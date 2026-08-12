#include "ggml-cpp.h"
#include "ggml-cuda.h"
#include "llama-expert-async-io.h"
#include "llama-expert-scheduler.h"
#include "llama-expert-storage.h"
#include "llama-expert-weight-provider.h"
#include "llama-mmap.h"

#include <array>
#include <atomic>
#include <cstdio>
#include <fcntl.h>
#include <stdexcept>
#include <unistd.h>

namespace {

std::atomic<int> prefetch_calls { 0 };
std::atomic<int> checksum_calls { 0 };
std::atomic<int> prefetch_failures { 0 };
std::atomic<int> checksum_fail_call { 0 };

struct failure_lifecycle_witness {
    int auto_prefetch_probes = 0;
    int auto_touch_calls = 0;
    uint64_t readiness_retry_generation = 0;
    bool stale_rejected = false;
    bool restored_capacity = false;
    uint64_t cancellation_cleanups = 0;
    bool cancellation_retry = false;
    uint64_t scheduler_active = 0;
} witness;

struct policy_pressure_witness {
    uint64_t safe_pool_bytes = 0;
    uint64_t effective_pool_bytes = 0;
    uint64_t pressure_samples = 0;
    uint64_t pressure_rejections = 0;
    bool autofit = false;
    bool explicit_policy = false;
    bool pressure_circuit = false;
    bool before_io = false;
    bool trim_zero_refs = false;
    bool surrender = false;
    bool fault_degraded = false;
    bool psi_circuit = false;
    bool compression_circuit = false;
} policy_witness;

llm_expert_uma_memory_sample controlled_memory;
bool memory_sample_failure = false;

void reset_memory_sample() {
    constexpr uint64_t GIB = UINT64_C(1024)*1024*1024;
    controlled_memory = {};
    controlled_memory.physical_ram_bytes = 128*GIB;
    controlled_memory.memory_available_bytes = 100*GIB;
    controlled_memory.cgroup_memory_max_bytes = 120*GIB;
    controlled_memory.cgroup_memory_current_bytes = 20*GIB;
    controlled_memory.process_rss_bytes = 2*GIB;
    controlled_memory.cgroup_v2 = true;
    controlled_memory.swap_counters_supported = true;
    controlled_memory.psi_full_supported = true;
    controlled_memory.zram_status_reason = "unsupported: no zram block device";
    controlled_memory.zswap_status_reason = "unsupported: zswap disabled";
    controlled_memory.nvidia_hmm_status_reason =
        "unsupported: no stable per-process NVIDIA HMM fault counter exposed";
    memory_sample_failure = false;
}

llm_expert_uma_result sample_controlled_memory(llm_expert_uma_memory_sample & output) {
    output = controlled_memory;
    return memory_sample_failure ?
        llm_expert_uma_result::failure(llm_expert_uma_error::unavailable_measurement) :
        llm_expert_uma_result::success();
}

int controlled_prefetch(ggml_backend_buffer_t buffer, size_t offset, size_t size) {
    prefetch_calls++;
    int remaining = prefetch_failures.load();
    if (remaining < 0 || (remaining > 0 && prefetch_failures.fetch_sub(1) > 0)) return 1;
    return ggml_backend_cuda_uma_prefetch(buffer, offset, size);
}

int controlled_checksum(ggml_backend_buffer_t buffer, size_t offset, size_t size, uint64_t * checksum) {
    const int call = ++checksum_calls;
    if (checksum_fail_call.load() == call) return 1;
    return ggml_backend_cuda_uma_checksum(buffer, offset, size, checksum);
}

bool abort_now(void *) { return true; }

void require(bool condition, const char * message) { if (!condition) throw std::runtime_error(message); }

struct temporary_source {
    char path[64] = "/tmp/phase11-uma-provider-XXXXXX";
    temporary_source() {
        const int fd = mkstemp(path);
        require(fd >= 0, "mkstemp failed");
        std::array<uint8_t, 288> bytes;
        for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = uint8_t(i*17U + 3U);
        require(write(fd, bytes.data(), bytes.size()) == ssize_t(bytes.size()), "source write failed");
        close(fd);
    }
    ~temporary_source() { unlink(path); }
};

struct async_fixture {
    llm_expert_async_transport transport;

    static llm_expert_async_config config() {
        llm_expert_async_config result;
        result.requested_queue_depth = 16;
        result.effective_hot_capacity = 2;
        result.request_capacity = 8;
        result.trace_capacity = 64;
        result.cold_cache_bytes = 8192;
        result.maximum_aligned_read_bytes = 4096;
        result.source_file_capacity = 1;
        result.force_positional_reads = true;
        return result;
    }

    explicit async_fixture(llm_expert_storage & storage) : transport(config()) {
        intptr_t handle = -1;
        size_t handle_count = 0;
        require(storage.copy_source_native_handles(&handle, 1, handle_count).is_ready() &&
            handle_count == 1, "source handle discovery failed");
        require(transport.register_files(&handle, 1) == llm_expert_async_result::ready,
            "source registration failed");
    }
};

llm_expert_bundle_descriptor make_bundle(ggml_context * ctx) {
    auto projection = [&](const char * name) {
        auto * weight = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, 3, 2);
        ggml_set_name(weight, name);
        return llm_expert_projection_descriptor::from(weight, nullptr, nullptr);
    };
    return { 0, 2, projection("up"), projection("gate"), {}, projection("down") };
}

std::vector<llm_expert_storage_span> spans(uint64_t base) {
    return {
        { 0, base + 0, 48, llm_expert_storage_projection::up, llm_expert_storage_sidecar::weight, 0, 48 },
        { 0, base + 48, 48, llm_expert_storage_projection::gate, llm_expert_storage_sidecar::weight, 48, 48 },
        { 0, base + 96, 48, llm_expert_storage_projection::down, llm_expert_storage_sidecar::weight, 96, 48 },
    };
}

void test_provider() {
    reset_memory_sample();
    temporary_source source;
    llama_file file(source.path, "rb");
    llm_expert_storage storage({ 1, 2, 2, 1024 }, { { 0, &file, 512, source.path, false } });
    require(storage.add_bundle({ 0, 0 }, spans(0)).is_ready(), "expert 0 directory failed");
    require(storage.add_bundle({ 0, 1 }, spans(144)).is_ready(), "expert 1 directory failed");
    require(storage.seal().is_ready(), "storage seal failed");
    async_fixture async(storage);
    llm_expert_scheduler scheduler({ 1, 2, 8, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1 });

    llm_uma_cache_config config;
    config.pool_bytes = 8192;
    config.hot_capacity = 1;
    config.n_expert_used = 1;
    config.routed_layer_count = 1;
    config.total_expert_keys = 2;
    config.buffer_type = ggml_backend_cuda_uma_buffer_type(0);
    config.target_device = ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), 0);
    config.storage = &storage;
    config.async_transport = &async.transport;
    config.scheduler = &scheduler;
    config.is_uma_buffer_type = ggml_backend_buft_is_cuda_uma;
    config.prefetch = ggml_backend_cuda_uma_prefetch;
    config.checksum = ggml_backend_cuda_uma_checksum;
    config.sample_memory = sample_controlled_memory;
    config.routed_layers = { 0 };
    auto provider = llm_create_uma_cache_expert_weight_provider(config);

    ggml_init_params params = { ggml_tensor_overhead()*32, nullptr, true };
    ggml_context_ptr ctx(ggml_init(params));
    require(bool(ctx), "context failed");
    auto bundle = make_bundle(ctx.get());
    auto * logical = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 1, 1);
    llm_expert_selection selection = { 0, 2, 1, 1, logical };
    bool owner = false;
    require(provider->begin_initialization(
        llm_expert_provider_initialization_stage::descriptors_before_scheduler_reserve, owner).is_ready() && owner,
        "initialization begin failed");
    llm_expert_graph_binding bootstrap;
    require(provider->bind_graph(ctx.get(), bundle, selection, bootstrap).is_ready() && bootstrap.bootstrap,
        "bootstrap bind failed");
    require(provider->bind_graph(ctx.get(), bundle, selection, bootstrap).is_ready(), "repeat bootstrap bind failed");
    require(provider->complete_descriptor_discovery(2, 2, 0, {}, {}).is_ready(), "descriptor discovery failed");
    require(provider->initialize_after_reserve().is_ready(), "UMA arena initialization failed");
    require(provider->finish_initialization(true).is_ready(), "initialization finish failed");

    llm_expert_graph_binding binding;
    require(provider->bind_graph(ctx.get(), bundle, selection, binding).is_ready() && !binding.bootstrap,
        "runtime bind failed");
    require(binding.up.weight->nb[2] == binding.gate.weight->nb[2] &&
        binding.up.weight->nb[2] == binding.down.weight->nb[2] && binding.up.weight->nb[2] % 4096 == 0,
        "bundle slots are not fixed interleaved page-aligned spans");
    for (int32_t expert = 0; expert < 2; ++expert) {
        llm_expert_execution_plan plan;
        require(provider->prepare({ binding }, plan).is_ready(), "prepare failed");
        int32_t execution = -1;
        require(provider->remap_checkpoint(binding, &expert, 1, &execution).is_ready(), "remap failed");
        require(execution >= 0 && execution < 2, "invalid execution slot");
        auto * data = static_cast<uint8_t *>(binding.up.weight->data) + size_t(execution)*binding.up.weight->nb[2];
        require(data[0] == uint8_t((size_t(expert)*144)*17U + 3U), "final slot has wrong source bytes");
        auto * base = static_cast<uint8_t *>(ggml_backend_buffer_get_base(binding.up.weight->buffer));
        uint64_t gpu_checksum = 0;
        require(ggml_backend_cuda_uma_checksum(binding.up.weight->buffer, size_t(data - base), 48,
            &gpu_checksum) == 0 && gpu_checksum != 0, "GPU did not consume final slot");
    }
    {
        const auto bytes_before = storage.diagnostics().read_bytes;
        llm_expert_execution_plan plan;
        require(provider->prepare({ binding }, plan).is_ready(), "cold-hit prepare failed");
        int32_t expert = 0, execution = -1;
        require(provider->remap_checkpoint(binding, &expert, 1, &execution).is_ready(), "cold-hit remap failed");
        require(storage.diagnostics().read_bytes == bytes_before && provider->hot_cache_diagnostics().cold_hits > 0,
            "hot miss did not reuse the retained cold slot");
        controlled_memory.major_faults++;
    }
    auto hit_diagnostics = provider->hot_cache_diagnostics();
    require(hit_diagnostics.uma_degraded_hits == 1 && hit_diagnostics.uma_unknown_residency_hits == 0,
        "major fault during a logical hit was not classified as degraded");
    policy_witness.fault_degraded = true;
    const auto provider_stats = provider->get_stats();
    const auto storage_stats = storage.diagnostics();
    const auto scheduler_stats = scheduler.diagnostics();
    const auto policy_stats = provider->hot_cache_diagnostics();
    require(provider_stats.tensor_copies == 0 && storage_stats.read_bytes == 288,
        "provider copied payload or storage byte count is wrong");
    require(scheduler_stats.terminal_complete == 2 && scheduler_stats.active_requests == 0,
        "readiness scheduler did not drain");
    require(policy_stats.policy.config.policy == LLAMA_EXPERT_CACHE_POLICY_LRU &&
        policy_stats.cold_policy.config.policy == LLAMA_EXPERT_CACHE_POLICY_LRU &&
        policy_stats.policy.config.admission == LLAMA_EXPERT_CACHE_ADMISSION_ALWAYS &&
        policy_stats.cold_policy.config.admission == LLAMA_EXPERT_CACHE_ADMISSION_ALWAYS,
        "null Phase 9 configuration did not preserve global LRU/ALWAYS defaults");
    const uint64_t bytes_before_trim = storage.diagnostics().read_bytes;
    require(provider->trim().is_ready(), "default-policy trim failed");
    {
        llm_expert_execution_plan plan;
        require(provider->prepare({ binding }, plan).is_ready(), "post-trim prepare failed");
        int32_t expert = 0, execution = -1;
        require(provider->remap_checkpoint(binding, &expert, 1, &execution).is_ready() &&
            storage.diagnostics().read_bytes == bytes_before_trim + 144,
            "post-trim demand did not reload the full bundle");
    }
}

void initialize_provider(
        llm_expert_weight_provider & provider,
        ggml_context * ctx,
        const llm_expert_bundle_descriptor & bundle,
        const llm_expert_selection & selection,
        llm_expert_graph_binding & binding) {
    bool owner = false;
    require(provider.begin_initialization(
        llm_expert_provider_initialization_stage::descriptors_before_scheduler_reserve, owner).is_ready() && owner,
        "initialization begin failed");
    llm_expert_graph_binding bootstrap;
    require(provider.bind_graph(ctx, bundle, selection, bootstrap).is_ready() && bootstrap.bootstrap,
        "bootstrap bind failed");
    require(provider.bind_graph(ctx, bundle, selection, bootstrap).is_ready(), "repeat bootstrap bind failed");
    require(provider.complete_descriptor_discovery(2, 2, 0, {}, {}).is_ready(), "descriptor discovery failed");
    require(provider.initialize_after_reserve().is_ready(), "UMA arena initialization failed");
    require(provider.finish_initialization(true).is_ready(), "initialization finish failed");
    require(provider.bind_graph(ctx, bundle, selection, binding).is_ready() && !binding.bootstrap,
        "runtime bind failed");
}

void test_readiness_selection_and_failed_generation_retry() {
    reset_memory_sample();
    temporary_source source;
    llama_file file(source.path, "rb");
    llm_expert_storage storage({ 1, 2, 2, 1024 }, { { 0, &file, 512, source.path, false } });
    require(storage.add_bundle({ 0, 0 }, spans(0)).is_ready(), "expert 0 directory failed");
    require(storage.add_bundle({ 0, 1 }, spans(144)).is_ready(), "expert 1 directory failed");
    require(storage.seal().is_ready(), "storage seal failed");
    async_fixture async(storage);
    ggml_init_params params = { ggml_tensor_overhead()*32, nullptr, true };
    ggml_context_ptr ctx(ggml_init(params));
    require(bool(ctx), "context failed");
    auto bundle = make_bundle(ctx.get());
    auto * logical = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 1, 1);
    llm_expert_selection selection = { 0, 2, 1, 1, logical };

    auto make_config = [&](llm_expert_scheduler & scheduler, llama_expert_uma_readiness readiness) {
        llm_uma_cache_config config;
        config.pool_bytes = 8192;
        config.hot_capacity = 1;
        config.n_expert_used = 1;
        config.routed_layer_count = 1;
        config.total_expert_keys = 2;
        config.buffer_type = ggml_backend_cuda_uma_buffer_type(0);
        config.target_device = ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), 0);
        config.storage = &storage;
        config.async_transport = &async.transport;
        config.scheduler = &scheduler;
        config.is_uma_buffer_type = ggml_backend_buft_is_cuda_uma;
        config.prefetch = controlled_prefetch;
        config.checksum = controlled_checksum;
        config.sample_memory = sample_controlled_memory;
        config.readiness = readiness;
        config.routed_layers = { 0 };
        return config;
    };

    {
        prefetch_calls = checksum_calls = 0;
        prefetch_failures = -1;
        checksum_fail_call = 0;
        llm_expert_scheduler scheduler({ 1, 2, 8, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1 });
        auto provider = llm_create_uma_cache_expert_weight_provider(
            make_config(scheduler, LLAMA_EXPERT_UMA_READINESS_CUDA_PREFETCH));
        bool owner = false;
        require(provider->begin_initialization(
            llm_expert_provider_initialization_stage::descriptors_before_scheduler_reserve, owner).is_ready() && owner,
            "explicit failure initialization begin failed");
        llm_expert_graph_binding bootstrap;
        require(provider->bind_graph(ctx.get(), bundle, selection, bootstrap).is_ready(),
            "explicit failure bootstrap bind failed");
        require(provider->bind_graph(ctx.get(), bundle, selection, bootstrap).is_ready(),
            "explicit failure repeat bind failed");
        require(provider->complete_descriptor_discovery(2, 2, 0, {}, {}).is_ready(),
            "explicit failure descriptor discovery failed");
        require(!provider->initialize_after_reserve().is_ready(),
            "unavailable explicit prefetch strategy was accepted");
        require(provider->finish_initialization(false).is_ready(),
            "failed explicit initialization did not roll back");
    }

    {
        prefetch_calls = checksum_calls = 0;
        prefetch_failures = -1;
        checksum_fail_call = 0;
        llm_expert_scheduler scheduler({ 1, 2, 8, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1 });
        auto provider = llm_create_uma_cache_expert_weight_provider(
            make_config(scheduler, LLAMA_EXPERT_UMA_READINESS_AUTO));
        llm_expert_graph_binding binding;
        initialize_provider(*provider, ctx.get(), bundle, selection, binding);
        llm_expert_execution_plan plan;
        require(provider->prepare({ binding }, plan).is_ready(), "AUTO prepare failed");
        int32_t logical_id = 0, execution = -1;
        require(provider->remap_checkpoint(binding, &logical_id, 1, &execution).is_ready(),
            "AUTO touch fallback failed");
        require(prefetch_calls == 1 && checksum_calls == 4,
            "AUTO readiness was not resolved exactly once to touch");
        witness.auto_prefetch_probes = prefetch_calls;
        witness.auto_touch_calls = checksum_calls;
    }

    {
        prefetch_calls = checksum_calls = 0;
        prefetch_failures = 0;
        checksum_fail_call = 2;
        llm_expert_scheduler scheduler({ 1, 2, 8, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1 });
        auto provider = llm_create_uma_cache_expert_weight_provider(
            make_config(scheduler, LLAMA_EXPERT_UMA_READINESS_CUDA_TOUCH));
        llm_expert_graph_binding binding;
        initialize_provider(*provider, ctx.get(), bundle, selection, binding);
        int32_t logical_id = 0, execution = -1;
        {
            llm_expert_execution_plan plan;
            require(provider->prepare({ binding }, plan).is_ready(), "failure prepare failed");
            require(!provider->remap_checkpoint(binding, &logical_id, 1, &execution).is_ready(),
                "injected readiness failure was accepted");
        }
        const auto failed = provider->hot_cache_diagnostics();
        require(failed.cold_failed_cleanups == 0 && failed.cold_slots[0].state ==
            llm_hot_cache_diagnostics::cold_slot::free, "readiness failure did not retire its generation");
        checksum_fail_call = 0;
        {
            llm_expert_execution_plan plan;
            require(provider->prepare({ binding }, plan).is_ready(), "retry prepare failed");
            require(provider->remap_checkpoint(binding, &logical_id, 1, &execution).is_ready(),
                "readiness retry failed");
        }
        const auto retried = provider->hot_cache_diagnostics();
        const uint64_t retry_generation = retried.cold_slots[execution].generation;
        const bool stale_rejected = !provider->validate_slot_generation(execution, retry_generation - 1).is_ready();
        require(retry_generation >= 2 && stale_rejected,
            "retry did not reject the stale generation");
        logical_id = 1;
        {
            llm_expert_execution_plan plan;
            require(provider->prepare({ binding }, plan).is_ready(), "restored-capacity prepare failed");
            require(provider->remap_checkpoint(binding, &logical_id, 1, &execution).is_ready(),
                "failed generation permanently reduced capacity");
        }
        require(scheduler.diagnostics().active_requests == 0, "readiness failure scheduler did not drain");
        witness.readiness_retry_generation = retry_generation;
        witness.stale_rejected = stale_rejected;
        witness.restored_capacity = true;
    }

    {
        prefetch_calls = checksum_calls = 0;
        prefetch_failures = 0;
        checksum_fail_call = 0;
        llm_expert_scheduler scheduler({ 1, 2, 8, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1 });
        auto provider = llm_create_uma_cache_expert_weight_provider(
            make_config(scheduler, LLAMA_EXPERT_UMA_READINESS_CUDA_PREFETCH));
        llm_expert_graph_binding binding;
        initialize_provider(*provider, ctx.get(), bundle, selection, binding);
        int32_t logical_id = 0, execution = -1;
        {
            llm_expert_execution_plan plan;
            require(provider->prepare({ binding }, plan).is_ready(), "cancel prepare failed");
            const auto cancelled = provider->remap_checkpoint(
                binding, &logical_id, 1, &execution, abort_now, nullptr);
            require(cancelled.error == llm_expert_provider_error::cancelled, "storage cancellation was not reported");
        }
        const auto cancelled = provider->hot_cache_diagnostics();
        require(cancelled.cold_failed_cleanups == 1 && cancelled.cold_slots[0].state ==
            llm_hot_cache_diagnostics::cold_slot::free, "cancelled generation was not cleaned");
        {
            llm_expert_execution_plan plan;
            require(provider->prepare({ binding }, plan).is_ready(), "cancel retry prepare failed");
            require(provider->remap_checkpoint(binding, &logical_id, 1, &execution).is_ready(),
                "cancelled generation was not reusable");
        }
        require(scheduler.diagnostics().active_requests == 0, "cancelled scheduler request did not drain");
        witness.cancellation_cleanups = cancelled.cold_failed_cleanups;
        witness.cancellation_retry = true;
        witness.scheduler_active += scheduler.diagnostics().active_requests;
    }
}

llm_expert_cache_policy_config_internal explicit_lfu(llm_expert_cache_policy_tier tier) {
    const llama_expert_cache_policy_config source = {
        LLAMA_EXPERT_CACHE_POLICY_VERSION_1,
        sizeof(llama_expert_cache_policy_config),
        LLAMA_EXPERT_CACHE_POLICY_LFU_AGING,
        LLAMA_EXPERT_CACHE_POLICY_SCOPE_GLOBAL,
        0,
        LLAMA_EXPERT_CACHE_ADMISSION_ALWAYS,
        0,
        LLAMA_EXPERT_CACHE_POLICY_LFU_AGING_DEFAULT_EVENTS,
        {},
    };
    llm_expert_cache_policy_config_internal result;
    require(llm_expert_cache_policy_copy_config(&source, tier, result).is_ready(), "explicit LFU copy failed");
    return result;
}

void test_autofit_policy_pressure_trim_and_surrender() {
    reset_memory_sample();
    temporary_source source;
    llama_file file(source.path, "rb");
    llm_expert_storage storage({ 1, 2, 2, 1024 }, { { 0, &file, 512, source.path, false } });
    require(storage.add_bundle({ 0, 0 }, spans(0)).is_ready(), "pressure expert 0 directory failed");
    require(storage.add_bundle({ 0, 1 }, spans(144)).is_ready(), "pressure expert 1 directory failed");
    require(storage.seal().is_ready(), "pressure storage seal failed");
    async_fixture async(storage);
    llm_expert_scheduler scheduler({ 1, 2, 8, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1 });
    llm_uma_cache_config config;
    config.pool_bytes = 0;
    config.hot_capacity = 1;
    config.n_expert_used = 1;
    config.routed_layer_count = 1;
    config.total_expert_keys = 2;
    config.buffer_type = ggml_backend_cuda_uma_buffer_type(0);
    config.target_device = ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), 0);
    config.storage = &storage;
    config.async_transport = &async.transport;
    config.scheduler = &scheduler;
    config.is_uma_buffer_type = ggml_backend_buft_is_cuda_uma;
    config.prefetch = ggml_backend_cuda_uma_prefetch;
    config.checksum = ggml_backend_cuda_uma_checksum;
    config.sample_memory = sample_controlled_memory;
    config.hot_cache_policy_config = explicit_lfu(llm_expert_cache_policy_tier::hot);
    config.cold_cache_policy_config = explicit_lfu(llm_expert_cache_policy_tier::cold);
    config.routed_layers = { 0 };
    auto provider = llm_create_uma_cache_expert_weight_provider(config);
    ggml_init_params params = { ggml_tensor_overhead()*32, nullptr, true };
    ggml_context_ptr ctx(ggml_init(params));
    require(bool(ctx), "pressure context failed");
    auto bundle = make_bundle(ctx.get());
    auto * logical = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 1, 1);
    llm_expert_selection selection = { 0, 2, 1, 1, logical };
    auto reject_initialization = [&](llm_uma_cache_config rejected_config,
                                     llm_expert_provider_status expected_status) {
        auto rejected_provider = llm_create_uma_cache_expert_weight_provider(std::move(rejected_config));
        bool owner = false;
        require(rejected_provider->begin_initialization(
            llm_expert_provider_initialization_stage::descriptors_before_scheduler_reserve, owner).is_ready() && owner,
            "negative initialization begin failed");
        llm_expert_graph_binding bootstrap;
        require(rejected_provider->bind_graph(ctx.get(), bundle, selection, bootstrap).is_ready() &&
            rejected_provider->bind_graph(ctx.get(), bundle, selection, bootstrap).is_ready(),
            "negative initialization bootstrap failed");
        require(rejected_provider->complete_descriptor_discovery(2, 2, 0, {}, {}).is_ready(),
            "negative descriptor discovery failed");
        const auto rejected = rejected_provider->initialize_after_reserve();
        require(rejected.status == expected_status && rejected_provider->finish_initialization(false).is_ready(),
            "unsafe or unmeasurable initialization did not fail transactionally");
    };
    memory_sample_failure = true;
    reject_initialization(config, llm_expert_provider_status::failed);
    memory_sample_failure = false;
    controlled_memory.psi_full_supported = false;
    reject_initialization(config, llm_expert_provider_status::failed);
    controlled_memory.psi_full_supported = true;
    auto unsafe_config = config;
    unsafe_config.pool_bytes = UINT64_C(80)*1024*1024*1024;
    reject_initialization(unsafe_config, llm_expert_provider_status::allocation_failed);
    llm_expert_graph_binding binding;
    initialize_provider(*provider, ctx.get(), bundle, selection, binding);
    auto diagnostics = provider->hot_cache_diagnostics();
    require(diagnostics.uma_autofit && diagnostics.uma_safe_pool_bytes >= diagnostics.uma_effective_pool_bytes &&
        diagnostics.uma_effective_pool_bytes == diagnostics.cold_actual_bytes,
        "autofit did not select the safe topology-bounded whole-slot arena");
    require(diagnostics.uma_model_capacity_bytes == diagnostics.cold_slot_footprint*2 &&
        diagnostics.uma_effective_slot_count == diagnostics.cold_effective_slots &&
        diagnostics.uma_model_cap_unused_safe_bytes ==
            diagnostics.uma_safe_pool_bytes - diagnostics.uma_effective_pool_bytes &&
        diagnostics.uma_alignment_remainder_bytes == diagnostics.uma_headroom_remainder_bytes,
        "autofit model-cap telemetry is incomplete");
    require(diagnostics.policy.config.policy == LLAMA_EXPERT_CACHE_POLICY_LFU_AGING &&
        diagnostics.cold_policy.config.policy == LLAMA_EXPERT_CACHE_POLICY_LFU_AGING &&
        diagnostics.policy.config.admission == LLAMA_EXPERT_CACHE_ADMISSION_ALWAYS &&
        diagnostics.cold_policy.config.admission == LLAMA_EXPERT_CACHE_ADMISSION_ALWAYS,
        "explicit Phase 9 policy was not preserved");

    controlled_memory.memory_available_bytes = UINT64_C(1024)*1024*1024;
    int32_t logical_id = 0, execution = -1;
    {
        llm_expert_execution_plan plan;
        const auto rejected = provider->prepare({ binding }, plan);
        require(rejected.status == llm_expert_provider_status::allocation_failed && storage.diagnostics().read_bytes == 0,
            "headroom pressure was not rejected before storage I/O");
    }
    controlled_memory.memory_available_bytes = UINT64_C(100)*1024*1024*1024;
    {
        llm_expert_execution_plan plan;
        require(provider->prepare({ binding }, plan).is_ready(), "pressure retry prepare failed");
        require(provider->remap_checkpoint(binding, &logical_id, 1, &execution).is_ready(),
            "recoverable headroom rejection poisoned retry");
    }
    const uint64_t bytes_before_swap = storage.diagnostics().read_bytes;
    controlled_memory.psi_full_total_usec = 1;
    controlled_memory.zram_present = true;
    controlled_memory.zram_counters_supported = true;
    controlled_memory.zram_write_bytes = 4096;
    controlled_memory.zswap_enabled = true;
    controlled_memory.zswap_counters_supported = true;
    controlled_memory.zswap_write_pages = 1;
    logical_id = 1;
    {
        llm_expert_execution_plan plan;
        const auto rejected = provider->prepare({ binding }, plan);
        require(rejected.status == llm_expert_provider_status::allocation_failed,
            "post-activation PSI/compression activity did not open the pressure circuit");
    }
    policy_witness.psi_circuit = true;
    policy_witness.compression_circuit = true;
    controlled_memory.psi_full_total_usec = 0;
    controlled_memory.zram_present = false;
    controlled_memory.zram_counters_supported = false;
    controlled_memory.zram_write_bytes = 0;
    controlled_memory.zswap_enabled = false;
    controlled_memory.zswap_counters_supported = false;
    controlled_memory.zswap_write_pages = 0;
    {
        llm_expert_execution_plan plan;
        require(provider->prepare({ binding }, plan).status ==
            llm_expert_provider_status::allocation_failed, "pressure circuit did not remain open");
    }
    diagnostics = provider->hot_cache_diagnostics();
    require(diagnostics.uma_pressure_circuit_open && diagnostics.uma_pressure_rejections == 3 &&
        diagnostics.uma_pressure_samples == 4 && storage.diagnostics().read_bytes == bytes_before_swap,
        "pressure telemetry or before-I/O circuit behavior is wrong");
    policy_witness.safe_pool_bytes = diagnostics.uma_safe_pool_bytes;
    policy_witness.effective_pool_bytes = diagnostics.uma_effective_pool_bytes;
    policy_witness.pressure_samples = diagnostics.uma_pressure_samples;
    policy_witness.pressure_rejections = diagnostics.uma_pressure_rejections;
    policy_witness.autofit = diagnostics.uma_autofit;
    policy_witness.explicit_policy = diagnostics.policy.config.policy == LLAMA_EXPERT_CACHE_POLICY_LFU_AGING &&
        diagnostics.cold_policy.config.policy == LLAMA_EXPERT_CACHE_POLICY_LFU_AGING;
    policy_witness.pressure_circuit = diagnostics.uma_pressure_circuit_open;
    policy_witness.before_io = storage.diagnostics().read_bytes == bytes_before_swap;
    require(provider->trim().is_ready(), "UMA trim failed");
    diagnostics = provider->hot_cache_diagnostics();
    require(diagnostics.cold_current_hot_refs == 0 && diagnostics.cold_current_request_refs == 0 &&
        diagnostics.cold_reclaimed_bytes > 0 && diagnostics.cold_reclaim_failures == 0,
        "UMA trim retained live references");
    policy_witness.trim_zero_refs = true;
    require(provider->surrender().error == llm_expert_provider_error::busy,
        "UMA surrender ignored a live graph allocation lease");
    binding = {};
    require(provider->surrender().is_ready() && provider->get_stats().pool_generations == 0,
        "UMA surrender did not release the arena");
    policy_witness.surrender = true;
}

} // namespace

int main() try {
    test_provider();
    test_readiness_selection_and_failed_generation_retry();
    test_autofit_policy_pressure_trim_and_surrender();
    printf("PHASE11_UMA_FAILURES\tauto_prefetch_probes=%d\tauto_touch_calls=%d"
        "\treadiness_retry_generation=%llu\tstale_rejected=%d\trestored_capacity=%d"
        "\tcancellation_cleanups=%llu\tcancellation_retry=%d\tscheduler_active=%llu\n",
        witness.auto_prefetch_probes, witness.auto_touch_calls,
        (unsigned long long) witness.readiness_retry_generation, witness.stale_rejected,
        witness.restored_capacity, (unsigned long long) witness.cancellation_cleanups,
        witness.cancellation_retry, (unsigned long long) witness.scheduler_active);
    printf("PHASE11_UMA_LIFECYCLE\tsafe_pool_bytes=%llu\teffective_pool_bytes=%llu"
        "\tautofit=%d\texplicit_policy=%d\tpressure_samples=%llu\tpressure_rejections=%llu"
        "\tpressure_circuit=%d\tbefore_io=%d\ttrim_zero_refs=%d\tsurrender=%d"
        "\tfault_degraded=%d\tpsi_circuit=%d\tcompression_circuit=%d\n",
        (unsigned long long) policy_witness.safe_pool_bytes,
        (unsigned long long) policy_witness.effective_pool_bytes, policy_witness.autofit,
        policy_witness.explicit_policy, (unsigned long long) policy_witness.pressure_samples,
        (unsigned long long) policy_witness.pressure_rejections, policy_witness.pressure_circuit,
        policy_witness.before_io, policy_witness.trim_zero_refs, policy_witness.surrender,
        policy_witness.fault_degraded, policy_witness.psi_circuit, policy_witness.compression_circuit);
    return 0;
} catch (const std::exception & error) {
    fprintf(stderr, "test-expert-uma-provider: %s\n", error.what());
    return 1;
}
