#include "ggml-cpp.h"
#include "ggml-cuda.h"
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
    temporary_source source;
    llama_file file(source.path, "rb");
    llm_expert_storage storage({ 1, 2, 2, 1024 }, { { 0, &file, 512, source.path, false } });
    require(storage.add_bundle({ 0, 0 }, spans(0)).is_ready(), "expert 0 directory failed");
    require(storage.add_bundle({ 0, 1 }, spans(144)).is_ready(), "expert 1 directory failed");
    require(storage.seal().is_ready(), "storage seal failed");
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
    config.scheduler = &scheduler;
    config.is_uma_buffer_type = ggml_backend_buft_is_cuda_uma;
    config.prefetch = ggml_backend_cuda_uma_prefetch;
    config.checksum = ggml_backend_cuda_uma_checksum;
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
    }
    const auto provider_stats = provider->get_stats();
    const auto storage_stats = storage.diagnostics();
    const auto scheduler_stats = scheduler.diagnostics();
    require(provider_stats.tensor_copies == 0 && storage_stats.read_bytes == 288,
        "provider copied payload or storage byte count is wrong");
    require(scheduler_stats.terminal_complete == 2 && scheduler_stats.active_requests == 0,
        "readiness scheduler did not drain");
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
    temporary_source source;
    llama_file file(source.path, "rb");
    llm_expert_storage storage({ 1, 2, 2, 1024 }, { { 0, &file, 512, source.path, false } });
    require(storage.add_bundle({ 0, 0 }, spans(0)).is_ready(), "expert 0 directory failed");
    require(storage.add_bundle({ 0, 1 }, spans(144)).is_ready(), "expert 1 directory failed");
    require(storage.seal().is_ready(), "storage seal failed");
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
        config.scheduler = &scheduler;
        config.is_uma_buffer_type = ggml_backend_buft_is_cuda_uma;
        config.prefetch = controlled_prefetch;
        config.checksum = controlled_checksum;
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

} // namespace

int main() try {
    test_provider();
    test_readiness_selection_and_failed_generation_retry();
    printf("PHASE11_UMA_FAILURES\tauto_prefetch_probes=%d\tauto_touch_calls=%d"
        "\treadiness_retry_generation=%llu\tstale_rejected=%d\trestored_capacity=%d"
        "\tcancellation_cleanups=%llu\tcancellation_retry=%d\tscheduler_active=%llu\n",
        witness.auto_prefetch_probes, witness.auto_touch_calls,
        (unsigned long long) witness.readiness_retry_generation, witness.stale_rejected,
        witness.restored_capacity, (unsigned long long) witness.cancellation_cleanups,
        witness.cancellation_retry, (unsigned long long) witness.scheduler_active);
    return 0;
} catch (const std::exception & error) {
    fprintf(stderr, "test-expert-uma-provider: %s\n", error.what());
    return 1;
}
