#include "llama-expert-async-io.h"
#include "llama-expert-scheduler.h"
#include "llama-expert-storage.h"
#include "llama-expert-weight-provider.h"

#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml.h"

#include <array>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <unistd.h>
#endif

namespace {

struct provider_fixture {
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buffer;
    ggml_tensor * up = nullptr;
    ggml_tensor * gate = nullptr;
    ggml_tensor * down = nullptr;
    ggml_tensor * ids = nullptr;

    provider_fixture() {
        ggml_init_params params = { ggml_tensor_overhead()*8, nullptr, true };
        ctx.reset(ggml_init(params));
        up = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 2, 4, 4);
        gate = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 2, 4, 4);
        down = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 4, 2, 4);
        ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 2, 2);
        buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), ggml_backend_cpu_buffer_type()));
        GGML_ASSERT(buffer);
    }

    llm_expert_bundle_descriptor bundle() const {
        return { 0, 4,
            llm_expert_projection_descriptor::from(up, nullptr, nullptr),
            llm_expert_projection_descriptor::from(gate, nullptr, nullptr), {},
            llm_expert_projection_descriptor::from(down, nullptr, nullptr) };
    }
};

struct storage_fixture {
    std::string path;
    std::vector<uint8_t> bytes;
    std::unique_ptr<llama_file> file;
    std::unique_ptr<llm_expert_storage> storage;
    std::unique_ptr<llm_expert_async_transport> transport;

    explicit storage_fixture(const provider_fixture & provider) : bytes(4*3*32) {
        for (size_t index = 0; index < bytes.size(); ++index) {
            bytes[index] = uint8_t(0x20 + index);
        }
#if defined(_WIN32)
        char name[L_tmpnam];
        GGML_ASSERT(std::tmpnam(name) != nullptr);
        path = name;
        FILE * output = std::fopen(path.c_str(), "wb");
#else
        char name[] = "/tmp/llama-phase10r-demand-XXXXXX";
        const int fd = mkstemp(name);
        GGML_ASSERT(fd >= 0);
        path = name;
        FILE * output = fdopen(fd, "wb");
#endif
        GGML_ASSERT(output != nullptr);
        GGML_ASSERT(std::fwrite(bytes.data(), bytes.size(), 1, output) == 1);
        GGML_ASSERT(std::fclose(output) == 0);

        file = std::make_unique<llama_file>(path.c_str(), "rb");
        storage = std::make_unique<llm_expert_storage>(
            llm_expert_storage_config { 1, 4, 4, 1024 },
            std::vector<llm_expert_storage_source> { { 0, file.get(), 32 } });
        const uint64_t up_extent = provider.up->nb[2];
        const uint64_t gate_extent = provider.gate->nb[2];
        const uint64_t down_extent = provider.down->nb[2];
        GGML_ASSERT(up_extent == 32 && gate_extent == 32 && down_extent == 32);
        for (int32_t expert = 0; expert < 4; ++expert) {
            const uint64_t base = uint64_t(expert)*96;
            GGML_ASSERT(storage->add_bundle({ 0, expert }, {
                { 0, base, up_extent, llm_expert_storage_projection::up,
                    llm_expert_storage_sidecar::weight, 0, up_extent },
                { 0, base + up_extent, gate_extent, llm_expert_storage_projection::gate,
                    llm_expert_storage_sidecar::weight, up_extent, gate_extent },
                { 0, base + up_extent + gate_extent, down_extent,
                    llm_expert_storage_projection::down,
                    llm_expert_storage_sidecar::weight,
                    up_extent + gate_extent, down_extent },
            }).is_ready());
        }
        GGML_ASSERT(storage->seal().is_ready());

        llm_expert_async_config async_config;
        async_config.requested_queue_depth = 8;
        async_config.effective_hot_capacity = 4;
        async_config.request_capacity = 16;
        async_config.trace_capacity = 32;
        async_config.cold_cache_bytes = 1U << 20;
        async_config.maximum_aligned_read_bytes = 2U << 20;
        transport = std::make_unique<llm_expert_async_transport>(async_config);
    }

    ~storage_fixture() {
        transport.reset();
        storage.reset();
        file.reset();
        std::remove(path.c_str());
    }
};

void test_duplicate_occurrences_use_one_flight_per_key(bool serial_control) {
    ggml_backend_load_all();
    provider_fixture fixture;
    llm_expert_scheduler scheduler({ 1, 4, 8, 2, 4, 0 });
    llm_expert_phase8_test_control control;

    llm_hot_cache_config config;
    config.capacity = 4;
    config.n_expert_used = 2;
    config.routed_layer_count = 1;
    config.total_expert_keys = 4;
    config.target_buffer_type = ggml_backend_cpu_buffer_type();
    config.allow_non_cuda_target_for_testing = true;
    config.cold_mode = true;
    config.cold_cache_bytes = 1U << 20;
    config.transfer_ring_bytes = 1U << 20;
    config.target_device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    config.force_pageable_transfer_for_testing = true;
    config.scheduler = &scheduler;
    config.miss_policy = LLAMA_EXPERT_MISS_POLICY_CPU_FALLBACK;
    config.phase8_test_control = &control;
    config.serial_current_layer_demand_for_testing = serial_control;
    auto provider = llm_create_cold_cache_expert_weight_provider(config);

    ggml_init_params graph_params = { ggml_tensor_overhead()*8, nullptr, true };
    ggml_context_ptr graph_ctx(ggml_init(graph_params));
    const llm_expert_selection selection = { 0, 4, 2, 2, fixture.ids };
    llm_expert_graph_binding binding;
    GGML_ASSERT(provider->bind_graph(graph_ctx.get(), fixture.bundle(), selection, binding).is_ready());
    GGML_ASSERT(binding.bootstrap);
    GGML_ASSERT(provider->initialize_after_reserve().is_ready());
    binding = {};
    GGML_ASSERT(provider->bind_graph(graph_ctx.get(), fixture.bundle(), selection, binding).is_ready());
    GGML_ASSERT(binding.hybrid);
    ggml_backend_buffer_ptr graph_buffer(
        ggml_backend_alloc_ctx_tensors_from_buft(graph_ctx.get(), ggml_backend_cpu_buffer_type()));
    GGML_ASSERT(graph_buffer);
    ggml_backend_ptr backend(ggml_backend_dev_init(config.target_device, nullptr));
    GGML_ASSERT(backend);

    const std::array<int32_t, 4> logical = { 3, 1, 3, 1 };
    ggml_tensor * checkpoint_ids = binding.hybrid ? binding.checkpoint_ids : binding.execution_ids;
    ggml_backend_tensor_set(checkpoint_ids, logical.data(), 0, sizeof(logical));
    llm_expert_execution_plan plan;
    GGML_ASSERT(provider->prepare({ binding }, plan).is_ready());
    GGML_ASSERT(provider->remap_checkpoint_tensor(binding, backend.get()).is_ready());

    std::array<int32_t, 4> gpu = {};
    std::array<int32_t, 4> cpu = {};
    ggml_backend_tensor_get(binding.checkpoint_ids, gpu.data(), 0, sizeof(gpu));
    ggml_backend_tensor_get(binding.cpu_execution_ids, cpu.data(), 0, sizeof(cpu));
    const std::array<int32_t, 4> expected_gpu = { -1, -1, -1, -1 };
    GGML_ASSERT(gpu == expected_gpu);
    GGML_ASSERT(cpu[0] == cpu[2] && cpu[1] == cpu[3] && cpu[0] != cpu[1]);

    const auto diagnostics = scheduler.diagnostics();
    GGML_ASSERT(diagnostics.current_layer_batches == 1);
    GGML_ASSERT(diagnostics.current_layer_batch_entries == 2);
    GGML_ASSERT(diagnostics.flights_created == 2);
    GGML_ASSERT(diagnostics.active_requests == 0);
    GGML_ASSERT(diagnostics.queued_requests == 0);
    GGML_ASSERT(diagnostics.current_layer_batch_rejections == 0);
    const auto provider_diagnostics = provider->hot_cache_diagnostics();
    const auto & event = provider_diagnostics.last_demand_event;
    GGML_ASSERT(provider_diagnostics.demand_event_records == 1);
    GGML_ASSERT(event.version == 1);
    GGML_ASSERT(event.mode == (serial_control ? llm_expert_demand_mode::serial_control :
        llm_expert_demand_mode::issue_ahead));
    GGML_ASSERT(event.selected_logical_ids ==
        std::vector<int32_t>(logical.begin(), logical.end()));
    GGML_ASSERT(event.occurrence_to_unique == std::vector<int32_t>({ 0, 1, 0, 1 }));
    GGML_ASSERT(event.canonical_unique_keys.size() == 2);
    GGML_ASSERT(event.canonical_unique_keys[0].expert == 1);
    GGML_ASSERT(event.canonical_unique_keys[1].expert == 3);
    GGML_ASSERT(event.canonical_occurrence_counts == std::vector<uint64_t>({ 2, 2 }));
    GGML_ASSERT(event.enqueued_before_first_take == 2);
    GGML_ASSERT(event.first_take_ordinal > 0);
    GGML_ASSERT(event.first_terminal_release_ordinal > event.first_take_ordinal);
    GGML_ASSERT(event.full_set_enqueued_before_first_take);
    GGML_ASSERT(event.full_set_enqueued_before_first_terminal_release);
    GGML_ASSERT(event.full_set_enqueued_before_first_blocking_wait);
    GGML_ASSERT(event.exact_generations_ready_before_use);
    GGML_ASSERT(event.bounds_ok);

    plan.reset();
    binding = {};
    GGML_ASSERT(provider->surrender().is_ready());
}

llm_expert_demand_event run_storage_backed_mode(bool serial_control) {
    ggml_backend_load_all();
    provider_fixture fixture;
    storage_fixture storage(fixture);
    llm_expert_scheduler scheduler({ 1, 4, 16, 2, 4, 0 });

    llm_hot_cache_config config;
    config.capacity = 4;
    config.n_expert_used = 2;
    config.routed_layer_count = 1;
    config.total_expert_keys = 4;
    config.target_buffer_type = ggml_backend_cpu_buffer_type();
    config.allow_non_cuda_target_for_testing = true;
    config.cold_mode = true;
    config.cold_cache_bytes = 1U << 20;
    config.transfer_ring_bytes = 1U << 20;
    config.target_device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    config.force_pageable_transfer_for_testing = true;
    config.scheduler = &scheduler;
    config.storage = storage.storage.get();
    config.async_transport = storage.transport.get();
    config.miss_policy = LLAMA_EXPERT_MISS_POLICY_PROMOTE_AND_GPU;
    config.serial_current_layer_demand_for_testing = serial_control;
    auto provider = llm_create_cold_cache_expert_weight_provider(config);

    ggml_init_params graph_params = { ggml_tensor_overhead()*8, nullptr, true };
    ggml_context_ptr graph_ctx(ggml_init(graph_params));
    const llm_expert_selection selection = { 0, 4, 2, 2, fixture.ids };
    llm_expert_graph_binding binding;
    GGML_ASSERT(provider->bind_graph(
        graph_ctx.get(), fixture.bundle(), selection, binding).is_ready());
    GGML_ASSERT(provider->initialize_after_reserve().is_ready());
    binding = {};
    GGML_ASSERT(provider->bind_graph(
        graph_ctx.get(), fixture.bundle(), selection, binding).is_ready());
    ggml_backend_buffer_ptr graph_buffer(
        ggml_backend_alloc_ctx_tensors_from_buft(
            graph_ctx.get(), ggml_backend_cpu_buffer_type()));
    GGML_ASSERT(graph_buffer);
    ggml_backend_ptr backend(ggml_backend_dev_init(config.target_device, nullptr));
    GGML_ASSERT(backend);

    const std::array<int32_t, 4> logical = { 3, 1, 3, 1 };
    ggml_tensor * checkpoint_ids = binding.hybrid ? binding.checkpoint_ids : binding.execution_ids;
    ggml_backend_tensor_set(checkpoint_ids, logical.data(), 0, sizeof(logical));
    llm_expert_execution_plan plan;
    GGML_ASSERT(provider->prepare({ binding }, plan).is_ready());
    GGML_ASSERT(provider->remap_checkpoint_tensor(binding, backend.get()).is_ready());
    const auto diagnostics = provider->hot_cache_diagnostics();
    GGML_ASSERT(diagnostics.demand_event_records == 1);
    const auto event = diagnostics.last_demand_event;
    plan.reset();
    binding = {};
    GGML_ASSERT(provider->surrender().is_ready());
    return event;
}

void test_storage_issue_ahead_and_serial_control() {
    const auto issue_ahead = run_storage_backed_mode(false);
    const auto serial = run_storage_backed_mode(true);
    GGML_ASSERT(issue_ahead.mode == llm_expert_demand_mode::issue_ahead);
    GGML_ASSERT(serial.mode == llm_expert_demand_mode::serial_control);
    GGML_ASSERT(issue_ahead.selected_logical_ids == serial.selected_logical_ids);
    GGML_ASSERT(issue_ahead.occurrence_to_unique == serial.occurrence_to_unique);
    GGML_ASSERT(issue_ahead.canonical_unique_keys.size() == 2);
    GGML_ASSERT(issue_ahead.acquired_read_count == 2);
    GGML_ASSERT(issue_ahead.submitted_before_first_storage_wait == 2);
    GGML_ASSERT(issue_ahead.all_acquired_reads_submitted_before_first_storage_wait);
    GGML_ASSERT(serial.acquired_read_count == 2);
    GGML_ASSERT(serial.submitted_before_first_storage_wait == 1);
    GGML_ASSERT(!serial.all_acquired_reads_submitted_before_first_storage_wait);
    GGML_ASSERT(issue_ahead.first_wait_kind ==
        llm_expert_demand_wait_kind::storage_completion);
    GGML_ASSERT(serial.first_wait_kind ==
        llm_expert_demand_wait_kind::storage_completion);
    GGML_ASSERT(issue_ahead.full_set_enqueued_before_first_blocking_wait);
    GGML_ASSERT(serial.full_set_enqueued_before_first_blocking_wait);
}

void test_insufficient_scheduler_reserve_is_rejected() {
    provider_fixture fixture;
    llm_expert_scheduler scheduler({ 1, 4, 8, 2, 3, 0 });
    llm_hot_cache_config config;
    config.capacity = 4;
    config.n_expert_used = 2;
    config.routed_layer_count = 1;
    config.total_expert_keys = 4;
    config.target_buffer_type = ggml_backend_cpu_buffer_type();
    config.scheduler = &scheduler;
    bool rejected = false;
    try {
        (void) llm_create_cold_cache_expert_weight_provider(config);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    GGML_ASSERT(rejected);
}

void test_full_set_is_visible_before_take() {
    llm_expert_scheduler scheduler({ 1, 8, 4, 4, 3, 0 });
    const llm_expert_current_layer_schedule_entry entries[] = {
        { { 0, 1 }, llm_expert_readiness::host_ready },
        { { 0, 2 }, llm_expert_readiness::device_ready },
        { { 0, 5 }, llm_expert_readiness::device_ready },
    };
    llm_expert_current_layer_schedule_result results[3];
    llm_expert_request_handle revoked[3];
    size_t revoked_count = 0;
    GGML_ASSERT(scheduler.enqueue_current_layer_batch(
        entries, 3, results, 3, revoked, 3, revoked_count) ==
        llm_expert_schedule_disposition::admitted);
    GGML_ASSERT(revoked_count == 0);
    const auto before_take = scheduler.diagnostics();
    GGML_ASSERT(before_take.active_requests == 3 && before_take.queued_requests == 3);
    for (size_t index = 0; index < 3; ++index) {
        llm_expert_request_snapshot snapshot;
        GGML_ASSERT(scheduler.take(results[index].handle, snapshot).accepted());
        GGML_ASSERT(snapshot.key.layer == entries[index].key.layer);
        GGML_ASSERT(snapshot.key.expert == entries[index].key.expert);
        GGML_ASSERT(snapshot.demand_owned);
    }
}

} // namespace

int main() {
    test_duplicate_occurrences_use_one_flight_per_key(false);
    test_duplicate_occurrences_use_one_flight_per_key(true);
    test_storage_issue_ahead_and_serial_control();
    test_insufficient_scheduler_reserve_is_rejected();
    test_full_set_is_visible_before_take();
    return 0;
}
