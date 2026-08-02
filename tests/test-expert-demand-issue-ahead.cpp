#include "llama-expert-scheduler.h"
#include "llama-expert-weight-provider.h"

#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml.h"

#include <array>
#include <stdexcept>

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
    ggml_backend_tensor_set(binding.checkpoint_ids, logical.data(), 0, sizeof(logical));
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
    test_insufficient_scheduler_reserve_is_rejected();
    test_full_set_is_visible_before_take();
    return 0;
}
