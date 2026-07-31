#include "llama-expert-transfer-ring.h"
#include "llama-expert-scheduler.h"

#include "ggml-cpp.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

struct fixture {
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buffer;
    ggml_tensor * up = nullptr;
    ggml_tensor * gate = nullptr;
    ggml_tensor * down = nullptr;
    int32_t n_expert;

    explicit fixture(int32_t n_expert, bool fill = true,
            ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type()) : n_expert(n_expert) {
        ggml_init_params params = { ggml_tensor_overhead()*8, nullptr, true };
        ctx.reset(ggml_init(params));
        GGML_ASSERT(ctx);
        up = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 8, 16, n_expert);
        gate = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 8, 16, n_expert);
        down = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 16, 8, n_expert);
        buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft));
        GGML_ASSERT(buffer);
        if (fill) {
            uint8_t pattern = 0x20;
            for (auto * tensor : { up, gate, down }) {
                for (int32_t expert = 0; expert < n_expert; ++expert) {
                    std::vector<uint8_t> bytes(tensor->nb[2], pattern + expert);
                    ggml_backend_tensor_set(tensor, bytes.data(), size_t(expert)*tensor->nb[2], bytes.size());
                }
                pattern += 0x20;
            }
        }
    }

    llm_expert_bundle_descriptor bundle(int32_t layer = 0) const {
        return {
            layer,
            n_expert,
            llm_expert_projection_descriptor::from(up, nullptr, nullptr),
            llm_expert_projection_descriptor::from(gate, nullptr, nullptr),
            {},
            llm_expert_projection_descriptor::from(down, nullptr, nullptr),
        };
    }
};

ggml_backend_dev_t cpu_device() {
    static const bool loaded = [] { ggml_backend_load_all(); return true; }();
    (void) loaded;
    auto * result = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    GGML_ASSERT(result);
    return result;
}

llm_cold_expert_cache make_cold(const fixture & source) {
    llm_cold_expert_cache cache({ 1U << 20, 2, 1, uint32_t(source.n_expert), 0 });
    GGML_ASSERT(cache.initialize(source.bundle()).is_ready());
    return cache;
}

llm_transfer_ring_config ring_config(uint64_t bytes, uint32_t minimum = 2, uint64_t generation = 0) {
    return { bytes, minimum, cpu_device(), true, true, false, generation };
}

void assert_slot_matches(const fixture & source, const fixture & hot, int32_t expert, uint32_t slot) {
    const auto source_bundle = source.bundle();
    const auto hot_bundle = hot.bundle();
    for (const auto & pair : {
            std::pair<const ggml_tensor *, const ggml_tensor *>(source_bundle.up.weight, hot_bundle.up.weight),
            std::pair<const ggml_tensor *, const ggml_tensor *>(source_bundle.gate.weight, hot_bundle.gate.weight),
            std::pair<const ggml_tensor *, const ggml_tensor *>(source_bundle.down.weight, hot_bundle.down.weight) }) {
        const size_t span = pair.first->nb[2];
        std::vector<uint8_t> expected(span), actual(span);
        ggml_backend_tensor_get(pair.first, expected.data(), size_t(expert)*span, span);
        ggml_backend_tensor_get(pair.second, actual.data(), size_t(slot)*span, span);
        GGML_ASSERT(expected == actual);
    }
}

void test_budget_fallback_and_wave() {
    fixture source(4);
    fixture hot(2, false);
    auto cold = make_cold(source);
    llm_cold_reference cold_zero, cold_one;
    GGML_ASSERT(cold.find_or_admit({ 0, 0 }, source.bundle(), cold_zero).is_ready());
    GGML_ASSERT(cold.find_or_admit({ 0, 1 }, source.bundle(), cold_one).is_ready());

    llm_expert_transfer_ring discovery(ring_config(1U << 20));
    GGML_ASSERT(discovery.initialize(source.bundle()).is_ready());
    const uint64_t two_lane_budget = discovery.diagnostics().lane_footprint*2;
    llm_expert_transfer_ring ring(ring_config(two_lane_budget));
    GGML_ASSERT(ring.initialize(source.bundle()).is_ready());
    auto diagnostics = ring.diagnostics();
    GGML_ASSERT(diagnostics.effective_lanes == 2);
    GGML_ASSERT(diagnostics.actual_bytes <= diagnostics.requested_bytes);
    GGML_ASSERT(diagnostics.pageable_fallback);
    GGML_ASSERT(diagnostics.pinned_or_registered_bytes == 0);
    GGML_ASSERT(diagnostics.acquisition_method == "pageable-cpu");

    llm_transfer_lane_reference lane_zero, lane_one;
    GGML_ASSERT(ring.reserve(cold, cold_zero, 0, 1, lane_zero).is_ready());
    GGML_ASSERT(ring.reserve(cold, cold_one, 1, 1, lane_one).is_ready());
    GGML_ASSERT(ring.stage(lane_zero, cold.bundle()).is_ready());
    GGML_ASSERT(ring.stage(lane_one, cold.bundle()).is_ready());
    GGML_ASSERT(ring.transfer_wave(nullptr, {
        { lane_zero, hot.bundle(), 0 }, { lane_one, hot.bundle(), 1 },
    }).is_ready());
    assert_slot_matches(source, hot, 0, 0);
    assert_slot_matches(source, hot, 1, 1);
    diagnostics = ring.diagnostics();
    GGML_ASSERT(diagnostics.waves == 1);
    GGML_ASSERT(diagnostics.wave_synchronizations == 0);
    GGML_ASSERT(diagnostics.async_enqueues == 0);
    GGML_ASSERT(diagnostics.synchronous_copies == 6);
    GGML_ASSERT(diagnostics.h2d_bytes == diagnostics.lane_payload_bytes*2);
    GGML_ASSERT(cold.diagnostics().current_transfer_refs == 0);
    GGML_ASSERT(ring.validate_invariants().is_ready());
}

void test_failures_cleanup_and_busy_surrender() {
    fixture source(4);
    fixture hot(2, false);
    auto cold = make_cold(source);
    llm_cold_reference cold_zero;
    GGML_ASSERT(cold.find_or_admit({ 0, 0 }, source.bundle(), cold_zero).is_ready());

    llm_expert_transfer_ring stage_failure(ring_config(1U << 20), { false, 1, false, false });
    GGML_ASSERT(stage_failure.initialize(source.bundle()).is_ready());
    llm_transfer_lane_reference lane;
    GGML_ASSERT(stage_failure.reserve(cold, cold_zero, 0, 1, lane).is_ready());
    GGML_ASSERT(stage_failure.surrender().error == llm_expert_provider_error::busy);
    GGML_ASSERT(stage_failure.stage(lane, cold.bundle()).error == llm_expert_provider_error::copy_failed);
    GGML_ASSERT(stage_failure.cleanup_failed_lanes().is_ready());
    GGML_ASSERT(cold.diagnostics().current_transfer_refs == 0);
    GGML_ASSERT(stage_failure.surrender().is_ready());

    llm_expert_transfer_ring enqueue_failure(ring_config(1U << 20), { false, SIZE_MAX, true, false });
    GGML_ASSERT(enqueue_failure.initialize(source.bundle()).is_ready());
    GGML_ASSERT(enqueue_failure.reserve(cold, cold_zero, 0, 2, lane).is_ready());
    GGML_ASSERT(enqueue_failure.stage(lane, cold.bundle()).is_ready());
    GGML_ASSERT(enqueue_failure.transfer_wave(nullptr, { { lane, hot.bundle(), 0 } }).error ==
        llm_expert_provider_error::copy_failed);
    GGML_ASSERT(enqueue_failure.cleanup_failed_lanes().is_ready());
    GGML_ASSERT(cold.diagnostics().current_transfer_refs == 0);

    {
        llm_expert_transfer_ring cleanup_failure(ring_config(1U << 20), { false, 0, false, true });
        GGML_ASSERT(cleanup_failure.initialize(source.bundle()).is_ready());
        GGML_ASSERT(cleanup_failure.reserve(cold, cold_zero, 0, 3, lane).is_ready());
        GGML_ASSERT(cleanup_failure.stage(lane, cold.bundle()).error == llm_expert_provider_error::copy_failed);
        GGML_ASSERT(cleanup_failure.cleanup_failed_lanes().error == llm_expert_provider_error::copy_failed);
        GGML_ASSERT(cold.diagnostics().current_transfer_refs == 1);
    }
    GGML_ASSERT(cold.diagnostics().current_transfer_refs == 0);

    llm_expert_transfer_ring allocation_failure(ring_config(1U << 20), { true });
    GGML_ASSERT(allocation_failure.initialize(source.bundle()).error ==
        llm_expert_provider_error::allocation_failed);
}

void test_budget_and_generation_rejection() {
    fixture source(4);
    llm_expert_transfer_ring discovery(ring_config(1U << 20));
    GGML_ASSERT(discovery.initialize(source.bundle()).is_ready());
    const uint64_t footprint = discovery.diagnostics().lane_footprint;
    llm_expert_transfer_ring insufficient(ring_config(footprint*2 - 1));
    GGML_ASSERT(insufficient.initialize(source.bundle()).error ==
        llm_expert_provider_error::unsupported_configuration);

    const uint64_t overflow_lanes = uint64_t(UINT32_MAX)/2 + 1;
    GGML_ASSERT(footprint <= UINT64_MAX/overflow_lanes);
    llm_expert_transfer_ring event_capacity_overflow(ring_config(footprint*overflow_lanes));
    GGML_ASSERT(event_capacity_overflow.initialize(source.bundle()).error ==
        llm_expert_provider_error::unsupported_configuration);
    const auto overflow = event_capacity_overflow.diagnostics();
    GGML_ASSERT(overflow.actual_bytes == 0 && overflow.effective_lanes == 0 &&
        overflow.pinned_or_registered_bytes == 0 && !overflow.dedicated_transfer_backend &&
        !overflow.event_capable && overflow.h2d_event_allocations == 0 &&
        overflow.compute_event_allocations == 0 && overflow.event_capacity == 0 &&
        overflow.live_events == 0);

    auto cold = make_cold(source);
    llm_cold_reference cold_zero;
    GGML_ASSERT(cold.find_or_admit({ 0, 0 }, source.bundle(), cold_zero).is_ready());
    llm_expert_transfer_ring exhausted(ring_config(footprint*2, 2, std::numeric_limits<uint64_t>::max()));
    GGML_ASSERT(exhausted.initialize(source.bundle()).is_ready());
    llm_transfer_lane_reference lane;
    GGML_ASSERT(exhausted.reserve(cold, cold_zero, 0, 1, lane).error ==
        llm_expert_provider_error::generation_exhausted);
}

void test_native_event_ordering_reuse_and_unload() {
    ggml_backend_load_all();
    auto * device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (device == nullptr) return;
    ggml_backend_ptr backend(ggml_backend_dev_init(device, nullptr));
    GGML_ASSERT(backend);
    fixture source(4);
    fixture hot(2, false, ggml_backend_dev_buffer_type(device));
    auto cold = make_cold(source);
    llm_cold_reference cold_zero, cold_one;
    GGML_ASSERT(cold.find_or_admit({ 0, 0 }, source.bundle(), cold_zero).is_ready());
    GGML_ASSERT(cold.find_or_admit({ 0, 1 }, source.bundle(), cold_one).is_ready());

    llm_expert_transfer_ring ring({ 1U << 20, 2, device, false, false, false, 0, 32 });
    GGML_ASSERT(ring.initialize(source.bundle()).is_ready());
    llm_transfer_lane_reference lane_zero, lane_one;
    GGML_ASSERT(ring.reserve(cold, cold_zero, 0, 1, lane_zero).is_ready());
    GGML_ASSERT(ring.reserve(cold, cold_one, 1, 1, lane_one).is_ready());
    GGML_ASSERT(ring.stage(lane_zero, cold.bundle()).is_ready());
    GGML_ASSERT(ring.stage(lane_one, cold.bundle()).is_ready());
    GGML_ASSERT(ring.transfer_wave(backend.get(), {
        { lane_zero, hot.bundle(), 0 }, { lane_one, hot.bundle(), 1 },
    }).is_ready());
    GGML_ASSERT(ring.wait_for_hot(backend.get(), 0, 2).error ==
        llm_expert_provider_error::stale_generation);
    GGML_ASSERT(ring.wait_for_hot(backend.get(), 0, 1).is_ready());
    GGML_ASSERT(ring.wait_for_hot(backend.get(), 1, 1).is_ready());
    GGML_ASSERT(ring.retire_hot(0, 1).is_ready());
    GGML_ASSERT(ring.retire_hot(1, 1).is_ready());
    assert_slot_matches(source, hot, 0, 0);
    assert_slot_matches(source, hot, 1, 1);
    const auto diagnostics = ring.diagnostics();
    GGML_ASSERT(diagnostics.event_capable && diagnostics.event_capacity == diagnostics.effective_lanes*2);
    GGML_ASSERT(diagnostics.h2d_event_capacity == diagnostics.effective_lanes &&
        diagnostics.compute_event_capacity == diagnostics.effective_lanes &&
        diagnostics.h2d_event_allocations == diagnostics.h2d_event_capacity &&
        diagnostics.compute_event_allocations == diagnostics.compute_event_capacity);
    GGML_ASSERT(diagnostics.event_records == 2 && diagnostics.compute_waits == 2);
    GGML_ASSERT(diagnostics.event_synchronizations == 2 && diagnostics.live_h2d_events == 0 &&
        diagnostics.live_compute_events == 0 && diagnostics.live_events == 0);
    const auto intervals = ring.completed_intervals();
    GGML_ASSERT(intervals.size() == 2 && intervals[0].h2d_enqueue_us != 0 &&
        intervals[0].h2d_complete_us >= intervals[0].h2d_enqueue_us &&
        intervals[0].bytes == diagnostics.lane_payload_bytes);
    GGML_ASSERT(cold.diagnostics().current_transfer_refs == 0);

    llm_transfer_ring_config background_config = {
        diagnostics.lane_footprint, 1, device, false, false, false, 0, 32,
    };
    background_config.delay_background_stage_ms_for_testing = 250;
    llm_expert_transfer_ring background(background_config);
    GGML_ASSERT(background.initialize(source.bundle()).is_ready());
    llm_transfer_lane_reference background_lane;
    GGML_ASSERT(background.try_queue_background_transfer(
        cold, cold_zero, 0, 3, cold.bundle(), hot.bundle(), background_lane).is_ready());
    bool background_complete = false;
    uint64_t remaining_submitted = UINT64_MAX;
    GGML_ASSERT(background.poll_h2d(
        background_lane, background_complete, &remaining_submitted).is_ready());
    GGML_ASSERT(!background_complete && remaining_submitted == 0);
    GGML_ASSERT(background.wait_for_hot(backend.get(), 0, 3).is_ready());
    GGML_ASSERT(background.retire_hot(0, 3).is_ready());
    GGML_ASSERT(background.surrender().is_ready());
    GGML_ASSERT(cold.diagnostics().current_transfer_refs == 0);

    llm_expert_transfer_ring failed_background(background_config, { false, 1, false, false });
    GGML_ASSERT(failed_background.initialize(source.bundle()).is_ready());
    llm_transfer_lane_reference failed_lane;
    GGML_ASSERT(failed_background.try_queue_background_transfer(
        cold, cold_zero, 0, 4, cold.bundle(), hot.bundle(), failed_lane).is_ready());
    llm_expert_provider_result failed_poll;
    const auto failure_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    do {
        failed_poll = failed_background.poll_h2d(failed_lane, background_complete);
        if (!failed_poll.is_ready()) break;
        std::this_thread::yield();
    } while (std::chrono::steady_clock::now() < failure_deadline);
    GGML_ASSERT(failed_poll.error == llm_expert_provider_error::copy_failed);
    GGML_ASSERT(failed_background.release_failed_background(failed_lane).is_ready());
    GGML_ASSERT(failed_background.surrender().is_ready());
    GGML_ASSERT(cold.diagnostics().current_transfer_refs == 0);

    {
        ggml_backend_ptr gate_backend(ggml_backend_dev_init(device, nullptr));
        GGML_ASSERT(gate_backend);
        ggml_init_params compute_params = { ggml_tensor_overhead()*4 + ggml_graph_overhead(), nullptr, true };
        ggml_context_ptr compute_ctx(ggml_init(compute_params));
        GGML_ASSERT(compute_ctx);
        ggml_tensor * a = ggml_new_tensor_2d(compute_ctx.get(), GGML_TYPE_F32, 1536, 1536);
        ggml_tensor * b = ggml_new_tensor_2d(compute_ctx.get(), GGML_TYPE_F32, 1536, 1536);
        ggml_tensor * c = ggml_mul_mat(compute_ctx.get(), a, b);
        ggml_backend_buffer_ptr compute_buffer(
            ggml_backend_alloc_ctx_tensors_from_buft(compute_ctx.get(), ggml_backend_dev_buffer_type(device)));
        GGML_ASSERT(compute_buffer);
        std::vector<float> zeros(size_t(1536)*1536, 0.0f);
        ggml_backend_tensor_set(a, zeros.data(), 0, zeros.size()*sizeof(float));
        ggml_backend_tensor_set(b, zeros.data(), 0, zeros.size()*sizeof(float));
        ggml_cgraph * graph = ggml_new_graph_custom(compute_ctx.get(), 8, false);
        ggml_build_forward_expand(graph, c);
        ggml_backend_event_t gate_event = ggml_backend_event_new(device);
        GGML_ASSERT(gate_event != nullptr);
        GGML_ASSERT(ggml_backend_graph_compute_async(gate_backend.get(), graph) == GGML_STATUS_SUCCESS);
        ggml_backend_event_record(gate_event, gate_backend.get());

        llm_transfer_ring_config cancel_config = { diagnostics.lane_footprint, 1, device };
        cancel_config.trace_capacity = 32;
        cancel_config.h2d_gate_event_for_testing = gate_event;
        llm_expert_transfer_ring cancelling(cancel_config);
        GGML_ASSERT(cancelling.initialize(source.bundle()).is_ready());
        llm_expert_scheduler scheduler({ 1, 4, 2, 1, 0 });
        const llm_expert_key key = { 0, 0 };
        const auto admitted = scheduler.enqueue(key, llm_expert_priority::demand_current_layer,
            llm_expert_readiness::device_ready);
        GGML_ASSERT(admitted.disposition == llm_expert_schedule_disposition::admitted);
        llm_expert_request_snapshot selected;
        GGML_ASSERT(scheduler.take_next(selected).disposition == llm_expert_schedule_disposition::admitted);
        GGML_ASSERT(scheduler.transition(admitted.handle, llm_expert_request_state::submitting,
            llm_expert_request_state::io_in_flight) == llm_expert_schedule_disposition::admitted);
        GGML_ASSERT(scheduler.transition(admitted.handle, llm_expert_request_state::io_in_flight,
            llm_expert_request_state::host_ready) == llm_expert_schedule_disposition::admitted);
        GGML_ASSERT(scheduler.transition(admitted.handle, llm_expert_request_state::host_ready,
            llm_expert_request_state::h2d_in_flight) == llm_expert_schedule_disposition::admitted);
        const llm_expert_flight_id cancelled_flight = {
            7, admitted.handle.slot, admitted.handle.generation, key,
        };
        llm_transfer_lane_reference cancelled_lane;
        GGML_ASSERT(cancelling.reserve(cold, cold_zero, 0, 7, cancelled_lane, cancelled_flight).is_ready());
        GGML_ASSERT(cancelling.stage(cancelled_lane, cold.bundle()).is_ready());
        GGML_ASSERT(cancelling.transfer_wave(
            backend.get(), { { cancelled_lane, hot.bundle(), 0 } }).is_ready());
        GGML_ASSERT(cancelling.diagnostics().live_h2d_events == 1);
        GGML_ASSERT(scheduler.transition(admitted.handle, llm_expert_request_state::h2d_in_flight,
            llm_expert_request_state::cancelling) == llm_expert_schedule_disposition::admitted);
        GGML_ASSERT(scheduler.transition(admitted.handle, llm_expert_request_state::cancelling,
            llm_expert_request_state::draining) == llm_expert_schedule_disposition::admitted);

        std::atomic<bool> cancel_done = false;
        std::atomic<bool> reuse_done = false;
        llm_transfer_lane_reference blocked_reuse;
        std::thread cancel_thread([&] {
            GGML_ASSERT(cancelling.cancel_after_h2d(cancelled_lane).is_ready());
            cancel_done.store(true, std::memory_order_release);
        });
        std::thread reuse_thread([&] {
            GGML_ASSERT(cancelling.reserve(cold, cold_one, 0, 8, blocked_reuse).is_ready());
            reuse_done.store(true, std::memory_order_release);
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        GGML_ASSERT(!cancel_done.load(std::memory_order_acquire));
        GGML_ASSERT(!reuse_done.load(std::memory_order_acquire));
        cancel_thread.join();
        reuse_thread.join();
        GGML_ASSERT(blocked_reuse.generation != cancelled_lane.generation);
        GGML_ASSERT(cancelling.cleanup_failed_lanes().is_ready());
        GGML_ASSERT(scheduler.finish(admitted.handle, llm_expert_request_state::cancelled) ==
            llm_expert_schedule_disposition::admitted);
        GGML_ASSERT(scheduler.release_terminal(admitted.handle) == llm_expert_schedule_disposition::admitted);
        GGML_ASSERT(scheduler.diagnostics().active_requests == 0);

        const auto retry = scheduler.enqueue(key, llm_expert_priority::demand_current_layer,
            llm_expert_readiness::device_ready);
        GGML_ASSERT(retry.disposition == llm_expert_schedule_disposition::admitted &&
            retry.handle.generation != admitted.handle.generation);
        GGML_ASSERT(scheduler.take_next(selected).disposition == llm_expert_schedule_disposition::admitted);
        GGML_ASSERT(scheduler.transition(retry.handle, llm_expert_request_state::submitting,
            llm_expert_request_state::io_in_flight) == llm_expert_schedule_disposition::admitted);
        GGML_ASSERT(scheduler.transition(retry.handle, llm_expert_request_state::io_in_flight,
            llm_expert_request_state::host_ready) == llm_expert_schedule_disposition::admitted);
        GGML_ASSERT(scheduler.transition(retry.handle, llm_expert_request_state::host_ready,
            llm_expert_request_state::h2d_in_flight) == llm_expert_schedule_disposition::admitted);
        const llm_expert_flight_id retry_flight = { 7, retry.handle.slot, retry.handle.generation, key };
        llm_transfer_lane_reference retry_lane;
        GGML_ASSERT(cancelling.reserve(cold, cold_zero, 0, 9, retry_lane, retry_flight).is_ready());
        GGML_ASSERT(cancelling.stage(retry_lane, cold.bundle()).is_ready());
        GGML_ASSERT(cancelling.transfer_wave(backend.get(), { { retry_lane, hot.bundle(), 0 } }).is_ready());
        GGML_ASSERT(cancelling.wait_for_hot(backend.get(), 0, 9).is_ready());
        GGML_ASSERT(scheduler.transition(retry.handle, llm_expert_request_state::h2d_in_flight,
            llm_expert_request_state::device_ready) == llm_expert_schedule_disposition::admitted);
        GGML_ASSERT(scheduler.finish(retry.handle, llm_expert_request_state::complete) ==
            llm_expert_schedule_disposition::admitted);
        GGML_ASSERT(scheduler.release_terminal(retry.handle) == llm_expert_schedule_disposition::admitted);
        GGML_ASSERT(cancelling.retire_hot(0, 9).is_ready());
        assert_slot_matches(source, hot, 0, 0);
        auto cancellation_diagnostics = cancelling.diagnostics();
        GGML_ASSERT(cancellation_diagnostics.h2d_event_cancellations == 1);
        GGML_ASSERT(cancellation_diagnostics.live_h2d_events == 0 &&
            cancellation_diagnostics.live_compute_events == 0 && cancellation_diagnostics.live_events == 0);
        GGML_ASSERT(cold.diagnostics().current_transfer_refs == 0);
        const auto cancellation_intervals = cancelling.completed_intervals();
        GGML_ASSERT(cancellation_intervals.size() == 2 && cancellation_intervals[0].cancelled &&
            cancellation_intervals[0].flight.request_generation == admitted.handle.generation &&
            !cancellation_intervals[1].cancelled &&
            cancellation_intervals[1].flight.request_generation == retry.handle.generation);
        GGML_ASSERT(cancelling.surrender().is_ready());
        cancellation_diagnostics = cancelling.diagnostics();
        GGML_ASSERT(cancellation_diagnostics.h2d_event_frees == 1 &&
            cancellation_diagnostics.compute_event_frees == 1 &&
            cancellation_diagnostics.h2d_event_allocations == cancellation_diagnostics.h2d_event_frees &&
            cancellation_diagnostics.compute_event_allocations == cancellation_diagnostics.compute_event_frees &&
            cancellation_diagnostics.event_capacity == 0);
        GGML_ASSERT(cancelling.validate_invariants().is_ready());
        ggml_backend_event_free(gate_event);
    }

    {
        llm_expert_transfer_ring unloading({ 1U << 20, 2, device, false, false, false, 0, 32 });
        GGML_ASSERT(unloading.initialize(source.bundle()).is_ready());
        GGML_ASSERT(unloading.reserve(cold, cold_zero, 0, 2, lane_zero).is_ready());
        GGML_ASSERT(unloading.stage(lane_zero, cold.bundle()).is_ready());
        GGML_ASSERT(unloading.transfer_wave(backend.get(), { { lane_zero, hot.bundle(), 0 } }).is_ready());
        GGML_ASSERT(cold.diagnostics().current_transfer_refs == 1);
    }
    GGML_ASSERT(cold.diagnostics().current_transfer_refs == 0);
}

} // namespace

int main() {
    test_budget_fallback_and_wave();
    test_failures_cleanup_and_busy_surrender();
    test_budget_and_generation_rejection();
    test_native_event_ordering_reuse_and_unload();
    std::cout << "expert transfer ring tests passed\n";
    return 0;
}
