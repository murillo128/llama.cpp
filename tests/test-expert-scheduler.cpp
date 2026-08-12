#include "llama-expert-scheduler.h"

#include "ggml.h"

#include <array>
#include <chrono>
#include <future>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

struct phase10_scheduler_evidence {
    uint64_t budget_rejections = 0;
    uint64_t demand_promotions = 0;
    uint64_t queued_speculative_cancellations = 0;
    uint64_t demand_preemptions = 0;
    bool retry_new_generation = false;
    bool submitted_shutdown_drained = false;
    bool zero_final_charges = false;
    bool cancelling_promotion_fenced = false;
    bool submitted_demand_set_headroom_preserved = false;
};

phase10_scheduler_evidence phase10_evidence;

template<class F> void expect_invalid(F fn) {
    bool rejected = false;
    try { fn(); } catch (const std::invalid_argument &) { rejected = true; }
    GGML_ASSERT(rejected);
}

llm_expert_scheduler_config config(uint32_t capacity = 4, uint64_t generation = 0) {
    return { 2, 8, capacity, 4, generation };
}

llm_expert_scheduler_config bounded_config() {
    return {
        2, 8, 4, 4, 0,
        2, 300, 200, 200, 150, 2, 2, 2,
    };
}

llm_expert_request_metadata speculative_metadata(
        uint64_t request, uint64_t token, int32_t layer,
        uint64_t storage, uint64_t h2d) {
    return {
        llm_expert_request_origin::speculative,
        0x1234,
        request,
        token,
        layer,
        token + 1,
        storage,
        h2d,
        1,
        1,
    };
}

void test_configuration() {
    expect_invalid([] { llm_expert_scheduler scheduler({}); });
    expect_invalid([] { llm_expert_scheduler scheduler({ 1, 1, 0, 1, 0 }); });
    expect_invalid([] { llm_expert_scheduler scheduler({ 1, 1, 1, 0, 0 }); });
    expect_invalid([] {
        llm_expert_scheduler scheduler({
            1, 2, 1, 1, 0,
            1, 1, 1, 1, 1, 1, 1,
        });
    });
    expect_invalid([] {
        llm_expert_scheduler scheduler({
            1, 8, 4, 4, 0,
            3, 300, 300, 100, 100, 3, 3, 2,
        });
    });
    llm_expert_scheduler scheduler(config());
    const auto diagnostics = scheduler.diagnostics();
    GGML_ASSERT(diagnostics.request_capacity == 4);
    GGML_ASSERT(diagnostics.waiters_per_request == 4);
    GGML_ASSERT(diagnostics.max_current_layer_demand_flights == 0);
    GGML_ASSERT(diagnostics.administration_bytes > 0);
}

void test_priority_fifo_and_promotion() {
    llm_expert_scheduler scheduler(config());
    const auto speculative = scheduler.enqueue(
        { 0, 0 }, llm_expert_priority::prefetch_speculative, llm_expert_readiness::host_ready);
    const auto first = scheduler.enqueue(
        { 0, 1 }, llm_expert_priority::demand_current_layer, llm_expert_readiness::host_ready);
    const auto second = scheduler.enqueue(
        { 0, 2 }, llm_expert_priority::demand_current_layer, llm_expert_readiness::host_ready);
    GGML_ASSERT(speculative.accepted() && first.accepted() && second.accepted());
    const auto joined = scheduler.enqueue(
        { 0, 0 }, llm_expert_priority::demand_current_layer, llm_expert_readiness::device_ready);
    GGML_ASSERT(joined.disposition == llm_expert_schedule_disposition::joined);
    GGML_ASSERT(joined.handle.slot == speculative.handle.slot && joined.handle.generation == speculative.handle.generation);

    llm_expert_request_snapshot request;
    GGML_ASSERT(scheduler.take_next(request).accepted());
    GGML_ASSERT(request.key.expert == 0);
    GGML_ASSERT(request.priority == llm_expert_priority::demand_current_layer);
    GGML_ASSERT(request.readiness == llm_expert_readiness::device_ready);
    GGML_ASSERT(scheduler.transition(request.handle, llm_expert_request_state::submitting,
        llm_expert_request_state::io_in_flight) == llm_expert_schedule_disposition::admitted);
    GGML_ASSERT(scheduler.transition(request.handle, llm_expert_request_state::io_in_flight,
        llm_expert_request_state::host_ready) == llm_expert_schedule_disposition::admitted);
    GGML_ASSERT(scheduler.finish(request.handle, llm_expert_request_state::complete) ==
        llm_expert_schedule_disposition::admitted);
    GGML_ASSERT(scheduler.release_terminal(request.handle) == llm_expert_schedule_disposition::admitted);

    GGML_ASSERT(scheduler.take_next(request).accepted() && request.key.expert == 1);
    GGML_ASSERT(scheduler.transition(request.handle, llm_expert_request_state::submitting,
        llm_expert_request_state::io_in_flight) == llm_expert_schedule_disposition::admitted);
    GGML_ASSERT(scheduler.transition(request.handle, llm_expert_request_state::io_in_flight,
        llm_expert_request_state::host_ready) == llm_expert_schedule_disposition::admitted);
    GGML_ASSERT(scheduler.finish(request.handle, llm_expert_request_state::complete) ==
        llm_expert_schedule_disposition::admitted);
    GGML_ASSERT(scheduler.release_terminal(request.handle) == llm_expert_schedule_disposition::admitted);
    GGML_ASSERT(scheduler.take_next(request).accepted() && request.key.expert == 2);

    const auto diagnostics = scheduler.diagnostics();
    GGML_ASSERT(diagnostics.joins == 1);
    GGML_ASSERT(diagnostics.promotions == 2);
}

void test_saturation_and_preemption() {
    llm_expert_scheduler scheduler(config(2));
    GGML_ASSERT(scheduler.enqueue(
        { 0, 0 }, llm_expert_priority::prefetch_speculative, llm_expert_readiness::host_ready).accepted());
    GGML_ASSERT(scheduler.enqueue(
        { 0, 1 }, llm_expert_priority::prefetch_next, llm_expert_readiness::host_ready).accepted());
    GGML_ASSERT(scheduler.enqueue(
        { 0, 2 }, llm_expert_priority::prefetch_speculative, llm_expert_readiness::host_ready).disposition ==
        llm_expert_schedule_disposition::dropped);
    const auto demand = scheduler.enqueue(
        { 0, 3 }, llm_expert_priority::demand_current_layer, llm_expert_readiness::device_ready);
    GGML_ASSERT(demand.accepted());
    const auto diagnostics = scheduler.diagnostics();
    GGML_ASSERT(diagnostics.active_requests == 2);
    GGML_ASSERT(diagnostics.peak_active_requests == 2);
    GGML_ASSERT(diagnostics.drops == 2);
    GGML_ASSERT(diagnostics.queued_preemptions == 1);
}

void test_transactional_batch_admission() {
    {
        llm_expert_scheduler scheduler(config(2));
        const std::array<llm_expert_schedule_batch_item, 3> items = {{
            { { 0, 0 }, llm_expert_priority::demand_current_layer,
                llm_expert_readiness::host_ready, {} },
            { { 0, 1 }, llm_expert_priority::demand_current_layer,
                llm_expert_readiness::host_ready, {} },
            { { 0, 2 }, llm_expert_priority::demand_current_layer,
                llm_expert_readiness::host_ready, {} },
        }};
        std::array<llm_expert_schedule_result, 3> results;
        GGML_ASSERT(scheduler.enqueue_batch(items.data(), items.size(), results.data()) ==
            llm_expert_schedule_disposition::busy);
        auto diagnostics = scheduler.diagnostics();
        GGML_ASSERT(diagnostics.active_requests == 0 && diagnostics.flights_created == 0 &&
            diagnostics.joins == 0);
        const auto first = scheduler.enqueue(
            { 0, 7 }, llm_expert_priority::demand_current_layer,
            llm_expert_readiness::host_ready);
        GGML_ASSERT(first.accepted() && first.handle.slot == 0 && first.handle.generation == 1);
    }

    {
        llm_expert_scheduler scheduler(config(3));
        const auto existing = scheduler.enqueue(
            { 0, 0 }, llm_expert_priority::demand_current_layer,
            llm_expert_readiness::host_ready);
        GGML_ASSERT(existing.accepted());
        const std::array<llm_expert_schedule_batch_item, 2> items = {{
            { { 0, 0 }, llm_expert_priority::demand_current_layer,
                llm_expert_readiness::host_ready, {} },
            { { 0, 1 }, llm_expert_priority::demand_current_layer,
                llm_expert_readiness::host_ready, {} },
        }};
        std::array<llm_expert_schedule_result, 2> results;
        GGML_ASSERT(scheduler.enqueue_batch(items.data(), items.size(), results.data()) ==
            llm_expert_schedule_disposition::admitted);
        GGML_ASSERT(results[0].disposition == llm_expert_schedule_disposition::joined &&
            results[0].handle.slot == existing.handle.slot &&
            results[0].handle.generation == existing.handle.generation &&
            results[1].disposition == llm_expert_schedule_disposition::admitted);
        const auto diagnostics = scheduler.diagnostics();
        GGML_ASSERT(diagnostics.active_requests == 2 && diagnostics.flights_created == 2 &&
            diagnostics.joins == 1);
        llm_expert_request_snapshot selected;
        GGML_ASSERT(scheduler.take(results[1].handle, selected).accepted() &&
            selected.key.expert == 1);
        GGML_ASSERT(scheduler.take_next(selected).accepted() && selected.key.expert == 0);
    }

    {
        llm_expert_scheduler scheduler(config(2));
        llm_expert_request_metadata class_one;
        class_one.layout_class_id = 1;
        const auto existing = scheduler.enqueue(
            { 0, 0 }, llm_expert_priority::demand_current_layer,
            llm_expert_readiness::host_ready, class_one);
        GGML_ASSERT(existing.accepted());
        const std::array<llm_expert_schedule_batch_item, 2> items = {{
            { { 0, 1 }, llm_expert_priority::demand_current_layer,
                llm_expert_readiness::host_ready, {} },
            { { 0, 0 }, llm_expert_priority::demand_current_layer,
                llm_expert_readiness::host_ready, {} },
        }};
        std::array<llm_expert_schedule_result, 2> results;
        GGML_ASSERT(scheduler.enqueue_batch(items.data(), items.size(), results.data()) ==
            llm_expert_schedule_disposition::invalid);
        const auto diagnostics = scheduler.diagnostics();
        GGML_ASSERT(diagnostics.active_requests == 1 && diagnostics.flights_created == 1 &&
            diagnostics.joins == 0);
        const auto next = scheduler.enqueue(
            { 0, 1 }, llm_expert_priority::demand_current_layer,
            llm_expert_readiness::host_ready);
        GGML_ASSERT(next.accepted() && next.handle.slot != existing.handle.slot);
    }

    {
        llm_expert_scheduler scheduler(config(3));
        const auto predecessor = scheduler.enqueue(
            { 0, 0 }, llm_expert_priority::demand_current_layer,
            llm_expert_readiness::host_ready);
        llm_expert_request_snapshot selected;
        GGML_ASSERT(predecessor.accepted() &&
            scheduler.take(predecessor.handle, selected).accepted() &&
            scheduler.transition(predecessor.handle, llm_expert_request_state::submitting,
                llm_expert_request_state::io_in_flight) ==
                llm_expert_schedule_disposition::admitted &&
            scheduler.transition(predecessor.handle, llm_expert_request_state::io_in_flight,
                llm_expert_request_state::draining) ==
                llm_expert_schedule_disposition::admitted);
        const std::array<llm_expert_schedule_batch_item, 2> items = {{
            { { 0, 0 }, llm_expert_priority::demand_current_layer,
                llm_expert_readiness::host_ready, {} },
            { { 0, 1 }, llm_expert_priority::demand_current_layer,
                llm_expert_readiness::host_ready, {} },
        }};
        std::array<llm_expert_schedule_result, 2> results;
        GGML_ASSERT(scheduler.enqueue_batch(items.data(), items.size(), results.data()) ==
            llm_expert_schedule_disposition::admitted);
        GGML_ASSERT(results[0].disposition == llm_expert_schedule_disposition::pending_successor &&
            results[0].handle.slot == predecessor.handle.slot &&
            results[0].handle.generation == predecessor.handle.generation + 1 &&
            results[0].blocking_handle.slot == predecessor.handle.slot &&
            results[0].blocking_handle.generation == predecessor.handle.generation &&
            results[1].disposition == llm_expert_schedule_disposition::admitted &&
            scheduler.take(results[1].handle, selected).accepted() && selected.key.expert == 1);
        GGML_ASSERT(scheduler.finish(predecessor.handle, llm_expert_request_state::failed) ==
            llm_expert_schedule_disposition::admitted);
        GGML_ASSERT(scheduler.release_terminal(predecessor.handle) ==
            llm_expert_schedule_disposition::admitted);
        GGML_ASSERT(scheduler.take(results[0].handle, selected).accepted() &&
            selected.key.expert == 0 && selected.handle.generation != predecessor.handle.generation);
        const auto diagnostics = scheduler.diagnostics();
        GGML_ASSERT(diagnostics.pending_successors == 1 &&
            diagnostics.successor_activations == 1 &&
            diagnostics.successor_cancellations == 0);
    }

    {
        llm_expert_scheduler scheduler(config(2));
        const auto predecessor = scheduler.enqueue(
            { 0, 0 }, llm_expert_priority::demand_current_layer,
            llm_expert_readiness::host_ready);
        llm_expert_request_snapshot selected;
        GGML_ASSERT(predecessor.accepted() &&
            scheduler.take(predecessor.handle, selected).accepted() &&
            scheduler.transition(predecessor.handle, llm_expert_request_state::submitting,
                llm_expert_request_state::draining) ==
                llm_expert_schedule_disposition::admitted);
        const llm_expert_schedule_batch_item item = {
            { 0, 0 }, llm_expert_priority::demand_current_layer,
            llm_expert_readiness::host_ready, {}
        };
        llm_expert_schedule_result result;
        GGML_ASSERT(scheduler.enqueue_batch(&item, 1, &result) ==
            llm_expert_schedule_disposition::admitted &&
            result.disposition == llm_expert_schedule_disposition::pending_successor &&
            scheduler.cancel_pending_successor(result.handle) ==
                llm_expert_schedule_disposition::admitted);
        GGML_ASSERT(scheduler.finish(predecessor.handle, llm_expert_request_state::failed) ==
            llm_expert_schedule_disposition::admitted &&
            scheduler.release_terminal(predecessor.handle) ==
                llm_expert_schedule_disposition::admitted);
        const auto diagnostics = scheduler.diagnostics();
        GGML_ASSERT(diagnostics.active_requests == 0 &&
            diagnostics.pending_successors == 1 &&
            diagnostics.successor_activations == 0 &&
            diagnostics.successor_cancellations == 1);
    }

    {
        llm_expert_scheduler scheduler(config(1));
        const auto predecessor = scheduler.enqueue(
            { 0, 0 }, llm_expert_priority::demand_current_layer,
            llm_expert_readiness::host_ready);
        llm_expert_request_snapshot selected;
        GGML_ASSERT(predecessor.accepted() &&
            scheduler.take(predecessor.handle, selected).accepted() &&
            scheduler.transition(predecessor.handle, llm_expert_request_state::submitting,
                llm_expert_request_state::draining) ==
                llm_expert_schedule_disposition::admitted &&
            scheduler.finish(predecessor.handle, llm_expert_request_state::failed) ==
                llm_expert_schedule_disposition::admitted);
        const llm_expert_schedule_batch_item item = {
            { 0, 0 }, llm_expert_priority::demand_current_layer,
            llm_expert_readiness::host_ready, {}
        };
        llm_expert_schedule_result result;
        GGML_ASSERT(scheduler.enqueue_batch(&item, 1, &result) ==
            llm_expert_schedule_disposition::admitted &&
            result.disposition == llm_expert_schedule_disposition::pending_successor &&
            result.blocking_handle.generation == predecessor.handle.generation &&
            scheduler.release_terminal(predecessor.handle) ==
                llm_expert_schedule_disposition::admitted &&
            scheduler.take(result.handle, selected).accepted() &&
            selected.handle.generation == predecessor.handle.generation + 1);
    }

    {
        llm_expert_scheduler scheduler(config(
            2, std::numeric_limits<uint64_t>::max() - 1));
        const auto predecessor = scheduler.enqueue(
            { 0, 0 }, llm_expert_priority::demand_current_layer,
            llm_expert_readiness::host_ready);
        llm_expert_request_snapshot selected;
        GGML_ASSERT(predecessor.accepted() &&
            predecessor.handle.generation == std::numeric_limits<uint64_t>::max() &&
            scheduler.take(predecessor.handle, selected).accepted() &&
            scheduler.transition(predecessor.handle, llm_expert_request_state::submitting,
                llm_expert_request_state::draining) ==
                llm_expert_schedule_disposition::admitted);
        const std::array<llm_expert_schedule_batch_item, 2> items = {{
            { { 0, 0 }, llm_expert_priority::demand_current_layer,
                llm_expert_readiness::host_ready, {} },
            { { 0, 1 }, llm_expert_priority::demand_current_layer,
                llm_expert_readiness::host_ready, {} },
        }};
        std::array<llm_expert_schedule_result, 2> results;
        GGML_ASSERT(scheduler.enqueue_batch(items.data(), items.size(), results.data()) ==
            llm_expert_schedule_disposition::generation_exhausted);
        const auto diagnostics = scheduler.diagnostics();
        GGML_ASSERT(diagnostics.active_requests == 1 &&
            diagnostics.flights_created == 1 && diagnostics.pending_successors == 0);
        GGML_ASSERT(scheduler.finish(predecessor.handle, llm_expert_request_state::failed) ==
            llm_expert_schedule_disposition::admitted &&
            scheduler.release_terminal(predecessor.handle) ==
                llm_expert_schedule_disposition::admitted);
    }
}

void test_stale_completion_and_generation_exhaustion() {
    llm_expert_scheduler scheduler(config(1));
    const auto admitted = scheduler.enqueue(
        { 0, 0 }, llm_expert_priority::demand_current_layer, llm_expert_readiness::host_ready);
    GGML_ASSERT(admitted.accepted());
    llm_expert_request_snapshot request;
    GGML_ASSERT(scheduler.take_next(request).accepted());
    GGML_ASSERT(scheduler.transition(admitted.handle, llm_expert_request_state::submitting,
        llm_expert_request_state::io_in_flight) == llm_expert_schedule_disposition::admitted);
    GGML_ASSERT(scheduler.transition(admitted.handle, llm_expert_request_state::io_in_flight,
        llm_expert_request_state::host_ready) == llm_expert_schedule_disposition::admitted);
    GGML_ASSERT(scheduler.finish(admitted.handle, llm_expert_request_state::complete) ==
        llm_expert_schedule_disposition::admitted);
    GGML_ASSERT(scheduler.release_terminal(admitted.handle) == llm_expert_schedule_disposition::admitted);
    const auto next = scheduler.enqueue(
        { 0, 1 }, llm_expert_priority::demand_current_layer, llm_expert_readiness::host_ready);
    GGML_ASSERT(next.accepted() && next.handle.generation != admitted.handle.generation);
    GGML_ASSERT(scheduler.finish(admitted.handle, llm_expert_request_state::complete) ==
        llm_expert_schedule_disposition::stale_generation);
    GGML_ASSERT(scheduler.diagnostics().stale_completions == 1);

    llm_expert_scheduler exhausted(config(1, std::numeric_limits<uint64_t>::max()));
    GGML_ASSERT(exhausted.enqueue(
        { 0, 0 }, llm_expert_priority::demand_current_layer, llm_expert_readiness::host_ready).disposition ==
        llm_expert_schedule_disposition::generation_exhausted);
}

void test_layout_class_identity_is_part_of_join_state() {
    llm_expert_scheduler scheduler(config(2));
    llm_expert_request_metadata class_two;
    class_two.layout_class_id = 2;
    const auto admitted = scheduler.enqueue(
        { 0, 0 }, llm_expert_priority::demand_current_layer,
        llm_expert_readiness::device_ready, class_two);
    GGML_ASSERT(admitted.disposition == llm_expert_schedule_disposition::admitted);

    llm_expert_request_metadata class_three;
    class_three.layout_class_id = 3;
    GGML_ASSERT(scheduler.enqueue(
        { 0, 0 }, llm_expert_priority::demand_current_layer,
        llm_expert_readiness::device_ready, class_three).disposition ==
        llm_expert_schedule_disposition::invalid);
    const auto joined = scheduler.enqueue(
        { 0, 0 }, llm_expert_priority::demand_current_layer,
        llm_expert_readiness::device_ready, class_two);
    GGML_ASSERT(joined.disposition == llm_expert_schedule_disposition::joined);
    GGML_ASSERT(joined.handle.slot == admitted.handle.slot &&
        joined.handle.generation == admitted.handle.generation);

    llm_expert_request_snapshot snapshot;
    GGML_ASSERT(scheduler.take_next(snapshot).accepted());
    GGML_ASSERT(snapshot.metadata.layout_class_id == class_two.layout_class_id);
    GGML_ASSERT(scheduler.transition(snapshot.handle, llm_expert_request_state::submitting,
        llm_expert_request_state::host_ready) == llm_expert_schedule_disposition::admitted);
    GGML_ASSERT(scheduler.finish(snapshot.handle, llm_expert_request_state::complete) ==
        llm_expert_schedule_disposition::admitted);
    GGML_ASSERT(scheduler.release_terminal(snapshot.handle) ==
        llm_expert_schedule_disposition::admitted);
}

void test_device_qualified_single_flight_and_backpressure() {
    for (int32_t expert = 0; expert < 16; ++expert) {
        GGML_ASSERT(llm_expert_owner_device(expert, 2) == llm_expert_device_id(expert & 1));
    }
    GGML_ASSERT(llm_expert_owner_device(-1, 2) == LLM_EXPERT_DEVICE_ID_INVALID);
    GGML_ASSERT(llm_expert_owner_device(0, 0) == LLM_EXPERT_DEVICE_ID_INVALID);

    auto scheduler_config = config(4);
    scheduler_config.device_count = 2;
    scheduler_config.per_device_request_capacity = 2;
    scheduler_config.per_device_inflight_capacity = 1;
    llm_expert_scheduler scheduler(scheduler_config);

    llm_expert_request_metadata gpu0;
    gpu0.target_device = 0;
    gpu0.reserved_storage_bytes = 10;
    gpu0.reserved_h2d_bytes = 20;
    llm_expert_request_metadata gpu1 = gpu0;
    gpu1.target_device = 1;

    const auto first0 = scheduler.enqueue(
        { 0, 0 }, llm_expert_priority::demand_current_layer,
        llm_expert_readiness::device_ready, gpu0);
    const auto first1 = scheduler.enqueue(
        { 0, 0 }, llm_expert_priority::demand_current_layer,
        llm_expert_readiness::device_ready, gpu1);
    GGML_ASSERT(first0.disposition == llm_expert_schedule_disposition::admitted);
    GGML_ASSERT(first1.disposition == llm_expert_schedule_disposition::admitted);
    GGML_ASSERT(first0.handle.target_device == 0 && first1.handle.target_device == 1);
    GGML_ASSERT(first0.handle.slot != first1.handle.slot);

    const auto joined0 = scheduler.enqueue(
        { 0, 0 }, llm_expert_priority::demand_current_layer,
        llm_expert_readiness::device_ready, gpu0);
    GGML_ASSERT(joined0.disposition == llm_expert_schedule_disposition::joined);
    GGML_ASSERT(joined0.handle.slot == first0.handle.slot);

    GGML_ASSERT(scheduler.enqueue(
        { 0, 1 }, llm_expert_priority::demand_current_layer,
        llm_expert_readiness::device_ready, gpu0).accepted());
    GGML_ASSERT(scheduler.enqueue(
        { 0, 2 }, llm_expert_priority::demand_current_layer,
        llm_expert_readiness::device_ready, gpu0).disposition ==
        llm_expert_schedule_disposition::busy);

    llm_expert_request_snapshot selected0;
    GGML_ASSERT(scheduler.take_next(selected0).accepted());
    GGML_ASSERT(selected0.metadata.target_device == 0);
    llm_expert_request_snapshot selected1;
    GGML_ASSERT(scheduler.take_next(selected1).accepted());
    GGML_ASSERT(selected1.metadata.target_device == 1);

    const auto wrong_device = llm_expert_request_handle {
        selected0.handle.slot, selected0.handle.generation, 1 };
    GGML_ASSERT(scheduler.transition(wrong_device,
        llm_expert_request_state::submitting,
        llm_expert_request_state::host_ready) ==
        llm_expert_schedule_disposition::stale_generation);

    const auto diagnostics = scheduler.diagnostics();
    GGML_ASSERT(diagnostics.devices.size() == 2);
    GGML_ASSERT(diagnostics.devices[0].active_requests == 2);
    GGML_ASSERT(diagnostics.devices[1].active_requests == 1);
    GGML_ASSERT(diagnostics.devices[0].inflight_requests == 1);
    GGML_ASSERT(diagnostics.devices[1].inflight_requests == 1);
    GGML_ASSERT(diagnostics.devices[0].reserved_storage_bytes == 20);
    GGML_ASSERT(diagnostics.devices[1].reserved_h2d_bytes == 20);
}

void test_exact_unequal_device_capacities() {
    auto exact = config(8);
    exact.device_count = 2;
    exact.device_request_capacities[0] = 2;
    exact.device_request_capacities[1] = 6;
    exact.device_inflight_capacities[0] = 1;
    exact.device_inflight_capacities[1] = 3;
    llm_expert_scheduler scheduler(exact);
    auto diagnostics = scheduler.diagnostics();
    GGML_ASSERT(diagnostics.devices.size() == 2);
    GGML_ASSERT(diagnostics.devices[0].request_capacity == 2);
    GGML_ASSERT(diagnostics.devices[1].request_capacity == 6);
    GGML_ASSERT(diagnostics.devices[0].inflight_capacity == 1);
    GGML_ASSERT(diagnostics.devices[1].inflight_capacity == 3);

    llm_expert_request_metadata first;
    first.target_device = 0;
    GGML_ASSERT(scheduler.enqueue({ 0, 0 }, llm_expert_priority::demand_current_layer,
        llm_expert_readiness::device_ready, first).accepted());
    GGML_ASSERT(scheduler.enqueue({ 0, 1 }, llm_expert_priority::demand_current_layer,
        llm_expert_readiness::device_ready, first).accepted());
    GGML_ASSERT(scheduler.enqueue({ 0, 2 }, llm_expert_priority::demand_current_layer,
        llm_expert_readiness::device_ready, first).disposition ==
        llm_expert_schedule_disposition::busy);

    llm_expert_request_metadata second;
    second.target_device = 1;
    for (int32_t expert = 2; expert < 6; ++expert) {
        GGML_ASSERT(scheduler.enqueue({ 1, expert }, llm_expert_priority::demand_current_layer,
            llm_expert_readiness::device_ready, second).accepted());
    }

    auto invalid = exact;
    invalid.device_request_capacities[1] = 7;
    expect_invalid([&] { llm_expert_scheduler rejected(invalid); });
    invalid = exact;
    invalid.device_request_capacities[1] = 0;
    expect_invalid([&] { llm_expert_scheduler rejected(invalid); });
}

void test_quiescent_shutdown() {
    llm_expert_scheduler scheduler(config());
    GGML_ASSERT(scheduler.enqueue(
        { 0, 0 }, llm_expert_priority::prefetch_next, llm_expert_readiness::host_ready).accepted());
    GGML_ASSERT(scheduler.shutdown());
    const auto diagnostics = scheduler.diagnostics();
    GGML_ASSERT(diagnostics.admission_closed);
    GGML_ASSERT(diagnostics.active_requests == 0);
    GGML_ASSERT(scheduler.enqueue(
        { 0, 1 }, llm_expert_priority::demand_current_layer, llm_expert_readiness::host_ready).disposition ==
        llm_expert_schedule_disposition::closed);
}

void test_post_h2d_cancellation_path() {
    llm_expert_scheduler scheduler(config(1));
    const auto admitted = scheduler.enqueue(
        { 0, 0 }, llm_expert_priority::demand_current_layer, llm_expert_readiness::device_ready);
    GGML_ASSERT(admitted.accepted());
    llm_expert_request_snapshot request;
    GGML_ASSERT(scheduler.take_next(request).accepted());
    GGML_ASSERT(scheduler.transition(admitted.handle, llm_expert_request_state::submitting,
        llm_expert_request_state::io_in_flight) == llm_expert_schedule_disposition::admitted);
    GGML_ASSERT(scheduler.transition(admitted.handle, llm_expert_request_state::io_in_flight,
        llm_expert_request_state::host_ready) == llm_expert_schedule_disposition::admitted);
    GGML_ASSERT(scheduler.transition(admitted.handle, llm_expert_request_state::host_ready,
        llm_expert_request_state::h2d_in_flight) == llm_expert_schedule_disposition::admitted);
    GGML_ASSERT(scheduler.begin_demand_cancellation(
        admitted.handle, llm_expert_request_state::h2d_in_flight) ==
        llm_expert_schedule_disposition::admitted);
    GGML_ASSERT(scheduler.transition(admitted.handle, llm_expert_request_state::cancelling,
        llm_expert_request_state::draining) == llm_expert_schedule_disposition::admitted);
    GGML_ASSERT(scheduler.finish(admitted.handle, llm_expert_request_state::cancelled) ==
        llm_expert_schedule_disposition::admitted);
    GGML_ASSERT(scheduler.release_terminal(admitted.handle) == llm_expert_schedule_disposition::admitted);
    GGML_ASSERT(scheduler.diagnostics().active_requests == 0);
}

void test_cold_hit_reaches_host_ready_without_io() {
    llm_expert_scheduler scheduler(config(1));
    const auto admitted = scheduler.enqueue(
        { 0, 0 }, llm_expert_priority::demand_current_layer, llm_expert_readiness::device_ready);
    GGML_ASSERT(admitted.accepted());
    llm_expert_request_snapshot request;
    GGML_ASSERT(scheduler.take_next(request).accepted());
    GGML_ASSERT(scheduler.transition(admitted.handle, llm_expert_request_state::submitting,
        llm_expert_request_state::host_ready) == llm_expert_schedule_disposition::admitted);
    GGML_ASSERT(scheduler.finish(admitted.handle, llm_expert_request_state::complete) ==
        llm_expert_schedule_disposition::admitted);
    GGML_ASSERT(scheduler.release_terminal(admitted.handle) == llm_expert_schedule_disposition::admitted);
}

void test_concurrent_duplicate_joins() {
    llm_expert_scheduler scheduler({ 1, 8, 2, 8, 0 });
    std::vector<llm_expert_schedule_result> results(8);
    std::vector<std::thread> threads;
    for (uint32_t index = 0; index < results.size(); ++index) {
        threads.emplace_back([&scheduler, &results, index] {
            results[index] = scheduler.enqueue(
                { 0, 3 }, llm_expert_priority::demand_current_layer, llm_expert_readiness::device_ready);
        });
    }
    for (auto & thread : threads) {
        thread.join();
    }
    uint32_t accepted = 0;
    for (const auto & result : results) {
        if (result.accepted()) {
            accepted++;
        }
    }
    GGML_ASSERT(accepted == 8);
    const auto diagnostics = scheduler.diagnostics();
    GGML_ASSERT(diagnostics.flights_created == 1);
    GGML_ASSERT(diagnostics.joins == 7);
    GGML_ASSERT(diagnostics.active_requests == 1);
    GGML_ASSERT(scheduler.shutdown());
}

void test_speculative_budgets_cancel_and_demand_promotion() {
    llm_expert_scheduler scheduler(bounded_config());
    const auto first = scheduler.enqueue(
        { 0, 1 }, llm_expert_priority::prefetch_next, llm_expert_readiness::host_ready,
        speculative_metadata(7, 3, 0, 100, 50));
    GGML_ASSERT(first.disposition == llm_expert_schedule_disposition::admitted);
    auto diagnostics = scheduler.diagnostics();
    GGML_ASSERT(diagnostics.active_speculative_flights == 1);
    GGML_ASSERT(diagnostics.speculative_storage_bytes_in_flight == 100);
    GGML_ASSERT(diagnostics.speculative_h2d_bytes_in_flight == 50);
    GGML_ASSERT(diagnostics.speculative_cold_slots == 1 && diagnostics.speculative_hot_slots == 1);

    // The per-token storage ceiling rejects the second flight without changing
    // any live charge or delaying later demand.
    const auto over_token = scheduler.enqueue(
        { 0, 2 }, llm_expert_priority::prefetch_next, llm_expert_readiness::host_ready,
        speculative_metadata(7, 3, 0, 101, 50));
    GGML_ASSERT(over_token.disposition == llm_expert_schedule_disposition::dropped);
    diagnostics = scheduler.diagnostics();
    GGML_ASSERT(diagnostics.speculative_budget_rejections == 1);
    GGML_ASSERT(diagnostics.active_speculative_flights == 1);

    // Exact demand joins and promotes the same generation. It becomes
    // demand-owned and can no longer be cancelled through the speculative seam.
    const auto demand = scheduler.enqueue(
        { 0, 1 }, llm_expert_priority::demand_current_layer,
        llm_expert_readiness::device_ready);
    GGML_ASSERT(demand.disposition == llm_expert_schedule_disposition::joined);
    GGML_ASSERT(demand.handle.slot == first.handle.slot && demand.handle.generation == first.handle.generation);
    GGML_ASSERT(scheduler.cancel_queued_speculative(first.handle) ==
        llm_expert_schedule_disposition::invalid);
    diagnostics = scheduler.diagnostics();
    GGML_ASSERT(diagnostics.demand_promotions == 1);
    GGML_ASSERT(diagnostics.active_speculative_flights == 0);
    llm_expert_request_snapshot promoted;
    GGML_ASSERT(scheduler.take_next(promoted).accepted());
    GGML_ASSERT(promoted.promoted_from_speculative);
    GGML_ASSERT(promoted.metadata.origin == llm_expert_request_origin::demand);
    GGML_ASSERT(promoted.handle.generation == first.handle.generation);
    GGML_ASSERT(scheduler.transition(promoted.handle, llm_expert_request_state::submitting,
        llm_expert_request_state::host_ready) == llm_expert_schedule_disposition::admitted);
    GGML_ASSERT(scheduler.finish(promoted.handle, llm_expert_request_state::complete) ==
        llm_expert_schedule_disposition::admitted);
    GGML_ASSERT(scheduler.release_terminal(promoted.handle) ==
        llm_expert_schedule_disposition::admitted);

    const auto cancellable = scheduler.enqueue(
        { 1, 0 }, llm_expert_priority::prefetch_speculative,
        llm_expert_readiness::device_ready,
        speculative_metadata(8, 0, 1, 100, 100));
    GGML_ASSERT(cancellable.accepted());
    GGML_ASSERT(scheduler.cancel_queued_speculative(cancellable.handle) ==
        llm_expert_schedule_disposition::admitted);
    diagnostics = scheduler.diagnostics();
    GGML_ASSERT(diagnostics.speculative_cancelled_before_submit == 1);
    GGML_ASSERT(diagnostics.active_requests == 0 && diagnostics.active_speculative_flights == 0);
    GGML_ASSERT(diagnostics.speculative_storage_bytes_in_flight == 0 &&
        diagnostics.speculative_h2d_bytes_in_flight == 0);
    phase10_evidence.budget_rejections = diagnostics.speculative_budget_rejections;
    phase10_evidence.demand_promotions = diagnostics.demand_promotions;
    phase10_evidence.queued_speculative_cancellations =
        diagnostics.speculative_cancelled_before_submit;
}

void test_demand_preemption_submitted_drain_retry_and_shutdown() {
    const llm_expert_scheduler_config single = {
        1, 4, 2, 4, 0,
        1, 200, 200, 200, 200, 1, 1, 1,
    };
    {
        llm_expert_scheduler scheduler(single);
        const auto speculative = scheduler.enqueue(
            { 0, 0 }, llm_expert_priority::prefetch_speculative,
            llm_expert_readiness::device_ready,
            speculative_metadata(1, 0, 0, 100, 100));
        GGML_ASSERT(speculative.accepted());
        GGML_ASSERT(scheduler.enqueue(
            { 0, 3 }, llm_expert_priority::demand_future_dependency,
            llm_expert_readiness::host_ready).accepted());
        const auto demand = scheduler.enqueue(
            { 0, 1 }, llm_expert_priority::demand_current_layer,
            llm_expert_readiness::device_ready);
        GGML_ASSERT(demand.disposition == llm_expert_schedule_disposition::admitted);
        GGML_ASSERT(demand.handle.generation != speculative.handle.generation);
        const auto diagnostics = scheduler.diagnostics();
        GGML_ASSERT(diagnostics.queued_preemptions == 1 && diagnostics.drops == 1);
        phase10_evidence.demand_preemptions = diagnostics.queued_preemptions;
        GGML_ASSERT(diagnostics.active_speculative_flights == 0);
        GGML_ASSERT(scheduler.cancel_queued_speculative(speculative.handle) ==
            llm_expert_schedule_disposition::stale_generation);
        llm_expert_request_snapshot selected;
        GGML_ASSERT(scheduler.take_next(selected).accepted());
        GGML_ASSERT(selected.priority == llm_expert_priority::demand_current_layer);
        GGML_ASSERT(scheduler.transition(demand.handle, llm_expert_request_state::submitting,
            llm_expert_request_state::host_ready) == llm_expert_schedule_disposition::admitted);
        GGML_ASSERT(scheduler.finish(demand.handle, llm_expert_request_state::complete) ==
            llm_expert_schedule_disposition::admitted);
        GGML_ASSERT(scheduler.release_terminal(demand.handle) ==
            llm_expert_schedule_disposition::admitted);
    }
    {
        llm_expert_scheduler scheduler(single);
        const auto speculative = scheduler.enqueue(
            { 0, 2 }, llm_expert_priority::prefetch_next,
            llm_expert_readiness::device_ready,
            speculative_metadata(2, 4, 0, 100, 100));
        GGML_ASSERT(speculative.accepted());
        llm_expert_request_snapshot selected;
        GGML_ASSERT(scheduler.take_next(selected).accepted());
        GGML_ASSERT(scheduler.cancel_queued_speculative(speculative.handle) ==
            llm_expert_schedule_disposition::invalid);
        const auto promoted = scheduler.enqueue(
            { 0, 2 }, llm_expert_priority::demand_current_layer,
            llm_expert_readiness::device_ready);
        GGML_ASSERT(promoted.disposition == llm_expert_schedule_disposition::joined);
        GGML_ASSERT(promoted.handle.generation == speculative.handle.generation);
        GGML_ASSERT(scheduler.diagnostics().active_speculative_flights == 0);
        GGML_ASSERT(scheduler.transition(promoted.handle, llm_expert_request_state::submitting,
            llm_expert_request_state::io_in_flight) == llm_expert_schedule_disposition::admitted);
        GGML_ASSERT(scheduler.transition(promoted.handle, llm_expert_request_state::io_in_flight,
            llm_expert_request_state::draining) == llm_expert_schedule_disposition::admitted);
        GGML_ASSERT(scheduler.finish(promoted.handle, llm_expert_request_state::failed) ==
            llm_expert_schedule_disposition::admitted);
        GGML_ASSERT(scheduler.release_terminal(promoted.handle) ==
            llm_expert_schedule_disposition::admitted);
        const auto retry = scheduler.enqueue(
            { 0, 2 }, llm_expert_priority::demand_current_layer,
            llm_expert_readiness::device_ready);
        GGML_ASSERT(retry.accepted() && retry.handle.generation > promoted.handle.generation);
        phase10_evidence.retry_new_generation = true;
        GGML_ASSERT(scheduler.shutdown());
    }
    {
        llm_expert_scheduler scheduler(single);
        const auto submitted = scheduler.enqueue(
            { 0, 3 }, llm_expert_priority::prefetch_speculative,
            llm_expert_readiness::host_ready,
            speculative_metadata(3, 1, 0, 100, 0));
        GGML_ASSERT(submitted.accepted());
        llm_expert_request_snapshot selected;
        GGML_ASSERT(scheduler.take_next(selected).accepted());
        GGML_ASSERT(!scheduler.shutdown());
        GGML_ASSERT(scheduler.diagnostics().admission_closed);
        GGML_ASSERT(scheduler.transition(submitted.handle, llm_expert_request_state::submitting,
            llm_expert_request_state::draining) == llm_expert_schedule_disposition::admitted);
        GGML_ASSERT(scheduler.finish(submitted.handle, llm_expert_request_state::cancelled) ==
            llm_expert_schedule_disposition::admitted);
        GGML_ASSERT(scheduler.release_terminal(submitted.handle) ==
            llm_expert_schedule_disposition::admitted);
        GGML_ASSERT(scheduler.shutdown());
        const auto diagnostics = scheduler.diagnostics();
        GGML_ASSERT(diagnostics.active_requests == 0 && diagnostics.active_speculative_flights == 0);
        phase10_evidence.submitted_shutdown_drained = true;
        phase10_evidence.zero_final_charges = diagnostics.speculative_storage_bytes_in_flight == 0 &&
            diagnostics.speculative_h2d_bytes_in_flight == 0 &&
            diagnostics.speculative_cold_slots == 0 && diagnostics.speculative_hot_slots == 0;
    }
}

void test_cancelling_promotion_fence_and_submitted_demand_set_headroom() {
    const llm_expert_scheduler_config bounded = {
        1, 8, 2, 4, 0,
        1, 200, 200, 200, 200, 1, 1, 1,
    };
    {
        llm_expert_scheduler scheduler(bounded);
        const auto speculative = scheduler.enqueue(
            { 0, 0 }, llm_expert_priority::prefetch_next,
            llm_expert_readiness::device_ready,
            speculative_metadata(20, 0, 0, 100, 100));
        GGML_ASSERT(speculative.accepted());
        llm_expert_request_snapshot selected;
        GGML_ASSERT(scheduler.take_next(selected).accepted());
        GGML_ASSERT(scheduler.transition(speculative.handle,
            llm_expert_request_state::submitting,
            llm_expert_request_state::io_in_flight) ==
            llm_expert_schedule_disposition::admitted);
        GGML_ASSERT(scheduler.begin_speculative_cancellation(
            speculative.handle, llm_expert_request_state::io_in_flight) ==
            llm_expert_schedule_disposition::admitted);
        const auto fenced = scheduler.enqueue(
            { 0, 0 }, llm_expert_priority::demand_current_layer,
            llm_expert_readiness::device_ready);
        GGML_ASSERT(fenced.disposition == llm_expert_schedule_disposition::busy &&
            fenced.handle.slot == speculative.handle.slot &&
            fenced.handle.generation == speculative.handle.generation);
        auto waiter = std::async(std::launch::async, [&] {
            return scheduler.wait_until_released(fenced.handle);
        });
        GGML_ASSERT(waiter.wait_for(std::chrono::milliseconds(1)) == std::future_status::timeout);
        GGML_ASSERT(scheduler.transition(speculative.handle,
            llm_expert_request_state::cancelling,
            llm_expert_request_state::draining) ==
            llm_expert_schedule_disposition::admitted);
        GGML_ASSERT(scheduler.finish(speculative.handle,
            llm_expert_request_state::cancelled) ==
            llm_expert_schedule_disposition::admitted);
        GGML_ASSERT(scheduler.release_terminal(speculative.handle) ==
            llm_expert_schedule_disposition::admitted);
        GGML_ASSERT(waiter.get() == llm_expert_schedule_disposition::admitted);
        const auto retry = scheduler.enqueue(
            { 0, 0 }, llm_expert_priority::demand_current_layer,
            llm_expert_readiness::device_ready);
        GGML_ASSERT(retry.accepted() && retry.handle.generation > speculative.handle.generation);
        GGML_ASSERT(scheduler.shutdown());
        phase10_evidence.cancelling_promotion_fenced = true;
    }
    {
        llm_expert_scheduler scheduler(bounded);
        const auto speculative = scheduler.enqueue(
            { 0, 1 }, llm_expert_priority::prefetch_next,
            llm_expert_readiness::device_ready,
            speculative_metadata(21, 0, 0, 100, 100));
        GGML_ASSERT(speculative.accepted());
        llm_expert_request_snapshot selected;
        GGML_ASSERT(scheduler.take_next(selected).accepted());
        const auto promoted = scheduler.enqueue(
            { 0, 1 }, llm_expert_priority::demand_current_layer,
            llm_expert_readiness::device_ready);
        GGML_ASSERT(promoted.disposition == llm_expert_schedule_disposition::joined);
        GGML_ASSERT(scheduler.begin_speculative_cancellation(
            promoted.handle, llm_expert_request_state::submitting) ==
            llm_expert_schedule_disposition::invalid);
        GGML_ASSERT(scheduler.begin_demand_cancellation(
            promoted.handle, llm_expert_request_state::submitting) ==
            llm_expert_schedule_disposition::admitted);
        GGML_ASSERT(scheduler.transition(promoted.handle,
            llm_expert_request_state::cancelling,
            llm_expert_request_state::draining) ==
            llm_expert_schedule_disposition::admitted);
        GGML_ASSERT(scheduler.finish(promoted.handle,
            llm_expert_request_state::cancelled) ==
            llm_expert_schedule_disposition::admitted);
        GGML_ASSERT(scheduler.release_terminal(promoted.handle) ==
            llm_expert_schedule_disposition::admitted);
        GGML_ASSERT(scheduler.shutdown());
    }
    {
        const llm_expert_scheduler_config demand_set_bounded = {
            1, 8, 4, 4, 0,
            2, 400, 400, 200, 200, 2, 2, 2,
        };
        llm_expert_scheduler scheduler(demand_set_bounded);
        const auto submitted_first = scheduler.enqueue(
            { 0, 2 }, llm_expert_priority::prefetch_next,
            llm_expert_readiness::device_ready,
            speculative_metadata(22, 0, 0, 100, 100));
        const auto submitted_second = scheduler.enqueue(
            { 0, 3 }, llm_expert_priority::prefetch_next,
            llm_expert_readiness::device_ready,
            speculative_metadata(23, 0, 0, 100, 100));
        GGML_ASSERT(submitted_first.accepted() && submitted_second.accepted());
        llm_expert_request_snapshot selected;
        GGML_ASSERT(scheduler.take_next(selected).accepted());
        GGML_ASSERT(scheduler.transition(submitted_first.handle,
            llm_expert_request_state::submitting,
            llm_expert_request_state::io_in_flight) ==
            llm_expert_schedule_disposition::admitted);
        GGML_ASSERT(scheduler.take_next(selected).accepted());
        GGML_ASSERT(scheduler.transition(submitted_second.handle,
            llm_expert_request_state::submitting,
            llm_expert_request_state::io_in_flight) ==
            llm_expert_schedule_disposition::admitted);
        GGML_ASSERT(scheduler.enqueue(
            { 0, 4 }, llm_expert_priority::prefetch_speculative,
            llm_expert_readiness::host_ready,
            speculative_metadata(24, 0, 0, 100, 0)).disposition ==
            llm_expert_schedule_disposition::dropped);
        const auto demand_first = scheduler.enqueue(
            { 0, 5 }, llm_expert_priority::demand_current_layer,
            llm_expert_readiness::device_ready);
        const auto demand_second = scheduler.enqueue(
            { 0, 6 }, llm_expert_priority::demand_current_layer,
            llm_expert_readiness::device_ready);
        GGML_ASSERT(demand_first.disposition == llm_expert_schedule_disposition::admitted &&
            demand_second.disposition == llm_expert_schedule_disposition::admitted);
        GGML_ASSERT(scheduler.diagnostics().max_current_layer_demand_flights == 2);
        GGML_ASSERT(demand_first.handle.slot != demand_second.handle.slot &&
            demand_first.handle.slot != submitted_first.handle.slot &&
            demand_first.handle.slot != submitted_second.handle.slot &&
            demand_second.handle.slot != submitted_first.handle.slot &&
            demand_second.handle.slot != submitted_second.handle.slot);
        phase10_evidence.submitted_demand_set_headroom_preserved = true;
        GGML_ASSERT(!scheduler.shutdown());
        GGML_ASSERT(scheduler.transition(submitted_first.handle,
            llm_expert_request_state::io_in_flight,
            llm_expert_request_state::draining) ==
            llm_expert_schedule_disposition::admitted);
        GGML_ASSERT(scheduler.finish(submitted_first.handle,
            llm_expert_request_state::failed) ==
            llm_expert_schedule_disposition::admitted);
        GGML_ASSERT(scheduler.release_terminal(submitted_first.handle) ==
            llm_expert_schedule_disposition::admitted);
        GGML_ASSERT(scheduler.transition(submitted_second.handle,
            llm_expert_request_state::io_in_flight,
            llm_expert_request_state::draining) ==
            llm_expert_schedule_disposition::admitted);
        GGML_ASSERT(scheduler.finish(submitted_second.handle,
            llm_expert_request_state::failed) ==
            llm_expert_schedule_disposition::admitted);
        GGML_ASSERT(scheduler.release_terminal(submitted_second.handle) ==
            llm_expert_schedule_disposition::admitted);
        GGML_ASSERT(scheduler.shutdown());
    }
}

} // namespace

int main() {
    test_configuration();
    test_priority_fifo_and_promotion();
    test_saturation_and_preemption();
    test_transactional_batch_admission();
    test_stale_completion_and_generation_exhaustion();
    test_layout_class_identity_is_part_of_join_state();
    test_device_qualified_single_flight_and_backpressure();
    test_exact_unequal_device_capacities();
    test_quiescent_shutdown();
    test_post_h2d_cancellation_path();
    test_cold_hit_reaches_host_ready_without_io();
    test_concurrent_duplicate_joins();
    test_speculative_budgets_cancel_and_demand_promotion();
    test_demand_preemption_submitted_drain_retry_and_shutdown();
    test_cancelling_promotion_fence_and_submitted_demand_set_headroom();
    std::cout << "PHASE10_SCHEDULER_LIFECYCLE"
              << "\tbudget_rejections=" << phase10_evidence.budget_rejections
              << "\tdemand_promotions=" << phase10_evidence.demand_promotions
              << "\tqueued_speculative_cancellations="
              << phase10_evidence.queued_speculative_cancellations
              << "\tdemand_preemptions=" << phase10_evidence.demand_preemptions
              << "\tretry_new_generation=" << phase10_evidence.retry_new_generation
              << "\tsubmitted_shutdown_drained=" << phase10_evidence.submitted_shutdown_drained
              << "\tzero_final_charges=" << phase10_evidence.zero_final_charges
              << "\tcancelling_promotion_fenced=" << phase10_evidence.cancelling_promotion_fenced
              << "\tsubmitted_demand_set_headroom_preserved="
              << phase10_evidence.submitted_demand_set_headroom_preserved << '\n';
    return 0;
}
