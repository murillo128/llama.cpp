#include "llama-expert-scheduler.h"

#include "ggml.h"

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
    bool submitted_headroom_preserved = false;
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
        2, 300, 200, 200, 150, 2, 2,
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
    llm_expert_scheduler scheduler(config());
    const auto diagnostics = scheduler.diagnostics();
    GGML_ASSERT(diagnostics.request_capacity == 4);
    GGML_ASSERT(diagnostics.waiters_per_request == 4);
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
        1, 200, 200, 200, 200, 1, 1,
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

void test_cancelling_promotion_fence_and_submitted_demand_headroom() {
    const llm_expert_scheduler_config bounded = {
        1, 8, 2, 4, 0,
        1, 200, 200, 200, 200, 1, 1,
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
        llm_expert_scheduler scheduler(bounded);
        const auto submitted = scheduler.enqueue(
            { 0, 2 }, llm_expert_priority::prefetch_next,
            llm_expert_readiness::device_ready,
            speculative_metadata(22, 0, 0, 100, 100));
        GGML_ASSERT(submitted.accepted());
        llm_expert_request_snapshot selected;
        GGML_ASSERT(scheduler.take_next(selected).accepted());
        GGML_ASSERT(scheduler.transition(submitted.handle,
            llm_expert_request_state::submitting,
            llm_expert_request_state::io_in_flight) ==
            llm_expert_schedule_disposition::admitted);
        GGML_ASSERT(scheduler.enqueue(
            { 0, 3 }, llm_expert_priority::prefetch_speculative,
            llm_expert_readiness::host_ready,
            speculative_metadata(23, 0, 0, 100, 0)).disposition ==
            llm_expert_schedule_disposition::dropped);
        const auto demand = scheduler.enqueue(
            { 0, 4 }, llm_expert_priority::demand_current_layer,
            llm_expert_readiness::device_ready);
        GGML_ASSERT(demand.disposition == llm_expert_schedule_disposition::admitted);
        GGML_ASSERT(demand.handle.slot != submitted.handle.slot);
        phase10_evidence.submitted_headroom_preserved = true;
        GGML_ASSERT(!scheduler.shutdown());
        GGML_ASSERT(scheduler.transition(submitted.handle,
            llm_expert_request_state::io_in_flight,
            llm_expert_request_state::draining) ==
            llm_expert_schedule_disposition::admitted);
        GGML_ASSERT(scheduler.finish(submitted.handle,
            llm_expert_request_state::failed) ==
            llm_expert_schedule_disposition::admitted);
        GGML_ASSERT(scheduler.release_terminal(submitted.handle) ==
            llm_expert_schedule_disposition::admitted);
        GGML_ASSERT(scheduler.shutdown());
    }
}

} // namespace

int main() {
    test_configuration();
    test_priority_fifo_and_promotion();
    test_saturation_and_preemption();
    test_stale_completion_and_generation_exhaustion();
    test_quiescent_shutdown();
    test_post_h2d_cancellation_path();
    test_cold_hit_reaches_host_ready_without_io();
    test_concurrent_duplicate_joins();
    test_speculative_budgets_cancel_and_demand_promotion();
    test_demand_preemption_submitted_drain_retry_and_shutdown();
    test_cancelling_promotion_fence_and_submitted_demand_headroom();
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
              << "\tsubmitted_headroom_preserved=" << phase10_evidence.submitted_headroom_preserved << '\n';
    return 0;
}
