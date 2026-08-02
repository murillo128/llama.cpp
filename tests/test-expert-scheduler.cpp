#include "llama-expert-scheduler.h"

#include "ggml.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

template<class F> void expect_invalid(F fn) {
    bool rejected = false;
    try { fn(); } catch (const std::invalid_argument &) { rejected = true; }
    GGML_ASSERT(rejected);
}

llm_expert_scheduler_config config(uint32_t capacity = 4, uint64_t generation = 0) {
    return { 2, 8, capacity, 4, std::min<uint32_t>(2, capacity), generation };
}

void test_configuration() {
    expect_invalid([] { llm_expert_scheduler scheduler({}); });
    expect_invalid([] { llm_expert_scheduler scheduler({ 1, 1, 0, 1, 1, 0 }); });
    expect_invalid([] { llm_expert_scheduler scheduler({ 1, 1, 1, 0, 1, 0 }); });
    expect_invalid([] { llm_expert_scheduler scheduler({ 1, 1, 1, 1, 0, 0 }); });
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
    llm_expert_scheduler scheduler({ 2, 8, 2, 4, 1, 0 });
    GGML_ASSERT(scheduler.enqueue(
        { 0, 0 }, llm_expert_priority::prefetch_speculative, llm_expert_readiness::host_ready).accepted());
    GGML_ASSERT(scheduler.enqueue(
        { 0, 1 }, llm_expert_priority::prefetch_next, llm_expert_readiness::host_ready).disposition ==
        llm_expert_schedule_disposition::dropped);
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
    GGML_ASSERT(diagnostics.queued_preemptions == 0);
}

void test_transactional_current_layer_batch() {
    llm_expert_scheduler scheduler({ 1, 8, 4, 4, 2, 0 });
    GGML_ASSERT(scheduler.enqueue(
        { 0, 6 }, llm_expert_priority::prefetch_next, llm_expert_readiness::host_ready).accepted());
    GGML_ASSERT(scheduler.enqueue(
        { 0, 7 }, llm_expert_priority::prefetch_speculative, llm_expert_readiness::host_ready).accepted());

    const llm_expert_current_layer_schedule_entry entries[] = {
        { { 0, 1 }, llm_expert_readiness::host_ready },
        { { 0, 3 }, llm_expert_readiness::device_ready },
    };
    llm_expert_current_layer_schedule_result results[2];
    llm_expert_request_handle revoked[2];
    size_t revoked_count = 0;
    GGML_ASSERT(scheduler.enqueue_current_layer_batch(
        entries, 2, results, 2, revoked, 2, revoked_count) ==
        llm_expert_schedule_disposition::admitted);
    GGML_ASSERT(revoked_count == 0);
    GGML_ASSERT(results[0].disposition ==
        llm_expert_current_layer_schedule_disposition::admitted_new);
    GGML_ASSERT(results[1].disposition ==
        llm_expert_current_layer_schedule_disposition::admitted_new);
    const auto diagnostics = scheduler.diagnostics();
    GGML_ASSERT(diagnostics.current_layer_demand_reserve == 2);
    GGML_ASSERT(diagnostics.current_layer_batches == 1);
    GGML_ASSERT(diagnostics.current_layer_batch_entries == 2);
    GGML_ASSERT(diagnostics.active_requests == 4);
}

void test_batch_preflight_has_no_partial_mutation() {
    llm_expert_scheduler scheduler({ 1, 8, 3, 1, 2, 0 });
    const auto optional = scheduler.enqueue(
        { 0, 1 }, llm_expert_priority::prefetch_next, llm_expert_readiness::host_ready);
    GGML_ASSERT(optional.accepted());
    const auto before = scheduler.diagnostics();
    const llm_expert_current_layer_schedule_entry noncanonical[] = {
        { { 0, 3 }, llm_expert_readiness::host_ready },
        { { 0, 2 }, llm_expert_readiness::host_ready },
    };
    llm_expert_current_layer_schedule_result results[2];
    llm_expert_request_handle revoked[2];
    size_t revoked_count = 0;
    GGML_ASSERT(scheduler.enqueue_current_layer_batch(
        noncanonical, 2, results, 2, revoked, 2, revoked_count) ==
        llm_expert_schedule_disposition::invalid);
    const auto after = scheduler.diagnostics();
    GGML_ASSERT(after.active_requests == before.active_requests);
    GGML_ASSERT(after.flights_created == before.flights_created);
    GGML_ASSERT(after.joins == before.joins);
    GGML_ASSERT(after.promotions == before.promotions);
    GGML_ASSERT(after.current_layer_batch_rejections == before.current_layer_batch_rejections + 1);

    const llm_expert_current_layer_schedule_entry join[] = {
        { { 0, 1 }, llm_expert_readiness::device_ready },
    };
    GGML_ASSERT(scheduler.enqueue_current_layer_batch(
        join, 1, results, 1, revoked, 1, revoked_count) ==
        llm_expert_schedule_disposition::busy);
    llm_expert_request_snapshot snapshot;
    GGML_ASSERT(scheduler.take_next(snapshot).accepted());
    GGML_ASSERT(snapshot.handle.slot == optional.handle.slot);
    GGML_ASSERT(snapshot.priority == llm_expert_priority::prefetch_next);
    GGML_ASSERT(!snapshot.demand_owned);
}

void test_optional_reclaim_and_demand_promotion() {
    llm_expert_scheduler scheduler({ 1, 8, 3, 4, 2, 0 });
    const auto optional = scheduler.enqueue(
        { 0, 0 }, llm_expert_priority::prefetch_speculative, llm_expert_readiness::host_ready);
    const auto existing_demand = scheduler.enqueue(
        { 0, 7 }, llm_expert_priority::demand_current_layer, llm_expert_readiness::host_ready);
    GGML_ASSERT(optional.accepted() && existing_demand.accepted());
    const llm_expert_current_layer_schedule_entry entries[] = {
        { { 0, 1 }, llm_expert_readiness::host_ready },
        { { 0, 2 }, llm_expert_readiness::device_ready },
    };
    llm_expert_current_layer_schedule_result results[2];
    llm_expert_request_handle revoked[2];
    size_t revoked_count = 0;
    GGML_ASSERT(scheduler.enqueue_current_layer_batch(
        entries, 2, results, 2, revoked, 2, revoked_count) ==
        llm_expert_schedule_disposition::admitted);
    GGML_ASSERT(revoked_count == 1);
    GGML_ASSERT(revoked[0].slot == optional.handle.slot &&
        revoked[0].generation == optional.handle.generation);
    GGML_ASSERT(scheduler.diagnostics().queued_preemptions == 1);

    llm_expert_scheduler promoted({ 1, 8, 3, 4, 2, 0 });
    const auto background = promoted.enqueue(
        { 0, 3 }, llm_expert_priority::prefetch_next, llm_expert_readiness::host_ready);
    GGML_ASSERT(background.accepted());
    const llm_expert_current_layer_schedule_entry demand[] = {
        { { 0, 3 }, llm_expert_readiness::device_ready },
    };
    revoked_count = 0;
    GGML_ASSERT(promoted.enqueue_current_layer_batch(
        demand, 1, results, 1, revoked, 1, revoked_count) ==
        llm_expert_schedule_disposition::admitted);
    GGML_ASSERT(results[0].disposition ==
        llm_expert_current_layer_schedule_disposition::promoted_exact_generation);
    GGML_ASSERT(results[0].handle.slot == background.handle.slot &&
        results[0].handle.generation == background.handle.generation);
    GGML_ASSERT(promoted.cancel_optional(
        background.handle, llm_expert_request_state::queued) ==
        llm_expert_schedule_disposition::busy);
    llm_expert_request_snapshot snapshot;
    GGML_ASSERT(promoted.take_next(snapshot).accepted());
    GGML_ASSERT(snapshot.demand_owned);
    GGML_ASSERT(snapshot.priority == llm_expert_priority::demand_current_layer);
    GGML_ASSERT(snapshot.readiness == llm_expert_readiness::device_ready);
    GGML_ASSERT(promoted.diagnostics().optional_cancellations_prevented == 1);
}

void test_pending_successor_is_generation_safe() {
    llm_expert_scheduler scheduler({ 1, 8, 3, 4, 2, 0 });
    const auto predecessor = scheduler.enqueue(
        { 0, 1 }, llm_expert_priority::demand_current_layer, llm_expert_readiness::device_ready);
    GGML_ASSERT(predecessor.accepted());
    llm_expert_request_snapshot snapshot;
    GGML_ASSERT(scheduler.take_next(snapshot).accepted());
    GGML_ASSERT(scheduler.transition(predecessor.handle, llm_expert_request_state::submitting,
        llm_expert_request_state::io_in_flight) == llm_expert_schedule_disposition::admitted);
    GGML_ASSERT(scheduler.transition(predecessor.handle, llm_expert_request_state::io_in_flight,
        llm_expert_request_state::cancelling) == llm_expert_schedule_disposition::admitted);
    GGML_ASSERT(scheduler.transition(predecessor.handle, llm_expert_request_state::cancelling,
        llm_expert_request_state::draining) == llm_expert_schedule_disposition::admitted);

    const llm_expert_current_layer_schedule_entry entries[] = {
        { { 0, 1 }, llm_expert_readiness::device_ready },
        { { 0, 2 }, llm_expert_readiness::host_ready },
    };
    llm_expert_current_layer_schedule_result results[2];
    llm_expert_request_handle revoked[2];
    size_t revoked_count = 0;
    GGML_ASSERT(scheduler.enqueue_current_layer_batch(
        entries, 2, results, 2, revoked, 2, revoked_count) ==
        llm_expert_schedule_disposition::admitted);
    GGML_ASSERT(results[0].disposition ==
        llm_expert_current_layer_schedule_disposition::pending_successor);
    GGML_ASSERT(results[0].handle.generation == predecessor.handle.generation + 1);
    GGML_ASSERT(results[1].disposition ==
        llm_expert_current_layer_schedule_disposition::admitted_new);

    GGML_ASSERT(scheduler.take_next(snapshot).accepted());
    GGML_ASSERT(snapshot.key.expert == 2);
    GGML_ASSERT(scheduler.finish(predecessor.handle, llm_expert_request_state::cancelled) ==
        llm_expert_schedule_disposition::admitted);
    GGML_ASSERT(scheduler.release_terminal(predecessor.handle) ==
        llm_expert_schedule_disposition::pending_successor);
    GGML_ASSERT(scheduler.transition(predecessor.handle, llm_expert_request_state::draining,
        llm_expert_request_state::cancelled) == llm_expert_schedule_disposition::stale_generation);
    GGML_ASSERT(scheduler.take_next(snapshot).accepted());
    GGML_ASSERT(snapshot.key.expert == 1);
    GGML_ASSERT(snapshot.handle.generation == results[0].handle.generation);
    const auto diagnostics = scheduler.diagnostics();
    GGML_ASSERT(diagnostics.current_layer_pending_successors == 1);
    GGML_ASSERT(diagnostics.current_layer_successor_activations == 1);

    llm_expert_scheduler cancelled({ 1, 8, 3, 4, 2, 0 });
    const auto old = cancelled.enqueue(
        { 0, 4 }, llm_expert_priority::demand_current_layer,
        llm_expert_readiness::device_ready);
    GGML_ASSERT(old.accepted());
    GGML_ASSERT(cancelled.take(old.handle, snapshot).accepted());
    GGML_ASSERT(cancelled.transition(old.handle, llm_expert_request_state::submitting,
        llm_expert_request_state::draining) == llm_expert_schedule_disposition::admitted);
    const llm_expert_current_layer_schedule_entry successor[] = {
        { { 0, 4 }, llm_expert_readiness::device_ready },
    };
    revoked_count = 0;
    GGML_ASSERT(cancelled.enqueue_current_layer_batch(
        successor, 1, results, 1, revoked, 1, revoked_count) ==
        llm_expert_schedule_disposition::admitted);
    GGML_ASSERT(results[0].disposition ==
        llm_expert_current_layer_schedule_disposition::pending_successor);
    GGML_ASSERT(cancelled.cancel_pending_successor(results[0].handle) ==
        llm_expert_schedule_disposition::admitted);
    GGML_ASSERT(cancelled.finish(old.handle, llm_expert_request_state::failed) ==
        llm_expert_schedule_disposition::admitted);
    GGML_ASSERT(cancelled.release_terminal(old.handle) ==
        llm_expert_schedule_disposition::admitted);
    GGML_ASSERT(cancelled.diagnostics().current_layer_successor_cancellations == 1);
}

void test_same_key_demand_promotes_every_joinable_state() {
    const std::array<llm_expert_request_state, 6> states = {
        llm_expert_request_state::queued,
        llm_expert_request_state::submitting,
        llm_expert_request_state::io_in_flight,
        llm_expert_request_state::host_ready,
        llm_expert_request_state::h2d_in_flight,
        llm_expert_request_state::device_ready,
    };
    for (const auto target : states) {
        llm_expert_scheduler scheduler({ 1, 8, 3, 4, 2, 0 });
        const auto optional = scheduler.enqueue(
            { 0, 3 }, llm_expert_priority::demand_future_dependency,
            llm_expert_readiness::host_ready);
        GGML_ASSERT(optional.accepted());
        llm_expert_request_snapshot snapshot;
        if (target != llm_expert_request_state::queued) {
            GGML_ASSERT(scheduler.take(optional.handle, snapshot).accepted());
        }
        if (target >= llm_expert_request_state::io_in_flight) {
            GGML_ASSERT(scheduler.transition(optional.handle,
                llm_expert_request_state::submitting,
                llm_expert_request_state::io_in_flight) ==
                llm_expert_schedule_disposition::admitted);
        }
        if (target >= llm_expert_request_state::host_ready) {
            GGML_ASSERT(scheduler.transition(optional.handle,
                llm_expert_request_state::io_in_flight,
                llm_expert_request_state::host_ready) ==
                llm_expert_schedule_disposition::admitted);
        }
        if (target >= llm_expert_request_state::h2d_in_flight) {
            GGML_ASSERT(scheduler.transition(optional.handle,
                llm_expert_request_state::host_ready,
                llm_expert_request_state::h2d_in_flight) ==
                llm_expert_schedule_disposition::admitted);
        }
        if (target >= llm_expert_request_state::device_ready) {
            GGML_ASSERT(scheduler.transition(optional.handle,
                llm_expert_request_state::h2d_in_flight,
                llm_expert_request_state::device_ready) ==
                llm_expert_schedule_disposition::admitted);
        }
        const llm_expert_current_layer_schedule_entry demand[] = {
            { { 0, 3 }, llm_expert_readiness::device_ready },
        };
        llm_expert_current_layer_schedule_result result;
        llm_expert_request_handle revoked;
        size_t revoked_count = 0;
        GGML_ASSERT(scheduler.enqueue_current_layer_batch(
            demand, 1, &result, 1, &revoked, 1, revoked_count) ==
            llm_expert_schedule_disposition::admitted);
        GGML_ASSERT(result.disposition ==
            llm_expert_current_layer_schedule_disposition::promoted_exact_generation);
        GGML_ASSERT(result.handle.slot == optional.handle.slot &&
            result.handle.generation == optional.handle.generation);
        GGML_ASSERT(scheduler.cancel_optional(optional.handle, target) ==
            llm_expert_schedule_disposition::busy);
        GGML_ASSERT(scheduler.diagnostics().flights_created == 1);
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
    GGML_ASSERT(scheduler.transition(admitted.handle, llm_expert_request_state::h2d_in_flight,
        llm_expert_request_state::cancelling) == llm_expert_schedule_disposition::admitted);
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
    llm_expert_scheduler scheduler({ 1, 8, 2, 8, 1, 0 });
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

} // namespace

int main() {
    test_configuration();
    test_priority_fifo_and_promotion();
    test_saturation_and_preemption();
    test_transactional_current_layer_batch();
    test_batch_preflight_has_no_partial_mutation();
    test_optional_reclaim_and_demand_promotion();
    test_pending_successor_is_generation_safe();
    test_same_key_demand_promotes_every_joinable_state();
    test_stale_completion_and_generation_exhaustion();
    test_quiescent_shutdown();
    test_post_h2d_cancellation_path();
    test_cold_hit_reaches_host_ready_without_io();
    test_concurrent_duplicate_joins();
    return 0;
}
