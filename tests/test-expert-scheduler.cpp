#include "llama-expert-scheduler.h"

#include "ggml.h"

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
    return { 2, 8, capacity, 4, generation };
}

void test_configuration() {
    expect_invalid([] { llm_expert_scheduler scheduler({}); });
    expect_invalid([] { llm_expert_scheduler scheduler({ 1, 1, 0, 1, 0 }); });
    expect_invalid([] { llm_expert_scheduler scheduler({ 1, 1, 1, 0, 0 }); });
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

} // namespace

int main() {
    test_configuration();
    test_priority_fifo_and_promotion();
    test_saturation_and_preemption();
    test_stale_completion_and_generation_exhaustion();
    test_quiescent_shutdown();
    test_concurrent_duplicate_joins();
    return 0;
}
