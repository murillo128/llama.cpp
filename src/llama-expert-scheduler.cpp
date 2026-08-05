#include "llama-expert-scheduler.h"
#include "llama-perfetto-trace.h"

#include <algorithm>
#include <condition_variable>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace {

bool same_key(llm_expert_key lhs, llm_expert_key rhs) {
    return lhs.layer == rhs.layer && lhs.expert == rhs.expert;
}

bool is_terminal(llm_expert_request_state state) {
    return state == llm_expert_request_state::complete ||
        state == llm_expert_request_state::cancelled ||
        state == llm_expert_request_state::failed;
}

bool is_submitted(llm_expert_request_state state) {
    return state != llm_expert_request_state::free &&
        state != llm_expert_request_state::queued &&
        !is_terminal(state);
}

bool valid_transition(llm_expert_request_state current, llm_expert_request_state next) {
    if (next == llm_expert_request_state::cancelling) {
        return current >= llm_expert_request_state::queued && current <= llm_expert_request_state::device_preparing;
    }
    if (next == llm_expert_request_state::draining) {
        return (current >= llm_expert_request_state::queued &&
                current <= llm_expert_request_state::device_preparing) ||
            current == llm_expert_request_state::cancelling;
    }
    switch (current) {
        case llm_expert_request_state::submitting:
            return next == llm_expert_request_state::io_in_flight ||
                next == llm_expert_request_state::host_ready;
        case llm_expert_request_state::io_in_flight:
            return next == llm_expert_request_state::host_ready;
        case llm_expert_request_state::host_ready:
            return next == llm_expert_request_state::h2d_in_flight ||
                next == llm_expert_request_state::device_preparing;
        case llm_expert_request_state::h2d_in_flight:
            return next == llm_expert_request_state::device_ready;
        case llm_expert_request_state::device_preparing:
            return next == llm_expert_request_state::device_ready;
        case llm_expert_request_state::cancelling:
            return next == llm_expert_request_state::draining;
        default:
            return false;
    }
}

bool checked_add(uint64_t lhs, uint64_t rhs, uint64_t & result) {
    if (rhs > UINT64_MAX - lhs) return false;
    result = lhs + rhs;
    return true;
}

} // namespace

struct llm_expert_scheduler::impl {
    struct request_record {
        llm_expert_key key = { -1, -1 };
        uint64_t generation = 0;
        uint64_t enqueue_ordinal = 0;
        llm_expert_priority priority = llm_expert_priority::prefetch_speculative;
        llm_expert_readiness readiness = llm_expert_readiness::host_ready;
        llm_expert_request_state state = llm_expert_request_state::free;
        uint32_t waiters = 0;
        llm_expert_request_metadata metadata;
        bool speculative_charge_active = false;
        bool promoted_from_speculative = false;
        bool cancellation_demand_owned = false;
    };

    llm_expert_scheduler_config config;
    std::vector<request_record> requests;
    mutable std::mutex mutex;
    std::condition_variable state_cv;
    llm_expert_scheduler_diagnostics counters;
    uint64_t next_ordinal = 1;

    bool speculative_budgets_enabled() const {
        return config.max_speculative_flights != 0;
    }

    bool can_charge(const llm_expert_request_metadata & metadata) const {
        if (!speculative_budgets_enabled()) return true;
        if (metadata.origin != llm_expert_request_origin::speculative ||
            metadata.profile_digest == 0 || metadata.owner_request == 0 ||
            metadata.target_layer < 0 || metadata.deadline_token < metadata.owner_token ||
            counters.active_speculative_flights >= config.max_speculative_flights ||
            counters.speculative_storage_bytes_in_flight >
                config.max_speculative_storage_bytes_in_flight ||
            counters.speculative_h2d_bytes_in_flight >
                config.max_speculative_h2d_bytes_in_flight ||
            counters.speculative_cold_slots > config.max_speculative_cold_slots ||
            counters.speculative_hot_slots > config.max_speculative_hot_slots ||
            metadata.reserved_storage_bytes >
                config.max_speculative_storage_bytes_in_flight - counters.speculative_storage_bytes_in_flight ||
            metadata.reserved_h2d_bytes >
                config.max_speculative_h2d_bytes_in_flight - counters.speculative_h2d_bytes_in_flight ||
            metadata.speculative_cold_slots >
                config.max_speculative_cold_slots - counters.speculative_cold_slots ||
            metadata.speculative_hot_slots >
                config.max_speculative_hot_slots - counters.speculative_hot_slots) {
            return false;
        }
        uint64_t token_storage = 0;
        uint64_t token_h2d = 0;
        for (const auto & request : requests) {
            if (!request.speculative_charge_active ||
                request.metadata.owner_request != metadata.owner_request ||
                request.metadata.owner_token != metadata.owner_token) continue;
            if (!checked_add(token_storage, request.metadata.reserved_storage_bytes, token_storage) ||
                !checked_add(token_h2d, request.metadata.reserved_h2d_bytes, token_h2d)) return false;
        }
        if (token_storage > config.max_speculative_storage_bytes_per_token ||
            token_h2d > config.max_speculative_h2d_bytes_per_token) return false;
        return metadata.reserved_storage_bytes <=
                config.max_speculative_storage_bytes_per_token - token_storage &&
            metadata.reserved_h2d_bytes <=
                config.max_speculative_h2d_bytes_per_token - token_h2d;
    }

    void charge(request_record & request) {
        if (!speculative_budgets_enabled()) return;
        request.speculative_charge_active = true;
        counters.active_speculative_flights++;
        counters.speculative_storage_bytes_in_flight += request.metadata.reserved_storage_bytes;
        counters.speculative_h2d_bytes_in_flight += request.metadata.reserved_h2d_bytes;
        counters.speculative_cold_slots += request.metadata.speculative_cold_slots;
        counters.speculative_hot_slots += request.metadata.speculative_hot_slots;
        counters.peak_speculative_flights = std::max(
            counters.peak_speculative_flights, counters.active_speculative_flights);
        counters.peak_speculative_storage_bytes_in_flight = std::max(
            counters.peak_speculative_storage_bytes_in_flight,
            counters.speculative_storage_bytes_in_flight);
        counters.peak_speculative_h2d_bytes_in_flight = std::max(
            counters.peak_speculative_h2d_bytes_in_flight,
            counters.speculative_h2d_bytes_in_flight);
        counters.peak_speculative_cold_slots = std::max(
            counters.peak_speculative_cold_slots, counters.speculative_cold_slots);
        counters.peak_speculative_hot_slots = std::max(
            counters.peak_speculative_hot_slots, counters.speculative_hot_slots);
    }

    void release_charge(request_record & request) {
        if (!request.speculative_charge_active) return;
        counters.active_speculative_flights--;
        counters.speculative_storage_bytes_in_flight -= request.metadata.reserved_storage_bytes;
        counters.speculative_h2d_bytes_in_flight -= request.metadata.reserved_h2d_bytes;
        counters.speculative_cold_slots -= request.metadata.speculative_cold_slots;
        counters.speculative_hot_slots -= request.metadata.speculative_hot_slots;
        request.speculative_charge_active = false;
    }

    request_record * find(llm_expert_request_handle handle) {
        if (!handle.valid() || handle.slot >= requests.size()) {
            return nullptr;
        }
        request_record & request = requests[handle.slot];
        return request.state != llm_expert_request_state::free && request.generation == handle.generation ? &request : nullptr;
    }

    void update_occupancy() {
        uint32_t active = 0;
        uint32_t queued = 0;
        for (const request_record & request : requests) {
            if (request.state != llm_expert_request_state::free) {
                active++;
            }
            if (request.state == llm_expert_request_state::queued) {
                queued++;
            }
        }
        counters.active_requests = active;
        counters.queued_requests = queued;
        counters.peak_active_requests = std::max(counters.peak_active_requests, active);
    }

    bool reset_for_reuse(request_record & request) {
        if (request.generation == std::numeric_limits<uint64_t>::max()) {
            counters.generation_exhaustions++;
            return false;
        }
        release_charge(request);
        request.key = { -1, -1 };
        request.generation++;
        request.enqueue_ordinal = 0;
        request.priority = llm_expert_priority::prefetch_speculative;
        request.readiness = llm_expert_readiness::host_ready;
        request.state = llm_expert_request_state::free;
        request.waiters = 0;
        request.metadata = {};
        request.promoted_from_speculative = false;
        request.cancellation_demand_owned = false;
        return true;
    }
};

llm_expert_scheduler::llm_expert_scheduler(llm_expert_scheduler_config config) : pimpl(std::make_unique<impl>()) {
    const bool any_speculative_budget = config.max_speculative_flights != 0 ||
        config.max_speculative_storage_bytes_in_flight != 0 ||
        config.max_speculative_h2d_bytes_in_flight != 0 ||
        config.max_speculative_storage_bytes_per_token != 0 ||
        config.max_speculative_h2d_bytes_per_token != 0 ||
        config.max_speculative_cold_slots != 0 || config.max_speculative_hot_slots != 0;
    const bool complete_speculative_budget = config.max_speculative_flights != 0 &&
        config.max_speculative_storage_bytes_in_flight != 0 &&
        config.max_speculative_h2d_bytes_in_flight != 0 &&
        config.max_speculative_storage_bytes_per_token != 0 &&
        config.max_speculative_h2d_bytes_per_token != 0 &&
        config.max_speculative_cold_slots != 0 && config.max_speculative_hot_slots != 0;
    if (config.layer_count == 0 || config.experts_per_layer == 0 || config.request_capacity == 0 ||
        config.waiters_per_request == 0 ||
        any_speculative_budget != complete_speculative_budget ||
        (complete_speculative_budget &&
            (config.max_current_layer_demand_flights == 0 ||
             config.max_current_layer_demand_flights > config.request_capacity ||
             config.max_speculative_flights >
                config.request_capacity - config.max_current_layer_demand_flights ||
             config.max_speculative_storage_bytes_per_token >
                config.max_speculative_storage_bytes_in_flight ||
             config.max_speculative_h2d_bytes_per_token >
                config.max_speculative_h2d_bytes_in_flight)) ||
        uint64_t(config.layer_count)*config.experts_per_layer > uint64_t(std::numeric_limits<uint32_t>::max())) {
        throw std::invalid_argument("invalid expert scheduler configuration");
    }
    pimpl->config = config;
    pimpl->requests.resize(config.request_capacity);
    for (auto & request : pimpl->requests) {
        request.generation = config.initial_generation_for_testing;
    }
    pimpl->counters.request_capacity = config.request_capacity;
    pimpl->counters.waiters_per_request = config.waiters_per_request;
    pimpl->counters.max_current_layer_demand_flights = config.max_current_layer_demand_flights;
    pimpl->counters.administration_bytes = sizeof(*pimpl) + pimpl->requests.capacity()*sizeof(impl::request_record);
}

llm_expert_scheduler::~llm_expert_scheduler() = default;

llm_expert_schedule_result llm_expert_scheduler::enqueue(
        llm_expert_key key,
        llm_expert_priority priority,
        llm_expert_readiness readiness,
        llm_expert_request_metadata metadata) noexcept {
    LLM_EXPERT_TRACE_SCOPE("k3.scheduler", "enqueue", "layer", key.layer, "original_expert_id", key.expert,
        "layout_class_id", metadata.layout_class_id, "priority", uint32_t(priority));
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    if (pimpl->counters.admission_closed) {
        return { llm_expert_schedule_disposition::closed, {} };
    }
    if (!key.is_valid(pimpl->config.layer_count, pimpl->config.experts_per_layer) ||
        metadata.layout_class_id >= LLM_EXPERT_LAYOUT_CLASS_MAX) {
        return { llm_expert_schedule_disposition::invalid, {} };
    }
    const bool speculative = priority >= llm_expert_priority::prefetch_next;
    if (speculative && !pimpl->speculative_budgets_enabled() &&
        metadata.origin == llm_expert_request_origin::demand) {
        metadata.origin = llm_expert_request_origin::speculative;
    }
    if ((speculative && metadata.origin != llm_expert_request_origin::speculative) ||
        (!speculative && metadata.origin == llm_expert_request_origin::speculative) ||
        (pimpl->speculative_budgets_enabled() && speculative && metadata.target_layer != key.layer)) {
        return { llm_expert_schedule_disposition::invalid, {} };
    }
    for (uint32_t slot = 0; slot < pimpl->requests.size(); ++slot) {
        impl::request_record & request = pimpl->requests[slot];
        if (request.state != llm_expert_request_state::free && !is_terminal(request.state) && same_key(request.key, key)) {
            if (request.metadata.layout_class_id != metadata.layout_class_id) {
                return { llm_expert_schedule_disposition::invalid, {} };
            }
            // Cancellation ownership is already committed once a speculative
            // request reaches cancelling/draining. Exact demand must retry with
            // a new generation after that drain; it must never be attached to
            // a generation which can still terminalize as speculative cancel.
            if (!speculative && (request.state == llm_expert_request_state::cancelling ||
                    request.state == llm_expert_request_state::draining)) {
                return { llm_expert_schedule_disposition::busy, { slot, request.generation } };
            }
            if (request.waiters == pimpl->config.waiters_per_request) {
                return { llm_expert_schedule_disposition::busy, {} };
            }
            request.waiters++;
            pimpl->counters.joins++;
            if (!speculative && request.metadata.origin == llm_expert_request_origin::speculative) {
                pimpl->release_charge(request);
                request.metadata = metadata;
                request.metadata.origin = llm_expert_request_origin::demand;
                request.promoted_from_speculative = true;
                pimpl->counters.demand_promotions++;
            }
            if (priority < request.priority) {
                request.priority = priority;
                pimpl->counters.promotions++;
            }
            if (readiness > request.readiness) {
                request.readiness = readiness;
                pimpl->counters.promotions++;
            }
            LLM_EXPERT_TRACE_INSTANT("k3.scheduler", "single_flight_join", "flight_id",
                llm_perfetto_trace_pair_id(llm_perfetto_trace_domain::flight, slot, uint32_t(request.generation)),
                "layer", key.layer, "original_expert_id", key.expert, "waiters", request.waiters);
            return { llm_expert_schedule_disposition::joined, { slot, request.generation } };
        }
    }

    uint32_t slot = UINT32_MAX;
    for (uint32_t index = 0; index < pimpl->requests.size(); ++index) {
        if (pimpl->requests[index].state == llm_expert_request_state::free) {
            slot = index;
            break;
        }
    }
    if (slot == UINT32_MAX && priority < llm_expert_priority::prefetch_next) {
        uint64_t oldest = std::numeric_limits<uint64_t>::max();
        for (uint32_t index = 0; index < pimpl->requests.size(); ++index) {
            const impl::request_record & request = pimpl->requests[index];
            if (request.state == llm_expert_request_state::queued &&
                request.priority > priority && request.enqueue_ordinal < oldest) {
                slot = index;
                oldest = request.enqueue_ordinal;
            }
        }
        if (slot != UINT32_MAX) {
            impl::request_record & request = pimpl->requests[slot];
            if (!pimpl->reset_for_reuse(request)) {
                pimpl->update_occupancy();
                return { llm_expert_schedule_disposition::generation_exhausted, {} };
            }
            pimpl->counters.queued_preemptions++;
            pimpl->counters.drops++;
        }
    }
    if (slot == UINT32_MAX) {
        if (priority >= llm_expert_priority::prefetch_next) {
            pimpl->counters.drops++;
            return { llm_expert_schedule_disposition::dropped, {} };
        }
        return { llm_expert_schedule_disposition::busy, {} };
    }

    impl::request_record & request = pimpl->requests[slot];
    if (request.generation == std::numeric_limits<uint64_t>::max()) {
        pimpl->counters.generation_exhaustions++;
        return { llm_expert_schedule_disposition::generation_exhausted, {} };
    }
    request.key = key;
    request.generation++;
    request.enqueue_ordinal = pimpl->next_ordinal++;
    request.priority = priority;
    request.readiness = readiness;
    request.metadata = metadata;
    request.promoted_from_speculative = false;
    request.cancellation_demand_owned = false;
    request.state = llm_expert_request_state::queued;
    request.waiters = 1;
    if (speculative && !pimpl->can_charge(metadata)) {
        request.key = { -1, -1 };
        request.enqueue_ordinal = 0;
        request.metadata = {};
        request.state = llm_expert_request_state::free;
        request.waiters = 0;
        pimpl->counters.speculative_budget_rejections++;
        pimpl->counters.drops++;
        return { llm_expert_schedule_disposition::dropped, {} };
    }
    if (speculative) pimpl->charge(request);
    pimpl->counters.flights_created++;
    pimpl->update_occupancy();
    [[maybe_unused]] const uint64_t flight_id = llm_perfetto_trace_pair_id(
        llm_perfetto_trace_domain::flight, slot, uint32_t(request.generation));
    LLM_EXPERT_TRACE_ASYNC_BEGIN("k3.scheduler", "flight", flight_id, "flight_id", flight_id,
        "layer", key.layer, "original_expert_id", key.expert, "layout_class_id", metadata.layout_class_id,
        "priority", uint32_t(priority), "queue_depth", pimpl->counters.queued_requests);
    LLM_EXPERT_TRACE_COUNTER("k3.resource", "scheduler_active_requests", 1, pimpl->counters.active_requests);
    LLM_EXPERT_TRACE_COUNTER("k3.resource", "scheduler_queue_depth", 2, pimpl->counters.queued_requests);
    pimpl->state_cv.notify_all();
    return { llm_expert_schedule_disposition::admitted, { slot, request.generation } };
}

llm_expert_schedule_result llm_expert_scheduler::take_next(llm_expert_request_snapshot & result) noexcept {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    uint32_t selected = UINT32_MAX;
    for (uint32_t slot = 0; slot < pimpl->requests.size(); ++slot) {
        const impl::request_record & request = pimpl->requests[slot];
        if (request.state != llm_expert_request_state::queued) {
            continue;
        }
        if (selected == UINT32_MAX || request.priority < pimpl->requests[selected].priority ||
            (request.priority == pimpl->requests[selected].priority &&
             request.enqueue_ordinal < pimpl->requests[selected].enqueue_ordinal)) {
            selected = slot;
        }
    }
    if (selected == UINT32_MAX) {
        return { pimpl->counters.admission_closed ? llm_expert_schedule_disposition::closed :
            llm_expert_schedule_disposition::busy, {} };
    }
    impl::request_record & request = pimpl->requests[selected];
    request.state = llm_expert_request_state::submitting;
    result = {
        request.key,
        { selected, request.generation },
        request.priority,
        request.readiness,
        request.state,
        request.waiters,
        request.enqueue_ordinal,
        request.metadata,
        request.promoted_from_speculative,
    };
    pimpl->update_occupancy();
    [[maybe_unused]] const uint64_t flight_id = llm_perfetto_trace_pair_id(
        llm_perfetto_trace_domain::flight, selected, uint32_t(request.generation));
    LLM_EXPERT_TRACE_INSTANT("k3.scheduler", "dispatch", "flight_id", flight_id,
        "layer", request.key.layer, "original_expert_id", request.key.expert,
        "queue_depth", pimpl->counters.queued_requests);
    LLM_EXPERT_TRACE_FLOW_BEGIN("k3.scheduler", "flight_dispatch", flight_id, "flight_id", flight_id);
    return { llm_expert_schedule_disposition::admitted, result.handle };
}

llm_expert_schedule_disposition llm_expert_scheduler::transition(
        llm_expert_request_handle handle,
        llm_expert_request_state expected,
        llm_expert_request_state next) noexcept {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    impl::request_record * request = pimpl->find(handle);
    if (request == nullptr) {
        pimpl->counters.stale_completions++;
        return llm_expert_schedule_disposition::stale_generation;
    }
    if (request->state != expected || next == llm_expert_request_state::cancelling ||
        !valid_transition(expected, next)) {
        return llm_expert_schedule_disposition::invalid;
    }
    request->state = next;
    pimpl->update_occupancy();
    [[maybe_unused]] const uint64_t flight_id = llm_perfetto_trace_pair_id(
        llm_perfetto_trace_domain::flight, handle.slot, uint32_t(handle.generation));
    LLM_EXPERT_TRACE_INSTANT("k3.scheduler", "state_transition", "flight_id", flight_id,
        "from_state", uint32_t(expected), "to_state", uint32_t(next));
    pimpl->state_cv.notify_all();
    return llm_expert_schedule_disposition::admitted;
}

llm_expert_schedule_disposition llm_expert_scheduler::begin_speculative_cancellation(
        llm_expert_request_handle handle,
        llm_expert_request_state expected) noexcept {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    impl::request_record * request = pimpl->find(handle);
    if (request == nullptr) {
        pimpl->counters.stale_completions++;
        return llm_expert_schedule_disposition::stale_generation;
    }
    if (request->state != expected ||
        !valid_transition(expected, llm_expert_request_state::cancelling) ||
        request->metadata.origin != llm_expert_request_origin::speculative ||
        request->promoted_from_speculative) {
        return llm_expert_schedule_disposition::invalid;
    }
    request->cancellation_demand_owned = false;
    request->state = llm_expert_request_state::cancelling;
    pimpl->update_occupancy();
    LLM_EXPERT_TRACE_INSTANT("k3.lifecycle", "speculative_cancel", "flight_id",
        llm_perfetto_trace_pair_id(llm_perfetto_trace_domain::flight, handle.slot, uint32_t(handle.generation)),
        "from_state", uint32_t(expected));
    pimpl->state_cv.notify_all();
    return llm_expert_schedule_disposition::admitted;
}

llm_expert_schedule_disposition llm_expert_scheduler::begin_demand_cancellation(
        llm_expert_request_handle handle,
        llm_expert_request_state expected) noexcept {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    impl::request_record * request = pimpl->find(handle);
    if (request == nullptr) {
        pimpl->counters.stale_completions++;
        return llm_expert_schedule_disposition::stale_generation;
    }
    if (request->state != expected ||
        !valid_transition(expected, llm_expert_request_state::cancelling) ||
        request->metadata.origin != llm_expert_request_origin::demand) {
        return llm_expert_schedule_disposition::invalid;
    }
    request->cancellation_demand_owned = true;
    request->state = llm_expert_request_state::cancelling;
    pimpl->update_occupancy();
    LLM_EXPERT_TRACE_INSTANT("k3.lifecycle", "demand_cancel", "flight_id",
        llm_perfetto_trace_pair_id(llm_perfetto_trace_domain::flight, handle.slot, uint32_t(handle.generation)),
        "from_state", uint32_t(expected));
    pimpl->state_cv.notify_all();
    return llm_expert_schedule_disposition::admitted;
}

llm_expert_schedule_disposition llm_expert_scheduler::finish(
        llm_expert_request_handle handle,
        llm_expert_request_state terminal) noexcept {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    impl::request_record * request = pimpl->find(handle);
    if (request == nullptr) {
        pimpl->counters.stale_completions++;
        return llm_expert_schedule_disposition::stale_generation;
    }
    const bool complete_ready = terminal == llm_expert_request_state::complete &&
        (request->state == llm_expert_request_state::host_ready ||
         request->state == llm_expert_request_state::device_ready);
    const bool drained_terminal = (terminal == llm_expert_request_state::cancelled ||
        terminal == llm_expert_request_state::failed) && request->state == llm_expert_request_state::draining;
    if (!complete_ready && !drained_terminal) {
        return llm_expert_schedule_disposition::invalid;
    }
    if (terminal == llm_expert_request_state::cancelled &&
        request->promoted_from_speculative && !request->cancellation_demand_owned) {
        return llm_expert_schedule_disposition::invalid;
    }
    request->state = terminal;
    if (terminal == llm_expert_request_state::complete) pimpl->counters.terminal_complete++;
    if (terminal == llm_expert_request_state::failed) pimpl->counters.terminal_failed++;
    if (terminal == llm_expert_request_state::cancelled) pimpl->counters.terminal_cancelled++;
    pimpl->update_occupancy();
    [[maybe_unused]] const uint64_t flight_id = llm_perfetto_trace_pair_id(
        llm_perfetto_trace_domain::flight, handle.slot, uint32_t(handle.generation));
    LLM_EXPERT_TRACE_FLOW_END("k3.scheduler", "flight_terminal", flight_id, "terminal_state", uint32_t(terminal));
    LLM_EXPERT_TRACE_ASYNC_END("k3.scheduler", flight_id, "terminal_state", uint32_t(terminal));
    LLM_EXPERT_TRACE_COUNTER("k3.resource", "scheduler_active_requests", 1, pimpl->counters.active_requests);
    pimpl->state_cv.notify_all();
    return llm_expert_schedule_disposition::admitted;
}

llm_expert_schedule_disposition llm_expert_scheduler::release_terminal(llm_expert_request_handle handle) noexcept {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    impl::request_record * request = pimpl->find(handle);
    if (request == nullptr) {
        pimpl->counters.stale_completions++;
        return llm_expert_schedule_disposition::stale_generation;
    }
    if (!is_terminal(request->state)) {
        return llm_expert_schedule_disposition::busy;
    }
    pimpl->release_charge(*request);
    request->key = { -1, -1 };
    request->enqueue_ordinal = 0;
    request->state = llm_expert_request_state::free;
    request->waiters = 0;
    request->metadata = {};
    request->promoted_from_speculative = false;
    request->cancellation_demand_owned = false;
    pimpl->counters.terminal_releases++;
    pimpl->update_occupancy();
    pimpl->state_cv.notify_all();
    return llm_expert_schedule_disposition::admitted;
}

llm_expert_schedule_disposition llm_expert_scheduler::wait_until_released(
        llm_expert_request_handle handle) noexcept {
    if (!handle.valid()) return llm_expert_schedule_disposition::invalid;
    std::unique_lock<std::mutex> lock(pimpl->mutex);
    if (handle.slot >= pimpl->requests.size()) {
        return llm_expert_schedule_disposition::stale_generation;
    }
    pimpl->state_cv.wait(lock, [&] {
        const auto & request = pimpl->requests[handle.slot];
        return request.generation != handle.generation ||
            request.state == llm_expert_request_state::free;
    });
    return llm_expert_schedule_disposition::admitted;
}

llm_expert_schedule_disposition llm_expert_scheduler::snapshot(
        llm_expert_request_handle handle,
        llm_expert_request_snapshot & snapshot) const noexcept {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    const impl::request_record * request = pimpl->find(handle);
    if (request == nullptr) {
        return llm_expert_schedule_disposition::stale_generation;
    }
    snapshot = {
        request->key,
        handle,
        request->priority,
        request->readiness,
        request->state,
        request->waiters,
        request->enqueue_ordinal,
        request->metadata,
        request->promoted_from_speculative,
    };
    return llm_expert_schedule_disposition::admitted;
}

llm_expert_schedule_disposition llm_expert_scheduler::cancel_queued_speculative(
        llm_expert_request_handle handle) noexcept {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    impl::request_record * request = pimpl->find(handle);
    if (request == nullptr) {
        pimpl->counters.stale_completions++;
        return llm_expert_schedule_disposition::stale_generation;
    }
    if (request->state != llm_expert_request_state::queued ||
        request->metadata.origin != llm_expert_request_origin::speculative ||
        request->promoted_from_speculative) {
        return llm_expert_schedule_disposition::invalid;
    }
    pimpl->counters.terminal_cancelled++;
    pimpl->counters.terminal_releases++;
    pimpl->counters.speculative_cancelled_before_submit++;
    pimpl->release_charge(*request);
    request->key = { -1, -1 };
    request->enqueue_ordinal = 0;
    request->priority = llm_expert_priority::prefetch_speculative;
    request->readiness = llm_expert_readiness::host_ready;
    request->state = llm_expert_request_state::free;
    request->waiters = 0;
    request->metadata = {};
    request->promoted_from_speculative = false;
    request->cancellation_demand_owned = false;
    pimpl->update_occupancy();
    pimpl->state_cv.notify_all();
    return llm_expert_schedule_disposition::admitted;
}

bool llm_expert_scheduler::shutdown() noexcept {
    LLM_EXPERT_TRACE_SCOPE("k3.lifecycle", "scheduler_shutdown");
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    pimpl->counters.admission_closed = true;
    for (const impl::request_record & request : pimpl->requests) {
        if (is_submitted(request.state)) {
            return false;
        }
    }
    for (impl::request_record & request : pimpl->requests) {
        pimpl->release_charge(request);
        request.key = { -1, -1 };
        request.enqueue_ordinal = 0;
        request.state = llm_expert_request_state::free;
        request.waiters = 0;
        request.metadata = {};
        request.promoted_from_speculative = false;
        request.cancellation_demand_owned = false;
    }
    pimpl->update_occupancy();
    pimpl->state_cv.notify_all();
    return true;
}

llm_expert_scheduler_diagnostics llm_expert_scheduler::diagnostics() const noexcept {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    return pimpl->counters;
}
