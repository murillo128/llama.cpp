#include "llama-expert-scheduler.h"

#include <algorithm>
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
    if (next == llm_expert_request_state::cancelling || next == llm_expert_request_state::draining) {
        return current >= llm_expert_request_state::queued && current <= llm_expert_request_state::h2d_in_flight;
    }
    switch (current) {
        case llm_expert_request_state::submitting:
            return next == llm_expert_request_state::io_in_flight;
        case llm_expert_request_state::io_in_flight:
            return next == llm_expert_request_state::host_ready;
        case llm_expert_request_state::host_ready:
            return next == llm_expert_request_state::h2d_in_flight;
        case llm_expert_request_state::h2d_in_flight:
            return next == llm_expert_request_state::device_ready;
        case llm_expert_request_state::cancelling:
            return next == llm_expert_request_state::draining;
        default:
            return false;
    }
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
    };

    llm_expert_scheduler_config config;
    std::vector<request_record> requests;
    mutable std::mutex mutex;
    llm_expert_scheduler_diagnostics counters;
    uint64_t next_ordinal = 1;

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
        request.key = { -1, -1 };
        request.generation++;
        request.enqueue_ordinal = 0;
        request.priority = llm_expert_priority::prefetch_speculative;
        request.readiness = llm_expert_readiness::host_ready;
        request.state = llm_expert_request_state::free;
        request.waiters = 0;
        return true;
    }
};

llm_expert_scheduler::llm_expert_scheduler(llm_expert_scheduler_config config) : pimpl(std::make_unique<impl>()) {
    if (config.layer_count == 0 || config.experts_per_layer == 0 || config.request_capacity == 0 ||
        config.waiters_per_request == 0 ||
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
    pimpl->counters.administration_bytes = sizeof(*pimpl) + pimpl->requests.capacity()*sizeof(impl::request_record);
}

llm_expert_scheduler::~llm_expert_scheduler() = default;

llm_expert_schedule_result llm_expert_scheduler::enqueue(
        llm_expert_key key,
        llm_expert_priority priority,
        llm_expert_readiness readiness) noexcept {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    if (pimpl->counters.admission_closed) {
        return { llm_expert_schedule_disposition::closed, {} };
    }
    if (!key.is_valid(pimpl->config.layer_count, pimpl->config.experts_per_layer)) {
        return { llm_expert_schedule_disposition::invalid, {} };
    }
    for (uint32_t slot = 0; slot < pimpl->requests.size(); ++slot) {
        impl::request_record & request = pimpl->requests[slot];
        if (request.state != llm_expert_request_state::free && !is_terminal(request.state) && same_key(request.key, key)) {
            if (request.waiters == pimpl->config.waiters_per_request) {
                return { llm_expert_schedule_disposition::busy, {} };
            }
            request.waiters++;
            pimpl->counters.joins++;
            if (priority < request.priority) {
                request.priority = priority;
                pimpl->counters.promotions++;
            }
            if (readiness > request.readiness) {
                request.readiness = readiness;
                pimpl->counters.promotions++;
            }
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
    if (slot == UINT32_MAX && priority == llm_expert_priority::demand_current_layer) {
        uint64_t oldest = std::numeric_limits<uint64_t>::max();
        for (uint32_t index = 0; index < pimpl->requests.size(); ++index) {
            const impl::request_record & request = pimpl->requests[index];
            if (request.state == llm_expert_request_state::queued &&
                request.priority >= llm_expert_priority::prefetch_next && request.enqueue_ordinal < oldest) {
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
    request.state = llm_expert_request_state::queued;
    request.waiters = 1;
    pimpl->counters.flights_created++;
    pimpl->update_occupancy();
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
    };
    pimpl->update_occupancy();
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
    if (request->state != expected || !valid_transition(expected, next)) {
        return llm_expert_schedule_disposition::invalid;
    }
    request->state = next;
    pimpl->update_occupancy();
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
    request->state = terminal;
    pimpl->update_occupancy();
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
    request->key = { -1, -1 };
    request->enqueue_ordinal = 0;
    request->state = llm_expert_request_state::free;
    request->waiters = 0;
    pimpl->update_occupancy();
    return llm_expert_schedule_disposition::admitted;
}

bool llm_expert_scheduler::shutdown() noexcept {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    pimpl->counters.admission_closed = true;
    for (const impl::request_record & request : pimpl->requests) {
        if (is_submitted(request.state)) {
            return false;
        }
    }
    for (impl::request_record & request : pimpl->requests) {
        request.key = { -1, -1 };
        request.enqueue_ordinal = 0;
        request.state = llm_expert_request_state::free;
        request.waiters = 0;
    }
    pimpl->update_occupancy();
    return true;
}

llm_expert_scheduler_diagnostics llm_expert_scheduler::diagnostics() const noexcept {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    return pimpl->counters;
}
