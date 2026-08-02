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

bool is_joinable(llm_expert_request_state state) {
    return state >= llm_expert_request_state::queued &&
        state <= llm_expert_request_state::device_ready;
}

bool is_optional(llm_expert_priority priority) {
    return priority > llm_expert_priority::demand_current_layer;
}

bool valid_transition(llm_expert_request_state current, llm_expert_request_state next) {
    if (next == llm_expert_request_state::cancelling) {
        return current >= llm_expert_request_state::queued && current <= llm_expert_request_state::h2d_in_flight;
    }
    if (next == llm_expert_request_state::draining) {
        return (current >= llm_expert_request_state::queued &&
                current <= llm_expert_request_state::h2d_in_flight) ||
            current == llm_expert_request_state::cancelling;
    }
    switch (current) {
        case llm_expert_request_state::submitting:
            return next == llm_expert_request_state::io_in_flight ||
                next == llm_expert_request_state::host_ready;
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
    struct pending_successor_record {
        bool active = false;
        uint64_t generation = 0;
        uint64_t enqueue_ordinal = 0;
        llm_expert_priority priority = llm_expert_priority::demand_current_layer;
        llm_expert_readiness readiness = llm_expert_readiness::host_ready;
        uint32_t waiters = 0;
    };

    struct request_record {
        llm_expert_key key = { -1, -1 };
        uint64_t generation = 0;
        uint64_t enqueue_ordinal = 0;
        llm_expert_priority priority = llm_expert_priority::prefetch_speculative;
        llm_expert_readiness readiness = llm_expert_readiness::host_ready;
        llm_expert_request_state state = llm_expert_request_state::free;
        uint32_t waiters = 0;
        bool demand_owned = false;
        pending_successor_record pending;
    };

    enum class batch_action : uint8_t {
        admit,
        join,
        promote,
        create_successor,
        join_successor,
    };

    struct batch_plan {
        batch_action action = batch_action::admit;
        uint32_t slot = UINT32_MAX;
        uint64_t generation = 0;
        uint64_t enqueue_ordinal = 0;
        bool revokes_optional = false;
    };

    llm_expert_scheduler_config config;
    std::vector<request_record> requests;
    std::vector<batch_plan> batch_plans;
    std::vector<uint8_t> batch_claimed_slots;
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
        request.demand_owned = false;
        request.pending = {};
        return true;
    }
};

llm_expert_scheduler::llm_expert_scheduler(llm_expert_scheduler_config config) : pimpl(std::make_unique<impl>()) {
    if (config.layer_count == 0 || config.experts_per_layer == 0 || config.request_capacity == 0 ||
        config.waiters_per_request == 0 || config.current_layer_demand_reserve == 0 ||
        config.current_layer_demand_reserve > config.request_capacity ||
        config.current_layer_demand_reserve > config.experts_per_layer ||
        uint64_t(config.layer_count)*config.experts_per_layer > uint64_t(std::numeric_limits<uint32_t>::max())) {
        throw std::invalid_argument("invalid expert scheduler configuration");
    }
    pimpl->config = config;
    pimpl->requests.resize(config.request_capacity);
    pimpl->batch_plans.resize(config.current_layer_demand_reserve);
    pimpl->batch_claimed_slots.resize(config.request_capacity);
    for (auto & request : pimpl->requests) {
        request.generation = config.initial_generation_for_testing;
    }
    pimpl->counters.request_capacity = config.request_capacity;
    pimpl->counters.waiters_per_request = config.waiters_per_request;
    pimpl->counters.current_layer_demand_reserve = config.current_layer_demand_reserve;
    pimpl->counters.administration_bytes = sizeof(*pimpl) +
        pimpl->requests.capacity()*sizeof(impl::request_record) +
        pimpl->batch_plans.capacity()*sizeof(impl::batch_plan) +
        pimpl->batch_claimed_slots.capacity()*sizeof(uint8_t);
}

llm_expert_scheduler::~llm_expert_scheduler() = default;

llm_expert_schedule_result llm_expert_scheduler::enqueue(
        llm_expert_key key,
        llm_expert_priority priority,
        llm_expert_readiness readiness) noexcept {
    if (priority == llm_expert_priority::demand_current_layer) {
        const llm_expert_current_layer_schedule_entry entry = { key, readiness };
        llm_expert_current_layer_schedule_result result;
        size_t revoked_count = 0;
        const auto disposition = enqueue_current_layer_batch(
            &entry, 1, &result, 1, nullptr, 0, revoked_count);
        if (disposition != llm_expert_schedule_disposition::admitted) {
            return { disposition, {} };
        }
        switch (result.disposition) {
            case llm_expert_current_layer_schedule_disposition::admitted_new:
                return { llm_expert_schedule_disposition::admitted, result.handle };
            case llm_expert_current_layer_schedule_disposition::joined_exact_generation:
            case llm_expert_current_layer_schedule_disposition::promoted_exact_generation:
                return { llm_expert_schedule_disposition::joined, result.handle };
            case llm_expert_current_layer_schedule_disposition::pending_successor:
                return { llm_expert_schedule_disposition::pending_successor, result.handle };
        }
    }

    std::lock_guard<std::mutex> lock(pimpl->mutex);
    if (pimpl->counters.admission_closed) {
        return { llm_expert_schedule_disposition::closed, {} };
    }
    if (!key.is_valid(pimpl->config.layer_count, pimpl->config.experts_per_layer)) {
        return { llm_expert_schedule_disposition::invalid, {} };
    }
    for (uint32_t slot = 0; slot < pimpl->requests.size(); ++slot) {
        impl::request_record & request = pimpl->requests[slot];
        if (request.state != llm_expert_request_state::free && is_joinable(request.state) &&
            !request.pending.active && same_key(request.key, key)) {
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
    const uint32_t optional_limit = pimpl->config.request_capacity -
        pimpl->config.current_layer_demand_reserve;
    for (uint32_t index = 0; index < optional_limit; ++index) {
        if (pimpl->requests[index].state == llm_expert_request_state::free) {
            slot = index;
            break;
        }
    }
    if (slot == UINT32_MAX) {
        if (is_optional(priority)) {
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
    request.demand_owned = false;
    request.pending = {};
    pimpl->counters.flights_created++;
    pimpl->update_occupancy();
    return { llm_expert_schedule_disposition::admitted, { slot, request.generation } };
}

llm_expert_schedule_disposition llm_expert_scheduler::enqueue_current_layer_batch(
        const llm_expert_current_layer_schedule_entry * entries,
        size_t count,
        llm_expert_current_layer_schedule_result * results,
        size_t result_capacity,
        llm_expert_request_handle * revoked_optional,
        size_t revoked_capacity,
        size_t & revoked_count) noexcept {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    revoked_count = 0;
    const auto reject = [&](llm_expert_schedule_disposition disposition) {
        pimpl->counters.current_layer_batch_rejections++;
        return disposition;
    };
    if (pimpl->counters.admission_closed) {
        return reject(llm_expert_schedule_disposition::closed);
    }
    if (count > pimpl->config.current_layer_demand_reserve || count > result_capacity ||
        (count != 0 && (entries == nullptr || results == nullptr))) {
        return reject(llm_expert_schedule_disposition::invalid);
    }
    if (count == 0) {
        pimpl->counters.current_layer_batches++;
        return llm_expert_schedule_disposition::admitted;
    }

    std::fill(pimpl->batch_claimed_slots.begin(), pimpl->batch_claimed_slots.end(), uint8_t(0));
    size_t ordinal_count = 0;
    size_t victim_count = 0;
    for (size_t index = 0; index < count; ++index) {
        const auto & entry = entries[index];
        if (!entry.key.is_valid(pimpl->config.layer_count, pimpl->config.experts_per_layer) ||
            (index != 0 && (entries[index - 1].key.layer > entry.key.layer ||
                (entries[index - 1].key.layer == entry.key.layer &&
                 entries[index - 1].key.expert >= entry.key.expert)))) {
            return reject(llm_expert_schedule_disposition::invalid);
        }

        impl::batch_plan & plan = pimpl->batch_plans[index];
        plan = {};
        uint32_t matching_slot = UINT32_MAX;
        for (uint32_t slot = 0; slot < pimpl->requests.size(); ++slot) {
            const auto & request = pimpl->requests[slot];
            if (request.state != llm_expert_request_state::free && same_key(request.key, entry.key)) {
                if (matching_slot != UINT32_MAX) {
                    return reject(llm_expert_schedule_disposition::invalid);
                }
                matching_slot = slot;
            }
        }
        if (matching_slot != UINT32_MAX) {
            const auto & request = pimpl->requests[matching_slot];
            plan.slot = matching_slot;
            if (request.pending.active) {
                if (request.pending.waiters == pimpl->config.waiters_per_request) {
                    return reject(llm_expert_schedule_disposition::busy);
                }
                plan.action = impl::batch_action::join_successor;
                plan.generation = request.pending.generation;
            } else if (is_joinable(request.state)) {
                if (request.waiters == pimpl->config.waiters_per_request) {
                    return reject(llm_expert_schedule_disposition::busy);
                }
                const bool promoted = !request.demand_owned ||
                    request.priority != llm_expert_priority::demand_current_layer ||
                    entry.readiness > request.readiness;
                plan.action = promoted ? impl::batch_action::promote : impl::batch_action::join;
                plan.generation = request.generation;
            } else if (request.state == llm_expert_request_state::cancelling ||
                       request.state == llm_expert_request_state::draining || is_terminal(request.state)) {
                if (request.generation == std::numeric_limits<uint64_t>::max()) {
                    return reject(llm_expert_schedule_disposition::generation_exhausted);
                }
                plan.action = impl::batch_action::create_successor;
                plan.generation = request.generation + 1;
                ordinal_count++;
            } else {
                return reject(llm_expert_schedule_disposition::invalid);
            }
            pimpl->batch_claimed_slots[matching_slot] = 1;
            continue;
        }

        uint32_t selected = UINT32_MAX;
        for (uint32_t slot = 0; slot < pimpl->requests.size(); ++slot) {
            if (!pimpl->batch_claimed_slots[slot] &&
                pimpl->requests[slot].state == llm_expert_request_state::free) {
                selected = slot;
                break;
            }
        }
        if (selected == UINT32_MAX) {
            for (uint32_t slot = 0; slot < pimpl->requests.size(); ++slot) {
                const auto & candidate = pimpl->requests[slot];
                if (pimpl->batch_claimed_slots[slot] || candidate.state != llm_expert_request_state::queued ||
                    candidate.demand_owned || !is_optional(candidate.priority)) {
                    continue;
                }
                if (selected == UINT32_MAX || candidate.priority > pimpl->requests[selected].priority ||
                    (candidate.priority == pimpl->requests[selected].priority &&
                     (candidate.enqueue_ordinal < pimpl->requests[selected].enqueue_ordinal ||
                      (candidate.enqueue_ordinal == pimpl->requests[selected].enqueue_ordinal && slot < selected)))) {
                    selected = slot;
                }
            }
        }
        if (selected == UINT32_MAX) {
            return reject(llm_expert_schedule_disposition::busy);
        }
        const auto & selected_request = pimpl->requests[selected];
        if (selected_request.generation == std::numeric_limits<uint64_t>::max()) {
            return reject(llm_expert_schedule_disposition::generation_exhausted);
        }
        plan.action = impl::batch_action::admit;
        plan.slot = selected;
        plan.generation = selected_request.generation + 1;
        plan.revokes_optional = selected_request.state == llm_expert_request_state::queued;
        victim_count += plan.revokes_optional;
        ordinal_count++;
        pimpl->batch_claimed_slots[selected] = 1;
    }

    if (victim_count > revoked_capacity || (victim_count != 0 && revoked_optional == nullptr)) {
        return reject(llm_expert_schedule_disposition::busy);
    }
    if (ordinal_count > std::numeric_limits<uint64_t>::max() - pimpl->next_ordinal) {
        return reject(llm_expert_schedule_disposition::generation_exhausted);
    }
    uint64_t next_ordinal = pimpl->next_ordinal;
    for (size_t index = 0; index < count; ++index) {
        auto & plan = pimpl->batch_plans[index];
        if (plan.action == impl::batch_action::admit ||
            plan.action == impl::batch_action::create_successor) {
            plan.enqueue_ordinal = next_ordinal++;
        }
    }

    for (size_t index = 0; index < count; ++index) {
        const auto & entry = entries[index];
        const auto & plan = pimpl->batch_plans[index];
        auto & request = pimpl->requests[plan.slot];
        auto & result = results[index];
        switch (plan.action) {
            case impl::batch_action::admit:
                if (plan.revokes_optional) {
                    revoked_optional[revoked_count++] = { plan.slot, request.generation };
                    pimpl->counters.queued_preemptions++;
                    pimpl->counters.drops++;
                }
                request.key = entry.key;
                request.generation = plan.generation;
                request.enqueue_ordinal = plan.enqueue_ordinal;
                request.priority = llm_expert_priority::demand_current_layer;
                request.readiness = entry.readiness;
                request.state = llm_expert_request_state::queued;
                request.waiters = 1;
                request.demand_owned = true;
                request.pending = {};
                result = { llm_expert_current_layer_schedule_disposition::admitted_new,
                    { plan.slot, plan.generation } };
                pimpl->counters.flights_created++;
                break;
            case impl::batch_action::join:
                request.waiters++;
                result = { llm_expert_current_layer_schedule_disposition::joined_exact_generation,
                    { plan.slot, plan.generation } };
                pimpl->counters.joins++;
                break;
            case impl::batch_action::promote:
                request.waiters++;
                pimpl->counters.promotions +=
                    (request.priority != llm_expert_priority::demand_current_layer) +
                    (entry.readiness > request.readiness);
                request.priority = llm_expert_priority::demand_current_layer;
                if (entry.readiness > request.readiness) request.readiness = entry.readiness;
                request.demand_owned = true;
                result = { llm_expert_current_layer_schedule_disposition::promoted_exact_generation,
                    { plan.slot, plan.generation } };
                pimpl->counters.joins++;
                break;
            case impl::batch_action::create_successor:
                request.pending.active = true;
                request.pending.generation = plan.generation;
                request.pending.enqueue_ordinal = plan.enqueue_ordinal;
                request.pending.priority = llm_expert_priority::demand_current_layer;
                request.pending.readiness = entry.readiness;
                request.pending.waiters = 1;
                result = { llm_expert_current_layer_schedule_disposition::pending_successor,
                    { plan.slot, plan.generation } };
                pimpl->counters.current_layer_pending_successors++;
                break;
            case impl::batch_action::join_successor:
                request.pending.waiters++;
                if (entry.readiness > request.pending.readiness) request.pending.readiness = entry.readiness;
                result = { llm_expert_current_layer_schedule_disposition::pending_successor,
                    { plan.slot, plan.generation } };
                pimpl->counters.joins++;
                break;
        }
    }
    pimpl->next_ordinal = next_ordinal;
    pimpl->counters.current_layer_batches++;
    pimpl->counters.current_layer_batch_entries += count;
    pimpl->update_occupancy();
    return llm_expert_schedule_disposition::admitted;
}

llm_expert_schedule_result llm_expert_scheduler::take(
        llm_expert_request_handle handle,
        llm_expert_request_snapshot & result) noexcept {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    if (!handle.valid() || handle.slot >= pimpl->requests.size()) {
        return { llm_expert_schedule_disposition::invalid, {} };
    }
    auto & request = pimpl->requests[handle.slot];
    if (request.pending.active && request.pending.generation == handle.generation) {
        return { llm_expert_schedule_disposition::busy, {} };
    }
    if (request.state == llm_expert_request_state::free || request.generation != handle.generation) {
        pimpl->counters.stale_completions++;
        return { llm_expert_schedule_disposition::stale_generation, {} };
    }
    if (request.state != llm_expert_request_state::queued) {
        return { llm_expert_schedule_disposition::busy, {} };
    }
    request.state = llm_expert_request_state::submitting;
    result = {
        request.key,
        handle,
        request.priority,
        request.readiness,
        request.state,
        request.waiters,
        request.enqueue_ordinal,
        request.demand_owned,
        request.pending.active,
        request.pending.generation,
    };
    pimpl->update_occupancy();
    return { llm_expert_schedule_disposition::admitted, handle };
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
        request.demand_owned,
        request.pending.active,
        request.pending.generation,
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
    if (terminal == llm_expert_request_state::complete) pimpl->counters.terminal_complete++;
    if (terminal == llm_expert_request_state::failed) pimpl->counters.terminal_failed++;
    if (terminal == llm_expert_request_state::cancelled) pimpl->counters.terminal_cancelled++;
    pimpl->update_occupancy();
    return llm_expert_schedule_disposition::admitted;
}

llm_expert_schedule_disposition llm_expert_scheduler::cancel_optional(
        llm_expert_request_handle handle,
        llm_expert_request_state expected) noexcept {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    impl::request_record * request = pimpl->find(handle);
    if (request == nullptr) {
        pimpl->counters.stale_completions++;
        return llm_expert_schedule_disposition::stale_generation;
    }
    if (request->demand_owned || request->pending.active) {
        pimpl->counters.optional_cancellations_prevented++;
        return llm_expert_schedule_disposition::busy;
    }
    if (request->state != expected || !is_optional(request->priority) ||
        !valid_transition(expected, llm_expert_request_state::cancelling)) {
        return llm_expert_schedule_disposition::invalid;
    }
    request->state = llm_expert_request_state::cancelling;
    pimpl->update_occupancy();
    return llm_expert_schedule_disposition::admitted;
}

llm_expert_schedule_disposition llm_expert_scheduler::cancel_pending_successor(
        llm_expert_request_handle handle) noexcept {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    if (!handle.valid() || handle.slot >= pimpl->requests.size()) {
        return llm_expert_schedule_disposition::invalid;
    }
    auto & request = pimpl->requests[handle.slot];
    if (!request.pending.active || request.pending.generation != handle.generation) {
        pimpl->counters.stale_completions++;
        return llm_expert_schedule_disposition::stale_generation;
    }
    request.pending = {};
    pimpl->counters.current_layer_successor_cancellations++;
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
    if (request->pending.active) {
        const auto successor = request->pending;
        request->generation = successor.generation;
        request->enqueue_ordinal = successor.enqueue_ordinal;
        request->priority = successor.priority;
        request->readiness = successor.readiness;
        request->state = llm_expert_request_state::queued;
        request->waiters = successor.waiters;
        request->demand_owned = true;
        request->pending = {};
        pimpl->counters.terminal_releases++;
        pimpl->counters.flights_created++;
        pimpl->counters.current_layer_successor_activations++;
        pimpl->update_occupancy();
        return llm_expert_schedule_disposition::pending_successor;
    }
    request->key = { -1, -1 };
    request->enqueue_ordinal = 0;
    request->state = llm_expert_request_state::free;
    request->waiters = 0;
    request->demand_owned = false;
    request->pending = {};
    pimpl->counters.terminal_releases++;
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
        request.demand_owned = false;
        request.pending = {};
    }
    pimpl->update_occupancy();
    return true;
}

llm_expert_scheduler_diagnostics llm_expert_scheduler::diagnostics() const noexcept {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    return pimpl->counters;
}
