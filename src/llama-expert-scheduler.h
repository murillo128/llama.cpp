#pragma once

#include "llama-expert-weight-provider.h"

#include <cstddef>
#include <cstdint>
#include <memory>

enum class llm_expert_priority : uint8_t {
    demand_current_layer,
    demand_future_dependency,
    prefetch_next,
    prefetch_speculative,
};

enum class llm_expert_readiness : uint8_t {
    host_ready,
    device_ready,
};

enum class llm_expert_request_state : uint8_t {
    free,
    queued,
    submitting,
    io_in_flight,
    host_ready,
    h2d_in_flight,
    device_ready,
    complete,
    cancelling,
    draining,
    cancelled,
    failed,
};

struct llm_expert_request_handle {
    uint32_t slot = UINT32_MAX;
    uint64_t generation = 0;

    bool valid() const { return slot != UINT32_MAX && generation != 0; }
};

struct llm_expert_scheduler_config {
    uint32_t layer_count = 0;
    uint32_t experts_per_layer = 0;
    uint32_t request_capacity = 0;
    uint32_t waiters_per_request = 0;
    uint32_t current_layer_demand_reserve = 0;
    uint64_t initial_generation_for_testing = 0;
};

enum class llm_expert_schedule_disposition {
    admitted,
    joined,
    dropped,
    busy,
    closed,
    invalid,
    stale_generation,
    generation_exhausted,
    pending_successor,
};

struct llm_expert_schedule_result {
    llm_expert_schedule_disposition disposition = llm_expert_schedule_disposition::invalid;
    llm_expert_request_handle handle;

    bool accepted() const {
        return disposition == llm_expert_schedule_disposition::admitted ||
            disposition == llm_expert_schedule_disposition::joined ||
            disposition == llm_expert_schedule_disposition::pending_successor;
    }
};

struct llm_expert_current_layer_schedule_entry {
    llm_expert_key key = { -1, -1 };
    llm_expert_readiness readiness = llm_expert_readiness::host_ready;
};

enum class llm_expert_current_layer_schedule_disposition : uint8_t {
    admitted_new,
    joined_exact_generation,
    promoted_exact_generation,
    pending_successor,
};

struct llm_expert_current_layer_schedule_result {
    llm_expert_current_layer_schedule_disposition disposition =
        llm_expert_current_layer_schedule_disposition::admitted_new;
    llm_expert_request_handle handle;
};

struct llm_expert_request_snapshot {
    llm_expert_key key = { -1, -1 };
    llm_expert_request_handle handle;
    llm_expert_priority priority = llm_expert_priority::prefetch_speculative;
    llm_expert_readiness readiness = llm_expert_readiness::host_ready;
    llm_expert_request_state state = llm_expert_request_state::free;
    uint32_t waiters = 0;
    uint64_t enqueue_ordinal = 0;
    bool demand_owned = false;
    bool has_pending_successor = false;
    uint64_t pending_successor_generation = 0;
};

struct llm_expert_scheduler_diagnostics {
    uint32_t request_capacity = 0;
    uint32_t active_requests = 0;
    uint32_t peak_active_requests = 0;
    uint32_t queued_requests = 0;
    uint32_t waiters_per_request = 0;
    uint32_t current_layer_demand_reserve = 0;
    uint64_t administration_bytes = 0;
    uint64_t flights_created = 0;
    uint64_t joins = 0;
    uint64_t promotions = 0;
    uint64_t drops = 0;
    uint64_t queued_preemptions = 0;
    uint64_t current_layer_batches = 0;
    uint64_t current_layer_batch_entries = 0;
    uint64_t current_layer_batch_rejections = 0;
    uint64_t current_layer_pending_successors = 0;
    uint64_t current_layer_successor_activations = 0;
    uint64_t current_layer_successor_cancellations = 0;
    uint64_t optional_cancellations_prevented = 0;
    uint64_t stale_completions = 0;
    uint64_t generation_exhaustions = 0;
    uint64_t terminal_complete = 0;
    uint64_t terminal_failed = 0;
    uint64_t terminal_cancelled = 0;
    uint64_t terminal_releases = 0;
    bool admission_closed = false;
};

class llm_expert_scheduler {
public:
    explicit llm_expert_scheduler(llm_expert_scheduler_config config);
    ~llm_expert_scheduler();

    llm_expert_scheduler(const llm_expert_scheduler &) = delete;
    llm_expert_scheduler & operator=(const llm_expert_scheduler &) = delete;

    llm_expert_schedule_result enqueue(
            llm_expert_key key,
            llm_expert_priority priority,
            llm_expert_readiness readiness) noexcept;
    llm_expert_schedule_disposition enqueue_current_layer_batch(
            const llm_expert_current_layer_schedule_entry * entries,
            size_t count,
            llm_expert_current_layer_schedule_result * results,
            size_t result_capacity,
            llm_expert_request_handle * revoked_optional,
            size_t revoked_capacity,
            size_t & revoked_count) noexcept;
    llm_expert_schedule_result take(
            llm_expert_request_handle handle,
            llm_expert_request_snapshot & request) noexcept;
    llm_expert_schedule_result take_next(llm_expert_request_snapshot & request) noexcept;
    llm_expert_schedule_disposition transition(
            llm_expert_request_handle handle,
            llm_expert_request_state expected,
            llm_expert_request_state next) noexcept;
    llm_expert_schedule_disposition finish(
            llm_expert_request_handle handle,
            llm_expert_request_state terminal) noexcept;
    llm_expert_schedule_disposition cancel_optional(
            llm_expert_request_handle handle,
            llm_expert_request_state expected) noexcept;
    llm_expert_schedule_disposition cancel_pending_successor(
            llm_expert_request_handle handle) noexcept;
    llm_expert_schedule_disposition release_terminal(llm_expert_request_handle handle) noexcept;
    bool shutdown() noexcept;
    llm_expert_scheduler_diagnostics diagnostics() const noexcept;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
