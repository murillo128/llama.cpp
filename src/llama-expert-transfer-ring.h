#pragma once

#include "llama-cold-expert-cache.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

enum class llm_transfer_lane_state {
    free,
    staging,
    in_flight,
    failed,
};

struct llm_transfer_lane_reference {
    uint32_t lane = 0;
    uint64_t generation = 0;
};

struct llm_transfer_ring_config {
    uint64_t byte_budget = 0;
    uint32_t minimum_lanes = 0;
    ggml_backend_dev_t target_device = nullptr;
    bool allow_non_cuda_target_for_testing = false;
    bool force_pageable_fallback_for_testing = false;
    bool force_no_events_for_testing = false;
    uint64_t initial_lane_generation_for_testing = 0;
};

struct llm_transfer_ring_faults {
    bool fail_allocation = false;
    size_t fail_stage_after_spans = SIZE_MAX;
    bool fail_pre_enqueue = false;
    bool fail_cleanup = false;
};

struct llm_transfer_binding {
    llm_transfer_lane_reference lane;
    llm_expert_bundle_descriptor destination;
    uint32_t hot_slot = 0;
};

struct llm_transfer_ring_diagnostics {
    uint64_t requested_bytes = 0;
    uint64_t actual_bytes = 0;
    uint64_t unused_budget_bytes = 0;
    uint64_t lane_payload_bytes = 0;
    uint64_t lane_footprint = 0;
    uint64_t alignment = 0;
    uint32_t effective_lanes = 0;
    std::string acquisition_method;
    std::string fallback_reason;
    bool pageable_fallback = false;
    uint64_t pinned_or_registered_bytes = 0;
    uint64_t fallback_count = 0;
    uint64_t lane_reservations = 0;
    uint64_t stage_bytes = 0;
    uint64_t stage_time_us = 0;
    uint64_t async_enqueues = 0;
    uint64_t synchronous_copies = 0;
    uint64_t waves = 0;
    uint64_t peak_in_flight_lanes = 0;
    uint64_t wave_synchronizations = 0;
    bool dedicated_transfer_backend = false;
    bool event_capable = false;
    uint32_t event_capacity = 0;
    uint32_t live_events = 0;
    uint32_t peak_live_events = 0;
    uint64_t event_records = 0;
    uint64_t compute_waits = 0;
    uint64_t event_synchronizations = 0;
    uint64_t first_h2d_enqueue_us = 0;
    uint64_t last_h2d_event_complete_us = 0;
    uint64_t h2d_compute_overlap_us = 0;
    uint64_t h2d_compute_overlap_bytes = 0;
    uint64_t h2d_bytes = 0;
    uint64_t h2d_time_us = 0;
    uint64_t failed_cleanups = 0;

    struct lane {
        uint64_t generation = 0;
        llm_transfer_lane_state state = llm_transfer_lane_state::free;
        llm_cold_reference cold;
        uint32_t hot_slot = 0;
        uint64_t hot_generation = 0;
    };
    std::vector<lane> lanes;
};

class llm_expert_transfer_ring {
public:
    explicit llm_expert_transfer_ring(
        llm_transfer_ring_config config,
        llm_transfer_ring_faults faults = {});
    ~llm_expert_transfer_ring();

    llm_expert_transfer_ring(const llm_expert_transfer_ring &) = delete;
    llm_expert_transfer_ring & operator=(const llm_expert_transfer_ring &) = delete;

    llm_expert_provider_result initialize(const llm_expert_bundle_descriptor & prototype) noexcept;
    llm_expert_provider_result reserve(
        llm_cold_expert_cache & cold_cache,
        llm_cold_reference cold,
        uint32_t hot_slot,
        uint64_t hot_generation,
        llm_transfer_lane_reference & lane) noexcept;
    llm_expert_provider_result stage(
        llm_transfer_lane_reference lane,
        const llm_expert_bundle_descriptor & cold_bundle) noexcept;
    llm_expert_provider_result transfer_wave(
        ggml_backend_t backend,
        const std::vector<llm_transfer_binding> & bindings) noexcept;
    llm_expert_provider_result wait_for_hot(
        ggml_backend_t compute_backend,
        uint32_t hot_slot,
        uint64_t hot_generation) noexcept;
    llm_expert_provider_result retire_hot(uint32_t hot_slot, uint64_t hot_generation) noexcept;
    llm_expert_provider_result cleanup_failed_lanes() noexcept;
    llm_expert_provider_result surrender() noexcept;
    llm_expert_provider_result validate_invariants() noexcept;
    llm_transfer_ring_diagnostics diagnostics() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
