#pragma once

#include "llama-cold-expert-cache.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct llm_expert_storage_destination;

enum class llm_transfer_lane_state {
    free,
    staging,
    in_flight,
    failed,
};

struct llm_transfer_lane_reference {
    uint32_t lane = 0;
    uint64_t generation = 0;
    llm_expert_layout_class_id layout_class_id = 0;
};

struct llm_transfer_ring_config {
    uint64_t byte_budget = 0;
    uint32_t minimum_lanes = 0;
    ggml_backend_dev_t target_device = nullptr;
    bool allow_non_cuda_target_for_testing = false;
    bool force_pageable_fallback_for_testing = false;
    bool force_no_events_for_testing = false;
    uint64_t initial_lane_generation_for_testing = 0;
    uint32_t trace_capacity = 256;
    uint32_t delay_event_monitor_ms_for_testing = 0;
    bool allow_controlled_compute_for_testing = false;
    ggml_backend_event_t h2d_gate_event_for_testing = nullptr;
    uint32_t delay_background_stage_ms_for_testing = 0;
    uint32_t delay_cold_fill_ms_for_testing = 0;
    llm_expert_phase8_test_control * phase8_test_control = nullptr;
};

struct llm_transfer_interval {
    llm_expert_flight_id flight;
    llm_cold_reference cold;
    llm_transfer_lane_reference lane;
    uint32_t hot_slot = 0;
    uint64_t hot_generation = 0;
    uint64_t h2d_enqueue_us = 0;
    uint64_t h2d_complete_us = 0;
    uint64_t compute_begin_us = 0;
    uint64_t compute_complete_us = 0;
    uint64_t bytes = 0;
    uint64_t compute_work_id = 0;
    uint64_t compute_work = 0;
    bool cancelled = false;
    bool direct_storage = false;
};

struct llm_transfer_ring_faults {
    bool fail_allocation = false;
    size_t fail_stage_after_spans = SIZE_MAX;
    bool fail_pre_enqueue = false;
    bool fail_cleanup = false;
    bool fail_cold_fill = false;
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
    uint32_t layout_class_count = 0;
    std::vector<uint64_t> class_payload_bytes;
    std::vector<uint64_t> class_padding_bytes;
    std::vector<uint64_t> role_offsets;
    std::vector<uint64_t> role_extents;
    uint64_t alignment = 0;
    uint32_t effective_lanes = 0;
    std::string acquisition_method;
    std::string fallback_reason;
    bool pageable_fallback = false;
    uint64_t pinned_or_registered_bytes = 0;
    uint64_t fallback_count = 0;
    uint64_t lane_reservations = 0;
    uint64_t direct_storage_reservations = 0;
    uint64_t direct_storage_completions = 0;
    uint64_t direct_storage_bytes = 0;
    uint64_t cold_fill_attempts = 0;
    uint64_t cold_fill_queued = 0;
    uint64_t cold_fill_dropped = 0;
    uint64_t cold_fill_completed = 0;
    uint64_t cold_fill_failed = 0;
    uint64_t cold_fill_bytes = 0;
    uint64_t cold_fill_time_us = 0;
    uint32_t cold_fill_active = 0;
    uint32_t cold_fill_peak_active = 0;
    uint64_t stage_bytes = 0;
    std::vector<uint64_t> class_stage_bundles;
    std::vector<uint64_t> class_stage_bytes;
    uint64_t stage_time_us = 0;
    uint64_t async_enqueues = 0;
    uint64_t synchronous_copies = 0;
    uint64_t waves = 0;
    uint64_t peak_in_flight_lanes = 0;
    uint64_t wave_synchronizations = 0;
    bool dedicated_transfer_backend = false;
    bool event_capable = false;
    uint32_t h2d_event_capacity = 0;
    uint32_t compute_event_capacity = 0;
    uint32_t event_capacity = 0;
    uint32_t live_h2d_events = 0;
    uint32_t peak_live_h2d_events = 0;
    uint32_t live_compute_events = 0;
    uint32_t peak_live_compute_events = 0;
    uint32_t live_events = 0;
    uint32_t peak_live_events = 0;
    uint64_t h2d_event_records = 0;
    uint64_t h2d_event_waits = 0;
    uint64_t h2d_event_synchronizations = 0;
    uint64_t event_records = 0;
    uint64_t compute_waits = 0;
    uint64_t event_synchronizations = 0;
    uint64_t compute_event_records = 0;
    uint64_t compute_event_waits = 0;
    uint64_t compute_event_synchronizations = 0;
    uint64_t h2d_event_cancellations = 0;
    uint64_t compute_event_cancellations = 0;
    uint64_t h2d_event_allocations = 0;
    uint64_t compute_event_allocations = 0;
    uint64_t h2d_event_frees = 0;
    uint64_t compute_event_frees = 0;
    uint64_t compute_work = 0;
    uint32_t trace_capacity = 0;
    uint64_t trace_records = 0;
    uint64_t trace_records_dropped = 0;
    uint64_t first_h2d_enqueue_us = 0;
    uint64_t last_h2d_event_complete_us = 0;
    uint64_t h2d_compute_overlap_us = 0;
    uint64_t h2d_compute_overlap_bytes = 0;
    uint64_t h2d_compute_overlap_work = 0;
    uint64_t h2d_compute_overlap_flights = 0;
    uint64_t h2d_bytes = 0;
    std::vector<uint64_t> class_h2d_bundles;
    std::vector<uint64_t> class_h2d_bytes;
    uint64_t h2d_time_us = 0;
    uint64_t failed_cleanups = 0;

    struct lane {
        uint64_t generation = 0;
        llm_expert_layout_class_id layout_class_id = LLM_EXPERT_LAYOUT_CLASS_INVALID;
        llm_transfer_lane_state state = llm_transfer_lane_state::free;
        llm_cold_reference cold;
        uint32_t hot_slot = 0;
        uint64_t hot_generation = 0;
        bool direct_storage = false;
        bool direct_storage_complete = false;
    };
    std::vector<lane> lanes;
};

struct llm_transfer_ring_closeout_diagnostics {
    uint32_t queued_workers = 0;
    uint32_t running_workers = 0;
    uint32_t non_free_lanes = 0;
    uint32_t live_events = 0;
    bool invariants_ok = false;
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
    llm_expert_provider_result initialize(const llm_expert_layout_registry & registry) noexcept;
    llm_expert_provider_result reserve(
        llm_cold_expert_cache & cold_cache,
        llm_cold_reference cold,
        uint32_t hot_slot,
        uint64_t hot_generation,
        llm_transfer_lane_reference & lane,
        llm_expert_flight_id flight = {}) noexcept;
    llm_expert_provider_result reserve_direct_storage(
        llm_expert_layout_class_id layout_class_id,
        uint32_t hot_slot,
        uint64_t hot_generation,
        llm_transfer_lane_reference & lane,
        llm_expert_flight_id flight = {},
        bool wait_for_lane = true) noexcept;
    llm_expert_provider_result storage_destinations(
        llm_transfer_lane_reference lane,
        llm_expert_storage_destination * destinations,
        size_t destination_capacity,
        size_t & destination_count) noexcept;
    llm_expert_provider_result complete_direct_storage(
        llm_transfer_lane_reference lane,
        uint64_t bytes) noexcept;
    llm_expert_provider_result try_queue_cold_fill(
        llm_cold_expert_cache & cold_cache,
        llm_transfer_lane_reference lane) noexcept;
    llm_expert_provider_result discard_staging(
        llm_transfer_lane_reference lane) noexcept;
    llm_expert_provider_result stage(
        llm_transfer_lane_reference lane,
        const llm_expert_bundle_descriptor & cold_bundle) noexcept;
    llm_expert_provider_result transfer_wave(
        ggml_backend_t backend,
        const std::vector<llm_transfer_binding> & bindings) noexcept;
    // Model-load seed path: use the ring's bounded transfer backend and wait
    // until every destination is device-ready before returning.
    llm_expert_provider_result transfer_wave_blocking(
        const std::vector<llm_transfer_binding> & bindings,
        llm_cold_expert_cache * cold_fill_cache = nullptr) noexcept;
    // Atomically reserve and queue a background transfer. The Phase 8 path
    // does not wait for the ring mutex; explicit Phase 9 policy ordering may
    // wait for that bounded critical section, but never for a free lane.
    llm_expert_provider_result try_queue_background_transfer(
        llm_cold_expert_cache & cold_cache,
        llm_cold_reference cold,
        uint32_t hot_slot,
        uint64_t hot_generation,
        const llm_expert_bundle_descriptor & cold_bundle,
        const llm_expert_bundle_descriptor & destination,
        llm_transfer_lane_reference & lane,
        llm_expert_flight_id flight = {},
        bool wait_for_mutex = false) noexcept;
    llm_expert_provider_result wait_for_hot(
        ggml_backend_t compute_backend,
        uint32_t hot_slot,
        uint64_t hot_generation,
        bool ordered_terminal = false) noexcept;
    // Arm event completion monitoring without adding a dependency to the
    // current compute stream. Used only by same-key background promotion.
    llm_expert_provider_result monitor_h2d(llm_transfer_lane_reference lane) noexcept;
    llm_expert_provider_result poll_h2d(
        llm_transfer_lane_reference lane,
        llm_expert_same_key_h2d_state & state,
        uint64_t & remaining_bytes) noexcept;
    // Wait for one background H2D terminal without releasing its held lane.
    // Phase 9 uses this at a logical boundary before publishing terminals in
    // immutable origin-operation order.
    llm_expert_provider_result wait_background_h2d(
        llm_transfer_lane_reference lane) noexcept;
    // Release a terminal failed or event-complete background lane. Never waits
    // for or cancels unrelated current-token work.
    llm_expert_provider_result release_terminal_background(
        llm_transfer_lane_reference lane) noexcept;
    llm_expert_provider_result begin_compute_work(
        ggml_backend_t compute_backend,
        uint64_t work) noexcept;
    llm_expert_provider_result cancel_after_h2d(
        llm_transfer_lane_reference lane) noexcept;
    // Internal test seam. The supplied event is borrowed and must outlive any
    // transfer submitted while it is installed.
    llm_expert_provider_result set_h2d_gate_event_for_testing(
        ggml_backend_event_t event) noexcept;
    llm_expert_provider_result set_phase8_test_control_for_testing(
        llm_expert_phase8_test_control * control) noexcept;
    llm_expert_provider_result retire_hot(uint32_t hot_slot, uint64_t hot_generation) noexcept;
    llm_expert_provider_result cleanup_failed_lanes() noexcept;
    llm_expert_provider_result surrender() noexcept;
    llm_expert_provider_result validate_invariants() noexcept;
    llm_transfer_ring_diagnostics diagnostics() const;
    llm_transfer_ring_closeout_diagnostics closeout_diagnostics() const noexcept;
    std::vector<llm_transfer_interval> completed_intervals() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
