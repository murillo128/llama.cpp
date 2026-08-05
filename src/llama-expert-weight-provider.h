#pragma once

#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"
#include "llama-expert-cache-policy.h"
#include "llama-expert-prefetch.h"
#include "llama-expert-uma.h"

#include <cstddef>
#include <cstdint>
#include <array>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

class llm_expert_storage;
class llm_expert_async_transport;
class llm_expert_scheduler;

enum class llm_expert_provider_status {
    ready,
    allocation_failed,
    failed,
    cancelled,
};

enum class llm_expert_provider_error {
    none,
    invalid_key,
    invalid_descriptor,
    invalid_selection,
    invalid_binding,
    initialization_failed,
    allocation_failed,
    preparation_failed,
    unsupported_configuration,
    busy,
    copy_failed,
    stale_generation,
    generation_exhausted,
    metadata_mismatch,
    cancelled,
};

struct llm_expert_provider_result {
    llm_expert_provider_status status = llm_expert_provider_status::ready;
    llm_expert_provider_error error = llm_expert_provider_error::none;

    bool is_ready() const;

    static llm_expert_provider_result success();
    static llm_expert_provider_result failure(llm_expert_provider_error error);
};

struct llm_expert_key {
    int32_t layer;
    int32_t expert;

    bool is_valid(int32_t n_layer, int32_t n_expert) const;
};

using llm_expert_layout_class_id = uint8_t;

static constexpr llm_expert_layout_class_id LLM_EXPERT_LAYOUT_CLASS_INVALID = UINT8_MAX;
static constexpr uint32_t LLM_EXPERT_LAYOUT_CLASS_MAX = 8;

// Internal, caller-owned Phase 8 evidence controller. Model-facing production
// construction leaves this absent. It can hold exactly one key/generation-bound
// directive and is deliberately not part of llama.h or the public ABI.
enum class llm_expert_phase8_test_gate : uint8_t {
    none,
    background_queued_before_stage,
    background_h2d_in_flight,
    background_h2d_complete_before_provider_publication,
    auto_after_normalization_before_background_snapshot,
    auto_after_decision_before_same_key_join,
    background_before_provider_publication,
};

enum class llm_expert_phase8_test_fault : uint8_t {
    none,
    stage_copy_failed,
    h2d_failed,
    metadata_mismatch,
    stale_generation,
    scheduler_join_mismatch,
    wait_failed,
    publication_failed,
};

class llm_expert_phase8_test_control {
public:
    llm_expert_provider_result arm_gate(
            llm_expert_phase8_test_gate gate,
            llm_expert_key key,
            uint64_t generation) noexcept;
    llm_expert_provider_result arm_fault(
            llm_expert_phase8_test_fault fault,
            llm_expert_key key,
            uint64_t generation) noexcept;
    void wait_until_reached() noexcept;
    llm_expert_provider_result release_gate() noexcept;
    llm_expert_provider_result release_gate_and_arm_gate(
            llm_expert_phase8_test_gate next_gate) noexcept;
    llm_expert_provider_result release_gate_and_arm_fault(
            llm_expert_phase8_test_fault next_fault) noexcept;
    bool gate_reached() const noexcept;
    uint64_t observations() const noexcept;
    void observe(
            llm_expert_phase8_test_gate observed_gate,
            llm_expert_key observed_key,
            uint64_t observed_generation) noexcept;
    void wait_until_observed(llm_expert_phase8_test_gate observed_gate) noexcept;

    // Runtime-side internal operations. Both are no-ops unless a matching
    // one-shot directive was explicitly armed by a focused test.
    bool pause_if_armed(
            llm_expert_phase8_test_gate gate,
            llm_expert_key key,
            uint64_t generation) noexcept;
    bool consume_fault(
            llm_expert_phase8_test_fault fault,
            llm_expert_key key,
            uint64_t generation) noexcept;

private:
    enum class directive_kind : uint8_t { none, gate, fault };
    mutable std::mutex mutex;
    std::condition_variable condition;
    directive_kind kind = directive_kind::none;
    llm_expert_phase8_test_gate gate = llm_expert_phase8_test_gate::none;
    llm_expert_phase8_test_fault fault = llm_expert_phase8_test_fault::none;
    llm_expert_key key = { -1, -1 };
    uint64_t generation = 0;
    bool reached = false;
    bool released = false;
    uint64_t directive_epoch = 0;
    uint64_t released_epoch = 0;
    uint64_t observation_count = 0;
    llm_expert_phase8_test_gate last_observed_gate = llm_expert_phase8_test_gate::none;
};

struct llm_expert_phase8_closeout_witness {
    bool written = false;
    uint32_t write_count = 0;
    uint64_t background_submitted = 0;
    uint64_t background_published_completed = 0;
    uint64_t background_dropped = 0;
    uint64_t background_useful = 0;
    uint64_t background_wasted = 0;
    uint64_t background_busy = 0;
    // empty, queued/staging, H2D-in-flight, complete-unpublished, published,
    // failed, and cancelled, in that order.
    std::array<uint32_t, 7> background_lifecycle = {};
    uint32_t scheduler_active = 0;
    uint32_t scheduler_queued = 0;
    uint64_t scheduler_terminal_complete = 0;
    uint64_t scheduler_terminal_failed = 0;
    uint64_t scheduler_terminal_cancelled = 0;
    uint64_t scheduler_terminal_releases = 0;
    uint32_t ring_queued_workers = 0;
    uint32_t ring_running_workers = 0;
    uint32_t ring_non_free_lanes = 0;
    uint32_t ring_live_events = 0;
    uint64_t cold_hot_refs = 0;
    uint64_t cold_transfer_refs = 0;
    uint64_t cold_request_refs = 0;
    uint64_t cold_cpu_execution_refs = 0;
    uint64_t hot_pins = 0;
    uint32_t published_forward_mappings = 0;
    bool final_invariants_ok = false;
};

struct llm_expert_flight_id {
    uint64_t transport_epoch = 0;
    uint32_t request_slot = UINT32_MAX;
    uint64_t request_generation = 0;
    llm_expert_key key = { -1, -1 };
    llm_expert_layout_class_id layout_class_id = 0;

    bool valid() const {
        return transport_epoch != 0 && request_slot != UINT32_MAX && request_generation != 0 &&
            key.layer >= 0 && key.expert >= 0;
    }
};

struct llm_expert_projection_descriptor {
    ggml_tensor * weight = nullptr;
    ggml_tensor * bias = nullptr;
    ggml_tensor * scale = nullptr;
    ggml_backend_buffer_type_t buffer_type = nullptr;

    static llm_expert_projection_descriptor from(
            ggml_tensor * weight,
            ggml_tensor * bias,
            ggml_tensor * scale);
};

struct llm_expert_bundle_descriptor {
    int32_t layer;
    int32_t n_expert;

    llm_expert_projection_descriptor up;
    llm_expert_projection_descriptor gate;
    llm_expert_projection_descriptor gate_up;
    llm_expert_projection_descriptor down;

    bool uses_merged_gate_up() const;
    llm_expert_provider_result validate() const;
};

struct llm_expert_layout_class_descriptor {
    llm_expert_layout_class_id id = LLM_EXPERT_LAYOUT_CLASS_INVALID;
    uint64_t canonical_digest = 0;
    uint64_t payload_bytes = 0;
    llm_expert_bundle_descriptor prototype = {};
};

struct llm_expert_layout_registry {
    std::vector<llm_expert_layout_class_descriptor> classes;
    std::vector<llm_expert_layout_class_id> layer_ids;

    bool sealed() const { return !classes.empty() && !layer_ids.empty(); }
};

struct llm_expert_selection {
    int32_t layer;
    int32_t n_expert;
    int32_t n_expert_used;
    int64_t n_tokens;
    ggml_tensor * logical_ids;

    llm_expert_provider_result validate() const;
};

// Validates the internal Phase 8 two-branch representation. Logical/public
// routes are validated separately and never permit negative IDs.
llm_expert_provider_result llm_validate_hybrid_execution_ids(
        const int32_t * gpu_ids,
        const int32_t * cpu_ids,
        size_t count,
        int32_t gpu_capacity,
        int32_t cpu_capacity) noexcept;

struct llm_expert_graph_binding {
    const void * provider_identity = nullptr;
    int32_t layer = -1;
    llm_expert_layout_class_id layout_class_id = LLM_EXPERT_LAYOUT_CLASS_INVALID;

    llm_expert_projection_descriptor up;
    llm_expert_projection_descriptor gate;
    llm_expert_projection_descriptor gate_up;
    llm_expert_projection_descriptor down;

    ggml_tensor * execution_ids = nullptr;

    // Keeps cache-owned tensor storage alive for as long as the graph binding
    // can be reused. Resident/bootstrap bindings leave this empty.
    std::shared_ptr<void> generation_lease;
    uint64_t graph_epoch = 0;
    bool bootstrap = false;
    ggml_tensor * logical_ids = nullptr;

    // Present only for the Phase 8 hybrid graph. The existing fields above
    // remain the GPU/default branch so PROMOTE_AND_GPU retains its Phase 7
    // aggregate layout and graph shape.
    llm_expert_projection_descriptor cpu_up;
    llm_expert_projection_descriptor cpu_gate;
    llm_expert_projection_descriptor cpu_gate_up;
    llm_expert_projection_descriptor cpu_down;
    ggml_tensor * checkpoint_ids = nullptr;
    ggml_tensor * cpu_execution_ids = nullptr;
    bool hybrid = false;

    bool uses_merged_gate_up() const;
    llm_expert_provider_result validate(const llm_expert_selection & selection) const;
};

struct llm_expert_provider_stats {
    uint64_t objects_created = 0;
    uint64_t bind_calls = 0;
    uint64_t prepare_calls = 0;
    uint64_t handles_acquired = 0;
    uint64_t handles_released = 0;
    uint64_t allocations = 0;
    uint64_t callbacks = 0;
    uint64_t tensor_copies = 0;
    uint64_t synchronizations = 0;
    uint64_t failures = 0;
    uint64_t cancellations = 0;
    uint64_t bundle_registrations = 0;
    uint64_t bundle_full_validations = 0;
    uint64_t bundle_fast_path_hits = 0;
    uint64_t requested_capacity = 0;
    uint64_t effective_capacity = 0;
    uint64_t pool_bytes = 0;
    uint64_t pool_generations = 0;
    uint64_t graph_epoch = 0;
    uint64_t bootstrap_bindings = 0;
    uint64_t hot_bindings = 0;
    uint64_t trims = 0;
    uint64_t surrender_successes = 0;
    uint64_t surrender_busy = 0;
};

struct llm_expert_phase10_lead_event {
    uint64_t sequence = 0;
    uint64_t ubatch_ordinal = 0;
    int32_t layer = -1;
    uint64_t steady_ns = 0;
};

struct llm_expert_phase10_scheduler_event {
    uint64_t sequence = 0;
    llm_expert_key key = { -1, -1 };
    uint64_t enqueue_ns = 0;
    uint64_t take_ns = 0;
};

struct llm_expert_phase10_issue_ahead_event {
    uint64_t sequence = 0;
    uint64_t request = 0;
    uint64_t ubatch_ordinal = 0;
    int32_t layer = -1;
    uint32_t logical_ids = 0;
    uint32_t unique_ids = 0;
    uint32_t demand_misses = 0;
    uint32_t scheduler_enqueue_attempts_before_first_wait = 0;
    uint32_t scheduler_enqueued_before_first_take = 0;
    uint32_t scheduler_release_waits = 0;
    uint32_t storage_reads = 0;
    uint32_t storage_reads_submitted_before_first_wait = 0;
    uint32_t demand_ready_before_use = 0;
    uint64_t first_take_us = 0;
    uint64_t first_wait_us = 0;
    uint64_t last_demand_ready_us = 0;
    uint64_t demand_completion_us = 0;
    bool first_take_after_all_demand_enqueues = false;
    bool first_wait_after_all_demand_enqueue_attempts = false;
    bool first_wait_after_all_storage_submissions = false;
    bool all_demand_ready_before_use = false;
    bool serial_control = false;
};

struct llm_expert_phase10_storage_event {
    struct source_segment {
        uint64_t file_offset = 0;
        uint64_t byte_count = 0;
    };

    llm_expert_flight_id flight;
    uint32_t operation_index = 0;
    uint64_t submit_us = 0;
    uint64_t complete_us = 0;
    uint64_t completed_bytes = 0;
    uint64_t useful_bytes = 0;
    uint64_t operation_file_offset = 0;
    uint32_t source_segment_count = 0;
    std::array<source_segment, 12> source_segments;
};

struct llm_expert_phase10_h2d_event {
    llm_expert_flight_id flight;
    uint64_t enqueue_us = 0;
    uint64_t complete_us = 0;
    uint64_t bytes = 0;
    bool cancelled = false;
};

enum class llm_expert_prefetch_outcome : uint8_t {
    pending,
    timely_useful,
    late_joined,
    wasted_unused,
    cancelled_before_io,
    cancelled_drained,
    rejected,
};

struct llm_expert_phase10_prediction_event {
    uint64_t sequence = 0;
    uint64_t request = 0;
    uint64_t token = 0;
    uint64_t deadline_token = 0;
    uint64_t config_digest = 0;
    uint64_t predictor_digest = 0;
    uint64_t post_event_digest = 0;
    uint64_t score = 0;
    uint64_t storage_bytes = 0;
    uint64_t h2d_bytes = 0;
    uint64_t predictor_compute_ns = 0;
    uint64_t enqueue_us = 0;
    uint64_t host_ready_us = 0;
    uint64_t device_ready_us = 0;
    int32_t source_layer = -1;
    llm_expert_key key = { -1, -1 };
    uint32_t rank = 0;
    uint32_t cold_slot = UINT32_MAX;
    uint64_t cold_generation = 0;
    uint32_t hot_slot = UINT32_MAX;
    uint64_t hot_generation = 0;
    uint32_t scheduler_slot = UINT32_MAX;
    uint64_t scheduler_generation = 0;
    llm_expert_prefetch_trigger trigger = llm_expert_prefetch_trigger::token_end;
    llama_expert_prefetch_readiness readiness = LLAMA_EXPERT_PREFETCH_READINESS_HOST_READY;
    uint8_t priority = 0;
    llm_expert_prefetch_outcome outcome = llm_expert_prefetch_outcome::pending;
    bool admitted = false;
    bool demand_claimed = false;
    bool cold_ready = false;
    bool device_ready = false;
    bool circuit_open_after = false;
};

struct llm_expert_phase10_route_event {
    uint64_t request = 0;
    uint64_t token = 0;
    uint64_t post_event_digest = 0;
    uint32_t id_offset = 0;
    uint32_t id_count = 0;
    int32_t layer = -1;
};

enum class llm_expert_residency_origin : uint8_t {
    demand,
    static_seed,
    speculative,
};

bool llm_expert_speculative_victim_precedes(
        uint64_t candidate_deadline,
        uint64_t candidate_utility,
        uint32_t candidate_slot,
        uint64_t incumbent_deadline,
        uint64_t incumbent_utility,
        uint32_t incumbent_slot) noexcept;

struct llm_hot_cache_diagnostics {
    struct auto_decision {
        uint64_t request = 0;
        int32_t layer = -1;
        int32_t expert = -1;
        llama_expert_auto_cost_model cost = {};
        bool prefill = false;
        uint64_t lanes = 0;
        uint64_t bundle_bytes = 0;
        uint64_t queued_cpu_work_ns = 0;
        uint64_t queued_h2d_work_ns = 0;
        uint64_t queued_gpu_work_ns = 0;
        bool same_key_h2d_present = false;
        uint8_t same_key_h2d_state = 0;
        uint64_t same_key_h2d_remaining_bytes = 0;
        uint64_t cpu_work_ns = 0;
        uint64_t h2d_work_ns = 0;
        uint64_t gpu_work_ns = 0;
        uint64_t cpu_finish_ns = UINT64_MAX;
        uint64_t gpu_finish_ns = UINT64_MAX;
        uint8_t backend = 1;
        uint8_t reason = 3;
        bool overflow = true;
    };

    llama_expert_miss_policy configured_miss_policy = LLAMA_EXPERT_MISS_POLICY_PROMOTE_AND_GPU;
    bool background_promotion_configured = false;
    uint32_t auto_cost_model_version = 0;
    uint64_t auto_cost_model_digest = 0;
    uint64_t hybrid_bindings = 0;
    uint64_t gpu_execution_lanes = 0;
    uint64_t cpu_execution_lanes = 0;
    uint64_t mixed_execution_layers = 0;
    uint64_t cpu_fallback_unique_keys = 0;
    uint64_t h2d_bytes_avoided_for_current_output = 0;
    uint64_t auto_cpu_decisions = 0;
    uint64_t auto_gpu_decisions = 0;
    uint64_t auto_tie_decisions = 0;
    uint64_t auto_overflow_decisions = 0;
    uint64_t auto_decision_records = 0;
    uint64_t auto_decision_records_dropped = 0;
    uint64_t auto_decision_digest = 0;
    std::vector<auto_decision> auto_decisions;
    uint64_t background_submitted = 0;
    uint64_t background_completed = 0;
    uint64_t background_useful = 0;
    uint64_t background_wasted = 0;
    uint64_t background_dropped = 0;
    uint64_t background_busy = 0;
    uint64_t background_later_joins = 0;
    uint64_t background_h2d_bytes = 0;
    uint32_t active_background_flights = 0;
    uint32_t peak_background_flights = 0;
    bool phase10_prefetch_configured = false;
    bool phase10_seed_configured = false;
    bool phase10_seed_complete = false;
    uint64_t phase10_seed_attempts = 0;
    uint64_t phase10_seed_failures = 0;
    uint64_t phase10_seed_entries = 0;
    uint64_t phase10_seed_storage_bytes = 0;
    uint64_t phase10_seed_h2d_bytes = 0;
    llm_expert_key phase10_seed_last_touch = { -1, -1 };
    uint64_t phase10_issue_ahead_events = 0;
    uint64_t phase10_issue_ahead_violations = 0;
    uint64_t phase10_prediction_events = 0;
    uint64_t phase10_prediction_events_dropped = 0;
    uint64_t phase10_route_events = 0;
    uint64_t phase10_route_events_dropped = 0;
    uint64_t phase10_predictions_admitted = 0;
    uint64_t phase10_predictions_rejected = 0;
    uint64_t phase10_timely_useful = 0;
    uint64_t phase10_late_joined = 0;
    uint64_t phase10_wasted_unused = 0;
    uint64_t phase10_cancelled_before_io = 0;
    uint64_t phase10_cancelled_drained = 0;
    uint64_t phase10_predictor_compute_ns = 0;
    uint64_t phase10_predictor_digest = 0;
    uint64_t phase10_predictor_state_digest = 0;
    uint64_t phase10_circuit_opens = 0;
    bool phase10_circuit_open = false;
    bool phase10_runtime_failed = false;
    std::vector<llm_expert_phase10_prediction_event> phase10_prediction_trace;
    std::vector<llm_expert_phase10_route_event> phase10_route_trace;
    std::vector<int32_t> phase10_route_ids;
    uint32_t requested_capacity = 0;
    uint32_t effective_capacity = 0;
    uint64_t pool_bytes = 0;
    uint32_t layout_class_count = 0;
    uint64_t layout_registry_administration_bytes = 0;
    uint32_t layout_preflight_consumer_count = 0;
    bool layout_preflight_passed = false;
    uint64_t hot_slot_stride = 0;
    std::vector<uint64_t> layout_class_digests;
    std::vector<uint64_t> layout_class_payload_bytes;
    std::vector<uint64_t> layout_class_hot_padding_bytes;
    std::vector<uint64_t> layout_class_cold_padding_bytes;
    std::vector<uint64_t> layout_class_lane_padding_bytes;
    std::vector<uint64_t> layout_class_stage_bundles;
    std::vector<uint64_t> layout_class_stage_bytes;
    std::vector<uint64_t> layout_class_h2d_bundles;
    std::vector<uint64_t> layout_class_h2d_bytes;
    std::vector<uint64_t> layout_hot_role_offsets;
    std::vector<uint64_t> layout_hot_role_extents;
    std::vector<uint64_t> layout_cold_role_offsets;
    std::vector<uint64_t> layout_cold_role_extents;
    std::vector<uint64_t> layout_lane_role_offsets;
    std::vector<uint64_t> layout_lane_role_extents;
    std::vector<llm_expert_layout_class_id> layout_layer_ids;
    uint64_t descriptor_discovery_graphs = 0;
    uint64_t descriptor_discovery_bindings = 0;
    uint64_t descriptor_discovery_scheduler_reserve_calls = 0;
    uint64_t descriptor_discovery_backend_bytes_before = 0;
    uint64_t descriptor_discovery_backend_bytes_after = 0;
    uint64_t final_bootstrap_source_bindings = 0;
    uint64_t complete_deferred_payload_bytes = 0;
    bool complete_deferred_payload_in_compute_workspace = false;
    std::vector<uint64_t> scheduler_backend_bytes_before_discovery;
    std::vector<uint64_t> scheduler_backend_bytes_after_discovery;
    std::vector<uint64_t> scheduler_backend_bytes_after_hierarchy;
    std::vector<uint64_t> scheduler_backend_bytes_after_final_reserve;
    uint64_t graph_epoch = 0;
    uint64_t generation = 0;
    uint32_t n_expert = 0;
    uint32_t n_expert_used = 0;
    uint32_t last_context_n_ctx = 0;
    uint32_t last_context_n_ubatch = 0;
    uint32_t last_context_extent = 0;
    uint32_t conservative_required_capacity = 0;
    uint64_t context_validations = 0;
    uint64_t context_rejections = 0;
    uint64_t requests = 0;
    uint64_t exclusive_busy_failures = 0;
    uint64_t remap_checkpoints = 0;
    uint64_t logical_ids = 0;
    uint64_t unique_ids = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t admissions = 0;
    uint64_t evictions = 0;
    uint64_t no_writeback_evictions = 0;
    uint64_t generation_changes = 0;
    uint64_t stale_generation_failures = 0;
    uint64_t copy_failures = 0;
    uint64_t failed_cleanups = 0;
    uint64_t pin_acquires = 0;
    uint64_t pin_releases = 0;
    uint64_t current_pins = 0;
    uint64_t peak_pins = 0;
    uint64_t h2d_bytes = 0;
    uint64_t h2d_time_us = 0;
    uint64_t execution_id_read_bytes = 0;
    uint64_t execution_id_write_bytes = 0;
    int32_t last_remap_layer = -1;
    std::vector<int32_t> last_logical_ids;
    std::vector<int32_t> last_execution_ids;
    uint64_t remap_dynamic_allocations = 0;
    uint64_t synchronization_checkpoints = 0;
    std::vector<uintptr_t> slot_tensor_addresses;
    struct slot {
        int32_t layer = -1;
        int32_t expert = -1;
        llm_expert_layout_class_id layout_class_id = LLM_EXPERT_LAYOUT_CLASS_INVALID;
        uint64_t generation = 0;
        uint64_t last_use = 0;
        uint32_t refcount = 0;
        uint32_t cold_slot = 0;
        uint64_t cold_generation = 0;
        bool has_cold_backing = false;
        llm_expert_residency_origin origin = llm_expert_residency_origin::demand;
        bool speculative_consumed = false;
        uint64_t speculative_deadline = 0;
        uint64_t speculative_utility = 0;
        enum state_type {
            free,
            reserved,
            loading,
            ready,
            pinned,
            evicting,
            failed,
        } state = free;
    };
    std::vector<slot> slots;
    struct cold_slot {
        int32_t layer = -1;
        int32_t expert = -1;
        llm_expert_layout_class_id layout_class_id = LLM_EXPERT_LAYOUT_CLASS_INVALID;
        uint64_t generation = 0;
        uint64_t last_use = 0;
        llm_expert_residency_origin origin = llm_expert_residency_origin::demand;
        enum state_type {
            free,
            reserved,
            loading,
            ready,
            evicting,
            failed,
        } state = free;
    };
    std::vector<cold_slot> cold_slots;
    ggml_backend_buffer_type_t source_buffer_type = nullptr;
    ggml_backend_buffer_type_t target_buffer_type = nullptr;
    bool source_pageable = false;
    uint64_t source_pinned_bytes = 0;
    uint64_t cold_requested_bytes = 0;
    uint64_t cold_actual_bytes = 0;
    uint64_t cold_unused_budget_bytes = 0;
    uint64_t cold_bundle_payload_bytes = 0;
    uint64_t cold_slot_footprint = 0;
    uint64_t cold_alignment = 0;
    uint32_t cold_effective_slots = 0;
    bool cold_pageable = false;
    uint64_t cold_requests = 0;
    uint64_t cold_hits = 0;
    uint64_t cold_misses = 0;
    uint64_t cold_admissions = 0;
    uint64_t cold_evictions = 0;
    uint64_t cold_speculative_admissions = 0;
    uint64_t cold_speculative_replacements = 0;
    uint64_t cold_speculative_rejections = 0;
    uint64_t cold_speculative_demand_consumptions = 0;
    uint64_t cold_source_copy_bundles = 0;
    uint64_t cold_source_copy_bytes = 0;
    uint64_t cold_source_copy_time_us = 0;
    uint64_t cold_failed_copies = 0;
    uint64_t cold_failed_cleanups = 0;
    uint64_t cold_generation_changes = 0;
    uint64_t cold_invariant_failures = 0;
    uint64_t cold_current_hot_refs = 0;
    uint64_t cold_peak_hot_refs = 0;
    uint64_t cold_current_transfer_refs = 0;
    uint64_t cold_peak_transfer_refs = 0;
    uint64_t cold_current_request_refs = 0;
    uint64_t cold_peak_request_refs = 0;
    uint64_t cold_current_cpu_execution_refs = 0;
    uint64_t cold_peak_cpu_execution_refs = 0;
    bool cold_residency_supported = false;
    std::string cold_residency_unavailable_reason;
    uint64_t cold_ready_logical_bytes = 0;
    uint64_t cold_ready_page_count = 0;
    uint64_t cold_resident_ready_page_count = 0;
    uint64_t cold_resident_ready_bytes = 0;
    uint64_t cold_reclaimed_bytes = 0;
    uint64_t cold_reclaim_failures = 0;
    uint64_t uma_physical_ram_bytes = 0;
    uint64_t uma_memory_available_bytes = 0;
    uint64_t uma_cgroup_memory_max_bytes = 0;
    uint64_t uma_cgroup_memory_current_bytes = 0;
    uint64_t uma_process_rss_bytes = 0;
    uint64_t uma_process_swap_bytes = 0;
    uint64_t uma_safe_pool_bytes = 0;
    uint64_t uma_effective_pool_bytes = 0;
    uint64_t uma_model_capacity_bytes = 0;
    uint64_t uma_model_cap_unused_safe_bytes = 0;
    uint64_t uma_alignment_remainder_bytes = 0;
    uint64_t uma_effective_slot_count = 0;
    uint64_t uma_system_reserve_bytes = 0;
    uint64_t uma_runtime_reserve_bytes = 0;
    uint64_t uma_runtime_delta_bytes = 0;
    uint64_t uma_headroom_remainder_bytes = 0;
    uint64_t uma_pressure_samples = 0;
    uint64_t uma_pressure_rejections = 0;
    uint64_t uma_storage_misses = 0;
    uint64_t uma_resident_hot_hits = 0;
    uint64_t uma_prepared_cold_hits = 0;
    uint64_t uma_degraded_hits = 0;
    uint64_t uma_unknown_residency_hits = 0;
    uint64_t uma_major_faults = 0;
    uint64_t uma_psi_full_total_usec = 0;
    uint64_t uma_zram_write_bytes = 0;
    uint64_t uma_zswap_write_pages = 0;
    bool uma_autofit = false;
    bool uma_pressure_circuit_open = false;
    bool uma_swap_counters_supported = false;
    bool uma_psi_full_supported = false;
    bool uma_zram_present = false;
    bool uma_zram_counters_supported = false;
    bool uma_zswap_enabled = false;
    bool uma_zswap_counters_supported = false;
    bool uma_nvidia_hmm_counters_supported = false;
    std::string uma_zram_status_reason;
    std::string uma_zswap_status_reason;
    std::string uma_nvidia_hmm_status_reason;
    std::string uma_pressure_rejection_reason;
    std::string uma_telemetry_unavailable_reason;
    uint64_t ring_requested_bytes = 0;
    uint64_t ring_actual_bytes = 0;
    uint64_t ring_lane_footprint = 0;
    uint32_t ring_effective_lanes = 0;
    uint64_t ring_pinned_or_registered_bytes = 0;
    std::string ring_acquisition_method;
    std::string ring_fallback_reason;
    bool ring_pageable_fallback = false;
    uint64_t ring_fallback_count = 0;
    uint64_t ring_lane_reservations = 0;
    uint64_t ring_stage_bytes = 0;
    uint64_t ring_stage_time_us = 0;
    uint64_t ring_async_enqueues = 0;
    uint64_t ring_synchronous_copies = 0;
    uint64_t ring_waves = 0;
    uint64_t ring_peak_in_flight_lanes = 0;
    uint64_t ring_wave_synchronizations = 0;
    bool ring_dedicated_transfer_backend = false;
    bool ring_event_capable = false;
    uint32_t ring_h2d_event_capacity = 0;
    uint32_t ring_compute_event_capacity = 0;
    uint32_t ring_event_capacity = 0;
    uint32_t ring_live_h2d_events = 0;
    uint32_t ring_peak_live_h2d_events = 0;
    uint32_t ring_live_compute_events = 0;
    uint32_t ring_peak_live_compute_events = 0;
    uint32_t ring_live_events = 0;
    uint32_t ring_peak_live_events = 0;
    uint64_t ring_h2d_event_records = 0;
    uint64_t ring_h2d_event_waits = 0;
    uint64_t ring_h2d_event_synchronizations = 0;
    uint64_t ring_event_records = 0;
    uint64_t ring_compute_waits = 0;
    uint64_t ring_event_synchronizations = 0;
    uint64_t ring_compute_event_records = 0;
    uint64_t ring_compute_event_waits = 0;
    uint64_t ring_compute_event_synchronizations = 0;
    uint64_t ring_h2d_event_cancellations = 0;
    uint64_t ring_compute_event_cancellations = 0;
    uint64_t ring_h2d_event_allocations = 0;
    uint64_t ring_compute_event_allocations = 0;
    uint64_t ring_h2d_event_frees = 0;
    uint64_t ring_compute_event_frees = 0;
    uint64_t ring_compute_work = 0;
    uint32_t ring_trace_capacity = 0;
    uint64_t ring_trace_records = 0;
    uint64_t ring_trace_records_dropped = 0;
    uint64_t ring_first_h2d_enqueue_us = 0;
    uint64_t ring_last_h2d_event_complete_us = 0;
    uint64_t ring_h2d_compute_overlap_us = 0;
    uint64_t ring_h2d_compute_overlap_bytes = 0;
    uint64_t ring_h2d_compute_overlap_work = 0;
    uint64_t ring_h2d_compute_overlap_flights = 0;
    uint64_t disk_h2d_overlap_us = 0;
    uint64_t disk_h2d_overlap_bytes = 0;
    uint64_t disk_h2d_overlap_read_bytes = 0;
    uint64_t disk_h2d_overlap_flights = 0;
    uint64_t disk_h2d_overlap_events = 0;
    uint64_t disk_h2d_overlap_read_flights = 0;
    uint64_t disk_h2d_overlap_transfer_flights = 0;
    uint64_t disk_h2d_overlap_pairs = 0;
    uint64_t disk_h2d_overlap_pair_digest = 0;
    uint64_t post_h2d_cancellations = 0;
    llm_expert_flight_id last_cancelled_flight;
    uint32_t last_cancelled_lane = UINT32_MAX;
    uint64_t last_cancelled_lane_generation = 0;
    uint32_t last_cancelled_hot_slot = UINT32_MAX;
    uint64_t last_cancelled_hot_generation = 0;
    llm_expert_flight_id last_completed_flight;
    uint32_t last_completed_lane = UINT32_MAX;
    uint64_t last_completed_lane_generation = 0;
    uint32_t last_completed_hot_slot = UINT32_MAX;
    uint64_t last_completed_hot_generation = 0;
    llm_expert_flight_id retry_after_cancel_flight;
    uint32_t retry_after_cancel_lane = UINT32_MAX;
    uint64_t retry_after_cancel_lane_generation = 0;
    uint32_t retry_after_cancel_hot_slot = UINT32_MAX;
    uint64_t retry_after_cancel_hot_generation = 0;
    int32_t last_execution_backend_device_type = -1;
    llm_expert_provider_error last_failure_error = llm_expert_provider_error::none;
    llm_expert_provider_error last_remap_error = llm_expert_provider_error::none;
    uint64_t ring_h2d_bytes = 0;
    uint64_t ring_h2d_time_us = 0;
    uint64_t ring_failed_cleanup = 0;
    llm_expert_cache_policy_diagnostics policy;
    std::vector<llm_expert_cache_policy_domain_diagnostics> policy_domains;
    std::vector<llm_expert_cache_policy_event> policy_events;
    llm_expert_cache_policy_diagnostics cold_policy;
    std::vector<llm_expert_cache_policy_domain_diagnostics> cold_policy_domains;
    std::vector<llm_expert_cache_policy_event> cold_policy_events;
    std::vector<llm_expert_phase10_lead_event> phase10_lead_events;
    uint32_t phase10_lead_event_capacity = 0;
    uint64_t phase10_lead_events_dropped = 0;
    std::vector<llm_expert_phase10_scheduler_event> phase10_scheduler_events;
    uint32_t phase10_scheduler_event_capacity = 0;
    uint64_t phase10_scheduler_events_dropped = 0;
    std::vector<llm_expert_phase10_issue_ahead_event> phase10_issue_ahead_trace;
    uint32_t phase10_issue_ahead_trace_capacity = 0;
    uint64_t phase10_issue_ahead_trace_dropped = 0;
    std::vector<llm_expert_phase10_storage_event> phase10_storage_events;
    uint32_t phase10_storage_event_capacity = 0;
    uint64_t phase10_storage_events_dropped = 0;
    std::vector<llm_expert_phase10_h2d_event> phase10_h2d_events;
    uint32_t phase10_h2d_event_capacity = 0;
    uint64_t phase10_h2d_events_dropped = 0;
};

enum class llm_expert_execution_backend : uint8_t {
    cpu,
    gpu,
};

enum class llm_expert_auto_reason : uint8_t {
    cpu_faster,
    gpu_faster_or_hysteresis,
    tie,
    overflow,
};

enum class llm_expert_same_key_h2d_state : uint8_t {
    none,
    queued_or_staging,
    h2d_in_flight,
    h2d_complete_unpublished,
};

struct llm_expert_auto_input {
    llama_expert_auto_cost_model cost = {};
    bool prefill = false;
    uint64_t lanes = 0;
    uint64_t bundle_bytes = 0;
    uint64_t queued_cpu_work_ns = 0;
    uint64_t queued_h2d_work_ns = 0;
    uint64_t queued_gpu_work_ns = 0;
    bool same_key_h2d_present = false;
    llm_expert_same_key_h2d_state same_key_h2d_state = llm_expert_same_key_h2d_state::none;
    uint64_t same_key_h2d_remaining_bytes = 0;
};

struct llm_expert_auto_result {
    llm_expert_execution_backend backend = llm_expert_execution_backend::gpu;
    llm_expert_auto_reason reason = llm_expert_auto_reason::overflow;
    uint64_t cpu_work_ns = 0;
    uint64_t h2d_work_ns = 0;
    uint64_t gpu_work_ns = 0;
    uint64_t cpu_finish_ns = UINT64_MAX;
    uint64_t gpu_finish_ns = UINT64_MAX;
    bool overflow = true;
};

llm_expert_auto_result llm_evaluate_expert_auto(const llm_expert_auto_input & input) noexcept;

struct llm_expert_graph_diagnostics {
    uint64_t operation_hash = 0;
    int32_t node_count = 0;
    int32_t binding_count = 0;
    int32_t binding_capacity = 0;
    uint64_t inflight_handles = 0;
    int32_t graphs_reused = 0;
};

class llm_expert_weight_provider;

enum class llm_expert_provider_initialization_stage : uint8_t {
    none,
    descriptors_before_scheduler_reserve,
    workspace_before_persistent_pool,
};

class llm_expert_handle {
public:
    llm_expert_handle() = default;
    llm_expert_handle(llm_expert_weight_provider * provider, uint64_t lease_id, uint64_t repetitions = 1);
    ~llm_expert_handle();

    llm_expert_handle(const llm_expert_handle &) = delete;
    llm_expert_handle & operator=(const llm_expert_handle &) = delete;

    llm_expert_handle(llm_expert_handle && other) noexcept;
    llm_expert_handle & operator=(llm_expert_handle && other) noexcept;

    bool is_valid() const;
    void reset();

private:
    friend class llm_expert_execution_plan;

    llm_expert_weight_provider * provider = nullptr;
    uint64_t lease_id = 0;
    uint64_t repetitions = 0;
};

class llm_expert_execution_plan {
public:
    llm_expert_execution_plan() = default;
    ~llm_expert_execution_plan();

    llm_expert_execution_plan(const llm_expert_execution_plan &) = delete;
    llm_expert_execution_plan & operator=(const llm_expert_execution_plan &) = delete;

    llm_expert_execution_plan(llm_expert_execution_plan && other) noexcept;
    llm_expert_execution_plan & operator=(llm_expert_execution_plan && other) noexcept;

    void reserve(size_t capacity);
    void add_handle(llm_expert_handle handle);
    void absorb(llm_expert_execution_plan && other);
    void set_result(llm_expert_provider_result result);
    void reset();

    const llm_expert_provider_result & get_result() const;
    size_t handle_count() const;

private:
    llm_expert_provider_result result;
    std::vector<llm_expert_handle> handles;
};

class llm_expert_weight_provider {
public:
    virtual ~llm_expert_weight_provider() = default;

    virtual llm_expert_provider_result bind(
            const llm_expert_bundle_descriptor & bundle,
            const llm_expert_selection & selection,
            llm_expert_graph_binding & binding) noexcept = 0;

    virtual llm_expert_provider_result bind_graph(
            ggml_context * graph_ctx,
            const llm_expert_bundle_descriptor & bundle,
            const llm_expert_selection & selection,
            llm_expert_graph_binding & binding) noexcept {
        (void) graph_ctx;
        return bind(bundle, selection, binding);
    }

    virtual llm_expert_provider_result prepare(
            const std::vector<llm_expert_graph_binding> & bindings,
            llm_expert_execution_plan & plan,
            uint64_t sequence_owner = 0,
            bool sequence_start = false) noexcept = 0;

    virtual llm_expert_provider_result end_prefetch_sequence(
            uint64_t sequence_owner) noexcept {
        (void) sequence_owner;
        return llm_expert_provider_result::success();
    }

    virtual llm_expert_provider_stats get_stats() const noexcept = 0;

    virtual llm_expert_provider_result validate_context_extent(
            uint32_t n_ctx,
            uint32_t n_ubatch) noexcept {
        (void) n_ctx;
        (void) n_ubatch;
        return llm_expert_provider_result::success();
    }
    virtual llm_expert_provider_result remap_checkpoint(
            const llm_expert_graph_binding & binding,
            const int32_t * logical_ids,
            size_t logical_id_count,
            int32_t * execution_ids,
            bool (*abort_callback)(void *) = nullptr,
            void * abort_callback_data = nullptr) noexcept {
        (void) binding;
        (void) logical_ids;
        (void) logical_id_count;
        (void) execution_ids;
        (void) abort_callback;
        (void) abort_callback_data;
        return llm_expert_provider_result::failure(llm_expert_provider_error::unsupported_configuration);
    }
    virtual llm_expert_provider_result remap_checkpoint_tensor(
            const llm_expert_graph_binding & binding,
            ggml_backend_t execution_backend = nullptr,
            bool (*abort_callback)(void *) = nullptr,
            void * abort_callback_data = nullptr) noexcept {
        (void) binding;
        (void) execution_backend;
        (void) abort_callback;
        (void) abort_callback_data;
        return llm_expert_provider_result::failure(llm_expert_provider_error::unsupported_configuration);
    }
    virtual llm_expert_provider_result cleanup_failed_slots() noexcept {
        return llm_expert_provider_result::failure(llm_expert_provider_error::unsupported_configuration);
    }
    virtual llm_expert_provider_result validate_slot_generation(
            uint32_t slot,
            uint64_t generation) noexcept {
        (void) slot;
        (void) generation;
        return llm_expert_provider_result::failure(llm_expert_provider_error::unsupported_configuration);
    }
    virtual llm_expert_provider_initialization_stage initialization_stage() const noexcept {
        return llm_expert_provider_initialization_stage::none;
    }
    virtual llm_expert_provider_result begin_initialization(
            llm_expert_provider_initialization_stage stage,
            bool & owner) noexcept {
        (void) stage;
        owner = false;
        return llm_expert_provider_result::success();
    }
    virtual llm_expert_provider_result complete_descriptor_discovery(
            uint64_t graph_count,
            uint64_t binding_count,
            uint64_t scheduler_reserve_calls,
            const std::vector<uint64_t> & backend_bytes_before,
            const std::vector<uint64_t> & backend_bytes_after) noexcept {
        (void) graph_count;
        (void) binding_count;
        (void) scheduler_reserve_calls;
        (void) backend_bytes_before;
        (void) backend_bytes_after;
        return llm_expert_provider_result::failure(llm_expert_provider_error::unsupported_configuration);
    }
    virtual llm_expert_provider_result record_initialization_telemetry(
            const std::vector<uint64_t> & backend_bytes_after_hierarchy,
            const std::vector<uint64_t> & backend_bytes_after_final_reserve,
            uint64_t final_bootstrap_source_bindings,
            uint64_t complete_deferred_payload_bytes) noexcept {
        (void) backend_bytes_after_hierarchy;
        (void) backend_bytes_after_final_reserve;
        (void) final_bootstrap_source_bindings;
        (void) complete_deferred_payload_bytes;
        return llm_expert_provider_result::success();
    }
    virtual llm_expert_provider_result finish_initialization(bool success) noexcept {
        (void) success;
        return llm_expert_provider_result::success();
    }
    virtual bool needs_post_reserve_initialization() const noexcept {
        return initialization_stage() != llm_expert_provider_initialization_stage::none;
    }
    virtual bool uses_hybrid_graph() const noexcept { return false; }
    virtual llm_expert_provider_result initialize_after_reserve() noexcept {
        return llm_expert_provider_result::success();
    }
    virtual llm_expert_provider_result trim() noexcept {
        return llm_expert_provider_result::success();
    }
    virtual llm_expert_provider_result surrender() noexcept {
        return llm_expert_provider_result::failure(llm_expert_provider_error::unsupported_configuration);
    }
    virtual uint64_t graph_epoch() const noexcept { return 0; }
    virtual llm_hot_cache_diagnostics hot_cache_diagnostics() const { return {}; }
    // Read-only standing-evidence seam. Production execution never calls this.
    virtual llm_expert_provider_result debug_copy_cold_bundle(
            llm_expert_key key,
            std::vector<uint8_t> & bytes) const noexcept {
        (void) key;
        bytes.clear();
        return llm_expert_provider_result::failure(llm_expert_provider_error::unsupported_configuration);
    }
    virtual llm_expert_provider_result debug_copy_hot_bundle(
            llm_expert_key key,
            std::vector<uint8_t> & bytes) const noexcept {
        (void) key;
        bytes.clear();
        return llm_expert_provider_result::failure(llm_expert_provider_error::unsupported_configuration);
    }
    virtual llm_expert_provider_result debug_set_miss_policy_for_testing(
            llama_expert_miss_policy policy) noexcept {
        (void) policy;
        return llm_expert_provider_result::failure(llm_expert_provider_error::unsupported_configuration);
    }
    virtual llm_expert_provider_result debug_set_auto_cost_model_for_testing(
            const llama_expert_auto_cost_model & cost) noexcept {
        (void) cost;
        return llm_expert_provider_result::failure(llm_expert_provider_error::unsupported_configuration);
    }
    // Internal focused-test seams. Production execution never calls these.
    virtual llm_expert_provider_result set_h2d_gate_event_for_testing(
            ggml_backend_event_t event) noexcept {
        (void) event;
        return llm_expert_provider_result::failure(llm_expert_provider_error::unsupported_configuration);
    }
    virtual llm_expert_provider_result set_phase8_test_control_for_testing(
            llm_expert_phase8_test_control * control) noexcept {
        (void) control;
        return llm_expert_provider_result::failure(llm_expert_provider_error::unsupported_configuration);
    }
    virtual llm_expert_provider_result set_phase8_closeout_witness_for_testing(
            llm_expert_phase8_closeout_witness * witness) noexcept {
        (void) witness;
        return llm_expert_provider_result::failure(llm_expert_provider_error::unsupported_configuration);
    }
    virtual bool debug_hot_mapping(
            llm_expert_key key,
            uint64_t * generation = nullptr,
            uint32_t * slot = nullptr) const noexcept {
        (void) key;
        if (generation) *generation = 0;
        if (slot) *slot = UINT32_MAX;
        return false;
    }
    virtual bool debug_cold_ready(
            llm_expert_key key,
            uint64_t * generation = nullptr,
            uint32_t * slot = nullptr) const noexcept {
        (void) key;
        if (generation) *generation = 0;
        if (slot) *slot = UINT32_MAX;
        return false;
    }

protected:
    friend class llm_expert_handle;
    virtual void release_handle(uint64_t lease_id) noexcept = 0;
};

struct llm_expert_provider_faults {
    llm_expert_provider_error initialization = llm_expert_provider_error::none;
    llm_expert_provider_error binding = llm_expert_provider_error::none;
    llm_expert_provider_error preparation = llm_expert_provider_error::none;
    size_t binding_successes_before_failure = 0;
    size_t fail_preparation_after_handles = SIZE_MAX;
    bool fail_pool_allocation = false;
    size_t fail_copy_after_tensors = SIZE_MAX;
};

struct llm_hot_cache_config {
    uint32_t capacity = 0;
    uint32_t n_expert_used = 0;
    uint32_t routed_layer_count = 0;
    uint32_t total_expert_keys = 0;
    ggml_backend_buffer_type_t target_buffer_type = nullptr;

    // Internal test seam. The model-facing path always leaves this false.
    bool allow_non_cuda_target_for_testing = false;
    uint64_t initial_slot_generation_for_testing = 0;
    bool cold_mode = false;
    uint64_t cold_cache_bytes = 0;
    uint64_t transfer_ring_bytes = 0;
    ggml_backend_dev_t target_device = nullptr;
    bool force_pageable_transfer_for_testing = false;
    llm_expert_storage * storage = nullptr;
    llm_expert_async_transport * async_transport = nullptr;
    llm_expert_scheduler * scheduler = nullptr;
    uint32_t trace_capacity = 256;
    llama_expert_miss_policy miss_policy = LLAMA_EXPERT_MISS_POLICY_PROMOTE_AND_GPU;
    bool background_promotion = false;
    bool phase10_lead_trace = false;
    llm_expert_prefetch_config_internal prefetch_config = {};
    llm_expert_prefetch_profile prefetch_profile = {};
    bool prefetch_profile_loaded = false;
    // Internal evidence seam. The model-facing path always leaves this false.
    bool phase10_serial_issue_for_testing = false;
    llama_expert_auto_cost_model auto_cost_model = {};
    uint64_t auto_cost_model_digest = 0;
    // Internal evidence seams. Model-facing construction always leaves these
    // null; focused tests may install them on a live provider.
    llm_expert_phase8_test_control * phase8_test_control = nullptr;
    llm_expert_phase8_closeout_witness * phase8_closeout_witness = nullptr;
    bool descriptor_only_source_for_testing = false;
    llm_expert_cache_policy_config_internal hot_cache_policy_config = {};
    llm_expert_cache_policy_config_internal cold_cache_policy_config = {};
    std::vector<int32_t> routed_layers;
};

struct llm_uma_cache_config {
    using is_buffer_type_fn = bool (*)(ggml_backend_buffer_type_t);
    using prepare_fn = int (*)(ggml_backend_buffer_t, size_t, size_t);
    using checksum_fn = int (*)(ggml_backend_buffer_t, size_t, size_t, uint64_t *);
    using sample_memory_fn = llm_expert_uma_result (*)(llm_expert_uma_memory_sample &);
    uint64_t pool_bytes = 0;
    uint32_t hot_capacity = 0;
    uint32_t n_expert_used = 0;
    uint32_t routed_layer_count = 0;
    uint32_t total_expert_keys = 0;
    ggml_backend_buffer_type_t buffer_type = nullptr;
    ggml_backend_dev_t target_device = nullptr;
    llm_expert_storage * storage = nullptr;
    llm_expert_scheduler * scheduler = nullptr;
    is_buffer_type_fn is_uma_buffer_type = nullptr;
    prepare_fn prefetch = nullptr;
    checksum_fn checksum = nullptr;
    sample_memory_fn sample_memory = nullptr;
    llama_expert_uma_readiness readiness = LLAMA_EXPERT_UMA_READINESS_AUTO;
    uint64_t min_system_headroom_bytes = 0;
    uint64_t min_runtime_headroom_bytes = 0;
    llm_expert_cache_policy_config_internal hot_cache_policy_config = {};
    llm_expert_cache_policy_config_internal cold_cache_policy_config = {};
    std::vector<int32_t> routed_layers;
};

std::unique_ptr<llm_expert_weight_provider> llm_create_resident_expert_weight_provider(
        llm_expert_provider_faults faults = {});

std::unique_ptr<llm_expert_weight_provider> llm_create_hot_cache_expert_weight_provider(
        llm_hot_cache_config config,
        llm_expert_provider_faults faults = {});

std::unique_ptr<llm_expert_weight_provider> llm_create_cold_cache_expert_weight_provider(
        llm_hot_cache_config config,
        llm_expert_provider_faults faults = {});

std::unique_ptr<llm_expert_weight_provider> llm_create_uma_cache_expert_weight_provider(
        llm_uma_cache_config config,
        llm_expert_provider_faults faults = {});
