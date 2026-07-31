#pragma once

#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"

#include <cstddef>
#include <cstdint>
#include <memory>
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

struct llm_expert_flight_id {
    uint64_t transport_epoch = 0;
    uint32_t request_slot = UINT32_MAX;
    uint64_t request_generation = 0;
    llm_expert_key key = { -1, -1 };

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

struct llm_hot_cache_diagnostics {
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
    uint32_t requested_capacity = 0;
    uint32_t effective_capacity = 0;
    uint64_t pool_bytes = 0;
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
        uint64_t generation = 0;
        uint64_t last_use = 0;
        uint32_t refcount = 0;
        uint32_t cold_slot = 0;
        uint64_t cold_generation = 0;
        bool has_cold_backing = false;
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
};

struct llm_expert_graph_diagnostics {
    uint64_t operation_hash = 0;
    int32_t node_count = 0;
    int32_t binding_count = 0;
    int32_t binding_capacity = 0;
    uint64_t inflight_handles = 0;
    int32_t graphs_reused = 0;
};

class llm_expert_weight_provider;

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
            llm_expert_execution_plan & plan) noexcept = 0;

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
    virtual bool needs_post_reserve_initialization() const noexcept { return false; }
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
    // Internal focused-test seams. Production execution never calls these.
    virtual llm_expert_provider_result set_h2d_gate_event_for_testing(
            ggml_backend_event_t event) noexcept {
        (void) event;
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
    llama_expert_auto_cost_model auto_cost_model = {};
    uint64_t auto_cost_model_digest = 0;
};

std::unique_ptr<llm_expert_weight_provider> llm_create_resident_expert_weight_provider(
        llm_expert_provider_faults faults = {});

std::unique_ptr<llm_expert_weight_provider> llm_create_hot_cache_expert_weight_provider(
        llm_hot_cache_config config,
        llm_expert_provider_faults faults = {});

std::unique_ptr<llm_expert_weight_provider> llm_create_cold_cache_expert_weight_provider(
        llm_hot_cache_config config,
        llm_expert_provider_faults faults = {});
