#pragma once

#include "ggml-backend.h"
#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

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
            int32_t * execution_ids) noexcept {
        (void) binding;
        (void) logical_ids;
        (void) logical_id_count;
        (void) execution_ids;
        return llm_expert_provider_result::failure(llm_expert_provider_error::unsupported_configuration);
    }
    virtual llm_expert_provider_result remap_checkpoint_tensor(
            const llm_expert_graph_binding & binding) noexcept {
        (void) binding;
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
};

std::unique_ptr<llm_expert_weight_provider> llm_create_resident_expert_weight_provider(
        llm_expert_provider_faults faults = {});

std::unique_ptr<llm_expert_weight_provider> llm_create_hot_cache_expert_weight_provider(
        llm_hot_cache_config config,
        llm_expert_provider_faults faults = {});
