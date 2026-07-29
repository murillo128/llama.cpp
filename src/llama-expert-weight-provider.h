#pragma once

#include "ggml-backend.h"
#include "ggml.h"

#include <cstddef>
#include <cstdint>
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
};

class llm_expert_weight_provider;

class llm_expert_handle {
public:
    llm_expert_handle() = default;
    llm_expert_handle(llm_expert_weight_provider * provider, uint64_t lease_id);
    ~llm_expert_handle();

    llm_expert_handle(const llm_expert_handle &) = delete;
    llm_expert_handle & operator=(const llm_expert_handle &) = delete;

    llm_expert_handle(llm_expert_handle && other) noexcept;
    llm_expert_handle & operator=(llm_expert_handle && other) noexcept;

    bool is_valid() const;
    void reset();

private:
    llm_expert_weight_provider * provider = nullptr;
    uint64_t lease_id = 0;
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

    virtual llm_expert_provider_result prepare(
            const std::vector<llm_expert_graph_binding> & bindings,
            llm_expert_execution_plan & plan) noexcept = 0;

    virtual llm_expert_provider_stats get_stats() const noexcept = 0;

protected:
    friend class llm_expert_handle;
    virtual void release_handle(uint64_t lease_id) noexcept = 0;
};

class llm_resident_expert_weight_provider;
