#pragma once

#include "llama-expert-scheduler.h"

#include <cstdint>
#include <memory>

struct llm_expert_async_config {
    uint32_t requested_queue_depth = 0;
    uint32_t effective_hot_capacity = 0;
    uint32_t request_capacity = 0;
    uint32_t trace_capacity = 0;
    uint64_t cold_cache_bytes = 0;
    uint64_t requested_staging_bytes = 0;
    uint64_t maximum_aligned_read_bytes = 0;
    bool staging_required = false;
    uint32_t initial_operation_generation_for_testing = 0;
};

struct llm_expert_async_ring_layout {
    uint32_t sq_entries = 0;
    uint32_t cq_entries = 0;
    uint32_t sq_head = 0;
    uint32_t sq_tail = 0;
    uint32_t sq_ring_mask = 0;
    uint32_t sq_ring_entries = 0;
    uint32_t sq_flags = 0;
    uint32_t sq_dropped = 0;
    uint32_t sq_array = 0;
    uint32_t cq_head = 0;
    uint32_t cq_tail = 0;
    uint32_t cq_ring_mask = 0;
    uint32_t cq_ring_entries = 0;
    uint32_t cq_overflow = 0;
    uint32_t cqes = 0;
};

struct llm_expert_async_mapping_sizes {
    uint64_t sq_ring_bytes = 0;
    uint64_t cq_ring_bytes = 0;
    uint64_t sqes_bytes = 0;
};

struct llm_expert_async_ring_probe {
    uint32_t sq_entries = 0;
    uint32_t cq_entries = 0;
    int native_error = 0;
    bool supported = false;
    bool layout_valid = false;
};

struct llm_expert_async_operation_identity {
    uint64_t transport_epoch = 0;
    llm_expert_request_handle request;
    uint32_t request_operation_index = 0;
    llm_expert_key key = { -1, -1 };
    llm_expert_readiness readiness = llm_expert_readiness::host_ready;
    llm_expert_priority priority = llm_expert_priority::prefetch_speculative;
};

enum class llm_expert_async_result {
    ready,
    busy,
    closed,
    invalid,
    stale_generation,
    generation_exhausted,
};

struct llm_expert_async_operation {
    llm_expert_async_result result = llm_expert_async_result::invalid;
    uint64_t user_data = 0;
};

struct llm_expert_async_diagnostics {
    uint32_t requested_sq_entries = 0;
    uint32_t requested_cq_entries = 0;
    uint32_t operation_capacity = 0;
    uint32_t trace_capacity = 0;
    uint32_t active_operations = 0;
    uint32_t peak_active_operations = 0;
    uint64_t staging_ceiling_bytes = 0;
    uint64_t administration_bytes = 0;
    uint64_t transport_epoch = 1;
    uint64_t operations_reserved = 0;
    uint64_t completions_consumed = 0;
    uint64_t stale_completions = 0;
    uint64_t trace_records = 0;
    uint64_t trace_records_dropped = 0;
    bool linux_uapi = false;
    bool admission_closed = false;
};

class llm_expert_async_transport {
public:
    explicit llm_expert_async_transport(llm_expert_async_config config);
    ~llm_expert_async_transport();

    llm_expert_async_transport(const llm_expert_async_transport &) = delete;
    llm_expert_async_transport & operator=(const llm_expert_async_transport &) = delete;

    static bool validate_ring_layout(
            const llm_expert_async_ring_layout & layout,
            uint64_t page_size,
            llm_expert_async_mapping_sizes & sizes) noexcept;
    static llm_expert_async_ring_probe probe_ring_for_testing(uint32_t queue_depth) noexcept;
    static uint64_t encode_user_data(uint32_t operation_slot, uint32_t operation_generation) noexcept;
    static bool decode_user_data(uint64_t user_data, uint32_t & operation_slot, uint32_t & operation_generation) noexcept;

    llm_expert_async_operation reserve_operation(const llm_expert_async_operation_identity & identity) noexcept;
    llm_expert_async_result consume_completion_for_testing(
            uint64_t user_data,
            llm_expert_async_operation_identity & identity) noexcept;
    void record_trace_for_testing() noexcept;
    bool shutdown() noexcept;
    llm_expert_async_diagnostics diagnostics() const noexcept;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
