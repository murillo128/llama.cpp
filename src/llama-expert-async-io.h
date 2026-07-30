#pragma once

#include "llama-expert-scheduler.h"
#include "llama-expert-storage.h"

#include <cstdint>
#include <memory>

struct llm_expert_async_read_override {
    virtual ~llm_expert_async_read_override() = default;
    virtual int64_t read_at(
            intptr_t native_handle,
            void * destination,
            size_t byte_count,
            uint64_t file_offset,
            int & native_error) noexcept = 0;
};

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
    llm_expert_async_read_override * read_override_for_testing = nullptr;
    bool direct_io_requested = false;
    uint64_t maximum_direct_alignment = 0;
    bool force_file_registration_failure_for_testing = false;
    bool force_buffer_registration_failure_for_testing = false;
    uint32_t partial_submit_count_before_error_for_testing = 0;
    int submit_error_for_testing = 0;
    uint32_t source_file_capacity = 0;
    bool pause_after_ring_submit_for_testing = false;
    bool inject_first_read_cqe_for_testing = false;
    int32_t first_read_cqe_result_for_testing = 0;
    uint32_t hide_cqes_after_cancel_polls_for_testing = 0;
};

enum class llm_expert_async_fallback_reason : uint64_t {
    none                = 0,
    ring_setup          = 1ULL << 0,
    ring_capability     = 1ULL << 1,
    ring_runtime        = 1ULL << 2,
    file_registration   = 1ULL << 3,
    buffer_registration = 1ULL << 4,
    direct_staging      = 1ULL << 5,
    direct_alignment    = 1ULL << 6,
    direct_eof          = 1ULL << 7,
    direct_capability   = 1ULL << 8,
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
    uint64_t fallback_reason_mask = 0;
    uint64_t fallback_diagnostics_emitted = 0;
    uint64_t operations_reserved = 0;
    uint64_t completions_consumed = 0;
    uint64_t stale_completions = 0;
    uint64_t trace_records = 0;
    uint64_t trace_records_dropped = 0;
    uint64_t read_requests_submitted = 0;
    uint64_t read_requests_completed = 0;
    uint64_t read_requests_cancelled = 0;
    uint64_t read_operations_completed = 0;
    uint64_t read_bytes_completed = 0;
    uint64_t synchronous_fallback_operations = 0;
    uint64_t interrupted_reads_retried = 0;
    uint64_t would_block_reads_retried = 0;
    uint64_t short_positive_reads = 0;
    uint64_t direct_read_operations = 0;
    uint64_t direct_useful_bytes = 0;
    uint64_t direct_aligned_bytes = 0;
    uint64_t direct_scatter_bytes = 0;
    uint64_t buffered_fallback_operations = 0;
    uint64_t buffered_fallback_bytes = 0;
    uint64_t direct_capability_retries = 0;
    uint64_t ring_submissions = 0;
    uint64_t ring_completions = 0;
    uint32_t peak_sq_occupancy = 0;
    uint32_t peak_cq_occupancy = 0;
    uint64_t ring_cancel_submissions = 0;
    uint64_t ring_cancel_completions = 0;
    uint64_t cq_empty_waits = 0;
    uint64_t cq_empty_waits_after_cancel = 0;
    uint32_t registered_file_count = 0;
    int file_registration_error = 0;
    uint32_t registered_buffer_count = 0;
    uint64_t registered_buffer_bytes = 0;
    int buffer_registration_error = 0;
    int direct_staging_error = 0;
    uint32_t active_read_requests = 0;
    uint32_t peak_active_read_requests = 0;
    bool linux_uapi = false;
    bool io_uring_enabled = false;
    int io_uring_setup_error = 0;
    int io_uring_probe_error = 0;
    int io_uring_runtime_error = 0;
    bool opcode_read = false;
    bool opcode_readv = false;
    bool opcode_async_cancel = false;
    bool opcode_read_fixed = false;
    uint32_t actual_sq_entries = 0;
    uint32_t actual_cq_entries = 0;
    bool worker_started = false;
    bool admission_closed = false;
};

struct llm_expert_async_read_completion {
    llm_expert_async_result result = llm_expert_async_result::invalid;
    int native_error = 0;
    uint64_t bytes_completed = 0;
    uint64_t digest = 1469598103934665603ULL;
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
    llm_expert_async_result submit_read_plan(
            const llm_expert_async_operation_identity & identity,
            const llm_expert_storage_read_operation * operations,
            size_t operation_count) noexcept;
    llm_expert_async_result register_files(const intptr_t * handles, size_t handle_count) noexcept;
    llm_expert_async_result wait_read(
            llm_expert_request_handle request,
            llm_expert_async_read_completion & completion,
            bool (*abort_callback)(void *) = nullptr,
            void * abort_callback_data = nullptr) noexcept;
    llm_expert_async_result cancel_read(llm_expert_request_handle request) noexcept;
    llm_expert_async_result release_read(llm_expert_request_handle request) noexcept;
    bool wait_until_ring_submitted_for_testing() noexcept;
    bool shutdown() noexcept;
    llm_expert_async_diagnostics diagnostics() const noexcept;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
