#include "llama-expert-async-io.h"

#include <algorithm>
#include <cerrno>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <vector>

#if defined(__linux__)
#include <linux/io_uring.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace {

bool checked_add(uint64_t lhs, uint64_t rhs, uint64_t & result) {
    if (lhs > std::numeric_limits<uint64_t>::max() - rhs) {
        return false;
    }
    result = lhs + rhs;
    return true;
}

bool checked_multiply(uint64_t lhs, uint64_t rhs, uint64_t & result) {
    if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max()/lhs) {
        return false;
    }
    result = lhs*rhs;
    return true;
}

bool is_power_of_two(uint64_t value) {
    return value != 0 && (value & (value - 1)) == 0;
}

bool include_field(uint64_t offset, uint64_t extent, uint64_t & size) {
    uint64_t end = 0;
    if (!checked_add(offset, extent, end)) {
        return false;
    }
    size = std::max(size, end);
    return true;
}

uint32_t auto_queue_depth(uint32_t hot_capacity) {
    uint64_t target = std::max<uint64_t>(32, uint64_t(hot_capacity)*4);
    target = std::min<uint64_t>(target, 256);
    uint32_t result = 1;
    while (result < target) {
        result <<= 1;
    }
    return result;
}

} // namespace

struct llm_expert_async_transport::impl {
    struct operation_record {
        llm_expert_async_operation_identity identity;
        uint32_t generation = 0;
        bool active = false;
    };

    struct trace_record {
        uint64_t sequence = 0;
    };

    llm_expert_async_config config;
    std::vector<operation_record> operations;
    std::vector<trace_record> traces;
    mutable std::mutex mutex;
    llm_expert_async_diagnostics counters;
};

llm_expert_async_transport::llm_expert_async_transport(llm_expert_async_config config) : pimpl(std::make_unique<impl>()) {
    if (config.effective_hot_capacity == 0 || config.request_capacity == 0 || config.trace_capacity == 0 ||
        config.cold_cache_bytes == 0 ||
        (config.requested_queue_depth != 0 &&
         (config.requested_queue_depth < 8 || config.requested_queue_depth > 4096))) {
        throw std::invalid_argument("invalid expert async transport configuration");
    }
    const uint32_t sq_entries = config.requested_queue_depth == 0 ?
        auto_queue_depth(config.effective_hot_capacity) : config.requested_queue_depth;
    if (sq_entries > std::numeric_limits<uint32_t>::max()/2) {
        throw std::invalid_argument("expert async completion queue overflow");
    }
    const uint32_t cq_entries = sq_entries*2;
    uint64_t staging_ceiling = config.requested_staging_bytes;
    if (staging_ceiling == 0 && config.staging_required && config.maximum_aligned_read_bytes != 0) {
        uint64_t minimum = 0;
        if (checked_multiply(config.maximum_aligned_read_bytes, 2, minimum)) {
            const uint64_t maximum = std::min<uint64_t>(512ULL*1024*1024, config.cold_cache_bytes/4);
            if (minimum <= maximum) {
                staging_ceiling = minimum;
            }
        }
    }

    pimpl->config = config;
    pimpl->operations.resize(cq_entries);
    pimpl->traces.resize(config.trace_capacity);
    for (auto & operation : pimpl->operations) {
        operation.generation = config.initial_operation_generation_for_testing;
    }
    pimpl->counters.requested_sq_entries = sq_entries;
    pimpl->counters.requested_cq_entries = cq_entries;
    pimpl->counters.operation_capacity = cq_entries;
    pimpl->counters.trace_capacity = config.trace_capacity;
    pimpl->counters.staging_ceiling_bytes = staging_ceiling;
    pimpl->counters.administration_bytes = sizeof(*pimpl) +
        pimpl->operations.capacity()*sizeof(impl::operation_record) +
        pimpl->traces.capacity()*sizeof(impl::trace_record);
#if defined(__linux__)
    static_assert(sizeof(io_uring_sqe) == 64, "unexpected io_uring SQE size");
    static_assert(sizeof(io_uring_cqe) == 16, "unexpected io_uring CQE size");
    pimpl->counters.linux_uapi = true;
#endif
}

llm_expert_async_transport::~llm_expert_async_transport() = default;

bool llm_expert_async_transport::validate_ring_layout(
        const llm_expert_async_ring_layout & layout,
        uint64_t page_size,
        llm_expert_async_mapping_sizes & sizes) noexcept {
    sizes = {};
    if (!is_power_of_two(page_size) || !is_power_of_two(layout.sq_entries) ||
        !is_power_of_two(layout.cq_entries) || layout.sq_entries < 8 ||
        layout.sq_entries > 4096 || layout.cq_entries < layout.sq_entries) {
        return false;
    }
    uint64_t sq_array_bytes = 0;
    uint64_t cqes_bytes = 0;
    if (!checked_multiply(layout.sq_entries, sizeof(uint32_t), sq_array_bytes) ||
        !checked_multiply(layout.cq_entries, 16, cqes_bytes) ||
        !checked_multiply(layout.sq_entries, 64, sizes.sqes_bytes)) {
        return false;
    }
    for (uint32_t offset : { layout.sq_head, layout.sq_tail, layout.sq_ring_mask,
            layout.sq_ring_entries, layout.sq_flags, layout.sq_dropped }) {
        if (!include_field(offset, sizeof(uint32_t), sizes.sq_ring_bytes)) {
            return false;
        }
    }
    if (!include_field(layout.sq_array, sq_array_bytes, sizes.sq_ring_bytes)) {
        return false;
    }
    for (uint32_t offset : { layout.cq_head, layout.cq_tail, layout.cq_ring_mask,
            layout.cq_ring_entries, layout.cq_overflow }) {
        if (!include_field(offset, sizeof(uint32_t), sizes.cq_ring_bytes)) {
            return false;
        }
    }
    if (!include_field(layout.cqes, cqes_bytes, sizes.cq_ring_bytes)) {
        return false;
    }
    uint64_t maximum_sq_ring = 0;
    uint64_t maximum_cq_ring = 0;
    if (!checked_add(page_size, sq_array_bytes, maximum_sq_ring) ||
        !checked_add(page_size, cqes_bytes, maximum_cq_ring)) {
        return false;
    }
    return sizes.sq_ring_bytes <= maximum_sq_ring && sizes.cq_ring_bytes <= maximum_cq_ring;
}

llm_expert_async_ring_probe llm_expert_async_transport::probe_ring_for_testing(uint32_t queue_depth) noexcept {
    llm_expert_async_ring_probe result;
    if (queue_depth < 8 || queue_depth > 4096) {
        result.native_error = EINVAL;
        return result;
    }
#if defined(__linux__)
    io_uring_params params{};
    params.flags = IORING_SETUP_CQSIZE;
    params.cq_entries = queue_depth*2;
    const int fd = int(syscall(__NR_io_uring_setup, queue_depth, &params));
    if (fd < 0) {
        result.native_error = errno;
        return result;
    }
    result.supported = true;
    result.sq_entries = params.sq_entries;
    result.cq_entries = params.cq_entries;
    const llm_expert_async_ring_layout layout = {
        params.sq_entries,
        params.cq_entries,
        params.sq_off.head,
        params.sq_off.tail,
        params.sq_off.ring_mask,
        params.sq_off.ring_entries,
        params.sq_off.flags,
        params.sq_off.dropped,
        params.sq_off.array,
        params.cq_off.head,
        params.cq_off.tail,
        params.cq_off.ring_mask,
        params.cq_off.ring_entries,
        params.cq_off.overflow,
        params.cq_off.cqes,
    };
    llm_expert_async_mapping_sizes sizes;
    const long page_size = sysconf(_SC_PAGESIZE);
    result.layout_valid = page_size > 0 && validate_ring_layout(layout, uint64_t(page_size), sizes);
    if (close(fd) != 0 && result.native_error == 0) {
        result.native_error = errno;
    }
    return result;
#else
    result.native_error = ENOSYS;
    return result;
#endif
}

uint64_t llm_expert_async_transport::encode_user_data(uint32_t operation_slot, uint32_t operation_generation) noexcept {
    if (operation_generation == 0) {
        return 0;
    }
    return uint64_t(operation_generation) << 32 | operation_slot;
}

bool llm_expert_async_transport::decode_user_data(
        uint64_t user_data,
        uint32_t & operation_slot,
        uint32_t & operation_generation) noexcept {
    operation_slot = uint32_t(user_data);
    operation_generation = uint32_t(user_data >> 32);
    return operation_generation != 0;
}

llm_expert_async_operation llm_expert_async_transport::reserve_operation(
        const llm_expert_async_operation_identity & identity) noexcept {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    if (pimpl->counters.admission_closed) {
        return { llm_expert_async_result::closed, 0 };
    }
    if (!identity.request.valid() || identity.transport_epoch != pimpl->counters.transport_epoch ||
        !identity.key.is_valid(std::numeric_limits<int32_t>::max(), std::numeric_limits<int32_t>::max())) {
        return { llm_expert_async_result::invalid, 0 };
    }
    for (uint32_t slot = 0; slot < pimpl->operations.size(); ++slot) {
        impl::operation_record & operation = pimpl->operations[slot];
        if (operation.active) {
            continue;
        }
        if (operation.generation == std::numeric_limits<uint32_t>::max()) {
            return { llm_expert_async_result::generation_exhausted, 0 };
        }
        operation.generation++;
        operation.identity = identity;
        operation.active = true;
        pimpl->counters.operations_reserved++;
        pimpl->counters.active_operations++;
        pimpl->counters.peak_active_operations = std::max(
            pimpl->counters.peak_active_operations, pimpl->counters.active_operations);
        return { llm_expert_async_result::ready, encode_user_data(slot, operation.generation) };
    }
    return { llm_expert_async_result::busy, 0 };
}

llm_expert_async_result llm_expert_async_transport::consume_completion_for_testing(
        uint64_t user_data,
        llm_expert_async_operation_identity & identity) noexcept {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    uint32_t slot = 0;
    uint32_t generation = 0;
    if (!decode_user_data(user_data, slot, generation) || slot >= pimpl->operations.size()) {
        pimpl->counters.stale_completions++;
        return llm_expert_async_result::stale_generation;
    }
    impl::operation_record & operation = pimpl->operations[slot];
    if (!operation.active || operation.generation != generation) {
        pimpl->counters.stale_completions++;
        return llm_expert_async_result::stale_generation;
    }
    identity = operation.identity;
    operation.active = false;
    pimpl->counters.active_operations--;
    pimpl->counters.completions_consumed++;
    return llm_expert_async_result::ready;
}

void llm_expert_async_transport::record_trace_for_testing() noexcept {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    if (pimpl->counters.trace_records == pimpl->traces.size()) {
        pimpl->counters.trace_records_dropped++;
        return;
    }
    pimpl->traces[pimpl->counters.trace_records].sequence = pimpl->counters.trace_records + 1;
    pimpl->counters.trace_records++;
}

bool llm_expert_async_transport::shutdown() noexcept {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    pimpl->counters.admission_closed = true;
    pimpl->counters.transport_epoch++;
    return pimpl->counters.active_operations == 0;
}

llm_expert_async_diagnostics llm_expert_async_transport::diagnostics() const noexcept {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    return pimpl->counters;
}
