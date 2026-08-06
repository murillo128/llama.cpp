#include "llama-expert-async-io.h"
#include "llama-perfetto-trace.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <linux/io_uring.h>
#include <sys/syscall.h>
#include <sys/mman.h>
#include <sys/uio.h>
#include <unistd.h>

#ifndef IORING_ASYNC_CANCEL_USERDATA
#define IORING_ASYNC_CANCEL_USERDATA 0
#endif
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

bool recoverable_registration_error(int error) {
    return error == EINVAL || error == ENOSYS || error == EOPNOTSUPP || error == ENOMEM ||
        error == EMFILE || error == ENFILE || error == EPERM;
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

#if defined(__linux__)
class io_uring_owner {
public:
    ~io_uring_owner() { close_ring(); }

    bool open_ring(uint32_t entries, int & native_error) {
        io_uring_params requested{};
        requested.flags = IORING_SETUP_CQSIZE;
        requested.cq_entries = entries*2;
        fd = int(syscall(__NR_io_uring_setup, entries, &requested));
        if (fd < 0 && errno == EINVAL) {
            requested = {};
            fd = int(syscall(__NR_io_uring_setup, entries, &requested));
        }
        if (fd < 0) {
            native_error = errno;
            return false;
        }
        params = requested;
        const llm_expert_async_ring_layout layout = {
            params.sq_entries, params.cq_entries,
            params.sq_off.head, params.sq_off.tail, params.sq_off.ring_mask, params.sq_off.ring_entries,
            params.sq_off.flags, params.sq_off.dropped, params.sq_off.array,
            params.cq_off.head, params.cq_off.tail, params.cq_off.ring_mask, params.cq_off.ring_entries,
            params.cq_off.overflow, params.cq_off.cqes,
        };
        llm_expert_async_mapping_sizes sizes;
        const long page_size = sysconf(_SC_PAGESIZE);
        if (page_size <= 0 || !llm_expert_async_transport::validate_ring_layout(
                layout, uint64_t(page_size), sizes, (params.features & IORING_FEAT_SINGLE_MMAP) != 0)) {
            native_error = EINVAL;
            close_ring();
            return false;
        }
        sq_ring_size = size_t(sizes.sq_ring_bytes);
        cq_ring_size = size_t(sizes.cq_ring_bytes);
        sqes_size = size_t(sizes.sqes_bytes);
        if (params.features & IORING_FEAT_SINGLE_MMAP) {
            sq_ring_size = cq_ring_size = std::max(sq_ring_size, cq_ring_size);
            sq_ring = mmap(nullptr, sq_ring_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE,
                fd, IORING_OFF_SQ_RING);
            cq_ring = sq_ring;
        } else {
            sq_ring = mmap(nullptr, sq_ring_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE,
                fd, IORING_OFF_SQ_RING);
            if (sq_ring != MAP_FAILED) {
                cq_ring = mmap(nullptr, cq_ring_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE,
                    fd, IORING_OFF_CQ_RING);
            }
        }
        if (sq_ring == MAP_FAILED || cq_ring == MAP_FAILED) {
            native_error = errno;
            close_ring();
            return false;
        }
        sqes = static_cast<io_uring_sqe *>(mmap(nullptr, sqes_size, PROT_READ | PROT_WRITE,
            MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_SQES));
        if (sqes == MAP_FAILED) {
            native_error = errno;
            close_ring();
            return false;
        }
        sq_head = pointer<uint32_t>(sq_ring, params.sq_off.head);
        sq_tail = pointer<uint32_t>(sq_ring, params.sq_off.tail);
        sq_mask = pointer<uint32_t>(sq_ring, params.sq_off.ring_mask);
        sq_entries = pointer<uint32_t>(sq_ring, params.sq_off.ring_entries);
        sq_array = pointer<uint32_t>(sq_ring, params.sq_off.array);
        cq_head = pointer<uint32_t>(cq_ring, params.cq_off.head);
        cq_tail = pointer<uint32_t>(cq_ring, params.cq_off.tail);
        cq_mask = pointer<uint32_t>(cq_ring, params.cq_off.ring_mask);
        cq_entries = pointer<uint32_t>(cq_ring, params.cq_off.ring_entries);
        cqes = pointer<io_uring_cqe>(cq_ring, params.cq_off.cqes);
        return true;
    }

    void close_ring() {
        if (sqes != MAP_FAILED) munmap(sqes, sqes_size);
        if (cq_ring != MAP_FAILED && cq_ring != sq_ring) munmap(cq_ring, cq_ring_size);
        if (sq_ring != MAP_FAILED) munmap(sq_ring, sq_ring_size);
        if (fd >= 0) close(fd);
        fd = -1;
        sq_ring = cq_ring = MAP_FAILED;
        sqes = static_cast<io_uring_sqe *>(MAP_FAILED);
        pending = 0;
        files_registered = false;
        buffer_registered = false;
    }

    io_uring_sqe * acquire_sqe() {
        const uint32_t head = __atomic_load_n(sq_head, __ATOMIC_ACQUIRE);
        const uint32_t tail = __atomic_load_n(sq_tail, __ATOMIC_RELAXED);
        if (tail - head >= *sq_entries) return nullptr;
        io_uring_sqe * sqe = &sqes[tail & *sq_mask];
        std::memset(sqe, 0, sizeof(*sqe));
        sq_array[tail & *sq_mask] = tail & *sq_mask;
        __atomic_store_n(sq_tail, tail + 1, __ATOMIC_RELEASE);
        pending++;
        return sqe;
    }

    int submit(int & native_error) {
        if (pending == 0) return 0;
        if (injected_submit_error != 0 && injected_submissions_remaining == 0) {
            native_error = injected_submit_error;
            return -1;
        }
        const uint32_t requested = injected_submit_error == 0 ? pending :
            std::min(pending, injected_submissions_remaining);
        int submitted = -1;
        for (uint32_t attempt = 0; attempt < 64; ++attempt) {
            submitted = int(syscall(__NR_io_uring_enter, fd, requested, 0, 0, nullptr, 0));
            if (submitted >= 0 || (errno != EINTR && errno != EAGAIN)) break;
            if (errno == EAGAIN) std::this_thread::yield();
        }
        if (submitted < 0) {
            native_error = errno;
            return -1;
        }
        pending -= uint32_t(submitted);
        if (injected_submit_error != 0) injected_submissions_remaining -= uint32_t(submitted);
        return submitted;
    }

    bool try_cqe(io_uring_cqe & result) {
        const uint32_t head = __atomic_load_n(cq_head, __ATOMIC_RELAXED);
        const uint32_t tail = __atomic_load_n(cq_tail, __ATOMIC_ACQUIRE);
        if (head == tail) return false;
        result = cqes[head & *cq_mask];
        __atomic_store_n(cq_head, head + 1, __ATOMIC_RELEASE);
        return true;
    }

    bool register_files(const int * handles, uint32_t count, int & native_error) {
        if (syscall(__NR_io_uring_register, fd, IORING_REGISTER_FILES, handles, count) < 0) {
            native_error = errno;
            return false;
        }
        files_registered = true;
        return true;
    }

    bool register_buffer(void * address, size_t size, int & native_error) {
        iovec buffer = { address, size };
        if (syscall(__NR_io_uring_register, fd, IORING_REGISTER_BUFFERS, &buffer, 1) < 0) {
            native_error = errno;
            return false;
        }
        buffer_registered = true;
        return true;
    }

    bool probe_opcodes(bool & read, bool & readv, bool & async_cancel,
            bool & read_fixed, int & native_error) {
        constexpr uint32_t operation_count = 256;
        std::vector<uint8_t> storage(sizeof(io_uring_probe) +
            operation_count*sizeof(io_uring_probe_op));
        auto * probe = reinterpret_cast<io_uring_probe *>(storage.data());
        if (syscall(__NR_io_uring_register, fd, IORING_REGISTER_PROBE, probe, operation_count) < 0) {
            native_error = errno;
            return false;
        }
        for (uint32_t index = 0; index < probe->ops_len; ++index) {
            if ((probe->ops[index].flags & IO_URING_OP_SUPPORTED) == 0) continue;
            read |= probe->ops[index].op == IORING_OP_READ;
            readv |= probe->ops[index].op == IORING_OP_READV;
            async_cancel |= probe->ops[index].op == IORING_OP_ASYNC_CANCEL;
            read_fixed |= probe->ops[index].op == IORING_OP_READ_FIXED;
        }
        return true;
    }

    uint32_t actual_sq_entries() const { return params.sq_entries; }
    uint32_t actual_cq_entries() const { return params.cq_entries; }
    uint32_t completion_occupancy() const {
        const uint32_t head = __atomic_load_n(cq_head, __ATOMIC_RELAXED);
        const uint32_t tail = __atomic_load_n(cq_tail, __ATOMIC_ACQUIRE);
        return tail - head;
    }
    void inject_submit_error_after(uint32_t count, int error) {
        injected_submissions_remaining = count;
        injected_submit_error = error;
    }

private:
    template<class T> static T * pointer(void * base, uint32_t offset) {
        return reinterpret_cast<T *>(static_cast<uint8_t *>(base) + offset);
    }

    int fd = -1;
    io_uring_params params{};
    void * sq_ring = MAP_FAILED;
    void * cq_ring = MAP_FAILED;
    io_uring_sqe * sqes = static_cast<io_uring_sqe *>(MAP_FAILED);
    size_t sq_ring_size = 0;
    size_t cq_ring_size = 0;
    size_t sqes_size = 0;
    uint32_t * sq_head = nullptr;
    uint32_t * sq_tail = nullptr;
    uint32_t * sq_mask = nullptr;
    uint32_t * sq_entries = nullptr;
    uint32_t * sq_array = nullptr;
    uint32_t * cq_head = nullptr;
    uint32_t * cq_tail = nullptr;
    uint32_t * cq_mask = nullptr;
    uint32_t * cq_entries = nullptr;
    io_uring_cqe * cqes = nullptr;
    uint32_t pending = 0;
    bool files_registered = false;
    bool buffer_registered = false;
    uint32_t injected_submissions_remaining = 0;
    int injected_submit_error = 0;
};
#endif

} // namespace

struct llm_expert_async_transport::impl {
    struct operation_record {
        llm_expert_async_operation_identity identity;
        llm_expert_storage_read_operation read;
        uint32_t generation = 0;
        bool active = false;
        bool read_operation = false;
        bool ring_completed = false;
        bool cancel_submitted = false;
        uint64_t submit_us = 0;
        uint64_t complete_us = 0;
        uint64_t completed_bytes = 0;
#if defined(__linux__)
        std::array<iovec, 12> iovecs;
#endif
    };

    enum class read_state { free, queued, running, complete };

    struct read_request_record {
        llm_expert_request_handle handle;
        uint64_t ordinal = 0;
        read_state state = read_state::free;
        bool cancel_requested = false;
        uint32_t operations_remaining = 0;
        uint64_t queued_us = 0;
        uint64_t started_us = 0;
        llm_expert_async_read_completion completion;
    };

    struct trace_record {
        uint64_t sequence = 0;
        llm_expert_async_read_interval read;
    };

    llm_expert_async_config config;
    std::vector<operation_record> operations;
    std::vector<read_request_record> read_requests;
    std::vector<trace_record> traces;
    std::vector<uint32_t> batch_slots;
    std::vector<llm_expert_request_handle> group_handles;
    mutable std::mutex mutex;
    std::condition_variable condition;
    std::vector<std::thread> workers;
    llm_expert_async_diagnostics counters;
    uint64_t next_read_ordinal = 1;
    bool worker_stop = false;
    bool deferred_batch_open = false;
    std::vector<int> registered_files;
    uint32_t registered_file_count = 0;
    void * staging = nullptr;
    uint64_t staging_bytes = 0;
    uint64_t staging_alignment = 0;
    std::vector<intptr_t> direct_disabled_handles;
    uint32_t direct_disabled_handle_count = 0;
    bool staging_registered = false;
    bool ring_submit_paused_for_testing = false;
    bool read_cqe_injected_for_testing = false;
    uint32_t hidden_cqe_polls_for_testing = 0;
#if defined(__linux__)
    io_uring_owner ring;
#endif

    ~impl() { std::free(staging); }

    void record_fallback(llm_expert_async_fallback_reason reason, int native_error, const char * detail) {
        const uint64_t bit = static_cast<uint64_t>(reason);
        const bool first = (counters.fallback_reason_mask & bit) == 0;
        counters.fallback_reason_mask |= bit;
        if (first) {
            counters.fallback_diagnostics_emitted++;
            std::fprintf(stderr, "warning: expert async fallback reason=%s error=%d\n", detail, native_error);
        }
    }

    read_request_record * find_read(llm_expert_request_handle handle) {
        if (!handle.valid() || handle.slot >= read_requests.size()) return nullptr;
        auto & request = read_requests[handle.slot];
        return request.state != read_state::free && request.handle.generation == handle.generation ? &request : nullptr;
    }

    void mark_request_running_locked(read_request_record & request) {
        request.state = read_state::running;
        request.started_us = uint64_t(ggml_time_us());
        const uint64_t queue_wait_us = request.queued_us != 0 && request.started_us >= request.queued_us ?
            request.started_us - request.queued_us : 0;
        LLM_EXPERT_TRACE_INSTANT("k3.storage", "request_start", "request_slot", request.handle.slot,
            "request_generation", request.handle.generation, "request_ordinal", request.ordinal,
            "queue_wait_us", queue_wait_us);
        if (request.queued_us != 0 && request.started_us >= request.queued_us) {
            counters.read_queue_wait_samples++;
            counters.read_queue_wait_us += queue_wait_us;
            counters.read_queue_wait_max_us = std::max(counters.read_queue_wait_max_us, queue_wait_us);
        }
    }

    bool direct_is_disabled(intptr_t handle) const {
        for (uint32_t index = 0; index < direct_disabled_handle_count; ++index) {
            if (direct_disabled_handles[index] == handle) return true;
        }
        return false;
    }

    bool disable_direct(intptr_t handle) {
        if (handle < 0 || direct_is_disabled(handle)) return handle >= 0;
        if (direct_disabled_handle_count == direct_disabled_handles.size()) return false;
        direct_disabled_handles[direct_disabled_handle_count++] = handle;
        return true;
    }

    bool prepare_direct(const operation_record & operation, uint64_t & aligned_offset,
            uint64_t & aligned_bytes) const {
        aligned_offset = 0;
        aligned_bytes = 0;
        if (!config.direct_io_requested || operation.read.direct_native_handle < 0 ||
            direct_is_disabled(operation.read.direct_native_handle) ||
            operation.read.direct_alignment == 0 || operation.read.direct_alignment > staging_alignment ||
            staging == nullptr || !is_power_of_two(operation.read.direct_alignment)) return false;
        uint64_t useful_end = 0;
        uint64_t rounded_end = 0;
        aligned_offset = operation.read.file_offset & ~(operation.read.direct_alignment - 1);
        if (!checked_add(operation.read.file_offset, operation.read.byte_count, useful_end) ||
            !checked_add(useful_end, operation.read.direct_alignment - 1, rounded_end)) return false;
        rounded_end &= ~(operation.read.direct_alignment - 1);
        if (rounded_end < aligned_offset || rounded_end > operation.read.source_size) return false;
        aligned_bytes = rounded_end - aligned_offset;
        return aligned_bytes != 0 && aligned_bytes <= staging_bytes && aligned_bytes <= UINT32_MAX;
    }

#if defined(__linux__)
    bool fill_read_sqe(uint32_t operation_slot, io_uring_sqe & sqe) {
        auto & operation = operations[operation_slot];
        uint64_t direct_offset = 0;
        uint64_t direct_bytes = 0;
        const bool direct = prepare_direct(operation, direct_offset, direct_bytes);
        if (direct) {
            sqe.opcode = staging_registered ? IORING_OP_READ_FIXED : IORING_OP_READ;
            sqe.addr = uint64_t(staging);
            sqe.len = uint32_t(direct_bytes);
            sqe.off = direct_offset;
            sqe.fd = int(operation.read.direct_native_handle);
            if (staging_registered) sqe.buf_index = 0;
        } else if (operation.read.segment_count == 1) {
            const auto & segment = operation.read.segments[0];
            sqe.opcode = IORING_OP_READ;
            sqe.addr = uint64_t(segment.data);
            sqe.len = uint32_t(segment.byte_count);
            sqe.off = segment.file_offset;
            sqe.fd = int(operation.read.native_handle);
        } else {
            for (uint8_t index = 0; index < operation.read.segment_count; ++index) {
                operation.iovecs[index] = {
                    operation.read.segments[index].data,
                    size_t(operation.read.segments[index].byte_count),
                };
            }
            sqe.opcode = IORING_OP_READV;
            sqe.addr = uint64_t(operation.iovecs.data());
            sqe.len = operation.read.segment_count;
            sqe.off = operation.read.file_offset;
            sqe.fd = int(operation.read.native_handle);
        }
        for (uint32_t file_index = 0; file_index < registered_file_count; ++file_index) {
            if (registered_files[file_index] == sqe.fd) {
                sqe.fd = int(file_index);
                sqe.flags |= IOSQE_FIXED_FILE;
                break;
            }
        }
        sqe.user_data = llm_expert_async_transport::encode_user_data(operation_slot, operation.generation);
        return true;
    }

    void record_request_traces_locked(const read_request_record & request) {
        for (const auto & operation : operations) {
            if (!operation.active || !operation.read_operation ||
                operation.identity.request.slot != request.handle.slot ||
                operation.identity.request.generation != request.handle.generation) continue;
            if (operation.submit_us == 0 || operation.complete_us < operation.submit_us) continue;
            if (counters.trace_records < traces.size()) {
                auto & trace = traces[counters.trace_records++];
                trace.sequence = counters.trace_records;
                trace.read = {};
                trace.read.flight = { operation.identity.transport_epoch, operation.identity.request.slot,
                    operation.identity.request.generation, operation.identity.key,
                    operation.identity.layout_class_id };
                trace.read.operation_index = operation.identity.request_operation_index;
                trace.read.queued_us = request.queued_us;
                trace.read.started_us = request.started_us;
                trace.read.submit_us = operation.submit_us;
                trace.read.complete_us = operation.complete_us;
                trace.read.bytes = operation.completed_bytes;
                trace.read.operation_file_offset = operation.read.file_offset;
                trace.read.source_segment_count = operation.read.segment_count;
                for (uint32_t index = 0; index < operation.read.segment_count; ++index) {
                    const auto & segment = operation.read.segments[index];
                    trace.read.source_segments[index] = { segment.file_offset, segment.byte_count };
                    trace.read.useful_bytes += segment.byte_count;
                }
            } else {
                counters.trace_records_dropped++;
            }
        }
    }

    void finalize_group_request_locked(read_request_record & request, bool used_ring) {
        if (request.cancel_requested) request.completion.result = llm_expert_async_result::closed;
        if (request.completion.result == llm_expert_async_result::ready) {
            std::array<const llm_expert_storage_read_segment *, 12> completed_segments{};
            size_t completed_segment_count = 0;
            for (const auto & operation : operations) {
                if (!operation.active || !operation.read_operation ||
                    operation.identity.request.slot != request.handle.slot ||
                    operation.identity.request.generation != request.handle.generation) continue;
                for (uint8_t index = 0; index < operation.read.segment_count; ++index) {
                    if (completed_segment_count < completed_segments.size()) {
                        completed_segments[completed_segment_count++] = &operation.read.segments[index];
                    }
                }
            }
            std::sort(completed_segments.begin(), completed_segments.begin() + completed_segment_count,
                [](const auto * lhs, const auto * rhs) {
                    const uint8_t lhs_identity = uint8_t(lhs->projection)*3 + uint8_t(lhs->sidecar);
                    const uint8_t rhs_identity = uint8_t(rhs->projection)*3 + uint8_t(rhs->sidecar);
                    return lhs_identity < rhs_identity;
                });
            LLM_EXPERT_TRACE_SCOPE("k3.storage", "integrity_digest", "request_slot", request.handle.slot,
                "request_generation", request.handle.generation, "segment_count", completed_segment_count);
            request.completion.digest = 1469598103934665603ULL;
            for (size_t segment_index = 0; segment_index < completed_segment_count; ++segment_index) {
                const auto & segment = *completed_segments[segment_index];
                const auto * bytes = static_cast<const uint8_t *>(segment.data);
                for (uint64_t byte_index = 0; byte_index < segment.byte_count; ++byte_index) {
                    request.completion.digest ^= bytes[byte_index];
                    request.completion.digest *= 1099511628211ULL;
                }
            }
        }
        request.completion.complete_us = uint64_t(ggml_time_us());
        request.completion.request = request.handle;
        record_request_traces_locked(request);
        [[maybe_unused]] const uint64_t trace_id = llm_perfetto_trace_pair_id(llm_perfetto_trace_domain::storage,
            request.handle.slot, uint32_t(request.handle.generation));
        LLM_EXPERT_TRACE_ASYNC_END("k3.storage", trace_id, "bytes", request.completion.bytes_completed,
            "result", uint32_t(request.completion.result), "native_error", request.completion.native_error,
            "used_io_uring", used_ring);
        request.state = read_state::complete;
        request.operations_remaining = 0;
        counters.read_requests_completed++;
        if (request.completion.result == llm_expert_async_result::closed) counters.read_requests_cancelled++;
        for (auto & operation : operations) {
            if (operation.active && operation.read_operation &&
                operation.identity.request.slot == request.handle.slot &&
                operation.identity.request.generation == request.handle.generation) {
                operation.active = false;
                operation.read_operation = false;
                operation.ring_completed = false;
                operation.cancel_submitted = false;
                operation.submit_us = 0;
                operation.complete_us = 0;
                operation.completed_bytes = 0;
                counters.active_operations--;
                counters.read_operations_completed++;
                if (!used_ring) counters.synchronous_fallback_operations++;
            }
        }
        counters.read_bytes_completed += request.completion.bytes_completed;
        condition.notify_all();
    }

    void run_buffered_group_fallback(size_t handle_count) {
        for (size_t handle_index = 0; handle_index < handle_count; ++handle_index) {
            const auto handle = group_handles[handle_index];
            for (auto & operation : operations) {
                if (!operation.active || !operation.read_operation ||
                    operation.identity.request.slot != handle.slot ||
                    operation.identity.request.generation != handle.generation) continue;
                operation.submit_us = uint64_t(ggml_time_us());
                operation.completed_bytes = 0;
                LLM_EXPERT_TRACE_ASYNC_BEGIN("k3.storage", "read_operation",
                    llm_perfetto_trace_operation_id(llm_perfetto_trace_domain::storage,
                        operation.identity.request.slot, uint32_t(operation.identity.request.generation),
                        operation.identity.request_operation_index),
                    "request_slot", operation.identity.request.slot, "request_generation",
                    operation.identity.request.generation, "operation_index",
                    operation.identity.request_operation_index, "submitted_bytes", operation.read.byte_count,
                    "file_offset", operation.read.file_offset, "io_uring", false);
                for (uint8_t segment_index = 0; segment_index < operation.read.segment_count; ++segment_index) {
                    const auto & segment = operation.read.segments[segment_index];
                    uint64_t completed = 0;
                    while (completed < segment.byte_count) {
                        {
                            std::lock_guard<std::mutex> guard(mutex);
                            auto * request = find_read(handle);
                            if (request == nullptr || request->cancel_requested) break;
                        }
                        const ssize_t result = pread(int(operation.read.native_handle),
                            static_cast<uint8_t *>(segment.data) + completed,
                            size_t(segment.byte_count - completed), off_t(segment.file_offset + completed));
                        if (result < 0 && (errno == EINTR || errno == EAGAIN)) continue;
                        std::lock_guard<std::mutex> guard(mutex);
                        auto * request = find_read(handle);
                        if (request == nullptr) break;
                        if (result <= 0) {
                            request->completion.result = llm_expert_async_result::invalid;
                            request->completion.native_error = result < 0 ? errno : 0;
                            break;
                        }
                        completed += uint64_t(result);
                        operation.completed_bytes += uint64_t(result);
                        request->completion.bytes_completed += uint64_t(result);
                    }
                }
                operation.complete_us = uint64_t(ggml_time_us());
                LLM_EXPERT_TRACE_ASYNC_END("k3.storage",
                    llm_perfetto_trace_operation_id(llm_perfetto_trace_domain::storage,
                        operation.identity.request.slot, uint32_t(operation.identity.request.generation),
                        operation.identity.request_operation_index),
                    "completed_bytes", operation.completed_bytes, "native_result", 0);
            }
            std::lock_guard<std::mutex> guard(mutex);
            auto * request = find_read(handle);
            if (request != nullptr) finalize_group_request_locked(*request, false);
        }
    }

    bool run_buffered_group(size_t handle_count) {
        size_t next_operation = 0;
        while (next_operation < operations.size()) {
            uint32_t batch = 0;
            {
                std::lock_guard<std::mutex> guard(mutex);
                for (; next_operation < operations.size() && batch < batch_slots.size(); ++next_operation) {
                    auto & operation = operations[next_operation];
                    if (!operation.active || !operation.read_operation) continue;
                    auto * request = find_read(operation.identity.request);
                    if (request == nullptr || request->state != read_state::running) continue;
                    io_uring_sqe * sqe = ring.acquire_sqe();
                    if (sqe == nullptr) break;
                    operation.ring_completed = false;
                    operation.cancel_submitted = false;
                    batch_slots[batch++] = uint32_t(next_operation);
                    (void) fill_read_sqe(uint32_t(next_operation), *sqe);
                }
            }
            if (batch == 0) break;
            int native_error = 0;
            uint32_t submitted = 0;
            while (submitted < batch) {
                const int count = ring.submit(native_error);
                if (count <= 0) {
                    ring.close_ring();
                    {
                        std::lock_guard<std::mutex> guard(mutex);
                        counters.io_uring_enabled = false;
                        counters.io_uring_runtime_error = count == 0 ? EAGAIN : native_error;
                        record_fallback(llm_expert_async_fallback_reason::ring_runtime,
                            counters.io_uring_runtime_error, "ring-runtime");
                    }
                    run_buffered_group_fallback(handle_count);
                    return false;
                }
                submitted += uint32_t(count);
            }
            {
                std::lock_guard<std::mutex> guard(mutex);
                counters.ring_submissions += submitted;
                counters.peak_sq_occupancy = std::max(counters.peak_sq_occupancy, submitted);
                const uint64_t submit_us = uint64_t(ggml_time_us());
                for (uint32_t index = 0; index < batch; ++index) {
                    auto & operation = operations[batch_slots[index]];
                    operation.submit_us = submit_us;
                    operation.complete_us = 0;
                    operation.completed_bytes = 0;
                    auto * request = find_read(operation.identity.request);
                    if (request != nullptr && request->completion.submit_us == 0) {
                        request->completion.submit_us = submit_us;
                        request->completion.request = request->handle;
                    }
                    [[maybe_unused]] const uint64_t operation_id = llm_perfetto_trace_operation_id(
                        llm_perfetto_trace_domain::storage, operation.identity.request.slot,
                        uint32_t(operation.identity.request.generation), operation.identity.request_operation_index);
                    LLM_EXPERT_TRACE_ASYNC_BEGIN("k3.storage", "read_operation", operation_id,
                        "request_slot", operation.identity.request.slot, "request_generation",
                        operation.identity.request.generation, "operation_index",
                        operation.identity.request_operation_index, "submitted_bytes", operation.read.byte_count,
                        "file_offset", operation.read.file_offset, "io_uring", true);
                }
                condition.notify_all();
            }
            if (config.delay_cq_drain_ms_for_testing != 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(config.delay_cq_drain_ms_for_testing));
            }
            uint32_t reads_left = submitted;
            uint32_t cancels_left = 0;
            while (reads_left != 0 || cancels_left != 0) {
                uint32_t cancels_prepared = 0;
                {
                    std::lock_guard<std::mutex> guard(mutex);
                    for (uint32_t index = 0; index < batch; ++index) {
                        auto & operation = operations[batch_slots[index]];
                        auto * request = find_read(operation.identity.request);
                        if (request == nullptr || !request->cancel_requested ||
                            operation.ring_completed || operation.cancel_submitted) continue;
                        io_uring_sqe * cancel = ring.acquire_sqe();
                        if (cancel == nullptr) break;
                        cancel->opcode = IORING_OP_ASYNC_CANCEL;
                        cancel->addr = llm_expert_async_transport::encode_user_data(
                            batch_slots[index], operation.generation);
                        cancel->cancel_flags = IORING_ASYNC_CANCEL_USERDATA;
                        cancel->user_data = llm_expert_async_transport::encode_user_data(
                            batch_slots[index] | 0x80000000U, operation.generation);
                        operation.cancel_submitted = true;
                        cancels_prepared++;
                    }
                }
                if (cancels_prepared != 0) {
                    const int count = ring.submit(native_error);
                    if (count > 0) {
                        cancels_left += uint32_t(count);
                        std::lock_guard<std::mutex> guard(mutex);
                        counters.ring_cancel_submissions += uint32_t(count);
                    }
                }
                io_uring_cqe cqe{};
                if (!ring.try_cqe(cqe)) {
                    std::unique_lock<std::mutex> guard(mutex);
                    counters.cq_empty_waits++;
                    condition.wait_for(guard, std::chrono::milliseconds(1));
                    continue;
                }
                uint32_t operation_slot = 0;
                uint32_t generation = 0;
                const bool decoded = llm_expert_async_transport::decode_user_data(
                    cqe.user_data, operation_slot, generation);
                const bool cancel_completion = (operation_slot & 0x80000000U) != 0;
                operation_slot &= 0x7fffffffU;
                std::lock_guard<std::mutex> guard(mutex);
                if (!decoded || operation_slot >= operations.size()) {
                    counters.stale_completions++;
                    if (cancel_completion) cancels_left--;
                    else reads_left--;
                    continue;
                }
                auto & operation = operations[operation_slot];
                auto * request = find_read(operation.identity.request);
                if (request == nullptr || operation.generation != generation) {
                    counters.stale_completions++;
                    if (cancel_completion) cancels_left--;
                    else reads_left--;
                    continue;
                }
                if (cancel_completion) {
                    cancels_left--;
                    counters.ring_cancel_completions++;
                    if (cqe.res < 0 && cqe.res != -ENOENT && cqe.res != -EALREADY &&
                        request->completion.result == llm_expert_async_result::ready) {
                        request->completion.result = llm_expert_async_result::invalid;
                        request->completion.native_error = -cqe.res;
                    }
                    continue;
                }
                if (cqe.res == -EINTR || cqe.res == -EAGAIN) {
                    io_uring_sqe * retry = ring.acquire_sqe();
                    int retry_error = 0;
                    if (retry != nullptr && fill_read_sqe(operation_slot, *retry) && ring.submit(retry_error) == 1) {
                        counters.ring_submissions++;
                        counters.ring_completions++;
                        if (cqe.res == -EINTR) counters.interrupted_reads_retried++;
                        else counters.would_block_reads_retried++;
                        continue;
                    }
                }
                operation.ring_completed = true;
                operation.complete_us = uint64_t(ggml_time_us());
                [[maybe_unused]] const uint64_t operation_id = llm_perfetto_trace_operation_id(
                    llm_perfetto_trace_domain::storage, operation.identity.request.slot,
                    uint32_t(operation.identity.request.generation), operation.identity.request_operation_index);
                LLM_EXPERT_TRACE_ASYNC_END("k3.storage", operation_id, "completed_bytes",
                    cqe.res > 0 ? uint64_t(cqe.res) : 0, "native_result", cqe.res);
                reads_left--;
                request->operations_remaining--;
                if (cqe.res < 0) {
                    if (cqe.res != -ECANCELED || !request->cancel_requested) {
                        request->completion.result = llm_expert_async_result::invalid;
                        request->completion.native_error = -cqe.res;
                    }
                } else if (uint64_t(cqe.res) != operation.read.byte_count) {
                    request->completion.result = llm_expert_async_result::invalid;
                } else {
                    operation.completed_bytes = uint64_t(cqe.res);
                    request->completion.bytes_completed += uint64_t(cqe.res);
                }
                counters.ring_completions++;
                if (request->operations_remaining == 0) finalize_group_request_locked(*request, true);
            }
        }
        return true;
    }

    bool run_ring(llm_expert_request_handle handle, llm_expert_async_read_completion & completion,
            intptr_t & direct_capability_handle, bool & transport_failed) {
        direct_capability_handle = -1;
        transport_failed = false;
        size_t next = 0;
        size_t remaining = 0;
        {
            std::lock_guard<std::mutex> guard(mutex);
            for (const auto & operation : operations) {
                if (operation.active && operation.read_operation &&
                    operation.identity.request.slot == handle.slot &&
                    operation.identity.request.generation == handle.generation) remaining++;
            }
        }
        while (remaining > 0) {
            uint32_t batch = 0;
            for (; next < operations.size(); ++next) {
                std::lock_guard<std::mutex> guard(mutex);
                auto & operation = operations[next];
                if (!operation.active || !operation.read_operation ||
                    operation.identity.request.slot != handle.slot ||
                    operation.identity.request.generation != handle.generation) continue;
                if (read_requests[handle.slot].cancel_requested) {
                    completion.result = llm_expert_async_result::closed;
                    return false;
                }
                io_uring_sqe * sqe = ring.acquire_sqe();
                if (sqe == nullptr) break;
                batch_slots[batch] = uint32_t(next);
                operation.ring_completed = false;
                operation.cancel_submitted = false;
                (void) fill_read_sqe(uint32_t(next), *sqe);
                batch++;
                if (config.direct_io_requested) {
                    ++next;
                    break;
                }
            }
            if (batch == 0) {
                completion.result = llm_expert_async_result::busy;
                return false;
            }
            int native_error = 0;
            uint32_t submitted = 0;
            while (submitted < batch) {
                const int count = ring.submit(native_error);
                if (count <= 0) {
                    // SQEs are published before enter. A hard enter failure after a partial
                    // submission cannot safely leave the unpublished tail for a later request.
                    // Closing the ring blocks until the kernel releases every submitted file,
                    // iovec, and destination reference; the complete unpublished bundle can
                    // then be retried by the buffered positional fallback.
                    ring.close_ring();
                    {
                        std::lock_guard<std::mutex> guard(mutex);
                        counters.io_uring_enabled = false;
                        counters.io_uring_runtime_error = count == 0 ? EAGAIN : native_error;
                        record_fallback(llm_expert_async_fallback_reason::ring_runtime,
                            counters.io_uring_runtime_error, "ring-runtime");
                    }
                    transport_failed = true;
                    completion.result = llm_expert_async_result::invalid;
                    completion.native_error = count == 0 ? EAGAIN : native_error;
                    return false;
                }
                submitted += uint32_t(count);
            }
            {
                std::lock_guard<std::mutex> guard(mutex);
                counters.ring_submissions += submitted;
                counters.peak_sq_occupancy = std::max(counters.peak_sq_occupancy, submitted);
                const uint64_t submit_us = uint64_t(ggml_time_us());
                for (uint32_t index = 0; index < batch; ++index) {
                    auto & operation = operations[batch_slots[index]];
                    operation.submit_us = submit_us;
                    operation.complete_us = 0;
                    operation.completed_bytes = 0;
                    LLM_EXPERT_TRACE_ASYNC_BEGIN("k3.storage", "read_operation",
                        llm_perfetto_trace_operation_id(llm_perfetto_trace_domain::storage,
                            operation.identity.request.slot, uint32_t(operation.identity.request.generation),
                            operation.identity.request_operation_index),
                        "request_slot", operation.identity.request.slot, "request_generation",
                        operation.identity.request.generation, "operation_index",
                        operation.identity.request_operation_index, "submitted_bytes", operation.read.byte_count,
                        "file_offset", operation.read.file_offset, "io_uring", true);
                }
            }
            if (config.pause_after_ring_submit_for_testing) {
                std::unique_lock<std::mutex> guard(mutex);
                ring_submit_paused_for_testing = true;
                condition.notify_all();
                condition.wait(guard, [&] {
                    return read_requests[handle.slot].cancel_requested || worker_stop;
                });
                ring_submit_paused_for_testing = false;
            }
            uint32_t read_completions_left = submitted;
            uint32_t cancel_completions_left = 0;
            bool cancel_requested = false;
            llm_expert_async_result first_error = llm_expert_async_result::ready;
            int first_native_error = 0;
            while (read_completions_left > 0 || cancel_completions_left > 0) {
                uint32_t cancels_prepared = 0;
                {
                    std::lock_guard<std::mutex> guard(mutex);
                    cancel_requested = read_requests[handle.slot].cancel_requested;
                    if (cancel_requested) {
                        for (uint32_t index = 0; index < batch; ++index) {
                            auto & operation = operations[batch_slots[index]];
                            if (operation.ring_completed || operation.cancel_submitted) continue;
                            io_uring_sqe * cancel = ring.acquire_sqe();
                            if (cancel == nullptr) break;
                            cancel->opcode = IORING_OP_ASYNC_CANCEL;
                            cancel->addr = llm_expert_async_transport::encode_user_data(
                                batch_slots[index], operation.generation);
                            cancel->cancel_flags = IORING_ASYNC_CANCEL_USERDATA;
                            cancel->user_data = llm_expert_async_transport::encode_user_data(
                                batch_slots[index] | 0x80000000U, operation.generation);
                            operation.cancel_submitted = true;
                            cancels_prepared++;
                        }
                    }
                }
                if (cancels_prepared != 0) {
                    uint32_t cancel_submitted = 0;
                    while (cancel_submitted < cancels_prepared) {
                        const int count = ring.submit(native_error);
                        if (count <= 0) {
                            if (first_error == llm_expert_async_result::ready) {
                                first_error = llm_expert_async_result::invalid;
                                first_native_error = count == 0 ? EAGAIN : native_error;
                            }
                            break;
                        }
                        cancel_submitted += uint32_t(count);
                    }
                    cancel_completions_left += cancel_submitted;
                    std::lock_guard<std::mutex> guard(mutex);
                    counters.ring_cancel_submissions += cancel_submitted;
                }
                io_uring_cqe cqe{};
                {
                    std::lock_guard<std::mutex> guard(mutex);
                    counters.peak_cq_occupancy = std::max(
                        counters.peak_cq_occupancy, ring.completion_occupancy());
                }
                const bool hide_cqe_for_testing = cancel_requested &&
                    hidden_cqe_polls_for_testing < config.hide_cqes_after_cancel_polls_for_testing;
                if (hide_cqe_for_testing) hidden_cqe_polls_for_testing++;
                if (hide_cqe_for_testing || !ring.try_cqe(cqe)) {
                    // The worker must remain responsive to cancellation and shutdown.
                    // Polling the bounded CQ with a condition-variable timeout lets the
                    // sole submitter observe cancellation and issue ASYNC_CANCEL SQEs.
                    std::unique_lock<std::mutex> guard(mutex);
                    counters.cq_empty_waits++;
                    if (cancel_requested || worker_stop) {
                        if (cancel_requested) counters.cq_empty_waits_after_cancel++;
                        condition.wait_for(guard, std::chrono::milliseconds(1));
                    } else {
                        condition.wait_for(guard, std::chrono::milliseconds(1), [&] {
                            return read_requests[handle.slot].cancel_requested || worker_stop;
                        });
                    }
                    continue;
                }
                uint32_t operation_slot = 0;
                uint32_t operation_generation = 0;
                const bool decoded = llm_expert_async_transport::decode_user_data(
                    cqe.user_data, operation_slot, operation_generation);
                const bool cancel_completion = (operation_slot & 0x80000000U) != 0;
                operation_slot &= 0x7fffffffU;
                std::lock_guard<std::mutex> guard(mutex);
                if (!decoded || operation_slot >= operations.size()) {
                    counters.stale_completions++;
                    if (first_error == llm_expert_async_result::ready) first_error = llm_expert_async_result::stale_generation;
                    if (cancel_completion) cancel_completions_left--;
                    else read_completions_left--;
                    continue;
                }
                auto & operation = operations[operation_slot];
                if (!operation.active || operation.generation != operation_generation ||
                    operation.identity.request.slot != handle.slot ||
                    operation.identity.request.generation != handle.generation) {
                    counters.stale_completions++;
                    if (first_error == llm_expert_async_result::ready) first_error = llm_expert_async_result::stale_generation;
                    if (cancel_completion) cancel_completions_left--;
                    else read_completions_left--;
                    continue;
                }
                if (cancel_completion) {
                    cancel_completions_left--;
                    counters.ring_cancel_completions++;
                    if (cqe.res < 0 && cqe.res != -ENOENT && cqe.res != -EALREADY &&
                        first_error == llm_expert_async_result::ready) {
                        first_error = llm_expert_async_result::invalid;
                        first_native_error = -cqe.res;
                    }
                    continue;
                }
                if (config.inject_first_read_cqe_for_testing && !read_cqe_injected_for_testing) {
                    cqe.res = config.first_read_cqe_result_for_testing;
                    read_cqe_injected_for_testing = true;
                }
                if (cqe.res == -EINTR || cqe.res == -EAGAIN) {
                    io_uring_sqe * retry = ring.acquire_sqe();
                    int retry_error = 0;
                    const bool retry_ready = retry != nullptr && fill_read_sqe(operation_slot, *retry) &&
                        ring.submit(retry_error) == 1;
                    counters.ring_completions++;
                    if (cqe.res == -EINTR) counters.interrupted_reads_retried++;
                    else counters.would_block_reads_retried++;
                    if (retry_ready) {
                        counters.ring_submissions++;
                        continue;
                    }
                    if (first_error == llm_expert_async_result::ready) {
                        first_error = llm_expert_async_result::invalid;
                        first_native_error = retry_error == 0 ? EAGAIN : retry_error;
                    }
                }
                operation.ring_completed = true;
                operation.complete_us = uint64_t(ggml_time_us());
                LLM_EXPERT_TRACE_ASYNC_END("k3.storage",
                    llm_perfetto_trace_operation_id(llm_perfetto_trace_domain::storage,
                        operation.identity.request.slot, uint32_t(operation.identity.request.generation),
                        operation.identity.request_operation_index),
                    "completed_bytes", cqe.res > 0 ? uint64_t(cqe.res) : 0, "native_result", cqe.res);
                read_completions_left--;
                if (cqe.res < 0) {
                    uint64_t ignored_offset = 0;
                    uint64_t ignored_bytes = 0;
                    const bool direct = prepare_direct(operation, ignored_offset, ignored_bytes);
                    if (direct && (cqe.res == -EINVAL || cqe.res == -EOPNOTSUPP || cqe.res == -ENOTSUP)) {
                        direct_capability_handle = operation.read.direct_native_handle;
                    }
                    if (cqe.res != -ECANCELED || !cancel_requested) {
                        if (first_error == llm_expert_async_result::ready) {
                            first_error = llm_expert_async_result::invalid;
                            first_native_error = -cqe.res;
                        }
                    }
                    counters.ring_completions++;
                    continue;
                }
                uint64_t direct_offset = 0;
                uint64_t direct_bytes = 0;
                const bool direct = prepare_direct(operation, direct_offset, direct_bytes);
                const uint64_t expected_bytes = direct ? direct_bytes : operation.read.byte_count;
                if (uint64_t(cqe.res) != expected_bytes) {
                    if (first_error == llm_expert_async_result::ready) first_error = llm_expert_async_result::invalid;
                    counters.ring_completions++;
                    continue;
                }
                if (direct) {
                    for (uint8_t segment_index = 0; segment_index < operation.read.segment_count; ++segment_index) {
                        const auto & segment = operation.read.segments[segment_index];
                        std::memcpy(segment.data,
                            static_cast<const uint8_t *>(staging) + (segment.file_offset - direct_offset),
                            size_t(segment.byte_count));
                    }
                    completion.bytes_completed += operation.read.byte_count;
                    operation.completed_bytes = operation.read.byte_count;
                    counters.direct_read_operations++;
                    counters.direct_useful_bytes += operation.read.byte_count;
                    counters.direct_aligned_bytes += direct_bytes;
                    counters.direct_scatter_bytes += operation.read.byte_count;
                } else {
                    completion.bytes_completed += uint64_t(cqe.res);
                    operation.completed_bytes = uint64_t(cqe.res);
                    if (config.direct_io_requested) {
                        counters.buffered_fallback_operations++;
                        counters.buffered_fallback_bytes += operation.read.byte_count;
                        uint64_t useful_end = 0;
                        uint64_t rounded_end = 0;
                        const bool eof_tail = operation.read.direct_alignment != 0 &&
                            checked_add(operation.read.file_offset, operation.read.byte_count, useful_end) &&
                            checked_add(useful_end, operation.read.direct_alignment - 1, rounded_end) &&
                            (rounded_end & ~(operation.read.direct_alignment - 1)) > operation.read.source_size;
                        const auto reason = eof_tail ? llm_expert_async_fallback_reason::direct_eof :
                            (staging == nullptr || staging_bytes == 0 ?
                                llm_expert_async_fallback_reason::direct_staging :
                                llm_expert_async_fallback_reason::direct_alignment);
                        record_fallback(reason, 0, eof_tail ? "direct-eof" :
                            (reason == llm_expert_async_fallback_reason::direct_staging ?
                                "direct-staging" : "direct-alignment"));
                    }
                }
                counters.ring_completions++;
            }
            if (cancel_requested) {
                completion.result = llm_expert_async_result::closed;
                return false;
            }
            if (first_error != llm_expert_async_result::ready) {
                completion.result = first_error;
                completion.native_error = first_native_error;
                return false;
            }
            remaining -= submitted;
        }
        return true;
    }
#endif

    void worker_main() {
        std::unique_lock<std::mutex> lock(mutex);
        for (;;) {
            condition.wait(lock, [&] {
                if (worker_stop) return true;
                if (deferred_batch_open) return false;
                for (const auto & request : read_requests) if (request.state == read_state::queued) return true;
                return false;
            });
#if defined(__linux__)
            if (counters.io_uring_enabled && config.read_override_for_testing == nullptr &&
                !config.direct_io_requested && config.submit_error_for_testing == 0 &&
                !config.pause_after_ring_submit_for_testing && !config.inject_first_read_cqe_for_testing &&
                config.hide_cqes_after_cancel_polls_for_testing == 0) {
                size_t handle_count = 0;
                for (auto & queued : read_requests) {
                    if (queued.state != read_state::queued) continue;
                    mark_request_running_locked(queued);
                    group_handles[handle_count++] = queued.handle;
                }
                if (handle_count != 0) {
                    counters.ring_request_batches++;
                    counters.peak_ring_batch_requests = std::max(
                        counters.peak_ring_batch_requests, uint32_t(handle_count));
                    lock.unlock();
                    (void) run_buffered_group(handle_count);
                    lock.lock();
                    continue;
                }
            }
#endif
            uint32_t selected = UINT32_MAX;
            for (uint32_t slot = 0; slot < read_requests.size(); ++slot) {
                if (read_requests[slot].state == read_state::queued &&
                    (selected == UINT32_MAX || (config.reverse_queued_requests_for_testing ?
                        read_requests[slot].ordinal > read_requests[selected].ordinal :
                        read_requests[slot].ordinal < read_requests[selected].ordinal))) {
                    selected = slot;
                }
            }
            if (selected == UINT32_MAX) {
                if (worker_stop) break;
                continue;
            }
            auto & request = read_requests[selected];
            mark_request_running_locked(request);
            const auto handle = request.handle;
            lock.unlock();

            llm_expert_async_read_completion completion;
            completion.result = llm_expert_async_result::ready;
            completion.submit_us = uint64_t(ggml_time_us());
            completion.request = handle;
            bool used_ring = false;
            bool execute_fallback = true;
            bool ring_fallback_buffered = false;
#if defined(__linux__)
            if (counters.io_uring_enabled && config.read_override_for_testing == nullptr) {
                used_ring = true;
                execute_fallback = false;
                intptr_t direct_capability_handle = -1;
                bool transport_failed = false;
                bool ring_ready = run_ring(handle, completion, direct_capability_handle, transport_failed);
                if (!ring_ready && !transport_failed &&
                    completion.result == llm_expert_async_result::invalid && direct_capability_handle >= 0 &&
                    (completion.native_error == EINVAL || completion.native_error == EOPNOTSUPP ||
                     completion.native_error == ENOTSUP)) {
                    {
                        std::lock_guard<std::mutex> guard(mutex);
                        (void) disable_direct(direct_capability_handle);
                        counters.direct_capability_retries++;
                        record_fallback(llm_expert_async_fallback_reason::direct_capability,
                            completion.native_error, "direct-capability");
                    }
                    completion = {};
                    completion.result = llm_expert_async_result::ready;
                    completion.submit_us = uint64_t(ggml_time_us());
                    completion.request = handle;
                    direct_capability_handle = -1;
                    ring_ready = run_ring(handle, completion, direct_capability_handle, transport_failed);
                }
                if (!ring_ready && transport_failed) {
                    completion = {};
                    completion.result = llm_expert_async_result::ready;
                    completion.submit_us = uint64_t(ggml_time_us());
                    completion.request = handle;
                    execute_fallback = true;
                    ring_fallback_buffered = true;
                    used_ring = false;
                }
            }
#endif
            if (execute_fallback) {
            bool force_buffered = ring_fallback_buffered;
            for (size_t operation_index = 0; operation_index < operations.size(); ++operation_index) {
                lock.lock();
                auto & stored = operations[operation_index];
                if (!stored.active || !stored.read_operation ||
                    stored.identity.request.slot != handle.slot ||
                    stored.identity.request.generation != handle.generation) {
                    lock.unlock();
                    continue;
                }
                const llm_expert_storage_read_operation operation = stored.read;
                stored.submit_us = uint64_t(ggml_time_us());
                stored.complete_us = 0;
                stored.completed_bytes = 0;
                LLM_EXPERT_TRACE_ASYNC_BEGIN("k3.storage", "read_operation",
                    llm_perfetto_trace_operation_id(llm_perfetto_trace_domain::storage,
                        stored.identity.request.slot, uint32_t(stored.identity.request.generation),
                        stored.identity.request_operation_index),
                    "request_slot", stored.identity.request.slot, "request_generation",
                    stored.identity.request.generation, "operation_index", stored.identity.request_operation_index,
                    "submitted_bytes", stored.read.byte_count, "file_offset", stored.read.file_offset,
                    "io_uring", false);
                if (read_requests[handle.slot].cancel_requested) {
                    completion.result = llm_expert_async_result::closed;
                }
                lock.unlock();
                if (completion.result != llm_expert_async_result::ready) break;
                auto read_all = [&](intptr_t native_handle, void * destination,
                                    uint64_t byte_count, uint64_t file_offset) {
                    uint64_t completed = 0;
                    while (completed < byte_count) {
#if defined(__linux__)
                        int native_error = 0;
                        const size_t remaining = size_t(byte_count - completed);
                        const int64_t result = config.read_override_for_testing != nullptr ?
                            config.read_override_for_testing->read_at(native_handle,
                                static_cast<uint8_t *>(destination) + completed, remaining,
                                file_offset + completed, native_error) :
                            int64_t(pread(int(native_handle),
                                static_cast<uint8_t *>(destination) + completed,
                                remaining, off_t(file_offset + completed)));
                        if (result < 0 && config.read_override_for_testing == nullptr) native_error = errno;
                        if (result < 0 && native_error == EINTR) {
                            std::lock_guard<std::mutex> guard(mutex);
                            counters.interrupted_reads_retried++;
                            continue;
                        }
                        if (result < 0 && native_error == EAGAIN) {
                            std::lock_guard<std::mutex> guard(mutex);
                            counters.would_block_reads_retried++;
                            if (read_requests[handle.slot].cancel_requested) {
                                completion.result = llm_expert_async_result::closed;
                                break;
                            }
                            continue;
                        }
                        if (result < 0) {
                            completion.result = llm_expert_async_result::invalid;
                            completion.native_error = native_error;
                            break;
                        }
                        if (result == 0 || uint64_t(result) > byte_count - completed) {
                            completion.result = llm_expert_async_result::invalid;
                            completion.native_error = 0;
                            break;
                        }
                        if (uint64_t(result) < byte_count - completed) {
                            std::lock_guard<std::mutex> guard(mutex);
                            counters.short_positive_reads++;
                        }
                        completed += uint64_t(result);
#else
                        completion.result = llm_expert_async_result::invalid;
                        completion.native_error = ENOSYS;
                        break;
#endif
                        std::lock_guard<std::mutex> guard(mutex);
                        if (read_requests[handle.slot].cancel_requested) {
                            completion.result = llm_expert_async_result::closed;
                            break;
                        }
                    }
                    return completion.result == llm_expert_async_result::ready;
                };

                uint64_t aligned_offset = 0;
                uint64_t aligned_bytes = 0;
                const bool direct = !force_buffered && prepare_direct(stored, aligned_offset, aligned_bytes);
                if (direct) {
                    if (read_all(operation.direct_native_handle, staging, aligned_bytes, aligned_offset)) {
                        for (uint8_t segment_index = 0; segment_index < operation.segment_count; ++segment_index) {
                            const auto & segment = operation.segments[segment_index];
                            std::memcpy(segment.data,
                                static_cast<const uint8_t *>(staging) + (segment.file_offset - aligned_offset),
                                size_t(segment.byte_count));
                            completion.bytes_completed += segment.byte_count;
                        }
                        std::lock_guard<std::mutex> guard(mutex);
                        counters.direct_read_operations++;
                        counters.direct_useful_bytes += operation.byte_count;
                        counters.direct_aligned_bytes += aligned_bytes;
                        counters.direct_scatter_bytes += operation.byte_count;
                    }
                    if (completion.result == llm_expert_async_result::invalid &&
                        (completion.native_error == EINVAL || completion.native_error == EOPNOTSUPP ||
                         completion.native_error == ENOTSUP)) {
                        const int direct_error = completion.native_error;
                        force_buffered = true;
                        const uint64_t submit_us = completion.submit_us;
                        completion = {};
                        completion.result = llm_expert_async_result::ready;
                        completion.submit_us = submit_us;
                        completion.request = handle;
                        operation_index = size_t(-1);
                        std::lock_guard<std::mutex> guard(mutex);
                        (void) disable_direct(operation.direct_native_handle);
                        counters.direct_capability_retries++;
                        record_fallback(llm_expert_async_fallback_reason::direct_capability,
                            direct_error, "direct-capability");
                        continue;
                    }
                } else {
                    {
                        std::lock_guard<std::mutex> guard(mutex);
                        if (config.direct_io_requested) {
                            counters.buffered_fallback_operations++;
                            counters.buffered_fallback_bytes += operation.byte_count;
                            uint64_t useful_end = 0;
                            uint64_t rounded_end = 0;
                            const bool eof_tail = operation.direct_alignment != 0 &&
                                checked_add(operation.file_offset, operation.byte_count, useful_end) &&
                                checked_add(useful_end, operation.direct_alignment - 1, rounded_end) &&
                                (rounded_end & ~(operation.direct_alignment - 1)) > operation.source_size;
                            const auto reason = eof_tail ? llm_expert_async_fallback_reason::direct_eof :
                                (staging == nullptr || staging_bytes == 0 ?
                                    llm_expert_async_fallback_reason::direct_staging :
                                    llm_expert_async_fallback_reason::direct_alignment);
                            record_fallback(reason, 0, eof_tail ? "direct-eof" :
                                (reason == llm_expert_async_fallback_reason::direct_staging ?
                                    "direct-staging" : "direct-alignment"));
                        }
                    }
                    for (uint8_t segment_index = 0; segment_index < operation.segment_count; ++segment_index) {
                        const auto & segment = operation.segments[segment_index];
                        if (!read_all(operation.native_handle, segment.data, segment.byte_count, segment.file_offset)) break;
                        completion.bytes_completed += segment.byte_count;
                    }
                }
                {
                    std::lock_guard<std::mutex> guard(mutex);
                    auto & completed = operations[operation_index];
                    completed.complete_us = uint64_t(ggml_time_us());
                    completed.completed_bytes = completion.result == llm_expert_async_result::ready ?
                        operation.byte_count : 0;
                    LLM_EXPERT_TRACE_ASYNC_END("k3.storage",
                        llm_perfetto_trace_operation_id(llm_perfetto_trace_domain::storage,
                            completed.identity.request.slot, uint32_t(completed.identity.request.generation),
                            completed.identity.request_operation_index),
                        "completed_bytes", completed.completed_bytes, "native_result", completion.native_error);
                }
                if (completion.result != llm_expert_async_result::ready) break;
            }
            }

            std::array<llm_expert_storage_read_segment, 12> completed_segments{};
            size_t completed_segment_count = 0;
            lock.lock();
            auto * current = find_read(handle);
            if (current != nullptr && current->cancel_requested) {
                completion.result = llm_expert_async_result::closed;
            }
            if (current != nullptr && completion.result == llm_expert_async_result::ready) {
                for (const auto & operation : operations) {
                    if (!operation.active || !operation.read_operation ||
                        operation.identity.request.slot != handle.slot ||
                        operation.identity.request.generation != handle.generation) continue;
                    for (uint8_t index = 0; index < operation.read.segment_count; ++index) {
                        if (completed_segment_count < completed_segments.size()) {
                            completed_segments[completed_segment_count++] = operation.read.segments[index];
                        }
                    }
                }
            }
            lock.unlock();
            if (completion.result == llm_expert_async_result::ready && completed_segment_count != 0) {
                LLM_EXPERT_TRACE_SCOPE("k3.storage", "integrity_digest", "request_slot", handle.slot,
                    "request_generation", handle.generation, "segment_count", completed_segment_count);
                std::sort(completed_segments.begin(), completed_segments.begin() + completed_segment_count,
                    [](const auto & lhs, const auto & rhs) {
                        const uint8_t lhs_identity = uint8_t(lhs.projection)*3 + uint8_t(lhs.sidecar);
                        const uint8_t rhs_identity = uint8_t(rhs.projection)*3 + uint8_t(rhs.sidecar);
                        return lhs_identity < rhs_identity;
                    });
                completion.digest = 1469598103934665603ULL;
                for (size_t segment_index = 0; segment_index < completed_segment_count; ++segment_index) {
                    const auto & segment = completed_segments[segment_index];
                    const auto * bytes = static_cast<const uint8_t *>(segment.data);
                    for (uint64_t byte_index = 0; byte_index < segment.byte_count; ++byte_index) {
                        completion.digest ^= bytes[byte_index];
                        completion.digest *= 1099511628211ULL;
                    }
                }
            }
            lock.lock();
            current = find_read(handle);
            if (current != nullptr) {
                if (current->cancel_requested) completion.result = llm_expert_async_result::closed;
                current->completion = completion;
                current->completion.complete_us = uint64_t(ggml_time_us());
                current->completion.request = handle;
                record_request_traces_locked(*current);
                [[maybe_unused]] const uint64_t trace_id = llm_perfetto_trace_pair_id(llm_perfetto_trace_domain::storage,
                    handle.slot, uint32_t(handle.generation));
                LLM_EXPERT_TRACE_ASYNC_END("k3.storage", trace_id, "bytes", completion.bytes_completed,
                    "result", uint32_t(completion.result), "native_error", completion.native_error,
                    "used_io_uring", false);
                current->state = read_state::complete;
                current->operations_remaining = 0;
                counters.read_requests_completed++;
                if (completion.result == llm_expert_async_result::closed) counters.read_requests_cancelled++;
                for (auto & operation : operations) {
                    if (operation.active && operation.read_operation &&
                        operation.identity.request.slot == handle.slot &&
                        operation.identity.request.generation == handle.generation) {
                        operation.active = false;
                        operation.read_operation = false;
                        operation.ring_completed = false;
                        operation.cancel_submitted = false;
                        operation.submit_us = 0;
                        operation.complete_us = 0;
                        operation.completed_bytes = 0;
                        counters.active_operations--;
                        counters.read_operations_completed++;
                        if (!used_ring) counters.synchronous_fallback_operations++;
                    }
                }
                counters.read_bytes_completed += completion.bytes_completed;
            }
            condition.notify_all();
        }
    }
};

llm_expert_async_transport::llm_expert_async_transport(llm_expert_async_config config) : pimpl(std::make_unique<impl>()) {
    if (config.effective_hot_capacity == 0 || config.request_capacity == 0 || config.trace_capacity == 0 ||
        config.cold_cache_bytes == 0 ||
        config.positional_worker_count == 0 || config.positional_worker_count > 4 ||
        (config.positional_worker_count > 1 && (!config.force_positional_reads || config.direct_io_requested)) ||
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
    pimpl->read_requests.resize(config.request_capacity);
    pimpl->traces.resize(config.trace_capacity);
    pimpl->batch_slots.resize(sq_entries);
    pimpl->group_handles.resize(config.request_capacity);
    pimpl->registered_files.resize(config.source_file_capacity == 0 ? 256 : config.source_file_capacity);
    pimpl->direct_disabled_handles.resize(pimpl->registered_files.size());
    pimpl->workers.reserve(config.positional_worker_count);
    for (auto & operation : pimpl->operations) {
        operation.generation = config.initial_operation_generation_for_testing;
    }
    pimpl->counters.requested_sq_entries = sq_entries;
    pimpl->counters.requested_cq_entries = cq_entries;
    pimpl->counters.operation_capacity = cq_entries;
    pimpl->counters.trace_capacity = config.trace_capacity;
    pimpl->counters.staging_ceiling_bytes = staging_ceiling;
    pimpl->counters.positional_reads_forced = config.force_positional_reads;
    if (config.direct_io_requested && config.maximum_direct_alignment != 0 && staging_ceiling == 0) {
        pimpl->counters.direct_staging_error = ENOBUFS;
    }
    pimpl->counters.administration_bytes = sizeof(*pimpl) +
        pimpl->operations.capacity()*sizeof(impl::operation_record) +
        pimpl->read_requests.capacity()*sizeof(impl::read_request_record) +
        pimpl->traces.capacity()*sizeof(impl::trace_record) +
        pimpl->batch_slots.capacity()*sizeof(uint32_t) +
        pimpl->group_handles.capacity()*sizeof(llm_expert_request_handle) +
        pimpl->registered_files.capacity()*sizeof(int) +
        pimpl->direct_disabled_handles.capacity()*sizeof(intptr_t) +
        pimpl->workers.capacity()*sizeof(std::thread);
#if defined(__linux__)
    if (config.direct_io_requested && staging_ceiling != 0 && config.maximum_direct_alignment != 0) {
        const uint64_t allocation_alignment = std::max<uint64_t>(config.maximum_direct_alignment, sizeof(void *));
        if (!is_power_of_two(config.maximum_direct_alignment) || !is_power_of_two(allocation_alignment) ||
            staging_ceiling > SIZE_MAX) {
            throw std::invalid_argument("invalid direct-I/O staging alignment");
        }
        if (allocation_alignment <= staging_ceiling) {
            void * staging = nullptr;
            const int allocation_error = posix_memalign(
                &staging, size_t(allocation_alignment), size_t(staging_ceiling));
            if (allocation_error == 0) {
                pimpl->staging = staging;
                pimpl->staging_bytes = staging_ceiling;
                pimpl->staging_alignment = allocation_alignment;
            } else {
                pimpl->counters.direct_staging_error = allocation_error;
            }
        } else {
            pimpl->counters.direct_staging_error = ENOBUFS;
        }
    }
    if (pimpl->counters.direct_staging_error != 0) {
        pimpl->record_fallback(llm_expert_async_fallback_reason::direct_staging,
            pimpl->counters.direct_staging_error, "direct-staging");
    }
#endif
#if defined(__linux__)
    static_assert(sizeof(io_uring_sqe) == 64, "unexpected io_uring SQE size");
    static_assert(sizeof(io_uring_cqe) == 16, "unexpected io_uring CQE size");
    pimpl->counters.linux_uapi = true;
    int ring_error = 0;
    if (!config.force_positional_reads && pimpl->ring.open_ring(sq_entries, ring_error)) {
        pimpl->counters.actual_sq_entries = pimpl->ring.actual_sq_entries();
        pimpl->counters.actual_cq_entries = pimpl->ring.actual_cq_entries();
        int probe_error = 0;
        const bool cq_capacity_safe = pimpl->counters.actual_sq_entries <= UINT32_MAX/2 &&
            pimpl->counters.actual_cq_entries >= pimpl->counters.actual_sq_entries*2;
        const bool probed = cq_capacity_safe && pimpl->ring.probe_opcodes(
                pimpl->counters.opcode_read,
                pimpl->counters.opcode_readv,
                pimpl->counters.opcode_async_cancel,
                pimpl->counters.opcode_read_fixed,
                probe_error);
        if (!cq_capacity_safe || !probed || !pimpl->counters.opcode_read || !pimpl->counters.opcode_readv ||
            !pimpl->counters.opcode_async_cancel) {
            pimpl->counters.io_uring_probe_error = !cq_capacity_safe ? ENOBUFS :
                (probe_error != 0 ? probe_error : EOPNOTSUPP);
            pimpl->record_fallback(llm_expert_async_fallback_reason::ring_capability,
                pimpl->counters.io_uring_probe_error, "ring-capability");
            pimpl->ring.close_ring();
        } else {
            pimpl->counters.io_uring_enabled = true;
            if (config.submit_error_for_testing != 0) {
                pimpl->ring.inject_submit_error_after(
                    config.partial_submit_count_before_error_for_testing,
                    config.submit_error_for_testing);
            }
        }
        if (pimpl->counters.io_uring_enabled && pimpl->staging != nullptr &&
            pimpl->counters.opcode_read_fixed) {
            int registration_error = 0;
            const bool registered = !config.force_buffer_registration_failure_for_testing &&
                pimpl->ring.register_buffer(pimpl->staging, size_t(pimpl->staging_bytes), registration_error);
            if (config.force_buffer_registration_failure_for_testing) registration_error = ENOMEM;
            if (registered) {
                pimpl->staging_registered = true;
                pimpl->counters.registered_buffer_count = 1;
                pimpl->counters.registered_buffer_bytes = pimpl->staging_bytes;
            } else if (recoverable_registration_error(registration_error)) {
                pimpl->counters.buffer_registration_error = registration_error;
                pimpl->record_fallback(llm_expert_async_fallback_reason::buffer_registration,
                    registration_error, "buffer-registration");
            } else {
                throw std::runtime_error("hard direct-I/O staging registration failure");
            }
        } else if (pimpl->counters.io_uring_enabled && pimpl->staging != nullptr &&
                   !pimpl->counters.opcode_read_fixed) {
            pimpl->counters.buffer_registration_error = EOPNOTSUPP;
            pimpl->record_fallback(llm_expert_async_fallback_reason::buffer_registration,
                EOPNOTSUPP, "buffer-registration");
        }
    } else if (!config.force_positional_reads) {
        pimpl->counters.io_uring_setup_error = ring_error;
        pimpl->record_fallback(llm_expert_async_fallback_reason::ring_setup,
            ring_error, "ring-setup");
    }
#endif
    try {
        for (uint32_t index = 0; index < config.positional_worker_count; ++index) {
            pimpl->workers.emplace_back([this] { pimpl->worker_main(); });
        }
    } catch (...) {
        {
            std::lock_guard<std::mutex> lock(pimpl->mutex);
            pimpl->worker_stop = true;
            pimpl->condition.notify_all();
        }
        for (auto & worker : pimpl->workers) {
            if (worker.joinable()) worker.join();
        }
        throw;
    }
    pimpl->counters.worker_count = uint32_t(pimpl->workers.size());
    pimpl->counters.worker_started = !pimpl->workers.empty();
}

llm_expert_async_transport::~llm_expert_async_transport() {
    (void) shutdown();
}

bool llm_expert_async_transport::validate_ring_layout(
        const llm_expert_async_ring_layout & layout,
        uint64_t page_size,
        llm_expert_async_mapping_sizes & sizes,
        bool single_mmap) noexcept {
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
    if (single_mmap) {
        uint64_t maximum_combined_ring = 0;
        if (!checked_add(maximum_sq_ring, cqes_bytes, maximum_combined_ring)) return false;
        return sizes.sq_ring_bytes <= maximum_combined_ring && sizes.cq_ring_bytes <= maximum_combined_ring;
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
    result.layout_valid = page_size > 0 && validate_ring_layout(
        layout, uint64_t(page_size), sizes, (params.features & IORING_FEAT_SINGLE_MMAP) != 0);
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
        !identity.key.is_valid(std::numeric_limits<int32_t>::max(), std::numeric_limits<int32_t>::max()) ||
        identity.layout_class_id >= LLM_EXPERT_LAYOUT_CLASS_MAX) {
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

llm_expert_async_result llm_expert_async_transport::submit_read_plan(
        const llm_expert_async_operation_identity & identity,
        const llm_expert_storage_read_operation * reads,
        size_t read_count,
        bool defer_worker) noexcept {
    LLM_EXPERT_TRACE_SCOPE("k3.storage", "request_submit", "flight_id",
        llm_perfetto_trace_pair_id(llm_perfetto_trace_domain::flight, identity.request.slot,
            uint32_t(identity.request.generation)), "layer", identity.key.layer,
        "original_expert_id", identity.key.expert, "layout_class_id", identity.layout_class_id,
        "operation_count", read_count);
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    if (pimpl->counters.admission_closed) return llm_expert_async_result::closed;
    if (!identity.request.valid() || identity.request.slot >= pimpl->read_requests.size() ||
        identity.transport_epoch != pimpl->counters.transport_epoch || reads == nullptr || read_count == 0 ||
        read_count > pimpl->operations.size() || identity.layout_class_id >= LLM_EXPERT_LAYOUT_CLASS_MAX) {
        return llm_expert_async_result::invalid;
    }
    for (size_t index = 0; index < read_count; ++index) {
        if (reads[index].layout_class_id != identity.layout_class_id) {
            return llm_expert_async_result::invalid;
        }
    }
    auto & request = pimpl->read_requests[identity.request.slot];
    if (request.state != impl::read_state::free) return llm_expert_async_result::busy;
    size_t inactive_operations = 0;
    size_t free_operations = 0;
    for (const auto & operation : pimpl->operations) {
        if (!operation.active) {
            inactive_operations++;
            if (operation.generation != std::numeric_limits<uint32_t>::max()) free_operations++;
        }
    }
    if (free_operations < read_count) return inactive_operations >= read_count ?
        llm_expert_async_result::generation_exhausted : llm_expert_async_result::busy;
    size_t source_index = 0;
    for (uint32_t slot = 0; slot < pimpl->operations.size() && source_index < read_count; ++slot) {
        auto & operation = pimpl->operations[slot];
        if (operation.active) continue;
        if (operation.generation == std::numeric_limits<uint32_t>::max()) continue;
        operation.generation++;
        operation.identity = identity;
        operation.identity.request_operation_index = uint32_t(source_index);
        operation.read = reads[source_index++];
        operation.active = true;
        operation.read_operation = true;
        operation.ring_completed = false;
        operation.cancel_submitted = false;
        operation.submit_us = 0;
        operation.complete_us = 0;
        operation.completed_bytes = 0;
        pimpl->counters.active_operations++;
        pimpl->counters.operations_reserved++;
    }
    pimpl->counters.peak_active_operations = std::max(
        pimpl->counters.peak_active_operations, pimpl->counters.active_operations);
    request.handle = identity.request;
    request.ordinal = pimpl->next_read_ordinal++;
    request.state = impl::read_state::queued;
    request.cancel_requested = false;
    request.operations_remaining = uint32_t(read_count);
    request.queued_us = uint64_t(ggml_time_us());
    request.started_us = 0;
    request.completion = {};
    request.completion.result = llm_expert_async_result::ready;
    request.completion.request = identity.request;
    pimpl->counters.read_requests_submitted++;
    pimpl->counters.active_read_requests++;
    pimpl->counters.peak_active_read_requests = std::max(
        pimpl->counters.peak_active_read_requests, pimpl->counters.active_read_requests);
    [[maybe_unused]] const uint64_t trace_id = llm_perfetto_trace_pair_id(llm_perfetto_trace_domain::storage,
        identity.request.slot, uint32_t(identity.request.generation));
    LLM_EXPERT_TRACE_ASYNC_BEGIN("k3.storage", "read_request", trace_id, "request_generation",
        identity.request.generation, "request_slot", identity.request.slot, "request_ordinal", request.ordinal,
        "layer", identity.key.layer, "original_expert_id", identity.key.expert,
        "layout_class_id", identity.layout_class_id, "operation_count", read_count);
    LLM_EXPERT_TRACE_COUNTER("k3.resource", "storage_active_requests", 3, pimpl->counters.active_read_requests);
    pimpl->deferred_batch_open = defer_worker;
    if (!defer_worker) pimpl->condition.notify_all();
    return llm_expert_async_result::ready;
}

void llm_expert_async_transport::start_deferred_reads() noexcept {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    pimpl->deferred_batch_open = false;
    pimpl->condition.notify_all();
}

llm_expert_async_result llm_expert_async_transport::register_files(
        const intptr_t * handles, size_t handle_count) noexcept {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    if (pimpl->counters.admission_closed) return llm_expert_async_result::closed;
    if (handles == nullptr || handle_count == 0 || handle_count > pimpl->registered_files.size() ||
        pimpl->counters.read_requests_submitted != 0) return llm_expert_async_result::invalid;
    for (size_t index = 0; index < handle_count; ++index) {
        if (handles[index] < 0 || handles[index] > std::numeric_limits<int>::max()) {
            return llm_expert_async_result::invalid;
        }
        pimpl->registered_files[index] = int(handles[index]);
    }
#if defined(__linux__)
    if (pimpl->counters.io_uring_enabled) {
        int native_error = 0;
        const bool registered = !pimpl->config.force_file_registration_failure_for_testing &&
            pimpl->ring.register_files(pimpl->registered_files.data(), uint32_t(handle_count), native_error);
        if (pimpl->config.force_file_registration_failure_for_testing) native_error = ENOMEM;
        if (!registered) {
            pimpl->counters.file_registration_error = native_error;
            pimpl->registered_file_count = 0;
            pimpl->record_fallback(llm_expert_async_fallback_reason::file_registration,
                native_error, "file-registration");
            return recoverable_registration_error(native_error) ?
                llm_expert_async_result::ready : llm_expert_async_result::invalid;
        }
        pimpl->registered_file_count = uint32_t(handle_count);
        pimpl->counters.registered_file_count = uint32_t(handle_count);
    }
#endif
    return llm_expert_async_result::ready;
}

llm_expert_async_result llm_expert_async_transport::wait_read(
        llm_expert_request_handle handle,
        llm_expert_async_read_completion & completion,
        bool (*abort_callback)(void *),
        void * abort_callback_data) noexcept {
    LLM_EXPERT_TRACE_SCOPE("k3.storage", "request_wait", "request_generation", handle.generation,
        "request_slot", handle.slot);
    std::unique_lock<std::mutex> lock(pimpl->mutex);
    auto * request = pimpl->find_read(handle);
    if (request == nullptr) return llm_expert_async_result::stale_generation;
    while (true) {
        const auto * current = pimpl->find_read(handle);
        if (current == nullptr || current->state == impl::read_state::complete) break;
        if (abort_callback != nullptr) {
            lock.unlock();
            const bool cancelled = abort_callback(abort_callback_data);
            lock.lock();
            current = pimpl->find_read(handle);
            if (current == nullptr) return llm_expert_async_result::stale_generation;
            if (cancelled) {
                auto * mutable_request = pimpl->find_read(handle);
                mutable_request->cancel_requested = true;
                pimpl->condition.notify_all();
            }
        }
        pimpl->condition.wait_for(lock, std::chrono::milliseconds(1));
    }
    request = pimpl->find_read(handle);
    if (request == nullptr) return llm_expert_async_result::stale_generation;
    completion = request->completion;
    return completion.result;
}

llm_expert_async_result llm_expert_async_transport::poll_read(
        llm_expert_request_handle handle,
        llm_expert_async_read_completion & completion) noexcept {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    const auto * request = pimpl->find_read(handle);
    if (request == nullptr) return llm_expert_async_result::stale_generation;
    if (request->state != impl::read_state::complete) return llm_expert_async_result::busy;
    completion = request->completion;
    return completion.result;
}

llm_expert_async_result llm_expert_async_transport::wait_any_read(
        const llm_expert_request_handle * handles,
        size_t handle_count,
        llm_expert_request_handle & completed_handle,
        llm_expert_async_read_completion & completion,
        bool (*abort_callback)(void *),
        void * abort_callback_data) noexcept {
    if (handles == nullptr || handle_count == 0) return llm_expert_async_result::invalid;
    std::unique_lock<std::mutex> lock(pimpl->mutex);
    while (true) {
        if (abort_callback != nullptr) {
            lock.unlock();
            const bool cancelled = abort_callback(abort_callback_data);
            lock.lock();
            if (cancelled) {
                for (size_t index = 0; index < handle_count; ++index) {
                    auto * request = pimpl->find_read(handles[index]);
                    if (request != nullptr) request->cancel_requested = true;
                }
                pimpl->condition.notify_all();
                return llm_expert_async_result::closed;
            }
        }
        bool live = false;
        for (size_t index = 0; index < handle_count; ++index) {
            const auto * request = pimpl->find_read(handles[index]);
            if (request == nullptr) continue;
            live = true;
            if (request->state == impl::read_state::complete) {
                completed_handle = handles[index];
                completion = request->completion;
                return completion.result;
            }
        }
        if (!live) return llm_expert_async_result::stale_generation;
        pimpl->condition.wait_for(lock, std::chrono::milliseconds(1));
    }
}

llm_expert_async_result llm_expert_async_transport::cancel_read(llm_expert_request_handle handle) noexcept {
    LLM_EXPERT_TRACE_INSTANT("k3.lifecycle", "storage_request_cancel", "request_generation", handle.generation,
        "request_slot", handle.slot);
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    auto * request = pimpl->find_read(handle);
    if (request == nullptr) return llm_expert_async_result::stale_generation;
    request->cancel_requested = true;
    pimpl->condition.notify_all();
    return llm_expert_async_result::ready;
}

llm_expert_async_result llm_expert_async_transport::release_read(llm_expert_request_handle handle) noexcept {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    auto * request = pimpl->find_read(handle);
    if (request == nullptr) return llm_expert_async_result::stale_generation;
    if (request->state != impl::read_state::complete) return llm_expert_async_result::busy;
    request->state = impl::read_state::free;
    request->handle = {};
    request->ordinal = 0;
    request->cancel_requested = false;
    request->operations_remaining = 0;
    request->queued_us = 0;
    request->started_us = 0;
    pimpl->counters.active_read_requests--;
    return llm_expert_async_result::ready;
}

bool llm_expert_async_transport::wait_until_ring_submitted_for_testing() noexcept {
    std::unique_lock<std::mutex> lock(pimpl->mutex);
    if (!pimpl->config.pause_after_ring_submit_for_testing || !pimpl->counters.io_uring_enabled) return false;
    return pimpl->condition.wait_for(lock, std::chrono::seconds(5), [&] {
        return pimpl->ring_submit_paused_for_testing || pimpl->counters.admission_closed;
    }) && pimpl->ring_submit_paused_for_testing;
}

bool llm_expert_async_transport::wait_until_read_submitted_for_testing(
        llm_expert_request_handle handle) noexcept {
    std::unique_lock<std::mutex> lock(pimpl->mutex);
    return pimpl->condition.wait_for(lock, std::chrono::seconds(5), [&] {
        const auto * request = pimpl->find_read(handle);
        return request == nullptr || request->completion.submit_us != 0 ||
            request->state == impl::read_state::complete || pimpl->counters.admission_closed;
    }) && pimpl->find_read(handle) != nullptr &&
        pimpl->find_read(handle)->completion.submit_us != 0;
}

bool llm_expert_async_transport::shutdown() noexcept {
    LLM_EXPERT_TRACE_SCOPE("k3.lifecycle", "async_io_shutdown");
    {
        std::lock_guard<std::mutex> lock(pimpl->mutex);
        if (!pimpl->counters.admission_closed) {
            pimpl->counters.admission_closed = true;
            pimpl->counters.transport_epoch++;
        }
        for (auto & request : pimpl->read_requests) {
            if (request.state != impl::read_state::free && request.state != impl::read_state::complete) {
                request.cancel_requested = true;
            }
        }
        pimpl->worker_stop = true;
        pimpl->deferred_batch_open = false;
        pimpl->condition.notify_all();
    }
    for (auto & worker : pimpl->workers) {
        if (worker.joinable() && worker.get_id() != std::this_thread::get_id()) {
            worker.join();
        }
    }
#if defined(__linux__)
    // The ring path has one worker; close it only after that worker drains every completion.
    pimpl->ring.close_ring();
#endif
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    bool drained = pimpl->counters.active_operations == 0;
    for (auto & request : pimpl->read_requests) {
        if (request.state == impl::read_state::complete) {
            request.state = impl::read_state::free;
            request.operations_remaining = 0;
            if (pimpl->counters.active_read_requests > 0) pimpl->counters.active_read_requests--;
        } else if (request.state != impl::read_state::free) {
            drained = false;
        }
    }
    return drained;
}

llm_expert_async_diagnostics llm_expert_async_transport::diagnostics() const noexcept {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    return pimpl->counters;
}

std::vector<llm_expert_async_read_interval> llm_expert_async_transport::completed_read_intervals() const {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    std::vector<llm_expert_async_read_interval> result;
    result.reserve(size_t(pimpl->counters.trace_records));
    for (uint64_t index = 0; index < pimpl->counters.trace_records; ++index) {
        const auto & read = pimpl->traces[index].read;
        if (read.submit_us != 0 && read.complete_us >= read.submit_us) result.push_back(read);
    }
    return result;
}
