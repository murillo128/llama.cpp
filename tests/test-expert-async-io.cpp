#include "llama-expert-async-io.h"
#include "llama-expert-weight-provider.h"
#include "llama-model.h"

#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <fcntl.h>
#include <unistd.h>
#endif

namespace {

uint64_t fallback_bit(llm_expert_async_fallback_reason reason) {
    return static_cast<uint64_t>(reason);
}

template<class F> void expect_invalid(F fn) {
    bool rejected = false;
    try { fn(); } catch (const std::invalid_argument &) { rejected = true; }
    GGML_ASSERT(rejected);
}

llm_expert_async_config config(uint32_t queue_depth = 0, uint32_t operation_generation = 0) {
    return {
        queue_depth,
        12,
        16,
        4,
        256U*1024U*1024U,
        0,
        2U*1024U*1024U,
        true,
        operation_generation,
    };
}

struct scripted_async_reader : llm_expert_async_read_override {
    enum class action { interrupted, would_block, partial, error, normal, block_then_would_block };
    std::vector<action> actions;
    std::array<uint8_t, 64> bytes{};
    size_t call = 0;
    std::mutex mutex;
    std::condition_variable condition;
    bool entered = false;
    bool released = false;
    int error_code = EIO;
    uint64_t source_size = std::numeric_limits<uint64_t>::max();

    scripted_async_reader() {
        for (size_t index = 0; index < bytes.size(); ++index) bytes[index] = uint8_t(0x50 + index);
    }

    int64_t read_at(intptr_t, void * destination, size_t byte_count, uint64_t file_offset,
            int & native_error) noexcept override {
        const action current = call < actions.size() ? actions[call++] : action::normal;
        if (current == action::block_then_would_block) {
            std::unique_lock<std::mutex> lock(mutex);
            entered = true;
            condition.notify_all();
            condition.wait(lock, [&] { return released; });
            native_error = EAGAIN;
            return -1;
        }
        if (current == action::interrupted) { native_error = EINTR; return -1; }
        if (current == action::would_block) { native_error = EAGAIN; return -1; }
        if (current == action::error) { native_error = error_code; return -1; }
        if (file_offset >= source_size) return 0;
        const size_t available = size_t(std::min<uint64_t>(source_size - file_offset, byte_count));
        const size_t count = current == action::partial ? std::min<size_t>(3, available) : available;
        std::memcpy(destination, bytes.data() + file_offset, count);
        return int64_t(count);
    }
};

struct concurrent_async_reader : llm_expert_async_read_override {
    std::mutex mutex;
    std::condition_variable condition;
    uint32_t entered = 0;
    bool released = false;

    int64_t read_at(intptr_t, void * destination, size_t byte_count, uint64_t file_offset,
            int & native_error) noexcept override {
        {
            std::unique_lock<std::mutex> lock(mutex);
            entered++;
            condition.notify_all();
            condition.wait(lock, [&] { return released; });
        }
        native_error = 0;
        auto * bytes = static_cast<uint8_t *>(destination);
        for (size_t index = 0; index < byte_count; ++index) {
            bytes[index] = uint8_t(file_offset + index);
        }
        return int64_t(byte_count);
    }
};

class provider_owned_destination final : public llm_expert_weight_provider {
public:
    provider_owned_destination(llm_expert_async_transport * transport, bool & drained_before_destroy) :
        transport(transport), drained_before_destroy(drained_before_destroy) {}

    ~provider_owned_destination() override {
        const auto diagnostics = transport->diagnostics();
        drained_before_destroy = diagnostics.admission_closed && diagnostics.active_operations == 0 &&
            diagnostics.active_read_requests == 0;
    }

    llm_expert_provider_result bind(
            const llm_expert_bundle_descriptor &,
            const llm_expert_selection &,
            llm_expert_graph_binding &) noexcept override {
        return llm_expert_provider_result::failure(llm_expert_provider_error::unsupported_configuration);
    }

    llm_expert_provider_result prepare(
            const std::vector<llm_expert_graph_binding> &,
            llm_expert_execution_plan &,
            uint64_t,
            bool) noexcept override {
        return llm_expert_provider_result::failure(llm_expert_provider_error::unsupported_configuration);
    }

    llm_expert_provider_stats get_stats() const noexcept override { return {}; }
    void release_handle(uint64_t) noexcept override {}

    std::array<uint8_t, 8> destination{};

private:
    llm_expert_async_transport * transport;
    bool & drained_before_destroy;
};

void test_configuration() {
    GGML_ASSERT(llm_expert_resolve_io_worker_count(0, 2, false) == 1);
    GGML_ASSERT(llm_expert_resolve_io_worker_count(0, 2, true) == 2);
    GGML_ASSERT(llm_expert_resolve_io_worker_count(0, 4, false) == 1);
    GGML_ASSERT(llm_expert_resolve_io_worker_count(2, 1, true) == 2);
    GGML_ASSERT(llm_expert_resolve_io_worker_count(1, 2, false) == 0);
    GGML_ASSERT(llm_expert_resolve_io_worker_count(9, 2, true) == 0);

    expect_invalid([] { llm_expert_async_transport transport({}); });
    expect_invalid([] { llm_expert_async_transport transport(config(7)); });
    expect_invalid([] { llm_expert_async_transport transport(config(4097)); });
    expect_invalid([] {
        auto invalid = config(8);
        invalid.integrity_mode = static_cast<llm_expert_integrity_mode>(UINT8_MAX);
        llm_expert_async_transport transport(invalid);
    });

    llm_expert_async_transport transport(config());
    const auto diagnostics = transport.diagnostics();
    GGML_ASSERT(diagnostics.integrity_mode == llm_expert_integrity_mode::none);
    GGML_ASSERT(diagnostics.integrity_digest_requests == 0 && diagnostics.integrity_digest_bytes == 0);
    GGML_ASSERT(diagnostics.requested_sq_entries == 64);
    GGML_ASSERT(diagnostics.requested_cq_entries == 128);
    GGML_ASSERT(diagnostics.operation_capacity == 128);
    GGML_ASSERT(diagnostics.trace_capacity == 4);
    GGML_ASSERT(diagnostics.staging_ceiling_bytes == 4U*1024U*1024U);
    GGML_ASSERT(diagnostics.administration_bytes > 0);
#if defined(__linux__)
    GGML_ASSERT(diagnostics.linux_uapi);
    if (diagnostics.io_uring_enabled) {
        GGML_ASSERT(diagnostics.actual_sq_entries >= diagnostics.requested_sq_entries);
        GGML_ASSERT(diagnostics.actual_cq_entries >= diagnostics.actual_sq_entries*2);
        GGML_ASSERT(diagnostics.io_uring_setup_error == 0);
    } else {
        GGML_ASSERT(diagnostics.io_uring_setup_error != 0 || diagnostics.io_uring_probe_error != 0);
        GGML_ASSERT((diagnostics.fallback_reason_mask &
            (fallback_bit(llm_expert_async_fallback_reason::ring_setup) |
             fallback_bit(llm_expert_async_fallback_reason::ring_capability))) != 0);
        GGML_ASSERT(diagnostics.fallback_diagnostics_emitted > 0);
    }
#else
    GGML_ASSERT(!diagnostics.linux_uapi);
#endif

    llm_expert_async_config bounded = config(8);
    bounded.requested_staging_bytes = 1024;
    llm_expert_async_transport explicit_transport(bounded);
    GGML_ASSERT(explicit_transport.diagnostics().staging_ceiling_bytes == 1024);

    auto undersized = config(8);
    undersized.direct_io_requested = true;
    undersized.maximum_direct_alignment = 8;
    undersized.requested_staging_bytes = 4;
    llm_expert_async_transport undersized_transport(undersized);
    GGML_ASSERT(undersized_transport.diagnostics().direct_staging_error == ENOBUFS);
    GGML_ASSERT((undersized_transport.diagnostics().fallback_reason_mask &
        fallback_bit(llm_expert_async_fallback_reason::direct_staging)) != 0);

    auto positional = config(8);
    positional.force_positional_reads = true;
    positional.worker_count = 2;
    llm_expert_async_transport positional_transport(positional);
    const auto positional_diagnostics = positional_transport.diagnostics();
    GGML_ASSERT(positional_diagnostics.positional_reads_forced);
    GGML_ASSERT(positional_diagnostics.worker_count == 2);
    GGML_ASSERT(!positional_diagnostics.io_uring_enabled);
    GGML_ASSERT(positional_diagnostics.fallback_reason_mask == 0);

    auto invalid_workers = config(8);
    invalid_workers.worker_count = 2;
    expect_invalid([&] { llm_expert_async_transport invalid(invalid_workers); });

    auto direct_positional_workers = config(8);
    direct_positional_workers.force_positional_reads = true;
    direct_positional_workers.direct_io_requested = true;
    direct_positional_workers.worker_count = llm_expert_resolve_io_worker_count(2, 1, true);
    llm_expert_async_transport direct_positional(direct_positional_workers);
    GGML_ASSERT(direct_positional.diagnostics().worker_count == 2);
}

void test_ring_layout_validation() {
    llm_expert_async_ring_layout layout = {
        8, 16,
        0, 4, 8, 12, 16, 20, 64,
        0, 4, 8, 12, 16, 64,
    };
    llm_expert_async_mapping_sizes sizes;
    GGML_ASSERT(llm_expert_async_transport::validate_ring_layout(layout, 4096, sizes));
    GGML_ASSERT(sizes.sq_ring_bytes == 96);
    GGML_ASSERT(sizes.cq_ring_bytes == 320);
    GGML_ASSERT(sizes.sqes_bytes == 512);

    llm_expert_async_ring_layout combined = {
        128, 256,
        0, 4, 16, 24, 36, 32, 4160,
        8, 12, 20, 28, 44, 64,
    };
    GGML_ASSERT(!llm_expert_async_transport::validate_ring_layout(combined, 4096, sizes));
    GGML_ASSERT(llm_expert_async_transport::validate_ring_layout(combined, 4096, sizes, true));
    GGML_ASSERT(sizes.sq_ring_bytes == 4672);
    GGML_ASSERT(sizes.cq_ring_bytes == 4160);
    GGML_ASSERT(sizes.sqes_bytes == 8192);

    layout.sq_entries = 7;
    GGML_ASSERT(!llm_expert_async_transport::validate_ring_layout(layout, 4096, sizes));
    layout.sq_entries = 8;
    layout.cqes = std::numeric_limits<uint32_t>::max();
    GGML_ASSERT(!llm_expert_async_transport::validate_ring_layout(layout, 4096, sizes));
    layout.cqes = 64;
    GGML_ASSERT(!llm_expert_async_transport::validate_ring_layout(layout, 3000, sizes));

    const auto probe = llm_expert_async_transport::probe_ring_for_testing(8);
    if (probe.supported) {
        GGML_ASSERT(probe.layout_valid);
        GGML_ASSERT(probe.sq_entries >= 8 && probe.cq_entries >= probe.sq_entries);
    } else {
        GGML_ASSERT(probe.native_error != 0);
    }
#if defined(__linux__)
    for (uint32_t queue_depth : { 128U, 256U }) {
        const auto large_probe = llm_expert_async_transport::probe_ring_for_testing(queue_depth);
        if (large_probe.supported) GGML_ASSERT(large_probe.layout_valid);
    }
#endif
    GGML_ASSERT(llm_expert_async_transport::probe_ring_for_testing(7).native_error == EINVAL);
}

void test_user_data_and_fake_completion() {
    uint32_t slot = 0;
    uint32_t generation = 0;
    const uint64_t encoded = llm_expert_async_transport::encode_user_data(17, 23);
    GGML_ASSERT(encoded != 0);
    GGML_ASSERT(llm_expert_async_transport::decode_user_data(encoded, slot, generation));
    GGML_ASSERT(slot == 17 && generation == 23);
    GGML_ASSERT(!llm_expert_async_transport::decode_user_data(0, slot, generation));

    llm_expert_async_transport transport(config(8));
    const llm_expert_async_operation_identity expected = {
        1,
        { 2, 9 },
        3,
        { 1, 4 },
        llm_expert_readiness::device_ready,
        llm_expert_priority::demand_current_layer,
    };
    const auto operation = transport.reserve_operation(expected);
    GGML_ASSERT(operation.result == llm_expert_async_result::ready && operation.user_data != 0);
    llm_expert_async_operation_identity actual;
    GGML_ASSERT(transport.consume_completion_for_testing(operation.user_data, actual) == llm_expert_async_result::ready);
    GGML_ASSERT(actual.transport_epoch == expected.transport_epoch);
    GGML_ASSERT(actual.request.slot == expected.request.slot);
    GGML_ASSERT(actual.request.generation == expected.request.generation);
    GGML_ASSERT(actual.request_operation_index == expected.request_operation_index);
    GGML_ASSERT(actual.key.layer == expected.key.layer && actual.key.expert == expected.key.expert);
    const auto reused = transport.reserve_operation(expected);
    GGML_ASSERT(reused.result == llm_expert_async_result::ready && reused.user_data != operation.user_data);
    GGML_ASSERT(transport.consume_completion_for_testing(operation.user_data, actual) ==
        llm_expert_async_result::stale_generation);
    GGML_ASSERT(transport.consume_completion_for_testing(reused.user_data, actual) ==
        llm_expert_async_result::ready);
    const auto diagnostics = transport.diagnostics();
    GGML_ASSERT(diagnostics.operations_reserved == 2);
    GGML_ASSERT(diagnostics.completions_consumed == 2);
    GGML_ASSERT(diagnostics.stale_completions == 1);
    GGML_ASSERT(diagnostics.active_operations == 0);
}

void test_bounded_trace_and_shutdown() {
    llm_expert_async_transport transport(config(8));
    for (int index = 0; index < 6; ++index) {
        transport.record_trace_for_testing();
    }
    auto diagnostics = transport.diagnostics();
    GGML_ASSERT(diagnostics.trace_records == 4);
    GGML_ASSERT(diagnostics.trace_records_dropped == 2);
    GGML_ASSERT(transport.shutdown());
    diagnostics = transport.diagnostics();
    GGML_ASSERT(diagnostics.admission_closed);
    GGML_ASSERT(diagnostics.transport_epoch == 2);

    const llm_expert_async_operation_identity identity = {
        2, { 0, 1 }, 0, { 0, 0 }, llm_expert_readiness::host_ready,
        llm_expert_priority::demand_current_layer,
    };
    GGML_ASSERT(transport.reserve_operation(identity).result == llm_expert_async_result::closed);
}

void test_generation_exhaustion() {
    llm_expert_async_transport transport(config(8, std::numeric_limits<uint32_t>::max()));
    const llm_expert_async_operation_identity identity = {
        1, { 0, 1 }, 0, { 0, 0 }, llm_expert_readiness::host_ready,
        llm_expert_priority::demand_current_layer,
    };
    GGML_ASSERT(transport.reserve_operation(identity).result == llm_expert_async_result::generation_exhausted);
}

void test_worker_read_and_drain() {
#if defined(__linux__)
    FILE * file = std::tmpfile();
    GGML_ASSERT(file != nullptr);
    std::array<uint8_t, 32> source;
    for (size_t index = 0; index < source.size(); ++index) source[index] = uint8_t(0x30 + index);
    GGML_ASSERT(std::fwrite(source.data(), source.size(), 1, file) == 1);
    GGML_ASSERT(std::fflush(file) == 0);
    std::array<uint8_t, 7> first{};
    std::array<uint8_t, 9> second{};
    llm_expert_storage_read_operation read;
    read.native_handle = fileno(file);
    read.file_offset = 3;
    read.byte_count = first.size() + second.size();
    read.segment_count = 2;
    read.segments[0] = { first.data(), first.size(), 3,
        llm_expert_storage_projection::up, llm_expert_storage_sidecar::weight };
    read.segments[1] = { second.data(), second.size(), 10,
        llm_expert_storage_projection::gate, llm_expert_storage_sidecar::weight };
    read.layout_class_id = 2;
    read.segments[0].layout_class_id = 2;
    read.segments[1].layout_class_id = 2;
    llm_expert_async_transport transport(config(8));
    const llm_expert_async_operation_identity identity = {
        1, { 1, 7 }, 0, { 0, 2 }, llm_expert_readiness::host_ready,
        llm_expert_priority::demand_current_layer, 2,
    };
    auto mismatched_read = read;
    mismatched_read.layout_class_id = 1;
    GGML_ASSERT(transport.submit_read_plan(identity, &mismatched_read, 1) ==
        llm_expert_async_result::invalid);
    GGML_ASSERT(transport.submit_read_plan(identity, &read, 1) == llm_expert_async_result::ready);
    llm_expert_async_read_completion completion;
    GGML_ASSERT(transport.wait_read(identity.request, completion) == llm_expert_async_result::ready);
    GGML_ASSERT(completion.bytes_completed == first.size() + second.size());
    GGML_ASSERT(completion.integrity_status == llm_expert_integrity_status::not_checked &&
        completion.digest == 0);
    GGML_ASSERT(std::memcmp(first.data(), source.data() + 3, first.size()) == 0);
    GGML_ASSERT(std::memcmp(second.data(), source.data() + 10, second.size()) == 0);
    GGML_ASSERT(transport.release_read(identity.request) == llm_expert_async_result::ready);
    const auto intervals = transport.completed_read_intervals();
    GGML_ASSERT(intervals.size() == 1);
    for (size_t index = 0; index < intervals.size(); ++index) {
        GGML_ASSERT(intervals[index].flight.transport_epoch == identity.transport_epoch);
        GGML_ASSERT(intervals[index].flight.request_slot == identity.request.slot);
        GGML_ASSERT(intervals[index].flight.request_generation == identity.request.generation);
        GGML_ASSERT(intervals[index].flight.key.layer == identity.key.layer &&
            intervals[index].flight.key.expert == identity.key.expert);
        GGML_ASSERT(intervals[index].flight.layout_class_id == identity.layout_class_id);
        GGML_ASSERT(intervals[index].operation_index == index);
        GGML_ASSERT(intervals[index].queued_us != 0);
        GGML_ASSERT(intervals[index].started_us >= intervals[index].queued_us);
        GGML_ASSERT(intervals[index].submit_us >= intervals[index].started_us);
        GGML_ASSERT(intervals[index].complete_us >= intervals[index].submit_us);
        GGML_ASSERT(intervals[index].bytes == read.byte_count);
    }
    const auto diagnostics = transport.diagnostics();
    GGML_ASSERT(diagnostics.read_requests_submitted == 1 && diagnostics.read_requests_completed == 1);
    GGML_ASSERT(diagnostics.read_operations_completed == 1);
    GGML_ASSERT(diagnostics.read_bytes_completed == first.size() + second.size());
    GGML_ASSERT(diagnostics.read_queue_wait_samples == 1);
    GGML_ASSERT(diagnostics.read_queue_wait_max_us <= diagnostics.read_queue_wait_us);
    GGML_ASSERT(diagnostics.active_read_requests == 0);

    first.fill(0);
    second.fill(0);
    auto checked_config = config(8);
    checked_config.force_positional_reads = true;
    checked_config.integrity_mode = llm_expert_integrity_mode::fnv64_end_to_end;
    llm_expert_async_transport checked_transport(checked_config);
    GGML_ASSERT(checked_transport.submit_read_plan(identity, &read, 1) == llm_expert_async_result::ready);
    llm_expert_async_read_completion checked_completion;
    GGML_ASSERT(checked_transport.wait_read(identity.request, checked_completion) == llm_expert_async_result::ready);
    uint64_t expected_digest = 1469598103934665603ULL;
    auto append_digest = [](uint64_t & digest, const auto & destination) {
        for (uint8_t byte : destination) {
            digest ^= byte;
            digest *= 1099511628211ULL;
        }
    };
    append_digest(expected_digest, first);
    append_digest(expected_digest, second);
    GGML_ASSERT(checked_completion.integrity_status == llm_expert_integrity_status::passed &&
        checked_completion.digest == expected_digest);
    const auto checked_diagnostics = checked_transport.diagnostics();
    GGML_ASSERT(checked_diagnostics.integrity_mode == llm_expert_integrity_mode::fnv64_end_to_end &&
        checked_diagnostics.integrity_digest_requests == 1 &&
        checked_diagnostics.integrity_digest_bytes == first.size() + second.size());
    first[0] ^= 0xff;
    uint64_t corrupted_digest = 1469598103934665603ULL;
    append_digest(corrupted_digest, first);
    append_digest(corrupted_digest, second);
    GGML_ASSERT(corrupted_digest != checked_completion.digest);
    GGML_ASSERT(checked_transport.release_read(identity.request) == llm_expert_async_result::ready);
    GGML_ASSERT(std::fclose(file) == 0);
#endif
}

void test_nonblocking_read_poll() {
    scripted_async_reader reader;
    reader.actions = { scripted_async_reader::action::block_then_would_block };
    auto cfg = config(8);
    cfg.read_override_for_testing = &reader;
    llm_expert_async_transport transport(cfg);
    std::array<uint8_t, 8> destination{};
    llm_expert_storage_read_operation read;
    read.native_handle = 17;
    read.file_offset = 0;
    read.byte_count = destination.size();
    read.segment_count = 1;
    read.segments[0] = { destination.data(), destination.size(), 0,
        llm_expert_storage_projection::up, llm_expert_storage_sidecar::weight };
    const llm_expert_async_operation_identity identity = {
        1, { 0, 2 }, 0, { 0, 4 }, llm_expert_readiness::host_ready,
        llm_expert_priority::prefetch_next,
    };
    GGML_ASSERT(transport.submit_read_plan(identity, &read, 1) ==
        llm_expert_async_result::ready);
    {
        std::unique_lock<std::mutex> lock(reader.mutex);
        reader.condition.wait(lock, [&] { return reader.entered; });
    }
    llm_expert_async_read_completion completion;
    GGML_ASSERT(transport.poll_read(identity.request, completion) ==
        llm_expert_async_result::busy);
    {
        std::lock_guard<std::mutex> lock(reader.mutex);
        reader.released = true;
        reader.condition.notify_all();
    }
    llm_expert_async_result result = llm_expert_async_result::busy;
    for (uint32_t attempt = 0; attempt < 100000 && result == llm_expert_async_result::busy; ++attempt) {
        result = transport.poll_read(identity.request, completion);
        std::this_thread::yield();
    }
    GGML_ASSERT(result == llm_expert_async_result::ready);
    GGML_ASSERT(completion.bytes_completed == destination.size());
    GGML_ASSERT(std::memcmp(destination.data(), reader.bytes.data(), destination.size()) == 0);
    GGML_ASSERT(transport.release_read(identity.request) == llm_expert_async_result::ready);
}

void test_bounded_positional_worker_parallelism() {
    concurrent_async_reader reader;
    auto cfg = config(8);
    cfg.force_positional_reads = true;
    cfg.worker_count = 2;
    cfg.read_override_for_testing = &reader;
    llm_expert_async_transport transport(cfg);
    std::array<std::array<uint8_t, 8>, 2> destinations{};
    std::array<llm_expert_storage_read_operation, 2> reads{};
    std::array<llm_expert_async_operation_identity, 2> identities{};
    for (uint32_t index = 0; index < 2; ++index) {
        reads[index].native_handle = 17;
        reads[index].source_size = 64;
        reads[index].file_offset = index*16;
        reads[index].byte_count = destinations[index].size();
        reads[index].segment_count = 1;
        reads[index].segments[0] = { destinations[index].data(), destinations[index].size(),
            reads[index].file_offset, llm_expert_storage_projection::up,
            llm_expert_storage_sidecar::weight };
        identities[index] = {
            1, { index, 1 }, index, { 0, int32_t(index) }, llm_expert_readiness::host_ready,
            llm_expert_priority::demand_current_layer,
        };
        GGML_ASSERT(transport.submit_read_plan(identities[index], &reads[index], 1, true) ==
            llm_expert_async_result::ready);
    }
    transport.start_deferred_reads();
    bool parallel = false;
    {
        std::unique_lock<std::mutex> lock(reader.mutex);
        parallel = reader.condition.wait_for(lock, std::chrono::seconds(5), [&] {
            return reader.entered == 2;
        });
        reader.released = true;
        reader.condition.notify_all();
    }
    GGML_ASSERT(parallel);
    for (uint32_t index = 0; index < 2; ++index) {
        llm_expert_async_read_completion completion;
        GGML_ASSERT(transport.wait_read(identities[index].request, completion) ==
            llm_expert_async_result::ready);
        GGML_ASSERT(completion.bytes_completed == destinations[index].size());
        GGML_ASSERT(destinations[index][0] == reads[index].file_offset);
        GGML_ASSERT(transport.release_read(identities[index].request) == llm_expert_async_result::ready);
    }
    const auto diagnostics = transport.diagnostics();
    GGML_ASSERT(diagnostics.worker_count == 2 && diagnostics.active_read_requests == 0 &&
        diagnostics.active_operations == 0);
}

void test_deferred_multi_request_ring_batch() {
#if defined(__linux__)
    FILE * file = std::tmpfile();
    GGML_ASSERT(file != nullptr);
    std::array<uint8_t, 32> source;
    for (size_t index = 0; index < source.size(); ++index) source[index] = uint8_t(0x70 + index);
    GGML_ASSERT(std::fwrite(source.data(), source.size(), 1, file) == 1);
    GGML_ASSERT(std::fflush(file) == 0);

    llm_expert_async_transport transport(config(8));
    if (!transport.diagnostics().io_uring_enabled) {
        GGML_ASSERT(std::fclose(file) == 0);
        return;
    }
    std::array<std::array<uint8_t, 4>, 3> destinations{};
    std::array<llm_expert_storage_read_operation, 3> reads{};
    std::array<llm_expert_async_operation_identity, 3> identities{};
    const std::array<uint32_t, 3> submit_order = { 2, 0, 1 };
    for (uint32_t index = 0; index < 3; ++index) {
        reads[index].native_handle = fileno(file);
        reads[index].source_size = source.size();
        reads[index].file_offset = 2 + index*7;
        reads[index].byte_count = destinations[index].size();
        reads[index].segment_count = 1;
        reads[index].segments[0] = { destinations[index].data(), destinations[index].size(),
            reads[index].file_offset, llm_expert_storage_projection::up,
            llm_expert_storage_sidecar::weight };
        identities[index] = { 1, { index, uint64_t(index) + 41 }, 0, { 0, int32_t(index) },
            llm_expert_readiness::host_ready, llm_expert_priority::demand_current_layer };
    }
    for (uint32_t index : submit_order) {
        GGML_ASSERT(transport.submit_read_plan(identities[index], &reads[index], 1, true) ==
            llm_expert_async_result::ready);
    }
    transport.start_deferred_reads();
    std::array<llm_expert_request_handle, 3> handles = {
        identities[0].request, identities[1].request, identities[2].request,
    };
    std::array<bool, 3> seen{};
    for (size_t completed = 0; completed < handles.size(); ++completed) {
        llm_expert_async_read_completion completion;
        llm_expert_request_handle handle;
        GGML_ASSERT(transport.wait_any_read(handles.data(), handles.size(), handle, completion) ==
            llm_expert_async_result::ready);
        GGML_ASSERT(handle.slot < seen.size() && !seen[handle.slot]);
        const uint32_t index = handle.slot;
        seen[index] = true;
        GGML_ASSERT(completion.bytes_completed == destinations[index].size());
        GGML_ASSERT(std::memcmp(destinations[index].data(), source.data() + reads[index].file_offset,
            destinations[index].size()) == 0);
        GGML_ASSERT(transport.release_read(identities[index].request) == llm_expert_async_result::ready);
    }
    GGML_ASSERT(std::all_of(seen.begin(), seen.end(), [](bool value) { return value; }));
    const auto diagnostics = transport.diagnostics();
    GGML_ASSERT(diagnostics.peak_ring_batch_requests == 3);
    GGML_ASSERT(diagnostics.ring_submissions == 3 && diagnostics.ring_completions == 3);
    GGML_ASSERT(diagnostics.active_read_requests == 0 && diagnostics.active_operations == 0);
    GGML_ASSERT(std::fclose(file) == 0);
#endif
}

void test_partial_ring_submission_falls_back_after_quiescence() {
#if defined(__linux__)
    FILE * file = std::tmpfile();
    GGML_ASSERT(file != nullptr);
    std::array<uint8_t, 32> source;
    for (size_t index = 0; index < source.size(); ++index) source[index] = uint8_t(0x20 + index);
    GGML_ASSERT(std::fwrite(source.data(), source.size(), 1, file) == 1);
    GGML_ASSERT(std::fflush(file) == 0);

    auto cfg = config(8);
    cfg.partial_submit_count_before_error_for_testing = 1;
    cfg.submit_error_for_testing = EBADF;
    llm_expert_async_transport transport(cfg);
    if (!transport.diagnostics().io_uring_enabled) {
        GGML_ASSERT(std::fclose(file) == 0);
        return;
    }
    std::array<uint8_t, 5> first{};
    std::array<uint8_t, 7> second{};
    std::array<llm_expert_storage_read_operation, 2> reads;
    reads[0].native_handle = fileno(file);
    reads[0].file_offset = 2;
    reads[0].byte_count = first.size();
    reads[0].segment_count = 1;
    reads[0].segments[0] = { first.data(), first.size(), 2,
        llm_expert_storage_projection::up, llm_expert_storage_sidecar::weight };
    reads[1].native_handle = fileno(file);
    reads[1].file_offset = 17;
    reads[1].byte_count = second.size();
    reads[1].segment_count = 1;
    reads[1].segments[0] = { second.data(), second.size(), 17,
        llm_expert_storage_projection::down, llm_expert_storage_sidecar::weight };
    const llm_expert_async_operation_identity identity = {
        1, { 0, 11 }, 0, { 0, 7 }, llm_expert_readiness::host_ready,
        llm_expert_priority::demand_current_layer,
    };
    GGML_ASSERT(transport.submit_read_plan(identity, reads.data(), reads.size()) ==
        llm_expert_async_result::ready);
    llm_expert_async_read_completion completion;
    GGML_ASSERT(transport.wait_read(identity.request, completion) == llm_expert_async_result::ready);
    GGML_ASSERT(completion.bytes_completed == first.size() + second.size());
    GGML_ASSERT(std::memcmp(first.data(), source.data() + 2, first.size()) == 0);
    GGML_ASSERT(std::memcmp(second.data(), source.data() + 17, second.size()) == 0);
    GGML_ASSERT(transport.release_read(identity.request) == llm_expert_async_result::ready);
    const auto diagnostics = transport.diagnostics();
    GGML_ASSERT(!diagnostics.io_uring_enabled);
    GGML_ASSERT(diagnostics.io_uring_runtime_error == EBADF);
    GGML_ASSERT((diagnostics.fallback_reason_mask &
        fallback_bit(llm_expert_async_fallback_reason::ring_runtime)) != 0);
    GGML_ASSERT(diagnostics.fallback_diagnostics_emitted > 0);
    GGML_ASSERT(diagnostics.synchronous_fallback_operations == reads.size());
    GGML_ASSERT(diagnostics.active_operations == 0 && diagnostics.active_read_requests == 0);
    GGML_ASSERT(std::fclose(file) == 0);
#endif
}

void test_file_registration_or_explicit_fallback() {
#if defined(__linux__)
    FILE * file = std::tmpfile();
    GGML_ASSERT(file != nullptr);
    const intptr_t handle = fileno(file);
    llm_expert_async_transport transport(config(8));
    GGML_ASSERT(transport.register_files(&handle, 1) == llm_expert_async_result::ready);
    const auto diagnostics = transport.diagnostics();
    if (diagnostics.io_uring_enabled && diagnostics.file_registration_error == 0) {
        GGML_ASSERT(diagnostics.registered_file_count == 1);
    } else {
        GGML_ASSERT(diagnostics.registered_file_count == 0);
        GGML_ASSERT(diagnostics.io_uring_setup_error != 0 || diagnostics.io_uring_probe_error != 0 ||
            diagnostics.file_registration_error != 0);
    }
    GGML_ASSERT(transport.register_files(nullptr, 0) == llm_expert_async_result::invalid);

    auto forced_cfg = config(8);
    forced_cfg.force_file_registration_failure_for_testing = true;
    llm_expert_async_transport forced(forced_cfg);
    GGML_ASSERT(forced.register_files(&handle, 1) == llm_expert_async_result::ready);
    const auto forced_diagnostics = forced.diagnostics();
    if (forced_diagnostics.io_uring_enabled) {
        GGML_ASSERT(forced_diagnostics.registered_file_count == 0);
        GGML_ASSERT(forced_diagnostics.file_registration_error == ENOMEM);
        GGML_ASSERT((forced_diagnostics.fallback_reason_mask &
            fallback_bit(llm_expert_async_fallback_reason::file_registration)) != 0);
    }

    auto buffer_cfg = config(8);
    buffer_cfg.direct_io_requested = true;
    buffer_cfg.maximum_direct_alignment = 8;
    buffer_cfg.force_buffer_registration_failure_for_testing = true;
    llm_expert_async_transport forced_buffer(buffer_cfg);
    const auto buffer_diagnostics = forced_buffer.diagnostics();
    if (buffer_diagnostics.io_uring_enabled) {
        GGML_ASSERT(buffer_diagnostics.registered_buffer_count == 0);
        GGML_ASSERT(buffer_diagnostics.registered_buffer_bytes == 0);
        GGML_ASSERT(buffer_diagnostics.buffer_registration_error == ENOMEM);
        GGML_ASSERT((buffer_diagnostics.fallback_reason_mask &
            fallback_bit(llm_expert_async_fallback_reason::buffer_registration)) != 0);
    }
    GGML_ASSERT(std::fclose(file) == 0);
#endif
}

void test_native_ring_cancel_and_shutdown_drain() {
#if defined(__linux__)
    auto make_read = [](FILE * file, std::array<uint8_t, 8> & destination) {
        llm_expert_storage_read_operation read;
        read.native_handle = fileno(file);
        read.source_size = destination.size();
        read.file_offset = 0;
        read.byte_count = destination.size();
        read.segment_count = 1;
        read.segments[0] = { destination.data(), destination.size(), 0,
            llm_expert_storage_projection::up, llm_expert_storage_sidecar::weight };
        return read;
    };
    FILE * file = std::tmpfile();
    GGML_ASSERT(file != nullptr);
    const std::array<uint8_t, 8> source = { 1, 2, 3, 4, 5, 6, 7, 8 };
    GGML_ASSERT(std::fwrite(source.data(), 1, source.size(), file) == source.size());
    GGML_ASSERT(std::fflush(file) == 0);

    auto cancel_cfg = config(8);
    cancel_cfg.pause_after_ring_submit_for_testing = true;
    cancel_cfg.hide_cqes_after_cancel_polls_for_testing = 8;
    llm_expert_async_transport cancel_transport(cancel_cfg);
    if (cancel_transport.diagnostics().io_uring_enabled) {
        std::array<uint8_t, 8> destination{};
        auto read = make_read(file, destination);
        const llm_expert_async_operation_identity identity = {
            1, { 0, 11 }, 0, { 0, 7 }, llm_expert_readiness::host_ready,
            llm_expert_priority::demand_current_layer,
        };
        GGML_ASSERT(cancel_transport.submit_read_plan(identity, &read, 1) == llm_expert_async_result::ready);
        GGML_ASSERT(cancel_transport.wait_until_ring_submitted_for_testing());
        const auto cancel_start = std::chrono::steady_clock::now();
        GGML_ASSERT(cancel_transport.cancel_read(identity.request) == llm_expert_async_result::ready);
        llm_expert_async_read_completion completion;
        GGML_ASSERT(cancel_transport.wait_read(identity.request, completion) == llm_expert_async_result::closed);
        const auto cancel_elapsed = std::chrono::steady_clock::now() - cancel_start;
        GGML_ASSERT(cancel_transport.release_read(identity.request) == llm_expert_async_result::ready);
        const auto diagnostics = cancel_transport.diagnostics();
        GGML_ASSERT(diagnostics.ring_cancel_submissions == 1);
        GGML_ASSERT(diagnostics.ring_cancel_completions == 1);
        GGML_ASSERT(diagnostics.ring_completions == 1);
        GGML_ASSERT(diagnostics.cq_empty_waits_after_cancel >= 8);
        GGML_ASSERT(cancel_elapsed >= std::chrono::milliseconds(4));
        GGML_ASSERT(diagnostics.active_operations == 0);
    }

    auto shutdown_cfg = config(8);
    shutdown_cfg.pause_after_ring_submit_for_testing = true;
    llm_expert_async_transport shutdown_transport(shutdown_cfg);
    if (shutdown_transport.diagnostics().io_uring_enabled) {
        std::array<uint8_t, 8> destination{};
        auto read = make_read(file, destination);
        const llm_expert_async_operation_identity identity = {
            1, { 0, 12 }, 0, { 0, 8 }, llm_expert_readiness::host_ready,
            llm_expert_priority::demand_current_layer,
        };
        GGML_ASSERT(shutdown_transport.submit_read_plan(identity, &read, 1) == llm_expert_async_result::ready);
        GGML_ASSERT(shutdown_transport.wait_until_ring_submitted_for_testing());
        GGML_ASSERT(shutdown_transport.shutdown());
        const auto diagnostics = shutdown_transport.diagnostics();
        GGML_ASSERT(diagnostics.ring_cancel_submissions == 1);
        GGML_ASSERT(diagnostics.ring_cancel_completions == 1);
        GGML_ASSERT(diagnostics.active_read_requests == 0);
        GGML_ASSERT(diagnostics.active_operations == 0);
    }
    GGML_ASSERT(std::fclose(file) == 0);
#endif
}

void test_native_ring_cqe_failure_matrix() {
#if defined(__linux__)
    FILE * file = std::tmpfile();
    GGML_ASSERT(file != nullptr);
    const std::array<uint8_t, 8> source = { 9, 8, 7, 6, 5, 4, 3, 2 };
    GGML_ASSERT(std::fwrite(source.data(), 1, source.size(), file) == source.size());
    GGML_ASSERT(std::fflush(file) == 0);

    auto run = [&](int32_t injected_result, uint64_t generation,
                   llm_expert_async_read_completion & completion) {
        auto cfg = config(8);
        cfg.inject_first_read_cqe_for_testing = true;
        cfg.first_read_cqe_result_for_testing = injected_result;
        llm_expert_async_transport transport(cfg);
        if (!transport.diagnostics().io_uring_enabled) return transport.diagnostics();
        std::array<uint8_t, 8> destination{};
        llm_expert_storage_read_operation read;
        read.native_handle = fileno(file);
        read.source_size = source.size();
        read.byte_count = destination.size();
        read.segment_count = 1;
        read.segments[0] = { destination.data(), destination.size(), 0,
            llm_expert_storage_projection::up, llm_expert_storage_sidecar::weight };
        const llm_expert_async_operation_identity identity = {
            1, { 0, generation }, 0, { 0, 9 }, llm_expert_readiness::host_ready,
            llm_expert_priority::demand_current_layer,
        };
        GGML_ASSERT(transport.submit_read_plan(identity, &read, 1) == llm_expert_async_result::ready);
        (void) transport.wait_read(identity.request, completion);
        const auto diagnostics = transport.diagnostics();
        if (completion.result == llm_expert_async_result::ready) {
            GGML_ASSERT(destination == source);
        }
        GGML_ASSERT(transport.release_read(identity.request) == llm_expert_async_result::ready);
        return diagnostics;
    };

    llm_expert_async_read_completion completion;
    const auto interrupted = run(-EINTR, 21, completion);
    if (interrupted.io_uring_enabled) {
        GGML_ASSERT(completion.result == llm_expert_async_result::ready);
        GGML_ASSERT(interrupted.interrupted_reads_retried == 1);
        GGML_ASSERT(interrupted.ring_submissions == 2 && interrupted.ring_completions == 2);
    }
    const auto would_block = run(-EAGAIN, 22, completion);
    if (would_block.io_uring_enabled) {
        GGML_ASSERT(completion.result == llm_expert_async_result::ready);
        GGML_ASSERT(would_block.would_block_reads_retried == 1);
        GGML_ASSERT(would_block.ring_submissions == 2 && would_block.ring_completions == 2);
    }
    const auto short_read = run(3, 23, completion);
    if (short_read.io_uring_enabled) {
        GGML_ASSERT(completion.result == llm_expert_async_result::invalid);
        GGML_ASSERT(completion.native_error == 0);
    }
    const auto hard_error = run(-EIO, 24, completion);
    if (hard_error.io_uring_enabled) {
        GGML_ASSERT(completion.result == llm_expert_async_result::invalid);
        GGML_ASSERT(completion.native_error == EIO);
        GGML_ASSERT(hard_error.interrupted_reads_retried == 0 && hard_error.would_block_reads_retried == 0);
    }
    GGML_ASSERT(std::fclose(file) == 0);
#endif
}

void test_model_owner_drains_before_provider_destination_destroy() {
#if defined(__linux__)
    FILE * file = std::tmpfile();
    GGML_ASSERT(file != nullptr);
    const std::array<uint8_t, 8> source = { 4, 8, 15, 16, 23, 42, 7, 9 };
    GGML_ASSERT(std::fwrite(source.data(), 1, source.size(), file) == source.size());
    GGML_ASSERT(std::fflush(file) == 0);

    auto cfg = config(8);
    cfg.pause_after_ring_submit_for_testing = true;
    auto transport = std::make_unique<llm_expert_async_transport>(cfg);
    auto scheduler = std::make_unique<llm_expert_scheduler>(llm_expert_scheduler_config{ 1, 1, 1, 1, 0 });
    bool drained_before_provider_destroy = false;
    auto destination_owner = std::make_unique<provider_owned_destination>(
        transport.get(), drained_before_provider_destroy);
    auto * owner = destination_owner.get();
    std::unique_ptr<llm_expert_weight_provider> provider = std::move(destination_owner);
    std::unique_ptr<llm_expert_storage> storage;

    if (transport->diagnostics().io_uring_enabled) {
        llm_expert_storage_read_operation read;
        read.native_handle = fileno(file);
        read.source_size = source.size();
        read.byte_count = owner->destination.size();
        read.segment_count = 1;
        read.segments[0] = { owner->destination.data(), owner->destination.size(), 0,
            llm_expert_storage_projection::up, llm_expert_storage_sidecar::weight };
        const llm_expert_async_operation_identity identity = {
            1, { 0, 31 }, 0, { 0, 10 }, llm_expert_readiness::host_ready,
            llm_expert_priority::demand_current_layer,
        };
        GGML_ASSERT(transport->submit_read_plan(identity, &read, 1) == llm_expert_async_result::ready);
        GGML_ASSERT(transport->wait_until_ring_submitted_for_testing());
    }

    llm_shutdown_expert_runtime(transport, scheduler, provider, storage);
    GGML_ASSERT(drained_before_provider_destroy);
    GGML_ASSERT(transport == nullptr && scheduler == nullptr && provider == nullptr && storage == nullptr);
    GGML_ASSERT(std::fclose(file) == 0);
#endif
}

void test_fallback_retry_error_and_cancel_races() {
#if defined(__linux__)
    auto run = [](scripted_async_reader & reader, llm_expert_async_read_completion & completion) {
        auto cfg = config(8);
        cfg.read_override_for_testing = &reader;
        llm_expert_async_transport transport(cfg);
        std::array<uint8_t, 8> destination{};
        llm_expert_storage_read_operation read;
        read.native_handle = 17;
        read.file_offset = 4;
        read.byte_count = destination.size();
        read.segment_count = 1;
        read.segments[0] = { destination.data(), destination.size(), 4,
            llm_expert_storage_projection::up, llm_expert_storage_sidecar::weight };
        const llm_expert_async_operation_identity identity = {
            1, { 0, 1 }, 0, { 0, 3 }, llm_expert_readiness::host_ready,
            llm_expert_priority::demand_current_layer,
        };
        GGML_ASSERT(transport.submit_read_plan(identity, &read, 1) == llm_expert_async_result::ready);
        const auto result = transport.wait_read(identity.request, completion);
        const auto diagnostics = transport.diagnostics();
        if (result == llm_expert_async_result::ready) {
            GGML_ASSERT(std::memcmp(destination.data(), reader.bytes.data() + 4, destination.size()) == 0);
        }
        GGML_ASSERT(transport.release_read(identity.request) == llm_expert_async_result::ready);
        return diagnostics;
    };

    scripted_async_reader retry_reader;
    retry_reader.actions = { scripted_async_reader::action::interrupted,
        scripted_async_reader::action::would_block, scripted_async_reader::action::partial };
    llm_expert_async_read_completion completion;
    const auto retry = run(retry_reader, completion);
    GGML_ASSERT(completion.result == llm_expert_async_result::ready && completion.bytes_completed == 8);
    GGML_ASSERT(retry.interrupted_reads_retried == 1 && retry.would_block_reads_retried == 1);
    GGML_ASSERT(retry.short_positive_reads == 1);

    scripted_async_reader error_reader;
    error_reader.actions = { scripted_async_reader::action::error };
    const auto hard_error = run(error_reader, completion);
    GGML_ASSERT(completion.result == llm_expert_async_result::invalid && completion.native_error == EIO);
    GGML_ASSERT(hard_error.read_bytes_completed == 0);

    scripted_async_reader cancel_reader;
    cancel_reader.actions = { scripted_async_reader::action::block_then_would_block };
    auto cfg = config(8);
    cfg.read_override_for_testing = &cancel_reader;
    llm_expert_async_transport transport(cfg);
    std::array<uint8_t, 8> destination{};
    llm_expert_storage_read_operation read;
    read.native_handle = 17;
    read.file_offset = 0;
    read.byte_count = destination.size();
    read.segment_count = 1;
    read.segments[0] = { destination.data(), destination.size(), 0,
        llm_expert_storage_projection::up, llm_expert_storage_sidecar::weight };
    const llm_expert_async_operation_identity identity = {
        1, { 0, 2 }, 0, { 0, 4 }, llm_expert_readiness::host_ready,
        llm_expert_priority::demand_current_layer,
    };
    GGML_ASSERT(transport.submit_read_plan(identity, &read, 1) == llm_expert_async_result::ready);
    {
        std::unique_lock<std::mutex> lock(cancel_reader.mutex);
        cancel_reader.condition.wait(lock, [&] { return cancel_reader.entered; });
    }
    GGML_ASSERT(transport.cancel_read(identity.request) == llm_expert_async_result::ready);
    {
        std::lock_guard<std::mutex> lock(cancel_reader.mutex);
        cancel_reader.released = true;
        cancel_reader.condition.notify_all();
    }
    GGML_ASSERT(transport.wait_read(identity.request, completion) == llm_expert_async_result::closed);
    GGML_ASSERT(transport.release_read(identity.request) == llm_expert_async_result::ready);
    GGML_ASSERT(transport.diagnostics().read_requests_cancelled == 1);

    cancel_reader.actions.clear();
    cancel_reader.call = 0;
    destination.fill(0);
    const llm_expert_async_operation_identity retry_identity = {
        1, { 0, 3 }, 0, { 0, 4 }, llm_expert_readiness::host_ready,
        llm_expert_priority::demand_current_layer,
    };
    GGML_ASSERT(transport.submit_read_plan(retry_identity, &read, 1) == llm_expert_async_result::ready);
    GGML_ASSERT(transport.wait_read(retry_identity.request, completion) == llm_expert_async_result::ready);
    GGML_ASSERT(std::memcmp(destination.data(), cancel_reader.bytes.data(), destination.size()) == 0);
    GGML_ASSERT(transport.wait_read(identity.request, completion) == llm_expert_async_result::stale_generation);
    GGML_ASSERT(transport.release_read(retry_identity.request) == llm_expert_async_result::ready);
#endif
}

void test_shutdown_drains_inflight_read() {
#if defined(__linux__)
    scripted_async_reader reader;
    reader.actions = { scripted_async_reader::action::block_then_would_block };
    auto cfg = config(8);
    cfg.read_override_for_testing = &reader;
    llm_expert_async_transport transport(cfg);
    std::array<uint8_t, 8> destination{};
    llm_expert_storage_read_operation read;
    read.native_handle = 17;
    read.file_offset = 0;
    read.byte_count = destination.size();
    read.segment_count = 1;
    read.segments[0] = { destination.data(), destination.size(), 0,
        llm_expert_storage_projection::up, llm_expert_storage_sidecar::weight };
    const llm_expert_async_operation_identity identity = {
        1, { 0, 7 }, 0, { 0, 6 }, llm_expert_readiness::host_ready,
        llm_expert_priority::demand_current_layer,
    };
    GGML_ASSERT(transport.submit_read_plan(identity, &read, 1) == llm_expert_async_result::ready);
    {
        std::unique_lock<std::mutex> lock(reader.mutex);
        reader.condition.wait(lock, [&] { return reader.entered; });
    }
    bool shutdown_result = false;
    std::thread shutdown([&] { shutdown_result = transport.shutdown(); });
    while (!transport.diagnostics().admission_closed) std::this_thread::yield();
    {
        std::lock_guard<std::mutex> lock(reader.mutex);
        reader.released = true;
        reader.condition.notify_all();
    }
    shutdown.join();
    GGML_ASSERT(shutdown_result);
    const auto diagnostics = transport.diagnostics();
    GGML_ASSERT(diagnostics.admission_closed);
    GGML_ASSERT(diagnostics.active_read_requests == 0);
    GGML_ASSERT(diagnostics.active_operations == 0);
    GGML_ASSERT(diagnostics.read_requests_cancelled == 1);
#endif
}

void test_direct_requirements_fail_closed() {
#if defined(__linux__)
    struct outcome {
        llm_expert_async_result submission = llm_expert_async_result::invalid;
        llm_expert_async_read_completion completion;
        llm_expert_async_diagnostics diagnostics;
    };
    auto run = [](scripted_async_reader & reader, uint64_t source_size,
                  uint64_t requested_staging_bytes = 0,
                  intptr_t direct_native_handle = 18) {
        outcome result;
        auto cfg = config(8);
        cfg.read_override_for_testing = &reader;
        cfg.direct_io_requested = true;
        cfg.maximum_direct_alignment = 8;
        cfg.requested_staging_bytes = requested_staging_bytes;
        reader.source_size = source_size;
        llm_expert_async_transport transport(cfg);
        std::array<uint8_t, 8> destination{};
        llm_expert_storage_read_operation read;
        read.native_handle = 17;
        read.direct_native_handle = direct_native_handle;
        read.direct_alignment = 8;
        read.source_size = source_size;
        read.file_offset = 3;
        read.byte_count = destination.size();
        read.segment_count = 1;
        read.segments[0] = { destination.data(), destination.size(), 3,
            llm_expert_storage_projection::up, llm_expert_storage_sidecar::weight };
        const llm_expert_async_operation_identity identity = {
            1, { 0, 3 }, 0, { 0, 5 }, llm_expert_readiness::host_ready,
            llm_expert_priority::demand_current_layer,
        };
        result.submission = transport.submit_read_plan(identity, &read, 1);
        if (result.submission == llm_expert_async_result::ready) {
            GGML_ASSERT(transport.wait_read(identity.request, result.completion) ==
                result.completion.result);
        }
        result.diagnostics = transport.diagnostics();
        if (result.completion.result == llm_expert_async_result::ready &&
            result.submission == llm_expert_async_result::ready) {
            GGML_ASSERT(std::memcmp(destination.data(), reader.bytes.data() + 3, destination.size()) == 0);
        }
        if (result.submission == llm_expert_async_result::ready) {
            GGML_ASSERT(transport.release_read(identity.request) == llm_expert_async_result::ready);
        }
        return result;
    };

    scripted_async_reader aligned_reader;
    const auto aligned = run(aligned_reader, 64);
    GGML_ASSERT(aligned.submission == llm_expert_async_result::ready &&
        aligned.completion.result == llm_expert_async_result::ready);
    GGML_ASSERT(aligned.diagnostics.direct_read_operations == 1 &&
        aligned.diagnostics.direct_useful_bytes == 8);
    GGML_ASSERT(aligned.diagnostics.direct_aligned_bytes == 16 &&
        aligned.diagnostics.direct_scatter_bytes == 8);
    GGML_ASSERT(aligned.diagnostics.buffered_fallback_operations == 0);

    scripted_async_reader tail_reader;
    const auto tail = run(tail_reader, 11);
    GGML_ASSERT(tail.submission == llm_expert_async_result::ready &&
        tail.completion.result == llm_expert_async_result::ready);
    GGML_ASSERT(tail.diagnostics.direct_read_operations == 1 &&
        tail.diagnostics.direct_aligned_bytes == 11 &&
        tail.diagnostics.direct_eof_short_reads == 1 &&
        tail.diagnostics.direct_eof_shortfall_bytes == 5 &&
        tail.diagnostics.short_positive_reads == 0 &&
        tail.diagnostics.buffered_fallback_operations == 0);

    scripted_async_reader past_eof_reader;
    const auto past_eof = run(past_eof_reader, 10);
    GGML_ASSERT(past_eof.submission == llm_expert_async_result::invalid);
    GGML_ASSERT(past_eof.diagnostics.direct_read_operations == 0 &&
        past_eof.diagnostics.buffered_fallback_operations == 0);
    GGML_ASSERT((past_eof.diagnostics.fallback_reason_mask &
        fallback_bit(llm_expert_async_fallback_reason::direct_eof)) != 0);

    scripted_async_reader staging_reader;
    const auto staging = run(staging_reader, 64, 4);
    GGML_ASSERT(staging.submission == llm_expert_async_result::invalid);
    GGML_ASSERT(staging.diagnostics.buffered_fallback_operations == 0 &&
        staging.diagnostics.direct_staging_error == ENOBUFS);
    GGML_ASSERT((staging.diagnostics.fallback_reason_mask &
        fallback_bit(llm_expert_async_fallback_reason::direct_staging)) != 0);

    scripted_async_reader unavailable_reader;
    const auto unavailable = run(unavailable_reader, 64, 0, -1);
    GGML_ASSERT(unavailable.submission == llm_expert_async_result::invalid);
    GGML_ASSERT(unavailable.diagnostics.direct_read_operations == 0 &&
        unavailable.diagnostics.buffered_fallback_operations == 0);
    GGML_ASSERT((unavailable.diagnostics.fallback_reason_mask &
        fallback_bit(llm_expert_async_fallback_reason::direct_alignment)) != 0);

    scripted_async_reader capability_reader;
    capability_reader.error_code = EINVAL;
    capability_reader.actions = { scripted_async_reader::action::error };
    const auto capability = run(capability_reader, 64);
    GGML_ASSERT(capability.submission == llm_expert_async_result::ready &&
        capability.completion.result == llm_expert_async_result::invalid &&
        capability.completion.native_error == EINVAL);
    GGML_ASSERT(capability.diagnostics.direct_capability_retries == 0 &&
        capability.diagnostics.buffered_fallback_operations == 0);

    scripted_async_reader hard_error_reader;
    hard_error_reader.actions = { scripted_async_reader::action::error };
    const auto hard_error = run(hard_error_reader, 64);
    GGML_ASSERT(hard_error.submission == llm_expert_async_result::ready &&
        hard_error.completion.result == llm_expert_async_result::invalid &&
        hard_error.completion.native_error == EIO);
    GGML_ASSERT(hard_error.diagnostics.direct_capability_retries == 0 &&
        hard_error.diagnostics.buffered_fallback_operations == 0);
#endif
}

void test_direct_positional_workers_use_independent_staging() {
#if defined(__linux__)
    struct concurrent_reader final : llm_expert_async_read_override {
        int64_t read_at(intptr_t, void * destination, size_t byte_count,
                uint64_t file_offset, int &) noexcept override {
            {
                std::unique_lock<std::mutex> lock(mutex);
                arrived++;
                condition.notify_all();
                condition.wait(lock, [&] { return arrived == 2; });
            }
            auto * bytes = static_cast<uint8_t *>(destination);
            for (size_t index = 0; index < byte_count; ++index) {
                bytes[index] = uint8_t(file_offset + index);
            }
            {
                std::unique_lock<std::mutex> lock(mutex);
                filled++;
                condition.notify_all();
                condition.wait(lock, [&] { return filled == 2; });
            }
            return int64_t(byte_count);
        }

        std::mutex mutex;
        std::condition_variable condition;
        uint32_t arrived = 0;
        uint32_t filled = 0;
    } reader;

    auto cfg = config(8);
    cfg.read_override_for_testing = &reader;
    cfg.direct_io_requested = true;
    cfg.maximum_direct_alignment = 8;
    cfg.force_positional_reads = true;
    cfg.worker_count = 2;
    llm_expert_async_transport transport(cfg);
    std::array<std::array<uint8_t, 8>, 2> destinations{};
    for (uint32_t index = 0; index < 2; ++index) {
        llm_expert_storage_read_operation read;
        read.native_handle = 17;
        read.direct_native_handle = 18;
        read.direct_alignment = 8;
        read.source_size = 64;
        read.file_offset = index*16;
        read.byte_count = destinations[index].size();
        read.segment_count = 1;
        read.segments[0] = { destinations[index].data(), destinations[index].size(),
            read.file_offset, llm_expert_storage_projection::up,
            llm_expert_storage_sidecar::weight };
        const llm_expert_async_operation_identity identity = {
            1, { index, uint64_t(index) + 1 }, 0, { 0, int32_t(index) },
            llm_expert_readiness::host_ready, llm_expert_priority::demand_current_layer,
        };
        GGML_ASSERT(transport.submit_read_plan(identity, &read, 1) ==
            llm_expert_async_result::ready);
    }
    for (uint32_t index = 0; index < 2; ++index) {
        const llm_expert_request_handle handle = { index, uint64_t(index) + 1 };
        llm_expert_async_read_completion completion;
        GGML_ASSERT(transport.wait_read(handle, completion) == llm_expert_async_result::ready);
        for (size_t byte = 0; byte < destinations[index].size(); ++byte) {
            GGML_ASSERT(destinations[index][byte] == uint8_t(index*16 + byte));
        }
        GGML_ASSERT(transport.release_read(handle) == llm_expert_async_result::ready);
    }
    const auto diagnostics = transport.diagnostics();
    GGML_ASSERT(diagnostics.direct_read_operations == 2);
    GGML_ASSERT(diagnostics.staging_ceiling_bytes == 8U*1024U*1024U);
#endif
}

void test_direct_positional_eof_tail() {
#if defined(__linux__)
    char path[] = "/tmp/llama-expert-direct-tail-XXXXXX";
    const int native_handle = mkstemp(path);
    GGML_ASSERT(native_handle >= 0);
    const std::array<uint8_t, 11> source = { 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 42 };
    GGML_ASSERT(write(native_handle, source.data(), source.size()) == int64_t(source.size()));
    GGML_ASSERT(fsync(native_handle) == 0);
    const int direct_native_handle = open(path, O_RDONLY | O_DIRECT);
    const int direct_open_error = direct_native_handle < 0 ? errno : 0;
    GGML_ASSERT(unlink(path) == 0);
    if (direct_native_handle < 0 &&
            (direct_open_error == EINVAL || direct_open_error == ENOTSUP ||
                direct_open_error == EOPNOTSUPP)) {
        GGML_ASSERT(close(native_handle) == 0);
        return;
    }
    GGML_ASSERT(direct_native_handle >= 0);

    auto cfg = config(8);
    cfg.direct_io_requested = true;
    cfg.maximum_direct_alignment = 4096;
    cfg.requested_staging_bytes = 4096;
    cfg.force_positional_reads = true;
    llm_expert_async_transport transport(cfg);
    std::array<uint8_t, 8> destination{};
    llm_expert_storage_read_operation read;
    read.native_handle = native_handle;
    read.direct_native_handle = direct_native_handle;
    read.direct_alignment = 4096;
    read.source_size = source.size();
    read.file_offset = 3;
    read.byte_count = destination.size();
    read.segment_count = 1;
    read.segments[0] = { destination.data(), destination.size(), read.file_offset,
        llm_expert_storage_projection::up, llm_expert_storage_sidecar::weight };
    const llm_expert_async_operation_identity identity = {
        1, { 0, 1 }, 0, { 0, 3 }, llm_expert_readiness::host_ready,
        llm_expert_priority::demand_current_layer,
    };
    GGML_ASSERT(transport.submit_read_plan(identity, &read, 1) ==
        llm_expert_async_result::ready);
    llm_expert_async_read_completion completion;
    GGML_ASSERT(transport.wait_read(identity.request, completion) ==
        llm_expert_async_result::ready);
    GGML_ASSERT(std::equal(destination.begin(), destination.end(), source.begin() + 3));
    const auto diagnostics = transport.diagnostics();
    GGML_ASSERT(diagnostics.direct_read_operations == 1 &&
        diagnostics.direct_useful_bytes == destination.size() &&
        diagnostics.direct_aligned_bytes == source.size() &&
        diagnostics.direct_scatter_bytes == destination.size());
    GGML_ASSERT(diagnostics.direct_eof_short_reads == 1 &&
        diagnostics.direct_eof_shortfall_bytes == 4096 - source.size() &&
        diagnostics.short_positive_reads == 0 &&
        diagnostics.buffered_fallback_operations == 0);
    GGML_ASSERT((diagnostics.fallback_reason_mask &
        fallback_bit(llm_expert_async_fallback_reason::direct_eof)) == 0);
    GGML_ASSERT(transport.release_read(identity.request) == llm_expert_async_result::ready);
    GGML_ASSERT(transport.shutdown());
    GGML_ASSERT(close(direct_native_handle) == 0);
    GGML_ASSERT(close(native_handle) == 0);
#endif
}

void test_concurrent_bounded_access() {
    llm_expert_async_transport transport(config(8));
    std::vector<std::thread> threads;
    for (uint32_t index = 0; index < 8; ++index) {
        threads.emplace_back([&transport, index] {
            const llm_expert_async_operation_identity expected = {
                1, { index, uint64_t(index) + 1 }, index, { 0, int32_t(index) },
                llm_expert_readiness::host_ready, llm_expert_priority::demand_current_layer,
            };
            const auto operation = transport.reserve_operation(expected);
            GGML_ASSERT(operation.result == llm_expert_async_result::ready);
            llm_expert_async_operation_identity actual;
            GGML_ASSERT(transport.consume_completion_for_testing(operation.user_data, actual) ==
                llm_expert_async_result::ready);
            GGML_ASSERT(actual.request.slot == index);
        });
    }
    for (auto & thread : threads) {
        thread.join();
    }
    const auto diagnostics = transport.diagnostics();
    GGML_ASSERT(diagnostics.operations_reserved == 8);
    GGML_ASSERT(diagnostics.completions_consumed == 8);
    GGML_ASSERT(diagnostics.active_operations == 0);
}

void test_reversed_fake_cq_and_saturation() {
    llm_expert_async_transport transport(config(8));
    std::array<uint64_t, 16> tokens{};
    for (uint32_t index = 0; index < tokens.size(); ++index) {
        const llm_expert_async_operation_identity identity = {
            1, { index, uint64_t(index) + 1 }, index, { 0, int32_t(index) },
            llm_expert_readiness::host_ready, llm_expert_priority::demand_current_layer,
        };
        const auto operation = transport.reserve_operation(identity);
        GGML_ASSERT(operation.result == llm_expert_async_result::ready);
        tokens[index] = operation.user_data;
    }
    const llm_expert_async_operation_identity overflow = {
        1, { 16, 17 }, 16, { 0, 16 }, llm_expert_readiness::host_ready,
        llm_expert_priority::demand_current_layer,
    };
    GGML_ASSERT(transport.reserve_operation(overflow).result == llm_expert_async_result::busy);
    for (size_t index = tokens.size(); index-- > 0;) {
        llm_expert_async_operation_identity identity;
        GGML_ASSERT(transport.consume_completion_for_testing(tokens[index], identity) ==
            llm_expert_async_result::ready);
        GGML_ASSERT(identity.request.slot == index);
    }
    const auto diagnostics = transport.diagnostics();
    GGML_ASSERT(diagnostics.peak_active_operations == tokens.size());
    GGML_ASSERT(diagnostics.active_operations == 0);
}

} // namespace

int main() {
    test_configuration();
    test_ring_layout_validation();
    test_user_data_and_fake_completion();
    test_bounded_trace_and_shutdown();
    test_generation_exhaustion();
    test_worker_read_and_drain();
    test_nonblocking_read_poll();
    test_bounded_positional_worker_parallelism();
    test_deferred_multi_request_ring_batch();
    test_partial_ring_submission_falls_back_after_quiescence();
    test_file_registration_or_explicit_fallback();
    test_native_ring_cancel_and_shutdown_drain();
    test_native_ring_cqe_failure_matrix();
    test_model_owner_drains_before_provider_destination_destroy();
    test_fallback_retry_error_and_cancel_races();
    test_shutdown_drains_inflight_read();
    test_direct_requirements_fail_closed();
    test_direct_positional_workers_use_independent_staging();
    test_direct_positional_eof_tail();
    test_concurrent_bounded_access();
    test_reversed_fake_cq_and_saturation();
    return 0;
}
