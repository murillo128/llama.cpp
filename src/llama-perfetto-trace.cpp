#include "llama-perfetto-trace.h"

#include <cupti.h>
#include <cupti_activity.h>
#include <cuda_runtime_api.h>

#include <atomic>
#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <climits>
#include <condition_variable>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <new>
#include <vector>
#include <time.h>
#include <unistd.h>

PERFETTO_TRACK_EVENT_STATIC_STORAGE();

namespace {

constexpr uint64_t k_cupti_retained_bytes_max = UINT64_C(256)*1024U*1024U;
constexpr size_t k_cupti_buffer_bytes = 1024U*1024U;
constexpr uint64_t k_cupti_activity_buffer_reserve_bytes = UINT64_C(32)*1024U*1024U;
std::atomic<bool> g_trace_active { false };
std::atomic<uint64_t> g_next_trace_id { 1 };
std::atomic<uint64_t> g_cupti_active_buffer_bytes { 0 };
std::atomic<uint64_t> g_cupti_buffer_limit { k_cupti_retained_bytes_max };
std::atomic<uint64_t> g_cupti_retained_capacity_bytes { 0 };

enum class cupti_record_kind : uint8_t {
    runtime,
    driver,
    kernel,
    memcpy,
    memset,
    synchronization,
    external_correlation,
};

struct retained_cupti_record {
    cupti_record_kind kind = cupti_record_kind::runtime;
    uint64_t start = 0;
    uint64_t end = 0;
    uint64_t queued = 0;
    uint64_t submitted = 0;
    uint64_t external_id = 0;
    uint64_t grid_id = 0;
    uint64_t bytes = 0;
    uint32_t correlation_id = 0;
    uint32_t runtime_correlation_id = 0;
    uint32_t device_id = 0;
    uint32_t context_id = 0;
    uint32_t stream_id = 0;
    uint32_t thread_id = 0;
    uint32_t subtype = 0;
    uint32_t return_value = 0;
    int32_t grid_x = 0;
    int32_t grid_y = 0;
    int32_t grid_z = 0;
    int32_t block_x = 0;
    int32_t block_y = 0;
    int32_t block_z = 0;
    std::array<char, 128> name {};
};

uint64_t monotonic_raw_ns() noexcept {
    timespec value {};
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &value) != 0) return 0;
    return uint64_t(value.tv_sec)*UINT64_C(1000000000) + uint64_t(value.tv_nsec);
}

uint64_t CUPTIAPI cupti_timestamp() {
    return monotonic_raw_ns();
}

void set_error(char * destination, size_t capacity, const char * message) noexcept {
    if (destination == nullptr || capacity == 0) return;
    std::snprintf(destination, capacity, "%s", message == nullptr ? "unknown tracing error" : message);
}

struct trace_state;

class trace_observer final : public perfetto::TrackEventSessionObserver {
public:
    explicit trace_observer(trace_state & state) : state(state) {}
    void OnStart(const perfetto::DataSourceBase::StartArgs &) override;
    void OnStop(const perfetto::DataSourceBase::StopArgs &) override;

private:
    trace_state & state;
};

struct trace_state {
    std::mutex mutex;
    std::condition_variable changed;
    llm_perfetto_trace_config config;
    llm_perfetto_trace_diagnostics diagnostics;
    trace_observer observer { *this };
    bool observer_registered = false;
    bool callback_failed = false;
    char callback_error[192] = {};
    std::vector<retained_cupti_record> cupti_records;
    std::array<CUpti_ActivityKind, 7> enabled_kinds {};
    size_t enabled_kind_count = 0;
};

trace_state & state() {
    static trace_state value;
    return value;
}

void record_cupti_error(trace_state & value, const char * operation, CUptiResult result) noexcept {
    const char * detail = nullptr;
    (void) cuptiGetResultString(result, &detail);
    value.diagnostics.cupti_errors++;
    value.callback_failed = true;
    std::snprintf(value.callback_error, sizeof(value.callback_error), "%s: %s (%u)", operation,
        detail == nullptr ? "CUPTI error" : detail, unsigned(result));
}

void record_callback_failure(trace_state & value, const char * message) noexcept {
    value.diagnostics.cupti_errors++;
    value.callback_failed = true;
    std::snprintf(value.callback_error, sizeof(value.callback_error), "%s", message);
}

bool valid_interval(uint64_t start, uint64_t end) noexcept {
    return start != 0 && end != 0 && start != CUPTI_TIMESTAMP_UNKNOWN &&
        end != CUPTI_TIMESTAMP_UNKNOWN && end >= start;
}

perfetto::TraceTimestamp monotonic_raw_timestamp(uint64_t value) noexcept {
    return {
        static_cast<uint32_t>(perfetto::protos::pbzero::BUILTIN_CLOCK_MONOTONIC_RAW),
        value,
    };
}

void retain_cupti_record(trace_state & value, const retained_cupti_record & record) noexcept {
    if (value.cupti_records.size() == value.cupti_records.capacity()) {
        record_callback_failure(value, "CUPTI retained-record bound exhausted");
        return;
    }
    const uint64_t next_bytes = uint64_t(value.cupti_records.size() + 1)*sizeof(retained_cupti_record);
    try {
        value.cupti_records.push_back(record);
        value.diagnostics.cupti_records++;
        value.diagnostics.cupti_retained_bytes = next_bytes;
    } catch (...) {
        record_callback_failure(value, "CUPTI retained-record allocation failed");
    }
}

void CUPTIAPI cupti_buffer_requested(uint8_t ** buffer, size_t * size, size_t * max_records) {
    *buffer = nullptr;
    *size = 0;
    *max_records = 0;
    const uint64_t buffer_limit = g_cupti_buffer_limit.load(std::memory_order_acquire);
    if (buffer_limit < k_cupti_buffer_bytes) {
        auto & value = state();
        std::lock_guard<std::mutex> lock(value.mutex);
        record_callback_failure(value, "CUPTI shared memory budget exhausted");
        return;
    }
    const uint64_t prior = g_cupti_active_buffer_bytes.fetch_add(k_cupti_buffer_bytes, std::memory_order_acq_rel);
    if (prior > buffer_limit - k_cupti_buffer_bytes) {
        g_cupti_active_buffer_bytes.fetch_sub(k_cupti_buffer_bytes, std::memory_order_acq_rel);
        auto & value = state();
        std::lock_guard<std::mutex> lock(value.mutex);
        record_callback_failure(value, "CUPTI activity-buffer bound exhausted");
        return;
    }
    void * allocation = nullptr;
    if (posix_memalign(&allocation, 8, k_cupti_buffer_bytes) != 0) {
        g_cupti_active_buffer_bytes.fetch_sub(k_cupti_buffer_bytes, std::memory_order_acq_rel);
        auto & value = state();
        std::lock_guard<std::mutex> lock(value.mutex);
        record_callback_failure(value, "CUPTI activity-buffer allocation failed");
        return;
    }
    *buffer = static_cast<uint8_t *>(allocation);
    *size = k_cupti_buffer_bytes;
    auto & value = state();
    std::lock_guard<std::mutex> lock(value.mutex);
    value.diagnostics.cupti_peak_buffer_bytes = std::max(
        value.diagnostics.cupti_peak_buffer_bytes, prior + k_cupti_buffer_bytes);
    value.diagnostics.cupti_peak_total_bytes = std::max(value.diagnostics.cupti_peak_total_bytes,
        g_cupti_retained_capacity_bytes.load(std::memory_order_acquire) + prior + k_cupti_buffer_bytes);
}

void CUPTIAPI cupti_buffer_completed(
        CUcontext context, uint32_t stream_id, uint8_t * buffer, size_t, size_t valid_size) {
    auto & value = state();
    if (valid_size != 0) {
        std::lock_guard<std::mutex> lock(value.mutex);
        CUpti_Activity * activity = nullptr;
        while (true) {
            const CUptiResult next = cuptiActivityGetNextRecord(buffer, valid_size, &activity);
            if (next == CUPTI_ERROR_MAX_LIMIT_REACHED) break;
            if (next != CUPTI_SUCCESS) {
                record_cupti_error(value, "cuptiActivityGetNextRecord", next);
                break;
            }
            retained_cupti_record record;
            switch (activity->kind) {
                case CUPTI_ACTIVITY_KIND_RUNTIME:
                case CUPTI_ACTIVITY_KIND_DRIVER: {
                    const auto * source = reinterpret_cast<const CUpti_ActivityAPI *>(activity);
                    record.kind = activity->kind == CUPTI_ACTIVITY_KIND_RUNTIME ?
                        cupti_record_kind::runtime : cupti_record_kind::driver;
                    record.start = source->start;
                    record.end = source->end;
                    record.correlation_id = source->correlationId;
                    record.thread_id = source->threadId;
                    record.subtype = uint32_t(source->cbid);
                    record.return_value = source->returnValue;
                    break;
                }
                case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
                    const auto * source = reinterpret_cast<const CUpti_ActivityKernel9 *>(activity);
                    record.kind = cupti_record_kind::kernel;
                    record.start = source->start;
                    record.end = source->end;
                    record.queued = source->queued;
                    record.submitted = source->submitted;
                    record.correlation_id = source->correlationId;
                    record.device_id = source->deviceId;
                    record.context_id = source->contextId;
                    record.stream_id = source->streamId;
                    record.grid_id = uint64_t(source->gridId);
                    record.grid_x = source->gridX;
                    record.grid_y = source->gridY;
                    record.grid_z = source->gridZ;
                    record.block_x = source->blockX;
                    record.block_y = source->blockY;
                    record.block_z = source->blockZ;
                    std::snprintf(record.name.data(), record.name.size(), "%s",
                        source->name == nullptr ? "unknown" : source->name);
                    break;
                }
                case CUPTI_ACTIVITY_KIND_MEMCPY: {
                    const auto * source = reinterpret_cast<const CUpti_ActivityMemcpy6 *>(activity);
                    record.kind = cupti_record_kind::memcpy;
                    record.start = source->start;
                    record.end = source->end;
                    record.correlation_id = source->correlationId;
                    record.runtime_correlation_id = source->runtimeCorrelationId;
                    record.device_id = source->deviceId;
                    record.context_id = source->contextId;
                    record.stream_id = source->streamId;
                    record.subtype = source->copyKind;
                    record.bytes = source->bytes;
                    break;
                }
                case CUPTI_ACTIVITY_KIND_MEMSET: {
                    const auto * source = reinterpret_cast<const CUpti_ActivityMemset4 *>(activity);
                    record.kind = cupti_record_kind::memset;
                    record.start = source->start;
                    record.end = source->end;
                    record.correlation_id = source->correlationId;
                    record.device_id = source->deviceId;
                    record.context_id = source->contextId;
                    record.stream_id = source->streamId;
                    record.bytes = source->bytes;
                    record.subtype = source->value;
                    break;
                }
                case CUPTI_ACTIVITY_KIND_SYNCHRONIZATION: {
                    const auto * source = reinterpret_cast<const CUpti_ActivitySynchronization2 *>(activity);
                    record.kind = cupti_record_kind::synchronization;
                    record.start = source->start;
                    record.end = source->end;
                    record.correlation_id = source->correlationId;
                    record.context_id = source->contextId;
                    record.stream_id = source->streamId;
                    record.subtype = uint32_t(source->type);
                    record.return_value = source->returnValue;
                    break;
                }
                case CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION: {
                    const auto * source = reinterpret_cast<const CUpti_ActivityExternalCorrelation *>(activity);
                    if (source->externalKind != CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0) continue;
                    record.kind = cupti_record_kind::external_correlation;
                    record.correlation_id = source->correlationId;
                    record.external_id = source->externalId;
                    break;
                }
                default:
                    continue;
            }
            if (record.kind != cupti_record_kind::external_correlation &&
                !valid_interval(record.start, record.end)) {
                value.diagnostics.cupti_unknown_timestamps++;
            }
            retain_cupti_record(value, record);
        }
        size_t dropped = 0;
        const CUptiResult dropped_result = cuptiActivityGetNumDroppedRecords(context, stream_id, &dropped);
        if (dropped_result != CUPTI_SUCCESS) {
            record_cupti_error(value, "cuptiActivityGetNumDroppedRecords", dropped_result);
        } else {
            value.diagnostics.cupti_dropped_records += dropped;
        }
    }
    std::free(buffer);
    g_cupti_active_buffer_bytes.fetch_sub(k_cupti_buffer_bytes, std::memory_order_acq_rel);
}

uint64_t external_id_for(
        const std::vector<std::pair<uint32_t, uint64_t>> & mappings, uint32_t correlation_id) noexcept {
    const auto found = std::lower_bound(mappings.begin(), mappings.end(), correlation_id,
        [](const auto & item, uint32_t value) { return item.first < value; });
    return found != mappings.end() && found->first == correlation_id ? found->second : 0;
}

void emit_retained_cupti_records(trace_state & value) {
    std::vector<std::pair<uint32_t, uint64_t>> mappings;
    mappings.reserve(value.cupti_records.size());
    for (const auto & record : value.cupti_records) {
        if (record.kind == cupti_record_kind::external_correlation) {
            mappings.emplace_back(record.correlation_id, record.external_id);
        }
    }
    std::sort(mappings.begin(), mappings.end());
    mappings.erase(std::unique(mappings.begin(), mappings.end(), [](const auto & lhs, const auto & rhs) {
        return lhs.first == rhs.first;
    }), mappings.end());
    std::stable_sort(value.cupti_records.begin(), value.cupti_records.end(), [](const auto & lhs, const auto & rhs) {
        const uint64_t lhs_time = lhs.kind == cupti_record_kind::external_correlation ? UINT64_MAX : lhs.start;
        const uint64_t rhs_time = rhs.kind == cupti_record_kind::external_correlation ? UINT64_MAX : rhs.start;
        if (lhs_time != rhs_time) return lhs_time < rhs_time;
        if (lhs.context_id != rhs.context_id) return lhs.context_id < rhs.context_id;
        if (lhs.stream_id != rhs.stream_id) return lhs.stream_id < rhs.stream_id;
        return uint8_t(lhs.kind) < uint8_t(rhs.kind);
    });

    for (const auto & record : value.cupti_records) {
        if (record.kind == cupti_record_kind::external_correlation || !valid_interval(record.start, record.end)) continue;
        const uint64_t external_id = external_id_for(mappings, record.correlation_id);
        if (record.correlation_id != 0 && external_id == 0) value.diagnostics.cupti_unmatched_correlations++;
        const uint64_t track_id = llm_perfetto_trace_operation_id(llm_perfetto_trace_domain::cuda,
            record.kind == cupti_record_kind::runtime || record.kind == cupti_record_kind::driver ?
                record.thread_id : record.context_id,
            record.stream_id, uint32_t(record.kind) + 1);
        switch (record.kind) {
            case cupti_record_kind::runtime: {
                const perfetto::NamedTrack track(perfetto::StaticString("CUDA runtime API"), track_id);
                TRACE_EVENT_BEGIN("k3.cuda", "runtime_api", track, monotonic_raw_timestamp(record.start),
                    "correlation_id", record.correlation_id, "application_correlation_id", external_id,
                    "cbid", record.subtype, "thread_id", record.thread_id);
                TRACE_EVENT_END(
                    "k3.cuda", track, monotonic_raw_timestamp(record.end), "return_value", record.return_value);
                break;
            }
            case cupti_record_kind::driver: {
                const perfetto::NamedTrack track(perfetto::StaticString("CUDA driver API"), track_id);
                TRACE_EVENT_BEGIN("k3.cuda", "driver_api", track, monotonic_raw_timestamp(record.start),
                    "correlation_id", record.correlation_id, "application_correlation_id", external_id,
                    "cbid", record.subtype, "thread_id", record.thread_id);
                TRACE_EVENT_END(
                    "k3.cuda", track, monotonic_raw_timestamp(record.end), "return_value", record.return_value);
                break;
            }
            case cupti_record_kind::kernel: {
                const perfetto::NamedTrack track(perfetto::StaticString("CUDA kernels"), track_id);
                TRACE_EVENT_BEGIN("k3.cuda", "kernel", track, monotonic_raw_timestamp(record.start),
                    "correlation_id", record.correlation_id, "application_correlation_id", external_id, "kernel_name",
                    record.name.data(), "device_id", record.device_id, "context_id", record.context_id,
                    "stream_id", record.stream_id, "grid_id", record.grid_id, "grid_x", record.grid_x,
                    "grid_y", record.grid_y, "grid_z", record.grid_z, "block_x", record.block_x,
                    "block_y", record.block_y, "block_z", record.block_z, "queued_ns", record.queued,
                    "submitted_ns", record.submitted);
                TRACE_EVENT_END("k3.cuda", track, monotonic_raw_timestamp(record.end));
                if (record.queued != CUPTI_TIMESTAMP_UNKNOWN && record.submitted != CUPTI_TIMESTAMP_UNKNOWN &&
                    record.queued != 0 && record.submitted >= record.queued) {
                    const uint64_t latency_track_id = llm_perfetto_trace_operation_id(llm_perfetto_trace_domain::cuda,
                        record.context_id, record.correlation_id, 0xfe);
                    const perfetto::NamedTrack latency_track(
                        perfetto::StaticString("CUDA kernel launch latency"), latency_track_id);
                    TRACE_EVENT_BEGIN("k3.cuda", "kernel_queued", latency_track,
                        monotonic_raw_timestamp(record.queued), "correlation_id", record.correlation_id,
                        "application_correlation_id", external_id);
                    TRACE_EVENT_END("k3.cuda", latency_track, monotonic_raw_timestamp(record.submitted));
                }
                break;
            }
            case cupti_record_kind::memcpy: {
                const perfetto::NamedTrack track(perfetto::StaticString("CUDA memcpy"), track_id);
                TRACE_EVENT_BEGIN("k3.cuda", "memcpy", track, monotonic_raw_timestamp(record.start),
                    "correlation_id", record.correlation_id, "runtime_correlation_id", record.runtime_correlation_id,
                    "application_correlation_id", external_id, "copy_kind", record.subtype,
                    "bytes", record.bytes, "device_id", record.device_id, "context_id", record.context_id,
                    "stream_id", record.stream_id);
                TRACE_EVENT_END("k3.cuda", track, monotonic_raw_timestamp(record.end));
                break;
            }
            case cupti_record_kind::memset: {
                const perfetto::NamedTrack track(perfetto::StaticString("CUDA memset"), track_id);
                TRACE_EVENT_BEGIN("k3.cuda", "memset", track, monotonic_raw_timestamp(record.start),
                    "correlation_id", record.correlation_id, "application_correlation_id", external_id,
                    "value", record.subtype,
                    "bytes", record.bytes, "device_id", record.device_id, "context_id", record.context_id,
                    "stream_id", record.stream_id);
                TRACE_EVENT_END("k3.cuda", track, monotonic_raw_timestamp(record.end));
                break;
            }
            case cupti_record_kind::synchronization: {
                const perfetto::NamedTrack track(perfetto::StaticString("CUDA synchronization"), track_id);
                TRACE_EVENT_BEGIN("k3.cuda", "synchronization", track, monotonic_raw_timestamp(record.start),
                    "correlation_id", record.correlation_id, "application_correlation_id", external_id,
                    "sync_type", record.subtype,
                    "context_id", record.context_id, "stream_id", record.stream_id);
                TRACE_EVENT_END(
                    "k3.cuda", track, monotonic_raw_timestamp(record.end), "return_value", record.return_value);
                break;
            }
            case cupti_record_kind::external_correlation:
                break;
        }
    }
}

bool finalize_cupti_activity(trace_state & value) {
    g_trace_active.store(false, std::memory_order_release);
    CUptiResult first_error = cuptiActivityFlushAll(0);
    for (size_t index = 0; index < value.enabled_kind_count; ++index) {
        const CUptiResult result = cuptiActivityDisable(value.enabled_kinds[index]);
        if (first_error == CUPTI_SUCCESS && result != CUPTI_SUCCESS) first_error = result;
    }
    const CUptiResult final_flush = cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
    if (first_error == CUPTI_SUCCESS && final_flush != CUPTI_SUCCESS) first_error = final_flush;
    const CUptiResult sync_disable = cuptiActivityEnableAllSyncRecords(0);
    if (first_error == CUPTI_SUCCESS && sync_disable != CUPTI_SUCCESS) first_error = sync_disable;
    (void) cuptiActivityEnableLatencyTimestamps(0);

    std::lock_guard<std::mutex> lock(value.mutex);
    if (!value.diagnostics.track_event_active || !value.diagnostics.cupti_active) {
        record_callback_failure(value, "CUPTI finalization requires one active trace session");
        return false;
    }
    value.enabled_kind_count = 0;
    if (first_error != CUPTI_SUCCESS) record_cupti_error(value, "CUPTI activity shutdown", first_error);
    if (g_cupti_active_buffer_bytes.load(std::memory_order_acquire) != 0) {
        record_callback_failure(value, "CUPTI activity buffers remained active after flush");
    }
    try {
        emit_retained_cupti_records(value);
    } catch (...) {
        record_callback_failure(value, "CUPTI TrackEvent emission failed");
    }
    value.diagnostics.clock_stop_ns = monotonic_raw_ns();
    LLM_EXPERT_TRACE_INSTANT("k3.lifecycle", "trace_session_stop", "clock_monotonic_raw_ns",
        value.diagnostics.clock_stop_ns, "cupti_errors", value.diagnostics.cupti_errors,
        "cupti_records", value.diagnostics.cupti_records,
        "cupti_dropped_records", value.diagnostics.cupti_dropped_records,
        "cupti_retained_capacity_bytes", value.diagnostics.cupti_retained_capacity_bytes,
        "cupti_peak_total_bytes", value.diagnostics.cupti_peak_total_bytes,
        "cupti_unknown_timestamps", value.diagnostics.cupti_unknown_timestamps,
        "cupti_unmatched_correlations", value.diagnostics.cupti_unmatched_correlations);
    value.diagnostics.cupti_active = false;
    return !value.callback_failed;
}

void trace_observer::OnStart(const perfetto::DataSourceBase::StartArgs &) {
    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.diagnostics.track_event_active || state.diagnostics.perfetto_sessions_started != 0) {
        state.callback_failed = true;
        std::snprintf(state.callback_error, sizeof(state.callback_error), "%s", "multiple tracing sessions are unsupported");
        state.changed.notify_all();
        return;
    }

    uint32_t version = 0;
    CUptiResult result = cuptiGetVersion(&version);
    if (result != CUPTI_SUCCESS) {
        record_cupti_error(state, "cuptiGetVersion", result);
        state.changed.notify_all();
        return;
    }
    state.diagnostics.cupti_version = version;
    state.diagnostics.cupti_capable = true;

    result = cuptiActivityRegisterTimestampCallback(cupti_timestamp);
    if (result != CUPTI_SUCCESS) {
        record_cupti_error(state, "cuptiActivityRegisterTimestampCallback", result);
        state.changed.notify_all();
        return;
    }
    result = cuptiActivityEnableLatencyTimestamps(1);
    if (result != CUPTI_SUCCESS) {
        record_cupti_error(state, "cuptiActivityEnableLatencyTimestamps", result);
        state.changed.notify_all();
        return;
    }

    result = cuptiActivityRegisterCallbacks(cupti_buffer_requested, cupti_buffer_completed);
    if (result != CUPTI_SUCCESS) {
        record_cupti_error(state, "cuptiActivityRegisterCallbacks", result);
        state.changed.notify_all();
        return;
    }
    result = cuptiActivityEnableAllSyncRecords(1);
    if (result != CUPTI_SUCCESS) {
        record_cupti_error(state, "cuptiActivityEnableAllSyncRecords", result);
        state.changed.notify_all();
        return;
    }

    state.cupti_records.clear();
    try {
        const uint64_t buffer_reserve = std::min<uint64_t>(
            k_cupti_activity_buffer_reserve_bytes, state.config.cupti_retained_bytes/2);
        const uint64_t retained_budget = state.config.cupti_retained_bytes - buffer_reserve;
        state.cupti_records.reserve(size_t(retained_budget/sizeof(retained_cupti_record)));
    } catch (...) {
        record_callback_failure(state, "CUPTI retained-record reserve failed");
        (void) cuptiActivityEnableAllSyncRecords(0);
        state.changed.notify_all();
        return;
    }
    const uint64_t retained_capacity_bytes =
        uint64_t(state.cupti_records.capacity())*sizeof(retained_cupti_record);
    if (retained_capacity_bytes > state.config.cupti_retained_bytes - k_cupti_buffer_bytes) {
        record_callback_failure(state, "CUPTI retained allocation exceeds shared memory budget");
        std::vector<retained_cupti_record>().swap(state.cupti_records);
        (void) cuptiActivityEnableAllSyncRecords(0);
        state.changed.notify_all();
        return;
    }
    state.diagnostics.cupti_retained_capacity_bytes = retained_capacity_bytes;
    state.diagnostics.cupti_peak_total_bytes = retained_capacity_bytes;
    g_cupti_retained_capacity_bytes.store(retained_capacity_bytes, std::memory_order_release);
    g_cupti_buffer_limit.store(
        state.config.cupti_retained_bytes - retained_capacity_bytes, std::memory_order_release);

    state.enabled_kinds = {
        CUPTI_ACTIVITY_KIND_RUNTIME,
        CUPTI_ACTIVITY_KIND_DRIVER,
        CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL,
        CUPTI_ACTIVITY_KIND_MEMCPY,
        CUPTI_ACTIVITY_KIND_MEMSET,
        CUPTI_ACTIVITY_KIND_SYNCHRONIZATION,
        CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION,
    };
    state.enabled_kind_count = 0;
    for (const auto kind : state.enabled_kinds) {
        result = cuptiActivityEnable(kind);
        if (result != CUPTI_SUCCESS) {
            record_cupti_error(state, "cuptiActivityEnable", result);
            for (size_t index = 0; index < state.enabled_kind_count; ++index) {
                (void) cuptiActivityDisable(state.enabled_kinds[index]);
            }
            state.enabled_kind_count = 0;
            (void) cuptiActivityEnableAllSyncRecords(0);
            state.changed.notify_all();
            return;
        }
        state.enabled_kind_count++;
    }
    state.diagnostics.track_event_active = true;
    state.diagnostics.cupti_active = true;
    g_trace_active.store(true, std::memory_order_release);
    state.diagnostics.clock_start_ns = monotonic_raw_ns();
    state.diagnostics.perfetto_sessions_started++;
    LLM_EXPERT_TRACE_INSTANT("k3.lifecycle", "trace_session_start", "clock_monotonic_raw_ns",
        state.diagnostics.clock_start_ns, "cupti_version", state.diagnostics.cupti_version,
        "cuda_device", state.config.cuda_device);
    state.changed.notify_all();
}

void trace_observer::OnStop(const perfetto::DataSourceBase::StopArgs & args) {
    auto stop = args.HandleStopAsynchronously();
    bool needs_finalization = false;
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        needs_finalization = state.diagnostics.cupti_active;
    }
    if (needs_finalization) (void) finalize_cupti_activity(state);
    perfetto::TrackEvent::Flush();
    if (stop) stop();
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        if (state.diagnostics.track_event_active) {
            state.diagnostics.track_event_active = false;
            state.diagnostics.perfetto_sessions_stopped++;
        } else {
            record_callback_failure(state, "trace session stopped before activation");
        }
        state.changed.notify_all();
    }
}

bool wait_for_state(
        bool active, uint32_t timeout_ms, char * error, size_t error_capacity) noexcept {
    auto & value = state();
    std::unique_lock<std::mutex> lock(value.mutex);
    const bool ready = value.changed.wait_for(lock, std::chrono::milliseconds(timeout_ms), [&] {
        return value.diagnostics.track_event_active == active || value.callback_failed;
    });
    if (!ready) {
        set_error(error, error_capacity, active ? "timed out waiting for Perfetto TrackEvent activation" :
            "timed out waiting for Perfetto TrackEvent stop");
        return false;
    }
    if (value.callback_failed) {
        set_error(error, error_capacity, value.callback_error);
        return false;
    }
    return true;
}

} // namespace

bool llm_perfetto_trace_initialize_system(
        const llm_perfetto_trace_config & config, char * error, size_t error_capacity) noexcept {
    auto & value = state();
    std::lock_guard<std::mutex> lock(value.mutex);
    if (value.diagnostics.initialized || value.diagnostics.shutdown) {
        set_error(error, error_capacity, "Perfetto tracing may be initialized only once per process");
        return false;
    }
    if (config.cuda_device != 0 || config.producer_shmem_kib == 0 || config.producer_shmem_kib > 32768 ||
        config.producer_shmem_kib % 4 != 0 || config.cupti_retained_bytes < k_cupti_buffer_bytes ||
        config.cupti_retained_bytes > k_cupti_retained_bytes_max) {
        set_error(error, error_capacity, "invalid bounded Perfetto/CUPTI configuration");
        return false;
    }

    value.config = config;
    value.diagnostics = {};
    value.diagnostics.compiled = true;
    value.callback_failed = false;
    value.callback_error[0] = '\0';
    if (!perfetto::TrackEvent::AddSessionObserver(&value.observer)) {
        set_error(error, error_capacity, "Perfetto TrackEvent observer capacity exhausted");
        return false;
    }
    value.observer_registered = true;

    perfetto::TracingInitArgs args;
    args.backends = perfetto::kSystemBackend;
    args.enable_system_consumer = false;
    args.supports_multiple_data_source_instances = false;
    args.use_monotonic_raw_clock = true;
    args.shmem_size_hint_kb = config.producer_shmem_kib;
    perfetto::Tracing::Initialize(args);
    perfetto::TrackEvent::Register();
    value.diagnostics.initialized = true;
    return true;
}

bool llm_perfetto_trace_wait_until_active(
        uint32_t timeout_ms, char * error, size_t error_capacity) noexcept {
    return wait_for_state(true, timeout_ms, error, error_capacity);
}

bool llm_perfetto_trace_wait_until_inactive(
        uint32_t timeout_ms, char * error, size_t error_capacity) noexcept {
    return wait_for_state(false, timeout_ms, error, error_capacity);
}

bool llm_perfetto_trace_request_stop(char * error, size_t error_capacity) noexcept {
    auto & value = state();
    {
        std::lock_guard<std::mutex> lock(value.mutex);
        if (!value.diagnostics.track_event_active || value.callback_failed) {
            set_error(error, error_capacity, value.callback_failed ? value.callback_error :
                "Perfetto stop trigger requires one active session");
            return false;
        }
    }
    if (!finalize_cupti_activity(value)) {
        std::lock_guard<std::mutex> lock(value.mutex);
        set_error(error, error_capacity, value.callback_error);
        return false;
    }
    perfetto::TrackEvent::Flush();
    const char * stop_fd_value = std::getenv("LLAMA_PERFETTO_STOP_FD");
    if (stop_fd_value != nullptr) {
        char * end = nullptr;
        errno = 0;
        const long stop_fd = std::strtol(stop_fd_value, &end, 10);
        if (errno != 0 || end == stop_fd_value || *end != '\0' || stop_fd < 0 || stop_fd > INT_MAX) {
            set_error(error, error_capacity, "invalid Perfetto stop descriptor");
            return false;
        }
        const uint8_t marker = 1;
        ssize_t result = -1;
        do {
            result = ::write(int(stop_fd), &marker, sizeof(marker));
        } while (result < 0 && errno == EINTR);
        if (result != 1) {
            set_error(error, error_capacity, "Perfetto stop descriptor write failed");
            return false;
        }
        return true;
    }
    try {
        perfetto::Tracing::ActivateTriggers({ "k3.stop" }, 5000);
    } catch (...) {
        set_error(error, error_capacity, "Perfetto stop trigger activation failed");
        return false;
    }
    return true;
}

bool llm_perfetto_trace_shutdown(char * error, size_t error_capacity) noexcept {
    auto & value = state();
    {
        std::lock_guard<std::mutex> lock(value.mutex);
        if (!value.diagnostics.initialized || value.diagnostics.track_event_active ||
            value.diagnostics.perfetto_sessions_started != value.diagnostics.perfetto_sessions_stopped) {
            set_error(error, error_capacity, "Perfetto shutdown requires one fully stopped session");
            return false;
        }
        if (value.callback_failed || value.diagnostics.cupti_errors != 0 ||
            value.diagnostics.cupti_dropped_records != 0 ||
            value.diagnostics.cupti_unknown_timestamps != 0) {
            set_error(error, error_capacity, value.callback_failed ? value.callback_error : "CUPTI trace integrity failure");
            return false;
        }
        perfetto::TrackEvent::RemoveSessionObserver(&value.observer);
        value.observer_registered = false;
    }
    perfetto::Tracing::Shutdown();
    {
        std::lock_guard<std::mutex> lock(value.mutex);
        std::vector<retained_cupti_record>().swap(value.cupti_records);
        value.diagnostics.cupti_retained_bytes = 0;
        value.diagnostics.cupti_retained_capacity_bytes = 0;
        g_cupti_retained_capacity_bytes.store(0, std::memory_order_release);
        value.diagnostics.shutdown = true;
    }
    return true;
}

llm_perfetto_trace_diagnostics llm_perfetto_trace_get_diagnostics() noexcept {
    auto & value = state();
    std::lock_guard<std::mutex> lock(value.mutex);
    auto result = value.diagnostics;
    result.compiled = true;
    return result;
}

bool llm_perfetto_trace_is_active() noexcept {
    return g_trace_active.load(std::memory_order_acquire);
}

uint64_t llm_perfetto_trace_next_id(llm_perfetto_trace_domain domain) noexcept {
    return llm_perfetto_trace_id(domain, g_next_trace_id.fetch_add(1, std::memory_order_relaxed));
}

bool llm_perfetto_trace_cuda_smoke() noexcept {
    const uint64_t correlation_id = llm_perfetto_trace_id(llm_perfetto_trace_domain::request, 1);
    LLM_EXPERT_TRACE_SCOPE("k3.request", "system_test_request", "request_id", 1);
    LLM_EXPERT_TRACE_CUDA_SCOPE(correlation_id);
    void * device = nullptr;
    uint32_t source[64] = {};
    uint32_t destination[64] = {};
    if (cudaMalloc(&device, sizeof(source)) != cudaSuccess) return false;
    bool success = cudaMemset(device, 0, sizeof(source)) == cudaSuccess &&
        cudaMemcpy(device, source, sizeof(source), cudaMemcpyHostToDevice) == cudaSuccess &&
        cudaMemcpy(destination, device, sizeof(destination), cudaMemcpyDeviceToHost) == cudaSuccess &&
        cudaDeviceSynchronize() == cudaSuccess;
    if (cudaFree(device) != cudaSuccess) success = false;
    return success;
}

llm_perfetto_correlation_scope::llm_perfetto_correlation_scope(uint64_t id) noexcept : id(id) {
    if (id == 0) return;
    const CUptiResult result = cuptiActivityPushExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, id);
    if (result == CUPTI_SUCCESS) {
        pushed = true;
        return;
    }
    auto & value = state();
    std::lock_guard<std::mutex> lock(value.mutex);
    record_cupti_error(value, "cuptiActivityPushExternalCorrelationId", result);
}

llm_perfetto_correlation_scope::~llm_perfetto_correlation_scope() {
    if (!pushed) return;
    uint64_t popped = 0;
    const CUptiResult result = cuptiActivityPopExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, &popped);
    if (result == CUPTI_SUCCESS && popped == id) return;
    auto & value = state();
    std::lock_guard<std::mutex> lock(value.mutex);
    if (result != CUPTI_SUCCESS) {
        record_cupti_error(value, "cuptiActivityPopExternalCorrelationId", result);
    } else {
        value.diagnostics.cupti_errors++;
        value.callback_failed = true;
        std::snprintf(value.callback_error, sizeof(value.callback_error), "%s", "CUPTI external correlation stack mismatch");
    }
}
