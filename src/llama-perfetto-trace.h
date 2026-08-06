#pragma once

#include <cstddef>
#include <cstdint>

#if defined(LLAMA_PERFETTO)

#include <perfetto.h>

PERFETTO_DEFINE_CATEGORIES(
    perfetto::Category("k3.request").SetDescription("Model, context, request, token, and teardown lifetimes"),
    perfetto::Category("k3.route").SetDescription("Router selection and publication"),
    perfetto::Category("k3.provider").SetDescription("Expert provider planning, binding, readiness, and release"),
    perfetto::Category("k3.scheduler").SetDescription("Expert single-flight scheduling and cancellation"),
    perfetto::Category("k3.cache.hot").SetDescription("Hot expert cache state transitions"),
    perfetto::Category("k3.cache.cold").SetDescription("Cold expert cache state transitions"),
    perfetto::Category("k3.storage").SetDescription("Expert storage planning and reads"),
    perfetto::Category("k3.transfer").SetDescription("Staging and host-to-device transfers"),
    perfetto::Category("k3.graph").SetDescription("Graph construction and execution"),
    perfetto::Category("k3.cuda").SetDescription("Correlated CUPTI activity"),
    perfetto::Category("k3.resource").SetDescription("Bounded queue and cache resource counters"),
    perfetto::Category("k3.lifecycle").SetDescription("Cancellation, drain, surrender, and unload"));

#define LLM_EXPERT_TRACE_SCOPE(category, name, ...) TRACE_EVENT(category, name, ##__VA_ARGS__)
#define LLM_EXPERT_TRACE_INSTANT(category, name, ...) TRACE_EVENT_INSTANT(category, name, ##__VA_ARGS__)
#define LLM_EXPERT_TRACE_ASYNC_BEGIN(category, name, id, ...) TRACE_EVENT_BEGIN(category, name, perfetto::Track(id), ##__VA_ARGS__)
#define LLM_EXPERT_TRACE_ASYNC_END(category, id, ...) TRACE_EVENT_END(category, perfetto::Track(id), ##__VA_ARGS__)
#define LLM_EXPERT_TRACE_ASYNC_BEGIN_AT(category, name, id, timestamp_ns, ...) TRACE_EVENT_BEGIN(category, name, perfetto::Track(id), uint64_t(timestamp_ns), ##__VA_ARGS__)
#define LLM_EXPERT_TRACE_ASYNC_END_AT(category, id, timestamp_ns, ...) TRACE_EVENT_END(category, perfetto::Track(id), uint64_t(timestamp_ns), ##__VA_ARGS__)
#define LLM_EXPERT_TRACE_FLOW_BEGIN(category, name, id, ...) TRACE_EVENT_INSTANT(category, name, perfetto::Flow::ProcessScoped(id), ##__VA_ARGS__)
#define LLM_EXPERT_TRACE_FLOW_END(category, name, id, ...) TRACE_EVENT_INSTANT(category, name, perfetto::TerminatingFlow::ProcessScoped(id), ##__VA_ARGS__)
#define LLM_EXPERT_TRACE_COUNTER(category, name, id, value) TRACE_COUNTER(category, perfetto::CounterTrack(name, id), value)
#define LLM_EXPERT_TRACE_CONCAT_INNER(a, b) a##b
#define LLM_EXPERT_TRACE_CONCAT(a, b) LLM_EXPERT_TRACE_CONCAT_INNER(a, b)
#define LLM_EXPERT_TRACE_CUDA_SCOPE(id) llm_perfetto_correlation_scope LLM_EXPERT_TRACE_CONCAT(llm_perfetto_correlation_, __LINE__)(llm_perfetto_trace_is_active() ? (id) : 0)

#else

#define LLM_EXPERT_TRACE_SCOPE(category, name, ...) do { } while (false)
#define LLM_EXPERT_TRACE_INSTANT(category, name, ...) do { } while (false)
#define LLM_EXPERT_TRACE_ASYNC_BEGIN(category, name, id, ...) do { } while (false)
#define LLM_EXPERT_TRACE_ASYNC_END(category, id, ...) do { } while (false)
#define LLM_EXPERT_TRACE_ASYNC_BEGIN_AT(category, name, id, timestamp_ns, ...) do { } while (false)
#define LLM_EXPERT_TRACE_ASYNC_END_AT(category, id, timestamp_ns, ...) do { } while (false)
#define LLM_EXPERT_TRACE_FLOW_BEGIN(category, name, id, ...) do { } while (false)
#define LLM_EXPERT_TRACE_FLOW_END(category, name, id, ...) do { } while (false)
#define LLM_EXPERT_TRACE_COUNTER(category, name, id, value) do { } while (false)
#define LLM_EXPERT_TRACE_CUDA_SCOPE(id) do { } while (false)

#endif

enum class llm_perfetto_trace_domain : uint8_t {
    request = 1,
    token = 2,
    graph = 3,
    flight = 4,
    storage = 5,
    transfer = 6,
    cuda = 7,
    resource = 8,
};

constexpr uint64_t llm_perfetto_trace_id(llm_perfetto_trace_domain domain, uint64_t value) noexcept {
    return (uint64_t(domain) << 56U) | (value & UINT64_C(0x00ffffffffffffff));
}

constexpr uint64_t llm_perfetto_trace_pair_id(
        llm_perfetto_trace_domain domain, uint32_t major, uint32_t minor) noexcept {
    return (uint64_t(domain) << 56U) | ((uint64_t(major) & UINT64_C(0x0fffffff)) << 28U) |
        (uint64_t(minor) & UINT64_C(0x0fffffff));
}

constexpr uint64_t llm_perfetto_trace_operation_id(
        llm_perfetto_trace_domain domain, uint32_t slot, uint32_t generation, uint32_t operation) noexcept {
    return (uint64_t(domain) << 56U) | ((uint64_t(slot) & UINT64_C(0xffff)) << 40U) |
        ((uint64_t(generation) & UINT64_C(0xffffffff)) << 8U) | (uint64_t(operation) & UINT64_C(0xff));
}

struct llm_perfetto_trace_config {
    uint32_t cuda_device = 0;
    uint32_t producer_shmem_kib = 32768;
    uint64_t cupti_retained_bytes = UINT64_C(256)*1024U*1024U;
};

struct llm_perfetto_trace_diagnostics {
    bool compiled = false;
    bool initialized = false;
    bool track_event_active = false;
    bool cupti_capable = false;
    bool cupti_active = false;
    bool shutdown = false;
    uint32_t perfetto_sessions_started = 0;
    uint32_t perfetto_sessions_stopped = 0;
    uint32_t perfetto_redundant_starts = 0;
    uint32_t perfetto_redundant_stops = 0;
    uint32_t cupti_version = 0;
    uint64_t clock_start_ns = 0;
    uint64_t clock_stop_ns = 0;
    uint64_t cupti_errors = 0;
    uint64_t cupti_records = 0;
    uint64_t cupti_dropped_records = 0;
    uint64_t cupti_retained_bytes = 0;
    uint64_t cupti_retained_capacity_bytes = 0;
    uint64_t cupti_peak_buffer_bytes = 0;
    uint64_t cupti_peak_total_bytes = 0;
    uint64_t cupti_unknown_timestamps = 0;
    uint64_t cupti_unmatched_correlations = 0;
};

#if defined(LLAMA_PERFETTO)

bool llm_perfetto_trace_initialize_system(
    const llm_perfetto_trace_config & config, char * error, size_t error_capacity) noexcept;
bool llm_perfetto_trace_wait_until_active(uint32_t timeout_ms, char * error, size_t error_capacity) noexcept;
bool llm_perfetto_trace_wait_until_inactive(uint32_t timeout_ms, char * error, size_t error_capacity) noexcept;
bool llm_perfetto_trace_request_stop(char * error, size_t error_capacity) noexcept;
bool llm_perfetto_trace_shutdown(char * error, size_t error_capacity) noexcept;
bool llm_perfetto_trace_is_active() noexcept;
uint64_t llm_perfetto_trace_next_id(llm_perfetto_trace_domain domain) noexcept;
bool llm_perfetto_trace_cuda_smoke() noexcept;
llm_perfetto_trace_diagnostics llm_perfetto_trace_get_diagnostics() noexcept;

class llm_perfetto_correlation_scope {
public:
    explicit llm_perfetto_correlation_scope(uint64_t id) noexcept;
    ~llm_perfetto_correlation_scope();

    llm_perfetto_correlation_scope(const llm_perfetto_correlation_scope &) = delete;
    llm_perfetto_correlation_scope & operator=(const llm_perfetto_correlation_scope &) = delete;

private:
    uint64_t id = 0;
    bool pushed = false;
};

#else

inline bool llm_perfetto_trace_is_active() noexcept {
    return false;
}

inline uint64_t llm_perfetto_trace_next_id(llm_perfetto_trace_domain) noexcept {
    return 0;
}

inline llm_perfetto_trace_diagnostics llm_perfetto_trace_get_diagnostics() noexcept {
    return {};
}

#endif
