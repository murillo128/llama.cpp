#include "llama-perfetto-trace.h"

#include <cstdint>
#include <cstdlib>

int main() {
    int instant = 0;
    int counter = 0;
    int async_begin = 0;
    int async_end = 0;
    int flow_begin = 0;
    int flow_end = 0;
    int cuda_scope = 0;
    LLM_EXPERT_TRACE_INSTANT("k3.provider", "inactive_argument", "value", ++instant);
    LLM_EXPERT_TRACE_COUNTER("k3.resource", "inactive_counter", 1, ++counter);
    LLM_EXPERT_TRACE_ASYNC_BEGIN("k3.storage", "inactive_async", ++async_begin, "value", 1);
    LLM_EXPERT_TRACE_ASYNC_END("k3.storage", 1, "value", ++async_end);
    LLM_EXPERT_TRACE_FLOW_BEGIN("k3.scheduler", "inactive_flow", ++flow_begin, "value", 1);
    LLM_EXPERT_TRACE_FLOW_END("k3.scheduler", "inactive_flow", 1, "value", ++flow_end);
    LLM_EXPERT_TRACE_CUDA_SCOPE(++cuda_scope);
    if (instant != 0) return 10;
    if (counter != 0) return 11;
    if (async_begin != 0) return 12;
    if (async_end != 0) return 13;
    if (flow_begin != 0) return 14;
    if (flow_end != 0) return 15;
    if (cuda_scope != 0) return 16;

    if (llm_perfetto_trace_id(llm_perfetto_trace_domain::request, 17) != UINT64_C(0x0100000000000011)) return 2;
    if (llm_perfetto_trace_pair_id(llm_perfetto_trace_domain::storage, 3, 5) != UINT64_C(0x0500000030000005)) return 3;
    const auto diagnostics = llm_perfetto_trace_get_diagnostics();
#if defined(LLAMA_PERFETTO)
    if (!diagnostics.compiled) return 4;
#else
    if (diagnostics.compiled) return 5;
#endif
    if (diagnostics.initialized || diagnostics.track_event_active || diagnostics.cupti_active) return 6;
    if (diagnostics.cupti_records != 0 || diagnostics.cupti_dropped_records != 0) return 7;
#if defined(LLAMA_PERFETTO)
    if (std::getenv("LLAMA_PERFETTO_SYSTEM_TEST") != nullptr) {
        char error[256] = {};
        llm_perfetto_trace_config config;
        config.cupti_retained_bytes = UINT64_C(16)*1024U*1024U;
        if (!llm_perfetto_trace_initialize_system(config, error, sizeof(error))) return 20;
        if (!llm_perfetto_trace_wait_until_active(15000, error, sizeof(error))) return 21;
        if (!llm_perfetto_trace_cuda_smoke()) return 22;
        if (!llm_perfetto_trace_wait_until_inactive(15000, error, sizeof(error))) return 28;
        const auto captured = llm_perfetto_trace_get_diagnostics();
        if (captured.perfetto_sessions_started != 1 || captured.perfetto_sessions_stopped != 1 ||
            captured.cupti_records == 0 || captured.cupti_dropped_records != 0 ||
            captured.cupti_unknown_timestamps != 0 || captured.cupti_errors != 0) return 29;
        if (!llm_perfetto_trace_shutdown(error, sizeof(error))) return 30;
        const auto stopped = llm_perfetto_trace_get_diagnostics();
        if (!stopped.shutdown || stopped.cupti_retained_bytes != 0) return 31;
    }
#endif
    return 0;
}
