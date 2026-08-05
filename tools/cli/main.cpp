#if defined(LLAMA_PERFETTO)
#include "llama-perfetto-trace.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#endif

int llama_cli(int argc, char ** argv);

int main(int argc, char ** argv) {
#if defined(LLAMA_PERFETTO)
    const char * requested = std::getenv("LLAMA_PERFETTO_CAPTURE");
    const bool trace = requested != nullptr && std::strcmp(requested, "1") == 0;
    char trace_error[256] = {};
    if (trace) {
        const llm_perfetto_trace_config config;
        if (!llm_perfetto_trace_initialize_system(config, trace_error, sizeof(trace_error)) ||
            !llm_perfetto_trace_wait_until_active(30000, trace_error, sizeof(trace_error))) {
            std::fprintf(stderr, "llama-cli: Perfetto/CUPTI activation failed: %s\n", trace_error);
            return 90;
        }
    }
#endif
    const int result = llama_cli(argc, argv);
#if defined(LLAMA_PERFETTO)
    if (trace && (!llm_perfetto_trace_request_stop(trace_error, sizeof(trace_error)) ||
            !llm_perfetto_trace_wait_until_inactive(30000, trace_error, sizeof(trace_error)) ||
            !llm_perfetto_trace_shutdown(trace_error, sizeof(trace_error)))) {
        std::fprintf(stderr, "llama-cli: Perfetto/CUPTI closeout failed: %s\n", trace_error);
        return 90;
    }
#endif
    return result;
}
