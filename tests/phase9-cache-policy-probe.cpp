#include "llama.h"
#include "llama-context.h"
#include "llama-expert-storage.h"
#include "llama-expert-async-io.h"
#include "llama-expert-weight-provider.h"
#include "llama-model.h"
#include "llama-perfetto-trace.h"
#include "llama-cpp.h"

#include "ggml-backend.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <map>
#include <string>
#include <sys/resource.h>
#include <vector>

using json = nlohmann::json;

namespace {

#if defined(LLAMA_PERFETTO)
class perfetto_evidence_owner {
public:
    perfetto_evidence_owner() {
        const char * requested = std::getenv("LLAMA_PERFETTO_CAPTURE");
        enabled = requested != nullptr && std::strcmp(requested, "1") == 0;
        if (!enabled) return;
        llm_perfetto_trace_config config;
        char error[256] = {};
        if (!llm_perfetto_trace_initialize_system(config, error, sizeof(error)) ||
            !llm_perfetto_trace_wait_until_active(30000, error, sizeof(error))) {
            throw std::runtime_error(std::string("Perfetto/CUPTI activation failed: ") + error);
        }
    }

    ~perfetto_evidence_owner() {
        if (!enabled) return;
        char error[256] = {};
        const bool stopped = llm_perfetto_trace_request_stop(error, sizeof(error)) &&
            llm_perfetto_trace_wait_until_inactive(30000, error, sizeof(error)) &&
            llm_perfetto_trace_shutdown(error, sizeof(error));
        if (!stopped) {
            std::fprintf(stderr, "phase9-cache-policy-probe: Perfetto/CUPTI closeout failed: %s\n", error);
            std::fflush(stderr);
            std::_Exit(90);
        }
    }

private:
    bool enabled = false;
};
#endif

struct arguments {
    std::string model;
    std::string output;
    std::string mode = "cold";
    std::string prompt = "According to all known laws";
    std::string hot_policy = "LRU";
    std::string cold_policy = "LRU";
    std::string scope = "GLOBAL";
    std::string admission = "ALWAYS";
    std::string miss_policy = "CPU_FALLBACK";
    uint32_t hot_slots = 16;
    uint64_t cold_bytes = 64U*1024U*1024U;
    uint64_t ring_bytes = 16U*1024U*1024U;
    uint32_t queue_depth = 0;
    uint32_t trace_capacity = 0;
    uint32_t ratio = 7500;
    uint32_t window = 1024;
    uint32_t aging = 1024;
    uint32_t n_ctx = 64;
    uint32_t n_batch = 64;
    uint32_t n_ubatch = 1;
    int max_generate = 2;
    bool background = false;
    bool observe_routes = true;
    std::string transport = "BUFFERED";
    std::string config_source = "EXPLICIT";
    std::string integrity = "NONE";
};

bool parse_u64(const char * text, uint64_t & value) {
    char * end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (end == text || *end != '\0') return false;
    value = parsed;
    return true;
}

bool parse_u32(const char * text, uint32_t & value) {
    uint64_t parsed = 0;
    if (!parse_u64(text, parsed) || parsed > UINT32_MAX) return false;
    value = uint32_t(parsed);
    return true;
}

std::string token_piece(const llama_vocab * vocab, llama_token token) {
    std::string piece(16, '\0');
    int size = llama_token_to_piece(vocab, token, piece.data(), piece.size(), 0, true);
    if (size < 0) {
        piece.resize(size_t(-size));
        size = llama_token_to_piece(vocab, token, piece.data(), piece.size(), 0, true);
    }
    if (size < 0) {
        return {};
    }
    piece.resize(size_t(size));
    return piece;
}

bool parse_arguments(int argc, char ** argv, arguments & result) {
    for (int index = 1; index < argc; ++index) {
        if (index + 1 >= argc) return false;
        const std::string option = argv[index];
        const char * value = argv[++index];
        if (option == "--model") result.model = value;
        else if (option == "--output") result.output = value;
        else if (option == "--mode") result.mode = value;
        else if (option == "--prompt") result.prompt = value;
        else if (option == "--hot-policy") result.hot_policy = value;
        else if (option == "--cold-policy") result.cold_policy = value;
        else if (option == "--scope") result.scope = value;
        else if (option == "--admission") result.admission = value;
        else if (option == "--miss-policy") result.miss_policy = value;
        else if (option == "--hot-slots") { if (!parse_u32(value, result.hot_slots)) return false; }
        else if (option == "--cold-bytes") { if (!parse_u64(value, result.cold_bytes)) return false; }
        else if (option == "--ring-bytes") { if (!parse_u64(value, result.ring_bytes)) return false; }
        else if (option == "--queue-depth") { if (!parse_u32(value, result.queue_depth)) return false; }
        else if (option == "--trace-capacity") { if (!parse_u32(value, result.trace_capacity)) return false; }
        else if (option == "--ratio") { if (!parse_u32(value, result.ratio)) return false; }
        else if (option == "--window") { if (!parse_u32(value, result.window)) return false; }
        else if (option == "--aging") { if (!parse_u32(value, result.aging)) return false; }
        else if (option == "--n-ctx") { if (!parse_u32(value, result.n_ctx)) return false; }
        else if (option == "--n-batch") { if (!parse_u32(value, result.n_batch)) return false; }
        else if (option == "--n-ubatch") { if (!parse_u32(value, result.n_ubatch)) return false; }
        else if (option == "--max-generate") {
            uint32_t parsed = 0; if (!parse_u32(value, parsed) || parsed == 0 || parsed > 128) return false;
            result.max_generate = int(parsed);
        } else if (option == "--background") {
            if (std::string(value) != "0" && std::string(value) != "1") return false;
            result.background = std::string(value) == "1";
        } else if (option == "--observe-routes") {
            if (std::string(value) != "0" && std::string(value) != "1") return false;
            result.observe_routes = std::string(value) == "1";
        } else if (option == "--transport") {
            result.transport = value;
            if (result.transport != "POSITIONAL" && result.transport != "BUFFERED" &&
                result.transport != "DIRECT_IO") return false;
        } else if (option == "--config-source") {
            result.config_source = value;
            if (result.config_source != "EXPLICIT" && result.config_source != "NULL") return false;
        } else if (option == "--integrity") {
            result.integrity = value;
            if (result.integrity != "NONE" && result.integrity != "FNV64_END_TO_END") return false;
        } else return false;
    }
    return !result.model.empty() && !result.output.empty() && result.hot_slots > 0 && result.n_ctx > 0 &&
        result.n_batch > 0 && result.n_ubatch > 0 && result.n_ubatch <= result.n_batch &&
        (result.mode == "disabled" || result.mode == "hot" || result.mode == "cold");
}

llama_expert_cache_policy policy_enum(const std::string & value) {
    if (value == "LRU") return LLAMA_EXPERT_CACHE_POLICY_LRU;
    if (value == "LFRU") return LLAMA_EXPERT_CACHE_POLICY_LFRU;
    if (value == "SLRU") return LLAMA_EXPERT_CACHE_POLICY_SLRU;
    if (value == "LFU_AGING") return LLAMA_EXPERT_CACHE_POLICY_LFU_AGING;
    throw std::runtime_error("unknown policy");
}

llama_expert_cache_policy_scope scope_enum(const std::string & value) {
    if (value == "GLOBAL") return LLAMA_EXPERT_CACHE_POLICY_SCOPE_GLOBAL;
    if (value == "PER_LAYER") return LLAMA_EXPERT_CACHE_POLICY_SCOPE_PER_LAYER;
    throw std::runtime_error("unknown scope");
}

llama_expert_cache_policy_config make_config(
        const std::string & policy, const arguments & args, bool hot) {
    const auto selected = policy_enum(policy);
    const bool slru = selected == LLAMA_EXPERT_CACHE_POLICY_SLRU;
    const bool lfu = selected == LLAMA_EXPERT_CACHE_POLICY_LFU_AGING;
    const bool window = hot && slru && args.admission == "FREQUENCY_WINDOW";
    return {
        LLAMA_EXPERT_CACHE_POLICY_VERSION_1,
        sizeof(llama_expert_cache_policy_config),
        selected,
        scope_enum(args.scope),
        slru ? args.ratio : 0,
        window ? LLAMA_EXPERT_CACHE_ADMISSION_FREQUENCY_WINDOW : LLAMA_EXPERT_CACHE_ADMISSION_ALWAYS,
        window ? args.window : 0,
        lfu ? args.aging : 0,
        {},
    };
}

llama_expert_miss_policy miss_policy_enum(const std::string & value) {
    if (value == "PROMOTE_AND_GPU") return LLAMA_EXPERT_MISS_POLICY_PROMOTE_AND_GPU;
    if (value == "CPU_FALLBACK") return LLAMA_EXPERT_MISS_POLICY_CPU_FALLBACK;
    if (value == "AUTO") return LLAMA_EXPERT_MISS_POLICY_AUTO;
    throw std::runtime_error("unknown miss policy");
}

const char * policy_name(llama_expert_cache_policy value) {
    static const char * names[] = { "LRU", "LFRU", "SLRU", "LFU_AGING" };
    return names[unsigned(value)];
}

const char * scope_name(llama_expert_cache_policy_scope value) {
    return value == LLAMA_EXPERT_CACHE_POLICY_SCOPE_GLOBAL ? "GLOBAL" : "PER_LAYER";
}

const char * admission_name(llama_expert_cache_admission value) {
    return value == LLAMA_EXPERT_CACHE_ADMISSION_ALWAYS ? "ALWAYS" : "FREQUENCY_WINDOW";
}

const char * integrity_mode_name(llm_expert_integrity_mode value) {
    return value == llm_expert_integrity_mode::none ? "none" : "fnv64_end_to_end";
}

const char * integrity_status_name(llm_expert_integrity_status value) {
    switch (value) {
        case llm_expert_integrity_status::not_checked: return "not_checked";
        case llm_expert_integrity_status::passed:      return "passed";
        case llm_expert_integrity_status::failed:      return "failed";
    }
    return "invalid";
}

json config_json(const llm_expert_cache_policy_diagnostics & diagnostics) {
    const auto & value = diagnostics.config;
    return {
        {"schema_version", "cache-policy-config-v1"}, {"policy", policy_name(value.policy)},
        {"scope", scope_name(value.scope)}, {"slru_protected_ratio_bps", value.slru_protected_ratio_bps},
        {"admission", admission_name(value.admission)}, {"admission_window_events", value.admission_window_events},
        {"lfu_aging_interval_events", value.lfu_aging_interval_events}, {"digest", value.digest},
    };
}

const char * event_name(llm_expert_cache_policy_event_type type) {
    static const char * names[] = { "REQUEST_BEGIN", "PHASE_TRANSITION", "DEMAND", "HIT",
        "VICTIM_SELECTED", "EVICT", "LOAD_BEGIN", "LOAD_COMPLETE", "LOAD_FAILED", "PIN",
        "UNPIN", "OPTIONAL_ADMISSION", "REQUEST_END", "RESET", "SURRENDER" };
    return names[unsigned(type)];
}

json event_json(const llm_expert_cache_policy_event & event) {
    return {
        {"schema_version", event.schema_version}, {"request_ordinal", event.request_ordinal},
        {"ubatch_ordinal", event.ubatch_ordinal},
        {"tier", event.tier == llm_expert_cache_policy_tier::hot ? "HOT" : "COLD"},
        {"type", event_name(event.type)}, {"event_sequence", event.event_sequence},
        {"demand_ordinal", event.demand_ordinal}, {"origin_operation_ordinal", event.origin_operation_ordinal},
        {"phase", event.phase == llm_expert_cache_policy_phase::prefill ? "PREFILL" : "DECODE"},
        {"layer", event.key.layer}, {"expert", event.key.expert},
        {"layout_class_id", event.key.layout_class_id},
        {"occurrence_count", event.occurrence_count}, {"logical_payload_bytes", event.logical_bundle_bytes},
        {"physical_slot_footprint_bytes", event.physical_slot_footprint_bytes},
        {"slot", event.slot == UINT32_MAX ? -1 : int64_t(event.slot)}, {"generation", event.generation},
        {"domain", event.domain == UINT32_MAX ? -1 : int64_t(event.domain)},
        {"eligible", event.eligible}, {"decision", event.decision}, {"reason", event.reason},
        {"state_digest", event.state_digest},
    };
}

json events_json(const std::vector<llm_expert_cache_policy_event> & events) {
    json result = json::array();
    for (const auto & event : events) result.push_back(event_json(event));
    return result;
}

struct route_record {
    uint64_t request = 0;
    uint64_t ubatch = 0;
    std::string phase;
    int32_t layer = -1;
    uint32_t n_tokens = 0;
    uint32_t n_expert_used = 0;
    std::vector<int32_t> ids;
    std::vector<float> weights;
};

struct route_capture { std::vector<route_record> records; };

bool capture_route(const llama_route_observation * observation, void * user_data) {
    auto & capture = *static_cast<route_capture *>(user_data);
    const size_t count = size_t(observation->n_tokens)*observation->n_expert_used;
    route_record record;
    record.request = observation->request_ordinal;
    record.ubatch = observation->ubatch_ordinal;
    record.phase = observation->phase == LLAMA_ROUTE_PHASE_PREFILL ? "PREFILL" : "DECODE";
    record.layer = observation->layer;
    record.n_tokens = observation->n_tokens;
    record.n_expert_used = observation->n_expert_used;
    if (count != 0) {
        record.ids.assign(observation->selected_experts, observation->selected_experts + count);
        record.weights.assign(observation->weights, observation->weights + count);
    }
    capture.records.push_back(std::move(record));
    return true;
}

uint64_t hash_bytes(uint64_t state, const void * data, size_t size) {
    const auto * bytes = static_cast<const uint8_t *>(data);
    for (size_t index = 0; index < size; ++index) {
        state ^= bytes[index];
        state *= 1099511628211ULL;
    }
    return state;
}

int finite_argmax(const float * logits, int count) {
    int result = -1;
    for (int index = 0; index < count; ++index) {
        if (!std::isfinite(logits[index])) return -1;
        if (result < 0 || logits[index] > logits[result]) result = index;
    }
    return result;
}

json route_json(const route_capture & capture) {
    json result = json::array();
    for (const auto & record : capture.records) {
        json weights = json::array();
        for (float value : record.weights) {
            uint32_t bits = 0;
            std::memcpy(&bits, &value, sizeof(bits));
            weights.push_back(bits);
        }
        result.push_back({
            {"request_ordinal", record.request}, {"ubatch_ordinal", record.ubatch},
            {"phase", record.phase}, {"layer", record.layer}, {"n_tokens", record.n_tokens},
            {"n_expert_used", record.n_expert_used}, {"selected_experts", record.ids},
            {"weight_f32_bits", weights},
        });
    }
    return result;
}

json diagnostics_json(const llm_expert_cache_policy_diagnostics & value) {
    return {
        {"events", value.events}, {"demands", value.demands}, {"hits", value.hits},
        {"misses", value.misses}, {"free_selections", value.free_selections},
        {"victim_selections", value.victim_selections}, {"mandatory_admissions", value.mandatory_admissions},
        {"optional_admission_accepts", value.optional_admission_accepts},
        {"optional_admission_rejects", value.optional_admission_rejects},
        {"transcript_records", value.transcript_records}, {"transcript_dropped", value.transcript_dropped},
        {"administration_actual_bytes", value.administration_actual_bytes}, {"state_digest", value.state_digest},
    };
}

json async_diagnostics_json(const llm_expert_async_diagnostics & value) {
    return {
        {"requested_sq_entries", value.requested_sq_entries}, {"requested_cq_entries", value.requested_cq_entries},
        {"actual_sq_entries", value.actual_sq_entries}, {"actual_cq_entries", value.actual_cq_entries},
        {"operation_capacity", value.operation_capacity}, {"trace_capacity", value.trace_capacity},
        {"active_operations", value.active_operations}, {"peak_active_operations", value.peak_active_operations},
        {"staging_ceiling_bytes", value.staging_ceiling_bytes},
        {"administration_bytes", value.administration_bytes}, {"transport_epoch", value.transport_epoch},
        {"fallback_reason_mask", value.fallback_reason_mask},
        {"fallback_diagnostics_emitted", value.fallback_diagnostics_emitted},
        {"operations_reserved", value.operations_reserved}, {"completions_consumed", value.completions_consumed},
        {"stale_completions", value.stale_completions}, {"trace_records", value.trace_records},
        {"trace_records_dropped", value.trace_records_dropped},
        {"read_requests_submitted", value.read_requests_submitted},
        {"read_requests_completed", value.read_requests_completed},
        {"read_requests_cancelled", value.read_requests_cancelled},
        {"read_operations_completed", value.read_operations_completed},
        {"read_bytes_completed", value.read_bytes_completed},
        {"read_queue_wait_samples", value.read_queue_wait_samples},
        {"read_queue_wait_us", value.read_queue_wait_us},
        {"read_queue_wait_max_us", value.read_queue_wait_max_us},
        {"integrity_mode", integrity_mode_name(value.integrity_mode)},
        {"integrity_digest_requests", value.integrity_digest_requests},
        {"integrity_digest_bytes", value.integrity_digest_bytes},
        {"synchronous_fallback_operations", value.synchronous_fallback_operations},
        {"interrupted_reads_retried", value.interrupted_reads_retried},
        {"would_block_reads_retried", value.would_block_reads_retried},
        {"short_positive_reads", value.short_positive_reads},
        {"direct_read_operations", value.direct_read_operations}, {"direct_useful_bytes", value.direct_useful_bytes},
        {"direct_aligned_bytes", value.direct_aligned_bytes}, {"direct_scatter_bytes", value.direct_scatter_bytes},
        {"buffered_fallback_operations", value.buffered_fallback_operations},
        {"buffered_fallback_bytes", value.buffered_fallback_bytes},
        {"direct_capability_retries", value.direct_capability_retries},
        {"ring_submissions", value.ring_submissions}, {"ring_completions", value.ring_completions},
        {"peak_sq_occupancy", value.peak_sq_occupancy}, {"peak_cq_occupancy", value.peak_cq_occupancy},
        {"ring_cancel_submissions", value.ring_cancel_submissions},
        {"ring_cancel_completions", value.ring_cancel_completions},
        {"ring_request_batches", value.ring_request_batches},
        {"peak_ring_batch_requests", value.peak_ring_batch_requests},
        {"cq_empty_waits", value.cq_empty_waits}, {"cq_empty_waits_after_cancel", value.cq_empty_waits_after_cancel},
        {"registered_file_count", value.registered_file_count}, {"file_registration_error", value.file_registration_error},
        {"registered_buffer_count", value.registered_buffer_count},
        {"registered_buffer_bytes", value.registered_buffer_bytes},
        {"buffer_registration_error", value.buffer_registration_error},
        {"direct_staging_error", value.direct_staging_error},
        {"active_read_requests", value.active_read_requests},
        {"peak_active_read_requests", value.peak_active_read_requests},
        {"linux_uapi", value.linux_uapi}, {"io_uring_enabled", value.io_uring_enabled},
        {"positional_reads_forced", value.positional_reads_forced},
        {"io_uring_setup_error", value.io_uring_setup_error}, {"io_uring_probe_error", value.io_uring_probe_error},
        {"io_uring_runtime_error", value.io_uring_runtime_error}, {"opcode_read", value.opcode_read},
        {"opcode_readv", value.opcode_readv}, {"opcode_async_cancel", value.opcode_async_cancel},
        {"opcode_read_fixed", value.opcode_read_fixed}, {"worker_started", value.worker_started},
        {"admission_closed", value.admission_closed},
    };
}

json async_read_intervals_json(const std::vector<llm_expert_async_read_interval> & intervals) {
    json result = json::array();
    for (const auto & interval : intervals) {
        json segments = json::array();
        for (uint32_t index = 0; index < interval.source_segment_count; ++index) {
            segments.push_back({
                {"file_offset", interval.source_segments[index].file_offset},
                {"byte_count", interval.source_segments[index].byte_count},
            });
        }
        result.push_back({
            {"transport_epoch", interval.flight.transport_epoch},
            {"request_slot", interval.flight.request_slot},
            {"request_generation", interval.flight.request_generation},
            {"layer", interval.flight.key.layer}, {"expert", interval.flight.key.expert},
            {"layout_class_id", interval.flight.layout_class_id},
            {"operation_index", interval.operation_index}, {"queued_us", interval.queued_us},
            {"started_us", interval.started_us}, {"submit_us", interval.submit_us},
            {"complete_us", interval.complete_us}, {"bytes", interval.bytes},
            {"useful_bytes", interval.useful_bytes},
            {"operation_file_offset", interval.operation_file_offset}, {"source_segments", segments},
        });
    }
    return result;
}

} // namespace

int main(int argc, char ** argv) {
    try {
        arguments args;
        if (!parse_arguments(argc, argv, args)) {
            std::fprintf(stderr,
                "usage: %s --model GGUF --output JSON [--mode disabled|hot|cold] "
                "[--integrity NONE|FNV64_END_TO_END (internal evidence only)] [policy/capacity options]\n",
                argv[0]);
            return 2;
        }
#if defined(_WIN32)
        if (_putenv_s("LLAMA_EXPERT_INTEGRITY_MODE", args.integrity.c_str()) != 0) return 2;
#else
        if (setenv("LLAMA_EXPERT_INTEGRITY_MODE", args.integrity.c_str(), 1) != 0) return 2;
#endif
#if defined(LLAMA_PERFETTO)
        perfetto_evidence_owner trace_owner;
#endif
        ggml_backend_load_all();
        if (ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU) == nullptr) {
            std::fprintf(stderr, "phase9-cache-policy-probe: CUDA/GPU backend required\n");
            return 2;
        }
        const auto hot_config = make_config(args.hot_policy, args, true);
        const auto cold_config = make_config(args.cold_policy, args, false);
        const llama_expert_auto_cost_model auto_cost = {
            LLAMA_EXPERT_AUTO_COST_MODEL_VERSION_1,
            sizeof(llama_expert_auto_cost_model),
            1000000, 1200000, 1, 1,
            500000, 600000, 1, 1,
            20000, 6000000000ULL, 1,
        };
        static const llama_model_tensor_buft_override overrides[] = {
            { "ffn_(gate|up|down)_exps\\.weight", ggml_backend_cpu_buffer_type() },
            { nullptr, nullptr },
        };
        auto model_params = llama_model_default_params();
        model_params.load_mode = args.transport == "DIRECT_IO" ? LLAMA_LOAD_MODE_DIRECT_IO : LLAMA_LOAD_MODE_MMAP;
        model_params.expert_io_queue_depth = args.queue_depth;
        model_params.expert_io_trace_capacity = args.trace_capacity;
        model_params.expert_io_force_positional_reads = args.transport == "POSITIONAL";
        model_params.n_gpu_layers = -1;
        model_params.tensor_buft_overrides = overrides;
        model_params.expert_weights_mode = args.mode == "disabled" ? LLAMA_EXPERT_WEIGHTS_MODE_DISABLED :
            args.mode == "cold" ? LLAMA_EXPERT_WEIGHTS_MODE_COLD_CACHE : LLAMA_EXPERT_WEIGHTS_MODE_HOT_CACHE;
        if (args.mode != "disabled") {
            model_params.expert_hot_cache_capacity = args.hot_slots;
            model_params.expert_hot_cache_policy = args.config_source == "NULL" ? nullptr : &hot_config;
        }
        if (args.mode == "cold") {
            model_params.expert_cold_cache_bytes = args.cold_bytes;
            model_params.expert_transfer_ring_bytes = args.ring_bytes;
            model_params.expert_miss_policy = miss_policy_enum(args.miss_policy);
            if (model_params.expert_miss_policy == LLAMA_EXPERT_MISS_POLICY_AUTO) {
                model_params.expert_auto_cost_model = &auto_cost;
            }
            model_params.expert_background_promotion = args.background;
            model_params.expert_cold_cache_policy = args.config_source == "NULL" ? nullptr : &cold_config;
        }
        llama_model_ptr model(llama_model_load_from_file(args.model.c_str(), model_params));
        if (!model) return 3;
        auto * provider = model->expert_weight_provider();
        if ((args.mode == "disabled") == (provider != nullptr)) return 4;
        const auto * vocab = llama_model_get_vocab(model.get());
        const std::string & prompt_text = args.prompt;
        const int prompt_count = -llama_tokenize(vocab, prompt_text.data(), prompt_text.size(), nullptr, 0, true, true);
        if (prompt_count <= 0) return 5;
        std::vector<llama_token> prompt(prompt_count);
        if (llama_tokenize(vocab, prompt_text.data(), prompt_text.size(), prompt.data(), prompt.size(), true, true) != prompt_count) return 5;
        auto context_params = llama_context_default_params();
        context_params.n_ctx = args.n_ctx;
        context_params.n_batch = args.n_batch;
        context_params.n_ubatch = args.n_ubatch;
        context_params.no_perf = false;
        llama_context_ptr context(llama_init_from_model(model.get(), context_params));
        if (!context) return 6;
        route_capture routes;
        if (args.observe_routes &&
            llama_set_route_observer(context.get(), capture_route, &routes) != LLAMA_ROUTE_OBSERVER_STATUS_OK) return 7;
        std::vector<llama_token> generated;
        std::string generated_text;
        std::vector<uint64_t> logits_digests;
        std::vector<uint64_t> latency_us;
        llama_batch batch = llama_batch_get_one(prompt.data(), prompt.size());
        const int n_vocab = llama_vocab_n_tokens(vocab);
        for (int step = 0; step < args.max_generate; ++step) {
            const auto phase = step == 0 ? LLAMA_ROUTE_PHASE_PREFILL : LLAMA_ROUTE_PHASE_DECODE;
            if (args.observe_routes &&
                llama_route_observer_begin(context.get(), uint64_t(step + 1), phase) != LLAMA_ROUTE_OBSERVER_STATUS_OK) return 8;
            const auto begin = std::chrono::steady_clock::now();
            const int decode_status = llama_decode(context.get(), batch);
            if (decode_status != 0) {
                if (provider == nullptr) {
                    std::fprintf(stderr, "phase9-cache-policy-probe: provider-disabled decode failed status=%d\n",
                        decode_status);
                    return 9;
                }
                const auto failed = provider->hot_cache_diagnostics();
                std::fprintf(stderr,
                    "phase9-cache-policy-probe: decode failed status=%d provider_error=%u remap_error=%u "
                    "hot_events=%llu hot_dropped=%llu cold_events=%llu cold_dropped=%llu\n",
                    decode_status, unsigned(failed.last_failure_error), unsigned(failed.last_remap_error),
                    (unsigned long long) failed.policy.events,
                    (unsigned long long) failed.policy.transcript_dropped,
                    (unsigned long long) failed.cold_policy.events,
                    (unsigned long long) failed.cold_policy.transcript_dropped);
                std::fprintf(stderr,
                    "phase9-cache-policy-probe: capacities hot=%u cold=%u last_ids=%zu "
                    "cpu_lanes=%llu gpu_lanes=%llu\n",
                    failed.effective_capacity, failed.cold_effective_slots, failed.last_logical_ids.size(),
                    (unsigned long long) failed.cpu_execution_lanes,
                    (unsigned long long) failed.gpu_execution_lanes);
                return 9;
            }
            llama_synchronize(context.get());
            const auto end = std::chrono::steady_clock::now();
            const float * logits = llama_get_logits_ith(context.get(), -1);
            if (!logits) {
                std::fprintf(stderr, "phase9-cache-policy-probe: missing logits at step=%d\n", step);
                return 10;
            }
            const int next = finite_argmax(logits, n_vocab);
            if (next < 0) {
                int bad_logit = -1;
                for (int index = 0; index < n_vocab; ++index) {
                    if (!std::isfinite(logits[index])) {
                        bad_logit = index;
                        break;
                    }
                }
                int bad_route = -1;
                int bad_layer = -1;
                for (size_t index = 0; index < routes.records.size() && bad_route < 0; ++index) {
                    for (float weight : routes.records[index].weights) {
                        if (!std::isfinite(weight)) {
                            bad_route = int(index);
                            bad_layer = routes.records[index].layer;
                            break;
                        }
                    }
                }
                if (provider == nullptr) {
                    std::fprintf(stderr,
                        "phase9-cache-policy-probe: provider-disabled non-finite logits step=%d first_bad=%d "
                        "first_bad_route=%d first_bad_layer=%d\n",
                        step, bad_logit, bad_route, bad_layer);
                    return 10;
                }
                const auto failed = provider->hot_cache_diagnostics();
                bool source_read = false;
                bool cold_read = false;
                bool hot_read = false;
                bool source_cold_equal = false;
                bool cold_hot_equal = false;
                uint64_t source_digest = 0;
                uint64_t cold_digest = 0;
                uint64_t hot_digest = 0;
                if (!routes.records.empty() && !routes.records.back().ids.empty()) {
                    const llm_expert_key key = {
                        routes.records.back().layer,
                        routes.records.back().ids.front(),
                    };
                    std::vector<uint8_t> cold_bytes;
                    std::vector<uint8_t> hot_bytes;
                    cold_read = provider->debug_copy_cold_bundle(key, cold_bytes).is_ready();
                    hot_read = provider->debug_copy_hot_bundle(key, hot_bytes).is_ready();
                    std::vector<uint8_t> source_bytes(cold_bytes.size());
                    auto * storage = model->expert_storage();
                    source_read = storage != nullptr && !source_bytes.empty() &&
                        storage->read_bundle(key, source_bytes.data(), source_bytes.size(), nullptr, nullptr).is_ready();
                    source_cold_equal = source_read && cold_read && source_bytes == cold_bytes;
                    cold_hot_equal = cold_read && hot_read && cold_bytes == hot_bytes;
                    if (source_read) source_digest = hash_bytes(1469598103934665603ULL,
                        source_bytes.data(), source_bytes.size());
                    if (cold_read) cold_digest = hash_bytes(1469598103934665603ULL,
                        cold_bytes.data(), cold_bytes.size());
                    if (hot_read) hot_digest = hash_bytes(1469598103934665603ULL,
                        hot_bytes.data(), hot_bytes.size());
                }
                std::fprintf(stderr,
                    "phase9-cache-policy-probe: non-finite logits step=%d first_bad=%d "
                    "first_bad_route=%d first_bad_layer=%d classes=%u hot_events=%llu cold_events=%llu "
                    "source_read=%d cold_read=%d hot_read=%d source_cold_equal=%d cold_hot_equal=%d "
                    "source_digest=%llu cold_digest=%llu hot_digest=%llu\n",
                    step, bad_logit, bad_route, bad_layer, failed.layout_class_count,
                    (unsigned long long) failed.policy.events,
                    (unsigned long long) failed.cold_policy.events,
                    source_read, cold_read, hot_read, source_cold_equal, cold_hot_equal,
                    (unsigned long long) source_digest, (unsigned long long) cold_digest,
                    (unsigned long long) hot_digest);
                return 10;
            }
            generated.push_back(next);
            generated_text += token_piece(vocab, next);
            logits_digests.push_back(hash_bytes(1469598103934665603ULL, logits, size_t(n_vocab)*sizeof(float)));
            latency_us.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());
            if (llama_vocab_is_eog(vocab, next)) break;
            batch = llama_batch_get_one(&generated.back(), 1);
        }
        // Releasing the context closes the provider request, drains or cancels
        // every background flight, and emits all reserved policy terminals.
        // Evidence must never accept a live provisional transcript prefix.
        context.reset();
        struct rusage usage {};
        getrusage(RUSAGE_SELF, &usage);
        const uint64_t cpu_user_time_us = uint64_t(usage.ru_utime.tv_sec)*1000000ULL + uint64_t(usage.ru_utime.tv_usec);
        const uint64_t cpu_system_time_us = uint64_t(usage.ru_stime.tv_sec)*1000000ULL + uint64_t(usage.ru_stime.tv_usec);
        json command = json::array();
        for (int index = 0; index < argc; ++index) command.push_back(argv[index]);
        if (provider == nullptr) {
            json output = {
                {"schema_version", "phase9-online-policy-capture-v1"}, {"status", "pass"},
                {"command", command}, {"model_path", args.model}, {"mode", args.mode},
                {"provider_enabled", false}, {"prompt", prompt_text},
                {"runtime", {
                    {"n_ctx", args.n_ctx},
                    {"n_batch", args.n_batch},
                    {"n_ubatch", args.n_ubatch},
                    {"max_generate", args.max_generate},
                }},
                {"prompt_ids", prompt}, {"generated_ids", generated}, {"generated_text", generated_text},
                {"logits_fnv64", logits_digests}, {"latency_us", latency_us},
                {"cpu_user_time_us", cpu_user_time_us}, {"cpu_system_time_us", cpu_system_time_us},
                {"peak_rss_kib", usage.ru_maxrss}, {"routes", route_json(routes)},
            };
            std::ofstream destination(args.output, std::ios::binary | std::ios::trunc);
            if (!destination) return 12;
            destination << output.dump(2) << '\n';
            destination.close();
            if (!destination) return 12;
            std::printf("PHASE9_POLICY_PROBE status=pass provider=disabled output=%s\n", args.output.c_str());
            return 0;
        }
        const auto diagnostics = provider->hot_cache_diagnostics();
        const auto storage_diagnostics = model->expert_storage()->diagnostics();
        const auto async_diagnostics = model->expert_async_diagnostics();
        const auto async_read_intervals = model->expert_async_read_intervals();
        if (diagnostics.policy.transcript_dropped != 0 || diagnostics.cold_policy.transcript_dropped != 0) {
            std::fprintf(stderr, "phase9-cache-policy-probe: policy transcript dropped events\n");
            return 11;
        }
        if (diagnostics.active_background_flights != 0) {
            std::fprintf(stderr, "phase9-cache-policy-probe: background transcript is not terminal\n");
            return 11;
        }
        std::vector<int32_t> routed_layers;
        for (const auto & record : routes.records) routed_layers.push_back(record.layer);
        if (routed_layers.empty()) {
            for (const auto & event : diagnostics.policy_events) {
                if (event.key.layer >= 0) routed_layers.push_back(event.key.layer);
            }
        }
        std::sort(routed_layers.begin(), routed_layers.end());
        routed_layers.erase(std::unique(routed_layers.begin(), routed_layers.end()), routed_layers.end());
        uint64_t hot_footprint = 0;
        for (const auto & event : diagnostics.policy_events) {
            if (event.physical_slot_footprint_bytes != 0) {
                hot_footprint = event.physical_slot_footprint_bytes;
                break;
            }
        }
        json output = {
            {"schema_version", "phase9-online-policy-capture-v1"}, {"status", "pass"},
            {"command", command}, {"model_path", args.model}, {"mode", args.mode},
            {"prompt", prompt_text},
            {"transport_requested", args.transport},
            {"config_source", args.config_source},
            {"miss_policy", args.miss_policy}, {"background", args.background},
            {"runtime", {
                {"n_ctx", args.n_ctx},
                {"n_batch", args.n_batch},
                {"n_ubatch", args.n_ubatch},
                {"max_generate", args.max_generate},
            }},
            {"prompt_ids", prompt}, {"generated_ids", generated}, {"generated_text", generated_text},
            {"logits_fnv64", logits_digests}, {"latency_us", latency_us},
            {"cpu_user_time_us", cpu_user_time_us}, {"cpu_system_time_us", cpu_system_time_us},
            {"peak_rss_kib", usage.ru_maxrss}, {"routes", route_json(routes)},
            {"topology", {
                {"routed_layers", routed_layers}, {"experts_per_layer", diagnostics.n_expert},
                {"hot_physical_slot_footprint_bytes", hot_footprint},
                {"cold_physical_slot_footprint_bytes", diagnostics.cold_slot_footprint},
                {"layout_registry", {
                    {"class_count", diagnostics.layout_class_count},
                    {"class_digests", diagnostics.layout_class_digests},
                    {"class_payload_bytes", diagnostics.layout_class_payload_bytes},
                    {"class_hot_padding_bytes", diagnostics.layout_class_hot_padding_bytes},
                    {"class_cold_padding_bytes", diagnostics.layout_class_cold_padding_bytes},
                    {"class_lane_padding_bytes", diagnostics.layout_class_lane_padding_bytes},
                    {"class_stage_bundles", diagnostics.layout_class_stage_bundles},
                    {"class_stage_bytes", diagnostics.layout_class_stage_bytes},
                    {"class_h2d_bundles", diagnostics.layout_class_h2d_bundles},
                    {"class_h2d_bytes", diagnostics.layout_class_h2d_bytes},
                    {"layer_class_ids", diagnostics.layout_layer_ids},
                    {"administration_bytes", diagnostics.layout_registry_administration_bytes},
                    {"preflight_consumer_count", diagnostics.layout_preflight_consumer_count},
                    {"preflight_passed", diagnostics.layout_preflight_passed},
                    {"hot_role_offsets", diagnostics.layout_hot_role_offsets},
                    {"hot_role_extents", diagnostics.layout_hot_role_extents},
                    {"cold_role_offsets", diagnostics.layout_cold_role_offsets},
                    {"cold_role_extents", diagnostics.layout_cold_role_extents},
                    {"lane_role_offsets", diagnostics.layout_lane_role_offsets},
                    {"lane_role_extents", diagnostics.layout_lane_role_extents},
                }},
                {"universal_hot_slot_stride", diagnostics.hot_slot_stride},
            }},
            {"capacities", {
                {"hot_requested_slots", args.hot_slots}, {"hot_effective_slots", diagnostics.effective_capacity},
                {"hot_pool_bytes", diagnostics.pool_bytes}, {"cold_requested_bytes", diagnostics.cold_requested_bytes},
                {"cold_actual_bytes", diagnostics.cold_actual_bytes}, {"cold_effective_slots", diagnostics.cold_effective_slots},
                {"cold_unused_budget_bytes", diagnostics.cold_unused_budget_bytes},
                {"cold_slot_footprint", diagnostics.cold_slot_footprint},
                {"ring_requested_bytes", diagnostics.ring_requested_bytes},
                {"ring_actual_bytes", diagnostics.ring_actual_bytes},
                {"ring_effective_lanes", diagnostics.ring_effective_lanes},
                {"ring_lane_footprint", diagnostics.ring_lane_footprint},
                {"ring_unused_budget_bytes", diagnostics.ring_requested_bytes - diagnostics.ring_actual_bytes},
                {"ring_pinned_or_registered_bytes", diagnostics.ring_pinned_or_registered_bytes},
                {"io_queue_depth_requested", args.queue_depth},
                {"io_trace_capacity_requested", args.trace_capacity},
            }},
            {"async_io", {
                {"diagnostics", async_diagnostics_json(async_diagnostics)},
                {"read_intervals", async_read_intervals_json(async_read_intervals)},
            }},
            {"cold_residency", {
                {"supported", diagnostics.cold_residency_supported},
                {"unavailable_reason", diagnostics.cold_residency_supported ? "" : diagnostics.cold_residency_unavailable_reason},
                {"ready_logical_bytes", diagnostics.cold_ready_logical_bytes},
                {"ready_page_count", diagnostics.cold_ready_page_count},
                {"resident_ready_page_count", diagnostics.cold_resident_ready_page_count},
                {"resident_ready_bytes", diagnostics.cold_resident_ready_bytes},
            }},
            {"hot", {
                {"config", config_json(diagnostics.policy)}, {"diagnostics", diagnostics_json(diagnostics.policy)},
                {"events", events_json(diagnostics.policy_events)},
            }},
            {"cold", {
                {"config", config_json(diagnostics.cold_policy)}, {"diagnostics", diagnostics_json(diagnostics.cold_policy)},
                {"events", events_json(diagnostics.cold_policy_events)},
            }},
            {"mechanism", {
                {"hot_hits", diagnostics.hits}, {"hot_misses", diagnostics.misses},
                {"hot_admissions", diagnostics.admissions}, {"hot_evictions", diagnostics.evictions},
                {"cold_hits", diagnostics.cold_hits}, {"cold_misses", diagnostics.cold_misses},
                {"cold_admissions", diagnostics.cold_admissions}, {"cold_evictions", diagnostics.cold_evictions},
                {"cold_source_copy_bundles", diagnostics.cold_source_copy_bundles},
                {"cold_source_copy_bytes", diagnostics.cold_source_copy_bytes},
                {"cold_source_copy_time_us", diagnostics.cold_source_copy_time_us},
                {"h2d_bytes", diagnostics.h2d_bytes}, {"background_useful", diagnostics.background_useful},
                {"background_wasted", diagnostics.background_wasted},
                {"active_background_flights", diagnostics.active_background_flights},
            }},
            {"transfer", {
                {"acquisition_method", diagnostics.ring_acquisition_method},
                {"pageable_fallback", diagnostics.ring_pageable_fallback},
                {"fallback_reason", diagnostics.ring_fallback_reason},
                {"lane_reservations", diagnostics.ring_lane_reservations},
                {"stage_bytes", diagnostics.ring_stage_bytes},
                {"stage_time_us", diagnostics.ring_stage_time_us},
                {"h2d_bytes", diagnostics.ring_h2d_bytes},
                {"h2d_time_us", diagnostics.ring_h2d_time_us},
                {"waves", diagnostics.ring_waves},
                {"peak_in_flight_lanes", diagnostics.ring_peak_in_flight_lanes},
                {"h2d_event_capacity", diagnostics.ring_h2d_event_capacity},
                {"compute_event_capacity", diagnostics.ring_compute_event_capacity},
                {"peak_live_h2d_events", diagnostics.ring_peak_live_h2d_events},
                {"peak_live_compute_events", diagnostics.ring_peak_live_compute_events},
                {"live_h2d_events", diagnostics.ring_live_h2d_events},
                {"live_compute_events", diagnostics.ring_live_compute_events},
                {"trace_capacity", diagnostics.ring_trace_capacity},
                {"trace_records", diagnostics.ring_trace_records},
                {"trace_records_dropped", diagnostics.ring_trace_records_dropped},
                {"failed_cleanup", diagnostics.ring_failed_cleanup},
            }},
            {"storage", {
                {"source_file_count", storage_diagnostics.source_file_count},
                {"directory_entry_count", storage_diagnostics.directory_entry_count},
                {"span_count", storage_diagnostics.span_count},
                {"administration_bytes", storage_diagnostics.administration_bytes},
                {"read_requests", storage_diagnostics.read_requests},
                {"read_operations", storage_diagnostics.read_chunks},
                {"read_bytes", storage_diagnostics.read_bytes},
                {"cancelled_reads", storage_diagnostics.cancelled_reads},
                {"short_reads", storage_diagnostics.short_reads},
                {"io_errors", storage_diagnostics.io_errors},
                {"integrity_checks", storage_diagnostics.integrity_checks},
                {"integrity_mismatches", storage_diagnostics.integrity_mismatches},
                {"integrity_not_checked", storage_diagnostics.integrity_not_checked},
                {"integrity_digest_bytes", storage_diagnostics.integrity_digest_bytes},
                {"integrity_mode", integrity_mode_name(storage_diagnostics.integrity_mode)},
                {"integrity_status", integrity_status_name(storage_diagnostics.integrity_status)},
                {"direct_source_count", storage_diagnostics.direct_source_count},
                {"direct_unsupported_source_count", storage_diagnostics.direct_unsupported_source_count},
                {"maximum_direct_alignment", storage_diagnostics.maximum_direct_alignment},
                {"maximum_bundle_bytes", storage_diagnostics.maximum_bundle_bytes},
                {"first_direct_error", storage_diagnostics.first_direct_error},
                {"first_native_error", storage_diagnostics.first_native_error},
                {"sealed", storage_diagnostics.sealed},
                {"poisoned", storage_diagnostics.poisoned},
            }},
            {"high_water", {
                {"hot_pins", diagnostics.peak_pins},
                {"cold_hot_refs", diagnostics.cold_peak_hot_refs},
                {"cold_transfer_refs", diagnostics.cold_peak_transfer_refs},
                {"cold_request_refs", diagnostics.cold_peak_request_refs},
                {"cold_cpu_execution_refs", diagnostics.cold_peak_cpu_execution_refs},
                {"background_flights", diagnostics.peak_background_flights},
                {"ring_in_flight_lanes", diagnostics.ring_peak_in_flight_lanes},
                {"ring_h2d_events", diagnostics.ring_peak_live_h2d_events},
                {"ring_compute_events", diagnostics.ring_peak_live_compute_events},
            }},
            {"context_validation", {
                {"last_n_ctx", diagnostics.last_context_n_ctx},
                {"last_n_ubatch", diagnostics.last_context_n_ubatch},
                {"last_extent", diagnostics.last_context_extent},
                {"conservative_required_capacity", diagnostics.conservative_required_capacity},
                {"validations", diagnostics.context_validations},
                {"rejections", diagnostics.context_rejections},
            }},
            {"lifecycle", {
                {"current_hot_pins", diagnostics.current_pins},
                {"cold_current_hot_refs", diagnostics.cold_current_hot_refs},
                {"cold_current_transfer_refs", diagnostics.cold_current_transfer_refs},
                {"cold_current_request_refs", diagnostics.cold_current_request_refs},
                {"cold_current_cpu_execution_refs", diagnostics.cold_current_cpu_execution_refs},
                {"active_background_flights", diagnostics.active_background_flights},
                {"hot_failed_cleanups", diagnostics.failed_cleanups},
                {"cold_failed_cleanups", diagnostics.cold_failed_cleanups},
                {"hot_transcript_dropped", diagnostics.policy.transcript_dropped},
                {"cold_transcript_dropped", diagnostics.cold_policy.transcript_dropped},
            }},
        };
        std::ofstream destination(args.output, std::ios::binary | std::ios::trunc);
        if (!destination) return 12;
        destination << output.dump(2) << '\n';
        destination.close();
        if (!destination) return 12;
        std::printf("PHASE9_POLICY_PROBE status=pass events=%zu cold_events=%zu output=%s\n",
            diagnostics.policy_events.size(), diagnostics.cold_policy_events.size(), args.output.c_str());
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "phase9-cache-policy-probe: %s\n", error.what());
        return 1;
    }
}
