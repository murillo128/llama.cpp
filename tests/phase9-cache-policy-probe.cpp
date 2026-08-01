#include "llama.h"
#include "llama-context.h"
#include "llama-expert-weight-provider.h"
#include "llama-model.h"
#include "llama-cpp.h"

#include "ggml-backend.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <sys/resource.h>
#include <vector>

using json = nlohmann::json;

namespace {

struct arguments {
    std::string model;
    std::string output;
    std::string mode = "cold";
    std::string hot_policy = "LRU";
    std::string cold_policy = "LRU";
    std::string scope = "GLOBAL";
    std::string admission = "ALWAYS";
    std::string miss_policy = "CPU_FALLBACK";
    uint32_t hot_slots = 16;
    uint64_t cold_bytes = 64U*1024U*1024U;
    uint64_t ring_bytes = 16U*1024U*1024U;
    uint32_t ratio = 7500;
    uint32_t window = 1024;
    uint32_t aging = 1024;
    uint32_t n_ubatch = 1;
    int max_generate = 2;
    bool background = false;
    bool observe_routes = true;
    std::string transport = "BUFFERED";
    std::string config_source = "EXPLICIT";
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

bool parse_arguments(int argc, char ** argv, arguments & result) {
    for (int index = 1; index < argc; ++index) {
        if (index + 1 >= argc) return false;
        const std::string option = argv[index];
        const char * value = argv[++index];
        if (option == "--model") result.model = value;
        else if (option == "--output") result.output = value;
        else if (option == "--mode") result.mode = value;
        else if (option == "--hot-policy") result.hot_policy = value;
        else if (option == "--cold-policy") result.cold_policy = value;
        else if (option == "--scope") result.scope = value;
        else if (option == "--admission") result.admission = value;
        else if (option == "--miss-policy") result.miss_policy = value;
        else if (option == "--hot-slots") { if (!parse_u32(value, result.hot_slots)) return false; }
        else if (option == "--cold-bytes") { if (!parse_u64(value, result.cold_bytes)) return false; }
        else if (option == "--ring-bytes") { if (!parse_u64(value, result.ring_bytes)) return false; }
        else if (option == "--ratio") { if (!parse_u32(value, result.ratio)) return false; }
        else if (option == "--window") { if (!parse_u32(value, result.window)) return false; }
        else if (option == "--aging") { if (!parse_u32(value, result.aging)) return false; }
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
            if (result.transport != "BUFFERED" && result.transport != "DIRECT_IO") return false;
        } else if (option == "--config-source") {
            result.config_source = value;
            if (result.config_source != "EXPLICIT" && result.config_source != "NULL") return false;
        } else return false;
    }
    return !result.model.empty() && !result.output.empty() && result.hot_slots > 0 && result.n_ubatch > 0 &&
        (result.mode == "hot" || result.mode == "cold");
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

} // namespace

int main(int argc, char ** argv) {
    try {
        arguments args;
        if (!parse_arguments(argc, argv, args)) {
            std::fprintf(stderr, "usage: %s --model GGUF --output JSON [--mode hot|cold] [policy/capacity options]\n", argv[0]);
            return 2;
        }
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
        model_params.n_gpu_layers = -1;
        model_params.tensor_buft_overrides = overrides;
        model_params.expert_weights_mode = args.mode == "cold" ?
            LLAMA_EXPERT_WEIGHTS_MODE_COLD_CACHE : LLAMA_EXPERT_WEIGHTS_MODE_HOT_CACHE;
        model_params.expert_hot_cache_capacity = args.hot_slots;
        model_params.expert_hot_cache_policy = args.config_source == "NULL" ? nullptr : &hot_config;
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
        if (!provider) return 4;
        const auto * vocab = llama_model_get_vocab(model.get());
        const std::string prompt_text = "According to all known laws";
        const int prompt_count = -llama_tokenize(vocab, prompt_text.data(), prompt_text.size(), nullptr, 0, true, true);
        if (prompt_count <= 0) return 5;
        std::vector<llama_token> prompt(prompt_count);
        if (llama_tokenize(vocab, prompt_text.data(), prompt_text.size(), prompt.data(), prompt.size(), true, true) != prompt_count) return 5;
        auto context_params = llama_context_default_params();
        context_params.n_ctx = 64;
        context_params.n_batch = 64;
        context_params.n_ubatch = args.n_ubatch;
        context_params.no_perf = false;
        llama_context_ptr context(llama_init_from_model(model.get(), context_params));
        if (!context) return 6;
        route_capture routes;
        if (args.observe_routes &&
            llama_set_route_observer(context.get(), capture_route, &routes) != LLAMA_ROUTE_OBSERVER_STATUS_OK) return 7;
        std::vector<llama_token> generated;
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
            if (!logits) return 10;
            const int next = finite_argmax(logits, n_vocab);
            if (next < 0) return 10;
            generated.push_back(next);
            logits_digests.push_back(hash_bytes(1469598103934665603ULL, logits, size_t(n_vocab)*sizeof(float)));
            latency_us.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());
            if (llama_vocab_is_eog(vocab, next)) break;
            batch = llama_batch_get_one(&generated.back(), 1);
        }
        // Releasing the context closes the provider request, drains or cancels
        // every background flight, and emits all reserved policy terminals.
        // Evidence must never accept a live provisional transcript prefix.
        context.reset();
        const auto diagnostics = provider->hot_cache_diagnostics();
        if (diagnostics.policy.transcript_dropped != 0 || diagnostics.cold_policy.transcript_dropped != 0) {
            std::fprintf(stderr, "phase9-cache-policy-probe: policy transcript dropped events\n");
            return 11;
        }
        if (diagnostics.active_background_flights != 0) {
            std::fprintf(stderr, "phase9-cache-policy-probe: background transcript is not terminal\n");
            return 11;
        }
        struct rusage usage {};
        getrusage(RUSAGE_SELF, &usage);
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
        json command = json::array();
        for (int index = 0; index < argc; ++index) command.push_back(argv[index]);
        json output = {
            {"schema_version", "phase9-online-policy-capture-v1"}, {"status", "pass"},
            {"command", command}, {"model_path", args.model}, {"mode", args.mode},
            {"transport_requested", args.transport},
            {"config_source", args.config_source},
            {"miss_policy", args.miss_policy}, {"background", args.background},
            {"prompt_ids", prompt}, {"generated_ids", generated}, {"logits_fnv64", logits_digests},
            {"latency_us", latency_us}, {"peak_rss_kib", usage.ru_maxrss}, {"routes", route_json(routes)},
            {"topology", {
                {"routed_layers", routed_layers}, {"experts_per_layer", diagnostics.n_expert},
                {"hot_physical_slot_footprint_bytes", hot_footprint},
                {"cold_physical_slot_footprint_bytes", diagnostics.cold_slot_footprint},
            }},
            {"capacities", {
                {"hot_requested_slots", args.hot_slots}, {"hot_effective_slots", diagnostics.effective_capacity},
                {"hot_pool_bytes", diagnostics.pool_bytes}, {"cold_requested_bytes", diagnostics.cold_requested_bytes},
                {"cold_actual_bytes", diagnostics.cold_actual_bytes}, {"cold_effective_slots", diagnostics.cold_effective_slots},
                {"cold_slot_footprint", diagnostics.cold_slot_footprint},
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
                {"h2d_bytes", diagnostics.h2d_bytes}, {"background_useful", diagnostics.background_useful},
                {"background_wasted", diagnostics.background_wasted},
                {"active_background_flights", diagnostics.active_background_flights},
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
