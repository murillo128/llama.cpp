#include "llama.h"
#include "llama-context.h"
#include "llama-expert-weight-provider.h"
#include "llama-model.h"

#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <sys/resource.h>
#include <vector>

namespace {

using clock_type = std::chrono::steady_clock;

struct arguments {
    std::string model;
    std::string mode;
    std::string routes;
    std::string logits;
    uint32_t capacity = 0;
    uint32_t n_ubatch = 0;
    int max_generate = 8;
};

struct route_record {
    uint64_t request = 0;
    uint64_t ubatch = 0;
    int32_t phase = 0;
    int32_t layer = 0;
    uint32_t n_tokens = 0;
    uint32_t n_expert_used = 0;
    std::vector<int32_t> ids;
    std::vector<float> weights;
};

struct route_capture {
    std::vector<route_record> records;
};

bool capture_route(const llama_route_observation * observation, void * user_data) {
    auto * capture = static_cast<route_capture *>(user_data);
    const size_t count = size_t(observation->n_tokens)*observation->n_expert_used;
    route_record record;
    record.request = observation->request_ordinal;
    record.ubatch = observation->ubatch_ordinal;
    record.phase = observation->phase;
    record.layer = observation->layer;
    record.n_tokens = observation->n_tokens;
    record.n_expert_used = observation->n_expert_used;
    record.ids.assign(observation->selected_experts, observation->selected_experts + count);
    record.weights.assign(observation->weights, observation->weights + count);
    capture->records.push_back(std::move(record));
    return true;
}

bool parse_u32(const char * text, uint32_t & value) {
    char * end = nullptr;
    const unsigned long parsed = std::strtoul(text, &end, 10);
    if (end == text || *end != '\0' || parsed > UINT32_MAX) {
        return false;
    }
    value = uint32_t(parsed);
    return true;
}

bool parse_int(const char * text, int & value) {
    char * end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || parsed < 1 || parsed > 128) {
        return false;
    }
    value = int(parsed);
    return true;
}

bool parse_arguments(int argc, char ** argv, arguments & result) {
    for (int index = 1; index < argc; ++index) {
        const std::string option = argv[index];
        if (index + 1 >= argc) {
            return false;
        }
        const char * value = argv[++index];
        if (option == "--model") {
            result.model = value;
        } else if (option == "--mode") {
            result.mode = value;
        } else if (option == "--capacity") {
            if (!parse_u32(value, result.capacity)) return false;
        } else if (option == "--n-ubatch") {
            if (!parse_u32(value, result.n_ubatch)) return false;
        } else if (option == "--max-generate") {
            if (!parse_int(value, result.max_generate)) return false;
        } else if (option == "--routes") {
            result.routes = value;
        } else if (option == "--logits") {
            result.logits = value;
        } else {
            return false;
        }
    }
    return !result.model.empty() && (result.mode == "disabled" || result.mode == "hot") &&
        result.n_ubatch > 0 && !result.routes.empty() && !result.logits.empty() &&
        (result.mode == "disabled" || result.capacity > 0);
}

double seconds(clock_type::time_point begin, clock_type::time_point end) {
    return std::chrono::duration<double>(end - begin).count();
}

double percentile(std::vector<double> values, double quantile) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const size_t index = size_t(std::ceil(quantile*values.size())) - 1;
    return values[std::min(index, values.size() - 1)];
}

int finite_argmax(const float * logits, int count) {
    int result = -1;
    for (int index = 0; index < count; ++index) {
        if (!std::isfinite(logits[index])) return -1;
        if (result < 0 || logits[index] > logits[result]) result = index;
    }
    return result;
}

uint64_t address_hash(const std::vector<uintptr_t> & addresses) {
    uint64_t hash = 1469598103934665603ULL;
    for (uintptr_t address : addresses) {
        for (size_t byte = 0; byte < sizeof(address); ++byte) {
            hash ^= uint8_t(address >> (byte*8));
            hash *= 1099511628211ULL;
        }
    }
    return hash;
}

template<typename T>
void print_csv(const std::vector<T> & values) {
    for (size_t index = 0; index < values.size(); ++index) {
        if (index) std::cout << ',';
        std::cout << values[index];
    }
}

bool write_routes(const std::string & path, const route_capture & capture) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) return false;
    for (const auto & record : capture.records) {
        output << record.request << '\t' << record.ubatch << '\t' << record.phase << '\t'
               << record.layer << '\t' << record.n_tokens << '\t' << record.n_expert_used;
        for (size_t index = 0; index < record.ids.size(); ++index) {
            uint32_t bits = 0;
            static_assert(sizeof(bits) == sizeof(record.weights[index]), "float width");
            std::memcpy(&bits, &record.weights[index], sizeof(bits));
            output << '\t' << record.ids[index] << ':' << std::hex << bits << std::dec;
        }
        output << '\n';
    }
    return bool(output);
}

void print_hot(const char * prefix, const llm_hot_cache_diagnostics & value) {
    std::cout << prefix
              << "\trequested_capacity=" << value.requested_capacity
              << "\teffective_capacity=" << value.effective_capacity
              << "\tpool_bytes=" << value.pool_bytes
              << "\taddress_hash=" << address_hash(value.slot_tensor_addresses)
              << "\taddress_count=" << value.slot_tensor_addresses.size()
              << "\tgraph_epoch=" << value.graph_epoch
              << "\tgeneration=" << value.generation
              << "\tn_expert=" << value.n_expert
              << "\tn_expert_used=" << value.n_expert_used
              << "\tn_ctx=" << value.last_context_n_ctx
              << "\tn_ubatch=" << value.last_context_n_ubatch
              << "\textent=" << value.last_context_extent
              << "\trequired_capacity=" << value.conservative_required_capacity
              << "\trequests=" << value.requests
              << "\tbusy_failures=" << value.exclusive_busy_failures
              << "\tremap_checkpoints=" << value.remap_checkpoints
              << "\tlogical_ids=" << value.logical_ids
              << "\tunique_ids=" << value.unique_ids
              << "\thits=" << value.hits
              << "\tmisses=" << value.misses
              << "\tadmissions=" << value.admissions
              << "\tevictions=" << value.evictions
              << "\tgeneration_changes=" << value.generation_changes
              << "\tstale_failures=" << value.stale_generation_failures
              << "\tcopy_failures=" << value.copy_failures
              << "\tpin_acquires=" << value.pin_acquires
              << "\tpin_releases=" << value.pin_releases
              << "\tcurrent_pins=" << value.current_pins
              << "\tpeak_pins=" << value.peak_pins
              << "\th2d_bytes=" << value.h2d_bytes
              << "\th2d_time_us=" << value.h2d_time_us
              << "\tid_read_bytes=" << value.execution_id_read_bytes
              << "\tid_write_bytes=" << value.execution_id_write_bytes
              << "\tsync_checkpoints=" << value.synchronization_checkpoints
              << "\tremap_dynamic_allocations=" << value.remap_dynamic_allocations
              << "\tsource_buffer=" << (value.source_buffer_type ? ggml_backend_buft_name(value.source_buffer_type) : "none")
              << "\ttarget_buffer=" << (value.target_buffer_type ? ggml_backend_buft_name(value.target_buffer_type) : "none")
              << '\n';
}

} // namespace

int main(int argc, char ** argv) {
    arguments args;
    if (!parse_arguments(argc, argv, args)) {
        std::cerr << "usage: phase4-hot-cache-probe --model PATH --mode disabled|hot --capacity N --n-ubatch N --max-generate N --routes PATH --logits PATH\n";
        return 2;
    }

    ggml_backend_load_all();
    const llama_model_tensor_buft_override overrides[] = {
        { "ffn_(gate|up|down)_exps\\.weight", ggml_backend_cpu_buffer_type() },
        { nullptr, nullptr },
    };
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = -1;
    if (args.mode == "hot") {
        model_params.tensor_buft_overrides = overrides;
        model_params.expert_weights_mode = LLAMA_EXPERT_WEIGHTS_MODE_HOT_CACHE;
        model_params.expert_hot_cache_capacity = args.capacity;
    }
    llama_model * model = llama_model_load_from_file(args.model.c_str(), model_params);
    if (!model) return 3;

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const std::string prompt_text = "According to all known laws";
    const int prompt_count = -llama_tokenize(vocab, prompt_text.data(), prompt_text.size(), nullptr, 0, true, true);
    std::vector<llama_token> prompt(prompt_count);
    if (prompt_count <= 0 || llama_tokenize(vocab, prompt_text.data(), prompt_text.size(),
            prompt.data(), prompt.size(), true, true) != prompt_count) return 4;

    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = 64;
    context_params.n_batch = 64;
    context_params.n_ubatch = args.n_ubatch;
    context_params.no_perf = false;
    llama_context * context = llama_init_from_model(model, context_params);
    if (!context) return 5;

    route_capture routes;
    if (llama_set_route_observer(context, capture_route, &routes) != LLAMA_ROUTE_OBSERVER_STATUS_OK) return 6;
    std::ofstream logits(args.logits, std::ios::binary | std::ios::trunc);
    if (!logits) return 7;

    std::vector<llama_token> generated;
    std::vector<double> latencies;
    llm_hot_cache_diagnostics first_hot;
    llama_batch batch = llama_batch_get_one(prompt.data(), prompt.size());
    const int vocab_count = llama_vocab_n_tokens(vocab);
    for (int step = 0; step < args.max_generate; ++step) {
        const llama_route_phase phase = step == 0 ? LLAMA_ROUTE_PHASE_PREFILL : LLAMA_ROUTE_PHASE_DECODE;
        if (llama_route_observer_begin(context, uint64_t(step), phase) != LLAMA_ROUTE_OBSERVER_STATUS_OK) return 8;
        const auto begin = clock_type::now();
        const int status = llama_decode(context, batch);
        const float * current_logits = status == 0 ? llama_get_logits_ith(context, -1) : nullptr;
        const auto end = clock_type::now();
        if (status != 0 || !current_logits) return 9;
        latencies.push_back(seconds(begin, end));
        logits.write(reinterpret_cast<const char *>(current_logits), size_t(vocab_count)*sizeof(float));
        const int next = finite_argmax(current_logits, vocab_count);
        if (!logits || next < 0) return 10;
        generated.push_back(next);
        if (step == 0 && args.mode == "hot") first_hot = model->expert_weight_provider()->hot_cache_diagnostics();
        if (llama_vocab_is_eog(vocab, next)) break;
        batch = llama_batch_get_one(&generated.back(), 1);
    }
    llama_synchronize(context);
    if (!write_routes(args.routes, routes)) return 11;

    const double ttft = latencies.front();
    const double decode_seconds = std::accumulate(latencies.begin() + 1, latencies.end(), 0.0);
    const auto graph = context->expert_graph_diagnostics();
    const auto provider = model->expert_weight_provider_stats();
    struct rusage usage {};
    getrusage(RUSAGE_SELF, &usage);
    std::cout << std::setprecision(17)
              << "PHASE4_RUN\tmode=" << args.mode
              << "\tcapacity=" << args.capacity
              << "\tn_ctx=64\tn_ubatch=" << args.n_ubatch
              << "\textent=" << std::min<uint32_t>(64, args.n_ubatch)
              << "\tprompt_tokens=" << prompt.size()
              << "\tgenerated_tokens=" << generated.size()
              << "\tttft_seconds=" << ttft
              << "\tprompt_tokens_per_second=" << prompt.size()/ttft
              << "\tdecode_tokens_per_second=" << (latencies.size() > 1 ? (latencies.size() - 1)/decode_seconds : 0.0)
              << "\ttoken_latency_p50_seconds=" << percentile(latencies, 0.50)
              << "\ttoken_latency_p95_seconds=" << percentile(latencies, 0.95)
              << "\ttoken_latency_p99_seconds=" << percentile(latencies, 0.99)
              << "\tpeak_rss_kib=" << usage.ru_maxrss
              << "\troute_records=" << routes.records.size()
              << "\tgraph_nodes=" << graph.node_count
              << "\tgraph_hash=" << graph.operation_hash
              << "\tgraph_bindings=" << graph.binding_count
              << "\tgraphs_reused=" << graph.graphs_reused
              << "\tprovider_objects=" << provider.objects_created
              << "\tprovider_callbacks=" << provider.callbacks
              << "\tprovider_copies=" << provider.tensor_copies
              << "\tprovider_syncs=" << provider.synchronizations
              << "\tprovider_failures=" << provider.failures
              << '\n';
    std::cout << "PHASE4_PROMPT_IDS\tvalues="; print_csv(prompt);
    std::cout << "\nPHASE4_GENERATED_IDS\tvalues="; print_csv(generated); std::cout << '\n';
    if (args.mode == "hot") {
        print_hot("PHASE4_FIRST_HOT", first_hot);
        const auto final_hot = model->expert_weight_provider()->hot_cache_diagnostics();
        print_hot("PHASE4_FINAL_HOT", final_hot);
        std::cout << "PHASE4_LAST_IDS\tlogical="; print_csv(final_hot.last_logical_ids);
        std::cout << "\texecution="; print_csv(final_hot.last_execution_ids); std::cout << '\n';
    }
    std::cout << "PHASE4_RESULT\texit=0\n";

    llama_free(context);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
