#include "llama-expert-prefetch.h"
#include "llama-expert-weight-provider.h"
#include "llama-model.h"
#include "llama.h"
#include "llama-cpp.h"

#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

using json = nlohmann::json;

namespace {

struct probe_error : std::runtime_error {
    using std::runtime_error::runtime_error;
};

int finite_argmax(const float * logits, int count) {
    int result = -1;
    float best = 0.0f;
    for (int index = 0; index < count; ++index) {
        if (!std::isfinite(logits[index]) || (result >= 0 && logits[index] <= best)) continue;
        result = index;
        best = logits[index];
    }
    return result;
}

uint64_t percentile(std::vector<uint64_t> values, uint32_t numerator, uint32_t denominator) {
    if (values.empty()) throw probe_error("empty measurement");
    std::sort(values.begin(), values.end());
    const size_t index = std::min(values.size() - 1, (values.size()*numerator + denominator - 1)/denominator - 1);
    return values[index];
}

uint64_t elapsed_ns(std::chrono::steady_clock::time_point begin) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - begin).count();
}

std::vector<uint64_t> benchmark_pread(const std::string & path, uint64_t bytes, bool direct, bool & supported) {
    supported = false;
    const size_t alignment = 4096;
    const size_t extent = direct ? size_t((bytes + alignment - 1)/alignment*alignment) : size_t(bytes);
    int flags = O_RDONLY;
#ifdef O_DIRECT
    if (direct) flags |= O_DIRECT;
#else
    if (direct) return {};
#endif
    const int descriptor = open(path.c_str(), flags);
    if (descriptor < 0) return {};
    void * buffer = nullptr;
    if (posix_memalign(&buffer, alignment, extent) != 0) {
        close(descriptor);
        return {};
    }
    std::vector<uint64_t> measurements;
    for (uint32_t repetition = 0; repetition < 21; ++repetition) {
        const auto begin = std::chrono::steady_clock::now();
        const ssize_t result = pread(descriptor, buffer, extent, 0);
        const uint64_t elapsed = elapsed_ns(begin);
        if (result != ssize_t(extent)) {
            std::free(buffer);
            close(descriptor);
            return {};
        }
        if (repetition != 0) measurements.push_back(elapsed);
    }
    std::free(buffer);
    close(descriptor);
    supported = true;
    return measurements;
}

std::vector<uint64_t> benchmark_h2d(uint64_t bytes, bool & supported) {
    supported = false;
    ggml_backend_dev_t device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (device == nullptr || bytes > SIZE_MAX) return {};
    ggml_backend_ptr backend(ggml_backend_dev_init(device, nullptr));
    if (!backend) return {};
    ggml_init_params params = { ggml_tensor_overhead(), nullptr, true };
    ggml_context_ptr context(ggml_init(params));
    if (!context) return {};
    ggml_tensor * tensor = ggml_new_tensor_1d(context.get(), GGML_TYPE_I8, int64_t(bytes));
    ggml_backend_buffer_ptr destination(
        ggml_backend_alloc_ctx_tensors_from_buft(context.get(), ggml_backend_dev_buffer_type(device)));
    ggml_backend_buffer_ptr source(ggml_backend_buft_alloc_buffer(ggml_backend_dev_host_buffer_type(device), size_t(bytes)));
    if (!destination || !source || !ggml_backend_buffer_is_host(source.get())) return {};
    ggml_backend_buffer_clear(source.get(), 0);
    std::vector<uint64_t> measurements;
    for (uint32_t repetition = 0; repetition < 21; ++repetition) {
        const auto begin = std::chrono::steady_clock::now();
        ggml_backend_tensor_set_async(backend.get(), tensor, ggml_backend_buffer_get_base(source.get()), 0, size_t(bytes));
        ggml_backend_synchronize(backend.get());
        const uint64_t elapsed = elapsed_ns(begin);
        if (repetition != 0) measurements.push_back(elapsed);
    }
    supported = true;
    return measurements;
}

struct predictor_benchmark {
    uint64_t upper_bound_p95_ns = 0;
    json measurements = json::array();
};

predictor_benchmark benchmark_predictor(
        const llm_expert_prefetch_profile & profile,
        const std::string & profile_path) {
    std::vector<std::vector<int32_t>> token(profile.target.routed_layers.size());
    for (size_t layer = 0; layer < token.size(); ++layer) {
        for (uint32_t rank = 0; rank < profile.target.experts_per_token; ++rank) {
            token[layer].push_back(int32_t((layer + rank)%profile.target.experts_per_layer));
        }
    }

    predictor_benchmark result;
    const auto measure = [&](llama_expert_prefetch_policy policy, uint32_t window, const char * name, bool cross) {
        llama_expert_prefetch_config_v1 value = {
            LLAMA_EXPERT_PREFETCH_VERSION_1, sizeof(llama_expert_prefetch_config_v1), policy,
            LLAMA_EXPERT_PREFETCH_READINESS_DEVICE_READY, LLAMA_EXPERT_PREFETCH_SEED_MODE_OFF, window,
            profile.target.experts_per_layer, 64U*1024U*1024U, 64, UINT64_C(1) << 30,
            UINT64_C(1) << 30, UINT64_C(1) << 30, UINT64_C(1) << 30, 64, 64, 64, 32, {},
        };
        llm_expert_prefetch_config_internal config;
        if (!llm_expert_prefetch_copy_config(&value, profile_path.c_str(), LLAMA_EXPERT_WEIGHTS_MODE_COLD_CACHE,
            profile.target.experts_per_layer, 64, UINT64_C(1) << 30, 64, UINT64_C(1) << 30,
            UINT64_C(1) << 30, config).is_ready()) throw probe_error("predictor configuration failed");
        llm_expert_prefetch_predictor predictor;
        if (!predictor.initialize(profile, config).is_ready() || !predictor.request_begin(1).is_ready()) {
            throw probe_error("predictor initialization failed");
        }
        for (uint64_t ordinal = 0; ordinal < 64; ++ordinal) {
            if (!predictor.commit_token(ordinal, token).is_ready()) throw probe_error("predictor warmup failed");
        }
        std::vector<uint64_t> measurements;
        for (uint32_t repetition = 0; repetition < 1000; ++repetition) {
            std::vector<llm_expert_prefetch_candidate> candidates;
            const auto begin = std::chrono::steady_clock::now();
            if (cross) {
                for (size_t layer = 0; layer + 1 < profile.target.routed_layers.size(); ++layer) {
                    if (!predictor.predict_cross_layer(64, profile.target.routed_layers[layer], token[layer].data(),
                        token[layer].size(), profile.target.routed_layers[layer + 1], candidates).is_ready()) {
                        throw probe_error("cross-layer predictor benchmark failed");
                    }
                }
            } else {
                for (const int32_t layer : profile.target.routed_layers) {
                    if (!predictor.predict_token_end(63, layer, candidates).is_ready()) {
                        throw probe_error("token-end predictor benchmark failed");
                    }
                }
            }
            measurements.push_back(elapsed_ns(begin));
        }
        const uint64_t p50 = percentile(measurements, 50, 100);
        const uint64_t p95 = percentile(measurements, 95, 100);
        const uint64_t p99 = percentile(measurements, 99, 100);
        result.upper_bound_p95_ns = std::max(result.upper_bound_p95_ns, p95);
        result.measurements.push_back({{"policy", name}, {"temporal_window_tokens", window},
            {"candidates_per_target", profile.target.experts_per_layer}, {"target_layers_per_call", cross ?
                profile.target.routed_layers.size() - 1 : profile.target.routed_layers.size()},
            {"repetitions", measurements.size()}, {"p50_ns", p50}, {"p95_ns", p95}, {"p99_ns", p99}});
    };
    measure(LLAMA_EXPERT_PREFETCH_POLICY_STATIC_LAYER, 0, "STATIC_LAYER", false);
    measure(LLAMA_EXPERT_PREFETCH_POLICY_PREVIOUS_TOKEN, 0, "PREVIOUS_TOKEN", false);
    measure(LLAMA_EXPERT_PREFETCH_POLICY_TEMPORAL_FREQUENCY, 64, "TEMPORAL_FREQUENCY", false);
    measure(LLAMA_EXPERT_PREFETCH_POLICY_CROSS_LAYER_TRANSITION, 0, "CROSS_LAYER_TRANSITION", true);
    measure(LLAMA_EXPERT_PREFETCH_POLICY_RANDOM_BASELINE, 0, "RANDOM_BASELINE", false);
    return result;
}

struct model_profile_validation {
    uint64_t load_ns = 0;
    json lead_measurements;
};

model_profile_validation validate_model_profile(
        const std::string & model_path,
        const std::string & profile_path,
        const llm_expert_prefetch_profile & profile,
        bool measure_lead) {
    const auto selected_cost = std::find_if(profile.costs.begin(), profile.costs.end(), [&](const auto & cost) {
        return cost.transport == profile.selected_transport && cost.readiness == profile.selected_readiness;
    });
    if (selected_cost == profile.costs.end()) throw probe_error("selected cost envelope is unavailable");
    static const llama_model_tensor_buft_override overrides[] = {
        { "ffn_(gate|up|down)_exps\\.weight", ggml_backend_cpu_buffer_type() },
        { nullptr, nullptr },
    };
    const auto selected_policy = [&]() {
        if (profile.selected_policy == "STATIC_LAYER") return LLAMA_EXPERT_PREFETCH_POLICY_STATIC_LAYER;
        if (profile.selected_policy == "PREVIOUS_TOKEN") return LLAMA_EXPERT_PREFETCH_POLICY_PREVIOUS_TOKEN;
        if (profile.selected_policy == "TEMPORAL_FREQUENCY") return LLAMA_EXPERT_PREFETCH_POLICY_TEMPORAL_FREQUENCY;
        if (profile.selected_policy == "CROSS_LAYER_TRANSITION") return LLAMA_EXPERT_PREFETCH_POLICY_CROSS_LAYER_TRANSITION;
        if (profile.selected_policy == "RANDOM_BASELINE") return LLAMA_EXPERT_PREFETCH_POLICY_RANDOM_BASELINE;
        if (profile.selected_policy == "BLOCKING_HOT") return LLAMA_EXPERT_PREFETCH_POLICY_OFF;
        throw probe_error("unsupported selected profile policy");
    }();
    const bool seed = profile.selected_policy == "BLOCKING_HOT";
    llama_expert_prefetch_config_v1 prefetch = {
        LLAMA_EXPERT_PREFETCH_VERSION_1, sizeof(llama_expert_prefetch_config_v1),
        selected_policy, profile.selected_readiness,
        seed ? LLAMA_EXPERT_PREFETCH_SEED_MODE_BLOCKING_HOT : LLAMA_EXPERT_PREFETCH_SEED_MODE_OFF,
        profile.selected_temporal_window, seed ? 0U : profile.selected_candidates,
        64U*1024U*1024U, seed ? 0U : 16U, seed ? 0U : 16U*1024U*1024U,
        seed ? 0U : 16U*1024U*1024U, seed ? 0U : 16U*1024U*1024U,
        seed ? 0U : 16U*1024U*1024U, seed ? 0U : 14U, seed ? 0U : 14U,
        seed ? 0U : selected_cost->utility_window_predictions,
        seed ? 0U : selected_cost->utility_min_observations, {},
    };
    auto params = llama_model_default_params();
    params.n_gpu_layers = -1;
    params.tensor_buft_overrides = overrides;
    params.expert_weights_mode = LLAMA_EXPERT_WEIGHTS_MODE_COLD_CACHE;
    params.expert_hot_cache_capacity = 16;
    params.expert_cold_cache_bytes = 64U*1024U*1024U;
    params.expert_transfer_ring_bytes = 16U*1024U*1024U;
    params.expert_prefetch_config = &prefetch;
    params.expert_prefetch_profile_path = profile_path.c_str();
    const auto begin = std::chrono::steady_clock::now();
    llama_model_ptr model(llama_model_load_from_file(model_path.c_str(), params));
    const uint64_t duration = elapsed_ns(begin);
    if (!model) throw probe_error("active profile model load failed");
    model_profile_validation result;
    result.load_ns = duration;
    if (!measure_lead) return result;

    const auto * vocab = llama_model_get_vocab(model.get());
    const std::string prompt_text = "According to all known laws";
    const int prompt_count = -llama_tokenize(vocab, prompt_text.data(), prompt_text.size(), nullptr, 0, true, true);
    if (prompt_count <= 0) throw probe_error("lead probe prompt tokenization failed");
    std::vector<llama_token> prompt(prompt_count);
    if (llama_tokenize(vocab, prompt_text.data(), prompt_text.size(), prompt.data(), prompt.size(), true, true) != prompt_count) {
        throw probe_error("lead probe prompt tokenization changed");
    }
    auto context_params = llama_context_default_params();
    context_params.n_ctx = 64;
    context_params.n_batch = 64;
    context_params.n_ubatch = 64;
    context_params.no_perf = false;
    llama_context_ptr context(llama_init_from_model(model.get(), context_params));
    if (!context) throw probe_error("lead probe context creation failed");
    auto * provider = model->expert_weight_provider();
    if (provider == nullptr) throw probe_error("lead probe expert provider is unavailable");
    llama_batch batch = llama_batch_get_one(prompt.data(), prompt.size());
    llama_token generated = 0;
    const int vocabulary = llama_vocab_n_tokens(vocab);
    std::vector<uint64_t> decode_begins;
    for (int step = 0; step < 6; ++step) {
        if (step != 0) {
            decode_begins.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
        }
        if (llama_decode(context.get(), batch) != 0) throw probe_error("lead probe decode failed");
        llama_synchronize(context.get());
        const float * logits = llama_get_logits_ith(context.get(), -1);
        const int next = logits == nullptr ? -1 : finite_argmax(logits, vocabulary);
        if (next < 0) throw probe_error("lead probe logits were unavailable");
        generated = next;
        batch = llama_batch_get_one(&generated, 1);
    }
    context.reset();
    const auto diagnostics = provider->hot_cache_diagnostics();
    if (diagnostics.phase10_lead_events_dropped != 0 || diagnostics.phase10_lead_events.empty()) {
        throw probe_error("lead probe provider timing transcript is incomplete");
    }
    std::vector<uint64_t> token_end_ns;
    std::vector<uint64_t> cross_layer_ns;
    size_t decode_index = 0;
    size_t layer_index = 0;
    uint64_t previous = 0;
    for (const auto & event : diagnostics.phase10_lead_events) {
        if (layer_index == 0) {
            if (decode_index >= decode_begins.size() || event.layer != profile.target.routed_layers.front() ||
                    event.steady_ns < decode_begins[decode_index]) {
                throw probe_error("lead probe provider timing transcript is not canonical");
            }
            previous = event.steady_ns;
        } else {
            if (event.layer != profile.target.routed_layers[layer_index] || event.steady_ns < previous) {
                throw probe_error("lead probe provider layer timing is not canonical");
            }
            cross_layer_ns.push_back(event.steady_ns - previous);
            previous = event.steady_ns;
        }
        token_end_ns.push_back(event.steady_ns - decode_begins[decode_index]);
        layer_index++;
        if (layer_index == profile.target.routed_layers.size()) {
            layer_index = 0;
            decode_index++;
        }
    }
    if (layer_index != 0 || decode_index != decode_begins.size() || token_end_ns.empty() || cross_layer_ns.empty()) {
        throw probe_error("lead probe captured an incomplete decode timing matrix");
    }
    const uint64_t token_end_p50 = percentile(token_end_ns, 50, 100);
    const uint64_t cross_layer_p50 = percentile(cross_layer_ns, 50, 100);
    const uint64_t conservative_p50 = std::min(token_end_p50, cross_layer_p50);
    if (conservative_p50 == 0) throw probe_error("lead probe produced a zero conservative interval");
    result.lead_measurements = {{"basis", "minimum p50 of observed token-end-to-router and adjacent-router intervals"},
        {"decode_submissions", decode_begins.size()}, {"token_end_samples", token_end_ns.size()},
        {"cross_layer_samples", cross_layer_ns.size()}, {"token_end_p50_ns", token_end_p50},
        {"cross_layer_p50_ns", cross_layer_p50}, {"conservative_lead_p50_ns", conservative_p50},
        {"provider_event_capacity", diagnostics.phase10_lead_event_capacity},
        {"provider_events_dropped", diagnostics.phase10_lead_events_dropped}};
    return result;
}

json envelope(
        const std::string & transport,
        const std::string & readiness,
        bool supported,
        const std::vector<uint64_t> & service,
        uint64_t predictor_p95,
        uint64_t storage_bytes,
        uint64_t h2d_bytes,
        uint64_t lead_p50) {
    const uint64_t p50 = supported ? percentile(service, 50, 100) : 0;
    const uint64_t p95 = supported ? percentile(service, 95, 100) : 0;
    return {{"transport", transport}, {"readiness", readiness}, {"supported", supported},
        {"lead_p50_ns", lead_p50}, {"demand_service_p50_ns", p50}, {"speculative_service_p95_ns", p95},
        {"predictor_compute_p95_ns", predictor_p95}, {"scheduler_demand_delay_p95_ns", 0},
        {"displacement_refill_p95_ns", p95}, {"storage_bytes", storage_bytes}, {"h2d_bytes", h2d_bytes},
        {"utility_window_predictions", 64}, {"utility_min_observations", 32}};
}

} // namespace

int main(int argc, char ** argv) {
    try {
        const bool validate_only = argc == 8 && std::string(argv[7]) == "--validate-only";
        if ((argc != 7 && !validate_only) || std::string(argv[1]) != "--profile" ||
            std::string(argv[3]) != "--model" || std::string(argv[5]) != "--identity") {
            throw probe_error("usage: phase10-prefetch-probe --profile FILE --model GGUF --identity PROJECT:NESTED [--validate-only]");
        }
        const std::string profile_path = argv[2];
        const std::string model_path = argv[4];
        const std::string identity = argv[6];
        const size_t separator = identity.find(':');
        if (separator == std::string::npos) throw probe_error("identity must be PROJECT:NESTED");
        ggml_backend_load_all();
        llm_expert_prefetch_profile profile;
        std::string profile_error;
        const auto profile_begin = std::chrono::steady_clock::now();
        if (!llm_expert_prefetch_load_profile(profile_path, 64U*1024U*1024U, nullptr, profile, profile_error).is_ready()) {
            throw probe_error("profile load failed: " + profile_error);
        }
        const uint64_t profile_parse_ns = elapsed_ns(profile_begin);
        const auto model_validation = validate_model_profile(model_path, profile_path, profile, !validate_only);
        if (validate_only) {
            std::cout << json({ {"schema_version", "phase10-profile-validation-v1"},
                {"project_head", identity.substr(0, separator)}, {"nested_head", identity.substr(separator + 1)},
                {"profile_sha256", profile.profile_sha256}, {"profile_parse_ns", profile_parse_ns},
                {"model_profile_load_ns", model_validation.load_ns} }).dump(2) << '\n';
            return 0;
        }
        const uint64_t bytes = profile.target.expert_bytes.front().physical_bytes;
        const auto predictor = benchmark_predictor(profile, profile_path);
        const uint64_t predictor_p95 = predictor.upper_bound_p95_ns;
        bool buffered_supported = false, direct_supported = false, h2d_supported = false;
        const auto buffered = benchmark_pread(model_path, bytes, false, buffered_supported);
        const auto direct = benchmark_pread(model_path, bytes, true, direct_supported);
        const auto h2d = benchmark_h2d(bytes, h2d_supported);
        std::vector<uint64_t> buffered_device, direct_device;
        if (buffered_supported && h2d_supported) {
            for (size_t index = 0; index < std::min(buffered.size(), h2d.size()); ++index) buffered_device.push_back(buffered[index] + h2d[index]);
        }
        if (direct_supported && h2d_supported) {
            for (size_t index = 0; index < std::min(direct.size(), h2d.size()); ++index) direct_device.push_back(direct[index] + h2d[index]);
        }
        json output = { {"schema_version", "phase10-transport-measurements-v1"},
            {"project_head", identity.substr(0, separator)}, {"nested_head", identity.substr(separator + 1)},
            {"host", "skynet"}, {"profile_sha256", profile.profile_sha256},
            {"profile_parse_ns", profile_parse_ns}, {"model_profile_load_ns", model_validation.load_ns},
            {"lead_measurements", model_validation.lead_measurements},
            {"predictor_upper_bound", {{"basis", "maximum p95 over full-token topology-capped declared predictors"},
                {"upper_bound_p95_ns", predictor.upper_bound_p95_ns}, {"measurements", predictor.measurements}}},
            {"envelopes", json::array()} };
        const uint64_t lead_p50 = model_validation.lead_measurements.at("conservative_lead_p50_ns");
        output["envelopes"].push_back(envelope("BUFFERED", "HOST_READY", buffered_supported, buffered, predictor_p95, bytes, 0, lead_p50));
        output["envelopes"].push_back(envelope("DIRECT_IO", "HOST_READY", direct_supported, direct, predictor_p95, bytes, 0, lead_p50));
        output["envelopes"].push_back(envelope("HOST_TO_DEVICE", "DEVICE_READY", h2d_supported, h2d, predictor_p95, 0, bytes, lead_p50));
        output["envelopes"].push_back(envelope("BUFFERED", "DEVICE_READY", !buffered_device.empty(), buffered_device, predictor_p95, bytes, bytes, lead_p50));
        output["envelopes"].push_back(envelope("DIRECT_IO", "DEVICE_READY", !direct_device.empty(), direct_device, predictor_p95, bytes, bytes, lead_p50));
        std::cout << output.dump(2) << '\n';
        return 0;
    } catch (const std::exception & exception) {
        std::fprintf(stderr, "phase10-prefetch-probe: %s\n", exception.what());
        return 1;
    }
}
