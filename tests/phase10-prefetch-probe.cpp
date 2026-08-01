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
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
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
            profile.target.experts_per_layer, 64, UINT64_C(1) << 30, 128,
            UINT64_C(1) << 30,
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
    llm_hot_cache_diagnostics diagnostics;
};

struct expected_bundle_path {
    uint64_t bytes = 0;
    std::vector<std::pair<uint64_t, uint64_t>> spans;
};

struct expected_storage_map {
    std::string sha256;
    std::string model_sha256;
    uint64_t model_size = 0;
    std::map<std::pair<int32_t, int32_t>, expected_bundle_path> bundles;
};

std::string read_text(const std::string & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw probe_error("cannot open storage map");
    std::ostringstream buffer;
    buffer << input.rdbuf();
    if (!input.good() && !input.eof()) throw probe_error("cannot read storage map");
    return buffer.str();
}

expected_storage_map load_storage_map(
        const std::string & path,
        const llm_expert_prefetch_profile & profile) {
    const std::string bytes = read_text(path);
    const json document = json::parse(bytes);
    if (document.value("schema_version", "") != "expert-storage-map-v1") {
        throw probe_error("unsupported storage map");
    }
    expected_storage_map result;
    result.sha256 = llm_expert_prefetch_sha256(bytes.data(), bytes.size());
    result.model_sha256 = document.at("model").at("sha256");
    result.model_size = document.at("model").at("size");
    if (profile.target.files.size() != 1 || profile.target.files[0].sha256 != result.model_sha256 ||
            profile.target.files[0].size != result.model_size) {
        throw probe_error("storage map model identity differs from the profile");
    }
    for (const auto & entry : document.at("entries")) {
        const std::pair<int32_t, int32_t> key = { entry.at("layer"), entry.at("expert_id") };
        expected_bundle_path bundle;
        bundle.bytes = entry.at("atomic_bundle_bytes");
        for (const char * projection : { "gate", "up", "down" }) {
            for (const auto & span : entry.at("projections").at(projection).at("spans")) {
                bundle.spans.emplace_back(span.at("file_offset"), span.at("length"));
            }
        }
        std::sort(bundle.spans.begin(), bundle.spans.end());
        if (!result.bundles.emplace(key, std::move(bundle)).second) {
            throw probe_error("duplicate storage-map expert key");
        }
    }
    if (result.bundles.size() != profile.target.expert_bytes.size()) {
        throw probe_error("storage-map/profile key count mismatch");
    }
    return result;
}

using flight_key = std::tuple<uint64_t, uint32_t, uint64_t, int32_t, int32_t>;

flight_key exact_flight(const llm_expert_flight_id & flight) {
    return { flight.transport_epoch, flight.request_slot, flight.request_generation,
        flight.key.layer, flight.key.expert };
}

struct runtime_path_measurements {
    std::vector<uint64_t> storage_service_ns;
    std::vector<uint64_t> storage_refill_ns;
    std::vector<uint64_t> h2d_service_ns;
    std::vector<uint64_t> h2d_refill_ns;
    std::vector<uint64_t> device_service_ns;
    std::vector<uint64_t> device_refill_ns;
    std::vector<uint64_t> scheduler_delay_ns;
    json provenance;
};

runtime_path_measurements measure_runtime_paths(
        const llm_hot_cache_diagnostics & diagnostics,
        const expected_storage_map & expected,
        const char * load_mode) {
    if (diagnostics.phase10_lead_events_dropped != 0 ||
            diagnostics.phase10_scheduler_events_dropped != 0 ||
            diagnostics.phase10_storage_events_dropped != 0 ||
            diagnostics.phase10_h2d_events_dropped != 0) {
        throw probe_error("Phase 10 runtime measurement transcript overflowed");
    }
    runtime_path_measurements result;
    for (const auto & event : diagnostics.phase10_scheduler_events) {
        if (event.take_ns < event.enqueue_ns) throw probe_error("scheduler timing regressed");
        result.scheduler_delay_ns.push_back(event.take_ns - event.enqueue_ns);
    }

    struct storage_flight {
        llm_expert_key key = { -1, -1 };
        uint64_t begin_us = UINT64_MAX;
        uint64_t end_us = 0;
        uint64_t completed_bytes = 0;
        uint64_t useful_bytes = 0;
        std::vector<std::pair<uint64_t, uint64_t>> spans;
    };
    std::map<flight_key, storage_flight> storage;
    for (const auto & event : diagnostics.phase10_storage_events) {
        if (!event.flight.valid() || event.complete_us < event.submit_us || event.source_segment_count == 0) {
            throw probe_error("invalid runtime storage timing event");
        }
        auto & flight = storage[exact_flight(event.flight)];
        flight.key = event.flight.key;
        flight.begin_us = std::min(flight.begin_us, event.submit_us);
        flight.end_us = std::max(flight.end_us, event.complete_us);
        flight.completed_bytes += event.completed_bytes;
        flight.useful_bytes += event.useful_bytes;
        for (uint32_t index = 0; index < event.source_segment_count; ++index) {
            flight.spans.emplace_back(event.source_segments[index].file_offset,
                event.source_segments[index].byte_count);
        }
    }
    std::vector<std::pair<flight_key, storage_flight *>> ordered_storage;
    for (auto & item : storage) ordered_storage.push_back({ item.first, &item.second });
    std::sort(ordered_storage.begin(), ordered_storage.end(), [](const auto & lhs, const auto & rhs) {
        return std::tie(lhs.second->begin_us, lhs.first) < std::tie(rhs.second->begin_us, rhs.first);
    });
    std::map<std::pair<int32_t, int32_t>, uint32_t> storage_occurrences;
    json representative = nullptr;
    for (const auto & ordered : ordered_storage) {
        auto & flight = *ordered.second;
        const std::pair<int32_t, int32_t> key = { flight.key.layer, flight.key.expert };
        const auto found = expected.bundles.find(key);
        if (found == expected.bundles.end()) throw probe_error("runtime storage key is absent from the storage map");
        std::sort(flight.spans.begin(), flight.spans.end());
        if (flight.spans != found->second.spans || flight.useful_bytes != found->second.bytes ||
                flight.end_us <= flight.begin_us) {
            throw probe_error("runtime storage flight differs from exact expert bundle spans");
        }
        const uint64_t duration_ns = (flight.end_us - flight.begin_us)*1000;
        result.storage_service_ns.push_back(duration_ns);
        if (storage_occurrences[key]++ != 0) result.storage_refill_ns.push_back(duration_ns);
        if (representative.is_null()) {
            representative = { {"layer", key.first}, {"expert", key.second},
                {"useful_bytes", flight.useful_bytes}, {"completed_bytes", flight.completed_bytes},
                {"spans", json::array()} };
            for (const auto & span : flight.spans) representative["spans"].push_back(
                { {"file_offset", span.first}, {"length", span.second} });
        }
    }

    std::vector<const llm_expert_phase10_h2d_event *> transfers;
    for (const auto & event : diagnostics.phase10_h2d_events) {
        if (!event.cancelled && event.flight.valid() && event.complete_us > event.enqueue_us) transfers.push_back(&event);
    }
    std::sort(transfers.begin(), transfers.end(), [](const auto * lhs, const auto * rhs) {
        return std::tie(lhs->enqueue_us, lhs->flight.request_slot, lhs->flight.request_generation) <
            std::tie(rhs->enqueue_us, rhs->flight.request_slot, rhs->flight.request_generation);
    });
    std::map<std::pair<int32_t, int32_t>, uint32_t> transfer_occurrences;
    for (const auto * transfer : transfers) {
        const std::pair<int32_t, int32_t> key = { transfer->flight.key.layer, transfer->flight.key.expert };
        const auto found = expected.bundles.find(key);
        if (found == expected.bundles.end() || transfer->bytes != found->second.bytes) {
            throw probe_error("runtime H2D flight differs from exact expert bundle bytes");
        }
        const uint64_t h2d_ns = (transfer->complete_us - transfer->enqueue_us)*1000;
        result.h2d_service_ns.push_back(h2d_ns);
        const bool refill = transfer_occurrences[key]++ != 0;
        if (refill) result.h2d_refill_ns.push_back(h2d_ns);
        uint64_t begin_us = transfer->enqueue_us;
        const auto storage_found = storage.find(exact_flight(transfer->flight));
        if (storage_found != storage.end()) begin_us = std::min(begin_us, storage_found->second.begin_us);
        if (transfer->complete_us <= begin_us) throw probe_error("combined runtime service timing is invalid");
        const uint64_t device_ns = (transfer->complete_us - begin_us)*1000;
        result.device_service_ns.push_back(device_ns);
        if (refill) result.device_refill_ns.push_back(device_ns);
    }
    result.provenance = { {"load_mode", load_mode}, {"path", "runtime COLD_CACHE scheduler -> exact ExpertStorage read plan -> cold cache -> transfer ring -> hot cache"},
        {"storage_map_sha256", expected.sha256}, {"model_sha256", expected.model_sha256},
        {"model_size", expected.model_size}, {"all_observed_spans_exact", true},
        {"storage_operations", diagnostics.phase10_storage_events.size()},
        {"storage_flights", storage.size()}, {"h2d_flights", transfers.size()},
        {"scheduler_samples", result.scheduler_delay_ns.size()},
        {"storage_refill_samples", result.storage_refill_ns.size()},
        {"h2d_refill_samples", result.h2d_refill_ns.size()},
        {"storage_trace_capacity", diagnostics.phase10_storage_event_capacity},
        {"h2d_trace_capacity", diagnostics.phase10_h2d_event_capacity},
        {"scheduler_trace_capacity", diagnostics.phase10_scheduler_event_capacity},
        {"representative_bundle", representative} };
    return result;
}

model_profile_validation validate_model_profile(
        const std::string & model_path,
        const std::string & profile_path,
        const llm_expert_prefetch_profile & profile,
        llama_load_mode load_mode,
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
    const uint64_t bundle_bytes = profile.target.expert_bytes.front().physical_bytes;
    if (bundle_bytes == 0 || bundle_bytes > UINT64_MAX/16) throw probe_error("invalid profile bundle bytes");
    const uint64_t speculative_bytes = 4*bundle_bytes;
    llama_expert_prefetch_config_v1 prefetch = {
        LLAMA_EXPERT_PREFETCH_VERSION_1, sizeof(llama_expert_prefetch_config_v1),
        selected_policy, profile.selected_readiness,
        seed ? LLAMA_EXPERT_PREFETCH_SEED_MODE_BLOCKING_HOT : LLAMA_EXPERT_PREFETCH_SEED_MODE_OFF,
        profile.selected_temporal_window, seed ? 0U : profile.selected_candidates,
        64U*1024U*1024U, seed ? 0U : 4U, seed ? 0U : speculative_bytes,
        seed ? 0U : speculative_bytes, seed ? 0U : speculative_bytes,
        seed ? 0U : speculative_bytes, seed ? 0U : 4U, seed ? 0U : 4U,
        seed ? 0U : selected_cost->utility_window_predictions,
        seed ? 0U : selected_cost->utility_min_observations, {},
    };
    auto params = llama_model_default_params();
    params.load_mode = load_mode;
    params.n_gpu_layers = -1;
    params.tensor_buft_overrides = overrides;
    params.expert_weights_mode = LLAMA_EXPERT_WEIGHTS_MODE_COLD_CACHE;
    params.expert_hot_cache_capacity = 16;
    params.expert_cold_cache_bytes = 16*bundle_bytes;
    params.expert_transfer_ring_bytes = 4*bundle_bytes;
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
    for (int step = 0; step < 13; ++step) {
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
    result.diagnostics = diagnostics;
    return result;
}

json envelope(
        const std::string & transport,
        const std::string & readiness,
        bool supported,
        const std::vector<uint64_t> & service,
        const std::vector<uint64_t> & refill,
        const std::vector<uint64_t> & scheduler,
        uint64_t predictor_p95,
        uint64_t storage_bytes,
        uint64_t h2d_bytes,
        uint64_t lead_p50,
        const json & measurement_basis) {
    supported = supported && !service.empty() && !refill.empty() && !scheduler.empty();
    const uint64_t p50 = supported ? percentile(service, 50, 100) : 0;
    const uint64_t p95 = supported ? percentile(service, 95, 100) : 0;
    return {{"transport", transport}, {"readiness", readiness}, {"supported", supported},
        {"lead_p50_ns", lead_p50}, {"demand_service_p50_ns", p50}, {"speculative_service_p95_ns", p95},
        {"predictor_compute_p95_ns", predictor_p95},
        {"scheduler_demand_delay_p95_ns", supported ? percentile(scheduler, 95, 100) : 0},
        {"displacement_refill_p95_ns", supported ? percentile(refill, 95, 100) : 0},
        {"storage_bytes", storage_bytes}, {"h2d_bytes", h2d_bytes},
        {"utility_window_predictions", 64}, {"utility_min_observations", 32},
        {"measurement_basis", measurement_basis}};
}

} // namespace

int main(int argc, char ** argv) {
    try {
        const bool validate_only = argc == 8 && std::string(argv[7]) == "--validate-only";
        if (((argc != 9 || std::string(argv[7]) != "--storage-map") && !validate_only) ||
                std::string(argv[1]) != "--profile" || std::string(argv[3]) != "--model" ||
                std::string(argv[5]) != "--identity") {
            throw probe_error("usage: phase10-prefetch-probe --profile FILE --model GGUF --identity PROJECT:NESTED --storage-map FILE | --validate-only");
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
        const auto model_validation = validate_model_profile(
            model_path, profile_path, profile, LLAMA_LOAD_MODE_MMAP, !validate_only);
        if (validate_only) {
            std::cout << json({ {"schema_version", "phase10-profile-validation-v1"},
                {"project_head", identity.substr(0, separator)}, {"nested_head", identity.substr(separator + 1)},
                {"profile_sha256", profile.profile_sha256}, {"profile_parse_ns", profile_parse_ns},
                {"model_profile_load_ns", model_validation.load_ns} }).dump(2) << '\n';
            return 0;
        }
        const uint64_t bytes = profile.target.expert_bytes.front().physical_bytes;
        const expected_storage_map storage_map = load_storage_map(argv[8], profile);
        const auto predictor = benchmark_predictor(profile, profile_path);
        const uint64_t predictor_p95 = predictor.upper_bound_p95_ns;
        const auto buffered = measure_runtime_paths(model_validation.diagnostics, storage_map, "MMAP_BUFFERED");
        std::optional<runtime_path_measurements> direct;
        std::string direct_unavailable_reason;
        try {
            const auto direct_validation = validate_model_profile(
                model_path, profile_path, profile, LLAMA_LOAD_MODE_DIRECT_IO, true);
            direct = measure_runtime_paths(direct_validation.diagnostics, storage_map, "DIRECT_IO");
        } catch (const std::exception & error) {
            direct_unavailable_reason = error.what();
        }
        json output = { {"schema_version", "phase10-transport-measurements-v1"},
            {"project_head", identity.substr(0, separator)}, {"nested_head", identity.substr(separator + 1)},
            {"host", "skynet"}, {"profile_sha256", profile.profile_sha256},
            {"profile_parse_ns", profile_parse_ns}, {"model_profile_load_ns", model_validation.load_ns},
            {"lead_measurements", model_validation.lead_measurements},
            {"predictor_upper_bound", {{"basis", "maximum p95 over full-token topology-capped declared predictors"},
                {"upper_bound_p95_ns", predictor.upper_bound_p95_ns}, {"measurements", predictor.measurements}}},
            {"path_provenance", {{"storage_map_sha256", storage_map.sha256},
                {"model_sha256", storage_map.model_sha256}, {"model_size", storage_map.model_size},
                {"exact_runtime_provider_path", true}, {"buffered", buffered.provenance},
                {"direct", direct ? direct->provenance : json({{"supported", false},
                    {"reason", direct_unavailable_reason}})}}},
            {"envelopes", json::array()} };
        const uint64_t lead_p50 = model_validation.lead_measurements.at("conservative_lead_p50_ns");
        output["envelopes"].push_back(envelope("BUFFERED", "HOST_READY", true,
            buffered.storage_service_ns, buffered.storage_refill_ns, buffered.scheduler_delay_ns,
            predictor_p95, bytes, 0, lead_p50, buffered.provenance));
        output["envelopes"].push_back(envelope("DIRECT_IO", "HOST_READY", direct.has_value(),
            direct ? direct->storage_service_ns : std::vector<uint64_t>{},
            direct ? direct->storage_refill_ns : std::vector<uint64_t>{},
            direct ? direct->scheduler_delay_ns : std::vector<uint64_t>{},
            predictor_p95, bytes, 0, lead_p50,
            direct ? direct->provenance : output["path_provenance"]["direct"]));
        output["envelopes"].push_back(envelope("HOST_TO_DEVICE", "DEVICE_READY", true,
            buffered.h2d_service_ns, buffered.h2d_refill_ns, buffered.scheduler_delay_ns,
            predictor_p95, 0, bytes, lead_p50, buffered.provenance));
        output["envelopes"].push_back(envelope("BUFFERED", "DEVICE_READY", true,
            buffered.device_service_ns, buffered.device_refill_ns, buffered.scheduler_delay_ns,
            predictor_p95, bytes, bytes, lead_p50, buffered.provenance));
        output["envelopes"].push_back(envelope("DIRECT_IO", "DEVICE_READY", direct.has_value(),
            direct ? direct->device_service_ns : std::vector<uint64_t>{},
            direct ? direct->device_refill_ns : std::vector<uint64_t>{},
            direct ? direct->scheduler_delay_ns : std::vector<uint64_t>{},
            predictor_p95, bytes, bytes, lead_p50,
            direct ? direct->provenance : output["path_provenance"]["direct"]));
        std::cout << output.dump(2) << '\n';
        return 0;
    } catch (const std::exception & exception) {
        std::fprintf(stderr, "phase10-prefetch-probe: %s\n", exception.what());
        return 1;
    }
}
