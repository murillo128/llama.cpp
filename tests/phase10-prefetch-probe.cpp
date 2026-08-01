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
    std::vector<int32_t> generated_tokens;
    std::vector<std::string> logit_sha256;
    llm_hot_cache_diagnostics initial_diagnostics;
    llm_hot_cache_diagnostics diagnostics;
};

struct online_runtime_options {
    llama_expert_weights_mode cache_mode = LLAMA_EXPERT_WEIGHTS_MODE_COLD_CACHE;
    llama_load_mode load_mode = LLAMA_LOAD_MODE_MMAP;
    llama_expert_miss_policy miss_policy = LLAMA_EXPERT_MISS_POLICY_PROMOTE_AND_GPU;
    uint32_t hot_slots = 16;
    uint32_t cold_slots = 16;
};

struct expected_bundle_path {
    uint64_t bytes = 0;
    std::vector<std::pair<uint64_t, uint64_t>> spans;
};

struct expected_storage_map {
    std::string sha256;
    std::string package_sha256;
    std::string model_sha256;
    uint64_t model_size = 0;
    uint64_t source_file_count = 0;
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
    const auto & model = document.at("model");
    result.package_sha256 = model.value("package_sha256", "");
    result.source_file_count = document.at("source_files").size();
    if (result.package_sha256.empty()) {
        result.model_sha256 = model.at("sha256");
        result.model_size = model.at("size");
        if (profile.target.files.size() != 1 || profile.target.files[0].sha256 != result.model_sha256 ||
                profile.target.files[0].size != result.model_size) {
            throw probe_error("storage map model identity differs from the profile");
        }
        result.package_sha256 = profile.target.package_sha256;
    } else {
        if (result.package_sha256 != profile.target.package_sha256 ||
                result.source_file_count != profile.target.files.size()) {
            throw probe_error("storage map package identity differs from the profile");
        }
        result.model_size = model.at("size");
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
    uint64_t storage_timer_censored_samples = 0;
    uint64_t h2d_timer_censored_samples = 0;
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
                flight.end_us < flight.begin_us) {
            json observed_spans = json::array();
            json expected_spans = json::array();
            for (const auto & span : flight.spans) observed_spans.push_back({span.first, span.second});
            for (const auto & span : found->second.spans) expected_spans.push_back({span.first, span.second});
            throw probe_error("runtime storage flight differs from exact expert bundle spans: key=" +
                std::to_string(key.first) + ":" + std::to_string(key.second) + " observed=" +
                observed_spans.dump() + " expected=" + expected_spans.dump() + " useful=" +
                std::to_string(flight.useful_bytes) + "/" + std::to_string(found->second.bytes) +
                " duration_us=" + std::to_string(flight.end_us - flight.begin_us));
        }
        const uint64_t duration_us = flight.end_us - flight.begin_us;
        result.storage_timer_censored_samples += duration_us == 0;
        const uint64_t duration_ns = std::max<uint64_t>(1, duration_us)*1000;
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
        if (!event.cancelled && event.flight.valid() && event.complete_us >= event.enqueue_us &&
                event.complete_us != 0) transfers.push_back(&event);
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
        const uint64_t h2d_us = transfer->complete_us - transfer->enqueue_us;
        result.h2d_timer_censored_samples += h2d_us == 0;
        const uint64_t h2d_ns = std::max<uint64_t>(1, h2d_us)*1000;
        result.h2d_service_ns.push_back(h2d_ns);
        const bool refill = transfer_occurrences[key]++ != 0;
        if (refill) result.h2d_refill_ns.push_back(h2d_ns);
        uint64_t begin_us = transfer->enqueue_us;
        const auto storage_found = storage.find(exact_flight(transfer->flight));
        if (storage_found != storage.end()) begin_us = std::min(begin_us, storage_found->second.begin_us);
        if (transfer->complete_us < begin_us) throw probe_error("combined runtime service timing is invalid");
        const uint64_t device_ns = std::max<uint64_t>(1, transfer->complete_us - begin_us)*1000;
        result.device_service_ns.push_back(device_ns);
        if (refill) result.device_refill_ns.push_back(device_ns);
    }
    result.provenance = { {"load_mode", load_mode}, {"path", "runtime COLD_CACHE scheduler -> exact ExpertStorage read plan -> cold cache -> transfer ring -> hot cache"},
        {"storage_map_sha256", expected.sha256}, {"package_sha256", expected.package_sha256},
        {"model_sha256", expected.model_sha256}, {"model_size", expected.model_size},
        {"source_file_count", expected.source_file_count}, {"all_observed_spans_exact", true},
        {"storage_operations", diagnostics.phase10_storage_events.size()},
        {"storage_flights", storage.size()}, {"h2d_flights", transfers.size()},
        {"scheduler_samples", result.scheduler_delay_ns.size()},
        {"storage_timer_censored_samples", result.storage_timer_censored_samples},
        {"h2d_timer_censored_samples", result.h2d_timer_censored_samples},
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
        bool measure_lead,
        bool enable_profile = true,
        const online_runtime_options & runtime = {},
        bool calculate_lead = true) {
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
    const llama_expert_auto_cost_model auto_cost = {
        LLAMA_EXPERT_AUTO_COST_MODEL_VERSION_1,
        sizeof(llama_expert_auto_cost_model),
        1, 1, 1, 1,
        UINT64_C(1000000000), UINT64_C(1000000000),
        UINT64_C(1000000000), UINT64_C(1000000000),
        UINT64_C(1000000000), 1, 1,
    };
    auto params = llama_model_default_params();
    params.load_mode = runtime.load_mode == LLAMA_LOAD_MODE_MMAP ? load_mode : runtime.load_mode;
    params.n_gpu_layers = -1;
    params.tensor_buft_overrides = overrides;
    params.expert_weights_mode = runtime.cache_mode;
    params.expert_hot_cache_capacity = runtime.hot_slots;
    params.expert_cold_cache_bytes = runtime.cache_mode == LLAMA_EXPERT_WEIGHTS_MODE_COLD_CACHE ?
        uint64_t(runtime.cold_slots)*bundle_bytes : 0;
    params.expert_transfer_ring_bytes = runtime.cache_mode == LLAMA_EXPERT_WEIGHTS_MODE_COLD_CACHE ?
        4*bundle_bytes : 0;
    params.expert_miss_policy = runtime.miss_policy;
    params.expert_auto_cost_model = runtime.miss_policy == LLAMA_EXPERT_MISS_POLICY_AUTO ?
        &auto_cost : nullptr;
    params.expert_prefetch_config = enable_profile ? &prefetch : nullptr;
    params.expert_prefetch_profile_path = enable_profile ? profile_path.c_str() : nullptr;
    const auto begin = std::chrono::steady_clock::now();
    llama_model_ptr model(llama_model_load_from_file(model_path.c_str(), params));
    const uint64_t duration = elapsed_ns(begin);
    if (!model) throw probe_error("active profile model load failed");
    model_profile_validation result;
    result.load_ns = duration;
    if (!measure_lead) return result;

    const auto * vocab = llama_model_get_vocab(model.get());
    const std::string prompt_text = profile.target.routed_layers.size() > 8 ? "A" :
        "According to all known laws";
    const int prompt_count = -llama_tokenize(vocab, prompt_text.data(), prompt_text.size(), nullptr, 0, true, true);
    if (prompt_count <= 0) throw probe_error("lead probe prompt tokenization failed");
    std::vector<llama_token> prompt(prompt_count);
    if (llama_tokenize(vocab, prompt_text.data(), prompt_text.size(), prompt.data(), prompt.size(), true, true) != prompt_count) {
        throw probe_error("lead probe prompt tokenization changed");
    }
    auto context_params = llama_context_default_params();
    context_params.n_ctx = 64;
    context_params.n_batch = 64;
    context_params.n_ubatch = std::max<uint32_t>(1, std::min<uint32_t>(
        64, runtime.hot_slots/profile.target.experts_per_token));
    context_params.no_perf = false;
    llama_context_ptr context(llama_init_from_model(model.get(), context_params));
    if (!context) throw probe_error("lead probe context creation failed");
    auto * provider = model->expert_weight_provider();
    if (provider == nullptr) throw probe_error("lead probe expert provider is unavailable");
    llama_batch batch = llama_batch_get_one(prompt.data(), prompt.size());
    llama_token generated = 0;
    const int vocabulary = llama_vocab_n_tokens(vocab);
    std::vector<uint64_t> decode_begins;
    const int online_steps = profile.target.routed_layers.size() > 8 ? 2 : 13;
    for (int step = 0; step < online_steps; ++step) {
        if (step != 0) {
            decode_begins.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
        }
        if (llama_decode(context.get(), batch) != 0) throw probe_error("lead probe decode failed");
        llama_synchronize(context.get());
        if (step == 0) result.initial_diagnostics = provider->hot_cache_diagnostics();
        const float * logits = llama_get_logits_ith(context.get(), -1);
        const int next = logits == nullptr ? -1 : finite_argmax(logits, vocabulary);
        if (next < 0) throw probe_error("lead probe logits were unavailable");
        generated = next;
        result.generated_tokens.push_back(next);
        result.logit_sha256.push_back(llm_expert_prefetch_sha256(
            logits, size_t(vocabulary)*sizeof(float)));
        batch = llama_batch_get_one(&generated, 1);
    }
    context.reset();
    const auto diagnostics = provider->hot_cache_diagnostics();
    result.diagnostics = diagnostics;
    if (enable_profile && calculate_lead &&
            (diagnostics.phase10_lead_events_dropped != 0 ||
             diagnostics.phase10_lead_events.empty())) {
        throw probe_error("lead probe provider timing transcript is incomplete");
    }
    if (!enable_profile || !calculate_lead) {
        result.lead_measurements = json::object();
        return result;
    }
    std::vector<uint64_t> token_end_ns;
    std::vector<uint64_t> cross_layer_ns;
    size_t decode_index = 0;
    size_t layer_index = 0;
    uint64_t previous = 0;
    for (const auto & event : diagnostics.phase10_lead_events) {
        if (!decode_begins.empty() && event.steady_ns < decode_begins.front()) {
            continue;
        }
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

const char * outcome_name(llm_expert_prefetch_outcome outcome) {
    switch (outcome) {
        case llm_expert_prefetch_outcome::pending:             return "PENDING";
        case llm_expert_prefetch_outcome::timely_useful:       return "TIMELY_USEFUL";
        case llm_expert_prefetch_outcome::late_joined:         return "LATE_JOINED";
        case llm_expert_prefetch_outcome::wasted_unused:       return "WASTED_UNUSED";
        case llm_expert_prefetch_outcome::cancelled_before_io: return "CANCELLED_BEFORE_IO";
        case llm_expert_prefetch_outcome::cancelled_drained:   return "CANCELLED_DRAINED";
        case llm_expert_prefetch_outcome::rejected:            return "REJECTED";
    }
    throw probe_error("unknown prediction outcome");
}

const char * origin_name(llm_expert_residency_origin origin) {
    switch (origin) {
        case llm_expert_residency_origin::demand:      return "DEMAND";
        case llm_expert_residency_origin::static_seed: return "STATIC_SEED";
        case llm_expert_residency_origin::speculative: return "SPECULATIVE";
    }
    throw probe_error("unknown residency origin");
}

json initial_resident(const llm_hot_cache_diagnostics & diagnostics) {
    using key_type = std::pair<int32_t, int32_t>;
    std::map<key_type, json> entries;
    for (size_t slot = 0; slot < diagnostics.cold_slots.size(); ++slot) {
        const auto & entry = diagnostics.cold_slots[slot];
        if (entry.state == llm_hot_cache_diagnostics::cold_slot::free) continue;
        if (entry.state != llm_hot_cache_diagnostics::cold_slot::ready ||
                entry.layer < 0 || entry.expert < 0 || entry.generation == 0) {
            throw probe_error("initial cold cache snapshot is not quiescent");
        }
        const key_type key = {entry.layer, entry.expert};
        if (!entries.emplace(key, json{
                {"layer", entry.layer}, {"expert", entry.expert},
                {"cold_slot", int32_t(slot)}, {"hot_slot", -1},
                {"cold_generation", entry.generation}, {"hot_generation", 0},
                {"cold_last_use", entry.last_use}, {"hot_last_use", 0},
                {"origin", origin_name(entry.origin)},
            }).second) {
            throw probe_error("duplicate key in initial cold cache snapshot");
        }
    }
    for (size_t slot = 0; slot < diagnostics.slots.size(); ++slot) {
        const auto & entry = diagnostics.slots[slot];
        if (entry.state == llm_hot_cache_diagnostics::slot::free) continue;
        if ((entry.state != llm_hot_cache_diagnostics::slot::ready &&
                entry.state != llm_hot_cache_diagnostics::slot::pinned) ||
                entry.layer < 0 || entry.expert < 0 || entry.generation == 0) {
            throw probe_error("initial hot cache snapshot is not quiescent");
        }
        const key_type key = {entry.layer, entry.expert};
        auto found = entries.find(key);
        if (found == entries.end()) {
            if (entry.has_cold_backing) {
                throw probe_error("initial hot cache backing is absent from cold snapshot");
            }
            found = entries.emplace(key, json{
                {"layer", entry.layer}, {"expert", entry.expert},
                {"cold_slot", -1}, {"hot_slot", int32_t(slot)},
                {"cold_generation", 0}, {"hot_generation", entry.generation},
                {"cold_last_use", 0}, {"hot_last_use", entry.last_use},
                {"origin", origin_name(entry.origin)},
            }).first;
        } else {
            if (!entry.has_cold_backing || entry.cold_slot != uint32_t(found->second.at("cold_slot")) ||
                    entry.cold_generation != found->second.at("cold_generation").get<uint64_t>() ||
                    found->second.at("origin") != origin_name(entry.origin)) {
                throw probe_error("initial hot/cold cache snapshot disagrees");
            }
            found->second["hot_slot"] = int32_t(slot);
            found->second["hot_generation"] = entry.generation;
            found->second["hot_last_use"] = entry.last_use;
        }
    }
    json result = json::array();
    for (auto & item : entries) result.push_back(std::move(item.second));
    return result;
}

json online_capture(
        const model_profile_validation & validation,
        const llm_expert_prefetch_profile & profile,
        const std::string & identity,
        bool enabled,
        const online_runtime_options & runtime) {
    const size_t separator = identity.find(':');
    if (separator == std::string::npos) throw probe_error("identity must be PROJECT:NESTED");
    const auto & diagnostics = validation.diagnostics;
    if (diagnostics.phase10_prediction_events_dropped != 0 ||
            diagnostics.phase10_route_events_dropped != 0) {
        throw probe_error("online prediction transcript overflowed");
    }
    if (enabled && (diagnostics.phase10_route_events == 0 ||
                    diagnostics.phase10_prediction_events == 0)) {
        throw probe_error("active online prediction transcript is empty: routes=" +
            std::to_string(diagnostics.phase10_route_events) + " predictions=" +
            std::to_string(diagnostics.phase10_prediction_events) + " circuit=" +
            std::to_string(diagnostics.phase10_circuit_open) + " runtime_failed=" +
            std::to_string(diagnostics.phase10_runtime_failed));
    }
    if (!enabled && (diagnostics.phase10_route_events != 0 ||
                     diagnostics.phase10_prediction_events != 0)) {
        throw probe_error("disabled online run created prediction state");
    }
    json routes = json::array();
    for (const auto & event : diagnostics.phase10_route_trace) {
        if (event.id_offset > diagnostics.phase10_route_ids.size() ||
                event.id_count > diagnostics.phase10_route_ids.size() - event.id_offset) {
            throw probe_error("online route transcript is malformed");
        }
        json ids = json::array();
        for (uint32_t index = 0; index < event.id_count; ++index) {
            ids.push_back(diagnostics.phase10_route_ids[event.id_offset + index]);
        }
        routes.push_back({{"request", event.request}, {"token", event.token},
            {"layer", event.layer}, {"selected_experts", ids},
            {"post_event_digest", event.post_event_digest}});
    }
    json predictions = json::array();
    for (const auto & event : diagnostics.phase10_prediction_trace) {
        predictions.push_back({
            {"sequence", event.sequence}, {"request", event.request},
            {"token", event.token}, {"deadline_token", event.deadline_token},
            {"source_layer", event.source_layer}, {"target_layer", event.key.layer},
            {"expert", event.key.expert}, {"rank", event.rank}, {"score", event.score},
            {"trigger", event.trigger == llm_expert_prefetch_trigger::token_end ?
                "TOKEN_END" : "ROUTER_RESULT"},
            {"readiness", event.readiness == LLAMA_EXPERT_PREFETCH_READINESS_HOST_READY ?
                "HOST_READY" : "DEVICE_READY"},
            {"priority", event.priority}, {"config_digest", event.config_digest},
            {"predictor_digest", event.predictor_digest},
            {"post_event_digest", event.post_event_digest},
            {"outcome", outcome_name(event.outcome)}, {"admitted", event.admitted},
            {"demand_claimed", event.demand_claimed}, {"cold_ready", event.cold_ready},
            {"device_ready", event.device_ready}, {"circuit_open_after", event.circuit_open_after},
            {"storage_bytes", event.storage_bytes}, {"h2d_bytes", event.h2d_bytes},
            {"cold_slot", event.cold_slot}, {"cold_generation", event.cold_generation},
            {"hot_slot", event.hot_slot}, {"hot_generation", event.hot_generation},
            {"scheduler_slot", event.scheduler_slot},
            {"scheduler_generation", event.scheduler_generation},
            {"timing", {{"predictor_compute_ns", event.predictor_compute_ns},
                {"enqueue_us", event.enqueue_us}, {"host_ready_us", event.host_ready_us},
                {"device_ready_us", event.device_ready_us}}},
        });
    }
    return {{"schema_version", "phase10-online-capture-v1"},
        {"project_head", identity.substr(0, separator)},
        {"nested_head", identity.substr(separator + 1)},
        {"profile_enabled", enabled}, {"profile_sha256", profile.profile_sha256},
        {"cache_mode", runtime.cache_mode == LLAMA_EXPERT_WEIGHTS_MODE_COLD_CACHE ?
            "COLD_CACHE" : "HOT_CACHE"},
        {"load_mode", runtime.load_mode == LLAMA_LOAD_MODE_DIRECT_IO ? "DIRECT_IO" : "BUFFERED"},
        {"miss_policy", runtime.miss_policy == LLAMA_EXPERT_MISS_POLICY_PROMOTE_AND_GPU ?
            "PROMOTE_AND_GPU" : (runtime.miss_policy == LLAMA_EXPERT_MISS_POLICY_CPU_FALLBACK ?
                "CPU_FALLBACK" : "AUTO")},
        {"hot_slots", runtime.hot_slots}, {"cold_slots", runtime.cold_slots},
        {"model_load_ns", validation.load_ns},
        {"generated_tokens", validation.generated_tokens},
        {"logit_sha256", validation.logit_sha256},
        {"initial_resident", initial_resident(validation.initial_diagnostics)},
        {"routes", routes}, {"predictions", predictions},
        {"summary", {
            {"route_events", diagnostics.phase10_route_events},
            {"prediction_events", diagnostics.phase10_prediction_events},
            {"admitted", diagnostics.phase10_predictions_admitted},
            {"rejected", diagnostics.phase10_predictions_rejected},
            {"timely_useful", diagnostics.phase10_timely_useful},
            {"late_joined", diagnostics.phase10_late_joined},
            {"wasted_unused", diagnostics.phase10_wasted_unused},
            {"cancelled_before_io", diagnostics.phase10_cancelled_before_io},
            {"cancelled_drained", diagnostics.phase10_cancelled_drained},
            {"predictor_compute_ns", diagnostics.phase10_predictor_compute_ns},
            {"predictor_digest", diagnostics.phase10_predictor_digest},
            {"predictor_state_digest", diagnostics.phase10_predictor_state_digest},
            {"circuit_opens", diagnostics.phase10_circuit_opens},
            {"circuit_open", diagnostics.phase10_circuit_open},
            {"runtime_failed", diagnostics.phase10_runtime_failed},
            {"active_background_flights", diagnostics.active_background_flights},
            {"current_pins", diagnostics.current_pins},
        }}};
}

json capture_fingerprint(const std::string & model_path) {
    auto params = llama_model_default_params();
    params.n_gpu_layers = 0;
    params.expert_weights_mode = LLAMA_EXPERT_WEIGHTS_MODE_DISABLED;
    llama_model_ptr model(llama_model_load_from_file(model_path.c_str(), params));
    if (!model) throw probe_error("fingerprint model load failed");
    const auto fingerprint = model->expert_prefetch_fingerprint();
    json files = json::array();
    for (const auto & file : fingerprint.files) {
        files.push_back({{"ordinal", file.ordinal}, {"name", file.name},
            {"size", file.size}, {"sha256", file.sha256}});
    }
    json expert_bytes = json::array();
    for (const auto & key : fingerprint.expert_bytes) {
        expert_bytes.push_back({{"layer", key.layer}, {"expert", key.expert},
            {"payload_bytes", key.payload_bytes}, {"physical_bytes", key.physical_bytes}});
    }
    return {{"schema_version", "phase10-target-fingerprint-v1"}, {"target", {
        {"package_sha256", fingerprint.package_sha256}, {"files", files},
        {"layer_count", fingerprint.layer_count}, {"routed_layers", fingerprint.routed_layers},
        {"experts_per_layer", fingerprint.experts_per_layer},
        {"experts_per_token", fingerprint.experts_per_token},
        {"tensor_layout_sha256", fingerprint.tensor_layout_sha256},
        {"expert_bytes", expert_bytes},
    }}};
}

} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc == 4 && std::string(argv[1]) == "--fingerprint" && std::string(argv[2]) == "--model") {
            ggml_backend_load_all();
            std::cout << capture_fingerprint(argv[3]).dump(2) << '\n';
            return 0;
        }
        if (argc < 8) {
            throw probe_error("usage: phase10-prefetch-probe --fingerprint --model GGUF | --profile FILE --model GGUF --identity PROJECT:NESTED --storage-map FILE | --validate-only | (--online | --online-disabled) [HOT_CACHE|COLD_CACHE BUFFERED|DIRECT_IO PROMOTE_AND_GPU|CPU_FALLBACK|AUTO HOT_SLOTS COLD_SLOTS]");
        }
        const bool validate_only_mode = std::string(argv[7]) == "--validate-only";
        const bool validate_only = validate_only_mode && (argc == 8 || argc == 13);
        const bool online_mode = argc >= 8 && std::string(argv[7]) == "--online";
        const bool online_disabled_mode = argc >= 8 && std::string(argv[7]) == "--online-disabled";
        const bool online = online_mode && (argc == 8 || argc == 13);
        const bool online_disabled = online_disabled_mode && (argc == 8 || argc == 13);
        if (((argc != 9 || std::string(argv[7]) != "--storage-map") &&
                !validate_only && !online && !online_disabled) ||
                std::string(argv[1]) != "--profile" || std::string(argv[3]) != "--model" ||
                std::string(argv[5]) != "--identity") {
            throw probe_error("usage: phase10-prefetch-probe --fingerprint --model GGUF | --profile FILE --model GGUF --identity PROJECT:NESTED --storage-map FILE | --validate-only | (--online | --online-disabled) [HOT_CACHE|COLD_CACHE BUFFERED|DIRECT_IO PROMOTE_AND_GPU|CPU_FALLBACK|AUTO HOT_SLOTS COLD_SLOTS]");
        }
        const std::string profile_path = argv[2];
        const std::string model_path = argv[4];
        const std::string identity = argv[6];
        const size_t separator = identity.find(':');
        if (separator == std::string::npos) throw probe_error("identity must be PROJECT:NESTED");
        online_runtime_options runtime;
        if ((validate_only || online || online_disabled) && argc == 13) {
            const std::string cache_mode = argv[8];
            const std::string runtime_load_mode = argv[9];
            const std::string miss_policy = argv[10];
            char * hot_end = nullptr;
            char * cold_end = nullptr;
            const unsigned long hot_slots = std::strtoul(argv[11], &hot_end, 10);
            const unsigned long cold_slots = std::strtoul(argv[12], &cold_end, 10);
            if (cache_mode == "HOT_CACHE") runtime.cache_mode = LLAMA_EXPERT_WEIGHTS_MODE_HOT_CACHE;
            else if (cache_mode != "COLD_CACHE") throw probe_error("unknown online cache mode");
            if (runtime_load_mode == "DIRECT_IO") runtime.load_mode = LLAMA_LOAD_MODE_DIRECT_IO;
            else if (runtime_load_mode != "BUFFERED") throw probe_error("unknown online load mode");
            if (miss_policy == "CPU_FALLBACK") runtime.miss_policy = LLAMA_EXPERT_MISS_POLICY_CPU_FALLBACK;
            else if (miss_policy == "AUTO") runtime.miss_policy = LLAMA_EXPERT_MISS_POLICY_AUTO;
            else if (miss_policy != "PROMOTE_AND_GPU") throw probe_error("unknown online miss policy");
            if (hot_end == argv[11] || *hot_end != '\0' || cold_end == argv[12] || *cold_end != '\0' ||
                    hot_slots < 4 || hot_slots > UINT32_MAX || cold_slots > UINT32_MAX ||
                    (runtime.cache_mode == LLAMA_EXPERT_WEIGHTS_MODE_COLD_CACHE && cold_slots < 4) ||
                    (runtime.cache_mode == LLAMA_EXPERT_WEIGHTS_MODE_HOT_CACHE && cold_slots != 0) ||
                    (runtime.cache_mode == LLAMA_EXPERT_WEIGHTS_MODE_HOT_CACHE &&
                        (runtime.miss_policy != LLAMA_EXPERT_MISS_POLICY_PROMOTE_AND_GPU ||
                         runtime.load_mode == LLAMA_LOAD_MODE_DIRECT_IO))) {
                throw probe_error("invalid online runtime configuration");
            }
            runtime.hot_slots = uint32_t(hot_slots);
            runtime.cold_slots = uint32_t(cold_slots);
        }
        ggml_backend_load_all();
        llm_expert_prefetch_profile profile;
        std::string profile_error;
        const auto profile_begin = std::chrono::steady_clock::now();
        if (!llm_expert_prefetch_load_profile(profile_path, 64U*1024U*1024U, nullptr, profile, profile_error).is_ready()) {
            throw probe_error("profile load failed: " + profile_error);
        }
        const uint64_t profile_parse_ns = elapsed_ns(profile_begin);
        const bool storage_measurement = argc == 9 && std::string(argv[7]) == "--storage-map";
        const llama_load_mode measurement_load_mode = storage_measurement &&
            profile.selected_transport == "DIRECT_IO" ? LLAMA_LOAD_MODE_DIRECT_IO : LLAMA_LOAD_MODE_MMAP;
        const auto model_validation = validate_model_profile(
            model_path, profile_path, profile, measurement_load_mode,
            !validate_only, !online_disabled, runtime, !online && !online_disabled);
        if (validate_only) {
            std::cout << json({ {"schema_version", "phase10-profile-validation-v1"},
                {"project_head", identity.substr(0, separator)}, {"nested_head", identity.substr(separator + 1)},
                {"profile_sha256", profile.profile_sha256}, {"profile_parse_ns", profile_parse_ns},
                {"model_profile_load_ns", model_validation.load_ns} }).dump(2) << '\n';
            return 0;
        }
        if (online || online_disabled) {
            std::cout << online_capture(
                model_validation, profile, identity, online, runtime).dump(2) << '\n';
            return 0;
        }
        const uint64_t bytes = profile.target.expert_bytes.front().physical_bytes;
        const expected_storage_map storage_map = load_storage_map(argv[8], profile);
        const auto predictor = benchmark_predictor(profile, profile_path);
        const uint64_t predictor_p95 = predictor.upper_bound_p95_ns;
        if (profile.selected_transport != "BUFFERED" && profile.selected_transport != "DIRECT_IO") {
            throw probe_error("transport measurement requires a selected storage transport");
        }
        const char * measurement_name = profile.selected_transport == "DIRECT_IO" ? "DIRECT_IO" : "MMAP_BUFFERED";
        const auto measured = measure_runtime_paths(model_validation.diagnostics, storage_map, measurement_name);
        json output = { {"schema_version", "phase10-transport-measurements-v1"},
            {"project_head", identity.substr(0, separator)}, {"nested_head", identity.substr(separator + 1)},
            {"host", "skynet"}, {"profile_sha256", profile.profile_sha256},
            {"profile_parse_ns", profile_parse_ns}, {"model_profile_load_ns", model_validation.load_ns},
            {"lead_measurements", model_validation.lead_measurements},
            {"predictor_upper_bound", {{"basis", "maximum p95 over full-token topology-capped declared predictors"},
                {"upper_bound_p95_ns", predictor.upper_bound_p95_ns}, {"measurements", predictor.measurements}}},
            {"path_provenance", {{"storage_map_sha256", storage_map.sha256},
                {"package_sha256", storage_map.package_sha256},
                {"model_sha256", storage_map.model_sha256}, {"model_size", storage_map.model_size},
                {"source_file_count", storage_map.source_file_count},
                {"exact_runtime_provider_path", true},
                {profile.selected_transport == "DIRECT_IO" ? "direct" : "buffered", measured.provenance}}},
            {"envelopes", json::array()} };
        const uint64_t lead_p50 = model_validation.lead_measurements.at("conservative_lead_p50_ns");
        output["envelopes"].push_back(envelope(profile.selected_transport, "HOST_READY", true,
            measured.storage_service_ns, measured.storage_refill_ns, measured.scheduler_delay_ns,
            predictor_p95, bytes, 0, lead_p50, measured.provenance));
        if (profile.selected_readiness == LLAMA_EXPERT_PREFETCH_READINESS_DEVICE_READY) {
            if (profile.selected_transport == "BUFFERED") {
                output["envelopes"].push_back(envelope("HOST_TO_DEVICE", "DEVICE_READY", true,
                    measured.h2d_service_ns, measured.h2d_refill_ns, measured.scheduler_delay_ns,
                    predictor_p95, 0, bytes, lead_p50, measured.provenance));
            }
            output["envelopes"].push_back(envelope(profile.selected_transport, "DEVICE_READY", true,
                measured.device_service_ns, measured.device_refill_ns, measured.scheduler_delay_ns,
                predictor_p95, bytes, bytes, lead_p50, measured.provenance));
        }
        std::cout << output.dump(2) << '\n';
        return 0;
    } catch (const std::exception & exception) {
        std::fprintf(stderr, "phase10-prefetch-probe: %s\n", exception.what());
        return 1;
    }
}
