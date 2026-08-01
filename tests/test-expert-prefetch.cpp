#include "llama-expert-prefetch.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;

namespace {

void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

llama_expert_prefetch_config_v1 config(llama_expert_prefetch_policy policy) {
    return {
        LLAMA_EXPERT_PREFETCH_VERSION_1,
        sizeof(llama_expert_prefetch_config_v1),
        policy,
        LLAMA_EXPERT_PREFETCH_READINESS_DEVICE_READY,
        LLAMA_EXPERT_PREFETCH_SEED_MODE_OFF,
        policy == LLAMA_EXPERT_PREFETCH_POLICY_TEMPORAL_FREQUENCY ? 4U : 0U,
        2,
        1024*1024,
        2,
        1024,
        1024,
        1024,
        1024,
        2,
        2,
        8,
        4,
        {},
    };
}

json profile_json() {
    const std::string hash(64, 'a');
    const std::string package_text = "0:10:model.gguf:1024:" + hash + "\n" + hash + "\n";
    const std::string package_hash = llm_expert_prefetch_sha256(package_text.data(), package_text.size());
    json bytes = json::array();
    for (int layer : {1, 2}) {
        for (int expert = 0; expert < 4; ++expert) {
            bytes.push_back({{"layer", layer}, {"expert", expert}, {"payload_bytes", 128}, {"physical_bytes", 160}});
        }
    }
    return {
        {"schema_version", "expert-prefetch-profile-v1"},
        {"profile_id", "test-profile"},
        {"tool", {{"name", "test"}, {"version", 1}}},
        {"source", {
            {"kind", "route_trace"},
            {"artifacts", json::array({{{"name", "trace.bin"}, {"size", 64}, {"sha256", hash}}})},
            {"fold", {{"index", 0}, {"training", json::array({"structured-en-small", "technical-en-large",
                "narrative-en-large", "narrative-es-large"})}, {"validation", "code-en-small"},
                {"test", "prose-en-small"}, {"training_rows", 40}, {"validation_rows", 10}, {"test_rows", 10}}},
        }},
        {"target", {
            {"package_sha256", package_hash},
            {"files", json::array({{{"ordinal", 0}, {"name", "model.gguf"}, {"size", 1024}, {"sha256", hash}}})},
            {"layer_count", 3}, {"routed_layers", json::array({1, 2})},
            {"experts_per_layer", 4}, {"experts_per_token", 2},
            {"tensor_layout_sha256", hash}, {"expert_bytes", bytes},
        }},
        {"static_counts", json::array({
            {{"layer", 1}, {"expert", 2}, {"count", 9}}, {{"layer", 1}, {"expert", 1}, {"count", 9}},
            {{"layer", 1}, {"expert", 0}, {"count", 2}}, {{"layer", 2}, {"expert", 3}, {"count", 7}}
        })},
        {"transitions", json::array({
            {{"source_layer", 1}, {"source_expert", 0}, {"target_layer", 2}, {"target_expert", 3}, {"count", 5}},
            {{"source_layer", 1}, {"source_expert", 1}, {"target_layer", 2}, {"target_expert", 2}, {"count", 4}},
            {{"source_layer", 1}, {"source_expert", 1}, {"target_layer", 2}, {"target_expert", 3}, {"count", 1}}
        })},
        {"costs", json::array({{{"transport", "BUFFERED"}, {"readiness", "DEVICE_READY"},
            {"lead_ns", 100}, {"demand_service_ns", 80}, {"speculative_service_ns", 20},
            {"predictor_compute_ns", 10}, {"scheduler_demand_delay_ns", 5}, {"displacement_refill_ns", 5},
            {"storage_bytes", 128}, {"h2d_bytes", 128}, {"break_even_bps", 3637},
            {"utility_window_predictions", 8}, {"utility_min_observations", 4}, {"utility_min_timely_successes", 3}}})},
        {"selection", {{"matrix_version", 1}, {"tuning_digest", hash}, {"fold_index", 0},
            {"policy", "TEMPORAL_FREQUENCY"}, {"candidates_per_target", 2}, {"temporal_window_tokens", 4}, {"readiness", "DEVICE_READY"},
            {"transport", "BUFFERED"}, {"break_even_bps", 3637}}},
        {"seed", json::array({{{"layer", 1}, {"expert", 1}, {"count", 9},
            {"payload_bytes", 128}, {"physical_bytes", 160}}})},
    };
}

std::string write_profile(const std::string & contents, const char * suffix) {
    const std::string path = std::string("test-expert-prefetch-") + suffix + ".json";
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output << contents;
    require(bool(output), "failed to write test profile");
    return path;
}

void test_config() {
    llm_expert_prefetch_config_internal copied;
    require(llm_expert_prefetch_copy_config(nullptr, nullptr, LLAMA_EXPERT_WEIGHTS_MODE_DISABLED,
        0, 0, 0, 0, 0, 0, copied).is_ready(), "null configuration failed");
    require(!copied.supplied && copied.digest == 0, "null configuration did work");

    llama_expert_prefetch_config_v1 disabled = {
        LLAMA_EXPERT_PREFETCH_VERSION_1, sizeof(llama_expert_prefetch_config_v1),
        LLAMA_EXPERT_PREFETCH_POLICY_OFF, LLAMA_EXPERT_PREFETCH_READINESS_DEVICE_READY,
        LLAMA_EXPERT_PREFETCH_SEED_MODE_OFF, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, {},
    };
    require(llm_expert_prefetch_copy_config(&disabled, nullptr, LLAMA_EXPERT_WEIGHTS_MODE_COLD_CACHE,
        4, 4, 4096, 8, 4096, 4096, copied).is_ready(), "explicit disabled configuration failed");
    require(copied.supplied && copied.value.policy == LLAMA_EXPERT_PREFETCH_POLICY_OFF,
        "explicit disabled configuration was not copied");

    auto value = config(LLAMA_EXPERT_PREFETCH_POLICY_TEMPORAL_FREQUENCY);
    require(llm_expert_prefetch_copy_config(&value, "profile.json", LLAMA_EXPERT_WEIGHTS_MODE_COLD_CACHE,
        4, 4, 4096, 8, 4096, 4096, copied).is_ready(), "valid configuration failed");
    require(copied.supplied && copied.digest != 0, "configuration was not copied");
    llm_expert_prefetch_config_internal moved_profile;
    require(llm_expert_prefetch_copy_config(&value, "moved/profile.json", LLAMA_EXPERT_WEIGHTS_MODE_COLD_CACHE,
        4, 4, 4096, 8, 4096, 4096, moved_profile).is_ready() && moved_profile.digest == copied.digest,
        "profile location changed configuration identity");
    value.temporal_window_tokens = 3;
    require(!llm_expert_prefetch_copy_config(&value, "profile.json", LLAMA_EXPERT_WEIGHTS_MODE_COLD_CACHE,
        4, 4, 4096, 8, 4096, 4096, copied).is_ready(), "non-power-of-two window accepted");
    value = config(LLAMA_EXPERT_PREFETCH_POLICY_PREVIOUS_TOKEN);
    value.max_speculative_hot_slots = 5;
    require(!llm_expert_prefetch_copy_config(&value, "profile.json", LLAMA_EXPERT_WEIGHTS_MODE_COLD_CACHE,
        4, 4, 4096, 8, 4096, 4096, copied).is_ready(), "oversized hot budget accepted");
    value = config(LLAMA_EXPERT_PREFETCH_POLICY_PREVIOUS_TOKEN);
    value.max_speculative_storage_bytes_per_token = value.max_speculative_storage_bytes_in_flight + 1;
    require(!llm_expert_prefetch_copy_config(&value, "profile.json", LLAMA_EXPERT_WEIGHTS_MODE_COLD_CACHE,
        4, 4, 4096, 8, 4096, 4096, copied).is_ready(), "oversized per-token budget accepted");
}

llm_expert_prefetch_profile load_profile(const std::string & path) {
    llm_expert_prefetch_profile profile;
    std::string error;
    require(llm_expert_prefetch_load_profile(path, 1024*1024, nullptr, profile, error).is_ready(), error.c_str());
    return profile;
}

void test_profile() {
    const std::string contents = profile_json().dump(2) + "\n";
    const std::string path = write_profile(contents, "valid");
    auto profile = load_profile(path);
    require(profile.static_counts[0].expert == 1 && profile.static_counts[1].expert == 2,
        "static tie did not prefer lower expert");
    require(profile.profile_sha256 == llm_expert_prefetch_sha256(contents.data(), contents.size()), "profile hash mismatch");
    auto expected = profile.target;
    require(load_profile(path).target.package_sha256 == expected.package_sha256, "profile reload changed identity");
    expected.files[0].size++;
    llm_expert_prefetch_profile rejected;
    std::string error;
    require(llm_expert_prefetch_load_profile(path, 1024*1024, &expected, rejected, error).error ==
        llm_expert_prefetch_error::profile_identity_mismatch, "wrong package accepted");
    require(llm_expert_prefetch_load_profile(path, 16, nullptr, rejected, error).error ==
        llm_expert_prefetch_error::profile_too_large, "profile ceiling not enforced");
    require(llm_expert_prefetch_load_profile(path, contents.size(), nullptr, rejected, error).error ==
        llm_expert_prefetch_error::profile_too_large, "parse allocation ceiling not enforced");
    auto wrong_cost = profile_json();
    wrong_cost["selection"]["break_even_bps"] = 3636;
    const std::string wrong_cost_path = write_profile(wrong_cost.dump(2) + "\n", "wrong-cost");
    require(!llm_expert_prefetch_load_profile(wrong_cost_path, 1024*1024, nullptr, rejected, error).is_ready(),
        "mismatched selected cost accepted");
    auto blocking = profile_json();
    blocking["selection"]["policy"] = "BLOCKING_HOT";
    blocking["selection"]["candidates_per_target"] = 0;
    blocking["selection"]["temporal_window_tokens"] = 0;
    const std::string blocking_path = write_profile(blocking.dump(2) + "\n", "blocking");
    require(llm_expert_prefetch_load_profile(blocking_path, 1024*1024, nullptr, rejected, error).is_ready(),
        "seed-only profile was rejected");
    blocking["selection"]["policy"] = "STATIC_LAYER";
    const std::string zero_predictor_path = write_profile(blocking.dump(2) + "\n", "zero-predictor");
    require(!llm_expert_prefetch_load_profile(zero_predictor_path, 1024*1024, nullptr, rejected, error).is_ready(),
        "active predictor accepted zero candidates");
    std::string duplicate = contents;
    duplicate.replace(duplicate.find("\"profile_id\""), std::string("\"profile_id\"").size(),
        "\"profile_id\": \"duplicate\", \"profile_id\"");
    const std::string duplicate_path = write_profile(duplicate, "duplicate");
    require(!llm_expert_prefetch_load_profile(duplicate_path, 1024*1024, nullptr, rejected, error).is_ready(),
        "duplicate key accepted");
    auto swapped_fold = profile_json();
    std::swap(swapped_fold["source"]["fold"]["validation"], swapped_fold["source"]["fold"]["test"]);
    const std::string swapped_fold_path = write_profile(swapped_fold.dump(2) + "\n", "swapped-fold");
    require(!llm_expert_prefetch_load_profile(swapped_fold_path, 1024*1024, nullptr, rejected, error).is_ready(),
        "swapped validation and test prompts accepted");
    auto arbitrary_fold = profile_json();
    arbitrary_fold["source"]["fold"]["training"][0] = "arbitrary-prompt";
    const std::string arbitrary_fold_path = write_profile(arbitrary_fold.dump(2) + "\n", "arbitrary-fold");
    require(!llm_expert_prefetch_load_profile(arbitrary_fold_path, 1024*1024, nullptr, rejected, error).is_ready(),
        "arbitrary fold prompt accepted");
    auto reordered_fold = profile_json();
    std::swap(reordered_fold["source"]["fold"]["training"][0], reordered_fold["source"]["fold"]["training"][1]);
    const std::string reordered_fold_path = write_profile(reordered_fold.dump(2) + "\n", "reordered-fold");
    require(!llm_expert_prefetch_load_profile(reordered_fold_path, 1024*1024, nullptr, rejected, error).is_ready(),
        "reordered fold training prompts accepted");
    std::remove(path.c_str());
    std::remove(wrong_cost_path.c_str());
    std::remove(blocking_path.c_str());
    std::remove(zero_predictor_path.c_str());
    std::remove(duplicate_path.c_str());
    std::remove(swapped_fold_path.c_str());
    std::remove(arbitrary_fold_path.c_str());
    std::remove(reordered_fold_path.c_str());
}

std::vector<llm_expert_prefetch_candidate> predict(
        const llm_expert_prefetch_profile & profile,
        llama_expert_prefetch_policy policy) {
    auto value = config(policy);
    llm_expert_prefetch_config_internal copied;
    require(llm_expert_prefetch_copy_config(&value, "profile.json", LLAMA_EXPERT_WEIGHTS_MODE_COLD_CACHE,
        4, 4, 4096, 8, 4096, 4096, copied).is_ready(), "predictor config failed");
    llm_expert_prefetch_predictor predictor;
    require(predictor.initialize(profile, copied).is_ready(), "predictor initialization failed");
    require(predictor.request_begin(7).is_ready(), "request begin failed");
    require(predictor.commit_token(0, {{0, 1}, {2, 3}}).is_ready(), "token commit failed");
    require(predictor.commit_token(1, {{1, 2}, {0, 3}}).is_ready(), "second token commit failed");
    std::vector<llm_expert_prefetch_candidate> result;
    require(predictor.predict_token_end(1, 1, result).is_ready(), "token prediction failed");
    require(!predictor.predict_token_end(0, 1, result).is_ready(), "future or out-of-order prediction accepted");
    require(predictor.predict_token_end(1, 1, result).is_ready(), "repeat deterministic prediction failed");
    return result;
}

void test_predictors() {
    const std::string path = write_profile(profile_json().dump(2) + "\n", "predictor");
    const auto profile = load_profile(path);
    const auto static_result = predict(profile, LLAMA_EXPERT_PREFETCH_POLICY_STATIC_LAYER);
    require(static_result.size() == 2 && static_result[0].key.expert == 1 && static_result[1].key.expert == 2,
        "static predictor ordering mismatch");
    const auto previous = predict(profile, LLAMA_EXPERT_PREFETCH_POLICY_PREVIOUS_TOKEN);
    require(previous.size() == 2 && previous[0].key.expert == 1 && previous[1].key.expert == 2,
        "previous predictor mismatch");
    const auto temporal = predict(profile, LLAMA_EXPERT_PREFETCH_POLICY_TEMPORAL_FREQUENCY);
    require(temporal.size() == 2 && temporal[0].key.expert == 1 && temporal[1].key.expert == 2,
        "temporal predictor mismatch");
    const auto random_a = predict(profile, LLAMA_EXPERT_PREFETCH_POLICY_RANDOM_BASELINE);
    const auto random_b = predict(profile, LLAMA_EXPERT_PREFETCH_POLICY_RANDOM_BASELINE);
    require(random_a.size() == random_b.size(), "random predictor size changed");
    for (size_t index = 0; index < random_a.size(); ++index) {
        require(random_a[index].key.expert == random_b[index].key.expert, "random predictor was nondeterministic");
    }

    auto value = config(LLAMA_EXPERT_PREFETCH_POLICY_CROSS_LAYER_TRANSITION);
    llm_expert_prefetch_config_internal copied;
    require(llm_expert_prefetch_copy_config(&value, "profile.json", LLAMA_EXPERT_WEIGHTS_MODE_COLD_CACHE,
        4, 4, 4096, 8, 4096, 4096, copied).is_ready(), "cross config failed");
    llm_expert_prefetch_predictor cross;
    require(cross.initialize(profile, copied).is_ready() && cross.request_begin(1).is_ready(), "cross initialization failed");
    const int32_t sources[] = {0, 1};
    std::vector<llm_expert_prefetch_candidate> candidates;
    require(cross.predict_cross_layer(0, 1, sources, 2, 2, candidates).is_ready(), "cross prediction failed");
    require(candidates.size() == 2 && candidates[0].key.expert == 3 && candidates[0].score == 6 &&
        candidates[1].key.expert == 2 && candidates[1].score == 4, "cross aggregation mismatch");
    std::remove(path.c_str());
}

void test_break_even() {
    uint64_t hidden = 0, waste = 0;
    uint32_t bps = 0;
    require(llm_expert_prefetch_break_even(100, 80, 10, 20, 5, 5, hidden, waste, bps).is_ready(),
        "valid break-even failed");
    require(hidden == 70 && waste == 40 && bps == 3637, "break-even math mismatch");
    require(!llm_expert_prefetch_break_even(10, 20, 10, 1, 1, 1, hidden, waste, bps).is_ready(),
        "nonpositive hidden benefit accepted");
}

} // namespace

int main() {
    try {
        test_config();
        test_profile();
        test_predictors();
        test_break_even();
    } catch (const std::exception & exception) {
        std::fprintf(stderr, "test-expert-prefetch: %s\n", exception.what());
        return 1;
    }
    return 0;
}
