#include "llama-expert-prefetch.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;

namespace {

struct replay_error : std::runtime_error {
    using std::runtime_error::runtime_error;
};

uint64_t unsigned_value(const json & value, const char * name) {
    if (!value.is_number_unsigned()) throw replay_error(std::string(name) + " must be an unsigned integer");
    return value.get<uint64_t>();
}

uint32_t uint32_value(const json & value, const char * name) {
    const uint64_t result = unsigned_value(value, name);
    if (result > UINT32_MAX) throw replay_error(std::string(name) + " exceeds uint32");
    return uint32_t(result);
}

std::string string_value(const json & value, const char * name) {
    if (!value.is_string()) throw replay_error(std::string(name) + " must be a string");
    const std::string result = value.get<std::string>();
    if (result.empty()) throw replay_error(std::string(name) + " must not be empty");
    return result;
}

void require(bool condition, const char * message) {
    if (!condition) throw replay_error(message);
}

void require_fields(const json & value, std::initializer_list<const char *> expected, const char * name) {
    require(value.is_object(), "expected JSON object");
    std::set<std::string> actual;
    for (const auto & item : value.items()) actual.insert(item.key());
    std::set<std::string> wanted;
    for (const char * field : expected) wanted.insert(field);
    if (actual != wanted) throw replay_error(std::string(name) + " fields do not match v1");
}

llama_expert_prefetch_policy policy_value(const std::string & value) {
    static const std::map<std::string, llama_expert_prefetch_policy> policies = {
        {"OFF", LLAMA_EXPERT_PREFETCH_POLICY_OFF},
        {"STATIC_LAYER", LLAMA_EXPERT_PREFETCH_POLICY_STATIC_LAYER},
        {"PREVIOUS_TOKEN", LLAMA_EXPERT_PREFETCH_POLICY_PREVIOUS_TOKEN},
        {"TEMPORAL_FREQUENCY", LLAMA_EXPERT_PREFETCH_POLICY_TEMPORAL_FREQUENCY},
        {"CROSS_LAYER_TRANSITION", LLAMA_EXPERT_PREFETCH_POLICY_CROSS_LAYER_TRANSITION},
        {"RANDOM_BASELINE", LLAMA_EXPERT_PREFETCH_POLICY_RANDOM_BASELINE},
    };
    const auto found = policies.find(value);
    if (found == policies.end()) throw replay_error("unknown policy");
    return found->second;
}

llama_expert_prefetch_readiness readiness_value(const std::string & value) {
    if (value == "HOST_READY") return LLAMA_EXPERT_PREFETCH_READINESS_HOST_READY;
    if (value == "DEVICE_READY") return LLAMA_EXPERT_PREFETCH_READINESS_DEVICE_READY;
    throw replay_error("unknown readiness");
}

json candidate_json(
        uint64_t trigger_token,
        const char * trigger,
        int32_t source_layer,
        const llm_expert_prefetch_candidate & candidate) {
    return {
        {"trigger_token", trigger_token}, {"trigger", trigger}, {"source_layer", source_layer},
        {"target_layer", candidate.key.layer}, {"expert", candidate.key.expert},
        {"rank", candidate.rank}, {"score", candidate.score},
    };
}

json replay(const json & input) {
    require_fields(input, {"schema_version", "profile_path", "policy", "readiness", "temporal_window_tokens",
        "candidates_per_target", "request_ordinal", "events", "completion_order"}, "replay");
    require(input.at("schema_version") == "phase10-prefetch-replay-v1", "unsupported replay schema");
    require(input.at("events").is_array(), "events must be an array");
    require(input.at("completion_order").is_array(), "completion_order must be an array");

    llm_expert_prefetch_profile profile;
    std::string profile_error;
    const std::string profile_path = string_value(input.at("profile_path"), "profile_path");
    const auto loaded = llm_expert_prefetch_load_profile(profile_path, 64U*1024U*1024U, nullptr, profile, profile_error);
    if (!loaded.is_ready()) throw replay_error("profile load failed: " + profile_error);
    const auto policy = policy_value(string_value(input.at("policy"), "policy"));
    const auto readiness = readiness_value(string_value(input.at("readiness"), "readiness"));
    const uint32_t temporal_window = uint32_value(input.at("temporal_window_tokens"), "temporal_window_tokens");
    const uint32_t candidate_count = uint32_value(input.at("candidates_per_target"), "candidates_per_target");
    const uint64_t request_ordinal = unsigned_value(input.at("request_ordinal"), "request_ordinal");
    require(candidate_count != 0 && request_ordinal != 0, "zero replay count or request ordinal");
    std::set<uint64_t> completion_ids;
    for (const auto & completion : input.at("completion_order")) {
        require(completion_ids.insert(unsigned_value(completion, "completion_order")).second,
            "duplicate completion ordinal");
    }

    json candidates = json::array();
    uint64_t state_digest = UINT64_C(1469598103934665603);
    if (policy != LLAMA_EXPERT_PREFETCH_POLICY_OFF) {
        llama_expert_prefetch_config_v1 public_config = {
            LLAMA_EXPERT_PREFETCH_VERSION_1,
            sizeof(llama_expert_prefetch_config_v1),
            policy,
            readiness,
            LLAMA_EXPERT_PREFETCH_SEED_MODE_OFF,
            temporal_window,
            candidate_count,
            64U*1024U*1024U,
            32,
            UINT64_C(1) << 30,
            UINT64_C(1) << 30,
            UINT64_C(1) << 30,
            UINT64_C(1) << 30,
            32,
            32,
            64,
            32,
            {},
        };
        llm_expert_prefetch_config_internal config;
        require(llm_expert_prefetch_copy_config(&public_config, profile_path.c_str(), LLAMA_EXPERT_WEIGHTS_MODE_COLD_CACHE,
            profile.target.experts_per_layer, 32, UINT64_C(1) << 30, 64, UINT64_C(1) << 30, UINT64_C(1) << 30,
            config).is_ready(), "invalid replay config");
        llm_expert_prefetch_predictor predictor;
        require(predictor.initialize(profile, config).is_ready(), "predictor initialize failed");
        require(predictor.request_begin(request_ordinal).is_ready(), "request begin failed");
        for (size_t token_index = 0; token_index < input.at("events").size(); ++token_index) {
            const auto & event = input.at("events")[token_index];
            require_fields(event, {"token", "layers"}, "event");
            require(unsigned_value(event.at("token"), "event.token") == token_index && event.at("layers").is_array() &&
                event.at("layers").size() == profile.target.routed_layers.size(), "noncanonical event");
            std::vector<std::vector<int32_t>> token;
            token.reserve(event.at("layers").size());
            for (size_t layer_index = 0; layer_index < event.at("layers").size(); ++layer_index) {
                const auto & layer = event.at("layers")[layer_index];
                require_fields(layer, {"layer", "experts"}, "layer event");
                require(uint32_value(layer.at("layer"), "layer") == uint32_t(profile.target.routed_layers[layer_index]) &&
                    layer.at("experts").is_array(), "noncanonical layer event");
                std::vector<int32_t> selected;
                for (const auto & expert : layer.at("experts")) {
                    const uint32_t value = uint32_value(expert, "expert");
                    require(value <= INT32_MAX, "expert exceeds int32");
                    selected.push_back(int32_t(value));
                }
                require(!selected.empty() && selected.size() <= profile.target.experts_per_token,
                    "invalid selected expert width");
                token.push_back(std::move(selected));
                if (policy == LLAMA_EXPERT_PREFETCH_POLICY_CROSS_LAYER_TRANSITION && layer_index + 1 < event.at("layers").size()) {
                    std::vector<llm_expert_prefetch_candidate> predicted;
                    require(predictor.predict_cross_layer(token_index, profile.target.routed_layers[layer_index],
                        token.back().data(), token.back().size(), profile.target.routed_layers[layer_index + 1], predicted).is_ready(),
                        "cross prediction failed");
                    for (const auto & candidate : predicted) candidates.push_back(candidate_json(
                        token_index, "ROUTER_RESULT", profile.target.routed_layers[layer_index], candidate));
                }
            }
            require(predictor.commit_token(token_index, token).is_ready(), "token commit failed");
            if (policy != LLAMA_EXPERT_PREFETCH_POLICY_CROSS_LAYER_TRANSITION) {
                for (int32_t layer : profile.target.routed_layers) {
                    std::vector<llm_expert_prefetch_candidate> predicted;
                    require(predictor.predict_token_end(token_index, layer, predicted).is_ready(), "token-end prediction failed");
                    for (const auto & candidate : predicted) candidates.push_back(candidate_json(token_index, "TOKEN_END", -1, candidate));
                }
            }
        }
        state_digest = predictor.state().digest;
    }
    const std::string event_dump = input.at("events").dump();
    return {
        {"schema_version", "phase10-prefetch-replay-output-v1"},
        {"profile_sha256", profile.profile_sha256},
        {"policy", input.at("policy")},
        {"candidate_stream", candidates},
        {"state_digest", state_digest},
        {"phase9_passthrough_sha256", llm_expert_prefetch_sha256(event_dump.data(), event_dump.size())},
    };
}

int self_test() {
    uint64_t hidden = 0, waste = 0;
    uint32_t bps = 0;
    require(llm_expert_prefetch_break_even(100, 80, 10, 20, 5, 5, hidden, waste, bps).is_ready(), "break-even self-test failed");
    require(hidden == 70 && waste == 40 && bps == 3637, "break-even self-test mismatch");
    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--self-test") return self_test();
        if (argc != 2) throw replay_error("usage: phase10-prefetch-replay INPUT.json");
        std::ifstream input(argv[1]);
        if (!input) throw replay_error("unable to open replay input");
        std::vector<std::set<std::string>> object_keys;
        bool duplicate_key = false;
        auto callback = [&](int, json::parse_event_t event, json & parsed) {
            if (event == json::parse_event_t::object_start) object_keys.emplace_back();
            if (event == json::parse_event_t::key) {
                if (object_keys.empty() || !object_keys.back().insert(parsed.get<std::string>()).second) duplicate_key = true;
            }
            if (event == json::parse_event_t::object_end && !object_keys.empty()) object_keys.pop_back();
            return !duplicate_key;
        };
        const json document = json::parse(input, callback, true, false);
        if (duplicate_key || document.is_discarded()) throw replay_error("duplicate JSON key");
        std::cout << replay(document).dump(2) << '\n';
        return 0;
    } catch (const std::exception & exception) {
        std::fprintf(stderr, "phase10-prefetch-replay: %s\n", exception.what());
        return 1;
    }
}
