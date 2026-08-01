#include "llama-expert-prefetch.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
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

int32_t int32_value(const json & value, const char * name) {
    if (!value.is_number_integer()) throw replay_error(std::string(name) + " must be an integer");
    const int64_t result = value.get<int64_t>();
    if (result < INT32_MIN || result > INT32_MAX) throw replay_error(std::string(name) + " exceeds int32");
    return int32_t(result);
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

constexpr uint64_t fnv_offset = UINT64_C(1469598103934665603);
constexpr uint64_t fnv_prime = UINT64_C(1099511628211);

void digest_append(uint64_t & digest, uint64_t value) {
    for (uint32_t byte = 0; byte < 8; ++byte) {
        digest ^= uint8_t(value >> (byte*8));
        digest *= fnv_prime;
    }
}

struct replay_limits {
    uint64_t cold_capacity_bytes = 0;
    uint32_t hot_capacity_slots = 0;
    uint32_t max_speculative_flights = 0;
    uint64_t max_speculative_storage_bytes_in_flight = 0;
    uint64_t max_speculative_h2d_bytes_in_flight = 0;
    uint64_t max_speculative_storage_bytes_per_token = 0;
    uint64_t max_speculative_h2d_bytes_per_token = 0;
    uint32_t max_speculative_cold_slots = 0;
    uint32_t max_speculative_hot_slots = 0;
};

replay_limits limits_value(const json & value, bool active, const std::string & cache_mode) {
    require_fields(value, { "cold_capacity_bytes", "hot_capacity_slots", "max_speculative_flights",
        "max_speculative_storage_bytes_in_flight", "max_speculative_h2d_bytes_in_flight",
        "max_speculative_storage_bytes_per_token", "max_speculative_h2d_bytes_per_token",
        "max_speculative_cold_slots", "max_speculative_hot_slots" }, "limits");
    replay_limits result;
    result.cold_capacity_bytes = unsigned_value(value.at("cold_capacity_bytes"), "cold_capacity_bytes");
    result.hot_capacity_slots = uint32_value(value.at("hot_capacity_slots"), "hot_capacity_slots");
    result.max_speculative_flights = uint32_value(value.at("max_speculative_flights"), "max_speculative_flights");
    result.max_speculative_storage_bytes_in_flight = unsigned_value(
        value.at("max_speculative_storage_bytes_in_flight"), "max_speculative_storage_bytes_in_flight");
    result.max_speculative_h2d_bytes_in_flight = unsigned_value(
        value.at("max_speculative_h2d_bytes_in_flight"), "max_speculative_h2d_bytes_in_flight");
    result.max_speculative_storage_bytes_per_token = unsigned_value(
        value.at("max_speculative_storage_bytes_per_token"), "max_speculative_storage_bytes_per_token");
    result.max_speculative_h2d_bytes_per_token = unsigned_value(
        value.at("max_speculative_h2d_bytes_per_token"), "max_speculative_h2d_bytes_per_token");
    result.max_speculative_cold_slots = uint32_value(value.at("max_speculative_cold_slots"), "max_speculative_cold_slots");
    result.max_speculative_hot_slots = uint32_value(value.at("max_speculative_hot_slots"), "max_speculative_hot_slots");
    require(result.hot_capacity_slots != 0 &&
        (cache_mode == "COLD_CACHE") == (result.cold_capacity_bytes != 0),
        "cache replay capacity disagrees with cache mode");
    const bool speculative_nonzero = result.max_speculative_flights != 0 &&
        result.max_speculative_storage_bytes_in_flight != 0 && result.max_speculative_h2d_bytes_in_flight != 0 &&
        result.max_speculative_storage_bytes_per_token != 0 && result.max_speculative_h2d_bytes_per_token != 0 &&
        result.max_speculative_cold_slots != 0 && result.max_speculative_hot_slots != 0;
    const bool speculative_zero = result.max_speculative_flights == 0 &&
        result.max_speculative_storage_bytes_in_flight == 0 && result.max_speculative_h2d_bytes_in_flight == 0 &&
        result.max_speculative_storage_bytes_per_token == 0 && result.max_speculative_h2d_bytes_per_token == 0 &&
        result.max_speculative_cold_slots == 0 && result.max_speculative_hot_slots == 0;
    require(active ? speculative_nonzero : speculative_zero,
        active ? "active replay speculative limits must be nonzero" : "disabled replay speculative limits must be zero");
    return result;
}

struct replay_entry {
    int32_t layer = -1;
    int32_t expert = -1;
    uint64_t payload_bytes = 0;
    uint64_t physical_bytes = 0;
    int32_t cold_slot = -1;
    int32_t hot_slot = -1;
    std::string origin;
    uint64_t deadline_token = 0;
    int32_t deadline_layer = -1;
    uint64_t score = 0;
    int64_t flight = -1;
    uint64_t cold_last_touch = 0;
    uint64_t hot_last_touch = 0;
    std::string phase;
    bool hot_speculative = false;
};

struct replay_counters {
    uint64_t predictions = 0;
    uint64_t accepted = 0;
    uint64_t timely_useful = 0;
    uint64_t late_joined = 0;
    uint64_t wasted_unused = 0;
    uint64_t cancelled_before_io = 0;
    uint64_t cancelled_drained = 0;
    uint64_t rejected = 0;
    uint64_t demand_keys = 0;
    uint64_t demand_hits = 0;
    uint64_t demand_loads = 0;
    uint64_t speculative_replacements = 0;
    uint64_t prevented_demand_evictions = 0;
    uint64_t storage_bytes = 0;
    uint64_t h2d_bytes = 0;
    uint64_t wasted_storage_bytes = 0;
    uint64_t wasted_h2d_bytes = 0;

    json output() const {
        return { {"predictions", predictions}, {"accepted", accepted}, {"timely_useful", timely_useful},
            {"late_joined", late_joined}, {"wasted_unused", wasted_unused},
            {"cancelled_before_io", cancelled_before_io}, {"cancelled_drained", cancelled_drained},
            {"rejected", rejected}, {"demand_keys", demand_keys}, {"demand_hits", demand_hits},
            {"demand_loads", demand_loads}, {"speculative_replacements", speculative_replacements},
            {"prevented_demand_evictions", prevented_demand_evictions}, {"storage_bytes", storage_bytes},
            {"h2d_bytes", h2d_bytes}, {"wasted_storage_bytes", wasted_storage_bytes},
            {"wasted_h2d_bytes", wasted_h2d_bytes} };
    }
};

class hierarchy_replay {
public:
    hierarchy_replay(
            const llm_expert_prefetch_profile & requested_profile,
            const json & requested_events,
            const json & requested_initial_resident,
            json & requested_candidates,
            std::set<uint64_t> requested_ready_before_deadline,
            std::string requested_policy,
            std::string requested_cache_mode,
            std::string requested_miss_policy,
            std::string requested_transport,
            std::string requested_readiness,
            replay_limits requested_limits,
            std::string requested_seed_mode,
            std::string requested_demand_mode) :
        profile(requested_profile), events(requested_events), initial_resident(requested_initial_resident),
        candidates(requested_candidates), ready_before_deadline(std::move(requested_ready_before_deadline)),
        policy(std::move(requested_policy)), cache_mode(std::move(requested_cache_mode)),
        miss_policy(std::move(requested_miss_policy)),
        transport(std::move(requested_transport)), readiness(std::move(requested_readiness)), limits(requested_limits),
        seed_mode(std::move(requested_seed_mode)), demand_mode(std::move(requested_demand_mode)) {
        for (size_t index = 0; index < profile.target.routed_layers.size(); ++index) {
            layer_indices[profile.target.routed_layers[index]] = index;
        }
        const auto selected_cost = std::find_if(profile.costs.begin(), profile.costs.end(), [&](const auto & item) {
            return item.transport == transport && (item.readiness == LLAMA_EXPERT_PREFETCH_READINESS_DEVICE_READY) ==
                (readiness == "DEVICE_READY");
        });
        require(selected_cost != profile.costs.end(), "replay transport/readiness cost is unavailable");
        cost = *selected_cost;
        uint64_t minimum_physical = UINT64_MAX;
        for (const auto & item : profile.target.expert_bytes) {
            sizes[{item.layer, item.expert}] = {item.payload_bytes, item.physical_bytes};
            minimum_physical = std::min(minimum_physical, item.physical_bytes);
        }
        require(minimum_physical != 0 && minimum_physical != UINT64_MAX,
            "profile expert byte map is empty");
        const uint64_t slots = limits.cold_capacity_bytes/minimum_physical;
        require((cache_mode != "COLD_CACHE" || slots != 0) && slots <= INT32_MAX &&
            limits.hot_capacity_slots <= INT32_MAX,
            "cache replay slot capacity is invalid");
        cold_slot_capacity = uint32_t(slots);
        for (size_t index = 0; index < candidates.size(); ++index) {
            candidates[index]["flight_ordinal"] = index;
            const auto & candidate = candidates[index];
            const int32_t batch_layer = candidate.at("trigger") == "ROUTER_RESULT" ?
                candidate.at("source_layer").get<int32_t>() : -1;
            batches[{candidate.at("trigger_token").get<uint64_t>(), candidate.at("trigger").get<std::string>(),
                batch_layer}].push_back(candidate);
        }
        counters.predictions = candidates.size();
    }

    json run() {
        initialize_resident();
        seed();
        for (size_t token = 0; token < events.size(); ++token) {
            for (const auto & layer : events[token].at("layers")) {
                const int32_t layer_id = layer.at("layer");
                const auto selected = layer.at("experts").get<std::vector<int32_t>>();
                issue_demands(token, layer_id, selected);
                discard_unused_deadlines(token, layer_id, selected);
                submit_batch(batch(token, "ROUTER_RESULT", layer_id));
                consume_demands(token, layer_id, selected);
                resolve_deadline(token, layer_id);
            }
            submit_batch(batch(token, "TOKEN_END", -1));
        }
        std::vector<key_type> remaining;
        for (const auto & item : entries) if (is_speculative(item.second)) remaining.push_back(item.first);
        for (const auto & key : remaining) discard(key, "request_end");
        json resident = json::array();
        for (const auto & item : entries) {
            resident.push_back({ {"layer", item.first.first}, {"expert", item.first.second},
                {"cold_slot", item.second.cold_slot}, {"hot_slot", item.second.hot_slot},
                {"origin", item.second.origin} });
        }
        return { {"action_stream", actions}, {"outcome_stream", outcomes}, {"state_digest", digest},
            {"summary", counters.output()}, {"resident", resident} };
    }

private:
    using key_type = std::pair<int32_t, int32_t>;
    using batch_type = std::tuple<uint64_t, std::string, int32_t>;
    const llm_expert_prefetch_profile & profile;
    const json & events;
    const json & initial_resident;
    json & candidates;
    std::set<uint64_t> ready_before_deadline;
    std::string policy;
    std::string cache_mode;
    std::string miss_policy;
    std::string transport;
    std::string readiness;
    llm_expert_prefetch_cost cost;
    replay_limits limits;
    std::string seed_mode;
    std::string demand_mode;
    uint32_t cold_slot_capacity = 0;
    uint64_t clock = 0;
    uint64_t digest = fnv_offset;
    json actions = json::array();
    json outcomes = json::array();
    replay_counters counters;
    std::map<key_type, std::pair<uint64_t, uint64_t>> sizes;
    std::map<int32_t, size_t> layer_indices;
    std::map<batch_type, std::vector<json>> batches;
    std::map<key_type, replay_entry> entries;
    std::map<uint64_t, uint64_t> token_storage;
    std::map<uint64_t, uint64_t> token_h2d;

    const std::vector<json> & batch(uint64_t token, const std::string & trigger, int32_t layer) const {
        static const std::vector<json> empty;
        const auto found = batches.find({token, trigger, layer});
        return found == batches.end() ? empty : found->second;
    }

    void emit(json & destination, json record) {
        static const std::map<std::string, uint64_t> codes = {
            {"DEMAND_ISSUE", 1}, {"DEMAND_HIT", 2}, {"DEMAND_LOAD", 3}, {"ENQUEUE", 4}, {"READY", 5},
            {"REJECTED", 6}, {"EVICT", 7}, {"TIMELY_USEFUL", 8}, {"LATE_JOINED", 9},
            {"WASTED_UNUSED", 10}, {"SEED_LOAD", 11}, {"SEED_TOUCH", 12},
            {"DEMAND_PROMOTE", 13}, {"CANCELLED_BEFORE_IO", 14}, {"CANCELLED_DRAINED", 15},
            {"DEMAND_COLD_EVICT", 16}, {"DEMAND_HOT_EVICT", 17},
        };
        record["sequence"] = actions.size() + outcomes.size();
        const auto code = codes.find(record.at("type").get<std::string>());
        require(code != codes.end(), "unknown replay action type");
        digest_append(digest, code->second);
        for (const char * name : {"flight_ordinal", "token", "layer", "expert", "cold_slot", "hot_slot"}) {
            const int64_t value = record.value(name, int64_t(-1));
            digest_append(digest, value < 0 ? UINT64_MAX : uint64_t(value));
        }
        destination.push_back(std::move(record));
    }

    uint64_t occupied_bytes() const {
        uint64_t result = 0;
        for (const auto & item : entries) {
            if (item.second.cold_slot < 0) continue;
            require(item.second.physical_bytes <= UINT64_MAX - result, "cold occupancy overflow");
            result += item.second.physical_bytes;
        }
        return result;
    }

    int32_t free_slot(bool cold) const {
        const uint32_t capacity = cold ? cold_slot_capacity : limits.hot_capacity_slots;
        std::set<int32_t> used;
        for (const auto & item : entries) {
            const int32_t slot = cold ? item.second.cold_slot : item.second.hot_slot;
            if (slot >= 0) used.insert(slot);
        }
        for (uint32_t slot = 0; slot < capacity; ++slot) {
            if (!used.count(int32_t(slot))) return int32_t(slot);
        }
        return -1;
    }

    static bool is_speculative(const replay_entry & entry) {
        return entry.origin == "SPECULATIVE" || entry.hot_speculative;
    }

    void discard(key_type key, const std::string & reason) {
        const auto found = entries.find(key);
        require(found != entries.end(), "discard target is not resident");
        replay_entry & mutable_entry = found->second;
        const replay_entry entry = mutable_entry;
        const bool speculative_hot_only = entry.hot_speculative && entry.origin != "SPECULATIVE";
        if (is_speculative(entry)) {
            std::string outcome;
            if (entry.phase == "QUEUED") {
                outcome = "CANCELLED_BEFORE_IO";
                counters.cancelled_before_io++;
            } else if (entry.phase == "SUBMITTED") {
                outcome = "CANCELLED_DRAINED";
                counters.cancelled_drained++;
                counters.wasted_storage_bytes += speculative_hot_only ? 0 : entry.physical_bytes;
                if (readiness == "DEVICE_READY") counters.wasted_h2d_bytes += entry.payload_bytes;
            } else {
                outcome = "WASTED_UNUSED";
                counters.wasted_unused++;
                counters.wasted_storage_bytes += speculative_hot_only ? 0 : entry.physical_bytes;
                if (readiness == "DEVICE_READY") counters.wasted_h2d_bytes += entry.payload_bytes;
            }
            emit(outcomes, { {"type", outcome}, {"flight_ordinal", entry.flight},
                {"token", entry.deadline_token}, {"layer", entry.layer}, {"expert", entry.expert},
                {"cold_slot", entry.cold_slot}, {"hot_slot", entry.hot_slot}, {"reason", reason},
                {"completion_phase", entry.phase} });
        }
        if (speculative_hot_only) {
            mutable_entry.hot_slot = -1;
            mutable_entry.hot_last_touch = 0;
            mutable_entry.hot_speculative = false;
            mutable_entry.deadline_token = 0;
            mutable_entry.score = 0;
            mutable_entry.flight = -1;
            mutable_entry.phase = "READY";
        } else {
            entries.erase(found);
        }
    }

    bool demand_victim(bool cold, key_type & result) const {
        bool found = false;
        std::tuple<uint64_t, int32_t, int32_t> best;
        for (const auto & item : entries) {
            const int32_t slot = cold ? item.second.cold_slot : item.second.hot_slot;
            if (slot < 0 || (cold && item.second.hot_slot >= 0)) continue;
            const auto order = std::make_tuple(cold ? item.second.cold_last_touch : item.second.hot_last_touch,
                item.second.layer, item.second.expert);
            if (!found || order < best) {
                found = true;
                best = order;
                result = item.first;
            }
        }
        return found;
    }

    void emit_demand_eviction(const char * type, uint64_t token, key_type key) {
        const auto & entry = entries.at(key);
        emit(actions, { {"type", type}, {"flight_ordinal", entry.flight}, {"token", token},
            {"layer", key.first}, {"expert", key.second}, {"cold_slot", entry.cold_slot},
            {"hot_slot", entry.hot_slot}, {"reason", "phase9_lru_capacity"} });
    }

    std::pair<int32_t, int32_t> ensure_demand_capacity(
            uint64_t token, uint64_t physical, bool needs_hot) {
        if (cache_mode == "COLD_CACHE") {
            while (physical > limits.cold_capacity_bytes - occupied_bytes() || free_slot(true) < 0) {
                key_type victim;
                require(demand_victim(true, victim), "mandatory cold demand cannot be admitted");
                emit_demand_eviction("DEMAND_COLD_EVICT", token, victim);
                discard(victim, "demand_cold_eviction");
            }
        }
        const int32_t cold_slot = cache_mode == "COLD_CACHE" ? free_slot(true) : -1;
        while (needs_hot && free_slot(false) < 0) {
            key_type victim;
            require(demand_victim(false, victim), "mandatory hot demand cannot be admitted");
            auto found = entries.find(victim);
            require(found != entries.end(), "hot victim disappeared");
            emit_demand_eviction("DEMAND_HOT_EVICT", token, victim);
            if (is_speculative(found->second)) discard(victim, "demand_hot_eviction");
            else found->second.hot_slot = -1;
        }
        return {cold_slot, needs_hot ? free_slot(false) : -1};
    }

    void issue_demands(uint64_t token, int32_t layer, const std::vector<int32_t> & selected) {
        if (demand_mode == "ISSUE_AHEAD") {
            for (int32_t expert : selected) {
                emit(actions, { {"type", "DEMAND_ISSUE"}, {"flight_ordinal", -1}, {"token", token},
                    {"layer", layer}, {"expert", expert}, {"cold_slot", -1}, {"hot_slot", -1},
                    {"priority", "DEMAND_CURRENT_LAYER"} });
            }
        }
    }

    void consume_demands(uint64_t token, int32_t layer, const std::vector<int32_t> & selected) {
        for (int32_t expert : selected) {
            if (demand_mode == "SERIAL") {
                emit(actions, { {"type", "DEMAND_ISSUE"}, {"flight_ordinal", -1}, {"token", token},
                    {"layer", layer}, {"expert", expert}, {"cold_slot", -1}, {"hot_slot", -1},
                    {"priority", "DEMAND_CURRENT_LAYER"} });
            }
            clock++;
            const key_type key = {layer, expert};
            counters.demand_keys++;
            auto found = entries.find(key);
            if (found != entries.end() && found->second.hot_speculative) {
                auto & entry = found->second;
                const std::string source_phase = entry.phase;
                if (entry.phase == "READY") {
                    counters.timely_useful++;
                    counters.demand_hits++;
                    emit(outcomes, { {"type", "TIMELY_USEFUL"}, {"flight_ordinal", entry.flight},
                        {"token", token}, {"layer", layer}, {"expert", expert},
                        {"cold_slot", entry.cold_slot}, {"hot_slot", entry.hot_slot},
                        {"reason", "ready_before_demand"} });
                    emit(actions, { {"type", "DEMAND_HIT"}, {"flight_ordinal", entry.flight},
                        {"token", token}, {"layer", layer}, {"expert", expert},
                        {"cold_slot", entry.cold_slot}, {"hot_slot", entry.hot_slot},
                        {"source_origin", "SPECULATIVE"}, {"source_tier", "HOT"} });
                } else {
                    counters.late_joined++;
                    counters.demand_loads++;
                    emit(outcomes, { {"type", "LATE_JOINED"}, {"flight_ordinal", entry.flight},
                        {"token", token}, {"layer", layer}, {"expert", expert},
                        {"cold_slot", entry.cold_slot}, {"hot_slot", entry.hot_slot},
                        {"reason", "demand_promoted_exact_generation"},
                        {"completion_phase", source_phase} });
                    emit(actions, { {"type", "DEMAND_PROMOTE"}, {"flight_ordinal", entry.flight},
                        {"token", token}, {"layer", layer}, {"expert", expert},
                        {"cold_slot", entry.cold_slot}, {"hot_slot", entry.hot_slot},
                        {"source_phase", source_phase}, {"priority", "DEMAND_CURRENT_LAYER"} });
                }
                entry.origin = "DEMAND";
                entry.phase = "READY";
                entry.hot_speculative = false;
                entry.cold_last_touch = clock;
                entry.hot_last_touch = clock;
                continue;
            }
            if (found != entries.end() && (found->second.origin != "SPECULATIVE" ||
                    found->second.phase == "READY")) {
                auto & entry = found->second;
                const std::string source_origin = entry.origin;
                const char * source_tier = entry.hot_slot >= 0 ? "HOT" : "COLD";
                if (entry.origin == "SPECULATIVE") {
                    counters.timely_useful++;
                    emit(outcomes, { {"type", "TIMELY_USEFUL"}, {"flight_ordinal", entry.flight},
                        {"token", token}, {"layer", layer}, {"expert", expert},
                        {"cold_slot", entry.cold_slot}, {"hot_slot", entry.hot_slot},
                        {"reason", "ready_before_demand"} });
                    entry.origin = "DEMAND";
                } else if (entry.origin == "STATIC_SEED") {
                    entry.origin = "DEMAND";
                }
                if (entry.hot_slot < 0 && miss_policy == "PROMOTE_AND_GPU") {
                    while (free_slot(false) < 0) {
                        key_type victim;
                        require(demand_victim(false, victim), "mandatory hot promotion cannot be admitted");
                        auto victim_entry = entries.find(victim);
                        require(victim_entry != entries.end(), "hot promotion victim disappeared");
                        emit_demand_eviction("DEMAND_HOT_EVICT", token, victim);
                        if (is_speculative(victim_entry->second)) discard(victim, "demand_hot_eviction");
                        else victim_entry->second.hot_slot = -1;
                    }
                    entry.hot_slot = free_slot(false);
                }
                entry.cold_last_touch = clock;
                entry.hot_last_touch = clock;
                counters.demand_hits++;
                emit(actions, { {"type", "DEMAND_HIT"}, {"flight_ordinal", entry.flight}, {"token", token},
                    {"layer", layer}, {"expert", expert}, {"cold_slot", entry.cold_slot},
                    {"hot_slot", entry.hot_slot}, {"source_origin", source_origin}, {"source_tier", source_tier} });
                continue;
            }
            if (found != entries.end() && found->second.origin == "SPECULATIVE") {
                auto & entry = found->second;
                const std::string source_phase = entry.phase;
                counters.late_joined++;
                counters.demand_loads++;
                emit(outcomes, { {"type", "LATE_JOINED"}, {"flight_ordinal", entry.flight},
                    {"token", token}, {"layer", layer}, {"expert", expert}, {"cold_slot", entry.cold_slot},
                    {"hot_slot", entry.hot_slot}, {"reason", "demand_promoted_exact_generation"},
                    {"completion_phase", source_phase} });
                entry.origin = "DEMAND";
                entry.phase = "READY";
                if (entry.hot_slot < 0) {
                    while (free_slot(false) < 0) {
                        key_type victim;
                        require(demand_victim(false, victim), "mandatory late promotion cannot be admitted");
                        auto victim_entry = entries.find(victim);
                        require(victim_entry != entries.end(), "late promotion victim disappeared");
                        emit_demand_eviction("DEMAND_HOT_EVICT", token, victim);
                        if (is_speculative(victim_entry->second)) discard(victim, "demand_hot_eviction");
                        else victim_entry->second.hot_slot = -1;
                    }
                    entry.hot_slot = free_slot(false);
                }
                entry.cold_last_touch = clock;
                entry.hot_last_touch = clock;
                emit(actions, { {"type", "DEMAND_PROMOTE"}, {"flight_ordinal", entry.flight},
                    {"token", token}, {"layer", layer}, {"expert", expert}, {"cold_slot", entry.cold_slot},
                    {"hot_slot", entry.hot_slot}, {"source_phase", source_phase},
                    {"priority", "DEMAND_CURRENT_LAYER"} });
                continue;
            }
            if (found != entries.end()) discard(key, "demand_replaced_unready");
            const auto size = sizes.find(key);
            require(size != sizes.end(), "demand key has no byte record");
            const auto slots = ensure_demand_capacity(token, size->second.second,
                cache_mode == "HOT_CACHE" || miss_policy == "PROMOTE_AND_GPU");
            entries[key] = {layer, expert, size->second.first, size->second.second, slots.first, slots.second,
                "DEMAND", token, layer, 0, -1, clock, clock, "READY"};
            counters.demand_loads++;
            emit(actions, { {"type", "DEMAND_LOAD"}, {"flight_ordinal", -1}, {"token", token},
                {"layer", layer}, {"expert", expert}, {"cold_slot", slots.first}, {"hot_slot", slots.second} });
        }
    }

    bool speculative_victims(
            uint64_t required_physical,
            bool needs_cold,
            bool needs_hot,
            bool require_cold_victim,
            bool require_hot_victim,
            const std::set<key_type> & excluded,
            std::vector<key_type> & result) const {
        uint64_t simulated_bytes = occupied_bytes();
        bool cold_free = !needs_cold || (free_slot(true) >= 0 && !require_cold_victim);
        bool hot_free = !needs_hot || (free_slot(false) >= 0 && !require_hot_victim);
        std::vector<std::pair<key_type, replay_entry>> available;
        for (const auto & item : entries) {
            if (is_speculative(item.second) && !excluded.count(item.first)) available.push_back(item);
        }
        std::sort(available.begin(), available.end(), [&](const auto & lhs, const auto & rhs) {
            const auto left = std::make_tuple(lhs.second.deadline_token, layer_indices.at(lhs.second.deadline_layer),
                lhs.second.score, lhs.second.cold_slot, lhs.second.hot_slot);
            const auto right = std::make_tuple(rhs.second.deadline_token, layer_indices.at(rhs.second.deadline_layer),
                rhs.second.score, rhs.second.cold_slot, rhs.second.hot_slot);
            return left < right;
        });
        for (const auto & item : available) {
            if (required_physical <= limits.cold_capacity_bytes - simulated_bytes && cold_free && hot_free) break;
            result.push_back(item.first);
            if (item.second.origin == "SPECULATIVE" && item.second.cold_slot >= 0) {
                require(item.second.physical_bytes <= simulated_bytes, "speculative occupancy underflow");
                simulated_bytes -= item.second.physical_bytes;
                cold_free = true;
            }
            if (item.second.hot_slot >= 0) hot_free = true;
        }
        return required_physical <= limits.cold_capacity_bytes - simulated_bytes && cold_free && hot_free;
    }

    void submit_batch(const std::vector<json> & requested) {
        uint32_t in_flight = 0;
        uint64_t storage_in_flight = 0;
        uint64_t h2d_in_flight = 0;
        for (const auto & item : entries) {
            if (!is_speculative(item.second) || item.second.phase == "READY") continue;
            in_flight++;
            if (item.second.origin == "SPECULATIVE" && item.second.cold_slot >= 0) {
                require(item.second.physical_bytes <= UINT64_MAX - storage_in_flight,
                    "speculative storage in-flight overflow");
                storage_in_flight += item.second.physical_bytes;
            }
            if (readiness == "DEVICE_READY") {
                require(item.second.payload_bytes <= UINT64_MAX - h2d_in_flight,
                    "speculative h2d in-flight overflow");
                h2d_in_flight += item.second.payload_bytes;
            }
        }
        std::vector<key_type> accepted;
        for (const auto & candidate : requested) {
            const uint64_t token = candidate.at("trigger_token");
            const uint64_t deadline_token = candidate.at("trigger") == "ROUTER_RESULT" ? token : token + 1;
            const int32_t deadline_layer = candidate.at("target_layer");
            const key_type key = {deadline_layer, candidate.at("expert")};
            const auto size = sizes.find(key);
            require(size != sizes.end(), "candidate key has no byte record");
            const uint64_t payload = size->second.first;
            const uint64_t physical = size->second.second;
            auto existing = entries.find(key);
            const bool promote_cold = readiness == "DEVICE_READY" && existing != entries.end() &&
                existing->second.cold_slot >= 0 && existing->second.hot_slot < 0 &&
                !is_speculative(existing->second);
            const uint64_t storage = cache_mode == "COLD_CACHE" && !promote_cold ? physical : 0;
            const uint64_t h2d = readiness == "DEVICE_READY" ? payload : 0;
            std::string reason;
            if (existing != entries.end() && !promote_cold) reason = "target_ready_or_higher_priority";
            else if (in_flight >= limits.max_speculative_flights) reason = "flight_budget";
            else if (storage > limits.max_speculative_storage_bytes_in_flight - storage_in_flight) {
                reason = "storage_in_flight_budget";
            } else if (h2d > limits.max_speculative_h2d_bytes_in_flight - h2d_in_flight) {
                reason = "h2d_in_flight_budget";
            } else if (storage > limits.max_speculative_storage_bytes_per_token - token_storage[token]) {
                reason = "storage_token_budget";
            } else if (h2d > limits.max_speculative_h2d_bytes_per_token - token_h2d[token]) {
                reason = "h2d_token_budget";
            }
            uint32_t cold_speculative = 0;
            uint32_t hot_speculative = 0;
            for (const auto & item : entries) {
                if (!is_speculative(item.second)) continue;
                if (item.second.origin == "SPECULATIVE" && item.second.cold_slot >= 0) cold_speculative++;
                if (item.second.hot_slot >= 0) hot_speculative++;
            }
            std::set<key_type> excluded(accepted.begin(), accepted.end());
            std::vector<key_type> victims;
            const bool needs_cold = cache_mode == "COLD_CACHE" && !promote_cold;
            const bool admitted = !reason.empty() ? false : speculative_victims(storage, needs_cold,
                readiness == "DEVICE_READY", needs_cold && cold_speculative >= limits.max_speculative_cold_slots,
                readiness == "DEVICE_READY" && hot_speculative >= limits.max_speculative_hot_slots,
                excluded, victims);
            if (reason.empty() && !admitted) {
                reason = "demand_state_protected";
                counters.prevented_demand_evictions++;
            }
            if (!reason.empty()) {
                counters.rejected++;
                emit(actions, { {"type", "REJECTED"}, {"flight_ordinal", candidate.at("flight_ordinal")},
                    {"token", token}, {"layer", key.first}, {"expert", key.second},
                    {"cold_slot", -1}, {"hot_slot", -1}, {"reason", reason} });
                continue;
            }
            for (const auto & victim : victims) {
                const replay_entry entry = entries.at(victim);
                emit(actions, { {"type", "EVICT"}, {"flight_ordinal", entry.flight}, {"token", token},
                    {"layer", victim.first}, {"expert", victim.second}, {"cold_slot", entry.cold_slot},
                    {"hot_slot", entry.hot_slot}, {"reason", "speculative_replacement"} });
                counters.speculative_replacements++;
                discard(victim, "speculative_replacement");
            }
            const int32_t cold_slot = promote_cold ? existing->second.cold_slot :
                (cache_mode == "COLD_CACHE" ? free_slot(true) : -1);
            const int32_t hot_slot = readiness == "DEVICE_READY" ? free_slot(false) : -1;
            const uint64_t flight_ordinal = candidate.at("flight_ordinal");
            const std::string phase = ready_before_deadline.count(flight_ordinal) ? "READY" :
                (cost.predictor_compute_ns >= cost.lead_ns ? "QUEUED" :
                    (cost.predictor_compute_ns + cost.speculative_service_ns > cost.lead_ns ?
                        "SUBMITTED" : "READY"));
            if (promote_cold) {
                auto & entry = existing->second;
                entry.hot_slot = hot_slot;
                entry.hot_speculative = true;
                entry.deadline_token = deadline_token;
                entry.deadline_layer = deadline_layer;
                entry.score = candidate.at("score");
                entry.flight = flight_ordinal;
                entry.hot_last_touch = clock;
                entry.phase = phase;
            } else {
                entries[key] = {key.first, key.second, payload, physical, cold_slot, hot_slot,
                    "SPECULATIVE", deadline_token, deadline_layer, candidate.at("score"),
                    int64_t(flight_ordinal), clock, hot_slot >= 0 ? clock : 0, phase,
                    readiness == "DEVICE_READY"};
            }
            accepted.push_back(key);
            in_flight++;
            storage_in_flight += storage;
            h2d_in_flight += h2d;
            token_storage[token] += storage;
            token_h2d[token] += h2d;
            counters.accepted++;
            const uint64_t submitted_storage = phase == "QUEUED" ? 0 : storage;
            const uint64_t submitted_h2d = phase == "QUEUED" ? 0 : h2d;
            counters.storage_bytes += submitted_storage;
            counters.h2d_bytes += submitted_h2d;
            emit(actions, { {"type", "ENQUEUE"}, {"flight_ordinal", candidate.at("flight_ordinal")},
                {"token", token}, {"layer", key.first}, {"expert", key.second}, {"cold_slot", cold_slot},
                {"hot_slot", hot_slot}, {"deadline_token", deadline_token},
                {"deadline_layer", deadline_layer},
                {"priority", policy == "STATIC_LAYER" || policy == "RANDOM_BASELINE" ?
                    "PREFETCH_SPECULATIVE" : "PREFETCH_NEXT"},
                {"storage_bytes", storage}, {"h2d_bytes", h2d},
                {"submitted_storage_bytes", submitted_storage}, {"submitted_h2d_bytes", submitted_h2d},
                {"completion_phase", phase} });
        }
        for (const auto & key : accepted) {
            const auto & entry = entries.at(key);
            if (entry.phase == "READY") {
                emit(actions, { {"type", "READY"}, {"flight_ordinal", entry.flight},
                    {"token", entry.deadline_token}, {"layer", key.first}, {"expert", key.second},
                    {"cold_slot", entry.cold_slot}, {"hot_slot", entry.hot_slot}, {"readiness", readiness} });
            }
        }
    }

    void resolve_deadline(uint64_t token, int32_t layer) {
        std::vector<key_type> expired;
        for (const auto & item : entries) {
            if (is_speculative(item.second) && item.second.deadline_token == token &&
                    item.second.deadline_layer == layer) expired.push_back(item.first);
        }
        for (const auto & key : expired) discard(key, "deadline_unused");
    }

    void discard_unused_deadlines(
            uint64_t token, int32_t layer, const std::vector<int32_t> & selected) {
        std::vector<key_type> expired;
        for (const auto & item : entries) {
            if (!is_speculative(item.second) || item.second.deadline_token != token ||
                    item.second.deadline_layer != layer ||
                    std::find(selected.begin(), selected.end(), item.first.second) != selected.end()) continue;
            expired.push_back(item.first);
        }
        for (const auto & key : expired) discard(key, "deadline_unused");
    }

    void initialize_resident() {
        require(initial_resident.is_array(), "initial_resident must be an array");
        std::set<int32_t> used_cold;
        std::set<int32_t> used_hot;
        for (const auto & item : initial_resident) {
            require_fields(item, {"layer", "expert", "cold_slot", "hot_slot", "cold_generation",
                "hot_generation", "cold_last_use", "hot_last_use", "origin"}, "initial resident");
            const int32_t layer = int32_value(item.at("layer"), "initial layer");
            const int32_t expert = int32_value(item.at("expert"), "initial expert");
            require(layer >= 0 && expert >= 0, "initial resident key is negative");
            const key_type key = {layer, expert};
            const auto size = sizes.find(key);
            require(size != sizes.end() && !entries.count(key), "invalid or duplicate initial resident key");
            const int32_t cold_slot = int32_value(item.at("cold_slot"), "initial cold slot");
            const int32_t hot_slot = int32_value(item.at("hot_slot"), "initial hot slot");
            require(cold_slot >= -1 && hot_slot >= -1 && (cold_slot >= 0 || hot_slot >= 0),
                "initial resident has invalid cache slots");
            require((cache_mode == "COLD_CACHE") == (cold_slot >= 0),
                "initial resident disagrees with cache mode");
            require(cold_slot < int32_t(cold_slot_capacity) && hot_slot < int32_t(limits.hot_capacity_slots),
                "initial resident slot exceeds capacity");
            require((cold_slot < 0 || used_cold.insert(cold_slot).second) &&
                (hot_slot < 0 || used_hot.insert(hot_slot).second), "duplicate initial resident slot");
            const uint64_t cold_generation = unsigned_value(item.at("cold_generation"), "initial cold generation");
            const uint64_t hot_generation = unsigned_value(item.at("hot_generation"), "initial hot generation");
            const uint64_t cold_touch = unsigned_value(item.at("cold_last_use"), "initial cold last use");
            const uint64_t hot_touch = unsigned_value(item.at("hot_last_use"), "initial hot last use");
            require((cold_slot >= 0) == (cold_generation != 0 && cold_touch != 0) &&
                (hot_slot >= 0) == (hot_generation != 0 && hot_touch != 0),
                "initial resident generation/recency disagrees with slots");
            const std::string origin = string_value(item.at("origin"), "initial origin");
            require(origin == "DEMAND" || origin == "STATIC_SEED",
                "initial resident must be quiescent demand or seed state");
            entries[key] = {layer, expert, size->second.first, size->second.second, cold_slot, hot_slot,
                origin, 0, layer, 0, -1, cold_touch, hot_touch, "READY"};
            clock = std::max(clock, std::max(cold_touch, hot_touch));
        }
        require(occupied_bytes() <= limits.cold_capacity_bytes,
            "initial resident bytes exceed cold capacity");
        require(initial_resident.empty() || seed_mode == "OFF",
            "replay cannot combine initial resident state and blocking seed");
    }

    void seed() {
        if (seed_mode != "BLOCKING_HOT") return;
        uint64_t seed_bytes = 0;
        for (const auto & item : profile.seed) {
            require(item.physical_bytes <= UINT64_MAX - seed_bytes, "seed bytes overflow");
            seed_bytes += item.physical_bytes;
        }
        require(!profile.seed.empty() && profile.seed.size() <= limits.hot_capacity_slots &&
            (cache_mode != "COLD_CACHE" || (seed_bytes <= limits.cold_capacity_bytes &&
                profile.seed.size() <= cold_slot_capacity)),
            "blocking seed does not fit exact replay capacity");
        for (const auto & item : profile.seed) {
            clock++;
            const key_type key = {item.layer, item.expert};
            const int32_t cold_slot = cache_mode == "COLD_CACHE" ? free_slot(true) : -1;
            const int32_t hot_slot = free_slot(false);
            entries[key] = {item.layer, item.expert, item.payload_bytes, item.physical_bytes,
                cold_slot, hot_slot, "STATIC_SEED", 0, item.layer, item.count, -1, clock, clock, "READY"};
            emit(actions, { {"type", "SEED_LOAD"}, {"flight_ordinal", -1}, {"token", 0},
                {"layer", item.layer}, {"expert", item.expert}, {"cold_slot", cold_slot},
                {"hot_slot", hot_slot},
                {"storage_bytes", cache_mode == "COLD_CACHE" ? item.physical_bytes : 0},
                {"h2d_bytes", item.payload_bytes} });
        }
        const auto highest = std::min_element(profile.seed.begin(), profile.seed.end(), [](const auto & lhs, const auto & rhs) {
            return lhs.count != rhs.count ? lhs.count > rhs.count :
                (lhs.layer != rhs.layer ? lhs.layer < rhs.layer : lhs.expert < rhs.expert);
        });
        require(highest != profile.seed.end(), "blocking seed is empty");
        clock++;
        auto & entry = entries.at({highest->layer, highest->expert});
        entry.cold_last_touch = clock;
        entry.hot_last_touch = clock;
        emit(actions, { {"type", "SEED_TOUCH"}, {"flight_ordinal", -1}, {"token", 0},
            {"layer", highest->layer}, {"expert", highest->expert}, {"cold_slot", entry.cold_slot},
            {"hot_slot", entry.hot_slot} });
    }
};

json replay(const json & input) {
    require_fields(input, {"schema_version", "profile_path", "policy", "cache_mode", "miss_policy",
        "transport", "readiness", "temporal_window_tokens",
        "candidates_per_target", "request_ordinal", "events", "completion_order", "ready_before_deadline",
        "limits", "seed_mode",
        "demand_mode", "initial_resident"}, "replay");
    require(input.at("schema_version") == "phase10-prefetch-replay-v1", "unsupported replay schema");
    require(input.at("events").is_array(), "events must be an array");
    require(input.at("completion_order").is_array(), "completion_order must be an array");
    require(input.at("ready_before_deadline").is_array(), "ready_before_deadline must be an array");
    require(input.at("initial_resident").is_array(), "initial_resident must be an array");

    llm_expert_prefetch_profile profile;
    std::string profile_error;
    const std::string profile_path = string_value(input.at("profile_path"), "profile_path");
    const auto loaded = llm_expert_prefetch_load_profile(profile_path, 64U*1024U*1024U, nullptr, profile, profile_error);
    if (!loaded.is_ready()) throw replay_error("profile load failed: " + profile_error);
    const auto policy = policy_value(string_value(input.at("policy"), "policy"));
    const std::string cache_mode = string_value(input.at("cache_mode"), "cache_mode");
    const std::string miss_policy = string_value(input.at("miss_policy"), "miss_policy");
    require(cache_mode == "HOT_CACHE" || cache_mode == "COLD_CACHE", "unknown cache mode");
    require(miss_policy == "PROMOTE_AND_GPU" || miss_policy == "CPU_FALLBACK" || miss_policy == "AUTO",
        "unknown miss policy");
    const std::string transport = string_value(input.at("transport"), "transport");
    require(transport == "BUFFERED" || transport == "DIRECT_IO" || transport == "HOST_TO_DEVICE",
        "unknown transport");
    require(transport == profile.selected_transport, "replay transport does not match profile selection");
    require((cache_mode == "HOT_CACHE" && transport == "HOST_TO_DEVICE") ||
            (cache_mode == "COLD_CACHE" && (transport == "BUFFERED" || transport == "DIRECT_IO")),
        "replay transport does not match cache mode");
    const auto readiness = readiness_value(string_value(input.at("readiness"), "readiness"));
    const std::string seed_mode = string_value(input.at("seed_mode"), "seed_mode");
    const std::string demand_mode = string_value(input.at("demand_mode"), "demand_mode");
    require(seed_mode == "OFF" || seed_mode == "BLOCKING_HOT", "unknown seed mode");
    require(demand_mode == "ISSUE_AHEAD" || demand_mode == "SERIAL", "unknown demand mode");
    require(policy == LLAMA_EXPERT_PREFETCH_POLICY_OFF || seed_mode == "OFF",
        "replay does not combine prediction and seed");
    require(cache_mode != "HOT_CACHE" ||
        (readiness == LLAMA_EXPERT_PREFETCH_READINESS_DEVICE_READY && miss_policy == "PROMOTE_AND_GPU"),
        "HOT_CACHE replay requires device readiness and GPU promotion");
    const auto limits = limits_value(input.at("limits"), policy != LLAMA_EXPERT_PREFETCH_POLICY_OFF, cache_mode);
    const uint32_t temporal_window = uint32_value(input.at("temporal_window_tokens"), "temporal_window_tokens");
    const uint32_t candidate_count = uint32_value(input.at("candidates_per_target"), "candidates_per_target");
    const uint64_t request_ordinal = unsigned_value(input.at("request_ordinal"), "request_ordinal");
    require(request_ordinal != 0, "zero replay request ordinal");
    require(policy == LLAMA_EXPERT_PREFETCH_POLICY_OFF ? candidate_count == 0 : candidate_count != 0,
        "replay candidate count does not match policy state");
    std::set<uint64_t> completion_ids;
    for (const auto & completion : input.at("completion_order")) {
        require(completion_ids.insert(unsigned_value(completion, "completion_order")).second,
            "duplicate completion ordinal");
    }

    json candidates = json::array();
    uint64_t predictor_state_digest = fnv_offset;
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
        require(llm_expert_prefetch_copy_config(&public_config, profile_path.c_str(),
            cache_mode == "COLD_CACHE" ? LLAMA_EXPERT_WEIGHTS_MODE_COLD_CACHE : LLAMA_EXPERT_WEIGHTS_MODE_HOT_CACHE,
            profile.target.experts_per_layer, 32, UINT64_C(1) << 30, 64,
            UINT64_C(1) << 30, UINT64_C(1) << 30,
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
        predictor_state_digest = predictor.state().digest;
    }
    if (!completion_ids.empty()) {
        require(completion_ids.size() == candidates.size(), "completion order is not a candidate permutation");
        uint64_t expected = 0;
        for (uint64_t completion : completion_ids) require(completion == expected++,
            "completion order is not a candidate permutation");
    }
    std::set<uint64_t> ready_ids;
    for (const auto & ready : input.at("ready_before_deadline")) {
        const uint64_t ordinal = unsigned_value(ready, "ready_before_deadline");
        require(ordinal < candidates.size() && ready_ids.insert(ordinal).second,
            "invalid ready-before-deadline ordinal");
    }
    hierarchy_replay hierarchy(profile, input.at("events"), input.at("initial_resident"), candidates,
        std::move(ready_ids), input.at("policy").get<std::string>(), cache_mode, miss_policy, transport,
        input.at("readiness").get<std::string>(), limits, seed_mode, demand_mode);
    json state = hierarchy.run();
    json output = {
        {"schema_version", "phase10-prefetch-replay-output-v1"},
        {"profile_sha256", profile.profile_sha256},
        {"policy", input.at("policy")},
        {"cache_mode", cache_mode}, {"miss_policy", miss_policy},
        {"transport", transport},
        {"seed_mode", seed_mode}, {"demand_mode", demand_mode},
        {"candidate_stream", candidates},
        {"predictor_state_digest", predictor_state_digest},
    };
    for (const char * field : {"action_stream", "outcome_stream", "state_digest", "summary", "resident"}) {
        output[field] = std::move(state[field]);
    }
    return output;
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
