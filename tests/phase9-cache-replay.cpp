#include "llama-expert-cache-policy.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;

namespace {

struct replay_error : std::runtime_error {
    using std::runtime_error::runtime_error;
};

void require(bool condition, const char * message) {
    if (!condition) throw replay_error(message);
}

void require_ready(llm_expert_cache_policy_result result, const std::string & message) {
    if (!result.is_ready()) {
        throw replay_error(message + " (policy error " + std::to_string(unsigned(result.error)) + ")");
    }
}

void require_fields(const json & value, std::initializer_list<const char *> expected, const char * name) {
    require(value.is_object(), "expected JSON object");
    std::set<std::string> actual;
    for (const auto & item : value.items()) actual.insert(item.key());
    std::set<std::string> wanted;
    for (const char * field : expected) wanted.insert(field);
    if (actual != wanted) throw replay_error(std::string(name) + " fields do not match version 1");
}

// Small self-contained SHA-256 used only to bind replay output to exact input bytes.
struct sha256 {
    std::array<uint32_t, 8> state = {
        0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
        0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U,
    };
    std::array<uint8_t, 64> block = {};
    uint64_t bytes = 0;
    size_t used = 0;

    static uint32_t rotate(uint32_t value, uint32_t count) { return value >> count | value << (32 - count); }

    void transform() {
        static constexpr uint32_t k[64] = {
            0x428a2f98U,0x71374491U,0xb5c0fbcfU,0xe9b5dba5U,0x3956c25bU,0x59f111f1U,0x923f82a4U,0xab1c5ed5U,
            0xd807aa98U,0x12835b01U,0x243185beU,0x550c7dc3U,0x72be5d74U,0x80deb1feU,0x9bdc06a7U,0xc19bf174U,
            0xe49b69c1U,0xefbe4786U,0x0fc19dc6U,0x240ca1ccU,0x2de92c6fU,0x4a7484aaU,0x5cb0a9dcU,0x76f988daU,
            0x983e5152U,0xa831c66dU,0xb00327c8U,0xbf597fc7U,0xc6e00bf3U,0xd5a79147U,0x06ca6351U,0x14292967U,
            0x27b70a85U,0x2e1b2138U,0x4d2c6dfcU,0x53380d13U,0x650a7354U,0x766a0abbU,0x81c2c92eU,0x92722c85U,
            0xa2bfe8a1U,0xa81a664bU,0xc24b8b70U,0xc76c51a3U,0xd192e819U,0xd6990624U,0xf40e3585U,0x106aa070U,
            0x19a4c116U,0x1e376c08U,0x2748774cU,0x34b0bcb5U,0x391c0cb3U,0x4ed8aa4aU,0x5b9cca4fU,0x682e6ff3U,
            0x748f82eeU,0x78a5636fU,0x84c87814U,0x8cc70208U,0x90befffaU,0xa4506cebU,0xbef9a3f7U,0xc67178f2U,
        };
        uint32_t words[64];
        for (size_t index = 0; index < 16; ++index) {
            words[index] = uint32_t(block[index*4]) << 24 | uint32_t(block[index*4 + 1]) << 16 |
                uint32_t(block[index*4 + 2]) << 8 | block[index*4 + 3];
        }
        for (size_t index = 16; index < 64; ++index) {
            const uint32_t s0 = rotate(words[index - 15], 7) ^ rotate(words[index - 15], 18) ^ (words[index - 15] >> 3);
            const uint32_t s1 = rotate(words[index - 2], 17) ^ rotate(words[index - 2], 19) ^ (words[index - 2] >> 10);
            words[index] = words[index - 16] + s0 + words[index - 7] + s1;
        }
        uint32_t a=state[0],b=state[1],c=state[2],d=state[3],e=state[4],f=state[5],g=state[6],h=state[7];
        for (size_t index = 0; index < 64; ++index) {
            const uint32_t s1 = rotate(e, 6) ^ rotate(e, 11) ^ rotate(e, 25);
            const uint32_t choice = (e & f) ^ (~e & g);
            const uint32_t temporary1 = h + s1 + choice + k[index] + words[index];
            const uint32_t s0 = rotate(a, 2) ^ rotate(a, 13) ^ rotate(a, 22);
            const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t temporary2 = s0 + majority;
            h=g; g=f; f=e; e=d+temporary1; d=c; c=b; b=a; a=temporary1+temporary2;
        }
        state[0]+=a; state[1]+=b; state[2]+=c; state[3]+=d;
        state[4]+=e; state[5]+=f; state[6]+=g; state[7]+=h;
    }

    void update(const std::string & input) {
        for (uint8_t byte : input) {
            block[used++] = byte;
            bytes++;
            if (used == block.size()) { transform(); used = 0; }
        }
    }

    std::string finish() {
        const uint64_t bits = bytes*8;
        block[used++] = 0x80;
        if (used > 56) { while (used < 64) block[used++] = 0; transform(); used = 0; }
        while (used < 56) block[used++] = 0;
        for (int shift = 56; shift >= 0; shift -= 8) block[used++] = uint8_t(bits >> shift);
        transform();
        std::ostringstream output;
        output << std::hex << std::setfill('0');
        for (uint32_t word : state) output << std::setw(8) << word;
        return output.str();
    }
};

std::string canonical_input_sha256(const json & input) {
    sha256 digest;
    digest.update(input.dump(2) + "\n");
    return digest.finish();
}

llama_expert_cache_policy policy_enum(const std::string & value) {
    if (value == "LRU") return LLAMA_EXPERT_CACHE_POLICY_LRU;
    if (value == "LFRU") return LLAMA_EXPERT_CACHE_POLICY_LFRU;
    if (value == "SLRU") return LLAMA_EXPERT_CACHE_POLICY_SLRU;
    if (value == "LFU_AGING") return LLAMA_EXPERT_CACHE_POLICY_LFU_AGING;
    throw replay_error("unknown policy");
}

llama_expert_cache_policy_scope scope_enum(const std::string & value) {
    if (value == "GLOBAL") return LLAMA_EXPERT_CACHE_POLICY_SCOPE_GLOBAL;
    if (value == "PER_LAYER") return LLAMA_EXPERT_CACHE_POLICY_SCOPE_PER_LAYER;
    throw replay_error("unknown scope");
}

llama_expert_cache_admission admission_enum(const std::string & value) {
    if (value == "ALWAYS") return LLAMA_EXPERT_CACHE_ADMISSION_ALWAYS;
    if (value == "FREQUENCY_WINDOW") return LLAMA_EXPERT_CACHE_ADMISSION_FREQUENCY_WINDOW;
    throw replay_error("unknown admission");
}

llm_expert_cache_policy_config_internal config_from_json(const json & value, llm_expert_cache_policy_tier tier) {
    require_fields(value, { "schema_version", "policy", "scope", "slru_protected_ratio_bps",
        "admission", "admission_window_events", "lfu_aging_interval_events" }, "config");
    require(value.at("schema_version") == "cache-policy-config-v1", "unsupported config schema");
    const llama_expert_cache_policy_config public_config = {
        LLAMA_EXPERT_CACHE_POLICY_VERSION_1,
        sizeof(llama_expert_cache_policy_config),
        policy_enum(value.at("policy")),
        scope_enum(value.at("scope")),
        value.at("slru_protected_ratio_bps"),
        admission_enum(value.at("admission")),
        value.at("admission_window_events"),
        value.at("lfu_aging_interval_events"),
        {},
    };
    llm_expert_cache_policy_config_internal result;
    require_ready(llm_expert_cache_policy_copy_config(&public_config, tier, result), "invalid config");
    return result;
}

std::string event_name(llm_expert_cache_policy_event_type type) {
    static const char * names[] = { "REQUEST_BEGIN", "PHASE_TRANSITION", "DEMAND", "HIT",
        "VICTIM_SELECTED", "EVICT", "LOAD_BEGIN", "LOAD_COMPLETE", "LOAD_FAILED", "PIN",
        "UNPIN", "OPTIONAL_ADMISSION", "REQUEST_END", "RESET", "SURRENDER" };
    return names[unsigned(type)];
}

json event_json(const llm_expert_cache_policy_event & event) {
    return {
        {"schema_version", event.schema_version},
        {"request_ordinal", event.request_ordinal},
        {"ubatch_ordinal", event.ubatch_ordinal},
        {"tier", event.tier == llm_expert_cache_policy_tier::hot ? "HOT" : "COLD"},
        {"type", event_name(event.type)},
        {"event_sequence", event.event_sequence},
        {"demand_ordinal", event.demand_ordinal},
        {"origin_operation_ordinal", event.origin_operation_ordinal},
        {"phase", event.phase == llm_expert_cache_policy_phase::prefill ? "PREFILL" : "DECODE"},
        {"layer", event.key.layer}, {"expert", event.key.expert},
        {"occurrence_count", event.occurrence_count},
        {"logical_payload_bytes", event.logical_bundle_bytes},
        {"physical_slot_footprint_bytes", event.physical_slot_footprint_bytes},
        {"slot", event.slot == UINT32_MAX ? -1 : int64_t(event.slot)},
        {"generation", event.generation},
        {"domain", event.domain == UINT32_MAX ? -1 : int64_t(event.domain)},
        {"eligible", event.eligible}, {"decision", event.decision}, {"reason", event.reason},
        {"state_digest", event.state_digest},
    };
}

struct mechanism_slot {
    llm_expert_cache_policy_key key = { -1, -1 };
    uint64_t generation = 0;
    uint64_t logical_bytes = 0;
    bool loading = false;
    bool ready = false;
};

bool key_less(const llm_expert_cache_policy_key & lhs, const llm_expert_cache_policy_key & rhs) {
    return lhs.layer != rhs.layer ? lhs.layer < rhs.layer : lhs.expert < rhs.expert;
}

bool key_equal(const llm_expert_cache_policy_key & lhs, const llm_expert_cache_policy_key & rhs) {
    return lhs.layer == rhs.layer && lhs.expert == rhs.expert;
}

struct tier_mechanism {
    llm_expert_cache_policy policy;
    llm_expert_cache_policy_tier tier;
    uint64_t footprint;
    std::vector<mechanism_slot> slots;
    std::map<std::pair<int32_t, int32_t>, uint32_t> mapping;
    uint64_t admissions = 0;
    uint64_t evictions = 0;
    uint64_t optional_rejections = 0;
    uint64_t initial_digest = 0;

    tier_mechanism(
            const json & value,
            llm_expert_cache_policy_tier requested_tier,
            const std::vector<int32_t> & layers,
            uint32_t experts_per_layer,
            uint64_t requested_footprint) :
        tier(requested_tier), footprint(requested_footprint), slots(value.at("slots").get<uint32_t>()) {
        require_fields(value, { "slots", "config" }, "tier");
        auto config = config_from_json(value.at("config"), tier);
        require_ready(policy.initialize(config, tier, layers.data(), layers.size(), experts_per_layer,
            1, slots.size(), footprint, 1000000), "policy initialization failed");
        initial_digest = policy.diagnostics().state_digest;
    }

    bool contains(llm_expert_cache_policy_key key) const {
        return mapping.count({ key.layer, key.expert }) != 0;
    }

    uint32_t mapped_slot(llm_expert_cache_policy_key key) const {
        const auto found = mapping.find({ key.layer, key.expert });
        if (found == mapping.end()) throw replay_error("missing mapping");
        return found->second;
    }

    void hit(llm_expert_cache_policy_key key) {
        const uint32_t slot = mapped_slot(key);
        require_ready(policy.hit(slot, slots[slot].generation), "policy hit failed");
    }

    bool remove(llm_expert_cache_policy_key key) {
        const auto found = mapping.find({ key.layer, key.expert });
        if (found == mapping.end()) return false;
        const uint32_t slot = found->second;
        require_ready(policy.evict(slot, slots[slot].generation), "forced inclusive eviction failed");
        mapping.erase(found);
        slots[slot].key = { -1, -1 };
        slots[slot].logical_bytes = 0;
        slots[slot].loading = false;
        slots[slot].ready = false;
        evictions++;
        return true;
    }

    std::vector<llm_expert_cache_policy_candidate> candidates(uint64_t incoming_logical) const {
        std::vector<llm_expert_cache_policy_candidate> result;
        result.reserve(slots.size());
        for (uint32_t index = 0; index < slots.size(); ++index) {
            const auto & slot = slots[index];
            result.push_back({ index, slot.generation, slot.key,
                slot.ready ? slot.logical_bytes : incoming_logical,
                footprint, !slot.loading && !slot.ready, !slot.loading && slot.ready });
        }
        return result;
    }

    std::pair<bool, llm_expert_cache_policy_key> admit(
            llm_expert_cache_policy_key key, uint64_t logical, const std::string & admission_class) {
        auto snapshot = candidates(logical);
        llm_expert_cache_policy_decision decision;
        llm_expert_cache_policy_result selected;
        if (admission_class == "MANDATORY_CURRENT_OUTPUT") {
            selected = policy.select(key, snapshot.data(), snapshot.size(), decision);
        } else {
            const auto admission = admission_class == "OPTIONAL_CPU_SERVED" ?
                llm_expert_cache_policy_admission::optional_cpu_served :
                llm_expert_cache_policy_admission::optional_background;
            selected = policy.optional_admission(key, admission, snapshot.data(), snapshot.size(), decision);
        }
        require_ready(selected, "policy selection failed");
        if (!decision.accept) {
            optional_rejections++;
            return { false, { -1, -1 } };
        }
        require(decision.slot < slots.size(), "policy returned invalid slot");
        auto & slot = slots[decision.slot];
        llm_expert_cache_policy_key victim = { -1, -1 };
        if (!decision.free) {
            require(slot.ready && slot.generation == decision.generation, "stale policy victim");
            victim = slot.key;
            require_ready(policy.evict(decision.slot, slot.generation), "policy eviction failed");
            mapping.erase({ victim.layer, victim.expert });
            evictions++;
        }
        require(slot.generation != UINT64_MAX, "generation exhausted");
        slot.generation++;
        slot.key = key;
        slot.logical_bytes = logical;
        slot.loading = true;
        slot.ready = false;
        require_ready(policy.load_begin(decision.slot, slot.generation, key, logical, footprint,
            admission_class != "OPTIONAL_BACKGROUND"), "load begin failed");
        require_ready(policy.load_complete(decision.slot, slot.generation), "load complete failed");
        slot.loading = false;
        slot.ready = true;
        mapping[{ key.layer, key.expert }] = decision.slot;
        admissions++;
        return { true, victim };
    }

    json output() const {
        json events = json::array();
        uint64_t demands = 0, hits = 0, victims = 0, optional_accepts = 0;
        for (size_t index = 0; index < policy.transcript_size(); ++index) {
            const auto & event = policy.transcript()[index];
            events.push_back(event_json(event));
            demands += event.type == llm_expert_cache_policy_event_type::demand;
            hits += event.type == llm_expert_cache_policy_event_type::hit;
            victims += event.type == llm_expert_cache_policy_event_type::victim_selected;
            optional_accepts += event.type == llm_expert_cache_policy_event_type::optional_admission && event.decision == 1;
        }
        json domains = json::array();
        for (const auto & domain : policy.domain_diagnostics()) {
            domains.push_back({ {"domain", domain.domain}, {"layer", domain.layer},
                {"slot_count", domain.slot_count}, {"quota_bytes", domain.quota_bytes},
                {"occupancy_bytes", domain.occupancy_bytes},
                {"protected_capacity_bytes", domain.protected_capacity_bytes},
                {"protected_occupancy_bytes", domain.protected_occupancy_bytes} });
        }
        json resident = json::array();
        for (uint32_t index = 0; index < slots.size(); ++index) {
            if (!slots[index].ready) continue;
            resident.push_back({ {"slot", index}, {"layer", slots[index].key.layer},
                {"expert", slots[index].key.expert}, {"generation", slots[index].generation} });
        }
        return {
            {"config_digest", policy.diagnostics().config.digest},
            {"initial_digest", initial_digest},
            {"final_digest", policy.diagnostics().state_digest},
            {"events", events},
            {"counters", {
                {"events", policy.transcript_size()}, {"demands", demands}, {"hits", hits},
                {"victims", victims}, {"admissions", admissions}, {"evictions", evictions},
                {"optional_accepts", optional_accepts}, {"optional_rejections", optional_rejections}
            }},
            {"domains", domains}, {"resident", resident},
        };
    }
};

json replay(const json & input) {
    require_fields(input, { "schema_version", "topology", "hot", "cold", "requests" }, "replay input");
    require(input.at("schema_version") == "cache-policy-replay-input-v1", "unsupported replay schema");
    const auto & topology = input.at("topology");
    require_fields(topology, { "routed_layers", "experts_per_layer", "physical_slot_footprint_bytes" }, "topology");
    const auto layers = topology.at("routed_layers").get<std::vector<int32_t>>();
    require(!layers.empty() && std::is_sorted(layers.begin(), layers.end()) &&
        std::adjacent_find(layers.begin(), layers.end()) == layers.end(), "noncanonical routed layers");
    const uint32_t experts_per_layer = topology.at("experts_per_layer");
    const uint64_t footprint = topology.at("physical_slot_footprint_bytes");
    require(experts_per_layer != 0 && footprint != 0, "invalid topology");
    tier_mechanism hot(input.at("hot"), llm_expert_cache_policy_tier::hot,
        layers, experts_per_layer, footprint);
    tier_mechanism cold(input.at("cold"), llm_expert_cache_policy_tier::cold,
        layers, experts_per_layer, footprint);
    uint64_t expected_request = 0;
    uint64_t logical_requests = 0, hot_hits = 0, cold_hits = 0, backing_hits = 0;
    uint64_t hot_bytes = 0, cold_bytes = 0, backing_bytes = 0;
    for (const auto & request : input.at("requests")) {
        require_fields(request, { "request_ordinal", "checkpoints", "outcome" }, "request");
        require(request.at("request_ordinal") == ++expected_request, "noncontiguous request ordinal");
        require_ready(hot.policy.request_begin(), "hot request begin failed");
        require_ready(cold.policy.request_begin(), "cold request begin failed");
        uint64_t previous_ubatch = 0;
        uint64_t previous_checkpoint = 0;
        auto current_phase = llm_expert_cache_policy_phase::prefill;
        for (const auto & checkpoint : request.at("checkpoints")) {
            require_fields(checkpoint, { "checkpoint_ordinal", "ubatch_ordinal", "phase", "demands" }, "checkpoint");
            require(checkpoint.at("checkpoint_ordinal") == ++previous_checkpoint,
                "noncontiguous checkpoint ordinal");
            const uint64_t ubatch = checkpoint.at("ubatch_ordinal");
            require(ubatch >= previous_ubatch, "noncanonical ubatch ordinal");
            if (ubatch != previous_ubatch) {
                require_ready(hot.policy.set_ubatch_ordinal(ubatch), "hot ubatch failed");
                require_ready(cold.policy.set_ubatch_ordinal(ubatch), "cold ubatch failed");
                previous_ubatch = ubatch;
            }
            const std::string phase = checkpoint.at("phase");
            require(phase == "PREFILL" || phase == "DECODE", "unknown phase");
            const auto requested_phase = phase == "PREFILL" ?
                llm_expert_cache_policy_phase::prefill : llm_expert_cache_policy_phase::decode;
            if (current_phase != requested_phase) {
                require_ready(hot.policy.phase_transition(requested_phase), "hot phase transition failed");
                require_ready(cold.policy.phase_transition(requested_phase), "cold phase transition failed");
                current_phase = requested_phase;
            }
            llm_expert_cache_policy_key previous = { -1, -1 };
            for (const auto & demand : checkpoint.at("demands")) {
                require_fields(demand, { "layer", "expert", "occurrence_count", "logical_payload_bytes", "hot_admission" }, "demand");
                const llm_expert_cache_policy_key key = { demand.at("layer"), demand.at("expert") };
                require(previous.layer < 0 || key_less(previous, key), "duplicate or noncanonical demand order");
                previous = key;
                const uint64_t logical = demand.at("logical_payload_bytes");
                const uint64_t occurrence = demand.at("occurrence_count");
                require(logical != 0 && occurrence != 0, "invalid demand payload");
                const std::string admission_class = demand.at("hot_admission");
                require(admission_class == "MANDATORY_CURRENT_OUTPUT" || admission_class == "OPTIONAL_CPU_SERVED" ||
                    admission_class == "OPTIONAL_BACKGROUND", "invalid admission class");
                require_ready(hot.policy.demand(key, occurrence, logical, footprint), "hot demand failed");
                logical_requests++;
                if (hot.contains(key)) {
                    hot.hit(key);
                    require_ready(cold.policy.demand(key, occurrence, logical, footprint),
                        "cold shadow demand failed");
                    cold.hit(key);
                    hot_hits++; hot_bytes += logical;
                    continue;
                }
                require_ready(cold.policy.demand(key, occurrence, logical, footprint), "cold demand failed");
                if (cold.contains(key)) {
                    cold.hit(key); cold_hits++; cold_bytes += logical;
                } else {
                    auto admitted = cold.admit(key, logical, "MANDATORY_CURRENT_OUTPUT");
                    require(admitted.first, "mandatory cold admission rejected");
                    if (admitted.second.layer >= 0) hot.remove(admitted.second);
                    backing_hits++; backing_bytes += logical;
                }
                auto admitted = hot.admit(key, logical, admission_class);
                require(admission_class != "MANDATORY_CURRENT_OUTPUT" || admitted.first,
                    "mandatory hot admission rejected");
            }
        }
        const std::string outcome = request.at("outcome");
        require(outcome == "SUCCESS" || outcome == "FAILURE" || outcome == "CANCELLED", "invalid outcome");
        const bool success = outcome == "SUCCESS";
        const bool cancelled = outcome == "CANCELLED";
        require_ready(hot.policy.request_end(success, cancelled), "hot request end failed");
        require_ready(cold.policy.request_end(success, cancelled), "cold request end failed");
    }
    return {
        {"schema_version", "cache-policy-replay-v1"}, {"status", "pass"},
        {"input_sha256", canonical_input_sha256(input)},
        {"tiers", { {"hot", hot.output()}, {"cold", cold.output()} }},
        {"summary", {
            {"logical_requests", logical_requests}, {"hot_hits", hot_hits}, {"cold_hits", cold_hits},
            {"backing_store_hits", backing_hits}, {"hot_bytes", hot_bytes}, {"cold_bytes", cold_bytes},
            {"backing_store_bytes", backing_bytes}
        }},
    };
}

json verify_capture_tier(const json & capture, const std::string & tier_name) {
    require(tier_name == "hot" || tier_name == "cold", "invalid capture tier");
    require(capture.at("schema_version") == "phase9-online-policy-capture-v1", "unsupported online capture");
    const auto & tier_value = capture.at(tier_name);
    const auto & expected = tier_value.at("events");
    if (tier_name == "cold" && capture.at("mode") != "cold") {
        require(expected.empty(), "inactive cold tier emitted events");
        return { {"status", "pass"}, {"tier", tier_name}, {"events", 0}, {"inactive", true} };
    }
    json supplied_config = tier_value.at("config");
    const uint64_t supplied_digest = supplied_config.at("digest");
    supplied_config.erase("digest");
    const auto & topology = capture.at("topology");
    const auto layers = topology.at("routed_layers").get<std::vector<int32_t>>();
    const uint32_t experts = topology.at("experts_per_layer");
    const uint64_t footprint = topology.at(tier_name + "_physical_slot_footprint_bytes");
    const uint32_t slot_count = capture.at("capacities").at(tier_name + "_effective_slots");
    require(!layers.empty() && experts != 0 && footprint != 0 && slot_count != 0, "capture topology is incomplete");
    const auto tier = tier_name == "hot" ? llm_expert_cache_policy_tier::hot : llm_expert_cache_policy_tier::cold;
    json tier_config = { {"slots", slot_count}, {"config", supplied_config} };
    tier_mechanism mechanism(tier_config, tier, layers, experts, footprint);
    require(mechanism.policy.diagnostics().config.digest == supplied_digest, "captured config digest mismatch");
    std::vector<int8_t> pending_load_demand_caused(slot_count, -1);
    std::set<std::pair<uint32_t, uint64_t>> live_loads;
    uint64_t logical_bundle_bytes = 0;
    for (const auto & event : expected) {
        if (event.at("logical_payload_bytes").get<uint64_t>() != 0) {
            logical_bundle_bytes = event.at("logical_payload_bytes");
            break;
        }
    }
    require(logical_bundle_bytes != 0, "capture logical bundle size is missing");
    size_t index = 0;
    uint64_t current_ubatch = 0;
    const auto accepted_slot_will_load = [&](size_t start, uint32_t slot) {
        for (size_t future = start; future < expected.size(); ++future) {
            if (expected.at(future).at("slot") != int64_t(slot)) continue;
            const std::string future_type = expected.at(future).at("type");
            if (future_type == "LOAD_BEGIN") return true;
            if (future_type == "VICTIM_SELECTED") return false;
        }
        return false;
    };
    while (index < expected.size()) {
        const auto & event = expected.at(index);
        require(event.at("tier") == (tier_name == "hot" ? "HOT" : "COLD"), "captured event tier mismatch");
        const std::string type = event.at("type");
        if (type != "REQUEST_BEGIN") {
            const uint64_t ubatch = event.at("ubatch_ordinal");
            if (ubatch > current_ubatch) {
                require_ready(mechanism.policy.set_ubatch_ordinal(ubatch), "captured ubatch failed");
                current_ubatch = ubatch;
            }
        }
        const size_t before = mechanism.policy.transcript_size();
        const llm_expert_cache_policy_key key = { event.at("layer"), event.at("expert") };
        if (type == "REQUEST_BEGIN") {
            require_ready(mechanism.policy.request_begin(), "captured request begin failed");
            current_ubatch = 0;
        } else if (type == "PHASE_TRANSITION") {
            const auto phase = event.at("phase") == "PREFILL" ?
                llm_expert_cache_policy_phase::prefill : llm_expert_cache_policy_phase::decode;
            require_ready(mechanism.policy.phase_transition(phase), "captured phase transition failed");
        } else if (type == "DEMAND") {
            require_ready(mechanism.policy.demand(key, event.at("occurrence_count"),
                event.at("logical_payload_bytes"), event.at("physical_slot_footprint_bytes")),
                "captured demand failed");
        } else if (type == "HIT") {
            require_ready(mechanism.policy.hit(event.at("slot"), event.at("generation")), "captured hit failed");
        } else if (type == "VICTIM_SELECTED") {
            auto candidates = mechanism.candidates(logical_bundle_bytes);
            llm_expert_cache_policy_decision decision;
            const bool optional = index + 1 < expected.size() && expected.at(index + 1).at("type") == "OPTIONAL_ADMISSION";
            if (optional) {
                const bool mandatory = expected.at(index + 1).at("decision") == 2;
                require_ready(mechanism.policy.optional_admission(key,
                    mandatory ? llm_expert_cache_policy_admission::mandatory_current_output :
                        llm_expert_cache_policy_admission::optional_background,
                    candidates.data(), candidates.size(), decision),
                    "captured optional selection failed at " + std::to_string(index));
                if (decision.accept) {
                    pending_load_demand_caused[decision.slot] = mandatory ? 1 : 0;
                    if (accepted_slot_will_load(index + 2, decision.slot)) {
                        mechanism.slots[decision.slot].loading = true;
                    }
                }
            } else {
                require_ready(mechanism.policy.select(key, candidates.data(), candidates.size(), decision),
                    "captured victim selection failed");
                pending_load_demand_caused[decision.slot] = 1;
                if (accepted_slot_will_load(index + 1, decision.slot)) {
                    mechanism.slots[decision.slot].loading = true;
                }
            }
        } else if (type == "OPTIONAL_ADMISSION") {
            auto candidates = mechanism.candidates(logical_bundle_bytes);
            llm_expert_cache_policy_decision decision;
            const bool mandatory = event.at("decision") == 2;
            const auto admitted = mechanism.policy.optional_admission(key,
                mandatory ? llm_expert_cache_policy_admission::mandatory_current_output :
                    llm_expert_cache_policy_admission::optional_background,
                candidates.data(), candidates.size(), decision);
            if (!admitted.is_ready()) {
                json snapshot = json::array();
                for (const auto & candidate : candidates) snapshot.push_back({
                    {"slot", candidate.slot}, {"generation", candidate.generation},
                    {"layer", candidate.key.layer}, {"expert", candidate.key.expert},
                    {"logical", candidate.logical_bundle_bytes}, {"free", candidate.free},
                    {"eligible", candidate.eligible},
                });
                throw replay_error("captured optional admission failed at " + std::to_string(index) +
                    " (policy error " + std::to_string(unsigned(admitted.error)) + "): " + snapshot.dump());
            }
            if (decision.accept) {
                pending_load_demand_caused[decision.slot] = mandatory ? 1 : 0;
                if (accepted_slot_will_load(index + 1, decision.slot)) {
                    mechanism.slots[decision.slot].loading = true;
                }
            }
        } else if (type == "EVICT") {
            const uint32_t slot = event.at("slot");
            require_ready(mechanism.policy.evict(slot, event.at("generation")), "captured eviction failed");
            mechanism.mapping.erase({ mechanism.slots[slot].key.layer, mechanism.slots[slot].key.expert });
            mechanism.slots[slot].key = { -1, -1 };
            mechanism.slots[slot].logical_bytes = 0;
            mechanism.slots[slot].loading = false;
            mechanism.slots[slot].ready = false;
        } else if (type == "LOAD_BEGIN") {
            const uint32_t slot = event.at("slot");
            mechanism.slots[slot].generation = event.at("generation");
            mechanism.slots[slot].key = key;
            mechanism.slots[slot].logical_bytes = event.at("logical_payload_bytes");
            mechanism.slots[slot].loading = true;
            mechanism.slots[slot].ready = false;
            const bool demand_caused = pending_load_demand_caused[slot] < 0 ? true :
                pending_load_demand_caused[slot] != 0;
            pending_load_demand_caused[slot] = -1;
            require_ready(mechanism.policy.load_begin(slot, event.at("generation"), key,
                event.at("logical_payload_bytes"), footprint, demand_caused), "captured load begin failed");
        } else if (type == "LOAD_COMPLETE") {
            require_ready(mechanism.policy.load_complete(event.at("slot"), event.at("generation")),
                "captured load complete failed");
        } else if (type == "LOAD_FAILED") {
            require_ready(mechanism.policy.load_failed(event.at("slot"), event.at("generation")),
                "captured load failed event failed");
        } else if (type == "PIN") {
            require_ready(mechanism.policy.pin(event.at("slot"), event.at("generation")), "captured pin failed");
        } else if (type == "UNPIN") {
            require_ready(mechanism.policy.unpin(event.at("slot"), event.at("generation")), "captured unpin failed");
        } else if (type == "REQUEST_END") {
            const uint8_t decision = event.at("decision");
            require_ready(mechanism.policy.request_end(decision == 1, decision == 2), "captured request end failed");
        } else {
            throw replay_error("unsupported captured event: " + type);
        }
        const size_t after = mechanism.policy.transcript_size();
        require(after > before, "captured operation emitted no event");
        require(index + after - before <= expected.size(), "captured operation emitted excess events");
        for (size_t offset = 0; offset < after - before; ++offset) {
            const auto produced = event_json(mechanism.policy.transcript()[before + offset]);
            if (produced != expected.at(index + offset)) {
                throw replay_error("captured event mismatch at " + std::to_string(index + offset) +
                    ": expected=" + expected.at(index + offset).dump() +
                    " produced=" + produced.dump());
            }
            const auto & emitted = mechanism.policy.transcript()[before + offset];
            const auto load_identity = std::make_pair(emitted.slot, emitted.generation);
            if (emitted.type == llm_expert_cache_policy_event_type::load_begin) {
                require(live_loads.insert(load_identity).second, "duplicate captured live load");
            } else if (emitted.type == llm_expert_cache_policy_event_type::load_complete ||
                    emitted.type == llm_expert_cache_policy_event_type::load_failed) {
                require(live_loads.erase(load_identity) == 1, "captured terminal has no live load");
            }
            if (emitted.type == llm_expert_cache_policy_event_type::load_complete) {
                auto & slot = mechanism.slots[emitted.slot];
                slot.loading = false;
                slot.ready = true;
                mechanism.mapping[{ slot.key.layer, slot.key.expert }] = emitted.slot;
            } else if (emitted.type == llm_expert_cache_policy_event_type::load_failed) {
                auto & slot = mechanism.slots[emitted.slot];
                slot.key = { -1, -1 };
                slot.logical_bytes = 0;
                slot.loading = false;
                slot.ready = false;
            }
        }
        index += after - before;
    }
    require(!expected.empty() && expected.back().at("type") == "REQUEST_END" && live_loads.empty(),
        "capture is a partial request transcript");
    const auto & diagnostics = tier_value.at("diagnostics");
    require(mechanism.policy.diagnostics().state_digest == diagnostics.at("state_digest"),
        "captured final digest mismatch");
    require(mechanism.policy.transcript_size() == diagnostics.at("events"), "captured final event count mismatch");
    return {
        {"status", "pass"}, {"tier", tier_name}, {"events", expected.size()},
        {"config_digest", supplied_digest}, {"final_digest", mechanism.policy.diagnostics().state_digest},
    };
}

json verify_capture(const json & capture) {
    return {
        {"schema_version", "phase9-online-policy-native-verification-v1"}, {"status", "pass"},
        {"hot", verify_capture_tier(capture, "hot")}, {"cold", verify_capture_tier(capture, "cold")},
        {"output_identity", {
            {"prompt_ids", capture.at("prompt_ids")}, {"generated_ids", capture.at("generated_ids")},
            {"logits_fnv64", capture.at("logits_fnv64")},
        }},
    };
}

json base_config(const char * policy = "LRU") {
    return { {"schema_version", "cache-policy-config-v1"}, {"policy", policy}, {"scope", "GLOBAL"},
        {"slru_protected_ratio_bps", std::string(policy) == "SLRU" ? 7500 : 0},
        {"admission", "ALWAYS"}, {"admission_window_events", 0},
        {"lfu_aging_interval_events", std::string(policy) == "LFU_AGING" ? 64 : 0} };
}

int self_test() {
    const json input = {
        {"schema_version", "cache-policy-replay-input-v1"},
        {"topology", { {"routed_layers", {0}}, {"experts_per_layer", 4},
            {"physical_slot_footprint_bytes", 128} }},
        {"hot", { {"slots", 2}, {"config", base_config()} }},
        {"cold", { {"slots", 3}, {"config", base_config()} }},
        {"requests", json::array({ {
            {"request_ordinal", 1}, {"outcome", "SUCCESS"},
            {"checkpoints", json::array({ {
                {"ubatch_ordinal", 1}, {"phase", "DECODE"},
                {"checkpoint_ordinal", 1},
                {"demands", json::array({
                    {{"layer",0},{"expert",0},{"occurrence_count",1},{"logical_payload_bytes",64},{"hot_admission","MANDATORY_CURRENT_OUTPUT"}},
                    {{"layer",0},{"expert",1},{"occurrence_count",1},{"logical_payload_bytes",64},{"hot_admission","MANDATORY_CURRENT_OUTPUT"}}
                })}
            } })}
        } })}
    };
    const auto output = replay(input);
    require(output.at("summary").at("backing_store_hits") == 2, "self-test backing count mismatch");
    std::puts(output.dump().c_str());
    return 0;
}

json benchmark_policy_cpu() {
    json rows = json::array();
    const std::vector<int32_t> layers = { 0, 1, 2, 3, 4, 5, 6 };
    for (const char * name : { "LRU", "LFRU", "SLRU", "LFU_AGING" }) {
        const json value = { {"slots", 64}, {"config", base_config(name)} };
        tier_mechanism mechanism(value, llm_expert_cache_policy_tier::cold, layers, 8, 786432);
        const uint64_t administration = mechanism.policy.diagnostics().administration_actual_bytes;
        uint64_t demands = 0;
        uint64_t elapsed = 0;
        const auto measured = [&](auto && operation) {
            const auto begin = std::chrono::steady_clock::now();
            const auto result = operation();
            elapsed += std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - begin).count();
            return result;
        };
        for (uint64_t request = 0; request < 1000; ++request) {
            require_ready(measured([&] { return mechanism.policy.request_begin(); }), "benchmark request begin failed");
            require_ready(measured([&] { return mechanism.policy.phase_transition(llm_expert_cache_policy_phase::decode); }),
                "benchmark phase transition failed");
            for (int32_t layer : layers) {
                for (int32_t lane = 0; lane < 2; ++lane) {
                    const llm_expert_cache_policy_key key = { layer, int32_t((request + lane + layer*3) % 8) };
                    require_ready(measured([&] { return mechanism.policy.demand(key, 1, 786432, 786432); }),
                        "benchmark demand failed");
                    demands++;
                    if (mechanism.contains(key)) {
                        const uint32_t slot = mechanism.mapped_slot(key);
                        require_ready(measured([&] { return mechanism.policy.hit(slot, mechanism.slots[slot].generation); }),
                            "benchmark hit failed");
                        continue;
                    }
                    auto candidates = mechanism.candidates(786432);
                    llm_expert_cache_policy_decision decision;
                    require_ready(measured([&] { return mechanism.policy.select(key, candidates.data(), candidates.size(), decision); }),
                        "benchmark select failed");
                    auto & slot = mechanism.slots[decision.slot];
                    if (slot.ready) {
                        require_ready(measured([&] { return mechanism.policy.evict(decision.slot, slot.generation); }),
                            "benchmark evict failed");
                        mechanism.mapping.erase({ slot.key.layer, slot.key.expert });
                    }
                    slot = { key, slot.generation + 1, 786432, true, false };
                    require_ready(measured([&] { return mechanism.policy.load_begin(
                        decision.slot, slot.generation, key, 786432, 786432); }),
                        "benchmark load begin failed");
                    require_ready(measured([&] { return mechanism.policy.load_complete(decision.slot, slot.generation); }),
                        "benchmark load complete failed");
                    slot.loading = false;
                    slot.ready = true;
                    mechanism.mapping[{ key.layer, key.expert }] = decision.slot;
                }
            }
            require_ready(measured([&] { return mechanism.policy.request_end(true, false); }),
                "benchmark request end failed");
        }
        const auto diagnostics = mechanism.policy.diagnostics();
        require(diagnostics.administration_actual_bytes == administration,
            "benchmark policy administration changed after initialization");
        rows.push_back({ {"policy", name}, {"requests", 1000}, {"demands", demands},
            {"elapsed_ns", elapsed}, {"ns_per_demand", double(elapsed)/demands},
            {"administration_bytes", administration}, {"steady_state_policy_allocations", 0},
            {"final_digest", diagnostics.state_digest} });
    }
    return { {"schema_version", "phase9-policy-cpu-benchmark-v1"}, {"status", "pass"},
        {"clock", "steady_clock"}, {"rows", rows} };
}

} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--self-test") return self_test();
        if (argc == 3 && std::string(argv[1]) == "--benchmark-output") {
            std::ofstream destination(argv[2]);
            if (!destination) throw replay_error("cannot open benchmark output");
            destination << benchmark_policy_cpu().dump(2) << '\n';
            return destination ? 0 : 1;
        }
        if (argc == 5 && std::string(argv[1]) == "--input" && std::string(argv[3]) == "--output") {
            std::ifstream source(argv[2]);
            if (!source) throw replay_error("cannot open replay input");
            json input;
            source >> input;
            const json output = replay(input);
            std::ofstream destination(argv[4]);
            if (!destination) throw replay_error("cannot open replay output");
            destination << output.dump(2) << '\n';
            return destination ? 0 : 1;
        }
        if (argc == 5 && std::string(argv[1]) == "--capture-input" && std::string(argv[3]) == "--output") {
            std::ifstream source(argv[2]);
            if (!source) throw replay_error("cannot open online capture input");
            json input;
            source >> input;
            const json output = verify_capture(input);
            std::ofstream destination(argv[4]);
            if (!destination) throw replay_error("cannot open capture verification output");
            destination << output.dump(2) << '\n';
            return destination ? 0 : 1;
        }
        std::fprintf(stderr, "usage: phase9-cache-replay --self-test | --benchmark-output OUTPUT | --input INPUT --output OUTPUT | --capture-input INPUT --output OUTPUT\n");
        return 2;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "phase9-cache-replay: %s\n", error.what());
        return 1;
    }
}
