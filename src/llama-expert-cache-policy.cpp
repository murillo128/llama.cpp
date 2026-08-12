#include "llama-expert-cache-policy.h"

#include <algorithm>
#include <limits>
#include <new>

namespace {

constexpr uint64_t fnv_offset = UINT64_C(1469598103934665603);
constexpr uint64_t fnv_prime = UINT64_C(1099511628211);

void hash_append(uint64_t & hash, uint64_t value) noexcept {
    for (uint32_t byte = 0; byte < 8; ++byte) {
        hash ^= uint8_t(value >> (byte*8));
        hash *= fnv_prime;
    }
}

bool is_power_of_two(uint32_t value) noexcept {
    return value != 0 && (value & (value - 1)) == 0;
}

bool checked_multiply(uint64_t lhs, uint64_t rhs, uint64_t & result) noexcept {
    if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max()/lhs) return false;
    result = lhs*rhs;
    return true;
}

bool checked_add(uint64_t lhs, uint64_t rhs, uint64_t & result) noexcept {
    if (rhs > std::numeric_limits<uint64_t>::max() - lhs) return false;
    result = lhs + rhs;
    return true;
}

struct uint128_product {
    uint64_t high;
    uint64_t low;
};

uint128_product multiply_64(uint64_t lhs, uint64_t rhs) noexcept {
    const uint64_t lhs_low = uint32_t(lhs);
    const uint64_t lhs_high = lhs >> 32;
    const uint64_t rhs_low = uint32_t(rhs);
    const uint64_t rhs_high = rhs >> 32;
    const uint64_t low_low = lhs_low*rhs_low;
    const uint64_t low_high = lhs_low*rhs_high;
    const uint64_t high_low = lhs_high*rhs_low;
    const uint64_t high_high = lhs_high*rhs_high;
    const uint64_t carry = (low_low >> 32) + uint32_t(low_high) + uint32_t(high_low);
    return { high_high + (low_high >> 32) + (high_low >> 32) + (carry >> 32),
             (carry << 32) | uint32_t(low_low) };
}

int compare_products(uint64_t lhs_a, uint64_t lhs_b, uint64_t rhs_a, uint64_t rhs_b) noexcept {
    const auto lhs = multiply_64(lhs_a, lhs_b);
    const auto rhs = multiply_64(rhs_a, rhs_b);
    if (lhs.high != rhs.high) return lhs.high < rhs.high ? -1 : 1;
    if (lhs.low != rhs.low) return lhs.low < rhs.low ? -1 : 1;
    return 0;
}

uint64_t window_contribution(uint64_t position, uint64_t key) noexcept {
    uint64_t hash = fnv_offset;
    hash_append(hash, position);
    hash_append(hash, key);
    return hash;
}

} // namespace

llm_expert_cache_policy_result llm_expert_cache_policy_copy_config(
        const llama_expert_cache_policy_config * source,
        llm_expert_cache_policy_tier tier,
        llm_expert_cache_policy_config_internal & destination) noexcept {
    llama_expert_cache_policy_config value = {
        LLAMA_EXPERT_CACHE_POLICY_VERSION_1,
        sizeof(llama_expert_cache_policy_config),
        LLAMA_EXPERT_CACHE_POLICY_LRU,
        LLAMA_EXPERT_CACHE_POLICY_SCOPE_GLOBAL,
        0,
        LLAMA_EXPERT_CACHE_ADMISSION_ALWAYS,
        0,
        0,
        {},
    };
    if (source != nullptr) value = *source;
    if (value.version != LLAMA_EXPERT_CACHE_POLICY_VERSION_1 ||
        value.struct_size != sizeof(llama_expert_cache_policy_config) ||
        value.policy < LLAMA_EXPERT_CACHE_POLICY_LRU || value.policy >= LLAMA_EXPERT_CACHE_POLICY_COUNT ||
        value.scope < LLAMA_EXPERT_CACHE_POLICY_SCOPE_GLOBAL || value.scope >= LLAMA_EXPERT_CACHE_POLICY_SCOPE_COUNT ||
        value.admission < LLAMA_EXPERT_CACHE_ADMISSION_ALWAYS ||
        value.admission >= LLAMA_EXPERT_CACHE_ADMISSION_COUNT) {
        return llm_expert_cache_policy_result::failure(llm_expert_cache_policy_error::invalid_configuration);
    }
    for (uint64_t reserved : value.reserved) {
        if (reserved != 0) {
            return llm_expert_cache_policy_result::failure(llm_expert_cache_policy_error::invalid_configuration);
        }
    }
    if ((value.policy == LLAMA_EXPERT_CACHE_POLICY_SLRU &&
            (value.slru_protected_ratio_bps < 1000 || value.slru_protected_ratio_bps > 9000)) ||
        (value.policy != LLAMA_EXPERT_CACHE_POLICY_SLRU && value.slru_protected_ratio_bps != 0)) {
        return llm_expert_cache_policy_result::failure(llm_expert_cache_policy_error::invalid_configuration);
    }
    if (value.admission == LLAMA_EXPERT_CACHE_ADMISSION_ALWAYS) {
        if (value.admission_window_events != 0) {
            return llm_expert_cache_policy_result::failure(llm_expert_cache_policy_error::invalid_configuration);
        }
    } else if (tier != llm_expert_cache_policy_tier::hot ||
               value.policy != LLAMA_EXPERT_CACHE_POLICY_SLRU ||
               !is_power_of_two(value.admission_window_events) ||
               value.admission_window_events < 64 || value.admission_window_events > 1048576) {
        return llm_expert_cache_policy_result::failure(llm_expert_cache_policy_error::invalid_configuration);
    }
    if ((value.policy == LLAMA_EXPERT_CACHE_POLICY_LFU_AGING &&
            (!is_power_of_two(value.lfu_aging_interval_events) ||
             value.lfu_aging_interval_events < 64 || value.lfu_aging_interval_events > 1048576)) ||
        (value.policy != LLAMA_EXPERT_CACHE_POLICY_LFU_AGING && value.lfu_aging_interval_events != 0)) {
        return llm_expert_cache_policy_result::failure(llm_expert_cache_policy_error::invalid_configuration);
    }
    uint64_t digest = fnv_offset;
    hash_append(digest, value.version);
    hash_append(digest, value.struct_size);
    hash_append(digest, value.policy);
    hash_append(digest, value.scope);
    hash_append(digest, value.slru_protected_ratio_bps);
    hash_append(digest, value.admission);
    hash_append(digest, value.admission_window_events);
    hash_append(digest, value.lfu_aging_interval_events);
    destination = {
        value.policy,
        value.scope,
        value.slru_protected_ratio_bps,
        value.admission,
        value.admission_window_events,
        value.lfu_aging_interval_events,
        true,
        source != nullptr,
        source != nullptr ? *source : llama_expert_cache_policy_config {},
        value,
        digest,
    };
    return llm_expert_cache_policy_result::success();
}

llm_expert_cache_policy_result llm_expert_cache_policy::initialize(
        const llm_expert_cache_policy_config_internal & requested_config,
        llm_expert_cache_policy_tier requested_tier,
        const int32_t * routed_layers,
        uint32_t routed_layer_count,
        uint32_t requested_experts_per_layer,
        uint32_t minimum_domain_slots,
        uint32_t slot_count,
        uint64_t physical_slot_footprint_bytes,
        uint32_t transcript_capacity,
        const uint64_t * domain_budget_bytes,
        uint32_t domain_budget_count) noexcept {
    if (initialized || routed_layers == nullptr || routed_layer_count == 0 || requested_experts_per_layer == 0 ||
        minimum_domain_slots == 0 || minimum_domain_slots > requested_experts_per_layer ||
        slot_count == 0 || physical_slot_footprint_bytes == 0 || transcript_capacity == 0 ||
        ((domain_budget_bytes == nullptr) != (domain_budget_count == 0)) ||
        requested_config.digest == 0 ||
        (requested_config.scope == LLAMA_EXPERT_CACHE_POLICY_SCOPE_PER_LAYER &&
         uint64_t(slot_count) < uint64_t(routed_layer_count)*minimum_domain_slots)) {
        return llm_expert_cache_policy_result::failure(llm_expert_cache_policy_error::invalid_configuration);
    }
    uint64_t key_count = 0;
    uint64_t budget = 0;
    if (!checked_multiply(routed_layer_count, requested_experts_per_layer, key_count) || key_count > SIZE_MAX ||
        !checked_multiply(slot_count, physical_slot_footprint_bytes, budget)) {
        return llm_expert_cache_policy_result::failure(llm_expert_cache_policy_error::overflow);
    }
    try {
        config = requested_config;
        tier = requested_tier;
        experts_per_layer = requested_experts_per_layer;
        slot_footprint = physical_slot_footprint_bytes;
        layers.assign(routed_layers, routed_layers + routed_layer_count);
        for (uint32_t index = 0; index < routed_layer_count; ++index) {
            if (layers[index] < 0 || (index != 0 && layers[index - 1] >= layers[index])) {
                return llm_expert_cache_policy_result::failure(llm_expert_cache_policy_error::invalid_configuration);
            }
        }
        keys.assign(size_t(key_count), {});
        slots.assign(slot_count, {});
        candidate_seen.assign(slot_count, 0);
        const uint32_t domain_count = config.scope == LLAMA_EXPERT_CACHE_POLICY_SCOPE_GLOBAL ? 1 : routed_layer_count;
        if (domain_budget_bytes != nullptr && domain_budget_count != domain_count) {
            return llm_expert_cache_policy_result::failure(llm_expert_cache_policy_error::invalid_configuration);
        }
        domains.assign(domain_count, {});
        for (uint32_t domain = 0; domain < domain_count; ++domain) {
            domains[domain].domain = domain;
            domains[domain].layer = config.scope == LLAMA_EXPERT_CACHE_POLICY_SCOPE_GLOBAL ?
                -1 : layers[domain];
        }
        for (uint32_t slot = 0; slot < slot_count; ++slot) {
            uint32_t domain = 0;
            if (config.scope == LLAMA_EXPERT_CACHE_POLICY_SCOPE_PER_LAYER) {
                const uint32_t base = slot_count/routed_layer_count;
                const uint32_t remainder = slot_count % routed_layer_count;
                uint32_t boundary = 0;
                for (uint32_t candidate = 0; candidate < routed_layer_count; ++candidate) {
                    boundary += base + (candidate < remainder ? 1 : 0);
                    if (slot < boundary) {
                        domain = candidate;
                        break;
                    }
                }
            }
            slots[slot].domain = domain;
            domains[domain].slot_count++;
            domains[domain].quota_bytes += slot_footprint;
        }
        if (domain_budget_bytes != nullptr) {
            for (uint32_t domain = 0; domain < domain_count; ++domain) {
                if (domain_budget_bytes[domain] == 0) {
                    return llm_expert_cache_policy_result::failure(
                        llm_expert_cache_policy_error::invalid_configuration);
                }
                domains[domain].quota_bytes = domain_budget_bytes[domain];
            }
        }
        for (auto & domain : domains) {
            uint64_t scaled = 0;
            if (!checked_multiply(domain.quota_bytes, config.slru_protected_ratio_bps, scaled)) {
                return llm_expert_cache_policy_result::failure(llm_expert_cache_policy_error::overflow);
            }
            domain.protected_capacity_bytes = (scaled/10000/slot_footprint)*slot_footprint;
        }
        if (config.admission == LLAMA_EXPERT_CACHE_ADMISSION_FREQUENCY_WINDOW) {
            frequency_window.assign(config.admission_window_events, UINT64_MAX);
        } else {
            frequency_window.clear();
        }
        if (config.state_attestation) {
            events.assign(transcript_capacity, {});
        } else {
            events.clear();
            events.shrink_to_fit();
        }
        frequency_window_write = 0;
        frequency_window_size = 0;
        event_write = 0;
        reserved_terminal_events = 0;
        terminal_operation_ordinal = 0;
        frequency_window_state_digest = 0;
        for (size_t index = 0; index < frequency_window.size(); ++index) {
            frequency_window_state_digest ^= window_contribution(index, frequency_window[index]);
        }
        request_active = false;
        phase = llm_expert_cache_policy_phase::prefill;
        counters = {};
        counters.config = config;
        uint64_t requested_admin = 0;
        uint64_t actual_admin = 0;
        const auto add_admin = [](uint64_t count, uint64_t width, uint64_t & total) {
            uint64_t bytes = 0;
            uint64_t next = 0;
            if (!checked_multiply(count, width, bytes) || !checked_add(total, bytes, next)) return false;
            total = next;
            return true;
        };
        if (!add_admin(layers.size(), sizeof(int32_t), requested_admin) ||
            !add_admin(keys.size(), sizeof(key_state), requested_admin) ||
            !add_admin(slots.size(), sizeof(slot_state), requested_admin) ||
            !add_admin(candidate_seen.size(), sizeof(uint8_t), requested_admin) ||
            !add_admin(domains.size(), sizeof(llm_expert_cache_policy_domain_diagnostics), requested_admin) ||
            !add_admin(frequency_window.size(), sizeof(uint64_t), requested_admin) ||
            !add_admin(events.size(), sizeof(llm_expert_cache_policy_event), requested_admin) ||
            !add_admin(layers.capacity(), sizeof(int32_t), actual_admin) ||
            !add_admin(keys.capacity(), sizeof(key_state), actual_admin) ||
            !add_admin(slots.capacity(), sizeof(slot_state), actual_admin) ||
            !add_admin(candidate_seen.capacity(), sizeof(uint8_t), actual_admin) ||
            !add_admin(domains.capacity(), sizeof(llm_expert_cache_policy_domain_diagnostics), actual_admin) ||
            !add_admin(frequency_window.capacity(), sizeof(uint64_t), actual_admin) ||
            !add_admin(events.capacity(), sizeof(llm_expert_cache_policy_event), actual_admin)) {
            return llm_expert_cache_policy_result::failure(llm_expert_cache_policy_error::overflow);
        }
        counters.administration_requested_bytes = requested_admin;
        counters.administration_actual_bytes = actual_admin;
        counters.administration_bytes = actual_admin;
        counters.peak_administration_bytes = counters.administration_bytes;
        refresh_state_digest();
        initialized = true;
        return llm_expert_cache_policy_result::success();
    } catch (const std::bad_alloc &) {
        return llm_expert_cache_policy_result::failure(llm_expert_cache_policy_error::invalid_configuration);
    }
}

int32_t llm_expert_cache_policy::layer_index(int32_t layer) const noexcept {
    const auto found = std::lower_bound(layers.begin(), layers.end(), layer);
    return found == layers.end() || *found != layer ? -1 : int32_t(found - layers.begin());
}

int64_t llm_expert_cache_policy::key_index(llm_expert_cache_policy_key key) const noexcept {
    const int32_t layer = layer_index(key.layer);
    if (layer < 0 || key.expert < 0 || key.expert >= int32_t(experts_per_layer)) return -1;
    return int64_t(layer)*experts_per_layer + key.expert;
}

uint32_t llm_expert_cache_policy::key_domain(llm_expert_cache_policy_key key) const noexcept {
    return config.scope == LLAMA_EXPERT_CACHE_POLICY_SCOPE_GLOBAL ? 0 : uint32_t(layer_index(key.layer));
}

bool llm_expert_cache_policy::key_matches(
        llm_expert_cache_policy_key lhs,
        llm_expert_cache_policy_key rhs) const noexcept {
    return lhs.layer == rhs.layer && lhs.expert == rhs.expert &&
        lhs.layout_class_id == rhs.layout_class_id;
}

llm_expert_cache_policy_result llm_expert_cache_policy::append_event(
        llm_expert_cache_policy_event_type type,
        llm_expert_cache_policy_key key,
        uint64_t occurrence_count,
        uint64_t logical_bundle_bytes,
        uint64_t physical_slot_footprint_bytes,
        uint32_t slot,
        uint64_t generation,
        uint8_t decision,
        uint64_t origin_operation_ordinal,
        bool eligible,
        uint8_t reason) noexcept {
    if (counters.event_sequence == UINT64_MAX) {
        return llm_expert_cache_policy_result::failure(llm_expert_cache_policy_error::sequence_exhausted);
    }
    counters.event_sequence++;
    counters.events++;
    refresh_state_digest();
    if (!config.state_attestation) {
        counters.transcript_records = 0;
        return llm_expert_cache_policy_result::success();
    }
    if (event_write >= events.size()) {
        counters.transcript_dropped++;
        return llm_expert_cache_policy_result::failure(llm_expert_cache_policy_error::transcript_full);
    }
    auto & event = events[event_write++];
    event = { 1, counters.request_ordinal, counters.ubatch_ordinal, tier, type,
        counters.event_sequence, counters.demand_ordinal,
        origin_operation_ordinal, phase, key, occurrence_count, logical_bundle_bytes,
        physical_slot_footprint_bytes, slot, generation,
        key_index(key) < 0 ? UINT32_MAX : key_domain(key), eligible, decision, reason,
        counters.state_digest };
    counters.transcript_records = event_write;
    return llm_expert_cache_policy_result::success();
}

llm_expert_cache_policy_result llm_expert_cache_policy::preflight_events(size_t count) const noexcept {
    if (config.state_attestation) {
        const size_t used = std::min(event_write, events.size());
        if (reserved_terminal_events > events.size() - used ||
            count > events.size() - used - reserved_terminal_events) {
            return llm_expert_cache_policy_result::failure(llm_expert_cache_policy_error::transcript_full);
        }
    }
    if (reserved_terminal_events > UINT64_MAX - counters.event_sequence ||
        count > UINT64_MAX - counters.event_sequence - reserved_terminal_events) {
        return llm_expert_cache_policy_result::failure(llm_expert_cache_policy_error::sequence_exhausted);
    }
    return llm_expert_cache_policy_result::success();
}

llm_expert_cache_policy_result llm_expert_cache_policy::validate_event_capacity(size_t count) const noexcept {
    return preflight_events(count);
}

llm_expert_cache_policy_result llm_expert_cache_policy::request_begin() noexcept {
    if (!initialized || request_active || counters.request_ordinal == UINT64_MAX) {
        return llm_expert_cache_policy_result::failure(
            counters.request_ordinal == UINT64_MAX ? llm_expert_cache_policy_error::sequence_exhausted :
            llm_expert_cache_policy_error::invalid_event);
    }
    const auto capacity = preflight_events();
    if (!capacity.is_ready()) return capacity;
    counters.request_ordinal++;
    counters.ubatch_ordinal = 0;
    request_active = true;
    phase = llm_expert_cache_policy_phase::prefill;
    return append_event(llm_expert_cache_policy_event_type::request_begin);
}

llm_expert_cache_policy_result llm_expert_cache_policy::set_ubatch_ordinal(uint64_t ordinal) noexcept {
    if (!request_active || ordinal == 0 || ordinal <= counters.ubatch_ordinal) {
        return llm_expert_cache_policy_result::failure(llm_expert_cache_policy_error::invalid_event);
    }
    counters.ubatch_ordinal = ordinal;
    refresh_state_digest();
    return llm_expert_cache_policy_result::success();
}

llm_expert_cache_policy_result llm_expert_cache_policy::phase_transition(
        llm_expert_cache_policy_phase requested_phase) noexcept {
    if (!request_active || requested_phase == phase) {
        return llm_expert_cache_policy_result::failure(llm_expert_cache_policy_error::invalid_event);
    }
    const auto capacity = preflight_events();
    if (!capacity.is_ready()) return capacity;
    phase = requested_phase;
    return append_event(llm_expert_cache_policy_event_type::phase_transition);
}

bool llm_expert_cache_policy::normalize_aging(
        slot_state & slot, uint64_t current_demand_ordinal) noexcept {
    const uint64_t epoch = current_demand_ordinal/config.lfu_aging_interval_events;
    if (epoch < slot.aging_epoch) return false;
    const uint64_t elapsed = std::min<uint64_t>(63, epoch - slot.aging_epoch);
    if (elapsed != 0 && slot.resident_frequency != 0) {
        slot.resident_frequency = std::max<uint64_t>(1, slot.resident_frequency >> elapsed);
    }
    slot.aging_epoch = epoch;
    return true;
}

bool llm_expert_cache_policy::candidate_precedes(uint32_t lhs_slot, uint32_t rhs_slot) const noexcept {
    const auto & lhs = slots[lhs_slot];
    const auto & rhs = slots[rhs_slot];
    bool precedes = false;
    bool equivalent = false;
    if (config.policy == LLAMA_EXPERT_CACHE_POLICY_LRU) {
        precedes = lhs.last_touch_sequence < rhs.last_touch_sequence;
        equivalent = lhs.last_touch_sequence == rhs.last_touch_sequence;
    } else if (config.policy == LLAMA_EXPERT_CACHE_POLICY_LFRU) {
        const uint64_t lhs_age = counters.demand_ordinal - lhs.last_touch_demand + 1;
        const uint64_t rhs_age = counters.demand_ordinal - rhs.last_touch_demand + 1;
        const int comparison = compare_products(lhs.resident_frequency, rhs_age,
            rhs.resident_frequency, lhs_age);
        precedes = comparison < 0 ||
            (comparison == 0 && lhs.last_touch_sequence < rhs.last_touch_sequence);
        equivalent = comparison == 0 && lhs.last_touch_sequence == rhs.last_touch_sequence;
    } else if (config.policy == LLAMA_EXPERT_CACHE_POLICY_SLRU) {
        precedes = (lhs.current_segment == segment::probationary &&
                rhs.current_segment == segment::protected_segment) ||
            (lhs.current_segment == rhs.current_segment &&
             lhs.last_touch_sequence < rhs.last_touch_sequence);
        equivalent = lhs.current_segment == rhs.current_segment &&
            lhs.last_touch_sequence == rhs.last_touch_sequence;
    } else {
        const uint64_t epoch = counters.demand_ordinal/config.lfu_aging_interval_events;
        const auto effective_frequency = [&](const slot_state & slot) {
            const uint64_t elapsed = std::min<uint64_t>(63, epoch - slot.aging_epoch);
            return elapsed == 0 || slot.resident_frequency == 0 ? slot.resident_frequency :
                std::max<uint64_t>(1, slot.resident_frequency >> elapsed);
        };
        const uint64_t lhs_frequency = effective_frequency(lhs);
        const uint64_t rhs_frequency = effective_frequency(rhs);
        precedes = lhs_frequency < rhs_frequency ||
            (lhs_frequency == rhs_frequency &&
             lhs.last_touch_sequence < rhs.last_touch_sequence);
        equivalent = lhs_frequency == rhs_frequency &&
            lhs.last_touch_sequence == rhs.last_touch_sequence;
    }
    return precedes || (equivalent && lhs_slot < rhs_slot);
}

llm_expert_cache_policy_result llm_expert_cache_policy::demand(
        llm_expert_cache_policy_key key,
        uint64_t occurrence_count,
        uint64_t logical_bundle_bytes,
        uint64_t physical_slot_footprint_bytes) noexcept {
    const int64_t index = key_index(key);
    if (!request_active || index < 0 || occurrence_count == 0 || logical_bundle_bytes == 0 ||
        physical_slot_footprint_bytes == 0) {
        return llm_expert_cache_policy_result::failure(llm_expert_cache_policy_error::invalid_key);
    }
    if (counters.demand_ordinal == UINT64_MAX) {
        return llm_expert_cache_policy_result::failure(llm_expert_cache_policy_error::sequence_exhausted);
    }
    const auto capacity = preflight_events();
    if (!capacity.is_ready()) return capacity;
    counters.demand_ordinal++;
    counters.demands++;
    auto & state = keys[size_t(index)];
    if (!frequency_window.empty()) {
        if (frequency_window_size == frequency_window.size()) {
            const uint64_t outgoing = frequency_window[frequency_window_write];
            if (outgoing >= keys.size() || keys[outgoing].window_frequency == 0) {
                return llm_expert_cache_policy_result::failure(llm_expert_cache_policy_error::metadata_mismatch);
            }
            keys[outgoing].window_frequency--;
        } else {
            frequency_window_size++;
        }
        frequency_window_state_digest ^=
            window_contribution(frequency_window_write, frequency_window[frequency_window_write]);
        frequency_window[frequency_window_write] = uint64_t(index);
        frequency_window_state_digest ^=
            window_contribution(frequency_window_write, frequency_window[frequency_window_write]);
        frequency_window_write = (frequency_window_write + 1) & (frequency_window.size() - 1);
        if (state.window_frequency != UINT64_MAX) state.window_frequency++;
        else counters.frequency_saturations++;
    }
    state.last_touch_demand = counters.demand_ordinal;
    state.last_demand_sequence = counters.event_sequence + 1;
    return append_event(llm_expert_cache_policy_event_type::demand, key, occurrence_count,
        logical_bundle_bytes, physical_slot_footprint_bytes);
}

void llm_expert_cache_policy::touch_slot(slot_state & slot) noexcept {
    const auto & key = keys[size_t(key_index(slot.key))];
    slot.last_touch_sequence = key.last_demand_sequence;
    slot.last_touch_demand = key.last_touch_demand;
}

void llm_expert_cache_policy::enforce_protected_capacity(uint32_t domain) noexcept {
    auto & state = domains[domain];
    while (state.protected_occupancy_bytes > state.protected_capacity_bytes) {
        int32_t demote = -1;
        bool pinned_blocked = false;
        for (uint32_t slot = 0; slot < slots.size(); ++slot) {
            const auto & candidate = slots[slot];
            if (!candidate.resident || candidate.domain != domain || candidate.current_segment != segment::protected_segment) continue;
            if (candidate.pin_count != 0) {
                pinned_blocked = true;
                continue;
            }
            if (demote < 0 || candidate.last_touch_sequence < slots[demote].last_touch_sequence ||
                (candidate.last_touch_sequence == slots[demote].last_touch_sequence && slot < uint32_t(demote))) {
                demote = int32_t(slot);
            }
        }
        if (demote < 0) {
            counters.pinned_blocked_demotions += pinned_blocked;
            break;
        }
        auto & candidate = slots[demote];
        candidate.current_segment = segment::probationary;
        state.protected_occupancy_bytes -= candidate.physical_slot_footprint_bytes;
        candidate.last_touch_sequence = counters.event_sequence + 1;
    }
}

llm_expert_cache_policy_result llm_expert_cache_policy::hit(uint32_t slot, uint64_t generation) noexcept {
    if (!request_active || slot >= slots.size()) {
        return llm_expert_cache_policy_result::failure(llm_expert_cache_policy_error::invalid_event);
    }
    auto & state = slots[slot];
    if (!state.resident || state.generation != generation) {
        counters.metadata_mismatches++;
        return llm_expert_cache_policy_result::failure(llm_expert_cache_policy_error::metadata_mismatch);
    }
    const auto capacity = preflight_events();
    if (!capacity.is_ready()) return capacity;
    const auto & key = keys[size_t(key_index(state.key))];
    if (config.policy == LLAMA_EXPERT_CACHE_POLICY_LFU_AGING &&
        !normalize_aging(state, key.last_touch_demand)) {
        return llm_expert_cache_policy_result::failure(llm_expert_cache_policy_error::invalid_event);
    }
    if (state.resident_frequency != UINT64_MAX) state.resident_frequency++;
    else counters.frequency_saturations++;
    if (config.policy == LLAMA_EXPERT_CACHE_POLICY_SLRU) {
        if (phase == llm_expert_cache_policy_phase::decode && state.current_segment == segment::probationary) {
            state.current_segment = segment::protected_segment;
            domains[state.domain].protected_occupancy_bytes += state.physical_slot_footprint_bytes;
            touch_slot(state);
            enforce_protected_capacity(state.domain);
        } else if (phase == llm_expert_cache_policy_phase::decode || state.current_segment == segment::probationary) {
            touch_slot(state);
        }
    } else {
        touch_slot(state);
    }
    counters.hits++;
    return append_event(llm_expert_cache_policy_event_type::hit, state.key, 0,
        state.logical_bundle_bytes, state.physical_slot_footprint_bytes, slot, generation);
}

llm_expert_cache_policy_result llm_expert_cache_policy::select(
        llm_expert_cache_policy_key key,
        const llm_expert_cache_policy_candidate * candidates,
        size_t candidate_count,
        llm_expert_cache_policy_decision & decision) noexcept {
    const int64_t requested_key = key_index(key);
    if (!request_active || requested_key < 0 || candidates == nullptr || candidate_count == 0) {
        return llm_expert_cache_policy_result::failure(llm_expert_cache_policy_error::invalid_key);
    }
    std::fill(candidate_seen.begin(), candidate_seen.end(), 0);
    for (size_t index = 0; index < candidate_count; ++index) {
        const auto & candidate = candidates[index];
        if (candidate.slot >= slots.size() || candidate.logical_bundle_bytes == 0 ||
            candidate.physical_slot_footprint_bytes != slot_footprint ||
            (candidate.free && candidate.eligible) || candidate_seen[candidate.slot] != 0) {
            counters.metadata_mismatches++;
            return llm_expert_cache_policy_result::failure(llm_expert_cache_policy_error::metadata_mismatch);
        }
        candidate_seen[candidate.slot] = 1;
    }
    const uint32_t domain = key_domain(key);
    const auto capacity = preflight_events();
    if (!capacity.is_ready()) return capacity;
    int32_t selected = -1;
    for (size_t index = 0; index < candidate_count; ++index) {
        const auto & candidate = candidates[index];
        if (candidate.slot >= slots.size() || slots[candidate.slot].domain != domain || !candidate.free) continue;
        if (selected < 0 || candidate.slot < candidates[selected].slot) selected = int32_t(index);
    }
    if (selected >= 0) {
        const auto & candidate = candidates[selected];
        decision = { candidate.slot, candidate.generation, true, true };
        counters.free_selections++;
        return append_event(llm_expert_cache_policy_event_type::victim_selected, key, 0,
            candidate.logical_bundle_bytes, candidate.physical_slot_footprint_bytes,
            candidate.slot, candidate.generation, 1, 0, true, 1);
    }
    for (size_t index = 0; index < candidate_count; ++index) {
        const auto & candidate = candidates[index];
        if (!candidate.eligible || candidate.free || candidate.slot >= slots.size() || slots[candidate.slot].domain != domain) continue;
        const auto & state = slots[candidate.slot];
        if (!state.resident || state.generation != candidate.generation || !key_matches(state.key, candidate.key)) {
            counters.metadata_mismatches++;
            return llm_expert_cache_policy_result::failure(llm_expert_cache_policy_error::metadata_mismatch);
        }
        if (selected < 0) {
            selected = int32_t(index);
            continue;
        }
        if (candidate_precedes(candidate.slot, candidates[selected].slot)) selected = int32_t(index);
    }
    if (selected < 0) {
        counters.misses++;
        return llm_expert_cache_policy_result::failure(llm_expert_cache_policy_error::no_victim);
    }
    const auto & candidate = candidates[selected];
    if (config.policy == LLAMA_EXPERT_CACHE_POLICY_SLRU &&
        slots[candidate.slot].current_segment == segment::protected_segment) {
        counters.protected_forced_victims++;
    }
    decision = { candidate.slot, candidate.generation, false, true };
    counters.victim_selections++;
    return append_event(llm_expert_cache_policy_event_type::victim_selected, key, 0,
        candidate.logical_bundle_bytes, candidate.physical_slot_footprint_bytes,
        candidate.slot, candidate.generation, 2, 0, true, 2);
}

llm_expert_cache_policy_result llm_expert_cache_policy::plan_evictions(
        llm_expert_cache_policy_key key,
        uint64_t logical_bundle_bytes,
        uint64_t physical_slot_footprint_bytes,
        const llm_expert_cache_policy_candidate * candidates,
        size_t candidate_count,
        llm_expert_cache_policy_decision * decisions,
        size_t decision_capacity,
        size_t & decision_count) noexcept {
    decision_count = 0;
    if (!request_active || key_index(key) < 0 || logical_bundle_bytes == 0 ||
        physical_slot_footprint_bytes == 0 || candidates == nullptr ||
        candidate_count == 0 || decisions == nullptr) {
        return llm_expert_cache_policy_result::failure(llm_expert_cache_policy_error::invalid_key);
    }
    const uint32_t domain = key_domain(key);
    const auto & domain_state = domains[domain];
    if (physical_slot_footprint_bytes > domain_state.quota_bytes) {
        return llm_expert_cache_policy_result::failure(llm_expert_cache_policy_error::no_victim);
    }
    std::fill(candidate_seen.begin(), candidate_seen.end(), 0);
    for (size_t index = 0; index < candidate_count; ++index) {
        const auto & candidate = candidates[index];
        if (candidate.slot >= slots.size() || candidate.free || !candidate.eligible ||
            slots[candidate.slot].domain != domain ||
            !slots[candidate.slot].resident ||
            slots[candidate.slot].generation != candidate.generation ||
            !key_matches(slots[candidate.slot].key, candidate.key) ||
            candidate.physical_slot_footprint_bytes !=
                slots[candidate.slot].physical_slot_footprint_bytes ||
            candidate_seen[candidate.slot] != 0) {
            counters.metadata_mismatches++;
            return llm_expert_cache_policy_result::failure(
                llm_expert_cache_policy_error::metadata_mismatch);
        }
        candidate_seen[candidate.slot] = 1;
    }
    const uint64_t available = domain_state.quota_bytes - domain_state.occupancy_bytes;
    if (physical_slot_footprint_bytes <= available) return llm_expert_cache_policy_result::success();
    const uint64_t required = physical_slot_footprint_bytes - available;
    uint64_t released = 0;
    while (released < required) {
        if (decision_count >= decision_capacity) {
            decision_count = 0;
            return llm_expert_cache_policy_result::failure(llm_expert_cache_policy_error::overflow);
        }
        int32_t selected = -1;
        for (size_t index = 0; index < candidate_count; ++index) {
            bool already_selected = false;
            for (size_t prior = 0; prior < decision_count; ++prior) {
                already_selected = already_selected || decisions[prior].slot == candidates[index].slot;
            }
            if (already_selected) continue;
            if (selected < 0 || candidate_precedes(candidates[index].slot,
                    candidates[size_t(selected)].slot)) {
                selected = int32_t(index);
            }
        }
        if (selected < 0) {
            decision_count = 0;
            return llm_expert_cache_policy_result::failure(llm_expert_cache_policy_error::no_victim);
        }
        const auto & candidate = candidates[size_t(selected)];
        decisions[decision_count++] = { candidate.slot, candidate.generation, false, true };
        uint64_t next = 0;
        if (!checked_add(released, candidate.physical_slot_footprint_bytes, next)) {
            decision_count = 0;
            return llm_expert_cache_policy_result::failure(llm_expert_cache_policy_error::overflow);
        }
        released = next;
    }
    const auto capacity = preflight_events(decision_count);
    if (!capacity.is_ready()) {
        decision_count = 0;
        return capacity;
    }
    for (size_t index = 0; index < decision_count; ++index) {
        const auto & decision = decisions[index];
        if (config.policy == LLAMA_EXPERT_CACHE_POLICY_SLRU &&
            slots[decision.slot].current_segment == segment::protected_segment) {
            counters.protected_forced_victims++;
        }
        counters.victim_selections++;
        const auto event = append_event(llm_expert_cache_policy_event_type::victim_selected,
            key, 0, logical_bundle_bytes, physical_slot_footprint_bytes,
            decision.slot, decision.generation, 2, 0, true, 2);
        if (!event.is_ready()) return event;
    }
    return llm_expert_cache_policy_result::success();
}

llm_expert_cache_policy_result llm_expert_cache_policy::optional_admission(
        llm_expert_cache_policy_key key,
        llm_expert_cache_policy_admission admission,
        const llm_expert_cache_policy_candidate * candidates,
        size_t candidate_count,
        llm_expert_cache_policy_decision & decision) noexcept {
    if (!request_active) {
        return llm_expert_cache_policy_result::failure(llm_expert_cache_policy_error::invalid_event);
    }
    if (admission == llm_expert_cache_policy_admission::mandatory_current_output ||
        config.admission != LLAMA_EXPERT_CACHE_ADMISSION_FREQUENCY_WINDOW) {
        if (admission == llm_expert_cache_policy_admission::mandatory_current_output) {
            const auto capacity = preflight_events(2);
            if (!capacity.is_ready()) return capacity;
            auto result = select(key, candidates, candidate_count, decision);
            if (!result.is_ready()) return result;
            const auto selected = std::find_if(candidates, candidates + candidate_count,
                [&](const llm_expert_cache_policy_candidate & candidate) {
                    return candidate.slot == decision.slot;
                });
            if (selected == candidates + candidate_count) {
                counters.metadata_mismatches++;
                return llm_expert_cache_policy_result::failure(
                    llm_expert_cache_policy_error::metadata_mismatch);
            }
            counters.mandatory_admissions++;
            return append_event(llm_expert_cache_policy_event_type::optional_admission, key, 0,
                selected->logical_bundle_bytes, selected->physical_slot_footprint_bytes,
                decision.slot, decision.generation, 2, 0, true, 5);
        }
        const auto capacity = preflight_events(2);
        if (!capacity.is_ready()) return capacity;
        auto result = select(key, candidates, candidate_count, decision);
        if (!result.is_ready()) return result;
        const auto selected = std::find_if(candidates, candidates + candidate_count,
            [&](const llm_expert_cache_policy_candidate & candidate) {
                return candidate.slot == decision.slot;
            });
        if (selected == candidates + candidate_count) {
            counters.metadata_mismatches++;
            return llm_expert_cache_policy_result::failure(llm_expert_cache_policy_error::metadata_mismatch);
        }
        counters.optional_admission_accepts++;
        return append_event(llm_expert_cache_policy_event_type::optional_admission, key, 0,
            selected->logical_bundle_bytes, selected->physical_slot_footprint_bytes,
            decision.slot, decision.generation, 1, 0, true, decision.free ? 1 : 2);
    }
    const int64_t candidate_key = key_index(key);
    if (candidate_key < 0 || candidates == nullptr || candidate_count == 0) {
        return llm_expert_cache_policy_result::failure(llm_expert_cache_policy_error::invalid_key);
    }
    const auto capacity = preflight_events(2);
    if (!capacity.is_ready()) return capacity;
    const uint32_t domain = key_domain(key);
    int32_t incumbent = -1;
    std::fill(candidate_seen.begin(), candidate_seen.end(), 0);
    for (size_t index = 0; index < candidate_count; ++index) {
        const auto & candidate = candidates[index];
        if (candidate.slot >= slots.size() || candidate.logical_bundle_bytes == 0 ||
            candidate.physical_slot_footprint_bytes != slot_footprint ||
            (candidate.free && candidate.eligible) || candidate_seen[candidate.slot] != 0) {
            counters.metadata_mismatches++;
            return llm_expert_cache_policy_result::failure(
                llm_expert_cache_policy_error::metadata_mismatch);
        }
        candidate_seen[candidate.slot] = 1;
        if (slots[candidate.slot].domain != domain) continue;
        if (candidate.free) {
            decision = { candidate.slot, candidate.generation, true, true };
            counters.free_selections++;
            auto selected = append_event(llm_expert_cache_policy_event_type::victim_selected,
                key, 0, candidate.logical_bundle_bytes, candidate.physical_slot_footprint_bytes,
                candidate.slot, candidate.generation, 1, 0, true, 1);
            if (!selected.is_ready()) return selected;
            counters.optional_admission_accepts++;
            return append_event(llm_expert_cache_policy_event_type::optional_admission, key, 0,
                candidate.logical_bundle_bytes, candidate.physical_slot_footprint_bytes,
                candidate.slot, candidate.generation, 1, 0, true, 1);
        }
        const auto & state = slots[candidate.slot];
        if (!candidate.eligible || state.current_segment != segment::probationary) continue;
        if (!state.resident || state.generation != candidate.generation ||
            !key_matches(state.key, candidate.key)) {
            counters.metadata_mismatches++;
            return llm_expert_cache_policy_result::failure(
                llm_expert_cache_policy_error::metadata_mismatch);
        }
        if (incumbent < 0) incumbent = int32_t(index);
        else {
            const auto & best = slots[candidates[incumbent].slot];
            const uint64_t frequency = keys[size_t(key_index(state.key))].window_frequency;
            const uint64_t best_frequency = keys[size_t(key_index(best.key))].window_frequency;
            if (frequency < best_frequency ||
                (frequency == best_frequency && (state.last_touch_sequence < best.last_touch_sequence ||
                 (state.last_touch_sequence == best.last_touch_sequence && candidate.slot < candidates[incumbent].slot)))) {
                incumbent = int32_t(index);
            }
        }
    }
    if (incumbent >= 0) {
        const auto & selected = candidates[incumbent];
        const auto & state = slots[selected.slot];
        if (keys[size_t(candidate_key)].window_frequency > keys[size_t(key_index(state.key))].window_frequency) {
            decision = { selected.slot, selected.generation, false, true };
            counters.victim_selections++;
            auto selected_event = append_event(llm_expert_cache_policy_event_type::victim_selected,
                key, 0, selected.logical_bundle_bytes, selected.physical_slot_footprint_bytes,
                selected.slot, selected.generation, 2, 0, true, 2);
            if (!selected_event.is_ready()) return selected_event;
            counters.optional_admission_accepts++;
            return append_event(llm_expert_cache_policy_event_type::optional_admission, key, 0,
                selected.logical_bundle_bytes, selected.physical_slot_footprint_bytes,
                selected.slot, selected.generation, 1, 0, true, 2);
        }
    }
    decision = {};
    counters.optional_admission_rejects++;
    return append_event(llm_expert_cache_policy_event_type::optional_admission, key, 0, 0, 0,
        UINT32_MAX, 0, 0, 0, false, incumbent < 0 ? 4 : 3);
}

llm_expert_cache_policy_result llm_expert_cache_policy::evict(uint32_t slot, uint64_t generation) noexcept {
    if (slot >= slots.size()) return llm_expert_cache_policy_result::failure(llm_expert_cache_policy_error::invalid_event);
    auto & state = slots[slot];
    if (!state.resident || state.generation != generation || state.pin_count != 0) {
        counters.metadata_mismatches++;
        return llm_expert_cache_policy_result::failure(llm_expert_cache_policy_error::metadata_mismatch);
    }
    const auto capacity = preflight_events();
    if (!capacity.is_ready()) return capacity;
    const auto key = state.key;
    const uint32_t domain = state.domain;
    const auto logical = state.logical_bundle_bytes;
    const auto physical = state.physical_slot_footprint_bytes;
    if (state.current_segment == segment::protected_segment) {
        domains[state.domain].protected_occupancy_bytes -= physical;
    }
    domains[state.domain].occupancy_bytes -= physical;
    state = {};
    state.domain = domain;
    return append_event(llm_expert_cache_policy_event_type::evict, key, 0, logical, physical, slot, generation);
}

llm_expert_cache_policy_result llm_expert_cache_policy::load_begin(
        uint32_t slot,
        uint64_t generation,
        llm_expert_cache_policy_key key,
        uint64_t logical_bundle_bytes,
        uint64_t physical_slot_footprint_bytes,
        bool demand_caused) noexcept {
    if (slot >= slots.size() || key_index(key) < 0 || generation == 0 || logical_bundle_bytes == 0 ||
        physical_slot_footprint_bytes == 0 || slots[slot].resident || slots[slot].loading ||
        slots[slot].domain != key_domain(key)) {
        return llm_expert_cache_policy_result::failure(llm_expert_cache_policy_error::metadata_mismatch);
    }
    const auto capacity = preflight_events(2);
    if (!capacity.is_ready()) return capacity;
    if (counters.operation_ordinal == UINT64_MAX) {
        return llm_expert_cache_policy_result::failure(llm_expert_cache_policy_error::sequence_exhausted);
    }
    if (physical_slot_footprint_bytes >
            domains[slots[slot].domain].quota_bytes - domains[slots[slot].domain].occupancy_bytes) {
        return llm_expert_cache_policy_result::failure(llm_expert_cache_policy_error::no_victim);
    }
    counters.operation_ordinal++;
    const uint32_t domain = slots[slot].domain;
    const auto & key_state = keys[size_t(key_index(key))];
    slots[slot] = { key, generation, key_state.last_demand_sequence, key_state.last_touch_demand,
        logical_bundle_bytes, physical_slot_footprint_bytes, demand_caused ? 1u : 0u,
        config.policy == LLAMA_EXPERT_CACHE_POLICY_LFU_AGING ?
            key_state.last_touch_demand/config.lfu_aging_interval_events : 0,
        counters.operation_ordinal, domain, 0, segment::none, false, false, true, false };
    domains[domain].occupancy_bytes += physical_slot_footprint_bytes;
    reserved_terminal_events++;
    const auto result = append_event(llm_expert_cache_policy_event_type::load_begin, key, 0,
        logical_bundle_bytes, physical_slot_footprint_bytes, slot, generation, 0,
        counters.operation_ordinal);
    if (!result.is_ready()) reserved_terminal_events--;
    return result;
}

llm_expert_cache_policy_result llm_expert_cache_policy::load_complete(uint32_t slot, uint64_t generation) noexcept {
    if (slot >= slots.size() || !slots[slot].loading || slots[slot].terminal_pending ||
        slots[slot].generation != generation) {
        counters.metadata_mismatches++;
        return llm_expert_cache_policy_result::failure(llm_expert_cache_policy_error::metadata_mismatch);
    }
    slots[slot].terminal_pending = true;
    slots[slot].terminal_success = true;
    return flush_terminal_events();
}

llm_expert_cache_policy_result llm_expert_cache_policy::load_failed(uint32_t slot, uint64_t generation) noexcept {
    if (slot >= slots.size() || !slots[slot].loading || slots[slot].terminal_pending ||
        slots[slot].generation != generation) {
        counters.metadata_mismatches++;
        return llm_expert_cache_policy_result::failure(llm_expert_cache_policy_error::metadata_mismatch);
    }
    slots[slot].terminal_pending = true;
    slots[slot].terminal_success = false;
    return flush_terminal_events();
}

llm_expert_cache_policy_result llm_expert_cache_policy::flush_terminal_events() noexcept {
    while (terminal_operation_ordinal < counters.operation_ordinal) {
        const uint64_t expected = terminal_operation_ordinal + 1;
        int32_t selected = -1;
        for (uint32_t slot = 0; slot < slots.size(); ++slot) {
            if (slots[slot].loading && slots[slot].origin_operation_ordinal == expected) {
                selected = int32_t(slot);
                break;
            }
        }
        if (selected < 0) {
            counters.metadata_mismatches++;
            return llm_expert_cache_policy_result::failure(
                llm_expert_cache_policy_error::metadata_mismatch);
        }
        auto & pending = slots[uint32_t(selected)];
        if (!pending.terminal_pending) return llm_expert_cache_policy_result::success();
        if (reserved_terminal_events == 0 ||
            (config.state_attestation && event_write >= events.size()) ||
            counters.event_sequence == UINT64_MAX) {
            return llm_expert_cache_policy_result::failure(
                reserved_terminal_events == 0 ? llm_expert_cache_policy_error::metadata_mismatch :
                config.state_attestation && event_write >= events.size() ?
                    llm_expert_cache_policy_error::transcript_full :
                llm_expert_cache_policy_error::sequence_exhausted);
        }
        const auto state = pending;
        terminal_operation_ordinal = expected;
        reserved_terminal_events--;
        if (state.terminal_success) {
            pending.terminal_pending = false;
            pending.terminal_success = false;
            pending.loading = false;
            pending.resident = true;
            pending.current_segment = segment::probationary;
            const auto event = append_event(llm_expert_cache_policy_event_type::load_complete,
                state.key, 0, state.logical_bundle_bytes, state.physical_slot_footprint_bytes,
                uint32_t(selected), state.generation, 0, state.origin_operation_ordinal);
            if (!event.is_ready()) return event;
        } else {
            const uint32_t domain = state.domain;
            pending = {};
            pending.domain = domain;
            domains[domain].occupancy_bytes -= state.physical_slot_footprint_bytes;
            const auto event = append_event(llm_expert_cache_policy_event_type::load_failed,
                state.key, 0, state.logical_bundle_bytes, state.physical_slot_footprint_bytes,
                uint32_t(selected), state.generation, 0, state.origin_operation_ordinal);
            if (!event.is_ready()) return event;
        }
    }
    return llm_expert_cache_policy_result::success();
}

llm_expert_cache_policy_result llm_expert_cache_policy::pin(uint32_t slot, uint64_t generation) noexcept {
    if (slot >= slots.size() || !slots[slot].resident || slots[slot].generation != generation) {
        counters.metadata_mismatches++;
        return llm_expert_cache_policy_result::failure(llm_expert_cache_policy_error::metadata_mismatch);
    }
    if (slots[slot].pin_count == UINT32_MAX) {
        return llm_expert_cache_policy_result::failure(llm_expert_cache_policy_error::overflow);
    }
    const auto capacity = preflight_events();
    if (!capacity.is_ready()) return capacity;
    slots[slot].pin_count++;
    const auto result = append_event(llm_expert_cache_policy_event_type::pin, slots[slot].key, 0,
        slots[slot].logical_bundle_bytes, slots[slot].physical_slot_footprint_bytes, slot, generation);
    if (!result.is_ready()) slots[slot].pin_count--;
    return result;
}

llm_expert_cache_policy_result llm_expert_cache_policy::unpin(uint32_t slot, uint64_t generation) noexcept {
    if (slot >= slots.size() || !slots[slot].resident || slots[slot].generation != generation) {
        counters.metadata_mismatches++;
        return llm_expert_cache_policy_result::failure(llm_expert_cache_policy_error::metadata_mismatch);
    }
    if (slots[slot].pin_count == 0) {
        counters.metadata_mismatches++;
        return llm_expert_cache_policy_result::failure(llm_expert_cache_policy_error::metadata_mismatch);
    }
    const auto capacity = preflight_events();
    if (!capacity.is_ready()) return capacity;
    slots[slot].pin_count--;
    if (config.policy == LLAMA_EXPERT_CACHE_POLICY_SLRU) {
        enforce_protected_capacity(slots[slot].domain);
    }
    const auto result = append_event(llm_expert_cache_policy_event_type::unpin, slots[slot].key, 0,
        slots[slot].logical_bundle_bytes, slots[slot].physical_slot_footprint_bytes, slot, generation);
    if (!result.is_ready()) slots[slot].pin_count++;
    return result;
}

llm_expert_cache_policy_result llm_expert_cache_policy::request_end(
        bool success, bool cancelled, bool allow_deferred_terminals) noexcept {
    if (!request_active || (success && cancelled) ||
        (!allow_deferred_terminals && reserved_terminal_events != 0)) {
        return llm_expert_cache_policy_result::failure(llm_expert_cache_policy_error::invalid_event);
    }
    const auto capacity = preflight_events();
    if (!capacity.is_ready()) return capacity;
    const auto result = append_event(llm_expert_cache_policy_event_type::request_end, {}, 0, 0, 0,
        UINT32_MAX, 0, cancelled ? 2 : success ? 1 : 0);
    request_active = false;
    return result;
}

llm_expert_cache_policy_result llm_expert_cache_policy::remove_resident(
        uint32_t slot, uint64_t generation) noexcept {
    return evict(slot, generation);
}

llm_expert_cache_policy_result llm_expert_cache_policy::reset() noexcept {
    if (!initialized || request_active) {
        return llm_expert_cache_policy_result::failure(llm_expert_cache_policy_error::invalid_event);
    }
    if (reserved_terminal_events != 0) {
        return llm_expert_cache_policy_result::failure(llm_expert_cache_policy_error::metadata_mismatch);
    }
    for (const auto & slot : slots) {
        if (slot.loading || slot.resident) {
            return llm_expert_cache_policy_result::failure(llm_expert_cache_policy_error::metadata_mismatch);
        }
    }
    const auto capacity = preflight_events();
    if (!capacity.is_ready()) return capacity;
    for (auto & key : keys) key = {};
    std::fill(frequency_window.begin(), frequency_window.end(), UINT64_MAX);
    frequency_window_write = 0;
    frequency_window_size = 0;
    frequency_window_state_digest = 0;
    for (size_t index = 0; index < frequency_window.size(); ++index) {
        frequency_window_state_digest ^= window_contribution(index, frequency_window[index]);
    }
    for (auto & domain : domains) {
        domain.occupancy_bytes = 0;
        domain.protected_occupancy_bytes = 0;
    }
    counters.demand_ordinal = 0;
    return append_event(llm_expert_cache_policy_event_type::reset);
}

llm_expert_cache_policy_result llm_expert_cache_policy::surrender() noexcept {
    const auto capacity = preflight_events(2);
    if (!capacity.is_ready()) return capacity;
    auto result = reset();
    if (!result.is_ready()) return result;
    result = append_event(llm_expert_cache_policy_event_type::surrender);
    if (result.is_ready()) initialized = false;
    return result;
}

bool llm_expert_cache_policy::validate_resident(
        uint32_t slot,
        uint64_t generation,
        llm_expert_cache_policy_key key) const noexcept {
    return slot < slots.size() && slots[slot].resident && slots[slot].generation == generation &&
        key_matches(slots[slot].key, key);
}

bool llm_expert_cache_policy::validate_loading(
        uint32_t slot,
        uint64_t generation,
        llm_expert_cache_policy_key key) const noexcept {
    return slot < slots.size() && slots[slot].loading && !slots[slot].resident &&
        slots[slot].generation == generation && key_matches(slots[slot].key, key);
}

bool llm_expert_cache_policy::resident_precedes(uint32_t lhs_slot, uint32_t rhs_slot) const noexcept {
    if (lhs_slot >= slots.size() || rhs_slot >= slots.size() ||
        !slots[lhs_slot].resident || !slots[rhs_slot].resident) {
        return false;
    }
    return candidate_precedes(lhs_slot, rhs_slot);
}

bool llm_expert_cache_policy::validate_free(uint32_t slot) const noexcept {
    return slot < slots.size() && !slots[slot].loading && !slots[slot].resident;
}

llm_expert_cache_policy_result llm_expert_cache_policy::validate_evictable(
        uint32_t slot, uint64_t generation) const noexcept {
    if (slot >= slots.size() || !slots[slot].resident || slots[slot].generation != generation ||
        slots[slot].pin_count != 0) {
        return llm_expert_cache_policy_result::failure(llm_expert_cache_policy_error::metadata_mismatch);
    }
    return preflight_events();
}

bool llm_expert_cache_policy::set_ordinals_for_testing(
        uint64_t event_sequence,
        uint64_t demand_ordinal,
        uint64_t operation_ordinal) noexcept {
    if (!config.state_attestation || !initialized || request_active || event_write != 0) return false;
    counters.event_sequence = event_sequence;
    counters.demand_ordinal = demand_ordinal;
    counters.operation_ordinal = operation_ordinal;
    refresh_state_digest();
    return true;
}

bool llm_expert_cache_policy::set_resident_frequency_for_testing(
        uint32_t slot, uint64_t frequency, uint64_t aging_epoch) noexcept {
    if (!config.state_attestation || !initialized || slot >= slots.size() || !slots[slot].resident) return false;
    slots[slot].resident_frequency = frequency;
    slots[slot].aging_epoch = aging_epoch;
    refresh_state_digest();
    return true;
}

void llm_expert_cache_policy::refresh_state_digest() noexcept {
    counters.state_digest = config.state_attestation ? hash_state() : 0;
}

uint64_t llm_expert_cache_policy::hash_state() const noexcept {
    uint64_t hash = fnv_offset;
    hash_append(hash, config.digest);
    hash_append(hash, counters.request_ordinal);
    hash_append(hash, counters.ubatch_ordinal);
    hash_append(hash, counters.event_sequence);
    hash_append(hash, counters.demand_ordinal);
    hash_append(hash, counters.operation_ordinal);
    hash_append(hash, reserved_terminal_events);
    hash_append(hash, terminal_operation_ordinal);
    hash_append(hash, frequency_window_write);
    hash_append(hash, frequency_window_size);
    hash_append(hash, frequency_window_state_digest);
    hash_append(hash, uint8_t(phase));
    for (const auto & key : keys) {
        hash_append(hash, key.window_frequency);
        hash_append(hash, key.last_touch_demand);
        hash_append(hash, key.last_demand_sequence);
    }
    for (const auto & slot : slots) {
        hash_append(hash, uint32_t(slot.key.layer));
        hash_append(hash, uint32_t(slot.key.expert));
        if (slot.key.layout_class_id != 0) hash_append(hash, slot.key.layout_class_id);
        hash_append(hash, slot.generation);
        hash_append(hash, slot.last_touch_sequence);
        hash_append(hash, slot.last_touch_demand);
        hash_append(hash, slot.logical_bundle_bytes);
        hash_append(hash, slot.physical_slot_footprint_bytes);
        hash_append(hash, slot.resident_frequency);
        hash_append(hash, slot.aging_epoch);
        hash_append(hash, slot.origin_operation_ordinal);
        hash_append(hash, slot.domain);
        hash_append(hash, slot.pin_count);
        hash_append(hash, uint8_t(slot.current_segment));
        hash_append(hash, slot.loading);
        hash_append(hash, slot.resident);
    }
    for (const auto & domain : domains) {
        hash_append(hash, domain.domain);
        hash_append(hash, uint32_t(domain.layer));
        hash_append(hash, domain.slot_count);
        hash_append(hash, domain.quota_bytes);
        hash_append(hash, domain.occupancy_bytes);
        hash_append(hash, domain.protected_capacity_bytes);
        hash_append(hash, domain.protected_occupancy_bytes);
    }
    return hash;
}
