#pragma once

#include "llama.h"

#include <cstddef>
#include <cstdint>
#include <vector>

enum class llm_expert_cache_policy_tier : uint8_t {
    hot,
    cold,
};

enum class llm_expert_cache_policy_phase : uint8_t {
    prefill,
    decode,
};

enum class llm_expert_cache_policy_event_type : uint8_t {
    request_begin,
    phase_transition,
    demand,
    hit,
    victim_selected,
    evict,
    load_begin,
    load_complete,
    load_failed,
    pin,
    unpin,
    optional_admission,
    request_end,
    reset,
    surrender,
};

enum class llm_expert_cache_policy_admission : uint8_t {
    mandatory_current_output,
    optional_cpu_served,
    optional_background,
};

enum class llm_expert_cache_policy_error : uint8_t {
    none,
    invalid_configuration,
    invalid_key,
    invalid_event,
    sequence_exhausted,
    no_victim,
    metadata_mismatch,
    overflow,
    transcript_full,
};

struct llm_expert_cache_policy_result {
    llm_expert_cache_policy_error error = llm_expert_cache_policy_error::none;

    bool is_ready() const noexcept { return error == llm_expert_cache_policy_error::none; }
    static llm_expert_cache_policy_result success() noexcept { return {}; }
    static llm_expert_cache_policy_result failure(llm_expert_cache_policy_error error) noexcept { return { error }; }
};

struct llm_expert_cache_policy_key {
    int32_t layer = -1;
    int32_t expert = -1;
    uint8_t layout_class_id = 0;
};

struct llm_expert_cache_policy_config_internal {
    llama_expert_cache_policy policy = LLAMA_EXPERT_CACHE_POLICY_LRU;
    llama_expert_cache_policy_scope scope = LLAMA_EXPERT_CACHE_POLICY_SCOPE_GLOBAL;
    uint32_t slru_protected_ratio_bps = 0;
    llama_expert_cache_admission admission = LLAMA_EXPERT_CACHE_ADMISSION_ALWAYS;
    uint32_t admission_window_events = 0;
    uint32_t lfu_aging_interval_events = 0;
    bool state_attestation = true;
    bool supplied = false;
    llama_expert_cache_policy_config supplied_config = {};
    llama_expert_cache_policy_config resolved_config = {};
    uint64_t digest = 0;
};

llm_expert_cache_policy_result llm_expert_cache_policy_copy_config(
        const llama_expert_cache_policy_config * source,
        llm_expert_cache_policy_tier tier,
        llm_expert_cache_policy_config_internal & destination) noexcept;

struct llm_expert_cache_policy_candidate {
    uint32_t slot = UINT32_MAX;
    uint64_t generation = 0;
    llm_expert_cache_policy_key key;
    uint64_t logical_bundle_bytes = 0;
    uint64_t physical_slot_footprint_bytes = 0;
    bool free = false;
    bool eligible = false;
};

struct llm_expert_cache_policy_decision {
    uint32_t slot = UINT32_MAX;
    uint64_t generation = 0;
    bool free = false;
    bool accept = false;
};

struct llm_expert_cache_policy_event {
    uint32_t schema_version = 1;
    uint64_t request_ordinal = 0;
    uint64_t ubatch_ordinal = 0;
    llm_expert_cache_policy_tier tier = llm_expert_cache_policy_tier::hot;
    llm_expert_cache_policy_event_type type = llm_expert_cache_policy_event_type::request_begin;
    uint64_t event_sequence = 0;
    uint64_t demand_ordinal = 0;
    uint64_t origin_operation_ordinal = 0;
    llm_expert_cache_policy_phase phase = llm_expert_cache_policy_phase::prefill;
    llm_expert_cache_policy_key key;
    uint64_t occurrence_count = 0;
    uint64_t logical_bundle_bytes = 0;
    uint64_t physical_slot_footprint_bytes = 0;
    uint32_t slot = UINT32_MAX;
    uint64_t generation = 0;
    uint32_t domain = UINT32_MAX;
    bool eligible = false;
    uint8_t decision = 0;
    uint8_t reason = 0;
    uint64_t state_digest = 0;
};

struct llm_expert_cache_policy_domain_diagnostics {
    uint32_t domain = UINT32_MAX;
    int32_t layer = -1;
    uint32_t slot_count = 0;
    uint64_t quota_bytes = 0;
    uint64_t occupancy_bytes = 0;
    uint64_t protected_capacity_bytes = 0;
    uint64_t protected_occupancy_bytes = 0;
};

struct llm_expert_cache_policy_diagnostics {
    llm_expert_cache_policy_config_internal config;
    uint64_t administration_requested_bytes = 0;
    uint64_t administration_actual_bytes = 0;
    uint64_t administration_bytes = 0;
    uint64_t peak_administration_bytes = 0;
    uint64_t request_ordinal = 0;
    uint64_t ubatch_ordinal = 0;
    uint64_t event_sequence = 0;
    uint64_t demand_ordinal = 0;
    uint64_t operation_ordinal = 0;
    uint64_t events = 0;
    uint64_t demands = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t free_selections = 0;
    uint64_t victim_selections = 0;
    uint64_t mandatory_admissions = 0;
    uint64_t optional_admission_accepts = 0;
    uint64_t optional_admission_rejects = 0;
    uint64_t protected_forced_victims = 0;
    uint64_t pinned_blocked_demotions = 0;
    uint64_t frequency_saturations = 0;
    uint64_t metadata_mismatches = 0;
    uint64_t transcript_records = 0;
    uint64_t transcript_dropped = 0;
    uint64_t state_digest = 0;
};

class llm_expert_cache_policy {
public:
    llm_expert_cache_policy() = default;

    llm_expert_cache_policy_result initialize(
            const llm_expert_cache_policy_config_internal & config,
            llm_expert_cache_policy_tier tier,
            const int32_t * routed_layers,
            uint32_t routed_layer_count,
            uint32_t experts_per_layer,
            uint32_t minimum_domain_slots,
            uint32_t slot_count,
            uint64_t physical_slot_footprint_bytes,
            uint32_t transcript_capacity,
            const uint64_t * domain_budget_bytes = nullptr,
            uint32_t domain_budget_count = 0) noexcept;

    llm_expert_cache_policy_result request_begin() noexcept;
    llm_expert_cache_policy_result set_ubatch_ordinal(uint64_t ordinal) noexcept;
    llm_expert_cache_policy_result phase_transition(llm_expert_cache_policy_phase phase) noexcept;
    llm_expert_cache_policy_result demand(
            llm_expert_cache_policy_key key,
            uint64_t occurrence_count,
            uint64_t logical_bundle_bytes,
            uint64_t physical_slot_footprint_bytes) noexcept;
    llm_expert_cache_policy_result hit(uint32_t slot, uint64_t generation) noexcept;
    llm_expert_cache_policy_result select(
            llm_expert_cache_policy_key key,
            const llm_expert_cache_policy_candidate * candidates,
            size_t candidate_count,
            llm_expert_cache_policy_decision & decision) noexcept;
    llm_expert_cache_policy_result plan_evictions(
            llm_expert_cache_policy_key key,
            uint64_t logical_bundle_bytes,
            uint64_t physical_slot_footprint_bytes,
            const llm_expert_cache_policy_candidate * candidates,
            size_t candidate_count,
            llm_expert_cache_policy_decision * decisions,
            size_t decision_capacity,
            size_t & decision_count) noexcept;
    llm_expert_cache_policy_result optional_admission(
            llm_expert_cache_policy_key key,
            llm_expert_cache_policy_admission admission,
            const llm_expert_cache_policy_candidate * candidates,
            size_t candidate_count,
            llm_expert_cache_policy_decision & decision) noexcept;
    llm_expert_cache_policy_result evict(uint32_t slot, uint64_t generation) noexcept;
    llm_expert_cache_policy_result load_begin(
            uint32_t slot,
            uint64_t generation,
            llm_expert_cache_policy_key key,
            uint64_t logical_bundle_bytes,
            uint64_t physical_slot_footprint_bytes,
            bool demand_caused = true) noexcept;
    llm_expert_cache_policy_result load_complete(uint32_t slot, uint64_t generation) noexcept;
    llm_expert_cache_policy_result load_failed(uint32_t slot, uint64_t generation) noexcept;
    llm_expert_cache_policy_result pin(uint32_t slot, uint64_t generation) noexcept;
    llm_expert_cache_policy_result unpin(uint32_t slot, uint64_t generation) noexcept;
    llm_expert_cache_policy_result request_end(bool success, bool cancelled) noexcept;
    llm_expert_cache_policy_result validate_event_capacity(size_t count) const noexcept;
    llm_expert_cache_policy_result remove_resident(uint32_t slot, uint64_t generation) noexcept;
    llm_expert_cache_policy_result reset() noexcept;
    llm_expert_cache_policy_result surrender() noexcept;

    bool validate_resident(uint32_t slot, uint64_t generation, llm_expert_cache_policy_key key) const noexcept;
    bool validate_loading(uint32_t slot, uint64_t generation, llm_expert_cache_policy_key key) const noexcept;
    bool resident_precedes(uint32_t lhs_slot, uint32_t rhs_slot) const noexcept;
    bool validate_free(uint32_t slot) const noexcept;
    llm_expert_cache_policy_result validate_evictable(uint32_t slot, uint64_t generation) const noexcept;
    const llm_expert_cache_policy_diagnostics & diagnostics() const noexcept { return counters; }
    const std::vector<llm_expert_cache_policy_domain_diagnostics> & domain_diagnostics() const noexcept {
        return domains;
    }
    const std::vector<llm_expert_cache_policy_event> & transcript() const noexcept { return events; }
    size_t transcript_size() const noexcept { return event_write; }
    bool set_ordinals_for_testing(
            uint64_t event_sequence,
            uint64_t demand_ordinal,
            uint64_t operation_ordinal) noexcept;
    bool set_resident_frequency_for_testing(uint32_t slot, uint64_t frequency, uint64_t aging_epoch) noexcept;

private:
    enum class segment : uint8_t { none, probationary, protected_segment };

    struct key_state {
        uint64_t window_frequency = 0;
        uint64_t last_touch_demand = 0;
        uint64_t last_demand_sequence = 0;
    };

    struct slot_state {
        llm_expert_cache_policy_key key;
        uint64_t generation = 0;
        uint64_t last_touch_sequence = 0;
        uint64_t last_touch_demand = 0;
        uint64_t logical_bundle_bytes = 0;
        uint64_t physical_slot_footprint_bytes = 0;
        uint64_t resident_frequency = 0;
        uint64_t aging_epoch = 0;
        uint64_t origin_operation_ordinal = 0;
        uint32_t domain = UINT32_MAX;
        uint32_t pin_count = 0;
        segment current_segment = segment::none;
        bool terminal_pending = false;
        bool terminal_success = false;
        bool loading = false;
        bool resident = false;
    };

    llm_expert_cache_policy_result append_event(
            llm_expert_cache_policy_event_type type,
            llm_expert_cache_policy_key key = {},
            uint64_t occurrence_count = 0,
            uint64_t logical_bundle_bytes = 0,
            uint64_t physical_slot_footprint_bytes = 0,
            uint32_t slot = UINT32_MAX,
            uint64_t generation = 0,
            uint8_t decision = 0,
            uint64_t origin_operation_ordinal = 0,
            bool eligible = false,
            uint8_t reason = 0) noexcept;
    llm_expert_cache_policy_result preflight_events(size_t count = 1) const noexcept;
    int32_t layer_index(int32_t layer) const noexcept;
    int64_t key_index(llm_expert_cache_policy_key key) const noexcept;
    uint32_t key_domain(llm_expert_cache_policy_key key) const noexcept;
    bool key_matches(llm_expert_cache_policy_key lhs, llm_expert_cache_policy_key rhs) const noexcept;
    bool normalize_aging(slot_state & slot, uint64_t current_demand_ordinal) noexcept;
    bool candidate_precedes(uint32_t lhs_slot, uint32_t rhs_slot) const noexcept;
    llm_expert_cache_policy_result flush_terminal_events() noexcept;
    void touch_slot(slot_state & slot) noexcept;
    void enforce_protected_capacity(uint32_t domain) noexcept;
    void refresh_state_digest() noexcept;
    uint64_t hash_state() const noexcept;

    llm_expert_cache_policy_config_internal config;
    llm_expert_cache_policy_tier tier = llm_expert_cache_policy_tier::hot;
    llm_expert_cache_policy_phase phase = llm_expert_cache_policy_phase::prefill;
    uint32_t experts_per_layer = 0;
    uint64_t slot_footprint = 0;
    bool initialized = false;
    bool request_active = false;
    std::vector<int32_t> layers;
    std::vector<key_state> keys;
    std::vector<slot_state> slots;
    std::vector<llm_expert_cache_policy_domain_diagnostics> domains;
    std::vector<uint64_t> frequency_window;
    size_t frequency_window_write = 0;
    size_t frequency_window_size = 0;
    std::vector<llm_expert_cache_policy_event> events;
    size_t event_write = 0;
    size_t reserved_terminal_events = 0;
    uint64_t terminal_operation_ordinal = 0;
    uint64_t frequency_window_state_digest = 0;
    llm_expert_cache_policy_diagnostics counters;
};
