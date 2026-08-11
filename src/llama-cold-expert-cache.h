#pragma once

#include "llama-expert-weight-provider.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

enum class llm_cold_slot_state {
    free,
    reserved,
    loading,
    ready,
    evicting,
    failed,
};

enum class llm_cold_reference_kind {
    hot,
    transfer,
    request,
    cpu_execution,
};

struct llm_cold_reference {
    uint32_t slot = 0;
    uint64_t generation = 0;
    llm_expert_layout_class_id layout_class_id = 0;
};

enum class llm_cold_demand_lookup {
    missing,
    reserved,
    joined_loading,
    ready,
};

struct llm_cold_hot_backing {
    llm_expert_key key = { -1, -1 };
    uint32_t cold_slot = 0;
    uint64_t cold_generation = 0;
    llm_expert_layout_class_id layout_class_id = 0;
};

struct llm_cold_cache_config {
    uint64_t byte_budget = 0;
    uint32_t minimum_slots = 0;
    uint32_t routed_layer_count = 0;
    uint32_t total_expert_keys = 0;
    uint64_t initial_slot_generation_for_testing = 0;
    uint32_t minimum_domain_slots = 1;
    llm_expert_cache_policy_config_internal cache_policy_config = {};
    std::vector<int32_t> routed_layers;
    uint32_t policy_trace_capacity = 4096;
    ggml_backend_buffer_type_t buffer_type = nullptr;
    bool reclaim_free_pages = false;
};

using llm_cold_cache_loader = llm_expert_provider_result (*)(
        void * user_data,
        llm_expert_key key,
        const llm_expert_bundle_descriptor & destination,
        uint32_t slot) noexcept;

struct llm_cold_cache_diagnostics {
    uint64_t requested_bytes = 0;
    uint64_t actual_bytes = 0;
    uint64_t unused_budget_bytes = 0;
    uint64_t bundle_payload_bytes = 0;
    uint64_t aligned_slot_footprint = 0;
    uint32_t layout_class_count = 0;
    std::vector<uint64_t> class_payload_bytes;
    std::vector<uint64_t> class_padding_bytes;
    std::vector<uint64_t> role_offsets;
    std::vector<uint64_t> role_extents;
    uint64_t alignment = 0;
    uint32_t effective_slots = 0;
    bool pageable = false;
    uint64_t requests = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t admissions = 0;
    uint64_t reservations = 0;
    uint64_t publications = 0;
    uint64_t failed_reservations = 0;
    uint64_t evictions = 0;
    uint64_t speculative_admissions = 0;
    uint64_t speculative_replacements = 0;
    uint64_t speculative_rejections = 0;
    uint64_t speculative_demand_consumptions = 0;
    uint64_t source_copy_bundles = 0;
    uint64_t source_copy_bytes = 0;
    uint64_t source_copy_time_us = 0;
    uint64_t failed_copies = 0;
    uint64_t failed_cleanups = 0;
    uint64_t generation_changes = 0;
    uint64_t invariant_failures = 0;
    uint64_t current_hot_refs = 0;
    uint64_t peak_hot_refs = 0;
    uint64_t current_transfer_refs = 0;
    uint64_t peak_transfer_refs = 0;
    uint64_t current_request_refs = 0;
    uint64_t peak_request_refs = 0;
    uint64_t current_cpu_execution_refs = 0;
    uint64_t peak_cpu_execution_refs = 0;
    bool residency_supported = false;
    std::string residency_unavailable_reason;
    uint64_t ready_logical_bytes = 0;
    uint64_t ready_page_count = 0;
    uint64_t resident_ready_page_count = 0;
    uint64_t resident_ready_bytes = 0;
    uint64_t reclaimed_bytes = 0;
    uint64_t reclaim_failures = 0;
    llm_expert_cache_policy_diagnostics policy;
    std::vector<llm_expert_cache_policy_domain_diagnostics> policy_domains;
    std::vector<llm_expert_cache_policy_event> policy_events;

    struct slot {
        llm_expert_key key = { -1, -1 };
        llm_expert_layout_class_id layout_class_id = LLM_EXPERT_LAYOUT_CLASS_INVALID;
        uint64_t generation = 0;
        uint64_t origin_operation_ordinal = 0;
        uint64_t last_use = 0;
        uint32_t hot_refs = 0;
        uint32_t transfer_refs = 0;
        uint32_t request_refs = 0;
        uint32_t cpu_execution_refs = 0;
        llm_expert_residency_origin origin = llm_expert_residency_origin::demand;
        bool speculative_consumed = false;
        uint64_t speculative_deadline = 0;
        uint64_t speculative_utility = 0;
        llm_cold_slot_state state = llm_cold_slot_state::free;
    };
    std::vector<slot> slots;
};

struct llm_cold_cache_scalar_diagnostics {
    bool available = false;
    uint64_t requested_bytes = 0;
    uint64_t actual_bytes = 0;
    uint32_t effective_slots = 0;
    uint32_t occupancy = 0;
    uint64_t requests = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t admissions = 0;
    uint64_t evictions = 0;
    uint64_t residency_digest = 0;
};

struct llm_cold_cache_counter_diagnostics {
    uint64_t requests = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t admissions = 0;
    uint64_t evictions = 0;
};

class llm_cold_expert_cache {
public:
    explicit llm_cold_expert_cache(llm_cold_cache_config config);
    ~llm_cold_expert_cache();

    llm_cold_expert_cache(const llm_cold_expert_cache &) = delete;
    llm_cold_expert_cache & operator=(const llm_cold_expert_cache &) = delete;
    llm_cold_expert_cache(llm_cold_expert_cache &&) noexcept;
    llm_cold_expert_cache & operator=(llm_cold_expert_cache &&) noexcept;

    llm_expert_provider_result initialize(const llm_expert_bundle_descriptor & prototype) noexcept;
    llm_expert_provider_result initialize(const llm_expert_layout_registry & registry) noexcept;
    static llm_expert_provider_result calculate_slot_footprint(
            const llm_expert_bundle_descriptor & prototype,
            ggml_backend_buffer_type_t buffer_type,
            uint64_t & footprint) noexcept;
    llm_expert_provider_result find_or_admit(
            llm_expert_key key,
            const llm_expert_bundle_descriptor & source,
            llm_cold_reference & reference,
            size_t fail_copy_after_tensors = SIZE_MAX) noexcept;
    llm_expert_provider_result find_or_admit_with_loader(
            llm_expert_key key,
            llm_cold_reference & reference,
            llm_cold_cache_loader loader,
            void * loader_data) noexcept;
    llm_expert_provider_result find_or_admit_speculative_with_loader(
            llm_expert_key key,
            uint64_t deadline,
            uint64_t utility,
            llm_cold_reference & reference,
            llm_cold_cache_loader loader,
            void * loader_data) noexcept;
    llm_expert_provider_result reserve_or_find(
            llm_expert_key key,
            llm_cold_reference & reference,
            bool & hit) noexcept;
    llm_expert_provider_result reserve_or_join_demand(
            llm_expert_key key,
            llm_cold_reference & reference,
            llm_cold_demand_lookup & lookup) noexcept;
    llm_expert_provider_result lookup_demand(
            llm_expert_key key,
            llm_cold_reference & reference,
            llm_cold_demand_lookup & lookup) noexcept;
    llm_expert_provider_result wait_until_ready(llm_cold_reference reference) noexcept;
    llm_expert_provider_result reserve_or_find_speculative(
            llm_expert_key key,
            uint64_t deadline,
            uint64_t utility,
            llm_cold_reference & reference,
            bool & hit) noexcept;
    llm_expert_provider_result reclassify(
            llm_cold_reference reference,
            llm_expert_residency_origin origin,
            bool consumed = false) noexcept;
    llm_expert_provider_result publish_ready(llm_expert_key key, llm_cold_reference reference) noexcept;
    llm_expert_provider_result fail_reservation(llm_expert_key key, llm_cold_reference reference) noexcept;
    llm_expert_provider_result acquire(
            llm_cold_reference reference,
            llm_cold_reference_kind kind) noexcept;
    llm_expert_provider_result policy_shadow_hit(
            llm_expert_key key,
            llm_cold_reference reference,
            uint64_t occurrence_count) noexcept;
    llm_expert_provider_result release(
            llm_cold_reference reference,
            llm_cold_reference_kind kind) noexcept;
    llm_expert_provider_result release_many(
            const llm_cold_reference * references,
            size_t reference_count,
            llm_cold_reference_kind kind) noexcept;
    bool ready(llm_cold_reference reference) const noexcept;
    bool contains_ready(llm_expert_key key) const noexcept;
    llm_expert_provider_result retire_ready(llm_cold_reference reference) noexcept;
    llm_expert_provider_result cleanup_failed_slots() noexcept;
    llm_expert_provider_result policy_request_begin() noexcept;
    llm_expert_provider_result policy_set_ubatch_ordinal(uint64_t ordinal) noexcept;
    llm_expert_provider_result policy_phase_transition(llm_expert_cache_policy_phase phase) noexcept;
    llm_expert_provider_result policy_request_end(
        bool success, bool cancelled, bool allow_deferred_terminals = false) noexcept;
    llm_expert_provider_result trim() noexcept;
    llm_expert_provider_result surrender() noexcept;
    llm_expert_provider_result validate_invariants(
            const std::vector<llm_cold_hot_backing> & hot_backings = {}) noexcept;

    const llm_expert_bundle_descriptor & bundle() const noexcept;
    const llm_expert_bundle_descriptor & bundle(llm_expert_layout_class_id layout_class_id) const noexcept;
    const llm_expert_bundle_descriptor & bundle_for_key(llm_expert_key key) const noexcept;
    ggml_backend_buffer_t buffer() const noexcept;
    std::shared_ptr<void> allocation_lease() const noexcept;
    llm_cold_cache_counter_diagnostics counter_diagnostics() const noexcept;
    llm_cold_cache_scalar_diagnostics scalar_diagnostics() const noexcept;
    llm_cold_cache_diagnostics diagnostics() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
