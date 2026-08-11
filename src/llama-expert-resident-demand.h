#pragma once

#include "llama-cold-expert-cache.h"
#include "llama-expert-async-io.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

struct llm_host_resident_demand_entry {
    llm_expert_key key = { -1, -1 };
    llm_cold_reference reference;
    llm_expert_request_handle scheduler_handle;
    uint64_t occurrence_count = 0;
    llm_cold_demand_lookup lookup = llm_cold_demand_lookup::missing;
    bool scheduler_owned = false;
    bool scheduler_joined = false;
    bool request_hold = false;
};

struct llm_host_resident_demand_event {
    uint64_t sequence = 0;
    int32_t layer = -1;
    uint32_t selected_occurrences = 0;
    uint32_t unique_keys = 0;
    uint32_t ready_hits = 0;
    uint32_t joined_loads = 0;
    uint32_t new_reservations = 0;
    uint32_t scheduler_enqueue_attempts = 0;
    uint32_t scheduler_admissions = 0;
    uint32_t scheduler_joins = 0;
    uint32_t read_plans = 0;
    uint32_t read_operations = 0;
    uint64_t read_bytes = 0;
    uint32_t host_ready = 0;
    uint32_t request_holds = 0;
    uint32_t peak_active_read_requests = 0;
    uint32_t peak_active_read_operations = 0;
    uint64_t last_enqueue_us = 0;
    uint64_t last_read_submit_us = 0;
    uint64_t first_wait_us = 0;
    bool first_wait_after_all_enqueue_attempts = false;
    bool first_wait_after_all_admissible_submissions = false;
    bool serial_control = true;
    std::vector<llm_expert_key> semantic_order;
    std::vector<llm_expert_key> physical_completion_order;
};

struct llm_host_resident_demand_batch {
    std::vector<llm_host_resident_demand_entry> entries;
    std::vector<uint32_t> occurrence_to_unique;
    std::vector<uint32_t> semantic_order;
    llm_host_resident_demand_event event;
    size_t occurrence_count = 0;
    size_t unique_count = 0;
    size_t resolved_semantic_count = 0;
    uint64_t transport_epoch = 0;
    bool semantic_order_frozen = false;
    bool finalized = false;
};

struct llm_host_resident_demand_diagnostics {
    uint64_t batches = 0;
    uint64_t failures = 0;
    uint64_t cancellations = 0;
    uint64_t stale_completions = 0;
    uint64_t physical_completion_digest = UINT64_C(1469598103934665603);
    uint64_t semantic_commit_digest = UINT64_C(1469598103934665603);
    uint64_t current_request_holds = 0;
    uint64_t peak_request_holds = 0;
    std::vector<llm_host_resident_demand_event> events;
};

struct llm_host_resident_demand_config {
    llm_cold_expert_cache * cache = nullptr;
    llm_expert_storage * storage = nullptr;
    llm_expert_scheduler * scheduler = nullptr;
    llm_expert_async_transport * transport = nullptr;
    const llm_expert_layout_registry * layout_registry = nullptr;
    llm_expert_integrity_mode integrity_mode = llm_expert_integrity_mode::none;
    uint32_t maximum_occurrences = 0;
    uint32_t maximum_unique_keys = 0;
    uint32_t trace_capacity = 0;
    bool serial_control = true;
    llm_expert_provider_result (*preflight)(void *, uint64_t) = nullptr;
    void * preflight_data = nullptr;
    uint64_t reservation_bytes = 0;
    llm_cold_reference_kind base_hold_kind = llm_cold_reference_kind::request;
};

class llm_host_resident_demand_coordinator {
public:
    explicit llm_host_resident_demand_coordinator(llm_host_resident_demand_config config);
    ~llm_host_resident_demand_coordinator();

    llm_host_resident_demand_coordinator(const llm_host_resident_demand_coordinator &) = delete;
    llm_host_resident_demand_coordinator & operator=(const llm_host_resident_demand_coordinator &) = delete;

    llm_expert_provider_result plan(
            int32_t layer,
            const int32_t * logical_ids,
            size_t logical_count,
            llm_host_resident_demand_batch & batch) noexcept;
    llm_expert_provider_result freeze_semantic_order(
            llm_host_resident_demand_batch & batch) noexcept;
    llm_expert_provider_result resolve_serial_next(
            llm_host_resident_demand_batch & batch,
            bool (*abort_callback)(void *) = nullptr,
            void * abort_data = nullptr) noexcept;
    llm_expert_provider_result finish_serial_batch(
            llm_host_resident_demand_batch & batch) noexcept;
    llm_expert_provider_result fail_serial_batch(
            llm_host_resident_demand_batch & batch,
            llm_expert_provider_error error,
            bool cancelled) noexcept;
    llm_expert_provider_result complete_host_scheduler(
            llm_host_resident_demand_entry & entry) noexcept;
    llm_expert_provider_result fail_host_scheduler(
            llm_host_resident_demand_entry & entry,
            bool cancelled) noexcept;
    llm_expert_provider_result fail_host_schedulers(
            llm_host_resident_demand_batch & batch,
            bool cancelled) noexcept;
    llm_expert_provider_result release_request_hold(
            llm_host_resident_demand_entry & entry) noexcept;
    llm_expert_provider_result transfer_request_hold(
            llm_host_resident_demand_entry & entry) noexcept;
    llm_expert_provider_result transfer_request_hold_to_cpu_execution(
            llm_host_resident_demand_entry & entry) noexcept;
    llm_expert_provider_result release_request_holds(
            llm_host_resident_demand_batch & batch) noexcept;
    llm_host_resident_demand_diagnostics diagnostics() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
