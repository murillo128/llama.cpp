#pragma once

#include "llama.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

enum class llm_expert_uma_error : uint8_t {
    none,
    invalid_configuration,
    overflow,
    unavailable_measurement,
    unsafe_capacity,
};

struct llm_expert_uma_result {
    llm_expert_uma_error error = llm_expert_uma_error::none;
    bool is_ready() const { return error == llm_expert_uma_error::none; }
    static llm_expert_uma_result success() { return {}; }
    static llm_expert_uma_result failure(llm_expert_uma_error error) { return { error }; }
};

struct llm_expert_uma_config_internal {
    llama_expert_uma_config_v1 value = {};
    bool supplied = false;
    uint64_t digest = 0;
};

llm_expert_uma_result llm_expert_uma_copy_config(
        const llama_expert_uma_config_v1 * source,
        llama_expert_weights_mode mode,
        llm_expert_uma_config_internal & destination) noexcept;

struct llm_expert_uma_headroom_input {
    uint64_t physical_ram_bytes = 0;
    uint64_t cgroup_memory_max_bytes = 0;
    uint64_t cgroup_memory_current_bytes = 0;
    uint64_t memory_available_bytes = 0;
    uint64_t measured_non_pool_committed_bytes = 0;
    uint64_t measured_runtime_delta_bytes = 0;
    uint64_t requested_pool_bytes = 0;
    uint64_t slot_stride = 0;
    uint64_t min_system_headroom_bytes = 0;
    uint64_t min_runtime_headroom_bytes = 0;
};

struct llm_expert_uma_headroom {
    uint64_t effective_limit_bytes = 0;
    uint64_t limit_headroom_bytes = 0;
    uint64_t available_headroom_bytes = 0;
    uint64_t system_reserve_bytes = 0;
    uint64_t runtime_reserve_bytes = 0;
    uint64_t safe_pool_bytes = 0;
    uint64_t effective_pool_bytes = 0;
    uint64_t slot_count = 0;
    uint64_t remainder_bytes = 0;
    bool autofit = false;
};

struct llm_expert_uma_memory_sample {
    uint64_t physical_ram_bytes = 0;
    uint64_t memory_available_bytes = 0;
    uint64_t cgroup_memory_max_bytes = 0;
    uint64_t cgroup_memory_current_bytes = 0;
    uint64_t cgroup_swap_current_bytes = 0;
    uint64_t process_rss_bytes = 0;
    uint64_t process_swap_bytes = 0;
    uint64_t major_faults = 0;
    uint64_t pswpin_pages = 0;
    uint64_t pswpout_pages = 0;
    uint64_t psi_full_total_usec = 0;
    uint64_t zram_write_bytes = 0;
    uint64_t zswap_write_pages = 0;
    bool cgroup_v2 = false;
    bool swap_counters_supported = false;
    bool psi_full_supported = false;
    bool zram_present = false;
    bool zram_counters_supported = false;
    bool zswap_enabled = false;
    bool zswap_counters_supported = false;
    bool nvidia_hmm_counters_supported = false;
    std::string zram_status_reason;
    std::string zswap_status_reason;
    std::string nvidia_hmm_status_reason;
    std::string unavailable_reason;
};

// System-memory cold tiers share this sizing and pressure authority.  The UMA
// names above are retained for source compatibility with the Phase 11 probes.
using llm_expert_system_memory_result = llm_expert_uma_result;
using llm_expert_system_memory_error = llm_expert_uma_error;
using llm_expert_system_memory_headroom_input = llm_expert_uma_headroom_input;
using llm_expert_system_memory_headroom = llm_expert_uma_headroom;
using llm_expert_system_memory_sample = llm_expert_uma_memory_sample;
using llm_expert_system_memory_sample_fn =
    llm_expert_system_memory_result (*)(llm_expert_system_memory_sample &);

struct llm_expert_system_memory_region {
    const void * address = nullptr;
    uint64_t bytes = 0;
    bool file_backed = false;
};

struct llm_expert_system_memory_diagnostics {
    llm_expert_system_memory_headroom headroom;
    llm_expert_system_memory_sample baseline_sample;
    llm_expert_system_memory_sample current_sample;
    uint64_t requested_pool_bytes = 0;
    uint64_t selected_pool_bytes = 0;
    uint64_t topology_bytes = 0;
    uint64_t measured_non_pool_committed_bytes = 0;
    uint64_t measured_runtime_obligation_bytes = 0;
    uint64_t reported_runtime_obligation_bytes = 0;
    uint64_t observed_runtime_obligation_bytes = 0;
    uint64_t credited_runtime_obligation_bytes = 0;
    uint64_t remaining_runtime_reserve_bytes = 0;
    uint64_t admission_safe_pool_bytes = 0;
    uint64_t model_file_virtual_bytes = 0;
    uint64_t model_file_cache_resident_bytes = 0;
    uint64_t model_file_resident_bytes = 0;
    uint64_t model_allocated_virtual_bytes = 0;
    uint64_t model_allocated_resident_bytes = 0;
    uint64_t other_process_resident_bytes = 0;
    uint64_t hysteresis_bytes = 0;
    uint64_t calculated_available_bytes = 0;
    uint64_t incoming_bytes = 0;
    uint64_t required_free_bytes = 0;
    uint64_t resolve_memory_current_bytes = 0;
    uint64_t resolve_memory_available_bytes = 0;
    uint64_t resolve_calculated_available_bytes = 0;
    uint64_t resolve_required_free_bytes = 0;
    uint64_t obligation_memory_current_bytes = 0;
    uint64_t obligation_memory_available_bytes = 0;
    uint64_t obligation_calculated_available_bytes = 0;
    uint64_t obligation_required_free_bytes = 0;
    uint64_t pressure_samples = 0;
    uint64_t pressure_rejections = 0;
    bool frozen = false;
    bool pressure_circuit_open = false;
    std::string stage;
    std::string pressure_rejection_reason;
    std::string residency_unavailable_reason;
};

class llm_expert_system_memory_budget {
public:
    llm_expert_system_memory_budget();
    ~llm_expert_system_memory_budget();
    llm_expert_system_memory_budget(llm_expert_system_memory_budget &&) noexcept;
    llm_expert_system_memory_budget & operator=(llm_expert_system_memory_budget &&) noexcept;

    llm_expert_system_memory_budget(const llm_expert_system_memory_budget &) = delete;
    llm_expert_system_memory_budget & operator=(const llm_expert_system_memory_budget &) = delete;

    void configure(
            llm_expert_system_memory_sample_fn sample_memory,
            uint64_t min_system_headroom_bytes,
            uint64_t min_runtime_headroom_bytes,
            std::vector<llm_expert_system_memory_region> regions = {});
    llm_expert_system_memory_result resolve(
            uint64_t requested_pool_bytes,
            uint64_t slot_stride,
            uint64_t topology_bytes,
            uint64_t minimum_slots,
            uint64_t & selected_pool_bytes) noexcept;
    llm_expert_system_memory_result record_runtime_obligation(uint64_t bytes) noexcept;
    llm_expert_system_memory_result revalidate(const char * stage = "revalidate") noexcept;
    llm_expert_system_memory_result preflight(uint64_t incoming_bytes) noexcept;
    llm_expert_system_memory_diagnostics diagnostics() const noexcept;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

llm_expert_uma_result llm_expert_uma_sample_memory(llm_expert_uma_memory_sample & output) noexcept;

llm_expert_system_memory_result llm_expert_system_memory_sample_memory(
        llm_expert_system_memory_sample & output) noexcept;

llm_expert_uma_result llm_expert_uma_calculate_headroom(
        const llm_expert_uma_headroom_input & input,
        llm_expert_uma_headroom & output) noexcept;

llm_expert_system_memory_result llm_expert_system_memory_calculate_headroom(
        const llm_expert_system_memory_headroom_input & input,
        llm_expert_system_memory_headroom & output) noexcept;
