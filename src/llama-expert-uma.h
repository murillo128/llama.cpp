#pragma once

#include "llama.h"

#include <cstdint>
#include <string>

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

llm_expert_uma_result llm_expert_uma_sample_memory(llm_expert_uma_memory_sample & output) noexcept;

llm_expert_uma_result llm_expert_uma_calculate_headroom(
        const llm_expert_uma_headroom_input & input,
        llm_expert_uma_headroom & output) noexcept;
