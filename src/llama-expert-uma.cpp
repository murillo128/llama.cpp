#include "llama-expert-uma.h"

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>

#ifdef __linux__
#include <sys/resource.h>
#include <sys/sysinfo.h>
#include <unistd.h>
#endif

namespace {

constexpr uint64_t GIB = UINT64_C(1024)*1024*1024;

bool checked_add(uint64_t lhs, uint64_t rhs, uint64_t & result) {
    if (rhs > std::numeric_limits<uint64_t>::max() - lhs) return false;
    result = lhs + rhs;
    return true;
}

bool checked_ceil_ratio(uint64_t value, uint64_t numerator, uint64_t denominator, uint64_t & result) {
    if (denominator == 0 || (value != 0 && numerator > std::numeric_limits<uint64_t>::max()/value)) return false;
    const uint64_t product = value*numerator;
    uint64_t rounded = 0;
    if (!checked_add(product, denominator - 1, rounded)) return false;
    result = rounded/denominator;
    return true;
}

uint64_t digest_config(const llama_expert_uma_config_v1 & value) {
    const auto * bytes = reinterpret_cast<const uint8_t *>(&value);
    uint64_t digest = UINT64_C(1469598103934665603);
    for (size_t i = 0; i < sizeof(value); ++i) {
        digest ^= bytes[i];
        digest *= UINT64_C(1099511628211);
    }
    return digest;
}

#ifdef __linux__
bool read_number(const std::string & path, uint64_t & value) {
    std::ifstream input(path);
    std::string text;
    if (!(input >> text) || text == "max") return false;
    try {
        size_t consumed = 0;
        value = std::stoull(text, &consumed);
        return consumed == text.size();
    } catch (...) {
        return false;
    }
}

bool read_kib_field(const char * path, const char * field, uint64_t & bytes) {
    std::ifstream input(path);
    std::string line;
    while (std::getline(input, line)) {
        std::istringstream fields(line);
        std::string key, unit;
        uint64_t value = 0;
        if (fields >> key >> value >> unit && key == field) {
            if (value > std::numeric_limits<uint64_t>::max()/1024) return false;
            bytes = value*1024;
            return true;
        }
    }
    return false;
}

bool read_vmstat(uint64_t & pswpin, uint64_t & pswpout) {
    std::ifstream input("/proc/vmstat");
    std::string key;
    uint64_t value = 0;
    bool have_in = false, have_out = false;
    while (input >> key >> value) {
        if (key == "pswpin") { pswpin = value; have_in = true; }
        if (key == "pswpout") { pswpout = value; have_out = true; }
    }
    return have_in && have_out;
}

bool read_psi_full(const std::string & path, uint64_t & total_usec) {
    std::ifstream input(path);
    std::string line;
    while (std::getline(input, line)) {
        if (line.rfind("full ", 0) != 0) continue;
        std::istringstream fields(line);
        std::string item;
        while (fields >> item) {
            if (item.rfind("total=", 0) != 0) continue;
            try {
                size_t consumed = 0;
                total_usec = std::stoull(item.substr(6), &consumed);
                return consumed == item.size() - 6;
            } catch (...) {
                return false;
            }
        }
    }
    return false;
}

bool read_zram_writes(llm_expert_uma_memory_sample & output) {
    const std::filesystem::path root("/sys/block");
    std::error_code error;
    for (const auto & entry : std::filesystem::directory_iterator(root, error)) {
        const std::string name = entry.path().filename().string();
        if (name.rfind("zram", 0) != 0) continue;
        output.zram_present = true;
        std::ifstream stat(entry.path()/"stat");
        uint64_t value = 0, sectors_written = 0;
        for (int index = 0; index <= 6; ++index) {
            if (!(stat >> value)) return false;
            if (index == 6) sectors_written = value;
        }
        if (sectors_written > std::numeric_limits<uint64_t>::max()/512 ||
            output.zram_write_bytes > std::numeric_limits<uint64_t>::max() - sectors_written*512) return false;
        output.zram_write_bytes += sectors_written*512;
    }
    if (error) return false;
    output.zram_counters_supported = output.zram_present;
    if (!output.zram_present) output.zram_status_reason = "unsupported: no zram block device";
    return true;
}

bool read_zswap(llm_expert_uma_memory_sample & output) {
    std::ifstream enabled("/sys/module/zswap/parameters/enabled");
    char value = 0;
    if (!(enabled >> value)) {
        output.zswap_status_reason = "unsupported: zswap module parameter unavailable";
        return true;
    }
    output.zswap_enabled = value == 'Y' || value == '1';
    if (!output.zswap_enabled) {
        output.zswap_status_reason = "unsupported: zswap disabled";
        return true;
    }
    if (!read_number("/sys/kernel/debug/zswap/written_back_pages", output.zswap_write_pages)) {
        output.zswap_status_reason = "unavailable: enabled zswap write counter is not readable";
        return false;
    }
    output.zswap_counters_supported = true;
    return true;
}
#endif

} // namespace

llm_expert_uma_result llm_expert_uma_sample_memory(llm_expert_uma_memory_sample & output) noexcept {
    output = {};
#ifdef __linux__
    try {
        struct sysinfo info = {};
        if (sysinfo(&info) != 0 || info.totalram == 0 || info.mem_unit == 0 ||
            uint64_t(info.totalram) > std::numeric_limits<uint64_t>::max()/uint64_t(info.mem_unit)) {
            output.unavailable_reason = "sysinfo physical RAM unavailable";
            return llm_expert_uma_result::failure(llm_expert_uma_error::unavailable_measurement);
        }
        output.physical_ram_bytes = uint64_t(info.totalram)*uint64_t(info.mem_unit);
        if (!read_kib_field("/proc/meminfo", "MemAvailable:", output.memory_available_bytes) ||
            !read_kib_field("/proc/self/status", "VmRSS:", output.process_rss_bytes) ||
            !read_kib_field("/proc/self/status", "VmSwap:", output.process_swap_bytes)) {
            output.unavailable_reason = "procfs memory fields unavailable";
            return llm_expert_uma_result::failure(llm_expert_uma_error::unavailable_measurement);
        }
        std::ifstream cgroup("/proc/self/cgroup");
        std::string line, relative;
        while (std::getline(cgroup, line)) {
            if (line.rfind("0::", 0) == 0) { relative = line.substr(3); break; }
        }
        if (relative.empty()) relative = "/";
        const std::string root = "/sys/fs/cgroup" + (relative == "/" ? "" : relative);
        if (!read_number(root + "/memory.current", output.cgroup_memory_current_bytes)) {
            output.unavailable_reason = "finite cgroup-v2 memory.current unavailable";
            return llm_expert_uma_result::failure(llm_expert_uma_error::unavailable_measurement);
        }
        output.cgroup_v2 = true;
        if (!read_number(root + "/memory.max", output.cgroup_memory_max_bytes)) {
            output.cgroup_memory_max_bytes = output.physical_ram_bytes;
        }
        output.swap_counters_supported = read_number(
            root + "/memory.swap.current", output.cgroup_swap_current_bytes) &&
            read_vmstat(output.pswpin_pages, output.pswpout_pages);
        output.psi_full_supported = read_psi_full(root + "/memory.pressure", output.psi_full_total_usec);
        if (!output.psi_full_supported || !read_zram_writes(output) || !read_zswap(output)) {
            output.unavailable_reason = !output.psi_full_supported ? "cgroup PSI-full counter unavailable" :
                output.zswap_status_reason.empty() ? "zram write counters unavailable" : output.zswap_status_reason;
            return llm_expert_uma_result::failure(llm_expert_uma_error::unavailable_measurement);
        }
        output.nvidia_hmm_status_reason = "unsupported: no stable per-process NVIDIA HMM fault counter exposed";
        struct rusage usage = {};
        if (getrusage(RUSAGE_SELF, &usage) != 0 || usage.ru_majflt < 0) {
            output.unavailable_reason = "getrusage major-fault counter unavailable";
            return llm_expert_uma_result::failure(llm_expert_uma_error::unavailable_measurement);
        }
        output.major_faults = uint64_t(usage.ru_majflt);
        if (!output.swap_counters_supported) output.unavailable_reason = "swap counters unavailable";
        return llm_expert_uma_result::success();
    } catch (...) {
        output = {};
        output.unavailable_reason = "memory sampling failed";
        return llm_expert_uma_result::failure(llm_expert_uma_error::unavailable_measurement);
    }
#else
    output.unavailable_reason = "Linux memory telemetry required";
    return llm_expert_uma_result::failure(llm_expert_uma_error::unavailable_measurement);
#endif
}

llm_expert_uma_result llm_expert_uma_copy_config(
        const llama_expert_uma_config_v1 * source,
        llama_expert_weights_mode mode,
        llm_expert_uma_config_internal & destination) noexcept {
    destination = {};
    destination.value.version = LLAMA_EXPERT_UMA_CONFIG_VERSION_1;
    destination.value.struct_size = sizeof(llama_expert_uma_config_v1);
    destination.value.readiness = LLAMA_EXPERT_UMA_READINESS_AUTO;
    if (source == nullptr) return llm_expert_uma_result::success();
    if (mode != LLAMA_EXPERT_WEIGHTS_MODE_UMA_CACHE ||
        source->version != LLAMA_EXPERT_UMA_CONFIG_VERSION_1 ||
        source->struct_size != sizeof(llama_expert_uma_config_v1) ||
        source->readiness < LLAMA_EXPERT_UMA_READINESS_AUTO ||
        source->readiness >= LLAMA_EXPERT_UMA_READINESS_COUNT || source->flags != 0) {
        return llm_expert_uma_result::failure(llm_expert_uma_error::invalid_configuration);
    }
    for (uint64_t reserved : source->reserved) {
        if (reserved != 0) return llm_expert_uma_result::failure(llm_expert_uma_error::invalid_configuration);
    }
    destination.value = *source;
    destination.supplied = true;
    destination.digest = digest_config(destination.value);
    return llm_expert_uma_result::success();
}

llm_expert_uma_result llm_expert_uma_calculate_headroom(
        const llm_expert_uma_headroom_input & input,
        llm_expert_uma_headroom & output) noexcept {
    output = {};
    if (input.physical_ram_bytes == 0 || input.cgroup_memory_max_bytes == 0 ||
        input.cgroup_memory_current_bytes > input.cgroup_memory_max_bytes ||
        input.memory_available_bytes == 0 || input.slot_stride == 0) {
        return llm_expert_uma_result::failure(llm_expert_uma_error::unavailable_measurement);
    }
    output.effective_limit_bytes = std::min(input.physical_ram_bytes, input.cgroup_memory_max_bytes);
    output.limit_headroom_bytes = output.effective_limit_bytes > input.measured_non_pool_committed_bytes ?
        output.effective_limit_bytes - input.measured_non_pool_committed_bytes : 0;
    const uint64_t cgroup_available = input.cgroup_memory_max_bytes - input.cgroup_memory_current_bytes;
    output.available_headroom_bytes = std::min(input.memory_available_bytes, cgroup_available);

    uint64_t proportional_system_reserve = 0;
    uint64_t proportional_runtime_reserve = 0;
    if (!checked_ceil_ratio(output.effective_limit_bytes, 1, 10, proportional_system_reserve) ||
        !checked_ceil_ratio(input.measured_runtime_delta_bytes, 5, 4, proportional_runtime_reserve)) {
        return llm_expert_uma_result::failure(llm_expert_uma_error::overflow);
    }
    output.system_reserve_bytes = std::max({ input.min_system_headroom_bytes, 8*GIB,
        proportional_system_reserve });
    output.runtime_reserve_bytes = std::max({ input.min_runtime_headroom_bytes, 16*GIB,
        proportional_runtime_reserve });
    uint64_t reserve_total = 0;
    if (!checked_add(output.system_reserve_bytes, output.runtime_reserve_bytes, reserve_total)) {
        return llm_expert_uma_result::failure(llm_expert_uma_error::overflow);
    }
    const uint64_t bound = std::min(output.limit_headroom_bytes, output.available_headroom_bytes);
    const uint64_t unrounded_safe = bound > reserve_total ? bound - reserve_total : 0;
    output.safe_pool_bytes = unrounded_safe/input.slot_stride*input.slot_stride;
    if (output.safe_pool_bytes < input.slot_stride) {
        return llm_expert_uma_result::failure(llm_expert_uma_error::unsafe_capacity);
    }
    output.autofit = input.requested_pool_bytes == 0;
    if (output.autofit) {
        output.effective_pool_bytes = output.safe_pool_bytes;
        output.remainder_bytes = unrounded_safe - output.safe_pool_bytes;
    } else {
        if (input.requested_pool_bytes > output.safe_pool_bytes) {
            return llm_expert_uma_result::failure(llm_expert_uma_error::unsafe_capacity);
        }
        output.effective_pool_bytes = input.requested_pool_bytes/input.slot_stride*input.slot_stride;
        output.remainder_bytes = input.requested_pool_bytes - output.effective_pool_bytes;
        if (output.effective_pool_bytes == 0) {
            return llm_expert_uma_result::failure(llm_expert_uma_error::unsafe_capacity);
        }
    }
    output.slot_count = output.effective_pool_bytes/input.slot_stride;
    return llm_expert_uma_result::success();
}
