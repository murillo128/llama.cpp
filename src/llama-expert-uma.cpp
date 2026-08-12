#include "llama-expert-uma.h"

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>

#ifdef __linux__
#include <cerrno>
#include <sys/mman.h>
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

llm_expert_system_memory_result llm_expert_system_memory_sample_memory(
        llm_expert_system_memory_sample & output) noexcept {
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
        (void) read_zram_writes(output);
        (void) read_zswap(output);
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

llm_expert_uma_result llm_expert_uma_sample_memory(llm_expert_uma_memory_sample & output) noexcept {
    return llm_expert_system_memory_sample_memory(output);
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

llm_expert_system_memory_result llm_expert_system_memory_calculate_headroom(
        const llm_expert_system_memory_headroom_input & input,
        llm_expert_system_memory_headroom & output) noexcept {
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

llm_expert_uma_result llm_expert_uma_calculate_headroom(
        const llm_expert_uma_headroom_input & input,
        llm_expert_uma_headroom & output) noexcept {
    return llm_expert_system_memory_calculate_headroom(input, output);
}

struct llm_expert_system_memory_budget::impl {
    mutable std::mutex mutex;
    llm_expert_system_memory_sample_fn sample_memory = llm_expert_system_memory_sample_memory;
    uint64_t min_system_headroom_bytes = 0;
    uint64_t min_runtime_headroom_bytes = 0;
    std::vector<llm_expert_system_memory_region> regions;
    llm_expert_system_memory_diagnostics state;

    bool measure_regions(llm_expert_system_memory_sample & sample) noexcept {
        state.model_file_virtual_bytes = 0;
        state.model_file_cache_resident_bytes = 0;
        state.model_file_resident_bytes = 0;
        state.model_allocated_virtual_bytes = 0;
        state.model_allocated_resident_bytes = 0;
        state.residency_unavailable_reason.clear();
#ifdef __linux__
        for (const auto & region : regions) {
            if (region.address == nullptr || region.bytes == 0) continue;
            uint64_t * virtual_bytes = region.file_backed ?
                &state.model_file_virtual_bytes : &state.model_allocated_virtual_bytes;
            if (!checked_add(*virtual_bytes, region.bytes, *virtual_bytes)) {
                state.residency_unavailable_reason = "model region virtual-byte accounting overflow";
                return false;
            }
            if (region.bytes > UINTPTR_MAX - reinterpret_cast<uintptr_t>(region.address)) {
                state.residency_unavailable_reason = "model region address overflow";
                return false;
            }
        }
        const long page_value = sysconf(_SC_PAGESIZE);
        if (page_value <= 0) {
            state.residency_unavailable_reason = "system page size unavailable";
            return false;
        }
        const uintptr_t page = uintptr_t(page_value);
        constexpr size_t chunk_pages = 16384;
        std::vector<unsigned char> residency;
        for (const auto & region : regions) {
            if (!region.file_backed || region.address == nullptr || region.bytes == 0) continue;
            const uintptr_t raw_begin = reinterpret_cast<uintptr_t>(region.address);
            if (region.bytes > UINTPTR_MAX - raw_begin || raw_begin + region.bytes > UINTPTR_MAX - (page - 1)) {
                state.residency_unavailable_reason = "model file region address overflow";
                return false;
            }
            const uintptr_t begin = raw_begin/page*page;
            const uintptr_t end = (raw_begin + region.bytes + page - 1)/page*page;
            for (uintptr_t cursor = begin; cursor < end;) {
                const size_t pages = size_t(std::min<uint64_t>((end - cursor)/page, chunk_pages));
                residency.resize(pages);
                if (mincore(reinterpret_cast<void *>(cursor), pages*size_t(page), residency.data()) != 0) {
                    state.residency_unavailable_reason = std::string("model file mincore failed: errno=") +
                        std::to_string(errno);
                    return false;
                }
                uint64_t resident_pages = 0;
                for (unsigned char value : residency) resident_pages += (value & 1) != 0;
                if (resident_pages > UINT64_MAX/uint64_t(page)) {
                    state.residency_unavailable_reason = "model file cache residency overflow";
                    return false;
                }
                const uint64_t resident_bytes = resident_pages*uint64_t(page);
                if (!checked_add(state.model_file_cache_resident_bytes, resident_bytes,
                        state.model_file_cache_resident_bytes)) {
                    state.residency_unavailable_reason = "model file cache residency overflow";
                    return false;
                }
                cursor += pages*size_t(page);
            }
        }
        std::ifstream smaps("/proc/self/smaps");
        std::string line;
        uintptr_t vma_begin = 0;
        uintptr_t vma_end = 0;
        bool have_vma = false;
        while (std::getline(smaps, line)) {
            const size_t dash = line.find('-');
            const size_t space = line.find(' ');
            if (dash != std::string::npos && space != std::string::npos && dash < space) {
                try {
                    size_t consumed_begin = 0;
                    size_t consumed_end = 0;
                    vma_begin = uintptr_t(std::stoull(line.substr(0, dash), &consumed_begin, 16));
                    vma_end = uintptr_t(std::stoull(line.substr(dash + 1, space - dash - 1), &consumed_end, 16));
                    have_vma = consumed_begin == dash && consumed_end == space - dash - 1 && vma_end > vma_begin;
                } catch (...) {
                    have_vma = false;
                }
                continue;
            }
            if (!have_vma || line.rfind("Rss:", 0) != 0) continue;
            std::istringstream rss_fields(line);
            std::string key, unit;
            uint64_t rss_kib = 0;
            if (!(rss_fields >> key >> rss_kib >> unit) || unit != "kB" ||
                rss_kib > UINT64_MAX/1024) {
                state.residency_unavailable_reason = "smaps RSS field is invalid";
                return false;
            }
            uint64_t file_overlap = 0;
            uint64_t allocated_overlap = 0;
            for (const auto & region : regions) {
                if (region.address == nullptr || region.bytes == 0) continue;
                const uintptr_t region_begin = reinterpret_cast<uintptr_t>(region.address);
                const uintptr_t region_end = region_begin + region.bytes;
                const uintptr_t overlap_begin = std::max(vma_begin, region_begin);
                const uintptr_t overlap_end = std::min(vma_end, region_end);
                if (overlap_end <= overlap_begin) continue;
                uint64_t & overlap = region.file_backed ? file_overlap : allocated_overlap;
                if (!checked_add(overlap, overlap_end - overlap_begin, overlap)) {
                    state.residency_unavailable_reason = "smaps model overlap accounting overflow";
                    return false;
                }
            }
            uint64_t total_overlap = 0;
            if (!checked_add(file_overlap, allocated_overlap, total_overlap)) {
                state.residency_unavailable_reason = "smaps model overlap accounting overflow";
                return false;
            }
            if (total_overlap == 0) continue;
            const uint64_t vma_bytes = vma_end - vma_begin;
            const uint64_t rss_bytes = rss_kib*1024;
            const auto resident_share = [&](uint64_t overlap) {
                return uint64_t(static_cast<long double>(rss_bytes)*overlap/vma_bytes);
            };
            const uint64_t file_resident = resident_share(file_overlap);
            const uint64_t allocated_resident = resident_share(allocated_overlap);
            if (!checked_add(state.model_file_resident_bytes, file_resident,
                    state.model_file_resident_bytes) ||
                !checked_add(state.model_allocated_resident_bytes, allocated_resident,
                    state.model_allocated_resident_bytes)) {
                state.residency_unavailable_reason = "smaps model RSS accounting overflow";
                return false;
            }
        }
        if (!smaps.eof()) {
            state.residency_unavailable_reason = "procfs smaps accounting unavailable";
            return false;
        }
        const uint64_t model_resident = state.model_file_resident_bytes <=
                std::numeric_limits<uint64_t>::max() - state.model_allocated_resident_bytes ?
            state.model_file_resident_bytes + state.model_allocated_resident_bytes :
            std::numeric_limits<uint64_t>::max();
        state.other_process_resident_bytes = sample.process_rss_bytes > model_resident ?
            sample.process_rss_bytes - model_resident : 0;
        return true;
#else
        (void) sample;
        state.residency_unavailable_reason = "Linux mincore model accounting required";
        return regions.empty();
#endif
    }

    llm_expert_system_memory_result sample_current(bool account_regions) noexcept {
        llm_expert_system_memory_sample current;
        auto result = sample_memory ? sample_memory(current) :
            llm_expert_system_memory_result::failure(
                llm_expert_system_memory_error::unavailable_measurement);
        state.current_sample = current;
        if (!result.is_ready()) return result;
        if (account_regions && !measure_regions(current)) {
            return llm_expert_system_memory_result::failure(
                llm_expert_system_memory_error::unavailable_measurement);
        }
        return result;
    }

    llm_expert_system_memory_result reject(const char * reason, bool open_circuit) noexcept {
        state.pressure_rejections++;
        state.pressure_circuit_open = state.pressure_circuit_open || open_circuit;
        state.pressure_rejection_reason = reason;
        return llm_expert_system_memory_result::failure(llm_expert_system_memory_error::unsafe_capacity);
    }

    llm_expert_system_memory_result check_pressure(uint64_t incoming_bytes) noexcept {
        state.pressure_samples++;
        if (state.pressure_circuit_open) return reject("pressure circuit already open", true);
        const auto sampled = sample_current(false);
        if (!sampled.is_ready()) return reject(
            state.current_sample.unavailable_reason.empty() ?
                "required memory capacity telemetry unavailable" :
                state.current_sample.unavailable_reason.c_str(), false);
        const auto & current = state.current_sample;
        const auto & baseline = state.baseline_sample;
        if ((current.process_swap_bytes > baseline.process_swap_bytes) ||
            (current.swap_counters_supported && baseline.swap_counters_supported &&
             (current.cgroup_swap_current_bytes > baseline.cgroup_swap_current_bytes ||
              current.pswpin_pages > baseline.pswpin_pages ||
              current.pswpout_pages > baseline.pswpout_pages)) ||
            (current.psi_full_supported && baseline.psi_full_supported &&
             current.psi_full_total_usec > baseline.psi_full_total_usec) ||
            (current.zram_counters_supported && baseline.zram_counters_supported &&
             current.zram_write_bytes > baseline.zram_write_bytes) ||
            (current.zswap_counters_supported && baseline.zswap_counters_supported &&
             current.zswap_write_pages > baseline.zswap_write_pages)) {
            return reject("swap or full-memory-pressure activity grew", true);
        }
        const uint64_t cgroup_available = current.cgroup_memory_current_bytes <=
                current.cgroup_memory_max_bytes ?
            current.cgroup_memory_max_bytes - current.cgroup_memory_current_bytes : 0;
        const uint64_t available = std::min(current.memory_available_bytes, cgroup_available);
        uint64_t required = 0;
        if (!checked_add(state.headroom.system_reserve_bytes,
                         state.headroom.runtime_reserve_bytes, required) ||
            !checked_add(required, state.hysteresis_bytes, required) ||
            !checked_add(required, incoming_bytes, required)) {
            return reject("pressure reservation arithmetic overflow", false);
        }
        if (available < required) {
            return reject("available memory fell below reserves plus hysteresis", false);
        }
        state.pressure_rejection_reason.clear();
        return llm_expert_system_memory_result::success();
    }
};

llm_expert_system_memory_budget::llm_expert_system_memory_budget() : pimpl(std::make_unique<impl>()) {}
llm_expert_system_memory_budget::~llm_expert_system_memory_budget() = default;
llm_expert_system_memory_budget::llm_expert_system_memory_budget(
        llm_expert_system_memory_budget &&) noexcept = default;
llm_expert_system_memory_budget & llm_expert_system_memory_budget::operator=(
        llm_expert_system_memory_budget &&) noexcept = default;

void llm_expert_system_memory_budget::configure(
        llm_expert_system_memory_sample_fn sample_memory,
        uint64_t min_system_headroom_bytes,
        uint64_t min_runtime_headroom_bytes,
        std::vector<llm_expert_system_memory_region> regions) {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    pimpl->sample_memory = sample_memory ? sample_memory : llm_expert_system_memory_sample_memory;
    pimpl->min_system_headroom_bytes = min_system_headroom_bytes;
    pimpl->min_runtime_headroom_bytes = min_runtime_headroom_bytes;
    pimpl->regions = std::move(regions);
}

llm_expert_system_memory_result llm_expert_system_memory_budget::resolve(
        uint64_t requested_pool_bytes,
        uint64_t slot_stride,
        uint64_t topology_bytes,
        uint64_t minimum_slots,
        uint64_t & selected_pool_bytes) noexcept {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    selected_pool_bytes = 0;
    if (pimpl->state.frozen || slot_stride == 0 || topology_bytes == 0 || minimum_slots == 0) {
        return llm_expert_system_memory_result::failure(
            llm_expert_system_memory_error::invalid_configuration);
    }
    auto sampled = pimpl->sample_current(true);
    if (!sampled.is_ready()) return sampled;
    const auto & current = pimpl->state.current_sample;
    uint64_t accounted_model_committed = 0;
    uint64_t required_model_committed = 0;
    uint64_t measured_non_pool_committed = 0;
    if (!checked_add(pimpl->state.model_file_cache_resident_bytes,
                     pimpl->state.model_allocated_resident_bytes, accounted_model_committed) ||
        !checked_add(pimpl->state.model_file_virtual_bytes,
                     pimpl->state.model_allocated_virtual_bytes, required_model_committed)) {
        return llm_expert_system_memory_result::failure(llm_expert_system_memory_error::overflow);
    }
    const uint64_t other_committed = current.cgroup_memory_current_bytes > accounted_model_committed ?
        current.cgroup_memory_current_bytes - accounted_model_committed : 0;
    if (!checked_add(other_committed, required_model_committed, measured_non_pool_committed)) {
        return llm_expert_system_memory_result::failure(llm_expert_system_memory_error::overflow);
    }
    llm_expert_system_memory_headroom_input input = {
        current.physical_ram_bytes,
        current.cgroup_memory_max_bytes,
        current.cgroup_memory_current_bytes,
        current.memory_available_bytes,
        measured_non_pool_committed,
        0,
        requested_pool_bytes,
        slot_stride,
        pimpl->min_system_headroom_bytes,
        pimpl->min_runtime_headroom_bytes,
    };
    auto result = llm_expert_system_memory_calculate_headroom(input, pimpl->state.headroom);
    if (!result.is_ready()) return result;
    pimpl->state.hysteresis_bytes = std::max<uint64_t>(
        slot_stride <= UINT64_MAX/2 ? 2*slot_stride : UINT64_MAX,
        GIB);
    const uint64_t admission_unrounded = pimpl->state.headroom.safe_pool_bytes >
            pimpl->state.hysteresis_bytes ?
        pimpl->state.headroom.safe_pool_bytes - pimpl->state.hysteresis_bytes : 0;
    pimpl->state.admission_safe_pool_bytes =
        admission_unrounded/slot_stride*slot_stride;
    if (pimpl->state.admission_safe_pool_bytes/slot_stride < minimum_slots ||
        (!pimpl->state.headroom.autofit &&
         pimpl->state.headroom.effective_pool_bytes > pimpl->state.admission_safe_pool_bytes)) {
        return llm_expert_system_memory_result::failure(
            llm_expert_system_memory_error::unsafe_capacity);
    }
    selected_pool_bytes = std::min(
        pimpl->state.headroom.autofit ? pimpl->state.admission_safe_pool_bytes :
            pimpl->state.headroom.effective_pool_bytes,
        topology_bytes);
    selected_pool_bytes = selected_pool_bytes/slot_stride*slot_stride;
    if (selected_pool_bytes/slot_stride < minimum_slots) {
        selected_pool_bytes = 0;
        return llm_expert_system_memory_result::failure(llm_expert_system_memory_error::unsafe_capacity);
    }
    pimpl->state.requested_pool_bytes = requested_pool_bytes;
    pimpl->state.selected_pool_bytes = selected_pool_bytes;
    pimpl->state.topology_bytes = topology_bytes;
    pimpl->state.measured_non_pool_committed_bytes = measured_non_pool_committed;
    pimpl->state.headroom.effective_pool_bytes = selected_pool_bytes;
    pimpl->state.headroom.slot_count = selected_pool_bytes/slot_stride;
    pimpl->state.baseline_sample = current;
    pimpl->state.frozen = true;
    return llm_expert_system_memory_result::success();
}

llm_expert_system_memory_result llm_expert_system_memory_budget::record_runtime_obligation(
        uint64_t bytes) noexcept {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    if (!pimpl->state.frozen || bytes > UINT64_MAX/5) {
        return llm_expert_system_memory_result::failure(
            llm_expert_system_memory_error::invalid_configuration);
    }
    const uint64_t required = (bytes*5 + 3)/4;
    if (required > pimpl->state.headroom.runtime_reserve_bytes) {
        return llm_expert_system_memory_result::failure(llm_expert_system_memory_error::unsafe_capacity);
    }
    pimpl->state.measured_runtime_obligation_bytes = bytes;
    return pimpl->check_pressure(0);
}

llm_expert_system_memory_result llm_expert_system_memory_budget::revalidate() noexcept {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    if (!pimpl->state.frozen) return llm_expert_system_memory_result::success();
    return pimpl->check_pressure(0);
}

llm_expert_system_memory_result llm_expert_system_memory_budget::preflight(
        uint64_t incoming_bytes) noexcept {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    if (!pimpl->state.frozen || incoming_bytes == 0) {
        return llm_expert_system_memory_result::failure(
            llm_expert_system_memory_error::invalid_configuration);
    }
    return pimpl->check_pressure(incoming_bytes);
}

llm_expert_system_memory_diagnostics
llm_expert_system_memory_budget::diagnostics() const noexcept {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    return pimpl->state;
}
