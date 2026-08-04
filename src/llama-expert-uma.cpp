#include "llama-expert-uma.h"

#include <algorithm>
#include <cstddef>
#include <limits>

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

} // namespace

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
