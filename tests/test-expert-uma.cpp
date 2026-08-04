#include "llama-expert-uma.h"

#include <stdexcept>

namespace {

void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

llama_expert_uma_config_v1 valid_config() {
    return { LLAMA_EXPERT_UMA_CONFIG_VERSION_1, sizeof(llama_expert_uma_config_v1),
        LLAMA_EXPERT_UMA_READINESS_AUTO, 0, 9ULL*1024*1024*1024, 17ULL*1024*1024*1024, {} };
}

void test_config_copy() {
    llm_expert_uma_config_internal copied;
    for (llama_expert_weights_mode mode : { LLAMA_EXPERT_WEIGHTS_MODE_DISABLED, LLAMA_EXPERT_WEIGHTS_MODE_RESIDENT,
            LLAMA_EXPERT_WEIGHTS_MODE_HOT_CACHE, LLAMA_EXPERT_WEIGHTS_MODE_COLD_CACHE }) {
        require(llm_expert_uma_copy_config(nullptr, mode, copied).is_ready(), "null config failed");
        require(!copied.supplied && copied.digest == 0, "null config did work");
    }
    auto value = valid_config();
    require(llm_expert_uma_copy_config(&value, LLAMA_EXPERT_WEIGHTS_MODE_UMA_CACHE, copied).is_ready(), "valid config failed");
    require(copied.supplied && copied.digest != 0, "config was not copied");
    value.min_system_headroom_bytes++;
    require(copied.value.min_system_headroom_bytes != value.min_system_headroom_bytes, "caller pointer retained");
    value = valid_config();
    require(!llm_expert_uma_copy_config(&value, LLAMA_EXPERT_WEIGHTS_MODE_COLD_CACHE, copied).is_ready(), "config accepted outside UMA");
    value = valid_config(); value.version++;
    require(!llm_expert_uma_copy_config(&value, LLAMA_EXPERT_WEIGHTS_MODE_UMA_CACHE, copied).is_ready(), "bad version accepted");
    value = valid_config(); value.struct_size--;
    require(!llm_expert_uma_copy_config(&value, LLAMA_EXPERT_WEIGHTS_MODE_UMA_CACHE, copied).is_ready(), "bad size accepted");
    value = valid_config(); value.readiness = LLAMA_EXPERT_UMA_READINESS_COUNT;
    require(!llm_expert_uma_copy_config(&value, LLAMA_EXPERT_WEIGHTS_MODE_UMA_CACHE, copied).is_ready(), "bad readiness accepted");
    value = valid_config(); value.flags = 1;
    require(!llm_expert_uma_copy_config(&value, LLAMA_EXPERT_WEIGHTS_MODE_UMA_CACHE, copied).is_ready(), "bad flags accepted");
    value = valid_config(); value.reserved[0] = 1;
    require(!llm_expert_uma_copy_config(&value, LLAMA_EXPERT_WEIGHTS_MODE_UMA_CACHE, copied).is_ready(), "bad reserved accepted");
}

void test_headroom() {
    constexpr uint64_t GIB = UINT64_C(1024)*1024*1024;
    llm_expert_uma_headroom_input input = { 128*GIB, 120*GIB, 20*GIB, 100*GIB,
        32*GIB, 8*GIB, 0, GIB, 0, 0 };
    llm_expert_uma_headroom output;
    require(llm_expert_uma_calculate_headroom(input, output).is_ready(), "autofit failed");
    require(output.safe_pool_bytes == 60*GIB && output.slot_count == 60 && output.autofit, "autofit wrong");
    input.requested_pool_bytes = 59*GIB + GIB/2;
    require(llm_expert_uma_calculate_headroom(input, output).is_ready(), "explicit capacity failed");
    require(output.effective_pool_bytes == 59*GIB && output.remainder_bytes == GIB/2, "rounding wrong");
    input.requested_pool_bytes = 61*GIB;
    require(llm_expert_uma_calculate_headroom(input, output).error == llm_expert_uma_error::unsafe_capacity,
        "unsafe capacity accepted");
    input.requested_pool_bytes = 0; input.cgroup_memory_max_bytes = 0;
    require(llm_expert_uma_calculate_headroom(input, output).error == llm_expert_uma_error::unavailable_measurement,
        "missing cgroup limit accepted");
}

void test_native_memory_sample() {
    llm_expert_uma_memory_sample sample;
#ifdef __linux__
    require(llm_expert_uma_sample_memory(sample).is_ready(), "Linux memory sample failed");
    require(sample.physical_ram_bytes > 0 && sample.memory_available_bytes > 0 &&
        sample.cgroup_memory_max_bytes > 0 && sample.cgroup_memory_current_bytes > 0 &&
        sample.cgroup_memory_current_bytes <= sample.cgroup_memory_max_bytes && sample.cgroup_v2,
        "Linux memory sample is incomplete");
#else
    require(!llm_expert_uma_sample_memory(sample).is_ready() && !sample.unavailable_reason.empty(),
        "unsupported memory sampling was reported as zero");
#endif
}

} // namespace

int main() {
    test_config_copy();
    test_headroom();
    test_native_memory_sample();
    return 0;
}
