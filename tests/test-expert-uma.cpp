#include "llama-expert-uma.h"

#include <stdexcept>

namespace {

llm_expert_system_memory_sample injected_sample;

llm_expert_system_memory_result sample_injected(llm_expert_system_memory_sample & output) {
    output = injected_sample;
    return llm_expert_system_memory_result::success();
}

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

void test_shared_system_memory_budget() {
    constexpr uint64_t GIB = UINT64_C(1024)*1024*1024;
    injected_sample = {};
    injected_sample.physical_ram_bytes = 128*GIB;
    injected_sample.cgroup_memory_max_bytes = 120*GIB;
    injected_sample.cgroup_memory_current_bytes = 24*GIB;
    injected_sample.memory_available_bytes = 100*GIB;
    injected_sample.process_rss_bytes = 12*GIB;

    llm_expert_system_memory_budget budget;
    budget.configure(sample_injected, 0, 0);
    uint64_t selected = 0;
    require(budget.resolve(0, GIB, 100*GIB, 2, selected).is_ready(),
        "shared AUTO resolve failed");
    require(selected == 66*GIB, "shared AUTO selected the wrong pool");
    const auto & resolved = budget.diagnostics();
    require(resolved.frozen && resolved.headroom.autofit &&
            resolved.measured_non_pool_committed_bytes == 24*GIB &&
            resolved.headroom.safe_pool_bytes == 68*GIB &&
            resolved.admission_safe_pool_bytes == 66*GIB &&
            resolved.hysteresis_bytes == 2*GIB &&
            resolved.remaining_runtime_reserve_bytes == 16*GIB &&
            resolved.calculated_available_bytes == 96*GIB &&
            resolved.required_free_bytes == 30*GIB &&
            resolved.resolve_memory_current_bytes == 24*GIB &&
            resolved.resolve_memory_available_bytes == 100*GIB &&
            resolved.resolve_calculated_available_bytes == 96*GIB &&
            resolved.resolve_required_free_bytes == 30*GIB &&
            resolved.stage == "resolve",
        "shared AUTO diagnostics are incomplete");

    injected_sample.cgroup_memory_current_bytes = 98*GIB;
    injected_sample.memory_available_bytes = 26*GIB;
    injected_sample.process_rss_bytes = 20*GIB;
    require(budget.record_runtime_obligation(4*GIB).is_ready(),
        "bounded runtime obligation was rejected");
    const auto & recorded = budget.diagnostics();
    require(recorded.reported_runtime_obligation_bytes == 4*GIB &&
            recorded.observed_runtime_obligation_bytes == 8*GIB &&
            recorded.measured_runtime_obligation_bytes == 8*GIB &&
            recorded.credited_runtime_obligation_bytes == 10*GIB &&
            recorded.remaining_runtime_reserve_bytes == 6*GIB &&
            recorded.calculated_available_bytes == 22*GIB &&
            recorded.required_free_bytes == 20*GIB &&
            recorded.obligation_memory_current_bytes == 98*GIB &&
            recorded.obligation_memory_available_bytes == 26*GIB &&
            recorded.obligation_calculated_available_bytes == 22*GIB &&
            recorded.obligation_required_free_bytes == 20*GIB &&
            recorded.stage == "record_runtime_obligation",
        "bounded runtime-obligation diagnostics are incomplete");
    require(budget.revalidate("provider_prepare").is_ready(),
        "legitimate runtime commitment was double charged");
    require(budget.preflight(GIB).is_ready(),
        "bounded incoming request was rejected");
    const auto & preflight = budget.diagnostics();
    require(preflight.stage == "cold_cache_preflight" &&
            preflight.incoming_bytes == GIB && preflight.required_free_bytes == 21*GIB,
        "preflight diagnostics are incomplete");

    injected_sample.memory_available_bytes = 23*GIB;
    injected_sample.cgroup_memory_current_bytes = 101*GIB;
    require(budget.revalidate("provider_prepare").error ==
            llm_expert_system_memory_error::unsafe_capacity,
        "external memory commitment borrowed the runtime credit");
    const auto & rejected = budget.diagnostics();
    require(rejected.stage == "provider_prepare" &&
            rejected.calculated_available_bytes == 19*GIB &&
            rejected.required_free_bytes == 20*GIB &&
            rejected.pressure_rejections == 1 &&
            rejected.pressure_rejection_reason ==
                "available memory fell below remaining reserves plus hysteresis" &&
            rejected.headroom.effective_pool_bytes == 66*GIB,
        "external-pressure rejection diagnostics are incomplete");

    injected_sample.cgroup_memory_current_bytes = 24*GIB;
    injected_sample.memory_available_bytes = 100*GIB;
    injected_sample.process_rss_bytes = 12*GIB;
    llm_expert_system_memory_budget explicit_budget;
    explicit_budget.configure(sample_injected, 0, 0);
    require(explicit_budget.resolve(60*GIB, GIB, 100*GIB, 2, selected).is_ready() &&
            selected == 60*GIB && !explicit_budget.diagnostics().headroom.autofit,
        "safe explicit capacity behavior changed");

    llm_expert_system_memory_budget unsafe_explicit_budget;
    unsafe_explicit_budget.configure(sample_injected, 0, 0);
    require(unsafe_explicit_budget.resolve(67*GIB, GIB, 100*GIB, 2, selected).error ==
            llm_expert_system_memory_error::unsafe_capacity,
        "explicit pool above the shared safe cap was accepted");

    llm_expert_system_memory_budget over_budget;
    over_budget.configure(sample_injected, 0, 0);
    require(over_budget.resolve(0, GIB, 100*GIB, 2, selected).is_ready(),
        "over-budget fixture AUTO resolve failed");
    require(over_budget.record_runtime_obligation(13*GIB).error ==
            llm_expert_system_memory_error::unsafe_capacity,
        "over-budget runtime obligation was accepted");
    const auto & over_budget_diagnostics = over_budget.diagnostics();
    require(over_budget_diagnostics.stage == "record_runtime_obligation" &&
            over_budget_diagnostics.credited_runtime_obligation_bytes == 17*GIB - 3*GIB/4 &&
            over_budget_diagnostics.pressure_rejection_reason ==
                "runtime obligation exceeds reserved allowance",
        "over-budget runtime diagnostics are incomplete");

    injected_sample.swap_counters_supported = true;
    injected_sample.psi_full_supported = true;
    injected_sample.psi_full_total_usec = 7;
    llm_expert_system_memory_budget pressure_budget;
    pressure_budget.configure(sample_injected, 0, 0);
    require(pressure_budget.resolve(0, GIB, 100*GIB, 2, selected).is_ready(),
        "pressure fixture AUTO resolve failed");
    injected_sample.psi_full_total_usec++;
    require(pressure_budget.revalidate("provider_prepare").error ==
            llm_expert_system_memory_error::unsafe_capacity,
        "full-memory pressure growth was accepted");
    const auto & pressure = pressure_budget.diagnostics();
    require(pressure.pressure_circuit_open &&
            pressure.pressure_rejection_reason == "swap or full-memory-pressure activity grew",
        "full-memory pressure did not open the circuit");

    injected_sample.psi_full_total_usec = 7;
    injected_sample.process_swap_bytes = 0;
    llm_expert_system_memory_budget swap_budget;
    swap_budget.configure(sample_injected, 0, 0);
    require(swap_budget.resolve(0, GIB, 100*GIB, 2, selected).is_ready(),
        "swap fixture AUTO resolve failed");
    injected_sample.process_swap_bytes = GIB;
    require(swap_budget.revalidate("provider_prepare").error ==
            llm_expert_system_memory_error::unsafe_capacity &&
            swap_budget.diagnostics().pressure_circuit_open,
        "process swap growth did not open the circuit");
}

} // namespace

int main() {
    test_config_copy();
    test_headroom();
    test_native_memory_sample();
    test_shared_system_memory_budget();
    return 0;
}
