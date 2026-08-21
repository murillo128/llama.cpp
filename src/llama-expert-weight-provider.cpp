#include "llama-expert-weight-provider.h"
#include "llama-perfetto-trace.h"
#include "llama-cold-expert-cache.h"
#include "llama-expert-storage.h"
#include "llama-expert-async-io.h"
#include "llama-expert-resident-demand.h"
#include "llama-expert-scheduler.h"
#include "llama-expert-transfer-ring.h"
#include "llama-hparams.h"

#include "ggml-alloc.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <future>
#include <limits>
#include <map>
#include <mutex>
#include <new>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

llm_expert_device_id llm_expert_owner_device(
        int32_t original_expert_id, uint32_t device_count) noexcept {
    if (original_expert_id < 0 || device_count == 0 || device_count > LLM_EXPERT_MAX_DEVICES) {
        return LLM_EXPERT_DEVICE_ID_INVALID;
    }
    return llm_expert_device_id(uint32_t(original_expert_id) % device_count);
}

llm_expert_provider_result llm_expert_phase8_test_control::arm_gate(
        llm_expert_phase8_test_gate requested_gate,
        llm_expert_key requested_key,
        uint64_t requested_generation) noexcept {
    std::lock_guard<std::mutex> lock(mutex);
    if (kind != directive_kind::none || requested_gate == llm_expert_phase8_test_gate::none ||
        requested_key.layer < 0 || requested_key.expert < 0 || requested_generation == 0) {
        return llm_expert_provider_result::failure(kind != directive_kind::none ?
            llm_expert_provider_error::busy : llm_expert_provider_error::invalid_key);
    }
    kind = directive_kind::gate;
    gate = requested_gate;
    fault = llm_expert_phase8_test_fault::none;
    key = requested_key;
    generation = requested_generation;
    reached = false;
    released = false;
    last_observed_gate = llm_expert_phase8_test_gate::none;
    directive_epoch++;
    return llm_expert_provider_result::success();
}

llm_expert_provider_result llm_expert_phase8_test_control::arm_fault(
        llm_expert_phase8_test_fault requested_fault,
        llm_expert_key requested_key,
        uint64_t requested_generation) noexcept {
    std::lock_guard<std::mutex> lock(mutex);
    if (kind != directive_kind::none || requested_fault == llm_expert_phase8_test_fault::none ||
        requested_key.layer < 0 || requested_key.expert < 0 || requested_generation == 0) {
        return llm_expert_provider_result::failure(kind != directive_kind::none ?
            llm_expert_provider_error::busy : llm_expert_provider_error::invalid_key);
    }
    kind = directive_kind::fault;
    gate = llm_expert_phase8_test_gate::none;
    fault = requested_fault;
    key = requested_key;
    generation = requested_generation;
    reached = false;
    released = false;
    last_observed_gate = llm_expert_phase8_test_gate::none;
    directive_epoch++;
    return llm_expert_provider_result::success();
}

void llm_expert_phase8_test_control::wait_until_reached() noexcept {
    std::unique_lock<std::mutex> lock(mutex);
    condition.wait(lock, [&] { return reached; });
}

llm_expert_provider_result llm_expert_phase8_test_control::release_gate() noexcept {
    std::lock_guard<std::mutex> lock(mutex);
    if (kind != directive_kind::gate || !reached || released) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::invalid_binding);
    }
    released = true;
    released_epoch = directive_epoch;
    condition.notify_all();
    return llm_expert_provider_result::success();
}

llm_expert_provider_result llm_expert_phase8_test_control::release_gate_and_arm_gate(
        llm_expert_phase8_test_gate next_gate) noexcept {
    std::lock_guard<std::mutex> lock(mutex);
    if (kind != directive_kind::gate || !reached || released ||
        next_gate == llm_expert_phase8_test_gate::none || directive_epoch == UINT64_MAX) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::invalid_binding);
    }
    released_epoch = directive_epoch;
    directive_epoch++;
    gate = next_gate;
    reached = false;
    released = false;
    condition.notify_all();
    return llm_expert_provider_result::success();
}

llm_expert_provider_result llm_expert_phase8_test_control::release_gate_and_arm_fault(
        llm_expert_phase8_test_fault next_fault) noexcept {
    std::lock_guard<std::mutex> lock(mutex);
    if (kind != directive_kind::gate || !reached || released ||
        next_fault == llm_expert_phase8_test_fault::none) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::invalid_binding);
    }
    released_epoch = directive_epoch;
    kind = directive_kind::fault;
    gate = llm_expert_phase8_test_gate::none;
    fault = next_fault;
    reached = false;
    released = false;
    condition.notify_all();
    return llm_expert_provider_result::success();
}

bool llm_expert_phase8_test_control::gate_reached() const noexcept {
    std::lock_guard<std::mutex> lock(mutex);
    return reached;
}

uint64_t llm_expert_phase8_test_control::observations() const noexcept {
    std::lock_guard<std::mutex> lock(mutex);
    return observation_count;
}

void llm_expert_phase8_test_control::observe(
        llm_expert_phase8_test_gate observed_gate,
        llm_expert_key observed_key,
        uint64_t observed_generation) noexcept {
    std::lock_guard<std::mutex> lock(mutex);
    if (kind == directive_kind::none || key.layer != observed_key.layer ||
        key.expert != observed_key.expert || generation != observed_generation) {
        return;
    }
    last_observed_gate = observed_gate;
    condition.notify_all();
}

void llm_expert_phase8_test_control::wait_until_observed(
        llm_expert_phase8_test_gate observed_gate) noexcept {
    std::unique_lock<std::mutex> lock(mutex);
    condition.wait(lock, [&] { return last_observed_gate == observed_gate; });
}

bool llm_expert_phase8_test_control::pause_if_armed(
        llm_expert_phase8_test_gate observed_gate,
        llm_expert_key observed_key,
        uint64_t observed_generation) noexcept {
    std::unique_lock<std::mutex> lock(mutex);
    if (kind != directive_kind::gate || gate != observed_gate ||
        key.layer != observed_key.layer || key.expert != observed_key.expert ||
        generation != observed_generation) {
        return false;
    }
    reached = true;
    observation_count++;
    last_observed_gate = observed_gate;
    const uint64_t observed_epoch = directive_epoch;
    condition.notify_all();
    condition.wait(lock, [&] { return released_epoch >= observed_epoch; });
    if (kind == directive_kind::gate && directive_epoch == observed_epoch && gate == observed_gate) {
        kind = directive_kind::none;
        gate = llm_expert_phase8_test_gate::none;
        key = { -1, -1 };
        generation = 0;
    }
    condition.notify_all();
    return true;
}

bool llm_expert_phase8_test_control::consume_fault(
        llm_expert_phase8_test_fault observed_fault,
        llm_expert_key observed_key,
        uint64_t observed_generation) noexcept {
    std::lock_guard<std::mutex> lock(mutex);
    if (kind != directive_kind::fault || fault != observed_fault ||
        key.layer != observed_key.layer || key.expert != observed_key.expert ||
        generation != observed_generation) {
        return false;
    }
    reached = true;
    observation_count++;
    kind = directive_kind::none;
    fault = llm_expert_phase8_test_fault::none;
    key = { -1, -1 };
    generation = 0;
    condition.notify_all();
    return true;
}

llm_expert_auto_result llm_evaluate_expert_auto(const llm_expert_auto_input & input) noexcept {
    llm_expert_auto_result result;
    bool overflow = false;
    const auto add = [&](uint64_t lhs, uint64_t rhs) {
        if (rhs > UINT64_MAX - lhs) {
            overflow = true;
            return UINT64_MAX;
        }
        return lhs + rhs;
    };
    const auto multiply = [&](uint64_t lhs, uint64_t rhs) {
        if (lhs != 0 && rhs > UINT64_MAX/lhs) {
            overflow = true;
            return UINT64_MAX;
        }
        return lhs*rhs;
    };
    const auto transfer = [&](uint64_t bytes) {
        if (input.cost.h2d_bytes_per_second == 0) {
            overflow = true;
            return UINT64_MAX;
        }
        const __uint128_t numerator = __uint128_t(bytes)*1000000000ULL +
            input.cost.h2d_bytes_per_second - 1;
        const __uint128_t value = numerator/input.cost.h2d_bytes_per_second;
        if (value > UINT64_MAX) {
            overflow = true;
            return UINT64_MAX;
        }
        return uint64_t(value);
    };

    const uint64_t cpu_fixed = input.prefill ? input.cost.cpu_fixed_prefill_ns : input.cost.cpu_fixed_decode_ns;
    const uint64_t cpu_per_lane = input.prefill ? input.cost.cpu_per_lane_prefill_ns :
        input.cost.cpu_per_lane_decode_ns;
    const uint64_t gpu_fixed = input.prefill ? input.cost.gpu_fixed_prefill_ns : input.cost.gpu_fixed_decode_ns;
    const uint64_t gpu_per_lane = input.prefill ? input.cost.gpu_per_lane_prefill_ns :
        input.cost.gpu_per_lane_decode_ns;
    result.cpu_work_ns = add(cpu_fixed, multiply(cpu_per_lane, input.lanes));
    const bool valid_same_key =
        (!input.same_key_h2d_present &&
            input.same_key_h2d_state == llm_expert_same_key_h2d_state::none &&
            input.same_key_h2d_remaining_bytes == 0) ||
        (input.same_key_h2d_present &&
            input.same_key_h2d_state == llm_expert_same_key_h2d_state::queued_or_staging &&
            input.same_key_h2d_remaining_bytes == input.bundle_bytes) ||
        (input.same_key_h2d_present &&
            input.same_key_h2d_state == llm_expert_same_key_h2d_state::h2d_in_flight &&
            input.same_key_h2d_remaining_bytes != 0 &&
            input.same_key_h2d_remaining_bytes <= input.bundle_bytes) ||
        (input.same_key_h2d_present &&
            input.same_key_h2d_state == llm_expert_same_key_h2d_state::h2d_complete_unpublished &&
            input.same_key_h2d_remaining_bytes == 0);
    if (!valid_same_key) overflow = true;
    if (input.same_key_h2d_state == llm_expert_same_key_h2d_state::h2d_complete_unpublished) {
        result.h2d_work_ns = 0;
    } else {
        const uint64_t transfer_bytes = input.same_key_h2d_present ?
            input.same_key_h2d_remaining_bytes : input.bundle_bytes;
        result.h2d_work_ns = add(input.cost.h2d_fixed_ns, transfer(transfer_bytes));
    }
    result.gpu_work_ns = add(gpu_fixed, multiply(gpu_per_lane, input.lanes));
    result.cpu_finish_ns = add(input.queued_cpu_work_ns, result.cpu_work_ns);
    result.gpu_finish_ns = add(add(input.queued_h2d_work_ns, result.h2d_work_ns),
        add(input.queued_gpu_work_ns, result.gpu_work_ns));
    const uint64_t cpu_with_hysteresis = add(result.cpu_finish_ns, input.cost.decision_hysteresis_ns);
    result.overflow = overflow;
    if (overflow) {
        result.backend = llm_expert_execution_backend::gpu;
        result.reason = llm_expert_auto_reason::overflow;
    } else if (result.cpu_finish_ns == result.gpu_finish_ns) {
        result.backend = llm_expert_execution_backend::gpu;
        result.reason = llm_expert_auto_reason::tie;
    } else if (cpu_with_hysteresis < result.gpu_finish_ns) {
        result.backend = llm_expert_execution_backend::cpu;
        result.reason = llm_expert_auto_reason::cpu_faster;
    } else {
        result.backend = llm_expert_execution_backend::gpu;
        result.reason = llm_expert_auto_reason::gpu_faster_or_hysteresis;
    }
    return result;
}

namespace {

llm_expert_provider_result system_memory_result(llm_expert_system_memory_result result) {
    if (result.is_ready()) return llm_expert_provider_result::success();
    return llm_expert_provider_result::failure(
        result.error == llm_expert_system_memory_error::unsafe_capacity ?
            llm_expert_provider_error::allocation_failed :
            llm_expert_provider_error::unsupported_configuration);
}

void copy_system_memory_diagnostics(
        const llm_expert_system_memory_diagnostics & source,
        llm_hot_cache_diagnostics & target) {
    target.system_memory_requested_pool_bytes = source.requested_pool_bytes;
    target.system_memory_selected_pool_bytes = source.selected_pool_bytes;
    target.system_memory_safe_pool_bytes = source.headroom.safe_pool_bytes;
    target.system_memory_admission_safe_pool_bytes = source.admission_safe_pool_bytes;
    target.system_memory_effective_limit_bytes = source.headroom.effective_limit_bytes;
    target.system_memory_limit_headroom_bytes = source.headroom.limit_headroom_bytes;
    target.system_memory_available_headroom_bytes = source.headroom.available_headroom_bytes;
    target.system_memory_measured_non_pool_committed_bytes = source.measured_non_pool_committed_bytes;
    target.system_memory_runtime_obligation_bytes = source.measured_runtime_obligation_bytes;
    target.system_memory_reported_runtime_obligation_bytes = source.reported_runtime_obligation_bytes;
    target.system_memory_observed_runtime_obligation_bytes = source.observed_runtime_obligation_bytes;
    target.system_memory_credited_runtime_obligation_bytes = source.credited_runtime_obligation_bytes;
    target.system_memory_remaining_runtime_reserve_bytes = source.remaining_runtime_reserve_bytes;
    target.system_memory_system_reserve_bytes = source.headroom.system_reserve_bytes;
    target.system_memory_runtime_reserve_bytes = source.headroom.runtime_reserve_bytes;
    target.system_memory_hysteresis_bytes = source.hysteresis_bytes;
    target.system_memory_model_file_virtual_bytes = source.model_file_virtual_bytes;
    target.system_memory_model_file_cache_resident_bytes = source.model_file_cache_resident_bytes;
    target.system_memory_model_file_resident_bytes = source.model_file_resident_bytes;
    target.system_memory_model_allocated_virtual_bytes = source.model_allocated_virtual_bytes;
    target.system_memory_model_allocated_resident_bytes = source.model_allocated_resident_bytes;
    target.system_memory_other_process_resident_bytes = source.other_process_resident_bytes;
    target.system_memory_current_bytes = source.current_sample.cgroup_memory_current_bytes;
    target.system_memory_available_bytes = source.current_sample.memory_available_bytes;
    target.system_memory_calculated_available_bytes = source.calculated_available_bytes;
    target.system_memory_incoming_bytes = source.incoming_bytes;
    target.system_memory_required_free_bytes = source.required_free_bytes;
    target.system_memory_selected_pool_slots = source.headroom.slot_count;
    target.system_memory_resolve_current_bytes = source.resolve_memory_current_bytes;
    target.system_memory_resolve_available_bytes = source.resolve_memory_available_bytes;
    target.system_memory_resolve_calculated_available_bytes = source.resolve_calculated_available_bytes;
    target.system_memory_resolve_required_free_bytes = source.resolve_required_free_bytes;
    target.system_memory_obligation_current_bytes = source.obligation_memory_current_bytes;
    target.system_memory_obligation_available_bytes = source.obligation_memory_available_bytes;
    target.system_memory_obligation_calculated_available_bytes = source.obligation_calculated_available_bytes;
    target.system_memory_obligation_required_free_bytes = source.obligation_required_free_bytes;
    target.system_memory_pressure_samples = source.pressure_samples;
    target.system_memory_pressure_rejections = source.pressure_rejections;
    target.system_memory_autofit = source.headroom.autofit;
    target.system_memory_budget_frozen = source.frozen;
    target.system_memory_pressure_circuit_open = source.pressure_circuit_open;
    target.system_memory_stage = source.stage;
    target.system_memory_pressure_rejection_reason = source.pressure_rejection_reason;
    target.system_memory_residency_unavailable_reason = source.residency_unavailable_reason;
}

uint64_t saturating_add_work(uint64_t lhs, uint64_t rhs) noexcept {
    return rhs > UINT64_MAX - lhs ? UINT64_MAX : lhs + rhs;
}

uint64_t auto_gpu_work(
        const llama_expert_auto_cost_model & cost,
        bool prefill,
        uint64_t lanes) noexcept {
    const uint64_t fixed = prefill ? cost.gpu_fixed_prefill_ns : cost.gpu_fixed_decode_ns;
    const uint64_t per_lane = prefill ? cost.gpu_per_lane_prefill_ns : cost.gpu_per_lane_decode_ns;
    if (per_lane != 0 && lanes > UINT64_MAX/per_lane) return UINT64_MAX;
    return saturating_add_work(fixed, per_lane*lanes);
}

uint64_t auto_h2d_work(const llama_expert_auto_cost_model & cost, uint64_t bytes) noexcept {
    if (cost.h2d_bytes_per_second == 0) return UINT64_MAX;
    const __uint128_t numerator = __uint128_t(bytes)*1000000000ULL + cost.h2d_bytes_per_second - 1;
    const __uint128_t transfer = numerator/cost.h2d_bytes_per_second;
    if (transfer > UINT64_MAX) return UINT64_MAX;
    return saturating_add_work(cost.h2d_fixed_ns, uint64_t(transfer));
}

bool same_flight(const llm_expert_flight_id & lhs, const llm_expert_flight_id & rhs) {
    return lhs.transport_epoch == rhs.transport_epoch && lhs.request_slot == rhs.request_slot &&
        lhs.request_generation == rhs.request_generation && lhs.key.layer == rhs.key.layer &&
        lhs.key.expert == rhs.key.expert && lhs.layout_class_id == rhs.layout_class_id &&
        lhs.target_device == rhs.target_device;
}

llm_expert_request_metadata demand_metadata(llm_expert_layout_class_id layout_class_id) {
    llm_expert_request_metadata metadata;
    metadata.layout_class_id = layout_class_id;
    return metadata;
}

uint64_t overlap_pair_hash(
        const llm_expert_async_read_interval & read,
        const llm_transfer_interval & transfer) {
    uint64_t hash = 1469598103934665603ULL;
    const auto append = [&](uint64_t value) {
        for (size_t byte = 0; byte < sizeof(value); ++byte) {
            hash ^= (value >> (byte*8)) & 0xffU;
            hash *= 1099511628211ULL;
        }
    };
    append(read.flight.transport_epoch);
    append(read.flight.request_slot);
    append(read.flight.request_generation);
    append(uint32_t(read.flight.key.layer));
    append(uint32_t(read.flight.key.expert));
    append(read.flight.layout_class_id);
    append(read.operation_index);
    append(transfer.flight.transport_epoch);
    append(transfer.flight.request_slot);
    append(transfer.flight.request_generation);
    append(uint32_t(transfer.flight.key.layer));
    append(uint32_t(transfer.flight.key.expert));
    append(transfer.flight.layout_class_id);
    append(transfer.lane.generation);
    append(transfer.hot_generation);
    return hash;
}

uint64_t overlap_union_us(
        std::vector<std::pair<uint64_t, uint64_t>> lhs,
        std::vector<std::pair<uint64_t, uint64_t>> rhs) {
    const auto merge = [](std::vector<std::pair<uint64_t, uint64_t>> values) {
        values.erase(std::remove_if(values.begin(), values.end(), [](const auto & value) {
            return value.second <= value.first;
        }), values.end());
        std::sort(values.begin(), values.end());
        size_t output = 0;
        for (const auto & value : values) {
            if (output == 0 || value.first > values[output - 1].second) {
                values[output++] = value;
            } else {
                values[output - 1].second = std::max(values[output - 1].second, value.second);
            }
        }
        values.resize(output);
        return values;
    };
    lhs = merge(std::move(lhs));
    rhs = merge(std::move(rhs));
    uint64_t total = 0;
    size_t left = 0, right = 0;
    while (left < lhs.size() && right < rhs.size()) {
        const uint64_t begin = std::max(lhs[left].first, rhs[right].first);
        const uint64_t end = std::min(lhs[left].second, rhs[right].second);
        if (end > begin) total += end - begin;
        if (lhs[left].second < rhs[right].second) left++;
        else right++;
    }
    return total;
}

bool tensor_is_expert_table(const ggml_tensor * tensor, int32_t n_expert) {
    return tensor != nullptr && ggml_n_dims(tensor) >= 3 && tensor->ne[2] == n_expert;
}

bool tensor_is_expert_sidecar(const ggml_tensor * tensor, int32_t n_expert) {
    return tensor == nullptr || (n_expert > 0 && ggml_nelements(tensor) % n_expert == 0);
}

bool projection_is_valid(const llm_expert_projection_descriptor & projection, int32_t n_expert) {
    if (!tensor_is_expert_table(projection.weight, n_expert)) {
        return false;
    }
    if (!tensor_is_expert_sidecar(projection.bias, n_expert) || !tensor_is_expert_sidecar(projection.scale, n_expert)) {
        return false;
    }

    const auto actual_buffer_type = projection.weight->buffer ? ggml_backend_buffer_get_type(projection.weight->buffer) : nullptr;
    return actual_buffer_type == projection.buffer_type;
}

bool projection_identity_matches(
        const llm_expert_projection_descriptor & lhs,
        const llm_expert_projection_descriptor & rhs) {
    return lhs.weight == rhs.weight && lhs.bias == rhs.bias && lhs.scale == rhs.scale && lhs.buffer_type == rhs.buffer_type;
}

bool bundle_identity_matches(
        const llm_expert_bundle_descriptor & lhs,
        const llm_expert_bundle_descriptor & rhs) {
    return lhs.layer == rhs.layer && lhs.n_expert == rhs.n_expert &&
        projection_identity_matches(lhs.up, rhs.up) &&
        projection_identity_matches(lhs.gate, rhs.gate) &&
        projection_identity_matches(lhs.gate_up, rhs.gate_up) &&
        projection_identity_matches(lhs.down, rhs.down);
}

bool binding_identity_matches(
        const llm_expert_graph_binding & binding,
        const llm_expert_bundle_descriptor & bundle) {
    return binding.layer == bundle.layer &&
        projection_identity_matches(binding.up, bundle.up) &&
        projection_identity_matches(binding.gate, bundle.gate) &&
        projection_identity_matches(binding.gate_up, bundle.gate_up) &&
        projection_identity_matches(binding.down, bundle.down) &&
        binding.execution_ids != nullptr && binding.execution_ids->type == GGML_TYPE_I32 &&
        binding.execution_ids->ne[0] > 0 && binding.execution_ids->ne[0] <= bundle.n_expert &&
        binding.execution_ids->ne[1] >= 0;
}

bool buffer_type_is_cuda(ggml_backend_buffer_type_t buft) {
    if (buft == nullptr) {
        return false;
    }
    ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
    if (dev == nullptr || ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
        return false;
    }
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    const char * name = reg ? ggml_backend_reg_name(reg) : nullptr;
    return name != nullptr && std::strcmp(name, "CUDA") == 0;
}

int expert_axis(const ggml_tensor * tensor, int32_t n_expert, bool weight) {
    if (tensor == nullptr) {
        return -1;
    }
    if (weight) {
        return ggml_n_dims(tensor) >= 3 && tensor->ne[2] == n_expert ? 2 : -1;
    }
    int result = -1;
    for (int axis = ggml_n_dims(tensor) - 1; axis >= 0; --axis) {
        if (tensor->ne[axis] == n_expert) {
            if (result >= 0) {
                return -1;
            }
            result = axis;
        }
    }
    return result;
}

bool tensor_layout_matches(
        const ggml_tensor * lhs,
        const ggml_tensor * rhs,
        int32_t n_expert,
        bool weight) {
    if (lhs == nullptr || rhs == nullptr) {
        return lhs == rhs;
    }
    const int lhs_axis = expert_axis(lhs, n_expert, weight);
    const int rhs_axis = expert_axis(rhs, n_expert, weight);
    if (lhs_axis < 0 || lhs_axis != rhs_axis || lhs->type != rhs->type || ggml_n_dims(lhs) != ggml_n_dims(rhs)) {
        return false;
    }
    for (int axis = 0; axis < GGML_MAX_DIMS; ++axis) {
        if (axis != lhs_axis && (lhs->ne[axis] != rhs->ne[axis] || lhs->nb[axis] != rhs->nb[axis])) {
            return false;
        }
    }
    return lhs->nb[lhs_axis] == rhs->nb[rhs_axis];
}

bool projection_layout_matches(
        const llm_expert_projection_descriptor & lhs,
        const llm_expert_projection_descriptor & rhs,
        int32_t n_expert) {
    return tensor_layout_matches(lhs.weight, rhs.weight, n_expert, true) &&
        tensor_layout_matches(lhs.bias, rhs.bias, n_expert, false) &&
        tensor_layout_matches(lhs.scale, rhs.scale, n_expert, false) &&
        lhs.buffer_type == rhs.buffer_type;
}

bool bundle_layout_matches(
        const llm_expert_bundle_descriptor & lhs,
        const llm_expert_bundle_descriptor & rhs) {
    return lhs.n_expert == rhs.n_expert && lhs.uses_merged_gate_up() == rhs.uses_merged_gate_up() &&
        projection_layout_matches(lhs.up, rhs.up, lhs.n_expert) &&
        projection_layout_matches(lhs.gate, rhs.gate, lhs.n_expert) &&
        projection_layout_matches(lhs.gate_up, rhs.gate_up, lhs.n_expert) &&
        projection_layout_matches(lhs.down, rhs.down, lhs.n_expert);
}

void append_layout_u64(std::vector<uint8_t> & encoding, uint64_t value) {
    for (size_t byte = 0; byte < sizeof(value); ++byte) {
        encoding.push_back(uint8_t(value >> (byte*8)));
    }
}

void append_layout_string(std::vector<uint8_t> & encoding, const char * value) {
    const size_t length = value == nullptr ? 0 : std::strlen(value);
    append_layout_u64(encoding, length);
    if (length != 0) encoding.insert(encoding.end(), value, value + length);
}

bool append_layout_tensor(
        std::vector<uint8_t> & encoding,
        const ggml_tensor * tensor,
        ggml_backend_buffer_type_t declared_buffer_type,
        int32_t n_expert,
        bool weight) {
    append_layout_u64(encoding, tensor != nullptr);
    if (tensor == nullptr) return true;
    const int axis = expert_axis(tensor, n_expert, weight);
    if (axis < 0) return false;
    append_layout_u64(encoding, uint32_t(tensor->type));
    append_layout_u64(encoding, ggml_n_dims(tensor));
    append_layout_u64(encoding, axis);
    for (int index = 0; index < GGML_MAX_DIMS; ++index) append_layout_u64(encoding, tensor->ne[index]);
    for (int index = 0; index < GGML_MAX_DIMS; ++index) append_layout_u64(encoding, tensor->nb[index]);
    append_layout_u64(encoding, tensor->nb[axis]);
    const auto buft = tensor->buffer == nullptr ? declared_buffer_type : ggml_backend_buffer_get_type(tensor->buffer);
    append_layout_string(encoding, buft == nullptr ? "METADATA_ONLY" : ggml_backend_buft_name(buft));
    append_layout_u64(encoding, buft == nullptr ? 0 : ggml_backend_buft_get_alignment(buft));
    return true;
}

bool canonical_layout_encoding(
        const llm_expert_bundle_descriptor & bundle,
        ggml_backend_buffer_type_t target_buffer_type,
        std::vector<uint8_t> & encoding) {
    encoding.clear();
    append_layout_u64(encoding, 1);
    append_layout_u64(encoding, bundle.n_expert);
    append_layout_u64(encoding, bundle.uses_merged_gate_up());
    uint64_t role = 0;
    for (const auto * projection : { &bundle.up, &bundle.gate, &bundle.gate_up, &bundle.down }) {
        append_layout_u64(encoding, role++);
        if (!append_layout_tensor(encoding, projection->weight, projection->buffer_type, bundle.n_expert, true) ||
            !append_layout_tensor(encoding, projection->bias, projection->buffer_type, bundle.n_expert, false) ||
            !append_layout_tensor(encoding, projection->scale, projection->buffer_type, bundle.n_expert, false)) {
            return false;
        }
    }
    append_layout_u64(encoding, 0); // runtime transform NONE
    append_layout_string(encoding,
        target_buffer_type == nullptr ? nullptr : ggml_backend_buft_name(target_buffer_type));
    append_layout_u64(encoding,
        target_buffer_type == nullptr ? 0 : ggml_backend_buft_get_alignment(target_buffer_type));
    return true;
}

uint64_t layout_encoding_digest(const std::vector<uint8_t> & encoding) {
    uint64_t digest = 1469598103934665603ULL;
    for (uint8_t value : encoding) {
        digest ^= value;
        digest *= 1099511628211ULL;
    }
    return digest;
}

bool tensor_logical_signature_matches(
        const ggml_tensor * lhs,
        const ggml_tensor * rhs,
        int32_t n_expert,
        bool weight) {
    if (lhs == nullptr || rhs == nullptr) return lhs == rhs;
    const int lhs_axis = expert_axis(lhs, n_expert, weight);
    const int rhs_axis = expert_axis(rhs, n_expert, weight);
    if (lhs_axis < 0 || lhs_axis != rhs_axis || ggml_n_dims(lhs) != ggml_n_dims(rhs)) return false;
    for (int axis = 0; axis < GGML_MAX_DIMS; ++axis) {
        if (axis != lhs_axis && lhs->ne[axis] != rhs->ne[axis]) return false;
    }
    return true;
}

bool projection_logical_signature_matches(
        const llm_expert_projection_descriptor & lhs,
        const llm_expert_projection_descriptor & rhs,
        int32_t n_expert) {
    return tensor_logical_signature_matches(lhs.weight, rhs.weight, n_expert, true) &&
        tensor_logical_signature_matches(lhs.bias, rhs.bias, n_expert, false) &&
        tensor_logical_signature_matches(lhs.scale, rhs.scale, n_expert, false);
}

bool bundle_logical_signature_matches(
        const llm_expert_bundle_descriptor & lhs,
        const llm_expert_bundle_descriptor & rhs) {
    return lhs.n_expert == rhs.n_expert && lhs.uses_merged_gate_up() == rhs.uses_merged_gate_up() &&
        projection_logical_signature_matches(lhs.up, rhs.up, lhs.n_expert) &&
        projection_logical_signature_matches(lhs.gate, rhs.gate, lhs.n_expert) &&
        projection_logical_signature_matches(lhs.gate_up, rhs.gate_up, lhs.n_expert) &&
        projection_logical_signature_matches(lhs.down, rhs.down, lhs.n_expert);
}

bool projection_is_host_accessible(const llm_expert_projection_descriptor & projection) {
    const std::array<ggml_tensor *, 3> tensors = { projection.weight, projection.bias, projection.scale };
    for (const auto * tensor : tensors) {
        if (tensor != nullptr && (tensor->buffer == nullptr || !ggml_backend_buffer_is_host(tensor->buffer))) {
            return false;
        }
    }
    return true;
}

bool projection_is_pageable_cpu(const llm_expert_projection_descriptor & projection) {
    const std::array<ggml_tensor *, 3> tensors = { projection.weight, projection.bias, projection.scale };
    for (const auto * tensor : tensors) {
        if (tensor == nullptr) {
            continue;
        }
        if (tensor->buffer == nullptr || !ggml_backend_buffer_is_host(tensor->buffer)) {
            return false;
        }
        const auto buft = ggml_backend_buffer_get_type(tensor->buffer);
        const auto dev = buft ? ggml_backend_buft_get_device(buft) : nullptr;
        if (dev != nullptr && ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU) {
            return false;
        }
    }
    return true;
}

struct storage_load_context {
    llm_expert_storage * storage = nullptr;
    llm_expert_async_transport * transport = nullptr;
    llm_expert_scheduler * scheduler = nullptr;
    llm_expert_integrity_mode integrity_mode = llm_expert_integrity_mode::none;
    std::unique_lock<std::mutex> * provider_lock = nullptr;
    llm_expert_request_handle completed_io_handle;
    bool completed_io_pending_publication = false;
    bool (*abort_callback)(void *) = nullptr;
    void * abort_callback_data = nullptr;
    const std::vector<llm_expert_layout_class_id> * layer_ids = nullptr;
    const llm_expert_layout_registry * layout_registry = nullptr;
    llm_expert_device_id target_device = 0;
};

bool finalize_payload_integrity(
        llm_expert_integrity_mode mode,
        llm_expert_storage & storage,
        const llm_expert_storage_destination * destinations,
        size_t destination_count,
        llm_expert_integrity_status transport_status,
        uint64_t transport_digest) {
    if (mode == llm_expert_integrity_mode::none) {
        LLM_EXPERT_TRACE_SCOPE("k3.provider", "integrity_finalize", "integrity_checked", false);
        if (transport_status != llm_expert_integrity_status::not_checked || transport_digest != 0) {
            storage.poison();
            return false;
        }
        storage.record_integrity_status(llm_expert_integrity_status::not_checked);
        return true;
    }

    if (transport_status != llm_expert_integrity_status::passed) {
        storage.record_integrity_status(llm_expert_integrity_status::failed);
        return false;
    }
    uint64_t destination_digest = 1469598103934665603ULL;
    uint64_t digest_bytes = 0;
    {
        LLM_EXPERT_TRACE_SCOPE("k3.provider", "integrity_digest", "destination_count", destination_count);
        for (size_t index = 0; index < destination_count; ++index) {
            const auto * bytes = static_cast<const uint8_t *>(destinations[index].data);
            digest_bytes += destinations[index].extent;
            for (uint64_t offset = 0; offset < destinations[index].extent; ++offset) {
                destination_digest ^= bytes[offset];
                destination_digest *= 1099511628211ULL;
            }
        }
    }
    const bool matches = destination_digest == transport_digest;
    {
        LLM_EXPERT_TRACE_SCOPE("k3.provider", "integrity_finalize", "integrity_checked", true,
            "integrity_matches", matches, "digest_bytes", digest_bytes);
        storage.record_integrity_status(matches ? llm_expert_integrity_status::passed :
            llm_expert_integrity_status::failed, digest_bytes);
    }
    return matches;
}

struct storage_async_flight {
    llm_expert_key key = { -1, -1 };
    llm_cold_reference cold;
    llm_expert_request_handle handle;
    llm_expert_request_handle deferred_release_handle;
    llm_expert_flight_id flight_id;
    std::array<llm_expert_storage_destination, 12> destinations;
    std::array<llm_expert_storage_read_operation, 12> operations;
    size_t destination_count = 0;
    size_t operation_count = 0;
    uint64_t scheduler_enqueue_ns = 0;
    bool cold_hit = false;
    bool direct_storage = false;
    bool direct_storage_completed = false;
    bool hot_prepared = false;
    llm_transfer_lane_reference lane;
    bool reserved = false;
    bool submitted = false;
    bool read_active = false;
    bool read_completed = false;
    bool scheduler_active = false;
    bool scheduler_joined = false;
    bool scheduler_deferred_retry = false;
    bool scheduler_taken = false;
    bool predictive_storage_joined = false;
    bool predictive_hot_joined = false;
    uint64_t prediction_sequence = 0;
    uint32_t predictive_hot_slot = UINT32_MAX;
    uint64_t predictive_hot_generation = 0;
    bool processed = false;
    bool h2d_submitted = false;
    bool device_ready = false;
    llm_expert_request_state scheduler_state = llm_expert_request_state::free;
};

struct predictive_storage_record {
    storage_async_flight flight;
    uint64_t prediction_sequence = 0;
    uint64_t deadline_token = 0;
    int32_t target_layer = -1;
    uint64_t utility = 0;
    bool active = false;
    bool demand_claimed = false;
};

bool append_storage_destination(
        std::array<llm_expert_storage_destination, 12> & destinations,
        size_t & count,
        ggml_tensor * tensor,
        const ggml_tensor * layout_tensor,
        int32_t target_n_expert,
        int32_t layout_n_expert,
        uint32_t slot,
        bool weight,
        llm_expert_storage_projection projection,
        llm_expert_storage_sidecar sidecar,
        llm_expert_layout_class_id layout_class_id) {
    if (tensor == nullptr || layout_tensor == nullptr) return tensor == layout_tensor;
    const int axis = expert_axis(tensor, target_n_expert, weight);
    const int layout_axis = expert_axis(layout_tensor, layout_n_expert, weight);
    if (axis < 0 || layout_axis != axis || tensor->data == nullptr ||
        tensor->nb[axis] < layout_tensor->nb[layout_axis] ||
        slot >= uint32_t(tensor->ne[axis]) || count >= destinations.size()) {
        return false;
    }
    for (int upper = axis + 1; upper < ggml_n_dims(tensor); ++upper) {
        if (tensor->ne[upper] != 1) {
            return false;
        }
    }
    destinations[count++] = {
        projection,
        sidecar,
        static_cast<uint8_t *>(tensor->data) + size_t(slot)*tensor->nb[axis],
        layout_tensor->nb[layout_axis],
        layout_class_id,
    };
    return true;
}

bool append_storage_projection(
        std::array<llm_expert_storage_destination, 12> & destinations,
        size_t & count,
        const llm_expert_projection_descriptor & projection,
        const llm_expert_projection_descriptor & layout_projection,
        int32_t target_n_expert,
        int32_t layout_n_expert,
        uint32_t slot,
        llm_expert_storage_projection identity,
        llm_expert_layout_class_id layout_class_id = 0) {
    return append_storage_destination(destinations, count, projection.weight, layout_projection.weight,
               target_n_expert, layout_n_expert, slot, true,
               identity, llm_expert_storage_sidecar::weight, layout_class_id) &&
        append_storage_destination(destinations, count, projection.bias, layout_projection.bias,
               target_n_expert, layout_n_expert, slot, false,
               identity, llm_expert_storage_sidecar::bias, layout_class_id) &&
        append_storage_destination(destinations, count, projection.scale, layout_projection.scale,
               target_n_expert, layout_n_expert, slot, false,
               identity, llm_expert_storage_sidecar::scale, layout_class_id);
}

bool build_storage_destinations(
        storage_async_flight & flight,
        const llm_expert_bundle_descriptor & destination,
        const llm_expert_bundle_descriptor & layout,
        uint32_t slot) {
    flight.destination_count = 0;
    const auto layout_class_id = flight.cold.layout_class_id;
    return append_storage_projection(flight.destinations, flight.destination_count,
               destination.up, layout.up, destination.n_expert, layout.n_expert, slot,
               llm_expert_storage_projection::up, layout_class_id) &&
        append_storage_projection(flight.destinations, flight.destination_count,
               destination.gate, layout.gate, destination.n_expert, layout.n_expert, slot,
               llm_expert_storage_projection::gate, layout_class_id) &&
        append_storage_projection(flight.destinations, flight.destination_count,
               destination.gate_up, layout.gate_up, destination.n_expert, layout.n_expert, slot,
               llm_expert_storage_projection::gate_up, layout_class_id) &&
        append_storage_projection(flight.destinations, flight.destination_count,
               destination.down, layout.down, destination.n_expert, layout.n_expert, slot,
               llm_expert_storage_projection::down, layout_class_id);
}

llm_expert_provider_result load_storage_bundle(
        void * user_data,
        llm_expert_key key,
        const llm_expert_bundle_descriptor & destination,
        uint32_t slot) noexcept {
    auto * context = static_cast<storage_load_context *>(user_data);
    std::array<llm_expert_storage_destination, 12> destinations;
    size_t count = 0;
    const auto layout_class_id = context != nullptr && context->layer_ids != nullptr && key.layer >= 0 &&
            size_t(key.layer) < context->layer_ids->size() ? (*context->layer_ids)[size_t(key.layer)] :
        LLM_EXPERT_LAYOUT_CLASS_INVALID;
    const auto * layout = context != nullptr && context->layout_registry != nullptr &&
            layout_class_id < context->layout_registry->classes.size() ?
        &context->layout_registry->classes[layout_class_id].prototype : nullptr;
    if (context == nullptr || context->storage == nullptr ||
        layout_class_id == LLM_EXPERT_LAYOUT_CLASS_INVALID || layout == nullptr ||
        !append_storage_projection(destinations, count, destination.up, layout->up,
            destination.n_expert, layout->n_expert, slot,
            llm_expert_storage_projection::up, layout_class_id) ||
        !append_storage_projection(destinations, count, destination.gate, layout->gate,
            destination.n_expert, layout->n_expert, slot,
            llm_expert_storage_projection::gate, layout_class_id) ||
        !append_storage_projection(destinations, count, destination.gate_up, layout->gate_up,
            destination.n_expert, layout->n_expert, slot,
            llm_expert_storage_projection::gate_up, layout_class_id) ||
        !append_storage_projection(destinations, count, destination.down, layout->down,
            destination.n_expert, layout->n_expert, slot,
            llm_expert_storage_projection::down, layout_class_id)) {
        if (context && context->storage) context->storage->poison();
        return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
    }
    if (context->transport != nullptr && context->scheduler != nullptr) {
        std::array<llm_expert_storage_read_operation, 12> operations;
        size_t operation_count = 0;
        const auto planned = context->storage->make_read_plan(
            key, destinations.data(), count, operations.data(), operations.size(), operation_count);
        if (!planned.is_ready()) {
            context->storage->poison();
            return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
        }
        llm_expert_request_metadata metadata;
        metadata.layout_class_id = layout_class_id;
        metadata.target_device = context->target_device;
        const auto scheduled = context->scheduler->enqueue(
            key, llm_expert_priority::demand_current_layer, llm_expert_readiness::host_ready, metadata);
        if (scheduled.disposition != llm_expert_schedule_disposition::admitted) {
            return llm_expert_provider_result::failure(
                scheduled.disposition == llm_expert_schedule_disposition::generation_exhausted ?
                    llm_expert_provider_error::generation_exhausted : llm_expert_provider_error::busy);
        }
        llm_expert_request_snapshot snapshot;
        const auto selected = context->scheduler->take_next(snapshot);
        if (selected.disposition != llm_expert_schedule_disposition::admitted ||
            selected.handle.slot != scheduled.handle.slot ||
            selected.handle.generation != scheduled.handle.generation ||
            selected.handle.target_device != scheduled.handle.target_device) {
            if (selected.disposition == llm_expert_schedule_disposition::admitted) {
                (void) context->scheduler->transition(selected.handle, llm_expert_request_state::submitting,
                    llm_expert_request_state::draining);
                (void) context->scheduler->finish(selected.handle, llm_expert_request_state::failed);
                (void) context->scheduler->release_terminal(selected.handle);
            }
            if (selected.handle.slot != scheduled.handle.slot ||
                selected.handle.generation != scheduled.handle.generation ||
                selected.handle.target_device != scheduled.handle.target_device) {
                (void) context->scheduler->transition(scheduled.handle, llm_expert_request_state::queued,
                    llm_expert_request_state::draining);
                (void) context->scheduler->finish(scheduled.handle, llm_expert_request_state::failed);
                (void) context->scheduler->release_terminal(scheduled.handle);
            }
            return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
        }
        const llm_expert_async_operation_identity identity = {
            context->transport->diagnostics().transport_epoch,
            selected.handle,
            0,
            key,
            llm_expert_readiness::host_ready,
            llm_expert_priority::demand_current_layer,
            layout_class_id,
        };
        const auto submitted = context->transport->submit_read_plan(identity, operations.data(), operation_count);
        if (submitted != llm_expert_async_result::ready) {
            (void) context->scheduler->transition(selected.handle, llm_expert_request_state::submitting,
                llm_expert_request_state::draining);
            (void) context->scheduler->finish(selected.handle, llm_expert_request_state::failed);
            (void) context->scheduler->release_terminal(selected.handle);
            return llm_expert_provider_result::failure(llm_expert_provider_error::copy_failed);
        }
        if (context->scheduler->transition(selected.handle, llm_expert_request_state::submitting,
                llm_expert_request_state::io_in_flight) != llm_expert_schedule_disposition::admitted) {
            (void) context->transport->cancel_read(selected.handle);
            llm_expert_async_read_completion discarded;
            if (context->provider_lock != nullptr) context->provider_lock->unlock();
            (void) context->transport->wait_read(selected.handle, discarded);
            if (context->provider_lock != nullptr) context->provider_lock->lock();
            (void) context->transport->release_read(selected.handle);
            (void) context->scheduler->transition(selected.handle, llm_expert_request_state::submitting,
                llm_expert_request_state::draining);
            (void) context->scheduler->finish(selected.handle, llm_expert_request_state::failed);
            (void) context->scheduler->release_terminal(selected.handle);
            return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
        }
        llm_expert_async_read_completion completion;
        if (context->provider_lock != nullptr) context->provider_lock->unlock();
        const auto waited = context->transport->wait_read(
            selected.handle, completion, context->abort_callback, context->abort_callback_data);
        if (context->provider_lock != nullptr) context->provider_lock->lock();
        const auto released = context->transport->release_read(selected.handle);
        const llm_expert_storage_error storage_error = waited == llm_expert_async_result::ready ?
            llm_expert_storage_error::none :
            (waited == llm_expert_async_result::closed ? llm_expert_storage_error::cancelled :
             (completion.native_error == 0 ? llm_expert_storage_error::short_read :
              llm_expert_storage_error::io_error));
        context->storage->record_async_read(count, completion.bytes_completed, storage_error, completion.native_error);
        bool integrity_matches = false;
        if (waited == llm_expert_async_result::ready && released == llm_expert_async_result::ready) {
            integrity_matches = finalize_payload_integrity(context->integrity_mode, *context->storage,
                destinations.data(), count, completion.integrity_status, completion.digest);
        }
        if (waited == llm_expert_async_result::ready && released == llm_expert_async_result::ready &&
            integrity_matches) {
            context->completed_io_handle = selected.handle;
            context->completed_io_pending_publication = true;
            return llm_expert_provider_result::success();
        }
        if (waited == llm_expert_async_result::closed) {
            (void) context->scheduler->begin_demand_cancellation(
                selected.handle, llm_expert_request_state::io_in_flight);
            (void) context->scheduler->transition(selected.handle, llm_expert_request_state::cancelling,
                llm_expert_request_state::draining);
            (void) context->scheduler->finish(selected.handle, llm_expert_request_state::cancelled);
            (void) context->scheduler->release_terminal(selected.handle);
            return llm_expert_provider_result::failure(llm_expert_provider_error::cancelled);
        }
        (void) context->scheduler->transition(selected.handle, llm_expert_request_state::io_in_flight,
            llm_expert_request_state::draining);
        (void) context->scheduler->finish(selected.handle, llm_expert_request_state::failed);
        (void) context->scheduler->release_terminal(selected.handle);
        return llm_expert_provider_result::failure(
            waited == llm_expert_async_result::ready &&
                released == llm_expert_async_result::ready && !integrity_matches ?
                llm_expert_provider_error::metadata_mismatch :
            (waited == llm_expert_async_result::stale_generation ?
                llm_expert_provider_error::stale_generation : llm_expert_provider_error::copy_failed));
    }
    const auto result = context->storage->read_bundle(
        key, destinations.data(), count, context->abort_callback, context->abort_callback_data);
    if (result.is_ready()) {
        const bool matches = finalize_payload_integrity(context->integrity_mode, *context->storage,
            destinations.data(), count, result.integrity_status, result.digest);
        return matches ? llm_expert_provider_result::success() :
            llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
    }
    if (result.error == llm_expert_storage_error::cancelled) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::cancelled);
    }
    if (result.error == llm_expert_storage_error::invalid_key ||
        result.error == llm_expert_storage_error::invalid_destination ||
        result.error == llm_expert_storage_error::invalid_directory ||
        result.error == llm_expert_storage_error::poisoned) {
        context->storage->poison();
        return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
    }
    return llm_expert_provider_result::failure(llm_expert_provider_error::copy_failed);
}

} // namespace

bool llm_expert_provider_result::is_ready() const {
    return status == llm_expert_provider_status::ready && error == llm_expert_provider_error::none;
}

llm_expert_provider_result llm_expert_provider_result::success() {
    return {};
}

llm_expert_provider_result llm_expert_provider_result::failure(llm_expert_provider_error error) {
    llm_expert_provider_status status = llm_expert_provider_status::failed;
    if (error == llm_expert_provider_error::allocation_failed) {
        status = llm_expert_provider_status::allocation_failed;
    } else if (error == llm_expert_provider_error::cancelled) {
        status = llm_expert_provider_status::cancelled;
    }
    return { status, error };
}

bool llm_expert_key::is_valid(int32_t n_layer, int32_t n_expert) const {
    return layer >= 0 && layer < n_layer && expert >= 0 && expert < n_expert;
}

bool llm_expert_speculative_victim_precedes(
        uint64_t candidate_deadline,
        uint64_t candidate_utility,
        uint32_t candidate_slot,
        uint64_t incumbent_deadline,
        uint64_t incumbent_utility,
        uint32_t incumbent_slot) noexcept {
    return candidate_deadline < incumbent_deadline ||
        (candidate_deadline == incumbent_deadline &&
         (candidate_utility < incumbent_utility ||
          (candidate_utility == incumbent_utility && candidate_slot < incumbent_slot)));
}

llm_expert_projection_descriptor llm_expert_projection_descriptor::from(
        ggml_tensor * weight,
        ggml_tensor * bias,
        ggml_tensor * scale) {
    return {
        weight,
        bias,
        scale,
        weight && weight->buffer ? ggml_backend_buffer_get_type(weight->buffer) : nullptr,
    };
}

bool llm_expert_bundle_descriptor::uses_merged_gate_up() const {
    return gate_up.weight != nullptr;
}

llm_expert_provider_result llm_expert_bundle_descriptor::validate() const {
    if (layer < 0 || n_expert <= 0 || !projection_is_valid(down, n_expert)) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::invalid_descriptor);
    }

    if (uses_merged_gate_up()) {
        if (!projection_is_valid(gate_up, n_expert) || gate_up.weight->ne[0] % 2 != 0 || up.weight || gate.weight) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::invalid_descriptor);
        }
    } else if (!projection_is_valid(up, n_expert)) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::invalid_descriptor);
    } else if (gate.weight && !projection_is_valid(gate, n_expert)) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::invalid_descriptor);
    }

    return llm_expert_provider_result::success();
}

llm_expert_provider_result llm_expert_selection::validate() const {
    if (layer < 0 || n_expert <= 0 || n_expert_used <= 0 || n_expert_used > n_expert || n_tokens < 0 || logical_ids == nullptr) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::invalid_selection);
    }
    if (logical_ids->type != GGML_TYPE_I32 || logical_ids->ne[0] != n_expert_used || logical_ids->ne[1] != n_tokens) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::invalid_selection);
    }
    return llm_expert_provider_result::success();
}

llm_expert_provider_result llm_validate_hybrid_execution_ids(
        const int32_t * gpu_ids,
        const int32_t * cpu_ids,
        size_t count,
        int32_t gpu_capacity,
        int32_t cpu_capacity) noexcept {
    if (gpu_ids == nullptr || cpu_ids == nullptr || count == 0 || gpu_capacity <= 0 || cpu_capacity <= 0) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::invalid_selection);
    }
    for (size_t i = 0; i < count; ++i) {
        const bool gpu_active = gpu_ids[i] >= 0 && gpu_ids[i] < gpu_capacity;
        const bool cpu_active = cpu_ids[i] >= 0 && cpu_ids[i] < cpu_capacity;
        if (gpu_ids[i] < -1 || cpu_ids[i] < -1 || gpu_ids[i] >= gpu_capacity || cpu_ids[i] >= cpu_capacity ||
            gpu_active == cpu_active) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::invalid_selection);
        }
    }
    return llm_expert_provider_result::success();
}

bool llm_expert_graph_binding::uses_merged_gate_up() const {
    return gate_up.weight != nullptr;
}

size_t llm_expert_graph_binding::execution_device_count() const noexcept {
    return remote_single ? 1 : devices.size();
}

const llm_expert_graph_binding::device_binding &
llm_expert_graph_binding::execution_device(size_t index) const {
    if (remote_single) {
        if (index != 0) throw std::out_of_range("remote expert execution device");
        return remote_device;
    }
    return devices.at(index);
}

llm_expert_provider_result llm_expert_graph_binding::validate(const llm_expert_selection & selection) const {
    if (provider_identity == nullptr || layer != selection.layer || execution_ids == nullptr ||
        (logical_ids != nullptr && logical_ids != selection.logical_ids)) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::invalid_binding);
    }
    if (execution_ids->type != GGML_TYPE_I32 || execution_ids->ne[0] != selection.n_expert_used || execution_ids->ne[1] != selection.n_tokens) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::invalid_binding);
    }
    if (down.weight == nullptr || (uses_merged_gate_up() ? (up.weight != nullptr || gate.weight != nullptr) : up.weight == nullptr)) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::invalid_binding);
    }
    if (remote_single) {
        const bool merged = remote_device.gate_up.weight != nullptr;
        if (hybrid || multi_device || !devices.empty() || checkpoint_ids == nullptr ||
            checkpoint_ids == execution_ids || checkpoint_ids == logical_ids ||
            checkpoint_ids->type != GGML_TYPE_I32 || checkpoint_ids->ne[0] != selection.n_expert_used ||
            checkpoint_ids->ne[1] != selection.n_tokens ||
            remote_device.device_id != 0 || remote_device.target_device == nullptr ||
            remote_device.execution_ids != execution_ids ||
            remote_device.down.weight == nullptr || merged != uses_merged_gate_up() ||
            (merged ? (remote_device.up.weight != nullptr || remote_device.gate.weight != nullptr) :
                remote_device.up.weight == nullptr) || !remote_device.generation_lease) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::invalid_binding);
        }
        return llm_expert_provider_result::success();
    }
    if (multi_device) {
        if (hybrid || checkpoint_ids == nullptr || checkpoint_ids == execution_ids ||
            checkpoint_ids == logical_ids || checkpoint_ids->type != GGML_TYPE_I32 ||
            checkpoint_ids->ne[0] != selection.n_expert_used ||
            checkpoint_ids->ne[1] != selection.n_tokens || devices.size() < 2) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::invalid_binding);
        }
        for (size_t index = 0; index < devices.size(); ++index) {
            const auto & device = devices[index];
            const bool merged = device.gate_up.weight != nullptr;
            if (device.device_id != index || device.target_device == nullptr ||
                device.execution_ids == nullptr || device.execution_ids->type != GGML_TYPE_I32 ||
                device.execution_ids->ne[0] != selection.n_expert_used ||
                device.execution_ids->ne[1] != selection.n_tokens ||
                device.down.weight == nullptr || merged != uses_merged_gate_up() ||
                (merged ? (device.up.weight != nullptr || device.gate.weight != nullptr) :
                    device.up.weight == nullptr) || !device.generation_lease) {
                return llm_expert_provider_result::failure(llm_expert_provider_error::invalid_binding);
            }
            for (size_t prior = 0; prior < index; ++prior) {
                if (devices[prior].target_device == device.target_device ||
                    devices[prior].execution_ids == device.execution_ids) {
                    return llm_expert_provider_result::failure(llm_expert_provider_error::invalid_binding);
                }
            }
        }
        return llm_expert_provider_result::success();
    }
    if (hybrid) {
        if (cpu_cold_only) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::invalid_binding);
        }
        const bool cpu_merged = cpu_gate_up.weight != nullptr;
        if (checkpoint_ids == nullptr || checkpoint_ids == execution_ids || checkpoint_ids == cpu_execution_ids ||
            checkpoint_ids == logical_ids ||
            checkpoint_ids->type != GGML_TYPE_I32 || checkpoint_ids->ne[0] != selection.n_expert_used ||
            checkpoint_ids->ne[1] != selection.n_tokens ||
            cpu_execution_ids == nullptr || cpu_execution_ids == execution_ids ||
            cpu_execution_ids == logical_ids || cpu_execution_ids->type != GGML_TYPE_I32 ||
            cpu_execution_ids->ne[0] != selection.n_expert_used ||
            cpu_execution_ids->ne[1] != selection.n_tokens || cpu_down.weight == nullptr ||
            cpu_merged != uses_merged_gate_up() ||
            (cpu_merged ? (cpu_up.weight != nullptr || cpu_gate.weight != nullptr) : cpu_up.weight == nullptr)) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::invalid_binding);
        }
    } else {
        const bool local_cached = default_target_device != nullptr && execution_ids != logical_ids;
        if (remote_device.target_device != nullptr ||
            cpu_execution_ids != nullptr || cpu_up.weight != nullptr || cpu_gate.weight != nullptr ||
            cpu_gate_up.weight != nullptr || cpu_down.weight != nullptr ||
            (local_cached && (checkpoint_ids == nullptr || checkpoint_ids == execution_ids ||
                checkpoint_ids == logical_ids || checkpoint_ids->type != GGML_TYPE_I32 ||
                checkpoint_ids->ne[0] != selection.n_expert_used ||
                checkpoint_ids->ne[1] != selection.n_tokens)) ||
            (!local_cached && checkpoint_ids != nullptr)) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::invalid_binding);
        }
        if (cpu_cold_only && (!local_cached || bootstrap || !generation_lease ||
            ggml_backend_dev_type(default_target_device) != GGML_BACKEND_DEVICE_TYPE_CPU)) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::invalid_binding);
        }
    }
    return llm_expert_provider_result::success();
}

llm_expert_handle::llm_expert_handle(
        llm_expert_weight_provider * provider,
        uint64_t lease_id,
        uint64_t repetitions) :
    provider(provider), lease_id(lease_id), repetitions(repetitions) {
}

llm_expert_handle::~llm_expert_handle() {
    reset();
}

llm_expert_handle::llm_expert_handle(llm_expert_handle && other) noexcept :
    provider(std::exchange(other.provider, nullptr)),
    lease_id(std::exchange(other.lease_id, 0)),
    repetitions(std::exchange(other.repetitions, 0)) {
}

llm_expert_handle & llm_expert_handle::operator=(llm_expert_handle && other) noexcept {
    if (this != &other) {
        reset();
        provider = std::exchange(other.provider, nullptr);
        lease_id = std::exchange(other.lease_id, 0);
        repetitions = std::exchange(other.repetitions, 0);
    }
    return *this;
}

bool llm_expert_handle::is_valid() const {
    return provider != nullptr && repetitions != 0;
}

void llm_expert_handle::reset() {
    if (provider) {
        auto * owner = std::exchange(provider, nullptr);
        const uint64_t id = std::exchange(lease_id, 0);
        const uint64_t count = std::exchange(repetitions, 0);
        for (uint64_t index = 0; index < count; ++index) {
            owner->release_handle(id);
        }
    }
}

llm_expert_execution_plan::~llm_expert_execution_plan() {
    reset();
}

llm_expert_execution_plan::llm_expert_execution_plan(llm_expert_execution_plan && other) noexcept :
    result(other.result), handles(std::move(other.handles)) {
    other.result = {};
}

llm_expert_execution_plan & llm_expert_execution_plan::operator=(llm_expert_execution_plan && other) noexcept {
    if (this != &other) {
        reset();
        result = other.result;
        handles = std::move(other.handles);
        other.result = {};
    }
    return *this;
}

void llm_expert_execution_plan::reserve(size_t capacity) {
    handles.reserve(capacity);
}

void llm_expert_execution_plan::add_handle(llm_expert_handle handle) {
    if (!handle.is_valid()) {
        return;
    }
    if (!handles.empty()) {
        auto & last = handles.back();
        if (last.provider == handle.provider &&
            last.lease_id == handle.lease_id) {
            last.repetitions += handle.repetitions;
            handle.provider = nullptr;
            handle.lease_id = 0;
            handle.repetitions = 0;
            return;
        }
    }
    handles.emplace_back(std::move(handle));
}

void llm_expert_execution_plan::absorb(llm_expert_execution_plan && other) {
    for (auto & handle : other.handles) {
        add_handle(std::move(handle));
    }
    other.handles.clear();
    if (!other.result.is_ready()) {
        result = other.result;
    }
    other.result = {};
}

void llm_expert_execution_plan::set_result(llm_expert_provider_result result) {
    this->result = result;
}

void llm_expert_execution_plan::reset() {
    for (auto it = handles.rbegin(); it != handles.rend(); ++it) {
        it->reset();
    }
    handles.clear();
    result = {};
}

const llm_expert_provider_result & llm_expert_execution_plan::get_result() const {
    return result;
}

size_t llm_expert_execution_plan::handle_count() const {
    size_t count = 0;
    for (const auto & handle : handles) {
        count += handle.repetitions;
    }
    return count;
}

namespace {

struct resident_provider_stats {
    std::atomic<uint64_t> bind_calls { 0 };
    std::atomic<uint64_t> prepare_calls { 0 };
    std::atomic<uint64_t> handles_acquired { 0 };
    std::atomic<uint64_t> handles_released { 0 };
    std::atomic<uint64_t> failures { 0 };
    std::atomic<uint64_t> cancellations { 0 };
    std::atomic<uint64_t> bundle_registrations { 0 };
    std::atomic<uint64_t> bundle_full_validations { 0 };
    std::atomic<uint64_t> bundle_fast_path_hits { 0 };
};

struct resident_bundle_registration {
    std::atomic<bool> registered { false };
    std::mutex mutex;
    llm_expert_bundle_descriptor bundle = {};
};

class llm_resident_expert_weight_provider final : public llm_expert_weight_provider {
public:
    explicit llm_resident_expert_weight_provider(llm_expert_provider_faults faults) : faults(faults) {
        if (faults.initialization != llm_expert_provider_error::none) {
            throw std::runtime_error("resident expert-weight provider initialization failed");
        }
    }

    llm_expert_provider_result bind(
            const llm_expert_bundle_descriptor & bundle,
            const llm_expert_selection & selection,
            llm_expert_graph_binding & binding) noexcept override {
        stats.bind_calls.fetch_add(1, std::memory_order_relaxed);
        auto result = selection.validate();
        if (!result.is_ready() || bundle.layer < 0 || bundle.layer >= LLAMA_MAX_LAYERS) {
            if (result.is_ready()) {
                result = llm_expert_provider_result::failure(llm_expert_provider_error::invalid_descriptor);
            }
            record_failure(result);
            return result;
        }

        auto & registration = registrations[bundle.layer];
        if (registration.registered.load(std::memory_order_acquire)) {
            stats.bundle_fast_path_hits.fetch_add(1, std::memory_order_relaxed);
            if (!bundle_identity_matches(registration.bundle, bundle)) {
                result = llm_expert_provider_result::failure(llm_expert_provider_error::invalid_descriptor);
            }
        } else {
            std::lock_guard<std::mutex> lock(registration.mutex);
            if (registration.registered.load(std::memory_order_relaxed)) {
                stats.bundle_fast_path_hits.fetch_add(1, std::memory_order_relaxed);
                if (!bundle_identity_matches(registration.bundle, bundle)) {
                    result = llm_expert_provider_result::failure(llm_expert_provider_error::invalid_descriptor);
                }
            } else {
                stats.bundle_full_validations.fetch_add(1, std::memory_order_relaxed);
                result = bundle.validate();
                if (result.is_ready()) {
                    registration.bundle = bundle;
                    registration.registered.store(true, std::memory_order_release);
                    stats.bundle_registrations.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
        if (result.is_ready() && faults.binding != llm_expert_provider_error::none &&
            successful_bindings.load(std::memory_order_relaxed) >= faults.binding_successes_before_failure) {
            result = llm_expert_provider_result::failure(faults.binding);
        }
        if (!result.is_ready()) {
            record_failure(result);
            return result;
        }

        binding = {};
        binding.provider_identity = this;
        binding.layer = bundle.layer;
        binding.up = bundle.up;
        binding.gate = bundle.gate;
        binding.gate_up = bundle.gate_up;
        binding.down = bundle.down;
        binding.execution_ids = selection.logical_ids;
        binding.logical_ids = selection.logical_ids;
        result = binding.validate(selection);
        if (result.is_ready()) {
            successful_bindings.fetch_add(1, std::memory_order_relaxed);
        } else {
            record_failure(result);
        }
        return result;
    }

    llm_expert_provider_result prepare(
            const std::vector<llm_expert_graph_binding> & bindings,
            llm_expert_execution_plan & plan,
            uint64_t,
            bool) noexcept override {
        stats.prepare_calls.fetch_add(1, std::memory_order_relaxed);
        plan.reset();
        auto fail = [&](llm_expert_provider_result result) {
            record_failure(result);
            plan.reset();
            plan.set_result(result);
            return result;
        };
        try {
            for (size_t index = 0; index < bindings.size(); ++index) {
                const auto & binding = bindings[index];
                if (binding.provider_identity != this || binding.layer < 0 || binding.layer >= LLAMA_MAX_LAYERS) {
                    return fail(llm_expert_provider_result::failure(llm_expert_provider_error::invalid_binding));
                }
                const auto & registration = registrations[binding.layer];
                if (!registration.registered.load(std::memory_order_acquire) ||
                    !binding_identity_matches(binding, registration.bundle)) {
                    return fail(llm_expert_provider_result::failure(llm_expert_provider_error::invalid_binding));
                }
            }

            if (bindings.empty()) {
                plan.set_result(llm_expert_provider_result::success());
                return llm_expert_provider_result::success();
            }

            const auto injected_preparation_error = faults.preparation == llm_expert_provider_error::none
                ? llm_expert_provider_error::preparation_failed
                : faults.preparation;
            if (faults.fail_preparation_after_handles == 0) {
                return fail(llm_expert_provider_result::failure(injected_preparation_error));
            }

            stats.handles_acquired.fetch_add(1, std::memory_order_relaxed);
            plan.add_handle({ this, 1 });

            if (faults.preparation != llm_expert_provider_error::none || faults.fail_preparation_after_handles != SIZE_MAX) {
                return fail(llm_expert_provider_result::failure(injected_preparation_error));
            }
        } catch (const std::bad_alloc &) {
            return fail(llm_expert_provider_result::failure(llm_expert_provider_error::allocation_failed));
        } catch (...) {
            return fail(llm_expert_provider_result::failure(llm_expert_provider_error::preparation_failed));
        }

        plan.set_result(llm_expert_provider_result::success());
        return llm_expert_provider_result::success();
    }

    llm_expert_provider_stats get_stats() const noexcept override {
        return {
            1,
            stats.bind_calls.load(std::memory_order_relaxed),
            stats.prepare_calls.load(std::memory_order_relaxed),
            stats.handles_acquired.load(std::memory_order_relaxed),
            stats.handles_released.load(std::memory_order_relaxed),
            0,
            0,
            0,
            0,
            stats.failures.load(std::memory_order_relaxed),
            stats.cancellations.load(std::memory_order_relaxed),
            stats.bundle_registrations.load(std::memory_order_relaxed),
            stats.bundle_full_validations.load(std::memory_order_relaxed),
            stats.bundle_fast_path_hits.load(std::memory_order_relaxed),
        };
    }

protected:
    void release_handle(uint64_t) noexcept override {
        stats.handles_released.fetch_add(1, std::memory_order_relaxed);
    }

private:
    void record_failure(llm_expert_provider_result result) noexcept {
        stats.failures.fetch_add(1, std::memory_order_relaxed);
        if (result.status == llm_expert_provider_status::cancelled) {
            stats.cancellations.fetch_add(1, std::memory_order_relaxed);
        }
    }

    llm_expert_provider_faults faults;
    resident_provider_stats stats;
    std::atomic<uint64_t> successful_bindings { 0 };
    std::array<resident_bundle_registration, LLAMA_MAX_LAYERS> registrations;
};

struct hot_pool_generation {
    uint64_t id = 0;
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buffer;
    std::vector<llm_expert_bundle_descriptor> bundles;
    uint64_t slot_stride = 0;
    std::array<uint64_t, 12> role_offsets = {};
    std::array<uint64_t, 12> role_extents = {};
    std::vector<uintptr_t> addresses;
};

using hot_slot_state = llm_hot_cache_diagnostics::slot::state_type;

struct hot_forward_entry {
    int32_t slot = -1;
    uint64_t generation = 0;
};

struct hot_slot_entry {
    llm_expert_key key = { -1, -1 };
    llm_expert_layout_class_id layout_class_id = LLM_EXPERT_LAYOUT_CLASS_INVALID;
    uint64_t generation = 0;
    uint64_t last_use = 0;
    uint32_t refcount = 0;
    uint32_t cold_slot = 0;
    uint64_t cold_generation = 0;
    bool has_cold_backing = false;
    llm_expert_residency_origin origin = llm_expert_residency_origin::demand;
    bool background_useful = false;
    uint64_t speculative_deadline = 0;
    uint64_t speculative_utility = 0;
    llm_expert_device_id device_id = 0;
    uint32_t device_slot = 0;
    hot_slot_state state = hot_slot_state::free;
};

struct background_promotion_record {
    enum class lifecycle : uint8_t {
        empty,
        queued_or_staging,
        h2d_in_flight,
        h2d_complete_unpublished,
        published,
        failed,
        cancelled,
    } state = lifecycle::empty;
    bool active = false;
    llm_expert_key key = { -1, -1 };
    llm_cold_reference cold;
    llm_transfer_lane_reference lane;
    llm_expert_request_handle scheduler_handle;
    llm_expert_request_state scheduler_state = llm_expert_request_state::free;
    uint32_t hot_slot = UINT32_MAX;
    uint64_t hot_generation = 0;
    uint64_t origin_operation_ordinal = 0;
    uint64_t remaining_bytes = 0;
    uint64_t speculative_deadline = 0;
    uint64_t speculative_utility = 0;
    uint64_t h2d_work_ns = 0;
    uint64_t prediction_sequence = 0;
    bool predictive = false;
    bool demand_claimed = false;
};

struct auto_decision_record {
    uint64_t request = 0;
    int32_t layer = -1;
    int32_t expert = -1;
    llm_expert_auto_input input;
    llm_expert_auto_result result;
};

struct hot_request_pin {
    uint32_t slot = 0;
    uint64_t generation = 0;
};

bool expert_key_matches(const llm_expert_key & lhs, const llm_expert_key & rhs) {
    return lhs.layer == rhs.layer && lhs.expert == rhs.expert;
}

template <typename T>
void phase10_digest_append(uint64_t & digest, T value) noexcept {
    const auto * bytes = reinterpret_cast<const uint8_t *>(&value);
    for (size_t index = 0; index < sizeof(value); ++index) {
        digest ^= bytes[index];
        digest *= 1099511628211ULL;
    }
}

llm_expert_provider_result cache_policy_result(llm_expert_cache_policy_result result) {
    if (result.is_ready()) return llm_expert_provider_result::success();
    switch (result.error) {
        case llm_expert_cache_policy_error::no_victim:
            return llm_expert_provider_result::failure(llm_expert_provider_error::busy);
        case llm_expert_cache_policy_error::metadata_mismatch:
            return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
        case llm_expert_cache_policy_error::sequence_exhausted:
        case llm_expert_cache_policy_error::overflow:
        case llm_expert_cache_policy_error::transcript_full:
            return llm_expert_provider_result::failure(llm_expert_provider_error::unsupported_configuration);
        case llm_expert_cache_policy_error::invalid_configuration:
        case llm_expert_cache_policy_error::invalid_key:
        case llm_expert_cache_policy_error::invalid_event:
            return llm_expert_provider_result::failure(llm_expert_provider_error::invalid_key);
        case llm_expert_cache_policy_error::none:
            break;
    }
    return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
}

bool expert_bundle_payload_bytes(const llm_expert_bundle_descriptor & bundle, uint64_t & bytes) {
    bytes = 0;
    for (const auto * projection : { &bundle.up, &bundle.gate, &bundle.gate_up, &bundle.down }) {
        size_t index = 0;
        for (const auto * tensor : { projection->weight, projection->bias, projection->scale }) {
            if (tensor == nullptr) {
                index++;
                continue;
            }
            const int axis = expert_axis(tensor, bundle.n_expert, index == 0);
            if (axis < 0 || tensor->nb[axis] > UINT64_MAX - bytes) return false;
            bytes += tensor->nb[axis];
            index++;
        }
    }
    return bytes != 0;
}

ggml_tensor * make_slot_tensor(
        ggml_context * ctx,
        const ggml_tensor * source,
        int32_t n_expert,
        uint32_t capacity,
        bool weight,
        const char * name) {
    if (source == nullptr) {
        return nullptr;
    }
    const int axis = expert_axis(source, n_expert, weight);
    if (axis < 0) {
        throw std::invalid_argument("expert sidecar has no unambiguous expert axis");
    }
    int64_t ne[GGML_MAX_DIMS];
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        ne[i] = source->ne[i];
    }
    ne[axis] = capacity;
    ggml_tensor * tensor = ggml_new_tensor(ctx, source->type, ggml_n_dims(source), ne);
    ggml_set_name(tensor, name);
    return tensor;
}

llm_expert_projection_descriptor make_slot_projection(
        ggml_context * ctx,
        const llm_expert_projection_descriptor & source,
        int32_t n_expert,
        uint32_t capacity,
        const char * prefix) {
    const std::string weight_name = std::string(prefix) + ".weight";
    const std::string bias_name = std::string(prefix) + ".bias";
    const std::string scale_name = std::string(prefix) + ".scale";
    return {
        make_slot_tensor(ctx, source.weight, n_expert, capacity, true, weight_name.c_str()),
        make_slot_tensor(ctx, source.bias, n_expert, capacity, false, bias_name.c_str()),
        make_slot_tensor(ctx, source.scale, n_expert, capacity, false, scale_name.c_str()),
        nullptr,
    };
}

void collect_projection_addresses(
        const llm_expert_projection_descriptor & projection,
        std::vector<uintptr_t> & addresses) {
    for (const auto * tensor : { projection.weight, projection.bias, projection.scale }) {
        if (tensor != nullptr) {
            addresses.push_back(reinterpret_cast<uintptr_t>(tensor->data));
        }
    }
}

ggml_tensor * bundle_member(
        llm_expert_bundle_descriptor & bundle,
        size_t projection_index,
        size_t member_index) {
    llm_expert_projection_descriptor * projection = nullptr;
    if (projection_index == 0) projection = &bundle.up;
    else if (projection_index == 1) projection = &bundle.gate;
    else if (projection_index == 2) projection = &bundle.gate_up;
    else if (projection_index == 3) projection = &bundle.down;
    if (projection == nullptr) return nullptr;
    return member_index == 0 ? projection->weight : member_index == 1 ? projection->bias : projection->scale;
}

bool checked_align_size(size_t value, size_t alignment, size_t & result) {
    if (alignment == 0 || value > SIZE_MAX - (alignment - 1)) return false;
    result = (value + alignment - 1)/alignment*alignment;
    return true;
}

bool checked_lcm_size(size_t lhs, size_t rhs, size_t & result) {
    if (lhs == 0 || rhs == 0) return false;
    const size_t reduced = lhs/std::gcd(lhs, rhs);
    if (reduced > SIZE_MAX/rhs) return false;
    result = reduced*rhs;
    return true;
}

struct hot_pool_physical_layout {
    struct role { size_t offset = 0; size_t extent = 0; };
    std::array<role, 12> roles;
    size_t slot_stride = 0;
};

bool derive_hot_pool_physical_layout(
        const llm_expert_layout_registry & registry,
        ggml_backend_buffer_type_t buffer_type,
        hot_pool_physical_layout & result) {
    if (!registry.sealed() || buffer_type == nullptr) return false;
    const size_t buffer_alignment = ggml_backend_buft_get_alignment(buffer_type);
    size_t slot_alignment = std::max<size_t>(64, buffer_alignment);
    size_t within_slot = 0;
    for (size_t projection = 0; projection < 4; ++projection) {
        for (size_t member_index = 0; member_index < 3; ++member_index) {
            const size_t role = projection*3 + member_index;
            size_t role_alignment = std::max<size_t>(64, buffer_alignment);
            for (const auto & layout_class : registry.classes) {
                auto source = layout_class.prototype;
                const ggml_tensor * tensor = bundle_member(source, projection, member_index);
                if (tensor == nullptr) continue;
                const int axis = expert_axis(tensor, source.n_expert, member_index == 0);
                if (axis < 0) return false;
                result.roles[role].extent = std::max(result.roles[role].extent, tensor->nb[axis]);
                role_alignment = std::max(role_alignment, ggml_type_size(tensor->type));
                if (!checked_lcm_size(slot_alignment, ggml_type_size(tensor->type), slot_alignment)) return false;
            }
            if (result.roles[role].extent == 0) continue;
            if (!checked_align_size(within_slot, role_alignment, result.roles[role].offset) ||
                result.roles[role].extent > SIZE_MAX - result.roles[role].offset) return false;
            within_slot = result.roles[role].offset + result.roles[role].extent;
        }
    }
    return checked_align_size(within_slot, slot_alignment, result.slot_stride) && result.slot_stride != 0;
}

bool preflight_hot_pool_consumers(
        const llm_expert_layout_registry & registry,
        uint32_t capacity,
        uint32_t n_expert_used,
        ggml_backend_dev_t target_device,
        ggml_backend_buffer_type_t buffer_type,
        uint32_t & consumer_count) {
    consumer_count = 0;
    if (registry.classes.size() == 1) return true;
    hot_pool_physical_layout physical;
    if (capacity == 0 || n_expert_used == 0 || target_device == nullptr ||
        !derive_hot_pool_physical_layout(registry, buffer_type, physical)) return false;
    for (const auto & layout_class : registry.classes) {
        for (const auto * projection : { &layout_class.prototype.up, &layout_class.prototype.gate,
                &layout_class.prototype.gate_up, &layout_class.prototype.down }) {
            const ggml_tensor * source = projection->weight;
            if (source == nullptr) continue;
            ggml_init_params params = {
                /*.mem_size   =*/ ggml_tensor_overhead()*5 + ggml_graph_overhead(),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            ggml_context_ptr ctx(ggml_init(params));
            if (!ctx) return false;
            ggml_tensor * weight = make_slot_tensor(
                ctx.get(), source, layout_class.prototype.n_expert, capacity, true, "layout-preflight.weight");
            const int axis = expert_axis(weight, capacity, true);
            if (axis < 0) return false;
            weight->nb[axis] = physical.slot_stride;
            for (int upper = axis + 1; upper < GGML_MAX_DIMS; ++upper) {
                if (weight->nb[upper - 1] > SIZE_MAX/weight->ne[upper - 1]) return false;
                weight->nb[upper] = weight->nb[upper - 1]*weight->ne[upper - 1];
            }
            ggml_tensor * input = ggml_new_tensor_3d(
                ctx.get(), GGML_TYPE_F32, weight->ne[0], n_expert_used, 1);
            ggml_tensor * ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, n_expert_used, 1);
            ggml_tensor * op = ggml_mul_mat_id(ctx.get(), weight, input, ids);
            if (!ggml_backend_dev_supports_op(target_device, op)) return false;
            consumer_count++;
        }
    }
    return true;
}

bool allocate_hot_pool(
        hot_pool_generation & pool,
        const llm_expert_layout_registry & registry,
        uint32_t capacity,
        ggml_backend_buffer_type_t buffer_type) {
    if (!registry.sealed() || registry.classes.size() > LLM_EXPERT_LAYOUT_CLASS_MAX ||
        capacity == 0 || buffer_type == nullptr) return false;
    pool.bundles.clear();
    pool.bundles.reserve(registry.classes.size());
    for (const auto & layout_class : registry.classes) {
        const auto & source = layout_class.prototype;
        llm_expert_bundle_descriptor bundle = {};
        bundle.layer = -1;
        bundle.n_expert = capacity;
        bundle.up = make_slot_projection(pool.ctx.get(), source.up, source.n_expert, capacity, "hot.up");
        bundle.gate = make_slot_projection(pool.ctx.get(), source.gate, source.n_expert, capacity, "hot.gate");
        bundle.gate_up = make_slot_projection(pool.ctx.get(), source.gate_up, source.n_expert, capacity, "hot.gate_up");
        bundle.down = make_slot_projection(pool.ctx.get(), source.down, source.n_expert, capacity, "hot.down");
        pool.bundles.push_back(bundle);
    }

    if (registry.classes.size() == 1) {
        pool.buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(pool.ctx.get(), buffer_type));
        if (!pool.buffer) return false;
        auto & bundle = pool.bundles.front();
        for (auto * projection : { &bundle.up, &bundle.gate, &bundle.gate_up, &bundle.down }) {
            if (projection->weight != nullptr) {
                projection->buffer_type = ggml_backend_buffer_get_type(projection->weight->buffer);
            }
        }
        collect_projection_addresses(bundle.up, pool.addresses);
        collect_projection_addresses(bundle.gate, pool.addresses);
        collect_projection_addresses(bundle.gate_up, pool.addresses);
        collect_projection_addresses(bundle.down, pool.addresses);
        const uint64_t bytes = ggml_backend_buffer_get_size(pool.buffer.get());
        pool.slot_stride = bytes/capacity + (bytes % capacity != 0);
        return pool.slot_stride != 0;
    }

    hot_pool_physical_layout physical;
    if (!derive_hot_pool_physical_layout(registry, buffer_type, physical) ||
        capacity > SIZE_MAX/physical.slot_stride) return false;
    const auto & roles = physical.roles;
    const size_t slot_stride = physical.slot_stride;
    pool.buffer.reset(ggml_backend_buft_alloc_buffer(buffer_type, slot_stride*capacity));
    if (!pool.buffer) return false;
    ggml_backend_buffer_clear(pool.buffer.get(), 0xa5);
    auto * base = static_cast<uint8_t *>(ggml_backend_buffer_get_base(pool.buffer.get()));
    if (base == nullptr) return false;
    for (auto & bundle : pool.bundles) {
        for (size_t projection = 0; projection < 4; ++projection) {
            for (size_t member_index = 0; member_index < 3; ++member_index) {
                ggml_tensor * tensor = bundle_member(bundle, projection, member_index);
                if (tensor == nullptr) continue;
                const int axis = expert_axis(tensor, capacity, member_index == 0);
                const size_t role = projection*3 + member_index;
                if (axis < 0 || tensor->nb[axis] > roles[role].extent) return false;
                tensor->nb[axis] = slot_stride;
                for (int upper = axis + 1; upper < GGML_MAX_DIMS; ++upper) {
                    if (tensor->nb[upper - 1] > SIZE_MAX/tensor->ne[upper - 1]) return false;
                    tensor->nb[upper] = tensor->nb[upper - 1]*tensor->ne[upper - 1];
                }
                if (ggml_backend_tensor_alloc(pool.buffer.get(), tensor, base + roles[role].offset) !=
                        GGML_STATUS_SUCCESS) return false;
            }
        }
        for (auto * projection : { &bundle.up, &bundle.gate, &bundle.gate_up, &bundle.down }) {
            if (projection->weight != nullptr) projection->buffer_type = buffer_type;
        }
        collect_projection_addresses(bundle.up, pool.addresses);
        collect_projection_addresses(bundle.gate, pool.addresses);
        collect_projection_addresses(bundle.gate_up, pool.addresses);
        collect_projection_addresses(bundle.down, pool.addresses);
    }
    pool.slot_stride = slot_stride;
    for (size_t role = 0; role < roles.size(); ++role) {
        pool.role_offsets[role] = roles[role].offset;
        pool.role_extents[role] = roles[role].extent;
    }
    return ggml_backend_buffer_get_size(pool.buffer.get()) >= slot_stride*capacity;
}

bool copy_expert_tensor(
        ggml_tensor * target,
        const ggml_tensor * source,
        int32_t n_expert,
        uint32_t capacity,
        int32_t expert,
        uint32_t slot,
        bool weight,
        size_t & bytes) {
    if (target == nullptr || source == nullptr) {
        return target == source;
    }
    const int source_axis = expert_axis(source, n_expert, weight);
    if (source_axis < 0 || target->ne[source_axis] != capacity ||
        source->data == nullptr || target->data == nullptr) {
        return false;
    }
    for (int axis = source_axis + 1; axis < ggml_n_dims(source); ++axis) {
        if (source->ne[axis] != 1 || target->ne[axis] != 1) {
            return false;
        }
    }
    const size_t span = source->nb[source_axis];
    if (target->nb[source_axis] < span) {
        return false;
    }
    const auto * source_data = static_cast<const uint8_t *>(source->data) + size_t(expert)*span;
    ggml_backend_tensor_set(target, source_data, size_t(slot)*target->nb[source_axis], span);
    bytes += span;
    return true;
}

bool copy_expert_projection(
        const llm_expert_projection_descriptor & target,
        const llm_expert_projection_descriptor & source,
        int32_t n_expert,
        uint32_t capacity,
        int32_t expert,
        uint32_t slot,
        size_t & bytes,
        size_t & copies,
        size_t fail_after) {
    const std::array<std::pair<ggml_tensor *, const ggml_tensor *>, 3> tensors = {{
        { target.weight, source.weight },
        { target.bias, source.bias },
        { target.scale, source.scale },
    }};
    for (size_t index = 0; index < tensors.size(); ++index) {
        if (tensors[index].first == nullptr && tensors[index].second == nullptr) {
            continue;
        }
        if (copies >= fail_after || !copy_expert_tensor(
                tensors[index].first,
                tensors[index].second,
                n_expert,
                capacity,
                expert,
                slot,
                index == 0,
                bytes)) {
            return false;
        }
        copies++;
    }
    return true;
}

class llm_hot_cache_expert_weight_provider final : public llm_expert_weight_provider {
public:
    llm_hot_cache_expert_weight_provider(llm_hot_cache_config config, llm_expert_provider_faults faults) :
        config(config), faults(faults),
        deterministic_policy_terminals(
            config.hot_cache_policy_config.supplied || config.cold_cache_policy_config.supplied ||
            (config.prefetch_config.supplied &&
             config.prefetch_config.value.policy != LLAMA_EXPERT_PREFETCH_POLICY_OFF)) {
        requested_cold_cache_bytes = this->config.cold_cache_bytes;
        system_memory_budget.configure(
            this->config.sample_memory,
            this->config.min_system_headroom_bytes,
            this->config.min_runtime_headroom_bytes,
            this->config.system_memory_regions);
        if (faults.initialization != llm_expert_provider_error::none) {
            throw std::runtime_error("hot-cache expert-weight provider initialization failed");
        }
        if (this->config.phase10_lead_trace) {
            phase10_lead_events = std::make_unique<std::array<llm_expert_phase10_lead_event, 256>>();
            phase10_scheduler_events = std::make_unique<std::array<llm_expert_phase10_scheduler_event, 256>>();
        }
        const bool phase10_active = this->config.prefetch_config.supplied &&
            (this->config.prefetch_config.value.policy != LLAMA_EXPERT_PREFETCH_POLICY_OFF ||
             this->config.prefetch_config.value.seed_mode != LLAMA_EXPERT_PREFETCH_SEED_MODE_OFF);
        if (phase10_active) {
            phase10_issue_ahead_trace =
                std::make_unique<std::array<llm_expert_phase10_issue_ahead_event, 256>>();
        }
        if (config.capacity < config.n_expert_used || config.capacity > config.total_expert_keys ||
            config.n_expert_used == 0 || config.routed_layer_count == 0 || config.total_expert_keys == 0 ||
            config.total_expert_keys % config.routed_layer_count != 0 || config.target_buffer_type == nullptr ||
            (config.integrity_mode != llm_expert_integrity_mode::none &&
             config.integrity_mode != llm_expert_integrity_mode::fnv64_end_to_end)) {
            throw std::invalid_argument("invalid hot-cache capacity or topology");
        }
        multi_device = config.devices.size() > 1;
        device_transfer_delay_us.assign(std::max<size_t>(config.devices.size(), 1), 0);
        device_transfer_failure_for_testing.assign(std::max<size_t>(config.devices.size(), 1), false);
        device_transfer_failure_decode_only_for_testing.assign(
            std::max<size_t>(config.devices.size(), 1), false);
        if (!config.devices.empty()) {
            uint64_t total_capacity = 0;
            std::string prior_bdf;
            for (size_t index = 0; index < config.devices.size(); ++index) {
                const auto & device = config.devices[index];
                if (device.device_id != index || device.target_device == nullptr ||
                    device.target_buffer_type == nullptr || device.capacity < config.n_expert_used ||
                    device.cuda_ordinal < 0 || device.pci_bdf.empty() || device.uuid.empty() ||
                    (!prior_bdf.empty() && prior_bdf >= device.pci_bdf) ||
                    device.capacity > UINT32_MAX - total_capacity) {
                    throw std::invalid_argument("invalid multi-device hot-cache topology");
                }
                for (size_t prior = 0; prior < index; ++prior) {
                    if (config.devices[prior].target_device == device.target_device ||
                        config.devices[prior].cuda_ordinal == device.cuda_ordinal ||
                        config.devices[prior].uuid == device.uuid) {
                        throw std::invalid_argument("duplicate multi-device hot-cache identity");
                    }
                }
                total_capacity += device.capacity;
                prior_bdf = device.pci_bdf;
            }
            if (total_capacity != config.capacity || config.devices.size() > LLM_EXPERT_MAX_DEVICES ||
                config.target_device != config.devices.front().target_device ||
                config.target_buffer_type != config.devices.front().target_buffer_type) {
                throw std::invalid_argument("inconsistent multi-device hot-cache capacity");
            }
        }
        if (config.remote_single != (config.devices.size() == 1) ||
            (!config.remote_single && !multi_device && !config.devices.empty())) {
            throw std::invalid_argument("inconsistent expert execution role shape");
        }
        const bool distributed_execution = config.remote_single || multi_device;
        if (distributed_execution && (!config.cold_mode ||
            config.miss_policy != LLAMA_EXPERT_MISS_POLICY_PROMOTE_AND_GPU ||
            config.background_promotion || phase10_active ||
            config.hot_cache_policy_config.policy != LLAMA_EXPERT_CACHE_POLICY_LRU ||
            config.hot_cache_policy_config.scope != LLAMA_EXPERT_CACHE_POLICY_SCOPE_GLOBAL ||
            config.hot_cache_policy_config.admission != LLAMA_EXPERT_CACHE_ADMISSION_ALWAYS ||
            (config.peer_transport == LLAMA_EXPERT_PEER_TRANSPORT_HOST_STAGED && config.peer_staging_bytes == 0) ||
            (config.peer_transport == LLAMA_EXPERT_PEER_TRANSPORT_P2P && config.peer_staging_bytes != 0))) {
            throw std::invalid_argument("unsupported multi-device expert policy configuration");
        }
        n_expert = config.total_expert_keys/config.routed_layer_count;
        if (this->config.hot_cache_policy_config.digest == 0) {
            const auto copied = llm_expert_cache_policy_copy_config(
                nullptr, llm_expert_cache_policy_tier::hot, this->config.hot_cache_policy_config);
            if (!copied.is_ready()) throw std::invalid_argument("invalid hot-cache policy configuration");
        }
        if (this->config.cold_cache_policy_config.digest == 0) {
            const auto copied = llm_expert_cache_policy_copy_config(
                nullptr, llm_expert_cache_policy_tier::cold, this->config.cold_cache_policy_config);
            if (!copied.is_ready()) throw std::invalid_argument("invalid cold-cache policy configuration");
        }
        if (!this->config.routed_layers.empty() &&
            this->config.routed_layers.size() != config.routed_layer_count) {
            throw std::invalid_argument("invalid hot-cache policy layer topology");
        }
        if (this->config.hot_cache_policy_config.scope == LLAMA_EXPERT_CACHE_POLICY_SCOPE_PER_LAYER &&
            config.capacity < uint64_t(config.routed_layer_count)*config.n_expert_used) {
            throw std::invalid_argument("per-layer hot-cache policy cannot satisfy simultaneous demand width");
        }
        if (config.n_expert_used > n_expert) {
            throw std::invalid_argument("invalid hot-cache expert topology");
        }
        if (!config.cpu_cold_only && !config.allow_non_cuda_target_for_testing &&
            !buffer_type_is_cuda(config.target_buffer_type)) {
            throw std::invalid_argument("hot-cache target must be one CUDA device");
        }
        if (config.cpu_cold_only && (!config.cold_mode ||
            config.transfer_ring_bytes != 0 || config.target_device == nullptr ||
            ggml_backend_dev_type(config.target_device) != GGML_BACKEND_DEVICE_TYPE_CPU ||
            buffer_type_is_cuda(config.target_buffer_type) || config.capacity != config.n_expert_used ||
            (!config.allow_non_cuda_target_for_testing && config.storage == nullptr) ||
            config.miss_policy != LLAMA_EXPERT_MISS_POLICY_CPU_FALLBACK || config.background_promotion ||
            config.async_cold_fill || phase10_active || config.remote_single || !config.devices.empty() ||
            config.peer_staging_bytes != 0 || config.auto_cost_model.version != 0 ||
            config.auto_cost_model_digest != 0)) {
            throw std::invalid_argument("invalid CPU cold-only expert topology");
        }
        if (config.cold_mode && !config.cpu_cold_only &&
            (config.transfer_ring_bytes == 0 ||
            config.target_device == nullptr || (!config.allow_non_cuda_target_for_testing &&
                config.storage == nullptr && config.phase8_test_control == nullptr) ||
            (config.descriptor_only_source_for_testing && !config.allow_non_cuda_target_for_testing))) {
            throw std::invalid_argument("cold-cache mode requires byte budgets and a target device");
        }
        const bool controlled_in_memory_scheduler = config.phase8_test_control != nullptr &&
            config.scheduler != nullptr && config.async_transport == nullptr && config.storage == nullptr;
        if (!controlled_in_memory_scheduler &&
            ((config.async_transport == nullptr) != (config.scheduler == nullptr) ||
             (config.async_transport != nullptr && config.storage == nullptr))) {
            throw std::invalid_argument("incomplete cold-cache async configuration");
        }
        if (!config.cold_mode && (config.cold_cache_bytes != 0 || config.transfer_ring_bytes != 0 ||
            config.force_pageable_transfer_for_testing || config.async_transport != nullptr || config.scheduler != nullptr)) {
            throw std::invalid_argument("cold-cache configuration outside cold mode");
        }
        const bool hybrid_policy = config.miss_policy == LLAMA_EXPERT_MISS_POLICY_CPU_FALLBACK ||
            config.miss_policy == LLAMA_EXPERT_MISS_POLICY_AUTO;
        if (config.miss_policy != LLAMA_EXPERT_MISS_POLICY_PROMOTE_AND_GPU && !hybrid_policy) {
            throw std::invalid_argument("invalid expert miss policy");
        }
        if (!config.cold_mode && (hybrid_policy || config.background_promotion ||
                                  config.async_cold_fill || config.auto_cost_model.version != 0 ||
                                  config.auto_cost_model_digest != 0)) {
            throw std::invalid_argument("miss-policy configuration outside cold mode");
        }
        if (config.async_cold_fill &&
            config.miss_policy != LLAMA_EXPERT_MISS_POLICY_PROMOTE_AND_GPU) {
            throw std::invalid_argument("async cold fill requires PROMOTE_AND_GPU");
        }
        if (config.prefetch_profile_loaded != phase10_active ||
            (!config.prefetch_config.supplied && config.phase10_serial_issue_for_testing)) {
            throw std::invalid_argument("incomplete expert prefetch configuration");
        }
        if (phase10_active && (config.prefetch_profile.profile_sha256.empty() ||
            config.prefetch_profile.target.experts_per_layer != n_expert ||
            config.prefetch_profile.target.routed_layers.size() != config.routed_layer_count)) {
            throw std::invalid_argument("expert prefetch profile topology mismatch");
        }
        if (config.prefetch_config.supplied &&
                config.prefetch_config.value.policy != LLAMA_EXPERT_PREFETCH_POLICY_OFF) {
            const auto initialized = prefetch_predictor.initialize(
                this->config.prefetch_profile, this->config.prefetch_config);
            if (!initialized.is_ready()) {
                throw std::invalid_argument("expert prefetch predictor initialization failed");
            }
            const auto selected_cost = std::find_if(
                this->config.prefetch_profile.costs.begin(),
                this->config.prefetch_profile.costs.end(), [&](const auto & cost) {
                    return cost.transport == this->config.prefetch_profile.selected_transport &&
                        cost.readiness == this->config.prefetch_config.value.readiness;
                });
            if (selected_cost == this->config.prefetch_profile.costs.end()) {
                throw std::invalid_argument("expert prefetch utility envelope missing");
            }
            phase10_utility_min_timely_successes =
                selected_cost->utility_min_timely_successes;
        }
        if (config.miss_policy == LLAMA_EXPERT_MISS_POLICY_AUTO) {
            const auto & cost = config.auto_cost_model;
            if (cost.version != LLAMA_EXPERT_AUTO_COST_MODEL_VERSION_1 ||
                cost.struct_size != sizeof(llama_expert_auto_cost_model) || config.auto_cost_model_digest == 0 ||
                cost.cpu_fixed_decode_ns == 0 || cost.cpu_fixed_prefill_ns == 0 ||
                cost.cpu_per_lane_decode_ns == 0 || cost.cpu_per_lane_prefill_ns == 0 ||
                cost.gpu_fixed_decode_ns == 0 || cost.gpu_fixed_prefill_ns == 0 ||
                cost.gpu_per_lane_decode_ns == 0 || cost.gpu_per_lane_prefill_ns == 0 ||
                cost.h2d_fixed_ns == 0 || cost.h2d_bytes_per_second == 0 ||
                cost.decision_hysteresis_ns == 0) {
                throw std::invalid_argument("invalid expert AUTO cost model");
            }
        } else if (config.auto_cost_model.version != 0 || config.auto_cost_model_digest != 0) {
            throw std::invalid_argument("expert AUTO cost model supplied for a non-AUTO policy");
        }
    }

    ~llm_hot_cache_expert_weight_provider() override {
        LLM_EXPERT_TRACE_SCOPE("k3.lifecycle", "provider_teardown");
        std::lock_guard<std::mutex> lock(mutex);
        finish_prediction_request_locked();
        auto ring_surrendered = llm_expert_provider_result::success();
        for (auto & ring : transfer_rings) {
            const auto result = ring->surrender();
            if (ring_surrendered.is_ready() && !result.is_ready()) ring_surrendered = result;
        }
        (void) release_request_pins_locked();
        while (true) {
            background_promotion_record * earliest = nullptr;
            for (auto & record : background_promotions) {
                if (record.active && (earliest == nullptr ||
                    record.origin_operation_ordinal < earliest->origin_operation_ordinal)) {
                    earliest = &record;
                }
            }
            if (earliest == nullptr) break;
            earliest->state = background_promotion_record::lifecycle::cancelled;
            discard_background_slot_locked(*earliest, false);
            background_dropped++;
        }
        if (active_request) {
            (void) end_policy_request_locked(false, true);
            active_request = false;
        }
        for (uint32_t slot = 0; slot < directory_slots.size(); ++slot) {
            auto & entry = directory_slots[slot];
            if (entry.origin == llm_expert_residency_origin::speculative &&
                !entry.background_useful) background_wasted++;
            if (entry.state == hot_slot_state::loading) {
                (void) hot_policy.load_failed(slot, entry.generation);
            } else if (entry.state == hot_slot_state::ready || entry.state == hot_slot_state::pinned) {
                LLM_EXPERT_TRACE_INSTANT("k3.cache.hot", "victim", "slot_id", slot,
                    "generation", entry.generation, "layer", entry.key.layer,
                    "original_expert_id", entry.key.expert);
                (void) hot_policy.remove_resident(slot, entry.generation);
                LLM_EXPERT_TRACE_INSTANT("k3.cache.hot", "eviction", "slot_id", slot,
                    "generation", entry.generation, "layer", entry.key.layer,
                    "original_expert_id", entry.key.expert);
            }
            clear_forward_locked(entry.key, slot, entry.generation);
            if (entry.has_cold_backing && cold_cache) {
                (void) cold_cache->release(
                    { entry.cold_slot, entry.cold_generation, entry.layout_class_id },
                    llm_cold_reference_kind::hot);
                entry.has_cold_backing = false;
            }
        }
        LLM_EXPERT_TRACE_COUNTER("k3.resource", "hot_cache_occupancy", 4, 0);
        active_background_flights = 0;
        const auto cold_surrendered = cold_cache ? cold_cache->surrender() :
            llm_expert_provider_result::success();
        if (!config.cpu_cold_only) (void) hot_policy.surrender();
        if (config.phase8_closeout_witness != nullptr && !config.phase8_closeout_witness->written) {
            auto & witness = *config.phase8_closeout_witness;
            witness.background_submitted = background_submitted;
            witness.background_published_completed = background_completed;
            witness.background_dropped = background_dropped;
            witness.background_useful = background_useful;
            witness.background_wasted = background_wasted;
            witness.background_busy = background_busy;
            for (const auto & record : background_promotions) {
                if (record.active) witness.background_lifecycle[size_t(record.state)]++;
            }
            if (config.scheduler != nullptr) {
                const auto scheduler = config.scheduler->diagnostics();
                witness.scheduler_active = scheduler.active_requests;
                witness.scheduler_queued = scheduler.queued_requests;
                witness.scheduler_terminal_complete = scheduler.terminal_complete;
                witness.scheduler_terminal_failed = scheduler.terminal_failed;
                witness.scheduler_terminal_cancelled = scheduler.terminal_cancelled;
                witness.scheduler_terminal_releases = scheduler.terminal_releases;
            }
            llm_transfer_ring_closeout_diagnostics ring;
            ring.invariants_ok = true;
            for (const auto & candidate : transfer_rings) {
                const auto current = candidate->closeout_diagnostics();
                ring.queued_workers += current.queued_workers;
                ring.running_workers += current.running_workers;
                ring.non_free_lanes += current.non_free_lanes;
                ring.live_events += current.live_events;
                ring.invariants_ok = ring.invariants_ok && current.invariants_ok;
            }
            witness.ring_queued_workers = ring.queued_workers;
            witness.ring_running_workers = ring.running_workers;
            witness.ring_non_free_lanes = ring.non_free_lanes;
            witness.ring_live_events = ring.live_events;
            witness.cold_hot_refs = cold_surrendered.is_ready() ? 0 : UINT64_MAX;
            witness.cold_transfer_refs = cold_surrendered.is_ready() ? 0 : UINT64_MAX;
            witness.cold_request_refs = cold_surrendered.is_ready() ? 0 : UINT64_MAX;
            witness.cold_cpu_execution_refs = cold_surrendered.is_ready() ? 0 : UINT64_MAX;
            witness.hot_pins = current_pins;
            for (const auto & forward : directory_forward) witness.published_forward_mappings += forward.slot >= 0;
            witness.final_invariants_ok = ring_surrendered.is_ready() && cold_surrendered.is_ready() &&
                ring.invariants_ok && witness.scheduler_active == 0 && witness.scheduler_queued == 0 &&
                witness.hot_pins == 0 && witness.published_forward_mappings == 0;
            witness.write_count = 1;
            witness.written = true;
        }
    }

    llm_expert_provider_result bind(
            const llm_expert_bundle_descriptor & bundle,
            const llm_expert_selection & selection,
            llm_expert_graph_binding & binding) noexcept override {
        return bind_impl(nullptr, bundle, selection, binding);
    }

    llm_expert_provider_result bind_graph(
            ggml_context * graph_ctx,
            const llm_expert_bundle_descriptor & bundle,
            const llm_expert_selection & selection,
            llm_expert_graph_binding & binding) noexcept override {
        return bind_impl(graph_ctx, bundle, selection, binding);
    }

    llm_expert_provider_result bind_impl(
            ggml_context * graph_ctx,
            const llm_expert_bundle_descriptor & bundle,
            const llm_expert_selection & selection,
            llm_expert_graph_binding & binding) noexcept {
        LLM_EXPERT_TRACE_SCOPE("k3.provider", "bind", "layer", bundle.layer,
            "n_tokens", selection.n_tokens, "n_expert_used", selection.n_expert_used);
        std::lock_guard<std::mutex> lock(mutex);
        counters.bind_calls++;
        auto result = selection.validate();
        if (!result.is_ready() || bundle.layer < 0 || bundle.layer >= LLAMA_MAX_LAYERS) {
            return fail(result.is_ready()
                ? llm_expert_provider_result::failure(llm_expert_provider_error::invalid_descriptor)
                : result);
        }

        auto it = registrations.find(bundle.layer);
        if (it == registrations.end()) {
            if (layout_registry.sealed()) {
                return fail(llm_expert_provider_result::failure(llm_expert_provider_error::invalid_descriptor));
            }
            result = validate_source_bundle(bundle);
            if (!result.is_ready()) {
                return fail(result);
            }
            if (!prototype.has_value()) {
                prototype = bundle;
            }
            registrations.emplace(bundle.layer, bundle);
            counters.bundle_registrations++;
            counters.bundle_full_validations++;
        } else {
            counters.bundle_fast_path_hits++;
            if (!bundle_identity_matches(it->second, bundle)) {
                return fail(llm_expert_provider_result::failure(llm_expert_provider_error::invalid_descriptor));
            }
        }

        if (faults.binding != llm_expert_provider_error::none &&
            successful_bindings >= faults.binding_successes_before_failure) {
            return fail(llm_expert_provider_result::failure(faults.binding));
        }

        if (!pool) {
            binding = {};
            binding.provider_identity = this;
            binding.layer = bundle.layer;
            binding.layout_class_id = layout_class_for_layer(bundle.layer);
            binding.up = bundle.up;
            binding.gate = bundle.gate;
            binding.gate_up = bundle.gate_up;
            binding.down = bundle.down;
            binding.execution_ids = selection.logical_ids;
            binding.graph_epoch = epoch;
            binding.bootstrap = true;
            counters.bootstrap_bindings++;
        } else {
            const auto class_id = layout_class_for_layer(bundle.layer);
            if (class_id == LLM_EXPERT_LAYOUT_CLASS_INVALID || class_id >= pool->bundles.size()) {
                return fail(llm_expert_provider_result::failure(llm_expert_provider_error::invalid_binding));
            }
            const auto & hot_bundle = pool->bundles[class_id];
            ggml_tensor * execution_ids = selection.logical_ids;
            ggml_tensor * checkpoint_ids = nullptr;
            ggml_tensor * cpu_execution_ids = nullptr;
            if (graph_ctx != nullptr && multi_device) {
                checkpoint_ids = ggml_dup(graph_ctx, selection.logical_ids);
                ggml_format_name(checkpoint_ids, "expert_multi_checkpoint_ids-%d", bundle.layer);
                execution_ids = ggml_new_tensor_2d(graph_ctx, GGML_TYPE_I32,
                    selection.n_expert_used, selection.n_tokens);
                ggml_set_input(execution_ids);
                ggml_format_name(execution_ids, "expert_device_0_execution_ids-%d", bundle.layer);
            } else if (graph_ctx != nullptr) {
                if (config.cpu_cold_only) {
                    checkpoint_ids = ggml_dup(graph_ctx, selection.logical_ids);
                    ggml_format_name(checkpoint_ids, "expert_cpu_cold_checkpoint_ids-%d", bundle.layer);
                    execution_ids = ggml_dup(graph_ctx, checkpoint_ids);
                    ggml_format_name(execution_ids, "expert_cpu_cold_execution_ids-%d", bundle.layer);
                } else if (config.cold_mode && config.miss_policy != LLAMA_EXPERT_MISS_POLICY_PROMOTE_AND_GPU) {
                    checkpoint_ids = ggml_dup(graph_ctx, selection.logical_ids);
                    ggml_format_name(checkpoint_ids, "expert_checkpoint_ids-%d", bundle.layer);
                    execution_ids = ggml_dup(graph_ctx, checkpoint_ids);
                    ggml_format_name(execution_ids, "expert_execution_ids-%d", bundle.layer);
                    cpu_execution_ids = ggml_new_tensor_2d(graph_ctx, GGML_TYPE_I32,
                        selection.n_expert_used, selection.n_tokens);
                    ggml_set_input(cpu_execution_ids);
                    ggml_format_name(cpu_execution_ids, "expert_cpu_execution_ids-%d", bundle.layer);
                } else {
                    checkpoint_ids = ggml_dup(graph_ctx, selection.logical_ids);
                    ggml_format_name(checkpoint_ids, "expert_checkpoint_ids-%d", bundle.layer);
                    execution_ids = ggml_dup(graph_ctx, checkpoint_ids);
                    ggml_format_name(execution_ids, "expert_execution_ids-%d", bundle.layer);
                }
            }
            binding = {};
            binding.provider_identity = this;
            binding.layer = bundle.layer;
            binding.layout_class_id = class_id;
            binding.up = hot_bundle.up;
            binding.gate = hot_bundle.gate;
            binding.gate_up = hot_bundle.gate_up;
            binding.down = hot_bundle.down;
            binding.execution_ids = execution_ids;
            binding.checkpoint_ids = checkpoint_ids;
            binding.default_target_device = config.target_device != nullptr ?
                config.target_device : ggml_backend_buft_get_device(config.target_buffer_type);
            binding.generation_lease = std::static_pointer_cast<void>(pool);
            binding.graph_epoch = epoch;
            binding.cpu_cold_only = config.cpu_cold_only;
            if (!config.cpu_cold_only) counters.hot_bindings++;
            if (config.remote_single) {
                binding.remote_single = true;
                binding.remote_device = {
                    0, config.devices.front().target_device,
                    hot_bundle.up, hot_bundle.gate, hot_bundle.gate_up, hot_bundle.down,
                    execution_ids, std::static_pointer_cast<void>(pool),
                    device_transfer_delay_us.front(), {},
                };
                remote_single_bindings++;
            } else if (multi_device) {
                binding.checkpoint_ids = checkpoint_ids;
                binding.multi_device = true;
                binding.devices.reserve(device_pools.size());
                for (size_t index = 0; index < device_pools.size(); ++index) {
                    const auto & device_bundle = device_pools[index]->bundles[class_id];
                    ggml_tensor * device_ids = execution_ids;
                    if (index != 0) {
                        device_ids = ggml_new_tensor_2d(graph_ctx, GGML_TYPE_I32,
                            selection.n_expert_used, selection.n_tokens);
                        ggml_set_input(device_ids);
                        ggml_format_name(device_ids, "expert_device_%zu_execution_ids-%d", index, bundle.layer);
                    }
                    binding.devices.push_back({
                        llm_expert_device_id(index), config.devices[index].target_device,
                        device_bundle.up, device_bundle.gate, device_bundle.gate_up, device_bundle.down,
                        device_ids, std::static_pointer_cast<void>(device_pools[index]),
                        device_transfer_delay_us[index], {},
                    });
                    binding.devices.back().h2d_dependencies.reserve(config.n_expert_used);
                }
                multi_device_bindings++;
                device_binding_vector_elements += binding.devices.size();
            } else if (cpu_execution_ids != nullptr) {
                const auto & cpu = cold_cache->bundle(class_id);
                binding.cpu_up = cpu.up;
                binding.cpu_gate = cpu.gate;
                binding.cpu_gate_up = cpu.gate_up;
                binding.cpu_down = cpu.down;
                binding.checkpoint_ids = checkpoint_ids;
                binding.cpu_execution_ids = cpu_execution_ids;
                binding.hybrid = true;
                hybrid_bindings++;
            }
        }
        binding.logical_ids = selection.logical_ids;
        successful_bindings++;
        result = binding.validate(selection);
        return result.is_ready() ? result : fail(result);
    }

    bool uses_hybrid_graph() const noexcept override {
        return config.remote_single || multi_device ||
            (!config.cpu_cold_only && config.cold_mode &&
             config.miss_policy != LLAMA_EXPERT_MISS_POLICY_PROMOTE_AND_GPU);
    }

    llm_expert_provider_result prepare(
            const std::vector<llm_expert_graph_binding> & bindings,
            llm_expert_execution_plan & plan,
            uint64_t sequence_owner,
            bool sequence_start) noexcept override {
        LLM_EXPERT_TRACE_SCOPE("k3.provider", "prepare", "binding_count", bindings.size(),
            "sequence_start", sequence_start);
        plan.reset();
        try {
            plan.reserve(1);
        } catch (...) {
            auto result = llm_expert_provider_result::failure(llm_expert_provider_error::allocation_failed);
            plan.set_result(result);
            return result;
        }
        std::lock_guard<std::mutex> lock(mutex);
        counters.prepare_calls++;
        if (bindings.empty()) {
            plan.set_result(llm_expert_provider_result::success());
            return llm_expert_provider_result::success();
        }

        if (!pool) {
            auto result = llm_expert_provider_result::failure(llm_expert_provider_error::initialization_failed);
            plan.set_result(result);
            return fail(result);
        }
        if (config.cold_mode) {
            const auto memory_result = system_memory_result(
                system_memory_budget.revalidate("provider_prepare"));
            if (!memory_result.is_ready()) {
                plan.set_result(memory_result);
                return fail(memory_result);
            }
        }

        size_t max_elements = 0;
        for (const auto & binding : bindings) {
            if (binding.provider_identity != this || binding.execution_ids == nullptr ||
                binding.execution_ids->ne[0] != int64_t(config.n_expert_used) || binding.execution_ids->ne[1] < 0 ||
                registrations.find(binding.layer) == registrations.end()) {
                auto result = llm_expert_provider_result::failure(llm_expert_provider_error::invalid_binding);
                plan.set_result(result);
                return fail(result);
            }
            if (!extent_is_safe(uint64_t(binding.execution_ids->ne[1]))) {
                auto result = llm_expert_provider_result::failure(llm_expert_provider_error::unsupported_configuration);
                plan.set_result(result);
                return fail(result);
            }
            if (binding.bootstrap || binding.graph_epoch != epoch || binding.generation_lease.get() != pool.get() ||
                !binding_uses_current_pool(binding)) {
                auto result = llm_expert_provider_result::failure(llm_expert_provider_error::invalid_binding);
                plan.set_result(result);
                return fail(result);
            }
            const uint64_t elements = uint64_t(binding.execution_ids->ne[0])*uint64_t(binding.execution_ids->ne[1]);
            if (elements > SIZE_MAX) {
                auto result = llm_expert_provider_result::failure(llm_expert_provider_error::unsupported_configuration);
                plan.set_result(result);
                return fail(result);
            }
            max_elements = std::max(max_elements, size_t(elements));
        }

        if (active_request) {
            exclusive_busy_failures++;
            auto result = llm_expert_provider_result::failure(llm_expert_provider_error::busy);
            plan.set_result(result);
            return fail(result);
        }

        if (config.prefetch_config.supplied &&
                config.prefetch_config.value.policy != LLAMA_EXPERT_PREFETCH_POLICY_OFF) {
            const auto sequence = begin_phase10_sequence_locked(
                sequence_owner, sequence_start);
            if (!sequence.is_ready()) {
                plan.set_result(sequence);
                return fail(sequence);
            }
        }

        try {
            if (element_unique.size() < max_elements || logical_id_scratch.size() < max_elements ||
                execution_id_scratch.size() < max_elements || cpu_execution_id_scratch.size() < max_elements ||
                last_logical_ids.size() < max_elements || last_execution_ids.size() < max_elements ||
                last_cpu_execution_ids.size() < max_elements) {
                element_unique.resize(max_elements);
                logical_id_scratch.resize(max_elements);
                execution_id_scratch.resize(max_elements);
                cpu_execution_id_scratch.resize(max_elements);
                last_logical_ids.resize(max_elements);
                last_execution_ids.resize(max_elements);
                last_cpu_execution_ids.resize(max_elements);
                scratch_reservations++;
            }
            if (multi_device) {
                device_execution_id_scratch.resize(device_pools.size());
                for (auto & ids : device_execution_id_scratch) ids.resize(max_elements, -1);
            }
        } catch (const std::bad_alloc &) {
            auto result = llm_expert_provider_result::failure(llm_expert_provider_error::allocation_failed);
            plan.set_result(result);
            return fail(result);
        }

        auto policy_started = config.cpu_cold_only ? cold_cache->policy_request_begin() :
            cache_policy_result(hot_policy.request_begin());
        const bool hot_policy_started = !config.cpu_cold_only && policy_started.is_ready();
        if (!config.cpu_cold_only && hot_policy_started && cold_cache) {
            policy_started = cold_cache->policy_request_begin();
        }
        if (!policy_started.is_ready()) {
            if (hot_policy_started) (void) hot_policy.request_end(false, false);
            plan.set_result(policy_started);
            return fail(policy_started);
        }
        hot_policy_phase = llm_expert_cache_policy_phase::prefill;
        request_ubatch_ordinal = 0;
        active_request_id = ++next_request_id;
        active_request = true;
        [[maybe_unused]] const uint64_t trace_request_id = llm_perfetto_trace_id(
            llm_perfetto_trace_domain::request, active_request_id);
        LLM_EXPERT_TRACE_ASYNC_BEGIN("k3.provider", "provider_request", trace_request_id,
            "request_id", active_request_id, "graph_epoch", epoch, "binding_count", bindings.size());
        request_pin_count = 0;
        cpu_execution_pin_count = 0;
        current_token_next_layer = 0;
        for (auto & route : current_token_routes) route.clear();
        last_remap_error = llm_expert_provider_error::none;
        requests++;
        counters.handles_acquired++;
        plan.add_handle({ this, active_request_id });
        plan.set_result(llm_expert_provider_result::success());
        return llm_expert_provider_result::success();
    }

    llm_expert_provider_stats get_stats() const noexcept override {
        std::lock_guard<std::mutex> lock(mutex);
        llm_expert_provider_stats result = counters;
        result.objects_created = 1;
        result.requested_capacity = config.cpu_cold_only ? 0 : config.capacity;
        result.effective_capacity = config.cpu_cold_only ? 0 : (pool ? config.capacity : 0);
        result.pool_bytes = config.cpu_cold_only ? 0 :
            (pool && pool->buffer ? ggml_backend_buffer_get_size(pool->buffer.get()) : 0);
        result.graph_epoch = epoch;
        return result;
    }

    llm_expert_cold_scalar_snapshot cold_cache_scalar_snapshot() const noexcept override {
        std::lock_guard<std::mutex> lock(mutex);
        if (!cold_cache) return {};
        const auto cold = cold_cache->scalar_diagnostics();
        return {
            cold.available,
            cold.requested_bytes,
            cold.actual_bytes,
            cold.effective_slots,
            cold.occupancy,
            cold.requests,
            cold.hits,
            cold.misses,
            cold.admissions,
            cold.evictions,
            cold.residency_digest,
        };
    }

    llm_expert_provider_result validate_context_extent(
            uint32_t n_ctx,
            uint32_t n_ubatch) noexcept override {
        std::lock_guard<std::mutex> lock(mutex);
        const uint32_t extent = std::min(n_ctx, n_ubatch);
        last_context_n_ctx = n_ctx;
        last_context_n_ubatch = n_ubatch;
        last_context_extent = extent;
        last_required_capacity = required_capacity(extent);
        context_validations++;
        if (n_ctx == 0 || n_ubatch == 0 || !extent_is_safe(extent)) {
            context_rejections++;
            return fail(llm_expert_provider_result::failure(llm_expert_provider_error::unsupported_configuration));
        }
        return llm_expert_provider_result::success();
    }

    llm_expert_provider_result remap_checkpoint(
            const llm_expert_graph_binding & binding,
            const int32_t * logical_ids,
            size_t logical_id_count,
            int32_t * execution_ids,
            bool (*abort_callback)(void *),
            void * abort_callback_data) noexcept override {
        std::unique_lock<std::mutex> ordered_lock(ordered_remap_mutex, std::defer_lock);
        if (deterministic_policy_terminals || config.cpu_cold_only) ordered_lock.lock();
        std::unique_lock<std::mutex> lock(mutex);
        return remap_checkpoint_locked(binding, logical_ids, logical_id_count, execution_ids, nullptr, nullptr,
            abort_callback, abort_callback_data, &lock);
    }

    llm_expert_provider_result remap_checkpoint_tensor(
            const llm_expert_graph_binding & binding,
            ggml_backend_t execution_backend,
            bool (*abort_callback)(void *),
            void * abort_callback_data) noexcept override {
        return remap_checkpoint_tensor_multi_device(
            binding, execution_backend, nullptr, 0, abort_callback, abort_callback_data);
    }

    llm_expert_provider_result remap_checkpoint_tensor_multi_device(
            const llm_expert_graph_binding & binding,
            ggml_backend_t execution_backend,
            const ggml_backend_t * device_backends,
            size_t device_backend_count,
            bool (*abort_callback)(void *),
            void * abort_callback_data) noexcept override {
        LLM_EXPERT_TRACE_SCOPE("k3.provider", "remap_checkpoint_tensor", "layer", binding.layer,
            "layout_class_id", binding.layout_class_id, "graph_epoch", binding.graph_epoch);
        std::unique_lock<std::mutex> ordered_lock(ordered_remap_mutex, std::defer_lock);
        if (deterministic_policy_terminals || config.cpu_cold_only) ordered_lock.lock();
        std::unique_lock<std::mutex> lock(mutex);
        (void) execution_backend;
        ggml_tensor * checkpoint_ids = binding.checkpoint_ids != nullptr ?
            binding.checkpoint_ids : binding.execution_ids;
        if (checkpoint_ids == nullptr || binding.execution_ids == nullptr ||
            binding.execution_ids == binding.logical_ids || binding.execution_ids->type != GGML_TYPE_I32 ||
            binding.execution_ids->ne[0] < 0 || binding.execution_ids->ne[1] < 0 ||
            (binding.hybrid && binding.cpu_execution_ids == nullptr) ||
            (binding.multi_device && binding.devices.size() != device_pools.size())) {
            return fail(llm_expert_provider_result::failure(llm_expert_provider_error::invalid_binding));
        }
        const uint64_t count64 = uint64_t(binding.execution_ids->ne[0])*uint64_t(binding.execution_ids->ne[1]);
        if (count64 > logical_id_scratch.size() || count64 > execution_id_scratch.size()) {
            return fail(llm_expert_provider_result::failure(llm_expert_provider_error::unsupported_configuration));
        }
        const size_t count = size_t(count64);
        const size_t bytes = count*sizeof(int32_t);
        ggml_backend_tensor_get(checkpoint_ids, logical_id_scratch.data(), 0, bytes);
        execution_id_read_bytes += bytes;
        auto result = binding.multi_device ?
            remap_checkpoint_multi_device_locked(
                binding, logical_id_scratch.data(), count,
                device_backends, device_backend_count,
                abort_callback, abort_callback_data, &lock) :
            remap_checkpoint_locked(
                binding, logical_id_scratch.data(), count, execution_id_scratch.data(),
                binding.hybrid ? cpu_execution_id_scratch.data() : nullptr, execution_backend,
                abort_callback, abort_callback_data, &lock);
        if (!result.is_ready()) {
            return result;
        }
        if (binding.multi_device) {
            for (size_t device = 0; device < binding.devices.size(); ++device) {
                ggml_backend_tensor_set(binding.devices[device].execution_ids,
                    device_execution_id_scratch[device].data(), 0, bytes);
            }
            execution_id_write_bytes += binding.devices.size()*bytes;
        } else if (binding.hybrid) {
            ggml_backend_tensor_set(binding.checkpoint_ids, execution_id_scratch.data(), 0, bytes);
            ggml_backend_tensor_set(binding.cpu_execution_ids, cpu_execution_id_scratch.data(), 0, bytes);
            execution_id_write_bytes += 2*bytes;
        } else {
            ggml_backend_tensor_set(checkpoint_ids, execution_id_scratch.data(), 0, bytes);
            execution_id_write_bytes += bytes;
        }
        counters.callbacks++;
        counters.synchronizations++;
        synchronization_checkpoints++;
        return llm_expert_provider_result::success();
    }

    llm_expert_provider_result remap_checkpoint_multi_device_locked(
            const llm_expert_graph_binding & binding,
            const int32_t * logical_ids,
            size_t logical_id_count,
            const ggml_backend_t * device_backends,
            size_t device_backend_count,
            bool (*abort_callback)(void *),
            void * abort_callback_data,
            std::unique_lock<std::mutex> * provider_lock) noexcept {
        (void) llm_perfetto_trace_maybe_start_decode_window(active_request_id, binding.layer);
        LLM_EXPERT_TRACE_SCOPE("k3.provider", "acquire_and_remap", "request_id", active_request_id,
            "layer", binding.layer, "layout_class_id", binding.layout_class_id,
            "selected_key_count", logical_id_count);
        LLM_EXPERT_TRACE_SCOPE("k3.graph", "expert_layer_execution", "request_id", active_request_id,
            "layer", binding.layer, "selected_key_count", logical_id_count);
        LLM_EXPERT_TRACE_CUDA_SCOPE(llm_perfetto_trace_id(llm_perfetto_trace_domain::request, active_request_id));
        if (!multi_device || !active_request || !pool || logical_ids == nullptr ||
            binding.provider_identity != this || binding.bootstrap ||
            binding.graph_epoch != epoch || binding.generation_lease.get() != pool.get() ||
            !binding_uses_current_pool(binding) || binding.devices.size() != device_pools.size() ||
            binding.execution_ids == nullptr || binding.execution_ids->ne[0] != int64_t(config.n_expert_used) ||
            binding.execution_ids->ne[1] < 0) {
            return fail(llm_expert_provider_result::failure(llm_expert_provider_error::invalid_binding));
        }
        const uint64_t expected_count = uint64_t(binding.execution_ids->ne[0])*
            uint64_t(binding.execution_ids->ne[1]);
        if (expected_count != logical_id_count || logical_id_count > element_unique.size() ||
            !extent_is_safe(uint64_t(binding.execution_ids->ne[1]))) {
            return fail(llm_expert_provider_result::failure(
                llm_expert_provider_error::unsupported_configuration));
        }
        const auto registration = registrations.find(binding.layer);
        if (registration == registrations.end() || !validate_request_pins_locked()) {
            return fail(llm_expert_provider_result::failure(
                registration == registrations.end() ? llm_expert_provider_error::invalid_binding :
                    llm_expert_provider_error::stale_generation));
        }
        const auto abort_requested = [&]() {
            if (abort_callback == nullptr) return false;
            if (provider_lock != nullptr) provider_lock->unlock();
            const bool requested = abort_callback(abort_callback_data);
            if (provider_lock != nullptr) provider_lock->lock();
            return requested;
        };
        if (abort_requested()) {
            return fail(llm_expert_provider_result::failure(llm_expert_provider_error::cancelled));
        }

        size_t unique_count = 0;
        for (size_t index = 0; index < logical_id_count; ++index) {
            const int32_t expert = logical_ids[index];
            if (expert < 0 || expert >= int32_t(n_expert)) {
                return fail(llm_expert_provider_result::failure(llm_expert_provider_error::invalid_key));
            }
            const llm_expert_key key = { binding.layer, expert };
            size_t unique_index = 0;
            while (unique_index < unique_count && !expert_key_matches(unique_keys[unique_index], key)) {
                unique_index++;
            }
            if (unique_index == unique_count) {
                // A prefill checkpoint can contain up to n_tokens distinct
                // top-k sets.  The provider scratch space is sized to the total
                // hot capacity, just as in the single-device path; limiting this
                // to one token's top-k width rejects every non-trivial prefill.
                if (unique_count >= config.capacity) {
                    return fail(llm_expert_provider_result::failure(
                        llm_expert_provider_error::unsupported_configuration));
                }
                unique_keys[unique_count] = key;
                unique_slots[unique_count] = -1;
                unique_lane_counts[unique_count] = 0;
                unique_index = unique_count++;
            }
            element_unique[index] = int32_t(unique_index);
            unique_lane_counts[unique_index]++;
        }

        const auto requested_phase = binding.execution_ids->ne[1] > 1 ?
            llm_expert_cache_policy_phase::prefill : llm_expert_cache_policy_phase::decode;
        const bool runtime_device_backends = device_backends != nullptr || device_backend_count != 0;
        if (runtime_device_backends &&
            (device_backends == nullptr || device_backend_count != device_pools.size())) {
            return fail(llm_expert_provider_result::failure(
                llm_expert_provider_error::invalid_binding));
        }
        if (runtime_device_backends &&
            device_transfer_event_capabilities.size() != binding.devices.size()) {
            return fail(llm_expert_provider_result::failure(
                llm_expert_provider_error::initialization_failed));
        }
        bool async_decode_transfers = runtime_device_backends &&
            requested_phase == llm_expert_cache_policy_phase::decode;
        for (size_t device = 0; device < binding.devices.size(); ++device) {
            if (!binding.devices[device].h2d_dependencies.empty()) {
                return fail(llm_expert_provider_result::failure(
                    llm_expert_provider_error::stale_generation));
            }
            if (!runtime_device_backends) continue;
            if (device_backends[device] == nullptr ||
                ggml_backend_get_device(device_backends[device]) != config.devices[device].target_device) {
                return fail(llm_expert_provider_result::failure(
                    llm_expert_provider_error::invalid_binding));
            }
            if (async_decode_transfers && !device_transfer_event_capabilities[device]) {
                return fail(llm_expert_provider_result::failure(
                    llm_expert_provider_error::unsupported_configuration));
            }
        }
        if (request_ubatch_ordinal == UINT64_MAX) {
            return fail(llm_expert_provider_result::failure(
                llm_expert_provider_error::unsupported_configuration));
        }
        const uint64_t ubatch_ordinal = request_ubatch_ordinal + 1;
        auto result = cache_policy_result(hot_policy.set_ubatch_ordinal(ubatch_ordinal));
        if (result.is_ready()) result = cold_cache->policy_set_ubatch_ordinal(ubatch_ordinal);
        if (!result.is_ready()) return fail(result);
        request_ubatch_ordinal = ubatch_ordinal;
        if (requested_phase != hot_policy_phase) {
            result = cache_policy_result(hot_policy.phase_transition(requested_phase));
            if (result.is_ready()) result = cold_cache->policy_phase_transition(requested_phase);
            if (!result.is_ready()) return fail(result);
            hot_policy_phase = requested_phase;
        }
        for (size_t index = 0; index < unique_count; ++index) policy_unique_indices[index] = uint32_t(index);
        std::sort(policy_unique_indices.begin(), policy_unique_indices.begin() + unique_count,
            [&](uint32_t lhs, uint32_t rhs) {
                return unique_keys[lhs].expert < unique_keys[rhs].expert;
            });
        for (size_t order = 0; order < unique_count; ++order) {
            const uint32_t index = policy_unique_indices[order];
            result = cache_policy_result(hot_policy.demand(
                policy_key_for(unique_keys[index]), unique_lane_counts[index],
                payload_for_key(unique_keys[index]), hot_physical_slot_footprint_bytes));
            if (!result.is_ready()) return fail(result);
        }

        std::fill(slot_selected.begin(), slot_selected.end(), uint8_t(0));
        size_t miss_count = 0;
        size_t hit_count = 0;
        for (size_t order = 0; order < unique_count; ++order) {
            const uint32_t index = policy_unique_indices[order];
            const auto & key = unique_keys[index];
            const auto owner = llm_expert_owner_device(key.expert, uint32_t(device_pools.size()));
            const auto & forward = directory_forward[forward_index(key)];
            if (forward.slot < 0) {
                device_runtime_stats[owner].misses++;
                miss_unique_indices[miss_count++] = index;
                continue;
            }
            if (!forward_entry_matches(key, forward) ||
                directory_slots[forward.slot].device_id != owner) {
                metadata_mismatches++;
                return fail(llm_expert_provider_result::failure(
                    llm_expert_provider_error::metadata_mismatch));
            }
            unique_slots[index] = forward.slot;
            slot_selected[forward.slot] = 1;
            auto & entry = directory_slots[forward.slot];
            result = cache_policy_result(hot_policy.hit(uint32_t(forward.slot), entry.generation));
            if (!result.is_ready()) return fail(result);
            if (entry.has_cold_backing) {
                const llm_cold_reference backing = {
                    entry.cold_slot, entry.cold_generation, entry.layout_class_id };
                result = cold_cache->policy_shadow_hit(key, backing, unique_lane_counts[index]);
                if (result.is_ready()) result = cold_cache->acquire(backing, llm_cold_reference_kind::request);
                if (result.is_ready()) result = cold_cache->release(backing, llm_cold_reference_kind::request);
                if (!result.is_ready()) return fail(result);
            }
            device_runtime_stats[owner].hits++;
            hit_count++;
        }
        result = release_request_pins_locked();
        if (!result.is_ready()) return fail(result);

        for (size_t index = 0; index < miss_count; ++index) {
            const uint32_t unique_index = miss_unique_indices[index];
            const auto owner = llm_expert_owner_device(
                unique_keys[unique_index].expert, uint32_t(device_pools.size()));
            int32_t selected = -1;
            result = select_hot_slot_for_device_locked(
                unique_keys[unique_index], owner, candidate_slots.data(), index, selected);
            if (!result.is_ready()) return fail(result);
            candidate_slots[index] = uint32_t(selected);
            unique_slots[unique_index] = selected;
        }

        const auto & device_lane_capacities = device_transfer_lane_capacities;
        if (device_lane_capacities.size() != device_pools.size()) {
            return fail(llm_expert_provider_result::failure(
                llm_expert_provider_error::initialization_failed));
        }
        std::vector<llm_expert_request_handle> scheduler_handles(miss_count);
        std::vector<llm_expert_request_state> scheduler_states(
            miss_count, llm_expert_request_state::free);
        std::vector<std::vector<llm_transfer_binding>> device_transfers(device_pools.size());
        std::vector<std::vector<uint64_t>> device_transfer_hot_generations(device_pools.size());
        for (size_t index = 0; index < miss_count; ++index) async_flights[index] = {};
        auto fail_multi = [&](llm_expert_provider_result failure) {
            // Direct storage reads target transfer-ring lanes. Drain every
            // producer before cancelling H2D work or releasing those lanes.
            for (size_t index = 0; index < miss_count; ++index) {
                auto & flight = async_flights[index];
                if (!flight.read_active) continue;
                (void) config.async_transport->cancel_read(flight.handle);
                llm_expert_async_read_completion discarded;
                if (provider_lock != nullptr) provider_lock->unlock();
                const auto drained = config.async_transport->wait_read(flight.handle, discarded);
                if (provider_lock != nullptr) provider_lock->lock();
                (void) config.async_transport->release_read(flight.handle);
                config.storage->record_async_read(
                    flight.destination_count, discarded.bytes_completed,
                    drained == llm_expert_async_result::closed ?
                        llm_expert_storage_error::cancelled : llm_expert_storage_error::io_error,
                    discarded.native_error);
                flight.read_active = false;
            }
            for (size_t device = 0; device < binding.devices.size(); ++device) {
                for (const auto & dependency : binding.devices[device].h2d_dependencies) {
                    (void) transfer_rings[device]->cancel_after_h2d({
                        dependency.lane, dependency.lane_generation, dependency.layout_class_id });
                }
                binding.devices[device].h2d_dependencies.clear();
            }
            for (auto & ring : transfer_rings) (void) ring->cleanup_failed_lanes();
            (void) release_request_pins_locked();
            for (size_t index = 0; index < miss_count; ++index) {
                auto & flight = async_flights[index];
                if (flight.reserved) {
                    (void) cold_cache->fail_reservation(flight.key, flight.cold);
                    flight.reserved = false;
                }
            }
            for (size_t index = 0; index < miss_count; ++index) {
                const uint32_t slot = candidate_slots[index];
                if (slot >= directory_slots.size()) continue;
                auto & entry = directory_slots[slot];
                if (entry.state == hot_slot_state::loading) {
                    (void) hot_policy.load_failed(slot, entry.generation);
                } else if (entry.state == hot_slot_state::ready || entry.state == hot_slot_state::pinned) {
                    (void) hot_policy.remove_resident(slot, entry.generation);
                }
                clear_forward_locked(entry.key, slot, entry.generation);
                if (entry.has_cold_backing) {
                    (void) cold_cache->release(
                        { entry.cold_slot, entry.cold_generation, entry.layout_class_id },
                        llm_cold_reference_kind::hot);
                }
                const auto device = entry.device_id;
                const auto device_slot = entry.device_slot;
                const auto generation = entry.generation;
                entry = {};
                entry.device_id = device;
                entry.device_slot = device_slot;
                entry.generation = generation;
            }
            for (size_t index = 0; index < scheduler_handles.size(); ++index) {
                if (!scheduler_handles[index].valid()) continue;
                auto state = scheduler_states[index];
                if (state >= llm_expert_request_state::queued &&
                    state <= llm_expert_request_state::device_preparing) {
                    (void) config.scheduler->transition(
                        scheduler_handles[index], state, llm_expert_request_state::draining);
                    (void) config.scheduler->finish(
                        scheduler_handles[index], llm_expert_request_state::failed);
                }
                (void) config.scheduler->release_terminal(scheduler_handles[index]);
            }
            last_remap_error = failure.error;
            return fail(failure);
        };
        auto flush_device_transfers = [&]() noexcept {
            auto wave_result = llm_expert_provider_result::success();
            size_t participating_devices = 0;
            bool inject_failure = false;
            const auto device_failure_active = [&](size_t device) {
                return device_transfer_failure_for_testing[device] &&
                    (!device_transfer_failure_decode_only_for_testing[device] ||
                        hot_policy_phase == llm_expert_cache_policy_phase::decode);
            };
            for (size_t device = 0; device < device_transfers.size(); ++device) {
                if (device_transfers[device].empty()) continue;
                participating_devices++;
                inject_failure = inject_failure || device_failure_active(device);
            }
            if (participating_devices == 0) return wave_result;

            if (async_decode_transfers) {
                if (inject_failure) {
                    injected_device_failure_waves++;
                    injected_device_failure_participants += participating_devices;
                }
                for (size_t device = 0; device < device_transfers.size(); ++device) {
                    if (device_transfers[device].empty() ||
                        device_failure_active(device)) continue;
                    wave_result = transfer_rings[device]->transfer_wave(
                        device_backends[device], device_transfers[device]);
                    if (!wave_result.is_ready()) break;
                    if (config.async_cold_fill) {
                        for (const auto & transfer : device_transfers[device]) {
                            (void) transfer_rings[device]->try_queue_cold_fill(
                                *cold_cache, transfer.lane);
                        }
                    }
                    auto & dependencies = binding.devices[device].h2d_dependencies;
                    GGML_ASSERT(device_transfers[device].size() ==
                        device_transfer_hot_generations[device].size());
                    for (size_t index = 0; index < device_transfers[device].size(); ++index) {
                        const auto & transfer = device_transfers[device][index];
                        dependencies.push_back({
                            transfer.lane.lane, transfer.lane.generation,
                            transfer.lane.layout_class_id, transfer.hot_slot,
                            device_transfer_hot_generations[device][index],
                        });
                    }
                }
                if (wave_result.is_ready() && inject_failure) {
                    wave_result = llm_expert_provider_result::failure(
                        llm_expert_provider_error::copy_failed);
                }
                if (!wave_result.is_ready()) {
                    for (size_t device = 0; device < binding.devices.size(); ++device) {
                        for (const auto & dependency : binding.devices[device].h2d_dependencies) {
                            (void) transfer_rings[device]->cancel_after_h2d({
                                dependency.lane, dependency.lane_generation,
                                dependency.layout_class_id });
                        }
                        binding.devices[device].h2d_dependencies.clear();
                    }
                    if (inject_failure) injected_device_failure_drained_waves++;
                    return wave_result;
                }
                for (size_t device = 0; device < device_transfers.size(); ++device) {
                    device_transfers[device].clear();
                    device_transfer_hot_generations[device].clear();
                }
                provider_h2d_async_decode_waves++;
                LLM_EXPERT_TRACE_INSTANT("k3.transfer", "provider_h2d_async_enqueue",
                    "layer", binding.layer, "participating_devices", participating_devices);
                return wave_result;
            }

            const auto join_started = std::chrono::steady_clock::now();
            try {
                if (inject_failure) {
                    injected_device_failure_waves++;
                    injected_device_failure_participants += participating_devices;
                }
                std::atomic<size_t> started_devices { 0 };
                std::vector<std::future<llm_expert_provider_result>> transfers;
                transfers.reserve(device_transfers.size());
                for (size_t device = 0; device < device_transfers.size(); ++device) {
                    if (device_transfers[device].empty()) continue;
                    transfers.emplace_back(std::async(std::launch::async, [&, device] {
                        if (inject_failure) {
                            started_devices.fetch_add(1, std::memory_order_release);
                            while (started_devices.load(std::memory_order_acquire) < participating_devices) {
                                std::this_thread::yield();
                            }
                            if (device_failure_active(device)) {
                                return llm_expert_provider_result::failure(
                                    llm_expert_provider_error::copy_failed);
                            }
                        }
                        return transfer_rings[device]->transfer_wave_blocking(
                            device_transfers[device], config.async_cold_fill ? cold_cache.get() : nullptr);
                    }));
                }
                for (auto & transfer : transfers) {
                    const auto transferred = transfer.get();
                    if (wave_result.is_ready() && !transferred.is_ready()) wave_result = transferred;
                }
                if (inject_failure) injected_device_failure_drained_waves++;
            } catch (...) {
                wave_result = llm_expert_provider_result::failure(
                    llm_expert_provider_error::allocation_failed);
            }
            const uint64_t join_ns = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - join_started).count());
            provider_h2d_join_waves++;
            provider_h2d_join_time_ns += join_ns;
            provider_h2d_join_max_ns = std::max(provider_h2d_join_max_ns, join_ns);
            if (hot_policy_phase == llm_expert_cache_policy_phase::decode) {
                provider_h2d_join_decode_waves++;
                provider_h2d_join_decode_time_ns += join_ns;
                provider_h2d_join_decode_max_ns = std::max(provider_h2d_join_decode_max_ns, join_ns);
            }
            LLM_EXPERT_TRACE_INSTANT("k3.transfer", "provider_h2d_join",
                "layer", binding.layer, "wait_ns", join_ns,
                "participating_devices", participating_devices);
            if (wave_result.is_ready()) {
                for (size_t device = 0; device < device_transfers.size(); ++device) {
                    device_transfers[device].clear();
                    device_transfer_hot_generations[device].clear();
                }
            }
            return wave_result;
        };

        const auto stage_miss = [&](size_t index) noexcept {
            const uint32_t unique_index = miss_unique_indices[index];
            const auto & key = unique_keys[unique_index];
            const auto owner = llm_expert_owner_device(key.expert, uint32_t(device_pools.size()));
            const uint32_t global_slot = candidate_slots[index];
            auto & flight = async_flights[index];
            auto staged = llm_expert_provider_result::success();
            if (flight.direct_storage) {
                const auto & loading = directory_slots[global_slot];
                if (loading.state != hot_slot_state::loading ||
                        !expert_key_matches(loading.key, key)) {
                    return llm_expert_provider_result::failure(
                        llm_expert_provider_error::metadata_mismatch);
                }
            } else {
                staged = prepare_hot_slot_locked(global_slot, key, cold_references[index]);
            }
            if (!staged.is_ready()) return staged;
            auto & entry = directory_slots[global_slot];
            auto * ring = ring_for_slot(global_slot);
            if (ring == nullptr || entry.device_id != owner) {
                return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
            }
            uint64_t transport_epoch = 0;
            {
                LLM_EXPERT_TRACE_SCOPE("k3.provider", "multi_device_transport_epoch_snapshot",
                    "phase", 2);
                transport_epoch = config.async_transport->diagnostics().transport_epoch;
            }
            if (flight.direct_storage) {
                transfer_lanes[index] = flight.lane;
            } else {
                staged = ring->reserve(*cold_cache, cold_references[index], entry.device_slot,
                    entry.generation, transfer_lanes[index], {
                        transport_epoch,
                        scheduler_handles[index].slot, scheduler_handles[index].generation, key,
                        binding.layout_class_id, owner,
                    });
                if (staged.is_ready()) staged = ring->stage(
                    transfer_lanes[index], cold_cache->bundle(cold_references[index].layout_class_id));
            }
            if (!staged.is_ready()) return staged;
            device_transfers[owner].push_back({
                transfer_lanes[index],
                device_pools[owner]->bundles[binding.layout_class_id],
                entry.device_slot,
            });
            device_transfer_hot_generations[owner].push_back(entry.generation);
            if (config.scheduler->transition(scheduler_handles[index],
                    llm_expert_request_state::host_ready,
                    llm_expert_request_state::h2d_in_flight) !=
                llm_expert_schedule_disposition::admitted) {
                return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
            }
            scheduler_states[index] = llm_expert_request_state::h2d_in_flight;
            const uint32_t lane_capacity = device_lane_capacities[owner];
            if (device_transfers[owner].size() >= lane_capacity) {
                staged = flush_device_transfers();
            }
            return staged;
        };

        const bool issue_ahead_storage = config.storage != nullptr && config.async_transport != nullptr;
        if (issue_ahead_storage) {
            size_t submitted_count = 0;
            for (size_t index = 0; index < miss_count; ++index) {
                if (abort_requested()) {
                    return fail_multi(llm_expert_provider_result::failure(
                        llm_expert_provider_error::cancelled));
                }
                const uint32_t unique_index = miss_unique_indices[index];
                const auto & key = unique_keys[unique_index];
                const auto owner = llm_expert_owner_device(key.expert, uint32_t(device_pools.size()));
                auto & flight = async_flights[index];
                flight.key = key;
                llm_expert_request_metadata metadata;
                metadata.layout_class_id = binding.layout_class_id;
                metadata.target_device = owner;
                metadata.reserved_storage_bytes = payload_for_key(key);
                metadata.reserved_h2d_bytes = payload_for_key(key);
                const auto scheduled = config.scheduler->enqueue(
                    key, llm_expert_priority::demand_current_layer,
                    llm_expert_readiness::device_ready, metadata);
                if (scheduled.disposition != llm_expert_schedule_disposition::admitted) {
                    return fail_multi(llm_expert_provider_result::failure(
                        scheduled.disposition == llm_expert_schedule_disposition::generation_exhausted ?
                            llm_expert_provider_error::generation_exhausted :
                            llm_expert_provider_error::busy));
                }
                scheduler_handles[index] = scheduled.handle;
                scheduler_states[index] = llm_expert_request_state::queued;
                flight.handle = scheduled.handle;
                flight.flight_id = {
                    config.async_transport->diagnostics().transport_epoch,
                    scheduled.handle.slot, scheduled.handle.generation,
                    key, binding.layout_class_id, owner,
                };
            }

            for (size_t index = 0; index < miss_count; ++index) {
                auto & flight = async_flights[index];
                llm_cold_demand_lookup lookup = llm_cold_demand_lookup::missing;
                result = cold_cache->lookup_demand(flight.key, flight.cold, lookup);
                if (!result.is_ready()) return fail_multi(result);
                if (lookup == llm_cold_demand_lookup::joined_loading) {
                    if (!config.async_cold_fill) {
                        return fail_multi(llm_expert_provider_result::failure(
                            llm_expert_provider_error::metadata_mismatch));
                    }
                    flight.cold = {};
                    lookup = llm_cold_demand_lookup::missing;
                }
                flight.cold_hit = lookup == llm_cold_demand_lookup::ready;
                flight.direct_storage = lookup == llm_cold_demand_lookup::missing;
                flight.reserved = false;
                cold_references[index] = flight.cold;
            }

            for (size_t taken_count = 0; taken_count < miss_count; ++taken_count) {
                llm_expert_request_snapshot snapshot;
                const auto selected = config.scheduler->take_next(snapshot);
                size_t index = miss_count;
                if (selected.disposition == llm_expert_schedule_disposition::admitted) {
                    for (size_t candidate = 0; candidate < miss_count; ++candidate) {
                        if (scheduler_states[candidate] != llm_expert_request_state::queued) continue;
                        const auto & expected = scheduler_handles[candidate];
                        if (selected.handle.slot == expected.slot &&
                            selected.handle.generation == expected.generation &&
                            selected.handle.target_device == expected.target_device) {
                            index = candidate;
                            break;
                        }
                    }
                }
                if (index == miss_count) {
                    return fail_multi(llm_expert_provider_result::failure(
                        llm_expert_provider_error::metadata_mismatch));
                }
                auto & flight = async_flights[index];
                flight.scheduler_taken = true;
                scheduler_states[index] = llm_expert_request_state::submitting;
                if (flight.cold_hit) {
                    if (config.scheduler->transition(flight.handle,
                            llm_expert_request_state::submitting,
                            llm_expert_request_state::host_ready) !=
                        llm_expert_schedule_disposition::admitted) {
                        return fail_multi(llm_expert_provider_result::failure(
                            llm_expert_provider_error::metadata_mismatch));
                    }
                    scheduler_states[index] = llm_expert_request_state::host_ready;
                    continue;
                }
            }
            const auto submit_admissible_direct_reads = [&]() {
                size_t admitted = 0;
                std::array<uint32_t, LLM_EXPERT_MAX_DEVICES> pending_cold_by_device = {};
                std::array<uint32_t, LLM_EXPERT_MAX_DEVICES> active_direct_by_device = {};
                std::array<uint32_t, LLM_EXPERT_MAX_DEVICES> admitted_direct_by_device = {};
                for (size_t index = 0; index < miss_count; ++index) {
                    const auto & flight = async_flights[index];
                    if (flight.flight_id.target_device >= device_pools.size()) {
                        return llm_expert_provider_result::failure(
                            llm_expert_provider_error::metadata_mismatch);
                    }
                    if (flight.cold_hit && !flight.processed) {
                        pending_cold_by_device[flight.flight_id.target_device]++;
                    }
                    if (flight.direct_storage && flight.submitted && !flight.processed) {
                        active_direct_by_device[flight.flight_id.target_device]++;
                    }
                }
                for (size_t index = 0; index < miss_count; ++index) {
                    auto & flight = async_flights[index];
                    if (!flight.direct_storage || flight.submitted || flight.read_completed ||
                            !flight.scheduler_taken) continue;
                    const auto owner = flight.flight_id.target_device;
                    // Every ring is initialized with at least two lanes. Keep
                    // one available for ready cold data instead of allowing
                    // direct reads to recreate a per-device phase barrier.
                    const uint32_t direct_lane_limit = device_lane_capacities[owner] -
                        uint32_t(pending_cold_by_device[owner] != 0);
                    if (active_direct_by_device[owner] + admitted_direct_by_device[owner] >=
                            direct_lane_limit) continue;
                    const uint32_t unique_index = miss_unique_indices[index];
                    const auto & key = unique_keys[unique_index];
                    const uint32_t global_slot = candidate_slots[index];
                    auto & entry = directory_slots[global_slot];
                    auto * ring = ring_for_slot(global_slot);
                    if (ring == nullptr || entry.device_id != flight.flight_id.target_device) {
                        return llm_expert_provider_result::failure(
                            llm_expert_provider_error::metadata_mismatch);
                    }
                    if (!flight.hot_prepared) {
                        auto prepared = prepare_hot_slot_locked(
                            global_slot, key, {}, true, false);
                        if (!prepared.is_ready()) return prepared;
                        flight.hot_prepared = true;
                    }
                    auto reserved = ring->reserve_direct_storage(
                        binding.layout_class_id, entry.device_slot, entry.generation,
                        flight.lane, flight.flight_id, false);
                    if (!reserved.is_ready()) {
                        if (reserved.error == llm_expert_provider_error::busy) continue;
                        return reserved;
                    }
                    auto destinations_ready = ring->storage_destinations(
                        flight.lane, flight.destinations.data(), flight.destinations.size(),
                        flight.destination_count);
                    if (!destinations_ready.is_ready()) return destinations_ready;
                    const auto planned = config.storage->make_read_plan(
                        flight.key, flight.destinations.data(), flight.destination_count,
                        flight.operations.data(), flight.operations.size(), flight.operation_count);
                    if (!planned.is_ready()) return llm_expert_provider_result::failure(
                        llm_expert_provider_error::metadata_mismatch);
                    const llm_expert_async_operation_identity identity = {
                        flight.flight_id.transport_epoch,
                        flight.handle, 0, flight.key, llm_expert_readiness::device_ready,
                        llm_expert_priority::demand_current_layer, binding.layout_class_id,
                    };
                    if (config.async_transport->submit_read_plan(
                            identity, flight.operations.data(), flight.operation_count, true) !=
                            llm_expert_async_result::ready) {
                        return llm_expert_provider_result::failure(
                            llm_expert_provider_error::copy_failed);
                    }
                    flight.submitted = true;
                    flight.read_active = true;
                    submitted_count++;
                    admitted++;
                    admitted_direct_by_device[owner]++;
                    if (config.scheduler->transition(flight.handle,
                            llm_expert_request_state::submitting,
                            llm_expert_request_state::io_in_flight) !=
                        llm_expert_schedule_disposition::admitted) {
                        return llm_expert_provider_result::failure(
                            llm_expert_provider_error::copy_failed);
                    }
                    scheduler_states[index] = llm_expert_request_state::io_in_flight;
                }
                if (admitted != 0) config.async_transport->start_deferred_reads();
                return llm_expert_provider_result::success();
            };
            const auto flush_and_wait_exact_h2d = [&](size_t index) {
                auto completed = flush_device_transfers();
                if (!completed.is_ready()) return completed;
                const uint32_t slot = candidate_slots[index];
                const auto & entry = directory_slots[slot];
                const auto owner = entry.device_id;
                if (async_decode_transfers) {
                    completed = transfer_rings[owner]->wait_for_hot(
                        device_backends[owner], entry.device_slot, entry.generation, false);
                    if (!completed.is_ready()) return completed;
                    binding.devices[owner].h2d_dependencies.clear();
                    provider_h2d_async_branch_waits++;
                }
                async_flights[index].device_ready = true;
                return completed;
            };

            size_t completed_count = 0;
            while (completed_count < miss_count) {
                result = submit_admissible_direct_reads();
                if (!result.is_ready()) return fail_multi(result);
                size_t index = miss_count;
                for (size_t candidate = 0; candidate < miss_count; ++candidate) {
                    if (async_flights[candidate].cold_hit &&
                            !async_flights[candidate].processed) {
                        index = candidate;
                        break;
                    }
                }
                if (index != miss_count) {
                    result = stage_miss(index);
                    if (result.is_ready()) result = flush_and_wait_exact_h2d(index);
                    if (!result.is_ready()) return fail_multi(result);
                    async_flights[index].h2d_submitted = true;
                    async_flights[index].processed = true;
                    completed_count++;
                    continue;
                }
                size_t pending_count = 0;
                for (size_t index = 0; index < miss_count; ++index) {
                    if (async_flights[index].read_active) {
                        async_handles[pending_count++] = async_flights[index].handle;
                    }
                }
                if (pending_count == 0) {
                    return fail_multi(llm_expert_provider_result::failure(
                        llm_expert_provider_error::metadata_mismatch));
                }
                llm_expert_async_read_completion completion;
                llm_expert_request_handle completed_handle;
                if (provider_lock != nullptr) provider_lock->unlock();
                const auto waited = config.async_transport->wait_any_read(
                    async_handles.data(), pending_count, completed_handle, completion,
                    abort_callback, abort_callback_data);
                if (provider_lock != nullptr) provider_lock->lock();
                index = miss_count;
                for (size_t candidate = 0; candidate < miss_count; ++candidate) {
                    const auto & flight = async_flights[candidate];
                    if (flight.read_active && flight.handle.slot == completed_handle.slot &&
                        flight.handle.generation == completed_handle.generation &&
                        flight.handle.target_device == completed_handle.target_device) {
                        index = candidate;
                        break;
                    }
                }
                if (index == miss_count) {
                    return fail_multi(llm_expert_provider_result::failure(
                        waited == llm_expert_async_result::closed ?
                            llm_expert_provider_error::cancelled :
                            llm_expert_provider_error::stale_generation));
                }
                auto & flight = async_flights[index];
                const auto released = config.async_transport->release_read(flight.handle);
                if (released == llm_expert_async_result::ready) flight.read_active = false;
                const auto storage_error = waited == llm_expert_async_result::ready ?
                    llm_expert_storage_error::none :
                    (waited == llm_expert_async_result::closed ? llm_expert_storage_error::cancelled :
                     (completion.native_error == 0 ? llm_expert_storage_error::short_read :
                      llm_expert_storage_error::io_error));
                config.storage->record_async_read(
                    flight.destination_count, completion.bytes_completed,
                    storage_error, completion.native_error);
                const bool integrity_matches = waited == llm_expert_async_result::ready &&
                    released == llm_expert_async_result::ready &&
                    finalize_payload_integrity(config.integrity_mode, *config.storage,
                        flight.destinations.data(), flight.destination_count,
                        completion.integrity_status, completion.digest);
                if (!integrity_matches) {
                    return fail_multi(llm_expert_provider_result::failure(
                        waited == llm_expert_async_result::closed ?
                            llm_expert_provider_error::cancelled : llm_expert_provider_error::copy_failed));
                }
                result = transfer_rings[flight.flight_id.target_device]->complete_direct_storage(
                    flight.lane, completion.bytes_completed);
                if (!result.is_ready()) return fail_multi(result);
                flight.direct_storage_completed = true;
                flight.read_completed = true;
                if (config.scheduler->transition(flight.handle,
                        llm_expert_request_state::io_in_flight,
                        llm_expert_request_state::host_ready) !=
                    llm_expert_schedule_disposition::admitted) {
                    return fail_multi(llm_expert_provider_result::failure(
                        llm_expert_provider_error::metadata_mismatch));
                }
                scheduler_states[index] = llm_expert_request_state::host_ready;
                result = stage_miss(index);
                if (result.is_ready()) result = flush_and_wait_exact_h2d(index);
                if (!result.is_ready()) return fail_multi(result);
                flight.h2d_submitted = true;
                flight.processed = true;
                completed_count++;
            }
            LLM_EXPERT_TRACE_INSTANT("k3.storage", "multi_device_issue_ahead",
                "layer", binding.layer, "demand_count", miss_count,
                "submitted_count", submitted_count);
        } else {
            for (size_t index = 0; index < miss_count; ++index) {
                if (abort_requested()) {
                    return fail_multi(llm_expert_provider_result::failure(
                        llm_expert_provider_error::cancelled));
                }
                const uint32_t unique_index = miss_unique_indices[index];
                const auto & key = unique_keys[unique_index];
                const auto owner = llm_expert_owner_device(key.expert, uint32_t(device_pools.size()));
                llm_expert_request_metadata metadata;
                metadata.layout_class_id = binding.layout_class_id;
                metadata.target_device = owner;
                metadata.reserved_storage_bytes = payload_for_key(key);
                metadata.reserved_h2d_bytes = payload_for_key(key);
                const auto scheduled = config.scheduler->enqueue(
                    key, llm_expert_priority::demand_current_layer,
                    llm_expert_readiness::device_ready, metadata);
                if (!scheduled.accepted()) {
                    return fail_multi(llm_expert_provider_result::failure(
                        scheduled.disposition == llm_expert_schedule_disposition::generation_exhausted ?
                            llm_expert_provider_error::generation_exhausted :
                            llm_expert_provider_error::busy));
                }
                scheduler_handles[index] = scheduled.handle;
                scheduler_states[index] = llm_expert_request_state::queued;
                llm_expert_request_snapshot snapshot;
                const auto selected = config.scheduler->take_next(snapshot);
                if (!selected.accepted() || selected.handle.slot != scheduled.handle.slot ||
                    selected.handle.generation != scheduled.handle.generation ||
                    selected.handle.target_device != scheduled.handle.target_device) {
                    return fail_multi(llm_expert_provider_result::failure(
                        llm_expert_provider_error::metadata_mismatch));
                }
                scheduler_states[index] = llm_expert_request_state::submitting;
                storage_load_context storage_context = {
                    config.storage, nullptr, nullptr, config.integrity_mode, provider_lock, {}, false,
                    abort_callback, abort_callback_data,
                };
                storage_context.layer_ids = &layout_registry.layer_ids;
                storage_context.layout_registry = &layout_registry;
                storage_context.target_device = owner;
                result = cold_cache->find_or_admit_with_loader(
                    key, cold_references[index], load_storage_bundle, &storage_context);
                if (!result.is_ready()) return fail_multi(result);
                if (config.scheduler->transition(scheduled.handle,
                        llm_expert_request_state::submitting,
                        llm_expert_request_state::host_ready) !=
                    llm_expert_schedule_disposition::admitted) {
                    return fail_multi(llm_expert_provider_result::failure(
                        llm_expert_provider_error::metadata_mismatch));
                }
                scheduler_states[index] = llm_expert_request_state::host_ready;
                result = stage_miss(index);
                if (!result.is_ready()) return fail_multi(result);
            }
        }

        result = flush_device_transfers();
        if (!result.is_ready()) return fail_multi(result);
        if (abort_requested()) {
            return fail_multi(llm_expert_provider_result::failure(
                llm_expert_provider_error::cancelled));
        }
        if (async_decode_transfers) {
            for (size_t device = 0; device < binding.devices.size(); ++device) {
                auto & dependencies = binding.devices[device].h2d_dependencies;
                for (const auto & dependency : dependencies) {
                    result = transfer_rings[device]->wait_for_hot(
                        device_backends[device], dependency.hot_slot,
                        dependency.hot_generation, false);
                    if (!result.is_ready()) return fail_multi(result);
                    provider_h2d_async_branch_waits++;
                }
                if (!dependencies.empty()) {
                    LLM_EXPERT_TRACE_INSTANT("k3.transfer", "device_h2d_branch_waits",
                        "layer", binding.layer, "device_id", device,
                        "event_count", dependencies.size());
                }
            }
        }

        // Storage and cold-ready flights may prepare hot-policy loads in a
        // different order than they complete. Mark every terminal before
        // publishing or pinning so deterministic terminal flushing can retire
        // the complete operation-ordinal prefix without a transport barrier.
        for (size_t index = 0; index < miss_count; ++index) {
            const uint32_t slot = candidate_slots[index];
            auto & entry = directory_slots[slot];
            result = cache_policy_result(hot_policy.load_complete(slot, entry.generation));
            if (!result.is_ready()) return fail_multi(result);
        }
        for (size_t index = 0; index < miss_count; ++index) {
            const uint32_t unique_index = miss_unique_indices[index];
            const uint32_t slot = candidate_slots[index];
            auto & entry = directory_slots[slot];
            entry.state = hot_slot_state::ready;
            entry.last_use = ++use_clock;
            directory_forward[forward_index(entry.key)] = { int32_t(slot), entry.generation };
            admissions++;
            device_runtime_stats[entry.device_id].admissions++;
            result = pin_slot_locked(slot);
            if (!result.is_ready()) return fail_multi(result);
            if (config.scheduler->transition(scheduler_handles[index],
                    llm_expert_request_state::h2d_in_flight,
                    llm_expert_request_state::device_ready) !=
                    llm_expert_schedule_disposition::admitted ||
                config.scheduler->finish(scheduler_handles[index],
                    llm_expert_request_state::complete) !=
                    llm_expert_schedule_disposition::admitted ||
                config.scheduler->release_terminal(scheduler_handles[index]) !=
                    llm_expert_schedule_disposition::admitted) {
                return fail_multi(llm_expert_provider_result::failure(
                    llm_expert_provider_error::metadata_mismatch));
            }
            scheduler_states[index] = llm_expert_request_state::free;
            scheduler_handles[index] = {};
            const uint64_t transferred_bytes = payload_for_key(unique_keys[unique_index]);
            h2d_bytes += transferred_bytes;
            device_runtime_stats[entry.device_id].h2d_bytes += transferred_bytes;
        }
        for (auto & device : binding.devices) device.h2d_dependencies.clear();
        for (size_t index = 0; index < unique_count; ++index) {
            if (unique_slots[index] < 0) continue;
            bool already_pinned = false;
            for (size_t request_pin = 0; request_pin < request_pin_count; ++request_pin) {
                already_pinned = already_pinned || request_pins[request_pin].slot == uint32_t(unique_slots[index]);
            }
            if (!already_pinned) {
                result = pin_slot_locked(uint32_t(unique_slots[index]));
                if (!result.is_ready()) return fail_multi(result);
            }
        }

        for (auto & ids : device_execution_id_scratch) {
            std::fill(ids.begin(), ids.begin() + logical_id_count, -1);
        }
        for (size_t index = 0; index < logical_id_count; ++index) {
            const size_t unique_index = size_t(element_unique[index]);
            const uint32_t slot = uint32_t(unique_slots[unique_index]);
            const auto & entry = directory_slots[slot];
            device_execution_id_scratch[entry.device_id][index] = int32_t(entry.device_slot);
            last_logical_ids[index] = logical_ids[index];
            last_execution_ids[index] = int32_t(entry.device_slot);
        }
        last_id_count = logical_id_count;
        last_remap_layer = binding.layer;
        remap_checkpoints++;
        logical_id_total += logical_id_count;
        unique_id_total += unique_count;
        hits += hit_count;
        misses += miss_count;
        gpu_execution_lanes += logical_id_count;
        return llm_expert_provider_result::success();
    }

    llm_expert_provider_result remap_checkpoint_locked(
            const llm_expert_graph_binding & binding,
            const int32_t * logical_ids,
            size_t logical_id_count,
            int32_t * execution_ids,
            int32_t * cpu_execution_ids,
            ggml_backend_t execution_backend,
            bool (*abort_callback)(void *),
            void * abort_callback_data,
            std::unique_lock<std::mutex> * provider_lock) noexcept {
        (void) llm_perfetto_trace_maybe_start_decode_window(active_request_id, binding.layer);
        LLM_EXPERT_TRACE_SCOPE("k3.provider", "acquire_and_remap", "request_id", active_request_id,
            "layer", binding.layer, "layout_class_id", binding.layout_class_id,
            "selected_key_count", logical_id_count);
        LLM_EXPERT_TRACE_SCOPE("k3.graph", "expert_layer_execution", "request_id", active_request_id,
            "layer", binding.layer, "selected_key_count", logical_id_count);
        LLM_EXPERT_TRACE_CUDA_SCOPE(llm_perfetto_trace_id(llm_perfetto_trace_domain::request, active_request_id));
        if (!active_request || !pool || logical_ids == nullptr || execution_ids == nullptr ||
            binding.provider_identity != this || binding.bootstrap || binding.graph_epoch != epoch ||
            binding.generation_lease.get() != pool.get() || !binding_uses_current_pool(binding) ||
            binding.execution_ids == nullptr || binding.execution_ids->ne[0] != int64_t(config.n_expert_used) ||
            binding.execution_ids->ne[1] < 0) {
            return fail(llm_expert_provider_result::failure(llm_expert_provider_error::invalid_binding));
        }
        if (binding.hybrid && cpu_execution_ids == nullptr) {
            return fail(llm_expert_provider_result::failure(llm_expert_provider_error::invalid_binding));
        }

        const auto execution_device = execution_backend == nullptr ? nullptr :
            ggml_backend_get_device(execution_backend);
        last_execution_backend_device_type = execution_device == nullptr ? -1 :
            int32_t(ggml_backend_dev_type(execution_device));

        const uint64_t expected_count = uint64_t(binding.execution_ids->ne[0])*uint64_t(binding.execution_ids->ne[1]);
        if (expected_count != logical_id_count || logical_id_count > element_unique.size() ||
            !extent_is_safe(uint64_t(binding.execution_ids->ne[1]))) {
            return fail(llm_expert_provider_result::failure(llm_expert_provider_error::unsupported_configuration));
        }

        auto registration = registrations.find(binding.layer);
        if (registration == registrations.end()) {
            return fail(llm_expert_provider_result::failure(llm_expert_provider_error::invalid_binding));
        }

        if (config.cpu_cold_only) {
            return remap_cpu_cold_only_locked(
                binding, logical_ids, logical_id_count, execution_ids,
                execution_backend, abort_callback, abort_callback_data, provider_lock);
        }

        if (!validate_request_pins_locked()) {
            return fail(llm_expert_provider_result::failure(llm_expert_provider_error::stale_generation));
        }
        if (config.cold_mode && config.background_promotion) {
            const auto normalized = deterministic_policy_terminals ?
                normalize_background_terminals_locked(provider_lock) :
                reap_background_locked(provider_lock);
            if (!normalized.is_ready()) return fail(normalized);
            if (config.phase8_test_control != nullptr) {
                for (const auto & record : background_promotions) {
                    if (!record.active) continue;
                    config.phase8_test_control->observe(
                        llm_expert_phase8_test_gate::auto_after_normalization_before_background_snapshot,
                        record.key, record.hot_generation);
                    if (provider_lock != nullptr) provider_lock->unlock();
                    const bool paused = config.phase8_test_control->pause_if_armed(
                        llm_expert_phase8_test_gate::auto_after_normalization_before_background_snapshot,
                        record.key, record.hot_generation);
                    if (provider_lock != nullptr) provider_lock->lock();
                    if (paused) {
                        const auto * revalidated = find_background_locked(record.key);
                        if (revalidated == nullptr || revalidated->hot_generation != record.hot_generation) {
                            return fail(llm_expert_provider_result::failure(
                                llm_expert_provider_error::stale_generation));
                        }
                        break;
                    }
                }
            }
        }

        phase10_before_remap_locked(
            binding, logical_ids, logical_id_count);

        size_t unique_count = 0;
        for (size_t index = 0; index < logical_id_count; ++index) {
            const int32_t expert = logical_ids[index];
            if (expert < 0 || expert >= int32_t(n_expert)) {
                return fail(llm_expert_provider_result::failure(llm_expert_provider_error::invalid_key));
            }
            const llm_expert_key key = { binding.layer, expert };
            size_t unique_index = 0;
            while (unique_index < unique_count && !expert_key_matches(unique_keys[unique_index], key)) {
                unique_index++;
            }
            if (unique_index == unique_count) {
                if (unique_count >= config.capacity) {
                    return fail(llm_expert_provider_result::failure(llm_expert_provider_error::unsupported_configuration));
                }
                unique_keys[unique_count] = key;
                unique_slots[unique_count] = -1;
                unique_lane_counts[unique_count] = 0;
                unique_gpu_assignment[unique_count] = 0;
                unique_index = unique_count++;
            }
            element_unique[index] = int32_t(unique_index);
            unique_lane_counts[unique_index]++;
        }

        const auto requested_phase = binding.execution_ids->ne[1] > 1 ?
            llm_expert_cache_policy_phase::prefill : llm_expert_cache_policy_phase::decode;
        if (request_ubatch_ordinal == UINT64_MAX) {
            return fail(llm_expert_provider_result::failure(
                llm_expert_provider_error::unsupported_configuration));
        }
        const uint64_t ubatch_ordinal = request_ubatch_ordinal + 1;
        if (phase10_lead_events && requested_phase == llm_expert_cache_policy_phase::decode) {
            if (phase10_lead_event_count < phase10_lead_events->size()) {
                const uint64_t steady_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()).count();
                (*phase10_lead_events)[phase10_lead_event_count] = {
                    phase10_lead_event_count, ubatch_ordinal, binding.layer, steady_ns,
                };
                phase10_lead_event_count++;
            } else {
                phase10_lead_events_dropped++;
            }
        }
        auto ubatch_set = cache_policy_result(hot_policy.set_ubatch_ordinal(ubatch_ordinal));
        if (ubatch_set.is_ready() && cold_cache) {
            ubatch_set = cold_cache->policy_set_ubatch_ordinal(ubatch_ordinal);
        }
        if (!ubatch_set.is_ready()) return fail(ubatch_set);
        request_ubatch_ordinal = ubatch_ordinal;
        if (requested_phase != hot_policy_phase) {
            auto transitioned = cache_policy_result(hot_policy.phase_transition(requested_phase));
            if (transitioned.is_ready() && cold_cache) {
                transitioned = cold_cache->policy_phase_transition(requested_phase);
            }
            if (!transitioned.is_ready()) return fail(transitioned);
            hot_policy_phase = requested_phase;
        }
        for (size_t index = 0; index < unique_count; ++index) policy_unique_indices[index] = uint32_t(index);
        std::sort(policy_unique_indices.begin(), policy_unique_indices.begin() + unique_count,
            [&](uint32_t lhs, uint32_t rhs) {
                const auto & lhs_key = unique_keys[lhs];
                const auto & rhs_key = unique_keys[rhs];
                return lhs_key.layer != rhs_key.layer ? lhs_key.layer < rhs_key.layer :
                    lhs_key.expert < rhs_key.expert;
            });
        for (size_t index = 0; index < unique_count; ++index) {
            const uint32_t unique_index = policy_unique_indices[index];
            const auto & key = unique_keys[unique_index];
            const auto observed = cache_policy_result(hot_policy.demand(
                policy_key_for(key), unique_lane_counts[unique_index],
                payload_for_key(key), hot_physical_slot_footprint_bytes));
            if (!observed.is_ready()) return fail(observed);
        }

        std::fill(slot_selected.begin(), slot_selected.end(), uint8_t(0));
        size_t miss_count = 0;
        for (size_t policy_index = 0; policy_index < unique_count; ++policy_index) {
            const size_t index = policy_unique_indices[policy_index];
            const auto & key = unique_keys[index];
            LLM_EXPERT_TRACE_INSTANT("k3.cache.hot", "lookup", "request_id", active_request_id,
                "layer", key.layer, "original_expert_id", key.expert);
            const auto & forward = directory_forward[forward_index(key)];
            if (forward.slot < 0) {
                LLM_EXPERT_TRACE_INSTANT("k3.cache.hot", "miss", "request_id", active_request_id,
                    "layer", key.layer, "original_expert_id", key.expert);
                miss_unique_indices[miss_count++] = uint32_t(index);
                continue;
            }
            if (!forward_entry_matches(key, forward)) {
                metadata_mismatches++;
                return fail(llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch));
            }
            unique_slots[index] = forward.slot;
            slot_selected[forward.slot] = 1;
        }
        for (size_t policy_index = 0; policy_index < unique_count; ++policy_index) {
            const size_t index = policy_unique_indices[policy_index];
            if (unique_slots[index] < 0) continue;
            auto & entry = directory_slots[unique_slots[index]];
            if (entry.origin == llm_expert_residency_origin::speculative &&
                !entry.background_useful) {
                entry.origin = llm_expert_residency_origin::demand;
                entry.background_useful = true;
                entry.speculative_deadline = 0;
                entry.speculative_utility = 0;
                background_useful++;
            }
            const auto touched = cache_policy_result(hot_policy.hit(
                uint32_t(unique_slots[index]), entry.generation));
            if (!touched.is_ready()) return fail(touched);
            LLM_EXPERT_TRACE_INSTANT("k3.cache.hot", "hit", "request_id", active_request_id,
                "layer", unique_keys[index].layer, "original_expert_id", unique_keys[index].expert,
                "slot_id", uint32_t(unique_slots[index]), "generation", entry.generation);
        }
        const auto prior_pins_released = release_request_pins_locked();
        if (!prior_pins_released.is_ready()) return fail(prior_pins_released);

        if (binding.hybrid && config.miss_policy == LLAMA_EXPERT_MISS_POLICY_AUTO) {
            std::sort(miss_unique_indices.begin(), miss_unique_indices.begin() + miss_count,
                [&](uint32_t lhs, uint32_t rhs) {
                    const auto & lhs_key = unique_keys[lhs];
                    const auto & rhs_key = unique_keys[rhs];
                    return lhs_key.layer != rhs_key.layer ? lhs_key.layer < rhs_key.layer :
                        lhs_key.expert < rhs_key.expert;
                });
        }

        if (binding.hybrid && config.miss_policy != LLAMA_EXPERT_MISS_POLICY_PROMOTE_AND_GPU) {
            const auto abort_requested = [&]() {
                if (abort_callback == nullptr) return false;
                if (provider_lock != nullptr) provider_lock->unlock();
                const bool requested = abort_callback(abort_callback_data);
                if (provider_lock != nullptr) provider_lock->lock();
                return requested;
            };
            if (abort_requested()) {
                return fail(llm_expert_provider_result::failure(llm_expert_provider_error::cancelled));
            }
            size_t hit_count = 0;
            for (size_t index = 0; index < unique_count; ++index) {
                unique_cpu_slots[index] = -1;
                if (unique_slots[index] < 0) continue;
                auto & entry = directory_slots[unique_slots[index]];
                auto touched = llm_expert_provider_result::success();
                if (entry.has_cold_backing) {
                    const llm_cold_reference backing = {
                        entry.cold_slot, entry.cold_generation, entry.layout_class_id };
                    touched = cold_cache->policy_shadow_hit(
                        unique_keys[index], backing, unique_lane_counts[index]);
                    if (touched.is_ready()) {
                        touched = cold_cache->acquire(backing, llm_cold_reference_kind::request);
                    }
                    if (touched.is_ready()) touched = cold_cache->release(backing, llm_cold_reference_kind::request);
                    if (!touched.is_ready()) return fail(touched);
                }
                touched = transfer_ring->wait_for_hot(execution_backend, uint32_t(unique_slots[index]),
                    entry.generation, deterministic_policy_terminals);
                if (!touched.is_ready()) return fail(touched);
                touched = pin_slot_locked(uint32_t(unique_slots[index]));
                if (!touched.is_ready()) return fail(touched);
                if (entry.origin == llm_expert_residency_origin::speculative &&
                    !entry.background_useful) {
                    entry.background_useful = true;
                    entry.origin = llm_expert_residency_origin::demand;
                    entry.speculative_deadline = 0;
                    entry.speculative_utility = 0;
                    background_useful++;
                }
                hit_count++;
            }

            storage_load_context storage_context = {
                config.storage, nullptr, nullptr, config.integrity_mode, provider_lock, {}, false,
                abort_callback, abort_callback_data,
            };
            storage_context.layer_ids = &layout_registry.layer_ids;
            storage_context.layout_registry = &layout_registry;
            auto result = llm_expert_provider_result::success();
            uint64_t queued_cpu_work = 0;
            uint64_t queued_h2d_work = 0;
            uint64_t queued_gpu_work = 0;
            const bool auto_prefill = binding.execution_ids->ne[1] > 1;
            if (config.miss_policy == LLAMA_EXPERT_MISS_POLICY_AUTO) {
                // Hot hits are already assigned GPU work for this bucket and
                // therefore precede every miss in the production queue model.
                for (size_t index = 0; index < unique_count; ++index) {
                    if (unique_slots[index] < 0) continue;
                    queued_gpu_work = saturating_add_work(queued_gpu_work,
                        auto_gpu_work(config.auto_cost_model, auto_prefill, unique_lane_counts[index]));
                }
                // Snapshot every live background transfer once. This is the
                // immutable AUTO operand set for this request.
                for (auto & record : background_promotions) {
                    if (!record.active) continue;
                    llm_expert_same_key_h2d_state state = llm_expert_same_key_h2d_state::none;
                    uint64_t remaining = 0;
                    const auto polled = transfer_ring->poll_h2d(record.lane, state, remaining);
                    if (!polled.is_ready()) {
                        const auto ordered = finish_background_before_locked(record.origin_operation_ordinal);
                        if (!ordered.is_ready()) {
                            result = ordered;
                            break;
                        }
                        const auto released = transfer_ring->release_terminal_background(record.lane);
                        if (!released.is_ready()) {
                            result = released;
                            break;
                        }
                        record.state = background_promotion_record::lifecycle::failed;
                        discard_background_slot_locked(record, false);
                        background_dropped++;
                        continue;
                    }
                    record.remaining_bytes = remaining;
                    if (state == llm_expert_same_key_h2d_state::queued_or_staging) {
                        record.state = background_promotion_record::lifecycle::queued_or_staging;
                    } else if (state == llm_expert_same_key_h2d_state::h2d_in_flight) {
                        record.state = background_promotion_record::lifecycle::h2d_in_flight;
                    } else {
                        record.state = background_promotion_record::lifecycle::h2d_complete_unpublished;
                    }
                    record.h2d_work_ns = state == llm_expert_same_key_h2d_state::h2d_complete_unpublished ?
                        0 : auto_h2d_work(config.auto_cost_model, remaining);
                }
            }
            size_t cpu_miss_count = 0;
            for (size_t index = 0; index < miss_count && result.is_ready(); ++index) {
                const uint32_t unique_index = miss_unique_indices[index];
                llm_cold_reference reference;
                result = config.storage ? cold_cache->find_or_admit_with_loader(
                    unique_keys[unique_index], reference, load_storage_bundle, &storage_context) :
                    cold_cache->find_or_admit(
                        unique_keys[unique_index], registration->second, reference,
                        faults.fail_copy_after_tensors);
                if (!result.is_ready()) break;
                cold_references[unique_index] = reference;

                bool use_gpu = false;
                if (config.miss_policy == LLAMA_EXPERT_MISS_POLICY_AUTO) {
                    const auto * background = find_background_locked(unique_keys[unique_index]);
                    uint64_t background_h2d_work = 0;
                    for (const auto & queued : background_promotions) {
                        if (!queued.active || &queued == background) continue;
                        background_h2d_work = saturating_add_work(background_h2d_work, queued.h2d_work_ns);
                    }
                    const llm_expert_auto_input input = {
                        config.auto_cost_model,
                        auto_prefill,
                        unique_lane_counts[unique_index],
                        payload_for_key(unique_keys[unique_index]),
                        queued_cpu_work,
                        saturating_add_work(queued_h2d_work, background_h2d_work),
                        queued_gpu_work,
                        background != nullptr,
                        background == nullptr ? llm_expert_same_key_h2d_state::none :
                            background->state == background_promotion_record::lifecycle::queued_or_staging ?
                                llm_expert_same_key_h2d_state::queued_or_staging :
                            background->state == background_promotion_record::lifecycle::h2d_in_flight ?
                                llm_expert_same_key_h2d_state::h2d_in_flight :
                                llm_expert_same_key_h2d_state::h2d_complete_unpublished,
                        background == nullptr ? 0 : background->remaining_bytes,
                    };
                    const auto decision = llm_evaluate_expert_auto(input);
                    record_auto_decision_locked({ active_request_id, binding.layer,
                        unique_keys[unique_index].expert, input, decision });
                    use_gpu = decision.backend == llm_expert_execution_backend::gpu;
                    if (use_gpu && background != nullptr && config.phase8_test_control != nullptr) {
                        const auto gated_key = background->key;
                        const uint64_t gated_generation = background->hot_generation;
                        config.phase8_test_control->observe(
                            llm_expert_phase8_test_gate::auto_after_decision_before_same_key_join,
                            gated_key, gated_generation);
                        if (provider_lock != nullptr) provider_lock->unlock();
                        const bool paused = config.phase8_test_control->pause_if_armed(
                            llm_expert_phase8_test_gate::auto_after_decision_before_same_key_join,
                            gated_key, gated_generation);
                        if (provider_lock != nullptr) provider_lock->lock();
                        if (paused) {
                            background = find_background_locked(gated_key);
                            if (background == nullptr || background->hot_generation != gated_generation) {
                                result = llm_expert_provider_result::failure(
                                    llm_expert_provider_error::stale_generation);
                            }
                        }
                    }
                    if (use_gpu) {
                        auto_gpu_decisions++;
                        if (background == nullptr) {
                            queued_h2d_work = saturating_add_work(queued_h2d_work, decision.h2d_work_ns);
                        }
                        queued_gpu_work = saturating_add_work(queued_gpu_work, decision.gpu_work_ns);
                    } else {
                        auto_cpu_decisions++;
                        queued_cpu_work = decision.cpu_finish_ns;
                    }
                    auto_tie_decisions += decision.reason == llm_expert_auto_reason::tie;
                    auto_overflow_decisions += decision.reason == llm_expert_auto_reason::overflow;
                }

                if (use_gpu) {
                    auto * background = find_background_locked(unique_keys[unique_index]);
                    if (background != nullptr) {
                        bool background_h2d_complete = false;
                        const auto joined = config.scheduler->enqueue(
                            unique_keys[unique_index], llm_expert_priority::demand_current_layer,
                            llm_expert_readiness::device_ready,
                            demand_metadata(layout_class_for_layer(unique_keys[unique_index].layer)));
                        const bool injected_join_mismatch = config.phase8_test_control != nullptr &&
                            config.phase8_test_control->consume_fault(
                                llm_expert_phase8_test_fault::scheduler_join_mismatch,
                                background->key, background->hot_generation);
                        if (injected_join_mismatch || joined.disposition != llm_expert_schedule_disposition::joined ||
                            joined.handle.slot != background->scheduler_handle.slot ||
                            joined.handle.generation != background->scheduler_handle.generation) {
                            result = llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
                        }
                        if (result.is_ready()) result = transfer_ring->wait_for_hot(
                            execution_backend, background->hot_slot, background->hot_generation,
                            deterministic_policy_terminals);
                        background_h2d_complete = result.is_ready();
                        if (!result.is_ready() && result.error == llm_expert_provider_error::copy_failed) {
                            const auto ordered = finish_background_before_locked(
                                background->origin_operation_ordinal);
                            if (ordered.is_ready()) {
                                const auto terminalized =
                                    transfer_ring->release_terminal_background(background->lane);
                                if (terminalized.is_ready()) {
                                    background->state = background_promotion_record::lifecycle::failed;
                                    discard_background_slot_locked(*background, false);
                                    background_dropped++;
                                    background = nullptr;
                                } else {
                                    result = terminalized;
                                }
                            } else {
                                result = ordered;
                            }
                        }
                        if (result.is_ready() && background != nullptr) {
                            if (background->hot_slot >= directory_slots.size()) {
                                result = llm_expert_provider_result::failure(
                                    llm_expert_provider_error::metadata_mismatch);
                            }
                        }
                        if (result.is_ready() && background != nullptr) {
                            const auto & entry = directory_slots[background->hot_slot];
                            const bool injected_metadata = config.phase8_test_control != nullptr &&
                                config.phase8_test_control->consume_fault(
                                    llm_expert_phase8_test_fault::metadata_mismatch,
                                    background->key, background->hot_generation);
                            if (injected_metadata || entry.state != hot_slot_state::loading ||
                                entry.generation != background->hot_generation ||
                                !expert_key_matches(entry.key, background->key)) {
                                result = llm_expert_provider_result::failure(
                                    llm_expert_provider_error::metadata_mismatch);
                            }
                        }
                        if (result.is_ready() && background != nullptr && config.phase8_test_control != nullptr) {
                            const auto gated_key = background->key;
                            const uint64_t gated_generation = background->hot_generation;
                            if (provider_lock != nullptr) provider_lock->unlock();
                            const bool paused = config.phase8_test_control->pause_if_armed(
                                llm_expert_phase8_test_gate::background_before_provider_publication,
                                gated_key, gated_generation);
                            if (provider_lock != nullptr) provider_lock->lock();
                            if (paused) {
                                background = find_background_locked(gated_key);
                                if (background == nullptr || background->hot_generation != gated_generation) {
                                    result = llm_expert_provider_result::failure(
                                        llm_expert_provider_error::stale_generation);
                                }
                            }
                        }
                        if (result.is_ready() && background != nullptr && config.phase8_test_control != nullptr &&
                            config.phase8_test_control->consume_fault(
                                llm_expert_phase8_test_fault::publication_failed,
                                background->key, background->hot_generation)) {
                            result = llm_expert_provider_result::failure(
                                llm_expert_provider_error::metadata_mismatch);
                        }
                        if (!result.is_ready() && background != nullptr && background_h2d_complete) {
                            const auto ordered = finish_background_before_locked(
                                background->origin_operation_ordinal);
                            if (ordered.is_ready()) {
                                const auto terminalized =
                                    transfer_ring->release_terminal_background(background->lane);
                                if (terminalized.is_ready()) {
                                    background->state = background_promotion_record::lifecycle::failed;
                                    discard_background_slot_locked(*background, false);
                                    background_dropped++;
                                } else {
                                    result = terminalized;
                                }
                            } else {
                                result = ordered;
                            }
                        }
                        if (result.is_ready() && background != nullptr) {
                            result = finish_background_before_locked(background->origin_operation_ordinal);
                        }
                        if (result.is_ready() && background != nullptr) {
                            auto & entry = directory_slots[background->hot_slot];
                            result = cache_policy_result(hot_policy.load_complete(
                                background->hot_slot, entry.generation));
                        }
                        if (result.is_ready() && background != nullptr) {
                            auto & entry = directory_slots[background->hot_slot];
                            entry.state = hot_slot_state::ready;
                            entry.origin = llm_expert_residency_origin::demand;
                            entry.background_useful = true;
                            entry.speculative_deadline = 0;
                            entry.speculative_utility = 0;
                            entry.last_use = ++use_clock;
                            directory_forward[forward_index(entry.key)] = {
                                int32_t(background->hot_slot), entry.generation };
                            background->state = background_promotion_record::lifecycle::published;
                            unique_slots[unique_index] = int32_t(background->hot_slot);
                            result = pin_slot_locked(background->hot_slot);
                            finish_background_scheduler_locked(*background, true);
                            background->active = false;
                            if (active_background_flights > 0) active_background_flights--;
                            background_completed++;
                            background_useful++;
                            background_later_joins++;
                            admissions++;
                            LLM_EXPERT_TRACE_INSTANT("k3.cache.hot", "admission",
                                "request_id", active_request_id, "layer", entry.key.layer,
                                "original_expert_id", entry.key.expert,
                                "slot_id", background->hot_slot, "generation", entry.generation);
                            LLM_EXPERT_TRACE_COUNTER("k3.resource", "hot_cache_occupancy", 4,
                                hot_cache_occupancy_locked());
                        }
                    } else {
                        unique_gpu_assignment[unique_index] = 1;
                    }
                } else {
                    result = cold_cache->acquire(reference, llm_cold_reference_kind::cpu_execution);
                    if (result.is_ready()) {
                        GGML_ASSERT(cpu_execution_pin_count < cpu_execution_pins.size());
                        cpu_execution_pins[cpu_execution_pin_count++] = reference;
                        unique_cpu_slots[unique_index] = int32_t(reference.slot);
                        cpu_miss_count++;
                    }
                }
                if (result.is_ready() && abort_requested()) {
                    result = llm_expert_provider_result::failure(llm_expert_provider_error::cancelled);
                }
            }
            if (!result.is_ready()) {
                const auto released = release_request_pins_locked();
                if (!released.is_ready()) result = released;
                last_remap_error = result.error;
                return fail(result);
            }

            size_t gpu_promotion_count = 0;
            transfer_bindings.clear();
            bool needs_new_gpu_promotion = false;
            for (size_t index = 0; index < miss_count; ++index) {
                const uint32_t unique_index = miss_unique_indices[index];
                needs_new_gpu_promotion = needs_new_gpu_promotion ||
                    (unique_gpu_assignment[unique_index] && unique_slots[unique_index] < 0);
            }
            if (result.is_ready() && needs_new_gpu_promotion) {
                result = finish_background_request_locked();
            }
            for (size_t index = 0; index < miss_count && result.is_ready(); ++index) {
                const uint32_t unique_index = miss_unique_indices[index];
                if (!unique_gpu_assignment[unique_index] || unique_slots[unique_index] >= 0) continue;
                int32_t selected = -1;
                result = select_hot_slot_locked(unique_keys[unique_index],
                    candidate_slots.data(), gpu_promotion_count, selected);
                if (!result.is_ready()) break;
                const uint32_t hot_slot = uint32_t(selected);
                candidate_slots[gpu_promotion_count] = hot_slot;
                result = prepare_hot_slot_locked(
                    hot_slot, unique_keys[unique_index], cold_references[unique_index]);
                if (result.is_ready()) result = transfer_ring->reserve(
                    *cold_cache, cold_references[unique_index], hot_slot,
                    directory_slots[hot_slot].generation, transfer_lanes[unique_index]);
                if (result.is_ready()) result = transfer_ring->stage(
                    transfer_lanes[unique_index],
                    cold_cache->bundle(cold_references[unique_index].layout_class_id));
                if (result.is_ready()) {
                    gpu_unique_indices[gpu_promotion_count++] = unique_index;
                    unique_slots[unique_index] = int32_t(hot_slot);
                    transfer_bindings.push_back({ transfer_lanes[unique_index],
                        pool->bundles[cold_references[unique_index].layout_class_id], hot_slot });
                }
            }
            bool injected_single_device_failure = false;
            if (result.is_ready() && !transfer_bindings.empty()) {
                const uint64_t started_us = uint64_t(ggml_time_us());
                result = transfer_ring->transfer_wave(execution_backend, transfer_bindings);
                const bool device_failure_active = config.remote_single &&
                    device_transfer_failure_for_testing.front() &&
                    (!device_transfer_failure_decode_only_for_testing.front() ||
                        hot_policy_phase == llm_expert_cache_policy_phase::decode);
                if (result.is_ready() && device_failure_active) {
                    injected_single_device_failure = true;
                    injected_device_failure_waves++;
                    injected_device_failure_participants++;
                    result = llm_expert_provider_result::failure(
                        llm_expert_provider_error::copy_failed);
                }
                for (size_t index = 0; index < gpu_promotion_count && result.is_ready(); ++index) {
                    const uint32_t unique_index = gpu_unique_indices[index];
                    const uint32_t hot_slot = uint32_t(unique_slots[unique_index]);
                    result = transfer_ring->wait_for_hot(
                        execution_backend, hot_slot, directory_slots[hot_slot].generation,
                        deterministic_policy_terminals);
                }
                if (result.is_ready()) {
                    for (size_t index = 0; index < gpu_promotion_count; ++index) {
                        const uint32_t unique_index = gpu_unique_indices[index];
                        const uint32_t hot_slot = uint32_t(unique_slots[unique_index]);
                        auto & entry = directory_slots[hot_slot];
                        result = cache_policy_result(hot_policy.load_complete(hot_slot, entry.generation));
                        if (!result.is_ready()) break;
                        entry.state = hot_slot_state::ready;
                        entry.last_use = ++use_clock;
                        directory_forward[forward_index(entry.key)] = { int32_t(hot_slot), entry.generation };
                        admissions++;
                        LLM_EXPERT_TRACE_INSTANT("k3.cache.hot", "admission",
                            "request_id", active_request_id, "layer", entry.key.layer,
                            "original_expert_id", entry.key.expert,
                            "slot_id", hot_slot, "generation", entry.generation);
                        LLM_EXPERT_TRACE_COUNTER("k3.resource", "hot_cache_occupancy", 4,
                            hot_cache_occupancy_locked());
                        result = pin_slot_locked(hot_slot);
                        if (!result.is_ready()) break;
                    }
                    uint64_t bytes = 0;
                    for (size_t index = 0; index < gpu_promotion_count; ++index) {
                        const uint64_t payload = payload_for_key(unique_keys[gpu_unique_indices[index]]);
                        bytes = payload > UINT64_MAX - bytes ? UINT64_MAX : bytes + payload;
                    }
                    h2d_bytes = bytes > UINT64_MAX - h2d_bytes ? UINT64_MAX : h2d_bytes + bytes;
                    h2d_time_us += uint64_t(ggml_time_us()) - started_us;
                }
            }
            if (!result.is_ready()) {
                (void) transfer_ring->cleanup_failed_lanes();
                if (injected_single_device_failure) injected_device_failure_drained_waves++;
                for (size_t index = 0; index < gpu_promotion_count; ++index) {
                    const uint32_t unique_index = gpu_unique_indices[index];
                    auto & entry = directory_slots[uint32_t(unique_slots[unique_index])];
                    const uint32_t hot_slot = uint32_t(unique_slots[unique_index]);
                    const bool was_resident = entry.state == hot_slot_state::ready ||
                        entry.state == hot_slot_state::pinned;
                    if (was_resident) {
                        LLM_EXPERT_TRACE_INSTANT("k3.cache.hot", "victim", "slot_id", hot_slot,
                            "generation", entry.generation, "layer", entry.key.layer,
                            "original_expert_id", entry.key.expert);
                    }
                    llm_expert_cache_policy_result policy_cleanup;
                    if (hot_policy.validate_resident(hot_slot, entry.generation,
                            policy_key_for(entry.key))) {
                        policy_cleanup = hot_policy.remove_resident(hot_slot, entry.generation);
                    } else if (entry.state == hot_slot_state::loading) {
                        policy_cleanup = hot_policy.load_failed(hot_slot, entry.generation);
                    }
                    if (!policy_cleanup.is_ready()) metadata_mismatches++;
                    if (entry.has_cold_backing) {
                        (void) cold_cache->release(
                            { entry.cold_slot, entry.cold_generation, entry.layout_class_id },
                            llm_cold_reference_kind::hot);
                    }
                    const uint64_t generation = entry.generation;
                    entry = {};
                    entry.generation = generation;
                    if (was_resident) {
                        LLM_EXPERT_TRACE_INSTANT("k3.cache.hot", "eviction", "slot_id", hot_slot,
                            "generation", generation, "layer", unique_keys[unique_index].layer,
                            "original_expert_id", unique_keys[unique_index].expert);
                        LLM_EXPERT_TRACE_COUNTER("k3.resource", "hot_cache_occupancy", 4,
                            hot_cache_occupancy_locked());
                    }
                    unique_slots[unique_index] = -1;
                }
                const auto released = release_request_pins_locked();
                if (!released.is_ready()) result = released;
                last_remap_error = result.error;
                return fail(result);
            }

            for (size_t index = 0; index < miss_count; ++index) {
                const uint32_t unique_index = miss_unique_indices[index];
                if (unique_slots[unique_index] < 0) {
                    enqueue_background_locked(unique_keys[unique_index], cold_references[unique_index]);
                }
            }

            size_t gpu_lanes = 0;
            size_t cpu_lanes = 0;
            for (size_t index = 0; index < logical_id_count; ++index) {
                const size_t unique_index = size_t(element_unique[index]);
                const bool gpu = unique_slots[unique_index] >= 0;
                execution_ids[index] = gpu ? unique_slots[unique_index] : -1;
                cpu_execution_ids[index] = gpu ? -1 : unique_cpu_slots[unique_index];
                last_logical_ids[index] = logical_ids[index];
                last_execution_ids[index] = execution_ids[index];
                last_cpu_execution_ids[index] = cpu_execution_ids[index];
                gpu_lanes += gpu;
                cpu_lanes += !gpu;
            }
            if (logical_id_count != 0) {
                result = llm_validate_hybrid_execution_ids(execution_ids, cpu_execution_ids, logical_id_count,
                    int32_t(config.capacity), int32_t(cold_cache->diagnostics().effective_slots));
            }
            if (!result.is_ready()) {
                const auto released = release_request_pins_locked();
                return fail(released.is_ready() ? result : released);
            }
            last_id_count = logical_id_count;
            last_remap_layer = binding.layer;
            remap_checkpoints++;
            logical_id_total += logical_id_count;
            unique_id_total += unique_count;
            hits += hit_count;
            misses += miss_count;
            gpu_execution_lanes += gpu_lanes;
            cpu_execution_lanes += cpu_lanes;
            cpu_fallback_unique_keys += cpu_miss_count;
            if (gpu_lanes != 0 && cpu_lanes != 0) mixed_execution_layers++;
            for (size_t index = 0; index < miss_count; ++index) {
                const uint32_t unique_index = miss_unique_indices[index];
                if (unique_gpu_assignment[unique_index]) continue;
                const uint64_t payload = payload_for_key(unique_keys[unique_index]);
                h2d_bytes_avoided_for_current_output =
                    payload > UINT64_MAX - h2d_bytes_avoided_for_current_output ? UINT64_MAX :
                    h2d_bytes_avoided_for_current_output + payload;
            }
            phase10_after_remap_locked(binding);
            return llm_expert_provider_result::success();
        }

        size_t candidate_count = 0;
        while (candidate_count < miss_count) {
            const uint32_t unique_index = miss_unique_indices[candidate_count];
            int32_t selected = -1;
            const auto selected_result = select_hot_slot_locked(
                unique_keys[unique_index], candidate_slots.data(), candidate_count, selected);
            if (!selected_result.is_ready()) return fail(selected_result);
            candidate_slots[candidate_count++] = uint32_t(selected);
        }

        for (size_t index = 0; index < candidate_count; ++index) {
            if (directory_slots[candidate_slots[index]].generation == std::numeric_limits<uint64_t>::max()) {
                return fail(llm_expert_provider_result::failure(llm_expert_provider_error::generation_exhausted));
            }
        }

        size_t hit_count = 0;
        for (size_t index = 0; index < unique_count; ++index) {
            if (unique_slots[index] >= 0) {
                auto & entry = directory_slots[unique_slots[index]];
                if (config.cold_mode) {
                    auto touched = llm_expert_provider_result::success();
                    if (entry.has_cold_backing) {
                        const llm_cold_reference backing = {
                            entry.cold_slot, entry.cold_generation, entry.layout_class_id };
                        touched = cold_cache->policy_shadow_hit(
                            unique_keys[index], backing, unique_lane_counts[index]);
                        if (touched.is_ready()) {
                            touched = cold_cache->acquire(backing, llm_cold_reference_kind::request);
                        }
                        if (touched.is_ready()) touched = cold_cache->release(backing, llm_cold_reference_kind::request);
                        if (!touched.is_ready()) return fail(touched);
                    }
                    touched = transfer_ring->wait_for_hot(execution_backend, uint32_t(unique_slots[index]),
                        entry.generation, deterministic_policy_terminals);
                    if (!touched.is_ready()) return fail(touched);
                }
                const auto pinned = pin_slot_locked(uint32_t(unique_slots[index]));
                if (!pinned.is_ready()) return fail(pinned);
                hit_count++;
            }
        }

        if (miss_count != 0 && config.background_promotion) {
            const auto ordered = finish_background_request_locked();
            if (!ordered.is_ready()) return fail(ordered);
        }
        for (size_t index = 0; index < miss_count; ++index) {
            const uint32_t unique_index = miss_unique_indices[index];
            const uint32_t slot = candidate_slots[index];
            auto & entry = directory_slots[slot];
            if (entry.state != hot_slot_state::free) {
                LLM_EXPERT_TRACE_INSTANT("k3.cache.hot", "victim", "slot_id", slot,
                    "generation", entry.generation, "layer", entry.key.layer,
                    "original_expert_id", entry.key.expert);
                const auto policy_valid = cache_policy_result(hot_policy.validate_evictable(slot, entry.generation));
                if (!policy_valid.is_ready()) return fail(policy_valid);
                if (config.cold_mode) {
                    auto retired = transfer_ring->retire_hot(slot, entry.generation);
                    if (!retired.is_ready()) return fail(retired);
                    if (entry.has_cold_backing) {
                        auto released = cold_cache->release(
                            { entry.cold_slot, entry.cold_generation, entry.layout_class_id },
                            llm_cold_reference_kind::hot);
                        if (!released.is_ready()) return fail(released);
                        entry.has_cold_backing = false;
                        entry.cold_slot = 0;
                        entry.cold_generation = 0;
                    }
                }
                const auto removed = cache_policy_result(hot_policy.evict(slot, entry.generation));
                if (!removed.is_ready()) return fail(removed);
                entry.state = hot_slot_state::evicting;
                clear_forward_locked(entry.key, slot, entry.generation);
                LLM_EXPERT_TRACE_INSTANT("k3.cache.hot", "eviction", "slot_id", slot,
                    "generation", entry.generation, "layer", entry.key.layer,
                    "original_expert_id", entry.key.expert);
                evictions++;
                LLM_EXPERT_TRACE_COUNTER("k3.resource", "hot_cache_occupancy", 4,
                    hot_cache_occupancy_locked());
                if (config.cold_mode) no_writeback_evictions++;
            }
            entry.state = hot_slot_state::reserved;
            entry.key = unique_keys[unique_index];
            entry.layout_class_id = layout_class_for_layer(entry.key.layer);
            entry.generation++;
            entry.refcount = 0;
            entry.has_cold_backing = false;
            entry.origin = llm_expert_residency_origin::demand;
            entry.background_useful = false;
            generation_changes++;
            entry.state = hot_slot_state::loading;
            const auto loading = cache_policy_result(hot_policy.load_begin(
                slot, entry.generation, policy_key_for(entry.key),
                payload_for_key(entry.key), hot_physical_slot_footprint_bytes));
            if (!loading.is_ready()) return fail(loading);
            LLM_EXPERT_TRACE_INSTANT("k3.cache.hot", "reserve", "request_id", active_request_id,
                "layer", entry.key.layer, "original_expert_id", entry.key.expert,
                "slot_id", slot, "generation", entry.generation,
                "layout_class_id", entry.layout_class_id);
            unique_slots[unique_index] = int32_t(slot);
        }

        size_t transaction_copies = 0;
        size_t transaction_bytes = 0;
        const int64_t copy_start_us = miss_count > 0 ? ggml_time_us() : 0;
        auto copy_result = llm_expert_provider_result::success();
        bool injected_remote_failure = false;
        const auto inject_remote_failure_after_enqueue = [&]() {
            const bool active = config.remote_single &&
                device_transfer_failure_for_testing.front() &&
                (!device_transfer_failure_decode_only_for_testing.front() ||
                    hot_policy_phase == llm_expert_cache_policy_phase::decode);
            if (!active) return llm_expert_provider_result::success();
            injected_remote_failure = true;
            injected_device_failure_waves++;
            injected_device_failure_participants++;
            return llm_expert_provider_result::failure(
                llm_expert_provider_error::copy_failed);
        };
        if (config.cold_mode) {
            size_t expected_bytes = 0;
            for (size_t index = 0; index < miss_count; ++index) {
                const uint64_t payload = payload_for_key(unique_keys[miss_unique_indices[index]]);
                if (payload > SIZE_MAX - expected_bytes) {
                    copy_result = llm_expert_provider_result::failure(llm_expert_provider_error::allocation_failed);
                    break;
                }
                expected_bytes += size_t(payload);
            }
        }
        if (config.cold_mode) {
            if (config.storage && config.async_transport && config.scheduler) {
                size_t submitted_count = 0;
                size_t scheduled_count = 0;
                size_t storage_read_count = 0;
                bool first_storage_wait = false;
                llm_expert_phase10_issue_ahead_event issue_ahead;
                issue_ahead.sequence = phase10_issue_ahead_events;
                issue_ahead.request = active_request_id;
                issue_ahead.ubatch_ordinal = request_ubatch_ordinal;
                issue_ahead.layer = binding.layer;
                issue_ahead.logical_ids = uint32_t(logical_id_count);
                issue_ahead.unique_ids = uint32_t(unique_count);
                issue_ahead.demand_misses = uint32_t(miss_count);
                issue_ahead.serial_control = config.phase10_serial_issue_for_testing;
                size_t scheduler_enqueue_attempts = 0;
                bool storage_planning_complete = false;
                const auto record_first_wait = [&]() {
                    if (issue_ahead.first_wait_us != 0) return;
                    issue_ahead.first_wait_us = uint64_t(ggml_time_us());
                    issue_ahead.scheduler_enqueue_attempts_before_first_wait =
                        uint32_t(scheduler_enqueue_attempts);
                    issue_ahead.storage_reads_submitted_before_first_wait = uint32_t(submitted_count);
                    issue_ahead.first_wait_after_all_demand_enqueue_attempts =
                        scheduler_enqueue_attempts >= miss_count;
                    const size_t immediately_admissible_reads =
                        std::min(storage_read_count, size_t(transfer_lane_capacity));
                    issue_ahead.first_wait_after_all_storage_submissions =
                        storage_planning_complete &&
                        submitted_count >= (config.phase10_serial_issue_for_testing ?
                            storage_read_count : immediately_admissible_reads);
                };
                const auto accept_scheduled = [&](size_t index, const llm_expert_schedule_result & scheduled) {
                    auto & flight = async_flights[index];
                    if (!scheduled.accepted()) {
                        copy_result = llm_expert_provider_result::failure(
                            scheduled.disposition == llm_expert_schedule_disposition::generation_exhausted ?
                                llm_expert_provider_error::generation_exhausted : llm_expert_provider_error::busy);
                        return;
                    }
                    flight.handle = scheduled.handle;
                    flight.flight_id = {
                        config.async_transport->diagnostics().transport_epoch,
                        scheduled.handle.slot,
                        scheduled.handle.generation,
                        flight.key,
                        layout_class_for_layer(flight.key.layer),
                    };
                    flight.scheduler_joined = scheduled.disposition == llm_expert_schedule_disposition::joined;
                    llm_expert_request_snapshot scheduler_snapshot;
                    if (config.scheduler->snapshot(scheduled.handle, scheduler_snapshot) !=
                            llm_expert_schedule_disposition::admitted) {
                        copy_result = llm_expert_provider_result::failure(
                            llm_expert_provider_error::stale_generation);
                        return;
                    }
                    flight.scheduler_state = scheduler_snapshot.state;
                    if (scheduler_snapshot.metadata.layout_class_id != flight.flight_id.layout_class_id) {
                        copy_result = llm_expert_provider_result::failure(
                            llm_expert_provider_error::metadata_mismatch);
                        return;
                    }
                    flight.scheduler_active = !flight.scheduler_joined ||
                        scheduler_snapshot.state == llm_expert_request_state::queued;
                    if (flight.scheduler_joined) {
                        auto * prediction = find_predictive_storage_locked(flight.key);
                        if (prediction != nullptr &&
                                prediction->flight.handle.slot == scheduled.handle.slot &&
                                prediction->flight.handle.generation == scheduled.handle.generation) {
                            flight.predictive_storage_joined = true;
                            flight.prediction_sequence = prediction->prediction_sequence;
                            flight.scheduler_active = true;
                            flight.scheduler_taken = scheduler_snapshot.state !=
                                llm_expert_request_state::queued;
                        } else {
                            auto * background = find_background_locked(flight.key);
                            if (background != nullptr && background->predictive &&
                                    background->scheduler_handle.slot == scheduled.handle.slot &&
                                    background->scheduler_handle.generation == scheduled.handle.generation) {
                                flight.predictive_hot_joined = true;
                                flight.prediction_sequence = background->prediction_sequence;
                                flight.predictive_hot_slot = background->hot_slot;
                                flight.predictive_hot_generation = background->hot_generation;
                                flight.scheduler_active = false;
                                flight.scheduler_taken = true;
                            }
                        }
                    }
                    scheduled_count++;
                };

                // Attempt every current-layer demand before any scheduler or
                // storage wait. A demand fenced by a cancelling speculative
                // generation is retried only after all other keys have had
                // their first enqueue attempt.
                for (size_t index = 0; index < miss_count && copy_result.is_ready(); ++index) {
                    auto & flight = async_flights[index];
                    flight = {};
                    const uint32_t unique_index = miss_unique_indices[index];
                    flight.key = unique_keys[unique_index];
                    flight.scheduler_enqueue_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now().time_since_epoch()).count();
                    auto scheduled = config.scheduler->enqueue(
                        flight.key, llm_expert_priority::demand_current_layer,
                        llm_expert_readiness::device_ready,
                        demand_metadata(layout_class_for_layer(flight.key.layer)));
                    scheduler_enqueue_attempts++;
                    if (scheduled.disposition == llm_expert_schedule_disposition::busy &&
                            scheduled.handle.valid()) {
                        flight.deferred_release_handle = scheduled.handle;
                        flight.scheduler_deferred_retry = true;
                        continue;
                    }
                    accept_scheduled(index, scheduled);
                }
                for (size_t index = 0; index < miss_count && copy_result.is_ready(); ++index) {
                    auto & flight = async_flights[index];
                    while (flight.scheduler_deferred_retry) {
                        issue_ahead.scheduler_release_waits++;
                        record_first_wait();
                        if (provider_lock != nullptr) provider_lock->unlock();
                        const auto drained = config.scheduler->wait_until_released(
                            flight.deferred_release_handle);
                        if (provider_lock != nullptr) provider_lock->lock();
                        if (drained != llm_expert_schedule_disposition::admitted) {
                            copy_result = llm_expert_provider_result::failure(
                                llm_expert_provider_error::stale_generation);
                            break;
                        }
                        const auto scheduled = config.scheduler->enqueue(
                            flight.key, llm_expert_priority::demand_current_layer,
                            llm_expert_readiness::device_ready,
                            demand_metadata(layout_class_for_layer(flight.key.layer)));
                        scheduler_enqueue_attempts++;
                        if (scheduled.disposition == llm_expert_schedule_disposition::busy &&
                                scheduled.handle.valid()) {
                            flight.deferred_release_handle = scheduled.handle;
                            continue;
                        }
                        flight.scheduler_deferred_retry = false;
                        accept_scheduled(index, scheduled);
                    }
                }

                // All scheduler demands are now admitted or joined. Cold-cache
                // joins and storage planning may wait, but cannot hold back a
                // later current-layer demand enqueue.
                for (size_t index = 0; index < miss_count && copy_result.is_ready(); ++index) {
                    auto & flight = async_flights[index];
                    if (flight.predictive_hot_joined) {
                        auto * background = find_background_locked(flight.key);
                        if (background == nullptr || !background->predictive ||
                                background->prediction_sequence != flight.prediction_sequence) {
                            copy_result = llm_expert_provider_result::failure(
                                llm_expert_provider_error::stale_generation);
                            break;
                        }
                        record_first_wait();
                        if (provider_lock != nullptr) provider_lock->unlock();
                        copy_result = transfer_ring->wait_for_hot(
                            execution_backend, background->hot_slot,
                            background->hot_generation,
                            deterministic_policy_terminals);
                        if (provider_lock != nullptr) provider_lock->lock();
                        const uint64_t completion_bound =
                            background->origin_operation_ordinal == UINT64_MAX ?
                                UINT64_MAX : background->origin_operation_ordinal + 1;
                        if (copy_result.is_ready()) {
                            copy_result = finish_background_before_locked(completion_bound);
                        }
                        if (!copy_result.is_ready()) break;
                        const uint32_t hot_slot = flight.predictive_hot_slot;
                        if (hot_slot >= directory_slots.size()) {
                            copy_result = llm_expert_provider_result::failure(
                                llm_expert_provider_error::metadata_mismatch);
                            break;
                        }
                        auto & predicted_entry = directory_slots[hot_slot];
                        if (predicted_entry.state != hot_slot_state::ready ||
                                predicted_entry.generation != flight.predictive_hot_generation ||
                                !expert_key_matches(predicted_entry.key, flight.key)) {
                            copy_result = llm_expert_provider_result::failure(
                                llm_expert_provider_error::metadata_mismatch);
                            break;
                        }
                        predicted_entry.origin = llm_expert_residency_origin::demand;
                        predicted_entry.background_useful = true;
                        predicted_entry.speculative_deadline = 0;
                        predicted_entry.speculative_utility = 0;
                        background_useful++;
                        background_later_joins++;
                        const auto touched = cache_policy_result(hot_policy.hit(
                            hot_slot, predicted_entry.generation));
                        if (!touched.is_ready()) {
                            copy_result = touched;
                            break;
                        }
                        const uint32_t abandoned_slot = candidate_slots[index];
                        auto & abandoned = directory_slots[abandoned_slot];
                        const auto abandoned_policy = cache_policy_result(
                            hot_policy.load_failed(abandoned_slot, abandoned.generation));
                        if (!abandoned_policy.is_ready()) {
                            copy_result = abandoned_policy;
                            break;
                        }
                        const uint64_t abandoned_generation = abandoned.generation;
                        abandoned = {};
                        abandoned.generation = abandoned_generation;
                        unique_slots[miss_unique_indices[index]] = int32_t(hot_slot);
                        flight.processed = true;
                        continue;
                    }
                    if (flight.predictive_storage_joined) {
                        auto * prediction = find_predictive_storage_locked(flight.key);
                        if (prediction == nullptr ||
                                prediction->prediction_sequence != flight.prediction_sequence) {
                            copy_result = llm_expert_provider_result::failure(
                                llm_expert_provider_error::stale_generation);
                            break;
                        }
                        record_first_wait();
                        copy_result = claim_predictive_storage_for_demand_locked(
                            *prediction, flight, provider_lock);
                        if (!copy_result.is_ready()) break;
                        cold_references[index] = flight.cold;
                        continue;
                    }
                    llm_cold_demand_lookup lookup = llm_cold_demand_lookup::missing;
                    copy_result = cold_cache->lookup_demand(flight.key, flight.cold, lookup);
                    if (!copy_result.is_ready()) break;
                    if (config.async_cold_fill && lookup == llm_cold_demand_lookup::joined_loading) {
                        flight.cold = {};
                        lookup = llm_cold_demand_lookup::missing;
                    } else if (!flight.scheduler_active &&
                            lookup == llm_cold_demand_lookup::joined_loading) {
                        record_first_wait();
                        if (provider_lock != nullptr) provider_lock->unlock();
                        copy_result = cold_cache->wait_until_ready(flight.cold);
                        if (provider_lock != nullptr) provider_lock->lock();
                        if (!copy_result.is_ready()) break;
                        lookup = llm_cold_demand_lookup::ready;
                    }
                    if (flight.scheduler_active && !flight.scheduler_joined &&
                        lookup == llm_cold_demand_lookup::joined_loading) {
                        copy_result = llm_expert_provider_result::failure(
                            llm_expert_provider_error::metadata_mismatch);
                        break;
                    }
                    flight.cold_hit = lookup == llm_cold_demand_lookup::ready;
                    flight.direct_storage = lookup == llm_cold_demand_lookup::missing;
                    flight.reserved = false;
                    cold_references[index] = flight.cold;
                    storage_read_count += flight.direct_storage;
                }
                storage_planning_complete = copy_result.is_ready();
                issue_ahead.scheduler_enqueued_before_first_take = uint32_t(scheduled_count);
                issue_ahead.storage_reads = uint32_t(storage_read_count);
                issue_ahead.first_take_after_all_demand_enqueues =
                    copy_result.is_ready() && scheduled_count == miss_count;

                const auto publish_completed_read = [&](size_t index,
                        const llm_expert_async_read_completion & completion,
                        llm_expert_async_result waited) {
                    auto & flight = async_flights[index];
                    const auto released = config.async_transport->release_read(flight.handle);
                    if (released == llm_expert_async_result::ready) flight.read_active = false;
                    const auto storage_error = waited == llm_expert_async_result::ready ?
                        llm_expert_storage_error::none :
                        (waited == llm_expert_async_result::closed ? llm_expert_storage_error::cancelled :
                         (completion.native_error == 0 ? llm_expert_storage_error::short_read :
                          llm_expert_storage_error::io_error));
                    config.storage->record_async_read(
                        flight.destination_count, completion.bytes_completed, storage_error,
                        completion.native_error);
                    const bool integrity_matches = waited == llm_expert_async_result::ready &&
                        released == llm_expert_async_result::ready &&
                        finalize_payload_integrity(config.integrity_mode, *config.storage,
                            flight.destinations.data(), flight.destination_count,
                            completion.integrity_status, completion.digest);
                    if (!integrity_matches) {
                        return llm_expert_provider_result::failure(
                            waited == llm_expert_async_result::closed ? llm_expert_provider_error::cancelled :
                            llm_expert_provider_error::copy_failed);
                    }
                    flight.read_completed = true;
                    if (flight.direct_storage) {
                        auto completed = transfer_ring->complete_direct_storage(
                            flight.lane, completion.bytes_completed);
                        if (completed.is_ready()) flight.direct_storage_completed = true;
                        return completed;
                    }
                    auto published = cold_cache->publish_ready(flight.key, flight.cold);
                    if (published.is_ready()) flight.reserved = false;
                    return published;
                };

                size_t scheduler_owned_count = 0;
                for (size_t index = 0; index < miss_count; ++index) {
                    scheduler_owned_count += async_flights[index].scheduler_active &&
                        !async_flights[index].scheduler_taken;
                }
                for (size_t owned = 0; owned < scheduler_owned_count && copy_result.is_ready(); ++owned) {
                    llm_expert_request_snapshot selected;
                    const auto taken = config.scheduler->take_next(selected);
                    const uint64_t scheduler_take_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now().time_since_epoch()).count();
                    if (issue_ahead.first_take_us == 0) {
                        issue_ahead.first_take_us = uint64_t(ggml_time_us());
                    }
                    size_t index = miss_count;
                    if (taken.disposition == llm_expert_schedule_disposition::admitted) {
                        for (size_t candidate = 0; candidate < miss_count; ++candidate) {
                            const auto & candidate_flight = async_flights[candidate];
                            if (candidate_flight.scheduler_active && !candidate_flight.scheduler_taken &&
                                candidate_flight.handle.slot == selected.handle.slot &&
                                candidate_flight.handle.generation == selected.handle.generation) {
                                index = candidate;
                                break;
                            }
                        }
                    }
                    if (index == miss_count) {
                        if (taken.disposition == llm_expert_schedule_disposition::admitted) {
                            (void) config.scheduler->transition(selected.handle,
                                llm_expert_request_state::submitting,
                                llm_expert_request_state::draining);
                            (void) config.scheduler->finish(
                                selected.handle, llm_expert_request_state::failed);
                            (void) config.scheduler->release_terminal(selected.handle);
                        }
                        copy_result = llm_expert_provider_result::failure(
                            llm_expert_provider_error::metadata_mismatch);
                        break;
                    }
                    auto & flight = async_flights[index];
                    flight.scheduler_taken = true;
                    if (phase10_scheduler_events) {
                        if (phase10_scheduler_event_count < phase10_scheduler_events->size()) {
                            (*phase10_scheduler_events)[phase10_scheduler_event_count] = {
                                phase10_scheduler_event_count, flight.key,
                                flight.scheduler_enqueue_ns, scheduler_take_ns,
                            };
                            phase10_scheduler_event_count++;
                        } else {
                            phase10_scheduler_events_dropped++;
                        }
                    }
                    flight.scheduler_state = llm_expert_request_state::submitting;
                    if (flight.cold_hit) {
                        if (config.scheduler->transition(flight.handle,
                                llm_expert_request_state::submitting,
                                llm_expert_request_state::host_ready) !=
                                llm_expert_schedule_disposition::admitted) {
                            copy_result = llm_expert_provider_result::failure(
                                llm_expert_provider_error::metadata_mismatch);
                            break;
                        }
                        flight.scheduler_state = llm_expert_request_state::host_ready;
                        continue;
                    }
                }

                const auto submit_admissible_direct_reads = [&]() {
                    size_t admitted = 0;
                    size_t pending_cold = 0;
                    size_t active_direct = 0;
                    for (size_t index = 0; index < miss_count; ++index) {
                        const auto & flight = async_flights[index];
                        pending_cold += !flight.processed && (flight.cold_hit ||
                            (!flight.direct_storage && flight.read_completed &&
                             cold_cache->ready(flight.cold)));
                        active_direct += flight.direct_storage && flight.submitted &&
                            !flight.processed;
                    }
                    // Leave one lane available while cold-ready work is
                    // pending; direct reads regain the full ring afterward.
                    const size_t direct_lane_limit = transfer_lane_capacity -
                        size_t(pending_cold != 0);
                    bool serial_read_active = false;
                    if (config.phase10_serial_issue_for_testing) {
                        for (size_t index = 0; index < miss_count; ++index) {
                            serial_read_active |= async_flights[index].read_active;
                        }
                    }
                    for (size_t index = 0; index < miss_count; ++index) {
                        auto & flight = async_flights[index];
                        if (!flight.direct_storage || flight.submitted || flight.read_completed ||
                                !flight.scheduler_taken) continue;
                        if (active_direct + admitted >= direct_lane_limit) break;
                        if (config.phase10_serial_issue_for_testing &&
                                (serial_read_active || admitted != 0)) break;
                        const uint32_t slot = candidate_slots[index];
                        auto & entry = directory_slots[slot];
                        auto reserved = transfer_ring->reserve_direct_storage(
                            flight.flight_id.layout_class_id, slot, entry.generation,
                            flight.lane, flight.flight_id, false);
                        if (!reserved.is_ready()) {
                            if (reserved.error == llm_expert_provider_error::busy) break;
                            return reserved;
                        }
                        auto destinations_ready = transfer_ring->storage_destinations(
                            flight.lane, flight.destinations.data(), flight.destinations.size(),
                            flight.destination_count);
                        if (!destinations_ready.is_ready()) return destinations_ready;
                        const auto planned = config.storage->make_read_plan(
                                flight.key, flight.destinations.data(), flight.destination_count,
                                flight.operations.data(), flight.operations.size(), flight.operation_count);
                        if (!planned.is_ready()) return llm_expert_provider_result::failure(
                            llm_expert_provider_error::metadata_mismatch);
                        const llm_expert_async_operation_identity identity = {
                            config.async_transport->diagnostics().transport_epoch,
                            flight.handle, 0, flight.key, llm_expert_readiness::device_ready,
                            llm_expert_priority::demand_current_layer,
                            flight.flight_id.layout_class_id,
                        };
                        if (config.async_transport->submit_read_plan(
                                identity, flight.operations.data(), flight.operation_count, true) !=
                                llm_expert_async_result::ready) {
                            return llm_expert_provider_result::failure(
                                llm_expert_provider_error::copy_failed);
                        }
                        flight.submitted = true;
                        flight.read_active = true;
                        submitted_count++;
                        admitted++;
                        if (config.scheduler->transition(flight.handle,
                                llm_expert_request_state::submitting,
                                llm_expert_request_state::io_in_flight) !=
                                llm_expert_schedule_disposition::admitted) {
                            return llm_expert_provider_result::failure(
                                llm_expert_provider_error::copy_failed);
                        }
                        flight.scheduler_state = llm_expert_request_state::io_in_flight;
                    }
                    if (admitted != 0) config.async_transport->start_deferred_reads();
                    return llm_expert_provider_result::success();
                };

                if (copy_result.is_ready()) copy_result = submit_admissible_direct_reads();

                size_t completed_count = 0;
                for (size_t index = 0; index < miss_count; ++index) {
                    completed_count += async_flights[index].processed;
                }
                while (completed_count < miss_count && copy_result.is_ready()) {
                    copy_result = submit_admissible_direct_reads();
                    if (!copy_result.is_ready()) break;
                    size_t index = miss_count;
                    for (size_t candidate = 0; candidate < miss_count; ++candidate) {
                        const auto & candidate_flight = async_flights[candidate];
                        if (!candidate_flight.processed && (candidate_flight.cold_hit ||
                                (!candidate_flight.direct_storage && candidate_flight.read_completed &&
                                 cold_cache->ready(candidate_flight.cold)))) {
                            index = candidate;
                            break;
                        }
                    }
                    if (index == miss_count) {
                        for (size_t candidate = 0; candidate < miss_count; ++candidate) {
                            const auto & candidate_flight = async_flights[candidate];
                            if (!candidate_flight.processed && candidate_flight.direct_storage &&
                                    candidate_flight.direct_storage_completed) {
                                index = candidate;
                                break;
                            }
                        }
                    }
                    if (index == miss_count) {
                        size_t pending_count = 0;
                        for (size_t candidate = 0; candidate < miss_count; ++candidate) {
                            const auto & candidate_flight = async_flights[candidate];
                            if (candidate_flight.submitted && !candidate_flight.read_completed) {
                                async_handles[pending_count++] = candidate_flight.handle;
                            }
                        }
                        if (pending_count == 0) {
                            copy_result = llm_expert_provider_result::failure(
                                llm_expert_provider_error::metadata_mismatch);
                            break;
                        }
                        llm_expert_async_read_completion completion;
                        llm_expert_request_handle completed_handle;
                        if (!first_storage_wait) {
                            first_storage_wait = true;
                            record_first_wait();
                        }
                        if (provider_lock != nullptr) provider_lock->unlock();
                        const auto waited = config.async_transport->wait_any_read(
                            async_handles.data(), pending_count, completed_handle, completion,
                            abort_callback, abort_callback_data);
                        if (provider_lock != nullptr) provider_lock->lock();
                        for (size_t candidate = 0; candidate < miss_count; ++candidate) {
                            const auto & candidate_flight = async_flights[candidate];
                            if (!candidate_flight.read_completed &&
                                candidate_flight.handle.slot == completed_handle.slot &&
                                candidate_flight.handle.generation == completed_handle.generation) {
                                index = candidate;
                                break;
                            }
                        }
                        if (index == miss_count) {
                            copy_result = llm_expert_provider_result::failure(
                                waited == llm_expert_async_result::closed ?
                                    llm_expert_provider_error::cancelled :
                                    llm_expert_provider_error::stale_generation);
                            break;
                        }
                        copy_result = publish_completed_read(index, completion, waited);
                        continue;
                    }

                    auto & flight = async_flights[index];
                    flight.processed = true;
                    completed_count++;
                    if (flight.submitted && config.scheduler->transition(flight.handle,
                            llm_expert_request_state::io_in_flight,
                            llm_expert_request_state::host_ready) != llm_expert_schedule_disposition::admitted) {
                        copy_result = llm_expert_provider_result::failure(
                            llm_expert_provider_error::metadata_mismatch);
                    }
                    if (copy_result.is_ready() && flight.submitted) {
                        flight.scheduler_state = llm_expert_request_state::host_ready;
                    }
                    if (!copy_result.is_ready()) break;

                    const uint32_t slot = candidate_slots[index];
                    auto & entry = directory_slots[slot];
                    if (copy_result.is_ready() && flight.scheduler_active &&
                        config.scheduler->transition(flight.handle, llm_expert_request_state::host_ready,
                            llm_expert_request_state::h2d_in_flight) != llm_expert_schedule_disposition::admitted) {
                        copy_result = llm_expert_provider_result::failure(
                            llm_expert_provider_error::metadata_mismatch);
                    }
                    if (copy_result.is_ready() && flight.scheduler_active) {
                        flight.scheduler_state = llm_expert_request_state::h2d_in_flight;
                    }
                    if (copy_result.is_ready() && flight.direct_storage) {
                        transfer_lanes[index] = flight.lane;
                    } else if (copy_result.is_ready()) {
                        copy_result = transfer_ring->reserve(*cold_cache, flight.cold, slot,
                            entry.generation, transfer_lanes[index], flight.flight_id);
                        if (copy_result.is_ready()) {
                            copy_result = transfer_ring->stage(transfer_lanes[index],
                                cold_cache->bundle(flight.cold.layout_class_id));
                        }
                    }
                    if (copy_result.is_ready()) {
                        transfer_bindings.clear();
                        transfer_bindings.push_back({ transfer_lanes[index],
                            pool->bundles[flight.flight_id.layout_class_id], slot });
                        copy_result = transfer_ring->transfer_wave(execution_backend, transfer_bindings);
                        if (copy_result.is_ready()) flight.h2d_submitted = true;
                        if (copy_result.is_ready() && config.async_cold_fill && flight.direct_storage) {
                            (void) transfer_ring->try_queue_cold_fill(
                                *cold_cache, transfer_lanes[index]);
                        }
                        if (copy_result.is_ready()) {
                            copy_result = inject_remote_failure_after_enqueue();
                        }
                    }
                    const auto abort_requested = [&]() {
                        if (abort_callback == nullptr) return false;
                        if (provider_lock != nullptr) provider_lock->unlock();
                        const bool requested = abort_callback(abort_callback_data);
                        if (provider_lock != nullptr) provider_lock->lock();
                        return requested;
                    };
                    const auto cancel_post_h2d = [&]() {
                        auto cancelled = transfer_ring->cancel_after_h2d(transfer_lanes[index]);
                        if (!cancelled.is_ready()) return cancelled;
                        if (flight.scheduler_active) {
                            if (config.scheduler->begin_demand_cancellation(
                                    flight.handle, llm_expert_request_state::h2d_in_flight) !=
                                    llm_expert_schedule_disposition::admitted) {
                                return llm_expert_provider_result::failure(
                                    llm_expert_provider_error::metadata_mismatch);
                            }
                            if (config.scheduler->transition(flight.handle,
                                    llm_expert_request_state::cancelling,
                                    llm_expert_request_state::draining) !=
                                    llm_expert_schedule_disposition::admitted) {
                                return llm_expert_provider_result::failure(
                                    llm_expert_provider_error::metadata_mismatch);
                            }
                            if (config.scheduler->finish(flight.handle,
                                    llm_expert_request_state::cancelled) !=
                                    llm_expert_schedule_disposition::admitted) {
                                return llm_expert_provider_result::failure(
                                    llm_expert_provider_error::metadata_mismatch);
                            }
                            if (config.scheduler->release_terminal(flight.handle) !=
                                    llm_expert_schedule_disposition::admitted) {
                                return llm_expert_provider_result::failure(
                                    llm_expert_provider_error::metadata_mismatch);
                            }
                            flight.scheduler_active = false;
                            flight.scheduler_state = llm_expert_request_state::free;
                        }
                        post_h2d_cancellations++;
                        last_cancelled_flight = flight.flight_id;
                        last_cancelled_lane = transfer_lanes[index].lane;
                        last_cancelled_lane_generation = transfer_lanes[index].generation;
                        last_cancelled_hot_slot = slot;
                        last_cancelled_hot_generation = entry.generation;
                        return llm_expert_provider_result::failure(llm_expert_provider_error::cancelled);
                    };
                    if (copy_result.is_ready() && abort_requested()) {
                        copy_result = cancel_post_h2d();
                    }
                    if (copy_result.is_ready()) {
                        copy_result = transfer_ring->wait_for_hot(
                            execution_backend, slot, entry.generation, deterministic_policy_terminals);
                        if (copy_result.is_ready()) flight.device_ready = true;
                    }
                    if (copy_result.is_ready()) {
                        issue_ahead.demand_ready_before_use++;
                        issue_ahead.last_demand_ready_us = uint64_t(ggml_time_us());
                    }
                    if (copy_result.is_ready() && abort_requested()) {
                        copy_result = cancel_post_h2d();
                    }
                    if (flight.scheduler_active) {
                        if (copy_result.is_ready()) {
                            const auto device_ready = config.scheduler->transition(flight.handle,
                                llm_expert_request_state::h2d_in_flight,
                                llm_expert_request_state::device_ready);
                            if (device_ready == llm_expert_schedule_disposition::admitted) {
                                flight.scheduler_state = llm_expert_request_state::device_ready;
                                (void) config.scheduler->finish(
                                    flight.handle, llm_expert_request_state::complete);
                                flight.scheduler_state = llm_expert_request_state::complete;
                            } else {
                                copy_result = llm_expert_provider_result::failure(
                                    llm_expert_provider_error::metadata_mismatch);
                            }
                        }
                        if (copy_result.is_ready()) {
                            last_completed_flight = flight.flight_id;
                            last_completed_lane = transfer_lanes[index].lane;
                            last_completed_lane_generation = transfer_lanes[index].generation;
                            last_completed_hot_slot = slot;
                            last_completed_hot_generation = entry.generation;
                            if (last_cancelled_flight.valid() &&
                                expert_key_matches(last_cancelled_flight.key, flight.key)) {
                                retry_after_cancel_flight = flight.flight_id;
                                retry_after_cancel_lane = transfer_lanes[index].lane;
                                retry_after_cancel_lane_generation = transfer_lanes[index].generation;
                                retry_after_cancel_hot_slot = slot;
                                retry_after_cancel_hot_generation = entry.generation;
                            }
                            (void) config.scheduler->release_terminal(flight.handle);
                            flight.scheduler_active = false;
                            flight.scheduler_state = llm_expert_request_state::free;
                        }
                    }
                }

                if (!copy_result.is_ready()) {
                    for (size_t index = 0; index < miss_count; ++index) {
                        auto & flight = async_flights[index];
                        if (flight.read_active) {
                            (void) config.async_transport->cancel_read(flight.handle);
                            llm_expert_async_read_completion discarded;
                            if (provider_lock != nullptr) provider_lock->unlock();
                            const auto drained = config.async_transport->wait_read(flight.handle, discarded);
                            if (provider_lock != nullptr) provider_lock->lock();
                            (void) config.async_transport->release_read(flight.handle);
                            const auto storage_error = copy_result.error == llm_expert_provider_error::cancelled ?
                                llm_expert_storage_error::cancelled :
                                (drained == llm_expert_async_result::ready ? llm_expert_storage_error::none :
                                 (discarded.native_error == 0 ? llm_expert_storage_error::short_read :
                                  llm_expert_storage_error::io_error));
                            config.storage->record_async_read(
                                flight.destination_count, discarded.bytes_completed,
                                storage_error, discarded.native_error);
                            flight.read_active = false;
                        }
                        if (flight.direct_storage && !flight.h2d_submitted &&
                            flight.lane.generation != 0) {
                            (void) transfer_ring->discard_staging(flight.lane);
                            flight.lane = {};
                        }
                        if (flight.scheduler_active) {
                            if (flight.scheduler_state >= llm_expert_request_state::queued &&
                                flight.scheduler_state <= llm_expert_request_state::h2d_in_flight) {
                                (void) config.scheduler->transition(flight.handle, flight.scheduler_state,
                                    llm_expert_request_state::draining);
                                flight.scheduler_state = llm_expert_request_state::draining;
                                (void) config.scheduler->finish(
                                    flight.handle, llm_expert_request_state::failed);
                                flight.scheduler_state = llm_expert_request_state::failed;
                            } else if (flight.scheduler_state == llm_expert_request_state::device_ready) {
                                (void) config.scheduler->finish(
                                    flight.handle, llm_expert_request_state::complete);
                                flight.scheduler_state = llm_expert_request_state::complete;
                            }
                            (void) config.scheduler->release_terminal(flight.handle);
                            flight.scheduler_active = false;
                            flight.scheduler_state = llm_expert_request_state::free;
                        }
                        if (flight.reserved) (void) cold_cache->fail_reservation(flight.key, flight.cold);
                    }
                }
                if (!first_storage_wait && storage_read_count == 0) {
                    issue_ahead.scheduler_enqueue_attempts_before_first_wait =
                        uint32_t(scheduler_enqueue_attempts);
                    issue_ahead.first_wait_after_all_demand_enqueue_attempts =
                        scheduler_enqueue_attempts >= miss_count;
                    issue_ahead.first_wait_after_all_storage_submissions = true;
                }
                if (copy_result.is_ready()) {
                    issue_ahead.demand_completion_us = uint64_t(ggml_time_us());
                    issue_ahead.all_demand_ready_before_use =
                        issue_ahead.demand_ready_before_use == miss_count;
                }
                if (phase10_issue_ahead_trace) {
                    if (!issue_ahead.serial_control &&
                        (!issue_ahead.first_wait_after_all_demand_enqueue_attempts ||
                         !issue_ahead.first_take_after_all_demand_enqueues ||
                         !issue_ahead.first_wait_after_all_storage_submissions)) {
                        phase10_issue_ahead_violations++;
                    }
                    if (phase10_issue_ahead_trace_count < phase10_issue_ahead_trace->size()) {
                        (*phase10_issue_ahead_trace)[phase10_issue_ahead_trace_count++] = issue_ahead;
                    } else {
                        phase10_issue_ahead_trace_dropped++;
                    }
                    phase10_issue_ahead_events++;
                }
            } else {
                storage_load_context storage_context = {
                    config.storage, nullptr, nullptr, config.integrity_mode, provider_lock, {}, false,
                    abort_callback, abort_callback_data,
                };
                storage_context.layer_ids = &layout_registry.layer_ids;
                storage_context.layout_registry = &layout_registry;
                for (size_t index = 0; index < miss_count && copy_result.is_ready(); ++index) {
                    const uint32_t unique_index = miss_unique_indices[index];
                    copy_result = config.storage ? cold_cache->find_or_admit_with_loader(
                        unique_keys[unique_index], cold_references[index], load_storage_bundle, &storage_context) :
                        cold_cache->find_or_admit(
                            unique_keys[unique_index], registration->second, cold_references[index],
                            faults.fail_copy_after_tensors);
                }
                for (size_t wave_begin = 0; wave_begin < miss_count && copy_result.is_ready();
                        wave_begin += transfer_lane_capacity) {
                    const size_t wave_end = std::min(miss_count, wave_begin + transfer_lane_capacity);
                    transfer_bindings.clear();
                    for (size_t index = wave_begin; index < wave_end && copy_result.is_ready(); ++index) {
                        const uint32_t slot = candidate_slots[index];
                        copy_result = transfer_ring->reserve(*cold_cache, cold_references[index], slot,
                            directory_slots[slot].generation, transfer_lanes[index]);
                        if (copy_result.is_ready()) {
                            copy_result = transfer_ring->stage(transfer_lanes[index],
                                cold_cache->bundle(cold_references[index].layout_class_id));
                        }
                        if (copy_result.is_ready()) {
                            transfer_bindings.push_back({ transfer_lanes[index],
                                pool->bundles[cold_references[index].layout_class_id], slot });
                        }
                    }
                    if (copy_result.is_ready()) {
                        copy_result = transfer_ring->transfer_wave(execution_backend, transfer_bindings);
                        if (copy_result.is_ready()) {
                            copy_result = inject_remote_failure_after_enqueue();
                        }
                    }
                    if (copy_result.is_ready()) {
                        for (const auto & transfer : transfer_bindings) {
                            copy_result = transfer_ring->wait_for_hot(execution_backend,
                                transfer.hot_slot, directory_slots[transfer.hot_slot].generation,
                                deterministic_policy_terminals);
                            if (!copy_result.is_ready()) break;
                        }
                    }
                }
            }
            if (copy_result.is_ready()) {
                transaction_bytes = 0;
                for (size_t index = 0; index < miss_count; ++index) {
                    if (async_flights[index].predictive_hot_joined) continue;
                    const uint64_t payload = payload_for_key(async_flights[index].key);
                    if (payload > SIZE_MAX - transaction_bytes) {
                        copy_result = llm_expert_provider_result::failure(
                            llm_expert_provider_error::allocation_failed);
                        break;
                    }
                    transaction_bytes += size_t(payload);
                }
            }
        } else {
            bool copied = true;
            for (size_t index = 0; index < miss_count && copied; ++index) {
                const uint32_t unique_index = miss_unique_indices[index];
                const uint32_t slot = candidate_slots[index];
                const auto & key = unique_keys[unique_index];
                const auto & source = registration->second;
                const auto & hot_bundle = pool->bundles[layout_class_for_layer(key.layer)];
                copied = copy_expert_projection(hot_bundle.up, source.up, source.n_expert, config.capacity,
                             key.expert, slot, transaction_bytes, transaction_copies, faults.fail_copy_after_tensors) &&
                    copy_expert_projection(hot_bundle.gate, source.gate, source.n_expert, config.capacity,
                             key.expert, slot, transaction_bytes, transaction_copies, faults.fail_copy_after_tensors) &&
                    copy_expert_projection(hot_bundle.gate_up, source.gate_up, source.n_expert, config.capacity,
                             key.expert, slot, transaction_bytes, transaction_copies, faults.fail_copy_after_tensors) &&
                    copy_expert_projection(hot_bundle.down, source.down, source.n_expert, config.capacity,
                             key.expert, slot, transaction_bytes, transaction_copies, faults.fail_copy_after_tensors);
            }
            if (!copied) copy_result = llm_expert_provider_result::failure(llm_expert_provider_error::copy_failed);
            counters.tensor_copies += transaction_copies;
        }
        h2d_bytes += transaction_bytes;
        if (miss_count > 0) h2d_time_us += ggml_time_us() - copy_start_us;

        if (!copy_result.is_ready()) {
            if (config.cold_mode) {
                (void) transfer_ring->cleanup_failed_lanes();
                if (injected_remote_failure) injected_device_failure_drained_waves++;
                for (size_t index = 0; index < miss_count; ++index) {
                    if (async_flights[index].predictive_hot_joined) continue;
                    auto & entry = directory_slots[candidate_slots[index]];
                    if (entry.has_cold_backing) {
                        (void) cold_cache->release(
                            { entry.cold_slot, entry.cold_generation, entry.layout_class_id },
                            llm_cold_reference_kind::hot);
                        entry.has_cold_backing = false;
                        entry.cold_slot = 0;
                        entry.cold_generation = 0;
                    }
                }
            }
            for (size_t index = 0; index < miss_count; ++index) {
                if (config.cold_mode && async_flights[index].predictive_hot_joined) continue;
                auto & entry = directory_slots[candidate_slots[index]];
                const auto failed = cache_policy_result(hot_policy.load_failed(
                    candidate_slots[index], entry.generation));
                if (!failed.is_ready()) {
                    metadata_mismatches++;
                    last_remap_error = failed.error;
                }
                entry.state = hot_slot_state::failed;
                entry.refcount = 0;
            }
            const auto released = release_request_pins_locked();
            if (!released.is_ready()) copy_result = released;
            faults.fail_copy_after_tensors = SIZE_MAX;
            copy_failures++;
            last_remap_error = copy_result.error;
            return fail(copy_result);
        }

        for (size_t index = 0; index < miss_count; ++index) {
            if (config.cold_mode && async_flights[index].predictive_hot_joined) {
                const uint32_t slot = async_flights[index].predictive_hot_slot;
                const auto pinned = pin_slot_locked(slot);
                if (!pinned.is_ready()) return fail(pinned);
                continue;
            }
            const uint32_t slot = candidate_slots[index];
            auto & entry = directory_slots[slot];
            const auto completed = cache_policy_result(hot_policy.load_complete(slot, entry.generation));
            if (!completed.is_ready()) return fail(completed);
            entry.state = hot_slot_state::ready;
            directory_forward[forward_index(entry.key)] = { int32_t(slot), entry.generation };
            LLM_EXPERT_TRACE_INSTANT("k3.cache.hot", "publish", "request_id", active_request_id,
                "layer", entry.key.layer, "original_expert_id", entry.key.expert,
                "slot_id", slot, "generation", entry.generation);
            admissions++;
            LLM_EXPERT_TRACE_INSTANT("k3.cache.hot", "admission", "request_id", active_request_id,
                "layer", entry.key.layer, "original_expert_id", entry.key.expert,
                "slot_id", slot, "generation", entry.generation);
            LLM_EXPERT_TRACE_COUNTER("k3.resource", "hot_cache_occupancy", 4,
                hot_cache_occupancy_locked());
            const auto pinned = pin_slot_locked(slot);
            if (!pinned.is_ready()) return fail(pinned);
        }

        if (config.cold_mode) {
            auto invariant = validate_tier_invariants_locked();
            if (!invariant.is_ready()) return fail(invariant);
        }

        for (size_t index = 0; index < logical_id_count; ++index) {
            execution_ids[index] = unique_slots[element_unique[index]];
            last_logical_ids[index] = logical_ids[index];
            last_execution_ids[index] = execution_ids[index];
            if (binding.hybrid) {
                cpu_execution_ids[index] = -1;
                last_cpu_execution_ids[index] = -1;
            }
        }
        if (binding.hybrid) {
            const auto hybrid_ids = llm_validate_hybrid_execution_ids(
                execution_ids, cpu_execution_ids, logical_id_count,
                int32_t(config.capacity), int32_t(cold_cache->diagnostics().effective_slots));
            if (!hybrid_ids.is_ready()) return fail(hybrid_ids);
            gpu_execution_lanes += logical_id_count;
        }
        last_id_count = logical_id_count;
        last_remap_layer = binding.layer;

        remap_checkpoints++;
        logical_id_total += logical_id_count;
        unique_id_total += unique_count;
        hits += hit_count;
        misses += miss_count;
        phase10_after_remap_locked(binding);
        return llm_expert_provider_result::success();
    }

    llm_expert_provider_result remap_cpu_cold_only_locked(
            const llm_expert_graph_binding & binding,
            const int32_t * logical_ids,
            size_t logical_id_count,
            int32_t * execution_ids,
            ggml_backend_t execution_backend,
            bool (*abort_callback)(void *),
            void * abort_callback_data,
            std::unique_lock<std::mutex> * provider_lock) noexcept {
        if (!binding.cpu_cold_only || binding.hybrid || binding.remote_single || binding.multi_device ||
            cold_cache == nullptr || binding.layout_class_id == LLM_EXPERT_LAYOUT_CLASS_INVALID ||
            ((config.storage == nullptr || config.async_transport == nullptr || config.scheduler == nullptr) &&
             !config.allow_non_cuda_target_for_testing)) {
            return fail(llm_expert_provider_result::failure(llm_expert_provider_error::invalid_binding));
        }
        const auto execution_device = execution_backend == nullptr ? nullptr :
            ggml_backend_get_device(execution_backend);
        if (execution_device != nullptr &&
            ggml_backend_dev_type(execution_device) != GGML_BACKEND_DEVICE_TYPE_CPU) {
            return fail(llm_expert_provider_result::failure(
                llm_expert_provider_error::unsupported_configuration));
        }

        auto released = release_request_pins_locked();
        if (!released.is_ready()) return fail(released);

        size_t unique_count = 0;
        for (size_t index = 0; index < logical_id_count; ++index) {
            const int32_t expert = logical_ids[index];
            if (expert < 0 || expert >= int32_t(n_expert)) {
                return fail(llm_expert_provider_result::failure(llm_expert_provider_error::invalid_key));
            }
            const llm_expert_key key = { binding.layer, expert };
            size_t unique_index = 0;
            while (unique_index < unique_count && !expert_key_matches(unique_keys[unique_index], key)) {
                unique_index++;
            }
            if (unique_index == unique_count) {
                if (unique_count >= config.capacity) {
                    return fail(llm_expert_provider_result::failure(
                        llm_expert_provider_error::unsupported_configuration));
                }
                unique_keys[unique_count] = key;
                unique_slots[unique_count] = -1;
                unique_lane_counts[unique_count] = 0;
                unique_index = unique_count++;
            }
            element_unique[index] = int32_t(unique_index);
            unique_lane_counts[unique_index]++;
        }

        if (request_ubatch_ordinal == UINT64_MAX) {
            return fail(llm_expert_provider_result::failure(
                llm_expert_provider_error::unsupported_configuration));
        }
        const uint64_t ubatch_ordinal = ++request_ubatch_ordinal;
        auto policy_result = cold_cache->policy_set_ubatch_ordinal(ubatch_ordinal);
        const auto requested_phase = binding.execution_ids->ne[1] > 1 ?
            llm_expert_cache_policy_phase::prefill : llm_expert_cache_policy_phase::decode;
        if (policy_result.is_ready() && requested_phase != hot_policy_phase) {
            policy_result = cold_cache->policy_phase_transition(requested_phase);
            if (policy_result.is_ready()) hot_policy_phase = requested_phase;
        }
        if (!policy_result.is_ready()) return fail(policy_result);

        for (size_t index = 0; index < unique_count; ++index) {
            policy_unique_indices[index] = uint32_t(index);
        }
        std::sort(policy_unique_indices.begin(), policy_unique_indices.begin() + unique_count,
            [&](uint32_t lhs, uint32_t rhs) {
                const auto & lhs_key = unique_keys[lhs];
                const auto & rhs_key = unique_keys[rhs];
                return lhs_key.layer != rhs_key.layer ? lhs_key.layer < rhs_key.layer :
                    lhs_key.expert < rhs_key.expert;
            });

        const auto cold_before = cold_cache->counter_diagnostics();
        if (config.storage != nullptr) {
            if (host_resident_demand == nullptr || provider_lock == nullptr) {
                return fail(llm_expert_provider_result::failure(
                    llm_expert_provider_error::invalid_binding));
            }
            auto loaded = host_resident_demand->plan(
                binding.layer, logical_ids, logical_id_count, host_resident_batch);
            if (!loaded.is_ready()) return fail(loaded);
            if (host_resident_batch.unique_count != unique_count ||
                host_resident_batch.occurrence_count != logical_id_count) {
                (void) host_resident_demand->release_request_holds(host_resident_batch);
                return fail(llm_expert_provider_result::failure(
                    llm_expert_provider_error::metadata_mismatch));
            }
            for (size_t order = 0; order < host_resident_batch.unique_count; ++order) {
                const uint32_t index = policy_unique_indices[order];
                if (index >= host_resident_batch.unique_count ||
                    !expert_key_matches(host_resident_batch.entries[index].key, unique_keys[index])) {
                    return fail(llm_expert_provider_result::failure(
                        llm_expert_provider_error::metadata_mismatch));
                }
                host_resident_batch.semantic_order[order] = index;
            }
            loaded = host_resident_demand->freeze_semantic_order(host_resident_batch);
            if (!loaded.is_ready()) return fail(loaded);
            if (!config.phase10_serial_issue_for_testing) {
                provider_lock->unlock();
                loaded = host_resident_demand->resolve_batch(
                    host_resident_batch, abort_callback, abort_callback_data);
                provider_lock->lock();
                if (!loaded.is_ready()) return fail(loaded);
            }
            for (size_t order = 0; order < host_resident_batch.unique_count; ++order) {
                const uint32_t index = host_resident_batch.semantic_order[order];
                auto & entry = host_resident_batch.entries[index];
                if (config.phase10_serial_issue_for_testing) {
                    provider_lock->unlock();
                    loaded = host_resident_demand->resolve_serial_next(
                        host_resident_batch, abort_callback, abort_callback_data);
                    provider_lock->lock();
                    if (!loaded.is_ready()) return fail(loaded);
                }
                loaded = host_resident_demand->complete_host_scheduler(entry);
                if (!loaded.is_ready()) {
                    (void) host_resident_demand->fail_serial_batch(
                        host_resident_batch, loaded.error, false);
                    return fail(loaded);
                }
                if (cpu_execution_pin_count >= cpu_execution_pins.size()) {
                    (void) host_resident_demand->fail_serial_batch(
                        host_resident_batch, llm_expert_provider_error::unsupported_configuration, false);
                    (void) release_request_pins_locked();
                    return fail(llm_expert_provider_result::failure(
                        llm_expert_provider_error::unsupported_configuration));
                }
                const auto acquired = host_resident_demand->transfer_request_hold_to_cpu_execution(entry);
                if (!acquired.is_ready()) {
                    (void) host_resident_demand->fail_serial_batch(
                        host_resident_batch, acquired.error, false);
                    (void) release_request_pins_locked();
                    return fail(acquired);
                }
                cpu_execution_pins[cpu_execution_pin_count++] = entry.reference;
                unique_slots[index] = int32_t(entry.reference.slot);
                cold_references[index] = entry.reference;
            }
            loaded = host_resident_demand->finish_serial_batch(host_resident_batch);
            if (!loaded.is_ready()) {
                (void) host_resident_demand->fail_serial_batch(
                    host_resident_batch, loaded.error, false);
                (void) release_request_pins_locked();
                return fail(loaded);
            }
            for (size_t occurrence = 0; occurrence < logical_id_count; ++occurrence) {
                element_unique[occurrence] = int32_t(
                    host_resident_batch.occurrence_to_unique[occurrence]);
            }
        } else {
            for (size_t policy_index = 0; policy_index < unique_count; ++policy_index) {
                const uint32_t unique_index = policy_unique_indices[policy_index];
                auto loaded = cold_cache->find_or_admit(
                    unique_keys[unique_index], registrations.at(binding.layer),
                    cold_references[unique_index]);
                if (!loaded.is_ready()) return fail(loaded);
                const auto acquired = cold_cache->acquire(
                    cold_references[unique_index], llm_cold_reference_kind::cpu_execution);
                if (!acquired.is_ready()) return fail(acquired);
                if (cpu_execution_pin_count >= cpu_execution_pins.size()) {
                    (void) cold_cache->release(
                        cold_references[unique_index], llm_cold_reference_kind::cpu_execution);
                    return fail(llm_expert_provider_result::failure(
                        llm_expert_provider_error::unsupported_configuration));
                }
                cpu_execution_pins[cpu_execution_pin_count++] = cold_references[unique_index];
                unique_slots[unique_index] = int32_t(cold_references[unique_index].slot);
            }
        }

        for (size_t index = 0; index < logical_id_count; ++index) {
            execution_ids[index] = unique_slots[size_t(element_unique[index])];
        }
        std::copy(logical_ids, logical_ids + logical_id_count, last_logical_ids.begin());
        std::copy(execution_ids, execution_ids + logical_id_count, last_execution_ids.begin());
        last_id_count = logical_id_count;
        last_remap_layer = binding.layer;

        const auto cold_after = cold_cache->counter_diagnostics();
        hits += cold_after.hits - cold_before.hits;
        misses += cold_after.misses - cold_before.misses;
        admissions += cold_after.admissions - cold_before.admissions;
        evictions += cold_after.evictions - cold_before.evictions;
        logical_id_total += logical_id_count;
        unique_id_total += unique_count;
        cpu_execution_lanes += logical_id_count;
        cpu_fallback_unique_keys += unique_count;
        h2d_bytes_avoided_for_current_output += cold_bundle_payload*unique_count;
        remap_checkpoints++;
        last_remap_error = llm_expert_provider_error::none;
        return llm_expert_provider_result::success();
    }

    llm_expert_provider_result cleanup_failed_slots() noexcept override {
        std::lock_guard<std::mutex> lock(mutex);
        if (active_request) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::busy);
        }
        for (auto & entry : directory_slots) {
            if (entry.state == hot_slot_state::failed) {
                if (config.cold_mode && entry.has_cold_backing) {
                    auto released = cold_cache->release(
                        { entry.cold_slot, entry.cold_generation, entry.layout_class_id },
                        llm_cold_reference_kind::hot);
                    if (!released.is_ready()) return fail(released);
                    entry.has_cold_backing = false;
                }
                entry.key = { -1, -1 };
                entry.layout_class_id = LLM_EXPERT_LAYOUT_CLASS_INVALID;
                entry.refcount = 0;
                entry.state = hot_slot_state::free;
                failed_cleanups++;
            }
        }
        if (config.cold_mode) {
            auto result = config.cpu_cold_only ? llm_expert_provider_result::success() :
                transfer_ring->cleanup_failed_lanes();
            if (result.is_ready()) result = cold_cache->cleanup_failed_slots();
            if (result.is_ready()) result = validate_tier_invariants_locked();
            if (!result.is_ready()) return fail(result);
        }
        return llm_expert_provider_result::success();
    }

    llm_expert_provider_result end_prefetch_sequence(
            uint64_t sequence_owner) noexcept override {
        std::lock_guard<std::mutex> lock(mutex);
        if (active_request) {
            return llm_expert_provider_result::failure(
                llm_expert_provider_error::busy);
        }
        if (!phase10_sequence_active ||
                phase10_sequence_owner != sequence_owner) {
            return llm_expert_provider_result::success();
        }
        return finish_phase10_sequence_locked();
    }

    llm_expert_provider_result debug_set_miss_policy_for_testing(
            llama_expert_miss_policy policy) noexcept override {
        std::lock_guard<std::mutex> lock(mutex);
        if (!config.cold_mode || active_request ||
            (policy != LLAMA_EXPERT_MISS_POLICY_PROMOTE_AND_GPU &&
             policy != LLAMA_EXPERT_MISS_POLICY_CPU_FALLBACK)) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::unsupported_configuration);
        }
        config.miss_policy = policy;
        return llm_expert_provider_result::success();
    }

    llm_expert_provider_result debug_set_host_resident_serial_issue_for_testing(
            bool serial_control) noexcept override {
        std::lock_guard<std::mutex> lock(mutex);
        if (!config.cpu_cold_only || active_request || pool || cold_cache ||
            host_resident_demand) {
            return llm_expert_provider_result::failure(
                llm_expert_provider_error::unsupported_configuration);
        }
        config.phase10_serial_issue_for_testing = serial_control;
        return llm_expert_provider_result::success();
    }

    llm_expert_provider_result debug_set_auto_cost_model_for_testing(
            const llama_expert_auto_cost_model & cost) noexcept override {
        std::lock_guard<std::mutex> lock(mutex);
        if (!config.cold_mode || active_request ||
            cost.version != LLAMA_EXPERT_AUTO_COST_MODEL_VERSION_1 ||
            cost.struct_size != sizeof(llama_expert_auto_cost_model) ||
            cost.cpu_fixed_decode_ns == 0 || cost.cpu_fixed_prefill_ns == 0 ||
            cost.cpu_per_lane_decode_ns == 0 || cost.cpu_per_lane_prefill_ns == 0 ||
            cost.gpu_fixed_decode_ns == 0 || cost.gpu_fixed_prefill_ns == 0 ||
            cost.gpu_per_lane_decode_ns == 0 || cost.gpu_per_lane_prefill_ns == 0 ||
            cost.h2d_fixed_ns == 0 || cost.h2d_bytes_per_second == 0 ||
            cost.decision_hysteresis_ns == 0) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::unsupported_configuration);
        }
        config.auto_cost_model = cost;
        const auto * bytes = reinterpret_cast<const uint8_t *>(&cost);
        config.auto_cost_model_digest = UINT64_C(1469598103934665603);
        for (size_t index = 0; index < sizeof(cost); ++index) {
            config.auto_cost_model_digest ^= bytes[index];
            config.auto_cost_model_digest *= UINT64_C(1099511628211);
        }
        config.miss_policy = LLAMA_EXPERT_MISS_POLICY_AUTO;
        return llm_expert_provider_result::success();
    }

    llm_expert_provider_result validate_slot_generation(
            uint32_t slot,
            uint64_t generation) noexcept override {
        std::lock_guard<std::mutex> lock(mutex);
        if (slot >= directory_slots.size() || directory_slots[slot].generation != generation ||
            (directory_slots[slot].state != hot_slot_state::ready &&
             directory_slots[slot].state != hot_slot_state::pinned)) {
            stale_generation_failures++;
            return fail(llm_expert_provider_result::failure(llm_expert_provider_error::stale_generation));
        }
        return llm_expert_provider_result::success();
    }

    llm_expert_provider_initialization_stage initialization_stage() const noexcept override {
        std::lock_guard<std::mutex> lock(mutex);
        if (pool != nullptr) {
            return llm_expert_provider_initialization_stage::none;
        }
        return config.cold_mode
            ? llm_expert_provider_initialization_stage::descriptors_before_scheduler_reserve
            : llm_expert_provider_initialization_stage::workspace_before_persistent_pool;
    }

    llm_expert_provider_result begin_initialization(
            llm_expert_provider_initialization_stage stage,
            bool & owner) noexcept override {
        std::lock_guard<std::mutex> lock(mutex);
        owner = false;
        if (pool != nullptr) {
            return llm_expert_provider_result::success();
        }
        if (initialization_failed) {
            return fail(llm_expert_provider_result::failure(llm_expert_provider_error::initialization_failed));
        }
        if (initialization_in_progress) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::busy);
        }
        const auto expected = config.cold_mode
            ? llm_expert_provider_initialization_stage::descriptors_before_scheduler_reserve
            : llm_expert_provider_initialization_stage::workspace_before_persistent_pool;
        if (stage != expected) {
            return fail(llm_expert_provider_result::failure(llm_expert_provider_error::invalid_descriptor));
        }
        initialization_in_progress = true;
        owner = true;
        return llm_expert_provider_result::success();
    }

    llm_expert_provider_result complete_descriptor_discovery(
            uint64_t graph_count,
            uint64_t binding_count,
            uint64_t scheduler_reserve_calls,
            const std::vector<uint64_t> & backend_bytes_before,
            const std::vector<uint64_t> & backend_bytes_after) noexcept override {
        std::lock_guard<std::mutex> lock(mutex);
        if (!config.cold_mode || !initialization_in_progress || pool != nullptr ||
            descriptor_discovery_complete || graph_count < 2 ||
            registrations.size() != config.routed_layer_count || !prototype.has_value() ||
            backend_bytes_before.size() != backend_bytes_after.size()) {
            return fail(llm_expert_provider_result::failure(llm_expert_provider_error::initialization_failed));
        }
        uint64_t before = 0;
        uint64_t after = 0;
        for (size_t index = 0; index < backend_bytes_before.size(); ++index) {
            if (backend_bytes_before[index] > UINT64_MAX - before ||
                backend_bytes_after[index] > UINT64_MAX - after) {
                return fail(llm_expert_provider_result::failure(llm_expert_provider_error::allocation_failed));
            }
            before += backend_bytes_before[index];
            after += backend_bytes_after[index];
        }
        if (graph_count > UINT64_MAX/config.routed_layer_count ||
            scheduler_reserve_calls != 0 || before != 0 || after != 0 ||
            binding_count != graph_count*config.routed_layer_count) {
            return fail(llm_expert_provider_result::failure(llm_expert_provider_error::initialization_failed));
        }
        const auto sealed = seal_layout_registry_locked();
        if (!sealed.is_ready()) return fail(sealed);
        descriptor_discovery_graphs = graph_count;
        descriptor_discovery_bindings = binding_count;
        descriptor_discovery_scheduler_reserve_calls = scheduler_reserve_calls;
        descriptor_discovery_backend_bytes_before = before;
        descriptor_discovery_backend_bytes_after = after;
        scheduler_backend_bytes_before_discovery = backend_bytes_before;
        scheduler_backend_bytes_after_discovery = backend_bytes_after;
        descriptor_discovery_complete = true;
        return llm_expert_provider_result::success();
    }

    llm_expert_provider_result record_initialization_telemetry(
            const std::vector<uint64_t> & backend_bytes_after_hierarchy,
            const std::vector<uint64_t> & backend_bytes_after_final_reserve,
            uint64_t final_source_bindings,
            uint64_t deferred_payload_bytes) noexcept override {
        std::lock_guard<std::mutex> lock(mutex);
        if (!config.cold_mode || pool == nullptr || !descriptor_discovery_complete ||
            backend_bytes_after_hierarchy.size() != backend_bytes_after_final_reserve.size() ||
            backend_bytes_after_hierarchy.size() != scheduler_backend_bytes_after_discovery.size() ||
            final_source_bindings != 0 || deferred_payload_bytes == 0) {
            return fail(llm_expert_provider_result::failure(llm_expert_provider_error::initialization_failed));
        }
        if (cold_cache == nullptr || (!config.cpu_cold_only && transfer_ring == nullptr)) {
            return fail(llm_expert_provider_result::failure(llm_expert_provider_error::initialization_failed));
        }
        const auto cold = cold_cache->diagnostics();
        if (cold.actual_bytes > config.cold_cache_bytes ||
            (!config.cpu_cold_only && device_pools.size() != transfer_rings.size()) ||
            (config.cpu_cold_only && (!device_pools.empty() || !transfer_rings.empty() ||
                                     pool->buffer != nullptr || pool->bundles.empty()))) {
            return fail(llm_expert_provider_result::failure(llm_expert_provider_error::allocation_failed));
        }
        for (size_t device = 0; device < device_pools.size(); ++device) {
            const uint32_t expected_capacity = !config.devices.empty() ?
                config.devices[device].capacity : config.capacity;
            const auto ring = transfer_rings[device]->diagnostics();
            if (ring.actual_bytes > config.transfer_ring_bytes ||
                device_pools[device]->bundles.empty() ||
                device_pools[device]->bundles.front().n_expert != int32_t(expected_capacity)) {
                return fail(llm_expert_provider_result::failure(
                    llm_expert_provider_error::allocation_failed));
            }
        }
        for (uint64_t bytes : backend_bytes_after_hierarchy) {
            if (bytes != 0) {
                return fail(llm_expert_provider_result::failure(llm_expert_provider_error::initialization_failed));
            }
        }
        uint64_t runtime_obligation = config.peer_staging_bytes;
        for (uint64_t bytes : backend_bytes_after_final_reserve) {
            if (bytes > UINT64_MAX - runtime_obligation) {
                return fail(llm_expert_provider_result::failure(
                    llm_expert_provider_error::unsupported_configuration));
            }
            runtime_obligation += bytes;
        }
        for (const auto & ring : transfer_rings) {
            const uint64_t bytes = ring ? ring->diagnostics().actual_bytes : 0;
            if (bytes > UINT64_MAX - runtime_obligation) {
                return fail(llm_expert_provider_result::failure(
                    llm_expert_provider_error::unsupported_configuration));
            }
            runtime_obligation += bytes;
        }
        const auto memory_validated = system_memory_result(
            system_memory_budget.record_runtime_obligation(runtime_obligation));
        if (!memory_validated.is_ready()) return fail(memory_validated);
        scheduler_backend_bytes_after_hierarchy = backend_bytes_after_hierarchy;
        scheduler_backend_bytes_after_final_reserve = backend_bytes_after_final_reserve;
        final_bootstrap_source_bindings = final_source_bindings;
        complete_deferred_payload_bytes = deferred_payload_bytes;
        complete_deferred_payload_in_compute_workspace = false;
        return llm_expert_provider_result::success();
    }

    llm_expert_provider_result revalidate_system_memory_budget() noexcept override {
        std::lock_guard<std::mutex> lock(mutex);
        return config.cold_mode ?
            system_memory_result(system_memory_budget.revalidate("context_sched_reserve")) :
            llm_expert_provider_result::success();
    }

    static llm_expert_provider_result system_memory_preflight_trampoline(
            void * data, uint64_t incoming_bytes) {
        auto * provider = static_cast<llm_hot_cache_expert_weight_provider *>(data);
        return system_memory_result(provider->system_memory_budget.preflight(incoming_bytes));
    }

    llm_expert_provider_result finish_initialization(bool success) noexcept override {
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialization_in_progress || (success && pool == nullptr) || (!success && pool != nullptr)) {
            return fail(llm_expert_provider_result::failure(llm_expert_provider_error::initialization_failed));
        }
        initialization_in_progress = false;
        initialization_failed = !success;
        return llm_expert_provider_result::success();
    }

    llm_expert_provider_result initialize_after_reserve() noexcept override {
        std::lock_guard<std::mutex> lock(mutex);
        auto fail_initialization = [&](const char * stage, llm_expert_provider_result result) {
            std::fprintf(stderr, "expert cache initialization failed at %s (provider error %d)\n",
                stage, int(result.error));
            return fail(result);
        };
        if (pool) {
            return llm_expert_provider_result::success();
        }
        if (config.cold_mode && initialization_in_progress && !descriptor_discovery_complete) {
            return fail(llm_expert_provider_result::failure(llm_expert_provider_error::initialization_failed));
        }
        if (!prototype.has_value() || registrations.size() != config.routed_layer_count) {
            return fail(llm_expert_provider_result::failure(llm_expert_provider_error::initialization_failed));
        }
        const auto sealed = seal_layout_registry_locked();
        if (!sealed.is_ready()) return fail(sealed);
        if (faults.fail_pool_allocation) {
            return fail(llm_expert_provider_result::failure(llm_expert_provider_error::allocation_failed));
        }

        if (config.cold_mode) {
            uint64_t slot_footprint = 0;
            auto calculated = llm_cold_expert_cache::calculate_slot_footprint(
                layout_registry, config.cpu_cold_only ? config.target_buffer_type : nullptr,
                slot_footprint);
            if (!calculated.is_ready() || slot_footprint > UINT64_MAX/config.total_expert_keys) {
                return fail_initialization("system-memory slot sizing",
                    calculated.is_ready() ? llm_expert_provider_result::failure(
                        llm_expert_provider_error::unsupported_configuration) : calculated);
            }
            const uint64_t topology_bytes = slot_footprint*config.total_expert_keys;
            uint64_t selected_pool_bytes = 0;
            const auto resolved = system_memory_result(system_memory_budget.resolve(
                requested_cold_cache_bytes, slot_footprint, topology_bytes,
                config.n_expert_used, selected_pool_bytes));
            if (!resolved.is_ready()) {
                return fail_initialization("system-memory cold-cache budget", resolved);
            }
            config.cold_cache_bytes = selected_pool_bytes;
        }

        try {
            if (config.cpu_cold_only) {
                std::vector<int32_t> policy_layers = config.routed_layers;
                if (policy_layers.empty()) {
                    for (const auto & registration : registrations) policy_layers.push_back(registration.first);
                }
                if (policy_layers.size() != config.routed_layer_count) {
                    return fail_initialization("CPU cold-only policy topology",
                        llm_expert_provider_result::failure(
                            llm_expert_provider_error::unsupported_configuration));
                }

                llm_cold_cache_config cold_config;
                cold_config.byte_budget = config.cold_cache_bytes;
                cold_config.minimum_slots = config.n_expert_used;
                cold_config.routed_layer_count = config.routed_layer_count;
                cold_config.total_expert_keys = config.total_expert_keys;
                cold_config.minimum_domain_slots = config.n_expert_used;
                cold_config.cache_policy_config = config.cold_cache_policy_config;
                cold_config.routed_layers = policy_layers;
                cold_config.policy_trace_capacity = std::max<uint32_t>(65536, config.trace_capacity);
                cold_config.buffer_type = config.target_buffer_type;
                cold_config.reclaim_free_pages = true;
                cold_config.preflight = system_memory_preflight_trampoline;
                cold_config.preflight_data = this;
                const auto memory_diagnostics = system_memory_budget.diagnostics();
                cold_config.reservation_bytes = memory_diagnostics.headroom.slot_count == 0 ?
                    0 : config.cold_cache_bytes/memory_diagnostics.headroom.slot_count;
                auto cold_candidate = std::make_unique<llm_cold_expert_cache>(std::move(cold_config));
                auto initialized = cold_candidate->initialize(layout_registry);
                if (!initialized.is_ready()) {
                    return fail_initialization("CPU cold-only cache", initialized);
                }
                const auto cold_diagnostics = cold_candidate->diagnostics();
                if (cold_diagnostics.bundle_payload_bytes == 0 ||
                    cold_diagnostics.effective_slots < config.n_expert_used ||
                    cold_diagnostics.actual_bytes > config.cold_cache_bytes) {
                    return fail_initialization("CPU cold-only cache capacity",
                        llm_expert_provider_result::failure(
                            llm_expert_provider_error::unsupported_configuration));
                }

                // The graph lease owns metadata only; the descriptors reference the cold allocation.
                auto generation_candidate = std::make_shared<hot_pool_generation>();
                generation_candidate->bundles.reserve(layout_registry.classes.size());
                for (size_t class_id = 0; class_id < layout_registry.classes.size(); ++class_id) {
                    generation_candidate->bundles.push_back(
                        cold_candidate->bundle(llm_expert_layout_class_id(class_id)));
                }
                generation_candidate->id = ++generation;

                config.routed_layers = std::move(policy_layers);
                unique_keys.resize(config.capacity);
                unique_slots.resize(config.capacity);
                unique_lane_counts.resize(config.capacity);
                policy_unique_indices.resize(config.capacity);
                cold_references.resize(config.capacity);
                cpu_execution_pins.resize(config.capacity);
                element_unique.clear();
                logical_id_scratch.clear();
                execution_id_scratch.clear();
                last_logical_ids.clear();
                last_execution_ids.clear();
                last_id_count = 0;
                last_remap_layer = -1;
                request_pin_count = 0;
                cpu_execution_pin_count = 0;
                active_request = false;
                request_ubatch_ordinal = 0;
                hot_policy_phase = llm_expert_cache_policy_phase::prefill;
                cold_bundle_payload = cold_diagnostics.bundle_payload_bytes;
                layout_preflight_consumer_count = 0;
                layout_preflight_passed = true;
                cold_cache = std::move(cold_candidate);
                if (config.storage != nullptr) {
                    host_resident_demand = std::make_unique<llm_host_resident_demand_coordinator>(
                        llm_host_resident_demand_config{
                            cold_cache.get(),
                            config.storage,
                            config.scheduler,
                            config.async_transport,
                            &layout_registry,
                            config.integrity_mode,
                            config.total_expert_keys,
                            config.capacity,
                            std::max<uint32_t>(config.trace_capacity, 1),
                            config.phase10_serial_issue_for_testing,
                            nullptr,
                            nullptr,
                            0,
                            config.phase10_serial_issue_for_testing ?
                                llm_cold_reference_kind::request :
                                llm_cold_reference_kind::batch,
                        });
                }
                pool = std::move(generation_candidate);
                epoch++;
                counters.allocations++;
                return llm_expert_provider_result::success();
            }

            std::vector<llm_hot_cache_config::device_config> pool_devices = config.devices;
            if (pool_devices.empty()) {
                pool_devices.push_back({
                    0, config.target_device, config.target_buffer_type, config.capacity, -1, {}, {},
                });
            }
            uint32_t preflight_consumer_count = 0;
            std::vector<std::shared_ptr<hot_pool_generation>> pool_candidates;
            pool_candidates.reserve(pool_devices.size());
            for (size_t device_index = 0; device_index < pool_devices.size(); ++device_index) {
                const auto & device = pool_devices[device_index];
                uint32_t device_consumers = 0;
                if (!preflight_hot_pool_consumers(layout_registry, device.capacity, config.n_expert_used,
                        device.target_device, device.target_buffer_type, device_consumers) ||
                    device_consumers > UINT32_MAX - preflight_consumer_count) {
                    return fail_initialization("hot-pool preflight", llm_expert_provider_result::failure(
                        llm_expert_provider_error::unsupported_configuration));
                }
                preflight_consumer_count += device_consumers;

                if (faults.fail_pool_allocation_device_for_testing == device_index) {
                    return fail(llm_expert_provider_result::failure(
                        llm_expert_provider_error::allocation_failed));
                }

                ggml_init_params params = {
                    /*.mem_size   =*/ ggml_tensor_overhead()*32,
                    /*.mem_buffer =*/ nullptr,
                    /*.no_alloc   =*/ true,
                };
                auto device_candidate = std::make_shared<hot_pool_generation>();
                device_candidate->ctx.reset(ggml_init(params));
                if (!device_candidate->ctx ||
                    !allocate_hot_pool(*device_candidate, layout_registry, device.capacity,
                        device.target_buffer_type) || !device_candidate->buffer ||
                    (!config.allow_non_cuda_target_for_testing &&
                     ggml_backend_buffer_is_host(device_candidate->buffer.get()))) {
                    return fail(llm_expert_provider_result::failure(
                        llm_expert_provider_error::allocation_failed));
                }
                for (size_t class_index = 0; class_index < layout_registry.classes.size(); ++class_index) {
                    if (!pool_layout_matches_source(device_candidate->bundles[class_index],
                            layout_registry.classes[class_index].prototype)) {
                        return fail(llm_expert_provider_result::failure(
                            llm_expert_provider_error::invalid_descriptor));
                    }
                }
                pool_candidates.push_back(std::move(device_candidate));
            }
            layout_preflight_consumer_count = preflight_consumer_count;
            layout_preflight_passed = true;
            auto candidate = pool_candidates.front();

            uint64_t hot_bundle_payload = 0;
            for (const auto & layout_class : layout_registry.classes) {
                hot_bundle_payload = std::max(hot_bundle_payload, layout_class.payload_bytes);
            }
            if (hot_bundle_payload == 0) return fail(
                llm_expert_provider_result::failure(llm_expert_provider_error::invalid_descriptor));
            uint64_t hot_slot_footprint = 0;
            for (const auto & device_candidate : pool_candidates) {
                hot_slot_footprint = std::max(hot_slot_footprint, device_candidate->slot_stride);
            }
            std::vector<int32_t> policy_layers = config.routed_layers;
            if (policy_layers.empty()) {
                for (const auto & registration : registrations) policy_layers.push_back(registration.first);
            }
            if (policy_layers.size() != config.routed_layer_count) {
                return fail_initialization("policy layer topology", llm_expert_provider_result::failure(
                    llm_expert_provider_error::unsupported_configuration));
            }
            llm_expert_cache_policy hot_policy_candidate;
            const uint32_t policy_trace_capacity = std::max<uint32_t>(65536, config.trace_capacity);
            auto policy_initialized = hot_policy_candidate.initialize(
                config.hot_cache_policy_config, llm_expert_cache_policy_tier::hot,
                policy_layers.data(), config.routed_layer_count, n_expert,
                config.n_expert_used, config.capacity, hot_slot_footprint, policy_trace_capacity);
            if (!policy_initialized.is_ready()) {
                return fail_initialization("hot policy", cache_policy_result(policy_initialized));
            }

            std::unique_ptr<llm_cold_expert_cache> cold_candidate;
            std::unique_ptr<llm_expert_transfer_ring> ring_candidate;
            std::vector<std::unique_ptr<llm_expert_transfer_ring>> ring_candidates;
            std::vector<uint32_t> device_transfer_lane_capacities_candidate;
            std::vector<uint8_t> device_transfer_event_capabilities_candidate;
            uint64_t cold_bundle_payload_candidate = 0;
            uint32_t transfer_lane_capacity_candidate = 0;
            if (config.cold_mode) {
                llm_cold_cache_config cold_config;
                cold_config.byte_budget = config.cold_cache_bytes;
                // PROMOTE_AND_GPU treats hot and cold as independent tiers.
                // A storage-backed hot miss bypasses durable cold admission,
                // so initialization only requires one bounded demand domain.
                cold_config.minimum_slots = uint32_t(std::min<uint64_t>(
                    config.total_expert_keys, config.n_expert_used));
                cold_config.routed_layer_count = config.routed_layer_count;
                cold_config.total_expert_keys = config.total_expert_keys;
                cold_config.minimum_domain_slots = config.n_expert_used;
                cold_config.cache_policy_config = config.cold_cache_policy_config;
                cold_config.routed_layers = policy_layers;
                cold_config.policy_trace_capacity = policy_trace_capacity;
                cold_config.reclaim_free_pages = true;
                cold_config.preflight = system_memory_preflight_trampoline;
                cold_config.preflight_data = this;
                const auto memory_diagnostics = system_memory_budget.diagnostics();
                cold_config.reservation_bytes = memory_diagnostics.headroom.slot_count == 0 ?
                    0 : config.cold_cache_bytes/memory_diagnostics.headroom.slot_count;
                cold_candidate = std::make_unique<llm_cold_expert_cache>(std::move(cold_config));
                auto initialized = cold_candidate->initialize(layout_registry);
                if (!initialized.is_ready()) {
                    return fail_initialization("shared cold cache", initialized);
                }
                ring_candidates.reserve(pool_devices.size());
                device_transfer_lane_capacities_candidate.reserve(pool_devices.size());
                device_transfer_event_capabilities_candidate.reserve(pool_devices.size());
                for (const auto & device : pool_devices) {
                    auto device_ring = std::make_unique<llm_expert_transfer_ring>(llm_transfer_ring_config {
                        config.transfer_ring_bytes,
                        2,
                        device.target_device,
                        config.allow_non_cuda_target_for_testing,
                        config.force_pageable_transfer_for_testing,
                        false,
                        0,
                        config.trace_capacity,
                    });
                    initialized = device_ring->initialize(layout_registry);
                    if (!initialized.is_ready()) {
                        return fail_initialization("device transfer ring", initialized);
                    }
                    if (config.phase8_test_control != nullptr) {
                        initialized = device_ring->set_phase8_test_control_for_testing(
                            config.phase8_test_control);
                        if (!initialized.is_ready()) return fail(initialized);
                    }
                    const auto device_ring_diagnostics = device_ring->diagnostics();
                    const uint32_t device_lane_capacity = device_ring_diagnostics.effective_lanes;
                    if (device_lane_capacity < 2) {
                        return fail_initialization("device transfer ring capacity",
                            llm_expert_provider_result::failure(
                                llm_expert_provider_error::unsupported_configuration));
                    }
                    device_transfer_lane_capacities_candidate.push_back(device_lane_capacity);
                    device_transfer_event_capabilities_candidate.push_back(
                        device_ring_diagnostics.event_capable ? 1 : 0);
                    ring_candidates.push_back(std::move(device_ring));
                }
                ring_candidate = std::move(ring_candidates.front());
                const auto cold_diagnostics = cold_candidate->diagnostics();
                cold_bundle_payload_candidate = cold_diagnostics.bundle_payload_bytes;
                transfer_lane_capacity_candidate = device_transfer_lane_capacities_candidate.front();
                if (cold_bundle_payload_candidate == 0) {
                    return fail_initialization("cold/ring capacity", llm_expert_provider_result::failure(
                        llm_expert_provider_error::unsupported_configuration));
                }
            }

            std::vector<hot_forward_entry> directory_forward_candidate(
                size_t(LLAMA_MAX_LAYERS)*n_expert*pool_devices.size());
            std::vector<hot_slot_entry> directory_slots_candidate(config.capacity);
            uint32_t global_slot = 0;
            for (const auto & device : pool_devices) {
                for (uint32_t device_slot = 0; device_slot < device.capacity; ++device_slot) {
                    auto & entry = directory_slots_candidate[global_slot++];
                    entry.generation = config.initial_slot_generation_for_testing;
                    entry.device_id = device.device_id;
                    entry.device_slot = device_slot;
                }
            }
            uint64_t seed_use_clock = 0;
            const bool blocking_seed = config.prefetch_config.supplied &&
                config.prefetch_config.value.seed_mode == LLAMA_EXPERT_PREFETCH_SEED_MODE_BLOCKING_HOT;
            if (blocking_seed) {
                phase10_seed_attempts++;
                const auto & seeds = config.prefetch_profile.seed;
                uint64_t seed_physical_bytes = 0;
                bool seed_fits = !seeds.empty() && seeds.size() <= config.capacity &&
                    hot_bundle_payload != 0;
                for (const auto & seed : seeds) {
                    const llm_expert_key seed_key = { seed.layer, seed.expert };
                    seed_fits = seed_fits && seed.layer >= 0 && seed.layer < LLAMA_MAX_LAYERS &&
                        seed.expert >= 0 && seed.expert < int32_t(n_expert) &&
                        seed.payload_bytes == payload_for_key(seed_key) &&
                        seed.physical_bytes <= UINT64_MAX - seed_physical_bytes;
                    if (!seed_fits) break;
                    seed_physical_bytes += seed.physical_bytes;
                }
                if (config.cold_mode) {
                    const auto cold = cold_candidate->diagnostics();
                    const auto scheduler = config.scheduler ?
                        config.scheduler->diagnostics() : llm_expert_scheduler_diagnostics{};
                    seed_fits = seed_fits && seeds.size() <= cold.effective_slots &&
                        seeds.size() <= transfer_lane_capacity_candidate &&
                        seeds.size() <= scheduler.request_capacity &&
                        scheduler.active_requests == 0 && !scheduler.admission_closed &&
                        seed_physical_bytes <= config.cold_cache_bytes &&
                        config.storage != nullptr && config.async_transport != nullptr &&
                        config.scheduler != nullptr;
                }
                auto seed_result = seed_fits ? llm_expert_provider_result::success() :
                    llm_expert_provider_result::failure(llm_expert_provider_error::unsupported_configuration);
                std::vector<llm_expert_cache_policy_candidate> seed_policy_candidates(config.capacity);
                std::vector<llm_cold_reference> seed_cold(seeds.size());
                size_t seed_tensor_copies = 0;
                size_t seed_copy_bytes = 0;
                if (seed_result.is_ready()) {
                    seed_result = cache_policy_result(hot_policy_candidate.request_begin());
                }
                storage_load_context storage_context = {
                    config.storage, config.async_transport, config.scheduler,
                    config.integrity_mode, nullptr, {}, false, nullptr, nullptr,
                };
                storage_context.layer_ids = &layout_registry.layer_ids;
                storage_context.layout_registry = &layout_registry;
                for (size_t index = 0; index < seeds.size() && seed_result.is_ready(); ++index) {
                    const auto & seed = seeds[index];
                    const llm_expert_key key = { seed.layer, seed.expert };
                    for (uint32_t slot = 0; slot < directory_slots_candidate.size(); ++slot) {
                        const auto & entry = directory_slots_candidate[slot];
                        seed_policy_candidates[slot] = {
                            slot, entry.generation, policy_key_for(entry.key),
                            entry.state == hot_slot_state::free ? payload_for_key(key) : payload_for_key(entry.key),
                            hot_slot_footprint,
                            entry.state == hot_slot_state::free, false,
                        };
                    }
                    llm_expert_cache_policy_decision decision;
                    seed_result = cache_policy_result(hot_policy_candidate.select(
                        policy_key_for(key), seed_policy_candidates.data(),
                        seed_policy_candidates.size(), decision));
                    if (!seed_result.is_ready()) break;
                    if (!decision.accept || !decision.free || decision.slot >= directory_slots_candidate.size()) {
                        seed_result = llm_expert_provider_result::failure(
                            llm_expert_provider_error::unsupported_configuration);
                        break;
                    }
                    auto & entry = directory_slots_candidate[decision.slot];
                    if (entry.state != hot_slot_state::free || entry.generation == UINT64_MAX) {
                        seed_result = llm_expert_provider_result::failure(
                            llm_expert_provider_error::generation_exhausted);
                        break;
                    }
                    entry.key = key;
                    entry.layout_class_id = layout_class_for_layer(key.layer);
                    entry.generation++;
                    entry.origin = llm_expert_residency_origin::static_seed;
                    entry.state = hot_slot_state::loading;
                    seed_result = cache_policy_result(hot_policy_candidate.load_begin(
                        decision.slot, entry.generation, policy_key_for(key),
                        payload_for_key(key), hot_slot_footprint, false));
                    if (!seed_result.is_ready()) break;

                    if (config.cold_mode) {
                        seed_result = cold_candidate->find_or_admit_with_loader(
                            key, seed_cold[index], load_storage_bundle, &storage_context);
                        if (storage_context.completed_io_pending_publication) {
                            const auto handle = storage_context.completed_io_handle;
                            const auto next = seed_result.is_ready() ?
                                llm_expert_request_state::host_ready :
                                llm_expert_request_state::draining;
                            const auto transitioned = config.scheduler->transition(
                                handle, llm_expert_request_state::io_in_flight, next);
                            const auto terminal = seed_result.is_ready() ?
                                llm_expert_request_state::complete :
                                llm_expert_request_state::failed;
                            const auto finished = transitioned == llm_expert_schedule_disposition::admitted ?
                                config.scheduler->finish(handle, terminal) :
                                llm_expert_schedule_disposition::invalid;
                            const auto released = finished == llm_expert_schedule_disposition::admitted ?
                                config.scheduler->release_terminal(handle) :
                                llm_expert_schedule_disposition::invalid;
                            storage_context.completed_io_pending_publication = false;
                            storage_context.completed_io_handle = {};
                            if ((transitioned != llm_expert_schedule_disposition::admitted ||
                                 finished != llm_expert_schedule_disposition::admitted ||
                                 released != llm_expert_schedule_disposition::admitted) &&
                                seed_result.is_ready()) {
                                seed_result = llm_expert_provider_result::failure(
                                    llm_expert_provider_error::metadata_mismatch);
                            }
                        }
                        if (seed_result.is_ready()) {
                            seed_result = cold_candidate->reclassify(
                                seed_cold[index], llm_expert_residency_origin::static_seed);
                        }
                        if (seed_result.is_ready()) {
                            entry.layout_class_id = seed_cold[index].layout_class_id;
                            llm_transfer_lane_reference lane;
                            seed_result = ring_candidate->reserve(*cold_candidate, seed_cold[index],
                                decision.slot, entry.generation, lane);
                            if (seed_result.is_ready()) {
                                seed_result = ring_candidate->stage(lane,
                                    cold_candidate->bundle(seed_cold[index].layout_class_id));
                            }
                            if (seed_result.is_ready()) {
                                std::vector<llm_transfer_binding> binding = {
                                    { lane, candidate->bundles[seed_cold[index].layout_class_id], decision.slot },
                                };
                                seed_result = ring_candidate->transfer_wave_blocking(binding);
                            }
                        }
                    } else {
                        const auto registration = registrations.find(key.layer);
                        bool copied = registration != registrations.end();
                        if (copied) {
                            const auto & source_bundle = registration->second;
                            const auto & hot_bundle = candidate->bundles[entry.layout_class_id];
                            copied = copy_expert_projection(hot_bundle.up, source_bundle.up,
                                         source_bundle.n_expert, config.capacity, key.expert, decision.slot,
                                         seed_copy_bytes, seed_tensor_copies, faults.fail_copy_after_tensors) &&
                                copy_expert_projection(hot_bundle.gate, source_bundle.gate,
                                         source_bundle.n_expert, config.capacity, key.expert, decision.slot,
                                         seed_copy_bytes, seed_tensor_copies, faults.fail_copy_after_tensors) &&
                                copy_expert_projection(hot_bundle.gate_up, source_bundle.gate_up,
                                         source_bundle.n_expert, config.capacity, key.expert, decision.slot,
                                         seed_copy_bytes, seed_tensor_copies, faults.fail_copy_after_tensors) &&
                                copy_expert_projection(hot_bundle.down, source_bundle.down,
                                         source_bundle.n_expert, config.capacity, key.expert, decision.slot,
                                         seed_copy_bytes, seed_tensor_copies, faults.fail_copy_after_tensors);
                        }
                        if (!copied) {
                            seed_result = llm_expert_provider_result::failure(
                                llm_expert_provider_error::copy_failed);
                        }
                    }
                    if (!seed_result.is_ready()) break;
                    seed_result = cache_policy_result(
                        hot_policy_candidate.load_complete(decision.slot, entry.generation));
                    if (!seed_result.is_ready()) break;
                    entry.state = hot_slot_state::ready;
                    entry.last_use = ++seed_use_clock;
                    directory_forward_candidate[size_t(key.layer)*n_expert + uint32_t(key.expert)] = {
                        int32_t(decision.slot), entry.generation,
                    };
                    seed_result = cache_policy_result(hot_policy_candidate.demand(
                        policy_key_for(key), 1, payload_for_key(key), hot_slot_footprint));
                    if (seed_result.is_ready()) {
                        seed_result = cache_policy_result(
                            hot_policy_candidate.hit(decision.slot, entry.generation));
                    }
                }
                llm_expert_key last_touch = { -1, -1 };
                if (seed_result.is_ready()) {
                    const auto highest = std::min_element(seeds.begin(), seeds.end(), [](const auto & lhs, const auto & rhs) {
                        return lhs.count != rhs.count ? lhs.count > rhs.count :
                            (lhs.layer != rhs.layer ? lhs.layer < rhs.layer : lhs.expert < rhs.expert);
                    });
                    last_touch = { highest->layer, highest->expert };
                    const auto & forward = directory_forward_candidate[
                        size_t(highest->layer)*n_expert + uint32_t(highest->expert)];
                    if (forward.slot < 0) {
                        seed_result = llm_expert_provider_result::failure(
                            llm_expert_provider_error::metadata_mismatch);
                    } else {
                        auto & entry = directory_slots_candidate[forward.slot];
                        seed_result = cache_policy_result(hot_policy_candidate.demand(
                            policy_key_for(last_touch), 1,
                            payload_for_key(last_touch), hot_slot_footprint));
                        if (seed_result.is_ready()) {
                            seed_result = cache_policy_result(
                                hot_policy_candidate.hit(uint32_t(forward.slot), entry.generation));
                        }
                        if (seed_result.is_ready() && config.cold_mode) {
                            size_t seed_index = 0;
                            while (seed_index < seeds.size() &&
                                (seeds[seed_index].layer != last_touch.layer ||
                                 seeds[seed_index].expert != last_touch.expert)) seed_index++;
                            if (seed_index == seeds.size()) {
                                seed_result = llm_expert_provider_result::failure(
                                    llm_expert_provider_error::metadata_mismatch);
                            } else {
                                seed_result = cold_candidate->policy_shadow_hit(
                                    last_touch, seed_cold[seed_index], 1);
                            }
                        }
                        if (seed_result.is_ready()) entry.last_use = ++seed_use_clock;
                    }
                }
                if (seed_result.is_ready()) {
                    seed_result = cache_policy_result(hot_policy_candidate.request_end(true, false));
                }
                if (config.cold_mode) {
                    const auto ended = cold_candidate->policy_request_end(seed_result.is_ready(), false);
                    if (seed_result.is_ready()) seed_result = ended;
                }
                if (!seed_result.is_ready()) {
                    phase10_seed_failures++;
                    return fail(seed_result);
                }
                counters.tensor_copies += seed_tensor_copies;
                phase10_seed_complete = true;
                phase10_seed_entries = seeds.size();
                phase10_seed_storage_bytes = config.cold_mode ? seed_physical_bytes : 0;
                phase10_seed_h2d_bytes = config.cold_mode ? hot_bundle_payload*seeds.size() : seed_copy_bytes;
                phase10_seed_last_touch = last_touch;
            }

            config.routed_layers = policy_layers;
            directory_forward = std::move(directory_forward_candidate);
            directory_slots = std::move(directory_slots_candidate);
            unique_keys.resize(config.capacity);
            unique_slots.resize(config.capacity);
            unique_cpu_slots.resize(config.capacity);
            unique_lane_counts.resize(config.capacity);
            unique_gpu_assignment.resize(config.capacity);
            gpu_unique_indices.resize(config.capacity);
            miss_unique_indices.resize(config.capacity);
            candidate_slots.resize(config.capacity);
            policy_candidate_slots.resize(config.capacity);
            policy_unique_indices.resize(config.capacity);
            slot_selected.resize(config.capacity);
            cold_references.resize(config.capacity);
            transfer_lanes.resize(config.capacity);
            async_flights.resize(config.capacity);
            async_handles.resize(config.capacity);
            transfer_bindings.clear();
            transfer_bindings.reserve(config.capacity);
            hot_backing_scratch.clear();
            hot_backing_scratch.reserve(config.capacity);
            request_pins.resize(config.capacity);
            cpu_execution_pins.resize(config.capacity);
            background_promotions.assign(config.capacity, {});
            predictive_storage_records.assign(
                config.prefetch_config.supplied ?
                    config.prefetch_config.value.max_speculative_flights : 0, {});
            phase10_prediction_trace.assign(
                config.prefetch_config.supplied &&
                        config.prefetch_config.value.policy != LLAMA_EXPERT_PREFETCH_POLICY_OFF ?
                    config.trace_capacity : 0, {});
            phase10_route_trace.assign(
                config.prefetch_config.supplied &&
                        config.prefetch_config.value.policy != LLAMA_EXPERT_PREFETCH_POLICY_OFF ?
                    config.trace_capacity : 0, {});
            phase10_route_ids.assign(
                config.prefetch_config.supplied &&
                        config.prefetch_config.value.policy != LLAMA_EXPERT_PREFETCH_POLICY_OFF ?
                    size_t(config.trace_capacity)*config.n_expert_used : 0, -1);
            predictor_candidates.clear();
            predictor_candidates.reserve(n_expert);
            current_token_routes.assign(config.routed_layer_count, {});
            for (auto & route : current_token_routes) route.reserve(config.n_expert_used);
            auto_decisions.assign(config.trace_capacity, {});
            auto_decision_write = 0;
            element_unique.clear();
            logical_id_scratch.clear();
            execution_id_scratch.clear();
            cpu_execution_id_scratch.clear();
            last_logical_ids.clear();
            last_execution_ids.clear();
            last_cpu_execution_ids.clear();
            last_id_count = 0;
            last_remap_layer = -1;
            request_pin_count = 0;
            cpu_execution_pin_count = 0;
            active_request = false;

            hot_policy = std::move(hot_policy_candidate);
            hot_policy_phase = llm_expert_cache_policy_phase::prefill;
            hot_logical_bundle_bytes = hot_bundle_payload;
            hot_physical_slot_footprint_bytes = hot_slot_footprint;
            cold_bundle_payload = cold_bundle_payload_candidate;
            transfer_lane_capacity = transfer_lane_capacity_candidate;
            device_transfer_lane_capacities = std::move(device_transfer_lane_capacities_candidate);
            device_transfer_event_capabilities = std::move(
                device_transfer_event_capabilities_candidate);
            use_clock = seed_use_clock;

            for (auto & device_pool : pool_candidates) device_pool->id = ++generation;
            device_pools = std::move(pool_candidates);
            device_runtime_stats.assign(device_pools.size(), {});
            GGML_ASSERT(device_transfer_delay_us.size() == device_pools.size());
            GGML_ASSERT(device_transfer_failure_for_testing.size() == device_pools.size());
            GGML_ASSERT(device_transfer_failure_decode_only_for_testing.size() == device_pools.size());
            pool = device_pools.front();
            cold_cache = std::move(cold_candidate);
            if (config.cold_mode) {
                transfer_rings.clear();
                transfer_rings.reserve(pool_devices.size());
                transfer_rings.push_back(std::move(ring_candidate));
                for (size_t index = 1; index < ring_candidates.size(); ++index) {
                    transfer_rings.push_back(std::move(ring_candidates[index]));
                }
                transfer_ring = transfer_rings.front().get();
            }
            if (llm_perfetto_trace_is_active()) {
                for (uint32_t slot = 0; slot < directory_slots.size(); ++slot) {
                    const auto & entry = directory_slots[slot];
                    if (entry.state != hot_slot_state::ready) continue;
                    LLM_EXPERT_TRACE_INSTANT("k3.cache.hot", "admission",
                        "layer", entry.key.layer, "original_expert_id", entry.key.expert,
                        "slot_id", slot, "generation", entry.generation);
                }
            }
            LLM_EXPERT_TRACE_COUNTER("k3.resource", "hot_cache_occupancy", 4,
                hot_cache_occupancy_locked());
            epoch++;
            counters.allocations++;
            counters.pool_generations++;
            return llm_expert_provider_result::success();
        } catch (const std::bad_alloc &) {
            return fail(llm_expert_provider_result::failure(llm_expert_provider_error::allocation_failed));
        } catch (...) {
            return fail(llm_expert_provider_result::failure(llm_expert_provider_error::initialization_failed));
        }
    }

    llm_expert_provider_result trim() noexcept override {
        LLM_EXPERT_TRACE_SCOPE("k3.lifecycle", "provider_trim");
        std::lock_guard<std::mutex> lock(mutex);
        if (!pool) {
            counters.trims++;
            return llm_expert_provider_result::success();
        }
        if (config.cold_mode && (active_request || active_background_flights != 0)) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::busy);
        }
        if (config.cold_mode) {
            auto invariant = validate_tier_invariants_locked();
            if (!invariant.is_ready()) return fail(invariant);
        }
        for (uint32_t slot = 0; slot < directory_slots.size(); ++slot) {
            auto & entry = directory_slots[slot];
            if (entry.state == hot_slot_state::ready && entry.refcount == 0) {
                LLM_EXPERT_TRACE_INSTANT("k3.cache.hot", "victim", "slot_id", slot,
                    "generation", entry.generation, "layer", entry.key.layer,
                    "original_expert_id", entry.key.expert);
                const auto policy_valid = cache_policy_result(hot_policy.validate_evictable(slot, entry.generation));
                if (!policy_valid.is_ready()) return fail(policy_valid);
                if (entry.origin == llm_expert_residency_origin::speculative &&
                    !entry.background_useful) background_wasted++;
                if (config.cold_mode && entry.has_cold_backing) {
                    auto * owner_ring = ring_for_slot(slot);
                    if (owner_ring == nullptr) {
                        return fail(llm_expert_provider_result::failure(
                            llm_expert_provider_error::metadata_mismatch));
                    }
                    auto retired = owner_ring->retire_hot(entry.device_slot, entry.generation);
                    if (!retired.is_ready()) return fail(retired);
                    auto released = cold_cache->release(
                        { entry.cold_slot, entry.cold_generation, entry.layout_class_id },
                        llm_cold_reference_kind::hot);
                    if (!released.is_ready()) return fail(released);
                    entry.has_cold_backing = false;
                    entry.cold_slot = 0;
                    entry.cold_generation = 0;
                }
                const auto policy_removed = cache_policy_result(hot_policy.remove_resident(slot, entry.generation));
                if (!policy_removed.is_ready()) return fail(policy_removed);
                clear_forward_locked(entry.key, slot, entry.generation);
                LLM_EXPERT_TRACE_INSTANT("k3.cache.hot", "eviction", "slot_id", slot,
                    "generation", entry.generation, "layer", entry.key.layer,
                    "original_expert_id", entry.key.expert);
                entry.key = { -1, -1 };
                entry.layout_class_id = LLM_EXPERT_LAYOUT_CLASS_INVALID;
                entry.state = hot_slot_state::free;
                LLM_EXPERT_TRACE_COUNTER("k3.resource", "hot_cache_occupancy", 4,
                    hot_cache_occupancy_locked());
            }
        }
        if (config.cold_mode) {
            auto result = cold_cache->trim();
            if (result.is_ready()) result = validate_tier_invariants_locked();
            if (!result.is_ready()) return fail(result);
        }
        counters.trims++;
        return llm_expert_provider_result::success();
    }

    llm_expert_provider_result surrender() noexcept override {
        LLM_EXPERT_TRACE_SCOPE("k3.lifecycle", "provider_surrender");
        std::lock_guard<std::mutex> lock(mutex);
        if (!pool) {
            return llm_expert_provider_result::success();
        }
        bool pool_busy = pool.use_count() != (device_pools.empty() ? 1 : 2);
        for (size_t index = 1; index < device_pools.size(); ++index) {
            pool_busy = pool_busy || device_pools[index].use_count() != 1;
        }
        if (active_request || active_background_flights != 0 || pool_busy) {
            counters.surrender_busy++;
            return llm_expert_provider_result::failure(llm_expert_provider_error::busy);
        }
        if (config.cold_mode) {
            auto result = validate_tier_invariants_locked();
            if (!result.is_ready()) return fail(result);
            for (auto & ring : transfer_rings) {
                result = ring->surrender();
                if (!result.is_ready()) {
                    counters.surrender_busy++;
                    return result;
                }
            }
            for (auto & entry : directory_slots) {
                if (entry.origin == llm_expert_residency_origin::speculative &&
                    !entry.background_useful) background_wasted++;
                if (entry.has_cold_backing) {
                    result = cold_cache->release(
                        { entry.cold_slot, entry.cold_generation, entry.layout_class_id },
                        llm_cold_reference_kind::hot);
                    if (!result.is_ready()) return fail(result);
                    entry.has_cold_backing = false;
                }
            }
            result = cold_cache->surrender();
            if (!result.is_ready()) return fail(result);
        }
        for (uint32_t slot = 0; slot < directory_slots.size(); ++slot) {
            auto & entry = directory_slots[slot];
            if (entry.state == hot_slot_state::loading) {
                auto result = cache_policy_result(hot_policy.load_failed(slot, entry.generation));
                if (!result.is_ready()) return fail(result);
            } else if (entry.state == hot_slot_state::ready || entry.state == hot_slot_state::pinned) {
                LLM_EXPERT_TRACE_INSTANT("k3.cache.hot", "victim", "slot_id", slot,
                    "generation", entry.generation, "layer", entry.key.layer,
                    "original_expert_id", entry.key.expert);
                auto result = cache_policy_result(hot_policy.remove_resident(slot, entry.generation));
                if (!result.is_ready()) return fail(result);
                LLM_EXPERT_TRACE_INSTANT("k3.cache.hot", "eviction", "slot_id", slot,
                    "generation", entry.generation, "layer", entry.key.layer,
                    "original_expert_id", entry.key.expert);
            }
        }
        if (!config.cpu_cold_only) {
            auto policy_surrendered = cache_policy_result(hot_policy.surrender());
            if (!policy_surrendered.is_ready()) return fail(policy_surrendered);
        }
        pool.reset();
        device_pools.clear();
        cold_cache.reset();
        transfer_ring = nullptr;
        transfer_rings.clear();
        device_transfer_lane_capacities.clear();
        device_transfer_event_capabilities.clear();
        directory_forward.clear();
        directory_slots.clear();
        LLM_EXPERT_TRACE_COUNTER("k3.resource", "hot_cache_occupancy", 4, 0);
        unique_keys.clear();
        unique_slots.clear();
        unique_cpu_slots.clear();
        unique_lane_counts.clear();
        unique_gpu_assignment.clear();
        gpu_unique_indices.clear();
        miss_unique_indices.clear();
        candidate_slots.clear();
        policy_candidate_slots.clear();
        policy_unique_indices.clear();
        slot_selected.clear();
        cold_references.clear();
        transfer_lanes.clear();
        async_flights.clear();
        async_handles.clear();
        transfer_bindings.clear();
        hot_backing_scratch.clear();
        element_unique.clear();
        logical_id_scratch.clear();
        execution_id_scratch.clear();
        cpu_execution_id_scratch.clear();
        last_logical_ids.clear();
        last_execution_ids.clear();
        last_cpu_execution_ids.clear();
        last_id_count = 0;
        last_remap_layer = -1;
        request_pins.clear();
        cpu_execution_pins.clear();
        background_promotions.clear();
        auto_decisions.clear();
        request_pin_count = 0;
        cpu_execution_pin_count = 0;
        cold_bundle_payload = 0;
        transfer_lane_capacity = 0;
        descriptor_discovery_complete = false;
        descriptor_discovery_graphs = 0;
        descriptor_discovery_bindings = 0;
        descriptor_discovery_scheduler_reserve_calls = 0;
        descriptor_discovery_backend_bytes_before = 0;
        descriptor_discovery_backend_bytes_after = 0;
        final_bootstrap_source_bindings = 0;
        complete_deferred_payload_bytes = 0;
        complete_deferred_payload_in_compute_workspace = false;
        scheduler_backend_bytes_before_discovery.clear();
        scheduler_backend_bytes_after_discovery.clear();
        scheduler_backend_bytes_after_hierarchy.clear();
        scheduler_backend_bytes_after_final_reserve.clear();
        epoch++;
        counters.surrender_successes++;
        return llm_expert_provider_result::success();
    }

    uint64_t graph_epoch() const noexcept override {
        std::lock_guard<std::mutex> lock(mutex);
        return epoch;
    }

    bool supports_route_service_tier_snapshot() const noexcept override {
        return true;
    }

    llm_expert_provider_result route_service_tier_snapshot(
            int32_t layer,
            const int32_t * experts,
            size_t expert_count,
            llama_route_service_tier * tiers) const noexcept override {
        std::lock_guard<std::mutex> lock(mutex);
        if (experts == nullptr || tiers == nullptr || expert_count == 0 ||
            !pool || (!config.cpu_cold_only && directory_forward.empty()) || layer < 0 ||
            size_t(layer) >= layout_registry.layer_ids.size() ||
            layout_class_for_layer(layer) == LLM_EXPERT_LAYOUT_CLASS_INVALID) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::stale_generation);
        }
        for (size_t index = 0; index < expert_count; ++index) {
            const llm_expert_key key = { layer, experts[index] };
            if (!key.is_valid(LLAMA_MAX_LAYERS, n_expert)) {
                return llm_expert_provider_result::failure(llm_expert_provider_error::invalid_key);
            }
            if (!config.cpu_cold_only &&
                forward_entry_matches(key, directory_forward[forward_index(key)])) {
                tiers[index] = LLAMA_ROUTE_SERVICE_TIER_HOT;
            } else if (cold_cache != nullptr && cold_cache->contains_ready(key)) {
                tiers[index] = LLAMA_ROUTE_SERVICE_TIER_COLD;
            } else {
                tiers[index] = LLAMA_ROUTE_SERVICE_TIER_BACKING;
            }
        }
        return llm_expert_provider_result::success();
    }

    llm_hot_cache_diagnostics hot_cache_diagnostics() const override {
        std::lock_guard<std::mutex> lock(mutex);
        llm_hot_cache_diagnostics result;
        if (config.cold_mode) {
            copy_system_memory_diagnostics(system_memory_budget.diagnostics(), result);
        }
        result.configured_miss_policy = config.miss_policy;
        result.cpu_cold_only = config.cpu_cold_only;
        result.background_promotion_configured = config.background_promotion;
        result.async_cold_fill_configured = config.async_cold_fill;
        result.auto_cost_model_version = config.auto_cost_model.version;
        result.auto_cost_model_digest = config.auto_cost_model_digest;
        result.hybrid_bindings = hybrid_bindings;
        result.remote_single_bindings = remote_single_bindings;
        result.multi_device_bindings = multi_device_bindings;
        result.device_binding_vector_elements = device_binding_vector_elements;
        result.gpu_execution_lanes = gpu_execution_lanes;
        result.cpu_execution_lanes = cpu_execution_lanes;
        result.mixed_execution_layers = mixed_execution_layers;
        result.cpu_fallback_unique_keys = cpu_fallback_unique_keys;
        result.h2d_bytes_avoided_for_current_output = h2d_bytes_avoided_for_current_output;
        result.auto_cpu_decisions = auto_cpu_decisions;
        result.auto_gpu_decisions = auto_gpu_decisions;
        result.auto_tie_decisions = auto_tie_decisions;
        result.auto_overflow_decisions = auto_overflow_decisions;
        result.auto_decision_records = auto_decision_records;
        result.auto_decision_records_dropped = auto_decision_records_dropped;
        result.auto_decision_digest = auto_decision_digest;
        const size_t retained_auto_decisions = std::min<size_t>(auto_decision_records, auto_decisions.size());
        result.auto_decisions.reserve(retained_auto_decisions);
        const size_t oldest_auto_decision = retained_auto_decisions == 0 ||
            auto_decision_records <= auto_decisions.size() ? 0 : auto_decision_write%auto_decisions.size();
        for (size_t offset = 0; offset < retained_auto_decisions; ++offset) {
            const auto & record = auto_decisions[(oldest_auto_decision + offset)%auto_decisions.size()];
            result.auto_decisions.push_back({
                record.request,
                record.layer,
                record.expert,
                record.input.cost,
                record.input.prefill,
                record.input.lanes,
                record.input.bundle_bytes,
                record.input.queued_cpu_work_ns,
                record.input.queued_h2d_work_ns,
                record.input.queued_gpu_work_ns,
                record.input.same_key_h2d_present,
                uint8_t(record.input.same_key_h2d_state),
                record.input.same_key_h2d_remaining_bytes,
                record.result.cpu_work_ns,
                record.result.h2d_work_ns,
                record.result.gpu_work_ns,
                record.result.cpu_finish_ns,
                record.result.gpu_finish_ns,
                uint8_t(record.result.backend),
                uint8_t(record.result.reason),
                record.result.overflow,
            });
        }
        result.background_submitted = background_submitted;
        result.background_completed = background_completed;
        result.background_useful = background_useful;
        result.background_wasted = background_wasted;
        result.background_dropped = background_dropped;
        result.background_busy = background_busy;
        result.background_later_joins = background_later_joins;
        result.background_h2d_bytes = background_h2d_bytes;
        result.active_background_flights = active_background_flights;
        result.peak_background_flights = peak_background_flights;
        result.phase10_prefetch_configured = config.prefetch_config.supplied &&
            config.prefetch_config.value.policy != LLAMA_EXPERT_PREFETCH_POLICY_OFF;
        result.phase10_seed_configured = config.prefetch_config.supplied &&
            config.prefetch_config.value.seed_mode == LLAMA_EXPERT_PREFETCH_SEED_MODE_BLOCKING_HOT;
        result.phase10_seed_complete = phase10_seed_complete;
        result.phase10_seed_attempts = phase10_seed_attempts;
        result.phase10_seed_failures = phase10_seed_failures;
        result.phase10_seed_entries = phase10_seed_entries;
        result.phase10_seed_storage_bytes = phase10_seed_storage_bytes;
        result.phase10_seed_h2d_bytes = phase10_seed_h2d_bytes;
        result.phase10_seed_last_touch = phase10_seed_last_touch;
        result.phase10_issue_ahead_events = phase10_issue_ahead_events;
        result.phase10_issue_ahead_violations = phase10_issue_ahead_violations;
        result.phase10_prediction_events = phase10_prediction_events;
        result.phase10_prediction_events_dropped = phase10_prediction_events_dropped;
        result.phase10_route_events = phase10_route_events;
        result.phase10_route_events_dropped = phase10_route_events_dropped;
        result.phase10_predictions_admitted = phase10_predictions_admitted;
        result.phase10_predictions_rejected = phase10_predictions_rejected;
        result.phase10_timely_useful = phase10_timely_useful;
        result.phase10_late_joined = phase10_late_joined;
        result.phase10_wasted_unused = phase10_wasted_unused;
        result.phase10_cancelled_before_io = phase10_cancelled_before_io;
        result.phase10_cancelled_drained = phase10_cancelled_drained;
        result.phase10_predictor_compute_ns = phase10_predictor_compute_ns;
        result.phase10_predictor_digest = phase10_predictor_digest;
        result.phase10_predictor_state_digest = config.prefetch_config.supplied &&
                config.prefetch_config.value.policy != LLAMA_EXPERT_PREFETCH_POLICY_OFF ?
            prefetch_predictor.state().digest : 1469598103934665603ULL;
        result.phase10_circuit_opens = phase10_circuit_opens;
        result.phase10_circuit_open = phase10_circuit_open;
        result.phase10_runtime_failed = phase10_runtime_failed;
        result.requested_capacity = config.cpu_cold_only ? 0 : config.capacity;
        result.effective_capacity = config.cpu_cold_only ? 0 : (pool ? config.capacity : 0);
        for (const auto & device_pool : device_pools) {
            if (device_pool && device_pool->buffer) {
                result.pool_bytes += ggml_backend_buffer_get_size(device_pool->buffer.get());
            }
        }
        result.layout_class_count = uint32_t(layout_registry.classes.size());
        result.layout_registry_administration_bytes = sizeof(layout_registry) +
            layout_registry.classes.capacity()*sizeof(llm_expert_layout_class_descriptor) +
            layout_registry.layer_ids.capacity()*sizeof(llm_expert_layout_class_id);
        result.layout_preflight_consumer_count = layout_preflight_consumer_count;
        result.layout_preflight_passed = layout_preflight_passed;
        result.hot_slot_stride = pool ? pool->slot_stride : 0;
        result.layout_layer_ids = layout_registry.layer_ids;
        result.layout_class_digests.reserve(layout_registry.classes.size());
        result.layout_class_payload_bytes.reserve(layout_registry.classes.size());
        result.layout_class_hot_padding_bytes.reserve(layout_registry.classes.size());
        for (const auto & layout_class : layout_registry.classes) {
            result.layout_class_digests.push_back(layout_class.canonical_digest);
            result.layout_class_payload_bytes.push_back(layout_class.payload_bytes);
            result.layout_class_hot_padding_bytes.push_back(layout_registry.classes.size() == 1 ? 0 :
                pool && pool->slot_stride >= layout_class.payload_bytes ?
                pool->slot_stride - layout_class.payload_bytes : 0);
        }
        result.descriptor_discovery_graphs = descriptor_discovery_graphs;
        result.descriptor_discovery_bindings = descriptor_discovery_bindings;
        result.descriptor_discovery_scheduler_reserve_calls = descriptor_discovery_scheduler_reserve_calls;
        result.descriptor_discovery_backend_bytes_before = descriptor_discovery_backend_bytes_before;
        result.descriptor_discovery_backend_bytes_after = descriptor_discovery_backend_bytes_after;
        result.final_bootstrap_source_bindings = final_bootstrap_source_bindings;
        result.complete_deferred_payload_bytes = complete_deferred_payload_bytes;
        result.complete_deferred_payload_in_compute_workspace = complete_deferred_payload_in_compute_workspace;
        result.scheduler_backend_bytes_before_discovery = scheduler_backend_bytes_before_discovery;
        result.scheduler_backend_bytes_after_discovery = scheduler_backend_bytes_after_discovery;
        result.scheduler_backend_bytes_after_hierarchy = scheduler_backend_bytes_after_hierarchy;
        result.scheduler_backend_bytes_after_final_reserve = scheduler_backend_bytes_after_final_reserve;
        result.graph_epoch = epoch;
        result.generation = pool ? pool->id : 0;
        result.n_expert = n_expert;
        result.n_expert_used = config.n_expert_used;
        result.last_context_n_ctx = last_context_n_ctx;
        result.last_context_n_ubatch = last_context_n_ubatch;
        result.last_context_extent = last_context_extent;
        result.conservative_required_capacity = last_required_capacity;
        result.context_validations = context_validations;
        result.context_rejections = context_rejections;
        result.requests = requests;
        result.exclusive_busy_failures = exclusive_busy_failures;
        result.remap_checkpoints = remap_checkpoints;
        result.logical_ids = logical_id_total;
        result.unique_ids = unique_id_total;
        result.hits = hits;
        result.misses = misses;
        result.admissions = admissions;
        result.evictions = evictions;
        result.no_writeback_evictions = no_writeback_evictions;
        result.generation_changes = generation_changes;
        result.stale_generation_failures = stale_generation_failures;
        result.copy_failures = copy_failures;
        result.failed_cleanups = failed_cleanups;
        result.pin_acquires = pin_acquires;
        result.pin_releases = pin_releases;
        result.current_pins = current_pins;
        result.peak_pins = peak_pins;
        result.h2d_bytes = h2d_bytes;
        result.h2d_time_us = h2d_time_us;
        result.execution_id_read_bytes = execution_id_read_bytes;
        result.execution_id_write_bytes = execution_id_write_bytes;
        result.last_remap_layer = last_remap_layer;
        result.last_logical_ids.assign(last_logical_ids.begin(), last_logical_ids.begin() + last_id_count);
        result.last_execution_ids.assign(last_execution_ids.begin(), last_execution_ids.begin() + last_id_count);
        result.post_h2d_cancellations = post_h2d_cancellations;
        result.last_cancelled_flight = last_cancelled_flight;
        result.last_cancelled_lane = last_cancelled_lane;
        result.last_cancelled_lane_generation = last_cancelled_lane_generation;
        result.last_cancelled_hot_slot = last_cancelled_hot_slot;
        result.last_cancelled_hot_generation = last_cancelled_hot_generation;
        result.last_completed_flight = last_completed_flight;
        result.last_completed_lane = last_completed_lane;
        result.last_completed_lane_generation = last_completed_lane_generation;
        result.last_completed_hot_slot = last_completed_hot_slot;
        result.last_completed_hot_generation = last_completed_hot_generation;
        result.retry_after_cancel_flight = retry_after_cancel_flight;
        result.retry_after_cancel_lane = retry_after_cancel_lane;
        result.retry_after_cancel_lane_generation = retry_after_cancel_lane_generation;
        result.retry_after_cancel_hot_slot = retry_after_cancel_hot_slot;
        result.retry_after_cancel_hot_generation = retry_after_cancel_hot_generation;
        result.last_execution_backend_device_type = last_execution_backend_device_type;
        result.last_failure_error = last_failure_error;
        result.last_remap_error = last_remap_error;
        result.remap_dynamic_allocations = 0;
        if (!config.cpu_cold_only) {
            result.policy = hot_policy.diagnostics();
            result.policy_domains = hot_policy.domain_diagnostics();
            result.policy_events.assign(
                hot_policy.transcript().begin(),
                hot_policy.transcript().begin() + hot_policy.transcript_size());
        }
        result.synchronization_checkpoints = synchronization_checkpoints;
        for (const auto & device_pool : device_pools) {
            if (device_pool) result.slot_tensor_addresses.insert(
                result.slot_tensor_addresses.end(), device_pool->addresses.begin(), device_pool->addresses.end());
        }
        if (pool) {
            result.layout_hot_role_offsets.assign(pool->role_offsets.begin(), pool->role_offsets.end());
            result.layout_hot_role_extents.assign(pool->role_extents.begin(), pool->role_extents.end());
        }
        result.slots.reserve(directory_slots.size());
        for (const auto & entry : directory_slots) {
            result.slots.push_back({
                entry.key.layer,
                entry.key.expert,
                entry.layout_class_id,
                entry.generation,
                entry.last_use,
                entry.refcount,
                entry.cold_slot,
                entry.cold_generation,
                entry.has_cold_backing,
                entry.origin,
                entry.background_useful,
                entry.speculative_deadline,
                entry.speculative_utility,
                entry.device_id,
                entry.device_slot,
                entry.state,
            });
        }
        result.source_buffer_type = prototype.has_value() ? prototype->down.buffer_type : nullptr;
        result.target_buffer_type = config.target_buffer_type;
        result.source_pageable = config.cold_mode && prototype.has_value();
        result.source_pinned_bytes = 0;
        if (cold_cache) {
            const auto cold = cold_cache->diagnostics();
            result.cold_requested_bytes = cold.requested_bytes;
            result.cold_actual_bytes = cold.actual_bytes;
            result.cold_unused_budget_bytes = cold.unused_budget_bytes;
            result.cold_bundle_payload_bytes = cold.bundle_payload_bytes;
            result.cold_slot_footprint = cold.aligned_slot_footprint;
            result.cold_alignment = cold.alignment;
            result.cold_effective_slots = cold.effective_slots;
            result.layout_class_cold_padding_bytes = cold.class_padding_bytes;
            result.layout_cold_role_offsets = cold.role_offsets;
            result.layout_cold_role_extents = cold.role_extents;
            result.cold_pageable = cold.pageable;
            result.cold_requests = cold.requests;
            result.cold_hits = cold.hits;
            result.cold_misses = cold.misses;
            result.cold_admissions = cold.admissions;
            result.cold_evictions = cold.evictions;
            result.cold_speculative_admissions = cold.speculative_admissions;
            result.cold_speculative_replacements = cold.speculative_replacements;
            result.cold_speculative_rejections = cold.speculative_rejections;
            result.cold_speculative_demand_consumptions = cold.speculative_demand_consumptions;
            result.cold_source_copy_bundles = cold.source_copy_bundles;
            result.cold_source_copy_bytes = cold.source_copy_bytes;
            result.cold_source_copy_time_us = cold.source_copy_time_us;
            result.cold_failed_copies = cold.failed_copies;
            result.cold_failed_cleanups = cold.failed_cleanups;
            result.cold_generation_changes = cold.generation_changes;
            result.cold_invariant_failures = cold.invariant_failures;
            result.cold_current_hot_refs = cold.current_hot_refs;
            result.cold_peak_hot_refs = cold.peak_hot_refs;
            result.cold_current_transfer_refs = cold.current_transfer_refs;
            result.cold_peak_transfer_refs = cold.peak_transfer_refs;
            result.cold_current_request_refs = cold.current_request_refs;
            result.cold_peak_request_refs = cold.peak_request_refs;
            result.cold_residency_supported = cold.residency_supported;
            result.cold_residency_unavailable_reason = cold.residency_unavailable_reason;
            result.cold_ready_logical_bytes = cold.ready_logical_bytes;
            result.cold_ready_page_count = cold.ready_page_count;
            result.cold_resident_ready_page_count = cold.resident_ready_page_count;
            result.cold_resident_ready_bytes = cold.resident_ready_bytes;
            result.cold_current_cpu_execution_refs = cold.current_cpu_execution_refs;
            result.cold_peak_cpu_execution_refs = cold.peak_cpu_execution_refs;
            result.cold_current_batch_refs = cold.current_batch_refs;
            result.cold_peak_batch_refs = cold.peak_batch_refs;
            result.cold_policy = cold.policy;
            result.cold_policy_domains = cold.policy_domains;
            result.cold_policy_events = cold.policy_events;
            result.cold_slots.reserve(cold.slots.size());
            for (const auto & entry : cold.slots) {
                llm_hot_cache_diagnostics::cold_slot::state_type state =
                    llm_hot_cache_diagnostics::cold_slot::free;
                switch (entry.state) {
                    case llm_cold_slot_state::free:
                        state = llm_hot_cache_diagnostics::cold_slot::free;
                        break;
                    case llm_cold_slot_state::reserved:
                        state = llm_hot_cache_diagnostics::cold_slot::reserved;
                        break;
                    case llm_cold_slot_state::loading:
                        state = llm_hot_cache_diagnostics::cold_slot::loading;
                        break;
                    case llm_cold_slot_state::ready:
                        state = llm_hot_cache_diagnostics::cold_slot::ready;
                        break;
                    case llm_cold_slot_state::evicting:
                        state = llm_hot_cache_diagnostics::cold_slot::evicting;
                        break;
                    case llm_cold_slot_state::failed:
                        state = llm_hot_cache_diagnostics::cold_slot::failed;
                        break;
                }
                result.cold_slots.push_back({
                    entry.key.layer,
                    entry.key.expert,
                    entry.layout_class_id,
                    entry.generation,
                    entry.last_use,
                    entry.origin,
                    state,
                });
            }
        }
        if (phase10_lead_events) result.phase10_lead_events.assign(
            phase10_lead_events->begin(), phase10_lead_events->begin() + phase10_lead_event_count);
        result.phase10_lead_event_capacity = phase10_lead_events ? phase10_lead_events->size() : 0;
        result.phase10_lead_events_dropped = phase10_lead_events_dropped;
        if (phase10_scheduler_events) result.phase10_scheduler_events.assign(
            phase10_scheduler_events->begin(),
            phase10_scheduler_events->begin() + phase10_scheduler_event_count);
        result.phase10_scheduler_event_capacity = phase10_scheduler_events ? phase10_scheduler_events->size() : 0;
        result.phase10_scheduler_events_dropped = phase10_scheduler_events_dropped;
        if (phase10_issue_ahead_trace) result.phase10_issue_ahead_trace.assign(
            phase10_issue_ahead_trace->begin(),
            phase10_issue_ahead_trace->begin() + phase10_issue_ahead_trace_count);
        result.phase10_issue_ahead_trace_capacity =
            phase10_issue_ahead_trace ? phase10_issue_ahead_trace->size() : 0;
        result.phase10_issue_ahead_trace_dropped = phase10_issue_ahead_trace_dropped;
        result.phase10_prediction_trace.assign(
            phase10_prediction_trace.begin(),
            phase10_prediction_trace.begin() + phase10_prediction_trace_count);
        result.phase10_route_trace.assign(
            phase10_route_trace.begin(),
            phase10_route_trace.begin() + phase10_route_trace_count);
        result.phase10_route_ids.assign(
            phase10_route_ids.begin(),
            phase10_route_ids.begin() + phase10_route_id_count);
        result.physical_feasibility_skips = physical_feasibility_skips;
        result.physical_feasibility_scan_calls = physical_feasibility_scan_calls;
        result.physical_feasibility_scan_time_ns = physical_feasibility_scan_time_ns;
        result.physical_feasibility_scan_max_ns = physical_feasibility_scan_max_ns;
        result.physical_feasibility_scan_decode_calls = physical_feasibility_scan_decode_calls;
        result.physical_feasibility_scan_decode_time_ns = physical_feasibility_scan_decode_time_ns;
        result.physical_feasibility_scan_decode_max_ns = physical_feasibility_scan_decode_max_ns;
        result.provider_h2d_join_waves = provider_h2d_join_waves;
        result.provider_h2d_join_time_ns = provider_h2d_join_time_ns;
        result.provider_h2d_join_max_ns = provider_h2d_join_max_ns;
        result.provider_h2d_join_decode_waves = provider_h2d_join_decode_waves;
        result.provider_h2d_join_decode_time_ns = provider_h2d_join_decode_time_ns;
        result.provider_h2d_join_decode_max_ns = provider_h2d_join_decode_max_ns;
        result.provider_h2d_async_decode_waves = provider_h2d_async_decode_waves;
        result.provider_h2d_async_branch_waits = provider_h2d_async_branch_waits;
        result.injected_device_failure_waves = injected_device_failure_waves;
        result.injected_device_failure_participants = injected_device_failure_participants;
        result.injected_device_failure_drained_waves = injected_device_failure_drained_waves;
        const uint32_t directory_device_count = multi_device ? uint32_t(config.devices.size()) : 1;
        result.directory_device_cells = uint64_t(config.total_expert_keys)*directory_device_count;
        if (multi_device) {
            for (int32_t layer : config.routed_layers) {
                for (uint32_t expert = 0; expert < n_expert; ++expert) {
                    const auto owner = llm_expert_owner_device(int32_t(expert), directory_device_count);
                    const size_t logical = size_t(layer)*n_expert + expert;
                    for (uint32_t device = 0; device < directory_device_count; ++device) {
                        const auto & cell = directory_forward[logical*directory_device_count + device];
                        if (device != owner && cell.slot >= 0) result.directory_owner_only_violations++;
                    }
                }
            }
        }
        result.peer_transport = config.peer_transport;
        result.peer_staging_bytes = config.peer_staging_bytes;
        const auto scheduler = config.scheduler ?
            config.scheduler->diagnostics() : llm_expert_scheduler_diagnostics{};
        result.devices.reserve(device_pools.size());
        for (size_t index = 0; index < device_pools.size(); ++index) {
            llm_hot_cache_diagnostics::device device;
            device.device_id = llm_expert_device_id(index);
            device.requested_capacity = !config.devices.empty() ? config.devices[index].capacity : config.capacity;
            device.effective_capacity = device_pools[index] ? device.requested_capacity : 0;
            if (!config.devices.empty()) {
                device.cuda_ordinal = config.devices[index].cuda_ordinal;
                device.pci_bdf = config.devices[index].pci_bdf;
                device.uuid = config.devices[index].uuid;
            }
            if (device_pools[index]) {
                device.pool_generation = device_pools[index]->id;
                if (device_pools[index]->buffer) {
                    device.pool_bytes = ggml_backend_buffer_get_size(device_pools[index]->buffer.get());
                }
            }
            for (const auto & entry : directory_slots) {
                if (entry.device_id == index && entry.state != hot_slot_state::free &&
                    entry.state != hot_slot_state::failed) {
                    device.occupancy++;
                }
            }
            if (multi_device && index < device_runtime_stats.size()) {
                const auto & counters = device_runtime_stats[index];
                device.hits = counters.hits;
                device.misses = counters.misses;
                device.admissions = counters.admissions;
                device.evictions = counters.evictions;
                device.h2d_bytes = counters.h2d_bytes;
            } else {
                device.hits = hits;
                device.misses = misses;
                device.admissions = admissions;
                device.evictions = evictions;
                device.h2d_bytes = h2d_bytes;
            }
            if (index < transfer_rings.size() && transfer_rings[index]) {
                const auto ring = transfer_rings[index]->diagnostics();
                result.ring_cold_fill_attempts += ring.cold_fill_attempts;
                result.ring_cold_fill_queued += ring.cold_fill_queued;
                result.ring_cold_fill_dropped += ring.cold_fill_dropped;
                result.ring_cold_fill_completed += ring.cold_fill_completed;
                result.ring_cold_fill_failed += ring.cold_fill_failed;
                result.ring_cold_fill_bytes += ring.cold_fill_bytes;
                result.ring_cold_fill_time_us += ring.cold_fill_time_us;
                result.ring_cold_fill_active += ring.cold_fill_active;
                result.ring_cold_fill_peak_active = std::max(
                    result.ring_cold_fill_peak_active, ring.cold_fill_peak_active);
                device.ring_requested_bytes = ring.requested_bytes;
                device.ring_actual_bytes = ring.actual_bytes;
                device.ring_pinned_or_registered_bytes = ring.pinned_or_registered_bytes;
                device.ring_lane_reservations = ring.lane_reservations;
                device.ring_direct_storage_reservations = ring.direct_storage_reservations;
                device.ring_direct_storage_completions = ring.direct_storage_completions;
                device.ring_direct_storage_bytes = ring.direct_storage_bytes;
                device.ring_stage_bytes = ring.stage_bytes;
                device.ring_h2d_bytes = ring.h2d_bytes;
                device.ring_h2d_time_us = ring.h2d_time_us;
                device.ring_waves = ring.waves;
                device.ring_async_enqueues = ring.async_enqueues;
                device.ring_h2d_event_records = ring.h2d_event_records;
                device.ring_h2d_event_waits = ring.h2d_event_waits;
                device.ring_h2d_event_synchronizations = ring.h2d_event_synchronizations;
                device.ring_live_events = ring.live_events;
                device.ring_peak_in_flight_lanes = ring.peak_in_flight_lanes;
                device.ring_first_h2d_enqueue_us = ring.first_h2d_enqueue_us;
                device.ring_last_h2d_complete_us = ring.last_h2d_event_complete_us;
            }
            if (index < scheduler.devices.size()) {
                const auto & scheduled = scheduler.devices[index];
                device.scheduler_request_capacity = scheduled.request_capacity;
                device.scheduler_inflight_capacity = scheduled.inflight_capacity;
                device.scheduler_active_requests = scheduled.active_requests;
                device.scheduler_peak_active_requests = scheduled.peak_active_requests;
                device.scheduler_queued_requests = scheduled.queued_requests;
                device.scheduler_inflight_requests = scheduled.inflight_requests;
                device.scheduler_peak_inflight_requests = scheduled.peak_inflight_requests;
                device.scheduler_reserved_storage_bytes = scheduled.reserved_storage_bytes;
                device.scheduler_peak_reserved_storage_bytes = scheduled.peak_reserved_storage_bytes;
                device.scheduler_reserved_h2d_bytes = scheduled.reserved_h2d_bytes;
                device.scheduler_peak_reserved_h2d_bytes = scheduled.peak_reserved_h2d_bytes;
                device.scheduler_terminal_complete = scheduled.terminal_complete;
                device.scheduler_terminal_failed = scheduled.terminal_failed;
                device.scheduler_terminal_cancelled = scheduled.terminal_cancelled;
                device.scheduler_terminal_releases = scheduled.terminal_releases;
                device.scheduler_stale_completions = scheduled.stale_completions;
            }
            result.devices.push_back(std::move(device));
        }
        if (phase10_lead_events && config.async_transport != nullptr) {
            const auto asynchronous = config.async_transport->diagnostics();
            const auto reads = config.async_transport->completed_read_intervals();
            result.phase10_storage_events.reserve(reads.size());
            for (const auto & read : reads) {
                llm_expert_phase10_storage_event event;
                event.flight = read.flight;
                event.operation_index = read.operation_index;
                event.submit_us = read.submit_us;
                event.complete_us = read.complete_us;
                event.completed_bytes = read.bytes;
                event.useful_bytes = read.useful_bytes;
                event.operation_file_offset = read.operation_file_offset;
                event.source_segment_count = read.source_segment_count;
                for (uint32_t index = 0; index < read.source_segment_count; ++index) {
                    event.source_segments[index] = {
                        read.source_segments[index].file_offset,
                        read.source_segments[index].byte_count,
                    };
                }
                result.phase10_storage_events.push_back(event);
            }
            result.phase10_storage_event_capacity = asynchronous.trace_capacity;
            result.phase10_storage_events_dropped = asynchronous.trace_records_dropped;
        }
        if (transfer_ring) {
            const auto ring = transfer_ring->diagnostics();
            result.ring_requested_bytes = ring.requested_bytes;
            result.ring_actual_bytes = ring.actual_bytes;
            result.ring_lane_footprint = ring.lane_footprint;
            result.ring_effective_lanes = ring.effective_lanes;
            result.ring_pinned_or_registered_bytes = ring.pinned_or_registered_bytes;
            result.ring_acquisition_method = ring.acquisition_method;
            result.ring_fallback_reason = ring.fallback_reason;
            result.ring_pageable_fallback = ring.pageable_fallback;
            result.ring_fallback_count = ring.fallback_count;
            result.ring_lane_reservations = ring.lane_reservations;
            result.ring_direct_storage_reservations = ring.direct_storage_reservations;
            result.ring_direct_storage_completions = ring.direct_storage_completions;
            result.ring_direct_storage_bytes = ring.direct_storage_bytes;
            result.ring_cold_fill_attempts = ring.cold_fill_attempts;
            result.ring_cold_fill_queued = ring.cold_fill_queued;
            result.ring_cold_fill_dropped = ring.cold_fill_dropped;
            result.ring_cold_fill_completed = ring.cold_fill_completed;
            result.ring_cold_fill_failed = ring.cold_fill_failed;
            result.ring_cold_fill_bytes = ring.cold_fill_bytes;
            result.ring_cold_fill_time_us = ring.cold_fill_time_us;
            result.ring_cold_fill_active = ring.cold_fill_active;
            result.ring_cold_fill_peak_active = ring.cold_fill_peak_active;
            result.ring_stage_bytes = ring.stage_bytes;
            result.ring_stage_time_us = ring.stage_time_us;
            result.ring_async_enqueues = ring.async_enqueues;
            result.ring_synchronous_copies = ring.synchronous_copies;
            result.ring_waves = ring.waves;
            result.ring_peak_in_flight_lanes = ring.peak_in_flight_lanes;
            result.ring_wave_synchronizations = ring.wave_synchronizations;
            result.ring_dedicated_transfer_backend = ring.dedicated_transfer_backend;
            result.ring_event_capable = ring.event_capable;
            result.ring_h2d_event_capacity = ring.h2d_event_capacity;
            result.ring_compute_event_capacity = ring.compute_event_capacity;
            result.ring_event_capacity = ring.event_capacity;
            result.ring_live_h2d_events = ring.live_h2d_events;
            result.ring_peak_live_h2d_events = ring.peak_live_h2d_events;
            result.ring_live_compute_events = ring.live_compute_events;
            result.ring_peak_live_compute_events = ring.peak_live_compute_events;
            result.ring_live_events = ring.live_events;
            result.ring_peak_live_events = ring.peak_live_events;
            result.ring_h2d_event_records = ring.h2d_event_records;
            result.ring_h2d_event_waits = ring.h2d_event_waits;
            result.ring_h2d_event_synchronizations = ring.h2d_event_synchronizations;
            result.ring_event_records = ring.event_records;
            result.ring_compute_waits = ring.compute_waits;
            result.ring_event_synchronizations = ring.event_synchronizations;
            result.ring_compute_event_records = ring.compute_event_records;
            result.ring_compute_event_waits = ring.compute_event_waits;
            result.ring_compute_event_synchronizations = ring.compute_event_synchronizations;
            result.ring_h2d_event_cancellations = ring.h2d_event_cancellations;
            result.ring_compute_event_cancellations = ring.compute_event_cancellations;
            result.ring_h2d_event_allocations = ring.h2d_event_allocations;
            result.ring_compute_event_allocations = ring.compute_event_allocations;
            result.ring_h2d_event_frees = ring.h2d_event_frees;
            result.ring_compute_event_frees = ring.compute_event_frees;
            result.ring_compute_work = ring.compute_work;
            result.ring_trace_capacity = ring.trace_capacity;
            result.ring_trace_records = ring.trace_records;
            result.ring_trace_records_dropped = ring.trace_records_dropped;
            result.ring_first_h2d_enqueue_us = ring.first_h2d_enqueue_us;
            result.ring_last_h2d_event_complete_us = ring.last_h2d_event_complete_us;
            result.ring_h2d_compute_overlap_us = ring.h2d_compute_overlap_us;
            result.ring_h2d_compute_overlap_bytes = ring.h2d_compute_overlap_bytes;
            result.ring_h2d_compute_overlap_work = ring.h2d_compute_overlap_work;
            result.ring_h2d_compute_overlap_flights = ring.h2d_compute_overlap_flights;
            if (phase10_lead_events) {
                const auto transfers = transfer_ring->completed_intervals();
                result.phase10_h2d_events.reserve(transfers.size());
                for (const auto & transfer : transfers) {
                    result.phase10_h2d_events.push_back({ transfer.flight, transfer.h2d_enqueue_us,
                        transfer.h2d_complete_us, transfer.bytes, transfer.cancelled });
                }
                result.phase10_h2d_event_capacity = ring.trace_capacity;
                result.phase10_h2d_events_dropped = ring.trace_records_dropped;
            }
            if (config.async_transport != nullptr && ring.event_capable) {
                const auto reads = config.async_transport->completed_read_intervals();
                const auto transfers = transfer_ring->completed_intervals();
                std::vector<std::pair<uint64_t, uint64_t>> read_ranges;
                std::vector<std::pair<uint64_t, uint64_t>> transfer_ranges;
                std::vector<bool> transfer_participates(transfers.size(), false);
                std::vector<llm_expert_flight_id> participating_read_flights;
                std::vector<llm_expert_flight_id> participating_transfer_flights;
                std::vector<llm_expert_flight_id> participating_flights;
                read_ranges.reserve(reads.size());
                transfer_ranges.reserve(transfers.size());
                for (const auto & read : reads) {
                    bool participates = false;
                    for (size_t transfer_index = 0; transfer_index < transfers.size(); ++transfer_index) {
                        const auto & transfer = transfers[transfer_index];
                        if (!read.flight.valid() || !transfer.flight.valid() ||
                            same_flight(read.flight, transfer.flight)) continue;
                        const uint64_t begin = std::max(transfer.h2d_enqueue_us, read.submit_us);
                        const uint64_t end = std::min(transfer.h2d_complete_us, read.complete_us);
                        if (end <= begin) continue;
                        participates = true;
                        transfer_participates[transfer_index] = true;
                        result.disk_h2d_overlap_events++;
                        result.disk_h2d_overlap_pairs++;
                        const uint64_t pair_hash = overlap_pair_hash(read, transfer);
                        result.disk_h2d_overlap_pair_digest ^=
                            pair_hash + 0x9e3779b97f4a7c15ULL + (pair_hash << 6) + (pair_hash >> 2);
                    }
                    if (!participates) continue;
                    read_ranges.emplace_back(read.submit_us, read.complete_us);
                    result.disk_h2d_overlap_read_bytes += read.bytes;
                    if (std::none_of(participating_read_flights.begin(), participating_read_flights.end(),
                            [&](const auto & flight) { return same_flight(flight, read.flight); })) {
                        participating_read_flights.push_back(read.flight);
                    }
                }
                for (size_t transfer_index = 0; transfer_index < transfers.size(); ++transfer_index) {
                    if (!transfer_participates[transfer_index]) continue;
                    const auto & transfer = transfers[transfer_index];
                    transfer_ranges.emplace_back(transfer.h2d_enqueue_us, transfer.h2d_complete_us);
                    result.disk_h2d_overlap_bytes += transfer.bytes;
                    if (std::none_of(participating_transfer_flights.begin(), participating_transfer_flights.end(),
                            [&](const auto & flight) { return same_flight(flight, transfer.flight); })) {
                        participating_transfer_flights.push_back(transfer.flight);
                    }
                }
                const auto add_total_flight = [&](const llm_expert_flight_id & flight) {
                    if (std::none_of(participating_flights.begin(), participating_flights.end(),
                            [&](const auto & existing) { return same_flight(existing, flight); })) {
                        participating_flights.push_back(flight);
                    }
                };
                for (const auto & flight : participating_read_flights) add_total_flight(flight);
                for (const auto & flight : participating_transfer_flights) add_total_flight(flight);
                result.disk_h2d_overlap_us = overlap_union_us(read_ranges, transfer_ranges);
                result.disk_h2d_overlap_read_flights = participating_read_flights.size();
                result.disk_h2d_overlap_transfer_flights = participating_transfer_flights.size();
                result.disk_h2d_overlap_flights = participating_flights.size();
            }
            result.ring_h2d_bytes = ring.h2d_bytes;
            result.layout_class_lane_padding_bytes = ring.class_padding_bytes;
            result.layout_lane_role_offsets = ring.role_offsets;
            result.layout_lane_role_extents = ring.role_extents;
            result.layout_class_stage_bundles = ring.class_stage_bundles;
            result.layout_class_stage_bytes = ring.class_stage_bytes;
            result.layout_class_h2d_bundles = ring.class_h2d_bundles;
            result.layout_class_h2d_bytes = ring.class_h2d_bytes;
            result.ring_h2d_time_us = ring.h2d_time_us;
            result.ring_failed_cleanup = ring.failed_cleanups;
        }
        return result;
    }

    llm_expert_provider_result debug_copy_cold_bundle(
            llm_expert_key key,
            std::vector<uint8_t> & bytes) const noexcept override {
        std::lock_guard<std::mutex> lock(mutex);
        bytes.clear();
        if (!config.cold_mode || !cold_cache ||
            !key.is_valid(LLAMA_MAX_LAYERS, n_expert)) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::unsupported_configuration);
        }
        const auto cold_diagnostics = cold_cache->diagnostics();
        uint32_t cold_slot_index = UINT32_MAX;
        llm_expert_layout_class_id layout_class_id = LLM_EXPERT_LAYOUT_CLASS_INVALID;
        for (uint32_t index = 0; index < cold_diagnostics.slots.size(); ++index) {
            const auto & candidate = cold_diagnostics.slots[index];
            if (candidate.state == llm_cold_slot_state::ready &&
                expert_key_matches(candidate.key, key)) {
                cold_slot_index = index;
                layout_class_id = candidate.layout_class_id;
                break;
            }
        }
        if (cold_slot_index == UINT32_MAX ||
            layout_class_id >= layout_registry.classes.size()) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::invalid_key);
        }
        std::array<llm_expert_storage_destination, 12> destinations;
        size_t count = 0;
        const auto & bundle = cold_cache->bundle(layout_class_id);
        const auto & layout = layout_registry.classes[layout_class_id].prototype;
        if (!append_storage_projection(destinations, count, bundle.up, layout.up,
                bundle.n_expert, layout.n_expert, cold_slot_index,
                llm_expert_storage_projection::up, layout_class_id) ||
            !append_storage_projection(destinations, count, bundle.gate, layout.gate,
                bundle.n_expert, layout.n_expert, cold_slot_index,
                llm_expert_storage_projection::gate, layout_class_id) ||
            !append_storage_projection(destinations, count, bundle.gate_up, layout.gate_up,
                bundle.n_expert, layout.n_expert, cold_slot_index,
                llm_expert_storage_projection::gate_up, layout_class_id) ||
            !append_storage_projection(destinations, count, bundle.down, layout.down,
                bundle.n_expert, layout.n_expert, cold_slot_index,
                llm_expert_storage_projection::down, layout_class_id)) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
        }
        try {
            uint64_t total = 0;
            for (size_t index = 0; index < count; ++index) {
                if (destinations[index].extent > SIZE_MAX - total) {
                    return llm_expert_provider_result::failure(llm_expert_provider_error::allocation_failed);
                }
                total += destinations[index].extent;
            }
            bytes.reserve(size_t(total));
            for (size_t index = 0; index < count; ++index) {
                const auto * begin = static_cast<const uint8_t *>(destinations[index].data);
                bytes.insert(bytes.end(), begin, begin + destinations[index].extent);
            }
        } catch (const std::bad_alloc &) {
            bytes.clear();
            return llm_expert_provider_result::failure(llm_expert_provider_error::allocation_failed);
        }
        return llm_expert_provider_result::success();
    }

    llm_expert_provider_result debug_set_device_delay_for_testing(
            llm_expert_device_id device, uint64_t delay_us) noexcept override {
        std::lock_guard<std::mutex> lock(mutex);
        if ((!multi_device && !config.remote_single) || active_request ||
            device >= config.devices.size() || delay_us > 1000000) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::unsupported_configuration);
        }
        device_transfer_delay_us[device] = delay_us;
        return llm_expert_provider_result::success();
    }

    llm_expert_provider_result debug_set_device_failure_for_testing(
            llm_expert_device_id device, bool fail_device, bool decode_only) noexcept override {
        std::lock_guard<std::mutex> lock(mutex);
        if ((!multi_device && !config.remote_single) || active_request || device >= config.devices.size()) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::unsupported_configuration);
        }
        device_transfer_failure_for_testing[device] = fail_device;
        device_transfer_failure_decode_only_for_testing[device] = fail_device && decode_only;
        return llm_expert_provider_result::success();
    }

    llm_expert_provider_result set_h2d_gate_event_for_testing(
            ggml_backend_event_t event) noexcept override {
        std::lock_guard<std::mutex> lock(mutex);
        if (!config.cold_mode || !transfer_ring || active_request) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::busy);
        }
        return transfer_ring->set_h2d_gate_event_for_testing(event);
    }

    llm_expert_provider_result set_phase8_test_control_for_testing(
            llm_expert_phase8_test_control * control) noexcept override {
        std::lock_guard<std::mutex> lock(mutex);
        if (!config.cold_mode || !transfer_ring || active_request) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::busy);
        }
        const auto installed = transfer_ring->set_phase8_test_control_for_testing(control);
        if (installed.is_ready()) config.phase8_test_control = control;
        return installed;
    }

    llm_expert_provider_result set_phase8_closeout_witness_for_testing(
            llm_expert_phase8_closeout_witness * witness) noexcept override {
        std::lock_guard<std::mutex> lock(mutex);
        if (!config.cold_mode || witness == nullptr || witness->written ||
            config.phase8_closeout_witness != nullptr) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::invalid_binding);
        }
        config.phase8_closeout_witness = witness;
        return llm_expert_provider_result::success();
    }

    llm_expert_provider_result debug_copy_hot_bundle(
            llm_expert_key key,
            std::vector<uint8_t> & bytes) const noexcept override {
        std::lock_guard<std::mutex> lock(mutex);
        bytes.clear();
        if (!pool || !key.is_valid(LLAMA_MAX_LAYERS, n_expert) ||
            directory_forward.empty()) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::unsupported_configuration);
        }
        const auto & forward = directory_forward[forward_index(key)];
        if (!forward_entry_matches(key, forward)) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::invalid_key);
        }
        const auto & hot_bundle = pool->bundles[layout_class_for_layer(key.layer)];
        const auto & layout = layout_registry.classes[layout_class_for_layer(key.layer)].prototype;
        const auto append_tensor = [&](ggml_tensor * tensor, const ggml_tensor * layout_tensor, bool weight) {
            if (tensor == nullptr || layout_tensor == nullptr) return tensor == layout_tensor;
            const int axis = expert_axis(tensor, hot_bundle.n_expert, weight);
            const int layout_axis = expert_axis(layout_tensor, layout.n_expert, weight);
            const size_t extent = layout_axis < 0 ? SIZE_MAX : layout_tensor->nb[layout_axis];
            if (axis < 0 || layout_axis != axis || tensor->nb[axis] < extent || forward.slot < 0 ||
                uint32_t(forward.slot) >= uint32_t(tensor->ne[axis]) ||
                extent > SIZE_MAX - bytes.size()) return false;
            const size_t begin = bytes.size();
            bytes.resize(begin + extent);
            ggml_backend_tensor_get(tensor, bytes.data() + begin,
                size_t(forward.slot)*tensor->nb[axis], extent);
            return true;
        };
        try {
            const std::array<std::pair<const llm_expert_projection_descriptor *,
                const llm_expert_projection_descriptor *>, 4> projections = {{
                { &hot_bundle.up, &layout.up }, { &hot_bundle.gate, &layout.gate },
                { &hot_bundle.gate_up, &layout.gate_up }, { &hot_bundle.down, &layout.down },
            }};
            for (const auto & pair : projections) {
                if (!append_tensor(pair.first->weight, pair.second->weight, true) ||
                    !append_tensor(pair.first->bias, pair.second->bias, false) ||
                    !append_tensor(pair.first->scale, pair.second->scale, false)) {
                    bytes.clear();
                    return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
                }
            }
        } catch (const std::bad_alloc &) {
            bytes.clear();
            return llm_expert_provider_result::failure(llm_expert_provider_error::allocation_failed);
        }
        return llm_expert_provider_result::success();
    }

    bool debug_hot_mapping(
            llm_expert_key key,
            uint64_t * generation,
            uint32_t * slot) const noexcept override {
        std::lock_guard<std::mutex> lock(mutex);
        if (!key.is_valid(LLAMA_MAX_LAYERS, n_expert) || directory_forward.empty()) {
            if (generation) *generation = 0;
            if (slot) *slot = UINT32_MAX;
            return false;
        }
        const auto & forward = directory_forward[forward_index(key)];
        const bool present = forward_entry_matches(key, forward);
        if (generation) *generation = present ? forward.generation : 0;
        if (slot) *slot = present ? uint32_t(forward.slot) : UINT32_MAX;
        return present;
    }

    llm_expert_provider_result debug_warm_all_cold_for_testing() noexcept override {
        std::unique_lock<std::mutex> lock(mutex);
        if (!config.cold_mode || !cold_cache || !config.storage || active_request) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::unsupported_configuration);
        }
        if (cold_cache->diagnostics().effective_slots < config.total_expert_keys) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::allocation_failed);
        }
        auto result = cold_cache->policy_request_begin();
        if (!result.is_ready()) return result;
        storage_load_context storage_context = {
            config.storage, nullptr, nullptr, config.integrity_mode, &lock, {}, false, nullptr, nullptr,
        };
        storage_context.layer_ids = &layout_registry.layer_ids;
        storage_context.layout_registry = &layout_registry;
        for (int32_t layer : config.routed_layers) {
            for (uint32_t expert = 0; expert < n_expert; ++expert) {
                llm_cold_reference reference;
                result = cold_cache->find_or_admit_with_loader(
                    { layer, int32_t(expert) }, reference, load_storage_bundle, &storage_context);
                if (!result.is_ready()) break;
            }
            if (!result.is_ready()) break;
        }
        const auto ended = cold_cache->policy_request_end(result.is_ready(), false);
        if (result.is_ready()) result = ended;
        if (result.is_ready()) result = validate_tier_invariants_locked();
        return result;
    }

    llm_expert_provider_result debug_warm_cold_key_for_testing(
            llm_expert_key key) noexcept override {
        std::unique_lock<std::mutex> lock(mutex);
        if (!config.cold_mode || !cold_cache || !config.storage || active_request ||
                !key.is_valid(LLAMA_MAX_LAYERS, n_expert)) {
            return llm_expert_provider_result::failure(
                llm_expert_provider_error::unsupported_configuration);
        }
        auto result = cold_cache->policy_request_begin();
        if (!result.is_ready()) return result;
        storage_load_context storage_context = {
            config.storage, nullptr, nullptr, config.integrity_mode, &lock, {}, false, nullptr, nullptr,
        };
        storage_context.layer_ids = &layout_registry.layer_ids;
        storage_context.layout_registry = &layout_registry;
        llm_cold_reference reference;
        result = cold_cache->find_or_admit_with_loader(
            key, reference, load_storage_bundle, &storage_context);
        const auto ended = cold_cache->policy_request_end(result.is_ready(), false);
        if (result.is_ready()) result = ended;
        if (result.is_ready()) result = validate_tier_invariants_locked();
        return result;
    }

    bool debug_cold_ready(
            llm_expert_key key,
            uint64_t * generation,
            uint32_t * slot) const noexcept override {
        std::lock_guard<std::mutex> lock(mutex);
        if (!config.cold_mode || !cold_cache || active_request ||
            !key.is_valid(LLAMA_MAX_LAYERS, n_expert)) return false;
        const auto diagnostics = cold_cache->diagnostics();
        for (uint32_t index = 0; index < diagnostics.slots.size(); ++index) {
            const auto & entry = diagnostics.slots[index];
            if (entry.state != llm_cold_slot_state::ready ||
                !expert_key_matches(entry.key, key)) continue;
            if (generation != nullptr) *generation = entry.generation;
            if (slot != nullptr) *slot = index;
            return true;
        }
        return false;
    }

protected:
    void release_handle(uint64_t lease_id) noexcept override {
        LLM_EXPERT_TRACE_SCOPE("k3.provider", "release", "request_id", lease_id);
        std::unique_lock<std::mutex> ordered_lock(ordered_remap_mutex, std::defer_lock);
        if (deterministic_policy_terminals || config.cpu_cold_only) ordered_lock.lock();
        std::lock_guard<std::mutex> lock(mutex);
        if (!active_request || lease_id != active_request_id || !validate_request_pins_locked()) {
            stale_generation_failures++;
        } else {
            const auto released = release_request_pins_locked();
            if (!released.is_ready()) metadata_mismatches++;
            const auto background_finished = config.cpu_cold_only ?
                llm_expert_provider_result::success() : finish_background_request_locked(true);
            if (!background_finished.is_ready()) metadata_mismatches++;
            const bool cancelled = last_remap_error == llm_expert_provider_error::cancelled;
            const bool success = released.is_ready() && background_finished.is_ready() &&
                last_remap_error == llm_expert_provider_error::none;
            const auto ended = end_policy_request_locked(success, cancelled);
            if (!ended.is_ready()) metadata_mismatches++;
            [[maybe_unused]] const uint64_t trace_request_id = llm_perfetto_trace_id(
                llm_perfetto_trace_domain::request, active_request_id);
            LLM_EXPERT_TRACE_ASYNC_END("k3.provider", trace_request_id, "request_id", active_request_id,
                "success", success, "cancelled", cancelled, "terminal_error", uint32_t(last_remap_error));
            active_request = false;
            active_request_id = 0;
            if (!success && phase10_sequence_active) {
                const auto sequence_finished = finish_phase10_sequence_locked();
                if (!sequence_finished.is_ready()) metadata_mismatches++;
            }
            if (config.cold_mode && !validate_tier_invariants_locked().is_ready()) {
                metadata_mismatches++;
            }
        }
        counters.handles_released++;
    }

private:
    uint64_t hot_cache_occupancy_locked() const noexcept {
        return std::count_if(directory_slots.begin(), directory_slots.end(), [](const auto & entry) {
            return entry.state == hot_slot_state::ready || entry.state == hot_slot_state::pinned;
        });
    }

    void reset_phase10_capture_locked() noexcept {
        phase10_prediction_events = 0;
        phase10_prediction_events_dropped = 0;
        phase10_route_events = 0;
        phase10_route_events_dropped = 0;
        phase10_predictions_admitted = 0;
        phase10_predictions_rejected = 0;
        phase10_timely_useful = 0;
        phase10_late_joined = 0;
        phase10_wasted_unused = 0;
        phase10_cancelled_before_io = 0;
        phase10_cancelled_drained = 0;
        phase10_predictor_compute_ns = 0;
        phase10_predictor_digest = UINT64_C(1469598103934665603);
        phase10_circuit_opens = 0;
        phase10_prediction_trace_count = 0;
        phase10_route_trace_count = 0;
        phase10_route_id_count = 0;
    }

    llm_expert_provider_result finish_phase10_sequence_locked() noexcept {
        finish_prediction_request_locked();
        auto result = finish_background_request_locked();
        phase10_sequence_active = false;
        phase10_sequence_owner = 0;
        current_decode_token = 0;
        current_token_next_layer = 0;
        for (auto & route : current_token_routes) route.clear();
        if (result.is_ready() && config.cold_mode) {
            result = validate_tier_invariants_locked();
        }
        return result;
    }

    llm_expert_provider_result begin_phase10_sequence_locked(
            uint64_t sequence_owner,
            bool sequence_start) noexcept {
        if (sequence_owner == 0) sequence_owner = UINT64_MAX;
        const bool replace = !phase10_sequence_active ||
            phase10_sequence_owner != sequence_owner || sequence_start;
        if (!replace) return llm_expert_provider_result::success();
        if (phase10_sequence_active) {
            const auto finished = finish_phase10_sequence_locked();
            if (!finished.is_ready()) return finished;
        }
        if (next_phase10_sequence_id == UINT64_MAX) {
            return llm_expert_provider_result::failure(
                llm_expert_provider_error::generation_exhausted);
        }
        reset_phase10_capture_locked();
        phase10_sequence_id = ++next_phase10_sequence_id;
        const auto predictor_started = prefetch_predictor.request_begin(
            phase10_sequence_id);
        if (!predictor_started.is_ready()) {
            phase10_sequence_id = 0;
            return llm_expert_provider_result::failure(
                llm_expert_provider_error::initialization_failed);
        }
        phase10_sequence_active = true;
        phase10_sequence_owner = sequence_owner;
        current_decode_token = 0;
        current_token_next_layer = 0;
        phase10_circuit_open = false;
        phase10_runtime_failed = false;
        phase10_utility_window_resolved = 0;
        phase10_utility_window_timely = 0;
        phase10_utility_failed_windows = 0;
        for (auto & route : current_token_routes) route.clear();
        for (auto & record : predictive_storage_records) record = {};
        return llm_expert_provider_result::success();
    }

    llm_expert_phase10_prediction_event * prediction_event_locked(uint64_t sequence) noexcept {
        if (sequence == 0) return nullptr;
        for (size_t index = 0; index < phase10_prediction_trace_count; ++index) {
            if (phase10_prediction_trace[index].sequence == sequence) {
                return &phase10_prediction_trace[index];
            }
        }
        return nullptr;
    }

    bool append_route_event_locked(
            uint64_t token,
            int32_t layer,
            const std::vector<int32_t> & ids) noexcept {
        phase10_route_events++;
        if (phase10_route_trace_count == phase10_route_trace.size() ||
                ids.size() > phase10_route_ids.size() - phase10_route_id_count) {
            phase10_route_events_dropped++;
            if (!phase10_circuit_open) {
                phase10_circuit_open = true;
                phase10_circuit_opens++;
            }
            return false;
        }
        llm_expert_phase10_route_event event;
        event.request = phase10_sequence_id;
        event.token = token;
        event.layer = layer;
        event.id_offset = uint32_t(phase10_route_id_count);
        event.id_count = uint32_t(ids.size());
        for (int32_t id : ids) {
            phase10_route_ids[phase10_route_id_count++] = id;
        }
        phase10_digest_append(phase10_predictor_digest, phase10_sequence_id);
        phase10_digest_append(phase10_predictor_digest, token);
        phase10_digest_append(phase10_predictor_digest, uint32_t(layer));
        phase10_digest_append(phase10_predictor_digest, uint32_t(ids.size()));
        for (int32_t id : ids) {
            phase10_digest_append(phase10_predictor_digest, uint32_t(id));
        }
        event.post_event_digest = phase10_predictor_digest;
        phase10_route_trace[phase10_route_trace_count++] = event;
        return true;
    }

    llm_expert_phase10_prediction_event * append_prediction_event_locked(
            llm_expert_phase10_prediction_event event) noexcept {
        phase10_prediction_events++;
        if (phase10_prediction_trace_count == phase10_prediction_trace.size()) {
            phase10_prediction_events_dropped++;
            if (!phase10_circuit_open) {
                phase10_circuit_open = true;
                phase10_circuit_opens++;
            }
            return nullptr;
        }
        event.sequence = phase10_prediction_events;
        event.request = phase10_sequence_id;
        event.config_digest = config.prefetch_config.digest;
        event.predictor_digest = prefetch_predictor.state().digest;
        phase10_digest_append(phase10_predictor_digest, event.sequence);
        phase10_digest_append(phase10_predictor_digest, event.token);
        phase10_digest_append(phase10_predictor_digest, event.deadline_token);
        phase10_digest_append(phase10_predictor_digest, uint32_t(event.source_layer));
        phase10_digest_append(phase10_predictor_digest, uint32_t(event.key.layer));
        phase10_digest_append(phase10_predictor_digest, uint32_t(event.key.expert));
        phase10_digest_append(phase10_predictor_digest, event.rank);
        phase10_digest_append(phase10_predictor_digest, event.score);
        phase10_digest_append(phase10_predictor_digest, uint32_t(event.trigger));
        phase10_digest_append(phase10_predictor_digest, uint32_t(event.readiness));
        phase10_digest_append(phase10_predictor_digest, uint32_t(event.priority));
        event.post_event_digest = phase10_predictor_digest;
        phase10_prediction_trace[phase10_prediction_trace_count] = event;
        return &phase10_prediction_trace[phase10_prediction_trace_count++];
    }

    void resolve_prediction_event_locked(
            llm_expert_phase10_prediction_event & event,
            llm_expert_prefetch_outcome outcome) noexcept {
        if (event.outcome != llm_expert_prefetch_outcome::pending) return;
        event.outcome = outcome;
        switch (outcome) {
            case llm_expert_prefetch_outcome::timely_useful:       phase10_timely_useful++; break;
            case llm_expert_prefetch_outcome::late_joined:        phase10_late_joined++; break;
            case llm_expert_prefetch_outcome::wasted_unused:      phase10_wasted_unused++; break;
            case llm_expert_prefetch_outcome::cancelled_before_io: phase10_cancelled_before_io++; break;
            case llm_expert_prefetch_outcome::cancelled_drained:  phase10_cancelled_drained++; break;
            case llm_expert_prefetch_outcome::rejected:           phase10_predictions_rejected++; break;
            case llm_expert_prefetch_outcome::pending:            return;
        }
        phase10_digest_append(phase10_predictor_digest, event.sequence);
        phase10_digest_append(phase10_predictor_digest, uint32_t(outcome));
        event.post_event_digest = phase10_predictor_digest;
        if (outcome == llm_expert_prefetch_outcome::rejected) return;
        phase10_utility_window_resolved++;
        phase10_utility_window_timely += outcome == llm_expert_prefetch_outcome::timely_useful;
        const uint32_t window = config.prefetch_config.value.utility_window_predictions;
        if (window == 0 || phase10_utility_window_resolved != window) return;
        const bool failed = phase10_utility_window_timely < phase10_utility_min_timely_successes;
        phase10_utility_failed_windows = failed ? phase10_utility_failed_windows + 1 : 0;
        phase10_utility_window_resolved = 0;
        phase10_utility_window_timely = 0;
        if (phase10_utility_failed_windows >= 2 && !phase10_circuit_open) {
            phase10_circuit_open = true;
            phase10_circuit_opens++;
            event.circuit_open_after = true;
        }
    }

    size_t forward_index(const llm_expert_key & key) const noexcept {
        const size_t logical = size_t(key.layer)*n_expert + uint32_t(key.expert);
        if (!multi_device) return logical;
        const auto owner = llm_expert_owner_device(key.expert, uint32_t(config.devices.size()));
        return logical*config.devices.size() + owner;
    }

    bool binding_uses_current_pool(const llm_expert_graph_binding & binding) const noexcept {
        if (binding.layout_class_id != layout_class_for_layer(binding.layer) ||
            binding.layout_class_id >= pool->bundles.size()) {
            return false;
        }
        const auto & bundle = pool->bundles[binding.layout_class_id];
        const bool primary_matches = projection_identity_matches(binding.up, bundle.up) &&
            projection_identity_matches(binding.gate, bundle.gate) &&
            projection_identity_matches(binding.gate_up, bundle.gate_up) &&
            projection_identity_matches(binding.down, bundle.down);
        if (config.remote_single) {
            const auto & remote = binding.remote_device;
            return primary_matches && binding.remote_single && !binding.multi_device && binding.devices.empty() &&
                remote.device_id == 0 && remote.target_device == config.devices.front().target_device &&
                remote.execution_ids == binding.execution_ids &&
                remote.generation_lease.get() == pool.get() &&
                remote.completion_delay_us_for_testing == device_transfer_delay_us.front() &&
                projection_identity_matches(remote.up, bundle.up) &&
                projection_identity_matches(remote.gate, bundle.gate) &&
                projection_identity_matches(remote.gate_up, bundle.gate_up) &&
                projection_identity_matches(remote.down, bundle.down);
        }
        if (!primary_matches || !multi_device) return primary_matches && !binding.multi_device;
        if (!binding.multi_device || binding.devices.size() != device_pools.size()) return false;
        for (size_t index = 0; index < binding.devices.size(); ++index) {
            const auto & device_binding = binding.devices[index];
            const auto & device_bundle = device_pools[index]->bundles[binding.layout_class_id];
            if (device_binding.device_id != index ||
                device_binding.target_device != config.devices[index].target_device ||
                device_binding.generation_lease.get() != device_pools[index].get() ||
                device_binding.completion_delay_us_for_testing != device_transfer_delay_us[index] ||
                !projection_identity_matches(device_binding.up, device_bundle.up) ||
                !projection_identity_matches(device_binding.gate, device_bundle.gate) ||
                !projection_identity_matches(device_binding.gate_up, device_bundle.gate_up) ||
                !projection_identity_matches(device_binding.down, device_bundle.down)) {
                return false;
            }
        }
        return true;
    }

    bool forward_entry_matches(
            const llm_expert_key & key,
            const hot_forward_entry & forward) const noexcept {
        if (forward.slot < 0 || uint32_t(forward.slot) >= directory_slots.size()) {
            return false;
        }
        const auto & entry = directory_slots[forward.slot];
        return expert_key_matches(entry.key, key) && entry.generation == forward.generation &&
            (entry.state == hot_slot_state::ready || entry.state == hot_slot_state::pinned);
    }

    void clear_forward_locked(
            const llm_expert_key & key,
            uint32_t slot,
            uint64_t generation) noexcept {
        if (!key.is_valid(LLAMA_MAX_LAYERS, n_expert)) {
            return;
        }
        auto & forward = directory_forward[forward_index(key)];
        if (forward.slot == int32_t(slot) && forward.generation == generation) {
            forward = {};
        }
    }

    background_promotion_record * find_background_locked(const llm_expert_key & key) noexcept {
        for (auto & record : background_promotions) {
            if (record.active && expert_key_matches(record.key, key)) return &record;
        }
        return nullptr;
    }

    predictive_storage_record * find_predictive_storage_locked(
            const llm_expert_key & key) noexcept {
        for (auto & record : predictive_storage_records) {
            if (record.active && expert_key_matches(record.flight.key, key)) return &record;
        }
        return nullptr;
    }

    void fail_predictive_scheduler_locked(
            storage_async_flight & flight,
            bool cancelled) noexcept {
        if (!flight.handle.valid() || config.scheduler == nullptr) return;
        if (flight.scheduler_state == llm_expert_request_state::queued) {
            (void) config.scheduler->cancel_queued_speculative(flight.handle);
        } else if (flight.scheduler_state >= llm_expert_request_state::submitting &&
                flight.scheduler_state <= llm_expert_request_state::h2d_in_flight) {
            if (cancelled) {
                llm_expert_request_snapshot snapshot;
                const bool demand_owned = config.scheduler->snapshot(
                    flight.handle, snapshot) ==
                        llm_expert_schedule_disposition::admitted &&
                    snapshot.metadata.origin == llm_expert_request_origin::demand;
                if (demand_owned) {
                    (void) config.scheduler->begin_demand_cancellation(
                        flight.handle, flight.scheduler_state);
                } else {
                    (void) config.scheduler->begin_speculative_cancellation(
                        flight.handle, flight.scheduler_state);
                }
                (void) config.scheduler->transition(
                    flight.handle, llm_expert_request_state::cancelling,
                    llm_expert_request_state::draining);
                (void) config.scheduler->finish(
                    flight.handle, llm_expert_request_state::cancelled);
            } else {
                (void) config.scheduler->transition(
                    flight.handle, flight.scheduler_state,
                    llm_expert_request_state::draining);
                (void) config.scheduler->finish(
                    flight.handle, llm_expert_request_state::failed);
            }
            (void) config.scheduler->release_terminal(flight.handle);
        }
        flight.handle = {};
        flight.scheduler_active = false;
        flight.scheduler_state = llm_expert_request_state::free;
    }

    llm_expert_provider_result queue_predictive_h2d_locked(
            predictive_storage_record & prediction) noexcept {
        auto & flight = prediction.flight;
        int32_t selected_slot = -1;
        auto result = select_optional_hot_slot_locked(
            flight.key, llm_expert_cache_policy_admission::optional_background,
            selected_slot);
        if (!result.is_ready() || selected_slot < 0) {
            fail_predictive_scheduler_locked(flight, false);
            if (auto * event = prediction_event_locked(prediction.prediction_sequence)) {
                resolve_prediction_event_locked(*event, llm_expert_prefetch_outcome::rejected);
            }
            prediction.active = false;
            return !result.is_ready() && result.error != llm_expert_provider_error::busy ?
                result : llm_expert_provider_result::success();
        }

        background_promotion_record background;
        background.state = background_promotion_record::lifecycle::queued_or_staging;
        background.key = flight.key;
        background.cold = flight.cold;
        background.scheduler_handle = flight.handle;
        background.scheduler_state = flight.scheduler_state;
        background.hot_slot = uint32_t(selected_slot);
        background.speculative_deadline = prediction.deadline_token + 1;
        background.speculative_utility = prediction.utility;
        background.prediction_sequence = prediction.prediction_sequence;
        background.predictive = true;
        result = prepare_hot_slot_locked(background.hot_slot, background.key, background.cold, false);
        if (result.is_ready()) {
            background.hot_generation = directory_slots[background.hot_slot].generation;
            background.origin_operation_ordinal = hot_policy.diagnostics().operation_ordinal;
            if (config.scheduler->transition(background.scheduler_handle,
                    background.scheduler_state, llm_expert_request_state::h2d_in_flight) !=
                    llm_expert_schedule_disposition::admitted) {
                result = llm_expert_provider_result::failure(
                    llm_expert_provider_error::metadata_mismatch);
            } else {
                background.scheduler_state = llm_expert_request_state::h2d_in_flight;
            }
        }
        if (result.is_ready()) {
            result = transfer_ring->try_queue_background_transfer(
                *cold_cache, background.cold, background.hot_slot,
                background.hot_generation,
                cold_cache->bundle(background.cold.layout_class_id),
                pool->bundles[background.cold.layout_class_id],
                background.lane,
                { config.async_transport->diagnostics().transport_epoch,
                  background.scheduler_handle.slot,
                  background.scheduler_handle.generation,
                  background.key,
                  background.cold.layout_class_id },
                deterministic_policy_terminals);
        }
        if (!result.is_ready()) {
            if (auto * event = prediction_event_locked(prediction.prediction_sequence)) {
                resolve_prediction_event_locked(*event, llm_expert_prefetch_outcome::rejected);
            }
            if (background.hot_generation != 0) {
                const auto ordered = finish_background_before_locked(
                    background.origin_operation_ordinal);
                if (!ordered.is_ready()) result = ordered;
                discard_background_slot_locked(background, false);
            } else {
                fail_predictive_scheduler_locked(flight, false);
            }
            prediction.active = false;
            return result.error == llm_expert_provider_error::busy ?
                llm_expert_provider_result::success() : result;
        }
        auto & installed = background_promotions[background.hot_slot];
        if (installed.active) {
            (void) transfer_ring->cancel_after_h2d(background.lane);
            discard_background_slot_locked(background, false);
            prediction.active = false;
            return llm_expert_provider_result::failure(
                llm_expert_provider_error::metadata_mismatch);
        }
        installed = background;
        installed.active = true;
        flight.handle = {};
        prediction.active = false;
        if (auto * event = prediction_event_locked(background.prediction_sequence)) {
            if (!event->admitted) {
                event->admitted = true;
                phase10_predictions_admitted++;
            }
            event->hot_slot = background.hot_slot;
            event->hot_generation = background.hot_generation;
            event->h2d_bytes = payload_for_key(background.key);
            if (event->enqueue_us == 0) event->enqueue_us = uint64_t(ggml_time_us());
        }
        background_submitted++;
        const uint64_t payload = payload_for_key(background.key);
        background_h2d_bytes = payload > UINT64_MAX - background_h2d_bytes ?
            UINT64_MAX : background_h2d_bytes + payload;
        h2d_bytes = payload > UINT64_MAX - h2d_bytes ? UINT64_MAX : h2d_bytes + payload;
        active_background_flights++;
        peak_background_flights = std::max(
            peak_background_flights, active_background_flights);
        return llm_expert_provider_result::success();
    }

    llm_expert_provider_result complete_predictive_read_locked(
            predictive_storage_record & prediction,
            const llm_expert_async_read_completion & completion,
            llm_expert_async_result waited) noexcept {
        auto & flight = prediction.flight;
        const auto released = config.async_transport->release_read(flight.handle);
        if (released == llm_expert_async_result::ready) flight.read_active = false;
        const auto storage_error = waited == llm_expert_async_result::ready ?
            llm_expert_storage_error::none :
            (waited == llm_expert_async_result::closed ?
                llm_expert_storage_error::cancelled :
                (completion.native_error == 0 ? llm_expert_storage_error::short_read :
                    llm_expert_storage_error::io_error));
        config.storage->record_async_read(
            flight.destination_count, completion.bytes_completed,
            storage_error, completion.native_error);
        const bool integrity_matches = waited == llm_expert_async_result::ready &&
            released == llm_expert_async_result::ready &&
            finalize_payload_integrity(config.integrity_mode, *config.storage,
                flight.destinations.data(), flight.destination_count,
                completion.integrity_status, completion.digest);
        if (!integrity_matches) {
            if (flight.reserved) {
                (void) cold_cache->fail_reservation(flight.key, flight.cold);
                flight.reserved = false;
            }
            fail_predictive_scheduler_locked(flight, waited == llm_expert_async_result::closed);
            if (auto * event = prediction_event_locked(prediction.prediction_sequence)) {
                resolve_prediction_event_locked(*event, llm_expert_prefetch_outcome::wasted_unused);
            }
            prediction.active = false;
            return llm_expert_provider_result::success();
        }
        auto published = cold_cache->publish_ready(flight.key, flight.cold);
        if (!published.is_ready()) {
            fail_predictive_scheduler_locked(flight, false);
            prediction.active = false;
            return published;
        }
        flight.reserved = false;
        flight.read_completed = true;
        if (config.scheduler->transition(
                flight.handle, llm_expert_request_state::io_in_flight,
                llm_expert_request_state::host_ready) !=
                llm_expert_schedule_disposition::admitted) {
            prediction.active = false;
            return llm_expert_provider_result::failure(
                llm_expert_provider_error::metadata_mismatch);
        }
        flight.scheduler_state = llm_expert_request_state::host_ready;
        if (auto * event = prediction_event_locked(prediction.prediction_sequence)) {
            event->cold_ready = true;
            event->cold_slot = flight.cold.slot;
            event->cold_generation = flight.cold.generation;
            event->host_ready_us = completion.complete_us;
        }
        if (config.prefetch_config.value.readiness ==
                LLAMA_EXPERT_PREFETCH_READINESS_HOST_READY) {
            if (config.scheduler->finish(
                    flight.handle, llm_expert_request_state::complete) !=
                    llm_expert_schedule_disposition::admitted ||
                config.scheduler->release_terminal(flight.handle) !=
                    llm_expert_schedule_disposition::admitted) {
                prediction.active = false;
                return llm_expert_provider_result::failure(
                    llm_expert_provider_error::metadata_mismatch);
            }
            flight.handle = {};
            flight.scheduler_state = llm_expert_request_state::free;
            prediction.active = false;
            return llm_expert_provider_result::success();
        }
        return queue_predictive_h2d_locked(prediction);
    }

    llm_expert_provider_result claim_predictive_storage_for_demand_locked(
            predictive_storage_record & prediction,
            storage_async_flight & demand,
            std::unique_lock<std::mutex> * provider_lock) noexcept {
        auto & speculative = prediction.flight;
        llm_expert_async_read_completion completion;
        if (provider_lock != nullptr) provider_lock->unlock();
        const auto waited = config.async_transport->wait_read(
            speculative.handle, completion);
        if (provider_lock != nullptr) provider_lock->lock();
        const auto released = config.async_transport->release_read(speculative.handle);
        speculative.read_active = false;
        const auto storage_error = waited == llm_expert_async_result::ready ?
            llm_expert_storage_error::none :
            (waited == llm_expert_async_result::closed ?
                llm_expert_storage_error::cancelled :
                (completion.native_error == 0 ? llm_expert_storage_error::short_read :
                    llm_expert_storage_error::io_error));
        config.storage->record_async_read(
            speculative.destination_count, completion.bytes_completed,
            storage_error, completion.native_error);
        const bool integrity_matches = waited == llm_expert_async_result::ready &&
            released == llm_expert_async_result::ready &&
            finalize_payload_integrity(config.integrity_mode, *config.storage,
                speculative.destinations.data(), speculative.destination_count,
                completion.integrity_status, completion.digest);
        if (!integrity_matches) {
            if (speculative.reserved) {
                (void) cold_cache->fail_reservation(
                    speculative.key, speculative.cold);
            }
            fail_predictive_scheduler_locked(speculative, false);
            prediction.active = false;
            return llm_expert_provider_result::failure(
                waited == llm_expert_async_result::closed ?
                    llm_expert_provider_error::cancelled :
                    llm_expert_provider_error::copy_failed);
        }
        auto published = cold_cache->publish_ready(
            speculative.key, speculative.cold);
        if (!published.is_ready()) {
            fail_predictive_scheduler_locked(speculative, false);
            prediction.active = false;
            return published;
        }
        speculative.reserved = false;
        if (config.scheduler->transition(
                speculative.handle, llm_expert_request_state::io_in_flight,
                llm_expert_request_state::host_ready) !=
                llm_expert_schedule_disposition::admitted) {
            prediction.active = false;
            return llm_expert_provider_result::failure(
                llm_expert_provider_error::metadata_mismatch);
        }
        demand.cold = speculative.cold;
        demand.cold_hit = true;
        demand.reserved = false;
        demand.handle = speculative.handle;
        demand.scheduler_active = true;
        demand.scheduler_joined = false;
        demand.scheduler_taken = true;
        demand.scheduler_state = llm_expert_request_state::host_ready;
        if (auto * event = prediction_event_locked(prediction.prediction_sequence)) {
            event->cold_ready = true;
            event->cold_slot = speculative.cold.slot;
            event->cold_generation = speculative.cold.generation;
            event->host_ready_us = completion.complete_us;
        }
        speculative.handle = {};
        speculative.scheduler_active = false;
        speculative.scheduler_state = llm_expert_request_state::free;
        prediction.active = false;
        return llm_expert_provider_result::success();
    }

    llm_expert_provider_result reap_predictive_storage_locked() noexcept {
        for (auto & prediction : predictive_storage_records) {
            if (!prediction.active || !prediction.flight.read_active) continue;
            llm_expert_async_read_completion completion;
            const auto polled = config.async_transport->poll_read(
                prediction.flight.handle, completion);
            if (polled == llm_expert_async_result::busy) continue;
            const auto completed = complete_predictive_read_locked(
                prediction, completion, polled);
            if (!completed.is_ready()) return completed;
        }
        return llm_expert_provider_result::success();
    }

    llm_expert_provider_result submit_cold_prediction_locked(
            llm_expert_phase10_prediction_event & event) noexcept {
        predictive_storage_record * record = nullptr;
        for (auto & candidate : predictive_storage_records) {
            if (!candidate.active) {
                record = &candidate;
                break;
            }
        }
        if (record == nullptr) {
            resolve_prediction_event_locked(event, llm_expert_prefetch_outcome::rejected);
            return llm_expert_provider_result::success();
        }

        llm_expert_request_metadata metadata;
        metadata.origin = llm_expert_request_origin::speculative;
        metadata.profile_digest = config.prefetch_config.digest;
        metadata.owner_request = phase10_sequence_id;
        metadata.owner_token = event.token;
        metadata.target_layer = event.key.layer;
        metadata.deadline_token = event.deadline_token;
        const uint64_t payload = payload_for_key(event.key);
        metadata.reserved_storage_bytes = payload;
        metadata.reserved_h2d_bytes = event.readiness == LLAMA_EXPERT_PREFETCH_READINESS_DEVICE_READY ?
            payload : 0;
        metadata.speculative_cold_slots = 1;
        metadata.speculative_hot_slots = event.readiness == LLAMA_EXPERT_PREFETCH_READINESS_DEVICE_READY ? 1 : 0;
        metadata.layout_class_id = layout_class_for_layer(event.key.layer);
        const auto scheduler_priority = static_cast<llm_expert_priority>(event.priority);
        const auto scheduler_readiness = event.readiness == LLAMA_EXPERT_PREFETCH_READINESS_DEVICE_READY ?
            llm_expert_readiness::device_ready : llm_expert_readiness::host_ready;
        const auto scheduled = config.scheduler->enqueue(
            event.key, scheduler_priority, scheduler_readiness, metadata);
        if (scheduled.disposition != llm_expert_schedule_disposition::admitted) {
            resolve_prediction_event_locked(event, llm_expert_prefetch_outcome::rejected);
            return llm_expert_provider_result::success();
        }

        *record = {};
        record->active = true;
        record->prediction_sequence = event.sequence;
        record->deadline_token = event.deadline_token;
        record->target_layer = event.key.layer;
        record->utility = event.score;
        auto & flight = record->flight;
        flight.key = event.key;
        flight.handle = scheduled.handle;
        flight.scheduler_active = true;
        flight.scheduler_state = llm_expert_request_state::queued;
        event.scheduler_slot = scheduled.handle.slot;
        event.scheduler_generation = scheduled.handle.generation;

        bool cold_hit = false;
        auto result = cold_cache->reserve_or_find_speculative(
            event.key, event.deadline_token + 1, event.score,
            flight.cold, cold_hit);
        if (!result.is_ready()) {
            fail_predictive_scheduler_locked(flight, false);
            record->active = false;
            resolve_prediction_event_locked(event, llm_expert_prefetch_outcome::rejected);
            return llm_expert_provider_result::success();
        }
        flight.reserved = !cold_hit;
        event.cold_slot = flight.cold.slot;
        event.cold_generation = flight.cold.generation;

        llm_expert_request_snapshot selected;
        const auto taken = config.scheduler->take_next(selected);
        if (taken.disposition != llm_expert_schedule_disposition::admitted ||
                selected.handle.slot != scheduled.handle.slot ||
                selected.handle.generation != scheduled.handle.generation) {
            if (flight.reserved) {
                (void) cold_cache->fail_reservation(flight.key, flight.cold);
            }
            fail_predictive_scheduler_locked(flight, false);
            record->active = false;
            resolve_prediction_event_locked(event, llm_expert_prefetch_outcome::rejected);
            return llm_expert_provider_result::failure(
                llm_expert_provider_error::metadata_mismatch);
        }
        flight.scheduler_taken = true;
        flight.scheduler_state = llm_expert_request_state::submitting;

        if (cold_hit) {
            if (event.readiness == LLAMA_EXPERT_PREFETCH_READINESS_HOST_READY) {
                fail_predictive_scheduler_locked(flight, false);
                record->active = false;
                resolve_prediction_event_locked(event, llm_expert_prefetch_outcome::rejected);
                return llm_expert_provider_result::success();
            }
            if (config.scheduler->transition(
                    flight.handle, llm_expert_request_state::submitting,
                    llm_expert_request_state::host_ready) !=
                    llm_expert_schedule_disposition::admitted) {
                record->active = false;
                return llm_expert_provider_result::failure(
                    llm_expert_provider_error::metadata_mismatch);
            }
            flight.scheduler_state = llm_expert_request_state::host_ready;
            event.cold_ready = true;
            event.host_ready_us = uint64_t(ggml_time_us());
            return queue_predictive_h2d_locked(*record);
        }

        if (!build_storage_destinations(flight,
                cold_cache->bundle(flight.cold.layout_class_id),
                layout_registry.classes[flight.cold.layout_class_id].prototype,
                flight.cold.slot)) {
            (void) cold_cache->fail_reservation(flight.key, flight.cold);
            fail_predictive_scheduler_locked(flight, false);
            record->active = false;
            resolve_prediction_event_locked(event, llm_expert_prefetch_outcome::rejected);
            return llm_expert_provider_result::failure(
                llm_expert_provider_error::metadata_mismatch);
        }
        const auto planned = config.storage->make_read_plan(
            flight.key, flight.destinations.data(), flight.destination_count,
            flight.operations.data(), flight.operations.size(), flight.operation_count);
        if (!planned.is_ready()) {
            (void) cold_cache->fail_reservation(flight.key, flight.cold);
            fail_predictive_scheduler_locked(flight, false);
            record->active = false;
            resolve_prediction_event_locked(event, llm_expert_prefetch_outcome::rejected);
            return llm_expert_provider_result::success();
        }
        const llm_expert_async_operation_identity identity = {
            config.async_transport->diagnostics().transport_epoch,
            flight.handle, 0, flight.key, scheduler_readiness, scheduler_priority,
            flight.cold.layout_class_id,
        };
        const auto submitted = config.async_transport->submit_read_plan(
            identity, flight.operations.data(), flight.operation_count);
        if (submitted != llm_expert_async_result::ready) {
            (void) cold_cache->fail_reservation(flight.key, flight.cold);
            fail_predictive_scheduler_locked(flight, false);
            record->active = false;
            resolve_prediction_event_locked(event, llm_expert_prefetch_outcome::rejected);
            return llm_expert_provider_result::success();
        }
        flight.submitted = true;
        flight.read_active = true;
        if (config.scheduler->transition(
                flight.handle, llm_expert_request_state::submitting,
                llm_expert_request_state::io_in_flight) !=
                llm_expert_schedule_disposition::admitted) {
            (void) config.async_transport->cancel_read(flight.handle);
            llm_expert_async_read_completion completion;
            (void) config.async_transport->wait_read(flight.handle, completion);
            (void) config.async_transport->release_read(flight.handle);
            flight.read_active = false;
            (void) cold_cache->fail_reservation(flight.key, flight.cold);
            fail_predictive_scheduler_locked(flight, false);
            record->active = false;
            return llm_expert_provider_result::failure(
                llm_expert_provider_error::metadata_mismatch);
        }
        flight.scheduler_state = llm_expert_request_state::io_in_flight;
        event.admitted = true;
        event.storage_bytes = payload_for_key(event.key);
        event.h2d_bytes = event.readiness == LLAMA_EXPERT_PREFETCH_READINESS_DEVICE_READY ?
            payload_for_key(event.key) : 0;
        event.enqueue_us = uint64_t(ggml_time_us());
        phase10_predictions_admitted++;
        return llm_expert_provider_result::success();
    }

    llm_expert_provider_result submit_hot_prediction_locked(
            llm_expert_phase10_prediction_event & event) noexcept {
        auto registration = registrations.find(event.key.layer);
        if (registration == registrations.end()) {
            resolve_prediction_event_locked(event, llm_expert_prefetch_outcome::rejected);
            return llm_expert_provider_result::failure(
                llm_expert_provider_error::metadata_mismatch);
        }
        int32_t selected = -1;
        auto result = select_optional_hot_slot_locked(
            event.key, llm_expert_cache_policy_admission::optional_background,
            selected);
        if (!result.is_ready() || selected < 0) {
            resolve_prediction_event_locked(event, llm_expert_prefetch_outcome::rejected);
            return llm_expert_provider_result::success();
        }
        const uint32_t slot = uint32_t(selected);
        auto & entry = directory_slots[slot];
        const bool replacing = entry.state != hot_slot_state::free;
        if (replacing) {
            LLM_EXPERT_TRACE_INSTANT("k3.cache.hot", "victim", "slot_id", slot,
                "generation", entry.generation, "layer", entry.key.layer,
                "original_expert_id", entry.key.expert);
            if (entry.origin == llm_expert_residency_origin::speculative &&
                    !entry.background_useful) {
                background_wasted++;
            }
            const auto removed = cache_policy_result(
                hot_policy.evict(slot, entry.generation));
            if (!removed.is_ready()) return removed;
            clear_forward_locked(entry.key, slot, entry.generation);
            evictions++;
            LLM_EXPERT_TRACE_INSTANT("k3.cache.hot", "eviction", "slot_id", slot,
                "generation", entry.generation, "layer", entry.key.layer,
                "original_expert_id", entry.key.expert);
        }
        if (entry.generation == UINT64_MAX) {
            resolve_prediction_event_locked(event, llm_expert_prefetch_outcome::rejected);
            return llm_expert_provider_result::failure(
                llm_expert_provider_error::generation_exhausted);
        }
        const uint64_t next_generation = entry.generation + 1;
        entry = {};
        if (replacing) {
            LLM_EXPERT_TRACE_COUNTER("k3.resource", "hot_cache_occupancy", 4,
                hot_cache_occupancy_locked());
        }
        entry.key = event.key;
        entry.layout_class_id = layout_class_for_layer(event.key.layer);
        entry.generation = next_generation;
        entry.state = hot_slot_state::loading;
        auto loading = cache_policy_result(hot_policy.load_begin(
            slot, entry.generation, policy_key_for(event.key),
            payload_for_key(event.key), hot_physical_slot_footprint_bytes, false));
        if (!loading.is_ready()) return loading;
        size_t bytes = 0;
        size_t copies = 0;
        const auto & source = registration->second;
        const auto & hot_bundle = pool->bundles[entry.layout_class_id];
        const bool copied = copy_expert_projection(
                hot_bundle.up, source.up, source.n_expert, config.capacity,
                event.key.expert, slot, bytes, copies, SIZE_MAX) &&
            copy_expert_projection(
                hot_bundle.gate, source.gate, source.n_expert, config.capacity,
                event.key.expert, slot, bytes, copies, SIZE_MAX) &&
            copy_expert_projection(
                hot_bundle.gate_up, source.gate_up, source.n_expert, config.capacity,
                event.key.expert, slot, bytes, copies, SIZE_MAX) &&
            copy_expert_projection(
                hot_bundle.down, source.down, source.n_expert, config.capacity,
                event.key.expert, slot, bytes, copies, SIZE_MAX);
        counters.tensor_copies += copies;
        if (!copied) {
            (void) hot_policy.load_failed(slot, entry.generation);
            const uint64_t generation = entry.generation;
            entry = {};
            entry.generation = generation;
            resolve_prediction_event_locked(event, llm_expert_prefetch_outcome::rejected);
            return llm_expert_provider_result::failure(
                llm_expert_provider_error::copy_failed);
        }
        const auto completed = cache_policy_result(
            hot_policy.load_complete(slot, entry.generation));
        if (!completed.is_ready()) return completed;
        entry.state = hot_slot_state::ready;
        entry.origin = llm_expert_residency_origin::speculative;
        entry.speculative_deadline = event.deadline_token + 1;
        entry.speculative_utility = event.score;
        directory_forward[forward_index(entry.key)] = {
            int32_t(slot), entry.generation };
        admissions++;
        LLM_EXPERT_TRACE_INSTANT("k3.cache.hot", "admission",
            "layer", entry.key.layer, "original_expert_id", entry.key.expert,
            "slot_id", slot, "generation", entry.generation);
        LLM_EXPERT_TRACE_COUNTER("k3.resource", "hot_cache_occupancy", 4,
            hot_cache_occupancy_locked());
        generation_changes++;
        h2d_bytes = bytes > UINT64_MAX - h2d_bytes ? UINT64_MAX : h2d_bytes + bytes;
        event.admitted = true;
        event.device_ready = true;
        event.hot_slot = slot;
        event.hot_generation = entry.generation;
        event.h2d_bytes = bytes;
        event.device_ready_us = uint64_t(ggml_time_us());
        phase10_predictions_admitted++;
        return llm_expert_provider_result::success();
    }

    void submit_prediction_candidates_locked(
            uint64_t token,
            uint64_t deadline_token,
            int32_t source_layer,
            llm_expert_prefetch_trigger trigger,
            const std::vector<llm_expert_prefetch_candidate> & candidates,
            uint64_t predictor_compute_ns) noexcept {
        phase10_predictor_compute_ns = predictor_compute_ns >
                UINT64_MAX - phase10_predictor_compute_ns ?
            UINT64_MAX : phase10_predictor_compute_ns + predictor_compute_ns;
        if (phase10_circuit_open || phase10_runtime_failed) return;
        for (const auto & candidate : candidates) {
            llm_expert_phase10_prediction_event event;
            event.token = token;
            event.deadline_token = deadline_token;
            event.source_layer = source_layer;
            event.key = { candidate.key.layer, candidate.key.expert };
            event.rank = candidate.rank;
            event.score = candidate.score;
            event.predictor_compute_ns = predictor_compute_ns;
            event.trigger = trigger;
            event.readiness = config.prefetch_config.value.readiness;
            const auto policy = config.prefetch_config.value.policy;
            event.priority = uint8_t(policy == LLAMA_EXPERT_PREFETCH_POLICY_STATIC_LAYER ||
                    policy == LLAMA_EXPERT_PREFETCH_POLICY_RANDOM_BASELINE ?
                llm_expert_priority::prefetch_speculative :
                llm_expert_priority::prefetch_next);
            auto * stored = append_prediction_event_locked(event);
            if (stored == nullptr) return;

            bool duplicate = false;
            for (size_t index = 0; index + 1 < phase10_prediction_trace_count; ++index) {
                const auto & prior = phase10_prediction_trace[index];
                duplicate = duplicate ||
                    (prior.outcome == llm_expert_prefetch_outcome::pending &&
                     prior.deadline_token == deadline_token &&
                     expert_key_matches(prior.key, stored->key));
            }
            const auto & forward = directory_forward[forward_index(stored->key)];
            const bool hot_ready = forward_entry_matches(stored->key, forward);
            if (duplicate || hot_ready || find_background_locked(stored->key) != nullptr ||
                    find_predictive_storage_locked(stored->key) != nullptr) {
                resolve_prediction_event_locked(
                    *stored, llm_expert_prefetch_outcome::rejected);
                continue;
            }

            if (!config.cold_mode) {
                uint32_t speculative_hot_slots = 0;
                for (const auto & entry : directory_slots) {
                    speculative_hot_slots +=
                        entry.origin == llm_expert_residency_origin::speculative &&
                        !entry.background_useful && entry.state == hot_slot_state::ready;
                }
                uint64_t token_h2d_bytes = 0;
                for (size_t index = 0; index + 1 < phase10_prediction_trace_count; ++index) {
                    const auto & prior = phase10_prediction_trace[index];
                    if (prior.token != token || !prior.admitted) continue;
                    token_h2d_bytes = prior.h2d_bytes > UINT64_MAX - token_h2d_bytes ?
                        UINT64_MAX : token_h2d_bytes + prior.h2d_bytes;
                }
                const uint64_t byte_limit =
                    config.prefetch_config.value.max_speculative_h2d_bytes_per_token;
                if (speculative_hot_slots >=
                        config.prefetch_config.value.max_speculative_hot_slots ||
                        token_h2d_bytes > byte_limit ||
                        payload_for_key(stored->key) > byte_limit - token_h2d_bytes) {
                    resolve_prediction_event_locked(
                        *stored, llm_expert_prefetch_outcome::rejected);
                    continue;
                }
            }

            const auto result = config.cold_mode ?
                submit_cold_prediction_locked(*stored) :
                submit_hot_prediction_locked(*stored);
            if (!result.is_ready()) {
                phase10_runtime_failed = true;
                if (stored->outcome == llm_expert_prefetch_outcome::pending) {
                    resolve_prediction_event_locked(
                        *stored, llm_expert_prefetch_outcome::rejected);
                }
                if (!phase10_circuit_open) {
                    phase10_circuit_open = true;
                    phase10_circuit_opens++;
                    stored->circuit_open_after = true;
                }
                return;
            }
        }
    }

    void cancel_prediction_sequence_locked(
            uint64_t sequence,
            llm_expert_prefetch_outcome outcome) noexcept {
        auto * event = prediction_event_locked(sequence);
        if (event != nullptr) {
            resolve_prediction_event_locked(*event, outcome);
        }
        for (auto & prediction : predictive_storage_records) {
            if (!prediction.active || prediction.prediction_sequence != sequence ||
                    prediction.demand_claimed) continue;
            auto & flight = prediction.flight;
            if (flight.read_active) {
                (void) config.async_transport->cancel_read(flight.handle);
                llm_expert_async_read_completion completion;
                const auto waited = config.async_transport->wait_read(
                    flight.handle, completion);
                (void) config.async_transport->release_read(flight.handle);
                flight.read_active = false;
                config.storage->record_async_read(
                    flight.destination_count, completion.bytes_completed,
                    llm_expert_storage_error::cancelled, completion.native_error);
                (void) waited;
            }
            if (flight.reserved) {
                (void) cold_cache->fail_reservation(flight.key, flight.cold);
                flight.reserved = false;
            }
            fail_predictive_scheduler_locked(flight, true);
            prediction.active = false;
        }
        for (auto & background : background_promotions) {
            if (!background.active || !background.predictive ||
                    background.prediction_sequence != sequence ||
                    background.demand_claimed) continue;
            (void) transfer_ring->cancel_after_h2d(background.lane);
            background.state = background_promotion_record::lifecycle::cancelled;
            discard_background_slot_locked(background, false);
            background_dropped++;
        }
        if (event != nullptr && !event->demand_claimed &&
                event->hot_slot < directory_slots.size()) {
            const uint32_t slot = event->hot_slot;
            auto & entry = directory_slots[slot];
            if (entry.generation == event->hot_generation &&
                    expert_key_matches(entry.key, event->key) &&
                    entry.origin == llm_expert_residency_origin::speculative &&
                    !entry.background_useful && entry.refcount == 0 &&
                    entry.state == hot_slot_state::ready) {
                LLM_EXPERT_TRACE_INSTANT("k3.cache.hot", "victim", "slot_id", slot,
                    "generation", entry.generation, "layer", entry.key.layer,
                    "original_expert_id", entry.key.expert);
                const auto removed = cache_policy_result(
                    hot_policy.evict(slot, entry.generation));
                if (!removed.is_ready()) {
                    metadata_mismatches++;
                    phase10_runtime_failed = true;
                    return;
                }
                if (entry.has_cold_backing) {
                    const auto released = cold_cache->release(
                        { entry.cold_slot, entry.cold_generation, entry.layout_class_id },
                        llm_cold_reference_kind::hot);
                    if (!released.is_ready()) {
                        metadata_mismatches++;
                        phase10_runtime_failed = true;
                        return;
                    }
                    entry.has_cold_backing = false;
                }
                clear_forward_locked(entry.key, slot, entry.generation);
                LLM_EXPERT_TRACE_INSTANT("k3.cache.hot", "eviction", "slot_id", slot,
                    "generation", entry.generation, "layer", entry.key.layer,
                    "original_expert_id", entry.key.expert);
                const uint64_t generation = entry.generation;
                entry = {};
                entry.generation = generation;
                LLM_EXPERT_TRACE_COUNTER("k3.resource", "hot_cache_occupancy", 4,
                    hot_cache_occupancy_locked());
            }
        }
    }

    void cancel_unclaimed_predictions_locked(
            llm_expert_prefetch_outcome outcome) noexcept {
        for (size_t index = 0; index < phase10_prediction_trace_count; ++index) {
            auto & event = phase10_prediction_trace[index];
            if (event.outcome == llm_expert_prefetch_outcome::pending &&
                    !event.demand_claimed) {
                cancel_prediction_sequence_locked(event.sequence, outcome);
            }
        }
    }

    void finish_prediction_request_locked() noexcept {
        for (auto & prediction : predictive_storage_records) {
            prediction.demand_claimed = false;
        }
        for (auto & background : background_promotions) {
            if (background.predictive) background.demand_claimed = false;
        }
        for (size_t index = 0; index < phase10_prediction_trace_count; ++index) {
            auto & event = phase10_prediction_trace[index];
            if (event.outcome == llm_expert_prefetch_outcome::pending) {
                cancel_prediction_sequence_locked(
                    event.sequence, llm_expert_prefetch_outcome::wasted_unused);
            } else {
                cancel_prediction_sequence_locked(event.sequence, event.outcome);
            }
        }
    }

    void resolve_prediction_deadline_locked(
            uint64_t token,
            int32_t layer,
            const int32_t * logical_ids,
            size_t logical_id_count) noexcept {
        for (size_t index = 0; index < phase10_prediction_trace_count; ++index) {
            auto & event = phase10_prediction_trace[index];
            if (event.outcome != llm_expert_prefetch_outcome::pending ||
                    event.deadline_token != token || event.key.layer != layer) continue;
            bool demanded = false;
            for (size_t logical = 0; logical < logical_id_count; ++logical) {
                demanded = demanded || logical_ids[logical] == event.key.expert;
            }
            const bool host_ready = event.cold_ready && config.cold_mode &&
                cold_cache->ready({ event.cold_slot, event.cold_generation,
                    layout_class_for_layer(event.key.layer) });
            const auto & forward = directory_forward[forward_index(event.key)];
            const bool device_ready = event.device_ready &&
                forward.slot == int32_t(event.hot_slot) &&
                forward.generation == event.hot_generation &&
                forward_entry_matches(event.key, forward);
            const bool ready = event.readiness == LLAMA_EXPERT_PREFETCH_READINESS_HOST_READY ?
                host_ready : device_ready;
            if (demanded) {
                event.demand_claimed = true;
                resolve_prediction_event_locked(event,
                    ready ? llm_expert_prefetch_outcome::timely_useful :
                        llm_expert_prefetch_outcome::late_joined);
                if (auto * storage = find_predictive_storage_locked(event.key)) {
                    if (storage->prediction_sequence == event.sequence) {
                        storage->demand_claimed = true;
                    }
                }
                if (auto * background = find_background_locked(event.key)) {
                    if (background->predictive &&
                            background->prediction_sequence == event.sequence) {
                        background->demand_claimed = true;
                    }
                }
            } else {
                cancel_prediction_sequence_locked(
                    event.sequence, llm_expert_prefetch_outcome::wasted_unused);
            }
        }
        if (phase10_circuit_open) {
            cancel_unclaimed_predictions_locked(
                llm_expert_prefetch_outcome::cancelled_drained);
        }
    }

    void phase10_before_remap_locked(
            const llm_expert_graph_binding & binding,
            const int32_t * logical_ids,
            size_t logical_id_count) noexcept {
        if (!config.prefetch_config.supplied ||
                config.prefetch_config.value.policy == LLAMA_EXPERT_PREFETCH_POLICY_OFF ||
                binding.execution_ids->ne[1] != 1 || phase10_runtime_failed) return;
        if (config.cold_mode) {
            bool reaped = reap_predictive_storage_locked().is_ready();
            for (size_t attempt = 0; reaped && attempt < background_promotions.size(); ++attempt) {
                const uint32_t before = active_background_flights;
                reaped = reap_background_locked(nullptr).is_ready();
                if (!reaped || active_background_flights == before) break;
            }
            if (!reaped) {
                phase10_runtime_failed = true;
                phase10_circuit_open = true;
                phase10_circuit_opens++;
                return;
            }
        }
        const auto layer_it = std::find(
            config.routed_layers.begin(), config.routed_layers.end(), binding.layer);
        if (layer_it == config.routed_layers.end()) return;
        const size_t layer_index = size_t(layer_it - config.routed_layers.begin());
        if (layer_index >= current_token_routes.size() ||
                !current_token_routes[layer_index].empty()) {
            phase10_runtime_failed = true;
            if (!phase10_circuit_open) {
                phase10_circuit_open = true;
                phase10_circuit_opens++;
            }
            return;
        }
        resolve_prediction_deadline_locked(
            current_decode_token, binding.layer, logical_ids, logical_id_count);

        auto & selected = current_token_routes[layer_index];
        selected.assign(logical_ids, logical_ids + logical_id_count);
        std::sort(selected.begin(), selected.end());
        selected.erase(std::unique(selected.begin(), selected.end()), selected.end());
        if (selected.empty() || selected.size() > config.n_expert_used) {
            phase10_runtime_failed = true;
            phase10_circuit_open = true;
            phase10_circuit_opens++;
            return;
        }
        while (current_token_next_layer < current_token_routes.size() &&
                !current_token_routes[current_token_next_layer].empty()) {
            const size_t canonical_index = current_token_next_layer;
            const int32_t canonical_layer = config.routed_layers[canonical_index];
            const auto & canonical_selected = current_token_routes[canonical_index];
            (void) append_route_event_locked(
                current_decode_token, canonical_layer, canonical_selected);
            if (config.prefetch_config.value.policy ==
                    LLAMA_EXPERT_PREFETCH_POLICY_CROSS_LAYER_TRANSITION &&
                    canonical_index + 1 < config.routed_layers.size() &&
                    !phase10_circuit_open) {
                const int32_t target_layer = config.routed_layers[canonical_index + 1];
                const auto started = std::chrono::steady_clock::now();
                const auto predicted = prefetch_predictor.predict_cross_layer(
                    current_decode_token, canonical_layer, canonical_selected.data(), canonical_selected.size(),
                    target_layer, predictor_candidates);
                const uint64_t compute_ns = uint64_t(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - started).count());
                if (!predicted.is_ready()) {
                    phase10_runtime_failed = true;
                    phase10_circuit_open = true;
                    phase10_circuit_opens++;
                    return;
                }
                submit_prediction_candidates_locked(
                    current_decode_token, current_decode_token, canonical_layer,
                    llm_expert_prefetch_trigger::router_result,
                    predictor_candidates, compute_ns);
            }
            current_token_next_layer++;
        }
    }

    void phase10_after_remap_locked(
            const llm_expert_graph_binding & binding) noexcept {
        if (!config.prefetch_config.supplied ||
                config.prefetch_config.value.policy == LLAMA_EXPERT_PREFETCH_POLICY_OFF ||
                binding.execution_ids->ne[1] != 1 || phase10_runtime_failed) return;
        const auto layer_it = std::find(
            config.routed_layers.begin(), config.routed_layers.end(), binding.layer);
        if (layer_it == config.routed_layers.end() ||
                current_token_next_layer != current_token_routes.size()) return;
        const auto committed = prefetch_predictor.commit_token(
            current_decode_token, current_token_routes);
        if (!committed.is_ready()) {
            phase10_runtime_failed = true;
            phase10_circuit_open = true;
            phase10_circuit_opens++;
            return;
        }
        if (config.prefetch_config.value.policy !=
                LLAMA_EXPERT_PREFETCH_POLICY_CROSS_LAYER_TRANSITION &&
                !phase10_circuit_open) {
            for (int32_t target_layer : config.routed_layers) {
                const auto started = std::chrono::steady_clock::now();
                const auto predicted = prefetch_predictor.predict_token_end(
                    current_decode_token, target_layer, predictor_candidates);
                const uint64_t compute_ns = uint64_t(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - started).count());
                if (!predicted.is_ready()) {
                    phase10_runtime_failed = true;
                    phase10_circuit_open = true;
                    phase10_circuit_opens++;
                    break;
                }
                submit_prediction_candidates_locked(
                    current_decode_token, current_decode_token + 1, -1,
                    llm_expert_prefetch_trigger::token_end,
                    predictor_candidates, compute_ns);
            }
        }
        current_decode_token++;
        current_token_next_layer = 0;
        for (auto & route : current_token_routes) route.clear();
    }

    void finish_background_scheduler_locked(background_promotion_record & record, bool success) noexcept {
        if (!record.scheduler_handle.valid() || config.scheduler == nullptr) return;
        if (success) {
            (void) config.scheduler->transition(record.scheduler_handle,
                record.scheduler_state, llm_expert_request_state::device_ready);
            (void) config.scheduler->finish(record.scheduler_handle, llm_expert_request_state::complete);
        } else {
            (void) config.scheduler->transition(record.scheduler_handle,
                record.scheduler_state, llm_expert_request_state::draining);
            (void) config.scheduler->finish(record.scheduler_handle, llm_expert_request_state::failed);
        }
        (void) config.scheduler->release_terminal(record.scheduler_handle);
        record.scheduler_handle = {};
    }

    void discard_background_slot_locked(background_promotion_record & record, bool count_wasted) noexcept {
        if (record.predictive) {
            if (auto * event = prediction_event_locked(record.prediction_sequence)) {
                resolve_prediction_event_locked(
                    *event, llm_expert_prefetch_outcome::wasted_unused);
            }
        }
        if (record.hot_slot < directory_slots.size()) {
            auto & entry = directory_slots[record.hot_slot];
            if (entry.generation == record.hot_generation && expert_key_matches(entry.key, record.key)) {
                const bool was_resident = entry.state == hot_slot_state::ready ||
                    entry.state == hot_slot_state::pinned;
                if (was_resident) {
                    LLM_EXPERT_TRACE_INSTANT("k3.cache.hot", "victim", "slot_id", record.hot_slot,
                        "generation", entry.generation, "layer", entry.key.layer,
                        "original_expert_id", entry.key.expert);
                }
                llm_expert_cache_policy_result policy_result_value;
                if (entry.state == hot_slot_state::loading) {
                    policy_result_value = hot_policy.load_failed(record.hot_slot, entry.generation);
                } else if (entry.state == hot_slot_state::ready || entry.state == hot_slot_state::pinned) {
                    policy_result_value = hot_policy.remove_resident(record.hot_slot, entry.generation);
                }
                if (!policy_result_value.is_ready()) metadata_mismatches++;
                clear_forward_locked(entry.key, record.hot_slot, entry.generation);
                if (entry.has_cold_backing) {
                    (void) cold_cache->release(
                        { entry.cold_slot, entry.cold_generation, entry.layout_class_id },
                        llm_cold_reference_kind::hot);
                }
                if (was_resident) {
                    LLM_EXPERT_TRACE_INSTANT("k3.cache.hot", "eviction", "slot_id", record.hot_slot,
                        "generation", entry.generation, "layer", entry.key.layer,
                        "original_expert_id", entry.key.expert);
                }
                const uint64_t generation = entry.generation;
                entry = {};
                entry.generation = generation;
                if (was_resident) {
                    LLM_EXPERT_TRACE_COUNTER("k3.resource", "hot_cache_occupancy", 4,
                        hot_cache_occupancy_locked());
                }
            }
        }
        finish_background_scheduler_locked(record, false);
        if (record.active && active_background_flights > 0) active_background_flights--;
        if (count_wasted) background_wasted++;
        record = {};
    }

    llm_expert_provider_result reap_background_locked(
            std::unique_lock<std::mutex> * provider_lock) noexcept {
        if (!transfer_ring) return llm_expert_provider_result::success();
        background_promotion_record * earliest = nullptr;
        for (auto & candidate : background_promotions) {
            if (candidate.active && (earliest == nullptr ||
                    candidate.origin_operation_ordinal < earliest->origin_operation_ordinal)) {
                earliest = &candidate;
            }
        }
        if (earliest == nullptr) return llm_expert_provider_result::success();
        auto & record = *earliest;
        if (config.phase8_test_control != nullptr &&
            config.phase8_test_control->consume_fault(
                llm_expert_phase8_test_fault::stale_generation,
                record.key, record.hot_generation)) {
            const auto cancelled = transfer_ring->cancel_after_h2d(record.lane);
            if (!cancelled.is_ready()) return cancelled;
            record.state = background_promotion_record::lifecycle::failed;
            discard_background_slot_locked(record, false);
            background_dropped++;
            return llm_expert_provider_result::success();
        }
        llm_expert_same_key_h2d_state state = llm_expert_same_key_h2d_state::none;
        uint64_t remaining = 0;
        const auto polled = transfer_ring->poll_h2d(record.lane, state, remaining);
        if (!polled.is_ready()) {
            const auto released = transfer_ring->release_terminal_background(record.lane);
            if (!released.is_ready()) return released;
            record.state = background_promotion_record::lifecycle::failed;
            discard_background_slot_locked(record, false);
            background_dropped++;
            return llm_expert_provider_result::success();
        }
        record.remaining_bytes = remaining;
        if (state == llm_expert_same_key_h2d_state::queued_or_staging) {
            record.state = background_promotion_record::lifecycle::queued_or_staging;
            return llm_expert_provider_result::success();
        }
        if (state == llm_expert_same_key_h2d_state::h2d_in_flight) {
            record.state = background_promotion_record::lifecycle::h2d_in_flight;
            return llm_expert_provider_result::success();
        }
        record.state = background_promotion_record::lifecycle::h2d_complete_unpublished;
        auto & entry = directory_slots[record.hot_slot];
        const bool injected_metadata = config.phase8_test_control != nullptr &&
            config.phase8_test_control->consume_fault(
                llm_expert_phase8_test_fault::metadata_mismatch,
                record.key, record.hot_generation);
        if (injected_metadata || entry.state != hot_slot_state::loading || entry.generation != record.hot_generation ||
            !expert_key_matches(entry.key, record.key)) {
            const auto released = transfer_ring->release_terminal_background(record.lane);
            if (!released.is_ready()) return released;
            record.state = background_promotion_record::lifecycle::failed;
            discard_background_slot_locked(record, false);
            background_dropped++;
            return llm_expert_provider_result::success();
        }
        if (config.phase8_test_control != nullptr) {
            const auto gated_key = record.key;
            const uint64_t gated_generation = record.hot_generation;
            if (provider_lock != nullptr) provider_lock->unlock();
            const bool paused = config.phase8_test_control->pause_if_armed(
                llm_expert_phase8_test_gate::background_before_provider_publication,
                gated_key, gated_generation);
            if (provider_lock != nullptr) provider_lock->lock();
            if (paused && (!record.active || record.hot_generation != gated_generation ||
                !expert_key_matches(record.key, gated_key))) {
                return llm_expert_provider_result::failure(llm_expert_provider_error::stale_generation);
            }
        }
        if (config.phase8_test_control != nullptr &&
            config.phase8_test_control->consume_fault(
                llm_expert_phase8_test_fault::publication_failed,
                record.key, record.hot_generation)) {
            const auto released = transfer_ring->release_terminal_background(record.lane);
            if (!released.is_ready()) return released;
            record.state = background_promotion_record::lifecycle::failed;
            discard_background_slot_locked(record, false);
            background_dropped++;
            return llm_expert_provider_result::success();
        }
        const auto released = transfer_ring->release_terminal_background(record.lane);
        if (!released.is_ready()) return released;
        const auto completed = cache_policy_result(hot_policy.load_complete(
            record.hot_slot, entry.generation));
        if (!completed.is_ready()) return completed;
        entry.state = hot_slot_state::ready;
        entry.origin = llm_expert_residency_origin::speculative;
        entry.background_useful = false;
        entry.speculative_deadline = record.speculative_deadline;
        entry.speculative_utility = record.speculative_utility;
        directory_forward[forward_index(entry.key)] = { int32_t(record.hot_slot), entry.generation };
        admissions++;
        LLM_EXPERT_TRACE_INSTANT("k3.cache.hot", "admission",
            "layer", entry.key.layer, "original_expert_id", entry.key.expert,
            "slot_id", record.hot_slot, "generation", entry.generation);
        LLM_EXPERT_TRACE_COUNTER("k3.resource", "hot_cache_occupancy", 4,
            hot_cache_occupancy_locked());
        if (record.predictive) {
            if (auto * event = prediction_event_locked(record.prediction_sequence)) {
                event->device_ready = true;
                event->hot_slot = record.hot_slot;
                event->hot_generation = record.hot_generation;
                event->device_ready_us = uint64_t(ggml_time_us());
            }
        }
        record.state = background_promotion_record::lifecycle::published;
        finish_background_scheduler_locked(record, true);
        record.active = false;
        record.scheduler_handle = {};
        if (active_background_flights > 0) active_background_flights--;
        background_completed++;
        return llm_expert_provider_result::success();
    }

    llm_expert_provider_result finish_background_before_locked(uint64_t operation_ordinal) noexcept {
        while (true) {
            background_promotion_record * earliest = nullptr;
            for (auto & record : background_promotions) {
                if (record.active && record.origin_operation_ordinal < operation_ordinal &&
                        (earliest == nullptr ||
                            record.origin_operation_ordinal < earliest->origin_operation_ordinal)) {
                    earliest = &record;
                }
            }
            if (earliest == nullptr) return llm_expert_provider_result::success();
            const auto lane = earliest->lane;
            const auto waited = transfer_ring->wait_background_h2d(lane);
            auto result = reap_background_locked(nullptr);
            if (!result.is_ready()) return result;
            if (!waited.is_ready() && earliest->active) return waited;
            if (earliest->active) {
                return llm_expert_provider_result::failure(
                    llm_expert_provider_error::metadata_mismatch);
            }
        }
    }

    llm_expert_provider_result normalize_background_terminals_locked(
            std::unique_lock<std::mutex> * provider_lock,
            bool retain_predictive = false) noexcept {
        while (active_background_flights != 0) {
            background_promotion_record * earliest = nullptr;
            for (auto & record : background_promotions) {
                if (record.active && (!retain_predictive || !record.predictive) &&
                        (earliest == nullptr ||
                        record.origin_operation_ordinal < earliest->origin_operation_ordinal)) {
                    earliest = &record;
                }
            }
            if (earliest == nullptr && retain_predictive) {
                return llm_expert_provider_result::success();
            }
            if (earliest == nullptr) {
                return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
            }
            const auto lane = earliest->lane;
            const auto waited = transfer_ring->wait_background_h2d(lane);
            auto result = reap_background_locked(provider_lock);
            if (!result.is_ready()) return result;
            if (!waited.is_ready() && earliest->active) return waited;
            if (earliest->active) {
                return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
            }
        }
        return llm_expert_provider_result::success();
    }

    llm_expert_provider_result finish_background_request_locked(
            bool retain_predictive = false) noexcept {
        if (retain_predictive) {
            // Cache-policy terminals cannot cross graph-request boundaries.
            // Complete predictive H2D work while retaining the published
            // speculative resident for the next logical decode token.
            return normalize_background_terminals_locked(nullptr);
        }
        if (deterministic_policy_terminals) {
            return normalize_background_terminals_locked(nullptr);
        }
        while (active_background_flights != 0) {
            auto result = reap_background_locked(nullptr);
            if (!result.is_ready()) return result;
            if (active_background_flights == 0) {
                return llm_expert_provider_result::success();
            }
            background_promotion_record * earliest = nullptr;
            for (auto & record : background_promotions) {
                if (record.active && (!retain_predictive || !record.predictive) &&
                        (earliest == nullptr ||
                    record.origin_operation_ordinal < earliest->origin_operation_ordinal)) {
                    earliest = &record;
                }
            }
            if (earliest == nullptr && retain_predictive) {
                return llm_expert_provider_result::success();
            }
            if (earliest == nullptr) {
                return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
            }
            if (earliest->predictive) {
                cancel_prediction_sequence_locked(
                    earliest->prediction_sequence,
                    llm_expert_prefetch_outcome::cancelled_drained);
                continue;
            }
            result = transfer_ring->cancel_after_h2d(earliest->lane);
            if (!result.is_ready()) return result;
            earliest->state = background_promotion_record::lifecycle::cancelled;
            discard_background_slot_locked(*earliest, false);
            background_dropped++;
        }
        return llm_expert_provider_result::success();
    }

    llm_expert_provider_result select_optional_hot_slot_locked(
            const llm_expert_key & key,
            llm_expert_cache_policy_admission admission,
            int32_t & selected_slot) noexcept {
        auto capacity = cache_policy_result(hot_policy.validate_event_capacity(5));
        if (!capacity.is_ready()) return capacity;
        int32_t speculative_victim = -1;
        for (uint32_t slot = 0; slot < directory_slots.size(); ++slot) {
            const auto & entry = directory_slots[slot];
            const bool policy_free = hot_policy.validate_free(slot);
            const bool policy_loading = hot_policy.validate_loading(
                slot, entry.generation, policy_key_for(entry.key));
            const bool policy_ready = hot_policy.validate_resident(
                slot, entry.generation, policy_key_for(entry.key));
            const bool mechanism_loading = entry.state == hot_slot_state::loading;
            const bool mechanism_ready = entry.state == hot_slot_state::ready ||
                entry.state == hot_slot_state::pinned;
            if ((entry.state == hot_slot_state::free) != policy_free ||
                    mechanism_loading != policy_loading || mechanism_ready != policy_ready) {
                metadata_mismatches++;
                return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
            }
            const bool eligible_speculative =
                entry.origin == llm_expert_residency_origin::speculative &&
                !entry.background_useful;
            const bool same_scope = config.hot_cache_policy_config.scope !=
                    LLAMA_EXPERT_CACHE_POLICY_SCOPE_PER_LAYER || entry.key.layer == key.layer;
            if (admission == llm_expert_cache_policy_admission::optional_background &&
                    eligible_speculative && same_scope && entry.state == hot_slot_state::ready &&
                    entry.refcount == 0 && !slot_selected[slot] &&
                    entry.speculative_deadline != 0 &&
                    (speculative_victim < 0 ||
                     llm_expert_speculative_victim_precedes(
                         entry.speculative_deadline,
                         entry.speculative_utility,
                         slot,
                         directory_slots[uint32_t(speculative_victim)].speculative_deadline,
                         directory_slots[uint32_t(speculative_victim)].speculative_utility,
                         uint32_t(speculative_victim)))) {
                speculative_victim = int32_t(slot);
            }
            policy_candidate_slots[slot] = {
                slot,
                entry.generation,
                policy_key_for(entry.key),
                entry.state == hot_slot_state::free ? payload_for_key(key) : payload_for_key(entry.key),
                hot_physical_slot_footprint_bytes,
                entry.state == hot_slot_state::free && policy_free,
                entry.state == hot_slot_state::ready && policy_ready &&
                    entry.refcount == 0 && !slot_selected[slot] &&
                    (admission != llm_expert_cache_policy_admission::optional_background ||
                     eligible_speculative),
            };
        }
        if (admission == llm_expert_cache_policy_admission::optional_background) {
            for (uint32_t slot = 0; slot < directory_slots.size(); ++slot) {
                if (!policy_candidate_slots[slot].free) {
                    policy_candidate_slots[slot].eligible = int32_t(slot) == speculative_victim;
                }
            }
        }
        llm_expert_cache_policy_decision decision;
        auto result = cache_policy_result(hot_policy.optional_admission(
            policy_key_for(key), admission,
            policy_candidate_slots.data(), policy_candidate_slots.size(), decision));
        if (!result.is_ready() || !decision.accept) {
            selected_slot = -1;
            return result;
        }
        if (decision.slot >= directory_slots.size()) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
        }
        const auto & entry = directory_slots[decision.slot];
        const bool valid = decision.free ?
            entry.state == hot_slot_state::free && entry.generation == decision.generation :
            policy_candidate_slots[decision.slot].eligible && entry.generation == decision.generation &&
                hot_policy.validate_resident(decision.slot, decision.generation,
                    policy_key_for(entry.key));
        if (!valid) {
            metadata_mismatches++;
            return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
        }
        selected_slot = int32_t(decision.slot);
        return llm_expert_provider_result::success();
    }

    llm_expert_provider_result select_hot_slot_locked(
            const llm_expert_key & key,
            const uint32_t * selected_candidates,
            size_t selected_candidate_count,
            int32_t & selected_slot) noexcept {
        auto capacity = cache_policy_result(hot_policy.validate_event_capacity(5));
        if (!capacity.is_ready()) return capacity;
        for (uint32_t slot = 0; slot < directory_slots.size(); ++slot) {
            const auto & entry = directory_slots[slot];
            bool already_candidate = false;
            for (size_t index = 0; index < selected_candidate_count; ++index) {
                already_candidate = already_candidate || selected_candidates[index] == slot;
            }
            const bool excluded = slot_selected[slot] || already_candidate;
            const bool policy_free = hot_policy.validate_free(slot);
            const bool policy_loading = hot_policy.validate_loading(
                slot, entry.generation, policy_key_for(entry.key));
            const bool policy_ready = hot_policy.validate_resident(
                slot, entry.generation, policy_key_for(entry.key));
            const bool mechanism_loading = entry.state == hot_slot_state::loading;
            const bool mechanism_ready = entry.state == hot_slot_state::ready ||
                entry.state == hot_slot_state::pinned;
            if ((entry.state == hot_slot_state::free) != policy_free ||
                    mechanism_loading != policy_loading || mechanism_ready != policy_ready) {
                metadata_mismatches++;
                return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
            }
            const bool free = !excluded && entry.state == hot_slot_state::free;
            const bool eligible = !excluded && entry.state == hot_slot_state::ready && entry.refcount == 0;
            policy_candidate_slots[slot] = {
                slot,
                entry.generation,
                policy_key_for(entry.key),
                entry.state == hot_slot_state::free ? payload_for_key(key) : payload_for_key(entry.key),
                hot_physical_slot_footprint_bytes,
                free,
                eligible,
            };
        }
        llm_expert_cache_policy_decision decision;
        auto result = cache_policy_result(hot_policy.optional_admission(
            policy_key_for(key), llm_expert_cache_policy_admission::mandatory_current_output,
            policy_candidate_slots.data(), policy_candidate_slots.size(), decision));
        if (!result.is_ready()) return result;
        if (decision.slot >= directory_slots.size()) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
        }
        const auto & entry = directory_slots[decision.slot];
        const bool valid = decision.free ?
            entry.state == hot_slot_state::free && entry.generation == decision.generation :
            policy_candidate_slots[decision.slot].eligible && entry.generation == decision.generation &&
                hot_policy.validate_resident(decision.slot, decision.generation,
                    policy_key_for(entry.key));
        if (!valid) {
            metadata_mismatches++;
            return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
        }
        selected_slot = int32_t(decision.slot);
        return llm_expert_provider_result::success();
    }

    llm_expert_provider_result select_hot_slot_for_device_locked(
            const llm_expert_key & key,
            llm_expert_device_id target_device,
            const uint32_t * selected_candidates,
            size_t selected_candidate_count,
            int32_t & selected_slot) noexcept {
        LLM_EXPERT_TRACE_SCOPE("k3.provider", "multi_device_hot_slot_select",
            "target_device", target_device, "selected_candidate_count", selected_candidate_count);
        auto capacity = cache_policy_result(hot_policy.validate_event_capacity(5));
        if (!capacity.is_ready()) return capacity;
        size_t candidate_count = 0;
        {
            LLM_EXPERT_TRACE_SCOPE("k3.provider", "multi_device_hot_slot_candidate_build",
                "target_device", target_device);
            for (uint32_t slot = 0; slot < directory_slots.size(); ++slot) {
                const auto & entry = directory_slots[slot];
                const bool policy_free = hot_policy.validate_free(slot);
                const bool policy_loading = hot_policy.validate_loading(
                    slot, entry.generation, policy_key_for(entry.key));
                const bool policy_ready = hot_policy.validate_resident(
                    slot, entry.generation, policy_key_for(entry.key));
                const bool mechanism_loading = entry.state == hot_slot_state::loading;
                const bool mechanism_ready = entry.state == hot_slot_state::ready ||
                    entry.state == hot_slot_state::pinned;
                if ((entry.state == hot_slot_state::free) != policy_free ||
                    mechanism_loading != policy_loading || mechanism_ready != policy_ready) {
                    metadata_mismatches++;
                    return llm_expert_provider_result::failure(
                        llm_expert_provider_error::metadata_mismatch);
                }
                if (entry.device_id != target_device) continue;
                bool already_candidate = false;
                for (size_t index = 0; index < selected_candidate_count; ++index) {
                    already_candidate = already_candidate || selected_candidates[index] == slot;
                }
                const bool excluded = slot_selected[slot] || already_candidate;
                policy_candidate_slots[candidate_count++] = {
                    slot,
                    entry.generation,
                    policy_key_for(entry.key),
                    entry.state == hot_slot_state::free ? payload_for_key(key) : payload_for_key(entry.key),
                    hot_physical_slot_footprint_bytes,
                    !excluded && entry.state == hot_slot_state::free,
                    !excluded && entry.state == hot_slot_state::ready && entry.refcount == 0,
                };
            }
        }
        if (candidate_count == 0) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::busy);
        }
        llm_expert_cache_policy_decision decision;
        llm_expert_provider_result result;
        {
            LLM_EXPERT_TRACE_SCOPE("k3.provider", "multi_device_hot_slot_policy_admission",
                "target_device", target_device, "candidate_count", candidate_count);
            result = cache_policy_result(hot_policy.optional_admission(
                policy_key_for(key), llm_expert_cache_policy_admission::mandatory_current_output,
                policy_candidate_slots.data(), candidate_count, decision));
        }
        if (!result.is_ready()) return result;
        if (decision.slot >= directory_slots.size() ||
            directory_slots[decision.slot].device_id != target_device) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
        }
        const auto & entry = directory_slots[decision.slot];
        const bool valid = decision.free ?
            entry.state == hot_slot_state::free && entry.generation == decision.generation :
            entry.state == hot_slot_state::ready && entry.refcount == 0 &&
                entry.generation == decision.generation &&
                hot_policy.validate_resident(decision.slot, decision.generation,
                    policy_key_for(entry.key));
        if (!valid) {
            metadata_mismatches++;
            return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
        }
        if (!decision.free) {
            LLM_EXPERT_TRACE_SCOPE("k3.provider", "multi_device_hot_slot_feasibility_accounting",
                "target_device", target_device);
            const auto scan_started = std::chrono::steady_clock::now();
            for (uint32_t slot = 0; slot < directory_slots.size(); ++slot) {
                const auto & skipped = directory_slots[slot];
                if (skipped.device_id != target_device &&
                    skipped.state == hot_slot_state::ready && skipped.refcount == 0 &&
                    hot_policy.resident_precedes(slot, decision.slot)) {
                    physical_feasibility_skips++;
                }
            }
            const uint64_t scan_ns = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - scan_started).count());
            physical_feasibility_scan_calls++;
            physical_feasibility_scan_time_ns += scan_ns;
            physical_feasibility_scan_max_ns = std::max(physical_feasibility_scan_max_ns, scan_ns);
            if (hot_policy_phase == llm_expert_cache_policy_phase::decode) {
                physical_feasibility_scan_decode_calls++;
                physical_feasibility_scan_decode_time_ns += scan_ns;
                physical_feasibility_scan_decode_max_ns = std::max(
                    physical_feasibility_scan_decode_max_ns, scan_ns);
            }
        }
        selected_slot = int32_t(decision.slot);
        return llm_expert_provider_result::success();
    }

    llm_expert_transfer_ring * ring_for_slot(uint32_t slot) noexcept {
        if (slot >= directory_slots.size()) return nullptr;
        const auto device = directory_slots[slot].device_id;
        return device < transfer_rings.size() ? transfer_rings[device].get() : nullptr;
    }

    llm_expert_provider_result prepare_hot_slot_locked(
            uint32_t slot,
            const llm_expert_key & key,
            llm_cold_reference cold,
            bool demand_caused = true,
            bool require_cold_reference = true) noexcept {
        auto & entry = directory_slots[slot];
        const auto layout_class_id = layout_class_for_layer(key.layer);
        if (layout_class_id == LLM_EXPERT_LAYOUT_CLASS_INVALID ||
            (require_cold_reference && cold.layout_class_id != layout_class_id) ||
            payload_for_key(key) == 0) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
        }
        if (entry.state != hot_slot_state::free) {
            LLM_EXPERT_TRACE_INSTANT("k3.cache.hot", "victim", "slot_id", slot,
                "generation", entry.generation, "layer", entry.key.layer,
                "original_expert_id", entry.key.expert);
            const auto policy_valid = cache_policy_result(hot_policy.validate_evictable(slot, entry.generation));
            if (!policy_valid.is_ready()) return policy_valid;
            if (entry.origin == llm_expert_residency_origin::speculative &&
                !entry.background_useful) background_wasted++;
            auto * owner_ring = ring_for_slot(slot);
            if (owner_ring == nullptr) {
                return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
            }
            auto retired = owner_ring->retire_hot(entry.device_slot, entry.generation);
            if (!retired.is_ready()) return retired;
            if (entry.has_cold_backing) {
                auto released = cold_cache->release(
                    { entry.cold_slot, entry.cold_generation, entry.layout_class_id },
                    llm_cold_reference_kind::hot);
                if (!released.is_ready()) return released;
            }
            const auto removed = cache_policy_result(hot_policy.evict(slot, entry.generation));
            if (!removed.is_ready()) return removed;
            clear_forward_locked(entry.key, slot, entry.generation);
            evictions++;
            no_writeback_evictions++;
            if (entry.device_id < device_runtime_stats.size()) {
                device_runtime_stats[entry.device_id].evictions++;
            }
            LLM_EXPERT_TRACE_INSTANT("k3.cache.hot", "eviction", "slot_id", slot,
                "generation", entry.generation, "layer", entry.key.layer,
                "original_expert_id", entry.key.expert);
        }
        if (entry.generation == UINT64_MAX) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::generation_exhausted);
        }
        const uint64_t next_generation = entry.generation + 1;
        const llm_expert_device_id owner_device = entry.device_id;
        const uint32_t owner_slot = entry.device_slot;
        const bool replaced = entry.state != hot_slot_state::free;
        entry = {};
        if (replaced) {
            LLM_EXPERT_TRACE_COUNTER("k3.resource", "hot_cache_occupancy", 4,
                hot_cache_occupancy_locked());
        }
        entry.key = key;
        entry.device_id = owner_device;
        entry.device_slot = owner_slot;
        entry.layout_class_id = layout_class_id;
        entry.generation = next_generation;
        entry.state = hot_slot_state::loading;
        auto policy_loading = cache_policy_result(hot_policy.load_begin(
            slot, entry.generation, policy_key_for(key),
            payload_for_key(key), hot_physical_slot_footprint_bytes, demand_caused));
        if (!policy_loading.is_ready()) return policy_loading;
        generation_changes++;
        return llm_expert_provider_result::success();
    }

    void enqueue_background_locked(
            const llm_expert_key & key,
            llm_cold_reference cold) noexcept {
        if (!config.background_promotion) return;
        if (find_background_locked(key) != nullptr) {
            return;
        }
        const auto ring = transfer_ring->diagnostics();
        if (!ring.event_capable || config.scheduler == nullptr) {
            background_dropped++;
            return;
        }
        int32_t selected_slot = -1;
        const auto admitted = select_optional_hot_slot_locked(
            key, llm_expert_cache_policy_admission::optional_background, selected_slot);
        if (!admitted.is_ready()) {
            background_dropped++;
            return;
        }
        if (selected_slot < 0) {
            background_busy++;
            return;
        }
        llm_expert_request_metadata metadata;
        metadata.layout_class_id = layout_class_for_layer(key.layer);
        const auto scheduled = config.scheduler->enqueue(
            key, llm_expert_priority::demand_future_dependency,
            llm_expert_readiness::device_ready, metadata);
        if (scheduled.disposition != llm_expert_schedule_disposition::admitted) {
            if (scheduled.disposition == llm_expert_schedule_disposition::busy) background_busy++;
            else background_dropped++;
            return;
        }
        background_promotion_record record;
        record.state = background_promotion_record::lifecycle::queued_or_staging;
        // A local record does not own an active-flight count until it is
        // installed below. Cleanup after a failed ring submission therefore
        // must not decrement the count belonging to another live flight.
        record.active = false;
        record.key = key;
        record.cold = cold;
        record.scheduler_handle = scheduled.handle;
        record.scheduler_state = llm_expert_request_state::queued;
        record.hot_slot = uint32_t(selected_slot);
        record.speculative_deadline = request_ubatch_ordinal == UINT64_MAX ?
            UINT64_MAX : request_ubatch_ordinal + 1;
        if (record.speculative_deadline == 0) record.speculative_deadline = 1;
        record.speculative_utility = 0;

        llm_expert_request_snapshot selected;
        auto result = llm_expert_provider_result::success();
        const auto taken = config.scheduler->take_next(selected);
        if (taken.disposition != llm_expert_schedule_disposition::admitted ||
            selected.handle.slot != scheduled.handle.slot || selected.handle.generation != scheduled.handle.generation) {
            result = llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
        } else {
            record.scheduler_state = llm_expert_request_state::submitting;
        }
        if (result.is_ready()) result = prepare_hot_slot_locked(record.hot_slot, key, cold, false);
        if (result.is_ready()) {
            record.hot_generation = directory_slots[record.hot_slot].generation;
            record.origin_operation_ordinal = hot_policy.diagnostics().operation_ordinal;
            if (config.scheduler->transition(record.scheduler_handle,
                    llm_expert_request_state::submitting, llm_expert_request_state::host_ready) !=
                    llm_expert_schedule_disposition::admitted) {
                result = llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
            } else {
                record.scheduler_state = llm_expert_request_state::host_ready;
            }
        }
        if (result.is_ready() && config.scheduler->transition(record.scheduler_handle,
                llm_expert_request_state::host_ready, llm_expert_request_state::h2d_in_flight) !=
                llm_expert_schedule_disposition::admitted) {
            result = llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
        } else if (result.is_ready()) {
            record.scheduler_state = llm_expert_request_state::h2d_in_flight;
        }
        if (result.is_ready()) result = transfer_ring->try_queue_background_transfer(
            *cold_cache, cold, record.hot_slot, record.hot_generation,
            cold_cache->bundle(cold.layout_class_id), pool->bundles[cold.layout_class_id], record.lane,
            { config.async_transport ? config.async_transport->diagnostics().transport_epoch : 1,
              record.scheduler_handle.slot,
              record.scheduler_handle.generation, key, cold.layout_class_id },
            deterministic_policy_terminals);
        if (!result.is_ready()) {
            if (record.hot_generation != 0) {
                const auto ordered = finish_background_before_locked(record.origin_operation_ordinal);
                if (!ordered.is_ready()) result = ordered;
            }
            discard_background_slot_locked(record, false);
            if (result.error == llm_expert_provider_error::busy) background_busy++;
            else background_dropped++;
            return;
        }
        auto & slot_record = background_promotions[record.hot_slot];
        GGML_ASSERT(!slot_record.active);
        slot_record = record;
        slot_record.active = true;
        background_submitted++;
        const uint64_t payload = payload_for_key(key);
        background_h2d_bytes = payload > UINT64_MAX - background_h2d_bytes ?
            UINT64_MAX : background_h2d_bytes + payload;
        h2d_bytes = payload > UINT64_MAX - h2d_bytes ? UINT64_MAX : h2d_bytes + payload;
        active_background_flights++;
        peak_background_flights = std::max(peak_background_flights, active_background_flights);
    }

    void record_auto_decision_locked(const auto_decision_record & record) noexcept {
        const auto append = [&](uint64_t value) {
            for (size_t byte = 0; byte < sizeof(value); ++byte) {
                auto_decision_digest ^= (value >> (byte*8)) & 0xffU;
                auto_decision_digest *= 1099511628211ULL;
            }
        };
        append(record.request);
        append(uint32_t(record.layer)); append(uint32_t(record.expert));
        append(record.input.prefill); append(record.input.lanes); append(record.input.bundle_bytes);
        append(record.input.queued_cpu_work_ns); append(record.input.queued_h2d_work_ns);
        append(record.input.queued_gpu_work_ns); append(record.input.same_key_h2d_present);
        append(uint8_t(record.input.same_key_h2d_state));
        append(record.input.same_key_h2d_remaining_bytes);
        append(record.result.cpu_work_ns); append(record.result.h2d_work_ns);
        append(record.result.gpu_work_ns); append(record.result.cpu_finish_ns);
        append(record.result.gpu_finish_ns); append(uint8_t(record.result.backend));
        append(uint8_t(record.result.reason)); append(record.result.overflow);
        if (!auto_decisions.empty()) {
            auto_decisions[auto_decision_write++%auto_decisions.size()] = record;
            if (auto_decision_records >= auto_decisions.size()) auto_decision_records_dropped++;
        }
        auto_decision_records++;
    }

    bool validate_request_pins_locked() noexcept {
        for (size_t index = 0; index < request_pin_count; ++index) {
            const auto & pin = request_pins[index];
            if (pin.slot >= directory_slots.size() || directory_slots[pin.slot].generation != pin.generation ||
                directory_slots[pin.slot].refcount == 0 || directory_slots[pin.slot].state != hot_slot_state::pinned) {
                stale_generation_failures++;
                return false;
            }
        }
        for (size_t index = 0; index < cpu_execution_pin_count; ++index) {
            auto result = cold_cache->acquire(
                cpu_execution_pins[index], llm_cold_reference_kind::cpu_execution);
            if (result.is_ready()) {
                result = cold_cache->release(
                    cpu_execution_pins[index], llm_cold_reference_kind::cpu_execution);
            }
            if (!result.is_ready()) {
                stale_generation_failures++;
                return false;
            }
        }
        return true;
    }

    llm_expert_provider_result release_request_pins_locked() noexcept {
        if (!config.cpu_cold_only) {
            auto capacity = cache_policy_result(hot_policy.validate_event_capacity(request_pin_count));
            if (!capacity.is_ready()) return capacity;
        } else if (request_pin_count != 0) {
            return llm_expert_provider_result::failure(
                llm_expert_provider_error::metadata_mismatch);
        }
        if (cold_cache && cpu_execution_pin_count != 0) {
            const auto released = cold_cache->release_many(
                cpu_execution_pins.data(), cpu_execution_pin_count,
                llm_cold_reference_kind::cpu_execution);
            if (!released.is_ready()) return released;
            cpu_execution_pin_count = 0;
        }
        for (size_t index = 0; index < request_pin_count; ++index) {
            auto & entry = directory_slots[request_pins[index].slot];
            GGML_ASSERT(entry.generation == request_pins[index].generation && entry.refcount > 0);
            const auto unpinned = cache_policy_result(hot_policy.unpin(
                request_pins[index].slot, request_pins[index].generation));
            if (!unpinned.is_ready()) return unpinned;
            entry.refcount--;
            LLM_EXPERT_TRACE_INSTANT("k3.cache.hot", "unpin", "slot_id", request_pins[index].slot,
                "generation", request_pins[index].generation, "refcount", entry.refcount);
            if (entry.refcount == 0) {
                entry.state = hot_slot_state::ready;
            }
            pin_releases++;
            current_pins--;
        }
        request_pin_count = 0;
        return llm_expert_provider_result::success();
    }

    llm_expert_provider_result end_policy_request_locked(bool success, bool cancelled) noexcept {
        if (config.cpu_cold_only) {
            return cold_cache ? cold_cache->policy_request_end(success, cancelled) :
                llm_expert_provider_result::failure(
                    llm_expert_provider_error::initialization_failed);
        }
        auto result = cache_policy_result(hot_policy.request_end(success, cancelled));
        if (result.is_ready() && cold_cache) {
            result = cold_cache->policy_request_end(success, cancelled, config.async_cold_fill);
        }
        return result;
    }

    llm_expert_provider_result pin_slot_locked(uint32_t slot) noexcept {
        auto & entry = directory_slots[slot];
        GGML_ASSERT(request_pin_count < request_pins.size());
        GGML_ASSERT(entry.state == hot_slot_state::ready || entry.state == hot_slot_state::pinned);
        const auto pinned = cache_policy_result(hot_policy.pin(slot, entry.generation));
        if (!pinned.is_ready()) return pinned;
        entry.refcount++;
        LLM_EXPERT_TRACE_INSTANT("k3.cache.hot", "pin", "slot_id", slot,
            "generation", entry.generation, "refcount", entry.refcount);
        entry.state = hot_slot_state::pinned;
        entry.last_use = ++use_clock;
        request_pins[request_pin_count++] = { slot, entry.generation };
        pin_acquires++;
        current_pins++;
        peak_pins = std::max(peak_pins, current_pins);
        return llm_expert_provider_result::success();
    }

    uint32_t required_capacity(uint64_t extent) const noexcept {
        if (extent >= (uint64_t(n_expert) + config.n_expert_used - 1)/config.n_expert_used) {
            return n_expert;
        }
        return uint32_t(extent*config.n_expert_used);
    }

    bool extent_is_safe(uint64_t extent) const noexcept {
        return config.capacity >= n_expert || extent <= config.capacity/config.n_expert_used;
    }

    llm_expert_provider_result validate_tier_invariants_locked() noexcept {
        if (config.cpu_cold_only) {
            return cold_cache ? cold_cache->validate_invariants() :
                llm_expert_provider_result::failure(
                    llm_expert_provider_error::initialization_failed);
        }
        for (uint32_t slot = 0; slot < directory_slots.size(); ++slot) {
            const auto & entry = directory_slots[slot];
            const auto key = policy_key_for(entry.key);
            const bool policy_matches = entry.state == hot_slot_state::loading ?
                hot_policy.validate_loading(slot, entry.generation, key) :
                (entry.state == hot_slot_state::ready || entry.state == hot_slot_state::pinned) ?
                    hot_policy.validate_resident(slot, entry.generation, key) : true;
            const bool active = entry.state == hot_slot_state::loading ||
                entry.state == hot_slot_state::ready || entry.state == hot_slot_state::pinned;
            const bool class_matches = active ?
                entry.layout_class_id == layout_class_for_layer(entry.key.layer) :
                entry.layout_class_id == LLM_EXPERT_LAYOUT_CLASS_INVALID ||
                    entry.state == hot_slot_state::reserved || entry.state == hot_slot_state::evicting ||
                    entry.state == hot_slot_state::failed;
            if (!policy_matches || !class_matches) {
                metadata_mismatches++;
                return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
            }
        }
        if (!config.cold_mode || !cold_cache || !transfer_ring) {
            return config.cold_mode ?
                llm_expert_provider_result::failure(llm_expert_provider_error::initialization_failed) :
                llm_expert_provider_result::success();
        }
        hot_backing_scratch.clear();
        for (const auto & entry : directory_slots) {
            if (entry.state != hot_slot_state::loading && entry.state != hot_slot_state::ready &&
                entry.state != hot_slot_state::pinned) {
                continue;
            }
            if (entry.has_cold_backing) {
                hot_backing_scratch.push_back({
                    entry.key, entry.cold_slot, entry.cold_generation, entry.layout_class_id });
            }
        }
        auto result = cold_cache->validate_invariants(hot_backing_scratch);
        for (const auto & ring : transfer_rings) {
            if (!result.is_ready()) break;
            result = ring->validate_invariants();
        }
        return result;
    }

    llm_expert_layout_class_id layout_class_for_layer(int32_t layer) const noexcept {
        if (!layout_registry.sealed() || layer < 0 || size_t(layer) >= layout_registry.layer_ids.size()) {
            return LLM_EXPERT_LAYOUT_CLASS_INVALID;
        }
        return layout_registry.layer_ids[size_t(layer)];
    }

    llm_expert_cache_policy_key policy_key_for(llm_expert_key key) const noexcept {
        return { key.layer, key.expert, layout_class_for_layer(key.layer) };
    }

    uint64_t payload_for_key(llm_expert_key key) const noexcept {
        const auto class_id = layout_class_for_layer(key.layer);
        return class_id < layout_registry.classes.size() ? layout_registry.classes[class_id].payload_bytes : 0;
    }

    llm_expert_provider_result seal_layout_registry_locked() {
        if (layout_registry.sealed()) return llm_expert_provider_result::success();
        if (registrations.size() != config.routed_layer_count || registrations.empty()) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::initialization_failed);
        }
        struct candidate {
            int32_t layer = -1;
            llm_expert_bundle_descriptor bundle = {};
            std::vector<uint8_t> canonical;
            uint64_t digest = 0;
            uint64_t payload = 0;
        };
        std::vector<candidate> candidates;
        candidates.reserve(registrations.size());
        const auto & logical_prototype = registrations.begin()->second;
        for (const auto & registration : registrations) {
            if (!bundle_logical_signature_matches(logical_prototype, registration.second)) {
                return llm_expert_provider_result::failure(llm_expert_provider_error::unsupported_configuration);
            }
            candidate value;
            value.layer = registration.first;
            value.bundle = registration.second;
            if (!canonical_layout_encoding(value.bundle, config.target_buffer_type, value.canonical) ||
                !expert_bundle_payload_bytes(value.bundle, value.payload)) {
                return llm_expert_provider_result::failure(llm_expert_provider_error::invalid_descriptor);
            }
            value.digest = layout_encoding_digest(value.canonical);
            candidates.push_back(std::move(value));
        }
        std::sort(candidates.begin(), candidates.end(), [](const auto & lhs, const auto & rhs) {
            if (lhs.canonical != rhs.canonical) return lhs.canonical < rhs.canonical;
            return lhs.layer < rhs.layer;
        });
        llm_expert_layout_registry sealed;
        sealed.layer_ids.assign(LLAMA_MAX_LAYERS, LLM_EXPERT_LAYOUT_CLASS_INVALID);
        std::vector<std::vector<uint8_t>> unique_canonical;
        for (const auto & value : candidates) {
            if (unique_canonical.empty() || unique_canonical.back() != value.canonical) {
                if (unique_canonical.size() == LLM_EXPERT_LAYOUT_CLASS_MAX) {
                    return llm_expert_provider_result::failure(
                        llm_expert_provider_error::unsupported_configuration);
                }
                unique_canonical.push_back(value.canonical);
                const auto id = llm_expert_layout_class_id(unique_canonical.size() - 1);
                sealed.classes.push_back({ id, value.digest, value.payload, value.bundle });
            } else {
                const auto & existing = sealed.classes.back();
                if (existing.canonical_digest != value.digest || existing.payload_bytes != value.payload) {
                    return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
                }
            }
            sealed.layer_ids[size_t(value.layer)] = llm_expert_layout_class_id(unique_canonical.size() - 1);
        }
        for (const auto & registration : registrations) {
            if (sealed.layer_ids[size_t(registration.first)] == LLM_EXPERT_LAYOUT_CLASS_INVALID) {
                return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
            }
        }
        layout_registry = std::move(sealed);
        prototype = layout_registry.classes.front().prototype;
        return llm_expert_provider_result::success();
    }

    llm_expert_provider_result validate_source_bundle(const llm_expert_bundle_descriptor & bundle) const {
        auto result = bundle.validate();
        if (!result.is_ready()) {
            return result;
        }
        if (!config.cpu_cold_only && config.cold_mode &&
            config.miss_policy != LLAMA_EXPERT_MISS_POLICY_PROMOTE_AND_GPU &&
            !config.allow_non_cuda_target_for_testing) {
            for (const auto * projection : { &bundle.up, &bundle.gate, &bundle.gate_up, &bundle.down }) {
                if (projection->weight != nullptr && projection->weight->type != GGML_TYPE_F16 &&
                    projection->weight->type != GGML_TYPE_MXFP4) {
                    return llm_expert_provider_result::failure(llm_expert_provider_error::unsupported_configuration);
                }
            }
        }
        if (config.cold_mode && config.storage != nullptr) {
            return result;
        }
        if (config.cold_mode && config.descriptor_only_source_for_testing) {
            return result;
        }
        for (const auto * projection : { &bundle.up, &bundle.gate, &bundle.gate_up, &bundle.down }) {
            if (!projection_is_host_accessible(*projection) ||
                (config.cold_mode && !projection_is_pageable_cpu(*projection))) {
                return llm_expert_provider_result::failure(llm_expert_provider_error::unsupported_configuration);
            }
        }
        return result;
    }

    static void set_projection_buffer_type(llm_expert_projection_descriptor & projection) {
        if (projection.weight) {
            projection.buffer_type = ggml_backend_buffer_get_type(projection.weight->buffer);
        }
    }

    static bool pool_tensor_matches_source(
            const ggml_tensor * target,
            const ggml_tensor * source,
            int32_t n_expert,
            uint32_t capacity,
            bool weight) {
        if (target == nullptr || source == nullptr) {
            return target == source;
        }
        const int source_axis = expert_axis(source, n_expert, weight);
        if (source_axis < 0 || target->ne[source_axis] != capacity || source->type != target->type) {
            return false;
        }
        for (int axis = 0; axis < source_axis; ++axis) {
            if (source->ne[axis] != target->ne[axis] || source->nb[axis] != target->nb[axis]) {
                return false;
            }
        }
        return source->nb[source_axis] <= target->nb[source_axis];
    }

    static bool pool_projection_matches_source(
            const llm_expert_projection_descriptor & target,
            const llm_expert_projection_descriptor & source,
            int32_t n_expert,
            uint32_t capacity) {
        return pool_tensor_matches_source(target.weight, source.weight, n_expert, capacity, true) &&
            pool_tensor_matches_source(target.bias, source.bias, n_expert, capacity, false) &&
            pool_tensor_matches_source(target.scale, source.scale, n_expert, capacity, false);
    }

    static bool pool_layout_matches_source(
            const llm_expert_bundle_descriptor & target,
            const llm_expert_bundle_descriptor & source) {
        return pool_projection_matches_source(target.up, source.up, source.n_expert, target.n_expert) &&
            pool_projection_matches_source(target.gate, source.gate, source.n_expert, target.n_expert) &&
            pool_projection_matches_source(target.gate_up, source.gate_up, source.n_expert, target.n_expert) &&
            pool_projection_matches_source(target.down, source.down, source.n_expert, target.n_expert);
    }

    llm_expert_provider_result fail(llm_expert_provider_result result) const {
        counters.failures++;
        last_failure_error = result.error;
        if (result.status == llm_expert_provider_status::cancelled) {
            counters.cancellations++;
        }
        return result;
    }

    llm_hot_cache_config config;
    llm_expert_provider_faults faults;
    llm_expert_system_memory_budget system_memory_budget;
    uint64_t requested_cold_cache_bytes = 0;
    const bool deterministic_policy_terminals = false;
    llm_expert_cache_policy hot_policy;
    uint32_t n_expert = 0;
    mutable std::mutex mutex;
    // Keeps model-owned Phase 9 logical remaps ordered even while the provider
    // mutex is temporarily released for abort callbacks or storage waits.
    std::mutex ordered_remap_mutex;
    mutable llm_expert_provider_stats counters;
    std::map<int32_t, llm_expert_bundle_descriptor> registrations;
    std::optional<llm_expert_bundle_descriptor> prototype;
    llm_expert_layout_registry layout_registry;
    uint32_t layout_preflight_consumer_count = 0;
    bool layout_preflight_passed = false;
    std::shared_ptr<hot_pool_generation> pool;
    std::vector<std::shared_ptr<hot_pool_generation>> device_pools;
    std::unique_ptr<llm_cold_expert_cache> cold_cache;
    std::unique_ptr<llm_host_resident_demand_coordinator> host_resident_demand;
    llm_host_resident_demand_batch host_resident_batch;
    std::vector<std::unique_ptr<llm_expert_transfer_ring>> transfer_rings;
    std::vector<uint32_t> device_transfer_lane_capacities;
    std::vector<uint8_t> device_transfer_event_capabilities;
    llm_expert_transfer_ring * transfer_ring = nullptr;
    bool multi_device = false;
    struct device_runtime_counters {
        uint64_t hits = 0;
        uint64_t misses = 0;
        uint64_t admissions = 0;
        uint64_t evictions = 0;
        uint64_t h2d_bytes = 0;
    };
    std::vector<device_runtime_counters> device_runtime_stats;
    std::vector<uint64_t> device_transfer_delay_us;
    std::vector<bool> device_transfer_failure_for_testing;
    std::vector<bool> device_transfer_failure_decode_only_for_testing;
    uint64_t successful_bindings = 0;
    bool initialization_in_progress = false;
    bool initialization_failed = false;
    bool descriptor_discovery_complete = false;
    uint64_t descriptor_discovery_graphs = 0;
    uint64_t descriptor_discovery_bindings = 0;
    uint64_t descriptor_discovery_scheduler_reserve_calls = 0;
    uint64_t descriptor_discovery_backend_bytes_before = 0;
    uint64_t descriptor_discovery_backend_bytes_after = 0;
    uint64_t final_bootstrap_source_bindings = 0;
    uint64_t complete_deferred_payload_bytes = 0;
    bool complete_deferred_payload_in_compute_workspace = false;
    std::vector<uint64_t> scheduler_backend_bytes_before_discovery;
    std::vector<uint64_t> scheduler_backend_bytes_after_discovery;
    std::vector<uint64_t> scheduler_backend_bytes_after_hierarchy;
    std::vector<uint64_t> scheduler_backend_bytes_after_final_reserve;
    uint64_t epoch = 0;
    uint64_t generation = 0;
    uint32_t last_context_n_ctx = 0;
    uint32_t last_context_n_ubatch = 0;
    uint64_t request_ubatch_ordinal = 0;
    std::unique_ptr<std::array<llm_expert_phase10_lead_event, 256>> phase10_lead_events;
    size_t phase10_lead_event_count = 0;
    uint64_t phase10_lead_events_dropped = 0;
    std::unique_ptr<std::array<llm_expert_phase10_scheduler_event, 256>> phase10_scheduler_events;
    size_t phase10_scheduler_event_count = 0;
    uint64_t phase10_scheduler_events_dropped = 0;
    std::unique_ptr<std::array<llm_expert_phase10_issue_ahead_event, 256>> phase10_issue_ahead_trace;
    size_t phase10_issue_ahead_trace_count = 0;
    uint64_t phase10_issue_ahead_trace_dropped = 0;
    uint64_t phase10_issue_ahead_events = 0;
    uint64_t phase10_issue_ahead_violations = 0;
    uint64_t phase10_seed_attempts = 0;
    uint64_t phase10_seed_failures = 0;
    uint64_t phase10_seed_entries = 0;
    uint64_t phase10_seed_storage_bytes = 0;
    uint64_t phase10_seed_h2d_bytes = 0;
    llm_expert_key phase10_seed_last_touch = { -1, -1 };
    bool phase10_seed_complete = false;
    uint32_t last_context_extent = 0;
    uint32_t last_required_capacity = 0;
    uint64_t context_validations = 0;
    uint64_t context_rejections = 0;
    std::vector<hot_forward_entry> directory_forward;
    std::vector<hot_slot_entry> directory_slots;
    std::vector<llm_expert_key> unique_keys;
    std::vector<int32_t> unique_slots;
    std::vector<int32_t> unique_cpu_slots;
    std::vector<uint64_t> unique_lane_counts;
    std::vector<uint8_t> unique_gpu_assignment;
    std::vector<uint32_t> gpu_unique_indices;
    std::vector<uint32_t> miss_unique_indices;
    std::vector<uint32_t> candidate_slots;
    std::vector<llm_expert_cache_policy_candidate> policy_candidate_slots;
    std::vector<uint32_t> policy_unique_indices;
    std::vector<uint8_t> slot_selected;
    std::vector<llm_cold_reference> cold_references;
    std::vector<llm_transfer_lane_reference> transfer_lanes;
    std::vector<storage_async_flight> async_flights;
    std::vector<llm_expert_request_handle> async_handles;
    std::vector<llm_transfer_binding> transfer_bindings;
    std::vector<llm_cold_hot_backing> hot_backing_scratch;
    std::vector<int32_t> element_unique;
    std::vector<int32_t> logical_id_scratch;
    std::vector<int32_t> execution_id_scratch;
    std::vector<int32_t> cpu_execution_id_scratch;
    std::vector<std::vector<int32_t>> device_execution_id_scratch;
    std::vector<int32_t> last_logical_ids;
    std::vector<int32_t> last_execution_ids;
    std::vector<int32_t> last_cpu_execution_ids;
    size_t last_id_count = 0;
    int32_t last_remap_layer = -1;
    std::vector<hot_request_pin> request_pins;
    std::vector<llm_cold_reference> cpu_execution_pins;
    std::vector<background_promotion_record> background_promotions;
    llm_expert_prefetch_predictor prefetch_predictor;
    std::vector<predictive_storage_record> predictive_storage_records;
    std::vector<llm_expert_prefetch_candidate> predictor_candidates;
    std::vector<std::vector<int32_t>> current_token_routes;
    std::vector<llm_expert_phase10_prediction_event> phase10_prediction_trace;
    size_t phase10_prediction_trace_count = 0;
    uint64_t phase10_prediction_events = 0;
    uint64_t phase10_prediction_events_dropped = 0;
    std::vector<llm_expert_phase10_route_event> phase10_route_trace;
    std::vector<int32_t> phase10_route_ids;
    size_t phase10_route_trace_count = 0;
    size_t phase10_route_id_count = 0;
    uint64_t phase10_route_events = 0;
    uint64_t phase10_route_events_dropped = 0;
    uint64_t phase10_predictions_admitted = 0;
    uint64_t phase10_predictions_rejected = 0;
    uint64_t phase10_timely_useful = 0;
    uint64_t phase10_late_joined = 0;
    uint64_t phase10_wasted_unused = 0;
    uint64_t phase10_cancelled_before_io = 0;
    uint64_t phase10_cancelled_drained = 0;
    uint64_t phase10_predictor_compute_ns = 0;
    uint64_t phase10_predictor_digest = 1469598103934665603ULL;
    uint64_t phase10_circuit_opens = 0;
    uint32_t phase10_utility_window_resolved = 0;
    uint32_t phase10_utility_window_timely = 0;
    uint32_t phase10_utility_failed_windows = 0;
    uint32_t phase10_utility_min_timely_successes = 0;
    size_t current_token_next_layer = 0;
    uint64_t current_decode_token = 0;
    bool phase10_sequence_active = false;
    uint64_t phase10_sequence_owner = 0;
    uint64_t phase10_sequence_id = 0;
    uint64_t next_phase10_sequence_id = 0;
    bool phase10_circuit_open = false;
    bool phase10_runtime_failed = false;
    std::vector<auto_decision_record> auto_decisions;
    size_t auto_decision_write = 0;
    size_t request_pin_count = 0;
    size_t cpu_execution_pin_count = 0;
    bool active_request = false;
    uint64_t active_request_id = 0;
    uint64_t next_request_id = 0;
    uint64_t use_clock = 0;
    llm_expert_cache_policy_phase hot_policy_phase = llm_expert_cache_policy_phase::prefill;
    uint64_t hot_logical_bundle_bytes = 0;
    uint64_t hot_physical_slot_footprint_bytes = 0;
    uint64_t requests = 0;
    uint64_t exclusive_busy_failures = 0;
    uint64_t remap_checkpoints = 0;
    uint64_t logical_id_total = 0;
    uint64_t unique_id_total = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t gpu_execution_lanes = 0;
    uint64_t cpu_execution_lanes = 0;
    uint64_t mixed_execution_layers = 0;
    uint64_t cpu_fallback_unique_keys = 0;
    uint64_t h2d_bytes_avoided_for_current_output = 0;
    uint64_t auto_cpu_decisions = 0;
    uint64_t auto_gpu_decisions = 0;
    uint64_t auto_tie_decisions = 0;
    uint64_t auto_overflow_decisions = 0;
    uint64_t auto_decision_records = 0;
    uint64_t auto_decision_records_dropped = 0;
    uint64_t auto_decision_digest = 1469598103934665603ULL;
    uint64_t background_submitted = 0;
    uint64_t background_completed = 0;
    uint64_t background_useful = 0;
    uint64_t background_wasted = 0;
    uint64_t background_dropped = 0;
    uint64_t background_busy = 0;
    uint64_t background_later_joins = 0;
    uint64_t background_h2d_bytes = 0;
    uint32_t active_background_flights = 0;
    uint32_t peak_background_flights = 0;
    uint64_t admissions = 0;
    uint64_t evictions = 0;
    uint64_t no_writeback_evictions = 0;
    uint64_t physical_feasibility_skips = 0;
    uint64_t physical_feasibility_scan_calls = 0;
    uint64_t physical_feasibility_scan_time_ns = 0;
    uint64_t physical_feasibility_scan_max_ns = 0;
    uint64_t physical_feasibility_scan_decode_calls = 0;
    uint64_t physical_feasibility_scan_decode_time_ns = 0;
    uint64_t physical_feasibility_scan_decode_max_ns = 0;
    uint64_t provider_h2d_join_waves = 0;
    uint64_t provider_h2d_join_time_ns = 0;
    uint64_t provider_h2d_join_max_ns = 0;
    uint64_t provider_h2d_join_decode_waves = 0;
    uint64_t provider_h2d_join_decode_time_ns = 0;
    uint64_t provider_h2d_join_decode_max_ns = 0;
    uint64_t provider_h2d_async_decode_waves = 0;
    uint64_t provider_h2d_async_branch_waits = 0;
    uint64_t injected_device_failure_waves = 0;
    uint64_t injected_device_failure_participants = 0;
    uint64_t injected_device_failure_drained_waves = 0;
    uint64_t generation_changes = 0;
    uint64_t stale_generation_failures = 0;
    uint64_t metadata_mismatches = 0;
    uint64_t copy_failures = 0;
    uint64_t failed_cleanups = 0;
    uint64_t post_h2d_cancellations = 0;
    llm_expert_flight_id last_cancelled_flight;
    uint32_t last_cancelled_lane = UINT32_MAX;
    uint64_t last_cancelled_lane_generation = 0;
    uint32_t last_cancelled_hot_slot = UINT32_MAX;
    uint64_t last_cancelled_hot_generation = 0;
    llm_expert_flight_id last_completed_flight;
    uint32_t last_completed_lane = UINT32_MAX;
    uint64_t last_completed_lane_generation = 0;
    uint32_t last_completed_hot_slot = UINT32_MAX;
    uint64_t last_completed_hot_generation = 0;
    llm_expert_flight_id retry_after_cancel_flight;
    uint32_t retry_after_cancel_lane = UINT32_MAX;
    uint64_t retry_after_cancel_lane_generation = 0;
    uint32_t retry_after_cancel_hot_slot = UINT32_MAX;
    uint64_t retry_after_cancel_hot_generation = 0;
    int32_t last_execution_backend_device_type = -1;
    mutable llm_expert_provider_error last_failure_error = llm_expert_provider_error::none;
    llm_expert_provider_error last_remap_error = llm_expert_provider_error::none;
    uint64_t pin_acquires = 0;
    uint64_t pin_releases = 0;
    uint64_t current_pins = 0;
    uint64_t peak_pins = 0;
    uint64_t h2d_bytes = 0;
    uint64_t h2d_time_us = 0;
    uint64_t execution_id_read_bytes = 0;
    uint64_t execution_id_write_bytes = 0;
    uint64_t scratch_reservations = 0;
    uint64_t synchronization_checkpoints = 0;
    uint64_t hybrid_bindings = 0;
    uint64_t remote_single_bindings = 0;
    uint64_t multi_device_bindings = 0;
    uint64_t device_binding_vector_elements = 0;
    uint64_t cold_bundle_payload = 0;
    uint32_t transfer_lane_capacity = 0;
};

} // namespace

std::unique_ptr<llm_expert_weight_provider> llm_create_resident_expert_weight_provider(
        llm_expert_provider_faults faults) {
    return std::make_unique<llm_resident_expert_weight_provider>(faults);
}

std::unique_ptr<llm_expert_weight_provider> llm_create_hot_cache_expert_weight_provider(
        llm_hot_cache_config config,
        llm_expert_provider_faults faults) {
    return std::make_unique<llm_hot_cache_expert_weight_provider>(config, faults);
}

std::unique_ptr<llm_expert_weight_provider> llm_create_cold_cache_expert_weight_provider(
        llm_hot_cache_config config,
        llm_expert_provider_faults faults) {
    config.cold_mode = true;
    return std::make_unique<llm_hot_cache_expert_weight_provider>(config, faults);
}
