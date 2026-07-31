#include "llama-expert-weight-provider.h"
#include "llama-cold-expert-cache.h"
#include "llama-expert-storage.h"
#include "llama-expert-async-io.h"
#include "llama-expert-scheduler.h"
#include "llama-expert-transfer-ring.h"
#include "llama-hparams.h"

#include "ggml-alloc.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

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
    const uint64_t transfer_bytes = input.same_key_submitted_bytes != 0 ?
        input.same_key_submitted_bytes : input.bundle_bytes;
    result.h2d_work_ns = add(input.cost.h2d_fixed_ns, transfer(transfer_bytes));
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
        lhs.key.expert == rhs.key.expert;
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
    append(read.operation_index);
    append(transfer.flight.transport_epoch);
    append(transfer.flight.request_slot);
    append(transfer.flight.request_generation);
    append(uint32_t(transfer.flight.key.layer));
    append(uint32_t(transfer.flight.key.expert));
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
    std::unique_lock<std::mutex> * provider_lock = nullptr;
    llm_expert_request_handle completed_io_handle;
    bool completed_io_pending_publication = false;
    bool (*abort_callback)(void *) = nullptr;
    void * abort_callback_data = nullptr;
};

struct storage_async_flight {
    llm_expert_key key = { -1, -1 };
    llm_cold_reference cold;
    llm_expert_request_handle handle;
    llm_expert_flight_id flight_id;
    std::array<llm_expert_storage_destination, 12> destinations;
    std::array<llm_expert_storage_read_operation, 12> operations;
    size_t destination_count = 0;
    size_t operation_count = 0;
    bool cold_hit = false;
    bool reserved = false;
    bool submitted = false;
    bool read_active = false;
    bool scheduler_active = false;
    bool processed = false;
    llm_expert_request_state scheduler_state = llm_expert_request_state::free;
};

bool append_storage_destination(
        std::array<llm_expert_storage_destination, 12> & destinations,
        size_t & count,
        ggml_tensor * tensor,
        int32_t n_expert,
        uint32_t slot,
        bool weight,
        llm_expert_storage_projection projection,
        llm_expert_storage_sidecar sidecar) {
    if (tensor == nullptr) {
        return true;
    }
    const int axis = expert_axis(tensor, n_expert, weight);
    if (axis < 0 || tensor->data == nullptr || slot >= uint32_t(tensor->ne[axis]) || count >= destinations.size()) {
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
        tensor->nb[axis],
    };
    return true;
}

bool append_storage_projection(
        std::array<llm_expert_storage_destination, 12> & destinations,
        size_t & count,
        const llm_expert_projection_descriptor & projection,
        int32_t n_expert,
        uint32_t slot,
        llm_expert_storage_projection identity) {
    return append_storage_destination(destinations, count, projection.weight, n_expert, slot, true,
               identity, llm_expert_storage_sidecar::weight) &&
        append_storage_destination(destinations, count, projection.bias, n_expert, slot, false,
               identity, llm_expert_storage_sidecar::bias) &&
        append_storage_destination(destinations, count, projection.scale, n_expert, slot, false,
               identity, llm_expert_storage_sidecar::scale);
}

bool build_storage_destinations(
        storage_async_flight & flight,
        const llm_expert_bundle_descriptor & destination,
        uint32_t slot) {
    flight.destination_count = 0;
    return append_storage_projection(flight.destinations, flight.destination_count,
               destination.up, destination.n_expert, slot, llm_expert_storage_projection::up) &&
        append_storage_projection(flight.destinations, flight.destination_count,
               destination.gate, destination.n_expert, slot, llm_expert_storage_projection::gate) &&
        append_storage_projection(flight.destinations, flight.destination_count,
               destination.gate_up, destination.n_expert, slot, llm_expert_storage_projection::gate_up) &&
        append_storage_projection(flight.destinations, flight.destination_count,
               destination.down, destination.n_expert, slot, llm_expert_storage_projection::down);
}

llm_expert_provider_result load_storage_bundle(
        void * user_data,
        llm_expert_key key,
        const llm_expert_bundle_descriptor & destination,
        uint32_t slot) noexcept {
    auto * context = static_cast<storage_load_context *>(user_data);
    std::array<llm_expert_storage_destination, 12> destinations;
    size_t count = 0;
    if (context == nullptr || context->storage == nullptr ||
        !append_storage_projection(destinations, count, destination.up, destination.n_expert, slot,
            llm_expert_storage_projection::up) ||
        !append_storage_projection(destinations, count, destination.gate, destination.n_expert, slot,
            llm_expert_storage_projection::gate) ||
        !append_storage_projection(destinations, count, destination.gate_up, destination.n_expert, slot,
            llm_expert_storage_projection::gate_up) ||
        !append_storage_projection(destinations, count, destination.down, destination.n_expert, slot,
            llm_expert_storage_projection::down)) {
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
        const auto scheduled = context->scheduler->enqueue(
            key, llm_expert_priority::demand_current_layer, llm_expert_readiness::host_ready);
        if (scheduled.disposition != llm_expert_schedule_disposition::admitted) {
            return llm_expert_provider_result::failure(
                scheduled.disposition == llm_expert_schedule_disposition::generation_exhausted ?
                    llm_expert_provider_error::generation_exhausted : llm_expert_provider_error::busy);
        }
        llm_expert_request_snapshot snapshot;
        const auto selected = context->scheduler->take_next(snapshot);
        if (selected.disposition != llm_expert_schedule_disposition::admitted ||
            selected.handle.slot != scheduled.handle.slot || selected.handle.generation != scheduled.handle.generation) {
            if (selected.disposition == llm_expert_schedule_disposition::admitted) {
                (void) context->scheduler->transition(selected.handle, llm_expert_request_state::submitting,
                    llm_expert_request_state::draining);
                (void) context->scheduler->finish(selected.handle, llm_expert_request_state::failed);
                (void) context->scheduler->release_terminal(selected.handle);
            }
            if (selected.handle.slot != scheduled.handle.slot || selected.handle.generation != scheduled.handle.generation) {
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
            uint64_t destination_digest = 1469598103934665603ULL;
            for (size_t index = 0; index < count; ++index) {
                const auto * bytes = static_cast<const uint8_t *>(destinations[index].data);
                for (uint64_t offset = 0; offset < destinations[index].extent; ++offset) {
                    destination_digest ^= bytes[offset];
                    destination_digest *= 1099511628211ULL;
                }
            }
            integrity_matches = destination_digest == completion.digest;
        }
        if (waited == llm_expert_async_result::ready && released == llm_expert_async_result::ready &&
            integrity_matches) {
            context->storage->record_integrity_check(true);
            context->completed_io_handle = selected.handle;
            context->completed_io_pending_publication = true;
            return llm_expert_provider_result::success();
        }
        if (waited == llm_expert_async_result::ready && released == llm_expert_async_result::ready) {
            context->storage->record_integrity_check(false);
        }
        if (waited == llm_expert_async_result::closed) {
            (void) context->scheduler->transition(selected.handle, llm_expert_request_state::io_in_flight,
                llm_expert_request_state::cancelling);
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
            !integrity_matches ? llm_expert_provider_error::metadata_mismatch :
            (waited == llm_expert_async_result::stale_generation ?
                llm_expert_provider_error::stale_generation : llm_expert_provider_error::copy_failed));
    }
    const auto result = context->storage->read_bundle(
        key, destinations.data(), count, context->abort_callback, context->abort_callback_data);
    if (result.is_ready()) {
        uint64_t destination_digest = 1469598103934665603ULL;
        for (size_t index = 0; index < count; ++index) {
            const auto * bytes = static_cast<const uint8_t *>(destinations[index].data);
            for (uint64_t offset = 0; offset < destinations[index].extent; ++offset) {
                destination_digest ^= bytes[offset];
                destination_digest *= 1099511628211ULL;
            }
        }
        const bool matches = destination_digest == result.digest;
        context->storage->record_integrity_check(matches);
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
    if (hybrid) {
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
    } else if (checkpoint_ids != nullptr || cpu_execution_ids != nullptr || cpu_up.weight != nullptr || cpu_gate.weight != nullptr ||
               cpu_gate_up.weight != nullptr || cpu_down.weight != nullptr) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::invalid_binding);
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
            llm_expert_execution_plan & plan) noexcept override {
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
    llm_expert_bundle_descriptor bundle = {};
    std::vector<uintptr_t> addresses;
};

using hot_slot_state = llm_hot_cache_diagnostics::slot::state_type;

struct hot_forward_entry {
    int32_t slot = -1;
    uint64_t generation = 0;
};

struct hot_slot_entry {
    llm_expert_key key = { -1, -1 };
    uint64_t generation = 0;
    uint64_t last_use = 0;
    uint32_t refcount = 0;
    uint32_t cold_slot = 0;
    uint64_t cold_generation = 0;
    bool has_cold_backing = false;
    bool background_origin = false;
    bool background_useful = false;
    hot_slot_state state = hot_slot_state::free;
};

struct background_promotion_record {
    bool active = false;
    llm_expert_key key = { -1, -1 };
    llm_cold_reference cold;
    llm_transfer_lane_reference lane;
    llm_expert_request_handle scheduler_handle;
    llm_expert_request_state scheduler_state = llm_expert_request_state::free;
    uint32_t hot_slot = UINT32_MAX;
    uint64_t hot_generation = 0;
    uint64_t remaining_submitted_bytes = 0;
    uint64_t h2d_work_ns = 0;
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

struct deterministic_lru_policy {
    int32_t select_victim(
            const std::vector<hot_slot_entry> & slots,
            const std::vector<uint8_t> & selected,
            const std::vector<uint32_t> & candidates,
            size_t candidate_count,
            const std::vector<uint32_t> & releasable) const noexcept {
        int32_t victim = -1;
        for (uint32_t slot = 0; slot < slots.size(); ++slot) {
            const auto & entry = slots[slot];
            bool already_candidate = false;
            for (size_t index = 0; index < candidate_count; ++index) {
                already_candidate = already_candidate || candidates[index] == slot;
            }
            if (selected[slot] || already_candidate ||
                (entry.state != hot_slot_state::ready && entry.state != hot_slot_state::pinned) ||
                entry.refcount != releasable[slot]) {
                continue;
            }
            if (victim < 0 || entry.last_use < slots[victim].last_use ||
                (entry.last_use == slots[victim].last_use && slot < uint32_t(victim))) {
                victim = int32_t(slot);
            }
        }
        return victim;
    }
};

bool expert_key_matches(const llm_expert_key & lhs, const llm_expert_key & rhs) {
    return lhs.layer == rhs.layer && lhs.expert == rhs.expert;
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
    if (span != target->nb[source_axis]) {
        return false;
    }
    const auto * source_data = static_cast<const uint8_t *>(source->data) + size_t(expert)*span;
    ggml_backend_tensor_set(target, source_data, size_t(slot)*span, span);
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
        config(config), faults(faults) {
        if (faults.initialization != llm_expert_provider_error::none) {
            throw std::runtime_error("hot-cache expert-weight provider initialization failed");
        }
        if (config.capacity < config.n_expert_used || config.capacity > config.total_expert_keys ||
            config.n_expert_used == 0 || config.routed_layer_count == 0 || config.total_expert_keys == 0 ||
            config.total_expert_keys % config.routed_layer_count != 0 || config.target_buffer_type == nullptr) {
            throw std::invalid_argument("invalid hot-cache capacity or topology");
        }
        n_expert = config.total_expert_keys/config.routed_layer_count;
        if (config.n_expert_used > n_expert) {
            throw std::invalid_argument("invalid hot-cache expert topology");
        }
        if (!config.allow_non_cuda_target_for_testing && !buffer_type_is_cuda(config.target_buffer_type)) {
            throw std::invalid_argument("hot-cache target must be one CUDA device");
        }
        if (config.cold_mode && (config.cold_cache_bytes == 0 || config.transfer_ring_bytes == 0 ||
            config.target_device == nullptr || (!config.allow_non_cuda_target_for_testing && config.storage == nullptr))) {
            throw std::invalid_argument("cold-cache mode requires byte budgets and a target device");
        }
        if ((config.async_transport == nullptr) != (config.scheduler == nullptr) ||
            (config.async_transport != nullptr && config.storage == nullptr)) {
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
                                  config.auto_cost_model.version != 0 || config.auto_cost_model_digest != 0)) {
            throw std::invalid_argument("miss-policy configuration outside cold mode");
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
        std::lock_guard<std::mutex> lock(mutex);
        if (!transfer_ring) return;
        if (active_background_flights != 0) (void) transfer_ring->surrender();
        for (auto & record : background_promotions) {
            if (!record.active) continue;
            finish_background_scheduler_locked(record, true);
            if (record.hot_slot < directory_slots.size()) {
                auto & entry = directory_slots[record.hot_slot];
                if (entry.has_cold_backing) {
                    (void) cold_cache->release(
                        { entry.cold_slot, entry.cold_generation }, llm_cold_reference_kind::hot);
                }
                const uint64_t generation = entry.generation;
                entry = {};
                entry.generation = generation;
            }
            background_wasted++;
            record = {};
        }
        for (auto & entry : directory_slots) {
            if (entry.background_origin && !entry.background_useful) background_wasted++;
            if (entry.has_cold_backing) {
                (void) cold_cache->release(
                    { entry.cold_slot, entry.cold_generation }, llm_cold_reference_kind::hot);
                entry.has_cold_backing = false;
            }
        }
        active_background_flights = 0;
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
            result = validate_source_bundle(bundle);
            if (!result.is_ready()) {
                return fail(result);
            }
            if (prototype.has_value() && !bundle_layout_matches(*prototype, bundle)) {
                return fail(llm_expert_provider_result::failure(llm_expert_provider_error::invalid_descriptor));
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
            binding.up = bundle.up;
            binding.gate = bundle.gate;
            binding.gate_up = bundle.gate_up;
            binding.down = bundle.down;
            binding.execution_ids = selection.logical_ids;
            binding.graph_epoch = epoch;
            binding.bootstrap = true;
            counters.bootstrap_bindings++;
        } else {
            ggml_tensor * execution_ids = selection.logical_ids;
            ggml_tensor * checkpoint_ids = nullptr;
            ggml_tensor * cpu_execution_ids = nullptr;
            if (graph_ctx != nullptr) {
                if (config.cold_mode && config.miss_policy != LLAMA_EXPERT_MISS_POLICY_PROMOTE_AND_GPU) {
                    checkpoint_ids = ggml_dup(graph_ctx, selection.logical_ids);
                    ggml_format_name(checkpoint_ids, "expert_checkpoint_ids-%d", bundle.layer);
                    execution_ids = ggml_dup(graph_ctx, checkpoint_ids);
                    ggml_format_name(execution_ids, "expert_execution_ids-%d", bundle.layer);
                    cpu_execution_ids = ggml_new_tensor_2d(graph_ctx, GGML_TYPE_I32,
                        selection.n_expert_used, selection.n_tokens);
                    ggml_set_input(cpu_execution_ids);
                    ggml_format_name(cpu_execution_ids, "expert_cpu_execution_ids-%d", bundle.layer);
                } else {
                    execution_ids = ggml_dup(graph_ctx, selection.logical_ids);
                    ggml_format_name(execution_ids, "expert_execution_ids-%d", bundle.layer);
                }
            }
            binding = {};
            binding.provider_identity = this;
            binding.layer = bundle.layer;
            binding.up = pool->bundle.up;
            binding.gate = pool->bundle.gate;
            binding.gate_up = pool->bundle.gate_up;
            binding.down = pool->bundle.down;
            binding.execution_ids = execution_ids;
            binding.generation_lease = std::static_pointer_cast<void>(pool);
            binding.graph_epoch = epoch;
            counters.hot_bindings++;
            if (cpu_execution_ids != nullptr) {
                const auto & cpu = cold_cache->bundle();
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
        return config.cold_mode && config.miss_policy != LLAMA_EXPERT_MISS_POLICY_PROMOTE_AND_GPU;
    }

    llm_expert_provider_result prepare(
            const std::vector<llm_expert_graph_binding> & bindings,
            llm_expert_execution_plan & plan) noexcept override {
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
        } catch (const std::bad_alloc &) {
            auto result = llm_expert_provider_result::failure(llm_expert_provider_error::allocation_failed);
            plan.set_result(result);
            return fail(result);
        }

        active_request = true;
        active_request_id = ++next_request_id;
        request_pin_count = 0;
        cpu_execution_pin_count = 0;
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
        result.requested_capacity = config.capacity;
        result.effective_capacity = pool ? config.capacity : 0;
        result.pool_bytes = pool && pool->buffer ? ggml_backend_buffer_get_size(pool->buffer.get()) : 0;
        result.graph_epoch = epoch;
        return result;
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
        std::unique_lock<std::mutex> lock(mutex);
        return remap_checkpoint_locked(binding, logical_ids, logical_id_count, execution_ids, nullptr, nullptr,
            abort_callback, abort_callback_data, &lock);
    }

    llm_expert_provider_result remap_checkpoint_tensor(
            const llm_expert_graph_binding & binding,
            ggml_backend_t execution_backend,
            bool (*abort_callback)(void *),
            void * abort_callback_data) noexcept override {
        std::unique_lock<std::mutex> lock(mutex);
        (void) execution_backend;
        ggml_tensor * checkpoint_ids = binding.hybrid ? binding.checkpoint_ids : binding.execution_ids;
        if (checkpoint_ids == nullptr || binding.execution_ids == nullptr ||
            binding.execution_ids == binding.logical_ids || binding.execution_ids->type != GGML_TYPE_I32 ||
            binding.execution_ids->ne[0] < 0 || binding.execution_ids->ne[1] < 0 ||
            (binding.hybrid && binding.cpu_execution_ids == nullptr)) {
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
        auto result = remap_checkpoint_locked(
            binding, logical_id_scratch.data(), count, execution_id_scratch.data(),
            binding.hybrid ? cpu_execution_id_scratch.data() : nullptr, execution_backend,
            abort_callback, abort_callback_data, &lock);
        if (!result.is_ready()) {
            return result;
        }
        if (binding.hybrid) {
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

        if (!validate_request_pins_locked()) {
            return fail(llm_expert_provider_result::failure(llm_expert_provider_error::stale_generation));
        }
        if (config.cold_mode && config.background_promotion) {
            reap_background_locked();
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

        std::fill(slot_selected.begin(), slot_selected.end(), uint8_t(0));
        std::fill(slot_releasable.begin(), slot_releasable.end(), uint32_t(0));
        for (size_t index = 0; index < request_pin_count; ++index) {
            slot_releasable[request_pins[index].slot]++;
        }

        size_t miss_count = 0;
        for (size_t index = 0; index < unique_count; ++index) {
            const auto & key = unique_keys[index];
            const auto & forward = directory_forward[forward_index(key)];
            if (forward.slot < 0) {
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
            release_request_pins_locked();
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
                if (!entry.has_cold_backing) {
                    metadata_mismatches++;
                    return fail(llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch));
                }
                const llm_cold_reference backing = { entry.cold_slot, entry.cold_generation };
                auto touched = cold_cache->acquire(backing, llm_cold_reference_kind::request);
                if (touched.is_ready()) touched = cold_cache->release(backing, llm_cold_reference_kind::request);
                if (!touched.is_ready()) return fail(touched);
                touched = transfer_ring->wait_for_hot(execution_backend, uint32_t(unique_slots[index]), entry.generation);
                if (!touched.is_ready()) return fail(touched);
                pin_slot_locked(uint32_t(unique_slots[index]));
                if (entry.background_origin && !entry.background_useful) {
                    entry.background_useful = true;
                    background_useful++;
                }
                hit_count++;
            }

            storage_load_context storage_context = {
                config.storage, nullptr, nullptr, provider_lock, {}, false,
                abort_callback, abort_callback_data,
            };
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
                // Snapshot every active background transfer once.  An event-
                // incomplete submission retains its exact immutable payload;
                // queued staging work has no submitted bytes but still carries
                // one full transfer cost in another key's backlog.
                for (auto & record : background_promotions) {
                    if (!record.active) continue;
                    bool complete = false;
                    uint64_t remaining = 0;
                    const auto polled = transfer_ring->poll_h2d(record.lane, complete, &remaining);
                    if (!polled.is_ready()) {
                        (void) transfer_ring->release_failed_background(record.lane);
                        discard_background_slot_locked(record, false);
                        background_dropped++;
                        continue;
                    }
                    record.remaining_submitted_bytes = complete ? 0 : remaining;
                    record.h2d_work_ns = complete ? 0 : auto_h2d_work(config.auto_cost_model,
                        remaining != 0 ? remaining : cold_bundle_payload);
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
                        cold_bundle_payload,
                        queued_cpu_work,
                        saturating_add_work(queued_h2d_work, background_h2d_work),
                        queued_gpu_work,
                        background == nullptr ? 0 : background->remaining_submitted_bytes,
                    };
                    const auto decision = llm_evaluate_expert_auto(input);
                    record_auto_decision_locked({ active_request_id, binding.layer,
                        unique_keys[unique_index].expert, input, decision });
                    use_gpu = decision.backend == llm_expert_execution_backend::gpu;
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
                        const auto joined = config.scheduler->enqueue(
                            unique_keys[unique_index], llm_expert_priority::demand_current_layer,
                            llm_expert_readiness::device_ready);
                        if (joined.disposition != llm_expert_schedule_disposition::joined ||
                            joined.handle.slot != background->scheduler_handle.slot ||
                            joined.handle.generation != background->scheduler_handle.generation) {
                            result = llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
                        }
                        if (result.is_ready()) result = transfer_ring->wait_for_hot(
                            execution_backend, background->hot_slot, background->hot_generation);
                        if (!result.is_ready() && result.error == llm_expert_provider_error::copy_failed) {
                            (void) transfer_ring->release_failed_background(background->lane);
                            discard_background_slot_locked(*background, false);
                            background_dropped++;
                            result = cold_cache->acquire(reference, llm_cold_reference_kind::cpu_execution);
                            if (result.is_ready()) {
                                GGML_ASSERT(cpu_execution_pin_count < cpu_execution_pins.size());
                                cpu_execution_pins[cpu_execution_pin_count++] = reference;
                                unique_cpu_slots[unique_index] = int32_t(reference.slot);
                                cpu_miss_count++;
                            }
                            background = nullptr;
                        }
                        if (result.is_ready() && background != nullptr) {
                            auto & entry = directory_slots[background->hot_slot];
                            entry.state = hot_slot_state::ready;
                            entry.background_origin = true;
                            entry.background_useful = true;
                            entry.last_use = ++use_clock;
                            directory_forward[forward_index(entry.key)] = {
                                int32_t(background->hot_slot), entry.generation };
                            unique_slots[unique_index] = int32_t(background->hot_slot);
                            pin_slot_locked(background->hot_slot);
                            finish_background_scheduler_locked(*background, true);
                            background->active = false;
                            if (active_background_flights > 0) active_background_flights--;
                            background_completed++;
                            background_useful++;
                            background_later_joins++;
                            admissions++;
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
                release_request_pins_locked();
                last_remap_error = result.error;
                return fail(result);
            }

            size_t gpu_promotion_count = 0;
            transfer_bindings.clear();
            for (size_t index = 0; index < miss_count && result.is_ready(); ++index) {
                const uint32_t unique_index = miss_unique_indices[index];
                if (!unique_gpu_assignment[unique_index] || unique_slots[unique_index] >= 0) continue;
                const int32_t selected = select_unpinned_hot_slot_locked();
                if (selected < 0) {
                    result = llm_expert_provider_result::failure(llm_expert_provider_error::busy);
                    break;
                }
                const uint32_t hot_slot = uint32_t(selected);
                result = prepare_hot_slot_locked(
                    hot_slot, unique_keys[unique_index], cold_references[unique_index]);
                if (result.is_ready()) result = transfer_ring->reserve(
                    *cold_cache, cold_references[unique_index], hot_slot,
                    directory_slots[hot_slot].generation, transfer_lanes[unique_index]);
                if (result.is_ready()) result = transfer_ring->stage(
                    transfer_lanes[unique_index], cold_cache->bundle());
                if (result.is_ready()) {
                    gpu_unique_indices[gpu_promotion_count++] = unique_index;
                    unique_slots[unique_index] = int32_t(hot_slot);
                    transfer_bindings.push_back({ transfer_lanes[unique_index], pool->bundle, hot_slot });
                }
            }
            if (result.is_ready() && !transfer_bindings.empty()) {
                const uint64_t started_us = uint64_t(ggml_time_us());
                result = transfer_ring->transfer_wave(execution_backend, transfer_bindings);
                for (size_t index = 0; index < gpu_promotion_count && result.is_ready(); ++index) {
                    const uint32_t unique_index = gpu_unique_indices[index];
                    const uint32_t hot_slot = uint32_t(unique_slots[unique_index]);
                    result = transfer_ring->wait_for_hot(
                        execution_backend, hot_slot, directory_slots[hot_slot].generation);
                }
                if (result.is_ready()) {
                    for (size_t index = 0; index < gpu_promotion_count; ++index) {
                        const uint32_t unique_index = gpu_unique_indices[index];
                        const uint32_t hot_slot = uint32_t(unique_slots[unique_index]);
                        auto & entry = directory_slots[hot_slot];
                        entry.state = hot_slot_state::ready;
                        entry.last_use = ++use_clock;
                        directory_forward[forward_index(entry.key)] = { int32_t(hot_slot), entry.generation };
                        admissions++;
                        pin_slot_locked(hot_slot);
                    }
                    const uint64_t bytes = cold_bundle_payload > UINT64_MAX/gpu_promotion_count ? UINT64_MAX :
                        cold_bundle_payload*gpu_promotion_count;
                    h2d_bytes = bytes > UINT64_MAX - h2d_bytes ? UINT64_MAX : h2d_bytes + bytes;
                    h2d_time_us += uint64_t(ggml_time_us()) - started_us;
                }
            }
            if (!result.is_ready()) {
                (void) transfer_ring->cleanup_failed_lanes();
                for (size_t index = 0; index < gpu_promotion_count; ++index) {
                    const uint32_t unique_index = gpu_unique_indices[index];
                    auto & entry = directory_slots[uint32_t(unique_slots[unique_index])];
                    if (entry.has_cold_backing) {
                        (void) cold_cache->release(
                            { entry.cold_slot, entry.cold_generation }, llm_cold_reference_kind::hot);
                    }
                    const uint64_t generation = entry.generation;
                    entry = {};
                    entry.generation = generation;
                    unique_slots[unique_index] = -1;
                }
                release_request_pins_locked();
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
            result = llm_validate_hybrid_execution_ids(execution_ids, cpu_execution_ids, logical_id_count,
                int32_t(config.capacity), int32_t(cold_cache->diagnostics().effective_slots));
            if (!result.is_ready()) {
                release_request_pins_locked();
                return fail(result);
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
            const uint64_t remaining = UINT64_MAX - h2d_bytes_avoided_for_current_output;
            h2d_bytes_avoided_for_current_output =
                cpu_miss_count > remaining/cold_bundle_payload ? UINT64_MAX :
                h2d_bytes_avoided_for_current_output + cold_bundle_payload*cpu_miss_count;
            return llm_expert_provider_result::success();
        }

        size_t candidate_count = 0;
        for (uint32_t slot = 0; slot < config.capacity && candidate_count < miss_count; ++slot) {
            if (directory_slots[slot].state == hot_slot_state::free) {
                candidate_slots[candidate_count++] = slot;
            }
        }
        while (candidate_count < miss_count) {
            const int32_t victim = lru_policy.select_victim(
                directory_slots, slot_selected, candidate_slots, candidate_count, slot_releasable);
            if (victim < 0) {
                return fail(llm_expert_provider_result::failure(llm_expert_provider_error::busy));
            }
            candidate_slots[candidate_count++] = uint32_t(victim);
        }

        for (size_t index = 0; index < candidate_count; ++index) {
            if (directory_slots[candidate_slots[index]].generation == std::numeric_limits<uint64_t>::max()) {
                return fail(llm_expert_provider_result::failure(llm_expert_provider_error::generation_exhausted));
            }
        }

        release_request_pins_locked();

        size_t hit_count = 0;
        for (size_t index = 0; index < unique_count; ++index) {
            if (unique_slots[index] >= 0) {
                auto & entry = directory_slots[unique_slots[index]];
                if (config.cold_mode) {
                    if (!entry.has_cold_backing) {
                        metadata_mismatches++;
                        return fail(llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch));
                    }
                    const llm_cold_reference backing = { entry.cold_slot, entry.cold_generation };
                    auto touched = cold_cache->acquire(backing, llm_cold_reference_kind::request);
                    if (touched.is_ready()) touched = cold_cache->release(backing, llm_cold_reference_kind::request);
                    if (!touched.is_ready()) return fail(touched);
                    touched = transfer_ring->wait_for_hot(execution_backend, uint32_t(unique_slots[index]),
                        entry.generation);
                    if (!touched.is_ready()) return fail(touched);
                }
                pin_slot_locked(uint32_t(unique_slots[index]));
                hit_count++;
            }
        }

        for (size_t index = 0; index < miss_count; ++index) {
            const uint32_t unique_index = miss_unique_indices[index];
            const uint32_t slot = candidate_slots[index];
            auto & entry = directory_slots[slot];
            if (entry.state != hot_slot_state::free) {
                if (config.cold_mode) {
                    auto retired = transfer_ring->retire_hot(slot, entry.generation);
                    if (!retired.is_ready()) return fail(retired);
                    if (!entry.has_cold_backing) {
                        metadata_mismatches++;
                        return fail(llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch));
                    }
                    auto released = cold_cache->release(
                        { entry.cold_slot, entry.cold_generation }, llm_cold_reference_kind::hot);
                    if (!released.is_ready()) return fail(released);
                    entry.has_cold_backing = false;
                    entry.cold_slot = 0;
                    entry.cold_generation = 0;
                }
                entry.state = hot_slot_state::evicting;
                clear_forward_locked(entry.key, slot, entry.generation);
                evictions++;
                if (config.cold_mode) no_writeback_evictions++;
            }
            entry.state = hot_slot_state::reserved;
            entry.key = unique_keys[unique_index];
            entry.generation++;
            entry.refcount = 0;
            entry.has_cold_backing = false;
            generation_changes++;
            entry.state = hot_slot_state::loading;
            unique_slots[unique_index] = int32_t(slot);
        }

        size_t transaction_copies = 0;
        size_t transaction_bytes = 0;
        const int64_t copy_start_us = miss_count > 0 ? ggml_time_us() : 0;
        auto copy_result = llm_expert_provider_result::success();
        if (config.cold_mode && miss_count != 0 && cold_bundle_payload > SIZE_MAX/miss_count) {
            copy_result = llm_expert_provider_result::failure(llm_expert_provider_error::allocation_failed);
        }
        if (config.cold_mode) {
            if (config.storage && config.async_transport && config.scheduler) {
                size_t submitted_count = 0;
                for (size_t index = 0; index < miss_count && copy_result.is_ready(); ++index) {
                    auto & flight = async_flights[index];
                    flight = {};
                    const uint32_t unique_index = miss_unique_indices[index];
                    flight.key = unique_keys[unique_index];
                    bool cold_hit = false;
                    copy_result = cold_cache->reserve_or_find(flight.key, flight.cold, cold_hit);
                    flight.cold_hit = cold_hit;
                    flight.reserved = copy_result.is_ready() && !cold_hit;
                    cold_references[index] = flight.cold;
                    if (!copy_result.is_ready()) continue;
                    if (!cold_hit) {
                        if (!build_storage_destinations(flight, cold_cache->bundle(), flight.cold.slot)) {
                            copy_result = llm_expert_provider_result::failure(
                                llm_expert_provider_error::metadata_mismatch);
                            break;
                        }
                        const auto planned = config.storage->make_read_plan(
                            flight.key, flight.destinations.data(), flight.destination_count,
                            flight.operations.data(), flight.operations.size(), flight.operation_count);
                        if (!planned.is_ready()) {
                            copy_result = llm_expert_provider_result::failure(
                                llm_expert_provider_error::metadata_mismatch);
                            break;
                        }
                    }
                    const auto scheduled = config.scheduler->enqueue(
                        flight.key, llm_expert_priority::demand_current_layer,
                        llm_expert_readiness::device_ready);
                    if (scheduled.disposition != llm_expert_schedule_disposition::admitted) {
                        copy_result = llm_expert_provider_result::failure(
                            scheduled.disposition == llm_expert_schedule_disposition::generation_exhausted ?
                                llm_expert_provider_error::generation_exhausted : llm_expert_provider_error::busy);
                        break;
                    }
                    flight.handle = scheduled.handle;
                    flight.flight_id = {
                        config.async_transport->diagnostics().transport_epoch,
                        scheduled.handle.slot,
                        scheduled.handle.generation,
                        flight.key,
                    };
                    flight.scheduler_active = true;
                    flight.scheduler_state = llm_expert_request_state::queued;
                    llm_expert_request_snapshot selected;
                    const auto taken = config.scheduler->take_next(selected);
                    if (taken.disposition != llm_expert_schedule_disposition::admitted ||
                        selected.handle.slot != scheduled.handle.slot ||
                        selected.handle.generation != scheduled.handle.generation) {
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
                    flight.scheduler_state = llm_expert_request_state::submitting;
                    if (cold_hit) {
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
                    const llm_expert_async_operation_identity identity = {
                        config.async_transport->diagnostics().transport_epoch,
                        flight.handle, 0, flight.key, llm_expert_readiness::device_ready,
                        llm_expert_priority::demand_current_layer,
                    };
                    const auto submitted = config.async_transport->submit_read_plan(
                        identity, flight.operations.data(), flight.operation_count, true);
                    if (submitted != llm_expert_async_result::ready) {
                        copy_result = llm_expert_provider_result::failure(llm_expert_provider_error::copy_failed);
                        break;
                    }
                    flight.submitted = true;
                    flight.read_active = true;
                    async_handles[submitted_count] = flight.handle;
                    submitted_count++;
                    if (config.scheduler->transition(flight.handle, llm_expert_request_state::submitting,
                            llm_expert_request_state::io_in_flight) != llm_expert_schedule_disposition::admitted) {
                        copy_result = llm_expert_provider_result::failure(llm_expert_provider_error::copy_failed);
                        break;
                    }
                    flight.scheduler_state = llm_expert_request_state::io_in_flight;
                }
                if (submitted_count != 0) config.async_transport->start_deferred_reads();

                for (size_t completed_count = 0;
                        completed_count < miss_count && copy_result.is_ready(); ++completed_count) {
                    size_t index = miss_count;
                    for (size_t candidate = 0; candidate < miss_count; ++candidate) {
                        if (async_flights[candidate].cold_hit && !async_flights[candidate].processed) {
                            index = candidate;
                            break;
                        }
                    }
                    llm_expert_async_read_completion completion;
                    llm_expert_async_result waited = llm_expert_async_result::ready;
                    if (index == miss_count) {
                        llm_expert_request_handle completed_handle;
                        if (provider_lock != nullptr) provider_lock->unlock();
                        waited = config.async_transport->wait_any_read(
                            async_handles.data(), submitted_count, completed_handle, completion,
                            abort_callback, abort_callback_data);
                        if (provider_lock != nullptr) provider_lock->lock();
                        for (size_t candidate = 0; candidate < miss_count; ++candidate) {
                            const auto & candidate_flight = async_flights[candidate];
                            if (!candidate_flight.processed && candidate_flight.handle.slot == completed_handle.slot &&
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
                    }
                    auto & flight = async_flights[index];
                    flight.processed = true;
                    if (flight.submitted) {
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
                        uint64_t digest = 1469598103934665603ULL;
                        if (waited == llm_expert_async_result::ready && released == llm_expert_async_result::ready) {
                            for (size_t destination = 0; destination < flight.destination_count; ++destination) {
                                const auto * bytes = static_cast<const uint8_t *>(flight.destinations[destination].data);
                                for (uint64_t offset = 0; offset < flight.destinations[destination].extent; ++offset) {
                                    digest ^= bytes[offset];
                                    digest *= 1099511628211ULL;
                                }
                            }
                        }
                        const bool integrity_matches = waited == llm_expert_async_result::ready &&
                            released == llm_expert_async_result::ready && digest == completion.digest;
                        config.storage->record_integrity_check(integrity_matches);
                        if (!integrity_matches) {
                            copy_result = llm_expert_provider_result::failure(
                                waited == llm_expert_async_result::closed ? llm_expert_provider_error::cancelled :
                                llm_expert_provider_error::copy_failed);
                            break;
                        }
                        copy_result = cold_cache->publish_ready(flight.key, flight.cold);
                        if (copy_result.is_ready()) flight.reserved = false;
                        if (copy_result.is_ready() && config.scheduler->transition(flight.handle,
                                llm_expert_request_state::io_in_flight,
                                llm_expert_request_state::host_ready) != llm_expert_schedule_disposition::admitted) {
                            copy_result = llm_expert_provider_result::failure(
                                llm_expert_provider_error::metadata_mismatch);
                        }
                        if (copy_result.is_ready()) flight.scheduler_state = llm_expert_request_state::host_ready;
                    }
                    if (!copy_result.is_ready()) break;

                    const uint32_t slot = candidate_slots[index];
                    auto & entry = directory_slots[slot];
                    copy_result = cold_cache->acquire(flight.cold, llm_cold_reference_kind::hot);
                    if (copy_result.is_ready()) {
                        entry.cold_slot = flight.cold.slot;
                        entry.cold_generation = flight.cold.generation;
                        entry.has_cold_backing = true;
                    }
                    if (copy_result.is_ready() && flight.scheduler_active &&
                        config.scheduler->transition(flight.handle, llm_expert_request_state::host_ready,
                            llm_expert_request_state::h2d_in_flight) != llm_expert_schedule_disposition::admitted) {
                        copy_result = llm_expert_provider_result::failure(
                            llm_expert_provider_error::metadata_mismatch);
                    }
                    if (copy_result.is_ready() && flight.scheduler_active) {
                        flight.scheduler_state = llm_expert_request_state::h2d_in_flight;
                    }
                    if (copy_result.is_ready()) {
                        copy_result = transfer_ring->reserve(*cold_cache, flight.cold, slot,
                            entry.generation, transfer_lanes[index], flight.flight_id);
                    }
                    if (copy_result.is_ready()) {
                        copy_result = transfer_ring->stage(transfer_lanes[index], cold_cache->bundle());
                    }
                    if (copy_result.is_ready()) {
                        transfer_bindings.clear();
                        transfer_bindings.push_back({ transfer_lanes[index], pool->bundle, slot });
                        copy_result = transfer_ring->transfer_wave(execution_backend, transfer_bindings);
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
                            if (config.scheduler->transition(flight.handle,
                                    llm_expert_request_state::h2d_in_flight,
                                    llm_expert_request_state::cancelling) !=
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
                            execution_backend, slot, entry.generation);
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
            } else {
                storage_load_context storage_context = {
                    config.storage, nullptr, nullptr, provider_lock, {}, false,
                    abort_callback, abort_callback_data,
                };
                for (size_t index = 0; index < miss_count && copy_result.is_ready(); ++index) {
                    const uint32_t unique_index = miss_unique_indices[index];
                    const uint32_t slot = candidate_slots[index];
                    copy_result = config.storage ? cold_cache->find_or_admit_with_loader(
                        unique_keys[unique_index], cold_references[index], load_storage_bundle, &storage_context) :
                        cold_cache->find_or_admit(
                            unique_keys[unique_index], registration->second, cold_references[index],
                            faults.fail_copy_after_tensors);
                    if (copy_result.is_ready()) {
                        copy_result = cold_cache->acquire(cold_references[index], llm_cold_reference_kind::hot);
                    }
                    if (copy_result.is_ready()) {
                        auto & entry = directory_slots[slot];
                        entry.cold_slot = cold_references[index].slot;
                        entry.cold_generation = cold_references[index].generation;
                        entry.has_cold_backing = true;
                    }
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
                            copy_result = transfer_ring->stage(transfer_lanes[index], cold_cache->bundle());
                        }
                        if (copy_result.is_ready()) {
                            transfer_bindings.push_back({ transfer_lanes[index], pool->bundle, slot });
                        }
                    }
                    if (copy_result.is_ready()) {
                        copy_result = transfer_ring->transfer_wave(execution_backend, transfer_bindings);
                    }
                    if (copy_result.is_ready()) {
                        for (const auto & transfer : transfer_bindings) {
                            copy_result = transfer_ring->wait_for_hot(execution_backend,
                                transfer.hot_slot, directory_slots[transfer.hot_slot].generation);
                            if (!copy_result.is_ready()) break;
                        }
                    }
                }
            }
            if (copy_result.is_ready()) {
                transaction_bytes = size_t(cold_bundle_payload)*miss_count;
            }
        } else {
            bool copied = true;
            for (size_t index = 0; index < miss_count && copied; ++index) {
                const uint32_t unique_index = miss_unique_indices[index];
                const uint32_t slot = candidate_slots[index];
                const auto & key = unique_keys[unique_index];
                const auto & source = registration->second;
                copied = copy_expert_projection(pool->bundle.up, source.up, source.n_expert, config.capacity,
                             key.expert, slot, transaction_bytes, transaction_copies, faults.fail_copy_after_tensors) &&
                    copy_expert_projection(pool->bundle.gate, source.gate, source.n_expert, config.capacity,
                             key.expert, slot, transaction_bytes, transaction_copies, faults.fail_copy_after_tensors) &&
                    copy_expert_projection(pool->bundle.gate_up, source.gate_up, source.n_expert, config.capacity,
                             key.expert, slot, transaction_bytes, transaction_copies, faults.fail_copy_after_tensors) &&
                    copy_expert_projection(pool->bundle.down, source.down, source.n_expert, config.capacity,
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
                for (size_t index = 0; index < miss_count; ++index) {
                    auto & entry = directory_slots[candidate_slots[index]];
                    if (entry.has_cold_backing) {
                        (void) cold_cache->release(
                            { entry.cold_slot, entry.cold_generation }, llm_cold_reference_kind::hot);
                        entry.has_cold_backing = false;
                        entry.cold_slot = 0;
                        entry.cold_generation = 0;
                    }
                }
            }
            for (size_t index = 0; index < miss_count; ++index) {
                auto & entry = directory_slots[candidate_slots[index]];
                entry.state = hot_slot_state::failed;
                entry.refcount = 0;
            }
            release_request_pins_locked();
            faults.fail_copy_after_tensors = SIZE_MAX;
            copy_failures++;
            last_remap_error = copy_result.error;
            return fail(copy_result);
        }

        for (size_t index = 0; index < miss_count; ++index) {
            const uint32_t slot = candidate_slots[index];
            auto & entry = directory_slots[slot];
            entry.state = hot_slot_state::ready;
            directory_forward[forward_index(entry.key)] = { int32_t(slot), entry.generation };
            admissions++;
            pin_slot_locked(slot);
        }

        if (config.cold_mode) {
            auto invariant = validate_inclusive_locked();
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
                        { entry.cold_slot, entry.cold_generation }, llm_cold_reference_kind::hot);
                    if (!released.is_ready()) return fail(released);
                    entry.has_cold_backing = false;
                }
                entry.key = { -1, -1 };
                entry.refcount = 0;
                entry.state = hot_slot_state::free;
                failed_cleanups++;
            }
        }
        if (config.cold_mode) {
            auto result = transfer_ring->cleanup_failed_lanes();
            if (result.is_ready()) result = cold_cache->cleanup_failed_slots();
            if (result.is_ready()) result = validate_inclusive_locked();
            if (!result.is_ready()) return fail(result);
        }
        return llm_expert_provider_result::success();
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

    bool needs_post_reserve_initialization() const noexcept override {
        std::lock_guard<std::mutex> lock(mutex);
        return pool == nullptr;
    }

    llm_expert_provider_result initialize_after_reserve() noexcept override {
        std::lock_guard<std::mutex> lock(mutex);
        if (pool) {
            return llm_expert_provider_result::success();
        }
        if (!prototype.has_value() || registrations.size() != config.routed_layer_count) {
            return fail(llm_expert_provider_result::failure(llm_expert_provider_error::initialization_failed));
        }
        if (faults.fail_pool_allocation) {
            return fail(llm_expert_provider_result::failure(llm_expert_provider_error::allocation_failed));
        }

        try {
            ggml_init_params params = {
                /*.mem_size   =*/ ggml_tensor_overhead()*32,
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            auto candidate = std::make_shared<hot_pool_generation>();
            candidate->ctx.reset(ggml_init(params));
            if (!candidate->ctx) {
                return fail(llm_expert_provider_result::failure(llm_expert_provider_error::allocation_failed));
            }

            const auto & source = *prototype;
            candidate->bundle.layer = -1;
            candidate->bundle.n_expert = config.capacity;
            candidate->bundle.up = make_slot_projection(candidate->ctx.get(), source.up, source.n_expert, config.capacity, "hot.up");
            candidate->bundle.gate = make_slot_projection(candidate->ctx.get(), source.gate, source.n_expert, config.capacity, "hot.gate");
            candidate->bundle.gate_up = make_slot_projection(candidate->ctx.get(), source.gate_up, source.n_expert, config.capacity, "hot.gate_up");
            candidate->bundle.down = make_slot_projection(candidate->ctx.get(), source.down, source.n_expert, config.capacity, "hot.down");

            candidate->buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(
                candidate->ctx.get(), config.target_buffer_type));
            if (!candidate->buffer ||
                (!config.allow_non_cuda_target_for_testing && ggml_backend_buffer_is_host(candidate->buffer.get()))) {
                return fail(llm_expert_provider_result::failure(llm_expert_provider_error::allocation_failed));
            }

            set_projection_buffer_type(candidate->bundle.up);
            set_projection_buffer_type(candidate->bundle.gate);
            set_projection_buffer_type(candidate->bundle.gate_up);
            set_projection_buffer_type(candidate->bundle.down);
            if (!pool_layout_matches_source(candidate->bundle, source)) {
                return fail(llm_expert_provider_result::failure(llm_expert_provider_error::invalid_descriptor));
            }

            collect_projection_addresses(candidate->bundle.up, candidate->addresses);
            collect_projection_addresses(candidate->bundle.gate, candidate->addresses);
            collect_projection_addresses(candidate->bundle.gate_up, candidate->addresses);
            collect_projection_addresses(candidate->bundle.down, candidate->addresses);

            std::unique_ptr<llm_cold_expert_cache> cold_candidate;
            std::unique_ptr<llm_expert_transfer_ring> ring_candidate;
            if (config.cold_mode) {
                cold_candidate = std::make_unique<llm_cold_expert_cache>(llm_cold_cache_config {
                    config.cold_cache_bytes,
                    config.capacity,
                    config.routed_layer_count,
                    config.total_expert_keys,
                    0,
                });
                auto initialized = cold_candidate->initialize(source);
                if (!initialized.is_ready()) {
                    return fail(initialized);
                }
                ring_candidate = std::make_unique<llm_expert_transfer_ring>(llm_transfer_ring_config {
                    config.transfer_ring_bytes,
                    config.n_expert_used,
                    config.target_device,
                    config.allow_non_cuda_target_for_testing,
                    config.force_pageable_transfer_for_testing,
                    false,
                    0,
                    config.trace_capacity,
                });
                initialized = ring_candidate->initialize(source);
                if (!initialized.is_ready()) {
                    return fail(initialized);
                }
                const auto cold_diagnostics = cold_candidate->diagnostics();
                const auto ring_diagnostics = ring_candidate->diagnostics();
                cold_bundle_payload = cold_diagnostics.bundle_payload_bytes;
                transfer_lane_capacity = ring_diagnostics.effective_lanes;
                if (cold_bundle_payload == 0 || transfer_lane_capacity < config.n_expert_used) {
                    return fail(llm_expert_provider_result::failure(
                        llm_expert_provider_error::unsupported_configuration));
                }
            }

            directory_forward.assign(size_t(LLAMA_MAX_LAYERS)*n_expert, {});
            directory_slots.assign(config.capacity, {});
            for (auto & entry : directory_slots) {
                entry.generation = config.initial_slot_generation_for_testing;
            }
            unique_keys.resize(config.capacity);
            unique_slots.resize(config.capacity);
            unique_cpu_slots.resize(config.capacity);
            unique_lane_counts.resize(config.capacity);
            unique_gpu_assignment.resize(config.capacity);
            gpu_unique_indices.resize(config.capacity);
            miss_unique_indices.resize(config.capacity);
            candidate_slots.resize(config.capacity);
            slot_selected.resize(config.capacity);
            slot_releasable.resize(config.capacity);
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

            candidate->id = ++generation;
            pool = std::move(candidate);
            cold_cache = std::move(cold_candidate);
            transfer_ring = std::move(ring_candidate);
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
        std::lock_guard<std::mutex> lock(mutex);
        if (!pool) {
            counters.trims++;
            return llm_expert_provider_result::success();
        }
        if (config.cold_mode && (active_request || active_background_flights != 0)) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::busy);
        }
        if (config.cold_mode) {
            auto invariant = validate_inclusive_locked();
            if (!invariant.is_ready()) return fail(invariant);
        }
        for (uint32_t slot = 0; slot < directory_slots.size(); ++slot) {
            auto & entry = directory_slots[slot];
            if (entry.state == hot_slot_state::ready && entry.refcount == 0) {
                if (entry.background_origin && !entry.background_useful) background_wasted++;
                if (config.cold_mode && entry.has_cold_backing) {
                    auto retired = transfer_ring->retire_hot(slot, entry.generation);
                    if (!retired.is_ready()) return fail(retired);
                    auto released = cold_cache->release(
                        { entry.cold_slot, entry.cold_generation }, llm_cold_reference_kind::hot);
                    if (!released.is_ready()) return fail(released);
                    entry.has_cold_backing = false;
                    entry.cold_slot = 0;
                    entry.cold_generation = 0;
                }
                clear_forward_locked(entry.key, slot, entry.generation);
                entry.key = { -1, -1 };
                entry.state = hot_slot_state::free;
            }
        }
        if (config.cold_mode) {
            auto result = cold_cache->trim();
            if (result.is_ready()) result = validate_inclusive_locked();
            if (!result.is_ready()) return fail(result);
        }
        counters.trims++;
        return llm_expert_provider_result::success();
    }

    llm_expert_provider_result surrender() noexcept override {
        std::lock_guard<std::mutex> lock(mutex);
        if (!pool) {
            return llm_expert_provider_result::success();
        }
        if (active_request || active_background_flights != 0 || pool.use_count() != 1) {
            counters.surrender_busy++;
            return llm_expert_provider_result::failure(llm_expert_provider_error::busy);
        }
        if (config.cold_mode) {
            auto result = validate_inclusive_locked();
            if (!result.is_ready()) return fail(result);
            result = transfer_ring->surrender();
            if (!result.is_ready()) {
                counters.surrender_busy++;
                return result;
            }
            for (auto & entry : directory_slots) {
                if (entry.background_origin && !entry.background_useful) background_wasted++;
                if (entry.has_cold_backing) {
                    result = cold_cache->release(
                        { entry.cold_slot, entry.cold_generation }, llm_cold_reference_kind::hot);
                    if (!result.is_ready()) return fail(result);
                    entry.has_cold_backing = false;
                }
            }
            result = cold_cache->surrender();
            if (!result.is_ready()) return fail(result);
        }
        pool.reset();
        cold_cache.reset();
        transfer_ring.reset();
        directory_forward.clear();
        directory_slots.clear();
        unique_keys.clear();
        unique_slots.clear();
        unique_cpu_slots.clear();
        unique_lane_counts.clear();
        unique_gpu_assignment.clear();
        gpu_unique_indices.clear();
        miss_unique_indices.clear();
        candidate_slots.clear();
        slot_selected.clear();
        slot_releasable.clear();
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
        epoch++;
        counters.surrender_successes++;
        return llm_expert_provider_result::success();
    }

    uint64_t graph_epoch() const noexcept override {
        std::lock_guard<std::mutex> lock(mutex);
        return epoch;
    }

    llm_hot_cache_diagnostics hot_cache_diagnostics() const override {
        std::lock_guard<std::mutex> lock(mutex);
        llm_hot_cache_diagnostics result;
        result.configured_miss_policy = config.miss_policy;
        result.background_promotion_configured = config.background_promotion;
        result.auto_cost_model_version = config.auto_cost_model.version;
        result.auto_cost_model_digest = config.auto_cost_model_digest;
        result.hybrid_bindings = hybrid_bindings;
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
                record.input.same_key_submitted_bytes,
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
        result.requested_capacity = config.capacity;
        result.effective_capacity = pool ? config.capacity : 0;
        result.pool_bytes = pool && pool->buffer ? ggml_backend_buffer_get_size(pool->buffer.get()) : 0;
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
        result.synchronization_checkpoints = synchronization_checkpoints;
        result.slot_tensor_addresses = pool ? pool->addresses : std::vector<uintptr_t> {};
        result.slots.reserve(directory_slots.size());
        for (const auto & entry : directory_slots) {
            result.slots.push_back({
                entry.key.layer,
                entry.key.expert,
                entry.generation,
                entry.last_use,
                entry.refcount,
                entry.cold_slot,
                entry.cold_generation,
                entry.has_cold_backing,
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
            result.cold_pageable = cold.pageable;
            result.cold_requests = cold.requests;
            result.cold_hits = cold.hits;
            result.cold_misses = cold.misses;
            result.cold_admissions = cold.admissions;
            result.cold_evictions = cold.evictions;
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
            result.cold_current_cpu_execution_refs = cold.current_cpu_execution_refs;
            result.cold_peak_cpu_execution_refs = cold.peak_cpu_execution_refs;
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
        if (!config.cold_mode || !cold_cache || active_request ||
            !key.is_valid(LLAMA_MAX_LAYERS, n_expert)) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::unsupported_configuration);
        }
        const auto & forward = directory_forward[forward_index(key)];
        if (!forward_entry_matches(key, forward)) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::invalid_key);
        }
        const auto & entry = directory_slots[forward.slot];
        if (!entry.has_cold_backing) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
        }
        const auto cold_diagnostics = cold_cache->diagnostics();
        if (entry.cold_slot >= cold_diagnostics.slots.size()) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
        }
        const auto & cold_slot = cold_diagnostics.slots[entry.cold_slot];
        if (cold_slot.state != llm_cold_slot_state::ready || cold_slot.generation != entry.cold_generation ||
            cold_slot.key.layer != key.layer || cold_slot.key.expert != key.expert) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
        }
        std::array<llm_expert_storage_destination, 12> destinations;
        size_t count = 0;
        const auto & bundle = cold_cache->bundle();
        if (!append_storage_projection(destinations, count, bundle.up, bundle.n_expert, entry.cold_slot,
                llm_expert_storage_projection::up) ||
            !append_storage_projection(destinations, count, bundle.gate, bundle.n_expert, entry.cold_slot,
                llm_expert_storage_projection::gate) ||
            !append_storage_projection(destinations, count, bundle.gate_up, bundle.n_expert, entry.cold_slot,
                llm_expert_storage_projection::gate_up) ||
            !append_storage_projection(destinations, count, bundle.down, bundle.n_expert, entry.cold_slot,
                llm_expert_storage_projection::down)) {
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

    llm_expert_provider_result set_h2d_gate_event_for_testing(
            ggml_backend_event_t event) noexcept override {
        std::lock_guard<std::mutex> lock(mutex);
        if (!config.cold_mode || !transfer_ring || active_request) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::busy);
        }
        return transfer_ring->set_h2d_gate_event_for_testing(event);
    }

    llm_expert_provider_result debug_copy_hot_bundle(
            llm_expert_key key,
            std::vector<uint8_t> & bytes) const noexcept override {
        std::lock_guard<std::mutex> lock(mutex);
        bytes.clear();
        if (!pool || active_request || !key.is_valid(LLAMA_MAX_LAYERS, n_expert) ||
            directory_forward.empty()) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::unsupported_configuration);
        }
        const auto & forward = directory_forward[forward_index(key)];
        if (!forward_entry_matches(key, forward)) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::invalid_key);
        }
        const auto append_tensor = [&](ggml_tensor * tensor, bool weight) {
            if (tensor == nullptr) return true;
            const int axis = expert_axis(tensor, pool->bundle.n_expert, weight);
            if (axis < 0 || tensor->nb[axis] > SIZE_MAX || forward.slot < 0 ||
                uint32_t(forward.slot) >= uint32_t(tensor->ne[axis]) ||
                tensor->nb[axis] > SIZE_MAX - bytes.size()) return false;
            const size_t begin = bytes.size();
            bytes.resize(begin + tensor->nb[axis]);
            ggml_backend_tensor_get(tensor, bytes.data() + begin,
                size_t(forward.slot)*tensor->nb[axis], tensor->nb[axis]);
            return true;
        };
        try {
            for (const auto & projection : { pool->bundle.up, pool->bundle.gate,
                    pool->bundle.gate_up, pool->bundle.down }) {
                if (!append_tensor(projection.weight, true) ||
                    !append_tensor(projection.bias, false) ||
                    !append_tensor(projection.scale, false)) {
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
        std::lock_guard<std::mutex> lock(mutex);
        if (!active_request || lease_id != active_request_id || !validate_request_pins_locked()) {
            stale_generation_failures++;
        } else {
            release_request_pins_locked();
            active_request = false;
            active_request_id = 0;
            if (config.cold_mode && !validate_inclusive_locked().is_ready()) {
                metadata_mismatches++;
            }
        }
        counters.handles_released++;
    }

private:
    size_t forward_index(const llm_expert_key & key) const noexcept {
        return size_t(key.layer)*n_expert + uint32_t(key.expert);
    }

    bool binding_uses_current_pool(const llm_expert_graph_binding & binding) const noexcept {
        return projection_identity_matches(binding.up, pool->bundle.up) &&
            projection_identity_matches(binding.gate, pool->bundle.gate) &&
            projection_identity_matches(binding.gate_up, pool->bundle.gate_up) &&
            projection_identity_matches(binding.down, pool->bundle.down);
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
        if (record.hot_slot < directory_slots.size()) {
            auto & entry = directory_slots[record.hot_slot];
            if (entry.generation == record.hot_generation && expert_key_matches(entry.key, record.key)) {
                clear_forward_locked(entry.key, record.hot_slot, entry.generation);
                if (entry.has_cold_backing) {
                    (void) cold_cache->release(
                        { entry.cold_slot, entry.cold_generation }, llm_cold_reference_kind::hot);
                }
                const uint64_t generation = entry.generation;
                entry = {};
                entry.generation = generation;
            }
        }
        finish_background_scheduler_locked(record, false);
        if (record.active && active_background_flights > 0) active_background_flights--;
        if (count_wasted) background_wasted++;
        record = {};
    }

    void reap_background_locked() noexcept {
        if (!transfer_ring) return;
        for (auto & record : background_promotions) {
            if (!record.active) continue;
            bool complete = false;
            const auto polled = transfer_ring->poll_h2d(record.lane, complete);
            if (!polled.is_ready()) {
                (void) transfer_ring->release_failed_background(record.lane);
                discard_background_slot_locked(record, false);
                background_dropped++;
                continue;
            }
            if (!complete) continue;
            auto & entry = directory_slots[record.hot_slot];
            if (entry.state != hot_slot_state::loading || entry.generation != record.hot_generation ||
                !expert_key_matches(entry.key, record.key) || !entry.has_cold_backing) {
                discard_background_slot_locked(record, false);
                background_dropped++;
                continue;
            }
            entry.state = hot_slot_state::ready;
            entry.background_origin = true;
            entry.background_useful = false;
            directory_forward[forward_index(entry.key)] = { int32_t(record.hot_slot), entry.generation };
            admissions++;
            finish_background_scheduler_locked(record, true);
            record.active = false;
            record.scheduler_handle = {};
            if (active_background_flights > 0) active_background_flights--;
            background_completed++;
        }
    }

    int32_t select_unpinned_hot_slot_locked() const noexcept {
        for (uint32_t slot = 0; slot < directory_slots.size(); ++slot) {
            if (directory_slots[slot].state == hot_slot_state::free) return int32_t(slot);
        }
        int32_t victim = -1;
        for (uint32_t slot = 0; slot < directory_slots.size(); ++slot) {
            const auto & entry = directory_slots[slot];
            if (entry.state != hot_slot_state::ready || entry.refcount != 0 || slot_selected[slot]) continue;
            if (victim < 0 || entry.last_use < directory_slots[victim].last_use ||
                (entry.last_use == directory_slots[victim].last_use && slot < uint32_t(victim))) {
                victim = int32_t(slot);
            }
        }
        return victim;
    }

    llm_expert_provider_result prepare_hot_slot_locked(
            uint32_t slot,
            const llm_expert_key & key,
            llm_cold_reference cold) noexcept {
        auto & entry = directory_slots[slot];
        if (entry.state != hot_slot_state::free) {
            if (entry.background_origin && !entry.background_useful) background_wasted++;
            auto retired = transfer_ring->retire_hot(slot, entry.generation);
            if (!retired.is_ready()) return retired;
            if (!entry.has_cold_backing) {
                return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
            }
            auto released = cold_cache->release(
                { entry.cold_slot, entry.cold_generation }, llm_cold_reference_kind::hot);
            if (!released.is_ready()) return released;
            clear_forward_locked(entry.key, slot, entry.generation);
            evictions++;
            no_writeback_evictions++;
        }
        if (entry.generation == UINT64_MAX) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::generation_exhausted);
        }
        const uint64_t next_generation = entry.generation + 1;
        entry = {};
        entry.key = key;
        entry.generation = next_generation;
        entry.state = hot_slot_state::loading;
        auto acquired = cold_cache->acquire(cold, llm_cold_reference_kind::hot);
        if (!acquired.is_ready()) return acquired;
        entry.cold_slot = cold.slot;
        entry.cold_generation = cold.generation;
        entry.has_cold_backing = true;
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
        for (uint32_t slot = 0; slot < directory_slots.size(); ++slot) {
            if (directory_slots[slot].state == hot_slot_state::free) {
                selected_slot = int32_t(slot);
                break;
            }
        }
        if (selected_slot < 0) {
            background_busy++;
            return;
        }
        const auto scheduled = config.scheduler->enqueue(
            key, llm_expert_priority::demand_future_dependency, llm_expert_readiness::device_ready);
        if (scheduled.disposition != llm_expert_schedule_disposition::admitted) {
            if (scheduled.disposition == llm_expert_schedule_disposition::busy) background_busy++;
            else background_dropped++;
            return;
        }
        background_promotion_record record;
        record.active = true;
        record.key = key;
        record.cold = cold;
        record.scheduler_handle = scheduled.handle;
        record.scheduler_state = llm_expert_request_state::queued;
        record.hot_slot = uint32_t(selected_slot);

        llm_expert_request_snapshot selected;
        auto result = llm_expert_provider_result::success();
        const auto taken = config.scheduler->take_next(selected);
        if (taken.disposition != llm_expert_schedule_disposition::admitted ||
            selected.handle.slot != scheduled.handle.slot || selected.handle.generation != scheduled.handle.generation) {
            result = llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
        } else {
            record.scheduler_state = llm_expert_request_state::submitting;
        }
        if (result.is_ready()) result = prepare_hot_slot_locked(record.hot_slot, key, cold);
        if (result.is_ready()) {
            record.hot_generation = directory_slots[record.hot_slot].generation;
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
            cold_cache->bundle(), pool->bundle, record.lane,
            { config.async_transport ? config.async_transport->diagnostics().transport_epoch : 1,
              record.scheduler_handle.slot,
              record.scheduler_handle.generation, key });
        if (!result.is_ready()) {
            discard_background_slot_locked(record, false);
            if (result.error == llm_expert_provider_error::busy) background_busy++;
            else background_dropped++;
            return;
        }
        auto & slot_record = background_promotions[record.hot_slot];
        GGML_ASSERT(!slot_record.active);
        slot_record = record;
        background_submitted++;
        background_h2d_bytes = cold_bundle_payload > UINT64_MAX - background_h2d_bytes ?
            UINT64_MAX : background_h2d_bytes + cold_bundle_payload;
        h2d_bytes = cold_bundle_payload > UINT64_MAX - h2d_bytes ?
            UINT64_MAX : h2d_bytes + cold_bundle_payload;
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
        append(record.input.queued_gpu_work_ns); append(record.input.same_key_submitted_bytes);
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

    void release_request_pins_locked() noexcept {
        for (size_t index = 0; index < request_pin_count; ++index) {
            auto & entry = directory_slots[request_pins[index].slot];
            GGML_ASSERT(entry.generation == request_pins[index].generation && entry.refcount > 0);
            entry.refcount--;
            if (entry.refcount == 0) {
                entry.state = hot_slot_state::ready;
            }
            pin_releases++;
            current_pins--;
        }
        request_pin_count = 0;
        for (size_t index = 0; index < cpu_execution_pin_count; ++index) {
            const auto released = cold_cache->release(
                cpu_execution_pins[index], llm_cold_reference_kind::cpu_execution);
            GGML_ASSERT(released.is_ready());
        }
        cpu_execution_pin_count = 0;
    }

    void pin_slot_locked(uint32_t slot) noexcept {
        auto & entry = directory_slots[slot];
        GGML_ASSERT(request_pin_count < request_pins.size());
        GGML_ASSERT(entry.state == hot_slot_state::ready || entry.state == hot_slot_state::pinned);
        entry.refcount++;
        entry.state = hot_slot_state::pinned;
        entry.last_use = ++use_clock;
        request_pins[request_pin_count++] = { slot, entry.generation };
        pin_acquires++;
        current_pins++;
        peak_pins = std::max(peak_pins, current_pins);
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

    llm_expert_provider_result validate_inclusive_locked() noexcept {
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
            if (!entry.has_cold_backing) {
                metadata_mismatches++;
                return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
            }
            hot_backing_scratch.push_back({ entry.key, entry.cold_slot, entry.cold_generation });
        }
        auto result = cold_cache->validate_invariants(hot_backing_scratch);
        return result.is_ready() ? transfer_ring->validate_invariants() : result;
    }

    llm_expert_provider_result validate_source_bundle(const llm_expert_bundle_descriptor & bundle) const {
        auto result = bundle.validate();
        if (!result.is_ready()) {
            return result;
        }
        if (config.cold_mode && config.miss_policy != LLAMA_EXPERT_MISS_POLICY_PROMOTE_AND_GPU &&
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
        return source->nb[source_axis] == target->nb[source_axis];
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
    deterministic_lru_policy lru_policy;
    uint32_t n_expert = 0;
    mutable std::mutex mutex;
    mutable llm_expert_provider_stats counters;
    std::map<int32_t, llm_expert_bundle_descriptor> registrations;
    std::optional<llm_expert_bundle_descriptor> prototype;
    std::shared_ptr<hot_pool_generation> pool;
    std::unique_ptr<llm_cold_expert_cache> cold_cache;
    std::unique_ptr<llm_expert_transfer_ring> transfer_ring;
    uint64_t successful_bindings = 0;
    uint64_t epoch = 0;
    uint64_t generation = 0;
    uint32_t last_context_n_ctx = 0;
    uint32_t last_context_n_ubatch = 0;
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
    std::vector<uint8_t> slot_selected;
    std::vector<uint32_t> slot_releasable;
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
    std::vector<int32_t> last_logical_ids;
    std::vector<int32_t> last_execution_ids;
    std::vector<int32_t> last_cpu_execution_ids;
    size_t last_id_count = 0;
    int32_t last_remap_layer = -1;
    std::vector<hot_request_pin> request_pins;
    std::vector<llm_cold_reference> cpu_execution_pins;
    std::vector<background_promotion_record> background_promotions;
    std::vector<auto_decision_record> auto_decisions;
    size_t auto_decision_write = 0;
    size_t request_pin_count = 0;
    size_t cpu_execution_pin_count = 0;
    bool active_request = false;
    uint64_t active_request_id = 0;
    uint64_t next_request_id = 0;
    uint64_t use_clock = 0;
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
