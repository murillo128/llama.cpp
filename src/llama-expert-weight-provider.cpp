#include "llama-expert-weight-provider.h"
#include "llama-cold-expert-cache.h"
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

namespace {

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

        binding = {
            this,
            bundle.layer,
            bundle.up,
            bundle.gate,
            bundle.gate_up,
            bundle.down,
            selection.logical_ids,
            {},
            0,
            false,
        };
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
    hot_slot_state state = hot_slot_state::free;
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
            config.target_device == nullptr)) {
            throw std::invalid_argument("cold-cache mode requires byte budgets and a target device");
        }
        if (!config.cold_mode && (config.cold_cache_bytes != 0 || config.transfer_ring_bytes != 0 ||
            config.force_pageable_transfer_for_testing)) {
            throw std::invalid_argument("cold-cache configuration outside cold mode");
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
            binding = {
                this,
                bundle.layer,
                bundle.up,
                bundle.gate,
                bundle.gate_up,
                bundle.down,
                selection.logical_ids,
                {},
                epoch,
                true,
            };
            counters.bootstrap_bindings++;
        } else {
            ggml_tensor * execution_ids = selection.logical_ids;
            if (graph_ctx != nullptr) {
                execution_ids = ggml_dup(graph_ctx, selection.logical_ids);
                ggml_format_name(execution_ids, "expert_execution_ids-%d", bundle.layer);
            }
            binding = {
                this,
                bundle.layer,
                pool->bundle.up,
                pool->bundle.gate,
                pool->bundle.gate_up,
                pool->bundle.down,
                execution_ids,
                std::static_pointer_cast<void>(pool),
                epoch,
                false,
            };
            counters.hot_bindings++;
        }
        binding.logical_ids = selection.logical_ids;
        successful_bindings++;
        result = binding.validate(selection);
        return result.is_ready() ? result : fail(result);
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
                execution_id_scratch.size() < max_elements || last_logical_ids.size() < max_elements ||
                last_execution_ids.size() < max_elements) {
                element_unique.resize(max_elements);
                logical_id_scratch.resize(max_elements);
                execution_id_scratch.resize(max_elements);
                last_logical_ids.resize(max_elements);
                last_execution_ids.resize(max_elements);
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
            int32_t * execution_ids) noexcept override {
        std::lock_guard<std::mutex> lock(mutex);
        return remap_checkpoint_locked(binding, logical_ids, logical_id_count, execution_ids, nullptr);
    }

    llm_expert_provider_result remap_checkpoint_tensor(
            const llm_expert_graph_binding & binding,
            ggml_backend_t execution_backend) noexcept override {
        std::lock_guard<std::mutex> lock(mutex);
        (void) execution_backend;
        if (binding.execution_ids == nullptr || binding.execution_ids == binding.logical_ids ||
            binding.execution_ids->type != GGML_TYPE_I32 || binding.execution_ids->ne[0] < 0 ||
            binding.execution_ids->ne[1] < 0) {
            return fail(llm_expert_provider_result::failure(llm_expert_provider_error::invalid_binding));
        }
        const uint64_t count64 = uint64_t(binding.execution_ids->ne[0])*uint64_t(binding.execution_ids->ne[1]);
        if (count64 > logical_id_scratch.size() || count64 > execution_id_scratch.size()) {
            return fail(llm_expert_provider_result::failure(llm_expert_provider_error::unsupported_configuration));
        }
        const size_t count = size_t(count64);
        const size_t bytes = count*sizeof(int32_t);
        ggml_backend_tensor_get(binding.execution_ids, logical_id_scratch.data(), 0, bytes);
        execution_id_read_bytes += bytes;
        auto result = remap_checkpoint_locked(
            binding, logical_id_scratch.data(), count, execution_id_scratch.data(), execution_backend);
        if (!result.is_ready()) {
            return result;
        }
        ggml_backend_tensor_set(binding.execution_ids, execution_id_scratch.data(), 0, bytes);
        execution_id_write_bytes += bytes;
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
            ggml_backend_t execution_backend) noexcept {
        if (!active_request || !pool || logical_ids == nullptr || execution_ids == nullptr ||
            binding.provider_identity != this || binding.bootstrap || binding.graph_epoch != epoch ||
            binding.generation_lease.get() != pool.get() || !binding_uses_current_pool(binding) ||
            binding.execution_ids == nullptr || binding.execution_ids->ne[0] != int64_t(config.n_expert_used) ||
            binding.execution_ids->ne[1] < 0) {
            return fail(llm_expert_provider_result::failure(llm_expert_provider_error::invalid_binding));
        }

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
                unique_index = unique_count++;
            }
            element_unique[index] = int32_t(unique_index);
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
            for (size_t index = 0; index < miss_count && copy_result.is_ready(); ++index) {
                const uint32_t unique_index = miss_unique_indices[index];
                const uint32_t slot = candidate_slots[index];
                auto & entry = directory_slots[slot];
                copy_result = cold_cache->find_or_admit(
                    unique_keys[unique_index], registration->second, cold_references[index],
                    faults.fail_copy_after_tensors);
                if (copy_result.is_ready()) {
                    copy_result = cold_cache->acquire(cold_references[index], llm_cold_reference_kind::hot);
                }
                if (copy_result.is_ready()) {
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
                    0,
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
            miss_unique_indices.resize(config.capacity);
            candidate_slots.resize(config.capacity);
            slot_selected.resize(config.capacity);
            slot_releasable.resize(config.capacity);
            cold_references.resize(config.capacity);
            transfer_lanes.resize(config.capacity);
            transfer_bindings.clear();
            transfer_bindings.reserve(config.capacity);
            hot_backing_scratch.clear();
            hot_backing_scratch.reserve(config.capacity);
            request_pins.resize(config.capacity);
            element_unique.clear();
            logical_id_scratch.clear();
            execution_id_scratch.clear();
            last_logical_ids.clear();
            last_execution_ids.clear();
            last_id_count = 0;
            last_remap_layer = -1;
            request_pin_count = 0;
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
        if (config.cold_mode && active_request) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::busy);
        }
        if (config.cold_mode) {
            auto invariant = validate_inclusive_locked();
            if (!invariant.is_ready()) return fail(invariant);
        }
        for (uint32_t slot = 0; slot < directory_slots.size(); ++slot) {
            auto & entry = directory_slots[slot];
            if (entry.state == hot_slot_state::ready && entry.refcount == 0) {
                if (config.cold_mode && entry.has_cold_backing) {
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
        if (active_request || pool.use_count() != 1) {
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
        miss_unique_indices.clear();
        candidate_slots.clear();
        slot_selected.clear();
        slot_releasable.clear();
        cold_references.clear();
        transfer_lanes.clear();
        transfer_bindings.clear();
        hot_backing_scratch.clear();
        element_unique.clear();
        logical_id_scratch.clear();
        execution_id_scratch.clear();
        last_logical_ids.clear();
        last_execution_ids.clear();
        last_id_count = 0;
        last_remap_layer = -1;
        request_pins.clear();
        request_pin_count = 0;
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
            result.cold_invariant_failures = cold.invariant_failures;
            result.cold_current_hot_refs = cold.current_hot_refs;
            result.cold_peak_hot_refs = cold.peak_hot_refs;
            result.cold_current_transfer_refs = cold.current_transfer_refs;
            result.cold_peak_transfer_refs = cold.peak_transfer_refs;
            result.cold_current_request_refs = cold.current_request_refs;
            result.cold_peak_request_refs = cold.peak_request_refs;
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
            result.ring_async_enqueues = ring.async_enqueues;
            result.ring_synchronous_copies = ring.synchronous_copies;
            result.ring_waves = ring.waves;
            result.ring_peak_in_flight_lanes = ring.peak_in_flight_lanes;
            result.ring_wave_synchronizations = ring.wave_synchronizations;
            result.ring_h2d_bytes = ring.h2d_bytes;
            result.ring_failed_cleanup = ring.failed_cleanups;
        }
        return result;
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

    bool validate_request_pins_locked() noexcept {
        for (size_t index = 0; index < request_pin_count; ++index) {
            const auto & pin = request_pins[index];
            if (pin.slot >= directory_slots.size() || directory_slots[pin.slot].generation != pin.generation ||
                directory_slots[pin.slot].refcount == 0 || directory_slots[pin.slot].state != hot_slot_state::pinned) {
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
    std::vector<uint32_t> miss_unique_indices;
    std::vector<uint32_t> candidate_slots;
    std::vector<uint8_t> slot_selected;
    std::vector<uint32_t> slot_releasable;
    std::vector<llm_cold_reference> cold_references;
    std::vector<llm_transfer_lane_reference> transfer_lanes;
    std::vector<llm_transfer_binding> transfer_bindings;
    std::vector<llm_cold_hot_backing> hot_backing_scratch;
    std::vector<int32_t> element_unique;
    std::vector<int32_t> logical_id_scratch;
    std::vector<int32_t> execution_id_scratch;
    std::vector<int32_t> last_logical_ids;
    std::vector<int32_t> last_execution_ids;
    size_t last_id_count = 0;
    int32_t last_remap_layer = -1;
    std::vector<hot_request_pin> request_pins;
    size_t request_pin_count = 0;
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
    uint64_t admissions = 0;
    uint64_t evictions = 0;
    uint64_t no_writeback_evictions = 0;
    uint64_t generation_changes = 0;
    uint64_t stale_generation_failures = 0;
    uint64_t metadata_mismatches = 0;
    uint64_t copy_failures = 0;
    uint64_t failed_cleanups = 0;
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
