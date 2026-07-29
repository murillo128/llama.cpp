#include "llama-expert-weight-provider.h"

#include <atomic>
#include <new>
#include <stdexcept>
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
    if (provider_identity == nullptr || layer != selection.layer || execution_ids == nullptr) {
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
        auto result = bundle.validate();
        if (result.is_ready()) {
            result = selection.validate();
        }
        if (result.is_ready() && faults.binding != llm_expert_provider_error::none) {
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
        };
        return binding.validate(selection);
    }

    llm_expert_provider_result prepare(
            const std::vector<llm_expert_graph_binding> & bindings,
            llm_expert_execution_plan & plan) noexcept override {
        stats.prepare_calls.fetch_add(1, std::memory_order_relaxed);
        plan.reset();
        try {
            plan.reserve(bindings.size());
            for (size_t index = 0; index < bindings.size(); ++index) {
                if (bindings[index].provider_identity != this) {
                    const auto result = llm_expert_provider_result::failure(llm_expert_provider_error::invalid_binding);
                    record_failure(result);
                    plan.set_result(result);
                    return result;
                }
                if (index == faults.fail_preparation_after_handles) {
                    const auto error = faults.preparation == llm_expert_provider_error::none
                        ? llm_expert_provider_error::preparation_failed
                        : faults.preparation;
                    const auto result = llm_expert_provider_result::failure(error);
                    record_failure(result);
                    plan.set_result(result);
                    return result;
                }

                stats.handles_acquired.fetch_add(1, std::memory_order_relaxed);
                // The resident provider has one immutable model-lifetime binding, so all
                // request leases share one opaque token and compact into bounded storage.
                plan.add_handle({ this, 1 });
            }
        } catch (const std::bad_alloc &) {
            const auto result = llm_expert_provider_result::failure(llm_expert_provider_error::allocation_failed);
            record_failure(result);
            plan.set_result(result);
            return result;
        } catch (...) {
            const auto result = llm_expert_provider_result::failure(llm_expert_provider_error::preparation_failed);
            record_failure(result);
            plan.set_result(result);
            return result;
        }

        if (faults.preparation != llm_expert_provider_error::none &&
            faults.fail_preparation_after_handles == SIZE_MAX) {
            const auto result = llm_expert_provider_result::failure(faults.preparation);
            record_failure(result);
            plan.set_result(result);
            return result;
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
};

} // namespace

std::unique_ptr<llm_expert_weight_provider> llm_create_resident_expert_weight_provider(
        llm_expert_provider_faults faults) {
    return std::make_unique<llm_resident_expert_weight_provider>(faults);
}
