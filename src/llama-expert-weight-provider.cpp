#include "llama-expert-weight-provider.h"
#include "llama-hparams.h"

#include "ggml-alloc.h"
#include "ggml-cpp.h"

#include <array>
#include <atomic>
#include <cstring>
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
    }

    llm_expert_provider_result bind(
            const llm_expert_bundle_descriptor & bundle,
            const llm_expert_selection & selection,
            llm_expert_graph_binding & binding) noexcept override {
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
            binding = {
                this,
                bundle.layer,
                pool->bundle.up,
                pool->bundle.gate,
                pool->bundle.gate_up,
                pool->bundle.down,
                selection.logical_ids,
                std::static_pointer_cast<void>(pool),
                epoch,
                false,
            };
            counters.hot_bindings++;
        }
        successful_bindings++;
        result = binding.validate(selection);
        return result.is_ready() ? result : fail(result);
    }

    llm_expert_provider_result prepare(
            const std::vector<llm_expert_graph_binding> & bindings,
            llm_expert_execution_plan & plan) noexcept override {
        plan.reset();
        std::lock_guard<std::mutex> lock(mutex);
        counters.prepare_calls++;
        if (bindings.empty()) {
            plan.set_result(llm_expert_provider_result::success());
            return llm_expert_provider_result::success();
        }

        for (const auto & binding : bindings) {
            if (binding.provider_identity != this || binding.execution_ids == nullptr ||
                binding.execution_ids->ne[0] != int64_t(config.n_expert_used) || binding.execution_ids->ne[1] < 0) {
                auto result = llm_expert_provider_result::failure(llm_expert_provider_error::invalid_binding);
                plan.set_result(result);
                return fail(result);
            }
            if (!extent_is_safe(uint64_t(binding.execution_ids->ne[1]))) {
                auto result = llm_expert_provider_result::failure(llm_expert_provider_error::unsupported_configuration);
                plan.set_result(result);
                return fail(result);
            }
        }

        // Runtime remapping is introduced by issue Phase 3. Reject submission
        // until then so bootstrap or unpopulated pool tensors cannot execute.
        auto result = llm_expert_provider_result::failure(llm_expert_provider_error::preparation_failed);
        plan.set_result(result);
        return fail(result);
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
            candidate->id = ++generation;
            pool = std::move(candidate);
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
        counters.trims++;
        return llm_expert_provider_result::success();
    }

    llm_expert_provider_result surrender() noexcept override {
        std::lock_guard<std::mutex> lock(mutex);
        if (!pool) {
            return llm_expert_provider_result::success();
        }
        if (pool.use_count() != 1) {
            counters.surrender_busy++;
            return llm_expert_provider_result::failure(llm_expert_provider_error::busy);
        }
        pool.reset();
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
        result.slot_tensor_addresses = pool ? pool->addresses : std::vector<uintptr_t> {};
        result.source_buffer_type = prototype.has_value() ? prototype->down.buffer_type : nullptr;
        result.target_buffer_type = config.target_buffer_type;
        return result;
    }

protected:
    void release_handle(uint64_t) noexcept override {
        std::lock_guard<std::mutex> lock(mutex);
        counters.handles_released++;
    }

private:
    uint32_t required_capacity(uint64_t extent) const noexcept {
        if (extent >= (uint64_t(n_expert) + config.n_expert_used - 1)/config.n_expert_used) {
            return n_expert;
        }
        return uint32_t(extent*config.n_expert_used);
    }

    bool extent_is_safe(uint64_t extent) const noexcept {
        return config.capacity >= n_expert || extent <= config.capacity/config.n_expert_used;
    }

    llm_expert_provider_result validate_source_bundle(const llm_expert_bundle_descriptor & bundle) const {
        auto result = bundle.validate();
        if (!result.is_ready()) {
            return result;
        }
        for (const auto * projection : { &bundle.up, &bundle.gate, &bundle.gate_up, &bundle.down }) {
            if (!projection_is_host_accessible(*projection)) {
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
        const int target_axis = expert_axis(target, capacity, weight);
        if (source_axis < 0 || target_axis != source_axis || source->type != target->type) {
            return false;
        }
        for (int axis = 0; axis < source_axis; ++axis) {
            if (source->ne[axis] != target->ne[axis] || source->nb[axis] != target->nb[axis]) {
                return false;
            }
        }
        return source->nb[source_axis] == target->nb[target_axis];
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
    uint32_t n_expert = 0;
    mutable std::mutex mutex;
    mutable llm_expert_provider_stats counters;
    std::map<int32_t, llm_expert_bundle_descriptor> registrations;
    std::optional<llm_expert_bundle_descriptor> prototype;
    std::shared_ptr<hot_pool_generation> pool;
    uint64_t successful_bindings = 0;
    uint64_t epoch = 0;
    uint64_t generation = 0;
    uint32_t last_context_n_ctx = 0;
    uint32_t last_context_n_ubatch = 0;
    uint32_t last_context_extent = 0;
    uint32_t last_required_capacity = 0;
    uint64_t context_validations = 0;
    uint64_t context_rejections = 0;
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
