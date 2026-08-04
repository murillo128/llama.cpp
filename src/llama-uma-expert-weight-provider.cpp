#include "llama-expert-weight-provider.h"

#include "llama-cold-expert-cache.h"
#include "llama-expert-scheduler.h"
#include "llama-expert-storage.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <map>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <unordered_map>

namespace {

bool same_key(llm_expert_key lhs, llm_expert_key rhs) {
    return lhs.layer == rhs.layer && lhs.expert == rhs.expert;
}

int expert_axis(const ggml_tensor * tensor, int32_t n_expert, bool weight) {
    if (tensor == nullptr) return -1;
    if (weight) {
        if (ggml_n_dims(tensor) < 3 || tensor->ne[2] != n_expert) return -1;
        for (int axis = 3; axis < ggml_n_dims(tensor); ++axis) if (tensor->ne[axis] != 1) return -1;
        return 2;
    }
    if (ggml_n_dims(tensor) < 2 || tensor->ne[1] != n_expert) return -1;
    for (int axis = 2; axis < ggml_n_dims(tensor); ++axis) if (tensor->ne[axis] != 1) return -1;
    return 1;
}

struct uma_load_context {
    llm_expert_storage * storage = nullptr;
    llm_expert_scheduler * scheduler = nullptr;
    bool (*abort_callback)(void *) = nullptr;
    void * abort_data = nullptr;
    llm_expert_request_handle handle;
    bool active = false;
};

bool build_destinations(
        const llm_expert_bundle_descriptor & bundle,
        uint32_t slot,
        std::array<llm_expert_storage_destination, 12> & destinations,
        size_t & count) {
    count = 0;
    const std::array<std::pair<llm_expert_storage_projection,
        const llm_expert_projection_descriptor *>, 4> projections = {{
        { llm_expert_storage_projection::up, &bundle.up },
        { llm_expert_storage_projection::gate, &bundle.gate },
        { llm_expert_storage_projection::gate_up, &bundle.gate_up },
        { llm_expert_storage_projection::down, &bundle.down },
    }};
    for (const auto & [projection_id, projection] : projections) {
        size_t sidecar = 0;
        for (ggml_tensor * tensor : { projection->weight, projection->bias, projection->scale }) {
            if (tensor == nullptr) { sidecar++; continue; }
            const int axis = expert_axis(tensor, bundle.n_expert, sidecar == 0);
            if (axis <= 0 || slot >= uint32_t(bundle.n_expert) || count == destinations.size()) return false;
            const uint64_t extent = tensor->nb[axis - 1]*uint64_t(tensor->ne[axis - 1]);
            if (extent == 0 || extent > tensor->nb[axis]) return false;
            destinations[count++] = { projection_id,
                static_cast<llm_expert_storage_sidecar>(sidecar),
                static_cast<uint8_t *>(tensor->data) + uint64_t(slot)*tensor->nb[axis], extent };
            sidecar++;
        }
    }
    return count != 0;
}

llm_expert_provider_result fail_scheduler(uma_load_context & context, bool cancelled) {
    if (!context.active) return llm_expert_provider_result::failure(
        cancelled ? llm_expert_provider_error::cancelled : llm_expert_provider_error::copy_failed);
    auto & scheduler = *context.scheduler;
    llm_expert_request_snapshot snapshot;
    const auto snapshotted = scheduler.snapshot(context.handle, snapshot);
    if (snapshotted == llm_expert_schedule_disposition::admitted &&
        snapshot.state != llm_expert_request_state::draining) {
        (void) scheduler.begin_demand_cancellation(context.handle, snapshot.state);
        (void) scheduler.transition(context.handle, llm_expert_request_state::cancelling,
            llm_expert_request_state::draining);
    }
    (void) scheduler.finish(context.handle,
        cancelled ? llm_expert_request_state::cancelled : llm_expert_request_state::failed);
    (void) scheduler.release_terminal(context.handle);
    context.active = false;
    return llm_expert_provider_result::failure(
        cancelled ? llm_expert_provider_error::cancelled : llm_expert_provider_error::copy_failed);
}

llm_expert_provider_result load_uma_bundle(
        void * user_data,
        llm_expert_key key,
        const llm_expert_bundle_descriptor & destination,
        uint32_t slot) noexcept {
    auto & context = *static_cast<uma_load_context *>(user_data);
    if (context.storage == nullptr || context.scheduler == nullptr) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::initialization_failed);
    }
    const auto enqueued = context.scheduler->enqueue(
        key, llm_expert_priority::demand_current_layer, llm_expert_readiness::device_ready);
    if (enqueued.disposition != llm_expert_schedule_disposition::admitted) {
        return llm_expert_provider_result::failure(enqueued.disposition == llm_expert_schedule_disposition::busy ?
            llm_expert_provider_error::busy : llm_expert_provider_error::metadata_mismatch);
    }
    context.handle = enqueued.handle;
    context.active = true;
    llm_expert_request_snapshot selected;
    const auto taken = context.scheduler->take_next(selected);
    if (taken.disposition != llm_expert_schedule_disposition::admitted ||
        selected.handle.slot != context.handle.slot || selected.handle.generation != context.handle.generation ||
        context.scheduler->transition(context.handle, llm_expert_request_state::submitting,
            llm_expert_request_state::io_in_flight) != llm_expert_schedule_disposition::admitted) {
        return fail_scheduler(context, false);
    }
    std::array<llm_expert_storage_destination, 12> destinations;
    size_t destination_count = 0;
    if (!build_destinations(destination, slot, destinations, destination_count)) {
        return fail_scheduler(context, false);
    }
    const auto read = context.storage->read_bundle(key, destinations.data(), destination_count,
        context.abort_callback, context.abort_data);
    if (!read.is_ready()) return fail_scheduler(context, read.error == llm_expert_storage_error::cancelled);
    uint64_t destination_digest = UINT64_C(1469598103934665603);
    for (size_t index = 0; index < destination_count; ++index) {
        const auto * bytes = static_cast<const uint8_t *>(destinations[index].data);
        for (uint64_t offset = 0; offset < destinations[index].extent; ++offset) {
            destination_digest ^= bytes[offset];
            destination_digest *= UINT64_C(1099511628211);
        }
    }
    const bool integrity_matches = destination_digest == read.digest;
    context.storage->record_integrity_check(integrity_matches);
    if (!integrity_matches) return fail_scheduler(context, false);
    return llm_expert_provider_result::success();
}

class llm_uma_expert_weight_provider final : public llm_expert_weight_provider {
public:
    llm_uma_expert_weight_provider(llm_uma_cache_config config, llm_expert_provider_faults faults) :
        config(std::move(config)), faults(faults) {
        if (faults.initialization != llm_expert_provider_error::none || this->config.pool_bytes == 0 ||
            this->config.hot_capacity < this->config.n_expert_used || this->config.n_expert_used == 0 ||
            this->config.routed_layer_count == 0 || this->config.total_expert_keys == 0 ||
            this->config.total_expert_keys % this->config.routed_layer_count != 0 ||
            this->config.buffer_type == nullptr || this->config.is_uma_buffer_type == nullptr ||
            !this->config.is_uma_buffer_type(this->config.buffer_type) ||
            this->config.target_device == nullptr || this->config.storage == nullptr || this->config.scheduler == nullptr ||
            this->config.prefetch == nullptr || this->config.checksum == nullptr ||
            this->config.routed_layers.size() != this->config.routed_layer_count) {
            throw std::invalid_argument("invalid UMA expert provider configuration");
        }
        n_expert = this->config.total_expert_keys/this->config.routed_layer_count;
        if (this->config.n_expert_used > n_expert) throw std::invalid_argument("invalid UMA expert topology");
        if (this->config.hot_cache_policy_config.digest == 0 && !llm_expert_cache_policy_copy_config(
                nullptr, llm_expert_cache_policy_tier::hot,
                this->config.hot_cache_policy_config).is_ready()) {
            throw std::invalid_argument("invalid UMA hot policy");
        }
        if (this->config.cold_cache_policy_config.digest == 0 && !llm_expert_cache_policy_copy_config(
                nullptr, llm_expert_cache_policy_tier::cold,
                this->config.cold_cache_policy_config).is_ready()) {
            throw std::invalid_argument("invalid UMA cold policy");
        }
    }

    ~llm_uma_expert_weight_provider() override {
        std::lock_guard<std::mutex> lock(mutex);
        release_pins_locked();
        if (cache) (void) cache->surrender();
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

    llm_expert_provider_result prepare(
            const std::vector<llm_expert_graph_binding> & bindings,
            llm_expert_execution_plan & plan,
            uint64_t,
            bool) noexcept override {
        plan.reset();
        std::lock_guard<std::mutex> lock(mutex);
        if (!cache || active_request || bindings.empty()) return set_plan_failure(plan,
            llm_expert_provider_error::busy);
        for (const auto & binding : bindings) {
            if (binding.provider_identity != this || binding.graph_epoch != epoch || binding.bootstrap) {
                return set_plan_failure(plan, llm_expert_provider_error::invalid_binding);
            }
        }
        auto started = cache->policy_request_begin();
        if (!started.is_ready()) return set_plan_failure(plan, started.error);
        active_request = true;
        const uint64_t lease = ++lease_id;
        try {
            plan.reserve(1);
            plan.add_handle(llm_expert_handle(this, lease));
        } catch (...) {
            active_request = false;
            (void) cache->policy_request_end(false, false);
            return set_plan_failure(plan, llm_expert_provider_error::allocation_failed);
        }
        stats.prepare_calls++;
        stats.handles_acquired++;
        return llm_expert_provider_result::success();
    }

    llm_expert_provider_result remap_checkpoint(
            const llm_expert_graph_binding & binding,
            const int32_t * logical_ids,
            size_t logical_count,
            int32_t * execution_ids,
            bool (*abort_callback)(void *),
            void * abort_data) noexcept override {
        std::lock_guard<std::mutex> lock(mutex);
        if (!active_request || !cache || binding.provider_identity != this || binding.graph_epoch != epoch ||
            binding.generation_lease.get() != cache->allocation_lease().get() || binding.layer < 0 ||
            logical_ids == nullptr || execution_ids == nullptr || logical_count == 0) {
            return failure(llm_expert_provider_error::invalid_binding);
        }
        const auto prior_pins_released = release_request_pins_locked();
        if (!prior_pins_released.is_ready()) return failure(prior_pins_released.error);
        std::vector<llm_expert_key> unique;
        std::vector<size_t> element_index(logical_count);
        try { unique.reserve(logical_count); } catch (...) { return failure(llm_expert_provider_error::allocation_failed); }
        for (size_t index = 0; index < logical_count; ++index) {
            if (logical_ids[index] < 0 || logical_ids[index] >= int32_t(n_expert)) {
                return failure(llm_expert_provider_error::invalid_key);
            }
            const llm_expert_key key = { binding.layer, logical_ids[index] };
            auto found = std::find_if(unique.begin(), unique.end(), [&](auto value) { return same_key(value, key); });
            if (found == unique.end()) { element_index[index] = unique.size(); unique.push_back(key); }
            else element_index[index] = size_t(found - unique.begin());
        }
        if (unique.size() > config.hot_capacity) return failure(llm_expert_provider_error::unsupported_configuration);
        std::vector<llm_cold_reference> references(unique.size());
        const size_t rollback_begin = request_pins.size();
        for (size_t index = 0; index < unique.size(); ++index) {
            uma_load_context load = { config.storage, config.scheduler, abort_callback, abort_data, {}, false };
            auto result = llm_expert_provider_result::success();
            if (!hot_key(unique[index]) && hot_count >= config.hot_capacity) result = make_hot_room();
            if (result.is_ready()) {
                result = cache->find_or_admit_with_loader(unique[index], references[index], load_uma_bundle, &load);
            }
            if (!result.is_ready()) {
                (void) cache->cleanup_failed_slots();
                rollback(rollback_begin);
                return failure(result.error);
            }
            ensure_slot_state(references[index]);
            auto & slot = slots[references[index].slot];
            if (!slot.hot) {
                result = make_hot_room();
                if (!result.is_ready() && load.active) (void) fail_scheduler(load, false);
                if (result.is_ready()) result = complete_readiness(load, references[index]);
                if (result.is_ready()) result = cache->acquire(references[index], llm_cold_reference_kind::hot);
                if (!result.is_ready()) {
                    if (cache->ready(references[index])) (void) cache->retire_ready(references[index]);
                    (void) cache->cleanup_failed_slots();
                    rollback(rollback_begin);
                    return failure(result.error);
                }
                slot.hot = true;
                hot_count++;
            } else if (load.active) {
                rollback(rollback_begin);
                return failure(llm_expert_provider_error::metadata_mismatch);
            }
            result = cache->acquire(references[index], llm_cold_reference_kind::request);
            if (!result.is_ready()) { rollback(rollback_begin); return failure(result.error); }
            slot.refs++;
            slot.last_use = ++use_clock;
            request_pins.push_back(references[index]);
        }
        for (size_t index = 0; index < logical_count; ++index) {
            execution_ids[index] = int32_t(references[element_index[index]].slot);
        }
        return llm_expert_provider_result::success();
    }

    llm_expert_provider_result remap_checkpoint_tensor(
            const llm_expert_graph_binding & binding,
            ggml_backend_t execution_backend,
            bool (*abort_callback)(void *),
            void * abort_data) noexcept override {
        if (binding.logical_ids == nullptr || binding.execution_ids == nullptr) {
            return failure(llm_expert_provider_error::invalid_binding);
        }
        const uint64_t count = uint64_t(binding.execution_ids->ne[0])*binding.execution_ids->ne[1];
        if (count == 0 || count > SIZE_MAX/sizeof(int32_t)) return failure(llm_expert_provider_error::invalid_binding);
        std::vector<int32_t> logical(count), execution(count);
        ggml_backend_tensor_get(binding.execution_ids, logical.data(), 0, count*sizeof(int32_t));
        auto result = remap_checkpoint(binding, logical.data(), logical.size(), execution.data(),
            abort_callback, abort_data);
        if (result.is_ready()) {
            ggml_backend_tensor_set(binding.execution_ids, execution.data(), 0, count*sizeof(int32_t));
            const auto device = execution_backend ? ggml_backend_get_device(execution_backend) : nullptr;
            std::lock_guard<std::mutex> lock(mutex);
            last_execution_backend_device_type = device ? int32_t(ggml_backend_dev_type(device)) : -1;
        }
        return result;
    }

    llm_expert_provider_stats get_stats() const noexcept override {
        std::lock_guard<std::mutex> lock(mutex);
        auto result = stats;
        result.graph_epoch = epoch;
        result.pool_generations = cache ? 1 : 0;
        if (cache) {
            const auto diagnostics = cache->diagnostics();
            result.pool_bytes = diagnostics.actual_bytes;
            result.effective_capacity = diagnostics.effective_slots;
        }
        result.requested_capacity = config.hot_capacity;
        return result;
    }

    llm_hot_cache_diagnostics hot_cache_diagnostics() const override {
        std::lock_guard<std::mutex> lock(mutex);
        llm_hot_cache_diagnostics result;
        result.requested_capacity = config.hot_capacity;
        result.effective_capacity = config.hot_capacity;
        result.graph_epoch = epoch;
        result.generation = cache ? 1 : 0;
        result.n_expert = n_expert;
        result.n_expert_used = config.n_expert_used;
        result.requests = stats.prepare_calls;
        result.source_buffer_type = config.buffer_type;
        result.target_buffer_type = config.buffer_type;
        result.source_pageable = cache != nullptr;
        result.cold_pageable = cache != nullptr;
        result.ring_requested_bytes = 0;
        if (!cache) return result;
        const auto cold = cache->diagnostics();
        result.pool_bytes = cold.actual_bytes;
        result.cold_requested_bytes = cold.requested_bytes;
        result.cold_actual_bytes = cold.actual_bytes;
        result.cold_unused_budget_bytes = cold.unused_budget_bytes;
        result.cold_bundle_payload_bytes = cold.bundle_payload_bytes;
        result.cold_slot_footprint = cold.aligned_slot_footprint;
        result.cold_alignment = cold.alignment;
        result.cold_effective_slots = cold.effective_slots;
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
        result.cold_current_request_refs = cold.current_request_refs;
        result.cold_peak_request_refs = cold.peak_request_refs;
        result.cold_residency_supported = cold.residency_supported;
        result.cold_residency_unavailable_reason = cold.residency_unavailable_reason;
        result.cold_ready_logical_bytes = cold.ready_logical_bytes;
        result.cold_ready_page_count = cold.ready_page_count;
        result.cold_resident_ready_page_count = cold.resident_ready_page_count;
        result.cold_resident_ready_bytes = cold.resident_ready_bytes;
        result.cold_policy = cold.policy;
        result.cold_policy_domains = cold.policy_domains;
        result.cold_policy_events = cold.policy_events;
        result.last_execution_backend_device_type = last_execution_backend_device_type;
        result.cold_slots.reserve(cold.slots.size());
        for (const auto & entry : cold.slots) {
            llm_hot_cache_diagnostics::cold_slot slot;
            slot.layer = entry.key.layer;
            slot.expert = entry.key.expert;
            slot.generation = entry.generation;
            slot.last_use = entry.last_use;
            slot.origin = entry.origin;
            slot.state = static_cast<llm_hot_cache_diagnostics::cold_slot::state_type>(entry.state);
            result.cold_slots.push_back(slot);
        }
        return result;
    }

    llm_expert_provider_initialization_stage initialization_stage() const noexcept override {
        std::lock_guard<std::mutex> lock(mutex);
        return cache ? llm_expert_provider_initialization_stage::none :
            llm_expert_provider_initialization_stage::descriptors_before_scheduler_reserve;
    }

    uint64_t graph_epoch() const noexcept override {
        std::lock_guard<std::mutex> lock(mutex);
        return epoch;
    }

    llm_expert_provider_result begin_initialization(
            llm_expert_provider_initialization_stage stage,
            bool & owner) noexcept override {
        std::lock_guard<std::mutex> lock(mutex);
        owner = false;
        if (cache) return llm_expert_provider_result::success();
        if (initialization_in_progress) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::busy);
        }
        if (stage != llm_expert_provider_initialization_stage::descriptors_before_scheduler_reserve) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::invalid_descriptor);
        }
        initialization_in_progress = true;
        owner = true;
        return llm_expert_provider_result::success();
    }

    llm_expert_provider_result complete_descriptor_discovery(
            uint64_t graph_count,
            uint64_t binding_count,
            uint64_t scheduler_reserve_calls,
            const std::vector<uint64_t> & before,
            const std::vector<uint64_t> & after) noexcept override {
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialization_in_progress || registrations.size() != config.routed_layer_count ||
            !prototype.has_value() || graph_count < 2 || binding_count != graph_count*config.routed_layer_count ||
            scheduler_reserve_calls != 0 || before != after) {
            return failure(llm_expert_provider_error::initialization_failed);
        }
        descriptors_complete = true;
        return llm_expert_provider_result::success();
    }

    llm_expert_provider_result initialize_after_reserve() noexcept override {
        std::lock_guard<std::mutex> lock(mutex);
        if (cache) return llm_expert_provider_result::success();
        if (!initialization_in_progress || !descriptors_complete || !prototype.has_value()) {
            return failure(llm_expert_provider_error::initialization_failed);
        }
        llm_cold_cache_config cold;
        cold.byte_budget = config.pool_bytes;
        cold.minimum_slots = config.n_expert_used;
        cold.routed_layer_count = config.routed_layer_count;
        cold.total_expert_keys = config.total_expert_keys;
        cold.minimum_domain_slots = config.n_expert_used;
        cold.cache_policy_config = config.cold_cache_policy_config;
        cold.routed_layers = config.routed_layers;
        cold.buffer_type = config.buffer_type;
        auto candidate = std::make_unique<llm_cold_expert_cache>(std::move(cold));
        auto result = candidate->initialize(*prototype);
        if (!result.is_ready()) return failure(result.error);
        const auto diagnostics = candidate->diagnostics();
        if (diagnostics.effective_slots < config.n_expert_used || config.hot_capacity > diagnostics.effective_slots ||
            !config.is_uma_buffer_type(ggml_backend_buffer_get_type(candidate->buffer()))) {
            return failure(llm_expert_provider_error::unsupported_configuration);
        }
        const size_t probe_bytes = std::min<size_t>(4096, ggml_backend_buffer_get_size(candidate->buffer()));
        int readiness_status = -1;
        if (config.readiness == LLAMA_EXPERT_UMA_READINESS_AUTO ||
                config.readiness == LLAMA_EXPERT_UMA_READINESS_CUDA_PREFETCH) {
            readiness_status = config.prefetch(candidate->buffer(), 0, probe_bytes);
            if (readiness_status == 0) resolved_readiness = LLAMA_EXPERT_UMA_READINESS_CUDA_PREFETCH;
        }
        if (readiness_status != 0 && (config.readiness == LLAMA_EXPERT_UMA_READINESS_AUTO ||
                config.readiness == LLAMA_EXPERT_UMA_READINESS_CUDA_TOUCH)) {
            uint64_t checksum = 0;
            readiness_status = config.checksum(candidate->buffer(), 0, probe_bytes, &checksum);
            if (readiness_status == 0) resolved_readiness = LLAMA_EXPERT_UMA_READINESS_CUDA_TOUCH;
        }
        if (readiness_status != 0) {
            return failure(llm_expert_provider_error::preparation_failed);
        }
        slots.assign(diagnostics.effective_slots, {});
        cache = std::move(candidate);
        epoch++;
        stats.allocations++;
        return llm_expert_provider_result::success();
    }

    llm_expert_provider_result finish_initialization(bool success) noexcept override {
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialization_in_progress || success != bool(cache)) {
            return failure(llm_expert_provider_error::initialization_failed);
        }
        initialization_in_progress = false;
        return llm_expert_provider_result::success();
    }

    llm_expert_provider_result validate_slot_generation(uint32_t slot, uint64_t generation) noexcept override {
        std::lock_guard<std::mutex> lock(mutex);
        return slot < slots.size() && slots[slot].generation == generation ?
            llm_expert_provider_result::success() :
            llm_expert_provider_result::failure(llm_expert_provider_error::stale_generation);
    }

    llm_expert_provider_result trim() noexcept override {
        std::lock_guard<std::mutex> lock(mutex);
        if (active_request) return llm_expert_provider_result::failure(llm_expert_provider_error::busy);
        for (uint32_t index = 0; index < slots.size(); ++index) {
            auto & slot = slots[index];
            if (slot.hot && slot.refs == 0) {
                auto released = cache->release({ index, slot.generation }, llm_cold_reference_kind::hot);
                if (!released.is_ready()) return released;
                slot.hot = false;
                hot_count--;
            }
        }
        return cache->trim();
    }

    llm_expert_provider_result surrender() noexcept override {
        std::lock_guard<std::mutex> lock(mutex);
        if (active_request) return llm_expert_provider_result::failure(llm_expert_provider_error::busy);
        if (!cache) return llm_expert_provider_result::success();
        for (uint32_t index = 0; index < slots.size(); ++index) {
            auto & slot = slots[index];
            if (!slot.hot) continue;
            if (slot.refs != 0) return llm_expert_provider_result::failure(llm_expert_provider_error::busy);
            auto released = cache->release({ index, slot.generation }, llm_cold_reference_kind::hot);
            if (!released.is_ready()) return released;
            slot.hot = false;
        }
        auto result = cache->surrender();
        if (!result.is_ready()) return result;
        cache.reset();
        slots.clear();
        hot_count = 0;
        epoch++;
        stats.surrender_successes++;
        return llm_expert_provider_result::success();
    }

protected:
    void release_handle(uint64_t) noexcept override {
        std::lock_guard<std::mutex> lock(mutex);
        release_pins_locked();
        stats.handles_released++;
    }

private:
    struct slot_state { uint64_t generation = 0; uint64_t last_use = 0; uint32_t refs = 0; bool hot = false; };

    llm_expert_provider_result bind_impl(
            ggml_context * graph_ctx,
            const llm_expert_bundle_descriptor & bundle,
            const llm_expert_selection & selection,
            llm_expert_graph_binding & binding) noexcept {
        std::lock_guard<std::mutex> lock(mutex);
        auto result = bundle.validate();
        if (!result.is_ready() || !selection.validate().is_ready() || selection.layer != bundle.layer) {
            return failure(llm_expert_provider_error::invalid_descriptor);
        }
        auto found = registrations.find(bundle.layer);
        if (found == registrations.end()) {
            if (prototype.has_value() && bundle.n_expert != prototype->n_expert) {
                return failure(llm_expert_provider_error::invalid_descriptor);
            }
            if (!prototype.has_value()) prototype = bundle;
            registrations.emplace(bundle.layer, bundle);
        }
        binding = {};
        binding.provider_identity = this;
        binding.layer = bundle.layer;
        binding.logical_ids = selection.logical_ids;
        binding.graph_epoch = epoch;
        if (!cache) {
            binding.up = bundle.up; binding.gate = bundle.gate;
            binding.gate_up = bundle.gate_up; binding.down = bundle.down;
            binding.execution_ids = selection.logical_ids;
            binding.bootstrap = true;
        } else {
            const auto & resident = cache->bundle();
            binding.up = resident.up; binding.gate = resident.gate;
            binding.gate_up = resident.gate_up; binding.down = resident.down;
            binding.execution_ids = graph_ctx ? ggml_dup(graph_ctx, selection.logical_ids) : selection.logical_ids;
            if (graph_ctx) ggml_format_name(binding.execution_ids, "expert_execution_ids-%d", bundle.layer);
            binding.generation_lease = cache->allocation_lease();
        }
        stats.bind_calls++;
        return binding.validate(selection);
    }

    llm_expert_provider_result set_plan_failure(llm_expert_execution_plan & plan, llm_expert_provider_error error) {
        auto result = llm_expert_provider_result::failure(error);
        plan.set_result(result);
        return result;
    }

    llm_expert_provider_result failure(llm_expert_provider_error error) const {
        stats.failures++;
        return llm_expert_provider_result::failure(error);
    }

    void ensure_slot_state(llm_cold_reference reference) {
        auto & state = slots[reference.slot];
        if (state.generation != reference.generation) state = { reference.generation, 0, 0, false };
    }

    llm_expert_provider_result make_hot_room() {
        if (hot_count < config.hot_capacity) return llm_expert_provider_result::success();
        uint32_t victim = UINT32_MAX;
        for (uint32_t index = 0; index < slots.size(); ++index) {
            if (slots[index].hot && slots[index].refs == 0 &&
                (victim == UINT32_MAX || slots[index].last_use < slots[victim].last_use ||
                 (slots[index].last_use == slots[victim].last_use && index < victim))) victim = index;
        }
        if (victim == UINT32_MAX) return llm_expert_provider_result::failure(llm_expert_provider_error::busy);
        auto result = cache->release({ victim, slots[victim].generation }, llm_cold_reference_kind::hot);
        if (result.is_ready()) { slots[victim].hot = false; hot_count--; }
        return result;
    }

    bool hot_key(llm_expert_key key) const {
        const auto diagnostics = cache->diagnostics();
        for (uint32_t index = 0; index < diagnostics.slots.size(); ++index) {
            const auto & entry = diagnostics.slots[index];
            if (same_key(entry.key, key) && entry.generation == slots[index].generation && slots[index].hot) {
                return true;
            }
        }
        return false;
    }

    llm_expert_provider_result complete_readiness(uma_load_context & load, llm_cold_reference reference) {
        if (load.active && config.scheduler->transition(load.handle, llm_expert_request_state::io_in_flight,
                llm_expert_request_state::host_ready) != llm_expert_schedule_disposition::admitted) {
            return fail_scheduler(load, false);
        }
        if (load.active && config.scheduler->transition(load.handle, llm_expert_request_state::host_ready,
                llm_expert_request_state::device_preparing) != llm_expert_schedule_disposition::admitted) {
            return fail_scheduler(load, false);
        }
        std::array<llm_expert_storage_destination, 12> destinations;
        size_t count = 0;
        if (!build_destinations(cache->bundle(), reference.slot, destinations, count)) {
            return load.active ? fail_scheduler(load, false) :
                llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
        }
        auto * base = static_cast<uint8_t *>(ggml_backend_buffer_get_base(cache->buffer()));
        for (size_t index = 0; index < count; ++index) {
            const size_t offset = static_cast<uint8_t *>(destinations[index].data) - base;
            int status = resolved_readiness == LLAMA_EXPERT_UMA_READINESS_CUDA_TOUCH ?
                config.checksum(cache->buffer(), offset, destinations[index].extent, &readiness_checksum) :
                config.prefetch(cache->buffer(), offset, destinations[index].extent);
            if (status != 0) return load.active ? fail_scheduler(load, false) :
                llm_expert_provider_result::failure(llm_expert_provider_error::preparation_failed);
        }
        if (load.active) {
            if (config.scheduler->transition(load.handle, llm_expert_request_state::device_preparing,
                    llm_expert_request_state::device_ready) != llm_expert_schedule_disposition::admitted ||
                config.scheduler->finish(load.handle, llm_expert_request_state::complete) !=
                    llm_expert_schedule_disposition::admitted ||
                config.scheduler->release_terminal(load.handle) != llm_expert_schedule_disposition::admitted) {
                return fail_scheduler(load, false);
            }
            load.active = false;
        }
        return llm_expert_provider_result::success();
    }

    void rollback(size_t begin) {
        while (request_pins.size() > begin) {
            auto reference = request_pins.back();
            request_pins.pop_back();
            (void) cache->release(reference, llm_cold_reference_kind::request);
            if (slots[reference.slot].refs != 0) slots[reference.slot].refs--;
        }
    }

    void release_pins_locked() {
        (void) release_request_pins_locked();
        if (active_request && cache) (void) cache->policy_request_end(true, false);
        active_request = false;
    }

    llm_expert_provider_result release_request_pins_locked() {
        while (!request_pins.empty()) {
            const auto reference = request_pins.back();
            auto result = cache->release(reference, llm_cold_reference_kind::request);
            if (!result.is_ready()) return result;
            request_pins.pop_back();
            if (slots[reference.slot].refs == 0) {
                return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
            }
            slots[reference.slot].refs--;
        }
        return llm_expert_provider_result::success();
    }

    llm_uma_cache_config config;
    llm_expert_provider_faults faults;
    uint32_t n_expert = 0;
    mutable std::mutex mutex;
    std::map<int32_t, llm_expert_bundle_descriptor> registrations;
    std::optional<llm_expert_bundle_descriptor> prototype;
    std::unique_ptr<llm_cold_expert_cache> cache;
    std::vector<slot_state> slots;
    std::vector<llm_cold_reference> request_pins;
    mutable llm_expert_provider_stats stats;
    uint64_t epoch = 0;
    uint64_t lease_id = 0;
    uint64_t use_clock = 0;
    uint64_t readiness_checksum = 0;
    uint32_t hot_count = 0;
    llama_expert_uma_readiness resolved_readiness = LLAMA_EXPERT_UMA_READINESS_AUTO;
    int32_t last_execution_backend_device_type = -1;
    bool initialization_in_progress = false;
    bool descriptors_complete = false;
    bool active_request = false;
};

} // namespace

std::unique_ptr<llm_expert_weight_provider> llm_create_uma_cache_expert_weight_provider(
        llm_uma_cache_config config,
        llm_expert_provider_faults faults) {
    return std::make_unique<llm_uma_expert_weight_provider>(std::move(config), faults);
}
