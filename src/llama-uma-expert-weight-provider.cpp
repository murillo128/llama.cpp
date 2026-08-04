#include "llama-expert-weight-provider.h"

#include "llama-cold-expert-cache.h"
#include "llama-expert-cache-policy.h"
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

#ifdef __linux__
#include <cerrno>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace {

bool same_key(llm_expert_key lhs, llm_expert_key rhs) {
    return lhs.layer == rhs.layer && lhs.expert == rhs.expert;
}

llm_expert_provider_result policy_result(llm_expert_cache_policy_result result) {
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
    llm_expert_provider_result (*preflight)(void *, uint64_t) = nullptr;
    void * preflight_data = nullptr;
    uint64_t incoming_bytes = 0;
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
    if (context.preflight != nullptr) {
        const auto preflight = context.preflight(context.preflight_data, context.incoming_bytes);
        if (!preflight.is_ready()) return preflight;
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
        if (this->config.sample_memory == nullptr) this->config.sample_memory = llm_expert_uma_sample_memory;
        if (faults.initialization != llm_expert_provider_error::none ||
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
        if (this->config.hot_cache_policy_config.scope == LLAMA_EXPERT_CACHE_POLICY_SCOPE_PER_LAYER &&
            this->config.hot_capacity < uint64_t(this->config.routed_layer_count)*this->config.n_expert_used) {
            throw std::invalid_argument("per-layer UMA hot policy cannot satisfy simultaneous demand width");
        }
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
        if (cache) {
            for (uint32_t index = 0; index < hot_entries.size(); ++index) {
                if (hot_entries[index].occupied) (void) demote_hot(index);
            }
            (void) hot_policy.surrender();
            (void) cache->surrender();
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
        auto hot_started = policy_result(hot_policy.request_begin());
        if (!hot_started.is_ready()) {
            (void) cache->policy_request_end(false, false);
            return set_plan_failure(plan, hot_started.error);
        }
        active_request = true;
        request_failed = false;
        request_cancelled = false;
        const uint64_t lease = ++lease_id;
        try {
            plan.reserve(1);
            plan.add_handle(llm_expert_handle(this, lease));
        } catch (...) {
            active_request = false;
            (void) cache->policy_request_end(false, false);
            (void) hot_policy.request_end(false, false);
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
            auto result = policy_result(hot_policy.demand(
                { unique[index].layer, unique[index].expert }, 1, bundle_payload_bytes, slot_footprint_bytes));
            if (!result.is_ready()) {
                rollback(rollback_begin);
                return failure(result.error);
            }
            int32_t hot_slot = find_hot_key(unique[index]);
            llm_expert_cache_policy_decision decision;
            if (hot_slot < 0) {
                result = select_hot_slot(unique[index], decision);
                if (result.is_ready() && !decision.free) result = demote_hot(decision.slot);
                if (!result.is_ready()) {
                    rollback(rollback_begin);
                    return failure(result.error);
                }
            }
            uma_load_context load = { config.storage, config.scheduler, abort_callback, abort_data,
                preflight_trampoline, this, slot_footprint_bytes, {}, false };
            result = cache->find_or_admit_with_loader(unique[index], references[index], load_uma_bundle, &load);
            if (!result.is_ready()) {
                (void) cache->cleanup_failed_slots();
                rollback(rollback_begin);
                return failure(result.error);
            }
            ensure_slot_state(references[index]);
            auto & slot = slots[references[index].slot];
            if (hot_slot >= 0 && find_hot_entry(unique[index], references[index]) != hot_slot) {
                rollback(rollback_begin);
                return failure(llm_expert_provider_error::metadata_mismatch);
            }
            const auto hit = classify_hit(load.active, hot_slot >= 0, references[index]);
            if (hot_slot < 0) {
                result = policy_result(hot_policy.load_begin(
                    decision.slot, references[index].generation,
                    { unique[index].layer, unique[index].expert }, bundle_payload_bytes, slot_footprint_bytes));
                if (!result.is_ready() && load.active) (void) fail_scheduler(load, false);
                if (result.is_ready()) result = complete_readiness(load, references[index]);
                bool hot_acquired = false;
                if (result.is_ready()) {
                    result = cache->acquire(references[index], llm_cold_reference_kind::hot);
                    hot_acquired = result.is_ready();
                }
                if (result.is_ready()) result = policy_result(
                    hot_policy.load_complete(decision.slot, references[index].generation));
                if (!result.is_ready()) {
                    if (hot_acquired) (void) cache->release(
                        references[index], llm_cold_reference_kind::hot);
                    if (decision.slot < hot_entries.size() && hot_policy.validate_loading(
                            decision.slot, references[index].generation,
                            { unique[index].layer, unique[index].expert })) {
                        (void) hot_policy.load_failed(decision.slot, references[index].generation);
                    }
                    if (cache->ready(references[index])) (void) cache->retire_ready(references[index]);
                    (void) cache->cleanup_failed_slots();
                    rollback(rollback_begin);
                    return failure(result.error);
                }
                hot_entries[decision.slot] = { unique[index], references[index], true };
                hot_slot = int32_t(decision.slot);
                slot.hot = true;
                hot_count++;
            } else if (load.active) {
                rollback(rollback_begin);
                return failure(llm_expert_provider_error::metadata_mismatch);
            } else {
                result = policy_result(hot_policy.hit(uint32_t(hot_slot), references[index].generation));
                if (!result.is_ready()) { rollback(rollback_begin); return failure(result.error); }
            }
            result = cache->acquire(references[index], llm_cold_reference_kind::request);
            if (result.is_ready()) result = policy_result(
                hot_policy.pin(uint32_t(hot_slot), references[index].generation));
            if (!result.is_ready()) { rollback(rollback_begin); return failure(result.error); }
            slot.refs++;
            slot.last_use = ++use_clock;
            request_pins.push_back({ references[index], uint32_t(hot_slot), hit });
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
        result.cold_reclaimed_bytes = cold.reclaimed_bytes;
        result.cold_reclaim_failures = cold.reclaim_failures;
        result.uma_physical_ram_bytes = memory_sample.physical_ram_bytes;
        result.uma_memory_available_bytes = memory_sample.memory_available_bytes;
        result.uma_cgroup_memory_max_bytes = memory_sample.cgroup_memory_max_bytes;
        result.uma_cgroup_memory_current_bytes = memory_sample.cgroup_memory_current_bytes;
        result.uma_process_rss_bytes = memory_sample.process_rss_bytes;
        result.uma_process_swap_bytes = memory_sample.process_swap_bytes;
        result.uma_safe_pool_bytes = headroom.safe_pool_bytes;
        result.uma_effective_pool_bytes = headroom.effective_pool_bytes;
        result.uma_system_reserve_bytes = headroom.system_reserve_bytes;
        result.uma_runtime_reserve_bytes = headroom.runtime_reserve_bytes;
        result.uma_runtime_delta_bytes = runtime_delta_bytes;
        result.uma_headroom_remainder_bytes = headroom.remainder_bytes;
        result.uma_pressure_samples = pressure_samples;
        result.uma_pressure_rejections = pressure_rejections;
        result.uma_storage_misses = storage_misses;
        result.uma_resident_hot_hits = resident_hot_hits;
        result.uma_prepared_cold_hits = prepared_cold_hits;
        result.uma_degraded_hits = degraded_hits;
        result.uma_unknown_residency_hits = unknown_residency_hits;
        result.uma_major_faults = memory_sample.major_faults;
        result.uma_psi_full_total_usec = memory_sample.psi_full_total_usec;
        result.uma_zram_write_bytes = memory_sample.zram_write_bytes;
        result.uma_zswap_write_pages = memory_sample.zswap_write_pages;
        result.uma_autofit = headroom.autofit;
        result.uma_pressure_circuit_open = pressure_circuit_open;
        result.uma_swap_counters_supported = memory_sample.swap_counters_supported;
        result.uma_psi_full_supported = memory_sample.psi_full_supported;
        result.uma_zram_present = memory_sample.zram_present;
        result.uma_zram_counters_supported = memory_sample.zram_counters_supported;
        result.uma_zswap_enabled = memory_sample.zswap_enabled;
        result.uma_zswap_counters_supported = memory_sample.zswap_counters_supported;
        result.uma_nvidia_hmm_counters_supported = memory_sample.nvidia_hmm_counters_supported;
        result.uma_zram_status_reason = memory_sample.zram_status_reason;
        result.uma_zswap_status_reason = memory_sample.zswap_status_reason;
        result.uma_nvidia_hmm_status_reason = memory_sample.nvidia_hmm_status_reason;
        result.uma_telemetry_unavailable_reason = !memory_sample.unavailable_reason.empty() ?
            memory_sample.unavailable_reason : residency_unavailable_reason;
        result.cold_policy = cold.policy;
        result.cold_policy_domains = cold.policy_domains;
        result.cold_policy_events = cold.policy_events;
        result.policy = hot_policy.diagnostics();
        result.policy_domains = hot_policy.domain_diagnostics();
        result.policy_events.assign(hot_policy.transcript().begin(),
            hot_policy.transcript().begin() + hot_policy.transcript_size());
        result.hits = result.policy.hits;
        result.misses = result.policy.misses;
        result.admissions = result.policy.mandatory_admissions;
        result.evictions = result.policy.victim_selections;
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
        uint64_t calculated_slot_footprint = 0;
        auto result = llm_cold_expert_cache::calculate_slot_footprint(
            *prototype, config.buffer_type, calculated_slot_footprint);
        if (!result.is_ready()) return failure(result.error);
        llm_expert_uma_memory_sample sampled;
        if (!config.sample_memory(sampled).is_ready() || !sampled.swap_counters_supported ||
            !sampled.psi_full_supported || (sampled.zram_present && !sampled.zram_counters_supported) ||
            (sampled.zswap_enabled && !sampled.zswap_counters_supported)) {
            memory_sample = sampled;
            return failure(llm_expert_provider_error::unsupported_configuration);
        }
        llm_expert_uma_headroom_input headroom_input = {
            sampled.physical_ram_bytes, sampled.cgroup_memory_max_bytes,
            sampled.cgroup_memory_current_bytes, sampled.memory_available_bytes,
            sampled.cgroup_memory_current_bytes, 0, config.pool_bytes, calculated_slot_footprint,
            config.min_system_headroom_bytes, config.min_runtime_headroom_bytes,
        };
        auto headroom_result = llm_expert_uma_calculate_headroom(headroom_input, headroom);
        if (!headroom_result.is_ready()) {
            memory_sample = sampled;
            return failure(headroom_result.error == llm_expert_uma_error::unsafe_capacity ?
                llm_expert_provider_error::allocation_failed :
                llm_expert_provider_error::unsupported_configuration);
        }
        if (calculated_slot_footprint > UINT64_MAX/config.total_expert_keys) {
            return failure(llm_expert_provider_error::unsupported_configuration);
        }
        const uint64_t topology_bytes = calculated_slot_footprint*config.total_expert_keys;
        const uint64_t selected_pool_bytes = std::min(headroom.effective_pool_bytes, topology_bytes);
        if (selected_pool_bytes/calculated_slot_footprint < config.n_expert_used ||
            selected_pool_bytes/calculated_slot_footprint < config.hot_capacity) {
            return failure(llm_expert_provider_error::allocation_failed);
        }
        headroom.effective_pool_bytes = selected_pool_bytes;
        headroom.slot_count = selected_pool_bytes/calculated_slot_footprint;
        memory_sample = sampled;
        baseline_memory_sample = sampled;
        llm_cold_cache_config cold;
        cold.byte_budget = selected_pool_bytes;
        cold.minimum_slots = config.n_expert_used;
        cold.routed_layer_count = config.routed_layer_count;
        cold.total_expert_keys = config.total_expert_keys;
        cold.minimum_domain_slots = config.n_expert_used;
        cold.cache_policy_config = config.cold_cache_policy_config;
        cold.routed_layers = config.routed_layers;
        cold.buffer_type = config.buffer_type;
        cold.reclaim_free_pages = true;
        auto candidate = std::make_unique<llm_cold_expert_cache>(std::move(cold));
        result = candidate->initialize(*prototype);
        if (!result.is_ready()) return failure(result.error);
        const auto diagnostics = candidate->diagnostics();
        if (diagnostics.effective_slots < config.n_expert_used || config.hot_capacity > diagnostics.effective_slots ||
            !config.is_uma_buffer_type(ggml_backend_buffer_get_type(candidate->buffer()))) {
            return failure(llm_expert_provider_error::unsupported_configuration);
        }
        auto hot_initialized = policy_result(hot_policy.initialize(
            config.hot_cache_policy_config, llm_expert_cache_policy_tier::hot,
            config.routed_layers.data(), config.routed_layer_count, n_expert, config.n_expert_used,
            config.hot_capacity, diagnostics.aligned_slot_footprint, 16384));
        if (!hot_initialized.is_ready()) return failure(hot_initialized.error);
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
        hot_entries.assign(config.hot_capacity, {});
        hot_candidates.assign(config.hot_capacity, {});
        bundle_payload_bytes = diagnostics.bundle_payload_bytes;
        slot_footprint_bytes = diagnostics.aligned_slot_footprint;
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

    llm_expert_provider_result record_initialization_telemetry(
            const std::vector<uint64_t> & backend_bytes_after_hierarchy,
            const std::vector<uint64_t> & backend_bytes_after_final_reserve,
            uint64_t final_bootstrap_source_bindings,
            uint64_t complete_deferred_payload_bytes) noexcept override {
        std::lock_guard<std::mutex> lock(mutex);
        if (!cache || backend_bytes_after_hierarchy.size() != backend_bytes_after_final_reserve.size() ||
            final_bootstrap_source_bindings != 0 || complete_deferred_payload_bytes == 0) {
            return failure(llm_expert_provider_error::initialization_failed);
        }
        uint64_t delta = 0;
        for (size_t index = 0; index < backend_bytes_after_hierarchy.size(); ++index) {
            const uint64_t before = backend_bytes_after_hierarchy[index];
            const uint64_t after = backend_bytes_after_final_reserve[index];
            const uint64_t growth = after > before ? after - before : 0;
            if (growth > UINT64_MAX - delta) return failure(llm_expert_provider_error::unsupported_configuration);
            delta += growth;
        }
        if (delta > UINT64_MAX/5) return failure(llm_expert_provider_error::unsupported_configuration);
        const uint64_t required = (delta*5 + 3)/4;
        if (required > headroom.runtime_reserve_bytes) {
            return failure(llm_expert_provider_error::allocation_failed);
        }
        runtime_delta_bytes = delta;
        return llm_expert_provider_result::success();
    }

    llm_expert_provider_result validate_slot_generation(uint32_t slot, uint64_t generation) noexcept override {
        std::lock_guard<std::mutex> lock(mutex);
        return slot < slots.size() && slots[slot].generation == generation ?
            llm_expert_provider_result::success() :
            llm_expert_provider_result::failure(llm_expert_provider_error::stale_generation);
    }

    llm_expert_provider_result cleanup_failed_slots() noexcept override {
        std::lock_guard<std::mutex> lock(mutex);
        return cache ? cache->cleanup_failed_slots() : llm_expert_provider_result::success();
    }

    llm_expert_provider_result trim() noexcept override {
        std::lock_guard<std::mutex> lock(mutex);
        if (active_request) return llm_expert_provider_result::failure(llm_expert_provider_error::busy);
        for (uint32_t index = 0; index < hot_entries.size(); ++index) {
            if (!hot_entries[index].occupied) continue;
            auto demoted = demote_hot(index);
            if (!demoted.is_ready()) return demoted;
        }
        auto result = cache->trim();
        if (result.is_ready()) result = policy_result(hot_policy.reset());
        if (result.is_ready()) hot_entries.assign(config.hot_capacity, {});
        return result;
    }

    llm_expert_provider_result surrender() noexcept override {
        std::lock_guard<std::mutex> lock(mutex);
        if (active_request) return llm_expert_provider_result::failure(llm_expert_provider_error::busy);
        if (!cache) return llm_expert_provider_result::success();
        auto allocation_lease = cache->allocation_lease();
        if (allocation_lease.use_count() != 2) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::busy);
        }
        allocation_lease.reset();
        for (uint32_t index = 0; index < hot_entries.size(); ++index) {
            if (!hot_entries[index].occupied) continue;
            auto demoted = demote_hot(index);
            if (!demoted.is_ready()) return demoted;
        }
        auto result = policy_result(hot_policy.surrender());
        if (!result.is_ready()) return result;
        result = cache->surrender();
        if (!result.is_ready()) return result;
        cache.reset();
        slots.clear();
        hot_entries.clear();
        hot_candidates.clear();
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
    enum class hit_kind : uint8_t { none, resident_hot, prepared_cold, degraded, unknown };
    struct hit_observation {
        hit_kind kind = hit_kind::none;
        llm_expert_uma_memory_sample before;
        bool sampled = false;
    };
    struct slot_state { uint64_t generation = 0; uint64_t last_use = 0; uint32_t refs = 0; bool hot = false; };
    struct hot_entry { llm_expert_key key; llm_cold_reference reference; bool occupied = false; };
    struct request_pin {
        llm_cold_reference reference;
        uint32_t hot_slot = UINT32_MAX;
        hit_observation hit;
    };

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

    llm_expert_provider_result failure(llm_expert_provider_error error) {
        stats.failures++;
        if (active_request) {
            request_failed = true;
            request_cancelled = request_cancelled || error == llm_expert_provider_error::cancelled;
        }
        return llm_expert_provider_result::failure(error);
    }

    void ensure_slot_state(llm_cold_reference reference) {
        auto & state = slots[reference.slot];
        if (state.generation != reference.generation) state = { reference.generation, 0, 0, false };
    }

    int32_t find_hot_entry(llm_expert_key key, llm_cold_reference reference) const {
        for (uint32_t index = 0; index < hot_entries.size(); ++index) {
            const auto & entry = hot_entries[index];
            if (entry.occupied && same_key(entry.key, key) && entry.reference.slot == reference.slot &&
                entry.reference.generation == reference.generation) return int32_t(index);
        }
        return -1;
    }

    int32_t find_hot_key(llm_expert_key key) const {
        for (uint32_t index = 0; index < hot_entries.size(); ++index) {
            if (hot_entries[index].occupied && same_key(hot_entries[index].key, key)) return int32_t(index);
        }
        return -1;
    }

    llm_expert_provider_result select_hot_slot(
            llm_expert_key key, llm_expert_cache_policy_decision & decision) {
        for (uint32_t index = 0; index < hot_entries.size(); ++index) {
            const auto & entry = hot_entries[index];
            hot_candidates[index] = { index, entry.reference.generation,
                { entry.key.layer, entry.key.expert }, bundle_payload_bytes, slot_footprint_bytes,
                !entry.occupied, entry.occupied && slots[entry.reference.slot].refs == 0 };
        }
        return policy_result(hot_policy.select({ key.layer, key.expert },
            hot_candidates.data(), hot_candidates.size(), decision));
    }

    llm_expert_provider_result demote_hot(uint32_t hot_slot) {
        if (hot_slot >= hot_entries.size() || !hot_entries[hot_slot].occupied) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
        }
        const auto entry = hot_entries[hot_slot];
        if (slots[entry.reference.slot].refs != 0) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::busy);
        }
        auto result = policy_result(hot_policy.validate_evictable(hot_slot, entry.reference.generation));
        if (result.is_ready()) result = cache->release(entry.reference, llm_cold_reference_kind::hot);
        if (result.is_ready()) result = policy_result(hot_policy.evict(hot_slot, entry.reference.generation));
        if (result.is_ready()) {
            slots[entry.reference.slot].hot = false;
            hot_entries[hot_slot] = {};
            hot_count--;
        }
        return result;
    }

    static llm_expert_provider_result preflight_trampoline(void * data, uint64_t incoming_bytes) {
        return static_cast<llm_uma_expert_weight_provider *>(data)->preflight_pressure(incoming_bytes);
    }

    llm_expert_provider_result preflight_pressure(uint64_t incoming_bytes) {
        pressure_samples++;
        if (pressure_circuit_open) {
            pressure_rejections++;
            return llm_expert_provider_result::failure(llm_expert_provider_error::allocation_failed);
        }
        llm_expert_uma_memory_sample current;
        if (!config.sample_memory(current).is_ready()) {
            memory_sample = current;
            pressure_rejections++;
            return llm_expert_provider_result::failure(llm_expert_provider_error::allocation_failed);
        }
        if (!current.swap_counters_supported || !current.psi_full_supported ||
            (current.zram_present && !current.zram_counters_supported) ||
            (current.zswap_enabled && !current.zswap_counters_supported) ||
            current.process_swap_bytes > baseline_memory_sample.process_swap_bytes ||
            current.cgroup_swap_current_bytes > baseline_memory_sample.cgroup_swap_current_bytes ||
            current.pswpin_pages > baseline_memory_sample.pswpin_pages ||
            current.pswpout_pages > baseline_memory_sample.pswpout_pages ||
            current.psi_full_total_usec > baseline_memory_sample.psi_full_total_usec ||
            current.zram_write_bytes > baseline_memory_sample.zram_write_bytes ||
            current.zswap_write_pages > baseline_memory_sample.zswap_write_pages) {
            pressure_circuit_open = true;
            pressure_rejections++;
            memory_sample = current;
            return llm_expert_provider_result::failure(llm_expert_provider_error::allocation_failed);
        }
        const uint64_t cgroup_available = current.cgroup_memory_current_bytes <= current.cgroup_memory_max_bytes ?
            current.cgroup_memory_max_bytes - current.cgroup_memory_current_bytes : 0;
        const uint64_t available = std::min(current.memory_available_bytes, cgroup_available);
        const uint64_t hysteresis = std::max<uint64_t>(
            slot_footprint_bytes <= UINT64_MAX/2 ? 2*slot_footprint_bytes : UINT64_MAX,
            UINT64_C(1024)*1024*1024);
        uint64_t required = headroom.system_reserve_bytes;
        if (headroom.runtime_reserve_bytes > UINT64_MAX - required) {
            pressure_rejections++;
            return llm_expert_provider_result::failure(llm_expert_provider_error::allocation_failed);
        }
        required += headroom.runtime_reserve_bytes;
        if (hysteresis > UINT64_MAX - required) {
            pressure_rejections++;
            return llm_expert_provider_result::failure(llm_expert_provider_error::allocation_failed);
        }
        required += hysteresis;
        if (incoming_bytes > UINT64_MAX - required) {
            pressure_rejections++;
            return llm_expert_provider_result::failure(llm_expert_provider_error::allocation_failed);
        }
        required += incoming_bytes;
        memory_sample = current;
        if (available < required) {
            pressure_rejections++;
            return llm_expert_provider_result::failure(llm_expert_provider_error::allocation_failed);
        }
        return llm_expert_provider_result::success();
    }

    hit_observation classify_hit(bool storage_miss, bool hot_hit, llm_cold_reference reference) {
        if (storage_miss) {
            storage_misses++;
            return {};
        }
        bool supported = false, resident = false;
        exact_slot_residency(reference, supported, resident);
        if (!supported) {
            residency_unavailable_reason = "exact-slot mincore telemetry unavailable";
            return { hit_kind::unknown, {}, false };
        }
        if (!resident) return { hit_kind::degraded, {}, false };
        hit_observation result;
        result.kind = hot_hit ? hit_kind::resident_hot : hit_kind::prepared_cold;
        result.sampled = config.sample_memory(result.before).is_ready();
        if (!result.sampled || !result.before.swap_counters_supported || !result.before.psi_full_supported ||
            (result.before.zram_present && !result.before.zram_counters_supported) ||
            (result.before.zswap_enabled && !result.before.zswap_counters_supported)) {
            result.kind = hit_kind::unknown;
            result.sampled = false;
            residency_unavailable_reason = !result.before.unavailable_reason.empty() ?
                result.before.unavailable_reason : "hit fault/page-in telemetry unavailable";
        }
        return result;
    }

    void complete_hit_observation(const hit_observation & observation) {
        if (observation.kind == hit_kind::none) return;
        if (observation.kind == hit_kind::unknown) { unknown_residency_hits++; return; }
        if (observation.kind == hit_kind::degraded) { degraded_hits++; return; }
        llm_expert_uma_memory_sample after;
        if (!observation.sampled || !config.sample_memory(after).is_ready() ||
            !after.swap_counters_supported || !after.psi_full_supported ||
            (after.zram_present && !after.zram_counters_supported) ||
            (after.zswap_enabled && !after.zswap_counters_supported)) {
            unknown_residency_hits++;
            residency_unavailable_reason = !after.unavailable_reason.empty() ?
                after.unavailable_reason : "hit fault/page-in telemetry unavailable";
            return;
        }
        memory_sample = after;
        const bool pressure_refill = after.process_swap_bytes > observation.before.process_swap_bytes ||
            after.cgroup_swap_current_bytes > observation.before.cgroup_swap_current_bytes ||
            after.pswpin_pages > observation.before.pswpin_pages ||
            after.pswpout_pages > observation.before.pswpout_pages ||
            after.psi_full_total_usec > observation.before.psi_full_total_usec ||
            after.zram_write_bytes > observation.before.zram_write_bytes ||
            after.zswap_write_pages > observation.before.zswap_write_pages;
        const bool faulted = after.major_faults > observation.before.major_faults;
        if (pressure_refill) pressure_circuit_open = true;
        if (faulted || pressure_refill) degraded_hits++;
        else if (observation.kind == hit_kind::resident_hot) resident_hot_hits++;
        else prepared_cold_hits++;
    }

    void exact_slot_residency(llm_cold_reference reference, bool & supported, bool & resident) {
        supported = false;
        resident = false;
#ifdef __linux__
        const long page_value = sysconf(_SC_PAGESIZE);
        if (page_value <= 0 || !cache->ready(reference)) return;
        const size_t page = size_t(page_value);
        std::array<llm_expert_storage_destination, 12> destinations;
        size_t count = 0;
        if (!build_destinations(cache->bundle(), reference.slot, destinations, count)) return;
        supported = true;
        resident = true;
        for (size_t index = 0; index < count; ++index) {
            const uintptr_t address = reinterpret_cast<uintptr_t>(destinations[index].data);
            const uintptr_t begin = address/page*page;
            const uintptr_t end_unrounded = address + destinations[index].extent;
            if (end_unrounded < address || end_unrounded > UINTPTR_MAX - (page - 1)) {
                supported = false;
                resident = false;
                return;
            }
            const uintptr_t end = (end_unrounded + page - 1)/page*page;
            std::vector<unsigned char> state((end - begin)/page);
            if (mincore(reinterpret_cast<void *>(begin), end - begin, state.data()) != 0) {
                supported = false;
                resident = false;
                return;
            }
            if (std::any_of(state.begin(), state.end(), [](unsigned char value) { return (value & 1) == 0; })) {
                resident = false;
            }
        }
#else
        (void) reference;
#endif
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
            auto pin = request_pins.back();
            request_pins.pop_back();
            (void) hot_policy.unpin(pin.hot_slot, pin.reference.generation);
            (void) cache->release(pin.reference, llm_cold_reference_kind::request);
            if (slots[pin.reference.slot].refs != 0) slots[pin.reference.slot].refs--;
        }
    }

    void release_pins_locked() {
        (void) release_request_pins_locked();
        if (active_request && cache) {
            (void) cache->policy_request_end(!request_failed, request_cancelled);
            (void) hot_policy.request_end(!request_failed, request_cancelled);
        }
        active_request = false;
    }

    llm_expert_provider_result release_request_pins_locked() {
        while (!request_pins.empty()) {
            const auto pin = request_pins.back();
            complete_hit_observation(pin.hit);
            auto result = policy_result(hot_policy.unpin(pin.hot_slot, pin.reference.generation));
            if (result.is_ready()) result = cache->release(pin.reference, llm_cold_reference_kind::request);
            if (!result.is_ready()) return result;
            request_pins.pop_back();
            if (slots[pin.reference.slot].refs == 0) {
                return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
            }
            slots[pin.reference.slot].refs--;
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
    std::vector<hot_entry> hot_entries;
    std::vector<llm_expert_cache_policy_candidate> hot_candidates;
    std::vector<request_pin> request_pins;
    llm_expert_cache_policy hot_policy;
    mutable llm_expert_provider_stats stats;
    uint64_t epoch = 0;
    uint64_t lease_id = 0;
    uint64_t use_clock = 0;
    uint64_t readiness_checksum = 0;
    uint64_t bundle_payload_bytes = 0;
    uint64_t slot_footprint_bytes = 0;
    uint64_t runtime_delta_bytes = 0;
    uint64_t pressure_samples = 0;
    uint64_t pressure_rejections = 0;
    uint64_t storage_misses = 0;
    uint64_t resident_hot_hits = 0;
    uint64_t prepared_cold_hits = 0;
    uint64_t degraded_hits = 0;
    uint64_t unknown_residency_hits = 0;
    llm_expert_uma_headroom headroom;
    llm_expert_uma_memory_sample baseline_memory_sample;
    llm_expert_uma_memory_sample memory_sample;
    std::string residency_unavailable_reason;
    uint32_t hot_count = 0;
    llama_expert_uma_readiness resolved_readiness = LLAMA_EXPERT_UMA_READINESS_AUTO;
    int32_t last_execution_backend_device_type = -1;
    bool initialization_in_progress = false;
    bool descriptors_complete = false;
    bool active_request = false;
    bool request_failed = false;
    bool request_cancelled = false;
    bool pressure_circuit_open = false;
};

} // namespace

std::unique_ptr<llm_expert_weight_provider> llm_create_uma_cache_expert_weight_provider(
        llm_uma_cache_config config,
        llm_expert_provider_faults faults) {
    return std::make_unique<llm_uma_expert_weight_provider>(std::move(config), faults);
}
