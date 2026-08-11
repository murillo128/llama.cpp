#include "llama-expert-resident-demand.h"

#include "llama-perfetto-trace.h"

#include <algorithm>
#include <array>
#include <limits>
#include <mutex>
#include <stdexcept>

namespace {

bool same_key(llm_expert_key lhs, llm_expert_key rhs) {
    return lhs.layer == rhs.layer && lhs.expert == rhs.expert;
}

int expert_axis(const ggml_tensor * tensor, int32_t n_expert, bool weight) {
    if (tensor == nullptr) return -1;
    if (weight) {
        if (ggml_n_dims(tensor) < 3 || tensor->ne[2] != n_expert) return -1;
        for (int axis = 3; axis < ggml_n_dims(tensor); ++axis) {
            if (tensor->ne[axis] != 1) return -1;
        }
        return 2;
    }
    if (ggml_n_dims(tensor) < 2 || tensor->ne[1] != n_expert) return -1;
    for (int axis = 2; axis < ggml_n_dims(tensor); ++axis) {
        if (tensor->ne[axis] != 1) return -1;
    }
    return 1;
}

bool append_destination(
        std::array<llm_expert_storage_destination, 12> & destinations,
        size_t & count,
        ggml_tensor * target,
        const ggml_tensor * layout,
        int32_t target_n_expert,
        int32_t layout_n_expert,
        uint32_t slot,
        bool weight,
        llm_expert_storage_projection projection,
        llm_expert_storage_sidecar sidecar,
        llm_expert_layout_class_id layout_class_id) {
    if (target == nullptr || layout == nullptr) return target == layout;
    const int target_axis = expert_axis(target, target_n_expert, weight);
    const int layout_axis = expert_axis(layout, layout_n_expert, weight);
    if (target_axis < 0 || target_axis != layout_axis || target->data == nullptr ||
        target->nb[target_axis] < layout->nb[layout_axis] || slot >= uint32_t(target->ne[target_axis]) ||
        count >= destinations.size()) {
        return false;
    }
    destinations[count++] = {
        projection,
        sidecar,
        static_cast<uint8_t *>(target->data) + size_t(slot)*target->nb[target_axis],
        layout->nb[layout_axis],
        layout_class_id,
    };
    return true;
}

bool append_projection(
        std::array<llm_expert_storage_destination, 12> & destinations,
        size_t & count,
        const llm_expert_projection_descriptor & target,
        const llm_expert_projection_descriptor & layout,
        int32_t target_n_expert,
        int32_t layout_n_expert,
        uint32_t slot,
        llm_expert_storage_projection projection,
        llm_expert_layout_class_id layout_class_id) {
    return append_destination(destinations, count, target.weight, layout.weight,
               target_n_expert, layout_n_expert, slot, true, projection,
               llm_expert_storage_sidecar::weight, layout_class_id) &&
        append_destination(destinations, count, target.bias, layout.bias,
               target_n_expert, layout_n_expert, slot, false, projection,
               llm_expert_storage_sidecar::bias, layout_class_id) &&
        append_destination(destinations, count, target.scale, layout.scale,
               target_n_expert, layout_n_expert, slot, false, projection,
               llm_expert_storage_sidecar::scale, layout_class_id);
}

bool build_destinations(
        const llm_expert_bundle_descriptor & target,
        const llm_expert_bundle_descriptor & layout,
        llm_cold_reference reference,
        std::array<llm_expert_storage_destination, 12> & destinations,
        size_t & count) {
    count = 0;
    return append_projection(destinations, count, target.up, layout.up,
               target.n_expert, layout.n_expert, reference.slot,
               llm_expert_storage_projection::up, reference.layout_class_id) &&
        append_projection(destinations, count, target.gate, layout.gate,
               target.n_expert, layout.n_expert, reference.slot,
               llm_expert_storage_projection::gate, reference.layout_class_id) &&
        append_projection(destinations, count, target.gate_up, layout.gate_up,
               target.n_expert, layout.n_expert, reference.slot,
               llm_expert_storage_projection::gate_up, reference.layout_class_id) &&
        append_projection(destinations, count, target.down, layout.down,
               target.n_expert, layout.n_expert, reference.slot,
               llm_expert_storage_projection::down, reference.layout_class_id) && count != 0;
}

bool finalize_integrity(
        llm_expert_integrity_mode mode,
        llm_expert_storage & storage,
        const llm_expert_storage_destination * destinations,
        size_t destination_count,
        const llm_expert_async_read_completion & completion) {
    if (mode == llm_expert_integrity_mode::none) {
        if (completion.integrity_status != llm_expert_integrity_status::not_checked || completion.digest != 0) {
            storage.poison();
            return false;
        }
        storage.record_integrity_status(llm_expert_integrity_status::not_checked);
        return true;
    }
    if (completion.integrity_status != llm_expert_integrity_status::passed) {
        storage.record_integrity_status(llm_expert_integrity_status::failed);
        return false;
    }
    uint64_t digest = UINT64_C(1469598103934665603);
    uint64_t digest_bytes = 0;
    for (size_t index = 0; index < destination_count; ++index) {
        const auto * bytes = static_cast<const uint8_t *>(destinations[index].data);
        digest_bytes += destinations[index].extent;
        for (uint64_t offset = 0; offset < destinations[index].extent; ++offset) {
            digest ^= bytes[offset];
            digest *= UINT64_C(1099511628211);
        }
    }
    const bool matches = digest == completion.digest;
    storage.record_integrity_status(matches ? llm_expert_integrity_status::passed :
        llm_expert_integrity_status::failed, digest_bytes);
    return matches;
}

uint64_t digest_value(uint64_t digest, uint64_t value) {
    for (uint32_t byte = 0; byte < 8; ++byte) {
        digest ^= uint8_t(value >> (byte*8));
        digest *= UINT64_C(1099511628211);
    }
    return digest;
}

llm_expert_provider_error schedule_error(llm_expert_schedule_disposition disposition) {
    if (disposition == llm_expert_schedule_disposition::stale_generation) {
        return llm_expert_provider_error::stale_generation;
    }
    if (disposition == llm_expert_schedule_disposition::generation_exhausted) {
        return llm_expert_provider_error::generation_exhausted;
    }
    if (disposition == llm_expert_schedule_disposition::busy ||
        disposition == llm_expert_schedule_disposition::closed) {
        return llm_expert_provider_error::busy;
    }
    return llm_expert_provider_error::metadata_mismatch;
}

llm_expert_provider_error async_error(llm_expert_async_result result) {
    if (result == llm_expert_async_result::busy) return llm_expert_provider_error::busy;
    if (result == llm_expert_async_result::closed) return llm_expert_provider_error::cancelled;
    if (result == llm_expert_async_result::stale_generation) {
        return llm_expert_provider_error::stale_generation;
    }
    if (result == llm_expert_async_result::generation_exhausted) {
        return llm_expert_provider_error::generation_exhausted;
    }
    return llm_expert_provider_error::copy_failed;
}

llm_expert_provider_error storage_error(llm_expert_storage_error error) {
    if (error == llm_expert_storage_error::cancelled) return llm_expert_provider_error::cancelled;
    if (error == llm_expert_storage_error::invalid_configuration) {
        return llm_expert_provider_error::unsupported_configuration;
    }
    if (error == llm_expert_storage_error::invalid_directory ||
        error == llm_expert_storage_error::invalid_key ||
        error == llm_expert_storage_error::invalid_destination ||
        error == llm_expert_storage_error::poisoned) {
        return llm_expert_provider_error::metadata_mismatch;
    }
    return llm_expert_provider_error::copy_failed;
}

} // namespace

struct llm_host_resident_demand_coordinator::impl {
    explicit impl(llm_host_resident_demand_config config) : config(config) {
        if (config.cache == nullptr || config.storage == nullptr || config.scheduler == nullptr ||
            config.transport == nullptr || config.layout_registry == nullptr || !config.layout_registry->sealed() ||
            config.maximum_occurrences == 0 || config.maximum_unique_keys == 0 ||
            config.maximum_unique_keys > config.maximum_occurrences || config.trace_capacity == 0 ||
            !config.serial_control || (config.preflight != nullptr && config.reservation_bytes == 0) ||
            (config.base_hold_kind != llm_cold_reference_kind::request &&
             config.base_hold_kind != llm_cold_reference_kind::batch)) {
            throw std::invalid_argument("invalid host-resident demand coordinator configuration");
        }
        counters.events.reserve(config.trace_capacity);
    }

    llm_expert_provider_result release_terminal_owned(
            llm_host_resident_demand_entry & entry) noexcept {
        const auto released = config.scheduler->release_terminal(entry.scheduler_handle);
        if (released == llm_expert_schedule_disposition::admitted) {
            entry.scheduler_owned = false;
            entry.scheduler_handle = {};
            return llm_expert_provider_result::success();
        }
        if (released == llm_expert_schedule_disposition::stale_generation) {
            counters.stale_completions++;
            entry.scheduler_owned = false;
            entry.scheduler_handle = {};
        }
        return llm_expert_provider_result::failure(schedule_error(released));
    }

    llm_expert_provider_result terminalize_owned(
            llm_host_resident_demand_entry & entry,
            bool cancelled) noexcept {
        if (!entry.scheduler_owned || !entry.scheduler_handle.valid()) {
            return llm_expert_provider_result::success();
        }
        llm_expert_request_snapshot snapshot;
        const auto found = config.scheduler->snapshot(entry.scheduler_handle, snapshot);
        if (found == llm_expert_schedule_disposition::stale_generation) {
            counters.stale_completions++;
            entry.scheduler_owned = false;
            entry.scheduler_handle = {};
            return llm_expert_provider_result::failure(llm_expert_provider_error::stale_generation);
        }
        if (found != llm_expert_schedule_disposition::admitted) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
        }
        if (snapshot.state == llm_expert_request_state::complete ||
            snapshot.state == llm_expert_request_state::cancelled ||
            snapshot.state == llm_expert_request_state::failed) {
            return release_terminal_owned(entry);
        }
        auto state = snapshot.state;
        if (state != llm_expert_request_state::draining) {
            if (cancelled && state != llm_expert_request_state::cancelling) {
                const auto cancelling = config.scheduler->begin_demand_cancellation(
                    entry.scheduler_handle, state);
                if (cancelling != llm_expert_schedule_disposition::admitted) {
                    return llm_expert_provider_result::failure(schedule_error(cancelling));
                }
                state = llm_expert_request_state::cancelling;
            }
            const auto draining = config.scheduler->transition(
                entry.scheduler_handle, state, llm_expert_request_state::draining);
            if (draining != llm_expert_schedule_disposition::admitted) {
                return llm_expert_provider_result::failure(schedule_error(draining));
            }
        }
        const auto finished = config.scheduler->finish(entry.scheduler_handle,
            cancelled ? llm_expert_request_state::cancelled : llm_expert_request_state::failed);
        if (finished != llm_expert_schedule_disposition::admitted) {
            if (finished == llm_expert_schedule_disposition::stale_generation) {
                counters.stale_completions++;
                entry.scheduler_owned = false;
                entry.scheduler_handle = {};
            }
            return llm_expert_provider_result::failure(schedule_error(finished));
        }
        return release_terminal_owned(entry);
    }

    llm_expert_provider_result release_holds(llm_host_resident_demand_batch & batch) noexcept {
        auto result = llm_expert_provider_result::success();
        for (size_t order = 0; order < batch.unique_count; ++order) {
            auto & entry = batch.entries[batch.canonical_order[order]];
            if (!entry.request_hold) continue;
            const auto released = config.cache->release(entry.reference, config.base_hold_kind);
            if (result.is_ready() && !released.is_ready()) result = released;
            if (released.is_ready()) {
                entry.request_hold = false;
                if (counters.current_request_holds != 0) counters.current_request_holds--;
            }
        }
        return result;
    }

    llm_expert_provider_result fail_batch(
            llm_host_resident_demand_batch & batch,
            size_t current,
            llm_expert_provider_error error,
            bool cancelled) noexcept {
        for (size_t index = 0; index <= current && index < batch.unique_count; ++index) {
            auto & entry = batch.entries[batch.canonical_order[index]];
            if (entry.scheduler_owned) (void) terminalize_owned(entry, cancelled);
            if (entry.lookup == llm_cold_demand_lookup::reserved && !config.cache->ready(entry.reference)) {
                (void) config.cache->fail_reservation(entry.key, entry.reference);
            }
        }
        (void) release_holds(batch);
        (void) config.cache->cleanup_failed_slots();
        counters.failures++;
        if (cancelled) counters.cancellations++;
        return llm_expert_provider_result::failure(cancelled ?
            llm_expert_provider_error::cancelled : error);
    }

    void record_hold(llm_host_resident_demand_event & event) noexcept {
        counters.current_request_holds++;
        counters.peak_request_holds = std::max(
            counters.peak_request_holds, counters.current_request_holds);
        event.request_holds++;
    }

    llm_host_resident_demand_config config;
    mutable std::mutex mutex;
    llm_host_resident_demand_diagnostics counters;
};

llm_host_resident_demand_coordinator::llm_host_resident_demand_coordinator(
        llm_host_resident_demand_config config) : pimpl(std::make_unique<impl>(config)) {}

llm_host_resident_demand_coordinator::~llm_host_resident_demand_coordinator() = default;

llm_expert_provider_result llm_host_resident_demand_coordinator::resolve(
        int32_t layer,
        const int32_t * logical_ids,
        size_t logical_count,
        llm_host_resident_demand_batch & batch,
        bool (*abort_callback)(void *),
        void * abort_data) noexcept {
    LLM_EXPERT_TRACE_SCOPE("k3.provider", "host_resident_demand", "layer", layer,
        "selected_occurrences", logical_count, "serial_control", pimpl->config.serial_control);
    std::lock_guard<std::mutex> guard(pimpl->mutex);
    batch.occurrence_count = 0;
    batch.unique_count = 0;
    if (logical_ids == nullptr || logical_count == 0 ||
        logical_count > pimpl->config.maximum_occurrences || layer < 0 ||
        size_t(layer) >= pimpl->config.layout_registry->layer_ids.size()) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::invalid_key);
    }
    const auto layer_class = pimpl->config.layout_registry->layer_ids[size_t(layer)];
    if (layer_class >= pimpl->config.layout_registry->classes.size()) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::invalid_key);
    }
    try {
        batch.entries.assign(pimpl->config.maximum_unique_keys, {});
        batch.occurrence_to_unique.assign(logical_count, 0);
        batch.canonical_order.assign(pimpl->config.maximum_unique_keys, 0);
    } catch (...) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::allocation_failed);
    }
    batch.occurrence_count = logical_count;
    const uint32_t experts_per_layer = uint32_t(
        pimpl->config.layout_registry->classes[layer_class].prototype.n_expert);
    for (size_t occurrence = 0; occurrence < logical_count; ++occurrence) {
        if (logical_ids[occurrence] < 0 || uint32_t(logical_ids[occurrence]) >= experts_per_layer) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::invalid_key);
        }
        const llm_expert_key key = { layer, logical_ids[occurrence] };
        size_t unique = 0;
        while (unique < batch.unique_count && !same_key(batch.entries[unique].key, key)) unique++;
        if (unique == batch.unique_count) {
            if (batch.unique_count == pimpl->config.maximum_unique_keys) {
                return llm_expert_provider_result::failure(
                    llm_expert_provider_error::unsupported_configuration);
            }
            batch.entries[unique].key = key;
            batch.canonical_order[unique] = uint32_t(unique);
            batch.unique_count++;
        }
        batch.occurrence_to_unique[occurrence] = uint32_t(unique);
        batch.entries[unique].occurrence_count++;
    }
    std::sort(batch.canonical_order.begin(), batch.canonical_order.begin() + batch.unique_count,
        [&](uint32_t lhs, uint32_t rhs) {
            const auto & lhs_key = batch.entries[lhs].key;
            const auto & rhs_key = batch.entries[rhs].key;
            return lhs_key.layer != rhs_key.layer ? lhs_key.layer < rhs_key.layer :
                lhs_key.expert < rhs_key.expert;
        });

    llm_host_resident_demand_event event;
    event.sequence = pimpl->counters.batches;
    event.layer = layer;
    event.selected_occurrences = uint32_t(logical_count);
    event.unique_keys = uint32_t(batch.unique_count);
    event.serial_control = true;
    const uint64_t transport_epoch = pimpl->config.transport->diagnostics().transport_epoch;
    for (size_t order = 0; order < batch.unique_count; ++order) {
        auto & entry = batch.entries[batch.canonical_order[order]];
        auto found = pimpl->config.cache->reserve_or_join_demand(entry.key, entry.reference, entry.lookup);
        if (!found.is_ready()) {
            return pimpl->fail_batch(batch, order, found.error, false);
        }
        if (entry.lookup == llm_cold_demand_lookup::ready) {
            event.ready_hits++;
            found = pimpl->config.cache->acquire(entry.reference, pimpl->config.base_hold_kind);
            if (!found.is_ready()) return pimpl->fail_batch(batch, order, found.error, false);
            entry.request_hold = true;
            pimpl->record_hold(event);
        } else if (entry.lookup == llm_cold_demand_lookup::joined_loading) {
            event.joined_loads++;
            llm_expert_request_metadata metadata;
            metadata.layout_class_id = entry.reference.layout_class_id;
            event.scheduler_enqueue_attempts++;
            event.last_enqueue_us = ggml_time_us();
            const auto scheduled = pimpl->config.scheduler->enqueue(
                entry.key, llm_expert_priority::demand_current_layer,
                llm_expert_readiness::host_ready, metadata);
            if (scheduled.disposition != llm_expert_schedule_disposition::joined) {
                if (scheduled.disposition == llm_expert_schedule_disposition::admitted) {
                    entry.scheduler_handle = scheduled.handle;
                    entry.scheduler_owned = true;
                }
                return pimpl->fail_batch(batch, order, schedule_error(scheduled.disposition), false);
            }
            entry.scheduler_handle = scheduled.handle;
            entry.scheduler_joined = true;
            event.scheduler_joins++;
            if (event.first_wait_us == 0) event.first_wait_us = ggml_time_us();
            found = pimpl->config.cache->wait_until_ready(entry.reference);
            if (!found.is_ready()) return pimpl->fail_batch(batch, order, found.error, false);
            found = pimpl->config.cache->acquire(entry.reference, pimpl->config.base_hold_kind);
            if (!found.is_ready()) return pimpl->fail_batch(batch, order, found.error, false);
            entry.request_hold = true;
            pimpl->record_hold(event);
        } else if (entry.lookup == llm_cold_demand_lookup::reserved) {
            event.new_reservations++;
            if (pimpl->config.preflight != nullptr) {
                found = pimpl->config.preflight(
                    pimpl->config.preflight_data, pimpl->config.reservation_bytes);
                if (!found.is_ready()) return pimpl->fail_batch(batch, order, found.error, false);
            }
            llm_expert_request_metadata metadata;
            metadata.layout_class_id = entry.reference.layout_class_id;
            event.scheduler_enqueue_attempts++;
            event.last_enqueue_us = ggml_time_us();
            const auto scheduled = pimpl->config.scheduler->enqueue(
                entry.key, llm_expert_priority::demand_current_layer,
                llm_expert_readiness::host_ready, metadata);
            if (scheduled.disposition != llm_expert_schedule_disposition::admitted) {
                return pimpl->fail_batch(batch, order, schedule_error(scheduled.disposition), false);
            }
            entry.scheduler_handle = scheduled.handle;
            entry.scheduler_owned = true;
            event.scheduler_admissions++;
            llm_expert_request_snapshot selected;
            const auto taken = pimpl->config.scheduler->take_next(selected);
            if (taken.disposition != llm_expert_schedule_disposition::admitted ||
                selected.handle.slot != entry.scheduler_handle.slot ||
                selected.handle.generation != entry.scheduler_handle.generation) {
                return pimpl->fail_batch(batch, order,
                    taken.disposition == llm_expert_schedule_disposition::admitted ?
                        llm_expert_provider_error::metadata_mismatch : schedule_error(taken.disposition), false);
            }
            const auto & target = pimpl->config.cache->bundle_for_key(entry.key);
            if (entry.reference.layout_class_id >= pimpl->config.layout_registry->classes.size()) {
                return pimpl->fail_batch(batch, order, llm_expert_provider_error::invalid_descriptor, false);
            }
            const auto & layout = pimpl->config.layout_registry->classes[
                entry.reference.layout_class_id].prototype;
            std::array<llm_expert_storage_destination, 12> destinations;
            std::array<llm_expert_storage_read_operation, 12> operations;
            size_t destination_count = 0;
            size_t operation_count = 0;
            if (!build_destinations(target, layout, entry.reference, destinations, destination_count)) {
                return pimpl->fail_batch(batch, order, llm_expert_provider_error::invalid_descriptor, false);
            }
            const auto planned = pimpl->config.storage->make_read_plan(
                entry.key, destinations.data(), destination_count,
                operations.data(), operations.size(), operation_count);
            if (!planned.is_ready() || operation_count == 0) {
                pimpl->config.storage->poison();
                return pimpl->fail_batch(batch, order,
                    planned.is_ready() ? llm_expert_provider_error::metadata_mismatch : storage_error(planned.error),
                    planned.error == llm_expert_storage_error::cancelled);
            }
            uint64_t read_bytes = 0;
            for (size_t operation = 0; operation < operation_count; ++operation) {
                if (operations[operation].byte_count > UINT64_MAX - read_bytes) {
                    return pimpl->fail_batch(batch, order,
                        llm_expert_provider_error::unsupported_configuration, false);
                }
                read_bytes += operations[operation].byte_count;
            }
            const llm_expert_async_operation_identity identity = {
                transport_epoch,
                entry.scheduler_handle,
                0,
                entry.key,
                llm_expert_readiness::host_ready,
                llm_expert_priority::demand_current_layer,
                entry.reference.layout_class_id,
            };
            const auto submitted = pimpl->config.transport->submit_read_plan(
                identity, operations.data(), operation_count);
            if (submitted != llm_expert_async_result::ready ||
                pimpl->config.scheduler->transition(entry.scheduler_handle,
                    llm_expert_request_state::submitting,
                    llm_expert_request_state::io_in_flight) != llm_expert_schedule_disposition::admitted) {
                if (submitted == llm_expert_async_result::ready) {
                    (void) pimpl->config.transport->cancel_read(entry.scheduler_handle);
                    llm_expert_async_read_completion discarded;
                    (void) pimpl->config.transport->wait_read(entry.scheduler_handle, discarded);
                    (void) pimpl->config.transport->release_read(entry.scheduler_handle);
                }
                return pimpl->fail_batch(batch, order,
                    submitted == llm_expert_async_result::ready ?
                        llm_expert_provider_error::metadata_mismatch : async_error(submitted),
                    submitted == llm_expert_async_result::closed);
            }
            event.read_plans++;
            event.read_operations += uint32_t(operation_count);
            event.read_bytes += read_bytes;
            event.last_read_submit_us = ggml_time_us();
            if (event.first_wait_us == 0) event.first_wait_us = ggml_time_us();
            const auto before_wait = pimpl->config.transport->diagnostics();
            event.peak_active_read_requests = std::max(
                event.peak_active_read_requests, before_wait.active_read_requests);
            event.peak_active_read_operations = std::max(
                event.peak_active_read_operations, before_wait.active_operations);
            llm_expert_async_read_completion completion;
            const auto waited = pimpl->config.transport->wait_read(
                entry.scheduler_handle, completion, abort_callback, abort_data);
            const auto released = pimpl->config.transport->release_read(entry.scheduler_handle);
            const bool cancelled = waited == llm_expert_async_result::closed;
            const llm_expert_storage_error storage_error = waited == llm_expert_async_result::ready ?
                llm_expert_storage_error::none : cancelled ? llm_expert_storage_error::cancelled :
                completion.native_error == 0 ? llm_expert_storage_error::short_read :
                llm_expert_storage_error::io_error;
            pimpl->config.storage->record_async_read(
                operation_count, completion.bytes_completed, storage_error, completion.native_error);
            const bool integrity_matches = waited == llm_expert_async_result::ready &&
                released == llm_expert_async_result::ready &&
                finalize_integrity(pimpl->config.integrity_mode, *pimpl->config.storage,
                    destinations.data(), destination_count, completion);
            if (waited != llm_expert_async_result::ready ||
                released != llm_expert_async_result::ready || !integrity_matches) {
                return pimpl->fail_batch(batch, order,
                    waited != llm_expert_async_result::ready ? async_error(waited) :
                    released != llm_expert_async_result::ready ? async_error(released) :
                        llm_expert_provider_error::metadata_mismatch,
                    cancelled);
            }
            pimpl->counters.physical_completion_digest = digest_value(
                pimpl->counters.physical_completion_digest, uint64_t(uint32_t(entry.key.layer)));
            pimpl->counters.physical_completion_digest = digest_value(
                pimpl->counters.physical_completion_digest, uint64_t(uint32_t(entry.key.expert)));
            found = pimpl->config.cache->publish_ready_and_acquire(
                entry.key, entry.reference, pimpl->config.base_hold_kind);
            if (!found.is_ready()) return pimpl->fail_batch(batch, order, found.error, false);
            entry.request_hold = true;
            pimpl->record_hold(event);
            const auto transitioned = pimpl->config.scheduler->transition(entry.scheduler_handle,
                llm_expert_request_state::io_in_flight, llm_expert_request_state::host_ready);
            if (transitioned != llm_expert_schedule_disposition::admitted) {
                return pimpl->fail_batch(batch, order, schedule_error(transitioned), false);
            }
        } else {
            return pimpl->fail_batch(batch, order, llm_expert_provider_error::metadata_mismatch, false);
        }
        event.host_ready++;
        pimpl->counters.canonical_commit_digest = digest_value(
            pimpl->counters.canonical_commit_digest, uint64_t(uint32_t(entry.key.layer)));
        pimpl->counters.canonical_commit_digest = digest_value(
            pimpl->counters.canonical_commit_digest, uint64_t(uint32_t(entry.key.expert)));
    }
    event.first_wait_after_all_enqueue_attempts = event.first_wait_us == 0 ||
        event.scheduler_enqueue_attempts <= 1;
    event.first_wait_after_all_admissible_submissions = event.first_wait_us == 0 || event.read_plans <= 1;
    pimpl->counters.batches++;
    if (pimpl->counters.events.size() == pimpl->config.trace_capacity) {
        pimpl->counters.events.erase(pimpl->counters.events.begin());
    }
    pimpl->counters.events.push_back(event);
    return llm_expert_provider_result::success();
}

llm_expert_provider_result llm_host_resident_demand_coordinator::complete_host_scheduler(
        llm_host_resident_demand_entry & entry) noexcept {
    std::lock_guard<std::mutex> guard(pimpl->mutex);
    if (!entry.scheduler_owned || !entry.scheduler_handle.valid()) {
        return llm_expert_provider_result::success();
    }
    llm_expert_request_snapshot snapshot;
    const auto found = pimpl->config.scheduler->snapshot(entry.scheduler_handle, snapshot);
    if (found != llm_expert_schedule_disposition::admitted) {
        if (found == llm_expert_schedule_disposition::stale_generation) {
            pimpl->counters.stale_completions++;
            entry.scheduler_owned = false;
            entry.scheduler_handle = {};
        }
        return llm_expert_provider_result::failure(schedule_error(found));
    }
    if (snapshot.state != llm_expert_request_state::complete) {
        const auto finished = pimpl->config.scheduler->finish(
            entry.scheduler_handle, llm_expert_request_state::complete);
        if (finished != llm_expert_schedule_disposition::admitted) {
            if (finished == llm_expert_schedule_disposition::stale_generation) {
                pimpl->counters.stale_completions++;
                entry.scheduler_owned = false;
                entry.scheduler_handle = {};
            }
            return llm_expert_provider_result::failure(schedule_error(finished));
        }
    }
    return pimpl->release_terminal_owned(entry);
}

llm_expert_provider_result llm_host_resident_demand_coordinator::fail_host_scheduler(
        llm_host_resident_demand_entry & entry,
        bool cancelled) noexcept {
    std::lock_guard<std::mutex> guard(pimpl->mutex);
    return pimpl->terminalize_owned(entry, cancelled);
}

llm_expert_provider_result llm_host_resident_demand_coordinator::fail_host_schedulers(
        llm_host_resident_demand_batch & batch,
        bool cancelled) noexcept {
    std::lock_guard<std::mutex> guard(pimpl->mutex);
    auto result = llm_expert_provider_result::success();
    for (size_t order = 0; order < batch.unique_count; ++order) {
        auto & entry = batch.entries[batch.canonical_order[order]];
        if (!entry.scheduler_owned) continue;
        const auto terminal = pimpl->terminalize_owned(entry, cancelled);
        if (result.is_ready() && !terminal.is_ready()) result = terminal;
    }
    return result;
}

llm_expert_provider_result llm_host_resident_demand_coordinator::release_request_hold(
        llm_host_resident_demand_entry & entry) noexcept {
    std::lock_guard<std::mutex> guard(pimpl->mutex);
    if (!entry.request_hold) return llm_expert_provider_result::success();
    const auto released = pimpl->config.cache->release(
        entry.reference, pimpl->config.base_hold_kind);
    if (released.is_ready()) {
        entry.request_hold = false;
        if (pimpl->counters.current_request_holds != 0) pimpl->counters.current_request_holds--;
    }
    return released;
}

llm_expert_provider_result llm_host_resident_demand_coordinator::transfer_request_hold(
        llm_host_resident_demand_entry & entry) noexcept {
    std::lock_guard<std::mutex> guard(pimpl->mutex);
    if (!entry.request_hold || pimpl->counters.current_request_holds == 0) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
    }
    if (pimpl->config.base_hold_kind == llm_cold_reference_kind::batch) {
        const auto converted = pimpl->config.cache->convert_batch_to_request(entry.reference);
        if (!converted.is_ready()) return converted;
    }
    entry.request_hold = false;
    pimpl->counters.current_request_holds--;
    return llm_expert_provider_result::success();
}

llm_expert_provider_result llm_host_resident_demand_coordinator::transfer_request_hold_to_cpu_execution(
        llm_host_resident_demand_entry & entry) noexcept {
    std::lock_guard<std::mutex> guard(pimpl->mutex);
    if (!entry.request_hold || pimpl->counters.current_request_holds == 0) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
    }
    if (pimpl->config.base_hold_kind != llm_cold_reference_kind::request) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
    }
    const auto converted = pimpl->config.cache->convert_request_to_cpu_execution(entry.reference);
    if (!converted.is_ready()) return converted;
    entry.request_hold = false;
    pimpl->counters.current_request_holds--;
    return llm_expert_provider_result::success();
}

llm_expert_provider_result llm_host_resident_demand_coordinator::release_request_holds(
        llm_host_resident_demand_batch & batch) noexcept {
    std::lock_guard<std::mutex> guard(pimpl->mutex);
    return pimpl->release_holds(batch);
}

llm_host_resident_demand_diagnostics llm_host_resident_demand_coordinator::diagnostics() const {
    std::lock_guard<std::mutex> guard(pimpl->mutex);
    return pimpl->counters;
}
