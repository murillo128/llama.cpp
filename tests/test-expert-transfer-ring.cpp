#include "llama-expert-transfer-ring.h"
#include "llama-expert-scheduler.h"
#include "llama-expert-storage.h"
#include "llama-hparams.h"

#include "ggml-cpp.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

struct fixture {
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buffer;
    ggml_tensor * up = nullptr;
    ggml_tensor * gate = nullptr;
    ggml_tensor * down = nullptr;
    int32_t n_expert;

    explicit fixture(int32_t n_expert, bool fill = true,
            ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type(),
            ggml_type type = GGML_TYPE_F32, int64_t width = 8) : n_expert(n_expert) {
        ggml_init_params params = { ggml_tensor_overhead()*8, nullptr, true };
        ctx.reset(ggml_init(params));
        GGML_ASSERT(ctx);
        up = ggml_new_tensor_3d(ctx.get(), type, width, 16, n_expert);
        gate = ggml_new_tensor_3d(ctx.get(), type, width, 16, n_expert);
        down = ggml_new_tensor_3d(ctx.get(), type, width, 16, n_expert);
        buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft));
        GGML_ASSERT(buffer);
        if (fill) {
            uint8_t pattern = 0x20;
            for (auto * tensor : { up, gate, down }) {
                for (int32_t expert = 0; expert < n_expert; ++expert) {
                    std::vector<uint8_t> bytes(tensor->nb[2], pattern + expert);
                    ggml_backend_tensor_set(tensor, bytes.data(), size_t(expert)*tensor->nb[2], bytes.size());
                }
                pattern += 0x20;
            }
        }
    }

    llm_expert_bundle_descriptor bundle(int32_t layer = 0) const {
        return {
            layer,
            n_expert,
            llm_expert_projection_descriptor::from(up, nullptr, nullptr),
            llm_expert_projection_descriptor::from(gate, nullptr, nullptr),
            {},
            llm_expert_projection_descriptor::from(down, nullptr, nullptr),
        };
    }
};

ggml_backend_dev_t cpu_device() {
    static const bool loaded = [] { ggml_backend_load_all(); return true; }();
    (void) loaded;
    auto * result = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    GGML_ASSERT(result);
    return result;
}

llm_cold_expert_cache make_cold(const fixture & source) {
    llm_cold_cache_config config;
    config.byte_budget = 1U << 20;
    config.minimum_slots = 2;
    config.routed_layer_count = 1;
    config.total_expert_keys = uint32_t(source.n_expert);
    llm_cold_expert_cache cache(config);
    GGML_ASSERT(cache.initialize(source.bundle()).is_ready());
    return cache;
}

llm_transfer_ring_config ring_config(uint64_t bytes, uint32_t minimum = 2, uint64_t generation = 0) {
    return { bytes, minimum, cpu_device(), true, true, false, generation };
}

void assert_slot_matches(const fixture & source, const fixture & hot, int32_t expert, uint32_t slot) {
    const auto source_bundle = source.bundle();
    const auto hot_bundle = hot.bundle();
    for (const auto & pair : {
            std::pair<const ggml_tensor *, const ggml_tensor *>(source_bundle.up.weight, hot_bundle.up.weight),
            std::pair<const ggml_tensor *, const ggml_tensor *>(source_bundle.gate.weight, hot_bundle.gate.weight),
            std::pair<const ggml_tensor *, const ggml_tensor *>(source_bundle.down.weight, hot_bundle.down.weight) }) {
        const size_t span = pair.first->nb[2];
        std::vector<uint8_t> expected(span), actual(span);
        ggml_backend_tensor_get(pair.first, expected.data(), size_t(expert)*span, span);
        ggml_backend_tensor_get(pair.second, actual.data(), size_t(slot)*span, span);
        GGML_ASSERT(expected == actual);
    }
}

void assert_bundle_slot_matches(
        const fixture & source,
        const llm_expert_bundle_descriptor & destination,
        int32_t expert,
        uint32_t slot) {
    const auto source_bundle = source.bundle();
    for (const auto & pair : {
            std::pair<const ggml_tensor *, const ggml_tensor *>(source_bundle.up.weight, destination.up.weight),
            std::pair<const ggml_tensor *, const ggml_tensor *>(source_bundle.gate.weight, destination.gate.weight),
            std::pair<const ggml_tensor *, const ggml_tensor *>(source_bundle.down.weight, destination.down.weight) }) {
        const size_t span = pair.first->nb[2];
        std::vector<uint8_t> expected(span), actual(span);
        ggml_backend_tensor_get(pair.first, expected.data(), size_t(expert)*span, span);
        ggml_backend_tensor_get(pair.second, actual.data(), size_t(slot)*pair.second->nb[2], span);
        GGML_ASSERT(expected == actual);
    }
}

uint64_t populate_direct_lane(
        llm_expert_transfer_ring & ring,
        llm_transfer_lane_reference lane,
        const fixture & source,
        int32_t expert) {
    std::array<llm_expert_storage_destination, 12> destinations;
    size_t destination_count = 0;
    GGML_ASSERT(ring.storage_destinations(
        lane, destinations.data(), destinations.size(), destination_count).is_ready());
    const auto source_bundle = source.bundle();
    uint64_t copied = 0;
    for (size_t index = 0; index < destination_count; ++index) {
        const auto & destination = destinations[index];
        const ggml_tensor * tensor = destination.projection == llm_expert_storage_projection::up ?
            source_bundle.up.weight : destination.projection == llm_expert_storage_projection::gate ?
                source_bundle.gate.weight : source_bundle.down.weight;
        GGML_ASSERT(destination.sidecar == llm_expert_storage_sidecar::weight && tensor != nullptr &&
            destination.extent == tensor->nb[2]);
        std::memcpy(destination.data,
            static_cast<const uint8_t *>(tensor->data) + size_t(expert)*tensor->nb[2], destination.extent);
        copied += destination.extent;
    }
    return copied;
}

void test_budget_fallback_and_wave() {
    fixture source(4);
    fixture hot(2, false);
    auto cold = make_cold(source);
    llm_cold_reference cold_zero, cold_one;
    GGML_ASSERT(cold.find_or_admit({ 0, 0 }, source.bundle(), cold_zero).is_ready());
    GGML_ASSERT(cold.find_or_admit({ 0, 1 }, source.bundle(), cold_one).is_ready());

    llm_expert_transfer_ring discovery(ring_config(1U << 20));
    GGML_ASSERT(discovery.initialize(source.bundle()).is_ready());
    const uint64_t two_lane_budget = discovery.diagnostics().lane_footprint*2;
    llm_expert_transfer_ring ring(ring_config(two_lane_budget));
    GGML_ASSERT(ring.initialize(source.bundle()).is_ready());
    auto diagnostics = ring.diagnostics();
    GGML_ASSERT(diagnostics.effective_lanes == 2);
    GGML_ASSERT(diagnostics.actual_bytes <= diagnostics.requested_bytes);
    GGML_ASSERT(diagnostics.pageable_fallback);
    GGML_ASSERT(diagnostics.pinned_or_registered_bytes == 0);
    GGML_ASSERT(diagnostics.acquisition_method == "pageable-cpu");

    llm_transfer_lane_reference lane_zero, lane_one;
    GGML_ASSERT(ring.reserve(cold, cold_zero, 0, 1, lane_zero).is_ready());
    GGML_ASSERT(ring.reserve(cold, cold_one, 1, 1, lane_one).is_ready());
    GGML_ASSERT(ring.stage(lane_zero, cold.bundle()).is_ready());
    GGML_ASSERT(ring.stage(lane_one, cold.bundle()).is_ready());
    GGML_ASSERT(ring.transfer_wave(nullptr, {
        { lane_zero, hot.bundle(), 0 }, { lane_one, hot.bundle(), 1 },
    }).is_ready());
    assert_slot_matches(source, hot, 0, 0);
    assert_slot_matches(source, hot, 1, 1);
    diagnostics = ring.diagnostics();
    GGML_ASSERT(diagnostics.waves == 1);
    GGML_ASSERT(diagnostics.wave_synchronizations == 0);
    GGML_ASSERT(diagnostics.async_enqueues == 0);
    GGML_ASSERT(diagnostics.synchronous_copies == 6);
    GGML_ASSERT(diagnostics.h2d_bytes == diagnostics.lane_payload_bytes*2);
    GGML_ASSERT(cold.diagnostics().current_transfer_refs == 0);
    GGML_ASSERT(ring.validate_invariants().is_ready());
}

void test_direct_storage_lane_bypasses_cold_staging() {
    fixture source(4);
    fixture hot(2, false);
    llm_expert_transfer_ring ring(ring_config(1U << 20));
    GGML_ASSERT(ring.initialize(source.bundle()).is_ready());

    llm_transfer_lane_reference lane;
    const llm_expert_flight_id flight = { 1, 2, 3, { 0, 2 }, 0, 0 };
    GGML_ASSERT(ring.reserve_direct_storage(0, 1, 7, lane, flight).is_ready());
    std::array<llm_expert_storage_destination, 12> destinations;
    size_t destination_count = 0;
    GGML_ASSERT(ring.storage_destinations(
        lane, destinations.data(), destinations.size(), destination_count).is_ready());
    GGML_ASSERT(destination_count == 3);
    uint64_t copied = 0;
    const auto source_bundle = source.bundle();
    for (size_t index = 0; index < destination_count; ++index) {
        const auto & destination = destinations[index];
        const ggml_tensor * tensor = destination.projection == llm_expert_storage_projection::up ?
            source_bundle.up.weight : destination.projection == llm_expert_storage_projection::gate ?
                source_bundle.gate.weight : source_bundle.down.weight;
        GGML_ASSERT(destination.sidecar == llm_expert_storage_sidecar::weight &&
            tensor != nullptr && destination.extent == tensor->nb[2]);
        std::memcpy(destination.data,
            static_cast<const uint8_t *>(tensor->data) + 2*tensor->nb[2], destination.extent);
        copied += destination.extent;
    }
    GGML_ASSERT(ring.complete_direct_storage(lane, copied).is_ready());
    GGML_ASSERT(ring.complete_direct_storage(lane, copied).error ==
        llm_expert_provider_error::metadata_mismatch);
    GGML_ASSERT(ring.diagnostics().lanes[lane.lane].direct_storage_complete);
    GGML_ASSERT(ring.transfer_wave(nullptr, { { lane, hot.bundle(), 1 } }).is_ready());
    assert_slot_matches(source, hot, 2, 1);
    auto diagnostics = ring.diagnostics();
    GGML_ASSERT(diagnostics.direct_storage_reservations == 1 &&
        diagnostics.direct_storage_completions == 1 && diagnostics.direct_storage_bytes == copied &&
        diagnostics.stage_bytes == 0 && diagnostics.lanes[lane.lane].state == llm_transfer_lane_state::free);
    GGML_ASSERT(ring.validate_invariants().is_ready());

    GGML_ASSERT(ring.reserve_direct_storage(0, 0, 8, lane).is_ready());
    GGML_ASSERT(ring.discard_staging(lane).is_ready());
    diagnostics = ring.diagnostics();
    GGML_ASSERT(diagnostics.direct_storage_reservations == 2 &&
        diagnostics.direct_storage_completions == 1 &&
        diagnostics.lanes[lane.lane].state == llm_transfer_lane_state::free);
    GGML_ASSERT(ring.validate_invariants().is_ready());
}

void test_three_layout_classes_reuse_universal_lanes() {
    fixture iq2(4, true, ggml_backend_cpu_buffer_type(), GGML_TYPE_IQ2_XS, 256);
    fixture iq3(4, true, ggml_backend_cpu_buffer_type(), GGML_TYPE_IQ3_XXS, 256);
    fixture mxfp4(4, true, ggml_backend_cpu_buffer_type(), GGML_TYPE_MXFP4, 256);
    const std::array<const fixture *, 3> sources = { &iq2, &iq3, &mxfp4 };

    llm_expert_layout_registry registry;
    registry.layer_ids.assign(LLAMA_MAX_LAYERS, LLM_EXPERT_LAYOUT_CLASS_INVALID);
    for (size_t index = 0; index < sources.size(); ++index) {
        const auto bundle = sources[index]->bundle(int32_t(index));
        const uint64_t payload = bundle.up.weight->nb[2] +
            bundle.gate.weight->nb[2] + bundle.down.weight->nb[2];
        registry.classes.push_back({ llm_expert_layout_class_id(index), index + 1, payload, bundle });
        registry.layer_ids[index] = llm_expert_layout_class_id(index);
    }

    llm_cold_cache_config cold_config;
    cold_config.byte_budget = 1U << 20;
    cold_config.minimum_slots = 2;
    cold_config.routed_layer_count = 3;
    cold_config.total_expert_keys = 12;
    cold_config.routed_layers = { 0, 1, 2 };
    llm_cold_expert_cache cold(cold_config);
    GGML_ASSERT(cold.initialize(registry).is_ready());

    llm_expert_transfer_ring ring(ring_config(1U << 20));
    GGML_ASSERT(ring.initialize(registry).is_ready());
    const auto initial = ring.diagnostics();
    GGML_ASSERT(initial.layout_class_count == 3);
    GGML_ASSERT(initial.class_payload_bytes.size() == 3);
    GGML_ASSERT(initial.role_offsets.size() == 12);
    GGML_ASSERT(initial.role_extents.size() == 12);
    GGML_ASSERT(initial.lane_footprint >= *std::max_element(
        initial.class_payload_bytes.begin(), initial.class_payload_bytes.end()));

    uint64_t expected_useful_bytes = 0;
    for (int32_t layer = 0; layer < 3; ++layer) {
        fixture hot(2, false, ggml_backend_cpu_buffer_type(),
            sources[size_t(layer)]->up->type, 256);
        llm_cold_reference cold_reference;
        GGML_ASSERT(cold.find_or_admit(
            { layer, 0 }, sources[size_t(layer)]->bundle(layer), cold_reference).is_ready());
        GGML_ASSERT(cold_reference.layout_class_id == registry.layer_ids[size_t(layer)]);
        llm_transfer_lane_reference lane;
        GGML_ASSERT(ring.reserve(cold, cold_reference, 0, uint64_t(layer + 1), lane).is_ready());
        GGML_ASSERT(lane.layout_class_id == cold_reference.layout_class_id);
        GGML_ASSERT(ring.stage(lane, cold.bundle(cold_reference.layout_class_id)).is_ready());
        GGML_ASSERT(ring.transfer_wave(nullptr, {
            { lane, hot.bundle(layer), 0 },
        }).is_ready());
        assert_slot_matches(*sources[size_t(layer)], hot, 0, 0);
        expected_useful_bytes += registry.classes[size_t(lane.layout_class_id)].payload_bytes;
    }
    const auto diagnostics = ring.diagnostics();
    GGML_ASSERT(diagnostics.h2d_bytes == expected_useful_bytes);
    GGML_ASSERT(diagnostics.h2d_bytes < diagnostics.lane_footprint*3);
    GGML_ASSERT(diagnostics.waves == 3);
    GGML_ASSERT(diagnostics.class_stage_bundles == std::vector<uint64_t>({ 1, 1, 1 }));
    GGML_ASSERT(diagnostics.class_stage_bytes == diagnostics.class_payload_bytes);
    GGML_ASSERT(diagnostics.class_h2d_bundles == std::vector<uint64_t>({ 1, 1, 1 }));
    GGML_ASSERT(diagnostics.class_h2d_bytes == diagnostics.class_payload_bytes);
    GGML_ASSERT(ring.validate_invariants().is_ready());
}

void test_failures_cleanup_and_busy_surrender() {
    fixture source(4);
    fixture hot(2, false);
    auto cold = make_cold(source);
    llm_cold_reference cold_zero;
    GGML_ASSERT(cold.find_or_admit({ 0, 0 }, source.bundle(), cold_zero).is_ready());

    llm_expert_transfer_ring stage_failure(ring_config(1U << 20), { false, 1, false, false });
    GGML_ASSERT(stage_failure.initialize(source.bundle()).is_ready());
    llm_transfer_lane_reference lane;
    GGML_ASSERT(stage_failure.reserve(cold, cold_zero, 0, 1, lane).is_ready());
    GGML_ASSERT(stage_failure.surrender().error == llm_expert_provider_error::busy);
    GGML_ASSERT(stage_failure.stage(lane, cold.bundle()).error == llm_expert_provider_error::copy_failed);
    GGML_ASSERT(stage_failure.cleanup_failed_lanes().is_ready());
    GGML_ASSERT(cold.diagnostics().current_transfer_refs == 0);
    GGML_ASSERT(stage_failure.surrender().is_ready());

    llm_expert_transfer_ring enqueue_failure(ring_config(1U << 20), { false, SIZE_MAX, true, false });
    GGML_ASSERT(enqueue_failure.initialize(source.bundle()).is_ready());
    GGML_ASSERT(enqueue_failure.reserve(cold, cold_zero, 0, 2, lane).is_ready());
    GGML_ASSERT(enqueue_failure.stage(lane, cold.bundle()).is_ready());
    GGML_ASSERT(enqueue_failure.transfer_wave(nullptr, { { lane, hot.bundle(), 0 } }).error ==
        llm_expert_provider_error::copy_failed);
    GGML_ASSERT(enqueue_failure.cleanup_failed_lanes().is_ready());
    GGML_ASSERT(cold.diagnostics().current_transfer_refs == 0);

    {
        llm_expert_transfer_ring cleanup_failure(ring_config(1U << 20), { false, 0, false, true });
        GGML_ASSERT(cleanup_failure.initialize(source.bundle()).is_ready());
        GGML_ASSERT(cleanup_failure.reserve(cold, cold_zero, 0, 3, lane).is_ready());
        GGML_ASSERT(cleanup_failure.stage(lane, cold.bundle()).error == llm_expert_provider_error::copy_failed);
        GGML_ASSERT(cleanup_failure.cleanup_failed_lanes().error == llm_expert_provider_error::copy_failed);
        GGML_ASSERT(cold.diagnostics().current_transfer_refs == 1);
    }
    GGML_ASSERT(cold.diagnostics().current_transfer_refs == 0);

    llm_expert_transfer_ring allocation_failure(ring_config(1U << 20), { true });
    GGML_ASSERT(allocation_failure.initialize(source.bundle()).error ==
        llm_expert_provider_error::allocation_failed);
}

void test_budget_and_generation_rejection() {
    fixture source(4);
    llm_expert_transfer_ring discovery(ring_config(1U << 20));
    GGML_ASSERT(discovery.initialize(source.bundle()).is_ready());
    const uint64_t footprint = discovery.diagnostics().lane_footprint;
    llm_expert_transfer_ring insufficient(ring_config(footprint*2 - 1));
    GGML_ASSERT(insufficient.initialize(source.bundle()).error ==
        llm_expert_provider_error::unsupported_configuration);

    const uint64_t overflow_lanes = uint64_t(UINT32_MAX)/2 + 1;
    GGML_ASSERT(footprint <= UINT64_MAX/overflow_lanes);
    llm_expert_transfer_ring event_capacity_overflow(ring_config(footprint*overflow_lanes));
    GGML_ASSERT(event_capacity_overflow.initialize(source.bundle()).error ==
        llm_expert_provider_error::unsupported_configuration);
    const auto overflow = event_capacity_overflow.diagnostics();
    GGML_ASSERT(overflow.actual_bytes == 0 && overflow.effective_lanes == 0 &&
        overflow.pinned_or_registered_bytes == 0 && !overflow.dedicated_transfer_backend &&
        !overflow.event_capable && overflow.h2d_event_allocations == 0 &&
        overflow.compute_event_allocations == 0 && overflow.event_capacity == 0 &&
        overflow.live_events == 0);

    auto cold = make_cold(source);
    llm_cold_reference cold_zero;
    GGML_ASSERT(cold.find_or_admit({ 0, 0 }, source.bundle(), cold_zero).is_ready());
    llm_expert_transfer_ring exhausted(ring_config(footprint*2, 2, std::numeric_limits<uint64_t>::max()));
    GGML_ASSERT(exhausted.initialize(source.bundle()).is_ready());
    llm_transfer_lane_reference lane;
    GGML_ASSERT(exhausted.reserve(cold, cold_zero, 0, 1, lane).error ==
        llm_expert_provider_error::generation_exhausted);
}

void test_native_event_ordering_reuse_and_unload() {
    ggml_backend_load_all();
    auto * device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (device == nullptr) return;
    ggml_backend_ptr backend(ggml_backend_dev_init(device, nullptr));
    GGML_ASSERT(backend);
    fixture source(4);
    fixture hot(2, false, ggml_backend_dev_buffer_type(device));
    auto cold = make_cold(source);
    llm_cold_reference cold_zero, cold_one;
    GGML_ASSERT(cold.find_or_admit({ 0, 0 }, source.bundle(), cold_zero).is_ready());
    GGML_ASSERT(cold.find_or_admit({ 0, 1 }, source.bundle(), cold_one).is_ready());

    llm_expert_transfer_ring ring({ 1U << 20, 2, device, false, false, false, 0, 32 });
    GGML_ASSERT(ring.initialize(source.bundle()).is_ready());
    llm_transfer_lane_reference lane_zero, lane_one;
    GGML_ASSERT(ring.reserve(cold, cold_zero, 0, 1, lane_zero).is_ready());
    GGML_ASSERT(ring.reserve(cold, cold_one, 1, 1, lane_one).is_ready());
    GGML_ASSERT(ring.stage(lane_zero, cold.bundle()).is_ready());
    GGML_ASSERT(ring.stage(lane_one, cold.bundle()).is_ready());
    GGML_ASSERT(ring.transfer_wave(backend.get(), {
        { lane_zero, hot.bundle(), 0 }, { lane_one, hot.bundle(), 1 },
    }).is_ready());
    GGML_ASSERT(ring.wait_for_hot(backend.get(), 0, 2).error ==
        llm_expert_provider_error::stale_generation);
    GGML_ASSERT(ring.wait_for_hot(backend.get(), 0, 1, true).is_ready());
    GGML_ASSERT(ring.wait_for_hot(backend.get(), 1, 1, true).is_ready());
    GGML_ASSERT(cold.diagnostics().current_transfer_refs == 0);
    GGML_ASSERT(ring.retire_hot(0, 1).is_ready());
    GGML_ASSERT(ring.retire_hot(1, 1).is_ready());
    assert_slot_matches(source, hot, 0, 0);
    assert_slot_matches(source, hot, 1, 1);
    const auto diagnostics = ring.diagnostics();
    GGML_ASSERT(diagnostics.event_capable && diagnostics.event_capacity == diagnostics.effective_lanes*2);
    GGML_ASSERT(diagnostics.h2d_event_capacity == diagnostics.effective_lanes &&
        diagnostics.compute_event_capacity == diagnostics.effective_lanes &&
        diagnostics.h2d_event_allocations == diagnostics.h2d_event_capacity &&
        diagnostics.compute_event_allocations == diagnostics.compute_event_capacity);
    GGML_ASSERT(diagnostics.event_records == 2 && diagnostics.compute_waits == 2);
    GGML_ASSERT(diagnostics.event_synchronizations == 2 && diagnostics.live_h2d_events == 0 &&
        diagnostics.live_compute_events == 0 && diagnostics.live_events == 0);
    const auto intervals = ring.completed_intervals();
    GGML_ASSERT(intervals.size() == 2 && intervals[0].h2d_enqueue_us != 0 &&
        intervals[0].h2d_complete_us >= intervals[0].h2d_enqueue_us &&
        intervals[0].bytes == diagnostics.lane_payload_bytes);
    GGML_ASSERT(cold.diagnostics().current_transfer_refs == 0);

    llm_transfer_ring_config background_config = {
        diagnostics.lane_footprint, 1, device, false, false, false, 0, 32,
    };
    background_config.delay_background_stage_ms_for_testing = 250;
    llm_expert_transfer_ring background(background_config);
    GGML_ASSERT(background.initialize(source.bundle()).is_ready());
    llm_transfer_lane_reference background_lane;
    GGML_ASSERT(background.try_queue_background_transfer(
        cold, cold_zero, 0, 3, cold.bundle(), hot.bundle(), background_lane).is_ready());
    llm_expert_same_key_h2d_state background_state = llm_expert_same_key_h2d_state::none;
    uint64_t remaining_submitted = UINT64_MAX;
    GGML_ASSERT(background.poll_h2d(
        background_lane, background_state, remaining_submitted).is_ready());
    GGML_ASSERT(background_state == llm_expert_same_key_h2d_state::queued_or_staging &&
        remaining_submitted == diagnostics.lane_payload_bytes);
    GGML_ASSERT(background.wait_for_hot(backend.get(), 0, 3).is_ready());
    GGML_ASSERT(background.retire_hot(0, 3).is_ready());
    GGML_ASSERT(background.surrender().is_ready());
    GGML_ASSERT(cold.diagnostics().current_transfer_refs == 0);

    auto state_config = background_config;
    state_config.delay_background_stage_ms_for_testing = 0;
    state_config.delay_event_monitor_ms_for_testing = 100;
    llm_expert_transfer_ring state_ring(state_config);
    GGML_ASSERT(state_ring.initialize(source.bundle()).is_ready());
    llm_transfer_lane_reference state_lane;
    GGML_ASSERT(state_ring.try_queue_background_transfer(
        cold, cold_zero, 0, 5, cold.bundle(), hot.bundle(), state_lane).is_ready());
    const auto in_flight_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    do {
        GGML_ASSERT(state_ring.poll_h2d(state_lane, background_state, remaining_submitted).is_ready());
        if (background_state == llm_expert_same_key_h2d_state::h2d_in_flight) break;
        std::this_thread::yield();
    } while (std::chrono::steady_clock::now() < in_flight_deadline);
    GGML_ASSERT(background_state == llm_expert_same_key_h2d_state::h2d_in_flight &&
        remaining_submitted == diagnostics.lane_payload_bytes);
    GGML_ASSERT(state_ring.wait_background_h2d(state_lane).is_ready());
    GGML_ASSERT(state_ring.poll_h2d(state_lane, background_state, remaining_submitted).is_ready());
    GGML_ASSERT(background_state == llm_expert_same_key_h2d_state::h2d_complete_unpublished &&
        remaining_submitted == 0);
    GGML_ASSERT(state_ring.release_terminal_background(state_lane).is_ready());
    GGML_ASSERT(state_ring.surrender().is_ready());
    GGML_ASSERT(cold.diagnostics().current_transfer_refs == 0);

    llm_expert_transfer_ring failed_background(background_config, { false, 1, false, false });
    GGML_ASSERT(failed_background.initialize(source.bundle()).is_ready());
    llm_transfer_lane_reference failed_lane;
    GGML_ASSERT(failed_background.try_queue_background_transfer(
        cold, cold_zero, 0, 4, cold.bundle(), hot.bundle(), failed_lane).is_ready());
    llm_expert_provider_result failed_poll;
    const auto failure_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    do {
        failed_poll = failed_background.poll_h2d(failed_lane, background_state, remaining_submitted);
        if (!failed_poll.is_ready()) break;
        std::this_thread::yield();
    } while (std::chrono::steady_clock::now() < failure_deadline);
    GGML_ASSERT(failed_poll.error == llm_expert_provider_error::copy_failed);
    GGML_ASSERT(failed_background.release_terminal_background(failed_lane).is_ready());
    GGML_ASSERT(failed_background.surrender().is_ready());
    GGML_ASSERT(cold.diagnostics().current_transfer_refs == 0);

    {
        ggml_backend_ptr gate_backend(ggml_backend_dev_init(device, nullptr));
        GGML_ASSERT(gate_backend);
        ggml_init_params compute_params = { ggml_tensor_overhead()*4 + ggml_graph_overhead(), nullptr, true };
        ggml_context_ptr compute_ctx(ggml_init(compute_params));
        GGML_ASSERT(compute_ctx);
        ggml_tensor * a = ggml_new_tensor_2d(compute_ctx.get(), GGML_TYPE_F32, 1536, 1536);
        ggml_tensor * b = ggml_new_tensor_2d(compute_ctx.get(), GGML_TYPE_F32, 1536, 1536);
        ggml_tensor * c = ggml_mul_mat(compute_ctx.get(), a, b);
        ggml_backend_buffer_ptr compute_buffer(
            ggml_backend_alloc_ctx_tensors_from_buft(compute_ctx.get(), ggml_backend_dev_buffer_type(device)));
        GGML_ASSERT(compute_buffer);
        std::vector<float> zeros(size_t(1536)*1536, 0.0f);
        ggml_backend_tensor_set(a, zeros.data(), 0, zeros.size()*sizeof(float));
        ggml_backend_tensor_set(b, zeros.data(), 0, zeros.size()*sizeof(float));
        ggml_cgraph * graph = ggml_new_graph_custom(compute_ctx.get(), 8, false);
        ggml_build_forward_expand(graph, c);
        ggml_backend_event_t gate_event = ggml_backend_event_new(device);
        GGML_ASSERT(gate_event != nullptr);
        GGML_ASSERT(ggml_backend_graph_compute_async(gate_backend.get(), graph) == GGML_STATUS_SUCCESS);
        ggml_backend_event_record(gate_event, gate_backend.get());

        llm_transfer_ring_config cancel_config = { diagnostics.lane_footprint, 1, device };
        cancel_config.trace_capacity = 32;
        cancel_config.h2d_gate_event_for_testing = gate_event;
        cancel_config.delay_event_monitor_ms_for_testing = 100;
        llm_expert_transfer_ring cancelling(cancel_config);
        GGML_ASSERT(cancelling.initialize(source.bundle()).is_ready());
        llm_expert_scheduler scheduler({ 1, 4, 2, 1, 0 });
        const llm_expert_key key = { 0, 0 };
        const auto admitted = scheduler.enqueue(key, llm_expert_priority::demand_current_layer,
            llm_expert_readiness::device_ready);
        GGML_ASSERT(admitted.disposition == llm_expert_schedule_disposition::admitted);
        llm_expert_request_snapshot selected;
        GGML_ASSERT(scheduler.take_next(selected).disposition == llm_expert_schedule_disposition::admitted);
        GGML_ASSERT(scheduler.transition(admitted.handle, llm_expert_request_state::submitting,
            llm_expert_request_state::io_in_flight) == llm_expert_schedule_disposition::admitted);
        GGML_ASSERT(scheduler.transition(admitted.handle, llm_expert_request_state::io_in_flight,
            llm_expert_request_state::host_ready) == llm_expert_schedule_disposition::admitted);
        GGML_ASSERT(scheduler.transition(admitted.handle, llm_expert_request_state::host_ready,
            llm_expert_request_state::h2d_in_flight) == llm_expert_schedule_disposition::admitted);
        const llm_expert_flight_id cancelled_flight = {
            7, admitted.handle.slot, admitted.handle.generation, key,
        };
        llm_transfer_lane_reference cancelled_lane;
        GGML_ASSERT(cancelling.reserve(cold, cold_zero, 0, 7, cancelled_lane, cancelled_flight).is_ready());
        GGML_ASSERT(cancelling.stage(cancelled_lane, cold.bundle()).is_ready());
        GGML_ASSERT(cancelling.transfer_wave(
            backend.get(), { { cancelled_lane, hot.bundle(), 0 } }).is_ready());
        GGML_ASSERT(cancelling.diagnostics().live_h2d_events == 1);
        GGML_ASSERT(scheduler.begin_demand_cancellation(
            admitted.handle, llm_expert_request_state::h2d_in_flight) ==
            llm_expert_schedule_disposition::admitted);
        GGML_ASSERT(scheduler.transition(admitted.handle, llm_expert_request_state::cancelling,
            llm_expert_request_state::draining) == llm_expert_schedule_disposition::admitted);

        std::atomic<bool> cancel_done = false;
        std::atomic<bool> reuse_done = false;
        llm_transfer_lane_reference blocked_reuse;
        std::thread cancel_thread([&] {
            GGML_ASSERT(cancelling.cancel_after_h2d(cancelled_lane).is_ready());
            cancel_done.store(true, std::memory_order_release);
        });
        std::thread reuse_thread([&] {
            GGML_ASSERT(cancelling.reserve(cold, cold_one, 0, 8, blocked_reuse).is_ready());
            reuse_done.store(true, std::memory_order_release);
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        GGML_ASSERT(!cancel_done.load(std::memory_order_acquire));
        GGML_ASSERT(!reuse_done.load(std::memory_order_acquire));
        cancel_thread.join();
        reuse_thread.join();
        GGML_ASSERT(blocked_reuse.generation != cancelled_lane.generation);
        GGML_ASSERT(cancelling.cleanup_failed_lanes().is_ready());
        GGML_ASSERT(scheduler.finish(admitted.handle, llm_expert_request_state::cancelled) ==
            llm_expert_schedule_disposition::admitted);
        GGML_ASSERT(scheduler.release_terminal(admitted.handle) == llm_expert_schedule_disposition::admitted);
        GGML_ASSERT(scheduler.diagnostics().active_requests == 0);

        const auto retry = scheduler.enqueue(key, llm_expert_priority::demand_current_layer,
            llm_expert_readiness::device_ready);
        GGML_ASSERT(retry.disposition == llm_expert_schedule_disposition::admitted &&
            retry.handle.generation != admitted.handle.generation);
        GGML_ASSERT(scheduler.take_next(selected).disposition == llm_expert_schedule_disposition::admitted);
        GGML_ASSERT(scheduler.transition(retry.handle, llm_expert_request_state::submitting,
            llm_expert_request_state::io_in_flight) == llm_expert_schedule_disposition::admitted);
        GGML_ASSERT(scheduler.transition(retry.handle, llm_expert_request_state::io_in_flight,
            llm_expert_request_state::host_ready) == llm_expert_schedule_disposition::admitted);
        GGML_ASSERT(scheduler.transition(retry.handle, llm_expert_request_state::host_ready,
            llm_expert_request_state::h2d_in_flight) == llm_expert_schedule_disposition::admitted);
        const llm_expert_flight_id retry_flight = { 7, retry.handle.slot, retry.handle.generation, key };
        llm_transfer_lane_reference retry_lane;
        GGML_ASSERT(cancelling.reserve(cold, cold_zero, 0, 9, retry_lane, retry_flight).is_ready());
        GGML_ASSERT(cancelling.stage(retry_lane, cold.bundle()).is_ready());
        GGML_ASSERT(cancelling.transfer_wave(backend.get(), { { retry_lane, hot.bundle(), 0 } }).is_ready());
        GGML_ASSERT(cancelling.wait_for_hot(backend.get(), 0, 9).is_ready());
        GGML_ASSERT(scheduler.transition(retry.handle, llm_expert_request_state::h2d_in_flight,
            llm_expert_request_state::device_ready) == llm_expert_schedule_disposition::admitted);
        GGML_ASSERT(scheduler.finish(retry.handle, llm_expert_request_state::complete) ==
            llm_expert_schedule_disposition::admitted);
        GGML_ASSERT(scheduler.release_terminal(retry.handle) == llm_expert_schedule_disposition::admitted);
        GGML_ASSERT(cancelling.retire_hot(0, 9).is_ready());
        assert_slot_matches(source, hot, 0, 0);
        auto cancellation_diagnostics = cancelling.diagnostics();
        GGML_ASSERT(cancellation_diagnostics.h2d_event_cancellations == 1);
        GGML_ASSERT(cancellation_diagnostics.live_h2d_events == 0 &&
            cancellation_diagnostics.live_compute_events == 0 && cancellation_diagnostics.live_events == 0);
        GGML_ASSERT(cold.diagnostics().current_transfer_refs == 0);
        const auto cancellation_intervals = cancelling.completed_intervals();
        GGML_ASSERT(cancellation_intervals.size() == 2 && cancellation_intervals[0].cancelled &&
            cancellation_intervals[0].flight.request_generation == admitted.handle.generation &&
            !cancellation_intervals[1].cancelled &&
            cancellation_intervals[1].flight.request_generation == retry.handle.generation);
        GGML_ASSERT(cancelling.surrender().is_ready());
        cancellation_diagnostics = cancelling.diagnostics();
        GGML_ASSERT(cancellation_diagnostics.h2d_event_frees == 1 &&
            cancellation_diagnostics.compute_event_frees == 1 &&
            cancellation_diagnostics.h2d_event_allocations == cancellation_diagnostics.h2d_event_frees &&
            cancellation_diagnostics.compute_event_allocations == cancellation_diagnostics.compute_event_frees &&
            cancellation_diagnostics.event_capacity == 0);
        GGML_ASSERT(cancelling.validate_invariants().is_ready());
        ggml_backend_event_free(gate_event);
    }

    {
        llm_expert_transfer_ring unloading({ 1U << 20, 2, device, false, false, false, 0, 32 });
        GGML_ASSERT(unloading.initialize(source.bundle()).is_ready());
        GGML_ASSERT(unloading.reserve(cold, cold_zero, 0, 2, lane_zero).is_ready());
        GGML_ASSERT(unloading.stage(lane_zero, cold.bundle()).is_ready());
        GGML_ASSERT(unloading.transfer_wave(backend.get(), { { lane_zero, hot.bundle(), 0 } }).is_ready());
        GGML_ASSERT(cold.diagnostics().current_transfer_refs == 1);
    }
    GGML_ASSERT(cold.diagnostics().current_transfer_refs == 0);
}

void test_native_async_cold_fill_is_best_effort_and_drained() {
    ggml_backend_load_all();
    auto * device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (device == nullptr) return;
    ggml_backend_ptr backend(ggml_backend_dev_init(device, nullptr));
    GGML_ASSERT(backend);
    fixture source(4);
    fixture hot(2, false, ggml_backend_dev_buffer_type(device));

    auto successful_cold = make_cold(source);
    llm_transfer_ring_config success_config = { 1U << 20, 3, device, false, false, false, 0, 32 };
    success_config.delay_cold_fill_ms_for_testing = 150;
    llm_expert_transfer_ring successful(success_config);
    GGML_ASSERT(successful.initialize(source.bundle()).is_ready());
    llm_transfer_lane_reference success_lane;
    const llm_expert_flight_id success_flight = { 9, 1, 1, { 0, 2 }, 0, 0 };
    GGML_ASSERT(successful.reserve_direct_storage(
        0, 0, 1, success_lane, success_flight).is_ready());
    const uint64_t success_bytes = populate_direct_lane(successful, success_lane, source, 2);
    GGML_ASSERT(successful.complete_direct_storage(success_lane, success_bytes).is_ready());
    GGML_ASSERT(successful.transfer_wave(
        backend.get(), { { success_lane, hot.bundle(), 0 } }).is_ready());
    GGML_ASSERT(successful.try_queue_cold_fill(successful_cold, success_lane).is_ready());
    const auto reservation_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    do {
        if (successful_cold.diagnostics().reservations == 1) break;
        std::this_thread::yield();
    } while (std::chrono::steady_clock::now() < reservation_deadline);
    GGML_ASSERT(successful_cold.diagnostics().reservations == 1);
    GGML_ASSERT(successful_cold.policy_request_end(true, false, true).is_ready());
    GGML_ASSERT(successful.wait_for_hot(backend.get(), 0, 1).is_ready());
    assert_slot_matches(source, hot, 2, 0);
    auto success_diagnostics = successful.diagnostics();
    GGML_ASSERT(success_diagnostics.cold_fill_active == 1 &&
        success_diagnostics.cold_fill_completed == 0);
    const auto success_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    do {
        success_diagnostics = successful.diagnostics();
        if (success_diagnostics.cold_fill_completed == 1) break;
        std::this_thread::yield();
    } while (std::chrono::steady_clock::now() < success_deadline);
    GGML_ASSERT(success_diagnostics.cold_fill_attempts == 1 &&
        success_diagnostics.cold_fill_queued == 1 && success_diagnostics.cold_fill_dropped == 0 &&
        success_diagnostics.cold_fill_completed == 1 && success_diagnostics.cold_fill_failed == 0 &&
        success_diagnostics.cold_fill_bytes == success_bytes && success_diagnostics.cold_fill_active == 0 &&
        success_diagnostics.cold_fill_peak_active == 1);
    llm_cold_reference filled_reference;
    llm_cold_demand_lookup filled_lookup = llm_cold_demand_lookup::missing;
    GGML_ASSERT(successful_cold.lookup_demand(
        { 0, 2 }, filled_reference, filled_lookup).is_ready());
    GGML_ASSERT(filled_lookup == llm_cold_demand_lookup::ready &&
        successful_cold.ready(filled_reference));
    assert_bundle_slot_matches(source, successful_cold.bundle(), 2, filled_reference.slot);
    GGML_ASSERT(successful.validate_invariants().is_ready());
    GGML_ASSERT(successful.surrender().is_ready());
    GGML_ASSERT(successful_cold.surrender().is_ready());

    auto failed_cold = make_cold(source);
    auto failure_config = success_config;
    failure_config.delay_cold_fill_ms_for_testing = 0;
    llm_transfer_ring_faults failure_faults;
    failure_faults.fail_cold_fill = true;
    llm_expert_transfer_ring failed(failure_config, failure_faults);
    GGML_ASSERT(failed.initialize(source.bundle()).is_ready());
    llm_transfer_lane_reference failed_lane;
    const llm_expert_flight_id failed_flight = { 9, 2, 1, { 0, 3 }, 0, 0 };
    GGML_ASSERT(failed.reserve_direct_storage(0, 0, 2, failed_lane, failed_flight).is_ready());
    const uint64_t failed_bytes = populate_direct_lane(failed, failed_lane, source, 3);
    GGML_ASSERT(failed.complete_direct_storage(failed_lane, failed_bytes).is_ready());
    GGML_ASSERT(failed.transfer_wave(backend.get(), { { failed_lane, hot.bundle(), 0 } }).is_ready());
    GGML_ASSERT(failed.try_queue_cold_fill(failed_cold, failed_lane).is_ready());
    GGML_ASSERT(failed.wait_for_hot(backend.get(), 0, 2).is_ready());
    assert_slot_matches(source, hot, 3, 0);
    const auto failure_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    llm_transfer_ring_diagnostics failure_diagnostics;
    do {
        failure_diagnostics = failed.diagnostics();
        if (failure_diagnostics.cold_fill_failed == 1) break;
        std::this_thread::yield();
    } while (std::chrono::steady_clock::now() < failure_deadline);
    GGML_ASSERT(failure_diagnostics.cold_fill_failed == 1 &&
        failure_diagnostics.cold_fill_completed == 0 && failure_diagnostics.cold_fill_active == 0);
    GGML_ASSERT(failed_cold.diagnostics().failed_copies == 1);
    GGML_ASSERT(failed_cold.cleanup_failed_slots().is_ready());
    GGML_ASSERT(failed.validate_invariants().is_ready());
    GGML_ASSERT(failed.surrender().is_ready());
    GGML_ASSERT(failed_cold.surrender().is_ready());

    auto cancelled_cold = make_cold(source);
    auto cancel_config = success_config;
    cancel_config.delay_cold_fill_ms_for_testing = 100;
    llm_expert_transfer_ring cancelled(cancel_config);
    GGML_ASSERT(cancelled.initialize(source.bundle()).is_ready());
    llm_transfer_lane_reference cancelled_lane;
    const llm_expert_flight_id cancelled_flight = { 9, 3, 1, { 0, 1 }, 0, 0 };
    GGML_ASSERT(cancelled.reserve_direct_storage(
        0, 0, 3, cancelled_lane, cancelled_flight).is_ready());
    const uint64_t cancelled_bytes = populate_direct_lane(cancelled, cancelled_lane, source, 1);
    GGML_ASSERT(cancelled.complete_direct_storage(cancelled_lane, cancelled_bytes).is_ready());
    GGML_ASSERT(cancelled.transfer_wave(
        backend.get(), { { cancelled_lane, hot.bundle(), 0 } }).is_ready());
    GGML_ASSERT(cancelled.try_queue_cold_fill(cancelled_cold, cancelled_lane).is_ready());
    GGML_ASSERT(cancelled.cancel_after_h2d(cancelled_lane).is_ready());
    const auto cancel_diagnostics = cancelled.diagnostics();
    GGML_ASSERT(cancel_diagnostics.cold_fill_active == 0 &&
        cancel_diagnostics.live_events == 0 && cancel_diagnostics.cold_fill_queued == 1 &&
        cancel_diagnostics.cold_fill_completed + cancel_diagnostics.cold_fill_dropped == 1);
    GGML_ASSERT(cancelled.validate_invariants().is_ready());
    GGML_ASSERT(cancelled.surrender().is_ready());
    GGML_ASSERT(cancelled_cold.surrender().is_ready());
}

} // namespace

int main() {
    test_budget_fallback_and_wave();
    test_direct_storage_lane_bypasses_cold_staging();
    test_three_layout_classes_reuse_universal_lanes();
    test_failures_cleanup_and_busy_surrender();
    test_budget_and_generation_rejection();
    test_native_event_ordering_reuse_and_unload();
    test_native_async_cold_fill_is_best_effort_and_drained();
    std::cout << "expert transfer ring tests passed\n";
    return 0;
}
