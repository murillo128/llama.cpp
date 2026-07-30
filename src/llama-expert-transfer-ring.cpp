#include "llama-expert-transfer-ring.h"

#include "ggml-cpp.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <stdexcept>

namespace {

int expert_axis(const ggml_tensor * tensor, int32_t n_expert, bool weight) {
    if (tensor == nullptr) return -1;
    if (weight) return ggml_n_dims(tensor) >= 3 && tensor->ne[2] == n_expert ? 2 : -1;
    int result = -1;
    for (int axis = ggml_n_dims(tensor) - 1; axis >= 0; --axis) {
        if (tensor->ne[axis] == n_expert) {
            if (result >= 0) return -1;
            result = axis;
        }
    }
    return result;
}

bool checked_add(uint64_t lhs, uint64_t rhs, uint64_t & result) {
    if (lhs > std::numeric_limits<uint64_t>::max() - rhs) return false;
    result = lhs + rhs;
    return true;
}

bool checked_mul(uint64_t lhs, uint64_t rhs, uint64_t & result) {
    if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max()/lhs) return false;
    result = lhs*rhs;
    return true;
}

bool round_up(uint64_t value, uint64_t alignment, uint64_t & result) {
    if (alignment == 0) return false;
    const uint64_t remainder = value % alignment;
    return remainder == 0 ? (result = value, true) : checked_add(value, alignment - remainder, result);
}

bool device_is_cuda(ggml_backend_dev_t device) {
    if (device == nullptr || ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_GPU) return false;
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(device);
    return reg != nullptr && std::strcmp(ggml_backend_reg_name(reg), "CUDA") == 0;
}

struct span_layout {
    uint8_t projection = 0;
    uint8_t member = 0;
    size_t offset = 0;
    size_t bytes = 0;
};

ggml_tensor * member(llm_expert_bundle_descriptor & bundle, uint8_t projection, uint8_t member_index) {
    llm_expert_projection_descriptor * value = nullptr;
    if (projection == 0) value = &bundle.up;
    else if (projection == 1) value = &bundle.gate;
    else if (projection == 2) value = &bundle.gate_up;
    else if (projection == 3) value = &bundle.down;
    if (value == nullptr) return nullptr;
    return member_index == 0 ? value->weight : member_index == 1 ? value->bias : value->scale;
}

const ggml_tensor * member(const llm_expert_bundle_descriptor & bundle, uint8_t projection, uint8_t member_index) {
    auto & mutable_bundle = const_cast<llm_expert_bundle_descriptor &>(bundle);
    return member(mutable_bundle, projection, member_index);
}

bool derive_layout(
        const llm_expert_bundle_descriptor & prototype,
        size_t alignment,
        std::vector<span_layout> & spans,
        uint64_t & payload,
        uint64_t & footprint) {
    spans.clear();
    payload = 0;
    footprint = 0;
    for (uint8_t projection = 0; projection < 4; ++projection) {
        for (uint8_t member_index = 0; member_index < 3; ++member_index) {
            const auto * tensor = member(prototype, projection, member_index);
            if (tensor == nullptr) continue;
            const int axis = expert_axis(tensor, prototype.n_expert, member_index == 0);
            uint64_t aligned = 0;
            if (axis < 0 || !round_up(footprint, alignment, aligned) ||
                tensor->nb[axis] > SIZE_MAX || aligned > SIZE_MAX ||
                !checked_add(aligned, tensor->nb[axis], footprint) ||
                !checked_add(payload, tensor->nb[axis], payload)) {
                return false;
            }
            spans.push_back({ projection, member_index, size_t(aligned), tensor->nb[axis] });
        }
    }
    return !spans.empty() && round_up(footprint, alignment, footprint);
}

} // namespace

struct llm_expert_transfer_ring::impl {
    struct lane_state : llm_transfer_ring_diagnostics::lane {
        llm_cold_expert_cache * cold_cache = nullptr;
    };

    impl(llm_transfer_ring_config config, llm_transfer_ring_faults faults) : config(config), faults(faults) {
        if (config.byte_budget == 0 || config.minimum_lanes == 0 || config.target_device == nullptr ||
            (!config.allow_non_cuda_target_for_testing && !device_is_cuda(config.target_device))) {
            throw std::invalid_argument("invalid transfer-ring budget or target");
        }
    }

    bool valid_lane(llm_transfer_lane_reference reference) const {
        return reference.lane < lanes.size() && lanes[reference.lane].generation == reference.generation &&
            (lanes[reference.lane].state == llm_transfer_lane_state::staging ||
             lanes[reference.lane].state == llm_transfer_lane_state::in_flight);
    }

    void release_lane(lane_state & lane) {
        if (lane.cold_cache != nullptr) {
            const auto released = lane.cold_cache->release(lane.cold, llm_cold_reference_kind::transfer);
            if (!released.is_ready()) counters.failed_cleanups++;
        }
        lane.cold_cache = nullptr;
        lane.cold = {};
        lane.hot_slot = 0;
        lane.hot_generation = 0;
        lane.state = llm_transfer_lane_state::free;
    }

    llm_transfer_ring_config config;
    llm_transfer_ring_faults faults;
    mutable std::mutex mutex;
    ggml_backend_buffer_ptr arena;
    uint8_t * base = nullptr;
    std::vector<span_layout> spans;
    std::vector<lane_state> lanes;
    llm_transfer_ring_diagnostics counters;
};

llm_expert_transfer_ring::llm_expert_transfer_ring(
        llm_transfer_ring_config config, llm_transfer_ring_faults faults) :
    pimpl(std::make_unique<impl>(config, faults)) {}

llm_expert_transfer_ring::~llm_expert_transfer_ring() {
    if (!pimpl) return;
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    for (auto & lane : pimpl->lanes) pimpl->release_lane(lane);
}

llm_expert_provider_result llm_expert_transfer_ring::initialize(
        const llm_expert_bundle_descriptor & prototype) noexcept {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    if (pimpl->arena) return llm_expert_provider_result::success();
    if (!prototype.validate().is_ready() || pimpl->faults.fail_allocation) {
        return llm_expert_provider_result::failure(pimpl->faults.fail_allocation ?
            llm_expert_provider_error::allocation_failed : llm_expert_provider_error::invalid_descriptor);
    }
    try {
        ggml_backend_buffer_type_t native_buft = ggml_backend_dev_host_buffer_type(pimpl->config.target_device);
        if (native_buft == nullptr) native_buft = ggml_backend_cpu_buffer_type();
        const size_t native_alignment = ggml_backend_buft_get_alignment(native_buft);
        uint64_t payload = 0, footprint = 0;
        if (!derive_layout(prototype, native_alignment, pimpl->spans, payload, footprint) || footprint == 0) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::unsupported_configuration);
        }
        uint64_t lane_count64 = pimpl->config.byte_budget/footprint;
        if (lane_count64 < pimpl->config.minimum_lanes || lane_count64 > UINT32_MAX) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::unsupported_configuration);
        }
        uint32_t lane_count = uint32_t(lane_count64);
        uint64_t requested = 0;
        if (!checked_mul(footprint, lane_count, requested) || requested > SIZE_MAX) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::unsupported_configuration);
        }

        ggml_backend_buffer_ptr candidate;
        bool pinned = false;
        std::string fallback_reason;
        if (!pimpl->config.force_pageable_fallback_for_testing && native_buft != ggml_backend_cpu_buffer_type()) {
            candidate.reset(ggml_backend_buft_alloc_buffer(native_buft, size_t(requested)));
            pinned = candidate && ggml_backend_buffer_get_type(candidate.get()) == native_buft;
            if (!pinned) fallback_reason = candidate ? "native-host-returned-pageable" : "native-host-allocation-failed";
        } else {
            fallback_reason = pimpl->config.force_pageable_fallback_for_testing ?
                "forced-for-testing" : "native-host-buffer-unavailable";
        }
        if (!pinned) {
            candidate.reset(ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size_t(requested)));
            if (!candidate) return llm_expert_provider_result::failure(llm_expert_provider_error::allocation_failed);
            std::fprintf(stderr, "warning: expert transfer ring using pageable synchronous fallback (%s)\n",
                fallback_reason.c_str());
        }
        const uint64_t actual = ggml_backend_buffer_get_size(candidate.get());
        if (actual > pimpl->config.byte_budget) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::allocation_failed);
        }
        pimpl->lanes.assign(lane_count, {});
        for (auto & lane : pimpl->lanes) lane.generation = pimpl->config.initial_lane_generation_for_testing;
        pimpl->base = static_cast<uint8_t *>(ggml_backend_buffer_get_base(candidate.get()));
        if (pimpl->base == nullptr) return llm_expert_provider_result::failure(llm_expert_provider_error::allocation_failed);
        pimpl->counters.requested_bytes = pimpl->config.byte_budget;
        pimpl->counters.actual_bytes = actual;
        pimpl->counters.unused_budget_bytes = pimpl->config.byte_budget - actual;
        pimpl->counters.lane_payload_bytes = payload;
        pimpl->counters.lane_footprint = footprint;
        pimpl->counters.alignment = native_alignment;
        pimpl->counters.effective_lanes = lane_count;
        pimpl->counters.acquisition_method = pinned ? "device-native-host" : "pageable-cpu";
        pimpl->counters.fallback_reason = pinned ? "" : fallback_reason;
        pimpl->counters.pageable_fallback = !pinned;
        pimpl->counters.pinned_or_registered_bytes = pinned ? actual : 0;
        pimpl->counters.fallback_count = pinned ? 0 : 1;
        pimpl->arena = std::move(candidate);
        return llm_expert_provider_result::success();
    } catch (const std::bad_alloc &) {
        pimpl->spans.clear(); pimpl->lanes.clear(); pimpl->base = nullptr;
        return llm_expert_provider_result::failure(llm_expert_provider_error::allocation_failed);
    } catch (...) {
        pimpl->spans.clear(); pimpl->lanes.clear(); pimpl->base = nullptr;
        return llm_expert_provider_result::failure(llm_expert_provider_error::initialization_failed);
    }
}

llm_expert_provider_result llm_expert_transfer_ring::reserve(
        llm_cold_expert_cache & cold_cache,
        llm_cold_reference cold,
        uint32_t hot_slot,
        uint64_t hot_generation,
        llm_transfer_lane_reference & reference) noexcept {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    if (!pimpl->arena) return llm_expert_provider_result::failure(llm_expert_provider_error::initialization_failed);
    uint32_t index = 0;
    while (index < pimpl->lanes.size() && pimpl->lanes[index].state != llm_transfer_lane_state::free) index++;
    if (index == pimpl->lanes.size()) return llm_expert_provider_result::failure(llm_expert_provider_error::busy);
    auto & lane = pimpl->lanes[index];
    if (lane.generation == std::numeric_limits<uint64_t>::max()) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::generation_exhausted);
    }
    const auto acquired = cold_cache.acquire(cold, llm_cold_reference_kind::transfer);
    if (!acquired.is_ready()) return acquired;
    lane.generation++;
    lane.state = llm_transfer_lane_state::staging;
    lane.cold = cold;
    lane.hot_slot = hot_slot;
    lane.hot_generation = hot_generation;
    lane.cold_cache = &cold_cache;
    reference = { index, lane.generation };
    pimpl->counters.lane_reservations++;
    return llm_expert_provider_result::success();
}

llm_expert_provider_result llm_expert_transfer_ring::stage(
        llm_transfer_lane_reference reference,
        const llm_expert_bundle_descriptor & cold_bundle) noexcept {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    if (!pimpl->valid_lane(reference) || pimpl->lanes[reference.lane].state != llm_transfer_lane_state::staging) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::stale_generation);
    }
    auto & lane = pimpl->lanes[reference.lane];
    size_t copied = 0;
    const int64_t start = ggml_time_us();
    for (const auto & span : pimpl->spans) {
        const auto * source = member(cold_bundle, span.projection, span.member);
        const int axis = expert_axis(source, cold_bundle.n_expert, span.member == 0);
        if (copied >= pimpl->faults.fail_stage_after_spans || source == nullptr || axis < 0 ||
            source->nb[axis] != span.bytes || lane.cold.slot >= uint32_t(source->ne[axis]) || source->data == nullptr) {
            lane.state = llm_transfer_lane_state::failed;
            return llm_expert_provider_result::failure(llm_expert_provider_error::copy_failed);
        }
        const auto * source_data = static_cast<const uint8_t *>(source->data) + size_t(lane.cold.slot)*span.bytes;
        std::memcpy(pimpl->base + size_t(reference.lane)*pimpl->counters.lane_footprint + span.offset,
            source_data, span.bytes);
        pimpl->counters.stage_bytes += span.bytes;
        copied++;
    }
    pimpl->counters.stage_time_us += ggml_time_us() - start;
    return llm_expert_provider_result::success();
}

llm_expert_provider_result llm_expert_transfer_ring::transfer_wave(
        ggml_backend_t backend,
        const std::vector<llm_transfer_binding> & bindings) noexcept {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    if (bindings.empty() || !pimpl->arena || (pimpl->counters.pinned_or_registered_bytes > 0 && backend == nullptr)) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::invalid_binding);
    }
    const auto fail_bindings = [&]() {
        for (const auto & binding : bindings) {
            if (binding.lane.lane < pimpl->lanes.size() &&
                pimpl->lanes[binding.lane.lane].generation == binding.lane.generation &&
                pimpl->lanes[binding.lane.lane].state == llm_transfer_lane_state::staging) {
                pimpl->lanes[binding.lane.lane].state = llm_transfer_lane_state::failed;
            }
        }
    };
    for (const auto & binding : bindings) {
        if (!pimpl->valid_lane(binding.lane) ||
            pimpl->lanes[binding.lane.lane].state != llm_transfer_lane_state::staging ||
            pimpl->lanes[binding.lane.lane].hot_slot != binding.hot_slot) {
            fail_bindings();
            return llm_expert_provider_result::failure(llm_expert_provider_error::stale_generation);
        }
        for (const auto & other : bindings) {
            if (&other != &binding && other.lane.lane == binding.lane.lane) {
                fail_bindings();
                return llm_expert_provider_result::failure(llm_expert_provider_error::invalid_binding);
            }
        }
        for (const auto & span : pimpl->spans) {
            const auto * target = member(binding.destination, span.projection, span.member);
            const int axis = expert_axis(target, binding.destination.n_expert, span.member == 0);
            if (target == nullptr || axis < 0 || target->nb[axis] != span.bytes ||
                binding.hot_slot >= uint32_t(target->ne[axis]) || target->data == nullptr) {
                fail_bindings();
                return llm_expert_provider_result::failure(llm_expert_provider_error::invalid_descriptor);
            }
        }
    }
    if (pimpl->faults.fail_pre_enqueue) {
        fail_bindings();
        return llm_expert_provider_result::failure(llm_expert_provider_error::copy_failed);
    }
    const bool async = pimpl->counters.pinned_or_registered_bytes > 0;
    const int64_t start = ggml_time_us();
    uint64_t bytes = 0;
    for (const auto & binding : bindings) {
        auto & lane = pimpl->lanes[binding.lane.lane];
        if (async) lane.state = llm_transfer_lane_state::in_flight;
        for (const auto & span : pimpl->spans) {
            auto destination = binding.destination;
            auto * target = member(destination, span.projection, span.member);
            const auto * data = pimpl->base + size_t(binding.lane.lane)*pimpl->counters.lane_footprint + span.offset;
            const size_t offset = size_t(binding.hot_slot)*span.bytes;
            if (async) {
                ggml_backend_tensor_set_async(backend, target, data, offset, span.bytes);
                pimpl->counters.async_enqueues++;
            } else {
                ggml_backend_tensor_set(target, data, offset, span.bytes);
                pimpl->counters.synchronous_copies++;
            }
            bytes += span.bytes;
        }
    }
    if (async) {
        pimpl->counters.peak_in_flight_lanes = std::max<uint64_t>(
            pimpl->counters.peak_in_flight_lanes, bindings.size());
        ggml_backend_synchronize(backend);
        pimpl->counters.wave_synchronizations++;
    }
    pimpl->counters.waves++;
    pimpl->counters.h2d_bytes += bytes;
    pimpl->counters.h2d_time_us += ggml_time_us() - start;
    for (const auto & binding : bindings) pimpl->release_lane(pimpl->lanes[binding.lane.lane]);
    return llm_expert_provider_result::success();
}

llm_expert_provider_result llm_expert_transfer_ring::cleanup_failed_lanes() noexcept {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    if (pimpl->faults.fail_cleanup) {
        pimpl->counters.failed_cleanups++;
        return llm_expert_provider_result::failure(llm_expert_provider_error::copy_failed);
    }
    for (auto & lane : pimpl->lanes) {
        if (lane.state == llm_transfer_lane_state::failed) pimpl->release_lane(lane);
    }
    return llm_expert_provider_result::success();
}

llm_expert_provider_result llm_expert_transfer_ring::surrender() noexcept {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    for (const auto & lane : pimpl->lanes) {
        if (lane.state != llm_transfer_lane_state::free) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::busy);
        }
    }
    pimpl->arena.reset(); pimpl->base = nullptr; pimpl->spans.clear(); pimpl->lanes.clear();
    pimpl->counters.actual_bytes = 0;
    pimpl->counters.pinned_or_registered_bytes = 0;
    pimpl->counters.effective_lanes = 0;
    pimpl->counters.unused_budget_bytes = pimpl->config.byte_budget;
    return llm_expert_provider_result::success();
}

llm_expert_provider_result llm_expert_transfer_ring::validate_invariants() noexcept {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    uint64_t inflight = 0;
    for (const auto & lane : pimpl->lanes) {
        if (lane.state == llm_transfer_lane_state::free && lane.cold_cache != nullptr) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
        }
        if (lane.state != llm_transfer_lane_state::free && lane.cold_cache == nullptr) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
        }
        inflight += lane.state == llm_transfer_lane_state::in_flight;
    }
    if (pimpl->counters.pageable_fallback &&
        (pimpl->counters.pinned_or_registered_bytes != 0 || pimpl->counters.async_enqueues != 0 || inflight != 0)) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
    }
    return llm_expert_provider_result::success();
}

llm_transfer_ring_diagnostics llm_expert_transfer_ring::diagnostics() const {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    auto result = pimpl->counters;
    result.lanes.assign(pimpl->lanes.begin(), pimpl->lanes.end());
    return result;
}
