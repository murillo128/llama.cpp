#include "llama-expert-transfer-ring.h"

#include "ggml-cpp.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <stdexcept>
#include <thread>

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

bool same_flight(const llm_expert_flight_id & lhs, const llm_expert_flight_id & rhs) {
    return lhs.transport_epoch == rhs.transport_epoch && lhs.request_slot == rhs.request_slot &&
        lhs.request_generation == rhs.request_generation && lhs.key.layer == rhs.key.layer &&
        lhs.key.expert == rhs.key.expert;
}

uint64_t interval_union_intersection_us(
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
        llm_expert_flight_id flight;
        llm_cold_expert_cache * cold_cache = nullptr;
        ggml_backend_event_t event = nullptr;
        ggml_backend_event_t compute_event = nullptr;
        bool wait_enqueued = false;
        bool event_complete = false;
        bool monitoring = false;
        bool hold_after_h2d = false;
        bool compute_pending = false;
        bool compute_complete = false;
        bool compute_monitoring = false;
        bool cancelled = false;
        bool background_queued = false;
        bool background_running = false;
        llm_expert_bundle_descriptor background_cold;
        llm_expert_bundle_descriptor background_destination;
        uint64_t failed_generation = 0;
        uint32_t failed_hot_slot = UINT32_MAX;
        uint64_t failed_hot_generation = 0;
        uint64_t h2d_enqueue_us = 0;
        uint64_t h2d_complete_us = 0;
        uint64_t compute_begin_us = 0;
        uint64_t compute_complete_us = 0;
        uint64_t compute_work = 0;
        uint64_t compute_work_id = 0;
    };

    impl(llm_transfer_ring_config config, llm_transfer_ring_faults faults) : config(config), faults(faults) {
        if (config.byte_budget == 0 || config.minimum_lanes == 0 || config.trace_capacity == 0 ||
            config.target_device == nullptr ||
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
        lane.flight = {};
        lane.wait_enqueued = false;
        lane.event_complete = false;
        lane.monitoring = false;
        lane.hold_after_h2d = false;
        lane.compute_pending = false;
        lane.compute_complete = false;
        lane.compute_monitoring = false;
        lane.cancelled = false;
        lane.background_queued = false;
        lane.background_running = false;
        lane.background_cold = {};
        lane.background_destination = {};
        lane.h2d_enqueue_us = 0;
        lane.h2d_complete_us = 0;
        lane.compute_begin_us = 0;
        lane.compute_complete_us = 0;
        lane.compute_work = 0;
        lane.compute_work_id = 0;
        lane.state = llm_transfer_lane_state::free;
    }

    void refresh_total_event_counters() {
        counters.live_events = counters.live_h2d_events + counters.live_compute_events;
        counters.peak_live_events = std::max(counters.peak_live_events, counters.live_events);
    }

    void free_event_objects(lane_state & lane) {
        if (lane.event != nullptr) {
            ggml_backend_event_free(lane.event);
            counters.h2d_event_frees++;
            lane.event = nullptr;
        }
        if (lane.compute_event != nullptr) {
            ggml_backend_event_free(lane.compute_event);
            counters.compute_event_frees++;
            lane.compute_event = nullptr;
        }
    }

    void finish_lane_if_ready(lane_state & lane) {
        if (!lane.event_complete || lane.hold_after_h2d ||
            (lane.compute_pending && !lane.compute_complete)) return;
        if (traces.size() != 0) {
            if (counters.trace_records < traces.size()) {
                traces[counters.trace_records++] = { lane.flight, lane.cold,
                    { uint32_t(&lane - lanes.data()), lane.generation }, lane.hot_slot, lane.hot_generation,
                    lane.h2d_enqueue_us, lane.h2d_complete_us, lane.compute_begin_us,
                    lane.compute_complete_us, counters.lane_payload_bytes,
                    lane.compute_work_id, lane.compute_work, lane.cancelled };
            } else {
                counters.trace_records_dropped++;
            }
        }
        release_lane(lane);
    }

    bool has_monitor_work() const {
        for (const auto & lane : lanes) {
            if (lane.state == llm_transfer_lane_state::in_flight && lane.wait_enqueued &&
                !lane.event_complete && !lane.monitoring) return true;
        }
        return false;
    }

    bool has_incomplete_events() const {
        for (const auto & lane : lanes) {
            if (lane.state == llm_transfer_lane_state::in_flight && !lane.event_complete) return true;
        }
        return false;
    }

    bool has_compute_monitor_work() const {
        for (const auto & lane : lanes) {
            if (lane.state == llm_transfer_lane_state::in_flight && lane.compute_pending &&
                !lane.compute_complete && !lane.compute_monitoring) return true;
        }
        return false;
    }

    bool has_background_work() const {
        for (const auto & lane : lanes) {
            if (lane.state == llm_transfer_lane_state::staging && lane.background_queued) return true;
        }
        return false;
    }

    bool has_running_background_work() const {
        for (const auto & lane : lanes) {
            if (lane.background_queued || lane.background_running) return true;
        }
        return false;
    }

    bool has_incomplete_compute_events() const {
        for (const auto & lane : lanes) {
            if (lane.state == llm_transfer_lane_state::in_flight && lane.compute_pending &&
                !lane.compute_complete) return true;
        }
        return false;
    }

    void monitor_main() {
        std::unique_lock<std::mutex> lock(mutex);
        for (;;) {
            condition.wait(lock, [&] { return monitor_stop || has_monitor_work(); });
            uint32_t selected = UINT32_MAX;
            for (uint32_t index = 0; index < lanes.size(); ++index) {
                const auto & lane = lanes[index];
                if (lane.state == llm_transfer_lane_state::in_flight && lane.wait_enqueued &&
                    !lane.event_complete && !lane.monitoring) {
                    selected = index;
                    break;
                }
            }
            if (selected == UINT32_MAX) {
                if (monitor_stop && !has_incomplete_events()) break;
                continue;
            }
            auto & lane = lanes[selected];
            lane.monitoring = true;
            const uint64_t generation = lane.generation;
            ggml_backend_event_t event = lane.event;
            lock.unlock();
            if (config.delay_event_monitor_ms_for_testing != 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(config.delay_event_monitor_ms_for_testing));
            }
            ggml_backend_event_synchronize(event);
            const uint64_t completed_us = uint64_t(ggml_time_us());
            lock.lock();
            auto & completed = lanes[selected];
            if (completed.generation == generation && completed.state == llm_transfer_lane_state::in_flight) {
                completed.monitoring = false;
                completed.event_complete = true;
                completed.h2d_complete_us = completed_us;
                counters.event_synchronizations++;
                counters.h2d_event_synchronizations++;
                if (counters.live_h2d_events > 0) counters.live_h2d_events--;
                refresh_total_event_counters();
                counters.last_h2d_event_complete_us = completed_us;
                finish_lane_if_ready(completed);
            }
            condition.notify_all();
        }
    }

    void compute_monitor_main() {
        std::unique_lock<std::mutex> lock(mutex);
        for (;;) {
            condition.wait(lock, [&] { return monitor_stop || has_compute_monitor_work(); });
            uint32_t selected = UINT32_MAX;
            for (uint32_t index = 0; index < lanes.size(); ++index) {
                const auto & lane = lanes[index];
                if (lane.state == llm_transfer_lane_state::in_flight && lane.compute_pending &&
                    !lane.compute_complete && !lane.compute_monitoring) {
                    selected = index;
                    break;
                }
            }
            if (selected == UINT32_MAX) {
                if (monitor_stop && !has_incomplete_compute_events()) break;
                continue;
            }
            auto & lane = lanes[selected];
            lane.compute_monitoring = true;
            const uint64_t generation = lane.generation;
            ggml_backend_event_t event = lane.compute_event;
            lock.unlock();
            ggml_backend_event_synchronize(event);
            const uint64_t completed_us = uint64_t(ggml_time_us());
            lock.lock();
            auto & completed = lanes[selected];
            if (completed.generation == generation && completed.state == llm_transfer_lane_state::in_flight) {
                completed.compute_monitoring = false;
                completed.compute_complete = true;
                completed.compute_complete_us = completed_us;
                counters.compute_event_synchronizations++;
                if (counters.live_compute_events > 0) counters.live_compute_events--;
                refresh_total_event_counters();
                finish_lane_if_ready(completed);
            }
            condition.notify_all();
        }
    }

    llm_transfer_ring_config config;
    llm_transfer_ring_faults faults;
    mutable std::mutex mutex;
    std::condition_variable condition;
    std::thread monitor;
    std::thread compute_monitor;
    std::thread background_worker;
    bool monitor_stop = false;
    bool background_stop = false;
    ggml_backend_buffer_ptr arena;
    ggml_backend_ptr transfer_backend;
    uint8_t * base = nullptr;
    std::vector<span_layout> spans;
    std::vector<lane_state> lanes;
    std::vector<llm_transfer_binding> background_bindings;
    llm_transfer_ring_diagnostics counters;
    std::vector<llm_transfer_interval> traces;
    uint64_t compute_begin_us = 0;
    uint64_t compute_work = 0;
    uint64_t next_compute_work_id = 1;
    uint64_t compute_work_id = 0;
    ggml_backend_dev_t compute_device = nullptr;
};

llm_expert_transfer_ring::llm_expert_transfer_ring(
        llm_transfer_ring_config config, llm_transfer_ring_faults faults) :
    pimpl(std::make_unique<impl>(config, faults)) {}

llm_expert_transfer_ring::~llm_expert_transfer_ring() {
    if (!pimpl) return;
    {
        std::lock_guard<std::mutex> lock(pimpl->mutex);
        pimpl->background_stop = true;
        pimpl->condition.notify_all();
    }
    if (pimpl->background_worker.joinable()) pimpl->background_worker.join();
    {
        std::lock_guard<std::mutex> lock(pimpl->mutex);
        pimpl->monitor_stop = true;
        for (auto & lane : pimpl->lanes) {
            if (lane.state == llm_transfer_lane_state::in_flight) lane.wait_enqueued = true;
        }
        pimpl->condition.notify_all();
    }
    if (pimpl->monitor.joinable()) pimpl->monitor.join();
    if (pimpl->compute_monitor.joinable()) pimpl->compute_monitor.join();
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    for (auto & lane : pimpl->lanes) {
        pimpl->release_lane(lane);
        pimpl->free_event_objects(lane);
    }
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
        uint64_t total_event_capacity = 0;
        if (!checked_mul(lane_count64, 2, total_event_capacity) || total_event_capacity > UINT32_MAX) {
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
        bool event_capable = false;
        if (pinned && !pimpl->config.force_no_events_for_testing) {
            ggml_backend_dev_props properties{};
            ggml_backend_dev_get_props(pimpl->config.target_device, &properties);
            if (properties.caps.async && properties.caps.events) {
                pimpl->transfer_backend.reset(ggml_backend_dev_init(pimpl->config.target_device, nullptr));
                event_capable = bool(pimpl->transfer_backend);
                if (event_capable) {
                    for (auto & lane : pimpl->lanes) {
                        lane.event = ggml_backend_event_new(pimpl->config.target_device);
                        if (lane.event != nullptr) pimpl->counters.h2d_event_allocations++;
                        lane.compute_event = ggml_backend_event_new(pimpl->config.target_device);
                        if (lane.compute_event != nullptr) pimpl->counters.compute_event_allocations++;
                        if (lane.event == nullptr || lane.compute_event == nullptr) {
                            event_capable = false;
                            break;
                        }
                    }
                }
            }
            if (!event_capable) fallback_reason = "transfer-events-unavailable";
        } else if (pinned) {
            fallback_reason = "forced-no-events-for-testing";
        }
        if (!event_capable) {
            for (auto & lane : pimpl->lanes) {
                pimpl->free_event_objects(lane);
            }
            pimpl->transfer_backend.reset();
        }
        pimpl->counters.pageable_fallback = !pinned;
        pimpl->counters.pinned_or_registered_bytes = pinned ? actual : 0;
        pimpl->counters.acquisition_method = pinned ?
            (event_capable ? "device-native-host-events" : "device-native-host-synchronous") : "pageable-cpu";
        pimpl->counters.fallback_reason = event_capable ? "" : fallback_reason;
        pimpl->counters.fallback_count = event_capable ? 0 : 1;
        pimpl->counters.dedicated_transfer_backend = event_capable;
        pimpl->counters.event_capable = event_capable;
        pimpl->counters.h2d_event_capacity = event_capable ? lane_count : 0;
        pimpl->counters.compute_event_capacity = event_capable ? lane_count : 0;
        pimpl->counters.event_capacity = event_capable ? uint32_t(total_event_capacity) : 0;
        pimpl->counters.trace_capacity = pimpl->config.trace_capacity;
        pimpl->traces.assign(pimpl->config.trace_capacity, {});
        pimpl->background_bindings.reserve(1);
        pimpl->arena = std::move(candidate);
        if (event_capable) {
            pimpl->monitor = std::thread([this] { pimpl->monitor_main(); });
            pimpl->compute_monitor = std::thread([this] { pimpl->compute_monitor_main(); });
            pimpl->background_worker = std::thread([this] {
                std::unique_lock<std::mutex> lock(pimpl->mutex);
                for (;;) {
                    pimpl->condition.wait(lock, [&] {
                        return pimpl->background_stop || pimpl->has_background_work();
                    });
                    uint32_t selected = UINT32_MAX;
                    for (uint32_t index = 0; index < pimpl->lanes.size(); ++index) {
                        const auto & lane = pimpl->lanes[index];
                        if (lane.state == llm_transfer_lane_state::staging && lane.background_queued) {
                            selected = index;
                            break;
                        }
                    }
                    if (selected == UINT32_MAX) {
                        if (pimpl->background_stop && !pimpl->has_running_background_work()) break;
                        continue;
                    }
                    auto & lane = pimpl->lanes[selected];
                    const llm_transfer_lane_reference reference = { selected, lane.generation };
                    const auto cold_bundle = lane.background_cold;
                    const auto destination = lane.background_destination;
                    const uint32_t hot_slot = lane.hot_slot;
                    lane.background_queued = false;
                    lane.background_running = true;
                    lock.unlock();

                    if (pimpl->config.delay_background_stage_ms_for_testing != 0) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(
                            pimpl->config.delay_background_stage_ms_for_testing));
                    }
                    auto result = stage(reference, cold_bundle);
                    if (result.is_ready()) {
                        pimpl->background_bindings.clear();
                        pimpl->background_bindings.push_back({ reference, destination, hot_slot });
                        result = transfer_wave(pimpl->transfer_backend.get(), pimpl->background_bindings);
                    }
                    if (result.is_ready()) result = monitor_h2d(reference);

                    lock.lock();
                    if (selected < pimpl->lanes.size()) {
                        auto & completed = pimpl->lanes[selected];
                        if (completed.generation == reference.generation) {
                            completed.background_running = false;
                            if (!result.is_ready() &&
                                (completed.state == llm_transfer_lane_state::failed ||
                                 completed.state == llm_transfer_lane_state::staging)) {
                                const uint32_t failed_hot_slot = completed.hot_slot;
                                const uint64_t failed_hot_generation = completed.hot_generation;
                                pimpl->release_lane(completed);
                                completed.failed_generation = reference.generation;
                                completed.failed_hot_slot = failed_hot_slot;
                                completed.failed_hot_generation = failed_hot_generation;
                            }
                        }
                    }
                    pimpl->condition.notify_all();
                }
            });
        }
        return llm_expert_provider_result::success();
    } catch (const std::bad_alloc &) {
        for (auto & lane : pimpl->lanes) {
            pimpl->free_event_objects(lane);
        }
        pimpl->transfer_backend.reset(); pimpl->arena.reset();
        pimpl->spans.clear(); pimpl->lanes.clear(); pimpl->base = nullptr;
        return llm_expert_provider_result::failure(llm_expert_provider_error::allocation_failed);
    } catch (...) {
        for (auto & lane : pimpl->lanes) {
            pimpl->free_event_objects(lane);
        }
        pimpl->transfer_backend.reset(); pimpl->arena.reset();
        pimpl->spans.clear(); pimpl->lanes.clear(); pimpl->base = nullptr;
        return llm_expert_provider_result::failure(llm_expert_provider_error::initialization_failed);
    }
}

llm_expert_provider_result llm_expert_transfer_ring::reserve(
        llm_cold_expert_cache & cold_cache,
        llm_cold_reference cold,
        uint32_t hot_slot,
        uint64_t hot_generation,
        llm_transfer_lane_reference & reference,
        llm_expert_flight_id flight) noexcept {
    std::unique_lock<std::mutex> lock(pimpl->mutex);
    if (!pimpl->arena) return llm_expert_provider_result::failure(llm_expert_provider_error::initialization_failed);
    uint32_t index = 0;
    while (index < pimpl->lanes.size() && pimpl->lanes[index].state != llm_transfer_lane_state::free) index++;
    if (index == pimpl->lanes.size()) {
        pimpl->condition.wait(lock, [&] {
            for (const auto & lane : pimpl->lanes) {
                if (lane.state == llm_transfer_lane_state::free ||
                    (lane.state == llm_transfer_lane_state::in_flight && lane.event_complete &&
                     !lane.hold_after_h2d && (!lane.compute_pending || lane.compute_complete))) return true;
            }
            return false;
        });
        for (index = 0; index < pimpl->lanes.size(); ++index) {
            if (pimpl->lanes[index].state == llm_transfer_lane_state::free) break;
            if (pimpl->lanes[index].state == llm_transfer_lane_state::in_flight &&
                pimpl->lanes[index].event_complete &&
                !pimpl->lanes[index].hold_after_h2d &&
                (!pimpl->lanes[index].compute_pending || pimpl->lanes[index].compute_complete)) {
                pimpl->release_lane(pimpl->lanes[index]);
                break;
            }
        }
    }
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
    lane.flight = flight;
    lane.cold_cache = &cold_cache;
    reference = { index, lane.generation };
    pimpl->counters.lane_reservations++;
    return llm_expert_provider_result::success();
}

llm_expert_provider_result llm_expert_transfer_ring::stage(
        llm_transfer_lane_reference reference,
        const llm_expert_bundle_descriptor & cold_bundle) noexcept {
    std::unique_lock<std::mutex> lock(pimpl->mutex);
    if (!pimpl->valid_lane(reference) || pimpl->lanes[reference.lane].state != llm_transfer_lane_state::staging) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::stale_generation);
    }
    auto & lane = pimpl->lanes[reference.lane];
    size_t copied = 0;
    uint64_t copied_bytes = 0;
    const int64_t start = ggml_time_us();
    lock.unlock();
    for (const auto & span : pimpl->spans) {
        const auto * source = member(cold_bundle, span.projection, span.member);
        const int axis = expert_axis(source, cold_bundle.n_expert, span.member == 0);
        if (copied >= pimpl->faults.fail_stage_after_spans || source == nullptr || axis < 0 ||
            source->nb[axis] != span.bytes || lane.cold.slot >= uint32_t(source->ne[axis]) || source->data == nullptr) {
            lock.lock();
            if (pimpl->valid_lane(reference)) lane.state = llm_transfer_lane_state::failed;
            pimpl->condition.notify_all();
            return llm_expert_provider_result::failure(llm_expert_provider_error::copy_failed);
        }
        const auto * source_data = static_cast<const uint8_t *>(source->data) + size_t(lane.cold.slot)*span.bytes;
        std::memcpy(pimpl->base + size_t(reference.lane)*pimpl->counters.lane_footprint + span.offset,
            source_data, span.bytes);
        copied_bytes += span.bytes;
        copied++;
    }
    lock.lock();
    if (!pimpl->valid_lane(reference) || lane.state != llm_transfer_lane_state::staging) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::stale_generation);
    }
    pimpl->counters.stage_bytes += copied_bytes;
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
    const bool async = pimpl->counters.event_capable;
    if (async && (backend == nullptr || ggml_backend_get_device(backend) != pimpl->config.target_device)) {
        std::fprintf(stderr, "error: expert transfer backend/device mismatch during enqueue\n");
        fail_bindings();
        return llm_expert_provider_result::failure(llm_expert_provider_error::invalid_binding);
    }
    const int64_t start = ggml_time_us();
    uint64_t bytes = 0;
    if (async && pimpl->config.h2d_gate_event_for_testing != nullptr) {
        ggml_backend_event_wait(pimpl->transfer_backend.get(), pimpl->config.h2d_gate_event_for_testing);
    }
    for (const auto & binding : bindings) {
        auto & lane = pimpl->lanes[binding.lane.lane];
        if (async) lane.state = llm_transfer_lane_state::in_flight;
        if (async) {
            lane.h2d_enqueue_us = uint64_t(start);
            if (pimpl->counters.first_h2d_enqueue_us == 0) {
                pimpl->counters.first_h2d_enqueue_us = uint64_t(start);
            }
        }
        for (const auto & span : pimpl->spans) {
            auto destination = binding.destination;
            auto * target = member(destination, span.projection, span.member);
            const auto * data = pimpl->base + size_t(binding.lane.lane)*pimpl->counters.lane_footprint + span.offset;
            const size_t offset = size_t(binding.hot_slot)*span.bytes;
            if (async) {
                ggml_backend_tensor_set_async(pimpl->transfer_backend.get(), target, data, offset, span.bytes);
                pimpl->counters.async_enqueues++;
            } else {
                ggml_backend_tensor_set(target, data, offset, span.bytes);
                pimpl->counters.synchronous_copies++;
            }
            bytes += span.bytes;
        }
        if (async) {
            ggml_backend_event_record(lane.event, pimpl->transfer_backend.get());
            pimpl->counters.event_records++;
            pimpl->counters.h2d_event_records++;
            pimpl->counters.live_h2d_events++;
            pimpl->counters.peak_live_h2d_events = std::max(
                pimpl->counters.peak_live_h2d_events, pimpl->counters.live_h2d_events);
            pimpl->refresh_total_event_counters();
        }
    }
    if (async) {
        pimpl->counters.peak_in_flight_lanes = std::max<uint64_t>(
            pimpl->counters.peak_in_flight_lanes, bindings.size());
    }
    pimpl->counters.waves++;
    pimpl->counters.h2d_bytes += bytes;
    pimpl->counters.h2d_time_us += ggml_time_us() - start;
    if (!async) {
        for (const auto & binding : bindings) pimpl->release_lane(pimpl->lanes[binding.lane.lane]);
    }
    return llm_expert_provider_result::success();
}

llm_expert_provider_result llm_expert_transfer_ring::try_queue_background_transfer(
        llm_cold_expert_cache & cold_cache,
        llm_cold_reference cold,
        uint32_t hot_slot,
        uint64_t hot_generation,
        const llm_expert_bundle_descriptor & cold_bundle,
        const llm_expert_bundle_descriptor & destination,
        llm_transfer_lane_reference & reference,
        llm_expert_flight_id flight) noexcept {
    std::unique_lock<std::mutex> lock(pimpl->mutex, std::try_to_lock);
    if (!lock.owns_lock() || pimpl->background_stop || !pimpl->counters.event_capable || !pimpl->arena) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::busy);
    }
    uint32_t index = 0;
    while (index < pimpl->lanes.size() && pimpl->lanes[index].state != llm_transfer_lane_state::free) index++;
    if (index == pimpl->lanes.size()) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::busy);
    }
    auto & selected = pimpl->lanes[index];
    if (selected.generation == std::numeric_limits<uint64_t>::max()) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::generation_exhausted);
    }
    const auto acquired = cold_cache.acquire(cold, llm_cold_reference_kind::transfer);
    if (!acquired.is_ready()) return acquired;
    selected.generation++;
    selected.state = llm_transfer_lane_state::staging;
    selected.cold = cold;
    selected.hot_slot = hot_slot;
    selected.hot_generation = hot_generation;
    selected.flight = flight;
    selected.cold_cache = &cold_cache;
    selected.background_cold = cold_bundle;
    selected.background_destination = destination;
    selected.background_queued = true;
    selected.hold_after_h2d = true;
    reference = { index, selected.generation };
    pimpl->counters.lane_reservations++;
    pimpl->condition.notify_all();
    return llm_expert_provider_result::success();
}

llm_expert_provider_result llm_expert_transfer_ring::wait_for_hot(
        ggml_backend_t compute_backend,
        uint32_t hot_slot,
        uint64_t hot_generation) noexcept {
    std::unique_lock<std::mutex> lock(pimpl->mutex);
    bool conflicting_generation = false;
    for (auto & lane : pimpl->lanes) {
        if (lane.failed_hot_slot == hot_slot && lane.failed_hot_generation == hot_generation) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::copy_failed);
        }
        if (lane.hot_slot == hot_slot && lane.state != llm_transfer_lane_state::free &&
            lane.hot_generation != hot_generation) conflicting_generation = true;
        if (lane.hot_slot != hot_slot || lane.hot_generation != hot_generation ||
            lane.state == llm_transfer_lane_state::free) continue;
        const uint64_t generation = lane.generation;
        if (lane.state == llm_transfer_lane_state::staging) {
            pimpl->condition.wait(lock, [&] {
                return lane.generation != generation || lane.state != llm_transfer_lane_state::staging;
            });
        }
        if (lane.failed_generation == generation) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::copy_failed);
        }
        if (lane.generation != generation || lane.state == llm_transfer_lane_state::failed) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::copy_failed);
        }
        if (lane.state == llm_transfer_lane_state::free) return llm_expert_provider_result::success();
        if (lane.state != llm_transfer_lane_state::in_flight) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::invalid_binding);
        }
        if (compute_backend == nullptr || ggml_backend_get_device(compute_backend) != pimpl->config.target_device ||
            lane.event == nullptr || lane.compute_event == nullptr) {
            std::fprintf(stderr, "error: expert transfer backend/device mismatch during compute wait\n");
            return llm_expert_provider_result::failure(llm_expert_provider_error::invalid_binding);
        }
        if (pimpl->compute_begin_us != 0 && pimpl->compute_work != 0 &&
            pimpl->compute_device == ggml_backend_get_device(compute_backend)) {
            ggml_backend_event_record(lane.compute_event, compute_backend);
            lane.compute_pending = true;
            lane.compute_complete = false;
            lane.compute_begin_us = pimpl->compute_begin_us;
            lane.compute_work = pimpl->compute_work;
            lane.compute_work_id = pimpl->compute_work_id;
            pimpl->counters.compute_event_records++;
            pimpl->counters.compute_work += pimpl->compute_work;
            pimpl->counters.live_compute_events++;
            pimpl->counters.peak_live_compute_events = std::max(
                pimpl->counters.peak_live_compute_events, pimpl->counters.live_compute_events);
            pimpl->refresh_total_event_counters();
        }
        ggml_backend_event_wait(compute_backend, lane.event);
        lane.hold_after_h2d = false;
        lane.wait_enqueued = true;
        pimpl->counters.compute_waits++;
        pimpl->counters.h2d_event_waits++;
        pimpl->compute_begin_us = 0;
        pimpl->compute_work = 0;
        pimpl->compute_work_id = 0;
        pimpl->compute_device = nullptr;
        pimpl->finish_lane_if_ready(lane);
        pimpl->condition.notify_all();
        return llm_expert_provider_result::success();
    }
    if (conflicting_generation) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::stale_generation);
    }
    return llm_expert_provider_result::success();
}

llm_expert_provider_result llm_expert_transfer_ring::monitor_h2d(
        llm_transfer_lane_reference reference) noexcept {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    if (reference.lane >= pimpl->lanes.size() ||
        pimpl->lanes[reference.lane].generation != reference.generation) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::stale_generation);
    }
    auto & lane = pimpl->lanes[reference.lane];
    if (lane.state == llm_transfer_lane_state::free) {
        return llm_expert_provider_result::success();
    }
    if (lane.state != llm_transfer_lane_state::in_flight || lane.event == nullptr) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::invalid_binding);
    }
    lane.wait_enqueued = true;
    pimpl->condition.notify_all();
    return llm_expert_provider_result::success();
}

llm_expert_provider_result llm_expert_transfer_ring::poll_h2d(
        llm_transfer_lane_reference reference,
        bool & complete,
        uint64_t * remaining_submitted_bytes) noexcept {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    complete = false;
    if (remaining_submitted_bytes != nullptr) *remaining_submitted_bytes = 0;
    if (reference.lane >= pimpl->lanes.size() ||
        pimpl->lanes[reference.lane].generation != reference.generation) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::stale_generation);
    }
    auto & lane = pimpl->lanes[reference.lane];
    const auto state = lane.state;
    if (state == llm_transfer_lane_state::free) {
        if (lane.failed_generation == reference.generation) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::copy_failed);
        }
        complete = true;
        return llm_expert_provider_result::success();
    }
    if (state == llm_transfer_lane_state::in_flight) {
        if (lane.event_complete) {
            complete = true;
            lane.hold_after_h2d = false;
            pimpl->finish_lane_if_ready(lane);
            pimpl->condition.notify_all();
        } else if (remaining_submitted_bytes != nullptr) {
            // Event-granularity accounting is deterministic: an incomplete
            // submitted copy retains its full immutable payload.
            *remaining_submitted_bytes = pimpl->counters.lane_payload_bytes;
        }
        return llm_expert_provider_result::success();
    }
    if (state == llm_transfer_lane_state::staging) {
        return llm_expert_provider_result::success();
    }
    return llm_expert_provider_result::failure(llm_expert_provider_error::copy_failed);
}

llm_expert_provider_result llm_expert_transfer_ring::release_failed_background(
        llm_transfer_lane_reference reference) noexcept {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    if (reference.lane >= pimpl->lanes.size() ||
        pimpl->lanes[reference.lane].generation != reference.generation) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::stale_generation);
    }
    auto & lane = pimpl->lanes[reference.lane];
    if (lane.state == llm_transfer_lane_state::free) return llm_expert_provider_result::success();
    if (lane.state != llm_transfer_lane_state::failed || lane.background_running || lane.background_queued) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::busy);
    }
    pimpl->release_lane(lane);
    pimpl->condition.notify_all();
    return llm_expert_provider_result::success();
}

llm_expert_provider_result llm_expert_transfer_ring::begin_compute_work(
        ggml_backend_t compute_backend,
        uint64_t work) noexcept {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    if (compute_backend == nullptr || work == 0 || !pimpl->config.allow_controlled_compute_for_testing ||
        !pimpl->counters.event_capable || pimpl->compute_begin_us != 0 ||
        ggml_backend_get_device(compute_backend) != pimpl->config.target_device) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::invalid_binding);
    }
    if (pimpl->next_compute_work_id == 0 || pimpl->next_compute_work_id == UINT64_MAX) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::generation_exhausted);
    }
    pimpl->compute_begin_us = uint64_t(ggml_time_us());
    pimpl->compute_work = work;
    pimpl->compute_work_id = pimpl->next_compute_work_id++;
    pimpl->compute_device = ggml_backend_get_device(compute_backend);
    return llm_expert_provider_result::success();
}

llm_expert_provider_result llm_expert_transfer_ring::cancel_after_h2d(
        llm_transfer_lane_reference reference) noexcept {
    std::unique_lock<std::mutex> lock(pimpl->mutex);
    if (reference.lane >= pimpl->lanes.size() ||
        pimpl->lanes[reference.lane].generation != reference.generation) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::stale_generation);
    }
    auto & lane = pimpl->lanes[reference.lane];
    if (lane.state == llm_transfer_lane_state::free) {
        return llm_expert_provider_result::success();
    }
    if (lane.state != llm_transfer_lane_state::in_flight) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::stale_generation);
    }
    lane.cancelled = true;
    lane.wait_enqueued = true;
    pimpl->counters.h2d_event_cancellations++;
    if (lane.compute_pending) pimpl->counters.compute_event_cancellations++;
    pimpl->condition.notify_all();
    pimpl->condition.wait(lock, [&] {
        return lane.generation != reference.generation || lane.state != llm_transfer_lane_state::in_flight ||
            (lane.event_complete && (!lane.compute_pending || lane.compute_complete));
    });
    if (lane.generation == reference.generation && lane.state == llm_transfer_lane_state::in_flight) {
        pimpl->release_lane(lane);
    }
    return llm_expert_provider_result::success();
}

llm_expert_provider_result llm_expert_transfer_ring::set_h2d_gate_event_for_testing(
        ggml_backend_event_t event) noexcept {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    for (const auto & lane : pimpl->lanes) {
        if (lane.state != llm_transfer_lane_state::free) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::busy);
        }
    }
    pimpl->config.h2d_gate_event_for_testing = event;
    return llm_expert_provider_result::success();
}

llm_expert_provider_result llm_expert_transfer_ring::retire_hot(
        uint32_t hot_slot,
        uint64_t hot_generation) noexcept {
    std::unique_lock<std::mutex> lock(pimpl->mutex);
    for (auto & lane : pimpl->lanes) {
        if (lane.hot_slot != hot_slot || lane.hot_generation != hot_generation ||
            lane.state != llm_transfer_lane_state::in_flight) continue;
        const uint64_t generation = lane.generation;
        pimpl->condition.wait(lock, [&] {
            return lane.generation != generation || lane.state != llm_transfer_lane_state::in_flight ||
                (lane.event_complete && (!lane.compute_pending || lane.compute_complete));
        });
        if (lane.generation == generation && lane.state == llm_transfer_lane_state::in_flight) {
            pimpl->release_lane(lane);
        }
    }
    return llm_expert_provider_result::success();
}

llm_expert_provider_result llm_expert_transfer_ring::cleanup_failed_lanes() noexcept {
    std::unique_lock<std::mutex> lock(pimpl->mutex);
    if (pimpl->faults.fail_cleanup) {
        pimpl->counters.failed_cleanups++;
        return llm_expert_provider_result::failure(llm_expert_provider_error::copy_failed);
    }
    pimpl->condition.wait(lock, [&] { return !pimpl->has_running_background_work(); });
    for (auto & lane : pimpl->lanes) {
        if (lane.state == llm_transfer_lane_state::in_flight) {
            const uint64_t generation = lane.generation;
            lane.wait_enqueued = true;
            pimpl->condition.notify_all();
            pimpl->condition.wait(lock, [&] {
                return lane.generation != generation || lane.state != llm_transfer_lane_state::in_flight ||
                    (lane.event_complete && (!lane.compute_pending || lane.compute_complete));
            });
            if (lane.generation == generation && lane.state == llm_transfer_lane_state::in_flight) {
                pimpl->release_lane(lane);
            }
            continue;
        }
        if (lane.state == llm_transfer_lane_state::failed || lane.state == llm_transfer_lane_state::staging) {
            pimpl->release_lane(lane);
        }
    }
    return llm_expert_provider_result::success();
}

llm_expert_provider_result llm_expert_transfer_ring::surrender() noexcept {
    std::unique_lock<std::mutex> lock(pimpl->mutex);
    pimpl->background_stop = true;
    pimpl->condition.notify_all();
    lock.unlock();
    if (pimpl->background_worker.joinable()) pimpl->background_worker.join();
    lock.lock();
    for (auto & lane : pimpl->lanes) {
        if (lane.state == llm_transfer_lane_state::in_flight) {
            const uint64_t generation = lane.generation;
            lane.wait_enqueued = true;
            pimpl->condition.notify_all();
            pimpl->condition.wait(lock, [&] {
                return lane.generation != generation || lane.state != llm_transfer_lane_state::in_flight ||
                    (lane.event_complete && (!lane.compute_pending || lane.compute_complete));
            });
            if (lane.generation == generation && lane.state == llm_transfer_lane_state::in_flight) {
                pimpl->release_lane(lane);
            }
        } else if (lane.state != llm_transfer_lane_state::free) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::busy);
        }
    }
    pimpl->monitor_stop = true;
    pimpl->condition.notify_all();
    lock.unlock();
    if (pimpl->monitor.joinable()) pimpl->monitor.join();
    if (pimpl->compute_monitor.joinable()) pimpl->compute_monitor.join();
    lock.lock();
    for (auto & lane : pimpl->lanes) {
        pimpl->free_event_objects(lane);
    }
    pimpl->transfer_backend.reset();
    pimpl->arena.reset(); pimpl->base = nullptr; pimpl->spans.clear(); pimpl->lanes.clear();
    pimpl->background_bindings.clear();
    pimpl->counters.actual_bytes = 0;
    pimpl->counters.pinned_or_registered_bytes = 0;
    pimpl->counters.effective_lanes = 0;
    pimpl->counters.h2d_event_capacity = 0;
    pimpl->counters.compute_event_capacity = 0;
    pimpl->counters.event_capacity = 0;
    pimpl->counters.event_capable = false;
    pimpl->counters.dedicated_transfer_backend = false;
    pimpl->counters.unused_budget_bytes = pimpl->config.byte_budget;
    return llm_expert_provider_result::success();
}

llm_expert_provider_result llm_expert_transfer_ring::validate_invariants() noexcept {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    uint64_t inflight = 0;
    uint64_t incomplete_h2d_events = 0;
    uint64_t incomplete_compute_events = 0;
    for (const auto & lane : pimpl->lanes) {
        if (lane.state == llm_transfer_lane_state::free && lane.cold_cache != nullptr) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
        }
        if (lane.state != llm_transfer_lane_state::free && lane.cold_cache == nullptr) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
        }
        inflight += lane.state == llm_transfer_lane_state::in_flight;
        incomplete_h2d_events += lane.state == llm_transfer_lane_state::in_flight && !lane.event_complete;
        incomplete_compute_events += lane.state == llm_transfer_lane_state::in_flight &&
            lane.compute_pending && !lane.compute_complete;
    }
    if (pimpl->counters.pageable_fallback &&
        (pimpl->counters.pinned_or_registered_bytes != 0 || pimpl->counters.async_enqueues != 0 || inflight != 0 ||
         pimpl->counters.h2d_event_capacity != 0 || pimpl->counters.compute_event_capacity != 0 ||
         pimpl->counters.event_capacity != 0 || pimpl->counters.h2d_event_records != 0 ||
         pimpl->counters.compute_event_records != 0 || pimpl->counters.live_events != 0)) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
    }
    if (incomplete_h2d_events != pimpl->counters.live_h2d_events ||
        incomplete_compute_events != pimpl->counters.live_compute_events ||
        pimpl->counters.live_events != pimpl->counters.live_h2d_events + pimpl->counters.live_compute_events ||
        pimpl->counters.event_records != pimpl->counters.h2d_event_records ||
        pimpl->counters.event_synchronizations != pimpl->counters.h2d_event_synchronizations ||
        pimpl->counters.h2d_event_records < pimpl->counters.h2d_event_synchronizations ||
        pimpl->counters.compute_event_records < pimpl->counters.compute_event_synchronizations ||
        pimpl->counters.h2d_event_allocations < pimpl->counters.h2d_event_frees ||
        pimpl->counters.compute_event_allocations < pimpl->counters.compute_event_frees ||
        pimpl->counters.h2d_event_allocations - pimpl->counters.h2d_event_frees !=
            pimpl->counters.h2d_event_capacity ||
        pimpl->counters.compute_event_allocations - pimpl->counters.compute_event_frees !=
            pimpl->counters.compute_event_capacity ||
        pimpl->counters.live_h2d_events !=
            pimpl->counters.h2d_event_records - pimpl->counters.h2d_event_synchronizations ||
        pimpl->counters.live_compute_events !=
            pimpl->counters.compute_event_records - pimpl->counters.compute_event_synchronizations ||
        (pimpl->counters.event_capable &&
            (pimpl->counters.h2d_event_capacity != pimpl->lanes.size() ||
             pimpl->counters.compute_event_capacity != pimpl->lanes.size() ||
             pimpl->counters.event_capacity != pimpl->lanes.size()*2))) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
    }
    return llm_expert_provider_result::success();
}

std::vector<llm_transfer_interval> llm_expert_transfer_ring::completed_intervals() const {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    return std::vector<llm_transfer_interval>(
        pimpl->traces.begin(), pimpl->traces.begin() + size_t(pimpl->counters.trace_records));
}

llm_transfer_ring_diagnostics llm_expert_transfer_ring::diagnostics() const {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    auto result = pimpl->counters;
    result.lanes.assign(pimpl->lanes.begin(), pimpl->lanes.end());
    result.h2d_compute_overlap_us = 0;
    result.h2d_compute_overlap_bytes = 0;
    result.h2d_compute_overlap_work = 0;
    result.h2d_compute_overlap_flights = 0;
    std::vector<std::pair<uint64_t, uint64_t>> h2d_ranges;
    std::vector<std::pair<uint64_t, uint64_t>> compute_ranges;
    h2d_ranges.reserve(size_t(pimpl->counters.trace_records));
    compute_ranges.reserve(size_t(pimpl->counters.trace_records));
    for (uint64_t index = 0; index < pimpl->counters.trace_records; ++index) {
        const auto & trace = pimpl->traces[index];
        if (trace.h2d_complete_us > trace.h2d_enqueue_us) {
            h2d_ranges.emplace_back(trace.h2d_enqueue_us, trace.h2d_complete_us);
        }
        if (trace.compute_work_id != 0 && trace.compute_complete_us > trace.compute_begin_us) {
            compute_ranges.emplace_back(trace.compute_begin_us, trace.compute_complete_us);
        }
    }
    result.h2d_compute_overlap_us = interval_union_intersection_us(h2d_ranges, compute_ranges);
    std::vector<llm_expert_flight_id> participating_flights;
    std::vector<uint64_t> participating_work;
    for (uint64_t h2d_index = 0; h2d_index < pimpl->counters.trace_records; ++h2d_index) {
        const auto & h2d = pimpl->traces[h2d_index];
        bool participates = false;
        for (uint64_t compute_index = 0; compute_index < pimpl->counters.trace_records; ++compute_index) {
            const auto & compute = pimpl->traces[compute_index];
            if (compute.compute_work_id == 0) continue;
            const uint64_t begin = std::max(h2d.h2d_enqueue_us, compute.compute_begin_us);
            const uint64_t end = std::min(h2d.h2d_complete_us, compute.compute_complete_us);
            if (end <= begin) continue;
            participates = true;
            if (std::find(participating_work.begin(), participating_work.end(), compute.compute_work_id) ==
                    participating_work.end()) {
                participating_work.push_back(compute.compute_work_id);
                result.h2d_compute_overlap_work += compute.compute_work;
            }
        }
        if (!participates) continue;
        result.h2d_compute_overlap_bytes += h2d.bytes;
        if (h2d.flight.valid() && std::none_of(participating_flights.begin(), participating_flights.end(),
                [&](const auto & flight) { return same_flight(flight, h2d.flight); })) {
            participating_flights.push_back(h2d.flight);
        }
    }
    result.h2d_compute_overlap_flights = participating_flights.size();
    return result;
}
