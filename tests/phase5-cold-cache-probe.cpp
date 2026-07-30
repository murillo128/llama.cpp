#include "llama-expert-transfer-ring.h"

#include "ggml-cpp.h"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct fixture {
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buffer;
    ggml_tensor * up = nullptr;
    ggml_tensor * gate = nullptr;
    ggml_tensor * down = nullptr;
    int32_t n_expert;

    fixture(int32_t n_expert, ggml_backend_buffer_type_t buft, bool fill) : n_expert(n_expert) {
        ggml_init_params params = { ggml_tensor_overhead()*8, nullptr, true };
        ctx.reset(ggml_init(params));
        if (!ctx) throw std::runtime_error("context allocation failed");
        up = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 8, 16, n_expert);
        gate = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 8, 16, n_expert);
        down = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 16, 8, n_expert);
        buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft));
        if (!buffer) throw std::runtime_error("tensor allocation failed");
        if (fill) {
            uint8_t pattern = 0x30;
            for (auto * tensor : { up, gate, down }) {
                for (int32_t expert = 0; expert < n_expert; ++expert) {
                    std::vector<uint8_t> bytes(tensor->nb[2], pattern + expert);
                    ggml_backend_tensor_set(tensor, bytes.data(), size_t(expert)*tensor->nb[2], bytes.size());
                }
                pattern += 0x20;
            }
        }
    }

    llm_expert_bundle_descriptor bundle() const {
        return {
            0, n_expert,
            llm_expert_projection_descriptor::from(up, nullptr, nullptr),
            llm_expert_projection_descriptor::from(gate, nullptr, nullptr),
            {},
            llm_expert_projection_descriptor::from(down, nullptr, nullptr),
        };
    }
};

bool slot_matches(const fixture & source, const fixture & hot, int32_t expert, uint32_t slot) {
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
        if (expected != actual) return false;
    }
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    bool force_pageable = false;
    if (argc == 2 && std::string(argv[1]) == "--force-pageable") force_pageable = true;
    else if (argc != 1) return 2;

    ggml_backend_load_all();
    auto * device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!device) return 3;
    ggml_backend_ptr backend(ggml_backend_dev_init(device, nullptr));
    if (!backend) return 4;

    fixture source(4, ggml_backend_cpu_buffer_type(), true);
    fixture hot(2, ggml_backend_dev_buffer_type(device), false);
    llm_cold_expert_cache cold({ 1U << 20, 2, 1, 4, 0 });
    if (!cold.initialize(source.bundle()).is_ready()) return 5;
    llm_cold_reference cold_zero, cold_one;
    if (!cold.find_or_admit({ 0, 0 }, source.bundle(), cold_zero).is_ready() ||
        !cold.find_or_admit({ 0, 1 }, source.bundle(), cold_one).is_ready()) return 6;

    llm_expert_transfer_ring ring({ 1U << 20, 2, device, false, force_pageable, 0 });
    if (!ring.initialize(source.bundle()).is_ready()) return 7;
    llm_transfer_lane_reference lane_zero, lane_one;
    if (!ring.reserve(cold, cold_zero, 0, 1, lane_zero).is_ready() ||
        !ring.reserve(cold, cold_one, 1, 1, lane_one).is_ready() ||
        !ring.stage(lane_zero, cold.bundle()).is_ready() ||
        !ring.stage(lane_one, cold.bundle()).is_ready()) return 8;
    if (!ring.transfer_wave(backend.get(), {
            { lane_zero, hot.bundle(), 0 }, { lane_one, hot.bundle(), 1 },
        }).is_ready()) return 9;
    if (!slot_matches(source, hot, 0, 0) || !slot_matches(source, hot, 1, 1)) return 10;

    const auto value = ring.diagnostics();
    if (value.waves != 1 || value.h2d_bytes != value.lane_payload_bytes*2 ||
        cold.diagnostics().current_transfer_refs != 0 || !ring.validate_invariants().is_ready()) return 11;
    if (force_pageable) {
        if (!value.pageable_fallback || value.pinned_or_registered_bytes != 0 ||
            value.async_enqueues != 0 || value.wave_synchronizations != 0 || value.synchronous_copies != 6) return 12;
    } else {
        if (value.pageable_fallback || value.pinned_or_registered_bytes == 0 ||
            value.async_enqueues != 6 || value.wave_synchronizations != 1 ||
            value.peak_in_flight_lanes != 2 || value.synchronous_copies != 0) return 13;
    }
    std::cout << "PHASE5_TRANSFER_RING"
              << "\tmode=" << (force_pageable ? "pageable" : "pinned")
              << "\tlanes=" << value.effective_lanes
              << "\tlane_footprint=" << value.lane_footprint
              << "\tactual_bytes=" << value.actual_bytes
              << "\tpinned_bytes=" << value.pinned_or_registered_bytes
              << "\tasync_enqueues=" << value.async_enqueues
              << "\tsynchronous_copies=" << value.synchronous_copies
              << "\twaves=" << value.waves
              << "\twave_synchronizations=" << value.wave_synchronizations
              << "\tpeak_in_flight_lanes=" << value.peak_in_flight_lanes
              << "\th2d_bytes=" << value.h2d_bytes
              << '\n';
    return 0;
}
