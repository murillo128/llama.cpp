#include "llama-expert-transfer-ring.h"

#include "ggml-cpp.h"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

struct fixture {
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buffer;
    ggml_tensor * up = nullptr;
    ggml_tensor * gate = nullptr;
    ggml_tensor * down = nullptr;
    int32_t n_expert;

    explicit fixture(int32_t n_expert, bool fill = true) : n_expert(n_expert) {
        ggml_init_params params = { ggml_tensor_overhead()*8, nullptr, true };
        ctx.reset(ggml_init(params));
        GGML_ASSERT(ctx);
        up = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 8, 16, n_expert);
        gate = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 8, 16, n_expert);
        down = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 16, 8, n_expert);
        buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), ggml_backend_cpu_buffer_type()));
        GGML_ASSERT(buffer);
        if (fill) {
            uint8_t pattern = 0x20;
            for (auto * tensor : { up, gate, down }) {
                for (int32_t expert = 0; expert < n_expert; ++expert) {
                    std::memset(static_cast<uint8_t *>(tensor->data) + size_t(expert)*tensor->nb[2],
                        pattern + expert, tensor->nb[2]);
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
    llm_cold_expert_cache cache({ 1U << 20, 2, 1, uint32_t(source.n_expert), 0 });
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
        GGML_ASSERT(std::memcmp(
            static_cast<const uint8_t *>(pair.first->data) + size_t(expert)*span,
            static_cast<const uint8_t *>(pair.second->data) + size_t(slot)*span,
            span) == 0);
    }
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

    auto cold = make_cold(source);
    llm_cold_reference cold_zero;
    GGML_ASSERT(cold.find_or_admit({ 0, 0 }, source.bundle(), cold_zero).is_ready());
    llm_expert_transfer_ring exhausted(ring_config(footprint*2, 2, std::numeric_limits<uint64_t>::max()));
    GGML_ASSERT(exhausted.initialize(source.bundle()).is_ready());
    llm_transfer_lane_reference lane;
    GGML_ASSERT(exhausted.reserve(cold, cold_zero, 0, 1, lane).error ==
        llm_expert_provider_error::generation_exhausted);
}

} // namespace

int main() {
    test_budget_fallback_and_wave();
    test_failures_cleanup_and_busy_surrender();
    test_budget_and_generation_rejection();
    std::cout << "expert transfer ring tests passed\n";
    return 0;
}
