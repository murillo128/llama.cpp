#include "llama-cold-expert-cache.h"

#include "ggml-cpp.h"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

struct fixture {
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buffer;
    ggml_tensor * up = nullptr;
    ggml_tensor * up_bias = nullptr;
    ggml_tensor * up_scale = nullptr;
    ggml_tensor * gate = nullptr;
    ggml_tensor * down = nullptr;
    ggml_tensor * ids = nullptr;
    int32_t n_expert = 4;

    fixture() {
        ggml_init_params params = { ggml_tensor_overhead()*12, nullptr, true };
        ctx.reset(ggml_init(params));
        GGML_ASSERT(ctx);
        up = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 8, 16, n_expert);
        up_bias = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 16, n_expert);
        up_scale = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 1, n_expert);
        gate = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 8, 16, n_expert);
        down = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 16, 8, n_expert);
        ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 2, 1);
        buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), ggml_backend_cpu_buffer_type()));
        GGML_ASSERT(buffer);
        uint8_t pattern = 0x10;
        for (auto * tensor : { up, up_bias, up_scale, gate, down }) {
            const int axis = tensor->ne[2] == n_expert ? 2 : 1;
            for (int32_t expert = 0; expert < n_expert; ++expert) {
                std::memset(static_cast<uint8_t *>(tensor->data) + size_t(expert)*tensor->nb[axis],
                    pattern + expert, tensor->nb[axis]);
            }
            pattern += 0x10;
        }
    }

    llm_expert_bundle_descriptor bundle(int32_t layer = 0) const {
        return {
            layer,
            n_expert,
            llm_expert_projection_descriptor::from(up, up_bias, up_scale),
            llm_expert_projection_descriptor::from(gate, nullptr, nullptr),
            {},
            llm_expert_projection_descriptor::from(down, nullptr, nullptr),
        };
    }
};

llm_cold_cache_config config(uint64_t bytes, uint32_t minimum = 1, uint64_t generation = 0) {
    return { bytes, minimum, 1, 4, generation };
}

template<class F> void expect_invalid(F fn) {
    bool rejected = false;
    try { fn(); } catch (const std::invalid_argument &) { rejected = true; }
    GGML_ASSERT(rejected);
}

uint64_t discover_budget(const fixture & tensors, uint32_t minimum = 1) {
    llm_cold_expert_cache cache(config(1U << 20, minimum));
    GGML_ASSERT(cache.initialize(tensors.bundle()).is_ready());
    return cache.diagnostics().actual_bytes;
}

uint64_t budget_for_slots(const fixture & tensors, uint32_t slots) {
    uint64_t low = 1;
    uint64_t high = discover_budget(tensors, 4);
    while (low < high) {
        const uint64_t middle = low + (high - low)/2;
        llm_cold_expert_cache probe(config(middle, slots));
        const auto result = probe.initialize(tensors.bundle());
        if (result.is_ready() && probe.diagnostics().effective_slots >= slots) {
            high = middle;
        } else {
            low = middle + 1;
        }
    }
    return low;
}

void assert_tensor_slot(
        const ggml_tensor * source,
        const ggml_tensor * target,
        int32_t expert,
        uint32_t slot,
        bool weight) {
    if (source == nullptr) {
        GGML_ASSERT(target == nullptr);
        return;
    }
    const int axis = weight ? 2 : 1;
    const size_t span = source->nb[axis];
    GGML_ASSERT(target->nb[axis] == span);
    GGML_ASSERT(std::memcmp(
        static_cast<const uint8_t *>(source->data) + size_t(expert)*span,
        static_cast<const uint8_t *>(target->data) + size_t(slot)*span,
        span) == 0);
}

void test_configuration_and_budget_edges() {
    fixture tensors;
    expect_invalid([] { llm_cold_expert_cache cache(config(0)); });
    expect_invalid([] { llm_cold_expert_cache cache({ 1024, 0, 1, 4, 0 }); });
    expect_invalid([] { llm_cold_expert_cache cache({ 1024, 1, 3, 4, 0 }); });

    const uint64_t full_budget = discover_budget(tensors, 4);
    llm_cold_expert_cache exact(config(full_budget, 4));
    GGML_ASSERT(exact.initialize(tensors.bundle()).is_ready());
    const auto diagnostics = exact.diagnostics();
    GGML_ASSERT(diagnostics.effective_slots == 4);
    GGML_ASSERT(diagnostics.actual_bytes <= diagnostics.requested_bytes);
    GGML_ASSERT(diagnostics.unused_budget_bytes == diagnostics.requested_bytes - diagnostics.actual_bytes);
    GGML_ASSERT(diagnostics.bundle_payload_bytes > 0);
    GGML_ASSERT(diagnostics.alignment > 0);
    GGML_ASSERT(diagnostics.pageable);

    llm_cold_expert_cache insufficient(config(full_budget - 1, 4));
    GGML_ASSERT(insufficient.initialize(tensors.bundle()).error ==
        llm_expert_provider_error::unsupported_configuration);
}

void test_publication_hits_and_copy_failure() {
    fixture tensors;
    llm_cold_expert_cache cache(config(budget_for_slots(tensors, 2), 2));
    GGML_ASSERT(cache.initialize(tensors.bundle()).is_ready());
    llm_cold_reference first;
    GGML_ASSERT(cache.find_or_admit({ 0, 2 }, tensors.bundle(), first).is_ready());
    assert_tensor_slot(tensors.up, cache.bundle().up.weight, 2, first.slot, true);
    assert_tensor_slot(tensors.up_bias, cache.bundle().up.bias, 2, first.slot, false);
    assert_tensor_slot(tensors.up_scale, cache.bundle().up.scale, 2, first.slot, false);
    llm_cold_reference hit;
    GGML_ASSERT(cache.find_or_admit({ 0, 2 }, tensors.bundle(), hit).is_ready());
    GGML_ASSERT(hit.slot == first.slot && hit.generation == first.generation);

    llm_cold_reference failed;
    GGML_ASSERT(cache.find_or_admit({ 0, 3 }, tensors.bundle(), failed, 1).error ==
        llm_expert_provider_error::copy_failed);
    auto diagnostics = cache.diagnostics();
    GGML_ASSERT(diagnostics.hits == 1 && diagnostics.misses == 2);
    GGML_ASSERT(diagnostics.admissions == 1 && diagnostics.failed_copies == 1);
    GGML_ASSERT(diagnostics.slots[failed.slot].state != llm_cold_slot_state::ready);
    GGML_ASSERT(cache.cleanup_failed_slots().is_ready());
    diagnostics = cache.diagnostics();
    GGML_ASSERT(diagnostics.failed_cleanups == 1);
}

void test_lru_references_and_inclusion() {
    fixture tensors;
    llm_cold_expert_cache cache(config(budget_for_slots(tensors, 2), 2));
    GGML_ASSERT(cache.initialize(tensors.bundle()).is_ready());
    GGML_ASSERT(cache.diagnostics().effective_slots == 2);
    llm_cold_reference zero, one, two;
    GGML_ASSERT(cache.find_or_admit({ 0, 0 }, tensors.bundle(), zero).is_ready());
    GGML_ASSERT(cache.find_or_admit({ 0, 1 }, tensors.bundle(), one).is_ready());
    GGML_ASSERT(cache.acquire(zero, llm_cold_reference_kind::hot).is_ready());
    GGML_ASSERT(cache.find_or_admit({ 0, 2 }, tensors.bundle(), two).is_ready());
    GGML_ASSERT(two.slot == one.slot);
    GGML_ASSERT(cache.acquire(one, llm_cold_reference_kind::request).error ==
        llm_expert_provider_error::stale_generation);
    GGML_ASSERT(cache.validate_invariants({ { { 0, 0 }, zero.slot, zero.generation } }).is_ready());
    GGML_ASSERT(cache.trim().is_ready());
    GGML_ASSERT(cache.diagnostics().slots[zero.slot].state == llm_cold_slot_state::ready);
    GGML_ASSERT(cache.surrender().error == llm_expert_provider_error::busy);
    GGML_ASSERT(cache.release(zero, llm_cold_reference_kind::hot).is_ready());
    GGML_ASSERT(cache.trim().is_ready());
    GGML_ASSERT(cache.surrender().is_ready());
}

void test_generation_wrap_and_reinitialize() {
    fixture tensors;
    llm_cold_expert_cache exhausted(config(discover_budget(tensors), 1,
        std::numeric_limits<uint64_t>::max()));
    GGML_ASSERT(exhausted.initialize(tensors.bundle()).is_ready());
    llm_cold_reference reference;
    GGML_ASSERT(exhausted.find_or_admit({ 0, 0 }, tensors.bundle(), reference).error ==
        llm_expert_provider_error::generation_exhausted);

    llm_cold_expert_cache cache(config(discover_budget(tensors), 1));
    GGML_ASSERT(cache.initialize(tensors.bundle()).is_ready());
    GGML_ASSERT(cache.surrender().is_ready());
    GGML_ASSERT(cache.initialize(tensors.bundle()).is_ready());
    GGML_ASSERT(cache.validate_invariants().is_ready());
}

struct loader_state {
    size_t calls = 0;
    llm_expert_provider_error failure = llm_expert_provider_error::none;
};

llm_expert_provider_result fill_slot(void * user_data, llm_expert_key key,
        const llm_expert_bundle_descriptor & destination, uint32_t slot) noexcept {
    auto & state = *static_cast<loader_state *>(user_data);
    state.calls++;
    if (state.failure != llm_expert_provider_error::none) {
        return llm_expert_provider_result::failure(state.failure);
    }
    uint8_t pattern = uint8_t(0x40 + key.expert);
    for (auto * tensor : { destination.up.weight, destination.up.bias, destination.up.scale,
            destination.gate.weight, destination.down.weight }) {
        const int axis = tensor->ne[2] == destination.n_expert ? 2 : 1;
        std::memset(static_cast<uint8_t *>(tensor->data) + size_t(slot)*tensor->nb[axis],
            pattern++, tensor->nb[axis]);
    }
    return llm_expert_provider_result::success();
}

void test_loader_publication_failure_cleanup_and_reread() {
    fixture tensors;
    llm_cold_expert_cache cache(config(budget_for_slots(tensors, 1), 1));
    GGML_ASSERT(cache.initialize(tensors.bundle()).is_ready());
    loader_state state;
    llm_cold_reference first;
    GGML_ASSERT(cache.find_or_admit_with_loader({ 0, 0 }, first, fill_slot, &state).is_ready());
    GGML_ASSERT(state.calls == 1);
    llm_cold_reference hit;
    GGML_ASSERT(cache.find_or_admit_with_loader({ 0, 0 }, hit, fill_slot, &state).is_ready());
    GGML_ASSERT(state.calls == 1 && hit.slot == first.slot && hit.generation == first.generation);

    state.failure = llm_expert_provider_error::cancelled;
    llm_cold_reference cancelled;
    GGML_ASSERT(cache.find_or_admit_with_loader({ 0, 1 }, cancelled, fill_slot, &state).error ==
        llm_expert_provider_error::cancelled);
    auto diagnostics = cache.diagnostics();
    GGML_ASSERT(diagnostics.slots[cancelled.slot].state == llm_cold_slot_state::failed);
    GGML_ASSERT(diagnostics.admissions == 1 && diagnostics.source_copy_bundles == 0);
    GGML_ASSERT(cache.cleanup_failed_slots().is_ready());

    state.failure = llm_expert_provider_error::none;
    llm_cold_reference retried;
    GGML_ASSERT(cache.find_or_admit_with_loader({ 0, 1 }, retried, fill_slot, &state).is_ready());
    GGML_ASSERT(retried.generation > cancelled.generation);
    llm_cold_reference reread;
    GGML_ASSERT(cache.find_or_admit_with_loader({ 0, 0 }, reread, fill_slot, &state).is_ready());
    GGML_ASSERT(state.calls == 4 && reread.generation > first.generation);
    diagnostics = cache.diagnostics();
    GGML_ASSERT(diagnostics.evictions >= 2 && diagnostics.source_copy_bytes == 0);
}

} // namespace

int main() {
    test_configuration_and_budget_edges();
    test_publication_hits_and_copy_failure();
    test_lru_references_and_inclusion();
    test_generation_wrap_and_reinitialize();
    test_loader_publication_failure_cleanup_and_reread();
    std::cout << "cold expert cache tests passed\n";
    return 0;
}
