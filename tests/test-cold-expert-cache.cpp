#include "llama-cold-expert-cache.h"
#include "llama-expert-async-io.h"

#include "ggml-cpp.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <future>
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
    llm_cold_cache_config result;
    result.byte_budget = bytes;
    result.minimum_slots = minimum;
    result.routed_layer_count = 1;
    result.total_expert_keys = 4;
    result.initial_slot_generation_for_testing = generation;
    return result;
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
    expect_invalid([] { llm_cold_expert_cache cache(config(1024, 0)); });
    expect_invalid([] {
        auto invalid = config(1024);
        invalid.routed_layer_count = 3;
        llm_cold_expert_cache cache(std::move(invalid));
    });

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
#ifdef __linux__
    GGML_ASSERT(diagnostics.residency_supported);
    GGML_ASSERT(diagnostics.ready_page_count == 0);
#endif

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

void test_runtime_policy_adapters() {
    fixture tensors;
    const uint64_t budget = budget_for_slots(tensors, 2);
    for (const auto policy_name : { LLAMA_EXPERT_CACHE_POLICY_LRU, LLAMA_EXPERT_CACHE_POLICY_LFRU,
            LLAMA_EXPERT_CACHE_POLICY_SLRU, LLAMA_EXPERT_CACHE_POLICY_LFU_AGING }) {
        auto cache_config = config(budget, 2);
        const llama_expert_cache_policy_config public_config = {
            LLAMA_EXPERT_CACHE_POLICY_VERSION_1,
            sizeof(llama_expert_cache_policy_config),
            policy_name,
            LLAMA_EXPERT_CACHE_POLICY_SCOPE_GLOBAL,
            policy_name == LLAMA_EXPERT_CACHE_POLICY_SLRU ? 7500u : 0u,
            LLAMA_EXPERT_CACHE_ADMISSION_ALWAYS,
            0,
            policy_name == LLAMA_EXPERT_CACHE_POLICY_LFU_AGING ? 64u : 0u,
            {},
        };
        GGML_ASSERT(llm_expert_cache_policy_copy_config(&public_config,
            llm_expert_cache_policy_tier::cold, cache_config.cache_policy_config).is_ready());
        llm_cold_expert_cache cache(cache_config);
        GGML_ASSERT(cache.initialize(tensors.bundle()).is_ready());
        llm_cold_reference reference;
        GGML_ASSERT(cache.find_or_admit({ 0, 0 }, tensors.bundle(), reference).is_ready());
        const llm_cold_reference first = reference;
        GGML_ASSERT(cache.find_or_admit({ 0, 1 }, tensors.bundle(), reference).is_ready());
        const auto before_shadow = cache.diagnostics();
        GGML_ASSERT(cache.policy_shadow_hit({ 0, 0 }, first, 3).is_ready());
        const auto after_shadow = cache.diagnostics();
        GGML_ASSERT(after_shadow.hits == before_shadow.hits &&
            after_shadow.misses == before_shadow.misses &&
            after_shadow.admissions == before_shadow.admissions &&
            after_shadow.policy.demands == before_shadow.policy.demands + 1 &&
            after_shadow.policy.hits == before_shadow.policy.hits + 1);
        GGML_ASSERT(cache.policy_shadow_hit({ 0, 3 }, first, 1).error ==
            llm_expert_provider_error::metadata_mismatch &&
            cache.diagnostics().policy.events == after_shadow.policy.events);
        GGML_ASSERT(cache.find_or_admit({ 0, 0 }, tensors.bundle(), reference).is_ready());
        GGML_ASSERT(cache.find_or_admit({ 0, 2 }, tensors.bundle(), reference).is_ready());
        const auto diagnostics = cache.diagnostics();
        GGML_ASSERT(diagnostics.policy.config.policy == policy_name);
        GGML_ASSERT(diagnostics.policy.config.supplied);
        GGML_ASSERT(diagnostics.policy.events > 0);
        GGML_ASSERT(cache.validate_invariants().is_ready());
        GGML_ASSERT(cache.surrender().is_ready());
    }
}

void test_transactional_multi_release_exhaustion() {
    fixture tensors;
    auto cache_config = config(budget_for_slots(tensors, 1), 1);
    cache_config.policy_trace_capacity = 8;
    llm_cold_expert_cache cache(cache_config);
    GGML_ASSERT(cache.initialize(tensors.bundle()).is_ready());
    llm_cold_reference reference;
    GGML_ASSERT(cache.find_or_admit({ 0, 0 }, tensors.bundle(), reference).is_ready());
    GGML_ASSERT(cache.acquire(reference, llm_cold_reference_kind::cpu_execution).is_ready());
    GGML_ASSERT(cache.acquire(reference, llm_cold_reference_kind::cpu_execution).is_ready());
    const llm_cold_reference references[] = { reference, reference };
    GGML_ASSERT(!cache.release_many(references, 2,
        llm_cold_reference_kind::cpu_execution).is_ready());
    const auto diagnostics = cache.diagnostics();
    GGML_ASSERT(diagnostics.current_cpu_execution_refs == 2);
    GGML_ASSERT(diagnostics.slots[reference.slot].cpu_execution_refs == 2);
    GGML_ASSERT(diagnostics.policy.transcript_records == 7);
}

void test_two_phase_publication_and_failure() {
    fixture tensors;
    llm_cold_expert_cache cache(config(budget_for_slots(tensors, 2), 2));
    GGML_ASSERT(cache.initialize(tensors.bundle()).is_ready());
    llm_cold_reference reserved;
    bool hit = true;
    GGML_ASSERT(cache.reserve_or_find({ 0, 2 }, reserved, hit).is_ready());
    GGML_ASSERT(!hit);
    auto diagnostics = cache.diagnostics();
    GGML_ASSERT(diagnostics.slots[reserved.slot].state == llm_cold_slot_state::loading);
    GGML_ASSERT(diagnostics.reservations == 1 && diagnostics.publications == 0);
    GGML_ASSERT(cache.surrender().error == llm_expert_provider_error::busy);
    GGML_ASSERT(cache.acquire(reserved, llm_cold_reference_kind::request).error ==
        llm_expert_provider_error::stale_generation);
    GGML_ASSERT(cache.publish_ready({ 0, 2 }, reserved).is_ready());
    llm_cold_reference found;
    GGML_ASSERT(cache.reserve_or_find({ 0, 2 }, found, hit).is_ready() && hit);
    GGML_ASSERT(found.slot == reserved.slot && found.generation == reserved.generation);

    llm_cold_reference failed;
    GGML_ASSERT(cache.reserve_or_find({ 0, 3 }, failed, hit).is_ready() && !hit);
    GGML_ASSERT(cache.fail_reservation({ 0, 3 }, failed).is_ready());
    GGML_ASSERT(cache.publish_ready({ 0, 3 }, failed).error ==
        llm_expert_provider_error::stale_generation);
    diagnostics = cache.diagnostics();
    GGML_ASSERT(diagnostics.publications == 1 && diagnostics.failed_reservations == 1);
    GGML_ASSERT(cache.cleanup_failed_slots().is_ready());
}

void test_reversed_two_phase_publication() {
    fixture tensors;
    llm_cold_expert_cache cache(config(budget_for_slots(tensors, 2), 2));
    GGML_ASSERT(cache.initialize(tensors.bundle()).is_ready());
    llm_cold_reference first, second;
    bool hit = true;
    GGML_ASSERT(cache.reserve_or_find({ 0, 0 }, first, hit).is_ready() && !hit);
    GGML_ASSERT(cache.reserve_or_find({ 0, 1 }, second, hit).is_ready() && !hit);
    GGML_ASSERT(cache.publish_ready({ 0, 1 }, second).is_ready());
    GGML_ASSERT(!cache.ready(second));
    auto diagnostics = cache.diagnostics();
    GGML_ASSERT(diagnostics.publications == 0 &&
        diagnostics.slots[second.slot].state == llm_cold_slot_state::loading);
    GGML_ASSERT(cache.publish_ready({ 0, 0 }, first).is_ready());
    GGML_ASSERT(cache.ready(first) && cache.ready(second));
    diagnostics = cache.diagnostics();
    GGML_ASSERT(diagnostics.publications == 2 && diagnostics.admissions == 2);
    GGML_ASSERT(cache.acquire(second, llm_cold_reference_kind::request).is_ready());
    GGML_ASSERT(cache.release(second, llm_cold_reference_kind::request).is_ready());
}

struct reversed_read_override : llm_expert_async_read_override {
    int64_t read_at(intptr_t, void * destination, size_t byte_count, uint64_t file_offset,
            int & native_error) noexcept override {
        std::memset(destination, int(file_offset + 1), byte_count);
        native_error = 0;
        return int64_t(byte_count);
    }
};

void test_wait_any_reversed_publication() {
    fixture tensors;
    llm_cold_expert_cache cache(config(budget_for_slots(tensors, 2), 2));
    GGML_ASSERT(cache.initialize(tensors.bundle()).is_ready());
    llm_cold_reference references[2];
    bool hit = true;
    GGML_ASSERT(cache.reserve_or_find({ 0, 0 }, references[0], hit).is_ready() && !hit);
    GGML_ASSERT(cache.reserve_or_find({ 0, 1 }, references[1], hit).is_ready() && !hit);

    reversed_read_override reader;
    llm_expert_async_config async_config = {
        8, 2, 2, 16, 1U << 20, 0, 4096, true, 0, &reader,
    };
    async_config.reverse_queued_requests_for_testing = true;
    llm_expert_async_transport transport(async_config);
    std::array<std::array<uint8_t, 8>, 2> destinations{};
    std::array<llm_expert_storage_read_operation, 2> reads{};
    std::array<llm_expert_async_operation_identity, 2> identities{};
    for (uint32_t index = 0; index < 2; ++index) {
        reads[index].native_handle = 0;
        reads[index].source_size = 16;
        reads[index].file_offset = index*8;
        reads[index].byte_count = destinations[index].size();
        reads[index].segment_count = 1;
        reads[index].segments[0] = { destinations[index].data(), destinations[index].size(),
            reads[index].file_offset, llm_expert_storage_projection::up,
            llm_expert_storage_sidecar::weight };
        identities[index] = { 1, { index, 1 }, 0, { 0, int32_t(index) },
            llm_expert_readiness::host_ready, llm_expert_priority::demand_current_layer };
        GGML_ASSERT(transport.submit_read_plan(identities[index], &reads[index], 1, true) ==
            llm_expert_async_result::ready);
    }
    transport.start_deferred_reads();
    const std::array<llm_expert_request_handle, 2> handles = {
        identities[1].request, identities[0].request,
    };
    llm_expert_request_handle completed;
    llm_expert_async_read_completion completion;
    GGML_ASSERT(transport.wait_any_read(handles.data(), handles.size(), completed, completion) ==
        llm_expert_async_result::ready);
    GGML_ASSERT(completed.slot == 1);
    GGML_ASSERT(transport.release_read(completed) == llm_expert_async_result::ready);
    GGML_ASSERT(cache.publish_ready({ 0, 1 }, references[1]).is_ready());
    GGML_ASSERT(!cache.ready(references[1]));
    GGML_ASSERT(transport.wait_any_read(handles.data(), handles.size(), completed, completion) ==
        llm_expert_async_result::ready);
    GGML_ASSERT(completed.slot == 0);
    GGML_ASSERT(transport.release_read(completed) == llm_expert_async_result::ready);
    GGML_ASSERT(cache.publish_ready({ 0, 0 }, references[0]).is_ready());
    GGML_ASSERT(cache.ready(references[0]) && cache.ready(references[1]));
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

void test_cpu_execution_reference_lifetime() {
    fixture tensors;
    llm_cold_expert_cache cache(config(budget_for_slots(tensors, 1), 1));
    GGML_ASSERT(cache.initialize(tensors.bundle()).is_ready());
    llm_cold_reference executing;
    GGML_ASSERT(cache.find_or_admit({ 0, 0 }, tensors.bundle(), executing).is_ready());
    GGML_ASSERT(cache.acquire(executing, llm_cold_reference_kind::cpu_execution).is_ready());
    auto diagnostics = cache.diagnostics();
    GGML_ASSERT(diagnostics.current_cpu_execution_refs == 1);
    GGML_ASSERT(diagnostics.peak_cpu_execution_refs == 1);

    llm_cold_reference blocked;
    GGML_ASSERT(cache.find_or_admit({ 0, 1 }, tensors.bundle(), blocked).error ==
        llm_expert_provider_error::busy);
    GGML_ASSERT(cache.trim().is_ready());
    GGML_ASSERT(cache.diagnostics().slots[executing.slot].state == llm_cold_slot_state::ready);
    GGML_ASSERT(cache.surrender().error == llm_expert_provider_error::busy);

    GGML_ASSERT(cache.release(executing, llm_cold_reference_kind::cpu_execution).is_ready());
    diagnostics = cache.diagnostics();
    GGML_ASSERT(diagnostics.current_cpu_execution_refs == 0);
    GGML_ASSERT(diagnostics.peak_cpu_execution_refs == 1);
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

void test_speculative_free_or_speculative_admission_and_reclassification() {
    fixture tensors;
    llm_cold_expert_cache cache(config(budget_for_slots(tensors, 2), 2));
    GGML_ASSERT(cache.initialize(tensors.bundle()).is_ready());
    loader_state state;

    llm_cold_reference demand;
    GGML_ASSERT(cache.find_or_admit({ 0, 0 }, tensors.bundle(), demand).is_ready());
    llm_cold_reference speculative;
    GGML_ASSERT(cache.find_or_admit_speculative_with_loader(
        { 0, 1 }, 5, 40, speculative, fill_slot, &state).is_ready());
    auto diagnostics = cache.diagnostics();
    GGML_ASSERT(diagnostics.slots[demand.slot].origin == llm_expert_residency_origin::demand);
    GGML_ASSERT(diagnostics.slots[speculative.slot].origin == llm_expert_residency_origin::speculative);
    GGML_ASSERT(diagnostics.speculative_admissions == 1);

    // Full cache: the upstream guard replaces only the unconsumed speculative
    // entry and preserves the demand-origin slot regardless of LRU order.
    llm_cold_reference replacement;
    GGML_ASSERT(cache.find_or_admit_speculative_with_loader(
        { 0, 2 }, 7, 30, replacement, fill_slot, &state).is_ready());
    GGML_ASSERT(replacement.slot == speculative.slot && replacement.generation > speculative.generation);
    diagnostics = cache.diagnostics();
    GGML_ASSERT(diagnostics.slots[demand.slot].key.expert == 0);
    GGML_ASSERT(diagnostics.speculative_replacements == 1);

    // Exact demand consumes the same generation and permanently removes this
    // slot from speculative-victim eligibility.
    llm_cold_reference consumed;
    GGML_ASSERT(cache.find_or_admit({ 0, 2 }, tensors.bundle(), consumed).is_ready());
    GGML_ASSERT(consumed.slot == replacement.slot && consumed.generation == replacement.generation);
    diagnostics = cache.diagnostics();
    GGML_ASSERT(diagnostics.slots[consumed.slot].origin == llm_expert_residency_origin::demand);
    GGML_ASSERT(diagnostics.slots[consumed.slot].speculative_consumed);
    GGML_ASSERT(diagnostics.speculative_demand_consumptions == 1);

    llm_cold_reference rejected;
    GGML_ASSERT(cache.find_or_admit_speculative_with_loader(
        { 0, 3 }, 9, 20, rejected, fill_slot, &state).error == llm_expert_provider_error::busy);
    diagnostics = cache.diagnostics();
    GGML_ASSERT(diagnostics.speculative_rejections == 1);
    GGML_ASSERT(diagnostics.slots[demand.slot].key.expert == 0);
    GGML_ASSERT(diagnostics.slots[consumed.slot].key.expert == 2);
    GGML_ASSERT(cache.validate_invariants().is_ready());
    GGML_ASSERT(cache.surrender().is_ready());
}

void test_demand_joins_loading_speculative_cold_generation() {
    fixture tensors;
    llm_cold_expert_cache cache(config(budget_for_slots(tensors, 2), 2));
    GGML_ASSERT(cache.initialize(tensors.bundle()).is_ready());
    llm_cold_reference speculative;
    bool hit = false;
    GGML_ASSERT(cache.reserve_or_find_speculative(
        { 0, 1 }, 5, 40, speculative, hit).is_ready());
    GGML_ASSERT(!hit);
    auto waiter = std::async(std::launch::async, [&] {
        return cache.wait_until_ready(speculative);
    });
    llm_cold_reference joined;
    llm_cold_demand_lookup lookup = llm_cold_demand_lookup::reserved;
    GGML_ASSERT(cache.reserve_or_join_demand({ 0, 1 }, joined, lookup).is_ready());
    GGML_ASSERT(lookup == llm_cold_demand_lookup::joined_loading);
    GGML_ASSERT(joined.slot == speculative.slot && joined.generation == speculative.generation);
    GGML_ASSERT(waiter.wait_for(std::chrono::milliseconds(1)) == std::future_status::timeout);
    GGML_ASSERT(cache.publish_ready({ 0, 1 }, speculative).is_ready());
    GGML_ASSERT(waiter.get().is_ready());
    const auto diagnostics = cache.diagnostics();
    GGML_ASSERT(diagnostics.slots[joined.slot].origin == llm_expert_residency_origin::demand);
    GGML_ASSERT(diagnostics.slots[joined.slot].speculative_consumed);
    GGML_ASSERT(diagnostics.speculative_demand_consumptions == 1);
    GGML_ASSERT(cache.surrender().is_ready());
}

void test_lookup_only_demand_does_not_admit_or_evict() {
    fixture tensors;
    llm_cold_expert_cache cache(config(budget_for_slots(tensors, 2), 2));
    GGML_ASSERT(cache.initialize(tensors.bundle()).is_ready());

    llm_cold_reference reference;
    llm_cold_demand_lookup lookup = llm_cold_demand_lookup::ready;
    GGML_ASSERT(cache.lookup_demand({ 0, 3 }, reference, lookup).is_ready());
    GGML_ASSERT(lookup == llm_cold_demand_lookup::missing);
    auto diagnostics = cache.diagnostics();
    GGML_ASSERT(diagnostics.requests == 1 && diagnostics.misses == 1 &&
        diagnostics.reservations == 0 && diagnostics.admissions == 0 && diagnostics.evictions == 0);

    llm_cold_reference loading;
    bool hit = false;
    GGML_ASSERT(cache.reserve_or_find({ 0, 1 }, loading, hit).is_ready() && !hit);
    GGML_ASSERT(cache.lookup_demand({ 0, 1 }, reference, lookup).is_ready());
    GGML_ASSERT(lookup == llm_cold_demand_lookup::joined_loading &&
        reference.slot == loading.slot && reference.generation == loading.generation);
    GGML_ASSERT(cache.publish_ready({ 0, 1 }, loading).is_ready());
    GGML_ASSERT(cache.lookup_demand({ 0, 1 }, reference, lookup).is_ready());
    GGML_ASSERT(lookup == llm_cold_demand_lookup::ready &&
        reference.slot == loading.slot && reference.generation == loading.generation);
    diagnostics = cache.diagnostics();
    GGML_ASSERT(diagnostics.reservations == 1 && diagnostics.admissions == 1 &&
        diagnostics.evictions == 0 && diagnostics.hits == 1);
    GGML_ASSERT(cache.validate_invariants().is_ready());
    GGML_ASSERT(cache.surrender().is_ready());
}

void test_batch_hold_publication_and_transfer() {
    fixture tensors;
    llm_cold_expert_cache cache(config(budget_for_slots(tensors, 1), 1));
    GGML_ASSERT(cache.initialize(tensors.bundle()).is_ready());
    llm_cold_reference reference;
    llm_cold_demand_lookup lookup = llm_cold_demand_lookup::ready;
    GGML_ASSERT(cache.reserve_or_join_demand({ 0, 2 }, reference, lookup).is_ready());
    GGML_ASSERT(lookup == llm_cold_demand_lookup::reserved);
    const uint64_t before_publication = cache.diagnostics().policy.events;
    GGML_ASSERT(cache.publish_ready_and_acquire(
        { 0, 2 }, reference, llm_cold_reference_kind::batch).is_ready());
    auto diagnostics = cache.diagnostics();
    GGML_ASSERT(diagnostics.current_batch_refs == 1 && diagnostics.current_request_refs == 0 &&
        diagnostics.policy.events == before_publication + 1);
    GGML_ASSERT(cache.surrender().error == llm_expert_provider_error::busy);
    GGML_ASSERT(cache.trim().is_ready() && cache.contains_ready({ 0, 2 }));

    const uint64_t before_transfer = diagnostics.policy.events;
    GGML_ASSERT(cache.convert_batch_to_request(reference).is_ready());
    diagnostics = cache.diagnostics();
    GGML_ASSERT(diagnostics.current_batch_refs == 0 && diagnostics.current_request_refs == 1 &&
        diagnostics.policy.events == before_transfer + 1);
    GGML_ASSERT(cache.release(reference, llm_cold_reference_kind::request).is_ready());
    GGML_ASSERT(cache.policy_request_end(true, false).is_ready());
    diagnostics = cache.diagnostics();
    GGML_ASSERT(diagnostics.current_batch_refs == 0 && diagnostics.current_request_refs == 0 &&
        cache.validate_invariants().is_ready());
    GGML_ASSERT(cache.surrender().is_ready());
}

} // namespace

int main() {
    test_runtime_policy_adapters();
    test_transactional_multi_release_exhaustion();
    test_configuration_and_budget_edges();
    test_publication_hits_and_copy_failure();
    test_two_phase_publication_and_failure();
    test_reversed_two_phase_publication();
    test_wait_any_reversed_publication();
    test_lru_references_and_inclusion();
    test_cpu_execution_reference_lifetime();
    test_generation_wrap_and_reinitialize();
    test_loader_publication_failure_cleanup_and_reread();
    test_speculative_free_or_speculative_admission_and_reclassification();
    test_demand_joins_loading_speculative_cold_generation();
    test_lookup_only_demand_does_not_admit_or_evict();
    test_batch_hold_publication_and_transfer();
    std::cout << "cold expert cache tests passed\n";
    return 0;
}
