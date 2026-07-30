#include "llama-expert-weight-provider.h"
#include "llama-context.h"
#include "llama-model.h"

#include "ggml-alloc.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

std::atomic<uint64_t> allocation_count { 0 };

void * operator new(std::size_t size) {
    allocation_count.fetch_add(1, std::memory_order_relaxed);
    if (void * memory = std::malloc(size)) {
        return memory;
    }
    throw std::bad_alloc();
}

void * operator new[](std::size_t size) {
    return ::operator new(size);
}

void operator delete(void * memory) noexcept {
    std::free(memory);
}

void operator delete[](void * memory) noexcept {
    std::free(memory);
}

void operator delete(void * memory, std::size_t) noexcept {
    std::free(memory);
}

void operator delete[](void * memory, std::size_t) noexcept {
    std::free(memory);
}

namespace {

struct tensor_fixture {
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buffer;
    ggml_tensor * up = nullptr;
    ggml_tensor * up_bias = nullptr;
    ggml_tensor * up_scale = nullptr;
    ggml_tensor * gate = nullptr;
    ggml_tensor * gate_bias = nullptr;
    ggml_tensor * gate_scale = nullptr;
    ggml_tensor * down = nullptr;
    ggml_tensor * down_bias = nullptr;
    ggml_tensor * down_scale = nullptr;
    ggml_tensor * ids = nullptr;
    int64_t n_expert = 4;
    int64_t n_expert_used = 2;
    int64_t n_tokens = 1;

    tensor_fixture(
            int64_t n_in = 8,
            int64_t n_hidden = 16,
            int64_t n_expert = 4,
            int64_t n_tokens = 1,
            int64_t n_expert_used = 2) :
        n_expert(n_expert), n_expert_used(n_expert_used), n_tokens(n_tokens) {
        ggml_init_params params = {
            /*.mem_size   =*/ ggml_tensor_overhead()*16,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ctx.reset(ggml_init(params));
        GGML_ASSERT(ctx);
        up = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, n_in, n_hidden, n_expert);
        up_bias = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, n_hidden, n_expert);
        up_scale = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 1, n_expert);
        gate = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, n_in, n_hidden, n_expert);
        gate_bias = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, n_hidden, n_expert);
        gate_scale = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 1, n_expert);
        down = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, n_hidden, n_in, n_expert);
        down_bias = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, n_in, n_expert);
        down_scale = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 1, n_expert);
        ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, n_expert_used, n_tokens);
        buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), ggml_backend_cpu_buffer_type()));
        GGML_ASSERT(buffer);
        uint8_t pattern = 0x10;
        for (auto * tensor : { up, up_bias, up_scale, gate, gate_bias, gate_scale, down, down_bias, down_scale }) {
            const int axis = tensor->ne[2] == n_expert ? 2 : 1;
            for (int64_t expert = 0; expert < n_expert; ++expert) {
                std::memset(static_cast<uint8_t *>(tensor->data) + expert*tensor->nb[axis],
                    pattern + expert, tensor->nb[axis]);
            }
            pattern += 0x10;
        }
    }

    llm_expert_bundle_descriptor bundle(int32_t layer) const {
        return {
            layer,
            int32_t(n_expert),
            llm_expert_projection_descriptor::from(up, up_bias, up_scale),
            llm_expert_projection_descriptor::from(gate, gate_bias, gate_scale),
            {},
            llm_expert_projection_descriptor::from(down, down_bias, down_scale),
        };
    }

    llm_expert_selection selection(int32_t layer) const {
        return { layer, int32_t(n_expert), int32_t(n_expert_used), n_tokens, ids };
    }
};

llm_hot_cache_config test_config(
        uint32_t capacity = 2,
        uint32_t routed_layers = 1,
        uint32_t total_keys = 4,
        uint32_t n_expert_used = 2,
        uint64_t initial_generation = 0) {
    return {
        capacity,
        n_expert_used,
        routed_layers,
        total_keys,
        ggml_backend_cpu_buffer_type(),
        true,
        initial_generation,
    };
}

llm_hot_cache_config cold_test_config(uint32_t capacity = 2) {
    static const bool loaded = [] { ggml_backend_load_all(); return true; }();
    (void) loaded;
    auto result = test_config(capacity, 1, 4, 2);
    result.cold_mode = true;
    result.cold_cache_bytes = 1U << 20;
    result.transfer_ring_bytes = 1U << 20;
    result.target_device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    result.force_pageable_transfer_for_testing = true;
    GGML_ASSERT(result.target_device);
    return result;
}

template<typename F>
void expect_invalid(F && fn) {
    bool rejected = false;
    try {
        fn();
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    GGML_ASSERT(rejected);
}

llm_expert_graph_binding initialize_hot_binding(
        llm_expert_weight_provider & provider,
        const tensor_fixture & tensors,
        int32_t layer = 0) {
    llm_expert_graph_binding binding;
    GGML_ASSERT(provider.bind(tensors.bundle(layer), tensors.selection(layer), binding).is_ready());
    GGML_ASSERT(binding.bootstrap);
    GGML_ASSERT(provider.initialize_after_reserve().is_ready());
    GGML_ASSERT(provider.bind(tensors.bundle(layer), tensors.selection(layer), binding).is_ready());
    GGML_ASSERT(!binding.bootstrap);
    return binding;
}

int source_expert_axis(const ggml_tensor * tensor, int64_t n_expert, bool weight) {
    if (weight) {
        GGML_ASSERT(tensor->ne[2] == n_expert);
        return 2;
    }
    int result = -1;
    for (int axis = 0; axis < ggml_n_dims(tensor); ++axis) {
        if (tensor->ne[axis] == n_expert) {
            GGML_ASSERT(result == -1);
            result = axis;
        }
    }
    GGML_ASSERT(result >= 0);
    return result;
}

void assert_tensor_slot_matches(
        const ggml_tensor * source,
        const ggml_tensor * target,
        int64_t n_expert,
        int32_t expert,
        int32_t slot,
        bool weight) {
    GGML_ASSERT((source == nullptr) == (target == nullptr));
    if (source == nullptr) {
        return;
    }
    const int axis = source_expert_axis(source, n_expert, weight);
    const size_t span = source->nb[axis];
    GGML_ASSERT(target->nb[axis] == span);
    const auto * source_bytes = static_cast<const uint8_t *>(source->data) + size_t(expert)*span;
    const auto * target_bytes = static_cast<const uint8_t *>(target->data) + size_t(slot)*span;
    GGML_ASSERT(std::memcmp(source_bytes, target_bytes, span) == 0);
}

void assert_projection_slot_matches(
        const llm_expert_projection_descriptor & source,
        const llm_expert_projection_descriptor & target,
        int64_t n_expert,
        int32_t expert,
        int32_t slot) {
    assert_tensor_slot_matches(source.weight, target.weight, n_expert, expert, slot, true);
    assert_tensor_slot_matches(source.bias, target.bias, n_expert, expert, slot, false);
    assert_tensor_slot_matches(source.scale, target.scale, n_expert, expert, slot, false);
}

void assert_bundle_slot_matches(
        const tensor_fixture & source,
        const llm_expert_graph_binding & target,
        int32_t expert,
        int32_t slot) {
    const auto bundle = source.bundle(target.layer);
    assert_projection_slot_matches(bundle.up, target.up, source.n_expert, expert, slot);
    assert_projection_slot_matches(bundle.gate, target.gate, source.n_expert, expert, slot);
    assert_projection_slot_matches(bundle.gate_up, target.gate_up, source.n_expert, expert, slot);
    assert_projection_slot_matches(bundle.down, target.down, source.n_expert, expert, slot);
}

int32_t find_slot(const llm_hot_cache_diagnostics & diagnostics, int32_t expert) {
    for (size_t slot = 0; slot < diagnostics.slots.size(); ++slot) {
        if (diagnostics.slots[slot].expert == expert) {
            return int32_t(slot);
        }
    }
    return -1;
}

struct eval_callback_counts {
    uint64_t asks = 0;
    uint64_t observations = 0;
};

struct route_capture {
    std::vector<int32_t> layers;
    std::vector<int32_t> ids;
    std::vector<float> weights;
};

bool capture_routes(const llama_route_observation * observation, void * user_data) {
    auto * capture = static_cast<route_capture *>(user_data);
    capture->layers.push_back(observation->layer);
    const size_t count = size_t(observation->n_tokens)*observation->n_expert_used;
    capture->ids.insert(capture->ids.end(), observation->selected_experts,
        observation->selected_experts + count);
    capture->weights.insert(capture->weights.end(), observation->weights,
        observation->weights + count);
    return true;
}

bool count_execution_id_callbacks(ggml_tensor * tensor, bool ask, void * user_data) {
    auto * counts = static_cast<eval_callback_counts *>(user_data);
    if (ask) {
        counts->asks++;
        return std::strncmp(tensor->name, "expert_execution_ids-", 21) == 0;
    }
    counts->observations++;
    return true;
}

void test_configuration_matrix() {
    expect_invalid([] { llm_create_hot_cache_expert_weight_provider(test_config(0)); });
    expect_invalid([] { llm_create_hot_cache_expert_weight_provider(test_config(1)); });
    expect_invalid([] { llm_create_hot_cache_expert_weight_provider(test_config(5)); });

    auto non_cuda = test_config(2);
    non_cuda.allow_non_cuda_target_for_testing = false;
    expect_invalid([&] { llm_create_hot_cache_expert_weight_provider(non_cuda); });

    GGML_ASSERT(llm_create_hot_cache_expert_weight_provider(test_config(2)) != nullptr);
    GGML_ASSERT(llm_create_hot_cache_expert_weight_provider(test_config(4)) != nullptr);

    auto hot_with_cold_budget = test_config(2);
    hot_with_cold_budget.cold_cache_bytes = 1;
    expect_invalid([&] { llm_create_hot_cache_expert_weight_provider(hot_with_cold_budget); });
    auto cold_without_budget = cold_test_config();
    cold_without_budget.transfer_ring_bytes = 0;
    expect_invalid([&] { llm_create_cold_cache_expert_weight_provider(cold_without_budget); });
}

void test_context_extent_matrix_and_prepare_revalidation() {
    auto exact_top_k = llm_create_hot_cache_expert_weight_provider(test_config(2));
    GGML_ASSERT(exact_top_k->validate_context_extent(64, 1).is_ready());
    auto diagnostics = exact_top_k->hot_cache_diagnostics();
    GGML_ASSERT(diagnostics.n_expert == 4);
    GGML_ASSERT(diagnostics.n_expert_used == 2);
    GGML_ASSERT(diagnostics.last_context_extent == 1);
    GGML_ASSERT(diagnostics.conservative_required_capacity == 2);
    GGML_ASSERT(diagnostics.effective_capacity == 0);

    const auto unsafe_exact = exact_top_k->validate_context_extent(64, 2);
    GGML_ASSERT(unsafe_exact.error == llm_expert_provider_error::unsupported_configuration);
    diagnostics = exact_top_k->hot_cache_diagnostics();
    GGML_ASSERT(diagnostics.last_context_extent == 2);
    GGML_ASSERT(diagnostics.conservative_required_capacity == 4);
    GGML_ASSERT(diagnostics.context_validations == 2);
    GGML_ASSERT(diagnostics.context_rejections == 1);
    GGML_ASSERT(diagnostics.effective_capacity == 0);

    auto capacity_three = llm_create_hot_cache_expert_weight_provider(test_config(3));
    GGML_ASSERT(capacity_three->validate_context_extent(64, 1).is_ready());
    GGML_ASSERT(capacity_three->validate_context_extent(64, 2).error ==
        llm_expert_provider_error::unsupported_configuration);

    auto all_experts = llm_create_hot_cache_expert_weight_provider(test_config(4));
    GGML_ASSERT(all_experts->validate_context_extent(64, 64).is_ready());
    diagnostics = all_experts->hot_cache_diagnostics();
    GGML_ASSERT(diagnostics.last_context_extent == 64);
    GGML_ASSERT(diagnostics.conservative_required_capacity == 4);

    tensor_fixture two_tokens(8, 16, 4, 2);
    llm_expert_graph_binding binding;
    GGML_ASSERT(exact_top_k->bind(two_tokens.bundle(0), two_tokens.selection(0), binding).is_ready());
    GGML_ASSERT(exact_top_k->initialize_after_reserve().is_ready());
    llm_expert_execution_plan unsafe_plan;
    GGML_ASSERT(exact_top_k->prepare({ binding }, unsafe_plan).error ==
        llm_expert_provider_error::unsupported_configuration);
    GGML_ASSERT(unsafe_plan.handle_count() == 0);

    GGML_ASSERT(capacity_three->bind(two_tokens.bundle(0), two_tokens.selection(0), binding).is_ready());
    GGML_ASSERT(capacity_three->initialize_after_reserve().is_ready());
    GGML_ASSERT(capacity_three->bind(two_tokens.bundle(0), two_tokens.selection(0), binding).is_ready());
    llm_expert_execution_plan capacity_three_plan;
    GGML_ASSERT(capacity_three->prepare({ binding }, capacity_three_plan).error ==
        llm_expert_provider_error::unsupported_configuration);
    GGML_ASSERT(capacity_three_plan.handle_count() == 0);

    GGML_ASSERT(all_experts->bind(two_tokens.bundle(0), two_tokens.selection(0), binding).is_ready());
    GGML_ASSERT(all_experts->initialize_after_reserve().is_ready());
    GGML_ASSERT(all_experts->bind(two_tokens.bundle(0), two_tokens.selection(0), binding).is_ready());
    llm_expert_execution_plan safe_extent_plan;
    GGML_ASSERT(all_experts->prepare({ binding }, safe_extent_plan).is_ready());
    safe_extent_plan.reset();
}

void test_pool_lifetime_trim_surrender_and_epoch() {
    tensor_fixture tensors;
    auto provider = llm_create_hot_cache_expert_weight_provider(test_config());

    llm_expert_graph_binding bootstrap;
    GGML_ASSERT(provider->bind(tensors.bundle(0), tensors.selection(0), bootstrap).is_ready());
    GGML_ASSERT(bootstrap.bootstrap);
    GGML_ASSERT(!bootstrap.generation_lease);
    GGML_ASSERT(bootstrap.up.weight == tensors.up);
    GGML_ASSERT(provider->needs_post_reserve_initialization());

    const auto init_result = provider->initialize_after_reserve();
    if (!init_result.is_ready()) {
        std::cerr << "hot-cache initialization failed: status=" << int(init_result.status)
                  << " error=" << int(init_result.error) << '\n';
    }
    GGML_ASSERT(init_result.is_ready());
    GGML_ASSERT(!provider->needs_post_reserve_initialization());
    const auto initialized = provider->hot_cache_diagnostics();
    GGML_ASSERT(initialized.requested_capacity == 2);
    GGML_ASSERT(initialized.effective_capacity == 2);
    GGML_ASSERT(initialized.pool_bytes > 0);
    GGML_ASSERT(initialized.generation == 1);
    GGML_ASSERT(!initialized.slot_tensor_addresses.empty());
    GGML_ASSERT(std::all_of(initialized.slot_tensor_addresses.begin(), initialized.slot_tensor_addresses.end(),
        [](uintptr_t address) { return address != 0; }));

    llm_expert_graph_binding hot;
    GGML_ASSERT(provider->bind(tensors.bundle(0), tensors.selection(0), hot).is_ready());
    GGML_ASSERT(!hot.bootstrap);
    GGML_ASSERT(hot.generation_lease);
    GGML_ASSERT(hot.up.weight != tensors.up);
    GGML_ASSERT(hot.up.weight->ne[2] == 2);
    const auto stable = provider->hot_cache_diagnostics();
    GGML_ASSERT(stable.slot_tensor_addresses == initialized.slot_tensor_addresses);
    GGML_ASSERT(provider->trim().is_ready());
    GGML_ASSERT(provider->hot_cache_diagnostics().slot_tensor_addresses == initialized.slot_tensor_addresses);

    const auto busy = provider->surrender();
    GGML_ASSERT(busy.error == llm_expert_provider_error::busy);
    GGML_ASSERT(provider->hot_cache_diagnostics().effective_capacity == 2);
    hot = {};

    const uint64_t epoch_before_surrender = provider->graph_epoch();
    GGML_ASSERT(provider->surrender().is_ready());
    GGML_ASSERT(provider->graph_epoch() > epoch_before_surrender);
    GGML_ASSERT(provider->needs_post_reserve_initialization());
    GGML_ASSERT(provider->hot_cache_diagnostics().effective_capacity == 0);

    llm_expert_graph_binding second_bootstrap;
    GGML_ASSERT(provider->bind(tensors.bundle(0), tensors.selection(0), second_bootstrap).is_ready());
    GGML_ASSERT(second_bootstrap.bootstrap);
    GGML_ASSERT(provider->initialize_after_reserve().is_ready());
    const auto reinitialized = provider->hot_cache_diagnostics();
    GGML_ASSERT(reinitialized.generation == 2);
    GGML_ASSERT(reinitialized.graph_epoch > initialized.graph_epoch);

    const auto stats = provider->get_stats();
    GGML_ASSERT(stats.allocations == 2);
    GGML_ASSERT(stats.pool_generations == 2);
    GGML_ASSERT(stats.bootstrap_bindings == 2);
    GGML_ASSERT(stats.hot_bindings == 1);
    GGML_ASSERT(stats.trims == 1);
    GGML_ASSERT(stats.surrender_busy == 1);
    GGML_ASSERT(stats.surrender_successes == 1);
}

void test_layout_host_and_partial_initialization_rejection() {
    tensor_fixture first;
    tensor_fixture incompatible(9, 16, 4);
    auto provider = llm_create_hot_cache_expert_weight_provider(test_config(2, 2, 8));
    llm_expert_graph_binding binding;
    GGML_ASSERT(provider->bind(first.bundle(0), first.selection(0), binding).is_ready());
    GGML_ASSERT(provider->bind(incompatible.bundle(1), incompatible.selection(1), binding).error ==
        llm_expert_provider_error::invalid_descriptor);
    GGML_ASSERT(provider->initialize_after_reserve().error == llm_expert_provider_error::initialization_failed);
    GGML_ASSERT(provider->hot_cache_diagnostics().effective_capacity == 0);

    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead()*4,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr unallocated_ctx(ggml_init(params));
    GGML_ASSERT(unallocated_ctx);
    ggml_tensor * up = ggml_new_tensor_3d(unallocated_ctx.get(), GGML_TYPE_F32, 8, 16, 4);
    ggml_tensor * gate = ggml_new_tensor_3d(unallocated_ctx.get(), GGML_TYPE_F32, 8, 16, 4);
    ggml_tensor * down = ggml_new_tensor_3d(unallocated_ctx.get(), GGML_TYPE_F32, 16, 8, 4);
    const llm_expert_bundle_descriptor non_host = {
        0, 4,
        llm_expert_projection_descriptor::from(up, nullptr, nullptr),
        llm_expert_projection_descriptor::from(gate, nullptr, nullptr),
        {},
        llm_expert_projection_descriptor::from(down, nullptr, nullptr),
    };
    const llm_expert_selection selection = { 0, 4, 2, 1, first.ids };
    auto host_checked = llm_create_hot_cache_expert_weight_provider(test_config());
    GGML_ASSERT(host_checked->bind(non_host, selection, binding).error ==
        llm_expert_provider_error::unsupported_configuration);
}

void test_allocation_failure_is_recoverable_and_empty_prepare_is_safe() {
    tensor_fixture tensors;
    llm_expert_provider_faults faults;
    faults.fail_pool_allocation = true;
    auto provider = llm_create_hot_cache_expert_weight_provider(test_config(), faults);
    llm_expert_graph_binding binding;
    GGML_ASSERT(provider->bind(tensors.bundle(0), tensors.selection(0), binding).is_ready());
    const auto failed = provider->initialize_after_reserve();
    GGML_ASSERT(failed.status == llm_expert_provider_status::allocation_failed);
    GGML_ASSERT(provider->needs_post_reserve_initialization());
    GGML_ASSERT(provider->hot_cache_diagnostics().effective_capacity == 0);

    llm_expert_execution_plan empty;
    GGML_ASSERT(provider->prepare({}, empty).is_ready());
    GGML_ASSERT(empty.handle_count() == 0);
}

void test_directory_hit_eviction_generation_and_copy() {
    tensor_fixture tensors(8, 16, 4, 1, 1);
    auto provider = llm_create_hot_cache_expert_weight_provider(test_config(1, 1, 4, 1));
    const auto binding = initialize_hot_binding(*provider, tensors);
    llm_expert_execution_plan plan;
    GGML_ASSERT(provider->prepare({ binding }, plan).is_ready());

    const int32_t first_ids[] = { 0 };
    int32_t execution_ids[] = { -1 };
    GGML_ASSERT(provider->remap_checkpoint(binding, first_ids, 1, execution_ids).is_ready());
    GGML_ASSERT(execution_ids[0] == 0);
    auto diagnostics = provider->hot_cache_diagnostics();
    GGML_ASSERT(diagnostics.slots[0].expert == 0);
    GGML_ASSERT(diagnostics.slots[0].generation == 1);
    GGML_ASSERT(diagnostics.slots[0].state == llm_hot_cache_diagnostics::slot::pinned);
    assert_bundle_slot_matches(tensors, binding, 0, 0);

    GGML_ASSERT(provider->remap_checkpoint(binding, first_ids, 1, execution_ids).is_ready());
    diagnostics = provider->hot_cache_diagnostics();
    GGML_ASSERT(diagnostics.slots[0].generation == 1);
    GGML_ASSERT(diagnostics.hits == 1);
    GGML_ASSERT(diagnostics.misses == 1);
    plan.reset();

    GGML_ASSERT(provider->prepare({ binding }, plan).is_ready());
    const int32_t second_ids[] = { 1 };
    GGML_ASSERT(provider->remap_checkpoint(binding, second_ids, 1, execution_ids).is_ready());
    diagnostics = provider->hot_cache_diagnostics();
    GGML_ASSERT(diagnostics.slots[0].expert == 1);
    GGML_ASSERT(diagnostics.slots[0].generation == 2);
    GGML_ASSERT(diagnostics.evictions == 1);
    GGML_ASSERT(provider->validate_slot_generation(0, 1).error ==
        llm_expert_provider_error::stale_generation);
    GGML_ASSERT(provider->validate_slot_generation(0, 2).is_ready());
    assert_bundle_slot_matches(tensors, binding, 1, 0);
    plan.reset();
}

void test_directory_multi_token_atomic_dedup_and_no_allocation() {
    tensor_fixture tensors(8, 16, 4, 2, 2);
    auto provider = llm_create_hot_cache_expert_weight_provider(test_config(4, 1, 4, 2));
    const auto binding = initialize_hot_binding(*provider, tensors);
    llm_expert_execution_plan plan;
    GGML_ASSERT(provider->prepare({ binding }, plan).is_ready());

    const int32_t logical_ids[] = { 0, 1, 2, 3 };
    int32_t execution_ids[] = { -1, -1, -1, -1 };
    const uint64_t allocations_before = allocation_count.load(std::memory_order_relaxed);
    GGML_ASSERT(provider->remap_checkpoint(binding, logical_ids, 4, execution_ids).is_ready());
    const uint64_t allocations_after = allocation_count.load(std::memory_order_relaxed);
    GGML_ASSERT(allocations_after == allocations_before);
    for (int32_t index = 0; index < 4; ++index) {
        GGML_ASSERT(execution_ids[index] == index);
        assert_bundle_slot_matches(tensors, binding, index, index);
    }

    const int32_t duplicate_ids[] = { 0, 0, 1, 1 };
    const int32_t expected_slots[] = { 0, 0, 1, 1 };
    const uint64_t duplicate_allocations_before = allocation_count.load(std::memory_order_relaxed);
    GGML_ASSERT(provider->remap_checkpoint(binding, duplicate_ids, 4, execution_ids).is_ready());
    GGML_ASSERT(allocation_count.load(std::memory_order_relaxed) == duplicate_allocations_before);
    GGML_ASSERT(std::memcmp(execution_ids, expected_slots, sizeof(expected_slots)) == 0);
    const auto diagnostics = provider->hot_cache_diagnostics();
    GGML_ASSERT(diagnostics.remap_checkpoints == 2);
    GGML_ASSERT(diagnostics.logical_ids == 8);
    GGML_ASSERT(diagnostics.unique_ids == 6);
    GGML_ASSERT(diagnostics.hits == 2);
    GGML_ASSERT(diagnostics.misses == 4);
    GGML_ASSERT(diagnostics.current_pins == 2);
    GGML_ASSERT(diagnostics.remap_dynamic_allocations == 0);
    plan.reset();
}

void test_cold_provider_inclusive_promotion_and_hits() {
    tensor_fixture tensors(8, 16, 4, 1, 2);
    auto provider = llm_create_cold_cache_expert_weight_provider(cold_test_config());
    const auto binding = initialize_hot_binding(*provider, tensors);
    llm_expert_execution_plan plan;
    GGML_ASSERT(provider->prepare({ binding }, plan).is_ready());

    const int32_t first_ids[] = { 0, 1 };
    int32_t execution_ids[] = { -1, -1 };
    const uint64_t allocations_before = allocation_count.load(std::memory_order_relaxed);
    GGML_ASSERT(provider->remap_checkpoint(binding, first_ids, 2, execution_ids).is_ready());
    GGML_ASSERT(allocation_count.load(std::memory_order_relaxed) == allocations_before);
    auto diagnostics = provider->hot_cache_diagnostics();
    GGML_ASSERT(diagnostics.misses == 2 && diagnostics.cold_misses == 2);
    GGML_ASSERT(diagnostics.cold_admissions == 2 && diagnostics.cold_source_copy_bundles == 2);
    GGML_ASSERT(diagnostics.ring_waves == 1 && diagnostics.ring_synchronous_copies == 18);
    GGML_ASSERT(diagnostics.ring_async_enqueues == 0 && diagnostics.ring_wave_synchronizations == 0);
    GGML_ASSERT(diagnostics.cold_current_hot_refs == 2);
    GGML_ASSERT(diagnostics.cold_current_transfer_refs == 0);
    for (int32_t index = 0; index < 2; ++index) {
        GGML_ASSERT(diagnostics.slots[execution_ids[index]].has_cold_backing);
        assert_bundle_slot_matches(tensors, binding, first_ids[index], execution_ids[index]);
    }

    const uint64_t source_bytes = diagnostics.cold_source_copy_bytes;
    GGML_ASSERT(provider->remap_checkpoint(binding, first_ids, 2, execution_ids).is_ready());
    diagnostics = provider->hot_cache_diagnostics();
    GGML_ASSERT(diagnostics.hits == 2 && diagnostics.cold_source_copy_bytes == source_bytes);
    GGML_ASSERT(diagnostics.cold_current_request_refs == 0);
    plan.reset();

    GGML_ASSERT(provider->trim().is_ready());
    diagnostics = provider->hot_cache_diagnostics();
    GGML_ASSERT(diagnostics.cold_current_hot_refs == 0);
    GGML_ASSERT(provider->surrender().error == llm_expert_provider_error::busy);
}

void test_cold_provider_copy_failure_cleanup_and_retry() {
    tensor_fixture tensors(8, 16, 4, 1, 2);
    llm_expert_provider_faults faults;
    faults.fail_copy_after_tensors = 1;
    auto provider = llm_create_cold_cache_expert_weight_provider(cold_test_config(), faults);
    const auto binding = initialize_hot_binding(*provider, tensors);
    llm_expert_execution_plan plan;
    GGML_ASSERT(provider->prepare({ binding }, plan).is_ready());
    const int32_t logical_ids[] = { 0, 1 };
    int32_t execution_ids[] = { -1, -1 };
    GGML_ASSERT(provider->remap_checkpoint(binding, logical_ids, 2, execution_ids).error ==
        llm_expert_provider_error::copy_failed);
    auto diagnostics = provider->hot_cache_diagnostics();
    GGML_ASSERT(diagnostics.cold_failed_copies == 1);
    GGML_ASSERT(diagnostics.cold_current_hot_refs == 0);
    GGML_ASSERT(diagnostics.cold_current_transfer_refs == 0);
    plan.reset();
    GGML_ASSERT(provider->cleanup_failed_slots().is_ready());

    GGML_ASSERT(provider->prepare({ binding }, plan).is_ready());
    GGML_ASSERT(provider->remap_checkpoint(binding, logical_ids, 2, execution_ids).is_ready());
    diagnostics = provider->hot_cache_diagnostics();
    GGML_ASSERT(diagnostics.cold_current_hot_refs == 2);
    GGML_ASSERT(diagnostics.cold_current_transfer_refs == 0);
    plan.reset();
}

void test_directory_composite_layer_expert_keys() {
    tensor_fixture first(8, 16, 4, 1, 1);
    tensor_fixture second(8, 16, 4, 1, 1);
    auto provider = llm_create_hot_cache_expert_weight_provider(test_config(2, 2, 8, 1));
    llm_expert_graph_binding first_binding;
    llm_expert_graph_binding second_binding;
    GGML_ASSERT(provider->bind(first.bundle(0), first.selection(0), first_binding).is_ready());
    GGML_ASSERT(provider->bind(second.bundle(1), second.selection(1), second_binding).is_ready());
    GGML_ASSERT(provider->initialize_after_reserve().is_ready());
    GGML_ASSERT(provider->bind(first.bundle(0), first.selection(0), first_binding).is_ready());
    GGML_ASSERT(provider->bind(second.bundle(1), second.selection(1), second_binding).is_ready());

    llm_expert_execution_plan plan;
    GGML_ASSERT(provider->prepare({ first_binding, second_binding }, plan).is_ready());
    const int32_t logical_ids[] = { 0 };
    int32_t first_execution[] = { -1 };
    int32_t second_execution[] = { -1 };
    GGML_ASSERT(provider->remap_checkpoint(first_binding, logical_ids, 1, first_execution).is_ready());
    GGML_ASSERT(provider->remap_checkpoint(second_binding, logical_ids, 1, second_execution).is_ready());
    GGML_ASSERT(first_execution[0] != second_execution[0]);
    const auto diagnostics = provider->hot_cache_diagnostics();
    GGML_ASSERT(diagnostics.slots[first_execution[0]].layer == 0);
    GGML_ASSERT(diagnostics.slots[second_execution[0]].layer == 1);
    assert_bundle_slot_matches(first, first_binding, 0, first_execution[0]);
    assert_bundle_slot_matches(second, second_binding, 0, second_execution[0]);
    plan.reset();
}

void test_directory_lru_pin_exclusion_and_request_exclusivity() {
    tensor_fixture tensors;
    auto provider = llm_create_hot_cache_expert_weight_provider(test_config());
    const auto binding = initialize_hot_binding(*provider, tensors);
    llm_expert_execution_plan plan;
    llm_expert_execution_plan competing;
    GGML_ASSERT(provider->prepare({ binding }, plan).is_ready());
    GGML_ASSERT(provider->prepare({ binding }, competing).error == llm_expert_provider_error::busy);

    const int32_t first_ids[] = { 0, 1 };
    int32_t execution_ids[] = { -1, -1 };
    GGML_ASSERT(provider->remap_checkpoint(binding, first_ids, 2, execution_ids).is_ready());
    const auto first = provider->hot_cache_diagnostics();
    const int32_t slot_zero = find_slot(first, 0);
    const int32_t slot_one = find_slot(first, 1);
    GGML_ASSERT(slot_zero >= 0 && slot_one >= 0 && slot_zero != slot_one);

    const int32_t second_ids[] = { 0, 2 };
    GGML_ASSERT(provider->remap_checkpoint(binding, second_ids, 2, execution_ids).is_ready());
    const auto second = provider->hot_cache_diagnostics();
    GGML_ASSERT(find_slot(second, 0) == slot_zero);
    GGML_ASSERT(find_slot(second, 2) == slot_one);
    GGML_ASSERT(second.slots[slot_zero].generation == first.slots[slot_zero].generation);
    plan.reset();

    GGML_ASSERT(provider->prepare({ binding }, plan).is_ready());
    const int32_t third_ids[] = { 2, 3 };
    GGML_ASSERT(provider->remap_checkpoint(binding, third_ids, 2, execution_ids).is_ready());
    const auto third = provider->hot_cache_diagnostics();
    GGML_ASSERT(find_slot(third, 2) == slot_one);
    GGML_ASSERT(find_slot(third, 3) == slot_zero);
    GGML_ASSERT(third.evictions == 2);
    GGML_ASSERT(third.exclusive_busy_failures == 1);
    plan.reset();
}

void test_directory_copy_failure_cleanup_and_reuse() {
    tensor_fixture tensors;
    llm_expert_provider_faults faults;
    faults.fail_copy_after_tensors = 4;
    auto provider = llm_create_hot_cache_expert_weight_provider(test_config(), faults);
    const auto binding = initialize_hot_binding(*provider, tensors);
    llm_expert_execution_plan plan;
    GGML_ASSERT(provider->prepare({ binding }, plan).is_ready());
    const int32_t logical_ids[] = { 0, 1 };
    int32_t execution_ids[] = { -1, -1 };
    GGML_ASSERT(provider->remap_checkpoint(binding, logical_ids, 2, execution_ids).error ==
        llm_expert_provider_error::copy_failed);
    auto diagnostics = provider->hot_cache_diagnostics();
    GGML_ASSERT(diagnostics.copy_failures == 1);
    GGML_ASSERT(diagnostics.current_pins == 0);
    GGML_ASSERT(std::all_of(diagnostics.slots.begin(), diagnostics.slots.end(), [](const auto & slot) {
        return slot.state == llm_hot_cache_diagnostics::slot::failed;
    }));
    GGML_ASSERT(provider->cleanup_failed_slots().error == llm_expert_provider_error::busy);
    plan.reset();
    GGML_ASSERT(provider->cleanup_failed_slots().is_ready());
    diagnostics = provider->hot_cache_diagnostics();
    GGML_ASSERT(diagnostics.failed_cleanups == 2);
    GGML_ASSERT(std::all_of(diagnostics.slots.begin(), diagnostics.slots.end(), [](const auto & slot) {
        return slot.state == llm_hot_cache_diagnostics::slot::free && slot.expert == -1;
    }));

    GGML_ASSERT(provider->prepare({ binding }, plan).is_ready());
    GGML_ASSERT(provider->remap_checkpoint(binding, logical_ids, 2, execution_ids).is_ready());
    GGML_ASSERT(execution_ids[0] != execution_ids[1]);
    plan.reset();
}

void test_directory_generation_exhaustion_trim_and_invalid_id() {
    tensor_fixture tensors(8, 16, 4, 1, 1);
    auto provider = llm_create_hot_cache_expert_weight_provider(
        test_config(1, 1, 4, 1, UINT64_MAX - 1));
    const auto binding = initialize_hot_binding(*provider, tensors);
    llm_expert_execution_plan plan;
    GGML_ASSERT(provider->prepare({ binding }, plan).is_ready());
    int32_t execution_ids[] = { -1 };
    const int32_t first_ids[] = { 0 };
    GGML_ASSERT(provider->remap_checkpoint(binding, first_ids, 1, execution_ids).is_ready());
    const auto before = provider->hot_cache_diagnostics();
    GGML_ASSERT(before.slots[0].generation == UINT64_MAX);
    plan.reset();

    GGML_ASSERT(provider->prepare({ binding }, plan).is_ready());
    const int32_t second_ids[] = { 1 };
    GGML_ASSERT(provider->remap_checkpoint(binding, second_ids, 1, execution_ids).error ==
        llm_expert_provider_error::generation_exhausted);
    auto after = provider->hot_cache_diagnostics();
    GGML_ASSERT(after.slots[0].expert == before.slots[0].expert);
    GGML_ASSERT(after.slots[0].generation == before.slots[0].generation);
    GGML_ASSERT(after.slots[0].state == llm_hot_cache_diagnostics::slot::ready);
    plan.reset();

    GGML_ASSERT(provider->prepare({ binding }, plan).is_ready());
    const int32_t invalid_ids[] = { 4 };
    GGML_ASSERT(provider->remap_checkpoint(binding, invalid_ids, 1, execution_ids).error ==
        llm_expert_provider_error::invalid_key);
    after = provider->hot_cache_diagnostics();
    GGML_ASSERT(after.slots[0].expert == 0 && after.slots[0].generation == UINT64_MAX);
    plan.reset();

    GGML_ASSERT(provider->prepare({ binding }, plan).is_ready());
    GGML_ASSERT(provider->remap_checkpoint(binding, first_ids, 1, execution_ids).is_ready());
    GGML_ASSERT(provider->trim().is_ready());
    GGML_ASSERT(provider->hot_cache_diagnostics().slots[0].state ==
        llm_hot_cache_diagnostics::slot::pinned);
    plan.reset();
    GGML_ASSERT(provider->trim().is_ready());
    after = provider->hot_cache_diagnostics();
    GGML_ASSERT(after.slots[0].state == llm_hot_cache_diagnostics::slot::free);
    GGML_ASSERT(after.slots[0].expert == -1);
    GGML_ASSERT(provider->validate_slot_generation(0, UINT64_MAX).error ==
        llm_expert_provider_error::stale_generation);
}

void test_cuda_model_pool_smoke(const char * model_path) {
    const llama_model_tensor_buft_override overrides[] = {
        { "ffn_(gate|up|down)_exps\\.weight", ggml_backend_cpu_buffer_type() },
        { nullptr, nullptr },
    };
    llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = -1;
    params.tensor_buft_overrides = overrides;
    params.expert_weights_mode = LLAMA_EXPERT_WEIGHTS_MODE_HOT_CACHE;
    params.expert_hot_cache_capacity = 2;
    llama_model * model = llama_model_load_from_file(model_path, params);
    GGML_ASSERT(model != nullptr);

    llama_context_params context_params = llama_context_default_params();
    eval_callback_counts callback_counts;
    context_params.n_ctx = 64;
    context_params.n_batch = 64;
    context_params.n_ubatch = 2;
    context_params.cb_eval = count_execution_id_callbacks;
    context_params.cb_eval_user_data = &callback_counts;
    llama_context * rejected = llama_init_from_model(model, context_params);
    GGML_ASSERT(rejected == nullptr);

    auto * provider = model->expert_weight_provider();
    GGML_ASSERT(provider != nullptr);
    auto diagnostics = provider->hot_cache_diagnostics();
    GGML_ASSERT(diagnostics.effective_capacity == 0);
    GGML_ASSERT(diagnostics.last_context_extent == 2);
    GGML_ASSERT(diagnostics.conservative_required_capacity == 4);
    GGML_ASSERT(diagnostics.context_rejections == 1);

    context_params.n_ubatch = 1;
    llama_context * context = llama_init_from_model(model, context_params);
    GGML_ASSERT(context != nullptr);

    diagnostics = provider->hot_cache_diagnostics();
    GGML_ASSERT(diagnostics.effective_capacity == 2);
    GGML_ASSERT(diagnostics.pool_bytes > 0);
    GGML_ASSERT(!diagnostics.slot_tensor_addresses.empty());
    GGML_ASSERT(diagnostics.last_context_extent == 1);
    GGML_ASSERT(diagnostics.conservative_required_capacity == 2);
    GGML_ASSERT(provider->surrender().error == llm_expert_provider_error::busy);

    context_params.n_ubatch = 2;
    rejected = llama_init_from_model(model, context_params);
    GGML_ASSERT(rejected == nullptr);
    GGML_ASSERT(provider->hot_cache_diagnostics().effective_capacity == 2);

    context_params.n_ubatch = 1;
    llama_context * second_context = llama_init_from_model(model, context_params);
    GGML_ASSERT(second_context != nullptr);

    llama_token token = 1;
    GGML_ASSERT(llama_decode(context, llama_batch_get_one(&token, 1)) == 0);
    GGML_ASSERT(llama_decode(second_context, llama_batch_get_one(&token, 1)) == -3);
    const float * first_logits = llama_get_logits_ith(context, -1);
    GGML_ASSERT(first_logits != nullptr);
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    GGML_ASSERT(std::all_of(first_logits, first_logits + n_vocab, [](float value) {
        return std::isfinite(value);
    }));
    diagnostics = provider->hot_cache_diagnostics();
    GGML_ASSERT(diagnostics.remap_checkpoints == 7);
    GGML_ASSERT(diagnostics.synchronization_checkpoints == 7);
    GGML_ASSERT(diagnostics.misses == 14);
    GGML_ASSERT(diagnostics.current_pins == 0);
    GGML_ASSERT(callback_counts.asks > 7);
    GGML_ASSERT(callback_counts.observations == 7);

    GGML_ASSERT(llama_decode(second_context, llama_batch_get_one(&token, 1)) == 0);
    const float * second_logits = llama_get_logits_ith(second_context, -1);
    GGML_ASSERT(second_logits != nullptr);
    GGML_ASSERT(std::equal(first_logits, first_logits + n_vocab, second_logits));
    diagnostics = provider->hot_cache_diagnostics();
    GGML_ASSERT(diagnostics.remap_checkpoints == 14);
    GGML_ASSERT(diagnostics.exclusive_busy_failures == 1);
    GGML_ASSERT(diagnostics.current_pins == 0);
    GGML_ASSERT(callback_counts.observations == 14);

    llama_context * cancelled_context = llama_init_from_model(model, context_params);
    GGML_ASSERT(cancelled_context != nullptr);
    llama_set_abort_callback(cancelled_context, [](void *) { return true; }, nullptr);
    GGML_ASSERT(llama_decode(cancelled_context, llama_batch_get_one(&token, 1)) == 2);
    llama_synchronize(cancelled_context);
    GGML_ASSERT(provider->hot_cache_diagnostics().current_pins == 0);
    llama_free(cancelled_context);

    llama_free(second_context);
    llama_free(context);
    GGML_ASSERT(provider->surrender().is_ready());
    llama_model_free(model);
}

void test_cuda_model_cross_epoch_hits(const char * model_path) {
    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = 64;
    context_params.n_batch = 64;
    context_params.n_ubatch = 1;
    llama_token token = 1;

    llama_model_params disabled_params = llama_model_default_params();
    disabled_params.n_gpu_layers = -1;
    llama_model * disabled_model = llama_model_load_from_file(model_path, disabled_params);
    GGML_ASSERT(disabled_model != nullptr);
    llama_context * disabled_context = llama_init_from_model(disabled_model, context_params);
    GGML_ASSERT(disabled_context != nullptr);
    route_capture disabled_routes;
    GGML_ASSERT(llama_set_route_observer(disabled_context, capture_routes, &disabled_routes) ==
        LLAMA_ROUTE_OBSERVER_STATUS_OK);
    GGML_ASSERT(llama_route_observer_begin(disabled_context, 1, LLAMA_ROUTE_PHASE_DECODE) ==
        LLAMA_ROUTE_OBSERVER_STATUS_OK);
    GGML_ASSERT(llama_decode(disabled_context, llama_batch_get_one(&token, 1)) == 0);
    const float * disabled_logits = llama_get_logits_ith(disabled_context, -1);
    GGML_ASSERT(disabled_logits != nullptr);
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(disabled_model));
    const std::vector<float> reference_logits(disabled_logits, disabled_logits + n_vocab);
    const auto disabled_graph = disabled_context->expert_graph_diagnostics();
    GGML_ASSERT(disabled_graph.binding_count == 0);
    llama_free(disabled_context);
    llama_model_free(disabled_model);

    const llama_model_tensor_buft_override overrides[] = {
        { "ffn_(gate|up|down)_exps\\.weight", ggml_backend_cpu_buffer_type() },
        { nullptr, nullptr },
    };
    llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = -1;
    params.tensor_buft_overrides = overrides;
    params.expert_weights_mode = LLAMA_EXPERT_WEIGHTS_MODE_HOT_CACHE;
    params.expert_hot_cache_capacity = 56;
    llama_model * model = llama_model_load_from_file(model_path, params);
    GGML_ASSERT(model != nullptr);

    llama_context * first = llama_init_from_model(model, context_params);
    llama_context * second = llama_init_from_model(model, context_params);
    GGML_ASSERT(first != nullptr && second != nullptr);
    route_capture hot_routes;
    GGML_ASSERT(llama_set_route_observer(first, capture_routes, &hot_routes) ==
        LLAMA_ROUTE_OBSERVER_STATUS_OK);
    GGML_ASSERT(llama_route_observer_begin(first, 1, LLAMA_ROUTE_PHASE_DECODE) ==
        LLAMA_ROUTE_OBSERVER_STATUS_OK);

    GGML_ASSERT(llama_decode(first, llama_batch_get_one(&token, 1)) == 0);
    const float * first_logits = llama_get_logits_ith(first, -1);
    GGML_ASSERT(first_logits != nullptr);
    GGML_ASSERT(std::equal(first_logits, first_logits + n_vocab, reference_logits.begin()));
    GGML_ASSERT(hot_routes.layers == disabled_routes.layers);
    GGML_ASSERT(hot_routes.ids == disabled_routes.ids);
    GGML_ASSERT(hot_routes.weights == disabled_routes.weights);
    const auto hot_graph = first->expert_graph_diagnostics();
    GGML_ASSERT(hot_graph.binding_count == 7);
    GGML_ASSERT(hot_graph.node_count == disabled_graph.node_count + 7);
    GGML_ASSERT(hot_graph.operation_hash != disabled_graph.operation_hash);
    auto * provider = model->expert_weight_provider();
    const auto cold = provider->hot_cache_diagnostics();
    GGML_ASSERT(cold.remap_checkpoints == 7);
    GGML_ASSERT(cold.misses == 14 && cold.hits == 0);
    GGML_ASSERT(cold.h2d_bytes > 0);
    GGML_ASSERT(cold.last_remap_layer == hot_routes.layers.back());
    GGML_ASSERT(cold.last_logical_ids.size() == 2 && cold.last_execution_ids.size() == 2);
    GGML_ASSERT(std::equal(cold.last_logical_ids.begin(), cold.last_logical_ids.end(), hot_routes.ids.end() - 2));
    GGML_ASSERT(std::all_of(cold.last_execution_ids.begin(), cold.last_execution_ids.end(), [](int32_t slot) {
        return slot >= 12 && slot < 14;
    }));

    GGML_ASSERT(llama_decode(second, llama_batch_get_one(&token, 1)) == 0);
    const float * second_logits = llama_get_logits_ith(second, -1);
    GGML_ASSERT(second_logits != nullptr);
    GGML_ASSERT(std::equal(first_logits, first_logits + n_vocab, second_logits));
    const auto warm = provider->hot_cache_diagnostics();
    GGML_ASSERT(warm.remap_checkpoints == 14);
    GGML_ASSERT(warm.misses == cold.misses);
    GGML_ASSERT(warm.hits == cold.hits + 14);
    GGML_ASSERT(warm.h2d_bytes == cold.h2d_bytes);
    GGML_ASSERT(warm.h2d_time_us == cold.h2d_time_us);
    GGML_ASSERT(warm.slot_tensor_addresses == cold.slot_tensor_addresses);
    GGML_ASSERT(warm.current_pins == 0);
    for (size_t slot = 0; slot < cold.slots.size(); ++slot) {
        GGML_ASSERT(warm.slots[slot].generation == cold.slots[slot].generation);
    }

    GGML_ASSERT(llama_decode(second, llama_batch_get_one(&token, 1)) == 0);
    GGML_ASSERT(llama_decode(second, llama_batch_get_one(&token, 1)) == 0);
    const float * reused_logits = llama_get_logits_ith(second, -1);
    GGML_ASSERT(reused_logits != nullptr);
    GGML_ASSERT(std::all_of(reused_logits, reused_logits + n_vocab, [](float value) {
        return std::isfinite(value);
    }));
    GGML_ASSERT(second->expert_graph_diagnostics().graphs_reused > 0);
    GGML_ASSERT(provider->hot_cache_diagnostics().current_pins == 0);

    llama_free(second);
    llama_free(first);
    GGML_ASSERT(provider->surrender().is_ready());
    llama_model_free(model);
}

void test_cuda_model_multi_token_all_expert_capacity(const char * model_path) {
    const llama_model_tensor_buft_override overrides[] = {
        { "ffn_(gate|up|down)_exps\\.weight", ggml_backend_cpu_buffer_type() },
        { nullptr, nullptr },
    };
    llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = -1;
    params.tensor_buft_overrides = overrides;
    params.expert_weights_mode = LLAMA_EXPERT_WEIGHTS_MODE_HOT_CACHE;
    params.expert_hot_cache_capacity = 8;
    llama_model * model = llama_model_load_from_file(model_path, params);
    GGML_ASSERT(model != nullptr);

    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = 64;
    context_params.n_batch = 64;
    context_params.n_ubatch = 2;
    llama_context * context = llama_init_from_model(model, context_params);
    GGML_ASSERT(context != nullptr);
    llama_token tokens[] = { 1, 1 };
    GGML_ASSERT(llama_decode(context, llama_batch_get_one(tokens, 2)) == 0);
    const float * logits = llama_get_logits_ith(context, -1);
    GGML_ASSERT(logits != nullptr);
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    GGML_ASSERT(std::all_of(logits, logits + n_vocab, [](float value) { return std::isfinite(value); }));
    auto * provider = model->expert_weight_provider();
    const auto diagnostics = provider->hot_cache_diagnostics();
    GGML_ASSERT(diagnostics.last_context_extent == 2);
    GGML_ASSERT(diagnostics.conservative_required_capacity == 4);
    GGML_ASSERT(diagnostics.remap_checkpoints == 7);
    GGML_ASSERT(diagnostics.logical_ids == 28);
    GGML_ASSERT(diagnostics.current_pins == 0);

    llama_free(context);
    GGML_ASSERT(provider->surrender().is_ready());
    llama_model_free(model);
}

void test_cuda_directory_copy() {
    ggml_backend_dev_t gpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    GGML_ASSERT(gpu != nullptr);
    tensor_fixture tensors;
    auto config = test_config();
    config.target_buffer_type = ggml_backend_dev_buffer_type(gpu);
    config.allow_non_cuda_target_for_testing = false;
    auto provider = llm_create_hot_cache_expert_weight_provider(config);
    const auto binding = initialize_hot_binding(*provider, tensors);
    llm_expert_execution_plan plan;
    GGML_ASSERT(provider->prepare({ binding }, plan).is_ready());
    const int32_t logical_ids[] = { 1, 3 };
    int32_t execution_ids[] = { -1, -1 };
    GGML_ASSERT(provider->remap_checkpoint(binding, logical_ids, 2, execution_ids).is_ready());

    const auto assert_cuda_tensor = [&](const ggml_tensor * source, const ggml_tensor * target,
                                        int32_t expert, int32_t slot, bool weight) {
        GGML_ASSERT((source == nullptr) == (target == nullptr));
        if (source == nullptr) {
            return;
        }
        const int axis = source_expert_axis(source, tensors.n_expert, weight);
        const size_t span = source->nb[axis];
        std::vector<uint8_t> actual(span);
        ggml_backend_tensor_get(target, actual.data(), size_t(slot)*span, span);
        const auto * expected = static_cast<const uint8_t *>(source->data) + size_t(expert)*span;
        GGML_ASSERT(std::memcmp(actual.data(), expected, span) == 0);
    };
    const auto assert_cuda_projection = [&](const auto & source, const auto & target,
                                             int32_t expert, int32_t slot) {
        assert_cuda_tensor(source.weight, target.weight, expert, slot, true);
        assert_cuda_tensor(source.bias, target.bias, expert, slot, false);
        assert_cuda_tensor(source.scale, target.scale, expert, slot, false);
    };
    const auto source = tensors.bundle(0);
    for (int32_t index = 0; index < 2; ++index) {
        assert_cuda_projection(source.up, binding.up, logical_ids[index], execution_ids[index]);
        assert_cuda_projection(source.gate, binding.gate, logical_ids[index], execution_ids[index]);
        assert_cuda_projection(source.down, binding.down, logical_ids[index], execution_ids[index]);
    }
    const auto diagnostics = provider->hot_cache_diagnostics();
    GGML_ASSERT(diagnostics.misses == 2);
    GGML_ASSERT(diagnostics.admissions == 2);
    GGML_ASSERT(diagnostics.h2d_bytes > 0);
    plan.reset();
}

} // namespace

int main(int argc, char ** argv) {
    test_configuration_matrix();
    test_context_extent_matrix_and_prepare_revalidation();
    test_pool_lifetime_trim_surrender_and_epoch();
    test_layout_host_and_partial_initialization_rejection();
    test_allocation_failure_is_recoverable_and_empty_prepare_is_safe();
    test_directory_hit_eviction_generation_and_copy();
    test_directory_multi_token_atomic_dedup_and_no_allocation();
    test_cold_provider_inclusive_promotion_and_hits();
    test_cold_provider_copy_failure_cleanup_and_retry();
    test_directory_composite_layer_expert_keys();
    test_directory_lru_pin_exclusion_and_request_exclusivity();
    test_directory_copy_failure_cleanup_and_reuse();
    test_directory_generation_exhaustion_trim_and_invalid_id();
    if (argc == 2) {
        ggml_backend_load_all();
        test_cuda_directory_copy();
        test_cuda_model_pool_smoke(argv[1]);
        test_cuda_model_cross_epoch_hits(argv[1]);
        test_cuda_model_multi_token_all_expert_capacity(argv[1]);
        llama_backend_free();
    } else if (argc != 1) {
        std::cerr << "usage: test-hot-expert-cache [MODEL]\n";
        return 2;
    }
    return 0;
}
