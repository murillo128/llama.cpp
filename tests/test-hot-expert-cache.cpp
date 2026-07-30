#include "llama-expert-weight-provider.h"
#include "llama-model.h"

#include "ggml-alloc.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct tensor_fixture {
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buffer;
    ggml_tensor * up = nullptr;
    ggml_tensor * gate = nullptr;
    ggml_tensor * down = nullptr;
    ggml_tensor * ids = nullptr;
    int64_t n_tokens = 1;

    tensor_fixture(
            int64_t n_in = 8,
            int64_t n_hidden = 16,
            int64_t n_expert = 4,
            int64_t n_tokens = 1) : n_tokens(n_tokens) {
        ggml_init_params params = {
            /*.mem_size   =*/ ggml_tensor_overhead()*8,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ctx.reset(ggml_init(params));
        GGML_ASSERT(ctx);
        up = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, n_in, n_hidden, n_expert);
        gate = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, n_in, n_hidden, n_expert);
        down = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, n_hidden, n_in, n_expert);
        ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 2, n_tokens);
        buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), ggml_backend_cpu_buffer_type()));
        GGML_ASSERT(buffer);
    }

    llm_expert_bundle_descriptor bundle(int32_t layer) const {
        return {
            layer,
            4,
            llm_expert_projection_descriptor::from(up, nullptr, nullptr),
            llm_expert_projection_descriptor::from(gate, nullptr, nullptr),
            {},
            llm_expert_projection_descriptor::from(down, nullptr, nullptr),
        };
    }

    llm_expert_selection selection(int32_t layer) const {
        return { layer, 4, 2, n_tokens, ids };
    }
};

llm_hot_cache_config test_config(
        uint32_t capacity = 2,
        uint32_t routed_layers = 1,
        uint32_t total_keys = 4) {
    return {
        capacity,
        2,
        routed_layers,
        total_keys,
        ggml_backend_cpu_buffer_type(),
        true,
    };
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

void test_configuration_matrix() {
    expect_invalid([] { llm_create_hot_cache_expert_weight_provider(test_config(0)); });
    expect_invalid([] { llm_create_hot_cache_expert_weight_provider(test_config(1)); });
    expect_invalid([] { llm_create_hot_cache_expert_weight_provider(test_config(5)); });

    auto non_cuda = test_config(2);
    non_cuda.allow_non_cuda_target_for_testing = false;
    expect_invalid([&] { llm_create_hot_cache_expert_weight_provider(non_cuda); });

    GGML_ASSERT(llm_create_hot_cache_expert_weight_provider(test_config(2)) != nullptr);
    GGML_ASSERT(llm_create_hot_cache_expert_weight_provider(test_config(4)) != nullptr);
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

    GGML_ASSERT(all_experts->bind(two_tokens.bundle(0), two_tokens.selection(0), binding).is_ready());
    GGML_ASSERT(all_experts->initialize_after_reserve().is_ready());
    llm_expert_execution_plan safe_extent_plan;
    GGML_ASSERT(all_experts->prepare({ binding }, safe_extent_plan).error ==
        llm_expert_provider_error::preparation_failed);
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
    context_params.n_ctx = 64;
    context_params.n_batch = 64;
    context_params.n_ubatch = 2;
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

    llama_free(second_context);
    llama_free(context);
    GGML_ASSERT(provider->surrender().is_ready());
    llama_model_free(model);
}

} // namespace

int main(int argc, char ** argv) {
    test_configuration_matrix();
    test_context_extent_matrix_and_prepare_revalidation();
    test_pool_lifetime_trim_surrender_and_epoch();
    test_layout_host_and_partial_initialization_rejection();
    test_allocation_failure_is_recoverable_and_empty_prepare_is_safe();
    if (argc == 2) {
        ggml_backend_load_all();
        test_cuda_model_pool_smoke(argv[1]);
        llama_backend_free();
    } else if (argc != 1) {
        std::cerr << "usage: test-hot-expert-cache [MODEL]\n";
        return 2;
    }
    return 0;
}
