#include "llama-expert-weight-provider.h"
#include "llama-model.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

class test_provider : public llm_expert_weight_provider {
public:
    llm_expert_provider_result bind(
            const llm_expert_bundle_descriptor & bundle,
            const llm_expert_selection & selection,
            llm_expert_graph_binding & binding) noexcept override {
        stats.bind_calls++;
        binding = {};
        binding.provider_identity = this;
        binding.layer = bundle.layer;
        binding.up = bundle.up;
        binding.gate = bundle.gate;
        binding.gate_up = bundle.gate_up;
        binding.down = bundle.down;
        binding.execution_ids = selection.logical_ids;
        return llm_expert_provider_result::success();
    }

    llm_expert_provider_result prepare(
            const std::vector<llm_expert_graph_binding> &,
            llm_expert_execution_plan &,
            uint64_t,
            bool) noexcept override {
        stats.prepare_calls++;
        return llm_expert_provider_result::success();
    }

    llm_expert_provider_stats get_stats() const noexcept override {
        return stats;
    }

    llm_expert_handle acquire(uint64_t lease_id) {
        stats.handles_acquired++;
        return { this, lease_id };
    }

    std::vector<uint64_t> released;

protected:
    void release_handle(uint64_t lease_id) noexcept override {
        released.push_back(lease_id);
        stats.handles_released++;
    }

private:
    llm_expert_provider_stats stats;
};

struct test_tensors {
    std::vector<uint8_t> memory;
    ggml_context_ptr ctx;
    ggml_tensor * up;
    ggml_tensor * gate;
    ggml_tensor * gate_up;
    ggml_tensor * down;
    ggml_tensor * bias;
    ggml_tensor * scale;
    ggml_tensor * ids;
    ggml_tensor * ids_f32;

    test_tensors() : memory(64*1024) {
        ggml_init_params params = {
            /*.mem_size   =*/ memory.size(),
            /*.mem_buffer =*/ memory.data(),
            /*.no_alloc   =*/ true,
        };
        ctx.reset(ggml_init(params));
        GGML_ASSERT(ctx);

        up = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 16, 8, 4);
        gate = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 16, 8, 4);
        gate_up = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 32, 8, 4);
        down = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 8, 16, 4);
        bias = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 16, 4);
        scale = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 4);
        ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 2, 3);
        ids_f32 = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 2, 3);
    }
};

llm_expert_bundle_descriptor separate_bundle(test_tensors & tensors) {
    return {
        2,
        4,
        llm_expert_projection_descriptor::from(tensors.up, tensors.bias, tensors.scale),
        llm_expert_projection_descriptor::from(tensors.gate, tensors.bias, tensors.scale),
        {},
        llm_expert_projection_descriptor::from(tensors.down, nullptr, tensors.scale),
    };
}

void test_default_and_model_ownership() {
    auto params = llama_model_default_params();
    GGML_ASSERT(params.expert_weights_mode == LLAMA_EXPERT_WEIGHTS_MODE_DISABLED);
    GGML_ASSERT(params.expert_prefetch_config == nullptr);
    GGML_ASSERT(params.expert_prefetch_profile_path == nullptr);
    GGML_ASSERT(params.expert_hot_cache_capacity == 0);
    GGML_ASSERT(params.expert_cold_cache_bytes == 0);
    GGML_ASSERT(params.expert_transfer_ring_bytes == 0);

    llama_model * model = llama_model_create(LLM_ARCH_KIMI_K3, params);
    GGML_ASSERT(model != nullptr);
    GGML_ASSERT(model->expert_weight_provider() == nullptr);
    const auto stats = model->expert_weight_provider_stats();
    GGML_ASSERT(stats.objects_created == 0);
    GGML_ASSERT(stats.bind_calls == 0);
    GGML_ASSERT(stats.prepare_calls == 0);
    GGML_ASSERT(stats.handles_acquired == 0);
    GGML_ASSERT(stats.handles_released == 0);
    GGML_ASSERT(stats.allocations == 0);
    GGML_ASSERT(stats.callbacks == 0);
    GGML_ASSERT(stats.tensor_copies == 0);
    GGML_ASSERT(stats.synchronizations == 0);
    delete model;

    params.expert_cold_cache_bytes = 4096;
    bool invalid = false;
    try {
        model = llama_model_create(LLM_ARCH_KIMI_K3, params);
    } catch (const std::invalid_argument &) {
        invalid = true;
    }
    GGML_ASSERT(invalid);
    params.expert_cold_cache_bytes = 0;

    params.expert_weights_mode = LLAMA_EXPERT_WEIGHTS_MODE_RESIDENT;
    model = llama_model_create(LLM_ARCH_KIMI_K3, params);
    model->init_expert_weight_provider();
    GGML_ASSERT(model->expert_weight_provider() != nullptr);
    GGML_ASSERT(model->expert_weight_provider_stats().objects_created == 1);
    delete model;

    params.expert_weights_mode = LLAMA_EXPERT_WEIGHTS_MODE_COUNT;
    invalid = false;
    try {
        model = llama_model_create(LLM_ARCH_KIMI_K3, params);
    } catch (const std::invalid_argument &) {
        invalid = true;
    }
    GGML_ASSERT(invalid);
}

void test_keys_and_descriptors() {
    test_tensors tensors;

    GGML_ASSERT((llm_expert_key { 2, 3 }).is_valid(4, 4));
    GGML_ASSERT(!(llm_expert_key { -1, 0 }).is_valid(4, 4));
    GGML_ASSERT(!(llm_expert_key { 0, 4 }).is_valid(4, 4));

    auto bundle = separate_bundle(tensors);
    GGML_ASSERT(bundle.validate().is_ready());
    GGML_ASSERT(!bundle.uses_merged_gate_up());

    bundle.down.weight = nullptr;
    GGML_ASSERT(bundle.validate().error == llm_expert_provider_error::invalid_descriptor);

    bundle = separate_bundle(tensors);
    bundle.n_expert = 5;
    GGML_ASSERT(bundle.validate().error == llm_expert_provider_error::invalid_descriptor);

    bundle = {
        2,
        4,
        {},
        {},
        llm_expert_projection_descriptor::from(tensors.gate_up, tensors.bias, tensors.scale),
        llm_expert_projection_descriptor::from(tensors.down, nullptr, tensors.scale),
    };
    GGML_ASSERT(bundle.validate().is_ready());
    GGML_ASSERT(bundle.uses_merged_gate_up());
}

void test_selection_binding_and_handles() {
    test_tensors tensors;
    auto bundle = separate_bundle(tensors);
    llm_expert_selection selection = { 2, 4, 2, 3, tensors.ids };
    GGML_ASSERT(selection.validate().is_ready());

    selection.logical_ids = tensors.ids_f32;
    GGML_ASSERT(selection.validate().error == llm_expert_provider_error::invalid_selection);
    selection.logical_ids = tensors.ids;

    test_provider provider;
    llm_expert_graph_binding binding;
    GGML_ASSERT(provider.bind(bundle, selection, binding).is_ready());
    GGML_ASSERT(binding.validate(selection).is_ready());
    GGML_ASSERT(binding.execution_ids == selection.logical_ids);
    GGML_ASSERT(binding.up.weight == bundle.up.weight);
    GGML_ASSERT(binding.gate.weight == bundle.gate.weight);
    GGML_ASSERT(binding.down.weight == bundle.down.weight);

    llm_expert_execution_plan plan;
    plan.reserve(3);
    plan.add_handle(provider.acquire(1));
    plan.add_handle(provider.acquire(2));
    plan.add_handle(provider.acquire(3));
    GGML_ASSERT(plan.handle_count() == 3);
    plan.reset();
    GGML_ASSERT((provider.released == std::vector<uint64_t> { 3, 2, 1 }));
    GGML_ASSERT(provider.get_stats().handles_acquired == 3);
    GGML_ASSERT(provider.get_stats().handles_released == 3);
}

void test_failed_graph_binding_is_never_reusable() {
    llm_graph_result graph_result(16);
    llm_graph_params params = {};
    graph_result.set_params(params);
    GGML_ASSERT(graph_result.can_reuse(params));

    graph_result.set_expert_provider_result(
        llm_expert_provider_result::failure(llm_expert_provider_error::invalid_binding));
    GGML_ASSERT(!graph_result.can_reuse(params));
    GGML_ASSERT(!graph_result.can_reuse(params));

    graph_result.reset();
    graph_result.set_params(params);
    GGML_ASSERT(graph_result.can_reuse(params));
}

void test_graph_binding_capacity_survives_reset() {
    llm_graph_result graph_result(16);
    GGML_ASSERT(graph_result.get_expert_binding_capacity() == 0);
    graph_result.reserve_expert_bindings(7);
    GGML_ASSERT(graph_result.get_expert_binding_capacity() >= 7);
    graph_result.reset();
    GGML_ASSERT(graph_result.get_expert_binding_capacity() >= 7);
}

void test_resident_provider_and_plan_retention() {
    test_tensors tensors;
    auto bundle = separate_bundle(tensors);
    const llm_expert_selection selection = { 2, 4, 2, 3, tensors.ids };
    auto provider = llm_create_resident_expert_weight_provider();

    llm_expert_graph_binding binding;
    GGML_ASSERT(provider->bind(bundle, selection, binding).is_ready());
    GGML_ASSERT(binding.provider_identity == provider.get());
    GGML_ASSERT(binding.execution_ids == selection.logical_ids);
    GGML_ASSERT(binding.up.weight == bundle.up.weight);
    GGML_ASSERT(binding.gate.weight == bundle.gate.weight);
    GGML_ASSERT(binding.down.weight == bundle.down.weight);

    llm_expert_graph_binding repeated_binding;
    GGML_ASSERT(provider->bind(bundle, selection, repeated_binding).is_ready());
    auto stats = provider->get_stats();
    GGML_ASSERT(stats.bundle_registrations == 1);
    GGML_ASSERT(stats.bundle_full_validations == 1);
    GGML_ASSERT(stats.bundle_fast_path_hits == 1);

    test_tensors conflicting_tensors;
    const auto conflicting_bundle = separate_bundle(conflicting_tensors);
    GGML_ASSERT(provider->bind(conflicting_bundle, selection, repeated_binding).error ==
        llm_expert_provider_error::invalid_descriptor);

    llm_expert_selection invalid_selection = selection;
    invalid_selection.logical_ids = tensors.ids_f32;
    GGML_ASSERT(provider->bind(bundle, invalid_selection, repeated_binding).error ==
        llm_expert_provider_error::invalid_selection);

    llm_expert_execution_plan request;
    request.reserve(1);
    std::vector<llm_expert_graph_binding> bindings = { binding, binding };
    GGML_ASSERT(provider->prepare(bindings, request).is_ready());
    GGML_ASSERT(request.handle_count() == 1);

    llm_expert_execution_plan inflight;
    inflight.reserve(1);
    inflight.absorb(std::move(request));
    GGML_ASSERT(request.handle_count() == 0);
    GGML_ASSERT(inflight.handle_count() == 1);
    GGML_ASSERT(provider->get_stats().handles_released == 0);
    inflight.reset();

    stats = provider->get_stats();
    GGML_ASSERT(stats.objects_created == 1);
    GGML_ASSERT(stats.bind_calls == 4);
    GGML_ASSERT(stats.prepare_calls == 1);
    GGML_ASSERT(stats.handles_acquired == 1);
    GGML_ASSERT(stats.handles_released == 1);
    GGML_ASSERT(stats.allocations == 0);
    GGML_ASSERT(stats.callbacks == 0);
    GGML_ASSERT(stats.tensor_copies == 0);
    GGML_ASSERT(stats.synchronizations == 0);

    llm_expert_provider_faults cached_fault;
    cached_fault.binding = llm_expert_provider_error::preparation_failed;
    cached_fault.binding_successes_before_failure = 1;
    provider = llm_create_resident_expert_weight_provider(cached_fault);
    GGML_ASSERT(provider->bind(bundle, selection, binding).is_ready());
    GGML_ASSERT(provider->bind(bundle, selection, binding).error == llm_expert_provider_error::preparation_failed);
    GGML_ASSERT(provider->get_stats().bundle_fast_path_hits == 1);

    auto concurrent_provider = llm_create_resident_expert_weight_provider();
    std::atomic<int> ready_count { 0 };
    std::vector<std::thread> binders;
    for (int index = 0; index < 8; ++index) {
        binders.emplace_back([&] {
            llm_expert_graph_binding concurrent_binding;
            if (concurrent_provider->bind(bundle, selection, concurrent_binding).is_ready()) {
                ready_count.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }
    for (auto & binder : binders) {
        binder.join();
    }
    const auto concurrent_stats = concurrent_provider->get_stats();
    GGML_ASSERT(ready_count.load(std::memory_order_relaxed) == 8);
    GGML_ASSERT(concurrent_stats.bundle_registrations == 1);
    GGML_ASSERT(concurrent_stats.bundle_full_validations == 1);
    GGML_ASSERT(concurrent_stats.bundle_fast_path_hits == 7);

    llm_expert_execution_plan empty;
    GGML_ASSERT(provider->prepare({}, empty).is_ready());
    GGML_ASSERT(empty.handle_count() == 0);

    auto invalid_binding = binding;
    invalid_binding.down.weight = conflicting_bundle.down.weight;
    GGML_ASSERT(provider->prepare({ invalid_binding }, empty).error == llm_expert_provider_error::invalid_binding);
    GGML_ASSERT(empty.handle_count() == 0);
    invalid_binding = binding;
    invalid_binding.execution_ids = tensors.ids_f32;
    GGML_ASSERT(provider->prepare({ invalid_binding }, empty).error == llm_expert_provider_error::invalid_binding);
    GGML_ASSERT(empty.handle_count() == 0);
}

void test_resident_provider_failures_cleanup_partially_acquired_handles() {
    test_tensors tensors;
    const auto bundle = separate_bundle(tensors);
    const llm_expert_selection selection = { 2, 4, 2, 3, tensors.ids };

    llm_expert_provider_faults bind_faults;
    bind_faults.binding = llm_expert_provider_error::preparation_failed;
    auto provider = llm_create_resident_expert_weight_provider(bind_faults);
    llm_expert_graph_binding binding;
    GGML_ASSERT(provider->bind(bundle, selection, binding).error == llm_expert_provider_error::preparation_failed);
    GGML_ASSERT(provider->get_stats().failures == 1);

    provider = llm_create_resident_expert_weight_provider();
    GGML_ASSERT(provider->bind(bundle, selection, binding).is_ready());

    llm_expert_provider_faults prepare_faults;
    prepare_faults.preparation = llm_expert_provider_error::allocation_failed;
    prepare_faults.fail_preparation_after_handles = 1;
    provider = llm_create_resident_expert_weight_provider(prepare_faults);
    GGML_ASSERT(provider->bind(bundle, selection, binding).is_ready());
    llm_expert_execution_plan plan;
    plan.reserve(1);
    const auto result = provider->prepare({ binding, binding }, plan);
    GGML_ASSERT(result.status == llm_expert_provider_status::allocation_failed);
    GGML_ASSERT(plan.handle_count() == 0);
    GGML_ASSERT(provider->get_stats().handles_acquired == 1);
    GGML_ASSERT(provider->get_stats().handles_released == 1);

    prepare_faults.preparation = llm_expert_provider_error::none;
    provider = llm_create_resident_expert_weight_provider(prepare_faults);
    GGML_ASSERT(provider->bind(bundle, selection, binding).is_ready());
    GGML_ASSERT(provider->prepare({ binding }, plan).error == llm_expert_provider_error::preparation_failed);
    GGML_ASSERT(plan.handle_count() == 0);
    GGML_ASSERT(provider->get_stats().handles_acquired == provider->get_stats().handles_released);

    llm_expert_provider_faults cancellation;
    cancellation.preparation = llm_expert_provider_error::cancelled;
    cancellation.fail_preparation_after_handles = 0;
    provider = llm_create_resident_expert_weight_provider(cancellation);
    GGML_ASSERT(provider->bind(bundle, selection, binding).is_ready());
    const auto cancelled = provider->prepare({ binding }, plan);
    GGML_ASSERT(cancelled.status == llm_expert_provider_status::cancelled);
    GGML_ASSERT(plan.handle_count() == 0);
    GGML_ASSERT(provider->get_stats().cancellations == 1);

    llm_expert_provider_faults init_faults;
    init_faults.initialization = llm_expert_provider_error::initialization_failed;
    bool init_failed = false;
    try {
        provider = llm_create_resident_expert_weight_provider(init_faults);
    } catch (const std::runtime_error &) {
        init_failed = true;
    }
    GGML_ASSERT(init_failed);
}

struct generation_result {
    std::vector<llama_token> generated;
    std::vector<float> logits;
};

generation_result run_generation(llama_model * model) {
    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = 64;
    context_params.n_batch = 64;
    context_params.n_ubatch = 64;
    llama_context * context = llama_init_from_model(model, context_params);
    GGML_ASSERT(context != nullptr);
    llama_set_n_threads(context, 4, 4);

    const llama_vocab * vocab = llama_model_get_vocab(model);
    constexpr const char * prompt_text = "According to all known laws";
    const int n_prompt = -llama_tokenize(vocab, prompt_text, std::strlen(prompt_text), nullptr, 0, true, true);
    GGML_ASSERT(n_prompt > 0);
    std::vector<llama_token> prompt(n_prompt);
    GGML_ASSERT(llama_tokenize(vocab, prompt_text, std::strlen(prompt_text), prompt.data(), prompt.size(), true, true) == n_prompt);

    generation_result result;
    llama_batch batch = llama_batch_get_one(prompt.data(), prompt.size());
    const int n_vocab = llama_vocab_n_tokens(vocab);
    uint64_t token_graph_bind_calls = 0;
    for (int step = 0; step < 4; ++step) {
        GGML_ASSERT(llama_decode(context, batch) == 0);
        const float * logits = llama_get_logits_ith(context, -1);
        GGML_ASSERT(logits != nullptr);
        GGML_ASSERT(std::all_of(logits, logits + n_vocab, [](float value) { return std::isfinite(value); }));
        result.logits.insert(result.logits.end(), logits, logits + n_vocab);
        const llama_token next = std::max_element(logits, logits + n_vocab) - logits;
        result.generated.push_back(next);
        batch = llama_batch_get_one(&result.generated.back(), 1);
        if (model->expert_weight_provider()) {
            const uint64_t bind_calls = model->expert_weight_provider_stats().bind_calls;
            if (step == 1) {
                token_graph_bind_calls = bind_calls;
            } else if (step > 1) {
                GGML_ASSERT(bind_calls == token_graph_bind_calls);
            }
        }
    }

    llama_free(context);
    return result;
}

std::vector<llama_token> integration_prompt(llama_model * model) {
    const llama_vocab * vocab = llama_model_get_vocab(model);
    constexpr const char * prompt_text = "According to all known laws";
    const int n_prompt = -llama_tokenize(vocab, prompt_text, std::strlen(prompt_text), nullptr, 0, true, true);
    GGML_ASSERT(n_prompt > 0);
    std::vector<llama_token> prompt(n_prompt);
    GGML_ASSERT(llama_tokenize(vocab, prompt_text, std::strlen(prompt_text), prompt.data(), prompt.size(), true, true) == n_prompt);
    return prompt;
}

llama_context * integration_context(llama_model * model) {
    llama_context_params params = llama_context_default_params();
    params.n_ctx = 64;
    params.n_batch = 64;
    params.n_ubatch = 64;
    llama_context * context = llama_init_from_model(model, params);
    GGML_ASSERT(context != nullptr);
    llama_set_n_threads(context, 4, 4);
    return context;
}

void test_shared_contexts_and_async_destruction(llama_model * model, const std::vector<float> & expected_logits) {
    const auto prompt = integration_prompt(model);
    llama_context * first = integration_context(model);
    llama_context * second = integration_context(model);
    GGML_ASSERT(llama_decode(first, llama_batch_get_one(const_cast<llama_token *>(prompt.data()), prompt.size())) == 0);
    GGML_ASSERT(llama_decode(second, llama_batch_get_one(const_cast<llama_token *>(prompt.data()), prompt.size())) == 0);

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const float * first_logits = llama_get_logits_ith(first, -1);
    const float * second_logits = llama_get_logits_ith(second, -1);
    GGML_ASSERT(first_logits != nullptr && second_logits != nullptr);
    GGML_ASSERT(std::equal(first_logits, first_logits + n_vocab, expected_logits.begin()));
    GGML_ASSERT(std::equal(second_logits, second_logits + n_vocab, expected_logits.begin()));
    llama_free(first);
    llama_free(second);

    llama_context * pending = integration_context(model);
    GGML_ASSERT(llama_decode(pending, llama_batch_get_one(const_cast<llama_token *>(prompt.data()), prompt.size())) == 0);
    const auto before_destroy = model->expert_weight_provider_stats();
    GGML_ASSERT(before_destroy.handles_acquired > before_destroy.handles_released);
    llama_free(pending);
    const auto after_destroy = model->expert_weight_provider_stats();
    GGML_ASSERT(after_destroy.handles_acquired == after_destroy.handles_released);
}

void test_model_failure_paths(llama_model * model) {
    const auto prompt = integration_prompt(model);

    llm_expert_provider_faults binding_fault;
    binding_fault.binding = llm_expert_provider_error::preparation_failed;
    model->replace_expert_weight_provider_for_testing(llm_create_resident_expert_weight_provider(binding_fault));
    llama_context * context = integration_context(model);
    GGML_ASSERT(llama_decode(context, llama_batch_get_one(const_cast<llama_token *>(prompt.data()), prompt.size())) == -3);
    GGML_ASSERT(llama_decode(context, llama_batch_get_one(const_cast<llama_token *>(prompt.data()), prompt.size())) == -3);
    auto stats = model->expert_weight_provider_stats();
    GGML_ASSERT(stats.bind_calls >= 2);
    GGML_ASSERT(stats.prepare_calls == 0);
    GGML_ASSERT(stats.handles_acquired == 0);
    llama_free(context);

    llm_expert_provider_faults allocation_fault;
    allocation_fault.preparation = llm_expert_provider_error::allocation_failed;
    allocation_fault.fail_preparation_after_handles = 1;
    model->replace_expert_weight_provider_for_testing(llm_create_resident_expert_weight_provider(allocation_fault));
    context = integration_context(model);
    GGML_ASSERT(llama_decode(context, llama_batch_get_one(const_cast<llama_token *>(prompt.data()), prompt.size())) == -2);
    stats = model->expert_weight_provider_stats();
    GGML_ASSERT(stats.prepare_calls == 1);
    GGML_ASSERT(stats.handles_acquired == 1);
    GGML_ASSERT(stats.handles_released == 1);
    llama_free(context);

    llm_expert_provider_faults preparation_fault;
    preparation_fault.preparation = llm_expert_provider_error::preparation_failed;
    preparation_fault.fail_preparation_after_handles = 1;
    model->replace_expert_weight_provider_for_testing(llm_create_resident_expert_weight_provider(preparation_fault));
    context = integration_context(model);
    GGML_ASSERT(llama_decode(context, llama_batch_get_one(const_cast<llama_token *>(prompt.data()), prompt.size())) == -3);
    stats = model->expert_weight_provider_stats();
    GGML_ASSERT(stats.handles_acquired == stats.handles_released);
    llama_free(context);

    llm_expert_provider_faults cancellation;
    cancellation.preparation = llm_expert_provider_error::cancelled;
    cancellation.fail_preparation_after_handles = 0;
    model->replace_expert_weight_provider_for_testing(llm_create_resident_expert_weight_provider(cancellation));
    context = integration_context(model);
    GGML_ASSERT(llama_decode(context, llama_batch_get_one(const_cast<llama_token *>(prompt.data()), prompt.size())) == -3);
    stats = model->expert_weight_provider_stats();
    GGML_ASSERT(stats.cancellations == 1);
    GGML_ASSERT(stats.handles_acquired == stats.handles_released);
    llama_free(context);

    model->replace_expert_weight_provider_for_testing(llm_create_resident_expert_weight_provider());
    context = integration_context(model);
    llama_set_abort_callback(context, [](void *) { return true; }, nullptr);
    GGML_ASSERT(llama_decode(context, llama_batch_get_one(const_cast<llama_token *>(prompt.data()), prompt.size())) == 2);
    stats = model->expert_weight_provider_stats();
    GGML_ASSERT(stats.handles_acquired > 0);
    GGML_ASSERT(stats.handles_acquired == stats.handles_released);
    llama_free(context);
}

void test_two_models(const char * disabled_path, const char * resident_path, int n_gpu_layers) {
    llama_model_params disabled_params = llama_model_default_params();
    disabled_params.n_gpu_layers = n_gpu_layers;
    llama_model * disabled_model = llama_model_load_from_file(disabled_path, disabled_params);
    GGML_ASSERT(disabled_model != nullptr);

    llama_model_params resident_params = disabled_params;
    resident_params.expert_weights_mode = LLAMA_EXPERT_WEIGHTS_MODE_RESIDENT;
    llama_model * resident_model = llama_model_load_from_file(resident_path, resident_params);
    GGML_ASSERT(resident_model != nullptr);

    const auto disabled_prompt = integration_prompt(disabled_model);
    const auto resident_prompt = integration_prompt(resident_model);
    llama_context * disabled_context = integration_context(disabled_model);
    llama_context * resident_context = integration_context(resident_model);
    GGML_ASSERT(llama_decode(disabled_context, llama_batch_get_one(
        const_cast<llama_token *>(disabled_prompt.data()), disabled_prompt.size())) == 0);
    GGML_ASSERT(llama_decode(resident_context, llama_batch_get_one(
        const_cast<llama_token *>(resident_prompt.data()), resident_prompt.size())) == 0);

    const float * disabled_logits = llama_get_logits_ith(disabled_context, -1);
    const float * resident_logits = llama_get_logits_ith(resident_context, -1);
    GGML_ASSERT(disabled_logits != nullptr && resident_logits != nullptr);
    GGML_ASSERT(std::all_of(disabled_logits,
        disabled_logits + llama_vocab_n_tokens(llama_model_get_vocab(disabled_model)),
        [](float value) { return std::isfinite(value); }));
    GGML_ASSERT(std::all_of(resident_logits,
        resident_logits + llama_vocab_n_tokens(llama_model_get_vocab(resident_model)),
        [](float value) { return std::isfinite(value); }));
    GGML_ASSERT(disabled_model->expert_weight_provider_stats().objects_created == 0);
    GGML_ASSERT(resident_model->expert_weight_provider_stats().handles_acquired > 0);

    llama_free(disabled_context);
    llama_free(resident_context);
    const auto resident_stats = resident_model->expert_weight_provider_stats();
    GGML_ASSERT(resident_stats.handles_acquired == resident_stats.handles_released);
    llama_model_free(disabled_model);
    llama_model_free(resident_model);
}

void test_model_integration(const char * model_path, int n_gpu_layers) {
    llama_model_params disabled_params = llama_model_default_params();
    disabled_params.n_gpu_layers = n_gpu_layers;
    llama_model * disabled_model = llama_model_load_from_file(model_path, disabled_params);
    GGML_ASSERT(disabled_model != nullptr);
    const auto disabled = run_generation(disabled_model);
    const auto disabled_stats = disabled_model->expert_weight_provider_stats();
    GGML_ASSERT(disabled_stats.objects_created == 0);
    GGML_ASSERT(disabled_stats.bind_calls == 0);
    GGML_ASSERT(disabled_stats.prepare_calls == 0);
    GGML_ASSERT(disabled_stats.handles_acquired == 0);
    GGML_ASSERT(disabled_stats.handles_released == 0);
    llama_model_params resident_params = disabled_params;
    resident_params.expert_weights_mode = LLAMA_EXPERT_WEIGHTS_MODE_RESIDENT;
    llama_model * resident_model = llama_model_load_from_file(model_path, resident_params);
    GGML_ASSERT(resident_model != nullptr);
    const auto resident_a = run_generation(resident_model);
    const auto resident_b = run_generation(resident_model);
    test_shared_contexts_and_async_destruction(
        resident_model,
        std::vector<float>(disabled.logits.begin(), disabled.logits.begin() + llama_vocab_n_tokens(llama_model_get_vocab(resident_model))));
    const auto resident_stats = resident_model->expert_weight_provider_stats();
    GGML_ASSERT(resident_stats.objects_created == 1);
    GGML_ASSERT(resident_stats.bind_calls > 0);
    GGML_ASSERT(resident_stats.prepare_calls == 11);
    GGML_ASSERT(resident_stats.handles_acquired == resident_stats.prepare_calls);
    GGML_ASSERT(resident_stats.handles_released == resident_stats.prepare_calls);
    GGML_ASSERT(resident_stats.bundle_registrations == 7);
    GGML_ASSERT(resident_stats.bundle_full_validations == 7);
    GGML_ASSERT(resident_stats.bundle_fast_path_hits > 0);
    GGML_ASSERT(resident_stats.allocations == 0);
    GGML_ASSERT(resident_stats.callbacks == 0);
    GGML_ASSERT(resident_stats.tensor_copies == 0);
    GGML_ASSERT(resident_stats.synchronizations == 0);
    GGML_ASSERT(resident_a.generated == disabled.generated);
    GGML_ASSERT(resident_a.logits == disabled.logits);
    GGML_ASSERT(resident_b.generated == disabled.generated);
    GGML_ASSERT(resident_b.logits == disabled.logits);
    if (n_gpu_layers == 0) {
        test_model_failure_paths(resident_model);
    }
    llama_model_free(disabled_model);
    llama_model_free(resident_model);
}

} // namespace

int main(int argc, char ** argv) {
    test_default_and_model_ownership();
    test_keys_and_descriptors();
    test_selection_binding_and_handles();
    test_failed_graph_binding_is_never_reusable();
    test_graph_binding_capacity_survives_reset();
    test_resident_provider_and_plan_retention();
    test_resident_provider_failures_cleanup_partially_acquired_handles();
    if (argc == 3 || argc == 4) {
        ggml_backend_load_all();
        test_model_integration(argv[1], std::stoi(argv[2]));
        if (argc == 4) {
            test_two_models(argv[1], argv[3], std::stoi(argv[2]));
        }
        llama_backend_free();
    } else if (argc != 1) {
        std::cerr << "usage: test-expert-weight-provider [MODEL GPU_LAYERS [SECOND_MODEL]]\n";
        return 2;
    }
    return 0;
}
