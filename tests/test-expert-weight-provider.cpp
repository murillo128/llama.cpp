#include "llama-expert-weight-provider.h"
#include "llama-model.h"

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {

class test_provider : public llm_expert_weight_provider {
public:
    llm_expert_provider_result bind(
            const llm_expert_bundle_descriptor & bundle,
            const llm_expert_selection & selection,
            llm_expert_graph_binding & binding) noexcept override {
        stats.bind_calls++;
        binding = {
            this,
            bundle.layer,
            bundle.up,
            bundle.gate,
            bundle.gate_up,
            bundle.down,
            selection.logical_ids,
        };
        return llm_expert_provider_result::success();
    }

    llm_expert_provider_result prepare(
            const std::vector<llm_expert_graph_binding> &,
            llm_expert_execution_plan &) noexcept override {
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

    params.expert_weights_mode = LLAMA_EXPERT_WEIGHTS_MODE_RESIDENT;
    model = llama_model_create(LLM_ARCH_KIMI_K3, params);
    bool unavailable = false;
    try {
        model->init_expert_weight_provider();
    } catch (const std::runtime_error &) {
        unavailable = true;
    }
    GGML_ASSERT(unavailable);
    delete model;

    params.expert_weights_mode = static_cast<llama_expert_weights_mode>(99);
    bool invalid = false;
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

} // namespace

int main() {
    test_default_and_model_ownership();
    test_keys_and_descriptors();
    test_selection_binding_and_handles();
    return 0;
}
