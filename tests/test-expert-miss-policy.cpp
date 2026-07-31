#include "llama-expert-weight-provider.h"
#include "llama-context.h"
#include "llama-model.h"
#include "llama-cpp.h"

#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace {

static_assert(std::is_standard_layout_v<llama_expert_auto_cost_model>);
static_assert(std::is_trivially_copyable_v<llama_expert_auto_cost_model>);

llama_expert_auto_cost_model valid_cost_model() {
    return {
        LLAMA_EXPERT_AUTO_COST_MODEL_VERSION_1,
        sizeof(llama_expert_auto_cost_model),
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
    };
}

struct validation_model final : llama_model {
    explicit validation_model(const llama_model_params & params) : llama_model(params) {}

    const llama_expert_auto_cost_model * copied_cost() const { return params.expert_auto_cost_model; }

    void load_stats(llama_model_loader &) override {}
    void load_hparams(llama_model_loader &) override {}
    void load_vocab(llama_model_loader &) override {}
    bool load_tensors(llama_model_loader &) override { return true; }
    void load_arch_hparams(llama_model_loader &) override {}
    void load_arch_tensors(llama_model_loader &) override {}
    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params &) const override { return {}; }
};

template<class F> void expect_invalid(F && fn) {
    bool rejected = false;
    try {
        fn();
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    GGML_ASSERT(rejected);
}

void test_model_parameter_validation_and_copy() {
    const auto defaults = llama_model_default_params();
    validation_model default_model(defaults);

    auto invalid_policy = defaults;
    invalid_policy.expert_miss_policy = static_cast<llama_expert_miss_policy>(LLAMA_EXPERT_MISS_POLICY_COUNT);
    expect_invalid([&] { validation_model model(invalid_policy); });

    auto outside_cold = defaults;
    outside_cold.expert_miss_policy = LLAMA_EXPERT_MISS_POLICY_CPU_FALLBACK;
    expect_invalid([&] { validation_model model(outside_cold); });
    outside_cold = defaults;
    outside_cold.expert_background_promotion = true;
    expect_invalid([&] { validation_model model(outside_cold); });

    auto cost = valid_cost_model();
    auto non_auto_cost = defaults;
    non_auto_cost.expert_auto_cost_model = &cost;
    expect_invalid([&] { validation_model model(non_auto_cost); });

    auto automatic = defaults;
    automatic.expert_weights_mode = LLAMA_EXPERT_WEIGHTS_MODE_COLD_CACHE;
    automatic.expert_hot_cache_capacity = 2;
    automatic.expert_cold_cache_bytes = 1U << 20;
    automatic.expert_transfer_ring_bytes = 1U << 20;
    automatic.expert_miss_policy = LLAMA_EXPERT_MISS_POLICY_AUTO;
    expect_invalid([&] { validation_model model(automatic); });
    automatic.expert_auto_cost_model = &cost;

    auto malformed = cost;
    malformed.struct_size--;
    automatic.expert_auto_cost_model = &malformed;
    expect_invalid([&] { validation_model model(automatic); });
    malformed = cost;
    malformed.cpu_per_lane_decode_ns = 0;
    automatic.expert_auto_cost_model = &malformed;
    expect_invalid([&] { validation_model model(automatic); });

    automatic.expert_auto_cost_model = &cost;
    validation_model copied(automatic);
    GGML_ASSERT(copied.copied_cost() != &cost);
    const auto original_decode_cost = copied.copied_cost()->cpu_fixed_decode_ns;
    cost.cpu_fixed_decode_ns++;
    GGML_ASSERT(copied.copied_cost()->cpu_fixed_decode_ns == original_decode_cost);
}

void mark_inactive_capable(ggml_tensor * tensor) {
    tensor->op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1] = 1;
}

void test_defaults_and_lane_validation() {
    const auto params = llama_model_default_params();
    GGML_ASSERT(params.expert_miss_policy == LLAMA_EXPERT_MISS_POLICY_PROMOTE_AND_GPU);
    GGML_ASSERT(!params.expert_background_promotion);
    GGML_ASSERT(params.expert_auto_cost_model == nullptr);

    const int32_t gpu[] = { 0, -1, 1, -1 };
    const int32_t cpu[] = { -1, 2, -1, 0 };
    GGML_ASSERT(llm_validate_hybrid_execution_ids(gpu, cpu, 4, 2, 3).is_ready());

    const int32_t below[] = { -2, 2, -1, 0 };
    GGML_ASSERT(!llm_validate_hybrid_execution_ids(gpu, below, 4, 2, 3).is_ready());
    const int32_t both[] = { 0, 2, -1, 0 };
    GGML_ASSERT(!llm_validate_hybrid_execution_ids(gpu, both, 4, 2, 3).is_ready());
    const int32_t neither[] = { -1, 2, -1, 0 };
    GGML_ASSERT(!llm_validate_hybrid_execution_ids(neither, cpu, 4, 2, 3).is_ready());
    GGML_ASSERT(!llm_validate_hybrid_execution_ids(gpu, cpu, 0, 2, 3).is_ready());
}

struct graph_fixture {
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buffer;
    ggml_tensor * weights = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * ids = nullptr;
    ggml_tensor * bias = nullptr;
    ggml_tensor * scale = nullptr;
    ggml_tensor * scale_rows = nullptr;
    ggml_tensor * matmul = nullptr;
    ggml_tensor * output = nullptr;
};

graph_fixture make_graph(ggml_backend_buffer_type_t buft, int64_t n_tokens, bool allow_inactive) {
    graph_fixture fixture;
    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead()*12 + ggml_graph_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    fixture.ctx.reset(ggml_init(params));
    GGML_ASSERT(fixture.ctx);
    fixture.weights = ggml_new_tensor_3d(fixture.ctx.get(), GGML_TYPE_F32, 2, 2, 3);
    fixture.input = ggml_new_tensor_3d(fixture.ctx.get(), GGML_TYPE_F32, 2, 3, n_tokens);
    fixture.ids = ggml_new_tensor_2d(fixture.ctx.get(), GGML_TYPE_I32, 3, n_tokens);
    fixture.bias = ggml_new_tensor_2d(fixture.ctx.get(), GGML_TYPE_F32, 2, 3);
    fixture.scale = ggml_new_tensor_1d(fixture.ctx.get(), GGML_TYPE_F32, 3);
    fixture.matmul = ggml_mul_mat_id(fixture.ctx.get(), fixture.weights, fixture.input, fixture.ids);
    if (allow_inactive) mark_inactive_capable(fixture.matmul);
    auto * expanded_scale = ggml_reshape_3d(fixture.ctx.get(), fixture.scale, 1, 3, 1);
    expanded_scale = ggml_repeat_4d(fixture.ctx.get(), expanded_scale, 1, 3, n_tokens, 1);
    fixture.scale_rows = ggml_get_rows(fixture.ctx.get(), expanded_scale, fixture.ids);
    if (allow_inactive) mark_inactive_capable(fixture.scale_rows);
    fixture.output = ggml_mul(fixture.ctx.get(), fixture.matmul, fixture.scale_rows);
    fixture.output = ggml_add_id(fixture.ctx.get(), fixture.output, fixture.bias, fixture.ids);
    if (allow_inactive) mark_inactive_capable(fixture.output);
    GGML_ASSERT((fixture.matmul->op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1] != 0) == allow_inactive);
    GGML_ASSERT((fixture.scale_rows->op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1] != 0) == allow_inactive);
    GGML_ASSERT((fixture.output->op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1] != 0) == allow_inactive);
    fixture.buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(fixture.ctx.get(), buft));
    GGML_ASSERT(fixture.buffer);
    GGML_ASSERT((fixture.matmul->op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1] != 0) == allow_inactive);
    return fixture;
}

std::vector<float> run_case(ggml_backend_dev_t device, const std::vector<int32_t> & ids, bool allow_inactive) {
    GGML_ASSERT(!ids.empty() && ids.size() % 3 == 0);
    const int64_t n_tokens = ids.size()/3;
    ggml_backend_ptr backend(ggml_backend_dev_init(device, nullptr));
    GGML_ASSERT(backend);
    auto fixture = make_graph(ggml_backend_get_default_buffer_type(backend.get()), n_tokens, allow_inactive);

    const std::array<float, 12> weights = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
    };
    const std::array<float, 12> input = {
        1, 2, 3, 4, 5, 6,
        7, 8, 9, 10, 11, 12,
    };
    const std::array<float, 6> bias = { 1, -1, 2, -2, 3, -3 };
    const std::array<float, 3> scale = { 2, 3, 4 };
    ggml_backend_tensor_set(fixture.weights, weights.data(), 0, sizeof(weights));
    ggml_backend_tensor_set(fixture.input, input.data(), 0, size_t(n_tokens)*6*sizeof(float));
    ggml_backend_tensor_set(fixture.ids, ids.data(), 0, ids.size()*sizeof(int32_t));
    ggml_backend_tensor_set(fixture.bias, bias.data(), 0, sizeof(bias));
    ggml_backend_tensor_set(fixture.scale, scale.data(), 0, sizeof(scale));

    ggml_cgraph * graph = ggml_new_graph(fixture.ctx.get());
    ggml_build_forward_expand(graph, fixture.output);
    GGML_ASSERT((fixture.matmul->op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1] != 0) == allow_inactive);
    GGML_ASSERT(ggml_backend_graph_compute(backend.get(), graph) == GGML_STATUS_SUCCESS);
    std::vector<float> result(size_t(n_tokens)*6);
    ggml_backend_tensor_get(fixture.output, result.data(), 0, result.size()*sizeof(float));
    return result;
}

std::vector<float> reference(const std::vector<int32_t> & ids) {
    GGML_ASSERT(!ids.empty() && ids.size() % 3 == 0);
    const int n_tokens = ids.size()/3;
    const std::array<float, 12> weights = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
    };
    const std::array<float, 12> input = {
        1, 2, 3, 4, 5, 6,
        7, 8, 9, 10, 11, 12,
    };
    const std::array<float, 6> bias = { 1, -1, 2, -2, 3, -3 };
    const std::array<float, 3> scale = { 2, 3, 4 };
    std::vector<float> result(size_t(n_tokens)*6, 0.0f);
    for (int token = 0; token < n_tokens; ++token) {
        for (int rank = 0; rank < 3; ++rank) {
            const int lane = token*3 + rank;
            const int expert = ids[lane];
            if (expert < 0) continue;
            for (int row = 0; row < 2; ++row) {
                float value = bias[expert*2 + row];
                for (int col = 0; col < 2; ++col) {
                    value += scale[expert]*weights[expert*4 + row*2 + col]*input[token*6 + rank*2 + col];
                }
                result[token*6 + rank*2 + row] = value;
            }
        }
    }
    return result;
}

void assert_case(ggml_backend_dev_t device, const std::vector<int32_t> & ids, bool allow_inactive = true) {
    const auto actual = run_case(device, ids, allow_inactive);
    const auto expected = reference(ids);
    GGML_ASSERT(actual.size() == expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        GGML_ASSERT(std::fabs(actual[i] - expected[i]) < 1e-5f);
    }
}

struct provider_fixture {
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buffer;
    ggml_tensor * up = nullptr;
    ggml_tensor * gate = nullptr;
    ggml_tensor * down = nullptr;
    ggml_tensor * ids = nullptr;

    provider_fixture() {
        ggml_init_params params = { ggml_tensor_overhead()*8, nullptr, true };
        ctx.reset(ggml_init(params));
        up = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 2, 4, 4);
        gate = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 2, 4, 4);
        down = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 4, 2, 4);
        ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 2, 1);
        buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), ggml_backend_cpu_buffer_type()));
        GGML_ASSERT(buffer);
    }

    llm_expert_bundle_descriptor bundle() const {
        return { 0, 4,
            llm_expert_projection_descriptor::from(up, nullptr, nullptr),
            llm_expert_projection_descriptor::from(gate, nullptr, nullptr), {},
            llm_expert_projection_descriptor::from(down, nullptr, nullptr) };
    }
};

void test_hybrid_binding() {
    ggml_backend_load_all();
    provider_fixture fixture;
    llm_hot_cache_config config;
    config.capacity = 2;
    config.n_expert_used = 2;
    config.routed_layer_count = 1;
    config.total_expert_keys = 4;
    config.target_buffer_type = ggml_backend_cpu_buffer_type();
    config.allow_non_cuda_target_for_testing = true;
    config.cold_mode = true;
    config.cold_cache_bytes = 1U << 20;
    config.transfer_ring_bytes = 1U << 20;
    config.target_device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    config.force_pageable_transfer_for_testing = true;
    config.miss_policy = LLAMA_EXPERT_MISS_POLICY_CPU_FALLBACK;
    auto provider = llm_create_cold_cache_expert_weight_provider(config);

    ggml_init_params graph_params = { ggml_tensor_overhead()*8, nullptr, true };
    ggml_context_ptr graph_ctx(ggml_init(graph_params));
    const llm_expert_selection selection = { 0, 4, 2, 1, fixture.ids };
    llm_expert_graph_binding binding;
    GGML_ASSERT(provider->bind_graph(graph_ctx.get(), fixture.bundle(), selection, binding).is_ready());
    GGML_ASSERT(binding.bootstrap && !binding.hybrid);
    GGML_ASSERT(provider->initialize_after_reserve().is_ready());
    binding = {};
    GGML_ASSERT(provider->bind_graph(graph_ctx.get(), fixture.bundle(), selection, binding).is_ready());
    GGML_ASSERT(!binding.bootstrap && binding.hybrid);
    GGML_ASSERT(binding.execution_ids != selection.logical_ids);
    GGML_ASSERT(binding.cpu_execution_ids != nullptr && binding.cpu_execution_ids != binding.execution_ids);
    GGML_ASSERT(binding.cpu_up.weight != nullptr && binding.cpu_gate.weight != nullptr && binding.cpu_down.weight != nullptr);
    const auto diagnostics = provider->hot_cache_diagnostics();
    GGML_ASSERT(diagnostics.configured_miss_policy == LLAMA_EXPERT_MISS_POLICY_CPU_FALLBACK);
    GGML_ASSERT(diagnostics.hybrid_bindings == 1);
    binding = {};
    GGML_ASSERT(provider->surrender().is_ready());
}

void test_hybrid_model_graph(const char * model_path) {
    const llama_model_tensor_buft_override overrides[] = {
        { "ffn_(gate|up|down)_exps\\.weight", ggml_backend_cpu_buffer_type() },
        { nullptr, nullptr },
    };
    auto params = llama_model_default_params();
    params.n_gpu_layers = -1;
    params.tensor_buft_overrides = overrides;
    params.expert_weights_mode = LLAMA_EXPERT_WEIGHTS_MODE_COLD_CACHE;
    params.expert_hot_cache_capacity = 16;
    params.expert_cold_cache_bytes = 64U*1024U*1024U;
    params.expert_transfer_ring_bytes = 16U*1024U*1024U;
    params.expert_miss_policy = LLAMA_EXPERT_MISS_POLICY_CPU_FALLBACK;
    llama_model_ptr model(llama_model_load_from_file(model_path, params));
    GGML_ASSERT(model);
    auto * provider = model->expert_weight_provider();
    GGML_ASSERT(provider);
    auto diagnostics = provider->hot_cache_diagnostics();
    GGML_ASSERT(diagnostics.configured_miss_policy == LLAMA_EXPERT_MISS_POLICY_CPU_FALLBACK);

    auto context_params = llama_context_default_params();
    context_params.n_ctx = 64;
    context_params.n_batch = 64;
    context_params.n_ubatch = 1;
    llama_context_ptr context(llama_init_from_model(model.get(), context_params));
    GGML_ASSERT(context);
    diagnostics = provider->hot_cache_diagnostics();
    GGML_ASSERT(diagnostics.hybrid_bindings > 0);
}

} // namespace

int main(int argc, char ** argv) {
    test_defaults_and_lane_validation();
    test_model_parameter_validation_and_copy();
    ggml_backend_load_all();
    const std::vector<int32_t> mixed = { 0, -1, 2, 1, 1, -1 };
    const std::vector<int32_t> active = { 0, 1, 2, 1, 1, 0 };
    const std::vector<int32_t> inactive = { -1, -1, -1, -1, -1, -1 };
    const std::vector<int32_t> mixed_fusion_shape = { 0, -1, 2 };
    const std::vector<int32_t> active_fast_path = { 0, 1, 1 };
    auto * cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    GGML_ASSERT(cpu);
    assert_case(cpu, mixed);
    assert_case(cpu, active);
    assert_case(cpu, inactive);
    assert_case(cpu, mixed_fusion_shape);
    assert_case(cpu, active_fast_path, false);
    if (auto * gpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU)) {
        assert_case(gpu, mixed);
        assert_case(gpu, active);
        assert_case(gpu, inactive);
        assert_case(gpu, mixed_fusion_shape);
        assert_case(gpu, active_fast_path, false);
    }
    test_hybrid_binding();
    if (argc > 1) test_hybrid_model_graph(argv[1]);
    return 0;
}
