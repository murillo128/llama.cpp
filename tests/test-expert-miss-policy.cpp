#include "llama-expert-weight-provider.h"
#include "llama-context.h"
#include "llama-model.h"
#include "llama-cpp.h"

#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <limits>
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

bool abort_immediately(void *) {
    return true;
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

void test_auto_evaluator() {
    auto cost = valid_cost_model();
    cost.cpu_fixed_decode_ns = 1;
    cost.cpu_per_lane_decode_ns = 1;
    cost.gpu_fixed_decode_ns = 1000;
    cost.gpu_per_lane_decode_ns = 1000;
    cost.h2d_fixed_ns = 1000;
    cost.h2d_bytes_per_second = 1000000000ULL;
    cost.decision_hysteresis_ns = 1;
    llm_expert_auto_input input = { cost, false, 2, 100, 0, 0, 0, false,
        llm_expert_same_key_h2d_state::none, 0 };
    auto result = llm_evaluate_expert_auto(input);
    GGML_ASSERT(result.backend == llm_expert_execution_backend::cpu);
    GGML_ASSERT(result.reason == llm_expert_auto_reason::cpu_faster && !result.overflow);

    input.cost.cpu_fixed_decode_ns = 10000;
    input.cost.cpu_per_lane_decode_ns = 10000;
    input.cost.gpu_fixed_decode_ns = 1;
    input.cost.gpu_per_lane_decode_ns = 1;
    input.cost.h2d_fixed_ns = 1;
    result = llm_evaluate_expert_auto(input);
    GGML_ASSERT(result.backend == llm_expert_execution_backend::gpu);
    GGML_ASSERT(result.reason == llm_expert_auto_reason::gpu_faster_or_hysteresis);

    input = {};
    input.cost = valid_cost_model();
    input.cost.cpu_fixed_decode_ns = 2;
    input.cost.cpu_per_lane_decode_ns = 1;
    input.cost.gpu_fixed_decode_ns = 1;
    input.cost.gpu_per_lane_decode_ns = 1;
    input.cost.h2d_fixed_ns = 1;
    input.cost.h2d_bytes_per_second = 1000000000ULL;
    input.cost.decision_hysteresis_ns = 1;
    input.lanes = 1;
    result = llm_evaluate_expert_auto(input);
    GGML_ASSERT(result.cpu_finish_ns == 3 && result.gpu_finish_ns == 3);
    GGML_ASSERT(result.backend == llm_expert_execution_backend::gpu);
    GGML_ASSERT(result.reason == llm_expert_auto_reason::tie);

    input.lanes = std::numeric_limits<uint64_t>::max();
    result = llm_evaluate_expert_auto(input);
    GGML_ASSERT(result.backend == llm_expert_execution_backend::gpu && result.overflow);
    GGML_ASSERT(result.reason == llm_expert_auto_reason::overflow);

    input.lanes = 2;
    input.bundle_bytes = 100;
    input.same_key_h2d_present = true;
    input.same_key_h2d_state = llm_expert_same_key_h2d_state::h2d_in_flight;
    input.same_key_h2d_remaining_bytes = 7;
    input.queued_cpu_work_ns = 11;
    input.queued_h2d_work_ns = 13;
    input.queued_gpu_work_ns = 17;
    const auto first = llm_evaluate_expert_auto(input);
    const auto second = llm_evaluate_expert_auto(input);
    GGML_ASSERT(first.backend == second.backend && first.reason == second.reason &&
        first.cpu_finish_ns == second.cpu_finish_ns && first.gpu_finish_ns == second.gpu_finish_ns);
    GGML_ASSERT(first.h2d_work_ns == input.cost.h2d_fixed_ns + 7);

    input.same_key_h2d_state = llm_expert_same_key_h2d_state::queued_or_staging;
    input.same_key_h2d_remaining_bytes = input.bundle_bytes;
    GGML_ASSERT(llm_evaluate_expert_auto(input).h2d_work_ns == input.cost.h2d_fixed_ns + input.bundle_bytes);
    input.same_key_h2d_state = llm_expert_same_key_h2d_state::h2d_complete_unpublished;
    input.same_key_h2d_remaining_bytes = 0;
    GGML_ASSERT(llm_evaluate_expert_auto(input).h2d_work_ns == 0);
    input.same_key_h2d_present = false;
    input.same_key_h2d_state = llm_expert_same_key_h2d_state::none;
    GGML_ASSERT(llm_evaluate_expert_auto(input).h2d_work_ns == input.cost.h2d_fixed_ns + input.bundle_bytes);
    input.same_key_h2d_present = true;
    GGML_ASSERT(llm_evaluate_expert_auto(input).overflow);
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
    ggml_backend_synchronize(backend.get());
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

void assert_logit_gate(const std::vector<float> & expected, const float * actual) {
    GGML_ASSERT(actual != nullptr && !expected.empty());
    double max_abs = 0.0;
    double mean_abs = 0.0;
    double dot = 0.0;
    double expected_norm = 0.0;
    double actual_norm = 0.0;
    std::vector<int32_t> expected_order(expected.size());
    std::vector<int32_t> actual_order(expected.size());
    for (size_t index = 0; index < expected.size(); ++index) {
        expected_order[index] = int32_t(index);
        actual_order[index] = int32_t(index);
        const double lhs = expected[index];
        const double rhs = actual[index];
        const double difference = std::fabs(lhs - rhs);
        max_abs = std::max(max_abs, difference);
        mean_abs += difference;
        dot += lhs*rhs;
        expected_norm += lhs*lhs;
        actual_norm += rhs*rhs;
    }
    const auto expected_compare = [&](int32_t lhs, int32_t rhs) {
        return expected[lhs] != expected[rhs] ? expected[lhs] > expected[rhs] : lhs < rhs;
    };
    const auto actual_compare = [&](int32_t lhs, int32_t rhs) {
        return actual[lhs] != actual[rhs] ? actual[lhs] > actual[rhs] : lhs < rhs;
    };
    std::partial_sort(expected_order.begin(), expected_order.begin() + 10, expected_order.end(), expected_compare);
    std::partial_sort(actual_order.begin(), actual_order.begin() + 10, actual_order.end(), actual_compare);
    if (!std::equal(expected_order.begin(), expected_order.begin() + 10, actual_order.begin())) {
        std::fprintf(stderr, "LOGIT_MISMATCH max_abs=%.9f expected_top=%d,%d,%d actual_top=%d,%d,%d\n",
            max_abs, expected_order[0], expected_order[1], expected_order[2],
            actual_order[0], actual_order[1], actual_order[2]);
    }
    GGML_ASSERT(std::equal(expected_order.begin(), expected_order.begin() + 10, actual_order.begin()));
    mean_abs /= expected.size();
    const double cosine = dot/std::sqrt(expected_norm*actual_norm);
    GGML_ASSERT(max_abs <= 0.10);
    GGML_ASSERT(mean_abs <= 0.02);
    GGML_ASSERT(cosine >= 0.999);
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
    GGML_ASSERT(binding.checkpoint_ids != binding.execution_ids);
    GGML_ASSERT(binding.cpu_up.weight != nullptr && binding.cpu_gate.weight != nullptr && binding.cpu_down.weight != nullptr);
    ggml_backend_buffer_ptr graph_buffer(
        ggml_backend_alloc_ctx_tensors_from_buft(graph_ctx.get(), ggml_backend_cpu_buffer_type()));
    GGML_ASSERT(graph_buffer);

    ggml_backend_ptr backend(ggml_backend_dev_init(config.target_device, nullptr));
    GGML_ASSERT(backend);
    const int32_t prime_ids[] = { 0, 0 };
    ggml_backend_tensor_set(binding.checkpoint_ids, prime_ids, 0, sizeof(prime_ids));
    GGML_ASSERT(provider->debug_set_miss_policy_for_testing(
        LLAMA_EXPERT_MISS_POLICY_PROMOTE_AND_GPU).is_ready());
    llm_expert_execution_plan plan;
    GGML_ASSERT(provider->prepare({ binding }, plan).is_ready());
    GGML_ASSERT(provider->remap_checkpoint_tensor(binding, backend.get()).is_ready());
    plan.reset();

    auto cpu_auto_cost = valid_cost_model();
    cpu_auto_cost.cpu_fixed_decode_ns = 1;
    cpu_auto_cost.cpu_per_lane_decode_ns = 1;
    cpu_auto_cost.gpu_fixed_decode_ns = 1000000;
    cpu_auto_cost.gpu_per_lane_decode_ns = 1000000;
    cpu_auto_cost.h2d_fixed_ns = 1000000;
    cpu_auto_cost.h2d_bytes_per_second = 1;
    const int32_t hot_plus_miss_ids[] = { 0, 1 };
    ggml_backend_tensor_set(binding.checkpoint_ids, hot_plus_miss_ids, 0, sizeof(hot_plus_miss_ids));
    GGML_ASSERT(provider->debug_set_auto_cost_model_for_testing(cpu_auto_cost).is_ready());
    GGML_ASSERT(provider->prepare({ binding }, plan).is_ready());
    GGML_ASSERT(provider->remap_checkpoint_tensor(binding, backend.get()).is_ready());
    auto auto_live = provider->hot_cache_diagnostics();
    GGML_ASSERT(!auto_live.auto_decisions.empty());
    const auto & hot_queued = auto_live.auto_decisions.back();
    GGML_ASSERT(hot_queued.expert == 1);
    GGML_ASSERT(hot_queued.queued_gpu_work_ns ==
        cpu_auto_cost.gpu_fixed_decode_ns + cpu_auto_cost.gpu_per_lane_decode_ns);
    plan.reset();

    const int32_t queued_miss_ids[] = { 2, 3 };
    ggml_backend_tensor_set(binding.checkpoint_ids, queued_miss_ids, 0, sizeof(queued_miss_ids));
    const uint64_t first_request = auto_live.auto_decisions.back().request + 1;
    GGML_ASSERT(provider->prepare({ binding }, plan).is_ready());
    GGML_ASSERT(provider->remap_checkpoint_tensor(binding, backend.get()).is_ready());
    auto_live = provider->hot_cache_diagnostics();
    std::vector<llm_hot_cache_diagnostics::auto_decision> queued_records;
    for (const auto & record : auto_live.auto_decisions) {
        if (record.request == first_request && record.layer == 0) queued_records.push_back(record);
    }
    GGML_ASSERT(queued_records.size() == 2);
    GGML_ASSERT(queued_records[0].expert == 2 && queued_records[1].expert == 3);
    GGML_ASSERT(queued_records[0].queued_cpu_work_ns == 0);
    GGML_ASSERT(queued_records[1].queued_cpu_work_ns == queued_records[0].cpu_finish_ns);
    plan.reset();

    const int32_t mixed_ids[] = { 0, 1 };
    ggml_backend_tensor_set(binding.checkpoint_ids, mixed_ids, 0, sizeof(mixed_ids));
    GGML_ASSERT(provider->debug_set_miss_policy_for_testing(
        LLAMA_EXPERT_MISS_POLICY_CPU_FALLBACK).is_ready());
    GGML_ASSERT(provider->prepare({ binding }, plan).is_ready());
    GGML_ASSERT(provider->remap_checkpoint_tensor(binding, backend.get()).is_ready());
    int32_t gpu_ids[2] = {};
    int32_t cpu_ids[2] = {};
    ggml_backend_tensor_get(binding.checkpoint_ids, gpu_ids, 0, sizeof(gpu_ids));
    ggml_backend_tensor_get(binding.cpu_execution_ids, cpu_ids, 0, sizeof(cpu_ids));
    GGML_ASSERT(gpu_ids[0] >= 0 && gpu_ids[1] == -1);
    GGML_ASSERT(cpu_ids[0] == -1 && cpu_ids[1] >= 0);
    GGML_ASSERT(llm_validate_hybrid_execution_ids(gpu_ids, cpu_ids, 2, 2, 2).is_ready());
    auto live = provider->hot_cache_diagnostics();
    GGML_ASSERT(live.cold_current_cpu_execution_refs == 1);
    GGML_ASSERT(live.mixed_execution_layers >= 1);
    GGML_ASSERT(provider->trim().error == llm_expert_provider_error::busy);
    GGML_ASSERT(provider->surrender().error == llm_expert_provider_error::busy);
    plan.reset();
    live = provider->hot_cache_diagnostics();
    GGML_ASSERT(live.cold_current_cpu_execution_refs == 0);

    const int32_t cancelled_ids[] = { 2, 3 };
    ggml_backend_tensor_set(binding.checkpoint_ids, cancelled_ids, 0, sizeof(cancelled_ids));
    GGML_ASSERT(provider->prepare({ binding }, plan).is_ready());
    const auto cancelled = provider->remap_checkpoint_tensor(
        binding, backend.get(), abort_immediately, nullptr);
    GGML_ASSERT(cancelled.status == llm_expert_provider_status::cancelled);
    plan.reset();
    GGML_ASSERT(provider->hot_cache_diagnostics().cold_current_cpu_execution_refs == 0);
    GGML_ASSERT(provider->trim().is_ready());
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

    GGML_ASSERT(provider->debug_set_miss_policy_for_testing(
        LLAMA_EXPERT_MISS_POLICY_PROMOTE_AND_GPU).is_ready());
    llama_token token = 1;
    GGML_ASSERT(llama_decode(context.get(), llama_batch_get_one(&token, 1)) == 0);
    llama_synchronize(context.get());
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
    const float * promote_logits_ptr = llama_get_logits_ith(context.get(), -1);
    GGML_ASSERT(promote_logits_ptr != nullptr && n_vocab > 0);
    const std::vector<float> promote_logits(promote_logits_ptr, promote_logits_ptr + n_vocab);
    GGML_ASSERT(provider->debug_set_miss_policy_for_testing(
        LLAMA_EXPERT_MISS_POLICY_CPU_FALLBACK).is_ready());
    token = 2;
    GGML_ASSERT(llama_decode(context.get(), llama_batch_get_one(&token, 1)) == 0);
    llama_synchronize(context.get());
    diagnostics = provider->hot_cache_diagnostics();
    GGML_ASSERT(diagnostics.cpu_execution_lanes > 0);
    GGML_ASSERT(diagnostics.gpu_execution_lanes > 0);
    GGML_ASSERT(diagnostics.mixed_execution_layers > 0);
    GGML_ASSERT(diagnostics.cold_current_cpu_execution_refs == 0);
    GGML_ASSERT(diagnostics.h2d_bytes_avoided_for_current_output > 0);
    GGML_ASSERT(diagnostics.background_submitted == 0 && diagnostics.background_h2d_bytes == 0);

    context.reset();
    GGML_ASSERT(provider->trim().is_ready());
    context.reset(llama_init_from_model(model.get(), context_params));
    GGML_ASSERT(context);
    token = 1;
    GGML_ASSERT(llama_decode(context.get(), llama_batch_get_one(&token, 1)) == 0);
    llama_synchronize(context.get());
    const float * cpu_logits = llama_get_logits_ith(context.get(), -1);
    GGML_ASSERT(cpu_logits != nullptr);
    assert_logit_gate(promote_logits, cpu_logits);

    context.reset();
    GGML_ASSERT(provider->trim().is_ready());
    auto prefill_params = context_params;
    prefill_params.n_batch = 4;
    prefill_params.n_ubatch = 4;
    GGML_ASSERT(provider->debug_set_miss_policy_for_testing(
        LLAMA_EXPERT_MISS_POLICY_PROMOTE_AND_GPU).is_ready());
    context.reset(llama_init_from_model(model.get(), prefill_params));
    GGML_ASSERT(context);
    llama_token prefill_tokens[] = { 1, 2, 3, 4 };
    GGML_ASSERT(llama_decode(context.get(), llama_batch_get_one(prefill_tokens, 4)) == 0);
    llama_synchronize(context.get());
    const float * promote_prefill_ptr = llama_get_logits_ith(context.get(), -1);
    GGML_ASSERT(promote_prefill_ptr != nullptr);
    const std::vector<float> promote_prefill(promote_prefill_ptr, promote_prefill_ptr + n_vocab);
    context.reset();
    GGML_ASSERT(provider->trim().is_ready());
    GGML_ASSERT(provider->debug_set_miss_policy_for_testing(
        LLAMA_EXPERT_MISS_POLICY_CPU_FALLBACK).is_ready());
    context.reset(llama_init_from_model(model.get(), prefill_params));
    GGML_ASSERT(context);
    GGML_ASSERT(llama_decode(context.get(), llama_batch_get_one(prefill_tokens, 4)) == 0);
    llama_synchronize(context.get());
    assert_logit_gate(promote_prefill, llama_get_logits_ith(context.get(), -1));

    context.reset();
    model.reset();

    const auto run_auto = [&](llama_expert_auto_cost_model cost, bool expect_cpu) {
        auto automatic_params = params;
        automatic_params.expert_miss_policy = LLAMA_EXPERT_MISS_POLICY_AUTO;
        automatic_params.expert_auto_cost_model = &cost;
        llama_model_ptr automatic_model(llama_model_load_from_file(model_path, automatic_params));
        GGML_ASSERT(automatic_model);
        llama_context_ptr automatic_context(llama_init_from_model(automatic_model.get(), context_params));
        GGML_ASSERT(automatic_context);
        llama_token automatic_token = 1;
        GGML_ASSERT(llama_decode(
            automatic_context.get(), llama_batch_get_one(&automatic_token, 1)) == 0);
        llama_synchronize(automatic_context.get());
        const auto automatic_diagnostics =
            automatic_model->expert_weight_provider()->hot_cache_diagnostics();
        if (!expect_cpu) {
            for (const auto & slot : automatic_diagnostics.slots) {
                if (slot.state != llm_hot_cache_diagnostics::slot::ready || slot.layer < 0) continue;
                std::vector<uint8_t> cold_bytes;
                std::vector<uint8_t> hot_bytes;
                const llm_expert_key key = { slot.layer, slot.expert };
                GGML_ASSERT(automatic_model->expert_weight_provider()->debug_copy_cold_bundle(
                    key, cold_bytes).is_ready());
                GGML_ASSERT(automatic_model->expert_weight_provider()->debug_copy_hot_bundle(
                    key, hot_bytes).is_ready());
                GGML_ASSERT(cold_bytes == hot_bytes);
            }
        }
        assert_logit_gate(promote_logits, llama_get_logits_ith(automatic_context.get(), -1));
        GGML_ASSERT(automatic_diagnostics.auto_decision_records > 0);
        GGML_ASSERT(automatic_diagnostics.auto_decision_digest != 1469598103934665603ULL);
        GGML_ASSERT(!automatic_diagnostics.auto_decisions.empty());
        for (size_t index = 0; index < automatic_diagnostics.auto_decisions.size(); ++index) {
            const auto & record = automatic_diagnostics.auto_decisions[index];
            GGML_ASSERT(!record.same_key_h2d_present &&
                record.same_key_h2d_state == uint8_t(llm_expert_same_key_h2d_state::none) &&
                record.same_key_h2d_remaining_bytes == 0);
            const llm_expert_auto_input input = {
                record.cost,
                record.prefill,
                record.lanes,
                record.bundle_bytes,
                record.queued_cpu_work_ns,
                record.queued_h2d_work_ns,
                record.queued_gpu_work_ns,
                record.same_key_h2d_present,
                llm_expert_same_key_h2d_state(record.same_key_h2d_state),
                record.same_key_h2d_remaining_bytes,
            };
            const auto replay = llm_evaluate_expert_auto(input);
            GGML_ASSERT(uint8_t(replay.backend) == record.backend);
            GGML_ASSERT(uint8_t(replay.reason) == record.reason);
            GGML_ASSERT(replay.cpu_work_ns == record.cpu_work_ns);
            GGML_ASSERT(replay.h2d_work_ns == record.h2d_work_ns);
            GGML_ASSERT(replay.gpu_work_ns == record.gpu_work_ns);
            GGML_ASSERT(replay.cpu_finish_ns == record.cpu_finish_ns);
            GGML_ASSERT(replay.gpu_finish_ns == record.gpu_finish_ns);
            GGML_ASSERT(replay.overflow == record.overflow);
            if (index != 0) {
                const auto & previous = automatic_diagnostics.auto_decisions[index - 1];
                if (previous.request == record.request && previous.layer == record.layer) {
                    GGML_ASSERT(previous.expert < record.expert);
                }
            }
        }
        if (expect_cpu) {
            GGML_ASSERT(automatic_diagnostics.auto_cpu_decisions > 0);
            GGML_ASSERT(automatic_diagnostics.auto_gpu_decisions == 0);
        } else {
            GGML_ASSERT(automatic_diagnostics.auto_gpu_decisions > 0);
            GGML_ASSERT(automatic_diagnostics.auto_cpu_decisions == 0);
        }
    };

    auto cpu_cost = valid_cost_model();
    cpu_cost.cpu_fixed_decode_ns = 1;
    cpu_cost.cpu_per_lane_decode_ns = 1;
    cpu_cost.gpu_fixed_decode_ns = 1000000000ULL;
    cpu_cost.gpu_per_lane_decode_ns = 1000000000ULL;
    cpu_cost.h2d_fixed_ns = 1000000000ULL;
    cpu_cost.h2d_bytes_per_second = 1;
    cpu_cost.decision_hysteresis_ns = 1;
    run_auto(cpu_cost, true);

    auto gpu_cost = valid_cost_model();
    gpu_cost.cpu_fixed_decode_ns = 1000000000000ULL;
    gpu_cost.cpu_per_lane_decode_ns = 1000000000000ULL;
    gpu_cost.gpu_fixed_decode_ns = 1;
    gpu_cost.gpu_per_lane_decode_ns = 1;
    gpu_cost.h2d_fixed_ns = 1;
    gpu_cost.h2d_bytes_per_second = UINT64_MAX;
    gpu_cost.decision_hysteresis_ns = 1;
    run_auto(gpu_cost, false);

    auto background_params = params;
    background_params.expert_background_promotion = true;
    llama_model_ptr background_model(llama_model_load_from_file(model_path, background_params));
    GGML_ASSERT(background_model);
    llama_context_ptr background_context(llama_init_from_model(background_model.get(), context_params));
    GGML_ASSERT(background_context);
    llama_token background_token = 1;
    GGML_ASSERT(llama_decode(
        background_context.get(), llama_batch_get_one(&background_token, 1)) == 0);
    llama_synchronize(background_context.get());
    GGML_ASSERT(background_model->expert_weight_provider()->hot_cache_diagnostics().background_submitted > 0);
    background_context.reset();
    background_context.reset(llama_init_from_model(background_model.get(), context_params));
    GGML_ASSERT(background_context);
    for (background_token = 1; background_token <= 3; ++background_token) {
        GGML_ASSERT(llama_decode(
            background_context.get(), llama_batch_get_one(&background_token, 1)) == 0);
        llama_synchronize(background_context.get());
    }
    const auto background_diagnostics =
        background_model->expert_weight_provider()->hot_cache_diagnostics();
    GGML_ASSERT(background_diagnostics.background_submitted > 0);
    GGML_ASSERT(background_diagnostics.background_h2d_bytes > 0);
    GGML_ASSERT(background_diagnostics.active_background_flights <=
        background_diagnostics.effective_capacity);
    GGML_ASSERT(background_diagnostics.background_useful +
        background_diagnostics.background_later_joins > 0);

    background_context.reset();
    background_model.reset();

    // A blocked background transfer must neither delay the CPU-served output nor
    // grow beyond the configured cache/ring bounds.  The unrecorded event holds
    // the transfer stream while llama_synchronize() completes the current token.
    auto * gpu_device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    GGML_ASSERT(gpu_device != nullptr);
    ggml_backend_ptr gate_backend(ggml_backend_dev_init(gpu_device, nullptr));
    GGML_ASSERT(gate_backend);
    ggml_backend_event_t gate_event = ggml_backend_event_new(gpu_device);
    GGML_ASSERT(gate_event != nullptr);
    constexpr size_t gate_copy_bytes = 256U*1024U*1024U;
    ggml_init_params gate_params = { ggml_tensor_overhead()*2, nullptr, true };
    ggml_context_ptr gate_ctx(ggml_init(gate_params));
    GGML_ASSERT(gate_ctx);
    ggml_tensor * gate_tensor = ggml_new_tensor_1d(gate_ctx.get(), GGML_TYPE_I8, gate_copy_bytes);
    ggml_backend_buffer_ptr gate_buffer(
        ggml_backend_alloc_ctx_tensors_from_buft(gate_ctx.get(), ggml_backend_dev_buffer_type(gpu_device)));
    GGML_ASSERT(gate_buffer);
    ggml_backend_buffer_ptr gate_source(ggml_backend_buft_alloc_buffer(
        ggml_backend_dev_host_buffer_type(gpu_device), gate_copy_bytes));
    GGML_ASSERT(gate_source && ggml_backend_buffer_is_host(gate_source.get()));
    ggml_backend_buffer_clear(gate_source.get(), 0);
    llama_model_ptr gated_model(llama_model_load_from_file(model_path, background_params));
    GGML_ASSERT(gated_model);
    auto * gated_provider = gated_model->expert_weight_provider();
    GGML_ASSERT(gated_provider != nullptr);
    llama_context_ptr gated_context(llama_init_from_model(gated_model.get(), prefill_params));
    GGML_ASSERT(gated_context);
    for (int repetition = 0; repetition < 64; ++repetition) {
        ggml_backend_tensor_set_async(gate_backend.get(), gate_tensor,
            ggml_backend_buffer_get_base(gate_source.get()), 0, gate_copy_bytes);
    }
    ggml_backend_event_record(gate_event, gate_backend.get());
    GGML_ASSERT(gated_provider->set_h2d_gate_event_for_testing(gate_event).is_ready());
    llama_token gated_tokens[] = { 1, 2, 3, 4 };
    GGML_ASSERT(llama_decode(gated_context.get(), llama_batch_get_one(gated_tokens, 4)) == 0);
    llama_synchronize(gated_context.get());
    const auto gated_diagnostics = gated_provider->hot_cache_diagnostics();
    GGML_ASSERT(gated_diagnostics.background_submitted > 0);
    GGML_ASSERT(gated_diagnostics.active_background_flights > 0);
    GGML_ASSERT(gated_diagnostics.active_background_flights <= gated_diagnostics.effective_capacity);
    GGML_ASSERT(gated_diagnostics.peak_background_flights <= gated_diagnostics.effective_capacity);
    GGML_ASSERT(gated_diagnostics.background_busy + gated_diagnostics.background_dropped > 0);
    GGML_ASSERT(gated_diagnostics.background_wasted == 0);
    GGML_ASSERT(gated_diagnostics.ring_h2d_event_waits == 0);

    GGML_ASSERT(gated_provider->debug_set_auto_cost_model_for_testing(gpu_cost).is_ready());
    llama_memory_clear(llama_get_memory(gated_context.get()), true);
    llama_token joined_token = 1;
    GGML_ASSERT(llama_decode(gated_context.get(), llama_batch_get_one(&joined_token, 1)) == 0);
    llama_synchronize(gated_context.get());
    const auto joined_diagnostics = gated_provider->hot_cache_diagnostics();
    GGML_ASSERT(joined_diagnostics.background_later_joins > 0);
    GGML_ASSERT(joined_diagnostics.active_background_flights == 0);
    bool saw_same_key_submitted = false;
    bool saw_background_backlog = false;
    bool saw_hot_gpu_queue = false;
    for (const auto & record : joined_diagnostics.auto_decisions) {
        saw_same_key_submitted = saw_same_key_submitted ||
            (record.same_key_h2d_present &&
             record.same_key_h2d_state == uint8_t(llm_expert_same_key_h2d_state::h2d_in_flight) &&
             record.same_key_h2d_remaining_bytes == record.bundle_bytes);
        saw_background_backlog = saw_background_backlog || record.queued_h2d_work_ns > 0;
        saw_hot_gpu_queue = saw_hot_gpu_queue || record.queued_gpu_work_ns > 0;
    }
    GGML_ASSERT(saw_same_key_submitted);
    GGML_ASSERT(saw_background_backlog);
    GGML_ASSERT(saw_hot_gpu_queue);
    GGML_ASSERT(joined_diagnostics.background_completed == gated_diagnostics.background_submitted);

    ggml_backend_synchronize(gate_backend.get());
    gated_context.reset();
    GGML_ASSERT(gated_provider->trim().is_ready());
    const auto wasted_diagnostics = gated_provider->hot_cache_diagnostics();
    GGML_ASSERT(wasted_diagnostics.background_useful + wasted_diagnostics.background_wasted ==
        gated_diagnostics.background_submitted);
    GGML_ASSERT(wasted_diagnostics.background_useful > 0);
    GGML_ASSERT(wasted_diagnostics.background_wasted > 0);
    gated_model.reset();
    ggml_backend_event_free(gate_event);
}

} // namespace

int main(int argc, char ** argv) {
    test_defaults_and_lane_validation();
    test_auto_evaluator();
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
