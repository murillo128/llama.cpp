#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

enum class probe_mode { cpu, gpu, mixed };

struct checkpoint_callback_data {
    ggml_tensor * checkpoint;
    ggml_tensor * cpu_ids;
    const int32_t * gpu_values;
    const int32_t * cpu_values;
    size_t bytes;
    bool fired;
};

bool checkpoint_callback(ggml_tensor * tensor, bool ask, void * user_data) {
    auto * data = static_cast<checkpoint_callback_data *>(user_data);
    if (tensor != data->checkpoint) {
        return false;
    }
    if (ask) {
        return true;
    }
    ggml_backend_tensor_set(data->checkpoint, data->gpu_values, 0, data->bytes);
    ggml_backend_tensor_set(data->cpu_ids, data->cpu_values, 0, data->bytes);
    data->fired = true;
    return true;
}

uint64_t elapsed_us(std::chrono::steady_clock::time_point begin) {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - begin).count();
}

uint64_t measure(
        probe_mode mode,
        ggml_backend_t cpu,
        ggml_backend_t gpu,
        int64_t extent) {
    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead()*12 + ggml_graph_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    GGML_ASSERT(ctx);

    ggml_tensor * cpu_result = nullptr;
    ggml_tensor * gpu_result = nullptr;
    ggml_tensor * output = nullptr;
    ggml_tensor * cpu_weights = nullptr;
    ggml_tensor * cpu_input = nullptr;
    ggml_tensor * cpu_ids = nullptr;
    ggml_tensor * gpu_weights = nullptr;
    ggml_tensor * gpu_input = nullptr;
    ggml_tensor * gpu_ids = nullptr;
    ggml_tensor * logical_ids = nullptr;
    constexpr int64_t n_expert_used = 2;
    constexpr int64_t n_tokens = 64;
    if (mode != probe_mode::gpu) {
        cpu_weights = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, extent, extent, 2);
        cpu_input = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, extent, n_expert_used, n_tokens);
        cpu_ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, n_expert_used, n_tokens);
        cpu_result = ggml_mul_mat_id(ctx.get(), cpu_weights, cpu_input, cpu_ids);
        cpu_result->op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1] = 1;
        output = cpu_result;
    }
    if (mode != probe_mode::cpu) {
        gpu_weights = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, extent, extent, 2);
        gpu_input = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, extent, n_expert_used, n_tokens);
        if (mode == probe_mode::mixed) {
            logical_ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, n_expert_used, n_tokens);
            gpu_ids = ggml_dup(ctx.get(), logical_ids);
        } else {
            gpu_ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, n_expert_used, n_tokens);
        }
        gpu_result = ggml_mul_mat_id(ctx.get(), gpu_weights, gpu_input, gpu_ids);
        gpu_result->op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1] = 1;
        output = gpu_result;
    }
    if (mode == probe_mode::mixed) {
        output = ggml_add(ctx.get(), gpu_result, cpu_result);
    }

    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, output);
    ggml_backend_t backends[] = { gpu, cpu };
    ggml_backend_sched_ptr sched(ggml_backend_sched_new(
        backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true));
    GGML_ASSERT(sched);
    if (cpu_result != nullptr) {
        ggml_backend_sched_set_tensor_backend(sched.get(), cpu_weights, cpu);
        ggml_backend_sched_set_tensor_backend(sched.get(), cpu_input, cpu);
        ggml_backend_sched_set_tensor_backend(sched.get(), cpu_ids, cpu);
        ggml_backend_sched_set_tensor_backend(sched.get(), cpu_result, cpu);
    }
    if (gpu_result != nullptr) {
        ggml_backend_sched_set_tensor_backend(sched.get(), gpu_weights, gpu);
        ggml_backend_sched_set_tensor_backend(sched.get(), gpu_input, gpu);
        ggml_backend_sched_set_tensor_backend(sched.get(), gpu_ids, gpu);
        ggml_backend_sched_set_tensor_backend(sched.get(), gpu_result, gpu);
    }
    if (mode == probe_mode::mixed) {
        ggml_backend_sched_set_tensor_backend(sched.get(), output, gpu);
    }
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched.get(), graph));

    std::vector<float> weights(size_t(2*extent*extent), 0.001f);
    std::vector<float> input(size_t(extent*n_expert_used*n_tokens), 0.001f);
    std::vector<int32_t> cpu_id_values(size_t(n_expert_used*n_tokens));
    std::vector<int32_t> gpu_id_values(size_t(n_expert_used*n_tokens));
    std::vector<int32_t> logical_id_values(size_t(n_expert_used*n_tokens));
    for (int64_t token = 0; token < n_tokens; ++token) {
        gpu_id_values[size_t(token*n_expert_used)] = 0;
        gpu_id_values[size_t(token*n_expert_used + 1)] = -1;
        cpu_id_values[size_t(token*n_expert_used)] = -1;
        cpu_id_values[size_t(token*n_expert_used + 1)] = 1;
        logical_id_values[size_t(token*n_expert_used)] = 0;
        logical_id_values[size_t(token*n_expert_used + 1)] = 1;
    }
    for (ggml_tensor * tensor : { cpu_weights, gpu_weights }) {
        if (tensor != nullptr) ggml_backend_tensor_set(tensor, weights.data(), 0, weights.size()*sizeof(float));
    }
    for (ggml_tensor * tensor : { cpu_input, gpu_input }) {
        if (tensor != nullptr) ggml_backend_tensor_set(tensor, input.data(), 0, input.size()*sizeof(float));
    }
    if (cpu_ids != nullptr) ggml_backend_tensor_set(cpu_ids, cpu_id_values.data(), 0, cpu_id_values.size()*sizeof(int32_t));
    if (logical_ids != nullptr) {
        ggml_backend_tensor_set(logical_ids, logical_id_values.data(), 0, logical_id_values.size()*sizeof(int32_t));
    } else if (gpu_ids != nullptr) {
        ggml_backend_tensor_set(gpu_ids, gpu_id_values.data(), 0, gpu_id_values.size()*sizeof(int32_t));
    }

    checkpoint_callback_data callback_data = {
        gpu_ids, cpu_ids, gpu_id_values.data(), cpu_id_values.data(),
        gpu_id_values.size()*sizeof(int32_t), false,
    };
    if (mode == probe_mode::mixed) {
        ggml_backend_sched_set_eval_callback(sched.get(), checkpoint_callback, &callback_data);
    }
    GGML_ASSERT(ggml_backend_sched_graph_compute_async(sched.get(), graph) == GGML_STATUS_SUCCESS);
    ggml_backend_sched_synchronize(sched.get());
    GGML_ASSERT(mode != probe_mode::mixed || callback_data.fired);

    std::vector<uint64_t> samples;
    for (int repetition = 0; repetition < 3; ++repetition) {
        callback_data.fired = false;
        const auto begin = std::chrono::steady_clock::now();
        GGML_ASSERT(ggml_backend_sched_graph_compute_async(sched.get(), graph) == GGML_STATUS_SUCCESS);
        ggml_backend_sched_synchronize(sched.get());
        GGML_ASSERT(mode != probe_mode::mixed || callback_data.fired);
        samples.push_back(elapsed_us(begin));
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size()/2];
}

} // namespace

int main() {
    ggml_backend_load_all();
    auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    auto * gpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (cpu_dev == nullptr || gpu_dev == nullptr) {
        std::puts("PHASE8_OVERLAP\tstatus=skip\treason=cpu-or-gpu-unavailable");
        return 0;
    }
    ggml_backend_ptr cpu(ggml_backend_dev_init(cpu_dev, nullptr));
    ggml_backend_ptr gpu(ggml_backend_dev_init(gpu_dev, nullptr));
    GGML_ASSERT(cpu && gpu);

    constexpr int64_t extent = 3072;
    const uint64_t cpu_us = measure(probe_mode::cpu, cpu.get(), gpu.get(), extent);
    const uint64_t gpu_us = measure(probe_mode::gpu, cpu.get(), gpu.get(), extent);
    const uint64_t mixed_us = measure(probe_mode::mixed, cpu.get(), gpu.get(), extent);
    const uint64_t sequential_us = cpu_us + gpu_us;
    const uint64_t overlap_us = sequential_us > mixed_us ? sequential_us - mixed_us : 0;
    std::printf("PHASE8_OVERLAP\tstatus=%s\textent=%lld\tcpu_us=%llu\tgpu_us=%llu\tmixed_us=%llu\tsequential_us=%llu\toverlap_us=%llu\n",
        overlap_us > 0 ? "pass" : "fail", (long long) extent,
        (unsigned long long) cpu_us, (unsigned long long) gpu_us,
        (unsigned long long) mixed_us, (unsigned long long) sequential_us,
        (unsigned long long) overlap_us);
    return overlap_us > 0 ? 0 : 1;
}
