#include "llama-expert-prefetch.h"
#include "llama-cpp.h"

#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

using json = nlohmann::json;

namespace {

struct probe_error : std::runtime_error {
    using std::runtime_error::runtime_error;
};

uint64_t percentile(std::vector<uint64_t> values, uint32_t numerator, uint32_t denominator) {
    if (values.empty()) throw probe_error("empty measurement");
    std::sort(values.begin(), values.end());
    const size_t index = std::min(values.size() - 1, (values.size()*numerator + denominator - 1)/denominator - 1);
    return values[index];
}

uint64_t elapsed_ns(std::chrono::steady_clock::time_point begin) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - begin).count();
}

std::vector<uint64_t> benchmark_pread(const std::string & path, uint64_t bytes, bool direct, bool & supported) {
    supported = false;
    const size_t alignment = 4096;
    const size_t extent = direct ? size_t((bytes + alignment - 1)/alignment*alignment) : size_t(bytes);
    int flags = O_RDONLY;
#ifdef O_DIRECT
    if (direct) flags |= O_DIRECT;
#else
    if (direct) return {};
#endif
    const int descriptor = open(path.c_str(), flags);
    if (descriptor < 0) return {};
    void * buffer = nullptr;
    if (posix_memalign(&buffer, alignment, extent) != 0) {
        close(descriptor);
        return {};
    }
    std::vector<uint64_t> measurements;
    for (uint32_t repetition = 0; repetition < 21; ++repetition) {
        const auto begin = std::chrono::steady_clock::now();
        const ssize_t result = pread(descriptor, buffer, extent, 0);
        const uint64_t elapsed = elapsed_ns(begin);
        if (result != ssize_t(extent)) {
            std::free(buffer);
            close(descriptor);
            return {};
        }
        if (repetition != 0) measurements.push_back(elapsed);
    }
    std::free(buffer);
    close(descriptor);
    supported = true;
    return measurements;
}

std::vector<uint64_t> benchmark_h2d(uint64_t bytes, bool & supported) {
    supported = false;
    ggml_backend_dev_t device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (device == nullptr || bytes > SIZE_MAX) return {};
    ggml_backend_ptr backend(ggml_backend_dev_init(device, nullptr));
    if (!backend) return {};
    ggml_init_params params = { ggml_tensor_overhead(), nullptr, true };
    ggml_context_ptr context(ggml_init(params));
    if (!context) return {};
    ggml_tensor * tensor = ggml_new_tensor_1d(context.get(), GGML_TYPE_I8, int64_t(bytes));
    ggml_backend_buffer_ptr destination(
        ggml_backend_alloc_ctx_tensors_from_buft(context.get(), ggml_backend_dev_buffer_type(device)));
    ggml_backend_buffer_ptr source(ggml_backend_buft_alloc_buffer(ggml_backend_dev_host_buffer_type(device), size_t(bytes)));
    if (!destination || !source || !ggml_backend_buffer_is_host(source.get())) return {};
    ggml_backend_buffer_clear(source.get(), 0);
    std::vector<uint64_t> measurements;
    for (uint32_t repetition = 0; repetition < 21; ++repetition) {
        const auto begin = std::chrono::steady_clock::now();
        ggml_backend_tensor_set_async(backend.get(), tensor, ggml_backend_buffer_get_base(source.get()), 0, size_t(bytes));
        ggml_backend_synchronize(backend.get());
        const uint64_t elapsed = elapsed_ns(begin);
        if (repetition != 0) measurements.push_back(elapsed);
    }
    supported = true;
    return measurements;
}

std::vector<uint64_t> benchmark_predictor(
        const llm_expert_prefetch_profile & profile,
        const std::string & profile_path) {
    llama_expert_prefetch_config_v1 value = {
        LLAMA_EXPERT_PREFETCH_VERSION_1, sizeof(llama_expert_prefetch_config_v1),
        LLAMA_EXPERT_PREFETCH_POLICY_TEMPORAL_FREQUENCY, LLAMA_EXPERT_PREFETCH_READINESS_DEVICE_READY,
        LLAMA_EXPERT_PREFETCH_SEED_MODE_OFF, 4, std::min<uint32_t>(profile.target.experts_per_token, 2),
        64U*1024U*1024U, 32, UINT64_C(1) << 30, UINT64_C(1) << 30, UINT64_C(1) << 30,
        UINT64_C(1) << 30, 32, 32, 64, 32, {},
    };
    llm_expert_prefetch_config_internal config;
    if (!llm_expert_prefetch_copy_config(&value, profile_path.c_str(), LLAMA_EXPERT_WEIGHTS_MODE_COLD_CACHE,
        profile.target.experts_per_layer, 32, UINT64_C(1) << 30, 64, UINT64_C(1) << 30,
        UINT64_C(1) << 30, config).is_ready()) throw probe_error("predictor configuration failed");
    llm_expert_prefetch_predictor predictor;
    if (!predictor.initialize(profile, config).is_ready() || !predictor.request_begin(1).is_ready()) {
        throw probe_error("predictor initialization failed");
    }
    std::vector<std::vector<int32_t>> token(profile.target.routed_layers.size());
    for (size_t layer = 0; layer < token.size(); ++layer) {
        for (uint32_t rank = 0; rank < profile.target.experts_per_token; ++rank) {
            token[layer].push_back(int32_t((layer + rank)%profile.target.experts_per_layer));
        }
    }
    for (uint64_t ordinal = 0; ordinal < 4; ++ordinal) {
        if (!predictor.commit_token(ordinal, token).is_ready()) throw probe_error("predictor warmup failed");
    }
    std::vector<uint64_t> measurements;
    for (uint32_t repetition = 0; repetition < 1000; ++repetition) {
        std::vector<llm_expert_prefetch_candidate> candidates;
        const auto begin = std::chrono::steady_clock::now();
        if (!predictor.predict_token_end(3, profile.target.routed_layers.front(), candidates).is_ready()) {
            throw probe_error("predictor benchmark failed");
        }
        measurements.push_back(elapsed_ns(begin));
    }
    return measurements;
}

uint64_t validate_model_profile(
        const std::string & model_path,
        const std::string & profile_path,
        const llm_expert_prefetch_profile & profile) {
    const auto selected_cost = std::find_if(profile.costs.begin(), profile.costs.end(), [&](const auto & cost) {
        return cost.transport == profile.selected_transport && cost.readiness == profile.selected_readiness;
    });
    if (selected_cost == profile.costs.end()) throw probe_error("selected cost envelope is unavailable");
    static const llama_model_tensor_buft_override overrides[] = {
        { "ffn_(gate|up|down)_exps\\.weight", ggml_backend_cpu_buffer_type() },
        { nullptr, nullptr },
    };
    llama_expert_prefetch_config_v1 prefetch = {
        LLAMA_EXPERT_PREFETCH_VERSION_1, sizeof(llama_expert_prefetch_config_v1),
        LLAMA_EXPERT_PREFETCH_POLICY_TEMPORAL_FREQUENCY, profile.selected_readiness,
        LLAMA_EXPERT_PREFETCH_SEED_MODE_OFF, profile.selected_temporal_window, profile.selected_candidates,
        64U*1024U*1024U, 16, 16U*1024U*1024U, 16U*1024U*1024U,
        16U*1024U*1024U, 16U*1024U*1024U, 14, 14,
        selected_cost->utility_window_predictions, selected_cost->utility_min_observations, {},
    };
    auto params = llama_model_default_params();
    params.n_gpu_layers = -1;
    params.tensor_buft_overrides = overrides;
    params.expert_weights_mode = LLAMA_EXPERT_WEIGHTS_MODE_COLD_CACHE;
    params.expert_hot_cache_capacity = 16;
    params.expert_cold_cache_bytes = 64U*1024U*1024U;
    params.expert_transfer_ring_bytes = 16U*1024U*1024U;
    params.expert_prefetch_config = &prefetch;
    params.expert_prefetch_profile_path = profile_path.c_str();
    const auto begin = std::chrono::steady_clock::now();
    llama_model_ptr model(llama_model_load_from_file(model_path.c_str(), params));
    const uint64_t duration = elapsed_ns(begin);
    if (!model) throw probe_error("active profile model load failed");
    return duration;
}

json envelope(
        const std::string & transport,
        const std::string & readiness,
        bool supported,
        const std::vector<uint64_t> & service,
        uint64_t predictor_p95,
        uint64_t storage_bytes,
        uint64_t h2d_bytes) {
    const uint64_t p50 = supported ? percentile(service, 50, 100) : 0;
    const uint64_t p95 = supported ? percentile(service, 95, 100) : 0;
    return {{"transport", transport}, {"readiness", readiness}, {"supported", supported},
        {"lead_p50_ns", p50}, {"demand_service_p50_ns", p50}, {"speculative_service_p95_ns", p95},
        {"predictor_compute_p95_ns", predictor_p95}, {"scheduler_demand_delay_p95_ns", 0},
        {"displacement_refill_p95_ns", p95}, {"storage_bytes", storage_bytes}, {"h2d_bytes", h2d_bytes},
        {"utility_window_predictions", 64}, {"utility_min_observations", 32}};
}

} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc != 7 || std::string(argv[1]) != "--profile" || std::string(argv[3]) != "--model" ||
            std::string(argv[5]) != "--identity") {
            throw probe_error("usage: phase10-prefetch-probe --profile FILE --model GGUF --identity PROJECT:NESTED");
        }
        const std::string profile_path = argv[2];
        const std::string model_path = argv[4];
        const std::string identity = argv[6];
        const size_t separator = identity.find(':');
        if (separator == std::string::npos) throw probe_error("identity must be PROJECT:NESTED");
        ggml_backend_load_all();
        llm_expert_prefetch_profile profile;
        std::string profile_error;
        const auto profile_begin = std::chrono::steady_clock::now();
        if (!llm_expert_prefetch_load_profile(profile_path, 64U*1024U*1024U, nullptr, profile, profile_error).is_ready()) {
            throw probe_error("profile load failed: " + profile_error);
        }
        const uint64_t profile_parse_ns = elapsed_ns(profile_begin);
        const uint64_t model_profile_load_ns = validate_model_profile(model_path, profile_path, profile);
        const uint64_t bytes = profile.target.expert_bytes.front().physical_bytes;
        const auto predictor = benchmark_predictor(profile, profile_path);
        const uint64_t predictor_p95 = percentile(predictor, 95, 100);
        bool buffered_supported = false, direct_supported = false, h2d_supported = false;
        const auto buffered = benchmark_pread(model_path, bytes, false, buffered_supported);
        const auto direct = benchmark_pread(model_path, bytes, true, direct_supported);
        const auto h2d = benchmark_h2d(bytes, h2d_supported);
        std::vector<uint64_t> buffered_device, direct_device;
        if (buffered_supported && h2d_supported) {
            for (size_t index = 0; index < std::min(buffered.size(), h2d.size()); ++index) buffered_device.push_back(buffered[index] + h2d[index]);
        }
        if (direct_supported && h2d_supported) {
            for (size_t index = 0; index < std::min(direct.size(), h2d.size()); ++index) direct_device.push_back(direct[index] + h2d[index]);
        }
        json output = { {"schema_version", "phase10-transport-measurements-v1"},
            {"project_head", identity.substr(0, separator)}, {"nested_head", identity.substr(separator + 1)},
            {"host", "skynet"}, {"profile_sha256", profile.profile_sha256},
            {"profile_parse_ns", profile_parse_ns}, {"model_profile_load_ns", model_profile_load_ns},
            {"envelopes", json::array()} };
        output["envelopes"].push_back(envelope("BUFFERED", "HOST_READY", buffered_supported, buffered, predictor_p95, bytes, 0));
        output["envelopes"].push_back(envelope("DIRECT_IO", "HOST_READY", direct_supported, direct, predictor_p95, bytes, 0));
        output["envelopes"].push_back(envelope("HOST_TO_DEVICE", "DEVICE_READY", h2d_supported, h2d, predictor_p95, 0, bytes));
        output["envelopes"].push_back(envelope("BUFFERED", "DEVICE_READY", !buffered_device.empty(), buffered_device, predictor_p95, bytes, bytes));
        output["envelopes"].push_back(envelope("DIRECT_IO", "DEVICE_READY", !direct_device.empty(), direct_device, predictor_p95, bytes, bytes));
        std::cout << output.dump(2) << '\n';
        return 0;
    } catch (const std::exception & exception) {
        std::fprintf(stderr, "phase10-prefetch-probe: %s\n", exception.what());
        return 1;
    }
}
