#include "ggml-cpp.h"
#include "ggml-cuda.h"
#include "llama-expert-scheduler.h"
#include "llama-expert-storage.h"
#include "llama-expert-weight-provider.h"
#include "llama-mmap.h"

#include <array>
#include <cstdio>
#include <fcntl.h>
#include <stdexcept>
#include <unistd.h>

namespace {

void require(bool condition, const char * message) { if (!condition) throw std::runtime_error(message); }

struct temporary_source {
    char path[64] = "/tmp/phase11-uma-provider-XXXXXX";
    temporary_source() {
        const int fd = mkstemp(path);
        require(fd >= 0, "mkstemp failed");
        std::array<uint8_t, 288> bytes;
        for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = uint8_t(i*17U + 3U);
        require(write(fd, bytes.data(), bytes.size()) == ssize_t(bytes.size()), "source write failed");
        close(fd);
    }
    ~temporary_source() { unlink(path); }
};

llm_expert_bundle_descriptor make_bundle(ggml_context * ctx) {
    auto projection = [&](const char * name) {
        auto * weight = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, 3, 2);
        ggml_set_name(weight, name);
        return llm_expert_projection_descriptor::from(weight, nullptr, nullptr);
    };
    return { 0, 2, projection("up"), projection("gate"), {}, projection("down") };
}

std::vector<llm_expert_storage_span> spans(uint64_t base) {
    return {
        { 0, base + 0, 48, llm_expert_storage_projection::up, llm_expert_storage_sidecar::weight, 0, 48 },
        { 0, base + 48, 48, llm_expert_storage_projection::gate, llm_expert_storage_sidecar::weight, 48, 48 },
        { 0, base + 96, 48, llm_expert_storage_projection::down, llm_expert_storage_sidecar::weight, 96, 48 },
    };
}

void test_provider() {
    temporary_source source;
    llama_file file(source.path, "rb");
    llm_expert_storage storage({ 1, 2, 2, 1024 }, { { 0, &file, 512, source.path, false } });
    require(storage.add_bundle({ 0, 0 }, spans(0)).is_ready(), "expert 0 directory failed");
    require(storage.add_bundle({ 0, 1 }, spans(144)).is_ready(), "expert 1 directory failed");
    require(storage.seal().is_ready(), "storage seal failed");
    llm_expert_scheduler scheduler({ 1, 2, 8, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1 });

    llm_uma_cache_config config;
    config.pool_bytes = 8192;
    config.hot_capacity = 1;
    config.n_expert_used = 1;
    config.routed_layer_count = 1;
    config.total_expert_keys = 2;
    config.buffer_type = ggml_backend_cuda_uma_buffer_type(0);
    config.target_device = ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), 0);
    config.storage = &storage;
    config.scheduler = &scheduler;
    config.is_uma_buffer_type = ggml_backend_buft_is_cuda_uma;
    config.prefetch = ggml_backend_cuda_uma_prefetch;
    config.checksum = ggml_backend_cuda_uma_checksum;
    config.routed_layers = { 0 };
    auto provider = llm_create_uma_cache_expert_weight_provider(config);

    ggml_init_params params = { ggml_tensor_overhead()*32, nullptr, true };
    ggml_context_ptr ctx(ggml_init(params));
    require(bool(ctx), "context failed");
    auto bundle = make_bundle(ctx.get());
    auto * logical = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 1, 1);
    llm_expert_selection selection = { 0, 2, 1, 1, logical };
    bool owner = false;
    require(provider->begin_initialization(
        llm_expert_provider_initialization_stage::descriptors_before_scheduler_reserve, owner).is_ready() && owner,
        "initialization begin failed");
    llm_expert_graph_binding bootstrap;
    require(provider->bind_graph(ctx.get(), bundle, selection, bootstrap).is_ready() && bootstrap.bootstrap,
        "bootstrap bind failed");
    require(provider->bind_graph(ctx.get(), bundle, selection, bootstrap).is_ready(), "repeat bootstrap bind failed");
    require(provider->complete_descriptor_discovery(2, 2, 0, {}, {}).is_ready(), "descriptor discovery failed");
    require(provider->initialize_after_reserve().is_ready(), "UMA arena initialization failed");
    require(provider->finish_initialization(true).is_ready(), "initialization finish failed");

    llm_expert_graph_binding binding;
    require(provider->bind_graph(ctx.get(), bundle, selection, binding).is_ready() && !binding.bootstrap,
        "runtime bind failed");
    require(binding.up.weight->nb[2] == binding.gate.weight->nb[2] &&
        binding.up.weight->nb[2] == binding.down.weight->nb[2] && binding.up.weight->nb[2] % 4096 == 0,
        "bundle slots are not fixed interleaved page-aligned spans");
    for (int32_t expert = 0; expert < 2; ++expert) {
        llm_expert_execution_plan plan;
        require(provider->prepare({ binding }, plan).is_ready(), "prepare failed");
        int32_t execution = -1;
        require(provider->remap_checkpoint(binding, &expert, 1, &execution).is_ready(), "remap failed");
        require(execution >= 0 && execution < 2, "invalid execution slot");
        auto * data = static_cast<uint8_t *>(binding.up.weight->data) + size_t(execution)*binding.up.weight->nb[2];
        require(data[0] == uint8_t((size_t(expert)*144)*17U + 3U), "final slot has wrong source bytes");
        auto * base = static_cast<uint8_t *>(ggml_backend_buffer_get_base(binding.up.weight->buffer));
        uint64_t gpu_checksum = 0;
        require(ggml_backend_cuda_uma_checksum(binding.up.weight->buffer, size_t(data - base), 48,
            &gpu_checksum) == 0 && gpu_checksum != 0, "GPU did not consume final slot");
    }
    const auto provider_stats = provider->get_stats();
    const auto storage_stats = storage.diagnostics();
    const auto scheduler_stats = scheduler.diagnostics();
    require(provider_stats.tensor_copies == 0 && storage_stats.read_bytes == 288,
        "provider copied payload or storage byte count is wrong");
    require(scheduler_stats.terminal_complete == 2 && scheduler_stats.active_requests == 0,
        "readiness scheduler did not drain");
}

} // namespace

int main() try {
    test_provider();
    return 0;
} catch (const std::exception & error) {
    fprintf(stderr, "test-expert-uma-provider: %s\n", error.what());
    return 1;
}
