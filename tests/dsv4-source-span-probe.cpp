#include "llama-expert-async-io.h"
#include "llama-expert-storage.h"
#include "llama-model.h"

#include "llama.h"

#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <string>
#include <unistd.h>
#include <vector>

using json = nlohmann::ordered_json;

namespace {

bool parse_u64(const char * text, uint64_t & result) {
    char * end = nullptr;
    const auto value = std::strtoull(text, &end, 10);
    if (end == text || *end != '\0') return false;
    result = value;
    return true;
}

uint64_t digest_bytes(const uint8_t * data, size_t size) {
    uint64_t digest = 1469598103934665603ULL;
    for (size_t index = 0; index < size; ++index) {
        digest ^= data[index];
        digest *= 1099511628211ULL;
    }
    return digest;
}

const char * projection_name(llm_expert_storage_projection value) {
    switch (value) {
        case llm_expert_storage_projection::up: return "up";
        case llm_expert_storage_projection::gate: return "gate";
        case llm_expert_storage_projection::gate_up: return "gate_up";
        case llm_expert_storage_projection::down: return "down";
    }
    return "unknown";
}

const char * sidecar_name(llm_expert_storage_sidecar value) {
    switch (value) {
        case llm_expert_storage_sidecar::weight: return "weight";
        case llm_expert_storage_sidecar::bias: return "bias";
        case llm_expert_storage_sidecar::scale: return "scale";
    }
    return "unknown";
}

bool pread_all(int fd, void * destination, size_t bytes, uint64_t offset) {
    size_t completed = 0;
    while (completed < bytes) {
        const ssize_t result = pread(fd, static_cast<uint8_t *>(destination) + completed,
            bytes - completed, off_t(offset + completed));
        if (result < 0 && errno == EINTR) continue;
        if (result <= 0) return false;
        completed += size_t(result);
    }
    return true;
}

size_t align_up(size_t value, size_t alignment) {
    if (alignment == 0 || value > SIZE_MAX - (alignment - 1)) return SIZE_MAX;
    return (value + alignment - 1)/alignment*alignment;
}

bool checked_lcm(size_t lhs, size_t rhs, size_t & result) {
    if (lhs == 0 || rhs == 0) return false;
    const size_t reduced = lhs/std::gcd(lhs, rhs);
    if (reduced > SIZE_MAX/rhs) return false;
    result = reduced*rhs;
    return true;
}

std::string routed_layout_signature(const llama_layer & layer) {
    std::string result;
    for (const ggml_tensor * tensor : {
            layer.ffn_up_exps, layer.ffn_gate_exps, layer.ffn_down_exps }) {
        if (tensor == nullptr) return {};
        result += std::to_string(unsigned(tensor->type)) + ":" +
            std::to_string(tensor->ne[0]) + ":" + std::to_string(tensor->ne[1]) + ":" +
            std::to_string(tensor->nb[0]) + ":" + std::to_string(tensor->nb[1]) + ":" +
            std::to_string(tensor->nb[2]) + ";";
    }
    return result;
}

bool derive_universal_hot_stride(
        const llama_model & model,
        ggml_backend_buffer_type_t target_buft,
        size_t & stride,
        std::array<size_t, 3> & role_offsets,
        std::array<size_t, 3> & role_extents) {
    if (target_buft == nullptr) return false;
    const size_t buffer_alignment = ggml_backend_buft_get_alignment(target_buft);
    size_t slot_alignment = std::max<size_t>(64, buffer_alignment);
    size_t within_slot = 0;
    for (size_t role = 0; role < role_extents.size(); ++role) {
        size_t role_alignment = std::max<size_t>(64, buffer_alignment);
        for (const auto & layer : model.layers) {
            const ggml_tensor * tensor = role == 0 ? layer.ffn_up_exps :
                role == 1 ? layer.ffn_gate_exps : layer.ffn_down_exps;
            if (tensor == nullptr) continue;
            role_extents[role] = std::max(role_extents[role], tensor->nb[2]);
            role_alignment = std::max(role_alignment, ggml_type_size(tensor->type));
            if (!checked_lcm(slot_alignment, ggml_type_size(tensor->type), slot_alignment)) return false;
        }
        if (role_extents[role] == 0) return false;
        role_offsets[role] = align_up(within_slot, role_alignment);
        if (role_offsets[role] == SIZE_MAX ||
                role_extents[role] > SIZE_MAX - role_offsets[role]) return false;
        within_slot = role_offsets[role] + role_extents[role];
    }
    stride = align_up(within_slot, slot_alignment);
    return stride != 0 && stride != SIZE_MAX;
}

json compare_cuda_projection(
        ggml_backend_t backend,
        ggml_backend_buffer_type_t buft,
        const ggml_tensor * source,
        const uint8_t * source_bytes,
        size_t source_size,
        size_t universal_stride,
        int32_t layer,
        const char * projection) {
    if (backend == nullptr || buft == nullptr || source == nullptr || source_bytes == nullptr ||
            source_size != source->nb[2] || universal_stride < source_size) {
        throw std::runtime_error("invalid CUDA projection comparison input");
    }
    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead()*8 + ggml_graph_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) throw std::runtime_error("cannot create CUDA comparison context");
    ggml_tensor * compact = ggml_new_tensor_3d(
        ctx.get(), source->type, source->ne[0], source->ne[1], 1);
    ggml_tensor * universal = ggml_new_tensor_3d(
        ctx.get(), source->type, source->ne[0], source->ne[1], 2);
    universal->nb[2] = universal_stride;
    universal->nb[3] = universal->nb[2]*universal->ne[2];
    ggml_tensor * input = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, source->ne[0], 1, 1);
    ggml_tensor * compact_ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 1, 1);
    ggml_tensor * universal_ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 1, 1);
    ggml_tensor * compact_output = ggml_mul_mat_id(ctx.get(), compact, input, compact_ids);
    ggml_tensor * universal_output = ggml_mul_mat_id(ctx.get(), universal, input, universal_ids);
    if (!ggml_backend_dev_supports_op(ggml_backend_get_device(backend), compact_output) ||
            !ggml_backend_dev_supports_op(ggml_backend_get_device(backend), universal_output)) {
        throw std::runtime_error("CUDA comparison operation is unsupported");
    }
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft));
    if (!buffer) throw std::runtime_error("cannot allocate CUDA comparison tensors");

    ggml_backend_tensor_set(compact, source_bytes, 0, source_size);
    std::vector<uint8_t> poisoned(ggml_nbytes(universal), 0xa5);
    ggml_backend_tensor_set(universal, poisoned.data(), 0, poisoned.size());
    ggml_backend_tensor_set(universal, source_bytes, universal_stride, source_size);
    std::vector<float> activation(size_t(source->ne[0]));
    for (size_t index = 0; index < activation.size(); ++index) {
        activation[index] = float(int(index % 31) - 15)/32.0f;
    }
    ggml_backend_tensor_set(input, activation.data(), 0, activation.size()*sizeof(float));
    const int32_t compact_id = 0;
    const int32_t universal_id = 1;
    ggml_backend_tensor_set(compact_ids, &compact_id, 0, sizeof(compact_id));
    ggml_backend_tensor_set(universal_ids, &universal_id, 0, sizeof(universal_id));

    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, compact_output);
    ggml_build_forward_expand(graph, universal_output);
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("CUDA projection comparison failed");
    }
    ggml_backend_synchronize(backend);
    std::vector<float> compact_values(size_t(ggml_nelements(compact_output)));
    std::vector<float> universal_values(compact_values.size());
    ggml_backend_tensor_get(
        compact_output, compact_values.data(), 0, compact_values.size()*sizeof(float));
    ggml_backend_tensor_get(
        universal_output, universal_values.data(), 0, universal_values.size()*sizeof(float));
    double max_abs_error = 0.0;
    double max_scaled_abs_error = 0.0;
    bool finite = true;
    for (size_t index = 0; index < compact_values.size(); ++index) {
        finite = finite && std::isfinite(compact_values[index]) && std::isfinite(universal_values[index]);
        max_abs_error = std::max(max_abs_error,
            std::abs(double(compact_values[index]) - double(universal_values[index])));
        max_scaled_abs_error = std::max(max_scaled_abs_error,
            std::abs(double(compact_values[index])*0.25 - double(universal_values[index])*0.25));
    }
    std::vector<uint8_t> slot_zero(source_size);
    ggml_backend_tensor_get(universal, slot_zero.data(), 0, slot_zero.size());
    const bool padding_guard = std::all_of(
        slot_zero.begin(), slot_zero.end(), [](uint8_t value) { return value == 0xa5; });
    const uint64_t compact_digest = digest_bytes(
        reinterpret_cast<const uint8_t *>(compact_values.data()), compact_values.size()*sizeof(float));
    const uint64_t universal_digest = digest_bytes(
        reinterpret_cast<const uint8_t *>(universal_values.data()), universal_values.size()*sizeof(float));
    return {
        { "layer", layer }, { "projection", projection },
        { "ggml_type", ggml_type_name(source->type) }, { "source_bytes", source_size },
        { "input_elements", activation.size() }, { "output_elements", compact_values.size() },
        { "compact_expert_id", compact_id }, { "universal_slot_id", universal_id },
        { "routing_weight_f32", 0.25 }, { "finite", finite },
        { "compact_output_fnv1a64", compact_digest },
        { "universal_output_fnv1a64", universal_digest },
        { "bit_exact", compact_values == universal_values },
        { "max_abs_error", max_abs_error }, { "max_scaled_abs_error", max_scaled_abs_error },
        { "padding_guard_intact", padding_guard },
    };
}

} // namespace

int main(int argc, char ** argv) {
    std::string model_path;
    std::string output_path;
    uint64_t cold_bytes = 1ULL << 30;
    uint64_t ring_bytes = 64ULL << 20;
    for (int index = 1; index < argc; ++index) {
        if (index + 1 >= argc) return 2;
        const std::string option = argv[index];
        const char * value = argv[++index];
        if (option == "--model") model_path = value;
        else if (option == "--output") output_path = value;
        else if (option == "--cold-bytes" && parse_u64(value, cold_bytes)) {}
        else if (option == "--ring-bytes" && parse_u64(value, ring_bytes)) {}
        else return 2;
    }
    if (model_path.empty() || output_path.empty() || cold_bytes == 0 || ring_bytes == 0) return 2;

    ggml_backend_load_all();
    const llama_model_tensor_buft_override overrides[] = {
        { "ffn_(gate|up|down)_exps\\.(weight|bias|scale)", ggml_backend_cpu_buffer_type() },
        { nullptr, nullptr },
    };
    llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = -1;
    params.load_mode = LLAMA_LOAD_MODE_MMAP;
    params.expert_weights_mode = LLAMA_EXPERT_WEIGHTS_MODE_COLD_CACHE;
    params.expert_hot_cache_capacity = 16;
    params.expert_cold_cache_bytes = cold_bytes;
    params.expert_transfer_ring_bytes = ring_bytes;
    params.expert_io_queue_depth = 8;
    params.tensor_buft_overrides = overrides;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), params);
    if (!model) return 3;

    int result_code = 4;
    try {
        auto * storage = model->expert_storage();
        if (storage == nullptr) throw std::runtime_error("expert storage unavailable");
        const auto initial_storage = storage->diagnostics();
        if (!initial_storage.sealed || initial_storage.read_requests != 0) {
            throw std::runtime_error("storage is not sealed and unread at probe start");
        }

        uint32_t file_count = 0;
        if (llama_model_source_file_count(model, &file_count) != LLAMA_MODEL_STORAGE_STATUS_OK || file_count == 0) {
            throw std::runtime_error("source file metadata unavailable");
        }
        std::vector<llama_model_source_file_metadata> files(file_count);
        std::vector<int> reference_fds(file_count, -1);
        for (uint32_t index = 0; index < file_count; ++index) {
            if (llama_model_get_source_file_metadata(model, index, &files[index]) != LLAMA_MODEL_STORAGE_STATUS_OK) {
                throw std::runtime_error("source file metadata incomplete");
            }
            reference_fds[index] = open(files[index].identity, O_RDONLY | O_CLOEXEC);
            if (reference_fds[index] < 0) throw std::runtime_error("cannot open source file for reference read");
        }

        const int32_t n_layer = int32_t(model->layers.size());
        const int32_t n_expert = int32_t(model->hparams.n_expert);
        std::vector<int32_t> representative_layers;
        std::set<std::string> observed_layouts;
        for (int32_t layer = 0; layer < n_layer; ++layer) {
            const std::string signature = routed_layout_signature(model->layers[size_t(layer)]);
            if (!signature.empty() && observed_layouts.insert(signature).second) {
                representative_layers.push_back(layer);
            }
        }
        if (representative_layers.size() != 3) {
            throw std::runtime_error("expected exactly three routed layout classes");
        }
        std::vector<llm_expert_key> samples;
        for (int32_t layer : { 0, n_layer/2, n_layer - 1 }) {
            for (int32_t expert : { 0, n_expert/2, n_expert - 1 }) {
                samples.push_back({ layer, expert });
            }
        }
        for (int32_t layer : representative_layers) {
            const bool already_sampled = std::any_of(samples.begin(), samples.end(), [&](const auto & key) {
                return key.layer == layer && key.expert == 0;
            });
            if (!already_sampled) samples.push_back({ layer, 0 });
        }
        bool cross_split_sample_added = false;
        for (int32_t layer = 0; layer < n_layer && !cross_split_sample_added; ++layer) {
            for (int32_t expert = 0; expert < n_expert; ++expert) {
                const auto * spans = storage->find({ layer, expert });
                if (spans == nullptr) continue;
                std::set<uint16_t> split_indices;
                for (const auto & span : *spans) split_indices.insert(span.split_index);
                if (split_indices.size() > 1) {
                    const bool already_sampled = std::any_of(samples.begin(), samples.end(), [&](const auto & key) {
                        return key.layer == layer && key.expert == expert;
                    });
                    if (!already_sampled) samples.push_back({ layer, expert });
                    cross_split_sample_added = true;
                    break;
                }
            }
        }

        std::vector<intptr_t> handles(file_count);
        size_t handle_count = 0;
        if (!storage->copy_source_native_handles(handles.data(), handles.size(), handle_count).is_ready() ||
                handle_count != file_count) {
            throw std::runtime_error("cannot copy storage source handles");
        }
        llm_expert_async_config async_config;
        async_config.requested_queue_depth = 8;
        async_config.effective_hot_capacity = 16;
        async_config.request_capacity = uint32_t(samples.size() + 1);
        async_config.trace_capacity = 1024;
        async_config.cold_cache_bytes = cold_bytes;
        async_config.requested_staging_bytes = ring_bytes;
        async_config.maximum_aligned_read_bytes = initial_storage.maximum_bundle_bytes +
            std::max<uint64_t>(1, initial_storage.maximum_direct_alignment);
        async_config.source_file_capacity = file_count;
        llm_expert_async_transport async(async_config);
        if (async.register_files(handles.data(), handles.size()) != llm_expert_async_result::ready) {
            throw std::runtime_error("async source registration failed");
        }

        json sample_results = json::array();
        bool all_equal = true;
        uint64_t total_sample_bytes = 0;
        std::map<int32_t, std::vector<uint8_t>> representative_bundles;
        for (size_t sample_index = 0; sample_index < samples.size(); ++sample_index) {
            const auto key = samples[sample_index];
            const auto * spans = storage->find(key);
            if (spans == nullptr || spans->empty()) throw std::runtime_error("sample storage entry missing");
            uint64_t bundle_bytes = 0;
            for (const auto & span : *spans) {
                bundle_bytes = std::max(bundle_bytes, span.destination_offset + span.destination_extent);
            }
            std::vector<uint8_t> sync_bytes(size_t(bundle_bytes), 0);
            std::vector<uint8_t> async_bytes(size_t(bundle_bytes), 0);
            std::vector<uint8_t> source_bytes(size_t(bundle_bytes), 0);
            const auto sync_result = storage->read_bundle(key, sync_bytes.data(), sync_bytes.size());
            if (!sync_result.is_ready()) throw std::runtime_error("synchronous storage read failed");

            std::vector<llm_expert_storage_destination> destinations;
            destinations.reserve(spans->size());
            json span_results = json::array();
            std::set<uint16_t> split_indices;
            for (const auto & span : *spans) {
                destinations.push_back({ span.projection, span.sidecar,
                    async_bytes.data() + span.destination_offset, span.destination_extent });
                if (!pread_all(reference_fds[span.split_index], source_bytes.data() + span.destination_offset,
                        size_t(span.byte_count), span.file_offset)) {
                    throw std::runtime_error("reference source read failed");
                }
                split_indices.insert(span.split_index);
                span_results.push_back({
                    { "split_index", span.split_index },
                    { "source_file_identity", files[span.split_index].identity },
                    { "file_offset", span.file_offset },
                    { "byte_count", span.byte_count },
                    { "projection", projection_name(span.projection) },
                    { "sidecar", sidecar_name(span.sidecar) },
                    { "destination_offset", span.destination_offset },
                });
            }
            std::array<llm_expert_storage_read_operation, 12> operations;
            size_t operation_count = 0;
            if (!storage->make_read_plan(key, destinations.data(), destinations.size(),
                    operations.data(), operations.size(), operation_count).is_ready() || operation_count == 0) {
                throw std::runtime_error("async read plan creation failed");
            }
            const llm_expert_async_operation_identity identity = {
                1, { uint32_t(sample_index), 1 }, 0, key,
                llm_expert_readiness::host_ready, llm_expert_priority::demand_current_layer,
            };
            if (async.submit_read_plan(identity, operations.data(), operation_count) != llm_expert_async_result::ready) {
                throw std::runtime_error("async read submission failed");
            }
            llm_expert_async_read_completion completion;
            if (async.wait_read(identity.request, completion) != llm_expert_async_result::ready ||
                    completion.bytes_completed != bundle_bytes ||
                    async.release_read(identity.request) != llm_expert_async_result::ready) {
                throw std::runtime_error("async read completion failed");
            }
            const bool sync_matches_source = sync_bytes == source_bytes;
            const bool async_matches_source = async_bytes == source_bytes;
            const bool sync_matches_async = sync_bytes == async_bytes;
            all_equal = all_equal && sync_matches_source && async_matches_source && sync_matches_async;
            if (key.expert == 0 && std::find(representative_layers.begin(), representative_layers.end(), key.layer) !=
                    representative_layers.end()) {
                representative_bundles.emplace(key.layer, sync_bytes);
            }
            total_sample_bytes += bundle_bytes;
            sample_results.push_back({
                { "layer", key.layer },
                { "expert", key.expert },
                { "bundle_bytes", bundle_bytes },
                { "split_indices", split_indices },
                { "crosses_source_files", split_indices.size() > 1 },
                { "operation_count", operation_count },
                { "sync_matches_source", sync_matches_source },
                { "async_matches_source", async_matches_source },
                { "sync_matches_async", sync_matches_async },
                { "source_digest_fnv1a64", digest_bytes(source_bytes.data(), source_bytes.size()) },
                { "spans", std::move(span_results) },
            });
        }

        auto * gpu_device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (gpu_device == nullptr) throw std::runtime_error("CUDA device unavailable for layout comparison");
        ggml_backend_ptr gpu_backend(ggml_backend_dev_init(gpu_device, nullptr));
        if (!gpu_backend) throw std::runtime_error("CUDA backend initialization failed");
        auto * target_buft = ggml_backend_get_default_buffer_type(gpu_backend.get());
        size_t universal_hot_stride = 0;
        std::array<size_t, 3> hot_role_offsets = {};
        std::array<size_t, 3> hot_role_extents = {};
        if (!derive_universal_hot_stride(
                *model, target_buft, universal_hot_stride, hot_role_offsets, hot_role_extents)) {
            throw std::runtime_error("cannot derive universal CUDA layout");
        }
        json kernel_comparisons = json::array();
        bool all_kernel_comparisons_pass = true;
        for (int32_t layer_index : representative_layers) {
            const auto bundle_it = representative_bundles.find(layer_index);
            const auto * spans = storage->find({ layer_index, 0 });
            if (bundle_it == representative_bundles.end() || spans == nullptr) {
                throw std::runtime_error("representative class bytes unavailable");
            }
            const auto & layer = model->layers[size_t(layer_index)];
            struct projection_input {
                llm_expert_storage_projection projection;
                const char * name;
                const ggml_tensor * tensor;
            };
            const std::array<projection_input, 3> projections = {{
                { llm_expert_storage_projection::up, "up", layer.ffn_up_exps },
                { llm_expert_storage_projection::gate, "gate", layer.ffn_gate_exps },
                { llm_expert_storage_projection::down, "down", layer.ffn_down_exps },
            }};
            for (const auto & projection : projections) {
                const auto span_it = std::find_if(spans->begin(), spans->end(), [&](const auto & span) {
                    return span.projection == projection.projection &&
                        span.sidecar == llm_expert_storage_sidecar::weight;
                });
                if (span_it == spans->end() ||
                        span_it->destination_offset > bundle_it->second.size() ||
                        span_it->byte_count > bundle_it->second.size() - span_it->destination_offset) {
                    throw std::runtime_error("representative projection span unavailable");
                }
                json comparison = compare_cuda_projection(
                    gpu_backend.get(), target_buft, projection.tensor,
                    bundle_it->second.data() + span_it->destination_offset,
                    size_t(span_it->byte_count), universal_hot_stride,
                    layer_index, projection.name);
                all_kernel_comparisons_pass = all_kernel_comparisons_pass &&
                    comparison["finite"].get<bool>() && comparison["bit_exact"].get<bool>() &&
                    comparison["padding_guard_intact"].get<bool>();
                kernel_comparisons.push_back(std::move(comparison));
            }
        }

        const auto final_storage = storage->diagnostics();
        const auto async_diagnostics = async.diagnostics();
        const bool shutdown = async.shutdown();
        for (int fd : reference_fds) if (fd >= 0) close(fd);

        json output = {
            { "schema", "dsv4-source-span-proof-v1" },
            { "model_path", model_path },
            { "samples", std::move(sample_results) },
            { "kernel_comparison", {
                { "status", all_kernel_comparisons_pass ? "pass" : "fail" },
                { "representative_layers", representative_layers },
                { "universal_hot_stride", universal_hot_stride },
                { "hot_role_offsets", hot_role_offsets },
                { "hot_role_extents", hot_role_extents },
                { "comparison_count", kernel_comparisons.size() },
                { "comparisons", std::move(kernel_comparisons) },
            } },
            { "summary", {
                { "sample_count", samples.size() },
                { "cross_split_sample_present", cross_split_sample_added },
                { "all_source_sync_async_equal", all_equal },
                { "sample_bytes_per_path", total_sample_bytes },
                { "storage_read_requests", final_storage.read_requests },
                { "storage_read_bytes", final_storage.read_bytes },
                { "storage_short_reads", final_storage.short_reads },
                { "storage_io_errors", final_storage.io_errors },
                { "async_io_uring_enabled", async_diagnostics.io_uring_enabled },
                { "async_fallback_reason_mask", async_diagnostics.fallback_reason_mask },
                { "async_read_requests_submitted", async_diagnostics.read_requests_submitted },
                { "async_read_requests_completed", async_diagnostics.read_requests_completed },
                { "async_read_operations_completed", async_diagnostics.read_operations_completed },
                { "async_read_bytes_completed", async_diagnostics.read_bytes_completed },
                { "async_active_operations", async_diagnostics.active_operations },
                { "async_active_read_requests", async_diagnostics.active_read_requests },
                { "async_shutdown", shutdown },
                { "layout_class_count", representative_layers.size() },
                { "cuda_kernel_comparisons", representative_layers.size()*3 },
                { "all_cuda_kernel_comparisons_pass", all_kernel_comparisons_pass },
            } },
        };
        std::ofstream stream(output_path);
        if (!stream) throw std::runtime_error("cannot create output file");
        stream << output.dump(2) << '\n';
        stream.close();
        if (!stream) throw std::runtime_error("cannot write output file");

        std::cout << "DSV4_SOURCE_SPANS"
                  << "\tsamples=" << samples.size()
                  << "\tcross_split=" << cross_split_sample_added
                  << "\tall_equal=" << all_equal
                  << "\tbytes_per_path=" << total_sample_bytes
                  << "\tio_uring=" << async_diagnostics.io_uring_enabled
                  << "\tfallback_mask=" << async_diagnostics.fallback_reason_mask
                  << "\tkernel_comparisons=" << representative_layers.size()*3
                  << "\tkernel_pass=" << all_kernel_comparisons_pass
                  << '\n';
        result_code = all_equal && cross_split_sample_added && shutdown && all_kernel_comparisons_pass ? 0 : 5;
    } catch (const std::exception & error) {
        std::cerr << "dsv4-source-span-probe: " << error.what() << '\n';
    }
    llama_model_free(model);
    return result_code;
}
