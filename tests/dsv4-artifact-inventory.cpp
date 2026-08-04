#include "llama-model.h"

#include "llama.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

namespace {

struct projection {
    const char * role;
    ggml_tensor * weight;
    ggml_tensor * bias;
    ggml_tensor * scale;
};

bool parse_u64(const char * text, uint64_t & result) {
    char * end = nullptr;
    const auto value = std::strtoull(text, &end, 10);
    if (end == text || *end != '\0') return false;
    result = value;
    return true;
}

uint64_t align_down(uint64_t value, uint64_t alignment) {
    return value - value % alignment;
}

bool align_up(uint64_t value, uint64_t alignment, uint64_t & result) {
    const uint64_t remainder = value % alignment;
    if (remainder == 0) {
        result = value;
        return true;
    }
    const uint64_t delta = alignment - remainder;
    if (value > UINT64_MAX - delta) return false;
    result = value + delta;
    return true;
}

std::string layout_component(const llama_model_tensor_storage_metadata & metadata, int expert_axis) {
    std::string result = std::to_string(int(metadata.type)) + ":" + std::to_string(metadata.n_dims) + ":";
    for (uint32_t axis = 0; axis < GGML_MAX_DIMS; ++axis) {
        if (int(axis) != expert_axis) {
            result += std::to_string(metadata.logical_shape[axis]) + "/" +
                std::to_string(metadata.physical_strides[axis]) + ",";
        }
    }
    result += "expert-stride=" + std::to_string(metadata.physical_strides[expert_axis]);
    return result;
}

json tensor_metadata(
        const llama_model * model,
        const ggml_tensor * tensor,
        const std::vector<llama_model_source_file_metadata> & files) {
    llama_model_tensor_storage_metadata metadata = {};
    const char * name = ggml_get_name(tensor);
    if (llama_model_get_tensor_storage_metadata(model, name, &metadata) != LLAMA_MODEL_STORAGE_STATUS_OK) {
        throw std::runtime_error(std::string("missing storage metadata for ") + name);
    }
    if (metadata.source_file_index >= files.size()) {
        throw std::runtime_error(std::string("invalid source file index for ") + name);
    }
    json shape = json::array();
    json strides = json::array();
    for (uint32_t axis = 0; axis < metadata.n_dims; ++axis) {
        shape.push_back(metadata.logical_shape[axis]);
        strides.push_back(metadata.physical_strides[axis]);
    }
    return {
        { "name", name },
        { "ggml_type", ggml_type_name(metadata.type) },
        { "ggml_type_id", int(metadata.type) },
        { "shape", std::move(shape) },
        { "physical_strides", std::move(strides) },
        { "logical_bytes", ggml_nbytes(tensor) },
        { "source_file_index", metadata.source_file_index },
        { "source_file_identity", files[metadata.source_file_index].identity },
        { "source_file_size", files[metadata.source_file_index].size },
        { "file_offset", metadata.file_offset },
        { "file_bytes", metadata.byte_size },
        { "gguf_alignment", metadata.gguf_alignment },
        { "runtime_buffer_type", metadata.runtime_buffer_type ? metadata.runtime_buffer_type : "" },
        { "runtime_layout_transform", metadata.runtime_layout_transform },
        { "runtime_backend_transform", metadata.runtime_backend_transform },
        { "runtime_repack", metadata.runtime_repack },
        { "data_allocated", tensor->data != nullptr },
        { "buffer_allocated", tensor->buffer != nullptr },
    };
}

} // namespace

int main(int argc, char ** argv) {
    std::string model_path;
    std::string output_path;
    uint64_t direct_alignment = 4096;
    for (int index = 1; index < argc; ++index) {
        if (index + 1 >= argc) return 2;
        const std::string option = argv[index];
        const char * value = argv[++index];
        if (option == "--model") model_path = value;
        else if (option == "--output") output_path = value;
        else if (option == "--direct-alignment" && parse_u64(value, direct_alignment)) {}
        else return 2;
    }
    if (model_path.empty() || output_path.empty() || direct_alignment == 0) return 2;

    ggml_backend_load_all();
    llama_model_params params = llama_model_default_params();
    params.no_alloc = true;
    params.load_mode = LLAMA_LOAD_MODE_NONE;
    params.n_gpu_layers = -1;
    params.expert_weights_mode = LLAMA_EXPERT_WEIGHTS_MODE_DISABLED;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), params);
    if (!model) return 3;

    try {
        uint32_t file_count = 0;
        if (llama_model_source_file_count(model, &file_count) != LLAMA_MODEL_STORAGE_STATUS_OK || file_count == 0) {
            throw std::runtime_error("source file identities are unavailable");
        }
        std::vector<llama_model_source_file_metadata> files(file_count);
        json source_files = json::array();
        for (uint32_t index = 0; index < file_count; ++index) {
            if (llama_model_get_source_file_metadata(model, index, &files[index]) != LLAMA_MODEL_STORAGE_STATUS_OK) {
                throw std::runtime_error("source file metadata is incomplete");
            }
            source_files.push_back({
                { "index", files[index].index },
                { "identity", files[index].identity },
                { "size", files[index].size },
                { "gguf_alignment", files[index].gguf_alignment },
            });
        }

        const int32_t n_expert = int32_t(model->hparams.n_expert);
        const int32_t n_expert_used = int32_t(model->hparams.n_expert_used);
        if (n_expert <= 0 || n_expert_used <= 0 || n_expert_used > n_expert) {
            throw std::runtime_error("model has no valid routed-expert topology");
        }

        json routed_layers = json::array();
        std::set<const ggml_tensor *> routed_tensors;
        std::map<std::string, std::vector<int32_t>> layout_layers;
        uint64_t max_bundle_bytes = 0;
        uint64_t max_direct_bytes = 0;
        uint64_t routed_payload_bytes = 0;
        uint64_t payload_backed_tensor_count = 0;
        bool all_expert_axes_contiguous = true;
        bool any_bundle_crosses_source_files = false;

        for (size_t layer_index = 0; layer_index < model->layers.size(); ++layer_index) {
            const auto & layer = model->layers[layer_index];
            const std::array<projection, 4> projections = {{
                { "up", layer.ffn_up_exps, layer.ffn_up_exps_b, layer.ffn_up_exps_s },
                { "gate", layer.ffn_gate_exps, layer.ffn_gate_exps_b, layer.ffn_gate_exps_s },
                { "gate_up", layer.ffn_gate_up_exps, layer.ffn_gate_up_exps_b, nullptr },
                { "down", layer.ffn_down_exps, layer.ffn_down_exps_b, layer.ffn_down_exps_s },
            }};
            if (std::none_of(projections.begin(), projections.end(), [](const projection & item) {
                    return item.weight != nullptr;
                })) {
                continue;
            }

            json tensors = json::array();
            std::string layout_fingerprint;
            uint64_t bundle_bytes = 0;
            uint64_t bundle_direct_bytes = 0;
            std::set<uint32_t> bundle_files;
            for (const auto & item : projections) {
                for (const auto & part : std::array<std::pair<const char *, ggml_tensor *>, 3> {{
                        { "weight", item.weight }, { "bias", item.bias }, { "scale", item.scale } }}) {
                    if (part.second == nullptr) continue;
                    routed_tensors.insert(part.second);
                    payload_backed_tensor_count += part.second->data != nullptr;
                    llama_model_tensor_storage_metadata metadata = {};
                    const char * name = ggml_get_name(part.second);
                    if (llama_model_get_tensor_storage_metadata(model, name, &metadata) != LLAMA_MODEL_STORAGE_STATUS_OK) {
                        throw std::runtime_error(std::string("missing routed metadata for ") + name);
                    }
                    const bool weight = std::string(part.first) == "weight";
                    int expert_axis = -1;
                    if (weight && metadata.n_dims >= 3 && metadata.logical_shape[2] == n_expert) {
                        expert_axis = 2;
                    } else if (!weight) {
                        for (int axis = int(metadata.n_dims) - 1; axis >= 0; --axis) {
                            if (metadata.logical_shape[axis] == n_expert) {
                                if (expert_axis >= 0) {
                                    expert_axis = -1;
                                    break;
                                }
                                expert_axis = axis;
                            }
                        }
                    }
                    if (expert_axis < 0 || metadata.physical_strides[expert_axis] == 0 ||
                            metadata.byte_size != metadata.physical_strides[expert_axis]*uint64_t(n_expert)) {
                        all_expert_axes_contiguous = false;
                        throw std::runtime_error(std::string("non-contiguous expert axis for ") + name);
                    }
                    const uint64_t expert_bytes = metadata.physical_strides[expert_axis];
                    uint64_t tensor_max_direct = 0;
                    for (int32_t expert = 0; expert < n_expert; ++expert) {
                        const uint64_t offset = metadata.file_offset + uint64_t(expert)*expert_bytes;
                        uint64_t aligned_end = 0;
                        if (offset < metadata.file_offset || expert_bytes > UINT64_MAX - offset ||
                                !align_up(offset + expert_bytes, direct_alignment, aligned_end)) {
                            throw std::runtime_error(std::string("direct-I/O extent overflow for ") + name);
                        }
                        tensor_max_direct = std::max(tensor_max_direct,
                            aligned_end - align_down(offset, direct_alignment));
                    }
                    json entry = tensor_metadata(model, part.second, files);
                    entry["projection"] = item.role;
                    entry["component"] = part.first;
                    entry["expert_axis"] = expert_axis;
                    entry["per_expert_bytes"] = expert_bytes;
                    entry["max_direct_aligned_per_expert_bytes"] = tensor_max_direct;
                    tensors.push_back(std::move(entry));
                    layout_fingerprint += std::string(item.role) + "." + part.first + "=" +
                        layout_component(metadata, expert_axis) + ";";
                    bundle_bytes += expert_bytes;
                    bundle_direct_bytes += tensor_max_direct;
                    routed_payload_bytes += metadata.byte_size;
                    bundle_files.insert(metadata.source_file_index);
                }
            }
            max_bundle_bytes = std::max(max_bundle_bytes, bundle_bytes);
            max_direct_bytes = std::max(max_direct_bytes, bundle_direct_bytes);
            any_bundle_crosses_source_files = any_bundle_crosses_source_files || bundle_files.size() > 1;
            layout_layers[layout_fingerprint].push_back(int32_t(layer_index));
            routed_layers.push_back({
                { "layer", layer_index },
                { "expert_count", n_expert },
                { "selected_top_k", n_expert_used },
                { "expert_id_domain", { 0, n_expert - 1 } },
                { "bundle_bytes", bundle_bytes },
                { "max_direct_aligned_bundle_bytes", bundle_direct_bytes },
                { "source_file_indices", bundle_files },
                { "crosses_source_files", bundle_files.size() > 1 },
                { "layout_fingerprint", layout_fingerprint },
                { "tensors", std::move(tensors) },
            });
        }

        json resident_tensors = json::array();
        json resident_shared_experts = json::array();
        uint64_t resident_payload_bytes = 0;
        for (const auto & named : model->tensors_by_name) {
            if (routed_tensors.count(named.second) != 0) continue;
            auto entry = tensor_metadata(model, named.second, files);
            payload_backed_tensor_count += named.second->data != nullptr;
            resident_payload_bytes += uint64_t(entry["file_bytes"]);
            if (named.first.find("shexp") != std::string::npos) {
                resident_shared_experts.push_back(entry);
            }
            resident_tensors.push_back(std::move(entry));
        }

        json layouts = json::array();
        for (const auto & item : layout_layers) {
            layouts.push_back({
                { "fingerprint", item.first },
                { "layers", item.second },
            });
        }

        json result = {
            { "schema", "dsv4-artifact-inventory-v1" },
            { "model_path", model_path },
            { "architecture", model->arch_name() },
            { "description", model->desc() },
            { "load", {
                { "expert_weights_mode", "DISABLED" },
                { "no_alloc", true },
                { "payload_backed_tensor_count", payload_backed_tensor_count },
                { "payload_read_bytes", 0 },
            } },
            { "topology", {
                { "model_layer_count", model->layers.size() },
                { "routed_layer_count", routed_layers.size() },
                { "expert_count", n_expert },
                { "selected_top_k", n_expert_used },
            } },
            { "source_files", std::move(source_files) },
            { "routed_layers", std::move(routed_layers) },
            { "resident_shared_experts", std::move(resident_shared_experts) },
            { "resident_non_routed_tensors", std::move(resident_tensors) },
            { "summary", {
                { "routed_tensor_count", routed_tensors.size() },
                { "routed_payload_bytes", routed_payload_bytes },
                { "resident_payload_bytes", resident_payload_bytes },
                { "max_bundle_bytes", max_bundle_bytes },
                { "direct_io_alignment", direct_alignment },
                { "max_direct_aligned_bundle_bytes", max_direct_bytes },
                { "all_expert_axes_contiguous", all_expert_axes_contiguous },
                { "any_bundle_crosses_source_files", any_bundle_crosses_source_files },
                { "layout_class_count", layout_layers.size() },
                { "single_fixed_slot_layout_representable", layout_layers.size() == 1 },
                { "layout_classes", std::move(layouts) },
            } },
        };

        std::ofstream output(output_path);
        if (!output) throw std::runtime_error("cannot create output file");
        output << result.dump(2) << '\n';
        output.close();
        if (!output) throw std::runtime_error("cannot write output file");

        std::cout << "DSV4_INVENTORY"
                  << "\trouted_layers=" << result["topology"]["routed_layer_count"]
                  << "\trouted_tensors=" << result["summary"]["routed_tensor_count"]
                  << "\tlayout_classes=" << result["summary"]["layout_class_count"]
                  << "\tsingle_fixed_slot_layout=" << result["summary"]["single_fixed_slot_layout_representable"]
                  << "\tmax_bundle_bytes=" << result["summary"]["max_bundle_bytes"]
                  << "\tmax_direct_bytes=" << result["summary"]["max_direct_aligned_bundle_bytes"]
                  << "\tpayload_backed_tensors=" << result["load"]["payload_backed_tensor_count"]
                  << '\n';
        llama_model_free(model);
        return layout_layers.size() == 1 ? 0 : 10;
    } catch (const std::exception & error) {
        std::cerr << "dsv4-artifact-inventory: " << error.what() << '\n';
        llama_model_free(model);
        return 4;
    }
}
