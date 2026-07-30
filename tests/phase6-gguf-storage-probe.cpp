#include "llama-model.h"
#include "llama-expert-storage.h"

#include "llama.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

namespace {

bool parse_u64(const char * text, uint64_t & result) {
    char * end = nullptr;
    const auto value = std::strtoull(text, &end, 10);
    if (end == text || *end != '\0') return false;
    result = value;
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    std::string model_path;
    uint64_t capacity = 2;
    uint64_t cold_bytes = 64U*1024U*1024U;
    uint64_t ring_bytes = 16U*1024U*1024U;
    for (int index = 1; index < argc; ++index) {
        if (index + 1 >= argc) return 2;
        const std::string option = argv[index];
        const char * value = argv[++index];
        if (option == "--model") model_path = value;
        else if (option == "--capacity" && parse_u64(value, capacity)) {}
        else if (option == "--cold-bytes" && parse_u64(value, cold_bytes)) {}
        else if (option == "--ring-bytes" && parse_u64(value, ring_bytes)) {}
        else return 2;
    }
    if (model_path.empty() || capacity == 0 || capacity > UINT32_MAX || cold_bytes == 0 || ring_bytes == 0) return 2;

    ggml_backend_load_all();
    const llama_model_tensor_buft_override overrides[] = {
        { "ffn_(gate|up|down)_exps\\.(weight|bias|scale)", ggml_backend_cpu_buffer_type() },
        { nullptr, nullptr },
    };
    llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = -1;
    params.load_mode = LLAMA_LOAD_MODE_MMAP;
    params.expert_weights_mode = LLAMA_EXPERT_WEIGHTS_MODE_COLD_CACHE;
    params.expert_hot_cache_capacity = uint32_t(capacity);
    params.expert_cold_cache_bytes = cold_bytes;
    params.expert_transfer_ring_bytes = ring_bytes;
    params.tensor_buft_overrides = overrides;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), params);
    if (!model) return 3;

    const auto deferred = model->deferred_expert_diagnostics();
    const auto * storage = model->expert_storage();
    if (!storage) return 4;
    const auto storage_diagnostics = storage->diagnostics();
    auto * provider = model->expert_weight_provider();
    const auto trim_result = provider ? provider->trim() : llm_expert_provider_result::failure(llm_expert_provider_error::initialization_failed);
    const bool trim_retains_storage = provider && model->expert_storage() == storage &&
        model->expert_storage()->diagnostics().sealed;
    const bool surrender_retains_storage = provider && provider->surrender().is_ready() &&
        model->expert_storage() == storage && model->expert_storage()->diagnostics().sealed;
    uint64_t routed_tensors = 0;
    uint64_t routed_null = 0;
    for (const auto & layer : model->layers) {
        for (const auto * tensor : {
                layer.ffn_up_exps, layer.ffn_up_exps_b, layer.ffn_up_exps_s,
                layer.ffn_gate_exps, layer.ffn_gate_exps_b, layer.ffn_gate_exps_s,
                layer.ffn_gate_up_exps, layer.ffn_gate_up_exps_b,
                layer.ffn_down_exps, layer.ffn_down_exps_b, layer.ffn_down_exps_s }) {
            if (!tensor) continue;
            routed_tensors++;
            if (tensor->data == nullptr && tensor->buffer == nullptr) routed_null++;
        }
    }
    const bool resident_loaded = model->tok_embd && model->tok_embd->data && model->tok_embd->buffer;
    std::cout << "PHASE6_LOAD"
              << "\tdeferred_tensors=" << deferred.tensor_count
              << "\tdeferred_payload_bytes=" << deferred.payload_bytes
              << "\tdeferred_allocated_bytes=" << deferred.allocated_bytes
              << "\tdeferred_mmap_bound_bytes=" << deferred.mmap_bound_bytes
              << "\tdeferred_prefetch_bytes=" << deferred.prefetched_bytes
              << "\tresident_loaded_bytes=" << deferred.resident_loaded_bytes
              << "\tprefetch_disabled=" << deferred.full_file_prefetch_disabled
              << "\trouted_tensors=" << routed_tensors
              << "\trouted_null=" << routed_null
              << "\tresident_loaded=" << resident_loaded
              << "\tstorage_files=" << storage_diagnostics.source_file_count
              << "\tstorage_entries=" << storage_diagnostics.directory_entry_count
              << "\tstorage_spans=" << storage_diagnostics.span_count
              << "\tstorage_admin_bytes=" << storage_diagnostics.administration_bytes
              << "\tstorage_reads=" << storage_diagnostics.read_requests
              << "\ttrim_retains_storage=" << trim_retains_storage
              << "\ttrim_ready=" << trim_result.is_ready()
              << "\tsurrender_retains_storage=" << surrender_retains_storage
              << '\n';
    const bool valid = deferred.tensor_count == routed_tensors && routed_tensors == routed_null &&
        deferred.payload_bytes > 0 && deferred.allocated_bytes == 0 && deferred.mmap_bound_bytes == 0 &&
        deferred.prefetched_bytes == 0 && deferred.full_file_prefetch_disabled && resident_loaded &&
        storage_diagnostics.sealed && storage_diagnostics.source_file_count > 0 &&
        storage_diagnostics.directory_entry_count > 0 && storage_diagnostics.span_count >= storage_diagnostics.directory_entry_count*2 &&
        storage_diagnostics.read_requests == 0 && trim_retains_storage && surrender_retains_storage;
    llama_model_free(model);
    return valid ? 0 : 5;
}
