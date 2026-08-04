#include "llama-expert-transfer-ring.h"
#include "llama-expert-storage.h"
#include "llama-expert-async-io.h"
#include "llama-expert-scheduler.h"
#include "llama-context.h"
#include "llama-model.h"
#include "llama.h"

#include "ggml-cpp.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <unistd.h>

namespace {

bool controlled_native_overlap(ggml_backend_dev_t device, ggml_backend_t backend);

uint64_t hash_bytes(uint64_t hash, const void * data, size_t size) {
    const auto * bytes = static_cast<const uint8_t *>(data);
    for (size_t index = 0; index < size; ++index) {
        hash ^= bytes[index];
        hash *= 1099511628211ULL;
    }
    return hash;
}

bool same_flight(const llm_expert_flight_id & lhs, const llm_expert_flight_id & rhs) {
    return lhs.transport_epoch == rhs.transport_epoch && lhs.request_slot == rhs.request_slot &&
        lhs.request_generation == rhs.request_generation && lhs.key.layer == rhs.key.layer &&
        lhs.key.expert == rhs.key.expert;
}

uint64_t overlap_pair_hash(
        const llm_expert_async_read_interval & read,
        const llm_transfer_interval & transfer) {
    uint64_t hash = 1469598103934665603ULL;
    const auto append = [&](uint64_t value) {
        for (size_t byte = 0; byte < sizeof(value); ++byte) {
            hash ^= (value >> (byte*8)) & 0xffU;
            hash *= 1099511628211ULL;
        }
    };
    append(read.flight.transport_epoch);
    append(read.flight.request_slot);
    append(read.flight.request_generation);
    append(uint32_t(read.flight.key.layer));
    append(uint32_t(read.flight.key.expert));
    append(read.operation_index);
    append(transfer.flight.transport_epoch);
    append(transfer.flight.request_slot);
    append(transfer.flight.request_generation);
    append(uint32_t(transfer.flight.key.layer));
    append(uint32_t(transfer.flight.key.expert));
    append(transfer.lane.generation);
    append(transfer.hot_generation);
    return hash;
}

struct route_hash {
    uint64_t value = 1469598103934665603ULL;
    uint64_t records = 0;
};

struct execution_id_placement {
    llama_context * context = nullptr;
    uint64_t cpu = 0;
    uint64_t non_cpu = 0;
    int32_t backend_device_type = -1;
    std::string backend_name;
    bool backend_consistent = true;
    uint64_t routing_cpu = 0;
    uint64_t routing_non_cpu = 0;
    int32_t routing_backend_device_type = -1;
    std::string routing_backend_name;
    bool routing_backend_consistent = true;
};

bool capture_execution_id_placement(ggml_tensor * tensor, bool ask, void * user_data) {
    auto * placement = static_cast<execution_id_placement *>(user_data);
    const bool execution_ids = std::strncmp(tensor->name, "expert_execution_ids-", 21) == 0;
    const bool routing_ids = std::strncmp(tensor->name, "ffn_moe_topk-", 13) == 0;
    if (!execution_ids && !routing_ids) return false;
    if (!ask) return true;
    if (placement->context == nullptr) return true;
    ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(
        placement->context->get_sched(), tensor);
    ggml_backend_dev_t device = backend == nullptr ? nullptr : ggml_backend_get_device(backend);
    const int32_t device_type = device == nullptr ? -1 : int32_t(ggml_backend_dev_type(device));
    const std::string backend_name = backend == nullptr ? "" : ggml_backend_name(backend);
    uint64_t & cpu = execution_ids ? placement->cpu : placement->routing_cpu;
    uint64_t & non_cpu = execution_ids ? placement->non_cpu : placement->routing_non_cpu;
    int32_t & observed_device_type = execution_ids ?
        placement->backend_device_type : placement->routing_backend_device_type;
    std::string & observed_backend_name = execution_ids ?
        placement->backend_name : placement->routing_backend_name;
    bool & consistent = execution_ids ?
        placement->backend_consistent : placement->routing_backend_consistent;
    if (cpu + non_cpu == 0) {
        observed_device_type = device_type;
        observed_backend_name = backend_name;
    } else if (observed_device_type != device_type || observed_backend_name != backend_name) {
        consistent = false;
    }
    if (device_type == GGML_BACKEND_DEVICE_TYPE_CPU) cpu++;
    else non_cpu++;
    return true;
}

bool capture_route(const llama_route_observation * observation, void * user_data) {
    auto * state = static_cast<route_hash *>(user_data);
    const size_t count = size_t(observation->n_tokens)*observation->n_expert_used;
    state->value = hash_bytes(state->value, &observation->layer, sizeof(observation->layer));
    state->value = hash_bytes(state->value, observation->selected_experts, count*sizeof(int32_t));
    state->value = hash_bytes(state->value, observation->weights, count*sizeof(float));
    state->records++;
    return true;
}

struct live_arguments {
    std::string model;
    std::string mode;
    uint32_t capacity = 0;
    uint64_t cold_bytes = 0;
    uint64_t ring_bytes = 0;
    int steps = 5;
    llama_load_mode load_mode = LLAMA_LOAD_MODE_MMAP;
    bool cancel_on_storage = false;
    bool cancel_after_h2d = false;
    bool require_overlap = false;
    std::string dump_cold_bundle;
};

bool parse_u64(const char * text, uint64_t & result) {
    char * end = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (end == text || *end != '\0') return false;
    result = value;
    return true;
}

bool parse_live(int argc, char ** argv, live_arguments & result) {
    for (int index = 1; index < argc; ++index) {
        if (std::string(argv[index]) == "--cancel-on-storage") {
            result.cancel_on_storage = true;
            continue;
        }
        if (std::string(argv[index]) == "--cancel-after-h2d") {
            result.cancel_after_h2d = true;
            continue;
        }
        if (std::string(argv[index]) == "--require-overlap") {
            result.require_overlap = true;
            continue;
        }
        if (index + 1 >= argc) return false;
        const std::string option = argv[index];
        const char * value = argv[++index];
        uint64_t parsed = 0;
        if (option == "--model") result.model = value;
        else if (option == "--mode") result.mode = value;
        else if (option == "--capacity" && parse_u64(value, parsed) && parsed <= UINT32_MAX) result.capacity = parsed;
        else if (option == "--cold-bytes" && parse_u64(value, result.cold_bytes)) {}
        else if (option == "--ring-bytes" && parse_u64(value, result.ring_bytes)) {}
        else if (option == "--steps" && parse_u64(value, parsed) && parsed > 0 && parsed <= 64) result.steps = parsed;
        else if (option == "--dump-cold-bundle") result.dump_cold_bundle = value;
        else if (option == "--load-mode") {
            try {
                result.load_mode = llama_load_mode_from_str(value);
            } catch (const std::invalid_argument &) {
                return false;
            }
        }
        else return false;
    }
    const bool cached_mode = result.mode == "hot" || result.mode == "cold" || result.mode == "uma";
    return !(result.cancel_on_storage && result.cancel_after_h2d) && !result.model.empty() &&
        (result.mode == "disabled" || result.mode == "resident" || cached_mode) &&
        (!cached_mode || result.capacity > 0) &&
        (result.mode != "cold" || (result.cold_bytes > 0 && result.ring_bytes > 0)) &&
        (result.mode != "uma" || result.ring_bytes == 0);
}

struct storage_cancel_state {
    const llama_model * model = nullptr;
};

struct post_h2d_cancel_state {
    const llama_model * model = nullptr;
    uint64_t initial_compute_waits = 0;
    bool observed_live_h2d = false;
};

bool cancel_after_provider_compute_wait(void * user_data) {
    auto * state = static_cast<post_h2d_cancel_state *>(user_data);
    if (state == nullptr || state->model == nullptr || state->model->expert_weight_provider() == nullptr) return false;
    const auto diagnostics = state->model->expert_weight_provider()->hot_cache_diagnostics();
    const bool cancel = diagnostics.ring_compute_waits > state->initial_compute_waits &&
        diagnostics.ring_live_h2d_events > 0 && !diagnostics.last_completed_flight.valid();
    state->observed_live_h2d = state->observed_live_h2d || cancel;
    return cancel;
}

bool cancel_after_first_storage_read(void * user_data) {
    const auto * state = static_cast<const storage_cancel_state *>(user_data);
    return state && state->model && state->model->expert_async_diagnostics().read_bytes_completed > 0;
}

int run_live(int argc, char ** argv) {
    live_arguments args;
    if (!parse_live(argc, argv, args)) return 20;
    ggml_backend_load_all();
    const llama_model_tensor_buft_override overrides[] = {
        { "ffn_(gate|up|down)_exps\\.weight", ggml_backend_cpu_buffer_type() },
        { nullptr, nullptr },
    };
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = -1;
    model_params.load_mode = args.load_mode;
    if (args.mode == "resident") {
        model_params.expert_weights_mode = LLAMA_EXPERT_WEIGHTS_MODE_RESIDENT;
    } else if (args.mode == "hot" || args.mode == "cold" || args.mode == "uma") {
        model_params.tensor_buft_overrides = overrides;
        model_params.expert_hot_cache_capacity = args.capacity;
        model_params.expert_weights_mode = args.mode == "hot" ? LLAMA_EXPERT_WEIGHTS_MODE_HOT_CACHE :
            args.mode == "cold" ? LLAMA_EXPERT_WEIGHTS_MODE_COLD_CACHE : LLAMA_EXPERT_WEIGHTS_MODE_UMA_CACHE;
        model_params.expert_cold_cache_bytes = args.cold_bytes;
        model_params.expert_transfer_ring_bytes = args.ring_bytes;
    }
    llama_model * model = llama_model_load_from_file(args.model.c_str(), model_params);
    if (!model) return 21;
    llama_context_params context_params = llama_context_default_params();
    execution_id_placement placement;
    context_params.n_ctx = 64;
    context_params.n_batch = 64;
    context_params.n_ubatch = 1;
    context_params.cb_eval = capture_execution_id_placement;
    context_params.cb_eval_user_data = &placement;
    llama_context * context = llama_init_from_model(model, context_params);
    if (!context) return 22;
    placement.context = context;

    if (args.cancel_after_h2d) {
        if (args.mode != "cold" || model->expert_storage() == nullptr ||
            model->expert_weight_provider() == nullptr) return 36;
        auto * provider = model->expert_weight_provider();
        const auto initial = provider->hot_cache_diagnostics();
        if (!initial.ring_event_capable) return 37;
        auto * device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        ggml_backend_ptr gate_backend(device == nullptr ? nullptr : ggml_backend_dev_init(device, nullptr));
        if (device == nullptr || !gate_backend) return 38;
        ggml_init_params compute_params = { ggml_tensor_overhead()*4 + ggml_graph_overhead(), nullptr, true };
        ggml_context_ptr compute_ctx(ggml_init(compute_params));
        if (!compute_ctx) return 39;
        constexpr int64_t gate_dimension = 6144;
        ggml_tensor * gate_a = ggml_new_tensor_2d(
            compute_ctx.get(), GGML_TYPE_F32, gate_dimension, gate_dimension);
        ggml_tensor * gate_b = ggml_new_tensor_2d(
            compute_ctx.get(), GGML_TYPE_F32, gate_dimension, gate_dimension);
        ggml_tensor * gate_c = ggml_mul_mat(compute_ctx.get(), gate_a, gate_b);
        ggml_backend_buffer_ptr compute_buffer(
            ggml_backend_alloc_ctx_tensors_from_buft(compute_ctx.get(), ggml_backend_dev_buffer_type(device)));
        if (!compute_buffer) return 40;
        std::vector<float> zeros(size_t(gate_dimension)*gate_dimension, 0.0f);
        ggml_backend_tensor_set(gate_a, zeros.data(), 0, zeros.size()*sizeof(float));
        ggml_backend_tensor_set(gate_b, zeros.data(), 0, zeros.size()*sizeof(float));
        ggml_cgraph * gate_graph = ggml_new_graph_custom(compute_ctx.get(), 8, false);
        ggml_build_forward_expand(gate_graph, gate_c);
        ggml_backend_event_t gate_event = ggml_backend_event_new(device);
        if (gate_event == nullptr ||
            ggml_backend_graph_compute_async(gate_backend.get(), gate_graph) != GGML_STATUS_SUCCESS) return 41;
        ggml_backend_event_record(gate_event, gate_backend.get());
        if (!provider->set_h2d_gate_event_for_testing(gate_event).is_ready()) return 42;

        post_h2d_cancel_state cancel_state = { model, initial.ring_compute_waits, false };
        context->set_expert_abort_callback_for_testing(cancel_after_provider_compute_wait, &cancel_state);
        llama_token token = 1;
        const int cancelled = llama_decode(context, llama_batch_get_one(&token, 1));
        llama_synchronize(context);
        const auto cancelled_cache = provider->hot_cache_diagnostics();
        const auto cancelled_provider_stats = provider->get_stats();
        const auto cancelled_scheduler = model->expert_scheduler_diagnostics();
        uint64_t cancelled_mapping_generation = 0;
        uint32_t cancelled_mapping_slot = UINT32_MAX;
        const bool cancelled_mapping = provider->debug_hot_mapping(
            cancelled_cache.last_cancelled_flight.key,
            &cancelled_mapping_generation, &cancelled_mapping_slot);
        uint64_t cancelled_cold_generation = 0;
        uint32_t cancelled_cold_slot = UINT32_MAX;
        const bool cancelled_cold_ready = provider->debug_cold_ready(
            cancelled_cache.last_cancelled_flight.key,
            &cancelled_cold_generation, &cancelled_cold_slot);
        std::cerr << "PHASE7_PROVIDER_H2D_CANCEL_CHECKPOINT"
                  << "\tdecode_status=" << cancelled
                  << "\tabort_observed=" << cancel_state.observed_live_h2d
                  << "\tpost_h2d_cancellations=" << cancelled_cache.post_h2d_cancellations
                  << "\tprovider_cancellations=" << cancelled_provider_stats.cancellations
                  << "\tprovider_failures=" << cancelled_provider_stats.failures
                  << "\tlast_failure_error=" << int(cancelled_cache.last_failure_error)
                  << "\tlast_remap_error=" << int(cancelled_cache.last_remap_error)
                  << "\tcopy_failures=" << cancelled_cache.copy_failures
                  << "\tlast_completed_valid=" << cancelled_cache.last_completed_flight.valid()
                  << "\th2d_event_cancellations=" << cancelled_cache.ring_h2d_event_cancellations
                  << "\tlive_events=" << cancelled_cache.ring_live_events
                  << "\tscheduler_active=" << cancelled_scheduler.active_requests
                  << "\tcold_ready=" << cancelled_cold_ready
                  << '\n';

        context->set_expert_abort_callback_for_testing(nullptr, nullptr);
        if (!provider->set_h2d_gate_event_for_testing(nullptr).is_ready()) return 43;
        route_hash retry_routes;
        if (llama_set_route_observer(context, capture_route, &retry_routes) != LLAMA_ROUTE_OBSERVER_STATUS_OK ||
            llama_route_observer_begin(context, 1, LLAMA_ROUTE_PHASE_DECODE) != LLAMA_ROUTE_OBSERVER_STATUS_OK) return 44;
        const int retry = llama_decode(context, llama_batch_get_one(&token, 1));
        llama_synchronize(context);
        const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
        const float * retry_logits_ptr = llama_get_logits_ith(context, -1);
        if (retry != 0 || retry_logits_ptr == nullptr) return 45;
        const std::vector<float> retry_logits(retry_logits_ptr, retry_logits_ptr + n_vocab);
        const llama_token retry_token = std::max_element(retry_logits.begin(), retry_logits.end()) - retry_logits.begin();
        const auto retry_cache = provider->hot_cache_diagnostics();
        const auto retry_scheduler = model->expert_scheduler_diagnostics();
        uint64_t retry_mapping_generation = 0;
        uint32_t retry_mapping_slot = UINT32_MAX;
        const bool retry_mapping = provider->debug_hot_mapping(
            cancelled_cache.last_cancelled_flight.key,
            &retry_mapping_generation, &retry_mapping_slot);
        std::vector<uint8_t> cold_bytes;
        std::vector<uint8_t> hot_bytes;
        const bool exact_retry_bytes = provider->debug_copy_cold_bundle(
                cancelled_cache.last_cancelled_flight.key, cold_bytes).is_ready() &&
            provider->debug_copy_hot_bundle(
                cancelled_cache.last_cancelled_flight.key, hot_bytes).is_ready() &&
            !cold_bytes.empty() && cold_bytes == hot_bytes;

        llama_free(context);
        placement.context = nullptr;
        ggml_backend_event_free(gate_event);
        llama_model_free(model);

        llama_model_params baseline_params = llama_model_default_params();
        baseline_params.n_gpu_layers = -1;
        llama_model * baseline_model = llama_model_load_from_file(args.model.c_str(), baseline_params);
        if (!baseline_model) return 46;
        llama_context_params baseline_context_params = llama_context_default_params();
        baseline_context_params.n_ctx = 64;
        baseline_context_params.n_batch = 64;
        baseline_context_params.n_ubatch = 1;
        llama_context * baseline_context = llama_init_from_model(baseline_model, baseline_context_params);
        if (!baseline_context) return 47;
        route_hash baseline_routes;
        if (llama_set_route_observer(baseline_context, capture_route, &baseline_routes) != LLAMA_ROUTE_OBSERVER_STATUS_OK ||
            llama_route_observer_begin(baseline_context, 1, LLAMA_ROUTE_PHASE_DECODE) != LLAMA_ROUTE_OBSERVER_STATUS_OK ||
            llama_decode(baseline_context, llama_batch_get_one(&token, 1)) != 0) return 48;
        llama_synchronize(baseline_context);
        const float * baseline_logits_ptr = llama_get_logits_ith(baseline_context, -1);
        const bool exact_logits = baseline_logits_ptr != nullptr &&
            std::equal(retry_logits.begin(), retry_logits.end(), baseline_logits_ptr);
        const llama_token baseline_token = baseline_logits_ptr == nullptr ? -1 :
            std::max_element(baseline_logits_ptr, baseline_logits_ptr + n_vocab) - baseline_logits_ptr;
        const bool exact_routes = retry_routes.records == baseline_routes.records &&
            retry_routes.value == baseline_routes.value;

        const bool new_generations = retry_cache.retry_after_cancel_flight.valid() &&
            retry_cache.retry_after_cancel_flight.key.layer == cancelled_cache.last_cancelled_flight.key.layer &&
            retry_cache.retry_after_cancel_flight.key.expert == cancelled_cache.last_cancelled_flight.key.expert &&
            retry_cache.retry_after_cancel_flight.request_generation !=
                cancelled_cache.last_cancelled_flight.request_generation &&
            retry_cache.retry_after_cancel_lane_generation != cancelled_cache.last_cancelled_lane_generation &&
            retry_cache.retry_after_cancel_hot_generation > cancelled_cache.last_cancelled_hot_generation;
        const bool valid = cancelled == 2 && cancel_state.observed_live_h2d &&
            cancelled_cache.post_h2d_cancellations == 1 && cancelled_cache.admissions == 0 &&
            cancelled_cache.ring_h2d_event_cancellations > initial.ring_h2d_event_cancellations &&
            !cancelled_mapping && cancelled_mapping_generation == 0 && cancelled_mapping_slot == UINT32_MAX &&
            cancelled_cold_ready && cancelled_cold_generation > 0 && cancelled_cold_slot != UINT32_MAX &&
            cancelled_cache.current_pins == 0 && cancelled_cache.cold_current_hot_refs == 0 &&
            cancelled_cache.cold_current_transfer_refs == 0 && cancelled_cache.cold_current_request_refs == 0 &&
            cancelled_cache.ring_live_h2d_events == 0 && cancelled_cache.ring_live_compute_events == 0 &&
            cancelled_cache.ring_live_events == 0 && cancelled_scheduler.active_requests == 0 &&
            retry == 0 && retry_mapping && retry_mapping_generation == retry_cache.retry_after_cancel_hot_generation &&
            retry_mapping_slot == retry_cache.retry_after_cancel_hot_slot && retry_cache.admissions > 0 &&
            retry_cache.current_pins == 0 && retry_cache.cold_current_hot_refs > 0 &&
            retry_cache.cold_current_transfer_refs == 0 && retry_cache.cold_current_request_refs == 0 &&
            retry_cache.ring_live_events == 0 && retry_scheduler.active_requests == 0 && new_generations &&
            placement.cpu > 0 && placement.non_cpu == 0 &&
            retry_cache.last_execution_backend_device_type == GGML_BACKEND_DEVICE_TYPE_GPU &&
            exact_retry_bytes && exact_routes && exact_logits && retry_token == baseline_token;
        std::cout << "PHASE7_PROVIDER_H2D_CANCEL"
                  << "\tcancelled_status=" << cancelled
                  << "\tobserved_live_h2d=" << cancel_state.observed_live_h2d
                  << "\tcancelled_flight_generation=" << cancelled_cache.last_cancelled_flight.request_generation
                  << "\tcancelled_lane_generation=" << cancelled_cache.last_cancelled_lane_generation
                  << "\tcancelled_hot_generation=" << cancelled_cache.last_cancelled_hot_generation
                  << "\tcancelled_hot_admissions=" << cancelled_cache.admissions
                  << "\tcancelled_hot_mapping=" << cancelled_mapping
                  << "\tcancelled_cold_ready=" << cancelled_cold_ready
                  << "\tcancelled_cold_slot=" << cancelled_cold_slot
                  << "\tcancelled_cold_generation=" << cancelled_cold_generation
                  << "\th2d_event_cancellations=" << cancelled_cache.ring_h2d_event_cancellations
                  << "\tcancelled_scheduler_active=" << cancelled_scheduler.active_requests
                  << "\tcancelled_live_events=" << cancelled_cache.ring_live_events
                  << "\tcancelled_hot_refs=" << cancelled_cache.cold_current_hot_refs
                  << "\tcancelled_transfer_refs=" << cancelled_cache.cold_current_transfer_refs
                  << "\tcancelled_request_refs=" << cancelled_cache.cold_current_request_refs
                  << "\tretry_status=" << retry
                  << "\tretry_flight_generation=" << retry_cache.retry_after_cancel_flight.request_generation
                  << "\tretry_lane_generation=" << retry_cache.retry_after_cancel_lane_generation
                  << "\tretry_hot_generation=" << retry_cache.retry_after_cancel_hot_generation
                  << "\tretry_hot_mapping=" << retry_mapping
                  << "\tretry_hot_admissions=" << retry_cache.admissions
                  << "\tretry_scheduler_active=" << retry_scheduler.active_requests
                  << "\tretry_live_events=" << retry_cache.ring_live_events
                  << "\texecution_ids_cpu=" << placement.cpu
                  << "\texecution_ids_non_cpu=" << placement.non_cpu
                  << "\texecution_backend_device_type=" << retry_cache.last_execution_backend_device_type
                  << "\texact_bytes=" << exact_retry_bytes
                  << "\texact_routes=" << exact_routes
                  << "\texact_logits=" << exact_logits
                  << "\texact_token=" << (retry_token == baseline_token)
                  << '\n';
        llama_free(baseline_context);
        llama_model_free(baseline_model);
        return valid ? 0 : 49;
    }

    if (args.cancel_on_storage) {
        if (args.mode != "cold" || model->expert_storage() == nullptr) return 27;
        storage_cancel_state cancel_state = { model };
        llama_set_abort_callback(context, cancel_after_first_storage_read, &cancel_state);
        llama_token token = 1;
        const int cancelled = llama_decode(context, llama_batch_get_one(&token, 1));
        llama_synchronize(context);
        const auto cancelled_cache = model->expert_weight_provider()->hot_cache_diagnostics();
        const auto cancelled_storage = model->expert_storage()->diagnostics();
        llama_set_abort_callback(context, nullptr, nullptr);
        const int retry = llama_decode(context, llama_batch_get_one(&token, 1));
        llama_synchronize(context);
        const auto retry_cache = model->expert_weight_provider()->hot_cache_diagnostics();
        const auto retry_storage = model->expert_storage()->diagnostics();
        std::cout << "PHASE6_CANCEL"
                  << "\tcancelled_status=" << cancelled
                  << "\tcancelled_storage_reads=" << cancelled_storage.read_requests
                  << "\tcancelled_storage_bytes=" << cancelled_storage.read_bytes
                  << "\tcancelled_storage_cancellations=" << cancelled_storage.cancelled_reads
                  << "\tcancelled_hot_admissions=" << cancelled_cache.admissions
                  << "\tcancelled_cold_admissions=" << cancelled_cache.cold_admissions
                  << "\tcancelled_hot_refs=" << cancelled_cache.cold_current_hot_refs
                  << "\tcancelled_transfer_refs=" << cancelled_cache.cold_current_transfer_refs
                  << "\tcancelled_request_refs=" << cancelled_cache.cold_current_request_refs
                  << "\tcancelled_failed_cleanups=" << cancelled_cache.failed_cleanups
                  << "\tcancelled_cold_failed_cleanups=" << cancelled_cache.cold_failed_cleanups
                  << "\tretry_status=" << retry
                  << "\tretry_storage_reads=" << retry_storage.read_requests
                  << "\tretry_hot_admissions=" << retry_cache.admissions
                  << "\tretry_cold_admissions=" << retry_cache.cold_admissions
                  << '\n';
        const bool valid = cancelled == 2 && cancelled_storage.read_requests > 0 &&
            cancelled_storage.read_bytes > 0 &&
            cancelled_storage.cancelled_reads == cancelled_storage.read_requests &&
            cancelled_cache.admissions == 0 && cancelled_cache.cold_admissions == 0 &&
            cancelled_cache.cold_current_hot_refs == 0 && cancelled_cache.cold_current_transfer_refs == 0 &&
            cancelled_cache.cold_current_request_refs == 0 && cancelled_cache.failed_cleanups > 0 &&
            cancelled_cache.cold_failed_cleanups > 0 && retry == 0 &&
            retry_storage.read_requests > cancelled_storage.read_requests && retry_cache.admissions > 0 &&
            retry_cache.cold_admissions > 0;
        llama_free(context);
        llama_model_free(model);
        return valid ? 0 : 28;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const std::string prompt_text = "According to all known laws";
    const int prompt_count = -llama_tokenize(vocab, prompt_text.data(), prompt_text.size(), nullptr, 0, true, true);
    std::vector<llama_token> prompt(prompt_count);
    if (prompt_count <= 0 || llama_tokenize(vocab, prompt_text.data(), prompt_text.size(),
            prompt.data(), prompt.size(), true, true) != prompt_count) return 23;
    route_hash routes;
    if (llama_set_route_observer(context, capture_route, &routes) != LLAMA_ROUTE_OBSERVER_STATUS_OK) return 24;
    uint64_t logits_hash = 1469598103934665603ULL;
    std::vector<llama_token> generated;
    std::vector<uint64_t> decode_us;
    llama_batch batch = llama_batch_get_one(prompt.data(), prompt.size());
    for (int step = 0; step < args.steps; ++step) {
        const auto phase = step == 0 ? LLAMA_ROUTE_PHASE_PREFILL : LLAMA_ROUTE_PHASE_DECODE;
        const int64_t begin_us = ggml_time_us();
        if (llama_route_observer_begin(context, step, phase) != LLAMA_ROUTE_OBSERVER_STATUS_OK ||
            llama_decode(context, batch) != 0) return 25;
        decode_us.push_back(uint64_t(ggml_time_us() - begin_us));
        const float * logits = llama_get_logits_ith(context, -1);
        const int32_t n_vocab = llama_vocab_n_tokens(vocab);
        if (!logits) return 26;
        logits_hash = hash_bytes(logits_hash, logits, size_t(n_vocab)*sizeof(float));
        int32_t next = 0;
        for (int32_t token = 1; token < n_vocab; ++token) if (logits[token] > logits[next]) next = token;
        generated.push_back(next);
        if (llama_vocab_is_eog(vocab, next)) break;
        batch = llama_batch_get_one(&generated.back(), 1);
    }
    llama_synchronize(context);
    const auto diagnostics = model->expert_weight_provider() ?
        model->expert_weight_provider()->hot_cache_diagnostics() : llm_hot_cache_diagnostics {};
    const auto provider_stats = model->expert_weight_provider() ?
        model->expert_weight_provider()->get_stats() : llm_expert_provider_stats {};
    const auto storage_diagnostics = model->expert_storage() ?
        model->expert_storage()->diagnostics() : llm_expert_storage_diagnostics {};
    const auto async_diagnostics = model->expert_async_diagnostics();
    const auto scheduler_diagnostics = model->expert_scheduler_diagnostics();
    const bool resident_runtime_quiet = args.mode != "resident" ||
        (diagnostics.requested_capacity == 0 && diagnostics.effective_capacity == 0 &&
         diagnostics.pool_bytes == 0 && diagnostics.requests == 0 && diagnostics.hits == 0 &&
         diagnostics.misses == 0 && diagnostics.admissions == 0 && diagnostics.evictions == 0 &&
         diagnostics.h2d_bytes == 0 && diagnostics.slots.empty() &&
         diagnostics.cold_requested_bytes == 0 && diagnostics.cold_actual_bytes == 0 &&
         diagnostics.cold_effective_slots == 0 && diagnostics.cold_requests == 0 &&
         diagnostics.cold_hits == 0 && diagnostics.cold_misses == 0 &&
         diagnostics.cold_admissions == 0 && diagnostics.cold_evictions == 0 &&
         diagnostics.ring_requested_bytes == 0 && diagnostics.ring_actual_bytes == 0 &&
         diagnostics.ring_effective_lanes == 0 && diagnostics.ring_async_enqueues == 0 &&
         diagnostics.ring_synchronous_copies == 0 && diagnostics.ring_waves == 0 &&
         diagnostics.ring_h2d_bytes == 0 && diagnostics.ring_event_capacity == 0 &&
         storage_diagnostics.read_requests == 0 && storage_diagnostics.read_bytes == 0 &&
         !async_diagnostics.io_uring_enabled && async_diagnostics.actual_sq_entries == 0 &&
         async_diagnostics.actual_cq_entries == 0 && async_diagnostics.registered_file_count == 0 &&
         async_diagnostics.read_requests_submitted == 0 &&
         async_diagnostics.read_operations_completed == 0 && async_diagnostics.read_bytes_completed == 0 &&
         async_diagnostics.active_read_requests == 0 && async_diagnostics.active_operations == 0 &&
         scheduler_diagnostics.flights_created == 0 && scheduler_diagnostics.active_requests == 0);
    if (!args.dump_cold_bundle.empty()) {
        if (args.mode != "cold" || !model->expert_storage() || !model->expert_weight_provider()) return 29;
        llm_expert_key key = { -1, -1 };
        for (const auto & slot : diagnostics.slots) {
            if (slot.state == llm_hot_cache_diagnostics::slot::ready && slot.has_cold_backing) {
                key = { slot.layer, slot.expert };
                break;
            }
        }
        std::vector<uint8_t> bytes;
        if (!key.is_valid(LLAMA_MAX_LAYERS, diagnostics.n_expert) ||
            !model->expert_weight_provider()->debug_copy_cold_bundle(key, bytes).is_ready()) return 30;
        const auto * spans = model->expert_storage()->find(key);
        if (!spans || spans->empty()) return 31;
        std::ofstream output(args.dump_cold_bundle, std::ios::binary | std::ios::trunc);
        if (!output || !output.write(reinterpret_cast<const char *>(bytes.data()), bytes.size())) return 32;
        std::ostringstream span_records;
        uint64_t source_bytes = 0;
        for (size_t index = 0; index < spans->size(); ++index) {
            if (index) span_records << ',';
            span_records << (*spans)[index].split_index << ':' << (*spans)[index].file_offset << ':'
                         << (*spans)[index].byte_count;
            source_bytes += (*spans)[index].byte_count;
        }
        if (source_bytes != bytes.size()) return 33;
        std::cout << "PHASE6_BUNDLE"
                  << "\tlayer=" << key.layer
                  << "\texpert=" << key.expert
                  << "\tbytes=" << bytes.size()
                  << "\tspans=" << span_records.str()
                  << "\tdump=" << args.dump_cold_bundle
                  << '\n';
    }
    std::ostringstream tokens;
    for (size_t index = 0; index < generated.size(); ++index) {
        if (index) tokens << ',';
        tokens << generated[index];
    }
    std::ostringstream prompt_ids;
    for (size_t index = 0; index < prompt.size(); ++index) {
        if (index) prompt_ids << ',';
        prompt_ids << prompt[index];
    }
    std::vector<uint64_t> sorted_us = decode_us;
    std::sort(sorted_us.begin(), sorted_us.end());
    const auto percentile = [&](size_t numerator, size_t denominator) {
        if (sorted_us.empty()) return uint64_t(0);
        const size_t index = std::min(sorted_us.size() - 1,
            (sorted_us.size()*numerator + denominator - 1)/denominator - 1);
        return sorted_us[index];
    };
    uint64_t decode_total_us = 0;
    for (size_t index = 1; index < decode_us.size(); ++index) decode_total_us += decode_us[index];
    std::cout << (args.mode == "uma" ? "PHASE11_UMA_LIVE" : "PHASE5_LIVE")
              << "\tmode=" << args.mode
              << "\tload_mode=" << llama_load_mode_name(args.load_mode)
              << "\tprompt_ids=" << prompt_ids.str()
              << "\ttokens=" << tokens.str()
              << "\tlogits_hash=" << logits_hash
              << "\troute_hash=" << routes.value
              << "\troute_records=" << routes.records
              << "\tttft_us=" << (decode_us.empty() ? 0 : decode_us.front())
              << "\tprompt_tokens_per_second=" << (decode_us.empty() || decode_us.front() == 0 ? 0.0 : double(prompt.size())*1e6/decode_us.front())
              << "\tdecode_tokens_per_second=" << (decode_total_us == 0 ? 0.0 : double(decode_us.size() - 1)*1e6/decode_total_us)
              << "\ttoken_p50_us=" << percentile(50, 100)
              << "\ttoken_p95_us=" << percentile(95, 100)
              << "\ttoken_p99_us=" << percentile(99, 100)
              << "\thot_requested_capacity=" << diagnostics.requested_capacity
              << "\thot_effective_capacity=" << diagnostics.effective_capacity
              << "\thot_slots=" << diagnostics.slots.size()
              << "\thot_hits=" << diagnostics.hits
              << "\thot_misses=" << diagnostics.misses
              << "\thot_admissions=" << diagnostics.admissions
              << "\thot_evictions=" << diagnostics.evictions
              << "\tprovider_pool_bytes=" << provider_stats.pool_bytes
              << "\tprovider_pool_generations=" << provider_stats.pool_generations
              << "\tprovider_effective_capacity=" << provider_stats.effective_capacity
              << "\tprovider_tensor_copies=" << provider_stats.tensor_copies
              << "\tprovider_failures=" << provider_stats.failures
              << "\tcold_hits=" << diagnostics.cold_hits
              << "\tcold_misses=" << diagnostics.cold_misses
              << "\tcold_evictions=" << diagnostics.cold_evictions
              << "\tcold_actual_bytes=" << diagnostics.cold_actual_bytes
              << "\tcold_requested_bytes=" << diagnostics.cold_requested_bytes
              << "\tcold_unused_bytes=" << diagnostics.cold_unused_budget_bytes
              << "\tcold_bundle_payload=" << diagnostics.cold_bundle_payload_bytes
              << "\tcold_slot_footprint=" << diagnostics.cold_slot_footprint
              << "\tcold_alignment=" << diagnostics.cold_alignment
              << "\tcold_slots=" << diagnostics.cold_effective_slots
              << "\tcold_source_bytes=" << diagnostics.cold_source_copy_bytes
              << "\tcold_source_time_us=" << diagnostics.cold_source_copy_time_us
              << "\tcold_failed_copies=" << diagnostics.cold_failed_copies
              << "\tcold_failed_cleanups=" << diagnostics.cold_failed_cleanups
              << "\tcold_generation_changes=" << diagnostics.cold_generation_changes
              << "\tcold_reclaimed_bytes=" << diagnostics.cold_reclaimed_bytes
              << "\tcold_reclaim_failures=" << diagnostics.cold_reclaim_failures
              << "\thot_policy=" << int(diagnostics.policy.config.policy)
              << "\thot_policy_scope=" << int(diagnostics.policy.config.scope)
              << "\thot_policy_admission=" << int(diagnostics.policy.config.admission)
              << "\thot_policy_digest=" << diagnostics.policy.state_digest
              << "\thot_policy_drops=" << diagnostics.policy.transcript_dropped
              << "\tcold_policy=" << int(diagnostics.cold_policy.config.policy)
              << "\tcold_policy_scope=" << int(diagnostics.cold_policy.config.scope)
              << "\tcold_policy_admission=" << int(diagnostics.cold_policy.config.admission)
              << "\tcold_policy_digest=" << diagnostics.cold_policy.state_digest
              << "\tcold_policy_drops=" << diagnostics.cold_policy.transcript_dropped
              << "\tuma_physical_ram_bytes=" << diagnostics.uma_physical_ram_bytes
              << "\tuma_memory_available_bytes=" << diagnostics.uma_memory_available_bytes
              << "\tuma_cgroup_memory_max_bytes=" << diagnostics.uma_cgroup_memory_max_bytes
              << "\tuma_cgroup_memory_current_bytes=" << diagnostics.uma_cgroup_memory_current_bytes
              << "\tuma_process_rss_bytes=" << diagnostics.uma_process_rss_bytes
              << "\tuma_process_swap_bytes=" << diagnostics.uma_process_swap_bytes
              << "\tuma_safe_pool_bytes=" << diagnostics.uma_safe_pool_bytes
              << "\tuma_effective_pool_bytes=" << diagnostics.uma_effective_pool_bytes
              << "\tuma_system_reserve_bytes=" << diagnostics.uma_system_reserve_bytes
              << "\tuma_runtime_reserve_bytes=" << diagnostics.uma_runtime_reserve_bytes
              << "\tuma_runtime_delta_bytes=" << diagnostics.uma_runtime_delta_bytes
              << "\tuma_headroom_remainder_bytes=" << diagnostics.uma_headroom_remainder_bytes
              << "\tuma_autofit=" << diagnostics.uma_autofit
              << "\tuma_pressure_samples=" << diagnostics.uma_pressure_samples
              << "\tuma_pressure_rejections=" << diagnostics.uma_pressure_rejections
              << "\tuma_pressure_circuit_open=" << diagnostics.uma_pressure_circuit_open
              << "\tuma_swap_counters_supported=" << diagnostics.uma_swap_counters_supported
              << "\tuma_major_faults=" << diagnostics.uma_major_faults
              << "\tuma_psi_full_total_usec=" << diagnostics.uma_psi_full_total_usec
              << "\tuma_psi_full_supported=" << diagnostics.uma_psi_full_supported
              << "\tuma_zram_write_bytes=" << diagnostics.uma_zram_write_bytes
              << "\tuma_zram_present=" << diagnostics.uma_zram_present
              << "\tuma_zram_counters_supported=" << diagnostics.uma_zram_counters_supported
              << "\tuma_zram_status_reason=" << diagnostics.uma_zram_status_reason
              << "\tuma_zswap_write_pages=" << diagnostics.uma_zswap_write_pages
              << "\tuma_zswap_enabled=" << diagnostics.uma_zswap_enabled
              << "\tuma_zswap_counters_supported=" << diagnostics.uma_zswap_counters_supported
              << "\tuma_zswap_status_reason=" << diagnostics.uma_zswap_status_reason
              << "\tuma_nvidia_hmm_counters_supported=" << diagnostics.uma_nvidia_hmm_counters_supported
              << "\tuma_nvidia_hmm_status_reason=" << diagnostics.uma_nvidia_hmm_status_reason
              << "\tuma_storage_misses=" << diagnostics.uma_storage_misses
              << "\tuma_resident_hot_hits=" << diagnostics.uma_resident_hot_hits
              << "\tuma_prepared_cold_hits=" << diagnostics.uma_prepared_cold_hits
              << "\tuma_degraded_hits=" << diagnostics.uma_degraded_hits
              << "\tuma_unknown_residency_hits=" << diagnostics.uma_unknown_residency_hits
              << "\tuma_telemetry_unavailable_reason=" << diagnostics.uma_telemetry_unavailable_reason
              << "\tstorage_read_requests=" << storage_diagnostics.read_requests
              << "\tstorage_read_chunks=" << storage_diagnostics.read_chunks
              << "\tstorage_read_bytes=" << storage_diagnostics.read_bytes
              << "\tstorage_cancelled_reads=" << storage_diagnostics.cancelled_reads
              << "\tstorage_short_reads=" << storage_diagnostics.short_reads
              << "\tstorage_io_errors=" << storage_diagnostics.io_errors
              << "\tstorage_integrity_checks=" << storage_diagnostics.integrity_checks
              << "\tstorage_integrity_mismatches=" << storage_diagnostics.integrity_mismatches
              << "\tstorage_poisoned=" << storage_diagnostics.poisoned
              << "\tstorage_direct_sources=" << storage_diagnostics.direct_source_count
              << "\tstorage_direct_unsupported=" << storage_diagnostics.direct_unsupported_source_count
              << "\tstorage_direct_alignment=" << storage_diagnostics.maximum_direct_alignment
              << "\tio_async=" << async_diagnostics.io_uring_enabled
              << "\tio_setup_error=" << async_diagnostics.io_uring_setup_error
              << "\tio_probe_error=" << async_diagnostics.io_uring_probe_error
              << "\tio_runtime_error=" << async_diagnostics.io_uring_runtime_error
              << "\tio_sq_entries=" << async_diagnostics.actual_sq_entries
              << "\tio_cq_entries=" << async_diagnostics.actual_cq_entries
              << "\tio_registered_files=" << async_diagnostics.registered_file_count
              << "\tio_file_registration_error=" << async_diagnostics.file_registration_error
              << "\tio_requests=" << async_diagnostics.read_requests_submitted
              << "\tio_operations=" << async_diagnostics.read_operations_completed
              << "\tio_request_batches=" << async_diagnostics.ring_request_batches
              << "\tio_peak_batch_requests=" << async_diagnostics.peak_ring_batch_requests
              << "\tio_peak_sq_occupancy=" << async_diagnostics.peak_sq_occupancy
              << "\tio_peak_cq_occupancy=" << async_diagnostics.peak_cq_occupancy
              << "\tio_bytes=" << async_diagnostics.read_bytes_completed
              << "\tio_sync_fallback_operations=" << async_diagnostics.synchronous_fallback_operations
              << "\tio_registered_buffers=" << async_diagnostics.registered_buffer_count
              << "\tio_registered_buffer_bytes=" << async_diagnostics.registered_buffer_bytes
              << "\tio_buffer_registration_error=" << async_diagnostics.buffer_registration_error
              << "\tio_direct_staging_error=" << async_diagnostics.direct_staging_error
              << "\tio_direct_operations=" << async_diagnostics.direct_read_operations
              << "\tio_direct_useful_bytes=" << async_diagnostics.direct_useful_bytes
              << "\tio_direct_aligned_bytes=" << async_diagnostics.direct_aligned_bytes
              << "\tio_direct_scatter_bytes=" << async_diagnostics.direct_scatter_bytes
              << "\tio_buffered_fallback_operations=" << async_diagnostics.buffered_fallback_operations
              << "\tio_buffered_fallback_bytes=" << async_diagnostics.buffered_fallback_bytes
              << "\tio_direct_capability_retries=" << async_diagnostics.direct_capability_retries
              << "\tio_active_requests=" << async_diagnostics.active_read_requests
              << "\tio_active_operations=" << async_diagnostics.active_operations
              << "\tio_trace_dropped=" << async_diagnostics.trace_records_dropped
              << "\tscheduler_flights=" << scheduler_diagnostics.flights_created
              << "\tscheduler_active=" << scheduler_diagnostics.active_requests
              << "\tsource_pageable=" << diagnostics.source_pageable
              << "\tsource_pinned_bytes=" << diagnostics.source_pinned_bytes
              << "\tno_writeback_evictions=" << diagnostics.no_writeback_evictions
              << "\tcold_hot_refs=" << diagnostics.cold_current_hot_refs
              << "\tcold_transfer_refs=" << diagnostics.cold_current_transfer_refs
              << "\tcold_request_refs=" << diagnostics.cold_current_request_refs
              << "\tcold_peak_hot_refs=" << diagnostics.cold_peak_hot_refs
              << "\tcold_peak_transfer_refs=" << diagnostics.cold_peak_transfer_refs
              << "\tcold_peak_request_refs=" << diagnostics.cold_peak_request_refs
              << "\tring_requested_bytes=" << diagnostics.ring_requested_bytes
              << "\tring_actual_bytes=" << diagnostics.ring_actual_bytes
              << "\tring_lane_footprint=" << diagnostics.ring_lane_footprint
              << "\tring_lanes=" << diagnostics.ring_effective_lanes
              << "\tring_pinned_bytes=" << diagnostics.ring_pinned_or_registered_bytes
              << "\tring_acquisition=" << diagnostics.ring_acquisition_method
              << "\tring_fallback_reason=" << diagnostics.ring_fallback_reason
              << "\tring_fallback=" << diagnostics.ring_pageable_fallback
              << "\tring_async_enqueues=" << diagnostics.ring_async_enqueues
              << "\tring_sync_copies=" << diagnostics.ring_synchronous_copies
              << "\tring_stage_bytes=" << diagnostics.ring_stage_bytes
              << "\tring_stage_time_us=" << diagnostics.ring_stage_time_us
              << "\tring_h2d_bytes=" << diagnostics.ring_h2d_bytes
              << "\tring_h2d_time_us=" << diagnostics.ring_h2d_time_us
              << "\tring_waves=" << diagnostics.ring_waves
              << "\tring_wave_syncs=" << diagnostics.ring_wave_synchronizations
              << "\tring_dedicated_backend=" << diagnostics.ring_dedicated_transfer_backend
              << "\tring_event_capable=" << diagnostics.ring_event_capable
              << "\tring_h2d_event_capacity=" << diagnostics.ring_h2d_event_capacity
              << "\tring_compute_event_capacity=" << diagnostics.ring_compute_event_capacity
              << "\tring_event_capacity=" << diagnostics.ring_event_capacity
              << "\tring_live_h2d_events=" << diagnostics.ring_live_h2d_events
              << "\tring_peak_live_h2d_events=" << diagnostics.ring_peak_live_h2d_events
              << "\tring_live_compute_events=" << diagnostics.ring_live_compute_events
              << "\tring_peak_live_compute_events=" << diagnostics.ring_peak_live_compute_events
              << "\tring_live_events=" << diagnostics.ring_live_events
              << "\tring_peak_live_events=" << diagnostics.ring_peak_live_events
              << "\tring_event_records=" << diagnostics.ring_event_records
              << "\tring_compute_waits=" << diagnostics.ring_compute_waits
              << "\tring_event_syncs=" << diagnostics.ring_event_synchronizations
              << "\tring_compute_event_records=" << diagnostics.ring_compute_event_records
              << "\tring_compute_event_syncs=" << diagnostics.ring_compute_event_synchronizations
              << "\tring_h2d_event_allocations=" << diagnostics.ring_h2d_event_allocations
              << "\tring_compute_event_allocations=" << diagnostics.ring_compute_event_allocations
              << "\tring_h2d_event_frees=" << diagnostics.ring_h2d_event_frees
              << "\tring_compute_event_frees=" << diagnostics.ring_compute_event_frees
              << "\tring_compute_work=" << diagnostics.ring_compute_work
              << "\tring_trace_capacity=" << diagnostics.ring_trace_capacity
              << "\tring_trace_records=" << diagnostics.ring_trace_records
              << "\tring_trace_dropped=" << diagnostics.ring_trace_records_dropped
              << "\tring_first_h2d_us=" << diagnostics.ring_first_h2d_enqueue_us
              << "\tring_last_h2d_complete_us=" << diagnostics.ring_last_h2d_event_complete_us
              << "\th2d_compute_overlap_us=" << diagnostics.ring_h2d_compute_overlap_us
              << "\th2d_compute_overlap_bytes=" << diagnostics.ring_h2d_compute_overlap_bytes
              << "\th2d_compute_overlap_work=" << diagnostics.ring_h2d_compute_overlap_work
              << "\th2d_compute_overlap_flights=" << diagnostics.ring_h2d_compute_overlap_flights
              << "\tdisk_h2d_overlap_us=" << diagnostics.disk_h2d_overlap_us
              << "\tdisk_h2d_overlap_bytes=" << diagnostics.disk_h2d_overlap_bytes
              << "\tdisk_h2d_overlap_read_bytes=" << diagnostics.disk_h2d_overlap_read_bytes
              << "\tdisk_h2d_overlap_flights=" << diagnostics.disk_h2d_overlap_flights
              << "\tdisk_h2d_overlap_events=" << diagnostics.disk_h2d_overlap_events
              << "\tdisk_h2d_overlap_read_flights=" << diagnostics.disk_h2d_overlap_read_flights
              << "\tdisk_h2d_overlap_transfer_flights=" << diagnostics.disk_h2d_overlap_transfer_flights
              << "\tdisk_h2d_overlap_pairs=" << diagnostics.disk_h2d_overlap_pairs
              << "\tdisk_h2d_overlap_pair_digest=" << diagnostics.disk_h2d_overlap_pair_digest
              << "\texecution_ids_cpu=" << placement.cpu
              << "\texecution_ids_non_cpu=" << placement.non_cpu
              << "\texecution_ids_backend_device_type=" << placement.backend_device_type
              << "\texecution_ids_backend_name=" << placement.backend_name
              << "\texecution_ids_backend_consistent=" << placement.backend_consistent
              << "\trouting_ids_cpu=" << placement.routing_cpu
              << "\trouting_ids_non_cpu=" << placement.routing_non_cpu
              << "\trouting_backend_device_type=" << placement.routing_backend_device_type
              << "\trouting_backend_name=" << placement.routing_backend_name
              << "\trouting_backend_consistent=" << placement.routing_backend_consistent
              << "\texecution_backend_device_type=" << diagnostics.last_execution_backend_device_type
              << "\tresident_runtime_quiet=" << resident_runtime_quiet
              << '\n';
    const bool cached_mode = args.mode == "hot" || args.mode == "cold" || args.mode == "uma";
    const bool routing_backend_valid = placement.routing_cpu + placement.routing_non_cpu > 0 &&
        placement.routing_backend_device_type >= 0 && placement.routing_backend_consistent;
    const auto expected_execution_device = args.mode == "uma" ?
        GGML_BACKEND_DEVICE_TYPE_IGPU : GGML_BACKEND_DEVICE_TYPE_GPU;
    const bool valid_placement = cached_mode ?
        placement.cpu > 0 && placement.non_cpu == 0 && placement.backend_consistent &&
            placement.backend_device_type == GGML_BACKEND_DEVICE_TYPE_CPU &&
            placement.routing_cpu == 0 && placement.routing_non_cpu > 0 && routing_backend_valid &&
            diagnostics.last_execution_backend_device_type == expected_execution_device :
        placement.cpu == 0 && placement.non_cpu == 0 && routing_backend_valid && resident_runtime_quiet;
    if (!valid_placement) {
        std::cerr << "invalid placement cached=" << cached_mode << " cpu=" << placement.cpu
                  << " non_cpu=" << placement.non_cpu << " backend_consistent=" << placement.backend_consistent
                  << " backend_type=" << placement.backend_device_type << " routing_cpu=" << placement.routing_cpu
                  << " routing_non_cpu=" << placement.routing_non_cpu << " routing_valid=" << routing_backend_valid
                  << " execution_backend=" << diagnostics.last_execution_backend_device_type << '\n';
    }
    bool valid_overlap = true;
    if (args.require_overlap) {
        valid_overlap = args.mode == "cold" && diagnostics.ring_event_capable &&
            diagnostics.ring_dedicated_transfer_backend && diagnostics.ring_wave_synchronizations == 0 &&
            diagnostics.ring_event_records == diagnostics.ring_waves &&
            diagnostics.ring_compute_waits == diagnostics.ring_waves &&
            diagnostics.ring_event_synchronizations == diagnostics.ring_waves &&
            diagnostics.ring_h2d_event_capacity == diagnostics.ring_effective_lanes &&
            diagnostics.ring_compute_event_capacity == diagnostics.ring_effective_lanes &&
            diagnostics.ring_event_capacity == diagnostics.ring_effective_lanes*2 &&
            diagnostics.ring_compute_event_records == 0 &&
            diagnostics.ring_live_h2d_events == 0 && diagnostics.ring_live_compute_events == 0 &&
            diagnostics.ring_live_events == 0 && diagnostics.ring_trace_records_dropped == 0 &&
            async_diagnostics.trace_records_dropped == 0;
    }
    llama_free(context);
    llama_model_free(model);
    if (args.require_overlap && valid_overlap) {
        auto * device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        ggml_backend_ptr backend(device == nullptr ? nullptr : ggml_backend_dev_init(device, nullptr));
        valid_overlap = device != nullptr && backend && controlled_native_overlap(device, backend.get());
    }
    if (!valid_overlap) return 34;
    if (!valid_placement) return 35;
    return 0;
}

struct fixture {
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buffer;
    ggml_tensor * up = nullptr;
    ggml_tensor * gate = nullptr;
    ggml_tensor * down = nullptr;
    int32_t n_expert;

    fixture(int32_t n_expert, ggml_backend_buffer_type_t buft, bool fill,
            int64_t rows = 8, int64_t columns = 16) : n_expert(n_expert) {
        ggml_init_params params = { ggml_tensor_overhead()*8, nullptr, true };
        ctx.reset(ggml_init(params));
        if (!ctx) throw std::runtime_error("context allocation failed");
        up = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, rows, columns, n_expert);
        gate = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, rows, columns, n_expert);
        down = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, columns, rows, n_expert);
        buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft));
        if (!buffer) throw std::runtime_error("tensor allocation failed");
        if (fill) {
            uint8_t pattern = 0x30;
            for (auto * tensor : { up, gate, down }) {
                for (int32_t expert = 0; expert < n_expert; ++expert) {
                    std::vector<uint8_t> bytes(tensor->nb[2], pattern + expert);
                    ggml_backend_tensor_set(tensor, bytes.data(), size_t(expert)*tensor->nb[2], bytes.size());
                }
                pattern += 0x20;
            }
        }
    }

    llm_expert_bundle_descriptor bundle() const {
        return {
            0, n_expert,
            llm_expert_projection_descriptor::from(up, nullptr, nullptr),
            llm_expert_projection_descriptor::from(gate, nullptr, nullptr),
            {},
            llm_expert_projection_descriptor::from(down, nullptr, nullptr),
        };
    }
};

bool slot_matches(const fixture & source, const fixture & hot, int32_t expert, uint32_t slot) {
    const auto source_bundle = source.bundle();
    const auto hot_bundle = hot.bundle();
    for (const auto & pair : {
            std::pair<const ggml_tensor *, const ggml_tensor *>(source_bundle.up.weight, hot_bundle.up.weight),
            std::pair<const ggml_tensor *, const ggml_tensor *>(source_bundle.gate.weight, hot_bundle.gate.weight),
            std::pair<const ggml_tensor *, const ggml_tensor *>(source_bundle.down.weight, hot_bundle.down.weight) }) {
        const size_t span = pair.first->nb[2];
        std::vector<uint8_t> expected(span), actual(span);
        ggml_backend_tensor_get(pair.first, expected.data(), size_t(expert)*span, span);
        ggml_backend_tensor_get(pair.second, actual.data(), size_t(slot)*span, span);
        if (expected != actual) return false;
    }
    return true;
}

bool controlled_native_overlap(ggml_backend_dev_t device, ggml_backend_t backend) {
    const auto fail = [](int step) { std::cerr << "controlled overlap failed at step " << step << '\n'; return false; };
    constexpr uint64_t disk_bytes = 192ULL << 20;
    fixture source(2, ggml_backend_cpu_buffer_type(), true, 2048, 4096);
    fixture hot(2, ggml_backend_dev_buffer_type(device), false, 2048, 4096);
    llm_cold_expert_cache cold({ 256ULL << 20, 2, 1, 2, 0 });
    const auto cold_initialized = cold.initialize(source.bundle());
    if (!cold_initialized.is_ready()) {
        std::cerr << "cold init error=" << int(cold_initialized.error) << '\n';
        return fail(1);
    }
    llm_cold_reference cold_ref;
    if (!cold.find_or_admit({ 0, 0 }, source.bundle(), cold_ref).is_ready()) return fail(2);
    llm_cold_reference cold_second;
    if (!cold.find_or_admit({ 0, 1 }, source.bundle(), cold_second).is_ready()) return fail(2);
    const llm_expert_flight_id controlled_flight = { 1, 0, 1, { 0, 0 } };
    const llm_expert_flight_id read_flight = { 1, 0, 1, { 0, 1 } };
    llm_expert_transfer_ring ring({ 128ULL << 20, 1, device, false, false, false, 0, 64, 0, true });
    if (!ring.initialize(source.bundle()).is_ready()) return fail(3);
    llm_transfer_lane_reference lane;
    const auto reserved = ring.reserve(cold, cold_ref, 0, 1, lane, controlled_flight);
    if (!reserved.is_ready()) { std::cerr << "reserve error=" << int(reserved.error) << '\n'; return fail(4); }
    const auto staged = ring.stage(lane, cold.bundle());
    if (!staged.is_ready()) { std::cerr << "stage error=" << int(staged.error) << '\n'; return fail(4); }

    char path[] = "./phase7-overlap-XXXXXX";
    const int fd = mkstemp(path);
    if (fd < 0) return fail(5);
    unlink(path);
    std::vector<uint8_t> file_block(4U << 20, 0x5a);
    uint64_t written = 0;
    while (written < disk_bytes) {
        const ssize_t count = write(fd, file_block.data(), file_block.size());
        if (count <= 0) { close(fd); return fail(6); }
        written += uint64_t(count);
    }
    fsync(fd);
#if defined(POSIX_FADV_DONTNEED)
    (void) posix_fadvise(fd, 0, off_t(disk_bytes), POSIX_FADV_DONTNEED);
#endif
    std::vector<uint8_t> disk_destination(size_t(disk_bytes), uint8_t(0));
    std::array<llm_expert_storage_read_operation, 2> read{};
    for (size_t index = 0; index < read.size(); ++index) {
        read[index].native_handle = fd;
        read[index].source_size = disk_bytes;
        read[index].file_offset = index*(disk_bytes/2);
        read[index].byte_count = disk_bytes/2;
        read[index].segment_count = 1;
        read[index].segments[0].data = disk_destination.data() + index*(disk_bytes/2);
        read[index].segments[0].file_offset = index*(disk_bytes/2);
        read[index].segments[0].byte_count = disk_bytes/2;
        read[index].segments[0].projection = llm_expert_storage_projection::up;
        read[index].segments[0].sidecar = index == 0 ?
            llm_expert_storage_sidecar::weight : llm_expert_storage_sidecar::bias;
    }
    llm_expert_async_config io_config;
    io_config.requested_queue_depth = 32;
    io_config.effective_hot_capacity = 4;
    io_config.request_capacity = 8;
    io_config.trace_capacity = 64;
    io_config.cold_cache_bytes = 256ULL << 20;
    io_config.maximum_aligned_read_bytes = disk_bytes;
    io_config.source_file_capacity = 1;
    io_config.delay_cq_drain_ms_for_testing = 10;
    llm_expert_async_transport transport(io_config);
    const llm_expert_async_operation_identity identity = {
        read_flight.transport_epoch,
        { read_flight.request_slot, read_flight.request_generation },
        0, read_flight.key, llm_expert_readiness::host_ready,
        llm_expert_priority::demand_current_layer,
    };
    ggml_init_params compute_params = { ggml_tensor_overhead()*4 + ggml_graph_overhead(), nullptr, true };
    ggml_context_ptr compute_ctx(ggml_init(compute_params));
    if (!compute_ctx) { close(fd); return fail(8); }
    ggml_tensor * a = ggml_new_tensor_2d(compute_ctx.get(), GGML_TYPE_F32, 1536, 1536);
    ggml_tensor * b = ggml_new_tensor_2d(compute_ctx.get(), GGML_TYPE_F32, 1536, 1536);
    ggml_tensor * c = ggml_mul_mat(compute_ctx.get(), a, b);
    ggml_backend_buffer_ptr compute_buffer(
        ggml_backend_alloc_ctx_tensors_from_buft(compute_ctx.get(), ggml_backend_dev_buffer_type(device)));
    if (!compute_buffer) { close(fd); return fail(9); }
    std::vector<float> zeros(size_t(1536)*1536, 0.0f);
    ggml_backend_tensor_set(a, zeros.data(), 0, zeros.size()*sizeof(float));
    ggml_backend_tensor_set(b, zeros.data(), 0, zeros.size()*sizeof(float));
    ggml_cgraph * graph = ggml_new_graph_custom(compute_ctx.get(), 8, false);
    ggml_build_forward_expand(graph, c);
    constexpr uint64_t compute_work = 2ULL*1536*1536*1536;
    if (transport.submit_read_plan(identity, read.data(), read.size()) != llm_expert_async_result::ready) {
        close(fd); return fail(7);
    }
    if (!transport.wait_until_read_submitted_for_testing(identity.request)) {
        close(fd); return fail(7);
    }
    if (!ring.transfer_wave(backend, { { lane, hot.bundle(), 0 } }).is_ready() ||
        !ring.begin_compute_work(backend, compute_work).is_ready() ||
        ggml_backend_graph_compute_async(backend, graph) != GGML_STATUS_SUCCESS ||
        !ring.wait_for_hot(backend, 0, 1).is_ready()) {
        close(fd); return fail(10);
    }
    std::atomic<bool> reuse_finished = false;
    llm_transfer_lane_reference reused_lane;
    std::thread reuse([&] {
        const auto result = ring.reserve(cold, cold_second, 0, 2, reused_lane);
        reuse_finished.store(result.is_ready(), std::memory_order_release);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    const bool reuse_was_blocked = !reuse_finished.load(std::memory_order_acquire);
    reuse.join();
    if (!reuse_finished.load(std::memory_order_acquire) ||
        reused_lane.generation == lane.generation || !ring.cleanup_failed_lanes().is_ready()) {
        close(fd); return fail(11);
    }
    llm_expert_async_read_completion completion;
    const auto waited = transport.wait_read(identity.request, completion);
    const auto released = transport.release_read(identity.request);
    const auto reads = transport.completed_read_intervals();
    const auto transfers = ring.completed_intervals();
    const auto diagnostics = ring.diagnostics();
    const auto io_diagnostics = transport.diagnostics();
    std::vector<std::pair<uint64_t, uint64_t>> read_ranges;
    std::vector<std::pair<uint64_t, uint64_t>> transfer_ranges;
    std::vector<bool> transfer_participates(transfers.size(), false);
    uint64_t overlap_read_bytes = 0;
    uint64_t overlap_transfer_bytes = 0;
    uint64_t overlap_pairs = 0;
    uint64_t overlap_pair_digest = 0;
    for (const auto & interval : reads) {
        bool participates = false;
        for (size_t transfer_index = 0; transfer_index < transfers.size(); ++transfer_index) {
            const auto & transfer = transfers[transfer_index];
            if (!interval.flight.valid() || !transfer.flight.valid() ||
                same_flight(interval.flight, transfer.flight)) continue;
            const uint64_t begin = std::max(interval.submit_us, transfer.h2d_enqueue_us);
            const uint64_t end = std::min(interval.complete_us, transfer.h2d_complete_us);
            if (end <= begin) continue;
            participates = true;
            transfer_participates[transfer_index] = true;
            overlap_pairs++;
            const uint64_t pair_hash = overlap_pair_hash(interval, transfer);
            overlap_pair_digest ^=
                pair_hash + 0x9e3779b97f4a7c15ULL + (pair_hash << 6) + (pair_hash >> 2);
            std::cout << "PHASE7_OVERLAP_PAIR"
                      << "\tread_epoch=" << interval.flight.transport_epoch
                      << "\tread_slot=" << interval.flight.request_slot
                      << "\tread_generation=" << interval.flight.request_generation
                      << "\tread_layer=" << interval.flight.key.layer
                      << "\tread_expert=" << interval.flight.key.expert
                      << "\tread_operation=" << interval.operation_index
                      << "\ttransfer_epoch=" << transfer.flight.transport_epoch
                      << "\ttransfer_slot=" << transfer.flight.request_slot
                      << "\ttransfer_generation=" << transfer.flight.request_generation
                      << "\ttransfer_layer=" << transfer.flight.key.layer
                      << "\ttransfer_expert=" << transfer.flight.key.expert
                      << "\tlane_generation=" << transfer.lane.generation
                      << "\thot_generation=" << transfer.hot_generation
                      << "\toverlap_us=" << end - begin
                      << "\tpair_hash=" << pair_hash
                      << '\n';
        }
        if (participates) {
            read_ranges.emplace_back(interval.submit_us, interval.complete_us);
            overlap_read_bytes += interval.bytes;
        }
        std::cout << "PHASE7_READ_INTERVAL"
                  << "\tepoch=" << interval.flight.transport_epoch
                  << "\tslot=" << interval.flight.request_slot
                  << "\tgeneration=" << interval.flight.request_generation
                  << "\tlayer=" << interval.flight.key.layer
                  << "\texpert=" << interval.flight.key.expert
                  << "\toperation=" << interval.operation_index
                  << "\tsubmit_us=" << interval.submit_us
                  << "\tcomplete_us=" << interval.complete_us
                  << "\tbytes=" << interval.bytes
                  << '\n';
    }
    for (size_t transfer_index = 0; transfer_index < transfers.size(); ++transfer_index) {
        const auto & transfer = transfers[transfer_index];
        if (transfer_participates[transfer_index]) {
            transfer_ranges.emplace_back(transfer.h2d_enqueue_us, transfer.h2d_complete_us);
            overlap_transfer_bytes += transfer.bytes;
        }
        std::cout << "PHASE7_H2D_INTERVAL"
                  << "\tepoch=" << transfer.flight.transport_epoch
                  << "\tslot=" << transfer.flight.request_slot
                  << "\tgeneration=" << transfer.flight.request_generation
                  << "\tlayer=" << transfer.flight.key.layer
                  << "\texpert=" << transfer.flight.key.expert
                  << "\tlane=" << transfer.lane.lane
                  << "\tlane_generation=" << transfer.lane.generation
                  << "\thot_slot=" << transfer.hot_slot
                  << "\thot_generation=" << transfer.hot_generation
                  << "\tenqueue_us=" << transfer.h2d_enqueue_us
                  << "\tcomplete_us=" << transfer.h2d_complete_us
                  << "\tbytes=" << transfer.bytes
                  << '\n';
    }
    const auto merge_ranges = [](std::vector<std::pair<uint64_t, uint64_t>> ranges) {
        std::sort(ranges.begin(), ranges.end());
        size_t output = 0;
        for (const auto & range : ranges) {
            if (range.second <= range.first) continue;
            if (output == 0 || range.first > ranges[output - 1].second) ranges[output++] = range;
            else ranges[output - 1].second = std::max(ranges[output - 1].second, range.second);
        }
        ranges.resize(output);
        return ranges;
    };
    read_ranges = merge_ranges(std::move(read_ranges));
    transfer_ranges = merge_ranges(std::move(transfer_ranges));
    uint64_t disk_h2d_overlap_us = 0;
    for (size_t left = 0, right = 0; left < read_ranges.size() && right < transfer_ranges.size();) {
        const uint64_t begin = std::max(read_ranges[left].first, transfer_ranges[right].first);
        const uint64_t end = std::min(read_ranges[left].second, transfer_ranges[right].second);
        if (end > begin) disk_h2d_overlap_us += end - begin;
        if (read_ranges[left].second < transfer_ranges[right].second) left++;
        else right++;
    }
    {
        llm_expert_transfer_ring unloading({ 128ULL << 20, 1, device, false, false, false, 0, 16 });
        llm_transfer_lane_reference unload_lane;
        if (!unloading.initialize(source.bundle()).is_ready() ||
            !unloading.reserve(cold, cold_ref, 0, 3, unload_lane).is_ready() ||
            !unloading.stage(unload_lane, cold.bundle()).is_ready() ||
            !unloading.transfer_wave(backend, { { unload_lane, hot.bundle(), 0 } }).is_ready()) {
            close(fd); return fail(12);
        }
    }
    const bool unload_drained = cold.diagnostics().current_transfer_refs == 0;
    const bool valid = waited == llm_expert_async_result::ready &&
        released == llm_expert_async_result::ready && transport.shutdown() &&
        completion.bytes_completed == disk_bytes && diagnostics.h2d_compute_overlap_us > 0 &&
        diagnostics.h2d_compute_overlap_bytes > 0 &&
        diagnostics.h2d_compute_overlap_work >= compute_work && diagnostics.compute_work >= compute_work &&
        diagnostics.compute_event_records == 1 && diagnostics.compute_event_synchronizations == 1 &&
        diagnostics.h2d_event_capacity == diagnostics.effective_lanes &&
        diagnostics.compute_event_capacity == diagnostics.effective_lanes &&
        diagnostics.event_capacity == diagnostics.effective_lanes*2 &&
        diagnostics.peak_live_h2d_events == 1 && diagnostics.peak_live_compute_events == 1 &&
        diagnostics.wave_synchronizations == 0 && diagnostics.live_h2d_events == 0 &&
        diagnostics.live_compute_events == 0 && diagnostics.live_events == 0 &&
        diagnostics.trace_records_dropped == 0 && io_diagnostics.trace_records_dropped == 0 &&
        disk_h2d_overlap_us > 0 && overlap_pairs > 0 && overlap_pair_digest != 0 &&
        overlap_read_bytes > 0 && overlap_transfer_bytes > 0 &&
        reuse_was_blocked && unload_drained &&
        reads.size() == 2 && reads[0].operation_index != reads[1].operation_index &&
        transfers.size() == 1 && reads[0].flight.valid() && reads[1].flight.valid() &&
        transfers[0].flight.valid() &&
        !same_flight(reads[0].flight, transfers[0].flight) &&
        !same_flight(reads[1].flight, transfers[0].flight) &&
        same_flight(reads[0].flight, reads[1].flight) &&
        slot_matches(source, hot, 0, 0);
    std::cout << "PHASE7_CONTROLLED_OVERLAP"
              << "\tdisk_h2d_overlap_us=" << disk_h2d_overlap_us
              << "\tdisk_h2d_overlap_read_flights=1"
              << "\tdisk_h2d_overlap_transfer_flights=1"
              << "\tdisk_h2d_overlap_flights=2"
              << "\tdisk_h2d_overlap_pairs=" << overlap_pairs
              << "\tdisk_h2d_overlap_pair_digest=" << overlap_pair_digest
              << "\tdisk_h2d_overlap_read_bytes=" << overlap_read_bytes
              << "\tdisk_h2d_overlap_bytes=" << overlap_transfer_bytes
              << "\th2d_compute_overlap_us=" << diagnostics.h2d_compute_overlap_us
              << "\th2d_compute_overlap_bytes=" << diagnostics.h2d_compute_overlap_bytes
              << "\th2d_compute_overlap_work=" << diagnostics.h2d_compute_overlap_work
              << "\tcompute_work=" << diagnostics.compute_work
              << "\tread_bytes=" << completion.bytes_completed
              << "\tevent_records=" << diagnostics.event_records
              << "\tcompute_event_records=" << diagnostics.compute_event_records
              << "\treuse_was_blocked=" << reuse_was_blocked
              << "\tunload_drained=" << unload_drained
              << "\tread_intervals=" << reads.size()
              << "\ttransfer_intervals=" << transfers.size()
              << "\tread_trace_dropped=" << io_diagnostics.trace_records_dropped
              << "\th2d_trace_dropped=" << diagnostics.trace_records_dropped
              << "\tread_submit_us=" << (reads.empty() ? 0 : reads[0].submit_us)
              << "\tread_complete_us=" << (reads.empty() ? 0 : reads[0].complete_us)
              << "\th2d_enqueue_us=" << (transfers.empty() ? 0 : transfers[0].h2d_enqueue_us)
              << "\th2d_complete_us=" << (transfers.empty() ? 0 : transfers[0].h2d_complete_us)
              << '\n';
    close(fd);
    return valid;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc > 1 && std::string(argv[1]) == "--model") return run_live(argc, argv);
    bool force_pageable = false;
    if (argc == 2 && std::string(argv[1]) == "--force-pageable") force_pageable = true;
    else if (argc != 1) return 2;

    ggml_backend_load_all();
    auto * device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!device) return 3;
    ggml_backend_ptr backend(ggml_backend_dev_init(device, nullptr));
    if (!backend) return 4;

    fixture source(4, ggml_backend_cpu_buffer_type(), true);
    fixture hot(2, ggml_backend_dev_buffer_type(device), false);
    llm_cold_expert_cache cold({ 1U << 20, 2, 1, 4, 0 });
    if (!cold.initialize(source.bundle()).is_ready()) return 5;
    llm_cold_reference cold_zero, cold_one;
    if (!cold.find_or_admit({ 0, 0 }, source.bundle(), cold_zero).is_ready() ||
        !cold.find_or_admit({ 0, 1 }, source.bundle(), cold_one).is_ready()) return 6;

    llm_expert_transfer_ring ring({ 1U << 20, 2, device, false, force_pageable, 0 });
    if (!ring.initialize(source.bundle()).is_ready()) return 7;
    llm_transfer_lane_reference lane_zero, lane_one;
    if (!ring.reserve(cold, cold_zero, 0, 1, lane_zero).is_ready() ||
        !ring.reserve(cold, cold_one, 1, 1, lane_one).is_ready() ||
        !ring.stage(lane_zero, cold.bundle()).is_ready() ||
        !ring.stage(lane_one, cold.bundle()).is_ready()) return 8;
    if (!ring.transfer_wave(backend.get(), {
            { lane_zero, hot.bundle(), 0 }, { lane_one, hot.bundle(), 1 },
        }).is_ready()) return 9;
    if (!ring.wait_for_hot(backend.get(), 0, 1).is_ready() ||
        !ring.wait_for_hot(backend.get(), 1, 1).is_ready()) return 9;
    if (!ring.retire_hot(0, 1).is_ready() || !ring.retire_hot(1, 1).is_ready()) return 9;
    if (!slot_matches(source, hot, 0, 0) || !slot_matches(source, hot, 1, 1)) return 10;

    const auto value = ring.diagnostics();
    if (value.waves != 1 || value.h2d_bytes != value.lane_payload_bytes*2 ||
        cold.diagnostics().current_transfer_refs != 0 || !ring.validate_invariants().is_ready()) return 11;
    if (force_pageable) {
        if (!value.pageable_fallback || value.pinned_or_registered_bytes != 0 ||
            value.async_enqueues != 0 || value.wave_synchronizations != 0 || value.synchronous_copies != 6 ||
            value.h2d_event_capacity != 0 || value.compute_event_capacity != 0 || value.event_capacity != 0 ||
            value.live_h2d_events != 0 || value.live_compute_events != 0 || value.live_events != 0 ||
            value.event_records != 0 || value.compute_event_records != 0) return 12;
    } else {
        if (value.pageable_fallback || value.pinned_or_registered_bytes == 0 ||
            value.async_enqueues != 6 || value.wave_synchronizations != 0 || value.event_records != 2 ||
            value.compute_waits != 2 || value.h2d_event_capacity != value.effective_lanes ||
            value.compute_event_capacity != value.effective_lanes ||
            value.event_capacity != value.effective_lanes*2 ||
            value.live_h2d_events != 0 || value.live_compute_events != 0 || value.live_events != 0 ||
            value.peak_in_flight_lanes != 2 || value.synchronous_copies != 0) return 13;
        if (!controlled_native_overlap(device, backend.get())) return 14;
    }
    std::cout << "PHASE5_TRANSFER_RING"
              << "\tmode=" << (force_pageable ? "pageable" : "pinned")
              << "\trequested_bytes=" << value.requested_bytes
              << "\tlanes=" << value.effective_lanes
              << "\tlane_footprint=" << value.lane_footprint
              << "\tactual_bytes=" << value.actual_bytes
              << "\tpinned_bytes=" << value.pinned_or_registered_bytes
              << "\tacquisition=" << value.acquisition_method
              << "\tfallback_reason=" << value.fallback_reason
              << "\tpageable_fallback=" << value.pageable_fallback
              << "\tasync_enqueues=" << value.async_enqueues
              << "\tsynchronous_copies=" << value.synchronous_copies
              << "\twaves=" << value.waves
              << "\twave_synchronizations=" << value.wave_synchronizations
              << "\tpeak_in_flight_lanes=" << value.peak_in_flight_lanes
              << "\th2d_bytes=" << value.h2d_bytes
              << '\n';
    return 0;
}
