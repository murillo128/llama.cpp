#include "llama-expert-transfer-ring.h"
#include "llama-expert-storage.h"
#include "llama-context.h"
#include "llama-model.h"
#include "llama.h"

#include "ggml-cpp.h"

#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

uint64_t hash_bytes(uint64_t hash, const void * data, size_t size) {
    const auto * bytes = static_cast<const uint8_t *>(data);
    for (size_t index = 0; index < size; ++index) {
        hash ^= bytes[index];
        hash *= 1099511628211ULL;
    }
    return hash;
}

struct route_hash {
    uint64_t value = 1469598103934665603ULL;
    uint64_t records = 0;
};

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
        else if (option == "--load-mode") {
            try {
                result.load_mode = llama_load_mode_from_str(value);
            } catch (const std::invalid_argument &) {
                return false;
            }
        }
        else return false;
    }
    return !result.model.empty() && (result.mode == "disabled" || result.mode == "hot" || result.mode == "cold") &&
        (result.mode == "disabled" || result.capacity > 0) &&
        (result.mode != "cold" || (result.cold_bytes > 0 && result.ring_bytes > 0));
}

struct storage_cancel_state {
    const llm_expert_storage * storage = nullptr;
};

bool cancel_after_first_storage_read(void * user_data) {
    const auto * state = static_cast<const storage_cancel_state *>(user_data);
    return state && state->storage && state->storage->diagnostics().read_bytes > 0;
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
    if (args.mode != "disabled") {
        model_params.tensor_buft_overrides = overrides;
        model_params.expert_hot_cache_capacity = args.capacity;
        model_params.expert_weights_mode = args.mode == "hot" ?
            LLAMA_EXPERT_WEIGHTS_MODE_HOT_CACHE : LLAMA_EXPERT_WEIGHTS_MODE_COLD_CACHE;
        model_params.expert_cold_cache_bytes = args.cold_bytes;
        model_params.expert_transfer_ring_bytes = args.ring_bytes;
    }
    llama_model * model = llama_model_load_from_file(args.model.c_str(), model_params);
    if (!model) return 21;
    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = 64;
    context_params.n_batch = 64;
    context_params.n_ubatch = 1;
    llama_context * context = llama_init_from_model(model, context_params);
    if (!context) return 22;

    if (args.cancel_on_storage) {
        if (args.mode != "cold" || model->expert_storage() == nullptr) return 27;
        storage_cancel_state cancel_state = { model->expert_storage() };
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
        const bool valid = cancelled == 2 && cancelled_storage.read_requests == 1 &&
            cancelled_storage.read_bytes > 0 && cancelled_storage.cancelled_reads == 1 &&
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
    const auto storage_diagnostics = model->expert_storage() ?
        model->expert_storage()->diagnostics() : llm_expert_storage_diagnostics {};
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
    std::cout << "PHASE5_LIVE"
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
              << "\thot_hits=" << diagnostics.hits
              << "\thot_misses=" << diagnostics.misses
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
              << "\tstorage_read_requests=" << storage_diagnostics.read_requests
              << "\tstorage_read_chunks=" << storage_diagnostics.read_chunks
              << "\tstorage_read_bytes=" << storage_diagnostics.read_bytes
              << "\tstorage_cancelled_reads=" << storage_diagnostics.cancelled_reads
              << "\tstorage_short_reads=" << storage_diagnostics.short_reads
              << "\tstorage_io_errors=" << storage_diagnostics.io_errors
              << "\tstorage_poisoned=" << storage_diagnostics.poisoned
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
              << '\n';
    llama_free(context);
    llama_model_free(model);
    return 0;
}

struct fixture {
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buffer;
    ggml_tensor * up = nullptr;
    ggml_tensor * gate = nullptr;
    ggml_tensor * down = nullptr;
    int32_t n_expert;

    fixture(int32_t n_expert, ggml_backend_buffer_type_t buft, bool fill) : n_expert(n_expert) {
        ggml_init_params params = { ggml_tensor_overhead()*8, nullptr, true };
        ctx.reset(ggml_init(params));
        if (!ctx) throw std::runtime_error("context allocation failed");
        up = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 8, 16, n_expert);
        gate = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 8, 16, n_expert);
        down = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 16, 8, n_expert);
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
    if (!slot_matches(source, hot, 0, 0) || !slot_matches(source, hot, 1, 1)) return 10;

    const auto value = ring.diagnostics();
    if (value.waves != 1 || value.h2d_bytes != value.lane_payload_bytes*2 ||
        cold.diagnostics().current_transfer_refs != 0 || !ring.validate_invariants().is_ready()) return 11;
    if (force_pageable) {
        if (!value.pageable_fallback || value.pinned_or_registered_bytes != 0 ||
            value.async_enqueues != 0 || value.wave_synchronizations != 0 || value.synchronous_copies != 6) return 12;
    } else {
        if (value.pageable_fallback || value.pinned_or_registered_bytes == 0 ||
            value.async_enqueues != 6 || value.wave_synchronizations != 1 ||
            value.peak_in_flight_lanes != 2 || value.synchronous_copies != 0) return 13;
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
