#include "llama.h"
#include "llama-cpp.h"
#include "llama-context.h"
#include "llama-expert-async-io.h"
#include "llama-expert-scheduler.h"
#include "llama-expert-storage.h"
#include "llama-expert-weight-provider.h"
#include "llama-model.h"

#include "ggml-backend.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <unistd.h>
#include <vector>

using json = nlohmann::ordered_json;
using steady_clock = std::chrono::steady_clock;

namespace {

struct arguments {
    std::string model;
    std::string prompt_corpus;
    std::string output;
    std::string point;
    std::string issue_mode = "BATCHED";
    uint64_t cold_cache_bytes = 96ULL*1024*1024*1024;
    uint32_t warmup_limit = 128;
    uint32_t decode_forwards = 64;
    uint32_t threads = 32;
    uint32_t n_ctx = 256;
    int32_t prompt_token = -1;
};

bool parse_u32(const char * text, uint32_t & value) {
    char * end = nullptr;
    const unsigned long parsed = std::strtoul(text, &end, 10);
    if (end == text || *end != '\0' || parsed > UINT32_MAX) return false;
    value = uint32_t(parsed);
    return true;
}

bool parse_u64(const char * text, uint64_t & value) {
    char * end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (end == text || *end != '\0') return false;
    value = uint64_t(parsed);
    return true;
}

bool parse_arguments(int argc, char ** argv, arguments & args) {
    for (int index = 1; index < argc; ++index) {
        const std::string option = argv[index];
        if (index + 1 >= argc) return false;
        const char * value = argv[++index];
        if (option == "--model") args.model = value;
        else if (option == "--prompt-corpus") args.prompt_corpus = value;
        else if (option == "--output") args.output = value;
        else if (option == "--point") args.point = value;
        else if (option == "--issue-mode") args.issue_mode = value;
        else if (option == "--cold-cache-bytes") {
            if (!parse_u64(value, args.cold_cache_bytes)) return false;
        } else if (option == "--warmup-limit") {
            if (!parse_u32(value, args.warmup_limit)) return false;
        } else if (option == "--decode-forwards") {
            if (!parse_u32(value, args.decode_forwards)) return false;
        } else if (option == "--threads") {
            if (!parse_u32(value, args.threads)) return false;
        } else if (option == "--n-ctx") {
            if (!parse_u32(value, args.n_ctx)) return false;
        } else if (option == "--prompt-token") {
            uint32_t parsed = 0;
            if (!parse_u32(value, parsed) || parsed > INT32_MAX) return false;
            args.prompt_token = int32_t(parsed);
        } else {
            return false;
        }
    }
    return !args.model.empty() && (!args.prompt_corpus.empty() || args.prompt_token >= 0) && !args.output.empty() &&
        (args.point == "EXACT" || args.point == "KNEE" || args.point == "AGGRESSIVE") &&
        (args.issue_mode == "SERIAL" || args.issue_mode == "BATCHED") &&
        args.warmup_limit != 0 && args.decode_forwards != 0 &&
        args.threads != 0 && args.n_ctx >= args.warmup_limit + args.decode_forwards;
}

double seconds(steady_clock::duration duration) {
    return std::chrono::duration<double>(duration).count();
}

double seconds(const timeval & value) {
    return double(value.tv_sec) + double(value.tv_usec)/1000000.0;
}

std::string hex_u64(uint64_t value) {
    std::ostringstream out;
    out << std::hex << std::setfill('0') << std::setw(16) << value;
    return out.str();
}

uint64_t token_hash(const std::vector<llama_token> & tokens) {
    uint64_t result = UINT64_C(1469598103934665603);
    for (llama_token token : tokens) {
        const uint32_t value = uint32_t(token);
        for (size_t byte = 0; byte < sizeof(value); ++byte) {
            result ^= (value >> (byte*8)) & 0xffU;
            result *= UINT64_C(1099511628211);
        }
    }
    return result;
}

int finite_argmax(const float * logits, int n_vocab) {
    int result = -1;
    float best = -std::numeric_limits<float>::infinity();
    for (int token = 0; token < n_vocab; ++token) {
        if (std::isfinite(logits[token]) && (result < 0 || logits[token] > best)) {
            result = token;
            best = logits[token];
        }
    }
    return result;
}

uint64_t vm_swap_kib() {
    std::ifstream status("/proc/self/status");
    std::string key;
    while (status >> key) {
        if (key == "VmSwap:") {
            uint64_t value = 0;
            status >> value;
            return value;
        }
        std::string remainder;
        std::getline(status, remainder);
    }
    return 0;
}

double percentile(std::vector<double> values, double quantile) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const size_t index = size_t(std::ceil(quantile*values.size())) - 1;
    return values[std::min(index, values.size() - 1)];
}

json cold_json(const llm_expert_cold_scalar_snapshot & value) {
    return {
        {"requested_bytes", value.requested_bytes},
        {"actual_bytes", value.actual_bytes},
        {"capacity", value.capacity},
        {"occupancy", value.occupancy},
        {"requests", value.requests},
        {"hits", value.hits},
        {"misses", value.misses},
        {"admissions", value.admissions},
        {"evictions", value.evictions},
        {"residency_digest", hex_u64(value.residency_digest)},
    };
}

json system_memory_json(const llm_hot_cache_diagnostics & value) {
    return {
        {"requested_pool_bytes", value.system_memory_requested_pool_bytes},
        {"selected_pool_bytes", value.system_memory_selected_pool_bytes},
        {"safe_pool_bytes", value.system_memory_safe_pool_bytes},
        {"admission_safe_pool_bytes", value.system_memory_admission_safe_pool_bytes},
        {"effective_limit_bytes", value.system_memory_effective_limit_bytes},
        {"limit_headroom_bytes", value.system_memory_limit_headroom_bytes},
        {"available_headroom_bytes", value.system_memory_available_headroom_bytes},
        {"measured_non_pool_committed_bytes", value.system_memory_measured_non_pool_committed_bytes},
        {"runtime_obligation_bytes", value.system_memory_runtime_obligation_bytes},
        {"system_reserve_bytes", value.system_memory_system_reserve_bytes},
        {"runtime_reserve_bytes", value.system_memory_runtime_reserve_bytes},
        {"hysteresis_bytes", value.system_memory_hysteresis_bytes},
        {"model_file_virtual_bytes", value.system_memory_model_file_virtual_bytes},
        {"model_file_cache_resident_bytes", value.system_memory_model_file_cache_resident_bytes},
        {"model_file_resident_bytes", value.system_memory_model_file_resident_bytes},
        {"model_allocated_virtual_bytes", value.system_memory_model_allocated_virtual_bytes},
        {"model_allocated_resident_bytes", value.system_memory_model_allocated_resident_bytes},
        {"other_process_resident_bytes", value.system_memory_other_process_resident_bytes},
        {"pressure_samples", value.system_memory_pressure_samples},
        {"pressure_rejections", value.system_memory_pressure_rejections},
        {"autofit", value.system_memory_autofit},
        {"budget_frozen", value.system_memory_budget_frozen},
        {"pressure_circuit_open", value.system_memory_pressure_circuit_open},
        {"pressure_rejection_reason", value.system_memory_pressure_rejection_reason},
        {"residency_unavailable_reason", value.system_memory_residency_unavailable_reason},
    };
}

json storage_json(const llm_expert_storage_diagnostics & value) {
    return {
        {"read_requests", value.read_requests},
        {"read_chunks", value.read_chunks},
        {"read_bytes", value.read_bytes},
        {"cancelled_reads", value.cancelled_reads},
        {"short_reads", value.short_reads},
        {"io_errors", value.io_errors},
        {"direct_source_count", value.direct_source_count},
        {"direct_unsupported_source_count", value.direct_unsupported_source_count},
        {"maximum_bundle_bytes", value.maximum_bundle_bytes},
    };
}

template<class T> T delta(T after, T before) {
    return after >= before ? after - before : 0;
}

json cold_delta_json(
        const llm_expert_cold_scalar_snapshot & before,
        const llm_expert_cold_scalar_snapshot & after) {
    return {
        {"requests", delta(after.requests, before.requests)},
        {"hits", delta(after.hits, before.hits)},
        {"misses", delta(after.misses, before.misses)},
        {"admissions", delta(after.admissions, before.admissions)},
        {"evictions", delta(after.evictions, before.evictions)},
        {"occupancy_before", before.occupancy},
        {"occupancy_after", after.occupancy},
    };
}

json storage_delta_json(
        const llm_expert_storage_diagnostics & before,
        const llm_expert_storage_diagnostics & after) {
    return {
        {"backing_loads", delta(after.read_requests, before.read_requests)},
        {"backing_chunks", delta(after.read_chunks, before.read_chunks)},
        {"backing_bytes", delta(after.read_bytes, before.read_bytes)},
        {"cancelled_reads", delta(after.cancelled_reads, before.cancelled_reads)},
        {"short_reads", delta(after.short_reads, before.short_reads)},
        {"io_errors", delta(after.io_errors, before.io_errors)},
    };
}

json async_delta_json(
        const llm_expert_async_diagnostics & before,
        const llm_expert_async_diagnostics & after) {
    return {
        {"read_requests_submitted", delta(after.read_requests_submitted,
                                           before.read_requests_submitted)},
        {"read_requests_completed", delta(after.read_requests_completed,
                                           before.read_requests_completed)},
        {"read_requests_cancelled", delta(after.read_requests_cancelled,
                                           before.read_requests_cancelled)},
        {"read_operations_completed", delta(after.read_operations_completed,
                                             before.read_operations_completed)},
        {"read_bytes_completed", delta(after.read_bytes_completed,
                                        before.read_bytes_completed)},
        {"queue_wait_samples", delta(after.read_queue_wait_samples,
                                      before.read_queue_wait_samples)},
        {"queue_wait_us", delta(after.read_queue_wait_us, before.read_queue_wait_us)},
        {"queue_wait_max_us_lifetime", after.read_queue_wait_max_us},
        {"ring_submissions", delta(after.ring_submissions, before.ring_submissions)},
        {"ring_completions", delta(after.ring_completions, before.ring_completions)},
        {"ring_request_batches", delta(after.ring_request_batches,
                                        before.ring_request_batches)},
        {"peak_ring_batch_requests_lifetime", after.peak_ring_batch_requests},
        {"peak_active_read_requests_lifetime", after.peak_active_read_requests},
        {"peak_active_operations_lifetime", after.peak_active_operations},
        {"peak_sq_occupancy_lifetime", after.peak_sq_occupancy},
        {"peak_cq_occupancy_lifetime", after.peak_cq_occupancy},
        {"cq_empty_waits", delta(after.cq_empty_waits, before.cq_empty_waits)},
        {"direct_read_operations", delta(after.direct_read_operations,
                                          before.direct_read_operations)},
        {"direct_useful_bytes", delta(after.direct_useful_bytes,
                                       before.direct_useful_bytes)},
        {"direct_aligned_bytes", delta(after.direct_aligned_bytes,
                                        before.direct_aligned_bytes)},
        {"buffered_fallback_operations", delta(after.buffered_fallback_operations,
                                                 before.buffered_fallback_operations)},
        {"synchronous_fallback_operations", delta(after.synchronous_fallback_operations,
                                                    before.synchronous_fallback_operations)},
    };
}

json scheduler_delta_json(
        const llm_expert_scheduler_diagnostics & before,
        const llm_expert_scheduler_diagnostics & after) {
    return {
        {"flights_created", delta(after.flights_created, before.flights_created)},
        {"joins", delta(after.joins, before.joins)},
        {"pending_successors", delta(after.pending_successors,
                                      before.pending_successors)},
        {"successor_activations", delta(after.successor_activations,
                                         before.successor_activations)},
        {"successor_cancellations", delta(after.successor_cancellations,
                                           before.successor_cancellations)},
        {"terminal_complete", delta(after.terminal_complete, before.terminal_complete)},
        {"terminal_failed", delta(after.terminal_failed, before.terminal_failed)},
        {"terminal_cancelled", delta(after.terminal_cancelled, before.terminal_cancelled)},
        {"terminal_releases", delta(after.terminal_releases, before.terminal_releases)},
        {"stale_completions", delta(after.stale_completions, before.stale_completions)},
        {"active_requests", after.active_requests},
        {"queued_requests", after.queued_requests},
        {"peak_active_requests_lifetime", after.peak_active_requests},
    };
}

json terminal_reference_json(const llm_hot_cache_diagnostics & value) {
    return {
        {"cold_hot_refs", value.cold_current_hot_refs},
        {"cold_transfer_refs", value.cold_current_transfer_refs},
        {"cold_request_refs", value.cold_current_request_refs},
        {"cold_cpu_execution_refs", value.cold_current_cpu_execution_refs},
        {"cold_batch_refs", value.cold_current_batch_refs},
        {"provider_pins", value.current_pins},
    };
}

std::vector<llama_token> load_prompt(const std::string & path, const llama_vocab * vocab) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("unable to open prompt corpus");
    json corpus;
    input >> corpus;
    const auto & cases = corpus.at("cases");
    const auto selected = std::find_if(cases.begin(), cases.end(), [](const json & item) {
        return item.value("id", "") == "issue73-decision-prompt";
    });
    if (selected == cases.end()) throw std::runtime_error("decision prompt is absent");
    const std::string prompt = selected->at("prompt").get<std::string>();
    const int expected = selected->at("expected_prompt_tokens").get<int>();
    const int count = -llama_tokenize(vocab, prompt.data(), prompt.size(), nullptr, 0, true, true);
    if (count != expected || count != 100) throw std::runtime_error("decision prompt token identity mismatch");
    std::vector<llama_token> tokens((size_t(count)));
    if (llama_tokenize(vocab, prompt.data(), prompt.size(), tokens.data(), tokens.size(), true, true) != count) {
        throw std::runtime_error("decision prompt tokenization failed");
    }
    return tokens;
}

} // namespace

int main(int argc, char ** argv) {
    arguments args;
    if (!parse_arguments(argc, argv, args)) {
        std::fprintf(stderr,
            "usage: %s --model GGUF --prompt-corpus JSON --output JSON --point EXACT|KNEE|AGGRESSIVE "
            "[--issue-mode SERIAL|BATCHED --cold-cache-bytes N --warmup-limit N "
            "--decode-forwards N --threads N --n-ctx N --prompt-token ID]\n",
            argv[0]);
        return 2;
    }

    try {
        const auto process_started = steady_clock::now();
        llama_log_set([](ggml_log_level level, const char * text, void *) {
            if (level == GGML_LOG_LEVEL_ERROR) std::fputs(text, stderr);
        }, nullptr);
        ggml_backend_load_all();
        uint32_t gpu_devices = 0;
        for (size_t index = 0; index < ggml_backend_dev_count(); ++index) {
            gpu_devices += ggml_backend_dev_type(ggml_backend_dev_get(index)) == GGML_BACKEND_DEVICE_TYPE_GPU;
        }
        if (gpu_devices != 0) throw std::runtime_error("GPU backend/device present in CPU-only decision probe");

        auto model_params = llama_model_default_params();
        model_params.n_gpu_layers = 0;
        model_params.use_extra_bufts = false;
        model_params.load_mode = LLAMA_LOAD_MODE_DIRECT_IO;
        model_params.expert_weights_mode = LLAMA_EXPERT_WEIGHTS_MODE_COLD_CACHE;
        model_params.expert_runtime_mode = LLAMA_EXPERT_RUNTIME_MODE_PERFORMANCE;
        model_params.expert_hot_cache_capacity = 0;
        model_params.expert_cold_cache_bytes = args.cold_cache_bytes;
        model_params.expert_transfer_ring_bytes = 0;
        model_params.expert_miss_policy = LLAMA_EXPERT_MISS_POLICY_CPU_FALLBACK;
        model_params.expert_io_trace_capacity = 0;
        model_params.expert_background_promotion = false;
        model_params.expert_async_cold_fill = false;

        const auto model_load_started = steady_clock::now();
        llama_model_ptr model(llama_model_load_from_file(args.model.c_str(), model_params));
        if (!model) throw std::runtime_error("model load failed");
        const auto model_loaded = steady_clock::now();
        if (!model->uses_cpu_cold_cache()) throw std::runtime_error("CPU cold-only model topology not selected");
        auto * provider = model->expert_weight_provider();
        if (provider == nullptr ||
            !provider->debug_set_host_resident_serial_issue_for_testing(
                args.issue_mode == "SERIAL").is_ready()) {
            throw std::runtime_error("unable to configure the internal issue-mode evidence seam");
        }
        const llama_vocab * vocab = llama_model_get_vocab(model.get());
        const int n_vocab = llama_vocab_n_tokens(vocab);
        const auto prompt = args.prompt_token >= 0 ? std::vector<llama_token>{args.prompt_token} :
            load_prompt(args.prompt_corpus, vocab);
        if (std::any_of(prompt.begin(), prompt.end(), [n_vocab](llama_token token) {
                return token < 0 || token >= n_vocab;
            })) throw std::runtime_error("prompt token is out of vocabulary");

        auto context_params = llama_context_default_params();
        context_params.n_ctx = args.n_ctx;
        context_params.n_batch = 1;
        context_params.n_ubatch = 1;
        context_params.n_threads = int32_t(args.threads);
        context_params.n_threads_batch = int32_t(args.threads);
        context_params.no_perf = true;
        const auto context_load_started = steady_clock::now();
        llama_context_ptr context(llama_init_from_model(model.get(), context_params));
        if (!context) throw std::runtime_error("context initialization failed");
        const auto context_loaded = steady_clock::now();

        auto * storage = model->expert_storage();
        if (provider == nullptr || storage == nullptr) throw std::runtime_error("CPU provider/storage unavailable");
        const auto initial_cold = provider->cold_cache_scalar_snapshot();
        const auto initial_storage = storage->diagnostics();
        const auto initial_async = model->expert_async_diagnostics();
        const auto initial_scheduler = model->expert_scheduler_diagnostics();
        const auto initial_full = provider->hot_cache_diagnostics();
        const uint64_t allowed_async_fallback_mask =
            uint64_t(llm_expert_async_fallback_reason::buffer_registration);
        if (!initial_cold.available || initial_cold.occupancy != 0 || initial_cold.capacity == 0 ||
            initial_full.system_memory_requested_pool_bytes != args.cold_cache_bytes ||
            initial_full.system_memory_selected_pool_bytes != initial_cold.requested_bytes ||
            !initial_full.system_memory_budget_frozen ||
            initial_full.system_memory_autofit != (args.cold_cache_bytes == 0) || !initial_full.cpu_cold_only ||
            initial_full.requested_capacity != 0 || initial_full.effective_capacity != 0 ||
            initial_full.pool_bytes != 0 || !initial_full.slots.empty() ||
            initial_storage.direct_source_count != initial_storage.source_file_count ||
            initial_storage.direct_unsupported_source_count != 0 || !initial_async.io_uring_enabled ||
            (initial_async.fallback_reason_mask & ~allowed_async_fallback_mask) != 0) {
            throw std::runtime_error("CPU production-path initial validation failed");
        }

        auto decode_one = [&](llama_token token, bool changed_routing, uint64_t request) -> llama_token {
            if (changed_routing && llama_cache_aware_routing_begin(
                    context.get(), request, LLAMA_ROUTE_PHASE_DECODE) != LLAMA_ROUTE_OBSERVER_STATUS_OK) {
                throw std::runtime_error("cache-aware routing begin failed");
            }
            llama_batch batch = llama_batch_get_one(&token, 1);
            if (llama_decode(context.get(), batch) != 0) throw std::runtime_error("decode failed");
            llama_synchronize(context.get());
            const float * logits = llama_get_logits_ith(context.get(), -1);
            const int next = logits == nullptr ? -1 : finite_argmax(logits, n_vocab);
            if (next < 0) throw std::runtime_error("non-finite logits");
            return llama_token(next);
        };

        const auto fill_started = steady_clock::now();
        llm_expert_cold_scalar_snapshot fill_cold;
        llm_expert_storage_diagnostics fill_storage;
        llama_token next_token = prompt.front();
        uint32_t tokens_to_full = 0;
        bool seam_residency_visible = false;
        for (uint32_t index = 0; index < args.warmup_limit; ++index) {
            const llama_token input = index < prompt.size() ? prompt[index] : next_token;
            next_token = decode_one(input, false, 0);
            tokens_to_full = index + 1;
            fill_cold = provider->cold_cache_scalar_snapshot();
            fill_storage = storage->diagnostics();
            if (index == 0) {
                if (fill_cold.misses <= initial_cold.misses ||
                    fill_storage.read_requests <= initial_storage.read_requests ||
                    fill_storage.read_bytes <= initial_storage.read_bytes) {
                    throw std::runtime_error("first real miss did not increase backing reads");
                }
                std::vector<int32_t> experts(size_t(model->hparams.n_expert));
                std::vector<llama_route_service_tier> tiers(size_t(model->hparams.n_expert));
                for (int32_t expert = 0; expert < int32_t(experts.size()); ++expert) experts[expert] = expert;
                bool saw_routed_layer = false;
                for (size_t layer = 0; layer < model->layers.size(); ++layer) {
                    if (model->layers[layer].ffn_down_exps == nullptr) continue;
                    saw_routed_layer = true;
                    const auto tier_result = provider->route_service_tier_snapshot(
                        int32_t(layer), experts.data(), experts.size(), tiers.data());
                    if (!tier_result.is_ready() ||
                        std::find(tiers.begin(), tiers.end(), LLAMA_ROUTE_SERVICE_TIER_HOT) != tiers.end()) {
                        throw std::runtime_error("CPU service-tier seam returned invalid HOT/stale state");
                    }
                    seam_residency_visible = seam_residency_visible ||
                        std::find(tiers.begin(), tiers.end(), LLAMA_ROUTE_SERVICE_TIER_COLD) != tiers.end();
                }
                if (!saw_routed_layer || !seam_residency_visible) {
                    throw std::runtime_error("real cold residency is not visible to routing");
                }
            }
            if (fill_cold.occupancy == fill_cold.capacity) break;
        }
        const auto fill_completed = steady_clock::now();
        if (fill_cold.occupancy != fill_cold.capacity) {
            throw std::runtime_error("real cold cache did not fill within the bounded warmup: occupancy=" +
                std::to_string(fill_cold.occupancy) + " capacity=" + std::to_string(fill_cold.capacity));
        }
        const auto fill_async = model->expert_async_diagnostics();
        const auto fill_scheduler = model->expert_scheduler_diagnostics();

        const bool changed_routing = args.point != "EXACT";
        uint32_t max_swaps = 0;
        float max_regret = 0.0f;
        if (args.point == "KNEE") {
            max_swaps = 1;
            max_regret = 0.0030885785818099976f;
        } else if (args.point == "AGGRESSIVE") {
            max_swaps = 4;
            max_regret = 0.007303759455680847f;
        }
        if (changed_routing) {
            const llama_cache_aware_routing_config routing = {
                true, 32, max_swaps, max_regret, nullptr, nullptr, nullptr,
            };
            if (llama_set_cache_aware_routing(context.get(), &routing) != LLAMA_ROUTE_OBSERVER_STATUS_OK) {
                throw std::runtime_error("provider-backed cache-aware routing configuration failed");
            }
            context->sched_reserve();
        }
        llama_cache_aware_routing_reset_stats(context.get());

        const auto before_cold = provider->cold_cache_scalar_snapshot();
        const auto before_storage = storage->diagnostics();
        const auto before_async = model->expert_async_diagnostics();
        const auto before_scheduler = model->expert_scheduler_diagnostics();
        std::vector<llama_token> generated;
        generated.reserve(args.decode_forwards);
        std::vector<double> forward_latency_s;
        forward_latency_s.reserve(args.decode_forwards);

        struct rusage measured_usage_before {};
        if (getrusage(RUSAGE_SELF, &measured_usage_before) != 0) {
            throw std::runtime_error("initial measured getrusage failed");
        }
        const auto measured_started = steady_clock::now();
        for (uint32_t index = 0; index < args.decode_forwards; ++index) {
            const auto forward_started = steady_clock::now();
            next_token = decode_one(next_token, changed_routing, 1);
            const auto forward_completed = steady_clock::now();
            generated.push_back(next_token);
            forward_latency_s.push_back(seconds(forward_completed - forward_started));
        }
        const auto measured_completed = steady_clock::now();
        struct rusage measured_usage_after {};
        if (getrusage(RUSAGE_SELF, &measured_usage_after) != 0) {
            throw std::runtime_error("final measured getrusage failed");
        }

        const auto post_started = steady_clock::now();
        const auto after_cold = provider->cold_cache_scalar_snapshot();
        const auto after_storage = storage->diagnostics();
        const auto after_async = model->expert_async_diagnostics();
        const auto after_scheduler = model->expert_scheduler_diagnostics();
        const auto after_full = provider->hot_cache_diagnostics();
        const auto routing_stats = llama_cache_aware_routing_get_stats(context.get());
        struct rusage usage {};
        if (getrusage(RUSAGE_SELF, &usage) != 0) throw std::runtime_error("getrusage failed");
        const uint64_t swap_kib = vm_swap_kib();
        const auto post_completed = steady_clock::now();

        if (after_storage.cancelled_reads != 0 || after_storage.short_reads != 0 ||
            after_storage.io_errors != 0 || after_async.buffered_fallback_operations != 0 ||
            after_async.synchronous_fallback_operations != 0 ||
            (after_async.fallback_reason_mask & ~allowed_async_fallback_mask) != 0 ||
            after_async.direct_read_operations <= initial_async.direct_read_operations ||
            after_scheduler.active_requests != 0 || after_scheduler.queued_requests != 0 ||
            after_scheduler.terminal_failed != 0 || after_scheduler.terminal_cancelled != 0 ||
            after_scheduler.stale_completions != 0 ||
            after_full.cold_current_hot_refs != 0 || after_full.cold_current_transfer_refs != 0 ||
            after_full.cold_current_request_refs != 0 ||
            after_full.cold_current_cpu_execution_refs != 0 ||
            after_full.cold_current_batch_refs != 0 || after_full.current_pins != 0 ||
            swap_kib != 0 || routing_stats.failures != 0 ||
            (!changed_routing && (routing_stats.ubatches != 0 || routing_stats.layers != 0 ||
                                  routing_stats.decisions != 0 || routing_stats.swaps != 0))) {
            throw std::runtime_error("measured production-path invariant failed");
        }

        const double measured_s = seconds(measured_completed - measured_started);
        const double measured_user_cpu_s =
            seconds(measured_usage_after.ru_utime) - seconds(measured_usage_before.ru_utime);
        const double measured_system_cpu_s =
            seconds(measured_usage_after.ru_stime) - seconds(measured_usage_before.ru_stime);
        const double init_s = seconds(context_loaded - process_started);
        const double fill_s = seconds(fill_completed - fill_started);
        const double post_s = seconds(post_completed - post_started);
        const double accounted_s = init_s + fill_s + measured_s + post_s;
        json command = json::array();
        for (int index = 0; index < argc; ++index) command.push_back(argv[index]);

        const json result = {
            {"schema_version", "phase13-6p-cpu-demand-v1"},
            {"status", "pass"},
            {"exit_status", 0},
            {"point", args.point},
            {"pid", getpid()},
            {"command", command},
            {"model_path", args.model},
            {"prompt_corpus", args.prompt_corpus},
            {"execution", {
                {"backend", "CPU"}, {"n_gpu_layers", 0}, {"gpu_device_count", gpu_devices},
                {"cuda_dependency", "none"}, {"load_mode", "DIRECT_IO"},
                {"runtime_mode", "PERFORMANCE"}, {"n_ctx", args.n_ctx},
                {"n_batch", 1}, {"n_ubatch", 1}, {"threads", args.threads},
                {"current_layer_issue_mode", args.issue_mode},
                {"serial_control", args.issue_mode == "SERIAL"},
                {"native_io_uring", initial_async.io_uring_enabled},
                {"direct_staging_bytes", initial_async.staging_ceiling_bytes},
                {"direct_staging_lanes", initial_async.direct_staging_lane_count},
                {"registered_file_count", initial_async.registered_file_count},
                {"registered_buffer_count", initial_async.registered_buffer_count},
                {"buffer_registration_error", initial_async.buffer_registration_error},
                {"async_fallback_reason_mask", initial_async.fallback_reason_mask},
            }},
            {"routing", {
                {"enabled", changed_routing}, {"candidate_count", changed_routing ? 32 : 0},
                {"max_swaps", max_swaps}, {"max_score_regret", max_regret},
                {"tier_source", changed_routing ? "real-provider-cold-cache" : "disabled"},
                {"stats", {
                    {"ubatches", routing_stats.ubatches}, {"layers", routing_stats.layers},
                    {"decisions", routing_stats.decisions},
                    {"changed_decisions", routing_stats.changed_decisions},
                    {"swaps", routing_stats.swaps},
                    {"cumulative_score_regret", routing_stats.cumulative_score_regret},
                    {"explicit_synchronizations", routing_stats.explicit_synchronizations},
                    {"failures", routing_stats.failures},
                }},
            }},
            {"preflight", {
                {"pass", true}, {"process_start_occupancy", initial_cold.occupancy},
                {"cpu_cold_only", initial_full.cpu_cold_only},
                {"hot_capacity", initial_full.effective_capacity},
                {"hot_pool_bytes", initial_full.pool_bytes},
                {"first_miss_backing_read", true},
                {"same_cache_residency_visible_to_routing", seam_residency_visible},
                {"initial_cold", cold_json(initial_cold)},
                {"initial_storage", storage_json(initial_storage)},
                {"initial_terminal_references", terminal_reference_json(initial_full)},
                {"system_memory", system_memory_json(initial_full)},
            }},
            {"fill", {
                {"tokens_to_full", tokens_to_full}, {"time_to_full_s", fill_s},
                {"cold", cold_json(fill_cold)},
                {"cold_delta", cold_delta_json(initial_cold, fill_cold)},
                {"storage_delta", storage_delta_json(initial_storage, fill_storage)},
                {"async_delta", async_delta_json(initial_async, fill_async)},
                {"scheduler_delta", scheduler_delta_json(initial_scheduler, fill_scheduler)},
            }},
            {"measured", {
                {"decode_forwards", args.decode_forwards}, {"decode_s", measured_s},
                {"decode_tok_s", double(args.decode_forwards)/measured_s},
                {"p50_forward_s", percentile(forward_latency_s, 0.50)},
                {"p95_forward_s", percentile(forward_latency_s, 0.95)},
                {"p99_forward_s", percentile(forward_latency_s, 0.99)},
                {"cold_before", cold_json(before_cold)}, {"cold_after", cold_json(after_cold)},
                {"cold_delta", cold_delta_json(before_cold, after_cold)},
                {"storage_delta", storage_delta_json(before_storage, after_storage)},
                {"async_delta", async_delta_json(before_async, after_async)},
                {"scheduler_delta", scheduler_delta_json(before_scheduler, after_scheduler)},
                {"user_cpu_s", measured_user_cpu_s},
                {"system_cpu_s", measured_system_cpu_s},
                {"process_cpu_utilization", measured_s == 0 ? 0 :
                    (measured_user_cpu_s + measured_system_cpu_s)/measured_s},
                {"minor_faults", delta(measured_usage_after.ru_minflt,
                                        measured_usage_before.ru_minflt)},
                {"major_faults", delta(measured_usage_after.ru_majflt,
                                        measured_usage_before.ru_majflt)},
            }},
            {"output", {
                {"generated_token_count", generated.size()},
                {"generated_token_hash", hex_u64(token_hash(generated))},
                {"generated_ids", generated},
            }},
            {"resources", {
                {"peak_rss_kib", usage.ru_maxrss}, {"vm_swap_kib", swap_kib},
                {"terminal_references", terminal_reference_json(after_full)},
                {"system_memory", system_memory_json(after_full)},
                {"terminal_scheduler_active_requests", after_scheduler.active_requests},
                {"terminal_scheduler_queued_requests", after_scheduler.queued_requests},
            }},
            {"wall_fractions", {
                {"model_load_s", seconds(model_loaded - model_load_started)},
                {"context_init_s", seconds(context_loaded - context_load_started)},
                {"initialization", accounted_s == 0 ? 0 : init_s/accounted_s},
                {"cache_fill", accounted_s == 0 ? 0 : fill_s/accounted_s},
                {"measured_decode", accounted_s == 0 ? 0 : measured_s/accounted_s},
                {"post_run", accounted_s == 0 ? 0 : post_s/accounted_s},
            }},
        };

        std::ofstream output(args.output, std::ios::trunc);
        if (!output) throw std::runtime_error("unable to open output file");
        output << result.dump(2) << '\n';
        if (!output) throw std::runtime_error("unable to write output file");
        std::printf("%s\n", result.dump().c_str());
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "phase13-mode-p-probe: %s\n", error.what());
        return 1;
    }
}
