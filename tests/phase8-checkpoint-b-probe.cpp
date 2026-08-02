#include "llama-expert-weight-provider.h"
#include "llama-expert-scheduler.h"
#include "llama-context.h"
#include "llama-model.h"
#include "llama-cpp.h"

#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <future>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

llama_expert_auto_cost_model gpu_cost_model() {
    return {
        LLAMA_EXPERT_AUTO_COST_MODEL_VERSION_1,
        sizeof(llama_expert_auto_cost_model),
        1000000000000ULL, 1000000000000ULL,
        1000000000000ULL, 1000000000000ULL,
        1, 1, 1, 1, 1, UINT64_MAX, 1,
    };
}

struct source_fixture {
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buffer;
    ggml_tensor * up = nullptr;
    ggml_tensor * gate = nullptr;
    ggml_tensor * down = nullptr;
    ggml_tensor * ids = nullptr;

    source_fixture() {
        ggml_init_params params = { ggml_tensor_overhead()*8, nullptr, true };
        ctx.reset(ggml_init(params));
        GGML_ASSERT(ctx);
        up = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 2, 4, 2);
        gate = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 2, 4, 2);
        down = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 4, 2, 2);
        ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 1, 1);
        buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), ggml_backend_cpu_buffer_type()));
        GGML_ASSERT(buffer);
    }

    llm_expert_bundle_descriptor bundle() const {
        return { 0, 2,
            llm_expert_projection_descriptor::from(up, nullptr, nullptr),
            llm_expert_projection_descriptor::from(gate, nullptr, nullptr), {},
            llm_expert_projection_descriptor::from(down, nullptr, nullptr) };
    }
};

struct evidence_counters {
    uint64_t background_submitted = 0;
    uint64_t background_published_completed = 0;
    uint64_t background_dropped = 0;
    uint64_t background_useful = 0;
    uint64_t background_wasted = 0;
    uint64_t background_busy = 0;
    uint64_t scheduler_terminal_complete = 0;
    uint64_t scheduler_terminal_failed = 0;
    uint64_t scheduler_terminal_cancelled = 0;
    uint64_t scheduler_terminal_releases = 0;
};

evidence_counters capture_evidence(
        llm_expert_weight_provider * provider,
        const llm_expert_scheduler_diagnostics & scheduler) {
    const auto diagnostics = provider->hot_cache_diagnostics();
    return {
        diagnostics.background_submitted,
        diagnostics.background_completed,
        diagnostics.background_dropped,
        diagnostics.background_useful,
        diagnostics.background_wasted,
        diagnostics.background_busy,
        scheduler.terminal_complete,
        scheduler.terminal_failed,
        scheduler.terminal_cancelled,
        scheduler.terminal_releases,
    };
}

struct provider_fixture {
    source_fixture source;
    llm_expert_phase8_test_control control;
    llm_expert_scheduler scheduler { { 1, 2, 8, 8, 2, 0 } };
    ggml_backend_ptr backend;
    ggml_context_ptr graph_ctx;
    ggml_backend_buffer_ptr graph_buffer;
    std::unique_ptr<llm_expert_weight_provider> provider;
    llm_expert_graph_binding binding;
    llm_expert_phase8_closeout_witness witness;

    explicit provider_fixture(bool background = true) {
        auto * gpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        GGML_ASSERT(gpu != nullptr);
        backend.reset(ggml_backend_dev_init(gpu, nullptr));
        GGML_ASSERT(backend);
        llm_hot_cache_config config;
        config.capacity = 2;
        config.n_expert_used = 1;
        config.routed_layer_count = 1;
        config.total_expert_keys = 2;
        config.target_buffer_type = ggml_backend_get_default_buffer_type(backend.get());
        config.allow_non_cuda_target_for_testing = true;
        config.cold_mode = true;
        config.cold_cache_bytes = 1U << 20;
        config.transfer_ring_bytes = 1U << 20;
        config.target_device = gpu;
        config.scheduler = &scheduler;
        config.miss_policy = LLAMA_EXPERT_MISS_POLICY_CPU_FALLBACK;
        config.background_promotion = background;
        config.phase8_test_control = &control;
        provider = llm_create_cold_cache_expert_weight_provider(config);

        ggml_init_params graph_params = { ggml_tensor_overhead()*16, nullptr, true };
        graph_ctx.reset(ggml_init(graph_params));
        GGML_ASSERT(graph_ctx);
        const llm_expert_selection selection = { 0, 2, 1, 1, source.ids };
        const auto bootstrap = provider->bind_graph(graph_ctx.get(), source.bundle(), selection, binding);
        if (!bootstrap.is_ready()) std::fprintf(stderr, "bootstrap bind failed status=%d error=%d\n",
            int(bootstrap.status), int(bootstrap.error));
        GGML_ASSERT(bootstrap.is_ready());
        GGML_ASSERT(binding.bootstrap);
        GGML_ASSERT(provider->initialize_after_reserve().is_ready());
        binding = {};
        GGML_ASSERT(provider->bind_graph(graph_ctx.get(), source.bundle(), selection, binding).is_ready());
        GGML_ASSERT(binding.hybrid);
        graph_buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(
            graph_ctx.get(), ggml_backend_cpu_buffer_type()));
        GGML_ASSERT(graph_buffer);
    }

    llm_expert_provider_result remap(int32_t expert) {
        ggml_backend_tensor_set(binding.checkpoint_ids, &expert, 0, sizeof(expert));
        llm_expert_execution_plan plan;
        auto result = provider->prepare({ binding }, plan);
        if (result.is_ready()) result = provider->remap_checkpoint_tensor(binding, backend.get());
        plan.reset();
        return result;
    }

    std::future<llm_expert_provider_result> remap_async(int32_t expert) {
        return std::async(std::launch::async, [this, expert] { return remap(expert); });
    }

    void select_auto_gpu() {
        const auto cost = gpu_cost_model();
        GGML_ASSERT(provider->debug_set_auto_cost_model_for_testing(cost).is_ready());
    }

    void select_cpu() {
        GGML_ASSERT(provider->debug_set_miss_policy_for_testing(
            LLAMA_EXPERT_MISS_POLICY_CPU_FALLBACK).is_ready());
    }

    void install_closeout() {
        GGML_ASSERT(provider->set_phase8_closeout_witness_for_testing(&witness).is_ready());
    }

    evidence_counters capture() const {
        return capture_evidence(provider.get(), scheduler.diagnostics());
    }

    void destroy() {
        if (!provider) return;
        install_closeout();
        provider.reset();
        GGML_ASSERT(witness.written && witness.write_count == 1 && witness.final_invariants_ok);
        GGML_ASSERT(witness.ring_queued_workers == 0 && witness.ring_running_workers == 0 &&
            witness.ring_non_free_lanes == 0 && witness.ring_live_events == 0);
        GGML_ASSERT(witness.cold_hot_refs == 0 && witness.cold_transfer_refs == 0 &&
            witness.cold_request_refs == 0 && witness.cold_cpu_execution_refs == 0);
    }
};

struct cuda_gate {
    ggml_backend_ptr backend;
    ggml_backend_event_t event = nullptr;
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr device_buffer;
    ggml_backend_buffer_ptr source_buffer;
    ggml_tensor * tensor = nullptr;

    explicit cuda_gate(provider_fixture & fixture) {
        auto * gpu = ggml_backend_get_device(fixture.backend.get());
        backend.reset(ggml_backend_dev_init(gpu, nullptr));
        GGML_ASSERT(backend);
        event = ggml_backend_event_new(gpu);
        GGML_ASSERT(event != nullptr);
        constexpr size_t gate_bytes = 256U*1024U*1024U;
        ggml_init_params params = { ggml_tensor_overhead()*2, nullptr, true };
        ctx.reset(ggml_init(params));
        GGML_ASSERT(ctx);
        tensor = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I8, gate_bytes);
        device_buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(
            ctx.get(), ggml_backend_dev_buffer_type(gpu)));
        source_buffer.reset(ggml_backend_buft_alloc_buffer(
            ggml_backend_dev_host_buffer_type(gpu), gate_bytes));
        GGML_ASSERT(device_buffer && source_buffer && ggml_backend_buffer_is_host(source_buffer.get()));
        ggml_backend_buffer_clear(source_buffer.get(), 0);
        for (int repetition = 0; repetition < 8; ++repetition) {
            ggml_backend_tensor_set_async(backend.get(), tensor,
                ggml_backend_buffer_get_base(source_buffer.get()), 0, gate_bytes);
        }
        ggml_backend_event_record(event, backend.get());
        GGML_ASSERT(fixture.provider->set_h2d_gate_event_for_testing(event).is_ready());
    }

    ~cuda_gate() {
        if (event != nullptr) ggml_backend_event_free(event);
    }

    void release() {
        ggml_backend_synchronize(backend.get());
    }
};

std::string auto_record_json(const llm_hot_cache_diagnostics::auto_decision & r) {
    static const char * states[] = { "NONE", "QUEUED_OR_STAGING", "H2D_IN_FLIGHT", "H2D_COMPLETE_UNPUBLISHED" };
    static const char * backends[] = { "cpu", "gpu" };
    static const char * reasons[] = { "cpu_faster", "gpu_faster_or_hysteresis", "tie", "overflow" };
    std::ostringstream out;
    out << "{\"request\":" << r.request
        << ",\"layer\":" << r.layer << ",\"expert\":" << r.expert
        << ",\"cost\":{\"version\":" << r.cost.version
        << ",\"struct_size\":" << r.cost.struct_size
        << ",\"cpu_fixed_decode_ns\":" << r.cost.cpu_fixed_decode_ns
        << ",\"cpu_fixed_prefill_ns\":" << r.cost.cpu_fixed_prefill_ns
        << ",\"cpu_per_lane_decode_ns\":" << r.cost.cpu_per_lane_decode_ns
        << ",\"cpu_per_lane_prefill_ns\":" << r.cost.cpu_per_lane_prefill_ns
        << ",\"gpu_fixed_decode_ns\":" << r.cost.gpu_fixed_decode_ns
        << ",\"gpu_fixed_prefill_ns\":" << r.cost.gpu_fixed_prefill_ns
        << ",\"gpu_per_lane_decode_ns\":" << r.cost.gpu_per_lane_decode_ns
        << ",\"gpu_per_lane_prefill_ns\":" << r.cost.gpu_per_lane_prefill_ns
        << ",\"h2d_fixed_ns\":" << r.cost.h2d_fixed_ns
        << ",\"h2d_bytes_per_second\":" << r.cost.h2d_bytes_per_second
        << ",\"decision_hysteresis_ns\":" << r.cost.decision_hysteresis_ns << "}"
        << ",\"prefill\":" << (r.prefill ? "true" : "false")
        << ",\"lanes\":" << r.lanes << ",\"bundle_bytes\":" << r.bundle_bytes
        << ",\"queued_cpu_work_ns\":" << r.queued_cpu_work_ns
        << ",\"queued_h2d_work_ns\":" << r.queued_h2d_work_ns
        << ",\"queued_gpu_work_ns\":" << r.queued_gpu_work_ns
        << ",\"same_key_h2d_present\":" << (r.same_key_h2d_present ? "true" : "false")
        << ",\"same_key_h2d_state\":\"" << states[r.same_key_h2d_state] << "\""
        << ",\"same_key_h2d_remaining_bytes\":" << r.same_key_h2d_remaining_bytes
        << ",\"result\":{\"backend\":\"" << backends[r.backend] << "\",\"reason\":\""
        << reasons[r.reason] << "\",\"cpu_work_ns\":" << r.cpu_work_ns
        << ",\"h2d_work_ns\":" << r.h2d_work_ns << ",\"gpu_work_ns\":" << r.gpu_work_ns
        << ",\"cpu_finish_ns\":" << r.cpu_finish_ns << ",\"gpu_finish_ns\":" << r.gpu_finish_ns
        << ",\"overflow\":" << (r.overflow ? "true" : "false") << "}}";
    return out.str();
}

const llm_hot_cache_diagnostics::auto_decision & only_last_decision(
        const llm_hot_cache_diagnostics & diagnostics,
        uint64_t before) {
    GGML_ASSERT(diagnostics.auto_decision_records == before + 1);
    GGML_ASSERT(!diagnostics.auto_decisions.empty());
    return diagnostics.auto_decisions.back();
}

std::string case_json(const char * name, const std::string & fields = {}) {
    std::ostringstream out;
    out << "\"" << name << "\":{\"status\":\"pass\"";
    if (!fields.empty()) out << "," << fields;
    out << "}";
    return out.str();
}

std::string json_string(const std::string & value) {
    std::ostringstream out;
    out << '"';
    for (unsigned char ch : value) {
        if (ch == '"' || ch == '\\') out << '\\' << char(ch);
        else if (ch == '\n') out << "\\n";
        else if (ch == '\r') out << "\\r";
        else if (ch == '\t') out << "\\t";
        else if (ch < 0x20) {
            static const char * hex = "0123456789abcdef";
            out << "\\u00" << hex[ch >> 4] << hex[ch & 0xf];
        } else out << char(ch);
    }
    out << '"';
    return out.str();
}

evidence_counters complete_evidence(
        const evidence_counters & baseline,
        const llm_expert_phase8_closeout_witness & closeout);

std::string evidence_json(
        const evidence_counters & baseline,
        const llm_expert_phase8_closeout_witness & closeout,
        const evidence_counters & delta);

std::string run_none_case() {
    provider_fixture fixture;
    fixture.select_auto_gpu();
    const auto baseline = fixture.capture();
    const auto before = fixture.provider->hot_cache_diagnostics().auto_decision_records;
    GGML_ASSERT(fixture.remap(0).is_ready());
    const auto diagnostics = fixture.provider->hot_cache_diagnostics();
    const auto & record = only_last_decision(diagnostics, before);
    GGML_ASSERT(!record.same_key_h2d_present && record.same_key_h2d_state == 0 &&
        record.same_key_h2d_remaining_bytes == 0);
    fixture.destroy();
    const auto delta = complete_evidence(baseline, fixture.witness);
    return case_json("auto_operands_none", "\"record\":" + auto_record_json(record) +
        ",\"decision_digest_entries\":1," + evidence_json(baseline, fixture.witness, delta));
}

std::string run_live_operand_case(
        const char * name,
        llm_expert_phase8_test_gate gate,
        uint8_t expected_state) {
    provider_fixture fixture;
    GGML_ASSERT(fixture.control.arm_gate(gate, { 0, 0 }, 1).is_ready());
    const auto baseline = fixture.capture();
    GGML_ASSERT(fixture.remap(0).is_ready());
    fixture.control.wait_until_reached();
    fixture.select_auto_gpu();
    const auto before = fixture.provider->hot_cache_diagnostics().auto_decision_records;
    auto future = fixture.remap_async(0);
    fixture.control.wait_until_observed(
        llm_expert_phase8_test_gate::auto_after_decision_before_same_key_join);
    GGML_ASSERT(fixture.control.release_gate().is_ready());
    GGML_ASSERT(future.get().is_ready());
    const auto diagnostics = fixture.provider->hot_cache_diagnostics();
    const auto & record = only_last_decision(diagnostics, before);
    GGML_ASSERT(record.same_key_h2d_present && record.same_key_h2d_state == expected_state);
    fixture.destroy();
    const auto delta = complete_evidence(baseline, fixture.witness);
    return case_json(name, "\"record\":" + auto_record_json(record) +
        ",\"decision_digest_entries\":1," + evidence_json(baseline, fixture.witness, delta));
}

std::string run_complete_operand_case() {
    provider_fixture fixture;
    cuda_gate gate(fixture);
    GGML_ASSERT(fixture.control.arm_gate(
        llm_expert_phase8_test_gate::auto_after_normalization_before_background_snapshot,
        { 0, 0 }, 1).is_ready());
    const auto baseline = fixture.capture();
    GGML_ASSERT(fixture.remap(0).is_ready());
    fixture.control.wait_until_observed(llm_expert_phase8_test_gate::background_h2d_in_flight);
    fixture.select_auto_gpu();
    const auto before = fixture.provider->hot_cache_diagnostics().auto_decision_records;
    auto future = fixture.remap_async(0);
    fixture.control.wait_until_reached();
    gate.release();
    fixture.control.wait_until_observed(
        llm_expert_phase8_test_gate::background_h2d_complete_before_provider_publication);
    GGML_ASSERT(fixture.control.release_gate().is_ready());
    GGML_ASSERT(future.get().is_ready());
    const auto diagnostics = fixture.provider->hot_cache_diagnostics();
    const auto & record = only_last_decision(diagnostics, before);
    GGML_ASSERT(record.same_key_h2d_present && record.same_key_h2d_state == 3 &&
        record.same_key_h2d_remaining_bytes == 0);
    fixture.destroy();
    const auto delta = complete_evidence(baseline, fixture.witness);
    return case_json("auto_operands_complete_unpublished",
        "\"record\":" + auto_record_json(record) + ",\"decision_digest_entries\":1," +
        evidence_json(baseline, fixture.witness, delta));
}

uint64_t checked_delta(uint64_t current, uint64_t baseline) {
    GGML_ASSERT(current >= baseline);
    return current - baseline;
}

evidence_counters observed_delta(
        const evidence_counters & baseline,
        const llm_expert_phase8_closeout_witness & closeout) {
    return {
        checked_delta(closeout.background_submitted, baseline.background_submitted),
        checked_delta(closeout.background_published_completed, baseline.background_published_completed),
        checked_delta(closeout.background_dropped, baseline.background_dropped),
        checked_delta(closeout.background_useful, baseline.background_useful),
        checked_delta(closeout.background_wasted, baseline.background_wasted),
        checked_delta(closeout.background_busy, baseline.background_busy),
        checked_delta(closeout.scheduler_terminal_complete, baseline.scheduler_terminal_complete),
        checked_delta(closeout.scheduler_terminal_failed, baseline.scheduler_terminal_failed),
        checked_delta(closeout.scheduler_terminal_cancelled, baseline.scheduler_terminal_cancelled),
        checked_delta(closeout.scheduler_terminal_releases, baseline.scheduler_terminal_releases),
    };
}

evidence_counters complete_evidence(
        const evidence_counters & baseline,
        const llm_expert_phase8_closeout_witness & closeout) {
    GGML_ASSERT(closeout.written && closeout.write_count == 1 && closeout.final_invariants_ok);
    GGML_ASSERT(closeout.scheduler_active == 0 && closeout.scheduler_queued == 0);
    GGML_ASSERT(closeout.ring_queued_workers == 0 && closeout.ring_running_workers == 0 &&
        closeout.ring_non_free_lanes == 0 && closeout.ring_live_events == 0);
    GGML_ASSERT(closeout.cold_hot_refs == 0 && closeout.cold_transfer_refs == 0 &&
        closeout.cold_request_refs == 0 && closeout.cold_cpu_execution_refs == 0);
    GGML_ASSERT(closeout.hot_pins == 0 && closeout.published_forward_mappings == 0);
    for (size_t state = 1; state < closeout.background_lifecycle.size(); ++state) {
        GGML_ASSERT(closeout.background_lifecycle[state] == 0);
    }
    const auto delta = observed_delta(baseline, closeout);
    GGML_ASSERT(delta.scheduler_terminal_releases == delta.scheduler_terminal_complete +
        delta.scheduler_terminal_failed + delta.scheduler_terminal_cancelled);
    GGML_ASSERT(delta.background_submitted ==
        delta.background_published_completed + delta.background_dropped);
    GGML_ASSERT(delta.background_useful + delta.background_wasted ==
        delta.background_published_completed);
    if (delta.background_dropped > 0) {
        GGML_ASSERT(delta.background_published_completed == 0 &&
            delta.background_useful == 0 && delta.background_wasted == 0);
    }
    return delta;
}

const char * terminal_from_delta(const evidence_counters & delta) {
    GGML_ASSERT(delta.scheduler_terminal_complete == 0);
    GGML_ASSERT(delta.scheduler_terminal_releases == 1);
    GGML_ASSERT((delta.scheduler_terminal_failed == 1 && delta.scheduler_terminal_cancelled == 0) ||
        (delta.scheduler_terminal_failed == 0 && delta.scheduler_terminal_cancelled == 1));
    return delta.scheduler_terminal_failed == 1 ? "failed" : "cancelled";
}

void assert_failed_flight(const evidence_counters & delta) {
    (void) terminal_from_delta(delta);
    GGML_ASSERT(delta.background_submitted == 1 && delta.background_published_completed == 0 &&
        delta.background_dropped == 1 && delta.background_useful == 0 &&
        delta.background_wasted == 0 && delta.background_busy == 0);
}

void assert_published_flight(
        const evidence_counters & delta,
        uint64_t useful,
        uint64_t wasted) {
    GGML_ASSERT(delta.background_submitted == 1 && delta.background_published_completed == 1 &&
        delta.background_dropped == 0 && delta.background_useful == useful &&
        delta.background_wasted == wasted && delta.background_busy == 0);
    GGML_ASSERT(delta.scheduler_terminal_complete == 1 && delta.scheduler_terminal_failed == 0 &&
        delta.scheduler_terminal_cancelled == 0 && delta.scheduler_terminal_releases == 1);
}

std::string counter_groups_json(const evidence_counters & counters) {
    std::ostringstream out;
    out << "{\"background\":{\"submitted\":" << counters.background_submitted
        << ",\"published_completed\":" << counters.background_published_completed
        << ",\"dropped\":" << counters.background_dropped
        << ",\"useful\":" << counters.background_useful
        << ",\"wasted\":" << counters.background_wasted
        << ",\"busy\":" << counters.background_busy
        << "},\"scheduler\":{\"terminal_complete\":" << counters.scheduler_terminal_complete
        << ",\"terminal_failed\":" << counters.scheduler_terminal_failed
        << ",\"terminal_cancelled\":" << counters.scheduler_terminal_cancelled
        << ",\"terminal_releases\":" << counters.scheduler_terminal_releases << "}}";
    return out.str();
}

std::string evidence_json(
        const evidence_counters & baseline,
        const llm_expert_phase8_closeout_witness & closeout,
        const evidence_counters & delta) {
    static const char * lifecycle[] = {
        "empty", "queued_or_staging", "h2d_in_flight", "h2d_complete_unpublished",
        "published", "failed", "cancelled",
    };
    std::ostringstream out;
    out << "\"baseline\":" << counter_groups_json(baseline)
        << ",\"closeout\":{\"written\":" << (closeout.written ? "true" : "false")
        << ",\"write_count\":" << closeout.write_count
        << ",\"final_invariants_ok\":" << (closeout.final_invariants_ok ? "true" : "false")
        << ",\"background\":{\"submitted\":" << closeout.background_submitted
        << ",\"published_completed\":" << closeout.background_published_completed
        << ",\"dropped\":" << closeout.background_dropped
        << ",\"useful\":" << closeout.background_useful
        << ",\"wasted\":" << closeout.background_wasted
        << ",\"busy\":" << closeout.background_busy << ",\"lifecycle\":{";
    for (size_t state = 0; state < closeout.background_lifecycle.size(); ++state) {
        if (state != 0) out << ',';
        out << '"' << lifecycle[state] << "\":" << closeout.background_lifecycle[state];
    }
    out << "}},\"scheduler\":{\"active\":" << closeout.scheduler_active
        << ",\"queued\":" << closeout.scheduler_queued
        << ",\"terminal_complete\":" << closeout.scheduler_terminal_complete
        << ",\"terminal_failed\":" << closeout.scheduler_terminal_failed
        << ",\"terminal_cancelled\":" << closeout.scheduler_terminal_cancelled
        << ",\"terminal_releases\":" << closeout.scheduler_terminal_releases
        << "},\"ring\":{\"queued_workers\":" << closeout.ring_queued_workers
        << ",\"running_workers\":" << closeout.ring_running_workers
        << ",\"non_free_lanes\":" << closeout.ring_non_free_lanes
        << ",\"live_events\":" << closeout.ring_live_events
        << "},\"cold\":{\"hot_refs\":" << closeout.cold_hot_refs
        << ",\"transfer_refs\":" << closeout.cold_transfer_refs
        << ",\"request_refs\":" << closeout.cold_request_refs
        << ",\"cpu_execution_refs\":" << closeout.cold_cpu_execution_refs
        << "},\"hot_pins\":" << closeout.hot_pins
        << ",\"published_forward_mappings\":" << closeout.published_forward_mappings
        << "},\"observed_delta\":" << counter_groups_json(delta);
    return out.str();
}

std::string run_frozen_failure_case(
        const char * name,
        llm_expert_phase8_test_fault fault) {
    provider_fixture fixture;
    cuda_gate gate(fixture);
    GGML_ASSERT(fixture.control.arm_gate(
        llm_expert_phase8_test_gate::auto_after_decision_before_same_key_join,
        { 0, 0 }, 1).is_ready());
    const auto baseline = fixture.capture();
    GGML_ASSERT(fixture.remap(0).is_ready());
    fixture.control.wait_until_observed(llm_expert_phase8_test_gate::background_h2d_in_flight);
    fixture.select_auto_gpu();
    const auto before = fixture.provider->hot_cache_diagnostics();
    auto future = fixture.remap_async(0);
    fixture.control.wait_until_reached();
    GGML_ASSERT(fixture.control.release_gate_and_arm_fault(fault).is_ready());
    if (fault != llm_expert_phase8_test_fault::publication_failed) {
        fixture.control.wait_until_reached();
    }
    gate.release();
    if (fault == llm_expert_phase8_test_fault::publication_failed) {
        fixture.control.wait_until_reached();
    }
    const auto result = future.get();
    GGML_ASSERT(!result.is_ready());
    const auto after = fixture.provider->hot_cache_diagnostics();
    const auto & record = only_last_decision(after, before.auto_decision_records);
    GGML_ASSERT(record.backend == uint8_t(llm_expert_execution_backend::gpu));
    GGML_ASSERT(after.cold_current_cpu_execution_refs == 0);
    GGML_ASSERT(after.cpu_execution_lanes == before.cpu_execution_lanes);
    fixture.destroy();
    const auto delta = complete_evidence(baseline, fixture.witness);
    assert_failed_flight(delta);
    const char * terminal = terminal_from_delta(delta);
    std::ostringstream fields;
    fields << "\"request_status\":\"" << terminal << "\",\"terminal\":\"" << terminal
        << "\",\"decision_count\":1,\"backend\":\"gpu\""
        << ",\"new_cpu_execution_refs\":0,\"active_cpu_lanes\":0,\"written_cpu_ids\":0"
        << ",\"evaluator_invocations\":1,\"backend_switches\":0,\"record\":"
        << auto_record_json(record) << "," << evidence_json(baseline, fixture.witness, delta);
    return case_json(name, fields.str());
}

std::string run_failed_normalization_case(
        const char * name,
        llm_expert_phase8_test_fault fault) {
    provider_fixture fixture;
    std::unique_ptr<cuda_gate> gate;
    if (fault == llm_expert_phase8_test_fault::stale_generation ||
        fault == llm_expert_phase8_test_fault::metadata_mismatch) {
        gate = std::make_unique<cuda_gate>(fixture);
    }
    GGML_ASSERT(fixture.control.arm_fault(fault, { 0, 0 }, 1).is_ready());
    const auto baseline = fixture.capture();
    GGML_ASSERT(fixture.remap(0).is_ready());
    if (gate) {
        fixture.control.wait_until_observed(llm_expert_phase8_test_gate::background_h2d_in_flight);
        if (fault == llm_expert_phase8_test_fault::metadata_mismatch) {
            gate->release();
            fixture.control.wait_until_observed(
                llm_expert_phase8_test_gate::background_h2d_complete_before_provider_publication);
        }
    } else {
        fixture.control.wait_until_reached();
    }
    fixture.select_auto_gpu();
    const auto before = fixture.provider->hot_cache_diagnostics();
    auto future = fixture.remap_async(0);
    if (fault == llm_expert_phase8_test_fault::stale_generation) {
        fixture.control.wait_until_reached();
        gate->release();
    }
    GGML_ASSERT(future.get().is_ready());
    const auto after = fixture.provider->hot_cache_diagnostics();
    const auto & record = only_last_decision(after, before.auto_decision_records);
    GGML_ASSERT(!record.same_key_h2d_present && record.same_key_h2d_state == 0);
    GGML_ASSERT(after.background_dropped == before.background_dropped + 1);
    fixture.destroy();
    const auto delta = complete_evidence(baseline, fixture.witness);
    assert_failed_flight(delta);
    const char * terminal = terminal_from_delta(delta);
    std::ostringstream fields;
    fields << "\"terminalized_before_evaluation\":true,\"same_key_h2d_present\":false"
        << ",\"terminal\":\"" << terminal << "\",\"record\":" << auto_record_json(record)
        << "," << evidence_json(baseline, fixture.witness, delta);
    return case_json(name, fields.str());
}

std::string run_unload_case(
        const char * name,
        llm_expert_phase8_test_gate gate_kind) {
    auto fixture = std::make_unique<provider_fixture>();
    GGML_ASSERT(fixture->control.arm_gate(gate_kind, { 0, 0 }, 1).is_ready());
    const auto baseline = fixture->capture();
    GGML_ASSERT(fixture->remap(0).is_ready());
    fixture->control.wait_until_reached();
    fixture->install_closeout();
    auto future = std::async(std::launch::async, [&] { fixture->provider.reset(); });
    GGML_ASSERT(fixture->control.release_gate().is_ready());
    future.get();
    GGML_ASSERT(fixture->witness.written && fixture->witness.final_invariants_ok);
    const auto delta = complete_evidence(baseline, fixture->witness);
    assert_failed_flight(delta);
    const char * terminal = terminal_from_delta(delta);
    std::ostringstream fields;
    fields << "\"terminal\":\"" << terminal << "\","
        << evidence_json(baseline, fixture->witness, delta);
    return case_json(name, fields.str());
}

std::string run_accounting_case(const char * name, bool consume) {
    provider_fixture fixture;
    GGML_ASSERT(fixture.control.arm_gate(
        llm_expert_phase8_test_gate::background_before_provider_publication,
        { 0, 0 }, 1).is_ready());
    const auto baseline = fixture.capture();
    GGML_ASSERT(fixture.remap(0).is_ready());
    fixture.control.wait_until_observed(
        llm_expert_phase8_test_gate::background_h2d_complete_before_provider_publication);
    if (!consume) {
        GGML_ASSERT(fixture.provider->debug_set_miss_policy_for_testing(
            LLAMA_EXPERT_MISS_POLICY_PROMOTE_AND_GPU).is_ready());
    }
    auto future = fixture.remap_async(consume ? 0 : 1);
    fixture.control.wait_until_reached();
    GGML_ASSERT(fixture.control.release_gate().is_ready());
    GGML_ASSERT(future.get().is_ready());
    const auto before_destroy = fixture.provider->hot_cache_diagnostics();
    if (consume) GGML_ASSERT(before_destroy.background_useful == 1);
    fixture.destroy();
    if (consume) {
        GGML_ASSERT(fixture.witness.background_useful == 1 && fixture.witness.background_wasted == 0);
    } else GGML_ASSERT(fixture.witness.background_useful == 0 && fixture.witness.background_wasted == 1);
    const auto delta = complete_evidence(baseline, fixture.witness);
    assert_published_flight(delta, consume ? 1 : 0, consume ? 0 : 1);
    std::ostringstream fields;
    fields << "\"useful_delta\":" << delta.background_useful
        << ",\"wasted_delta\":" << delta.background_wasted << ","
        << evidence_json(baseline, fixture.witness, delta);
    return case_json(name, fields.str());
}

std::string run_current_output_nonblocking_case() {
    provider_fixture fixture;
    GGML_ASSERT(fixture.control.arm_gate(
        llm_expert_phase8_test_gate::background_queued_before_stage, { 0, 0 }, 1).is_ready());
    const auto baseline = fixture.capture();
    const auto current = fixture.remap(0);
    GGML_ASSERT(current.is_ready());
    fixture.control.wait_until_reached();
    const auto held = fixture.provider->hot_cache_diagnostics();
    GGML_ASSERT(held.active_background_flights == 1 && fixture.control.gate_reached());
    fixture.select_auto_gpu();
    const auto before = fixture.provider->hot_cache_diagnostics().auto_decision_records;
    auto later = fixture.remap_async(0);
    fixture.control.wait_until_observed(
        llm_expert_phase8_test_gate::auto_after_decision_before_same_key_join);
    GGML_ASSERT(fixture.control.release_gate().is_ready());
    GGML_ASSERT(later.get().is_ready());
    const auto joined = fixture.provider->hot_cache_diagnostics();
    GGML_ASSERT(joined.background_later_joins == 1 && joined.active_background_flights == 0);
    const auto & record = only_last_decision(joined, before);
    fixture.destroy();
    const auto delta = complete_evidence(baseline, fixture.witness);
    assert_published_flight(delta, 1, 0);
    return case_json("current_output_nonblocking",
        "\"current_remap_returned_while_gate_closed\":true,\"later_join_same_scheduler\":true,"
        "\"later_join_same_lane\":true,\"later_join_same_hot_generation\":true,\"record\":" +
        auto_record_json(record) + ",\"useful_delta\":" + std::to_string(delta.background_useful) +
        ",\"wasted_delta\":" + std::to_string(delta.background_wasted) + "," +
        evidence_json(baseline, fixture.witness, delta));
}

struct model_smoke_result {
    std::string json;
    llm_expert_key gated_key = { -1, -1 };
    llm_expert_key reap_key = { -1, -1 };
};

struct model_route_capture {
    std::vector<llm_expert_key> keys;
};

bool capture_model_route(
        const llama_route_observation * observation,
        void * user_data) {
    auto & capture = *static_cast<model_route_capture *>(user_data);
    for (uint32_t token = 0; token < observation->n_tokens; ++token) {
        for (uint32_t rank = 0; rank < observation->n_expert_used; ++rank) {
            capture.keys.push_back({
                observation->layer,
                observation->selected_experts[token*observation->n_expert_used + rank],
            });
        }
    }
    return true;
}

llama_model_params model_params(bool background) {
    static const llama_model_tensor_buft_override overrides[] = {
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
    params.expert_background_promotion = background;
    return params;
}

llama_context_params model_context_params() {
    auto params = llama_context_default_params();
    params.n_ctx = 64;
    params.n_batch = 64;
    params.n_ubatch = 1;
    return params;
}

void wait_for_model_background_idle(
        llm_expert_weight_provider * provider,
        llama_model * model) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (std::chrono::steady_clock::now() < deadline) {
        const auto background = provider->hot_cache_diagnostics();
        const auto scheduler = model->expert_scheduler_diagnostics();
        if (background.active_background_flights == 0 &&
            scheduler.active_requests == 0 && scheduler.queued_requests == 0) {
            return;
        }
        std::this_thread::yield();
    }
    GGML_ABORT("timed out waiting for model background work to become idle");
}

void decode_model_token(llama_model * model) {
    llama_context_ptr context(llama_init_from_model(model, model_context_params()));
    GGML_ASSERT(context);
    llama_token token = 1;
    GGML_ASSERT(llama_decode(context.get(), llama_batch_get_one(&token, 1)) == 0);
    llama_synchronize(context.get());
}

model_smoke_result run_model_smoke(const char * name, const std::string & path) {
    llama_model_ptr model(llama_model_load_from_file(path.c_str(), model_params(false)));
    GGML_ASSERT(model);
    auto * provider = model->expert_weight_provider();
    GGML_ASSERT(provider);
    llama_context_ptr context(llama_init_from_model(model.get(), model_context_params()));
    GGML_ASSERT(context);
    model_route_capture routes;
    GGML_ASSERT(llama_set_route_observer(context.get(), capture_model_route, &routes) ==
        LLAMA_ROUTE_OBSERVER_STATUS_OK);
    GGML_ASSERT(llama_route_observer_begin(context.get(), 1, LLAMA_ROUTE_PHASE_DECODE) ==
        LLAMA_ROUTE_OBSERVER_STATUS_OK);
    const auto baseline = capture_evidence(provider, model->expert_scheduler_diagnostics());
    llama_token token = 1;
    GGML_ASSERT(llama_decode(context.get(), llama_batch_get_one(&token, 1)) == 0);
    llama_synchronize(context.get());
    const auto diagnostics = provider->hot_cache_diagnostics();
    GGML_ASSERT(diagnostics.cpu_execution_lanes > 0 && diagnostics.cold_current_cpu_execution_refs == 0);
    GGML_ASSERT(diagnostics.last_remap_layer >= 0 && !diagnostics.last_logical_ids.empty());
    GGML_ASSERT(routes.keys.size() > 1);
    const llm_expert_key key = routes.keys.back();
    GGML_ASSERT(key.expert >= 0);
    context.reset();
    llm_expert_phase8_closeout_witness witness;
    GGML_ASSERT(provider->set_phase8_closeout_witness_for_testing(&witness).is_ready());
    model.reset();
    GGML_ASSERT(witness.written && witness.write_count == 1 && witness.final_invariants_ok);
    const auto delta = complete_evidence(baseline, witness);
    std::ostringstream fields;
    fields << "\"model_path\":\"" << path << "\",\"decode_status\":0"
        << ",\"cpu_execution_lanes_positive\":true,\"route_key\":{\"layer\":"
        << key.layer << ",\"expert\":" << key.expert << "},"
        << evidence_json(baseline, witness, delta);
    return { case_json(name, fields.str()), key, routes.keys.front() };
}

void reap_model_background(
        llm_expert_weight_provider * provider,
        llama_model * model,
        llm_expert_key key) {
    ggml_init_params params = { ggml_tensor_overhead()*16, nullptr, true };
    ggml_context_ptr context(ggml_init(params));
    GGML_ASSERT(context);
    ggml_tensor * logical_ids = ggml_new_tensor_2d(
        context.get(), GGML_TYPE_I32, model->hparams.n_expert_used, 1);
    const auto & layer = model->layers[key.layer];
    const llm_expert_bundle_descriptor bundle = {
        key.layer,
        int32_t(model->hparams.n_expert),
        llm_expert_projection_descriptor::from(
            layer.ffn_up_exps, layer.ffn_up_exps_b, layer.ffn_up_exps_s),
        llm_expert_projection_descriptor::from(
            layer.ffn_gate_exps, layer.ffn_gate_exps_b, layer.ffn_gate_exps_s),
        llm_expert_projection_descriptor::from(
            layer.ffn_gate_up_exps, layer.ffn_gate_up_exps_b, nullptr),
        llm_expert_projection_descriptor::from(
            layer.ffn_down_exps, layer.ffn_down_exps_b, layer.ffn_down_exps_s),
    };
    const llm_expert_selection selection = {
        key.layer,
        int32_t(model->hparams.n_expert),
        int32_t(model->hparams.n_expert_used),
        1,
        logical_ids,
    };
    llm_expert_graph_binding binding;
    GGML_ASSERT(provider->bind_graph(context.get(), bundle, selection, binding).is_ready());
    GGML_ASSERT(binding.hybrid && !binding.bootstrap && binding.checkpoint_ids != nullptr);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors_from_buft(
        context.get(), ggml_backend_cpu_buffer_type()));
    GGML_ASSERT(buffer);
    const size_t count = size_t(binding.checkpoint_ids->ne[0])*size_t(binding.checkpoint_ids->ne[1]);
    std::vector<int32_t> checkpoint_values(count, key.expert);
    ggml_backend_tensor_set(
        binding.checkpoint_ids, checkpoint_values.data(), 0,
        checkpoint_values.size()*sizeof(checkpoint_values[0]));
    llm_expert_execution_plan plan;
    GGML_ASSERT(provider->prepare({ binding }, plan).is_ready());
    GGML_ASSERT(provider->remap_checkpoint_tensor(binding).is_ready());
    plan.reset();
}

std::string run_model_destroy_case(
        const char * name,
        const std::string & path,
        llm_expert_key key,
        llm_expert_key reap_key,
        llm_expert_phase8_test_gate gate) {
    llama_model_ptr model(llama_model_load_from_file(path.c_str(), model_params(true)));
    GGML_ASSERT(model);
    auto * provider = model->expert_weight_provider();
    GGML_ASSERT(provider);
    llm_expert_phase8_test_control control;
    llm_expert_phase8_closeout_witness witness;
    llama_context_ptr context(llama_init_from_model(model.get(), model_context_params()));
    GGML_ASSERT(context);
    GGML_ASSERT(provider->set_phase8_test_control_for_testing(&control).is_ready());
    GGML_ASSERT(provider->set_phase8_closeout_witness_for_testing(&witness).is_ready());

    // Prime all non-target generations and classify them as useful before the
    // measured window. The target is deliberately failed in both warm passes,
    // leaving exactly one production-path retry for the destruction case.
    for (uint64_t generation = 1; generation <= 2; ++generation) {
        GGML_ASSERT(control.arm_fault(
            llm_expert_phase8_test_fault::h2d_failed, key, generation).is_ready());
        decode_model_token(model.get());
        control.wait_until_reached();
    }
    reap_model_background(provider, model.get(), reap_key);
    wait_for_model_background_idle(provider, model.get());
    GGML_ASSERT(!provider->debug_hot_mapping(key));
    const auto primed = provider->hot_cache_diagnostics();
    GGML_ASSERT(primed.background_completed > 0 &&
        primed.background_useful == primed.background_completed &&
        primed.background_wasted == 0 && primed.active_background_flights == 0);

    GGML_ASSERT(control.arm_gate(gate, key, 3).is_ready());
    const auto baseline = capture_evidence(provider, model->expert_scheduler_diagnostics());
    llama_token token = 1;
    GGML_ASSERT(llama_decode(context.get(), llama_batch_get_one(&token, 1)) == 0);
    llama_synchronize(context.get());
    control.wait_until_reached();
    context.reset();
    auto destroy = std::async(std::launch::async, [&] { model.reset(); });
    GGML_ASSERT(control.release_gate().is_ready());
    destroy.get();
    GGML_ASSERT(witness.written && witness.write_count == 1 && witness.final_invariants_ok);
    GGML_ASSERT(witness.scheduler_active == 0 && witness.scheduler_queued == 0 &&
        witness.ring_queued_workers == 0 && witness.ring_running_workers == 0 &&
        witness.ring_non_free_lanes == 0 && witness.ring_live_events == 0);
    GGML_ASSERT(witness.cold_hot_refs == 0 && witness.cold_transfer_refs == 0 &&
        witness.cold_request_refs == 0 && witness.cold_cpu_execution_refs == 0);
    const auto delta = complete_evidence(baseline, witness);
    assert_failed_flight(delta);
    const char * terminal = terminal_from_delta(delta);
    std::ostringstream fields;
    fields << "\"model_path\":\"" << path << "\",\"gate_reached_before_model_reset\":true"
        << ",\"terminal\":\"" << terminal << "\"," << evidence_json(baseline, witness, delta);
    return case_json(name, fields.str());
}

} // namespace

int main(int argc, char ** argv) {
    ggml_backend_load_all();
    std::string output_path;
    std::string outer_head;
    std::string nested_head;
    std::string f16_path;
    std::string mxfp4_path;
    for (int index = 1; index < argc; ++index) {
        const std::string arg = argv[index];
        if (arg == "--output" && index + 1 < argc) output_path = argv[++index];
        else if (arg == "--outer-head" && index + 1 < argc) outer_head = argv[++index];
        else if (arg == "--nested-head" && index + 1 < argc) nested_head = argv[++index];
        else if (arg == "--f16" && index + 1 < argc) f16_path = argv[++index];
        else if (arg == "--mxfp4" && index + 1 < argc) mxfp4_path = argv[++index];
    }
    if (output_path.empty() || outer_head.empty() || nested_head.empty() || f16_path.empty() || mxfp4_path.empty() ||
        ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU) == nullptr) {
        std::fprintf(stderr, "usage: %s --output PATH --outer-head SHA --nested-head SHA --f16 GGUF --mxfp4 GGUF (CUDA required)\n", argv[0]);
        return 2;
    }

    std::vector<std::string> cases;
    std::fprintf(stderr, "case auto_operands_none\n");
    cases.push_back(run_none_case());
    std::fprintf(stderr, "case auto_operands_queued\n");
    cases.push_back(run_live_operand_case("auto_operands_queued",
        llm_expert_phase8_test_gate::background_queued_before_stage, 1));
    std::fprintf(stderr, "case auto_operands_in_flight\n");
    cases.push_back(run_live_operand_case("auto_operands_in_flight",
        llm_expert_phase8_test_gate::background_h2d_in_flight, 2));
    std::fprintf(stderr, "case auto_operands_complete_unpublished\n");
    cases.push_back(run_complete_operand_case());
    std::fprintf(stderr, "case frozen_gpu_scheduler_join_mismatch\n");
    cases.push_back(run_frozen_failure_case("frozen_gpu_scheduler_join_mismatch",
        llm_expert_phase8_test_fault::scheduler_join_mismatch));
    std::fprintf(stderr, "case frozen_gpu_wait_failed\n");
    cases.push_back(run_frozen_failure_case("frozen_gpu_wait_failed",
        llm_expert_phase8_test_fault::wait_failed));
    std::fprintf(stderr, "case frozen_gpu_publication_failed\n");
    cases.push_back(run_frozen_failure_case("frozen_gpu_publication_failed",
        llm_expert_phase8_test_fault::publication_failed));
    std::fprintf(stderr, "case normalization_failed\n");
    cases.push_back(run_failed_normalization_case("normalization_failed",
        llm_expert_phase8_test_fault::stage_copy_failed));
    std::fprintf(stderr, "case normalization_stale\n");
    cases.push_back(run_failed_normalization_case("normalization_stale",
        llm_expert_phase8_test_fault::stale_generation));
    std::fprintf(stderr, "case prepublication_h2d_failure\n");
    cases.push_back(run_failed_normalization_case("prepublication_h2d_failure",
        llm_expert_phase8_test_fault::h2d_failed));
    std::fprintf(stderr, "case prepublication_metadata_invalid\n");
    cases.push_back(run_failed_normalization_case("prepublication_metadata_invalid",
        llm_expert_phase8_test_fault::metadata_mismatch));
    std::fprintf(stderr, "case prepublication_unload_queued\n");
    cases.push_back(run_unload_case("prepublication_unload_queued",
        llm_expert_phase8_test_gate::background_queued_before_stage));
    std::fprintf(stderr, "case prepublication_unload_in_flight\n");
    cases.push_back(run_unload_case("prepublication_unload_in_flight",
        llm_expert_phase8_test_gate::background_h2d_in_flight));
    std::fprintf(stderr, "case prepublication_unload_complete_unpublished\n");
    cases.push_back(run_unload_case("prepublication_unload_complete_unpublished",
        llm_expert_phase8_test_gate::background_h2d_complete_before_provider_publication));
    std::fprintf(stderr, "case current_output_nonblocking\n");
    cases.push_back(run_current_output_nonblocking_case());
    std::fprintf(stderr, "case published_wasted\n");
    cases.push_back(run_accounting_case("published_wasted", false));
    std::fprintf(stderr, "case published_useful\n");
    cases.push_back(run_accounting_case("published_useful", true));
    std::fprintf(stderr, "case model_f16_smoke\n");
    const auto f16 = run_model_smoke("model_f16_smoke", f16_path);
    cases.push_back(f16.json);
    std::fprintf(stderr, "case model_mxfp4_smoke\n");
    const auto mxfp4 = run_model_smoke("model_mxfp4_smoke", mxfp4_path);
    cases.push_back(mxfp4.json);
    for (const auto & entry : std::array<std::pair<const char *, llm_expert_phase8_test_gate>, 3> {{
            { "queued", llm_expert_phase8_test_gate::background_queued_before_stage },
            { "in_flight", llm_expert_phase8_test_gate::background_h2d_in_flight },
            { "complete_unpublished", llm_expert_phase8_test_gate::background_h2d_complete_before_provider_publication },
        }}) {
        const std::string f16_name = std::string("model_f16_destroy_") + entry.first;
        std::fprintf(stderr, "case %s\n", f16_name.c_str());
        cases.push_back(run_model_destroy_case(
            f16_name.c_str(), f16_path, f16.gated_key, f16.reap_key, entry.second));
        const std::string mxfp4_name = std::string("model_mxfp4_destroy_") + entry.first;
        std::fprintf(stderr, "case %s\n", mxfp4_name.c_str());
        cases.push_back(run_model_destroy_case(
            mxfp4_name.c_str(), mxfp4_path, mxfp4.gated_key, mxfp4.reap_key, entry.second));
    }

    std::ostringstream json;
    json << "{\n  \"schema\":\"phase8-checkpoint-b-probe-v2\",\n"
         << "  \"outer_head\":\"" << outer_head << "\",\n"
         << "  \"nested_head\":\"" << nested_head << "\",\n"
         << "  \"cwd\":" << json_string(std::filesystem::current_path().string()) << ",\n"
         << "  \"exit_code\":0,\n  \"command\":[";
    for (int index = 0; index < argc; ++index) {
        if (index != 0) json << ',';
        json << json_string(argv[index]);
    }
    json << "],\n  \"models\":{\"f16\":{\"path\":" << json_string(f16_path)
         << "},\"mxfp4\":{\"path\":" << json_string(mxfp4_path) << "}},\n"
         << "  \"cases\":{";
    for (size_t index = 0; index < cases.size(); ++index) {
        json << (index == 0 ? "\n    " : ",\n    ") << cases[index];
    }
    json << "\n  }\n}\n";
    std::ofstream output(output_path, std::ios::binary | std::ios::trunc);
    if (!output) return 3;
    output << json.str();
    output.close();
    if (!output) return 4;
    std::printf("PHASE8_CHECKPOINT_B_PROBE status=pass cases=%zu output=%s\n",
        cases.size(), output_path.c_str());
    return 0;
}
