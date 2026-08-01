#pragma once

#include "llama.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

enum class llm_expert_prefetch_error : uint8_t {
    none,
    invalid_configuration,
    invalid_profile,
    profile_too_large,
    profile_io,
    profile_identity_mismatch,
    overflow,
    transcript_full,
};

struct llm_expert_prefetch_result {
    llm_expert_prefetch_error error = llm_expert_prefetch_error::none;
    bool is_ready() const noexcept { return error == llm_expert_prefetch_error::none; }
    static llm_expert_prefetch_result success() noexcept { return {}; }
    static llm_expert_prefetch_result failure(llm_expert_prefetch_error error) noexcept { return { error }; }
};

struct llm_expert_prefetch_config_internal {
    bool supplied = false;
    llama_expert_prefetch_config_v1 value = {};
    uint64_t digest = 0;
};

llm_expert_prefetch_result llm_expert_prefetch_copy_config(
        const llama_expert_prefetch_config_v1 * source,
        const char * profile_path,
        llama_expert_weights_mode mode,
        uint32_t experts_per_layer,
        uint32_t hot_capacity,
        uint64_t cold_capacity_bytes,
        uint32_t scheduler_capacity,
        uint64_t storage_capacity_bytes,
        uint64_t h2d_capacity_bytes,
        llm_expert_prefetch_config_internal & destination) noexcept;

struct llm_expert_prefetch_file_identity {
    uint32_t ordinal = 0;
    std::string name;
    uint64_t size = 0;
    std::string sha256;
};

struct llm_expert_prefetch_key_bytes {
    int32_t layer = -1;
    int32_t expert = -1;
    uint64_t payload_bytes = 0;
    uint64_t physical_bytes = 0;
};

struct llm_expert_prefetch_fingerprint {
    std::string package_sha256;
    std::vector<llm_expert_prefetch_file_identity> files;
    uint32_t layer_count = 0;
    std::vector<int32_t> routed_layers;
    uint32_t experts_per_layer = 0;
    uint32_t experts_per_token = 0;
    std::string tensor_layout_sha256;
    std::vector<llm_expert_prefetch_key_bytes> expert_bytes;
};

struct llm_expert_prefetch_count {
    int32_t layer = -1;
    int32_t expert = -1;
    uint64_t count = 0;
};

struct llm_expert_prefetch_transition {
    int32_t source_layer = -1;
    int32_t source_expert = -1;
    int32_t target_layer = -1;
    int32_t target_expert = -1;
    uint64_t count = 0;
};

struct llm_expert_prefetch_cost {
    std::string transport;
    llama_expert_prefetch_readiness readiness = LLAMA_EXPERT_PREFETCH_READINESS_HOST_READY;
    uint64_t lead_ns = 0;
    uint64_t demand_service_ns = 0;
    uint64_t speculative_service_ns = 0;
    uint64_t predictor_compute_ns = 0;
    uint64_t scheduler_demand_delay_ns = 0;
    uint64_t displacement_refill_ns = 0;
    uint64_t storage_bytes = 0;
    uint64_t h2d_bytes = 0;
    uint32_t break_even_bps = 0;
    uint32_t utility_window_predictions = 0;
    uint32_t utility_min_observations = 0;
    uint32_t utility_min_timely_successes = 0;
};

struct llm_expert_prefetch_seed {
    int32_t layer = -1;
    int32_t expert = -1;
    uint64_t count = 0;
    uint64_t payload_bytes = 0;
    uint64_t physical_bytes = 0;
};

struct llm_expert_prefetch_profile {
    std::string profile_id;
    std::string source_kind;
    uint32_t fold_index = 0;
    std::vector<std::string> training_prompts;
    std::string validation_prompt;
    std::string test_prompt;
    uint64_t training_rows = 0;
    uint64_t validation_rows = 0;
    uint64_t test_rows = 0;
    llm_expert_prefetch_fingerprint target;
    std::vector<llm_expert_prefetch_count> static_counts;
    std::vector<llm_expert_prefetch_transition> transitions;
    std::vector<llm_expert_prefetch_cost> costs;
    std::vector<llm_expert_prefetch_seed> seed;
    uint32_t selected_candidates = 0;
    uint32_t selected_temporal_window = 0;
    std::string selected_policy;
    std::string selected_transport;
    llama_expert_prefetch_readiness selected_readiness = LLAMA_EXPERT_PREFETCH_READINESS_HOST_READY;
    uint32_t selected_break_even_bps = 0;
    std::string profile_sha256;
    uint64_t profile_bytes = 0;
};

llm_expert_prefetch_result llm_expert_prefetch_load_profile(
        const std::string & path,
        uint64_t max_profile_bytes,
        const llm_expert_prefetch_fingerprint * expected,
        llm_expert_prefetch_profile & profile,
        std::string & error) noexcept;

struct llm_expert_prefetch_key {
    int32_t layer = -1;
    int32_t expert = -1;
};

enum class llm_expert_prefetch_trigger : uint8_t {
    token_end,
    router_result,
};

struct llm_expert_prefetch_candidate {
    llm_expert_prefetch_key key;
    uint32_t rank = 0;
    uint64_t score = 0;
};

struct llm_expert_prefetch_request_state {
    uint64_t request_ordinal = 0;
    uint64_t completed_tokens = 0;
    uint64_t digest = 0;
    std::vector<std::vector<std::vector<int32_t>>> token_history;
};

class llm_expert_prefetch_predictor {
public:
    llm_expert_prefetch_result initialize(
            const llm_expert_prefetch_profile & profile,
            const llm_expert_prefetch_config_internal & config) noexcept;
    llm_expert_prefetch_result request_begin(uint64_t request_ordinal) noexcept;
    llm_expert_prefetch_result commit_token(
            uint64_t token_ordinal,
            const std::vector<std::vector<int32_t>> & routed_experts) noexcept;
    llm_expert_prefetch_result predict_token_end(
            uint64_t token_ordinal,
            int32_t target_layer,
            std::vector<llm_expert_prefetch_candidate> & candidates) const noexcept;
    llm_expert_prefetch_result predict_cross_layer(
            uint64_t token_ordinal,
            int32_t source_layer,
            const int32_t * source_experts,
            size_t source_count,
            int32_t target_layer,
            std::vector<llm_expert_prefetch_candidate> & candidates) const noexcept;
    const llm_expert_prefetch_request_state & state() const noexcept { return request; }

private:
    const llm_expert_prefetch_profile * profile = nullptr;
    llm_expert_prefetch_config_internal config;
    llm_expert_prefetch_request_state request;
};

llm_expert_prefetch_result llm_expert_prefetch_break_even(
        uint64_t lead_ns,
        uint64_t demand_service_ns,
        uint64_t predictor_compute_ns,
        uint64_t speculative_service_ns,
        uint64_t scheduler_demand_delay_ns,
        uint64_t displacement_refill_ns,
        uint64_t & hidden_benefit_ns,
        uint64_t & waste_cost_ns,
        uint32_t & break_even_bps) noexcept;

std::string llm_expert_prefetch_sha256(const void * data, size_t size);
llm_expert_prefetch_result llm_expert_prefetch_sha256_file(
        const std::string & path,
        uint64_t expected_size,
        std::string & digest) noexcept;
