#pragma once

#include "llama.h"

#include <cstddef>
#include <cstdint>

enum class llm_cache_aware_routing_error : uint8_t {
    none,
    invalid_configuration,
    invalid_candidate,
    invalid_tier,
    invalid_result,
};

struct llm_cache_aware_routing_result {
    llm_cache_aware_routing_error error = llm_cache_aware_routing_error::none;
    uint32_t swaps = 0;
    double cumulative_score_regret = 0.0;

    bool is_ready() const noexcept {
        return error == llm_cache_aware_routing_error::none;
    }
};

// Pure, allocation-free selection for one routed layer/token. candidates must
// be a unique, score-ordered exact top-k extension and tiers must describe the
// same contemporaneous snapshot.
llm_cache_aware_routing_result llm_select_cache_aware_route(
        const int32_t * exact_experts,
        const int32_t * candidates,
        const float * selection_scores,
        const llama_route_service_tier * tiers,
        uint32_t n_expert,
        uint32_t n_expert_used,
        uint32_t candidate_count,
        uint32_t max_swaps,
        float max_score_regret,
        int32_t * final_experts) noexcept;
