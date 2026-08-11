#include "llama-cache-aware-routing.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <tuple>

namespace {

constexpr uint32_t MAX_EXPERT_USED = 16;
constexpr uint32_t MAX_CANDIDATES = 64;

struct option {
    uint32_t selected_rank = 0;
    uint32_t candidate_rank = 0;
    int32_t selected_expert = -1;
    int32_t candidate_expert = -1;
    uint8_t improvement = 0;
    float regret = 0.0f;
};

bool valid_tier(llama_route_service_tier tier) {
    return tier == LLAMA_ROUTE_SERVICE_TIER_HOT ||
        tier == LLAMA_ROUTE_SERVICE_TIER_COLD ||
        tier == LLAMA_ROUTE_SERVICE_TIER_BACKING;
}

} // namespace

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
        int32_t * final_experts) noexcept {
    llm_cache_aware_routing_result result;
    if (exact_experts == nullptr || candidates == nullptr || selection_scores == nullptr ||
        tiers == nullptr || final_experts == nullptr || n_expert == 0 || n_expert_used == 0 ||
        n_expert_used > MAX_EXPERT_USED || candidate_count < n_expert_used ||
        candidate_count > std::min<uint32_t>(MAX_CANDIDATES, n_expert) ||
        max_swaps > n_expert_used || !std::isfinite(max_score_regret) || max_score_regret < 0.0f) {
        result.error = llm_cache_aware_routing_error::invalid_configuration;
        return result;
    }

    for (uint32_t rank = 0; rank < candidate_count; ++rank) {
        if (candidates[rank] < 0 || uint32_t(candidates[rank]) >= n_expert ||
            !std::isfinite(selection_scores[rank]) || !valid_tier(tiers[rank]) ||
            (rank < n_expert_used && exact_experts[rank] != candidates[rank]) ||
            (rank > 0 && selection_scores[rank] > selection_scores[rank - 1])) {
            result.error = llm_cache_aware_routing_error::invalid_candidate;
            return result;
        }
        for (uint32_t previous = 0; previous < rank; ++previous) {
            if (candidates[previous] == candidates[rank]) {
                result.error = llm_cache_aware_routing_error::invalid_candidate;
                return result;
            }
        }
    }
    std::copy(exact_experts, exact_experts + n_expert_used, final_experts);
    if (max_score_regret == 0.0f || max_swaps == 0 || candidate_count == n_expert_used) {
        return result;
    }

    std::array<option, MAX_EXPERT_USED*MAX_CANDIDATES> options {};
    size_t option_count = 0;
    for (uint32_t selected_rank = 0; selected_rank < n_expert_used; ++selected_rank) {
        const uint8_t selected_cost = uint8_t(tiers[selected_rank]);
        for (uint32_t candidate_rank = n_expert_used; candidate_rank < candidate_count; ++candidate_rank) {
            const uint8_t candidate_cost = uint8_t(tiers[candidate_rank]);
            const float regret = selection_scores[selected_rank] - selection_scores[candidate_rank];
            if (candidate_cost >= selected_cost || regret < 0.0f || regret > max_score_regret) {
                continue;
            }
            options[option_count++] = {
                selected_rank,
                candidate_rank,
                exact_experts[selected_rank],
                candidates[candidate_rank],
                uint8_t(selected_cost - candidate_cost),
                regret,
            };
        }
    }
    std::sort(options.begin(), options.begin() + option_count, [](const option & lhs, const option & rhs) {
        return std::tuple(-int(lhs.improvement), lhs.regret, lhs.candidate_rank,
                   lhs.selected_rank, lhs.candidate_expert, lhs.selected_expert) <
               std::tuple(-int(rhs.improvement), rhs.regret, rhs.candidate_rank,
                   rhs.selected_rank, rhs.candidate_expert, rhs.selected_expert);
    });

    std::array<bool, MAX_EXPERT_USED> used_slots {};
    std::array<bool, MAX_CANDIDATES> used_candidates {};
    for (size_t index = 0; index < option_count && result.swaps < max_swaps; ++index) {
        const auto & candidate = options[index];
        if (used_slots[candidate.selected_rank] || used_candidates[candidate.candidate_rank]) {
            continue;
        }
        final_experts[candidate.selected_rank] = candidate.candidate_expert;
        used_slots[candidate.selected_rank] = true;
        used_candidates[candidate.candidate_rank] = true;
        result.swaps++;
        result.cumulative_score_regret += candidate.regret;
    }

    for (uint32_t rank = 0; rank < n_expert_used; ++rank) {
        if (final_experts[rank] < 0 || uint32_t(final_experts[rank]) >= n_expert) {
            result.error = llm_cache_aware_routing_error::invalid_result;
            return result;
        }
        for (uint32_t previous = 0; previous < rank; ++previous) {
            if (final_experts[previous] == final_experts[rank]) {
                result.error = llm_cache_aware_routing_error::invalid_result;
                return result;
            }
        }
    }
    return result;
}
