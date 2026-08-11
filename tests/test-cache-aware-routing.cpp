#include "../src/llama-cache-aware-routing.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace {

void require(bool condition) {
    if (!condition) std::abort();
}

struct fixture {
    std::array<int32_t, 16> exact {};
    std::array<int32_t, 20> candidates {};
    std::array<float, 20> scores {};
    std::array<llama_route_service_tier, 20> tiers {};

    fixture() {
        for (int32_t rank = 0; rank < 20; ++rank) {
            candidates[rank] = rank;
            scores[rank] = 1.0f - float(rank)*0.01f;
            tiers[rank] = LLAMA_ROUTE_SERVICE_TIER_HOT;
            if (rank < 16) exact[rank] = rank;
        }
        scores[16] = 0.849f;
        scores[17] = 0.848f;
        scores[18] = 0.847f;
        scores[19] = 0.846f;
        tiers[15] = LLAMA_ROUTE_SERVICE_TIER_BACKING;
        tiers[16] = LLAMA_ROUTE_SERVICE_TIER_COLD;
        tiers[17] = LLAMA_ROUTE_SERVICE_TIER_HOT;
    }
};

llm_cache_aware_routing_result select(
        fixture & value, uint32_t count, uint32_t swaps, float regret,
        std::array<int32_t, 16> & final) {
    return llm_select_cache_aware_route(
        value.exact.data(), value.candidates.data(), value.scores.data(), value.tiers.data(),
        896, 16, count, swaps, regret, final.data());
}

void test_exact_controls() {
    fixture value;
    for (const auto control : std::array<std::array<float, 3>, 3> {{
            {{ 20, 0, 1.0f }}, {{ 16, 16, 1.0f }}, {{ 20, 16, 0.0f }},
        }}) {
        std::array<int32_t, 16> final {};
        const auto result = select(value, uint32_t(control[0]), uint32_t(control[1]), control[2], final);
        require(result.is_ready() && result.swaps == 0 && final == value.exact);
    }
}

void test_bounded_deterministic_selection() {
    fixture value;
    std::array<int32_t, 16> first {};
    std::array<int32_t, 16> second {};
    auto result = select(value, 20, 1, 0.01f, first);
    require(result.is_ready() && result.swaps == 1);
    require(first[15] == 17); // HOT wins before the lower-regret COLD candidate.
    for (uint32_t rank = 0; rank < 15; ++rank) require(first[rank] == value.exact[rank]);
    result = select(value, 20, 1, 0.01f, second);
    require(result.is_ready() && first == second);

    std::array<int32_t, 16> outside {};
    result = select(value, 20, 1, 0.0005f, outside);
    require(result.is_ready() && result.swaps == 0 && outside == value.exact);
}

void test_multi_swap_unique_cardinality() {
    fixture value;
    value.tiers[14] = LLAMA_ROUTE_SERVICE_TIER_BACKING;
    std::array<int32_t, 16> final {};
    const auto result = select(value, 20, 2, 0.02f, final);
    require(result.is_ready() && result.swaps == 2);
    for (uint32_t rank = 0; rank < final.size(); ++rank) {
        require(final[rank] >= 0 && final[rank] < 896);
        for (uint32_t previous = 0; previous < rank; ++previous) {
            require(final[rank] != final[previous]);
        }
    }
}

void test_equal_score_tie_break_preserves_better_selected_rank() {
    fixture value;
    value.tiers[14] = LLAMA_ROUTE_SERVICE_TIER_BACKING;
    value.scores[14] = value.scores[15];
    value.scores[16] = value.scores[15] - 0.001f;
    value.tiers[16] = LLAMA_ROUTE_SERVICE_TIER_HOT;
    value.tiers[17] = LLAMA_ROUTE_SERVICE_TIER_BACKING;
    std::array<int32_t, 16> final {};
    const auto result = select(value, 20, 1, 0.01f, final);
    require(result.is_ready() && result.swaps == 1);
    require(final[14] == 16 && final[15] == value.exact[15]);
}

void test_invalid_inputs_fail_closed() {
    fixture value;
    std::array<int32_t, 16> final {};
    value.candidates[16] = value.candidates[15];
    require(!select(value, 20, 1, 0.1f, final).is_ready());
    value = fixture {};
    value.scores[16] = NAN;
    require(!select(value, 20, 1, 0.1f, final).is_ready());
    value = fixture {};
    require(!select(value, 20, 1, INFINITY, final).is_ready());
    require(!select(value, 20, 17, 0.1f, final).is_ready());
}

} // namespace

int main() {
    test_exact_controls();
    test_bounded_deterministic_selection();
    test_multi_swap_unique_cardinality();
    test_equal_score_tie_break_preserves_better_selected_rank();
    test_invalid_inputs_fail_closed();
    std::puts("cache-aware routing tests passed");
    return 0;
}
