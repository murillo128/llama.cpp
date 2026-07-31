#include "llama-expert-cache-policy.h"

#include <cinttypes>
#include <cstdio>
#include <cstring>

namespace {

bool ready(llm_expert_cache_policy_result result, const char * step) {
    if (result.is_ready()) return true;
    std::fprintf(stderr, "phase9-cache-replay: %s failed (%u)\n", step, unsigned(result.error));
    return false;
}

llm_expert_cache_policy_candidate candidate(
        uint32_t slot,
        uint64_t generation,
        int32_t expert,
        bool free,
        bool eligible) {
    return { slot, generation, { 0, expert }, 64, 128, free, eligible };
}

int self_test() {
    const llama_expert_cache_policy_config public_config = {
        LLAMA_EXPERT_CACHE_POLICY_VERSION_1,
        sizeof(llama_expert_cache_policy_config),
        LLAMA_EXPERT_CACHE_POLICY_LRU,
        LLAMA_EXPERT_CACHE_POLICY_SCOPE_GLOBAL,
        0,
        LLAMA_EXPERT_CACHE_ADMISSION_ALWAYS,
        0,
        0,
        {},
    };
    llm_expert_cache_policy_config_internal config;
    if (!ready(llm_expert_cache_policy_copy_config(
            &public_config, llm_expert_cache_policy_tier::hot, config), "config")) return 1;

    const int32_t layers[] = { 0 };
    llm_expert_cache_policy policy;
    if (!ready(policy.initialize(config, llm_expert_cache_policy_tier::hot,
            layers, 1, 4, 1, 2, 128, 64), "initialize") ||
        !ready(policy.request_begin(), "request_begin") ||
        !ready(policy.demand({ 0, 0 }, 1, 64, 128), "demand_0") ||
        !ready(policy.load_begin(0, 1, { 0, 0 }, 64, 128), "load_begin_0") ||
        !ready(policy.load_complete(0, 1), "load_complete_0") ||
        !ready(policy.demand({ 0, 1 }, 1, 64, 128), "demand_1") ||
        !ready(policy.load_begin(1, 1, { 0, 1 }, 64, 128), "load_begin_1") ||
        !ready(policy.load_complete(1, 1), "load_complete_1") ||
        !ready(policy.demand({ 0, 0 }, 1, 64, 128), "demand_hit") ||
        !ready(policy.hit(0, 1), "hit_0") ||
        !ready(policy.demand({ 0, 2 }, 1, 64, 128), "demand_2")) return 1;

    const llm_expert_cache_policy_candidate candidates[] = {
        candidate(0, 1, 0, false, true),
        candidate(1, 1, 1, false, true),
    };
    llm_expert_cache_policy_decision decision;
    if (!ready(policy.select({ 0, 2 }, candidates, 2, decision), "select") ||
        decision.slot != 1 || decision.free || !decision.accept ||
        !ready(policy.evict(1, 1), "evict") ||
        !ready(policy.load_begin(1, 2, { 0, 2 }, 64, 128), "load_begin_2") ||
        !ready(policy.load_complete(1, 2), "load_complete_2") ||
        !ready(policy.request_end(true, false), "request_end")) {
        std::fprintf(stderr, "phase9-cache-replay: hand-computed LRU victim mismatch\n");
        return 1;
    }

    const auto & diagnostics = policy.diagnostics();
    const auto & transcript = policy.transcript();
    for (size_t index = 0; index < policy.transcript_size(); ++index) {
        if (transcript[index].event_sequence != index + 1 || transcript[index].state_digest == 0) {
            std::fprintf(stderr, "phase9-cache-replay: noncanonical transcript at %zu\n", index);
            return 1;
        }
    }
    std::printf(
        "{\"schema\":\"cache-policy-replay-v1\",\"policy\":\"LRU\","
        "\"victim_slot\":1,\"events\":%zu,\"operations\":%" PRIu64
        ",\"final_digest\":%" PRIu64 "}\n",
        policy.transcript_size(), diagnostics.operation_ordinal, diagnostics.state_digest);
    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--self-test") == 0) return self_test();
    std::fprintf(stderr, "usage: phase9-cache-replay --self-test\n");
    return 2;
}
