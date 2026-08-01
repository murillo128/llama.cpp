#include "llama-expert-cache-policy.h"

#include <cstdio>
#include <cstdlib>

namespace {

void require(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "FAILED: %s\n", message);
        std::exit(1);
    }
}

llm_expert_cache_policy_config_internal make_config(
        llama_expert_cache_policy policy,
        llama_expert_cache_policy_scope scope = LLAMA_EXPERT_CACHE_POLICY_SCOPE_GLOBAL,
        uint32_t ratio = UINT32_MAX,
        llama_expert_cache_admission admission = LLAMA_EXPERT_CACHE_ADMISSION_ALWAYS,
        uint32_t window = UINT32_MAX,
        uint32_t aging = UINT32_MAX) {
    if (ratio == UINT32_MAX) {
        ratio = policy == LLAMA_EXPERT_CACHE_POLICY_SLRU ?
            LLAMA_EXPERT_CACHE_POLICY_SLRU_PROTECTED_RATIO_DEFAULT_BPS : 0;
    }
    if (window == UINT32_MAX) {
        window = admission == LLAMA_EXPERT_CACHE_ADMISSION_FREQUENCY_WINDOW ?
            LLAMA_EXPERT_CACHE_POLICY_FREQUENCY_WINDOW_DEFAULT_EVENTS : 0;
    }
    if (aging == UINT32_MAX) {
        aging = policy == LLAMA_EXPERT_CACHE_POLICY_LFU_AGING ?
            LLAMA_EXPERT_CACHE_POLICY_LFU_AGING_DEFAULT_EVENTS : 0;
    }
    const llama_expert_cache_policy_config source = {
        LLAMA_EXPERT_CACHE_POLICY_VERSION_1,
        sizeof(llama_expert_cache_policy_config),
        policy,
        scope,
        ratio,
        admission,
        window,
        aging,
        {},
    };
    llm_expert_cache_policy_config_internal result;
    require(llm_expert_cache_policy_copy_config(
        &source, llm_expert_cache_policy_tier::hot, result).is_ready(), "copy policy config");
    return result;
}

llm_expert_cache_policy_candidate candidate(
        uint32_t slot,
        uint64_t generation,
        int32_t layer,
        int32_t expert,
        bool free,
        bool eligible) {
    return { slot, generation, { layer, expert }, 64, 128, free, eligible };
}

void begin(llm_expert_cache_policy & policy) {
    require(policy.request_begin().is_ready(), "request begin");
}

void demand(llm_expert_cache_policy & policy, int32_t layer, int32_t expert) {
    require(policy.demand({ layer, expert }, 1, 64, 128).is_ready(), "demand");
}

void load(llm_expert_cache_policy & policy, uint32_t slot, uint64_t generation, int32_t layer, int32_t expert) {
    require(policy.load_begin(slot, generation, { layer, expert }, 64, 128).is_ready(), "load begin");
    require(policy.load_complete(slot, generation).is_ready(), "load complete");
}

void test_config() {
    llm_expert_cache_policy_config_internal first;
    llm_expert_cache_policy_config_internal second;
    require(llm_expert_cache_policy_copy_config(
        nullptr, llm_expert_cache_policy_tier::hot, first).is_ready(), "null config resolves");
    require(llm_expert_cache_policy_copy_config(
        nullptr, llm_expert_cache_policy_tier::hot, second).is_ready(), "null config repeats");
    require(first.policy == LLAMA_EXPERT_CACHE_POLICY_LRU &&
            first.scope == LLAMA_EXPERT_CACHE_POLICY_SCOPE_GLOBAL && first.digest == second.digest,
        "null config is deterministic global LRU");

    llama_expert_cache_policy_config invalid = {
        LLAMA_EXPERT_CACHE_POLICY_VERSION_1,
        sizeof(llama_expert_cache_policy_config),
        LLAMA_EXPERT_CACHE_POLICY_LRU,
        LLAMA_EXPERT_CACHE_POLICY_SCOPE_GLOBAL,
        5000,
        LLAMA_EXPERT_CACHE_ADMISSION_ALWAYS,
        0,
        0,
        {},
    };
    require(!llm_expert_cache_policy_copy_config(
        &invalid, llm_expert_cache_policy_tier::hot, first).is_ready(),
        "irrelevant noncanonical field rejected");
    invalid.policy = LLAMA_EXPERT_CACHE_POLICY_SLRU;
    invalid.admission = LLAMA_EXPERT_CACHE_ADMISSION_FREQUENCY_WINDOW;
    invalid.admission_window_events = 63;
    require(!llm_expert_cache_policy_copy_config(
        &invalid, llm_expert_cache_policy_tier::hot, first).is_ready(), "invalid window rejected");

    invalid.admission_window_events = 64;
    require(!llm_expert_cache_policy_copy_config(
        &invalid, llm_expert_cache_policy_tier::cold, first).is_ready(),
        "cold frequency-window admission rejected");
    invalid.admission = LLAMA_EXPERT_CACHE_ADMISSION_ALWAYS;
    require(!llm_expert_cache_policy_copy_config(
        &invalid, llm_expert_cache_policy_tier::hot, first).is_ready(),
        "ALWAYS rejects a nonzero admission window");
    invalid.admission_window_events = 0;
    invalid.reserved[2] = 1;
    require(!llm_expert_cache_policy_copy_config(
        &invalid, llm_expert_cache_policy_tier::hot, first).is_ready(),
        "nonzero reserved field rejected");
    invalid.reserved[2] = 0;
    invalid.version++;
    require(!llm_expert_cache_policy_copy_config(
        &invalid, llm_expert_cache_policy_tier::hot, first).is_ready(), "unknown version rejected");
}

void test_free_pin_failure_and_exhaustion() {
    const int32_t layers[] = { 0 };
    llm_expert_cache_policy policy;
    require(policy.initialize(make_config(LLAMA_EXPERT_CACHE_POLICY_LRU),
        llm_expert_cache_policy_tier::hot, layers, 1, 4, 1, 3, 128, 64).is_ready(),
        "initialize free/pin fixture");
    begin(policy);
    demand(policy, 0, 0);
    const llm_expert_cache_policy_candidate free_candidates[] = {
        candidate(2, 0, -1, -1, true, false), candidate(1, 0, -1, -1, true, false),
        candidate(0, 0, -1, -1, true, false),
    };
    llm_expert_cache_policy_decision decision;
    require(policy.select({ 0, 0 }, free_candidates, 3, decision).is_ready() && decision.slot == 0,
        "FREE slots use ascending physical order");
    const llm_expert_cache_policy_candidate duplicate_candidates[] = {
        candidate(0, 0, -1, -1, true, false), candidate(0, 0, -1, -1, true, false),
    };
    require(policy.select({ 0, 0 }, duplicate_candidates, 2, decision).error ==
            llm_expert_cache_policy_error::metadata_mismatch,
        "duplicate candidate slots are rejected");
    load(policy, 0, 1, 0, 0);
    require(policy.pin(0, 1).is_ready(), "pin resident");
    require(!policy.evict(0, 1).is_ready(), "pinned resident cannot be evicted");
    require(policy.unpin(0, 1).is_ready(), "unpin resident");
    require(!policy.unpin(0, 1).is_ready(), "duplicate unpin is a hard mismatch");
    require(policy.evict(0, 1).is_ready(), "unpinned resident can be evicted");
    require(policy.request_end(false, false).is_ready(), "failed request end");

    llm_expert_cache_policy bounded;
    require(bounded.initialize(make_config(LLAMA_EXPERT_CACHE_POLICY_LRU),
        llm_expert_cache_policy_tier::hot, layers, 1, 4, 1, 1, 128, 1).is_ready(),
        "initialize bounded transcript");
    begin(bounded);
    const auto before = bounded.diagnostics();
    require(bounded.demand({ 0, 0 }, 1, 64, 128).error ==
            llm_expert_cache_policy_error::transcript_full,
        "transcript exhaustion is a hard failure");
    require(bounded.diagnostics().demand_ordinal == before.demand_ordinal,
        "transcript exhaustion is transactional");

    llm_expert_cache_policy terminal_bounded;
    require(terminal_bounded.initialize(make_config(LLAMA_EXPERT_CACHE_POLICY_LRU),
        llm_expert_cache_policy_tier::hot, layers, 1, 4, 1, 1, 128, 3).is_ready(),
        "initialize terminal reservation fixture");
    begin(terminal_bounded);
    demand(terminal_bounded, 0, 0);
    require(terminal_bounded.load_begin(0, 1, { 0, 0 }, 64, 128).error ==
            llm_expert_cache_policy_error::transcript_full &&
            !terminal_bounded.validate_loading(0, 1, { 0, 0 }),
        "load begin reserves terminal transcript capacity before state mutation");

    llm_expert_cache_policy active_load;
    require(active_load.initialize(make_config(LLAMA_EXPERT_CACHE_POLICY_LRU),
        llm_expert_cache_policy_tier::hot, layers, 1, 4, 1, 1, 128, 5).is_ready(),
        "initialize active-load request fixture");
    begin(active_load);
    demand(active_load, 0, 0);
    require(active_load.load_begin(0, 1, { 0, 0 }, 64, 128).is_ready(),
        "reserve active load terminal");
    require(!active_load.validate_free(0),
        "policy loading state cannot be presented as a free mechanism slot");
    require(!active_load.request_end(true, false).is_ready(),
        "request cannot end before reserved load terminal");
    require(active_load.load_failed(0, 1).is_ready() &&
        active_load.request_end(false, true).is_ready(),
        "failed terminal precedes cancelled request end");
    require(active_load.validate_free(0),
        "failed ordered terminal restores policy free state");

    llm_expert_cache_policy multi_pin;
    require(multi_pin.initialize(make_config(LLAMA_EXPERT_CACHE_POLICY_LRU),
        llm_expert_cache_policy_tier::hot, layers, 1, 4, 1, 1, 128, 7).is_ready(),
        "initialize multi-pin capacity fixture");
    begin(multi_pin);
    demand(multi_pin, 0, 0);
    load(multi_pin, 0, 1, 0, 0);
    require(multi_pin.pin(0, 1).is_ready() && multi_pin.pin(0, 1).is_ready(),
        "acquire two policy pins");
    const uint64_t pin_digest = multi_pin.diagnostics().state_digest;
    require(multi_pin.validate_event_capacity(2).error ==
            llm_expert_cache_policy_error::transcript_full &&
            multi_pin.diagnostics().state_digest == pin_digest,
        "multi-pin release preflight fails before any ownership mutation");

    llm_expert_cache_policy exhausted;
    require(exhausted.initialize(make_config(LLAMA_EXPERT_CACHE_POLICY_LRU),
        llm_expert_cache_policy_tier::hot, layers, 1, 4, 1, 1, 128, 8).is_ready(),
        "initialize ordinal exhaustion fixture");
    require(exhausted.set_ordinals_for_testing(0, 0, UINT64_MAX), "install operation exhaustion seam");
    begin(exhausted);
    demand(exhausted, 0, 0);
    require(exhausted.load_begin(0, 1, { 0, 0 }, 64, 128).error ==
            llm_expert_cache_policy_error::sequence_exhausted,
        "operation ordinal never wraps");
    require(!exhausted.validate_loading(0, 1, { 0, 0 }), "exhausted operation creates no provisional state");
}

void test_eligibility_and_domain_edges() {
    const int32_t layers[] = { 2, 5 };
    llm_expert_cache_policy policy;
    require(policy.initialize(make_config(LLAMA_EXPERT_CACHE_POLICY_LRU,
        LLAMA_EXPERT_CACHE_POLICY_SCOPE_PER_LAYER), llm_expert_cache_policy_tier::hot,
        layers, 2, 4, 2, 5, 128, 128).is_ready(), "initialize remainder domains");
    const auto & domains = policy.domain_diagnostics();
    require(domains.size() == 2 && domains[0].slot_count == 3 && domains[1].slot_count == 2,
        "lower routed layer receives quotient remainder first");
    begin(policy);
    demand(policy, 2, 0);
    load(policy, 0, 1, 2, 0);
    demand(policy, 2, 1);
    load(policy, 1, 1, 2, 1);
    demand(policy, 2, 2);
    const llm_expert_cache_policy_candidate pinned_oldest[] = {
        candidate(0, 1, 2, 0, false, false), candidate(1, 1, 2, 1, false, true),
    };
    llm_expert_cache_policy_decision decision;
    require(policy.select({ 2, 2 }, pinned_oldest, 2, decision).is_ready() && decision.slot == 1,
        "ineligible oldest resident is never selected");
    const llm_expert_cache_policy_candidate all_ineligible[] = {
        candidate(0, 1, 2, 0, false, false), candidate(1, 1, 2, 1, false, false),
    };
    require(policy.select({ 2, 2 }, all_ineligible, 2, decision).error ==
            llm_expert_cache_policy_error::no_victim,
        "all-ineligible domain has no victim and no fallback");
    const llm_expert_cache_policy_candidate foreign_free[] = {
        candidate(3, 0, -1, -1, true, false), candidate(4, 0, -1, -1, true, false),
    };
    require(policy.select({ 2, 2 }, foreign_free, 2, decision).error ==
            llm_expert_cache_policy_error::no_victim,
        "hard per-layer domain cannot borrow free foreign slots");

    const uint64_t tight_budgets[] = { 255, 256 };
    llm_expert_cache_policy tight;
    require(tight.initialize(make_config(LLAMA_EXPERT_CACHE_POLICY_LRU,
        LLAMA_EXPERT_CACHE_POLICY_SCOPE_PER_LAYER), llm_expert_cache_policy_tier::hot,
        layers, 2, 4, 2, 5, 128, 64, tight_budgets, 2).is_ready(),
        "initialize explicit byte quotas");
    begin(tight);
    demand(tight, 2, 0);
    load(tight, 0, 1, 2, 0);
    demand(tight, 2, 1);
    require(tight.load_begin(1, 1, { 2, 1 }, 128, 128).error ==
            llm_expert_cache_policy_error::no_victim,
        "domain byte quota rejects occupancy overflow");
}

void test_lru() {
    const int32_t layers[] = { 0 };
    llm_expert_cache_policy policy;
    require(policy.initialize(make_config(LLAMA_EXPERT_CACHE_POLICY_LRU),
        llm_expert_cache_policy_tier::hot, layers, 1, 4, 1, 2, 128, 128).is_ready(), "initialize LRU");
    begin(policy);
    require(policy.set_ubatch_ordinal(1).is_ready(), "set first ubatch ordinal");
    require(!policy.set_ubatch_ordinal(1).is_ready(), "ubatch ordinal must increase");
    demand(policy, 0, 0);
    require(policy.transcript()[policy.transcript_size() - 1].ubatch_ordinal == 1,
        "demand records current ubatch ordinal");
    load(policy, 0, 1, 0, 0);
    demand(policy, 0, 1);
    load(policy, 1, 1, 0, 1);
    demand(policy, 0, 2);
    const llm_expert_cache_policy_candidate candidates[] = {
        candidate(0, 1, 0, 0, false, true), candidate(1, 1, 0, 1, false, true),
    };
    llm_expert_cache_policy_decision decision;
    require(policy.select({ 0, 2 }, candidates, 2, decision).is_ready(), "LRU select");
    require(decision.slot == 0 && !decision.free, "LRU chooses oldest slot");
    require(policy.evict(0, 1).is_ready(), "LRU evict");
    require(policy.load_begin(0, 2, { 0, 2 }, 64, 128).is_ready(), "LRU reuse slot");
    require(policy.load_failed(0, 2).is_ready(), "LRU failed load cleanup");
    require(!policy.validate_resident(0, 2, { 0, 2 }), "failed load not resident");
    require(policy.request_end(true, false).is_ready(), "LRU request end");
}

void test_always_optional_admission_event() {
    const int32_t layers[] = { 0 };
    llm_expert_cache_policy policy;
    require(policy.initialize(make_config(LLAMA_EXPERT_CACHE_POLICY_LRU),
        llm_expert_cache_policy_tier::hot, layers, 1, 4, 1, 1, 128, 32).is_ready(),
        "initialize always-admit fixture");
    begin(policy);
    demand(policy, 0, 0);
    const llm_expert_cache_policy_candidate candidates[] = {
        candidate(0, 0, -1, -1, true, false),
    };
    llm_expert_cache_policy_decision decision;
    require(policy.optional_admission({ 0, 0 },
        llm_expert_cache_policy_admission::optional_background,
        candidates, 1, decision).is_ready() && decision.accept && decision.free,
        "always admission accepts free slot");
    const auto & transcript = policy.transcript();
    const size_t count = policy.transcript_size();
    require(count >= 2 && transcript[count - 2].type ==
            llm_expert_cache_policy_event_type::victim_selected &&
            transcript[count - 1].type == llm_expert_cache_policy_event_type::optional_admission,
        "always admission emits selection and admission events");
    require(policy.diagnostics().optional_admission_accepts == 1,
        "always optional admission is counted");
    require(policy.optional_admission({ 0, 0 },
        llm_expert_cache_policy_admission::mandatory_current_output,
        candidates, 1, decision).is_ready() &&
        policy.transcript()[policy.transcript_size() - 1].type ==
            llm_expert_cache_policy_event_type::optional_admission &&
        policy.transcript()[policy.transcript_size() - 1].decision == 2 &&
        policy.diagnostics().mandatory_admissions == 1,
        "mandatory retention records an explicit not-applicable gate decision");
}

void test_heterogeneous_multi_victim_planning() {
    const int32_t layers[] = { 0 };
    const uint64_t budgets[] = { 300 };
    llm_expert_cache_policy policy;
    require(policy.initialize(make_config(LLAMA_EXPERT_CACHE_POLICY_LRU),
        llm_expert_cache_policy_tier::hot, layers, 1, 4, 1, 3, 128, 128,
        budgets, 1).is_ready(), "initialize heterogeneous-byte fixture");
    begin(policy);
    const uint64_t sizes[] = { 50, 100, 150 };
    for (uint32_t slot = 0; slot < 3; ++slot) {
        require(policy.demand({ 0, int32_t(slot) }, 1, sizes[slot], sizes[slot]).is_ready(),
            "heterogeneous demand");
        require(policy.load_begin(slot, 1, { 0, int32_t(slot) },
            sizes[slot], sizes[slot]).is_ready(), "heterogeneous load begin");
        require(policy.load_complete(slot, 1).is_ready(), "heterogeneous load complete");
    }
    require(policy.demand({ 0, 3 }, 1, 150, 150).is_ready(), "multi-victim demand");
    const llm_expert_cache_policy_candidate candidates[] = {
        { 0, 1, { 0, 0 }, 50, 50, false, true },
        { 1, 1, { 0, 1 }, 100, 100, false, true },
        { 2, 1, { 0, 2 }, 150, 150, false, true },
    };
    llm_expert_cache_policy_decision decisions[3];
    size_t decision_count = 0;
    require(policy.plan_evictions({ 0, 3 }, 150, 150, candidates, 3,
        decisions, 1, decision_count).error == llm_expert_cache_policy_error::overflow &&
        decision_count == 0, "bounded victim output rejects insufficient capacity transactionally");
    require(policy.plan_evictions({ 0, 3 }, 150, 150, candidates, 1,
        decisions, 3, decision_count).error == llm_expert_cache_policy_error::no_victim &&
        decision_count == 0, "insufficient eligible bytes are rejected");
    require(policy.plan_evictions({ 0, 3 }, 150, 150, candidates, 3,
        decisions, 3, decision_count).is_ready() && decision_count == 2 &&
        decisions[0].slot == 0 && decisions[1].slot == 1,
        "heterogeneous LRU returns the smallest ordered sufficient victim list");
    require(policy.evict(0, 1).is_ready() && policy.evict(1, 1).is_ready(),
        "commit planned heterogeneous victims");
    require(policy.load_begin(0, 2, { 0, 3 }, 150, 150).is_ready() &&
        policy.load_complete(0, 2).is_ready(), "reuse heterogeneous capacity");
    require(policy.domain_diagnostics()[0].quota_bytes == 300 &&
        policy.domain_diagnostics()[0].occupancy_bytes == 300,
        "heterogeneous physical occupancy matches domain quota");
}

void test_lfru() {
    const int32_t layers[] = { 0 };
    llm_expert_cache_policy policy;
    require(policy.initialize(make_config(LLAMA_EXPERT_CACHE_POLICY_LFRU),
        llm_expert_cache_policy_tier::hot, layers, 1, 4, 1, 2, 128, 256).is_ready(), "initialize LFRU");
    begin(policy);
    demand(policy, 0, 0);
    load(policy, 0, 1, 0, 0);
    demand(policy, 0, 1);
    load(policy, 1, 1, 0, 1);
    for (int repeat = 0; repeat < 3; ++repeat) {
        demand(policy, 0, 0);
        require(policy.hit(0, 1).is_ready(), "LFRU hot hit");
    }
    demand(policy, 0, 2);
    const llm_expert_cache_policy_candidate candidates[] = {
        candidate(0, 1, 0, 0, false, true), candidate(1, 1, 0, 1, false, true),
    };
    llm_expert_cache_policy_decision decision;
    require(policy.select({ 0, 2 }, candidates, 2, decision).is_ready(), "LFRU select");
    require(decision.slot == 1, "LFRU protects frequent recent key");
}

void test_batched_lfru_origin_demand() {
    const int32_t layers[] = { 0 };
    llm_expert_cache_policy policy;
    require(policy.initialize(make_config(LLAMA_EXPERT_CACHE_POLICY_LFRU),
        llm_expert_cache_policy_tier::hot, layers, 1, 4, 1, 2, 128, 128).is_ready(),
        "initialize batched LFRU fixture");
    begin(policy);
    demand(policy, 0, 0);
    load(policy, 0, 1, 0, 0);
    demand(policy, 0, 1);
    load(policy, 1, 1, 0, 1);
    require(policy.set_resident_frequency_for_testing(0, 2, 0) &&
        policy.set_resident_frequency_for_testing(1, 1, 0),
        "seed discriminating LFRU frequencies");
    demand(policy, 0, 0);
    demand(policy, 0, 1);
    demand(policy, 0, 2);
    require(policy.hit(0, 1).is_ready() && policy.hit(1, 1).is_ready(),
        "observe batched LFRU hits");
    const llm_expert_cache_policy_candidate candidates[] = {
        candidate(0, 1, 0, 0, false, true), candidate(1, 1, 0, 1, false, true),
    };
    llm_expert_cache_policy_decision decision;
    require(policy.select({ 0, 2 }, candidates, 2, decision).is_ready() && decision.slot == 0,
        "LFRU uses each key origin demand rather than the final batch ordinal");
}

void test_shuffled_terminal_order() {
    const int32_t layers[] = { 0 };
    auto run = [&](bool shuffled) {
        llm_expert_cache_policy policy;
        require(policy.initialize(make_config(LLAMA_EXPERT_CACHE_POLICY_LRU),
            llm_expert_cache_policy_tier::hot, layers, 1, 4, 1, 2, 128, 64).is_ready(),
            "initialize terminal-order fixture");
        begin(policy);
        demand(policy, 0, 0);
        require(policy.load_begin(0, 1, { 0, 0 }, 64, 128).is_ready(), "begin terminal op one");
        demand(policy, 0, 1);
        require(policy.load_begin(1, 1, { 0, 1 }, 64, 128).is_ready(), "begin terminal op two");
        if (shuffled) {
            const size_t before = policy.transcript_size();
            require(policy.load_complete(1, 1).is_ready() && policy.transcript_size() == before,
                "later terminal waits for its operation ordinal");
            require(policy.load_complete(0, 1).is_ready(), "earlier terminal flushes ordered queue");
        } else {
            require(policy.load_complete(0, 1).is_ready() && policy.load_complete(1, 1).is_ready(),
                "serial terminals complete");
        }
        return policy;
    };
    const auto serial = run(false);
    const auto shuffled = run(true);
    require(serial.transcript_size() == shuffled.transcript_size() &&
        serial.diagnostics().state_digest == shuffled.diagnostics().state_digest,
        "shuffled terminals preserve transcript length and final digest");
    for (size_t index = 0; index < serial.transcript_size(); ++index) {
        const auto & lhs = serial.transcript()[index];
        const auto & rhs = shuffled.transcript()[index];
        require(lhs.type == rhs.type && lhs.event_sequence == rhs.event_sequence &&
            lhs.origin_operation_ordinal == rhs.origin_operation_ordinal &&
            lhs.slot == rhs.slot && lhs.generation == rhs.generation &&
            lhs.state_digest == rhs.state_digest,
            "shuffled terminals produce identical canonical events");
    }
}

void test_frequency_window_digest_order() {
    const int32_t layers[] = { 0 };
    auto run = [&](const int32_t * sequence) {
        llm_expert_cache_policy policy;
        require(policy.initialize(make_config(LLAMA_EXPERT_CACHE_POLICY_SLRU,
            LLAMA_EXPERT_CACHE_POLICY_SCOPE_GLOBAL, 5000,
            LLAMA_EXPERT_CACHE_ADMISSION_FREQUENCY_WINDOW, 64),
            llm_expert_cache_policy_tier::hot, layers, 1, 8, 1, 1, 128, 64).is_ready(),
            "initialize window-digest fixture");
        begin(policy);
        for (size_t index = 0; index < 6; ++index) demand(policy, 0, sequence[index]);
        return policy.diagnostics().state_digest;
    };
    const int32_t first[] = { 0, 1, 0, 1, 2, 3 };
    const int32_t second[] = { 1, 0, 0, 1, 2, 3 };
    require(run(first) != run(second),
        "digest distinguishes equal counts and last touches with different ring order");
}

void test_slru_and_optional_admission() {
    const int32_t layers[] = { 0 };
    llm_expert_cache_policy policy;
    require(policy.initialize(make_config(LLAMA_EXPERT_CACHE_POLICY_SLRU,
        LLAMA_EXPERT_CACHE_POLICY_SCOPE_GLOBAL, 5000,
        LLAMA_EXPERT_CACHE_ADMISSION_FREQUENCY_WINDOW, 64),
        llm_expert_cache_policy_tier::hot, layers, 1, 4, 1, 2, 128, 256).is_ready(), "initialize SLRU");
    begin(policy);
    demand(policy, 0, 0);
    load(policy, 0, 1, 0, 0);
    require(policy.phase_transition(llm_expert_cache_policy_phase::decode).is_ready(), "decode transition");
    demand(policy, 0, 0);
    require(policy.hit(0, 1).is_ready(), "SLRU promote");
    demand(policy, 0, 1);
    load(policy, 1, 1, 0, 1);
    demand(policy, 0, 2);
    const llm_expert_cache_policy_candidate candidates[] = {
        candidate(0, 1, 0, 0, false, true), candidate(1, 1, 0, 1, false, true),
    };
    llm_expert_cache_policy_decision decision;
    require(policy.select({ 0, 2 }, candidates, 2, decision).is_ready(), "SLRU select");
    require(decision.slot == 1, "SLRU chooses probationary before protected");
    demand(policy, 0, 2);
    require(policy.optional_admission({ 0, 2 }, llm_expert_cache_policy_admission::optional_background,
        candidates, 2, decision).is_ready(), "optional admission");
    require(decision.accept && decision.slot == 1, "higher window frequency replaces probationary incumbent");
    const size_t event_count = policy.transcript_size();
    require(event_count >= 2 && policy.transcript()[event_count - 2].type ==
            llm_expert_cache_policy_event_type::victim_selected &&
            policy.transcript()[event_count - 1].type ==
            llm_expert_cache_policy_event_type::optional_admission,
        "frequency-gated acceptance emits victim selection before admission");
}

void test_slru_prefill_and_rejection() {
    const int32_t layers[] = { 0 };
    llm_expert_cache_policy policy;
    require(policy.initialize(make_config(LLAMA_EXPERT_CACHE_POLICY_SLRU,
        LLAMA_EXPERT_CACHE_POLICY_SCOPE_GLOBAL, 7500,
        LLAMA_EXPERT_CACHE_ADMISSION_FREQUENCY_WINDOW, 64),
        llm_expert_cache_policy_tier::hot, layers, 1, 8, 1, 4, 128, 256).is_ready(),
        "initialize SLRU prefill fixture");
    begin(policy);
    for (int expert = 0; expert < 4; ++expert) {
        demand(policy, 0, expert);
        load(policy, uint32_t(expert), 1, 0, expert);
    }
    require(policy.phase_transition(llm_expert_cache_policy_phase::decode).is_ready(), "enter decode");
    demand(policy, 0, 0);
    require(policy.hit(0, 1).is_ready(), "promote protected slot zero");
    demand(policy, 0, 1);
    require(policy.hit(1, 1).is_ready(), "promote protected slot one");
    require(policy.phase_transition(llm_expert_cache_policy_phase::prefill).is_ready(), "return to prefill");
    demand(policy, 0, 0);
    require(policy.hit(0, 1).is_ready(), "prefill protected hit");
    demand(policy, 0, 4);
    const llm_expert_cache_policy_candidate protected_only[] = {
        candidate(0, 1, 0, 0, false, true), candidate(1, 1, 0, 1, false, true),
        candidate(2, 1, 0, 2, false, false), candidate(3, 1, 0, 3, false, false),
    };
    llm_expert_cache_policy_decision decision;
    require(policy.select({ 0, 4 }, protected_only, 4, decision).is_ready() && decision.slot == 0,
        "prefill hit does not refresh protected order");
    require(policy.diagnostics().protected_forced_victims == 1, "forced protected victim counted");

    demand(policy, 0, 5);
    const llm_expert_cache_policy_candidate probationary[] = {
        candidate(2, 1, 0, 2, false, true), candidate(3, 1, 0, 3, false, true),
    };
    require(policy.optional_admission({ 0, 5 }, llm_expert_cache_policy_admission::optional_background,
        probationary, 2, decision).is_ready() && !decision.accept,
        "frequency tie preserves probationary incumbent");
    const llm_expert_cache_policy_candidate no_probationary[] = {
        candidate(0, 1, 0, 0, false, true), candidate(1, 1, 0, 1, false, true),
    };
    require(policy.optional_admission({ 0, 5 }, llm_expert_cache_policy_admission::optional_cpu_served,
        no_probationary, 2, decision).is_ready() && !decision.accept,
        "optional admission never evicts protected state");
}

void test_slru_pinned_blocked_demotion() {
    const int32_t layers[] = { 0 };
    llm_expert_cache_policy policy;
    require(policy.initialize(make_config(LLAMA_EXPERT_CACHE_POLICY_SLRU,
        LLAMA_EXPERT_CACHE_POLICY_SCOPE_GLOBAL, 5000),
        llm_expert_cache_policy_tier::hot, layers, 1, 4, 1, 2, 128, 128).is_ready(),
        "initialize pinned SLRU fixture");
    begin(policy);
    for (uint32_t slot = 0; slot < 2; ++slot) {
        demand(policy, 0, int32_t(slot));
        load(policy, slot, 1, 0, int32_t(slot));
        require(policy.pin(slot, 1).is_ready(), "pin SLRU resident");
    }
    require(policy.phase_transition(llm_expert_cache_policy_phase::decode).is_ready(),
        "enter pinned SLRU decode");
    for (uint32_t slot = 0; slot < 2; ++slot) {
        demand(policy, 0, int32_t(slot));
        require(policy.hit(slot, 1).is_ready(), "promote pinned SLRU resident");
    }
    require(policy.diagnostics().pinned_blocked_demotions == 1,
        "pinned protected overflow is counted without illegal demotion");
    require(policy.unpin(0, 1).is_ready(), "release first protected pin");
    require(policy.domain_diagnostics()[0].protected_occupancy_bytes == 128,
        "unpin reconciles protected occupancy to its byte budget");
    require(policy.unpin(1, 1).is_ready(), "release second protected pin");
    demand(policy, 0, 2);
    const llm_expert_cache_policy_candidate candidates[] = {
        candidate(0, 1, 0, 0, false, true), candidate(1, 1, 0, 1, false, true),
    };
    llm_expert_cache_policy_decision decision;
    require(policy.select({ 0, 2 }, candidates, 2, decision).is_ready() && decision.slot == 0,
        "post-unpin victim selection observes the reconciled probationary segment");
}

void test_lfu_aging_and_per_layer() {
    const int32_t layers[] = { 2, 5 };
    llm_expert_cache_policy policy;
    require(policy.initialize(make_config(LLAMA_EXPERT_CACHE_POLICY_LFU_AGING,
        LLAMA_EXPERT_CACHE_POLICY_SCOPE_PER_LAYER, 0,
        LLAMA_EXPERT_CACHE_ADMISSION_ALWAYS, 0, 64),
        llm_expert_cache_policy_tier::cold, layers, 2, 4, 1, 4, 128, 256).is_ready(),
        "initialize per-layer LFU aging");
    begin(policy);
    demand(policy, 5, 0);
    const llm_expert_cache_policy_candidate free_candidates[] = {
        candidate(0, 0, -1, -1, true, false), candidate(1, 0, -1, -1, true, false),
        candidate(2, 0, -1, -1, true, false), candidate(3, 0, -1, -1, true, false),
    };
    llm_expert_cache_policy_decision decision;
    require(policy.select({ 5, 0 }, free_candidates, 4, decision).is_ready(), "per-layer free select");
    require(decision.slot == 2, "per-layer domain cannot borrow lower slots");
    const auto & domains = policy.domain_diagnostics();
    require(domains.size() == 2 && domains[0].layer == 2 && domains[1].layer == 5 &&
            domains[0].slot_count == 2 && domains[1].slot_count == 2 &&
            domains[0].quota_bytes == 256 && domains[1].quota_bytes == 256,
        "per-layer quotas are reported in slots and physical bytes");
    load(policy, 2, 1, 5, 0);
    require(!policy.hit(2, 2).is_ready(), "stale generation is hard failure");
    require(policy.diagnostics().metadata_mismatches == 1, "metadata mismatch counted");

    llm_expert_cache_policy infeasible;
    require(!infeasible.initialize(make_config(LLAMA_EXPERT_CACHE_POLICY_LRU,
        LLAMA_EXPERT_CACHE_POLICY_SCOPE_PER_LAYER), llm_expert_cache_policy_tier::hot,
        layers, 2, 4, 2, 3, 128, 64).is_ready(),
        "per-layer minimum simultaneous width is enforced");
}

void test_reset_surrender_and_reinitialize() {
    const int32_t layers[] = { 0 };
    llm_expert_cache_policy policy;
    const auto config = make_config(LLAMA_EXPERT_CACHE_POLICY_LFU_AGING,
        LLAMA_EXPERT_CACHE_POLICY_SCOPE_GLOBAL, 0,
        LLAMA_EXPERT_CACHE_ADMISSION_ALWAYS, 0, 64);
    require(policy.initialize(config, llm_expert_cache_policy_tier::hot,
        layers, 1, 4, 1, 1, 128, 64).is_ready(), "initialize surrender fixture");
    begin(policy);
    demand(policy, 0, 0);
    load(policy, 0, 1, 0, 0);
    require(policy.set_resident_frequency_for_testing(0, UINT64_MAX, 0), "install saturation seam");
    demand(policy, 0, 0);
    require(policy.hit(0, 1).is_ready(), "saturated LFU hit");
    require(policy.diagnostics().frequency_saturations == 1, "frequency saturation counted");
    require(policy.evict(0, 1).is_ready(), "evict before surrender");
    require(policy.request_end(true, false).is_ready(), "end before surrender");
    require(policy.surrender().is_ready(), "quiescent policy surrender");
    require(policy.initialize(config, llm_expert_cache_policy_tier::hot,
        layers, 1, 4, 1, 1, 128, 64).is_ready(), "reinitialize surrendered policy");
    require(policy.diagnostics().event_sequence == 0 && policy.transcript_size() == 0,
        "reinitialize starts a fresh bounded transcript");
}

} // namespace

int main() {
    test_config();
    test_free_pin_failure_and_exhaustion();
    test_eligibility_and_domain_edges();
    test_lru();
    test_always_optional_admission_event();
    test_heterogeneous_multi_victim_planning();
    test_lfru();
    test_batched_lfru_origin_demand();
    test_shuffled_terminal_order();
    test_frequency_window_digest_order();
    test_slru_and_optional_admission();
    test_slru_prefill_and_rejection();
    test_slru_pinned_blocked_demotion();
    test_lfu_aging_and_per_layer();
    test_reset_surrender_and_reinitialize();
    std::puts("expert cache policy tests passed");
    return 0;
}
