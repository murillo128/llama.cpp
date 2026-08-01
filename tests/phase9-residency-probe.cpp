#include "llama-cold-expert-cache.h"

#include "ggml-cpp.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <sys/resource.h>
#include <vector>

using json = nlohmann::json;

namespace {

struct arguments {
    std::string output;
    uint64_t budget_bytes = 0;
    uint64_t projection_bytes = 5849088;
    uint32_t layers = 92;
    uint32_t experts = 896;
    uint32_t experts_used = 8;
    uint32_t touch_slots = 0;
    uint32_t classification_samples = 0;
    llama_expert_cache_policy policy = LLAMA_EXPERT_CACHE_POLICY_LRU;
    bool self_test = false;
};

bool parse_u64(const char * text, uint64_t & value) {
    char * end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (end == text || *end != '\0') return false;
    value = parsed;
    return true;
}

bool parse_u32(const char * text, uint32_t & value) {
    uint64_t parsed = 0;
    if (!parse_u64(text, parsed) || parsed > UINT32_MAX) return false;
    value = uint32_t(parsed);
    return true;
}

bool parse_arguments(int argc, char ** argv, arguments & result) {
    if (argc == 2 && std::string(argv[1]) == "--self-test") {
        result.self_test = true;
        result.output = "/tmp/phase9-residency-probe-self-test.json";
        result.projection_bytes = 65536;
        result.layers = 2;
        result.experts = 4;
        result.experts_used = 2;
        result.touch_slots = 4;
        result.classification_samples = 4;
        result.budget_bytes = 3*result.projection_bytes*result.touch_slots + 1024*1024;
        return true;
    }
    for (int index = 1; index < argc; ++index) {
        if (index + 1 >= argc) return false;
        const std::string option = argv[index];
        const char * value = argv[++index];
        if (option == "--output") result.output = value;
        else if (option == "--budget-bytes") { if (!parse_u64(value, result.budget_bytes)) return false; }
        else if (option == "--projection-bytes") { if (!parse_u64(value, result.projection_bytes)) return false; }
        else if (option == "--layers") { if (!parse_u32(value, result.layers)) return false; }
        else if (option == "--experts-per-layer") { if (!parse_u32(value, result.experts)) return false; }
        else if (option == "--experts-used") { if (!parse_u32(value, result.experts_used)) return false; }
        else if (option == "--touch-slots") { if (!parse_u32(value, result.touch_slots)) return false; }
        else if (option == "--classification-samples") { if (!parse_u32(value, result.classification_samples)) return false; }
        else if (option == "--policy") {
            const std::string policy = value;
            if (policy == "LRU") result.policy = LLAMA_EXPERT_CACHE_POLICY_LRU;
            else if (policy == "LFRU") result.policy = LLAMA_EXPERT_CACHE_POLICY_LFRU;
            else if (policy == "SLRU") result.policy = LLAMA_EXPERT_CACHE_POLICY_SLRU;
            else if (policy == "LFU_AGING") result.policy = LLAMA_EXPERT_CACHE_POLICY_LFU_AGING;
            else return false;
        } else return false;
    }
    return !result.output.empty() && result.budget_bytes > 0 && result.projection_bytes > 0 &&
        result.layers > 0 && result.experts > 0 && result.experts_used > 0 &&
        result.experts_used <= result.experts && result.touch_slots > 0;
}

const char * policy_name(llama_expert_cache_policy policy) {
    static const char * names[] = { "LRU", "LFRU", "SLRU", "LFU_AGING" };
    return names[unsigned(policy)];
}

struct prototype {
    ggml_context_ptr ctx;
    ggml_tensor * up = nullptr;
    ggml_tensor * gate = nullptr;
    ggml_tensor * down = nullptr;

    prototype(uint64_t projection_bytes, uint32_t experts) {
        ggml_init_params params = { ggml_tensor_overhead()*4, nullptr, true };
        ctx.reset(ggml_init(params));
        if (!ctx) throw std::runtime_error("ggml context allocation failed");
        up = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_I8, projection_bytes, 1, experts);
        gate = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_I8, projection_bytes, 1, experts);
        down = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_I8, projection_bytes, 1, experts);
    }

    llm_expert_bundle_descriptor bundle() const {
        return { 0, int32_t(up->ne[2]),
            llm_expert_projection_descriptor::from(up, nullptr, nullptr),
            llm_expert_projection_descriptor::from(gate, nullptr, nullptr), {},
            llm_expert_projection_descriptor::from(down, nullptr, nullptr) };
    }
};

llm_expert_provider_result zero_loader(
        void *, llm_expert_key, const llm_expert_bundle_descriptor & destination, uint32_t slot) noexcept {
    for (const auto * tensor : { destination.up.weight, destination.gate.weight, destination.down.weight }) {
        if (tensor == nullptr || tensor->data == nullptr || tensor->ne[2] != destination.n_expert) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::invalid_descriptor);
        }
        std::memset(static_cast<uint8_t *>(tensor->data) + uint64_t(slot)*tensor->nb[2], 0, tensor->nb[2]);
    }
    return llm_expert_provider_result::success();
}

std::map<std::string, uint64_t> read_kib(const std::string & path) {
    std::ifstream source(path);
    std::map<std::string, uint64_t> result;
    std::string key;
    uint64_t value = 0;
    std::string unit;
    while (source >> key >> value) {
        if (!key.empty() && key.back() == ':') key.pop_back();
        std::getline(source, unit);
        result[key] = value;
    }
    return result;
}

json rusage_json(const rusage & value) {
    return { {"minor_faults", value.ru_minflt}, {"major_faults", value.ru_majflt},
        {"max_rss_kib", value.ru_maxrss}, {"user_us", value.ru_utime.tv_sec*1000000LL + value.ru_utime.tv_usec},
        {"system_us", value.ru_stime.tv_sec*1000000LL + value.ru_stime.tv_usec} };
}

json kib_json(const std::map<std::string, uint64_t> & values) {
    json result = json::object();
    for (const auto & [key, value] : values) result[key] = value;
    return result;
}

} // namespace

int main(int argc, char ** argv) {
    arguments args;
    if (!parse_arguments(argc, argv, args)) return 2;
    try {
        prototype tensors(args.projection_bytes, args.experts);
        llm_cold_cache_config config;
        config.byte_budget = args.budget_bytes;
        config.minimum_slots = 1;
        config.routed_layer_count = args.layers;
        config.total_expert_keys = args.layers*args.experts;
        config.routed_layers.resize(args.layers);
        for (uint32_t layer = 0; layer < args.layers; ++layer) config.routed_layers[layer] = int32_t(layer);
        config.policy_trace_capacity = std::max<uint32_t>(8192, args.touch_slots*6 + args.classification_samples*4 + 16);
        const llama_expert_cache_policy_config public_policy = {
            LLAMA_EXPERT_CACHE_POLICY_VERSION_1, sizeof(llama_expert_cache_policy_config), args.policy,
            LLAMA_EXPERT_CACHE_POLICY_SCOPE_GLOBAL,
            args.policy == LLAMA_EXPERT_CACHE_POLICY_SLRU ? 7500u : 0u,
            LLAMA_EXPERT_CACHE_ADMISSION_ALWAYS, 0,
            args.policy == LLAMA_EXPERT_CACHE_POLICY_LFU_AGING ? 1024u : 0u, {},
        };
        if (!llm_expert_cache_policy_copy_config(&public_policy,
                llm_expert_cache_policy_tier::cold, config.cache_policy_config).is_ready()) return 3;
        rusage usage_before {};
        getrusage(RUSAGE_SELF, &usage_before);
        const auto meminfo_before = read_kib("/proc/meminfo");
        const auto smaps_before = read_kib("/proc/self/smaps_rollup");
        const auto start = std::chrono::steady_clock::now();
        llm_cold_expert_cache cache(config);
        if (!cache.initialize(tensors.bundle()).is_ready()) return 4;
        const auto initialized = cache.diagnostics();
        if (initialized.effective_slots == 0 || args.touch_slots > args.layers*args.experts) return 5;
        std::vector<llm_expert_key> keys;
        keys.reserve(args.touch_slots);
        for (uint32_t expert = 0; expert < args.experts && keys.size() < args.touch_slots; ++expert) {
            for (uint32_t layer = 0; layer < args.layers && keys.size() < args.touch_slots; ++layer) {
                if (expert < args.experts_used || keys.size() >= uint64_t(args.layers)*args.experts_used) {
                    keys.push_back({ int32_t(layer), int32_t(expert) });
                }
            }
        }
        if (keys.size() != args.touch_slots) return 6;
        for (const auto & key : keys) {
            llm_cold_reference reference;
            if (!cache.find_or_admit_with_loader(key, reference, zero_loader, nullptr).is_ready()) return 7;
        }
        uint32_t classified = 0;
        uint32_t classified_resident = 0;
        for (uint32_t sample = 0; sample < args.classification_samples; ++sample) {
            const auto before = cache.diagnostics();
            llm_cold_reference reference;
            if (!cache.find_or_admit_with_loader(keys[sample % keys.size()], reference, zero_loader, nullptr).is_ready()) return 8;
            const auto after = cache.diagnostics();
            if (before.residency_supported && after.residency_supported && before.ready_page_count != 0) {
                classified++;
                classified_resident += before.ready_page_count == before.resident_ready_page_count;
            }
        }
        if (!cache.policy_request_end(true, false).is_ready() || !cache.validate_invariants().is_ready()) return 9;
        const auto diagnostics = cache.diagnostics();
        const auto end = std::chrono::steady_clock::now();
        rusage usage_after {};
        getrusage(RUSAGE_SELF, &usage_after);
        const auto meminfo_after = read_kib("/proc/meminfo");
        const auto smaps_after = read_kib("/proc/self/smaps_rollup");
        const json output = {
            {"schema_version", "phase9-residency-probe-v1"}, {"status", "pass"},
            {"mode", args.classification_samples ? "residency-classification" : "performance"},
            {"policy", policy_name(args.policy)},
            {"layout", {{"layers", args.layers}, {"experts_per_layer", args.experts},
                {"experts_used", args.experts_used}, {"projection_bytes", args.projection_bytes},
                {"projections_per_bundle", 3}, {"logical_bundle_bytes", 3*args.projection_bytes},
                {"theoretical_token_working_set_bytes", 3*args.projection_bytes*args.layers*args.experts_used}}},
            {"capacity", {{"requested_bytes", args.budget_bytes}, {"actual_bytes", diagnostics.actual_bytes},
                {"effective_slots", diagnostics.effective_slots}, {"touched_slots", args.touch_slots},
                {"slot_footprint_bytes", diagnostics.aligned_slot_footprint}}},
            {"residency", {{"supported", diagnostics.residency_supported},
                {"unavailable_reason", diagnostics.residency_supported ? "" : diagnostics.residency_unavailable_reason},
                {"ready_logical_bytes", diagnostics.ready_logical_bytes}, {"ready_pages", diagnostics.ready_page_count},
                {"resident_ready_pages", diagnostics.resident_ready_page_count},
                {"resident_ready_bytes", diagnostics.resident_ready_bytes},
                {"classification_samples", classified}, {"fully_resident_samples", classified_resident},
                {"distribution_status", classified >= 100 ? "supported" : "insufficient-samples"}}},
            {"faults_before", rusage_json(usage_before)}, {"faults_after", rusage_json(usage_after)},
            {"meminfo_before_kib", kib_json(meminfo_before)}, {"meminfo_after_kib", kib_json(meminfo_after)},
            {"smaps_before_kib", kib_json(smaps_before)}, {"smaps_after_kib", kib_json(smaps_after)},
            {"elapsed_us", std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()},
            {"policy_digest", diagnostics.policy.state_digest}, {"self_test", args.self_test},
        };
        std::ofstream destination(args.output, std::ios::binary | std::ios::trunc);
        if (!destination) return 10;
        destination << output.dump(2) << '\n';
        destination.close();
        if (!destination) return 10;
        if (args.self_test && (!diagnostics.residency_supported || diagnostics.ready_page_count == 0 ||
                diagnostics.ready_page_count != diagnostics.resident_ready_page_count)) return 11;
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "phase9-residency-probe: %s\n", error.what());
        return 12;
    }
}
