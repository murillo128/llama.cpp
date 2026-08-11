#include "llama.h"
#include "llama-cpp.h"

#include "ggml-backend.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <array>
#include <string_view>
#include <string>
#include <sys/resource.h>
#include <vector>

using json = nlohmann::json;

namespace {

struct arguments {
    std::string model;
    std::string output;
    std::string quality_trace_output;
    std::string prompt = "Cache-aware routing opportunity probe.";
    std::vector<llama_token> teacher_forced_ids;
    int32_t prompt_token = -1;
    uint32_t candidate_count = 32;
    uint32_t max_generate = 2;
    uint32_t n_ctx = 512;
    uint32_t n_batch = 512;
    uint32_t n_ubatch = 512;
    int32_t threads = 16;
    bool routing_enabled = false;
    uint32_t routing_capacity_slots = 0;
    uint32_t routing_max_swaps = 0;
    float routing_max_score_regret = 0.0f;
};

bool parse_u32(const char * text, uint32_t & result) {
    try {
        size_t consumed = 0;
        const unsigned long value = std::stoul(text, &consumed, 10);
        if (text[consumed] != '\0' || value > UINT32_MAX) return false;
        result = uint32_t(value);
        return true;
    } catch (...) {
        return false;
    }
}

bool parse_i32(const char * text, int32_t & result) {
    try {
        size_t consumed = 0;
        const long value = std::stol(text, &consumed, 10);
        if (text[consumed] != '\0' || value < INT32_MIN || value > INT32_MAX) return false;
        result = int32_t(value);
        return true;
    } catch (...) {
        return false;
    }
}

bool parse_f32(const char * text, float & result) {
    try {
        size_t consumed = 0;
        result = std::stof(text, &consumed);
        return text[consumed] == '\0' && std::isfinite(result);
    } catch (...) {
        return false;
    }
}

bool parse_token_list(const char * text, std::vector<llama_token> & result) {
    if (text == nullptr || text[0] == '\0') return false;
    std::string_view remaining(text);
    while (!remaining.empty()) {
        const size_t delimiter = remaining.find(',');
        const std::string_view item = remaining.substr(0, delimiter);
        int32_t value = -1;
        const auto parsed = std::from_chars(item.data(), item.data() + item.size(), value);
        if (item.empty() || parsed.ec != std::errc() || parsed.ptr != item.data() + item.size() || value < 0) {
            return false;
        }
        result.push_back(value);
        if (delimiter == std::string_view::npos) break;
        remaining.remove_prefix(delimiter + 1);
    }
    return !result.empty();
}

bool parse_arguments(int argc, char ** argv, arguments & args) {
    for (int index = 1; index < argc; ++index) {
        const std::string option = argv[index];
        if (index + 1 >= argc) return false;
        const char * value = argv[++index];
        if (option == "--model") args.model = value;
        else if (option == "--output") args.output = value;
        else if (option == "--quality-trace-output") args.quality_trace_output = value;
        else if (option == "--prompt") args.prompt = value;
        else if (option == "--teacher-forced-ids") {
            if (!args.teacher_forced_ids.empty() || !parse_token_list(value, args.teacher_forced_ids)) return false;
        }
        else if (option == "--prompt-token") {
            if (!parse_i32(value, args.prompt_token) || args.prompt_token < 0) return false;
        }
        else if (option == "--candidate-count") {
            if (!parse_u32(value, args.candidate_count)) return false;
        } else if (option == "--max-generate") {
            if (!parse_u32(value, args.max_generate)) return false;
        } else if (option == "--n-ctx") {
            if (!parse_u32(value, args.n_ctx)) return false;
        } else if (option == "--n-batch") {
            if (!parse_u32(value, args.n_batch)) return false;
        } else if (option == "--n-ubatch") {
            if (!parse_u32(value, args.n_ubatch)) return false;
        } else if (option == "--threads") {
            if (!parse_i32(value, args.threads)) return false;
        } else if (option == "--routing-enabled") {
            uint32_t enabled = 0;
            if (!parse_u32(value, enabled) || enabled > 1) return false;
            args.routing_enabled = enabled != 0;
        } else if (option == "--routing-capacity-slots") {
            if (!parse_u32(value, args.routing_capacity_slots)) return false;
        } else if (option == "--routing-max-swaps") {
            if (!parse_u32(value, args.routing_max_swaps)) return false;
        } else if (option == "--routing-max-score-regret") {
            if (!parse_f32(value, args.routing_max_score_regret)) return false;
        } else {
            return false;
        }
    }
    return !args.model.empty() && !args.output.empty() && args.candidate_count != 0 &&
        args.max_generate != 0 && args.n_ctx != 0 && args.n_batch != 0 && args.n_ubatch != 0 &&
        args.threads > 0 && (args.teacher_forced_ids.empty() ||
            args.teacher_forced_ids.size() == args.max_generate) && (!args.routing_enabled ||
            (args.routing_capacity_slots > 0 && args.routing_max_swaps <= 16 &&
             args.routing_max_score_regret >= 0.0f));
}

enum quality_record_type : uint8_t {
    QUALITY_RECORD_MOE_OUTPUT = 1,
    QUALITY_RECORD_HIDDEN_STATE = 2,
    QUALITY_RECORD_LOGITS = 3,
};

struct quality_trace {
    std::string path;
    std::string error;
    std::ofstream destination;
    std::vector<uint8_t> raw;
    std::vector<float> values;
    std::vector<float> block;
    uint32_t current_step = 0;
    bool capture_internal = false;
    bool finished = false;
    uint64_t records = 0;
    uint64_t moe_records = 0;
    uint64_t hidden_records = 0;
    uint64_t logits_records = 0;
    uint64_t payload_bytes = 0;
    uint64_t file_bytes = 0;

    template<typename T>
    bool write_value(const T & value) {
        destination.write(reinterpret_cast<const char *>(&value), sizeof(value));
        file_bytes += sizeof(value);
        return bool(destination);
    }

    bool write_bytes(const void * data, size_t size) {
        destination.write(static_cast<const char *>(data), size);
        file_bytes += size;
        return bool(destination);
    }

    bool initialize(const std::string & output_path, const json & metadata, size_t reserve_values) {
        const uint32_t endian_probe = 1;
        if (*reinterpret_cast<const uint8_t *>(&endian_probe) != 1 || output_path.empty() ||
            reserve_values > SIZE_MAX/sizeof(float)) return false;
        path = output_path;
        try {
            raw.reserve(reserve_values*sizeof(float));
            values.reserve(reserve_values);
            block.reserve(256);
        } catch (const std::bad_alloc &) {
            return false;
        }
        destination.open(path, std::ios::binary | std::ios::trunc);
        if (!destination) return false;
        const char magic[8] = {'P', '1', '3', 'Q', 'T', 'R', '1', '\n'};
        const std::string header = metadata.dump();
        if (header.size() > UINT32_MAX) return false;
        const uint32_t header_size = uint32_t(header.size());
        return write_bytes(magic, sizeof(magic)) && write_value(header_size) &&
            write_bytes(header.data(), header.size());
    }

    static bool parse_layer(std::string_view name, std::string_view prefix, int32_t & layer) {
        if (name.size() < prefix.size() || name.substr(0, prefix.size()) != prefix) return false;
        const std::string_view suffix = name.substr(prefix.size());
        const auto parsed = std::from_chars(suffix.data(), suffix.data() + suffix.size(), layer);
        return !suffix.empty() && parsed.ec == std::errc() && parsed.ptr == suffix.data() + suffix.size() &&
            layer >= 0;
    }

    bool wants(const ggml_tensor * tensor, quality_record_type & type, int32_t & layer) const {
        if (!capture_internal || tensor == nullptr) return false;
        const std::string_view name(tensor->name);
        if (parse_layer(name, "ffn_moe_out-", layer)) {
            type = QUALITY_RECORD_MOE_OUTPUT;
            return true;
        }
        if (parse_layer(name, "l_out-", layer)) {
            type = QUALITY_RECORD_HIDDEN_STATE;
            return true;
        }
        return false;
    }

    bool tensor_to_float(const ggml_tensor * tensor) {
        const int64_t element_count = ggml_nelements(tensor);
        const size_t raw_bytes = ggml_nbytes(tensor);
        const size_t block_size = ggml_blck_size(tensor->type);
        const auto * traits = ggml_get_type_traits(tensor->type);
        if (element_count <= 0 || raw_bytes == 0 || block_size == 0 || traits == nullptr) return false;
        try {
            raw.resize(raw_bytes);
            values.clear();
            values.reserve(size_t(element_count));
            block.resize(block_size);
        } catch (const std::bad_alloc &) {
            return false;
        }
        ggml_backend_tensor_get(tensor, raw.data(), 0, raw.size());
        for (int64_t i3 = 0; i3 < tensor->ne[3]; ++i3) {
            for (int64_t i2 = 0; i2 < tensor->ne[2]; ++i2) {
                for (int64_t i1 = 0; i1 < tensor->ne[1]; ++i1) {
                    for (int64_t i0 = 0; i0 < tensor->ne[0]; i0 += block_size) {
                        const size_t offset = size_t(i3)*tensor->nb[3] + size_t(i2)*tensor->nb[2] +
                            size_t(i1)*tensor->nb[1] + size_t(i0/block_size)*tensor->nb[0];
                        if (offset >= raw.size()) return false;
                        if (tensor->type == GGML_TYPE_F32) {
                            float value = 0.0f;
                            if (offset + sizeof(value) > raw.size()) return false;
                            std::memcpy(&value, raw.data() + offset, sizeof(value));
                            values.push_back(value);
                        } else if (tensor->type == GGML_TYPE_F16) {
                            ggml_fp16_t value = 0;
                            if (offset + sizeof(value) > raw.size()) return false;
                            std::memcpy(&value, raw.data() + offset, sizeof(value));
                            values.push_back(ggml_fp16_to_fp32(value));
                        } else if (tensor->type == GGML_TYPE_BF16) {
                            ggml_bf16_t value {};
                            if (offset + sizeof(value) > raw.size()) return false;
                            std::memcpy(&value, raw.data() + offset, sizeof(value));
                            values.push_back(ggml_bf16_to_fp32(value));
                        } else if (traits->is_quantized && traits->to_float != nullptr) {
                            if (offset + traits->type_size > raw.size()) return false;
                            traits->to_float(raw.data() + offset, block.data(), block_size);
                            const size_t remaining = size_t(element_count) - values.size();
                            values.insert(values.end(), block.begin(), block.begin() + std::min(block_size, remaining));
                        } else {
                            return false;
                        }
                    }
                }
            }
        }
        return values.size() == size_t(element_count) &&
            std::all_of(values.begin(), values.end(), [](float value) { return std::isfinite(value); });
    }

    bool write_record(
            quality_record_type type,
            uint32_t step,
            int32_t layer,
            int32_t target_token,
            uint32_t n_tokens,
            const float * data,
            uint64_t count) {
        const uint8_t reserved[3] = {};
        if (data == nullptr || count == 0 || count > SIZE_MAX/sizeof(float) ||
            !write_value(uint8_t(type)) || !write_bytes(reserved, sizeof(reserved)) ||
            !write_value(step) || !write_value(layer) || !write_value(target_token) ||
            !write_value(n_tokens) || !write_value(count) ||
            !write_bytes(data, size_t(count)*sizeof(float))) {
            error = "quality trace write failed";
            return false;
        }
        records++;
        payload_bytes += count*sizeof(float);
        moe_records += type == QUALITY_RECORD_MOE_OUTPUT;
        hidden_records += type == QUALITY_RECORD_HIDDEN_STATE;
        logits_records += type == QUALITY_RECORD_LOGITS;
        return true;
    }

    bool capture_tensor(ggml_tensor * tensor, quality_record_type type, int32_t layer) {
        if (!tensor_to_float(tensor) || tensor->ne[1] <= 0 || tensor->ne[1] > UINT32_MAX) {
            error = "quality tensor conversion failed";
            return false;
        }
        return write_record(type, current_step, layer, -1, uint32_t(tensor->ne[1]),
            values.data(), values.size());
    }

    bool capture_logits(uint32_t step, llama_token target, const float * logits, uint32_t count) {
        if (logits == nullptr || count == 0 || target < 0 || uint32_t(target) >= count ||
            !std::all_of(logits, logits + count, [](float value) { return std::isfinite(value); })) {
            error = "quality logits are invalid";
            return false;
        }
        return write_record(QUALITY_RECORD_LOGITS, step, -1, target, 1, logits, count);
    }

    bool finish() {
        if (finished) return error.empty();
        finished = true;
        destination.flush();
        if (!destination) {
            error = "quality trace flush failed";
            return false;
        }
        destination.close();
        return bool(destination) && error.empty();
    }
};

bool capture_quality_tensor(ggml_tensor * tensor, bool ask, void * user_data) {
    auto & trace = *static_cast<quality_trace *>(user_data);
    quality_record_type type = QUALITY_RECORD_HIDDEN_STATE;
    int32_t layer = -1;
    if (!trace.wants(tensor, type, layer)) return false;
    return ask || trace.capture_tensor(tensor, type, layer);
}

struct route_lru {
    static constexpr uint32_t max_layers = 256;
    static constexpr uint32_t experts_per_layer = 896;
    static constexpr uint64_t empty_key = UINT64_MAX;

    uint32_t capacity = 0;
    uint32_t occupancy = 0;
    uint64_t use_clock = 0;
    std::vector<int32_t> key_to_slot;
    std::vector<uint64_t> slot_keys;
    std::vector<uint64_t> last_use;
    std::vector<uint64_t> unique_scratch;
    uint64_t snapshots = 0;
    uint64_t commits = 0;
    uint64_t requests = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;

    static uint64_t key(int32_t layer, int32_t expert) {
        return (uint64_t(uint32_t(layer)) << 32) | uint32_t(expert);
    }

    bool initialize(uint32_t requested_capacity, uint32_t n_ubatch) {
        if (requested_capacity == 0 || requested_capacity > max_layers*experts_per_layer ||
            n_ubatch == 0 || n_ubatch > SIZE_MAX/16) return false;
        capacity = requested_capacity;
        try {
            key_to_slot.assign(size_t(max_layers)*experts_per_layer, -1);
            slot_keys.assign(capacity, empty_key);
            last_use.assign(capacity, 0);
            unique_scratch.reserve(size_t(n_ubatch)*16);
        } catch (const std::bad_alloc &) {
            return false;
        }
        return true;
    }

    int32_t find(uint64_t value) const {
        const uint32_t layer = uint32_t(value >> 32);
        const uint32_t expert = uint32_t(value);
        if (layer >= max_layers || expert >= experts_per_layer) return -1;
        const int32_t slot = key_to_slot[size_t(layer)*experts_per_layer + expert];
        return slot >= 0 && uint32_t(slot) < occupancy && slot_keys[uint32_t(slot)] == value ? slot : -1;
    }

    void set_mapping(uint64_t value, int32_t slot) {
        const uint32_t layer = uint32_t(value >> 32);
        const uint32_t expert = uint32_t(value);
        key_to_slot[size_t(layer)*experts_per_layer + expert] = slot;
    }

    bool snapshot(
            const llama_cache_aware_routing_query * query,
            llama_route_service_tier * tiers) {
        if (query == nullptr || tiers == nullptr || capacity == 0) return false;
        const size_t count = size_t(query->n_tokens)*query->n_candidates;
        for (size_t index = 0; index < count; ++index) {
            tiers[index] = query->phase == LLAMA_ROUTE_PHASE_PREFILL ?
                LLAMA_ROUTE_SERVICE_TIER_HOT :
                (find(key(query->layer, query->candidate_experts[index])) >= 0 ?
                    LLAMA_ROUTE_SERVICE_TIER_COLD : LLAMA_ROUTE_SERVICE_TIER_BACKING);
        }
        snapshots++;
        return true;
    }

    bool commit(const llama_cache_aware_routing_query * query, const int32_t * final_experts) {
        if (query == nullptr || final_experts == nullptr || capacity == 0) return false;
        unique_scratch.clear();
        for (uint32_t token = 0; token < query->n_tokens; ++token) {
            for (uint32_t rank = 0; rank < query->n_expert_used; ++rank) {
                const uint64_t value = key(query->layer,
                    final_experts[size_t(token)*query->n_expert_used + rank]);
                if (std::find(unique_scratch.begin(), unique_scratch.end(), value) == unique_scratch.end()) {
                    if (unique_scratch.size() == unique_scratch.capacity()) return false;
                    unique_scratch.push_back(value);
                }
            }
        }
        for (const uint64_t value : unique_scratch) {
            requests++;
            const int32_t found = find(value);
            if (found >= 0) {
                hits++;
                last_use[uint32_t(found)] = ++use_clock;
                continue;
            }
            misses++;
            uint32_t slot = occupancy;
            if (occupancy < capacity) {
                occupancy++;
            } else {
                slot = uint32_t(std::min_element(last_use.begin(), last_use.end()) - last_use.begin());
                set_mapping(slot_keys[slot], -1);
            }
            slot_keys[slot] = value;
            last_use[slot] = ++use_clock;
            set_mapping(value, int32_t(slot));
        }
        commits++;
        return true;
    }
};

bool snapshot_route_tiers(
        const llama_cache_aware_routing_query * query,
        llama_route_service_tier * tiers,
        void * user_data) {
    return static_cast<route_lru *>(user_data)->snapshot(query, tiers);
}

bool commit_route(
        const llama_cache_aware_routing_query * query,
        const int32_t * final_experts,
        void * user_data) {
    return static_cast<route_lru *>(user_data)->commit(query, final_experts);
}

struct route_record {
    uint64_t request = 0;
    uint64_t ubatch = 0;
    std::string phase;
    int32_t layer = -1;
    uint32_t n_tokens = 0;
    uint32_t n_expert_used = 0;
    uint32_t n_candidates = 0;
    std::vector<llama_pos> positions;
    std::vector<int32_t> selected_experts;
    std::vector<float> weights;
    std::vector<int32_t> candidate_experts;
    std::vector<float> candidate_selection_scores;
    std::vector<float> candidate_probabilities;
};

struct route_capture {
    std::vector<route_record> records;
    std::string error;
    uint32_t expected_candidate_count = 0;
    bool routing_enabled = false;
    uint32_t routing_max_swaps = 0;
    float routing_max_score_regret = 0.0f;
};

bool capture_route(const llama_route_observation * observation, void * user_data) {
    auto & capture = *static_cast<route_capture *>(user_data);
    const size_t selected_count = size_t(observation->n_tokens)*observation->n_expert_used;
    const size_t candidate_count = size_t(observation->n_tokens)*observation->n_candidates;
    if (observation->n_candidates != capture.expected_candidate_count ||
        observation->candidate_experts == nullptr ||
        observation->candidate_selection_scores == nullptr ||
        observation->candidate_probabilities == nullptr) {
        capture.error = "candidate payload is incomplete or has the wrong count";
        return false;
    }

    route_record record;
    record.request = observation->request_ordinal;
    record.ubatch = observation->ubatch_ordinal;
    record.phase = observation->phase == LLAMA_ROUTE_PHASE_PREFILL ? "PREFILL" : "DECODE";
    record.layer = observation->layer;
    record.n_tokens = observation->n_tokens;
    record.n_expert_used = observation->n_expert_used;
    record.n_candidates = observation->n_candidates;
    if (observation->n_pos > 0 && observation->positions != nullptr) {
        record.positions.reserve(observation->n_tokens);
        for (uint32_t token = 0; token < observation->n_tokens; ++token) {
            record.positions.push_back(observation->positions[size_t(token)*observation->n_pos]);
        }
    }
    record.selected_experts.assign(observation->selected_experts,
        observation->selected_experts + selected_count);
    record.weights.assign(observation->weights, observation->weights + selected_count);
    record.candidate_experts.assign(observation->candidate_experts,
        observation->candidate_experts + candidate_count);
    record.candidate_selection_scores.assign(observation->candidate_selection_scores,
        observation->candidate_selection_scores + candidate_count);
    record.candidate_probabilities.assign(observation->candidate_probabilities,
        observation->candidate_probabilities + candidate_count);

    for (uint32_t token = 0; token < observation->n_tokens; ++token) {
        const size_t selected_offset = size_t(token)*observation->n_expert_used;
        const size_t candidate_offset = size_t(token)*observation->n_candidates;
        uint32_t changed = 0;
        for (uint32_t rank = 0; rank < observation->n_expert_used; ++rank) {
            const int32_t selected = record.selected_experts[selected_offset + rank];
            const int32_t exact = record.candidate_experts[candidate_offset + rank];
            if (selected == exact) continue;
            changed++;
            const auto begin = record.candidate_experts.begin() + candidate_offset + observation->n_expert_used;
            const auto end = record.candidate_experts.begin() + candidate_offset + observation->n_candidates;
            const auto replacement = std::find(begin, end, selected);
            if (!capture.routing_enabled || changed > capture.routing_max_swaps || replacement == end) {
                capture.error = "selected route is outside the bounded top-M substitution policy";
                return false;
            }
            const size_t replacement_index = size_t(replacement - record.candidate_experts.begin());
            const float regret = record.candidate_selection_scores[candidate_offset + rank] -
                record.candidate_selection_scores[replacement_index];
            if (regret < 0.0f || regret > capture.routing_max_score_regret) {
                capture.error = "selected route exceeds the configured score-regret bound";
                return false;
            }
        }
        for (uint32_t rank = 0; rank < observation->n_candidates; ++rank) {
            const size_t index = candidate_offset + rank;
            if (!std::isfinite(record.candidate_selection_scores[index]) ||
                !std::isfinite(record.candidate_probabilities[index])) {
                capture.error = "non-finite candidate value";
                return false;
            }
            if (rank > 0 && record.candidate_selection_scores[index] >
                record.candidate_selection_scores[index - 1]) {
                capture.error = "candidate scores are not ordered";
                return false;
            }
            if (std::find(record.candidate_experts.begin() + candidate_offset,
                    record.candidate_experts.begin() + index,
                    record.candidate_experts[index]) != record.candidate_experts.begin() + index) {
                capture.error = "duplicate candidate expert";
                return false;
            }
        }
    }

    capture.records.push_back(std::move(record));
    return true;
}

int finite_argmax(const float * logits, int count) {
    int result = -1;
    for (int index = 0; index < count; ++index) {
        if (!std::isfinite(logits[index])) return -1;
        if (result < 0 || logits[index] > logits[result]) result = index;
    }
    return result;
}

std::string token_piece(const llama_vocab * vocab, llama_token token) {
    std::string piece(16, '\0');
    int size = llama_token_to_piece(vocab, token, piece.data(), piece.size(), 0, true);
    if (size < 0) {
        piece.resize(size_t(-size));
        size = llama_token_to_piece(vocab, token, piece.data(), piece.size(), 0, true);
    }
    if (size < 0) return {};
    piece.resize(size_t(size));
    return piece;
}

json route_json(const route_capture & capture) {
    json result = json::array();
    for (const auto & record : capture.records) {
        result.push_back({
            {"request_ordinal", record.request},
            {"ubatch_ordinal", record.ubatch},
            {"phase", record.phase},
            {"layer", record.layer},
            {"n_tokens", record.n_tokens},
            {"n_expert_used", record.n_expert_used},
            {"n_candidates", record.n_candidates},
            {"positions", record.positions},
            {"selected_experts", record.selected_experts},
            {"weights", record.weights},
            {"candidate_experts", record.candidate_experts},
            {"candidate_selection_scores", record.candidate_selection_scores},
            {"candidate_probabilities", record.candidate_probabilities},
        });
    }
    return result;
}

} // namespace

int main(int argc, char ** argv) {
    arguments args;
    if (!parse_arguments(argc, argv, args)) {
        std::fprintf(stderr,
            "usage: %s --model GGUF --output JSON --candidate-count M --max-generate N "
            "[--prompt TEXT | --prompt-token ID] [--n-ctx N] [--n-batch N] [--n-ubatch N] [--threads N] "
            "[--teacher-forced-ids ID,ID,...] [--quality-trace-output BINARY] "
            "[--routing-enabled 0|1 --routing-capacity-slots N --routing-max-swaps N "
            "--routing-max-score-regret F]\n",
            argv[0]);
        return 2;
    }

    ggml_backend_load_all();
    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    model_params.use_extra_bufts = false;
    llama_model_ptr model(llama_model_load_from_file(args.model.c_str(), model_params));
    if (!model) return 3;

    const llama_vocab * vocab = llama_model_get_vocab(model.get());
    const int n_vocab = llama_vocab_n_tokens(vocab);
    if (std::any_of(args.teacher_forced_ids.begin(), args.teacher_forced_ids.end(),
            [n_vocab](llama_token token) { return token < 0 || token >= n_vocab; })) return 4;
    std::vector<llama_token> prompt;
    if (args.prompt_token >= 0) {
        if (args.prompt_token >= llama_vocab_n_tokens(vocab)) return 4;
        prompt.push_back(args.prompt_token);
    } else {
        const int prompt_count = -llama_tokenize(
            vocab, args.prompt.data(), args.prompt.size(), nullptr, 0, true, true);
        if (prompt_count <= 0 || uint32_t(prompt_count) >= args.n_ctx) return 4;
        prompt.resize(prompt_count);
        if (llama_tokenize(vocab, args.prompt.data(), args.prompt.size(), prompt.data(),
                prompt.size(), true, true) != prompt_count) return 4;
    }

    json command = json::array();
    for (int index = 0; index < argc; ++index) command.push_back(argv[index]);
    quality_trace trace;
    const bool quality_trace_enabled = !args.quality_trace_output.empty();
    if (quality_trace_enabled) {
        const int32_t n_embd = llama_model_n_embd(model.get());
        if (n_embd <= 0 || args.n_ubatch > SIZE_MAX/size_t(n_embd) ||
            !trace.initialize(args.quality_trace_output, {
                {"schema_version", "phase13-quality-trace-v1"},
                {"encoding", "little-endian-float32"},
                {"record_header", "<B3xIiiIQ"},
                {"record_types", {
                    {"1", "ffn_moe_out"}, {"2", "l_out"}, {"3", "logits"},
                }},
                {"command", command},
                {"model_path", args.model},
                {"prompt", args.prompt},
                {"prompt_token", args.prompt_token},
                {"prompt_ids", prompt},
                {"teacher_forced_ids", args.teacher_forced_ids},
                {"capture_internal_phase", "DECODE"},
                {"candidate_count", args.candidate_count},
                {"cache_aware_routing", {
                    {"enabled", args.routing_enabled},
                    {"capacity_slots", args.routing_capacity_slots},
                    {"max_swaps", args.routing_max_swaps},
                    {"max_score_regret", args.routing_max_score_regret},
                }},
            }, size_t(n_embd)*args.n_ubatch)) return 5;
    }

    auto context_params = llama_context_default_params();
    context_params.n_ctx = args.n_ctx;
    context_params.n_batch = args.n_batch;
    context_params.n_ubatch = args.n_ubatch;
    context_params.n_threads = args.threads;
    context_params.n_threads_batch = args.threads;
    context_params.no_perf = false;
    if (quality_trace_enabled) {
        context_params.cb_eval = capture_quality_tensor;
        context_params.cb_eval_user_data = &trace;
    }
    llama_context_ptr context(llama_init_from_model(model.get(), context_params));
    if (!context) return 5;

    route_lru routing_lru;
    if (args.routing_enabled && !routing_lru.initialize(
            args.routing_capacity_slots, args.n_ubatch)) return 6;
    if (args.routing_enabled) {
        const llama_cache_aware_routing_config config = {
            true,
            args.candidate_count,
            args.routing_max_swaps,
            args.routing_max_score_regret,
            snapshot_route_tiers,
            commit_route,
            &routing_lru,
        };
        if (llama_set_cache_aware_routing(context.get(), &config) !=
            LLAMA_ROUTE_OBSERVER_STATUS_OK) return 6;
    }
    if (llama_set_route_observer_candidate_count(context.get(), args.candidate_count) !=
        LLAMA_ROUTE_OBSERVER_STATUS_OK) {
        std::fprintf(stderr, "phase13-routing-probe: invalid candidate count %u\n", args.candidate_count);
        return 6;
    }
    route_capture routes;
    routes.expected_candidate_count = args.candidate_count;
    routes.routing_enabled = args.routing_enabled;
    routes.routing_max_swaps = args.routing_max_swaps;
    routes.routing_max_score_regret = args.routing_max_score_regret;
    if (llama_set_route_observer(context.get(), capture_route, &routes) !=
        LLAMA_ROUTE_OBSERVER_STATUS_OK) return 6;

    std::vector<llama_token> generated;
    std::vector<llama_token> argmax_ids;
    std::string generated_text;
    std::vector<uint64_t> latency_us;
    llama_batch batch = llama_batch_get_one(prompt.data(), prompt.size());
    for (uint32_t step = 0; step < args.max_generate; ++step) {
        const llama_route_phase phase = step == 0 ? LLAMA_ROUTE_PHASE_PREFILL : LLAMA_ROUTE_PHASE_DECODE;
        trace.current_step = step;
        trace.capture_internal = quality_trace_enabled && phase == LLAMA_ROUTE_PHASE_DECODE;
        if (llama_route_observer_begin(context.get(), 1, phase) !=
            LLAMA_ROUTE_OBSERVER_STATUS_OK) return 7;
        const auto begin = std::chrono::steady_clock::now();
        if (llama_decode(context.get(), batch) != 0) {
            std::fprintf(stderr, "phase13-routing-probe: decode failed at step %u\n", step);
            return 8;
        }
        llama_synchronize(context.get());
        const auto end = std::chrono::steady_clock::now();
        if (!routes.error.empty()) {
            std::fprintf(stderr, "phase13-routing-probe: %s\n", routes.error.c_str());
            return 9;
        }
        const float * logits = llama_get_logits_ith(context.get(), -1);
        const int next = logits == nullptr ? -1 : finite_argmax(logits, n_vocab);
        if (next < 0) return 10;
        argmax_ids.push_back(next);
        const llama_token accepted = args.teacher_forced_ids.empty() ? next : args.teacher_forced_ids[step];
        if (quality_trace_enabled && !trace.capture_logits(step, accepted, logits, uint32_t(n_vocab))) {
            std::fprintf(stderr, "phase13-routing-probe: %s\n", trace.error.c_str());
            return 10;
        }
        generated.push_back(accepted);
        if (args.prompt_token < 0) generated_text += token_piece(vocab, accepted);
        latency_us.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());
        if (args.teacher_forced_ids.empty() && args.prompt_token < 0 && llama_vocab_is_eog(vocab, accepted)) break;
        batch = llama_batch_get_one(&generated.back(), 1);
    }
    trace.capture_internal = false;
    if (quality_trace_enabled && !trace.finish()) {
        std::fprintf(stderr, "phase13-routing-probe: %s\n", trace.error.c_str());
        return 11;
    }

    const llama_route_observer_stats stats = llama_route_observer_get_stats(context.get());
    const llama_cache_aware_routing_stats routing_stats = llama_cache_aware_routing_get_stats(context.get());
    struct rusage usage {};
    getrusage(RUSAGE_SELF, &usage);
    const json output = {
        {"schema_version", "phase13-exact-topm-capture-v1"},
        {"status", "pass"},
        {"command", command},
        {"model_path", args.model},
        {"candidate_count", args.candidate_count},
        {"prompt", args.prompt},
        {"prompt_ids", prompt},
        {"execution", {
            {"backend", "CPU"},
            {"n_gpu_layers", 0},
            {"weight_repacking", false},
            {"n_ctx", args.n_ctx},
            {"n_batch", args.n_batch},
            {"n_ubatch", args.n_ubatch},
            {"threads", args.threads},
        }},
        {"sampling", {
            {"temperature", 0.0}, {"selection", "argmax"},
            {"teacher_forced", !args.teacher_forced_ids.empty()},
        }},
        {"cache_aware_routing", {
            {"enabled", args.routing_enabled},
            {"candidate_count", args.candidate_count},
            {"capacity_slots", args.routing_capacity_slots},
            {"max_swaps", args.routing_max_swaps},
            {"max_score_regret", args.routing_max_score_regret},
            {"prefill_rerouting", false},
            {"tier_source", args.routing_enabled ? "deterministic-exclusive-lru" : "disabled"},
            {"stats", {
                {"ubatches", routing_stats.ubatches},
                {"layers", routing_stats.layers},
                {"decisions", routing_stats.decisions},
                {"changed_decisions", routing_stats.changed_decisions},
                {"swaps", routing_stats.swaps},
                {"cumulative_score_regret", routing_stats.cumulative_score_regret},
                {"explicit_synchronizations", routing_stats.explicit_synchronizations},
                {"failures", routing_stats.failures},
            }},
            {"cache", {
                {"requests", routing_lru.requests}, {"hits", routing_lru.hits},
                {"misses", routing_lru.misses}, {"occupancy", routing_lru.occupancy},
                {"snapshots", routing_lru.snapshots}, {"commits", routing_lru.commits},
            }},
        }},
        {"generated_ids", generated},
        {"argmax_ids", argmax_ids},
        {"teacher_forced_ids", args.teacher_forced_ids},
        {"generated_text", generated_text},
        {"latency_us", latency_us},
        {"peak_rss_kib", usage.ru_maxrss},
        {"quality_trace", {
            {"enabled", quality_trace_enabled},
            {"path", trace.path},
            {"capture_internal_phase", "DECODE"},
            {"records", trace.records},
            {"moe_records", trace.moe_records},
            {"hidden_records", trace.hidden_records},
            {"logits_records", trace.logits_records},
            {"payload_bytes", trace.payload_bytes},
            {"file_bytes", trace.file_bytes},
            {"failures", trace.error.empty() ? 0 : 1},
        }},
        {"observer_stats", {
            {"ubatches", stats.ubatches},
            {"layers", stats.layers},
            {"copy_bytes", stats.copy_bytes},
            {"explicit_synchronizations", stats.explicit_synchronizations},
            {"failures", stats.failures},
        }},
        {"routes", route_json(routes)},
    };
    std::ofstream destination(args.output, std::ios::binary | std::ios::trunc);
    if (!destination) return 11;
    destination << output.dump(2) << '\n';
    destination.close();
    if (!destination) return 11;

    std::printf("PHASE13_ROUTING_PROBE status=pass routes=%zu output=%s\n",
        routes.records.size(), args.output.c_str());
    context.reset();
    model.reset();
    llama_backend_free();
    return 0;
}
