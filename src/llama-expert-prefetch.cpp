#include "llama-expert-prefetch.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <sys/stat.h>
#include <utility>

namespace {

struct json_allocation_ceiling {
    static thread_local bool enabled;
    static thread_local size_t remaining;
};

thread_local bool json_allocation_ceiling::enabled = false;
thread_local size_t json_allocation_ceiling::remaining = 0;

template<typename T>
struct bounded_json_allocator {
    using value_type = T;
    using is_always_equal = std::true_type;

    bounded_json_allocator() noexcept = default;
    template<typename U> bounded_json_allocator(const bounded_json_allocator<U> &) noexcept {}

    T * allocate(size_t count) {
        if (count > std::numeric_limits<size_t>::max()/sizeof(T)) throw std::bad_alloc();
        const size_t bytes = count*sizeof(T);
        if (json_allocation_ceiling::enabled) {
            if (bytes > json_allocation_ceiling::remaining) throw std::bad_alloc();
            json_allocation_ceiling::remaining -= bytes;
        }
        return std::allocator<T>{}.allocate(count);
    }

    void deallocate(T * pointer, size_t count) noexcept {
        std::allocator<T>{}.deallocate(pointer, count);
    }
};

template<typename T, typename U>
bool operator==(const bounded_json_allocator<T> &, const bounded_json_allocator<U> &) noexcept { return true; }

template<typename T, typename U>
bool operator!=(const bounded_json_allocator<T> &, const bounded_json_allocator<U> &) noexcept { return false; }

using bounded_json_string = std::basic_string<char, std::char_traits<char>, bounded_json_allocator<char>>;
using bounded_json_binary = std::vector<uint8_t, bounded_json_allocator<uint8_t>>;
using json = nlohmann::basic_json<std::map, std::vector, bounded_json_string, bool, int64_t, uint64_t, double,
    bounded_json_allocator, nlohmann::adl_serializer, bounded_json_binary>;
using bounded_json_key_set = std::set<bounded_json_string, std::less<bounded_json_string>,
    bounded_json_allocator<bounded_json_string>>;
using bounded_json_key_stack = std::vector<bounded_json_key_set, bounded_json_allocator<bounded_json_key_set>>;

struct json_ceiling_guard {
    explicit json_ceiling_guard(size_t budget) noexcept {
        json_allocation_ceiling::remaining = budget;
        json_allocation_ceiling::enabled = true;
    }
    ~json_ceiling_guard() {
        json_allocation_ceiling::enabled = false;
        json_allocation_ceiling::remaining = 0;
    }
};

constexpr uint64_t fnv_offset = UINT64_C(1469598103934665603);
constexpr uint64_t fnv_prime = UINT64_C(1099511628211);

constexpr const char * canonical_fold_prompts[6][6] = {
    { "prose-en-small",       "code-en-small",        "structured-en-small", "technical-en-large",  "narrative-en-large", "narrative-es-large" },
    { "code-en-small",        "structured-en-small",  "prose-en-small",      "technical-en-large",  "narrative-en-large", "narrative-es-large" },
    { "structured-en-small",  "technical-en-large",   "prose-en-small",      "code-en-small",       "narrative-en-large", "narrative-es-large" },
    { "technical-en-large",   "narrative-en-large",   "prose-en-small",      "code-en-small",       "structured-en-small", "narrative-es-large" },
    { "narrative-en-large",   "narrative-es-large",   "prose-en-small",      "code-en-small",       "structured-en-small", "technical-en-large" },
    { "narrative-es-large",   "prose-en-small",       "code-en-small",       "structured-en-small", "technical-en-large", "narrative-en-large" },
};

bool checked_add(uint64_t lhs, uint64_t rhs, uint64_t & result) noexcept {
    if (rhs > std::numeric_limits<uint64_t>::max() - lhs) return false;
    result = lhs + rhs;
    return true;
}

bool is_power_of_two(uint32_t value) noexcept {
    return value != 0 && (value & (value - 1)) == 0;
}

void hash_append(uint64_t & hash, uint64_t value) noexcept {
    for (uint32_t byte = 0; byte < 8; ++byte) {
        hash ^= uint8_t(value >> (byte*8));
        hash *= fnv_prime;
    }
}

void hash_append(uint64_t & hash, const char * value) noexcept {
    const size_t size = value == nullptr ? 0 : std::strlen(value);
    hash_append(hash, size);
    for (size_t index = 0; index < size; ++index) {
        const uint8_t byte = uint8_t(value[index]);
        hash ^= byte;
        hash *= fnv_prime;
    }
}

struct sha256 {
    std::array<uint32_t, 8> state = {
        0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
        0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U,
    };
    std::array<uint8_t, 64> block = {};
    uint64_t bytes = 0;
    size_t used = 0;

    static uint32_t rotate(uint32_t value, uint32_t count) noexcept {
        return value >> count | value << (32 - count);
    }

    void transform() noexcept {
        static constexpr uint32_t constants[64] = {
            0x428a2f98U,0x71374491U,0xb5c0fbcfU,0xe9b5dba5U,0x3956c25bU,0x59f111f1U,0x923f82a4U,0xab1c5ed5U,
            0xd807aa98U,0x12835b01U,0x243185beU,0x550c7dc3U,0x72be5d74U,0x80deb1feU,0x9bdc06a7U,0xc19bf174U,
            0xe49b69c1U,0xefbe4786U,0x0fc19dc6U,0x240ca1ccU,0x2de92c6fU,0x4a7484aaU,0x5cb0a9dcU,0x76f988daU,
            0x983e5152U,0xa831c66dU,0xb00327c8U,0xbf597fc7U,0xc6e00bf3U,0xd5a79147U,0x06ca6351U,0x14292967U,
            0x27b70a85U,0x2e1b2138U,0x4d2c6dfcU,0x53380d13U,0x650a7354U,0x766a0abbU,0x81c2c92eU,0x92722c85U,
            0xa2bfe8a1U,0xa81a664bU,0xc24b8b70U,0xc76c51a3U,0xd192e819U,0xd6990624U,0xf40e3585U,0x106aa070U,
            0x19a4c116U,0x1e376c08U,0x2748774cU,0x34b0bcb5U,0x391c0cb3U,0x4ed8aa4aU,0x5b9cca4fU,0x682e6ff3U,
            0x748f82eeU,0x78a5636fU,0x84c87814U,0x8cc70208U,0x90befffaU,0xa4506cebU,0xbef9a3f7U,0xc67178f2U,
        };
        uint32_t words[64];
        for (size_t index = 0; index < 16; ++index) {
            words[index] = uint32_t(block[index*4]) << 24 | uint32_t(block[index*4 + 1]) << 16 |
                uint32_t(block[index*4 + 2]) << 8 | block[index*4 + 3];
        }
        for (size_t index = 16; index < 64; ++index) {
            const uint32_t s0 = rotate(words[index - 15], 7) ^ rotate(words[index - 15], 18) ^ (words[index - 15] >> 3);
            const uint32_t s1 = rotate(words[index - 2], 17) ^ rotate(words[index - 2], 19) ^ (words[index - 2] >> 10);
            words[index] = words[index - 16] + s0 + words[index - 7] + s1;
        }
        uint32_t a=state[0],b=state[1],c=state[2],d=state[3],e=state[4],f=state[5],g=state[6],h=state[7];
        for (size_t index = 0; index < 64; ++index) {
            const uint32_t s1 = rotate(e, 6) ^ rotate(e, 11) ^ rotate(e, 25);
            const uint32_t choice = (e & f) ^ (~e & g);
            const uint32_t temporary1 = h + s1 + choice + constants[index] + words[index];
            const uint32_t s0 = rotate(a, 2) ^ rotate(a, 13) ^ rotate(a, 22);
            const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t temporary2 = s0 + majority;
            h=g; g=f; f=e; e=d+temporary1; d=c; c=b; b=a; a=temporary1+temporary2;
        }
        state[0]+=a; state[1]+=b; state[2]+=c; state[3]+=d;
        state[4]+=e; state[5]+=f; state[6]+=g; state[7]+=h;
    }

    void update(const void * source, size_t size) noexcept {
        const auto * input = static_cast<const uint8_t *>(source);
        while (size != 0) {
            const size_t count = std::min(size, block.size() - used);
            std::memcpy(block.data() + used, input, count);
            used += count;
            input += count;
            size -= count;
            bytes += count;
            if (used == block.size()) {
                transform();
                used = 0;
            }
        }
    }

    std::string finish() noexcept {
        const uint64_t bits = bytes*8;
        block[used++] = 0x80;
        if (used > 56) {
            while (used < block.size()) block[used++] = 0;
            transform();
            used = 0;
        }
        while (used < 56) block[used++] = 0;
        for (int shift = 56; shift >= 0; shift -= 8) block[used++] = uint8_t(bits >> shift);
        transform();
        std::ostringstream output;
        output << std::hex << std::setfill('0');
        for (uint32_t word : state) output << std::setw(8) << word;
        return output.str();
    }
};

bool valid_sha256(const std::string & value) noexcept {
    if (value.size() != 64) return false;
    for (char byte : value) {
        if (!((byte >= '0' && byte <= '9') || (byte >= 'a' && byte <= 'f'))) return false;
    }
    return true;
}

void require_fields(const json & value, std::initializer_list<const char *> fields, const char * name) {
    if (!value.is_object()) throw std::runtime_error(std::string(name) + " must be an object");
    std::set<std::string> actual;
    for (const auto & item : value.items()) actual.emplace(item.key().data(), item.key().size());
    std::set<std::string> expected;
    for (const char * field : fields) expected.insert(field);
    if (actual != expected) throw std::runtime_error(std::string(name) + " fields do not match v1");
}

uint64_t unsigned_value(const json & value, const char * name) {
    if (!value.is_number_unsigned()) throw std::runtime_error(std::string(name) + " must be an unsigned integer");
    return value.get<uint64_t>();
}

uint32_t uint32_value(const json & value, const char * name) {
    const uint64_t result = unsigned_value(value, name);
    if (result > UINT32_MAX) throw std::runtime_error(std::string(name) + " exceeds uint32");
    return uint32_t(result);
}

std::string string_value(const json & value, const char * name) {
    if (!value.is_string()) throw std::runtime_error(std::string(name) + " must be a string");
    const std::string result = value.get<std::string>();
    if (result.empty()) throw std::runtime_error(std::string(name) + " must not be empty");
    return result;
}

llama_expert_prefetch_readiness readiness_value(const json & value) {
    const std::string name = string_value(value, "readiness");
    if (name == "HOST_READY") return LLAMA_EXPERT_PREFETCH_READINESS_HOST_READY;
    if (name == "DEVICE_READY") return LLAMA_EXPERT_PREFETCH_READINESS_DEVICE_READY;
    throw std::runtime_error("unsupported readiness");
}

bool key_less(const llm_expert_prefetch_key & lhs, const llm_expert_prefetch_key & rhs) noexcept {
    return lhs.layer != rhs.layer ? lhs.layer < rhs.layer : lhs.expert < rhs.expert;
}

bool fingerprint_equal(
        const llm_expert_prefetch_fingerprint & lhs,
        const llm_expert_prefetch_fingerprint & rhs) noexcept {
    if (lhs.package_sha256 != rhs.package_sha256 || lhs.layer_count != rhs.layer_count ||
        lhs.routed_layers != rhs.routed_layers || lhs.experts_per_layer != rhs.experts_per_layer ||
        lhs.experts_per_token != rhs.experts_per_token || lhs.tensor_layout_sha256 != rhs.tensor_layout_sha256 ||
        lhs.files.size() != rhs.files.size() || lhs.expert_bytes.size() != rhs.expert_bytes.size()) return false;
    for (size_t index = 0; index < lhs.files.size(); ++index) {
        const auto & a = lhs.files[index];
        const auto & b = rhs.files[index];
        if (a.ordinal != b.ordinal || a.name != b.name || a.size != b.size || a.sha256 != b.sha256) return false;
    }
    for (size_t index = 0; index < lhs.expert_bytes.size(); ++index) {
        const auto & a = lhs.expert_bytes[index];
        const auto & b = rhs.expert_bytes[index];
        if (a.layer != b.layer || a.expert != b.expert || a.payload_bytes != b.payload_bytes ||
            a.physical_bytes != b.physical_bytes) return false;
    }
    return true;
}

uint64_t splitmix64(uint64_t & state) noexcept {
    uint64_t value = (state += UINT64_C(0x9e3779b97f4a7c15));
    value = (value ^ (value >> 30))*UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27))*UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}

} // namespace

std::string llm_expert_prefetch_sha256(const void * data, size_t size) {
    sha256 digest;
    digest.update(data, size);
    return digest.finish();
}

llm_expert_prefetch_result llm_expert_prefetch_sha256_file(
        const std::string & path,
        uint64_t expected_size,
        std::string & digest) noexcept {
    digest.clear();
    std::ifstream input(path, std::ios::binary);
    if (!input) return llm_expert_prefetch_result::failure(llm_expert_prefetch_error::profile_io);
    sha256 hash;
    std::array<char, 1024*1024> buffer;
    uint64_t bytes = 0;
    while (input) {
        input.read(buffer.data(), buffer.size());
        const size_t count = size_t(input.gcount());
        if (count != 0) {
            if (!checked_add(bytes, count, bytes)) return llm_expert_prefetch_result::failure(llm_expert_prefetch_error::overflow);
            hash.update(buffer.data(), count);
        }
    }
    if (!input.eof() || bytes != expected_size) return llm_expert_prefetch_result::failure(llm_expert_prefetch_error::profile_io);
    digest = hash.finish();
    return llm_expert_prefetch_result::success();
}

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
        llm_expert_prefetch_config_internal & destination) noexcept {
    destination = {};
    if (source == nullptr) {
        if (profile_path != nullptr) return llm_expert_prefetch_result::failure(llm_expert_prefetch_error::invalid_configuration);
        return llm_expert_prefetch_result::success();
    }
    const auto value = *source;
    if (value.version != LLAMA_EXPERT_PREFETCH_VERSION_1 || value.struct_size != sizeof(value) ||
        value.policy < LLAMA_EXPERT_PREFETCH_POLICY_OFF || value.policy >= LLAMA_EXPERT_PREFETCH_POLICY_COUNT ||
        value.readiness < LLAMA_EXPERT_PREFETCH_READINESS_HOST_READY || value.readiness >= LLAMA_EXPERT_PREFETCH_READINESS_COUNT ||
        value.seed_mode < LLAMA_EXPERT_PREFETCH_SEED_MODE_OFF || value.seed_mode >= LLAMA_EXPERT_PREFETCH_SEED_MODE_COUNT) {
        return llm_expert_prefetch_result::failure(llm_expert_prefetch_error::invalid_configuration);
    }
    for (uint64_t reserved : value.reserved) {
        if (reserved != 0) return llm_expert_prefetch_result::failure(llm_expert_prefetch_error::invalid_configuration);
    }
    const bool prediction = value.policy != LLAMA_EXPERT_PREFETCH_POLICY_OFF;
    const bool seed = value.seed_mode == LLAMA_EXPERT_PREFETCH_SEED_MODE_BLOCKING_HOT;
    if (!prediction && !seed) {
        const bool any = value.temporal_window_tokens != 0 || value.candidates_per_target != 0 ||
            value.max_profile_bytes != 0 || value.max_speculative_flights != 0 ||
            value.max_speculative_storage_bytes_in_flight != 0 || value.max_speculative_h2d_bytes_in_flight != 0 ||
            value.max_speculative_storage_bytes_per_token != 0 || value.max_speculative_h2d_bytes_per_token != 0 ||
            value.max_speculative_cold_slots != 0 || value.max_speculative_hot_slots != 0 ||
            value.utility_window_predictions != 0 || value.utility_min_observations != 0 || profile_path != nullptr;
        if (any) return llm_expert_prefetch_result::failure(llm_expert_prefetch_error::invalid_configuration);
    } else if (profile_path == nullptr || profile_path[0] == '\0' || value.max_profile_bytes == 0) {
        return llm_expert_prefetch_result::failure(llm_expert_prefetch_error::invalid_configuration);
    }
    if (mode != LLAMA_EXPERT_WEIGHTS_MODE_HOT_CACHE && mode != LLAMA_EXPERT_WEIGHTS_MODE_COLD_CACHE) {
        return llm_expert_prefetch_result::failure(llm_expert_prefetch_error::invalid_configuration);
    }
    if (value.readiness == LLAMA_EXPERT_PREFETCH_READINESS_HOST_READY && mode != LLAMA_EXPERT_WEIGHTS_MODE_COLD_CACHE) {
        return llm_expert_prefetch_result::failure(llm_expert_prefetch_error::invalid_configuration);
    }
    if (seed && (hot_capacity == 0 || (mode == LLAMA_EXPERT_WEIGHTS_MODE_COLD_CACHE && cold_capacity_bytes == 0))) {
        return llm_expert_prefetch_result::failure(llm_expert_prefetch_error::invalid_configuration);
    }
    if (value.policy == LLAMA_EXPERT_PREFETCH_POLICY_TEMPORAL_FREQUENCY) {
        if (!is_power_of_two(value.temporal_window_tokens) || value.temporal_window_tokens < 2 || value.temporal_window_tokens > 64) {
            return llm_expert_prefetch_result::failure(llm_expert_prefetch_error::invalid_configuration);
        }
    } else if (value.temporal_window_tokens != 0) {
        return llm_expert_prefetch_result::failure(llm_expert_prefetch_error::invalid_configuration);
    }
    if (prediction) {
        if (value.candidates_per_target == 0 || (experts_per_layer != 0 && value.candidates_per_target > experts_per_layer) ||
            value.max_speculative_flights == 0 || value.max_speculative_storage_bytes_in_flight == 0 ||
            value.max_speculative_h2d_bytes_in_flight == 0 || value.max_speculative_storage_bytes_per_token == 0 ||
            value.max_speculative_h2d_bytes_per_token == 0 || value.max_speculative_cold_slots == 0 ||
            value.max_speculative_hot_slots == 0 || !is_power_of_two(value.utility_window_predictions) ||
            value.utility_window_predictions < 2 || value.utility_min_observations == 0 ||
            value.utility_min_observations > value.utility_window_predictions) {
            return llm_expert_prefetch_result::failure(llm_expert_prefetch_error::invalid_configuration);
        }
        const uint32_t max_current_layer_demand_flights = experts_per_layer == 0 ? 0 :
            std::min(experts_per_layer, hot_capacity);
        if ((scheduler_capacity != 0 && max_current_layer_demand_flights != 0 &&
                (max_current_layer_demand_flights > scheduler_capacity ||
                 value.max_speculative_flights >
                    scheduler_capacity - max_current_layer_demand_flights)) ||
            (storage_capacity_bytes != 0 && value.max_speculative_storage_bytes_in_flight > storage_capacity_bytes) ||
            (h2d_capacity_bytes != 0 && value.max_speculative_h2d_bytes_in_flight > h2d_capacity_bytes) ||
            value.max_speculative_storage_bytes_per_token > value.max_speculative_storage_bytes_in_flight ||
            value.max_speculative_h2d_bytes_per_token > value.max_speculative_h2d_bytes_in_flight ||
            (mode == LLAMA_EXPERT_WEIGHTS_MODE_COLD_CACHE && value.max_speculative_cold_slots > scheduler_capacity) ||
            value.max_speculative_hot_slots > hot_capacity) {
            return llm_expert_prefetch_result::failure(llm_expert_prefetch_error::invalid_configuration);
        }
    } else if (value.candidates_per_target != 0 || value.max_speculative_flights != 0 ||
               value.max_speculative_storage_bytes_in_flight != 0 || value.max_speculative_h2d_bytes_in_flight != 0 ||
               value.max_speculative_storage_bytes_per_token != 0 || value.max_speculative_h2d_bytes_per_token != 0 ||
               value.max_speculative_cold_slots != 0 || value.max_speculative_hot_slots != 0 ||
               value.utility_window_predictions != 0 || value.utility_min_observations != 0) {
        return llm_expert_prefetch_result::failure(llm_expert_prefetch_error::invalid_configuration);
    }
    destination.supplied = true;
    destination.value = value;
    destination.digest = fnv_offset;
    hash_append(destination.digest, value.version);
    hash_append(destination.digest, value.struct_size);
    hash_append(destination.digest, uint32_t(value.policy));
    hash_append(destination.digest, uint32_t(value.readiness));
    hash_append(destination.digest, uint32_t(value.seed_mode));
    hash_append(destination.digest, value.temporal_window_tokens);
    hash_append(destination.digest, value.candidates_per_target);
    hash_append(destination.digest, value.max_profile_bytes);
    hash_append(destination.digest, value.max_speculative_flights);
    hash_append(destination.digest, value.max_speculative_storage_bytes_in_flight);
    hash_append(destination.digest, value.max_speculative_h2d_bytes_in_flight);
    hash_append(destination.digest, value.max_speculative_storage_bytes_per_token);
    hash_append(destination.digest, value.max_speculative_h2d_bytes_per_token);
    hash_append(destination.digest, value.max_speculative_cold_slots);
    hash_append(destination.digest, value.max_speculative_hot_slots);
    hash_append(destination.digest, value.utility_window_predictions);
    hash_append(destination.digest, value.utility_min_observations);
    for (uint64_t reserved : value.reserved) hash_append(destination.digest, reserved);
    return llm_expert_prefetch_result::success();
}

llm_expert_prefetch_result llm_expert_prefetch_load_profile(
        const std::string & path,
        uint64_t max_profile_bytes,
        const llm_expert_prefetch_fingerprint * expected,
        llm_expert_prefetch_profile & profile,
        std::string & error) noexcept {
    profile = {};
    error.clear();
    try {
        struct stat status = {};
        if (path.empty() || max_profile_bytes == 0 || stat(path.c_str(), &status) != 0 || status.st_size <= 0) {
            error = "profile is unavailable";
            return llm_expert_prefetch_result::failure(llm_expert_prefetch_error::profile_io);
        }
        if (uint64_t(status.st_size) > max_profile_bytes || uint64_t(status.st_size) > size_t(-1)) {
            error = "profile exceeds max_profile_bytes";
            return llm_expert_prefetch_result::failure(llm_expert_prefetch_error::profile_too_large);
        }
        std::ifstream input(path, std::ios::binary);
        if (!input) {
            error = "profile open failed";
            return llm_expert_prefetch_result::failure(llm_expert_prefetch_error::profile_io);
        }
        std::string bytes(size_t(status.st_size), '\0');
        input.read(bytes.data(), std::streamsize(bytes.size()));
        if (!input || input.peek() != std::ifstream::traits_type::eof()) {
            error = "profile read failed";
            return llm_expert_prefetch_result::failure(llm_expert_prefetch_error::profile_io);
        }

        json_ceiling_guard allocation_ceiling(size_t(max_profile_bytes) - bytes.size());
        bounded_json_key_stack object_keys;
        bool duplicate_key = false;
        auto callback = [&](int, json::parse_event_t event, json & parsed) {
            if (event == json::parse_event_t::object_start) object_keys.emplace_back();
            if (event == json::parse_event_t::key) {
                if (object_keys.empty() || !object_keys.back().insert(parsed.get<json::string_t>()).second) duplicate_key = true;
            }
            if (event == json::parse_event_t::object_end && !object_keys.empty()) object_keys.pop_back();
            return !duplicate_key;
        };
        const json document = json::parse(bytes, callback, true, false);
        if (duplicate_key || document.is_discarded()) throw std::runtime_error("duplicate JSON key");
        require_fields(document, { "schema_version", "profile_id", "tool", "source", "target", "static_counts",
            "transitions", "costs", "selection", "seed" }, "profile");
        if (string_value(document.at("schema_version"), "schema_version") != "expert-prefetch-profile-v1") {
            throw std::runtime_error("unsupported profile schema");
        }
        profile.profile_id = string_value(document.at("profile_id"), "profile_id");

        const auto & tool = document.at("tool");
        require_fields(tool, { "name", "version" }, "tool");
        string_value(tool.at("name"), "tool.name");
        uint32_value(tool.at("version"), "tool.version");

        const auto & source = document.at("source");
        require_fields(source, { "kind", "artifacts", "fold" }, "source");
        profile.source_kind = string_value(source.at("kind"), "source.kind");
        if (profile.source_kind != "route_trace" && profile.source_kind != "imatrix_derived") {
            throw std::runtime_error("unsupported source kind");
        }
        if (!source.at("artifacts").is_array() || source.at("artifacts").empty()) {
            throw std::runtime_error("source.artifacts must be non-empty");
        }
        for (const auto & artifact : source.at("artifacts")) {
            require_fields(artifact, { "name", "size", "sha256" }, "source artifact");
            string_value(artifact.at("name"), "artifact.name");
            if (unsigned_value(artifact.at("size"), "artifact.size") == 0 ||
                !valid_sha256(string_value(artifact.at("sha256"), "artifact.sha256"))) {
                throw std::runtime_error("invalid source artifact identity");
            }
        }
        const auto & fold = source.at("fold");
        require_fields(fold, { "index", "training", "validation", "test", "training_rows", "validation_rows", "test_rows" }, "fold");
        profile.fold_index = uint32_value(fold.at("index"), "fold.index");
        if (profile.fold_index >= 6 || !fold.at("training").is_array() || fold.at("training").size() != 4) {
            throw std::runtime_error("invalid fold membership");
        }
        profile.training_prompts.reserve(4);
        std::set<std::string> prompt_members;
        for (const auto & prompt : fold.at("training")) {
            profile.training_prompts.push_back(string_value(prompt, "fold.training"));
            if (!prompt_members.insert(profile.training_prompts.back()).second) throw std::runtime_error("duplicate fold prompt");
        }
        profile.validation_prompt = string_value(fold.at("validation"), "fold.validation");
        profile.test_prompt = string_value(fold.at("test"), "fold.test");
        if (!prompt_members.insert(profile.validation_prompt).second || !prompt_members.insert(profile.test_prompt).second ||
            prompt_members.size() != 6) throw std::runtime_error("overlapping fold membership");
        const auto & expected_fold = canonical_fold_prompts[profile.fold_index];
        if (profile.test_prompt != expected_fold[0] || profile.validation_prompt != expected_fold[1]) {
            throw std::runtime_error("noncanonical fold validation or test prompt");
        }
        for (size_t index = 0; index < profile.training_prompts.size(); ++index) {
            if (profile.training_prompts[index] != expected_fold[index + 2]) {
                throw std::runtime_error("noncanonical fold training prompts");
            }
        }
        profile.training_rows = unsigned_value(fold.at("training_rows"), "fold.training_rows");
        profile.validation_rows = unsigned_value(fold.at("validation_rows"), "fold.validation_rows");
        profile.test_rows = unsigned_value(fold.at("test_rows"), "fold.test_rows");
        if (profile.training_rows == 0 || profile.validation_rows == 0 || profile.test_rows == 0) {
            throw std::runtime_error("empty fold rows");
        }

        const auto & target = document.at("target");
        require_fields(target, { "package_sha256", "files", "layer_count", "routed_layers", "experts_per_layer",
            "experts_per_token", "tensor_layout_sha256", "expert_bytes" }, "target");
        profile.target.package_sha256 = string_value(target.at("package_sha256"), "target.package_sha256");
        profile.target.tensor_layout_sha256 = string_value(target.at("tensor_layout_sha256"), "target.tensor_layout_sha256");
        if (!valid_sha256(profile.target.package_sha256) || !valid_sha256(profile.target.tensor_layout_sha256)) {
            throw std::runtime_error("invalid target digest");
        }
        profile.target.layer_count = uint32_value(target.at("layer_count"), "target.layer_count");
        profile.target.experts_per_layer = uint32_value(target.at("experts_per_layer"), "target.experts_per_layer");
        profile.target.experts_per_token = uint32_value(target.at("experts_per_token"), "target.experts_per_token");
        if (profile.target.layer_count == 0 || profile.target.experts_per_layer == 0 || profile.target.experts_per_token == 0 ||
            profile.target.experts_per_token > profile.target.experts_per_layer) throw std::runtime_error("invalid target topology");
        if (!target.at("files").is_array() || target.at("files").empty()) throw std::runtime_error("target files missing");
        profile.target.files.reserve(target.at("files").size());
        for (const auto & file : target.at("files")) {
            require_fields(file, { "ordinal", "name", "size", "sha256" }, "target file");
            llm_expert_prefetch_file_identity identity;
            identity.ordinal = uint32_value(file.at("ordinal"), "file.ordinal");
            identity.name = string_value(file.at("name"), "file.name");
            identity.size = unsigned_value(file.at("size"), "file.size");
            identity.sha256 = string_value(file.at("sha256"), "file.sha256");
            if (identity.ordinal != profile.target.files.size() || identity.size == 0 || !valid_sha256(identity.sha256)) {
                throw std::runtime_error("invalid target file identity");
            }
            profile.target.files.push_back(std::move(identity));
        }
        if (!target.at("routed_layers").is_array() || target.at("routed_layers").empty()) throw std::runtime_error("routed layers missing");
        profile.target.routed_layers.reserve(target.at("routed_layers").size());
        int32_t previous_layer = -1;
        for (const auto & layer_value : target.at("routed_layers")) {
            const uint32_t layer = uint32_value(layer_value, "routed layer");
            if (layer >= profile.target.layer_count || int32_t(layer) <= previous_layer) throw std::runtime_error("routed layers not canonical");
            profile.target.routed_layers.push_back(int32_t(layer));
            previous_layer = int32_t(layer);
        }
        const uint64_t expected_keys = uint64_t(profile.target.routed_layers.size())*profile.target.experts_per_layer;
        if (expected_keys > SIZE_MAX || !target.at("expert_bytes").is_array() || target.at("expert_bytes").size() != expected_keys) {
            throw std::runtime_error("expert byte map is incomplete");
        }
        profile.target.expert_bytes.reserve(size_t(expected_keys));
        for (const auto & item : target.at("expert_bytes")) {
            require_fields(item, { "layer", "expert", "payload_bytes", "physical_bytes" }, "expert bytes");
            llm_expert_prefetch_key_bytes key;
            key.layer = int32_t(uint32_value(item.at("layer"), "expert layer"));
            key.expert = int32_t(uint32_value(item.at("expert"), "expert id"));
            key.payload_bytes = unsigned_value(item.at("payload_bytes"), "payload bytes");
            key.physical_bytes = unsigned_value(item.at("physical_bytes"), "physical bytes");
            const size_t index = profile.target.expert_bytes.size();
            const int32_t expected_layer = profile.target.routed_layers[index/profile.target.experts_per_layer];
            const int32_t expected_expert = int32_t(index%profile.target.experts_per_layer);
            if (key.layer != expected_layer || key.expert != expected_expert || key.payload_bytes == 0 ||
                key.physical_bytes < key.payload_bytes) throw std::runtime_error("expert byte map is not canonical");
            profile.target.expert_bytes.push_back(key);
        }
        std::ostringstream package_identity;
        for (const auto & file : profile.target.files) {
            package_identity << file.ordinal << ':' << file.name.size() << ':' << file.name << ':'
                             << file.size << ':' << file.sha256 << '\n';
        }
        package_identity << profile.target.tensor_layout_sha256 << '\n';
        const std::string package_text = package_identity.str();
        if (llm_expert_prefetch_sha256(package_text.data(), package_text.size()) != profile.target.package_sha256) {
            throw std::runtime_error("target package digest is inconsistent");
        }

        if (!document.at("static_counts").is_array()) throw std::runtime_error("static_counts must be an array");
        profile.static_counts.reserve(document.at("static_counts").size());
        std::set<std::pair<int32_t, int32_t>> count_keys;
        for (const auto & item : document.at("static_counts")) {
            require_fields(item, { "layer", "expert", "count" }, "static count");
            llm_expert_prefetch_count count;
            count.layer = int32_t(uint32_value(item.at("layer"), "count.layer"));
            count.expert = int32_t(uint32_value(item.at("expert"), "count.expert"));
            count.count = unsigned_value(item.at("count"), "count.count");
            if (std::find(profile.target.routed_layers.begin(), profile.target.routed_layers.end(), count.layer) == profile.target.routed_layers.end() ||
                count.expert < 0 || uint32_t(count.expert) >= profile.target.experts_per_layer || count.count == 0 ||
                !count_keys.insert({count.layer, count.expert}).second) throw std::runtime_error("invalid static count");
            profile.static_counts.push_back(count);
        }
        std::sort(profile.static_counts.begin(), profile.static_counts.end(), [](const auto & lhs, const auto & rhs) {
            return lhs.layer != rhs.layer ? lhs.layer < rhs.layer :
                (lhs.count != rhs.count ? lhs.count > rhs.count : lhs.expert < rhs.expert);
        });

        if (!document.at("transitions").is_array()) throw std::runtime_error("transitions must be an array");
        profile.transitions.reserve(document.at("transitions").size());
        std::set<std::array<int32_t, 4>> transition_keys;
        for (const auto & item : document.at("transitions")) {
            require_fields(item, { "source_layer", "source_expert", "target_layer", "target_expert", "count" }, "transition");
            llm_expert_prefetch_transition transition;
            transition.source_layer = int32_t(uint32_value(item.at("source_layer"), "source_layer"));
            transition.source_expert = int32_t(uint32_value(item.at("source_expert"), "source_expert"));
            transition.target_layer = int32_t(uint32_value(item.at("target_layer"), "target_layer"));
            transition.target_expert = int32_t(uint32_value(item.at("target_expert"), "target_expert"));
            transition.count = unsigned_value(item.at("count"), "transition.count");
            const auto source_it = std::find(profile.target.routed_layers.begin(), profile.target.routed_layers.end(), transition.source_layer);
            if (source_it == profile.target.routed_layers.end() || source_it + 1 == profile.target.routed_layers.end() ||
                *(source_it + 1) != transition.target_layer || transition.source_expert < 0 || transition.target_expert < 0 ||
                uint32_t(transition.source_expert) >= profile.target.experts_per_layer ||
                uint32_t(transition.target_expert) >= profile.target.experts_per_layer || transition.count == 0 ||
                !transition_keys.insert({transition.source_layer, transition.source_expert, transition.target_layer, transition.target_expert}).second) {
                throw std::runtime_error("invalid sparse transition");
            }
            profile.transitions.push_back(transition);
        }
        std::sort(profile.transitions.begin(), profile.transitions.end(), [](const auto & lhs, const auto & rhs) {
            return std::tie(lhs.source_layer, lhs.source_expert, lhs.target_layer, lhs.target_expert) <
                std::tie(rhs.source_layer, rhs.source_expert, rhs.target_layer, rhs.target_expert);
        });

        if (!document.at("costs").is_array() || document.at("costs").empty()) throw std::runtime_error("costs missing");
        profile.costs.reserve(document.at("costs").size());
        std::set<std::pair<std::string, int>> cost_keys;
        for (const auto & item : document.at("costs")) {
            require_fields(item, { "transport", "readiness", "lead_ns", "demand_service_ns", "speculative_service_ns",
                "predictor_compute_ns", "scheduler_demand_delay_ns", "displacement_refill_ns", "storage_bytes", "h2d_bytes",
                "break_even_bps", "utility_window_predictions", "utility_min_observations", "utility_min_timely_successes" }, "cost");
            llm_expert_prefetch_cost cost;
            cost.transport = string_value(item.at("transport"), "cost.transport");
            cost.readiness = readiness_value(item.at("readiness"));
            cost.lead_ns = unsigned_value(item.at("lead_ns"), "cost.lead_ns");
            cost.demand_service_ns = unsigned_value(item.at("demand_service_ns"), "cost.demand_service_ns");
            cost.speculative_service_ns = unsigned_value(item.at("speculative_service_ns"), "cost.speculative_service_ns");
            cost.predictor_compute_ns = unsigned_value(item.at("predictor_compute_ns"), "cost.predictor_compute_ns");
            cost.scheduler_demand_delay_ns = unsigned_value(item.at("scheduler_demand_delay_ns"), "cost.scheduler_demand_delay_ns");
            cost.displacement_refill_ns = unsigned_value(item.at("displacement_refill_ns"), "cost.displacement_refill_ns");
            cost.storage_bytes = unsigned_value(item.at("storage_bytes"), "cost.storage_bytes");
            cost.h2d_bytes = unsigned_value(item.at("h2d_bytes"), "cost.h2d_bytes");
            cost.break_even_bps = uint32_value(item.at("break_even_bps"), "cost.break_even_bps");
            cost.utility_window_predictions = uint32_value(item.at("utility_window_predictions"), "cost.utility_window_predictions");
            cost.utility_min_observations = uint32_value(item.at("utility_min_observations"), "cost.utility_min_observations");
            cost.utility_min_timely_successes = uint32_value(item.at("utility_min_timely_successes"), "cost.utility_min_timely_successes");
            uint64_t hidden = 0, waste = 0;
            uint32_t recomputed = 0;
            const auto computed = llm_expert_prefetch_break_even(cost.lead_ns, cost.demand_service_ns,
                cost.predictor_compute_ns, cost.speculative_service_ns, cost.scheduler_demand_delay_ns,
                cost.displacement_refill_ns, hidden, waste, recomputed);
            const uint64_t expected_successes =
                (uint64_t(recomputed)*cost.utility_window_predictions + 9999)/10000;
            const bool buffered = cost.transport == "BUFFERED" || cost.transport == "DIRECT_IO";
            const bool h2d_only = cost.transport == "HOST_TO_DEVICE";
            const bool valid_path =
                (buffered && cost.readiness == LLAMA_EXPERT_PREFETCH_READINESS_HOST_READY && cost.storage_bytes != 0 && cost.h2d_bytes == 0) ||
                (buffered && cost.readiness == LLAMA_EXPERT_PREFETCH_READINESS_DEVICE_READY && cost.storage_bytes != 0 && cost.h2d_bytes != 0) ||
                (h2d_only && cost.readiness == LLAMA_EXPERT_PREFETCH_READINESS_DEVICE_READY && cost.storage_bytes == 0 && cost.h2d_bytes != 0);
            if (!computed.is_ready() || recomputed != cost.break_even_bps || !is_power_of_two(cost.utility_window_predictions) ||
                cost.utility_min_observations == 0 || cost.utility_min_observations > cost.utility_window_predictions ||
                cost.utility_min_timely_successes != expected_successes || !valid_path ||
                !cost_keys.insert({cost.transport, int(cost.readiness)}).second) throw std::runtime_error("invalid cost envelope");
            profile.costs.push_back(cost);
        }

        const auto & selection = document.at("selection");
        require_fields(selection, { "matrix_version", "tuning_digest", "fold_index", "policy", "candidates_per_target",
            "temporal_window_tokens", "transport", "readiness", "break_even_bps" }, "selection");
        if (uint32_value(selection.at("matrix_version"), "matrix_version") != 1 ||
            !valid_sha256(string_value(selection.at("tuning_digest"), "tuning_digest")) ||
            uint32_value(selection.at("fold_index"), "selection.fold_index") != profile.fold_index) throw std::runtime_error("invalid selection provenance");
        profile.selected_policy = string_value(selection.at("policy"), "selection.policy");
        static const std::set<std::string> selected_policies = {
            "STATIC_LAYER", "PREVIOUS_TOKEN", "TEMPORAL_FREQUENCY", "CROSS_LAYER_TRANSITION",
            "RANDOM_BASELINE", "BLOCKING_HOT",
        };
        if (selected_policies.count(profile.selected_policy) == 0) throw std::runtime_error("invalid selected policy");
        profile.selected_candidates = uint32_value(selection.at("candidates_per_target"), "selection.candidates_per_target");
        profile.selected_temporal_window = uint32_value(selection.at("temporal_window_tokens"), "selection.temporal_window_tokens");
        profile.selected_transport = string_value(selection.at("transport"), "selection.transport");
        profile.selected_readiness = readiness_value(selection.at("readiness"));
        profile.selected_break_even_bps = uint32_value(selection.at("break_even_bps"), "selection.break_even_bps");
        if ((profile.selected_policy == "BLOCKING_HOT") != (profile.selected_candidates == 0) ||
            profile.selected_candidates > profile.target.experts_per_layer ||
            (profile.selected_policy == "TEMPORAL_FREQUENCY" &&
                (!is_power_of_two(profile.selected_temporal_window) || profile.selected_temporal_window < 2 ||
                 profile.selected_temporal_window > 64)) ||
            (profile.selected_policy != "TEMPORAL_FREQUENCY" && profile.selected_temporal_window != 0) ||
            profile.selected_break_even_bps > 10000) throw std::runtime_error("invalid selected tuning");
        const auto selected_cost = std::find_if(profile.costs.begin(), profile.costs.end(), [&](const auto & cost) {
            return cost.transport == profile.selected_transport && cost.readiness == profile.selected_readiness;
        });
        if (selected_cost == profile.costs.end() || selected_cost->break_even_bps != profile.selected_break_even_bps) {
            throw std::runtime_error("selected cost envelope mismatch");
        }

        if (!document.at("seed").is_array()) throw std::runtime_error("seed must be an array");
        profile.seed.reserve(document.at("seed").size());
        std::set<std::pair<int32_t, int32_t>> seed_keys;
        for (const auto & item : document.at("seed")) {
            require_fields(item, { "layer", "expert", "count", "payload_bytes", "physical_bytes" }, "seed");
            llm_expert_prefetch_seed seed;
            seed.layer = int32_t(uint32_value(item.at("layer"), "seed.layer"));
            seed.expert = int32_t(uint32_value(item.at("expert"), "seed.expert"));
            seed.count = unsigned_value(item.at("count"), "seed.count");
            seed.payload_bytes = unsigned_value(item.at("payload_bytes"), "seed.payload_bytes");
            seed.physical_bytes = unsigned_value(item.at("physical_bytes"), "seed.physical_bytes");
            if (seed.count == 0 || seed.payload_bytes == 0 || seed.physical_bytes < seed.payload_bytes ||
                !seed_keys.insert({seed.layer, seed.expert}).second) throw std::runtime_error("invalid seed entry");
            const auto key = std::find_if(profile.target.expert_bytes.begin(), profile.target.expert_bytes.end(), [&](const auto & value) {
                return value.layer == seed.layer && value.expert == seed.expert;
            });
            if (key == profile.target.expert_bytes.end() || key->payload_bytes != seed.payload_bytes ||
                key->physical_bytes != seed.physical_bytes) throw std::runtime_error("seed byte map mismatch");
            profile.seed.push_back(seed);
        }
        std::sort(profile.seed.begin(), profile.seed.end(), [](const auto & lhs, const auto & rhs) {
            return lhs.count != rhs.count ? lhs.count < rhs.count :
                (lhs.layer != rhs.layer ? lhs.layer < rhs.layer : lhs.expert < rhs.expert);
        });
        if (profile.seed.size() > profile.static_counts.size()) throw std::runtime_error("seed set exceeds static counts");
        auto ranked_counts = profile.static_counts;
        std::sort(ranked_counts.begin(), ranked_counts.end(), [](const auto & lhs, const auto & rhs) {
            return lhs.count != rhs.count ? lhs.count > rhs.count :
                (lhs.layer != rhs.layer ? lhs.layer < rhs.layer : lhs.expert < rhs.expert);
        });
        for (size_t index = 0; index < profile.seed.size(); ++index) {
            const auto selected = std::find_if(profile.seed.begin(), profile.seed.end(), [&](const auto & seed) {
                return seed.layer == ranked_counts[index].layer && seed.expert == ranked_counts[index].expert;
            });
            if (selected == profile.seed.end() || selected->count != ranked_counts[index].count) {
                throw std::runtime_error("seed set is not the ranked complete prefix");
            }
        }

        if (expected != nullptr && !fingerprint_equal(profile.target, *expected)) {
            error = "profile target fingerprint mismatch";
            return llm_expert_prefetch_result::failure(llm_expert_prefetch_error::profile_identity_mismatch);
        }
        profile.profile_sha256 = llm_expert_prefetch_sha256(bytes.data(), bytes.size());
        profile.profile_bytes = bytes.size();
        return llm_expert_prefetch_result::success();
    } catch (const std::bad_alloc &) {
        error = "profile allocation failed";
        profile = {};
        return llm_expert_prefetch_result::failure(llm_expert_prefetch_error::profile_too_large);
    } catch (const std::exception & exception) {
        error = exception.what();
        profile = {};
        return llm_expert_prefetch_result::failure(llm_expert_prefetch_error::invalid_profile);
    }
}

llm_expert_prefetch_result llm_expert_prefetch_break_even(
        uint64_t lead_ns,
        uint64_t demand_service_ns,
        uint64_t predictor_compute_ns,
        uint64_t speculative_service_ns,
        uint64_t scheduler_demand_delay_ns,
        uint64_t displacement_refill_ns,
        uint64_t & hidden_benefit_ns,
        uint64_t & waste_cost_ns,
        uint32_t & break_even_bps) noexcept {
    hidden_benefit_ns = 0;
    waste_cost_ns = 0;
    break_even_bps = 0;
    const uint64_t hidden_before_predictor = std::min(lead_ns, demand_service_ns);
    if (hidden_before_predictor <= predictor_compute_ns) {
        return llm_expert_prefetch_result::failure(llm_expert_prefetch_error::invalid_profile);
    }
    hidden_benefit_ns = hidden_before_predictor - predictor_compute_ns;
    if (!checked_add(predictor_compute_ns, speculative_service_ns, waste_cost_ns) ||
        !checked_add(waste_cost_ns, scheduler_demand_delay_ns, waste_cost_ns) ||
        !checked_add(waste_cost_ns, displacement_refill_ns, waste_cost_ns)) {
        return llm_expert_prefetch_result::failure(llm_expert_prefetch_error::overflow);
    }
    uint64_t denominator = 0;
    if (!checked_add(hidden_benefit_ns, waste_cost_ns, denominator) || denominator == 0 ||
        waste_cost_ns > std::numeric_limits<uint64_t>::max()/10000) {
        return llm_expert_prefetch_result::failure(llm_expert_prefetch_error::overflow);
    }
    const uint64_t numerator = waste_cost_ns*10000;
    const uint64_t rounded = numerator/denominator + (numerator%denominator != 0);
    if (rounded > 10000) return llm_expert_prefetch_result::failure(llm_expert_prefetch_error::overflow);
    break_even_bps = uint32_t(rounded);
    return llm_expert_prefetch_result::success();
}

llm_expert_prefetch_result llm_expert_prefetch_predictor::initialize(
        const llm_expert_prefetch_profile & profile,
        const llm_expert_prefetch_config_internal & config) noexcept {
    this->profile = nullptr;
    this->config = {};
    request = {};
    if (!config.supplied || config.value.policy == LLAMA_EXPERT_PREFETCH_POLICY_OFF ||
        profile.target.routed_layers.empty() || profile.target.experts_per_layer == 0 || !valid_sha256(profile.profile_sha256) ||
        config.value.candidates_per_target == 0 || config.value.candidates_per_target > profile.target.experts_per_layer) {
        return llm_expert_prefetch_result::failure(llm_expert_prefetch_error::invalid_configuration);
    }
    try {
        request.token_history.reserve(config.value.temporal_window_tokens == 0 ? 1 : config.value.temporal_window_tokens);
    } catch (...) {
        return llm_expert_prefetch_result::failure(llm_expert_prefetch_error::overflow);
    }
    this->profile = &profile;
    this->config = config;
    hash_append(this->config.digest, profile.profile_sha256.c_str());
    return llm_expert_prefetch_result::success();
}

llm_expert_prefetch_result llm_expert_prefetch_predictor::request_begin(uint64_t request_ordinal) noexcept {
    if (profile == nullptr || request_ordinal == 0) {
        return llm_expert_prefetch_result::failure(llm_expert_prefetch_error::invalid_configuration);
    }
    request.request_ordinal = request_ordinal;
    request.completed_tokens = 0;
    request.digest = fnv_offset;
    request.token_history.clear();
    hash_append(request.digest, request_ordinal);
    return llm_expert_prefetch_result::success();
}

llm_expert_prefetch_result llm_expert_prefetch_predictor::commit_token(
        uint64_t token_ordinal,
        const std::vector<std::vector<int32_t>> & routed_experts) noexcept {
    if (profile == nullptr || request.request_ordinal == 0 || token_ordinal != request.completed_tokens ||
        routed_experts.size() != profile->target.routed_layers.size()) {
        return llm_expert_prefetch_result::failure(llm_expert_prefetch_error::invalid_profile);
    }
    try {
        std::vector<std::vector<int32_t>> canonical;
        canonical.reserve(routed_experts.size());
        for (const auto & selected : routed_experts) {
            if (selected.empty() || selected.size() > profile->target.experts_per_token) {
                return llm_expert_prefetch_result::failure(llm_expert_prefetch_error::invalid_profile);
            }
            canonical.push_back(selected);
            auto & values = canonical.back();
            std::sort(values.begin(), values.end());
            if (std::adjacent_find(values.begin(), values.end()) != values.end()) {
                return llm_expert_prefetch_result::failure(llm_expert_prefetch_error::invalid_profile);
            }
            for (int32_t expert : values) {
                if (expert < 0 || uint32_t(expert) >= profile->target.experts_per_layer) {
                    return llm_expert_prefetch_result::failure(llm_expert_prefetch_error::invalid_profile);
                }
            }
        }
        const uint32_t window = config.value.temporal_window_tokens == 0 ? 1 : config.value.temporal_window_tokens;
        if (request.token_history.size() == window) request.token_history.erase(request.token_history.begin());
        request.token_history.push_back(std::move(canonical));
        hash_append(request.digest, token_ordinal);
        for (const auto & layer : request.token_history.back()) {
            hash_append(request.digest, layer.size());
            for (int32_t expert : layer) hash_append(request.digest, uint32_t(expert));
        }
        request.completed_tokens++;
        return llm_expert_prefetch_result::success();
    } catch (...) {
        return llm_expert_prefetch_result::failure(llm_expert_prefetch_error::overflow);
    }
}

llm_expert_prefetch_result llm_expert_prefetch_predictor::predict_token_end(
        uint64_t token_ordinal,
        int32_t target_layer,
        std::vector<llm_expert_prefetch_candidate> & candidates) const noexcept {
    candidates.clear();
    if (profile == nullptr || request.request_ordinal == 0 || request.completed_tokens == 0 ||
        token_ordinal + 1 != request.completed_tokens) {
        return llm_expert_prefetch_result::failure(llm_expert_prefetch_error::invalid_profile);
    }
    const auto layer_it = std::find(profile->target.routed_layers.begin(), profile->target.routed_layers.end(), target_layer);
    if (layer_it == profile->target.routed_layers.end()) {
        return llm_expert_prefetch_result::failure(llm_expert_prefetch_error::invalid_profile);
    }
    const size_t layer_index = size_t(layer_it - profile->target.routed_layers.begin());
    try {
        switch (config.value.policy) {
            case LLAMA_EXPERT_PREFETCH_POLICY_STATIC_LAYER:
                for (const auto & count : profile->static_counts) {
                    if (count.layer == target_layer) candidates.push_back({{target_layer, count.expert}, 0, count.count});
                }
                break;
            case LLAMA_EXPERT_PREFETCH_POLICY_PREVIOUS_TOKEN:
                for (int32_t expert : request.token_history.back()[layer_index]) {
                    candidates.push_back({{target_layer, expert}, 0, 1});
                }
                break;
            case LLAMA_EXPERT_PREFETCH_POLICY_TEMPORAL_FREQUENCY: {
                struct score { uint64_t count = 0; uint64_t recency = 0; };
                std::vector<score> scores(profile->target.experts_per_layer);
                for (size_t history_index = 0; history_index < request.token_history.size(); ++history_index) {
                    for (int32_t expert : request.token_history[history_index][layer_index]) {
                        if (scores[expert].count != UINT64_MAX) scores[expert].count++;
                        scores[expert].recency = history_index + 1;
                    }
                }
                for (uint32_t expert = 0; expert < scores.size(); ++expert) {
                    if (scores[expert].count != 0) candidates.push_back({{target_layer, int32_t(expert)}, 0,
                        scores[expert].count*128 + scores[expert].recency});
                }
                std::sort(candidates.begin(), candidates.end(), [](const auto & lhs, const auto & rhs) {
                    return lhs.score != rhs.score ? lhs.score > rhs.score : lhs.key.expert < rhs.key.expert;
                });
                break;
            }
            case LLAMA_EXPERT_PREFETCH_POLICY_RANDOM_BASELINE: {
                std::vector<int32_t> eligible(profile->target.experts_per_layer);
                for (uint32_t index = 0; index < eligible.size(); ++index) eligible[index] = int32_t(index);
                uint64_t state = config.digest;
                hash_append(state, profile->fold_index);
                hash_append(state, request.request_ordinal);
                hash_append(state, token_ordinal);
                hash_append(state, uint32_t(target_layer));
                for (size_t index = 0; index < eligible.size(); ++index) {
                    const size_t selected = index + splitmix64(state)%(eligible.size() - index);
                    std::swap(eligible[index], eligible[selected]);
                }
                for (int32_t expert : eligible) candidates.push_back({{target_layer, expert}, 0, 0});
                break;
            }
            case LLAMA_EXPERT_PREFETCH_POLICY_CROSS_LAYER_TRANSITION:
            case LLAMA_EXPERT_PREFETCH_POLICY_OFF:
            case LLAMA_EXPERT_PREFETCH_POLICY_COUNT:
                return llm_expert_prefetch_result::failure(llm_expert_prefetch_error::invalid_configuration);
        }
        if (config.value.policy != LLAMA_EXPERT_PREFETCH_POLICY_TEMPORAL_FREQUENCY) {
            std::stable_sort(candidates.begin(), candidates.end(), [](const auto & lhs, const auto & rhs) {
                return lhs.score != rhs.score ? lhs.score > rhs.score : lhs.key.expert < rhs.key.expert;
            });
        }
        if (candidates.size() > config.value.candidates_per_target) candidates.resize(config.value.candidates_per_target);
        for (uint32_t rank = 0; rank < candidates.size(); ++rank) candidates[rank].rank = rank;
        return llm_expert_prefetch_result::success();
    } catch (...) {
        candidates.clear();
        return llm_expert_prefetch_result::failure(llm_expert_prefetch_error::overflow);
    }
}

llm_expert_prefetch_result llm_expert_prefetch_predictor::predict_cross_layer(
        uint64_t token_ordinal,
        int32_t source_layer,
        const int32_t * source_experts,
        size_t source_count,
        int32_t target_layer,
        std::vector<llm_expert_prefetch_candidate> & candidates) const noexcept {
    candidates.clear();
    if (profile == nullptr || config.value.policy != LLAMA_EXPERT_PREFETCH_POLICY_CROSS_LAYER_TRANSITION ||
        request.request_ordinal == 0 || token_ordinal != request.completed_tokens || source_experts == nullptr ||
        source_count == 0 || source_count > profile->target.experts_per_token) {
        return llm_expert_prefetch_result::failure(llm_expert_prefetch_error::invalid_configuration);
    }
    const auto source_layer_it = std::find(profile->target.routed_layers.begin(), profile->target.routed_layers.end(), source_layer);
    if (source_layer_it == profile->target.routed_layers.end() || source_layer_it + 1 == profile->target.routed_layers.end() ||
        *(source_layer_it + 1) != target_layer) return llm_expert_prefetch_result::failure(llm_expert_prefetch_error::invalid_profile);
    try {
        std::vector<int32_t> sources(source_experts, source_experts + source_count);
        std::sort(sources.begin(), sources.end());
        if (std::adjacent_find(sources.begin(), sources.end()) != sources.end()) {
            return llm_expert_prefetch_result::failure(llm_expert_prefetch_error::invalid_profile);
        }
        std::vector<uint64_t> scores(profile->target.experts_per_layer);
        for (int32_t source : sources) {
            if (source < 0 || uint32_t(source) >= profile->target.experts_per_layer) {
                return llm_expert_prefetch_result::failure(llm_expert_prefetch_error::invalid_profile);
            }
            for (const auto & transition : profile->transitions) {
                if (transition.source_layer != source_layer || transition.source_expert != source ||
                    transition.target_layer != target_layer) continue;
                uint64_t updated = 0;
                scores[transition.target_expert] = checked_add(scores[transition.target_expert], transition.count, updated) ? updated : UINT64_MAX;
            }
        }
        for (uint32_t expert = 0; expert < scores.size(); ++expert) {
            if (scores[expert] != 0) candidates.push_back({{target_layer, int32_t(expert)}, 0, scores[expert]});
        }
        std::sort(candidates.begin(), candidates.end(), [](const auto & lhs, const auto & rhs) {
            return lhs.score != rhs.score ? lhs.score > rhs.score : lhs.key.expert < rhs.key.expert;
        });
        if (candidates.size() > config.value.candidates_per_target) candidates.resize(config.value.candidates_per_target);
        for (uint32_t rank = 0; rank < candidates.size(); ++rank) candidates[rank].rank = rank;
        return llm_expert_prefetch_result::success();
    } catch (...) {
        candidates.clear();
        return llm_expert_prefetch_result::failure(llm_expert_prefetch_error::overflow);
    }
}
