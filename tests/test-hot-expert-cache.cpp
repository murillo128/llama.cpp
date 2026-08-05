#include "llama-expert-weight-provider.h"
#include "llama-expert-async-io.h"
#include "llama-expert-scheduler.h"
#include "llama-expert-storage.h"
#include "llama-context.h"
#include "llama-model.h"

#include "ggml-alloc.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <iostream>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if !defined(_WIN32)
#include <unistd.h>
#endif

std::atomic<uint64_t> allocation_count { 0 };

void * operator new(std::size_t size) {
    allocation_count.fetch_add(1, std::memory_order_relaxed);
    if (void * memory = std::malloc(size)) {
        return memory;
    }
    throw std::bad_alloc();
}

void * operator new[](std::size_t size) {
    return ::operator new(size);
}

void operator delete(void * memory) noexcept {
    std::free(memory);
}

void operator delete[](void * memory) noexcept {
    std::free(memory);
}

void operator delete(void * memory, std::size_t) noexcept {
    std::free(memory);
}

void operator delete[](void * memory, std::size_t) noexcept {
    std::free(memory);
}

namespace {

struct phase10_provider_evidence {
    uint64_t seed_entries = 0;
    uint64_t seed_storage_bytes = 0;
    uint64_t seed_h2d_bytes = 0;
    bool seed_ordinary_lru = false;
    bool seed_failure_rolled_back = false;
    bool seed_scheduler_drained = false;
    uint32_t parallel_misses = 0;
    uint32_t parallel_enqueued_before_take = 0;
    uint32_t parallel_submitted_before_wait = 0;
    uint32_t parallel_ready_before_use = 0;
    uint32_t serial_submitted_before_wait = 0;
    bool routes_equal = false;
    bool joined_same_generation = false;
    bool joined_submitted_ready = false;
    bool deferred_retry_multi_key_ordered = false;
    bool hot_speculative_victim_order = false;
};

phase10_provider_evidence phase10_evidence;

struct tensor_fixture {
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buffer;
    ggml_tensor * up = nullptr;
    ggml_tensor * up_bias = nullptr;
    ggml_tensor * up_scale = nullptr;
    ggml_tensor * gate = nullptr;
    ggml_tensor * gate_bias = nullptr;
    ggml_tensor * gate_scale = nullptr;
    ggml_tensor * down = nullptr;
    ggml_tensor * down_bias = nullptr;
    ggml_tensor * down_scale = nullptr;
    ggml_tensor * ids = nullptr;
    int64_t n_expert = 4;
    int64_t n_expert_used = 2;
    int64_t n_tokens = 1;

    tensor_fixture(
            int64_t n_in = 8,
            int64_t n_hidden = 16,
            int64_t n_expert = 4,
            int64_t n_tokens = 1,
            int64_t n_expert_used = 2,
            ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type(),
            ggml_type type = GGML_TYPE_F32) :
        n_expert(n_expert), n_expert_used(n_expert_used), n_tokens(n_tokens) {
        ggml_init_params params = {
            /*.mem_size   =*/ ggml_tensor_overhead()*16,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ctx.reset(ggml_init(params));
        GGML_ASSERT(ctx);
        up = ggml_new_tensor_3d(ctx.get(), type, n_in, n_hidden, n_expert);
        up_bias = ggml_new_tensor_2d(ctx.get(), type, n_hidden, n_expert);
        up_scale = ggml_new_tensor_2d(ctx.get(), type, 1, n_expert);
        gate = ggml_new_tensor_3d(ctx.get(), type, n_in, n_hidden, n_expert);
        gate_bias = ggml_new_tensor_2d(ctx.get(), type, n_hidden, n_expert);
        gate_scale = ggml_new_tensor_2d(ctx.get(), type, 1, n_expert);
        down = ggml_new_tensor_3d(ctx.get(), type, n_hidden, n_in, n_expert);
        down_bias = ggml_new_tensor_2d(ctx.get(), type, n_in, n_expert);
        down_scale = ggml_new_tensor_2d(ctx.get(), type, 1, n_expert);
        ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, n_expert_used, n_tokens);
        buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft));
        GGML_ASSERT(buffer);
        uint8_t pattern = 0x10;
        for (auto * tensor : { up, up_bias, up_scale, gate, gate_bias, gate_scale, down, down_bias, down_scale }) {
            const int axis = tensor->ne[2] == n_expert ? 2 : 1;
            for (int64_t expert = 0; expert < n_expert; ++expert) {
                std::memset(static_cast<uint8_t *>(tensor->data) + expert*tensor->nb[axis],
                    pattern + expert, tensor->nb[axis]);
            }
            pattern += 0x10;
        }
    }

    llm_expert_bundle_descriptor bundle(int32_t layer) const {
        return {
            layer,
            int32_t(n_expert),
            llm_expert_projection_descriptor::from(up, up_bias, up_scale),
            llm_expert_projection_descriptor::from(gate, gate_bias, gate_scale),
            {},
            llm_expert_projection_descriptor::from(down, down_bias, down_scale),
        };
    }

    llm_expert_selection selection(int32_t layer) const {
        return { layer, int32_t(n_expert), int32_t(n_expert_used), n_tokens, ids };
    }
};

struct metadata_only_tensor_fixture {
    ggml_context_ptr ctx;
    ggml_tensor * up = nullptr;
    ggml_tensor * gate = nullptr;
    ggml_tensor * down = nullptr;
    ggml_tensor * ids = nullptr;
    int64_t n_expert = 4;
    int64_t n_expert_used = 2;

    metadata_only_tensor_fixture(
            int64_t n_in = 8,
            int64_t n_hidden = 16,
            ggml_type type = GGML_TYPE_F16) {
        ggml_init_params params = {
            /*.mem_size   =*/ ggml_tensor_overhead()*8,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ctx.reset(ggml_init(params));
        GGML_ASSERT(ctx);
        up = ggml_new_tensor_3d(ctx.get(), type, n_in, n_hidden, n_expert);
        gate = ggml_new_tensor_3d(ctx.get(), type, n_in, n_hidden, n_expert);
        down = ggml_new_tensor_3d(ctx.get(), type, n_hidden, n_in, n_expert);
        ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, n_expert_used, 1);
        GGML_ASSERT(up->data == nullptr && up->buffer == nullptr);
        GGML_ASSERT(gate->data == nullptr && gate->buffer == nullptr);
        GGML_ASSERT(down->data == nullptr && down->buffer == nullptr);
    }

    llm_expert_bundle_descriptor bundle(int32_t layer) const {
        return {
            layer,
            int32_t(n_expert),
            llm_expert_projection_descriptor::from(up, nullptr, nullptr),
            llm_expert_projection_descriptor::from(gate, nullptr, nullptr),
            {},
            llm_expert_projection_descriptor::from(down, nullptr, nullptr),
        };
    }

    llm_expert_selection selection(int32_t layer) const {
        return { layer, int32_t(n_expert), int32_t(n_expert_used), 1, ids };
    }

    uint64_t payload_bytes() const {
        return ggml_nbytes(up) + ggml_nbytes(gate) + ggml_nbytes(down);
    }
};

llm_hot_cache_config test_config(
        uint32_t capacity = 2,
        uint32_t routed_layers = 1,
        uint32_t total_keys = 4,
        uint32_t n_expert_used = 2,
        uint64_t initial_generation = 0) {
    llm_hot_cache_config result;
    result.capacity = capacity;
    result.n_expert_used = n_expert_used;
    result.routed_layer_count = routed_layers;
    result.total_expert_keys = total_keys;
    result.target_buffer_type = ggml_backend_cpu_buffer_type();
    result.allow_non_cuda_target_for_testing = true;
    result.initial_slot_generation_for_testing = initial_generation;
    return result;
}

llm_hot_cache_config cold_test_config(uint32_t capacity = 2) {
    static const bool loaded = [] { ggml_backend_load_all(); return true; }();
    (void) loaded;
    auto result = test_config(capacity, 1, 4, 2);
    result.cold_mode = true;
    result.cold_cache_bytes = 1U << 20;
    result.transfer_ring_bytes = 1U << 20;
    result.target_device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    result.force_pageable_transfer_for_testing = true;
    GGML_ASSERT(result.target_device);
    return result;
}

llm_hot_cache_config descriptor_only_cold_test_config(uint32_t routed_layers = 1) {
    auto result = cold_test_config();
    result.routed_layer_count = routed_layers;
    result.total_expert_keys = routed_layers*4;
    result.descriptor_only_source_for_testing = true;
    return result;
}

uint64_t expert_payload_bytes(const tensor_fixture & tensors) {
    uint64_t result = 0;
    const auto bundle = tensors.bundle(0);
    for (const auto * projection : { &bundle.up, &bundle.gate, &bundle.gate_up, &bundle.down }) {
        size_t member_index = 0;
        for (const auto * tensor : { projection->weight, projection->bias, projection->scale }) {
            if (tensor != nullptr) {
                const int axis = member_index == 0 ? 2 : 1;
                result += tensor->nb[axis];
            }
            member_index++;
        }
    }
    return result;
}

struct temporary_expert_storage_file {
    std::string path;
    std::vector<uint8_t> bytes;
    std::vector<std::vector<llm_expert_storage_span>> bundles;

    explicit temporary_expert_storage_file(const tensor_fixture & tensors) : bundles(size_t(tensors.n_expert)) {
#if defined(_WIN32)
        char name[L_tmpnam];
        GGML_ASSERT(std::tmpnam(name) != nullptr);
        path = name;
        FILE * file = std::fopen(path.c_str(), "wb");
#else
        char name[] = "/tmp/llama-phase10-experts-XXXXXX";
        const int fd = mkstemp(name);
        GGML_ASSERT(fd >= 0);
        path = name;
        FILE * file = fdopen(fd, "wb");
#endif
        GGML_ASSERT(file != nullptr);
        uint64_t file_offset = 0;
        const auto source = tensors.bundle(0);
        for (int32_t expert = 0; expert < tensors.n_expert; ++expert) {
            uint64_t destination_offset = 0;
            const auto append_member = [&](const ggml_tensor * tensor,
                    llm_expert_storage_projection projection,
                    llm_expert_storage_sidecar sidecar,
                    bool weight) {
                if (tensor == nullptr) return;
                int axis = weight ? 2 : -1;
                if (!weight) {
                    for (int candidate = 0; candidate < ggml_n_dims(tensor); ++candidate) {
                        if (tensor->ne[candidate] == tensors.n_expert) {
                            GGML_ASSERT(axis == -1);
                            axis = candidate;
                        }
                    }
                }
                GGML_ASSERT(axis >= 0);
                const uint64_t extent = tensor->nb[axis];
                const auto * source_bytes =
                    static_cast<const uint8_t *>(tensor->data) + uint64_t(expert)*extent;
                bytes.insert(bytes.end(), source_bytes, source_bytes + extent);
                GGML_ASSERT(std::fwrite(source_bytes, 1, size_t(extent), file) == extent);
                bundles[size_t(expert)].push_back({
                    0, file_offset, extent, projection, sidecar, destination_offset, extent,
                });
                file_offset += extent;
                destination_offset += extent;
            };
            const auto append_projection = [&](const llm_expert_projection_descriptor & projection,
                    llm_expert_storage_projection identity) {
                append_member(projection.weight, identity, llm_expert_storage_sidecar::weight, true);
                append_member(projection.bias, identity, llm_expert_storage_sidecar::bias, false);
                append_member(projection.scale, identity, llm_expert_storage_sidecar::scale, false);
            };
            append_projection(source.up, llm_expert_storage_projection::up);
            append_projection(source.gate, llm_expert_storage_projection::gate);
            append_projection(source.gate_up, llm_expert_storage_projection::gate_up);
            append_projection(source.down, llm_expert_storage_projection::down);
        }
        GGML_ASSERT(std::fclose(file) == 0);
    }

    ~temporary_expert_storage_file() { std::remove(path.c_str()); }

    void populate(llm_expert_storage & storage) const {
        for (int32_t expert = 0; expert < int32_t(bundles.size()); ++expert) {
            GGML_ASSERT(storage.add_bundle({ 0, expert }, bundles[size_t(expert)]).is_ready());
        }
        GGML_ASSERT(storage.seal().is_ready());
    }
};

struct failing_async_reader final : llm_expert_async_read_override {
    const std::vector<uint8_t> & bytes;
    size_t successful_reads_before_failure = 0;
    size_t calls = 0;

    failing_async_reader(const std::vector<uint8_t> & bytes, size_t successful_reads) :
        bytes(bytes), successful_reads_before_failure(successful_reads) {}

    int64_t read_at(
            intptr_t,
            void * data,
            size_t size,
            uint64_t offset,
            int & native_error) noexcept override {
        if (calls++ >= successful_reads_before_failure) {
            native_error = EIO;
            return -1;
        }
        if (offset > bytes.size() || size > bytes.size() - size_t(offset)) return 0;
        std::memcpy(data, bytes.data() + offset, size);
        return int64_t(size);
    }
};

struct gated_async_reader final : llm_expert_async_read_override {
    const std::vector<uint8_t> & bytes;
    uint64_t gated_offset = 0;
    std::atomic<bool> release { false };
    std::atomic<uint64_t> entered { 0 };

    gated_async_reader(const std::vector<uint8_t> & bytes, uint64_t gated_offset) :
        bytes(bytes), gated_offset(gated_offset) {}

    int64_t read_at(
            intptr_t,
            void * data,
            size_t size,
            uint64_t offset,
            int &) noexcept override {
        entered.fetch_add(1, std::memory_order_release);
        while (offset >= gated_offset && !release.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        if (offset > bytes.size() || size > bytes.size() - size_t(offset)) return 0;
        std::memcpy(data, bytes.data() + offset, size);
        return int64_t(size);
    }
};

llm_hot_cache_config blocking_seed_config(const tensor_fixture & tensors) {
    auto result = test_config(2);
    result.prefetch_config.supplied = true;
    result.prefetch_config.digest = 0x1234;
    result.prefetch_config.value.version = LLAMA_EXPERT_PREFETCH_VERSION_1;
    result.prefetch_config.value.struct_size = sizeof(llama_expert_prefetch_config_v1);
    result.prefetch_config.value.policy = LLAMA_EXPERT_PREFETCH_POLICY_OFF;
    result.prefetch_config.value.readiness = LLAMA_EXPERT_PREFETCH_READINESS_DEVICE_READY;
    result.prefetch_config.value.seed_mode = LLAMA_EXPERT_PREFETCH_SEED_MODE_BLOCKING_HOT;
    result.prefetch_config.value.max_profile_bytes = 4096;
    result.prefetch_profile_loaded = true;
    result.prefetch_profile.profile_sha256 = std::string(64, 'a');
    result.prefetch_profile.target.experts_per_layer = uint32_t(tensors.n_expert);
    result.prefetch_profile.target.routed_layers = { 0 };
    const uint64_t payload = expert_payload_bytes(tensors);
    // Runtime loader canonical form is increasing (count, layer, expert).
    result.prefetch_profile.seed = {
        { 0, 2, 40, payload, payload },
        { 0, 0, 100, payload, payload },
    };
    return result;
}

llm_hot_cache_config phase10_issue_ahead_config(
        llm_expert_storage & storage,
        llm_expert_async_transport & transport,
        llm_expert_scheduler & scheduler,
        bool serial_control,
        uint32_t hot_capacity = 4) {
    auto result = cold_test_config(hot_capacity);
    result.storage = &storage;
    result.async_transport = &transport;
    result.scheduler = &scheduler;
    result.trace_capacity = 64;
    result.phase10_lead_trace = true;
    result.phase10_serial_issue_for_testing = serial_control;
    result.prefetch_config.supplied = true;
    result.prefetch_config.digest = 0x5678;
    auto & prefetch = result.prefetch_config.value;
    prefetch.version = LLAMA_EXPERT_PREFETCH_VERSION_1;
    prefetch.struct_size = sizeof(llama_expert_prefetch_config_v1);
    prefetch.policy = LLAMA_EXPERT_PREFETCH_POLICY_STATIC_LAYER;
    prefetch.readiness = LLAMA_EXPERT_PREFETCH_READINESS_DEVICE_READY;
    prefetch.seed_mode = LLAMA_EXPERT_PREFETCH_SEED_MODE_OFF;
    prefetch.candidates_per_target = 1;
    prefetch.max_profile_bytes = 4096;
    prefetch.max_speculative_flights = 4;
    prefetch.max_speculative_storage_bytes_in_flight = 1U << 20;
    prefetch.max_speculative_h2d_bytes_in_flight = 1U << 20;
    prefetch.max_speculative_storage_bytes_per_token = 1U << 20;
    prefetch.max_speculative_h2d_bytes_per_token = 1U << 20;
    prefetch.max_speculative_cold_slots = 4;
    prefetch.max_speculative_hot_slots = hot_capacity;
    prefetch.utility_window_predictions = 8;
    prefetch.utility_min_observations = 1;
    result.prefetch_profile_loaded = true;
    result.prefetch_profile.profile_sha256 = std::string(64, 'b');
    result.prefetch_profile.target.experts_per_layer = 4;
    result.prefetch_profile.target.experts_per_token = 2;
    result.prefetch_profile.target.routed_layers = { 0 };
    result.prefetch_profile.selected_transport = "buffered";
    result.prefetch_profile.selected_readiness =
        LLAMA_EXPERT_PREFETCH_READINESS_DEVICE_READY;
    result.prefetch_profile.costs.push_back({
        "buffered", LLAMA_EXPERT_PREFETCH_READINESS_DEVICE_READY,
        10, 10, 1, 1, 1, 1, 1, 1, 1000, 8, 1, 1,
    });
    return result;
}

llm_hot_cache_config predictive_static_config(
        llm_expert_storage & storage,
        llm_expert_async_transport & transport,
        llm_expert_scheduler & scheduler,
        llama_expert_prefetch_readiness readiness = LLAMA_EXPERT_PREFETCH_READINESS_HOST_READY,
        uint32_t hot_capacity = 2) {
    auto result = phase10_issue_ahead_config(
        storage, transport, scheduler, false, hot_capacity);
    auto & prefetch = result.prefetch_config.value;
    prefetch.policy = LLAMA_EXPERT_PREFETCH_POLICY_STATIC_LAYER;
    prefetch.readiness = readiness;
    prefetch.candidates_per_target = 2;
    result.prefetch_profile.selected_policy = "static-layer";
    result.prefetch_profile.selected_readiness =
        readiness;
    result.prefetch_profile.static_counts = {
        { 0, 2, 20 },
        { 0, 3, 10 },
    };
    result.prefetch_profile.costs.clear();
    result.prefetch_profile.costs.push_back({
        "buffered", readiness,
        10, 10, 1, 1, 1, 1, 1,
        readiness == LLAMA_EXPERT_PREFETCH_READINESS_DEVICE_READY ? 1U : 0U,
        1000, 8, 1, 1,
    });
    return result;
}

llm_hot_cache_config predictive_hot_circuit_config() {
    auto result = test_config(4, 1, 4, 2);
    result.trace_capacity = 64;
    result.prefetch_config.supplied = true;
    result.prefetch_config.digest = 0xdef0;
    auto & prefetch = result.prefetch_config.value;
    prefetch.version = LLAMA_EXPERT_PREFETCH_VERSION_1;
    prefetch.struct_size = sizeof(llama_expert_prefetch_config_v1);
    prefetch.policy = LLAMA_EXPERT_PREFETCH_POLICY_STATIC_LAYER;
    prefetch.readiness = LLAMA_EXPERT_PREFETCH_READINESS_DEVICE_READY;
    prefetch.seed_mode = LLAMA_EXPERT_PREFETCH_SEED_MODE_OFF;
    prefetch.candidates_per_target = 2;
    prefetch.max_profile_bytes = 4096;
    prefetch.max_speculative_flights = 2;
    prefetch.max_speculative_storage_bytes_in_flight = 1U << 20;
    prefetch.max_speculative_h2d_bytes_in_flight = 1U << 20;
    prefetch.max_speculative_storage_bytes_per_token = 1U << 20;
    prefetch.max_speculative_h2d_bytes_per_token = 1U << 20;
    prefetch.max_speculative_cold_slots = 2;
    prefetch.max_speculative_hot_slots = 2;
    prefetch.utility_window_predictions = 2;
    prefetch.utility_min_observations = 2;
    result.prefetch_profile_loaded = true;
    result.prefetch_profile.profile_sha256 = std::string(64, 'd');
    result.prefetch_profile.target.experts_per_layer = 4;
    result.prefetch_profile.target.experts_per_token = 2;
    result.prefetch_profile.target.routed_layers = { 0 };
    result.prefetch_profile.selected_policy = "static-layer";
    result.prefetch_profile.selected_transport = "resident";
    result.prefetch_profile.selected_readiness =
        LLAMA_EXPERT_PREFETCH_READINESS_DEVICE_READY;
    result.prefetch_profile.static_counts = {
        { 0, 2, 20 },
        { 0, 3, 10 },
    };
    result.prefetch_profile.costs.push_back({
        "resident", LLAMA_EXPERT_PREFETCH_READINESS_DEVICE_READY,
        10, 10, 1, 1, 1, 1, 0, 1, 1000, 2, 2, 1,
    });
    return result;
}

llm_hot_cache_config predictive_hot_cross_config() {
    auto result = test_config(4, 2, 8, 2);
    result.trace_capacity = 64;
    result.prefetch_config.supplied = true;
    result.prefetch_config.digest = 0xe123;
    auto & prefetch = result.prefetch_config.value;
    prefetch.version = LLAMA_EXPERT_PREFETCH_VERSION_1;
    prefetch.struct_size = sizeof(llama_expert_prefetch_config_v1);
    prefetch.policy = LLAMA_EXPERT_PREFETCH_POLICY_CROSS_LAYER_TRANSITION;
    prefetch.readiness = LLAMA_EXPERT_PREFETCH_READINESS_DEVICE_READY;
    prefetch.candidates_per_target = 2;
    prefetch.max_profile_bytes = 4096;
    prefetch.max_speculative_flights = 2;
    prefetch.max_speculative_storage_bytes_in_flight = 1U << 20;
    prefetch.max_speculative_h2d_bytes_in_flight = 1U << 20;
    prefetch.max_speculative_storage_bytes_per_token = 1U << 20;
    prefetch.max_speculative_h2d_bytes_per_token = 1U << 20;
    prefetch.max_speculative_cold_slots = 2;
    prefetch.max_speculative_hot_slots = 2;
    prefetch.utility_window_predictions = 8;
    prefetch.utility_min_observations = 1;
    result.prefetch_profile_loaded = true;
    result.prefetch_profile.profile_sha256 = std::string(64, 'e');
    result.prefetch_profile.target.experts_per_layer = 4;
    result.prefetch_profile.target.experts_per_token = 2;
    result.prefetch_profile.target.routed_layers = { 0, 1 };
    result.prefetch_profile.selected_policy = "cross-layer-transition";
    result.prefetch_profile.selected_transport = "resident";
    result.prefetch_profile.selected_readiness =
        LLAMA_EXPERT_PREFETCH_READINESS_DEVICE_READY;
    result.prefetch_profile.transitions = {
        { 0, 0, 1, 2, 20 },
        { 0, 1, 1, 3, 10 },
    };
    result.prefetch_profile.costs.push_back({
        "resident", LLAMA_EXPERT_PREFETCH_READINESS_DEVICE_READY,
        10, 10, 1, 1, 1, 1, 0, 1, 1000, 8, 1, 1,
    });
    return result;
}

llm_hot_cache_config predictive_hot_multilayer_circuit_config() {
    auto result = predictive_hot_circuit_config();
    result.capacity = 12;
    result.routed_layer_count = 3;
    result.total_expert_keys = 12;
    result.prefetch_config.value.max_speculative_flights = 6;
    result.prefetch_config.value.max_speculative_cold_slots = 6;
    result.prefetch_config.value.max_speculative_hot_slots = 6;
    result.prefetch_profile.target.routed_layers = { 0, 1, 2 };
    result.prefetch_profile.static_counts = {
        { 0, 2, 20 }, { 0, 3, 10 },
        { 1, 2, 20 }, { 1, 3, 10 },
        { 2, 2, 20 }, { 2, 3, 10 },
    };
    return result;
}

template<typename F>
void expect_invalid(F && fn) {
    bool rejected = false;
    try {
        fn();
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    GGML_ASSERT(rejected);
}

llm_expert_graph_binding initialize_hot_binding(
        llm_expert_weight_provider & provider,
        const tensor_fixture & tensors,
        int32_t layer = 0) {
    llm_expert_graph_binding binding;
    GGML_ASSERT(provider.bind(tensors.bundle(layer), tensors.selection(layer), binding).is_ready());
    GGML_ASSERT(binding.bootstrap);
    GGML_ASSERT(provider.initialize_after_reserve().is_ready());
    GGML_ASSERT(provider.bind(tensors.bundle(layer), tensors.selection(layer), binding).is_ready());
    GGML_ASSERT(!binding.bootstrap);
    return binding;
}

void test_runtime_policy_adapters_and_bounded_metadata() {
    tensor_fixture tensors;
    for (const auto policy_name : { LLAMA_EXPERT_CACHE_POLICY_LRU, LLAMA_EXPERT_CACHE_POLICY_LFRU,
            LLAMA_EXPERT_CACHE_POLICY_SLRU, LLAMA_EXPERT_CACHE_POLICY_LFU_AGING }) {
        auto config = test_config();
        const llama_expert_cache_policy_config public_config = {
            LLAMA_EXPERT_CACHE_POLICY_VERSION_1,
            sizeof(llama_expert_cache_policy_config),
            policy_name,
            LLAMA_EXPERT_CACHE_POLICY_SCOPE_GLOBAL,
            policy_name == LLAMA_EXPERT_CACHE_POLICY_SLRU ? 7500u : 0u,
            LLAMA_EXPERT_CACHE_ADMISSION_ALWAYS,
            0,
            policy_name == LLAMA_EXPERT_CACHE_POLICY_LFU_AGING ? 64u : 0u,
            {},
        };
        GGML_ASSERT(llm_expert_cache_policy_copy_config(&public_config,
            llm_expert_cache_policy_tier::hot, config.hot_cache_policy_config).is_ready());
        auto provider = llm_create_hot_cache_expert_weight_provider(config);
        auto binding = initialize_hot_binding(*provider, tensors);
        llm_expert_execution_plan plan;
        int32_t execution_ids[] = { -1, -1 };
        const int32_t first[] = { 0, 1 };
        GGML_ASSERT(provider->prepare({ binding }, plan).is_ready());
        GGML_ASSERT(provider->remap_checkpoint(binding, first, 2, execution_ids).is_ready());
        plan.reset();
        const int32_t second[] = { 0, 2 };
        GGML_ASSERT(provider->prepare({ binding }, plan).is_ready());
        const uint64_t allocations_before = allocation_count.load(std::memory_order_relaxed);
        GGML_ASSERT(provider->remap_checkpoint(binding, second, 2, execution_ids).is_ready());
        GGML_ASSERT(allocation_count.load(std::memory_order_relaxed) == allocations_before);
        plan.reset();
        const auto diagnostics = provider->hot_cache_diagnostics();
        GGML_ASSERT(diagnostics.phase10_lead_event_capacity == 0);
        GGML_ASSERT(diagnostics.phase10_lead_events.empty());
        GGML_ASSERT(diagnostics.phase10_lead_events_dropped == 0);
        GGML_ASSERT(diagnostics.phase10_scheduler_event_capacity == 0);
        GGML_ASSERT(diagnostics.phase10_scheduler_events.empty());
        GGML_ASSERT(diagnostics.phase10_scheduler_events_dropped == 0);
        GGML_ASSERT(diagnostics.phase10_issue_ahead_trace_capacity == 0);
        GGML_ASSERT(diagnostics.phase10_issue_ahead_trace.empty());
        GGML_ASSERT(diagnostics.phase10_issue_ahead_events == 0);
        GGML_ASSERT(diagnostics.phase10_route_events == 0);
        GGML_ASSERT(diagnostics.phase10_route_events_dropped == 0);
        GGML_ASSERT(diagnostics.phase10_route_trace.empty());
        GGML_ASSERT(diagnostics.phase10_route_ids.empty());
        GGML_ASSERT(!diagnostics.phase10_prefetch_configured &&
            !diagnostics.phase10_seed_configured && !diagnostics.phase10_seed_complete);
        GGML_ASSERT(diagnostics.phase10_storage_event_capacity == 0);
        GGML_ASSERT(diagnostics.phase10_storage_events.empty());
        GGML_ASSERT(diagnostics.phase10_storage_events_dropped == 0);
        GGML_ASSERT(diagnostics.phase10_h2d_event_capacity == 0);
        GGML_ASSERT(diagnostics.phase10_h2d_events.empty());
        GGML_ASSERT(diagnostics.phase10_h2d_events_dropped == 0);
        GGML_ASSERT(diagnostics.policy.config.policy == policy_name);
        GGML_ASSERT(diagnostics.policy.config.supplied);
        GGML_ASSERT(diagnostics.policy.events > 0);
        GGML_ASSERT(diagnostics.policy.administration_requested_bytes > 0);
        GGML_ASSERT(diagnostics.policy.administration_actual_bytes >=
            diagnostics.policy.administration_requested_bytes);
        GGML_ASSERT(provider->trim().is_ready());
        binding = {};
        GGML_ASSERT(provider->surrender().is_ready());
    }
}

void test_predictor_runtime_has_no_heap_allocations() {
    llm_expert_prefetch_profile profile;
    profile.profile_sha256 = std::string(64, 'c');
    profile.target.experts_per_layer = 4;
    profile.target.experts_per_token = 2;
    profile.target.routed_layers = { 0, 1 };
    profile.static_counts = {
        { 0, 2, 20 },
        { 0, 3, 10 },
    };
    profile.transitions = {
        { 0, 0, 1, 2, 20 },
        { 0, 1, 1, 3, 10 },
    };
    llm_expert_prefetch_config_internal config;
    config.supplied = true;
    config.digest = 0x9abc;
    config.value.version = LLAMA_EXPERT_PREFETCH_VERSION_1;
    config.value.struct_size = sizeof(llama_expert_prefetch_config_v1);
    config.value.readiness = LLAMA_EXPERT_PREFETCH_READINESS_HOST_READY;
    config.value.candidates_per_target = 2;
    config.value.temporal_window_tokens = 2;
    config.value.max_profile_bytes = 4096;

    std::vector<std::vector<int32_t>> routes = {
        { 0, 1 },
        { 2, 3 },
    };
    std::vector<llm_expert_prefetch_candidate> candidates;
    candidates.reserve(4);
    uint64_t request = 10;
    for (const auto policy : {
            LLAMA_EXPERT_PREFETCH_POLICY_STATIC_LAYER,
            LLAMA_EXPERT_PREFETCH_POLICY_PREVIOUS_TOKEN,
            LLAMA_EXPERT_PREFETCH_POLICY_TEMPORAL_FREQUENCY,
            LLAMA_EXPERT_PREFETCH_POLICY_RANDOM_BASELINE }) {
        config.value.policy = policy;
        llm_expert_prefetch_predictor predictor;
        GGML_ASSERT(predictor.initialize(profile, config).is_ready());
        GGML_ASSERT(predictor.request_begin(request++).is_ready());
        const uint64_t allocations_before = allocation_count.load(std::memory_order_relaxed);
        GGML_ASSERT(predictor.commit_token(0, routes).is_ready());
        GGML_ASSERT(predictor.predict_token_end(0, 0, candidates).is_ready());
        GGML_ASSERT(allocation_count.load(std::memory_order_relaxed) == allocations_before);
    }

    config.value.policy = LLAMA_EXPERT_PREFETCH_POLICY_CROSS_LAYER_TRANSITION;
    llm_expert_prefetch_predictor cross;
    GGML_ASSERT(cross.initialize(profile, config).is_ready());
    GGML_ASSERT(cross.request_begin(request).is_ready());
    const int32_t sources[] = { 1, 0 };
    const uint64_t allocations_before = allocation_count.load(std::memory_order_relaxed);
    GGML_ASSERT(cross.predict_cross_layer(
        0, 0, sources, 2, 1, candidates).is_ready());
    GGML_ASSERT(allocation_count.load(std::memory_order_relaxed) == allocations_before);
}

void assert_bundle_slot_matches(
        const tensor_fixture & source,
        const llm_expert_graph_binding & target,
        int32_t expert,
        int32_t slot);
int32_t find_slot(const llm_hot_cache_diagnostics & diagnostics, int32_t expert);

void test_blocking_hot_seed_atomic_publication_and_failure() {
    tensor_fixture tensors;
    auto provider = llm_create_hot_cache_expert_weight_provider(blocking_seed_config(tensors));
    llm_expert_graph_binding binding;
    GGML_ASSERT(provider->bind(tensors.bundle(0), tensors.selection(0), binding).is_ready());
    GGML_ASSERT(binding.bootstrap);
    binding = {};
    GGML_ASSERT(provider->initialize_after_reserve().is_ready());
    auto diagnostics = provider->hot_cache_diagnostics();
    GGML_ASSERT(diagnostics.phase10_seed_configured && diagnostics.phase10_seed_complete);
    GGML_ASSERT(diagnostics.phase10_seed_attempts == 1 && diagnostics.phase10_seed_failures == 0);
    GGML_ASSERT(diagnostics.phase10_seed_entries == 2);
    GGML_ASSERT(diagnostics.phase10_seed_storage_bytes == 0);
    GGML_ASSERT(diagnostics.phase10_seed_h2d_bytes == 2*expert_payload_bytes(tensors));
    GGML_ASSERT(diagnostics.phase10_seed_last_touch.layer == 0 &&
        diagnostics.phase10_seed_last_touch.expert == 0);
    const int32_t slot_zero = find_slot(diagnostics, 0);
    const int32_t slot_two = find_slot(diagnostics, 2);
    GGML_ASSERT(slot_zero >= 0 && slot_two >= 0);
    GGML_ASSERT(diagnostics.slots[slot_zero].origin == llm_expert_residency_origin::static_seed);
    GGML_ASSERT(diagnostics.slots[slot_two].origin == llm_expert_residency_origin::static_seed);
    GGML_ASSERT(diagnostics.slots[slot_zero].refcount == 0 && diagnostics.slots[slot_two].refcount == 0);
    GGML_ASSERT(diagnostics.slots[slot_zero].last_use > diagnostics.slots[slot_two].last_use);
    GGML_ASSERT(provider->bind(tensors.bundle(0), tensors.selection(0), binding).is_ready());
    assert_bundle_slot_matches(tensors, binding, 0, slot_zero);
    assert_bundle_slot_matches(tensors, binding, 2, slot_two);
    binding = {};
    GGML_ASSERT(provider->surrender().is_ready());

    llm_expert_provider_faults faults;
    faults.fail_copy_after_tensors = 2;
    auto failing = llm_create_hot_cache_expert_weight_provider(blocking_seed_config(tensors), faults);
    GGML_ASSERT(failing->bind(tensors.bundle(0), tensors.selection(0), binding).is_ready());
    binding = {};
    GGML_ASSERT(failing->initialize_after_reserve().error == llm_expert_provider_error::copy_failed);
    diagnostics = failing->hot_cache_diagnostics();
    GGML_ASSERT(!diagnostics.phase10_seed_complete);
    GGML_ASSERT(diagnostics.phase10_seed_attempts == 1 && diagnostics.phase10_seed_failures == 1);
    GGML_ASSERT(diagnostics.effective_capacity == 0 && diagnostics.pool_bytes == 0);
    GGML_ASSERT(diagnostics.slots.empty());
    GGML_ASSERT(failing->surrender().is_ready());
}

void test_blocking_cold_seed_uses_storage_ring_and_ordinary_lru() {
    tensor_fixture tensors;
    temporary_expert_storage_file source_file(tensors);
    llama_file loader(source_file.path.c_str(), "rb");
    llm_expert_storage storage({ 1, 4, 4, 1U << 20 }, {
        { 0, &loader, 1, source_file.path.c_str() },
    });
    source_file.populate(storage);
    std::array<intptr_t, 1> handles{};
    size_t handle_count = 0;
    GGML_ASSERT(storage.copy_source_native_handles(
        handles.data(), handles.size(), handle_count).is_ready());
    GGML_ASSERT(handle_count == handles.size());

    llm_expert_scheduler scheduler({ 1, 4, 4, 2, 0 });
    llm_expert_async_config async_config;
    async_config.requested_queue_depth = 8;
    async_config.effective_hot_capacity = 2;
    async_config.request_capacity = 4;
    async_config.trace_capacity = 32;
    async_config.cold_cache_bytes = 1U << 20;
    async_config.source_file_capacity = 1;
    llm_expert_async_transport transport(async_config);
    GGML_ASSERT(transport.register_files(handles.data(), handle_count) == llm_expert_async_result::ready);

    auto config = cold_test_config(2);
    const auto seed = blocking_seed_config(tensors);
    config.prefetch_config = seed.prefetch_config;
    config.prefetch_profile = seed.prefetch_profile;
    config.prefetch_profile_loaded = true;
    config.storage = &storage;
    config.async_transport = &transport;
    config.scheduler = &scheduler;
    auto provider = llm_create_cold_cache_expert_weight_provider(config);
    llm_expert_graph_binding binding;
    GGML_ASSERT(provider->bind(tensors.bundle(0), tensors.selection(0), binding).is_ready());
    binding = {};
    GGML_ASSERT(provider->initialize_after_reserve().is_ready());
    auto diagnostics = provider->hot_cache_diagnostics();
    const uint64_t payload = expert_payload_bytes(tensors);
    GGML_ASSERT(diagnostics.phase10_seed_complete && diagnostics.phase10_seed_entries == 2);
    GGML_ASSERT(diagnostics.phase10_seed_storage_bytes == 2*payload);
    GGML_ASSERT(diagnostics.phase10_seed_h2d_bytes == 2*payload);
    GGML_ASSERT(diagnostics.cold_admissions == 2 && diagnostics.cold_source_copy_bundles == 0);
    const int32_t slot_zero = find_slot(diagnostics, 0);
    const int32_t slot_two = find_slot(diagnostics, 2);
    GGML_ASSERT(slot_zero >= 0 && slot_two >= 0);
    GGML_ASSERT(diagnostics.slots[slot_zero].has_cold_backing &&
        diagnostics.slots[slot_two].has_cold_backing);
    GGML_ASSERT(diagnostics.slots[slot_zero].origin == llm_expert_residency_origin::static_seed &&
        diagnostics.slots[slot_two].origin == llm_expert_residency_origin::static_seed);
    GGML_ASSERT(diagnostics.slots[slot_zero].last_use > diagnostics.slots[slot_two].last_use);

    GGML_ASSERT(provider->bind(tensors.bundle(0), tensors.selection(0), binding).is_ready());
    llm_expert_execution_plan plan;
    GGML_ASSERT(provider->prepare({ binding }, plan).is_ready());
    const int32_t logical_ids[] = { 1, 3 };
    int32_t execution_ids[] = { -1, -1 };
    GGML_ASSERT(provider->remap_checkpoint(binding, logical_ids, 2, execution_ids).is_ready());
    diagnostics = provider->hot_cache_diagnostics();
    GGML_ASSERT(find_slot(diagnostics, 0) < 0 && find_slot(diagnostics, 2) < 0);
    GGML_ASSERT(find_slot(diagnostics, 1) >= 0 && find_slot(diagnostics, 3) >= 0);
    phase10_evidence.seed_entries = 2;
    phase10_evidence.seed_storage_bytes = 2*payload;
    phase10_evidence.seed_h2d_bytes = 2*payload;
    phase10_evidence.seed_ordinary_lru = true;
    plan.reset();
    binding = {};
    GGML_ASSERT(provider->trim().is_ready());
    GGML_ASSERT(provider->surrender().is_ready());
    provider.reset();
    GGML_ASSERT(transport.shutdown());
    GGML_ASSERT(scheduler.diagnostics().active_requests == 0);

    // The first seed reaches the candidate caches, then the second storage
    // load fails. The transaction publishes none of the candidate mappings.
    failing_async_reader failing_reader(
        source_file.bytes, source_file.bundles[2].size());
    llm_expert_scheduler failing_scheduler({ 1, 4, 4, 2, 0 });
    async_config.read_override_for_testing = &failing_reader;
    llm_expert_async_transport failing_transport(async_config);
    GGML_ASSERT(failing_transport.register_files(handles.data(), handle_count) ==
        llm_expert_async_result::ready);
    config.async_transport = &failing_transport;
    config.scheduler = &failing_scheduler;
    auto failing = llm_create_cold_cache_expert_weight_provider(config);
    GGML_ASSERT(failing->bind(tensors.bundle(0), tensors.selection(0), binding).is_ready());
    binding = {};
    GGML_ASSERT(failing->initialize_after_reserve().error == llm_expert_provider_error::copy_failed);
    diagnostics = failing->hot_cache_diagnostics();
    GGML_ASSERT(!diagnostics.phase10_seed_complete && diagnostics.phase10_seed_failures == 1);
    GGML_ASSERT(diagnostics.effective_capacity == 0 && diagnostics.pool_bytes == 0);
    GGML_ASSERT(diagnostics.slots.empty() && diagnostics.cold_effective_slots == 0);
    phase10_evidence.seed_failure_rolled_back = true;
    GGML_ASSERT(failing->surrender().is_ready());
    failing.reset();
    GGML_ASSERT(failing_transport.shutdown());
    GGML_ASSERT(failing_scheduler.diagnostics().active_requests == 0);
    phase10_evidence.seed_scheduler_drained = true;
}

struct issue_ahead_case_result {
    llm_hot_cache_diagnostics diagnostics;
    std::array<int32_t, 4> execution_ids{};
    uint64_t demand_promotions = 0;
};

issue_ahead_case_result run_issue_ahead_case(bool serial_control, bool prequeue_speculative = false) {
    tensor_fixture tensors(8, 16, 4, 2, 2);
    temporary_expert_storage_file source_file(tensors);
    llama_file loader(source_file.path.c_str(), "rb");
    llm_expert_storage storage({ 1, 4, 4, 1U << 20 }, {
        { 0, &loader, 1, source_file.path.c_str() },
    });
    source_file.populate(storage);

    llm_expert_scheduler scheduler({
        1, 4, 16, 4, 0,
        4, 1U << 20, 1U << 20, 1U << 20, 1U << 20, 4, 4, 2,
    });
    llm_expert_async_config async_config;
    async_config.requested_queue_depth = 8;
    async_config.effective_hot_capacity = 4;
    async_config.request_capacity = 16;
    async_config.trace_capacity = 64;
    async_config.cold_cache_bytes = 1U << 20;
    async_config.source_file_capacity = 1;
    llm_expert_async_transport transport(async_config);
    std::array<intptr_t, 1> handles{};
    size_t handle_count = 0;
    GGML_ASSERT(storage.copy_source_native_handles(
        handles.data(), handles.size(), handle_count).is_ready());
    GGML_ASSERT(handle_count == handles.size());
    GGML_ASSERT(transport.register_files(handles.data(), handle_count) == llm_expert_async_result::ready);

    llm_expert_request_handle speculative_handle;
    if (prequeue_speculative) {
        llm_expert_request_metadata metadata;
        metadata.origin = llm_expert_request_origin::speculative;
        metadata.profile_digest = 0x5678;
        metadata.owner_request = 1;
        metadata.owner_token = 0;
        metadata.target_layer = 0;
        metadata.deadline_token = 1;
        metadata.reserved_storage_bytes = expert_payload_bytes(tensors);
        metadata.reserved_h2d_bytes = expert_payload_bytes(tensors);
        metadata.speculative_cold_slots = 1;
        metadata.speculative_hot_slots = 1;
        const auto speculative = scheduler.enqueue(
            { 0, 0 }, llm_expert_priority::prefetch_next,
            llm_expert_readiness::device_ready, metadata);
        GGML_ASSERT(speculative.disposition == llm_expert_schedule_disposition::admitted);
        speculative_handle = speculative.handle;
    }

    auto provider = llm_create_cold_cache_expert_weight_provider(
        phase10_issue_ahead_config(storage, transport, scheduler, serial_control));
    auto binding = initialize_hot_binding(*provider, tensors);
    llm_expert_execution_plan plan;
    GGML_ASSERT(provider->prepare({ binding }, plan).is_ready());
    const int32_t logical_ids[] = { 0, 1, 0, 1 };
    issue_ahead_case_result result;
    GGML_ASSERT(provider->remap_checkpoint(
        binding, logical_ids, 4, result.execution_ids.data()).is_ready());
    GGML_ASSERT(result.execution_ids[0] == result.execution_ids[2]);
    GGML_ASSERT(result.execution_ids[1] == result.execution_ids[3]);
    GGML_ASSERT(result.execution_ids[0] != result.execution_ids[1]);
    result.diagnostics = provider->hot_cache_diagnostics();
    result.demand_promotions = scheduler.diagnostics().demand_promotions;
    if (prequeue_speculative) {
        GGML_ASSERT(result.demand_promotions == 1);
        llm_expert_request_snapshot released;
        GGML_ASSERT(scheduler.snapshot(speculative_handle, released) ==
            llm_expert_schedule_disposition::stale_generation);
    }
    plan.reset();
    binding = {};
    GGML_ASSERT(provider->trim().is_ready());
    GGML_ASSERT(provider->surrender().is_ready());
    provider.reset();
    GGML_ASSERT(transport.shutdown());
    return result;
}

void test_cancelling_speculative_retry_attempts_all_demands_before_wait() {
    tensor_fixture tensors(8, 16, 4, 2, 2);
    temporary_expert_storage_file source_file(tensors);
    llama_file loader(source_file.path.c_str(), "rb");
    llm_expert_storage storage({ 1, 4, 4, 1U << 20 }, {
        { 0, &loader, 1, source_file.path.c_str() },
    });
    source_file.populate(storage);

    llm_expert_scheduler scheduler({
        1, 4, 16, 4, 0,
        4, 1U << 20, 1U << 20, 1U << 20, 1U << 20, 4, 4, 2,
    });
    llm_expert_async_config async_config;
    async_config.requested_queue_depth = 8;
    async_config.effective_hot_capacity = 4;
    async_config.request_capacity = 16;
    async_config.trace_capacity = 64;
    async_config.cold_cache_bytes = 1U << 20;
    async_config.source_file_capacity = 1;
    llm_expert_async_transport transport(async_config);
    std::array<intptr_t, 1> handles{};
    size_t handle_count = 0;
    GGML_ASSERT(storage.copy_source_native_handles(
        handles.data(), handles.size(), handle_count).is_ready());
    GGML_ASSERT(handle_count == handles.size());
    GGML_ASSERT(transport.register_files(handles.data(), handle_count) ==
        llm_expert_async_result::ready);

    llm_expert_request_metadata metadata;
    metadata.origin = llm_expert_request_origin::speculative;
    metadata.profile_digest = 0x5678;
    metadata.owner_request = 3;
    metadata.owner_token = 0;
    metadata.target_layer = 0;
    metadata.deadline_token = 1;
    metadata.reserved_storage_bytes = expert_payload_bytes(tensors);
    metadata.reserved_h2d_bytes = expert_payload_bytes(tensors);
    metadata.speculative_cold_slots = 1;
    metadata.speculative_hot_slots = 1;
    const auto speculative = scheduler.enqueue(
        { 0, 0 }, llm_expert_priority::prefetch_next,
        llm_expert_readiness::device_ready, metadata);
    GGML_ASSERT(speculative.disposition == llm_expert_schedule_disposition::admitted);
    llm_expert_request_snapshot selected;
    GGML_ASSERT(scheduler.take_next(selected).accepted());
    GGML_ASSERT(scheduler.transition(speculative.handle,
        llm_expert_request_state::submitting,
        llm_expert_request_state::io_in_flight) ==
        llm_expert_schedule_disposition::admitted);
    GGML_ASSERT(scheduler.begin_speculative_cancellation(speculative.handle,
        llm_expert_request_state::io_in_flight) ==
        llm_expert_schedule_disposition::admitted);

    auto provider = llm_create_cold_cache_expert_weight_provider(
        phase10_issue_ahead_config(storage, transport, scheduler, false));
    auto binding = initialize_hot_binding(*provider, tensors);
    llm_expert_execution_plan plan;
    GGML_ASSERT(provider->prepare({ binding }, plan).is_ready());
    const int32_t logical_ids[] = { 0, 1, 0, 1 };
    std::array<int32_t, 4> execution_ids = { -1, -1, -1, -1 };
    auto remap = std::async(std::launch::async, [&] {
        return provider->remap_checkpoint(
            binding, logical_ids, 4, execution_ids.data());
    });

    bool later_demand_enqueued = false;
    for (uint32_t attempt = 0; attempt < 100000; ++attempt) {
        if (scheduler.diagnostics().active_requests == 2) {
            later_demand_enqueued = true;
            break;
        }
        std::this_thread::yield();
    }
    GGML_ASSERT(scheduler.transition(speculative.handle,
        llm_expert_request_state::cancelling,
        llm_expert_request_state::draining) ==
        llm_expert_schedule_disposition::admitted);
    GGML_ASSERT(scheduler.finish(speculative.handle,
        llm_expert_request_state::cancelled) ==
        llm_expert_schedule_disposition::admitted);
    GGML_ASSERT(scheduler.release_terminal(speculative.handle) ==
        llm_expert_schedule_disposition::admitted);
    const auto remapped = remap.get();
    GGML_ASSERT(later_demand_enqueued && remapped.is_ready());
    GGML_ASSERT(execution_ids[0] == execution_ids[2] &&
        execution_ids[1] == execution_ids[3] &&
        execution_ids[0] != execution_ids[1]);

    const auto diagnostics = provider->hot_cache_diagnostics();
    GGML_ASSERT(diagnostics.phase10_issue_ahead_events == 1);
    GGML_ASSERT(diagnostics.phase10_issue_ahead_violations == 1);
    GGML_ASSERT(diagnostics.phase10_issue_ahead_trace.size() == 1);
    const auto & event = diagnostics.phase10_issue_ahead_trace.front();
    GGML_ASSERT(event.demand_misses == 2 &&
        event.scheduler_enqueue_attempts_before_first_wait == 2 &&
        event.scheduler_release_waits == 1 &&
        event.first_wait_after_all_demand_enqueue_attempts &&
        event.scheduler_enqueued_before_first_take == 2 &&
        event.first_take_after_all_demand_enqueues &&
        !event.first_wait_after_all_storage_submissions);
    phase10_evidence.deferred_retry_multi_key_ordered = true;

    plan.reset();
    binding = {};
    GGML_ASSERT(provider->trim().is_ready());
    GGML_ASSERT(provider->surrender().is_ready());
    provider.reset();
    GGML_ASSERT(transport.shutdown());
    GGML_ASSERT(scheduler.shutdown());
}

void test_predictive_runtime_late_join_same_generation() {
    tensor_fixture tensors;
    temporary_expert_storage_file source_file(tensors);
    llama_file loader(source_file.path.c_str(), "rb");
    llm_expert_storage storage({ 1, 4, 4, 1U << 20 }, {
        { 0, &loader, 1, source_file.path.c_str() },
    });
    source_file.populate(storage);

    llm_expert_scheduler scheduler({
        1, 4, 16, 2, 0,
        4, 1U << 20, 1U << 20, 1U << 20, 1U << 20, 4, 2, 2,
    });
    GGML_ASSERT(!source_file.bundles[2].empty());
    gated_async_reader reader(
        source_file.bytes, source_file.bundles[2].front().file_offset);
    llm_expert_async_config async_config;
    async_config.requested_queue_depth = 8;
    async_config.effective_hot_capacity = 2;
    async_config.request_capacity = 16;
    async_config.trace_capacity = 64;
    async_config.cold_cache_bytes = 1U << 20;
    async_config.source_file_capacity = 1;
    async_config.read_override_for_testing = &reader;
    llm_expert_async_transport transport(async_config);
    std::array<intptr_t, 1> handles{};
    size_t handle_count = 0;
    GGML_ASSERT(storage.copy_source_native_handles(
        handles.data(), handles.size(), handle_count).is_ready());
    GGML_ASSERT(handle_count == handles.size());
    GGML_ASSERT(transport.register_files(handles.data(), handle_count) ==
        llm_expert_async_result::ready);

    auto provider = llm_create_cold_cache_expert_weight_provider(
        predictive_static_config(storage, transport, scheduler));
    auto binding = initialize_hot_binding(*provider, tensors);
    llm_expert_execution_plan plan;
    GGML_ASSERT(provider->prepare({ binding }, plan).is_ready());

    const int32_t first_ids[] = { 0, 1 };
    int32_t first_execution[] = { -1, -1 };
    GGML_ASSERT(provider->remap_checkpoint(
        binding, first_ids, 2, first_execution).is_ready());
    GGML_ASSERT(first_execution[0] != first_execution[1]);
    auto scheduler_diagnostics = scheduler.diagnostics();
    auto diagnostics = provider->hot_cache_diagnostics();
    GGML_ASSERT(scheduler_diagnostics.active_speculative_flights == 2);
    GGML_ASSERT(diagnostics.phase10_prediction_events == 2);
    GGML_ASSERT(diagnostics.phase10_predictions_admitted == 2);
    GGML_ASSERT(diagnostics.phase10_prediction_trace.size() == 2);

    const int32_t second_ids[] = { 2, 3 };
    int32_t second_execution[] = { -1, -1 };
    auto remap = std::async(std::launch::async, [&] {
        return provider->remap_checkpoint(
            binding, second_ids, 2, second_execution);
    });
    bool joined_both = false;
    for (uint32_t attempt = 0; attempt < 100000; ++attempt) {
        if (scheduler.diagnostics().demand_promotions == 2) {
            joined_both = true;
            break;
        }
        std::this_thread::yield();
    }
    GGML_ASSERT(joined_both);
    reader.release.store(true, std::memory_order_release);
    GGML_ASSERT(remap.get().is_ready());
    GGML_ASSERT(second_execution[0] != second_execution[1]);

    diagnostics = provider->hot_cache_diagnostics();
    scheduler_diagnostics = scheduler.diagnostics();
    GGML_ASSERT(diagnostics.phase10_prediction_events == 4);
    GGML_ASSERT(diagnostics.phase10_predictions_admitted == 2);
    GGML_ASSERT(diagnostics.phase10_predictions_rejected == 2);
    GGML_ASSERT(diagnostics.phase10_late_joined == 2);
    GGML_ASSERT(diagnostics.phase10_timely_useful == 0);
    GGML_ASSERT(diagnostics.phase10_prediction_events_dropped == 0);
    GGML_ASSERT(!diagnostics.phase10_circuit_open);
    GGML_ASSERT(diagnostics.phase10_prediction_trace[0].outcome ==
        llm_expert_prefetch_outcome::late_joined);
    GGML_ASSERT(diagnostics.phase10_prediction_trace[1].outcome ==
        llm_expert_prefetch_outcome::late_joined);
    GGML_ASSERT(diagnostics.phase10_prediction_trace[0].demand_claimed &&
        diagnostics.phase10_prediction_trace[1].demand_claimed);
    GGML_ASSERT(scheduler_diagnostics.demand_promotions == 2);

    plan.reset();
    GGML_ASSERT(scheduler.diagnostics().active_requests == 0);
    binding = {};
    GGML_ASSERT(provider->trim().is_ready());
    GGML_ASSERT(provider->surrender().is_ready());
    provider.reset();
    GGML_ASSERT(transport.shutdown());
    GGML_ASSERT(scheduler.shutdown());
}

void test_predictive_request_end_cancels_and_drains_storage() {
    tensor_fixture tensors;
    temporary_expert_storage_file source_file(tensors);
    llama_file loader(source_file.path.c_str(), "rb");
    llm_expert_storage storage({ 1, 4, 4, 1U << 20 }, {
        { 0, &loader, 1, source_file.path.c_str() },
    });
    source_file.populate(storage);
    llm_expert_scheduler scheduler({
        1, 4, 16, 2, 0,
        4, 1U << 20, 1U << 20, 1U << 20, 1U << 20, 4, 2, 2,
    });
    GGML_ASSERT(!source_file.bundles[2].empty());
    gated_async_reader reader(
        source_file.bytes, source_file.bundles[2].front().file_offset);
    llm_expert_async_config async_config;
    async_config.requested_queue_depth = 8;
    async_config.effective_hot_capacity = 2;
    async_config.request_capacity = 16;
    async_config.trace_capacity = 64;
    async_config.cold_cache_bytes = 1U << 20;
    async_config.source_file_capacity = 1;
    async_config.read_override_for_testing = &reader;
    llm_expert_async_transport transport(async_config);
    std::array<intptr_t, 1> handles{};
    size_t handle_count = 0;
    GGML_ASSERT(storage.copy_source_native_handles(
        handles.data(), handles.size(), handle_count).is_ready());
    GGML_ASSERT(transport.register_files(handles.data(), handle_count) ==
        llm_expert_async_result::ready);
    auto provider = llm_create_cold_cache_expert_weight_provider(
        predictive_static_config(storage, transport, scheduler));
    auto binding = initialize_hot_binding(*provider, tensors);
    llm_expert_execution_plan plan;
    GGML_ASSERT(provider->prepare({ binding }, plan).is_ready());
    const int32_t logical_ids[] = { 0, 1 };
    int32_t execution_ids[] = { -1, -1 };
    GGML_ASSERT(provider->remap_checkpoint(
        binding, logical_ids, 2, execution_ids).is_ready());
    GGML_ASSERT(scheduler.diagnostics().active_speculative_flights == 2);

    auto reset = std::async(std::launch::async, [&] {
        plan.reset();
        GGML_ASSERT(provider->end_prefetch_sequence(UINT64_MAX).is_ready());
    });
    for (uint32_t attempt = 0; attempt < 100000; ++attempt) std::this_thread::yield();
    reader.release.store(true, std::memory_order_release);
    reset.get();
    const auto diagnostics = provider->hot_cache_diagnostics();
    GGML_ASSERT(diagnostics.phase10_wasted_unused == 2);
    GGML_ASSERT(diagnostics.phase10_prediction_events_dropped == 0);
    GGML_ASSERT(scheduler.diagnostics().active_requests == 0);
    GGML_ASSERT(transport.diagnostics().active_read_requests == 0);
    GGML_ASSERT(transport.diagnostics().read_requests_cancelled != 0);

    binding = {};
    GGML_ASSERT(provider->trim().is_ready());
    GGML_ASSERT(provider->surrender().is_ready());
    provider.reset();
    GGML_ASSERT(transport.shutdown());
    GGML_ASSERT(scheduler.shutdown());
}

void run_predictive_host_ready_hybrid_consumption(bool automatic) {
    tensor_fixture tensors;
    temporary_expert_storage_file source_file(tensors);
    llama_file loader(source_file.path.c_str(), "rb");
    llm_expert_storage storage({ 1, 4, 4, 1U << 20 }, {
        { 0, &loader, 1, source_file.path.c_str() },
    });
    source_file.populate(storage);
    llm_expert_scheduler scheduler({
        1, 4, 16, 2, 0,
        4, 1U << 20, 1U << 20, 1U << 20, 1U << 20, 4, 2, 2,
    });
    llm_expert_async_config async_config;
    async_config.requested_queue_depth = 8;
    async_config.effective_hot_capacity = 2;
    async_config.request_capacity = 16;
    async_config.trace_capacity = 64;
    async_config.cold_cache_bytes = 1U << 20;
    async_config.source_file_capacity = 1;
    llm_expert_async_transport transport(async_config);
    std::array<intptr_t, 1> handles{};
    size_t handle_count = 0;
    GGML_ASSERT(storage.copy_source_native_handles(
        handles.data(), handles.size(), handle_count).is_ready());
    GGML_ASSERT(transport.register_files(handles.data(), handle_count) ==
        llm_expert_async_result::ready);
    auto config = predictive_static_config(storage, transport, scheduler);
    config.miss_policy = automatic ?
        LLAMA_EXPERT_MISS_POLICY_AUTO : LLAMA_EXPERT_MISS_POLICY_CPU_FALLBACK;
    if (automatic) {
        config.auto_cost_model = {
            LLAMA_EXPERT_AUTO_COST_MODEL_VERSION_1,
            sizeof(llama_expert_auto_cost_model),
            1, 1, 1, 1,
            UINT64_C(1000000000), UINT64_C(1000000000),
            UINT64_C(1000000000), UINT64_C(1000000000),
            UINT64_C(1000000000), 1, 1,
        };
        config.auto_cost_model_digest = UINT64_C(1469598103934665603);
        const auto * bytes = reinterpret_cast<const uint8_t *>(&config.auto_cost_model);
        for (size_t index = 0; index < sizeof(config.auto_cost_model); ++index) {
            config.auto_cost_model_digest ^= bytes[index];
            config.auto_cost_model_digest *= UINT64_C(1099511628211);
        }
    }
    auto provider = llm_create_cold_cache_expert_weight_provider(config);
    auto binding = initialize_hot_binding(*provider, tensors);
    binding = {};
    ggml_init_params graph_params = { ggml_tensor_overhead()*32, nullptr, true };
    ggml_context_ptr graph_ctx(ggml_init(graph_params));
    GGML_ASSERT(graph_ctx);
    GGML_ASSERT(provider->bind_graph(
        graph_ctx.get(), tensors.bundle(0), tensors.selection(0), binding).is_ready());
    ggml_backend_buffer_ptr graph_buffer(
        ggml_backend_alloc_ctx_tensors_from_buft(
            graph_ctx.get(), ggml_backend_cpu_buffer_type()));
    GGML_ASSERT(graph_buffer);
    GGML_ASSERT(binding.hybrid && binding.checkpoint_ids != nullptr &&
        binding.cpu_execution_ids != nullptr);
    llm_expert_execution_plan plan;
    GGML_ASSERT(provider->prepare({ binding }, plan).is_ready());
    const int32_t first_ids[] = { 0, 1 };
    ggml_backend_tensor_set(binding.checkpoint_ids, first_ids, 0, sizeof(first_ids));
    GGML_ASSERT(provider->remap_checkpoint_tensor(binding).is_ready());
    bool predictions_read = false;
    for (uint32_t attempt = 0; attempt < 100000; ++attempt) {
        if (transport.diagnostics().read_requests_completed >= 2) {
            predictions_read = true;
            break;
        }
        std::this_thread::yield();
    }
    GGML_ASSERT(predictions_read);
    const int32_t second_ids[] = { 2, 3 };
    ggml_backend_tensor_set(binding.checkpoint_ids, second_ids, 0, sizeof(second_ids));
    GGML_ASSERT(provider->remap_checkpoint_tensor(binding).is_ready());
    std::array<int32_t, 2> gpu_ids{};
    std::array<int32_t, 2> cpu_ids{};
    ggml_backend_tensor_get(binding.checkpoint_ids, gpu_ids.data(), 0, sizeof(gpu_ids));
    ggml_backend_tensor_get(binding.cpu_execution_ids, cpu_ids.data(), 0, sizeof(cpu_ids));
    GGML_ASSERT(gpu_ids[0] == -1 && gpu_ids[1] == -1);
    GGML_ASSERT(cpu_ids[0] >= 0 && cpu_ids[1] >= 0 && cpu_ids[0] != cpu_ids[1]);
    const auto diagnostics = provider->hot_cache_diagnostics();
    GGML_ASSERT(diagnostics.phase10_prediction_events == 4);
    GGML_ASSERT(diagnostics.phase10_predictions_admitted == 2);
    GGML_ASSERT(diagnostics.phase10_predictions_rejected == 2);
    GGML_ASSERT(diagnostics.phase10_timely_useful == 2);
    GGML_ASSERT(diagnostics.phase10_late_joined == 0);
    GGML_ASSERT(diagnostics.cpu_execution_lanes == 4);
    GGML_ASSERT(diagnostics.gpu_execution_lanes == 0);
    GGML_ASSERT((diagnostics.auto_cpu_decisions > 0) == automatic);
    GGML_ASSERT(diagnostics.auto_gpu_decisions == 0);
    GGML_ASSERT(scheduler.diagnostics().demand_promotions == 0);

    plan.reset();
    GGML_ASSERT(scheduler.diagnostics().active_requests == 0);
    binding = {};
    GGML_ASSERT(provider->trim().is_ready());
    GGML_ASSERT(provider->surrender().is_ready());
    provider.reset();
    graph_buffer.reset();
    graph_ctx.reset();
    GGML_ASSERT(transport.shutdown());
    GGML_ASSERT(scheduler.shutdown());
}

void test_predictive_device_readiness_rejects_eventless_transport() {
    tensor_fixture tensors;
    temporary_expert_storage_file source_file(tensors);
    llama_file loader(source_file.path.c_str(), "rb");
    llm_expert_storage storage({ 1, 4, 4, 1U << 20 }, {
        { 0, &loader, 1, source_file.path.c_str() },
    });
    source_file.populate(storage);
    llm_expert_scheduler scheduler({
        1, 4, 16, 2, 0,
        4, 1U << 20, 1U << 20, 1U << 20, 1U << 20, 4, 2, 2,
    });
    llm_expert_async_config async_config;
    async_config.requested_queue_depth = 8;
    async_config.effective_hot_capacity = 4;
    async_config.request_capacity = 16;
    async_config.trace_capacity = 64;
    async_config.cold_cache_bytes = 1U << 20;
    async_config.source_file_capacity = 1;
    llm_expert_async_transport transport(async_config);
    std::array<intptr_t, 1> handles{};
    size_t handle_count = 0;
    GGML_ASSERT(storage.copy_source_native_handles(
        handles.data(), handles.size(), handle_count).is_ready());
    GGML_ASSERT(transport.register_files(handles.data(), handle_count) ==
        llm_expert_async_result::ready);
    auto provider = llm_create_cold_cache_expert_weight_provider(
        predictive_static_config(storage, transport, scheduler,
            LLAMA_EXPERT_PREFETCH_READINESS_DEVICE_READY, 4));
    auto binding = initialize_hot_binding(*provider, tensors);
    llm_expert_execution_plan plan;
    GGML_ASSERT(provider->prepare({ binding }, plan).is_ready());

    const int32_t first_ids[] = { 0, 1 };
    int32_t first_execution[] = { -1, -1 };
    GGML_ASSERT(provider->remap_checkpoint(
        binding, first_ids, 2, first_execution).is_ready());
    bool predictions_read = false;
    for (uint32_t attempt = 0; attempt < 100000; ++attempt) {
        if (transport.diagnostics().read_requests_completed >= 4) {
            predictions_read = true;
            break;
        }
        std::this_thread::yield();
    }
    GGML_ASSERT(predictions_read);

    const int32_t second_ids[] = { 2, 3 };
    int32_t second_execution[] = { -1, -1 };
    GGML_ASSERT(provider->remap_checkpoint(
        binding, second_ids, 2, second_execution).is_ready());
    const auto diagnostics = provider->hot_cache_diagnostics();
    GGML_ASSERT(diagnostics.phase10_prediction_events == 4);
    GGML_ASSERT(diagnostics.phase10_predictions_admitted == 2);
    GGML_ASSERT(diagnostics.phase10_predictions_rejected == 4);
    GGML_ASSERT(diagnostics.phase10_timely_useful == 0);
    GGML_ASSERT(diagnostics.phase10_late_joined == 0);
    GGML_ASSERT(!diagnostics.phase10_circuit_open);
    GGML_ASSERT(!diagnostics.phase10_prediction_trace[0].device_ready &&
        !diagnostics.phase10_prediction_trace[1].device_ready);
    GGML_ASSERT(diagnostics.phase10_prediction_trace[0].outcome ==
        llm_expert_prefetch_outcome::rejected);
    GGML_ASSERT(diagnostics.phase10_prediction_trace[1].outcome ==
        llm_expert_prefetch_outcome::rejected);
    GGML_ASSERT(scheduler.diagnostics().demand_promotions == 0);
    GGML_ASSERT(second_execution[0] != second_execution[1]);

    plan.reset();
    GGML_ASSERT(scheduler.diagnostics().active_requests == 0);
    binding = {};
    GGML_ASSERT(provider->trim().is_ready());
    GGML_ASSERT(provider->surrender().is_ready());
    provider.reset();
    GGML_ASSERT(transport.shutdown());
    GGML_ASSERT(scheduler.shutdown());
}

void test_predictive_hot_expiry_and_deterministic_circuit() {
    tensor_fixture tensors;
    auto provider = llm_create_hot_cache_expert_weight_provider(
        predictive_hot_circuit_config());
    auto binding = initialize_hot_binding(*provider, tensors);
    llm_expert_execution_plan plan;
    GGML_ASSERT(provider->prepare({ binding }, plan).is_ready());
    const int32_t logical_ids[] = { 0, 1 };
    int32_t execution_ids[] = { -1, -1 };

    GGML_ASSERT(provider->remap_checkpoint(
        binding, logical_ids, 2, execution_ids).is_ready());
    auto diagnostics = provider->hot_cache_diagnostics();
    GGML_ASSERT(diagnostics.phase10_prediction_events == 2);
    GGML_ASSERT(diagnostics.phase10_predictions_admitted == 2);
    GGML_ASSERT(diagnostics.phase10_wasted_unused == 0);
    GGML_ASSERT(find_slot(diagnostics, 2) >= 0 && find_slot(diagnostics, 3) >= 0);

    GGML_ASSERT(provider->remap_checkpoint(
        binding, logical_ids, 2, execution_ids).is_ready());
    diagnostics = provider->hot_cache_diagnostics();
    GGML_ASSERT(diagnostics.phase10_prediction_events == 4);
    GGML_ASSERT(diagnostics.phase10_predictions_admitted == 4);
    GGML_ASSERT(diagnostics.phase10_wasted_unused == 2);
    GGML_ASSERT(!diagnostics.phase10_circuit_open);
    GGML_ASSERT(find_slot(diagnostics, 2) >= 0 && find_slot(diagnostics, 3) >= 0);

    GGML_ASSERT(provider->remap_checkpoint(
        binding, logical_ids, 2, execution_ids).is_ready());
    diagnostics = provider->hot_cache_diagnostics();
    GGML_ASSERT(diagnostics.phase10_prediction_events == 4);
    GGML_ASSERT(diagnostics.phase10_predictions_admitted == 4);
    GGML_ASSERT(diagnostics.phase10_wasted_unused == 4);
    GGML_ASSERT(diagnostics.phase10_circuit_open);
    GGML_ASSERT(diagnostics.phase10_circuit_opens == 1);
    GGML_ASSERT(find_slot(diagnostics, 2) < 0 && find_slot(diagnostics, 3) < 0);
    GGML_ASSERT(find_slot(diagnostics, 0) >= 0 && find_slot(diagnostics, 1) >= 0);

    plan.reset();
    binding = {};
    GGML_ASSERT(provider->trim().is_ready());
    GGML_ASSERT(provider->surrender().is_ready());
}

void test_predictive_circuit_cancels_future_layer_work() {
    std::array<tensor_fixture, 3> tensors;
    auto provider = llm_create_hot_cache_expert_weight_provider(
        predictive_hot_multilayer_circuit_config());
    std::array<llm_expert_graph_binding, 3> bindings;
    for (int32_t layer = 0; layer < 3; ++layer) {
        GGML_ASSERT(provider->bind(
            tensors[layer].bundle(layer), tensors[layer].selection(layer),
            bindings[layer]).is_ready());
        GGML_ASSERT(bindings[layer].bootstrap);
        bindings[layer] = {};
    }
    GGML_ASSERT(provider->initialize_after_reserve().is_ready());
    std::vector<llm_expert_graph_binding> prepared;
    prepared.reserve(bindings.size());
    for (int32_t layer = 0; layer < 3; ++layer) {
        GGML_ASSERT(provider->bind(
            tensors[layer].bundle(layer), tensors[layer].selection(layer),
            bindings[layer]).is_ready());
        prepared.push_back(bindings[layer]);
    }
    llm_expert_execution_plan plan;
    GGML_ASSERT(provider->prepare(prepared, plan).is_ready());
    const int32_t logical_ids[] = { 0, 1 };
    int32_t execution_ids[] = { -1, -1 };
    for (int32_t layer = 0; layer < 3; ++layer) {
        GGML_ASSERT(provider->remap_checkpoint(
            bindings[layer], logical_ids, 2, execution_ids).is_ready());
    }
    auto diagnostics = provider->hot_cache_diagnostics();
    GGML_ASSERT(diagnostics.phase10_prediction_events == 6);
    GGML_ASSERT(diagnostics.phase10_predictions_admitted == 6);

    GGML_ASSERT(provider->remap_checkpoint(
        bindings[0], logical_ids, 2, execution_ids).is_ready());
    diagnostics = provider->hot_cache_diagnostics();
    GGML_ASSERT(diagnostics.phase10_wasted_unused == 2);
    GGML_ASSERT(!diagnostics.phase10_circuit_open);
    GGML_ASSERT(provider->remap_checkpoint(
        bindings[1], logical_ids, 2, execution_ids).is_ready());
    diagnostics = provider->hot_cache_diagnostics();
    GGML_ASSERT(diagnostics.phase10_wasted_unused == 4);
    GGML_ASSERT(diagnostics.phase10_cancelled_drained == 2);
    GGML_ASSERT(diagnostics.phase10_circuit_open);
    GGML_ASSERT(diagnostics.phase10_circuit_opens == 1);
    GGML_ASSERT(diagnostics.phase10_prediction_trace[4].outcome ==
        llm_expert_prefetch_outcome::cancelled_drained);
    GGML_ASSERT(diagnostics.phase10_prediction_trace[5].outcome ==
        llm_expert_prefetch_outcome::cancelled_drained);
    GGML_ASSERT(diagnostics.slots[
        diagnostics.phase10_prediction_trace[4].hot_slot].state ==
        llm_hot_cache_diagnostics::slot::free);
    GGML_ASSERT(diagnostics.slots[
        diagnostics.phase10_prediction_trace[5].hot_slot].state ==
        llm_hot_cache_diagnostics::slot::free);
    GGML_ASSERT(provider->remap_checkpoint(
        bindings[2], logical_ids, 2, execution_ids).is_ready());
    GGML_ASSERT(provider->hot_cache_diagnostics().phase10_prediction_events == 6);

    plan.reset();
    prepared.clear();
    bindings = {};
    GGML_ASSERT(provider->trim().is_ready());
    GGML_ASSERT(provider->surrender().is_ready());
}

void test_predictive_trace_overflow_disables_prediction() {
    tensor_fixture tensors;
    auto config = predictive_hot_circuit_config();
    config.trace_capacity = 2;
    config.prefetch_config.value.utility_window_predictions = 8;
    config.prefetch_profile.costs.front().utility_window_predictions = 8;
    auto provider = llm_create_hot_cache_expert_weight_provider(config);
    auto binding = initialize_hot_binding(*provider, tensors);
    llm_expert_execution_plan plan;
    GGML_ASSERT(provider->prepare({ binding }, plan).is_ready());
    const int32_t first_ids[] = { 0, 1 };
    const int32_t second_ids[] = { 2, 3 };
    int32_t execution_ids[] = { -1, -1 };
    GGML_ASSERT(provider->remap_checkpoint(
        binding, first_ids, 2, execution_ids).is_ready());
    GGML_ASSERT(provider->remap_checkpoint(
        binding, second_ids, 2, execution_ids).is_ready());
    const auto diagnostics = provider->hot_cache_diagnostics();
    GGML_ASSERT(diagnostics.phase10_prediction_events == 3);
    GGML_ASSERT(diagnostics.phase10_prediction_events_dropped == 1);
    GGML_ASSERT(diagnostics.phase10_prediction_trace.size() == 2);
    GGML_ASSERT(diagnostics.phase10_predictions_admitted == 2);
    GGML_ASSERT(diagnostics.phase10_timely_useful == 2);
    GGML_ASSERT(diagnostics.phase10_circuit_open);
    GGML_ASSERT(diagnostics.phase10_circuit_opens == 1);
    plan.reset();
    binding = {};
    GGML_ASSERT(provider->trim().is_ready());
    GGML_ASSERT(provider->surrender().is_ready());
}

void test_predictive_hot_slot_budget_saturation() {
    tensor_fixture tensors;
    auto config = predictive_hot_circuit_config();
    config.prefetch_config.value.max_speculative_flights = 1;
    config.prefetch_config.value.max_speculative_hot_slots = 1;
    config.prefetch_config.value.utility_window_predictions = 8;
    config.prefetch_profile.costs.front().utility_window_predictions = 8;
    auto provider = llm_create_hot_cache_expert_weight_provider(config);
    auto binding = initialize_hot_binding(*provider, tensors);
    llm_expert_execution_plan plan;
    GGML_ASSERT(provider->prepare({ binding }, plan).is_ready());
    const int32_t logical_ids[] = { 0, 1 };
    int32_t execution_ids[] = { -1, -1 };
    GGML_ASSERT(provider->remap_checkpoint(
        binding, logical_ids, 2, execution_ids).is_ready());
    const auto diagnostics = provider->hot_cache_diagnostics();
    GGML_ASSERT(diagnostics.phase10_prediction_events == 2);
    GGML_ASSERT(diagnostics.phase10_predictions_admitted == 1);
    GGML_ASSERT(diagnostics.phase10_predictions_rejected == 1);
    GGML_ASSERT(find_slot(diagnostics, 2) >= 0);
    GGML_ASSERT(find_slot(diagnostics, 3) < 0);
    plan.reset();
    binding = {};
    GGML_ASSERT(provider->trim().is_ready());
    GGML_ASSERT(provider->surrender().is_ready());
}

void test_predictive_cross_layer_router_trigger_and_prefill_exclusion() {
    tensor_fixture layer_zero;
    tensor_fixture layer_one;
    auto provider = llm_create_hot_cache_expert_weight_provider(
        predictive_hot_cross_config());
    llm_expert_graph_binding binding_zero;
    llm_expert_graph_binding binding_one;
    GGML_ASSERT(provider->bind(
        layer_zero.bundle(0), layer_zero.selection(0), binding_zero).is_ready());
    GGML_ASSERT(provider->bind(
        layer_one.bundle(1), layer_one.selection(1), binding_one).is_ready());
    GGML_ASSERT(binding_zero.bootstrap && binding_one.bootstrap);
    binding_zero = {};
    binding_one = {};
    GGML_ASSERT(provider->initialize_after_reserve().is_ready());
    GGML_ASSERT(provider->bind(
        layer_zero.bundle(0), layer_zero.selection(0), binding_zero).is_ready());
    GGML_ASSERT(provider->bind(
        layer_one.bundle(1), layer_one.selection(1), binding_one).is_ready());
    llm_expert_execution_plan plan;
    GGML_ASSERT(provider->prepare({ binding_zero, binding_one }, plan).is_ready());

    const int32_t source_ids[] = { 1, 0 };
    int32_t source_execution[] = { -1, -1 };
    GGML_ASSERT(provider->remap_checkpoint(
        binding_zero, source_ids, 2, source_execution).is_ready());
    auto diagnostics = provider->hot_cache_diagnostics();
    GGML_ASSERT(diagnostics.phase10_prediction_events == 2);
    GGML_ASSERT(diagnostics.phase10_predictions_admitted == 2);
    GGML_ASSERT(diagnostics.phase10_prediction_trace[0].trigger ==
        llm_expert_prefetch_trigger::router_result);
    GGML_ASSERT(diagnostics.phase10_prediction_trace[1].trigger ==
        llm_expert_prefetch_trigger::router_result);
    GGML_ASSERT(diagnostics.phase10_prediction_trace[0].key.layer == 1 &&
        diagnostics.phase10_prediction_trace[0].key.expert == 2);
    GGML_ASSERT(diagnostics.phase10_prediction_trace[1].key.layer == 1 &&
        diagnostics.phase10_prediction_trace[1].key.expert == 3);

    const int32_t target_ids[] = { 2, 3 };
    int32_t target_execution[] = { -1, -1 };
    GGML_ASSERT(provider->remap_checkpoint(
        binding_one, target_ids, 2, target_execution).is_ready());
    diagnostics = provider->hot_cache_diagnostics();
    GGML_ASSERT(diagnostics.phase10_prediction_events == 2);
    GGML_ASSERT(diagnostics.phase10_timely_useful == 2);
    GGML_ASSERT(target_execution[0] ==
        int32_t(diagnostics.phase10_prediction_trace[0].hot_slot));
    GGML_ASSERT(target_execution[1] ==
        int32_t(diagnostics.phase10_prediction_trace[1].hot_slot));
    plan.reset();
    binding_zero = {};
    binding_one = {};
    GGML_ASSERT(provider->trim().is_ready());
    GGML_ASSERT(provider->surrender().is_ready());

    tensor_fixture prefill(8, 16, 4, 2, 2);
    auto prefill_provider = llm_create_hot_cache_expert_weight_provider(
        predictive_hot_circuit_config());
    auto prefill_binding = initialize_hot_binding(*prefill_provider, prefill);
    GGML_ASSERT(prefill_provider->prepare({ prefill_binding }, plan).is_ready());
    const int32_t prefill_ids[] = { 0, 1, 0, 1 };
    int32_t prefill_execution[] = { -1, -1, -1, -1 };
    GGML_ASSERT(prefill_provider->remap_checkpoint(
        prefill_binding, prefill_ids, 4, prefill_execution).is_ready());
    diagnostics = prefill_provider->hot_cache_diagnostics();
    GGML_ASSERT(diagnostics.phase10_prediction_events == 0);
    GGML_ASSERT(diagnostics.phase10_predictions_admitted == 0);
    plan.reset();
    prefill_binding = {};
    GGML_ASSERT(prefill_provider->trim().is_ready());
    GGML_ASSERT(prefill_provider->surrender().is_ready());
}

void test_predictive_reversed_layer_callbacks_are_canonical() {
    tensor_fixture layer_zero;
    tensor_fixture layer_one;
    auto provider = llm_create_hot_cache_expert_weight_provider(
        predictive_hot_cross_config());
    llm_expert_graph_binding bindings[2];
    GGML_ASSERT(provider->bind(
        layer_zero.bundle(0), layer_zero.selection(0), bindings[0]).is_ready());
    GGML_ASSERT(provider->bind(
        layer_one.bundle(1), layer_one.selection(1), bindings[1]).is_ready());
    bindings[0] = {};
    bindings[1] = {};
    GGML_ASSERT(provider->initialize_after_reserve().is_ready());
    GGML_ASSERT(provider->bind(
        layer_zero.bundle(0), layer_zero.selection(0), bindings[0]).is_ready());
    GGML_ASSERT(provider->bind(
        layer_one.bundle(1), layer_one.selection(1), bindings[1]).is_ready());
    llm_expert_execution_plan plan;
    GGML_ASSERT(provider->prepare({ bindings[0], bindings[1] }, plan).is_ready());

    const int32_t source_ids[] = { 1, 0 };
    const int32_t target_ids[] = { 3, 2 };
    int32_t execution_ids[] = { -1, -1 };
    for (uint64_t token = 0; token < 2; ++token) {
        GGML_ASSERT(provider->remap_checkpoint(
            bindings[1], target_ids, 2, execution_ids).is_ready());
        if (token == 0) {
            const auto before_prefix = provider->hot_cache_diagnostics();
            GGML_ASSERT(before_prefix.phase10_route_events == 0);
            GGML_ASSERT(before_prefix.phase10_prediction_events == 0);
        }
        GGML_ASSERT(provider->remap_checkpoint(
            bindings[0], source_ids, 2, execution_ids).is_ready());
    }
    const auto diagnostics = provider->hot_cache_diagnostics();
    GGML_ASSERT(!diagnostics.phase10_runtime_failed);
    GGML_ASSERT(diagnostics.phase10_route_events == 4);
    GGML_ASSERT(diagnostics.phase10_prediction_events == 4);
    GGML_ASSERT(diagnostics.phase10_route_trace.size() == 4);
    for (size_t index = 0; index < diagnostics.phase10_route_trace.size(); ++index) {
        GGML_ASSERT(diagnostics.phase10_route_trace[index].token == index/2);
        GGML_ASSERT(diagnostics.phase10_route_trace[index].layer == int32_t(index%2));
    }
    GGML_ASSERT(diagnostics.phase10_prediction_trace[0].trigger ==
        llm_expert_prefetch_trigger::router_result);
    GGML_ASSERT(diagnostics.phase10_prediction_trace[0].key.layer == 1);

    plan.reset();
    bindings[0] = {};
    bindings[1] = {};
    GGML_ASSERT(provider->trim().is_ready());
    GGML_ASSERT(provider->surrender().is_ready());
}

void test_exact_issue_ahead_and_serial_evidence_control() {
    const auto parallel = run_issue_ahead_case(false);
    const auto serial = run_issue_ahead_case(true);
    const auto joined = run_issue_ahead_case(false, true);
    GGML_ASSERT(parallel.execution_ids == serial.execution_ids);
    GGML_ASSERT(parallel.diagnostics.phase10_issue_ahead_events == 1);
    GGML_ASSERT(parallel.diagnostics.phase10_issue_ahead_violations == 0);
    GGML_ASSERT(parallel.diagnostics.phase10_issue_ahead_trace.size() == 1);
    GGML_ASSERT(serial.diagnostics.phase10_issue_ahead_trace.size() == 1);
    const auto & issued = parallel.diagnostics.phase10_issue_ahead_trace.front();
    const auto & control = serial.diagnostics.phase10_issue_ahead_trace.front();
    GGML_ASSERT(issued.logical_ids == 4 && issued.unique_ids == 2 && issued.demand_misses == 2);
    GGML_ASSERT(issued.scheduler_enqueue_attempts_before_first_wait == 2 &&
        issued.scheduler_release_waits == 0 &&
        issued.first_wait_after_all_demand_enqueue_attempts &&
        issued.scheduler_enqueued_before_first_take == 2 &&
        issued.first_take_after_all_demand_enqueues);
    GGML_ASSERT(issued.storage_reads == 2 &&
        issued.storage_reads_submitted_before_first_wait == 2 &&
        issued.first_wait_after_all_storage_submissions && !issued.serial_control);
    GGML_ASSERT(issued.demand_ready_before_use == 2 && issued.all_demand_ready_before_use);
    GGML_ASSERT(issued.first_take_us != 0 && issued.first_wait_us >= issued.first_take_us &&
        issued.last_demand_ready_us >= issued.first_wait_us &&
        issued.demand_completion_us >= issued.last_demand_ready_us);
    GGML_ASSERT(control.logical_ids == issued.logical_ids && control.unique_ids == issued.unique_ids &&
        control.demand_misses == issued.demand_misses && control.storage_reads == issued.storage_reads);
    GGML_ASSERT(control.scheduler_enqueue_attempts_before_first_wait == 2 &&
        control.scheduler_release_waits == 0 &&
        control.first_wait_after_all_demand_enqueue_attempts &&
        control.scheduler_enqueued_before_first_take == 2 &&
        control.first_take_after_all_demand_enqueues);
    GGML_ASSERT(control.storage_reads_submitted_before_first_wait == 1 &&
        !control.first_wait_after_all_storage_submissions && control.serial_control);
    GGML_ASSERT(control.demand_ready_before_use == 2 && control.all_demand_ready_before_use);
    GGML_ASSERT(parallel.diagnostics.phase10_storage_events.size() >= 2);
    GGML_ASSERT(serial.diagnostics.phase10_storage_events.size() >= 2);
    phase10_evidence.parallel_misses = issued.demand_misses;
    phase10_evidence.parallel_enqueued_before_take = issued.scheduler_enqueued_before_first_take;
    phase10_evidence.parallel_submitted_before_wait = issued.storage_reads_submitted_before_first_wait;
    phase10_evidence.parallel_ready_before_use = issued.demand_ready_before_use;
    phase10_evidence.serial_submitted_before_wait = control.storage_reads_submitted_before_first_wait;
    phase10_evidence.routes_equal = parallel.execution_ids == serial.execution_ids;
    GGML_ASSERT(joined.execution_ids == parallel.execution_ids && joined.demand_promotions == 1);
    phase10_evidence.joined_same_generation = true;
}

void test_exact_demand_joins_submitted_same_generation_with_ready_cold_data() {
    tensor_fixture tensors(8, 16, 4, 1, 2);
    temporary_expert_storage_file source_file(tensors);
    llama_file loader(source_file.path.c_str(), "rb");
    llm_expert_storage storage({ 1, 4, 4, 1U << 20 }, {
        { 0, &loader, 1, source_file.path.c_str() },
    });
    source_file.populate(storage);
    llm_expert_scheduler scheduler({
        1, 4, 16, 4, 0,
        4, 1U << 20, 1U << 20, 1U << 20, 1U << 20, 4, 2, 2,
    });
    llm_expert_async_config async_config;
    async_config.requested_queue_depth = 8;
    async_config.effective_hot_capacity = 2;
    async_config.request_capacity = 16;
    async_config.trace_capacity = 64;
    async_config.cold_cache_bytes = 16U << 20;
    async_config.source_file_capacity = 1;
    llm_expert_async_transport transport(async_config);
    std::array<intptr_t, 1> handles{};
    size_t handle_count = 0;
    GGML_ASSERT(storage.copy_source_native_handles(
        handles.data(), handles.size(), handle_count).is_ready());
    GGML_ASSERT(handle_count == handles.size());
    GGML_ASSERT(transport.register_files(handles.data(), handle_count) ==
        llm_expert_async_result::ready);

    auto provider_config = phase10_issue_ahead_config(storage, transport, scheduler, false, 2);
    provider_config.cold_cache_bytes = 16U << 20;
    auto provider = llm_create_cold_cache_expert_weight_provider(provider_config);
    auto binding = initialize_hot_binding(*provider, tensors);
    llm_expert_execution_plan plan;
    GGML_ASSERT(provider->prepare({ binding }, plan).is_ready());
    std::array<int32_t, 2> execution{};
    const int32_t first[] = { 0, 1 };
    GGML_ASSERT(provider->remap_checkpoint(binding, first, 2, execution.data()).is_ready());
    const int32_t evict[] = { 2, 3 };
    GGML_ASSERT(provider->remap_checkpoint(binding, evict, 2, execution.data()).is_ready());
    GGML_ASSERT(!provider->debug_hot_mapping({ 0, 0 }));

    llm_expert_request_metadata metadata;
    metadata.origin = llm_expert_request_origin::speculative;
    metadata.profile_digest = 0x5678;
    metadata.owner_request = 2;
    metadata.owner_token = 0;
    metadata.target_layer = 0;
    metadata.deadline_token = 1;
    metadata.reserved_storage_bytes = expert_payload_bytes(tensors);
    metadata.reserved_h2d_bytes = expert_payload_bytes(tensors);
    metadata.speculative_cold_slots = 1;
    metadata.speculative_hot_slots = 1;
    const auto speculative = scheduler.enqueue(
        { 0, 0 }, llm_expert_priority::prefetch_next,
        llm_expert_readiness::device_ready, metadata);
    GGML_ASSERT(speculative.disposition == llm_expert_schedule_disposition::admitted);
    llm_expert_request_snapshot selected;
    GGML_ASSERT(scheduler.take_next(selected).accepted());
    GGML_ASSERT(scheduler.transition(speculative.handle,
        llm_expert_request_state::submitting,
        llm_expert_request_state::io_in_flight) ==
        llm_expert_schedule_disposition::admitted);

    const int32_t demand[] = { 0, 0 };
    GGML_ASSERT(provider->remap_checkpoint(binding, demand, 2, execution.data()).is_ready());
    GGML_ASSERT(execution[0] == execution[1]);
    llm_expert_request_snapshot promoted;
    GGML_ASSERT(scheduler.snapshot(speculative.handle, promoted) ==
        llm_expert_schedule_disposition::admitted);
    GGML_ASSERT(promoted.promoted_from_speculative &&
        promoted.metadata.origin == llm_expert_request_origin::demand &&
        promoted.handle.generation == speculative.handle.generation &&
        promoted.state == llm_expert_request_state::io_in_flight);
    GGML_ASSERT(scheduler.transition(speculative.handle,
        llm_expert_request_state::io_in_flight,
        llm_expert_request_state::draining) ==
        llm_expert_schedule_disposition::admitted);
    GGML_ASSERT(scheduler.finish(speculative.handle,
        llm_expert_request_state::failed) ==
        llm_expert_schedule_disposition::admitted);
    GGML_ASSERT(scheduler.release_terminal(speculative.handle) ==
        llm_expert_schedule_disposition::admitted);
    phase10_evidence.joined_submitted_ready = true;

    plan.reset();
    binding = {};
    GGML_ASSERT(provider->trim().is_ready());
    GGML_ASSERT(provider->surrender().is_ready());
    provider.reset();
    GGML_ASSERT(transport.shutdown());
}

void test_hot_speculative_victim_deadline_utility_slot_order() {
    GGML_ASSERT(llm_expert_speculative_victim_precedes(4, 100, 3, 5, 0, 0));
    GGML_ASSERT(!llm_expert_speculative_victim_precedes(6, 0, 0, 5, 100, 3));
    GGML_ASSERT(llm_expert_speculative_victim_precedes(5, 40, 3, 5, 50, 0));
    GGML_ASSERT(!llm_expert_speculative_victim_precedes(5, 60, 0, 5, 50, 3));
    GGML_ASSERT(llm_expert_speculative_victim_precedes(5, 50, 2, 5, 50, 3));
    GGML_ASSERT(!llm_expert_speculative_victim_precedes(5, 50, 4, 5, 50, 3));
    phase10_evidence.hot_speculative_victim_order = true;
}

int source_expert_axis(const ggml_tensor * tensor, int64_t n_expert, bool weight) {
    if (weight) {
        GGML_ASSERT(tensor->ne[2] == n_expert);
        return 2;
    }
    int result = -1;
    for (int axis = 0; axis < ggml_n_dims(tensor); ++axis) {
        if (tensor->ne[axis] == n_expert) {
            GGML_ASSERT(result == -1);
            result = axis;
        }
    }
    GGML_ASSERT(result >= 0);
    return result;
}

void assert_tensor_slot_matches(
        const ggml_tensor * source,
        const ggml_tensor * target,
        int64_t n_expert,
        int32_t expert,
        int32_t slot,
        bool weight) {
    GGML_ASSERT((source == nullptr) == (target == nullptr));
    if (source == nullptr) {
        return;
    }
    const int axis = source_expert_axis(source, n_expert, weight);
    const size_t span = source->nb[axis];
    GGML_ASSERT(target->nb[axis] >= span);
    const auto * source_bytes = static_cast<const uint8_t *>(source->data) + size_t(expert)*span;
    const auto * target_bytes = static_cast<const uint8_t *>(target->data) + size_t(slot)*target->nb[axis];
    GGML_ASSERT(std::memcmp(source_bytes, target_bytes, span) == 0);
}

void assert_projection_slot_matches(
        const llm_expert_projection_descriptor & source,
        const llm_expert_projection_descriptor & target,
        int64_t n_expert,
        int32_t expert,
        int32_t slot) {
    assert_tensor_slot_matches(source.weight, target.weight, n_expert, expert, slot, true);
    assert_tensor_slot_matches(source.bias, target.bias, n_expert, expert, slot, false);
    assert_tensor_slot_matches(source.scale, target.scale, n_expert, expert, slot, false);
}

void assert_bundle_slot_matches(
        const tensor_fixture & source,
        const llm_expert_graph_binding & target,
        int32_t expert,
        int32_t slot) {
    const auto bundle = source.bundle(target.layer);
    assert_projection_slot_matches(bundle.up, target.up, source.n_expert, expert, slot);
    assert_projection_slot_matches(bundle.gate, target.gate, source.n_expert, expert, slot);
    assert_projection_slot_matches(bundle.gate_up, target.gate_up, source.n_expert, expert, slot);
    assert_projection_slot_matches(bundle.down, target.down, source.n_expert, expert, slot);
}

int32_t find_slot(const llm_hot_cache_diagnostics & diagnostics, int32_t expert) {
    for (size_t slot = 0; slot < diagnostics.slots.size(); ++slot) {
        if (diagnostics.slots[slot].expert == expert) {
            return int32_t(slot);
        }
    }
    return -1;
}

struct eval_callback_counts {
    uint64_t asks = 0;
    uint64_t observations = 0;
};

struct route_capture {
    std::vector<int32_t> layers;
    std::vector<int32_t> ids;
    std::vector<float> weights;
};

bool capture_routes(const llama_route_observation * observation, void * user_data) {
    auto * capture = static_cast<route_capture *>(user_data);
    capture->layers.push_back(observation->layer);
    const size_t count = size_t(observation->n_tokens)*observation->n_expert_used;
    capture->ids.insert(capture->ids.end(), observation->selected_experts,
        observation->selected_experts + count);
    capture->weights.insert(capture->weights.end(), observation->weights,
        observation->weights + count);
    return true;
}

bool count_execution_id_callbacks(ggml_tensor * tensor, bool ask, void * user_data) {
    auto * counts = static_cast<eval_callback_counts *>(user_data);
    if (ask) {
        counts->asks++;
        return std::strncmp(tensor->name, "expert_execution_ids-", 21) == 0;
    }
    counts->observations++;
    return true;
}

void test_initialization_stage_and_descriptor_only_scale() {
    auto hot = llm_create_hot_cache_expert_weight_provider(test_config());
    GGML_ASSERT(hot->initialization_stage() ==
        llm_expert_provider_initialization_stage::workspace_before_persistent_pool);

    {
        auto concurrent = llm_create_cold_cache_expert_weight_provider(descriptor_only_cold_test_config());
        GGML_ASSERT(concurrent->initialization_stage() ==
            llm_expert_provider_initialization_stage::descriptors_before_scheduler_reserve);
        bool owner = false;
        GGML_ASSERT(concurrent->begin_initialization(
            llm_expert_provider_initialization_stage::descriptors_before_scheduler_reserve, owner).is_ready());
        GGML_ASSERT(owner);
        bool second_owner = false;
        GGML_ASSERT(concurrent->begin_initialization(
            llm_expert_provider_initialization_stage::descriptors_before_scheduler_reserve,
            second_owner).error == llm_expert_provider_error::busy);
        GGML_ASSERT(!second_owner);
        GGML_ASSERT(concurrent->finish_initialization(false).is_ready());
    }

    // These tensors describe 6 GiB of routed payload but own no data or backend
    // buffer.  Two graph variants register the same descriptors without any
    // scheduler reservation or backend allocation.
    metadata_only_tensor_fixture large(16384, 16384);
    GGML_ASSERT(large.payload_bytes() > UINT64_C(4)*1024*1024*1024);
    auto scale = llm_create_cold_cache_expert_weight_provider(descriptor_only_cold_test_config());
    bool owner = false;
    GGML_ASSERT(scale->begin_initialization(
        llm_expert_provider_initialization_stage::descriptors_before_scheduler_reserve, owner).is_ready());
    GGML_ASSERT(owner);
    llm_expert_graph_binding binding;
    GGML_ASSERT(scale->bind(large.bundle(0), large.selection(0), binding).is_ready());
    GGML_ASSERT(binding.bootstrap && binding.generation_lease == nullptr);
    binding = {};
    GGML_ASSERT(scale->bind(large.bundle(0), large.selection(0), binding).is_ready());
    binding = {};
    GGML_ASSERT(scale->complete_descriptor_discovery(2, 2, 0, { 0, 0 }, { 0, 0 }).is_ready());
    auto scale_diagnostics = scale->hot_cache_diagnostics();
    GGML_ASSERT(scale_diagnostics.descriptor_discovery_graphs == 2);
    GGML_ASSERT(scale_diagnostics.descriptor_discovery_backend_bytes_before == 0);
    GGML_ASSERT(scale_diagnostics.descriptor_discovery_backend_bytes_after == 0);
    GGML_ASSERT(scale_diagnostics.effective_capacity == 0 && scale_diagnostics.pool_bytes == 0);
    GGML_ASSERT(scale->finish_initialization(false).is_ready());

    // A small metadata-only source proves the final fixed-capacity hot/cold
    // bindings are distinct from the all-expert descriptor tensors.
    metadata_only_tensor_fixture small;
    auto bounded_config = descriptor_only_cold_test_config();
    bounded_config.miss_policy = LLAMA_EXPERT_MISS_POLICY_CPU_FALLBACK;
    auto bounded = llm_create_cold_cache_expert_weight_provider(bounded_config);
    owner = false;
    GGML_ASSERT(bounded->begin_initialization(
        llm_expert_provider_initialization_stage::descriptors_before_scheduler_reserve, owner).is_ready());
    GGML_ASSERT(owner);
    GGML_ASSERT(bounded->bind(small.bundle(0), small.selection(0), binding).is_ready());
    binding = {};
    GGML_ASSERT(bounded->bind(small.bundle(0), small.selection(0), binding).is_ready());
    binding = {};
    GGML_ASSERT(bounded->complete_descriptor_discovery(2, 2, 0, { 0 }, { 0 }).is_ready());
    GGML_ASSERT(bounded->initialize_after_reserve().is_ready());

    ggml_init_params graph_params = { ggml_tensor_overhead()*16, nullptr, true };
    ggml_context_ptr graph_ctx(ggml_init(graph_params));
    GGML_ASSERT(graph_ctx);
    GGML_ASSERT(bounded->bind_graph(
        graph_ctx.get(), small.bundle(0), small.selection(0), binding).is_ready());
    GGML_ASSERT(!binding.bootstrap && binding.hybrid && binding.generation_lease != nullptr);
    GGML_ASSERT(binding.up.weight != small.up && binding.cpu_up.weight != small.up);
    GGML_ASSERT(bounded->record_initialization_telemetry(
        { 0 }, { 512 }, 0, small.payload_bytes()).is_ready());
    GGML_ASSERT(bounded->finish_initialization(true).is_ready());
    const auto bounded_diagnostics = bounded->hot_cache_diagnostics();
    GGML_ASSERT(bounded_diagnostics.pool_bytes > 0);
    GGML_ASSERT(bounded_diagnostics.pool_bytes < small.payload_bytes());
    GGML_ASSERT(bounded_diagnostics.complete_deferred_payload_bytes == small.payload_bytes());
    GGML_ASSERT(!bounded_diagnostics.complete_deferred_payload_in_compute_workspace);
    GGML_ASSERT(bounded->initialization_stage() == llm_expert_provider_initialization_stage::none);
    binding = {};
    graph_ctx.reset();
    GGML_ASSERT(bounded->surrender().is_ready());

    // Incomplete, duplicate-identity, and incompatible-logical discovery all
    // fail before persistent allocation.
    auto incomplete = llm_create_cold_cache_expert_weight_provider(descriptor_only_cold_test_config(2));
    owner = false;
    GGML_ASSERT(incomplete->begin_initialization(
        llm_expert_provider_initialization_stage::descriptors_before_scheduler_reserve, owner).is_ready());
    GGML_ASSERT(incomplete->bind(small.bundle(0), small.selection(0), binding).is_ready());
    binding = {};
    GGML_ASSERT(incomplete->complete_descriptor_discovery(2, 2, 0, { 0 }, { 0 }).error ==
        llm_expert_provider_error::initialization_failed);
    GGML_ASSERT(incomplete->hot_cache_diagnostics().pool_bytes == 0);
    GGML_ASSERT(incomplete->finish_initialization(false).is_ready());

    metadata_only_tensor_fixture incompatible_layout(8, 17);
    auto incompatible = llm_create_cold_cache_expert_weight_provider(descriptor_only_cold_test_config(2));
    owner = false;
    GGML_ASSERT(incompatible->begin_initialization(
        llm_expert_provider_initialization_stage::descriptors_before_scheduler_reserve, owner).is_ready());
    GGML_ASSERT(incompatible->bind(small.bundle(0), small.selection(0), binding).is_ready());
    binding = {};
    GGML_ASSERT(incompatible->bind(
        incompatible_layout.bundle(1), incompatible_layout.selection(1), binding).is_ready());
    binding = {};
    GGML_ASSERT(incompatible->bind(small.bundle(0), small.selection(0), binding).is_ready());
    binding = {};
    GGML_ASSERT(incompatible->bind(
        incompatible_layout.bundle(1), incompatible_layout.selection(1), binding).is_ready());
    binding = {};
    GGML_ASSERT(incompatible->complete_descriptor_discovery(
        2, 4, 0, { 0, 0 }, { 0, 0 }).error ==
        llm_expert_provider_error::unsupported_configuration);
    GGML_ASSERT(incompatible->hot_cache_diagnostics().pool_bytes == 0);
    GGML_ASSERT(incompatible->finish_initialization(false).is_ready());

    metadata_only_tensor_fixture duplicate_identity(8, 16);
    auto duplicate = llm_create_cold_cache_expert_weight_provider(descriptor_only_cold_test_config());
    owner = false;
    GGML_ASSERT(duplicate->begin_initialization(
        llm_expert_provider_initialization_stage::descriptors_before_scheduler_reserve, owner).is_ready());
    GGML_ASSERT(duplicate->bind(small.bundle(0), small.selection(0), binding).is_ready());
    binding = {};
    GGML_ASSERT(duplicate->bind(
        duplicate_identity.bundle(0), duplicate_identity.selection(0), binding).error ==
        llm_expert_provider_error::invalid_descriptor);
    GGML_ASSERT(duplicate->hot_cache_diagnostics().pool_bytes == 0);
    GGML_ASSERT(duplicate->finish_initialization(false).is_ready());

    for (size_t failure_after = 2; failure_after <= 4; ++failure_after) {
        llm_expert_provider_faults faults;
        faults.binding = llm_expert_provider_error::invalid_descriptor;
        faults.binding_successes_before_failure = failure_after;
        auto failing = llm_create_cold_cache_expert_weight_provider(
            bounded_config, faults);
        owner = false;
        GGML_ASSERT(failing->begin_initialization(
            llm_expert_provider_initialization_stage::descriptors_before_scheduler_reserve, owner).is_ready());
        GGML_ASSERT(failing->bind(small.bundle(0), small.selection(0), binding).is_ready());
        binding = {};
        GGML_ASSERT(failing->bind(small.bundle(0), small.selection(0), binding).is_ready());
        binding = {};
        GGML_ASSERT(failing->complete_descriptor_discovery(2, 2, 0, { 0 }, { 0 }).is_ready());
        GGML_ASSERT(failing->initialize_after_reserve().is_ready());
        std::vector<llm_expert_graph_binding> leases;
        for (size_t graph = 2; graph < failure_after; ++graph) {
            GGML_ASSERT(failing->bind(small.bundle(0), small.selection(0), binding).is_ready());
            leases.push_back(std::move(binding));
        }
        GGML_ASSERT(failing->bind(small.bundle(0), small.selection(0), binding).error ==
            llm_expert_provider_error::invalid_descriptor);
        binding = {};
        leases.clear();
        GGML_ASSERT(failing->surrender().is_ready());
        GGML_ASSERT(failing->hot_cache_diagnostics().pool_bytes == 0);
        GGML_ASSERT(failing->finish_initialization(false).is_ready());
    }
}

void test_layout_class_determinism_and_bounded_cap() {
    metadata_only_tensor_fixture f32(8, 16, GGML_TYPE_F32);
    metadata_only_tensor_fixture f16(8, 16, GGML_TYPE_F16);
    metadata_only_tensor_fixture bf16(8, 16, GGML_TYPE_BF16);
    const std::array<const metadata_only_tensor_fixture *, 3> fixtures = { &f32, &f16, &bf16 };
    const auto discover = [&](const std::array<int32_t, 3> & order) {
        auto provider = llm_create_cold_cache_expert_weight_provider(
            descriptor_only_cold_test_config(3));
        bool owner = false;
        GGML_ASSERT(provider->begin_initialization(
            llm_expert_provider_initialization_stage::descriptors_before_scheduler_reserve,
            owner).is_ready() && owner);
        llm_expert_graph_binding binding;
        for (uint32_t graph = 0; graph < 2; ++graph) {
            for (int32_t layer : order) {
                GGML_ASSERT(provider->bind(
                    fixtures[size_t(layer)]->bundle(layer),
                    fixtures[size_t(layer)]->selection(layer), binding).is_ready());
                binding = {};
            }
        }
        GGML_ASSERT(provider->complete_descriptor_discovery(
            2, 6, 0, { 0, 0, 0 }, { 0, 0, 0 }).is_ready());
        auto diagnostics = provider->hot_cache_diagnostics();
        GGML_ASSERT(provider->finish_initialization(false).is_ready());
        return diagnostics;
    };
    const auto forward = discover({ 0, 1, 2 });
    const auto reverse = discover({ 2, 1, 0 });
    GGML_ASSERT(forward.layout_class_count == 3 && reverse.layout_class_count == 3);
    GGML_ASSERT(forward.layout_class_digests == reverse.layout_class_digests);
    GGML_ASSERT(forward.layout_class_payload_bytes == reverse.layout_class_payload_bytes);
    GGML_ASSERT(forward.layout_layer_ids == reverse.layout_layer_ids);

    const std::array<ggml_type, 9> types = {
        GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16,
        GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0,
        GGML_TYPE_Q5_1, GGML_TYPE_Q8_0, GGML_TYPE_Q6_K,
    };
    std::array<std::unique_ptr<metadata_only_tensor_fixture>, 9> too_many;
    auto provider = llm_create_cold_cache_expert_weight_provider(
        descriptor_only_cold_test_config(uint32_t(too_many.size())));
    bool owner = false;
    GGML_ASSERT(provider->begin_initialization(
        llm_expert_provider_initialization_stage::descriptors_before_scheduler_reserve,
        owner).is_ready() && owner);
    llm_expert_graph_binding binding;
    for (size_t layer = 0; layer < too_many.size(); ++layer) {
        too_many[layer] = std::make_unique<metadata_only_tensor_fixture>(256, 256, types[layer]);
        GGML_ASSERT(provider->bind(
            too_many[layer]->bundle(int32_t(layer)),
            too_many[layer]->selection(int32_t(layer)), binding).is_ready());
        binding = {};
    }
    for (size_t layer = 0; layer < too_many.size(); ++layer) {
        GGML_ASSERT(provider->bind(
            too_many[layer]->bundle(int32_t(layer)),
            too_many[layer]->selection(int32_t(layer)), binding).is_ready());
        binding = {};
    }
    GGML_ASSERT(provider->complete_descriptor_discovery(
        2, 2*too_many.size(), 0,
        std::vector<uint64_t>(too_many.size(), 0),
        std::vector<uint64_t>(too_many.size(), 0)).error ==
        llm_expert_provider_error::unsupported_configuration);
    GGML_ASSERT(provider->hot_cache_diagnostics().layout_class_count == 0);
    GGML_ASSERT(provider->finish_initialization(false).is_ready());
}

void test_configuration_matrix() {
    expect_invalid([] { llm_create_hot_cache_expert_weight_provider(test_config(0)); });
    expect_invalid([] { llm_create_hot_cache_expert_weight_provider(test_config(1)); });
    expect_invalid([] { llm_create_hot_cache_expert_weight_provider(test_config(5)); });

    auto non_cuda = test_config(2);
    non_cuda.allow_non_cuda_target_for_testing = false;
    expect_invalid([&] { llm_create_hot_cache_expert_weight_provider(non_cuda); });

    GGML_ASSERT(llm_create_hot_cache_expert_weight_provider(test_config(2)) != nullptr);
    GGML_ASSERT(llm_create_hot_cache_expert_weight_provider(test_config(4)) != nullptr);

    auto hot_with_cold_budget = test_config(2);
    hot_with_cold_budget.cold_cache_bytes = 1;
    expect_invalid([&] { llm_create_hot_cache_expert_weight_provider(hot_with_cold_budget); });
    auto cold_without_budget = cold_test_config();
    cold_without_budget.transfer_ring_bytes = 0;
    expect_invalid([&] { llm_create_cold_cache_expert_weight_provider(cold_without_budget); });
}

void test_cold_provider_uses_bounded_transfer_waves() {
    tensor_fixture tensors(8, 16, 8, 1, 6);
    auto source = tensors.bundle(0);
    for (auto * projection : { &source.up, &source.gate, &source.down }) {
        projection->bias = nullptr;
        projection->scale = nullptr;
    }
    auto config = cold_test_config(6);
    config.n_expert_used = 6;
    config.total_expert_keys = 8;
    uint64_t lane_footprint = 0;
    {
        auto sizing_provider = llm_create_cold_cache_expert_weight_provider(config);
        llm_expert_graph_binding sizing_binding;
        GGML_ASSERT(sizing_provider->bind(
            source, tensors.selection(0), sizing_binding).is_ready());
        const auto initialized = sizing_provider->initialize_after_reserve();
        if (!initialized.is_ready()) {
            std::fprintf(stderr, "bounded-wave sizing initialization failed: status=%u error=%u\n",
                unsigned(initialized.status), unsigned(initialized.error));
        }
        GGML_ASSERT(initialized.is_ready());
        GGML_ASSERT(sizing_provider->bind(
            source, tensors.selection(0), sizing_binding).is_ready());
        GGML_ASSERT(sizing_binding.layout_class_id == 0);
        lane_footprint = sizing_provider->hot_cache_diagnostics().ring_lane_footprint;
        GGML_ASSERT(lane_footprint > 0);
    }

    config.transfer_ring_bytes = lane_footprint*2;
    auto provider = llm_create_cold_cache_expert_weight_provider(config);
    llm_expert_graph_binding binding;
    GGML_ASSERT(provider->bind(source, tensors.selection(0), binding).is_ready());
    GGML_ASSERT(provider->initialize_after_reserve().is_ready());
    GGML_ASSERT(provider->bind(source, tensors.selection(0), binding).is_ready());
    auto diagnostics = provider->hot_cache_diagnostics();
    GGML_ASSERT(diagnostics.ring_effective_lanes == 2);
    GGML_ASSERT(diagnostics.ring_actual_bytes == config.transfer_ring_bytes);

    llm_expert_execution_plan plan;
    GGML_ASSERT(provider->prepare({ binding }, plan).is_ready());
    const int32_t logical_ids[] = { 0, 1, 2, 3, 4, 5 };
    int32_t execution_ids[] = { -1, -1, -1, -1, -1, -1 };
    GGML_ASSERT(provider->remap_checkpoint(
        binding, logical_ids, 6, execution_ids).is_ready());
    diagnostics = provider->hot_cache_diagnostics();
    GGML_ASSERT(diagnostics.ring_waves == 3);
    GGML_ASSERT(diagnostics.ring_peak_in_flight_lanes <= 2);
    GGML_ASSERT(diagnostics.ring_live_events == 0);
    plan.reset();
}

void test_context_extent_matrix_and_prepare_revalidation() {
    auto exact_top_k = llm_create_hot_cache_expert_weight_provider(test_config(2));
    GGML_ASSERT(exact_top_k->validate_context_extent(64, 1).is_ready());
    auto diagnostics = exact_top_k->hot_cache_diagnostics();
    GGML_ASSERT(diagnostics.n_expert == 4);
    GGML_ASSERT(diagnostics.n_expert_used == 2);
    GGML_ASSERT(diagnostics.last_context_extent == 1);
    GGML_ASSERT(diagnostics.conservative_required_capacity == 2);
    GGML_ASSERT(diagnostics.effective_capacity == 0);

    const auto unsafe_exact = exact_top_k->validate_context_extent(64, 2);
    GGML_ASSERT(unsafe_exact.error == llm_expert_provider_error::unsupported_configuration);
    diagnostics = exact_top_k->hot_cache_diagnostics();
    GGML_ASSERT(diagnostics.last_context_extent == 2);
    GGML_ASSERT(diagnostics.conservative_required_capacity == 4);
    GGML_ASSERT(diagnostics.context_validations == 2);
    GGML_ASSERT(diagnostics.context_rejections == 1);
    GGML_ASSERT(diagnostics.effective_capacity == 0);

    auto capacity_three = llm_create_hot_cache_expert_weight_provider(test_config(3));
    GGML_ASSERT(capacity_three->validate_context_extent(64, 1).is_ready());
    GGML_ASSERT(capacity_three->validate_context_extent(64, 2).error ==
        llm_expert_provider_error::unsupported_configuration);

    auto all_experts = llm_create_hot_cache_expert_weight_provider(test_config(4));
    GGML_ASSERT(all_experts->validate_context_extent(64, 64).is_ready());
    diagnostics = all_experts->hot_cache_diagnostics();
    GGML_ASSERT(diagnostics.last_context_extent == 64);
    GGML_ASSERT(diagnostics.conservative_required_capacity == 4);

    tensor_fixture two_tokens(8, 16, 4, 2);
    llm_expert_graph_binding binding;
    GGML_ASSERT(exact_top_k->bind(two_tokens.bundle(0), two_tokens.selection(0), binding).is_ready());
    GGML_ASSERT(exact_top_k->initialize_after_reserve().is_ready());
    llm_expert_execution_plan unsafe_plan;
    GGML_ASSERT(exact_top_k->prepare({ binding }, unsafe_plan).error ==
        llm_expert_provider_error::unsupported_configuration);
    GGML_ASSERT(unsafe_plan.handle_count() == 0);

    GGML_ASSERT(capacity_three->bind(two_tokens.bundle(0), two_tokens.selection(0), binding).is_ready());
    GGML_ASSERT(capacity_three->initialize_after_reserve().is_ready());
    GGML_ASSERT(capacity_three->bind(two_tokens.bundle(0), two_tokens.selection(0), binding).is_ready());
    llm_expert_execution_plan capacity_three_plan;
    GGML_ASSERT(capacity_three->prepare({ binding }, capacity_three_plan).error ==
        llm_expert_provider_error::unsupported_configuration);
    GGML_ASSERT(capacity_three_plan.handle_count() == 0);

    GGML_ASSERT(all_experts->bind(two_tokens.bundle(0), two_tokens.selection(0), binding).is_ready());
    GGML_ASSERT(all_experts->initialize_after_reserve().is_ready());
    GGML_ASSERT(all_experts->bind(two_tokens.bundle(0), two_tokens.selection(0), binding).is_ready());
    llm_expert_execution_plan safe_extent_plan;
    GGML_ASSERT(all_experts->prepare({ binding }, safe_extent_plan).is_ready());
    safe_extent_plan.reset();
}

void test_cold_provider_rejects_cuda_host_source() {
    ggml_backend_load_all();
    auto * device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!device) return;
    auto * host_buft = ggml_backend_dev_host_buffer_type(device);
    if (!host_buft) return;
    tensor_fixture tensors(8, 16, 4, 1, 2, host_buft);
    auto provider = llm_create_cold_cache_expert_weight_provider(cold_test_config());
    llm_expert_graph_binding binding;
    GGML_ASSERT(provider->bind(tensors.bundle(0), tensors.selection(0), binding).error ==
        llm_expert_provider_error::unsupported_configuration);
}

void test_pool_lifetime_trim_surrender_and_epoch() {
    tensor_fixture tensors;
    auto provider = llm_create_hot_cache_expert_weight_provider(test_config());

    llm_expert_graph_binding bootstrap;
    GGML_ASSERT(provider->bind(tensors.bundle(0), tensors.selection(0), bootstrap).is_ready());
    GGML_ASSERT(bootstrap.bootstrap);
    GGML_ASSERT(!bootstrap.generation_lease);
    GGML_ASSERT(bootstrap.up.weight == tensors.up);
    GGML_ASSERT(provider->needs_post_reserve_initialization());

    const auto init_result = provider->initialize_after_reserve();
    if (!init_result.is_ready()) {
        std::cerr << "hot-cache initialization failed: status=" << int(init_result.status)
                  << " error=" << int(init_result.error) << '\n';
    }
    GGML_ASSERT(init_result.is_ready());
    GGML_ASSERT(!provider->needs_post_reserve_initialization());
    const auto initialized = provider->hot_cache_diagnostics();
    GGML_ASSERT(initialized.requested_capacity == 2);
    GGML_ASSERT(initialized.effective_capacity == 2);
    GGML_ASSERT(initialized.pool_bytes > 0);
    GGML_ASSERT(initialized.generation == 1);
    GGML_ASSERT(!initialized.slot_tensor_addresses.empty());
    GGML_ASSERT(std::all_of(initialized.slot_tensor_addresses.begin(), initialized.slot_tensor_addresses.end(),
        [](uintptr_t address) { return address != 0; }));

    llm_expert_graph_binding hot;
    GGML_ASSERT(provider->bind(tensors.bundle(0), tensors.selection(0), hot).is_ready());
    GGML_ASSERT(!hot.bootstrap);
    GGML_ASSERT(hot.generation_lease);
    GGML_ASSERT(hot.up.weight != tensors.up);
    GGML_ASSERT(hot.up.weight->ne[2] == 2);
    const auto stable = provider->hot_cache_diagnostics();
    GGML_ASSERT(stable.slot_tensor_addresses == initialized.slot_tensor_addresses);
    GGML_ASSERT(provider->trim().is_ready());
    GGML_ASSERT(provider->hot_cache_diagnostics().slot_tensor_addresses == initialized.slot_tensor_addresses);

    const auto busy = provider->surrender();
    GGML_ASSERT(busy.error == llm_expert_provider_error::busy);
    GGML_ASSERT(provider->hot_cache_diagnostics().effective_capacity == 2);
    hot = {};

    const uint64_t epoch_before_surrender = provider->graph_epoch();
    GGML_ASSERT(provider->surrender().is_ready());
    GGML_ASSERT(provider->graph_epoch() > epoch_before_surrender);
    GGML_ASSERT(provider->needs_post_reserve_initialization());
    GGML_ASSERT(provider->hot_cache_diagnostics().effective_capacity == 0);

    llm_expert_graph_binding second_bootstrap;
    GGML_ASSERT(provider->bind(tensors.bundle(0), tensors.selection(0), second_bootstrap).is_ready());
    GGML_ASSERT(second_bootstrap.bootstrap);
    GGML_ASSERT(provider->initialize_after_reserve().is_ready());
    const auto reinitialized = provider->hot_cache_diagnostics();
    GGML_ASSERT(reinitialized.generation == 2);
    GGML_ASSERT(reinitialized.graph_epoch > initialized.graph_epoch);

    const auto stats = provider->get_stats();
    GGML_ASSERT(stats.allocations == 2);
    GGML_ASSERT(stats.pool_generations == 2);
    GGML_ASSERT(stats.bootstrap_bindings == 2);
    GGML_ASSERT(stats.hot_bindings == 1);
    GGML_ASSERT(stats.trims == 1);
    GGML_ASSERT(stats.surrender_busy == 1);
    GGML_ASSERT(stats.surrender_successes == 1);
}

void test_layout_host_and_partial_initialization_rejection() {
    tensor_fixture first;
    tensor_fixture incompatible(9, 16, 4);
    auto provider = llm_create_hot_cache_expert_weight_provider(test_config(2, 2, 8));
    llm_expert_graph_binding binding;
    GGML_ASSERT(provider->bind(first.bundle(0), first.selection(0), binding).is_ready());
    binding = {};
    GGML_ASSERT(provider->bind(incompatible.bundle(1), incompatible.selection(1), binding).is_ready());
    binding = {};
    GGML_ASSERT(provider->initialize_after_reserve().error ==
        llm_expert_provider_error::unsupported_configuration);
    GGML_ASSERT(provider->hot_cache_diagnostics().effective_capacity == 0);

    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead()*4,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr unallocated_ctx(ggml_init(params));
    GGML_ASSERT(unallocated_ctx);
    ggml_tensor * up = ggml_new_tensor_3d(unallocated_ctx.get(), GGML_TYPE_F32, 8, 16, 4);
    ggml_tensor * gate = ggml_new_tensor_3d(unallocated_ctx.get(), GGML_TYPE_F32, 8, 16, 4);
    ggml_tensor * down = ggml_new_tensor_3d(unallocated_ctx.get(), GGML_TYPE_F32, 16, 8, 4);
    const llm_expert_bundle_descriptor non_host = {
        0, 4,
        llm_expert_projection_descriptor::from(up, nullptr, nullptr),
        llm_expert_projection_descriptor::from(gate, nullptr, nullptr),
        {},
        llm_expert_projection_descriptor::from(down, nullptr, nullptr),
    };
    const llm_expert_selection selection = { 0, 4, 2, 1, first.ids };
    auto host_checked = llm_create_hot_cache_expert_weight_provider(test_config());
    GGML_ASSERT(host_checked->bind(non_host, selection, binding).error ==
        llm_expert_provider_error::unsupported_configuration);
}

void test_allocation_failure_is_recoverable_and_empty_prepare_is_safe() {
    tensor_fixture tensors;
    llm_expert_provider_faults faults;
    faults.fail_pool_allocation = true;
    auto provider = llm_create_hot_cache_expert_weight_provider(test_config(), faults);
    llm_expert_graph_binding binding;
    GGML_ASSERT(provider->bind(tensors.bundle(0), tensors.selection(0), binding).is_ready());
    const auto failed = provider->initialize_after_reserve();
    GGML_ASSERT(failed.status == llm_expert_provider_status::allocation_failed);
    GGML_ASSERT(provider->needs_post_reserve_initialization());
    GGML_ASSERT(provider->hot_cache_diagnostics().effective_capacity == 0);

    llm_expert_execution_plan empty;
    GGML_ASSERT(provider->prepare({}, empty).is_ready());
    GGML_ASSERT(empty.handle_count() == 0);
}

void test_directory_hit_eviction_generation_and_copy() {
    tensor_fixture tensors(8, 16, 4, 1, 1);
    auto provider = llm_create_hot_cache_expert_weight_provider(test_config(1, 1, 4, 1));
    const auto binding = initialize_hot_binding(*provider, tensors);
    llm_expert_execution_plan plan;
    GGML_ASSERT(provider->prepare({ binding }, plan).is_ready());

    const int32_t first_ids[] = { 0 };
    int32_t execution_ids[] = { -1 };
    GGML_ASSERT(provider->remap_checkpoint(binding, first_ids, 1, execution_ids).is_ready());
    GGML_ASSERT(execution_ids[0] == 0);
    auto diagnostics = provider->hot_cache_diagnostics();
    GGML_ASSERT(diagnostics.slots[0].expert == 0);
    GGML_ASSERT(diagnostics.slots[0].generation == 1);
    GGML_ASSERT(diagnostics.slots[0].state == llm_hot_cache_diagnostics::slot::pinned);
    assert_bundle_slot_matches(tensors, binding, 0, 0);

    GGML_ASSERT(provider->remap_checkpoint(binding, first_ids, 1, execution_ids).is_ready());
    diagnostics = provider->hot_cache_diagnostics();
    GGML_ASSERT(diagnostics.slots[0].generation == 1);
    GGML_ASSERT(diagnostics.hits == 1);
    GGML_ASSERT(diagnostics.misses == 1);
    plan.reset();

    GGML_ASSERT(provider->prepare({ binding }, plan).is_ready());
    const int32_t second_ids[] = { 1 };
    GGML_ASSERT(provider->remap_checkpoint(binding, second_ids, 1, execution_ids).is_ready());
    diagnostics = provider->hot_cache_diagnostics();
    GGML_ASSERT(diagnostics.slots[0].expert == 1);
    GGML_ASSERT(diagnostics.slots[0].generation == 2);
    GGML_ASSERT(diagnostics.evictions == 1);
    GGML_ASSERT(provider->validate_slot_generation(0, 1).error ==
        llm_expert_provider_error::stale_generation);
    GGML_ASSERT(provider->validate_slot_generation(0, 2).is_ready());
    assert_bundle_slot_matches(tensors, binding, 1, 0);
    plan.reset();
}

void test_directory_multi_token_atomic_dedup_and_no_allocation() {
    tensor_fixture tensors(8, 16, 4, 2, 2);
    auto provider = llm_create_hot_cache_expert_weight_provider(test_config(4, 1, 4, 2));
    const auto binding = initialize_hot_binding(*provider, tensors);
    llm_expert_execution_plan plan;
    GGML_ASSERT(provider->prepare({ binding }, plan).is_ready());

    const int32_t logical_ids[] = { 0, 1, 2, 3 };
    int32_t execution_ids[] = { -1, -1, -1, -1 };
    const uint64_t allocations_before = allocation_count.load(std::memory_order_relaxed);
    GGML_ASSERT(provider->remap_checkpoint(binding, logical_ids, 4, execution_ids).is_ready());
    const uint64_t allocations_after = allocation_count.load(std::memory_order_relaxed);
    GGML_ASSERT(allocations_after == allocations_before);
    for (int32_t index = 0; index < 4; ++index) {
        GGML_ASSERT(execution_ids[index] == index);
        assert_bundle_slot_matches(tensors, binding, index, index);
    }

    const int32_t duplicate_ids[] = { 0, 0, 1, 1 };
    const int32_t expected_slots[] = { 0, 0, 1, 1 };
    const uint64_t duplicate_allocations_before = allocation_count.load(std::memory_order_relaxed);
    GGML_ASSERT(provider->remap_checkpoint(binding, duplicate_ids, 4, execution_ids).is_ready());
    GGML_ASSERT(allocation_count.load(std::memory_order_relaxed) == duplicate_allocations_before);
    GGML_ASSERT(std::memcmp(execution_ids, expected_slots, sizeof(expected_slots)) == 0);
    const auto diagnostics = provider->hot_cache_diagnostics();
    GGML_ASSERT(diagnostics.remap_checkpoints == 2);
    GGML_ASSERT(diagnostics.logical_ids == 8);
    GGML_ASSERT(diagnostics.unique_ids == 6);
    GGML_ASSERT(diagnostics.hits == 2);
    GGML_ASSERT(diagnostics.misses == 4);
    GGML_ASSERT(diagnostics.current_pins == 2);
    GGML_ASSERT(diagnostics.remap_dynamic_allocations == 0);
    plan.reset();
}

void test_cold_provider_inclusive_promotion_and_hits() {
    tensor_fixture tensors(8, 16, 4, 1, 2);
    auto provider = llm_create_cold_cache_expert_weight_provider(cold_test_config());
    const auto binding = initialize_hot_binding(*provider, tensors);
    llm_expert_execution_plan plan;
    GGML_ASSERT(provider->prepare({ binding }, plan).is_ready());

    const int32_t first_ids[] = { 0, 1 };
    int32_t execution_ids[] = { -1, -1 };
    const uint64_t allocations_before = allocation_count.load(std::memory_order_relaxed);
    GGML_ASSERT(provider->remap_checkpoint(binding, first_ids, 2, execution_ids).is_ready());
    GGML_ASSERT(allocation_count.load(std::memory_order_relaxed) == allocations_before);
    auto diagnostics = provider->hot_cache_diagnostics();
    GGML_ASSERT(diagnostics.misses == 2 && diagnostics.cold_misses == 2);
    GGML_ASSERT(diagnostics.cold_admissions == 2 && diagnostics.cold_source_copy_bundles == 2);
    GGML_ASSERT(diagnostics.ring_waves == 1 && diagnostics.ring_synchronous_copies == 18);
    GGML_ASSERT(diagnostics.ring_async_enqueues == 0 && diagnostics.ring_wave_synchronizations == 0);
    GGML_ASSERT(diagnostics.cold_current_hot_refs == 2);
    GGML_ASSERT(diagnostics.cold_current_transfer_refs == 0);
    GGML_ASSERT(diagnostics.source_pageable && diagnostics.source_pinned_bytes == 0);
    GGML_ASSERT(diagnostics.ring_acquisition_method == "pageable-cpu");
    GGML_ASSERT(diagnostics.ring_fallback_reason == "forced-for-testing");
    for (int32_t index = 0; index < 2; ++index) {
        GGML_ASSERT(diagnostics.slots[execution_ids[index]].has_cold_backing);
        assert_bundle_slot_matches(tensors, binding, first_ids[index], execution_ids[index]);
    }

    const uint64_t source_bytes = diagnostics.cold_source_copy_bytes;
    GGML_ASSERT(provider->remap_checkpoint(binding, first_ids, 2, execution_ids).is_ready());
    diagnostics = provider->hot_cache_diagnostics();
    GGML_ASSERT(diagnostics.hits == 2 && diagnostics.cold_source_copy_bytes == source_bytes);
    GGML_ASSERT(diagnostics.cold_current_request_refs == 0);
    const int32_t second_ids[] = { 2, 3 };
    GGML_ASSERT(provider->remap_checkpoint(binding, second_ids, 2, execution_ids).is_ready());
    diagnostics = provider->hot_cache_diagnostics();
    GGML_ASSERT(diagnostics.evictions == 2 && diagnostics.no_writeback_evictions == 2);
    plan.reset();

    GGML_ASSERT(provider->trim().is_ready());
    diagnostics = provider->hot_cache_diagnostics();
    GGML_ASSERT(diagnostics.cold_current_hot_refs == 0);
    GGML_ASSERT(provider->surrender().error == llm_expert_provider_error::busy);
}

void test_three_layout_classes_share_one_cache_and_full_data_path() {
    tensor_fixture f32(8, 16, 4, 1, 2, ggml_backend_cpu_buffer_type(), GGML_TYPE_F32);
    tensor_fixture f16(8, 16, 4, 1, 2, ggml_backend_cpu_buffer_type(), GGML_TYPE_F16);
    tensor_fixture bf16(8, 16, 4, 1, 2, ggml_backend_cpu_buffer_type(), GGML_TYPE_BF16);
    const std::array<const tensor_fixture *, 3> sources = { &f32, &f16, &bf16 };

    auto config = cold_test_config(2);
    config.routed_layer_count = 3;
    config.total_expert_keys = 12;
    config.routed_layers = { 0, 1, 2 };
    auto provider = llm_create_cold_cache_expert_weight_provider(config);

    llm_expert_graph_binding binding;
    for (int32_t layer = 0; layer < 3; ++layer) {
        GGML_ASSERT(provider->bind(
            sources[size_t(layer)]->bundle(layer),
            sources[size_t(layer)]->selection(layer), binding).is_ready());
        GGML_ASSERT(binding.bootstrap);
        binding = {};
    }
    GGML_ASSERT(provider->initialize_after_reserve().is_ready());

    ggml_init_params graph_params = { ggml_tensor_overhead()*64, nullptr, true };
    ggml_context_ptr graph_ctx(ggml_init(graph_params));
    GGML_ASSERT(graph_ctx);
    std::array<llm_expert_graph_binding, 3> bindings;
    for (int32_t layer = 0; layer < 3; ++layer) {
        GGML_ASSERT(provider->bind_graph(
            graph_ctx.get(), sources[size_t(layer)]->bundle(layer),
            sources[size_t(layer)]->selection(layer), bindings[size_t(layer)]).is_ready());
        GGML_ASSERT(!bindings[size_t(layer)].bootstrap);
    }

    auto diagnostics = provider->hot_cache_diagnostics();
    GGML_ASSERT(diagnostics.layout_class_count == 3);
    GGML_ASSERT(diagnostics.layout_class_digests.size() == 3);
    GGML_ASSERT(diagnostics.layout_class_payload_bytes.size() == 3);
    GGML_ASSERT(diagnostics.layout_class_hot_padding_bytes.size() == 3);
    GGML_ASSERT(diagnostics.layout_class_cold_padding_bytes.size() == 3);
    GGML_ASSERT(diagnostics.layout_class_lane_padding_bytes.size() == 3);
    GGML_ASSERT(diagnostics.layout_layer_ids.size() == LLAMA_MAX_LAYERS);
    GGML_ASSERT(diagnostics.layout_layer_ids[0] != diagnostics.layout_layer_ids[1]);
    GGML_ASSERT(diagnostics.layout_layer_ids[0] != diagnostics.layout_layer_ids[2]);
    GGML_ASSERT(diagnostics.layout_layer_ids[1] != diagnostics.layout_layer_ids[2]);
    GGML_ASSERT(diagnostics.layout_registry_administration_bytes < (1U << 20));
    GGML_ASSERT(diagnostics.layout_preflight_passed);
    GGML_ASSERT(diagnostics.layout_preflight_consumer_count == 9);
    GGML_ASSERT(diagnostics.layout_hot_role_offsets.size() == 12);
    GGML_ASSERT(diagnostics.layout_hot_role_extents.size() == 12);
    GGML_ASSERT(diagnostics.layout_cold_role_offsets.size() == 12);
    GGML_ASSERT(diagnostics.layout_cold_role_extents.size() == 12);
    GGML_ASSERT(diagnostics.layout_lane_role_offsets.size() == 12);
    GGML_ASSERT(diagnostics.layout_lane_role_extents.size() == 12);
    for (const auto payload : diagnostics.layout_class_payload_bytes) {
        GGML_ASSERT(payload > 0 && payload <= diagnostics.hot_slot_stride);
    }
    GGML_ASSERT(bindings[0].up.weight->data == bindings[1].up.weight->data);
    GGML_ASSERT(bindings[0].up.weight->data == bindings[2].up.weight->data);

    auto mismatched_binding = bindings[0];
    mismatched_binding.layout_class_id = bindings[1].layout_class_id;
    mismatched_binding.up = bindings[1].up;
    mismatched_binding.gate = bindings[1].gate;
    mismatched_binding.gate_up = bindings[1].gate_up;
    mismatched_binding.down = bindings[1].down;
    llm_expert_execution_plan mismatched_plan;
    GGML_ASSERT(provider->prepare({ mismatched_binding }, mismatched_plan).error ==
        llm_expert_provider_error::invalid_binding);
    mismatched_binding = {};

    const int32_t logical_ids[] = { 0, 1 };
    for (int32_t layer = 0; layer < 3; ++layer) {
        llm_expert_execution_plan plan;
        GGML_ASSERT(provider->prepare({ bindings[size_t(layer)] }, plan).is_ready());
        int32_t execution_ids[] = { -1, -1 };
        GGML_ASSERT(provider->remap_checkpoint(
            bindings[size_t(layer)], logical_ids, 2, execution_ids).is_ready());
        GGML_ASSERT(execution_ids[0] != execution_ids[1]);
        assert_bundle_slot_matches(
            *sources[size_t(layer)], bindings[size_t(layer)], 0, execution_ids[0]);
        plan.reset();

        std::vector<uint8_t> cold_bytes;
        std::vector<uint8_t> hot_bytes;
        GGML_ASSERT(provider->debug_copy_cold_bundle({ layer, 0 }, cold_bytes).is_ready());
        GGML_ASSERT(provider->debug_copy_hot_bundle({ layer, 0 }, hot_bytes).is_ready());
        GGML_ASSERT(!cold_bytes.empty() && cold_bytes == hot_bytes);
        diagnostics = provider->hot_cache_diagnostics();
        GGML_ASSERT(diagnostics.evictions == uint64_t(layer*2));
        bool class_observed = false;
        uint32_t ready_slots = 0;
        for (const auto & slot : diagnostics.slots) {
            if (slot.layer == layer && slot.expert == 0) {
                class_observed = slot.layout_class_id == bindings[size_t(layer)].layout_class_id;
            }
            if (slot.state == llm_hot_cache_diagnostics::slot::ready) {
                ready_slots++;
                GGML_ASSERT(slot.layout_class_id == bindings[size_t(layer)].layout_class_id);
            }
        }
        GGML_ASSERT(class_observed && ready_slots == config.capacity);
    }
    diagnostics = provider->hot_cache_diagnostics();
    GGML_ASSERT(diagnostics.evictions >= 4);
    GGML_ASSERT(diagnostics.slots.size() == config.capacity);
    GGML_ASSERT(diagnostics.policy.config.scope == LLAMA_EXPERT_CACHE_POLICY_SCOPE_GLOBAL);
    GGML_ASSERT(diagnostics.cold_invariant_failures == 0);
    GGML_ASSERT(diagnostics.ring_stage_bytes == diagnostics.ring_h2d_bytes);

    bindings = {};
    graph_ctx.reset();
    GGML_ASSERT(provider->trim().is_ready());
    GGML_ASSERT(provider->surrender().is_ready());
}

void test_cold_provider_copy_failure_cleanup_and_retry() {
    tensor_fixture tensors(8, 16, 4, 1, 2);
    llm_expert_provider_faults faults;
    faults.fail_copy_after_tensors = 1;
    auto provider = llm_create_cold_cache_expert_weight_provider(cold_test_config(), faults);
    const auto binding = initialize_hot_binding(*provider, tensors);
    llm_expert_execution_plan plan;
    GGML_ASSERT(provider->prepare({ binding }, plan).is_ready());
    const int32_t logical_ids[] = { 0, 1 };
    int32_t execution_ids[] = { -1, -1 };
    GGML_ASSERT(provider->remap_checkpoint(binding, logical_ids, 2, execution_ids).error ==
        llm_expert_provider_error::copy_failed);
    auto diagnostics = provider->hot_cache_diagnostics();
    GGML_ASSERT(diagnostics.cold_failed_copies == 1);
    GGML_ASSERT(diagnostics.cold_current_hot_refs == 0);
    GGML_ASSERT(diagnostics.cold_current_transfer_refs == 0);
    plan.reset();
    GGML_ASSERT(provider->cleanup_failed_slots().is_ready());

    GGML_ASSERT(provider->prepare({ binding }, plan).is_ready());
    GGML_ASSERT(provider->remap_checkpoint(binding, logical_ids, 2, execution_ids).is_ready());
    diagnostics = provider->hot_cache_diagnostics();
    GGML_ASSERT(diagnostics.cold_current_hot_refs == 2);
    GGML_ASSERT(diagnostics.cold_current_transfer_refs == 0);
    plan.reset();
}

void test_directory_composite_layer_expert_keys() {
    tensor_fixture first(8, 16, 4, 1, 1);
    tensor_fixture second(8, 16, 4, 1, 1);
    auto provider = llm_create_hot_cache_expert_weight_provider(test_config(2, 2, 8, 1));
    llm_expert_graph_binding first_binding;
    llm_expert_graph_binding second_binding;
    GGML_ASSERT(provider->bind(first.bundle(0), first.selection(0), first_binding).is_ready());
    GGML_ASSERT(provider->bind(second.bundle(1), second.selection(1), second_binding).is_ready());
    GGML_ASSERT(provider->initialize_after_reserve().is_ready());
    GGML_ASSERT(provider->bind(first.bundle(0), first.selection(0), first_binding).is_ready());
    GGML_ASSERT(provider->bind(second.bundle(1), second.selection(1), second_binding).is_ready());

    llm_expert_execution_plan plan;
    GGML_ASSERT(provider->prepare({ first_binding, second_binding }, plan).is_ready());
    const int32_t logical_ids[] = { 0 };
    int32_t first_execution[] = { -1 };
    int32_t second_execution[] = { -1 };
    GGML_ASSERT(provider->remap_checkpoint(first_binding, logical_ids, 1, first_execution).is_ready());
    GGML_ASSERT(provider->remap_checkpoint(second_binding, logical_ids, 1, second_execution).is_ready());
    GGML_ASSERT(first_execution[0] != second_execution[0]);
    const auto diagnostics = provider->hot_cache_diagnostics();
    GGML_ASSERT(diagnostics.slots[first_execution[0]].layer == 0);
    GGML_ASSERT(diagnostics.slots[second_execution[0]].layer == 1);
    assert_bundle_slot_matches(first, first_binding, 0, first_execution[0]);
    assert_bundle_slot_matches(second, second_binding, 0, second_execution[0]);
    plan.reset();
}

void test_directory_lru_pin_exclusion_and_request_exclusivity() {
    tensor_fixture tensors;
    auto provider = llm_create_hot_cache_expert_weight_provider(test_config());
    const auto binding = initialize_hot_binding(*provider, tensors);
    llm_expert_execution_plan plan;
    llm_expert_execution_plan competing;
    GGML_ASSERT(provider->prepare({ binding }, plan).is_ready());
    GGML_ASSERT(provider->prepare({ binding }, competing).error == llm_expert_provider_error::busy);

    const int32_t first_ids[] = { 0, 1 };
    int32_t execution_ids[] = { -1, -1 };
    GGML_ASSERT(provider->remap_checkpoint(binding, first_ids, 2, execution_ids).is_ready());
    const auto first = provider->hot_cache_diagnostics();
    const int32_t slot_zero = find_slot(first, 0);
    const int32_t slot_one = find_slot(first, 1);
    GGML_ASSERT(slot_zero >= 0 && slot_one >= 0 && slot_zero != slot_one);

    const int32_t second_ids[] = { 0, 2 };
    GGML_ASSERT(provider->remap_checkpoint(binding, second_ids, 2, execution_ids).is_ready());
    const auto second = provider->hot_cache_diagnostics();
    GGML_ASSERT(find_slot(second, 0) == slot_zero);
    GGML_ASSERT(find_slot(second, 2) == slot_one);
    GGML_ASSERT(second.slots[slot_zero].generation == first.slots[slot_zero].generation);
    plan.reset();

    GGML_ASSERT(provider->prepare({ binding }, plan).is_ready());
    const int32_t third_ids[] = { 2, 3 };
    GGML_ASSERT(provider->remap_checkpoint(binding, third_ids, 2, execution_ids).is_ready());
    const auto third = provider->hot_cache_diagnostics();
    GGML_ASSERT(find_slot(third, 2) == slot_one);
    GGML_ASSERT(find_slot(third, 3) == slot_zero);
    GGML_ASSERT(third.evictions == 2);
    GGML_ASSERT(third.exclusive_busy_failures == 1);
    plan.reset();
}

void test_directory_copy_failure_cleanup_and_reuse() {
    tensor_fixture tensors;
    llm_expert_provider_faults faults;
    faults.fail_copy_after_tensors = 4;
    auto provider = llm_create_hot_cache_expert_weight_provider(test_config(), faults);
    const auto binding = initialize_hot_binding(*provider, tensors);
    llm_expert_execution_plan plan;
    GGML_ASSERT(provider->prepare({ binding }, plan).is_ready());
    const int32_t logical_ids[] = { 0, 1 };
    int32_t execution_ids[] = { -1, -1 };
    GGML_ASSERT(provider->remap_checkpoint(binding, logical_ids, 2, execution_ids).error ==
        llm_expert_provider_error::copy_failed);
    auto diagnostics = provider->hot_cache_diagnostics();
    GGML_ASSERT(diagnostics.copy_failures == 1);
    GGML_ASSERT(diagnostics.current_pins == 0);
    GGML_ASSERT(std::all_of(diagnostics.slots.begin(), diagnostics.slots.end(), [](const auto & slot) {
        return slot.state == llm_hot_cache_diagnostics::slot::failed;
    }));
    GGML_ASSERT(provider->cleanup_failed_slots().error == llm_expert_provider_error::busy);
    plan.reset();
    GGML_ASSERT(provider->cleanup_failed_slots().is_ready());
    diagnostics = provider->hot_cache_diagnostics();
    GGML_ASSERT(diagnostics.failed_cleanups == 2);
    GGML_ASSERT(std::all_of(diagnostics.slots.begin(), diagnostics.slots.end(), [](const auto & slot) {
        return slot.state == llm_hot_cache_diagnostics::slot::free && slot.expert == -1;
    }));

    GGML_ASSERT(provider->prepare({ binding }, plan).is_ready());
    GGML_ASSERT(provider->remap_checkpoint(binding, logical_ids, 2, execution_ids).is_ready());
    GGML_ASSERT(execution_ids[0] != execution_ids[1]);
    plan.reset();
}

void test_directory_generation_exhaustion_trim_and_invalid_id() {
    tensor_fixture tensors(8, 16, 4, 1, 1);
    auto provider = llm_create_hot_cache_expert_weight_provider(
        test_config(1, 1, 4, 1, UINT64_MAX - 1));
    const auto binding = initialize_hot_binding(*provider, tensors);
    llm_expert_execution_plan plan;
    GGML_ASSERT(provider->prepare({ binding }, plan).is_ready());
    int32_t execution_ids[] = { -1 };
    const int32_t first_ids[] = { 0 };
    GGML_ASSERT(provider->remap_checkpoint(binding, first_ids, 1, execution_ids).is_ready());
    const auto before = provider->hot_cache_diagnostics();
    GGML_ASSERT(before.slots[0].generation == UINT64_MAX);
    plan.reset();

    GGML_ASSERT(provider->prepare({ binding }, plan).is_ready());
    const int32_t second_ids[] = { 1 };
    GGML_ASSERT(provider->remap_checkpoint(binding, second_ids, 1, execution_ids).error ==
        llm_expert_provider_error::generation_exhausted);
    auto after = provider->hot_cache_diagnostics();
    GGML_ASSERT(after.slots[0].expert == before.slots[0].expert);
    GGML_ASSERT(after.slots[0].generation == before.slots[0].generation);
    GGML_ASSERT(after.slots[0].state == llm_hot_cache_diagnostics::slot::ready);
    plan.reset();

    GGML_ASSERT(provider->prepare({ binding }, plan).is_ready());
    const int32_t invalid_ids[] = { 4 };
    GGML_ASSERT(provider->remap_checkpoint(binding, invalid_ids, 1, execution_ids).error ==
        llm_expert_provider_error::invalid_key);
    after = provider->hot_cache_diagnostics();
    GGML_ASSERT(after.slots[0].expert == 0 && after.slots[0].generation == UINT64_MAX);
    plan.reset();

    GGML_ASSERT(provider->prepare({ binding }, plan).is_ready());
    GGML_ASSERT(provider->remap_checkpoint(binding, first_ids, 1, execution_ids).is_ready());
    GGML_ASSERT(provider->trim().is_ready());
    GGML_ASSERT(provider->hot_cache_diagnostics().slots[0].state ==
        llm_hot_cache_diagnostics::slot::pinned);
    plan.reset();
    GGML_ASSERT(provider->trim().is_ready());
    after = provider->hot_cache_diagnostics();
    GGML_ASSERT(after.slots[0].state == llm_hot_cache_diagnostics::slot::free);
    GGML_ASSERT(after.slots[0].expert == -1);
    GGML_ASSERT(provider->validate_slot_generation(0, UINT64_MAX).error ==
        llm_expert_provider_error::stale_generation);
}

void test_cuda_model_pool_smoke(const char * model_path) {
    const llama_model_tensor_buft_override overrides[] = {
        { "ffn_(gate|up|down)_exps\\.weight", ggml_backend_cpu_buffer_type() },
        { nullptr, nullptr },
    };
    llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = -1;
    params.tensor_buft_overrides = overrides;
    params.expert_weights_mode = LLAMA_EXPERT_WEIGHTS_MODE_HOT_CACHE;
    params.expert_hot_cache_capacity = 2;
    llama_model * model = llama_model_load_from_file(model_path, params);
    GGML_ASSERT(model != nullptr);

    llama_context_params context_params = llama_context_default_params();
    eval_callback_counts callback_counts;
    context_params.n_ctx = 64;
    context_params.n_batch = 64;
    context_params.n_ubatch = 2;
    context_params.cb_eval = count_execution_id_callbacks;
    context_params.cb_eval_user_data = &callback_counts;
    llama_context * rejected = llama_init_from_model(model, context_params);
    GGML_ASSERT(rejected == nullptr);

    auto * provider = model->expert_weight_provider();
    GGML_ASSERT(provider != nullptr);
    auto diagnostics = provider->hot_cache_diagnostics();
    GGML_ASSERT(diagnostics.effective_capacity == 0);
    GGML_ASSERT(diagnostics.last_context_extent == 2);
    GGML_ASSERT(diagnostics.conservative_required_capacity == 4);
    GGML_ASSERT(diagnostics.context_rejections == 1);

    context_params.n_ubatch = 1;
    llama_context * context = llama_init_from_model(model, context_params);
    GGML_ASSERT(context != nullptr);

    diagnostics = provider->hot_cache_diagnostics();
    GGML_ASSERT(diagnostics.effective_capacity == 2);
    GGML_ASSERT(diagnostics.pool_bytes > 0);
    GGML_ASSERT(!diagnostics.slot_tensor_addresses.empty());
    GGML_ASSERT(diagnostics.last_context_extent == 1);
    GGML_ASSERT(diagnostics.conservative_required_capacity == 2);
    GGML_ASSERT(provider->surrender().error == llm_expert_provider_error::busy);

    context_params.n_ubatch = 2;
    rejected = llama_init_from_model(model, context_params);
    GGML_ASSERT(rejected == nullptr);
    GGML_ASSERT(provider->hot_cache_diagnostics().effective_capacity == 2);

    context_params.n_ubatch = 1;
    llama_context * second_context = llama_init_from_model(model, context_params);
    GGML_ASSERT(second_context != nullptr);

    llama_token token = 1;
    GGML_ASSERT(llama_decode(context, llama_batch_get_one(&token, 1)) == 0);
    GGML_ASSERT(llama_decode(second_context, llama_batch_get_one(&token, 1)) == -3);
    const float * first_logits = llama_get_logits_ith(context, -1);
    GGML_ASSERT(first_logits != nullptr);
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    GGML_ASSERT(std::all_of(first_logits, first_logits + n_vocab, [](float value) {
        return std::isfinite(value);
    }));
    diagnostics = provider->hot_cache_diagnostics();
    GGML_ASSERT(diagnostics.remap_checkpoints == 7);
    GGML_ASSERT(diagnostics.synchronization_checkpoints == 7);
    GGML_ASSERT(diagnostics.misses == 14);
    GGML_ASSERT(diagnostics.current_pins == 0);
    GGML_ASSERT(callback_counts.asks > 7);
    GGML_ASSERT(callback_counts.observations == 7);

    GGML_ASSERT(llama_decode(second_context, llama_batch_get_one(&token, 1)) == 0);
    const float * second_logits = llama_get_logits_ith(second_context, -1);
    GGML_ASSERT(second_logits != nullptr);
    GGML_ASSERT(std::equal(first_logits, first_logits + n_vocab, second_logits));
    diagnostics = provider->hot_cache_diagnostics();
    GGML_ASSERT(diagnostics.remap_checkpoints == 14);
    GGML_ASSERT(diagnostics.exclusive_busy_failures == 1);
    GGML_ASSERT(diagnostics.current_pins == 0);
    GGML_ASSERT(callback_counts.observations == 14);

    llama_context * cancelled_context = llama_init_from_model(model, context_params);
    GGML_ASSERT(cancelled_context != nullptr);
    llama_set_abort_callback(cancelled_context, [](void *) { return true; }, nullptr);
    GGML_ASSERT(llama_decode(cancelled_context, llama_batch_get_one(&token, 1)) == 2);
    llama_synchronize(cancelled_context);
    GGML_ASSERT(provider->hot_cache_diagnostics().current_pins == 0);
    llama_free(cancelled_context);

    llama_free(second_context);
    llama_free(context);
    GGML_ASSERT(provider->surrender().is_ready());
    llama_model_free(model);
}

void test_cuda_model_cross_epoch_hits(const char * model_path) {
    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = 64;
    context_params.n_batch = 64;
    context_params.n_ubatch = 1;
    llama_token token = 1;

    llama_model_params disabled_params = llama_model_default_params();
    disabled_params.n_gpu_layers = -1;
    llama_model * disabled_model = llama_model_load_from_file(model_path, disabled_params);
    GGML_ASSERT(disabled_model != nullptr);
    llama_context * disabled_context = llama_init_from_model(disabled_model, context_params);
    GGML_ASSERT(disabled_context != nullptr);
    route_capture disabled_routes;
    GGML_ASSERT(llama_set_route_observer(disabled_context, capture_routes, &disabled_routes) ==
        LLAMA_ROUTE_OBSERVER_STATUS_OK);
    GGML_ASSERT(llama_route_observer_begin(disabled_context, 1, LLAMA_ROUTE_PHASE_DECODE) ==
        LLAMA_ROUTE_OBSERVER_STATUS_OK);
    GGML_ASSERT(llama_decode(disabled_context, llama_batch_get_one(&token, 1)) == 0);
    const float * disabled_logits = llama_get_logits_ith(disabled_context, -1);
    GGML_ASSERT(disabled_logits != nullptr);
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(disabled_model));
    const std::vector<float> reference_logits(disabled_logits, disabled_logits + n_vocab);
    const auto disabled_graph = disabled_context->expert_graph_diagnostics();
    GGML_ASSERT(disabled_graph.binding_count == 0);
    llama_free(disabled_context);
    llama_model_free(disabled_model);

    const llama_model_tensor_buft_override overrides[] = {
        { "ffn_(gate|up|down)_exps\\.weight", ggml_backend_cpu_buffer_type() },
        { nullptr, nullptr },
    };
    llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = -1;
    params.tensor_buft_overrides = overrides;
    params.expert_weights_mode = LLAMA_EXPERT_WEIGHTS_MODE_HOT_CACHE;
    params.expert_hot_cache_capacity = 56;
    llama_model * model = llama_model_load_from_file(model_path, params);
    GGML_ASSERT(model != nullptr);

    llama_context * first = llama_init_from_model(model, context_params);
    llama_context * second = llama_init_from_model(model, context_params);
    GGML_ASSERT(first != nullptr && second != nullptr);
    route_capture hot_routes;
    GGML_ASSERT(llama_set_route_observer(first, capture_routes, &hot_routes) ==
        LLAMA_ROUTE_OBSERVER_STATUS_OK);
    GGML_ASSERT(llama_route_observer_begin(first, 1, LLAMA_ROUTE_PHASE_DECODE) ==
        LLAMA_ROUTE_OBSERVER_STATUS_OK);

    GGML_ASSERT(llama_decode(first, llama_batch_get_one(&token, 1)) == 0);
    const float * first_logits = llama_get_logits_ith(first, -1);
    GGML_ASSERT(first_logits != nullptr);
    GGML_ASSERT(std::equal(first_logits, first_logits + n_vocab, reference_logits.begin()));
    GGML_ASSERT(hot_routes.layers == disabled_routes.layers);
    GGML_ASSERT(hot_routes.ids == disabled_routes.ids);
    GGML_ASSERT(hot_routes.weights == disabled_routes.weights);
    const auto hot_graph = first->expert_graph_diagnostics();
    GGML_ASSERT(hot_graph.binding_count == 7);
    GGML_ASSERT(hot_graph.node_count == disabled_graph.node_count + 7);
    GGML_ASSERT(hot_graph.operation_hash != disabled_graph.operation_hash);
    auto * provider = model->expert_weight_provider();
    const auto cold = provider->hot_cache_diagnostics();
    GGML_ASSERT(cold.remap_checkpoints == 7);
    GGML_ASSERT(cold.misses == 14 && cold.hits == 0);
    GGML_ASSERT(cold.h2d_bytes > 0);
    GGML_ASSERT(cold.last_remap_layer == hot_routes.layers.back());
    GGML_ASSERT(cold.last_logical_ids.size() == 2 && cold.last_execution_ids.size() == 2);
    GGML_ASSERT(std::equal(cold.last_logical_ids.begin(), cold.last_logical_ids.end(), hot_routes.ids.end() - 2));
    GGML_ASSERT(std::all_of(cold.last_execution_ids.begin(), cold.last_execution_ids.end(), [](int32_t slot) {
        return slot >= 12 && slot < 14;
    }));

    GGML_ASSERT(llama_decode(second, llama_batch_get_one(&token, 1)) == 0);
    const float * second_logits = llama_get_logits_ith(second, -1);
    GGML_ASSERT(second_logits != nullptr);
    GGML_ASSERT(std::equal(first_logits, first_logits + n_vocab, second_logits));
    const auto warm = provider->hot_cache_diagnostics();
    GGML_ASSERT(warm.remap_checkpoints == 14);
    GGML_ASSERT(warm.misses == cold.misses);
    GGML_ASSERT(warm.hits == cold.hits + 14);
    GGML_ASSERT(warm.h2d_bytes == cold.h2d_bytes);
    GGML_ASSERT(warm.h2d_time_us == cold.h2d_time_us);
    GGML_ASSERT(warm.slot_tensor_addresses == cold.slot_tensor_addresses);
    GGML_ASSERT(warm.current_pins == 0);
    for (size_t slot = 0; slot < cold.slots.size(); ++slot) {
        GGML_ASSERT(warm.slots[slot].generation == cold.slots[slot].generation);
    }

    GGML_ASSERT(llama_decode(second, llama_batch_get_one(&token, 1)) == 0);
    GGML_ASSERT(llama_decode(second, llama_batch_get_one(&token, 1)) == 0);
    const float * reused_logits = llama_get_logits_ith(second, -1);
    GGML_ASSERT(reused_logits != nullptr);
    GGML_ASSERT(std::all_of(reused_logits, reused_logits + n_vocab, [](float value) {
        return std::isfinite(value);
    }));
    GGML_ASSERT(second->expert_graph_diagnostics().graphs_reused > 0);
    GGML_ASSERT(provider->hot_cache_diagnostics().current_pins == 0);

    llama_free(second);
    llama_free(first);
    GGML_ASSERT(provider->surrender().is_ready());
    llama_model_free(model);
}

void test_cuda_model_multi_token_all_expert_capacity(const char * model_path) {
    const llama_model_tensor_buft_override overrides[] = {
        { "ffn_(gate|up|down)_exps\\.weight", ggml_backend_cpu_buffer_type() },
        { nullptr, nullptr },
    };
    llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = -1;
    params.tensor_buft_overrides = overrides;
    params.expert_weights_mode = LLAMA_EXPERT_WEIGHTS_MODE_HOT_CACHE;
    params.expert_hot_cache_capacity = 8;
    llama_model * model = llama_model_load_from_file(model_path, params);
    GGML_ASSERT(model != nullptr);

    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = 64;
    context_params.n_batch = 64;
    context_params.n_ubatch = 2;
    llama_context * context = llama_init_from_model(model, context_params);
    GGML_ASSERT(context != nullptr);
    llama_token tokens[] = { 1, 1 };
    GGML_ASSERT(llama_decode(context, llama_batch_get_one(tokens, 2)) == 0);
    const float * logits = llama_get_logits_ith(context, -1);
    GGML_ASSERT(logits != nullptr);
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    GGML_ASSERT(std::all_of(logits, logits + n_vocab, [](float value) { return std::isfinite(value); }));
    auto * provider = model->expert_weight_provider();
    const auto diagnostics = provider->hot_cache_diagnostics();
    GGML_ASSERT(diagnostics.last_context_extent == 2);
    GGML_ASSERT(diagnostics.conservative_required_capacity == 4);
    GGML_ASSERT(diagnostics.remap_checkpoints == 7);
    GGML_ASSERT(diagnostics.logical_ids == 28);
    GGML_ASSERT(diagnostics.current_pins == 0);

    llama_free(context);
    GGML_ASSERT(provider->surrender().is_ready());
    llama_model_free(model);
}

void test_cuda_directory_copy() {
    ggml_backend_dev_t gpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    GGML_ASSERT(gpu != nullptr);
    tensor_fixture tensors;
    auto config = test_config();
    config.target_buffer_type = ggml_backend_dev_buffer_type(gpu);
    config.allow_non_cuda_target_for_testing = false;
    auto provider = llm_create_hot_cache_expert_weight_provider(config);
    const auto binding = initialize_hot_binding(*provider, tensors);
    llm_expert_execution_plan plan;
    GGML_ASSERT(provider->prepare({ binding }, plan).is_ready());
    const int32_t logical_ids[] = { 1, 3 };
    int32_t execution_ids[] = { -1, -1 };
    GGML_ASSERT(provider->remap_checkpoint(binding, logical_ids, 2, execution_ids).is_ready());

    const auto assert_cuda_tensor = [&](const ggml_tensor * source, const ggml_tensor * target,
                                        int32_t expert, int32_t slot, bool weight) {
        GGML_ASSERT((source == nullptr) == (target == nullptr));
        if (source == nullptr) {
            return;
        }
        const int axis = source_expert_axis(source, tensors.n_expert, weight);
        const size_t span = source->nb[axis];
        std::vector<uint8_t> actual(span);
        ggml_backend_tensor_get(target, actual.data(), size_t(slot)*span, span);
        const auto * expected = static_cast<const uint8_t *>(source->data) + size_t(expert)*span;
        GGML_ASSERT(std::memcmp(actual.data(), expected, span) == 0);
    };
    const auto assert_cuda_projection = [&](const auto & source, const auto & target,
                                             int32_t expert, int32_t slot) {
        assert_cuda_tensor(source.weight, target.weight, expert, slot, true);
        assert_cuda_tensor(source.bias, target.bias, expert, slot, false);
        assert_cuda_tensor(source.scale, target.scale, expert, slot, false);
    };
    const auto source = tensors.bundle(0);
    for (int32_t index = 0; index < 2; ++index) {
        assert_cuda_projection(source.up, binding.up, logical_ids[index], execution_ids[index]);
        assert_cuda_projection(source.gate, binding.gate, logical_ids[index], execution_ids[index]);
        assert_cuda_projection(source.down, binding.down, logical_ids[index], execution_ids[index]);
    }
    const auto diagnostics = provider->hot_cache_diagnostics();
    GGML_ASSERT(diagnostics.misses == 2);
    GGML_ASSERT(diagnostics.admissions == 2);
    GGML_ASSERT(diagnostics.h2d_bytes > 0);
    plan.reset();
}

} // namespace

int main(int argc, char ** argv) {
    test_runtime_policy_adapters_and_bounded_metadata();
    test_predictor_runtime_has_no_heap_allocations();
    test_initialization_stage_and_descriptor_only_scale();
    test_configuration_matrix();
    test_cold_provider_uses_bounded_transfer_waves();
    test_context_extent_matrix_and_prepare_revalidation();
    test_cold_provider_rejects_cuda_host_source();
    test_pool_lifetime_trim_surrender_and_epoch();
    test_layout_host_and_partial_initialization_rejection();
    test_layout_class_determinism_and_bounded_cap();
    test_allocation_failure_is_recoverable_and_empty_prepare_is_safe();
    test_directory_hit_eviction_generation_and_copy();
    test_directory_multi_token_atomic_dedup_and_no_allocation();
    test_cold_provider_inclusive_promotion_and_hits();
    test_three_layout_classes_share_one_cache_and_full_data_path();
    test_cold_provider_copy_failure_cleanup_and_retry();
    test_directory_composite_layer_expert_keys();
    test_directory_lru_pin_exclusion_and_request_exclusivity();
    test_directory_copy_failure_cleanup_and_reuse();
    test_directory_generation_exhaustion_trim_and_invalid_id();
    test_blocking_hot_seed_atomic_publication_and_failure();
    test_blocking_cold_seed_uses_storage_ring_and_ordinary_lru();
    test_exact_issue_ahead_and_serial_evidence_control();
    test_cancelling_speculative_retry_attempts_all_demands_before_wait();
    test_predictive_runtime_late_join_same_generation();
    test_predictive_request_end_cancels_and_drains_storage();
    run_predictive_host_ready_hybrid_consumption(false);
    run_predictive_host_ready_hybrid_consumption(true);
    test_predictive_device_readiness_rejects_eventless_transport();
    test_predictive_hot_expiry_and_deterministic_circuit();
    test_predictive_circuit_cancels_future_layer_work();
    test_predictive_trace_overflow_disables_prediction();
    test_predictive_hot_slot_budget_saturation();
    test_predictive_cross_layer_router_trigger_and_prefill_exclusion();
    test_predictive_reversed_layer_callbacks_are_canonical();
    test_exact_demand_joins_submitted_same_generation_with_ready_cold_data();
    test_hot_speculative_victim_deadline_utility_slot_order();
    if (argc == 2) {
        ggml_backend_load_all();
        test_cuda_directory_copy();
        test_cuda_model_pool_smoke(argv[1]);
        test_cuda_model_cross_epoch_hits(argv[1]);
        test_cuda_model_multi_token_all_expert_capacity(argv[1]);
        llama_backend_free();
    } else if (argc != 1) {
        std::cerr << "usage: test-hot-expert-cache [MODEL]\n";
        return 2;
    }
    std::cout << "PHASE10_STATIC_SEED"
              << "\tentries=" << phase10_evidence.seed_entries
              << "\tstorage_bytes=" << phase10_evidence.seed_storage_bytes
              << "\th2d_bytes=" << phase10_evidence.seed_h2d_bytes
              << "\tordinary_lru=" << phase10_evidence.seed_ordinary_lru
              << "\tfailure_rolled_back=" << phase10_evidence.seed_failure_rolled_back
              << "\tscheduler_drained=" << phase10_evidence.seed_scheduler_drained
              << "\thot_speculative_victim_order=" << phase10_evidence.hot_speculative_victim_order << '\n';
    std::cout << "PHASE10_EXACT_ISSUE_AHEAD"
              << "\tmisses=" << phase10_evidence.parallel_misses
              << "\tenqueued_before_take=" << phase10_evidence.parallel_enqueued_before_take
              << "\tparallel_submitted_before_wait=" << phase10_evidence.parallel_submitted_before_wait
              << "\tparallel_ready_before_use=" << phase10_evidence.parallel_ready_before_use
              << "\tserial_submitted_before_wait=" << phase10_evidence.serial_submitted_before_wait
              << "\troutes_equal=" << phase10_evidence.routes_equal
              << "\tjoined_same_generation=" << phase10_evidence.joined_same_generation
              << "\tjoined_submitted_ready=" << phase10_evidence.joined_submitted_ready
              << "\tdeferred_retry_multi_key_ordered="
              << phase10_evidence.deferred_retry_multi_key_ordered << '\n';
    return 0;
}
