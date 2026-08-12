#include "llama-expert-resident-demand.h"

#include "ggml-cpp.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <future>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unistd.h>

namespace {

void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

struct temporary_source {
    char path[64] = "/tmp/expert-resident-demand-XXXXXX";

    temporary_source() {
        const int fd = mkstemp(path);
        require(fd >= 0, "mkstemp failed");
        std::array<uint8_t, 576> bytes;
        for (size_t index = 0; index < bytes.size(); ++index) bytes[index] = uint8_t(index*17U + 3U);
        require(write(fd, bytes.data(), bytes.size()) == ssize_t(bytes.size()), "source write failed");
        close(fd);
    }

    ~temporary_source() {
        unlink(path);
    }
};

struct wide_temporary_source {
    char path[64] = "/tmp/expert-resident-batch-XXXXXX";

    explicit wide_temporary_source(uint32_t expert_count) {
        const int fd = mkstemp(path);
        require(fd >= 0, "wide mkstemp failed");
        std::vector<uint8_t> bytes(size_t(expert_count)*144);
        for (size_t index = 0; index < bytes.size(); ++index) bytes[index] = uint8_t(index*17U + 3U);
        require(write(fd, bytes.data(), bytes.size()) == ssize_t(bytes.size()),
            "wide source write failed");
        close(fd);
    }

    ~wide_temporary_source() {
        unlink(path);
    }
};

llm_expert_bundle_descriptor make_bundle(ggml_context * ctx, int64_t n_expert) {
    auto projection = [&](const char * name) {
        auto * weight = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, 3, n_expert);
        ggml_set_name(weight, name);
        return llm_expert_projection_descriptor::from(weight, nullptr, nullptr);
    };
    return { 0, int32_t(n_expert), projection("up"), projection("gate"), {}, projection("down") };
}

llm_expert_bundle_descriptor make_bundle(ggml_context * ctx) {
    return make_bundle(ctx, 4);
}

std::vector<llm_expert_storage_span> spans(uint64_t base) {
    return {
        { 0, base + 0, 48, llm_expert_storage_projection::up,
            llm_expert_storage_sidecar::weight, 0, 48 },
        { 0, base + 48, 48, llm_expert_storage_projection::gate,
            llm_expert_storage_sidecar::weight, 48, 48 },
        { 0, base + 96, 48, llm_expert_storage_projection::down,
            llm_expert_storage_sidecar::weight, 96, 48 },
    };
}

struct loader_context {
    llm_expert_storage * storage = nullptr;
};

llm_expert_provider_result load_bundle(
        void * user_data,
        llm_expert_key key,
        const llm_expert_bundle_descriptor & destination,
        uint32_t slot) noexcept {
    auto & context = *static_cast<loader_context *>(user_data);
    std::array<llm_expert_storage_destination, 3> destinations;
    const std::array<std::pair<llm_expert_storage_projection, ggml_tensor *>, 3> projections = {{
        { llm_expert_storage_projection::up, destination.up.weight },
        { llm_expert_storage_projection::gate, destination.gate.weight },
        { llm_expert_storage_projection::down, destination.down.weight },
    }};
    for (size_t index = 0; index < projections.size(); ++index) {
        auto * tensor = projections[index].second;
        if (tensor == nullptr || tensor->data == nullptr || slot >= uint32_t(tensor->ne[2])) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::invalid_descriptor);
        }
        destinations[index] = {
            projections[index].first,
            llm_expert_storage_sidecar::weight,
            static_cast<uint8_t *>(tensor->data) + size_t(slot)*tensor->nb[2],
            tensor->nb[1]*uint64_t(tensor->ne[1]),
            0,
        };
    }
    std::array<llm_expert_storage_read_operation, 3> operations;
    size_t operation_count = 0;
    const auto planned = context.storage->make_read_plan(
        key, destinations.data(), destinations.size(),
        operations.data(), operations.size(), operation_count);
    if (!planned.is_ready() || operation_count == 0) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
    }
    uint64_t completed = 0;
    for (size_t operation = 0; operation < operation_count; ++operation) {
        for (uint8_t segment = 0; segment < operations[operation].segment_count; ++segment) {
            const auto & item = operations[operation].segments[segment];
            const ssize_t read = pread(
                operations[operation].native_handle, item.data, item.byte_count, item.file_offset);
            if (read != ssize_t(item.byte_count)) {
                context.storage->record_async_read(
                    operation_count, completed,
                    read < 0 ? llm_expert_storage_error::io_error : llm_expert_storage_error::short_read);
                return llm_expert_provider_result::failure(llm_expert_provider_error::copy_failed);
            }
            completed += item.byte_count;
        }
    }
    context.storage->record_async_read(
        operation_count, completed, llm_expert_storage_error::none);
    return llm_expert_provider_result::success();
}

bool same_policy_event(
        const llm_expert_cache_policy_event & lhs,
        const llm_expert_cache_policy_event & rhs) {
    return lhs.request_ordinal == rhs.request_ordinal && lhs.ubatch_ordinal == rhs.ubatch_ordinal &&
        lhs.tier == rhs.tier && lhs.type == rhs.type && lhs.event_sequence == rhs.event_sequence &&
        lhs.demand_ordinal == rhs.demand_ordinal &&
        lhs.origin_operation_ordinal == rhs.origin_operation_ordinal && lhs.phase == rhs.phase &&
        lhs.key.layer == rhs.key.layer && lhs.key.expert == rhs.key.expert &&
        lhs.key.layout_class_id == rhs.key.layout_class_id &&
        lhs.occurrence_count == rhs.occurrence_count &&
        lhs.logical_bundle_bytes == rhs.logical_bundle_bytes &&
        lhs.physical_slot_footprint_bytes == rhs.physical_slot_footprint_bytes &&
        lhs.slot == rhs.slot && lhs.generation == rhs.generation && lhs.domain == rhs.domain &&
        lhs.eligible == rhs.eligible && lhs.decision == rhs.decision && lhs.reason == rhs.reason &&
        lhs.state_digest == rhs.state_digest;
}

bool same_key_sequence(
        const std::vector<llm_expert_key> & lhs,
        const std::vector<llm_expert_key> & rhs) {
    return lhs.size() == rhs.size() && std::equal(
        lhs.begin(), lhs.end(), rhs.begin(), [](const auto & left, const auto & right) {
            return left.layer == right.layer && left.expert == right.expert;
        });
}

void validate_slot(
        const llm_expert_bundle_descriptor & bundle,
        uint32_t slot,
        int32_t expert) {
    for (const auto * projection : { &bundle.up, &bundle.gate, &bundle.down }) {
        const auto * data = static_cast<const uint8_t *>(projection->weight->data) +
            size_t(slot)*projection->weight->nb[2];
        const size_t projection_index = projection == &bundle.up ? 0 : projection == &bundle.gate ? 1 : 2;
        const uint64_t offset = uint64_t(expert)*144 + projection_index*48;
        require(data[0] == uint8_t(offset*17U + 3U), "final slot has wrong source bytes");
    }
}

void test_serial_host_ready_equivalence() {
    temporary_source source;
    llama_file file(source.path, "rb");
    llm_expert_storage storage({ 1, 4, 4, 1024 }, { { 0, &file, 512, source.path, false } });
    for (int32_t expert = 0; expert < 4; ++expert) {
        require(storage.add_bundle({ 0, expert }, spans(uint64_t(expert)*144)).is_ready(),
            "storage directory failed");
    }
    require(storage.seal().is_ready(), "storage seal failed");

    ggml_init_params context_params = { ggml_tensor_overhead()*16, nullptr, true };
    ggml_context_ptr context(ggml_init(context_params));
    require(bool(context), "context failed");
    const auto prototype = make_bundle(context.get());

    llm_cold_cache_config cache_config;
    cache_config.byte_budget = 1U << 20;
    cache_config.minimum_slots = 4;
    cache_config.routed_layer_count = 1;
    cache_config.total_expert_keys = 4;
    cache_config.routed_layers = { 0 };

    llama_file baseline_file(source.path, "rb");
    llm_expert_storage baseline_storage(
        { 1, 4, 4, 1024 }, { { 0, &baseline_file, 512, source.path, false } });
    for (int32_t expert = 0; expert < 4; ++expert) {
        require(baseline_storage.add_bundle({ 0, expert }, spans(uint64_t(expert)*144)).is_ready(),
            "baseline storage directory failed");
    }
    require(baseline_storage.seal().is_ready(), "baseline storage seal failed");
    llm_cold_expert_cache baseline_cache(cache_config);
    require(baseline_cache.initialize(prototype).is_ready(), "baseline cache initialization failed");
    loader_context baseline_loader = { &baseline_storage };
    const std::array<int32_t, 3> baseline_experts = { 2, 0, 1 };
    const std::array<uint32_t, 3> canonical_unique = { 1, 2, 0 };
    std::array<llm_cold_reference, 3> baseline_by_unique;
    std::array<llm_cold_reference, 3> baseline_release;
    require(baseline_cache.policy_request_begin().is_ready(), "baseline policy begin failed");
    for (size_t order = 0; order < canonical_unique.size(); ++order) {
        const uint32_t unique = canonical_unique[order];
        require(baseline_cache.find_or_admit_with_loader(
            { 0, baseline_experts[unique] }, baseline_by_unique[unique],
            load_bundle, &baseline_loader).is_ready(), "baseline serial load failed");
        require(baseline_cache.acquire(
            baseline_by_unique[unique], llm_cold_reference_kind::cpu_execution).is_ready(),
            "baseline CPU execution acquisition failed");
        baseline_release[order] = baseline_by_unique[unique];
    }
    const std::array<int32_t, 4> baseline_execution = {
        int32_t(baseline_by_unique[0].slot), int32_t(baseline_by_unique[1].slot),
        int32_t(baseline_by_unique[0].slot), int32_t(baseline_by_unique[2].slot),
    };
    require(baseline_cache.release_many(baseline_release.data(), baseline_release.size(),
        llm_cold_reference_kind::cpu_execution).is_ready(), "baseline CPU execution release failed");
    require(baseline_cache.policy_request_end(true, false).is_ready(), "baseline policy close failed");
    const auto baseline_diagnostics = baseline_cache.diagnostics();

    llm_cold_expert_cache cache(cache_config);
    require(cache.initialize(prototype).is_ready(), "cold cache initialization failed");

    llm_expert_layout_registry registry;
    registry.classes = { { 0, 0, cache.diagnostics().bundle_payload_bytes, prototype } };
    registry.layer_ids = { 0 };

    llm_expert_scheduler_config scheduler_config;
    scheduler_config.layer_count = 1;
    scheduler_config.experts_per_layer = 4;
    scheduler_config.request_capacity = 8;
    scheduler_config.waiters_per_request = 4;
    scheduler_config.max_current_layer_demand_flights = 4;
    scheduler_config.device_count = 1;
    scheduler_config.per_device_request_capacity = 8;
    scheduler_config.per_device_inflight_capacity = 8;
    llm_expert_scheduler scheduler(scheduler_config);

    llm_expert_async_config async_config;
    async_config.requested_queue_depth = 16;
    async_config.effective_hot_capacity = 4;
    async_config.request_capacity = 8;
    async_config.trace_capacity = 64;
    async_config.cold_cache_bytes = cache.diagnostics().actual_bytes;
    async_config.maximum_aligned_read_bytes = 4096;
    async_config.source_file_capacity = 1;
    async_config.force_positional_reads = true;
    llm_expert_async_transport transport(async_config);
    intptr_t handle = -1;
    size_t handle_count = 0;
    require(storage.copy_source_native_handles(&handle, 1, handle_count).is_ready() && handle_count == 1,
        "source handle discovery failed");
    require(transport.register_files(&handle, 1) == llm_expert_async_result::ready,
        "source registration failed");

    llm_host_resident_demand_config demand_config;
    demand_config.cache = &cache;
    demand_config.storage = &storage;
    demand_config.scheduler = &scheduler;
    demand_config.transport = &transport;
    demand_config.layout_registry = &registry;
    demand_config.maximum_occurrences = 16;
    demand_config.maximum_unique_keys = 4;
    demand_config.trace_capacity = 16;
    demand_config.serial_control = true;
    llm_host_resident_demand_coordinator coordinator(demand_config);

    const int32_t logical_ids[] = { 2, 0, 2, 1 };
    llm_host_resident_demand_batch batch;
    require(coordinator.plan(0, logical_ids, 4, batch).is_ready(), "serial demand plan failed");
    require(batch.unique_count == 3 && batch.occurrence_count == 4,
        "occurrence plan has wrong width");
    require(batch.entries[0].key.expert == 2 && batch.entries[0].occurrence_count == 2 &&
        batch.entries[1].key.expert == 0 && batch.entries[2].key.expert == 1,
        "stable first-occurrence table changed");
    std::copy(canonical_unique.begin(), canonical_unique.end(), batch.semantic_order.begin());
    require(coordinator.freeze_semantic_order(batch).is_ready(), "semantic order freeze failed");
    require(batch.semantic_order[0] == 1 && batch.semantic_order[1] == 2 &&
        batch.semantic_order[2] == 0,
        "CPU semantic key order changed");
    require(batch.occurrence_to_unique[0] == 0 && batch.occurrence_to_unique[1] == 1 &&
        batch.occurrence_to_unique[2] == 0 && batch.occurrence_to_unique[3] == 2,
        "occurrence reconstruction changed");

    std::array<llm_cold_reference, 3> execution_by_unique;
    std::array<llm_cold_reference, 3> execution_refs;
    for (size_t order = 0; order < batch.unique_count; ++order) {
        const uint32_t index = batch.semantic_order[order];
        auto & entry = batch.entries[index];
        require(coordinator.resolve_serial_next(batch).is_ready(), "serial demand resolve failed");
        require(coordinator.complete_host_scheduler(entry).is_ready(), "host scheduler completion failed");
        require(coordinator.transfer_request_hold_to_cpu_execution(entry).is_ready(),
            "CPU execution hold transfer failed");
        execution_by_unique[index] = entry.reference;
        execution_refs[order] = entry.reference;
        validate_slot(cache.bundle(), entry.reference.slot, entry.key.expert);
    }
    require(coordinator.finish_serial_batch(batch).is_ready(), "serial demand finish failed");
    require(coordinator.release_request_holds(batch).is_ready(), "request hold release failed");
    require(cache.release_many(execution_refs.data(), execution_refs.size(),
        llm_cold_reference_kind::cpu_execution).is_ready(), "CPU execution release failed");
    require(cache.policy_request_end(true, false).is_ready(), "cold policy close failed");

    const std::array<int32_t, 4> execution = {
        int32_t(execution_by_unique[0].slot), int32_t(execution_by_unique[1].slot),
        int32_t(execution_by_unique[0].slot), int32_t(execution_by_unique[2].slot),
    };
    const auto serial_diagnostics = cache.diagnostics();
    require(execution == baseline_execution, "serial execution IDs changed");
    require(serial_diagnostics.requests == baseline_diagnostics.requests &&
        serial_diagnostics.hits == baseline_diagnostics.hits &&
        serial_diagnostics.misses == baseline_diagnostics.misses &&
        serial_diagnostics.admissions == baseline_diagnostics.admissions &&
        serial_diagnostics.evictions == baseline_diagnostics.evictions &&
        serial_diagnostics.generation_changes == baseline_diagnostics.generation_changes &&
        serial_diagnostics.actual_bytes == baseline_diagnostics.actual_bytes &&
        serial_diagnostics.policy.state_digest == baseline_diagnostics.policy.state_digest &&
        serial_diagnostics.policy_events.size() == baseline_diagnostics.policy_events.size(),
        "serial cache or policy summary changed");
    for (size_t index = 0; index < serial_diagnostics.policy_events.size(); ++index) {
        require(same_policy_event(
            serial_diagnostics.policy_events[index], baseline_diagnostics.policy_events[index]),
            "serial policy transcript changed");
    }
    const auto storage_diagnostics = storage.diagnostics();
    const auto baseline_storage_diagnostics = baseline_storage.diagnostics();
    require(storage_diagnostics.read_requests == baseline_storage_diagnostics.read_requests &&
        storage_diagnostics.read_chunks == baseline_storage_diagnostics.read_chunks &&
        storage_diagnostics.read_bytes == baseline_storage_diagnostics.read_bytes,
        "serial backing read plan or bytes changed");

    const auto first = coordinator.diagnostics();
    require(first.batches == 1 && first.failures == 0 && first.current_request_holds == 0 &&
        first.peak_request_holds == 1 && first.events.size() == 1,
        "serial coordinator counters are inconsistent");
    const auto & event = first.events.front();
    require(event.new_reservations == 3 && event.ready_hits == 0 && event.read_plans == 3 &&
        event.read_operations == 3 && event.host_ready == 3 &&
        !event.first_wait_after_all_enqueue_attempts &&
        !event.first_wait_after_all_admissible_submissions && event.serial_control,
        "serial-control witness changed");
    require(event.semantic_order.size() == 3 && event.semantic_order[0].expert == 0 &&
        event.semantic_order[1].expert == 1 && event.semantic_order[2].expert == 2 &&
        event.physical_completion_order.size() == 3,
        "semantic and physical order telemetry changed");
    require(scheduler.diagnostics().active_requests == 0 &&
        transport.diagnostics().active_read_requests == 0,
        "serial demand resources did not drain");

    require(coordinator.plan(0, logical_ids, 4, batch).is_ready(), "all-hit plan failed");
    std::copy(canonical_unique.begin(), canonical_unique.end(), batch.semantic_order.begin());
    require(coordinator.freeze_semantic_order(batch).is_ready(), "all-hit semantic freeze failed");
    for (size_t order = 0; order < batch.unique_count; ++order) {
        auto & entry = batch.entries[batch.semantic_order[order]];
        require(coordinator.resolve_serial_next(batch).is_ready(), "all-hit resolve failed");
        require(coordinator.transfer_request_hold_to_cpu_execution(entry).is_ready(),
            "all-hit CPU execution hold transfer failed");
        execution_refs[order] = entry.reference;
    }
    require(coordinator.finish_serial_batch(batch).is_ready(), "all-hit finish failed");
    require(cache.release_many(execution_refs.data(), batch.unique_count,
        llm_cold_reference_kind::cpu_execution).is_ready(), "all-hit CPU execution release failed");
    require(cache.policy_request_end(true, false).is_ready(), "all-hit policy close failed");
    const auto second = coordinator.diagnostics();
    require(second.events.back().ready_hits == 3 && second.events.back().new_reservations == 0 &&
        second.events.back().read_plans == 0 && storage.diagnostics().read_bytes == 432,
        "all-hit path performed storage or scheduler work");
    require(cache.validate_invariants().is_ready(), "cold cache invariant failed");
}

struct semantic_hot_entry {
    llm_expert_key key = { -1, -1 };
    llm_cold_reference reference;
    bool occupied = false;
    bool pinned = false;
};

llm_expert_cache_policy_decision select_hot(
        llm_expert_cache_policy & policy,
        const std::array<semantic_hot_entry, 3> & entries,
        llm_expert_key key,
        uint64_t payload_bytes,
        uint64_t slot_bytes) {
    std::array<llm_expert_cache_policy_candidate, 3> candidates;
    for (size_t index = 0; index < entries.size(); ++index) {
        const auto & entry = entries[index];
        candidates[index] = { uint32_t(index), entry.reference.generation,
            { entry.key.layer, entry.key.expert }, payload_bytes, slot_bytes,
            !entry.occupied, entry.occupied && !entry.pinned };
    }
    llm_expert_cache_policy_decision decision;
    require(policy.select({ key.layer, key.expert }, candidates.data(), candidates.size(), decision).is_ready(),
        "hot semantic selection failed");
    return decision;
}

void initialize_hot_policy(
        llm_expert_cache_policy & policy,
        uint64_t slot_bytes) {
    llm_expert_cache_policy_config_internal config;
    require(llm_expert_cache_policy_copy_config(
        nullptr, llm_expert_cache_policy_tier::hot, config).is_ready(),
        "hot semantic policy config failed");
    const int32_t layers[] = { 0 };
    require(policy.initialize(config, llm_expert_cache_policy_tier::hot,
        layers, 1, 4, 3, 3, slot_bytes, 128).is_ready(),
        "hot semantic policy initialization failed");
}

void test_uma_stable_first_semantic_equivalence() {
    temporary_source source;
    llama_file baseline_file(source.path, "rb");
    llama_file common_file(source.path, "rb");
    llm_expert_storage baseline_storage(
        { 1, 4, 4, 1024 }, { { 0, &baseline_file, 512, source.path, false } });
    llm_expert_storage common_storage(
        { 1, 4, 4, 1024 }, { { 0, &common_file, 512, source.path, false } });
    for (int32_t expert = 0; expert < 4; ++expert) {
        require(baseline_storage.add_bundle({ 0, expert }, spans(uint64_t(expert)*144)).is_ready() &&
            common_storage.add_bundle({ 0, expert }, spans(uint64_t(expert)*144)).is_ready(),
            "UMA semantic storage directory failed");
    }
    require(baseline_storage.seal().is_ready() && common_storage.seal().is_ready(),
        "UMA semantic storage seal failed");

    ggml_init_params context_params = { ggml_tensor_overhead()*16, nullptr, true };
    ggml_context_ptr context(ggml_init(context_params));
    require(bool(context), "UMA semantic context failed");
    const auto prototype = make_bundle(context.get());
    llm_cold_cache_config cache_config;
    cache_config.byte_budget = 1U << 20;
    cache_config.minimum_slots = 4;
    cache_config.routed_layer_count = 1;
    cache_config.total_expert_keys = 4;
    cache_config.routed_layers = { 0 };
    llm_cold_expert_cache baseline_cache(cache_config);
    llm_cold_expert_cache common_cache(cache_config);
    require(baseline_cache.initialize(prototype).is_ready() && common_cache.initialize(prototype).is_ready(),
        "UMA semantic cache initialization failed");
    const uint64_t payload_bytes = baseline_cache.diagnostics().bundle_payload_bytes;
    const uint64_t slot_bytes = baseline_cache.diagnostics().aligned_slot_footprint;

    llm_expert_cache_policy baseline_hot;
    llm_expert_cache_policy common_hot;
    initialize_hot_policy(baseline_hot, slot_bytes);
    initialize_hot_policy(common_hot, slot_bytes);
    std::array<semantic_hot_entry, 3> baseline_hot_entries;
    std::array<semantic_hot_entry, 3> common_hot_entries;
    loader_context baseline_loader = { &baseline_storage };
    loader_context common_loader = { &common_storage };
    require(baseline_cache.policy_request_begin().is_ready() && common_cache.policy_request_begin().is_ready() &&
        baseline_hot.request_begin().is_ready() && common_hot.request_begin().is_ready(),
        "UMA semantic setup request begin failed");
    const llm_expert_key prior_key = { 0, 3 };
    require(baseline_hot.demand({ 0, 3 }, 1, payload_bytes, slot_bytes).is_ready() &&
        common_hot.demand({ 0, 3 }, 1, payload_bytes, slot_bytes).is_ready(),
        "UMA semantic setup demand failed");
    const auto baseline_prior_decision = select_hot(
        baseline_hot, baseline_hot_entries, prior_key, payload_bytes, slot_bytes);
    const auto common_prior_decision = select_hot(
        common_hot, common_hot_entries, prior_key, payload_bytes, slot_bytes);
    llm_cold_reference baseline_prior_ref;
    llm_cold_reference common_prior_ref;
    require(baseline_prior_decision.free && common_prior_decision.free &&
        baseline_cache.find_or_admit_with_loader(
            prior_key, baseline_prior_ref, load_bundle, &baseline_loader).is_ready() &&
        common_cache.find_or_admit_with_loader(
            prior_key, common_prior_ref, load_bundle, &common_loader).is_ready() &&
        baseline_hot.load_begin(baseline_prior_decision.slot, baseline_prior_ref.generation,
            { 0, 3 }, payload_bytes, slot_bytes).is_ready() &&
        common_hot.load_begin(common_prior_decision.slot, common_prior_ref.generation,
            { 0, 3 }, payload_bytes, slot_bytes).is_ready() &&
        baseline_cache.acquire(baseline_prior_ref, llm_cold_reference_kind::hot).is_ready() &&
        common_cache.acquire(common_prior_ref, llm_cold_reference_kind::hot).is_ready() &&
        baseline_hot.load_complete(baseline_prior_decision.slot, baseline_prior_ref.generation).is_ready() &&
        common_hot.load_complete(common_prior_decision.slot, common_prior_ref.generation).is_ready(),
        "UMA semantic setup publication failed");
    baseline_hot_entries[baseline_prior_decision.slot] = { prior_key, baseline_prior_ref, true, false };
    common_hot_entries[common_prior_decision.slot] = { prior_key, common_prior_ref, true, false };
    require(baseline_hot.request_end(true, false).is_ready() && common_hot.request_end(true, false).is_ready() &&
        baseline_cache.policy_request_end(true, false).is_ready() &&
        common_cache.policy_request_end(true, false).is_ready(),
        "UMA semantic setup request end failed");
    require(baseline_cache.policy_request_begin().is_ready() && common_cache.policy_request_begin().is_ready() &&
        baseline_hot.request_begin().is_ready() && common_hot.request_begin().is_ready(),
        "UMA semantic request begin failed");

    const std::array<int32_t, 3> stable_experts = { 2, 0, 1 };
    std::array<llm_cold_reference, 3> baseline_refs;
    std::array<llm_cold_reference, 3> common_refs;
    std::array<uint32_t, 3> baseline_hot_slots;
    std::array<uint32_t, 3> common_hot_slots;
    std::array<bool, 3> baseline_free_decisions;
    std::array<bool, 3> common_free_decisions;
    std::vector<int32_t> baseline_readiness;
    std::vector<int32_t> common_readiness;
    for (size_t unique = 0; unique < stable_experts.size(); ++unique) {
        const llm_expert_key key = { 0, stable_experts[unique] };
        require(baseline_hot.demand(
            { key.layer, key.expert }, unique == 0 ? 2 : 1, payload_bytes, slot_bytes).is_ready(),
            "baseline UMA hot demand failed");
        const auto decision = select_hot(baseline_hot, baseline_hot_entries, key, payload_bytes, slot_bytes);
        baseline_free_decisions[unique] = decision.free;
        if (!decision.free) {
            const auto victim = baseline_hot_entries[decision.slot];
            require(victim.occupied && !victim.pinned &&
                baseline_cache.release(victim.reference, llm_cold_reference_kind::hot).is_ready() &&
                baseline_hot.evict(decision.slot, victim.reference.generation).is_ready(),
                "baseline UMA victim demotion failed");
            baseline_hot_entries[decision.slot] = {};
        }
        require(baseline_cache.find_or_admit_with_loader(
            key, baseline_refs[unique], load_bundle, &baseline_loader).is_ready(),
            "baseline UMA cold load failed");
        require(baseline_hot.load_begin(decision.slot, baseline_refs[unique].generation,
            { key.layer, key.expert }, payload_bytes, slot_bytes).is_ready(),
            "baseline UMA hot load begin failed");
        baseline_readiness.push_back(key.expert);
        require(baseline_cache.acquire(
            baseline_refs[unique], llm_cold_reference_kind::hot).is_ready() &&
            baseline_hot.load_complete(decision.slot, baseline_refs[unique].generation).is_ready() &&
            baseline_cache.acquire(baseline_refs[unique], llm_cold_reference_kind::request).is_ready() &&
            baseline_hot.pin(decision.slot, baseline_refs[unique].generation).is_ready(),
            "baseline UMA final-reference publication failed");
        baseline_hot_entries[decision.slot] = { key, baseline_refs[unique], true, true };
        baseline_hot_slots[unique] = decision.slot;
    }

    llm_expert_layout_registry registry;
    registry.classes = { { 0, 0, common_cache.diagnostics().bundle_payload_bytes, prototype } };
    registry.layer_ids = { 0 };
    llm_expert_scheduler_config scheduler_config;
    scheduler_config.layer_count = 1;
    scheduler_config.experts_per_layer = 4;
    scheduler_config.request_capacity = 8;
    scheduler_config.waiters_per_request = 4;
    scheduler_config.max_current_layer_demand_flights = 4;
    scheduler_config.device_count = 1;
    scheduler_config.per_device_request_capacity = 8;
    scheduler_config.per_device_inflight_capacity = 8;
    llm_expert_scheduler scheduler(scheduler_config);
    llm_expert_async_config async_config;
    async_config.requested_queue_depth = 16;
    async_config.effective_hot_capacity = 4;
    async_config.request_capacity = 8;
    async_config.trace_capacity = 64;
    async_config.cold_cache_bytes = common_cache.diagnostics().actual_bytes;
    async_config.maximum_aligned_read_bytes = 4096;
    async_config.source_file_capacity = 1;
    async_config.force_positional_reads = true;
    llm_expert_async_transport transport(async_config);
    intptr_t handle = -1;
    size_t handle_count = 0;
    require(common_storage.copy_source_native_handles(&handle, 1, handle_count).is_ready() &&
        handle_count == 1 && transport.register_files(&handle, 1) == llm_expert_async_result::ready,
        "UMA semantic transport registration failed");
    llm_host_resident_demand_config demand_config;
    demand_config.cache = &common_cache;
    demand_config.storage = &common_storage;
    demand_config.scheduler = &scheduler;
    demand_config.transport = &transport;
    demand_config.layout_registry = &registry;
    demand_config.maximum_occurrences = 16;
    demand_config.maximum_unique_keys = 4;
    demand_config.trace_capacity = 16;
    demand_config.serial_control = true;
    demand_config.base_hold_kind = llm_cold_reference_kind::batch;
    llm_host_resident_demand_coordinator coordinator(demand_config);
    const int32_t logical_ids[] = { 2, 0, 2, 1 };
    llm_host_resident_demand_batch batch;
    require(coordinator.plan(0, logical_ids, 4, batch).is_ready(), "UMA semantic plan failed");
    for (size_t unique = 0; unique < batch.unique_count; ++unique) {
        batch.semantic_order[unique] = uint32_t(unique);
    }
    require(batch.occurrence_to_unique == std::vector<uint32_t>({ 0, 1, 0, 2 }),
        "UMA occurrence reconstruction changed");
    require(coordinator.freeze_semantic_order(batch).is_ready(), "UMA semantic freeze failed");
    for (size_t order = 0; order < batch.unique_count; ++order) {
        const uint32_t unique = batch.semantic_order[order];
        auto & entry = batch.entries[unique];
        const auto key = entry.key;
        require(common_hot.demand(
            { key.layer, key.expert }, entry.occurrence_count, payload_bytes, slot_bytes).is_ready(),
            "common UMA hot demand failed");
        const auto decision = select_hot(common_hot, common_hot_entries, key, payload_bytes, slot_bytes);
        common_free_decisions[unique] = decision.free;
        if (!decision.free) {
            const auto victim = common_hot_entries[decision.slot];
            require(victim.occupied && !victim.pinned &&
                common_cache.release(victim.reference, llm_cold_reference_kind::hot).is_ready() &&
                common_hot.evict(decision.slot, victim.reference.generation).is_ready(),
                "common UMA victim demotion failed");
            common_hot_entries[decision.slot] = {};
        }
        require(coordinator.resolve_serial_next(batch).is_ready(), "common UMA cold resolve failed");
        require(common_hot.load_begin(decision.slot, entry.reference.generation,
            { key.layer, key.expert }, payload_bytes, slot_bytes).is_ready(),
            "common UMA hot load begin failed");
        common_readiness.push_back(key.expert);
        require(coordinator.complete_host_scheduler(entry).is_ready(),
            "common UMA readiness terminal failed");
        require(common_cache.acquire(entry.reference, llm_cold_reference_kind::hot).is_ready() &&
            common_hot.load_complete(decision.slot, entry.reference.generation).is_ready() &&
            coordinator.transfer_request_hold(entry).is_ready() &&
            common_hot.pin(decision.slot, entry.reference.generation).is_ready(),
            "common UMA final-reference publication failed");
        common_refs[unique] = entry.reference;
        common_hot_entries[decision.slot] = { key, entry.reference, true, true };
        common_hot_slots[unique] = decision.slot;
    }
    require(coordinator.finish_serial_batch(batch).is_ready(), "common UMA semantic finish failed");

    for (size_t reverse = stable_experts.size(); reverse-- > 0;) {
        require(baseline_hot.unpin(baseline_hot_slots[reverse], baseline_refs[reverse].generation).is_ready() &&
            baseline_cache.release(baseline_refs[reverse], llm_cold_reference_kind::request).is_ready() &&
            common_hot.unpin(common_hot_slots[reverse], common_refs[reverse].generation).is_ready() &&
            common_cache.release(common_refs[reverse], llm_cold_reference_kind::request).is_ready(),
            "UMA semantic request release failed");
    }
    require(baseline_hot.request_end(true, false).is_ready() && common_hot.request_end(true, false).is_ready() &&
        baseline_cache.policy_request_end(true, false).is_ready() &&
        common_cache.policy_request_end(true, false).is_ready(),
        "UMA semantic request end failed");

    require(baseline_readiness == common_readiness &&
        common_readiness == std::vector<int32_t>({ 2, 0, 1 }),
        "UMA readiness order was canonicalized");
    const std::array<int32_t, 4> baseline_execution = {
        int32_t(baseline_refs[0].slot), int32_t(baseline_refs[1].slot),
        int32_t(baseline_refs[0].slot), int32_t(baseline_refs[2].slot),
    };
    const std::array<int32_t, 4> common_execution = {
        int32_t(common_refs[0].slot), int32_t(common_refs[1].slot),
        int32_t(common_refs[0].slot), int32_t(common_refs[2].slot),
    };
    require(baseline_execution == common_execution && baseline_hot_slots == common_hot_slots &&
        baseline_free_decisions == common_free_decisions &&
        baseline_free_decisions == std::array<bool, 3>({ true, true, false }),
        "UMA execution IDs or hot admission decisions changed");
    for (size_t unique = 0; unique < stable_experts.size(); ++unique) {
        require(baseline_refs[unique].slot == common_refs[unique].slot &&
            baseline_refs[unique].generation == common_refs[unique].generation,
            "UMA cold slot or generation changed");
    }
    const auto baseline_cold = baseline_cache.diagnostics();
    const auto common_cold = common_cache.diagnostics();
    require(baseline_cold.policy.state_digest == common_cold.policy.state_digest &&
        baseline_cold.policy_events.size() == common_cold.policy_events.size() &&
        baseline_cold.current_hot_refs == 3 && common_cold.current_hot_refs == 3 &&
        baseline_cold.current_request_refs == 0 && common_cold.current_request_refs == 0 &&
        baseline_cold.current_transfer_refs == 0 && common_cold.current_transfer_refs == 0 &&
        baseline_cold.current_cpu_execution_refs == 0 && common_cold.current_cpu_execution_refs == 0 &&
        baseline_cold.current_batch_refs == 0 && common_cold.current_batch_refs == 0 &&
        baseline_cold.slots.size() == common_cold.slots.size(),
        "UMA cold policy or terminal references changed");
    for (size_t index = 0; index < baseline_cold.policy_events.size(); ++index) {
        require(same_policy_event(baseline_cold.policy_events[index], common_cold.policy_events[index]),
            "UMA cold policy transcript changed");
    }
    for (size_t index = 0; index < baseline_cold.slots.size(); ++index) {
        const auto & lhs = baseline_cold.slots[index];
        const auto & rhs = common_cold.slots[index];
        require(lhs.key.layer == rhs.key.layer && lhs.key.expert == rhs.key.expert &&
            lhs.layout_class_id == rhs.layout_class_id && lhs.generation == rhs.generation &&
            lhs.state == rhs.state &&
            lhs.hot_refs == rhs.hot_refs && lhs.request_refs == rhs.request_refs &&
            lhs.transfer_refs == rhs.transfer_refs &&
            lhs.cpu_execution_refs == rhs.cpu_execution_refs && lhs.batch_refs == rhs.batch_refs,
            "UMA cold directory contents changed");
    }
    require(baseline_hot.diagnostics().state_digest == common_hot.diagnostics().state_digest &&
        baseline_hot.transcript_size() == common_hot.transcript_size(),
        "UMA hot policy digest changed");
    for (size_t index = 0; index < baseline_hot.transcript_size(); ++index) {
        require(same_policy_event(baseline_hot.transcript()[index], common_hot.transcript()[index]),
            "UMA hot policy transcript changed");
    }
    const auto baseline_reads = baseline_storage.diagnostics();
    const auto common_reads = common_storage.diagnostics();
    require(baseline_reads.read_requests == common_reads.read_requests &&
        baseline_reads.read_chunks == common_reads.read_chunks &&
        baseline_reads.read_bytes == common_reads.read_bytes,
        "UMA backing read plan or bytes changed");
    const auto demand_diagnostics = coordinator.diagnostics();
    require(demand_diagnostics.events.size() == 1 &&
        demand_diagnostics.events[0].semantic_order.size() == 3 &&
        demand_diagnostics.events[0].semantic_order[0].expert == 2 &&
        demand_diagnostics.events[0].semantic_order[1].expert == 0 &&
        demand_diagnostics.events[0].semantic_order[2].expert == 1 &&
        demand_diagnostics.events[0].physical_completion_order.size() == 3,
        "UMA semantic and physical telemetry were not separated");
    require(scheduler.diagnostics().active_requests == 0,
        "UMA semantic scheduler did not drain");
    require(transport.diagnostics().active_read_requests == 0,
        "UMA semantic transport did not drain");
}

enum class batch_reader_mode {
    ready,
    short_read,
    io_error,
};

struct deterministic_batch_reader : llm_expert_async_read_override {
    batch_reader_mode mode = batch_reader_mode::ready;
    bool slow_first_expert = false;
    bool stagger_by_expert = false;

    int64_t read_at(
            intptr_t,
            void * destination,
            size_t byte_count,
            uint64_t file_offset,
            int & native_error) noexcept override {
        const uint64_t expert = file_offset/144;
        const uint64_t stagger_us = stagger_by_expert ?
            (1 + ((expert*5 + 3)%7))*1000 : 1000;
        usleep(slow_first_expert && file_offset < 144 ? 30000 : stagger_us);
        if (mode == batch_reader_mode::io_error) {
            native_error = EIO;
            return -1;
        }
        auto * bytes = static_cast<uint8_t *>(destination);
        for (size_t index = 0; index < byte_count; ++index) {
            bytes[index] = uint8_t((file_offset + index)*17U + 3U);
        }
        native_error = 0;
        return mode == batch_reader_mode::short_read && byte_count != 0 ?
            int64_t(byte_count - 1) : int64_t(byte_count);
    }
};

struct batch_case_result {
    llm_expert_provider_error error = llm_expert_provider_error::none;
    uint64_t failures = 0;
    uint64_t cancellations = 0;
    llm_host_resident_demand_event event;
    llm_cold_cache_diagnostics cache;
    llm_expert_storage_diagnostics storage;
    std::vector<int32_t> execution_ids;
};

struct batch_case_options {
    bool reverse_completion = false;
    bool reverse_issue = false;
    bool staggered_completion = false;
    bool cancel_on_first_wait = false;
    bool cancel_after_completion_before_publication = false;
    bool cancel_after_host_ready = false;
    bool close_transport_before_resolve = false;
    bool integrity_mismatch = false;
    bool expect_failure = false;
    bool retry_after_failure = false;
    batch_reader_mode reader_mode = batch_reader_mode::ready;
    uint32_t scheduler_capacity = 32;
    uint32_t transport_request_capacity = 32;
    uint64_t initial_cache_generation = 0;
};

batch_case_result run_batch_case(
        const std::vector<int32_t> & logical_ids,
        const std::vector<int32_t> & warm_ids = {},
        batch_case_options options = {}) {
    constexpr uint32_t expert_count = 16;
    wide_temporary_source source(expert_count);
    llama_file file(source.path, "rb");
    llm_expert_storage storage(
        { 1, expert_count, expert_count, 4096 }, { { 0, &file, 512, source.path, false } });
    for (int32_t expert = 0; expert < int32_t(expert_count); ++expert) {
        require(storage.add_bundle({ 0, expert }, spans(uint64_t(expert)*144)).is_ready(),
            "batch storage directory failed");
    }
    require(storage.seal().is_ready(), "batch storage seal failed");

    ggml_init_params context_params = { ggml_tensor_overhead()*32, nullptr, true };
    ggml_context_ptr context(ggml_init(context_params));
    require(bool(context), "batch context failed");
    const auto prototype = make_bundle(context.get(), expert_count);

    llm_cold_cache_config cache_config;
    cache_config.byte_budget = 1U << 20;
    cache_config.minimum_slots = expert_count;
    cache_config.routed_layer_count = 1;
    cache_config.total_expert_keys = expert_count;
    cache_config.initial_slot_generation_for_testing = options.initial_cache_generation;
    cache_config.routed_layers = { 0 };
    llm_cold_expert_cache cache(cache_config);
    require(cache.initialize(prototype).is_ready(), "batch cache initialization failed");

    llm_expert_layout_registry registry;
    registry.classes = { { 0, 0, cache.diagnostics().bundle_payload_bytes, prototype } };
    registry.layer_ids = { 0 };

    llm_expert_scheduler_config scheduler_config;
    scheduler_config.layer_count = 1;
    scheduler_config.experts_per_layer = expert_count;
    scheduler_config.request_capacity = options.scheduler_capacity;
    scheduler_config.waiters_per_request = 16;
    scheduler_config.max_current_layer_demand_flights = expert_count;
    scheduler_config.device_count = 1;
    scheduler_config.per_device_request_capacity = options.scheduler_capacity;
    scheduler_config.per_device_inflight_capacity = options.scheduler_capacity;
    llm_expert_scheduler scheduler(scheduler_config);

    deterministic_batch_reader reader;
    reader.mode = options.reader_mode;
    reader.stagger_by_expert = options.staggered_completion;
    llm_expert_async_config async_config;
    async_config.requested_queue_depth = 64;
    async_config.effective_hot_capacity = expert_count;
    async_config.request_capacity = options.transport_request_capacity;
    async_config.trace_capacity = 128;
    async_config.cold_cache_bytes = cache.diagnostics().actual_bytes;
    async_config.maximum_aligned_read_bytes = 4096;
    async_config.source_file_capacity = 1;
    async_config.read_override_for_testing = &reader;
    async_config.reverse_queued_requests_for_testing = options.reverse_completion;
    async_config.force_positional_reads = true;
    async_config.worker_count = options.staggered_completion ? 4 : 1;
    llm_expert_async_transport transport(async_config);
    intptr_t handle = -1;
    size_t handle_count = 0;
    require(storage.copy_source_native_handles(&handle, 1, handle_count).is_ready() &&
        handle_count == 1 && transport.register_files(&handle, 1) == llm_expert_async_result::ready,
        "batch source registration failed");

    llm_host_resident_demand_config demand_config;
    demand_config.cache = &cache;
    demand_config.storage = &storage;
    demand_config.scheduler = &scheduler;
    demand_config.transport = &transport;
    demand_config.layout_registry = &registry;
    demand_config.maximum_occurrences = 32;
    demand_config.maximum_unique_keys = expert_count;
    demand_config.trace_capacity = 16;
    demand_config.serial_control = false;
    demand_config.base_hold_kind = llm_cold_reference_kind::batch;
    bool publication_cancel_requested = false;
    if (options.cancel_after_completion_before_publication) {
        demand_config.before_semantic_publication_for_testing = [](void * data) {
            *static_cast<bool *>(data) = true;
        };
        demand_config.before_semantic_publication_data_for_testing =
            &publication_cancel_requested;
    }
    if (options.integrity_mismatch) {
        demand_config.integrity_mode = llm_expert_integrity_mode::fnv64_end_to_end;
    }
    llm_host_resident_demand_coordinator coordinator(demand_config);

    llm_expert_provider_error last_error = llm_expert_provider_error::none;
    bool transport_closed = false;
    auto execute = [&](const std::vector<int32_t> & ids, bool alternate_issue, bool may_fail) {
        require(!ids.empty(), "batch execution requires logical occurrences");
        require(cache.policy_request_begin().is_ready(), "batch policy begin failed");
        llm_host_resident_demand_batch batch;
        require(coordinator.plan(0, ids.data(), ids.size(), batch).is_ready(),
            "batch demand plan failed");
        std::iota(batch.semantic_order.begin(), batch.semantic_order.end(), 0U);
        std::sort(batch.semantic_order.begin(), batch.semantic_order.end(), [&](uint32_t lhs, uint32_t rhs) {
            return batch.entries[lhs].key.expert < batch.entries[rhs].key.expert;
        });
        if (alternate_issue) {
            batch.issue_order = batch.semantic_order;
            std::reverse(batch.issue_order.begin(), batch.issue_order.end());
        }
        require(coordinator.freeze_semantic_order(batch).is_ready(),
            "batch semantic freeze failed");
        if (options.close_transport_before_resolve && !transport_closed) {
            require(transport.shutdown(), "batch pre-submit transport shutdown failed");
            transport_closed = true;
        }
        bool cancel_requested = options.cancel_on_first_wait;
        const auto abort_callback = [](void * data) {
            return *static_cast<bool *>(data);
        };
        bool (*callback)(void *) =
            (options.cancel_on_first_wait || options.cancel_after_completion_before_publication) ?
                +abort_callback : nullptr;
        bool * callback_data = options.cancel_after_completion_before_publication ?
            &publication_cancel_requested : &cancel_requested;
        const auto resolved = coordinator.resolve_batch(
            batch, callback, callback != nullptr ? callback_data : nullptr);
        if (!resolved.is_ready()) {
            last_error = resolved.error;
            require(may_fail, "batched demand resolve failed");
            require(cache.policy_request_end(false,
                resolved.error == llm_expert_provider_error::cancelled).is_ready(),
                "failed batch policy end failed");
            return std::vector<int32_t>{};
        }
        require(!may_fail || options.cancel_after_host_ready,
            "batched demand unexpectedly succeeded");
        if (options.cancel_after_host_ready) {
            const auto cancelled = coordinator.fail_serial_batch(
                batch, llm_expert_provider_error::cancelled, true);
            require(cancelled.error == llm_expert_provider_error::cancelled,
                "HOST_READY cancellation status changed");
            last_error = cancelled.error;
            require(cache.policy_request_end(false, true).is_ready(),
                "HOST_READY cancellation policy end failed");
            return std::vector<int32_t>{};
        }

        std::vector<llm_cold_reference> execution_refs;
        execution_refs.reserve(batch.unique_count);
        for (size_t order = 0; order < batch.unique_count; ++order) {
            auto & entry = batch.entries[batch.semantic_order[order]];
            require(coordinator.complete_host_scheduler(entry).is_ready(),
                "batch host scheduler completion failed");
            require(coordinator.transfer_request_hold_to_cpu_execution(entry).is_ready(),
                "batch CPU execution hold transfer failed");
            execution_refs.push_back(entry.reference);
            validate_slot(cache.bundle(), entry.reference.slot, entry.key.expert);
        }
        require(coordinator.finish_serial_batch(batch).is_ready(), "batch finish failed");
        std::vector<int32_t> result(ids.size(), -1);
        for (size_t occurrence = 0; occurrence < ids.size(); ++occurrence) {
            result[occurrence] = int32_t(batch.entries[
                batch.occurrence_to_unique[occurrence]].reference.slot);
        }
        require(cache.release_many(execution_refs.data(), execution_refs.size(),
            llm_cold_reference_kind::cpu_execution).is_ready(),
            "batch CPU execution release failed");
        require(cache.policy_request_end(true, false).is_ready(), "batch policy end failed");
        return result;
    };

    if (!warm_ids.empty()) (void) execute(warm_ids, false, false);
    batch_case_result result;
    result.execution_ids = execute(logical_ids, options.reverse_issue, options.expect_failure);
    if (options.retry_after_failure) {
        require(options.expect_failure && last_error != llm_expert_provider_error::none,
            "batch retry requires a fully drained first failure");
        reader.mode = batch_reader_mode::ready;
        result.execution_ids = execute(logical_ids, options.reverse_issue, false);
    }
    const auto coordinator_diagnostics = coordinator.diagnostics();
    result.error = last_error;
    result.failures = coordinator_diagnostics.failures;
    result.cancellations = coordinator_diagnostics.cancellations;
    if (!options.expect_failure || options.retry_after_failure) {
        require(!coordinator_diagnostics.events.empty(), "batch event missing");
        result.event = coordinator_diagnostics.events.back();
    }
    result.cache = cache.diagnostics();
    result.storage = storage.diagnostics();
    require(result.cache.current_request_refs == 0 && result.cache.current_cpu_execution_refs == 0 &&
        result.cache.current_batch_refs == 0 && scheduler.diagnostics().active_requests == 0 &&
        transport.diagnostics().active_read_requests == 0 &&
        transport.diagnostics().active_operations == 0,
        "batch terminal resources did not drain");
    require(cache.validate_invariants().is_ready(), "batch cache invariant failed");
    return result;
}

void require_same_batch_semantics(
        const batch_case_result & lhs,
        const batch_case_result & rhs,
        const char * message) {
    require(lhs.execution_ids == rhs.execution_ids &&
        lhs.cache.policy.state_digest == rhs.cache.policy.state_digest &&
        lhs.cache.requests == rhs.cache.requests && lhs.cache.hits == rhs.cache.hits &&
        lhs.cache.misses == rhs.cache.misses && lhs.cache.admissions == rhs.cache.admissions &&
        lhs.cache.evictions == rhs.cache.evictions &&
        lhs.cache.generation_changes == rhs.cache.generation_changes &&
        lhs.cache.policy_events.size() == rhs.cache.policy_events.size() &&
        lhs.cache.slots.size() == rhs.cache.slots.size() &&
        lhs.storage.read_requests == rhs.storage.read_requests &&
        lhs.storage.read_chunks == rhs.storage.read_chunks &&
        lhs.storage.read_bytes == rhs.storage.read_bytes,
        message);
    for (size_t index = 0; index < lhs.cache.policy_events.size(); ++index) {
        require(same_policy_event(lhs.cache.policy_events[index], rhs.cache.policy_events[index]), message);
    }
    for (size_t index = 0; index < lhs.cache.slots.size(); ++index) {
        const auto & left = lhs.cache.slots[index];
        const auto & right = rhs.cache.slots[index];
        require(left.key.layer == right.key.layer && left.key.expert == right.key.expert &&
            left.layout_class_id == right.layout_class_id && left.generation == right.generation &&
            left.last_use == right.last_use && left.state == right.state &&
            left.hot_refs == right.hot_refs && left.request_refs == right.request_refs &&
            left.transfer_refs == right.transfer_refs &&
            left.cpu_execution_refs == right.cpu_execution_refs && left.batch_refs == right.batch_refs,
            message);
    }
}

void test_batched_issue_ahead_and_determinism() {
    for (uint32_t width : { 1U, 2U, 4U, 8U, 16U }) {
        std::vector<int32_t> ids(width);
        std::iota(ids.begin(), ids.end(), 0);
        const auto result = run_batch_case(ids);
        require(result.event.new_reservations == width && result.event.ready_hits == 0 &&
            result.event.scheduler_enqueue_attempts == width &&
            result.event.scheduler_admissions == width && result.event.read_plans == width &&
            result.event.host_ready == width && result.event.adapter_ready == width &&
            result.event.first_wait_after_all_enqueue_attempts &&
            result.event.first_wait_after_all_admissible_submissions &&
            result.event.first_wait_reason ==
                llm_host_resident_wait_reason::read_completion &&
            !result.event.serial_control && result.event.issue_order.size() == width &&
            result.event.physical_completion_order.size() == width,
            "batch issue-ahead witness changed");
        require(result.event.peak_active_read_requests == width &&
            result.event.peak_active_read_operations >= width,
            "batch did not expose complete current-layer read concurrency");
    }

    const std::vector<int32_t> four = { 0, 1, 2, 3 };
    const auto all_hit = run_batch_case(four, four);
    require(all_hit.event.ready_hits == 4 && all_hit.event.new_reservations == 0 &&
        all_hit.event.scheduler_enqueue_attempts == 0 && all_hit.event.read_plans == 0 &&
        all_hit.event.first_wait_us == 0 &&
        all_hit.event.first_wait_reason == llm_host_resident_wait_reason::none &&
        all_hit.event.first_wait_after_all_enqueue_attempts &&
        all_hit.event.first_wait_after_all_admissible_submissions,
        "batch all-hit path performed scheduler or storage work");

    const auto mixed = run_batch_case(four, { 0, 1 });
    require(mixed.event.ready_hits == 2 && mixed.event.new_reservations == 2 &&
        mixed.event.scheduler_admissions == 2 && mixed.event.read_plans == 2 &&
        mixed.event.peak_active_read_requests == 2,
        "batch mixed hit/miss classification changed");

    const std::vector<int32_t> duplicates = { 2, 0, 2, 1 };
    const auto duplicate = run_batch_case(duplicates);
    require(duplicate.event.selected_occurrences == 4 && duplicate.event.unique_keys == 3 &&
        duplicate.event.new_reservations == 3 && duplicate.event.semantic_order.size() == 3 &&
        duplicate.event.semantic_order[0].expert == 0 &&
        duplicate.event.semantic_order[1].expert == 1 &&
        duplicate.event.semantic_order[2].expert == 2 &&
        duplicate.execution_ids[0] == duplicate.execution_ids[2],
        "batch duplicate occurrence reconstruction changed");

    const std::vector<int32_t> permuted = { 5, 1, 7, 3 };
    const auto normal = run_batch_case(permuted);
    batch_case_options reversed_options;
    reversed_options.reverse_completion = true;
    const auto reversed = run_batch_case(permuted, {}, reversed_options);
    batch_case_options issue_options;
    issue_options.reverse_issue = true;
    const auto alternate_issue = run_batch_case(permuted, {}, issue_options);
    batch_case_options staggered_options;
    staggered_options.staggered_completion = true;
    const auto staggered = run_batch_case(permuted, {}, staggered_options);
    require_same_batch_semantics(normal, reversed,
        "reversed completion changed batch semantic state");
    require_same_batch_semantics(normal, alternate_issue,
        "alternate issue order changed batch semantic state");
    require_same_batch_semantics(normal, staggered,
        "staggered completion changed batch semantic state");
    require(same_key_sequence(normal.event.semantic_order, reversed.event.semantic_order) &&
        same_key_sequence(normal.event.semantic_order, alternate_issue.event.semantic_order) &&
        same_key_sequence(normal.event.semantic_order, staggered.event.semantic_order) &&
        !same_key_sequence(normal.event.issue_order, alternate_issue.event.issue_order) &&
        !same_key_sequence(normal.event.physical_completion_order,
            reversed.event.physical_completion_order) &&
        !same_key_sequence(normal.event.physical_completion_order,
            staggered.event.physical_completion_order),
        "batch occurrence, semantic, issue, and completion orders were not separated");
}

void require_failed_batch_drained(
        const batch_case_result & result,
        llm_expert_provider_error expected,
        bool cancelled = false) {
    require(result.error == expected && result.failures == 1 &&
        result.cancellations == uint64_t(cancelled) &&
        result.cache.current_hot_refs == 0 && result.cache.current_request_refs == 0 &&
        result.cache.current_transfer_refs == 0 &&
        result.cache.current_cpu_execution_refs == 0 && result.cache.current_batch_refs == 0 &&
        std::none_of(result.cache.slots.begin(), result.cache.slots.end(), [](const auto & slot) {
            return slot.state == llm_cold_slot_state::loading ||
                slot.state == llm_cold_slot_state::reserved ||
                slot.state == llm_cold_slot_state::failed;
        }),
        "failed batch did not drain exact generation ownership");
}

void test_batched_failure_cleanup() {
    const std::vector<int32_t> ids = { 0, 1, 2, 3 };

    batch_case_options short_read;
    short_read.expect_failure = true;
    short_read.retry_after_failure = true;
    short_read.reader_mode = batch_reader_mode::short_read;
    const auto retried = run_batch_case(ids, {}, short_read);
    require_failed_batch_drained(retried, llm_expert_provider_error::copy_failed);
    require(retried.execution_ids.size() == ids.size() && retried.cache.admissions == ids.size(),
        "batch retry after complete drain did not publish a fresh exact generation");

    batch_case_options io_error;
    io_error.expect_failure = true;
    io_error.reader_mode = batch_reader_mode::io_error;
    require_failed_batch_drained(
        run_batch_case(ids, {}, io_error), llm_expert_provider_error::copy_failed);

    batch_case_options cancellation;
    cancellation.expect_failure = true;
    cancellation.cancel_on_first_wait = true;
    require_failed_batch_drained(
        run_batch_case(ids, {}, cancellation), llm_expert_provider_error::cancelled, true);

    batch_case_options pre_submit_cancellation;
    pre_submit_cancellation.expect_failure = true;
    pre_submit_cancellation.close_transport_before_resolve = true;
    const auto cancelled_before_submit = run_batch_case(ids, {}, pre_submit_cancellation);
    require_failed_batch_drained(
        cancelled_before_submit, llm_expert_provider_error::cancelled, true);
    require(cancelled_before_submit.storage.read_requests == 0,
        "pre-submit cancellation reached backing storage");

    batch_case_options completed_cancellation;
    completed_cancellation.expect_failure = true;
    completed_cancellation.cancel_after_completion_before_publication = true;
    require_failed_batch_drained(
        run_batch_case(ids, {}, completed_cancellation),
        llm_expert_provider_error::cancelled, true);

    batch_case_options host_ready_cancellation;
    host_ready_cancellation.expect_failure = true;
    host_ready_cancellation.cancel_after_host_ready = true;
    const auto cancelled_after_host_ready = run_batch_case(ids, {}, host_ready_cancellation);
    require_failed_batch_drained(
        cancelled_after_host_ready, llm_expert_provider_error::cancelled, true);
    require(cancelled_after_host_ready.cache.admissions == ids.size(),
        "HOST_READY cancellation discarded valid ready content");

    batch_case_options integrity_failure;
    integrity_failure.expect_failure = true;
    integrity_failure.integrity_mismatch = true;
    require_failed_batch_drained(
        run_batch_case(ids, {}, integrity_failure),
        llm_expert_provider_error::metadata_mismatch);

    batch_case_options scheduler_saturation;
    scheduler_saturation.expect_failure = true;
    scheduler_saturation.scheduler_capacity = 2;
    const auto scheduler_failure = run_batch_case(ids, {}, scheduler_saturation);
    require_failed_batch_drained(scheduler_failure, llm_expert_provider_error::busy);
    require(scheduler_failure.storage.read_requests == 0,
        "scheduler saturation submitted partial storage work");

    batch_case_options transport_saturation;
    transport_saturation.expect_failure = true;
    transport_saturation.transport_request_capacity = 2;
    require_failed_batch_drained(
        run_batch_case(ids, {}, transport_saturation), llm_expert_provider_error::copy_failed);

    batch_case_options generation_exhaustion;
    generation_exhaustion.expect_failure = true;
    generation_exhaustion.initial_cache_generation = std::numeric_limits<uint64_t>::max();
    const auto generation_failure = run_batch_case(ids, {}, generation_exhaustion);
    require_failed_batch_drained(
        generation_failure, llm_expert_provider_error::generation_exhausted);
    require(generation_failure.storage.read_requests == 0,
        "generation exhaustion submitted storage work");
}

void run_batched_join_case(bool predecessor_queued, bool predecessor_draining = false) {
    temporary_source source;
    llama_file file(source.path, "rb");
    llm_expert_storage storage({ 1, 4, 4, 1024 }, { { 0, &file, 512, source.path, false } });
    for (int32_t expert = 0; expert < 4; ++expert) {
        require(storage.add_bundle({ 0, expert }, spans(uint64_t(expert)*144)).is_ready(),
            "join storage directory failed");
    }
    require(storage.seal().is_ready(), "join storage seal failed");

    ggml_init_params context_params = { ggml_tensor_overhead()*16, nullptr, true };
    ggml_context_ptr context(ggml_init(context_params));
    require(bool(context), "join context failed");
    const auto prototype = make_bundle(context.get());
    llm_cold_cache_config cache_config;
    cache_config.byte_budget = 1U << 20;
    cache_config.minimum_slots = 4;
    cache_config.routed_layer_count = 1;
    cache_config.total_expert_keys = 4;
    cache_config.routed_layers = { 0 };
    llm_cold_expert_cache cache(cache_config);
    require(cache.initialize(prototype).is_ready(), "join cache initialization failed");
    require(cache.policy_request_begin().is_ready(), "join policy begin failed");

    llm_expert_layout_registry registry;
    registry.classes = { { 0, 0, cache.diagnostics().bundle_payload_bytes, prototype } };
    registry.layer_ids = { 0 };
    llm_expert_scheduler_config scheduler_config;
    scheduler_config.layer_count = 1;
    scheduler_config.experts_per_layer = 4;
    scheduler_config.request_capacity = 8;
    scheduler_config.waiters_per_request = 4;
    scheduler_config.max_current_layer_demand_flights = 4;
    scheduler_config.device_count = 1;
    scheduler_config.per_device_request_capacity = 8;
    scheduler_config.per_device_inflight_capacity = 8;
    llm_expert_scheduler scheduler(scheduler_config);

    deterministic_batch_reader reader;
    reader.slow_first_expert = true;
    llm_expert_async_config async_config;
    async_config.requested_queue_depth = 16;
    async_config.effective_hot_capacity = 4;
    async_config.request_capacity = 8;
    async_config.trace_capacity = 64;
    async_config.cold_cache_bytes = cache.diagnostics().actual_bytes;
    async_config.maximum_aligned_read_bytes = 4096;
    async_config.source_file_capacity = 1;
    async_config.read_override_for_testing = &reader;
    async_config.force_positional_reads = true;
    async_config.worker_count = 2;
    llm_expert_async_transport transport(async_config);
    intptr_t source_handle = -1;
    size_t handle_count = 0;
    require(storage.copy_source_native_handles(&source_handle, 1, handle_count).is_ready() &&
        handle_count == 1 && transport.register_files(&source_handle, 1) == llm_expert_async_result::ready,
        "join source registration failed");

    llm_cold_reference predecessor_reference;
    llm_cold_demand_lookup predecessor_lookup = llm_cold_demand_lookup::missing;
    require(cache.reserve_or_join_demand(
        { 0, 0 }, predecessor_reference, predecessor_lookup).is_ready() &&
        predecessor_lookup == llm_cold_demand_lookup::reserved,
        "join predecessor reservation failed");
    llm_expert_request_metadata predecessor_metadata;
    predecessor_metadata.layout_class_id = predecessor_reference.layout_class_id;
    const auto predecessor = scheduler.enqueue(
        { 0, 0 }, llm_expert_priority::demand_current_layer,
        llm_expert_readiness::host_ready, predecessor_metadata);
    require(predecessor.accepted(), "join predecessor scheduler admission failed");

    std::array<llm_expert_storage_destination, 3> destinations;
    const auto & target = cache.bundle();
    const std::array<std::pair<llm_expert_storage_projection, ggml_tensor *>, 3> projections = {{
        { llm_expert_storage_projection::up, target.up.weight },
        { llm_expert_storage_projection::gate, target.gate.weight },
        { llm_expert_storage_projection::down, target.down.weight },
    }};
    for (size_t index = 0; index < projections.size(); ++index) {
        auto * tensor = projections[index].second;
        destinations[index] = {
            projections[index].first,
            llm_expert_storage_sidecar::weight,
            static_cast<uint8_t *>(tensor->data) +
                size_t(predecessor_reference.slot)*tensor->nb[2],
            tensor->nb[1]*uint64_t(tensor->ne[1]),
            0,
        };
    }
    std::array<llm_expert_storage_read_operation, 3> operations;
    size_t operation_count = 0;
    require(storage.make_read_plan(
        { 0, 0 }, destinations.data(), destinations.size(), operations.data(),
        operations.size(), operation_count).is_ready() && operation_count != 0,
        "join predecessor read plan failed");

    std::promise<void> predecessor_submitted;
    auto predecessor_submitted_future = predecessor_submitted.get_future();
    auto predecessor_worker = std::async(std::launch::async, [&]() {
        if (predecessor_queued) usleep(20000);
        llm_expert_request_snapshot selected;
        if (!scheduler.take(predecessor.handle, selected).accepted()) {
            predecessor_submitted.set_value();
            return false;
        }
        if (predecessor_draining) {
            if (scheduler.transition(predecessor.handle, llm_expert_request_state::submitting,
                    llm_expert_request_state::draining) !=
                    llm_expert_schedule_disposition::admitted) {
                predecessor_submitted.set_value();
                return false;
            }
            if (!cache.fail_reservation({ 0, 0 }, predecessor_reference).is_ready() ||
                !cache.cleanup_failed_slots().is_ready()) {
                predecessor_submitted.set_value();
                return false;
            }
            predecessor_submitted.set_value();
            usleep(30000);
            return scheduler.finish(predecessor.handle, llm_expert_request_state::failed) ==
                    llm_expert_schedule_disposition::admitted &&
                scheduler.release_terminal(predecessor.handle) ==
                    llm_expert_schedule_disposition::admitted;
        }
        const llm_expert_async_operation_identity identity = {
            transport.diagnostics().transport_epoch,
            predecessor.handle,
            0,
            { 0, 0 },
            llm_expert_readiness::host_ready,
            llm_expert_priority::demand_current_layer,
            predecessor_reference.layout_class_id,
        };
        if (transport.submit_read_plan(identity, operations.data(), operation_count) !=
                llm_expert_async_result::ready ||
            scheduler.transition(predecessor.handle, llm_expert_request_state::submitting,
                llm_expert_request_state::io_in_flight) !=
                llm_expert_schedule_disposition::admitted) {
            predecessor_submitted.set_value();
            return false;
        }
        predecessor_submitted.set_value();
        llm_expert_async_read_completion completion;
        const auto waited = transport.wait_read(predecessor.handle, completion);
        const auto released = transport.release_read(predecessor.handle);
        storage.record_async_read(
            operation_count, completion.bytes_completed,
            waited == llm_expert_async_result::ready ? llm_expert_storage_error::none :
                llm_expert_storage_error::io_error,
            completion.native_error);
        if (waited != llm_expert_async_result::ready ||
            released != llm_expert_async_result::ready ||
            !cache.publish_ready({ 0, 0 }, predecessor_reference).is_ready() ||
            scheduler.transition(predecessor.handle, llm_expert_request_state::io_in_flight,
                llm_expert_request_state::host_ready) !=
                llm_expert_schedule_disposition::admitted ||
            scheduler.finish(predecessor.handle, llm_expert_request_state::complete) !=
                llm_expert_schedule_disposition::admitted ||
            scheduler.release_terminal(predecessor.handle) !=
                llm_expert_schedule_disposition::admitted) {
            return false;
        }
        return true;
    });
    if (!predecessor_queued) predecessor_submitted_future.wait();

    llm_host_resident_demand_config demand_config;
    demand_config.cache = &cache;
    demand_config.storage = &storage;
    demand_config.scheduler = &scheduler;
    demand_config.transport = &transport;
    demand_config.layout_registry = &registry;
    demand_config.maximum_occurrences = 4;
    demand_config.maximum_unique_keys = 4;
    demand_config.trace_capacity = 8;
    demand_config.serial_control = false;
    demand_config.base_hold_kind = llm_cold_reference_kind::batch;
    llm_host_resident_demand_coordinator coordinator(demand_config);
    const int32_t logical_ids[] = { 0, 1, 2 };
    llm_host_resident_demand_batch batch;
    require(coordinator.plan(0, logical_ids, 3, batch).is_ready(), "join batch plan failed");
    std::iota(batch.semantic_order.begin(), batch.semantic_order.end(), 0U);
    require(coordinator.freeze_semantic_order(batch).is_ready(), "join semantic freeze failed");
    const auto joined_resolve = coordinator.resolve_batch(batch);
    require(joined_resolve.is_ready(), "joined batch resolve failed");
    require(predecessor_worker.get(), "join predecessor worker failed");

    std::array<llm_cold_reference, 3> execution_refs;
    for (size_t order = 0; order < batch.unique_count; ++order) {
        auto & entry = batch.entries[batch.semantic_order[order]];
        require(coordinator.complete_host_scheduler(entry).is_ready(),
            "join scheduler completion failed");
        require(coordinator.transfer_request_hold_to_cpu_execution(entry).is_ready(),
            "join CPU execution transfer failed");
        execution_refs[order] = entry.reference;
    }
    require(coordinator.finish_serial_batch(batch).is_ready(), "join batch finish failed");
    require(cache.release_many(execution_refs.data(), batch.unique_count,
        llm_cold_reference_kind::cpu_execution).is_ready(), "join execution release failed");
    require(cache.policy_request_end(true, false).is_ready(), "join policy end failed");
    const auto event = coordinator.diagnostics().events.back();
    const uint32_t expected_new_reservations = predecessor_draining ? 3 : 2;
    const uint32_t expected_scheduler_joins = predecessor_draining ? 0 : 1;
    const uint32_t expected_scheduler_admissions = predecessor_draining ? 3 : 2;
    const uint32_t expected_read_plans = predecessor_draining ? 3 : 2;
    const uint32_t expected_joined_loads = predecessor_draining ? 0 : 1;
    require(event.joined_loads == expected_joined_loads &&
        event.new_reservations == expected_new_reservations &&
        event.scheduler_joins == expected_scheduler_joins &&
        event.scheduler_deferred_successors == uint32_t(predecessor_draining) &&
        event.scheduler_admissions == expected_scheduler_admissions &&
        event.read_plans == expected_read_plans && event.peak_active_read_requests >= 2 &&
        event.first_wait_after_all_enqueue_attempts &&
        event.first_wait_after_all_admissible_submissions &&
        cache.validate_invariants().is_ready() && scheduler.diagnostics().active_requests == 0 &&
        transport.diagnostics().active_read_requests == 0,
        "queued/in-flight join changed issue-ahead or terminal state");
}

void test_batched_queued_and_inflight_joins() {
    run_batched_join_case(true);
    run_batched_join_case(false);
    run_batched_join_case(false, true);
}

void test_cpu_provider_adapter(bool serial_control) {
    temporary_source source;
    llama_file file(source.path, "rb");
    llm_expert_storage storage({ 1, 4, 4, 1024 }, { { 0, &file, 512, source.path, false } });
    for (int32_t expert = 0; expert < 4; ++expert) {
        require(storage.add_bundle({ 0, expert }, spans(uint64_t(expert)*144)).is_ready(),
            "provider storage directory failed");
    }
    require(storage.seal().is_ready(), "provider storage seal failed");

    llm_expert_scheduler_config scheduler_config;
    scheduler_config.layer_count = 1;
    scheduler_config.experts_per_layer = 4;
    scheduler_config.request_capacity = 8;
    scheduler_config.waiters_per_request = 4;
    scheduler_config.max_current_layer_demand_flights = 4;
    scheduler_config.device_count = 1;
    scheduler_config.per_device_request_capacity = 8;
    scheduler_config.per_device_inflight_capacity = 8;
    llm_expert_scheduler scheduler(scheduler_config);

    llm_expert_async_config async_config;
    async_config.requested_queue_depth = 16;
    async_config.effective_hot_capacity = 4;
    async_config.request_capacity = 8;
    async_config.trace_capacity = 64;
    async_config.cold_cache_bytes = 1U << 20;
    async_config.maximum_aligned_read_bytes = 4096;
    async_config.source_file_capacity = 1;
    async_config.force_positional_reads = true;
    llm_expert_async_transport transport(async_config);
    intptr_t handle = -1;
    size_t handle_count = 0;
    require(storage.copy_source_native_handles(&handle, 1, handle_count).is_ready() && handle_count == 1,
        "provider source handle discovery failed");
    require(transport.register_files(&handle, 1) == llm_expert_async_result::ready,
        "provider source registration failed");

    ggml_init_params params = { ggml_tensor_overhead()*32, nullptr, true };
    ggml_context_ptr context(ggml_init(params));
    require(bool(context), "provider context failed");
    const auto prototype = make_bundle(context.get());
    auto * selected = ggml_new_tensor_2d(context.get(), GGML_TYPE_I32, 4, 1);
    llm_expert_selection selection = { 0, 4, 4, 1, selected };

    llm_hot_cache_config config;
    config.capacity = 4;
    config.n_expert_used = 4;
    config.routed_layer_count = 1;
    config.total_expert_keys = 4;
    config.target_buffer_type = ggml_backend_cpu_buffer_type();
    config.target_device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    config.allow_non_cuda_target_for_testing = true;
    config.cold_mode = true;
    config.cpu_cold_only = true;
    config.cold_cache_bytes = 1U << 20;
    config.miss_policy = LLAMA_EXPERT_MISS_POLICY_CPU_FALLBACK;
    config.storage = &storage;
    config.scheduler = &scheduler;
    config.async_transport = &transport;
    config.phase10_serial_issue_for_testing = serial_control;
    config.prefetch_config.supplied = true;
    config.prefetch_config.value.version = LLAMA_EXPERT_PREFETCH_VERSION_1;
    config.prefetch_config.value.struct_size = sizeof(llama_expert_prefetch_config_v1);
    config.prefetch_config.value.policy = LLAMA_EXPERT_PREFETCH_POLICY_OFF;
    config.prefetch_config.value.seed_mode = LLAMA_EXPERT_PREFETCH_SEED_MODE_OFF;
    config.trace_capacity = 64;
    auto provider = llm_create_cold_cache_expert_weight_provider(config);

    llm_expert_graph_binding binding;
    require(provider->bind_graph(context.get(), prototype, selection, binding).is_ready() && binding.bootstrap,
        "provider bootstrap bind failed");
    require(provider->initialize_after_reserve().is_ready(), "provider initialization failed");
    binding = {};
    require(provider->bind_graph(context.get(), prototype, selection, binding).is_ready() &&
        binding.cpu_cold_only, "provider runtime bind failed");
    ggml_backend_buffer_ptr graph_buffer(
        ggml_backend_alloc_ctx_tensors_from_buft(context.get(), ggml_backend_cpu_buffer_type()));
    require(bool(graph_buffer), "provider graph allocation failed");

    const int32_t logical_ids[] = { 2, 0, 2, 1 };
    int32_t execution_ids[4] = { -1, -1, -1, -1 };
    llm_expert_execution_plan plan;
    require(provider->prepare({ binding }, plan).is_ready(), "provider prepare failed");
    require(provider->remap_checkpoint(
        binding, logical_ids, 4, execution_ids).is_ready(), "provider serial remap failed");
    require(execution_ids[0] == execution_ids[2] && execution_ids[0] >= 0 &&
        execution_ids[1] >= 0 && execution_ids[3] >= 0 &&
        execution_ids[0] != execution_ids[1] && execution_ids[0] != execution_ids[3] &&
        execution_ids[1] != execution_ids[3], "provider occurrence reconstruction changed");
    const llm_expert_bundle_descriptor resident = {
        binding.layer, 4, binding.up, binding.gate, binding.gate_up, binding.down,
    };
    for (size_t index = 0; index < 4; ++index) {
        validate_slot(resident, uint32_t(execution_ids[index]), logical_ids[index]);
    }
    plan.reset();
    auto diagnostics = provider->hot_cache_diagnostics();
    require(diagnostics.cold_admissions == 3 && diagnostics.cold_misses == 3 &&
        diagnostics.cold_current_cpu_execution_refs == 0 && diagnostics.cold_current_request_refs == 0 &&
        scheduler.diagnostics().active_requests == 0 &&
        transport.diagnostics().active_read_requests == 0,
        "provider serial resources did not drain");
    const uint64_t read_bytes = storage.diagnostics().read_bytes;

    require(provider->prepare({ binding }, plan).is_ready(), "provider all-hit prepare failed");
    require(provider->remap_checkpoint(
        binding, logical_ids, 4, execution_ids).is_ready(), "provider all-hit remap failed");
    plan.reset();
    diagnostics = provider->hot_cache_diagnostics();
    require(diagnostics.cold_hits == 3 && diagnostics.cold_misses == 3 &&
        storage.diagnostics().read_bytes == read_bytes &&
        diagnostics.cold_current_cpu_execution_refs == 0 && diagnostics.cold_current_request_refs == 0,
        "provider all-hit path changed storage or lifetime state");

    const int32_t invalid_ids[] = { 2, 4, 0, 1 };
    std::fill(std::begin(execution_ids), std::end(execution_ids), -7);
    require(provider->prepare({ binding }, plan).is_ready(), "provider failure prepare failed");
    const auto invalid = provider->remap_checkpoint(binding, invalid_ids, 4, execution_ids);
    require(!invalid.is_ready() && invalid.error == llm_expert_provider_error::invalid_key,
        "provider failure mapping changed");
    require(std::all_of(std::begin(execution_ids), std::end(execution_ids),
        [](int32_t value) { return value == -7; }), "provider published a partial failed mapping");
    plan.reset();
    diagnostics = provider->hot_cache_diagnostics();
    require(diagnostics.cold_current_cpu_execution_refs == 0 && diagnostics.cold_current_request_refs == 0 &&
        diagnostics.cold_current_batch_refs == 0 && scheduler.diagnostics().active_requests == 0 &&
        transport.diagnostics().active_read_requests == 0,
        "provider failure resources did not drain");

    binding = {};
    graph_buffer.reset();
    context.reset();
    require(provider->trim().is_ready(), "provider trim failed");
    require(provider->surrender().is_ready(), "provider surrender failed");
}

} // namespace

int main() {
    test_serial_host_ready_equivalence();
    test_uma_stable_first_semantic_equivalence();
    test_batched_issue_ahead_and_determinism();
    test_batched_failure_cleanup();
    test_batched_queued_and_inflight_joins();
    test_cpu_provider_adapter(true);
    test_cpu_provider_adapter(false);
    std::puts("EXPERT_RESIDENT_DEMAND_OK");
    return 0;
}
