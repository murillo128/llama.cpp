#include "llama-expert-resident-demand.h"

#include "ggml-cpp.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
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

llm_expert_bundle_descriptor make_bundle(ggml_context * ctx) {
    auto projection = [&](const char * name) {
        auto * weight = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, 3, 4);
        ggml_set_name(weight, name);
        return llm_expert_projection_descriptor::from(weight, nullptr, nullptr);
    };
    return { 0, 4, projection("up"), projection("gate"), {}, projection("down") };
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
    require(coordinator.resolve(0, logical_ids, 4, batch).is_ready(), "serial demand resolve failed");
    require(batch.unique_count == 3 && batch.occurrence_count == 4,
        "occurrence plan has wrong width");
    require(batch.entries[0].key.expert == 2 && batch.entries[0].occurrence_count == 2 &&
        batch.entries[1].key.expert == 0 && batch.entries[2].key.expert == 1,
        "stable first-occurrence table changed");
    require(batch.canonical_order[0] == 1 && batch.canonical_order[1] == 2 &&
        batch.canonical_order[2] == 0,
        "canonical key order changed");
    require(batch.occurrence_to_unique[0] == 0 && batch.occurrence_to_unique[1] == 1 &&
        batch.occurrence_to_unique[2] == 0 && batch.occurrence_to_unique[3] == 2,
        "occurrence reconstruction changed");

    std::array<llm_cold_reference, 3> execution_by_unique;
    std::array<llm_cold_reference, 3> execution_refs;
    for (size_t order = 0; order < batch.unique_count; ++order) {
        const uint32_t index = batch.canonical_order[order];
        auto & entry = batch.entries[index];
        require(coordinator.complete_host_scheduler(entry).is_ready(), "host scheduler completion failed");
        require(coordinator.transfer_request_hold_to_cpu_execution(entry).is_ready(),
            "CPU execution hold transfer failed");
        execution_by_unique[index] = entry.reference;
        execution_refs[order] = entry.reference;
        validate_slot(cache.bundle(), entry.reference.slot, entry.key.expert);
    }
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
        first.peak_request_holds == 3 && first.events.size() == 1,
        "serial coordinator counters are inconsistent");
    const auto & event = first.events.front();
    require(event.new_reservations == 3 && event.ready_hits == 0 && event.read_plans == 3 &&
        event.read_operations == 3 && event.host_ready == 3 &&
        !event.first_wait_after_all_enqueue_attempts &&
        !event.first_wait_after_all_admissible_submissions && event.serial_control,
        "serial-control witness changed");
    require(scheduler.diagnostics().active_requests == 0 &&
        transport.diagnostics().active_read_requests == 0,
        "serial demand resources did not drain");

    require(coordinator.resolve(0, logical_ids, 4, batch).is_ready(), "all-hit resolve failed");
    for (size_t order = 0; order < batch.unique_count; ++order) {
        auto & entry = batch.entries[batch.canonical_order[order]];
        require(coordinator.transfer_request_hold_to_cpu_execution(entry).is_ready(),
            "all-hit CPU execution hold transfer failed");
        execution_refs[order] = entry.reference;
    }
    require(cache.release_many(execution_refs.data(), batch.unique_count,
        llm_cold_reference_kind::cpu_execution).is_ready(), "all-hit CPU execution release failed");
    require(cache.policy_request_end(true, false).is_ready(), "all-hit policy close failed");
    const auto second = coordinator.diagnostics();
    require(second.events.back().ready_hits == 3 && second.events.back().new_reservations == 0 &&
        second.events.back().read_plans == 0 && storage.diagnostics().read_bytes == 432,
        "all-hit path performed storage or scheduler work");
    require(cache.validate_invariants().is_ready(), "cold cache invariant failed");
}

void test_cpu_provider_serial_adapter() {
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
    test_cpu_provider_serial_adapter();
    std::puts("EXPERT_RESIDENT_DEMAND_SERIAL_OK");
    return 0;
}
