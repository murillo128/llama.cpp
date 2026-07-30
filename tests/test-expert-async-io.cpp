#include "llama-expert-async-io.h"

#include "ggml.h"

#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

template<class F> void expect_invalid(F fn) {
    bool rejected = false;
    try { fn(); } catch (const std::invalid_argument &) { rejected = true; }
    GGML_ASSERT(rejected);
}

llm_expert_async_config config(uint32_t queue_depth = 0, uint32_t operation_generation = 0) {
    return {
        queue_depth,
        12,
        16,
        4,
        256U*1024U*1024U,
        0,
        2U*1024U*1024U,
        true,
        operation_generation,
    };
}

void test_configuration() {
    expect_invalid([] { llm_expert_async_transport transport({}); });
    expect_invalid([] { llm_expert_async_transport transport(config(7)); });
    expect_invalid([] { llm_expert_async_transport transport(config(4097)); });

    llm_expert_async_transport transport(config());
    const auto diagnostics = transport.diagnostics();
    GGML_ASSERT(diagnostics.requested_sq_entries == 64);
    GGML_ASSERT(diagnostics.requested_cq_entries == 128);
    GGML_ASSERT(diagnostics.operation_capacity == 128);
    GGML_ASSERT(diagnostics.trace_capacity == 4);
    GGML_ASSERT(diagnostics.staging_ceiling_bytes == 4U*1024U*1024U);
    GGML_ASSERT(diagnostics.administration_bytes > 0);
#if defined(__linux__)
    GGML_ASSERT(diagnostics.linux_uapi);
#else
    GGML_ASSERT(!diagnostics.linux_uapi);
#endif

    llm_expert_async_config bounded = config(8);
    bounded.requested_staging_bytes = 1024;
    llm_expert_async_transport explicit_transport(bounded);
    GGML_ASSERT(explicit_transport.diagnostics().staging_ceiling_bytes == 1024);
}

void test_ring_layout_validation() {
    llm_expert_async_ring_layout layout = {
        8, 16,
        0, 4, 8, 12, 16, 20, 64,
        0, 4, 8, 12, 16, 64,
    };
    llm_expert_async_mapping_sizes sizes;
    GGML_ASSERT(llm_expert_async_transport::validate_ring_layout(layout, 4096, sizes));
    GGML_ASSERT(sizes.sq_ring_bytes == 96);
    GGML_ASSERT(sizes.cq_ring_bytes == 320);
    GGML_ASSERT(sizes.sqes_bytes == 512);

    layout.sq_entries = 7;
    GGML_ASSERT(!llm_expert_async_transport::validate_ring_layout(layout, 4096, sizes));
    layout.sq_entries = 8;
    layout.cqes = std::numeric_limits<uint32_t>::max();
    GGML_ASSERT(!llm_expert_async_transport::validate_ring_layout(layout, 4096, sizes));
    layout.cqes = 64;
    GGML_ASSERT(!llm_expert_async_transport::validate_ring_layout(layout, 3000, sizes));

    const auto probe = llm_expert_async_transport::probe_ring_for_testing(8);
    if (probe.supported) {
        GGML_ASSERT(probe.layout_valid);
        GGML_ASSERT(probe.sq_entries >= 8 && probe.cq_entries >= probe.sq_entries);
    } else {
        GGML_ASSERT(probe.native_error != 0);
    }
    GGML_ASSERT(llm_expert_async_transport::probe_ring_for_testing(7).native_error == EINVAL);
}

void test_user_data_and_fake_completion() {
    uint32_t slot = 0;
    uint32_t generation = 0;
    const uint64_t encoded = llm_expert_async_transport::encode_user_data(17, 23);
    GGML_ASSERT(encoded != 0);
    GGML_ASSERT(llm_expert_async_transport::decode_user_data(encoded, slot, generation));
    GGML_ASSERT(slot == 17 && generation == 23);
    GGML_ASSERT(!llm_expert_async_transport::decode_user_data(0, slot, generation));

    llm_expert_async_transport transport(config(8));
    const llm_expert_async_operation_identity expected = {
        1,
        { 2, 9 },
        3,
        { 1, 4 },
        llm_expert_readiness::device_ready,
        llm_expert_priority::demand_current_layer,
    };
    const auto operation = transport.reserve_operation(expected);
    GGML_ASSERT(operation.result == llm_expert_async_result::ready && operation.user_data != 0);
    llm_expert_async_operation_identity actual;
    GGML_ASSERT(transport.consume_completion_for_testing(operation.user_data, actual) == llm_expert_async_result::ready);
    GGML_ASSERT(actual.transport_epoch == expected.transport_epoch);
    GGML_ASSERT(actual.request.slot == expected.request.slot);
    GGML_ASSERT(actual.request.generation == expected.request.generation);
    GGML_ASSERT(actual.request_operation_index == expected.request_operation_index);
    GGML_ASSERT(actual.key.layer == expected.key.layer && actual.key.expert == expected.key.expert);
    GGML_ASSERT(transport.consume_completion_for_testing(operation.user_data, actual) ==
        llm_expert_async_result::stale_generation);
    const auto diagnostics = transport.diagnostics();
    GGML_ASSERT(diagnostics.operations_reserved == 1);
    GGML_ASSERT(diagnostics.completions_consumed == 1);
    GGML_ASSERT(diagnostics.stale_completions == 1);
    GGML_ASSERT(diagnostics.active_operations == 0);
}

void test_bounded_trace_and_shutdown() {
    llm_expert_async_transport transport(config(8));
    for (int index = 0; index < 6; ++index) {
        transport.record_trace_for_testing();
    }
    auto diagnostics = transport.diagnostics();
    GGML_ASSERT(diagnostics.trace_records == 4);
    GGML_ASSERT(diagnostics.trace_records_dropped == 2);
    GGML_ASSERT(transport.shutdown());
    diagnostics = transport.diagnostics();
    GGML_ASSERT(diagnostics.admission_closed);
    GGML_ASSERT(diagnostics.transport_epoch == 2);

    const llm_expert_async_operation_identity identity = {
        2, { 0, 1 }, 0, { 0, 0 }, llm_expert_readiness::host_ready,
        llm_expert_priority::demand_current_layer,
    };
    GGML_ASSERT(transport.reserve_operation(identity).result == llm_expert_async_result::closed);
}

void test_generation_exhaustion() {
    llm_expert_async_transport transport(config(8, std::numeric_limits<uint32_t>::max()));
    const llm_expert_async_operation_identity identity = {
        1, { 0, 1 }, 0, { 0, 0 }, llm_expert_readiness::host_ready,
        llm_expert_priority::demand_current_layer,
    };
    GGML_ASSERT(transport.reserve_operation(identity).result == llm_expert_async_result::generation_exhausted);
}

void test_concurrent_bounded_access() {
    llm_expert_async_transport transport(config(8));
    std::vector<std::thread> threads;
    for (uint32_t index = 0; index < 8; ++index) {
        threads.emplace_back([&transport, index] {
            const llm_expert_async_operation_identity expected = {
                1, { index, uint64_t(index) + 1 }, index, { 0, int32_t(index) },
                llm_expert_readiness::host_ready, llm_expert_priority::demand_current_layer,
            };
            const auto operation = transport.reserve_operation(expected);
            GGML_ASSERT(operation.result == llm_expert_async_result::ready);
            llm_expert_async_operation_identity actual;
            GGML_ASSERT(transport.consume_completion_for_testing(operation.user_data, actual) ==
                llm_expert_async_result::ready);
            GGML_ASSERT(actual.request.slot == index);
        });
    }
    for (auto & thread : threads) {
        thread.join();
    }
    const auto diagnostics = transport.diagnostics();
    GGML_ASSERT(diagnostics.operations_reserved == 8);
    GGML_ASSERT(diagnostics.completions_consumed == 8);
    GGML_ASSERT(diagnostics.active_operations == 0);
}

} // namespace

int main() {
    test_configuration();
    test_ring_layout_validation();
    test_user_data_and_fake_completion();
    test_bounded_trace_and_shutdown();
    test_generation_exhaustion();
    test_concurrent_bounded_access();
    return 0;
}
