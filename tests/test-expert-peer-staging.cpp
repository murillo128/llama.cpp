#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <array>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

constexpr size_t transfer_bytes = 12U*1024U*1024U;
constexpr size_t staging_bytes_per_device = 32U*1024U*1024U;
constexpr uint32_t transfer_count = 4;

using configure_transport_fn = int (*)(ggml_backend_t, int, size_t, uint32_t, uint32_t);
using configure_edge_fn = int (*)(ggml_backend_t, ggml_backend_t);
using diagnostics_fn = int (*)(ggml_backend_t, uint32_t, uint64_t *, size_t);
using cancel_edge_fn = int (*)(ggml_backend_t, uint32_t);
using peer_test_gate_fn = int (*)(ggml_backend_t, uint32_t, uint32_t);
using slot_generation_fn = int (*)(ggml_backend_t, uint32_t, uint32_t, uint64_t *);
using complete_slot_fn = int (*)(ggml_backend_t, uint32_t, uint32_t, uint64_t);

constexpr uint32_t test_phase_d2h = 1;
constexpr uint32_t test_phase_h2d = 2;

struct peer_fixture {
    std::array<ggml_backend_dev_t, 2> devices = {};
    std::array<ggml_backend_ptr, 2> backends;
    std::array<ggml_context_ptr, 2> contexts;
    std::array<ggml_backend_buffer_ptr, 2> buffers;
    std::array<std::array<ggml_tensor *, transfer_count>, 2> sources = {};
    std::array<ggml_tensor *, 2> destinations = {};
    diagnostics_fn diagnostics = nullptr;
    cancel_edge_fn cancel_edge = nullptr;
    peer_test_gate_fn set_test_gate = nullptr;
    peer_test_gate_fn wait_test_gate = nullptr;
    slot_generation_fn slot_generation = nullptr;
    complete_slot_fn complete_slot = nullptr;

    peer_fixture() {
        ggml_backend_load_all();
        uint32_t found = 0;
        for (size_t index = 0; index < ggml_backend_dev_count() && found < devices.size(); ++index) {
            ggml_backend_dev_t device = ggml_backend_dev_get(index);
            if (ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_GPU ||
                std::strncmp(ggml_backend_dev_name(device), "CUDA", 4) != 0) {
                continue;
            }
            devices[found++] = device;
        }
        if (found != devices.size()) {
            return;
        }

        for (uint32_t device = 0; device < devices.size(); ++device) {
            backends[device].reset(ggml_backend_dev_init(devices[device], nullptr));
            GGML_ASSERT(backends[device]);
            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(devices[device]);
            auto configure = reinterpret_cast<configure_transport_fn>(ggml_backend_reg_get_proc_address(
                reg, "ggml_backend_cuda_expert_configure_peer_transport"));
            auto edge = reinterpret_cast<configure_edge_fn>(ggml_backend_reg_get_proc_address(
                reg, "ggml_backend_cuda_expert_configure_peer_edge"));
            auto query = reinterpret_cast<diagnostics_fn>(ggml_backend_reg_get_proc_address(
                reg, "ggml_backend_cuda_expert_peer_diagnostics"));
            auto cancel = reinterpret_cast<cancel_edge_fn>(ggml_backend_reg_get_proc_address(
                reg, "ggml_backend_cuda_expert_cancel_peer_edge"));
            auto set_gate = reinterpret_cast<peer_test_gate_fn>(ggml_backend_reg_get_proc_address(
                reg, "ggml_backend_cuda_expert_set_peer_test_gate"));
            auto wait_gate = reinterpret_cast<peer_test_gate_fn>(ggml_backend_reg_get_proc_address(
                reg, "ggml_backend_cuda_expert_wait_peer_test_gate"));
            auto generation = reinterpret_cast<slot_generation_fn>(ggml_backend_reg_get_proc_address(
                reg, "ggml_backend_cuda_expert_peer_slot_generation_for_testing"));
            auto complete = reinterpret_cast<complete_slot_fn>(ggml_backend_reg_get_proc_address(
                reg, "ggml_backend_cuda_expert_complete_peer_slot_for_testing"));
            GGML_ASSERT(configure != nullptr && edge != nullptr && query != nullptr &&
                cancel != nullptr && set_gate != nullptr && wait_gate != nullptr &&
                generation != nullptr && complete != nullptr);
            GGML_ASSERT(configure(backends[device].get(), 0, staging_bytes_per_device,
                device, devices.size()) == 0);
            diagnostics = query;
            cancel_edge = cancel;
            set_test_gate = set_gate;
            wait_test_gate = wait_gate;
            slot_generation = generation;
            complete_slot = complete;
        }
        GGML_ASSERT(diagnostics != nullptr);
        GGML_ASSERT(reinterpret_cast<configure_edge_fn>(ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(devices[0]),
            "ggml_backend_cuda_expert_configure_peer_edge"))(
                backends[0].get(), backends[1].get()) == 0);
        GGML_ASSERT(reinterpret_cast<configure_edge_fn>(ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(devices[1]),
            "ggml_backend_cuda_expert_configure_peer_edge"))(
                backends[1].get(), backends[0].get()) == 0);

        for (uint32_t device = 0; device < devices.size(); ++device) {
            const size_t tensor_count = transfer_count + 1;
            ggml_init_params params = { ggml_tensor_overhead()*tensor_count, nullptr, true };
            contexts[device].reset(ggml_init(params));
            GGML_ASSERT(contexts[device]);
            for (uint32_t index = 0; index < transfer_count; ++index) {
                sources[device][index] = ggml_new_tensor_1d(
                    contexts[device].get(), GGML_TYPE_I8, transfer_bytes);
            }
            destinations[device] = ggml_new_tensor_1d(
                contexts[device].get(), GGML_TYPE_I8, transfer_bytes);
            buffers[device].reset(ggml_backend_alloc_ctx_tensors_from_buft(
                contexts[device].get(), ggml_backend_dev_buffer_type(devices[device])));
            GGML_ASSERT(buffers[device]);
            for (uint32_t index = 0; index < transfer_count; ++index) {
                ggml_backend_tensor_memset(sources[device][index],
                    uint8_t(0x10U*device + index + 1), 0, transfer_bytes);
            }
            ggml_backend_tensor_memset(destinations[device], 0, 0, transfer_bytes);
        }
    }

    bool available() const {
        return backends[0] != nullptr && backends[1] != nullptr;
    }

    std::array<uint64_t, 24> query(uint32_t source, uint32_t destination) {
        std::array<uint64_t, 24> values = {};
        GGML_ASSERT(diagnostics(backends[destination].get(), source,
            values.data(), values.size()) == 0);
        return values;
    }

    void arm(uint32_t source, uint32_t destination, uint32_t phase) {
        GGML_ASSERT(set_test_gate(backends[destination].get(), source, phase) == 0);
    }

    void wait_until_held(uint32_t source, uint32_t destination, uint32_t phase) {
        GGML_ASSERT(wait_test_gate(backends[destination].get(), source, phase) == 0);
    }

    void cancel(uint32_t source, uint32_t destination) {
        GGML_ASSERT(cancel_edge(backends[destination].get(), source) == 0);
    }

    uint64_t generation(uint32_t source, uint32_t destination, uint32_t slot) {
        uint64_t value = 0;
        GGML_ASSERT(slot_generation(backends[destination].get(), source, slot, &value) == 0);
        return value;
    }

    void inject_stale_completion(
            uint32_t source, uint32_t destination, uint32_t slot, uint64_t expected) {
        GGML_ASSERT(complete_slot(backends[destination].get(), source, slot, expected) == 1);
    }
};

void assert_pattern(const ggml_tensor * tensor, uint8_t expected) {
    std::vector<uint8_t> bytes(transfer_bytes);
    ggml_backend_tensor_get(tensor, bytes.data(), 0, bytes.size());
    for (size_t offset : { size_t(0), bytes.size()/3, 2*bytes.size()/3, bytes.size() - 1 }) {
        GGML_ASSERT(bytes[offset] == expected);
    }
}

} // namespace

int main() {
    {
        peer_fixture availability;
        if (!availability.available()) {
            return 0;
        }
    }

    {
        peer_fixture fixture;
        // Alternate both directed edges without a host synchronization. Four
        // copies per edge force each fixed two-slot ring to reuse both slots.
        for (uint32_t index = 0; index < transfer_count; ++index) {
            ggml_backend_tensor_copy_async(fixture.backends[0].get(), fixture.backends[1].get(),
                fixture.sources[0][index], fixture.destinations[1]);
            ggml_backend_tensor_copy_async(fixture.backends[1].get(), fixture.backends[0].get(),
                fixture.sources[1][index], fixture.destinations[0]);
        }
        const auto forward_live = fixture.query(0, 1);
        const auto reverse_live = fixture.query(1, 0);
        for (const auto & values : { forward_live, reverse_live }) {
            GGML_ASSERT(values[2] == transfer_bytes*transfer_count);
            GGML_ASSERT(values[3] == transfer_count && values[4] == 2);
            GGML_ASSERT(values[5] == 2 && values[6] > 0);
            GGML_ASSERT(values[8] == 0 && values[9] == transfer_count && values[11] == 0);
            GGML_ASSERT(values[17] == 0 && values[18] == 0 && values[22] == 0);
        }

        // Synchronize in the reverse of the enqueue order. The final contents
        // are still selected by stream order, never device completion order.
        ggml_backend_synchronize(fixture.backends[1].get());
        ggml_backend_synchronize(fixture.backends[0].get());
        const auto forward_done = fixture.query(0, 1);
        const auto reverse_done = fixture.query(1, 0);
        GGML_ASSERT(forward_done[10] == transfer_count && reverse_done[10] == transfer_count);
        GGML_ASSERT(forward_done[23] == 0 && reverse_done[23] == 0);
        assert_pattern(fixture.destinations[1], uint8_t(transfer_count));
        assert_pattern(fixture.destinations[0], uint8_t(0x10U + transfer_count));

        // Leave work live on both directed edges. Backend teardown must drain
        // D2H/H2D before the still-live tensor buffers are released.
        ggml_backend_tensor_copy_async(fixture.backends[0].get(), fixture.backends[1].get(),
            fixture.sources[0][0], fixture.destinations[1]);
        ggml_backend_tensor_copy_async(fixture.backends[1].get(), fixture.backends[0].get(),
            fixture.sources[1][0], fixture.destinations[0]);
        fixture.backends[1].reset();
        fixture.backends[0].reset();
    }

    {
        peer_fixture fixture;
        ggml_backend_tensor_copy_async(fixture.backends[0].get(), fixture.backends[1].get(),
            fixture.sources[0][0], fixture.destinations[1]);
        const uint64_t generation = fixture.generation(0, 1, 0);
        GGML_ASSERT(generation > 0);
        fixture.inject_stale_completion(0, 1, 0, generation - 1);
        const auto stale_live = fixture.query(0, 1);
        GGML_ASSERT(stale_live[17] == 1);
        GGML_ASSERT(stale_live[9] == 1 && stale_live[10] <= 1);
        ggml_backend_synchronize(fixture.backends[1].get());
        const auto stale_done = fixture.query(0, 1);
        GGML_ASSERT(stale_done[17] == 1 && stale_done[10] == 1 && stale_done[23] == 0);
        assert_pattern(fixture.destinations[1], uint8_t(1));
    }

    for (const uint32_t phase : { test_phase_d2h, test_phase_h2d }) {
        peer_fixture fixture;
        fixture.arm(0, 1, phase);
        ggml_backend_tensor_copy_async(fixture.backends[0].get(), fixture.backends[1].get(),
            fixture.sources[0][0], fixture.destinations[1]);
        fixture.wait_until_held(0, 1, phase);
        fixture.cancel(0, 1);
        const auto cancelled = fixture.query(0, 1);
        GGML_ASSERT(cancelled[9] == 1 && cancelled[10] == 1);
        GGML_ASSERT(cancelled[17] == 0 && cancelled[18] == 1);
        GGML_ASSERT(cancelled[19] == (phase == test_phase_d2h ? 1U : 0U));
        GGML_ASSERT(cancelled[20] == (phase == test_phase_h2d ? 1U : 0U));
        GGML_ASSERT(cancelled[21] == 1 && cancelled[22] == 0 && cancelled[23] == 0);
    }
    std::printf(
        "{\"schema_version\":\"phase13-peer-staging-lifetime-v1\","
        "\"status\":\"pass\",\"stale_generation_discarded\":1,"
        "\"cancellation_during_d2h\":1,\"cancellation_during_h2d\":1,"
        "\"cancelled_generations_drained\":2,\"live_slots_after_cancel\":0}\n");
    return 0;
}
