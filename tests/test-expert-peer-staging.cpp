#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <array>
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

struct peer_fixture {
    std::array<ggml_backend_dev_t, 2> devices = {};
    std::array<ggml_backend_ptr, 2> backends;
    std::array<ggml_context_ptr, 2> contexts;
    std::array<ggml_backend_buffer_ptr, 2> buffers;
    std::array<std::array<ggml_tensor *, transfer_count>, 2> sources = {};
    std::array<ggml_tensor *, 2> destinations = {};
    diagnostics_fn diagnostics = nullptr;

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
            GGML_ASSERT(configure != nullptr && edge != nullptr && query != nullptr);
            GGML_ASSERT(configure(backends[device].get(), 0, staging_bytes_per_device,
                device, devices.size()) == 0);
            diagnostics = query;
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

    std::array<uint64_t, 17> query(uint32_t source, uint32_t destination) {
        std::array<uint64_t, 17> values = {};
        GGML_ASSERT(diagnostics(backends[destination].get(), source,
            values.data(), values.size()) == 0);
        return values;
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
    peer_fixture fixture;
    if (!fixture.available()) {
        return 0;
    }

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
    }

    // Synchronize in the reverse of the enqueue order. The final contents are
    // still selected by stream order, never by relative device completion.
    ggml_backend_synchronize(fixture.backends[1].get());
    ggml_backend_synchronize(fixture.backends[0].get());
    const auto forward_done = fixture.query(0, 1);
    const auto reverse_done = fixture.query(1, 0);
    GGML_ASSERT(forward_done[10] == transfer_count && reverse_done[10] == transfer_count);
    assert_pattern(fixture.destinations[1], uint8_t(transfer_count));
    assert_pattern(fixture.destinations[0], uint8_t(0x10U + transfer_count));

    // Leave work live on both directed edges. Backend teardown must drain its
    // D2H/H2D events before the still-live tensor buffers are released.
    ggml_backend_tensor_copy_async(fixture.backends[0].get(), fixture.backends[1].get(),
        fixture.sources[0][0], fixture.destinations[1]);
    ggml_backend_tensor_copy_async(fixture.backends[1].get(), fixture.backends[0].get(),
        fixture.sources[1][0], fixture.destinations[0]);
    fixture.backends[1].reset();
    fixture.backends[0].reset();
    return 0;
}
