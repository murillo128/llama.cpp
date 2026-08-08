#include <cuda_runtime_api.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <sys/statvfs.h>
#include <sys/utsname.h>
#include <vector>

using json = nlohmann::ordered_json;

namespace {

constexpr size_t transfer_bytes = 64ULL << 20;
constexpr int transfer_repetitions = 8;

std::string read_text(const std::filesystem::path & path) {
    std::ifstream input(path);
    std::string value;
    std::getline(input, value);
    return value;
}

std::string uuid_string(const cudaUUID_t & value) {
    const auto * b = reinterpret_cast<const unsigned char *>(value.bytes);
    char text[64] = {};
    std::snprintf(text, sizeof(text),
        "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
        b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]);
    return text;
}

struct device {
    int ordinal = -1;
    std::string pci_bdf;
    cudaDeviceProp properties = {};
    void * allocation = nullptr;
};

double timed_device_copy(int ordinal, void * device_memory, void * host_memory, cudaMemcpyKind kind) {
    cudaSetDevice(ordinal);
    cudaStream_t stream = nullptr;
    cudaEvent_t begin = nullptr;
    cudaEvent_t end = nullptr;
    if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess ||
        cudaEventCreate(&begin) != cudaSuccess || cudaEventCreate(&end) != cudaSuccess) {
        throw std::runtime_error("CUDA event/stream allocation failed");
    }
    const void * source = kind == cudaMemcpyHostToDevice ? host_memory : device_memory;
    void * destination = kind == cudaMemcpyHostToDevice ? device_memory : host_memory;
    cudaEventRecord(begin, stream);
    for (int repetition = 0; repetition < transfer_repetitions; ++repetition) {
        if (cudaMemcpyAsync(destination, source, transfer_bytes, kind, stream) != cudaSuccess) {
            throw std::runtime_error("CUDA host/device bandwidth copy failed");
        }
    }
    cudaEventRecord(end, stream);
    cudaEventSynchronize(end);
    float elapsed_ms = 0;
    cudaEventElapsedTime(&elapsed_ms, begin, end);
    cudaEventDestroy(end);
    cudaEventDestroy(begin);
    cudaStreamDestroy(stream);
    return double(transfer_bytes)*transfer_repetitions/(elapsed_ms/1000.0)/1.0e9;
}

double timed_peer_copy(const device & source, const device & destination) {
    cudaSetDevice(destination.ordinal);
    cudaStream_t stream = nullptr;
    cudaEvent_t begin = nullptr;
    cudaEvent_t end = nullptr;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    cudaEventCreate(&begin);
    cudaEventCreate(&end);
    cudaEventRecord(begin, stream);
    for (int repetition = 0; repetition < transfer_repetitions; ++repetition) {
        if (cudaMemcpyPeerAsync(destination.allocation, destination.ordinal,
                source.allocation, source.ordinal, transfer_bytes, stream) != cudaSuccess) {
            throw std::runtime_error("CUDA peer bandwidth copy failed");
        }
    }
    cudaEventRecord(end, stream);
    cudaEventSynchronize(end);
    float elapsed_ms = 0;
    cudaEventElapsedTime(&elapsed_ms, begin, end);
    cudaEventDestroy(end);
    cudaEventDestroy(begin);
    cudaStreamDestroy(stream);
    return double(transfer_bytes)*transfer_repetitions/(elapsed_ms/1000.0)/1.0e9;
}

double timed_host_staged_copy(const device & source, const device & destination, void * host_memory) {
    const auto begin = std::chrono::steady_clock::now();
    for (int repetition = 0; repetition < transfer_repetitions; ++repetition) {
        cudaSetDevice(source.ordinal);
        if (cudaMemcpy(host_memory, source.allocation, transfer_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
            throw std::runtime_error("CUDA staged D2H copy failed");
        }
        cudaSetDevice(destination.ordinal);
        if (cudaMemcpy(destination.allocation, host_memory, transfer_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            throw std::runtime_error("CUDA staged H2D copy failed");
        }
    }
    const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
    return double(transfer_bytes)*transfer_repetitions/elapsed/1.0e9;
}

json backing_store(const std::string & requested_path) {
    const auto path = std::filesystem::canonical(requested_path);
    struct stat metadata {};
    struct statvfs filesystem {};
    if (stat(path.c_str(), &metadata) != 0 || statvfs(path.c_str(), &filesystem) != 0) {
        throw std::runtime_error("backing path stat failed");
    }
    std::string selected_mount;
    std::string selected_fs;
    std::string selected_source;
    std::ifstream mounts("/proc/self/mountinfo");
    for (std::string line; std::getline(mounts, line);) {
        std::istringstream fields(line);
        std::vector<std::string> parts;
        for (std::string part; fields >> part;) parts.push_back(part);
        auto separator = std::find(parts.begin(), parts.end(), "-");
        if (parts.size() < 6 || separator == parts.end() || separator + 2 >= parts.end()) continue;
        const std::string & mount = parts[4];
        const std::string pathname = path.string();
        if (pathname.compare(0, mount.size(), mount) == 0 && mount.size() >= selected_mount.size()) {
            selected_mount = mount;
            selected_fs = *(separator + 1);
            selected_source = *(separator + 2);
        }
    }
    return {
        {"path", path.string()}, {"mountpoint", selected_mount}, {"filesystem", selected_fs},
        {"source", selected_source}, {"stat_device_major", major(metadata.st_dev)},
        {"stat_device_minor", minor(metadata.st_dev)},
        {"capacity_bytes", uint64_t(filesystem.f_blocks)*filesystem.f_frsize},
        {"available_bytes", uint64_t(filesystem.f_bavail)*filesystem.f_frsize},
    };
}

} // namespace

int main(int argc, char ** argv) {
    std::string output_path;
    std::string backing_path;
    for (int index = 1; index < argc; ++index) {
        if (index + 1 >= argc) return 2;
        const std::string option = argv[index];
        const char * value = argv[++index];
        if (option == "--output") output_path = value;
        else if (option == "--backing-path") backing_path = value;
        else return 2;
    }
    if (output_path.empty() || backing_path.empty()) return 2;

    try {
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count != 2) {
            throw std::runtime_error("Phase 13 requires exactly two visible CUDA devices");
        }
        std::vector<device> devices;
        for (int ordinal = 0; ordinal < count; ++ordinal) {
            device value;
            value.ordinal = ordinal;
            char pci[32] = {};
            if (cudaGetDeviceProperties(&value.properties, ordinal) != cudaSuccess ||
                cudaDeviceGetPCIBusId(pci, sizeof(pci), ordinal) != cudaSuccess) {
                throw std::runtime_error("CUDA device identity query failed");
            }
            value.pci_bdf = pci;
            std::transform(value.pci_bdf.begin(), value.pci_bdf.end(), value.pci_bdf.begin(),
                [](unsigned char character) { return char(std::tolower(character)); });
            devices.push_back(value);
        }
        std::sort(devices.begin(), devices.end(), [](const device & lhs, const device & rhs) {
            return lhs.pci_bdf < rhs.pci_bdf;
        });

        void * host_memory = nullptr;
        if (cudaHostAlloc(&host_memory, transfer_bytes, cudaHostAllocPortable) != cudaSuccess) {
            throw std::runtime_error("bounded pinned topology buffer allocation failed");
        }
        for (auto & value : devices) {
            cudaSetDevice(value.ordinal);
            if (cudaMalloc(&value.allocation, transfer_bytes) != cudaSuccess) {
                throw std::runtime_error("bounded device topology buffer allocation failed");
            }
            cudaMemset(value.allocation, value.ordinal + 1, transfer_bytes);
        }

        json gpu_json = json::array();
        for (size_t id = 0; id < devices.size(); ++id) {
            const auto & value = devices[id];
            const auto sysfs = std::filesystem::path("/sys/bus/pci/devices")/value.pci_bdf;
            int numa_node = -1;
            const std::string numa_text = read_text(sysfs/"numa_node");
            if (!numa_text.empty()) numa_node = std::stoi(numa_text);
            std::error_code ec;
            const auto upstream = std::filesystem::canonical(sysfs, ec);
            gpu_json.push_back({
                {"device_id", id}, {"cuda_ordinal", value.ordinal},
                {"uuid", uuid_string(value.properties.uuid)}, {"pci_bdf", value.pci_bdf},
                {"name", value.properties.name}, {"numa_node", numa_node},
                {"pci_path", ec ? "" : upstream.string()},
                {"current_link_speed", read_text(sysfs/"current_link_speed")},
                {"current_link_width", read_text(sysfs/"current_link_width")},
                {"max_link_speed", read_text(sysfs/"max_link_speed")},
                {"max_link_width", read_text(sysfs/"max_link_width")},
                {"h2d_gbps", timed_device_copy(value.ordinal, value.allocation, host_memory, cudaMemcpyHostToDevice)},
                {"d2h_gbps", timed_device_copy(value.ordinal, value.allocation, host_memory, cudaMemcpyDeviceToHost)},
            });
        }

        json peer_matrix = json::array();
        json peer_bandwidth = json::array();
        json staged_bandwidth = json::array();
        for (size_t src = 0; src < devices.size(); ++src) {
            json row = json::array();
            for (size_t dst = 0; dst < devices.size(); ++dst) {
                int capable = src == dst ? 1 : 0;
                int enable_status = 0;
                if (src != dst) {
                    cudaDeviceCanAccessPeer(&capable, devices[src].ordinal, devices[dst].ordinal);
                    if (capable) {
                        cudaSetDevice(devices[src].ordinal);
                        const cudaError_t status = cudaDeviceEnablePeerAccess(devices[dst].ordinal, 0);
                        enable_status = status == cudaSuccess || status == cudaErrorPeerAccessAlreadyEnabled ? 1 : -int(status);
                        if (status == cudaErrorPeerAccessAlreadyEnabled) (void) cudaGetLastError();
                    }
                    staged_bandwidth.push_back({
                        {"source_device_id", src}, {"destination_device_id", dst},
                        {"gbps", timed_host_staged_copy(devices[src], devices[dst], host_memory)},
                    });
                    if (capable && enable_status == 1) {
                        peer_bandwidth.push_back({
                            {"source_device_id", src}, {"destination_device_id", dst},
                            {"gbps", timed_peer_copy(devices[src], devices[dst])},
                        });
                    }
                }
                row.push_back({{"capable", capable != 0}, {"enable_result", enable_status}});
            }
            peer_matrix.push_back(std::move(row));
        }

        int driver_version = 0;
        int runtime_version = 0;
        cudaDriverGetVersion(&driver_version);
        cudaRuntimeGetVersion(&runtime_version);
        struct utsname system {};
        uname(&system);
        json result = {
            {"schema", "phase13-topology-v1"}, {"status", "pass"},
            {"host", {{"nodename", system.nodename}, {"kernel", system.release},
                {"online_numa_nodes", read_text("/sys/devices/system/node/online")}}},
            {"cuda", {{"driver_version", driver_version}, {"runtime_version", runtime_version}}},
            {"device_order", "ascending_pci_domain_bus_device_function"},
            {"gpus", gpu_json}, {"directed_peer_matrix", peer_matrix},
            {"peer_bandwidth", peer_bandwidth}, {"host_staged_bandwidth", staged_bandwidth},
            {"transfer_bytes", transfer_bytes}, {"transfer_repetitions", transfer_repetitions},
            {"backing_store", backing_store(backing_path)},
        };
        std::ofstream output(output_path, std::ios::binary | std::ios::trunc);
        output << result.dump(2) << '\n';
        if (!output) throw std::runtime_error("topology output write failed");
        for (auto & value : devices) {
            cudaSetDevice(value.ordinal);
            cudaFree(value.allocation);
        }
        cudaFreeHost(host_memory);
        std::cout << "PHASE13_TOPOLOGY status=pass output=" << output_path << '\n';
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "phase13-topology-probe: " << error.what() << '\n';
        return 3;
    }
}
