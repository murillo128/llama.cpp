#include "ggml-cpp.h"
#include "ggml-cuda.h"

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <linux/io_uring.h>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <vector>

namespace {

void require(bool condition, const char * message) { if (!condition) throw std::runtime_error(message); }
uint64_t checksum(const uint8_t * bytes, size_t size) {
    uint64_t result = 0;
    for (size_t i = 0; i < size; ++i) result += bytes[i];
    return result;
}
void fill(uint8_t * bytes, size_t size, uint8_t seed) {
    for (size_t i = 0; i < size; ++i) bytes[i] = uint8_t(seed + uint8_t(i*131U));
}
std::string read_text(const char * path) {
    FILE * file = fopen(path, "rb");
    if (file == nullptr) return "unavailable";
    char value[256] = {};
    const size_t size = fread(value, 1, sizeof(value) - 1, file);
    fclose(file);
    while (size != 0 && (value[strlen(value) - 1] == '\n' || value[strlen(value) - 1] == '\r')) value[strlen(value) - 1] = 0;
    return value;
}

} // namespace

int main() try {
    constexpr size_t bytes = 4*1024*1024;
    constexpr size_t page = 4096;
    ggml_backend_cuda_uma_capabilities capabilities = {};
    capabilities.struct_size = sizeof(capabilities);
    require(ggml_backend_cuda_uma_get_capabilities(0, &capabilities) == 0, "capability query failed");
    require(capabilities.device_count == 1 && capabilities.compute_capability_major == 12 &&
        capabilities.pageable_memory_access != 0 && capabilities.pageable_memory_access_uses_host_page_tables != 0,
        "GB10 coherent pageable access unavailable");
    ggml_backend_buffer_type_t buft = ggml_backend_cuda_uma_buffer_type(0);
    ggml_backend_buffer_ptr buffer(ggml_backend_buft_alloc_buffer(buft, bytes));
    require(bool(buffer), "anonymous UMA allocation failed");
    auto * base = static_cast<uint8_t *>(ggml_backend_buffer_get_base(buffer.get()));
    require(base != nullptr && uintptr_t(base) % page == 0, "allocation alignment failed");
    const auto handoff_start = std::chrono::steady_clock::now();
    fill(base, bytes, 17);
    uint64_t gpu_checksum = 0;
    require(ggml_backend_cuda_uma_checksum(buffer.get(), 0, bytes, &gpu_checksum) == 0 &&
        gpu_checksum == checksum(base, bytes), "CPU-write/GPU-checksum failed");
    const auto handoff_end = std::chrono::steady_clock::now();
    require(ggml_backend_cuda_uma_fill(buffer.get(), 0, bytes, 29) == 0 &&
        checksum(base, bytes) == uint64_t(bytes)*255/2, "GPU-write/CPU-read failed");
    for (uint8_t iteration = 0; iteration < 8; ++iteration) {
        fill(base + page/2, 3*page, iteration);
        require(ggml_backend_cuda_uma_checksum(buffer.get(), page/2, 3*page, &gpu_checksum) == 0 &&
            gpu_checksum == checksum(base + page/2, 3*page), "cross-page handoff failed");
    }
    const auto readiness_start = std::chrono::steady_clock::now();
    require(ggml_backend_cuda_uma_prefetch(buffer.get(), 0, bytes) == 0, "prefetch readiness failed");
    const auto readiness_end = std::chrono::steady_clock::now();

    require(ggml_backend_cuda_uma_run_allocator_negative_controls(0) == 0,
        "CUDA allocator negative controls failed");

    char path[] = "phase11-uma-probe-XXXXXX";
    const int fd = mkstemp(path);
    require(fd >= 0, "source creation failed");
    fill(base, bytes, 41);
    const uint64_t source_checksum = checksum(base, bytes);
    require(write(fd, base, bytes) == ssize_t(bytes) && fsync(fd) == 0, "source write failed");
    memset(base, 0, bytes);
    const auto pread_start = std::chrono::steady_clock::now();
    require(pread(fd, base, bytes, 0) == ssize_t(bytes), "buffered pread failed");
    const auto pread_end = std::chrono::steady_clock::now();
    require(ggml_backend_cuda_uma_checksum(buffer.get(), 0, bytes, &gpu_checksum) == 0 &&
        gpu_checksum == source_checksum, "buffered pread/GPU verification failed");
    struct statx direct_alignment = {};
    require(statx(fd, "", AT_EMPTY_PATH, STATX_DIOALIGN, &direct_alignment) == 0,
        "direct-I/O alignment query failed");
    close(fd);
    unlink(path);

    io_uring_params ring_params = {};
    const int ring_fd = int(syscall(__NR_io_uring_setup, 8, &ring_params));
    const int ring_error = ring_fd < 0 ? errno : 0;
    if (ring_fd >= 0) close(ring_fd);
    require(ring_error == EPERM, "io_uring capability did not match amended host contract");
    std::vector<unsigned char> residency(bytes/page);
    require(mincore(base, bytes, residency.data()) == 0, "mincore failed");
    size_t resident_pages = 0;
    for (unsigned char value : residency) resident_pages += (value & 1U) != 0;
    require(resident_pages != 0, "residency unavailable");
    require(madvise(base + bytes - page, page, MADV_DONTNEED) == 0, "dead-span madvise failed");
    rusage usage = {};
    require(getrusage(RUSAGE_SELF, &usage) == 0, "fault counters unavailable");
    rlimit memlock = {};
    require(getrlimit(RLIMIT_MEMLOCK, &memlock) == 0, "memlock limit unavailable");
    require(memlock.rlim_cur == 8*1024*1024, "memlock limit did not match amended host contract");
    const std::string board = read_text("/sys/class/dmi/id/board_name");
    const std::string product = read_text("/sys/class/dmi/id/product_name");
    require(board.find("EdgeXpert") != std::string::npos && board.find("MS-C931") != std::string::npos &&
        product == "MS-C931", "probe is not running on the authorized MSI EdgeXpert MS-C931 host");
    printf("{\"schema_version\":\"phase11-uma-probe-v1\",\"board\":\"%s\",\"product\":\"%s\","
        "\"gpu\":\"%s\",\"compute_capability\":\"%d.%d\",\"device_count\":%d,"
        "\"cuda_driver_version\":%d,\"cuda_runtime_version\":%d,"
        "\"pageable_memory_access\":%d,\"pageable_uses_host_page_tables\":%d,"
        "\"concurrent_managed_access\":%d,\"direct_managed_host_access\":%d,"
        "\"host_native_atomic\":%d,\"unified_addressing\":%d,\"integrated\":%d,"
        "\"native_io_uring\":\"unavailable_host_seccomp\",\"io_uring_errno\":%d,"
        "\"memlock_limit_bytes\":%llu,\"registered_large_buffer_evidence\":\"unavailable\","
        "\"storage_transport\":\"buffered_pread\",\"direct_io\":\"not_selected_buffered_fallback\","
        "\"direct_io_mem_alignment\":%u,\"direct_io_offset_alignment\":%u,"
        "\"negative_controls\":[\"cudaMalloc_device\",\"cudaMallocManaged_managed\",\"cudaHostAlloc_pinned\"],"
        "\"allocation_bytes\":%zu,\"memory_handoff_service_ns\":%lld,"
        "\"readiness_service_ns\":%lld,\"buffered_pread_service_ns\":%lld,"
        "\"resident_pages\":%zu,\"minor_faults\":%ld,\"major_faults\":%ld,\"status\":\"pass\"}\n",
        board.c_str(), product.c_str(), capabilities.device_name, capabilities.compute_capability_major,
        capabilities.compute_capability_minor, capabilities.device_count, capabilities.cuda_driver_version,
        capabilities.cuda_runtime_version, capabilities.pageable_memory_access,
        capabilities.pageable_memory_access_uses_host_page_tables, capabilities.concurrent_managed_access,
        capabilities.direct_managed_mem_access_from_host, capabilities.host_native_atomic_supported,
        capabilities.unified_addressing, capabilities.integrated, ring_error,
        static_cast<unsigned long long>(memlock.rlim_cur), direct_alignment.stx_dio_mem_align,
        direct_alignment.stx_dio_offset_align, bytes,
        static_cast<long long>(std::chrono::duration_cast<std::chrono::nanoseconds>(handoff_end - handoff_start).count()),
        static_cast<long long>(std::chrono::duration_cast<std::chrono::nanoseconds>(readiness_end - readiness_start).count()),
        static_cast<long long>(std::chrono::duration_cast<std::chrono::nanoseconds>(pread_end - pread_start).count()),
        resident_pages,
        usage.ru_minflt, usage.ru_majflt);
    return 0;
} catch (const std::exception & error) {
    fprintf(stderr, "phase11-uma-probe: %s\n", error.what());
    return 1;
}
