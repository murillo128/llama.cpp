#include "ggml-cuda.h"
#include "ggml-backend-impl.h"
#include "ggml-cuda/common.cuh"

#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>

#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif

struct ggml_backend_cuda_uma_buffer_type_context {
    int device;
    std::string name;
};

static __global__ void ggml_cuda_uma_checksum_kernel(
        const uint8_t * bytes, size_t size, unsigned long long * checksum);
static __global__ void ggml_cuda_uma_fill_kernel(uint8_t * bytes, size_t size, uint8_t seed);

static bool ggml_backend_cuda_uma_authorized_board() {
#if defined(__linux__) && defined(__aarch64__)
    auto read_value = [](const char * path, char * value, size_t size) {
        FILE * file = fopen(path, "rb");
        if (file == nullptr) return false;
        const size_t count = fread(value, 1, size - 1, file);
        fclose(file);
        while (count != 0 && (value[strlen(value) - 1] == '\n' || value[strlen(value) - 1] == '\r')) {
            value[strlen(value) - 1] = 0;
        }
        return count != 0;
    };
    char board[256] = {};
    char product[256] = {};
    return read_value("/sys/class/dmi/id/board_name", board, sizeof(board)) &&
        read_value("/sys/class/dmi/id/product_name", product, sizeof(product)) &&
        strcmp(board, "EdgeXpert (MS-C931)") == 0 && strcmp(product, "MS-C931") == 0;
#else
    return false;
#endif
}

static const char * ggml_backend_cuda_uma_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    auto * ctx = static_cast<ggml_backend_cuda_uma_buffer_type_context *>(buft->context);
    return ctx->name.c_str();
}

bool ggml_backend_buft_is_cuda_uma(ggml_backend_buffer_type_t buft) {
    return buft != nullptr && buft->iface.get_name == ggml_backend_cuda_uma_buffer_type_get_name;
}

int ggml_backend_cuda_uma_get_capabilities(
        int device, ggml_backend_cuda_uma_capabilities * capabilities) {
    if (capabilities == nullptr || capabilities->struct_size != sizeof(*capabilities) ||
        device < 0 || device >= ggml_backend_cuda_get_device_count()) return int(cudaErrorInvalidValue);
    ggml_backend_cuda_uma_capabilities result = {};
    result.struct_size = sizeof(result);
    result.device = device;
    result.device_count = ggml_backend_cuda_get_device_count();
    cudaDeviceProp properties = {};
    cudaError_t error = cudaGetDeviceProperties(&properties, device);
    if (error != cudaSuccess) return int(error);
    result.compute_capability_major = properties.major;
    result.compute_capability_minor = properties.minor;
    result.unified_addressing = properties.unifiedAddressing;
    result.integrated = properties.integrated;
    error = cudaDriverGetVersion(&result.cuda_driver_version);
    if (error != cudaSuccess) return int(error);
    error = cudaRuntimeGetVersion(&result.cuda_runtime_version);
    if (error != cudaSuccess) return int(error);
    snprintf(result.device_name, sizeof(result.device_name), "%s", properties.name);
    const cudaDeviceAttr attributes[] = {
        cudaDevAttrPageableMemoryAccess,
        cudaDevAttrPageableMemoryAccessUsesHostPageTables,
        cudaDevAttrConcurrentManagedAccess,
        cudaDevAttrDirectManagedMemAccessFromHost,
        cudaDevAttrHostNativeAtomicSupported,
    };
    int * outputs[] = {
        &result.pageable_memory_access,
        &result.pageable_memory_access_uses_host_page_tables,
        &result.concurrent_managed_access,
        &result.direct_managed_mem_access_from_host,
        &result.host_native_atomic_supported,
    };
    for (size_t i = 0; i < sizeof(attributes)/sizeof(attributes[0]); ++i) {
        error = cudaDeviceGetAttribute(outputs[i], attributes[i], device);
        if (error != cudaSuccess) return int(error);
    }
    *capabilities = result;
    return 0;
}

static bool ggml_backend_cuda_uma_supported(int device) {
    ggml_backend_cuda_uma_capabilities capabilities = {};
    capabilities.struct_size = sizeof(capabilities);
    return ggml_backend_cuda_uma_get_capabilities(device, &capabilities) == 0 &&
        capabilities.device_count == 1 && strcmp(capabilities.device_name, "NVIDIA GB10") == 0 &&
        capabilities.compute_capability_major == 12 && capabilities.compute_capability_minor == 1 &&
        capabilities.integrated != 0 && capabilities.pageable_memory_access != 0 &&
        capabilities.pageable_memory_access_uses_host_page_tables != 0 && capabilities.unified_addressing != 0 &&
        ggml_backend_cuda_uma_authorized_board();
}

static bool ggml_backend_cuda_uma_native_qualification(int device) {
#if defined(__linux__) && defined(__aarch64__)
    static std::once_flag flags[GGML_CUDA_MAX_DEVICES];
    static bool qualified[GGML_CUDA_MAX_DEVICES] = {};
    if (device < 0 || device >= GGML_CUDA_MAX_DEVICES) return false;
    std::call_once(flags[device], [device]() {
        if (!ggml_backend_cuda_uma_supported(device)) return;
        constexpr size_t size = 4096;
        void * mapping = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (mapping == MAP_FAILED) return;
        auto * bytes = static_cast<uint8_t *>(mapping);
        unsigned long long expected = 0;
        for (size_t i = 0; i < size; ++i) {
            bytes[i] = uint8_t(17 + uint8_t(i*131U));
            expected += bytes[i];
        }
        ggml_cuda_set_device(device);
        unsigned long long * device_checksum = nullptr;
        cudaError_t error = cudaMalloc(&device_checksum, sizeof(*device_checksum));
        if (error == cudaSuccess) error = cudaMemset(device_checksum, 0, sizeof(*device_checksum));
        if (error == cudaSuccess) {
            ggml_cuda_uma_checksum_kernel<<<1, 256>>>(bytes, size, device_checksum);
            error = cudaGetLastError();
        }
        unsigned long long observed = 0;
        if (error == cudaSuccess) {
            error = cudaMemcpy(&observed, device_checksum, sizeof(observed), cudaMemcpyDeviceToHost);
        }
        if (error == cudaSuccess && observed == expected) {
            ggml_cuda_uma_fill_kernel<<<1, 256>>>(bytes, size, 29);
            error = cudaGetLastError();
        }
        if (error == cudaSuccess) error = cudaDeviceSynchronize();
        if (error == cudaSuccess) {
            for (size_t i = 0; i < size; ++i) {
                if (bytes[i] != uint8_t(29 + uint8_t(i*131U))) {
                    error = cudaErrorUnknown;
                    break;
                }
            }
        }
        if (device_checksum != nullptr) cudaFree(device_checksum);
        const bool coherent = error == cudaSuccess;
        const int unmap_result = munmap(mapping, size);
        qualified[device] = coherent && unmap_result == 0;
    });
    return qualified[device];
#else
    GGML_UNUSED(device);
    return false;
#endif
}

static void ggml_backend_cuda_uma_buffer_free_buffer(ggml_backend_buffer_t buffer) {
#if defined(__linux__)
    if (buffer->context != nullptr && buffer->size != 0) GGML_ASSERT(munmap(buffer->context, buffer->size) == 0);
#else
    GGML_UNUSED(buffer);
#endif
}

static ggml_backend_buffer_t ggml_backend_cuda_uma_buffer_type_alloc_buffer(
        ggml_backend_buffer_type_t buft, size_t size) {
#if defined(__linux__) && defined(__aarch64__)
    auto * buft_ctx = static_cast<ggml_backend_cuda_uma_buffer_type_context *>(buft->context);
    if (size == 0 || !ggml_backend_cuda_uma_native_qualification(buft_ctx->device)) return nullptr;
    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0 || size > SIZE_MAX - size_t(page_size - 1)) return nullptr;
    const size_t mapped_size = (size + size_t(page_size - 1))/size_t(page_size)*size_t(page_size);
    void * ptr = mmap(nullptr, mapped_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (ptr == MAP_FAILED) return nullptr;
    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, mapped_size);
    if (buffer == nullptr) {
        munmap(ptr, mapped_size);
        return nullptr;
    }
    buffer->buft = buft;
    buffer->iface.free_buffer = ggml_backend_cuda_uma_buffer_free_buffer;
    return buffer;
#else
    GGML_UNUSED(buft);
    GGML_UNUSED(size);
    return nullptr;
#endif
}

static size_t ggml_backend_cuda_uma_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
#if defined(__linux__)
    const long page_size = sysconf(_SC_PAGESIZE);
    return page_size > 0 ? size_t(page_size) : 4096;
#else
    return 4096;
#endif
}

static size_t ggml_backend_cuda_uma_buffer_type_get_alloc_size(
        ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    GGML_UNUSED(buft);
    return ggml_nbytes(tensor);
}

static bool ggml_backend_cuda_uma_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return true;
}

static const ggml_backend_buffer_type_i ggml_backend_cuda_uma_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_cuda_uma_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_cuda_uma_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_cuda_uma_buffer_type_get_alignment,
    /* .get_max_size     = */ nullptr,
    /* .get_alloc_size   = */ ggml_backend_cuda_uma_buffer_type_get_alloc_size,
    /* .is_host          = */ ggml_backend_cuda_uma_buffer_type_is_host,
};

ggml_backend_buffer_type_t ggml_backend_cuda_uma_buffer_type(int device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    if (device < 0 || device >= ggml_backend_cuda_get_device_count()) return nullptr;
    static ggml_backend_buffer_type types[GGML_CUDA_MAX_DEVICES];
    static bool initialized = false;
    if (!initialized) {
        for (int i = 0; i < ggml_backend_cuda_get_device_count(); ++i) {
            types[i] = {
                /* .iface    = */ ggml_backend_cuda_uma_buffer_type_interface,
                /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), i),
                /* .context  = */ new ggml_backend_cuda_uma_buffer_type_context{
                    i, GGML_CUDA_NAME + std::string("_UMA") + std::to_string(i)},
            };
        }
        initialized = true;
    }
    return &types[device];
}

static __global__ void ggml_cuda_uma_checksum_kernel(
        const uint8_t * bytes, size_t size, unsigned long long * checksum) {
    unsigned long long local = 0;
    for (size_t index = blockIdx.x*blockDim.x + threadIdx.x; index < size;
            index += blockDim.x*gridDim.x) local += bytes[index];
    atomicAdd(checksum, local);
}

static __global__ void ggml_cuda_uma_fill_kernel(uint8_t * bytes, size_t size, uint8_t seed) {
    for (size_t index = blockIdx.x*blockDim.x + threadIdx.x; index < size;
            index += blockDim.x*gridDim.x) bytes[index] = uint8_t(seed + uint8_t(index*131U));
}

static bool ggml_backend_cuda_uma_validate_range(
        ggml_backend_buffer_t buffer, size_t offset, size_t size,
        ggml_backend_cuda_uma_buffer_type_context *& context) {
    if (buffer == nullptr || !ggml_backend_buft_is_cuda_uma(buffer->buft) ||
        offset > buffer->size || size > buffer->size - offset) return false;
    context = static_cast<ggml_backend_cuda_uma_buffer_type_context *>(buffer->buft->context);
    return context != nullptr;
}

int ggml_backend_cuda_uma_prefetch(ggml_backend_buffer_t buffer, size_t offset, size_t size) {
    ggml_backend_cuda_uma_buffer_type_context * context = nullptr;
    if (!ggml_backend_cuda_uma_validate_range(buffer, offset, size, context)) return int(cudaErrorInvalidValue);
    ggml_cuda_set_device(context->device);
    cudaStream_t stream = nullptr;
    cudaError_t error = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (error == cudaSuccess) {
        const cudaMemLocation location = { cudaMemLocationTypeDevice, context->device };
        error = cudaMemPrefetchAsync(static_cast<uint8_t *>(buffer->context) + offset, size, location, 0, stream);
    }
    if (error == cudaSuccess) error = cudaStreamSynchronize(stream);
    if (stream != nullptr) {
        const cudaError_t destroy_error = cudaStreamDestroy(stream);
        if (error == cudaSuccess) error = destroy_error;
    }
    return int(error);
}

int ggml_backend_cuda_uma_checksum(
        ggml_backend_buffer_t buffer, size_t offset, size_t size, uint64_t * checksum) {
    ggml_backend_cuda_uma_buffer_type_context * context = nullptr;
    if (checksum == nullptr || !ggml_backend_cuda_uma_validate_range(buffer, offset, size, context)) return int(cudaErrorInvalidValue);
    ggml_cuda_set_device(context->device);
    unsigned long long * device_checksum = nullptr;
    cudaError_t error = cudaMalloc(&device_checksum, sizeof(*device_checksum));
    if (error == cudaSuccess) error = cudaMemset(device_checksum, 0, sizeof(*device_checksum));
    if (error == cudaSuccess) {
        ggml_cuda_uma_checksum_kernel<<<128, 256>>>(
            static_cast<const uint8_t *>(buffer->context) + offset, size, device_checksum);
        error = cudaGetLastError();
    }
    unsigned long long result = 0;
    if (error == cudaSuccess) error = cudaMemcpy(&result, device_checksum, sizeof(result), cudaMemcpyDeviceToHost);
    if (device_checksum != nullptr) cudaFree(device_checksum);
    if (error == cudaSuccess) *checksum = uint64_t(result);
    return int(error);
}

int ggml_backend_cuda_uma_fill(
        ggml_backend_buffer_t buffer, size_t offset, size_t size, uint8_t seed) {
    ggml_backend_cuda_uma_buffer_type_context * context = nullptr;
    if (!ggml_backend_cuda_uma_validate_range(buffer, offset, size, context)) return int(cudaErrorInvalidValue);
    ggml_cuda_set_device(context->device);
    ggml_cuda_uma_fill_kernel<<<128, 256>>>(static_cast<uint8_t *>(buffer->context) + offset, size, seed);
    cudaError_t error = cudaGetLastError();
    if (error == cudaSuccess) error = cudaDeviceSynchronize();
    return int(error);
}

int ggml_backend_cuda_uma_run_allocator_negative_controls(int device) {
    if (device < 0 || device >= ggml_backend_cuda_get_device_count()) return int(cudaErrorInvalidDevice);
    ggml_cuda_set_device(device);
    constexpr size_t size = 4096;
    void * pointers[3] = {};
    cudaPointerAttributes attributes = {};
    cudaError_t error = cudaMalloc(&pointers[0], size);
    if (error == cudaSuccess) error = cudaPointerGetAttributes(&attributes, pointers[0]);
    if (error == cudaSuccess && attributes.type != cudaMemoryTypeDevice) error = cudaErrorInvalidValue;
    if (error == cudaSuccess) error = cudaMallocManaged(&pointers[1], size);
    if (error == cudaSuccess) error = cudaPointerGetAttributes(&attributes, pointers[1]);
    if (error == cudaSuccess && attributes.type != cudaMemoryTypeManaged) error = cudaErrorInvalidValue;
    if (error == cudaSuccess) error = cudaHostAlloc(&pointers[2], size, cudaHostAllocDefault);
    if (error == cudaSuccess) error = cudaPointerGetAttributes(&attributes, pointers[2]);
    if (error == cudaSuccess && attributes.type != cudaMemoryTypeHost) error = cudaErrorInvalidValue;
    if (pointers[0] != nullptr) {
        const cudaError_t free_error = cudaFree(pointers[0]);
        if (error == cudaSuccess) error = free_error;
    }
    if (pointers[1] != nullptr) {
        const cudaError_t free_error = cudaFree(pointers[1]);
        if (error == cudaSuccess) error = free_error;
    }
    if (pointers[2] != nullptr) {
        const cudaError_t free_error = cudaFreeHost(pointers[2]);
        if (error == cudaSuccess) error = free_error;
    }
    return int(error);
}
