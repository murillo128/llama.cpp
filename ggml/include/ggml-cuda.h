#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#ifdef GGML_USE_HIP
#define GGML_CUDA_NAME "ROCm"
#define GGML_CUBLAS_NAME "hipBLAS"
#elif defined(GGML_USE_MUSA)
#define GGML_CUDA_NAME "MUSA"
#define GGML_CUBLAS_NAME "muBLAS"
#else
#define GGML_CUDA_NAME "CUDA"
#define GGML_CUBLAS_NAME "cuBLAS"
#endif
#define GGML_CUDA_MAX_DEVICES       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_cuda_init(int device);

GGML_BACKEND_API bool ggml_backend_is_cuda(ggml_backend_t backend);

// device buffer
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device);

// conduct allreduce operation between devices
GGML_BACKEND_API bool ggml_backend_cuda_allreduce_tensor(ggml_backend_t * backends, struct ggml_tensor ** tensors, size_t n_backends);

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type(void);

struct ggml_backend_cuda_uma_capabilities {
    uint32_t struct_size;
    int32_t device;
    int32_t device_count;
    int32_t compute_capability_major;
    int32_t compute_capability_minor;
    int32_t pageable_memory_access;
    int32_t pageable_memory_access_uses_host_page_tables;
    int32_t concurrent_managed_access;
    int32_t direct_managed_mem_access_from_host;
    int32_t host_native_atomic_supported;
    int32_t unified_addressing;
    int32_t integrated;
    char device_name[256];
};

GGML_BACKEND_API int ggml_backend_cuda_uma_get_capabilities(int device, struct ggml_backend_cuda_uma_capabilities * capabilities);
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_uma_buffer_type(int device);
GGML_BACKEND_API bool ggml_backend_buft_is_cuda_uma(ggml_backend_buffer_type_t buft);
GGML_BACKEND_API int ggml_backend_cuda_uma_prefetch(ggml_backend_buffer_t buffer, size_t offset, size_t size);
GGML_BACKEND_API int ggml_backend_cuda_uma_checksum(ggml_backend_buffer_t buffer, size_t offset, size_t size, uint64_t * checksum);
GGML_BACKEND_API int ggml_backend_cuda_uma_fill(ggml_backend_buffer_t buffer, size_t offset, size_t size, uint8_t seed);
GGML_BACKEND_API int ggml_backend_cuda_uma_run_allocator_negative_controls(int device);

GGML_BACKEND_API int  ggml_backend_cuda_get_device_count(void);
GGML_BACKEND_API void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);

GGML_BACKEND_API bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size);
GGML_BACKEND_API void ggml_backend_cuda_unregister_host_buffer(void * buffer);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cuda_reg(void);

#ifdef  __cplusplus
}
#endif
