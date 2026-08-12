llama_add_compile_flags()

#
# libraries
#

# llama

file(GLOB LLAMA_MODELS_SOURCES "models/*.cpp")

add_library(llama
            ../include/llama.h
            llama.cpp
            llama-adapter.cpp
            llama-arch.cpp
            llama-batch.cpp
            llama-chat.cpp
            llama-context.cpp
            llama-cparams.cpp
            llama-cache-aware-routing.cpp
            llama-cold-expert-cache.cpp
            llama-expert-cache-policy.cpp
            llama-expert-prefetch.cpp
            llama-expert-async-io.cpp
            llama-expert-resident-demand.cpp
            llama-expert-scheduler.cpp
            llama-expert-transfer-ring.cpp
    llama-expert-uma.cpp
    llama-uma-expert-weight-provider.cpp
            llama-expert-storage.cpp
            llama-expert-weight-provider.cpp
            llama-grammar.cpp
            llama-graph.cpp
            llama-hparams.cpp
            llama-impl.cpp
            llama-io.cpp
            llama-kv-cache.cpp
            llama-kv-cache-iswa.cpp
            llama-kv-cache-dsa.cpp
            llama-kv-cache-dsv4.cpp
            llama-memory.cpp
            llama-memory-hybrid.cpp
            llama-memory-hybrid-iswa.cpp
            llama-memory-recurrent.cpp
            llama-mmap.cpp
            llama-model-loader.cpp
            llama-model-saver.cpp
            llama-model.cpp
            llama-quant.cpp
            llama-sampler.cpp
            llama-vocab.cpp
            unicode-data.cpp
            unicode.cpp
            unicode.h
            ${LLAMA_MODELS_SOURCES}
            )

set_target_properties(llama PROPERTIES
    VERSION ${LLAMA_INSTALL_VERSION}
    SOVERSION 0
    MACHO_CURRENT_VERSION 0 # keep macOS linker from seeing oversized version number
)

target_include_directories(llama PRIVATE .)
target_include_directories(llama PRIVATE ../vendor)
target_include_directories(llama PUBLIC ../include)
target_compile_features   (llama PRIVATE cxx_std_17) # don't bump

target_link_libraries(llama PUBLIC ggml)

if (LLAMA_PERFETTO)
    if (NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
        message(FATAL_ERROR "LLAMA_PERFETTO requires Linux system tracing")
    endif()
    if (NOT GGML_CUDA)
        message(FATAL_ERROR "LLAMA_PERFETTO requires GGML_CUDA for CUPTI activity correlation")
    endif()

    find_package(Threads REQUIRED)
    find_package(CUDAToolkit REQUIRED)
    if (NOT TARGET CUDA::cupti)
        message(FATAL_ERROR "LLAMA_PERFETTO requires the CUDA Toolkit CUPTI development library")
    endif()

    add_library(llama-perfetto-sdk STATIC ../vendor/perfetto-v50.1/sdk/perfetto.cc)
    target_include_directories(llama-perfetto-sdk PUBLIC ../vendor/perfetto-v50.1/sdk)
    target_compile_features(llama-perfetto-sdk PRIVATE cxx_std_17)
    target_link_libraries(llama-perfetto-sdk PRIVATE Threads::Threads ${CMAKE_DL_LIBS})
    set_target_properties(llama-perfetto-sdk PROPERTIES POSITION_INDEPENDENT_CODE ON)

    target_sources(llama PRIVATE llama-perfetto-trace.cpp)
    target_include_directories(llama PRIVATE ../vendor/perfetto-v50.1/sdk)
    target_compile_definitions(llama PRIVATE LLAMA_PERFETTO=1)
    target_link_libraries(llama PRIVATE llama-perfetto-sdk CUDA::cupti CUDA::cudart Threads::Threads ${CMAKE_DL_LIBS})
endif()

if (BUILD_SHARED_LIBS)
    set_target_properties(llama PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(llama PRIVATE LLAMA_BUILD)
    target_compile_definitions(llama PUBLIC  LLAMA_SHARED)
endif()
