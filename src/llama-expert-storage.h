#pragma once

#include "llama-expert-weight-provider.h"
#include "llama-mmap.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

enum class llm_expert_storage_projection : uint8_t { up, gate, gate_up, down };
enum class llm_expert_storage_sidecar : uint8_t { weight, bias, scale };

struct llm_expert_storage_span {
    uint16_t split_index = 0;
    uint64_t file_offset = 0;
    uint64_t byte_count = 0;
    llm_expert_storage_projection projection = llm_expert_storage_projection::up;
    llm_expert_storage_sidecar sidecar = llm_expert_storage_sidecar::weight;
    uint64_t destination_offset = 0;
    uint64_t destination_extent = 0;
};

struct llm_expert_storage_source {
    uint16_t split_index = 0;
    const llama_file * file = nullptr;
    uint64_t alignment = 1;
};

struct llm_expert_storage_config {
    uint32_t layer_count = 0;
    uint32_t experts_per_layer = 0;
    uint32_t expected_bundle_count = 0;
    uint64_t maximum_read_chunk = 8U*1024U*1024U;
};

enum class llm_expert_storage_error {
    none,
    invalid_configuration,
    invalid_directory,
    invalid_key,
    invalid_destination,
    cancelled,
    short_read,
    io_error,
    poisoned,
};

struct llm_expert_storage_result {
    llm_expert_storage_error error = llm_expert_storage_error::none;
    int native_error = 0;
    bool is_ready() const { return error == llm_expert_storage_error::none; }
};

using llm_expert_storage_abort = bool (*)(void * user_data);

struct llm_expert_storage_read_override {
    virtual ~llm_expert_storage_read_override() = default;
    virtual int64_t read_at(uint16_t split_index, void * data, size_t size, uint64_t offset, int & native_error) = 0;
};

struct llm_expert_storage_diagnostics {
    uint64_t source_file_count = 0;
    uint64_t directory_entry_count = 0;
    uint64_t span_count = 0;
    uint64_t administration_bytes = 0;
    uint64_t read_requests = 0;
    uint64_t read_chunks = 0;
    uint64_t read_bytes = 0;
    uint64_t cancelled_reads = 0;
    uint64_t short_reads = 0;
    uint64_t io_errors = 0;
    int first_native_error = 0;
    bool sealed = false;
    bool poisoned = false;
};

class llm_expert_storage {
public:
    llm_expert_storage(llm_expert_storage_config config,
            const std::vector<llm_expert_storage_source> & sources,
            llm_expert_storage_read_override * read_override = nullptr);
    ~llm_expert_storage();

    llm_expert_storage(const llm_expert_storage &) = delete;
    llm_expert_storage & operator=(const llm_expert_storage &) = delete;
    llm_expert_storage(llm_expert_storage &&) noexcept;
    llm_expert_storage & operator=(llm_expert_storage &&) noexcept;

    llm_expert_storage_result add_bundle(llm_expert_key key, std::vector<llm_expert_storage_span> spans) noexcept;
    llm_expert_storage_result seal() noexcept;
    const std::vector<llm_expert_storage_span> * find(llm_expert_key key) const noexcept;
    llm_expert_storage_result read_bundle(llm_expert_key key, void * destination, uint64_t destination_size,
            llm_expert_storage_abort abort = nullptr, void * abort_data = nullptr) noexcept;
    void poison() noexcept;
    llm_expert_storage_diagnostics diagnostics() const noexcept;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
