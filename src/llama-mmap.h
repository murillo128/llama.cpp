#pragma once

#include <cstdint>
#include <memory>
#include <vector>
#include <cstdio>

struct llama_file;
struct llama_file_read_handle;
struct llama_mmap;
struct llama_mlock;

using llama_files  = std::vector<std::unique_ptr<llama_file>>;
using llama_mmaps  = std::vector<std::unique_ptr<llama_mmap>>;
using llama_mlocks = std::vector<std::unique_ptr<llama_mlock>>;

struct llama_file {
    llama_file(const char * fname, const char * mode, bool use_direct_io = false);
    llama_file(FILE * file);
    ~llama_file();

    size_t tell() const;
    size_t size() const;

    int file_id() const; // fileno overload

    llama_file_read_handle duplicate_read_handle() const;

    void seek(size_t offset, int whence) const;

    void read_raw(void * ptr, size_t len);
    void read_raw_unsafe(void * ptr, size_t len);
    void read_aligned_chunk(void * dest, size_t size);
    uint32_t read_u32();

    void write_raw(const void * ptr, size_t len) const;
    void write_u32(uint32_t val) const;

    size_t read_alignment() const;
    bool has_direct_io() const;
private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

struct llama_file_identity {
    uint64_t device = 0;
    uint64_t file = 0;
    bool valid = false;

    bool operator==(const llama_file_identity & other) const {
        return valid && other.valid && device == other.device && file == other.file;
    }
};

// A model-lifetime, cursor-independent read handle duplicated from a loader file.
// read_at() performs one positional system call and never advances shared state.
struct llama_file_read_handle {
    llama_file_read_handle();
    ~llama_file_read_handle();
    llama_file_read_handle(const llama_file_read_handle &) = delete;
    llama_file_read_handle & operator=(const llama_file_read_handle &) = delete;
    llama_file_read_handle(llama_file_read_handle &&) noexcept;
    llama_file_read_handle & operator=(llama_file_read_handle &&) noexcept;

    bool valid() const;
    uint64_t size() const;
    llama_file_identity identity() const;
    int64_t read_at(void * data, size_t size, uint64_t offset, int & native_error) const noexcept;

private:
    friend struct llama_file;
    struct impl;
    explicit llama_file_read_handle(std::unique_ptr<impl> impl);
    std::unique_ptr<impl> pimpl;
};

struct llama_mmap {
    llama_mmap(const llama_mmap &) = delete;
    llama_mmap(struct llama_file * file, size_t prefetch = (size_t) -1, bool numa = false);
    ~llama_mmap();

    size_t size() const;
    void * addr() const;

    void unmap_fragment(size_t first, size_t last);

    static const bool SUPPORTED;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

struct llama_mlock {
    llama_mlock();
    ~llama_mlock();

    void init(void * ptr);
    void grow_to(size_t target_size);

    static const bool SUPPORTED;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

size_t llama_path_max();
