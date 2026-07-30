#include "llama-expert-storage.h"

#include "ggml.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <dirent.h>
#include <unistd.h>
#endif

namespace {

struct temporary_file {
    std::string path;
    std::vector<uint8_t> bytes;

    explicit temporary_file(uint8_t seed, size_t size = 128) : bytes(size) {
        for (size_t index = 0; index < size; ++index) bytes[index] = uint8_t(seed + index);
#if defined(_WIN32)
        char name[L_tmpnam];
        GGML_ASSERT(std::tmpnam(name) != nullptr);
        path = name;
        FILE * file = std::fopen(path.c_str(), "wb");
#else
        char name[] = "/tmp/llama-expert-storage-XXXXXX";
        const int fd = mkstemp(name);
        GGML_ASSERT(fd >= 0);
        path = name;
        FILE * file = fdopen(fd, "wb");
#endif
        GGML_ASSERT(file != nullptr);
        GGML_ASSERT(std::fwrite(bytes.data(), bytes.size(), 1, file) == 1);
        GGML_ASSERT(std::fclose(file) == 0);
    }

    ~temporary_file() { std::remove(path.c_str()); }
};

llm_expert_storage_span span(uint16_t split, uint64_t file_offset, uint64_t count,
        llm_expert_storage_projection projection, uint64_t destination,
        llm_expert_storage_sidecar sidecar = llm_expert_storage_sidecar::weight) {
    return { split, file_offset, count, projection, sidecar, destination, count };
}

std::vector<llm_expert_storage_span> separate_bundle() {
    return {
        span(0, 3, 5, llm_expert_storage_projection::up, 0),
        span(1, 7, 6, llm_expert_storage_projection::gate, 5),
        span(0, 124, 4, llm_expert_storage_projection::down, 11),
    };
}

std::vector<llm_expert_storage_span> merged_bundle() {
    return {
        span(1, 20, 7, llm_expert_storage_projection::gate_up, 0),
        span(0, 40, 3, llm_expert_storage_projection::down, 7),
    };
}

template<class F> void expect_invalid(F function) {
    bool rejected = false;
    try { function(); } catch (const std::invalid_argument &) { rejected = true; }
    GGML_ASSERT(rejected);
}

void populate_and_seal(llm_expert_storage & storage) {
    GGML_ASSERT(storage.add_bundle({ 0, 0 }, separate_bundle()).is_ready());
    GGML_ASSERT(storage.add_bundle({ 0, 1 }, merged_bundle()).is_ready());
    GGML_ASSERT(storage.seal().is_ready());
}

void test_configuration_directory_and_real_reads() {
    temporary_file first(0x10), second(0x80);
    llama_file first_loader(first.path.c_str(), "rb");
    llama_file second_loader(second.path.c_str(), "rb");
    const std::vector<llm_expert_storage_source> sources = {
        { 0, &first_loader, 32 }, { 1, &second_loader, 32 },
    };
    expect_invalid([&] { llm_expert_storage storage({ 0, 2, 8 }, sources); });
    expect_invalid([&] { llm_expert_storage storage({ 1, 2, 8U*1024U*1024U + 1 }, sources); });
    expect_invalid([&] {
        llm_expert_storage storage({ 1, 2, 8 }, { sources[0], sources[0] });
    });
    expect_invalid([&] {
        llm_expert_storage storage({ 1, 2, 8 }, { sources[1] });
    });

    llm_expert_storage storage({ 1, 2, 4 }, sources);
    auto malformed = separate_bundle();
    malformed[0].split_index = 7;
    GGML_ASSERT(storage.add_bundle({ 0, 0 }, malformed).error == llm_expert_storage_error::invalid_directory);
    malformed = separate_bundle();
    malformed[0].file_offset = std::numeric_limits<uint64_t>::max() - 1;
    GGML_ASSERT(storage.add_bundle({ 0, 0 }, malformed).error == llm_expert_storage_error::invalid_directory);
    malformed = separate_bundle();
    malformed[2].byte_count = 5;
    malformed[2].destination_extent = 5;
    GGML_ASSERT(storage.add_bundle({ 0, 0 }, malformed).error == llm_expert_storage_error::invalid_directory);
    malformed = separate_bundle();
    malformed.push_back(malformed[0]);
    GGML_ASSERT(storage.add_bundle({ 0, 0 }, malformed).error == llm_expert_storage_error::invalid_directory);
    malformed = separate_bundle();
    malformed.erase(malformed.begin() + 1);
    GGML_ASSERT(storage.add_bundle({ 0, 0 }, malformed).error == llm_expert_storage_error::invalid_directory);
    malformed = separate_bundle();
    malformed.push_back(span(0, 50, 2, llm_expert_storage_projection::gate_up, 15,
        llm_expert_storage_sidecar::bias));
    GGML_ASSERT(storage.add_bundle({ 0, 0 }, malformed).error == llm_expert_storage_error::invalid_directory);
    malformed = separate_bundle();
    malformed[1].destination_offset = 3;
    GGML_ASSERT(storage.add_bundle({ 0, 0 }, malformed).error == llm_expert_storage_error::invalid_directory);

    populate_and_seal(storage);
    GGML_ASSERT(storage.add_bundle({ 0, 0 }, separate_bundle()).error == llm_expert_storage_error::invalid_key);
    const auto diagnostics = storage.diagnostics();
    GGML_ASSERT(diagnostics.sealed && diagnostics.source_file_count == 2);
    GGML_ASSERT(diagnostics.directory_entry_count == 2 && diagnostics.span_count == 5);
    GGML_ASSERT(diagnostics.administration_bytes > 0);
    GGML_ASSERT(storage.find({ 0, 0 }) != nullptr && storage.find({ 1, 0 }) == nullptr);

    first_loader.seek(17, SEEK_SET);
    std::array<uint8_t, 16> destination{};
    GGML_ASSERT(storage.read_bundle({ 0, 0 }, destination.data(), destination.size()).is_ready());
    GGML_ASSERT(first_loader.tell() == 17);
    GGML_ASSERT(std::memcmp(destination.data(), first.bytes.data() + 3, 5) == 0);
    GGML_ASSERT(std::memcmp(destination.data() + 5, second.bytes.data() + 7, 6) == 0);
    GGML_ASSERT(std::memcmp(destination.data() + 11, first.bytes.data() + 124, 4) == 0);
    GGML_ASSERT(storage.diagnostics().read_chunks == 5);
    GGML_ASSERT(storage.diagnostics().read_bytes == 15);
    GGML_ASSERT(storage.read_bundle({ 0, 0 }, destination.data(), 14).error ==
        llm_expert_storage_error::invalid_destination);
}

struct scripted_reader : llm_expert_storage_read_override {
    enum class action { interrupted, partial, eof, error, normal };
    std::vector<action> actions;
    std::vector<uint8_t> first;
    std::vector<uint8_t> second;
    size_t call = 0;

    int64_t read_at(uint16_t split_index, void * data, size_t size, uint64_t offset, int & native_error) override {
        const action current = call < actions.size() ? actions[call++] : action::normal;
        if (current == action::interrupted) { native_error = EINTR; return -1; }
        if (current == action::eof) return 0;
        if (current == action::error) { native_error = EIO; return -1; }
        const auto & bytes = split_index == 0 ? first : second;
        const size_t count = current == action::partial ? std::min<size_t>(2, size) : size;
        std::memcpy(data, bytes.data() + offset, count);
        return int64_t(count);
    }
};

struct abort_counter { size_t calls = 0; size_t cancel_at = SIZE_MAX; };
bool abort_after(void * data) {
    auto & state = *static_cast<abort_counter *>(data);
    return state.calls++ >= state.cancel_at;
}

void test_retry_short_error_cancel_and_poison() {
    temporary_file first(0x20), second(0x60);
    llama_file first_loader(first.path.c_str(), "rb");
    llama_file second_loader(second.path.c_str(), "rb");
    scripted_reader reader;
    reader.first = first.bytes;
    reader.second = second.bytes;
    llm_expert_storage storage({ 1, 2, 4 }, {
        { 0, &first_loader, 32 }, { 1, &second_loader, 32 },
    }, &reader);
    populate_and_seal(storage);
    std::array<uint8_t, 16> destination{};

    reader.actions = { scripted_reader::action::interrupted, scripted_reader::action::partial };
    GGML_ASSERT(storage.read_bundle({ 0, 0 }, destination.data(), destination.size()).is_ready());
    GGML_ASSERT(std::memcmp(destination.data(), first.bytes.data() + 3, 5) == 0);

    reader.actions = { scripted_reader::action::eof };
    reader.call = 0;
    GGML_ASSERT(storage.read_bundle({ 0, 0 }, destination.data(), destination.size()).error ==
        llm_expert_storage_error::short_read);
    reader.actions = { scripted_reader::action::error };
    reader.call = 0;
    const auto io = storage.read_bundle({ 0, 0 }, destination.data(), destination.size());
    GGML_ASSERT(io.error == llm_expert_storage_error::io_error && io.native_error == EIO);

    reader.actions.clear();
    reader.call = 0;
    abort_counter abort{ 0, 3 };
    GGML_ASSERT(storage.read_bundle({ 0, 0 }, destination.data(), destination.size(), abort_after, &abort).error ==
        llm_expert_storage_error::cancelled);
    const auto diagnostics = storage.diagnostics();
    GGML_ASSERT(diagnostics.short_reads == 1 && diagnostics.io_errors == 1);
    GGML_ASSERT(diagnostics.cancelled_reads == 1 && diagnostics.first_native_error == EIO);

    storage.poison();
    GGML_ASSERT(storage.read_bundle({ 0, 0 }, destination.data(), destination.size()).error ==
        llm_expert_storage_error::poisoned);
}

void test_duplicate_lifetime_and_transactional_close() {
    temporary_file first(0x30), second(0x70);
#if !defined(_WIN32)
    auto descriptor_count = [] {
        DIR * directory = opendir("/proc/self/fd");
        GGML_ASSERT(directory != nullptr);
        size_t count = 0;
        while (readdir(directory) != nullptr) count++;
        closedir(directory);
        return count;
    };
    const size_t baseline = descriptor_count();
#endif
    {
        auto first_loader = std::make_unique<llama_file>(first.path.c_str(), "rb");
        auto second_loader = std::make_unique<llama_file>(second.path.c_str(), "rb");
#if !defined(_WIN32)
        const size_t loaders_open = descriptor_count();
#endif
        bool partial_rejected = false;
        try {
            llm_expert_storage rejected({ 1, 2, 4 }, {
                { 0, first_loader.get(), 32 }, { 1, nullptr, 32 },
            });
        } catch (const std::invalid_argument &) {
            partial_rejected = true;
        }
        GGML_ASSERT(partial_rejected);
#if !defined(_WIN32)
        GGML_ASSERT(descriptor_count() == loaders_open);
#endif
        auto storage = std::make_unique<llm_expert_storage>(llm_expert_storage_config{ 1, 2, 4 },
            std::vector<llm_expert_storage_source>{ { 0, first_loader.get(), 32 }, { 1, second_loader.get(), 32 } });
        populate_and_seal(*storage);
        first_loader.reset();
        second_loader.reset();
        std::array<uint8_t, 16> destination{};
        GGML_ASSERT(storage->read_bundle({ 0, 0 }, destination.data(), destination.size()).is_ready());
#if !defined(_WIN32)
        GGML_ASSERT(descriptor_count() == baseline + 2);
#endif
        storage.reset();
    }
#if !defined(_WIN32)
    GGML_ASSERT(descriptor_count() == baseline);
#endif
}

} // namespace

int main() {
    test_configuration_directory_and_real_reads();
    test_retry_short_error_cancel_and_poison();
    test_duplicate_lifetime_and_transactional_close();
    std::cout << "expert storage tests passed\n";
    return 0;
}
