#include "llama-expert-storage.h"
#include "llama-model.h"

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

struct storage_lifetime_evidence {
    size_t baseline = 0;
    size_t peak = 0;
    size_t final = 0;
    bool supported = false;
};

storage_lifetime_evidence lifetime_evidence;

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
    expect_invalid([&] { llm_expert_storage storage({ 0, 2, 2, 8 }, sources); });
    expect_invalid([&] { llm_expert_storage storage({ 1, 2, 2, 8U*1024U*1024U + 1 }, sources); });
    expect_invalid([&] {
        llm_expert_storage storage({ 1, 2, 2, 8 }, { sources[0], sources[0] });
    });
    expect_invalid([&] {
        llm_expert_storage storage({ 1, 2, 2, 8 }, { sources[1] });
    });

    llm_expert_storage storage({ 1, 2, 2, 4, llm_expert_integrity_mode::fnv64_end_to_end }, sources);
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
    std::array<intptr_t, 2> native_handles{};
    size_t native_handle_count = 0;
    GGML_ASSERT(storage.copy_source_native_handles(
        native_handles.data(), native_handles.size(), native_handle_count).is_ready());
    GGML_ASSERT(native_handle_count == native_handles.size());
    GGML_ASSERT(native_handles[0] >= 0 && native_handles[1] >= 0);
    size_t rejected_handle_count = 99;
    GGML_ASSERT(storage.copy_source_native_handles(
        native_handles.data(), 1, rejected_handle_count).error == llm_expert_storage_error::invalid_destination);
    GGML_ASSERT(rejected_handle_count == 0);
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

    std::array<uint8_t, 5> up{};
    std::array<uint8_t, 6> gate{};
    std::array<uint8_t, 4> down{};
    const std::array<llm_expert_storage_destination, 3> scattered = {{
        { llm_expert_storage_projection::up, llm_expert_storage_sidecar::weight, up.data(), up.size(), 2 },
        { llm_expert_storage_projection::gate, llm_expert_storage_sidecar::weight, gate.data(), gate.size(), 2 },
        { llm_expert_storage_projection::down, llm_expert_storage_sidecar::weight, down.data(), down.size(), 2 },
    }};
    llm_expert_storage unchecked_storage({ 1, 2, 2, 4 }, sources);
    populate_and_seal(unchecked_storage);
    const auto unchecked_read = unchecked_storage.read_bundle({ 0, 0 }, scattered.data(), scattered.size());
    const auto unchecked_diagnostics = unchecked_storage.diagnostics();
    GGML_ASSERT(unchecked_read.is_ready() && unchecked_read.digest == 0 &&
        unchecked_read.integrity_status == llm_expert_integrity_status::not_checked);
    GGML_ASSERT(unchecked_diagnostics.integrity_mode == llm_expert_integrity_mode::none &&
        unchecked_diagnostics.integrity_status == llm_expert_integrity_status::not_checked &&
        unchecked_diagnostics.integrity_checks == 0 && unchecked_diagnostics.integrity_digest_bytes == 0);
    GGML_ASSERT(storage.read_bundle({ 0, 0 }, scattered.data(), scattered.size()).is_ready());
    const auto scattered_read = storage.read_bundle({ 0, 0 }, scattered.data(), scattered.size());
    uint64_t independent_digest = 1469598103934665603ULL;
    for (const auto & destination : scattered) {
        const auto * bytes = static_cast<const uint8_t *>(destination.data);
        for (uint64_t index = 0; index < destination.extent; ++index) {
            independent_digest ^= bytes[index]; independent_digest *= 1099511628211ULL;
        }
    }
    GGML_ASSERT(scattered_read.digest == independent_digest);
    GGML_ASSERT(scattered_read.integrity_status == llm_expert_integrity_status::passed);
    storage.record_integrity_status(llm_expert_integrity_status::passed, 15);
    GGML_ASSERT(storage.diagnostics().integrity_checks == 1 &&
        storage.diagnostics().integrity_digest_bytes == 15 &&
        storage.diagnostics().integrity_status == llm_expert_integrity_status::passed);
    GGML_ASSERT(std::memcmp(up.data(), first.bytes.data() + 3, up.size()) == 0);
    GGML_ASSERT(std::memcmp(gate.data(), second.bytes.data() + 7, gate.size()) == 0);
    GGML_ASSERT(std::memcmp(down.data(), first.bytes.data() + 124, down.size()) == 0);
    std::array<llm_expert_storage_read_operation, 12> read_plan;
    size_t read_operation_count = 0;
    GGML_ASSERT(storage.make_read_plan({ 0, 0 }, scattered.data(), scattered.size(),
        read_plan.data(), read_plan.size(), read_operation_count).is_ready());
    GGML_ASSERT(read_operation_count == 3);
    for (size_t index = 0; index < read_operation_count; ++index) {
        GGML_ASSERT(read_plan[index].native_handle >= 0);
        GGML_ASSERT(read_plan[index].layout_class_id == 2);
        GGML_ASSERT(read_plan[index].segment_count == 1);
        GGML_ASSERT(read_plan[index].segments[0].layout_class_id == 2);
        GGML_ASSERT(read_plan[index].byte_count == read_plan[index].segments[0].byte_count);
    }
    auto malformed_destination = scattered;
    malformed_destination[1].layout_class_id = 3;
    GGML_ASSERT(storage.make_read_plan({ 0, 0 }, malformed_destination.data(), malformed_destination.size(),
        read_plan.data(), read_plan.size(), read_operation_count).error ==
        llm_expert_storage_error::invalid_destination);
    malformed_destination = scattered;
    malformed_destination[1].extent--;
    GGML_ASSERT(storage.read_bundle({ 0, 0 }, malformed_destination.data(), malformed_destination.size()).error ==
        llm_expert_storage_error::invalid_destination);
    GGML_ASSERT(storage.read_bundle({ 0, 0 }, destination.data(), 14).error ==
        llm_expert_storage_error::invalid_destination);
    up[0] ^= 0xff;
    uint64_t corrupted_digest = 1469598103934665603ULL;
    for (const auto & item : scattered) {
        const auto * bytes = static_cast<const uint8_t *>(item.data);
        for (uint64_t index = 0; index < item.extent; ++index) {
            corrupted_digest ^= bytes[index]; corrupted_digest *= 1099511628211ULL;
        }
    }
    GGML_ASSERT(corrupted_digest != scattered_read.digest);
    storage.record_integrity_status(llm_expert_integrity_status::failed, 15);
    GGML_ASSERT(storage.diagnostics().integrity_mismatches == 1 && storage.diagnostics().poisoned);
    GGML_ASSERT(storage.read_bundle({ 0, 0 }, scattered.data(), scattered.size()).error ==
        llm_expert_storage_error::poisoned);
}

void test_exact_adjacent_read_plan() {
    temporary_file first(0x20), second(0x90);
    llama_file first_loader(first.path.c_str(), "rb");
    llama_file second_loader(second.path.c_str(), "rb");
    llm_expert_storage storage({ 1, 1, 1, 64 }, {
        { 0, &first_loader, 32 }, { 1, &second_loader, 32 },
    });
    GGML_ASSERT(storage.add_bundle({ 0, 0 }, {
        span(0, 8, 5, llm_expert_storage_projection::up, 0),
        span(0, 13, 7, llm_expert_storage_projection::gate, 5),
        span(1, 4, 6, llm_expert_storage_projection::down, 12),
    }).is_ready());
    GGML_ASSERT(storage.seal().is_ready());
    std::array<uint8_t, 5> up{};
    std::array<uint8_t, 7> gate{};
    std::array<uint8_t, 6> down{};
    const std::array<llm_expert_storage_destination, 3> destinations = {{
        { llm_expert_storage_projection::up, llm_expert_storage_sidecar::weight, up.data(), up.size() },
        { llm_expert_storage_projection::gate, llm_expert_storage_sidecar::weight, gate.data(), gate.size() },
        { llm_expert_storage_projection::down, llm_expert_storage_sidecar::weight, down.data(), down.size() },
    }};
    std::array<llm_expert_storage_read_operation, 12> operations;
    size_t count = 0;
    GGML_ASSERT(storage.make_read_plan({ 0, 0 }, destinations.data(), destinations.size(),
        operations.data(), operations.size(), count).is_ready());
    GGML_ASSERT(count == 2);
    GGML_ASSERT(operations[0].split_index == 0 && operations[0].segment_count == 2);
    GGML_ASSERT(operations[0].file_offset == 8 && operations[0].byte_count == 12);
    GGML_ASSERT(operations[1].split_index == 1 && operations[1].segment_count == 1);
}

void test_direct_source_identity_and_explicit_support() {
#if defined(__linux__)
    temporary_file first(0x21, 8192), second(0x91, 8192);
    llama_file buffered(first.path.c_str(), "rb");
    llama_file buffered_second(second.path.c_str(), "rb");
    llm_expert_storage storage({ 1, 1, 1, 64 }, {
        { 0, &buffered, 32, first.path.c_str(), true },
        { 1, &buffered_second, 32, second.path.c_str(), true },
    });
    GGML_ASSERT(storage.add_bundle({ 0, 0 }, separate_bundle()).is_ready());
    GGML_ASSERT(storage.seal().is_ready());
    const auto diagnostics = storage.diagnostics();
    GGML_ASSERT(diagnostics.direct_source_count == 2);
    GGML_ASSERT(diagnostics.direct_unsupported_source_count == 0);
    std::array<uint8_t, 5> up{};
    std::array<uint8_t, 6> gate{};
    std::array<uint8_t, 4> down{};
    const std::array<llm_expert_storage_destination, 3> targets = {{
        { llm_expert_storage_projection::up, llm_expert_storage_sidecar::weight, up.data(), up.size() },
        { llm_expert_storage_projection::gate, llm_expert_storage_sidecar::weight, gate.data(), gate.size() },
        { llm_expert_storage_projection::down, llm_expert_storage_sidecar::weight, down.data(), down.size() },
    }};
    std::array<llm_expert_storage_read_operation, 3> operations;
    size_t count = 0;
    GGML_ASSERT(storage.make_read_plan({ 0, 0 }, targets.data(), targets.size(),
        operations.data(), operations.size(), count).is_ready());
    GGML_ASSERT(count == 3 && operations[0].source_size == first.bytes.size());
    std::array<intptr_t, 2> transport_handles{};
    size_t transport_handle_count = 0;
    GGML_ASSERT(storage.copy_source_native_handles(transport_handles.data(), transport_handles.size(),
        transport_handle_count, true).is_ready());
    GGML_ASSERT(transport_handle_count == 2);
    GGML_ASSERT(operations[0].direct_native_handle >= 0);
    GGML_ASSERT(operations[0].direct_alignment == diagnostics.maximum_direct_alignment);
    GGML_ASSERT(transport_handles[0] == operations[0].direct_native_handle);
    bool identity_rejected = false;
    try {
        llm_expert_storage rejected({ 1, 1, 1, 64 }, {
            { 0, &buffered, 32, second.path.c_str(), true },
            { 1, &buffered_second, 32, second.path.c_str(), true },
        });
    } catch (const std::runtime_error &) {
        identity_rejected = true;
    }
    GGML_ASSERT(identity_rejected);

    expect_invalid([&] {
        llm_expert_storage unsupported({ 1, 1, 1, 64 }, {
            { 0, &buffered, 32, nullptr, true },
            { 1, &buffered_second, 32, nullptr, true },
        });
    });
#endif
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
    llm_expert_storage storage({ 1, 2, 2, 4 }, {
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
    std::array<uint8_t, 5> up{};
    std::array<uint8_t, 6> gate{};
    std::array<uint8_t, 4> down{};
    const std::array<llm_expert_storage_destination, 3> scattered = {{
        { llm_expert_storage_projection::up, llm_expert_storage_sidecar::weight, up.data(), up.size() },
        { llm_expert_storage_projection::gate, llm_expert_storage_sidecar::weight, gate.data(), gate.size() },
        { llm_expert_storage_projection::down, llm_expert_storage_sidecar::weight, down.data(), down.size() },
    }};
    abort = { 0, 2 };
    GGML_ASSERT(storage.read_bundle({ 0, 0 }, scattered.data(), scattered.size(), abort_after, &abort).error ==
        llm_expert_storage_error::cancelled);
    const auto diagnostics = storage.diagnostics();
    GGML_ASSERT(diagnostics.short_reads == 1 && diagnostics.io_errors == 1);
    GGML_ASSERT(diagnostics.cancelled_reads == 2 && diagnostics.first_native_error == EIO);

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
    lifetime_evidence.baseline = baseline;
    lifetime_evidence.supported = true;
#endif
    {
        auto first_loader = std::make_unique<llama_file>(first.path.c_str(), "rb");
        auto second_loader = std::make_unique<llama_file>(second.path.c_str(), "rb");
#if !defined(_WIN32)
        const size_t loaders_open = descriptor_count();
#endif
        bool partial_rejected = false;
        try {
            llm_expert_storage rejected({ 1, 2, 2, 4 }, {
                { 0, first_loader.get(), 32 }, { 1, nullptr, 32 },
            });
        } catch (const std::invalid_argument &) {
            partial_rejected = true;
        }
        GGML_ASSERT(partial_rejected);
#if !defined(_WIN32)
        GGML_ASSERT(descriptor_count() == loaders_open);
#endif
        auto storage = std::make_unique<llm_expert_storage>(llm_expert_storage_config{ 1, 2, 2, 4 },
            std::vector<llm_expert_storage_source>{ { 0, first_loader.get(), 32 }, { 1, second_loader.get(), 32 } });
        populate_and_seal(*storage);
        first_loader.reset();
        second_loader.reset();
        std::array<uint8_t, 16> destination{};
        GGML_ASSERT(storage->read_bundle({ 0, 0 }, destination.data(), destination.size()).is_ready());
#if !defined(_WIN32)
        GGML_ASSERT(descriptor_count() == baseline + 2);
        lifetime_evidence.peak = descriptor_count();
#endif
        storage.reset();
    }
#if !defined(_WIN32)
    GGML_ASSERT(descriptor_count() == baseline);
    lifetime_evidence.final = descriptor_count();
#endif
}

void test_cold_mode_accepts_mmap_and_direct_before_loading() {
    llama_model_params params = llama_model_default_params();
    params.expert_weights_mode = LLAMA_EXPERT_WEIGHTS_MODE_COLD_CACHE;
    params.expert_hot_cache_capacity = 2;
    params.expert_cold_cache_bytes = 1U << 20;
    params.expert_transfer_ring_bytes = 1U << 20;
    for (const auto mode : { LLAMA_LOAD_MODE_NONE, LLAMA_LOAD_MODE_MLOCK,
            LLAMA_LOAD_MODE_MMAP_MLOCK }) {
        params.load_mode = mode;
        expect_invalid([&] {
            std::unique_ptr<llama_model> model(llama_model_create(LLM_ARCH_KIMI_K3, params));
        });
    }
    for (const auto mode : { LLAMA_LOAD_MODE_MMAP, LLAMA_LOAD_MODE_DIRECT_IO }) {
        params.load_mode = mode;
        std::unique_ptr<llama_model> valid(llama_model_create(LLM_ARCH_KIMI_K3, params));
        GGML_ASSERT(valid != nullptr);
    }
}

} // namespace

int main() {
    test_configuration_directory_and_real_reads();
    test_exact_adjacent_read_plan();
    test_direct_source_identity_and_explicit_support();
    test_retry_short_error_cancel_and_poison();
    test_duplicate_lifetime_and_transactional_close();
    test_cold_mode_accepts_mmap_and_direct_before_loading();
    std::cout << "PHASE6_STORAGE_LIFETIME"
              << "\tsupported=" << lifetime_evidence.supported
              << "\tbaseline=" << lifetime_evidence.baseline
              << "\tpeak=" << lifetime_evidence.peak
              << "\tfinal=" << lifetime_evidence.final
              << "\tbalanced=" << (lifetime_evidence.supported &&
                    lifetime_evidence.final == lifetime_evidence.baseline)
              << '\n';
    std::cout << "expert storage tests passed\n";
    return 0;
}
