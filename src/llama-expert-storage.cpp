#include "llama-expert-storage.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <limits>
#include <mutex>
#include <new>
#include <stdexcept>

namespace {

bool checked_add(uint64_t lhs, uint64_t rhs, uint64_t & result) {
    if (lhs > std::numeric_limits<uint64_t>::max() - rhs) {
        return false;
    }
    result = lhs + rhs;
    return true;
}

size_t identity_index(llm_expert_storage_projection projection, llm_expert_storage_sidecar sidecar) {
    return size_t(projection)*3 + size_t(sidecar);
}

} // namespace

struct llm_expert_storage::impl {
    struct source {
        uint16_t split_index = 0;
        uint64_t alignment = 1;
        llama_file_read_handle handle;
    };
    struct entry {
        bool present = false;
        std::vector<llm_expert_storage_span> spans;
    };

    llm_expert_storage_config config;
    std::vector<source> sources;
    std::vector<entry> directory;
    llm_expert_storage_read_override * read_override = nullptr;
    mutable std::mutex directory_mutex;
    mutable std::mutex diagnostics_mutex;
    std::atomic<bool> sealed{ false };
    std::atomic<bool> poisoned{ false };
    llm_expert_storage_diagnostics counters;

    size_t index(llm_expert_key key) const {
        return size_t(key.layer)*config.experts_per_layer + uint32_t(key.expert);
    }

    const source * find_source(uint16_t split_index) const {
        auto it = std::lower_bound(sources.begin(), sources.end(), split_index,
            [](const source & value, uint16_t requested) { return value.split_index < requested; });
        return it != sources.end() && it->split_index == split_index ? &*it : nullptr;
    }
};

llm_expert_storage::llm_expert_storage(llm_expert_storage_config config,
        const std::vector<llm_expert_storage_source> & sources,
        llm_expert_storage_read_override * read_override) : pimpl(std::make_unique<impl>()) {
    if (config.layer_count == 0 || config.experts_per_layer == 0 || config.expected_bundle_count == 0 ||
        config.expected_bundle_count > uint64_t(config.layer_count)*config.experts_per_layer || config.maximum_read_chunk == 0 ||
        config.maximum_read_chunk > 8U*1024U*1024U || sources.empty()) {
        throw std::invalid_argument("invalid expert storage configuration");
    }
    pimpl->config = config;
    pimpl->read_override = read_override;
    pimpl->directory.resize(size_t(config.layer_count)*config.experts_per_layer);
    pimpl->sources.reserve(sources.size());
    for (const auto & item : sources) {
        if (item.file == nullptr || item.alignment == 0) {
            throw std::invalid_argument("invalid expert storage source");
        }
        for (const auto & prior : pimpl->sources) {
            if (prior.split_index == item.split_index) {
                throw std::invalid_argument("duplicate expert storage split index");
            }
        }
        auto handle = item.file->duplicate_read_handle();
        if (!handle.valid() || !handle.identity().valid || handle.size() != item.file->size()) {
            throw std::runtime_error("unable to establish duplicated source identity");
        }
        pimpl->sources.push_back({ item.split_index, item.alignment, std::move(handle) });
    }
    std::sort(pimpl->sources.begin(), pimpl->sources.end(),
        [](const impl::source & lhs, const impl::source & rhs) { return lhs.split_index < rhs.split_index; });
    for (size_t index = 0; index < pimpl->sources.size(); ++index) {
        if (pimpl->sources[index].split_index != index) {
            throw std::invalid_argument("expert storage split indices must be contiguous from zero");
        }
    }
    pimpl->counters.source_file_count = pimpl->sources.size();
}

llm_expert_storage::~llm_expert_storage() = default;
llm_expert_storage::llm_expert_storage(llm_expert_storage &&) noexcept = default;
llm_expert_storage & llm_expert_storage::operator=(llm_expert_storage &&) noexcept = default;

llm_expert_storage_result llm_expert_storage::add_bundle(
        llm_expert_key key, std::vector<llm_expert_storage_span> spans) noexcept {
    std::lock_guard<std::mutex> lock(pimpl->directory_mutex);
    if (pimpl->sealed.load() || !key.is_valid(pimpl->config.layer_count, pimpl->config.experts_per_layer) || spans.empty()) {
        return { llm_expert_storage_error::invalid_key, 0 };
    }
    auto & entry = pimpl->directory[pimpl->index(key)];
    if (entry.present) {
        return { llm_expert_storage_error::invalid_directory, 0 };
    }
    std::array<bool, 12> identities{};
    bool up = false, gate = false, gate_up = false, down = false;
    for (const auto & span : spans) {
        const impl::source * source = pimpl->find_source(span.split_index);
        uint64_t file_end = 0;
        uint64_t destination_end = 0;
        const size_t identity = identity_index(span.projection, span.sidecar);
        if (source == nullptr || span.byte_count == 0 || span.destination_extent != span.byte_count ||
            !checked_add(span.file_offset, span.byte_count, file_end) || file_end > source->handle.size() ||
            !checked_add(span.destination_offset, span.destination_extent, destination_end) ||
            identity >= identities.size() || identities[identity]) {
            return { llm_expert_storage_error::invalid_directory, 0 };
        }
        identities[identity] = true;
        if (span.sidecar == llm_expert_storage_sidecar::weight) {
            switch (span.projection) {
                case llm_expert_storage_projection::up: up = true; break;
                case llm_expert_storage_projection::gate: gate = true; break;
                case llm_expert_storage_projection::gate_up: gate_up = true; break;
                case llm_expert_storage_projection::down: down = true; break;
            }
        }
    }
    for (size_t projection = 0; projection < 4; ++projection) {
        if ((identities[projection*3 + size_t(llm_expert_storage_sidecar::bias)] ||
             identities[projection*3 + size_t(llm_expert_storage_sidecar::scale)]) &&
            !identities[projection*3 + size_t(llm_expert_storage_sidecar::weight)]) {
            return { llm_expert_storage_error::invalid_directory, 0 };
        }
    }
    if (!down || !((up && gate && !gate_up) || (!up && !gate && gate_up))) {
        return { llm_expert_storage_error::invalid_directory, 0 };
    }
    for (size_t left = 0; left < spans.size(); ++left) {
        const uint64_t left_end = spans[left].destination_offset + spans[left].destination_extent;
        for (size_t right = left + 1; right < spans.size(); ++right) {
            const uint64_t right_end = spans[right].destination_offset + spans[right].destination_extent;
            if (spans[left].destination_offset < right_end && spans[right].destination_offset < left_end) {
                return { llm_expert_storage_error::invalid_directory, 0 };
            }
        }
    }
    std::sort(spans.begin(), spans.end(), [](const auto & lhs, const auto & rhs) {
        return identity_index(lhs.projection, lhs.sidecar) < identity_index(rhs.projection, rhs.sidecar);
    });
    entry.present = true;
    entry.spans = std::move(spans);
    return {};
}

llm_expert_storage_result llm_expert_storage::seal() noexcept {
    std::lock_guard<std::mutex> lock(pimpl->directory_mutex);
    if (pimpl->sealed.load()) {
        return {};
    }
    uint64_t span_count = 0;
    uint32_t entry_count = 0;
    for (const auto & entry : pimpl->directory) {
        if (entry.present) {
            entry_count++;
            span_count += entry.spans.size();
        }
    }
    if (entry_count != pimpl->config.expected_bundle_count) {
        return { llm_expert_storage_error::invalid_directory, 0 };
    }
    pimpl->counters.directory_entry_count = entry_count;
    pimpl->counters.span_count = span_count;
    pimpl->counters.administration_bytes = sizeof(*pimpl) + pimpl->sources.capacity()*sizeof(impl::source) +
        pimpl->directory.capacity()*sizeof(impl::entry) + span_count*sizeof(llm_expert_storage_span);
    pimpl->sealed.store(true);
    return {};
}

const std::vector<llm_expert_storage_span> * llm_expert_storage::find(llm_expert_key key) const noexcept {
    if (!pimpl->sealed.load() || !key.is_valid(pimpl->config.layer_count, pimpl->config.experts_per_layer)) {
        return nullptr;
    }
    const auto & entry = pimpl->directory[pimpl->index(key)];
    return entry.present ? &entry.spans : nullptr;
}

llm_expert_storage_result llm_expert_storage::read_bundle(llm_expert_key key, void * destination,
        uint64_t destination_size, llm_expert_storage_abort abort, void * abort_data) noexcept {
    if (pimpl->poisoned.load()) {
        return { llm_expert_storage_error::poisoned, 0 };
    }
    if (!pimpl->sealed.load() || !key.is_valid(pimpl->config.layer_count, pimpl->config.experts_per_layer) ||
        !pimpl->directory[pimpl->index(key)].present) {
        return { llm_expert_storage_error::invalid_key, 0 };
    }
    if (destination == nullptr) {
        return { llm_expert_storage_error::invalid_destination, 0 };
    }
    {
        std::lock_guard<std::mutex> lock(pimpl->diagnostics_mutex);
        pimpl->counters.read_requests++;
    }
    const auto & spans = pimpl->directory[pimpl->index(key)].spans;
    for (const auto & span : spans) {
        uint64_t destination_end = 0;
        if (!checked_add(span.destination_offset, span.byte_count, destination_end) || destination_end > destination_size) {
            return { llm_expert_storage_error::invalid_destination, 0 };
        }
        if (abort && abort(abort_data)) {
            std::lock_guard<std::mutex> lock(pimpl->diagnostics_mutex);
            pimpl->counters.cancelled_reads++;
            return { llm_expert_storage_error::cancelled, 0 };
        }
        const impl::source * source = pimpl->find_source(span.split_index);
        uint64_t completed = 0;
        while (completed < span.byte_count) {
            if (abort && abort(abort_data)) {
                std::lock_guard<std::mutex> lock(pimpl->diagnostics_mutex);
                pimpl->counters.cancelled_reads++;
                return { llm_expert_storage_error::cancelled, 0 };
            }
            const size_t chunk = size_t(std::min<uint64_t>(pimpl->config.maximum_read_chunk, span.byte_count - completed));
            size_t chunk_completed = 0;
            while (chunk_completed < chunk) {
                int native_error = 0;
                auto * target = static_cast<uint8_t *>(destination) + span.destination_offset + completed + chunk_completed;
                const size_t remaining = chunk - chunk_completed;
                const int64_t result = pimpl->read_override ?
                    pimpl->read_override->read_at(span.split_index, target, remaining,
                        span.file_offset + completed + chunk_completed, native_error) :
                    source->handle.read_at(target, remaining, span.file_offset + completed + chunk_completed, native_error);
                if (result < 0 && native_error == EINTR) {
                    continue;
                }
                if (result < 0) {
                    std::lock_guard<std::mutex> lock(pimpl->diagnostics_mutex);
                    pimpl->counters.io_errors++;
                    if (pimpl->counters.first_native_error == 0) pimpl->counters.first_native_error = native_error;
                    return { llm_expert_storage_error::io_error, native_error };
                }
                if (result == 0 || uint64_t(result) > remaining) {
                    std::lock_guard<std::mutex> lock(pimpl->diagnostics_mutex);
                    pimpl->counters.short_reads++;
                    return { llm_expert_storage_error::short_read, 0 };
                }
                chunk_completed += size_t(result);
                {
                    std::lock_guard<std::mutex> lock(pimpl->diagnostics_mutex);
                    pimpl->counters.read_bytes += uint64_t(result);
                }
                if (abort && abort(abort_data)) {
                    std::lock_guard<std::mutex> lock(pimpl->diagnostics_mutex);
                    pimpl->counters.cancelled_reads++;
                    return { llm_expert_storage_error::cancelled, 0 };
                }
            }
            completed += chunk;
            std::lock_guard<std::mutex> lock(pimpl->diagnostics_mutex);
            pimpl->counters.read_chunks++;
        }
    }
    return {};
}

llm_expert_storage_result llm_expert_storage::read_bundle(llm_expert_key key,
        const llm_expert_storage_destination * destinations, size_t destination_count,
        llm_expert_storage_abort abort, void * abort_data) noexcept {
    if (pimpl->poisoned.load()) {
        return { llm_expert_storage_error::poisoned, 0 };
    }
    if (!pimpl->sealed.load() || !key.is_valid(pimpl->config.layer_count, pimpl->config.experts_per_layer) ||
        !pimpl->directory[pimpl->index(key)].present) {
        return { llm_expert_storage_error::invalid_key, 0 };
    }
    if (destinations == nullptr || destination_count == 0 || destination_count > 12) {
        return { llm_expert_storage_error::invalid_destination, 0 };
    }
    std::array<bool, 12> seen{};
    for (size_t index = 0; index < destination_count; ++index) {
        const size_t identity = identity_index(destinations[index].projection, destinations[index].sidecar);
        if (identity >= seen.size() || seen[identity] || destinations[index].data == nullptr || destinations[index].extent == 0) {
            return { llm_expert_storage_error::invalid_destination, 0 };
        }
        seen[identity] = true;
    }
    {
        std::lock_guard<std::mutex> lock(pimpl->diagnostics_mutex);
        pimpl->counters.read_requests++;
    }
    const auto & spans = pimpl->directory[pimpl->index(key)].spans;
    for (const auto & span : spans) {
        const llm_expert_storage_destination * destination = nullptr;
        for (size_t index = 0; index < destination_count; ++index) {
            if (destinations[index].projection == span.projection && destinations[index].sidecar == span.sidecar) {
                destination = &destinations[index];
                break;
            }
        }
        if (destination == nullptr || destination->extent != span.destination_extent) {
            return { llm_expert_storage_error::invalid_destination, 0 };
        }
        if (abort && abort(abort_data)) {
            std::lock_guard<std::mutex> lock(pimpl->diagnostics_mutex);
            pimpl->counters.cancelled_reads++;
            return { llm_expert_storage_error::cancelled, 0 };
        }
        const impl::source * source = pimpl->find_source(span.split_index);
        uint64_t completed = 0;
        while (completed < span.byte_count) {
            if (abort && abort(abort_data)) {
                std::lock_guard<std::mutex> lock(pimpl->diagnostics_mutex);
                pimpl->counters.cancelled_reads++;
                return { llm_expert_storage_error::cancelled, 0 };
            }
            const size_t chunk = size_t(std::min<uint64_t>(pimpl->config.maximum_read_chunk, span.byte_count - completed));
            size_t chunk_completed = 0;
            while (chunk_completed < chunk) {
                int native_error = 0;
                auto * target = static_cast<uint8_t *>(destination->data) + completed + chunk_completed;
                const size_t remaining = chunk - chunk_completed;
                const int64_t result = pimpl->read_override ?
                    pimpl->read_override->read_at(span.split_index, target, remaining,
                        span.file_offset + completed + chunk_completed, native_error) :
                    source->handle.read_at(target, remaining, span.file_offset + completed + chunk_completed, native_error);
                if (result < 0 && native_error == EINTR) {
                    continue;
                }
                if (result < 0) {
                    std::lock_guard<std::mutex> lock(pimpl->diagnostics_mutex);
                    pimpl->counters.io_errors++;
                    if (pimpl->counters.first_native_error == 0) pimpl->counters.first_native_error = native_error;
                    return { llm_expert_storage_error::io_error, native_error };
                }
                if (result == 0 || uint64_t(result) > remaining) {
                    std::lock_guard<std::mutex> lock(pimpl->diagnostics_mutex);
                    pimpl->counters.short_reads++;
                    return { llm_expert_storage_error::short_read, 0 };
                }
                chunk_completed += size_t(result);
                {
                    std::lock_guard<std::mutex> lock(pimpl->diagnostics_mutex);
                    pimpl->counters.read_bytes += uint64_t(result);
                }
                if (abort && abort(abort_data)) {
                    std::lock_guard<std::mutex> lock(pimpl->diagnostics_mutex);
                    pimpl->counters.cancelled_reads++;
                    return { llm_expert_storage_error::cancelled, 0 };
                }
            }
            completed += chunk;
            std::lock_guard<std::mutex> lock(pimpl->diagnostics_mutex);
            pimpl->counters.read_chunks++;
        }
    }
    return {};
}

void llm_expert_storage::poison() noexcept {
    std::lock_guard<std::mutex> lock(pimpl->diagnostics_mutex);
    pimpl->poisoned.store(true);
}

llm_expert_storage_diagnostics llm_expert_storage::diagnostics() const noexcept {
    std::lock_guard<std::mutex> lock(pimpl->diagnostics_mutex);
    auto result = pimpl->counters;
    result.sealed = pimpl->sealed.load();
    result.poisoned = pimpl->poisoned.load();
    return result;
}
