#include "llama-cold-expert-cache.h"
#include "llama-perfetto-trace.h"

#include "llama-hparams.h"

#include "ggml-alloc.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_set>

#ifdef __linux__
#include <cerrno>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace {

int expert_axis(const ggml_tensor * tensor, int32_t n_expert, bool weight) {
    if (tensor == nullptr) {
        return -1;
    }
    if (weight) {
        return ggml_n_dims(tensor) >= 3 && tensor->ne[2] == n_expert ? 2 : -1;
    }
    int result = -1;
    for (int axis = ggml_n_dims(tensor) - 1; axis >= 0; --axis) {
        if (tensor->ne[axis] == n_expert) {
            if (result >= 0) {
                return -1;
            }
            result = axis;
        }
    }
    return result;
}

bool checked_add(uint64_t lhs, uint64_t rhs, uint64_t & result) {
    if (lhs > std::numeric_limits<uint64_t>::max() - rhs) {
        return false;
    }
    result = lhs + rhs;
    return true;
}

bool key_matches(const llm_expert_key & lhs, const llm_expert_key & rhs) {
    return lhs.layer == rhs.layer && lhs.expert == rhs.expert;
}

llm_expert_provider_result policy_result(llm_expert_cache_policy_result result) {
    if (result.is_ready()) return llm_expert_provider_result::success();
    switch (result.error) {
        case llm_expert_cache_policy_error::no_victim:
            return llm_expert_provider_result::failure(llm_expert_provider_error::busy);
        case llm_expert_cache_policy_error::metadata_mismatch:
            return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
        case llm_expert_cache_policy_error::sequence_exhausted:
        case llm_expert_cache_policy_error::overflow:
        case llm_expert_cache_policy_error::transcript_full:
            return llm_expert_provider_result::failure(llm_expert_provider_error::unsupported_configuration);
        case llm_expert_cache_policy_error::invalid_configuration:
        case llm_expert_cache_policy_error::invalid_key:
        case llm_expert_cache_policy_error::invalid_event:
            return llm_expert_provider_result::failure(llm_expert_provider_error::invalid_key);
        case llm_expert_cache_policy_error::none:
            break;
    }
    return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
}

bool projection_is_host_accessible(const llm_expert_projection_descriptor & projection) {
    for (const auto * tensor : { projection.weight, projection.bias, projection.scale }) {
        if (tensor != nullptr && (tensor->buffer == nullptr || !ggml_backend_buffer_is_host(tensor->buffer))) {
            return false;
        }
    }
    return true;
}

bool bundle_is_host_accessible(const llm_expert_bundle_descriptor & bundle) {
    return projection_is_host_accessible(bundle.up) && projection_is_host_accessible(bundle.gate) &&
        projection_is_host_accessible(bundle.gate_up) && projection_is_host_accessible(bundle.down);
}

ggml_tensor * make_slot_tensor(
        ggml_context * ctx,
        const ggml_tensor * source,
        int32_t n_expert,
        uint32_t capacity,
        bool weight,
        const char * name) {
    if (source == nullptr) {
        return nullptr;
    }
    const int axis = expert_axis(source, n_expert, weight);
    if (axis < 0) {
        throw std::invalid_argument("expert tensor has no unambiguous expert axis");
    }
    int64_t ne[GGML_MAX_DIMS];
    for (int index = 0; index < GGML_MAX_DIMS; ++index) {
        ne[index] = source->ne[index];
    }
    ne[axis] = capacity;
    ggml_tensor * tensor = ggml_new_tensor(ctx, source->type, ggml_n_dims(source), ne);
    ggml_set_name(tensor, name);
    return tensor;
}

llm_expert_projection_descriptor make_slot_projection(
        ggml_context * ctx,
        const llm_expert_projection_descriptor & source,
        int32_t n_expert,
        uint32_t capacity,
        const char * prefix) {
    const std::string base(prefix);
    return {
        make_slot_tensor(ctx, source.weight, n_expert, capacity, true, (base + ".weight").c_str()),
        make_slot_tensor(ctx, source.bias, n_expert, capacity, false, (base + ".bias").c_str()),
        make_slot_tensor(ctx, source.scale, n_expert, capacity, false, (base + ".scale").c_str()),
        nullptr,
    };
}

struct cold_allocation {
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buffer;
    llm_expert_bundle_descriptor bundle = {};
    std::vector<llm_expert_bundle_descriptor> bundles;
    std::vector<llm_expert_layout_class_id> layer_ids;
    std::vector<uint64_t> payload_bytes;
    uint64_t slot_stride = 0;
    std::array<uint64_t, 12> role_offsets = {};
    std::array<uint64_t, 12> role_extents = {};
};

ggml_tensor * bundle_member(
        llm_expert_bundle_descriptor & bundle,
        size_t projection_index,
        size_t member_index) {
    llm_expert_projection_descriptor * projection = nullptr;
    if (projection_index == 0) projection = &bundle.up;
    else if (projection_index == 1) projection = &bundle.gate;
    else if (projection_index == 2) projection = &bundle.gate_up;
    else if (projection_index == 3) projection = &bundle.down;
    if (projection == nullptr) return nullptr;
    return member_index == 0 ? projection->weight : member_index == 1 ? projection->bias : projection->scale;
}

#ifdef __linux__
void add_tensor_slot_pages(
        std::unordered_set<uintptr_t> & pages,
        const ggml_tensor * tensor,
        uint32_t slot,
        int32_t n_expert,
        bool weight,
        uintptr_t page_mask) {
    if (tensor == nullptr || tensor->data == nullptr) return;
    const int axis = expert_axis(tensor, n_expert, weight);
    if (axis < 0) return;
    const uintptr_t begin = reinterpret_cast<uintptr_t>(tensor->data) + uintptr_t(slot)*tensor->nb[axis];
    const uintptr_t end = begin + tensor->nb[axis];
    for (uintptr_t page = begin & page_mask; page < end; page += ~page_mask + 1) pages.insert(page);
}

void sample_ready_residency(
        const cold_allocation & arena,
        const std::vector<llm_cold_cache_diagnostics::slot> & slots,
        llm_cold_cache_diagnostics & result) {
    const long page_size_value = sysconf(_SC_PAGESIZE);
    if (page_size_value <= 0 || (uint64_t(page_size_value) & (uint64_t(page_size_value) - 1)) != 0) {
        result.residency_unavailable_reason = "sysconf(_SC_PAGESIZE) returned an invalid value";
        return;
    }
    const uintptr_t page_size = uintptr_t(page_size_value);
    const uintptr_t page_mask = ~(page_size - 1);
    std::unordered_set<uintptr_t> page_set;
    const auto add_projection = [&](const llm_expert_projection_descriptor & projection,
                                    uint32_t slot, int32_t capacity) {
        add_tensor_slot_pages(page_set, projection.weight, slot, capacity, true, page_mask);
        add_tensor_slot_pages(page_set, projection.bias, slot, capacity, false, page_mask);
        add_tensor_slot_pages(page_set, projection.scale, slot, capacity, false, page_mask);
    };
    for (uint32_t slot = 0; slot < slots.size(); ++slot) {
        if (slots[slot].state != llm_cold_slot_state::ready) continue;
        const auto class_id = slots[slot].layout_class_id;
        if (class_id >= arena.bundles.size()) continue;
        const auto & bundle = arena.bundles[class_id];
        result.ready_logical_bytes += arena.payload_bytes[class_id];
        add_projection(bundle.up, slot, bundle.n_expert);
        add_projection(bundle.gate, slot, bundle.n_expert);
        add_projection(bundle.gate_up, slot, bundle.n_expert);
        add_projection(bundle.down, slot, bundle.n_expert);
    }
    result.ready_page_count = page_set.size();
    if (page_set.empty()) {
        result.residency_supported = true;
        return;
    }
    std::vector<uintptr_t> pages(page_set.begin(), page_set.end());
    std::sort(pages.begin(), pages.end());
    size_t begin = 0;
    while (begin < pages.size()) {
        size_t end = begin + 1;
        while (end < pages.size() && pages[end] == pages[end - 1] + page_size) end++;
        std::vector<unsigned char> status(end - begin);
        if (mincore(reinterpret_cast<void *>(pages[begin]), (end - begin)*page_size, status.data()) != 0) {
            result.residency_unavailable_reason = std::string("mincore failed: errno=") + std::to_string(errno);
            result.ready_page_count = 0;
            result.resident_ready_page_count = 0;
            result.resident_ready_bytes = 0;
            return;
        }
        for (unsigned char value : status) result.resident_ready_page_count += (value & 1) != 0;
        begin = end;
    }
    result.resident_ready_bytes = result.resident_ready_page_count*page_size;
    result.residency_supported = true;
}
#endif

size_t align_up(size_t value, size_t alignment) {
    if (alignment == 0 || value > SIZE_MAX - (alignment - 1)) return SIZE_MAX;
    return (value + alignment - 1)/alignment*alignment;
}

size_t universal_slot_stride(
        const llm_expert_layout_registry & registry,
        ggml_backend_buffer_type_t buffer_type,
        std::array<size_t, 12> * offsets = nullptr,
        std::array<size_t, 12> * extents = nullptr) {
    if (!registry.sealed() || registry.classes.size() > LLM_EXPERT_LAYOUT_CLASS_MAX) return SIZE_MAX;
    const auto buft = buffer_type == nullptr ? ggml_backend_cpu_buffer_type() : buffer_type;
    const size_t buffer_alignment = ggml_backend_buft_get_alignment(buft);
    std::array<size_t, 12> role_offsets = {};
    std::array<size_t, 12> role_extents = {};
    size_t slot_alignment = std::max<size_t>(4096, buffer_alignment);
    size_t within_slot = 0;
    for (size_t projection = 0; projection < 4; ++projection) {
        for (size_t member_index = 0; member_index < 3; ++member_index) {
            const size_t role = projection*3 + member_index;
            const size_t role_alignment = std::max<size_t>(4096, buffer_alignment);
            for (const auto & layout_class : registry.classes) {
                auto source = layout_class.prototype;
                const auto * tensor = bundle_member(source, projection, member_index);
                if (tensor == nullptr) continue;
                const int axis = expert_axis(tensor, source.n_expert, member_index == 0);
                if (axis < 0) return SIZE_MAX;
                role_extents[role] = std::max(role_extents[role], tensor->nb[axis]);
            }
            if (role_extents[role] == 0) continue;
            role_offsets[role] = align_up(within_slot, role_alignment);
            if (role_offsets[role] == SIZE_MAX || role_extents[role] > SIZE_MAX - role_offsets[role]) return SIZE_MAX;
            within_slot = role_offsets[role] + role_extents[role];
        }
    }
    const size_t stride = align_up(within_slot, slot_alignment);
    if (offsets != nullptr) *offsets = role_offsets;
    if (extents != nullptr) *extents = role_extents;
    return stride;
}

std::shared_ptr<cold_allocation> make_universal_allocation(
        const llm_expert_layout_registry & registry,
        uint32_t capacity,
        ggml_backend_buffer_type_t buffer_type) {
    const auto buft = buffer_type == nullptr ? ggml_backend_cpu_buffer_type() : buffer_type;
    std::array<size_t, 12> offsets;
    std::array<size_t, 12> extents;
    const size_t slot_stride = universal_slot_stride(registry, buft, &offsets, &extents);
    if (slot_stride == SIZE_MAX || slot_stride == 0 || capacity > SIZE_MAX/slot_stride) return nullptr;
    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead()*32*registry.classes.size(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    auto result = std::make_shared<cold_allocation>();
    result->ctx.reset(ggml_init(params));
    if (!result->ctx) return nullptr;
    result->buffer.reset(ggml_backend_buft_alloc_buffer(buft, slot_stride*capacity));
    if (!result->buffer || !ggml_backend_buffer_is_host(result->buffer.get())) return nullptr;
    auto * base = static_cast<uint8_t *>(ggml_backend_buffer_get_base(result->buffer.get()));
    if (base == nullptr) return nullptr;
    std::memset(base, 0xa5, slot_stride*capacity);
    result->bundles.reserve(registry.classes.size());
    result->payload_bytes.reserve(registry.classes.size());
    for (const auto & layout_class : registry.classes) {
        const auto & source = layout_class.prototype;
        llm_expert_bundle_descriptor bundle = {};
        bundle.layer = -1;
        bundle.n_expert = capacity;
        bundle.up = make_slot_projection(result->ctx.get(), source.up, source.n_expert, capacity, "cold.up");
        bundle.gate = make_slot_projection(result->ctx.get(), source.gate, source.n_expert, capacity, "cold.gate");
        bundle.gate_up = make_slot_projection(result->ctx.get(), source.gate_up, source.n_expert, capacity, "cold.gate_up");
        bundle.down = make_slot_projection(result->ctx.get(), source.down, source.n_expert, capacity, "cold.down");
        for (size_t projection = 0; projection < 4; ++projection) {
            for (size_t member_index = 0; member_index < 3; ++member_index) {
                ggml_tensor * tensor = bundle_member(bundle, projection, member_index);
                if (tensor == nullptr) continue;
                const int axis = expert_axis(tensor, capacity, member_index == 0);
                if (axis < 0) return nullptr;
                tensor->nb[axis] = slot_stride;
                for (int upper = axis + 1; upper < GGML_MAX_DIMS; ++upper) {
                    if (tensor->nb[upper - 1] > SIZE_MAX/tensor->ne[upper - 1]) return nullptr;
                    tensor->nb[upper] = tensor->nb[upper - 1]*tensor->ne[upper - 1];
                }
                if (ggml_backend_tensor_alloc(result->buffer.get(), tensor,
                        base + offsets[projection*3 + member_index]) != GGML_STATUS_SUCCESS) return nullptr;
            }
        }
        for (auto * projection : { &bundle.up, &bundle.gate, &bundle.gate_up, &bundle.down }) {
            if (projection->weight != nullptr) projection->buffer_type = buft;
        }
        result->bundles.push_back(bundle);
        result->payload_bytes.push_back(layout_class.payload_bytes);
    }
    result->bundle = result->bundles.front();
    result->layer_ids = registry.layer_ids;
    result->slot_stride = slot_stride;
    std::copy(offsets.begin(), offsets.end(), result->role_offsets.begin());
    std::copy(extents.begin(), extents.end(), result->role_extents.begin());
    return result;
}

std::shared_ptr<cold_allocation> make_allocation(
        const llm_expert_bundle_descriptor & source,
        uint32_t capacity,
        bool allocate,
        ggml_backend_buffer_type_t buffer_type) {
    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead()*32,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    auto result = std::make_shared<cold_allocation>();
    result->ctx.reset(ggml_init(params));
    if (!result->ctx) {
        return nullptr;
    }
    result->bundle.layer = -1;
    result->bundle.n_expert = capacity;
    result->bundle.up = make_slot_projection(result->ctx.get(), source.up, source.n_expert, capacity, "cold.up");
    result->bundle.gate = make_slot_projection(result->ctx.get(), source.gate, source.n_expert, capacity, "cold.gate");
    result->bundle.gate_up = make_slot_projection(result->ctx.get(), source.gate_up, source.n_expert, capacity, "cold.gate_up");
    result->bundle.down = make_slot_projection(result->ctx.get(), source.down, source.n_expert, capacity, "cold.down");
    if (allocate && buffer_type != nullptr) {
        struct tensor_layout { ggml_tensor * tensor; int axis; size_t offset; size_t span; };
        std::vector<tensor_layout> layouts;
        size_t within_slot = 0;
        for (auto * projection : { &result->bundle.up, &result->bundle.gate,
                &result->bundle.gate_up, &result->bundle.down }) {
            size_t sidecar = 0;
            for (ggml_tensor * tensor : { projection->weight, projection->bias, projection->scale }) {
                if (tensor == nullptr) { sidecar++; continue; }
                const int axis = expert_axis(tensor, capacity, sidecar == 0);
                const size_t span = tensor->nb[axis];
                within_slot = align_up(within_slot, std::max<size_t>(64, ggml_type_size(tensor->type)));
                if (axis < 0 || span == 0 || within_slot == SIZE_MAX || span > SIZE_MAX - within_slot) return nullptr;
                layouts.push_back({ tensor, axis, within_slot, span });
                within_slot += span;
                sidecar++;
            }
        }
        const size_t alignment = std::max<size_t>(4096, ggml_backend_buft_get_alignment(buffer_type));
        const size_t slot_stride = align_up(within_slot, alignment);
        if (slot_stride == SIZE_MAX || capacity > SIZE_MAX/slot_stride) return nullptr;
        result->buffer.reset(ggml_backend_buft_alloc_buffer(buffer_type, slot_stride*capacity));
        if (!result->buffer) return nullptr;
        auto * base = static_cast<uint8_t *>(ggml_backend_buffer_get_base(result->buffer.get()));
        for (const auto & layout : layouts) {
            layout.tensor->nb[layout.axis] = slot_stride;
            for (int axis = layout.axis + 1; axis < GGML_MAX_DIMS; ++axis) {
                layout.tensor->nb[axis] = layout.tensor->nb[axis - 1]*layout.tensor->ne[axis - 1];
            }
            if (ggml_backend_tensor_alloc(result->buffer.get(), layout.tensor, base + layout.offset) != GGML_STATUS_SUCCESS) {
                return nullptr;
            }
        }
    } else if (allocate) {
        result->buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(
            result->ctx.get(), ggml_backend_cpu_buffer_type()));
        if (!result->buffer || !ggml_backend_buffer_is_host(result->buffer.get())) {
            return nullptr;
        }
        for (auto * projection : { &result->bundle.up, &result->bundle.gate, &result->bundle.gate_up, &result->bundle.down }) {
            if (projection->weight) {
                projection->buffer_type = ggml_backend_buffer_get_type(projection->weight->buffer);
            }
        }
    }
    return result;
}

uint64_t allocation_size(
        const llm_expert_bundle_descriptor & source,
        uint32_t capacity,
        ggml_backend_buffer_type_t buffer_type) {
    if (buffer_type != nullptr) {
        size_t within_slot = 0;
        for (const auto * projection : { &source.up, &source.gate, &source.gate_up, &source.down }) {
            size_t sidecar = 0;
            for (const ggml_tensor * tensor : { projection->weight, projection->bias, projection->scale }) {
                if (tensor == nullptr) { sidecar++; continue; }
                const int axis = expert_axis(tensor, source.n_expert, sidecar == 0);
                if (axis < 0) return UINT64_MAX;
                within_slot = align_up(within_slot, std::max<size_t>(64, ggml_type_size(tensor->type)));
                if (within_slot == SIZE_MAX || tensor->nb[axis] > SIZE_MAX - within_slot) return UINT64_MAX;
                within_slot += tensor->nb[axis];
                sidecar++;
            }
        }
        const size_t stride = align_up(within_slot,
            std::max<size_t>(4096, ggml_backend_buft_get_alignment(buffer_type)));
        return stride == SIZE_MAX || capacity > UINT64_MAX/stride ? UINT64_MAX : uint64_t(stride)*capacity;
    }
    auto allocation = make_allocation(source, capacity, false, nullptr);
    if (!allocation) {
        return std::numeric_limits<uint64_t>::max();
    }
    return ggml_backend_alloc_ctx_tensors_from_buft_size(
        allocation->ctx.get(), ggml_backend_cpu_buffer_type());
}

bool bundle_payload(const llm_expert_bundle_descriptor & source, uint64_t & bytes) {
    bytes = 0;
    for (const auto * projection : { &source.up, &source.gate, &source.gate_up, &source.down }) {
        size_t index = 0;
        for (const auto * tensor : { projection->weight, projection->bias, projection->scale }) {
            if (tensor == nullptr) {
                index++;
                continue;
            }
            const int axis = expert_axis(tensor, source.n_expert, index == 0);
            if (axis < 0 || !checked_add(bytes, tensor->nb[axis], bytes)) {
                return false;
            }
            index++;
        }
    }
    return bytes > 0;
}

bool copy_tensor(
        ggml_tensor * target,
        const ggml_tensor * source,
        int32_t n_expert,
        uint32_t capacity,
        int32_t expert,
        uint32_t slot,
        bool weight,
        size_t & bytes) {
    if (target == nullptr || source == nullptr) {
        return target == source;
    }
    const int axis = expert_axis(source, n_expert, weight);
    if (axis < 0 || target->ne[axis] != capacity || source->data == nullptr || target->data == nullptr ||
        target->nb[axis] < source->nb[axis]) {
        return false;
    }
    for (int upper = axis + 1; upper < ggml_n_dims(source); ++upper) {
        if (source->ne[upper] != 1 || target->ne[upper] != 1) {
            return false;
        }
    }
    const size_t span = source->nb[axis];
    const auto * source_data = static_cast<const uint8_t *>(source->data) + size_t(expert)*span;
    std::memcpy(static_cast<uint8_t *>(target->data) + size_t(slot)*target->nb[axis], source_data, span);
    bytes += span;
    return true;
}

bool copy_projection(
        const llm_expert_projection_descriptor & target,
        const llm_expert_projection_descriptor & source,
        int32_t n_expert,
        uint32_t capacity,
        int32_t expert,
        uint32_t slot,
        size_t & bytes,
        size_t & copies,
        size_t fail_after) {
    const std::array<std::pair<ggml_tensor *, const ggml_tensor *>, 3> tensors = {{
        { target.weight, source.weight }, { target.bias, source.bias }, { target.scale, source.scale },
    }};
    for (size_t index = 0; index < tensors.size(); ++index) {
        if (tensors[index].first == nullptr && tensors[index].second == nullptr) {
            continue;
        }
        if (copies >= fail_after || !copy_tensor(tensors[index].first, tensors[index].second,
                n_expert, capacity, expert, slot, index == 0, bytes)) {
            return false;
        }
        copies++;
    }
    return true;
}

} // namespace

struct llm_cold_expert_cache::impl {
    struct forward_entry { int32_t slot = -1; uint64_t generation = 0; };

    explicit impl(llm_cold_cache_config config) : config(config) {
        if (config.byte_budget == 0 || config.minimum_slots == 0 || config.routed_layer_count == 0 ||
            config.total_expert_keys == 0 || config.minimum_slots > config.total_expert_keys ||
            config.total_expert_keys % config.routed_layer_count != 0) {
            throw std::invalid_argument("invalid cold-cache budget or topology");
        }
        n_expert = config.total_expert_keys/config.routed_layer_count;
        if (this->config.cache_policy_config.digest == 0) {
            const auto copied = llm_expert_cache_policy_copy_config(
                nullptr, llm_expert_cache_policy_tier::cold, this->config.cache_policy_config);
            if (!copied.is_ready()) throw std::invalid_argument("invalid cold-cache policy configuration");
        }
        if (this->config.routed_layers.empty()) {
            this->config.routed_layers.resize(config.routed_layer_count);
            for (uint32_t layer = 0; layer < config.routed_layer_count; ++layer) {
                this->config.routed_layers[layer] = int32_t(layer);
            }
        }
        if (this->config.routed_layers.size() != config.routed_layer_count ||
            this->config.policy_trace_capacity == 0) {
            throw std::invalid_argument("invalid cold-cache policy topology");
        }
    }

    size_t forward_index(llm_expert_key key) const {
        return size_t(key.layer)*n_expert + uint32_t(key.expert);
    }

    llm_expert_layout_class_id class_for_key(llm_expert_key key) const {
        if (!arena || key.layer < 0 || size_t(key.layer) >= arena->layer_ids.size()) {
            return LLM_EXPERT_LAYOUT_CLASS_INVALID;
        }
        return arena->layer_ids[size_t(key.layer)];
    }

    llm_expert_cache_policy_key policy_key(llm_expert_key key) const {
        return { key.layer, key.expert, class_for_key(key) };
    }

    uint64_t payload_for_key(llm_expert_key key) const {
        const auto class_id = class_for_key(key);
        return arena && class_id < arena->payload_bytes.size() ? arena->payload_bytes[class_id] : 0;
    }

    const llm_expert_bundle_descriptor * bundle_for_key(llm_expert_key key) const {
        const auto class_id = class_for_key(key);
        return arena && class_id < arena->bundles.size() ? &arena->bundles[class_id] : nullptr;
    }

    bool valid_reference(llm_cold_reference reference) const {
        return reference.slot < slots.size() && slots[reference.slot].generation == reference.generation &&
            slots[reference.slot].layout_class_id == reference.layout_class_id &&
            slots[reference.slot].state == llm_cold_slot_state::ready;
    }

    bool no_refs(const llm_cold_cache_diagnostics::slot & slot) const {
        return slot.hot_refs == 0 && slot.transfer_refs == 0 && slot.request_refs == 0 &&
            slot.cpu_execution_refs == 0;
    }

    uint64_t occupancy() const noexcept {
        return std::count_if(slots.begin(), slots.end(), [](const auto & slot) {
            return slot.state == llm_cold_slot_state::ready;
        });
    }

    void clear_forward(uint32_t slot) {
        const auto & entry = slots[slot];
        if (!entry.key.is_valid(LLAMA_MAX_LAYERS, n_expert)) {
            return;
        }
        auto & forward = directory[forward_index(entry.key)];
        if (forward.slot == int32_t(slot) && forward.generation == entry.generation) {
            forward = {};
        }
    }

    llm_cold_cache_config config;
    uint32_t n_expert = 0;
    mutable std::mutex mutex;
    std::condition_variable ready_cv;
    std::shared_ptr<cold_allocation> arena;
    std::vector<forward_entry> directory;
    std::vector<llm_cold_cache_diagnostics::slot> slots;
    llm_cold_cache_diagnostics counters;
    uint64_t use_clock = 0;
    llm_expert_cache_policy policy;
    std::vector<llm_expert_cache_policy_candidate> policy_candidates;
    bool policy_request_active = false;
    llm_expert_cache_policy_phase policy_phase = llm_expert_cache_policy_phase::prefill;

    llm_expert_provider_result commit_ready_policy_slots() {
        while (true) {
            int32_t selected = -1;
            for (uint32_t index = 0; index < slots.size(); ++index) {
                const auto & candidate = slots[index];
                if (candidate.state != llm_cold_slot_state::loading ||
                    !policy.validate_resident(index, candidate.generation,
                        policy_key(candidate.key))) {
                    continue;
                }
                if (candidate.origin_operation_ordinal == 0) {
                    counters.invariant_failures++;
                    return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
                }
                if (selected < 0 || candidate.origin_operation_ordinal <
                        slots[uint32_t(selected)].origin_operation_ordinal) {
                    selected = int32_t(index);
                }
            }
            if (selected < 0) break;
            const uint32_t index = uint32_t(selected);
            auto & slot = slots[index];
            auto & forward = directory[forward_index(slot.key)];
            if (forward.slot >= 0) {
                counters.invariant_failures++;
                return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
            }
            slot.state = llm_cold_slot_state::ready;
            slot.last_use = ++use_clock;
            forward = { int32_t(index), slot.generation };
            counters.admissions++;
            counters.publications++;
            LLM_EXPERT_TRACE_INSTANT("k3.cache.cold", "admission", "layer", slot.key.layer,
                "original_expert_id", slot.key.expert, "slot_id", index,
                "generation", slot.generation);
            LLM_EXPERT_TRACE_COUNTER("k3.resource", "cold_cache_occupancy", 5,
                occupancy());
        }
        return llm_expert_provider_result::success();
    }

    llm_expert_provider_result ensure_policy_request() {
        if (policy_request_active) return llm_expert_provider_result::success();
        const auto result = policy_result(policy.request_begin());
        if (result.is_ready()) {
            policy_request_active = true;
            policy_phase = llm_expert_cache_policy_phase::prefill;
        }
        return result;
    }

    llm_expert_provider_result observe_demand(llm_expert_key key, uint64_t occurrence_count = 1) {
        auto result = ensure_policy_request();
        if (!result.is_ready()) return result;
        return policy_result(policy.demand(policy_key(key), occurrence_count,
            payload_for_key(key), counters.aligned_slot_footprint));
    }

    llm_expert_provider_result select_slot(llm_expert_key key, int32_t & selected) {
        auto capacity = policy_result(policy.validate_event_capacity(4));
        if (!capacity.is_ready()) return capacity;
        for (uint32_t index = 0; index < slots.size(); ++index) {
            const auto & slot = slots[index];
            auto & candidate = policy_candidates[index];
            candidate = { index, slot.generation, policy_key(slot.key),
                slot.state == llm_cold_slot_state::free ? payload_for_key(key) : payload_for_key(slot.key),
                counters.aligned_slot_footprint,
                slot.state == llm_cold_slot_state::free,
                slot.state == llm_cold_slot_state::ready && no_refs(slot) };
        }
        llm_expert_cache_policy_decision decision;
        auto result = policy_result(policy.select(policy_key(key),
            policy_candidates.data(), policy_candidates.size(), decision));
        if (!result.is_ready()) return result;
        if (decision.slot >= slots.size()) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
        }
        const auto & slot = slots[decision.slot];
        const bool valid = decision.free ?
            slot.state == llm_cold_slot_state::free && slot.generation == decision.generation :
            slot.state == llm_cold_slot_state::ready && no_refs(slot) &&
                slot.generation == decision.generation &&
                policy.validate_resident(decision.slot, decision.generation,
                    policy_key(slot.key));
        if (!valid) {
            counters.invariant_failures++;
            return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
        }
        selected = int32_t(decision.slot);
        return llm_expert_provider_result::success();
    }
};

llm_cold_expert_cache::llm_cold_expert_cache(llm_cold_cache_config config) :
    pimpl(std::make_unique<impl>(config)) {}

llm_cold_expert_cache::~llm_cold_expert_cache() = default;
llm_cold_expert_cache::llm_cold_expert_cache(llm_cold_expert_cache &&) noexcept = default;
llm_cold_expert_cache & llm_cold_expert_cache::operator=(llm_cold_expert_cache &&) noexcept = default;

llm_expert_provider_result llm_cold_expert_cache::calculate_slot_footprint(
        const llm_expert_bundle_descriptor & prototype,
        ggml_backend_buffer_type_t buffer_type,
        uint64_t & footprint) noexcept {
    footprint = 0;
    if (!prototype.validate().is_ready() || buffer_type == nullptr) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::invalid_descriptor);
    }
    try {
        const size_t size = allocation_size(prototype, 1, buffer_type);
        if (size == 0) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::unsupported_configuration);
        }
        footprint = size;
        return llm_expert_provider_result::success();
    } catch (const std::bad_alloc &) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::allocation_failed);
    } catch (...) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::initialization_failed);
    }
}

llm_expert_provider_result llm_cold_expert_cache::initialize(
        const llm_expert_bundle_descriptor & prototype) noexcept {
    llm_expert_layout_registry registry;
    uint64_t payload = 0;
    if (!prototype.validate().is_ready() || !bundle_payload(prototype, payload)) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::invalid_descriptor);
    }
    registry.classes.push_back({ 0, 0, payload, prototype });
    registry.layer_ids.assign(LLAMA_MAX_LAYERS, LLM_EXPERT_LAYOUT_CLASS_INVALID);
    for (int32_t layer : pimpl->config.routed_layers) {
        if (layer < 0 || layer >= LLAMA_MAX_LAYERS) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::invalid_descriptor);
        }
        registry.layer_ids[size_t(layer)] = 0;
    }
    return initialize(registry);
}

llm_expert_provider_result llm_cold_expert_cache::initialize(
        const llm_expert_layout_registry & registry) noexcept {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    if (pimpl->arena) {
        return llm_expert_provider_result::success();
    }
    if (!registry.sealed() || registry.classes.size() > LLM_EXPERT_LAYOUT_CLASS_MAX ||
        registry.layer_ids.size() != LLAMA_MAX_LAYERS) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::invalid_descriptor);
    }
    try {
        uint64_t payload = 0;
        for (size_t index = 0; index < registry.classes.size(); ++index) {
            const auto & layout_class = registry.classes[index];
            uint64_t actual_payload = 0;
            if (layout_class.id != index || !layout_class.prototype.validate().is_ready() ||
                !bundle_payload(layout_class.prototype, actual_payload) ||
                actual_payload != layout_class.payload_bytes) {
                return llm_expert_provider_result::failure(llm_expert_provider_error::invalid_descriptor);
            }
            payload = std::max(payload, actual_payload);
        }
        for (int32_t layer : pimpl->config.routed_layers) {
            if (layer < 0 || layer >= LLAMA_MAX_LAYERS ||
                registry.layer_ids[size_t(layer)] >= registry.classes.size()) {
                return llm_expert_provider_result::failure(llm_expert_provider_error::invalid_descriptor);
            }
        }
        uint32_t low = 0;
        size_t stride = 0;
        std::shared_ptr<cold_allocation> candidate;
        if (registry.classes.size() == 1) {
            uint32_t high = pimpl->config.total_expert_keys;
            while (low < high) {
                const uint32_t middle = low + (high - low + 1)/2;
                if (allocation_size(registry.classes.front().prototype, middle,
                        pimpl->config.buffer_type) <= pimpl->config.byte_budget) {
                    low = middle;
                } else {
                    high = middle - 1;
                }
            }
        } else {
            stride = universal_slot_stride(registry, pimpl->config.buffer_type);
            if (stride == SIZE_MAX || stride == 0) {
                return llm_expert_provider_result::failure(llm_expert_provider_error::unsupported_configuration);
            }
            const uint64_t available_slots = pimpl->config.byte_budget/stride;
            low = uint32_t(std::min<uint64_t>(available_slots, pimpl->config.total_expert_keys));
        }
        if (low < pimpl->config.minimum_slots) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::unsupported_configuration);
        }
        if (pimpl->config.cache_policy_config.scope == LLAMA_EXPERT_CACHE_POLICY_SCOPE_PER_LAYER &&
            uint64_t(low) < uint64_t(pimpl->config.routed_layer_count)*pimpl->config.minimum_domain_slots) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::unsupported_configuration);
        }
        if (registry.classes.size() == 1) {
            candidate = make_allocation(
                registry.classes.front().prototype, low, true, pimpl->config.buffer_type);
            if (candidate && candidate->buffer) {
                candidate->bundles = { candidate->bundle };
                candidate->layer_ids = registry.layer_ids;
                candidate->payload_bytes = { registry.classes.front().payload_bytes };
                const uint64_t actual = ggml_backend_buffer_get_size(candidate->buffer.get());
                candidate->slot_stride = actual/low + (actual % low != 0);
                stride = candidate->slot_stride;
            }
        } else {
            candidate = make_universal_allocation(registry, low, pimpl->config.buffer_type);
        }
        if (!candidate || !candidate->buffer) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::allocation_failed);
        }
        const uint64_t actual = ggml_backend_buffer_get_size(candidate->buffer.get());
        if (actual > pimpl->config.byte_budget) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::allocation_failed);
        }
        pimpl->directory.assign(size_t(LLAMA_MAX_LAYERS)*pimpl->n_expert, {});
        pimpl->slots.assign(low, {});
        for (auto & slot : pimpl->slots) {
            slot.generation = pimpl->config.initial_slot_generation_for_testing;
        }
        pimpl->counters.requested_bytes = pimpl->config.byte_budget;
        pimpl->counters.actual_bytes = actual;
        pimpl->counters.unused_budget_bytes = pimpl->config.byte_budget - actual;
        pimpl->counters.bundle_payload_bytes = payload;
        pimpl->counters.aligned_slot_footprint = stride;
        pimpl->counters.alignment = ggml_backend_buft_get_alignment(
            pimpl->config.buffer_type ? pimpl->config.buffer_type : ggml_backend_cpu_buffer_type());
        pimpl->counters.effective_slots = low;
        pimpl->counters.pageable = ggml_backend_buffer_is_host(candidate->buffer.get());
        pimpl->policy_candidates.assign(low, {});
        const auto policy_initialized = pimpl->policy.initialize(
            pimpl->config.cache_policy_config,
            llm_expert_cache_policy_tier::cold,
            pimpl->config.routed_layers.data(),
            pimpl->config.routed_layer_count,
            pimpl->n_expert,
            pimpl->config.minimum_domain_slots,
            low,
            pimpl->counters.aligned_slot_footprint,
            pimpl->config.policy_trace_capacity);
        if (!policy_initialized.is_ready()) {
            return policy_result(policy_initialized);
        }
        pimpl->arena = std::move(candidate);
        return llm_expert_provider_result::success();
    } catch (const std::bad_alloc &) {
        pimpl->arena.reset();
        pimpl->directory.clear();
        pimpl->slots.clear();
        return llm_expert_provider_result::failure(llm_expert_provider_error::allocation_failed);
    } catch (...) {
        pimpl->arena.reset();
        pimpl->directory.clear();
        pimpl->slots.clear();
        return llm_expert_provider_result::failure(llm_expert_provider_error::initialization_failed);
    }
}

llm_expert_provider_result llm_cold_expert_cache::find_or_admit_with_loader(
        llm_expert_key key,
        llm_cold_reference & reference,
        llm_cold_cache_loader loader,
        void * loader_data) noexcept {
    if (loader == nullptr) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::invalid_key);
    }
    bool hit = false;
    auto reserved = reserve_or_find(key, reference, hit);
    if (!reserved.is_ready() || hit) {
        return reserved;
    }
    const auto * destination = pimpl->bundle_for_key(key);
    if (destination == nullptr) {
        (void) fail_reservation(key, reference);
        return llm_expert_provider_result::failure(llm_expert_provider_error::invalid_descriptor);
    }
    const auto result = loader(loader_data, key, *destination, reference.slot);
    if (!result.is_ready()) {
        (void) fail_reservation(key, reference);
        return result;
    }
    const auto published = publish_ready(key, reference);
    if (!published.is_ready()) (void) fail_reservation(key, reference);
    return published;
}

llm_expert_provider_result llm_cold_expert_cache::find_or_admit_speculative_with_loader(
        llm_expert_key key,
        uint64_t deadline,
        uint64_t utility,
        llm_cold_reference & reference,
        llm_cold_cache_loader loader,
        void * loader_data) noexcept {
    if (loader == nullptr) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::invalid_key);
    }
    bool hit = false;
    auto reserved = reserve_or_find_speculative(key, deadline, utility, reference, hit);
    if (!reserved.is_ready() || hit) return reserved;
    const auto * destination = pimpl->bundle_for_key(key);
    if (destination == nullptr) {
        (void) fail_reservation(key, reference);
        return llm_expert_provider_result::failure(llm_expert_provider_error::invalid_descriptor);
    }
    const auto loaded = loader(loader_data, key, *destination, reference.slot);
    if (!loaded.is_ready()) {
        (void) fail_reservation(key, reference);
        return loaded;
    }
    const auto published = publish_ready(key, reference);
    if (!published.is_ready()) {
        (void) fail_reservation(key, reference);
        return published;
    }
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    pimpl->counters.speculative_admissions++;
    return llm_expert_provider_result::success();
}

llm_expert_provider_result llm_cold_expert_cache::reserve_or_find(
        llm_expert_key key,
        llm_cold_reference & reference,
        bool & hit) noexcept {
    LLM_EXPERT_TRACE_SCOPE("k3.cache.cold", "lookup", "layer", key.layer, "original_expert_id", key.expert);
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    pimpl->counters.requests++;
    hit = false;
    if (!pimpl->arena || !key.is_valid(LLAMA_MAX_LAYERS, pimpl->n_expert)) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::invalid_key);
    }
    auto observed = pimpl->observe_demand(key);
    if (!observed.is_ready()) return observed;
    auto & forward = pimpl->directory[pimpl->forward_index(key)];
    if (forward.slot >= 0) {
        if (uint32_t(forward.slot) >= pimpl->slots.size()) {
            pimpl->counters.invariant_failures++;
            return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
        }
        auto & existing = pimpl->slots[forward.slot];
        if (!key_matches(existing.key, key) || existing.generation != forward.generation ||
            existing.state != llm_cold_slot_state::ready) {
            pimpl->counters.invariant_failures++;
            return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
        }
        existing.last_use = ++pimpl->use_clock;
        const auto touched = policy_result(pimpl->policy.hit(uint32_t(forward.slot), forward.generation));
        if (!touched.is_ready()) return touched;
        if (existing.origin == llm_expert_residency_origin::speculative) {
            existing.origin = llm_expert_residency_origin::demand;
            existing.speculative_consumed = true;
            existing.speculative_deadline = 0;
            existing.speculative_utility = 0;
            pimpl->counters.speculative_demand_consumptions++;
        }
        reference = { uint32_t(forward.slot), forward.generation, existing.layout_class_id };
        pimpl->counters.hits++;
        hit = true;
        LLM_EXPERT_TRACE_INSTANT("k3.cache.cold", "hit", "layer", key.layer,
            "original_expert_id", key.expert, "slot_id", reference.slot, "generation", reference.generation);
        return llm_expert_provider_result::success();
    }
    pimpl->counters.misses++;
    LLM_EXPERT_TRACE_INSTANT("k3.cache.cold", "miss", "layer", key.layer, "original_expert_id", key.expert);
    int32_t victim = -1;
    auto selected = pimpl->select_slot(key, victim);
    if (!selected.is_ready()) return selected;
    auto & slot = pimpl->slots[victim];
    if (slot.generation == std::numeric_limits<uint64_t>::max()) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::generation_exhausted);
    }
    if (slot.state == llm_cold_slot_state::ready) {
        LLM_EXPERT_TRACE_INSTANT("k3.cache.cold", "victim", "layer", slot.key.layer,
            "original_expert_id", slot.key.expert, "slot_id", uint32_t(victim),
            "generation", slot.generation);
        auto removed = policy_result(pimpl->policy.evict(uint32_t(victim), slot.generation));
        if (!removed.is_ready()) return removed;
        slot.state = llm_cold_slot_state::evicting;
        pimpl->clear_forward(victim);
        pimpl->counters.evictions++;
        LLM_EXPERT_TRACE_INSTANT("k3.cache.cold", "eviction", "layer", slot.key.layer,
            "original_expert_id", slot.key.expert, "slot_id", uint32_t(victim),
            "generation", slot.generation);
        LLM_EXPERT_TRACE_COUNTER("k3.resource", "cold_cache_occupancy", 5,
            pimpl->occupancy());
    }
    slot.state = llm_cold_slot_state::reserved;
    slot.key = key;
    slot.layout_class_id = pimpl->class_for_key(key);
    if (slot.layout_class_id == LLM_EXPERT_LAYOUT_CLASS_INVALID) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::invalid_descriptor);
    }
    slot.generation++;
    slot.last_use = 0;
    slot.hot_refs = slot.transfer_refs = slot.request_refs = slot.cpu_execution_refs = 0;
    slot.origin = llm_expert_residency_origin::demand;
    slot.speculative_consumed = false;
    slot.speculative_deadline = 0;
    slot.speculative_utility = 0;
    pimpl->counters.generation_changes++;
    slot.state = llm_cold_slot_state::loading;
    const auto loading = policy_result(pimpl->policy.load_begin(uint32_t(victim), slot.generation,
        pimpl->policy_key(key), pimpl->payload_for_key(key),
        pimpl->counters.aligned_slot_footprint));
    if (!loading.is_ready()) return loading;
    slot.origin_operation_ordinal = pimpl->policy.diagnostics().operation_ordinal;
    reference = { uint32_t(victim), slot.generation, slot.layout_class_id };
    pimpl->counters.reservations++;
    LLM_EXPERT_TRACE_INSTANT("k3.cache.cold", "reserve", "layer", key.layer,
        "original_expert_id", key.expert, "slot_id", reference.slot, "generation", reference.generation,
        "layout_class_id", reference.layout_class_id);
    return llm_expert_provider_result::success();
}

llm_expert_provider_result llm_cold_expert_cache::reserve_or_join_demand(
        llm_expert_key key,
        llm_cold_reference & reference,
        llm_cold_demand_lookup & lookup) noexcept {
    LLM_EXPERT_TRACE_SCOPE("k3.cache.cold", "demand_lookup", "layer", key.layer,
        "original_expert_id", key.expert);
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    pimpl->counters.requests++;
    lookup = llm_cold_demand_lookup::reserved;
    if (!pimpl->arena || !key.is_valid(LLAMA_MAX_LAYERS, pimpl->n_expert)) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::invalid_key);
    }
    auto observed = pimpl->observe_demand(key);
    if (!observed.is_ready()) return observed;
    auto & forward = pimpl->directory[pimpl->forward_index(key)];
    if (forward.slot >= 0) {
        if (uint32_t(forward.slot) >= pimpl->slots.size()) {
            pimpl->counters.invariant_failures++;
            return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
        }
        auto & existing = pimpl->slots[forward.slot];
        if (!key_matches(existing.key, key) || existing.generation != forward.generation ||
            existing.state != llm_cold_slot_state::ready) {
            pimpl->counters.invariant_failures++;
            return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
        }
        existing.last_use = ++pimpl->use_clock;
        const auto touched = policy_result(pimpl->policy.hit(uint32_t(forward.slot), forward.generation));
        if (!touched.is_ready()) return touched;
        if (existing.origin == llm_expert_residency_origin::speculative) {
            existing.origin = llm_expert_residency_origin::demand;
            existing.speculative_consumed = true;
            existing.speculative_deadline = 0;
            existing.speculative_utility = 0;
            pimpl->counters.speculative_demand_consumptions++;
        }
        reference = { uint32_t(forward.slot), forward.generation, existing.layout_class_id };
        pimpl->counters.hits++;
        lookup = llm_cold_demand_lookup::ready;
        LLM_EXPERT_TRACE_INSTANT("k3.cache.cold", "hit", "layer", key.layer,
            "original_expert_id", key.expert, "slot_id", reference.slot, "generation", reference.generation);
        return llm_expert_provider_result::success();
    }
    for (uint32_t index = 0; index < pimpl->slots.size(); ++index) {
        auto & existing = pimpl->slots[index];
        if (existing.state != llm_cold_slot_state::loading || !key_matches(existing.key, key)) continue;
        if (existing.origin == llm_expert_residency_origin::speculative) {
            existing.origin = llm_expert_residency_origin::demand;
            existing.speculative_consumed = true;
            existing.speculative_deadline = 0;
            existing.speculative_utility = 0;
            pimpl->counters.speculative_demand_consumptions++;
        }
        reference = { index, existing.generation, existing.layout_class_id };
        lookup = llm_cold_demand_lookup::joined_loading;
        LLM_EXPERT_TRACE_INSTANT("k3.cache.cold", "join_loading", "layer", key.layer,
            "original_expert_id", key.expert, "slot_id", reference.slot, "generation", reference.generation);
        return llm_expert_provider_result::success();
    }
    pimpl->counters.misses++;
    int32_t victim = -1;
    auto selected = pimpl->select_slot(key, victim);
    if (!selected.is_ready()) return selected;
    auto & slot = pimpl->slots[victim];
    if (slot.generation == std::numeric_limits<uint64_t>::max()) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::generation_exhausted);
    }
    if (slot.state == llm_cold_slot_state::ready) {
        LLM_EXPERT_TRACE_INSTANT("k3.cache.cold", "victim", "layer", slot.key.layer,
            "original_expert_id", slot.key.expert, "slot_id", uint32_t(victim),
            "generation", slot.generation);
        auto removed = policy_result(pimpl->policy.evict(uint32_t(victim), slot.generation));
        if (!removed.is_ready()) return removed;
        slot.state = llm_cold_slot_state::evicting;
        pimpl->clear_forward(victim);
        pimpl->counters.evictions++;
        LLM_EXPERT_TRACE_INSTANT("k3.cache.cold", "eviction", "layer", slot.key.layer,
            "original_expert_id", slot.key.expert, "slot_id", uint32_t(victim),
            "generation", slot.generation);
        LLM_EXPERT_TRACE_COUNTER("k3.resource", "cold_cache_occupancy", 5,
            pimpl->occupancy());
    }
    slot.state = llm_cold_slot_state::reserved;
    slot.key = key;
    slot.layout_class_id = pimpl->class_for_key(key);
    if (slot.layout_class_id == LLM_EXPERT_LAYOUT_CLASS_INVALID) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::invalid_descriptor);
    }
    slot.generation++;
    slot.last_use = 0;
    slot.hot_refs = slot.transfer_refs = slot.request_refs = slot.cpu_execution_refs = 0;
    slot.origin = llm_expert_residency_origin::demand;
    slot.speculative_consumed = false;
    slot.speculative_deadline = 0;
    slot.speculative_utility = 0;
    pimpl->counters.generation_changes++;
    slot.state = llm_cold_slot_state::loading;
    const auto loading = policy_result(pimpl->policy.load_begin(uint32_t(victim), slot.generation,
        pimpl->policy_key(key), pimpl->payload_for_key(key),
        pimpl->counters.aligned_slot_footprint));
    if (!loading.is_ready()) return loading;
    slot.origin_operation_ordinal = pimpl->policy.diagnostics().operation_ordinal;
    reference = { uint32_t(victim), slot.generation, slot.layout_class_id };
    pimpl->counters.reservations++;
    LLM_EXPERT_TRACE_INSTANT("k3.cache.cold", "reserve", "layer", key.layer,
        "original_expert_id", key.expert, "slot_id", reference.slot, "generation", reference.generation,
        "layout_class_id", reference.layout_class_id);
    return llm_expert_provider_result::success();
}

llm_expert_provider_result llm_cold_expert_cache::lookup_demand(
        llm_expert_key key,
        llm_cold_reference & reference,
        llm_cold_demand_lookup & lookup) noexcept {
    LLM_EXPERT_TRACE_SCOPE("k3.cache.cold", "demand_lookup_only", "layer", key.layer,
        "original_expert_id", key.expert);
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    pimpl->counters.requests++;
    lookup = llm_cold_demand_lookup::missing;
    reference = {};
    if (!pimpl->arena || !key.is_valid(LLAMA_MAX_LAYERS, pimpl->n_expert)) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::invalid_key);
    }
    auto observed = pimpl->observe_demand(key);
    if (!observed.is_ready()) return observed;
    auto & forward = pimpl->directory[pimpl->forward_index(key)];
    if (forward.slot >= 0) {
        if (uint32_t(forward.slot) >= pimpl->slots.size()) {
            pimpl->counters.invariant_failures++;
            return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
        }
        auto & existing = pimpl->slots[forward.slot];
        if (!key_matches(existing.key, key) || existing.generation != forward.generation ||
            existing.state != llm_cold_slot_state::ready) {
            pimpl->counters.invariant_failures++;
            return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
        }
        existing.last_use = ++pimpl->use_clock;
        const auto touched = policy_result(pimpl->policy.hit(uint32_t(forward.slot), forward.generation));
        if (!touched.is_ready()) return touched;
        if (existing.origin == llm_expert_residency_origin::speculative) {
            existing.origin = llm_expert_residency_origin::demand;
            existing.speculative_consumed = true;
            existing.speculative_deadline = 0;
            existing.speculative_utility = 0;
            pimpl->counters.speculative_demand_consumptions++;
        }
        reference = { uint32_t(forward.slot), forward.generation, existing.layout_class_id };
        pimpl->counters.hits++;
        lookup = llm_cold_demand_lookup::ready;
        LLM_EXPERT_TRACE_INSTANT("k3.cache.cold", "hit", "layer", key.layer,
            "original_expert_id", key.expert, "slot_id", reference.slot, "generation", reference.generation);
        return llm_expert_provider_result::success();
    }
    for (uint32_t index = 0; index < pimpl->slots.size(); ++index) {
        auto & existing = pimpl->slots[index];
        if (existing.state != llm_cold_slot_state::loading || !key_matches(existing.key, key)) continue;
        if (existing.origin == llm_expert_residency_origin::speculative) {
            existing.origin = llm_expert_residency_origin::demand;
            existing.speculative_consumed = true;
            existing.speculative_deadline = 0;
            existing.speculative_utility = 0;
            pimpl->counters.speculative_demand_consumptions++;
        }
        reference = { index, existing.generation, existing.layout_class_id };
        lookup = llm_cold_demand_lookup::joined_loading;
        LLM_EXPERT_TRACE_INSTANT("k3.cache.cold", "join_loading", "layer", key.layer,
            "original_expert_id", key.expert, "slot_id", reference.slot, "generation", reference.generation);
        return llm_expert_provider_result::success();
    }
    pimpl->counters.misses++;
    LLM_EXPERT_TRACE_INSTANT("k3.cache.cold", "miss_without_admission", "layer", key.layer,
        "original_expert_id", key.expert);
    return llm_expert_provider_result::success();
}

llm_expert_provider_result llm_cold_expert_cache::wait_until_ready(
        llm_cold_reference reference) noexcept {
    std::unique_lock<std::mutex> lock(pimpl->mutex);
    if (reference.slot >= pimpl->slots.size() ||
        pimpl->slots[reference.slot].generation != reference.generation ||
        pimpl->slots[reference.slot].layout_class_id != reference.layout_class_id) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::stale_generation);
    }
    pimpl->ready_cv.wait(lock, [&] {
        if (reference.slot >= pimpl->slots.size()) return true;
        const auto & slot = pimpl->slots[reference.slot];
        return slot.generation != reference.generation || slot.layout_class_id != reference.layout_class_id ||
            slot.state == llm_cold_slot_state::ready ||
            slot.state == llm_cold_slot_state::failed ||
            slot.state == llm_cold_slot_state::free;
    });
    if (reference.slot >= pimpl->slots.size()) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::stale_generation);
    }
    const auto & slot = pimpl->slots[reference.slot];
    if (slot.generation != reference.generation || slot.layout_class_id != reference.layout_class_id) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::stale_generation);
    }
    return slot.state == llm_cold_slot_state::ready ?
        llm_expert_provider_result::success() :
        llm_expert_provider_result::failure(llm_expert_provider_error::copy_failed);
}

llm_expert_provider_result llm_cold_expert_cache::reserve_or_find_speculative(
        llm_expert_key key,
        uint64_t deadline,
        uint64_t utility,
        llm_cold_reference & reference,
        bool & hit) noexcept {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    pimpl->counters.requests++;
    hit = false;
    if (!pimpl->arena || deadline == 0 || !key.is_valid(LLAMA_MAX_LAYERS, pimpl->n_expert)) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::invalid_key);
    }
    auto policy_request = pimpl->ensure_policy_request();
    if (!policy_request.is_ready()) return policy_request;
    auto & forward = pimpl->directory[pimpl->forward_index(key)];
    if (forward.slot >= 0) {
        if (uint32_t(forward.slot) >= pimpl->slots.size()) {
            pimpl->counters.invariant_failures++;
            return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
        }
        const auto & existing = pimpl->slots[forward.slot];
        if (!key_matches(existing.key, key) || existing.generation != forward.generation ||
            existing.state != llm_cold_slot_state::ready) {
            pimpl->counters.invariant_failures++;
            return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
        }
        reference = { uint32_t(forward.slot), forward.generation, existing.layout_class_id };
        pimpl->counters.hits++;
        hit = true;
        return llm_expert_provider_result::success();
    }
    pimpl->counters.misses++;
    int32_t speculative_victim = -1;
    for (uint32_t index = 0; index < pimpl->slots.size(); ++index) {
        const auto & candidate = pimpl->slots[index];
        if (candidate.state != llm_cold_slot_state::ready || !pimpl->no_refs(candidate) ||
            candidate.origin != llm_expert_residency_origin::speculative ||
            candidate.speculative_consumed ||
            (pimpl->config.cache_policy_config.scope == LLAMA_EXPERT_CACHE_POLICY_SCOPE_PER_LAYER &&
             candidate.key.layer != key.layer)) continue;
        if (speculative_victim < 0 ||
            candidate.speculative_deadline < pimpl->slots[speculative_victim].speculative_deadline ||
            (candidate.speculative_deadline == pimpl->slots[speculative_victim].speculative_deadline &&
             (candidate.speculative_utility < pimpl->slots[speculative_victim].speculative_utility ||
              (candidate.speculative_utility == pimpl->slots[speculative_victim].speculative_utility &&
               index < uint32_t(speculative_victim))))) {
            speculative_victim = int32_t(index);
        }
    }
    for (uint32_t index = 0; index < pimpl->slots.size(); ++index) {
        const auto & slot = pimpl->slots[index];
        pimpl->policy_candidates[index] = {
            index, slot.generation, pimpl->policy_key(slot.key),
            slot.state == llm_cold_slot_state::free ? pimpl->payload_for_key(key) : pimpl->payload_for_key(slot.key),
            pimpl->counters.aligned_slot_footprint,
            slot.state == llm_cold_slot_state::free,
            speculative_victim == int32_t(index),
        };
    }
    llm_expert_cache_policy_decision decision;
    auto selected = policy_result(pimpl->policy.optional_admission(
        pimpl->policy_key(key), llm_expert_cache_policy_admission::optional_background,
        pimpl->policy_candidates.data(), pimpl->policy_candidates.size(), decision));
    if (!selected.is_ready() || !decision.accept || decision.slot >= pimpl->slots.size()) {
        pimpl->counters.speculative_rejections++;
        return selected.is_ready() ?
            llm_expert_provider_result::failure(llm_expert_provider_error::busy) : selected;
    }
    auto & slot = pimpl->slots[decision.slot];
    const bool valid = decision.free ?
        slot.state == llm_cold_slot_state::free && slot.generation == decision.generation :
        int32_t(decision.slot) == speculative_victim &&
            slot.state == llm_cold_slot_state::ready && pimpl->no_refs(slot) &&
            slot.origin == llm_expert_residency_origin::speculative &&
            !slot.speculative_consumed && slot.generation == decision.generation;
    if (!valid) {
        pimpl->counters.invariant_failures++;
        return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
    }
    if (!decision.free) {
        LLM_EXPERT_TRACE_INSTANT("k3.cache.cold", "victim", "layer", slot.key.layer,
            "original_expert_id", slot.key.expert, "slot_id", decision.slot,
            "generation", slot.generation);
        auto removed = policy_result(pimpl->policy.evict(decision.slot, slot.generation));
        if (!removed.is_ready()) return removed;
        slot.state = llm_cold_slot_state::evicting;
        pimpl->clear_forward(decision.slot);
        pimpl->counters.evictions++;
        pimpl->counters.speculative_replacements++;
        LLM_EXPERT_TRACE_INSTANT("k3.cache.cold", "eviction", "layer", slot.key.layer,
            "original_expert_id", slot.key.expert, "slot_id", decision.slot,
            "generation", slot.generation);
        LLM_EXPERT_TRACE_COUNTER("k3.resource", "cold_cache_occupancy", 5,
            pimpl->occupancy());
    }
    if (slot.generation == UINT64_MAX) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::generation_exhausted);
    }
    slot.state = llm_cold_slot_state::reserved;
    slot.key = key;
    slot.layout_class_id = pimpl->class_for_key(key);
    if (slot.layout_class_id == LLM_EXPERT_LAYOUT_CLASS_INVALID) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::invalid_descriptor);
    }
    slot.generation++;
    slot.last_use = 0;
    slot.hot_refs = slot.transfer_refs = slot.request_refs = slot.cpu_execution_refs = 0;
    slot.origin = llm_expert_residency_origin::speculative;
    slot.speculative_consumed = false;
    slot.speculative_deadline = deadline;
    slot.speculative_utility = utility;
    pimpl->counters.generation_changes++;
    slot.state = llm_cold_slot_state::loading;
    const auto loading = policy_result(pimpl->policy.load_begin(decision.slot, slot.generation,
        pimpl->policy_key(key), pimpl->payload_for_key(key),
        pimpl->counters.aligned_slot_footprint, false));
    if (!loading.is_ready()) return loading;
    slot.origin_operation_ordinal = pimpl->policy.diagnostics().operation_ordinal;
    pimpl->counters.reservations++;
    reference = { decision.slot, slot.generation, slot.layout_class_id };
    return llm_expert_provider_result::success();
}

llm_expert_provider_result llm_cold_expert_cache::reclassify(
        llm_cold_reference reference,
        llm_expert_residency_origin origin,
        bool consumed) noexcept {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    if (!pimpl->valid_reference(reference) || origin == llm_expert_residency_origin::speculative) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::stale_generation);
    }
    auto & slot = pimpl->slots[reference.slot];
    slot.origin = origin;
    slot.speculative_consumed = consumed;
    slot.speculative_deadline = 0;
    slot.speculative_utility = 0;
    return llm_expert_provider_result::success();
}

llm_expert_provider_result llm_cold_expert_cache::publish_ready(
        llm_expert_key key, llm_cold_reference reference) noexcept {
    LLM_EXPERT_TRACE_SCOPE("k3.cache.cold", "publish", "layer", key.layer,
        "original_expert_id", key.expert, "slot_id", reference.slot, "generation", reference.generation);
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    if (reference.slot >= pimpl->slots.size()) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::stale_generation);
    }
    auto & slot = pimpl->slots[reference.slot];
    if (slot.generation != reference.generation || slot.layout_class_id != reference.layout_class_id ||
        slot.state != llm_cold_slot_state::loading ||
        !key_matches(slot.key, key)) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::stale_generation);
    }
    auto & forward = pimpl->directory[pimpl->forward_index(key)];
    if (forward.slot >= 0) {
        pimpl->counters.invariant_failures++;
        return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
    }
    const auto completed = policy_result(pimpl->policy.load_complete(reference.slot, reference.generation));
    if (!completed.is_ready()) return completed;
    const auto result = pimpl->commit_ready_policy_slots();
    pimpl->ready_cv.notify_all();
    return result;
}

llm_expert_provider_result llm_cold_expert_cache::fail_reservation(
        llm_expert_key key, llm_cold_reference reference) noexcept {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    if (reference.slot >= pimpl->slots.size()) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::stale_generation);
    }
    auto & slot = pimpl->slots[reference.slot];
    if (slot.generation != reference.generation || slot.layout_class_id != reference.layout_class_id ||
        slot.state != llm_cold_slot_state::loading ||
        !key_matches(slot.key, key)) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::stale_generation);
    }
    slot.state = llm_cold_slot_state::failed;
    pimpl->counters.failed_copies++;
    pimpl->counters.failed_reservations++;
    const auto failed = policy_result(pimpl->policy.load_failed(reference.slot, reference.generation));
    if (!failed.is_ready()) return failed;
    const auto result = pimpl->commit_ready_policy_slots();
    pimpl->ready_cv.notify_all();
    return result;
}

llm_expert_provider_result llm_cold_expert_cache::find_or_admit(
        llm_expert_key key,
        const llm_expert_bundle_descriptor & source,
        llm_cold_reference & reference,
        size_t fail_copy_after_tensors) noexcept {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    pimpl->counters.requests++;
    if (!pimpl->arena || !key.is_valid(LLAMA_MAX_LAYERS, pimpl->n_expert) || source.layer != key.layer ||
        source.n_expert != int32_t(pimpl->n_expert) || !source.validate().is_ready() ||
        !bundle_is_host_accessible(source)) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::invalid_key);
    }
    auto observed = pimpl->observe_demand(key);
    if (!observed.is_ready()) return observed;
    auto & forward = pimpl->directory[pimpl->forward_index(key)];
    if (forward.slot >= 0) {
        if (uint32_t(forward.slot) >= pimpl->slots.size()) {
            pimpl->counters.invariant_failures++;
            return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
        }
        auto & slot = pimpl->slots[forward.slot];
        if (!key_matches(slot.key, key) || slot.generation != forward.generation ||
            slot.state != llm_cold_slot_state::ready) {
            pimpl->counters.invariant_failures++;
            return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
        }
        slot.last_use = ++pimpl->use_clock;
        const auto touched = policy_result(pimpl->policy.hit(uint32_t(forward.slot), forward.generation));
        if (!touched.is_ready()) return touched;
        if (slot.origin == llm_expert_residency_origin::speculative) {
            slot.origin = llm_expert_residency_origin::demand;
            slot.speculative_consumed = true;
            slot.speculative_deadline = 0;
            slot.speculative_utility = 0;
            pimpl->counters.speculative_demand_consumptions++;
        }
        reference = { uint32_t(forward.slot), forward.generation, slot.layout_class_id };
        pimpl->counters.hits++;
        return llm_expert_provider_result::success();
    }
    pimpl->counters.misses++;
    int32_t victim = -1;
    auto selected = pimpl->select_slot(key, victim);
    if (!selected.is_ready()) return selected;
    auto & slot = pimpl->slots[victim];
    if (slot.generation == std::numeric_limits<uint64_t>::max()) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::generation_exhausted);
    }
    if (slot.state == llm_cold_slot_state::ready) {
        LLM_EXPERT_TRACE_INSTANT("k3.cache.cold", "victim", "layer", slot.key.layer,
            "original_expert_id", slot.key.expert, "slot_id", uint32_t(victim),
            "generation", slot.generation);
        auto removed = policy_result(pimpl->policy.evict(uint32_t(victim), slot.generation));
        if (!removed.is_ready()) return removed;
        slot.state = llm_cold_slot_state::evicting;
        pimpl->clear_forward(victim);
        pimpl->counters.evictions++;
        LLM_EXPERT_TRACE_INSTANT("k3.cache.cold", "eviction", "layer", slot.key.layer,
            "original_expert_id", slot.key.expert, "slot_id", uint32_t(victim),
            "generation", slot.generation);
        LLM_EXPERT_TRACE_COUNTER("k3.resource", "cold_cache_occupancy", 5,
            pimpl->occupancy());
    }
    slot.state = llm_cold_slot_state::reserved;
    slot.key = key;
    slot.layout_class_id = pimpl->class_for_key(key);
    if (slot.layout_class_id == LLM_EXPERT_LAYOUT_CLASS_INVALID) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::invalid_descriptor);
    }
    slot.generation++;
    slot.last_use = 0;
    slot.hot_refs = slot.transfer_refs = slot.request_refs = slot.cpu_execution_refs = 0;
    slot.origin = llm_expert_residency_origin::demand;
    slot.speculative_consumed = false;
    slot.speculative_deadline = 0;
    slot.speculative_utility = 0;
    pimpl->counters.generation_changes++;
    slot.state = llm_cold_slot_state::loading;
    const auto loading = policy_result(pimpl->policy.load_begin(uint32_t(victim), slot.generation,
        pimpl->policy_key(key), pimpl->payload_for_key(key),
        pimpl->counters.aligned_slot_footprint));
    if (!loading.is_ready()) return loading;
    slot.origin_operation_ordinal = pimpl->policy.diagnostics().operation_ordinal;
    reference = { uint32_t(victim), slot.generation, slot.layout_class_id };

    size_t bytes = 0;
    size_t copies = 0;
    const int64_t start = ggml_time_us();
    const auto * target_ptr = pimpl->bundle_for_key(key);
    if (target_ptr == nullptr) {
        slot.state = llm_cold_slot_state::failed;
        (void) pimpl->policy.load_failed(uint32_t(victim), slot.generation);
        return llm_expert_provider_result::failure(llm_expert_provider_error::invalid_descriptor);
    }
    const auto & target = *target_ptr;
    const bool copied =
        copy_projection(target.up, source.up, source.n_expert, target.n_expert, key.expert, victim,
            bytes, copies, fail_copy_after_tensors) &&
        copy_projection(target.gate, source.gate, source.n_expert, target.n_expert, key.expert, victim,
            bytes, copies, fail_copy_after_tensors) &&
        copy_projection(target.gate_up, source.gate_up, source.n_expert, target.n_expert, key.expert, victim,
            bytes, copies, fail_copy_after_tensors) &&
        copy_projection(target.down, source.down, source.n_expert, target.n_expert, key.expert, victim,
            bytes, copies, fail_copy_after_tensors);
    pimpl->counters.source_copy_time_us += ggml_time_us() - start;
    if (!copied) {
        slot.state = llm_cold_slot_state::failed;
        pimpl->counters.failed_copies++;
        const auto failed = policy_result(pimpl->policy.load_failed(uint32_t(victim), slot.generation));
        pimpl->ready_cv.notify_all();
        if (!failed.is_ready()) return failed;
        return llm_expert_provider_result::failure(llm_expert_provider_error::copy_failed);
    }
    const auto completed = policy_result(pimpl->policy.load_complete(uint32_t(victim), slot.generation));
    if (!completed.is_ready()) return completed;
    slot.state = llm_cold_slot_state::ready;
    slot.last_use = ++pimpl->use_clock;
    pimpl->directory[pimpl->forward_index(key)] = { victim, slot.generation };
    pimpl->counters.admissions++;
    LLM_EXPERT_TRACE_INSTANT("k3.cache.cold", "admission", "layer", key.layer,
        "original_expert_id", key.expert, "slot_id", uint32_t(victim),
        "generation", slot.generation);
    LLM_EXPERT_TRACE_COUNTER("k3.resource", "cold_cache_occupancy", 5,
        pimpl->occupancy());
    pimpl->counters.source_copy_bundles++;
    pimpl->counters.source_copy_bytes += bytes;
    reference = { uint32_t(victim), slot.generation, slot.layout_class_id };
    pimpl->ready_cv.notify_all();
    return llm_expert_provider_result::success();
}

llm_expert_provider_result llm_cold_expert_cache::acquire(
        llm_cold_reference reference, llm_cold_reference_kind kind) noexcept {
    LLM_EXPERT_TRACE_INSTANT("k3.cache.cold", "pin", "slot_id", reference.slot,
        "generation", reference.generation, "reference_kind", uint32_t(kind));
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    if (!pimpl->valid_reference(reference)) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::stale_generation);
    }
    auto & slot = pimpl->slots[reference.slot];
    uint32_t * counter = nullptr;
    uint64_t * current = nullptr;
    uint64_t * peak = nullptr;
    if (kind == llm_cold_reference_kind::hot) {
        counter = &slot.hot_refs; current = &pimpl->counters.current_hot_refs; peak = &pimpl->counters.peak_hot_refs;
    } else if (kind == llm_cold_reference_kind::transfer) {
        counter = &slot.transfer_refs; current = &pimpl->counters.current_transfer_refs; peak = &pimpl->counters.peak_transfer_refs;
    } else if (kind == llm_cold_reference_kind::request) {
        counter = &slot.request_refs; current = &pimpl->counters.current_request_refs; peak = &pimpl->counters.peak_request_refs;
    } else {
        counter = &slot.cpu_execution_refs;
        current = &pimpl->counters.current_cpu_execution_refs;
        peak = &pimpl->counters.peak_cpu_execution_refs;
    }
    if (*counter == std::numeric_limits<uint32_t>::max()) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::unsupported_configuration);
    }
    const auto pinned = policy_result(pimpl->policy.pin(reference.slot, reference.generation));
    if (!pinned.is_ready()) return pinned;
    (*counter)++; (*current)++; *peak = std::max(*peak, *current);
    slot.last_use = ++pimpl->use_clock;
    return llm_expert_provider_result::success();
}

llm_expert_provider_result llm_cold_expert_cache::policy_shadow_hit(
        llm_expert_key key,
        llm_cold_reference reference,
        uint64_t occurrence_count) noexcept {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    if (occurrence_count == 0 || !pimpl->valid_reference(reference) ||
        !key.is_valid(LLAMA_MAX_LAYERS, pimpl->n_expert)) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::invalid_key);
    }
    const auto & slot = pimpl->slots[reference.slot];
    if (!key_matches(slot.key, key)) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
    }
    if (!pimpl->policy.validate_resident(
            reference.slot, reference.generation, pimpl->policy_key(key))) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
    }
    auto result = pimpl->ensure_policy_request();
    if (result.is_ready()) result = policy_result(pimpl->policy.validate_event_capacity(2));
    if (result.is_ready()) result = pimpl->observe_demand(key, occurrence_count);
    if (result.is_ready()) {
        result = policy_result(pimpl->policy.hit(reference.slot, reference.generation));
    }
    return result;
}

llm_expert_provider_result llm_cold_expert_cache::release(
        llm_cold_reference reference, llm_cold_reference_kind kind) noexcept {
    LLM_EXPERT_TRACE_INSTANT("k3.cache.cold", "unpin", "slot_id", reference.slot,
        "generation", reference.generation, "reference_kind", uint32_t(kind));
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    if (!pimpl->valid_reference(reference)) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::stale_generation);
    }
    auto & slot = pimpl->slots[reference.slot];
    uint32_t * counter = nullptr;
    uint64_t * current = nullptr;
    if (kind == llm_cold_reference_kind::hot) {
        counter = &slot.hot_refs; current = &pimpl->counters.current_hot_refs;
    } else if (kind == llm_cold_reference_kind::transfer) {
        counter = &slot.transfer_refs; current = &pimpl->counters.current_transfer_refs;
    } else if (kind == llm_cold_reference_kind::request) {
        counter = &slot.request_refs; current = &pimpl->counters.current_request_refs;
    } else {
        counter = &slot.cpu_execution_refs; current = &pimpl->counters.current_cpu_execution_refs;
    }
    if (*counter == 0 || *current == 0) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
    }
    const auto unpinned = policy_result(pimpl->policy.unpin(reference.slot, reference.generation));
    if (!unpinned.is_ready()) return unpinned;
    (*counter)--; (*current)--;
    return llm_expert_provider_result::success();
}

llm_expert_provider_result llm_cold_expert_cache::release_many(
        const llm_cold_reference * references,
        size_t reference_count,
        llm_cold_reference_kind kind) noexcept {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    if (reference_count != 0 && references == nullptr) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::invalid_key);
    }
    auto capacity = policy_result(pimpl->policy.validate_event_capacity(reference_count));
    if (!capacity.is_ready()) return capacity;
    const uint64_t current_references = kind == llm_cold_reference_kind::hot ?
        pimpl->counters.current_hot_refs : kind == llm_cold_reference_kind::transfer ?
        pimpl->counters.current_transfer_refs : kind == llm_cold_reference_kind::request ?
        pimpl->counters.current_request_refs : pimpl->counters.current_cpu_execution_refs;
    if (reference_count > current_references) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
    }
    for (size_t index = 0; index < reference_count; ++index) {
        if (!pimpl->valid_reference(references[index])) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::stale_generation);
        }
        const auto & slot = pimpl->slots[references[index].slot];
        const uint32_t counter = kind == llm_cold_reference_kind::hot ? slot.hot_refs :
            kind == llm_cold_reference_kind::transfer ? slot.transfer_refs :
            kind == llm_cold_reference_kind::request ? slot.request_refs : slot.cpu_execution_refs;
        size_t occurrences = 0;
        for (size_t prior = 0; prior <= index; ++prior) {
            occurrences += references[prior].slot == references[index].slot &&
                references[prior].generation == references[index].generation;
        }
        if (occurrences > counter) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
        }
    }
    for (size_t index = 0; index < reference_count; ++index) {
        auto & slot = pimpl->slots[references[index].slot];
        uint32_t * counter = nullptr;
        uint64_t * current = nullptr;
        if (kind == llm_cold_reference_kind::hot) {
            counter = &slot.hot_refs; current = &pimpl->counters.current_hot_refs;
        } else if (kind == llm_cold_reference_kind::transfer) {
            counter = &slot.transfer_refs; current = &pimpl->counters.current_transfer_refs;
        } else if (kind == llm_cold_reference_kind::request) {
            counter = &slot.request_refs; current = &pimpl->counters.current_request_refs;
        } else {
            counter = &slot.cpu_execution_refs; current = &pimpl->counters.current_cpu_execution_refs;
        }
        if (*counter == 0 || *current == 0) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
        }
        const auto unpinned = policy_result(
            pimpl->policy.unpin(references[index].slot, references[index].generation));
        if (!unpinned.is_ready()) return unpinned;
        (*counter)--;
        (*current)--;
    }
    return llm_expert_provider_result::success();
}

bool llm_cold_expert_cache::ready(llm_cold_reference reference) const noexcept {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    return pimpl->valid_reference(reference);
}

llm_expert_provider_result llm_cold_expert_cache::retire_ready(llm_cold_reference reference) noexcept {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    if (!pimpl->valid_reference(reference)) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::stale_generation);
    }
    auto & slot = pimpl->slots[reference.slot];
    if (slot.state != llm_cold_slot_state::ready || !pimpl->no_refs(slot)) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::busy);
    }
    LLM_EXPERT_TRACE_INSTANT("k3.cache.cold", "victim", "layer", slot.key.layer,
        "original_expert_id", slot.key.expert, "slot_id", reference.slot,
        "generation", slot.generation);
    const auto removed = policy_result(pimpl->policy.remove_resident(reference.slot, reference.generation));
    if (!removed.is_ready()) return removed;
    pimpl->clear_forward(reference.slot);
    LLM_EXPERT_TRACE_INSTANT("k3.cache.cold", "eviction", "layer", slot.key.layer,
        "original_expert_id", slot.key.expert, "slot_id", reference.slot,
        "generation", slot.generation);
    slot.key = { -1, -1 };
    slot.layout_class_id = LLM_EXPERT_LAYOUT_CLASS_INVALID;
    slot.state = llm_cold_slot_state::free;
    LLM_EXPERT_TRACE_COUNTER("k3.resource", "cold_cache_occupancy", 5,
        pimpl->occupancy());
    return llm_expert_provider_result::success();
}

llm_expert_provider_result llm_cold_expert_cache::cleanup_failed_slots() noexcept {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    for (auto & slot : pimpl->slots) {
        if (slot.state == llm_cold_slot_state::failed && pimpl->no_refs(slot)) {
            slot.key = { -1, -1 };
            slot.layout_class_id = LLM_EXPERT_LAYOUT_CLASS_INVALID;
            slot.state = llm_cold_slot_state::free;
            pimpl->counters.failed_cleanups++;
        }
    }
    return llm_expert_provider_result::success();
}

llm_expert_provider_result llm_cold_expert_cache::policy_request_begin() noexcept {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    return pimpl->ensure_policy_request();
}

llm_expert_provider_result llm_cold_expert_cache::policy_set_ubatch_ordinal(uint64_t ordinal) noexcept {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    auto result = pimpl->ensure_policy_request();
    if (result.is_ready()) result = policy_result(pimpl->policy.set_ubatch_ordinal(ordinal));
    return result;
}

llm_expert_provider_result llm_cold_expert_cache::policy_phase_transition(
        llm_expert_cache_policy_phase phase) noexcept {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    auto result = pimpl->ensure_policy_request();
    if (!result.is_ready() || pimpl->policy_phase == phase) return result;
    result = policy_result(pimpl->policy.phase_transition(phase));
    if (result.is_ready()) pimpl->policy_phase = phase;
    return result;
}

llm_expert_provider_result llm_cold_expert_cache::policy_request_end(
        bool success, bool cancelled, bool allow_deferred_terminals) noexcept {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    if (!pimpl->policy_request_active) return llm_expert_provider_result::success();
    const auto result = policy_result(pimpl->policy.request_end(
        success, cancelled, allow_deferred_terminals));
    if (result.is_ready()) pimpl->policy_request_active = false;
    return result;
}

llm_expert_provider_result llm_cold_expert_cache::trim() noexcept {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    for (uint32_t index = 0; index < pimpl->slots.size(); ++index) {
        auto & slot = pimpl->slots[index];
        if (slot.state == llm_cold_slot_state::ready && pimpl->no_refs(slot)) {
            LLM_EXPERT_TRACE_INSTANT("k3.cache.cold", "victim", "layer", slot.key.layer,
                "original_expert_id", slot.key.expert, "slot_id", index,
                "generation", slot.generation);
            const auto removed = policy_result(pimpl->policy.remove_resident(index, slot.generation));
            if (!removed.is_ready()) return removed;
            pimpl->clear_forward(index);
            LLM_EXPERT_TRACE_INSTANT("k3.cache.cold", "eviction", "layer", slot.key.layer,
                "original_expert_id", slot.key.expert, "slot_id", index,
                "generation", slot.generation);
            slot.key = { -1, -1 };
            slot.layout_class_id = LLM_EXPERT_LAYOUT_CLASS_INVALID;
            slot.state = llm_cold_slot_state::free;
            LLM_EXPERT_TRACE_COUNTER("k3.resource", "cold_cache_occupancy", 5,
                pimpl->occupancy());
            if (pimpl->config.reclaim_free_pages) {
#ifdef __linux__
                const uint64_t reclaimed = pimpl->counters.aligned_slot_footprint;
                auto * base = static_cast<uint8_t *>(ggml_backend_buffer_get_base(pimpl->arena->buffer.get()));
                if (reclaimed == 0 || reclaimed > SIZE_MAX || index > SIZE_MAX/size_t(reclaimed) ||
                    madvise(base + size_t(index)*size_t(reclaimed), size_t(reclaimed), MADV_DONTNEED) != 0 ||
                    reclaimed > UINT64_MAX - pimpl->counters.reclaimed_bytes) {
                    pimpl->counters.reclaim_failures++;
                    return llm_expert_provider_result::failure(llm_expert_provider_error::preparation_failed);
                }
                pimpl->counters.reclaimed_bytes += reclaimed;
#else
                pimpl->counters.reclaim_failures++;
                return llm_expert_provider_result::failure(llm_expert_provider_error::unsupported_configuration);
#endif
            }
        }
    }
    return llm_expert_provider_result::success();
}

llm_expert_provider_result llm_cold_expert_cache::surrender() noexcept {
    LLM_EXPERT_TRACE_SCOPE("k3.lifecycle", "cold_cache_surrender");
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    if (pimpl->arena && pimpl->arena.use_count() != 1) {
        return llm_expert_provider_result::failure(llm_expert_provider_error::busy);
    }
    for (const auto & slot : pimpl->slots) {
        if (!pimpl->no_refs(slot) || slot.state == llm_cold_slot_state::loading) {
            return llm_expert_provider_result::failure(llm_expert_provider_error::busy);
        }
    }
    if (pimpl->policy_request_active) {
        const auto ended = policy_result(pimpl->policy.request_end(true, false));
        if (!ended.is_ready()) return ended;
        pimpl->policy_request_active = false;
    }
    for (uint32_t index = 0; index < pimpl->slots.size(); ++index) {
        const auto & slot = pimpl->slots[index];
        if (slot.state == llm_cold_slot_state::ready) {
            LLM_EXPERT_TRACE_INSTANT("k3.cache.cold", "victim", "layer", slot.key.layer,
                "original_expert_id", slot.key.expert, "slot_id", index,
                "generation", slot.generation);
            const auto removed = policy_result(pimpl->policy.remove_resident(index, slot.generation));
            if (!removed.is_ready()) return removed;
            LLM_EXPERT_TRACE_INSTANT("k3.cache.cold", "eviction", "layer", slot.key.layer,
                "original_expert_id", slot.key.expert, "slot_id", index,
                "generation", slot.generation);
        }
    }
    const auto policy_surrendered = policy_result(pimpl->policy.surrender());
    if (!policy_surrendered.is_ready()) return policy_surrendered;
    pimpl->arena.reset();
    pimpl->directory.clear();
    pimpl->slots.clear();
    LLM_EXPERT_TRACE_COUNTER("k3.resource", "cold_cache_occupancy", 5, 0);
    pimpl->counters.actual_bytes = 0;
    pimpl->counters.unused_budget_bytes = pimpl->config.byte_budget;
    pimpl->counters.effective_slots = 0;
    pimpl->counters.pageable = false;
    pimpl->policy = {};
    pimpl->policy_candidates.clear();
    pimpl->ready_cv.notify_all();
    return llm_expert_provider_result::success();
}

llm_expert_provider_result llm_cold_expert_cache::validate_invariants(
        const std::vector<llm_cold_hot_backing> & hot_backings) noexcept {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    for (const auto & backing : hot_backings) {
        if (backing.cold_slot >= pimpl->slots.size()) {
            pimpl->counters.invariant_failures++;
            return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
        }
        const auto & slot = pimpl->slots[backing.cold_slot];
        if (slot.state != llm_cold_slot_state::ready || slot.generation != backing.cold_generation ||
            slot.layout_class_id != backing.layout_class_id || !key_matches(slot.key, backing.key)) {
            pimpl->counters.invariant_failures++;
            return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
        }
    }
    for (uint32_t index = 0; index < pimpl->slots.size(); ++index) {
        const auto & slot = pimpl->slots[index];
        const bool occupied = slot.state != llm_cold_slot_state::free;
        if ((occupied && slot.layout_class_id != pimpl->class_for_key(slot.key)) ||
            (!occupied && slot.layout_class_id != LLM_EXPERT_LAYOUT_CLASS_INVALID)) {
            pimpl->counters.invariant_failures++;
            return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
        }
        if (slot.state == llm_cold_slot_state::ready || slot.state == llm_cold_slot_state::loading) {
            const bool speculative = slot.origin == llm_expert_residency_origin::speculative;
            if ((speculative && (slot.speculative_consumed || slot.speculative_deadline == 0)) ||
                (!speculative && (slot.speculative_deadline != 0 || slot.speculative_utility != 0))) {
                pimpl->counters.invariant_failures++;
                return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
            }
        }
        uint32_t expected_hot = 0;
        for (const auto & backing : hot_backings) {
            expected_hot += backing.cold_slot == index;
        }
        if (slot.hot_refs != expected_hot) {
            pimpl->counters.invariant_failures++;
            return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
        }
        if (slot.state == llm_cold_slot_state::ready) {
            if (!slot.key.is_valid(LLAMA_MAX_LAYERS, pimpl->n_expert)) {
                pimpl->counters.invariant_failures++;
                return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
            }
            const auto & forward = pimpl->directory[pimpl->forward_index(slot.key)];
            if (forward.slot != int32_t(index) || forward.generation != slot.generation) {
                pimpl->counters.invariant_failures++;
                return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
            }
            if (!pimpl->policy.validate_resident(index, slot.generation,
                    pimpl->policy_key(slot.key))) {
                pimpl->counters.invariant_failures++;
                return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
            }
        } else if (slot.state == llm_cold_slot_state::loading &&
                   !pimpl->policy.validate_loading(index, slot.generation,
                       pimpl->policy_key(slot.key))) {
            pimpl->counters.invariant_failures++;
            return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
        }
    }
    for (size_t index = 0; index < pimpl->directory.size(); ++index) {
        const auto & forward = pimpl->directory[index];
        if (forward.slot < 0) {
            continue;
        }
        if (uint32_t(forward.slot) >= pimpl->slots.size()) {
            pimpl->counters.invariant_failures++;
            return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
        }
        const auto & slot = pimpl->slots[forward.slot];
        const llm_expert_key expected = { int32_t(index/pimpl->n_expert), int32_t(index%pimpl->n_expert) };
        if (slot.state != llm_cold_slot_state::ready || slot.generation != forward.generation ||
            !key_matches(slot.key, expected)) {
            pimpl->counters.invariant_failures++;
            return llm_expert_provider_result::failure(llm_expert_provider_error::metadata_mismatch);
        }
    }
    return llm_expert_provider_result::success();
}

const llm_expert_bundle_descriptor & llm_cold_expert_cache::bundle() const noexcept {
    static const llm_expert_bundle_descriptor empty = {};
    return pimpl->arena ? pimpl->arena->bundle : empty;
}

const llm_expert_bundle_descriptor & llm_cold_expert_cache::bundle(
        llm_expert_layout_class_id layout_class_id) const noexcept {
    static const llm_expert_bundle_descriptor empty = {};
    return pimpl->arena && layout_class_id < pimpl->arena->bundles.size() ?
        pimpl->arena->bundles[layout_class_id] : empty;
}

const llm_expert_bundle_descriptor & llm_cold_expert_cache::bundle_for_key(
        llm_expert_key key) const noexcept {
    static const llm_expert_bundle_descriptor empty = {};
    const auto * result = pimpl->bundle_for_key(key);
    return result == nullptr ? empty : *result;
}

ggml_backend_buffer_t llm_cold_expert_cache::buffer() const noexcept {
    return pimpl->arena ? pimpl->arena->buffer.get() : nullptr;
}

std::shared_ptr<void> llm_cold_expert_cache::allocation_lease() const noexcept {
    return std::static_pointer_cast<void>(pimpl->arena);
}

llm_cold_cache_diagnostics llm_cold_expert_cache::diagnostics() const {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    auto result = pimpl->counters;
    if (pimpl->arena) {
        result.layout_class_count = uint32_t(pimpl->arena->payload_bytes.size());
        result.class_payload_bytes = pimpl->arena->payload_bytes;
        result.role_offsets.assign(pimpl->arena->role_offsets.begin(), pimpl->arena->role_offsets.end());
        result.role_extents.assign(pimpl->arena->role_extents.begin(), pimpl->arena->role_extents.end());
        result.class_padding_bytes.reserve(pimpl->arena->payload_bytes.size());
        for (uint64_t payload : pimpl->arena->payload_bytes) {
            result.class_padding_bytes.push_back(
                pimpl->arena->payload_bytes.size() == 1 ? 0 :
                pimpl->counters.aligned_slot_footprint >= payload ?
                    pimpl->counters.aligned_slot_footprint - payload : 0);
        }
    }
    result.policy = pimpl->policy.diagnostics();
    result.policy_domains = pimpl->policy.domain_diagnostics();
    result.policy_events.assign(
        pimpl->policy.transcript().begin(),
        pimpl->policy.transcript().begin() + pimpl->policy.transcript_size());
    result.slots = pimpl->slots;
#ifdef __linux__
    if (pimpl->arena) sample_ready_residency(*pimpl->arena, pimpl->slots, result);
#else
    result.residency_unavailable_reason = "mincore residency telemetry is supported only on Linux";
#endif
    return result;
}
