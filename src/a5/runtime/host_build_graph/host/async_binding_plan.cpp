/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include "async_binding_plan.h"

#include <algorithm>
#include <cstring>
#include <map>
#include <new>
#include <stdexcept>
#include <tuple>

#include "host_build_graph/runtime_types.h"
#include "scheduler/scheduler_graph_binding.h"

namespace {
struct Interval {
    uint64_t address{0}, bytes{0};
};

bool contains(Interval span, Interval requested) {
    return async_resource_detail::contains(span.address, span.bytes, requested.address, requested.bytes);
}
bool overlaps(Interval a, Interval b) {
    return a.bytes != 0 && b.bytes != 0 && a.address < b.address + b.bytes && b.address < a.address + a.bytes;
}
bool fail(std::string *error, const char *message) {
    if (error != nullptr) *error = message;
    return false;
}

struct Budget {
    uint64_t remaining;
    bool take(uint64_t count, uint64_t stride) {
        if (count > remaining / stride) return false;
        remaining -= count * stride;
        return true;
    }
};

template <typename T>
const T *within(const AsyncBindingTaskImage &image, const T *pointer, uint64_t count = 1) {
    if (count > UINT64_MAX / sizeof(T)) return nullptr;
    const Interval storage{reinterpret_cast<uint64_t>(image.host_base), image.bytes};
    return contains(storage, {reinterpret_cast<uint64_t>(pointer), count * sizeof(T)}) ? pointer : nullptr;
}

const ChipTaskStorage *task_storage(const AsyncBindingTaskImage &image, uint32_t index) {
    if (index >= image.task_count || image.storage_offset > image.bytes ||
        image.task_count > (image.bytes - image.storage_offset) / sizeof(ChipTaskStorage))
        return nullptr;
    return reinterpret_cast<const ChipTaskStorage *>(
               static_cast<const std::byte *>(image.host_base) + image.storage_offset
           ) +
           index;
}

template <typename T>
const T *definition_element(const NativeGraphDefinitionPlan &plan, uint32_t offset, int32_t count, int32_t index) {
    if (index < 0 || count < 0 || index >= count || offset > plan.image.size() ||
        static_cast<uint64_t>(count) > (plan.image.size() - offset) / sizeof(T) || offset % alignof(T) != 0)
        return nullptr;
    return reinterpret_cast<const T *>(plan.image.data() + offset) + index;
}

struct TaskArguments {
    std::vector<simpler::hbg::TensorData> tensors;
    uint32_t scalar_count{0};
    const AsyncHostDeclarations::Uses *uses{nullptr};
    Interval predicate{};
};

bool decode_task(
    uint32_t task, const AsyncBindingTaskImage &image, const NativeGraphHostPlan &native,
    const AsyncHostDeclarations &declarations, TaskArguments *out
) {
    const SchedulerNodeBinding *binding = native.instances.empty() ? nullptr : &native.bindings[task];
    if (binding == nullptr || binding->instance_index == SCHEDULER_NATIVE_NO_INSTANCE) {
        const uint32_t original = binding == nullptr ? task : binding->source_index;
        const auto *storage = task_storage(image, original);
        if (storage == nullptr) return false;
        const auto &payload = storage->payload;
        if (payload.tensor_count < 0 || payload.scalar_count < 0 || payload.tensor_count > CORE_MAX_TENSOR_ARGS ||
            payload.scalar_count > CORE_MAX_SCALAR_ARGS)
            return false;
        const auto *tensors = within(image, payload.tensor_data(), payload.tensor_count);
        if (payload.tensor_count != 0 && tensors == nullptr) return false;
        for (int32_t i = 0; i < payload.tensor_count; ++i)
            out->tensors.push_back(tensors[i]);
        out->scalar_count = payload.scalar_count;
        out->uses = declarations.find(false, 0, original);
        out->predicate = {payload.predicate.addr, payload.predicate.elem_size};
        return true;
    }
    if (binding->kind != SchedulerNodeKind::CHILD_KERNEL || binding->instance_index >= native.instances.size())
        return false;
    auto instance = native.instances[binding->instance_index];
    if (instance.definition_index >= native.definitions.size()) return false;
    const auto &plan = native.definitions[instance.definition_index];
    const auto &def = plan.definition;
    const auto *child =
        definition_element<SubTaskDefinition>(plan, def.off_sub_tasks, def.task_count, binding->source_index);
    const auto *shell = task_storage(image, instance.original_shell_id);
    if (child == nullptr || shell == nullptr || shell->payload.tensor_count < 0 ||
        instance.boundary_tensor_count != static_cast<uint32_t>(shell->payload.tensor_count))
        return false;
    const auto *boundary = within(image, shell->payload.tensor_data(), instance.boundary_tensor_count);
    if (instance.boundary_tensor_count != 0 && boundary == nullptr) return false;
    const uint64_t expected_boundary =
        instance.boundary_tensor_count == 0 ?
            0 :
            image.device_base + (reinterpret_cast<uint64_t>(boundary) - reinterpret_cast<uint64_t>(image.host_base));
    if (instance.boundary_tensors_address != expected_boundary ||
        instance.heap_base != reinterpret_cast<uint64_t>(shell->task.packed_buffer_base))
        return false;
    instance.boundary_tensors_address = reinterpret_cast<uint64_t>(boundary);
    SchedulerGraphTemplate graph_template{};
    graph_template.required_heap = def.required_heap;
    SchedulerNativeNode node{nullptr, binding, &instance, &graph_template};
    for (int32_t arg = 0; arg < child->tensor_count; ++arg) {
        const auto *record = definition_element<simpler::hbg::TensorData>(
            plan, def.off_tensors, def.tensor_arg_count, child->tensor_offset + arg
        );
        SchedulerTensorBinding bound{};
        if (record == nullptr || !scheduler_graph_relocate_tensor(*record, node, &bound)) return false;
        auto tensor = *record;
        tensor.buffer = {bound.address, bound.bytes};
        tensor.start_offset = bound.origin;
        tensor.address_space = bound.address_space;
        out->tensors.push_back(tensor);
    }
    out->scalar_count = child->scalar_count;
    out->uses = declarations.find(true, def.graph_key, binding->source_index);
    if (child->predicate_slot != 0) {
        const auto *predicate = definition_element<GraphPredicate>(
            plan, def.off_predicates, def.predicate_count, child->predicate_slot - 1
        );
        SchedulerTensorBinding bound{};
        if (predicate == nullptr || !scheduler_graph_relocate_tensor(predicate->operand, node, &bound) ||
            predicate->elem_size == 0 || predicate->elem_offset > UINT64_MAX - bound.origin)
            return false;
        const uint64_t element = bound.origin + predicate->elem_offset;
        if (bound.bytes < predicate->elem_size || element > (bound.bytes - predicate->elem_size) / predicate->elem_size)
            return false;
        out->predicate = {bound.address + element * predicate->elem_size, predicate->elem_size};
    }
    return true;
}

// Exact positive-stride geometry. Collapse contiguous axes first, then retain
// only the union of actual byte runs; holes never become record permissions.
bool tensor_runs(const simpler::hbg::TensorData &tensor, uint64_t byte_budget, std::vector<Interval> *out) {
    const uint64_t width = scheduler_graph_element_size(tensor.dtype);
    if (width == 0 || tensor.ndims == 0 || tensor.ndims > MAX_TENSOR_DIMS ||
        tensor.address_space != AddressSpace::DEVICE || tensor.buffer.size > UINT64_MAX - tensor.buffer.addr ||
        tensor.start_offset > tensor.buffer.size / width)
        return false;
    std::vector<std::pair<uint64_t, uint32_t>> axes;
    uint64_t elements = 1, last = 0;
    for (uint32_t i = 0; i < tensor.ndims; ++i) {
        if (tensor.shapes[i] == 0) return true;
        if (tensor.strides[i] == 0 || !scheduler_layout_checked_mul(elements, tensor.shapes[i], &elements))
            return false;
        const uint64_t extent = static_cast<uint64_t>(tensor.shapes[i] - 1) * tensor.strides[i];
        if (!scheduler_layout_checked_add(last, extent, &last)) return false;
        if (tensor.shapes[i] > 1) axes.emplace_back(tensor.strides[i], tensor.shapes[i]);
    }
    if (last >= tensor.buffer.size / width - tensor.start_offset) return false;
    std::sort(axes.begin(), axes.end());
    uint64_t contiguous = 1;
    size_t first = 0;
    while (first < axes.size() && axes[first].first == contiguous)
        contiguous *= axes[first++].second;
    const uint64_t count = elements / contiguous;
    if (count > byte_budget / sizeof(Interval)) return false;
    const uint64_t origin = tensor.buffer.addr + tensor.start_offset * width;
    for (uint64_t run = 0; run < count; ++run) {
        uint64_t ordinal = run, offset = 0;
        for (size_t dim = first; dim < axes.size(); ++dim) {
            offset += ordinal % axes[dim].second * axes[dim].first;
            ordinal /= axes[dim].second;
        }
        out->push_back({origin + offset * width, contiguous * width});
    }
    std::sort(out->begin(), out->end(), [](Interval a, Interval b) {
        return a.address < b.address;
    });
    size_t kept = 0;
    for (const auto span : *out) {
        if (kept != 0 && span.address <= (*out)[kept - 1].address + (*out)[kept - 1].bytes) {
            auto &previous = (*out)[kept - 1];
            previous.bytes = std::max(previous.address + previous.bytes, span.address + span.bytes) - previous.address;
        } else (*out)[kept++] = span;
    }
    out->resize(kept);
    return true;
}

struct Builder {
    const HostApi &api;
    Budget budget;
    AsyncBindingHostPlan plan;
    std::map<std::pair<uint64_t, uint64_t>, uint64_t> span_indices;
    std::map<std::tuple<uint64_t, uint64_t, uint64_t, uint64_t>, uint64_t> range_indices;

    bool range(Interval logical, uint64_t access, uint64_t *index) {
        if (logical.bytes == 0 || logical.address == 0 || logical.bytes > UINT64_MAX - logical.address ||
            logical.address + logical.bytes > UINT64_MAX - 63)
            return false;
        const uint64_t first = logical.address & ~UINT64_C(63);
        const uint64_t end = (logical.address + logical.bytes + 63) & ~UINT64_C(63);
        HostAsyncRetainedSpan owner{};
        if (!api.query_async_retained_span(first, end - first, &owner) ||
            owner.owner_kind == HostAsyncRetainedOwner::UNKNOWN || owner.reserved != 0 ||
            !contains({owner.base, owner.bytes}, {first, end - first}))
            return false;
        const auto key = std::make_pair(owner.base, owner.bytes);
        auto found = span_indices.find(key);
        if (found == span_indices.end()) {
            for (const auto &existing : plan.spans)
                if (overlaps({existing.address, existing.bytes}, {owner.base, owner.bytes})) return false;
            if (!budget.take(1, sizeof(AsyncRetainedSpan))) return false;
            found = span_indices.emplace(key, plan.spans.size()).first;
            plan.spans.push_back({owner.base, owner.bytes});
        }
        const auto range_key = std::make_tuple(found->second, logical.address - owner.base, logical.bytes, access);
        auto saved = range_indices.find(range_key);
        if (saved == range_indices.end()) {
            if (!budget.take(1, sizeof(AsyncResourceRange))) return false;
            saved = range_indices.emplace(range_key, plan.ranges.size()).first;
            plan.ranges.push_back({found->second, logical.address - owner.base, logical.bytes, access});
        }
        *index = saved->second;
        return true;
    }

    Interval interval(uint64_t index) const {
        const auto &range = plan.ranges[index];
        return {plan.spans[range.span_index].address + range.byte_offset, range.bytes};
    }
};

struct RecordUse {
    bool control_only{false};
    uint64_t record_range{0};
    std::vector<uint64_t> payloads;
};

bool build(
    const std::vector<SchedulerTaskMetadata> &metadata, const std::vector<SchedulerAsyncCapabilities> &capabilities,
    const AsyncBindingTaskImage &image, const NativeGraphHostPlan &native, const AsyncHostDeclarations &declarations,
    const HostApi &api, uint64_t byte_budget, AsyncBindingHostPlan *out, std::string *error
) {
    if (out == nullptr || api.run_epoch() == 0 || image.host_base == nullptr ||
        metadata.size() != capabilities.size() || metadata.size() > INT32_MAX ||
        (native.instances.empty() ? image.task_count != metadata.size() : native.bindings.size() != metadata.size()))
        return fail(error, "async bindings require final task arguments and the live native preparation");
    if (image.bytes > UINT64_MAX - reinterpret_cast<uint64_t>(image.host_base) ||
        image.bytes > UINT64_MAX - image.device_base)
        return fail(error, "async argument image extent overflows");
    SchedulerAsyncLayout skeleton{};
    if (!scheduler_plan_async_layout(metadata, capabilities, api.run_epoch(), byte_budget, {}, &skeleton, error))
        return false;
    Builder builder{api, Budget{byte_budget}, {}, {}, {}};
    auto &plan = builder.plan;
    plan.run_epoch = api.run_epoch();
    if (!builder.budget.take(1, sizeof(AsyncResourceHeader)) ||
        !builder.budget.take(metadata.size(), sizeof(AsyncConsumerTaskMap)))
        return fail(error, "async binding tables exceed byte budget");
    plan.consumer_maps.resize(metadata.size());
    std::vector<TaskArguments> args(metadata.size());
    std::vector<std::vector<uint64_t>> consumer_inputs(metadata.size());
    std::vector<Interval> payloads;
    for (uint32_t task = 0; task < metadata.size(); ++task) {
        const auto &meta = metadata[task];
        if (!scheduler_task_is_executable(meta.flags) || scheduler_task_is_inline(meta.flags)) continue;
        if (!decode_task(task, image, native, declarations, &args[task]))
            return fail(error, "async bindings cannot decode final ordinary/native Graph arguments");
        auto &actual = args[task];
        std::map<uint64_t, RecordUse> records;
        bool declared_wait = false;
        if (actual.uses != nullptr)
            for (const auto &use : *actual.uses) {
                if (!async_use_declaration_valid(use, actual.tensors.size(), actual.scalar_count))
                    return fail(error, "async declaration no longer matches this invocation's argument positions");
                if (use.kind == AsyncUseDeclarationKind::SDMA_TRANSFER)
                    return fail(
                        error,
                        "SDMA binding requires enrolled domain, public record, channel and submit-cache qualification"
                    );
                if (use.cache_contract != AsyncUseCacheContract::PUBLISHED_CLEAN_LINES)
                    return fail(error, "async use lacks an explicit PUBLISHED_CLEAN_LINES author/compiler contract");
                std::vector<uint64_t> use_payloads;
                if (use.kind != AsyncUseDeclarationKind::COUNTER_ONLY) {
                    std::vector<Interval> runs;
                    if (!tensor_runs(actual.tensors[use.payload_arg], byte_budget, &runs))
                        return fail(error, "async payload has invalid or unbudgeted exact view geometry");
                    for (const auto span : runs) {
                        uint64_t index = 0;
                        const uint64_t access =
                            ASYNC_RANGE_READ |
                            (use.kind == AsyncUseDeclarationKind::CONSUMER_CLEAN_READ ? ASYNC_RANGE_CONSUMER_CLEAN : 0);
                        if (!builder.range(span, access, &index))
                            return fail(
                                error,
                                "async payload/cache line has no retained owner for this preparation or exceeds budget"
                            );
                        use_payloads.push_back(index);
                        if (use.kind == AsyncUseDeclarationKind::CONSUMER_CLEAN_READ)
                            consumer_inputs[task].push_back(index);
                        else payloads.push_back(span);
                    }
                }
                if (use.kind == AsyncUseDeclarationKind::CONSUMER_CLEAN_READ) continue;
                declared_wait = true;
                const auto &signal = actual.tensors[use.condition_arg];
                std::vector<Interval> runs;
                if (signal.dtype != DataType::INT32 || !tensor_runs(signal, byte_budget, &runs))
                    return fail(error, "COUNTER declaration requires an exact INT32 device view");
                const uint64_t origin = signal.buffer.addr + signal.start_offset * sizeof(int32_t);
                for (const auto span : runs) {
                    const uint64_t delta = span.address - origin;
                    const uint64_t skip = (use.record_stride - delta % use.record_stride) % use.record_stride;
                    if (skip > span.bytes || span.bytes - skip < sizeof(int32_t)) continue;
                    const uint64_t count = (span.bytes - skip - sizeof(int32_t)) / use.record_stride + 1;
                    if (count > byte_budget / sizeof(AsyncConditionContract))
                        return fail(error, "COUNTER allowed-record set exceeds byte budget");
                    for (uint64_t n = 0; n < count; ++n) {
                        const uint64_t address = span.address + skip + n * use.record_stride;
                        uint64_t range = 0;
                        if (!builder.range(
                                {address, sizeof(int32_t)}, ASYNC_RANGE_READ | ASYNC_RANGE_CONDITION_CLEAN, &range
                            ))
                            return fail(error, "COUNTER record/cache line has no retained owner or exceeds budget");
                        const bool only = use.kind == AsyncUseDeclarationKind::COUNTER_ONLY;
                        auto [entry, inserted] = records.try_emplace(address, RecordUse{only, range, {}});
                        if (!inserted && entry->second.control_only != only)
                            return fail(error, "aliased COUNTER selectors disagree on control-only/payload semantics");
                        for (uint64_t payload : use_payloads)
                            if (std::find(entry->second.payloads.begin(), entry->second.payloads.end(), payload) ==
                                entry->second.payloads.end())
                                entry->second.payloads.push_back(payload);
                    }
                }
            }
        std::map<std::vector<uint64_t>, std::pair<uint64_t, uint64_t>> sets;
        for (uint32_t block = 0; block < meta.logical_block_num; ++block)
            for (uint32_t lane = 0; lane < 3; ++lane) {
                if (!meta.active_mask.has(static_cast<SubtaskSlot>(lane)) ||
                    capabilities[task][lane].mode != CoreAsyncMode::DEFERRED)
                    continue;
                if (!declared_wait) return fail(error, "deferred execution has no declared allowed condition uses");
                if ((capabilities[task][lane].condition_types & CORE_ASYNC_CONDITION_COUNTER) == 0)
                    return fail(error, "COUNTER use does not match the registered lane capability");
                if (records.size() > UINT32_MAX)
                    return fail(error, "allowed-use set exceeds its uint32 index contract");
                if (!builder.budget.take(records.size(), sizeof(AsyncConditionContract)))
                    return fail(error, "per-execution allowed-use sets exceed byte budget");
                const uint64_t begin = plan.conditions.size();
                for (const auto &[address, record] : records) {
                    (void)address;
                    uint64_t payload_begin = ASYNC_RESOURCE_NONE, payload_count = 0;
                    if (!record.control_only && !record.payloads.empty()) {
                        auto cached = sets.find(record.payloads);
                        if (cached == sets.end()) {
                            if (!builder.budget.take(record.payloads.size(), sizeof(AsyncResourceRange)))
                                return fail(error, "condition payload sets exceed byte budget");
                            const auto start = plan.ranges.size();
                            for (auto index : record.payloads)
                                plan.ranges.push_back(plan.ranges[index]);
                            cached = sets.emplace(record.payloads, std::make_pair(start, record.payloads.size())).first;
                        }
                        std::tie(payload_begin, payload_count) = cached->second;
                    }
                    plan.conditions.push_back(
                        {static_cast<int64_t>(task), block, lane, static_cast<uint32_t>(plan.conditions.size() - begin),
                         ASYNC_CORE_AIC | ASYNC_CORE_AIV, COMPLETION_TYPE_COUNTER, payload_count == 0 ? 1u : 0u,
                         record.record_range, 1, sizeof(int32_t), ASYNC_RESOURCE_NONE, payload_begin, payload_count}
                    );
                }
                plan.groups.push_back({begin, plan.conditions.size() - begin});
            }
    }
    // Every potential payload reader must explicitly declare its acquisition.
    // This is a conservative argument-alias check, not analysis of kernel code.
    for (uint32_t task = 0; task < metadata.size(); ++task) {
        const auto &meta = metadata[task];
        if (!scheduler_task_is_executable(meta.flags) || scheduler_task_is_inline(meta.flags)) continue;
        auto &inputs = consumer_inputs[task];
        std::sort(inputs.begin(), inputs.end());
        inputs.erase(std::unique(inputs.begin(), inputs.end()), inputs.end());
        for (const auto &tensor : args[task].tensors) {
            std::vector<Interval> runs;
            if (!tensor_runs(tensor, byte_budget, &runs))
                return fail(error, "task argument geometry is not valid for async alias qualification");
            for (const auto run : runs)
                for (const auto payload : payloads) {
                    if (!overlaps(run, payload)) continue;
                    const Interval overlap{
                        std::max(run.address, payload.address),
                        std::min(run.address + run.bytes, payload.address + payload.bytes) -
                            std::max(run.address, payload.address)
                    };
                    if (std::none_of(inputs.begin(), inputs.end(), [&](uint64_t index) {
                            return contains(builder.interval(index), overlap);
                        }))
                        return fail(
                            error, "task aliases an async payload without an explicit clean-consumer declaration"
                        );
                }
        }
        if (scheduler_task_has_predicate(meta.flags)) {
            const auto predicate = args[task].predicate;
            if (predicate.bytes == 0 || std::none_of(inputs.begin(), inputs.end(), [&](uint64_t index) {
                    return contains(builder.interval(index), predicate);
                }))
                return fail(error, "predicate's earlier scalar read has no declared retained clean input range");
        }
        uint32_t mask = 0;
        for (uint32_t lane = 0; lane < 3; ++lane)
            if (meta.active_mask.has(static_cast<SubtaskSlot>(lane))) mask |= 1u << lane;
        auto &map = plan.consumer_maps[task];
        map.block_count = meta.logical_block_num;
        map.lane_mask = mask;
        map.kind = inputs.empty() ? AsyncConsumerMapKind::NONE : AsyncConsumerMapKind::REQUIRED;
        if (inputs.empty()) continue;
        map.contract_base = plan.consumers.size();
        const uint64_t executions = static_cast<uint64_t>(meta.logical_block_num) * meta.active_mask.count();
        if (!builder.budget.take(executions, sizeof(AsyncConsumerContract)) ||
            !builder.budget.take(inputs.size(), sizeof(uint64_t)))
            return fail(error, "consumer execution tables exceed byte budget");
        const uint64_t begin = plan.consumer_ranges.size();
        plan.consumer_ranges.insert(plan.consumer_ranges.end(), inputs.begin(), inputs.end());
        for (uint32_t block = 0; block < meta.logical_block_num; ++block)
            for (uint32_t lane = 0; lane < 3; ++lane)
                if (meta.active_mask.has(static_cast<SubtaskSlot>(lane)))
                    plan.consumers.push_back(
                        {static_cast<int64_t>(task), block, lane, begin, inputs.size(), ASYNC_CORE_AIC | ASYNC_CORE_AIV,
                         0}
                    );
    }
    plan.auxiliary.resources = sizeof(AsyncResourceHeader) + plan.spans.size() * sizeof(AsyncRetainedSpan) +
                               plan.ranges.size() * sizeof(AsyncResourceRange);
    plan.auxiliary.uses = plan.conditions.size() * sizeof(AsyncConditionContract);
    plan.auxiliary.consumers = plan.consumer_maps.size() * sizeof(AsyncConsumerTaskMap) +
                               plan.consumers.size() * sizeof(AsyncConsumerContract) +
                               plan.consumer_ranges.size() * sizeof(uint64_t);
    SchedulerAsyncLayout complete{};
    if (!scheduler_plan_async_layout(
            metadata, capabilities, api.run_epoch(), byte_budget, plan.auxiliary, &complete, error
        ) ||
        plan.groups.size() != complete.group_count)
        return fail(error, "complete async state and binding tables exceed budget or execution map");
    *out = std::move(plan);
    return true;
}
}  // namespace

bool scheduler_plan_async_bindings(
    const std::vector<SchedulerTaskMetadata> &metadata, const std::vector<SchedulerAsyncCapabilities> &capabilities,
    const AsyncBindingTaskImage &image, const NativeGraphHostPlan &native, const AsyncHostDeclarations &declarations,
    const HostApi &api, uint64_t byte_budget, AsyncBindingHostPlan *out, std::string *error
) {
    try {
        return build(metadata, capabilities, image, native, declarations, api, byte_budget, out, error);
    } catch (const std::bad_alloc &) {
        return fail(error, "cannot allocate async binding plan");
    } catch (const std::length_error &) {
        return fail(error, "async binding plan exceeds host container capacity");
    }
}

bool scheduler_initialize_async_bindings(
    const AsyncBindingHostPlan &plan, const SchedulerAsyncLayout &layout, void *region, uint64_t region_bytes,
    std::string *error
) {
    uint64_t resources = sizeof(AsyncResourceHeader), uses = 0, consumers = 0;
    if (!async_resource_detail::add_rows(plan.spans.size(), sizeof(AsyncRetainedSpan), resources) ||
        !async_resource_detail::add_rows(plan.ranges.size(), sizeof(AsyncResourceRange), resources) ||
        !async_resource_detail::add_rows(plan.conditions.size(), sizeof(AsyncConditionContract), uses) ||
        !async_resource_detail::add_rows(plan.consumer_maps.size(), sizeof(AsyncConsumerTaskMap), consumers) ||
        !async_resource_detail::add_rows(plan.consumers.size(), sizeof(AsyncConsumerContract), consumers) ||
        !async_resource_detail::add_rows(plan.consumer_ranges.size(), sizeof(uint64_t), consumers) ||
        resources != plan.auxiliary.resources || uses != plan.auxiliary.uses || consumers != plan.auxiliary.consumers)
        return fail(error, "async binding vectors changed after sizing");
    if (region == nullptr || layout.total_bytes < sizeof(SchedulerAsyncLayout) || region_bytes < layout.total_bytes ||
        plan.run_epoch != layout.run_epoch || plan.groups.size() != layout.group_count ||
        plan.consumer_maps.size() != layout.task_count || plan.auxiliary.resources != layout.resources_bytes ||
        plan.auxiliary.uses != layout.uses_bytes || plan.auxiliary.consumers != layout.consumers_bytes)
        return fail(error, "async binding image does not match its planned layout");
    if (layout.version != SCHEDULER_ASYNC_LAYOUT_VERSION ||
        layout.total_bytes > UINTPTR_MAX - reinterpret_cast<uintptr_t>(region))
        return fail(error, "async binding destination is invalid");
    // The state initializer already re-planned and validated every section,
    // including writer/lease/progress rows which this exporter does not own.
    // Bind only to that exact image; a changed offset must not overwrite them.
    if (std::memcmp(region, &layout, sizeof(layout)) != 0)
        return fail(error, "async binding layout differs from its initialized scheduler image");
    uint64_t end = sizeof(SchedulerAsyncLayout);
    for (const auto section :
         {Interval{layout.group_identity_offset, layout.group_count * sizeof(SchedulerAsyncGroupIdentity)},
          Interval{layout.resources_offset, layout.resources_bytes}, Interval{layout.uses_offset, layout.uses_bytes},
          Interval{layout.consumers_offset, layout.consumers_bytes}}) {
        if (section.bytes == 0) {
            if (section.address != 0 &&
                (section.address % 128 != 0 || section.address < end || section.address > layout.total_bytes))
                return fail(error, "empty async binding section has an invalid boundary");
            end = std::max(end, section.address);
            continue;
        }
        if (section.address % 128 != 0 || section.address < end || !contains({0, layout.total_bytes}, section))
            return fail(error, "async binding section is overlapping or outside its allocation");
        end = section.address + section.bytes;
    }
    for (const auto &group : plan.groups)
        if (group.begin > plan.conditions.size() || group.count > plan.conditions.size() - group.begin)
            return fail(error, "async group allowed-use range is invalid");
    auto *bytes = static_cast<std::byte *>(region);
    const auto copy = [&](uint64_t offset, const auto &values) {
        using Value = typename std::decay_t<decltype(values)>::value_type;
        if (!values.empty()) std::memcpy(bytes + offset, values.data(), values.size() * sizeof(Value));
    };
    AsyncResourceHeader header{ASYNC_RESOURCE_VERSION,      plan.run_epoch,
                               plan.spans.size(),           plan.ranges.size(),
                               plan.conditions.size(),      plan.consumers.size(),
                               plan.consumer_ranges.size(), 0};
    std::memcpy(bytes + layout.resources_offset, &header, sizeof(header));
    copy(layout.resources_offset + sizeof(header), plan.spans);
    copy(layout.resources_offset + sizeof(header) + plan.spans.size() * sizeof(AsyncRetainedSpan), plan.ranges);
    copy(layout.uses_offset, plan.conditions);
    copy(layout.consumers_offset, plan.consumer_maps);
    copy(layout.consumers_offset + plan.consumer_maps.size() * sizeof(AsyncConsumerTaskMap), plan.consumers);
    copy(
        layout.consumers_offset + plan.consumer_maps.size() * sizeof(AsyncConsumerTaskMap) +
            plan.consumers.size() * sizeof(AsyncConsumerContract),
        plan.consumer_ranges
    );
    for (uint64_t index = 0; index < plan.groups.size(); ++index) {
        SchedulerAsyncGroupIdentity identity{};
        auto *record = bytes + layout.group_identity_offset + index * sizeof(identity);
        std::memcpy(&identity, record, sizeof(identity));
        identity.binding_use_count = plan.groups[index].count;
        identity.binding_uses_offset =
            identity.binding_use_count == 0 ?
                0 :
                layout.uses_offset + plan.groups[index].begin * sizeof(AsyncConditionContract);
        std::memcpy(record, &identity, sizeof(identity));
    }
    return true;
}
