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
    std::vector<uint64_t> scalars;
    const AsyncHostDeclarations::Uses *uses{nullptr};
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
        const auto *scalars = within(image, payload.scalar_data(), payload.scalar_count);
        if (payload.scalar_count != 0 && scalars == nullptr) return false;
        for (int32_t i = 0; i < payload.scalar_count; ++i)
            out->scalars.push_back(scalars[i]);
        out->uses = declarations.find(false, 0, original);
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
    if (child->scalar_count < 0 || child->scalar_count > CORE_MAX_SCALAR_ARGS || child->scalar_offset < 0 ||
        child->scalar_offset > def.scalar_arg_count ||
        child->scalar_count > def.scalar_arg_count - child->scalar_offset)
        return false;
    out->scalar_count = child->scalar_count;
    for (int32_t arg = 0; arg < child->scalar_count; ++arg) {
        const auto *inheritance = definition_element<GraphScalarInheritance>(
            plan, def.off_scalar_inheritance, def.scalar_arg_count, child->scalar_offset + arg
        );
        const auto *value =
            definition_element<uint64_t>(plan, def.off_scalars, def.scalar_arg_count, child->scalar_offset + arg);
        if (inheritance == nullptr || value == nullptr || !inheritance->valid()) return false;
        if (inheritance->inherited()) {
            if (shell->payload.scalar_count < 0 ||
                instance.boundary_scalar_count != static_cast<uint32_t>(shell->payload.scalar_count) ||
                inheritance->boundary_index() >= instance.boundary_scalar_count)
                return false;
            const auto *boundary_scalars = within(image, shell->payload.scalar_data(), instance.boundary_scalar_count);
            if (boundary_scalars == nullptr ||
                instance.boundary_scalars_address != image.device_base + (reinterpret_cast<uint64_t>(boundary_scalars) -
                                                                          reinterpret_cast<uint64_t>(image.host_base)))
                return false;
            value = boundary_scalars + inheritance->boundary_index();
        }
        out->scalars.push_back(*value);
    }
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

struct SdmaUse {
    uint64_t source_range{ASYNC_RESOURCE_NONE};
    uint64_t payload_range{ASYNC_RESOURCE_NONE};
    uint64_t workspace_input{SCHEDULER_ASYNC_INDEX_NONE};
    std::vector<uint32_t> context_scalars;
};

bool bind_sdma_use(
    Builder &builder, const TaskArguments &actual, const AsyncUseDeclaration &use, uint64_t byte_budget,
    std::vector<SdmaUse> *out, std::string *error
) {
    if (use.source_peer_scalar == UINT32_MAX || use.source_peer_scalar >= actual.scalars.size() ||
        use.context_scalar >= actual.scalars.size() ||
        use.cache_contract != AsyncUseCacheContract::PUBLISHED_CLEAN_LINES)
        return fail(error, "SDMA TGET requires explicit peer scalar and PUBLISHED_CLEAN_LINES declaration");
    const uint64_t context_address = actual.scalars[use.context_scalar];
    const uint64_t peer = actual.scalars[use.source_peer_scalar];
    CommDomainSnapshot domain{};
    CommDomainExecution execution = CommDomainExecution::UNKNOWN;
    CommSdmaOwnerInfoV1 owner{};
    if (!builder.api.query_async_domain_context(context_address, &domain, &execution) ||
        !comm_domain_snapshot_valid(domain) || domain.use.device_context != context_address ||
        execution != CommDomainExecution::EXCLUSIVE_RUN ||
        !builder.api.query_async_sdma_owner(context_address, &owner) || owner.access != CommSdmaAccess::SDK_ONLY ||
        owner.queue_num == 0 || owner.queue_num > SCHEDULER_SUBMISSION_SDMA_CHANNELS)
        return fail(error, "SDMA context lacks selected exclusive domain and fixed SDK-only workspace owner");
    const auto &comm = domain.context;
    const auto &workspace = comm_workspace<CommWorkspaceKind::SDMA>(comm);
    if (peer >= comm.rankNum || comm.windowsIn[peer] == 0 || workspace.addr == 0 ||
        workspace.addr != owner.workspace_addr || workspace.size != owner.workspace_bytes || workspace.addr % 64 != 0 ||
        workspace.size < (SCHEDULER_SUBMISSION_SDMA_CHANNELS + 1) * 64)
        return fail(error, "SDMA peer or workspace does not match the retained allocation");
    std::vector<Interval> sources, destinations;
    if (!tensor_runs(actual.tensors[use.source_arg], byte_budget, &sources) ||
        !tensor_runs(actual.tensors[use.payload_arg], byte_budget, &destinations) ||
        sources.size() != destinations.size() || sources.size() > 1)
        return fail(error, "SDMA TGET currently requires one exact contiguous source and destination view");
    if (sources.empty()) return true;
    if (!contains({comm.windowsIn[comm.rankId], comm.winSize}, sources[0]))
        return fail(error, "SDMA local source reference must fit its selected input window");
    const uint64_t source_offset = sources[0].address - comm.windowsIn[comm.rankId];
    const Interval remote{comm.windowsIn[peer] + source_offset, sources[0].bytes};
    if (!contains({comm.windowsIn[peer], comm.winSize}, remote))
        return fail(error, "SDMA peer source view exceeds its selected input window");
    uint64_t source = 0, destination = 0;
    if (!builder.range(remote, ASYNC_RANGE_READ, &source) ||
        !builder.range(destinations[0], ASYNC_RANGE_READ | ASYNC_RANGE_WRITE, &destination))
        return fail(error, "SDMA source/destination cache lines lack an existing retained owner");
    uint64_t workspace_index = 0;
    auto &workspaces = builder.plan.submission_workspaces;
    for (; workspace_index < workspaces.size(); ++workspace_index) {
        const auto &saved = workspaces[workspace_index];
        if (saved.address != workspace.addr) continue;
        if (saved.bytes != workspace.size || saved.queue_num != owner.queue_num)
            return fail(error, "one SDMA workspace owner has conflicting retained configuration");
        break;
    }
    if (workspace_index == workspaces.size()) {
        if (!builder.budget.take(1, sizeof(SchedulerSubmissionWorkspaceInput)))
            return fail(error, "SDMA workspace declarations exceed byte budget");
        workspaces.push_back(
            {workspace.addr, workspace.size, builder.api.run_epoch(), owner.queue_num,
             SCHEDULER_SUBMISSION_SDMA_CHANNELS, SchedulerSubmissionPoolQualification::EXCLUSIVE_RUN}
        );
    }
    for (auto &saved : *out) {
        if (saved.workspace_input != workspace_index) continue;
        if (saved.source_range == source && saved.payload_range == destination) {
            if (std::find(saved.context_scalars.begin(), saved.context_scalars.end(), use.context_scalar) ==
                saved.context_scalars.end())
                saved.context_scalars.push_back(use.context_scalar);
            return true;
        }
        if (overlaps(builder.interval(saved.source_range), remote) &&
            overlaps(builder.interval(saved.payload_range), destinations[0]))
            return fail(error, "SDMA declarations admit an ambiguous transfer within one workspace");
    }
    out->push_back({source, destination, workspace_index, {use.context_scalar}});
    return true;
}

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
    if (!builder.budget.take(1, sizeof(AsyncResourceHeader)))
        return fail(error, "async binding tables exceed byte budget");
    bool submission_image = false;
    if (!scheduler_has_sdma_uses(metadata, native, declarations, &submission_image, error)) return false;
    bool core_only = true;
    bool has_execution = false;
    for (uint32_t task = 0; task < metadata.size(); ++task) {
        const auto &meta = metadata[task];
        if (!scheduler_task_is_executable(meta.flags) || scheduler_task_is_inline(meta.flags)) continue;
        TaskArguments actual{};
        if (!decode_task(task, image, native, declarations, &actual))
            return fail(error, "async bindings cannot decode final ordinary/native Graph arguments");
        has_execution = true;
        bool declared_core_only = false;
        std::map<uint64_t, RecordUse> records;
        std::vector<SdmaUse> sdma_uses;
        bool declared_wait = false;
        if (actual.uses != nullptr)
            for (const auto &use : *actual.uses) {
                if (!async_use_declaration_valid(use, actual.tensors.size(), actual.scalar_count))
                    return fail(error, "async declaration no longer matches this invocation's argument positions");
                if (use.kind == AsyncUseDeclarationKind::CONSUMER_CLEAN_READ) continue;
                if (use.kind == AsyncUseDeclarationKind::CORE_PIPELINES_ONLY) {
                    declared_core_only = true;
                    continue;
                }
                if (use.kind == AsyncUseDeclarationKind::SDMA_TRANSFER) {
                    declared_wait = true;
                    if (!bind_sdma_use(builder, actual, use, byte_budget, &sdma_uses, error)) return false;
                    continue;
                }
                if (use.cache_contract != AsyncUseCacheContract::PUBLISHED_CLEAN_LINES)
                    return fail(error, "async use lacks an explicit PUBLISHED_CLEAN_LINES author/compiler contract");
                std::vector<uint64_t> use_payloads;
                if (use.kind != AsyncUseDeclarationKind::COUNTER_ONLY) {
                    std::vector<Interval> runs;
                    if (!tensor_runs(actual.tensors[use.payload_arg], byte_budget, &runs))
                        return fail(error, "async payload has invalid or unbudgeted exact view geometry");
                    for (const auto span : runs) {
                        uint64_t index = 0;
                        if (!builder.range(span, ASYNC_RANGE_READ, &index))
                            return fail(
                                error,
                                "async payload/cache line has no retained owner for this preparation or exceeds budget"
                            );
                        use_payloads.push_back(index);
                    }
                }
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
        core_only &= declared_core_only;
        bool counter_lane = false, sdma_lane = false;
        for (uint32_t lane = 0; lane < 3; ++lane) {
            if (!meta.active_mask.has(static_cast<SubtaskSlot>(lane))) continue;
            const auto cap = core_async_effective_capability(capabilities[task][lane]);
            counter_lane |=
                cap.mode == CoreAsyncMode::DEFERRED && (cap.condition_types & CORE_ASYNC_CONDITION_COUNTER) != 0;
            sdma_lane |= lane != 0 && (cap.mode == CoreAsyncMode::SYNCHRONOUS ||
                                       (cap.condition_types & CORE_ASYNC_CONDITION_SDMA_POST_DONE) != 0);
        }
        if ((!records.empty() && !counter_lane) || (!sdma_uses.empty() && !sdma_lane))
            return fail(error, "declared conditions have no compatible registered execution lane");
        if (submission_image && !builder.budget.take(
                                    static_cast<uint64_t>(meta.logical_block_num) * meta.active_mask.count(),
                                    sizeof(SchedulerSubmissionExecutionUseSet)
                                ))
            return fail(error, "submission execution use maps exceed byte budget");
        std::map<std::vector<uint64_t>, std::pair<uint64_t, uint64_t>> sets;
        for (uint32_t block = 0; block < meta.logical_block_num; ++block)
            for (uint32_t lane = 0; lane < 3; ++lane) {
                if (!meta.active_mask.has(static_cast<SubtaskSlot>(lane))) continue;
                const auto capability = core_async_effective_capability(capabilities[task][lane]);
                const bool deferred = capability.mode == CoreAsyncMode::DEFERRED;
                const bool counter = deferred && (capability.condition_types & CORE_ASYNC_CONDITION_COUNTER) != 0;
                const bool sdma =
                    lane != 0 && (capability.mode == CoreAsyncMode::SYNCHRONOUS ||
                                  (capability.condition_types & CORE_ASYNC_CONDITION_SDMA_POST_DONE) != 0);
                const uint64_t use_count = (counter ? records.size() : 0) + (sdma ? sdma_uses.size() : 0);
                if (use_count > UINT32_MAX) return fail(error, "allowed-use set exceeds its uint32 index contract");
                if (!builder.budget.take(use_count, sizeof(AsyncConditionContract)))
                    return fail(error, "per-execution allowed-use sets exceed byte budget");
                const uint64_t begin = plan.conditions.size();
                for (const auto &[address, record] : records) {
                    if (!counter) break;
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
                if (sdma)
                    for (const auto &use : sdma_uses) {
                        const auto &workspace = plan.submission_workspaces[use.workspace_input];
                        plan.conditions.push_back(
                            {static_cast<int64_t>(task), block, lane,
                             static_cast<uint32_t>(plan.conditions.size() - begin), ASYNC_CORE_AIC | ASYNC_CORE_AIV,
                             COMPLETION_TYPE_SDMA_EVENT_RECORD, 0, ASYNC_RESOURCE_NONE, 0, 0, use.source_range,
                             use.payload_range, 1, workspace.address}
                        );
                        if (!builder.budget.take(use.context_scalars.size(), sizeof(SchedulerSubmissionExecutionUse)))
                            return fail(error, "SDMA submission request inputs exceed byte budget");
                        for (auto context : use.context_scalars)
                            plan.submission_uses.push_back(
                                {static_cast<int64_t>(task), block, lane, context, use.workspace_input}
                            );
                    }
                if (submission_image) {
                    const uint64_t count = plan.conditions.size() - begin;
                    plan.execution_use_sets.push_back(
                        {static_cast<int64_t>(task), block, lane, count == 0 ? SCHEDULER_ASYNC_INDEX_NONE : begin,
                         count}
                    );
                }
                if (deferred && !declared_wait && sdma) plan.needs_default_sdma_guards = true;
                if (deferred)
                    plan.groups.push_back(
                        {declared_wait ? begin : SCHEDULER_ASYNC_USES_DYNAMIC, plan.conditions.size() - begin}
                    );
            }
    }
    plan.execution_contract = core_only && has_execution && !submission_image ? ASYNC_EXECUTION_CORE_PIPELINES_ONLY :
                                                                                ASYNC_EXECUTION_UNQUALIFIED;
    plan.auxiliary.resources = sizeof(AsyncResourceHeader) + plan.spans.size() * sizeof(AsyncRetainedSpan) +
                               plan.ranges.size() * sizeof(AsyncResourceRange);
    plan.auxiliary.uses = plan.conditions.size() * sizeof(AsyncConditionContract);
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

bool scheduler_has_sdma_uses(
    const std::vector<SchedulerTaskMetadata> &metadata, const NativeGraphHostPlan &native,
    const AsyncHostDeclarations &declarations, bool *out, std::string *error
) {
    if (out == nullptr || metadata.size() > INT32_MAX ||
        (!native.instances.empty() && native.bindings.size() != metadata.size()))
        return fail(error, "submission declarations require the actual expanded task map");
    bool found = false;
    for (uint32_t task = 0; task < metadata.size(); ++task) {
        if (!scheduler_task_is_executable(metadata[task].flags) || scheduler_task_is_inline(metadata[task].flags))
            continue;
        const AsyncHostDeclarations::Uses *uses = nullptr;
        if (native.instances.empty()) uses = declarations.find(false, 0, task);
        else {
            const auto &binding = native.bindings[task];
            if (binding.instance_index == SCHEDULER_NATIVE_NO_INSTANCE)
                uses = declarations.find(false, 0, binding.source_index);
            else {
                if (binding.instance_index >= native.instances.size())
                    return fail(error, "submission declaration references an invalid Graph instance");
                const auto definition = native.instances[binding.instance_index].definition_index;
                if (definition >= native.definitions.size() || binding.kind != SchedulerNodeKind::CHILD_KERNEL)
                    return fail(error, "submission declaration references an invalid Graph definition");
                uses =
                    declarations.find(true, native.definitions[definition].definition.graph_key, binding.source_index);
            }
        }
        if (uses != nullptr)
            for (const auto &use : *uses)
                found |= use.kind == AsyncUseDeclarationKind::SDMA_TRANSFER;
    }
    *out = found;
    return true;
}

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
    uint64_t resources = sizeof(AsyncResourceHeader), uses = 0;
    if (!async_execution_contract_valid(plan.execution_contract) ||
        (plan.execution_contract == ASYNC_EXECUTION_CORE_PIPELINES_ONLY &&
         (!plan.submission_uses.empty() ||
          std::any_of(plan.conditions.begin(), plan.conditions.end(), [](const auto &use) {
              return use.completion_type == COMPLETION_TYPE_SDMA_EVENT_RECORD;
          }))))
        return fail(error, "async execution contract is invalid or contradicts its submission permissions");
    if (!async_resource_detail::add_rows(plan.spans.size(), sizeof(AsyncRetainedSpan), resources) ||
        !async_resource_detail::add_rows(plan.ranges.size(), sizeof(AsyncResourceRange), resources) ||
        !async_resource_detail::add_rows(plan.conditions.size(), sizeof(AsyncConditionContract), uses) ||
        resources != plan.auxiliary.resources || uses != plan.auxiliary.uses)
        return fail(error, "async binding vectors changed after sizing");
    if (region == nullptr || layout.total_bytes < sizeof(SchedulerAsyncLayout) || region_bytes < layout.total_bytes ||
        plan.run_epoch != layout.run_epoch || plan.groups.size() != layout.group_count ||
        plan.auxiliary.resources != layout.resources_bytes || plan.auxiliary.uses != layout.uses_bytes)
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
          Interval{layout.resources_offset, layout.resources_bytes}, Interval{layout.uses_offset, layout.uses_bytes}}) {
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
        if (group.begin == SCHEDULER_ASYNC_USES_DYNAMIC ?
                group.count != 0 :
                (group.begin > plan.conditions.size() || group.count > plan.conditions.size() - group.begin))
            return fail(error, "async group allowed-use range is invalid");
    auto *bytes = static_cast<std::byte *>(region);
    const auto copy = [&](uint64_t offset, const auto &values) {
        using Value = typename std::decay_t<decltype(values)>::value_type;
        if (!values.empty()) std::memcpy(bytes + offset, values.data(), values.size() * sizeof(Value));
    };
    AsyncResourceHeader header{ASYNC_RESOURCE_VERSION, plan.run_epoch,         plan.spans.size(),
                               plan.ranges.size(),     plan.conditions.size(), plan.execution_contract};
    std::memcpy(bytes + layout.resources_offset, &header, sizeof(header));
    copy(layout.resources_offset + sizeof(header), plan.spans);
    copy(layout.resources_offset + sizeof(header) + plan.spans.size() * sizeof(AsyncRetainedSpan), plan.ranges);
    copy(layout.uses_offset, plan.conditions);
    for (uint64_t index = 0; index < plan.groups.size(); ++index) {
        SchedulerAsyncGroupIdentity identity{};
        auto *record = bytes + layout.group_identity_offset + index * sizeof(identity);
        std::memcpy(&identity, record, sizeof(identity));
        identity.binding_use_count = plan.groups[index].count;
        identity.binding_uses_offset =
            plan.groups[index].begin == SCHEDULER_ASYNC_USES_DYNAMIC ?
                SCHEDULER_ASYNC_USES_DYNAMIC :
            identity.binding_use_count == 0 ?
                0 :
                layout.uses_offset + plan.groups[index].begin * sizeof(AsyncConditionContract);
        std::memcpy(record, &identity, sizeof(identity));
    }
    return true;
}
