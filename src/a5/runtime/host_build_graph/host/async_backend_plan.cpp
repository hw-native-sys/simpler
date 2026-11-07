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

#include "async_backend_plan.h"

#include <algorithm>
#include <cstring>
#include <map>
#include <new>
#include <stdexcept>

#include "host_build_graph/runtime_types.h"
#include "host_build_graph/async_retirement.h"
#include "scheduler/scheduler_graph_binding.h"

namespace {
struct Interval {
    uint64_t address{0}, bytes{0};
};

bool contains(Interval span, Interval requested) {
    return async_backend_detail::contains(span.address, span.bytes, requested.address, requested.bytes);
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
const T *within(const AsyncBackendTaskImage &image, const T *pointer, uint64_t count = 1) {
    if (count > UINT64_MAX / sizeof(T)) return nullptr;
    const Interval storage{reinterpret_cast<uint64_t>(image.host_base), image.bytes};
    return contains(storage, {reinterpret_cast<uint64_t>(pointer), count * sizeof(T)}) ? pointer : nullptr;
}

const ChipTaskStorage *task_storage(const AsyncBackendTaskImage &image, uint32_t index) {
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

bool task_uses(
    uint32_t task, const NativeGraphHostPlan &native, const AsyncHostDeclarations &declarations,
    const AsyncHostDeclarations::Uses **out
) {
    if (native.instances.empty()) {
        *out = declarations.find(false, 0, task);
        return true;
    }
    if (task >= native.bindings.size()) return false;
    const auto &binding = native.bindings[task];
    if (binding.instance_index == SCHEDULER_NATIVE_NO_INSTANCE) {
        *out = declarations.find(false, 0, binding.source_index);
        return true;
    }
    if (binding.instance_index >= native.instances.size() || binding.kind != SchedulerNodeKind::CHILD_KERNEL)
        return false;
    const auto definition = native.instances[binding.instance_index].definition_index;
    if (definition >= native.definitions.size()) return false;
    *out = declarations.find(true, native.definitions[definition].definition.graph_key, binding.source_index);
    return true;
}

bool decode_task(
    uint32_t task, const AsyncBackendTaskImage &image, const NativeGraphHostPlan &native, TaskArguments *out
) {
    if (image.host_base == nullptr || image.bytes > UINT64_MAX - reinterpret_cast<uint64_t>(image.host_base) ||
        image.bytes > UINT64_MAX - image.device_base)
        return false;
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

struct Workspace {
    uint64_t address, bytes, queue_num;
};
struct Builder {
    const HostApi &api;
    Budget budget;
    std::vector<Interval> owners;
    std::vector<Workspace> workspaces;

    bool validate(Interval logical) {
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
        const Interval retained{owner.base, owner.bytes};
        for (const auto existing : owners) {
            if (existing.address == retained.address && existing.bytes == retained.bytes) return true;
            if (overlaps(existing, retained)) return false;
        }
        if (!budget.take(1, sizeof(Interval))) return false;
        owners.push_back(retained);
        return true;
    }
};

bool bind_sdma_use(
    Builder &builder, const TaskArguments &actual, const AsyncUseDeclaration &use, uint64_t byte_budget,
    std::vector<AsyncSdmaTransfer> *out, std::string *error
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
        owner.queue_num == 0 || owner.queue_num > ASYNC_SDMA_CHANNELS)
        return fail(error, "SDMA context lacks selected exclusive domain and fixed SDK-only workspace owner");
    const auto &comm = domain.context;
    const auto &workspace = comm_workspace<CommWorkspaceKind::SDMA>(comm);
    if (peer >= comm.rankNum || comm.windowsIn[peer] == 0 || workspace.addr == 0 ||
        workspace.addr != owner.workspace_addr || workspace.size != owner.workspace_bytes || workspace.addr % 64 != 0 ||
        workspace.size < (ASYNC_SDMA_CHANNELS + 1) * 64)
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
    if (!builder.validate(remote) || !builder.validate(destinations[0]))
        return fail(error, "SDMA source/destination cache lines lack an existing retained owner");
    auto &workspaces = builder.workspaces;
    const auto saved_workspace = std::find_if(workspaces.begin(), workspaces.end(), [&](const Workspace &saved) {
        return saved.address == workspace.addr;
    });
    if (saved_workspace == workspaces.end()) {
        if (!builder.budget.take(1, sizeof(Workspace)))
            return fail(error, "SDMA workspace declarations exceed byte budget");
        workspaces.push_back({workspace.addr, workspace.size, owner.queue_num});
    } else if (saved_workspace->bytes != workspace.size || saved_workspace->queue_num != owner.queue_num) {
        return fail(error, "one SDMA workspace owner has conflicting retained configuration");
    }
    const auto destination = destinations[0];
    for (const auto &saved : *out) {
        if (saved.workspace != workspace.addr) continue;
        if (saved.source == remote.address && saved.source_bytes == remote.bytes &&
            saved.destination == destination.address && saved.destination_bytes == destination.bytes)
            return true;
        if (overlaps({saved.source, saved.source_bytes}, remote) &&
            overlaps({saved.destination, saved.destination_bytes}, destination))
            return fail(error, "SDMA declarations admit an ambiguous transfer within one workspace");
    }
    if (!builder.budget.take(1, sizeof(AsyncSdmaTransfer)))
        return fail(error, "SDMA transfer declarations exceed byte budget");
    out->push_back(
        {workspace.addr, workspace.size, remote.address, remote.bytes, destination.address, destination.bytes,
         owner.queue_num}
    );
    return true;
}

bool build(
    const std::vector<SchedulerTaskMetadata> &metadata, const std::vector<SchedulerAsyncCapabilities> &capabilities,
    const AsyncBackendTaskImage &image, const NativeGraphHostPlan &native, const AsyncHostDeclarations &declarations,
    const HostApi &api, uint64_t byte_budget, AsyncBackendHostPlan *out, std::string *error
) {
    if (out == nullptr || api.run_epoch() == 0 || metadata.size() != capabilities.size() ||
        (!native.instances.empty() && native.bindings.size() != metadata.size()))
        return fail(error, "async backend planning requires final arguments and a live native preparation");
    AsyncBackendHostPlan plan;
    plan.run_epoch = api.run_epoch();
    Builder builder{api, Budget{byte_budget}, {}, {}};
    std::vector<AsyncBackendTask> tasks;
    std::vector<AsyncCounterRange> counters;
    std::vector<AsyncSdmaTransfer> transfers;
    bool core_only = true;
    bool has_execution = false;
    for (uint32_t task = 0; task < metadata.size(); ++task) {
        const auto &meta = metadata[task];
        if (!scheduler_task_is_executable(meta.flags) || scheduler_task_is_inline(meta.flags)) continue;
        TaskArguments actual{};
        if (!task_uses(task, native, declarations, &actual.uses))
            return fail(error, "async declaration references an invalid native Graph task");
        if (actual.uses != nullptr && !decode_task(task, image, native, &actual))
            return fail(error, "async backend cannot decode final ordinary/native Graph arguments");
        has_execution = true;
        bool declared_core_only = false;
        std::map<uint64_t, bool> records;
        std::vector<AsyncSdmaTransfer> sdma_uses;
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
                if (use.kind != AsyncUseDeclarationKind::COUNTER_ONLY) {
                    std::vector<Interval> runs;
                    if (!tensor_runs(actual.tensors[use.payload_arg], byte_budget, &runs))
                        return fail(error, "async payload has invalid or unbudgeted exact view geometry");
                    for (const auto span : runs) {
                        if (!builder.validate(span))
                            return fail(
                                error,
                                "async payload/cache line has no retained owner for this preparation or exceeds budget"
                            );
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
                    if (count > byte_budget / sizeof(std::pair<const uint64_t, bool>))
                        return fail(error, "COUNTER allowed-record set exceeds byte budget");
                    for (uint64_t n = 0; n < count; ++n) {
                        const uint64_t address = span.address + skip + n * use.record_stride;
                        if (!builder.validate({address, sizeof(int32_t)}))
                            return fail(error, "COUNTER record/cache line has no retained owner or exceeds budget");
                        const bool only = use.kind == AsyncUseDeclarationKind::COUNTER_ONLY;
                        auto [entry, inserted] = records.try_emplace(address, only);
                        if (!inserted && entry->second != only)
                            return fail(error, "aliased COUNTER selectors disagree on control-only/payload semantics");
                        if (inserted && !builder.budget.take(1, sizeof(std::pair<const uint64_t, bool>)))
                            return fail(error, "COUNTER allowed-record set exceeds byte budget");
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
        for (uint32_t lane = 1; lane < 3; ++lane) {
            if (!meta.active_mask.has(static_cast<SubtaskSlot>(lane))) continue;
            const auto cap = core_async_effective_capability(capabilities[task][lane]);
            plan.needs_sdma_guards |= !declared_wait && !declared_core_only &&
                                      (cap.mode == CoreAsyncMode::SYNCHRONOUS ||
                                       (cap.condition_types & CORE_ASYNC_CONDITION_SDMA_POST_DONE) != 0);
        }
        if (!declared_wait) continue;
        AsyncBackendTask contract{task, counters.size(), 0, transfers.size(), 0};
        for (const auto &[address, record] : records) {
            (void)record;
            if (contract.counter_count != 0) {
                auto &previous = counters.back();
                const uint64_t delta = address - previous.address;
                if (previous.count == 1) {
                    previous.stride = delta;
                    previous.count = 2;
                    continue;
                }
                if (previous.count <= UINT64_MAX / previous.stride && delta == previous.count * previous.stride) {
                    ++previous.count;
                    continue;
                }
            }
            counters.push_back({address, 1, sizeof(int32_t)});
            ++contract.counter_count;
        }
        contract.transfer_count = sdma_uses.size();
        transfers.insert(transfers.end(), sdma_uses.begin(), sdma_uses.end());
        plan.needs_sdma_guards |= !sdma_uses.empty();
        tasks.push_back(contract);
    }
    plan.execution_contract = core_only && has_execution && transfers.empty() ? HBG_ASYNC_CONTRACT_CORE_PIPELINES_ONLY :
                                                                                HBG_ASYNC_CONTRACT_UNQUALIFIED;
    if (!tasks.empty()) {
        AsyncBackendLayout layout{};
        layout.task_count = tasks.size();
        layout.counter_count = counters.size();
        layout.transfer_count = transfers.size();
        uint64_t cursor = sizeof(layout);
        auto append = [&](uint64_t count, uint64_t stride, uint64_t &offset) {
            if (count == 0) return true;
            if (cursor > byte_budget || count > (byte_budget - cursor) / stride) return false;
            offset = cursor;
            cursor += count * stride;
            return true;
        };
        if (!append(tasks.size(), sizeof(AsyncBackendTask), layout.tasks_offset) ||
            !append(counters.size(), sizeof(AsyncCounterRange), layout.counters_offset) ||
            !append(transfers.size(), sizeof(AsyncSdmaTransfer), layout.transfers_offset))
            return fail(error, "async backend declarations exceed byte budget");
        layout.total_bytes = cursor;
        plan.image.resize(cursor);
        std::memcpy(plan.image.data(), &layout, sizeof(layout));
        const auto copy = [&](uint64_t offset, const auto &rows) {
            using T = typename std::decay_t<decltype(rows)>::value_type;
            if (!rows.empty()) std::memcpy(plan.image.data() + offset, rows.data(), rows.size() * sizeof(T));
        };
        copy(layout.tasks_offset, tasks);
        copy(layout.counters_offset, counters);
        copy(layout.transfers_offset, transfers);
    }
    *out = std::move(plan);
    return true;
}
}  // namespace

bool async_backend_has_sdma_uses(
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
        if (!task_uses(task, native, declarations, &uses))
            return fail(error, "submission declaration references an invalid native Graph task");
        if (uses != nullptr)
            for (const auto &use : *uses)
                found |= use.kind == AsyncUseDeclarationKind::SDMA_TRANSFER;
    }
    *out = found;
    return true;
}

bool scheduler_plan_async_backend(
    const std::vector<SchedulerTaskMetadata> &metadata, const std::vector<SchedulerAsyncCapabilities> &capabilities,
    const AsyncBackendTaskImage &image, const NativeGraphHostPlan &native, const AsyncHostDeclarations &declarations,
    const HostApi &api, uint64_t byte_budget, AsyncBackendHostPlan *out, std::string *error
) {
    try {
        return build(metadata, capabilities, image, native, declarations, api, byte_budget, out, error);
    } catch (const std::bad_alloc &) {
        return fail(error, "cannot allocate async backend declarations");
    } catch (const std::length_error &) {
        return fail(error, "async backend declarations exceed container capacity");
    }
}
