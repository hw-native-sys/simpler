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
#include "launch.h"
#include "sram_layout.h"
#include "wse/block_lifecycle.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <new>
#include <string>
#include "cpu_sim_context.h"
#include "host/raii_scope_guard.h"
#include "host_build_graph/submit_types.h"
#include "scheduler/scheduler_graph_binding.h"

#include <atomic>
#include <cstdio>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace {
class ImageBuilder {
public:
    explicit ImageBuilder(const wse::SramRegion *regions, uint32_t count) :
        regions_(regions, regions + count) {
        for (const auto &r : regions_)
            cursor_.push_back(r.reserve);
    }
    template <class T>
    uint64_t allocate(uint32_t core, uint64_t count, WseSectionKind kind) {
        const auto &r = regions_.at(core);
        if (count > SIZE_MAX / sizeof(T)) throw std::invalid_argument("WSE SRAM layout multiplication overflow");
        const uint64_t bytes = count * sizeof(T);
        if (!bytes) return 0;
        if (bytes > cursor_[core]) exhausted(core, bytes, kind);
        const auto offset = (cursor_[core] - bytes) & ~(uint64_t{alignof(T)} - 1);
        cursor_[core] = offset;
        sections.push_back({r.runtime_base + offset, bytes, core, kind});
        auto *objects = reinterpret_cast<T *>(r.runtime_base + offset);
        for (uint64_t i = 0; i < count; ++i)
            new (objects + i) T{};
        return r.runtime_base - r.base + offset;
    }
    template <class T>
    T *at(uint32_t core, uint64_t offset) {
        return reinterpret_cast<T *>(regions_[core].base + offset);
    }
    template <class T>
    T *objects(uint32_t core, uint64_t count, WseSectionKind kind) {
        return count ? at<T>(core, allocate<T>(core, count, kind)) : nullptr;
    }
    std::vector<WseMemorySection> sections;
    std::vector<uint64_t> usage() const {
        std::vector<uint64_t> result;
        result.reserve(regions_.size());
        for (size_t i = 0; i < regions_.size(); ++i)
            result.push_back(regions_[i].reserve - cursor_[i]);
        return result;
    }

private:
    [[noreturn]] void exhausted(uint32_t core, uint64_t required, WseSectionKind kind) const {
        throw std::runtime_error(
            "WSE Runtime SRAM capacity exhausted: core=" + std::to_string(core) + " section=" + wse_section_name(kind) +
            " required=" + std::to_string(required) + " available=" + std::to_string(cursor_[core]) +
            " reserve=" + std::to_string(regions_[core].reserve)
        );
    }
    std::vector<wse::SramRegion> regions_;
    std::vector<uint64_t> cursor_;
};

const std::byte *descriptor_region(
    const WseGraphDescriptor &descriptor, uint64_t offset, uint64_t count, uint64_t stride, uint64_t alignment
) {
    if (count == 0 && offset == 0) return nullptr;
    if (offset < sizeof(WseGraphDescriptor) || offset > descriptor.descriptor_bytes || offset % alignment ||
        count > (descriptor.descriptor_bytes - offset) / stride)
        throw std::invalid_argument("invalid WSE descriptor table");
    return reinterpret_cast<const std::byte *>(&descriptor) + offset;
}

template <class T>
const T *descriptor_table(const WseGraphDescriptor &descriptor, uint64_t offset, uint64_t count) {
    return reinterpret_cast<const T *>(descriptor_region(descriptor, offset, count, sizeof(T), alignof(T)));
}

struct TaskCalendar {
    std::vector<uint32_t> row_by_block;
    uint64_t first_tick = UINT64_MAX;
    uint64_t finish_tick = 0;
};

// A logical Graph boundary has no Block placement of its own. Choose the closest
// explicitly placed neighbor in the dependency DAG (ties follow node ID). A
// component with no kernel has no device work or SRAM owner and settles at bind.
std::vector<uint32_t>
task_owners(const WseGraphDescriptor &d, const WseNodePlan *plans, uint32_t count, uint32_t cores) {
    std::vector<uint32_t> owners(count, UINT32_MAX), queue;
    std::vector<std::vector<uint32_t>> neighbors(count);
    for (uint32_t ref = 0; ref < count; ++ref) {
        const auto &plan = plans[ref];
        if (plan.host_done && plan.wait_count) throw std::invalid_argument("host-completed WSE task has waits");
        if (plan.entry) {
            const auto *blocks = descriptor_table<WseBlockPlan>(d, plan.block_plans_offset, plan.block_plan_count);
            if (blocks[0].core_id >= cores) throw std::invalid_argument("invalid WSE Block Core");
            owners[ref] = blocks[0].core_id;
            queue.push_back(ref);
        }
        const auto *waits = descriptor_table<uint32_t>(d, plan.waits_offset, plan.wait_count);
        for (uint32_t i = 0; i < plan.wait_count; ++i) {
            const uint32_t producer = waits[i];
            if (producer >= ref || plans[producer].host_done) throw std::invalid_argument("invalid WSE calendar wait");
            if (plan.entry && plans[producer].entry) continue;
            neighbors[ref].push_back(producer);
            neighbors[producer].push_back(ref);
        }
    }
    for (size_t i = 0; i < queue.size(); ++i) {
        const uint32_t ref = queue[i];
        for (uint32_t neighbor : neighbors[ref]) {
            if (owners[neighbor] != UINT32_MAX) continue;
            owners[neighbor] = owners[ref];
            queue.push_back(neighbor);
        }
    }
    return owners;
}

struct RunAborted {};
}  // namespace

struct WseRun {
    // The simulator binds worker IDs to actual Core SRAM. All graph-sized
    // execution state lives in the Core records, queues and signals below.
    std::vector<wse::CoreControl *> core_controls;
    const uint64_t expected_tasks;
    const uint64_t application_base;  // Kernel ABI origin for the whole application Buffer.
    std::atomic<uint64_t> completed{0};
    uint32_t ready_pes = 0, retired_cores = 0;
    bool configured = false, go = false, stop = false, failed = false;
    std::vector<WseEvent> host_events;
    std::mutex mutex;
    std::condition_variable changed;
    std::atomic<uint64_t> sequence{0};
    std::vector<std::vector<WseEvent>> events;
    std::vector<uint64_t> used;
    std::vector<std::unique_ptr<wse::BlockOperationStorage>> block_storage;
    std::vector<WseMemorySection> sections;

    template <class Predicate>
    void wait(std::unique_lock<std::mutex> &lock, Predicate ready) {
        changed.wait(lock, [&] {
            return failed || ready();
        });
        if (failed) throw RunAborted{};
    }

    void abort(uint32_t worker, const char *reason) {
        {
            std::lock_guard lock(mutex);
            if (failed) return;
            failed = true;
        }
        changed.notify_all();
        std::fprintf(stderr, "WSE PE %u failed: %s\n", worker, reason);
        for (auto &storage : block_storage)
            storage->cancel();
    }

    wse::CoreControl &local(uint32_t core) const { return *core_controls[core]; }
    uint64_t completed_tasks() const { return completed.load(); }
    wse::Work &head(uint32_t core) const {
        const auto &c = local(core);
        return reinterpret_cast<wse::Work *>(c.base + c.queue.work_offset)[c.queue.head];
    }
    bool dependencies_ready(const wse::TaskRecord &task) const {
        for (uint32_t i = 0; i < task.predecessor_count; ++i)
            if (!task.predecessors[i].signal->completed) return false;
        return true;
    }
    void complete_task(uint32_t worker, wse::TaskRecord &task) {
        if (task.signal->completed) throw std::logic_error("WSE duplicate Graph-node completion");
        task.signal->completed = 1;
        record(worker, WseEventKind::TASK_DONE, task.ref, 0, ++completed);
        changed.notify_all();
    }
    void advance_controls(uint32_t worker) {
        auto &first = local(worker / WSE_PE_PER_CORE).first_control;
        // A blocked exit cannot trap an independent entry behind it. Unlink
        // completed boundaries so later wake-ups only visit outstanding work.
        for (auto **link = &first; *link;) {
            auto &task = **link;
            if (dependencies_ready(task)) {
                *link = task.next_control;
                complete_task(worker, task);
            } else {
                link = &task.next_control;
            }
        }
    }
    int ready_thread(uint32_t worker) {
        advance_controls(worker);
        const uint32_t core = worker / WSE_PE_PER_CORE, pe = worker % WSE_PE_PER_CORE;
        auto &c = local(core);
        if (c.queue.head == c.queue.tail) return -1;
        const auto &work = head(core);
        auto &task = *work.task;
        auto &signal = *task.signal;
        if (c.block_state != wse::BlockState::RUNNING && pe == 0 && dependencies_ready(task)) {
            if (signal.remaining_arrivals) {
                if (c.block_state == wse::BlockState::WAITING_FOR_SYNC) return -1;
                c.block_state = wse::BlockState::WAITING_FOR_SYNC;
                record(worker, WseEventKind::BLOCK_READY, task.ref, work.block);
                if (--signal.remaining_arrivals) return -1;
                record(worker, WseEventKind::SYNC_RELEASE, task.ref, work.block, task.blocks);
            }
            c.block_state = wse::BlockState::RUNNING;
            c.started = c.finished = 0;
            c.execute = !task.has_predicate ||
                        DispatchPredicate{
                            task.predicate.addr, task.predicate.target, task.predicate.elem_size,
                            static_cast<PredicateOp>(task.predicate.op)
                        }
                            .pass();
            __atomic_store_n(&c.block_epoch, ++c.block_serial, __ATOMIC_RELEASE);
            record(worker, WseEventKind::BLOCK_BEGIN, task.ref, work.block, work.thread_num);
            if (!c.execute) record(worker, WseEventKind::BLOCK_SKIPPED, task.ref, work.block);
            changed.notify_all();
        }
        if (c.block_state != wse::BlockState::RUNNING) return -1;
        for (uint32_t t = 0; t < work.thread_num; ++t)
            if (work.thread_pes[t] == pe && !(c.started & (1u << t))) return t;
        return -1;
    }

    WseRun(
        const WseGraphDescriptor *d, uint32_t cores, uint64_t epoch, const wse::SramRegion *storage,
        uint32_t region_count, const wse::SramAllocation *allocations, uint32_t allocation_count
    ) :
        expected_tasks(d->expected_tasks),
        application_base(storage->base),
        events(cores * WSE_PE_PER_CORE) {
        for (uint32_t core = 0; core < cores; ++core)
            block_storage.push_back(std::make_unique<wse::BlockOperationStorage>());
        const SchedulerGraphAccess graph(d->graph);
        if (!scheduler_graph_view_valid(graph)) throw std::invalid_argument("invalid WSE graph");
        const auto count = graph.task_count;
        const auto *source_plans =
            descriptor_table<WseNodePlan>(*d, d->nodes_address - reinterpret_cast<uintptr_t>(d), count);
        if (d->calendar_core_count != cores) throw std::invalid_argument("WSE calendar Core count mismatch");
        const auto *indices = descriptor_table<uint32_t>(*d, d->calendar_indices_offset, cores + 1);
        const auto *rows = descriptor_table<wse::CalendarRow>(*d, d->calendar_rows_offset, d->calendar_row_count);
        if (indices[0] != 0 || indices[cores] != d->calendar_row_count)
            throw std::invalid_argument("invalid WSE calendar boundaries");
        const auto *pe_indices =
            descriptor_table<uint32_t>(*d, d->pe_calendar_indices_offset, cores * WSE_PE_PER_CORE + 1);
        const auto *pe_rows = descriptor_table<uint32_t>(*d, d->pe_calendar_rows_offset, d->pe_calendar_row_count);
        if (pe_indices[0] || pe_indices[cores * WSE_PE_PER_CORE] != d->pe_calendar_row_count ||
            !std::is_sorted(pe_indices, pe_indices + cores * WSE_PE_PER_CORE + 1))
            throw std::invalid_argument("invalid WSE PE calendar boundaries");
        uint64_t expected_rows = 0;
        for (uint32_t ref = 0; ref < count; ++ref) {
            const auto &plan = source_plans[ref];
            if (plan.sync_start > 1 ||
                (plan.sync_start && (!plan.entry || plan.blocks < 2 || plan.blocks > cores || plan.has_predicate)))
                throw std::invalid_argument("invalid WSE synchronous SPMD task");
            if (plan.entry) {
                if (plan.host_done || !plan.blocks || plan.block_plan_count != plan.blocks)
                    throw std::invalid_argument("WSE requires a placement for every kernel Block");
                expected_rows += plan.blocks;
            } else if (plan.block_plan_count || plan.block_plans_offset) {
                throw std::invalid_argument("WSE control nodes cannot carry Block placements");
            }
        }
        if (expected_rows != d->calendar_row_count) throw std::invalid_argument("WSE calendar Block count mismatch");
        ImageBuilder image(storage, region_count);
        const auto owners = task_owners(*d, source_plans, count, cores);
        std::vector<wse::TaskSignal *> signals(count);
        std::vector<wse::TaskRecord *> control_tails(cores);
        std::vector<TaskCalendar> calendar(count);
        for (uint32_t core = 0; core < cores; ++core) {
            const uint32_t begin = indices[core], end = indices[core + 1];
            const auto *pe_ranges = pe_indices + core * WSE_PE_PER_CORE;
            std::array<uint32_t, WSE_PE_PER_CORE> cursor;
            std::copy_n(pe_ranges, WSE_PE_PER_CORE, cursor.begin());
            if (begin > end || end > d->calendar_row_count)
                throw std::invalid_argument("invalid WSE Core calendar range");
            auto *c = image.objects<wse::CoreControl>(core, 1, WseSectionKind::CORE_CONTROL);
            core_controls.push_back(c);
            c->epoch = epoch;
            c->base = storage[core].base;
            auto &q = c->queue;
            q.capacity = q.tail = end - begin;
            q.work_offset = image.allocate<wse::Work>(core, q.capacity, WseSectionKind::BLOCK_QUEUE);
            if (q.capacity) {
                c->access.region_count = region_count;
                auto *regions = image.objects<wse::SramRegion>(core, region_count, WseSectionKind::REGION_MAP);
                c->access.regions_offset = reinterpret_cast<uint64_t>(regions) - application_base;
                std::copy_n(storage, region_count, regions);
            }
            for (uint32_t i = begin; i < end; ++i) {
                const auto &row = rows[i];
                if (row.task >= count || (i > begin && row.tick <= rows[i - 1].tick))
                    throw std::invalid_argument("invalid WSE calendar task or Core order");
                const auto &plan = source_plans[row.task];
                if (!plan.entry || plan.host_done || plan.blocks > INT16_MAX || row.block >= plan.blocks)
                    throw std::invalid_argument("invalid WSE calendar Block");
                const auto *blocks = descriptor_table<WseBlockPlan>(*d, plan.block_plans_offset, plan.block_plan_count);
                const auto &placement = blocks[row.block];
                if (!placement.thread_num || placement.thread_num > WSE_PE_PER_CORE || placement.core_id != core)
                    throw std::invalid_argument("invalid WSE calendar Block placement");
                uint32_t occupied = 0;
                for (uint32_t t = 0; t < placement.thread_num; ++t) {
                    const uint32_t pe = placement.thread_pes[t];
                    if (pe >= WSE_PE_PER_CORE || (occupied & (1u << pe)))
                        throw std::invalid_argument("WSE Block Threads must map to distinct local PEs");
                    occupied |= 1u << pe;
                    if (cursor[pe] == pe_ranges[pe + 1]) throw std::invalid_argument("missing WSE PE calendar row");
                    if (pe_rows[cursor[pe]++] != i)
                        throw std::invalid_argument("WSE PE calendar row disagrees with Block placement or order");
                }
                auto &task = calendar[row.task];
                if (task.row_by_block.empty()) task.row_by_block.resize(plan.blocks, UINT32_MAX);
                if (task.row_by_block[row.block] != UINT32_MAX)
                    throw std::invalid_argument("duplicate WSE calendar Block");
                task.row_by_block[row.block] = i;
                task.first_tick = std::min(task.first_tick, uint64_t{row.tick});
                task.finish_tick = std::max(task.finish_tick, uint64_t{row.tick} + 1);
            }
            if (!std::equal(cursor.begin(), cursor.end(), pe_ranges + 1))
                throw std::invalid_argument("extra WSE PE calendar row");
        }
        std::vector<wse::SramAllocation> used_allocations;
        auto validate_storage = [&](uint64_t address, uint64_t bytes) {
            if (bytes > UINT64_MAX - address) throw std::invalid_argument("WSE tensor address overflow");
            bool is_sram = false;
            for (uint32_t i = 0; i < region_count; ++i) {
                const auto &r = storage[i];
                if (address < r.base + r.capacity - r.reserve && (address >= r.base || address + bytes > r.base))
                    is_sram = true;
            }
            if (!is_sram) throw std::invalid_argument("WSE tensor must reside in SRAM");
            for (uint32_t i = 0; i < allocation_count; ++i) {
                const auto &a = allocations[i];
                if (address < a.base || address - a.base > a.bytes || bytes > a.bytes - (address - a.base)) continue;
                if (std::none_of(used_allocations.begin(), used_allocations.end(), [&](const auto &old) {
                        return old.base == a.base;
                    }))
                    used_allocations.push_back(a);
                return;
            }
            throw std::invalid_argument("WSE tensor names unallocated or private SRAM");
        };
        uint64_t expected = 0;
        for (uint64_t ref = 0; ref < count; ++ref) {
            const auto &source = source_plans[ref];
            if (source.host_done) continue;
            ++expected;
            if (!source.blocks || source.blocks > INT16_MAX)
                throw std::invalid_argument("invalid WSE task block count");
            if (owners[ref] == UINT32_MAX) {
                // No path to a Block: every predecessor in this component has
                // already settled in descriptor order. There is no PE dispatch.
                host_events.push_back(
                    {sequence++, ref, ++completed, UINT32_MAX, UINT32_MAX, 0, WseEventKind::TASK_DONE, 0, 0}
                );
                continue;
            }
            signals[ref] = image.objects<wse::TaskSignal>(owners[ref], 1, WseSectionKind::TASK_SIGNAL);
            signals[ref]->remaining_blocks = source.blocks;
            signals[ref]->remaining_arrivals = source.sync_start ? source.blocks : 0;
            const SchedulerDispatchPredicate *predicate = nullptr;
            if (source.entry && scheduler_bind_graph_arguments(graph, ref) != SchedulerGraphBindingError::NONE)
                throw std::invalid_argument("invalid WSE graph argument binding");
            if (source.entry) {
                SchedulerTaskPayloadArguments args{};
                if (scheduler_parse_task_payload_arguments(graph, ref, &args) != SchedulerGraphResult::OK)
                    throw std::invalid_argument("invalid WSE task arguments");
                if (source.has_predicate) {
                    predicate = scheduler_node_predicate(graph, ref);
                    if (!predicate || !scheduler_dispatch_predicate_metadata_valid(
                                          predicate->addr, predicate->elem_size, predicate->op
                                      ))
                        throw std::invalid_argument("invalid WSE predicate");
                }
            }
            const auto *waits = descriptor_table<uint32_t>(*d, source.waits_offset, source.wait_count);
            uint64_t ready_tick = 0;
            for (uint32_t i = 0; i < source.wait_count; ++i)
                ready_tick = std::max(ready_tick, calendar[waits[i]].finish_tick);
            auto make_local_task = [&](uint32_t core) {
                auto *task = image.objects<wse::TaskRecord>(core, 1, WseSectionKind::LOCAL_TASK);
                task->ref = ref;
                task->entry = source.entry;
                task->blocks = source.blocks;
                task->signal = signals[ref];
                task->predecessor_count = source.wait_count;
                task->predecessors =
                    image.objects<wse::TaskDependency>(core, source.wait_count, WseSectionKind::DEPENDENCIES);
                for (uint32_t i = 0; i < source.wait_count; ++i)
                    task->predecessors[i].signal = signals[waits[i]];
                if (predicate) {
                    task->has_predicate = true;
                    task->predicate = *predicate;
                    validate_storage(predicate->addr, predicate->elem_size);
                }
                return task;
            };
            // Controls advance at their predecessors' join. Kernel rows must be
            // reachable in both the Core order and the whole-task wait order.
            auto &task_calendar = calendar[ref];
            if (!source.entry) {
                const auto owner = owners[ref];
                auto *task = make_local_task(owner);
                auto *&tail = control_tails[owner];
                if (tail) tail->next_control = task;
                else local(owner).first_control = task;
                tail = task;
                task_calendar.finish_tick = ready_tick;
                continue;
            }
            if (task_calendar.row_by_block.size() != source.blocks || task_calendar.first_tick < ready_tick)
                throw std::invalid_argument("missing or premature WSE calendar task");
            // A Core's row ticks strictly increase. Equal ticks across all members therefore also require distinct
            // Cores.
            if (source.sync_start && task_calendar.finish_tick != task_calendar.first_tick + 1)
                throw std::invalid_argument("WSE sync-start Blocks require a common tick on distinct Cores");
            const auto *block_plans =
                descriptor_table<WseBlockPlan>(*d, source.block_plans_offset, source.block_plan_count);
            SchedulerNativeNode native_node{};
            const bool child = graph.native_header_offset && scheduler_native_node_checked(graph, ref, &native_node) &&
                               native_node.binding->kind == SchedulerNodeKind::CHILD_KERNEL;
            if (source.sync_start && child) throw std::invalid_argument("WSE sync-start requires an outer task list");
            std::vector<wse::TaskRecord *> local_tasks(cores);
            for (uint32_t b = 0; b < source.blocks; ++b) {
                const auto &placement = block_plans[b];
                const uint32_t core = placement.core_id;
                if (placement.tensor_count > CORE_MAX_TENSOR_ARGS || placement.scalar_count > CORE_MAX_SCALAR_ARGS)
                    throw std::invalid_argument("invalid WSE Block argument count");
                const uint32_t row = task_calendar.row_by_block[b];
                if (core >= cores || row < indices[core] || row >= indices[core + 1])
                    throw std::invalid_argument("WSE calendar Block missing or assigned to the wrong Core");
                auto &work = image.at<wse::Work>(core, local(core).queue.work_offset)[row - indices[core]];
                if (!local_tasks[core]) local_tasks[core] = make_local_task(core);
                work.task = local_tasks[core];
                work.block = b;
                work.thread_num = placement.thread_num;
                std::copy_n(placement.thread_pes, WSE_PE_PER_CORE, work.thread_pes);
                work.tensor_count = placement.tensor_count;
                work.scalar_count = placement.scalar_count;
                work.tensors_offset = image.allocate<Tensor>(core, work.tensor_count, WseSectionKind::TENSORS);
                work.scalars_offset = image.allocate<uint64_t>(core, work.scalar_count, WseSectionKind::SCALARS);
                const auto *tensor_bytes =
                    descriptor_region(*d, placement.tensors_offset, work.tensor_count, sizeof(Tensor), alignof(Tensor));
                for (uint32_t i = 0; i < work.tensor_count; ++i) {
                    auto &tensor = image.at<Tensor>(core, work.tensors_offset)[i];
                    std::memcpy(&tensor, tensor_bytes + i * sizeof(Tensor), sizeof(Tensor));
                    if (child) {
                        SchedulerTensorBinding bound{};
                        if (!scheduler_graph_relocate_tensor(tensor, native_node, &bound))
                            throw std::invalid_argument("invalid WSE Block Tensor Graph binding");
                        tensor.buffer.addr = bound.address;
                        tensor.buffer.size = bound.bytes;
                        tensor.start_offset = bound.origin;
                        tensor.address_space = bound.address_space;
                    }
                    uint64_t address = 0, bytes = 0;
                    if (!wse::tensor_span(tensor, address, bytes))
                        throw std::invalid_argument("invalid WSE Tensor span");
                    validate_storage(tensor.buffer.addr, tensor.buffer.size);
                }
                const auto *values = reinterpret_cast<const uint64_t *>(descriptor_region(
                    *d, placement.scalars_offset, work.scalar_count, sizeof(uint64_t) + sizeof(int32_t),
                    alignof(uint64_t)
                ));
                for (uint32_t i = 0; i < work.scalar_count; ++i) {
                    uint64_t value = values[i];
                    if (child) {
                        const int32_t parameter = reinterpret_cast<const int32_t *>(values + work.scalar_count)[i];
                        if (parameter >= 0) {
                            if (static_cast<uint32_t>(parameter) >= native_node.instance->boundary_scalar_count)
                                throw std::invalid_argument("invalid WSE Block scalar Graph binding");
                            value = reinterpret_cast<const uint64_t *>(
                                native_node.instance->boundary_scalars_address
                            )[parameter];
                        }
                    }
                    image.at<uint64_t>(core, work.scalars_offset)[i] = value;
                }
            }
        }
        if (expected != d->expected_tasks) throw std::invalid_argument("WSE task count mismatch");
        for (uint32_t core = 0; core < cores; ++core) {
            if (!local(core).queue.tail) continue;
            // A kernel can resolve a peer buffer passed by scalar address. Keep
            // the existing kernel ABI's contiguous authorization table on each
            // executing Core; it scales with distinct buffers, not graph nodes.
            auto &access = local(core).access;
            access.allocation_count = used_allocations.size();
            if (!used_allocations.empty()) {
                auto *bindings =
                    image.objects<wse::SramAllocation>(core, used_allocations.size(), WseSectionKind::ALLOCATIONS);
                access.allocations_offset = reinterpret_cast<uint64_t>(bindings) - application_base;
                std::copy(used_allocations.begin(), used_allocations.end(), bindings);
            }
        }
        used = image.usage();
        sections = std::move(image.sections);
    }
    void record(
        uint32_t worker, WseEventKind kind, uint64_t task = UINT64_MAX, uint32_t block = 0, uint64_t value = 0,
        uint32_t slot = 0, uint64_t generation = 0
    ) {
        events[worker].push_back(
            {sequence.fetch_add(1), task, value, worker / WSE_PE_PER_CORE, worker % WSE_PE_PER_CORE, block, kind, slot,
             generation}
        );
    }
};

namespace {
struct OperationBinding {
    WseRun *run;
    uint32_t worker;
    uint64_t task;
    uint32_t block;
};
void operation_trace(void *opaque, wse::OperationEvent kind, uint32_t slot, uint64_t generation) {
    auto &binding = *static_cast<OperationBinding *>(opaque);
    binding.run->record(
        binding.worker, static_cast<WseEventKind>(kind), binding.task, binding.block, 0, slot, generation
    );
}
static_assert(static_cast<uint32_t>(wse::OperationEvent::PUBLISH) == static_cast<uint32_t>(WseEventKind::OP_PUBLISH));
static_assert(static_cast<uint32_t>(wse::OperationEvent::CONSUMED) == static_cast<uint32_t>(WseEventKind::OP_CONSUMED));
}  // namespace

extern "C" WseRun *wse_create_run_v6(
    const WseGraphDescriptor *d, uint32_t cores, uint64_t epoch, const wse::SramRegion *regions, uint32_t region_count,
    const wse::SramAllocation *allocations, uint32_t allocation_count, char *error, size_t error_size
) {
    try {
        if (!d || !cores || cores > region_count || region_count > wse::MAX_CORE_COUNT || !epoch || !regions)
            throw std::invalid_argument("invalid WSE SRAM run descriptor");
        if (d->descriptor_bytes < sizeof(*d) || d->nodes_address - reinterpret_cast<uintptr_t>(d) < sizeof(*d))
            throw std::invalid_argument("WSE descriptor lacks per-PE calendar header");
        if (d->pe_calendar_version != wse::PE_CALENDAR_VERSION)
            throw std::invalid_argument("incompatible WSE PE calendar version");
        return new WseRun(d, cores, epoch, regions, region_count, allocations, allocation_count);
    } catch (const std::exception &e) {
        if (error && error_size) std::snprintf(error, error_size, "%s", e.what());
        return nullptr;
    }
}
extern "C" const WseMemorySection *wse_memory_sections(const WseRun *run, uint64_t *count) {
    *count = run->sections.size();
    return run->sections.data();
}
extern "C" uint64_t wse_runtime_used(const WseRun *run, uint32_t core) {
    return core < run->used.size() ? run->used[core] : 0;
}
extern "C" uint64_t wse_control_address(const WseRun *run, uint32_t core) {
    return core < run->core_controls.size() ? reinterpret_cast<uint64_t>(&run->local(core)) : 0;
}
extern "C" void wse_destroy_run(WseRun *run) { delete run; }
extern "C" uint64_t wse_completed_tasks(const WseRun *run) { return run->completed_tasks(); }
extern "C" const WseEvent *wse_run_events(const WseRun *run, uint32_t worker, uint64_t *count) {
    if (!run || !count || (worker != UINT32_MAX && worker >= run->events.size())) return nullptr;
    const auto &stream = worker == UINT32_MAX ? run->host_events : run->events[worker];
    *count = stream.size();
    return stream.data();
}

static void execute_pe(WseRun *run) {
    const int32_t physical_worker = wse::current_worker_id();
    if (physical_worker < 0) throw std::runtime_error("WSE PE has no bound physical identity");
    const auto worker = static_cast<uint32_t>(physical_worker);
    const uint32_t core = worker / WSE_PE_PER_CORE;
    const uint32_t pe = worker % WSE_PE_PER_CORE;
    std::unique_lock lock(run->mutex);
    if (!run->configured) {
        clear_cpu_sim_shared_storage();
        run->record(worker, WseEventKind::CONFIG_READY, UINT64_MAX, 0, run->expected_tasks);
        run->configured = true;
    }
    auto &local = run->local(core);
    if (!local.initialized) {
        local.initialized = true;
        run->record(worker, WseEventKind::CORE_INIT, UINT64_MAX, 0, run->expected_tasks);
    }
    sim_context_set_cluster_id(core);
    sim_context_set_subblock_id(0);
    run->record(worker, WseEventKind::PE_READY);
    ++local.ready;
    if (++run->ready_pes == run->events.size()) {
        run->record(worker, WseEventKind::GO);
        run->go = true;
        run->changed.notify_all();
    }
    run->wait(lock, [&] {
        return run->go;
    });
    while (true) {
        int thread = -1;
        run->wait(lock, [&] {
            thread = run->ready_thread(worker);
            return thread >= 0 || run->completed_tasks() == run->expected_tasks;
        });
        if (run->completed_tasks() == run->expected_tasks) break;
        const wse::Work work = run->head(core);
        local.started |= 1u << thread;
        const bool execute = local.execute;
        const uint64_t block_epoch = local.block_epoch;
        lock.unlock();
        const auto &plan = *work.task;
        run->record(worker, WseEventKind::THREAD_BEGIN, plan.ref, work.block, pe);
        if (execute) {
            int64_t args[SPMD_GLOBAL_CONTEXT_INDEX + 1]{};
            int index = 0;
            const uint64_t core_base = local.base;
            for (uint32_t i = 0; i < work.tensor_count; ++i)
                args[index++] =
                    reinterpret_cast<int64_t>(reinterpret_cast<Tensor *>(core_base + work.tensors_offset) + i);
            for (uint32_t i = 0; i < work.scalar_count; ++i)
                args[index++] = reinterpret_cast<const uint64_t *>(core_base + work.scalars_offset)[i];
            LocalContext context{};
            context.block_idx = work.block;
            context.block_num = plan.blocks;
            context.legacy_abi_slot = thread;
            context.thread_num = work.thread_num;
            context.block_scope = {
                &local.block_epoch,
                block_epoch,
                &local.epoch,
                local.epoch,
                run->block_storage[core].get(),
                wse::BlockOperationStorage::get
            };
            GlobalContext global{};
            global.core_id = core;
            global.pe_id = pe;
            global.sram_base = run->application_base;
            global.sram_access_offset = reinterpret_cast<uint64_t>(&local.access) - global.sram_base;
            OperationBinding binding{run, worker, plan.ref, work.block};
            global.operation_trace = {&binding, operation_trace};
            args[SPMD_LOCAL_CONTEXT_INDEX] = reinterpret_cast<int64_t>(&context);
            args[SPMD_GLOBAL_CONTEXT_INDEX] = reinterpret_cast<int64_t>(&global);
            reinterpret_cast<void (*)(int64_t *)>(plan.entry)(args);
        }
        run->record(worker, WseEventKind::THREAD_END, plan.ref, work.block, pe);
        lock.lock();
        if (run->failed) throw RunAborted{};
        local.finished |= 1u << thread;
        if (local.finished == (1u << work.thread_num) - 1) {
            // Current CPU tile operations return only after their work drains.
            __atomic_store_n(&local.block_epoch, uint64_t{0}, __ATOMIC_RELEASE);
            run->block_storage[core]->clear();
            run->record(worker, WseEventKind::BLOCK_END, plan.ref, work.block);
            local.block_state = wse::BlockState::IDLE;
            ++local.queue.head;
            if (--work.task->signal->remaining_blocks == 0) run->complete_task(worker, *work.task);
        }
        run->changed.notify_all();
    }
    if (!run->stop) {
        run->record(worker, WseEventKind::STOP);
        run->stop = true;
    }
    run->record(worker, WseEventKind::QUIESCENT);
    if (++local.quiet == WSE_PE_PER_CORE) {
        local.retired = true;
        run->record(worker, WseEventKind::CORE_RETIRED);
        if (++run->retired_cores == run->core_controls.size())
            run->record(worker, WseEventKind::RUN_DONE, UINT64_MAX, 0, run->completed_tasks());
    }
}

extern "C" int wse_execute_pe(WseRun *run, uint32_t worker) {
    if (!run || worker >= run->events.size()) return -1;
    try {
        void *previous = sim_context_get_invocation_context();
        auto *encoded = reinterpret_cast<void *>(static_cast<uintptr_t>(worker) + 1);
        if (sim_context_bind_invocation_context(encoded) != 0)
            throw std::runtime_error("WSE physical PE identity binding failed");
        auto restore = RAIIScopeGuard([previous] {
            if (sim_context_bind_invocation_context(previous) != 0) std::terminate();
        });
        execute_pe(run);
        return 0;
    } catch (const RunAborted &) {
        // The failing PE already published the error and woke the run.
    } catch (const std::exception &error) {
        run->abort(worker, error.what());
    } catch (...) {
        run->abort(worker, "unknown exception");
    }
    return -1;
}
