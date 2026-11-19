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
#include "wse_launch.h"
#include "sram_layout.h"
#include "wse/block_lifecycle.h"
#include <algorithm>
#include <cstring>
#include <new>
#include <iostream>
#include <string>
#include "cpu_sim_context.h"
#include "host_build_graph/submit_types.h"
#include "scheduler/scheduler_graph_binding.h"

#include <atomic>
#include <cstdio>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

static_assert(alignof(wse::RunControl) >= alignof(uint64_t));

namespace {
class ImageBuilder {
public:
    explicit ImageBuilder(const wse::SramRegion *regions, uint32_t count) :
        regions_(regions, regions + count) {
        for (const auto &r : regions_)
            cursor_.push_back(r.reserve);
    }
    template <class T>
    uint64_t allocate(uint32_t core, uint64_t count = 1, uint32_t kind = 0) {
        const auto &r = regions_.at(core);
        if (count > SIZE_MAX / sizeof(T)) throw std::invalid_argument("WSE SRAM layout multiplication overflow");
        const uint64_t bytes = count * sizeof(T);
        if (bytes > cursor_[core]) exhausted(core, bytes);
        const auto offset = (cursor_[core] - bytes) & ~(uint64_t{alignof(T)} - 1);
        cursor_[core] = offset;
        if (bytes) sections.push_back({r.runtime_base + offset, bytes, core, kind});
        auto *objects = reinterpret_cast<T *>(r.runtime_base + offset);
        for (uint64_t i = 0; i < count; ++i)
            new (objects + i) T{};
        return r.runtime_base - r.base + offset;
    }
    template <class T>
    T *at(uint32_t core, uint64_t offset) {
        return reinterpret_cast<T *>(regions_[core].base + offset);
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
    [[noreturn]] void exhausted(uint32_t core, uint64_t required) const {
        throw std::runtime_error(
            "WSE Runtime SRAM capacity exhausted: core=" + std::to_string(core) +
            " required=" + std::to_string(required) + " available=" + std::to_string(regions_[core].reserve)
        );
    }
    std::vector<wse::SramRegion> regions_;
    std::vector<uint64_t> cursor_;
};
}  // namespace

struct WseRun {
    uint64_t base;
    wse::RunControl *control;
    std::mutex mutex;
    std::condition_variable changed;
    std::atomic<uint64_t> sequence{0};
    std::vector<std::vector<WseEvent>> events;
    std::vector<uint64_t> used;
    std::vector<std::unique_ptr<wse::BlockOperationStorage>> block_storage;
    std::vector<WseMemorySection> sections;

    template <class T>
    T *at(uint64_t offset) const {
        return reinterpret_cast<T *>(base + offset);
    }
    const wse::SramRegion *regions() const { return at<wse::SramRegion>(control->access.regions_offset); }
    wse::CoreControl &local(uint32_t core) const {
        return *reinterpret_cast<wse::CoreControl *>(
            regions()[core].base + at<uint64_t>(control->core_offsets_offset)[core]
        );
    }
    wse::TaskRecord *plans() const { return at<wse::TaskRecord>(control->tasks_offset); }
    uint64_t completed_tasks() const { return __atomic_load_n(&control->completed, __ATOMIC_ACQUIRE); }
    wse::Work &head(uint32_t core) const {
        const auto &q = local(core).queue;
        return reinterpret_cast<wse::Work *>(regions()[core].base + q.work_offset)[q.head];
    }
    bool dependencies_ready(const wse::TaskRecord &task) const {
        const auto *predecessors = at<uint64_t>(task.predecessors_offset);
        for (uint32_t i = 0; i < task.predecessor_count; ++i)
            if (!plans()[predecessors[i]].completed) return false;
        return true;
    }
    void complete_task(uint32_t worker, uint64_t ref) {
        auto &task = plans()[ref];
        if (task.completed) throw std::logic_error("WSE duplicate Graph-node completion");
        task.completed = 1;
        const uint64_t count = __atomic_fetch_add(&control->completed, uint64_t{1}, __ATOMIC_ACQ_REL) + 1;
        record(worker, WseEventKind::TASK_DONE, ref, 0, count);
        changed.notify_all();
    }
    void advance_controls(uint32_t worker) {
        // Controls have no Block queue. The lock gives each ready node one executor,
        // without a waiting Graph exit trapping an independent Graph entry behind it.
        for (uint32_t i = 0; i < control->task_count; ++i) {
            auto &task = plans()[i];
            if (!task.entry && !task.completed && dependencies_ready(task)) complete_task(worker, i);
        }
    }
    int ready_thread(uint32_t worker) {
        advance_controls(worker);
        const uint32_t core = worker / WSE_PE_PER_CORE, pe = worker % WSE_PE_PER_CORE;
        auto &c = local(core);
        if (c.queue.head == c.queue.tail) return -1;
        const auto &work = head(core);
        const auto &task = plans()[work.task];
        if (!c.active && pe == 0 && dependencies_ready(task)) {
            c.active = 1;
            c.started = c.finished = 0;
            c.execute = !task.has_predicate ||
                        DispatchPredicate{
                            task.predicate.addr, task.predicate.target, task.predicate.elem_size,
                            static_cast<PredicateOp>(task.predicate.op)
                        }
                            .pass();
            __atomic_store_n(&c.block_epoch, ++c.block_serial, __ATOMIC_RELEASE);
            record(worker, WseEventKind::BLOCK_BEGIN, work.task, work.block, work.thread_num);
            if (!c.execute) record(worker, WseEventKind::BLOCK_SKIPPED, work.task, work.block);
            changed.notify_all();
        }
        if (!c.active) return -1;
        for (uint32_t t = 0; t < work.thread_num; ++t)
            if (work.thread_pes[t] == pe && !(c.started & (1u << t))) return t;
        return -1;
    }

    WseRun(
        const WseGraphDescriptor *d, uint32_t cores, uint64_t epoch, const wse::SramRegion *storage,
        uint32_t region_count, const wse::SramAllocation *allocations, uint32_t allocation_count
    ) :
        base(storage[0].base),
        events(cores * WSE_PE_PER_CORE) {
        for (uint32_t core = 0; core < cores; ++core)
            block_storage.push_back(std::make_unique<wse::BlockOperationStorage>());
        const SchedulerGraphAccess graph(d->graph);
        if (!scheduler_graph_view_valid(graph)) throw std::invalid_argument("invalid WSE graph");
        const auto *source_plans = reinterpret_cast<const WseNodePlan *>(d->nodes_address);
        const auto count = graph.task_count;
        ImageBuilder image(storage, region_count);
        control = image.at<wse::RunControl>(0, image.allocate<wse::RunControl>(0, 1, 1));
        control->epoch = epoch;
        control->expected_tasks = d->expected_tasks;
        control->core_count = cores;
        control->task_count = count;
        control->tasks_offset = image.allocate<wse::TaskRecord>(0, count, 2);
        control->core_offsets_offset = image.allocate<uint64_t>(0, cores, 3);
        control->access.region_count = region_count;
        control->access.regions_offset = image.allocate<wse::SramRegion>(0, region_count, 4);
        std::copy(storage, storage + region_count, at<wse::SramRegion>(control->access.regions_offset));
        std::vector<std::vector<wse::Work>> queues(cores);
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
            auto &task = plans()[ref];
            task.entry = source.entry;
            task.blocks = source.blocks;
            task.host_done = source.host_done;
            task.has_predicate = source.has_predicate;
            task.completed = source.host_done;
            if (source.host_done) continue;
            ++expected;
            if (!source.blocks || source.blocks > INT16_MAX)
                throw std::invalid_argument("invalid WSE task block count");
            if (source.entry && scheduler_bind_graph_arguments(graph, ref) != SchedulerGraphBindingError::NONE)
                throw std::invalid_argument("invalid WSE graph argument binding");
            SchedulerTaskPayloadArguments args{};
            if (source.entry) {
                if (scheduler_parse_task_payload_arguments(graph, ref, &args) != SchedulerGraphResult::OK)
                    throw std::invalid_argument("invalid WSE task arguments");
                if (source.has_predicate) {
                    const auto *predicate = scheduler_node_predicate(graph, ref);
                    if (!predicate || !scheduler_dispatch_predicate_metadata_valid(
                                          predicate->addr, predicate->elem_size, predicate->op
                                      ))
                        throw std::invalid_argument("invalid WSE predicate");
                    task.predicate = *predicate;
                    validate_storage(predicate->addr, predicate->elem_size);
                }
            }
            uint32_t fanins = 0;
            if (scheduler_node_fanin_count(graph, ref, &fanins) != SchedulerGraphResult::OK)
                throw std::invalid_argument("invalid WSE dependencies");
            std::vector<uint64_t> predecessors;
            for (uint32_t i = 0; i < fanins; ++i) {
                int64_t producer = -1;
                if (scheduler_node_fanin_ref(graph, ref, i, &producer) != SchedulerGraphResult::OK || producer < 0 ||
                    static_cast<uint64_t>(producer) >= ref)
                    throw std::invalid_argument("invalid WSE dependency order");
                if (!source_plans[producer].host_done) {
                    predecessors.push_back(producer);
                }
            }
            task.predecessor_count = predecessors.size();
            task.predecessors_offset = image.allocate<uint64_t>(0, predecessors.size(), 7);
            std::copy(predecessors.begin(), predecessors.end(), at<uint64_t>(task.predecessors_offset));
            if (!source.entry) continue;
            auto source_region = [&](uint64_t offset, uint64_t elements, uint64_t stride, uint64_t alignment) {
                if (offset < sizeof(WseGraphDescriptor) || offset > d->descriptor_bytes || offset % alignment ||
                    elements > (d->descriptor_bytes - offset) / stride)
                    throw std::invalid_argument("invalid WSE Block argument image");
                return reinterpret_cast<const std::byte *>(d) + offset;
            };
            const WseBlockPlan *block_plans = nullptr;
            if (source.block_plan_count) {
                if (source.block_plan_count != source.blocks)
                    throw std::invalid_argument("invalid WSE Block plan count");
                block_plans = reinterpret_cast<const WseBlockPlan *>(source_region(
                    source.block_plans_offset, source.block_plan_count, sizeof(WseBlockPlan), alignof(WseBlockPlan)
                ));
            } else if (source.block_plans_offset) throw std::invalid_argument("unexpected WSE Block plans offset");
            SchedulerNativeNode native_node{};
            const bool child = graph.native_header_offset && scheduler_native_node_checked(graph, ref, &native_node) &&
                               native_node.binding->kind == SchedulerNodeKind::CHILD_KERNEL;
            for (uint32_t b = 0; b < task.blocks; ++b) {
                WseBlockPlan placement{};
                placement.core_id = b % cores;
                placement.thread_num = 1;
                placement.tensor_count = args.tensor_count;
                placement.scalar_count = args.scalar_count;
                if (block_plans) placement = block_plans[b];
                const uint32_t core = placement.core_id;
                if (core >= cores || !placement.thread_num || placement.thread_num > WSE_PE_PER_CORE ||
                    placement.tensor_count > CORE_MAX_TENSOR_ARGS || placement.scalar_count > CORE_MAX_SCALAR_ARGS)
                    throw std::invalid_argument("invalid WSE Block Core, Threads or arguments");
                uint32_t mask = 0;
                for (uint32_t t = 0; t < placement.thread_num; ++t) {
                    const uint32_t pe = placement.thread_pes[t];
                    if (pe >= WSE_PE_PER_CORE || (mask & (1u << pe)))
                        throw std::invalid_argument("WSE Block Threads must map to distinct local PEs");
                    mask |= 1u << pe;
                }
                wse::Work work{};
                work.task = ref;
                work.block = b;
                work.thread_num = placement.thread_num;
                std::copy_n(placement.thread_pes, WSE_PE_PER_CORE, work.thread_pes);
                work.tensor_count = placement.tensor_count;
                work.scalar_count = placement.scalar_count;
                work.tensors_offset = image.allocate<Tensor>(core, work.tensor_count, 5);
                work.scalars_offset = image.allocate<uint64_t>(core, work.scalar_count, 6);
                const auto *tensor_bytes =
                    block_plans ?
                        source_region(placement.tensors_offset, work.tensor_count, sizeof(Tensor), alignof(Tensor)) :
                        reinterpret_cast<const std::byte *>(args.tensors);
                for (uint32_t i = 0; i < work.tensor_count; ++i) {
                    auto &tensor = image.at<Tensor>(core, work.tensors_offset)[i];
                    std::memcpy(&tensor, tensor_bytes + i * sizeof(Tensor), sizeof(Tensor));
                    if (block_plans && child) {
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
                const auto *values = block_plans ? reinterpret_cast<const uint64_t *>(source_region(
                                                       placement.scalars_offset, work.scalar_count,
                                                       sizeof(uint64_t) + sizeof(int32_t), alignof(uint64_t)
                                                   )) :
                                                   args.scalars;
                for (uint32_t i = 0; i < work.scalar_count; ++i) {
                    uint64_t value = values[i];
                    if (block_plans && child) {
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
                queues[core].push_back(work);
            }
        }
        if (expected != d->expected_tasks) throw std::invalid_argument("WSE task count mismatch");
        for (uint32_t core = 0; core < cores; ++core) {
            const auto offset = image.allocate<wse::CoreControl>(core, 1, 8);
            at<uint64_t>(control->core_offsets_offset)[core] = offset;
            auto &q = local(core).queue;
            q.capacity = q.tail = queues[core].size();
            q.work_offset = image.allocate<wse::Work>(core, q.capacity, 9);
            std::copy(queues[core].begin(), queues[core].end(), image.at<wse::Work>(core, q.work_offset));
        }
        control->access.allocation_count = used_allocations.size();
        control->access.allocations_offset = image.allocate<wse::SramAllocation>(0, used_allocations.size(), 10);
        std::copy(
            used_allocations.begin(), used_allocations.end(),
            at<wse::SramAllocation>(control->access.allocations_offset)
        );
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
    void bootstrap() {
        clear_cpu_sim_shared_storage();
        for (uint32_t i = 0; i < control->task_count; ++i) {
            auto &task = plans()[i];
            task.remaining_blocks = task.blocks;
        }
        __atomic_store_n(&control->completed, uint64_t{0}, __ATOMIC_RELAXED);
        record(0, WseEventKind::CONFIG_READY, UINT64_MAX, 0, control->expected_tasks);
        control->configured = true;
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

extern "C" WseRun *wse_create_run(
    const WseGraphDescriptor *d, uint32_t cores, uint64_t epoch, const wse::SramRegion *regions, uint32_t region_count,
    const wse::SramAllocation *allocations, uint32_t allocation_count, char *error, size_t error_size
) {
    try {
        if (!d || !cores || cores > region_count || region_count > wse::MAX_CORE_COUNT || !epoch || !regions)
            throw std::invalid_argument("invalid WSE SRAM run descriptor");
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
    return core < run->control->core_count ? reinterpret_cast<uint64_t>(&run->local(core)) : 0;
}
extern "C" void wse_destroy_run(WseRun *run) { delete run; }
extern "C" uint64_t wse_completed_tasks(const WseRun *run) { return run->completed_tasks(); }
extern "C" const WseEvent *wse_run_events(const WseRun *run, uint32_t worker, uint64_t *count) {
    if (!run || !count || worker >= run->events.size()) return nullptr;
    *count = run->events[worker].size();
    return run->events[worker].data();
}

extern "C" int wse_execute_pe(WseRun *run, uint32_t worker) {
    const uint32_t core = worker / WSE_PE_PER_CORE;
    const uint32_t pe = worker % WSE_PE_PER_CORE;
    std::unique_lock lock(run->mutex);
    if (worker == 0) {
        try {
            run->bootstrap();
        } catch (const std::exception &error) {
            std::cerr << "WSE bootstrap rejected: " << error.what() << '\n';
            run->control->init_error = -1;
        }
        run->changed.notify_all();
    }
    run->changed.wait(lock, [&] {
        return run->control->configured || run->control->init_error;
    });
    if (run->control->init_error) return run->control->init_error;
    auto &local = run->local(core);
    if (pe == 0) {
        local.queue.head = 0;
        local.ready = 0;
        local.quiet = 0;
        local.retired = false;
        local.expected_tasks = run->control->expected_tasks;
        local.initialized = true;
        run->record(worker, WseEventKind::CORE_INIT, UINT64_MAX, 0, local.expected_tasks);
        run->changed.notify_all();
    }
    run->changed.wait(lock, [&] {
        return local.initialized;
    });
    sim_context_set_cluster_id(core);
    sim_context_set_subblock_id(0);
    run->record(worker, WseEventKind::PE_READY);
    ++local.ready;
    run->changed.notify_all();
    if (worker == 0) {
        run->changed.wait(lock, [&] {
            for (uint32_t c = 0; c < run->control->core_count; ++c)
                if (run->local(c).ready != WSE_PE_PER_CORE) return false;
            return true;
        });
        run->record(worker, WseEventKind::GO);
        run->control->go = true;
        run->changed.notify_all();
    }
    run->changed.wait(lock, [&] {
        return run->control->go;
    });
    while (true) {
        int thread = -1;
        run->changed.wait(lock, [&] {
            thread = run->ready_thread(worker);
            return thread >= 0 || run->completed_tasks() == local.expected_tasks;
        });
        if (run->completed_tasks() == local.expected_tasks) break;
        const wse::Work work = run->head(core);
        local.started |= 1u << thread;
        const bool execute = local.execute;
        const uint64_t block_epoch = local.block_epoch;
        lock.unlock();
        run->record(worker, WseEventKind::THREAD_BEGIN, work.task, work.block, thread);
        const auto &plan = run->plans()[work.task];
        if (execute) {
            int64_t args[SPMD_GLOBAL_CONTEXT_INDEX + 1]{};
            int index = 0;
            const uint64_t core_base = run->regions()[core].base;
            for (uint32_t i = 0; i < work.tensor_count; ++i)
                args[index++] =
                    reinterpret_cast<int64_t>(reinterpret_cast<Tensor *>(core_base + work.tensors_offset) + i);
            for (uint32_t i = 0; i < work.scalar_count; ++i)
                args[index++] = reinterpret_cast<const uint64_t *>(core_base + work.scalars_offset)[i];
            LocalContext context{};
            context.block_idx = work.block;
            context.block_num = plan.blocks;
            context.thread_idx = thread;
            context.thread_num = work.thread_num;
            context.block_scope = {
                &local.block_epoch,
                block_epoch,
                &run->control->epoch,
                run->control->epoch,
                run->block_storage[core].get(),
                wse::BlockOperationStorage::get
            };
            GlobalContext global{};
            global.core_id = core;
            global.pe_id = pe;
            global.sram_base = run->base;
            global.sram_access_offset = reinterpret_cast<uint64_t>(&run->control->access) - run->base;
            OperationBinding binding{run, worker, work.task, work.block};
            global.operation_trace = {&binding, operation_trace};
            args[SPMD_LOCAL_CONTEXT_INDEX] = reinterpret_cast<int64_t>(&context);
            args[SPMD_GLOBAL_CONTEXT_INDEX] = reinterpret_cast<int64_t>(&global);
            reinterpret_cast<void (*)(int64_t *)>(plan.entry)(args);
        }
        run->record(worker, WseEventKind::THREAD_END, work.task, work.block, thread);
        lock.lock();
        local.finished |= 1u << thread;
        if (local.finished == (1u << work.thread_num) - 1) {
            // Current CPU tile operations return only after their work drains.
            __atomic_store_n(&local.block_epoch, uint64_t{0}, __ATOMIC_RELEASE);
            run->block_storage[core]->clear();
            run->record(worker, WseEventKind::BLOCK_END, work.task, work.block);
            local.active = 0;
            ++local.queue.head;
            auto &node = run->plans()[work.task];
            if (--node.remaining_blocks == 0) run->complete_task(worker, work.task);
        }
        run->changed.notify_all();
    }
    if (worker == 0) {
        run->record(worker, WseEventKind::STOP);
        run->control->stop = true;
        run->changed.notify_all();
    }
    run->changed.wait(lock, [&] {
        return run->control->stop;
    });
    run->record(worker, WseEventKind::QUIESCENT);
    ++local.quiet;
    run->changed.notify_all();
    if (pe == 0) {
        run->changed.wait(lock, [&] {
            return local.quiet == WSE_PE_PER_CORE;
        });
        local.retired = true;
        run->record(worker, WseEventKind::CORE_RETIRED);
        run->changed.notify_all();
    }
    if (worker == 0) {
        run->changed.wait(lock, [&] {
            for (uint32_t c = 0; c < run->control->core_count; ++c)
                if (!run->local(c).retired) return false;
            return true;
        });
        run->record(
            worker, WseEventKind::RUN_DONE, UINT64_MAX, 0, __atomic_load_n(&run->control->completed, __ATOMIC_SEQ_CST)
        );
    }
    return 0;
}
