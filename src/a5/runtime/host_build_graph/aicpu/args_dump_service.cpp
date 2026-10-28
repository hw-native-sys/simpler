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

#include "args_dump_service.h"

#include "aicpu/cache_maintenance.h"
#include "aicpu/device_run_result_base_aicpu.h"
#include "common/memory_barrier.h"
#include "host_build_graph/runtime_types.h"

namespace {
template <typename T>
T *image_array(SchedulerArgsDumpHeader *image, uint64_t offset) {
    return reinterpret_cast<T *>(reinterpret_cast<uint8_t *>(image) + offset);
}

bool same_phase(const SchedulerArgsDumpRequest &a, const SchedulerArgsDumpRequest &b) {
    return a.run_epoch == b.run_epoch && a.task_id == b.task_id && a.stage == b.stage && a.generation == b.generation &&
           a.arg_index == b.arg_index;
}

uint64_t callable_object(uint64_t objects, uint64_t count, int32_t function_id) {
    if (objects == 0 || function_id < 0 || static_cast<uint64_t>(function_id) >= count) return 0;
    const auto *table = reinterpret_cast<const uint64_t *>(objects);
    return table[function_id];
}
}  // namespace

bool initialize_resident_args_dump_outputs(
    void *base, const SchedulerGraphView &graph, uint64_t objects, uint64_t count
) {
    if (base == nullptr) return true;
    auto *image = static_cast<SchedulerArgsDumpHeader *>(base);
    cache_invalidate_range(image, sizeof(*image));
    rmb();
    cache_invalidate_range(image, image->total_size);
    rmb();
    if (image->task_count != graph.task_count || image->initialized_count != image->selected_count ||
        image->run_epoch != get_platform_run_result_epoch())
        return false;
    auto *tasks = image_array<SchedulerArgsDumpTask>(image, image->tasks_offset);
    for (uint64_t index = 0; index < image->selected_count; ++index) {
        auto &task = tasks[index];
        if (task.task_id >= graph.task_count) return false;
        auto *descriptor = reinterpret_cast<TaskDescriptor *>(scheduler_graph_descriptor(graph, task.task_id));
        auto *payload = reinterpret_cast<TaskPayload *>(scheduler_graph_payload(graph, task.task_id));
        cache_invalidate_range(descriptor, sizeof(*descriptor));
        cache_invalidate_range(payload, sizeof(*payload));
        rmb();
        if (TaskId::to_uint64(descriptor->task_id) != task.task_id || payload->tensor_count < 0 ||
            payload->tensor_count > CORE_MAX_TENSOR_ARGS || payload->scalar_count < 0 ||
            payload->scalar_count > CORE_MAX_SCALAR_ARGS)
            return false;
        LaneMask lanes;
        for (uint32_t slot = 0; slot < SUBTASK_SLOT_COUNT; ++slot) {
            if ((task.active_mask & (UINT64_C(1) << slot)) == 0) continue;
            lanes.activate(static_cast<SubtaskSlot>(slot));
            const uint64_t address = callable_object(objects, count, descriptor->kernel_id[slot]);
            if (address == 0) return false;
            cache_invalidate_range(reinterpret_cast<void *>(address), sizeof(CoreCallable));
            rmb();
            const auto *signature = reinterpret_cast<const CoreCallable *>(address);
            if (signature->sig_count() < 0 || signature->sig_count() > CORE_MAX_TENSOR_ARGS) return false;
        }
        rmb();
        uint32_t cursor = 0;
        ArgsDumpInfo info{};
        while (next_dump_arg_for_task<SUBTASK_SLOT_COUNT>(
            0, *descriptor, *payload, lanes, ArgsDumpStage::AFTER_COMPLETION,
            [=](int32_t id) {
                return callable_object(objects, count, id);
            },
            &cursor, &info, &payload->dump_metadata
        )) {
            if (info.kind == static_cast<uint8_t>(ArgsDumpKind::TENSOR) && info.arg_index < CORE_MAX_TENSOR_ARGS &&
                (image->dump_level != static_cast<uint64_t>(DumpArgsLevel::HYBRID) || info.capture_payload != 0))
                task.output_tensor_mask |= UINT64_C(1) << info.arg_index;
        }
        // All AICores still await their context/READY publication here. No
        // phase word can have been modified by a participant yet.
        cache_flush_range(&task, sizeof(task));
    }
    wmb();
    return true;
}

bool ResidentArgsDumpService::initialize(
    void *base, const SchedulerGraphView &graph, uint64_t objects, uint64_t count, int32_t thread, int32_t threads
) {
    if (image_ != nullptr || reservation_.open()) return false;
    if (base == nullptr) return true;
    if (thread < 0 || threads <= 0 || thread >= threads || threads > PLATFORM_MAX_AICPU_THREADS) return false;
    auto *image = static_cast<SchedulerArgsDumpHeader *>(base);
    cache_invalidate_range(image, sizeof(*image));
    rmb();
    if (image->task_count != graph.task_count || image->worker_count == 0 ||
        image->worker_count > SCHEDULER_WORKER_CAPACITY || objects == 0 || count == 0)
        return false;
    cache_invalidate_range(image, image->total_size);
    rmb();
    image_ = image;
    graph_ = graph;
    callable_objects_ = objects;
    callable_count_ = count;
    thread_ = thread;
    lo_ = static_cast<uint32_t>(thread * image->worker_count / threads);
    hi_ = static_cast<uint32_t>((thread + 1) * image->worker_count / threads);
    head_ = 0;
    pending_ = 0;
    owner_ = UINT32_MAX;
    return true;
}

uint64_t ResidentArgsDumpService::callable(int32_t id) const {
    return callable_object(callable_objects_, callable_count_, id);
}

void ResidentArgsDumpService::acknowledge(
    uint32_t worker, const SchedulerArgsDumpRequest &request, SchedulerArgsDumpStatus status, uint64_t cursor,
    uint64_t bytes
) {
    auto *ack = &image_array<SchedulerArgsDumpAck>(image_, image_->acknowledgements_offset)[worker];
    __atomic_store_n(&ack->status, UINT64_C(0), __ATOMIC_RELEASE);
    cache_flush_range(ack, sizeof(*ack));
    wmb();
    __atomic_store_n(&ack->run_epoch, request.run_epoch, __ATOMIC_RELAXED);
    __atomic_store_n(&ack->task_id, request.task_id, __ATOMIC_RELAXED);
    __atomic_store_n(&ack->stage, request.stage, __ATOMIC_RELAXED);
    __atomic_store_n(&ack->generation, request.generation, __ATOMIC_RELAXED);
    __atomic_store_n(&ack->arg_index, cursor, __ATOMIC_RELAXED);
    __atomic_store_n(&ack->byte_cursor, bytes, __ATOMIC_RELAXED);
    __atomic_store_n(&ack->tick, request.tick, __ATOMIC_RELEASE);
    cache_flush_range(ack, sizeof(*ack));
    wmb();
    __atomic_store_n(
        &ack->status, scheduler_args_dump_ack_commit(request.generation, request.stage, status), __ATOMIC_RELEASE
    );
    cache_flush_range(ack, sizeof(*ack));
    wmb();
}

bool ResidentArgsDumpService::service(uint32_t worker, const SchedulerArgsDumpRequest &request) {
    auto *task = scheduler_args_dump_task(image_, request.task_id);
    const auto stage = static_cast<ArgsDumpStage>(request.stage);
    bool valid = task != nullptr && request.run_epoch == image_->run_epoch &&
                 request.stage <= static_cast<uint64_t>(ArgsDumpStage::AFTER_COMPLETION) && request.generation != 0 &&
                 request.generation <= UINT32_MAX && request.arg_index <= CORE_MAX_TENSOR_ARGS + CORE_MAX_SCALAR_ARGS;
    if (task != nullptr) {
        // Observation only: AICPU never writes a cache line containing the
        // concurrently updated GM arrival/departure/abort words.
        cache_invalidate_range(task, sizeof(*task));
        rmb();
        if (valid) {
            const uint64_t publisher =
                stage == ArgsDumpStage::BEFORE_DISPATCH ? task->publisher_before : task->publisher_after;
            valid = publisher == scheduler_args_dump_publisher(worker, request.generation) &&
                    (scheduler_args_dump_masks(task, stage) & UINT64_C(7)) == task->active_mask && task->aborted == 0;
            if (stage == ArgsDumpStage::AFTER_COMPLETION) valid = valid && scheduler_args_dump_can_execute(task);
        }
    }
    if (!valid) {
        if (owner_ == worker && reservation_.open()) (void)dump_arg_abandon(&reservation_);
        if (owner_ == worker) owner_ = UINT32_MAX;
        acknowledge(worker, request, SchedulerArgsDumpStatus::Aborted, request.arg_index, request.byte_cursor);
        return false;
    }
    if (reservation_.open() && owner_ != worker) {
        acknowledge(worker, request, SchedulerArgsDumpStatus::WouldBlock, request.arg_index, request.byte_cursor);
        return false;
    }
    if (reservation_.open()) {
        if (!same_phase(request, owner_request_) || request.byte_cursor != reservation_.copied_bytes()) {
            (void)dump_arg_abandon(&reservation_);
            owner_ = UINT32_MAX;
            acknowledge(worker, request, SchedulerArgsDumpStatus::Aborted, request.arg_index, request.byte_cursor);
            return false;
        }
    } else {
        if (request.byte_cursor != 0) {
            acknowledge(worker, request, SchedulerArgsDumpStatus::Aborted, request.arg_index, request.byte_cursor);
            return false;
        }
        auto *descriptor = reinterpret_cast<TaskDescriptor *>(scheduler_graph_descriptor(graph_, request.task_id));
        auto *payload = reinterpret_cast<TaskPayload *>(scheduler_graph_payload(graph_, request.task_id));
        LaneMask lanes;
        for (uint32_t slot = 0; slot < SUBTASK_SLOT_COUNT; ++slot)
            if ((task->active_mask & (UINT64_C(1) << slot)) != 0) lanes.activate(static_cast<SubtaskSlot>(slot));
        uint32_t cursor = static_cast<uint32_t>(request.arg_index);
        ArgsDumpInfo info{};
        if (!next_dump_arg_for_task<SUBTASK_SLOT_COUNT>(
                thread_, *descriptor, *payload, lanes, stage,
                [this](int32_t id) {
                    return callable(id);
                },
                &cursor, &info, &payload->dump_metadata
            )) {
            const auto flushed = dump_args_try_flush(thread_);
            acknowledge(
                worker, request,
                flushed == DumpArgFlushResult::Complete   ? SchedulerArgsDumpStatus::Sealed :
                flushed == DumpArgFlushResult::WouldBlock ? SchedulerArgsDumpStatus::WouldBlock :
                                                            SchedulerArgsDumpStatus::Aborted,
                request.arg_index, 0
            );
            return flushed == DumpArgFlushResult::Complete;
        }
        const auto begun = dump_arg_try_begin(thread_, info, &reservation_);
        if (begun != DumpArgBeginResult::Ready) {
            acknowledge(
                worker, request,
                begun == DumpArgBeginResult::WouldBlock ? SchedulerArgsDumpStatus::WouldBlock :
                                                          SchedulerArgsDumpStatus::Aborted,
                request.arg_index, 0
            );
            return false;
        }
        owner_ = worker;
    }
    owner_request_ = request;
    const uint64_t before = reservation_.copied_bytes();
    constexpr uint64_t kCopyBytesPerStep = 64 * 1024;
    const auto copied = dump_arg_copy_chunk(&reservation_, kCopyBytesPerStep);
    const uint64_t after = reservation_.copied_bytes();
    if (copied == DumpArgCopyResult::Complete && dump_arg_commit(&reservation_)) {
        owner_ = UINT32_MAX;
        acknowledge(worker, request, SchedulerArgsDumpStatus::InProgress, request.arg_index + 1, 0);
        return true;
    }
    if (copied == DumpArgCopyResult::Invalid || copied == DumpArgCopyResult::Complete) {
        (void)dump_arg_abandon(&reservation_);
        owner_ = UINT32_MAX;
        acknowledge(worker, request, SchedulerArgsDumpStatus::Aborted, request.arg_index, after);
        return false;
    }
    acknowledge(worker, request, SchedulerArgsDumpStatus::InProgress, request.arg_index, after);
    return after != before;
}

bool ResidentArgsDumpService::step() {
    if (image_ == nullptr || lo_ == hi_) return false;
    auto *requests = image_array<SchedulerArgsDumpRequest>(image_, image_->requests_offset);
    auto *observed = image_array<uint64_t>(image_, image_->observed_ticks_offset);
    auto *fifo = image_array<uint32_t>(image_, image_->pending_workers_offset) + lo_;
    const uint32_t capacity = hi_ - lo_;
    for (uint32_t worker = lo_; worker < hi_; ++worker) {
        cache_invalidate_range(&requests[worker], sizeof(requests[worker]));
        rmb();
        if (requests[worker].tick == 0 || requests[worker].tick == observed[worker]) continue;
        // One outstanding request per worker: it cannot republish until its
        // matching ack, which is emitted only after removing it from this FIFO.
        if (pending_ == capacity) break;
        observed[worker] = requests[worker].tick;
        fifo[(head_ + pending_) % capacity] = worker;
        ++pending_;
    }
    if (pending_ == 0) return false;
    const uint32_t worker = fifo[head_];
    head_ = (head_ + 1) % capacity;
    --pending_;
    const SchedulerArgsDumpRequest request = requests[worker];
    return service(worker, request);
}

void ResidentArgsDumpService::stop() {
    if (image_ == nullptr) return;
    if (reservation_.open()) (void)dump_arg_abandon(&reservation_);
    auto *requests = image_array<SchedulerArgsDumpRequest>(image_, image_->requests_offset);
    for (uint32_t worker = lo_; worker < hi_; ++worker) {
        cache_invalidate_range(&requests[worker], sizeof(requests[worker]));
        rmb();
        const auto request = requests[worker];
        if (request.tick == 0) continue;
        auto *task = scheduler_args_dump_task(image_, request.task_id);
        if (task == nullptr) continue;
        cache_invalidate_range(task, sizeof(*task));
        rmb();
        if (!scheduler_args_dump_can_complete(task))
            acknowledge(worker, request, SchedulerArgsDumpStatus::Aborted, request.arg_index, request.byte_cursor);
    }
    owner_ = UINT32_MAX;
    if (!reservation_.open()) image_ = nullptr;
}
