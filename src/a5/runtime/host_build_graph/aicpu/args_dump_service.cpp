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
#include "host_build_graph/tensor.h"

namespace {
template <typename T>
T *image_array(SchedulerArgsDumpHeader *image, uint64_t offset) {
    return reinterpret_cast<T *>(reinterpret_cast<uint8_t *>(image) + offset);
}

bool same_phase(const SchedulerArgsDumpRequest &a, const SchedulerArgsDumpRequest &b) {
    return a.run_epoch == b.run_epoch && a.task_id == b.task_id && a.stage == b.stage && a.publisher == b.publisher &&
           a.arg_index == b.arg_index;
}

struct DumpDescriptorView {
    TaskId task_id{};
    int32_t kernel_id[SUBTASK_SLOT_COUNT]{};
};

struct DumpPayloadView {
    int32_t tensor_count{0};
    int32_t scalar_count{0};
    const simpler::hbg::Tensor *tensors{nullptr};
    const uint64_t *scalars{nullptr};
    const simpler::hbg::Tensor *tensor_data() const { return tensors; }
    const uint64_t *scalar_data() const { return scalars; }
};

bool observe_graph_arguments(
    const SchedulerArgsDumpTask &task, DumpDescriptorView *descriptor, DumpPayloadView *payload
) {
    if (task.tensor_count > CORE_MAX_TENSOR_ARGS || task.scalar_count > CORE_MAX_SCALAR_ARGS ||
        task.tensor_count + task.scalar_count > SPMD_LOCAL_CONTEXT_INDEX || task.argument_region_base == 0 ||
        task.argument_region_bytes > UINT64_MAX - task.argument_region_base)
        return false;
    const auto valid_region = [&](uint64_t address, uint64_t bytes, uint64_t alignment) {
        return address % alignment == 0 && address >= task.argument_region_base &&
               address - task.argument_region_base <= task.argument_region_bytes &&
               bytes <= task.argument_region_bytes - (address - task.argument_region_base);
    };
    const uint64_t tensor_bytes = task.tensor_count * sizeof(simpler::hbg::Tensor);
    const uint64_t scalar_bytes = task.scalar_count * sizeof(uint64_t);
    if ((tensor_bytes != 0 && !valid_region(task.tensors_address, tensor_bytes, alignof(simpler::hbg::Tensor))) ||
        (scalar_bytes != 0 && !valid_region(task.scalars_address, scalar_bytes, alignof(uint64_t))))
        return false;
    payload->tensor_count = task.tensor_count;
    payload->scalar_count = task.scalar_count;
    payload->tensors = reinterpret_cast<const simpler::hbg::Tensor *>(task.tensors_address);
    payload->scalars = reinterpret_cast<const uint64_t *>(task.scalars_address);
    TaskId::assign(descriptor->task_id, task.task_id);
    for (uint32_t slot = 0; slot < SUBTASK_SLOT_COUNT; ++slot)
        descriptor->kernel_id[slot] = task.kernel_id[slot];
    // Native child arguments are published by AICore bootstrap before a request.
    // Both complete source regions are bounded before either cache observation.
    if (tensor_bytes != 0) cache_invalidate_range(payload->tensors, tensor_bytes);
    if (scalar_bytes != 0) cache_invalidate_range(payload->scalars, scalar_bytes);
    rmb();
    return true;
}

uint64_t callable_object(uint64_t objects, uint64_t count, int32_t function_id) {
    if (objects == 0 || function_id < 0 || static_cast<uint64_t>(function_id) >= count) return 0;
    const auto *table = reinterpret_cast<const uint64_t *>(objects);
    return table[function_id];
}
}  // namespace

bool initialize_resident_args_dump_outputs(void *base, uint64_t objects, uint64_t count) {
    if (base == nullptr) return true;
    auto *image = static_cast<SchedulerArgsDumpHeader *>(base);
    cache_invalidate_range(image, sizeof(*image));
    rmb();
    cache_invalidate_range(image, image->total_size);
    rmb();
    if (image->initialized_count != image->selected_count || image->run_epoch != get_platform_run_result_epoch())
        return false;
    auto *tasks = image_array<SchedulerArgsDumpTask>(image, image->tasks_offset);
    for (uint64_t index = 0; index < image->selected_count; ++index) {
        auto &task = tasks[index];
        if (task.task_ref >= image->task_count || task.tensor_count > CORE_MAX_TENSOR_ARGS ||
            task.scalar_count > CORE_MAX_SCALAR_ARGS ||
            task.tensor_count + task.scalar_count > SPMD_LOCAL_CONTEXT_INDEX)
            return false;
        const CoreCallable *signature = nullptr;
        for (uint32_t slot = 0; slot < SUBTASK_SLOT_COUNT; ++slot) {
            if ((task.active_mask & (UINT64_C(1) << slot)) == 0) continue;
            const uint64_t address = callable_object(objects, count, task.kernel_id[slot]);
            if (address == 0) return false;
            cache_invalidate_range(reinterpret_cast<void *>(address), sizeof(CoreCallable));
            rmb();
            const auto *candidate = reinterpret_cast<const CoreCallable *>(address);
            if (candidate->sig_count() < 0 || candidate->sig_count() > CORE_MAX_TENSOR_ARGS) return false;
            if (signature == nullptr || candidate->sig_count() > signature->sig_count()) signature = candidate;
        }
        if (signature == nullptr) return false;
        task.output_tensor_mask = 0;
        for (int32_t arg = 0; arg < signature->sig_count() && arg < task.tensor_count; ++arg) {
            ArgsDumpRole role;
            if (get_dump_arg_role_from_direction(signature->sig(arg), &role) &&
                should_dump_arg_at_stage(role, ArgsDumpStage::AFTER_COMPLETION) &&
                should_dump_arg(task.dump_metadata.dump_arg_mask, arg) &&
                (image->dump_level != static_cast<uint64_t>(DumpArgsLevel::HYBRID) ||
                 has_dump_arg_flag(task.dump_metadata.dump_arg_mask, arg)))
                task.output_tensor_mask |= UINT64_C(1) << arg;
        }
        // No child tensor value is read before AICore bootstrap binds it.
        // The mask lies outside the mutable GM-atomic state cache line.
        cache_flush_range(&task.task_id, sizeof(task) - offsetof(SchedulerArgsDumpTask, task_id));
    }
    wmb();
    return true;
}

bool ResidentArgsDumpService::initialize(
    void *base, uint64_t objects, uint64_t count, int32_t thread, int32_t threads
) {
    if (image_ != nullptr || reservation_.open()) return false;
    if (base == nullptr) return true;
    if (thread < 0 || threads <= 0 || thread >= threads || threads > PLATFORM_MAX_AICPU_THREADS) return false;
    auto *image = static_cast<SchedulerArgsDumpHeader *>(base);
    cache_invalidate_range(image, sizeof(*image));
    rmb();
    if (image->worker_count == 0 || image->worker_count > SCHEDULER_WORKER_CAPACITY || objects == 0 || count == 0)
        return false;
    cache_invalidate_range(image, image->total_size);
    rmb();
    image_ = image;
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
    __atomic_store_n(&ack->publisher, request.publisher, __ATOMIC_RELAXED);
    __atomic_store_n(&ack->arg_index, cursor, __ATOMIC_RELAXED);
    __atomic_store_n(&ack->byte_cursor, bytes, __ATOMIC_RELAXED);
    __atomic_store_n(&ack->tick, request.tick, __ATOMIC_RELEASE);
    cache_flush_range(ack, sizeof(*ack));
    wmb();
    __atomic_store_n(
        &ack->status, scheduler_args_dump_ack_commit(request.task_id, request.stage, status), __ATOMIC_RELEASE
    );
    cache_flush_range(ack, sizeof(*ack));
    wmb();
}

bool ResidentArgsDumpService::service(uint32_t worker, const SchedulerArgsDumpRequest &request) {
    auto *task = scheduler_args_dump_task(image_, request.task_id);
    const auto stage = static_cast<ArgsDumpStage>(request.stage);
    bool valid = task != nullptr && request.run_epoch == image_->run_epoch &&
                 request.stage <= static_cast<uint64_t>(ArgsDumpStage::AFTER_COMPLETION) &&
                 request.publisher == scheduler_args_dump_publisher(worker) &&
                 request.arg_index <= CORE_MAX_TENSOR_ARGS + CORE_MAX_SCALAR_ARGS;
    if (task != nullptr) {
        // Observation only: AICPU never writes a cache line containing the
        // concurrently updated GM arrival/departure/abort words.
        cache_invalidate_range(task, sizeof(*task));
        rmb();
        if (valid) {
            const uint64_t publisher =
                stage == ArgsDumpStage::BEFORE_DISPATCH ? task->publisher_before : task->publisher_after;
            valid = publisher == request.publisher &&
                    (stage == ArgsDumpStage::BEFORE_DISPATCH ?
                         task->before_state == 1 :
                         task->returned_participants == task->expected_participants) &&
                    task->aborted == 0;
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
        DumpDescriptorView descriptor{};
        DumpPayloadView payload{};
        if (!observe_graph_arguments(*task, &descriptor, &payload)) {
            acknowledge(worker, request, SchedulerArgsDumpStatus::Aborted, request.arg_index, 0);
            return false;
        }
        LaneMask lanes;
        for (uint32_t slot = 0; slot < SUBTASK_SLOT_COUNT; ++slot)
            if ((task->active_mask & (UINT64_C(1) << slot)) != 0) lanes.activate(static_cast<SubtaskSlot>(slot));
        uint32_t cursor = static_cast<uint32_t>(request.arg_index);
        ArgsDumpInfo info{};
        if (!next_dump_arg_for_task<SUBTASK_SLOT_COUNT>(
                thread_, descriptor, payload, lanes, stage,
                [this](int32_t id) {
                    return callable(id);
                },
                &cursor, &info, &task->dump_metadata
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
