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

#pragma once

// This header is included inside the lane implementation's namespace.
inline __aicore__ bool calendar_args_dump_request(
    __gm__ SchedulerArgsDumpRequest *request, uint64_t epoch, uint64_t task, ArgsDumpStage stage, uint64_t publisher,
    uint64_t argument, uint64_t byte_cursor
) {
    const uint64_t tick = request->tick;
    if (tick == UINT64_MAX) return false;
    request->run_epoch = epoch;
    request->task_id = task;
    request->stage = static_cast<uint64_t>(stage);
    request->publisher = publisher;
    request->arg_index = argument;
    request->byte_cursor = byte_cursor;
    OUT_OF_ORDER_STORE_BARRIER();
    request->tick = tick + 1;
    dcci(request, SINGLE_CACHE_LINE, CACHELINE_OUT);
    percounter_memory_barrier();
    return true;
}

struct CalendarArgsDumpOutput {
    __gm__ const SchedulerArgsDumpTask *task;

    __aicore__ bool operator()() const {
#if defined(__CCE_AICORE__)
        pipe_barrier(PIPE_ALL);
#endif
        auto *tensors = reinterpret_cast<const __gm__ simpler::hbg::Tensor *>(task->tensors_address);
        for (uint32_t arg = 0; arg < task->tensor_count; ++arg) {
            if ((task->output_tensor_mask & (UINT64_C(1) << arg)) != 0 &&
                !scheduler_args_dump_tensor_lines(tensors + arg, [](uint64_t line) __aicore__ {
#if defined(__CCE_AICORE__)
                    dcci(reinterpret_cast<__gm__ void *>(line), SINGLE_CACHE_LINE, CACHELINE_OUT);
#else
                    (void)line;
#endif
                }))
                return false;
        }
        return true;
    }
};

inline __attribute__((noinline)) __aicore__ bool
calendar_args_dump_boundary(const LaneConfig &cfg, uint32_t task_ref, ArgsDumpStage stage) {
    auto *image = scheduler_state_at<SchedulerArgsDumpHeader>(cfg.control, cfg.control->args_dump_offset);
    const bool before = stage == ArgsDumpStage::BEFORE_DISPATCH;
    if (task_ref >= image->task_count) return false;
    auto *indices = scheduler_state_at<uint32_t>(image, image->task_indices_offset);
    percounter_observe_cache_line(indices + task_ref);
    auto *task = scheduler_args_dump_task(image, task_ref);
    if (task != nullptr) {
        percounter_observe_cache_line(&task->task_id);
        percounter_observe_cache_line(&task->dump_metadata);
    }
    if (before) {
        auto *context = &cfg.ssbuf->contexts[cfg.cluster_lane].value.global;
        percounter_ssbuf_store_field(
            &context->async_output_capture_refused,
            task != nullptr && task->output_tensor_mask != 0 ? SCHEDULER_ARGS_DUMP_ASYNC_OUTPUT_REFUSED : uint32_t{0}
        );
    }
    if (task == nullptr) return true;

    const uint32_t worker = cfg.cluster_lane == 0 ?
                                static_cast<uint32_t>(cfg.cluster_index) :
                                cfg.cluster_count + cfg.cluster_index * cfg.aiv_per_cluster + cfg.cluster_lane - 1;
    if (worker >= image->worker_count) return false;
    const auto arrival =
        before ? scheduler_args_dump_enter(task) : scheduler_args_dump_return(task, CalendarArgsDumpOutput{task});
    if (arrival == SchedulerArgsDumpArrival::Ready || (!before && arrival == SchedulerArgsDumpArrival::Waiting))
        return true;
    if (arrival == SchedulerArgsDumpArrival::Invalid) return false;

    const bool publisher = arrival == SchedulerArgsDumpArrival::Publisher;
    auto &publisher_word = before ? task->publisher_before : task->publisher_after;
    auto *request = scheduler_state_at<SchedulerArgsDumpRequest>(image, image->requests_offset) + worker;
    auto *acks = scheduler_state_at<SchedulerArgsDumpAck>(image, image->acknowledgements_offset);
    const uint64_t epoch = image->run_epoch;
    if (publisher) {
        const uint64_t id = scheduler_args_dump_publisher(worker);
        percounter_gm_store(publisher_word, id);
        percounter_memory_barrier();
        percounter_observe_cache_line(request);
        if (!calendar_args_dump_request(request, epoch, task_ref, stage, id, 0, 0)) return false;
    }

    const uint64_t timeout = cfg.control->scheduler_timeout_cycles;
    uint64_t progress_cycles = get_sys_cnt_aicore();
    uint64_t last_arg = 0;
    uint64_t last_bytes = 0;
    while (!percounter_should_stop(cfg)) {
        if (before && scheduler_args_dump_can_execute(task)) return true;
        const uint64_t id = scheduler_args_dump_load(publisher_word);
        if (id != 0 && id <= image->worker_count) {
            auto *ack = acks + id - 1;
            percounter_observe_cache_line(ack);
            SchedulerArgsDumpAck observed{};
            if (scheduler_args_dump_observe_ack(ack, &observed) &&
                scheduler_args_dump_ack_matches_phase(observed, epoch, task_ref, stage, id)) {
                const auto status = scheduler_args_dump_ack_status(observed);
                if (status == SchedulerArgsDumpStatus::Aborted) return false;
                if (publisher && status == SchedulerArgsDumpStatus::Sealed &&
                    scheduler_args_dump_seal(task, stage, observed, epoch, request->tick))
                    return true;
                if (status == SchedulerArgsDumpStatus::InProgress &&
                    (observed.arg_index > last_arg ||
                     (observed.arg_index == last_arg && observed.byte_cursor > last_bytes))) {
                    last_arg = observed.arg_index;
                    last_bytes = observed.byte_cursor;
                    progress_cycles = get_sys_cnt_aicore();
                }
                if (publisher && observed.tick == request->tick &&
                    (status == SchedulerArgsDumpStatus::InProgress || status == SchedulerArgsDumpStatus::WouldBlock) &&
                    !calendar_args_dump_request(
                        request, epoch, task_ref, stage, id, observed.arg_index, observed.byte_cursor
                    ))
                    return false;
            }
        }
        if (scheduler_watchdog_expired(progress_cycles, get_sys_cnt_aicore(), timeout)) {
            percounter_gm_store(task->aborted, UINT64_C(1));
            percounter_fail(cfg, SchedulerGraphResult::TIMEOUT, task_ref);
            return false;
        }
        SPIN_WAIT_HINT();
    }
    return false;
}
