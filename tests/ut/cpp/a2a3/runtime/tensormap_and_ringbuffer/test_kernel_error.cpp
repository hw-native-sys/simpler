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
#include <gtest/gtest.h>
#include <memory>
#include <cstring>
#include <thread>
#include <future>
#include <chrono>
#include <vector>
#include "kernel_error_api.h"
#include "runtime_core.h"
#include "scheduler/scheduler_context.h"
#include "aicpu/device_run_result_base_aicpu.h"

// Reuse the normal test logger, intercepting only the failure diagnostic so
// the orchestrator can run between primary publication and scheduler metadata.
#define unified_log_error default_unified_log_error
#include "../../../support/unified_log_stubs.cpp"
#undef unified_log_error
thread_local std::promise<void> *diagnostic_entered = nullptr;
thread_local std::shared_future<void> release_diagnostic;
extern "C" void unified_log_error(const char *func, const char *fmt, ...) {
    if (diagnostic_entered != nullptr && std::strstr(fmt, "Kernel failure:") != nullptr) {
        diagnostic_entered->set_value();
        release_diagnostic.wait();
    }
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "[ERROR] %s: ", func);
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
    va_end(args);
}

// Replace only the hardware clock. The second read is inside the actual
// blocked wait, after its first fatal-channel poll. No timing sleeps.
thread_local std::promise<void> *wait_entered = nullptr;
thread_local std::shared_future<void> release_wait;
thread_local int clock_reads = 0;
uint64_t get_sys_cnt_aicpu() {
    if (wait_entered != nullptr && ++clock_reads == 2) {
        wait_entered->set_value();
        release_wait.wait();
    }
    return 0;
}

class SchedulerContextTestPeer {
public:
    static void bind(SchedulerContext &context, SchedulerState &scheduler) { context.sched_ = &scheduler; }
    static void finish(
        SchedulerContext &context, ChipTaskSlotState &producer, int core, int block, const DeferredCompletionSlab &slab
    ) {
        context.deferred_slab_per_core_[core][0] = slab;
        context.payload_per_core_[core][0].local_context.block_idx = block;
        int completed = 0, released = 0;
        ChipTaskSlotState *release[1]{};
        context.complete_slot_task(
            producer, 0, SubtaskSlot::AIV0, core, core, nullptr, completed, release, released
#if SIMPLER_DFX
            ,
            0, 0
#endif
        );
        EXPECT_EQ(completed, 0);
        EXPECT_EQ(released, 0);
    }
};

namespace {

TEST(KernelError, PublicApiPreservesFirstError) {
    DeferredCompletionSlab slab{};
    auto context = AsyncCtx::make(TaskId::make(1, 264), &slab);
    ASSERT_TRUE(report_kernel_error(context, KernelErrorCode::INDEX_OUT_OF_RANGE));
    EXPECT_EQ(slab.error_code, 106);
    EXPECT_FALSE(report_kernel_error(context, KernelErrorCode::INVALID_OUTPUT));
    EXPECT_EQ(slab.count, 0u);
}

TEST(KernelError, MissingContextAndInvalidCodeDoNotWrite) {
    AsyncCtx missing{};
    EXPECT_FALSE(report_kernel_error(missing, KernelErrorCode::INVALID_OUTPUT));
    DeferredCompletionSlab slab{};
    auto context = AsyncCtx::make(TaskId::make(0, 1), &slab);
    EXPECT_FALSE(report_kernel_error(context, static_cast<KernelErrorCode>(0)));
    EXPECT_EQ(slab.error_code, 0);
}

TEST(KernelError, FailedBlockDoesNotCompleteOrReleaseConsumers) {
    auto context = std::make_unique<SchedulerContext>();
    auto scheduler = std::make_unique<SchedulerState>();
    auto header = std::make_unique<SharedMemoryHeader>();
    scheduler->sm_header = header.get();
    SchedulerContextTestPeer::bind(*context, *scheduler);
    TaskDescriptor task{};
    task.task_id = TaskId::make(1, 264);
    task.kernel_id[static_cast<int>(SubtaskSlot::AIV0)] = 219;
    TaskPayload payload{};
    ChipTaskSlotState producer{}, consumer{};
    producer.task = &task;
    producer.payload = &payload;
    producer.total_required_subtasks = 1;
    producer.task_state.store(CHIP_TASK_PENDING);
    consumer.fanin_count = 2;
    consumer.task_state.store(CHIP_TASK_PENDING);
    DepListEntry edge{&consumer, nullptr};
    producer.fanout_head = &edge;
    producer.fanout_count = 1;
    DeferredCompletionSlab slab{};
    auto kernel_context = AsyncCtx::make(task.task_id, &slab);
    ASSERT_TRUE(report_kernel_error(kernel_context, KernelErrorCode::INDEX_OUT_OF_RANGE));
    set_platform_run_result(0, 63);
    SchedulerContextTestPeer::finish(*context, producer, 0, 3, slab);
    EXPECT_TRUE(context->is_completed());
    EXPECT_EQ(header->sched_error_code.load(), 106);
    EXPECT_EQ(header->orch_error_code.load(), 106);
    EXPECT_EQ(producer.completed_subtasks.load(), 0);
    EXPECT_EQ(producer.task_state.load(), CHIP_TASK_PENDING);
    EXPECT_EQ(consumer.fanin_refcount.load(), 0);
    scheduler->on_task_complete(producer);
    EXPECT_EQ(consumer.fanin_refcount.load(), 1);
}

TEST(KernelError, ConcurrentBlocksPreserveFirstCode) {
    auto context = std::make_unique<SchedulerContext>();
    auto scheduler = std::make_unique<SchedulerState>();
    auto header = std::make_unique<SharedMemoryHeader>();
    scheduler->sm_header = header.get();
    SchedulerContextTestPeer::bind(*context, *scheduler);
    TaskDescriptor task{};
    task.task_id = TaskId::make(0, 17);
    TaskPayload payload{};
    ChipTaskSlotState producer{};
    producer.task = &task;
    producer.payload = &payload;
    producer.total_required_subtasks = 8;
    std::vector<std::thread> threads;
    for (int i = 0; i < 8; ++i) {
        threads.emplace_back([&, i] {
            DeferredCompletionSlab slab{};
            auto ctx = AsyncCtx::make(task.task_id, &slab);
            ASSERT_TRUE(report_kernel_error(ctx, static_cast<KernelErrorCode>(105 + i % 3)));
            SchedulerContextTestPeer::finish(*context, producer, i, i, slab);
        });
    }
    for (auto &thread : threads)
        thread.join();
    EXPECT_TRUE(is_kernel_error(header->sched_error_code.load()));
    EXPECT_TRUE(is_kernel_error(header->orch_error_code.load()));
    EXPECT_EQ(
        runtime_status_from_error_codes(header->orch_error_code.load(), header->sched_error_code.load()),
        -header->orch_error_code.load()
    );
    EXPECT_EQ(producer.completed_subtasks.load(), 0);
}
}  // namespace

TEST(KernelError, FailureWakesAlreadyWaitingOrchestrator) {
    DeviceArena sm_arena, runtime_arena;
    auto *sm = SharedMemoryHandle::create_and_init_default(sm_arena);
    ASSERT_NE(sm, nullptr);
    auto rt = std::make_unique<RuntimeContext>();
    int32_t windows[CHIP_MAX_RING_DEPTH];
    std::fill_n(windows, CHIP_MAX_RING_DEPTH, CHIP_TASK_WINDOW_SIZE);
    auto orch_layout = OrchestratorState::reserve_layout(runtime_arena, windows);
    auto sched_layout = SchedulerState::reserve_layout(runtime_arena);
    ASSERT_NE(runtime_arena.commit(), nullptr);
    std::vector<char> heap(4096 * CHIP_MAX_RING_DEPTH);
    ASSERT_TRUE(rt->orchestrator.init_data_from_layout(
        orch_layout, runtime_arena, sm->sm_base, heap.data(), 4096, CHIP_TASK_WINDOW_SIZE
    ));
    ASSERT_TRUE(rt->scheduler.init_data_from_layout(sched_layout, runtime_arena, sm->sm_base));
    rt->scheduler.wire_arena_pointers(sched_layout, runtime_arena);
    rt->orchestrator.wire_arena_pointers(orch_layout, runtime_arena, &rt->scheduler);
    runtime_finalize_after_wire(rt.get(), 1, 1);
    rt->orchestrator.begin_scope();
    uint32_t shape[] = {1};
    TensorCreateInfo info(shape, 1, DataType::INT32);
    CoreTaskArgs args;
    args.add_output(info);
    auto output = rt->orchestrator.submit_dummy_task(args);
    ASSERT_TRUE(output.task_id().is_valid());
    auto &slot = sm->header->rings[output.task_id().ring()].get_slot_state_by_task_id(output.task_id().local_id());
    slot.task_state.store(CHIP_TASK_PENDING);
    auto context = std::make_unique<SchedulerContext>();
    SchedulerContextTestPeer::bind(*context, rt->scheduler);
    std::promise<void> entered, resume;
    auto entered_event = entered.get_future();
    auto resume_event = resume.get_future().share();
    auto waiter = std::async(std::launch::async, [&] {
        wait_entered = &entered;
        release_wait = resume_event;
        uint32_t index[] = {0};
        return rt->ops->get_tensor_data(rt.get(), output.get_ref(0), 1, index);
    });
    ASSERT_EQ(entered_event.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    DeferredCompletionSlab slab{};
    auto ctx = AsyncCtx::make(output.task_id(), &slab);
    ASSERT_TRUE(report_kernel_error(ctx, KernelErrorCode::INDEX_OUT_OF_RANGE));
    SchedulerContextTestPeer::finish(*context, slot, 0, 0, slab);
    resume.set_value();
    ASSERT_EQ(waiter.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    EXPECT_EQ(waiter.get(), 0u);
    EXPECT_TRUE(rt->ops->is_fatal(rt.get()));
    rt->orchestrator.report_fatal(SIMPLER_ERROR_TENSOR_WAIT_TIMEOUT, "test", nullptr);
    EXPECT_EQ(sm->header->orch_error_code.load(), 106);
    EXPECT_EQ(sm->header->sched_error_code.load(), 106);
    EXPECT_EQ(slot.task_state.load(), CHIP_TASK_PENDING);
    rt->orchestrator.destroy();
    rt->scheduler.destroy();
}

TEST(KernelError, FailureWakesAllocationBackpressure) {
    auto context = std::make_unique<SchedulerContext>();
    auto scheduler = std::make_unique<SchedulerState>();
    auto header = std::make_unique<SharedMemoryHeader>();
    scheduler->sm_header = header.get();
    SchedulerContextTestPeer::bind(*context, *scheduler);
    TaskDescriptor descriptors[2]{};
    TaskPayload payload{};
    ChipTaskSlotState producer{};
    descriptors[0].task_id = TaskId::make(0, 0);
    producer.task = &descriptors[0];
    producer.payload = &payload;
    std::atomic<int32_t> current{0}, alive{0};
    char heap[256]{};
    TaskAllocator allocator;
    allocator.init(descriptors, 2, &current, &alive, heap, sizeof(heap), &header->orch_error_code, nullptr, 1);
    std::promise<void> entered, resume;
    auto entered_event = entered.get_future();
    auto resume_event = resume.get_future().share();
    auto waiter = std::async(std::launch::async, [&] {
        wait_entered = &entered;
        release_wait = resume_event;
        return allocator.alloc(0).failed();
    });
    ASSERT_EQ(entered_event.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    DeferredCompletionSlab slab{};
    auto ctx = AsyncCtx::make(descriptors[0].task_id, &slab);
    ASSERT_TRUE(report_kernel_error(ctx, KernelErrorCode::INDEX_OUT_OF_RANGE));
    SchedulerContextTestPeer::finish(*context, producer, 0, 0, slab);
    resume.set_value();
    ASSERT_EQ(waiter.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    EXPECT_TRUE(waiter.get());
    EXPECT_EQ(header->orch_error_code.load(), 106);
}

TEST(KernelError, OrchestrationStopObservesSharedFailure) {
    auto rt = std::make_unique<RuntimeContext>();
    auto header = std::make_unique<SharedMemoryHeader>();
    rt->orchestrator.sm_header = header.get();
    runtime_finalize_after_wire(rt.get(), 1, 1);
    EXPECT_FALSE(rt->ops->is_fatal(rt.get()));
    header->orch_error_code.store(106, std::memory_order_release);
    EXPECT_TRUE(rt->ops->is_fatal(rt.get()));
    EXPECT_TRUE(rt->orchestrator.fatal);
}

TEST(KernelError, LaterPoolFailuresPreserveValidationCode) {
    std::atomic<int32_t> error{106};
    FaninPool fanin{};
    fanin.error_code_ptr = &error;
    EXPECT_EQ(fanin.alloc(), nullptr);
    EXPECT_EQ(error.load(), 106);
    DepListPool fanout{};
    fanout.error_code_ptr = &error;
    EXPECT_EQ(fanout.alloc(), nullptr);
    EXPECT_EQ(error.load(), 106);
}

TEST(KernelError, TimeoutBetweenPrimaryPublicationAndDiagnosticsKeepsKernelCode) {
    auto context = std::make_unique<SchedulerContext>();
    auto scheduler = std::make_unique<SchedulerState>();
    auto header = std::make_unique<SharedMemoryHeader>();
    scheduler->sm_header = header.get();
    SchedulerContextTestPeer::bind(*context, *scheduler);
    OrchestratorState orch{};
    orch.sm_header = header.get();
    TaskDescriptor task{};
    task.task_id = TaskId::make(0, 1);
    TaskPayload payload{};
    ChipTaskSlotState producer{};
    producer.task = &task;
    producer.payload = &payload;
    DeferredCompletionSlab slab{};
    auto ctx = AsyncCtx::make(task.task_id, &slab);
    ASSERT_TRUE(report_kernel_error(ctx, KernelErrorCode::INDEX_OUT_OF_RANGE));
    std::promise<void> entered, resume;
    auto entered_event = entered.get_future();
    auto resume_event = resume.get_future().share();
    auto completion = std::async(std::launch::async, [&] {
        diagnostic_entered = &entered;
        release_diagnostic = resume_event;
        SchedulerContextTestPeer::finish(*context, producer, 0, 0, slab);
    });
    ASSERT_EQ(entered_event.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    // Run the real timeout publication while completion is paused at its log.
    orch.report_fatal(SIMPLER_ERROR_TENSOR_WAIT_TIMEOUT, "test", nullptr);
    resume.set_value();
    completion.get();
    EXPECT_EQ(header->orch_error_code.load(), 106);
    EXPECT_EQ(runtime_status_from_error_codes(header->orch_error_code.load(), header->sched_error_code.load()), -106);
}

TEST(KernelError, EarlierOrchestrationFailureRemainsPrimary) {
    auto context = std::make_unique<SchedulerContext>();
    auto scheduler = std::make_unique<SchedulerState>();
    auto header = std::make_unique<SharedMemoryHeader>();
    scheduler->sm_header = header.get();
    SchedulerContextTestPeer::bind(*context, *scheduler);
    OrchestratorState orch{};
    orch.sm_header = header.get();
    orch.report_fatal(SIMPLER_ERROR_TENSOR_WAIT_TIMEOUT, "test", nullptr);
    TaskDescriptor task{};
    task.task_id = TaskId::make(0, 1);
    TaskPayload payload{};
    ChipTaskSlotState producer{};
    producer.task = &task;
    producer.payload = &payload;
    DeferredCompletionSlab slab{};
    auto ctx = AsyncCtx::make(task.task_id, &slab);
    ASSERT_TRUE(report_kernel_error(ctx, KernelErrorCode::INDEX_OUT_OF_RANGE));
    SchedulerContextTestPeer::finish(*context, producer, 0, 0, slab);
    EXPECT_EQ(header->sched_error_code.load(), 106);
    EXPECT_EQ(header->orch_error_code.load(), SIMPLER_ERROR_TENSOR_WAIT_TIMEOUT);
    EXPECT_EQ(runtime_status_from_error_codes(header->orch_error_code.load(), header->sched_error_code.load()), -8);
}

TEST(KernelError, ExistingSchedulerFailureRemainsPrimary) {
    auto context = std::make_unique<SchedulerContext>();
    auto scheduler = std::make_unique<SchedulerState>();
    auto header = std::make_unique<SharedMemoryHeader>();
    scheduler->sm_header = header.get();
    header->sched_error_code.store(SIMPLER_ERROR_SCHEDULER_TIMEOUT);
    SchedulerContextTestPeer::bind(*context, *scheduler);
    TaskDescriptor task{};
    task.task_id = TaskId::make(0, 1);
    TaskPayload payload{};
    ChipTaskSlotState producer{};
    producer.task = &task;
    producer.payload = &payload;
    DeferredCompletionSlab slab{};
    auto ctx = AsyncCtx::make(task.task_id, &slab);
    ASSERT_TRUE(report_kernel_error(ctx, KernelErrorCode::INDEX_OUT_OF_RANGE));
    SchedulerContextTestPeer::finish(*context, producer, 0, 0, slab);
    EXPECT_EQ(header->sched_error_code.load(), SIMPLER_ERROR_SCHEDULER_TIMEOUT);
    EXPECT_EQ(runtime_status_from_error_codes(header->orch_error_code.load(), header->sched_error_code.load()), -100);
}
