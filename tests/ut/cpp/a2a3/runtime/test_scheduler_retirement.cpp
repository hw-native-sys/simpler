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

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <future>
#include <memory>
#include <mutex>
#include <thread>

#include "runtime.h"
#include "scheduler/scheduler_context.h"
#ifdef PTO_ASYNC_WAIT_H
#include "runtime_core.h"
#else
#include "common/host_build_graph/hbg_graph_image_test_support.h"
#include "common/native_invocation_evidence.h"
// Exercise the private HBG executor that snapshots the scheduler header.
// NOLINTNEXTLINE(bugprone-suspicious-include)
#include "aicpu/aicpu_executor.cpp"
#endif

namespace {
constexpr int kCores = 6;
std::array<std::atomic<int>, kCores> opened{};
std::array<std::atomic<int>, kCores> retired{};
std::array<uint32_t, kCores> conditions{};
std::atomic<int> batches{0};
std::atomic<int> deadlines{0};
std::mutex init_mutex;
std::condition_variable init_cv;
bool pause_open = false;
bool open_paused = false;
bool resume_open = false;
bool silent_first_core = false;
std::shared_future<void> peer_dispatch_exits;
#ifndef PTO_ASYNC_WAIT_H
struct ExitProbe {
    alignas(8) uint64_t data_main = AICPU_IDLE_TASK_ID;
    bool pause_doorbell = false;
    std::promise<void> doorbell_entered;
    std::promise<void> resume_doorbell;
    std::promise<void> first_sweep;
    std::promise<void> resume_ack;
    bool ack = false;
};
ExitProbe *exit_probe = nullptr;
#endif

template <typename Context>
auto initialize(Context &context, Runtime *runtime, uint64_t regs, int)
    -> decltype(context.pre_handshake_init(runtime, 3, 2, regs)) {
    return context.pre_handshake_init(runtime, 3, 2, regs);
}
template <typename Context>
auto initialize(Context &context, Runtime *runtime, uint64_t regs, long)
    -> decltype(context.pre_handshake_init(runtime, 3, regs)) {
    return context.pre_handshake_init(runtime, 3, regs);
}

template <typename Context>
auto handshake_owned(Context &context, Runtime *runtime, int owner, int)
    -> decltype(context.handshake_owned_clusters(runtime, owner, 2), bool()) {
    context.handshake_owned_clusters(runtime, owner, 2);
    context.assign_own_clusters(runtime, owner);
    return true;
}
template <typename Context>
bool handshake_owned(Context &, Runtime *, int, long) {
    return false;
}

template <typename Context>
auto handshake_owned_only(Context &context, Runtime *runtime, int owner, int)
    -> decltype(context.handshake_owned_clusters(runtime, owner, 2), bool()) {
    context.handshake_owned_clusters(runtime, owner, 2);
    return true;
}

template <typename Context>
bool handshake_owned_only(Context &, Runtime *, int, long) {
    return false;
}

template <typename Context>
auto assign_owned(Context &context, Runtime *runtime, int owner, int)
    -> decltype(context.assign_own_clusters(runtime, owner), bool()) {
    context.assign_own_clusters(runtime, owner);
    return true;
}

template <typename Context>
bool assign_owned(Context &, Runtime *, int, long) {
    return false;
}
}  // namespace

uint32_t platform_get_physical_cores_count() { return kCores; }
static std::atomic<int> nonce_batches{0};
#ifdef PTO_ASYNC_WAIT_H
static uint64_t test_report_epoch = 0;
extern "C" uint64_t get_platform_run_result_epoch() { return test_report_epoch; }
static void set_test_epoch(uint64_t epoch) { test_report_epoch = epoch; }
#else
static void set_test_epoch(uint64_t epoch) { set_platform_run_result(0, epoch); }
#endif
uint64_t platform_aicore_exit_deadline() { return ++deadlines; }
uint64_t read_reg(uint64_t, RegId) { return AICORE_IDLE_VALUE; }
volatile uint32_t *get_reg_ptr(uint64_t address, RegId reg) {
#ifndef PTO_ASYNC_WAIT_H
    if (exit_probe && reg == RegId::DATA_MAIN_BASE && address == 2) {
        if (exit_probe->pause_doorbell) {
            exit_probe->doorbell_entered.set_value();
            EXPECT_EQ(
                exit_probe->resume_doorbell.get_future().wait_for(std::chrono::seconds(3)), std::future_status::ready
            );
        }
        return reinterpret_cast<volatile uint32_t *>(&exit_probe->data_main);
    }
#endif
    return &conditions.at(address - 1);
}

#ifdef PTO_ASYNC_WAIT_H
void AsyncWaitList::log_diagnostics(AICoreCompletionMailbox *, const char *) {}
#endif

void platform_init_aicore_regs(uint64_t address) {
    ++opened.at(address - 1);
    std::unique_lock<std::mutex> lock(init_mutex);
    if (pause_open && address == kCores) {
        open_paused = true;
        init_cv.notify_all();
        init_cv.wait(lock, [] {
            return resume_open;
        });
    }
}

int32_t platform_retire_aicore_group(
    const AicoreExitTarget *targets, size_t count, uint64_t, bool *released, AicoreExitReadiness readiness
) {
    ++nonce_batches;
    ++batches;
    if (peer_dispatch_exits.valid()) {
        EXPECT_EQ(peer_dispatch_exits.wait_for(std::chrono::seconds(2)), std::future_status::ready);
    }
    // Mirror one platform sweep. The test controls when the ACK deadline expires.
    bool sent[PLATFORM_MAX_CORES]{};

    for (size_t i = 0; i < count; ++i) {
        const auto core = targets[i].reg_addr - 1;
        EXPECT_LT(core, kCores);
        if (core >= kCores) continue;
        EXPECT_EQ(opened[core].load(), 1);
        EXPECT_NE(targets[i].teardown, nullptr);
        sent[i] = !readiness.ready || readiness.ready(readiness.context, i);
        if (!sent[i]) continue;
        ++retired[core];
#ifndef PTO_ASYNC_WAIT_H
        if (exit_probe && core == 1) exit_probe->data_main = AICORE_EXIT_SIGNAL;
#endif
    }
    bool timed_out = false;
    for (size_t i = 0; i < count; ++i) {
        const auto core = targets[i].reg_addr - 1;
        if (core >= kCores) continue;
        bool success = sent[i] && !(silent_first_core && core == 0);
#ifndef PTO_ASYNC_WAIT_H
        if (exit_probe && core == 1) {
            exit_probe->first_sweep.set_value();
            EXPECT_EQ(exit_probe->resume_ack.get_future().wait_for(std::chrono::seconds(3)), std::future_status::ready);
            exit_probe->ack = sent[i] && exit_probe->data_main == AICORE_EXIT_SIGNAL;
            success = exit_probe->ack;
        }
#endif
        if (released != nullptr) released[i] = success;
        if (success && targets[i].teardown != nullptr) {
            __atomic_store_n(&targets[i].teardown->post_close_release, AICORE_POST_CLOSE_RELEASE, __ATOMIC_RELEASE);
        }
        timed_out |= !success;
    }
    return timed_out ? -1 : 0;
}

class SchedulerContextTestPeer {
public:
#ifdef PTO_ASYNC_WAIT_H
    static void bind_async(SchedulerContext &context, RuntimeContext &runtime) {
        context.rt_ = &runtime;
        context.sched_ = &runtime.scheduler;
    }
    static DeferredCompletionSlab &slab(SchedulerContext &context, TaskId task) {
        context.payload_per_core_[0][0].local_context.async_ctx.task_token = task;
        return context.deferred_slab_per_core_[0][0];
    }
#else
    static void fail_requeue(SchedulerContext &ctx, Runtime *runtime) {
        ctx.fail_scheduler(runtime, 0, SIMPLER_ERROR_READY_QUEUE_OVERFLOW);
    }
    static auto prepared_token(uint64_t &data_main) {
        return SchedulerContext::PublishHandle{
            reinterpret_cast<uint64_t>(&data_main) - reg_offset(RegId::DATA_MAIN_BASE), 17, 1, nullptr,
            TASK_TIMING_SLOT_NONE
        };
    }
    static void publish_prepared(SchedulerContext &ctx, const SchedulerContext::PublishHandle &handle) {
        RegisterPublications::CoreMask cores{};
        ctx.publish_task_batch(1, &handle, 1, cores, [&] {
            ctx.publish_subtask_to_core(handle, 0, 1);
        });
    }
    static DeferredCompletionSlab &finish_first_core(SchedulerContext &context, int thread, ChipTaskSlotState &slot) {
        auto &tracker = context.core_trackers_[thread];
        const int core_id = tracker.get_core_id_by_offset(0);
        auto &core = context.core_exec_states_[core_id];
        core.running_slot_state = &slot;
        core.running_reg_task_id = 0;
        core.running_subslot = SubtaskSlot::AIC;
        tracker.change_core_state(0);
        conditions[core_id] = MAKE_FIN_VALUE(0);
        return context.deferred_slab_per_core_[core_id][0];
    }

    static CompletedTaskQueue &completion_queue(SchedulerContext &context, int thread) {
        return context.sp_queues_[thread];
    }

    // Production retains the buffers on its static executor for the next attach.
    static void release_completed_queues(SchedulerContext &context) {
        for (auto &queue : context.sp_queues_)
            queue.destroy();
    }
#endif
    static int32_t request_all(SchedulerContext &context, Runtime *runtime) {
        return context.retire_all_cores(runtime);
    }
    static void emergency(SchedulerContext &context, Runtime *runtime) { context.emergency_shutdown(runtime); }
    static bool handshake_failed(const SchedulerContext &context) {
        return context.handshake_failed_.load(std::memory_order_acquire);
    }
    static int32_t request(SchedulerContext &context, Runtime *runtime, const int32_t *ids, int count) {
        return context.retire_cores(runtime, ids, count);
    }
    static void clear_register_address(SchedulerContext &context, int core) {
        context.core_exec_states_[core].reg_addr = 0;
    }
    static void clear_ownership(SchedulerContext &context) {
        context.core_trackers_[0].init(0);
        context.core_trackers_[1].init(0);
    }
    static void force_assignment_failure(SchedulerContext &context) {
        // Fault-injected discovery geometry stays within the allocated tables.
        context.cores_total_num_ = PLATFORM_MAX_CORES;
        context.aicpu_thread_num_ = 2;
        for (int i = 0; i < PLATFORM_MAX_CORES; ++i)
            context.core_type_compact_[i] = static_cast<uint8_t>(CoreType::AIC);
    }
    static void assign(SchedulerContext &context) {
        context.core_trackers_[0].init(1);
        context.core_trackers_[0].set_cluster(0, 0, 2, 3);
        context.core_trackers_[1].init(1);
        context.core_trackers_[1].set_cluster(0, 1, 4, 5);
    }
};

class SchedulerRetirement : public testing::Test {
protected:
#ifdef PTO_ASYNC_WAIT_H
    std::unique_ptr<SchedulerContext> context = std::make_unique<SchedulerContext>();
#else
    std::unique_ptr<AicpuExecutor> executor = std::make_unique<AicpuExecutor>();
    SchedulerContext *context = &executor->sched_ctx_;
#endif
    std::unique_ptr<Runtime> runtime = std::make_unique<Runtime>();
    std::array<uint64_t, kCores> regs{};

    void SetUp() override {
        set_test_epoch(0);
        nonce_batches = 0;
        batches = 0;
        deadlines = 0;
        pause_open = false;
        open_paused = false;
        resume_open = false;
        silent_first_core = false;
        for (int i = 0; i < kCores; ++i) {
            opened[i] = 0;
            retired[i] = 0;
            regs[i] = i + 1;
        }
        runtime->set_worker_count(kCores);
        ASSERT_EQ(initialize(*context, runtime.get(), reinterpret_cast<uint64_t>(regs.data()), 0), 0);
        for (int i = 0; i < kCores; ++i) {
            auto &handshake = runtime->get_workers()[i];
            handshake.physical_core_id = i;
            handshake.core_type = i < 2 ? CoreType::AIC : CoreType::AIV;
            handshake.aicore_done = 1;
        }
        SchedulerContextTestPeer::assign(*context);
    }
#ifndef PTO_ASYNC_WAIT_H
    void TearDown() override {
        exit_probe = nullptr;
        peer_dispatch_exits = {};
        SchedulerContextTestPeer::release_completed_queues(*context);
    }
#endif
    void handshake() {
        context->handshake_partition(runtime.get(), 0, 1);
        const bool failed = SchedulerContextTestPeer::handshake_failed(*context);
        EXPECT_EQ(context->post_handshake_init(runtime.get()), failed ? -1 : 0);
    }
    int32_t request_all() { return SchedulerContextTestPeer::request_all(*context, runtime.get()); }
    void emergency() { SchedulerContextTestPeer::emergency(*context, runtime.get()); }
    void expect_once(int unacknowledged_core = -1) {
        for (int i = 0; i < kCores; ++i) {
            EXPECT_EQ(retired[i].load(), 1) << "core " << i;
            EXPECT_EQ(
                runtime->get_teardown_gates()[i].post_close_release,
                ((silent_first_core && i == 0) || i == unacknowledged_core) ? 0U : AICORE_POST_CLOSE_RELEASE
            ) << "core "
              << i;
        }
    }
};

#ifndef PTO_ASYNC_WAIT_H
namespace {
void seed_two_slots(ChipReadyQueue &queue, ChipReadyQueueSlot (&slots)[2]) {
    queue.slots = slots;
    queue.capacity = 2;
    queue.mask = 1;
    queue.seed_slots();
}
}  // namespace

class HbgDispatchFailure : public SchedulerRetirement {
protected:
    static constexpr uint64_t epoch = 43;
    NativeInvocationResultStorage storage{};
    SharedMemoryHeader header{};
    SchedulerState scheduler{};
    RuntimeContext rt{};
    ChipReadyQueueSlot ready_slots[NUM_RESOURCE_SHAPES][2]{}, sync_slots[NUM_RESOURCE_SHAPES][2]{};
    ChipReadyQueueSlot early_slots[NUM_RESOURCE_SHAPES][2]{}, prepare_slots[2]{}, early_sync_slots[2]{},
        dummy_slots[2]{};

    void SetUp() override {
        SchedulerRetirement::SetUp();
        ASSERT_EQ(aicpu_begin_native_invocation(3, reinterpret_cast<uint64_t>(&storage), epoch), 0);
        for (int core = 0; core < kCores; ++core)
            runtime->get_workers()[core].report_epoch = epoch;
        handshake();
        runtime->set_gm_sm_ptr(&header);
        scheduler.sm_header = &header;
        rt.scheduler = &scheduler;
        context->bind_runtime(&rt);
        seed_two_slots(scheduler.graph_prepare_queue, prepare_slots);
        seed_two_slots(scheduler.early_sync_start_queue, early_sync_slots);
        seed_two_slots(scheduler.dummy_ready_queue, dummy_slots);
        for (int shape = 0; shape < NUM_RESOURCE_SHAPES; ++shape) {
            seed_two_slots(scheduler.ready_queues[shape], ready_slots[shape]);
            seed_two_slots(scheduler.ready_sync_queues[shape], sync_slots[shape]);
            seed_two_slots(scheduler.early_dispatch_queues[shape], early_slots[shape]);
        }
    }

    void attach_tasks(int32_t count) {
        header.tasks.total_tasks = count;
        context->on_graph_attached(&rt, 0, header.tasks.total_tasks);
    }

    void expect_header_terminal(int32_t code) {
        executor->snapshot_run_terminal(runtime.get());
        executor->handoff_run_terminal();
        EXPECT_EQ(aicpu_finish_native_invocation(0, 0), 0);
        EXPECT_EQ(aicpu_finish_native_invocation(1, 0), 0);
        EXPECT_EQ(aicpu_finish_native_invocation(2, 0), -code);
        auto terminal = device_run_result_terminal(storage.result, epoch);
        EXPECT_EQ(terminal.state, DeviceRunTerminalState::Failed);
        EXPECT_EQ(terminal.source, DeviceRunCodeSource::Header);
        EXPECT_EQ(terminal.code, -code);
    }
};

class HbgSlabFailure : public HbgDispatchFailure,
                       public testing::WithParamInterface<int32_t> {};

TEST_P(HbgSlabFailure, FailedFinPreservesReporterRetiresCoresAndReturnsNoCompletion) {
    attach_tasks(1);
    ChipTaskStorage task{};
    auto &slab = SchedulerContextTestPeer::finish_first_core(*context, 1, task.slot);
    const int32_t expected = GetParam();
    if (expected == SIMPLER_ERROR_ASYNC_REGISTRATION_FAILED) {
        slab.count = MAX_COMPLETIONS_PER_TASK + 1;
    } else {
        slab.error_code = expected;
    }

    EXPECT_EQ(context->resolve_and_dispatch(runtime.get(), 1), 0);
    EXPECT_EQ(header.sched_error_code.load(), expected);
    EXPECT_EQ(header.sched_error_thread.load(), 1);
    EXPECT_EQ(header.sched_error_bitmap.load(), 1u << 1);
    EXPECT_EQ(task.slot.completed_subtasks.load(), 0);
    EXPECT_EQ(batches.load(), 1);
    expect_once();
    expect_header_terminal(expected);
}
INSTANTIATE_TEST_SUITE_P(
    DeferredSlab, HbgSlabFailure,
    testing::Values(
        SIMPLER_ERROR_ASYNC_COMPLETION_INVALID, SIMPLER_ERROR_ASYNC_WAIT_OVERFLOW,
        SIMPLER_ERROR_ASYNC_REGISTRATION_FAILED
    )
);

class HbgWakeStop : public HbgDispatchFailure,
                    public testing::WithParamInterface<bool> {};

TEST_P(HbgWakeStop, FailedFirstWaiterLeavesSecondWaiterUnrouted) {
    std::array<ChipTaskStorage, 3> tasks{};
    std::array<std::atomic<ChipTaskState>, 3> states{};
    header.tasks.task_storage = tasks.data();
    header.tasks.task_states = states.data();
    header.tasks.total_tasks = 3;
    scheduler.task_view.tasks = &header.tasks;
    GraphExecution graph{};
    int32_t offsets[] = {0, 0, 1, 2};
    graph.fanin_offsets = offsets;
    graph.definition_offset = GRAPH_DEFINITION_OBJECT_ALIGN;
    graph.task_storage = tasks.data();
    graph.task_states = states.data();
    graph.task_count = 3;
    graph.remaining_tasks.store(3);
    graph_execution_set_phase(graph, GraphExecutionPhase::MATERIALIZED);
    for (int i = 0; i < 3; ++i) {
        header.tasks.reset_task_state(i);
        tasks[i].task.task_id = TaskId::make_global(i);
        tasks[i].slot.sub_task_local_id = i;
        tasks[i].slot.graph_context = GetParam() ? &graph : nullptr;
    }
    tasks[0].slot.wake_list_head.store(&tasks[1].slot);
    tasks[1].slot.next_in_wake_list = &tasks[2].slot;
    tasks[1].payload.fanin_count = tasks[2].payload.fanin_count = 1;
    tasks[1].slot.active_mask = LaneMask::aic();
    // The first waiter's AIC queue is full. The second waiter's DUMMY queue has room.
    auto &full = scheduler.ready_queues[static_cast<int>(ResourceShape::AIC)];
    ASSERT_TRUE(full.push(&tasks[0].slot));
    ASSERT_TRUE(full.push(&tasks[0].slot));
    const auto outcome = scheduler.complete_task(tasks[0].slot, 1);
    EXPECT_EQ(outcome.error_code, SIMPLER_ERROR_READY_QUEUE_OVERFLOW);
    EXPECT_EQ(outcome.stream_tasks_completed, 0);
    EXPECT_EQ(header.sched_error_code.load(), SIMPLER_ERROR_READY_QUEUE_OVERFLOW);
    EXPECT_EQ(header.sched_error_thread.load(), 1);
    EXPECT_EQ(scheduler.dummy_ready_queue.size(), 0u);
}
INSTANTIATE_TEST_SUITE_P(OrdinaryAndGraph, HbgWakeStop, testing::Bool());

TEST_F(HbgDispatchFailure, NormalFinWaitsForSpaceOnFullCompletionQueue) {
    attach_tasks(1);
    auto &queue = SchedulerContextTestPeer::completion_queue(*context, 0);
    ChipTaskStorage queued{}, next{};
    const std::atomic<bool> stopped{false};
    ASSERT_EQ(queue.cap, 256u);
    for (uint64_t i = 0; i < queue.cap; ++i)
        ASSERT_TRUE(queue.push(&queued.slot, stopped));
    auto producer = std::async(std::launch::async, [&] {
        return queue.push(&next.slot, stopped);
    });
    EXPECT_EQ(producer.wait_for(std::chrono::milliseconds(20)), std::future_status::timeout);
    EXPECT_EQ(queue.pop(), &queued.slot);
    EXPECT_TRUE(producer.get());
    for (uint64_t i = 1; i < queue.cap; ++i)
        EXPECT_EQ(queue.pop(), &queued.slot);
    EXPECT_EQ(queue.pop(), &next.slot);
}

TEST_F(HbgDispatchFailure, FullCompletionQueueReleasesOrdinaryFinProducerOnStop) {
    attach_tasks(2);
    ChipTaskStorage queued{}, blocked{};
    blocked.task.task_id = TaskId::make_global(0);
    blocked.slot.total_required_subtasks = 1;
    SchedulerContextTestPeer::finish_first_core(*context, 0, blocked.slot);
    auto &queue = SchedulerContextTestPeer::completion_queue(*context, 0);
    const std::atomic<bool> stopped{false};
    for (uint64_t i = 0; i < queue.cap; ++i)
        ASSERT_TRUE(queue.push(&queued.slot, stopped));
    auto producer = std::async(std::launch::async, [&] {
        return context->resolve_and_dispatch(runtime.get(), 0);
    });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!blocked.slot.completed_subtasks.load() && std::chrono::steady_clock::now() < deadline)
        std::this_thread::yield();
    EXPECT_EQ(blocked.slot.completed_subtasks.load(), 1);
    SchedulerContextTestPeer::fail_requeue(*context, runtime.get());
    if (producer.wait_for(std::chrono::seconds(2)) != std::future_status::ready) {
        ADD_FAILURE() << "ordinary FIN waited for a stopped completion consumer";
        queue.pop();  // Unblock a regressed producer so the suite can finish.
    }
    EXPECT_EQ(producer.get(), 0);
    EXPECT_EQ(queue.tail.load(), queue.cap);
    expect_header_terminal(SIMPLER_ERROR_READY_QUEUE_OVERFLOW);
}

class HbgMailboxStop : public HbgDispatchFailure,
                       public testing::WithParamInterface<uint32_t> {};

TEST_P(HbgMailboxStop, BlockedFinExitsWhenAnotherSchedulerStopsTheRun) {
    auto mailbox = std::make_unique<AICoreCompletionMailbox>();
    mailbox->init_empty();
    rt.aicore_mailbox = mailbox.get();
    attach_tasks(2);
    ChipTaskStorage blocked{}, failed{};
    blocked.task.task_id = TaskId::make_global(0);
    blocked.slot.total_required_subtasks = 1;
    auto &slab = SchedulerContextTestPeer::finish_first_core(*context, 0, blocked.slot);
    if (GetParam() == MSG_KIND_CONDITION) {
        slab.count = 1;  // The CONDITION cannot be published.
    } else {
        blocked.slot.mark_any_subtask_deferred();  // The last subtask must publish NORMAL_DONE.
    }
    SchedulerContextTestPeer::finish_first_core(*context, 1, failed.slot).error_code =
        SIMPLER_ERROR_ASYNC_COMPLETION_INVALID;
    for (uint64_t i = 0; i < AICORE_COMPLETION_MAILBOX_CAPACITY; ++i)
        ASSERT_TRUE(mailbox->try_push_normal_done(blocked.task.task_id, 0));

    auto producer = std::async(std::launch::async, [&] {
        return context->resolve_and_dispatch(runtime.get(), 0);
    });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (scheduler.async_wait_list.mpsc_skipped_count.load() == 0 && std::chrono::steady_clock::now() < deadline)
        std::this_thread::yield();
    EXPECT_GT(scheduler.async_wait_list.mpsc_skipped_count.load(), 0u);
    EXPECT_EQ(context->resolve_and_dispatch(runtime.get(), 1), 0);
    if (producer.wait_for(std::chrono::seconds(2)) != std::future_status::ready) {
        ADD_FAILURE() << "FIN producer still waits on a stopped mailbox consumer";
        // Unblock the old producer so a regression fails instead of hanging the suite.
        AICoreCompletionMsgView msg{};
        mailbox->try_pop(msg);
        mailbox->try_pop(msg);
    }
    EXPECT_EQ(producer.get(), 0);
    EXPECT_EQ(mailbox->head.load(), AICORE_COMPLETION_MAILBOX_CAPACITY);
    EXPECT_EQ(header.sched_error_code.load(), SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
    EXPECT_EQ(header.sched_error_thread.load(), 1);
    EXPECT_EQ(header.sched_error_bitmap.load(), 1u << 1);
    EXPECT_EQ(batches.load(), 1);
    expect_once();
    expect_header_terminal(SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
}
INSTANTIATE_TEST_SUITE_P(FullMailbox, HbgMailboxStop, testing::Values(MSG_KIND_CONDITION, MSG_KIND_TASK_NORMAL_DONE));

TEST_F(HbgDispatchFailure, GraphBodyReadyOverflowStopsBeforeDispatchingQueuedTask) {
    auto definition = make_test_definition(42, 0, 1, TestRoot::ORDINARY, TestEdDefect::NONE, TestBoundary::NONE);
    TestDefinitionObject image(definition);
    const auto view = image.view();
    runtime->publish_graph_section(
        GraphSectionSource::HostSnapshot, reinterpret_cast<uint64_t>(view.section), view.bytes
    );
    OuterHeap heap(definition);
    ChipTaskStorage graph{};
    std::atomic<ChipTaskState> state{};
    attach_tasks(1);
    header.tasks.task_storage = &graph;
    header.tasks.task_states = &state;
    scheduler.task_view.tasks = &header.tasks;
    uint64_t boundary_scalar = 17;
    graph.payload.scalar_count = 1;
    graph.payload.bind_regions(nullptr, &boundary_scalar, nullptr);
    graph.slot.task_kind = TaskKind::GRAPH;
    graph.task.task_id = TaskId::make_global(0);
    graph.task.graph_definition_offset = image.image_offset();
    graph.task.packed_buffer_base = heap.base();
    graph.task.packed_buffer_end = heap.end();
    header.tasks.reset_task_state(0);
    context->classify_partition(0, 1, view);
    ASSERT_NE(graph.slot.graph_context, nullptr);
    ASSERT_EQ(scheduler.graph_prepare_queue.size(), 1u);
    ASSERT_EQ(header.sched_error_code.load(), SIMPLER_ERROR_NONE);

    // A second ready task still has blocks to dispatch when the Graph root finds the AIC queue full.
    ChipTaskStorage waiting{};
    waiting.slot.active_mask = LaneMask::aic();
    waiting.slot.logical_block_num = 2;
    auto &queue = scheduler.ready_queues[static_cast<int>(ResourceShape::AIC)];
    ChipTaskStorage other{};
    other.slot.active_mask = LaneMask::aic();
    ASSERT_TRUE(queue.push(&other.slot));
    ASSERT_TRUE(queue.push(&waiting.slot));
    // A consumer has claimed the first cell but has not freed it. Dispatch can still take the second.
    queue.dequeue_pos.store(1);
    EXPECT_EQ(context->resolve_and_dispatch(runtime.get(), 1), 0);
    EXPECT_EQ(header.sched_error_code.load(), SIMPLER_ERROR_READY_QUEUE_OVERFLOW);
    EXPECT_EQ(header.sched_error_thread.load(), 1);
    EXPECT_EQ(header.sched_error_bitmap.load(), 1u << 1);
    EXPECT_EQ(queue.dequeue_pos.load(), 1u);
    EXPECT_EQ(waiting.slot.next_block_idx.load(), 0);
    EXPECT_EQ(batches.load(), 1);
    expect_once();
    expect_header_terminal(SIMPLER_ERROR_READY_QUEUE_OVERFLOW);
}

TEST_F(SchedulerRetirement, ClassifyPartitionReportsOrdinaryReadyQueueOverflowWithThreadIdentity) {
    SharedMemoryHeader header{};
    SchedulerState scheduler{};
    scheduler.sm_header = &header;
    RuntimeContext rt{};
    rt.scheduler = &scheduler;
    context->bind_runtime(&rt);
    context->on_graph_attached(&rt, 0, 3);
    std::array<ChipTaskStorage, 3> tasks{};
    std::array<std::atomic<ChipTaskState>, 3> states{};
    header.tasks.total_tasks = 3;
    header.tasks.task_storage = tasks.data();
    header.tasks.task_states = states.data();
    scheduler.task_view.tasks = &header.tasks;
    ChipReadyQueueSlot slots[2]{};
    auto &queue = scheduler.ready_queues[static_cast<int>(ResourceShape::AIC)];
    seed_two_slots(queue, slots);
    for (int i = 0; i < 3; ++i) {
        tasks[i].slot.active_mask = {true, false, false};
        tasks[i].task.task_id = TaskId::make_global(i);
        header.tasks.reset_task_state(i);
        context->classify_partition(i, 3, {});
        if (i < 2) ASSERT_EQ(header.sched_error_code.load(), SIMPLER_ERROR_NONE);
    }
    EXPECT_EQ(queue.size(), 2u);
    EXPECT_EQ(header.sched_error_code.load(), SIMPLER_ERROR_READY_QUEUE_OVERFLOW);
    EXPECT_EQ(header.sched_error_thread.load(), 2);
    EXPECT_EQ(header.sched_error_bitmap.load(), 1u << 2);
}

class HbgEarlySyncCancel : public HbgDispatchFailure,
                           public testing::WithParamInterface<bool> {};

TEST_P(HbgEarlySyncCancel, ReadyCohortReentersTheSyncQueueExactlyOnceOrReportsOverflow) {
    ChipTaskStorage task{}, first{}, second{};
    task.slot.active_mask = LaneMask::aic();
    task.slot.logical_block_num = 2;
    task.slot.ed_flags = ED_FLAG_CANDIDATE;
    task.slot.task_attrs.set_sync_start();
    task.task.task_id = TaskId::make_global(7);
    auto &ready = scheduler.ready_sync_queues[static_cast<int>(ResourceShape::AIC)];
    if (GetParam()) {
        ASSERT_TRUE(ready.push_sync_start(&first.slot, TaskId::make_global(1)));
        ASSERT_TRUE(ready.push_sync_start(&second.slot, TaskId::make_global(2)));
    }
    scheduler.enqueue_early_dispatch_candidate(task.slot);
    TaskId id;
    auto captured = scheduler.early_sync_start_queue.pop_sync_start(&id);
    ASSERT_EQ(captured.ptr(), &task.slot);
    ASSERT_TRUE(SchedulerState::try_claim_early_sync_drain(task.payload));
    ASSERT_TRUE(scheduler.push_ready_routed(&task.slot, 0));
    EXPECT_EQ(scheduler.cancel_early_sync_drain(task.slot, captured, id, 1), !GetParam());
    EXPECT_TRUE(scheduler.cancel_early_sync_drain(task.slot, captured, id, 1));
    EXPECT_EQ(ready.size(), GetParam() ? 2u : 1u);
    EXPECT_EQ(header.sched_error_code.load(), GetParam() ? SIMPLER_ERROR_READY_QUEUE_OVERFLOW : SIMPLER_ERROR_NONE);
    if (GetParam()) EXPECT_EQ(header.sched_error_thread.load(), 1);
    else EXPECT_EQ(ready.pop_sync_start(&id).ptr(), &task.slot);
}

INSTANTIATE_TEST_SUITE_P(ReadyQueueFull, HbgEarlySyncCancel, testing::Bool());

TEST_F(HbgDispatchFailure, FullDispatchRequeueElectsOneGlobalRetirementAndReleasesObserver) {
    attach_tasks(2);
    auto &queue = scheduler.ready_queues[static_cast<int>(ResourceShape::AIC)];
    ChipTaskStorage first{}, second{};
    second.slot.active_mask = {true, false, false};
    second.slot.logical_block_num = 2;  // The reporter owns one AIC, so it must requeue the remaining block.
    ASSERT_TRUE(queue.push(&first.slot));
    ASSERT_TRUE(queue.push(&second.slot));
    // Pause a competing dequeue after it has claimed cell 0 and before it
    // releases the sequence. The reporter can pop cell 1, but cannot refill cell 0.
    queue.dequeue_pos.store(1);
    std::promise<void> observer_left;
    peer_dispatch_exits = observer_left.get_future().share();
    std::thread observer([&] {
        while (!context->is_completed())
            std::this_thread::yield();
        EXPECT_EQ(context->run_resolution_thread(runtime.get(), 2), 0);
        observer_left.set_value();
    });
    EXPECT_EQ(context->resolve_and_dispatch(runtime.get(), 0), 0);
    observer.join();
    EXPECT_EQ(second.slot.next_block_idx.load(), 1);
    EXPECT_EQ(header.sched_error_code.load(), SIMPLER_ERROR_READY_QUEUE_OVERFLOW);
    EXPECT_EQ(header.sched_error_thread.load(), 0);
    EXPECT_EQ(header.sched_error_bitmap.load(), 1u);
    EXPECT_EQ(batches.load(), 1);
    EXPECT_EQ(deadlines.load(), 1);
    expect_once();
    expect_header_terminal(SIMPLER_ERROR_READY_QUEUE_OVERFLOW);
}

TEST_F(HbgDispatchFailure, RetirementWaitsForBothWritersOfAnUpperMaskCore) {
    auto &publications = scheduler.register_publications;
    const int core = 70;
    RegisterPublications::CoreMask upper{0, 1ULL << (core - 64)};
    RegisterPublications::CoreMask peer{1, 0};
    // Keep both logical writers inside publication while the retiree closes
    // their shared target. Completing either one alone cannot permit EXIT.
    EXPECT_TRUE(publications.publish(0, upper, [&] {
        EXPECT_TRUE(publications.publish(1, upper, [&] {
            EXPECT_TRUE(publications.close_and_check_writers(upper));
            EXPECT_FALSE(publications.idle(core));
        }));
        EXPECT_FALSE(publications.idle(core));
        EXPECT_FALSE(publications.publish(2, upper, [] {
            ADD_FAILURE() << "write to retiring core";
        }));
        EXPECT_TRUE(publications.publish(2, peer, [] {}));
    }));
    EXPECT_TRUE(publications.idle(core));
}

TEST_F(HbgDispatchFailure, PreparedTokenAndDoorbellCannotOverwriteExit) {
    attach_tasks(2);
    ExitProbe probe;
    exit_probe = &probe;
    auto first_sweep = probe.first_sweep.get_future();
    auto handle = SchedulerContextTestPeer::prepared_token(probe.data_main);
    auto shutdown = std::async(std::launch::async, [&] {
        SchedulerContextTestPeer::fail_requeue(*context, runtime.get());
    });
    ASSERT_EQ(first_sweep.wait_for(std::chrono::seconds(2)), std::future_status::ready);
    SchedulerContextTestPeer::publish_prepared(*context, handle);
    scheduler.early_dispatch_doorbell_table[1] = {2, 19};
    scheduler.ring_staged_doorbells({1ULL << 1}, 2);
    EXPECT_EQ(probe.data_main, AICORE_EXIT_SIGNAL);
    probe.resume_ack.set_value();
    shutdown.get();
    ASSERT_TRUE(probe.ack);
    ASSERT_EQ(runtime->get_teardown_gates()[1].post_close_release, AICORE_POST_CLOSE_RELEASE);
    SchedulerContextTestPeer::publish_prepared(*context, handle);
    EXPECT_FALSE(scheduler.ring_staged_doorbells({1ULL << 1}, 2));
    EXPECT_EQ(probe.data_main, AICORE_EXIT_SIGNAL);
    EXPECT_EQ(context->retirement_error(), 0);
    expect_once();
    expect_header_terminal(SIMPLER_ERROR_READY_QUEUE_OVERFLOW);
}

TEST_F(HbgDispatchFailure, RetirementFinishesWhileDoorbellWriterIsPaused) {
    attach_tasks(2);
    ExitProbe probe;
    exit_probe = &probe;
    probe.pause_doorbell = true;
    scheduler.early_dispatch_doorbell_table[1] = {2, 19};
    auto writer = std::async(std::launch::async, [&] {
        scheduler.ring_staged_doorbells({1ULL << 1}, 2);
    });
    ASSERT_EQ(probe.doorbell_entered.get_future().wait_for(std::chrono::seconds(2)), std::future_status::ready);
    auto shutdown = std::async(std::launch::async, [&] {
        SchedulerContextTestPeer::fail_requeue(*context, runtime.get());
    });
    EXPECT_EQ(probe.first_sweep.get_future().wait_for(std::chrono::seconds(2)), std::future_status::ready);
    EXPECT_EQ(deadlines.load(), 1);
    for (int i = 0; i < kCores; ++i)
        EXPECT_EQ(retired[i].load(), i == 1 ? 0 : 1);
    probe.resume_ack.set_value();
    // Five peers can release while the sixth writer is still parked.
    EXPECT_EQ(shutdown.wait_for(std::chrono::seconds(2)), std::future_status::ready);
    probe.resume_doorbell.set_value();
    writer.get();
    shutdown.get();
    EXPECT_FALSE(probe.ack);
    EXPECT_EQ(context->retirement_error(), -1);
    for (int i = 0; i < kCores; ++i) {
        EXPECT_EQ(runtime->get_teardown_gates()[i].post_close_release, i == 1 ? 0U : AICORE_POST_CLOSE_RELEASE);
        EXPECT_EQ(runtime->get_workers()[i].aicore_done != 0, i == 1);
    }
    expect_header_terminal(SIMPLER_ERROR_READY_QUEUE_OVERFLOW);
}

TEST_F(HbgDispatchFailure, LastArrivalRetainsMissingAckAlongsideQueueFailure) {
    attach_tasks(2);
    ExitProbe probe;
    exit_probe = &probe;
    silent_first_core = true;
    auto first_sweep = probe.first_sweep.get_future();
    auto arrive = [&] {
        executor->completion_gate_.arrive_and_finalize_if_last(2, [&] {
            executor->snapshot_run_terminal(runtime.get());
        });
    };
    auto reporter = std::async(std::launch::async, [&] {
        SchedulerContextTestPeer::fail_requeue(*context, runtime.get());
        arrive();
    });
    ASSERT_EQ(first_sweep.wait_for(std::chrono::seconds(2)), std::future_status::ready);
    EXPECT_EQ(context->shutdown(1, runtime.get()), 0);
    arrive();
    probe.resume_ack.set_value();
    reporter.get();
    executor->handoff_run_terminal();
    EXPECT_EQ(aicpu_finish_native_invocation(0, 0), 0);
    EXPECT_EQ(aicpu_finish_native_invocation(1, 0), 0);
    EXPECT_EQ(aicpu_finish_native_invocation(2, 0), -SIMPLER_ERROR_READY_QUEUE_OVERFLOW);
    const auto terminal = device_run_result_terminal(storage.result, epoch);
    EXPECT_EQ(terminal.source, DeviceRunCodeSource::Header);
    EXPECT_EQ(terminal.code, -SIMPLER_ERROR_READY_QUEUE_OVERFLOW);
    ASSERT_EQ(storage.result.payload_bytes, sizeof(int32_t));
    int32_t retirement_rc = 0;
    std::memcpy(&retirement_rc, storage.result.payload, sizeof(retirement_rc));
    EXPECT_EQ(retirement_rc, -1);
    EXPECT_EQ(batches.load(), 1);
    expect_once();
}

TEST_F(SchedulerRetirement, DeferredHandshakeRetirementFailureSurvivesSuccessfulPeerShutdown) {
    // Request retirement before any core is READY. The handshake publisher
    // owns the eventual return code, and the peer's shutdown claims no core.
    silent_first_core = true;
    EXPECT_EQ(request_all(), 0);
    handshake();
    EXPECT_EQ(context->shutdown(1, runtime.get()), 0);
    EXPECT_EQ(context->retirement_error(), -1);
    executor->snapshot_run_terminal(runtime.get());
    EXPECT_TRUE(executor->terminal_.has_failure());
    EXPECT_EQ(executor->terminal_.source(), DeviceRunCodeSource::ShutdownRc);
    ASSERT_EQ(initialize(*context, runtime.get(), reinterpret_cast<uint64_t>(regs.data()), 0), 0);
    EXPECT_EQ(context->retirement_error(), 0);
}

TEST_F(SchedulerRetirement, GraphPrepareOverflowStopsSchedulersAndPublishesHeaderError) {
    constexpr uint64_t epoch = 42;
    NativeInvocationResultStorage storage{};
    ASSERT_EQ(aicpu_begin_native_invocation(3, reinterpret_cast<uint64_t>(&storage), epoch), 0);
    ASSERT_EQ(initialize(*context, runtime.get(), reinterpret_cast<uint64_t>(regs.data()), 0), 0);
    for (int i = 0; i < kCores; ++i)
        runtime->get_workers()[i].report_epoch = epoch;
    SchedulerContextTestPeer::assign(*context);
    handshake();

    SharedMemoryHeader header{};
    runtime->set_gm_sm_ptr(&header);
    SchedulerState scheduler{};
    scheduler.sm_header = &header;
    RuntimeContext rt_context{};
    rt_context.scheduler = &scheduler;
    context->bind_runtime(&rt_context);
    context->on_graph_attached(&rt_context, 0, 3);
    ChipReadyQueueSlot slots[2]{};
    auto &queue = scheduler.graph_prepare_queue;
    seed_two_slots(queue, slots);
    // Each valid graph has its own execution storage. Clamp only the fixture's
    // queue capacity, as a correct Host bind sizes it for all three graphs.
    auto definition = make_test_definition(42, 0, 0, TestRoot::ORDINARY, TestEdDefect::NONE, TestBoundary::NONE);
    TestDefinitionObject image(definition);
    std::array<OuterHeap, 3> heaps{OuterHeap(definition), OuterHeap(definition), OuterHeap(definition)};
    std::array<ChipTaskStorage, 3> graphs{};
    std::array<std::atomic<ChipTaskState>, 3> states{};
    header.tasks.total_tasks = 3;
    header.tasks.task_storage = graphs.data();
    header.tasks.task_states = states.data();
    scheduler.task_view.tasks = &header.tasks;
    for (int i = 0; i < 3; ++i) {
        auto &graph = graphs[i];
        graph.slot.task_kind = TaskKind::GRAPH;
        graph.task.task_id = TaskId::make_global(i);
        graph.task.graph_definition_offset = image.image_offset();
        graph.task.packed_buffer_base = heaps[i].base();
        graph.task.packed_buffer_end = heaps[i].end();
        header.tasks.reset_task_state(i);
        context->classify_partition(i, 3, image.view());
        ASSERT_NE(graph.slot.graph_context, nullptr);
        if (i < 2) ASSERT_EQ(header.sched_error_code.load(), SIMPLER_ERROR_NONE);
    }
    ASSERT_EQ(queue.enqueue_pos.load(), 2u);
    ASSERT_EQ(header.sched_error_code.load(), SIMPLER_ERROR_READY_QUEUE_OVERFLOW);
    ASSERT_FALSE(context->is_completed());
    ASSERT_EQ(batches.load(), 0);

    // Release both dispatchers and the resolution thread on the same fatal header.
    std::array<int32_t, 3> counts{};
    std::atomic<int> arrived{0};
    std::atomic<int> returned{0};
    // Keep the EXIT owner inside the platform call until its two peers have left
    // their scheduler loops. A serialized second shutdown cannot pass this gate.
    std::promise<void> two_peers_returned;
    peer_dispatch_exits = two_peers_returned.get_future().share();
    std::array<std::thread, 3> workers;
    for (int i = 0; i < 3; ++i) {
        workers[i] = std::thread([&, i] {
            ++arrived;
            while (arrived.load() != 3)
                std::this_thread::yield();
            counts[i] = i == 2 ? context->run_resolution_thread(runtime.get(), i) :
                                 context->resolve_and_dispatch(runtime.get(), i);
            if (++returned == 2) two_peers_returned.set_value();
        });
    }
    for (auto &worker : workers)
        worker.join();
    EXPECT_TRUE(context->is_completed());
    EXPECT_EQ(queue.dequeue_pos.load(), 0u);
    EXPECT_EQ(header.sched_error_code.load(), SIMPLER_ERROR_READY_QUEUE_OVERFLOW);
    EXPECT_EQ(header.sched_error_thread.load(), 2);
    EXPECT_EQ(header.sched_error_bitmap.load(), 1u << 2);
    EXPECT_EQ(batches.load(), 1);
    EXPECT_EQ(deadlines.load(), 1);
    expect_once();

    // A thread can return a successful completion count even though the run failed.
    for (int32_t count : counts)
        EXPECT_EQ(count, 0);
    executor->snapshot_run_terminal(runtime.get());
    executor->handoff_run_terminal();
    EXPECT_EQ(aicpu_finish_native_invocation(0, counts[0]), 0);
    EXPECT_EQ(aicpu_finish_native_invocation(1, counts[1]), 0);
    EXPECT_EQ(aicpu_finish_native_invocation(2, counts[2]), -SIMPLER_ERROR_READY_QUEUE_OVERFLOW);
    auto terminal = device_run_result_terminal(storage.result, epoch);
    EXPECT_EQ(terminal.state, DeviceRunTerminalState::Failed);
    EXPECT_EQ(terminal.source, DeviceRunCodeSource::Header);
    EXPECT_EQ(terminal.code, -SIMPLER_ERROR_READY_QUEUE_OVERFLOW);
    EXPECT_EQ(device_run_result_terminal(storage.result, epoch + 1).state, DeviceRunTerminalState::Undecided);
}
#endif

TEST_F(SchedulerRetirement, RequestBeforeHandshakeIsServicedByPublisher) {
    EXPECT_EQ(request_all(), 0);
    EXPECT_EQ(batches.load(), 0);
    handshake();
    expect_once();
    EXPECT_EQ(batches.load(), 1);
    EXPECT_EQ(deadlines.load(), 1);
    EXPECT_EQ(request_all(), 0);
    expect_once();
}

TEST_F(SchedulerRetirement, RequestAfterHandshakeUsesOneBatchAndDeadline) {
    handshake();
    EXPECT_EQ(batches.load(), 0);
    EXPECT_EQ(request_all(), 0);
    expect_once();
    EXPECT_EQ(batches.load(), 1);
    EXPECT_EQ(deadlines.load(), 1);
}

TEST_F(SchedulerRetirement, RequestAfterHandshakeWaitsForInitialization) {
    SchedulerContextTestPeer::clear_ownership(*context);
    context->handshake_partition(runtime.get(), 0, 1);
    EXPECT_EQ(request_all(), 0);
    EXPECT_EQ(batches.load(), 0);
    for (int i = 0; i < kCores; ++i) {
        EXPECT_EQ(opened[i].load(), 1);
        EXPECT_EQ(retired[i].load(), 0);
        EXPECT_EQ(runtime->get_teardown_gates()[i].post_close_release, 0U);
    }
    EXPECT_EQ(context->post_handshake_init(runtime.get()), 0);
    expect_once();
    EXPECT_EQ(batches.load(), 1);
    EXPECT_EQ(request_all(), 0);
    expect_once();
}

TEST_F(SchedulerRetirement, OwnedHandshakeWaitsForAssignment) {
    SchedulerContextTestPeer::clear_ownership(*context);
    if (!handshake_owned_only(*context, runtime.get(), 0, 0)) {
        GTEST_SKIP() << "HBG has only the partition handshake";
    }
    EXPECT_EQ(request_all(), 0);
    EXPECT_EQ(batches.load(), 0);
    ASSERT_TRUE(assign_owned(*context, runtime.get(), 0, 0));
    const std::array<int, 3> owned = {0, 2, 3};
    for (int core : owned)
        EXPECT_EQ(retired[core].load(), 1);
    EXPECT_EQ(retired[1].load(), 0);
    EXPECT_EQ(retired[4].load(), 0);
    EXPECT_EQ(retired[5].load(), 0);
    EXPECT_EQ(batches.load(), 1);
    ASSERT_TRUE(handshake_owned(*context, runtime.get(), 1, 0));
    expect_once();
    EXPECT_EQ(request_all(), 0);
    expect_once();
}

TEST_F(SchedulerRetirement, AssignmentFailurePublishesOpenedWindows) {
    SchedulerContextTestPeer::clear_ownership(*context);
    context->handshake_partition(runtime.get(), 0, 1);
    SchedulerContextTestPeer::force_assignment_failure(*context);
    EXPECT_EQ(request_all(), 0);
    EXPECT_EQ(batches.load(), 0);
    EXPECT_EQ(context->post_handshake_init(runtime.get()), -1);
    expect_once();
    EXPECT_EQ(batches.load(), 1);
    EXPECT_EQ(request_all(), 0);
    expect_once();
}

TEST_F(SchedulerRetirement, EmergencyBeforeHandshakeIsServicedByPublisher) {
    emergency();
    EXPECT_EQ(batches.load(), 0);
    handshake();
    expect_once();
    emergency();
    EXPECT_EQ(request_all(), 0);
    expect_once();
    EXPECT_EQ(batches.load(), 1);
}

TEST_F(SchedulerRetirement, RequestWhileWindowsOpenBeforeStatePublicationIsRetained) {
    pause_open = true;
    std::thread publisher([&] {
        handshake();
    });
    bool reached;
    {
        std::unique_lock<std::mutex> lock(init_mutex);
        reached = init_cv.wait_for(lock, std::chrono::seconds(5), [] {
            return open_paused;
        });
    }
    EXPECT_TRUE(reached);
    if (reached) {
        EXPECT_EQ(request_all(), 0);
        EXPECT_EQ(batches.load(), 0);
    }
    {
        std::lock_guard<std::mutex> lock(init_mutex);
        resume_open = true;
    }
    init_cv.notify_all();
    publisher.join();
    expect_once();
}

TEST_F(SchedulerRetirement, OwnedClusterPublishersServiceEarlyRequest) {
    EXPECT_EQ(request_all(), 0);
    if (!handshake_owned(*context, runtime.get(), 0, 0)) {
        GTEST_SKIP() << "HBG has only the partition handshake";
    }
    ASSERT_TRUE(handshake_owned(*context, runtime.get(), 1, 0));
    expect_once();
    EXPECT_EQ(batches.load(), 2);
    EXPECT_EQ(deadlines.load(), 2);
}

TEST_F(SchedulerRetirement, OverlappingSetsAndDuplicateIdsRetireOnlyOnce) {
    handshake();
    const int32_t subset[] = {-1, 0, 3, 0, kCores};
    EXPECT_EQ(SchedulerContextTestPeer::request(*context, runtime.get(), subset, 5), 0);
    EXPECT_EQ(request_all(), 0);
    expect_once();
    EXPECT_EQ(batches.load(), 2);
}

TEST_F(SchedulerRetirement, NormalAndEmergencyRetireReadyCoresOnce) {
    handshake();
    std::thread normal([&] {
        context->shutdown(0, runtime.get());
    });
    std::thread fatal([&] {
        emergency();
    });
    context->shutdown(1, runtime.get());
    normal.join();
    fatal.join();
    expect_once();
}

TEST_F(SchedulerRetirement, PublicationAndEmergencyCanRace) {
    std::thread publisher([&] {
        handshake();
    });
    std::thread fatal([&] {
        emergency();
    });
    publisher.join();
    fatal.join();
    expect_once();
}

TEST_F(SchedulerRetirement, PartialTimeoutDoesNotReleaseSilentCoreOrRetryClaim) {
    handshake();
    silent_first_core = true;
    EXPECT_EQ(request_all(), -1);
    expect_once();
    EXPECT_EQ(batches.load(), 1);
    EXPECT_EQ(deadlines.load(), 1);
    EXPECT_EQ(request_all(), 0);
    expect_once();
}

TEST_F(SchedulerRetirement, EarlyRequestPreservesPartialTimeout) {
    silent_first_core = true;
    EXPECT_EQ(request_all(), 0);
    handshake();
    expect_once();
    EXPECT_EQ(batches.load(), 1);
    EXPECT_EQ(request_all(), 0);
    expect_once();
}

#ifdef PTO_ASYNC_WAIT_H
TEST_F(SchedulerRetirement, ShutdownPreservesEmergencyRetirementFailure) {
    handshake();
    silent_first_core = true;
    emergency();
    EXPECT_EQ(context->shutdown(0, runtime.get()), -1);
    EXPECT_EQ(context->shutdown(1, runtime.get()), -1);
    expect_once();
    EXPECT_EQ(batches.load(), 1);
}

TEST_F(SchedulerRetirement, ShutdownPreservesPublisherRetirementFailure) {
    silent_first_core = true;
    EXPECT_EQ(request_all(), 0);
    handshake();
    EXPECT_EQ(context->shutdown(0, runtime.get()), -1);
    EXPECT_EQ(context->shutdown(1, runtime.get()), -1);
    expect_once();
    EXPECT_EQ(batches.load(), 1);
}

TEST_F(SchedulerRetirement, NewRunClearsRetirementFailure) {
    handshake();
    silent_first_core = true;
    emergency();
    EXPECT_EQ(context->retirement_error(), -1);
    ASSERT_EQ(initialize(*context, runtime.get(), reinterpret_cast<uint64_t>(regs.data()), 0), 0);
    EXPECT_EQ(context->retirement_error(), 0);
}

TEST_F(SchedulerRetirement, CancelledDrainCoversAllOwnersWithoutCompletingTasks) {
    handshake();
    ASSERT_EQ(request_all(), 0);
    auto rt = std::make_unique<RuntimeContext>();
    auto mailbox = std::make_unique<AICoreCompletionMailbox>();
    rt->aicore_mailbox = mailbox.get();
    SchedulerContextTestPeer::bind_async(*context, *rt);
    const TaskId task = TaskId::make(0, 19);
    uint32_t counter = 3;
    uint64_t sdma_record = 9;
    auto &waits = rt->scheduler.async_wait_list;
    waits.count.store(1);
    waits.entries[0].task_token = task;
    waits.entries[0].normal_done = true;
    // Any accidental success callback would dereference this invalid slot.
    waits.entries[0].slot_state = reinterpret_cast<ChipTaskSlotState *>(1);
    int32_t error = 0;
    ASSERT_TRUE(waits.append_condition_locked(
        waits.entries[0], reinterpret_cast<uint64_t>(&counter), 0, 3, ASYNC_ENGINE_SDMA, COMPLETION_TYPE_COUNTER, error
    ));
    ASSERT_TRUE(mailbox->try_push_condition(
        task, reinterpret_cast<uint64_t>(&sdma_record), 9, 0, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_SDMA_EVENT_RECORD
    ));
    ASSERT_TRUE(mailbox->try_push_normal_done(task, 1));
    auto &slab = SchedulerContextTestPeer::slab(*context, task);
    slab.count = 1;
    slab.entries[0] = {reinterpret_cast<uint64_t>(&counter), 0, 3, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_COUNTER, 0};
    KernelErrorCleanup result{};
    context->drain_cancelled_run(UINT64_MAX, result);
    EXPECT_EQ(result.complete, 1u);
    EXPECT_EQ(result.released_cores, kCores);
    EXPECT_EQ(result.drained_conditions, 3u);
    EXPECT_TRUE(waits.entries[0].conditions[0].retired);
    EXPECT_EQ(waits.count.load(), 1);
    EXPECT_EQ(sdma_record, 9u);
}

TEST_F(SchedulerRetirement, CancelledDrainRejectsPendingSlabAndPreservesIdentity) {
    handshake();
    ASSERT_EQ(request_all(), 0);
    auto rt = std::make_unique<RuntimeContext>();
    SchedulerContextTestPeer::bind_async(*context, *rt);
    const TaskId task = TaskId::make(1, 20);
    uint32_t counter = 0;
    auto &slab = SchedulerContextTestPeer::slab(*context, task);
    slab.count = 1;
    slab.entries[0] = {reinterpret_cast<uint64_t>(&counter), 0, 1, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_COUNTER, 0};
    KernelErrorCleanup result{};
    context->drain_cancelled_run(0, result);
    EXPECT_EQ(result.complete, 0u);
    EXPECT_EQ(result.async_error, SIMPLER_ERROR_SCHEDULER_TIMEOUT);
    EXPECT_EQ(result.source, 4u);
    EXPECT_EQ(result.task_token, TaskId::to_uint64(task));
    EXPECT_EQ(slab.count, 1u);
    counter = 1;
    context->drain_cancelled_run(UINT64_MAX, result);
    EXPECT_EQ(result.complete, 1u);
}

TEST_F(SchedulerRetirement, CancelledDrainRequiresEveryCoreRelease) {
    handshake();
    silent_first_core = true;
    emergency();
    KernelErrorCleanup result{};
    context->drain_cancelled_run(UINT64_MAX, result);
    EXPECT_EQ(result.complete, 0u);
    EXPECT_EQ(result.core_error, -1);
    EXPECT_EQ(result.source, 1u);
    EXPECT_EQ(result.core, 0);
    EXPECT_EQ(result.released_cores, kCores - 1);
}
#endif

TEST_F(SchedulerRetirement, InvalidPhysicalCoreIsNotPublishedOrRetired) {
    runtime->get_workers()[0].physical_core_id = kCores;
    EXPECT_EQ(request_all(), 0);
    handshake();
    EXPECT_TRUE(SchedulerContextTestPeer::handshake_failed(*context));
    EXPECT_EQ(retired[0].load(), 0);
    EXPECT_EQ(runtime->get_teardown_gates()[0].post_close_release, 0U);
    for (int i = 1; i < kCores; ++i)
        EXPECT_EQ(retired[i].load(), 1);
    EXPECT_EQ(request_all(), 0);
    EXPECT_EQ(retired[0].load(), 0);
}

TEST_F(SchedulerRetirement, UnmappedWindowIsNotOpenedOrPublished) {
    regs[0] = 0;
    EXPECT_EQ(request_all(), 0);
    handshake();
    EXPECT_TRUE(SchedulerContextTestPeer::handshake_failed(*context));
    EXPECT_EQ(opened[0].load(), 0);
    EXPECT_EQ(retired[0].load(), 0);
    EXPECT_EQ(runtime->get_teardown_gates()[0].post_close_release, 0U);
    for (int i = 1; i < kCores; ++i)
        EXPECT_EQ(retired[i].load(), 1);
    EXPECT_EQ(request_all(), 0);
    EXPECT_EQ(retired[0].load(), 0);
}

TEST_F(SchedulerRetirement, OwnedClusterPublishersExcludeUnmappedWindow) {
    regs[0] = 0;
    EXPECT_EQ(request_all(), 0);
    if (!handshake_owned(*context, runtime.get(), 0, 0)) {
        GTEST_SKIP() << "HBG has only the partition handshake";
    }
    ASSERT_TRUE(handshake_owned(*context, runtime.get(), 1, 0));
    EXPECT_TRUE(SchedulerContextTestPeer::handshake_failed(*context));
    EXPECT_EQ(opened[0].load(), 0);
    EXPECT_EQ(retired[0].load(), 0);
    EXPECT_EQ(runtime->get_teardown_gates()[0].post_close_release, 0U);
    for (int i = 1; i < kCores; ++i)
        EXPECT_EQ(retired[i].load(), 1);
    EXPECT_EQ(batches.load(), 2);
    EXPECT_EQ(request_all(), 0);
}

TEST_F(SchedulerRetirement, PublishedZeroAddressDoesNotRetireOrReleaseTarget) {
    handshake();
    // Fault injection after READY exercises the retirement boundary guard.
    SchedulerContextTestPeer::clear_register_address(*context, 3);
    silent_first_core = true;
    EXPECT_EQ(request_all(), -1);
    for (int i = 0; i < kCores; ++i) {
        EXPECT_EQ(retired[i].load(), i == 3 ? 0 : 1);
        EXPECT_EQ(
            runtime->get_teardown_gates()[i].post_close_release, i == 0 || i == 3 ? 0U : AICORE_POST_CLOSE_RELEASE
        );
    }
    EXPECT_EQ(batches.load(), 1);
    EXPECT_EQ(deadlines.load(), 1);
    EXPECT_EQ(request_all(), 0);
    EXPECT_EQ(batches.load(), 1);
}

TEST_F(SchedulerRetirement, EmptyValidatedBatchDoesNotAllocateDeadline) {
    handshake();
    for (int i = 0; i < kCores; ++i)
        SchedulerContextTestPeer::clear_register_address(*context, i);
    EXPECT_EQ(request_all(), 0);
    EXPECT_EQ(batches.load(), 0);
    EXPECT_EQ(deadlines.load(), 0);
    for (int i = 0; i < kCores; ++i) {
        EXPECT_EQ(retired[i].load(), 0);
        EXPECT_EQ(runtime->get_teardown_gates()[i].post_close_release, 0U);
    }
}

TEST_F(SchedulerRetirement, NewGenerationClearsPendingRequestsAndReturnGates) {
    EXPECT_EQ(request_all(), 0);
    for (int i = 0; i < kCores; ++i)
        runtime->get_teardown_gates()[i].post_close_release = AICORE_POST_CLOSE_RELEASE;
    ASSERT_EQ(initialize(*context, runtime.get(), reinterpret_cast<uint64_t>(regs.data()), 0), 0);
    handshake();
    EXPECT_EQ(batches.load(), 0);
    for (int i = 0; i < kCores; ++i)
        EXPECT_EQ(runtime->get_teardown_gates()[i].post_close_release, 0U);
    EXPECT_EQ(request_all(), 0);
    expect_once();
}

TEST_F(SchedulerRetirement, InitializationClearsRequestAndReleaseForEveryCore) {
    for (int i = 0; i < kCores; ++i) {
        auto &control = runtime->get_teardown_gates()[i];
        control.post_close_release = AICORE_POST_CLOSE_RELEASE;
        control.exit_request = 9;
    }
    ASSERT_EQ(initialize(*context, runtime.get(), reinterpret_cast<uint64_t>(regs.data()), 0), 0);
    for (int i = 0; i < kCores; ++i) {
        const auto &control = runtime->get_teardown_gates()[i];
        EXPECT_EQ(control.post_close_release, 0u);
        EXPECT_EQ(control.exit_request, 0u);
    }
}

TEST_F(SchedulerRetirement, NativeModeClaimsOnlyAfterCurrentReportAndUsesCondNonce) {
    set_test_epoch(42);
    ASSERT_EQ(initialize(*context, runtime.get(), reinterpret_cast<uint64_t>(regs.data()), 0), 0);
    for (int i = 0; i < kCores; ++i) {
        EXPECT_EQ(runtime->get_teardown_gates()[i].exit_request, 0U);
        EXPECT_EQ(runtime->get_teardown_gates()[i].post_close_release, 0U);
        runtime->get_workers()[i].report_epoch = 42;
    }
    SchedulerContextTestPeer::assign(*context);
    EXPECT_EQ(request_all(), 0);
    EXPECT_EQ(nonce_batches.load(), 0);
    handshake();
    expect_once();
    EXPECT_EQ(nonce_batches.load(), 1);
    set_test_epoch(0);
}
TEST_F(SchedulerRetirement, UnstampedModeUsesNonceAfterOwnershipPublication) {
    handshake();
    EXPECT_EQ(request_all(), 0);
    expect_once();
    EXPECT_EQ(nonce_batches.load(), 1);
}

TEST_F(SchedulerRetirement, ReportsAreInvalidatedOnlyAfterTheirOwnExitAcknowledgement) {
    handshake();
    for (int i = 0; i < kCores; ++i) {
        runtime->get_workers()[i].aicpu_ready = 2;
        runtime->get_workers()[i].report_epoch = 42;
    }
    silent_first_core = true;
    EXPECT_NE(request_all(), 0);
    for (int i = 0; i < kCores; ++i) {
        const auto &report = runtime->get_workers()[i];
        EXPECT_EQ(report.aicore_done, i == 0 ? 1u : 0u);
        EXPECT_EQ(report.aicpu_ready, i == 0 ? 2u : 0u);
        EXPECT_EQ(report.report_epoch, 42u);
        EXPECT_EQ(aicore_report_accepted(&report, 42), i == 0);
    }
    const int previous_batches = batches.load();
    emergency();
    EXPECT_EQ(batches.load(), previous_batches);
    EXPECT_EQ(runtime->get_workers()[0].aicore_done, 1u);
}

TEST_F(SchedulerRetirement, MissingReportDoesNotTurnAcceptedRetirementIntoExitProof) {
    EXPECT_EXIT(
        {
            SchedulerContextTestPeer::clear_ownership(*context);
            runtime->get_workers()[kCores - 1].aicore_done = 0;
            context->handshake_partition(runtime.get(), 0, 2);
            for (int i = 0; i < kCores / 2; ++i)
                if (opened[i].load() != 1) std::_Exit(1);

            std::atomic<bool> entered{false};
            std::atomic<bool> returned{false};
            std::thread missing_report([&] {
                entered.store(true, std::memory_order_release);
                context->handshake_partition(runtime.get(), 1, 2);
                returned.store(true, std::memory_order_release);
            });
            const auto startup_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
            while (!entered.load(std::memory_order_acquire)) {
                if (std::chrono::steady_clock::now() >= startup_deadline) std::_Exit(2);
            }
            if (request_all() != 0) std::_Exit(3);
            const auto observation_deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
            while (std::chrono::steady_clock::now() < observation_deadline) {
                if (returned.load(std::memory_order_acquire)) std::_Exit(4);
            }
            // READY is published only after the handshake join. An accepted
            // request is still pending, even for the already-open first slice.
            if (batches.load() != 0 || deadlines.load() != 0) std::_Exit(5);
            for (int i = 0; i < kCores; ++i) {
                if (retired[i].load() != 0 || runtime->get_teardown_gates()[i].post_close_release != 0) std::_Exit(6);
            }
            if (opened[kCores - 1].load() != 0) std::_Exit(7);
            // Only the fixture process ends here; no physical-exit fact is minted.
            std::_Exit(0);
        },
        testing::ExitedWithCode(0), ""
    );
}
