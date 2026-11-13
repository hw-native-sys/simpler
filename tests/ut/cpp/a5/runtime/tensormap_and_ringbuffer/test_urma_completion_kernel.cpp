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

#define __CPU_SIM 1
#define PTO_URMA_SUPPORTED 1

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace pto::comm {
enum class DmaEngine : uint8_t;
enum class AsyncSubmitMode : uint8_t { IMMEDIATE, DEFER, DEFER_AND_SUBMIT };
}  // namespace pto::comm

namespace {
uint32_t total_wait_calls = 0;
struct TestSession {
    bool valid{true};
    pto::comm::DmaEngine engine{static_cast<pto::comm::DmaEngine>(1)};
    uint8_t *contextGm{};
    uint32_t destRankId{1};
    uint32_t qpIdxBase{2};
    pto::comm::AsyncSubmitMode submitMode{pto::comm::AsyncSubmitMode::IMMEDIATE};
    uint32_t batchSize{UINT32_MAX};
    mutable uint32_t batchLogicalWqes{};
    mutable uint32_t batchStagedOps{};
    mutable std::array<uint32_t, 8> batchStagedWqes{};
    struct {
        struct {
            uint32_t qpCount{2};
        } urma;
    } cfg;
};

struct TestEvent {
    uint64_t handle{(uint64_t{1} << 32) | 4};
    pto::comm::DmaEngine engine{static_cast<pto::comm::DmaEngine>(1)};
    uint32_t urmaTargetCqe{};
    std::array<uint32_t, 8> urmaTargetBbPerJetty{4, 9};
    std::array<uint32_t, 8> urmaTargetCqePerJetty{3, 7};
    uint16_t urmaJettyBase{2};
    uint16_t urmaJettyCount{2};
    bool postFailed{};
    mutable uint32_t wait_calls{};

    bool Wait(const TestSession &) const {
        ++wait_calls;
        ++total_wait_calls;
        return false;
    }
};

}  // namespace

// Host-only doubles replace the ISA session and posting boundary. Adapter
// validation, slicing, posting and completion registration remain real.
namespace pto {
inline constexpr int64_t DYNAMIC = -1;

template <int64_t... Dims>
struct Shape {
    std::array<int64_t, 5> values;
    Shape(int64_t a, int64_t b, int64_t c, int64_t d, int64_t e) :
        values{a, b, c, d, e} {}
};

template <int64_t... Dims>
using Stride = Shape<Dims...>;

template <typename T>
struct TestAddress {
    uint64_t bytes;
    explicit operator const void *() const { return reinterpret_cast<const void *>(bytes); }
    TestAddress operator+(uint64_t elements) const { return {bytes + elements * sizeof(T)}; }
};

template <typename T, typename ShapeT, typename StrideT>
struct GlobalTensor {
    using RawDType = T;
    TestAddress<T> address;
    ShapeT shape;
    StrideT stride;
    GlobalTensor(TestAddress<T> ptr, ShapeT shp, StrideT steps) :
        address(ptr),
        shape(shp),
        stride(steps) {}
    TestAddress<T> data() const { return address; }
    int64_t GetShape(int dim) const { return shape.values[dim]; }
    int64_t GetStride(int dim) const { return stride.values[dim]; }
};

namespace comm {
using AsyncSession = TestSession;
using AsyncEvent = TestEvent;

template <DmaEngine Engine>
bool BuildAsyncSession(uint8_t *, uint32_t, AsyncSession &) = delete;
namespace detail {
template <DmaEngine Engine>
void ActivateEngineState(AsyncSession &session);
}  // namespace detail
template <DmaEngine Engine, typename Dst, typename Src>
AsyncEvent TGET_ASYNC(Dst dst, Src src, const AsyncSession &session);
template <DmaEngine Engine, typename Dst, typename Src>
AsyncEvent TPUT_ASYNC(Dst dst, Src src, const AsyncSession &session);
namespace urma {
inline uint64_t UrmaPeerMrBaseAddr(uint8_t *, uint32_t) { return 0; }
}  // namespace urma
}  // namespace comm
}  // namespace pto

// The adapter must not depend on the implicit CCE block identity.
uint32_t get_block_idx() = delete;

#include "urma/urma_request.h"
#include "backend/urma/urma_completion_kernel.h"
#include "runtime_core.h"
#include "scheduler/scheduler_context.h"
#include <pto/comm/async/urma/urma_types.hpp>

namespace {
struct Submission {
    bool put;
    uint64_t dst;
    uint64_t src;
    std::array<int64_t, 5> shape;
    std::array<int64_t, 5> stride;
    pto::comm::AsyncSubmitMode mode;
};
struct PostScript {
    std::vector<TestEvent> events;
    std::vector<Submission> submissions;
    uint32_t activation_calls{};
    bool post_from_queue{};
    uint32_t fail_batch_post{};
    bool invalid_flush_event{};
    const volatile uint32_t *batch_completion_count{};
    std::vector<uint32_t> completion_counts_before_post;
    std::vector<uint32_t> doorbells;
};
PostScript *active_script = nullptr;
}  // namespace

namespace pto::comm {
namespace detail {
template <DmaEngine Engine>
void ActivateEngineState(AsyncSession &session) {
    static_assert(Engine == DmaEngine::URMA);
    ++active_script->activation_calls;
    session.engine = Engine;
    session.cfg.urma.qpCount = 0;
}
}  // namespace detail
AsyncEvent post_batch_for_test(const AsyncSession &session, uint32_t wqes) {
    auto &script = *active_script;
    script.completion_counts_before_post.push_back(static_cast<uint32_t>(*script.batch_completion_count));
    const uint32_t jetties = session.cfg.urma.qpCount;
    for (uint32_t n = 0; n < wqes; ++n) {
        ++session.batchStagedWqes[session.batchLogicalWqes++ % jetties];
    }
    const bool flush =
        ++session.batchStagedOps == session.batchSize || session.submitMode == AsyncSubmitMode::DEFER_AND_SUBMIT;
    const bool fail = script.submissions.size() == script.fail_batch_post;
    auto *info = reinterpret_cast<urma::UrmaInfo *>(session.contextGm);
    auto *queues = reinterpret_cast<urma::UrmaWQCtx *>(info->sqPtr);
    TestEvent event{};
    event.urmaJettyBase = session.qpIdxBase;
    event.urmaJettyCount = jetties;
    for (uint32_t j = 0; j < jetties; ++j) {
        uint32_t row = 0;
        EXPECT_TRUE(simpler::urma::queue_row_for(session.contextGm, session.destRankId, session.qpIdxBase + j, row));
        auto &queue = queues[row];
        auto *head = reinterpret_cast<uint32_t *>(queue.headAddr);
        const uint32_t staged = session.batchStagedWqes[j];
        if (flush && staged != 0 && !fail) {
            *head += staged;
            queue.submittedWqeCount += staged;
            *reinterpret_cast<uint32_t *>(queue.dbAddr) = *head;
            script.doorbells.push_back(row);
        }
        event.urmaTargetBbPerJetty[j] = *head + (flush ? 0 : staged);
        event.urmaTargetCqePerJetty[j] = queue.submittedWqeCount + (flush ? 0 : staged);
    }
    if (flush) {
        session.batchStagedWqes = {};
        session.batchStagedOps = 0;
    }
    if (flush && script.invalid_flush_event) event.urmaJettyCount = 0;
    event.handle = fail ? 0 : (static_cast<uint64_t>(session.destRankId) << 32) | event.urmaTargetBbPerJetty[0];
    event.postFailed = fail;
    return event;
}

template <typename Dst, typename Src>
AsyncEvent post_for_test(bool put, Dst dst, Src src, const AsyncSession &session) {
    auto &script = *active_script;
    script.submissions.push_back(
        {put, dst.data().bytes, src.data().bytes, dst.shape.values, dst.stride.values, session.submitMode}
    );
    EXPECT_EQ(src.shape.values, dst.shape.values);
    EXPECT_EQ(src.stride.values, dst.stride.values);
    if (session.submitMode != AsyncSubmitMode::IMMEDIATE) {
        EXPECT_TRUE(put);
        const uint64_t bytes = static_cast<uint64_t>(src.shape.values[4]) * sizeof(typename Src::RawDType);
        return post_batch_for_test(session, simpler::urma::chunk_count(bytes));
    }
    if (script.post_from_queue) {
        auto *info = reinterpret_cast<urma::UrmaInfo *>(session.contextGm);
        auto *queues = reinterpret_cast<urma::UrmaWQCtx *>(info->sqPtr);
        TestEvent event{};
        event.urmaJettyBase = session.qpIdxBase;
        event.urmaJettyCount = session.cfg.urma.qpCount;
        const uint64_t bytes = static_cast<uint64_t>(src.shape.values[4]) * sizeof(typename Src::RawDType);
        const uint64_t wqes = simpler::urma::chunk_count(bytes);
        for (uint32_t j = 0; j < event.urmaJettyCount; ++j) {
            uint32_t row = 0;
            EXPECT_TRUE(
                simpler::urma::queue_row_for(session.contextGm, session.destRankId, session.qpIdxBase + j, row)
            );
            auto &queue = queues[row];
            auto *head = reinterpret_cast<uint32_t *>(queue.headAddr);
            // Non-rail ISA posts rotate within one call and restart at jetty zero on the next call.
            const uint32_t posted = wqes / event.urmaJettyCount + (j < wqes % event.urmaJettyCount ? 1 : 0);
            if (posted != 0) {
                *head += posted;
                queue.submittedWqeCount += posted;
                *reinterpret_cast<uint32_t *>(queue.dbAddr) = *head;
            }
            event.urmaTargetBbPerJetty[j] = *head;
            event.urmaTargetCqePerJetty[j] = queue.submittedWqeCount;
        }
        event.handle = (static_cast<uint64_t>(session.destRankId) << 32) | event.urmaTargetBbPerJetty[0];
        return event;
    }
    const auto index = script.submissions.size() - 1;
    EXPECT_LT(index, script.events.size());
    return script.events.at(index);
}
template <DmaEngine Engine, typename Dst, typename Src>
AsyncEvent TGET_ASYNC(Dst dst, Src src, const AsyncSession &session) {
    static_assert(Engine == DmaEngine::URMA);
    return post_for_test(false, dst, src, session);
}
template <DmaEngine Engine, typename Dst, typename Src>
AsyncEvent TPUT_ASYNC(Dst dst, Src src, const AsyncSession &session) {
    static_assert(Engine == DmaEngine::URMA);
    return post_for_test(true, dst, src, session);
}
}  // namespace pto::comm

class SchedulerRetirementTestPeer {
public:
    static bool finish_failed_submission(
        SchedulerContext &context, SchedulerState &scheduler, const DeferredCompletionSlab &slab,
        ChipTaskSlotState &producer, int32_t &completed, ChipTaskSlotState **release, int32_t &release_count,
        RuntimeContext *runtime = nullptr, bool aborting = false
    ) {
        context.sched_ = &scheduler;
        context.rt_ = runtime;
        context.deferred_slab_per_core_[0][0] = slab;
        context.deferred_slabs_initialized_[0] = true;
        context.cores_total_num_ = 1;
        context.completed_.store(aborting);
        return context.complete_slot_task(
            producer, 0, SubtaskSlot::AIV0, 0, 0, nullptr, completed, release, release_count
#if SIMPLER_DFX
            ,
            0, 0
#endif
        );
    }

    static uint32_t preserved_count(const SchedulerContext &context) {
        return context.deferred_slab_per_core_[0][0].count;
    }
};

namespace {
struct RegistrationFixture {
    pto::comm::urma::UrmaInfo info{};
    std::array<pto::comm::urma::UrmaWQCtx, 8> sq{};
    std::array<pto::comm::urma::UrmaCqCtx, 8> cq{};
    std::array<uint32_t, 8> sq_head{};
    std::array<uint32_t, 8> sq_tail{};
    std::array<uint32_t, 8> sq_doorbell{};
    std::array<uint32_t, 8> cq_tail{};
    std::array<uint32_t, 8> cq_doorbell{};
    std::array<std::array<uint8_t, 16 * 64>, 8> sq_storage{};
    alignas(64) std::array<std::array<uint8_t, 16 * 64>, 8> cq_storage{};
    DeferredCompletionSlab slab{};
    AsyncCtx ctx{AsyncCtx::make(TaskId::make(0, 7), &slab)};
    simpler::urma::TmrRequestAdapter runtime{ctx};
    TestSession session{};
    TestEvent event{};

    RegistrationFixture() {
        ctx.urma_sender_id = 1;
        info.rowsPerPeer = 4;
        info.rankCount = 2;
        info.layout = pto::comm::urma::UrmaLayout::PER_PEER;
        info.jettyCount = 8;
        info.jettiesPerCore = 2;
        info.sqPtr = reinterpret_cast<uint64_t>(sq.data());
        info.scqPtr = reinterpret_cast<uint64_t>(cq.data());
        for (uint32_t row = 0; row < sq.size(); ++row) {
            sq[row].bufAddr = reinterpret_cast<uint64_t>(sq_storage[row].data());
            sq[row].wqeShiftSize = 6;
            sq[row].depth = 16;
            sq[row].headAddr = reinterpret_cast<uint64_t>(&sq_head[row]);
            sq[row].tailAddr = reinterpret_cast<uint64_t>(&sq_tail[row]);
            sq[row].dbAddr = reinterpret_cast<uint64_t>(&sq_doorbell[row]);
            cq[row].bufAddr = reinterpret_cast<uint64_t>(cq_storage[row].data());
            cq[row].cqeShiftSize = 6;
            cq[row].depth = 16;
            cq[row].tailAddr = reinterpret_cast<uint64_t>(&cq_tail[row]);
            cq[row].dbAddr = reinterpret_cast<uint64_t>(&cq_doorbell[row]);
        }
        session.contextGm = workspace();
    }

    uint8_t *workspace() { return reinterpret_cast<uint8_t *>(&info); }

    bool register_event() { return simpler::urma::register_urma_async_event(runtime, event, session, workspace()); }
};

}  // namespace

TEST(A5UrmaCompletionKernel, RegistersEveryActualJettyAndIndependentTarget) {
    RegistrationFixture fixture;

    ASSERT_TRUE(fixture.register_event());
    ASSERT_EQ(fixture.slab.count, 2u);
    EXPECT_EQ(fixture.slab.entries[0].addr, ((static_cast<uint64_t>(6) << 32) | 4));
    EXPECT_EQ(fixture.slab.entries[1].addr, ((static_cast<uint64_t>(7) << 32) | 9));
    EXPECT_EQ(fixture.slab.entries[0].expected_value, 3u);
    EXPECT_EQ(fixture.slab.entries[1].expected_value, 7u);
    EXPECT_EQ(fixture.slab.entries[0].backend_cookie, reinterpret_cast<uint64_t>(fixture.workspace()));
    EXPECT_EQ(fixture.slab.entries[0].completion_type, COMPLETION_TYPE_URMA_EVENT_HANDLE);
    EXPECT_EQ(fixture.slab.error_code, SIMPLER_ERROR_NONE);
    EXPECT_EQ(fixture.event.wait_calls, 0u);
}

TEST(A5UrmaCompletionKernel, ZeroHandleWithRealTargetsIsRegistered) {
    RegistrationFixture fixture;
    fixture.session.destRankId = 0;
    fixture.session.qpIdxBase = 0;
    fixture.session.cfg.urma.qpCount = 1;
    fixture.info.jettiesPerCore = 1;
    fixture.event.handle = 0;
    fixture.event.urmaJettyBase = 0;
    fixture.event.urmaJettyCount = 1;
    fixture.event.urmaTargetBbPerJetty[0] = 0;
    fixture.event.urmaTargetCqePerJetty[0] = 0;

    ASSERT_TRUE(fixture.register_event());
    ASSERT_EQ(fixture.slab.count, 1u);
    EXPECT_EQ(fixture.slab.entries[0].addr, 0u);
    EXPECT_EQ(fixture.slab.entries[0].expected_value, 0u);
    EXPECT_EQ(fixture.event.wait_calls, 0u);
}

TEST(A5UrmaCompletionKernel, NoCapacityForWholeEventRegistersNothing) {
    RegistrationFixture fixture;
    fixture.ctx.completion_capacity = 1;

    EXPECT_FALSE(fixture.register_event());
    EXPECT_EQ(fixture.slab.count, 0u);
    EXPECT_EQ(fixture.slab.error_code, SIMPLER_ERROR_ASYNC_REGISTRATION_FAILED);
    EXPECT_EQ(fixture.event.wait_calls, 0u);
}

TEST(A5UrmaCompletionKernel, ExistingConditionsSurviveCapacityFailure) {
    RegistrationFixture fixture;
    fixture.ctx.completion_capacity = 2;
    fixture.slab.count = 1;
    fixture.slab.entries[0].addr = 0x1234;

    EXPECT_FALSE(fixture.register_event());
    EXPECT_EQ(fixture.slab.count, 1u);
    EXPECT_EQ(fixture.slab.entries[0].addr, 0x1234u);
    EXPECT_EQ(fixture.event.wait_calls, 0u);
}

TEST(A5UrmaCompletionKernel, FailedPostWithNoSubmittedTargetsFails) {
    RegistrationFixture fixture;
    fixture.event.postFailed = true;
    fixture.event.handle = 0;
    fixture.event.urmaJettyCount = 0;

    EXPECT_FALSE(fixture.register_event());
    EXPECT_EQ(fixture.slab.count, 0u);
    EXPECT_EQ(fixture.slab.error_code, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
    EXPECT_EQ(fixture.event.wait_calls, 0u);
}

TEST(A5UrmaCompletionKernel, PartialFailedPostRetainsAllSubmittedTargets) {
    RegistrationFixture fixture;
    fixture.event.postFailed = true;

    EXPECT_FALSE(fixture.register_event());
    ASSERT_EQ(fixture.slab.count, 2u);
    EXPECT_EQ(fixture.slab.entries[0].addr, ((static_cast<uint64_t>(6) << 32) | 4));
    EXPECT_EQ(fixture.slab.entries[1].expected_value, 7u);
    EXPECT_EQ(fixture.slab.error_code, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
    EXPECT_EQ(fixture.event.wait_calls, 0u);
}

TEST(A5UrmaCompletionKernel, InvalidDeferredContextDoesNotWaitOrSucceed) {
    RegistrationFixture fixture;
    fixture.ctx.completion_entries = nullptr;

    EXPECT_FALSE(fixture.register_event());
    EXPECT_EQ(fixture.slab.count, 0u);
    EXPECT_EQ(fixture.event.wait_calls, 0u);
}

TEST(A5UrmaCompletionKernel, WrongEngineFailsWithoutWaiting) {
    RegistrationFixture fixture;
    fixture.event.engine = pto::comm::DmaEngine::SDMA;

    EXPECT_FALSE(fixture.register_event());
    EXPECT_EQ(fixture.slab.count, 0u);
    EXPECT_EQ(fixture.slab.error_code, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
    EXPECT_EQ(fixture.event.wait_calls, 0u);
}

TEST(A5UrmaCompletionKernel, InvalidJettyRangeRegistersNothing) {
    RegistrationFixture fixture;
    fixture.event.urmaJettyBase = 3;
    fixture.event.urmaJettyCount = 2;

    EXPECT_FALSE(fixture.register_event());
    EXPECT_EQ(fixture.slab.count, 0u);
    EXPECT_EQ(fixture.event.wait_calls, 0u);
}

TEST(A5UrmaCompletionKernel, InvalidSessionRegistersNothing) {
    RegistrationFixture fixture;
    fixture.session.valid = false;

    EXPECT_FALSE(fixture.register_event());
    EXPECT_EQ(fixture.slab.count, 0u);
    EXPECT_EQ(fixture.event.wait_calls, 0u);
}

TEST(A5UrmaCompletionKernel, RepeatedEventsAppendWithoutLosingEarlierTargets) {
    RegistrationFixture fixture;
    ASSERT_TRUE(fixture.register_event());
    fixture.event.urmaTargetBbPerJetty[0] = 5;
    fixture.event.urmaTargetCqePerJetty[0] = 4;
    fixture.event.handle = (uint64_t{1} << 32) | 5;

    ASSERT_TRUE(fixture.register_event());
    ASSERT_EQ(fixture.slab.count, 4u);
    EXPECT_EQ(fixture.slab.entries[0].addr, ((static_cast<uint64_t>(6) << 32) | 4));
    EXPECT_EQ(fixture.slab.entries[2].addr, ((static_cast<uint64_t>(6) << 32) | 5));
    EXPECT_EQ(fixture.slab.entries[2].expected_value, 4u);
}

TEST(A5UrmaCompletionKernel, CapacityBudgetAccountsForEachParticipatingChunk) {
    RegistrationFixture fixture;
    fixture.ctx.completion_capacity = 8;
    fixture.slab.count = 3;

    EXPECT_TRUE(simpler::urma::preflight_urma_request(fixture.runtime, fixture.session, 5));
    EXPECT_FALSE(simpler::urma::preflight_urma_request(fixture.runtime, fixture.session, 6));
    EXPECT_EQ(fixture.slab.count, 3u);
}

TEST(A5UrmaCompletionKernel, InvalidPublishedCountCannotUnderflowCapacity) {
    RegistrationFixture fixture;
    fixture.ctx.completion_capacity = 2;
    fixture.slab.count = 3;

    EXPECT_FALSE(fixture.runtime.capacity_available(1));
    EXPECT_FALSE(fixture.register_event());
    EXPECT_EQ(fixture.slab.count, 3u);
    EXPECT_EQ(fixture.event.wait_calls, 0u);
}

TEST(A5UrmaCompletionKernel, FailedContextCannotAdmitMoreWork) {
    RegistrationFixture fixture;
    fixture.slab.error_code = SIMPLER_ERROR_ASYNC_COMPLETION_INVALID;

    EXPECT_FALSE(fixture.runtime.capacity_available(1));
    EXPECT_FALSE(fixture.register_event());
    EXPECT_EQ(fixture.slab.count, 0u);
    EXPECT_EQ(fixture.slab.error_code, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
}

TEST(A5UrmaCompletionKernel, ChunkBudgetHandlesEmptyBoundaryAndMaximumSize) {
    constexpr uint64_t limit = pto::comm::urma::kUrmaMaxWqeTransferBytes;
    EXPECT_EQ(simpler::urma::chunk_count(0), 0u);
    EXPECT_EQ(simpler::urma::chunk_count(limit), 1u);
    EXPECT_EQ(simpler::urma::chunk_count(limit + 1), 2u);
    EXPECT_EQ(simpler::urma::chunk_count(UINT64_MAX), 1u + (UINT64_MAX - 1u) / limit);
}

TEST(A5UrmaCompletionKernel, OversizedChunkBudgetFailsBeforeQueueAccess) {
    RegistrationFixture fixture;

    EXPECT_FALSE(simpler::urma::preflight_urma_request(fixture.runtime, fixture.session, UINT64_MAX));
    EXPECT_EQ(fixture.slab.error_code, SIMPLER_ERROR_ASYNC_REGISTRATION_FAILED);
    EXPECT_EQ(fixture.slab.count, 0u);
}

TEST(A5UrmaCompletionKernel, SharedPoolResolvesQueueIndependentlyOfPeer) {
    RegistrationFixture fixture;
    fixture.info.layout = pto::comm::urma::UrmaLayout::SHARED_POOL;
    fixture.info.rowsPerPeer = 1;
    fixture.info.jettyCount = 4;

    ASSERT_TRUE(fixture.register_event());
    ASSERT_EQ(fixture.slab.count, 2u);
    EXPECT_EQ(fixture.slab.entries[0].addr, ((static_cast<uint64_t>(2) << 32) | 4));
    EXPECT_EQ(fixture.slab.entries[1].addr, ((static_cast<uint64_t>(3) << 32) | 9));
}

TEST(A5UrmaCompletionKernel, WholeRequestPreflightAcceptsRoomInEveryQueue) {
    RegistrationFixture fixture;

    EXPECT_TRUE(simpler::urma::preflight_urma_request(fixture.runtime, fixture.session, 3));
    EXPECT_EQ(fixture.slab.count, 0u);
    EXPECT_EQ(fixture.slab.error_code, SIMPLER_ERROR_NONE);
    EXPECT_EQ(fixture.sq_head[6], 0u);
    EXPECT_EQ(fixture.sq[6].submittedWqeCount, 0u);
}

TEST(A5UrmaCompletionKernel, PreflightReservesOnlyTheChunksAssignedToEachQueue) {
    RegistrationFixture fixture;
    fixture.ctx.completion_capacity = 3;
    fixture.sq_head[6] = 14;
    fixture.sq[6].submittedWqeCount = 14;
    fixture.sq_head[7] = 15;
    fixture.sq[7].submittedWqeCount = 15;

    EXPECT_TRUE(simpler::urma::preflight_urma_request(fixture.runtime, fixture.session, 3));
    EXPECT_EQ(fixture.slab.count, 0u);
    EXPECT_EQ(fixture.slab.error_code, SIMPLER_ERROR_NONE);
    EXPECT_EQ(fixture.sq_doorbell[6], 0u);
    EXPECT_EQ(fixture.sq_doorbell[7], 0u);
}

TEST(A5UrmaCompletionKernel, FullUnusedJettyDoesNotRejectSmallRequest) {
    RegistrationFixture fixture;
    fixture.sq_head[7] = 16;
    fixture.sq[7].submittedWqeCount = 16;

    EXPECT_TRUE(simpler::urma::preflight_urma_request(fixture.runtime, fixture.session, 1));
    EXPECT_EQ(fixture.slab.error_code, SIMPLER_ERROR_NONE);
    EXPECT_EQ(fixture.sq_head[7], 16u);
}

TEST(A5UrmaCompletionKernel, SharedPoolKeepsFullGroupCompletionBudget) {
    RegistrationFixture fixture;
    fixture.info.layout = pto::comm::urma::UrmaLayout::SHARED_POOL;
    fixture.info.rowsPerPeer = 1;
    fixture.info.jettyCount = 4;
    fixture.ctx.completion_capacity = 3;

    EXPECT_FALSE(simpler::urma::preflight_urma_request(fixture.runtime, fixture.session, 2));
    EXPECT_EQ(fixture.slab.error_code, SIMPLER_ERROR_ASYNC_REGISTRATION_FAILED);
    EXPECT_EQ(fixture.slab.count, 0u);
}

TEST(A5UrmaCompletionKernel, PerPeerCannotUseRailSelectedQueueOffsets) {
    RegistrationFixture fixture;
    fixture.info.railCycleLen = 1;
    fixture.info.railCycle[0] = 1;

    EXPECT_FALSE(simpler::urma::preflight_urma_request(fixture.runtime, fixture.session, 2));
    EXPECT_EQ(fixture.slab.error_code, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
    EXPECT_EQ(fixture.slab.count, 0u);
    EXPECT_EQ(fixture.sq_doorbell[6], 0u);
    EXPECT_EQ(fixture.sq_doorbell[7], 0u);
}

TEST(A5UrmaCompletionKernel, FullSqInOneJettyRejectsWholeRequest) {
    RegistrationFixture fixture;
    fixture.sq_head[6] = 16;

    EXPECT_FALSE(simpler::urma::preflight_urma_request(fixture.runtime, fixture.session, 1));
    EXPECT_EQ(fixture.slab.count, 0u);
    EXPECT_EQ(fixture.slab.error_code, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
    EXPECT_EQ(fixture.sq_head[6], 16u);
    EXPECT_EQ(fixture.sq_head[7], 0u);
    EXPECT_EQ(fixture.sq_doorbell[6], 0u);
    EXPECT_EQ(fixture.sq_doorbell[7], 0u);
}

TEST(A5UrmaCompletionKernel, FullCqInAnotherJettyRejectsWholeRequest) {
    RegistrationFixture fixture;
    fixture.sq[7].submittedWqeCount = 16;

    EXPECT_FALSE(simpler::urma::preflight_urma_request(fixture.runtime, fixture.session, 2));
    EXPECT_EQ(fixture.slab.count, 0u);
    EXPECT_EQ(fixture.slab.error_code, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
    EXPECT_EQ(fixture.sq[7].submittedWqeCount, 16u);
    EXPECT_EQ(fixture.sq_head[6], 0u);
    EXPECT_EQ(fixture.sq_doorbell[6], 0u);
}

TEST(A5UrmaCompletionKernel, PreflightReservesAllChunksBeforeFirstSubmission) {
    RegistrationFixture fixture;
    fixture.sq_head[6] = 14;
    fixture.sq[6].submittedWqeCount = 14;

    EXPECT_FALSE(simpler::urma::preflight_urma_request(fixture.runtime, fixture.session, 5));
    EXPECT_EQ(fixture.sq_head[6], 14u);
    EXPECT_EQ(fixture.sq[6].submittedWqeCount, 14u);
    EXPECT_EQ(fixture.slab.count, 0u);
}

TEST(A5UrmaCompletionKernel, ExactRemainingRingCapacityCanBeReserved) {
    RegistrationFixture fixture;
    fixture.sq_head[6] = 14;
    fixture.sq[6].submittedWqeCount = 14;

    EXPECT_TRUE(simpler::urma::preflight_urma_request(fixture.runtime, fixture.session, 3));
    EXPECT_EQ(fixture.slab.error_code, SIMPLER_ERROR_NONE);
    EXPECT_EQ(fixture.sq_head[6], 14u);
    EXPECT_EQ(fixture.sq[6].submittedWqeCount, 14u);
}

TEST(A5UrmaCompletionKernel, PreflightUsesUnsignedDistancesAcrossCounterWrap) {
    RegistrationFixture fixture;
    fixture.sq_head[6] = 1;
    fixture.sq_tail[6] = UINT32_MAX;
    fixture.sq[6].submittedWqeCount = 2;
    fixture.cq_tail[6] = UINT32_MAX;

    EXPECT_TRUE(simpler::urma::preflight_urma_request(fixture.runtime, fixture.session, 3));
    EXPECT_EQ(fixture.slab.error_code, SIMPLER_ERROR_NONE);
    EXPECT_EQ(fixture.sq_tail[6], UINT32_MAX);
    EXPECT_EQ(fixture.cq_tail[6], UINT32_MAX);
}

TEST(A5UrmaCompletionKernel, CompletionBudgetRejectsBeforeReadingQueueTables) {
    RegistrationFixture fixture;
    fixture.ctx.completion_capacity = 2;
    fixture.info.sqPtr = 0;
    fixture.info.scqPtr = 0;

    EXPECT_FALSE(simpler::urma::preflight_urma_request(fixture.runtime, fixture.session, 3));
    EXPECT_EQ(fixture.slab.error_code, SIMPLER_ERROR_ASYNC_REGISTRATION_FAILED);
    EXPECT_EQ(fixture.slab.count, 0u);
}

TEST(A5UrmaCompletionKernel, PreflightRejectsMalformedRingDepth) {
    RegistrationFixture fixture;
    fixture.cq[7].depth = 15;

    EXPECT_FALSE(simpler::urma::preflight_urma_request(fixture.runtime, fixture.session, 1));
    EXPECT_EQ(fixture.slab.error_code, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
    EXPECT_EQ(fixture.sq_head[6], 0u);
    EXPECT_EQ(fixture.sq_head[7], 0u);
    EXPECT_EQ(fixture.slab.count, 0u);
}

TEST(A5UrmaCompletionKernel, PreflightRejectsJettyOutsideWorkspace) {
    RegistrationFixture fixture;
    fixture.session.qpIdxBase = 3;

    EXPECT_FALSE(simpler::urma::preflight_urma_request(fixture.runtime, fixture.session, 1));
    EXPECT_EQ(fixture.slab.error_code, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
    EXPECT_EQ(fixture.sq_head[7], 0u);
    EXPECT_EQ(fixture.slab.count, 0u);
}

class A5UrmaSessionBinding : public testing::TestWithParam<pto::comm::urma::UrmaLayout>,
                             public RegistrationFixture {
protected:
    PostScript script;

    void SetUp() override {
        info.layout = GetParam();
        if (info.layout == pto::comm::urma::UrmaLayout::SHARED_POOL) {
            info.rowsPerPeer = 1;
            info.jettyCount = 4;
        }
        active_script = &script;
    }
    void TearDown() override { active_script = nullptr; }

    bool build(uint32_t sender, TestSession &bound) {
        ctx.urma_sender_id = sender;
        return simpler::urma::build_urma_session(ctx.urma_sender_id, workspace(), 1, bound);
    }

    uint32_t completions_per_small_request() const {
        return info.layout == pto::comm::urma::UrmaLayout::PER_PEER ? 1 : 2;
    }

    uint32_t first_row() const { return info.layout == pto::comm::urma::UrmaLayout::PER_PEER ? 6 : 2; }

    void post_two_unretired_tasks(DeferredCompletionSlab &next_slab) {
        script.post_from_queue = true;
        const uint32_t row = first_row();
        sq_head[row] = 15;
        sq[row].submittedWqeCount = 15;
        sq_tail[row] = 14;
        cq_tail[row] = 14;
        cq_doorbell[row] = 14;
        using Shape = pto::Shape<1, 1, 1, 1, pto::DYNAMIC>;
        using Stride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;
        using Tensor = pto::GlobalTensor<float, Shape, Stride>;
        Tensor dst({0x100000000ULL}, Shape(1, 1, 1, 1, 8), Stride(8, 8, 8, 8, 1));
        Tensor src({0x200000000ULL}, Shape(1, 1, 1, 1, 8), Stride(8, 8, 8, 8, 1));
        ASSERT_TRUE(send_request_entry(ctx, UrmaTput(dst, src, workspace(), 1)));
        ASSERT_EQ(sq_head[row], 16u);
        ASSERT_EQ(sq[row].submittedWqeCount, 16u);
        ASSERT_EQ(sq_tail[row], 14u);
        ASSERT_EQ(cq_tail[row], 14u);
        ASSERT_EQ(slab.count, completions_per_small_request());

        AsyncCtx next = AsyncCtx::make(TaskId::make(0, 99), &next_slab);
        next.urma_sender_id = ctx.urma_sender_id;
        EXPECT_NE(TaskId::to_uint64(next.task_token), TaskId::to_uint64(ctx.task_token));
        ASSERT_TRUE(send_request_entry(next, UrmaTput(dst, src, workspace(), 1)));
        ASSERT_EQ(next_slab.count, completions_per_small_request());
        EXPECT_EQ(script.activation_calls, 2u);
        EXPECT_EQ(sq_head[row], 17u);
        EXPECT_EQ(sq[row].submittedWqeCount, 17u);
        EXPECT_EQ(sq_tail[row], 14u);
        EXPECT_EQ(cq_tail[row], 14u);
        EXPECT_EQ(cq_doorbell[row], 14u);
        EXPECT_EQ(slab.entries[0].addr, (static_cast<uint64_t>(row) << 32) | 16u);
        EXPECT_EQ(slab.entries[0].expected_value, 16u);
        EXPECT_EQ(next_slab.entries[0].addr, (static_cast<uint64_t>(row) << 32) | 17u);
        EXPECT_EQ(next_slab.entries[0].expected_value, 17u);
        EXPECT_EQ(slab.error_code, SIMPLER_ERROR_NONE);
        EXPECT_EQ(next_slab.error_code, SIMPLER_ERROR_NONE);
    }

    void set_cqe(uint32_t sequence, bool ready, uint32_t status = 0) {
        const uint32_t lap = (sequence / 16) & 1u;
        const uint32_t owner = ready ? lap ^ 1u : lap;
        auto *dw0 = reinterpret_cast<uint32_t *>(cq_storage[first_row()].data() + (sequence % 16) * 64);
        *dw0 = (owner << 2) | (status << 24);
    }

    CompletionPollResult poll(const volatile DeferredCompletionEntry &entry) {
        return pto2::urma_backend::poll_urma_event_handle(entry.addr, entry.backend_cookie, entry.expected_value);
    }
};

TEST_P(A5UrmaSessionBinding, CrossTaskReusePreservesOutstandingTargetsAcrossWrap) {
    DeferredCompletionSlab next_slab{};
    post_two_unretired_tasks(next_slab);
    ASSERT_EQ(next_slab.count, completions_per_small_request());
    set_cqe(14, false);
    set_cqe(15, true);
    set_cqe(16, true);
    EXPECT_EQ(poll(next_slab.entries[0]).state, CompletionPollState::PENDING);
    EXPECT_EQ(sq_tail[first_row()], 14u);
    EXPECT_EQ(cq_tail[first_row()], 14u);

    set_cqe(14, true);
    EXPECT_EQ(poll(next_slab.entries[0]).state, CompletionPollState::READY);
    if (next_slab.count == 2) EXPECT_EQ(poll(next_slab.entries[1]).state, CompletionPollState::READY);
    EXPECT_EQ(poll(slab.entries[0]).state, CompletionPollState::READY);
    if (slab.count == 2) EXPECT_EQ(poll(slab.entries[1]).state, CompletionPollState::READY);
    EXPECT_EQ(sq_tail[first_row()], 17u);
    EXPECT_EQ(cq_tail[first_row()], 17u);
    EXPECT_EQ(cq_doorbell[first_row()], 17u);
    EXPECT_EQ(sq_head[first_row()], 17u);
    EXPECT_EQ(sq[first_row()].submittedWqeCount, 17u);
}

TEST_P(A5UrmaSessionBinding, CrossTaskReuseCannotHideAnOlderRequestError) {
    DeferredCompletionSlab next_slab{};
    post_two_unretired_tasks(next_slab);
    ASSERT_EQ(next_slab.count, completions_per_small_request());
    set_cqe(14, true);
    set_cqe(15, true, 7);
    set_cqe(16, true);
    EXPECT_EQ(poll(next_slab.entries[0]).state, CompletionPollState::FAILED);
    EXPECT_EQ(poll(slab.entries[0]).state, CompletionPollState::FAILED);
    EXPECT_EQ(sq_tail[first_row()], 14u);
    EXPECT_EQ(cq_tail[first_row()], 14u);
    EXPECT_EQ(cq_doorbell[first_row()], 14u);
}

TEST_P(A5UrmaSessionBinding, IndependentLogicalBlockZeroTasksUseResidentAivQueues) {
    std::array<LocalContext, 2> local{};
    std::array<GlobalContext, 2> global{};
    std::array<TestSession, 2> bound{};
    for (uint32_t lane = 0; lane < 2; ++lane) {
        local[lane].block_idx = 0;
        local[lane].block_num = 1;
        local[lane].async_ctx = ctx;
        local[lane].async_ctx.urma_sender_id = lane;
        local[lane].async_ctx.task_token = TaskId::make(0, 7 + lane);
        global[lane].sub_block_id = lane;
        std::array<int64_t, 50> args{};
        args[PAYLOAD_LOCAL_CONTEXT_INDEX] = reinterpret_cast<int64_t>(&local[lane]);
        args[PAYLOAD_GLOBAL_CONTEXT_INDEX] = reinterpret_cast<int64_t>(&global[lane]);
        EXPECT_EQ(get_block_idx(args.data()), 0);
        EXPECT_EQ(get_sub_block_id(args.data()), lane);
        AsyncCtx loaded = get_async_ctx(args.data());
        EXPECT_EQ(loaded.urma_sender_id, lane);
        ASSERT_TRUE(simpler::urma::build_urma_session(loaded.urma_sender_id, workspace(), 1, bound[lane]));
        EXPECT_EQ(bound[lane].qpIdxBase, lane * 2);
    }
    EXPECT_NE(bound[0].qpIdxBase, bound[1].qpIdxBase);
    // The default ISA builder is deleted in this translation unit. Even a block
    // identity outside the workspace cannot override a valid resident sender.
    local[1].block_idx = INT32_MAX;
    std::array<int64_t, 50> args{};
    args[PAYLOAD_LOCAL_CONTEXT_INDEX] = reinterpret_cast<int64_t>(&local[1]);
    args[PAYLOAD_GLOBAL_CONTEXT_INDEX] = reinterpret_cast<int64_t>(&global[1]);
    AsyncCtx loaded = get_async_ctx(args.data());
    ASSERT_TRUE(simpler::urma::build_urma_session(loaded.urma_sender_id, workspace(), 1, bound[1]));
    EXPECT_EQ(bound[1].qpIdxBase, 2u);
    EXPECT_EQ(bound[1].cfg.urma.qpCount, 2u);
    EXPECT_EQ(bound[1].engine, pto::comm::DmaEngine::URMA);
    EXPECT_EQ(bound[1].contextGm, workspace());
    EXPECT_EQ(bound[1].destRankId, 1u);
    EXPECT_TRUE(bound[1].valid);
}

TEST_P(A5UrmaSessionBinding, SameResidentSenderReusesQueueAcrossTasksAndPeers) {
    TestSession first;
    ASSERT_TRUE(build(1, first));
    ctx.task_token = TaskId::make(0, 99);
    TestSession next;
    ASSERT_TRUE(build(1, next));
    EXPECT_EQ(first.qpIdxBase, 2u);
    EXPECT_EQ(next.qpIdxBase, first.qpIdxBase);
    ASSERT_TRUE(simpler::urma::build_urma_session(ctx.urma_sender_id, workspace(), 0, next));
    EXPECT_EQ(next.qpIdxBase, first.qpIdxBase);
    EXPECT_EQ(next.destRankId, 0u);
}

TEST_P(A5UrmaSessionBinding, RejectsMissingAndOutOfRangeSenders) {
    TestSession bound;
    EXPECT_FALSE(build(UINT32_MAX, bound));
    EXPECT_FALSE(bound.valid);
    EXPECT_FALSE(build(2, bound));
    EXPECT_FALSE(bound.valid);
}

TEST_P(A5UrmaSessionBinding, RejectsMalformedLayoutAndQueueGeometry) {
    TestSession bound;
    info.layout = static_cast<pto::comm::urma::UrmaLayout>(99);
    EXPECT_FALSE(build(0, bound));
    EXPECT_FALSE(bound.valid);
    info.layout = GetParam();
    info.jettiesPerCore = 0;
    EXPECT_FALSE(build(0, bound));
    info.jettiesPerCore = 9;
    EXPECT_FALSE(build(0, bound));
    info.jettiesPerCore = 2;
    info.jettyCount = 3;
    EXPECT_FALSE(build(1, bound));
}

TEST_P(A5UrmaSessionBinding, RejectsMissingWorkspaceAndOutOfRangePeer) {
    TestSession bound;
    EXPECT_FALSE(simpler::urma::build_urma_session(ctx.urma_sender_id, nullptr, 1, bound));
    EXPECT_FALSE(bound.valid);
    EXPECT_FALSE(simpler::urma::build_urma_session(ctx.urma_sender_id, workspace(), info.rankCount, bound));
    EXPECT_FALSE(bound.valid);
}

INSTANTIATE_TEST_SUITE_P(
    QueueLayouts, A5UrmaSessionBinding,
    testing::Values(pto::comm::urma::UrmaLayout::PER_PEER, pto::comm::urma::UrmaLayout::SHARED_POOL)
);

class A5UrmaChunkedSubmission : public testing::TestWithParam<UrmaOp>,
                                public RegistrationFixture {
protected:
    static constexpr uint64_t kLimit = pto::comm::urma::kUrmaMaxWqeTransferBytes;
    static constexpr uint64_t kElements = 2 * (kLimit / sizeof(float)) + 8192;
    static constexpr uint64_t kDstBase = 0x100000000ULL;
    static constexpr uint64_t kSrcBase = 0x200000000ULL;
    PostScript script;

    void SetUp() override {
        total_wait_calls = 0;
        for (uint32_t chunk = 0; chunk < 3; ++chunk) {
            TestEvent outcome = event;
            outcome.handle += chunk;
            outcome.urmaJettyBase = 2 + chunk % 2;
            outcome.urmaJettyCount = 1;
            for (uint32_t jetty = 0; jetty < 2; ++jetty) {
                outcome.urmaTargetBbPerJetty[jetty] += chunk;
                outcome.urmaTargetCqePerJetty[jetty] += chunk;
            }
            script.events.push_back(outcome);
        }
        active_script = &script;
    }
    void TearDown() override { active_script = nullptr; }

    template <typename T = float>
    bool submit(uint64_t elements = kElements) {
        return submit_with<T>(runtime, elements);
    }

    template <typename T = float, typename Runtime>
    bool submit_with(Runtime &adapter, uint64_t elements = kElements) {
        using Shape = pto::Shape<1, 1, 1, 1, pto::DYNAMIC>;
        using Stride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;
        using Tensor = pto::GlobalTensor<T, Shape, Stride>;
        Tensor dst({kDstBase}, Shape(1, 1, 1, 1, elements), Stride(elements, elements, elements, elements, 1));
        Tensor src({kSrcBase}, Shape(1, 1, 1, 1, elements), Stride(elements, elements, elements, elements, 1));
        return simpler::urma::submit_chunked_urma_request(
            adapter, UrmaRequestDescriptor<Tensor, Tensor>{GetParam(), dst, src, workspace(), session.destRankId}
        );
    }

    void expect_consumer_blocked() {
        auto context = std::make_unique<SchedulerContext>();
        auto scheduler = std::make_unique<SchedulerState>();
        auto header = std::make_unique<SharedMemoryHeader>();
        scheduler->sm_header = header.get();
        header->sched_error_code.store(SIMPLER_ERROR_NONE);
        ChipTaskSlotState producer{}, consumer{};
        TaskPayload payload{};
        TaskDescriptor task{};
        TaskId::assign(task.task_id, ctx.task_token);
        producer.task = &task;
        producer.payload = &payload;
        producer.total_required_subtasks = 1;
        producer.task_state.store(CHIP_TASK_PENDING);
        consumer.fanin_count = 2;
        consumer.fanin_refcount.store(0);
        consumer.task_state.store(CHIP_TASK_PENDING);
        DepListEntry edge{&consumer, nullptr};
        producer.fanout_head = &edge;
        producer.fanout_count = 1;
        int32_t completed = 0, release_count = 0;
        ChipTaskSlotState *release[1]{};

        SchedulerRetirementTestPeer::finish_failed_submission(
            *context, *scheduler, slab, producer, completed, release, release_count
        );
        EXPECT_TRUE(context->is_completed());
        EXPECT_EQ(header->sched_error_code.load(), slab.error_code);
        EXPECT_EQ(producer.completed_subtasks.load(), 0);
        EXPECT_EQ(producer.task_state.load(), CHIP_TASK_PENDING);
        ASSERT_EQ(consumer.fanin_refcount.load(), 0);
        EXPECT_EQ(consumer.task_state.load(), CHIP_TASK_PENDING);
        EXPECT_EQ(completed, 0);
        EXPECT_EQ(release_count, 0);

        // The same edge releases a dependency when normal completion is called.
        // This checks the fanout wiring, rather than observing an unrelated slot.
        scheduler->on_task_complete(producer);
        EXPECT_EQ(consumer.fanin_refcount.load(), 1);
    }
};

TEST_P(A5UrmaChunkedSubmission, PostsEveryChunkOnceWithExactOffsetsAndPackedExtents) {
    ASSERT_TRUE(submit());
    ASSERT_EQ(script.submissions.size(), 3u);
    EXPECT_EQ(script.activation_calls, 1u);
    const uint64_t offsets[] = {0, kLimit, 2 * kLimit};
    const int64_t lengths[] = {kLimit / sizeof(float), kLimit / sizeof(float), 8192};
    for (size_t i = 0; i < script.submissions.size(); ++i) {
        const auto &post = script.submissions[i];
        EXPECT_EQ(post.put, GetParam() == UrmaOp::TPUT);
        EXPECT_EQ(post.dst, kDstBase + offsets[i]);
        EXPECT_EQ(post.src, kSrcBase + offsets[i]);
        EXPECT_EQ(post.shape, (std::array<int64_t, 5>{1, 1, 1, 1, lengths[i]}));
        EXPECT_EQ(post.stride, (std::array<int64_t, 5>{lengths[i], lengths[i], lengths[i], lengths[i], 1}));
    }
    EXPECT_EQ(slab.count, 3u);
    EXPECT_EQ(slab.error_code, SIMPLER_ERROR_NONE);
    EXPECT_EQ(total_wait_calls, 0u);
}

TEST_P(A5UrmaChunkedSubmission, RotatesActualQueuePostsWithinOneLargeRequest) {
    script.post_from_queue = true;
    ASSERT_TRUE(submit());
    ASSERT_EQ(script.submissions.size(), 3u);
    EXPECT_EQ(sq_head[6], 2u);
    EXPECT_EQ(sq[6].submittedWqeCount, 2u);
    EXPECT_EQ(sq_doorbell[6], 2u);
    EXPECT_EQ(sq_head[7], 1u);
    EXPECT_EQ(sq[7].submittedWqeCount, 1u);
    EXPECT_EQ(sq_doorbell[7], 1u);
    ASSERT_EQ(slab.count, 3u);
    EXPECT_EQ(slab.entries[0].addr, (uint64_t{6} << 32) | 1);
    EXPECT_EQ(slab.entries[1].addr, (uint64_t{7} << 32) | 1);
    EXPECT_EQ(slab.entries[2].addr, (uint64_t{6} << 32) | 2);
    EXPECT_EQ(slab.entries[0].expected_value, 1u);
    EXPECT_EQ(slab.entries[1].expected_value, 1u);
    EXPECT_EQ(slab.entries[2].expected_value, 2u);
    EXPECT_EQ(sq_tail[6], 0u);
    EXPECT_EQ(sq_tail[7], 0u);
    EXPECT_EQ(cq_tail[6], 0u);
    EXPECT_EQ(cq_tail[7], 0u);
}

TEST_P(A5UrmaChunkedSubmission, QueueUsageMatchesChunkBoundariesAndTail) {
    script.post_from_queue = true;
    for (const uint64_t bytes :
         {kLimit - sizeof(float), kLimit, kLimit + sizeof(float), 2 * kLimit, 2 * kLimit + 32768}) {
        SCOPED_TRACE(bytes);
        sq_head.fill(0);
        for (auto &queue : sq)
            queue.submittedWqeCount = 0;
        slab.count = 0;
        script.submissions.clear();
        ASSERT_TRUE(submit(bytes / sizeof(float)));
        const uint32_t chunks = bytes > 2 * kLimit ? 3 : (bytes > kLimit ? 2 : 1);
        EXPECT_EQ(sq_head[6], (chunks + 1) / 2);
        EXPECT_EQ(sq_head[7], chunks / 2);
        EXPECT_EQ(slab.count, chunks);
        ASSERT_EQ(script.submissions.size(), chunks);
        uint64_t posted_bytes = 0;
        for (const auto &post : script.submissions) {
            const uint64_t length = post.shape[4] * sizeof(float);
            EXPECT_LE(length, kLimit);
            EXPECT_EQ(post.dst, kDstBase + posted_bytes);
            EXPECT_EQ(post.src, kSrcBase + posted_bytes);
            posted_bytes += length;
        }
        EXPECT_EQ(posted_bytes, bytes);
    }
}

TEST_P(A5UrmaChunkedSubmission, ConsecutiveSmallRequestsStartAtFirstJetty) {
    script.post_from_queue = true;
    for (uint32_t request = 0; request < 3; ++request)
        ASSERT_TRUE(submit(8));
    EXPECT_EQ(sq_head[6], 3u);
    EXPECT_EQ(sq[6].submittedWqeCount, 3u);
    EXPECT_EQ(sq_head[7], 0u);
    EXPECT_EQ(sq[7].submittedWqeCount, 0u);
    ASSERT_EQ(slab.count, 3u);
    for (uint32_t request = 0; request < 3; ++request) {
        EXPECT_EQ(slab.entries[request].addr, (uint64_t{6} << 32) | (request + 1));
        EXPECT_EQ(slab.entries[request].expected_value, request + 1);
    }
}

TEST_P(A5UrmaChunkedSubmission, SingleJettyKeepsEveryChunkOnItsSenderQueue) {
    script.post_from_queue = true;
    info.jettiesPerCore = 1;
    ASSERT_TRUE(submit());
    EXPECT_EQ(sq_head[5], 3u);
    ASSERT_EQ(slab.count, 3u);
    for (uint32_t chunk = 0; chunk < 3; ++chunk) {
        EXPECT_EQ(slab.entries[chunk].addr, (uint64_t{5} << 32) | (chunk + 1));
        EXPECT_EQ(slab.entries[chunk].expected_value, chunk + 1);
    }
    EXPECT_EQ(sq_head[4], 0u);
    EXPECT_EQ(sq_head[6], 0u);
}

TEST_P(A5UrmaChunkedSubmission, ThreeJettiesDistributeWholeChunksAndTail) {
    script.post_from_queue = true;
    info.jettiesPerCore = 3;
    info.rowsPerPeer = 3;
    info.jettyCount = 6;
    ctx.urma_sender_id = 0;
    ASSERT_TRUE(submit(3 * kLimit / sizeof(float) + 8192));
    EXPECT_EQ(sq_head[3], 2u);
    EXPECT_EQ(sq_head[4], 1u);
    EXPECT_EQ(sq_head[5], 1u);
    ASSERT_EQ(slab.count, 4u);
    EXPECT_EQ(slab.entries[0].addr, (uint64_t{3} << 32) | 1);
    EXPECT_EQ(slab.entries[1].addr, (uint64_t{4} << 32) | 1);
    EXPECT_EQ(slab.entries[2].addr, (uint64_t{5} << 32) | 1);
    EXPECT_EQ(slab.entries[3].addr, (uint64_t{3} << 32) | 2);
}

TEST_P(A5UrmaChunkedSubmission, NextTaskPreservesBothUnretiredQueueTargets) {
    script.post_from_queue = true;
    ASSERT_TRUE(submit());
    DeferredCompletionSlab next_slab{};
    ctx = AsyncCtx::make(TaskId::make(0, 99), &next_slab);
    ctx.urma_sender_id = 1;
    ASSERT_TRUE(submit());
    ASSERT_EQ(slab.count, 3u);
    ASSERT_EQ(next_slab.count, 3u);
    EXPECT_EQ(slab.entries[2].addr, (uint64_t{6} << 32) | 2);
    EXPECT_EQ(slab.entries[1].addr, (uint64_t{7} << 32) | 1);
    EXPECT_EQ(next_slab.entries[0].addr, (uint64_t{6} << 32) | 3);
    EXPECT_EQ(next_slab.entries[1].addr, (uint64_t{7} << 32) | 2);
    EXPECT_EQ(next_slab.entries[2].addr, (uint64_t{6} << 32) | 4);
    EXPECT_EQ(sq_head[6], 4u);
    EXPECT_EQ(sq_head[7], 2u);
    EXPECT_EQ(sq_tail[6], 0u);
    EXPECT_EQ(sq_tail[7], 0u);
    EXPECT_EQ(cq_tail[6], 0u);
    EXPECT_EQ(cq_tail[7], 0u);
}

TEST_P(A5UrmaChunkedSubmission, BothQueueTargetsSurviveCounterWrap) {
    script.post_from_queue = true;
    for (uint32_t row : {6u, 7u}) {
        sq_head[row] = UINT32_MAX;
        sq[row].submittedWqeCount = UINT32_MAX;
        sq_tail[row] = UINT32_MAX - 1;
        cq_tail[row] = UINT32_MAX - 1;
    }
    ASSERT_TRUE(submit());
    ASSERT_EQ(slab.count, 3u);
    EXPECT_EQ(slab.entries[0].addr, uint64_t{6} << 32);
    EXPECT_EQ(slab.entries[1].addr, uint64_t{7} << 32);
    EXPECT_EQ(slab.entries[2].addr, (uint64_t{6} << 32) | 1);
    EXPECT_EQ(slab.entries[0].expected_value, 0u);
    EXPECT_EQ(slab.entries[1].expected_value, 0u);
    EXPECT_EQ(slab.entries[2].expected_value, 1u);
    EXPECT_EQ(sq_tail[6], UINT32_MAX - 1);
    EXPECT_EQ(sq_tail[7], UINT32_MAX - 1);

    DeviceArena sm_arena, sched_arena;
    auto scheduler = std::make_unique<SchedulerState>();
    auto *sm = SharedMemoryHandle::create_and_init_default(sm_arena);
    ASSERT_NE(sm, nullptr);
    const auto layout = SchedulerState::reserve_layout(sched_arena);
    ASSERT_NE(sched_arena.commit(), nullptr);
    ASSERT_TRUE(scheduler->init_data_from_layout(layout, sched_arena, sm->header));
    scheduler->wire_arena_pointers(layout, sched_arena);
    ChipTaskSlotState producer{}, consumer{};
    TaskPayload payload{};
    producer.task_state.store(CHIP_TASK_PENDING);
    producer.active_mask = LaneMask::aiv0();
    producer.total_required_subtasks = 1;
    producer.logical_block_num = 1;
    producer.payload = &payload;
    TaskPayload consumer_payload{};
    TaskDescriptor consumer_task{};
    consumer.payload = &consumer_payload;
    consumer.task = &consumer_task;
    consumer.task_state.store(CHIP_TASK_PENDING);
    consumer.active_mask = LaneMask::aiv0();
    consumer.logical_block_num = 1;
    consumer.total_required_subtasks = 1;
    consumer.fanin_count = 1;
    consumer.fanin_refcount.store(0);
    auto &ready = scheduler->ready_queues[static_cast<int32_t>(consumer.active_mask.to_shape())];
    DepListEntry edge{&consumer, nullptr};
    producer.fanout_head = &edge;
    producer.fanout_count = 1;
    auto &waits = scheduler->async_wait_list;
    waits.count = 1;
    auto &entry = waits.entries[0];
    entry.slot_state = &producer;
    entry.task_token = ctx.task_token;
    entry.normal_done = true;
    int32_t error = SIMPLER_ERROR_NONE;
    for (uint32_t i = 0; i < slab.count; ++i) {
        const auto &condition = slab.entries[i];
        ASSERT_TRUE(waits.append_condition_locked(
            entry, condition.addr, condition.backend_cookie, condition.expected_value, ASYNC_ENGINE_URMA,
            COMPLETION_TYPE_URMA_EVENT_HANDLE, error
        ));
    }
    ASSERT_EQ(error, SIMPLER_ERROR_NONE);
    auto set_cqe = [&](uint32_t row, uint32_t sequence, bool ready) {
        const uint32_t phase = (sequence / 16) & 1u;
        const uint32_t owner = ready ? phase ^ 1u : phase;
        auto *dw0 = reinterpret_cast<uint32_t *>(cq_storage[row].data() + (sequence % 16) * 64);
        *dw0 = owner << 2;
    };
    for (uint32_t row : {6u, 7u}) {
        set_cqe(row, UINT32_MAX - 1, false);
        set_cqe(row, UINT32_MAX, false);
    }
    set_cqe(6, 0, false);
    ChipTaskSlotState *releases[1]{};
    int32_t release_count = 0;
    auto poll = [&] {
        return waits.poll_and_complete<false>(nullptr, scheduler.get(), releases, release_count, 1);
    };
    auto expect_pending = [&](int32_t remaining) {
        const auto result = poll();
        EXPECT_EQ(result.error_code, SIMPLER_ERROR_NONE);
        EXPECT_EQ(result.completed, 0);
        EXPECT_EQ(release_count, 0);
        EXPECT_EQ(producer.task_state.load(), CHIP_TASK_PENDING);
        EXPECT_EQ(consumer.fanin_refcount.load(), 0);
        EXPECT_EQ(ready.pop(), nullptr);
        ASSERT_EQ(waits.count, 1);
        EXPECT_EQ(waits.entries[0].waiting_completion_count, remaining);
    };
    expect_pending(3);
    for (uint32_t sequence : {UINT32_MAX - 1, UINT32_MAX, 0u})
        set_cqe(6, sequence, true);
    expect_pending(1);
    EXPECT_EQ(sq_tail[6], 1u);
    EXPECT_EQ(cq_tail[6], 1u);
    EXPECT_EQ(sq_tail[7], UINT32_MAX - 1);
    EXPECT_EQ(cq_tail[7], UINT32_MAX - 1);
    // Completing only the older CQE must not satisfy the wrapped zero target.
    set_cqe(7, UINT32_MAX - 1, true);
    expect_pending(1);
    set_cqe(7, UINT32_MAX, true);
    const auto complete = poll();
    EXPECT_EQ(complete.error_code, SIMPLER_ERROR_NONE);
    EXPECT_EQ(complete.completed, 1);
    EXPECT_EQ(release_count, 1);
    EXPECT_EQ(releases[0], &producer);
    EXPECT_EQ(producer.task_state.load(), CHIP_TASK_COMPLETED);
    EXPECT_EQ(consumer.fanin_refcount.load(), 1);
    EXPECT_EQ(ready.pop(), &consumer);
    EXPECT_EQ(ready.pop(), nullptr);
    EXPECT_EQ(sq_tail[7], 0u);
    EXPECT_EQ(cq_tail[7], 0u);
    EXPECT_EQ(waits.count, 0);
    scheduler->destroy();
}

TEST_P(A5UrmaChunkedSubmission, SecondChunkPartialFailureRetainsTargetsAndBlocksConsumer) {
    script.events[1].postFailed = true;
    ASSERT_FALSE(submit());
    ASSERT_EQ(script.submissions.size(), 2u);
    ASSERT_EQ(slab.count, 2u);
    EXPECT_EQ(slab.entries[0].addr, (uint64_t{6} << 32) | 4);
    EXPECT_EQ(slab.entries[0].expected_value, 3u);
    EXPECT_EQ(slab.entries[1].addr, (uint64_t{7} << 32) | 5);
    EXPECT_EQ(slab.entries[1].expected_value, 4u);
    for (uint32_t i = 0; i < slab.count; ++i)
        EXPECT_EQ(slab.entries[i].backend_cookie, reinterpret_cast<uint64_t>(workspace()));
    EXPECT_EQ(slab.error_code, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
    runtime.set_error(simpler::urma::RequestError::REGISTRATION_FAILED);
    EXPECT_EQ(slab.error_code, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
    EXPECT_EQ(total_wait_calls, 0u);
    expect_consumer_blocked();
}

TEST_P(A5UrmaChunkedSubmission, SecondChunkUnsubmittedFailureKeepsFirstChunkTargets) {
    script.events[1].postFailed = true;
    script.events[1].handle = 0;
    script.events[1].urmaJettyCount = 0;
    ASSERT_FALSE(submit());
    ASSERT_EQ(script.submissions.size(), 2u);
    ASSERT_EQ(slab.count, 1u);
    EXPECT_EQ(slab.entries[0].addr, (uint64_t{6} << 32) | 4);
    EXPECT_EQ(slab.entries[0].expected_value, 3u);
    EXPECT_EQ(slab.error_code, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
    expect_consumer_blocked();
}

TEST_P(A5UrmaChunkedSubmission, WholeRequestBudgetRejectsBeforeFirstPost) {
    ctx.completion_capacity = 2;
    EXPECT_FALSE(submit());
    EXPECT_EQ(script.activation_calls, 1u);
    EXPECT_TRUE(script.submissions.empty());
    EXPECT_EQ(slab.count, 0u);
    EXPECT_EQ(slab.error_code, SIMPLER_ERROR_ASYNC_REGISTRATION_FAILED);
}

TEST_P(A5UrmaChunkedSubmission, ExistingErrorRejectsBeforePostingAndStaysFirst) {
    slab.error_code = SIMPLER_ERROR_ASYNC_WAIT_OVERFLOW;
    EXPECT_FALSE(submit());
    EXPECT_TRUE(script.submissions.empty());
    EXPECT_EQ(slab.count, 0u);
    EXPECT_EQ(slab.error_code, SIMPLER_ERROR_ASYNC_WAIT_OVERFLOW);
}

TEST_P(A5UrmaChunkedSubmission, SessionFailureDoesNotPost) {
    ctx.urma_sender_id = UINT32_MAX;
    EXPECT_FALSE(submit());
    EXPECT_EQ(script.activation_calls, 1u);
    EXPECT_TRUE(script.submissions.empty());
    EXPECT_EQ(slab.error_code, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
}

TEST_P(A5UrmaChunkedSubmission, EmptyRequestDoesNotBuildSessionOrPost) {
    EXPECT_TRUE(submit(0));
    EXPECT_EQ(script.activation_calls, 0u);
    EXPECT_TRUE(script.submissions.empty());
    EXPECT_EQ(slab.error_code, SIMPLER_ERROR_NONE);
}

namespace {
struct IndependentRuntime {
    uint32_t sender{1};
    uint64_t capacity{8};
    uint64_t reserved{};
    uint32_t finishes{};
    bool reject_reservation{};
    bool failed{};
    simpler::urma::RequestError error{};
    std::vector<simpler::urma::CompletionTarget> targets;

    uint32_t sender_id() const { return sender; }
    bool capacity_available(uint64_t count) const { return !failed && count <= capacity - targets.size(); }
    bool reserve(uint64_t count) {
        EXPECT_TRUE(active_script->submissions.empty());
        if (reject_reservation) return false;
        reserved = count;
        return true;
    }
    bool register_target(const simpler::urma::CompletionTarget &target) {
        EXPECT_LT(targets.size(), reserved);
        targets.push_back(target);
        return true;
    }
    void set_error(simpler::urma::RequestError value) {
        if (!failed) error = value;
        failed = true;
    }
    bool finish() {
        ++finishes;
        return !failed;
    }
};
}  // namespace

TEST_P(A5UrmaChunkedSubmission, IndependentRuntimeReceivesExactQueueTargets) {
    IndependentRuntime adapter;
    ASSERT_TRUE(submit_with(adapter));
    ASSERT_EQ(adapter.targets.size(), 3u);
    EXPECT_EQ(adapter.reserved, 3u);
    EXPECT_EQ(adapter.finishes, 1u);
    EXPECT_EQ(slab.count, 0u);
    for (uint32_t i = 0; i < 3; ++i) {
        EXPECT_EQ(adapter.targets[i].workspace, workspace());
        EXPECT_EQ(adapter.targets[i].queue_row, 6 + i % 2);
        EXPECT_EQ(adapter.targets[i].target_bb, script.events[i].urmaTargetBbPerJetty[0]);
        EXPECT_EQ(adapter.targets[i].target_cqe, script.events[i].urmaTargetCqePerJetty[0]);
    }
}

TEST_P(A5UrmaChunkedSubmission, FailedReservationDoesNotPost) {
    IndependentRuntime adapter;
    adapter.reject_reservation = true;
    EXPECT_FALSE(submit_with(adapter));
    EXPECT_TRUE(script.submissions.empty());
    EXPECT_TRUE(adapter.targets.empty());
    EXPECT_EQ(adapter.finishes, 0u);
    EXPECT_EQ(adapter.error, simpler::urma::RequestError::REGISTRATION_FAILED);
}

TEST_P(A5UrmaChunkedSubmission, PartialFailureFinishesReservationWithSubmittedPrefix) {
    IndependentRuntime adapter;
    script.events[1].postFailed = true;
    EXPECT_FALSE(submit_with(adapter));
    ASSERT_EQ(script.submissions.size(), 2u);
    ASSERT_EQ(adapter.targets.size(), 2u);
    EXPECT_EQ(adapter.targets[1].target_bb, script.events[1].urmaTargetBbPerJetty[0]);
    EXPECT_EQ(adapter.targets[1].target_cqe, script.events[1].urmaTargetCqePerJetty[0]);
    EXPECT_EQ(adapter.finishes, 1u);
    EXPECT_EQ(adapter.error, simpler::urma::RequestError::INVALID);
    EXPECT_EQ(total_wait_calls, 0u);
}

INSTANTIATE_TEST_SUITE_P(GetAndPut, A5UrmaChunkedSubmission, testing::Values(UrmaOp::TGET, UrmaOp::TPUT));

class A5UrmaBatchSubmission : public testing::Test,
                              public RegistrationFixture {
protected:
    using Shape = pto::Shape<1, 1, 1, 1, pto::DYNAMIC>;
    using Stride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;
    using Tensor = pto::GlobalTensor<float, Shape, Stride>;
    using Request = UrmaRequestDescriptor<Tensor, Tensor>;
    using Mode = pto::comm::AsyncSubmitMode;

    static constexpr uint64_t kSrcBase = 0x200000000ULL;
    static constexpr uint64_t kDstBase = 0x100000000ULL;
    static constexpr uint64_t kPeerMappedBase = 0x300000000ULL;
    CommContext domain{};
    GlobalContext global{};
    std::array<pto::comm::urma::UrmaMemInfo, 8> peer_mrs{};
    PostScript script;

    void SetUp() override {
        domain.rankNum = 2;
        domain.winSize = 512 * 1024 * 1024;
        domain.windowsIn[0] = kSrcBase;
        domain.windowsIn[1] = kPeerMappedBase;
        ASSERT_TRUE(
            set_comm_workspace(domain, CommWorkspaceKind::URMA, reinterpret_cast<uint64_t>(workspace()), sizeof(info))
        );
        info.memPtr = reinterpret_cast<uint64_t>(peer_mrs.data());
        for (uint32_t row = info.rowsPerPeer; row < peer_mrs.size(); ++row) {
            peer_mrs[row].addr = kDstBase;
            peer_mrs[row].len = domain.winSize;
        }
        global.urma_batch_guard = URMA_BATCH_GUARD;
        script.batch_completion_count = ctx.completion_count;
        active_script = &script;
    }
    void TearDown() override { active_script = nullptr; }

    Request put(uint64_t offset = 0, int64_t elems = 8) {
        const Stride stride(elems, elems, elems, elems, 1);
        const Shape shape(1, 1, 1, 1, elems);
        return UrmaTput(
            Tensor({kDstBase + offset}, shape, stride), Tensor({kSrcBase + offset}, shape, stride), workspace(), 1
        );
    }

    bool submit_batch(const Request *requests, uint32_t count, uint32_t batch_size = UINT32_MAX) {
        LocalContext local{};
        local.async_ctx = ctx;
        std::array<int64_t, PAYLOAD_GLOBAL_CONTEXT_INDEX + 1> args{};
        args[PAYLOAD_LOCAL_CONTEXT_INDEX] = reinterpret_cast<int64_t>(&local);
        args[PAYLOAD_GLOBAL_CONTEXT_INDEX] = reinterpret_cast<int64_t>(&global);
        return send_urma_put_batch(args.data(), domain, requests, count, batch_size);
    }

    void expect_accounted_error(int32_t code) { EXPECT_EQ(slab.error_code, code | DEFERRED_ERROR_DMA_ACCOUNTED); }

    void expect_target(uint32_t slot, uint32_t row, uint32_t bb, uint32_t cqe) {
        EXPECT_EQ(slab.entries[slot].addr, (static_cast<uint64_t>(row) << 32) | bb);
        EXPECT_EQ(slab.entries[slot].expected_value, cqe);
        EXPECT_EQ(slab.entries[slot].engine, COMPLETION_ENGINE_URMA);
        EXPECT_EQ(slab.entries[slot].completion_type, COMPLETION_TYPE_URMA_EVENT_HANDLE);
        EXPECT_EQ(slab.entries[slot].backend_cookie, reinterpret_cast<uint64_t>(workspace()));
    }
};

TEST_F(A5UrmaBatchSubmission, ThresholdPublishesEveryJettyAndLastNonemptyRequestFlushesTail) {
    const Request requests[]{put(0, 0), put(32), put(64), put(96, 0), put(128), put(160, 0)};
    sq_head[6] = 14;
    sq_tail[6] = 14;
    sq[6].submittedWqeCount = 11;
    cq_tail[6] = 11;
    ASSERT_TRUE(submit_batch(requests, std::size(requests), 2));
    EXPECT_EQ(script.activation_calls, 1u);
    ASSERT_EQ(script.submissions.size(), 3u);
    EXPECT_EQ(script.submissions[0].mode, Mode::DEFER);
    EXPECT_EQ(script.submissions[1].mode, Mode::DEFER);
    EXPECT_EQ(script.submissions[2].mode, Mode::DEFER_AND_SUBMIT);
    EXPECT_EQ(script.submissions[2].src, requests[4].src.data().bytes);
    EXPECT_EQ(script.completion_counts_before_post, (std::vector<uint32_t>{0, 0, 0}));
    EXPECT_EQ(script.doorbells, (std::vector<uint32_t>{6, 7, 6}));
    ASSERT_EQ(slab.count, 2u);
    expect_target(0, 6, 16, 13);
    expect_target(1, 7, 1, 1);
}

TEST_F(A5UrmaBatchSubmission, BatchSizeOneFlushesEachRequestAndPreservesRotation) {
    const Request requests[]{put(0), put(32), put(64)};
    ASSERT_TRUE(submit_batch(requests, std::size(requests), 1));
    EXPECT_EQ(script.completion_counts_before_post, (std::vector<uint32_t>{0, 0, 0}));
    EXPECT_EQ(script.doorbells, (std::vector<uint32_t>{6, 7, 6}));
    ASSERT_EQ(slab.count, 2u);
    expect_target(0, 6, 2, 2);
    expect_target(1, 7, 1, 1);
}

TEST_F(A5UrmaBatchSubmission, FinalFlushIncludesFragmentedRequestAndCounterWrap) {
    constexpr int64_t chunk_elems = pto::comm::urma::kUrmaMaxWqeTransferBytes / sizeof(float);
    const Request requests[]{put(0, chunk_elems + 32), put((chunk_elems + 32) * sizeof(float))};
    sq_head[6] = UINT32_MAX - 1;
    sq_tail[6] = UINT32_MAX - 1;
    sq[6].submittedWqeCount = UINT32_MAX;
    cq_tail[6] = UINT32_MAX;
    ASSERT_TRUE(submit_batch(requests, std::size(requests)));
    ASSERT_EQ(script.submissions.size(), 2u);
    EXPECT_EQ(script.submissions[0].shape.back(), chunk_elems + 32);
    EXPECT_EQ(script.completion_counts_before_post, (std::vector<uint32_t>{0, 0}));
    EXPECT_EQ(script.doorbells, (std::vector<uint32_t>{6, 7}));
    ASSERT_EQ(slab.count, 2u);
    expect_target(0, 6, 0, 1);
    expect_target(1, 7, 1, 1);
}

TEST_F(A5UrmaBatchSubmission, RejectsPublicationWithoutHostReclamationGuard) {
    global.urma_batch_guard = 0;
    const Request requests[]{put()};
    EXPECT_FALSE(submit_batch(requests, 1));
    expect_accounted_error(SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
    EXPECT_TRUE(script.submissions.empty());
    EXPECT_EQ(slab.count, 0u);
}

TEST_F(A5UrmaBatchSubmission, QueueAndSlabCapacityRejectWholeListBeforeAnyPost) {
    const Request requests[]{put(0), put(32), put(64)};
    ctx.completion_capacity = 1;
    ASSERT_FALSE(submit_batch(requests, std::size(requests), 1));
    expect_accounted_error(SIMPLER_ERROR_ASYNC_REGISTRATION_FAILED);
    EXPECT_TRUE(script.submissions.empty());
    EXPECT_EQ(slab.count, 0u);

    slab.error_code = SIMPLER_ERROR_NONE;
    ctx.completion_capacity = MAX_COMPLETIONS_PER_TASK;
    sq_head[6] = 15;
    ASSERT_FALSE(submit_batch(requests, std::size(requests), 1));
    expect_accounted_error(SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
    EXPECT_TRUE(script.submissions.empty());
    EXPECT_EQ(slab.count, 0u);
}

TEST_F(A5UrmaBatchSubmission, PreexistingErrorKeepsItsOriginalDmaStatus) {
    const Request request = put();
    slab.error_code = SIMPLER_ERROR_ASYNC_WAIT_OVERFLOW;
    EXPECT_FALSE(submit_batch(&request, 1));
    EXPECT_EQ(slab.error_code, SIMPLER_ERROR_ASYNC_WAIT_OVERFLOW);
    EXPECT_TRUE(script.submissions.empty());
}

TEST_F(A5UrmaBatchSubmission, RejectsPeerWorkspaceOrOperationChangeBeforeFirstPost) {
    Request requests[]{put(0), put(32)};
    requests[1].remote_rank = 0;
    EXPECT_FALSE(submit_batch(requests, std::size(requests), 1));
    slab.error_code = SIMPLER_ERROR_NONE;
    requests[1] = put(32);
    requests[1].workspace = nullptr;
    EXPECT_FALSE(submit_batch(requests, std::size(requests), 1));
    slab.error_code = SIMPLER_ERROR_NONE;
    requests[1] = put(32);
    requests[1].op = UrmaOp::TGET;
    EXPECT_FALSE(submit_batch(requests, std::size(requests), 1));
    expect_accounted_error(SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
    EXPECT_TRUE(script.submissions.empty());
}

TEST_F(A5UrmaBatchSubmission, RejectsUnregisteredSourceOrDestinationBeforeAnyPost) {
    Request requests[]{put(0), put(32)};
    requests[1].src.address.bytes = kSrcBase + domain.winSize - 1;
    EXPECT_FALSE(submit_batch(requests, std::size(requests), 1));
    slab.error_code = SIMPLER_ERROR_NONE;
    requests[1] = put(32);
    requests[1].dst.address.bytes = kPeerMappedBase;
    EXPECT_FALSE(submit_batch(requests, std::size(requests), 1));
    slab.error_code = SIMPLER_ERROR_NONE;
    requests[1] = put(32);
    peer_mrs[info.rowsPerPeer].len = 63;
    EXPECT_FALSE(submit_batch(requests, std::size(requests), 1));
    expect_accounted_error(SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
    EXPECT_TRUE(script.submissions.empty());
    EXPECT_EQ(slab.count, 0u);
}

TEST_F(A5UrmaBatchSubmission, RejectsDomainsWithoutAnOrdinaryDevicePeerMapping) {
    const Request request = put();
    domain.windowsIn[1] = 0;
    EXPECT_FALSE(submit_batch(&request, 1));
    expect_accounted_error(SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
    EXPECT_TRUE(script.submissions.empty());
    EXPECT_EQ(slab.count, 0u);
}

TEST_F(A5UrmaBatchSubmission, InvalidIsaExtentInLaterRequestRejectsBeforeFirstPost) {
    const Request requests[]{put(0), put(32, int64_t{INT32_MAX} + 1)};
    EXPECT_FALSE(submit_batch(requests, std::size(requests), 1));
    expect_accounted_error(SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
    EXPECT_TRUE(script.submissions.empty());
    EXPECT_EQ(slab.count, 0u);
}

TEST_F(A5UrmaBatchSubmission, EmptyAndZeroLengthListsDoNotBuildOrPublish) {
    const Request requests[]{put(0, 0), put(32, 0)};
    EXPECT_TRUE(submit_batch(static_cast<const Request *>(nullptr), 0));
    EXPECT_TRUE(submit_batch(requests, std::size(requests)));
    EXPECT_EQ(script.activation_calls, 0u);
    EXPECT_TRUE(script.submissions.empty());
    EXPECT_EQ(slab.count, 0u);
}

TEST_F(A5UrmaBatchSubmission, FailureAfterThresholdRetainsPublishedTargetsAndDropsStagedTail) {
    const Request requests[]{put(0), put(32), put(64), put(96)};
    script.fail_batch_post = 3;
    ASSERT_FALSE(submit_batch(requests, std::size(requests), 2));
    expect_accounted_error(SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
    EXPECT_EQ(script.submissions.size(), 3u);
    EXPECT_EQ(script.doorbells, (std::vector<uint32_t>{6, 7}));
    ASSERT_EQ(slab.count, 2u);
    expect_target(0, 6, 1, 1);
    expect_target(1, 7, 1, 1);
}

TEST_F(A5UrmaBatchSubmission, RejectedTailFlushRetainsOnlyPreviouslyPublishedTargets) {
    const Request requests[]{put(0), put(32), put(64)};
    script.fail_batch_post = 3;
    ASSERT_FALSE(submit_batch(requests, std::size(requests), 2));
    expect_accounted_error(SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
    EXPECT_EQ(script.doorbells, (std::vector<uint32_t>{6, 7}));
    ASSERT_EQ(slab.count, 2u);
    expect_target(0, 6, 1, 1);
    expect_target(1, 7, 1, 1);
}

TEST_F(A5UrmaBatchSubmission, MalformedFlushEventCannotSupplyCompletionProof) {
    const Request request = put();
    script.invalid_flush_event = true;
    ASSERT_FALSE(submit_batch(&request, 1));
    EXPECT_EQ(slab.error_code, SIMPLER_ERROR_ASYNC_REGISTRATION_FAILED);
    EXPECT_EQ(script.doorbells, (std::vector<uint32_t>{6}));
    EXPECT_EQ(slab.count, 0u);
}

TEST_F(A5UrmaBatchSubmission, FailedBatchFinLeavesDmaForAbortDespitePendingNotification) {
    const Request requests[]{put(0), put(32), put(64), put(96)};
    script.fail_batch_post = 3;
    ASSERT_FALSE(submit_batch(requests, std::size(requests), 2));
    ASSERT_EQ(slab.count, 2u);
    uint32_t notification = 0;
    ASSERT_TRUE(register_completion_condition(
        ctx, {reinterpret_cast<uint64_t>(&notification), 1, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_COUNTER, 0}
    ));

    auto context = std::make_unique<SchedulerContext>();
    auto runtime = std::make_unique<RuntimeContext>();
    auto header = std::make_unique<SharedMemoryHeader>();
    auto mailbox = std::make_unique<AICoreCompletionMailbox>();
    runtime->aicore_mailbox = mailbox.get();
    runtime->scheduler.sm_header = header.get();
    runtime->orchestrator.sm_header = header.get();
    runtime_finalize_after_wire(runtime.get(), 1, 1);
    header->sched_error_code.store(SIMPLER_ERROR_NONE);
    ChipTaskSlotState producer{};
    TaskPayload payload{};
    TaskDescriptor task{};
    TaskId::assign(task.task_id, ctx.task_token);
    producer.task = &task;
    producer.payload = &payload;
    producer.total_required_subtasks = 1;
    producer.task_state.store(CHIP_TASK_PENDING);
    int32_t completed = 0, release_count = 0;
    ChipTaskSlotState *release[1]{};
    EXPECT_FALSE(
        SchedulerRetirementTestPeer::finish_failed_submission(
            *context, runtime->scheduler, slab, producer, completed, release, release_count, runtime.get()
        )
    );

    EXPECT_TRUE(context->is_completed());
    uint32_t bucket_epochs[2]{};
    runtime->orchestrator.tensor_map.num_buckets = 2;
    runtime->orchestrator.tensor_map.bucket_epochs = bucket_epochs;
    uint32_t value = 42;
    const uint32_t shape[]{1};
    const uint32_t index[]{0};
    const auto tensor = simpler::tmr::make_tensor_external(&value, shape, 1, DataType::INT32);
    EXPECT_EQ(get_tensor_data(runtime.get(), tensor, 1, index), 0u);
    EXPECT_TRUE(runtime->ops->is_fatal(runtime.get()));
    EXPECT_EQ(header->orch_error_code.load(), SIMPLER_ERROR_NONE);
    EXPECT_EQ(header->sched_error_code.load(), SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
    EXPECT_EQ(context->accounted_urma_submit_error(), SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
    EXPECT_EQ(SchedulerRetirementTestPeer::preserved_count(*context), slab.count);
    EXPECT_EQ(producer.completed_subtasks.load(), 0);
    EXPECT_EQ(completed, 0);
    EXPECT_EQ(release_count, 0);
    EXPECT_FALSE(mailbox->has_pending());

    for (uint32_t row : {6u, 7u}) {
        *reinterpret_cast<uint32_t *>(cq_storage[row].data()) = 1u << 2;
    }
    EXPECT_TRUE(context->settle_aborted_dma());
    EXPECT_EQ(notification, 0u);
    EXPECT_EQ(cq_tail[6], 1u);
    EXPECT_EQ(cq_tail[7], 1u);
}

TEST_F(A5UrmaBatchSubmission, NoTokenFailureKeepsAccountedProvenanceOnlyWhenItsLatchWins) {
    const Request request = put();
    ASSERT_FALSE(submit_batch(&request, 1, 0));
    ASSERT_EQ(slab.count, 0u);

    for (bool earlier_same_code : {false, true}) {
        auto context = std::make_unique<SchedulerContext>();
        auto runtime = std::make_unique<RuntimeContext>();
        auto header = std::make_unique<SharedMemoryHeader>();
        runtime->scheduler.sm_header = header.get();
        header->sched_error_code.store(earlier_same_code ? SIMPLER_ERROR_ASYNC_COMPLETION_INVALID : SIMPLER_ERROR_NONE);
        header->sched_error_thread.store(earlier_same_code ? 1 : -1);
        header->sched_error_bitmap.store(earlier_same_code ? 2u : 0u);
        ChipTaskSlotState producer{};
        TaskPayload payload{};
        producer.payload = &payload;
        int32_t completed = 0, release_count = 0;
        ChipTaskSlotState *release[1]{};

        EXPECT_FALSE(
            SchedulerRetirementTestPeer::finish_failed_submission(
                *context, runtime->scheduler, slab, producer, completed, release, release_count, runtime.get()
            )
        );
        EXPECT_EQ(header->sched_error_code.load(), SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        EXPECT_EQ(header->sched_error_thread.load(), earlier_same_code ? 1 : 0);
        EXPECT_EQ(header->sched_error_bitmap.load(), earlier_same_code ? 3u : 1u);
        EXPECT_EQ(
            context->accounted_urma_submit_error(),
            earlier_same_code ? SIMPLER_ERROR_NONE : SIMPLER_ERROR_ASYNC_COMPLETION_INVALID
        );
    }
}

TEST_F(A5UrmaBatchSubmission, FullMailboxAtAbortPreservesUnharvestedSlab) {
    const Request requests[]{put(0), put(32)};
    ASSERT_TRUE(submit_batch(requests, std::size(requests)));
    ASSERT_EQ(slab.count, 2u);

    auto context = std::make_unique<SchedulerContext>();
    auto runtime = std::make_unique<RuntimeContext>();
    auto mailbox = std::make_unique<AICoreCompletionMailbox>();
    mailbox->head.store(AICORE_COMPLETION_MAILBOX_CAPACITY);
    runtime->aicore_mailbox = mailbox.get();
    ChipTaskSlotState producer{};
    TaskPayload payload{};
    TaskDescriptor task{};
    TaskId::assign(task.task_id, ctx.task_token);
    producer.task = &task;
    producer.payload = &payload;
    producer.total_required_subtasks = 1;
    int32_t completed = 0, release_count = 0;
    ChipTaskSlotState *release[1]{};

    EXPECT_FALSE(
        SchedulerRetirementTestPeer::finish_failed_submission(
            *context, runtime->scheduler, slab, producer, completed, release, release_count, runtime.get(), true
        )
    );
    EXPECT_EQ(SchedulerRetirementTestPeer::preserved_count(*context), slab.count);
    EXPECT_EQ(producer.completed_subtasks.load(), 0);
    EXPECT_EQ(completed, 0);
    EXPECT_EQ(release_count, 0);
}
