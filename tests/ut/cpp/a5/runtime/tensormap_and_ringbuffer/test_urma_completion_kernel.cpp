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
}

namespace {
uint32_t total_wait_calls = 0;
struct TestSession {
    bool valid{true};
    pto::comm::DmaEngine engine{static_cast<pto::comm::DmaEngine>(1)};
    uint8_t *contextGm{};
    uint32_t destRankId{1};
    uint32_t qpIdxBase{2};
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

#include "backend/urma/urma_completion_kernel.h"
#include "scheduler/scheduler_context.h"
#include <pto/comm/async/urma/urma_types.hpp>

namespace {
struct Submission {
    bool put;
    uint64_t dst;
    uint64_t src;
    std::array<int64_t, 5> shape;
    std::array<int64_t, 5> stride;
};
struct PostScript {
    std::vector<TestEvent> events;
    std::vector<Submission> submissions;
    uint32_t activation_calls{};
    bool post_from_queue{};
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
template <typename Dst, typename Src>
AsyncEvent post_for_test(bool put, Dst dst, Src src, const AsyncSession &session) {
    auto &script = *active_script;
    script.submissions.push_back({put, dst.data().bytes, src.data().bytes, dst.shape.values, dst.stride.values});
    EXPECT_EQ(src.shape.values, dst.shape.values);
    EXPECT_EQ(src.stride.values, dst.stride.values);
    if (script.post_from_queue) {
        auto *info = reinterpret_cast<urma::UrmaInfo *>(session.contextGm);
        auto *queues = reinterpret_cast<urma::UrmaWQCtx *>(info->sqPtr);
        TestEvent event{};
        event.urmaJettyBase = session.qpIdxBase;
        event.urmaJettyCount = session.cfg.urma.qpCount;
        const uint64_t bytes = static_cast<uint64_t>(src.shape.values[4]) * sizeof(typename Src::RawDType);
        const uint64_t wqes = pto2::urma_backend::chunk_count(bytes);
        for (uint32_t j = 0; j < event.urmaJettyCount; ++j) {
            uint32_t row = 0;
            EXPECT_TRUE(
                pto2::urma_backend::queue_row_for(session.contextGm, session.destRankId, session.qpIdxBase + j, row)
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
    static void finish_failed_submission(
        SchedulerContext &context, SchedulerState &scheduler, const DeferredCompletionSlab &slab,
        ChipTaskSlotState &producer, int32_t &completed, ChipTaskSlotState **release, int32_t &release_count
    ) {
        context.sched_ = &scheduler;
        context.deferred_slab_per_core_[0][0] = slab;
        context.complete_slot_task(producer, 0, SubtaskSlot::AIV0, 0, 0, nullptr, completed, release, release_count);
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

    bool register_event() { return pto2::detail::register_urma_async_event(ctx, event, session, workspace()); }
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

TEST(A5UrmaCompletionKernel, CapacityBudgetAccountsForAllChunksAndJetties) {
    RegistrationFixture fixture;
    fixture.ctx.completion_capacity = 8;
    fixture.slab.count = 3;

    EXPECT_TRUE(pto2::urma_backend::completion_capacity_available(fixture.ctx, 2u * 2u));
    EXPECT_FALSE(pto2::urma_backend::completion_capacity_available(fixture.ctx, 3u * 2u));
    EXPECT_EQ(fixture.slab.count, 3u);
}

TEST(A5UrmaCompletionKernel, InvalidPublishedCountCannotUnderflowCapacity) {
    RegistrationFixture fixture;
    fixture.ctx.completion_capacity = 2;
    fixture.slab.count = 3;

    EXPECT_FALSE(pto2::urma_backend::completion_capacity_available(fixture.ctx, 1));
    EXPECT_FALSE(fixture.register_event());
    EXPECT_EQ(fixture.slab.count, 3u);
    EXPECT_EQ(fixture.event.wait_calls, 0u);
}

TEST(A5UrmaCompletionKernel, FailedContextCannotAdmitMoreWork) {
    RegistrationFixture fixture;
    fixture.slab.error_code = SIMPLER_ERROR_ASYNC_COMPLETION_INVALID;

    EXPECT_FALSE(pto2::urma_backend::completion_capacity_available(fixture.ctx, 1));
    EXPECT_FALSE(fixture.register_event());
    EXPECT_EQ(fixture.slab.count, 0u);
    EXPECT_EQ(fixture.slab.error_code, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
}

TEST(A5UrmaCompletionKernel, ChunkBudgetHandlesEmptyBoundaryAndMaximumSize) {
    constexpr uint64_t limit = pto::comm::urma::kUrmaMaxWqeTransferBytes;
    EXPECT_EQ(pto2::urma_backend::chunk_count(0), 0u);
    EXPECT_EQ(pto2::urma_backend::chunk_count(limit), 1u);
    EXPECT_EQ(pto2::urma_backend::chunk_count(limit + 1), 2u);
    EXPECT_EQ(pto2::urma_backend::chunk_count(UINT64_MAX), 1u + (UINT64_MAX - 1u) / limit);
}

TEST(A5UrmaCompletionKernel, OversizedChunkBudgetFailsBeforeQueueAccess) {
    RegistrationFixture fixture;

    EXPECT_FALSE(pto2::urma_backend::preflight_urma_request(fixture.ctx, fixture.session, UINT64_MAX));
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

    EXPECT_TRUE(pto2::urma_backend::preflight_urma_request(fixture.ctx, fixture.session, 3));
    EXPECT_EQ(fixture.slab.count, 0u);
    EXPECT_EQ(fixture.slab.error_code, SIMPLER_ERROR_NONE);
    EXPECT_EQ(fixture.sq_head[6], 0u);
    EXPECT_EQ(fixture.sq[6].submittedWqeCount, 0u);
}

TEST(A5UrmaCompletionKernel, FullSqInOneJettyRejectsWholeRequest) {
    RegistrationFixture fixture;
    fixture.sq_head[6] = 16;

    EXPECT_FALSE(pto2::urma_backend::preflight_urma_request(fixture.ctx, fixture.session, 1));
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

    EXPECT_FALSE(pto2::urma_backend::preflight_urma_request(fixture.ctx, fixture.session, 1));
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

    EXPECT_FALSE(pto2::urma_backend::preflight_urma_request(fixture.ctx, fixture.session, 3));
    EXPECT_EQ(fixture.sq_head[6], 14u);
    EXPECT_EQ(fixture.sq[6].submittedWqeCount, 14u);
    EXPECT_EQ(fixture.slab.count, 0u);
}

TEST(A5UrmaCompletionKernel, ExactRemainingRingCapacityCanBeReserved) {
    RegistrationFixture fixture;
    fixture.sq_head[6] = 14;
    fixture.sq[6].submittedWqeCount = 14;

    EXPECT_TRUE(pto2::urma_backend::preflight_urma_request(fixture.ctx, fixture.session, 2));
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

    EXPECT_TRUE(pto2::urma_backend::preflight_urma_request(fixture.ctx, fixture.session, 3));
    EXPECT_EQ(fixture.slab.error_code, SIMPLER_ERROR_NONE);
    EXPECT_EQ(fixture.sq_tail[6], UINT32_MAX);
    EXPECT_EQ(fixture.cq_tail[6], UINT32_MAX);
}

TEST(A5UrmaCompletionKernel, CompletionBudgetRejectsBeforeReadingQueueTables) {
    RegistrationFixture fixture;
    fixture.ctx.completion_capacity = 5;
    fixture.info.sqPtr = 0;
    fixture.info.scqPtr = 0;

    EXPECT_FALSE(pto2::urma_backend::preflight_urma_request(fixture.ctx, fixture.session, 3));
    EXPECT_EQ(fixture.slab.error_code, SIMPLER_ERROR_ASYNC_REGISTRATION_FAILED);
    EXPECT_EQ(fixture.slab.count, 0u);
}

TEST(A5UrmaCompletionKernel, PreflightRejectsMalformedRingDepth) {
    RegistrationFixture fixture;
    fixture.cq[7].depth = 15;

    EXPECT_FALSE(pto2::urma_backend::preflight_urma_request(fixture.ctx, fixture.session, 1));
    EXPECT_EQ(fixture.slab.error_code, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
    EXPECT_EQ(fixture.sq_head[6], 0u);
    EXPECT_EQ(fixture.sq_head[7], 0u);
    EXPECT_EQ(fixture.slab.count, 0u);
}

TEST(A5UrmaCompletionKernel, PreflightRejectsJettyOutsideWorkspace) {
    RegistrationFixture fixture;
    fixture.session.qpIdxBase = 3;

    EXPECT_FALSE(pto2::urma_backend::preflight_urma_request(fixture.ctx, fixture.session, 1));
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
        return pto2::urma_backend::build_urma_session(ctx, workspace(), 1, bound);
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
        ASSERT_EQ(slab.count, 2u);

        AsyncCtx next = AsyncCtx::make(TaskId::make(0, 99), &next_slab);
        next.urma_sender_id = ctx.urma_sender_id;
        EXPECT_NE(TaskId::to_uint64(next.task_token), TaskId::to_uint64(ctx.task_token));
        ASSERT_TRUE(send_request_entry(next, UrmaTput(dst, src, workspace(), 1)));
        ASSERT_EQ(next_slab.count, 2u);
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
    ASSERT_EQ(next_slab.count, 2u);
    set_cqe(14, false);
    set_cqe(15, true);
    set_cqe(16, true);
    EXPECT_EQ(poll(next_slab.entries[0]).state, CompletionPollState::PENDING);
    EXPECT_EQ(sq_tail[first_row()], 14u);
    EXPECT_EQ(cq_tail[first_row()], 14u);

    set_cqe(14, true);
    EXPECT_EQ(poll(next_slab.entries[0]).state, CompletionPollState::READY);
    EXPECT_EQ(poll(next_slab.entries[1]).state, CompletionPollState::READY);
    EXPECT_EQ(poll(slab.entries[0]).state, CompletionPollState::READY);
    EXPECT_EQ(poll(slab.entries[1]).state, CompletionPollState::READY);
    EXPECT_EQ(sq_tail[first_row()], 17u);
    EXPECT_EQ(cq_tail[first_row()], 17u);
    EXPECT_EQ(cq_doorbell[first_row()], 17u);
    EXPECT_EQ(sq_head[first_row()], 17u);
    EXPECT_EQ(sq[first_row()].submittedWqeCount, 17u);
}

TEST_P(A5UrmaSessionBinding, CrossTaskReuseCannotHideAnOlderRequestError) {
    DeferredCompletionSlab next_slab{};
    post_two_unretired_tasks(next_slab);
    ASSERT_EQ(next_slab.count, 2u);
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
        ASSERT_TRUE(pto2::urma_backend::build_urma_session(loaded, workspace(), 1, bound[lane]));
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
    ASSERT_TRUE(pto2::urma_backend::build_urma_session(loaded, workspace(), 1, bound[1]));
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
    ASSERT_TRUE(pto2::urma_backend::build_urma_session(ctx, workspace(), 0, next));
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
    EXPECT_FALSE(pto2::urma_backend::build_urma_session(ctx, nullptr, 1, bound));
    EXPECT_FALSE(bound.valid);
    EXPECT_FALSE(pto2::urma_backend::build_urma_session(ctx, workspace(), info.rankCount, bound));
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
        using Shape = pto::Shape<1, 1, 1, 1, pto::DYNAMIC>;
        using Stride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;
        using Tensor = pto::GlobalTensor<T, Shape, Stride>;
        Tensor dst({kDstBase}, Shape(1, 1, 1, 1, elements), Stride(elements, elements, elements, elements, 1));
        Tensor src({kSrcBase}, Shape(1, 1, 1, 1, elements), Stride(elements, elements, elements, elements, 1));
        return pto2::urma_backend::submit_chunked_urma_request(
            ctx, UrmaRequestDescriptor<Tensor, Tensor>{GetParam(), dst, src, workspace(), session.destRankId}
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
    EXPECT_EQ(slab.count, 6u);
    EXPECT_EQ(slab.error_code, SIMPLER_ERROR_NONE);
    EXPECT_EQ(total_wait_calls, 0u);
}

TEST_P(A5UrmaChunkedSubmission, SecondChunkPartialFailureRetainsTargetsAndBlocksConsumer) {
    script.events[1].postFailed = true;
    script.events[1].urmaJettyCount = 1;
    ASSERT_FALSE(submit());
    ASSERT_EQ(script.submissions.size(), 2u);
    ASSERT_EQ(slab.count, 3u);
    EXPECT_EQ(slab.entries[0].addr, (uint64_t{6} << 32) | 4);
    EXPECT_EQ(slab.entries[0].expected_value, 3u);
    EXPECT_EQ(slab.entries[1].addr, (uint64_t{7} << 32) | 9);
    EXPECT_EQ(slab.entries[1].expected_value, 7u);
    EXPECT_EQ(slab.entries[2].addr, (uint64_t{6} << 32) | 5);
    EXPECT_EQ(slab.entries[2].expected_value, 4u);
    for (uint32_t i = 0; i < slab.count; ++i)
        EXPECT_EQ(slab.entries[i].backend_cookie, reinterpret_cast<uint64_t>(workspace()));
    EXPECT_EQ(slab.error_code, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
    pto2::detail::set_urma_error(ctx, SIMPLER_ERROR_ASYNC_REGISTRATION_FAILED);
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
    ASSERT_EQ(slab.count, 2u);
    EXPECT_EQ(slab.entries[0].addr, (uint64_t{6} << 32) | 4);
    EXPECT_EQ(slab.entries[1].expected_value, 7u);
    EXPECT_EQ(slab.error_code, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
    expect_consumer_blocked();
}

TEST_P(A5UrmaChunkedSubmission, WholeRequestBudgetRejectsBeforeFirstPost) {
    ctx.completion_capacity = 5;
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

INSTANTIATE_TEST_SUITE_P(GetAndPut, A5UrmaChunkedSubmission, testing::Values(UrmaOp::TGET, UrmaOp::TPUT));
