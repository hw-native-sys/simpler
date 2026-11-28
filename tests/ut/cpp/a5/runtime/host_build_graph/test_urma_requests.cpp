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

#include <cstring>

#include "inner_kernel.h"
#include "runtime_types.h"
#include "async_context.h"
#include "../urma_queue_fixture.h"
#include "../urma_request_fixture.h"
#include "backend/urma/urma_completion_kernel.h"

namespace {

struct RunImage {
    simpler::percounter::PercounterControl control{};
    pto2::resident::AsyncGroupData group{};
    alignas(128) DeferredCompletionEntry entries[MAX_COMPLETIONS_PER_TASK]{};
};

class HbgUrmaRequest : public testing::TestWithParam<UrmaOp> {
protected:
    static constexpr uint32_t row = 4;  // First row for peer 1, sender 0.
    static constexpr uint64_t limit = pto::comm::urma::kUrmaMaxWqeTransferBytes;
    struct alignas(pto::comm::urma::kCacheLineSize) Header {
        pto::comm::urma::UrmaInfo info;
    };
    static_assert(sizeof(Header) == simpler::urma::sender_view_offset(sizeof(Header::info), 0));
    RunImage image{};
    simpler::test::UrmaQueueFixture queues;
    alignas(128) std::array<Header, 4> headers{};
    UrmaRowOwner owners[simpler::test::kRows]{};
    CommWorkspace domain{reinterpret_cast<uint64_t>(&headers[0].info), sizeof(queues.info)};
    UrmaRunBinding binding{domain.addr, domain.size, reinterpret_cast<uint64_t>(owners), 0, row, 2, 2};
    AsyncCtx ctx{};
    PostScript script{};

    void SetUp() override {
        ctx.urma_sender_id = 0;
        ctx.resident_enabled = 1;
        ctx.resident = {reinterpret_cast<uint8_t *>(&image), sizeof(image), 1, offsetof(RunImage, group), 0};
        image.group.run_epoch = ctx.resident.run_epoch;
        image.group.capacity = MAX_COMPLETIONS_PER_TASK;
        image.group.entries_offset = offsetof(RunImage, entries);
        image.group.entries_bytes = pto2::resident::async_group_entry_bytes(image.group.capacity);
        image.control.urma = {reinterpret_cast<uint64_t>(&binding), 1};
        headers[0].info = queues.info;
        headers[0].info.gatherRuntimeBase = queues.addr();
        headers[0].info.gatherRuntimeStride = 128;
        native_block_idx = 0;
        script.post_from_queue = true;
        active_script = &script;
    }

    void TearDown() override { active_script = nullptr; }

    bool send(
        uint64_t bytes, const CommWorkspace &workspace, simpler::urma::UrmaSession *session = nullptr, bool raw = false
    ) {
        using Shape = pto::Shape<1, 1, 1, 1, pto::DYNAMIC>;
        using Stride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;
        using Tensor = pto::GlobalTensor<uint8_t, Shape, Stride>;
        const Shape shape(1, 1, 1, 1, bytes);
        const Stride stride(bytes, bytes, bytes, bytes, 1);
        const Tensor dst({0x10000}, shape, stride), src({0x20000}, shape, stride);
        auto request = GetParam() == UrmaOp::TGET ? UrmaTget(dst, src, workspace, 1) : UrmaTput(dst, src, workspace, 1);
        if (raw) request.domain = nullptr;
        return session ? send_request_entry(ctx, request, *session) : send_request_entry(ctx, request);
    }

    void expect_rejected(uint64_t bytes, const CommWorkspace &workspace, int32_t error) {
        const auto heads = queues.sq_head;
        const auto queues_before = queues.sq;
        const size_t posts = script.submissions.size();
        const uint32_t count = image.group.count;
        EXPECT_FALSE(send(bytes, workspace));
        EXPECT_EQ(int32_t{image.group.error_code}, error);
        EXPECT_EQ(uint32_t{image.group.count}, count);
        EXPECT_EQ(queues.sq_head, heads);
        for (uint32_t i = 0; i < queues.sq.size(); ++i)
            EXPECT_EQ(queues.sq[i].submittedWqeCount, queues_before[i].submittedWqeCount);
        EXPECT_EQ(script.submissions.size(), posts);
    }

    void expect_target(uint32_t slot, uint32_t queue, uint32_t bb, uint32_t cqe) {
        const auto &entry = image.entries[slot];
        EXPECT_EQ(entry.engine, COMPLETION_ENGINE_URMA);
        EXPECT_EQ(entry.completion_type, COMPLETION_TYPE_URMA_EVENT_HANDLE);
        EXPECT_EQ(entry.backend_cookie, domain.addr);
        EXPECT_EQ(entry.addr >> 32, queue);
        EXPECT_EQ(static_cast<uint32_t>(entry.addr), bb);
        EXPECT_EQ(entry.expected_value, cqe);
    }
};

TEST_P(HbgUrmaRequest, WqeLimitAndUnevenTailRegisterOnlyThePostedRows) {
    simpler::urma::UrmaSession session;
    ASSERT_TRUE(simpler::urma::build_urma_session(ctx, domain, session, 1));
    session.rt.urma.nextWqeSeq = UINT32_MAX;
    queues.sq_head[row] = queues.sq_tail[row] = UINT32_MAX;
    queues.sq[row].submittedWqeCount = queues.cq_tail[row] = 9;
    ASSERT_TRUE(send(limit, domain, &session));
    ASSERT_TRUE(send(limit + 17, domain, &session));
    ASSERT_TRUE(send(16, domain, &session));
    ASSERT_EQ(image.group.count, 4u);
    ASSERT_TRUE(send(16, domain, &session));
    ASSERT_EQ(image.group.count, 5u);
    ASSERT_EQ(image.group.reservation_limit, 0u);
    ASSERT_EQ(script.submissions.size(), 5u);
    EXPECT_EQ(script.submissions[1].shape[4], static_cast<int64_t>(limit));
    EXPECT_EQ(script.submissions[2].shape[4], 17);
    EXPECT_EQ(script.submissions[2].src, script.submissions[1].src + limit);
    EXPECT_EQ(script.submissions[2].dst, script.submissions[1].dst + limit);
    expect_target(0, row + 1, 1, 1);
    expect_target(1, row, 0, 10);
    expect_target(2, row + 1, 2, 2);
    expect_target(3, row, 1, 11);
    expect_target(4, row + 1, 3, 3);
}

TEST_P(HbgUrmaRequest, FullRequestOf64FitsAndThe65thDoesNotPost) {
    std::array<std::array<simpler::test::TestCqe, 32>, 2> ring{};
    for (uint32_t j = 0; j < ring.size(); ++j) {
        queues.sq[row + j].bufAddr = queues.cq[row + j].bufAddr = reinterpret_cast<uint64_t>(ring[j].data());
        queues.sq[row + j].depth = queues.cq[row + j].depth = ring[j].size();
    }
    ASSERT_TRUE(send(64 * limit, domain));
    ASSERT_EQ(image.group.count, 64u);
    for (uint32_t i = 0; i < 64; ++i)
        expect_target(i, row + i % 2, 1 + i / 2, 1 + i / 2);
    for (uint32_t j = 0; j < ring.size(); ++j)
        queues.sq_tail[row + j] = queues.cq_tail[row + j] = 32;
    expect_rejected(16, domain, SIMPLER_ERROR_ASYNC_WAIT_OVERFLOW);
}

TEST_P(HbgUrmaRequest, OversizedRequestAndMixedConditionsRejectBeforeTheFirstPost) {
    alignas(64) uint32_t counter = 0;
    ASSERT_TRUE(register_completion_condition(
        ctx, {reinterpret_cast<uint64_t>(&counter), 1, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_COUNTER, 0}
    ));
    expect_rejected(64 * limit, domain, SIMPLER_ERROR_ASYNC_WAIT_OVERFLOW);
}

TEST_P(HbgUrmaRequest, RequestRequiring65WqesCannotPostAPrefix) {
    expect_rejected(65 * limit, domain, SIMPLER_ERROR_ASYNC_WAIT_OVERFLOW);
}

TEST_P(HbgUrmaRequest, EmptyRequestDoesNotBuildASessionOrRegisterACondition) {
    EXPECT_TRUE(send(0, domain));
    EXPECT_TRUE(script.submissions.empty());
    EXPECT_EQ(script.activation_calls, 0U);
    EXPECT_EQ(image.group.count, 0U);
}

TEST_P(HbgUrmaRequest, QueueTargetsShareTheGroupWithAnEarlierCounterCondition) {
    alignas(64) uint32_t counter = 0;
    ASSERT_TRUE(register_completion_condition(
        ctx, {reinterpret_cast<uint64_t>(&counter), 1, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_COUNTER, 0}
    ));
    ASSERT_TRUE(send(limit + 1, domain));
    ASSERT_EQ(image.group.count, 3U);
    EXPECT_EQ(image.entries[0].addr, reinterpret_cast<uint64_t>(&counter));
    EXPECT_EQ(image.entries[0].completion_type, COMPLETION_TYPE_COUNTER);
    expect_target(1, row, 1, 1);
    expect_target(2, row + 1, 1, 1);
}

TEST_P(HbgUrmaRequest, SharedPoolWorkspaceIsRefusedBeforePosting) {
    headers[0].info.layout = pto::comm::urma::UrmaLayout::SHARED_POOL;
    expect_rejected(16, domain, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
}

TEST_P(HbgUrmaRequest, LogicalBlockUsesThePhysicalSendersDomainHeaderEvenWithARawWorkspace) {
    native_block_idx = 1;
    ASSERT_TRUE(send(32, domain, nullptr, true));
    ASSERT_EQ(script.submissions.size(), 1u);
    EXPECT_EQ(script.submissions[0].workspace, reinterpret_cast<const uint8_t *>(&headers[1].info));
    EXPECT_EQ(
        headers[1].info.gatherRuntimeBase + headers[1].info.gatherRuntimeStride, headers[0].info.gatherRuntimeBase
    );
    expect_target(0, row, 1, 1);
}

TEST_P(HbgUrmaRequest, PartialSubmissionPreservesEveryReturnedTargetBeforeReportingTheError) {
    script.post_from_queue = false;
    TestEvent first{}, second{};
    first.urmaJettyBase = 0;
    first.urmaJettyCount = 1;
    second.urmaJettyBase = 1;
    second.urmaJettyCount = 1;
    second.postFailed = true;
    script.events = {first, second};
    EXPECT_FALSE(send(3 * limit, domain));
    EXPECT_EQ(image.group.error_code, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
    ASSERT_EQ(image.group.count, 2u);
    EXPECT_EQ(image.group.reservation_limit, 0u);
    EXPECT_EQ(script.submissions.size(), 2u);
    expect_target(0, row, first.urmaTargetBbPerJetty[0], first.urmaTargetCqePerJetty[0]);
    expect_target(1, row + 1, second.urmaTargetBbPerJetty[0], second.urmaTargetCqePerJetty[0]);
}

TEST_P(HbgUrmaRequest, SingleGatherOrScatterUsesTheInlineEndCqeAndFinishesItsReservation) {
    using namespace pto::comm;
    TestEvent event{};
    event.urmaJettyCount = urma::kUrmaGatherInlineEvent;
    event.urmaJettyBase = 0;
    event.urmaTargetCqe = 12;
    event.urmaTargetCqePerJetty[0] = 11;
    event.urmaTargetBbPerJetty[0] = 19;
    script.events.push_back(event);
    uint8_t remote_data[32]{}, local_data[32]{};
    if (GetParam() == UrmaOp::TPUT) {
        GatherDestination remote{remote_data, 1, 1};
        GatherSegment local{local_data, 32};
        ASSERT_TRUE(send_request_entry(ctx, UrmaTput(&remote, 1, &local, 1), domain));
    } else {
        ScatterSource remote{remote_data, 1, 1};
        ScatterSegment local{local_data, 32};
        simpler::urma::UrmaSession session;
        ASSERT_TRUE(simpler::urma::build_urma_session(ctx, domain, session, 1));
        ASSERT_TRUE(send_request_entry(ctx, UrmaTget(&local, 1, &remote, 1), session));
    }
    ASSERT_EQ(image.group.count, 1u);
    EXPECT_EQ(image.group.reservation_limit, 0u);
    expect_target(0, row, 19, 12);
}

TEST_P(HbgUrmaRequest, FullSecondQueuePreventsPostingOnFirstQueue) {
    queues.sq_head[row + 1] = queues.sq[row + 1].depth;
    expect_rejected(limit + 1, domain, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
}

TEST_P(HbgUrmaRequest, ForeignWorkspaceIsAnInvalidRequest) {
    CommWorkspace other = domain;
    other.addr = reinterpret_cast<uint64_t>(owners);
    expect_rejected(16, other, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
}

TEST_P(HbgUrmaRequest, PublicSessionRejectsBadBindingsBeforeReadingOrWritingSenderViews) {
    simpler::test::UrmaQueueFixture other;
    alignas(128) std::array<Header, 4> foreign{};
    foreign[0].info = other.info;
    foreign[0].info.gatherRuntimeBase = other.addr();
    foreign[0].info.gatherRuntimeStride = 128;
    const CommWorkspace foreign_domain{reinterpret_cast<uint64_t>(foreign.data()), domain.size};
    // The forged larger size would put the ISA header one slot later, still inside this CPU fixture.
    const CommWorkspace wrong_size{domain.addr, domain.size + sizeof(Header)};
    std::array<uint8_t, sizeof(headers)> original{}, other_original{};
    std::memcpy(original.data(), headers.data(), sizeof(headers));
    std::memcpy(other_original.data(), foreign.data(), sizeof(foreign));
    simpler::urma::UrmaSession session;
    ASSERT_TRUE(simpler::urma::build_urma_session(ctx, domain, session, 1));
    const auto activations = script.activation_calls;
    const auto reject = [&](const CommWorkspace &ws, uint32_t peer) {
        EXPECT_FALSE(simpler::urma::build_urma_session(ctx, ws, session, peer));
        EXPECT_FALSE(session.valid);
        EXPECT_EQ(session.contextGm, nullptr);
        EXPECT_EQ(session.domain_workspace, nullptr);
        EXPECT_EQ(script.activation_calls, activations);
        EXPECT_EQ(std::memcmp(headers.data(), original.data(), sizeof(headers)), 0);
        EXPECT_EQ(std::memcmp(foreign.data(), other_original.data(), sizeof(foreign)), 0);
    };
    for (uint32_t sender = 0; sender < 2; ++sender) {
        ctx.urma_sender_id = sender;
        native_block_idx = 1 - sender;
        reject(foreign_domain, 1);
        reject(wrong_size, 1);
        reject(domain, binding.rank_count);
    }
    ctx.urma_sender_id = 2;  // Two physical senders in this domain.
    reject(domain, 1);
    EXPECT_FALSE(send(16, domain, &session));
    EXPECT_TRUE(script.submissions.empty());
    EXPECT_EQ(image.group.count, 0U);
    for (uint32_t i = 0; i < queues.sq.size(); ++i) {
        EXPECT_EQ(queues.sq_head[i], 0U);
        EXPECT_EQ(other.sq_head[i], 0U);
        EXPECT_EQ(queues.sq[i].submittedWqeCount, 0U);
        EXPECT_EQ(other.sq[i].submittedWqeCount, 0U);
    }
}

TEST_P(HbgUrmaRequest, PublicSessionMayChooseItsPeerAtPostTimeAndUsesTheRegisteredSenderView) {
    ctx.urma_sender_id = 1;
    native_block_idx = 0;
    simpler::urma::UrmaSession session;
    ASSERT_TRUE(simpler::urma::build_urma_session(ctx, domain, session));
    EXPECT_EQ(session.contextGm, reinterpret_cast<uint8_t *>(&headers[2].info));
    ASSERT_TRUE(send(16, domain, &session));
    ASSERT_EQ(script.submissions.size(), 1U);
    EXPECT_EQ(queues.sq_head[row + 2], 1U);
    expect_target(0, row + 2, 1, 1);
}

INSTANTIATE_TEST_SUITE_P(GetAndPut, HbgUrmaRequest, testing::Values(UrmaOp::TGET, UrmaOp::TPUT));

}  // namespace
