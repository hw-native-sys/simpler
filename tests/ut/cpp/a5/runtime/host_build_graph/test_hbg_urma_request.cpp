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

#include "helpers/urma_request_test.h"
#include "backend/urma/urma_request_adapter.h"

namespace {
struct HbgUrmaFixture {
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
    alignas(128) std::array<uint8_t, 8192> image{};
    pto2::resident::AsyncWriter writer{image.data(), image.size(), 91, 128, 7};
    simpler::urma::HbgRequestAdapter runtime{writer, 1};
    PostScript script{};
    pto2::resident::AsyncGroupData &group() {
        return *reinterpret_cast<pto2::resident::AsyncGroupData *>(image.data() + 128);
    }
    DeferredCompletionEntry &entry(uint32_t i) {
        return reinterpret_cast<DeferredCompletionEntry *>(image.data() + 4096)[i];
    }
    TestSession session{};
    TestEvent event{};

    HbgUrmaFixture() {
        auto &g = group();
        g.run_epoch = 91;
        g.task_ref = 11;
        g.group_index = 7;
        g.subtask_slot = 1;
        g.capacity = 4;
        g.entries_offset = 4096;
        g.entries_bytes = pto2::resident::async_group_entry_bytes(4);
        active_script = &script;
        script.post_from_queue = true;
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

    bool submit(UrmaOp op, uint64_t chunks = 3) {
        using S = pto::Shape<1, 1, 1, 1, pto::DYNAMIC>;
        using T = pto::GlobalTensor<float, S, S>;
        const auto n = chunks * pto::comm::urma::kUrmaMaxWqeTransferBytes / sizeof(float);
        T dst({0x100000}, S(1, 1, 1, 1, n), S(n, n, n, n, 1));
        T src({0x200000}, S(1, 1, 1, 1, n), S(n, n, n, n, 1));
        return simpler::urma::submit_chunked_urma_request(
            runtime, UrmaRequestDescriptor<T, T>{op, dst, src, workspace(), 1}
        );
    }
    bool register_event() { return simpler::urma::register_urma_async_event(runtime, event, session, workspace()); }
};

}  // namespace

TEST(HbgUrmaRequest, GetAndPutReserveRegisterAndFinishRealWriter) {
    for (auto op : {UrmaOp::TGET, UrmaOp::TPUT}) {
        HbgUrmaFixture f;
        ASSERT_TRUE(f.submit(op));
        ASSERT_EQ(f.group().count, 3u);
        EXPECT_EQ(f.group().reservation_limit, 0u);
        EXPECT_EQ(f.entry(0).addr, uint64_t{6} << 32 | 1);
        EXPECT_EQ(f.entry(1).addr, uint64_t{7} << 32 | 1);
        EXPECT_EQ(f.entry(2).addr, uint64_t{6} << 32 | 2);
        EXPECT_EQ(f.entry(2).expected_value, 2u);
        EXPECT_EQ(f.entry(0).backend_cookie, reinterpret_cast<uint64_t>(f.workspace()));
        EXPECT_TRUE(pto2::resident::seal_conditions(f.writer));
    }
}
TEST(HbgUrmaRequest, CapacityAndMalformedWorkspaceRejectBeforePosting) {
    for (bool capacity : {false, true}) {
        HbgUrmaFixture f;
        if (capacity) f.group().capacity = 1;
        else f.info.rowsPerPeer = 0;
        f.group().entries_bytes = pto2::resident::async_group_entry_bytes(f.group().capacity);
        EXPECT_FALSE(f.submit(UrmaOp::TPUT));
        EXPECT_TRUE(f.script.submissions.empty());
        EXPECT_EQ(f.group().count, 0u);
    }
}
TEST(HbgUrmaRequest, InvalidSenderRejectsBeforePosting) {
    HbgUrmaFixture f;
    f.runtime = simpler::urma::HbgRequestAdapter(f.writer, UINT32_MAX);
    EXPECT_FALSE(f.submit(UrmaOp::TGET));
    EXPECT_TRUE(f.script.submissions.empty());
}
TEST(HbgUrmaRequest, ReservedCapacitySurvivesFreePoolExhaustion) {
    HbgUrmaFixture f;
    ASSERT_TRUE(f.runtime.reserve(4));
    ASSERT_TRUE(f.runtime.capacity_available(4));
    ASSERT_TRUE(f.runtime.register_target({f.workspace(), 6, 1, 1}));
    EXPECT_TRUE(f.runtime.capacity_available(3));
    EXPECT_FALSE(f.runtime.capacity_available(4));
    EXPECT_TRUE(f.runtime.finish());
    EXPECT_EQ(f.group().count, 1u);
    EXPECT_EQ(f.group().reservation_limit, 0u);
}
TEST(HbgUrmaRequest, PartialPostRetainsTargetsAndPoisonsSeal) {
    for (auto op : {UrmaOp::TGET, UrmaOp::TPUT}) {
        HbgUrmaFixture f;
        f.script.post_from_queue = false;
        for (uint32_t i = 0; i < 2; ++i) {
            TestEvent e{};
            e.urmaJettyBase = 2 + i;
            e.urmaJettyCount = 1;
            e.urmaTargetBbPerJetty[0] = i + 7;
            e.urmaTargetCqePerJetty[0] = i + 3;
            e.postFailed = i == 1;
            f.script.events.push_back(e);
        }
        EXPECT_FALSE(f.submit(op));
        EXPECT_EQ(f.script.submissions.size(), 2u);
        EXPECT_EQ(f.group().count, 2u);
        EXPECT_EQ(f.group().reservation_limit, 0u);
        EXPECT_EQ(f.entry(1).addr, (uint64_t{7} << 32) | 8);
        EXPECT_FALSE(pto2::resident::seal_conditions(f.writer));
        EXPECT_EQ(f.group().error_code, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
    }
}
TEST(HbgUrmaRequest, StaleReservationCannotRegisterOrEraseAcceptedPrefix) {
    for (bool epoch : {false, true}) {
        HbgUrmaFixture f;
        ASSERT_TRUE(f.runtime.reserve(2));
        ASSERT_TRUE(f.runtime.register_target({f.workspace(), 6, 1, 1}));
        if (epoch) ++f.group().run_epoch;
        else ++f.group().reservation_serial;
        EXPECT_FALSE(f.runtime.register_target({f.workspace(), 7, 2, 2}));
        EXPECT_FALSE(f.runtime.finish());
        EXPECT_EQ(f.group().count, 1u);
        EXPECT_EQ(f.group().reservation_limit, 2u);
    }
}

TEST(HbgUrmaRequest, MisalignedWorkspaceIsRejectedBeforeQueueAccess) {
    HbgUrmaFixture fixture;
    TestSession session;
    EXPECT_FALSE(simpler::urma::build_urma_session(0, fixture.workspace() + 1, 0, session));
}
TEST(HbgUrmaRequest, UrmaTokenRequiresItsOwnTypeAndAlignedWorkspace) {
    CompletionToken token{0, UINT32_MAX, COMPLETION_ENGINE_URMA, COMPLETION_TYPE_URMA_EVENT_HANDLE, 0x10000};
    EXPECT_TRUE(pto2::resident::detail::token_valid(token));
    token.backend_cookie = 0;
    EXPECT_FALSE(pto2::resident::detail::token_valid(token));
    token.backend_cookie = 0x10001;
    EXPECT_FALSE(pto2::resident::detail::token_valid(token));
    token.backend_cookie = 0x10000;
    token.completion_type = COMPLETION_TYPE_COUNTER;
    EXPECT_FALSE(pto2::resident::detail::token_valid(token));
}

TEST(HbgUrmaRequest, StaleWriterRejectsBeforePosting) {
    for (bool epoch : {false, true}) {
        HbgUrmaFixture f;
        if (epoch) ++f.group().run_epoch;
        else ++f.group().group_index;
        EXPECT_FALSE(f.submit(UrmaOp::TPUT));
        EXPECT_TRUE(f.script.submissions.empty());
        EXPECT_EQ(f.group().count, 0u);
    }
}
