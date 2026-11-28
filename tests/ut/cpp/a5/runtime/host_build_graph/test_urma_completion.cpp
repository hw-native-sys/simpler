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

#include <cstddef>

#include "async_condition_writer.h"

#include "inner_kernel.h"
#include "runtime_types.h"
#include "../urma_queue_fixture.h"
#include "backend/urma/urma_completion.h"

namespace {
using simpler::percounter::poll_urma;
using simpler::percounter::urma_token_valid;

struct CoreObserver : simpler::test::UrmaQueueFixture {
    UrmaRowOwner owners[simpler::test::kRows]{};
    UrmaRunBinding binding{addr(), sizeof(info), reinterpret_cast<uint64_t>(owners), 1, 4, 2, 1};

    CoreObserver() { info.jettiesPerCore = 1; }

    CompletionToken event(uint32_t bb, uint32_t cqe, uint32_t row = 0) {
        return {(uint64_t{row} << 32) | bb, cqe, COMPLETION_ENGINE_URMA, COMPLETION_TYPE_URMA_EVENT_HANDLE, addr()};
    }
};

struct Invocation {
    alignas(128) pto2::resident::AsyncGroupData group{};
    alignas(128) DeferredCompletionEntry entries[4]{};
    pto2::resident::AsyncWriter writer{reinterpret_cast<uint8_t *>(this), sizeof(*this), 1, 0, 0};

    Invocation() {
        group.run_epoch = writer.run_epoch;
        group.capacity = 4;
        group.entries_offset = offsetof(Invocation, entries);
        group.entries_bytes = pto2::resident::async_group_entry_bytes(group.capacity);
    }
};

TEST(HbgUrma, CountersAndUrmaShareTheFourSlots) {
    using namespace pto2::resident;
    CoreObserver row;
    Invocation call;
    const CompletionToken counter{0x10080, 1, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_COUNTER, 0};
    for (unsigned i = 0; i < 3; ++i)
        ASSERT_TRUE(register_completion_condition(call.writer, counter));
    const auto last = reserve_conditions(call.writer, 1);
    const auto token = row.event(0, 0);
    ASSERT_TRUE(detail::commit_validated_condition(call.writer, last, token, urma_token_valid(token, row.binding, 1)));
    ASSERT_TRUE(finish_reservation(call.writer, last, 1));
    EXPECT_EQ(call.entries[3].backend_cookie, row.addr());
    EXPECT_EQ(call.entries[3].addr, 0u);
    EXPECT_EQ(reserve_conditions(call.writer, 1).serial, 0u);
    EXPECT_EQ(call.group.count, 4u);
    EXPECT_EQ(call.group.error_code, SIMPLER_ERROR_ASYNC_WAIT_OVERFLOW);
}

TEST(HbgUrma, FailedRegistrationOrFinishKeepsEarlierTarget) {
    using namespace pto2::resident;
    CoreObserver row;
    for (const bool bad_target : {false, true}) {
        Invocation call;
        const auto first = reserve_conditions(call.writer, 1);
        const auto token = row.event(3, 1);
        ASSERT_TRUE(
            detail::commit_validated_condition(call.writer, first, token, urma_token_valid(token, row.binding, 1))
        );
        ASSERT_TRUE(finish_reservation(call.writer, first, 1));
        const auto next = reserve_conditions(call.writer, 1);
        if (bad_target) {
            auto foreign = token;
            ++foreign.backend_cookie;
            EXPECT_FALSE(
                detail::commit_validated_condition(
                    call.writer, next, foreign, urma_token_valid(foreign, row.binding, 1)
                )
            );
        } else {
            EXPECT_FALSE(finish_reservation(call.writer, next, 1));
        }
        EXPECT_FALSE(seal_conditions(call.writer));
        EXPECT_EQ(AsyncPublicationState{call.group.state}, AsyncPublicationState::FAILED);
        EXPECT_EQ(call.group.count, 1u);
        EXPECT_EQ(call.entries[0].addr, token.addr);
        EXPECT_EQ(call.entries[0].backend_cookie, row.addr());
    }
}

TEST(HbgUrma, BusyRowAndPendingPrefixLeaveBothTargetsAlone) {
    CoreObserver row;
    row.owners[0].value = 1;
    row.set_cqe(0, 0, true);
    EXPECT_EQ(poll_urma(row.event(2, 1), row.binding, 1).state, CompletionPollState::PENDING);
    EXPECT_EQ(row.owners[0].value, 1u);
    row.owners[0].value = 0;
    EXPECT_EQ(poll_urma(row.event(3, 2), row.binding, 1).state, CompletionPollState::PENDING);
    EXPECT_EQ(row.cq_tail[0], 0u);
    EXPECT_EQ(row.cq_doorbell[0], 0u);
    EXPECT_EQ(row.sq_tail[0], 0u);
    EXPECT_EQ(row.owners[0].value, 0u);
    row.set_cqe(0, 1, true, 1);
    EXPECT_EQ(poll_urma(row.event(3, 2), row.binding, 1).state, CompletionPollState::FAILED);
    EXPECT_EQ(row.cq_tail[0], 0u);
    EXPECT_EQ(row.sq_tail[0], 0u);
}

TEST(HbgUrma, NewerObserverCanSettleBeforeOlderObserver) {
    CoreObserver row;
    row.cq_tail[0] = 8;
    row.sq_tail[0] = 8;
    row.sq_doorbell[0] = 123;
    for (uint32_t seq = 8; seq < 12; ++seq)
        row.set_cqe(0, seq, true);
    EXPECT_EQ(poll_urma(row.event(12, 12), row.binding, 1).state, CompletionPollState::READY);
    EXPECT_EQ(poll_urma(row.event(10, 10), row.binding, 1).state, CompletionPollState::READY);
    EXPECT_EQ(row.cq_tail[0], 12u);
    EXPECT_EQ(row.sq_tail[0], 12u);
    EXPECT_EQ(row.cq_doorbell[0], 12u);
    EXPECT_EQ(row.sq_doorbell[0], 123u);
}

TEST(HbgUrma, ZeroPackedTargetAfterWrapStillRequiresTheBoundPeerAndSender) {
    CoreObserver row;
    const auto token = row.event(0, 0);
    ASSERT_EQ(token.addr, 0u);
    EXPECT_FALSE(urma_token_valid(token, row.binding, 0));
    EXPECT_FALSE(urma_token_valid(token, row.binding, 2));
    auto foreign = row.binding;
    ++foreign.workspace;
    EXPECT_FALSE(urma_token_valid(token, foreign, 1));
    row.cq_tail[0] = row.sq_tail[0] = UINT32_MAX;
    row.set_cqe(0, UINT32_MAX, false);
    EXPECT_EQ(poll_urma(token, row.binding, 1).state, CompletionPollState::PENDING);
    row.set_cqe(0, UINT32_MAX, true);
    EXPECT_EQ(poll_urma(token, row.binding, 1).state, CompletionPollState::READY);
    EXPECT_EQ(row.sq_tail[0], 0u);
    EXPECT_EQ(row.cq_tail[0], 0u);
}

TEST(HbgUrma, IndependentJettyCanRetireWhileFirstRowIsBusy) {
    CoreObserver row;
    row.binding.jetties_per_core = row.info.jettiesPerCore = 2;
    row.owners[0].value = 1;
    row.set_cqe(1, 0, true);
    EXPECT_EQ(poll_urma(row.event(2, 1, 0), row.binding, 1).state, CompletionPollState::PENDING);
    EXPECT_EQ(poll_urma(row.event(2, 1, 1), row.binding, 1).state, CompletionPollState::READY);
    EXPECT_EQ(row.sq_tail[1], 2u);
    EXPECT_EQ(row.cq_tail[1], 1u);
    EXPECT_EQ(row.cq_doorbell[1], 1u);
    EXPECT_EQ(row.sq_tail[0], 0u);
    EXPECT_EQ(row.owners[0].value, 1u);
    EXPECT_EQ(row.owners[1].value, 0u);
}

TEST(HbgUrma, ResolveDomainAndValidatePeerSenderAndJettyBeforePolling) {
    CoreObserver a, b;
    b.binding.rank_count = 4;
    b.binding.rows_per_peer = 8;
    b.binding.jetties_per_core = 2;
    const UrmaRunBinding retained[] = {a.binding, b.binding};
    const UrmaRunBindings domains{reinterpret_cast<uint64_t>(retained), 2};
    using simpler::percounter::find_urma_binding;
    EXPECT_EQ(find_urma_binding(domains, a.addr()), retained);
    EXPECT_EQ(find_urma_binding(domains, b.addr()), retained + 1);
    EXPECT_EQ(find_urma_binding(domains, 42), nullptr);
    // Local rank 1, physical AIV 2: jetties 0 and 1 to remote rank 3 own rows 28 and 29.
    for (uint32_t index : {28u, 29u})
        EXPECT_TRUE(urma_token_valid(b.event(2, 1, index), b.binding, 3));
    for (uint32_t invalid : {12u, 27u, 30u, 36u})
        EXPECT_FALSE(urma_token_valid(b.event(2, 1, invalid), b.binding, 3));
    EXPECT_FALSE(urma_token_valid(b.event(2, 1, 28), a.binding, 3));
    EXPECT_FALSE(urma_token_valid(b.event(2, 1, 28), b.binding, 5));
    // A mismatch between the retained geometry and the published workspace must not reach the queue arrays.
    EXPECT_EQ(poll_urma(b.event(2, 1, 28), b.binding, 3).state, CompletionPollState::FAILED);
}
}  // namespace
