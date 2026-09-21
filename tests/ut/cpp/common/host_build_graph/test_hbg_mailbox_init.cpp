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
/**
 * The completion mailbox is reserved in the arena's device-only zone, so it
 * reaches the AICPU holding whatever the pooled allocation last had.
 * init_empty() is what makes it an empty ring.
 *
 * host_build_graph only: the tensormap_and_ringbuffer mailbox is uploaded with
 * the rest of its image and has no such entry point.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <new>

#include "aicore_completion_mailbox.h"

namespace {

// The mailbox stores a token and compares it for identity; it never decodes one,
// and the encoding belongs to whichever runtime minted it, so any distinct 64-bit
// value serves here.
TaskId make_token(uint32_t local) { return TaskId{local}; }

// A mailbox on memory holding a prior generation's bytes, which is what the
// pooled arena hands the AICPU: the region is never uploaded, so nothing zeroes
// it on the way in.
AICoreCompletionMailbox *dirty_mailbox(uint8_t fill) {
    void *raw = ::operator new(sizeof(AICoreCompletionMailbox));
    std::memset(raw, fill, sizeof(AICoreCompletionMailbox));
    return new (raw) AICoreCompletionMailbox{};
}

void destroy_mailbox(AICoreCompletionMailbox *mb) {
    mb->~AICoreCompletionMailbox();
    ::operator delete(mb);
}

}  // namespace

// The window init_empty() closes: a producer bumps head before it stores seq, so
// between those two a consumer already sees t < head and reads entries[t].seq. A
// residual seq equal to t + 1 passes the publication gate and hands out a message
// the producer has not written. Seeding entries[0].seq = 1 is exactly the value
// the gate accepts for the first pop.
TEST(HbgMailboxInit, ClosesThePrePublicationWindowOnDirtyMemory) {
    AICoreCompletionMailbox *mb = dirty_mailbox(0xAA);
    mb->entries[0].seq.store(1, std::memory_order_relaxed);

    mb->init_empty();

    // Stand in for a producer that has claimed slot 0 but not yet published.
    mb->head.store(1, std::memory_order_relaxed);
    AICoreCompletionMsgView out{};
    EXPECT_FALSE(mb->try_pop(out)) << "a residual seq must not pass the publication gate";

    // The producer's own release-store is still observed.
    mb->entries[0].seq.store(1, std::memory_order_release);
    EXPECT_TRUE(mb->try_pop(out));

    destroy_mailbox(mb);
}

TEST(HbgMailboxInit, DirtyMemoryStartsEmpty) {
    AICoreCompletionMailbox *mb = dirty_mailbox(0xAA);

    mb->init_empty();

    EXPECT_FALSE(mb->has_pending());
    AICoreCompletionMsgView out{};
    EXPECT_FALSE(mb->try_pop(out));

    destroy_mailbox(mb);
}

// A full round trip on dirty memory, so the fields try_pop copies out are proven
// to come from the producer rather than from the fill.
TEST(HbgMailboxInit, DirtyMemoryYieldsAUsableRing) {
    AICoreCompletionMailbox *mb = dirty_mailbox(0xAA);
    mb->init_empty();

    const TaskId token = make_token(11);
    ASSERT_TRUE(
        mb->try_push_condition(token, /*addr=*/0x4000, /*expected_value=*/7, /*engine=*/1, /*completion_type=*/0)
    );
    ASSERT_TRUE(mb->has_pending());

    AICoreCompletionMsgView out{};
    ASSERT_TRUE(mb->try_pop(out));
    EXPECT_EQ(out.task_token.raw, token.raw);
    EXPECT_EQ(out.addr, 0x4000ULL);
    EXPECT_EQ(out.expected_value, 7U);
    EXPECT_FALSE(mb->try_pop(out));

    destroy_mailbox(mb);
}
