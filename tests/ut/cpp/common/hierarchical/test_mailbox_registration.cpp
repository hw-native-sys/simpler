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
 * Registering a mailbox: the grant is a claim, the byte extent is the fact.
 *
 * An endpoint addresses `mailbox_run_entry_count(granted_sets)` entries and
 * clears every one of them in its constructor, so a registration whose region
 * is smaller than the grant implies writes past the mapping. The Python facade
 * sizes region and grant from the same place and cannot produce that pair, but
 * the C++ and nanobind registration surfaces are reachable directly, and a
 * direct caller's grant is nothing but an assertion until it is checked against
 * what it says it mapped.
 *
 * Every case here uses a heap region filled with a byte no endpoint ever
 * writes, so "refused before any access" is an assertion about the bytes and
 * not about the order of the source.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "ring.h"
#include "worker_manager.h"

namespace {

// Not 0: a zeroed entry is what a successful construction leaves behind, so a
// zero-filled region could not tell an untouched table from a cleared one.
constexpr char kCanary = static_cast<char>(0xAB);

std::vector<char> canary_region(size_t bytes) { return std::vector<char>(bytes, kCanary); }

bool all_canary(const char *begin, const char *end) {
    return std::all_of(begin, end, [](char byte) {
        return byte == kCanary;
    });
}

}  // namespace

TEST(MailboxRegistration, AGrantPastTheMappedRegionIsRefusedBeforeAnyWrite) {
    // One set mapped, two claimed: the second set's entry starts inside the last
    // 384 bytes of nothing.
    std::vector<char> region = canary_region(mailbox_size(1));
    EXPECT_THROW(
        { LocalMailboxEndpoint endpoint(0, region.data(), region.size(), /*child_pid=*/-1, /*granted_sets=*/2); },
        std::invalid_argument
    );
    EXPECT_TRUE(all_canary(region.data(), region.data() + region.size()))
        << "the constructor wrote into the region it had just been told was too small";
}

TEST(MailboxRegistration, TheExactRegionAGrantNeedsIsAcceptedAndOnlyTheTableIsCleared) {
    constexpr uint32_t kGrantedSets = 4;
    std::vector<char> region = canary_region(mailbox_size(kGrantedSets));
    EXPECT_NO_THROW({
        LocalMailboxEndpoint endpoint(0, region.data(), region.size(), /*child_pid=*/-1, kGrantedSets);
    });
    // The frames are the caller's to initialize and the entry table is the
    // endpoint's, so the boundary between them is where the canary stops.
    EXPECT_TRUE(all_canary(region.data(), region.data() + MAILBOX_RUN_ENTRY_TABLE_OFFSET))
        << "the control and request frames are not the constructor's to clear";
    const char *table = region.data() + MAILBOX_RUN_ENTRY_TABLE_OFFSET;
    EXPECT_FALSE(all_canary(table, region.data() + region.size())) << "the entry table was never cleared";
    for (size_t index = 0; index < mailbox_run_entry_count(kGrantedSets); ++index) {
        const MailboxRunEntry *entry = mailbox_run_entry(region.data(), index);
        EXPECT_EQ(entry->entry_generation, 0u) << "entry " << index << " starts at an odd generation";
        EXPECT_EQ(entry->state, 0) << "entry " << index << " starts non-free";
    }
}

TEST(MailboxRegistration, AGrantWithNoRepresentableSizeIsRefusedWhateverExtentIsClaimed) {
    // Above `MAILBOX_MAX_GRANTED_SETS`, so the bound a bounds check would use is
    // itself the wrapped number — which is why this is refused on the grant and
    // not on the extent. The extent claimed is the largest there is.
    constexpr uint32_t kUnrepresentable = 3'000'000'000u;
    static_assert(static_cast<uint64_t>(kUnrepresentable) > MAILBOX_MAX_GRANTED_SETS, "pick a grant past the bound");
    std::vector<char> region = canary_region(mailbox_size(1));
    EXPECT_THROW(
        {
            LocalMailboxEndpoint endpoint(
                0, region.data(), std::numeric_limits<size_t>::max(), /*child_pid=*/-1, kUnrepresentable
            );
        },
        std::invalid_argument
    );
    EXPECT_TRUE(all_canary(region.data(), region.data() + region.size()));
    EXPECT_EQ(mailbox_size_for_grant(kUnrepresentable), 0u);
}

TEST(MailboxRegistration, TheRepresentableBoundIsTheLastGrantWithASize) {
    EXPECT_EQ(mailbox_size_for_grant(0), 0u) << "a grant of nothing has no mailbox";
    EXPECT_GT(mailbox_size_for_grant(MAILBOX_MAX_GRANTED_SETS), 0u);
    EXPECT_EQ(mailbox_size_for_grant(MAILBOX_MAX_GRANTED_SETS + 1), 0u);
    EXPECT_EQ(mailbox_size_for_grant(MAILBOX_DEFAULT_GRANTED_SETS), mailbox_size(MAILBOX_DEFAULT_GRANTED_SETS))
        << "the checked and unchecked sizes disagree inside the bound";
}

TEST(MailboxRegistration, ANullMailboxIsStillRefused) {
    EXPECT_THROW(
        { LocalMailboxEndpoint endpoint(0, nullptr, mailbox_size(1), /*child_pid=*/-1, /*granted_sets=*/1); },
        std::invalid_argument
    );
}

TEST(MailboxRegistration, AnUndersizedRegistrationInstallsNoNextLevelWorker) {
    Ring allocator;
    allocator.init(/*heap_bytes=*/1ULL << 20);
    WorkerManager manager;

    std::vector<char> too_small = canary_region(mailbox_size(1));
    EXPECT_THROW(
        manager.add_next_level_at(/*worker_id=*/7, too_small.data(), too_small.size(), /*child_pid=*/-1,
                                  /*granted_sets=*/2),
        std::invalid_argument
    );
    EXPECT_TRUE(all_canary(too_small.data(), too_small.data() + too_small.size()));

    // The proof that the refusal installed nothing: worker_id 7 is still free.
    // Had the rejected registration been recorded, `start` would refuse the pair
    // as a duplicate NEXT_LEVEL worker_id instead of standing one lane up.
    std::vector<char> region(mailbox_size(2), 0);
    manager.add_next_level_at(/*worker_id=*/7, region.data(), region.size(), /*child_pid=*/-1, /*granted_sets=*/2);
    manager.start(&allocator, [](WorkerCompletion) {}, [](WorkerDispatch) {});
    EXPECT_EQ(manager.next_level_worker_ids(), std::vector<int32_t>{7});
    manager.stop();
}

TEST(MailboxRegistration, AnUndersizedSubRegistrationIsRefused) {
    WorkerManager manager;
    // A SUB endpoint takes the default grant of one set, and its entry table
    // sits at the same offset, so one byte short of `mailbox_size(1)` is short.
    std::vector<char> region = canary_region(mailbox_size(1) - 1);
    EXPECT_THROW(manager.add_sub(region.data(), region.size()), std::invalid_argument);
    EXPECT_TRUE(all_canary(region.data(), region.data() + region.size()));
}
