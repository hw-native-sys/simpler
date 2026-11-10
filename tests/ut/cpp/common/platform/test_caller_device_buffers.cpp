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

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <new>

#include "host/caller_device_buffers.h"

namespace {
thread_local size_t fail_after = std::numeric_limits<size_t>::max();
}  // namespace

void *operator new(std::size_t bytes) {
    if (fail_after != std::numeric_limits<size_t>::max()) {
        if (fail_after-- == 0) {
            fail_after = std::numeric_limits<size_t>::max();
            throw std::bad_alloc();
        }
    }
    if (void *storage = std::malloc(bytes == 0 ? 1 : bytes)) return storage;
    throw std::bad_alloc();
}

namespace {

// Stand-in device addresses. Nothing here dereferences them; the table is pure bookkeeping over
// addresses its owner minted, so an address only has to be distinguishable and non-null.
void *at(uintptr_t value) { return reinterpret_cast<void *>(value); }
constexpr uintptr_t kBase = 0x10000;
constexpr uint64_t kSize = 0x1000;
constexpr uint64_t kRunA = 1;
constexpr uint64_t kRunB = 2;

CallerDeviceBuffers::Span span(uintptr_t addr, uint64_t bytes) {
    return CallerDeviceBuffers::Span{static_cast<uint64_t>(addr), bytes};
}

}  // namespace

TEST(CallerDeviceBuffers, OnlyARecordedMintResolves) {
    CallerDeviceBuffers buffers;
    CallerDeviceBuffers::Allocation allocation{};

    EXPECT_FALSE(buffers.resolve(kBase, kSize, &allocation));
    buffers.record(at(kBase), kSize);
    ASSERT_TRUE(buffers.resolve(kBase, kSize, &allocation));
    EXPECT_EQ(allocation.base, kBase);
    EXPECT_EQ(allocation.bytes, kSize);

    // An address the owner allocated for itself never reaches `record`, so it reads exactly like
    // one that was never allocated: a run naming it cannot prove an owner.
    EXPECT_FALSE(buffers.resolve(kBase + 0x100000, kSize, &allocation));
}

TEST(CallerDeviceBuffers, AnInteriorSpanResolvesToItsContainingAllocation) {
    CallerDeviceBuffers buffers;
    buffers.record(at(kBase), kSize);
    CallerDeviceBuffers::Allocation allocation{};

    // A tensor may sit at an offset inside a larger buffer, and release is per allocation, so the
    // unit of proof is the allocation that covers the whole span.
    ASSERT_TRUE(buffers.resolve(kBase + 0x10, 0x20, &allocation));
    EXPECT_EQ(allocation.base, kBase);

    // Both ends are bounded: a span that starts inside but runs past the end is not covered.
    EXPECT_TRUE(buffers.resolve(kBase + kSize - 1, 1, &allocation));
    EXPECT_FALSE(buffers.resolve(kBase + kSize - 1, 2, &allocation));
    EXPECT_FALSE(buffers.resolve(kBase - 1, 1, &allocation));
    // A length that would overflow the address must not wrap into a pass.
    EXPECT_FALSE(buffers.resolve(kBase + 0x10, UINT64_MAX, &allocation));
    // An empty span names no bytes, so there is nothing to own.
    EXPECT_FALSE(buffers.resolve(kBase, 0, &allocation));
}

TEST(CallerDeviceBuffers, UnknownSpansRefuseJoiningButRetainKnownAllocations) {
    for (bool unknown_first : {false, true}) {
        for (bool keep : {false, true}) {
            SCOPED_TRACE(unknown_first);
            SCOPED_TRACE(keep);
            CallerDeviceBuffers buffers;
            buffers.record(at(kBase), kSize);
            const auto known = span(kBase, 0x10);
            const auto unknown = span(kBase + 0x100000, 0x10);
            const CallerDeviceBuffers::Span spans[] = {
                unknown_first ? unknown : known, unknown_first ? known : unknown
            };
            EXPECT_FALSE(buffers.borrow(kRunA, spans, 2));
            EXPECT_EQ(buffers.borrow_count(), 1u);
            EXPECT_FALSE(buffers.forget_if_unborrowed(at(kBase)));
            // Only the external OUTPUT is declared; the owned INPUT needs its borrow.
            size_t unresolved = 0;
            EXPECT_TRUE(buffers.declare_writes(kRunA, &unknown, 1, &unresolved));
            EXPECT_EQ(unresolved, 1u);
            buffers.release(kRunA, keep);
            EXPECT_EQ(buffers.forget_if_unborrowed(at(kBase)), !keep);
        }
    }
}

TEST(CallerDeviceBuffers, AReleaseIsRefusedWhileARunHoldsTheAllocation) {
    CallerDeviceBuffers buffers;
    buffers.record(at(kBase), kSize);
    const CallerDeviceBuffers::Span held[] = {span(kBase + 0x40, 0x40)};

    ASSERT_TRUE(buffers.borrow(kRunA, held, 1));
    EXPECT_TRUE(buffers.borrowed(at(kBase)));
    EXPECT_FALSE(buffers.forget_if_unborrowed(at(kBase)));
    // Refused rather than deferred, and nothing changed: the entry is still there to refuse again.
    EXPECT_TRUE(buffers.recorded(at(kBase)));

    buffers.release(kRunA, /*keep=*/false);
    EXPECT_FALSE(buffers.borrowed(at(kBase)));
    EXPECT_TRUE(buffers.forget_if_unborrowed(at(kBase)));
    EXPECT_FALSE(buffers.recorded(at(kBase)));
}

TEST(CallerDeviceBuffers, OneRunsReleaseDoesNotDischargeAnothersBorrow) {
    CallerDeviceBuffers buffers;
    buffers.record(at(kBase), kSize);
    const CallerDeviceBuffers::Span shared[] = {span(kBase, kSize)};

    ASSERT_TRUE(buffers.borrow(kRunA, shared, 1));
    ASSERT_TRUE(buffers.borrow(kRunB, shared, 1));
    EXPECT_EQ(buffers.borrow_count(), 2u);

    buffers.release(kRunA, /*keep=*/false);
    EXPECT_TRUE(buffers.borrowed(at(kBase)));
    EXPECT_FALSE(buffers.forget_if_unborrowed(at(kBase)));

    buffers.release(kRunB, /*keep=*/false);
    EXPECT_FALSE(buffers.borrowed(at(kBase)));
}

TEST(CallerDeviceBuffers, ReBorrowingUnderOneIdentityReplacesThatIdentitysSet) {
    CallerDeviceBuffers buffers;
    buffers.record(at(kBase), kSize);
    buffers.record(at(kBase + kSize), kSize);
    const CallerDeviceBuffers::Span first[] = {span(kBase, kSize)};
    const CallerDeviceBuffers::Span second[] = {span(kBase + kSize, kSize)};

    ASSERT_TRUE(buffers.borrow(kRunA, first, 1));
    // A re-prepared run names its spans again; keeping both sets would leak a reference it no
    // longer holds.
    ASSERT_TRUE(buffers.borrow(kRunA, second, 1));
    EXPECT_FALSE(buffers.borrowed(at(kBase)));
    EXPECT_TRUE(buffers.borrowed(at(kBase + kSize)));
    EXPECT_EQ(buffers.borrow_count(), 1u);
}

TEST(CallerDeviceBuffers, IncompleteReBorrowReplacesOrClearsThePreviousSet) {
    for (bool has_known_span : {false, true}) {
        SCOPED_TRACE(has_known_span);
        CallerDeviceBuffers buffers;
        buffers.record(at(kBase), kSize);
        buffers.record(at(kBase + kSize), kSize);
        const auto first = span(kBase, kSize);
        ASSERT_TRUE(buffers.borrow(kRunA, &first, 1));
        const CallerDeviceBuffers::Span replacement[] = {span(kBase + 0x100000, kSize), span(kBase + kSize, kSize)};
        EXPECT_FALSE(buffers.borrow(kRunA, replacement, has_known_span ? 2 : 1));
        EXPECT_TRUE(buffers.forget_if_unborrowed(at(kBase)));
        EXPECT_EQ(buffers.borrowed(at(kBase + kSize)), has_known_span);
        EXPECT_EQ(buffers.borrow_count(), has_known_span ? 1u : 0u);
        buffers.release(kRunA, /*keep=*/false);
        EXPECT_TRUE(buffers.forget_if_unborrowed(at(kBase + kSize)));
    }
}

TEST(CallerDeviceBuffers, AnUnprovenLastConsumerRetainsTheAllocationForGood) {
    CallerDeviceBuffers buffers;
    buffers.record(at(kBase), kSize);
    const CallerDeviceBuffers::Span held[] = {span(kBase, kSize)};
    ASSERT_TRUE(buffers.borrow(kRunA, held, 1));

    buffers.release(kRunA, /*keep=*/true);
    EXPECT_EQ(buffers.borrow_count(), 0u);
    EXPECT_EQ(buffers.retained_count(), 1u);
    // The device may still name these bytes and nothing later can prove otherwise, so the release
    // stays refused rather than being delayed.
    EXPECT_TRUE(buffers.borrowed(at(kBase)));
    EXPECT_FALSE(buffers.forget_if_unborrowed(at(kBase)));

    // Not even a fresh borrow-and-clean-release reopens it.
    ASSERT_TRUE(buffers.borrow(kRunB, held, 1));
    buffers.release(kRunB, /*keep=*/false);
    EXPECT_FALSE(buffers.forget_if_unborrowed(at(kBase)));
}

// Holding is not producing. This is the property that keeps every path outside the device chain
// on its old behaviour: two runs may take one immutable device input, and neither may be told the
// other's bytes are unreadable.
TEST(CallerDeviceBuffers, ABorrowAloneNeverMakesASpanUnreadable) {
    CallerDeviceBuffers buffers;
    buffers.record(at(kBase), kSize);
    const CallerDeviceBuffers::Span held[] = {span(kBase, kSize)};
    ASSERT_TRUE(buffers.borrow(kRunA, held, 1));
    ASSERT_TRUE(buffers.borrow(kRunB, held, 1));

    EXPECT_FALSE(buffers.written_by_other_run(kRunA, kBase, kSize));
    EXPECT_FALSE(buffers.written_by_other_run(kRunB, kBase, kSize));
}

TEST(CallerDeviceBuffers, ADeclaredWriteIsWhatMakesASpanUnreadableToOtherRuns) {
    CallerDeviceBuffers buffers;
    buffers.record(at(kBase), kSize);
    const CallerDeviceBuffers::Span produced[] = {span(kBase, kSize)};
    ASSERT_TRUE(buffers.declare_writes(kRunA, produced, 1));

    // The declaring run is excluded: it is building the graph that will write these bytes.
    EXPECT_FALSE(buffers.written_by_other_run(kRunA, kBase, kSize));
    EXPECT_TRUE(buffers.written_by_other_run(kRunB, kBase, kSize));
    // Including a sub-span, since a declaration covers the containing allocation.
    EXPECT_TRUE(buffers.written_by_other_run(kRunB, kBase + 0x20, 0x8));
    EXPECT_EQ(buffers.write_declaration_count(), 1u);

    // A declaration lives exactly as long as its run.
    buffers.release(kRunA, /*keep=*/false);
    EXPECT_FALSE(buffers.written_by_other_run(kRunB, kBase, kSize));
    EXPECT_EQ(buffers.write_declaration_count(), 0u);
}

// The production shape of the case above: a run holds a borrow *and* a declaration, and its
// ordinary retirement has to discharge both. A declaration left behind would make its own output
// unreadable to every later run — including the next run in this slot, which inherits the
// identity.
TEST(CallerDeviceBuffers, AProvenReleaseDischargesTheBorrowAndTheDeclaration) {
    CallerDeviceBuffers buffers;
    buffers.record(at(kBase), kSize);
    const CallerDeviceBuffers::Span produced[] = {span(kBase, kSize)};
    ASSERT_TRUE(buffers.borrow(kRunA, produced, 1));
    ASSERT_TRUE(buffers.declare_writes(kRunA, produced, 1));
    ASSERT_TRUE(buffers.written_by_other_run(kRunB, kBase, kSize));

    buffers.release(kRunA, /*keep=*/false);
    EXPECT_EQ(buffers.borrow_count(), 0u);
    EXPECT_EQ(buffers.write_declaration_count(), 0u);
    // A later run reading this completed producer's output is not refused, and the caller's
    // release is not refused either.
    EXPECT_FALSE(buffers.written_by_other_run(kRunB, kBase, kSize));
    EXPECT_FALSE(buffers.borrowed(at(kBase)));
    EXPECT_TRUE(buffers.forget_if_unborrowed(at(kBase)));
}

TEST(CallerDeviceBuffers, ReDeclaringReplacesThatRunsWriteSet) {
    CallerDeviceBuffers buffers;
    buffers.record(at(kBase), kSize);
    buffers.record(at(kBase + kSize), kSize);
    const CallerDeviceBuffers::Span first[] = {span(kBase, kSize)};
    const CallerDeviceBuffers::Span second[] = {span(kBase + kSize, kSize)};

    ASSERT_TRUE(buffers.declare_writes(kRunA, first, 1));
    // A re-prepared run states its writes again; the previous set must not linger.
    ASSERT_TRUE(buffers.declare_writes(kRunA, second, 1));
    EXPECT_FALSE(buffers.written_by_other_run(kRunB, kBase, kSize));
    EXPECT_TRUE(buffers.written_by_other_run(kRunB, kBase + kSize, kSize));

    // An empty declaration is how a run with no device output says so.
    ASSERT_TRUE(buffers.declare_writes(kRunA, nullptr, 0));
    EXPECT_FALSE(buffers.written_by_other_run(kRunB, kBase + kSize, kSize));
    // A declaration no run could name is refused rather than recorded under some default.
    EXPECT_FALSE(buffers.declare_writes(0, first, 1));
}

TEST(CallerDeviceBuffers, ADeclarationOverAnUnrecordedSpanProtectsNothing) {
    CallerDeviceBuffers buffers;
    const CallerDeviceBuffers::Span produced[] = {span(kBase, kSize)};
    // Not a caller mint, so there is no allocation to speak for.
    ASSERT_TRUE(buffers.declare_writes(kRunA, produced, 1));
    EXPECT_FALSE(buffers.written_by_other_run(kRunB, kBase, kSize));
    EXPECT_EQ(buffers.write_declaration_count(), 0u);
}

// The fail-closed half of an unproven last consumer. The run that said it produces these bytes
// ended without its finalize proving the device was done with them, so nothing can establish that
// the write completed — and unlike the borrow, this mark cannot be keyed on the run: the run is
// gone and the next one to occupy its slot gets its identity.
TEST(CallerDeviceBuffers, AnUnprovenProducersBytesStayUnreadableForGood) {
    CallerDeviceBuffers buffers;
    buffers.record(at(kBase), kSize);
    const CallerDeviceBuffers::Span held[] = {span(kBase, kSize)};
    ASSERT_TRUE(buffers.borrow(kRunA, held, 1));
    ASSERT_TRUE(buffers.declare_writes(kRunA, held, 1));

    buffers.release(kRunA, /*keep=*/true);
    EXPECT_EQ(buffers.write_declaration_count(), 0u);
    EXPECT_TRUE(buffers.borrowed(at(kBase)));
    EXPECT_TRUE(buffers.written_by_other_run(kRunB, kBase, kSize));
    // Not even to a run under the identity the unproven one used, which is what a slot reuse
    // hands out, and not after a later run's own clean release.
    EXPECT_TRUE(buffers.written_by_other_run(kRunA, kBase, kSize));
    ASSERT_TRUE(buffers.borrow(kRunA, held, 1));
    buffers.release(kRunA, /*keep=*/false);
    EXPECT_TRUE(buffers.written_by_other_run(kRunA, kBase, kSize));
}

// A retained borrow on its own is not that: holding an allocation is not producing it, so a run
// that named an immutable input and then failed to prove its consumer done leaves the bytes
// readable. Only what was declared becomes unreadable.
TEST(CallerDeviceBuffers, ARetainedBorrowWithNoDeclarationLeavesTheBytesReadable) {
    CallerDeviceBuffers buffers;
    buffers.record(at(kBase), kSize);
    const CallerDeviceBuffers::Span held[] = {span(kBase, kSize)};
    ASSERT_TRUE(buffers.borrow(kRunA, held, 1));
    buffers.release(kRunA, /*keep=*/true);

    EXPECT_TRUE(buffers.borrowed(at(kBase)));
    EXPECT_FALSE(buffers.forget_if_unborrowed(at(kBase)));
    EXPECT_FALSE(buffers.written_by_other_run(kRunB, kBase, kSize));
}

TEST(CallerDeviceBuffers, ADeclarationWithoutABorrowIsStillReleased) {
    CallerDeviceBuffers buffers;
    buffers.record(at(kBase), kSize);
    const CallerDeviceBuffers::Span produced[] = {span(kBase, kSize)};
    // A run whose borrow was refused can still have declared its writes, so the release has to
    // reach the declaration on its own.
    ASSERT_TRUE(buffers.declare_writes(kRunA, produced, 1));
    ASSERT_EQ(buffers.borrow_count(), 0u);
    buffers.release(kRunA, /*keep=*/false);
    EXPECT_EQ(buffers.write_declaration_count(), 0u);
    EXPECT_FALSE(buffers.written_by_other_run(kRunB, kBase, kSize));
}

TEST(CallerDeviceBuffers, AnAddressThatWasNeverRecordedIsNeitherBorrowedNorForgettable) {
    CallerDeviceBuffers buffers;
    EXPECT_FALSE(buffers.borrowed(at(kBase)));
    // Nothing holds it, so this reports that the caller may proceed — and the caller's own free
    // path is what decides what an unrecorded address means.
    EXPECT_TRUE(buffers.forget_if_unborrowed(at(kBase)));
    EXPECT_FALSE(buffers.written_by_other_run(kRunA, kBase, kSize));
    EXPECT_EQ(buffers.allocation_count(), 0u);
    EXPECT_FALSE(buffers.borrowed(nullptr));
    EXPECT_FALSE(buffers.forget_if_unborrowed(nullptr));
}

TEST(CallerDeviceBuffers, ARunNamingNoSpanHoldsNoBorrow) {
    CallerDeviceBuffers buffers;
    buffers.record(at(kBase), kSize);
    // An empty span list is not a refusal: a run with no device argument is admissible and simply
    // holds nothing.
    EXPECT_TRUE(buffers.borrow(kRunA, nullptr, 0));
    EXPECT_EQ(buffers.borrow_count(), 0u);
    EXPECT_TRUE(buffers.forget_if_unborrowed(at(kBase)));
    // A borrow with no identity is refused: a release could not name it.
    EXPECT_FALSE(buffers.borrow(0, nullptr, 0));
}

TEST(CallerDeviceBuffers, RetainedAliasesDetachBeforeTheSameSlotAcquiresADifferentWriter) {
    CallerDeviceBuffers buffers;
    buffers.record(at(kBase), kSize);
    buffers.record(at(kBase + kSize), kSize);
    const auto a = span(kBase, 64), b = span(kBase + kSize, 64);
    int ar, br;
    ASSERT_TRUE(buffers.borrow(kRunA, &a, 1));
    ASSERT_TRUE(buffers.declare_writes(kRunA, &a, 1));
    EXPECT_TRUE(buffers.written_by_other_run(0, a.addr, a.bytes));
    ASSERT_TRUE(buffers.retain_reference(&ar, kRunA));
    EXPECT_FALSE(buffers.written_by_other_run(kRunA, a.addr, a.bytes));
    EXPECT_TRUE(buffers.written_by_other_run(0, a.addr, a.bytes));
    EXPECT_EQ(buffers.borrow_count(), 1u);
    buffers.detach_reference(&ar);
    buffers.release(kRunA, false);
    EXPECT_TRUE(buffers.written_by_other_run(kRunA, a.addr, a.bytes));
    EXPECT_TRUE(buffers.written_by_other_run(0, a.addr, a.bytes));
    ASSERT_TRUE(buffers.borrow(kRunA, &b, 1));
    ASSERT_TRUE(buffers.declare_writes(kRunA, &b, 1));
    ASSERT_TRUE(buffers.retain_reference(&br, kRunA));
    EXPECT_FALSE(buffers.written_by_other_run(kRunA, b.addr, b.bytes));
    EXPECT_TRUE(buffers.borrowed(at(kBase)));
    buffers.release_reference(&ar, false);
    EXPECT_FALSE(buffers.borrowed(at(kBase)));
    EXPECT_TRUE(buffers.borrowed(at(kBase + kSize)));
    EXPECT_TRUE(buffers.written_by_other_run(kRunB, b.addr, b.bytes));
    buffers.release_reference(&br, false);
    EXPECT_TRUE(buffers.borrowed(at(kBase + kSize)));
    buffers.release(kRunA, false);
    EXPECT_FALSE(buffers.borrowed(at(kBase + kSize)));
}

TEST(CallerDeviceBuffers, SnapshotAllocationFailuresRollbackOnlyTheUnpublishedReference) {
    size_t failures = 0;
    for (size_t cut = 0; cut < 12; ++cut) {
        CallerDeviceBuffers buffers;
        buffers.record(at(kBase), kSize);
        buffers.record(at(kBase + kSize), kSize);
        const auto a = span(kBase, 64), b = span(kBase + kSize, 64);
        int old, next;
        ASSERT_TRUE(buffers.borrow(kRunB, &b, 1));
        ASSERT_TRUE(buffers.declare_writes(kRunB, &b, 1));
        ASSERT_TRUE(buffers.retain_reference(&old, kRunB));
        buffers.detach_reference(&old);
        buffers.release(kRunB, false);
        ASSERT_TRUE(buffers.borrow(kRunA, &a, 1));
        ASSERT_TRUE(buffers.declare_writes(kRunA, &a, 1));
        fail_after = cut;
        bool injected = false;
        try {
            EXPECT_TRUE(buffers.retain_reference(&next, kRunA));
        } catch (const std::bad_alloc &) {
            injected = true;
        }
        fail_after = std::numeric_limits<size_t>::max();
        if (injected) {
            ++failures;
            EXPECT_TRUE(buffers.borrowed(at(kBase)));
            EXPECT_FALSE(buffers.written_by_other_run(kRunA, a.addr, a.bytes));
            buffers.release(kRunA, false);
            EXPECT_FALSE(buffers.borrowed(at(kBase)));
            EXPECT_FALSE(buffers.written_by_other_run(kRunB, a.addr, a.bytes));
            EXPECT_TRUE(buffers.borrowed(at(kBase + kSize)));
            EXPECT_TRUE(buffers.written_by_other_run(kRunA, b.addr, b.bytes));
            ASSERT_TRUE(buffers.borrow(kRunA, &a, 1));
            ASSERT_TRUE(buffers.declare_writes(kRunA, &a, 1));
            EXPECT_TRUE(buffers.retain_reference(&next, kRunA));
        }
        buffers.release_reference(&next, false);
        buffers.release(kRunA, false);
        buffers.release_reference(&old, false);
        EXPECT_FALSE(buffers.borrowed(at(kBase)));
        EXPECT_FALSE(buffers.borrowed(at(kBase + kSize)));
    }
    EXPECT_GE(failures, 4u);
}

TEST(CallerDeviceBuffers, PartialKnownSnapshotAndAbandonKeepPermanentRefusalFacts) {
    CallerDeviceBuffers buffers;
    buffers.record(at(kBase), kSize);
    const CallerDeviceBuffers::Span spans[] = {span(kBase, 64), span(kBase + 0x100000, 64)};
    int reference;
    EXPECT_FALSE(buffers.borrow(kRunA, spans, 2));
    ASSERT_TRUE(buffers.declare_writes(kRunA, spans, 2));
    ASSERT_TRUE(buffers.retain_reference(&reference, kRunA));
    buffers.detach_reference(&reference);
    buffers.release(kRunA, false);
    buffers.release_reference(&reference, true);
    EXPECT_TRUE(buffers.borrowed(at(kBase)));
    EXPECT_FALSE(buffers.forget_if_unborrowed(at(kBase)));
    EXPECT_TRUE(buffers.written_by_other_run(kRunA, kBase, 64));
    EXPECT_EQ(buffers.retained_count(), 1u);
}
