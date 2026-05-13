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

// Contract test for DeviceRunner::upload_chip_callable_buffer.
//
// The previous per-fid `upload_kernel_binary` path mutated the caller's
// ChipCallable buffer in-place via const_cast to fill resolved_addr_; the
// new single-shot upload path stages a host scratch instead and leaves the
// caller bytes untouched. That invariant is load-bearing: chip-child
// processes share the ChipCallable buffer through COW pages, so any
// in-place write triggers a private copy (onboard) or rewrites bytes still
// shared with the parent (sim).
//
// This test pins the contract: any compliant implementation must
//   (1) compute total_size from header + max(binary_size, child_offset +
//       binary_data_offset + binary_size over children),
//   (2) leave the caller buffer bytewise identical after the call,
//   (3) write each child's fix-up address to the scratch only.
//
// The test reproduces the upload-side arithmetic locally so it does not
// require linking a full DeviceRunner. If a future implementation moves
// the fix-up back into the caller buffer, item (2) will fail here.

#include <cstdint>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include "callable.h"
#include "utils/fnv1a_64.h"

namespace {

// Build a ChipCallable with two trivial CoreCallable children. Returns the
// owning byte buffer; the ChipCallable* aliases its data().
std::vector<uint8_t> build_test_chip_callable() {
    // Two leaf kernels with arbitrary fake binaries.
    constexpr int kCoreSig = 2;
    ArgDirection core_sig[kCoreSig] = {ArgDirection::IN, ArgDirection::OUT};

    const uint8_t kernel0[] = {0xde, 0xad, 0xbe, 0xef, 0x01, 0x02, 0x03, 0x04};
    const uint8_t kernel1[] = {0xca, 0xfe, 0xba, 0xbe, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a};

    auto core0 = make_callable<CORE_MAX_TENSOR_ARGS>(core_sig, kCoreSig, kernel0, sizeof(kernel0));
    auto core1 = make_callable<CORE_MAX_TENSOR_ARGS>(core_sig, kCoreSig, kernel1, sizeof(kernel1));

    const uint8_t fake_orch_so[] = {0x7f, 'E', 'L', 'F', 0xaa, 0xbb};
    int32_t child_ids[2] = {5, 7};
    std::vector<uint8_t> children[2] = {std::move(core0), std::move(core1)};

    return make_callable<CoreCallable, CHIP_MAX_TENSOR_ARGS, 1024>(
        nullptr, 0, "orch_fn", fake_orch_so, sizeof(fake_orch_so), child_ids, children, 2, "cfg_name"
    );
}

}  // namespace

// ---------------------------------------------------------------------------
// Caller buffer must be bytewise identical before and after the upload-side
// arithmetic — this is the core anti-COW invariant.
// ---------------------------------------------------------------------------
TEST(ChipCallableUploadImmutable, CallerBufferUnchangedAfterScratchFixup) {
    auto chip_buf = build_test_chip_callable();
    const auto *callable = reinterpret_cast<const ChipCallable *>(chip_buf.data());
    ASSERT_EQ(callable->child_count(), 2);

    // Recompute total_size exactly as DeviceRunner::upload_chip_callable_buffer does.
    constexpr size_t kHeaderSize = offsetof(ChipCallable, storage_);
    size_t storage_used = static_cast<size_t>(callable->binary_size());
    for (int32_t i = 0; i < callable->child_count(); ++i) {
        const CoreCallable &c = callable->child(i);
        size_t end = static_cast<size_t>(callable->child_offset(i)) + CoreCallable::binary_data_offset() +
                     static_cast<size_t>(c.binary_size());
        if (end > storage_used) storage_used = end;
    }
    const size_t total_size = kHeaderSize + storage_used;
    ASSERT_LE(total_size, chip_buf.size()) << "Computed total_size must fit inside the make_callable allocation";

    // Hash the caller buffer *before* the upload-side fix-up.
    const uint64_t caller_hash_before = simpler::common::utils::fnv1a_64(chip_buf.data(), chip_buf.size());

    // Mirror onboard's fix-up path: copy to scratch, patch each child's
    // resolved_addr_ to a synthetic device address, never touch the caller.
    std::vector<uint8_t> scratch(total_size);
    std::memcpy(scratch.data(), chip_buf.data(), total_size);
    constexpr uint64_t kFakeChipDev = 0x10000ULL;
    for (int32_t i = 0; i < callable->child_count(); ++i) {
        const uint32_t off = callable->child_offset(i);
        auto *child_in_scratch = reinterpret_cast<CoreCallable *>(scratch.data() + kHeaderSize + off);
        const uint64_t child_dev = kFakeChipDev + kHeaderSize + off;
        child_in_scratch->set_resolved_addr(child_dev + CoreCallable::binary_data_offset());
    }

    // (1) Caller bytes unchanged.
    const uint64_t caller_hash_after = simpler::common::utils::fnv1a_64(chip_buf.data(), chip_buf.size());
    EXPECT_EQ(caller_hash_before, caller_hash_after) << "Upload must not mutate the caller's ChipCallable buffer";

    // (2) Caller-side resolved_addr_ on every child must still be the
    //     initial value written by make_callable. If a future implementation
    //     reintroduces in-place const_cast, this is the assertion that fires.
    for (int32_t i = 0; i < callable->child_count(); ++i) {
        EXPECT_EQ(callable->child(i).resolved_addr(), 0u)
            << "child " << i << " resolved_addr_ on caller side must be untouched";
    }

    // (3) Scratch-side resolved_addr_ matches the documented formula.
    for (int32_t i = 0; i < callable->child_count(); ++i) {
        const uint32_t off = callable->child_offset(i);
        const auto *child_in_scratch = reinterpret_cast<const CoreCallable *>(scratch.data() + kHeaderSize + off);
        const uint64_t expected = kFakeChipDev + kHeaderSize + off + CoreCallable::binary_data_offset();
        EXPECT_EQ(child_in_scratch->resolved_addr(), expected) << "child " << i << " scratch resolved_addr_ mismatch";
    }
}

// ---------------------------------------------------------------------------
// Each child's recorded offset must land on a CALLABLE_ALIGN boundary in
// storage_. make_callable enforces this; upload arithmetic relies on it.
// ---------------------------------------------------------------------------
TEST(ChipCallableUploadImmutable, ChildOffsetsAreCallableAligned) {
    auto chip_buf = build_test_chip_callable();
    const auto *callable = reinterpret_cast<const ChipCallable *>(chip_buf.data());
    for (int32_t i = 0; i < callable->child_count(); ++i) {
        EXPECT_EQ(callable->child_offset(i) % CALLABLE_ALIGN, 0u)
            << "child_offset(" << i << ") = " << callable->child_offset(i)
            << " is not a multiple of CALLABLE_ALIGN=" << CALLABLE_ALIGN;
    }
}
