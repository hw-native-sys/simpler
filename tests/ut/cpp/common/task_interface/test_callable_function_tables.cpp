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
 * A callable's two function tables are a pure function of its own bytes and the
 * base its registration block becomes readable at.
 *
 * Registration builds them once and every run that binds the callable
 * references them, so what has to hold is that the derivation is complete
 * before publication and identical for every caller of one block: the length
 * covers exactly the callable's func_id domain, holes read zero, an
 * unaddressable func_id is refused rather than sized around, and the two views
 * differ only in the one step that separates a CoreCallable object from the
 * address its consumer dispatches through.
 *
 * These cases drive the shared derivation directly, without a device: the
 * platform halves differ only in the base they pass and whether they copy the
 * result.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "callable.h"
#include "chip_callable_layout.h"

namespace {

// Stands in for the runtime's own RUNTIME_MAX_FUNC_ID, which is the exclusive
// bound registration validates a child's func_id against.
constexpr uint32_t kFuncIdDomain = 1024;

// A device base far from any host address, so an entry that came from the host
// scratch instead of the patched device base is visible in the value.
constexpr uint64_t kDeviceBase = 0x7000'0000'0000ull;

std::vector<uint8_t> make_core_child(uint8_t fill, uint32_t binary_size) {
    std::vector<uint8_t> binary(binary_size, fill);
    return make_callable<CORE_MAX_TENSOR_ARGS>(nullptr, 0, binary.data(), binary_size);
}

// A ChipCallable whose children claim `func_ids`, with one distinctly filled
// kernel binary each.
std::vector<uint8_t> make_chip_with_children(const std::vector<int32_t> &func_ids) {
    std::vector<std::vector<uint8_t>> children;
    children.reserve(func_ids.size());
    for (size_t i = 0; i < func_ids.size(); ++i) {
        children.push_back(make_core_child(static_cast<uint8_t>(0xB0 + i), 128));
    }
    const uint8_t orch[64] = {0x5a};
    return make_callable<CoreCallable, CHIP_MAX_TENSOR_ARGS, 1024>(
        nullptr, 0, "orch_entry", orch, sizeof(orch), func_ids.data(), children.data(),
        static_cast<int32_t>(func_ids.size()), "orch_config"
    );
}

const ChipCallable *as_chip(const std::vector<uint8_t> &buffer) {
    return reinterpret_cast<const ChipCallable *>(buffer.data());
}

TEST(CallableFunctionTables, TheLengthIsOnePastTheLargestFuncId) {
    struct Case {
        std::vector<int32_t> func_ids;
        uint32_t expected;
    };
    const Case cases[] = {
        {{0, 1}, 2},
        {{7}, 8},
        {{3, 0, 9}, 10},
        {{static_cast<int32_t>(kFuncIdDomain) - 1}, kFuncIdDomain},
    };
    for (const Case &c : cases) {
        const std::vector<uint8_t> buffer = make_chip_with_children(c.func_ids);
        uint32_t length = 0;
        int32_t bad_func_id = -1;
        ASSERT_TRUE(chip_callable_table_length(as_chip(buffer), kFuncIdDomain, &length, &bad_func_id));
        EXPECT_EQ(length, c.expected) << "largest func_id decides the length, not the child count";
    }
}

// A callable with no children needs no table at all. Registration must not
// publish an empty one: the descriptor then names nothing and the device guard
// short-circuits instead of dereferencing an address no table lives at.
TEST(CallableFunctionTables, ACallableWithNoChildrenNeedsNoTable) {
    const std::vector<uint8_t> buffer = make_chip_with_children({});
    uint32_t length = 0xffffffffu;
    int32_t bad_func_id = -1;
    ASSERT_TRUE(chip_callable_table_length(as_chip(buffer), kFuncIdDomain, &length, &bad_func_id));
    EXPECT_EQ(length, 0u);
}

// The bound is the runtime's, so a child naming a func_id at or past it is
// refused — and named, because the caller logs it and returns before it
// allocates or copies anything.
TEST(CallableFunctionTables, AFuncIdOutsideTheDomainIsRefusedAndNamed) {
    const std::vector<uint8_t> buffer = make_chip_with_children({0, static_cast<int32_t>(kFuncIdDomain)});
    uint32_t length = 0;
    int32_t bad_func_id = -1;
    EXPECT_FALSE(chip_callable_table_length(as_chip(buffer), kFuncIdDomain, &length, &bad_func_id));
    EXPECT_EQ(bad_func_id, static_cast<int32_t>(kFuncIdDomain));
}

TEST(CallableFunctionTables, ANegativeFuncIdIsRefusedAndNamed) {
    const std::vector<uint8_t> buffer = make_chip_with_children({-1});
    uint32_t length = 0;
    int32_t bad_func_id = 0;
    EXPECT_FALSE(chip_callable_table_length(as_chip(buffer), kFuncIdDomain, &length, &bad_func_id));
    EXPECT_EQ(bad_func_id, -1);
}

// The onboard shape: the scratch is patched for a device base, so the object
// view is where each CoreCallable lands on the device and the entry view is
// that object plus the header the code sits behind. Sparse func_ids leave
// holes, and a hole has to read zero — an unmapped func_id is what the
// device-side callable check rejects.
TEST(CallableFunctionTables, TheObjectAndEntryViewsComeFromOnePatchedScratch) {
    const std::vector<int32_t> func_ids = {5, 0, 2};
    const std::vector<uint8_t> buffer = make_chip_with_children(func_ids);
    const ChipCallable *callable = as_chip(buffer);
    const ChipCallableLayout layout = compute_chip_callable_layout(callable);

    uint32_t length = 0;
    int32_t bad_func_id = -1;
    ASSERT_TRUE(chip_callable_table_length(callable, kFuncIdDomain, &length, &bad_func_id));
    ASSERT_EQ(length, 6u);

    std::vector<uint8_t> scratch(layout.total_size);
    std::memcpy(scratch.data(), buffer.data(), layout.total_size);
    patch_chip_callable_scratch_for_device(callable, layout, kDeviceBase, scratch.data());

    std::vector<uint64_t> object(length, 0xdead);
    std::vector<uint64_t> entry(length, 0xdead);
    chip_callable_fill_tables(callable, layout, scratch.data(), kDeviceBase, length, object.data(), entry.data());

    for (int32_t i = 0; i < callable->child_count(); ++i) {
        const uint32_t func_id = static_cast<uint32_t>(callable->child_func_id(i));
        const uint64_t expected_object = kDeviceBase + layout.header_size + callable->child_offset(i);
        EXPECT_EQ(object[func_id], expected_object) << "func_id=" << func_id;
        EXPECT_EQ(entry[func_id], expected_object + CoreCallable::binary_data_offset()) << "func_id=" << func_id;
    }
    for (uint32_t func_id : {1u, 3u, 4u}) {
        EXPECT_EQ(object[func_id], 0u) << "hole func_id=" << func_id;
        EXPECT_EQ(entry[func_id], 0u) << "hole func_id=" << func_id;
    }
}

// The sim shape: no patch for a device base and no copy. The scratch is the
// block, so the object view holds its own host addresses, and each child's
// resolved_addr_ is the host function pointer registration's dlopen produced —
// which is what the entry view must carry, through the same one formula.
TEST(CallableFunctionTables, TheEntryViewCarriesWhateverResolvedAddrHolds) {
    const std::vector<uint8_t> buffer = make_chip_with_children({1, 4});
    const ChipCallable *callable = as_chip(buffer);
    const ChipCallableLayout layout = compute_chip_callable_layout(callable);

    std::vector<uint8_t> scratch(layout.total_size);
    std::memcpy(scratch.data(), buffer.data(), layout.total_size);
    const uint64_t host_base = reinterpret_cast<uint64_t>(scratch.data());
    const uint64_t kHostEntries[] = {0x1111'2222ull, 0x3333'4444ull};
    for (int32_t i = 0; i < callable->child_count(); ++i) {
        auto *child = reinterpret_cast<CoreCallable *>(scratch.data() + layout.header_size + callable->child_offset(i));
        child->set_resolved_addr(kHostEntries[i]);
    }

    std::vector<uint64_t> object(5, 0);
    std::vector<uint64_t> entry(5, 0);
    chip_callable_fill_tables(callable, layout, scratch.data(), host_base, 5, object.data(), entry.data());

    for (int32_t i = 0; i < callable->child_count(); ++i) {
        const uint32_t func_id = static_cast<uint32_t>(callable->child_func_id(i));
        EXPECT_EQ(object[func_id], host_base + layout.header_size + callable->child_offset(i));
        EXPECT_EQ(entry[func_id], kHostEntries[i]) << "func_id=" << func_id;
    }
}

// A runtime whose device consumers resolve the entry out of the object asks for
// no entry view, and must then get a complete object view and no writes past it.
TEST(CallableFunctionTables, TheEntryViewIsOptional) {
    const std::vector<uint8_t> buffer = make_chip_with_children({0});
    const ChipCallable *callable = as_chip(buffer);
    const ChipCallableLayout layout = compute_chip_callable_layout(callable);

    std::vector<uint8_t> scratch(layout.total_size);
    std::memcpy(scratch.data(), buffer.data(), layout.total_size);
    patch_chip_callable_scratch_for_device(callable, layout, kDeviceBase, scratch.data());

    std::vector<uint64_t> object(1, 0xdead);
    chip_callable_fill_tables(callable, layout, scratch.data(), kDeviceBase, 1, object.data(), nullptr);
    EXPECT_EQ(object[0], kDeviceBase + layout.header_size + callable->child_offset(0));
}

// What makes one pooled block a consistent owner for every callable_id that
// dedups onto it: the content hash covers the child func_ids and offsets the
// tables are derived from, so identical bytes at one base give identical
// tables. Differing bytes give a different hash, so they never share a block.
TEST(CallableFunctionTables, IdenticalBytesYieldIdenticalTables) {
    const std::vector<uint8_t> first = make_chip_with_children({2, 6});
    const std::vector<uint8_t> second = make_chip_with_children({2, 6});
    ASSERT_EQ(first, second);

    const ChipCallableLayout first_layout = compute_chip_callable_layout(as_chip(first));
    const ChipCallableLayout second_layout = compute_chip_callable_layout(as_chip(second));
    ASSERT_EQ(first_layout.content_hash, second_layout.content_hash);

    auto build = [](const std::vector<uint8_t> &buffer, const ChipCallableLayout &layout) {
        std::vector<uint8_t> scratch(layout.total_size);
        std::memcpy(scratch.data(), buffer.data(), layout.total_size);
        patch_chip_callable_scratch_for_device(as_chip(buffer), layout, kDeviceBase, scratch.data());
        std::vector<uint64_t> object(7, 0);
        std::vector<uint64_t> entry(7, 0);
        chip_callable_fill_tables(as_chip(buffer), layout, scratch.data(), kDeviceBase, 7, object.data(), entry.data());
        object.insert(object.end(), entry.begin(), entry.end());
        return object;
    };
    EXPECT_EQ(build(first, first_layout), build(second, second_layout));

    const std::vector<uint8_t> other = make_chip_with_children({2, 7});
    EXPECT_NE(compute_chip_callable_layout(as_chip(other)).content_hash, first_layout.content_hash)
        << "a different func_id domain must not dedup onto the same block";
}

}  // namespace
