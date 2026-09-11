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

#include <cstring>

#include "host/kernel_ctx_control.h"

namespace {

SimplerKernelCtxControl make_configure(
    uint32_t mode = SIMPLER_MODE_PROGRAM, uint64_t gm_heap = 0, uint64_t gm_sm = 0, uint64_t runtime_arena = 0
) {
    SimplerKernelCtxControl control{};
    control.abi_version = SIMPLER_KERNEL_CTX_CONTROL_ABI_VERSION;
    control.struct_size = sizeof(SimplerKernelCtxControl);
    control.action = SIMPLER_KERNEL_CTX_CONFIGURE;
    control.mode = mode;
    control.gm_heap_bytes = gm_heap;
    control.gm_sm_bytes = gm_sm;
    control.runtime_arena_bytes = runtime_arena;
    return control;
}

SimplerKernelCtxControl make_freeze() {
    SimplerKernelCtxControl control{};
    control.abi_version = SIMPLER_KERNEL_CTX_CONTROL_ABI_VERSION;
    control.struct_size = sizeof(SimplerKernelCtxControl);
    control.action = SIMPLER_KERNEL_CTX_FREEZE;
    return control;
}

constexpr KernelCtxControlState::Environment kFresh{false, false};
constexpr KernelCtxControlState::Environment kInitNoCapacity{true, false};
constexpr KernelCtxControlState::Environment kInitWithCapacity{true, true};
constexpr KernelCtxControlState::Capabilities kStub{false, false};
constexpr KernelCtxControlState::Capabilities kFull{true, true};

TEST(KernelCtxControlWire, LayoutIsPinned) {
    EXPECT_EQ(sizeof(SimplerKernelCtxControl), 72u);
    EXPECT_EQ(offsetof(SimplerKernelCtxControl, abi_version), 0u);
    EXPECT_EQ(offsetof(SimplerKernelCtxControl, struct_size), 4u);
    EXPECT_EQ(offsetof(SimplerKernelCtxControl, action), 8u);
    EXPECT_EQ(offsetof(SimplerKernelCtxControl, mode), 12u);
    EXPECT_EQ(offsetof(SimplerKernelCtxControl, gm_heap_bytes), 16u);
    EXPECT_EQ(offsetof(SimplerKernelCtxControl, gm_sm_bytes), 24u);
    EXPECT_EQ(offsetof(SimplerKernelCtxControl, runtime_arena_bytes), 32u);
    EXPECT_EQ(offsetof(SimplerKernelCtxControl, reserved), 40u);
    EXPECT_EQ(SIMPLER_MODE_PROGRAM, 0);
    EXPECT_EQ(SIMPLER_MODE_KERNEL, 1);
    EXPECT_EQ(SIMPLER_KERNEL_CTX_CONFIGURE, 1);
    EXPECT_EQ(SIMPLER_KERNEL_CTX_FREEZE, 2);
}

TEST(KernelCtxControlStructural, NullControlRejected) {
    KernelCtxControlState state;
    EXPECT_EQ(state.apply(nullptr, kFresh, kFull), PTO_RUNTIME_ERR_INTERNAL);
}

TEST(KernelCtxControlStructural, WrongVersionRejected) {
    KernelCtxControlState state;
    auto control = make_configure();
    control.abi_version = 2;
    EXPECT_EQ(state.apply(&control, kFresh, kFull), PTO_RUNTIME_ERR_INTERNAL);
}

TEST(KernelCtxControlStructural, StructSizeMustMatchExactly) {
    KernelCtxControlState state;
    auto control = make_configure();
    control.struct_size = sizeof(SimplerKernelCtxControl) + 8;
    EXPECT_EQ(state.apply(&control, kFresh, kFull), PTO_RUNTIME_ERR_INTERNAL);
    control.struct_size = sizeof(SimplerKernelCtxControl) - 8;
    EXPECT_EQ(state.apply(&control, kFresh, kFull), PTO_RUNTIME_ERR_INTERNAL);
}

TEST(KernelCtxControlStructural, UnknownActionRejected) {
    KernelCtxControlState state;
    auto control = make_configure();
    control.action = 3;
    EXPECT_EQ(state.apply(&control, kFresh, kFull), PTO_RUNTIME_ERR_INTERNAL);
    control.action = 0;
    EXPECT_EQ(state.apply(&control, kFresh, kFull), PTO_RUNTIME_ERR_INTERNAL);
}

TEST(KernelCtxControlStructural, UnknownModeRejected) {
    KernelCtxControlState state;
    auto control = make_configure(/*mode=*/2);
    EXPECT_EQ(state.apply(&control, kFresh, kFull), PTO_RUNTIME_ERR_INTERNAL);
}

TEST(KernelCtxControlStructural, NonzeroReservedRejected) {
    for (size_t i = 0; i < 4; ++i) {
        KernelCtxControlState state;
        auto control = make_configure();
        control.reserved[i] = 1;
        EXPECT_EQ(state.apply(&control, kFresh, kFull), PTO_RUNTIME_ERR_INTERNAL) << "reserved[" << i << "]";
    }
}

TEST(KernelCtxControlStructural, FreezePayloadMustBeZero) {
    KernelCtxControlState state;
    auto control = make_freeze();
    control.mode = SIMPLER_MODE_KERNEL;
    EXPECT_EQ(state.apply(&control, kInitWithCapacity, kFull), PTO_RUNTIME_ERR_INTERNAL);
    control = make_freeze();
    control.gm_heap_bytes = 4096;
    EXPECT_EQ(state.apply(&control, kInitWithCapacity, kFull), PTO_RUNTIME_ERR_INTERNAL);
}

TEST(KernelCtxControlOrdering, ConfigureBeforeInitAcceptedAndIdempotent) {
    KernelCtxControlState state;
    auto control = make_configure(SIMPLER_MODE_KERNEL, 1 << 20, 0, 1 << 16);
    EXPECT_EQ(state.apply(&control, kFresh, kFull), 0);
    EXPECT_EQ(state.apply(&control, kFresh, kFull), 0);
    EXPECT_EQ(state.configured_mode(), SIMPLER_MODE_KERNEL);
}

TEST(KernelCtxControlOrdering, ConfigureWithDifferentTupleRejected) {
    KernelCtxControlState state;
    auto first = make_configure(SIMPLER_MODE_KERNEL, 1 << 20);
    EXPECT_EQ(state.apply(&first, kFresh, kFull), 0);
    auto second = make_configure(SIMPLER_MODE_KERNEL, 2 << 20);
    EXPECT_EQ(state.apply(&second, kFresh, kFull), PTO_RUNTIME_ERR_INVALID_STATE);
    auto third = make_configure(SIMPLER_MODE_PROGRAM, 1 << 20);
    EXPECT_EQ(state.apply(&third, kFresh, kFull), PTO_RUNTIME_ERR_INVALID_STATE);
}

TEST(KernelCtxControlOrdering, ConfigureAfterInitRejected) {
    KernelCtxControlState state;
    auto control = make_configure();
    EXPECT_EQ(state.apply(&control, kInitNoCapacity, kFull), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_EQ(state.apply(&control, kInitWithCapacity, kFull), PTO_RUNTIME_ERR_INVALID_STATE);
}

TEST(KernelCtxControlOrdering, FreezeRequiresInitAndCapacity) {
    KernelCtxControlState state;
    auto configure = make_configure(SIMPLER_MODE_KERNEL);
    ASSERT_EQ(state.apply(&configure, kFresh, kFull), 0);
    auto freeze = make_freeze();
    EXPECT_EQ(state.apply(&freeze, kFresh, kFull), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_EQ(state.apply(&freeze, kInitNoCapacity, kFull), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_FALSE(state.frozen());
}

TEST(KernelCtxControlOrdering, FreezeRequiresKernelConfigure) {
    KernelCtxControlState state;
    auto freeze = make_freeze();
    EXPECT_EQ(state.apply(&freeze, kInitWithCapacity, kFull), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_FALSE(state.frozen());
}

TEST(KernelCtxControlOrdering, FreezeAfterProgramConfigureRejected) {
    KernelCtxControlState state;
    auto configure = make_configure();
    ASSERT_EQ(state.apply(&configure, kFresh, kFull), 0);
    auto freeze = make_freeze();
    EXPECT_EQ(state.apply(&freeze, kInitWithCapacity, kFull), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_FALSE(state.frozen());
}

TEST(KernelCtxControlOrdering, FreezeSucceedsOnceThenFailsClosed) {
    KernelCtxControlState state;
    auto configure = make_configure(SIMPLER_MODE_KERNEL);
    ASSERT_EQ(state.apply(&configure, kFresh, kFull), 0);
    auto freeze = make_freeze();
    EXPECT_EQ(state.apply(&freeze, kInitWithCapacity, kFull), 0);
    EXPECT_TRUE(state.frozen());
    EXPECT_EQ(state.apply(&freeze, kInitWithCapacity, kFull), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_TRUE(state.frozen());
}

TEST(KernelCtxControlOrdering, StubNeverReachesASuccessfulFreeze) {
    // A stub rejects the kernel CONFIGURE, so no ordering sequence available
    // on it can satisfy FREEZE's kernel-configure precondition.
    KernelCtxControlState state;
    auto configure = make_configure(SIMPLER_MODE_KERNEL);
    ASSERT_EQ(state.apply(&configure, kFresh, kStub), PTO_RUNTIME_ERR_UNSUPPORTED);
    auto freeze = make_freeze();
    EXPECT_EQ(state.apply(&freeze, kInitWithCapacity, kStub), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_FALSE(state.frozen());
}

TEST(KernelCtxControlCapability, StubRejectsKernelModeAfterSharedChecks) {
    KernelCtxControlState state;
    auto control = make_configure(SIMPLER_MODE_KERNEL);
    EXPECT_EQ(state.apply(&control, kFresh, kStub), PTO_RUNTIME_ERR_UNSUPPORTED);
    // Ordering still wins over the capability split: the same request after
    // init reports the ordering violation, not unsupported.
    EXPECT_EQ(state.apply(&control, kInitNoCapacity, kStub), PTO_RUNTIME_ERR_INVALID_STATE);
}

TEST(KernelCtxControlCapability, StubRejectsNonzeroCapacityIntent) {
    KernelCtxControlState state;
    auto control = make_configure(SIMPLER_MODE_PROGRAM, /*gm_heap=*/4096);
    EXPECT_EQ(state.apply(&control, kFresh, kStub), PTO_RUNTIME_ERR_UNSUPPORTED);
}

TEST(KernelCtxControlCapability, StubAcceptsProgramDefaultConfigure) {
    KernelCtxControlState state;
    auto control = make_configure();
    EXPECT_EQ(state.apply(&control, kFresh, kStub), 0);
    EXPECT_EQ(state.apply(&control, kFresh, kStub), 0);
    EXPECT_EQ(state.configured_mode(), SIMPLER_MODE_PROGRAM);
}

TEST(KernelCtxControlCapability, RejectionMutatesNothing) {
    KernelCtxControlState state;
    auto kernel_configure = make_configure(SIMPLER_MODE_KERNEL);
    EXPECT_EQ(state.apply(&kernel_configure, kFresh, kStub), PTO_RUNTIME_ERR_UNSUPPORTED);
    // The rejected kernel tuple was not recorded: a later program-mode
    // CONFIGURE is a first CONFIGURE, not a tuple change.
    auto program_configure = make_configure();
    EXPECT_EQ(state.apply(&program_configure, kFresh, kStub), 0);
}

}  // namespace
