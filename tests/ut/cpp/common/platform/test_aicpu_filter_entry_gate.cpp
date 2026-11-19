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

#include "support/aicpu_entry_gate_cases.h"
TEST(AicpuFilterEntryGate, InitializationPrecedesEverySurvivorAcrossSerialInvocations) {
    entry_gate_test::check_cycles(entry_gate_test::filter);
}
TEST(AicpuFilterEntryGate, InvalidInputNeverInvokesInitializer) {
    entry_gate_test::initialized = 0;
    EXPECT_FALSE(platform_aicpu_affinity_gate_filter(nullptr, 2, 3, entry_gate_test::initialize_entry));
    EXPECT_FALSE(entry_gate_test::filter(2, 1, entry_gate_test::initialize_entry));
    EXPECT_EQ(entry_gate_test::initialized.load(), 0);
}

TEST(AicpuFilterEntryGate, NativeIndicesIncludeFilteredCallbacksAndFinishExactlyOnce) {
    entry_gate_test::check_native(entry_gate_test::native_filter, false, 0);
}
TEST(AicpuFilterEntryGate, OneInvalidParticipantRejectsEveryoneBeforeWork) {
    entry_gate_test::check_native(entry_gate_test::native_filter, true, 0);
}
TEST(AicpuFilterEntryGate, InitializationFailureReachesEveryCallback) {
    entry_gate_test::check_native(entry_gate_test::native_filter, false, -29);
}
TEST(AicpuFilterEntryGate, UntrustedCountDoesNotInitializeOrClaimIdentity) {
    entry_gate_test::NativeState state;
    const auto result =
        entry_gate_test::native_filter(4, MAX_GATE_THREADS + 1, {entry_gate_test::initialize_native, &state});
    EXPECT_EQ(result.callback_index, -1);
    EXPECT_NE(result.status, 0);
    EXPECT_FALSE(result.execute);
    EXPECT_EQ(state.initializations, 0);
}
