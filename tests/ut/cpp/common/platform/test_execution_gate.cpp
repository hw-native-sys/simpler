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

#include "common/execution_gate.h"
#include "common/kernel_args.h"

TEST(ExecutionGate, OnlyMatchingTerminalDecisionsPermitEntry) {
    EXPECT_EQ(execution_gate_outcome(1, 1, 17, 9, 17, 9), ExecutionGateOutcome::Go);
    EXPECT_EQ(execution_gate_outcome(1, 2, 17, 9, 17, 9), ExecutionGateOutcome::Cancel);
    EXPECT_EQ(execution_gate_outcome(1, 0, 17, 9, 17, 9), ExecutionGateOutcome::Invalid);
    EXPECT_EQ(execution_gate_outcome(1, 3, 17, 9, 17, 9), ExecutionGateOutcome::Invalid);
    EXPECT_EQ(execution_gate_outcome(2, 1, 17, 9, 17, 9), ExecutionGateOutcome::Invalid);
}

TEST(ExecutionGate, RejectsStaleIdentityAndRecycledGenerationEvenForCancellation) {
    for (uint32_t decision : {1u, 2u}) {
        EXPECT_EQ(execution_gate_outcome(1, decision, 16, 9, 17, 9), ExecutionGateOutcome::Invalid);
        EXPECT_EQ(execution_gate_outcome(1, decision, 17, 8, 17, 9), ExecutionGateOutcome::Invalid);
        EXPECT_EQ(execution_gate_outcome(1, decision, 0, 9, 0, 9), ExecutionGateOutcome::Invalid);
        EXPECT_EQ(execution_gate_outcome(1, decision, 17, 0, 17, 0), ExecutionGateOutcome::Invalid);
    }
}

TEST(ExecutionGate, FreshRecordIsPendingAndHasNoEntryReports) {
    ExecutionGateRecord record{};
    EXPECT_EQ(record.header.version, EXECUTION_GATE_VERSION);
    EXPECT_EQ(record.header.decision, static_cast<uint32_t>(ExecutionGateDecision::Pending));
    EXPECT_EQ(record.aicpu.outcome, 0u);
    EXPECT_EQ(record.aic.outcome, 0u);
    EXPECT_EQ(record.aiv.outcome, 0u);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(&record.header) % 64, 0u);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(&record.aicpu) - reinterpret_cast<uintptr_t>(&record.header), 64u);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(&record.aic) - reinterpret_cast<uintptr_t>(&record.aicpu), 64u);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(&record.aiv) - reinterpret_cast<uintptr_t>(&record.aic), 64u);
}

TEST(ExecutionGate, CoreProjectionCarriesTheSameGateAndIdentity) {
    KernelArgs args{};
    AicoreLaunchArgs projected{};
    fill_shared_launch_args(projected, args);
    EXPECT_EQ(projected.execution_gate_address, 0u);
    EXPECT_EQ(projected.execution_gate_identity, 0u);
    EXPECT_EQ(projected.execution_gate_generation, 0u);
    args.execution_gate_address = 0x123400;
    args.execution_gate_identity = 17;
    args.execution_gate_generation = 9;
    fill_shared_launch_args(projected, args);
    EXPECT_EQ(projected.execution_gate_address, args.execution_gate_address);
    EXPECT_EQ(projected.execution_gate_identity, args.execution_gate_identity);
    EXPECT_EQ(projected.execution_gate_generation, args.execution_gate_generation);
    EXPECT_GE(LAUNCH_ENVELOPE_HEADER_BYTES, sizeof(KernelArgs));
    EXPECT_EQ(LAUNCH_ENVELOPE_HEADER_BYTES % 64, 0u);
}
