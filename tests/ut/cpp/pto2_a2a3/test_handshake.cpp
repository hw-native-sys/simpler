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
 * Unit tests for Handshake Protocol macros.
 *
 * Tests the ACK/FIN dual-state register encoding/decoding defined in
 * platform_config.h: MAKE_ACK_VALUE, MAKE_FIN_VALUE, EXTRACT_TASK_ID,
 * EXTRACT_TASK_STATE, and reserved ID guards.
 */

#include <gtest/gtest.h>
#include "common/platform_config.h"

// =============================================================================
// ACK value encoding (bit 31 = 0)
// =============================================================================

TEST(HandshakeProtocol, MakeAckValue_Bit31Clear) {
    uint64_t ack = MAKE_ACK_VALUE(42);
    // bit 31 must be 0 for ACK
    EXPECT_EQ(ack & TASK_STATE_MASK, 0u);
    EXPECT_EQ(EXTRACT_TASK_STATE(ack), TASK_ACK_STATE);
}

TEST(HandshakeProtocol, MakeAckValue_PreservesTaskId) {
    for (int task_id : {0, 1, 100, 1000000, 0x7FFFFFFF}) {
        uint64_t ack = MAKE_ACK_VALUE(task_id);
        EXPECT_EQ(EXTRACT_TASK_ID(ack), task_id);
    }
}

// =============================================================================
// FIN value encoding (bit 31 = 1)
// =============================================================================

TEST(HandshakeProtocol, MakeFinValue_Bit31Set) {
    uint64_t fin = MAKE_FIN_VALUE(42);
    // bit 31 must be 1 for FIN
    EXPECT_NE(fin & TASK_STATE_MASK, 0u);
    EXPECT_EQ(EXTRACT_TASK_STATE(fin), TASK_FIN_STATE);
}

TEST(HandshakeProtocol, MakeFinValue_PreservesTaskId) {
    for (int task_id : {0, 1, 100, 1000000, 0x7FFFFFFF}) {
        uint64_t fin = MAKE_FIN_VALUE(task_id);
        EXPECT_EQ(EXTRACT_TASK_ID(fin), task_id);
    }
}

// =============================================================================
// Roundtrip: encode -> decode
// =============================================================================

TEST(HandshakeProtocol, AckRoundtrip) {
    for (int id = 0; id < 1000; id++) {
        uint64_t ack = MAKE_ACK_VALUE(id);
        EXPECT_EQ(EXTRACT_TASK_ID(ack), id);
        EXPECT_EQ(EXTRACT_TASK_STATE(ack), TASK_ACK_STATE);
    }
}

TEST(HandshakeProtocol, FinRoundtrip) {
    for (int id = 0; id < 1000; id++) {
        uint64_t fin = MAKE_FIN_VALUE(id);
        EXPECT_EQ(EXTRACT_TASK_ID(fin), id);
        EXPECT_EQ(EXTRACT_TASK_STATE(fin), TASK_FIN_STATE);
    }
}

// =============================================================================
// Reserved task IDs
// =============================================================================

TEST(HandshakeProtocol, ReservedIdGuard_IdleAndExit) {
    // IDLE and EXIT task IDs must be distinct
    EXPECT_NE(AICORE_IDLE_TASK_ID, AICORE_EXIT_TASK_ID);

    // Both must be in the reserved range (high values)
    EXPECT_GT(AICORE_IDLE_TASK_ID, 0x7FFFFFF0u);
    EXPECT_GT(AICORE_EXIT_TASK_ID, 0x7FFFFFF0u);
}

TEST(HandshakeProtocol, ReservedIdGuard_IdleValue) {
    // AICORE_IDLE_VALUE should encode IDLE_TASK_ID with FIN state
    uint64_t idle = AICORE_IDLE_VALUE;
    EXPECT_EQ(EXTRACT_TASK_STATE(idle), TASK_FIN_STATE);
    EXPECT_EQ(EXTRACT_TASK_ID(idle), (int)AICORE_IDLE_TASK_ID);
}

TEST(HandshakeProtocol, ReservedIdGuard_ExitValue) {
    // AICORE_EXITED_VALUE should encode EXIT_TASK_ID with FIN state
    uint64_t exited = AICORE_EXITED_VALUE;
    EXPECT_EQ(EXTRACT_TASK_STATE(exited), TASK_FIN_STATE);
    EXPECT_EQ(EXTRACT_TASK_ID(exited), (int)AICORE_EXIT_TASK_ID);
}

// =============================================================================
// Exit signal
// =============================================================================

TEST(HandshakeProtocol, ExitSignalValue) {
    // AICORE_EXIT_SIGNAL is a special dispatch value
    EXPECT_EQ(AICORE_EXIT_SIGNAL, 0x7FFFFFF0u);
}

// =============================================================================
// Invalid task ID sentinel
// =============================================================================

TEST(HandshakeProtocol, InvalidTaskSentinel) { EXPECT_EQ(AICPU_TASK_INVALID, -1); }
