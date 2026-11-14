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
#include "platform_comm/ccu/ccu_completion.h"
#include "platform_comm/ccu/ccu_ready.h"
namespace {
using namespace simpler::ccu;
TEST(CcuCompletion, ClearsDoneBeforePublishingCompletionAndSupportsReuse) {
    uint64_t event = 0;
    const auto address = reinterpret_cast<uint64_t>(&event);
    EXPECT_EQ(poll_completion(address, 1).state, CompletionPollState::pending);
    for (int i = 0; i < 1000; ++i) {
        event = 1;
        EXPECT_EQ(poll_completion(address, 1).state, CompletionPollState::ready);
        EXPECT_EQ(event, 0U);
        EXPECT_EQ(poll_completion(address, 1).state, CompletionPollState::pending);
    }
}
struct StuckIo {
    static uint64_t load(uint64_t) { return 1; }
    static void clear(uint64_t) {}
};
TEST(CcuCompletion, RejectsUnconfirmedClearAndInvalidMapping) {
    EXPECT_EQ(poll_completion<StuckIo>(8, 1).state, CompletionPollState::protocol_error);
    EXPECT_EQ(poll_completion(0, 1).state, CompletionPollState::protocol_error);
    EXPECT_EQ(poll_completion(8, 2).state, CompletionPollState::protocol_error);
    EXPECT_EQ(poll_completion(9, 1).state, CompletionPollState::protocol_error);
}

struct ReadyFixture {
    uint64_t events[2]{1, 0};
    SimplerCcuControl controls[2]{};
    SimplerCcuWorkspace workspace{};

    ReadyFixture() {
        workspace.struct_size = sizeof(workspace);
        workspace.ccu_count = 2;
        workspace.run_generation = 7;
        workspace.controls_addr = reinterpret_cast<uint64_t>(controls);
        workspace.control_size = sizeof(controls[0]);
        for (uint32_t i = 0; i < 2; ++i) {
            controls[i].struct_size = sizeof(controls[i]);
            controls[i].ccu_id = i;
            controls[i].run_generation = 7;
            controls[i].ready = {reinterpret_cast<uint64_t>(&events[i]), 1};
        }
    }

    uint64_t address() { return reinterpret_cast<uint64_t>(&workspace); }
};

TEST(CcuReady, ConsumesEveryLocalReadyAndConfirmsClear) {
    ReadyFixture fixture;
    uint32_t spins = 0;
    EXPECT_EQ(
        wait_local_ready(
            fixture.address(),
            [] {
                return false;
            },
            [&] {
                EXPECT_EQ(fixture.events[0], 0U);
                fixture.events[1] = 1;
                ++spins;
            }
        ),
        0
    );
    EXPECT_EQ(spins, 1U);
    EXPECT_EQ(fixture.events[0], 0U);
    EXPECT_EQ(fixture.events[1], 0U);
}

TEST(CcuReady, ReportsTimeoutInvalidGenerationAndUnconfirmedClear) {
    ReadyFixture fixture;
    EXPECT_EQ(
        wait_local_ready(
            fixture.address(),
            [] {
                return true;
            },
            [] {}
        ),
        -2
    );
    fixture.controls[0].run_generation = 6;
    fixture.events[0] = 1;
    EXPECT_EQ(
        wait_local_ready(
            fixture.address(),
            [] {
                return false;
            },
            [] {}
        ),
        -1
    );
    EXPECT_EQ(fixture.events[0], 1U);
    fixture.controls[0].run_generation = 7;
    EXPECT_EQ(
        wait_local_ready<StuckIo>(
            fixture.address(),
            [] {
                return false;
            },
            [] {}
        ),
        -1
    );
    EXPECT_EQ(
        wait_local_ready(
            0,
            [] {
                return false;
            },
            [] {}
        ),
        0
    );
}
}  // namespace
