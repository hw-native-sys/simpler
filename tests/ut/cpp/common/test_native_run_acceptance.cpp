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

#include <cstddef>
#include <cstdint>

#include "native_run_context.h"
#include "native_run_execution_test_peer.h"

namespace {

// NativeRunContext is a template over the runner only for its execution types;
// publish_acceptance touches neither, so the smallest runner that names them is
// enough to exercise the acceptance gate.
struct FakeRunner {
    struct PreparedExecution {};
    struct ActiveExecution {};
};

using FakeRunContext = NativeRunContext<FakeRunner>;

constexpr uint32_t kPipelineSlot = 1;
constexpr int32_t kAcceptedValue = 17;

NativeRunDescriptor make_descriptor(volatile int32_t *accepted_state) {
    NativeRunDescriptor descriptor{};
    descriptor.run_epoch = 11;
    descriptor.generation = 13;
    descriptor.dispatch_id = 17;
    descriptor.pipeline_slot = kPipelineSlot;
    descriptor.accepted_state = accepted_state;
    descriptor.accepted_value = kAcceptedValue;
    descriptor.flags = 0;
    return descriptor;
}

LaunchReceipt complete_receipt(const NativeRunIdentity &identity) {
    LaunchTransactionResult launched = exact_launch_transaction(
        identity, NativeRunExecutionTestPeer::mint(identity),
        []() {
            return 0;
        },
        []() {
            return 0;
        }
    );
    EXPECT_EQ(launched.progress, LaunchProgress::Complete);
    return std::move(launched.receipt);
}

}  // namespace

TEST(NativeRunDescriptorAbi, FlagsSitAfterAcceptedValueAndStayZeroByDefault) {
    NativeRunDescriptor descriptor{};
    EXPECT_EQ(descriptor.flags, 0u);
    descriptor.flags = PTO_NATIVE_RUN_FLAG_INTERNAL_PREWARM;
    EXPECT_EQ(descriptor.flags, PTO_NATIVE_RUN_FLAG_INTERNAL_PREWARM);
    EXPECT_EQ(offsetof(NativeRunDescriptor, flags), offsetof(NativeRunDescriptor, accepted_value) + sizeof(int32_t));
}

TEST(NativeRunAcceptanceTest, MatchingReceiptStoresTheAcceptedValue) {
    volatile int32_t accepted_state = 0;
    FakeRunner runner;
    CallConfig config{};
    FakeRunContext context(&runner, config, 0, make_descriptor(&accepted_state), nullptr);

    EXPECT_TRUE(context.publish_acceptance(complete_receipt(context.identity())));
    EXPECT_EQ(__atomic_load_n(&accepted_state, __ATOMIC_ACQUIRE), kAcceptedValue);
}

TEST(NativeRunAcceptanceTest, StaleReceiptCannotPublishAcceptance) {
    volatile int32_t accepted_state = 5;
    FakeRunner runner;
    CallConfig config{};
    FakeRunContext context(&runner, config, 0, make_descriptor(&accepted_state), nullptr);

    NativeRunIdentity stale = context.identity();
    stale.generation++;

    EXPECT_FALSE(context.publish_acceptance(complete_receipt(stale)));
    EXPECT_EQ(__atomic_load_n(&accepted_state, __ATOMIC_ACQUIRE), 5);
}

TEST(NativeRunAcceptanceTest, AbsentAcceptedStateIsAccepted) {
    FakeRunner runner;
    CallConfig config{};
    FakeRunContext context(&runner, config, 0, make_descriptor(nullptr), nullptr);

    EXPECT_TRUE(context.publish_acceptance(complete_receipt(context.identity())));
}
