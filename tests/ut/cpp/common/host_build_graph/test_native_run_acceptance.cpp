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

#include "native_run_context.h"
#include "host_build_graph/runtime_status.h"
#include "common/support/native_run_execution_peer.h"

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
    return descriptor;
}

LaunchReceipt complete_receipt(const NativeRunIdentity &identity) {
    LaunchTransactionResult launched = exact_launch_transaction(
        identity, NativeRunExecutionTestPeer::mint(identity),
        [](LaunchProgressSink &sink) {
            sink.mark_submitted();
            return 0;
        },
        [](LaunchProgressSink &sink) {
            sink.mark_submitted();
            return 0;
        }
    );
    EXPECT_EQ(launched.progress, LaunchProgress::Complete);
    return std::move(launched.receipt);
}

}  // namespace

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

TEST(NativeRunDrainTest, UnconfirmedExecutionRetainsItsCarrierAndPreservesTheFirstFailure) {
    FakeRunner runner;
    FakeRunContext context(&runner, CallConfig{}, 0, make_descriptor(nullptr), nullptr);
    context.completion_rc = 0;

    EXPECT_EQ(context.check_finalize_drain(/*launched=*/false), 0);
    EXPECT_EQ(context.check_finalize_drain(/*launched=*/true), PTO_RUNTIME_ERR_INTERNAL);
    context.record_drain_outcome(DrainOutcome::device_error(-SIMPLER_ERROR_SCHEDULER_TIMEOUT));
    EXPECT_EQ(context.check_finalize_drain(true), -SIMPLER_ERROR_SCHEDULER_TIMEOUT);
    EXPECT_FALSE(context.physical_complete);

    context.record_drain_outcome(DrainOutcome::device_complete(/*diagnostics_rc=*/PTO_RUNTIME_ERR_INTERNAL));
    EXPECT_EQ(context.check_finalize_drain(true), 0);
    EXPECT_EQ(context.completion_rc, -SIMPLER_ERROR_SCHEDULER_TIMEOUT);
    EXPECT_TRUE(context.physical_complete);

    context.record_drain_outcome(DrainOutcome::device_error(PTO_RUNTIME_ERR_INTERNAL));
    EXPECT_TRUE(context.physical_complete);
    EXPECT_EQ(context.completion_rc, -SIMPLER_ERROR_SCHEDULER_TIMEOUT);
}
