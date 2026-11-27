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
#include <functional>

#include "native_run_context.h"
#include "common/support/native_run_execution_peer.h"

namespace {

// NativeRunContext is a template over the runner only for its execution types;
// publish_acceptance touches neither, so the smallest runner that names them is
// enough to exercise the acceptance gate.
struct FakeRunner {
    struct PreparedExecution {};
    struct ActiveExecution {};
};

struct CancelRunner : FakeRunner {
    std::function<int()> poll;
    int timeout_ms{-1};
    int request_rc{0};
    int requests{0};
    int drains{0};
    uint32_t requested_slot{0};
    uint64_t requested_epoch{0};
    int poll_execution(const ActiveExecution &) { return poll(); }
    int peer_cancel_wait_timeout_ms() const { return timeout_ms; }
    int request_peer_cancel(uint32_t slot, uint64_t epoch) {
        ++requests;
        requested_slot = slot;
        requested_epoch = epoch;
        return request_rc;
    }
    DrainOutcome drain_execution(ActiveExecution &) {
        ++drains;
        return DrainOutcome::device_error(-108, true);
    }
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

TEST(NativeRunAcceptanceTest, BlockingObserverDeliversPeerCancellationBeforeDraining) {
    CancelRunner runner;
    int32_t request = 0;
    NativeRunContext<CancelRunner> context(&runner, CallConfig{}, 0, make_descriptor(nullptr), nullptr);
    context.active_execution = std::make_unique<CancelRunner::ActiveExecution>();
    context.peer_cancel_request = &request;
    runner.poll = [&] {
        if (runner.requests == 0) {
            __atomic_store_n(&request, 1, __ATOMIC_RELEASE);
            return SIMPLER_NATIVE_RUN_POLL_NOT_READY;
        }
        return SIMPLER_NATIVE_RUN_POLL_COMPLETE;
    };
    const auto result = context.drain_with_peer_cancel();
    EXPECT_EQ(runner.requests, 1);
    EXPECT_EQ(runner.requested_slot, kPipelineSlot);
    EXPECT_EQ(runner.requested_epoch, 11u);
    EXPECT_EQ(runner.drains, 1);
    EXPECT_EQ(result.device_rc, -108);
    EXPECT_TRUE(result.physical_complete);
    EXPECT_EQ(context.poll_peer_cancel(), 0);
    EXPECT_EQ(runner.requests, 1);
}

TEST(NativeRunAcceptanceTest, UndeliveredCancellationAndExpiredWaitDoNotProveRetirement) {
    for (bool transfer_failure : {false, true}) {
        CancelRunner runner;
        int32_t request = transfer_failure ? 1 : 0;
        NativeRunContext<CancelRunner> context(&runner, CallConfig{}, 0, make_descriptor(nullptr), nullptr);
        context.active_execution = std::make_unique<CancelRunner::ActiveExecution>();
        context.peer_cancel_request = &request;
        runner.poll = [] {
            return SIMPLER_NATIVE_RUN_POLL_NOT_READY;
        };
        runner.timeout_ms = 0;
        runner.request_rc = -123;
        const auto result = context.drain_with_peer_cancel();
        EXPECT_EQ(result.device_rc, transfer_failure ? -123 : SIMPLER_NATIVE_RUN_WAIT_INCOMPLETE);
        EXPECT_FALSE(result.physical_complete);
        EXPECT_EQ(runner.drains, 0);
        EXPECT_FALSE(context.peer_cancel_sent);
        runner.request_rc = 0;
        runner.timeout_ms = -1;
        __atomic_store_n(&request, 1, __ATOMIC_RELEASE);
        runner.poll = [&] {
            return context.peer_cancel_sent ? SIMPLER_NATIVE_RUN_POLL_COMPLETE : SIMPLER_NATIVE_RUN_POLL_NOT_READY;
        };
        const auto retry = context.drain_with_peer_cancel();
        EXPECT_EQ(retry.device_rc, -108);
        EXPECT_TRUE(retry.physical_complete);
        EXPECT_EQ(runner.drains, 1);
    }
}
