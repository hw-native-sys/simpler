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

#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "host/run_caller_order.h"

namespace {
using Role = RunCompletionFence::StreamRole;
constexpr NativeRunIdentity kRun{1, 1, 1, 0};
constexpr NativeRunIdentity kNext{2, 1, 2, 0};

class CallerOrderTest : public testing::Test {
protected:
    int tokens[16]{};
    int created = 0, destroyed = 0, operation = 0, fail_at = 0;
    int sync_error = 0;
    bool completed = false;
    std::vector<std::string> trace;
    void *caller = &tokens[10], *cpu = &tokens[11], *core = &tokens[12];
    RunCompletionFence::DeviceEventOps ops() {
        return {
            [this](void **out) {
                *out = &tokens[created++];
                return 0;
            },
            [this](void *, void *stream) {
                return submit("record", stream);
            },
            [this](void *, bool *out) {
                *out = completed;
                return 0;
            },
            [this](void *, int) {
                trace.push_back("event_wait");
                return completed ? 0 : -91;
            },
            [this](void *) {
                ++destroyed;
                return 0;
            },
        };
    }
    int submit(const char *op, void *stream) {
        trace.push_back(std::string(op) + (stream == caller ? " caller" : stream == cpu ? " cpu" : " core"));
        return ++operation == fail_at ? -77 : 0;
    }
    RunCompletionFence fence{ops()};
    RunCallerOrder order{
        ops(),
        [this](void *stream, void *) {
            return submit("wait", stream);
        },
        [this](void *stream, int) {
            trace.push_back(stream == caller ? "sync caller" : "sync private");
            return sync_error;
        }
    };

    void arm() {
        ASSERT_EQ(fence.arm(kRun), 0);
        ASSERT_EQ(order.arm(kRun, caller, cpu, core), 0);
    }
    void publish_whole_boundary() {
        fence.note_kernel_submitted(kRun, Role::Aicore);
        ASSERT_EQ(fence.record(kRun, Role::Aicore, core), 0);
        fence.note_kernel_submitted(kRun, Role::Aicpu);
        ASSERT_EQ(fence.record(kRun, Role::Aicpu, cpu), 0);
    }
};

TEST_F(CallerOrderTest, CaptureArmingRequiresReservedEventsAndDoesNotCreateThem) {
    EXPECT_EQ(fence.arm(kRun, false), PTO_RUNTIME_ERR_UNSUPPORTED);
    EXPECT_EQ(order.arm(kRun, caller, cpu, core, false), PTO_RUNTIME_ERR_UNSUPPORTED);
    EXPECT_EQ(created, 0);
    ASSERT_EQ(fence.reserve_events(), 0);
    ASSERT_EQ(order.reserve_events(), 0);
    EXPECT_EQ(created, 5);
    ASSERT_TRUE(fence.events_ready());
    ASSERT_TRUE(order.events_ready());
    EXPECT_EQ(fence.arm(kRun, false), 0);
    EXPECT_EQ(order.arm(kRun, caller, cpu, core, false), 0);
    EXPECT_EQ(created, 5);
    EXPECT_EQ(order.retire(kRun), 0);
    EXPECT_EQ(order.release(), 0);
    EXPECT_EQ(fence.release(), 0);
    EXPECT_FALSE(order.events_ready());
    EXPECT_FALSE(fence.events_ready());
    EXPECT_EQ(fence.arm(kNext, false), PTO_RUNTIME_ERR_UNSUPPORTED);
    EXPECT_EQ(order.arm(kNext, caller, cpu, core, false), PTO_RUNTIME_ERR_UNSUPPORTED);
    EXPECT_EQ(created, 5);
}

TEST_F(CallerOrderTest, DestroyedNeverReplayedGraphReleasesAfterStreamDrainWithoutEventWait) {
    arm();
    ASSERT_EQ(order.mark_captured(kRun), 0);
    LaunchProgressSink sink;
    ASSERT_EQ(
        order.fork(
            kRun, sink,
            [] {
                return 0;
            }
        ),
        0
    );
    publish_whole_boundary();
    ASSERT_EQ(order.join(kRun, fence), 0);
    EXPECT_FALSE(completed);
    EXPECT_NE(order.release(), 0);
    trace.clear();
    ASSERT_EQ(order.release_after_graph_drain(), 0);
    EXPECT_TRUE(trace.empty());
    EXPECT_EQ(fence.release(), 0);
    EXPECT_EQ(destroyed, created);
    EXPECT_EQ(order.release_after_graph_drain(), 0);
}

TEST_F(CallerOrderTest, CallerWaitPinsTheKernelFenceUntilItsOwnProofCompletes) {
    arm();
    LaunchProgressSink sink;
    ASSERT_EQ(
        order.fork(
            kRun, sink,
            [] {
                return 0;
            }
        ),
        0
    );
    EXPECT_TRUE(sink.submitted());
    EXPECT_EQ(trace, (std::vector<std::string>{"record caller", "wait cpu", "record cpu", "wait core"}));
    publish_whole_boundary();
    ASSERT_EQ(order.join(kRun, fence), 0);
    EXPECT_EQ(trace[6], "wait caller");
    EXPECT_EQ(trace[7], "record caller");
    EXPECT_NE(fence.retire(kRun), 0);
    EXPECT_NE(fence.arm(kNext), 0);
    EXPECT_NE(order.release(), 0);
    EXPECT_EQ(order.poll(kRun), RunCallerOrder::Completion::Pending);
    EXPECT_NE(order.discharge(kRun, 100), 0);
    EXPECT_NE(fence.retire(kRun), 0);
    completed = true;
    EXPECT_EQ(order.discharge(kRun, 100), 0);
    EXPECT_EQ(order.retire(kRun), 0);
    EXPECT_EQ(fence.retire(kRun), 0);
    const int allocations = created;
    EXPECT_EQ(order.arm(kNext, &tokens[13], cpu, core), 0);
    EXPECT_EQ(created, allocations);
    EXPECT_EQ(order.retire(kNext), 0);
    EXPECT_EQ(order.release(), 0);
    EXPECT_EQ(destroyed, 3);
}

TEST_F(CallerOrderTest, EveryForkFailureIsSubmittedAndNeedsAllThreeStreams) {
    for (int fault = 1; fault <= 4; ++fault) {
        SCOPED_TRACE(fault);
        arm();
        operation = 0;
        fail_at = fault;
        LaunchProgressSink sink;
        EXPECT_EQ(
            order.fork(
                kRun, sink,
                [] {
                    return 0;
                }
            ),
            -77
        );
        EXPECT_TRUE(sink.submitted());
        EXPECT_NE(order.retire(kRun), 0);
        sync_error = -88;
        EXPECT_EQ(order.discharge(kRun, 100), -88);
        EXPECT_NE(order.release(), 0);
        sync_error = 0;
        trace.clear();
        EXPECT_EQ(order.discharge(kRun, 100), 0);
        EXPECT_EQ(trace, (std::vector<std::string>{"sync caller", "sync private", "sync private"}));
        EXPECT_EQ(order.retire(kRun), 0);
        EXPECT_EQ(fence.retire(kRun), 0);
    }
}

TEST_F(CallerOrderTest, FailedExitWaitOrTailRecordStillPinsTheBoundary) {
    for (int fault = 1; fault <= 2; ++fault) {
        SCOPED_TRACE(fault);
        fail_at = 0;
        arm();
        LaunchProgressSink sink;
        ASSERT_EQ(
            order.fork(
                kRun, sink,
                [] {
                    return 0;
                }
            ),
            0
        );
        publish_whole_boundary();
        operation = 0;
        fail_at = fault;
        EXPECT_EQ(order.join(kRun, fence), -77);
        EXPECT_NE(fence.retire(kRun), 0);
        sync_error = -88;
        EXPECT_EQ(order.discharge(kRun, 100), -88);
        EXPECT_NE(fence.retire(kRun), 0);
        sync_error = 0;
        EXPECT_EQ(order.discharge(kRun, 100), 0);
        EXPECT_EQ(order.retire(kRun), 0);
        EXPECT_EQ(fence.retire(kRun), 0);
    }
}

TEST_F(CallerOrderTest, ArmingFailureAndStaleIdentityNeverQueueWork) {
    EXPECT_NE(order.arm(kRun, cpu, cpu, core), 0);
    arm();
    LaunchProgressSink sink;
    EXPECT_NE(
        order.fork(
            kNext, sink,
            [] {
                return 0;
            }
        ),
        0
    );
    EXPECT_FALSE(sink.submitted());
    EXPECT_TRUE(trace.empty());
    EXPECT_NE(order.arm(kNext, caller, cpu, core), 0);
    EXPECT_EQ(order.retire(kRun), 0);
    EXPECT_EQ(order.retire(kRun), 0);
}

TEST_F(CallerOrderTest, AbandonAfterFenceQuarantineMakesNoDeviceCalls) {
    arm();
    LaunchProgressSink sink;
    ASSERT_EQ(
        order.fork(
            kRun, sink,
            [] {
                return 0;
            }
        ),
        0
    );
    publish_whole_boundary();
    ASSERT_EQ(order.join(kRun, fence), 0);
    fence.abandon();
    order.abandon();
    EXPECT_EQ(order.release(), 0);
    EXPECT_EQ(destroyed, 0);
}
}  // namespace

TEST_F(CallerOrderTest, PublicationPrecedesTheCoreForkAfterTheCallerDependency) {
    arm();
    LaunchProgressSink sink;
    ASSERT_EQ(
        order.fork(
            kRun, sink,
            [&] {
                trace.push_back("publish cpu");
                sink.mark_submitted();
                return 0;
            }
        ),
        0
    );
    EXPECT_EQ(trace, (std::vector<std::string>{"record caller", "wait cpu", "publish cpu", "record cpu", "wait core"}));
    publish_whole_boundary();
    ASSERT_EQ(order.join(kRun, fence), 0);
    completed = true;
    ASSERT_EQ(order.discharge(kRun, 100), 0);
    EXPECT_EQ(order.retire(kRun), 0);
}

TEST_F(CallerOrderTest, PublicationWithoutACallerOrdersBothPrivateStreams) {
    ASSERT_EQ(order.arm(kRun, nullptr, cpu, core), 0);
    EXPECT_EQ(created, 1);
    LaunchProgressSink sink;
    ASSERT_EQ(
        order.fork(
            kRun, sink,
            [&] {
                trace.push_back("publish cpu");
                sink.mark_submitted();
                return 0;
            }
        ),
        0
    );
    EXPECT_EQ(trace, (std::vector<std::string>{"publish cpu", "record cpu", "wait core"}));
    EXPECT_NE(order.retire(kRun), 0);
    ASSERT_EQ(order.discharge(kRun, 100, true), 0);
    EXPECT_EQ(trace.size(), 3U) << "run boundaries consume the entry wait without a whole-stream sync";
    EXPECT_EQ(order.retire(kRun), 0);
    EXPECT_EQ(order.release(), 0);
    EXPECT_EQ(destroyed, 1);
}

TEST_F(CallerOrderTest, AnUnchangedDescriptorWithoutACallerQueuesNoFork) {
    ASSERT_EQ(order.arm(kRun, nullptr, cpu, core), 0);
    LaunchProgressSink sink;
    ASSERT_EQ(
        order.fork(
            kRun, sink,
            [] {
                return 0;
            }
        ),
        0
    );
    EXPECT_FALSE(sink.submitted());
    EXPECT_TRUE(trace.empty());
    EXPECT_EQ(order.discharge(kRun, 100), 0);
    EXPECT_TRUE(trace.empty());
    EXPECT_EQ(order.retire(kRun), 0);
}

TEST_F(CallerOrderTest, PublicationFailureRetainsItsPrivateStreamReaders) {
    ASSERT_EQ(order.arm(kRun, nullptr, cpu, core), 0);
    LaunchProgressSink sink;
    EXPECT_EQ(
        order.fork(
            kRun, sink,
            [&] {
                sink.mark_submitted();
                return -82;
            }
        ),
        -82
    );
    EXPECT_TRUE(sink.submitted());
    EXPECT_TRUE(trace.empty());
    EXPECT_NE(order.retire(kRun), 0);
    sync_error = -88;
    EXPECT_EQ(order.discharge(kRun, 100), -88);
    EXPECT_NE(order.retire(kRun), 0);
    sync_error = 0;
    EXPECT_EQ(order.discharge(kRun, 100), 0);
    EXPECT_EQ(order.retire(kRun), 0);
    EXPECT_EQ(trace, (std::vector<std::string>{"sync private", "sync private", "sync private", "sync private"}));
}

TEST_F(CallerOrderTest, AThrowAfterCopySubmissionStillRequiresDischarge) {
    ASSERT_EQ(order.arm(kRun, nullptr, cpu, core), 0);
    LaunchProgressSink sink;
    EXPECT_THROW(
        order.fork(
            kRun, sink,
            [&]() -> int {
                sink.mark_submitted();
                throw std::runtime_error("copy");
            }
        ),
        std::runtime_error
    );
    EXPECT_NE(order.retire(kRun), 0);
    ASSERT_EQ(order.discharge(kRun, 100), 0);
    EXPECT_EQ(trace, (std::vector<std::string>{"sync private", "sync private"}));
    EXPECT_EQ(order.retire(kRun), 0);
}

TEST_F(CallerOrderTest, FailedPublicationForkRecordDoesNotDiscardTheCopy) {
    ASSERT_EQ(order.arm(kRun, nullptr, cpu, core), 0);
    fail_at = 1;
    LaunchProgressSink sink;
    EXPECT_EQ(
        order.fork(
            kRun, sink,
            [&] {
                sink.mark_submitted();
                return 0;
            }
        ),
        -77
    );
    EXPECT_TRUE(sink.submitted());
    EXPECT_EQ(trace, (std::vector<std::string>{"record cpu"}));
    EXPECT_NE(order.retire(kRun), 0);
    ASSERT_EQ(order.discharge(kRun, 100), 0);
    EXPECT_EQ(order.retire(kRun), 0);
}
