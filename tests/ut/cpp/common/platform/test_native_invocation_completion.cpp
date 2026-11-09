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

#include <array>
#include <atomic>
#include <thread>
#include <vector>

#include "aicpu/native_invocation_completion.h"

namespace {
constexpr RunTerminalSelection success{DeviceRunVerdict::Ok, 0, DeviceRunCodeSource::None};
struct Publication {
    int count{0};
    RunTerminalSelection result{};
    std::vector<uint8_t> payload;
    NativeTerminalPublication operator()(RunTerminalSelection value, const void *data, size_t bytes) {
        ++count;
        result = value;
        if (bytes != 0) {
            const auto *begin = static_cast<const uint8_t *>(data);
            payload.assign(begin, begin + bytes);
        }
        return NativeTerminalPublication::Published;
    }
};
}  // namespace

TEST(NativeInvocationCompletion, DroppedCallbackCanFinishLastAndReturnEarlierError) {
    NativeInvocationCompletion completion;
    ASSERT_TRUE(completion.begin(6));
    ASSERT_TRUE(completion.accept_runtime_terminal(success, nullptr, 0));
    Publication publish;
    std::array<int, 5> returns{};
    std::vector<std::thread> workers;
    for (int i = 0; i < 5; ++i) {
        workers.emplace_back([&, i] {
            returns[i] = completion.finish(i, i == 1 ? -71 : 0, publish);
        });
    }
    for (auto &worker : workers)
        worker.join();
    for (int rc : returns)
        EXPECT_EQ(rc, 0);
    EXPECT_EQ(publish.count, 0);
    EXPECT_EQ(completion.finish(5, 0, publish), -71);
    EXPECT_EQ(publish.count, 1);
    EXPECT_EQ(publish.result.source, DeviceRunCodeSource::ThreadRc);
}

TEST(NativeInvocationCompletion, LastContributorObservesAllPriorWrites) {
    NativeInvocationCompletion completion;
    ASSERT_TRUE(completion.begin(6));
    std::array<int, 6> writes{};
    std::array<int, 6> returns{};
    std::atomic<bool> start{false};
    std::atomic<int> published{0};
    std::vector<std::thread> workers;
    for (int i = 0; i < 6; ++i) {
        workers.emplace_back([&, i] {
            while (!start.load(std::memory_order_acquire))
                std::this_thread::yield();
            writes[i] = i + 1;
            if (i == 0) EXPECT_TRUE(completion.accept_runtime_terminal(success, nullptr, 0));
            returns[i] = completion.finish(i, 0, [&](RunTerminalSelection, const void *, size_t) {
                for (int j = 0; j < 6; ++j)
                    EXPECT_EQ(writes[j], j + 1);
                published.fetch_add(1);
                return NativeTerminalPublication::Published;
            });
        });
    }
    start.store(true, std::memory_order_release);
    for (auto &worker : workers)
        worker.join();
    EXPECT_EQ(published.load(), 1);
    for (int rc : returns)
        EXPECT_EQ(rc, 0);
}

TEST(NativeInvocationCompletion, RuntimeFailureOutranksEntryErrorAndOwnsPayload) {
    NativeInvocationCompletion completion;
    ASSERT_TRUE(completion.begin(2));
    uint8_t bytes[] = {1, 2, 3};
    ASSERT_TRUE(completion.accept_runtime_terminal(
        {DeviceRunVerdict::Error, -93, DeviceRunCodeSource::Header}, bytes, sizeof(bytes)
    ));
    bytes[0] = 99;
    Publication publish;
    EXPECT_EQ(completion.finish(0, -72, publish), 0);
    EXPECT_EQ(completion.finish(1, 0, publish), -93);
    EXPECT_EQ(publish.result.source, DeviceRunCodeSource::Header);
    EXPECT_EQ(publish.payload, (std::vector<uint8_t>{1, 2, 3}));
}

TEST(NativeInvocationCompletion, MissingTerminalIsUnknownButEarlyErrorCanBePublished) {
    NativeInvocationCompletion completion;
    Publication publish;
    ASSERT_TRUE(completion.begin(1));
    EXPECT_EQ(completion.finish(0, 0, publish), NativeInvocationCompletion::kInternalError);
    EXPECT_EQ(publish.count, 0);
    ASSERT_TRUE(completion.begin(1));
    EXPECT_EQ(completion.finish(0, -17, publish), -17);
    EXPECT_EQ(publish.count, 1);
}

TEST(NativeInvocationCompletion, DuplicateAndInvalidIndicesDoNotCompleteMissingParticipant) {
    NativeInvocationCompletion completion;
    ASSERT_TRUE(completion.begin(2));
    ASSERT_TRUE(completion.accept_runtime_terminal(success, nullptr, 0));
    Publication publish;
    EXPECT_EQ(completion.finish(0, 0, publish), 0);
    EXPECT_EQ(completion.finish(0, 0, publish), NativeInvocationCompletion::kInternalError);
    EXPECT_EQ(completion.finish(-1, 0, publish), NativeInvocationCompletion::kInternalError);
    EXPECT_EQ(completion.finish(2, 0, publish), NativeInvocationCompletion::kInternalError);
    EXPECT_EQ(publish.count, 0);
    EXPECT_EQ(completion.finish(1, 0, publish), 0);
    EXPECT_EQ(publish.count, 1);
}

TEST(NativeInvocationCompletion, SerialReuseDoesNotInheritSuccessOrPayload) {
    NativeInvocationCompletion completion;
    for (int round = 0; round < 3; ++round) {
        Publication publish;
        ASSERT_TRUE(completion.begin(1));
        ASSERT_TRUE(completion.accept_runtime_terminal(success, nullptr, 0));
        EXPECT_EQ(completion.finish(0, 0, publish), 0);
        EXPECT_EQ(publish.count, 1);
        ASSERT_TRUE(completion.begin(1));
        EXPECT_EQ(completion.finish(0, 0, publish), NativeInvocationCompletion::kInternalError);
        EXPECT_EQ(publish.count, 1);
    }
}

TEST(NativeInvocationCompletion, InvalidHandoffAndPublicationFailureCannotSucceed) {
    NativeInvocationCompletion completion;
    Publication publish;
    ASSERT_TRUE(completion.begin(1));
    EXPECT_FALSE(completion.accept_runtime_terminal(success, nullptr, 1));
    EXPECT_EQ(completion.finish(0, 0, publish), NativeInvocationCompletion::kInternalError);
    EXPECT_EQ(publish.count, 0);
    ASSERT_TRUE(completion.begin(1));
    ASSERT_TRUE(completion.accept_runtime_terminal(success, nullptr, 0));
    EXPECT_FALSE(completion.accept_runtime_terminal(success, nullptr, 0));
    EXPECT_EQ(completion.finish(0, 0, publish), NativeInvocationCompletion::kInternalError);
    ASSERT_TRUE(completion.begin(1));
    ASSERT_TRUE(completion.accept_runtime_terminal(success, nullptr, 0));
    EXPECT_EQ(
        completion.finish(
            0, 0,
            [](RunTerminalSelection, const void *, size_t) {
                return NativeTerminalPublication::InvalidState;
            }
        ),
        NativeInvocationCompletion::kInternalError
    );
}

TEST(NativeInvocationCompletion, RejectsUntrustedCountsAndUninitializedFinish) {
    NativeInvocationCompletion completion;
    Publication publish;
    EXPECT_FALSE(completion.begin(0));
    EXPECT_FALSE(completion.begin(MAX_GATE_THREADS + 1));
    EXPECT_EQ(completion.finish(0, 0, publish), NativeInvocationCompletion::kInternalError);
    EXPECT_EQ(publish.count, 0);
}

TEST(NativeInvocationCompletion, ColdSuccessDoesNotCreateFaultEvidence) {
    NativeInvocationCompletion completion;
    NativeInvocationEvidence evidence;
    Publication publish;
    ASSERT_TRUE(completion.begin(1, &evidence));
    EXPECT_FALSE(evidence.acknowledge_after_exit());
    ASSERT_TRUE(completion.accept_runtime_terminal(success, nullptr, 0));
    EXPECT_EQ(completion.finish(0, 0, publish), 0);
    EXPECT_EQ(evidence.read_after_exit().kind, NativeInvocationEvidence::Kind::None);
    EXPECT_EQ(evidence.read_after_exit().fault_count, 0);
}

TEST(NativeInvocationCompletion, IndependentSuccessDoesNotEraseUnacknowledgedFailure) {
    NativeInvocationCompletion completion;
    NativeInvocationEvidence evidence;
    Publication publish;
    uint8_t bytes[] = {4, 5};
    ASSERT_TRUE(completion.begin(1, &evidence));
    ASSERT_TRUE(completion.accept_runtime_terminal(
        {DeviceRunVerdict::Error, -51, DeviceRunCodeSource::Header}, bytes, sizeof(bytes)
    ));
    EXPECT_EQ(completion.finish(0, 0, publish), -51);
    bytes[0] = 99;
    ASSERT_TRUE(completion.begin(1, &evidence));
    ASSERT_TRUE(completion.accept_runtime_terminal(success, nullptr, 0));
    EXPECT_EQ(completion.finish(0, 0, publish), 0);
    auto saved = evidence.read_after_exit();
    EXPECT_EQ(saved.kind, NativeInvocationEvidence::Kind::Error);
    EXPECT_EQ(saved.terminal.code, -51);
    EXPECT_EQ(saved.payload[0], 4);
    EXPECT_EQ(saved.fault_count, 1);
    ASSERT_TRUE(completion.begin(1, &evidence));
    EXPECT_EQ(completion.finish(0, -72, publish), -72);
    saved = evidence.read_after_exit();
    EXPECT_EQ(saved.terminal.code, -51);
    EXPECT_EQ(saved.fault_count, 2);
    EXPECT_TRUE(evidence.acknowledge_after_exit());
    EXPECT_EQ(evidence.read_after_exit().kind, NativeInvocationEvidence::Kind::None);
}

TEST(NativeInvocationCompletion, MissingCallbackBecomesUnknownBeforeSerialReset) {
    NativeInvocationCompletion completion;
    NativeInvocationEvidence evidence;
    Publication publish;
    ASSERT_TRUE(completion.begin(2, &evidence));
    EXPECT_EQ(completion.finish(0, 0, publish), 0);
    // The fixture models physical exit of an incomplete invocation. begin is
    // not a means to force a still-running invocation out of the device.
    ASSERT_TRUE(completion.begin(1, &evidence));
    ASSERT_TRUE(completion.accept_runtime_terminal(success, nullptr, 0));
    EXPECT_EQ(completion.finish(0, 0, publish), 0);
    const auto saved = evidence.read_after_exit();
    EXPECT_EQ(saved.kind, NativeInvocationEvidence::Kind::MissingTerminal);
    EXPECT_EQ(saved.fault_count, 1);
    EXPECT_EQ(publish.count, 1);
}

TEST(NativeInvocationCompletion, PostExitReadDetectsMissingTerminalWithoutAnotherReplay) {
    NativeInvocationCompletion completion;
    NativeInvocationEvidence evidence;
    ASSERT_TRUE(completion.begin(1, &evidence));
    EXPECT_EQ(evidence.read_after_exit().kind, NativeInvocationEvidence::Kind::MissingTerminal);
    EXPECT_EQ(evidence.read_after_exit().fault_count, 1);
    EXPECT_EQ(evidence.read_after_exit().fault_count, 1);
    EXPECT_TRUE(evidence.acknowledge_after_exit());
}

TEST(NativeInvocationCompletion, FailedPublicationRetainsOriginalErrorAndPayload) {
    NativeInvocationCompletion completion;
    NativeInvocationEvidence evidence;
    auto refuse = [](RunTerminalSelection, const void *, size_t) {
        return NativeTerminalPublication::InvalidState;
    };
    ASSERT_TRUE(completion.begin(1, &evidence));
    ASSERT_TRUE(completion.accept_runtime_terminal(success, nullptr, 0));
    EXPECT_EQ(completion.finish(0, 0, refuse), NativeInvocationCompletion::kInternalError);
    EXPECT_EQ(evidence.read_after_exit().kind, NativeInvocationEvidence::Kind::PublicationFailed);
    ASSERT_TRUE(evidence.acknowledge_after_exit());
    ASSERT_TRUE(completion.begin(1, &evidence));
    uint8_t bytes[] = {17};
    ASSERT_TRUE(completion.accept_runtime_terminal(
        {DeviceRunVerdict::Error, -93, DeviceRunCodeSource::ShutdownRc}, bytes, sizeof(bytes)
    ));
    EXPECT_EQ(completion.finish(0, 0, refuse), -93);
    const auto saved = evidence.read_after_exit();
    EXPECT_EQ(saved.kind, NativeInvocationEvidence::Kind::Error);
    EXPECT_EQ(saved.terminal.code, -93);
    EXPECT_EQ(saved.terminal.source, DeviceRunCodeSource::ShutdownRc);
    EXPECT_FALSE(saved.publication_succeeded);
    EXPECT_EQ(saved.payload[0], 17);
}

TEST(NativeInvocationCompletion, EvidenceBelongsToPreparedOwnerNotExecutorSingleton) {
    NativeInvocationCompletion completion;
    NativeInvocationEvidence first, second;
    Publication publish;
    ASSERT_TRUE(completion.begin(1, &first));
    EXPECT_EQ(completion.finish(0, -37, publish), -37);
    ASSERT_TRUE(completion.begin(1, &second));
    ASSERT_TRUE(completion.accept_runtime_terminal(success, nullptr, 0));
    EXPECT_EQ(completion.finish(0, 0, publish), 0);
    EXPECT_EQ(first.read_after_exit().terminal.code, -37);
    EXPECT_EQ(second.read_after_exit().kind, NativeInvocationEvidence::Kind::None);
}

TEST(NativeInvocationCompletion, UnrequestedPrivateRecordDoesNotGateSuccessOrLoseFunctionalError) {
    NativeInvocationCompletion completion;
    NativeInvocationEvidence evidence;
    const auto omitted = [](RunTerminalSelection, const void *, size_t) {
        return NativeTerminalPublication::NotRequested;
    };
    ASSERT_TRUE(completion.begin(1, &evidence));
    ASSERT_TRUE(completion.accept_runtime_terminal(success, nullptr, 0));
    EXPECT_EQ(completion.finish(0, 0, omitted), 0);
    EXPECT_EQ(evidence.read_after_exit().kind, NativeInvocationEvidence::Kind::None);
    ASSERT_TRUE(completion.begin(1, &evidence));
    ASSERT_TRUE(
        completion.accept_runtime_terminal({DeviceRunVerdict::Error, -73, DeviceRunCodeSource::Header}, nullptr, 0)
    );
    EXPECT_EQ(completion.finish(0, 0, omitted), -73);
    EXPECT_EQ(evidence.read_after_exit().terminal.code, -73);
    EXPECT_FALSE(evidence.read_after_exit().publication_succeeded);
}

TEST(NativeInvocationCompletion, RetiredCcuFailureCompletesStreamsAndPreservesBusinessEvidence) {
    NativeInvocationCompletion completion;
    NativeInvocationEvidence evidence;
    Publication publish;
    ASSERT_TRUE(completion.begin(3, &evidence));
    uint8_t bytes[] = {7, 19};
    ASSERT_TRUE(completion.accept_runtime_terminal(
        {DeviceRunVerdict::Error, -105, DeviceRunCodeSource::Header}, bytes, sizeof(bytes), true
    ));
    bytes[0] = 99;
    EXPECT_EQ(completion.finish(0, -105, publish), 0);
    EXPECT_EQ(completion.finish(1, 0, publish), 0);
    EXPECT_EQ(publish.count, 0);
    EXPECT_EQ(completion.finish(2, -105, publish), 0);
    EXPECT_EQ(publish.count, 1);
    EXPECT_EQ(publish.result.verdict, DeviceRunVerdict::Error);
    EXPECT_EQ(publish.result.code, -105);
    EXPECT_EQ(publish.payload, (std::vector<uint8_t>{7, 19}));
    const auto saved = evidence.read_after_exit();
    EXPECT_EQ(saved.kind, NativeInvocationEvidence::Kind::Error);
    EXPECT_EQ(saved.terminal.code, -105);
    EXPECT_TRUE(saved.publication_succeeded);
    EXPECT_EQ(saved.payload_bytes, 2);
    EXPECT_EQ(saved.payload[0], 7);
}

TEST(NativeInvocationCompletion, CcuFailureRequiresExplicitRetirementProof) {
    NativeInvocationCompletion completion;
    Publication publish;
    ASSERT_TRUE(completion.begin(2));
    ASSERT_TRUE(
        completion.accept_runtime_terminal({DeviceRunVerdict::Error, -105, DeviceRunCodeSource::Header}, nullptr, 0)
    );
    EXPECT_EQ(completion.finish(0, -105, publish), 0);
    EXPECT_EQ(completion.finish(1, 0, publish), -105);
}

TEST(NativeInvocationCompletion, OtherCallbackErrorsVetoRetiredCcuFailureInEitherOrder) {
    for (int other_error : {-1, -106, -507018}) {
        for (bool other_first : {false, true}) {
            NativeInvocationCompletion completion;
            Publication publish;
            ASSERT_TRUE(completion.begin(3));
            ASSERT_TRUE(completion.accept_runtime_terminal(
                {DeviceRunVerdict::Error, -105, DeviceRunCodeSource::Header}, nullptr, 0, true
            ));
            EXPECT_EQ(completion.finish(0, other_first ? other_error : -105, publish), 0);
            EXPECT_EQ(completion.finish(1, other_first ? -105 : other_error, publish), 0);
            EXPECT_NE(completion.finish(2, 0, publish), 0);
            EXPECT_EQ(publish.result.code, -105);
        }
    }
}

TEST(NativeInvocationCompletion, RetiredCcuFailureCannotNormalizeUnpublishedRecord) {
    for (auto publication : {NativeTerminalPublication::NotRequested, NativeTerminalPublication::InvalidState}) {
        NativeInvocationCompletion completion;
        NativeInvocationEvidence evidence;
        ASSERT_TRUE(completion.begin(1, &evidence));
        ASSERT_TRUE(completion.accept_runtime_terminal(
            {DeviceRunVerdict::Error, -105, DeviceRunCodeSource::Header}, nullptr, 0, true
        ));
        EXPECT_EQ(
            completion.finish(
                0, -105,
                [publication](RunTerminalSelection, const void *, size_t) {
                    return publication;
                }
            ),
            -105
        );
        const auto saved = evidence.read_after_exit();
        EXPECT_EQ(saved.terminal.code, -105);
        EXPECT_FALSE(saved.publication_succeeded);
    }
}

TEST(NativeInvocationCompletion, RetirementProofRejectsOtherTerminalKindsAndInvalidHandoff) {
    for (const auto terminal :
         {success, RunTerminalSelection{DeviceRunVerdict::Error, -106, DeviceRunCodeSource::Header},
          RunTerminalSelection{DeviceRunVerdict::Error, -105, DeviceRunCodeSource::None}}) {
        NativeInvocationCompletion completion;
        Publication publish;
        ASSERT_TRUE(completion.begin(1));
        EXPECT_FALSE(completion.accept_runtime_terminal(terminal, nullptr, 0, true));
        EXPECT_EQ(completion.finish(0, -105, publish), NativeInvocationCompletion::kInternalError);
        EXPECT_EQ(publish.count, 0);
    }
    NativeInvocationCompletion completion;
    Publication publish;
    const RunTerminalSelection failed{DeviceRunVerdict::Error, -105, DeviceRunCodeSource::Header};
    ASSERT_TRUE(completion.begin(1));
    EXPECT_FALSE(completion.accept_runtime_terminal(failed, nullptr, 1, true));
    EXPECT_EQ(completion.finish(0, -105, publish), NativeInvocationCompletion::kInternalError);
    ASSERT_TRUE(completion.begin(1));
    ASSERT_TRUE(completion.accept_runtime_terminal(failed, nullptr, 0, true));
    EXPECT_FALSE(completion.accept_runtime_terminal(failed, nullptr, 0, true));
    EXPECT_EQ(completion.finish(0, -105, publish), NativeInvocationCompletion::kInternalError);
    EXPECT_EQ(publish.count, 0);
}

TEST(NativeInvocationCompletion, BeginClearsRetirementProofAndOtherErrorVeto) {
    NativeInvocationCompletion completion;
    const RunTerminalSelection failed{DeviceRunVerdict::Error, -105, DeviceRunCodeSource::Header};
    Publication publish;
    ASSERT_TRUE(completion.begin(1));
    ASSERT_TRUE(completion.accept_runtime_terminal(failed, nullptr, 0, true));
    EXPECT_EQ(completion.finish(0, -106, publish), -105);
    ASSERT_TRUE(completion.begin(1));
    ASSERT_TRUE(completion.accept_runtime_terminal(failed, nullptr, 0, true));
    EXPECT_EQ(completion.finish(0, -105, publish), 0);
    ASSERT_TRUE(completion.begin(1));
    ASSERT_TRUE(completion.accept_runtime_terminal(failed, nullptr, 0));
    EXPECT_EQ(completion.finish(0, -105, publish), -105);
    ASSERT_TRUE(completion.begin(1));
    ASSERT_TRUE(completion.accept_runtime_terminal(success, nullptr, 0));
    EXPECT_EQ(completion.finish(0, 0, publish), 0);
}

TEST(NativeInvocationCompletion, ConcurrentCallbacksCannotHideOtherErrorBehindCcuFailure) {
    NativeInvocationCompletion completion;
    ASSERT_TRUE(completion.begin(6));
    std::atomic<bool> start{false};
    std::atomic<int> published{0};
    std::array<int, 6> returns{};
    std::vector<std::thread> workers;
    for (int i = 0; i < 6; ++i) {
        workers.emplace_back([&, i] {
            while (!start.load(std::memory_order_acquire))
                std::this_thread::yield();
            if (i == 0) {
                EXPECT_TRUE(completion.accept_runtime_terminal(
                    {DeviceRunVerdict::Error, -105, DeviceRunCodeSource::Header}, nullptr, 0, true
                ));
            }
            returns[i] =
                completion.finish(i, i == 4 ? -106 : -105, [&](RunTerminalSelection result, const void *, size_t) {
                    EXPECT_EQ(result.code, -105);
                    published.fetch_add(1);
                    return NativeTerminalPublication::Published;
                });
        });
    }
    start.store(true, std::memory_order_release);
    for (auto &worker : workers)
        worker.join();
    EXPECT_EQ(published.load(), 1);
    int failures = 0;
    for (int rc : returns)
        failures += rc != 0;
    EXPECT_EQ(failures, 1);
}
