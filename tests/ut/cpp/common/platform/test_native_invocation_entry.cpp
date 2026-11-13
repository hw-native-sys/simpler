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
#include "aicpu/device_run_result_aicpu.h"
#include "aicpu/native_invocation_entry.h"
#include "aicpu/platform_aicpu_affinity.h"
#include "common/native_invocation_evidence.h"
namespace {
int32_t initialize(void *storage) { return aicpu_begin_native_invocation(6, reinterpret_cast<uint64_t>(storage), 7); }
void run(NativeInvocationResultStorage &storage, int32_t error, bool handoff) {
    std::atomic<int> finished{0}, destructed{0}, last_index{-1};
    std::array<int32_t, 6> returns{};
    std::vector<std::thread> threads;
    for (int i = 0; i < 6; ++i)
        threads.emplace_back([&, i] {
            const auto gate = platform_aicpu_entry_gate(4, 6, {initialize, &storage});
            int expected = -1;
            const bool last = !gate.execute && last_index.compare_exchange_strong(expected, i);
            [&] {
                struct Scope {
                    std::atomic<int> &counter;
                    ~Scope() { counter.fetch_add(1); }
                } scope{destructed};
                if (gate.execute && gate.callback_index == 0 && handoff) {
                    RunTerminalPublisher publisher;
                    const RunTerminalSelection selection =
                        error == 0 ? RunTerminalSelection{DeviceRunVerdict::Ok, 0, DeviceRunCodeSource::None} :
                                     RunTerminalSelection{DeviceRunVerdict::Error, error, DeviceRunCodeSource::Header};
                    const uint8_t payload[] = {42};
                    publisher.take(selection, error == 0 ? nullptr : payload, error == 0 ? 0 : 1);
                    EXPECT_TRUE(publisher.handoff());
                }
            }();
            if (last) {
                while (finished.load(std::memory_order_acquire) != 5)
                    std::this_thread::yield();
                EXPECT_EQ(destructed.load(), 6);
                EXPECT_EQ(storage.result.published, 0);
            }
            returns[i] = aicpu_finish_native_invocation(gate.callback_index, gate.status);
            finished.fetch_add(1, std::memory_order_release);
        });
    for (auto &thread : threads)
        thread.join();
    ASSERT_GE(last_index.load(), 0);
    for (int i = 0; i < 6; ++i)
        EXPECT_EQ(returns[i], i == last_index.load() ? (handoff ? error : -1) : 0);
}
}  // namespace
TEST(NativeInvocationEntry, ActualBeginHandoffFinishRepeatWithSameBinding) {
    NativeInvocationResultStorage storage;
    for (int round = 0; round < 3; ++round) {
        run(storage, 0, true);
        EXPECT_EQ(device_run_result_terminal(storage.result, 7).state, DeviceRunTerminalState::Succeeded);
    }
}
TEST(NativeInvocationEntry, AccountedUrmaFailurePublishesItsSemanticErrorWithSuccessfulTransport) {
    NativeInvocationResultStorage storage;
    ASSERT_EQ(initialize(&storage), 0);
    RunTerminalPublisher publisher;
    const uint8_t scene[] = {43};
    publisher.take({DeviceRunVerdict::Error, -101, DeviceRunCodeSource::AccountedUrmaSubmit}, scene, sizeof(scene));
    ASSERT_TRUE(publisher.handoff());
    for (int i = 0; i < 5; ++i)
        EXPECT_EQ(aicpu_finish_native_invocation(i, i == 2 ? -101 : 0), 0);
    EXPECT_EQ(storage.result.published, 0u);
    EXPECT_EQ(aicpu_finish_native_invocation(5, -101), 0);

    const auto terminal = device_run_result_terminal(storage.result, 7);
    EXPECT_EQ(terminal.state, DeviceRunTerminalState::Failed);
    EXPECT_EQ(terminal.code, -101);
    EXPECT_EQ(terminal.source, DeviceRunCodeSource::AccountedUrmaSubmit);
    EXPECT_EQ(storage.result.payload[0], 43);
    const auto saved = storage.evidence.read_after_exit();
    EXPECT_EQ(saved.kind, NativeInvocationEvidence::Kind::Error);
    EXPECT_EQ(saved.terminal.code, -101);
}

TEST(NativeInvocationEntry, ErrorSurvivesFollowingSuccessWithoutSkippingIt) {
    NativeInvocationResultStorage storage;
    run(storage, -77, true);
    EXPECT_EQ(device_run_result_terminal(storage.result, 7).code, -77);
    run(storage, 0, true);
    EXPECT_EQ(device_run_result_terminal(storage.result, 7).state, DeviceRunTerminalState::Succeeded);
    const auto evidence = storage.evidence.read_after_exit();
    EXPECT_EQ(evidence.terminal.code, -77);
    EXPECT_EQ(evidence.payload[0], 42);
}
TEST(NativeInvocationEntry, MissingTerminalCannotReusePreviouslyPublishedSuccess) {
    NativeInvocationResultStorage storage;
    run(storage, 0, true);
    run(storage, 0, false);
    EXPECT_EQ(device_run_result_terminal(storage.result, 7).state, DeviceRunTerminalState::Undecided);
    EXPECT_EQ(storage.evidence.read_after_exit().kind, NativeInvocationEvidence::Kind::MissingTerminal);
}
TEST(NativeInvocationEntry, MissingRegionCompatibilityAndHalfBindingRefusal) {
    ASSERT_EQ(aicpu_begin_native_invocation(1, 0, 0), 0);
    RunTerminalPublisher publisher;
    publisher.take({DeviceRunVerdict::Ok, 0, DeviceRunCodeSource::None}, nullptr, 0);
    ASSERT_TRUE(publisher.handoff());
    EXPECT_EQ(aicpu_finish_native_invocation(0, 0), 0);
    NativeInvocationResultStorage storage;
    ASSERT_NE(aicpu_begin_native_invocation(1, reinterpret_cast<uint64_t>(&storage), 0), 0);
    EXPECT_NE(aicpu_finish_native_invocation(0, -1), 0);
}

TEST(NativeInvocationEntry, UnrequestedRecordKeepsTheInvocationIdentityAndSignedErrors) {
    for (int error : {0, -73}) {
        ASSERT_EQ(aicpu_begin_native_invocation(2, 0, 719), 0);
        EXPECT_EQ(get_platform_run_result_epoch(), 719u);
        RunTerminalPublisher publisher;
        publisher.take({DeviceRunVerdict::Ok, 0, DeviceRunCodeSource::None}, nullptr, 0);
        ASSERT_TRUE(publisher.handoff());
        EXPECT_EQ(aicpu_finish_native_invocation(0, error), 0);
        EXPECT_EQ(aicpu_finish_native_invocation(1, 0), error);
    }
    ASSERT_EQ(aicpu_begin_native_invocation(1, 0, 720), 0);
    EXPECT_NE(aicpu_finish_native_invocation(0, 0), 0);
    ASSERT_EQ(aicpu_begin_native_invocation(1, 0, 721), 0);
    ASSERT_TRUE(aicpu_accept_runtime_terminal({DeviceRunVerdict::Error, -74, DeviceRunCodeSource::Header}, nullptr, 0));
    EXPECT_EQ(aicpu_finish_native_invocation(0, 0), -74);
}

TEST(NativeInvocationEntry, InitializerFailureIsReturnedOnlyByLastCallback) {
    NativeInvocationResultStorage storage;
    std::array<int, 6> returns{};
    std::atomic<int> initialized{0}, executed{0};
    struct Context {
        NativeInvocationResultStorage *storage;
        std::atomic<int> *initialized;
    } context{&storage, &initialized};
    const auto reject = [](void *opaque) -> int32_t {
        auto &ctx = *static_cast<Context *>(opaque);
        EXPECT_EQ(aicpu_begin_native_invocation(6, reinterpret_cast<uint64_t>(ctx.storage), 7), 0);
        ctx.initialized->fetch_add(1);
        return -51;
    };
    std::vector<std::thread> threads;
    for (int i = 0; i < 6; ++i)
        threads.emplace_back([&, i] {
            const auto gate = platform_aicpu_entry_gate(4, 6, {reject, &context});
            if (gate.execute) executed.fetch_add(1);
            EXPECT_EQ(gate.status, -51);
            returns[i] = aicpu_finish_native_invocation(gate.callback_index, gate.status);
        });
    for (auto &thread : threads)
        thread.join();
    int failed = 0;
    for (int rc : returns) {
        EXPECT_TRUE(rc == 0 || rc == -51);
        failed += rc == -51;
    }
    EXPECT_EQ(failed, 1);
    EXPECT_EQ(initialized.load(), 1);
    EXPECT_EQ(executed.load(), 0);
    const auto terminal = device_run_result_terminal(storage.result, 7);
    EXPECT_EQ(terminal.state, DeviceRunTerminalState::Failed);
    EXPECT_EQ(terminal.code, -51);
    EXPECT_EQ(terminal.source, DeviceRunCodeSource::ThreadRc);
    EXPECT_EQ(storage.evidence.read_after_exit().terminal.code, -51);
    // Every callback has joined. Reuse the exact binding as serial replay does.
    for (int round = 0; round < 3; ++round) {
        run(storage, 0, true);
        EXPECT_EQ(device_run_result_terminal(storage.result, 7).state, DeviceRunTerminalState::Succeeded);
        const auto saved = storage.evidence.read_after_exit();
        EXPECT_EQ(saved.kind, NativeInvocationEvidence::Kind::Error);
        EXPECT_EQ(saved.terminal.code, -51);
        EXPECT_EQ(saved.fault_count, 1u);
    }
}

TEST(NativeInvocationEntry, ControlledEarlyReturnWithoutRuntimeHandoffStillReportsOriginalError) {
    NativeInvocationResultStorage storage;
    ASSERT_EQ(aicpu_begin_native_invocation(6, reinterpret_cast<uint64_t>(&storage), 7), 0);
    EXPECT_EQ(aicpu_finish_native_invocation(0, -67), 0);
    for (int i = 1; i < 5; ++i)
        EXPECT_EQ(aicpu_finish_native_invocation(i, 0), 0);
    EXPECT_EQ(storage.result.published, 0);
    EXPECT_EQ(aicpu_finish_native_invocation(5, 0), -67);
    EXPECT_EQ(device_run_result_terminal(storage.result, 7).code, -67);
    EXPECT_EQ(storage.evidence.read_after_exit().terminal.code, -67);
}

TEST(NativeInvocationEntry, MissingContributionNeverPublishesEvenWithRuntimeSuccess) {
    NativeInvocationResultStorage storage;
    run(storage, 0, true);
    ASSERT_EQ(aicpu_begin_native_invocation(3, reinterpret_cast<uint64_t>(&storage), 7), 0);
    RunTerminalPublisher publisher;
    publisher.take({DeviceRunVerdict::Ok, 0, DeviceRunCodeSource::None}, nullptr, 0);
    ASSERT_TRUE(publisher.handoff());
    EXPECT_EQ(aicpu_finish_native_invocation(0, 0), 0);
    EXPECT_EQ(aicpu_finish_native_invocation(1, 0), 0);
    EXPECT_EQ(storage.result.published, 0);
    EXPECT_EQ(device_run_result_terminal(storage.result, 7).state, DeviceRunTerminalState::Undecided);
    // All calls made by this CPU fixture have returned; no device task exists.
    // This does not infer physical exit of a missing callback on real hardware.
    const auto evidence = storage.evidence.read_after_exit();
    EXPECT_EQ(evidence.kind, NativeInvocationEvidence::Kind::MissingTerminal);
    EXPECT_EQ(evidence.fault_count, 1);
    // There is no outstanding worker in this fixture. This is not permission
    // to reset a real invocation whose missing callback could still run.
    for (int round = 0; round < 3; ++round) {
        run(storage, 0, true);
        EXPECT_EQ(device_run_result_terminal(storage.result, 7).state, DeviceRunTerminalState::Succeeded);
        const auto saved = storage.evidence.read_after_exit();
        EXPECT_EQ(saved.kind, NativeInvocationEvidence::Kind::MissingTerminal);
        EXPECT_EQ(saved.fault_count, 1u);
    }
    ASSERT_TRUE(storage.evidence.acknowledge_after_exit());
    EXPECT_EQ(storage.evidence.read_after_exit().kind, NativeInvocationEvidence::Kind::None);
    run(storage, 0, true);
    EXPECT_EQ(device_run_result_terminal(storage.result, 7).state, DeviceRunTerminalState::Succeeded);
    EXPECT_EQ(storage.evidence.read_after_exit().fault_count, 0u);
}

TEST(NativeInvocationEntry, MalformedRuntimeTerminalCannotPublishSuccess) {
    NativeInvocationResultStorage storage;
    ASSERT_EQ(aicpu_begin_native_invocation(2, reinterpret_cast<uint64_t>(&storage), 7), 0);
    EXPECT_FALSE(aicpu_accept_runtime_terminal({DeviceRunVerdict::Ok, -3, DeviceRunCodeSource::ThreadRc}, nullptr, 0));
    EXPECT_EQ(aicpu_finish_native_invocation(0, 0), 0);
    EXPECT_EQ(aicpu_finish_native_invocation(1, 0), -1);
    EXPECT_EQ(storage.result.published, 0);
    EXPECT_EQ(storage.evidence.read_after_exit().kind, NativeInvocationEvidence::Kind::InvalidTerminal);
}

TEST(NativeInvocationEntry, MissingOptionalPayloadKeepsTheRealFunctionalFailure) {
    NativeInvocationResultStorage storage;
    ASSERT_EQ(aicpu_begin_native_invocation(1, reinterpret_cast<uint64_t>(&storage), 7), 0);
    RunTerminalPublisher publisher;
    publisher.take(
        {DeviceRunVerdict::Error, -79, DeviceRunCodeSource::Header}, nullptr, DEVICE_RUN_RESULT_PAYLOAD_BYTES + 1
    );
    ASSERT_TRUE(publisher.handoff());
    EXPECT_EQ(aicpu_finish_native_invocation(0, 0), -79);
    EXPECT_EQ(device_run_result_terminal(storage.result, 7).code, -79);
    EXPECT_EQ(storage.result.payload_bytes, 0u);
}

TEST(NativeInvocationEntry, InvalidPublicationBindingFailsSuccessButPreservesOriginalError) {
    for (const int code : {0, -79}) {
        NativeInvocationResultStorage storage;
        ASSERT_EQ(aicpu_begin_native_invocation(1, reinterpret_cast<uint64_t>(&storage), 7), 0);
        RunTerminalPublisher publisher;
        publisher.take(
            code == 0 ? RunTerminalSelection{DeviceRunVerdict::Ok, 0, DeviceRunCodeSource::None} :
                        RunTerminalSelection{DeviceRunVerdict::Error, code, DeviceRunCodeSource::Header},
            nullptr, 0
        );
        ASSERT_TRUE(publisher.handoff());
        // Deliberately violate the binding contract, not a simulated DMA fault.
        set_platform_run_result(reinterpret_cast<uint64_t>(&storage), 0);
        EXPECT_EQ(aicpu_finish_native_invocation(0, 0), code == 0 ? -1 : code);
        EXPECT_EQ(storage.result.published, 0u);
        EXPECT_EQ(
            storage.evidence.read_after_exit().kind,
            code == 0 ? NativeInvocationEvidence::Kind::PublicationFailed : NativeInvocationEvidence::Kind::Error
        );
    }
    set_platform_run_result(0, 0);
}

TEST(NativeInvocationEntry, ObservedFailureDoesNotRepeatOnIndependentSuccess) {
    NativeInvocationResultStorage storage;
    run(storage, -77, true);
    const auto saved = storage.evidence.read_after_exit();
    ASSERT_EQ(saved.terminal.code, -77);
    ASSERT_TRUE(storage.evidence.acknowledge_after_exit());
    run(storage, 0, true);
    EXPECT_EQ(storage.evidence.read_after_exit().kind, NativeInvocationEvidence::Kind::None);
    run(storage, -79, true);
    EXPECT_EQ(storage.evidence.read_after_exit().terminal.code, -79);
    EXPECT_EQ(saved.terminal.code, -77);
}

TEST(NativeInvocationEntry, SharedAdapterFinishesAfterWorkAndItsCleanup) {
    NativeInvocationResultStorage storage;
    NativeInvocationEntry entry{1,     reinterpret_cast<uint64_t>(&storage), 7, aicpu_begin_native_invocation,
                                [] {}, aicpu_finish_native_invocation};
    ASSERT_EQ(NativeInvocationEntry::initialize(&entry), 0);
    bool cleaned = false;
    EXPECT_EQ(
        entry.complete(
            {0, 0, true},
            [&] {
                struct Cleanup {
                    bool &value;
                    ~Cleanup() { value = true; }
                } cleanup{cleaned};
                RunTerminalPublisher publisher;
                publisher.take({DeviceRunVerdict::Ok, 0, DeviceRunCodeSource::None}, nullptr, 0);
                EXPECT_TRUE(publisher.handoff());
                EXPECT_EQ(storage.result.published, 0);
                return 0;
            }
        ),
        0
    );
    EXPECT_TRUE(cleaned);
    EXPECT_EQ(device_run_result_terminal(storage.result, 7).state, DeviceRunTerminalState::Succeeded);
}
