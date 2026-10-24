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

#include <cstring>
#include <vector>

#include "tensormap_and_ringbuffer/kernel_capture_device.h"
#include "tensormap_and_ringbuffer/kernel_launch_admission.h"

namespace {
using namespace simpler::tmr;

class KernelCaptureDevice : public ::testing::Test {
protected:
    void SetUp() override {
        control.context_generation = 11;
        control.capture_generation = 17;
        control.runtime_address = 0x9000;
        control.binding_epoch = 29;
        control.callable_generation = 31;
        control.callable_id = 0;
        control.tensor_count = 1;
        control.scalar_count = 1;
        control.launched_threads = 2;
        control.worker_count = 3;
        registration.reference = {reinterpret_cast<uint64_t>(&control), 17, 11};
        registration.runtime_address = 0x9000;
        registration.binding_epoch = 29;
        registration.run_result_address = reinterpret_cast<uint64_t>(&result);
        registration.callable_generation = 31;
        registration.tensor_count = registration.scalar_count = 1;
        registration.launched_threads = 2;
        registration.worker_count = 3;
        registration.callable_id = 0;
        packet.resize(KERNEL_CAPTURE_INVOCATION_OFFSET + kernel_invocation_bytes(1, 1));
        registration.packet_bytes = static_cast<uint32_t>(packet.size());
        header.runtime_args = reinterpret_cast<Runtime *>(0x9000);
        header.run_result_epoch = 29;
        header.run_result_data_base = registration.run_result_address;
        header.entry_args_source = static_cast<uint32_t>(EntryArgsSource::KernelCaptureInvocation);
        header.entry_args_offset = LAUNCH_ENVELOPE_HEADER_BYTES;
        header.entry_tensor_count = header.entry_scalar_count = 1;
        std::memcpy(packet.data(), &header, sizeof(header));
        std::memcpy(
            packet.data() + LAUNCH_ENVELOPE_HEADER_BYTES, &registration.reference, sizeof(registration.reference)
        );
        ChipTensor tensor{};
        tensor.buffer = {0x10000, 64};
        tensor.dtype = DataType::FLOAT32;
        tensor.address_space = AddressSpace::DEVICE;
        tensor.ndims = 1;
        tensor.shapes[0] = 4;
        tensor.strides[0] = 1;
        EntryArgsStorage args;
        args.add_tensor(Tensor::from_boundary(tensor));
        args.add_scalar(7);
        ASSERT_TRUE(encode_kernel_invocation(
            args, identity, {0x9000, 29}, packet.data() + KERNEL_CAPTURE_INVOCATION_OFFSET,
            packet.size() - KERNEL_CAPTURE_INVOCATION_OFFSET
        ));
    }
    auto lookup() {
        return [this](int32_t id) {
            return id == 0 ? identity : KernelCallableIdentity{};
        };
    }
    void register_owner() {
        ASSERT_EQ(registry.update(registration, lookup(), [](const void *, size_t) {}), 0);
    }
    void finish_round(bool publish_ok = true) {
        if (publish_ok) {
            result.published = control.replay_epoch;
            result.verdict = static_cast<uint32_t>(DeviceRunVerdict::Ok);
        }
        registry.finish(packet.data(), control.replay_epoch, 0);
        registry.finish(packet.data(), control.replay_epoch, 0);
    }
    KernelCaptureControl control{};
    DeviceRunResultRegion result{};
    KernelCaptureRegistrationArgs registration{};
    KernelCallableIdentity identity{0, 31, 1, 1};
    KernelArgs header{};
    std::vector<unsigned char> packet;
    KernelCaptureRegistry registry;
};

TEST_F(KernelCaptureDevice, PointerAndGenerationCannotSelfRegister) {
    const auto before = control;
    EXPECT_NE(registry.admit(packet.data(), lookup()), 0);
    EXPECT_EQ(std::memcmp(&before, &control, sizeof(control)), 0);
}

TEST_F(KernelCaptureDevice, HeaderPaddingCannotSupplyCaptureIdentity) {
    register_owner();
    const auto before = control;
    std::memset(packet.data() + LAUNCH_ENVELOPE_HEADER_BYTES, 0, sizeof(KernelCaptureReference));
    std::memcpy(packet.data() + sizeof(KernelArgs), &registration.reference, sizeof(registration.reference));
    EXPECT_NE(registry.admit(packet.data(), lookup()), 0);
    EXPECT_EQ(std::memcmp(&before, &control, sizeof(control)), 0);
}

TEST_F(KernelCaptureDevice, ExecutionDecodesCapturePayloadAndOrdinaryEagerAtTheirOwnOffsets) {
    Runtime runtime;
    runtime.dev.entry_args_source_ = static_cast<uint32_t>(EntryArgsSource::KernelInvocation);
    runtime.dev.entry_tensor_count_ = runtime.dev.entry_scalar_count_ = 1;
    runtime.set_active_callable_id(0);
    EntryArgsStorage input;
    ASSERT_TRUE(consume_kernel_invocation(
        packet.data() + KERNEL_CAPTURE_INVOCATION_OFFSET, packet.size() - KERNEL_CAPTURE_INVOCATION_OFFSET, identity,
        {0x9000, 29}, &input
    ));
    ASSERT_TRUE(encode_kernel_invocation(
        input, identity, {reinterpret_cast<uint64_t>(&runtime), 29}, packet.data() + KERNEL_CAPTURE_INVOCATION_OFFSET,
        packet.size() - KERNEL_CAPTURE_INVOCATION_OFFSET
    ));
    PlatformEntryArgs view{};
    view.args_base = packet.data();
    view.offset = LAUNCH_ENVELOPE_HEADER_BYTES;
    view.tensor_count = view.scalar_count = 1;
    view.source = static_cast<uint32_t>(EntryArgsSource::KernelCaptureInvocation);
    view.binding_epoch = 29;
    EntryArgsStorage output;
    ASSERT_TRUE(admit_kernel_launch_entry(runtime, view, identity, 29, &output));
    EXPECT_EQ(output.scalar(0), 7U);
    auto eager = packet;
    eager.erase(eager.begin() + LAUNCH_ENVELOPE_HEADER_BYTES, eager.begin() + KERNEL_CAPTURE_INVOCATION_OFFSET);
    view.args_base = eager.data();
    view.source = static_cast<uint32_t>(EntryArgsSource::KernelInvocation);
    view.binding_epoch = 0;
    ASSERT_TRUE(admit_kernel_launch_entry(runtime, view, identity, 29, &output));
    EXPECT_EQ(output.scalar(0), 7U);
}

TEST_F(KernelCaptureDevice, SnapshotBindingStaysFixedWhileReplayEpochChanges) {
    register_owner();
    for (uint64_t epoch : {1U, 2U, 3U}) {
        ASSERT_EQ(registry.admit(packet.data(), lookup()), 0);
        auto decoded = header;
        uint64_t binding = 0;
        ASSERT_EQ(registry.enter(packet.data(), &decoded, &binding), 1);
        EXPECT_EQ(decoded.run_result_epoch, epoch);
        EXPECT_EQ(binding, 29U);
        finish_round();
        EXPECT_EQ(control.completed_epoch, epoch);
    }
}

TEST_F(KernelCaptureDevice, SlotReuseRejectsAndRetainsTheFirstFailure) {
    register_owner();
    ++identity.generation;
    ASSERT_EQ(registry.admit(packet.data(), lookup()), 0);
    EXPECT_EQ(control.decision, static_cast<uint32_t>(CaptureDecision::Rejected));
    EXPECT_NE(control.first_error, 0);
    const int error = control.first_error;
    finish_round();
    --identity.generation;
    ASSERT_EQ(registry.admit(packet.data(), lookup()), 0);
    EXPECT_EQ(control.decision, static_cast<uint32_t>(CaptureDecision::Rejected));
    EXPECT_EQ(control.first_error, error);
}

TEST_F(KernelCaptureDevice, UnregisterRevokesEvenBeforeAnIdIsReused) {
    register_owner();
    auto revoke = registration;
    revoke.action = static_cast<uint32_t>(CaptureRegistrationAction::RevokeCallable);
    ASSERT_EQ(registry.update(revoke, lookup(), [](const void *, size_t) {}), 0);
    ASSERT_EQ(registry.admit(packet.data(), lookup()), 0);
    auto decoded = header;
    uint64_t binding = 0;
    EXPECT_EQ(registry.enter(packet.data(), &decoded, &binding), 0);
    EXPECT_EQ(control.decision, static_cast<uint32_t>(CaptureDecision::Rejected));
}

TEST_F(KernelCaptureDevice, StaleContextIsRejectedBeforeControlWrites) {
    register_owner();
    auto stale = registration.reference;
    ++stale.context_generation;
    std::memcpy(packet.data() + LAUNCH_ENVELOPE_HEADER_BYTES, &stale, sizeof(stale));
    const auto before = control;
    EXPECT_NE(registry.admit(packet.data(), lookup()), 0);
    EXPECT_EQ(std::memcmp(&before, &control, sizeof(control)), 0);
}

TEST_F(KernelCaptureDevice, MalformedTensorCannotEnterTheExecutionImage) {
    register_owner();
    ChipTensor malformed{};
    const size_t tensor_offset =
        KERNEL_CAPTURE_INVOCATION_OFFSET + sizeof(SimplerKernelInvocationHeader) + sizeof(EagerKernelBinding);
    std::memcpy(packet.data() + tensor_offset, &malformed, sizeof(malformed));
    ASSERT_EQ(registry.admit(packet.data(), lookup()), 0);
    EXPECT_EQ(control.decision, static_cast<uint32_t>(CaptureDecision::Rejected));
}

TEST_F(KernelCaptureDevice, ReleaseRevokesRegistrationAndIsRetryable) {
    register_owner();
    auto revoke = registration;
    revoke.action = static_cast<uint32_t>(CaptureRegistrationAction::Revoke);
    ASSERT_EQ(registry.update(revoke, lookup(), [](const void *, size_t) {}), 0);
    EXPECT_EQ(registry.update(revoke, lookup(), [](const void *, size_t) {}), 0);
    const auto before = control;
    EXPECT_NE(registry.admit(packet.data(), lookup()), 0);
    EXPECT_EQ(std::memcmp(&before, &control, sizeof(control)), 0);
}

TEST_F(KernelCaptureDevice, MissingMainlineTerminalIsNotSuccess) {
    register_owner();
    ASSERT_EQ(registry.admit(packet.data(), lookup()), 0);
    finish_round(false);
    EXPECT_EQ(control.first_error, PTO_RUNTIME_ERR_INTERNAL);
    ASSERT_EQ(registry.admit(packet.data(), lookup()), 0);
    EXPECT_EQ(control.decision, static_cast<uint32_t>(CaptureDecision::Rejected));
}

TEST_F(KernelCaptureDevice, MainlineTerminalPreservesSignedErrorBeforeQueuedReplay) {
    register_owner();
    ASSERT_EQ(registry.admit(packet.data(), lookup()), 0);
    result.published = 1;
    result.verdict = static_cast<uint32_t>(DeviceRunVerdict::Error);
    result.completion_code = -9;
    result.code_source = static_cast<uint32_t>(DeviceRunCodeSource::Header);
    finish_round(false);
    EXPECT_EQ(control.first_error, -9);
    ASSERT_EQ(registry.admit(packet.data(), lookup()), 0);
    EXPECT_EQ(control.decision, static_cast<uint32_t>(CaptureDecision::Rejected));
    finish_round(false);
    EXPECT_EQ(control.first_error, -9);
}

TEST_F(KernelCaptureDevice, AnIncompletePreviousGroupCannotReleaseAnotherRuntime) {
    register_owner();
    ASSERT_EQ(registry.admit(packet.data(), lookup()), 0);
    // Model/runtime failure prevented one native CPU participant from returning.
    registry.finish(packet.data(), 1, 0);
    ASSERT_EQ(registry.admit(packet.data(), lookup()), 0);
    EXPECT_EQ(control.first_error, PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_EQ(control.decision, static_cast<uint32_t>(CaptureDecision::Rejected));
    EXPECT_EQ(control.replay_epoch, 1U);
}

TEST_F(KernelCaptureDevice, ColdRegistrationFreshensReusedControlBeforeReadingIt) {
    register_owner();
    ASSERT_EQ(registry.admit(packet.data(), lookup()), 0);
    finish_round();
    auto revoke = registration;
    revoke.action = static_cast<uint32_t>(CaptureRegistrationAction::Revoke);
    ASSERT_EQ(registry.update(revoke, lookup(), [](const void *, size_t) {}), 0);
    // Model a retained old cache image and a new host-DMA image at the same
    // address. This proves callback ordering, not hardware cache behavior.
    auto initialized = control;
    initialized.capture_generation = ++registration.reference.capture_generation;
    initialized.replay_epoch = initialized.completed_epoch = 0;
    initialized.finished_threads = initialized.arrived_threads = 0;
    initialized.decision = static_cast<uint32_t>(CaptureDecision::Idle);
    EXPECT_NE(registry.update(registration, lookup(), [](const void *, size_t) {}), 0);
    unsigned freshens = 0;
    ASSERT_EQ(
        registry.update(
            registration, lookup(),
            [&](const void *address, size_t bytes) {
                EXPECT_EQ(address, &control);
                EXPECT_EQ(bytes, sizeof(control));
                ++freshens;
                control = initialized;
            }
        ),
        0
    );
    EXPECT_EQ(freshens, 1u);
    std::memcpy(packet.data() + LAUNCH_ENVELOPE_HEADER_BYTES, &registration.reference, sizeof(registration.reference));
    ASSERT_EQ(registry.admit(packet.data(), lookup()), 0);
    EXPECT_EQ(control.replay_epoch, 1u);
    finish_round();
    EXPECT_EQ(control.completed_epoch, 1u);
}

TEST_F(KernelCaptureDevice, InvalidIdentityAndRevocationDoNotFreshenControl) {
    unsigned freshens = 0;
    auto freshen = [&](const void *, size_t) {
        ++freshens;
    };
    auto invalid = registration;
    invalid.callable_generation++;
    EXPECT_NE(registry.update(invalid, lookup(), freshen), 0);
    EXPECT_EQ(freshens, 0u);
    ASSERT_EQ(registry.update(registration, lookup(), freshen), 0);
    EXPECT_EQ(freshens, 1u);
    EXPECT_NE(registry.update(registration, lookup(), freshen), 0);
    auto revoke = registration;
    revoke.action = static_cast<uint32_t>(CaptureRegistrationAction::Revoke);
    EXPECT_EQ(registry.update(revoke, lookup(), freshen), 0);
    EXPECT_EQ(freshens, 1u);
}
}  // namespace
