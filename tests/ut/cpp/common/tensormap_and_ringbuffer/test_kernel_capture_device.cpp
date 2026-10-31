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
        control.runtime_address = reinterpret_cast<uint64_t>(&runtime_image);
        control.binding_epoch = 29;
        control.replay_epoch = control.completed_epoch = control.binding_epoch - 1;
        control.callable_generation = 31;
        control.callable_id = 0;
        control.tensor_count = 1;
        control.scalar_count = 1;
        control.launched_threads = 2;
        control.worker_count = 3;
        registration.reference = {reinterpret_cast<uint64_t>(&control), 17, 11};
        registration.runtime_address = control.runtime_address;
        registration.binding_epoch = 29;
        registration.run_result_address = reinterpret_cast<uint64_t>(&result);
        registration.callable_generation = 31;
        registration.tensor_count = registration.scalar_count = 1;
        registration.launched_threads = 2;
        registration.worker_count = 3;
        registration.callable_id = 0;
        packet.resize(LAUNCH_ENVELOPE_HEADER_BYTES + kernel_invocation_bytes(1, 1));
        registration.packet_bytes = static_cast<uint32_t>(packet.size());
        header.runtime_args = &runtime_image;
        header.run_result_epoch = 29;
        header.run_result_data_base = registration.run_result_address;
        header.entry_args_source = static_cast<uint32_t>(EntryArgsSource::KernelInvocation);
        header.entry_args_offset = LAUNCH_ENVELOPE_HEADER_BYTES;
        header.entry_tensor_count = header.entry_scalar_count = 1;
        save_registered_header();
        std::memcpy(packet.data(), &header, sizeof(header));
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
            args, identity, {control.runtime_address, 29, registration.reference},
            packet.data() + LAUNCH_ENVELOPE_HEADER_BYTES, packet.size() - LAUNCH_ENVELOPE_HEADER_BYTES
        ));
    }
    static constexpr size_t reference_offset() {
        return LAUNCH_ENVELOPE_HEADER_BYTES + sizeof(SimplerKernelInvocationHeader) +
               offsetof(EagerKernelBinding, reference);
    }
    auto lookup() {
        return [this](int32_t id) {
            return id == 0 ? identity : KernelCallableIdentity{};
        };
    }
    void register_owner() {
        save_registered_header();
        ASSERT_EQ(registry.update(registration, lookup(), [](const void *, size_t) {}), 0);
    }
    void save_registered_header() {
        registration.header_bytes = sizeof(header);
        std::memcpy(registration.launch_header, &header, sizeof(header));
    }
    void finish_round(bool publish_ok = true) {
        if (publish_ok) {
            result.published = registration.binding_epoch;
            result.verdict = static_cast<uint32_t>(DeviceRunVerdict::Ok);
        }
        registry.finish(packet.data(), control.replay_epoch, 0);
        registry.finish(packet.data(), control.replay_epoch, 0);
    }
    Runtime runtime_image;
    KernelCaptureControl control{};
    DeviceRunResultRegion result{};
    KernelCaptureRegistrationArgs registration{};
    KernelCallableIdentity identity{0, 31, 1, 1};
    KernelArgs header{};
    std::vector<unsigned char> packet;
    KernelCaptureRegistry registry;
};

TEST_F(KernelCaptureDevice, SingleExecutionRegistrationRetiresAfterAllCpuParticipantsReturn) {
    registration.action = static_cast<uint32_t>(simpler::tmr::CaptureRegistrationAction::RegisterOnce);
    ASSERT_EQ(registry.update(registration, lookup(), [](const void *, size_t) {}), 0);
    ASSERT_EQ(registry.admit(packet.data(), lookup()), 0);
    result.published = registration.binding_epoch;
    result.verdict = static_cast<uint32_t>(DeviceRunVerdict::Ok);
    registry.finish(packet.data(), control.replay_epoch, 0);
    KernelArgs decoded = header;
    uint64_t binding_epoch = 0;
    EXPECT_EQ(registry.enter(packet.data(), &decoded, &binding_epoch), 1);
    registry.finish(packet.data(), control.replay_epoch, 0);
    EXPECT_EQ(control.completed_epoch, control.replay_epoch);
    EXPECT_NE(registry.enter(packet.data(), &decoded, &binding_epoch), 1);
    EXPECT_NE(registry.admit(packet.data(), lookup()), 0);
}

TEST_F(KernelCaptureDevice, AdmissionClearsOldReportsBeforeAcceptedAndRejectedExecutions) {
    register_owner();
    for (const bool rejected : {false, true}) {
        SCOPED_TRACE(rejected);
        for (uint32_t i = 0; i < registration.worker_count; ++i) {
            auto &worker = runtime_image.dev.workers[i];
            worker.aicore_done = i + 1;
            worker.report_epoch = 97;
            worker.task = 0x1000;
            ASSERT_TRUE(aicore_report_accepted(&worker, 0));
        }
        if (rejected) ++identity.generation;
        ASSERT_EQ(registry.admit(packet.data(), lookup()), 0);
        EXPECT_EQ(
            control.decision, static_cast<uint32_t>(rejected ? CaptureDecision::Rejected : CaptureDecision::Admitted)
        );
        for (uint32_t i = 0; i < registration.worker_count; ++i) {
            const auto &worker = runtime_image.dev.workers[i];
            EXPECT_FALSE(aicore_report_accepted(&worker, 0));
            EXPECT_EQ(worker.report_epoch, 97u);
            EXPECT_EQ(worker.task, 0x1000u);
        }
        finish_round();
    }
}

TEST_F(KernelCaptureDevice, RegistrationBoundsTheReportResetRange) {
    registration.worker_count = control.worker_count = RUNTIME_MAX_WORKER + 1;
    EXPECT_NE(registry.update(registration, lookup(), [](const void *, size_t) {}), 0);
}

TEST_F(KernelCaptureDevice, ReplaySequenceDoesNotReplaceTheNativeResultIdentity) {
    register_owner();
    ASSERT_EQ(registry.admit(packet.data(), lookup()), 0);
    finish_round();
    ASSERT_EQ(registry.admit(packet.data(), lookup()), 0);
    KernelArgs decoded = header;
    uint64_t sequence = 0;
    ASSERT_EQ(registry.enter(packet.data(), &decoded, &sequence), 1);
    EXPECT_EQ(decoded.run_result_epoch, registration.binding_epoch);
    EXPECT_EQ(sequence, control.replay_epoch);
    EXPECT_NE(sequence, decoded.run_result_epoch);
}

TEST_F(KernelCaptureDevice, AdmissionClearsThePreviousTerminalBeforeTheNextFork) {
    register_owner();
    ASSERT_EQ(registry.admit(packet.data(), lookup()), 0);
    finish_round();
    ASSERT_NE(result.published, 0u);
    ASSERT_EQ(registry.admit(packet.data(), lookup()), 0);
    EXPECT_EQ(result.published, 0u);
    finish_round(false);
    EXPECT_EQ(control.first_error, PTO_RUNTIME_ERR_INTERNAL);
}

TEST_F(KernelCaptureDevice, PointerAndGenerationCannotSelfRegister) {
    const auto before = control;
    EXPECT_NE(registry.admit(packet.data(), lookup()), 0);
    EXPECT_EQ(std::memcmp(&before, &control, sizeof(control)), 0);
}

TEST_F(KernelCaptureDevice, InitialDeviceEpochIsBoundToTheOrdinaryRunIdentity) {
    control.replay_epoch = control.completed_epoch = 0;
    EXPECT_NE(registry.update(registration, lookup(), [](const void *, size_t) {}), 0);
    control.replay_epoch = control.completed_epoch = registration.binding_epoch - 1;
    register_owner();
    ASSERT_EQ(registry.admit(packet.data(), lookup()), 0);
    EXPECT_EQ(control.replay_epoch, registration.binding_epoch);
    finish_round();
    ASSERT_EQ(registry.admit(packet.data(), lookup()), 0);
    EXPECT_EQ(control.replay_epoch, registration.binding_epoch + 1);
}

TEST_F(KernelCaptureDevice, OrdinaryDiagnosticsUseTheRegisteredLaunchMetadata) {
    registration.enable_profiling_flag = 3;
    registration.device_wall_data_base = 0x80000;
    header.enable_profiling_flag = registration.enable_profiling_flag;
    header.device_wall_data_base = registration.device_wall_data_base;
    std::memcpy(packet.data(), &header, sizeof(header));
    register_owner();
    ASSERT_EQ(registry.admit(packet.data(), lookup()), 0);
    EXPECT_EQ(control.decision, static_cast<uint32_t>(CaptureDecision::Admitted));
    finish_round();
    header.enable_profiling_flag ^= 4;
    std::memcpy(packet.data(), &header, sizeof(header));
    ASSERT_EQ(registry.admit(packet.data(), lookup()), 0);
    EXPECT_EQ(control.decision, static_cast<uint32_t>(CaptureDecision::Rejected));
    EXPECT_EQ(control.first_error, PTO_RUNTIME_ERR_INVALID_ARGUMENT);
    auto rejected_header = header;
    uint64_t binding_epoch = 0;
    EXPECT_EQ(registry.enter(packet.data(), &rejected_header, &binding_epoch), 0);
    EXPECT_EQ(rejected_header.enable_profiling_flag, registration.enable_profiling_flag);
    EXPECT_EQ(rejected_header.device_wall_data_base, registration.device_wall_data_base);
    EXPECT_EQ(rejected_header.runtime_args, &runtime_image);
    EXPECT_EQ(rejected_header.run_result_epoch, registration.binding_epoch);
    EXPECT_EQ(binding_epoch, control.replay_epoch);
}

TEST_F(KernelCaptureDevice, DiagnosticPaddingIsNotASecondRegistrationChannel) {
    auto invalid = registration;
    invalid.diagnostics_reserved = 1;
    EXPECT_NE(registry.update(invalid, lookup(), [](const void *, size_t) {}), 0);
    EXPECT_NE(registry.admit(packet.data(), lookup()), 0);
}

TEST_F(KernelCaptureDevice, ColdHeaderVersionAndExtentAreValidatedBeforeControlAccess) {
    for (const auto invalid_bytes :
         {0u, static_cast<uint32_t>(sizeof(header) - 1),
          static_cast<uint32_t>(sizeof(registration.launch_header) + 1)}) {
        SCOPED_TRACE(invalid_bytes);
        auto invalid = registration;
        invalid.header_bytes = invalid_bytes;
        unsigned reads = 0;
        EXPECT_NE(
            registry.update(
                invalid, lookup(),
                [&](const void *, size_t) {
                    ++reads;
                }
            ),
            0
        );
        EXPECT_EQ(reads, 0u);
    }
    auto invalid = registration;
    invalid.version = 1;
    unsigned reads = 0;
    EXPECT_NE(
        registry.update(
            invalid, lookup(),
            [&](const void *, size_t) {
                ++reads;
            }
        ),
        0
    );
    EXPECT_EQ(reads, 0u);
}

TEST_F(KernelCaptureDevice, RejectedHeaderRestoresTrustedAddressesAndRetainsResultIdentity) {
    header.regs = 0x5000;
    std::memcpy(packet.data(), &header, sizeof(header));
    register_owner();
    auto invalid = header;
    invalid.regs = 0xdead0000;
    invalid.runtime_args = reinterpret_cast<Runtime *>(0xbad0000);
    invalid.run_result_data_base = 0xbad1000;
    std::memcpy(packet.data(), &invalid, sizeof(invalid));
    ASSERT_EQ(registry.admit(packet.data(), lookup()), 0);
    ASSERT_EQ(control.decision, static_cast<uint32_t>(CaptureDecision::Rejected));
    uint64_t binding = 0;
    ASSERT_EQ(registry.enter(packet.data(), &invalid, &binding), 0);
    EXPECT_EQ(invalid.regs, header.regs);
    EXPECT_EQ(invalid.runtime_args, &runtime_image);
    EXPECT_EQ(invalid.run_result_data_base, registration.run_result_address);
    EXPECT_EQ(invalid.run_result_epoch, registration.binding_epoch);
    EXPECT_EQ(binding, control.replay_epoch);
    finish_round(false);
    EXPECT_EQ(control.completed_epoch, control.replay_epoch);
    EXPECT_EQ(control.first_error, PTO_RUNTIME_ERR_INVALID_ARGUMENT);
}

TEST_F(KernelCaptureDevice, NativeHeaderPaddingDoesNotChangeAdmissionIdentity) {
    register_owner();
    const auto change_padding = [](auto &value) {
        using Header = std::decay_t<decltype(value)>;
        if constexpr (!std::has_unique_object_representations_v<Header>) {
            constexpr size_t field_end = offsetof(Header, force_simt_anchor) + sizeof(value.force_simt_anchor);
            static_assert(field_end < sizeof(Header));
            std::memset(reinterpret_cast<unsigned char *>(&value) + field_end, 0xa5, sizeof(Header) - field_end);
        }
    };
    change_padding(header);
    std::memcpy(packet.data(), &header, sizeof(header));
    ASSERT_EQ(registry.admit(packet.data(), lookup()), 0);
    EXPECT_EQ(control.decision, static_cast<uint32_t>(CaptureDecision::Admitted));
    uint64_t binding = 0;
    EXPECT_EQ(registry.enter(packet.data(), &header, &binding), 1);
}

TEST_F(KernelCaptureDevice, NonzeroReservedRequestCannotRegisterOrRevoke) {
    auto invalid = registration;
    invalid.reserved = 1;
    EXPECT_NE(registry.update(invalid, lookup(), [](const void *, size_t) {}), 0);
    EXPECT_NE(registry.admit(packet.data(), lookup()), 0);
    register_owner();
    for (auto action : {CaptureRegistrationAction::Revoke, CaptureRegistrationAction::RevokeCallable}) {
        SCOPED_TRACE(static_cast<uint32_t>(action));
        invalid.action = static_cast<uint32_t>(action);
        EXPECT_NE(registry.update(invalid, lookup(), [](const void *, size_t) {}), 0);
        ASSERT_EQ(registry.admit(packet.data(), lookup()), 0);
        EXPECT_EQ(control.decision, static_cast<uint32_t>(CaptureDecision::Admitted));
        finish_round();
    }
}

TEST_F(KernelCaptureDevice, ColdRegistrationRejectsEveryNonzeroReservedControlField) {
    const auto initialized = control;
    for (size_t field = 0; field <= sizeof(control.reserved_tail) / sizeof(control.reserved_tail[0]); ++field) {
        SCOPED_TRACE(field);
        control = initialized;
        if (field == 0) control.reserved = 1;
        else control.reserved_tail[field - 1] = 1;
        KernelCaptureRegistry candidate;
        unsigned freshens = 0;
        EXPECT_NE(
            candidate.update(
                registration, lookup(),
                [&](const void *, size_t) {
                    ++freshens;
                }
            ),
            0
        );
        EXPECT_EQ(freshens, 1U);
        EXPECT_NE(candidate.admit(packet.data(), lookup()), 0);
    }
    control = initialized;
    register_owner();
    ASSERT_EQ(registry.admit(packet.data(), lookup()), 0);
    finish_round();
    EXPECT_EQ(control.completed_epoch, 29U);
}

TEST_F(KernelCaptureDevice, HeaderPaddingCannotSupplyRegisteredIdentity) {
    register_owner();
    const auto before = control;
    std::memset(packet.data() + reference_offset(), 0, sizeof(KernelCaptureReference));
    std::memcpy(packet.data() + sizeof(KernelArgs), &registration.reference, sizeof(registration.reference));
    EXPECT_NE(registry.admit(packet.data(), lookup()), 0);
    EXPECT_EQ(std::memcmp(&before, &control, sizeof(control)), 0);
}

TEST_F(KernelCaptureDevice, EagerAndRecordedExecutionDecodeTheSameRegisteredInvocation) {
    Runtime runtime;
    runtime.dev.entry_args_source_ = static_cast<uint32_t>(EntryArgsSource::KernelInvocation);
    runtime.dev.entry_tensor_count_ = runtime.dev.entry_scalar_count_ = 1;
    runtime.set_active_callable_id(0);
    EntryArgsStorage input;
    ASSERT_TRUE(consume_kernel_invocation(
        packet.data() + LAUNCH_ENVELOPE_HEADER_BYTES, packet.size() - LAUNCH_ENVELOPE_HEADER_BYTES, identity,
        {control.runtime_address, 29, registration.reference}, &input
    ));
    ASSERT_TRUE(encode_kernel_invocation(
        input, identity, {reinterpret_cast<uint64_t>(&runtime), 29, registration.reference},
        packet.data() + LAUNCH_ENVELOPE_HEADER_BYTES, packet.size() - LAUNCH_ENVELOPE_HEADER_BYTES
    ));
    PlatformEntryArgs view{};
    view.args_base = packet.data();
    view.offset = LAUNCH_ENVELOPE_HEADER_BYTES;
    view.tensor_count = view.scalar_count = 1;
    view.source = static_cast<uint32_t>(EntryArgsSource::KernelInvocation);
    EntryArgsStorage output;
    ASSERT_TRUE(admit_kernel_launch_entry(runtime, view, identity, 29, &output));
    EXPECT_EQ(output.scalar(0), 7U);
    view.invocation_rejected = true;
    EXPECT_FALSE(admit_kernel_launch_entry(runtime, view, identity, 29, &output));
    EXPECT_EQ(output.scalar(0), 7U);
    view.invocation_rejected = false;
    EXPECT_FALSE(admit_kernel_launch_entry(runtime, view, identity, 0, &output));
    view.source = 3;
    EXPECT_FALSE(admit_kernel_launch_entry(runtime, view, identity, 29, &output));
}

TEST_F(KernelCaptureDevice, SnapshotBindingStaysFixedWhileReplayEpochChanges) {
    register_owner();
    for (uint64_t epoch : {29U, 30U, 31U}) {
        ASSERT_EQ(registry.admit(packet.data(), lookup()), 0);
        auto decoded = header;
        uint64_t binding = 0;
        ASSERT_EQ(registry.enter(packet.data(), &decoded, &binding), 1);
        EXPECT_EQ(decoded.run_result_epoch, registration.binding_epoch);
        EXPECT_EQ(binding, epoch);
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
    std::memcpy(packet.data() + reference_offset(), &stale, sizeof(stale));
    const auto before = control;
    EXPECT_NE(registry.admit(packet.data(), lookup()), 0);
    EXPECT_EQ(std::memcmp(&before, &control, sizeof(control)), 0);
}

TEST_F(KernelCaptureDevice, MalformedTensorCannotEnterTheExecutionImage) {
    register_owner();
    ChipTensor malformed{};
    const size_t tensor_offset =
        LAUNCH_ENVELOPE_HEADER_BYTES + sizeof(SimplerKernelInvocationHeader) + sizeof(EagerKernelBinding);
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
    result.published = registration.binding_epoch;
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
    registry.finish(packet.data(), control.replay_epoch, 0);
    runtime_image.dev.workers[0].aicore_done = 7;
    result.published = 71;
    EXPECT_NE(registry.admit(packet.data(), lookup()), 0);
    EXPECT_EQ(control.first_error, PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_EQ(control.decision, static_cast<uint32_t>(CaptureDecision::Admitted));
    EXPECT_EQ(control.replay_epoch, 29U);
    EXPECT_EQ(runtime_image.dev.workers[0].aicore_done, 7u);
    EXPECT_EQ(result.published, 71u);
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
    initialized.replay_epoch = initialized.completed_epoch = initialized.binding_epoch - 1;
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
    std::memcpy(packet.data() + reference_offset(), &registration.reference, sizeof(registration.reference));
    ASSERT_EQ(registry.admit(packet.data(), lookup()), 0);
    EXPECT_EQ(control.replay_epoch, 29u);
    finish_round();
    EXPECT_EQ(control.completed_epoch, 29u);
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
