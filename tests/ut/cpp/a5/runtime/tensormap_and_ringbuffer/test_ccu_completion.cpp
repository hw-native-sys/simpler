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
#include <memory>

#include "backend/ccu/ccu_completion_kernel.h"
#include "scheduler/scheduler.h"
#include "utils/device_arena.h"

namespace {
namespace wire = simpler::ccu::device;

struct alignas(128) Image {
    wire::Workspace header{};
    wire::MemoryRange ranges[1]{};
    wire::ProgramBinding bindings[2]{};
    wire::Request request{};
};
struct alignas(128) Directory {
    wire::DomainDirectory header{};
    wire::InstanceEntry entries[2]{};
};
struct alignas(128) PlatformImage {
    CcuDeviceContext header{};
    CcuAdmissionRecord admission{};
    CcuProgramBinding bindings[2]{};
    CcuMemoryRange range{};
    CcuRequestRecord record{};
    CcuPositionalArguments arguments{};
};

// Replaces only the hardware boundary. Snapshot, validation, submit, polling,
// retirement and scheduler dependency handling below use production code.
struct FakeIo {
    static inline uint64_t clock = 100;
    static inline uint32_t triggers = 0;
    static inline bool finish_on_trigger = false;
    static inline bool owner_seen = false;
    static inline DeferredCompletionSlab *owner = nullptr;
    static inline Image *image = nullptr;
    static uint64_t now_ns() { return clock; }
    static void sync(void *, uint64_t) {}
    static void finish(wire::Outcome outcome = wire::Outcome::SUCCEEDED) {
        auto &r = image->request;
        auto &c = r.completion;
        c.generation = r.generation;
        c.binding_index = r.binding_index;
        c.outcome = outcome;
        c.sequence = r.sequence;
    }
    static void trigger(const wire::ProgramBinding &) {
        ++triggers;
        owner_seen = owner && owner->count != 0;
        if (!finish_on_trigger) return;
        const auto &args = image->request.arguments;
        std::memcpy(
            reinterpret_cast<void *>(args.tensors[1].addr), reinterpret_cast<const void *>(args.tensors[0].addr),
            args.scalars[0].bits * sizeof(float)
        );
        finish();
    }
};
using Protocol = wire::Protocol<FakeIo>;

class CcuCompletionTest : public ::testing::Test {
protected:
    Image image{};
    Directory directory{};
    CommContext domain{};
    DeferredCompletionSlab slab{};
    AsyncCtx task{};
    float data[512]{};
    PlatformImage platform_image{};
    uint64_t doorbell{};

    void SetUp() override {
        FakeIo::clock = 100;
        FakeIo::triggers = 0;
        FakeIo::finish_on_trigger = false;
        FakeIo::owner_seen = false;
        FakeIo::owner = &slab;
        FakeIo::image = &image;
        auto &w = image.header;
        w.version = wire::PROTOCOL_VERSION;
        w.header_bytes = sizeof(w);
        w.image_bytes = sizeof(image);
        w.domain_owner = 10;
        w.domain_id = 20;
        w.generation = 3;
        w.timeout_ns = 1000;
        w.ranges_offset = offsetof(Image, ranges);
        w.range_count = 1;
        w.bindings_offset = offsetof(Image, bindings);
        w.binding_count = 2;
        w.rank_count = 2;
        image.ranges[0] = {reinterpret_cast<uint64_t>(data), sizeof(data), 123};
        for (uint32_t i = 0; i < 2; ++i)
            image.bindings[i] = {7, i, offsetof(Image, request), 0x1000, 1, i + 100};
        auto &d = directory.header;
        d.version = wire::PROTOCOL_VERSION;
        d.header_bytes = sizeof(d);
        d.image_bytes = sizeof(directory);
        d.domain_owner = w.domain_owner;
        d.domain_id = w.domain_id;
        d.generation = w.generation;
        d.entries_offset = offsetof(Directory, entries);
        d.entry_count = 1;
        d.rank_count = 2;
        directory.entries[0] = {7, 0, reinterpret_cast<uint64_t>(&image), sizeof(image)};
        domain.rankNum = 2;
        set_comm_workspace(domain, CommWorkspaceKind::CCU, reinterpret_cast<uint64_t>(&directory), sizeof(directory));
        task = AsyncCtx::make(TaskId::make(0, 1), &slab);
        task.callable_id = 7;
    }

    Tensor tensor(uint32_t offset = 0, uint32_t count = 64) {
        Tensor t{};
        const uint32_t shape[] = {512};
        t.init_external(data, sizeof(data), shape, 1, DataType::FLOAT32, 0);
        t.start_offset = offset;
        t.shapes[0] = count;
        return t;
    }
    CcuCallArgs arguments(uint32_t offset = 0, uint32_t count = 64) {
        CcuCallArgs args;
        args.add_scalar(uint64_t(count));  // Grouped tensors still precede scalars.
        args.add_tensor(tensor(offset, count));
        args.add_tensor(tensor(256 + offset, count));
        return args;
    }
    wire::Reference reference() { return {&image.header, 0, image.request.sequence}; }
    bool send(const CcuCallDescriptor &call) { return simpler::ccu::detail::send<FakeIo>(task, call); }
    void use_platform() {
        auto &h = platform_image.header;
        h.abi_version = CCU_DEVICE_ABI_VERSION;
        h.struct_size = sizeof(h);
        h.image_size = sizeof(platform_image);
        h.allocation_id = image.header.domain_id;
        h.domain_owner = image.header.domain_owner;
        h.callable_id = 7;
        h.session_epoch = image.header.generation;
        h.next_sequence = 1;
        h.rank_count = 2;
        h.timeout_ns = 5000000000;
        h.admission_offset = offsetof(PlatformImage, admission);
        h.bindings_offset = offsetof(PlatformImage, bindings);
        h.binding_count = 2;
        h.ranges_offset = offsetof(PlatformImage, range);
        h.range_count = 1;
        h.record_offset = offsetof(PlatformImage, record);
        h.arguments_offset = offsetof(PlatformImage, arguments);
        h.arguments_size = sizeof(CcuPositionalArguments);
        platform_image.range = {1, reinterpret_cast<uint64_t>(data), sizeof(data), 456};
        for (uint32_t i = 0; i < 2; ++i) {
            auto &binding = platform_image.bindings[i];
            binding.session_epoch = h.session_epoch;
            binding.program_index = i;
            binding.gate_addr = reinterpret_cast<uint64_t>(&doorbell);
            binding.argument_bytes = sizeof(CcuPositionalArguments);
            binding.reserved = i + 1;
        }
        image.header.platform_context = reinterpret_cast<uint64_t>(&platform_image);
        image.header.platform_context_bytes = sizeof(platform_image);
    }
    void finish_platform() {
        auto &r = platform_image.record;
        r.completed_epoch = r.session_epoch;
        r.completed_program = r.program_index;
        r.completed_state = CcuRequestState::SUCCEEDED;
        r.status = CcuStatus::OK;
        r.completed_sequence = r.sequence;
    }
};

TEST_F(CcuCompletionTest, InvalidTensorDiagnosticsPreserveFirstFailureAndMetadata) {
    auto bad = tensor();
    bad.dtype = DataType::INT32;
    bad.address_space = AddressSpace::HOST;
    CcuCallArgs args;
    args.add_tensor(tensor());
    args.add_tensor(bad);
    args.add_scalar(uint64_t{4});
    args.add_tensor(tensor());
    const uint32_t failure = args.failure_reason();
    EXPECT_FALSE(args.valid());
    EXPECT_EQ(failure & 0xf, 2U);
    EXPECT_EQ((failure >> 4) & 0x1f, 1U);
    EXPECT_EQ((failure >> 9) & 0xff, static_cast<uint32_t>(DataType::INT32));
    EXPECT_EQ((failure >> 17) & 0xff, static_cast<uint32_t>(AddressSpace::HOST));
    EXPECT_EQ(failure >> 25, 1U);
    ASSERT_FALSE(send(CcuCall(&domain, 0, args)));
    EXPECT_EQ(slab.entries[0].expected_value, failure);
    EXPECT_EQ((slab.entries[0].addr >> 32) & 0xff, 1U);
    EXPECT_EQ(FakeIo::triggers, 0U);
}

TEST_F(CcuCompletionTest, InvalidTensorDiagnosticsDistinguishViewChecks) {
    for (uint32_t reason = 3; reason <= 11; ++reason) {
        auto bad = tensor();
        if (reason == 3) bad.address_space = AddressSpace::HOST;
        if (reason == 4) bad.ndims = 0;
        if (reason == 5) bad.shapes[0] = 0;
        if (reason == 6) bad.strides[0] = 2;
        if (reason == 7) {
            bad.ndims = 3;
            bad.shapes[0] = UINT32_MAX;
            bad.shapes[1] = 3;
            bad.shapes[2] = UINT32_MAX / 3;
            bad.strides[0] = UINT32_MAX;
            bad.strides[1] = UINT32_MAX / 3;
            bad.strides[2] = 1;
        }
        if (reason == 8) bad.start_offset = UINT64_MAX;
        if (reason == 9) bad.buffer.size = UINT64_MAX;
        if (reason == 10) ++bad.buffer.addr;
        if (reason == 11) bad.start_offset = 512;
        CcuCallArgs args;
        args.add_tensor(bad);
        EXPECT_FALSE(args.valid()) << reason;
        EXPECT_EQ(args.failure_reason() & 0xf, reason);
    }
}

TEST_F(CcuCompletionTest, RejectionDiagnosticPreservesAcceptedCompletions) {
    ASSERT_TRUE(send(CcuCall(&domain, 0, arguments())));
    const DeferredCompletionEntry accepted = slab.entries[0];
    ASSERT_FALSE(send(CcuCall(&domain, 99, arguments())));
    EXPECT_EQ(slab.error_code, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
    EXPECT_EQ(slab.count, 1U);
    EXPECT_EQ(std::memcmp(&accepted, &slab.entries[0], sizeof(accepted)), 0);
    const auto &diagnostic = slab.entries[1];
    EXPECT_EQ(diagnostic._pad, CCU_ACCEPT_DIAGNOSTIC_MAGIC);
    EXPECT_EQ((diagnostic.addr >> 32) & 0xff, 3U);
    EXPECT_EQ((diagnostic.addr >> 40) & 0xff, 15U);
    EXPECT_EQ(static_cast<int32_t>(diagnostic.backend_cookie), 7);
    EXPECT_EQ(diagnostic.backend_cookie >> 32, 0U);
    EXPECT_EQ(diagnostic.addr >> 48, 2U);
    EXPECT_EQ(FakeIo::triggers, 1U);
}

TEST_F(CcuCompletionTest, RejectionDiagnosticRespectsCapacityAndFirstError) {
    const auto call = CcuCall(&domain, 0, arguments());
    slab.entries[0]._pad = 123;
    task.completion_capacity = 0;
    EXPECT_FALSE(simpler::ccu::detail::reject_accept(task, call, 1));
    EXPECT_EQ(slab.entries[0]._pad, 123U);
    EXPECT_EQ(slab.count, 0U);
    EXPECT_EQ(slab.error_code, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
    task.completion_capacity = MAX_COMPLETIONS_PER_TASK + 1;
    slab.count = MAX_COMPLETIONS_PER_TASK;
    slab.error_code = SIMPLER_ERROR_NONE;
    EXPECT_FALSE(simpler::ccu::detail::reject_accept(task, call, 2));
    EXPECT_EQ(slab.count, MAX_COMPLETIONS_PER_TASK);
    slab.count = 0;
    slab.error_code = SIMPLER_ERROR_CCU_FAILED;
    EXPECT_FALSE(simpler::ccu::detail::reject_accept(task, call, 3));
    EXPECT_EQ(slab.error_code, SIMPLER_ERROR_CCU_FAILED);
    EXPECT_EQ(slab.entries[0]._pad, 123U);
}

TEST_F(CcuCompletionTest, RejectionDiagnosticBoundsDirectoryReads) {
    directory.header.entries_offset = UINT64_MAX;
    ASSERT_FALSE(send(CcuCall(&domain, 0, arguments())));
    EXPECT_EQ(slab.entries[0]._pad, CCU_ACCEPT_DIAGNOSTIC_MAGIC);
    EXPECT_EQ((slab.entries[0].addr >> 32) & 0xff, 2U);
    EXPECT_EQ(slab.entries[0].expected_value, 1U << 7);
    EXPECT_EQ(slab.count, 0U);
}

TEST_F(CcuCompletionTest, PlatformOwnsSnapshotAndCompletionAcrossAivReturn) {
    use_platform();
    auto args = arguments();
    ASSERT_TRUE(send(CcuCall(&domain, 0, args)));
    EXPECT_EQ(slab.count, 1u);
    EXPECT_EQ(doorbell, (uint64_t{1} << 63) | 1);
    EXPECT_EQ(FakeIo::triggers, 0u);
    EXPECT_EQ(platform_image.arguments.tensors[0].token, 456u);
    EXPECT_EQ(platform_image.arguments.scalar_count, 1u);
    args.add_scalar(uint64_t{99});
    EXPECT_EQ(platform_image.arguments.scalar_count, 1u);
    wire::Reference first{&image.header, 0, platform_image.record.sequence};
    EXPECT_EQ(Protocol::poll(first), wire::Outcome::PENDING);
    EXPECT_FALSE(Protocol::retire(first));
    finish_platform();
    EXPECT_EQ(Protocol::poll(first), wire::Outcome::SUCCEEDED);
    EXPECT_TRUE(Protocol::synchronize_arguments(first));
    ASSERT_TRUE(Protocol::retire(first));
    EXPECT_FALSE(Protocol::retire(first));
    wire::Reference second;
    ASSERT_TRUE(Protocol::resolve(&image.header, sizeof(image), 7, 1, second));
    CcuStatus status = CcuStatus::INVALID_ARGUMENT;
    ASSERT_TRUE(Protocol::reserve(second, arguments(8, 32).values(), &status));
    EXPECT_EQ(status, CcuStatus::OK);
    EXPECT_EQ(second.sequence, first.sequence + 1);
    EXPECT_FALSE(Protocol::reserve(second, arguments(8, 32).values(), &status));
    EXPECT_EQ(status, CcuStatus::BUSY);
    EXPECT_EQ(platform_image.record.program_index, 1u);
    EXPECT_FALSE(Protocol::cancel(first));
    EXPECT_TRUE(Protocol::cancel(second));
}

TEST_F(CcuCompletionTest, PlatformRejectsStaleContextInvalidIdAndOutsideTensorWithoutTrigger) {
    use_platform();
    wire::Reference ref;
    EXPECT_FALSE(Protocol::resolve(&image.header, sizeof(image), 8, 0, ref));
    EXPECT_FALSE(Protocol::resolve(&image.header, sizeof(image), 7, 2, ref));
    ++platform_image.header.session_epoch;
    EXPECT_FALSE(Protocol::resolve(&image.header, sizeof(image), 7, 0, ref));
    --platform_image.header.session_epoch;
    ++platform_image.header.callable_id;
    EXPECT_FALSE(Protocol::resolve(&image.header, sizeof(image), 7, 0, ref));
    --platform_image.header.callable_id;
    ++platform_image.header.domain_owner;
    EXPECT_FALSE(Protocol::resolve(&image.header, sizeof(image), 7, 0, ref));
    --platform_image.header.domain_owner;
    ++platform_image.header.allocation_id;
    EXPECT_FALSE(Protocol::resolve(&image.header, sizeof(image), 7, 0, ref));
    --platform_image.header.allocation_id;
    ASSERT_TRUE(Protocol::resolve(&image.header, sizeof(image), 7, 0, ref));
    auto args = arguments().values();
    args.tensors[0].addr += sizeof(data);
    CcuStatus status = CcuStatus::OK;
    EXPECT_FALSE(Protocol::reserve(ref, args, &status));
    EXPECT_EQ(status, CcuStatus::INVALID_ARGUMENT);
    EXPECT_EQ(CcuSlotState(platform_image.record.slot_state), CcuSlotState::FREE);
    EXPECT_EQ(doorbell, 0u);
}

TEST_F(CcuCompletionTest, PlatformCloseBeforeSubmitCanCancelWithoutOrphanRequest) {
    use_platform();
    wire::Reference ref;
    ASSERT_TRUE(Protocol::resolve(&image.header, sizeof(image), 7, 0, ref));
    ASSERT_TRUE(Protocol::reserve(ref, arguments().values()));
    platform_image.admission.closed = 1;
    EXPECT_FALSE(Protocol::submit(ref));
    EXPECT_TRUE(Protocol::cancel(ref));
    EXPECT_EQ(CcuSlotState(platform_image.record.slot_state), CcuSlotState::FREE);
    EXPECT_EQ(doorbell, 0u);
}

TEST_F(CcuCompletionTest, PlatformWrongCompletionGenerationNeverReleasesRequest) {
    use_platform();
    ASSERT_TRUE(send(CcuCall(&domain, 0, arguments())));
    wire::Reference ref{&image.header, 0, platform_image.record.sequence};
    finish_platform();
    ++platform_image.record.completed_epoch;
    EXPECT_EQ(Protocol::poll(ref), wire::Outcome::UNCONFIRMED);
    EXPECT_FALSE(Protocol::retire(ref));
    EXPECT_EQ(wire::Admission(image.header.admission), wire::Admission::UNCONFIRMED);
    EXPECT_EQ(CcuSlotState(platform_image.record.slot_state), CcuSlotState::SUBMITTED);
}

TEST_F(CcuCompletionTest, BothTransportsRejectUnsupportedScalarRepresentations) {
    for (const bool platform : {false, true}) {
        if (platform) use_platform();
        wire::Reference ref;
        ASSERT_TRUE(Protocol::resolve(&image.header, sizeof(image), 7, 0, ref));
        auto args = arguments().values();
        args.scalars[0].width = 16;
        EXPECT_FALSE(Protocol::reserve(ref, args));
        args.scalars[0].width = 32;
        args.scalars[0].bits = uint64_t{1} << 32;
        EXPECT_FALSE(Protocol::reserve(ref, args));
        args.scalars[0].bits = 1;
        args.scalars[0].kind = static_cast<wire::ScalarKind>(99);
        EXPECT_FALSE(Protocol::reserve(ref, args));
        EXPECT_EQ(wire::Slot(image.request.slot), wire::Slot::FREE);
        EXPECT_EQ(CcuSlotState(platform_image.record.slot_state), CcuSlotState::FREE);
    }
    EXPECT_EQ(doorbell, 0u);
    EXPECT_EQ(FakeIo::triggers, 0u);
}

TEST_F(CcuCompletionTest, OldDirectoryVersionIsRejectedBeforeAccessingPlatformContext) {
    use_platform();
    directory.header.version = wire::PROTOCOL_VERSION - 1;
    EXPECT_FALSE(send(CcuCall(&domain, 0, arguments())));
    EXPECT_EQ(doorbell, 0u);
    EXPECT_EQ(CcuSlotState(platform_image.record.slot_state), CcuSlotState::FREE);
}

TEST_F(CcuCompletionTest, ConstructionDoesNotReserveAndAcceptanceOwnsSnapshot) {
    auto args = arguments();
    auto call = CcuCall(&domain, 0, args);
    args.add_scalar(uint64_t(99));
    EXPECT_EQ(wire::Slot(image.request.slot), wire::Slot::FREE);
    EXPECT_EQ(FakeIo::triggers, 0u);
    ASSERT_TRUE(send(call));
    EXPECT_TRUE(FakeIo::owner_seen);
    EXPECT_EQ(image.request.arguments.scalar_count, 1u);
    EXPECT_EQ(image.request.arguments.scalars[0].bits, 64u);
    EXPECT_EQ(image.request.arguments.tensors[0].credential, 123u);
    call.parameters.add_scalar(uint64_t(2));
    EXPECT_EQ(image.request.arguments.scalar_count, 1u);
    EXPECT_EQ(wire::Slot(image.request.slot), wire::Slot::SUBMITTED);
}

TEST_F(CcuCompletionTest, FullSlabAndInvalidContextNeverTriggerOrReserve) {
    slab.count = task.completion_capacity;
    EXPECT_FALSE(send(CcuCall(&domain, 0, arguments())));
    EXPECT_EQ(wire::Slot(image.request.slot), wire::Slot::FREE);
    EXPECT_EQ(FakeIo::triggers, 0u);
    slab.count = 0;
    slab.error_code = 0;
    task.completion_entries = nullptr;
    EXPECT_FALSE(send(CcuCall(&domain, 0, arguments())));
    EXPECT_EQ(FakeIo::triggers, 0u);
}

TEST_F(CcuCompletionTest, InvalidViewsIdsAndExpiredDomainsFailBeforeSubmission) {
    auto expect_rejected = [&](const CcuCallDescriptor &call) {
        slab.error_code = 0;
        EXPECT_FALSE(send(call));
        EXPECT_EQ(wire::Slot(image.request.slot), wire::Slot::FREE);
        EXPECT_EQ(FakeIo::triggers, 0u);
    };
    expect_rejected(CcuCall(&domain, 2, arguments()));
    task.callable_id = 8;
    expect_rejected(CcuCall(&domain, 0, arguments()));
    task.callable_id = 7;
    directory.header.generation++;
    expect_rejected(CcuCall(&domain, 0, arguments()));
    directory.header.generation--;
    for (int variant = 0; variant < 5; ++variant) {
        auto t = tensor();
        if (variant == 0) t.buffer.addr += sizeof(data);
        if (variant == 1) t.start_offset = 500;
        if (variant == 2) t.strides[0] = 2;
        if (variant == 3) t.start_offset = UINT64_MAX;
        if (variant == 4) t.dtype = DataType::INT32;
        CcuCallArgs args;
        args.add_tensor(t);
        expect_rejected(CcuCall(&domain, 0, args));
    }
    CcuCallArgs full;
    for (uint32_t i = 0; i <= wire::MAX_SCALARS; ++i)
        full.add_scalar(i);
    expect_rejected(CcuCall(&domain, 0, full));
}

TEST_F(CcuCompletionTest, DuplicateSubmissionAndStaleCompletionCannotCompleteNextRound) {
    const auto call = CcuCall(&domain, 0, arguments());
    ASSERT_TRUE(send(call));
    const auto first = reference();
    EXPECT_FALSE(Protocol::submit(first));
    EXPECT_FALSE(Protocol::cancel(first));
    FakeIo::finish();
    ASSERT_TRUE(Protocol::retire(first));
    EXPECT_FALSE(Protocol::retire(first));
    slab.count = 0;
    ASSERT_TRUE(send(call));
    EXPECT_EQ(image.request.sequence, first.sequence + 1);
    EXPECT_EQ(Protocol::poll(reference()), wire::Outcome::PENDING);
    image.request.completion.sequence = image.request.sequence;
    image.request.completion.generation--;
    EXPECT_EQ(Protocol::poll(reference()), wire::Outcome::PENDING);
    FakeIo::finish();
    EXPECT_TRUE(Protocol::retire(reference()));
}

TEST_F(CcuCompletionTest, TimeoutPoisonsAdmissionAndRetainsActiveSlot) {
    ASSERT_TRUE(send(CcuCall(&domain, 0, arguments())));
    FakeIo::clock += image.header.timeout_ns;
    EXPECT_EQ(Protocol::poll(reference()), wire::Outcome::UNCONFIRMED);
    EXPECT_FALSE(Protocol::retire(reference()));
    EXPECT_EQ(wire::Slot(image.request.slot), wire::Slot::SUBMITTED);
    EXPECT_EQ(wire::Admission(image.header.admission), wire::Admission::UNCONFIRMED);
    EXPECT_FALSE(send(CcuCall(&domain, 1, arguments())));
    EXPECT_EQ(FakeIo::triggers, 1u);
}

TEST_F(CcuCompletionTest, WaitExecutes1024IndependentRoundsAndPreservesGuards) {
    FakeIo::finish_on_trigger = true;
    for (uint32_t round = 0; round < 1024; ++round) {
        const uint32_t offset = (round % 4) * 16;
        const uint32_t count = (1 + round % 4) * 16;
        for (uint32_t j = 0; j < 512; ++j)
            data[j] = j < 256 ? float(round + j) : -1;
        auto call = CcuCall(&domain, round % 2, arguments(offset, count));
        ASSERT_TRUE(simpler::ccu::detail::wait<FakeIo>(task, call));
        EXPECT_EQ(wire::Slot(image.request.slot), wire::Slot::FREE);
        EXPECT_EQ(image.request.sequence, uint64_t(round) + 1);
        for (uint32_t j = 0; j < 256; ++j)
            EXPECT_EQ(data[256 + j], j >= offset && j < offset + count ? data[j] : -1);
    }
    EXPECT_EQ(FakeIo::triggers, 1024u);
    EXPECT_EQ(slab.count, 0u);
}

TEST_F(CcuCompletionTest, CallableZeroIdsResolveToDifferentStableInstances) {
    Image other = image;
    for (auto &b : other.bindings)
        b.callable_id = 8;
    directory.entries[1] = {8, 0, reinterpret_cast<uint64_t>(&other), sizeof(other)};
    directory.header.entry_count = 2;
    ASSERT_TRUE(send(CcuCall(&domain, 0, arguments())));
    task.callable_id = 8;
    ASSERT_TRUE(send(CcuCall(&domain, 0, arguments())));
    EXPECT_EQ(wire::Slot(other.request.slot), wire::Slot::SUBMITTED);
    EXPECT_NE(slab.entries[0].addr, slab.entries[1].addr);
}

TEST_F(CcuCompletionTest, SchedulerRequiresRetirementAndNormalAivCompletion) {
    ASSERT_TRUE(send(CcuCall(&domain, 0, arguments())));
    // The real scheduler uses a real clock. Keep this request pending until
    // explicitly completed by the test's platform publication.
    image.request.submitted_ns = wire::Transport::now_ns();
    image.header.timeout_ns = UINT64_MAX;
    auto scheduler = std::make_unique<SchedulerState>();
    DeviceArena sm_arena;
    DeviceArena scheduler_arena;
    auto *sm = SharedMemoryHandle::create_and_init_default(sm_arena);
    ASSERT_NE(sm, nullptr);
    const auto layout = SchedulerState::reserve_layout(scheduler_arena);
    ASSERT_NE(scheduler_arena.commit(), nullptr);
    ASSERT_TRUE(scheduler->init_data_from_layout(layout, scheduler_arena, sm->header));
    scheduler->wire_arena_pointers(layout, scheduler_arena);
    ChipTaskSlotState producer{};
    TaskPayload payload{};
    producer.task_state.store(CHIP_TASK_PENDING);
    producer.active_mask = LaneMask::aiv0();
    producer.total_required_subtasks = 1;
    producer.logical_block_num = 1;
    producer.fanout_count = 1;
    producer.payload = &payload;
    auto &waits = scheduler->async_wait_list;
    waits.count = 1;
    auto &entry = waits.entries[0];
    entry.task_token = task.task_token;
    entry.slot_state = &producer;
    entry.normal_done = false;
    int32_t error = 0;
    ASSERT_TRUE(waits.append_condition_locked(
        entry, slab.entries[0].addr, slab.entries[0].backend_cookie, slab.entries[0].expected_value, ASYNC_ENGINE_CCU,
        COMPLETION_TYPE_CCU_REQUEST, error
    ));
    ChipTaskSlotState *releases[1]{};
    int32_t release_count = 0;
    auto poll = [&] {
        return waits.poll_and_complete<false>(nullptr, scheduler.get(), releases, release_count, 1);
    };
    EXPECT_EQ(poll().completed, 0);
    EXPECT_FALSE(entry.conditions[0].satisfied);
    FakeIo::finish();
    EXPECT_EQ(poll().completed, 0);  // Ordinary AIV still owns the task.
    EXPECT_TRUE(entry.conditions[0].retired);
    EXPECT_EQ(wire::Slot(image.request.slot), wire::Slot::FREE);
    EXPECT_EQ(release_count, 0);
    EXPECT_EQ(producer.task_state.load(), CHIP_TASK_PENDING);
    entry.normal_done = true;
    EXPECT_EQ(poll().completed, 1);
    EXPECT_EQ(producer.task_state.load(), CHIP_TASK_COMPLETED);
    EXPECT_EQ(release_count, 1);
    EXPECT_EQ(releases[0], &producer);
    EXPECT_EQ(poll().completed, 0);
    scheduler->destroy();
    scheduler_arena.release();
    sm_arena.release();
}

TEST_F(CcuCompletionTest, HiddenIdentityIsCopiedWithoutChangingPayloadSize) {
    static_assert(sizeof(AsyncCtxStorage) == 40);
    static_assert(sizeof(AsyncCtx) == 48);
    LocalContext local{};
    local.block_num = 1;
    local.async_ctx = task;
    local.async_ctx.urma_sender_id = 5;
    GlobalContext global{};
    global.callable_id = task.callable_id;
    int64_t args[50]{};
    args[PAYLOAD_LOCAL_CONTEXT_INDEX] = reinterpret_cast<int64_t>(&local);
    args[SPMD_GLOBAL_CONTEXT_INDEX] = reinterpret_cast<int64_t>(&global);
    EXPECT_EQ(get_async_ctx(args).callable_id, 7);
    EXPECT_EQ(get_async_ctx(args).urma_sender_id, 5u);
    local.block_num = 2;
    EXPECT_EQ(get_async_ctx(args).callable_id, -1);
}

TEST_F(CcuCompletionTest, PretriggerCloseCanCancelAndReuseReservedSlot) {
    wire::Reference ref;
    ASSERT_TRUE(Protocol::resolve(&image.header, sizeof(image), 7, 0, ref));
    ASSERT_TRUE(Protocol::reserve(ref, arguments().values()));
    image.header.admission = wire::Admission::CLOSED;
    EXPECT_FALSE(Protocol::submit(ref));
    EXPECT_TRUE(Protocol::cancel(ref));
    EXPECT_EQ(wire::Slot(image.request.slot), wire::Slot::FREE);
    EXPECT_EQ(FakeIo::triggers, 0u);
}

TEST_F(CcuCompletionTest, ScalarRepresentationPreservesWidthsAndSignedValues) {
    CcuCallArgs args;
    args.add_scalar(int32_t(-3));
    args.add_scalar(float(1.25));
    args.add_tensor(tensor());
    ASSERT_TRUE(send(CcuCall(&domain, 0, args)));
    const auto &stored = image.request.arguments;
    EXPECT_EQ(stored.tensor_count, 1u);
    EXPECT_EQ(stored.scalars[0].width, 32u);
    EXPECT_EQ(stored.scalars[0].kind, wire::ScalarKind::SIGNED);
    EXPECT_EQ(stored.scalars[0].bits, static_cast<uint64_t>(uint32_t(-3)));
    float value = 0;
    std::memcpy(&value, &stored.scalars[1].bits, sizeof(value));
    EXPECT_EQ(value, 1.25f);
}

TEST_F(CcuCompletionTest, ConfirmedFailureRetiresButNeverReleasesTask) {
    ASSERT_TRUE(send(CcuCall(&domain, 0, arguments())));
    FakeIo::finish(wire::Outcome::FAILED);
    auto scheduler = std::make_unique<SchedulerState>();
    auto &waits = scheduler->async_wait_list;
    waits.count = 1;
    auto &entry = waits.entries[0];
    entry.task_token = task.task_token;
    entry.normal_done = true;
    int32_t error = 0;
    ASSERT_TRUE(waits.append_condition_locked(
        entry, slab.entries[0].addr, slab.entries[0].backend_cookie, slab.entries[0].expected_value, ASYNC_ENGINE_CCU,
        COMPLETION_TYPE_CCU_REQUEST, error
    ));
    ChipTaskSlotState *releases[1]{};
    int32_t release_count = 0;
    auto result = waits.poll_and_complete<false>(nullptr, scheduler.get(), releases, release_count, 1);
    EXPECT_EQ(result.error_code, SIMPLER_ERROR_CCU_FAILED);
    EXPECT_EQ(result.failed_task_token, task.task_token);
    EXPECT_EQ(result.completed, 0);
    EXPECT_EQ(release_count, 0);
    EXPECT_EQ(wire::Slot(image.request.slot), wire::Slot::FREE);
    EXPECT_FALSE(entry.conditions[0].satisfied);
}

class CcuFailureDiagnosticTest : public ::testing::Test {
protected:
    DeviceArena sm_arena;
    DeviceArena scheduler_arena;
    SharedMemoryHandle handle{};
    std::unique_ptr<SchedulerState> scheduler = std::make_unique<SchedulerState>();
    ChipTaskSlotState *producer{};
    ChipTaskSlotState *consumer{};
    DepListEntry *edge{};

    void SetUp() override {
        constexpr uint64_t window = 8;
        const auto size = SharedMemoryHandle::calculate_size(window);
        const auto offset = sm_arena.reserve(size, CHIP_ALIGN_SIZE);
        ASSERT_NE(sm_arena.commit(), nullptr);
        ASSERT_TRUE(handle.init(sm_arena.region_ptr(offset), size, window, 1024));
        const auto layout = SchedulerState::reserve_layout(scheduler_arena, 8);
        ASSERT_NE(scheduler_arena.commit(), nullptr);
        ASSERT_TRUE(scheduler->init_data_from_layout(layout, scheduler_arena, handle.header));
        scheduler->wire_arena_pointers(layout, scheduler_arena);
        auto &ring = handle.header->rings[0];
        ring.fc.current_task_index.store(2);
        for (int32_t i = 0; i < 2; ++i) {
            auto &slot = ring.slot_states[i];
            slot.bind_ring(0);
            slot.bind_buffers(&ring.task_payloads[i], &ring.task_descriptors[i]);
            slot.reset_for_reuse();
            slot.task_state.store(CHIP_TASK_PENDING);
            slot.task->task_id = TaskId::make(0, i);
        }
        producer = &ring.slot_states[0];
        consumer = &ring.slot_states[1];
        consumer->payload->fanin_actual_count = 1;
        consumer->payload->fanin_inline_edges[0].set(producer, DEP_WAIT | DEP_RETAIN);
        edge = scheduler->ring_sched_states[0].dep_pool.alloc();
        ASSERT_NE(edge, nullptr);
        edge->slot_state = consumer;
        edge->next = nullptr;
        producer->fanout_head = edge;
        seed();
    }
    void TearDown() override {
        scheduler->destroy();
        scheduler_arena.release();
        sm_arena.release();
    }
    void seed() {
        handle.header->sched_error_code.store(SIMPLER_ERROR_CCU_FAILED);
        handle.header->ccu_dependency_failure = CcuDependencyFailureDiagnostic::recorded(TaskId::make(0, 0));
    }
    const CcuDependencyFailureDiagnostic &capture() {
        scheduler->capture_ccu_failure_successors();
        return handle.header->ccu_dependency_failure;
    }
};

TEST_F(CcuFailureDiagnosticTest, ReportsPublishedSuccessorWithoutDispatchOrCompletion) {
    const auto &diagnostic = capture();
    EXPECT_TRUE(diagnostic.scan_complete());
    EXPECT_TRUE(diagnostic.successors_unstarted());
    EXPECT_EQ(diagnostic.failed_token, 0U);
    EXPECT_EQ(diagnostic.first_successor_token, 1U);
    EXPECT_EQ(diagnostic.successor_count, 1U);
    EXPECT_EQ(diagnostic.first_successor_state, CHIP_TASK_PENDING);
    EXPECT_EQ(diagnostic.dispatched_blocks, 0U);
    EXPECT_EQ(producer->task_state.load(), CHIP_TASK_PENDING);
    EXPECT_EQ(consumer->task_state.load(), CHIP_TASK_PENDING);
}

TEST_F(CcuFailureDiagnosticTest, DispatchAndCompletionCannotClaimUnstarted) {
    consumer->next_block_idx.store(1);
    EXPECT_TRUE(capture().scan_complete());
    EXPECT_FALSE(capture().successors_unstarted());
    EXPECT_EQ(capture().dispatched_blocks, 1U);
    seed();
    consumer->next_block_idx.store(0);
    consumer->mark_completed();
    EXPECT_TRUE(capture().scan_complete());
    EXPECT_FALSE(capture().successors_unstarted());
}

TEST_F(CcuFailureDiagnosticTest, InvalidEdgesAndLostIdentitiesRemainIncomplete) {
    producer->fanout_head = reinterpret_cast<DepListEntry *>(uintptr_t{1});
    EXPECT_FALSE(capture().scan_complete());
    EXPECT_FALSE(capture().successors_unstarted());
    seed();
    producer->fanout_head = edge;
    edge->next = edge;
    EXPECT_FALSE(capture().scan_complete());
    EXPECT_FALSE(capture().successors_unstarted());
    seed();
    edge->next = nullptr;
    consumer->task->task_id = TaskId::make(0, 9);
    EXPECT_FALSE(capture().scan_complete());
    seed();
    consumer->task->task_id = TaskId::make(0, 1);
    consumer->payload->fanin_inline_edges[0].clear();
    EXPECT_FALSE(capture().scan_complete());
    seed();
    handle.header->ccu_dependency_failure = CcuDependencyFailureDiagnostic::recorded(TaskId::invalid());
    EXPECT_FALSE(capture().scan_complete());
    seed();
    handle.header->ccu_dependency_failure = CcuDependencyFailureDiagnostic::recorded(TaskId::make(3, 0));
    EXPECT_FALSE(capture().scan_complete());
}

TEST_F(CcuFailureDiagnosticTest, ReclaimedEdgesAndUnpublishedSlotsRemainIncomplete) {
    auto &pool = scheduler->ring_sched_states[0].dep_pool;
    const auto tail = pool.tail;
    pool.tail = pool.top;
    EXPECT_FALSE(capture().scan_complete());
    seed();
    pool.tail = tail;
    handle.header->rings[0].fc.current_task_index.store(1);
    EXPECT_FALSE(capture().scan_complete());
    seed();
    handle.header->rings[0].fc.current_task_index.store(2);
    edge->slot_state = reinterpret_cast<ChipTaskSlotState *>(uintptr_t{1});
    EXPECT_FALSE(capture().scan_complete());
}

TEST_F(CcuFailureDiagnosticTest, EmptyAndOtherErrorRecordsDoNotProveSuccessorSafety) {
    producer->fanout_head = nullptr;
    EXPECT_TRUE(capture().scan_complete());
    EXPECT_FALSE(capture().successors_unstarted());
    seed();
    handle.header->sched_error_code.store(SIMPLER_ERROR_CCU_UNCONFIRMED);
    EXPECT_FALSE(capture().scan_complete());
    seed();
    handle.header->ccu_accept_diagnostic = {};
    EXPECT_FALSE(capture().scan_complete());
}

}  // namespace
