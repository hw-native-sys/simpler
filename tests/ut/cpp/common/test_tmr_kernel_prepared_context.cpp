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
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>

#include "tensormap_and_ringbuffer/kernel_prepared_context.h"

namespace {
using namespace simpler::tmr;

// Real aligned storage models allocation ranges only. This fixture does not
// initialize a runnable runtime image or act as a production resource provider.
struct alignas(DeviceArena::kDefaultBaseAlign) ArenaStorage {
    std::array<uint8_t, sizeof(RuntimeContext)> image{};
    TmrLaunchControl control{};
    std::array<TmrCoreReport, 3> reports{};
    TmrKernelContextDescriptor descriptor{};
};

void expect_equal(const PreparedKernelContext &actual, const PreparedKernelContext &expected) {
    EXPECT_EQ(std::memcmp(&actual.descriptor, &expected.descriptor, sizeof(actual.descriptor)), 0);
    EXPECT_EQ(actual.binding.identity.device_binding_addr, expected.binding.identity.device_binding_addr);
    EXPECT_EQ(actual.binding.identity.context_generation, expected.binding.identity.context_generation);
    EXPECT_EQ(actual.binding.resident, expected.binding.resident);
    EXPECT_EQ(actual.binding.sm.base, expected.binding.sm.base);
    EXPECT_EQ(actual.binding.sm.capacity, expected.binding.sm.capacity);
    EXPECT_EQ(actual.binding.sm.required_bytes, expected.binding.sm.required_bytes);
    EXPECT_EQ(actual.binding.arena.base, expected.binding.arena.base);
    EXPECT_EQ(actual.binding.arena.capacity, expected.binding.arena.capacity);
    EXPECT_EQ(actual.binding.arena.required_bytes, expected.binding.arena.required_bytes);
    EXPECT_EQ(actual.binding.runtime_offset, expected.binding.runtime_offset);
    EXPECT_EQ(actual.handshake.control, expected.handshake.control);
    EXPECT_EQ(actual.handshake.reports, expected.handshake.reports);
    EXPECT_EQ(actual.handshake.worker_count, expected.handshake.worker_count);
    EXPECT_EQ(actual.handshake.epoch, expected.handshake.epoch);
    EXPECT_EQ(actual.allowed_cpus, expected.allowed_cpus);
    EXPECT_EQ(actual.residency_base, expected.residency_base);
    EXPECT_EQ(actual.residency_count, expected.residency_count);
    EXPECT_EQ(actual.stride, expected.stride);
    EXPECT_EQ(actual.register_table, expected.register_table);
    EXPECT_EQ(actual.arch_argument, expected.arch_argument);
    EXPECT_EQ(actual.ready_queue_shards, expected.ready_queue_shards);
    EXPECT_EQ(actual.serial_orch_sched, expected.serial_orch_sched);
}

class TmrKernelPreparedContextTest : public ::testing::Test {
protected:
    void SetUp() override {
        resident->dev.worker_count = 3;
        resident->dev.aicpu_thread_num = 2;
        resident->dev.aicpu_launch_count = 3;
        resident->dev.aicpu_allowed_cpu_count = 2;
        resident->dev.aicpu_allowed_cpus[0] = 7;
        resident->dev.aicpu_allowed_cpus[1] = 19;
        auto &d = arena.descriptor;
        d.version = kTmrKernelContextVersion;
        d.bytes = sizeof(d);
        d.context_generation = 17;
        d.self_address = reinterpret_cast<uint64_t>(&d);
        d.resident_runtime = reinterpret_cast<uint64_t>(resident.get());
        d.resident_kernel_args = reinterpret_cast<uint64_t>(&kernel_args);
        d.heap_base = reinterpret_cast<uint64_t>(heap.data());
        d.heap_capacity = d.heap_required = sizeof(heap);
        d.sm_base = reinterpret_cast<uint64_t>(&sm);
        d.sm_capacity = d.sm_required = sizeof(sm);
        d.arena_base = reinterpret_cast<uint64_t>(arena.image.data());
        d.arena_capacity = sizeof(arena);
        d.arena_required = sizeof(arena.image);
        d.control_address = reinterpret_cast<uint64_t>(&arena.control);
        d.control_bytes = sizeof(arena.control);
        d.reports_address = reinterpret_cast<uint64_t>(arena.reports.data());
        d.reports_bytes = sizeof(arena.reports);
        d.launch_threads = 3;
        d.execution_threads = 2;
        d.worker_count = 3;
        registration = {
            d.self_address, d.context_generation, reinterpret_cast<uint64_t>(residencies.data()),
            static_cast<uint32_t>(residencies.size()), sizeof(KernelCallableDeviceResidency)
        };
        ASSERT_TRUE(make_prepared_kernel_context(registration, d, *resident, &prepared));
    }

    void reject(const TmrContextRegistrationArgs &r, const TmrKernelContextDescriptor &d) {
        auto out = prepared;
        EXPECT_FALSE(make_prepared_kernel_context(r, d, *resident, &out));
        expect_equal(out, prepared);
    }

    std::unique_ptr<Runtime> resident{std::make_unique<Runtime>()};
    alignas(64) KernelArgs kernel_args{};
    alignas(DeviceArena::kDefaultBaseAlign) std::array<uint8_t, 1024> heap{};
    alignas(DeviceArena::kDefaultBaseAlign) SharedMemoryHeader sm{};
    ArenaStorage arena{};
    std::array<KernelCallableDeviceResidency, MAX_REGISTERED_CALLABLE_IDS> residencies{};
    TmrContextRegistrationArgs registration{};
    PreparedKernelContext prepared{};
};

TEST_F(TmrKernelPreparedContextTest, BorrowsRegionsAndCopiesImmutableMetadata) {
    EXPECT_EQ(prepared.binding.resident, resident.get());
    EXPECT_EQ(prepared.binding.sm.base, &sm);
    EXPECT_EQ(prepared.binding.arena.base, arena.image.data());
    EXPECT_EQ(prepared.handshake.control, &arena.control);
    EXPECT_EQ(prepared.handshake.reports, arena.reports.data());
    EXPECT_EQ(prepared.handshake.epoch, 0u);
    EXPECT_EQ(prepared.allowed_cpus[0], 7);
    EXPECT_EQ(prepared.allowed_cpus[1], 19);
    EXPECT_EQ(prepared.allowed_cpus[2], 0);
    resident->dev.aicpu_allowed_cpus[0] = 23;
    arena.descriptor.context_generation = 99;
    EXPECT_EQ(prepared.allowed_cpus[0], 7);
    EXPECT_EQ(prepared.descriptor.context_generation, 17u);
    EXPECT_FALSE(make_prepared_kernel_context(registration, arena.descriptor, *resident, nullptr));
}

TEST_F(TmrKernelPreparedContextTest, AcceptsIndependentSnapshotsAndSeparatelyAllocatedAuxiliaryRegions) {
    auto mirror = std::make_unique<Runtime>(*resident);
    TmrKernelContextDescriptor descriptor = arena.descriptor;
    TmrLaunchControl control{};
    std::array<TmrCoreReport, 3> reports{};
    descriptor.self_address = reinterpret_cast<uint64_t>(&descriptor);
    descriptor.control_address = reinterpret_cast<uint64_t>(&control);
    descriptor.reports_address = reinterpret_cast<uint64_t>(reports.data());
    auto r = registration;
    r.descriptor_address = descriptor.self_address;
    PreparedKernelContext out;
    ASSERT_TRUE(make_prepared_kernel_context(r, descriptor, *mirror, &out));
    EXPECT_EQ(out.binding.resident, resident.get());
    EXPECT_NE(out.binding.resident, mirror.get());
    EXPECT_EQ(out.handshake.control, &control);
    EXPECT_EQ(out.handshake.reports, reports.data());
}

TEST_F(TmrKernelPreparedContextTest, RejectsHeaderIdentityAndReservedFieldsTransactionally) {
    for (uint32_t version : {0u, 2u}) {
        auto d = arena.descriptor;
        d.version = version;
        reject(registration, d);
    }
    for (uint32_t bytes : {0u, static_cast<uint32_t>(sizeof(TmrKernelContextDescriptor) - 1)}) {
        auto d = arena.descriptor;
        d.bytes = bytes;
        reject(registration, d);
    }
    for (uint64_t generation : {uint64_t{0}, uint64_t{18}}) {
        auto d = arena.descriptor;
        d.context_generation = generation;
        reject(registration, d);
        auto r = registration;
        r.context_generation = generation;
        reject(r, arena.descriptor);
    }
    auto d = arena.descriptor;
    d.flags = 1;
    reject(registration, d);
    for (size_t i = 0; i < std::size(d.reserved); ++i) {
        d = arena.descriptor;
        d.reserved[i] = 1;
        reject(registration, d);
    }
    auto r = registration;
    r.descriptor_address += alignof(TmrKernelContextDescriptor);
    reject(r, arena.descriptor);
}

TEST_F(TmrKernelPreparedContextTest, RejectsTopologyAndNonemptyResidentInputs) {
    for (int32_t count : {-1, 0, 1, PLATFORM_MAX_AICPU_THREADS + 1}) {
        auto d = arena.descriptor;
        d.execution_threads = count;
        reject(registration, d);
    }
    for (int32_t count : {-1, 0, 1, MAX_GATE_THREADS + 1}) {
        auto d = arena.descriptor;
        d.launch_threads = count;
        reject(registration, d);
    }
    for (int32_t count : {-1, 0, 2, RUNTIME_MAX_WORKER + 1}) {
        auto d = arena.descriptor;
        d.worker_count = count;
        reject(registration, d);
    }
    auto &config = resident->dev;
    auto non_mixed = arena.descriptor;
    non_mixed.worker_count = config.worker_count = 2;
    non_mixed.reports_bytes = 2 * sizeof(TmrCoreReport);
    reject(registration, non_mixed);
    config.worker_count = 3;
    for (int *field :
         {&config.aicpu_thread_num, &config.aicpu_launch_count, &config.aicpu_allowed_cpu_count, &config.worker_count,
          &config.ready_queue_shards}) {
        const int value = *field;
        *field = 0;
        reject(registration, arena.descriptor);
        *field = value;
    }
    for (int32_t cpu : {-1, config.aicpu_allowed_cpus[0]}) {
        config.aicpu_allowed_cpus[1] = cpu;
        reject(registration, arena.descriptor);
    }
    config.aicpu_allowed_cpus[1] = 19;
    config.active_callable_id_ = 0;
    reject(registration, arena.descriptor);
    config.active_callable_id_ = -1;
    config.func_id_to_addr_[RUNTIME_MAX_FUNC_ID - 1] = 1;
    reject(registration, arena.descriptor);
    config.func_id_to_addr_[RUNTIME_MAX_FUNC_ID - 1] = 0;
    for (void **field : {&config.gm_sm_ptr_, &config.prebuilt_arena_base_}) {
        *field = &sm;
        reject(registration, arena.descriptor);
        *field = nullptr;
    }
    config.prebuilt_runtime_offset_ = 1;
    reject(registration, arena.descriptor);
    config.prebuilt_runtime_offset_ = 0;
    for (int32_t *field : {&config.orch_args_storage_.tensor_count_, &config.orch_args_storage_.scalar_count_}) {
        *field = 1;
        reject(registration, arena.descriptor);
        *field = 0;
    }
    config.aicpu_launch_count = MAX_GATE_THREADS;
    auto d = arena.descriptor;
    d.launch_threads = MAX_GATE_THREADS;
    EXPECT_TRUE(make_prepared_kernel_context(registration, d, *resident, &prepared));
}

TEST_F(TmrKernelPreparedContextTest, RejectsArenaBoundsAlignmentAndAliasing) {
    struct ArenaFields {
        uint64_t TmrKernelContextDescriptor::*base;
        uint64_t TmrKernelContextDescriptor::*capacity;
        uint64_t TmrKernelContextDescriptor::*required;
    };
    const ArenaFields fields[] = {
        {&TmrKernelContextDescriptor::heap_base, &TmrKernelContextDescriptor::heap_capacity,
         &TmrKernelContextDescriptor::heap_required},
        {&TmrKernelContextDescriptor::sm_base, &TmrKernelContextDescriptor::sm_capacity,
         &TmrKernelContextDescriptor::sm_required},
        {&TmrKernelContextDescriptor::arena_base, &TmrKernelContextDescriptor::arena_capacity,
         &TmrKernelContextDescriptor::arena_required},
    };
    for (const auto &field : fields) {
        for (uint64_t base :
             {uint64_t{0}, (arena.descriptor.*field.base) + 1, std::numeric_limits<uint64_t>::max() - 1023}) {
            auto d = arena.descriptor;
            d.*field.base = base;
            d.*field.capacity = std::numeric_limits<uint64_t>::max();
            reject(registration, d);
        }
        for (uint64_t required : {uint64_t{0}, (arena.descriptor.*field.capacity) + 1}) {
            auto d = arena.descriptor;
            d.*field.required = required;
            reject(registration, d);
        }
        auto d = arena.descriptor;
        d.*field.capacity = 0;
        reject(registration, d);
    }
    auto d = arena.descriptor;
    d.heap_base = d.sm_base;
    reject(registration, d);
    d = arena.descriptor;
    d.sm_base = d.arena_base;
    reject(registration, d);
    d = arena.descriptor;
    d.arena_base = d.heap_base;
    reject(registration, d);
    d = arena.descriptor;
    d.sm_required = sizeof(SharedMemoryHeader) - 1;
    reject(registration, d);
    for (uint64_t offset :
         {uint64_t{1}, d.arena_required, d.arena_required + 1, std::numeric_limits<uint64_t>::max()}) {
        d = arena.descriptor;
        d.runtime_offset = offset;
        reject(registration, d);
    }
}

TEST_F(TmrKernelPreparedContextTest, RejectsInvalidStaticAndClearSpansWithoutDereferencingAddresses) {
    for (auto field :
         {&TmrKernelContextDescriptor::self_address, &TmrKernelContextDescriptor::resident_runtime,
          &TmrKernelContextDescriptor::resident_kernel_args, &TmrKernelContextDescriptor::control_address,
          &TmrKernelContextDescriptor::reports_address}) {
        for (uint64_t address :
             {uint64_t{0}, (arena.descriptor.*field) + 1, std::numeric_limits<uint64_t>::max() - 63}) {
            auto d = arena.descriptor;
            d.*field = address;
            auto r = registration;
            r.descriptor_address = d.self_address;
            reject(r, d);
        }
    }
    for (auto field : {&TmrKernelContextDescriptor::control_bytes, &TmrKernelContextDescriptor::reports_bytes}) {
        for (uint64_t bytes : {uint64_t{0}, (arena.descriptor.*field) + 1}) {
            auto d = arena.descriptor;
            d.*field = bytes;
            reject(registration, d);
        }
    }
    for (uint64_t address :
         {arena.descriptor.heap_base, arena.descriptor.sm_base, arena.descriptor.arena_base,
          arena.descriptor.self_address, arena.descriptor.resident_runtime, arena.descriptor.resident_kernel_args}) {
        auto d = arena.descriptor;
        d.control_address = address;
        reject(registration, d);
    }
    auto d = arena.descriptor;
    d.reports_address = d.control_address + 64;
    reject(registration, d);
    d = arena.descriptor;
    d.control_address = d.arena_base + d.arena_capacity - 64;
    reject(registration, d);
    auto r = registration;
    r.callable_descriptor_base = arena.descriptor.control_address;
    reject(r, arena.descriptor);
    d = arena.descriptor;
    d.resident_kernel_args = d.resident_runtime;
    reject(registration, d);
}

TEST_F(TmrKernelPreparedContextTest, ResidencyMembershipRequiresExactBoundedSlotStarts) {
    EXPECT_FALSE(PreparedKernelContext{}.contains_residency(0));
    const uint64_t base = registration.callable_descriptor_base;
    const uint64_t stride = registration.callable_descriptor_stride;
    for (uint32_t i = 0; i < registration.callable_descriptor_count; ++i)
        EXPECT_TRUE(prepared.contains_residency(base + i * stride));
    EXPECT_FALSE(prepared.contains_residency(base - stride));
    EXPECT_FALSE(prepared.contains_residency(base + 8));
    EXPECT_FALSE(prepared.contains_residency(base + registration.callable_descriptor_count * stride));
    EXPECT_FALSE(prepared.contains_residency(std::numeric_limits<uint64_t>::max()));
    auto r = registration;
    r.callable_descriptor_count /= 2;
    r.callable_descriptor_stride *= 2;
    ASSERT_TRUE(make_prepared_kernel_context(r, arena.descriptor, *resident, &prepared));
    EXPECT_TRUE(prepared.contains_residency(base + 2 * stride));
    EXPECT_FALSE(prepared.contains_residency(base + stride));
    for (uint32_t count : {0u, static_cast<uint32_t>(MAX_REGISTERED_CALLABLE_IDS + 1)}) {
        r = registration;
        r.callable_descriptor_count = count;
        reject(r, arena.descriptor);
    }
    for (uint32_t bad_stride :
         {0u, static_cast<uint32_t>(sizeof(KernelCallableDeviceResidency) - 1),
          static_cast<uint32_t>(sizeof(KernelCallableDeviceResidency) + 1)}) {
        r = registration;
        r.callable_descriptor_stride = bad_stride;
        reject(r, arena.descriptor);
    }
    for (uint64_t address : {uint64_t{0}, base + 1, std::numeric_limits<uint64_t>::max() - 7}) {
        r = registration;
        r.callable_descriptor_base = address;
        reject(r, arena.descriptor);
    }
}

}  // namespace
