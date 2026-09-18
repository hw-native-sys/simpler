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
#include <cstring>

#include "tensormap_and_ringbuffer/kernel_registration.h"

namespace {
using namespace simpler::tmr;

// Only executor metadata and cache operations are modeled. The production
// revoke template and prepared-context/callable representations are unchanged.
struct ExecutorModel {
    struct Gate {
        bool quiescent{true};
        bool idle() const { return quiescent; }
    } kernel_gate_;
    struct Invocation {
        bool borrowed{false};
        bool active() const { return borrowed; }
    } kernel_invocation_;
    struct Slot {
        PreparedKernelCallable kernel;
        bool in_use{false};
        bool kernel_owned{false};
        uint64_t handle{0};
    };
    bool kernel_control_attached_{false};
    bool kernel_context_ready_{false};
    PreparedKernelContext kernel_context_{};
    std::array<Slot, 3> orch_so_table_{};
};

struct CacheObservation {
    ExecutorModel *executor{nullptr};
    const void *address{nullptr};
    int invalidations{0};
    int flushes{0};
    bool registered_at_invalidate{false};
    bool registered_at_flush{false};
    bool slot_borrowed_at_flush{false};
    TmrContextRevokeReceipt published{};
} observation;
}  // namespace

uint32_t platform_get_physical_cores_count() { return 75; }

namespace aicpu_cache_maintenance {
void invalidate_range_impl(const void *address, size_t bytes) {
    EXPECT_EQ(address, observation.address);
    EXPECT_EQ(bytes, sizeof(simpler::tmr::TmrContextRevokeReceipt));
    ++observation.invalidations;
    observation.registered_at_invalidate = observation.executor->kernel_context_ready_;
}
void flush_range_impl(const void *address, size_t bytes) {
    EXPECT_EQ(address, observation.address);
    EXPECT_EQ(bytes, sizeof(simpler::tmr::TmrContextRevokeReceipt));
    ++observation.flushes;
    observation.registered_at_flush = observation.executor->kernel_context_ready_;
    for (const auto &slot : observation.executor->orch_so_table_)
        observation.slot_borrowed_at_flush |= slot.kernel.device_address != 0 || !slot.kernel.functions.empty();
    std::memcpy(&observation.published, address, sizeof(observation.published));
}
}  // namespace aicpu_cache_maintenance

namespace {
class TmrKernelRevokeTest : public ::testing::Test {
protected:
    void SetUp() override {
        args = {0x100000, 17, reinterpret_cast<uint64_t>(&receipt), sizeof(receipt)};
        receipt = {args.descriptor_address, args.context_generation, 0, 0, {}};
        observation = {};
        observation.executor = &executor;
        observation.address = &receipt;
        executor.orch_so_table_[0].in_use = true;
        executor.orch_so_table_[0].handle = 31;
        // A cached SO without a device image/table is not a device borrower.
        executor.orch_so_table_[2].in_use = true;
        executor.orch_so_table_[2].kernel_owned = true;
        executor.orch_so_table_[2].handle = 47;
    }

    void register_context() {
        executor.kernel_context_ready_ = true;
        auto &context = executor.kernel_context_;
        auto &d = context.descriptor;
        d.self_address = args.descriptor_address;
        d.context_generation = args.context_generation;
        // These integer ranges are never dereferenced by revoke.
        d.resident_runtime = 0x200000;
        d.resident_kernel_args = 0x300000;
        d.heap_base = 0x400000;
        d.heap_capacity = 1024;
        d.sm_base = 0x500000;
        d.sm_capacity = 1024;
        d.arena_base = 0x600000;
        d.arena_capacity = 2048;
        d.arena_required = 1024;
        d.control_address = 0x700000;
        d.control_bytes = sizeof(TmrLaunchControl);
        d.reports_address = 0x800000;
        d.reports_bytes = 3 * sizeof(TmrCoreReport);
        context.register_table = 0x900000;
        auto &slot = executor.orch_so_table_[1];
        slot.in_use = true;
        slot.kernel_owned = true;
        slot.handle = 41;
        slot.kernel.device_address = 0xa00000;
        slot.kernel.bytes = 512;
        slot.kernel.identity = {7, 1, 1};
        slot.kernel.functions = {0, 0xa00080};
    }

    void expect_cached_handles() {
        EXPECT_TRUE(executor.orch_so_table_[0].in_use);
        EXPECT_FALSE(executor.orch_so_table_[0].kernel_owned);
        EXPECT_EQ(executor.orch_so_table_[0].handle, 31u);
        EXPECT_TRUE(executor.orch_so_table_[2].in_use);
        EXPECT_TRUE(executor.orch_so_table_[2].kernel_owned);
        EXPECT_EQ(executor.orch_so_table_[2].handle, 47u);
    }

    void expect_rejected(const TmrContextRevokeArgs &invalid, bool receipt_read = false) {
        const auto previous_receipt = receipt;
        const auto previous_context = executor.kernel_context_;
        const bool previous_ready = executor.kernel_context_ready_;
        const auto previous_callable = executor.orch_so_table_[1].kernel;
        const auto *const previous_storage = executor.orch_so_table_[1].kernel.functions.data();
        const int previous_invalidations = observation.invalidations;
        const int previous_flushes = observation.flushes;
        EXPECT_NE(revoke_kernel_context(executor, &invalid), 0);
        EXPECT_EQ(std::memcmp(&receipt, &previous_receipt, sizeof(receipt)), 0);
        EXPECT_EQ(executor.kernel_context_ready_, previous_ready);
        EXPECT_EQ(
            std::memcmp(
                &executor.kernel_context_.descriptor, &previous_context.descriptor, sizeof(previous_context.descriptor)
            ),
            0
        );
        const auto &kernel = executor.orch_so_table_[1].kernel;
        EXPECT_EQ(kernel.device_address, previous_callable.device_address);
        EXPECT_EQ(kernel.bytes, previous_callable.bytes);
        EXPECT_EQ(kernel.identity.callable_id, previous_callable.identity.callable_id);
        EXPECT_EQ(kernel.functions, previous_callable.functions);
        EXPECT_EQ(kernel.functions.data(), previous_storage);
        EXPECT_EQ(observation.invalidations, previous_invalidations + (receipt_read ? 1 : 0));
        EXPECT_EQ(observation.flushes, previous_flushes);
        expect_cached_handles();
    }

    ExecutorModel executor;
    TmrContextRevokeReceipt receipt{};
    TmrContextRevokeArgs args{};
};

TEST_F(TmrKernelRevokeTest, PublishesReceiptOnlyAfterMetadataRevocationAndPreservesCachedSos) {
    register_context();
    EXPECT_FALSE(valid_tmr_context_revoke_receipt(receipt, args, TmrRevokeCompletion::Complete));
    ASSERT_EQ(revoke_kernel_context(executor, &args), 0);
    EXPECT_EQ(observation.invalidations, 1);
    EXPECT_EQ(observation.flushes, 1);
    EXPECT_TRUE(observation.registered_at_invalidate);
    EXPECT_FALSE(observation.registered_at_flush);
    EXPECT_FALSE(observation.slot_borrowed_at_flush);
    EXPECT_TRUE(valid_tmr_context_revoke_receipt(observation.published, args, TmrRevokeCompletion::Complete));
    EXPECT_EQ(std::memcmp(&observation.published, &receipt, sizeof(receipt)), 0);
    EXPECT_FALSE(executor.kernel_context_ready_);
    EXPECT_EQ(executor.kernel_context_.descriptor.self_address, 0u);
    EXPECT_EQ(executor.kernel_context_.register_table, 0u);
    EXPECT_EQ(executor.orch_so_table_[1].kernel.device_address, 0u);
    EXPECT_TRUE(executor.orch_so_table_[1].kernel.functions.empty());
    EXPECT_EQ(executor.orch_so_table_[1].handle, 41u);
    EXPECT_TRUE(executor.orch_so_table_[1].in_use);
    EXPECT_TRUE(executor.orch_so_table_[1].kernel_owned);
    expect_cached_handles();
    // A duplicate task cannot overwrite an already published one-shot receipt.
    expect_rejected(args, true);
}

TEST_F(TmrKernelRevokeTest, AbsentRegistrationCanConfirmEmptyRevocationWithoutReadingDescriptor) {
    ASSERT_EQ(revoke_kernel_context(executor, &args), 0);
    EXPECT_EQ(observation.invalidations, 1);
    EXPECT_EQ(observation.flushes, 1);
    EXPECT_FALSE(observation.registered_at_invalidate);
    EXPECT_FALSE(observation.registered_at_flush);
    EXPECT_TRUE(valid_tmr_context_revoke_receipt(receipt, args, TmrRevokeCompletion::Complete));
    expect_cached_handles();
}

TEST_F(TmrKernelRevokeTest, DifferentContextDoesNotAccessReceiptOrReplaceExistingState) {
    register_context();
    for (bool different_address : {false, true}) {
        auto other = args;
        if (different_address) other.descriptor_address += sizeof(TmrKernelContextDescriptor);
        else ++other.context_generation;
        expect_rejected(other);
    }
}

TEST_F(TmrKernelRevokeTest, ActiveOrRetiringConsumersLeaveReceiptAndMetadataUntouched) {
    register_context();
    executor.kernel_gate_.quiescent = false;
    expect_rejected(args);
    executor.kernel_gate_.quiescent = true;
    executor.kernel_invocation_.borrowed = true;
    expect_rejected(args);
    executor.kernel_invocation_.borrowed = false;
    executor.kernel_control_attached_ = true;
    expect_rejected(args);
}

TEST_F(TmrKernelRevokeTest, AbsentRegistrationCannotHideOrphanedContextOrCallableBorrowers) {
    auto &context = executor.kernel_context_;
    context.descriptor.context_generation = 13;
    expect_rejected(args);
    context = {};
    context.binding.resident = reinterpret_cast<Runtime *>(0x200000);
    expect_rejected(args);
    context = {};
    context.handshake.control = reinterpret_cast<TmrLaunchControl *>(0x700000);
    expect_rejected(args);
    context = {};
    context.register_table = 0x900000;
    expect_rejected(args);
    context = {};
    auto &kernel = executor.orch_so_table_[1].kernel;
    kernel.device_address = 0xa00000;
    expect_rejected(args);
    kernel = {};
    kernel.bytes = 512;
    expect_rejected(args);
    kernel = {};
    kernel.functions = {0xa00080};
    expect_rejected(args);
}

TEST_F(TmrKernelRevokeTest, RejectsMalformedArgsBeforeAnyReceiptAccess) {
    register_context();
    EXPECT_NE(revoke_kernel_context(executor, nullptr), 0);
    for (uint64_t address : {uint64_t{0}, args.receipt_address + 1, UINT64_MAX - 63}) {
        auto invalid = args;
        invalid.receipt_address = address;
        expect_rejected(invalid);
    }
    for (uint64_t bytes : {uint64_t{0}, uint64_t{63}, uint64_t{65}, UINT64_MAX}) {
        auto invalid = args;
        invalid.receipt_bytes = bytes;
        expect_rejected(invalid);
    }
    for (uint64_t address : {uint64_t{0}, args.descriptor_address + 1, UINT64_MAX - 63}) {
        auto invalid = args;
        invalid.descriptor_address = address;
        expect_rejected(invalid);
    }
    auto invalid = args;
    invalid.context_generation = 0;
    expect_rejected(invalid);
}

TEST_F(TmrKernelRevokeTest, RejectsReceiptAliasingRegisteredRegionsBeforeCacheMaintenance) {
    register_context();
    const auto &d = executor.kernel_context_.descriptor;
    const std::array<uint64_t, 10> addresses{
        d.self_address,
        d.resident_runtime,
        d.resident_kernel_args,
        d.heap_base,
        d.sm_base,
        d.arena_base,
        d.control_address,
        d.reports_address,
        executor.kernel_context_.register_table,
        executor.orch_so_table_[1].kernel.device_address
    };
    for (uint64_t address : addresses) {
        auto invalid = args;
        invalid.receipt_address = address;
        expect_rejected(invalid);
    }
    executor.kernel_context_.descriptor.arena_capacity += 32;
    auto invalid = args;
    invalid.receipt_address = d.arena_base + d.arena_capacity - 32;
    expect_rejected(invalid);
}

TEST_F(TmrKernelRevokeTest, ReceiptCannotAliasEvenUnusedArenaCapacity) {
    register_context();
    const auto &d = executor.kernel_context_.descriptor;
    auto invalid = args;
    invalid.receipt_address = d.arena_base + d.arena_required;
    ASSERT_LE(invalid.receipt_address + sizeof(receipt), d.arena_base + d.arena_capacity);
    expect_rejected(invalid);
}

TEST_F(TmrKernelRevokeTest, MismatchedOrNonpendingReceiptCannotRevokeMetadata) {
    register_context();
    const auto pending = receipt;
    for (int fault = 0; fault < 5; ++fault) {
        receipt = pending;
        if (fault == 0) ++receipt.descriptor_address;
        if (fault == 1) ++receipt.context_generation;
        if (fault == 2) receipt.status = -47;
        if (fault == 3) receipt.complete = static_cast<uint32_t>(TmrRevokeCompletion::Complete);
        if (fault == 4) receipt.reserved[4] = 1;
        expect_rejected(args, true);
    }
}

TEST_F(TmrKernelRevokeTest, HostReceiptValidationRequiresExactIdentityStatusAndCompletion) {
    const auto pending = receipt;
    for (int fault = 0; fault < 6; ++fault) {
        receipt = pending;
        receipt.complete = static_cast<uint32_t>(TmrRevokeCompletion::Complete);
        if (fault == 1) ++receipt.descriptor_address;
        if (fault == 2) ++receipt.context_generation;
        if (fault == 3) receipt.status = -47;
        if (fault == 4) receipt.complete = 0;
        if (fault == 5) receipt.reserved[0] = 1;
        EXPECT_EQ(valid_tmr_context_revoke_receipt(receipt, args, TmrRevokeCompletion::Complete), fault == 0);
    }
}
}  // namespace
