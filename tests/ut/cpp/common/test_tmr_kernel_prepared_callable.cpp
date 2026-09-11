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
#include <vector>

#include "tensormap_and_ringbuffer/kernel_prepared_callable.h"

namespace {
using namespace simpler::tmr;

// Owned Host bytes exercise registration parsing and borrowed table addresses.
// No HBM allocation, executable device image or production provider is created.
std::vector<uint8_t> make_image(int32_t cached_scalars = 0) {
    const uint8_t binary[] = {1, 2, 3, 4};
    const std::array<ArgDirection, 4> signature{
        ArgDirection::IN, ArgDirection::OUT, ArgDirection::SCALAR, ArgDirection::SCALAR
    };
    const std::array<int32_t, 2> function_ids{2, 7};
    const std::array<std::vector<uint8_t>, 2> children{
        make_callable<CORE_MAX_TENSOR_ARGS>(nullptr, 0, binary, sizeof(binary)),
        make_callable<CORE_MAX_TENSOR_ARGS>(nullptr, 0, binary, sizeof(binary))
    };
    auto image = make_callable<CoreCallable, CHIP_MAX_TENSOR_ARGS, 1024>(
        signature.data(), signature.size(), cached_scalars, "orch", binary, sizeof(binary), function_ids.data(),
        children.data(), children.size(), "config"
    );
    auto *chip = reinterpret_cast<ChipCallable *>(image.data());
    for (int32_t i = 0; i < chip->child_count(); ++i) {
        auto *child = reinterpret_cast<CoreCallable *>(chip->storage_ + chip->child_offset(i));
        child->set_resolved_addr(reinterpret_cast<uint64_t>(child->binary_data()));
    }
    return image;
}

void expect_equal(const PreparedKernelCallable &actual, const PreparedKernelCallable &expected) {
    EXPECT_EQ(actual.residency_address, expected.residency_address);
    EXPECT_EQ(std::memcmp(&actual.residency, &expected.residency, sizeof(actual.residency)), 0);
    EXPECT_EQ(actual.identity.callable_id, expected.identity.callable_id);
    EXPECT_EQ(actual.identity.tensor_count, expected.identity.tensor_count);
    EXPECT_EQ(actual.identity.scalar_count, expected.identity.scalar_count);
    EXPECT_EQ(actual.identity.slot_generation, expected.identity.slot_generation);
    EXPECT_EQ(actual.functions, expected.functions);
}

class TmrKernelPreparedCallableTest : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_EQ(reinterpret_cast<uintptr_t>(image.data()) % alignof(ChipCallable), 0u);
        resident = {13, reinterpret_cast<uint64_t>(image.data()), image.size(), 3, 0};
        ASSERT_TRUE(make_prepared_kernel_callable(descriptor_address(), resident, &prepared));
    }

    uint64_t descriptor_address() const { return reinterpret_cast<uint64_t>(&resident); }
    ChipCallable &chip() { return *reinterpret_cast<ChipCallable *>(image.data()); }
    CoreCallable &child(int32_t index) {
        return *reinterpret_cast<CoreCallable *>(chip().storage_ + chip().child_offset(index));
    }

    void reject(const KernelCallableDeviceResidency &candidate, uint64_t address) {
        auto out = prepared;
        const auto *storage = out.functions.data();
        const size_t capacity = out.functions.capacity();
        EXPECT_FALSE(make_prepared_kernel_callable(address, candidate, &out));
        expect_equal(out, prepared);
        EXPECT_EQ(out.functions.data(), storage);
        EXPECT_EQ(out.functions.capacity(), capacity);
    }
    void reject() { reject(resident, descriptor_address()); }

    std::vector<uint8_t> image{make_image()};
    KernelCallableDeviceResidency resident{};
    PreparedKernelCallable prepared{};
};

TEST_F(TmrKernelPreparedCallableTest, DerivesLegacyZeroScalarCountAndMatchesRecordedCount) {
    EXPECT_EQ(chip().scalar_count(), 0);
    EXPECT_EQ(prepared.identity.callable_id, 3);
    EXPECT_EQ(prepared.identity.slot_generation, 13u);
    EXPECT_EQ(prepared.identity.tensor_count, 2);
    EXPECT_EQ(prepared.identity.scalar_count, 2);
    const auto original = image;
    PreparedKernelCallable legacy;
    ASSERT_TRUE(make_prepared_kernel_callable(descriptor_address(), resident, &legacy));
    EXPECT_EQ(image, original);
    chip().scalar_count_ = 2;
    PreparedKernelCallable recorded;
    ASSERT_TRUE(make_prepared_kernel_callable(descriptor_address(), resident, &recorded));
    EXPECT_EQ(recorded.identity.tensor_count, legacy.identity.tensor_count);
    EXPECT_EQ(recorded.identity.scalar_count, legacy.identity.scalar_count);
    EXPECT_EQ(recorded.functions, legacy.functions);

    chip().sig_count_ = 2;
    chip().scalar_count_ = 0;
    ASSERT_TRUE(make_prepared_kernel_callable(descriptor_address(), resident, &recorded));
    EXPECT_EQ(recorded.identity.tensor_count, 2);
    EXPECT_EQ(recorded.identity.scalar_count, 0);
}

TEST_F(TmrKernelPreparedCallableTest, SparseTablePointsToCoreCallableDescriptorsNotInstructions) {
    ASSERT_EQ(prepared.functions.size(), 8u);
    const auto view = prepared.view();
    EXPECT_EQ(view.functions.entries, prepared.functions.data());
    EXPECT_EQ(view.functions.count, prepared.functions.size());
    EXPECT_EQ(view.identity.callable_id, resident.callable_id);
    EXPECT_EQ(view.identity.slot_generation, resident.generation);
    for (int32_t id = 0; id < 8; ++id) {
        const CoreCallable *expected = id == 2 ? &child(0) : id == 7 ? &child(1) : nullptr;
        EXPECT_EQ(view.functions.lookup(id), reinterpret_cast<uint64_t>(expected));
        if (expected != nullptr) EXPECT_NE(view.functions.lookup(id), expected->resolved_addr());
    }
    EXPECT_EQ(view.functions.lookup(-1), 0u);
    EXPECT_EQ(view.functions.lookup(8), 0u);
    chip().child_func_ids_[1] = RUNTIME_MAX_FUNC_ID - 1;
    ASSERT_TRUE(make_prepared_kernel_callable(descriptor_address(), resident, &prepared));
    EXPECT_EQ(prepared.functions.size(), static_cast<size_t>(RUNTIME_MAX_FUNC_ID));
    EXPECT_EQ(prepared.view().functions.lookup(RUNTIME_MAX_FUNC_ID - 1), reinterpret_cast<uint64_t>(&child(1)));
}

TEST_F(TmrKernelPreparedCallableTest, IndependentCallablesWithSameFunctionIdsDoNotShareTables) {
    auto image_b = make_image(2);
    auto *chip_b = reinterpret_cast<ChipCallable *>(image_b.data());
    KernelCallableDeviceResidency resident_b{27, reinterpret_cast<uint64_t>(image_b.data()), image_b.size(), 4, 0};
    PreparedKernelCallable prepared_b;
    ASSERT_TRUE(make_prepared_kernel_callable(reinterpret_cast<uint64_t>(&resident_b), resident_b, &prepared_b));
    EXPECT_NE(prepared.functions.data(), prepared_b.functions.data());
    for (int32_t i = 0; i < chip().child_count(); ++i) {
        const int32_t id = chip().child_func_id(i);
        EXPECT_EQ(prepared.view().functions.lookup(id), reinterpret_cast<uint64_t>(&chip().child(i)));
        EXPECT_EQ(prepared_b.view().functions.lookup(id), reinterpret_cast<uint64_t>(&chip_b->child(i)));
        EXPECT_NE(prepared.view().functions.lookup(id), prepared_b.view().functions.lookup(id));
    }
    const auto before = prepared;
    resident_b.generation = 28;
    ASSERT_TRUE(make_prepared_kernel_callable(reinterpret_cast<uint64_t>(&resident_b), resident_b, &prepared_b));
    EXPECT_EQ(prepared_b.identity.slot_generation, 28u);
    expect_equal(prepared, before);
}

TEST_F(TmrKernelPreparedCallableTest, RejectsIdentityBoundsWithoutChangingExistingOutput) {
    EXPECT_FALSE(make_prepared_kernel_callable(descriptor_address(), resident, nullptr));
    reject(resident, 0);
    for (int32_t id : {-1, MAX_REGISTERED_CALLABLE_IDS}) {
        auto invalid = resident;
        invalid.callable_id = id;
        reject(invalid, descriptor_address());
    }
    auto invalid = resident;
    invalid.generation = 0;
    reject(invalid, descriptor_address());
    invalid = resident;
    invalid.reserved = 1;
    reject(invalid, descriptor_address());
    invalid = resident;
    invalid.device_address = 0;
    reject(invalid, descriptor_address());
    invalid = resident;
    ++invalid.device_address;
    reject(invalid, descriptor_address());
    invalid = resident;
    invalid.device_address = std::numeric_limits<uintptr_t>::max() - 7;
    reject(invalid, descriptor_address());
    for (uint64_t bytes : {uint64_t{0}, uint64_t{sizeof(ChipCallable) - 1}, resident.bytes - 1, resident.bytes + 1}) {
        invalid = resident;
        invalid.bytes = bytes;
        reject(invalid, descriptor_address());
    }
}

TEST_F(TmrKernelPreparedCallableTest, RejectsInvalidDuplicateAndUnresolvedChildMappingsTransactionally) {
    for (int32_t id : {-1, RUNTIME_MAX_FUNC_ID, std::numeric_limits<int32_t>::max()}) {
        chip().child_func_ids_[1] = id;
        reject();
    }
    chip().child_func_ids_[1] = chip().child_func_ids_[0];
    reject();
    chip().child_func_ids_[1] = 7;
    child(1).set_resolved_addr(0);
    reject();
}

TEST_F(TmrKernelPreparedCallableTest, RejectsMalformedSignaturesAndImageLayoutBeforeChildAccess) {
    const auto original = image;
    auto reset = [&] {
        std::memcpy(image.data(), original.data(), image.size());
    };
    for (int32_t count : {-1, CHIP_MAX_TENSOR_ARGS + 1}) {
        chip().sig_count_ = count;
        reject();
        reset();
    }
    for (int32_t scalars : {-1, 1, CHIP_MAX_SCALAR_ARGS + 1}) {
        chip().scalar_count_ = scalars;
        reject();
        reset();
    }
    chip().signature_[0] = static_cast<ArgDirection>(99);
    reject();
    reset();
    chip().signature_[0] = ArgDirection::SCALAR;
    reject();
    reset();
    for (int32_t count : {-1, 1025}) {
        chip().child_count_ = count;
        reject();
        reset();
    }
    for (uint32_t offset :
         {chip().child_offsets_[0], chip().child_offsets_[1] + 1, std::numeric_limits<uint32_t>::max()}) {
        chip().child_offsets_[1] = offset;
        reject();
        reset();
    }
    chip().binary_size_ = std::numeric_limits<uint32_t>::max();
    reject();
    reset();
    child(1).binary_size_ = std::numeric_limits<uint32_t>::max();
    reject();
    reset();
    child(1).sig_count_ = CORE_MAX_TENSOR_ARGS + 1;
    reject();
    reset();
    chip().func_name_len_ = CALLABLE_FUNC_NAME_MAX;
    reject();
    reset();
    chip().config_name_[chip().config_name_len_] = 'x';
    reject();
}

TEST_F(TmrKernelPreparedCallableTest, EmptyChildTableHasNoPhantomFunctionEntry) {
    auto empty = make_callable<CoreCallable, CHIP_MAX_TENSOR_ARGS, 1024>(
        nullptr, 0, 0, "orch", nullptr, 0, nullptr, nullptr, 0, "config"
    );
    KernelCallableDeviceResidency no_children{1, reinterpret_cast<uint64_t>(empty.data()), empty.size(), 0, 0};
    PreparedKernelCallable out;
    ASSERT_TRUE(make_prepared_kernel_callable(reinterpret_cast<uint64_t>(&no_children), no_children, &out));
    EXPECT_EQ(out.identity.tensor_count, 0);
    EXPECT_EQ(out.identity.scalar_count, 0);
    EXPECT_TRUE(out.functions.empty());
    EXPECT_EQ(out.view().functions.lookup(0), 0u);
}
}  // namespace
