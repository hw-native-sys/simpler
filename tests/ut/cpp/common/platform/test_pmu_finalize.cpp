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

#include <cstdlib>
#include <map>
#include <utility>
#include <vector>

#include "aicpu/platform_regs.h"
#include "aicpu/pmu_collector_aicpu.h"
#include "host/pmu_collector.h"

namespace {
using Register = std::pair<uint64_t, RegId>;
struct RegisterWrite {
    Register target;
    uint64_t value;
};
uint64_t register_table = 0;
std::map<Register, uint32_t> registers;
std::vector<RegisterWrite> writes;
size_t read_count = 0;

uint32_t saved_value(RegId reg) { return 0x4ac00000U + static_cast<uint32_t>(reg); }
}  // namespace

extern "C" {
void set_platform_pmu_reg_addrs(uint64_t value) { register_table = value; }
uint64_t get_platform_pmu_reg_addrs() { return register_table; }
uint64_t get_platform_regs() { return register_table; }
}

uint64_t read_reg(uint64_t, RegId reg) {
    ++read_count;
    return saved_value(reg);
}

volatile uint32_t *get_reg_ptr(uint64_t base, RegId reg) { return &registers[{base, reg}]; }

uint32_t reg_load_acquire(const volatile uint32_t *pointer) {
    ++read_count;
    return *pointer;
}

void reg_store_release(volatile uint32_t *pointer, uint32_t value) {
    for (auto &entry : registers) {
        if (&entry.second == pointer) {
            writes.push_back({entry.first, value});
            entry.second = value;
            return;
        }
    }
#if !defined(SIMPLER_TEST_PMU_ARCH_A5)
    for (uint64_t base : {0x10000000ULL, 0x20000000ULL}) {
        for (int index = static_cast<int>(RegId::PMU_CTRL_0); index <= static_cast<int>(RegId::PMU_CNT7_IDX); ++index) {
            RegId reg = static_cast<RegId>(index);
            if (reinterpret_cast<uint64_t>(pointer) == base + reg_offset(reg)) {
                writes.push_back({{base, reg}, value});
                return;
            }
        }
    }
#endif
    ADD_FAILURE() << "write outside the modeled PMU register window";
}

#if defined(SIMPLER_TEST_PMU_ARCH_A5)
void write_reg(uint64_t base, RegId reg, uint64_t value) { writes.push_back({{base, reg}, value}); }
#endif

namespace {
class PmuFinalizeTest : public ::testing::Test {
protected:
    uint64_t bases_[2] = {0x10000000, 0x20000000};
    uint32_t physical_[2] = {1, 0};
    PmuCollector collector_;

    static void *allocate(size_t size) { return std::calloc(1, size); }
    static int release(void *pointer) {
        std::free(pointer);
        return 0;
    }

    void SetUp() override {
        collector_.begin_run("/tmp/pmu_finalize_test.csv", PmuEventType::PIPE_UTILIZATION);
        ASSERT_EQ(collector_.init(2, 1, allocate, nullptr, release, 0), 0);
        set_pmu_enabled(true);
        set_platform_pmu_base(reinterpret_cast<uint64_t>(collector_.get_pmu_shm_device_ptr()));
        set_platform_pmu_reg_addrs(reinterpret_cast<uint64_t>(bases_));
        pmu_aicpu_init(physical_, 2);
        clear_trace();
    }

    void TearDown() override {
        set_pmu_enabled(false);
        set_platform_pmu_base(0);
        set_platform_pmu_reg_addrs(0);
        collector_.finalize(nullptr, release);
        registers.clear();
    }

    static void clear_trace() {
        writes.clear();
        read_count = 0;
    }

    static constexpr size_t writes_per_core() {
#if defined(SIMPLER_TEST_PMU_ARCH_A5)
        return 2;
#else
        return 1;
#endif
    }
};

TEST_F(PmuFinalizeTest, RestoresOnlyOwnedCoreControlsWithoutPolling) {
    const int owned[] = {0};
    pmu_aicpu_finalize(owned, 1);
    EXPECT_EQ(read_count, 0U);
    ASSERT_EQ(writes.size(), writes_per_core());
    for (const auto &write : writes) {
        EXPECT_EQ(write.target.first, bases_[physical_[0]]);
        EXPECT_EQ(write.value, saved_value(write.target.second));
    }
    EXPECT_EQ(writes[0].target.second, RegId::PMU_CTRL_0);
#if defined(SIMPLER_TEST_PMU_ARCH_A5)
    EXPECT_EQ(writes[1].target.second, RegId::PMU_CTRL_1);
#endif
}

TEST_F(PmuFinalizeTest, AllCoreFinalizationHasOneFixedWriteSequencePerCore) {
    const int owned[] = {0, 1};
    pmu_aicpu_finalize(owned, 2);
    EXPECT_EQ(read_count, 0U);
    ASSERT_EQ(writes.size(), 2 * writes_per_core());
    for (size_t i = 0; i < writes.size(); ++i) {
        EXPECT_EQ(writes[i].target.first, bases_[physical_[i / writes_per_core()]]);
        EXPECT_EQ(writes[i].value, saved_value(writes[i].target.second));
    }
}

TEST_F(PmuFinalizeTest, EmptyInvalidAndUnboundCoresDoNotTouchMmio) {
    bases_[physical_[1]] = 0;
    pmu_aicpu_init(physical_, 2);
    clear_trace();
    const int invalid_or_unbound[] = {-1, PLATFORM_MAX_CORES, 1};
    pmu_aicpu_finalize(nullptr, 0);
    pmu_aicpu_finalize(invalid_or_unbound, 3);
    EXPECT_EQ(read_count, 0U);
    EXPECT_TRUE(writes.empty());
}
}  // namespace
