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
#include <tuple>
#include <utility>
#include <vector>

#include "aicpu/platform_regs.h"
#include "aicpu/pmu_collector_aicpu.h"
#include "host/pmu_collector.h"

namespace {
using Register = std::pair<uint64_t, RegId>;
enum class Access { Read, Write };
using RegisterAccess = std::tuple<Access, Register, uint32_t>;
uint64_t register_table = 0;
std::map<Register, uint32_t> registers;
std::vector<RegisterAccess> accesses;
size_t read_count = 0;

uint32_t saved_value(uint64_t base, RegId reg) {
    return 0x4ac00000U + static_cast<uint32_t>(base >> 16) + static_cast<uint32_t>(reg);
}
uint32_t close_value(uint64_t base, RegId reg) { return saved_value(base, reg) ^ 0x10000000U; }

uint32_t &register_value(uint64_t base, RegId reg) {
    return registers.try_emplace({base, reg}, saved_value(base, reg)).first->second;
}

void record_write(const Register &target, uint32_t value) {
    accesses.emplace_back(Access::Write, target, value);
    registers[target] = value;
}
}  // namespace

extern "C" {
void set_platform_pmu_reg_addrs(uint64_t value) { register_table = value; }
uint64_t get_platform_pmu_reg_addrs() { return register_table; }
uint64_t get_platform_regs() { return register_table; }
}

uint64_t read_reg(uint64_t base, RegId reg) {
    ++read_count;
    const uint32_t value = register_value(base, reg);
    accesses.emplace_back(Access::Read, Register{base, reg}, value);
    return value;
}

volatile uint32_t *get_reg_ptr(uint64_t base, RegId reg) { return &register_value(base, reg); }

uint32_t reg_load_acquire(const volatile uint32_t *pointer) {
    for (const auto &entry : registers) {
        if (&entry.second == pointer) return static_cast<uint32_t>(read_reg(entry.first.first, entry.first.second));
    }
    ADD_FAILURE() << "read outside the modeled PMU register window";
    return 0;
}

void reg_store_release(volatile uint32_t *pointer, uint32_t value) {
    for (auto &entry : registers) {
        if (&entry.second == pointer) {
            record_write(entry.first, value);
            return;
        }
    }
#if !defined(SIMPLER_TEST_PMU_ARCH_A5)
    for (uint64_t base : {0x10000000ULL, 0x20000000ULL}) {
        for (int index = static_cast<int>(RegId::PMU_CTRL_0); index <= static_cast<int>(RegId::PMU_CNT7_IDX); ++index) {
            RegId reg = static_cast<RegId>(index);
            if (reinterpret_cast<uint64_t>(pointer) == base + reg_offset(reg)) {
                record_write({base, reg}, value);
                return;
            }
        }
    }
#endif
    ADD_FAILURE() << "write outside the modeled PMU register window";
}

#if defined(SIMPLER_TEST_PMU_ARCH_A5)
void write_reg(uint64_t base, RegId reg, uint64_t value) { record_write({base, reg}, value); }
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
        for (uint64_t base : bases_) {
            for (RegId reg : controls())
                register_value(base, reg) = close_value(base, reg);
        }
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
        accesses.clear();
        read_count = 0;
    }

    static std::vector<RegId> controls() {
#if defined(SIMPLER_TEST_PMU_ARCH_A5)
        return {RegId::PMU_CTRL_0, RegId::PMU_CTRL_1};
#else
        return {RegId::PMU_CTRL_0};
#endif
    }

    void expect_finalization(const std::vector<int> &owned) {
        std::vector<RegisterAccess> expected;
        for (int core : owned) {
            const uint64_t base = bases_[physical_[core]];
            for (RegId reg : controls())
                expected.emplace_back(Access::Read, Register{base, reg}, close_value(base, reg));
            for (RegId reg : controls()) {
                expected.emplace_back(Access::Write, Register{base, reg}, saved_value(base, reg));
                EXPECT_EQ(register_value(base, reg), saved_value(base, reg));
            }
        }
        EXPECT_EQ(read_count, owned.size() * controls().size());
        EXPECT_EQ(accesses, expected);
    }
};

TEST_F(PmuFinalizeTest, RestoresOnlyOwnedCoreControlsWithoutPolling) {
    const int owned[] = {0};
    pmu_aicpu_finalize(owned, 1);
    expect_finalization({0});
    for (RegId reg : controls()) {
        const uint64_t unowned_base = bases_[physical_[1]];
        EXPECT_EQ(register_value(unowned_base, reg), close_value(unowned_base, reg));
    }
}

TEST_F(PmuFinalizeTest, AllCoreFinalizationHasOneFixedWriteSequencePerCore) {
    const int owned[] = {0, 1};
    pmu_aicpu_finalize(owned, 2);
    expect_finalization({0, 1});
}

TEST_F(PmuFinalizeTest, EmptyInvalidAndUnboundCoresDoNotTouchMmio) {
    bases_[physical_[1]] = 0;
    pmu_aicpu_init(physical_, 2);
    clear_trace();
    const int invalid_or_unbound[] = {-1, PLATFORM_MAX_CORES, 1};
    pmu_aicpu_finalize(nullptr, 0);
    pmu_aicpu_finalize(invalid_or_unbound, 3);
    EXPECT_EQ(read_count, 0U);
    EXPECT_TRUE(accesses.empty());
}
}  // namespace
