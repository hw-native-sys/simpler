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

/**
 * The per-core counting window AICPU opens around a run and closes after it.
 *
 * `pmu_aicpu_init` resolves each core's register window through that core's
 * physical id, programs the event selectors, and enables the control word
 * after reading the one it replaces; `pmu_aicpu_finalize` writes that word
 * back. Both run here as production code — the window is plain host memory,
 * one block per core addressed at the same offsets the MMIO path uses, so the
 * words a run wrote are readable afterwards.
 *
 * **Nothing here simulates counting.** A host block does not increment, so a
 * counter's *value* is out of scope, exactly as in
 * `test_pmu_run_identity.cpp`. What is in scope is which window a core's
 * programming lands in and which control word each run leaves behind, because
 * that word is what the next run on the same core inherits.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <unistd.h>
#include <vector>

#include "aicpu/device_run_result_base_aicpu.h"
#include "aicpu/platform_regs.h"
#include "aicpu/pmu_collector_aicpu.h"
#include "common/memory_barrier.h"
#include "common/pmu_profiling.h"
#include "host/pmu_collector.h"

#if defined(SIMPLER_TEST_PMU_ARCH_A5)
constexpr int kPmuCounterCount = PMU_COUNTER_COUNT_A5;
#else
constexpr int kPmuCounterCount = PMU_COUNTER_COUNT_A2A3;
#endif

namespace {
// The per-core window table, indexed by physical core id, as the collector
// reads it: a2a3 through the PMU register accessor, a5 through the platform
// register accessor. Both hold the same table here.
uint64_t g_pmu_test_reg_addrs = 0;
}  // namespace

extern "C" {
void set_platform_pmu_reg_addrs(uint64_t pmu_regs) { g_pmu_test_reg_addrs = pmu_regs; }
uint64_t get_platform_pmu_reg_addrs() { return g_pmu_test_reg_addrs; }
uint64_t get_platform_regs() { return g_pmu_test_reg_addrs; }
}

// The register accessors, over the host blocks the fixture allocates. Defined
// here rather than linked for the reason `test_pmu_run_identity.cpp` gives —
// the two arches keep them in different translation units, and both of those
// reach further into the runtime than a PMU case needs — but backed by real
// per-offset storage rather than a refusal, which is what makes the control
// word readable.
volatile uint32_t *get_reg_ptr(uint64_t reg_base_addr, RegId reg) {
    return reinterpret_cast<volatile uint32_t *>(reg_base_addr + reg_offset(reg));
}

uint32_t reg_load_acquire(const volatile uint32_t *p) { return __atomic_load_n(p, __ATOMIC_ACQUIRE); }

void reg_store_release(volatile uint32_t *p, uint32_t v) { __atomic_store_n(p, v, __ATOMIC_RELEASE); }

uint64_t read_reg(uint64_t reg_base_addr, RegId reg) {
    return static_cast<uint64_t>(reg_load_acquire(get_reg_ptr(reg_base_addr, reg)));
}

// a5 declares `write_reg` out-of-line where a2a3 has it inline over
// `reg_store_release`; the arch's headers carry no preprocessor discriminator,
// so the build sets this one.
#if defined(SIMPLER_TEST_PMU_ARCH_A5)
void write_reg(uint64_t reg_base_addr, RegId reg, uint64_t value) {
    reg_store_release(get_reg_ptr(reg_base_addr, reg), static_cast<uint32_t>(value));
}
#endif

namespace {

void *pmu_test_alloc(size_t size) { return std::calloc(1, size); }

int pmu_test_free(void *ptr) {
    std::free(ptr);
    return 0;
}

class PmuCountingWindowTest : public ::testing::Test {
protected:
    static constexpr int kCores = 2;
    static constexpr int kThreads = 1;
    // Every PMU offset either arch defines falls inside this much of a core's
    // block (a2a3's highest is the CNT7 selector at 0x129C, a5's the stop-cycle
    // high half at 0x42AC).
    static constexpr size_t kWindowWords = 0x8000 / sizeof(uint32_t);
    // Logical core i runs on physical core kPhysical[i]: reversed, so a window
    // resolved by the logical index instead of the physical id lands on the
    // wrong block rather than on the right one by coincidence.
    static constexpr uint32_t kPhysical[kCores] = {1, 0};

    void SetUp() override {
        dir_ = std::filesystem::temp_directory_path() /
               ("pmu_window_" + std::to_string(::getpid()) + "_" + std::to_string(++instance_));
        std::filesystem::remove_all(dir_);
        ASSERT_TRUE(std::filesystem::create_directories(dir_));
        csv_ = dir_ / "pmu.csv";

        collector_.begin_run(csv_.string(), PmuEventType::PIPE_UTILIZATION);
        ASSERT_EQ(collector_.init(kCores, kThreads, pmu_test_alloc, nullptr, pmu_test_free, /*device_id=*/0), 0);
        shm_ = collector_.get_pmu_shm_device_ptr();
        ASSERT_NE(shm_, nullptr);

        windows_.assign(static_cast<size_t>(kCores) * kWindowWords, 0);
        for (int core = 0; core < kCores; core++)
            reg_table_[core] = window_base(core);

        set_pmu_enabled(true);
        set_platform_pmu_base(reinterpret_cast<uint64_t>(shm_));
        set_platform_pmu_reg_addrs(reinterpret_cast<uint64_t>(reg_table_));
    }

    void TearDown() override {
        set_pmu_enabled(false);
        set_platform_pmu_base(0);
        set_platform_pmu_reg_addrs(0);
        set_platform_run_result(0, 0);
        collector_.finalize(nullptr, pmu_test_free);
        std::filesystem::remove_all(dir_);
    }

    uint64_t window_base(int core) const {
        return reinterpret_cast<uint64_t>(windows_.data() + static_cast<size_t>(core) * kWindowWords);
    }

    // A word of one core's window, by the same offset the collector addresses.
    uint32_t word(int core, RegId reg) const { return windows_[static_cast<size_t>(core) * kWindowWords + index(reg)]; }

    void set_word(int core, RegId reg, uint32_t value) {
        windows_[static_cast<size_t>(core) * kWindowWords + index(reg)] = value;
    }

    // The run's device-side register sequence, with no records in between:
    // what each core's window holds while the run is open, and what it holds
    // once the run's own teardown has closed it, is all this case reads.
    void open_run(uint64_t epoch, int num_cores = kCores, PmuEventType event_type = PmuEventType::PIPE_UTILIZATION) {
        set_platform_run_result(/*region_base=*/0, epoch);
        pmu_aicpu_init(kPhysical, num_cores, event_type);
    }

    void close_run(int num_cores = kCores) {
        int cores[kCores];
        for (int i = 0; i < num_cores; i++)
            cores[i] = i;
        pmu_aicpu_finalize(cores, num_cores);
    }

    const PmuEventConfig &events(PmuEventType event_type = PmuEventType::PIPE_UTILIZATION) const {
#if defined(SIMPLER_TEST_PMU_ARCH_A5)
        return *pmu_resolve_event_config_a5(event_type);
#else
        return *pmu_resolve_event_config_a2a3(event_type);
#endif
    }

    static size_t index(RegId reg) { return reg_offset(reg) / sizeof(uint32_t); }

    PmuCollector collector_;
    void *shm_ = nullptr;
    std::vector<uint32_t> windows_;
    uint64_t reg_table_[kCores] = {0, 0};
    std::filesystem::path dir_;
    std::filesystem::path csv_;
    static int instance_;
};

int PmuCountingWindowTest::instance_ = 0;

TEST_F(PmuCountingWindowTest, DeviceEntryResetsAccountingWithoutATerminalBank) {
    for (int core = 0; core < kCores; ++core) {
        auto *state = get_pmu_buffer_state(shm_, core);
        state->total_record_count = 37;
        state->dropped_record_count = 11;
#if defined(SIMPLER_TEST_PMU_ARCH_A5)
        state->mismatch_record_count = 13;
#endif
        state->published_record_count = 23;
        state->published_buffer_count = 2;
    }
    open_run(99);
    for (int core = 0; core < kCores; ++core) {
        auto *state = get_pmu_buffer_state(shm_, core);
        EXPECT_EQ(state->total_record_count, 0U);
        EXPECT_EQ(state->dropped_record_count, 0U);
#if defined(SIMPLER_TEST_PMU_ARCH_A5)
        EXPECT_EQ(state->mismatch_record_count, 0U);
#endif
        EXPECT_EQ(state->published_record_count, 0U);
        EXPECT_EQ(state->published_buffer_count, 0U);
    }
    close_run();
}

TEST_F(PmuCountingWindowTest, InvocationSelectionProgramsEventsDespiteAConflictingHeader) {
    get_pmu_header(shm_)->event_type = static_cast<uint32_t>(PmuEventType::PIPE_UTILIZATION);
    for (PmuEventType selected : {PmuEventType::MEMORY, PmuEventType::RESOURCE_CONFLICT}) {
        collector_.begin_run(csv_.string(), selected);
        open_run(/*epoch=*/101, kCores, selected);
        bool differs_from_header = false;
        for (int i = 0; i < kPmuCounterCount; ++i) {
            differs_from_header = differs_from_header || events(selected).event_ids[i] != events().event_ids[i];
            for (int core = 0; core < kCores; ++core) {
                EXPECT_EQ(word(core, reg_index(RegId::PMU_CNT0_IDX, i)), events(selected).event_ids[i]);
            }
        }
        EXPECT_TRUE(differs_from_header);
        EXPECT_EQ(get_pmu_header(shm_)->event_type, static_cast<uint32_t>(PmuEventType::PIPE_UTILIZATION));
        close_run();
    }
}

// The window a core's counters are programmed in is the one its physical id
// names. A resolution that used the logical index would program the other
// core's block, and the records read back through the same resolution would
// then report a core that never ran this run's work.
TEST_F(PmuCountingWindowTest, ProgramsTheWindowEachCoresPhysicalIdNames) {
    open_run(/*epoch=*/1, /*num_cores=*/1);

    EXPECT_EQ(word(/*core=*/1, RegId::PMU_CTRL_0), REG_MMIO_PMU_CTRL_0_ENABLE_VAL)
        << "logical core 0 runs on physical core 1, so its counting window is physical core 1's block";
    for (int i = 0; i < kPmuCounterCount; i++) {
        EXPECT_EQ(word(/*core=*/1, reg_index(RegId::PMU_CNT0_IDX, i)), events().event_ids[i])
            << "event selector " << i << " belongs in the window this core's physical id names";
    }
    EXPECT_EQ(word(/*core=*/0, RegId::PMU_CTRL_0), 0u) << "physical core 0 carries no logical core in this run";
    EXPECT_EQ(word(/*core=*/0, reg_index(RegId::PMU_CNT0_IDX, 0)), 0u);

    close_run(/*num_cores=*/1);
}

// What a run leaves behind is the word it found, not a zero and not the
// enable: the chip is shared, and a run that invents the pre-run state hands
// the next owner of the core a control word nobody wrote.
TEST_F(PmuCountingWindowTest, FinalizeRestoresTheControlWordInitReplaced) {
    constexpr uint32_t kForeign0 = 0x5Au;
    for (int core = 0; core < kCores; core++)
        set_word(core, RegId::PMU_CTRL_0, kForeign0);
#if defined(SIMPLER_TEST_PMU_ARCH_A5)
    constexpr uint32_t kForeign1 = 0x3Cu;
    for (int core = 0; core < kCores; core++)
        set_word(core, RegId::PMU_CTRL_1, kForeign1);
#endif

    open_run(/*epoch=*/1);
    for (int core = 0; core < kCores; core++) {
        EXPECT_EQ(word(core, RegId::PMU_CTRL_0), REG_MMIO_PMU_CTRL_0_ENABLE_VAL) << "core " << core << " is counting";
        EXPECT_EQ(word(core, RegId::PMU_START_CYC0), 0u);
        EXPECT_EQ(word(core, RegId::PMU_STOP_CYC0), 0xFFFFFFFFu);
#if defined(SIMPLER_TEST_PMU_ARCH_A5)
        EXPECT_EQ(word(core, RegId::PMU_CTRL_1), REG_MMIO_PMU_CTRL_1_ENABLE_VAL);
#endif
    }

    close_run();
    for (int core = 0; core < kCores; core++) {
        EXPECT_EQ(word(core, RegId::PMU_CTRL_0), kForeign0) << "core " << core << "'s window is closed again";
#if defined(SIMPLER_TEST_PMU_ARCH_A5)
        EXPECT_EQ(word(core, RegId::PMU_CTRL_1), kForeign1);
#endif
    }
}

// The saved word is each run's own observation. Runs on one device operator
// are serial — a run's teardown precedes its successor's bring-up — so a
// successor that inherited a predecessor's saved word would restore a state
// that was already stale when it started.
TEST_F(PmuCountingWindowTest, EachRunRestoresTheWordItsOwnInitFound) {
    constexpr uint32_t kFirst = 0x11u;
    constexpr uint32_t kSecond = 0x22u;

    for (int core = 0; core < kCores; core++)
        set_word(core, RegId::PMU_CTRL_0, kFirst);
    open_run(/*epoch=*/1);
    close_run();
    for (int core = 0; core < kCores; core++)
        ASSERT_EQ(word(core, RegId::PMU_CTRL_0), kFirst);

    // Whatever else runs on this chip between the two runs.
    for (int core = 0; core < kCores; core++)
        set_word(core, RegId::PMU_CTRL_0, kSecond);
    open_run(/*epoch=*/2);
    for (int core = 0; core < kCores; core++)
        EXPECT_EQ(word(core, RegId::PMU_CTRL_0), REG_MMIO_PMU_CTRL_0_ENABLE_VAL)
            << "core " << core << " counts for the second run too";
    close_run();
    for (int core = 0; core < kCores; core++)
        EXPECT_EQ(word(core, RegId::PMU_CTRL_0), kSecond) << "core " << core << " restored a word from an earlier run";
}

// A core with no window is not a core with window zero: the sim path resolves
// no register table at all, and a run that wrote through it anyway would store
// to a null page.
TEST_F(PmuCountingWindowTest, NoRegisterTableLeavesEveryWindowAlone) {
    set_platform_pmu_reg_addrs(0);
    for (int core = 0; core < kCores; core++)
        set_word(core, RegId::PMU_CTRL_0, 0x77u);

    open_run(/*epoch=*/1);
    close_run();

    for (int core = 0; core < kCores; core++) {
        EXPECT_EQ(word(core, RegId::PMU_CTRL_0), 0x77u) << "core " << core << "'s window was written without a table";
        EXPECT_EQ(word(core, reg_index(RegId::PMU_CNT0_IDX, 0)), 0u);
    }
}

}  // namespace
