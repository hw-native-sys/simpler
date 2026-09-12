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

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <vector>

#include "tensormap_and_ringbuffer/kernel_core_group.h"

namespace {
using namespace simpler::tmr;

enum class EventKind { Open, Readback, PublishOpen, Cancel, SignalExit, ReadCond, Close, Release, Status };

struct Event {
    EventKind kind;
    int32_t worker;
};

uint32_t load_word(const uint32_t &word) { return __atomic_load_n(&word, __ATOMIC_ACQUIRE); }
void store_word(uint32_t &word, uint32_t value) { __atomic_store_n(&word, value, __ATOMIC_RELEASE); }

struct RegisterCell {
    std::atomic<uint32_t> dispatch{AICPU_IDLE_TASK_ID};
    std::atomic<uint32_t> condition{AICORE_IDLE_VALUE};
    bool opened{false};
    bool read_back{false};
    bool closed{false};
};

// Only the platform boundary is modeled. Logical time advances polling and
// callbacks publish device-owned words; no callback executes the group algorithm.
struct PlatformModel {
    static constexpr uint32_t kPhysicalCount = 75;
    static constexpr uint64_t kEpoch = 17;
    TmrLaunchControl control{};
    std::array<TmrCoreReport, 3> reports{};
    std::array<uint64_t, kPhysicalCount> registers{};
    std::array<RegisterCell, kPhysicalCount> cells{};
    std::vector<Event> events;
    std::atomic<uint64_t> ticks{0};
    uint64_t budget{8};
    std::function<void(uint64_t)> on_tick;

    static uint64_t register_base(uint32_t physical) {
        return 0x1000000ull + static_cast<uint64_t>(physical) * 0x100000ull;
    }

    void reset() {
        control = {};
        reports = {};
        events.clear();
        ticks.store(0);
        on_tick = {};
        for (uint32_t i = 0; i < kPhysicalCount; ++i) {
            registers[i] = register_base(i);
            cells[i].dispatch.store(AICPU_IDLE_TASK_ID);
            cells[i].condition.store(AICORE_IDLE_VALUE);
            cells[i].opened = false;
            cells[i].read_back = false;
            cells[i].closed = false;
        }
    }

    void report_ready(int32_t worker, uint32_t physical) {
        auto &report = reports[worker];
        report.physical_core_id = physical;
        report.core_type = static_cast<uint32_t>(worker == 0 ? CoreType::AIC : CoreType::AIV);
        store_word(report.ready, static_cast<uint32_t>(worker + 1));
    }

    void ready_all() {
        for (int32_t i = 0; i < static_cast<int32_t>(reports.size()); ++i)
            report_ready(i, static_cast<uint32_t>(i));
    }

    KernelHandshakeView view() { return {&control, reports.data(), static_cast<int32_t>(reports.size()), kEpoch}; }

    int32_t physical_index(uint64_t base) const {
        for (uint32_t i = 0; i < kPhysicalCount; ++i)
            if (register_base(i) == base) return static_cast<int32_t>(i);
        ADD_FAILURE() << "Unexpected register base: " << base;
        return 0;
    }

    int32_t worker_index(uint32_t physical) const {
        for (int32_t i = 0; i < static_cast<int32_t>(reports.size()); ++i)
            if (reports[i].physical_core_id == physical) return i;
        ADD_FAILURE() << "Unreported physical core: " << physical;
        return 0;
    }

    size_t first(EventKind kind, int32_t worker) const {
        for (size_t i = 0; i < events.size(); ++i)
            if (events[i].kind == kind && events[i].worker == worker) return i;
        return events.size();
    }

    size_t count(EventKind kind) const {
        return static_cast<size_t>(std::count_if(events.begin(), events.end(), [&](const Event &event) {
            return event.kind == kind;
        }));
    }
};

PlatformModel *active = nullptr;
}  // namespace

uint64_t get_sys_cnt_aicpu() {
    const uint64_t now = active->ticks.fetch_add(1) + 1;
    if (active->on_tick) active->on_tick(now);
    return now;
}

uint64_t platform_aicore_exit_deadline() { return active->ticks.load() + active->budget; }

void platform_init_aicore_regs(uint64_t base) {
    const auto physical = active->physical_index(base);
    const auto worker = active->worker_index(static_cast<uint32_t>(physical));
    auto &cell = active->cells[physical];
    EXPECT_FALSE(cell.opened);
    EXPECT_EQ(load_word(active->reports[worker].command), static_cast<uint32_t>(TmrCoreCommand::Wait));
    EXPECT_EQ(active->reports[worker].round_epoch, 0u);
    cell.opened = true;
    cell.dispatch.store(AICPU_IDLE_TASK_ID, std::memory_order_release);
    active->events.push_back({EventKind::Open, worker});
}

uint64_t read_reg(uint64_t base, RegId reg) {
    const auto physical = active->physical_index(base);
    const auto worker = active->worker_index(static_cast<uint32_t>(physical));
    auto &cell = active->cells[physical];
    if (reg == RegId::DATA_MAIN_BASE) {
        EXPECT_TRUE(cell.opened);
        EXPECT_EQ(load_word(active->reports[worker].command), static_cast<uint32_t>(TmrCoreCommand::Wait));
        EXPECT_EQ(active->reports[worker].round_epoch, 0u);
        cell.read_back = true;
        active->events.push_back({EventKind::Readback, worker});
        return cell.dispatch.load(std::memory_order_acquire);
    }
    EXPECT_EQ(reg, RegId::COND);
    active->events.push_back({EventKind::ReadCond, worker});
    return cell.condition.load(std::memory_order_acquire);
}

void reg_store_release(volatile uint32_t *address, uint32_t value) {
    const auto raw = reinterpret_cast<uintptr_t>(address);
    for (uint32_t physical = 0; physical < PlatformModel::kPhysicalCount; ++physical) {
        if (raw != PlatformModel::register_base(physical) + reg_offset(RegId::DATA_MAIN_BASE)) continue;
        EXPECT_EQ(value, AICORE_EXIT_SIGNAL);
        EXPECT_TRUE(active->cells[physical].opened);
        const auto worker = active->worker_index(physical);
        active->cells[physical].dispatch.store(value, std::memory_order_release);
        active->events.push_back({EventKind::SignalExit, worker});
        return;
    }
    ADD_FAILURE() << "Unexpected register write: " << raw;
}

void platform_close_aicore_window(uint64_t base) {
    const auto physical = active->physical_index(base);
    const auto worker = active->worker_index(static_cast<uint32_t>(physical));
    auto &cell = active->cells[physical];
    EXPECT_TRUE(cell.opened);
    EXPECT_FALSE(cell.closed);
    EXPECT_EQ(cell.condition.load(std::memory_order_acquire), AICORE_EXITED_VALUE);
    EXPECT_EQ(load_word(active->reports[worker].exited), static_cast<uint32_t>(worker + 1));
    EXPECT_EQ(load_word(active->reports[worker].release), static_cast<uint32_t>(TmrCoreRelease::Wait));
    cell.closed = true;
    active->events.push_back({EventKind::Close, worker});
}

namespace aicpu_cache_maintenance {
void invalidate_range_impl(const void *address, size_t bytes) {
    for (const auto &report : active->reports) {
        if (address != &report) continue;
        EXPECT_TRUE(bytes == sizeof(TmrCoreReport) || bytes == offsetof(TmrCoreReport, command));
        return;
    }
    EXPECT_EQ(address, &active->control);
    EXPECT_EQ(bytes, sizeof(TmrLaunchControl));
}

void flush_range_impl(const void *address, size_t bytes) {
    for (int32_t i = 0; i < static_cast<int32_t>(active->reports.size()); ++i) {
        auto &report = active->reports[i];
        if (address != &report.command) continue;
        EXPECT_EQ(bytes, sizeof(TmrCoreReport) - offsetof(TmrCoreReport, command));
        const uint32_t command = load_word(report.command);
        if (load_word(report.release) == static_cast<uint32_t>(TmrCoreRelease::Release)) {
            const auto physical = report.physical_core_id;
            EXPECT_TRUE(active->cells[physical].closed);
            active->events.push_back({EventKind::Release, i});
        } else if (command == static_cast<uint32_t>(TmrCoreCommand::Open)) {
            const auto physical = report.physical_core_id;
            EXPECT_TRUE(active->cells[physical].opened);
            EXPECT_TRUE(active->cells[physical].read_back);
            EXPECT_EQ(report.round_epoch, PlatformModel::kEpoch);
            active->events.push_back({EventKind::PublishOpen, i});
        } else {
            EXPECT_EQ(command, static_cast<uint32_t>(TmrCoreCommand::Cancel));
            active->events.push_back({EventKind::Cancel, i});
        }
        return;
    }
    EXPECT_EQ(address, &active->control.runtime_status);
    EXPECT_EQ(bytes, sizeof(TmrLaunchControl) - offsetof(TmrLaunchControl, runtime_status));
    active->events.push_back({EventKind::Status, -1});
}
}  // namespace aicpu_cache_maintenance

namespace {
class TmrKernelCoreGroupTest : public ::testing::Test {
protected:
    void SetUp() override {
        active = &model;
        model.reset();
    }
    void TearDown() override { active = nullptr; }

    void open_all(KernelCoreGroup &group) {
        model.ready_all();
        ASSERT_TRUE(group.attach(model.view()));
        ASSERT_EQ(group.collect_reports(model.registers.data(), PlatformModel::kPhysicalCount), 0);
        for (int32_t i = 0; i < static_cast<int32_t>(model.reports.size()); ++i)
            group.open(i);
    }

    PlatformModel model;
};

TEST_F(TmrKernelCoreGroupTest, InvalidDuplicateAndWrongTypeReportsNeverOpenRegisters) {
    for (int invalid = 0; invalid < 6; ++invalid) {
        SCOPED_TRACE(invalid);
        model.reset();
        model.ready_all();
        if (invalid == 0) model.reports[2].physical_core_id = PlatformModel::kPhysicalCount;
        if (invalid == 1) model.reports[2].physical_core_id = model.reports[0].physical_core_id;
        if (invalid == 2) model.reports[2].core_type = 99;
        if (invalid == 3) model.registers[2] = 0;
        if (invalid == 4) model.reports[1].core_type = static_cast<uint32_t>(CoreType::AIC);
        if (invalid == 5) model.reports[0].core_type = static_cast<uint32_t>(CoreType::AIV);
        KernelCoreGroup group;
        ASSERT_TRUE(group.attach(model.view()));
        EXPECT_NE(group.collect_reports(model.registers.data(), PlatformModel::kPhysicalCount), 0);
        EXPECT_EQ(model.count(EventKind::Open), 0u);
        EXPECT_EQ(model.count(EventKind::PublishOpen), 0u);
        EXPECT_EQ(model.count(EventKind::SignalExit), 0u);
    }
}

TEST_F(TmrKernelCoreGroupTest, PhysicalCoreAboveLogicalWorkerLimitIsValid) {
    model.ready_all();
    model.reports[2].physical_core_id = 74;
    KernelCoreGroup group;
    ASSERT_TRUE(group.attach(model.view()));
    ASSERT_EQ(group.collect_reports(model.registers.data(), PlatformModel::kPhysicalCount), 0);
    EXPECT_EQ(group.physical_id(2), 74u);
    EXPECT_EQ(group.register_address(2), model.registers[74]);
    group.open(2);
    EXPECT_TRUE(model.cells[74].opened);
    EXPECT_FALSE(model.cells[2].opened);
    EXPECT_EQ(model.count(EventKind::Open), 1u);
}

TEST_F(TmrKernelCoreGroupTest, ReadbackPrecedesEveryOpenPublication) {
    KernelCoreGroup group;
    ASSERT_NO_FATAL_FAILURE(open_all(group));
    EXPECT_EQ(model.count(EventKind::Open), model.reports.size());
    EXPECT_EQ(model.count(EventKind::Readback), model.reports.size());
    EXPECT_EQ(model.count(EventKind::PublishOpen), model.reports.size());
    for (int32_t i = 0; i < static_cast<int32_t>(model.reports.size()); ++i) {
        EXPECT_LT(model.first(EventKind::Open, i), model.first(EventKind::Readback, i));
        EXPECT_LT(model.first(EventKind::Readback, i), model.first(EventKind::PublishOpen, i));
    }
    // The production inline barrier is compiled here; its hardware completion
    // semantics require the device probe, not this host operation trace.
}

TEST_F(TmrKernelCoreGroupTest, CancelPreservesOnlyAnActuallyOpenedWindowsEpoch) {
    model.ready_all();
    KernelCoreGroup group;
    ASSERT_TRUE(group.attach(model.view()));
    ASSERT_EQ(group.collect_reports(model.registers.data(), PlatformModel::kPhysicalCount), 0);
    group.open(0);
    model.reports[1].round_epoch = 999;
    model.reports[2].round_epoch = 999;
    group.cancel();
    EXPECT_EQ(model.reports[0].round_epoch, PlatformModel::kEpoch);
    EXPECT_EQ(model.reports[1].round_epoch, 0u);
    EXPECT_EQ(model.reports[2].round_epoch, 0u);
    for (const auto &report : model.reports) {
        EXPECT_EQ(load_word(report.command), static_cast<uint32_t>(TmrCoreCommand::Cancel));
        EXPECT_EQ(load_word(report.release), static_cast<uint32_t>(TmrCoreRelease::Wait));
    }
}

TEST_F(TmrKernelCoreGroupTest, LatePreWindowReportsCanExitWithoutAnyMmio) {
    KernelCoreGroup group;
    ASSERT_TRUE(group.attach(model.view()));
    bool late_report_published = false;
    model.on_tick = [&](uint64_t now) {
        if (now != 3) return;
        late_report_published = true;
        for (int32_t i = 0; i < static_cast<int32_t>(model.reports.size()); ++i) {
            EXPECT_EQ(load_word(model.reports[i].command), static_cast<uint32_t>(TmrCoreCommand::Cancel));
            EXPECT_EQ(model.reports[i].round_epoch, 0u);
            model.report_ready(i, static_cast<uint32_t>(i));
            store_word(model.reports[i].exited, static_cast<uint32_t>(i + 1));
        }
    };
    EXPECT_EQ(group.finish(), 0);
    EXPECT_TRUE(late_report_published);
    EXPECT_EQ(model.count(EventKind::Open), 0u);
    EXPECT_EQ(model.count(EventKind::Readback), 0u);
    EXPECT_EQ(model.count(EventKind::SignalExit), 0u);
    EXPECT_EQ(model.count(EventKind::ReadCond), 0u);
    EXPECT_EQ(model.count(EventKind::Close), 0u);
    EXPECT_EQ(model.count(EventKind::Release), 0u);
}

TEST_F(TmrKernelCoreGroupTest, OpenedWindowNeedsBothReportAndRegisterAcknowledgment) {
    for (bool missing_report : {false, true}) {
        SCOPED_TRACE(missing_report);
        model.reset();
        model.ready_all();
        KernelCoreGroup group;
        ASSERT_TRUE(group.attach(model.view()));
        ASSERT_EQ(group.collect_reports(model.registers.data(), PlatformModel::kPhysicalCount), 0);
        group.open(0);
        for (int32_t i = 1; i < static_cast<int32_t>(model.reports.size()); ++i)
            store_word(model.reports[i].exited, static_cast<uint32_t>(i + 1));
        if (missing_report) {
            model.cells[0].condition.store(AICORE_EXITED_VALUE, std::memory_order_release);
        } else {
            store_word(model.reports[0].exited, 1);
        }
        EXPECT_NE(group.finish(), 0);
        EXPECT_EQ(model.count(EventKind::Close), 0u);
        EXPECT_EQ(model.count(EventKind::Release), 0u);
        EXPECT_EQ(load_word(model.reports[0].release), static_cast<uint32_t>(TmrCoreRelease::Wait));
    }
}

TEST_F(TmrKernelCoreGroupTest, TimeoutReleasesOnlyFullyAcknowledgedWindows) {
    KernelCoreGroup group;
    ASSERT_NO_FATAL_FAILURE(open_all(group));
    store_word(model.reports[0].exited, 1);
    model.cells[0].condition.store(AICORE_EXITED_VALUE, std::memory_order_release);
    store_word(model.reports[1].exited, 2);
    model.cells[2].condition.store(AICORE_EXITED_VALUE, std::memory_order_release);
    const int32_t cleanup = group.finish();
    EXPECT_NE(cleanup, 0);
    group.publish_status(0, cleanup);
    EXPECT_EQ(model.control.cleanup_status, cleanup);
    EXPECT_EQ(model.count(EventKind::SignalExit), model.reports.size());
    EXPECT_EQ(model.count(EventKind::Close), 1u);
    EXPECT_EQ(model.count(EventKind::Release), 1u);
    EXPECT_LT(model.first(EventKind::SignalExit, 2), model.first(EventKind::Close, 0));
    EXPECT_LT(model.first(EventKind::Close, 0), model.first(EventKind::Release, 0));
    EXPECT_EQ(load_word(model.reports[0].release), static_cast<uint32_t>(TmrCoreRelease::Release));
    EXPECT_EQ(load_word(model.reports[1].release), static_cast<uint32_t>(TmrCoreRelease::Wait));
    EXPECT_EQ(load_word(model.reports[2].release), static_cast<uint32_t>(TmrCoreRelease::Wait));
    EXPECT_FALSE(model.cells[1].closed);
    EXPECT_FALSE(model.cells[2].closed);
}

TEST_F(TmrKernelCoreGroupTest, SuccessfulFinishClosesAllWindowsBeforeAnyRelease) {
    KernelCoreGroup group;
    ASSERT_NO_FATAL_FAILURE(open_all(group));
    for (int32_t i = 0; i < static_cast<int32_t>(model.reports.size()); ++i) {
        store_word(model.reports[i].exited, static_cast<uint32_t>(i + 1));
        model.cells[i].condition.store(AICORE_EXITED_VALUE, std::memory_order_release);
    }
    EXPECT_EQ(group.finish(), 0);
    EXPECT_EQ(model.count(EventKind::Close), model.reports.size());
    EXPECT_EQ(model.count(EventKind::Release), model.reports.size());
    for (int32_t closed = 0; closed < static_cast<int32_t>(model.reports.size()); ++closed)
        for (int32_t released = 0; released < static_cast<int32_t>(model.reports.size()); ++released)
            EXPECT_LT(model.first(EventKind::Close, closed), model.first(EventKind::Release, released));
}

TEST_F(TmrKernelCoreGroupTest, FinalStatusWritesOnlyTheDeviceOwnedControlLine) {
    KernelCoreGroup group;
    ASSERT_TRUE(group.attach(model.view()));
    store_word(model.control.host_cancel, kTmrHostCancel);
    std::fill(std::begin(model.control.host_reserved), std::end(model.control.host_reserved), uint8_t{0xa5});
    group.publish_status(-7, -9);
    EXPECT_EQ(model.control.runtime_status, -7);
    EXPECT_EQ(model.control.cleanup_status, -9);
    EXPECT_EQ(model.control.round_epoch, PlatformModel::kEpoch);
    EXPECT_EQ(load_word(model.control.completion), static_cast<uint32_t>(TmrCompletion::Complete));
    EXPECT_EQ(load_word(model.control.host_cancel), kTmrHostCancel);
    EXPECT_TRUE(
        std::all_of(std::begin(model.control.host_reserved), std::end(model.control.host_reserved), [](uint8_t value) {
            return value == 0xa5;
        })
    );
    EXPECT_EQ(model.count(EventKind::Status), 1u);
}

}  // namespace
