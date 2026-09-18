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

#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <functional>
#include <future>
#include <thread>
#include <utility>
#include <vector>

#include "host/device_fault_monitor.h"

namespace {

/** A stand-in driver: counts registrations and can be made to refuse. */
struct FakeDriver {
    int installs{0};
    int uninstalls{0};
    int install_rc{0};
    int uninstall_rc{0};
    long pid{4242};
    bool callback_held{false};

    DeviceFaultMonitor::Ops ops() {
        return DeviceFaultMonitor::Ops{
            [this]() {
                ++installs;
                if (install_rc != 0) return install_rc;
                callback_held = true;
                return 0;
            },
            [this]() {
                ++uninstalls;
                if (uninstall_rc != 0) return uninstall_rc;
                callback_held = false;
                return 0;
            },
            [this]() {
                return pid;
            },
        };
    }
};

DeviceFaultNotice notice(uint32_t device, uint32_t stream, uint32_t code) {
    DeviceFaultNotice out;
    out.device_id = device;
    out.stream_id = stream;
    out.error_code = code;
    return out;
}

}  // namespace

// ===== Refcounted ownership of the single process slot =====

TEST(DeviceFaultMonitorTest, TheFirstReferenceInstallsAndTheLastRetires) {
    FakeDriver driver;
    DeviceFaultMonitor monitor(driver.ops());

    ASSERT_EQ(monitor.acquire(), 0);
    EXPECT_EQ(driver.installs, 1);
    EXPECT_TRUE(monitor.installed());

    ASSERT_EQ(monitor.acquire(), 0);
    ASSERT_EQ(monitor.acquire(), 0);
    // Several runners in one process must not re-register: the slot holds one
    // callback and the last writer wins silently.
    EXPECT_EQ(driver.installs, 1);
    EXPECT_EQ(monitor.references(), 3u);

    monitor.release();
    monitor.release();
    EXPECT_EQ(driver.uninstalls, 0);
    EXPECT_TRUE(monitor.installed());

    monitor.release();
    EXPECT_EQ(driver.uninstalls, 1);
    EXPECT_FALSE(monitor.installed());
    EXPECT_FALSE(driver.callback_held);
}

TEST(DeviceFaultMonitorTest, AFailedInstallTakesNoReference) {
    FakeDriver driver;
    driver.install_rc = -7;
    DeviceFaultMonitor monitor(driver.ops());

    EXPECT_EQ(monitor.acquire(), -7);
    EXPECT_EQ(monitor.references(), 0u);
    EXPECT_FALSE(monitor.installed());

    // A later caller retries rather than inheriting a registration that was
    // never made.
    driver.install_rc = 0;
    ASSERT_EQ(monitor.acquire(), 0);
    EXPECT_EQ(driver.installs, 2);
    EXPECT_EQ(monitor.references(), 1u);
}

TEST(DeviceFaultMonitorTest, AFailedRetireStillDropsTheReference) {
    FakeDriver driver;
    driver.uninstall_rc = -9;
    DeviceFaultMonitor monitor(driver.ops());

    ASSERT_EQ(monitor.acquire(), 0);
    monitor.release();
    // Holding a reference for a registration we cannot retire would pin the
    // owner for the process lifetime.
    EXPECT_EQ(monitor.references(), 0u);
    EXPECT_FALSE(monitor.installed());
    EXPECT_EQ(monitor.last_uninstall_rc(), -9);

    ASSERT_EQ(monitor.acquire(), 0);
    EXPECT_EQ(driver.installs, 2);
}

TEST(DeviceFaultMonitorTest, ReleasingWithoutAReferenceIsANoOp) {
    FakeDriver driver;
    DeviceFaultMonitor monitor(driver.ops());
    monitor.release();
    monitor.release();
    EXPECT_EQ(driver.uninstalls, 0);
    EXPECT_EQ(monitor.references(), 0u);
}

// ===== fork =====

TEST(DeviceFaultMonitorTest, AChildDoesNotInheritTheParentsRegistration) {
    FakeDriver driver;
    DeviceFaultMonitor monitor(driver.ops());

    ASSERT_EQ(monitor.acquire(), 0);
    ASSERT_EQ(monitor.acquire(), 0);
    ASSERT_EQ(driver.installs, 1);

    // The child image sees the parent's refcount and installed flag through
    // copy-on-write. Whether the driver-side registration crossed the fork is
    // not measured, so the owner must not take either answer on trust.
    driver.pid = 9999;

    EXPECT_EQ(monitor.references(), 0u);
    EXPECT_FALSE(monitor.installed());
    ASSERT_EQ(monitor.acquire(), 0);
    EXPECT_EQ(driver.installs, 2) << "the child must register for its own driver context";
    EXPECT_EQ(monitor.references(), 1u);
}

TEST(DeviceFaultMonitorTest, AChildDoesNotReportItsParentsFaults) {
    FakeDriver driver;
    DeviceFaultMonitor monitor(driver.ops());
    ASSERT_EQ(monitor.acquire(), 0);
    monitor.report(notice(3, 44, 507018));
    monitor.report(notice(3, 43, 507015));
    ASSERT_EQ(monitor.sequence(), 2u);

    driver.pid = 9999;
    ASSERT_EQ(monitor.acquire(), 0);
    EXPECT_EQ(monitor.sequence(), 0u) << "inherited notice counts belong to the parent's device work";

    DeviceFaultNotice out;
    EXPECT_EQ(monitor.read_notice(0, &out), DeviceFaultNoticeRead::Pending)
        << "an inherited ring slot must be unreachable";
}

TEST(DeviceFaultMonitorTest, AChildsOwnRetireDoesNotTouchTheParentsSlot) {
    FakeDriver driver;
    DeviceFaultMonitor monitor(driver.ops());
    ASSERT_EQ(monitor.acquire(), 0);

    driver.pid = 9999;
    // The child holds nothing yet, so this release must not call the driver —
    // retiring here would be the child retiring a registration it never made.
    monitor.release();
    EXPECT_EQ(driver.uninstalls, 0);
}

// ===== device reset =====

TEST(DeviceFaultMonitorTest, AConfirmedResetReRegistersWhileReferencesAreHeld) {
    FakeDriver driver;
    DeviceFaultMonitor monitor(driver.ops());
    ASSERT_EQ(monitor.acquire(), 0);
    ASSERT_EQ(driver.installs, 1);

    // Survival across a force reset is unmeasured, so the registration is
    // remade rather than assumed.
    ASSERT_EQ(monitor.reinstall_after_device_reset(), 0);
    EXPECT_EQ(driver.installs, 2);
    EXPECT_TRUE(monitor.installed());
    EXPECT_EQ(monitor.references(), 1u) << "re-registering is not an extra reference";

    monitor.release();
    EXPECT_EQ(driver.uninstalls, 1);
}

TEST(DeviceFaultMonitorTest, AResetWithNoReferencesLeavesTheNextAcquireToInstall) {
    FakeDriver driver;
    DeviceFaultMonitor monitor(driver.ops());

    ASSERT_EQ(monitor.reinstall_after_device_reset(), 0);
    EXPECT_EQ(driver.installs, 0) << "nobody is listening, so nothing is registered";
    EXPECT_FALSE(monitor.installed());

    ASSERT_EQ(monitor.acquire(), 0);
    EXPECT_EQ(driver.installs, 1);
}

TEST(DeviceFaultMonitorTest, AFailedReInstallIsReportedAndLeavesTheSlotUninstalled) {
    FakeDriver driver;
    DeviceFaultMonitor monitor(driver.ops());
    ASSERT_EQ(monitor.acquire(), 0);

    driver.install_rc = -5;
    EXPECT_EQ(monitor.reinstall_after_device_reset(), -5);
    EXPECT_FALSE(monitor.installed());
    // The reference is still held, so the next acquire re-tries the install
    // rather than reporting a registration that is not there.
    EXPECT_EQ(monitor.references(), 1u);
    driver.install_rc = 0;
    ASSERT_EQ(monitor.acquire(), 0);
    EXPECT_TRUE(monitor.installed());
}

// ===== late notifications =====

TEST(DeviceFaultMonitorTest, ANotificationAfterRetirementIsStillRecorded) {
    FakeDriver driver;
    DeviceFaultMonitor monitor(driver.ops());
    ASSERT_EQ(monitor.acquire(), 0);
    monitor.release();
    ASSERT_FALSE(monitor.installed());

    // Nothing orders a driver callback against the retire that preceded it, so
    // the handler has to be safe with no reference held.
    monitor.report(notice(5, 43, 507015));
    ASSERT_EQ(monitor.sequence(), 1u);
    DeviceFaultNotice out;
    ASSERT_EQ(monitor.read_notice(0, &out), DeviceFaultNoticeRead::Ok);
    EXPECT_EQ(out.device_id, 5u);
    EXPECT_EQ(out.error_code, 507015u);
}

TEST(DeviceFaultMonitorTest, ANotificationBeforeAnyReferenceIsRecorded) {
    FakeDriver driver;
    DeviceFaultMonitor monitor(driver.ops());
    monitor.report(notice(1, 44, 507018));
    EXPECT_EQ(monitor.sequence(), 1u);
}

// ===== the notice ring =====

TEST(DeviceFaultMonitorTest, NoticesReadBackInReportOrderWithTheirOwnFields) {
    FakeDriver driver;
    DeviceFaultMonitor monitor(driver.ops());
    monitor.report(notice(2, 44, 507018));
    monitor.report(notice(2, 43, 507015));

    DeviceFaultNotice first;
    DeviceFaultNotice second;
    ASSERT_EQ(monitor.read_notice(0, &first), DeviceFaultNoticeRead::Ok);
    ASSERT_EQ(monitor.read_notice(1, &second), DeviceFaultNoticeRead::Ok);
    EXPECT_EQ(first.stream_id, 44u);
    EXPECT_EQ(first.error_code, 507018u);
    EXPECT_EQ(second.stream_id, 43u);
    EXPECT_EQ(second.error_code, 507015u);
}

TEST(DeviceFaultMonitorTest, AnIndexNeverReportedReadsAsAbsentNotAsAZeroNotice) {
    FakeDriver driver;
    DeviceFaultMonitor monitor(driver.ops());
    DeviceFaultNotice out;
    EXPECT_EQ(monitor.read_notice(0, &out), DeviceFaultNoticeRead::Pending);
    monitor.report(notice(1, 44, 507018));
    EXPECT_EQ(monitor.read_notice(0, &out), DeviceFaultNoticeRead::Ok);
    EXPECT_EQ(monitor.read_notice(1, &out), DeviceFaultNoticeRead::Pending);
}

TEST(DeviceFaultMonitorTest, AnOverwrittenNoticeReadsAsAbsentRatherThanAsAnotherNotice) {
    FakeDriver driver;
    DeviceFaultMonitor monitor(driver.ops());
    const uint64_t retained = DeviceFaultMonitor::retained_notices();
    for (uint64_t i = 0; i < retained + 3; ++i) {
        monitor.report(notice(static_cast<uint32_t>(i), 44, static_cast<uint32_t>(i + 1)));
    }
    EXPECT_EQ(monitor.sequence(), retained + 3);

    DeviceFaultNotice out;
    // The three oldest are gone, and must not read back as whatever wrapped
    // onto their slots.
    EXPECT_EQ(monitor.read_notice(0, &out), DeviceFaultNoticeRead::Lost);
    EXPECT_EQ(monitor.read_notice(2, &out), DeviceFaultNoticeRead::Lost);
    ASSERT_EQ(monitor.read_notice(3, &out), DeviceFaultNoticeRead::Ok);
    EXPECT_EQ(out.device_id, 3u);
    ASSERT_EQ(monitor.read_notice(retained + 2, &out), DeviceFaultNoticeRead::Ok);
    EXPECT_EQ(out.device_id, static_cast<uint32_t>(retained + 2));
}

TEST(DeviceFaultMonitorTest, ConcurrentDriverThreadsEachGetTheirOwnIndex) {
    FakeDriver driver;
    DeviceFaultMonitor monitor(driver.ops());
    constexpr int kThreads = 8;
    std::atomic<bool> go{false};
    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int i = 0; i < kThreads; ++i) {
        threads.emplace_back([&monitor, &go, i] {
            while (!go.load(std::memory_order_acquire)) {}
            monitor.report(notice(static_cast<uint32_t>(i), 44, static_cast<uint32_t>(100 + i)));
        });
    }
    go.store(true, std::memory_order_release);
    for (std::thread &t : threads)
        t.join();

    ASSERT_EQ(monitor.sequence(), static_cast<uint64_t>(kThreads));
    // Every notice is readable and internally consistent: a device id paired
    // with another report's error code would be a torn slot.
    int seen = 0;
    for (uint64_t index = 0; index < static_cast<uint64_t>(kThreads); ++index) {
        DeviceFaultNotice out;
        ASSERT_EQ(monitor.read_notice(index, &out), DeviceFaultNoticeRead::Ok) << "index " << index;
        EXPECT_EQ(out.error_code, out.device_id + 100u);
        ++seen;
    }
    EXPECT_EQ(seen, kThreads);
}

TEST(DeviceFaultMonitorTest, ReportingCompletesWhileTheOwnershipMutexIsHeld) {
    // The handler runs on a driver thread, so if it took the mutex that
    // serializes install/retire it could block behind a host teardown.
    //
    // Interleaving short acquire/release cycles with reports does not test
    // that: a `report` that *did* take the mutex would simply take it between
    // two cycles and still finish, so the assertion could not fail for the
    // reason it names. The mutex has to be genuinely held for the whole
    // report — which the fake driver can do, because `install` runs under it.
    FakeDriver driver;
    std::atomic<bool> install_entered{false};
    std::atomic<bool> allow_install_to_return{false};
    DeviceFaultMonitor::Ops ops = driver.ops();
    const std::function<int()> real_install = ops.install;
    ops.install = [&install_entered, &allow_install_to_return, real_install]() {
        install_entered.store(true, std::memory_order_release);
        while (!allow_install_to_return.load(std::memory_order_acquire)) {}
        return real_install();
    };
    DeviceFaultMonitor monitor(std::move(ops));

    // Parks inside `install`, holding the ownership mutex for as long as we like.
    auto owner = std::async(std::launch::async, [&monitor] {
        return monitor.acquire();
    });
    while (!install_entered.load(std::memory_order_acquire)) {}

    // A regression here would block until the gate opens, so the report is
    // given a bounded budget rather than being allowed to hang the suite.
    auto reporter = std::async(std::launch::async, [&monitor] {
        monitor.report(notice(7, 43, 507015));
    });
    const bool reported_in_time = reporter.wait_for(std::chrono::seconds(5)) == std::future_status::ready;

    allow_install_to_return.store(true, std::memory_order_release);
    EXPECT_EQ(owner.get(), 0);
    if (reported_in_time) {
        reporter.get();
    } else {
        reporter.wait();
        FAIL() << "report() did not finish while the ownership mutex was held — the handler is taking it";
    }
    EXPECT_EQ(monitor.sequence(), 1u);
    DeviceFaultNotice out;
    ASSERT_EQ(monitor.read_notice(0, &out), DeviceFaultNoticeRead::Ok);
    EXPECT_EQ(out.error_code, 507015u);
}

// ===== The snapshot protocol: no mixed records =====

TEST(DeviceFaultMonitorTest, AnOverwriteInProgressReadsAsLostNotAsAMixedRecord) {
    // The interleaving a published-marker bracket cannot catch: the overwriting
    // writer has replaced the fields but not yet published, so the old marker
    // is still in place at both ends of the reader's read.
    FakeDriver driver;
    DeviceFaultMonitor monitor(driver.ops());
    const uint64_t retained = DeviceFaultMonitor::retained_notices();

    monitor.report(notice(1, 44, 507018));
    DeviceFaultNotice first;
    ASSERT_EQ(monitor.read_notice(0, &first), DeviceFaultNoticeRead::Ok);
    ASSERT_EQ(first.device_id, 1u);

    // Fill the ring so the next report lands back on slot 0.
    for (uint64_t i = 1; i < retained; ++i) {
        monitor.report(notice(static_cast<uint32_t>(100 + i), 44, static_cast<uint32_t>(200 + i)));
    }

    std::atomic<bool> holding{false};
    std::atomic<bool> release{false};
    const std::function<void()> pause = [&holding, &release] {
        holding.store(true, std::memory_order_release);
        while (!release.load(std::memory_order_acquire)) {}
    };
    std::thread overwriter([&monitor, &pause] {
        monitor.report_with_pause(notice(2, 43, 507015), ReportPause::BeforePublish, &pause);
    });
    while (!holding.load(std::memory_order_acquire)) {}

    // Slot 0's fields are now the new notice's; its publication is not.
    DeviceFaultNotice out;
    const DeviceFaultNoticeRead read = monitor.read_notice(0, &out);
    EXPECT_EQ(read, DeviceFaultNoticeRead::Lost) << "an index the ring has reclaimed must not read back at all";
    if (read == DeviceFaultNoticeRead::Ok) {
        // If this protocol regresses, say which half leaked rather than only
        // that the state was wrong.
        EXPECT_EQ(out.device_id, first.device_id);
        EXPECT_EQ(out.error_code, first.error_code);
    }

    release.store(true, std::memory_order_release);
    overwriter.join();
    ASSERT_EQ(monitor.read_notice(retained, &out), DeviceFaultNoticeRead::Ok);
    EXPECT_EQ(out.device_id, 2u);
    EXPECT_EQ(out.error_code, 507015u);
}

TEST(DeviceFaultMonitorTest, ASlotStillBeingWrittenIsNotEvictedByALaterReport) {
    // The exclusion, stated from the loser's side. Two reporters a full ring
    // apart target one slot; the first still holds it, so the second writes
    // nothing and is counted. The newest notice is therefore *not* guaranteed
    // to win — a report in flight is never evicted, which is what makes "the
    // fields belong to the generation the state names" true at all times.
    FakeDriver driver;
    DeviceFaultMonitor monitor(driver.ops());
    const uint64_t retained = DeviceFaultMonitor::retained_notices();

    std::atomic<bool> holding{false};
    std::atomic<bool> release{false};
    const std::function<void()> pause = [&holding, &release] {
        holding.store(true, std::memory_order_release);
        while (!release.load(std::memory_order_acquire)) {}
    };
    std::thread owner([&monitor, &pause] {
        monitor.report_with_pause(notice(7, 44, 111), ReportPause::BeforePublish, &pause);
    });
    while (!holding.load(std::memory_order_acquire)) {}

    // Advance to the same slot. This report finds it still being written.
    for (uint64_t i = 1; i < retained; ++i) {
        monitor.report(notice(static_cast<uint32_t>(50 + i), 44, static_cast<uint32_t>(60 + i)));
    }
    monitor.report(notice(9, 43, 999));
    EXPECT_GE(monitor.dropped(), 1u) << "a report that cannot take its slot must be counted, not written";

    release.store(true, std::memory_order_release);
    owner.join();

    DeviceFaultNotice out;
    ASSERT_EQ(monitor.read_notice(0, &out), DeviceFaultNoticeRead::Ok);
    EXPECT_EQ(out.device_id, 7u) << "the owner's own notice must survive intact";
    EXPECT_EQ(out.error_code, 111u);
    // And the report that lost reads as gone rather than parking the consumer.
    EXPECT_EQ(monitor.read_notice(retained, &out), DeviceFaultNoticeRead::Lost);
}

TEST(DeviceFaultMonitorTest, AReportThatLosesItsSlotExecutesNothingAfterTheExchange) {
    // The exclusion, asserted directly rather than through its consequences.
    // A flag-then-write scheme lets a loser continue into its field writes,
    // and whether that corrupts anything then depends on which store happens
    // to land last — so the invariant to pin is that the loser never gets
    // there at all. The losing report's own pause is the witness: if it runs,
    // the loser reached code it must not reach.
    FakeDriver driver;
    DeviceFaultMonitor monitor(driver.ops());
    const uint64_t retained = DeviceFaultMonitor::retained_notices();

    std::atomic<bool> holding{false};
    std::atomic<bool> release{false};
    const std::function<void()> owner_pause = [&holding, &release] {
        holding.store(true, std::memory_order_release);
        while (!release.load(std::memory_order_acquire)) {}
    };
    std::thread owner([&monitor, &owner_pause] {
        monitor.report_with_pause(notice(3, 44, 507018), ReportPause::AfterClaim, &owner_pause);
    });
    while (!holding.load(std::memory_order_acquire)) {}

    DeviceFaultNotice out;
    EXPECT_EQ(monitor.read_notice(0, &out), DeviceFaultNoticeRead::Pending)
        << "a slot whose fields are not written yet must not read as complete";

    for (uint64_t i = 1; i < retained; ++i) {
        monitor.report(notice(static_cast<uint32_t>(70 + i), 44, static_cast<uint32_t>(80 + i)));
    }

    // This report wants slot 0, which the owner still holds.
    std::atomic<bool> loser_ran_past_the_exchange{false};
    const std::function<void()> loser_pause = [&loser_ran_past_the_exchange] {
        loser_ran_past_the_exchange.store(true, std::memory_order_release);
    };
    monitor.report_with_pause(notice(4, 43, 507015), ReportPause::AfterClaim, &loser_pause);
    EXPECT_FALSE(loser_ran_past_the_exchange.load(std::memory_order_acquire))
        << "a report that lost its slot must not execute a single store into it";

    release.store(true, std::memory_order_release);
    owner.join();

    ASSERT_EQ(monitor.read_notice(0, &out), DeviceFaultNoticeRead::Ok);
    EXPECT_EQ(out.device_id, 3u) << "the owner's fields, not the later report's";
    EXPECT_EQ(out.error_code, 507018u);
    EXPECT_EQ(monitor.read_notice(retained, &out), DeviceFaultNoticeRead::Lost);
}

TEST(DeviceFaultMonitorTest, ADroppedReportNeverParksTheConsumerOnItsIndex) {
    // A report that loses its slot writes nothing, so its index would sit at
    // Pending forever unless the loss is recorded. The consumer has to be able
    // to move past it.
    FakeDriver driver;
    DeviceFaultMonitor monitor(driver.ops());
    DeviceFaultNoticeCursor cursor;
    const uint64_t retained = DeviceFaultMonitor::retained_notices();

    std::atomic<bool> holding{false};
    std::atomic<bool> release{false};
    const std::function<void()> pause = [&holding, &release] {
        holding.store(true, std::memory_order_release);
        while (!release.load(std::memory_order_acquire)) {}
    };
    std::thread owner([&monitor, &pause] {
        monitor.report_with_pause(notice(1, 44, 11), ReportPause::BeforePublish, &pause);
    });
    while (!holding.load(std::memory_order_acquire)) {}
    for (uint64_t i = 1; i < retained; ++i) {
        monitor.report(notice(static_cast<uint32_t>(i), 44, static_cast<uint32_t>(i)));
    }
    monitor.report(notice(99, 43, 99));  // loses slot 0, writes nothing
    release.store(true, std::memory_order_release);
    owner.join();

    // One more report so the cursor has something beyond the lost index.
    monitor.report(notice(123, 44, 123));

    std::vector<DeviceFaultNotice> seen;
    DeviceFaultNoticeCursor::Progress progress = cursor.consume(monitor, [&seen](const DeviceFaultNotice &n) {
        seen.push_back(n);
    });
    EXPECT_FALSE(progress.pending) << "the lost index must not park the cursor";
    EXPECT_GE(progress.lost, 1u);
    EXPECT_EQ(cursor.position(), monitor.sequence());
    ASSERT_FALSE(seen.empty());
    EXPECT_EQ(seen.back().device_id, 123u) << "reports after the lost one must still arrive";
}

// ===== The consume rule: a pending notice is never skipped =====

namespace {

std::vector<DeviceFaultNotice> drain(
    DeviceFaultNoticeCursor &cursor, const DeviceFaultMonitor &monitor,
    DeviceFaultNoticeCursor::Progress *progress_out = nullptr
) {
    std::vector<DeviceFaultNotice> seen;
    const DeviceFaultNoticeCursor::Progress progress = cursor.consume(monitor, [&seen](const DeviceFaultNotice &n) {
        seen.push_back(n);
    });
    if (progress_out != nullptr) *progress_out = progress;
    return seen;
}

}  // namespace

TEST(DeviceFaultNoticeCursorTest, APendingNoticeIsDeliveredOnTheNextReadNotLost) {
    // The sequence counts reserved slots, so a reader can see a count that
    // includes a report still in flight. Consuming that index would lose the
    // notice the instant it became readable.
    FakeDriver driver;
    DeviceFaultMonitor monitor(driver.ops());
    DeviceFaultNoticeCursor cursor;

    std::atomic<bool> holding{false};
    std::atomic<bool> release{false};
    const std::function<void()> pause = [&holding, &release] {
        holding.store(true, std::memory_order_release);
        while (!release.load(std::memory_order_acquire)) {}
    };
    std::thread reporter([&monitor, &pause] {
        monitor.report_with_pause(notice(4, 44, 507018), ReportPause::BeforePublish, &pause);
    });
    while (!holding.load(std::memory_order_acquire)) {}

    ASSERT_EQ(monitor.sequence(), 1u) << "the slot is reserved, so the count already includes it";
    DeviceFaultNoticeCursor::Progress first;
    EXPECT_TRUE(drain(cursor, monitor, &first).empty());
    EXPECT_TRUE(first.pending);
    EXPECT_EQ(cursor.position(), 0u) << "the cursor must not move past a report in flight";

    release.store(true, std::memory_order_release);
    reporter.join();

    DeviceFaultNoticeCursor::Progress second;
    const std::vector<DeviceFaultNotice> seen = drain(cursor, monitor, &second);
    ASSERT_EQ(seen.size(), 1u) << "the notice that was pending must arrive, exactly once";
    EXPECT_EQ(seen[0].device_id, 4u);
    EXPECT_EQ(second.delivered, 1u);
    EXPECT_FALSE(second.pending);
    EXPECT_EQ(cursor.position(), 1u);
}

TEST(DeviceFaultNoticeCursorTest, EachNoticeIsDeliveredExactlyOnce) {
    FakeDriver driver;
    DeviceFaultMonitor monitor(driver.ops());
    DeviceFaultNoticeCursor cursor;

    monitor.report(notice(1, 44, 11));
    monitor.report(notice(2, 43, 22));
    EXPECT_EQ(drain(cursor, monitor).size(), 2u);
    EXPECT_TRUE(drain(cursor, monitor).empty());

    monitor.report(notice(3, 44, 33));
    const std::vector<DeviceFaultNotice> seen = drain(cursor, monitor);
    ASSERT_EQ(seen.size(), 1u);
    EXPECT_EQ(seen[0].device_id, 3u);
}

TEST(DeviceFaultNoticeCursorTest, OverwrittenNoticesAreCountedAsLostAndSkipped) {
    FakeDriver driver;
    DeviceFaultMonitor monitor(driver.ops());
    DeviceFaultNoticeCursor cursor;
    const uint64_t retained = DeviceFaultMonitor::retained_notices();

    for (uint64_t i = 0; i < retained + 3; ++i) {
        monitor.report(notice(static_cast<uint32_t>(i), 44, static_cast<uint32_t>(i + 1)));
    }
    DeviceFaultNoticeCursor::Progress progress;
    const std::vector<DeviceFaultNotice> seen = drain(cursor, monitor, &progress);
    EXPECT_EQ(progress.lost, 3u);
    EXPECT_EQ(seen.size(), retained);
    EXPECT_FALSE(progress.pending);
    EXPECT_EQ(cursor.position(), retained + 3);
}

TEST(DeviceFaultNoticeCursorTest, ANewReaderStartsFromTheCurrentPositionNotFromZero) {
    FakeDriver driver;
    DeviceFaultMonitor monitor(driver.ops());
    monitor.report(notice(1, 44, 11));

    DeviceFaultNoticeCursor cursor;
    cursor.skip_to_current(monitor);
    EXPECT_TRUE(drain(cursor, monitor).empty()) << "a notice reported before this reader existed is not its to report";

    monitor.report(notice(2, 43, 22));
    ASSERT_EQ(drain(cursor, monitor).size(), 1u);
}

// ===== The fork boundary, with a real fork =====

TEST(DeviceFaultMonitorTest, AChildCanAcquireAfterForkingWhileTheLockWasHeld) {
    // The hazard a pid stamp cannot fix: the child inherits `mutex_` locked by
    // a thread that does not exist there, and the pid check lives under that
    // mutex. The atfork entry points are what make the child's first
    // operation reachable at all.
    FakeDriver driver;
    DeviceFaultMonitor monitor(driver.ops());
    ASSERT_EQ(monitor.acquire(), 0);

    monitor.before_fork();  // what pthread_atfork's prepare handler does
    const pid_t pid = fork();
    ASSERT_GE(pid, 0);
    if (pid == 0) {
        monitor.after_fork_in_child();
        // Must not hang, and must install for this process rather than count
        // a reference to the parent's registration.
        const bool acquired = monitor.acquire() == 0;
        const bool installed = monitor.installed();
        const bool own_refcount = monitor.references() == 1;
        _exit((acquired && installed && own_refcount) ? 0 : 1);
    }
    monitor.after_fork_in_parent();

    int status = 0;
    bool reaped = false;
    // A child that deadlocked would never be reaped; bound the wait rather
    // than hanging the suite.
    for (int waited_ms = 0; waited_ms < 5000 && !reaped; ++waited_ms) {
        const pid_t done = waitpid(pid, &status, WNOHANG);
        ASSERT_NE(done, -1);
        if (done == pid) {
            reaped = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (!reaped) {
        kill(pid, SIGKILL);
        (void)waitpid(pid, &status, 0);
        FAIL() << "the child never completed its first acquire — it inherited a lock it cannot take";
    }
    ASSERT_TRUE(WIFEXITED(status));
    EXPECT_EQ(WEXITSTATUS(status), 0);
    // The parent keeps its own reference and registration.
    EXPECT_EQ(monitor.references(), 1u);
    EXPECT_TRUE(monitor.installed());
}

TEST(DeviceFaultMonitorTest, AChildDoesNotInheritTheParentsNoticesAcrossARealFork) {
    FakeDriver driver;
    DeviceFaultMonitor monitor(driver.ops());
    ASSERT_EQ(monitor.acquire(), 0);
    monitor.report(notice(3, 44, 507018));
    ASSERT_EQ(monitor.sequence(), 1u);

    monitor.before_fork();
    const pid_t pid = fork();
    ASSERT_GE(pid, 0);
    if (pid == 0) {
        monitor.after_fork_in_child();
        DeviceFaultNotice out;
        const bool empty = monitor.sequence() == 0;
        const bool unreachable = monitor.read_notice(0, &out) != DeviceFaultNoticeRead::Ok;
        _exit((empty && unreachable) ? 0 : 1);
    }
    monitor.after_fork_in_parent();

    int status = 0;
    ASSERT_EQ(waitpid(pid, &status, 0), pid);
    ASSERT_TRUE(WIFEXITED(status));
    EXPECT_EQ(WEXITSTATUS(status), 0);
    // The parent's own notice is untouched.
    DeviceFaultNotice out;
    EXPECT_EQ(monitor.read_notice(0, &out), DeviceFaultNoticeRead::Ok);
    EXPECT_EQ(monitor.sequence(), 1u);
}
