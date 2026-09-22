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

#include <atomic>
#include <thread>
#include <vector>

#include "host/device_fault_monitor.h"
#include "host/device_health_state.h"

namespace {

constexpr uint32_t kFaultCode = 507018;
constexpr uint32_t kLaterCode = 507015;

}  // namespace

// ===== A fault naming this device quarantines it =====

TEST(DeviceHealthState, StartsHealthy) {
    const DeviceHealthState health;
    EXPECT_FALSE(health.suspect());
    EXPECT_EQ(health.first_error_code(), 0u);
    EXPECT_EQ(health.generation(), 0u);
    EXPECT_EQ(health.own_faults_total(), 0u);
}

TEST(DeviceHealthState, AFaultNamingThisDeviceMakesItSuspect) {
    DeviceHealthState health;
    EXPECT_TRUE(health.note_own_device_fault(kFaultCode));
    EXPECT_TRUE(health.suspect());
    EXPECT_EQ(health.first_error_code(), kFaultCode);
    EXPECT_EQ(health.own_faults_total(), 1u);
}

// The return distinguishes the first fault from its cascade, so a caller can
// recover once per fault instead of once per notice — a fault typically reports
// on both streams (measured in #2303: 2 notices for one aicore_hang).
TEST(DeviceHealthState, OnlyTheFirstFaultOfAGenerationReportsAsNew) {
    DeviceHealthState health;
    EXPECT_TRUE(health.note_own_device_fault(kFaultCode));
    EXPECT_FALSE(health.note_own_device_fault(kLaterCode));
    EXPECT_FALSE(health.note_own_device_fault(kLaterCode));
    EXPECT_EQ(health.own_faults_total(), 3u);
}

// A cascade's later codes are consequences of its first, so the first is the one
// worth keeping. Same rule as the per-run terminal record's.
TEST(DeviceHealthState, TheFirstErrorCodeSurvivesLaterFaults) {
    DeviceHealthState health;
    health.note_own_device_fault(kFaultCode);
    health.note_own_device_fault(kLaterCode);
    EXPECT_EQ(health.first_error_code(), kFaultCode);
}

// A code of 0 would be indistinguishable from "healthy", so it is stored as a
// sentinel rather than dropped — the fault is real even when its code is not.
TEST(DeviceHealthState, AFaultWithNoCodeIsStillSuspect) {
    DeviceHealthState health;
    EXPECT_TRUE(health.note_own_device_fault(0));
    EXPECT_TRUE(health.suspect());
    EXPECT_EQ(health.first_error_code(), DeviceHealthState::kUnspecifiedFault);
}

// ===== A fault this runner cannot attribute does not =====

// A notice naming another device, or a stream no run of this runner submits on,
// identifies nothing this runner can act on. Measured on a2a3: a 507018 on
// stream_id=45/46 with task_id=10 arrives while every run on the card succeeds,
// and quarantining on it refuses the next healthy run.
TEST(DeviceHealthState, AnUnattributableFaultLeavesThisDeviceHealthy) {
    DeviceHealthState health;
    health.note_unattributed_fault();
    health.note_unattributed_fault();
    EXPECT_FALSE(health.suspect());
    EXPECT_EQ(health.first_error_code(), 0u);
    EXPECT_EQ(health.unattributed_faults_total(), 2u);
    EXPECT_EQ(health.own_faults_total(), 0u);
}

// ===== An undelivered notice is counted, not acted on =====

// An undelivered notice names nothing, so it cannot be told from one that would
// have named another device or an unrelated stream. Counting it keeps the
// channel's lossiness visible; quarantining on it would refuse healthy work on
// evidence that identifies no fault.
TEST(DeviceHealthState, ALostNoticeIsCountedAndDoesNotQuarantine) {
    DeviceHealthState health;
    health.note_undelivered_notices(/*lost=*/1, /*dropped=*/0);
    EXPECT_FALSE(health.suspect());
    EXPECT_EQ(health.first_error_code(), 0u);
    EXPECT_EQ(health.undelivered_notices_total(), 1u);
}

TEST(DeviceHealthState, ADroppedNoticeIsCountedAndDoesNotQuarantine) {
    DeviceHealthState health;
    health.note_undelivered_notices(/*lost=*/0, /*dropped=*/3);
    EXPECT_FALSE(health.suspect());
    EXPECT_EQ(health.undelivered_notices_total(), 3u);
}

TEST(DeviceHealthState, NoUndeliveredNoticesChangesNothing) {
    DeviceHealthState health;
    health.note_undelivered_notices(0, 0);
    EXPECT_EQ(health.undelivered_notices_total(), 0u);
    EXPECT_FALSE(health.suspect());
}

// The channel being lossy does not disturb a fault it did deliver.
TEST(DeviceHealthState, AnUndeliveredNoticeDoesNotDisturbARealFault) {
    DeviceHealthState health;
    health.note_own_device_fault(kFaultCode);
    health.note_undelivered_notices(1, 0);
    EXPECT_TRUE(health.suspect());
    EXPECT_EQ(health.first_error_code(), kFaultCode);
}

// ===== The reset fence is what lets a card come back =====

TEST(DeviceHealthState, AConfirmedResetRetiresTheSuspectGeneration) {
    DeviceHealthState health;
    health.note_own_device_fault(kFaultCode);
    EXPECT_TRUE(health.retire_generation());
    EXPECT_FALSE(health.suspect());
    EXPECT_EQ(health.first_error_code(), 0u);
    EXPECT_EQ(health.generation(), 1u);
}

// The return distinguishes a recovery from a reset that had nothing to recover,
// so a caller can say which happened instead of logging both the same way.
TEST(DeviceHealthState, RetiringAHealthyGenerationClearsNothing) {
    DeviceHealthState health;
    EXPECT_FALSE(health.retire_generation());
    EXPECT_EQ(health.generation(), 1u);
}

// The whole point of the fence: without it one recovered fault would quarantine
// the card for the life of the process.
TEST(DeviceHealthState, ADeviceIsUsableAgainAfterTheResetThatRecoveredIt) {
    DeviceHealthState health;
    health.note_own_device_fault(kFaultCode);
    ASSERT_TRUE(health.suspect());
    health.retire_generation();
    EXPECT_FALSE(health.suspect());
}

// And the fence does not make the device permanently trusted: the generation
// after a reset is judged on its own notices.
TEST(DeviceHealthState, AFaultAfterTheResetMakesTheNewGenerationSuspect) {
    DeviceHealthState health;
    health.note_own_device_fault(kFaultCode);
    health.retire_generation();
    EXPECT_TRUE(health.note_own_device_fault(kLaterCode));
    EXPECT_TRUE(health.suspect());
    EXPECT_EQ(health.first_error_code(), kLaterCode);
    EXPECT_EQ(health.generation(), 1u);
}

TEST(DeviceHealthState, EachResetAdvancesTheGeneration) {
    DeviceHealthState health;
    for (uint64_t expected = 1; expected <= 4; ++expected) {
        health.retire_generation();
        EXPECT_EQ(health.generation(), expected);
    }
}

// ===== Concurrency =====

// The notices are consumed on whichever host thread finalizes a run, while the
// quarantine is read from admission on another; the counts must not tear and
// exactly one caller may see the first-fault transition.
TEST(DeviceHealthState, ConcurrentFaultsCountOnceAndElectOneFirstReporter) {
    constexpr int kThreads = 8;
    constexpr int kPerThread = 200;
    DeviceHealthState health;
    std::atomic<int> first_reports{0};

    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&health, &first_reports] {
            for (int i = 0; i < kPerThread; ++i) {
                if (health.note_own_device_fault(kFaultCode)) first_reports.fetch_add(1);
            }
        });
    }
    for (std::thread &thread : threads)
        thread.join();

    EXPECT_EQ(health.own_faults_total(), static_cast<uint64_t>(kThreads) * kPerThread);
    EXPECT_TRUE(health.suspect());
    EXPECT_EQ(first_reports.load(), 1) << "more than one caller was told it saw the first fault";
}

// ===== Attributing a notice to a stream =====
//
// The filter itself, which the state tests above do not reach. Its inputs are
// driver stream ids captured when a run launched — not handles read later, which
// at teardown may already have been invalidated by a force reset.

using Attribution = RunStreamIdentities::Attribution;

TEST(RunStreamIdentities, WithNoHistoryEveryNoticeIsNotMine) {
    const RunStreamIdentities ids;
    EXPECT_EQ(ids.attribute(44), Attribution::NotMine);
    EXPECT_TRUE(ids.complete());
    EXPECT_EQ(ids.size(), 0u);
}

TEST(RunStreamIdentities, ARecordedStreamIsMine) {
    RunStreamIdentities ids;
    ids.note(43);
    ids.note(44);
    EXPECT_EQ(ids.attribute(43), Attribution::Mine);
    EXPECT_EQ(ids.attribute(44), Attribution::Mine);
    EXPECT_EQ(ids.size(), 2u);
}

// The measured false positives: a 507018 on stream 45/46 while every run
// succeeds. Those ids were never launched on, so they are not this runner's.
TEST(RunStreamIdentities, AnUnrecordedStreamIsNotMine) {
    RunStreamIdentities ids;
    ids.note(43);
    ids.note(44);
    EXPECT_EQ(ids.attribute(45), Attribution::NotMine);
    EXPECT_EQ(ids.attribute(46), Attribution::NotMine);
}

TEST(RunStreamIdentities, ARepeatedIdIsStoredOnce) {
    RunStreamIdentities ids;
    ids.note(43);
    ids.note(43);
    EXPECT_EQ(ids.size(), 1u);
    EXPECT_EQ(ids.attribute(43), Attribution::Mine);
}

// A stream the runner uses whose id could not be read. Skipping it silently
// would leave the history claiming to be whole while missing exactly the entry a
// later notice on that stream would carry — and that notice would then read as
// another runner's.
TEST(RunStreamIdentities, AStreamWhoseIdCouldNotBeReadMakesTheHistoryIncomplete) {
    RunStreamIdentities ids;
    ids.note(43);
    ids.note_unidentified_stream();

    EXPECT_FALSE(ids.complete());
    EXPECT_EQ(ids.attribute(43), Attribution::Mine) << "a known id stays decidable";
    EXPECT_EQ(ids.attribute(44), Attribution::Undecided) << "an unknown id must not read as another runner's";
}

// A failed query yields a negative id, so that is the shape `note` has to treat
// as unidentified rather than ignore.
TEST(RunStreamIdentities, ANegativeIdIsTreatedAsUnidentifiedNotIgnored) {
    RunStreamIdentities ids;
    ids.note(43);
    ids.note(-1);
    EXPECT_FALSE(ids.complete());
    EXPECT_EQ(ids.attribute(99), Attribution::Undecided);
}

TEST(RunStreamIdentities, AResetRestoresACompleteHistoryAfterAFailedQuery) {
    RunStreamIdentities ids;
    ids.note_unidentified_stream();
    ASSERT_FALSE(ids.complete());
    ids.retire_generation();
    EXPECT_TRUE(ids.complete());
    EXPECT_EQ(ids.attribute(44), Attribution::NotMine);
}

// The reason ids are kept rather than read from the live handles: publishing
// AICore code marks a2a3's stream stale and the next launch replaces it, so a
// notice from the run that used the old stream arrives naming an id no longer
// live. Forgetting it would read that run's own fault as someone else's.
TEST(RunStreamIdentities, AReplacedStreamStaysAttributable) {
    RunStreamIdentities ids;
    ids.note(43);  // the run's AICore stream
    ids.note(44);
    ids.note(47);  // its replacement after a code publication
    EXPECT_EQ(ids.attribute(43), Attribution::Mine) << "a late notice from the replaced stream lost its owner";
    EXPECT_EQ(ids.attribute(47), Attribution::Mine);
}

// Capacity is finite, so history can be incomplete — and that is reported rather
// than silently turning into "not mine", because an evicted id is exactly what a
// stale notice would carry.
TEST(RunStreamIdentities, AnOverflowingHistoryReportsAttributionAsUndecided) {
    RunStreamIdentities ids;
    for (int32_t id = 0; id < static_cast<int32_t>(RunStreamIdentities::kCapacity); ++id)
        ids.note(id);
    ASSERT_TRUE(ids.complete());

    ids.note(9000);
    EXPECT_FALSE(ids.complete());
    // Ids it still holds stay decidable.
    EXPECT_EQ(ids.attribute(0), Attribution::Mine);
    // Anything else can no longer be ruled out.
    EXPECT_EQ(ids.attribute(9999), Attribution::Undecided);
}

TEST(RunStreamIdentities, AConfirmedResetForgetsTheGenerationsStreams) {
    RunStreamIdentities ids;
    ids.note(43);
    ids.retire_generation();
    EXPECT_EQ(ids.attribute(43), Attribution::NotMine);
    EXPECT_EQ(ids.size(), 0u);
    EXPECT_TRUE(ids.complete());
}

TEST(RunStreamIdentities, AResetAlsoRestoresACompleteHistory) {
    RunStreamIdentities ids;
    for (int32_t id = 0; id <= static_cast<int32_t>(RunStreamIdentities::kCapacity); ++id)
        ids.note(id);
    ASSERT_FALSE(ids.complete());
    ids.retire_generation();
    EXPECT_TRUE(ids.complete());
    EXPECT_EQ(ids.attribute(0), Attribution::NotMine);
}

// The production contract is **one writer, many readers**: `note` runs on the
// launch path under the execution claim, `attribute` on whichever thread
// finalizes. So this drives exactly that shape, with an explicit publication
// point — the reader only asserts an id *after* the writer has published that it
// wrote it — because without one "not visible yet" and "lost" are the same
// observation and the test would assert a race rather than a contract.
//
// It is deliberately not a four-writer test. `note`'s count-load → slot-store →
// count-store is not a multi-writer insert algorithm, nothing in the product
// inserts concurrently, and asserting that it were would argue for a lock on the
// launch path that no caller needs.
TEST(RunStreamIdentities, AConcurrentReaderSeesEveryPublishedId) {
    constexpr int32_t kIds = 16;
    RunStreamIdentities ids;
    std::atomic<int32_t> published{-1};
    std::atomic<bool> reader_failed{false};

    std::thread reader([&] {
        for (int32_t id = 0; id < kIds; ++id) {
            // Wait for the writer to publish that this id is in, then require it.
            while (published.load(std::memory_order_acquire) < id) {}
            if (ids.attribute(static_cast<uint32_t>(id)) != Attribution::Mine) {
                reader_failed.store(true, std::memory_order_release);
                return;
            }
            // Everything published so far must still be there — a later insert
            // must not displace an earlier one.
            for (int32_t seen = 0; seen <= id; ++seen) {
                if (ids.attribute(static_cast<uint32_t>(seen)) != Attribution::Mine) {
                    reader_failed.store(true, std::memory_order_release);
                    return;
                }
            }
        }
    });

    for (int32_t id = 0; id < kIds; ++id) {
        ids.note(id);
        published.store(id, std::memory_order_release);
    }
    reader.join();

    EXPECT_FALSE(reader_failed.load(std::memory_order_acquire))
        << "a concurrent reader missed an id the writer had already published";
    EXPECT_EQ(ids.size(), static_cast<size_t>(kIds));
    EXPECT_TRUE(ids.complete());
}

// ===== A confirmed reset retires the local evidence even with no monitor =====
//
// The wiring, not the state classes: `retire_after_confirmed_device_reset` is
// what `DeviceRunnerBase::retire_device_generation_after_confirmed_reset` calls,
// and the bug it exists to prevent is an early return on a null monitor skipping
// the local half. Driven with the real `DeviceFaultMonitor` and cursor — no fake
// — because that header carries its own injected ops for exactly this reason.

namespace {

/** A monitor whose install/uninstall are counted rather than performed. */
struct CountingMonitorOps {
    int install_rc{0};
    int installs{0};
    DeviceFaultMonitor::Ops make() {
        return DeviceFaultMonitor::Ops{
            [this] {
                ++installs;
                return install_rc;
            },
            [] {
                return 0;
            },
            [] {
                return static_cast<long>(1234);
            },
        };
    }
};

DeviceFaultNotice notice_on(uint32_t stream_id) {
    DeviceFaultNotice notice;
    notice.device_id = 0;
    notice.stream_id = stream_id;
    notice.error_code = kFaultCode;
    return notice;
}

}  // namespace

// A runner whose `acquire` failed still runs work and still records stream ids
// (`ensure_device_initialized` does not fail it), so a confirmed reset must
// invalidate that evidence whether or not a callback was ever installed.
TEST(ConfirmedDeviceReset, RetiresLocalEvidenceWithNoMonitorHeld) {
    DeviceHealthState health;
    RunStreamIdentities ids;
    DeviceFaultNoticeCursor notices;

    ids.note(43);
    ids.note(44);
    health.note_own_device_fault(kFaultCode);
    ASSERT_TRUE(health.suspect());

    const DeviceGenerationRetirement retirement =
        retire_after_confirmed_device_reset(health, ids, static_cast<DeviceFaultMonitor *>(nullptr), notices);

    EXPECT_FALSE(retirement.monitor_reinstalled) << "there was no monitor to re-install";
    EXPECT_EQ(retirement.monitor_reinstall_rc, 0) << "no monitor is not a failure";
    EXPECT_TRUE(retirement.cleared_suspicion);

    // The point of the test: the local half happened anyway.
    EXPECT_FALSE(health.suspect());
    EXPECT_EQ(health.generation(), 1u);
    EXPECT_EQ(ids.size(), 0u);
    EXPECT_EQ(ids.attribute(43), Attribution::NotMine)
        << "a stream id from the retired generation was still attributed to the new one";
}

// The counters are runner-lifetime on purpose, so the same reset must leave them
// alone. This is the distinction the naming now carries.
TEST(ConfirmedDeviceReset, LeavesRunnerLifetimeTotalsAlone) {
    DeviceHealthState health;
    RunStreamIdentities ids;
    DeviceFaultNoticeCursor notices;

    health.note_own_device_fault(kFaultCode);
    health.note_unattributed_fault();
    health.note_undelivered_notices(2, 1);

    (void)retire_after_confirmed_device_reset(health, ids, static_cast<DeviceFaultMonitor *>(nullptr), notices);

    EXPECT_FALSE(health.suspect()) << "generation-scoped state retires";
    EXPECT_EQ(health.first_error_code(), 0u);
    EXPECT_EQ(health.own_faults_total(), 1u) << "runner-lifetime totals do not";
    EXPECT_EQ(health.unattributed_faults_total(), 1u);
    EXPECT_EQ(health.undelivered_notices_total(), 3u);
}

// With a monitor held, the conditional half runs too: the callback is remade and
// the cursor skips the notices that described the retired generation.
TEST(ConfirmedDeviceReset, AlsoFencesTheMonitorWhenOneIsHeld) {
    CountingMonitorOps ops;
    DeviceFaultMonitor monitor(ops.make());
    ASSERT_EQ(monitor.acquire(), 0);

    DeviceHealthState health;
    RunStreamIdentities ids;
    DeviceFaultNoticeCursor notices;
    ids.note(43);
    monitor.report(notice_on(43));
    monitor.report(notice_on(44));
    ASSERT_EQ(monitor.sequence(), 2u);

    const DeviceGenerationRetirement retirement = retire_after_confirmed_device_reset(health, ids, &monitor, notices);

    EXPECT_TRUE(retirement.monitor_reinstalled);
    EXPECT_EQ(retirement.monitor_reinstall_rc, 0);
    EXPECT_EQ(notices.position(), 2u) << "notices from the retired generation are not the new one's to read";
    EXPECT_EQ(ids.size(), 0u);
    EXPECT_EQ(health.generation(), 1u);
}

// A re-install that failed is still reported, and does not stop the retirement:
// the device was reset regardless of what the callback slot did afterwards.
TEST(ConfirmedDeviceReset, ReportsAFailedReinstallWithoutBlockingRetirement) {
    CountingMonitorOps ops;
    DeviceFaultMonitor monitor(ops.make());
    ASSERT_EQ(monitor.acquire(), 0);
    ops.install_rc = -7;

    DeviceHealthState health;
    RunStreamIdentities ids;
    DeviceFaultNoticeCursor notices;
    ids.note(44);
    health.note_own_device_fault(kFaultCode);

    const DeviceGenerationRetirement retirement = retire_after_confirmed_device_reset(health, ids, &monitor, notices);

    EXPECT_EQ(retirement.monitor_reinstall_rc, -7) << "the rc has to reach the caller";
    EXPECT_FALSE(health.suspect());
    EXPECT_EQ(ids.size(), 0u);
}

// ===== Admission under device health =====
//
// `device_admits_new_run` is the predicate the shared c_api asks through
// `DeviceRunnerBase::accepts_new_run`. What reaches it is already covered above
// — which notice kinds make a device suspect, and what `attribute` answers for
// each stream id. These cases pin the consequence: a notice the runner matched
// to one of its own run streams refuses the *next* run, and nothing else does.

TEST(AdmissionUnderDeviceHealth, AHealthyRunnerAdmits) {
    const DeviceHealthState health;
    EXPECT_TRUE(device_admits_new_run(/*arch_quarantined=*/false, health));
}

TEST(AdmissionUnderDeviceHealth, AMatchedNoticeRefusesTheNextRun) {
    DeviceHealthState health;
    RunStreamIdentities ids;
    ids.note(61);
    ASSERT_EQ(ids.attribute(61), Attribution::Mine);

    health.note_own_device_fault(kFaultCode);

    EXPECT_FALSE(device_admits_new_run(/*arch_quarantined=*/false, health))
        << "a matched notice has to refuse admission on its own, with no arch quarantine set";
    EXPECT_EQ(health.first_error_code(), kFaultCode);
}

// A missing id is exactly what a stale notice carries, so an incomplete history
// weakens only the *negative* answer. A positive match is still a match, and it
// still refuses.
TEST(AdmissionUnderDeviceHealth, AMatchedNoticeStillRefusesWhenTheHistoryIsIncomplete) {
    DeviceHealthState health;
    RunStreamIdentities ids;
    ids.note(61);
    ids.note_unidentified_stream();
    ASSERT_FALSE(ids.complete());
    ASSERT_EQ(ids.attribute(61), Attribution::Mine);

    health.note_own_device_fault(kFaultCode);

    EXPECT_FALSE(device_admits_new_run(/*arch_quarantined=*/false, health));
}

// Both of the consumer's non-matching outcomes reach the same record call, so
// both leave admission open: a stream no run used while the history is whole,
// and any stream once an id is missing. A notice naming another device reaches
// it too, without consulting the history at all.
TEST(AdmissionUnderDeviceHealth, AnUnattributedNoticeAdmits) {
    DeviceHealthState health;
    RunStreamIdentities whole;
    whole.note(61);
    ASSERT_EQ(whole.attribute(45), Attribution::NotMine);

    RunStreamIdentities partial;
    partial.note_unidentified_stream();
    ASSERT_EQ(partial.attribute(45), Attribution::Undecided);

    health.note_unattributed_fault();
    health.note_unattributed_fault();

    EXPECT_TRUE(device_admits_new_run(/*arch_quarantined=*/false, health));
    EXPECT_EQ(health.unattributed_faults_total(), 2u) << "recorded, and still admitting";
    EXPECT_EQ(health.first_error_code(), 0u);
}

TEST(AdmissionUnderDeviceHealth, LostAndDroppedNoticesAdmit) {
    DeviceHealthState health;
    health.note_undelivered_notices(/*lost=*/3, /*dropped=*/2);

    EXPECT_TRUE(device_admits_new_run(/*arch_quarantined=*/false, health))
        << "a notice that named nothing cannot name a resource to refuse for";
    EXPECT_EQ(health.undelivered_notices_total(), 5u);
}

TEST(AdmissionUnderDeviceHealth, AConfirmedResetRestoresAdmission) {
    CountingMonitorOps ops;
    DeviceFaultMonitor monitor(ops.make());
    ASSERT_EQ(monitor.acquire(), 0);

    DeviceHealthState health;
    RunStreamIdentities ids;
    DeviceFaultNoticeCursor notices;
    ids.note(61);
    health.note_own_device_fault(kFaultCode);
    ASSERT_FALSE(device_admits_new_run(/*arch_quarantined=*/false, health));

    const DeviceGenerationRetirement retirement = retire_after_confirmed_device_reset(health, ids, &monitor, notices);

    EXPECT_TRUE(retirement.cleared_suspicion);
    EXPECT_TRUE(device_admits_new_run(/*arch_quarantined=*/false, health))
        << "the reset that recovered the card has to give it back";
}

// Only a confirmed reset retires a generation, so a reset that was not confirmed
// reaches no retirement and the refusal stands.
TEST(AdmissionUnderDeviceHealth, AResetThatWasNotConfirmedLeavesAdmissionRefused) {
    DeviceHealthState health;
    health.note_own_device_fault(kFaultCode);

    EXPECT_FALSE(device_admits_new_run(/*arch_quarantined=*/false, health));
    EXPECT_EQ(health.generation(), 0u) << "no generation was retired";
}

// The two refusals are independent, and only one of them is this channel's.
TEST(AdmissionUnderDeviceHealth, TheArchQuarantineRefusesOnItsOwnAndIsNotRetiredHere) {
    DeviceHealthState health;
    EXPECT_FALSE(device_admits_new_run(/*arch_quarantined=*/true, health));

    health.retire_generation();
    EXPECT_FALSE(device_admits_new_run(/*arch_quarantined=*/true, health))
        << "retiring the health generation cannot clear the arch's own flag";
    EXPECT_TRUE(device_admits_new_run(/*arch_quarantined=*/false, health));
}
