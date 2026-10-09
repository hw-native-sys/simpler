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
 * The hand-off's own identity, and what the transport may conclude from it.
 *
 * Two separate guarantees are exercised here:
 *
 *  - the ready-queue descriptor carries the run that produced the buffer and
 *    the record count that producer committed, so a payload the host later
 *    cannot read is still attributable;
 *  - a consumer-index advance whose write failed is resolved against the
 *    device's own head rather than assumed, because "the write returned an
 *    error" and "the device did not see it" are different statements.
 */

#include <gtest/gtest.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <vector>
#include <optional>
#include <string>
#include <thread>
#include <utility>

#include "common/chip_swimlane_profiling.h"
#include "host/chip_swimlane_collector.h"
#include "host/profiler_base.h"

namespace fs = std::filesystem;

namespace {

// ---------------------------------------------------------------------------
// The wire layout the two sides agree on
// ---------------------------------------------------------------------------

TEST(SwimlaneHandoffIdentityTest, ReadyEntryCarriesIdentityWithoutGrowing) {
    // The device queue arrays are sized from this, so a change here is a change
    // to every ready queue's footprint on the device.
    EXPECT_EQ(sizeof(ReadyQueueEntry), 32u);
    EXPECT_EQ(alignof(ReadyQueueEntry), 32u);
    EXPECT_EQ(offsetof(ReadyQueueEntry, buffer_ptr), 8u);
    EXPECT_EQ(offsetof(ReadyQueueEntry, buffer_seq), 16u);
    EXPECT_EQ(offsetof(ReadyQueueEntry, record_count), 20u);
    EXPECT_EQ(offsetof(ReadyQueueEntry, run_epoch), 24u);
}

TEST(SwimlaneHandoffIdentityTest, OnlyADeclaredSchemaMakesTheIdentityFieldsReadable) {
    // Value-initializing the new type proves nothing about an old producer's
    // padding, so the guard is not "these bytes are zero" -- it is that the
    // producer declared which schema it writes. A region the host zeroed and an
    // old producer never stamped reads as `handoff_schema == 0`, which is not
    // the supported value, and the host then treats every hand-off from it as
    // carrying no identity rather than reading stale bytes as a run id.
    ChipSwimlaneDataHeader header{};
    EXPECT_NE(header.handoff_schema, kChipSwimlaneHandoffSchema);
    header.handoff_schema = kChipSwimlaneHandoffSchema;
    EXPECT_EQ(header.handoff_schema, kChipSwimlaneHandoffSchema);
}

// ---------------------------------------------------------------------------
// Acknowledgement: three outcomes, one of which may not be guessed
// ---------------------------------------------------------------------------

constexpr uint32_t kQueueSize = 8;

struct AckHeader {
    uint32_t queue_heads[4]{};
    uint32_t queue_tails[4]{};
};

// Only the traits the acknowledgement path names. The rest of the Module
// contract is unused here on purpose: this covers the consumer-index state
// machine, not buffer resolution.
struct AckFreeQueue {
    uint32_t head{0};
    uint32_t tail{0};
};

struct AckReadyEntry {
    uint64_t buffer_ptr{0};
};

struct AckBufferInfo {
    void *dev_buffer_ptr{nullptr};
};

struct AckModule {
    using DataHeader = AckHeader;
    using ReadyEntry = AckReadyEntry;
    using ReadyBufferInfo = AckBufferInfo;
    using FreeQueue = AckFreeQueue;
    static constexpr uint32_t kReadyQueueSize = kQueueSize;
    static constexpr const char *kSubsystemName = "AckTest";
};

/**
 * A manager whose device transfers can be made to fail, and whose device-side
 * head is a separate value from the host shadow.
 *
 * Keeping them separate is the whole point: the production bug this guards is
 * the host restoring its own shadow and calling that proof about the device.
 */
struct AckMgr {
    uint32_t device_head{0};
    bool write_fails{false};
    bool read_fails{false};
    uint32_t forced_device_head{UINT32_MAX};  // UINT32_MAX = report `device_head`

    int write_range_to_device(void *src, size_t bytes) {
        if (write_fails) return -1;
        std::memcpy(&device_head, src, bytes);
        return 0;
    }
    int read_range_from_device(void *dst, size_t bytes) {
        if (read_fails) return -1;
        const uint32_t value = forced_device_head == UINT32_MAX ? device_head : forced_device_head;
        std::memcpy(dst, &value, bytes);
        return 0;
    }
};

using Alg = profiling_common::ProfilerAlgorithms<AckModule>;
using profiling_common::AckOutcome;

TEST(SwimlaneAckOutcomeTest, SuccessfulWriteConsumesTheEntry) {
    AckHeader header;
    AckMgr mgr;
    header.queue_heads[0] = 3;
    mgr.device_head = 3;

    EXPECT_EQ(Alg::ack_aicpu_entry_checked(mgr, &header, 0), AckOutcome::kConsumed);
    EXPECT_EQ(header.queue_heads[0], 4u);
    EXPECT_EQ(mgr.device_head, 4u);
}

TEST(SwimlaneAckOutcomeTest, FailedWriteThatDidNotLandIsRetryable) {
    AckHeader header;
    AckMgr mgr;
    header.queue_heads[0] = 3;
    mgr.device_head = 3;  // the write never reached the device
    mgr.write_fails = true;

    EXPECT_EQ(Alg::ack_aicpu_entry_checked(mgr, &header, 0), AckOutcome::kNotConsumed);
    // Restored, so the same entry is served again rather than skipped.
    EXPECT_EQ(header.queue_heads[0], 3u);
}

TEST(SwimlaneAckOutcomeTest, FailedWriteThatDidLandIsNotRetried) {
    AckHeader header;
    AckMgr mgr;
    header.queue_heads[0] = 3;
    mgr.write_fails = true;
    // The transfer reported failure after the bytes were already in place.
    mgr.forced_device_head = 4;

    EXPECT_EQ(Alg::ack_aicpu_entry_checked(mgr, &header, 0), AckOutcome::kConsumed);
    // Retrying here would deliver this buffer twice; the shadow stays forward.
    EXPECT_EQ(header.queue_heads[0], 4u);
}

TEST(SwimlaneAckOutcomeTest, UnreadableHeadIsUnknownAndNotServedAgain) {
    AckHeader header;
    AckMgr mgr;
    header.queue_heads[0] = 3;
    mgr.write_fails = true;
    mgr.read_fails = true;

    EXPECT_EQ(Alg::ack_aicpu_entry_checked(mgr, &header, 0), AckOutcome::kUnknown);
    // Undecidable: the entry is given up rather than risked twice.
    EXPECT_EQ(header.queue_heads[0], 4u);
}

TEST(SwimlaneAckOutcomeTest, HeadNoSingleConsumerCouldProduceIsUnknown) {
    AckHeader header;
    AckMgr mgr;
    header.queue_heads[0] = 3;
    mgr.write_fails = true;
    mgr.forced_device_head = 7;  // neither 3 nor 4

    EXPECT_EQ(Alg::ack_aicpu_entry_checked(mgr, &header, 0), AckOutcome::kUnknown);
    EXPECT_EQ(header.queue_heads[0], 4u);
}

TEST(SwimlaneAckOutcomeTest, WrapAroundStillResolvesAgainstTheDeviceHead) {
    AckHeader header;
    AckMgr mgr;
    header.queue_heads[0] = kQueueSize - 1;
    mgr.write_fails = true;
    mgr.forced_device_head = 0;  // the wrapped successor, so the write landed

    EXPECT_EQ(Alg::ack_aicpu_entry_checked(mgr, &header, 0), AckOutcome::kConsumed);
    EXPECT_EQ(header.queue_heads[0], 0u);
}

// The boolean wrapper the existing call sites use must mean "the device will
// not show this entry again" and nothing weaker: an unconsumed acknowledgement
// reported as success is how a buffer gets delivered and freed twice.
TEST(SwimlaneAckOutcomeTest, BooleanWrapperIsTrueOnlyWhenConsumed) {
    AckHeader header;
    AckMgr mgr;
    header.queue_heads[0] = 1;
    mgr.device_head = 1;
    EXPECT_TRUE(Alg::ack_aicpu_entry(mgr, &header, 0));

    AckHeader not_consumed;
    AckMgr failing;
    not_consumed.queue_heads[0] = 1;
    failing.device_head = 1;
    failing.write_fails = true;
    EXPECT_FALSE(Alg::ack_aicpu_entry(failing, &not_consumed, 0));
}

// ---------------------------------------------------------------------------
// R1: the producer's declaration is read from the device, not the host shadow
// ---------------------------------------------------------------------------

/**
 * A manager with genuinely separate device and host regions, as a platform
 * without shared mapping has. The host side starts zeroed and only a narrow
 * read moves bytes across -- which is exactly the shape in which a declaration
 * written on the device never reaches a host that does not read it.
 */
struct SplitRegionMgr {
    ChipSwimlaneDataHeader device{};
    ChipSwimlaneDataHeader host{};
    bool read_fails{false};
    int reads{0};

    int read_range_from_device(void *host_field, size_t bytes) {
        if (read_fails) return -1;
        reads++;
        const auto base_h = reinterpret_cast<uintptr_t>(&host);
        const auto off = reinterpret_cast<uintptr_t>(host_field) - base_h;
        std::memcpy(host_field, reinterpret_cast<const char *>(&device) + off, bytes);
        return 0;
    }
};

TEST(SwimlaneSchemaTransportTest, DeclarationIsReadThroughTheProductionRefresh) {
    SplitRegionMgr mgr;
    mgr.device.handoff_schema = kChipSwimlaneHandoffSchema;  // the producer stamped it
    ASSERT_EQ(mgr.host.handoff_schema, 0u) << "a separate host region starts zeroed";

    // The production hook, not a direct assignment to the mirror.
    EXPECT_EQ(
        ChipSwimlaneModule::refresh_handoff_schema(mgr, &mgr.host), profiling_common::HandoffSchemaRead::kDeclared
    );
    EXPECT_EQ(mgr.host.handoff_schema, kChipSwimlaneHandoffSchema);
}

TEST(SwimlaneSchemaTransportTest, AFailedReadIsNotADeclaration) {
    SplitRegionMgr mgr;
    mgr.device.handoff_schema = kChipSwimlaneHandoffSchema;
    mgr.read_fails = true;

    EXPECT_EQ(ChipSwimlaneModule::refresh_handoff_schema(mgr, &mgr.host), profiling_common::HandoffSchemaRead::kFailed)
        << "a refused transfer is a fact about the link, never about what the producer declared";
    EXPECT_EQ(mgr.host.handoff_schema, 0u) << "a read that did not happen may not publish a declaration";
}

TEST(SwimlaneSchemaTransportTest, IdentitySurvivesResolveOnlyAfterTheRefresh) {
    SplitRegionMgr mgr;
    mgr.device.handoff_schema = kChipSwimlaneHandoffSchema;
    mgr.host.num_cores = 4;
    mgr.device.num_cores = 4;

    ChipSwimlaneAicpuTaskBuffer payload{};
    ReadyQueueEntry entry{};
    entry.core_index = 1;
    entry.kind = ChipSwimlaneBufferKind::AicpuTask;
    entry.buffer_ptr = reinterpret_cast<uint64_t>(&payload);
    entry.buffer_seq = 3;
    entry.record_count = 2;
    entry.run_epoch = 0x4242;

    // Untrusted is the state before the owning shard has confirmed a
    // declaration -- the A5 shape that silently discarded every epoch.
    auto before = ChipSwimlaneModule::resolve_entry(&mgr.host, &mgr.host, 0, entry, /*identity_trusted=*/false);
    ASSERT_TRUE(before.has_value());
    EXPECT_EQ(before->info.run_epoch, 0u);
    EXPECT_EQ(before->info.record_count, 0u);

    ASSERT_EQ(
        ChipSwimlaneModule::refresh_handoff_schema(mgr, &mgr.host), profiling_common::HandoffSchemaRead::kDeclared
    );
    auto after = ChipSwimlaneModule::resolve_entry(&mgr.host, &mgr.host, 0, entry, /*identity_trusted=*/true);
    ASSERT_TRUE(after.has_value());
    EXPECT_EQ(after->info.run_epoch, 0x4242u);
    EXPECT_EQ(after->info.record_count, 2u);
    EXPECT_EQ(after->info.buffer_seq, 3u);
}

TEST(SwimlaneSchemaTransportTest, AnOverCapacityCountRejectsTheEntry) {
    SplitRegionMgr mgr;
    mgr.device.handoff_schema = kChipSwimlaneHandoffSchema;
    mgr.host.num_cores = 4;
    ASSERT_EQ(
        ChipSwimlaneModule::refresh_handoff_schema(mgr, &mgr.host), profiling_common::HandoffSchemaRead::kDeclared
    );

    ChipSwimlaneAicpuTaskBuffer payload{};
    ReadyQueueEntry entry{};
    entry.core_index = 0;
    entry.kind = ChipSwimlaneBufferKind::AicpuTask;
    entry.buffer_ptr = reinterpret_cast<uint64_t>(&payload);
    entry.run_epoch = 7;
    entry.record_count = UINT32_MAX;  // no buffer holds this many

    EXPECT_FALSE(
        ChipSwimlaneModule::resolve_entry(&mgr.host, &mgr.host, 0, entry, /*identity_trusted=*/true).has_value()
    ) << "an unreadable payload must not be charged an invented number of records";
}

// ---------------------------------------------------------------------------
// R3: a cut armed after a queue stops never publishes success for it
// ---------------------------------------------------------------------------

/**
 * Drives the real cut methods on a real `ProfilerBase`, because the defect is
 * in `run_drain_boundary`'s capture: `cut_arm` zeroes `qstate`, so a cut armed
 * after a queue was quarantined used to capture a fresh target that its
 * consumed count already met -- head==tail on a queue nobody drains -- reach
 * stage one, and read back as a cut with no failed queues.
 */
struct CutHeader {
    // Four slots, which is what makes a legal multi-entry state expressible:
    // the device producer refuses a push when `(tail + 1) % size == head`
    // (`profiler_device_engine.h`), so a ring of N ever holds N-1 entries and a
    // ring of 2 holds exactly one. A case that needs a predecessor's entry and
    // a successor's beside it cannot be written at size 2 without constructing
    // a state no producer can reach.
    ReadyQueueEntry queues[PLATFORM_MAX_AICPU_THREADS][4]{};
    uint32_t queue_heads[PLATFORM_MAX_AICPU_THREADS]{};
    uint32_t queue_tails[PLATFORM_MAX_AICPU_THREADS]{};
    uint32_t handoff_schema{0};
};

struct CutFreeQueue {
    uint32_t head{0};
    uint32_t tail{0};
    uint64_t buffer_ptrs[1]{};
};

struct CutModule {
    using DataHeader = CutHeader;
    using ReadyEntry = ReadyQueueEntry;
    using ReadyBufferInfo = ::ReadyBufferInfo;
    using FreeQueue = CutFreeQueue;
    static constexpr int kBufferKinds = 1;
    static constexpr uint32_t kReadyQueueSize = 4;
    static constexpr uint32_t kSlotCount = 1;
    static constexpr int kMaxCollectorThreads = 1;
    static constexpr const char *kSubsystemName = "CutTest";
    static DataHeader *header_from_shm(void *shm) { return static_cast<DataHeader *>(shm); }
    static int batch_size(int) { return 1; }

    static std::optional<profiling_common::EntrySite<CutModule>>
    resolve_entry(void *, DataHeader *, int, const ReadyEntry &) {
        return std::nullopt;
    }
    template <typename Cb>
    static void for_each_instance(void *, DataHeader *, Cb &&) {}
};

/** Exposes the protected cut surface the drain owner uses. */
class CutProbe : public profiling_common::ProfilerBase<CutProbe, CutModule> {
public:
    using Base = profiling_common::ProfilerBase<CutProbe, CutModule>;
    static constexpr const char *kSubsystemName = "CutTest";
    void on_buffer_collected(const ::ReadyBufferInfo &, int) {}

    using Base::cut_arm;
    using Base::cut_failed_queues;
    using Base::cut_stage1_done;
    using Base::note_buffer_retired;
    using Base::quarantine_queue;
    using Base::run_drain_boundary;
    using Base::set_aicpu_thread_num;
    using Base::set_run_counters;
};

TEST(SwimlaneCutQuarantineTest, ACutArmedAfterAQueueStopsReportsItFailed) {
    CutProbe probe;
    CutHeader header{};
    probe.set_aicpu_thread_num(1);
    probe.set_run_counters(true);

    // The queue looks drained -- head==tail -- which is exactly the state an
    // unknown acknowledgement that actually landed leaves behind.
    header.queue_heads[0] = 1;
    header.queue_tails[0] = 1;

    probe.quarantine_queue(0);

    // Armed AFTER the quarantine, so its qstate starts at zero.
    uint64_t request = 0;
    const int slot = probe.cut_arm(&request);
    ASSERT_GE(slot, 0);

    probe.run_drain_boundary(&header, /*queue_start=*/0, /*queue_stride=*/1);

    int failed = 0;
    const bool known = probe.cut_failed_queues(slot, request, &failed);
    EXPECT_TRUE(known);
    EXPECT_EQ(failed, 1) << "a stopped queue must never be captured as reached";
}

TEST(SwimlaneCutQuarantineTest, AQueueStillServedIsCapturedNormally) {
    CutProbe probe;
    CutHeader header{};
    probe.set_aicpu_thread_num(1);
    probe.set_run_counters(true);
    header.queue_heads[0] = 0;
    header.queue_tails[0] = 0;

    uint64_t request = 0;
    const int slot = probe.cut_arm(&request);
    ASSERT_GE(slot, 0);
    probe.run_drain_boundary(&header, 0, 1);

    int failed = 0;
    EXPECT_TRUE(probe.cut_failed_queues(slot, request, &failed));
    EXPECT_EQ(failed, 0);
}

TEST(SwimlaneCutProgressTest, SuccessorTrafficCannotSatisfyThePredecessorsCut) {
    // Two runs collecting at once share every ready queue, so a predecessor's
    // cut is captured on queues its successor keeps pushing into. Two things
    // have to hold and are separate: the target is `consumed + outstanding`
    // taken once at capture and does not grow with later pushes, and one
    // queue reaching its target does not answer for another queue's.
    CutProbe probe;
    CutHeader header{};
    probe.set_aicpu_thread_num(2);
    probe.set_run_counters(true);

    // Each queue holds what the predecessor left outstanding. Both states are
    // reachable: a ring of four admits up to three entries.
    header.queue_heads[0] = 0;
    header.queue_tails[0] = 2;  // two of the predecessor's, undrained
    header.queue_heads[1] = 0;
    header.queue_tails[1] = 1;  // one of the predecessor's, undrained

    uint64_t request = 0;
    const int slot = probe.cut_arm(&request);
    ASSERT_GE(slot, 0);

    probe.run_drain_boundary(&header, /*queue_start=*/0, /*queue_stride=*/1);
    EXPECT_FALSE(probe.cut_stage1_done(slot)) << "nothing captured has been consumed yet";

    // The successor pushes a third entry onto queue 0, legally. The capture
    // already happened, so this must not move queue 0's target.
    header.queue_tails[0] = 3;
    probe.run_drain_boundary(&header, /*queue_start=*/0, /*queue_stride=*/1);
    EXPECT_FALSE(probe.cut_stage1_done(slot)) << "a successor's push satisfied a cut on its own";

    // Queue 0 drains exactly the two the predecessor owed. If the target had
    // grown with the successor's push it would now stand at three and this
    // queue would still be short -- which the final assertion would catch.
    probe.note_buffer_retired(0);
    probe.note_buffer_retired(0);
    probe.run_drain_boundary(&header, /*queue_start=*/0, /*queue_stride=*/1);
    EXPECT_FALSE(probe.cut_stage1_done(slot))
        << "queue 0's progress answered for queue 1's predecessor, which has not retired";

    // Only queue 1's own entry leaving completes the cut.
    probe.note_buffer_retired(1);
    probe.run_drain_boundary(&header, /*queue_start=*/0, /*queue_stride=*/1);
    EXPECT_TRUE(probe.cut_stage1_done(slot))
        << "every queue reached the target it was captured with and the cut did not notice";

    int failed = 0;
    EXPECT_TRUE(probe.cut_failed_queues(slot, request, &failed));
    EXPECT_EQ(failed, 0) << "no queue was stopped, so none may be reported failed";
}

TEST(SwimlaneCutProgressTest, AnUndecidableQueueIsReportedWithoutHoldingTheOthers) {
    // An acknowledgement nobody can decide stops one queue for good. The cut
    // must still complete on the queues that are still served -- a cut no
    // queue can finish never returns its slot -- and the stopped queue must be
    // reported rather than counted as reached.
    //
    // Scope: this is the cut's own state machine. It is not evidence about the
    // ACK retirement that decides a queue is undecidable in the first place,
    // nor about a retained bucket being handed back; `quarantine_queue` is
    // called here directly.
    CutProbe probe;
    CutHeader header{};
    probe.set_aicpu_thread_num(2);
    probe.set_run_counters(true);
    header.queue_heads[0] = 0;
    header.queue_tails[0] = 1;  // still served, one outstanding
    header.queue_heads[1] = 1;
    header.queue_tails[1] = 1;  // reads drained, but nobody is draining it

    probe.quarantine_queue(1);

    uint64_t request = 0;
    const int slot = probe.cut_arm(&request);
    ASSERT_GE(slot, 0);

    probe.run_drain_boundary(&header, /*queue_start=*/0, /*queue_stride=*/1);
    EXPECT_FALSE(probe.cut_stage1_done(slot)) << "queue 0 still owes the entry it was captured with";

    probe.note_buffer_retired(0);
    probe.run_drain_boundary(&header, /*queue_start=*/0, /*queue_stride=*/1);
    EXPECT_TRUE(probe.cut_stage1_done(slot)) << "a stopped queue held a cut the served queues had reached";

    int failed = 0;
    EXPECT_TRUE(probe.cut_failed_queues(slot, request, &failed));
    EXPECT_EQ(failed, 1) << "the stopped queue completed the cut instead of being reported";
}

// ---------------------------------------------------------------------------
// R2: a late charge never lands on the run that reused the bucket
// ---------------------------------------------------------------------------

void *loss_alloc(size_t bytes) { return std::malloc(bytes); }
int loss_free(void *p) {
    std::free(p);
    return 0;
}
std::thread loss_thread(std::function<void()> fn) { return std::thread(std::move(fn)); }

profiling_common::RetiredHandoff<ChipSwimlaneModule> handoff(uint64_t epoch, uint32_t records) {
    profiling_common::RetiredHandoff<ChipSwimlaneModule> r;
    r.identified = true;
    r.info.run_epoch = epoch;
    r.info.record_count = records;
    return r;
}

/** A retaining collector, brought up the way the runner brings one up. */
struct LossFixture {
    ChipSwimlaneCollector collector;
    fs::path root;
    std::string dir;

    explicit LossFixture(const char *name) :
        // Named with this process's pid, and removed by this fixture alone.
        // The case name on its own is one absolute path shared by every
        // process on the host: this file builds PER_ARCH PER_RUNTIME, the host
        // lane runs `ctest -j4`, and `/tmp` is shared between users. A root
        // another process owns cannot be reserved into, and one this fixture
        // never removed grows a `swimlane-K` subdirectory per run forever.
        root(fs::temp_directory_path() / ("simpler-ut-" + std::string(name) + "-" + std::to_string(::getpid()))),
        dir(root.string()) {
        std::error_code ec;
        fs::remove_all(root, ec);
        collector.configure_retained_runs(true, simpler::dfx::runs::kDefaultBudgetBytes);
        EXPECT_EQ(collector.initialize(1, 1, 0, ChipSwimlaneLevel::TASK_TIMING, loss_alloc, nullptr, loss_free), 0);
        collector.start(loss_thread);
    }
    ~LossFixture() {
        collector.finish_retained_runs();
        collector.stop();
        collector.finalize(nullptr, loss_free);
        // After the joins above, so nothing is still publishing into it.
        std::error_code ec;
        fs::remove_all(root, ec);
    }
};

TEST(SwimlaneLossGenerationTest, AnOpenRunIsChargedItsOwnLoss) {
    LossFixture fx("loss-open");
    ASSERT_TRUE(fx.collector.run_begin(11, fx.dir, ChipSwimlaneLevel::TASK_TIMING, nullptr, 0, false));

    fx.collector.on_handoff_retired(handoff(11, 5));

    const auto loss = fx.collector.transport_loss_for_test(11);
    EXPECT_EQ(loss.first, 1u);
    EXPECT_EQ(loss.second, 5u);
    EXPECT_EQ(fx.collector.unattributable_handoffs_for_test(), 0u);
}

TEST(SwimlaneLossGenerationTest, APredecessorsLateChargeNeverReachesTheSuccessor) {
    LossFixture fx("loss-reuse");
    ASSERT_TRUE(fx.collector.run_begin(21, fx.dir, ChipSwimlaneLevel::TASK_TIMING, nullptr, 0, false));
    // Withdraw run 21 without launching it, which is the production way a slot
    // goes back; then reuse it for 22.
    ASSERT_TRUE(fx.collector.abandon_run(21));
    ASSERT_TRUE(fx.collector.run_begin(22, fx.dir, ChipSwimlaneLevel::TASK_TIMING, nullptr, 0, false));

    fx.collector.on_handoff_retired(handoff(22, 3));  // the successor's own loss
    fx.collector.on_handoff_retired(handoff(21, 9));  // the predecessor's, arriving late

    const auto successor = fx.collector.transport_loss_for_test(22);
    EXPECT_EQ(successor.first, 1u) << "the late charge must not be added to the successor";
    EXPECT_EQ(successor.second, 3u) << "and must not be subtracted from it either";
    EXPECT_EQ(fx.collector.unattributable_handoffs_for_test(), 1u);
}

TEST(SwimlaneLossGenerationTest, AnUnknownEpochIsChargedToNobody) {
    LossFixture fx("loss-unknown");
    ASSERT_TRUE(fx.collector.run_begin(31, fx.dir, ChipSwimlaneLevel::TASK_TIMING, nullptr, 0, false));

    fx.collector.on_handoff_retired(handoff(999, 4));

    EXPECT_EQ(fx.collector.transport_loss_for_test(31).first, 0u);
    EXPECT_EQ(fx.collector.unattributable_handoffs_for_test(), 1u);
}

// ---------------------------------------------------------------------------
// The interleavings a label and a later re-check cannot survive
// ---------------------------------------------------------------------------

/**
 * Releases a parked charge and joins its thread however the test leaves --
 * including through a failed `ASSERT_*`, which returns from the body with the
 * charger still blocked. Without this a failing assertion would destroy a
 * joinable `std::thread` and take the process with it.
 */
struct ParkedCharge {
    std::atomic<bool> parked{false};
    std::atomic<bool> released{false};
    std::atomic<bool> park_next{true};
    std::thread charger;

    /**
     * Park the first charge to arrive, and only that one. The hook is never
     * reassigned while a thread is inside it: a later charge takes the same
     * callable and walks straight through, so nothing destroys a callable its
     * caller is still executing.
     */
    void arm(ChipSwimlaneCollector &collector) {
        collector.set_pre_charge_hook_for_test([this] {
            if (!park_next.exchange(false)) return;
            parked.store(true);
            while (!released.load())
                std::this_thread::yield();
        });
    }

    void start(ChipSwimlaneCollector &collector, uint64_t epoch, uint32_t records) {
        charger = std::thread([&collector, epoch, records] {
            collector.on_handoff_retired(handoff(epoch, records));
        });
        while (!parked.load())
            std::this_thread::yield();
    }

    void release() {
        released.store(true);
        if (charger.joinable()) charger.join();
    }

    ~ParkedCharge() { release(); }
};

TEST(SwimlaneLossExclusionTest, AChargeInFlightWhenTheBucketIsReusedNeverReachesTheSuccessor) {
    LossFixture fx("loss-reuse-race");
    ASSERT_TRUE(fx.collector.run_begin(41, fx.dir, ChipSwimlaneLevel::TASK_TIMING, nullptr, 0, false));

    // The charge is held in the window the review names: past its decision to
    // charge 41, before it can take the bucket's accounting lock.
    ParkedCharge held;
    held.arm(fx.collector);
    held.start(fx.collector, 41, 9);

    // 41 goes away and 42 takes its slot while that charge is in flight.
    ASSERT_TRUE(fx.collector.abandon_run(41));
    ASSERT_TRUE(fx.collector.run_begin(42, fx.dir, ChipSwimlaneLevel::TASK_TIMING, nullptr, 0, false));
    held.release();

    const auto successor = fx.collector.transport_loss_for_test(42);
    EXPECT_EQ(successor.first, 0u) << "a predecessor's in-flight charge must not land on the successor";
    EXPECT_EQ(successor.second, 0u);
    EXPECT_EQ(fx.collector.unattributable_handoffs_for_test(), 1u);
}

TEST(SwimlaneLossExclusionTest, AParkedChargeDoesNotDisturbTheSuccessorsOwn) {
    LossFixture fx("loss-reuse-mixed");
    ASSERT_TRUE(fx.collector.run_begin(61, fx.dir, ChipSwimlaneLevel::TASK_TIMING, nullptr, 0, false));

    ParkedCharge held;
    held.arm(fx.collector);
    held.start(fx.collector, 61, 4);

    ASSERT_TRUE(fx.collector.abandon_run(61));
    ASSERT_TRUE(fx.collector.run_begin(62, fx.dir, ChipSwimlaneLevel::TASK_TIMING, nullptr, 0, false));
    // Runs through the same hook the parked thread is still inside.
    fx.collector.on_handoff_retired(handoff(62, 3));
    held.release();

    const auto successor = fx.collector.transport_loss_for_test(62);
    EXPECT_EQ(successor.first, 1u) << "the successor keeps exactly its own charge";
    EXPECT_EQ(successor.second, 3u);
    EXPECT_EQ(fx.collector.unattributable_handoffs_for_test(), 1u);
}

TEST(SwimlaneLossExclusionTest, AChargeInFlightWhenTheSlotIsReleasedIsRefused) {
    LossFixture fx("loss-release-race");
    ASSERT_TRUE(fx.collector.run_begin(71, fx.dir, ChipSwimlaneLevel::TASK_TIMING, nullptr, 0, false));
    fx.collector.on_handoff_retired(handoff(71, 2));

    ParkedCharge held;
    held.arm(fx.collector);
    held.start(fx.collector, 71, 7);

    // The slot goes back while the charge is parked, and nothing claims it
    // afterwards: the charge resumes against a genuinely free bucket rather
    // than against a successor that happens to have overwritten the identity.
    ASSERT_TRUE(fx.collector.abandon_run(71));
    held.release();

    const auto after = fx.collector.transport_loss_for_test(71);
    EXPECT_EQ(after.first, 0u) << "a released slot holds no run's accounting, not even the one it last held";
    EXPECT_EQ(after.second, 0u);
    EXPECT_EQ(fx.collector.unattributable_handoffs_for_test(), 1u)
        << "a charge that missed the release is reported, never silently dropped";
}

// ---------------------------------------------------------------------------
// A released slot holds no run, so nothing may be charged to it
// ---------------------------------------------------------------------------

TEST(SwimlaneLossAuthorityTest, AWithdrawnRunIsNotChargeableBeforeItsSuccessorExists) {
    LossFixture fx("loss-withdrawn");
    ASSERT_TRUE(fx.collector.run_begin(81, fx.dir, ChipSwimlaneLevel::TASK_TIMING, nullptr, 0, false));
    ASSERT_TRUE(fx.collector.abandon_run(81));

    // No successor has claimed the slot yet, which is what used to let the
    // stale identity still match and the charge land on an orphan bucket.
    fx.collector.on_handoff_retired(handoff(81, 9));

    EXPECT_EQ(fx.collector.transport_loss_for_test(81).first, 0u);
    EXPECT_EQ(fx.collector.transport_loss_for_test(81).second, 0u);
    EXPECT_EQ(fx.collector.unattributable_handoffs_for_test(), 1u)
        << "a descriptor naming a run this collector no longer holds is unattributed, not charged";

    // And the successor that later takes the slot starts from zero.
    ASSERT_TRUE(fx.collector.run_begin(82, fx.dir, ChipSwimlaneLevel::TASK_TIMING, nullptr, 0, false));
    EXPECT_EQ(fx.collector.transport_loss_for_test(82).first, 0u);
}

// ---------------------------------------------------------------------------
// The drain settles the producer's declaration before it consumes an entry
// ---------------------------------------------------------------------------

/**
 * Drives `ProfilerBase::mgmt_drain_loop` itself -- the function both the sweep
 * and the final pass live in -- over a split device/host region, so what is
 * under test is the decision the drain actually makes about an entry, not a
 * helper's return value.
 *
 * The buffer each entry names is deliberately unmappable, which is the
 * attribution case that matters: the descriptor validated and the payload did
 * not arrive, so the loss has an owner and a count if, and only if, the drain
 * settled the declaration before it resolved the entry.
 */
struct DrainModule {
    using DataHeader = CutHeader;
    using ReadyEntry = ReadyQueueEntry;
    using ReadyBufferInfo = ::ReadyBufferInfo;
    using FreeQueue = CutFreeQueue;
    static constexpr int kBufferKinds = 1;
    static constexpr uint32_t kReadyQueueSize = 2;
    static constexpr uint32_t kSlotCount = 1;
    static constexpr int kMaxCollectorThreads = 1;
    static constexpr const char *kSubsystemName = "DrainTest";
    static DataHeader *header_from_shm(void *shm) { return static_cast<DataHeader *>(shm); }
    static int batch_size(int) { return 1; }

    // The production shape of the hook: one narrow device read, and a verdict
    // that separates "the link refused" from "the producer declared nothing".
    template <typename Mgr>
    static profiling_common::HandoffSchemaRead refresh_handoff_schema(Mgr &mgr, DataHeader *header) {
        if (mgr.read_range_from_device(&header->handoff_schema, sizeof(header->handoff_schema)) != 0) {
            return profiling_common::HandoffSchemaRead::kFailed;
        }
        return header->handoff_schema == kChipSwimlaneHandoffSchema ? profiling_common::HandoffSchemaRead::kDeclared :
                                                                      profiling_common::HandoffSchemaRead::kUndeclared;
    }

    static CutFreeQueue free_queue;

    static std::optional<profiling_common::EntrySite<DrainModule>>
    resolve_entry(void *, DataHeader *, int, const ReadyEntry &entry, bool identity_trusted) {
        profiling_common::EntrySite<DrainModule> site{};
        site.kind = 0;
        site.free_queue = &free_queue;
        site.buffer_size = 32;
        site.info.index = entry.core_index;
        site.info.dev_buffer_ptr = reinterpret_cast<void *>(entry.buffer_ptr);
        site.info.buffer_seq = entry.buffer_seq;
        site.info.record_count = identity_trusted ? entry.record_count : 0;
        site.info.run_epoch = identity_trusted ? entry.run_epoch : 0;
        return site;
    }

    template <typename Cb>
    static void for_each_instance(void *, DataHeader *, Cb &&) {}
};

CutFreeQueue DrainModule::free_queue{};

class DrainProbe : public profiling_common::ProfilerBase<DrainProbe, DrainModule> {
public:
    using Base = profiling_common::ProfilerBase<DrainProbe, DrainModule>;
    static constexpr const char *kSubsystemName = "DrainTest";
    void on_buffer_collected(const ::ReadyBufferInfo &, int) {}

    std::vector<profiling_common::RetiredHandoff<DrainModule>> retired;
    void on_handoff_retired(const profiling_common::RetiredHandoff<DrainModule> &r) { retired.push_back(r); }

    using Base::handoff_trust;
    using Base::kMaxSchemaReadAttempts;
    using Base::mgmt_drain_loop;
    using Base::set_aicpu_thread_num;
    using Base::set_memory_context;
};

/**
 * A device region and a host shadow with no mapping between them, so every
 * word the host believes it had to read. `schema_refusals` makes the narrow
 * declaration read -- and only that read -- fail.
 */
struct DrainBed {
    CutHeader device{};
    CutHeader host{};
    int schema_refusals{0};
    int schema_reads{0};

    void bind(DrainProbe &probe) {
        probe.set_memory_context(
            loss_alloc, nullptr, loss_free,
            [](void *dst, const void *src, size_t n) {
                std::memcpy(dst, src, n);
                return 0;
            },
            [this](void *dst, const void *src, size_t n) {
                const auto off = reinterpret_cast<uintptr_t>(src) - reinterpret_cast<uintptr_t>(&device);
                if (off == offsetof(CutHeader, handoff_schema)) {
                    schema_reads++;
                    if (schema_refusals > 0) {
                        schema_refusals--;
                        return -1;
                    }
                }
                std::memcpy(dst, src, n);
                return 0;
            },
            &device, &host, sizeof(CutHeader), 0
        );
    }

    /** Publish one entry on `q`, the way a producer does: payload, then tail. */
    void publish(int q, uint64_t epoch, uint32_t records) {
        ReadyQueueEntry entry{};
        entry.core_index = 0;
        entry.kind = ChipSwimlaneBufferKind::AicpuTask;
        entry.buffer_ptr = 0x5000;  // never mapped on this host
        entry.buffer_seq = 77;
        entry.record_count = records;
        entry.run_epoch = epoch;
        device.queues[q][device.queue_tails[q]] = entry;
        device.queue_tails[q] = (device.queue_tails[q] + 1) % 2;
    }
};

TEST(SwimlaneDrainIdentityTest, AShardThatNeverOwnedQueueZeroStillKeepsTheIdentity) {
    DrainProbe probe;
    DrainBed bed;
    probe.set_aicpu_thread_num(2);
    bed.bind(probe);
    bed.device.handoff_schema = kChipSwimlaneHandoffSchema;
    bed.publish(/*q=*/1, /*epoch=*/0x900d, /*records=*/6);

    // Shard 1 only. Shard 0 -- the former sole owner of the declaration --
    // never runs, which is the interleaving that used to erase identity.
    probe.mgmt_drain_loop(/*queue_start=*/1, /*queue_stride=*/2);

    ASSERT_EQ(probe.retired.size(), 1u);
    EXPECT_TRUE(probe.retired[0].identified);
    EXPECT_EQ(probe.retired[0].info.run_epoch, 0x900du) << "the producer named this run; the drain must keep it";
    EXPECT_EQ(probe.retired[0].info.record_count, 6u);
    EXPECT_EQ(bed.schema_reads, 1);
}

TEST(SwimlaneDrainIdentityTest, NoDeclarationIsSampledBeforeAnEntryExists) {
    DrainProbe probe;
    DrainBed bed;
    probe.set_aicpu_thread_num(1);
    bed.bind(probe);

    // The producer has not initialized yet: no stamp, nothing published.
    probe.mgmt_drain_loop(0, 1);
    EXPECT_EQ(bed.schema_reads, 0) << "a sample taken before the producer exists would answer about nobody";
    EXPECT_EQ(probe.handoff_trust(), profiling_common::HandoffTrust::kUnsettled);

    // Now the producer comes up in the order the device enforces: the stamp
    // first, then the entry.
    bed.device.handoff_schema = kChipSwimlaneHandoffSchema;
    bed.publish(0, /*epoch=*/0xbeef, /*records=*/2);
    probe.mgmt_drain_loop(0, 1);

    EXPECT_EQ(bed.schema_reads, 1);
    ASSERT_EQ(probe.retired.size(), 1u);
    EXPECT_EQ(probe.retired[0].info.run_epoch, 0xbeefu);
    EXPECT_EQ(probe.retired[0].info.record_count, 2u);
}

TEST(SwimlaneDrainIdentityTest, ALegacyProducerIsSettledOnceAndCarriesNoIdentity) {
    DrainProbe probe;
    DrainBed bed;
    probe.set_aicpu_thread_num(1);
    bed.bind(probe);
    bed.device.handoff_schema = 0;  // a build that predates the identity fields
    bed.publish(0, /*epoch=*/5, /*records=*/3);

    probe.mgmt_drain_loop(0, 1);

    ASSERT_EQ(probe.retired.size(), 1u);
    EXPECT_TRUE(probe.retired[0].identified);
    EXPECT_EQ(probe.retired[0].info.run_epoch, 0u) << "an undeclared producer's bytes are not an identity";
    EXPECT_EQ(probe.retired[0].info.record_count, 0u);
    EXPECT_EQ(probe.handoff_trust(), profiling_common::HandoffTrust::kUndeclared);

    bed.publish(0, 5, 3);
    probe.mgmt_drain_loop(0, 1);
    EXPECT_EQ(bed.schema_reads, 1) << "a settled declaration is never read again";
}

TEST(SwimlaneDrainIdentityTest, ARefusedDeclarationHoldsTheEntryAndThenGivesUp) {
    DrainProbe probe;
    DrainBed bed;
    probe.set_aicpu_thread_num(1);
    bed.bind(probe);
    bed.device.handoff_schema = kChipSwimlaneHandoffSchema;
    bed.publish(0, /*epoch=*/9, /*records=*/1);

    // One fewer refusal than the bound, so the last attempt succeeds and the
    // entry is consumed with the identity its producer did publish.
    bed.schema_refusals = static_cast<int>(DrainProbe::kMaxSchemaReadAttempts) - 1;
    probe.mgmt_drain_loop(0, 1);
    ASSERT_EQ(probe.retired.size(), 1u);
    EXPECT_EQ(probe.retired[0].info.run_epoch, 9u) << "a link that answers late still answers";
    EXPECT_EQ(probe.handoff_trust(), profiling_common::HandoffTrust::kDeclared);

    // A link that never answers must not stall the drain for good.
    DrainProbe stuck;
    DrainBed dead;
    stuck.set_aicpu_thread_num(1);
    dead.bind(stuck);
    dead.device.handoff_schema = kChipSwimlaneHandoffSchema;
    dead.publish(0, 11, 4);
    dead.schema_refusals = 1000;
    stuck.mgmt_drain_loop(0, 1);
    EXPECT_EQ(dead.schema_reads, static_cast<int>(DrainProbe::kMaxSchemaReadAttempts))
        << "the refused read is bounded, not retried forever";
    ASSERT_EQ(stuck.retired.size(), 1u);
    EXPECT_EQ(stuck.retired[0].info.run_epoch, 0u);
    EXPECT_EQ(stuck.handoff_trust(), profiling_common::HandoffTrust::kUnreadable)
        << "a link that could not answer is not the same state as a producer that declared nothing";
}

TEST(SwimlaneDrainIdentityTest, TrustDoesNotCarryIntoANewRegion) {
    DrainProbe probe;
    DrainBed bed;
    probe.set_aicpu_thread_num(1);
    bed.bind(probe);
    bed.device.handoff_schema = kChipSwimlaneHandoffSchema;
    bed.publish(0, 3, 1);
    probe.mgmt_drain_loop(0, 1);
    ASSERT_EQ(probe.handoff_trust(), profiling_common::HandoffTrust::kDeclared);

    DrainBed second;
    second.bind(probe);
    EXPECT_EQ(probe.handoff_trust(), profiling_common::HandoffTrust::kUnsettled)
        << "a new region carries a new producer's declaration, which has not been read";
}

}  // namespace
