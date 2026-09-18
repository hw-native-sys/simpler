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
 * @file device_fault_monitor.h
 * @brief Process-level owner of the device fault-notification callback.
 *
 * The driver's notification callback (`aclrtSetExceptionInfoCallback` onboard)
 * is **one slot per process**, and the last registration wins silently. So no
 * runner may own that slot: it is owned here, refcounted, and installed on
 * first use.
 *
 * **This class is only half the ownership story.** One instance per process is
 * what makes the refcount mean anything, and that is a property of where the
 * instance and the driver's function pointer live — not of this header. See
 * the singleton accessor's contract at the bottom.
 *
 * **What a notification can and cannot say.** It carries the device, the
 * faulting stream, a task id and the error code — and on the measured silicon
 * the task id is `0` for every error class, while both runs of a pipelined pair
 * share one stream pair. So a notification places a fault on a *device*, never
 * on a run. This class records notices and counts them; it attributes nothing,
 * and a caller that wants per-run blame must get it from the run's own
 * device-published record instead.
 *
 * ## The three lifetime hazards
 *
 *   - **A notification can arrive during or after retirement.** The driver
 *     calls on its own thread; nothing orders that call against an uninstall.
 *     So `report` is safe with no reference held, and the storage it writes
 *     must outlive every owner of it.
 *   - **`fork`.** A level-2 `Worker` forks the child that owns the device.
 *     A child inherits this object's bytes — including `mutex_`, possibly
 *     **locked by a thread that does not exist in the child**. A pid stamp
 *     cannot fix that: the stamp is read under the lock the child can never
 *     take. So the fork boundary is handled explicitly, by the three
 *     `*_fork` entry points below, which the singleton's owner registers with
 *     `pthread_atfork`. The pid stamp stays as a second line of defence for a
 *     child created by a path that runs no atfork handler.
 *   - **Device reset.** Whether a registration survives
 *     `aclrtResetDeviceForce` is not measured, so
 *     `reinstall_after_device_reset` re-registers rather than assuming
 *     persistence. Registering twice is harmless (the slot takes the last
 *     writer), which is what makes re-registering the safe direction.
 *
 * ## The handler takes no lock
 *
 * `acquire`/`release`/`reinstall` serialize on a mutex because they call into
 * the driver; `report` touches only atomics. A driver thread therefore never
 * blocks behind a drain, and a drain never blocks behind a driver thread.
 *
 * ## One word per slot, so a loser writes nothing
 *
 * A notice's fields are plain atomics, so a reader needs to know that no
 * writer touched them mid-read — and a *flag* raised before the writes cannot
 * give it that. Two independent words fail in both directions: a writer
 * preempted between checking the flag and writing the fields still clobbers a
 * newer notice, and one preempted before storing its publication marker can
 * store a stale one afterwards, which parks every later read on `Pending`
 * forever.
 *
 * So a slot has **one** state word, and taking the slot is the same decision
 * as completing it. A writer compare-exchanges from its predecessor's complete
 * state into its own writing state: winning is exclusive, and **losing means
 * touching nothing at all**. Only the winner can move the word to complete,
 * and no other generation's exchange accepts that value, so a completed notice
 * cannot be reverted. A loser records that its generation never arrived, which
 * is what lets a reader distinguish "still coming" from "never coming" instead
 * of waiting on it.
 */

#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <utility>

/** One notification, as the driver delivered it. */
struct DeviceFaultNotice {
    uint32_t device_id{0};
    uint32_t stream_id{0};
    uint32_t task_id{0};
    uint32_t error_code{0};
    uint32_t thread_id{0};
};

/** What reading one notice index found. */
enum class DeviceFaultNoticeRead : uint8_t {
    /** Read, complete, and this index's own. */
    Ok = 0,
    /**
     * Reserved by a report that has not finished. The index is **not
     * consumed**: a caller must leave its cursor here and read again, or the
     * notice is lost the moment the report completes.
     */
    Pending = 1,
    /** Overwritten by a later notice. Gone for good; account for it as lost. */
    Lost = 2,
};

/** Where `report_with_pause` stops. A test seam; see that overload. */
enum class ReportPause : uint8_t {
    /** Between taking the slot and writing the first field. */
    AfterClaim = 0,
    /** Between the last field and publishing it. */
    BeforePublish = 1,
};

class DeviceFaultMonitor {
public:
    /**
     * Platform hooks, injected so the owner's lifecycle is exercisable without
     * a device. `install` registers the process callback and `uninstall`
     * retires it; both return the platform's rc. `current_pid` exists for the
     * same reason — a test has to be able to drive a fork boundary.
     */
    struct Ops {
        std::function<int()> install;
        std::function<int()> uninstall;
        std::function<long()> current_pid;
    };

    explicit DeviceFaultMonitor(Ops ops) :
        ops_(std::move(ops)) {
        owner_pid_ = current_pid();
    }

    DeviceFaultMonitor(const DeviceFaultMonitor &) = delete;
    DeviceFaultMonitor &operator=(const DeviceFaultMonitor &) = delete;

    /**
     * Take a reference, installing the callback on the first one held in this
     * process. Returns the install rc; a failed install takes no reference, so
     * a later caller retries rather than inheriting a registration that was
     * never made.
     */
    int acquire() {
        std::scoped_lock lock(mutex_);
        adopt_process_locked();
        if (references_ > 0 && installed_) {
            ++references_;
            return 0;
        }
        const int rc = ops_.install ? ops_.install() : 0;
        if (rc != 0) return rc;
        installed_ = true;
        ++references_;
        return 0;
    }

    /**
     * Drop a reference, retiring the callback with the last one. A retire
     * failure still drops the reference: the slot's contents are the driver's
     * to decide, and holding a reference for a registration we can no longer
     * retire would pin the owner for the process lifetime.
     */
    void release() {
        std::scoped_lock lock(mutex_);
        adopt_process_locked();
        if (references_ == 0) return;
        if (--references_ > 0) return;
        if (installed_ && ops_.uninstall) {
            last_uninstall_rc_ = ops_.uninstall();
        }
        installed_ = false;
    }

    /**
     * Re-register after a confirmed device reset, which may or may not have
     * cleared the slot. With references outstanding this re-registers now;
     * with none it clears the installed state so the next `acquire` installs
     * instead of taking a reference to a registration that may be gone.
     */
    int reinstall_after_device_reset() {
        std::scoped_lock lock(mutex_);
        adopt_process_locked();
        if (references_ == 0) {
            installed_ = false;
            return 0;
        }
        const int rc = ops_.install ? ops_.install() : 0;
        installed_ = rc == 0;
        return rc;
    }

    /**
     * Record a notification. Called on a driver thread, at any time, including
     * while no reference is held — so it neither locks, allocates, nor
     * consults the installed state.
     *
     * A report that loses its slot to a newer one is dropped and counted
     * rather than written partially: the ring keeps recent notices, and a
     * notice too old to keep must be visibly absent, never a mixture.
     */
    void report(const DeviceFaultNotice &notice) { report_with_pause(notice, ReportPause::BeforePublish, nullptr); }

    /**
     * `report`, with a pause at one of the two points where a second reporter
     * can interleave.
     *
     * A test seam, and the only way to hold a slot at either point
     * deterministically — both windows are a few instructions wide otherwise.
     * Production calls `report`, which passes `nullptr`, so the handler path
     * pays one null compare.
     */
    void report_with_pause(const DeviceFaultNotice &notice, ReportPause stage, const std::function<void()> *pause) {
        const uint64_t generation = sequence_.fetch_add(1, std::memory_order_acq_rel) + 1;
        Slot &slot = ring_[(generation - 1) % kNoticeRing];

        // Take the slot, or write nothing at all. The expected value is this
        // slot's previous lap, complete: a writer whose predecessor is still
        // mid-write, or whose slot a later lap already took, loses the
        // exchange and leaves every field alone. That is the whole exclusion —
        // a flag checked before the writes cannot provide it, because losing
        // the race after the check still leaves the writes to come.
        uint64_t expected = generation > kNoticeRing ? complete_state(generation - kNoticeRing) : 0;
        if (!slot.state.compare_exchange_strong(
                expected, writing_state(generation), std::memory_order_acq_rel, std::memory_order_acquire
            )) {
            abandon(slot, generation);
            return;
        }

        if (stage == ReportPause::AfterClaim && pause != nullptr && *pause) (*pause)();

        slot.device_id.store(notice.device_id, std::memory_order_relaxed);
        slot.stream_id.store(notice.stream_id, std::memory_order_relaxed);
        slot.task_id.store(notice.task_id, std::memory_order_relaxed);
        slot.error_code.store(notice.error_code, std::memory_order_relaxed);
        slot.thread_id.store(notice.thread_id, std::memory_order_relaxed);

        if (stage == ReportPause::BeforePublish && pause != nullptr && *pause) (*pause)();

        // A plain store, and it cannot be reverted: this writer holds the slot
        // in its writing state, and every other generation's exchange expects
        // a different value, so nobody else can move it. An older writer
        // storing a stale marker here is what would stall a consumer for good.
        slot.state.store(complete_state(generation), std::memory_order_release);
    }

    /** Notification slots reserved in this process, ever. */
    uint64_t sequence() const { return sequence_.load(std::memory_order_acquire); }

    /** Reports this process abandoned rather than write partially. */
    uint64_t dropped() const { return dropped_.load(std::memory_order_acquire); }

    /** How many of the most recent notices the ring can hold. */
    static constexpr uint64_t retained_notices() { return kNoticeRing; }

    /**
     * Read the notice at `index` in report order.
     *
     * Three outcomes, and a caller must not collapse them: only `Ok` consumes
     * the index, `Pending` must be re-read, and `Lost` is gone. Treating
     * `Pending` as consumed loses a notice that was merely mid-flight.
     */
    DeviceFaultNoticeRead read_notice(uint64_t index, DeviceFaultNotice *out) const {
        if (out == nullptr) return DeviceFaultNoticeRead::Lost;
        const uint64_t generation = index + 1;
        const Slot &slot = ring_[index % kNoticeRing];

        const uint64_t state = slot.state.load(std::memory_order_acquire);
        if (state == writing_state(generation)) return DeviceFaultNoticeRead::Pending;
        if (state == complete_state(generation)) {
            out->device_id = slot.device_id.load(std::memory_order_relaxed);
            out->stream_id = slot.stream_id.load(std::memory_order_relaxed);
            out->task_id = slot.task_id.load(std::memory_order_relaxed);
            out->error_code = slot.error_code.load(std::memory_order_relaxed);
            out->thread_id = slot.thread_id.load(std::memory_order_relaxed);
            // Unchanged across the read, so no later lap began overwriting
            // these fields while they were being copied out.
            if (slot.state.load(std::memory_order_acquire) == complete_state(generation)) {
                return DeviceFaultNoticeRead::Ok;
            }
            return DeviceFaultNoticeRead::Lost;
        }
        // A later lap owns the slot, so this generation's notice is gone.
        if ((state >> 1) > generation) return DeviceFaultNoticeRead::Lost;
        // Its writer lost the slot and wrote nothing. Without this the
        // consumer would sit on a reserved index that will never be filled.
        if (slot.abandoned.load(std::memory_order_acquire) >= generation) return DeviceFaultNoticeRead::Lost;
        // Reserved, and its writer has not taken the slot yet.
        return DeviceFaultNoticeRead::Pending;
    }

    /**
     * References and installed state held by **this** process.
     *
     * A child reads zero and false before it has acquired anything, even
     * though the fork copied the parent's values: reporting the parent's
     * registration here would describe a slot this process does not own. The
     * answer is computed rather than adopted, so an observer stays const.
     */
    uint32_t references() const {
        std::scoped_lock lock(mutex_);
        return owned_by_this_process_locked() ? references_ : 0;
    }

    bool installed() const {
        std::scoped_lock lock(mutex_);
        return owned_by_this_process_locked() && installed_;
    }

    /** The rc of this process's last retire attempt; 0 if it has made none. */
    int last_uninstall_rc() const {
        std::scoped_lock lock(mutex_);
        return owned_by_this_process_locked() ? last_uninstall_rc_ : 0;
    }

    // ===== The fork boundary =====
    //
    // Registered with `pthread_atfork` by whoever owns the process's instance,
    // and callable directly so a test can drive the boundary without forking.
    // `mutex_` is held from before_fork() until one of the two after_fork
    // entries runs, which is what stops a child from inheriting a lock whose
    // owner does not exist there.

    void before_fork() { mutex_.lock(); }

    void after_fork_in_parent() { mutex_.unlock(); }

    /**
     * Claim the inherited state for the child. The child is single-threaded
     * here, and the registration and notices it inherited belong to the
     * parent's driver context and device work.
     */
    void after_fork_in_child() {
        owner_pid_ = -1;  // force adopt_process_locked() to reset, whatever the pid is
        adopt_process_locked();
        mutex_.unlock();
    }

private:
    // Deep enough that a burst of notifications arriving while the host is
    // between reads is retained, small enough to stay a fixed-size global: the
    // handler runs on a driver thread and may not allocate, so this cannot be
    // a growable container.
    static constexpr uint64_t kNoticeRing = 16;

    struct Slot {
        // One word, so taking the slot and completing it are the same
        // decision. `(generation << 1) | 1` means that generation owns the
        // slot and is writing; `generation << 1` means its notice is
        // complete; 0 means never written. A writer only ever moves this from
        // its predecessor's complete state, which is what makes losing the
        // exchange mean "write nothing".
        std::atomic<uint64_t> state{0};
        // The highest generation that wanted this slot and lost it. A reader
        // needs this to tell "reserved, still coming" from "reserved, never
        // arriving" — without it a lost report parks the consumer forever.
        std::atomic<uint64_t> abandoned{0};
        std::atomic<uint32_t> device_id{0};
        std::atomic<uint32_t> stream_id{0};
        std::atomic<uint32_t> task_id{0};
        std::atomic<uint32_t> error_code{0};
        std::atomic<uint32_t> thread_id{0};
    };

    static constexpr uint64_t writing_state(uint64_t generation) { return (generation << 1) | 1u; }
    static constexpr uint64_t complete_state(uint64_t generation) { return generation << 1; }

    /** Record that `generation` never got its slot, and count it. */
    void abandon(Slot &slot, uint64_t generation) {
        uint64_t seen = slot.abandoned.load(std::memory_order_acquire);
        while (seen < generation && !slot.abandoned.compare_exchange_weak(
                                        seen, generation, std::memory_order_acq_rel, std::memory_order_acquire
                                    )) {}
        dropped_.fetch_add(1, std::memory_order_acq_rel);
    }

    long current_pid() const { return ops_.current_pid ? ops_.current_pid() : 0; }

    /**
     * Claim this process's ownership, discarding anything a parent left behind.
     *
     * Everything below the mutex is per process image: a registration made
     * before a `fork` belongs to the parent's driver context, and notices
     * counted before it belong to the parent's device work. A child that
     * inherited either would otherwise take a reference to a registration it
     * does not hold, or report its parent's fault as its own.
     */
    void adopt_process_locked() {
        if (owned_by_this_process_locked()) return;
        owner_pid_ = current_pid();
        references_ = 0;
        installed_ = false;
        last_uninstall_rc_ = 0;
        // Both the counters and the ring are cleared. Resetting the count
        // alone would leave the parent's slots claimed and published, and
        // `read_notice` takes an index from its caller rather than deriving
        // one from the count — so an inherited notice would still read back as
        // this process's.
        //
        // Unsynchronized against `report` by design, and safe where it runs: a
        // freshly forked child has one thread, so no driver callback can be in
        // flight the first time the child claims ownership.
        for (Slot &slot : ring_) {
            slot.state.store(0, std::memory_order_relaxed);
            slot.abandoned.store(0, std::memory_order_relaxed);
        }
        dropped_.store(0, std::memory_order_relaxed);
        sequence_.store(0, std::memory_order_release);
    }

    bool owned_by_this_process_locked() const { return owner_pid_ == current_pid(); }

    mutable std::mutex mutex_;
    Ops ops_;
    uint32_t references_{0};
    bool installed_{false};
    int last_uninstall_rc_{0};
    long owner_pid_{-1};

    std::atomic<uint64_t> sequence_{0};
    std::atomic<uint64_t> dropped_{0};
    Slot ring_[kNoticeRing]{};
};

/**
 * One reader's position in the monitor's notice stream.
 *
 * The advance rule is the whole content of this type, and it is the one thing
 * a naive loop gets wrong: **a pending index is not consumed.** The sequence
 * counts slots that have been *reserved*, so a reader can see a count that
 * includes a report still in flight. Skipping that index and moving the cursor
 * past it loses the notice permanently — it becomes readable a moment later
 * and nobody ever looks again. An overwritten index, by contrast, is gone and
 * must be counted rather than waited for.
 *
 * Lives here rather than in the runner so the rule is exercisable without a
 * device.
 */
class DeviceFaultNoticeCursor {
public:
    struct Progress {
        uint64_t delivered{0};      // notices handed to the sink
        uint64_t lost{0};           // indices the ring overwrote before this read
        uint64_t newly_dropped{0};  // reports the writer abandoned since the last read
        bool pending{false};        // stopped on a report still in flight
    };

    /**
     * Hand every readable notice since the last call to `sink`, in report
     * order, and stop at the first one still in flight.
     */
    template <typename Sink>
    Progress consume(const DeviceFaultMonitor &monitor, Sink &&sink) {
        Progress progress;
        const uint64_t reserved = monitor.sequence();
        uint64_t index = position_;
        for (; index < reserved; ++index) {
            DeviceFaultNotice notice;
            const DeviceFaultNoticeRead read = monitor.read_notice(index, &notice);
            if (read == DeviceFaultNoticeRead::Pending) {
                progress.pending = true;
                break;
            }
            if (read == DeviceFaultNoticeRead::Lost) {
                ++progress.lost;
                continue;
            }
            sink(notice);
            ++progress.delivered;
        }
        position_ = index;

        const uint64_t dropped = monitor.dropped();
        if (dropped > dropped_seen_) {
            progress.newly_dropped = dropped - dropped_seen_;
            dropped_seen_ = dropped;
        }
        return progress;
    }

    /** Index this cursor will read next. */
    uint64_t position() const { return position_; }

    /**
     * Start from wherever the stream already stands, so notices reported
     * before this reader existed are not its to report.
     */
    void skip_to_current(const DeviceFaultMonitor &monitor) {
        position_ = monitor.sequence();
        dropped_seen_ = monitor.dropped();
    }

private:
    uint64_t position_{0};
    uint64_t dropped_seen_{0};
};

/**
 * The device-fault monitor this module may use, or `nullptr` when nothing
 * bound one.
 *
 * **Deliberately not an instance-returning accessor.** One monitor per process
 * is what makes the refcount mean anything, and a runtime `host_runtime.so`
 * cannot provide that: it is opened `RTLD_LOCAL` and `dlclose`d by
 * `ChipWorker::finalize`, so a copy per loaded runtime would give the process
 * several refcounts fighting over one driver slot, and unloading one would pull
 * the trampoline out from under a notification still in flight. The instance
 * and the trampoline therefore live in the module that is loaded once per
 * process and never unloaded, and a loader binds the address in here.
 *
 * `nullptr` is a supported answer: a module nobody bound installs nothing. A
 * test constructs its own instance rather than reaching for this one — the
 * process's monitor is process state, and a test that mutated it would leak
 * into the next.
 */
DeviceFaultMonitor *device_fault_monitor();
