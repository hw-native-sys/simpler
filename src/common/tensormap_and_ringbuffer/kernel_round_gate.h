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

#pragma once

#include <atomic>
#include <cstdint>
#include <limits>

#include "aicpu/platform_aicpu_affinity.h"

namespace simpler::tmr {

struct KernelRoundTicket {
    uint64_t epoch{0};
    int32_t launch_index{-1};
};

struct KernelRoundAdmission {
    int32_t status{0};
    int32_t execution_index{-1};
};

struct KernelFinalStatus {
    int32_t runtime_status{0};
    int32_t cleanup_status{0};
};

enum class RoundArrival { Invalid, Peer, Finalizer };
enum class RoundDeparture { Invalid, Peer, Last };

// Process-local rendezvous for one native task's launch group. Every launched
// thread joins, including threads excluded from execution by affinity. Native
// tasks must be serialized externally: an unjoined thread has no task identity
// with which this gate could distinguish overlapping invocations.
//
// Each ticket has one owner. Phase transitions reject duplicates/stale tickets;
// callers must not concurrently execute different operations on the same ticket.
// A successful transition pins its slot until depart, so non-atomic payloads
// cannot be overwritten by a following epoch. No pointer or atomic is a wire.
class KernelRoundGate {
public:
    bool join(int32_t launched, int32_t reported_cpu, KernelRoundTicket *out) noexcept {
        if (out == nullptr || launched <= 0 || launched > MAX_GATE_THREADS) return false;
        uint64_t state = lifecycle_.load(std::memory_order_acquire);
        for (;;) {
            const uint64_t epoch = state >> kStageBits;
            if (stage(state) == Stage::Idle) {
                if (epoch == kMaxEpoch) return false;
                if (!lifecycle_.compare_exchange_weak(
                        state, stamp(epoch + 1, Stage::Starting), std::memory_order_acq_rel
                    ))
                    continue;
                launched_.store(launched, std::memory_order_relaxed);
                execution_count_ = 0;
                admission_ = 0;
                init_status_ = 0;
                final_ = {};
                initialized_.store(0, std::memory_order_relaxed);
                arrived_.store(0, std::memory_order_relaxed);
                departed_.store(0, std::memory_order_relaxed);
                for (auto &slot : slots_) {
                    slot.cpu = -1;
                    slot.execution_index = -1;
                    slot.init_status = 0;
                    slot.run_status = 0;
                    slot.phase.store(stamp(epoch + 1, Stage::Vacant), std::memory_order_relaxed);
                }
                slots_[0].cpu = reported_cpu;
                slots_[0].phase.store(stamp(epoch + 1, Stage::Joined), std::memory_order_relaxed);
                claims_.store(stamp(epoch + 1, 1), std::memory_order_relaxed);
                published_.store(1, std::memory_order_relaxed);
                *out = {epoch + 1, 0};
                lifecycle_.store(stamp(epoch + 1, Stage::Joining), std::memory_order_release);
                return true;
            }
            if (stage(state) == Stage::Starting) {
                state = lifecycle_.load(std::memory_order_acquire);
                continue;
            }
            if (stage(state) != Stage::Joining || launched_.load(std::memory_order_relaxed) != launched) return false;
            uint64_t claim = claims_.load(std::memory_order_relaxed);
            for (;;) {
                const int32_t index = static_cast<int32_t>(stage(claim));
                if ((claim >> kStageBits) != epoch || index >= launched) return false;
                if (!claims_.compare_exchange_weak(claim, stamp(epoch, index + 1), std::memory_order_acq_rel)) continue;
                slots_[index].cpu = reported_cpu;
                slots_[index].phase.store(stamp(epoch, Stage::Joined), std::memory_order_release);
                *out = {epoch, index};
                // A claim alone does not publish the slot's CPU report.
                published_.fetch_add(1, std::memory_order_acq_rel);
                return true;
            }
        }
    }

    // The leader prepares/validates invocation inputs before publishing this
    // verdict. Classification keeps exact CPU matches, then fills missing roles
    // in report order, without touching program's static affinity gate or TLS.
    bool publish_admission(
        const KernelRoundTicket &leader, const int32_t *allowed_cpus, int32_t count, int32_t status
    ) noexcept {
        if (leader.epoch == 0 || leader.epoch > kMaxEpoch || leader.launch_index != 0 || allowed_cpus == nullptr ||
            count <= 0 || count > MAX_GATE_THREADS)
            return false;
        for (int32_t i = 0; i < count; ++i) {
            if (allowed_cpus[i] < 0) return false;
            for (int32_t j = 0; j < i; ++j)
                if (allowed_cpus[i] == allowed_cpus[j]) return false;
        }
        const uint64_t joining = stamp(leader.epoch, Stage::Joining);
        int32_t launched;
        for (;;) {
            if (lifecycle_.load(std::memory_order_acquire) != joining) return false;
            launched = launched_.load(std::memory_order_relaxed);
            if (count > launched) return false;
            if (published_.load(std::memory_order_acquire) == launched) break;
        }
        uint64_t expected = joining;
        if (!lifecycle_.compare_exchange_strong(
                expected, stamp(leader.epoch, Stage::Admitting), std::memory_order_acq_rel
            ))
            return false;
        bool filled[MAX_GATE_THREADS]{};
        for (int32_t i = 0; i < launched; ++i) {
            for (int32_t role = 0; role < count; ++role) {
                if (!filled[role] && slots_[i].cpu == allowed_cpus[role]) {
                    slots_[i].execution_index = role;
                    filled[role] = true;
                    break;
                }
            }
        }
        int32_t role = 0;
        for (int32_t i = 0; i < launched; ++i) {
            if (slots_[i].execution_index >= 0) continue;
            while (role < count && filled[role])
                ++role;
            if (role == count) break;
            slots_[i].execution_index = role;
            filled[role++] = true;
        }
        execution_count_ = count;
        admission_ = status;
        lifecycle_.store(stamp(leader.epoch, Stage::Admitted), std::memory_order_release);
        return true;
    }

    bool wait_admission(const KernelRoundTicket &ticket, KernelRoundAdmission *out) noexcept {
        if (out == nullptr || !transition(ticket, Stage::Joined, Stage::ReadingAdmission)) return false;
        wait_phase(ticket.epoch, Stage::Admitted);
        *out = {admission_, slots_[ticket.launch_index].execution_index};
        slots_[ticket.launch_index].phase.store(stamp(ticket.epoch, Stage::AdmissionRead), std::memory_order_release);
        return true;
    }

    bool report_init(const KernelRoundTicket &ticket, int32_t status) noexcept {
        if (!transition(ticket, Stage::AdmissionRead, Stage::InitWriting)) return false;
        auto &slot = slots_[ticket.launch_index];
        if (admission_ != 0 || slot.execution_index < 0) {
            slot.phase.store(stamp(ticket.epoch, Stage::AdmissionRead), std::memory_order_release);
            return false;
        }
        slot.init_status = status;
        slot.phase.store(stamp(ticket.epoch, Stage::InitDone), std::memory_order_release);
        initialized_.fetch_add(1, std::memory_order_acq_rel);
        return true;
    }

    // Called after the leader has reported its own init, if selected. A
    // decoupled orchestrator still waits here before entering orchestration.
    bool publish_init_verdict(const KernelRoundTicket &leader) noexcept {
        return publish_init_verdict(leader, []() noexcept {
            return 0;
        });
    }

    template <typename CompleteInit>
    bool publish_init_verdict(const KernelRoundTicket &leader, CompleteInit complete_init) noexcept {
        if (leader.launch_index != 0) return false;
        Stage previous = Stage::InitDone;
        if (!transition(leader, previous, Stage::InitPublishing)) {
            previous = Stage::AdmissionRead;
            if (!transition(leader, previous, Stage::InitPublishing)) return false;
        }
        if (admission_ != 0 || (previous == Stage::AdmissionRead && slots_[leader.launch_index].execution_index >= 0)) {
            slots_[leader.launch_index].phase.store(stamp(leader.epoch, previous), std::memory_order_release);
            return false;
        }
        uint64_t expected = stamp(leader.epoch, Stage::Admitted);
        if (!lifecycle_.compare_exchange_strong(
                expected, stamp(leader.epoch, Stage::Initializing), std::memory_order_acq_rel
            )) {
            slots_[leader.launch_index].phase.store(stamp(leader.epoch, previous), std::memory_order_release);
            return false;
        }
        while (initialized_.load(std::memory_order_acquire) != execution_count_) {}
        for (int32_t i = 0; i < launched_.load(std::memory_order_relaxed); ++i) {
            if (slots_[i].init_status != 0) {
                init_status_ = slots_[i].init_status;
                break;
            }
        }
        if (init_status_ == 0) init_status_ = complete_init();
        slots_[leader.launch_index].phase.store(stamp(leader.epoch, previous), std::memory_order_release);
        lifecycle_.store(stamp(leader.epoch, Stage::Initialized), std::memory_order_release);
        return true;
    }

    bool wait_init_verdict(const KernelRoundTicket &ticket, int32_t *out) noexcept {
        if (out == nullptr) return false;
        if (!transition(ticket, Stage::InitDone, Stage::ReadingInit)) {
            if (!transition(ticket, Stage::AdmissionRead, Stage::ReadingInit)) return false;
            // Only filtered threads skip initialization.
            if (admission_ != 0 || slots_[ticket.launch_index].execution_index >= 0) {
                slots_[ticket.launch_index].phase.store(
                    stamp(ticket.epoch, Stage::AdmissionRead), std::memory_order_release
                );
                return false;
            }
        }
        wait_phase(ticket.epoch, Stage::Initialized);
        *out = init_status_;
        slots_[ticket.launch_index].phase.store(stamp(ticket.epoch, Stage::ReadyToRun), std::memory_order_release);
        return true;
    }

    RoundArrival arrive(const KernelRoundTicket &ticket, int32_t runtime_status) noexcept {
        if (!transition(ticket, Stage::ReadyToRun, Stage::Arriving)) {
            if (!transition(ticket, Stage::AdmissionRead, Stage::Arriving)) return RoundArrival::Invalid;
            if (admission_ == 0) {
                slots_[ticket.launch_index].phase.store(
                    stamp(ticket.epoch, Stage::AdmissionRead), std::memory_order_release
                );
                return RoundArrival::Invalid;
            }
        }
        auto &slot = slots_[ticket.launch_index];
        slot.run_status = runtime_status;
        slot.phase.store(stamp(ticket.epoch, Stage::Arrived), std::memory_order_release);
        const int32_t launched = launched_.load(std::memory_order_relaxed);
        if (arrived_.fetch_add(1, std::memory_order_acq_rel) + 1 != launched) return RoundArrival::Peer;
        slot.phase.store(stamp(ticket.epoch, Stage::Finalizer), std::memory_order_release);
        return RoundArrival::Finalizer;
    }

    // Fallible shutdown/destroy finishes before this call. Preserve cleanup
    // independently; execution priority is admission, init, SM, then first
    // launch-index error, never a race between error-reporting threads.
    bool publish_final_status(const KernelRoundTicket &finalizer, int32_t sm_status, int32_t cleanup_status) noexcept {
        return publish_final_status(finalizer, sm_status, cleanup_status, [](const KernelFinalStatus &) noexcept {});
    }

    // Publish external completion evidence before allowing any native worker
    // to return an error: transport may begin cancellation on that return.
    template <typename PublishReport>
    bool publish_final_status(
        const KernelRoundTicket &finalizer, int32_t sm_status, int32_t cleanup_status, PublishReport publish_report
    ) noexcept {
        if (!transition(finalizer, Stage::Finalizer, Stage::Finalizing)) return false;
        int32_t status = admission_ != 0 ? admission_ : init_status_;
        if (status == 0) status = sm_status;
        if (status == 0) {
            for (int32_t i = 0; i < launched_.load(std::memory_order_relaxed); ++i) {
                if (slots_[i].run_status != 0) {
                    status = slots_[i].run_status;
                    break;
                }
            }
        }
        final_ = {status, cleanup_status};
        publish_report(final_);
        slots_[finalizer.launch_index].phase.store(stamp(finalizer.epoch, Stage::Arrived), std::memory_order_release);
        lifecycle_.store(stamp(finalizer.epoch, Stage::Final), std::memory_order_release);
        return true;
    }

    bool read_final_status(const KernelRoundTicket &ticket, KernelFinalStatus *out) noexcept {
        if (out == nullptr || !transition(ticket, Stage::Arrived, Stage::ReadingFinal)) return false;
        wait_phase(ticket.epoch, Stage::Final);
        *out = final_;
        slots_[ticket.launch_index].phase.store(stamp(ticket.epoch, Stage::FinalRead), std::memory_order_release);
        return true;
    }

    RoundDeparture depart(const KernelRoundTicket &ticket) noexcept {
        if (!transition(ticket, Stage::FinalRead, Stage::Departed)) return RoundDeparture::Invalid;
        const int32_t launched = launched_.load(std::memory_order_relaxed);
        if (departed_.fetch_add(1, std::memory_order_acq_rel) + 1 != launched) return RoundDeparture::Peer;
        slots_[ticket.launch_index].phase.store(stamp(ticket.epoch, Stage::Retiring), std::memory_order_release);
        lifecycle_.store(stamp(ticket.epoch, Stage::Retiring), std::memory_order_release);
        return RoundDeparture::Last;
    }

    // The last reader performs noexcept storage/round cleanup before this
    // call. Publishing Idle is the last shared operation; a new leader can
    // overwrite every field immediately. Return only thread-local results.
    bool complete_departure(const KernelRoundTicket &last) noexcept {
        if (!transition(last, Stage::Retiring, Stage::Retired)) return false;
        lifecycle_.store(stamp(last.epoch, Stage::Idle), std::memory_order_release);
        return true;
    }

    bool idle() const noexcept { return stage(lifecycle_.load(std::memory_order_acquire)) == Stage::Idle; }

private:
    enum class Stage : uint8_t {
        Idle,
        Starting,
        Joining,
        Admitting,
        Admitted,
        Initializing,
        Initialized,
        Final,
        Vacant,
        Joined,
        ReadingAdmission,
        AdmissionRead,
        InitWriting,
        InitDone,
        InitPublishing,
        ReadingInit,
        ReadyToRun,
        Arriving,
        Arrived,
        Finalizer,
        Finalizing,
        ReadingFinal,
        FinalRead,
        Departed,
        Retiring,
        Retired
    };
    static constexpr int kStageBits = 8;
    static constexpr uint64_t kMaxEpoch = std::numeric_limits<uint64_t>::max() >> kStageBits;
    static uint64_t stamp(uint64_t epoch, int32_t value) noexcept { return (epoch << kStageBits) | value; }
    static uint64_t stamp(uint64_t epoch, Stage value) noexcept { return stamp(epoch, static_cast<int32_t>(value)); }
    static Stage stage(uint64_t value) noexcept { return static_cast<Stage>(value & 0xff); }

    struct Slot {
        std::atomic<uint64_t> phase{0};
        int32_t cpu{-1};
        int32_t execution_index{-1};
        int32_t init_status{0};
        int32_t run_status{0};
    };

    bool transition(const KernelRoundTicket &ticket, Stage from, Stage to) noexcept {
        if (ticket.epoch == 0 || ticket.epoch > kMaxEpoch || ticket.launch_index < 0 ||
            ticket.launch_index >= MAX_GATE_THREADS)
            return false;
        uint64_t expected = stamp(ticket.epoch, from);
        return slots_[ticket.launch_index].phase.compare_exchange_strong(
            expected, stamp(ticket.epoch, to), std::memory_order_acq_rel
        );
    }

    void wait_phase(uint64_t epoch, Stage phase) const noexcept {
        // The caller holds a live slot that cannot depart until this wait
        // returns; lifecycle cannot advance to another epoch underneath it.
        while (stage(lifecycle_.load(std::memory_order_acquire)) < phase ||
               (lifecycle_.load(std::memory_order_acquire) >> kStageBits) != epoch) {}
    }

    std::atomic<uint64_t> lifecycle_{0};
    std::atomic<uint64_t> claims_{0};
    std::atomic<int32_t> launched_{0};
    std::atomic<int32_t> published_{0};
    std::atomic<int32_t> initialized_{0};
    std::atomic<int32_t> arrived_{0};
    std::atomic<int32_t> departed_{0};
    Slot slots_[MAX_GATE_THREADS];
    int32_t execution_count_{0};
    int32_t admission_{0};
    int32_t init_status_{0};
    KernelFinalStatus final_{};
};

}  // namespace simpler::tmr
