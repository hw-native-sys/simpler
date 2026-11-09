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
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "aicpu/platform_aicpu_affinity.h"
#include "common/run_terminal_accumulator.h"
#include "common/native_invocation_evidence.h"

// The private record is optional; its requested binding and functional result
// must nevertheless be internally consistent. InvalidState is a producer
// contract violation, not missing DFX detail or an SDK/DMA failure status.
enum class NativeTerminalPublication { Published, NotRequested, InvalidState };

// begin requires exclusive access after the preceding invocation's physical
// exit. The entry gate must publish its initialization before any contributor.
// Runtime handoff has one owner and precedes that owner's finish. Neither the
// completion count nor a published verdict authorizes reuse of this storage.
class NativeInvocationCompletion {
public:
    static constexpr int32_t kInternalError = -1;
    static constexpr int32_t kRetiredCcuFailure = -105;

    bool begin(int32_t callbacks, NativeInvocationEvidence *evidence = nullptr) {
        if (callbacks <= 0 || callbacks > MAX_GATE_THREADS) return false;
        evidence_ = evidence;
        if (evidence_ != nullptr) evidence_->begin_after_exit();
        expected_ = callbacks;
        claimed_.store(0, std::memory_order_relaxed);
        finished_.store(0, std::memory_order_relaxed);
        entry_errors_.reset();
        other_callback_error_.store(false, std::memory_order_relaxed);
        runtime_terminal_ = {};
        payload_bytes_ = 0;
        handed_off_ = false;
        invalid_handoff_ = false;
        ccu_failure_retired_ = false;
        return true;
    }

    // Copy while the runtime snapshot is alive; cleanup must not leave borrowed
    // payload pointers for the last callback to dereference.
    bool accept_runtime_terminal(
        RunTerminalSelection terminal, const void *payload, size_t bytes, bool ccu_failure_retired = false
    ) {
        if (handed_off_) {
            invalid_handoff_ = true;
            return false;
        }
        handed_off_ = true;
        const bool success = terminal.verdict == DeviceRunVerdict::Ok && terminal.code == 0 &&
                             terminal.source == DeviceRunCodeSource::None && bytes == 0;
        const bool failure = terminal.verdict == DeviceRunVerdict::Error && terminal.code != 0 &&
                             device_run_code_source_is_failure(terminal.source);
        if ((!success && !failure) || bytes > sizeof(payload_) || (bytes != 0 && payload == nullptr) ||
            (ccu_failure_retired && (!failure || terminal.code != kRetiredCcuFailure))) {
            invalid_handoff_ = true;
            return false;
        }
        runtime_terminal_ = terminal;
        // The runtime owner supplies this only after every AICore has acknowledged
        // retirement. The terminal record itself is never a retirement proof.
        ccu_failure_retired_ = ccu_failure_retired;
        payload_bytes_ = bytes;
        if (bytes != 0) std::memcpy(payload_, payload, bytes);
        return true;
    }

    // publish(selection, payload, bytes) returns NativeTerminalPublication.
    // NotRequested cannot turn a valid execution into an error.
    // It is invoked only by the last unique contributor, and must not throw.
    template <typename Publish>
    int32_t finish(int32_t callback_index, int32_t rc, Publish &&publish) {
        return finish(callback_index, rc, static_cast<Publish &&>(publish), [] {});
    }

    template <typename Publish, typename Complete>
    int32_t finish(int32_t callback_index, int32_t rc, Publish &&publish, Complete &&complete) {
        if (callback_index < 0 || callback_index >= expected_) return kInternalError;
        const uint32_t bit = uint32_t{1} << callback_index;
        if ((claimed_.fetch_or(bit, std::memory_order_relaxed) & bit) != 0) return kInternalError;
        entry_errors_.record(rc, DeviceRunCodeSource::ThreadRc);
        // Keep an independent veto: the first recorded error may already be -105.
        if (rc != 0 && rc != kRetiredCcuFailure) other_callback_error_.store(true, std::memory_order_relaxed);
        // The claim precedes the contribution; only this separate release RMW
        // certifies its completion. The final acquire observes all contributors.
        if (finished_.fetch_add(1, std::memory_order_acq_rel) + 1 != expected_) return 0;
        if (invalid_handoff_) {
            if (evidence_ != nullptr) evidence_->complete_unknown(NativeInvocationEvidence::Kind::InvalidTerminal);
            complete();
            return kInternalError;
        }
        RunTerminalSelection result = runtime_terminal_;
        size_t bytes = payload_bytes_;
        if (result.verdict != DeviceRunVerdict::Error && entry_errors_.has_failure()) {
            result = {DeviceRunVerdict::Error, entry_errors_.code(), entry_errors_.source()};
            bytes = 0;
        }
        if (result.verdict == DeviceRunVerdict::None) {
            if (evidence_ != nullptr) evidence_->complete_unknown(NativeInvocationEvidence::Kind::MissingTerminal);
            complete();
            return kInternalError;
        }
        const void *payload = bytes != 0 ? payload_ : nullptr;
        const NativeTerminalPublication publication = publish(result, payload, bytes);
        if (evidence_ != nullptr)
            evidence_->complete(
                result, payload, bytes, publication == NativeTerminalPublication::Published,
                publication != NativeTerminalPublication::NotRequested
            );
        complete();
        // Preserve the business failure and its payload for Host observation,
        // while allowing normal stream completion only for the explicitly
        // retired CCU failure whose own record was successfully published.
        if (ccu_failure_retired_ && result.verdict == DeviceRunVerdict::Error && result.code == kRetiredCcuFailure &&
            !other_callback_error_.load(std::memory_order_relaxed) &&
            publication == NativeTerminalPublication::Published) {
            return 0;
        }
        // Preserve a real execution error even if its private record is invalid.
        return publication != NativeTerminalPublication::InvalidState || result.code != 0 ? result.code :
                                                                                            kInternalError;
    }

private:
    static_assert(MAX_GATE_THREADS <= 32);
    NativeInvocationEvidence *evidence_{nullptr};
    int32_t expected_{0};
    std::atomic<uint32_t> claimed_{0};
    std::atomic<int32_t> finished_{0};
    RunTerminalAccumulator entry_errors_;
    std::atomic<bool> other_callback_error_{false};
    RunTerminalSelection runtime_terminal_{};
    size_t payload_bytes_{0};
    uint8_t payload_[DEVICE_RUN_RESULT_PAYLOAD_BYTES]{};
    bool handed_off_{false};
    bool invalid_handoff_{false};
    bool ccu_failure_retired_{false};
};
