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

#include <cstdint>
#include <mutex>

enum class CcuRequestPhaseV1 {
    IDLE,
    PREPARED,
    ARMING,
    ARMED,
    TRIGGER_PENDING,
    IN_FLIGHT,
    DATA_READY,
    DRAINED,
    RETIRING,
    RETIRED,
    CANCELLED,
    RETAINED,
    CLOSED
};
enum class CcuSubmissionV1 { NOT_SUBMITTED, SUBMITTED, UNKNOWN };
enum class CcuStateResultV1 {
    OK,
    PENDING,
    BUSY,
    INVALID_STATE,
    STALE_SEQUENCE,
    COMPLETION_MISMATCH,
    SUBMISSION_NOT_ACCEPTED,
    RETAINED
};

class CcuRequestStateV1 {
public:
    CcuStateResultV1 Prepare(uint64_t sequence);
    CcuStateResultV1 BeginArm();
    CcuStateResultV1 FinishArm(CcuSubmissionV1 submission, bool peers_armed);
    CcuStateResultV1 BeginTrigger();
    CcuStateResultV1 FinishTrigger(CcuSubmissionV1 submission);
    CcuStateResultV1 ObserveCompletion(uint64_t completed_sequence);
    CcuStateResultV1 MarkDrained(bool trigger_stream_drained, bool ccu_stream_drained);
    CcuStateResultV1 RetainPending();
    CcuStateResultV1 RetainFault();
    CcuStateResultV1 CancelPrepared(bool peers_cancelled);
    CcuStateResultV1 BeginRetire();
    CcuStateResultV1 FinishRetire(bool peers_retired);
    CcuStateResultV1 RecoverDrained(uint64_t completed_sequence, bool trigger_stream_drained, bool ccu_stream_drained);
    CcuStateResultV1 Close(bool quiescence_proven);
    CcuRequestPhaseV1 Phase() const;
    uint64_t Sequence() const;
    bool Poisoned() const;

private:
    CcuStateResultV1 RetainLocked(bool poison);
    mutable std::mutex mutex_;
    CcuRequestPhaseV1 phase_{CcuRequestPhaseV1::IDLE};
    CcuRequestPhaseV1 retained_from_{CcuRequestPhaseV1::IDLE};
    uint64_t last_sequence_{0};
    uint64_t active_sequence_{0};
    bool poisoned_{false};
    bool data_ready_{false};
    bool trigger_drained_{false};
    bool ccu_drained_{false};
};
