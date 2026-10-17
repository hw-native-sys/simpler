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
 * @file credit_call_gate.h
 * @brief Who is allowed inside the runtime's diagnostic credit entries, and
 *        how a close proves they have all left.
 *
 * This lives in the extension image the interpreter never unloads, because
 * that is the only side that survives the `dlclose` it guards against.
 */

#pragma once

#include "chip_worker.h"

#include <condition_variable>
#include <mutex>

/**
 * The right to be inside one of the runtime's credit entries, owned by the
 * extension rather than by the runtime.
 *
 * A count kept inside the runtime cannot prove unload safety: it cannot see a
 * caller that has loaded the function pointer but not yet entered, and it
 * cannot see one that has decremented but not yet executed the runtime's
 * return epilogue. Both windows are real — `ChipWorker::finalize` `dlclose`s
 * before it nulls its pointers — so the state that decides has to live, and
 * stay valid, on the side that survives the unload. This lives in
 * `ChipRunLaneState`, which a `ChipRun` holds by `shared_ptr` and which
 * therefore outlives the lane and the worker both.
 *
 * The targets are copied in once, while the worker is demonstrably alive, and
 * handed out only *by* an acquisition under this mutex — so there is no
 * "pointer in hand, lease not yet taken" state. Cancellation is counted like
 * any other call; it is exempt only from the `Closing` refusal, which is what
 * lets the teardown owner wake every waiter without waiting on its own lease.
 */
enum class GateState : int { Open, Closing, Detached };

struct CreditCallGate {
    std::mutex mu;
    std::condition_variable idle;     // calls_in_flight reached 0
    std::condition_variable settled;  // state reached Detached; followers join here
    GateState state = GateState::Open;
    unsigned calls_in_flight = 0;
    ChipWorker::CreditEntries entries{};
};

/** Why a call was admitted, which decides whether `Closing` refuses it. */
enum class Admit : int { Ordinary, Cancel };

/**
 * An admitted call, holding its lease for exactly as long as the runtime frame
 * it guards.
 *
 * The destructor is extension code, so the decrement cannot precede the
 * runtime's return — which is the whole of what a runtime-local count could
 * not promise. An unadmitted instance is falsy and carries no targets, and
 * every refusal has a local result the caller takes instead.
 */
class CreditCall {
public:
    CreditCall(CreditCallGate &gate, Admit kind) :
        gate_(&gate) {
        std::lock_guard<std::mutex> lk(gate.mu);
        if (gate.state == GateState::Detached) return;
        if (gate.state == GateState::Closing && kind == Admit::Ordinary) return;
        gate.calls_in_flight++;
        entries_ = gate.entries;
        admitted_ = true;
    }

    ~CreditCall() {
        if (!admitted_) return;
        bool idle = false;
        {
            std::lock_guard<std::mutex> lk(gate_->mu);
            idle = (--gate_->calls_in_flight == 0);
        }
        if (idle) gate_->idle.notify_all();
    }

    CreditCall(const CreditCall &) = delete;
    CreditCall &operator=(const CreditCall &) = delete;

    explicit operator bool() const { return admitted_; }
    const ChipWorker::CreditEntries &entries() const { return entries_; }

private:
    CreditCallGate *gate_;
    ChipWorker::CreditEntries entries_{};
    bool admitted_ = false;
};

enum class CloseOutcome : int { Owner, Already };

/**
 * Claim teardown of the gate, wake every waiter, and detach the targets.
 *
 * Exactly one thread becomes the owner; every other close joins it and returns
 * only once detaching is a fact, so "close returned" means "detached" for
 * every caller. The claim precedes the cancel so a second thread is a follower
 * rather than a second owner, the cancel is counted so no other thread's drain
 * can read zero while it is in flight, and the drain follows the cancel so it
 * terminates on the waiters' own return path instead of waiting out their
 * budget. The drain holds the gate mutex only — no lane mutex, no collector
 * lock — which is what a returning call needs to decrement.
 */
inline CloseOutcome close_credit_gate(CreditCallGate &gate) {
    std::unique_lock<std::mutex> lk(gate.mu);
    if (gate.state == GateState::Detached) return CloseOutcome::Already;
    if (gate.state == GateState::Closing) {
        gate.settled.wait(lk, [&gate] {
            return gate.state == GateState::Detached;
        });
        return CloseOutcome::Already;
    }
    gate.state = GateState::Closing;
    const ChipWorker::CreditEntries entries = gate.entries;
    gate.calls_in_flight++;
    lk.unlock();
    if (entries.cancel != nullptr && entries.ctx != nullptr) entries.cancel(entries.ctx);
    lk.lock();
    gate.calls_in_flight--;
    gate.idle.wait(lk, [&gate] {
        return gate.calls_in_flight == 0;
    });
    gate.entries = ChipWorker::CreditEntries{};
    gate.state = GateState::Detached;
    gate.settled.notify_all();
    return CloseOutcome::Owner;
}
