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

/**
 * Whether one device context may be initialized again on the same handle.
 *
 * The program init entry installs workspace ownership for every caller, so a
 * second initialization of a handle that still owns blocks would run the
 * bring-up over resources nothing has discharged. What decides that is this
 * one fact — what the last teardown proved about its own completion — and the
 * three transitions below are the whole of the decision.
 *
 * Its own header rather than members of the runner because the entry and the
 * cleanup each own one end of it, and because a rule this small is worth
 * testing as the rule rather than through a device.
 *
 * Holds no resource and is never published: not a field of any report, not on
 * any wire, and not reachable through an exported entry.
 */
enum class TeardownProof : std::uint32_t {
    /** No teardown has run on this context. */
    NotAttempted = 0,
    /** The ordinary managed cleanup completed and left no unproven block. */
    SweptClean,
    /** That cleanup completed and the whole teardown returned zero. */
    Proven,
    /** A teardown ran without proving it completed, or an initialization failed. */
    Unresolved,
};

/**
 * Whether the init entry may proceed on this handle.
 *
 * Two admissible states and nothing else. `Proven` is a handle whose last
 * close discharged what it owned. `NotAttempted` with no device bound is a
 * handle that never ran; `NotAttempted` *with* a device bound is one that is
 * live or half-built, which is why the device id is part of the question
 * rather than the mode latch — re-latching PROGRAM is idempotent and says
 * nothing about whether this context is already running.
 *
 * `runs_outstanding` refuses ahead of everything else it would otherwise walk
 * into: a prepared or launched run still holds slots, bindings and blocks.
 *
 * @param proof            what the last teardown proved
 * @param runs_outstanding whether a native run of this context is still live
 * @param device_bound     whether a device is currently bound to it
 */
inline bool may_initialize_context(TeardownProof proof, bool runs_outstanding, bool device_bound) {
    if (runs_outstanding) return false;
    if (proof == TeardownProof::Proven) return true;
    return proof == TeardownProof::NotAttempted && !device_bound;
}

/**
 * What a finished teardown proved, from what its cleanup recorded and what it
 * returned.
 *
 * Promotion is only ever *from* `SweptClean`, which the ordinary managed
 * cleanup is the only writer of. That is what keeps a fatal teardown — which
 * skips that cleanup entirely — from earning a proof by returning zero.
 * Everything else is unresolved, including a clean sweep whose device reset
 * then failed.
 */
inline TeardownProof resolve_teardown_proof(TeardownProof swept, int teardown_rc) {
    if (teardown_rc == 0 && swept == TeardownProof::SweptClean) return TeardownProof::Proven;
    return TeardownProof::Unresolved;
}

/**
 * Spend a proof, which authorizes exactly one initialization.
 *
 * Called once the entry's pure checks have passed and before the first step
 * that can change a resource, so a later failure in that initialization cannot
 * fall back on the previous close. Any other state is returned unchanged: a
 * refusal must leave the record alone, or a caller that goes on to close
 * properly would lose the proof that close earns.
 */
inline TeardownProof consume_teardown_proof(TeardownProof proof) {
    return proof == TeardownProof::Proven ? TeardownProof::NotAttempted : proof;
}
