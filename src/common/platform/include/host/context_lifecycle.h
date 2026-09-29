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

#include "host/teardown_proof.h"
#include "runtime_c_api.h"

/**
 * The order a program context's initialization and teardown take their
 * lifecycle steps in, separated from what each step does.
 *
 * The orders are the load-bearing part: a proof spent after the first step
 * that changes a resource would let a failed initialization inherit the
 * previous close, and a teardown that invalidates its proof after the step
 * that can refuse or throw would leave a stale one behind. Both are ordering
 * properties of the whole sequence rather than of any one call, so they live
 * here, where the sequence is one function that a caller supplies the steps
 * to. `simpler_init` and `finalize_device` supply the runner's; a test
 * supplies steps that fail on purpose.
 *
 * The slot is the proof's storage: the entries read and write it through the
 * caller's own accessors, so there is one copy of the state and no write-back
 * for a caller to forget.
 */
struct TeardownProofSlot {
    void *ctx{nullptr};
    TeardownProof (*get)(void *ctx){nullptr};
    void (*set)(void *ctx, TeardownProof proof){nullptr};
};

/** Why an initialization was refused, or that it was admitted. */
enum class ContextInitAdmission : std::uint32_t {
    /** Admitted, and the proof has been spent. */
    Admitted = 0,
    /** A prepared or launched run of this context is still live. */
    RefusedOutstandingRun,
    /** Live or half-built: a device is bound and no teardown proved otherwise. */
    RefusedLive,
    /** A teardown ran without proving it completed, or an initialization failed. */
    RefusedUnproven,
};

/**
 * Decide whether this handle may be initialized, and spend the proof if so.
 *
 * A refusal leaves the slot untouched, so a caller that goes on to close
 * properly keeps whatever proof that close earns. An admission consumes before
 * returning, which is what puts the spend ahead of every step below.
 */
inline ContextInitAdmission
admit_context_init(const TeardownProofSlot &slot, bool runs_outstanding, bool device_bound) {
    const TeardownProof proof = slot.get(slot.ctx);
    if (runs_outstanding) return ContextInitAdmission::RefusedOutstandingRun;
    if (!may_initialize_context(proof, runs_outstanding, device_bound)) {
        return proof == TeardownProof::NotAttempted ? ContextInitAdmission::RefusedLive :
                                                      ContextInitAdmission::RefusedUnproven;
    }
    slot.set(slot.ctx, consume_teardown_proof(proof));
    return ContextInitAdmission::Admitted;
}

/** The steps an initialization runs once it is admitted, in order. */
struct ContextInstallSteps {
    void *ctx{nullptr};
    /** Put the workspace regions under this context's manager. May throw. */
    int (*install_workspace)(void *ctx){nullptr};
    /** Forget a staging that never became an installation. */
    void (*clear_staging)(void *ctx){nullptr};
    /** Level the device log session, which must precede the device attach. May throw. */
    void (*configure_logging)(void *ctx){nullptr};
};

/**
 * Run the steps between the spend and the device attach, fail-closed.
 *
 * Every exit other than success records that this context is no longer proved,
 * including an escaping exception: the proof is already spent by the time this
 * runs, so an exit that left the slot alone would look like a handle that had
 * never been initialized.
 *
 * @return 0 on success, the step's own code on failure, or
 *         PTO_RUNTIME_ERR_INTERNAL when a step threw
 */
inline int run_context_install(const ContextInstallSteps &steps, const TeardownProofSlot &slot) {
    try {
        const int install_rc = steps.install_workspace(steps.ctx);
        if (install_rc != 0) {
            steps.clear_staging(steps.ctx);
            slot.set(slot.ctx, TeardownProof::Unresolved);
            return install_rc;
        }
        steps.configure_logging(steps.ctx);
    } catch (...) {
        slot.set(slot.ctx, TeardownProof::Unresolved);
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    return 0;
}

/** The steps a teardown runs, in order. */
struct ContextTeardownSteps {
    void *ctx{nullptr};
    /**
     * Whether a prepared or launched run of this context is still live. Takes
     * the runner's lock and logs its refusal, so it may throw.
     */
    bool (*runs_outstanding)(void *ctx){nullptr};
    /** Publish whatever diagnostics are still retained. May throw. */
    void (*publish_retained)(void *ctx){nullptr};
    /**
     * The device cleanup. Its ordinary managed path records `SweptClean` into
     * the slot itself; a fatal path skips that cleanup and records nothing,
     * which is why a zero return from one cannot be told from the other here
     * and the slot is what decides.
     */
    int (*cleanup)(void *ctx){nullptr};
};

/**
 * Run a teardown and leave behind what it proved.
 *
 * The invalidation comes first — before the refusal and before any step that
 * can throw — so a refused, throwing or fatal teardown cannot leave an older
 * proof standing. Only the cleanup's own record, promoted by a zero return,
 * survives it.
 *
 * Every step is inside the guard, the refusal check included: this runs behind
 * a C entry, so no step may unwind past it.
 *
 * @return 0 on success, PTO_RUNTIME_ERR_INTERNAL for an outstanding run or a
 *         throwing step, else the cleanup's code
 */
inline int run_context_teardown(const ContextTeardownSteps &steps, const TeardownProofSlot &slot) {
    slot.set(slot.ctx, TeardownProof::Unresolved);
    try {
        if (steps.runs_outstanding(steps.ctx)) return PTO_RUNTIME_ERR_INTERNAL;
        steps.publish_retained(steps.ctx);
        const int rc = steps.cleanup(steps.ctx);
        slot.set(slot.ctx, resolve_teardown_proof(slot.get(slot.ctx), rc));
        return rc;
    } catch (...) {
        return PTO_RUNTIME_ERR_INTERNAL;
    }
}
