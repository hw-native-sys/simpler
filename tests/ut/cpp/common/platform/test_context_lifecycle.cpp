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
 * The order the program entry and the teardown take their lifecycle steps in.
 *
 * `simpler_init` and `finalize_device` supply the runner's steps to these same
 * two sequences; here the steps are fakes that record when they ran and fail
 * or throw on demand. What the cases derive is the end state — no case assigns
 * the state it is checking for, and no case tells a sequence which outcome to
 * produce.
 *
 * What the fakes stand in for is only the step bodies: the workspace install,
 * the log levelling, the retained-diagnostics publish and the device cleanup.
 * The ordering under test is the production sequence's own.
 */

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "host/context_lifecycle.h"

namespace {

/** One context's proof storage plus a log of the steps that ran on it. */
struct FakeContext {
    TeardownProof proof{TeardownProof::NotAttempted};
    std::vector<std::string> steps;
    // What each fake step does when it runs.
    int install_rc{0};
    bool install_throws{false};
    bool logging_throws{false};
    bool publish_throws{false};
    bool runs_outstanding{false};
    bool outstanding_check_throws{false};
    int cleanup_rc{0};
    // Whether the cleanup sweeps. The ordinary managed path records its own
    // fact; a fatal path skips that cleanup and records nothing.
    bool cleanup_sweeps{true};
    bool cleanup_throws{false};

    // The proof as each step sees it when it is entered, which is how a case
    // asserts what had already happened by then.
    TeardownProof proof_at_outstanding_check{TeardownProof::NotAttempted};
    TeardownProof proof_at_install{TeardownProof::NotAttempted};
};

TeardownProofSlot slot_for(FakeContext &fake) {
    TeardownProofSlot slot;
    slot.ctx = &fake;
    slot.get = [](void *ctx) {
        return static_cast<FakeContext *>(ctx)->proof;
    };
    slot.set = [](void *ctx, TeardownProof proof) {
        static_cast<FakeContext *>(ctx)->proof = proof;
    };
    return slot;
}

ContextInstallSteps install_steps_for(FakeContext &fake) {
    ContextInstallSteps steps;
    steps.ctx = &fake;
    steps.install_workspace = [](void *ctx) {
        auto *self = static_cast<FakeContext *>(ctx);
        self->steps.emplace_back("install");
        self->proof_at_install = self->proof;
        if (self->install_throws) throw std::runtime_error("install threw");
        return self->install_rc;
    };
    steps.clear_staging = [](void *ctx) {
        static_cast<FakeContext *>(ctx)->steps.emplace_back("clear_staging");
    };
    steps.configure_logging = [](void *ctx) {
        auto *self = static_cast<FakeContext *>(ctx);
        self->steps.emplace_back("logging");
        if (self->logging_throws) throw std::runtime_error("logging threw");
    };
    return steps;
}

ContextTeardownSteps teardown_steps_for(FakeContext &fake) {
    ContextTeardownSteps steps;
    steps.ctx = &fake;
    steps.runs_outstanding = [](void *ctx) {
        auto *self = static_cast<FakeContext *>(ctx);
        self->steps.emplace_back("outstanding_check");
        self->proof_at_outstanding_check = self->proof;
        if (self->outstanding_check_throws) throw std::runtime_error("outstanding check threw");
        return self->runs_outstanding;
    };
    steps.publish_retained = [](void *ctx) {
        auto *self = static_cast<FakeContext *>(ctx);
        self->steps.emplace_back("publish_retained");
        if (self->publish_throws) throw std::runtime_error("publish threw");
    };
    steps.cleanup = [](void *ctx) {
        auto *self = static_cast<FakeContext *>(ctx);
        self->steps.emplace_back("cleanup");
        if (self->cleanup_throws) throw std::runtime_error("cleanup threw");
        // The ordinary managed cleanup's own record, written where the real one
        // writes it: inside the cleanup, after its sweep.
        if (self->cleanup_sweeps) self->proof = TeardownProof::SweptClean;
        return self->cleanup_rc;
    };
    return steps;
}

/** A whole ordinary close, as the teardown entry runs it. */
int close_context(FakeContext &fake) { return run_context_teardown(teardown_steps_for(fake), slot_for(fake)); }

/** An admitted initialization's lifecycle half, as the init entry runs it. */
int install_context(FakeContext &fake) { return run_context_install(install_steps_for(fake), slot_for(fake)); }

TEST(ContextLifecycle, ANewContextIsInstalledWithNothingStagedAndBeforeAnythingElse) {
    FakeContext fake;
    EXPECT_EQ(
        admit_context_init(slot_for(fake), /*runs_outstanding=*/false, /*device_bound=*/false),
        ContextInitAdmission::Admitted
    );
    EXPECT_EQ(install_context(fake), 0);
    // The install is the first lifecycle step, and it ran without anything
    // having been staged for it.
    EXPECT_EQ(fake.steps, std::vector<std::string>({"install", "logging"}));
    EXPECT_EQ(fake.proof, TeardownProof::NotAttempted);
}

TEST(ContextLifecycle, TheProofIsSpentBeforeTheInstallRuns) {
    FakeContext fake;
    fake.cleanup_sweeps = true;
    EXPECT_EQ(close_context(fake), 0);
    ASSERT_EQ(fake.proof, TeardownProof::Proven);

    fake.steps.clear();
    ASSERT_EQ(admit_context_init(slot_for(fake), false, /*device_bound=*/true), ContextInitAdmission::Admitted);
    EXPECT_EQ(install_context(fake), 0);
    // Seen from inside the first step that can change a resource: already
    // spent, so a failure after this point cannot inherit that close.
    EXPECT_EQ(fake.proof_at_install, TeardownProof::NotAttempted);
}

TEST(ContextLifecycle, NewInitOrdinaryCloseAndReinitAllSucceed) {
    FakeContext fake;
    ASSERT_EQ(admit_context_init(slot_for(fake), false, false), ContextInitAdmission::Admitted);
    ASSERT_EQ(install_context(fake), 0);

    // An ordinary close: it sweeps and returns zero, so it earns the proof the
    // next initialization needs — derived from those two facts, not assigned.
    ASSERT_EQ(close_context(fake), 0);
    EXPECT_EQ(fake.proof, TeardownProof::Proven);
    EXPECT_EQ(fake.steps.back(), "cleanup");

    // And the same handle is admitted again, device still bound.
    EXPECT_EQ(admit_context_init(slot_for(fake), false, /*device_bound=*/true), ContextInitAdmission::Admitted);
    EXPECT_EQ(install_context(fake), 0);
    // Spent: a second consecutive entry has nothing left to admit it.
    EXPECT_EQ(admit_context_init(slot_for(fake), false, /*device_bound=*/true), ContextInitAdmission::RefusedLive);
}

TEST(ContextLifecycle, ALiveOrOutstandingEntryIsRefusedWithoutRunningAStep) {
    FakeContext fake;
    // Live: a device is bound and no teardown proved otherwise.
    EXPECT_EQ(admit_context_init(slot_for(fake), false, /*device_bound=*/true), ContextInitAdmission::RefusedLive);
    // Outstanding: refused ahead of everything, even with a device unbound.
    EXPECT_EQ(
        admit_context_init(slot_for(fake), /*runs_outstanding=*/true, false),
        ContextInitAdmission::RefusedOutstandingRun
    );
    // Neither refusal ran a step or touched the record, so a caller that goes
    // on to close properly still earns its proof.
    EXPECT_TRUE(fake.steps.empty());
    EXPECT_EQ(fake.proof, TeardownProof::NotAttempted);
}

TEST(ContextLifecycle, AFailedInstallAfterTheConsumeRefusesTheNextEntry) {
    FakeContext fake;
    fake.cleanup_sweeps = true;
    ASSERT_EQ(close_context(fake), 0);
    ASSERT_EQ(fake.proof, TeardownProof::Proven);

    fake.steps.clear();
    ASSERT_EQ(admit_context_init(slot_for(fake), false, true), ContextInitAdmission::Admitted);
    fake.install_rc = PTO_RUNTIME_ERR_INVALID_STATE;
    EXPECT_EQ(install_context(fake), PTO_RUNTIME_ERR_INVALID_STATE);
    // The staging is dropped and the state is derived from the failure.
    EXPECT_EQ(fake.steps, std::vector<std::string>({"install", "clear_staging"}));
    EXPECT_EQ(admit_context_init(slot_for(fake), false, false), ContextInitAdmission::RefusedUnproven);
}

TEST(ContextLifecycle, AThrowingInstallOrLoggingStepRefusesTheNextEntry) {
    for (const bool throwing_install : {true, false}) {
        FakeContext fake;
        fake.cleanup_sweeps = true;
        ASSERT_EQ(close_context(fake), 0);
        ASSERT_EQ(admit_context_init(slot_for(fake), false, true), ContextInitAdmission::Admitted);
        fake.install_throws = throwing_install;
        fake.logging_throws = !throwing_install;
        // Nothing escapes the sequence, and a spent proof does not survive it.
        EXPECT_EQ(install_context(fake), PTO_RUNTIME_ERR_INTERNAL);
        EXPECT_EQ(admit_context_init(slot_for(fake), false, false), ContextInitAdmission::RefusedUnproven);
    }
}

TEST(ContextLifecycle, ACloseThatSkipsItsSweepEarnsNothingEvenReturningZero) {
    FakeContext fake;
    fake.cleanup_sweeps = true;
    ASSERT_EQ(close_context(fake), 0);
    ASSERT_EQ(fake.proof, TeardownProof::Proven);

    // The fatal shape: a cleanup that abandons instead of sweeping, and still
    // returns zero. It records nothing, so the sequence derives no proof from
    // its return code.
    fake.cleanup_sweeps = false;
    fake.cleanup_rc = 0;
    EXPECT_EQ(close_context(fake), 0);
    EXPECT_EQ(fake.proof, TeardownProof::Unresolved);
    EXPECT_EQ(admit_context_init(slot_for(fake), false, false), ContextInitAdmission::RefusedUnproven);
}

TEST(ContextLifecycle, ASweepingCloseWhoseTeardownFailedEarnsNothing) {
    FakeContext fake;
    fake.cleanup_sweeps = true;
    fake.cleanup_rc = 507018;
    EXPECT_EQ(close_context(fake), 507018);
    EXPECT_EQ(fake.proof, TeardownProof::Unresolved);
    EXPECT_EQ(admit_context_init(slot_for(fake), false, false), ContextInitAdmission::RefusedUnproven);
}

TEST(ContextLifecycle, ACloseInvalidatesItsProofBeforeItCanRefuseOrThrow) {
    FakeContext fake;
    fake.cleanup_sweeps = true;
    ASSERT_EQ(close_context(fake), 0);
    ASSERT_EQ(fake.proof, TeardownProof::Proven);

    // Refused for an outstanding run: the check sees a record already
    // invalidated, so the older proof cannot outlive this attempt.
    fake.runs_outstanding = true;
    EXPECT_EQ(close_context(fake), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(fake.proof_at_outstanding_check, TeardownProof::Unresolved);
    EXPECT_EQ(fake.proof, TeardownProof::Unresolved);
    EXPECT_EQ(admit_context_init(slot_for(fake), false, false), ContextInitAdmission::RefusedUnproven);
}

TEST(ContextLifecycle, AThrowingPublishOrCleanupLeavesNoProofAndNoEscape) {
    for (const bool throwing_publish : {true, false}) {
        FakeContext fake;
        fake.cleanup_sweeps = true;
        ASSERT_EQ(close_context(fake), 0);
        ASSERT_EQ(fake.proof, TeardownProof::Proven);

        fake.publish_throws = throwing_publish;
        fake.cleanup_throws = !throwing_publish;
        EXPECT_EQ(close_context(fake), PTO_RUNTIME_ERR_INTERNAL);
        EXPECT_EQ(fake.proof, TeardownProof::Unresolved);
        // A publish that throws never reaches the cleanup at all.
        if (throwing_publish) EXPECT_EQ(fake.steps.back(), "publish_retained");
    }
}

TEST(ContextLifecycle, TheTeardownRunsItsStepsInOneOrder) {
    FakeContext fake;
    fake.cleanup_sweeps = true;
    ASSERT_EQ(close_context(fake), 0);
    EXPECT_EQ(fake.steps, std::vector<std::string>({"outstanding_check", "publish_retained", "cleanup"}));
}

TEST(ContextLifecycle, AThrowingOutstandingCheckIsContainedAndRunsNothingElse) {
    FakeContext fake;
    fake.cleanup_sweeps = true;
    ASSERT_EQ(close_context(fake), 0);
    ASSERT_EQ(fake.proof, TeardownProof::Proven);

    // The check takes the runner's lock and logs its refusal, so it is a step
    // that can throw. This sequence runs behind a C entry, so the throw has to
    // become a code rather than unwind past it.
    fake.steps.clear();
    fake.outstanding_check_throws = true;
    EXPECT_EQ(close_context(fake), PTO_RUNTIME_ERR_INTERNAL);
    // Invalidated before the check ran, so the earlier proof does not survive
    // a teardown that could not even decide whether to refuse.
    EXPECT_EQ(fake.proof_at_outstanding_check, TeardownProof::Unresolved);
    EXPECT_EQ(fake.proof, TeardownProof::Unresolved);
    // And nothing after it ran: no retained publish, no cleanup.
    EXPECT_EQ(fake.steps, std::vector<std::string>({"outstanding_check"}));
    EXPECT_EQ(admit_context_init(slot_for(fake), false, false), ContextInitAdmission::RefusedUnproven);
}

}  // namespace
