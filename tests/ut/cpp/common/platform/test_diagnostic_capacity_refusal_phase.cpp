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
 * What a diagnostic capacity refusal leaves behind at the simulated C boundary.
 *
 * The lane keeps such a run `PREPARED` and plans to launch it again once
 * credit returns. Whether that plan is realisable is decided one layer down,
 * inside `simpler_launch_run`, and nowhere else: every other non-zero launch
 * result with no active execution latches `NativeRunPhase::Complete` and
 * publishes a completed run's wall span, and a `Complete` carrier is refused
 * by the phase check at the top of the next launch. A capacity refusal taking
 * that branch would make the whole wait-and-resume unreachable — the retry
 * would be rejected for its phase before reaching the device.
 *
 * **Scope: the sim entry only.** All four targets build `sim/host/c_api_shared.cpp`
 * over `sim/host/device_runner_base.cpp`, so what is proved here is the sim
 * claim policy — the claim is taken at prepare, kept across a refusal, and only
 * the permit is re-issued. The onboard entry's opposite policy, where the
 * refusal releases the claim for the retry to re-acquire, is **not** proved
 * here and cannot be: `tests/ut/cpp` is a CANN-free build (see
 * `tests/ut/cpp/a2a3/platform/CMakeLists.txt`, which cannot even open
 * `ascend_hal.h`), and no target in it links an onboard runner translation
 * unit. That branch is exercised by the onboard rows of the four
 * `test_dfx_joined_launch_attribution.py` suites, which submit through the
 * public API on real silicon.
 *
 * The runner beneath the entry is a test double, deliberately: the layer whose
 * state is in question is the C entry's carrier, and a double *above* the entry
 * — a `ChipWorker` stub returning the code — changes no carrier and would pass
 * whichever branch the entry took. `RetryRunner` returns exactly what a
 * production arch runner returns on a capacity refusal: the code, `NotStarted`,
 * and the prepared execution handed back; and on a launch it does submit, it
 * grades itself through the production `ExactLaunchTransaction`, so its receipt
 * is minted from the run's own permit rather than asserted.
 *
 * The decisive sequence is in `ARefusedRunLaunchesOnceCapacityReturns`:
 * refusal, then release, then the *same* carrier launching. The other cases
 * hold the surrounding facts that make that sequence mean something — that no
 * completion was latched, that the caller was told nothing, that the claim is
 * still this run's, and that a result which may have reached the device is
 * never retried.
 */

#include "runtime_c_api.h"

#include <gtest/gtest.h>

#include "device_runner_base.h"
#include "native_run_context.h"
#include "native_run_execution.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

/* ===========================================================================
 * The runtime-maker half, which this target does not link.
 *
 * `simpler_prepare_run` and `simpler_finalize_run` name these unconditionally
 * and a real preparation runs through all of them; the production definitions
 * live in each runtime's `runtime_maker.cpp`, which only the production
 * host_runtime.so builds. The weak placeholders in
 * `tests/ut/cpp/support/weak_link_placeholders.cpp` resolve them for targets
 * that never call them, and report failure because that is the honest answer
 * where no runtime maker exists. A case that drives the real entries does call
 * them, so it supplies its own: strong definitions override the weak ones for
 * this target alone.
 *
 * Each reports success because this run never reaches a device — the double's
 * `launch_execution` is the device boundary, and what the image and the input
 * copy would have done is the business of the runtime-maker suites. Their
 * failure modes are not what this case is about; their *reachability* is, and
 * a placeholder that failed would stop the preparation before the phase under
 * test existed.
 * =========================================================================== */
extern "C" {

int bind_callable_to_runtime_impl(
    Runtime * /*runtime*/, const HostApi * /*api*/, const ChipStorageTaskArgs * /*orch_args*/,
    void * /*host_orch_func_ptr*/, const ArgDirection * /*signature*/, int /*sig_count*/,
    const uint64_t * /*ring_task_window*/, const uint64_t * /*ring_heap*/, const uint64_t * /*ring_dep_pool*/
) {
    return 0;
}

int publish_run_image_impl(Runtime * /*runtime*/, const HostApi * /*api*/) { return 0; }

int copy_in_run_inputs_impl(const Runtime * /*runtime*/, const HostApi * /*api*/) { return 0; }

int copy_back_run_outputs_impl(
    const Runtime * /*runtime*/, const HostApi * /*api*/, int /*execution_rc*/, int /*launched*/
) {
    return 0;
}

int release_run_bindings_impl(Runtime * /*runtime*/, const HostApi * /*api*/) { return 0; }

}  // extern "C"

namespace {

constexpr int32_t kCallableId = 11;
constexpr int kSimDeviceId = 0;
/** What a caller's acceptance sink holds before anything publishes into it. */
constexpr int32_t kUnaccepted = 0;
constexpr int32_t kAccepted = 7;

/**
 * A sim runner whose launch result each case chooses.
 *
 * Everything else is the production base: the execution claim, the permit, the
 * launch transaction and the native-run bookkeeping the C entry reads are the
 * real ones, which is what makes the phase assertions below about production
 * state.
 */
class RetryRunner : public SimDeviceRunnerBase {
public:
    RetryRunner() {
        // The two registration facts a real `simpler_prepare_run` checks: the
        // callable is registered, and the retained block its hash names exists.
        // Standing them up directly keeps the case off the dlopen/image path,
        // which is covered by its own suite and is not what this is about. The
        // block's function tables are deliberately empty — the bind installs
        // no reference for a callable that dispatches nothing.
        callables_[kCallableId] = CallableState{};
        chip_callable_buffers_[CallableState{}.chip_buffer_hash] = ChipCallableBuffer{};
    }

    /** Everything a real prepare needs from the runner, and nothing more. */
    bool stand_up() {
        if (attach_current_thread(kSimDeviceId) != 0) return false;
        // One run-resource set, which is what makes pipeline slot 0 and arena
        // bank 0 name a resource this context was granted.
        return set_run_resource_grant(1);
    }

    int prepare_execution(
        Runtime &runtime, const CallConfig &config, uint32_t pipeline_slot, const NativeRunIdentity &identity,
        std::unique_ptr<PreparedExecution> *prepared
    ) override {
        *prepared = std::make_unique<PreparedExecution>(identity, runtime, config, pipeline_slot);
        prepare_calls++;
        return 0;
    }

    LaunchOutcome launch_execution(std::unique_ptr<PreparedExecution> prepared, LaunchPermit permit) override {
        launch_calls++;
        LaunchOutcome outcome;
        if (refuse_for_capacity) {
            // Exactly what an arch runner returns when an enabled owner has no
            // slot free: the refusal is decided before the transaction, so
            // nothing is submitted, the permit is never consumed, and the
            // prepared execution is handed back so the caller can leave the run
            // launchable.
            outcome.rc = SIMPLER_RUNTIME_ERR_DIAGNOSTIC_CAPACITY_FULL;
            outcome.progress = LaunchProgress::NotStarted;
            outcome.prepared = std::move(prepared);
            return outcome;
        }
        // The production transaction, consuming this run's own permit: only
        // both submissions succeeding mints a receipt, and that receipt is
        // what the entry publishes acceptance from.
        const NativeRunIdentity identity = prepared->identity;
        const bool partial = fail_after_partial_submission;
        LaunchTransactionResult result = exact_launch_transaction(
            identity, std::move(permit),
            [partial](LaunchProgressSink &sink) {
                if (!partial) return 0;
                // The device accepted a submission and a later step failed,
                // which is what grades a run `Partial`. The code it carries is
                // the capacity one, to pin that the entry grades by progress
                // and not by the value.
                sink.mark_submitted();
                return static_cast<int>(SIMPLER_RUNTIME_ERR_DIAGNOSTIC_CAPACITY_FULL);
            },
            [](LaunchProgressSink &) {
                return 0;
            }
        );
        outcome.rc = result.rc;
        outcome.progress = result.progress;
        outcome.receipt = std::move(result.receipt);
        if (result.progress == LaunchProgress::NotStarted) {
            outcome.prepared = std::move(prepared);
        } else {
            outcome.active = std::make_unique<ActiveExecution>(std::move(prepared), result.progress);
        }
        return outcome;
    }

    void abandon_prepared_execution(PreparedExecution & /*prepared*/) noexcept override {}
    int poll_execution(const ActiveExecution & /*active*/) override { return SIMPLER_NATIVE_RUN_POLL_COMPLETE; }
    DrainOutcome drain_execution(ActiveExecution & /*active*/) override { return DrainOutcome::device_complete(0); }
    int finalize() override { return 0; }
    int ensure_binaries_loaded() override { return 0; }
    int invoke_device_register(const RegisterCallableArgs & /*reg_args*/) override { return 0; }

    bool refuse_for_capacity{true};
    bool fail_after_partial_submission{false};
    int prepare_calls{0};
    int launch_calls{0};

    /** Whether this owner still holds the execution claim. */
    bool claim_held_by(const void *owner) const { return native_run_owned_by(owner); }
};

/** Runtime storage with the alignment and zeroing `simpler_prepare_run` requires. */
struct RuntimeStorage {
    std::vector<unsigned char> bytes;

    RuntimeStorage() :
        bytes(get_runtime_size() + get_runtime_alignment(), 0) {}

    void *handle() {
        auto address = reinterpret_cast<uintptr_t>(bytes.data());
        const size_t align = get_runtime_alignment();
        const uintptr_t aligned = (address + align - 1) & ~static_cast<uintptr_t>(align - 1);
        return reinterpret_cast<void *>(aligned);
    }
};

/** A config with one owner on, which is what makes credit a term at all. */
CallConfig diagnostic_config() {
    CallConfig config;
    config.enable_scope_stats = 1;
    std::snprintf(config.output_prefix, sizeof(config.output_prefix), "%s", "credit-retry");
    return config;
}

struct PreparedRun {
    RetryRunner runner;
    RuntimeStorage storage;
    /** The caller's acceptance sink, which only a matching receipt may write. */
    volatile int32_t accepted{kUnaccepted};

    /** Drive the real `simpler_prepare_run` to a real PREPARED carrier. */
    int prepare() {
        const CallConfig config = diagnostic_config();
        NativeRunDescriptor descriptor{};
        descriptor.pipeline_slot = 0;
        descriptor.arena_bank = 0;
        descriptor.run_id = 4100;
        descriptor.generation = 1;
        descriptor.dispatch_id = 2;
        descriptor.run_epoch = 9100;
        descriptor.accepted_state = &accepted;
        descriptor.accepted_value = kAccepted;
        return simpler_prepare_run(
            static_cast<DeviceContextHandle>(&runner), storage.handle(), kCallableId, nullptr, &config, &descriptor
        );
    }

    int launch() { return simpler_launch_run(static_cast<DeviceContextHandle>(&runner), storage.handle()); }
    int finalize() { return simpler_finalize_run(static_cast<DeviceContextHandle>(&runner), storage.handle()); }

    const NativeRunContext<SimDeviceRunnerBase> *state() const {
        return reinterpret_cast<const NativeRunContext<SimDeviceRunnerBase> *>(
            const_cast<RuntimeStorage &>(storage).handle()
        );
    }

    NativeRunPhase phase() const { return state()->phase.load(std::memory_order_acquire); }
    int completion_rc() const { return state()->completion_rc; }
    /** What the caller's sink holds, read as a value rather than as storage. */
    int32_t acceptance() const { return accepted; }
};

/**
 * The value `completion_rc` carries before any phase writes one.
 *
 * Not zero: `NativeRunContext` initializes it to -1 so that "no result yet" is
 * distinguishable from "completed with 0". A capacity refusal must leave this
 * sentinel exactly as prepare left it, which is a different claim from leaving
 * it at some benign value.
 */
constexpr int kNoCompletionYet = -1;

}  // namespace

TEST(DiagnosticCapacityRefusalPhaseTest, ARefusedRunLaunchesOnceCapacityReturns) {
    // The sequence the whole wait-and-resume rests on, through the real entry.
    PreparedRun run;
    ASSERT_TRUE(run.runner.stand_up());
    ASSERT_EQ(run.prepare(), 0);
    ASSERT_EQ(run.phase(), NativeRunPhase::Prepared);
    ASSERT_EQ(run.runner.prepare_calls, 1) << "the preparation never reached the runner";

    // First launch: every owner's capacity is held by a run ahead.
    EXPECT_EQ(run.launch(), SIMPLER_RUNTIME_ERR_DIAGNOSTIC_CAPACITY_FULL);
    EXPECT_EQ(run.phase(), NativeRunPhase::Prepared)
        << "a refusal that submitted nothing latched a phase the next launch refuses";
    EXPECT_EQ(run.completion_rc(), kNoCompletionYet) << "a run that never started carries a completion code";
    EXPECT_EQ(run.acceptance(), kUnaccepted) << "a run that submitted nothing was published to its caller as accepted";

    // Capacity comes back, and the *same* carrier launches. Under the defect
    // this second call is rejected by the phase check before it reaches the
    // runner at all, which `launch_calls` is what distinguishes.
    run.runner.refuse_for_capacity = false;
    EXPECT_EQ(run.launch(), 0) << "the retry was refused rather than launched";
    EXPECT_EQ(run.runner.launch_calls, 2) << "the retry never reached the runner";
    EXPECT_EQ(run.phase(), NativeRunPhase::Running);
    EXPECT_EQ(run.completion_rc(), 0);
    // The retry's permit was re-minted and consumed by the transaction, which
    // only a matching receipt could have come out of — and only that receipt
    // publishes acceptance.
    EXPECT_EQ(run.acceptance(), kAccepted) << "the relaunched run never published its acceptance";

    EXPECT_EQ(run.finalize(), 0) << "the relaunched run did not retire cleanly";
    EXPECT_FALSE(run.runner.native_run_active()) << "finalize left the execution claim held";
}

TEST(DiagnosticCapacityRefusalPhaseTest, ARefusalKeepsThisRunsExecutionClaim) {
    // This runner takes its claim at prepare and holds it through finalize,
    // so a refusal must not give it away: the retry's own phase check reads
    // `runner_claimed`, and a released claim would fail it with no way back.
    PreparedRun run;
    ASSERT_TRUE(run.runner.stand_up());
    ASSERT_EQ(run.prepare(), 0);
    const void *owner = run.storage.handle();
    ASSERT_TRUE(run.runner.claim_held_by(owner));

    ASSERT_EQ(run.launch(), SIMPLER_RUNTIME_ERR_DIAGNOSTIC_CAPACITY_FULL);
    EXPECT_TRUE(run.runner.claim_held_by(owner)) << "a retryable refusal gave away the claim its retry needs";

    EXPECT_EQ(run.finalize(), 0);
}

TEST(DiagnosticCapacityRefusalPhaseTest, ASubmissionThatMayHaveReachedTheDeviceIsNeverRetried) {
    // The same code from a `Partial` transaction. Progress past `NotStarted`
    // means this run may own device work, and nothing about it is retryable
    // however it was reported — so it takes the ordinary failure path.
    PreparedRun run;
    run.runner.refuse_for_capacity = false;
    run.runner.fail_after_partial_submission = true;
    ASSERT_TRUE(run.runner.stand_up());
    ASSERT_EQ(run.prepare(), 0);

    const int rc = run.launch();
    EXPECT_EQ(rc, SIMPLER_RUNTIME_ERR_DIAGNOSTIC_CAPACITY_FULL);
    EXPECT_NE(run.phase(), NativeRunPhase::Prepared) << "a partial submission was left launchable again";
    EXPECT_EQ(run.completion_rc(), rc) << "a run that may own device work carries no completion code";
    EXPECT_EQ(run.acceptance(), kUnaccepted) << "a transaction that minted no receipt published an acceptance";

    EXPECT_NE(run.finalize(), 0) << "a partial submission retired as though it had succeeded";
}

TEST(DiagnosticCapacityRefusalPhaseTest, RepeatedRefusalsLeaveTheCarrierLaunchableEachTime) {
    // A run can be refused many times before its deadline. Each refusal must
    // leave the same launchable state, or the second one quietly ends the run
    // the first one preserved.
    PreparedRun run;
    ASSERT_TRUE(run.runner.stand_up());
    ASSERT_EQ(run.prepare(), 0);
    for (int attempt = 0; attempt < 4; attempt++) {
        EXPECT_EQ(run.launch(), SIMPLER_RUNTIME_ERR_DIAGNOSTIC_CAPACITY_FULL) << "attempt " << attempt;
        EXPECT_EQ(run.phase(), NativeRunPhase::Prepared) << "attempt " << attempt;
        EXPECT_EQ(run.completion_rc(), kNoCompletionYet) << "attempt " << attempt;
    }
    run.runner.refuse_for_capacity = false;
    EXPECT_EQ(run.launch(), 0);
    EXPECT_EQ(run.acceptance(), kAccepted);

    EXPECT_EQ(run.finalize(), 0);
}

TEST(DiagnosticCapacityRefusalPhaseTest, ARefusedRunRetiresWithoutEverHavingLaunched) {
    // The deadline path's lower half: a run the lane gives up on is finalized
    // from `PREPARED`, having submitted nothing. It owes no device drain, its
    // claim goes back, and the caller is never told it was accepted.
    PreparedRun run;
    ASSERT_TRUE(run.runner.stand_up());
    ASSERT_EQ(run.prepare(), 0);
    ASSERT_EQ(run.launch(), SIMPLER_RUNTIME_ERR_DIAGNOSTIC_CAPACITY_FULL);

    EXPECT_EQ(run.finalize(), 0) << "retiring a never-launched run invented a device failure";
    EXPECT_FALSE(run.runner.native_run_active()) << "a never-launched run kept the execution claim";
    EXPECT_EQ(run.acceptance(), kUnaccepted);
}

TEST(DiagnosticCapacityRefusalPhaseTest, TheCapacityCodeCannotBeConfusedWithARunsOwnResult) {
    // A device-latched code is a run's own outcome; this one is the absence of
    // an outcome. They share no value, and it is distinct from every other
    // host-side code, so a caller switching on it cannot absorb an unrelated
    // failure as "merely waiting".
    EXPECT_LT(SIMPLER_RUNTIME_ERR_DIAGNOSTIC_CAPACITY_FULL, -PTO_RUNTIME_LATCHED_CODE_MAX);
    EXPECT_NE(SIMPLER_RUNTIME_ERR_DIAGNOSTIC_CAPACITY_FULL, PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_NE(SIMPLER_RUNTIME_ERR_DIAGNOSTIC_CAPACITY_FULL, PTO_RUNTIME_ERR_UNSUPPORTED);
    EXPECT_NE(SIMPLER_RUNTIME_ERR_DIAGNOSTIC_CAPACITY_FULL, PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_NE(SIMPLER_RUNTIME_ERR_DIAGNOSTIC_CAPACITY_FULL, PTO_RUNTIME_ERR_INVALID_ARGUMENT);
    EXPECT_NE(SIMPLER_RUNTIME_ERR_DIAGNOSTIC_CAPACITY_FULL, PTO_RUNTIME_ERR_PREPARED_INCOMPATIBLE);
}
