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

#include "cpu_sim_context.h"
#include "device_runner_base.h"
#include "native_run_context.h"
#include "native_run_execution.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <pthread.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <cstdlib>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <array>
#include <filesystem>
#include <fstream>
#include <pthread.h>
#include <unistd.h>
#include "common/host_log_state.h"
#include "common/host_span.h"
#include "common/log_level.h"
#include "utils/tensor_lease_release.h"

namespace finalize_prefix_test {
struct AllocationFault {
    bool armed{false};
    size_t consumed{0};
    size_t bytes{0};
};
thread_local AllocationFault prefix_allocation_fault;
thread_local size_t copyback_calls{0};
}  // namespace finalize_prefix_test

namespace {
thread_local bool refuse_result_retention_allocation = false;
thread_local bool sample_trace_allocation = false;
thread_local size_t trace_allocation_bytes = 0;
thread_local size_t trace_allocation_failures = 0;
thread_local const NativeRunContext<SimDeviceRunnerBase> *trace_retirement_fault_owner = nullptr;
}  // namespace

void *operator new(std::size_t bytes) {
    if (sample_trace_allocation && trace_allocation_bytes == 0) trace_allocation_bytes = bytes;
    if (trace_retirement_fault_owner != nullptr && trace_retirement_fault_owner->physical_complete &&
        trace_retirement_fault_owner->runner_claimed && bytes == trace_allocation_bytes) {
        trace_retirement_fault_owner = nullptr;
        ++trace_allocation_failures;
        throw std::bad_alloc();
    }
    if (std::exchange(refuse_result_retention_allocation, false)) throw std::bad_alloc();
    auto &fault = finalize_prefix_test::prefix_allocation_fault;
    if (std::exchange(fault.armed, false)) {
        ++fault.consumed;
        fault.bytes = bytes;
        throw std::bad_alloc();
    }
    if (void *storage = std::malloc(bytes == 0 ? 1 : bytes)) return storage;
    throw std::bad_alloc();
}

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

int copy_back_run_outputs_impl(const Runtime *runtime, const HostApi *api, int execution_rc, int launched) {
    ++finalize_prefix_test::copyback_calls;
    if (execution_rc == 0 && launched != 0 && runtime != nullptr) {
        for (const auto &lease : runtime->tensor_leases()) {
            if (lease.needs_copy_back) {
                const int rc = api->copy_from_device(lease.host_ptr, lease.dev_ptr, lease.size);
                if (rc != 0) return rc;
            }
        }
    }
    return 0;
}

int release_run_bindings_impl(Runtime *runtime, const HostApi *api) {
    if (runtime != nullptr) release_tensor_leases(runtime->tensor_leases(), api);
    return 0;
}

}  // extern "C"

#if defined(__linux__)
namespace {
std::atomic<bool> reject_device_binding{false};
}
extern "C" int __real_pthread_setspecific(pthread_key_t key, const void *value);
extern "C" int __wrap_pthread_setspecific(pthread_key_t key, const void *value) {
    if (value == reinterpret_cast<void *>(uintptr_t{1}) && reject_device_binding.load()) return ENOMEM;
    return __real_pthread_setspecific(key, value);
}
#endif

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
    enum class ProgressException { None, Allocation, Other };
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

    ~RetryRunner() override {
        pto_cpu_sim_bind_device(-1);
        pto_cpu_sim_release_device(kSimDeviceId);
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

    LaunchTransactionResult launch_execution_impl(PreparedExecution &prepared, LaunchPermit permit) override {
        launch_calls++;
        if (refuse_for_capacity) {
            // Exactly what an arch runner returns when an enabled owner has no
            // slot free: the refusal is decided before the transaction, so
            // nothing is submitted, the permit is never consumed, and the
            // base returns the prepared execution so the caller can leave the
            // run launchable.
            LaunchTransactionResult result;
            result.rc = SIMPLER_RUNTIME_ERR_DIAGNOSTIC_CAPACITY_FULL;
            return result;
        }
        // The production transaction, consuming this run's own permit: only
        // both submissions succeeding mints a receipt, and that receipt is
        // what the entry publishes acceptance from.
        const NativeRunIdentity identity = prepared.identity;
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
        if (consume_drain_once) active_for_drain = true;
        return result;
    }

    void abandon_prepared_execution(PreparedExecution & /*prepared*/) noexcept override {}
    int poll_execution(const ActiveExecution & /*active*/) override {
        if (exception_from_poll) throw_progress_exception();
        return SIMPLER_NATIVE_RUN_POLL_COMPLETE;
    }
    DrainOutcome drain_execution(ActiveExecution &active) override {
        ++drain_calls;
        if (!exception_from_poll) throw_progress_exception();
        if (consume_drain_once) {
            if (!std::exchange(active_for_drain, false)) {
                ++empty_drain_calls;
                return DrainOutcome::device_error(PTO_RUNTIME_ERR_INTERNAL);
            }
            auto *region =
                reinterpret_cast<DeviceRunResultRegion *>(run_result_region_base(active.prepared->pipeline_slot));
            *region = {};
            region->published = active.prepared->identity.run_epoch;
            region->verdict = static_cast<uint32_t>(drain_rc == 0 ? DeviceRunVerdict::Ok : DeviceRunVerdict::Error);
            region->completion_code = drain_rc;
            if (drain_rc != 0) {
                region->code_source = static_cast<uint32_t>(DeviceRunCodeSource::ThreadRc);
                region->payload_bytes = sizeof(uint64_t);
                std::memcpy(region->payload, &region->published, sizeof(uint64_t));
            }
            if (clock_timing) device_phase_ns_[static_cast<int>(AicpuPhase::RunWall)] = 1000;
            // Both production sim backends consume active_run_ before returning.
            if (std::exchange(fail_result_retention, false)) refuse_result_retention_allocation = true;
        }
        return drain_rc == 0 ? DrainOutcome::device_complete(0) : DrainOutcome::device_error(drain_rc, true);
    }
    int finalize() override { return 0; }
    int ensure_binaries_loaded() override { return 0; }
    int invoke_device_register(const RegisterCallableArgs & /*reg_args*/) override { return 0; }

    bool refuse_for_capacity{true};
    bool fail_after_partial_submission{false};
    int prepare_calls{0};
    int launch_calls{0};
    bool consume_drain_once{false};
    bool active_for_drain{false};
    bool fail_result_retention{false};
    int drain_calls{0};
    ProgressException progress_exception{ProgressException::None};
    bool exception_from_poll{false};
    int progress_exception_calls{0};
    int empty_drain_calls{0};
    int drain_rc{0};
    bool clock_timing{false};

    /** Whether this owner still holds the execution claim. */
    bool claim_held_by(const void *owner) const { return native_run_owned_by(owner); }
    bool result_retained(uint64_t run_epoch) const {
        std::lock_guard<std::mutex> lock(retained_result_mu_);
        return retained_run_results_.find(run_epoch) != retained_run_results_.end();
    }

private:
    void throw_progress_exception() {
        const auto failure = std::exchange(progress_exception, ProgressException::None);
        if (failure == ProgressException::None) return;
        ++progress_exception_calls;
        if (failure == ProgressException::Allocation) throw std::bad_alloc();
        throw std::runtime_error("simulated physical progress failed");
    }
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
    int prepare(const CallConfig &config = diagnostic_config()) {
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

TEST(DiagnosticCapacityRefusalPhaseTest, PhysicalProgressExceptionsPreserveTheUnconsumedOwner) {
    for (bool blocking : {false, true}) {
        for (bool poll_fault : {false, true}) {
            if (blocking && poll_fault) continue;
            for (bool allocation : {false, true}) {
                SCOPED_TRACE(blocking);
                SCOPED_TRACE(poll_fault);
                SCOPED_TRACE(allocation);
                PreparedRun run;
                run.runner.refuse_for_capacity = false;
                run.runner.consume_drain_once = true;
                ASSERT_TRUE(run.runner.stand_up());
                ASSERT_EQ(run.prepare(), 0);
                ASSERT_EQ(run.launch(), 0);
                auto *const runtime_owner = run.state()->runtime.get();
                auto *const execution_owner = run.state()->active_execution.get();
                run.runner.exception_from_poll = poll_fault;
                run.runner.progress_exception =
                    allocation ? RetryRunner::ProgressException::Allocation : RetryRunner::ProgressException::Other;
                bool escaped = false;
                int rc = 12345;
                try {
                    rc = blocking ? simpler_wait_run_completion(&run.runner, run.storage.handle()) :
                                    simpler_poll_run(&run.runner, run.storage.handle());
                } catch (...) {
                    escaped = true;
                }
                EXPECT_FALSE(escaped);
                const int expected_rc = blocking ? static_cast<int>(PTO_RUNTIME_ERR_INTERNAL) :
                                                   static_cast<int>(SIMPLER_NATIVE_RUN_POLL_ERROR);
                EXPECT_EQ(rc, expected_rc);
                EXPECT_EQ(run.runner.progress_exception_calls, 1);
                EXPECT_EQ(run.state()->runtime.get(), runtime_owner);
                EXPECT_EQ(run.state()->active_execution.get(), execution_owner);
                EXPECT_FALSE(run.state()->physical_complete);
                EXPECT_FALSE(run.state()->execution_retired);
                EXPECT_FALSE(run.state()->result_observed);
                EXPECT_EQ(run.phase(), NativeRunPhase::Running);
                EXPECT_EQ(run.completion_rc(), 0);
                EXPECT_TRUE(run.runner.claim_held_by(run.storage.handle()));
                EXPECT_TRUE(run.runner.active_for_drain);
                EXPECT_EQ(run.runner.drain_calls, poll_fault ? 0 : 1);
                EXPECT_EQ(run.runner.empty_drain_calls, 0);
                std::fprintf(
                    stderr, "SIM_PROGRESS_EXCEPTION entry=%s poll_fault=%d allocation=%d escaped=%d rc=%d drains=%d\n",
                    blocking ? "wait_completion" : "poll", poll_fault, allocation, escaped, rc, run.runner.drain_calls
                );
                if (blocking) ASSERT_EQ(simpler_wait_run_completion(&run.runner, run.storage.handle()), 0);
                else ASSERT_EQ(simpler_poll_run(&run.runner, run.storage.handle()), SIMPLER_NATIVE_RUN_POLL_COMPLETE);
                EXPECT_TRUE(run.state()->physical_complete);
                EXPECT_TRUE(run.state()->execution_retired);
                EXPECT_FALSE(run.runner.claim_held_by(run.storage.handle()));
                EXPECT_FALSE(run.runner.active_for_drain);
                EXPECT_EQ(run.runner.drain_calls, poll_fault ? 1 : 2);
                EXPECT_EQ(run.runner.empty_drain_calls, 0);
                EXPECT_EQ(simpler_wait_run(&run.runner, run.storage.handle()), 0);
                EXPECT_EQ(run.finalize(), 0);
                EXPECT_FALSE(run.runner.native_runs_outstanding());
            }
        }
    }
}

TEST(DiagnosticCapacityRefusalPhaseTest, TraceAllocationFailureKeepsStagingAndRetirementRetryable) {
    for (bool blocking : {false, true}) {
        SCOPED_TRACE(blocking);
        EXPECT_EXIT(
            ([blocking]() {
                static SimplerHostLogState log_state{};
                log_state.threshold = static_cast<int>(simpler::log::LogLevel::TIMING);
                log_state.sink_owner_pid = log_state.sink_process_pid = getpid();
                log_state.sink_context = &log_state;
                log_state.sink_enqueue = [](void *, SimplerHostLogState *state, const char *record, uint32_t size) {
                    const bool written = std::fwrite(record, 1, size, stderr) == size;
                    __atomic_fetch_sub(&state->sink_producer_state, UINT64_C(1), __ATOMIC_RELEASE);
                    return written ? 1 : 0;
                };
                pthread_key_t key;
                if (pthread_key_create(&key, std::free) != 0) std::_Exit(90);
                log_state.trace_key_plus_one = static_cast<uint64_t>(key) + 1;
                if (simpler_host_log_bind_state(&log_state) != 0) std::_Exit(91);
                SimplerHostSpan span{};
                span.name = "chip.run.runner_run";
                span.attributes = "";
                sample_trace_allocation = true;
                unified_log_host_span(&span);
                sample_trace_allocation = false;
                if (trace_allocation_bytes == 0 || trace_allocation_bytes >= sizeof(DeviceRunResultRegion))
                    std::_Exit(92);

                PreparedRun run;
                run.runner.refuse_for_capacity = false;
                run.runner.consume_drain_once = true;
                if (!run.runner.stand_up() || run.prepare(CallConfig{}) != 0) std::_Exit(93);
                auto *const runtime_owner = run.state()->runtime.get();
                std::array<unsigned char, 16> output;
                output.fill(0x71);
                void *const backing = run.runner.allocate_tensor(output.size());
                if (backing == nullptr) std::_Exit(94);
                std::memset(backing, 0x42, output.size());
                run.runner.set_retained_temp_buffer(0, backing, output.size());
                run.state()->runtime->tensor_leases().push_back(
                    TensorLease{output.data(), backing, output.size(), false, true, TensorReleaseKind::BufferNoop}
                );
                if (run.launch() != 0) std::_Exit(95);
                auto *const execution_owner = run.state()->active_execution.get();
                trace_retirement_fault_owner = run.state();
                int rc = 12345;
                bool escaped = false;
                try {
                    rc = blocking ? simpler_wait_run_completion(&run.runner, run.storage.handle()) :
                                    simpler_poll_run(&run.runner, run.storage.handle());
                } catch (...) {
                    escaped = true;
                }
                const bool injected = trace_allocation_failures == 1 && trace_retirement_fault_owner == nullptr;
                trace_retirement_fault_owner = nullptr;
                void *slot_staging = nullptr;
                size_t slot_bytes = 0;
                run.runner.get_retained_temp_buffer(0, &slot_staging, &slot_bytes);
                const bool retained_result = run.runner.result_retained(run.state()->descriptor.run_epoch);
                const bool staging_owned = (slot_staging == backing && run.state()->retired_host_staging == nullptr) ||
                                           (slot_staging == nullptr && run.state()->retired_host_staging == backing);
                const int expected_rc = blocking ? static_cast<int>(PTO_RUNTIME_ERR_INTERNAL) :
                                                   static_cast<int>(SIMPLER_NATIVE_RUN_POLL_ERROR);
                bool ok = !escaped && injected && rc == expected_rc && retained_result && staging_owned &&
                          run.state()->physical_complete && run.runner.claim_held_by(run.storage.handle()) &&
                          run.state()->runtime.get() == runtime_owner &&
                          run.state()->active_execution.get() == execution_owner &&
                          std::all_of(output.begin(), output.end(), [](unsigned char value) {
                              return value == 0x71;
                          });
                const int retry_rc = blocking ? simpler_wait_run_completion(&run.runner, run.storage.handle()) :
                                                simpler_poll_run(&run.runner, run.storage.handle());
                const bool claim_after_retry = run.runner.claim_held_by(run.storage.handle());
                run.runner.get_retained_temp_buffer(0, &slot_staging, &slot_bytes);
                ok = ok && retry_rc == (blocking ? 0 : SIMPLER_NATIVE_RUN_POLL_COMPLETE) && !claim_after_retry &&
                     run.state()->execution_retired && run.state()->retired_host_staging == backing &&
                     slot_staging == nullptr && run.runner.drain_calls == 1 && run.runner.empty_drain_calls == 0;
                const int wait_rc = simpler_wait_run(&run.runner, run.storage.handle());
                const int finalize_rc = run.finalize();
                ok = ok && wait_rc == 0 && finalize_rc == 0 && !run.runner.native_runs_outstanding() &&
                     std::all_of(
                         output.begin(), output.end(),
                         [](unsigned char value) {
                             return value == 0x42;
                         }
                     ) &&
                     log_state.sink_producer_state == 0 && log_state.pending_record_count == 0 &&
                     log_state.dropped_record_count == 0;
                std::printf(
                    "SIM_TRACE_RETIRE entry=%s bytes=%zu injected=%d rc=%d retained_result=%d staging_owned=%d "
                    "retry_rc=%d claim_after_retry=%d drains=%d finalize=%d ok=%d\n",
                    blocking ? "wait_completion" : "poll", trace_allocation_bytes, injected, rc, retained_result,
                    staging_owned, retry_rc, claim_after_retry, run.runner.drain_calls, finalize_rc, ok
                );
                std::fflush(stdout);
                std::_Exit(ok ? 0 : 17);
            }()),
            ::testing::ExitedWithCode(0), ""
        );
    }
}

TEST(DiagnosticCapacityRefusalPhaseTest, ResultRetentionAllocationFailureRetriesAfterConsumedDrain) {
    for (bool blocking : {false, true}) {
        SCOPED_TRACE(blocking ? "wait_completion" : "poll");
        PreparedRun run;
        run.runner.refuse_for_capacity = false;
        run.runner.consume_drain_once = true;
        ASSERT_TRUE(run.runner.stand_up());
        ASSERT_EQ(run.prepare(), 0);
        ASSERT_EQ(run.launch(), 0);
        auto *const runtime_owner = run.state()->runtime.get();
        auto *const active_owner = run.state()->active_execution.get();
        run.runner.fail_result_retention = true;
        bool escaped = false;
        int rc = 12345;
        try {
            rc = blocking ? simpler_wait_run_completion(&run.runner, run.storage.handle()) :
                            simpler_poll_run(&run.runner, run.storage.handle());
        } catch (const std::bad_alloc &) {
            escaped = true;
        }
        const bool consumed = !std::exchange(refuse_result_retention_allocation, false);
        EXPECT_TRUE(consumed);
        EXPECT_FALSE(escaped);
        EXPECT_EQ(rc, PTO_RUNTIME_ERR_INTERNAL);
        EXPECT_EQ(run.runner.drain_calls, 1);
        EXPECT_EQ(run.runner.empty_drain_calls, 0);
        EXPECT_TRUE(run.state()->physical_complete);
        EXPECT_FALSE(run.state()->execution_retired);
        EXPECT_EQ(run.phase(), NativeRunPhase::Running);
        EXPECT_EQ(run.state()->runtime.get(), runtime_owner);
        EXPECT_EQ(run.state()->active_execution.get(), active_owner);
        EXPECT_TRUE(run.runner.claim_held_by(run.storage.handle()));
        EXPECT_EQ(run.completion_rc(), 0);
        std::fprintf(
            stderr, "SIM_RETENTION entry=%s consumed=%d escaped=%d rc=%d drains=%d physical=%d retired=%d\n",
            blocking ? "wait_completion" : "poll", consumed, escaped, rc, run.runner.drain_calls,
            run.state()->physical_complete, run.state()->execution_retired
        );
        if (blocking) EXPECT_EQ(simpler_wait_run_completion(&run.runner, run.storage.handle()), 0);
        else EXPECT_EQ(simpler_poll_run(&run.runner, run.storage.handle()), SIMPLER_NATIVE_RUN_POLL_COMPLETE);
        EXPECT_TRUE(run.state()->execution_retired);
        EXPECT_FALSE(run.runner.claim_held_by(run.storage.handle()));
        EXPECT_EQ(run.runner.drain_calls, 1);
        EXPECT_EQ(run.runner.empty_drain_calls, 0);
        EXPECT_EQ(simpler_wait_run(&run.runner, run.storage.handle()), 0);
        EXPECT_EQ(run.finalize(), 0);
        EXPECT_FALSE(run.runner.native_runs_outstanding());
        EXPECT_EQ(run.runner.drain_calls, 1);
    }
}

TEST(DiagnosticCapacityRefusalPhaseTest, ObserversReuseConsumedDrainAndKeepItsBusinessResult) {
    for (int business_rc : {0, -4321}) {
        for (bool finalize_directly : {false, true}) {
            SCOPED_TRACE(business_rc);
            SCOPED_TRACE(finalize_directly);
            PreparedRun run;
            run.runner.refuse_for_capacity = false;
            run.runner.consume_drain_once = true;
            run.runner.drain_rc = business_rc;
            ASSERT_TRUE(run.runner.stand_up());
            ASSERT_EQ(run.prepare(), 0);
            std::array<unsigned char, 16> output;
            output.fill(0x71);
            void *const backing = run.runner.allocate_tensor(output.size());
            ASSERT_NE(backing, nullptr);
            std::memset(backing, 0x42, output.size());
            run.runner.set_retained_temp_buffer(0, backing, output.size());
            run.state()->runtime->tensor_leases().push_back(
                TensorLease{output.data(), backing, output.size(), false, true, TensorReleaseKind::BufferNoop}
            );
            ASSERT_EQ(run.launch(), 0);
            run.runner.fail_result_retention = true;
            ASSERT_EQ(simpler_poll_run(&run.runner, run.storage.handle()), SIMPLER_NATIVE_RUN_POLL_ERROR);
            ASSERT_TRUE(run.state()->physical_complete);
            EXPECT_FALSE(run.state()->execution_retired);
            EXPECT_EQ(run.completion_rc(), business_rc);
            if (!finalize_directly) {
                EXPECT_EQ(simpler_wait_run(&run.runner, run.storage.handle()), business_rc);
                EXPECT_EQ(simpler_poll_run(&run.runner, run.storage.handle()), SIMPLER_NATIVE_RUN_POLL_COMPLETE);
                EXPECT_TRUE(run.state()->execution_retired);
            }
            EXPECT_EQ(run.finalize(), business_rc);
            EXPECT_TRUE(std::all_of(output.begin(), output.end(), [&](unsigned char value) {
                return value == (business_rc == 0 ? 0x42 : 0x71);
            }));
            EXPECT_EQ(run.runner.drain_calls, 1);
            EXPECT_EQ(run.runner.empty_drain_calls, 0);
            EXPECT_FALSE(run.runner.native_runs_outstanding());
        }
    }
}

TEST(DiagnosticCapacityRefusalPhaseTest, StrongWaitThenPollStillFinishesExecutionRetirement) {
    PreparedRun run;
    run.runner.refuse_for_capacity = false;
    run.runner.consume_drain_once = true;
    ASSERT_TRUE(run.runner.stand_up());
    ASSERT_EQ(run.prepare(), 0);
    ASSERT_EQ(run.launch(), 0);
    ASSERT_EQ(simpler_wait_run(&run.runner, run.storage.handle()), 0);
    ASSERT_EQ(run.phase(), NativeRunPhase::Complete);
    EXPECT_TRUE(run.state()->physical_complete);
    EXPECT_FALSE(run.state()->execution_retired);
    EXPECT_TRUE(run.runner.claim_held_by(run.storage.handle()));
    ASSERT_EQ(simpler_poll_run(&run.runner, run.storage.handle()), SIMPLER_NATIVE_RUN_POLL_COMPLETE);
    EXPECT_TRUE(run.state()->execution_retired);
    EXPECT_FALSE(run.runner.claim_held_by(run.storage.handle()));
    EXPECT_EQ(run.finalize(), 0);
    EXPECT_EQ(run.runner.drain_calls, 1);
    EXPECT_EQ(run.runner.empty_drain_calls, 0);
    EXPECT_FALSE(run.runner.native_runs_outstanding());
}

TEST(DiagnosticCapacityRefusalPhaseTest, RetiredResultsAndOutputSurviveTheSuccessorUsingTheirSlot) {
    for (int business_rc : {0, -4321}) {
        SCOPED_TRACE(business_rc);
        PreparedRun run;
        run.runner.refuse_for_capacity = false;
        run.runner.consume_drain_once = true;
        run.runner.drain_rc = business_rc;
        ASSERT_TRUE(run.runner.stand_up());
        ASSERT_EQ(run.prepare(), 0);
        std::array<unsigned char, 16> output;
        output.fill(0x71);
        void *const backing = run.runner.allocate_tensor(output.size());
        ASSERT_NE(backing, nullptr);
        std::memset(backing, 0x42, output.size());
        run.runner.set_retained_temp_buffer(0, backing, output.size());
        run.state()->runtime->tensor_leases().push_back(
            TensorLease{output.data(), backing, output.size(), false, true, TensorReleaseKind::BufferNoop}
        );
        ASSERT_EQ(run.launch(), 0);
        run.runner.fail_result_retention = true;
        ASSERT_EQ(simpler_wait_run_completion(&run.runner, run.storage.handle()), PTO_RUNTIME_ERR_INTERNAL);
        EXPECT_EQ(run.state()->runtime->tensor_leases().front().dev_ptr, backing);
        EXPECT_EQ(output.front(), 0x71);
        ASSERT_EQ(simpler_wait_run_completion(&run.runner, run.storage.handle()), 0);
        EXPECT_EQ(run.state()->retired_host_staging, backing);

        RuntimeStorage successor;
        NativeRunDescriptor descriptor = run.state()->descriptor;
        descriptor.run_epoch = 9200;
        descriptor.generation++;
        descriptor.dispatch_id++;
        CallConfig config{};
        ASSERT_EQ(simpler_prepare_run(&run.runner, successor.handle(), kCallableId, nullptr, &config, &descriptor), 0);
        run.runner.drain_rc = -6543;
        ASSERT_EQ(simpler_launch_run(&run.runner, successor.handle()), 0);
        ASSERT_EQ(simpler_wait_run_completion(&run.runner, successor.handle()), 0);
        const auto terminal = run.runner.device_run_terminal(0, 9100);
        EXPECT_EQ(
            terminal.state, business_rc == 0 ? DeviceRunTerminalState::Succeeded : DeviceRunTerminalState::Failed
        );
        EXPECT_EQ(terminal.code, business_rc);
        if (business_rc != 0) {
            size_t bytes = 0;
            const void *payload = run.state()->host_api.run_result(&bytes);
            ASSERT_NE(payload, nullptr);
            ASSERT_EQ(bytes, sizeof(uint64_t));
            uint64_t epoch = 0;
            std::memcpy(&epoch, payload, sizeof(epoch));
            EXPECT_EQ(epoch, 9100u);
        }
        EXPECT_EQ(run.state()->runtime->tensor_leases().front().dev_ptr, backing);
        EXPECT_EQ(output.front(), 0x71);
        EXPECT_EQ(simpler_finalize_run(&run.runner, successor.handle()), -6543);
        EXPECT_EQ(simpler_wait_run(&run.runner, run.storage.handle()), business_rc);
        EXPECT_EQ(run.finalize(), business_rc);
        EXPECT_TRUE(std::all_of(output.begin(), output.end(), [&](unsigned char value) {
            return value == (business_rc == 0 ? 0x42 : 0x71);
        }));
        EXPECT_EQ(run.runner.drain_calls, 2);
        EXPECT_EQ(run.runner.empty_drain_calls, 0);
        EXPECT_FALSE(run.runner.native_runs_outstanding());
    }
}

TEST(DiagnosticCapacityRefusalPhaseTest, OutputDeliveryDoesNotChangeObservationOrCleanup) {
    for (int delivery : {-1, 0, 1}) {
        for (int business_rc : {0, -735}) {
            SCOPED_TRACE(delivery);
            SCOPED_TRACE(business_rc);
            PreparedRun run;
            run.runner.refuse_for_capacity = false;
            run.runner.consume_drain_once = true;
            run.runner.drain_rc = business_rc;
            ASSERT_TRUE(run.runner.stand_up());
            ASSERT_EQ(run.prepare(CallConfig{}), 0);
            std::array<unsigned char, 64> output;
            output.fill(0x71);
            void *const backing = run.runner.allocate_tensor(output.size());
            ASSERT_NE(backing, nullptr);
            std::memset(backing, 0x42, output.size());
            run.runner.set_retained_temp_buffer(0, backing, output.size());
            run.state()->runtime->tensor_leases().push_back(
                TensorLease{output.data(), backing, output.size(), false, true, TensorReleaseKind::BufferNoop}
            );
            ASSERT_EQ(run.launch(), 0);
            ASSERT_EQ(simpler_wait_run_completion(&run.runner, run.storage.handle()), 0);
            EXPECT_EQ(output.front(), 0x71);
            EXPECT_EQ(simpler_wait_run(&run.runner, run.storage.handle()), business_rc);
            const int rc = delivery < 0 ?
                               run.finalize() :
                               simpler_finalize_run_with_delivery(&run.runner, run.storage.handle(), delivery);
            EXPECT_EQ(rc, business_rc);
            EXPECT_TRUE(std::all_of(output.begin(), output.end(), [&](unsigned char value) {
                return value == (delivery != 0 && business_rc == 0 ? 0x42 : 0x71);
            }));
            EXPECT_EQ(run.runner.drain_calls, 1);
            EXPECT_EQ(run.runner.empty_drain_calls, 0);
            EXPECT_FALSE(run.runner.native_runs_outstanding());
            std::fprintf(
                stderr, "SIM_FINALIZE_DELIVERY flag=%d native=%d finalize=%d host=%u drain=%d\n", delivery, business_rc,
                rc, output.front(), run.runner.drain_calls
            );
        }
    }
}

TEST(DiagnosticCapacityRefusalPhaseTest, WithheldUnsubmittedDeliveryKeepsTheCleanRefusalProof) {
    PreparedRun run;
    ASSERT_TRUE(run.runner.stand_up());
    ASSERT_EQ(run.prepare(CallConfig{}), 0);
    EXPECT_EQ(
        simpler_finalize_run_with_delivery(&run.runner, run.storage.handle(), 2), PTO_RUNTIME_ERR_INVALID_ARGUMENT
    );
    EXPECT_EQ(run.phase(), NativeRunPhase::Prepared);
    EXPECT_TRUE(run.runner.claim_held_by(run.storage.handle()));
    ASSERT_EQ(run.launch(), SIMPLER_RUNTIME_ERR_DIAGNOSTIC_CAPACITY_FULL);
    EXPECT_EQ(run.phase(), NativeRunPhase::Prepared);
    EXPECT_EQ(simpler_finalize_run_with_delivery(&run.runner, run.storage.handle(), 0), 0);
    EXPECT_TRUE(consume_released_preparation(run.storage.handle(), get_runtime_size(), 9100));
    EXPECT_EQ(run.runner.drain_calls, 0);
    EXPECT_FALSE(run.runner.native_runs_outstanding());
}

TEST(DiagnosticCapacityRefusalPhaseTest, ClockExportFollowsItsRootAfterSlotReuse) {
    char path[] = "/tmp/simpler-sim-clock-XXXXXX";
    ASSERT_NE(mkdtemp(path), nullptr);
    const std::filesystem::path directory(path);
    static SimplerHostLogState log_state{};
    log_state.threshold = static_cast<int>(simpler::log::LogLevel::TIMING);
    log_state.sink_owner_pid = log_state.sink_process_pid = getpid();
    log_state.sink_context = &log_state;
    log_state.sink_enqueue = [](void *, SimplerHostLogState *state, const char *record, uint32_t size) {
        std::ofstream file(
            std::filesystem::path(state->log_directory) / ("host." + std::to_string(getpid()) + ".log"), std::ios::app
        );
        file.write(record, size);
        file.close();
        const bool written = static_cast<bool>(file);
        __atomic_fetch_sub(&state->sink_producer_state, UINT64_C(1), __ATOMIC_RELEASE);
        return written ? 1 : 0;
    };
    std::memcpy(log_state.log_directory, path, sizeof(path));
    log_state.log_directory_bound = 1;
    pthread_key_t key;
    ASSERT_EQ(pthread_key_create(&key, std::free), 0);
    log_state.trace_key_plus_one = static_cast<uint64_t>(key) + 1;
    ASSERT_EQ(simpler_host_log_bind_state(&log_state), 0);

    PreparedRun run;
    run.runner.refuse_for_capacity = false;
    run.runner.consume_drain_once = run.runner.clock_timing = true;
    ASSERT_TRUE(run.runner.stand_up());
    CallConfig config{};
    std::memcpy(config.output_prefix, path, sizeof(path));
    ASSERT_EQ(run.prepare(config), 0);
    const uint64_t invocation = run.state()->trace_inv;
    const auto runner_records = [&] {
        std::ifstream process_log(directory / ("host." + std::to_string(getpid()) + ".log"));
        size_t count = 0;
        for (std::string line; std::getline(process_log, line);) {
            if (line.find(" name=chip.run.runner_run ") != std::string::npos &&
                line.find(" inv=" + std::to_string(invocation) + " ") != std::string::npos)
                ++count;
        }
        return count;
    };
    DfxRunConfig dfx;
    dfx.chip_swimlane_level = ChipSwimlaneLevel::SCHED_PHASES;
    dfx.output_prefix = path;
    (void)run.runner.host_phase_pool_arm(0, false);
    run.runner.begin_host_phase_run(0, dfx);
    EXPECT_FALSE(run.runner.host_clock_alignment_log_required(0));
    ASSERT_NE(run.runner.host_phase_pool_arm(0, true), nullptr);
    EXPECT_FALSE(run.runner.host_clock_alignment_log_required(0));
    run.runner.host_phase_pool_finish(0, 0, invocation);
    ASSERT_TRUE(run.runner.host_clock_alignment_log_required(0));
    ASSERT_EQ(run.launch(), 0);
    ASSERT_EQ(simpler_wait_run_completion(&run.runner, run.storage.handle()), 0);
    EXPECT_EQ(run.completion_rc(), 0);
    EXPECT_EQ(runner_records(), 1u);
    const auto target = directory / ("host_clock_alignment." + std::to_string(getpid()) + ".log");
    EXPECT_FALSE(std::filesystem::exists(target));

    RuntimeStorage successor;
    NativeRunDescriptor descriptor = run.state()->descriptor;
    descriptor.run_epoch = 9200;
    descriptor.generation++;
    descriptor.dispatch_id++;
    CallConfig ordinary{};
    ASSERT_EQ(simpler_prepare_run(&run.runner, successor.handle(), kCallableId, nullptr, &ordinary, &descriptor), 0);
    ASSERT_FALSE(run.runner.host_clock_alignment_log_required(0));
    const uint64_t successor_invocation =
        static_cast<NativeRunContext<SimDeviceRunnerBase> *>(successor.handle())->trace_inv;
    EXPECT_EQ(run.finalize(), 0);
    EXPECT_EQ(runner_records(), 1u);
    std::ifstream file(target);
    EXPECT_TRUE(file);
    bool root = false, execution = false, wall = false;
    std::string exported;
    for (std::string line; std::getline(file, line);) {
        EXPECT_NE(line.find(" inv=" + std::to_string(invocation) + " "), std::string::npos);
        EXPECT_EQ(line.find(" inv=" + std::to_string(successor_invocation) + " "), std::string::npos);
        root |= line.find(" name=chip.run ") != std::string::npos;
        execution |= line.find(" name=chip.run.runner_run ") != std::string::npos;
        wall |= line.find(" name=chip.run.runner_run.device_wall ") != std::string::npos;
        exported += line + '\n';
    }
    EXPECT_TRUE(root && execution && wall);
    EXPECT_EQ(simpler_finalize_run(&run.runner, successor.handle()), 0);
    if (std::filesystem::exists(target)) {
        std::ifstream unchanged(target);
        EXPECT_EQ(std::string(std::istreambuf_iterator<char>(unchanged), {}), exported);
    }
    EXPECT_EQ(log_state.sink_producer_state, 0u);
    EXPECT_EQ(log_state.pending_record_count, 0u);
    EXPECT_EQ(log_state.dropped_record_count, 0u);
    std::fprintf(
        stderr, "SIM_CLOCK_EXPORT path=%s inv=%llu successor_inv=%llu\n", target.c_str(),
        static_cast<unsigned long long>(invocation), static_cast<unsigned long long>(successor_invocation)
    );
}

namespace {
void check_sim_finalize_prefix_allocation_failure(bool launched, bool delivery_entry) {
    PreparedRun run;
    ASSERT_TRUE(run.runner.stand_up());
    CallConfig config = diagnostic_config();
    config.enable_scope_stats = 0;
    const std::string prefix(160, 'p');
    std::memcpy(config.output_prefix, prefix.c_str(), prefix.size() + 1);
    ASSERT_EQ(run.prepare(config), 0);
    if (launched) {
        run.runner.refuse_for_capacity = false;
        ASSERT_EQ(run.launch(), 0);
    }
    const auto phase = run.phase();
    const auto completion = run.completion_rc();
    const auto acceptance = run.acceptance();
    const auto *prepared_owner = run.state()->prepared_execution.get();
    const auto *active_owner = run.state()->active_execution.get();
    const auto descriptor = run.state()->descriptor;
    const auto launches = run.runner.launch_calls;
    ASSERT_TRUE(run.runner.claim_held_by(run.storage.handle()));
    std::vector<unsigned char> carrier(get_runtime_size());
    std::memcpy(carrier.data(), run.storage.handle(), carrier.size());
    const auto finalize = [&] {
        return delivery_entry ? simpler_finalize_run_with_delivery(&run.runner, run.storage.handle(), 1) :
                                run.finalize();
    };
    auto &fault = finalize_prefix_test::prefix_allocation_fault;
    fault = {true, 0, 0};
    finalize_prefix_test::copyback_calls = 0;
    bool escaped = false;
    int rc = PTO_RUNTIME_ERR_INTERNAL;
    try {
        rc = finalize();
    } catch (const std::bad_alloc &) {
        escaped = true;
    }
    const bool fault_unconsumed = std::exchange(fault.armed, false);
    std::printf(
        "FINALIZE_PREFIX backend=sim launched=%d delivery_entry=%d consumed=%zu escaped=%d rc=%d bytes=%zu\n", launched,
        delivery_entry, fault.consumed, escaped, rc, fault.bytes
    );
    EXPECT_FALSE(fault_unconsumed);
    EXPECT_EQ(fault.consumed, 1u);
    EXPECT_GE(fault.bytes, prefix.size() + 1);
    EXPECT_FALSE(escaped);
    EXPECT_EQ(rc, PTO_RUNTIME_ERR_INTERNAL);
    ASSERT_EQ(run.state()->magic, NativeRunContext<SimDeviceRunnerBase>::kMagic);
    EXPECT_EQ(std::memcmp(carrier.data(), run.storage.handle(), carrier.size()), 0);
    EXPECT_EQ(run.phase(), phase);
    EXPECT_EQ(run.completion_rc(), completion);
    EXPECT_EQ(run.acceptance(), acceptance);
    EXPECT_EQ(run.state()->prepared_execution.get(), prepared_owner);
    EXPECT_EQ(run.state()->active_execution.get(), active_owner);
    EXPECT_TRUE(run.runner.claim_held_by(run.storage.handle()));
    EXPECT_EQ(run.runner.launch_calls, launches);
    EXPECT_EQ(finalize_prefix_test::copyback_calls, 0u);

    ASSERT_EQ(finalize(), 0);
    uint64_t magic = 1;
    std::memcpy(&magic, run.storage.handle(), sizeof(magic));
    EXPECT_EQ(magic, 0u);
    EXPECT_FALSE(run.runner.native_run_active());
    EXPECT_EQ(finalize_prefix_test::copyback_calls, 1u);
    RuntimeStorage successor;
    auto next = descriptor;
    ++next.run_id;
    ++next.generation;
    ++next.dispatch_id;
    ++next.run_epoch;
    ASSERT_EQ(simpler_prepare_run(&run.runner, successor.handle(), kCallableId, nullptr, &config, &next), 0);
    run.runner.refuse_for_capacity = false;
    ASSERT_EQ(simpler_launch_run(&run.runner, successor.handle()), 0);
    EXPECT_EQ(simpler_finalize_run(&run.runner, successor.handle()), 0);
    EXPECT_FALSE(run.runner.native_run_active());
}
}  // namespace

TEST(DiagnosticCapacityRefusalPhaseTest, FinalizePrefixAllocationFailureKeepsPreparedOwnerRetryable) {
    for (bool delivery_entry : {false, true}) {
        SCOPED_TRACE(delivery_entry);
        check_sim_finalize_prefix_allocation_failure(false, delivery_entry);
    }
}

TEST(DiagnosticCapacityRefusalPhaseTest, FinalizePrefixAllocationFailureKeepsLaunchedOwnerRetryable) {
    for (bool delivery_entry : {false, true}) {
        SCOPED_TRACE(delivery_entry);
        check_sim_finalize_prefix_allocation_failure(true, delivery_entry);
    }
}
#if defined(__linux__)
TEST(SimRunBindingFailure, WaitFailureKeepsTheRunDrainable) {
    PreparedRun run;
    ASSERT_TRUE(run.runner.stand_up());
    ASSERT_EQ(run.prepare(), 0);
    run.runner.refuse_for_capacity = false;
    ASSERT_EQ(run.launch(), 0);
    reject_device_binding = true;
    const int rc = simpler_wait_run(&run.runner, run.storage.handle());
    reject_device_binding = false;
    EXPECT_EQ(rc, SIMPLER_NATIVE_RUN_WAIT_INCOMPLETE);
    EXPECT_FALSE(run.state()->result_observed);
    EXPECT_FALSE(run.state()->physical_complete);
    EXPECT_EQ(run.phase(), NativeRunPhase::Running);
    EXPECT_EQ(run.runner.drain_calls, 0);
    EXPECT_TRUE(run.runner.claim_held_by(run.storage.handle()));
    EXPECT_EQ(run.finalize(), 0);
    EXPECT_EQ(run.runner.drain_calls, 1);
    EXPECT_FALSE(run.runner.native_run_active());
}

TEST(SimRunBindingFailure, FinalizeFailurePreservesTheCarrierAndClaimForRetry) {
    for (bool already_drained : {false, true}) {
        PreparedRun run;
        ASSERT_TRUE(run.runner.stand_up());
        ASSERT_EQ(run.prepare(), 0);
        run.runner.refuse_for_capacity = false;
        ASSERT_EQ(run.launch(), 0);
        if (already_drained) ASSERT_EQ(simpler_wait_run(&run.runner, run.storage.handle()), 0);
        reject_device_binding = true;
        const int rc = run.finalize();
        reject_device_binding = false;
        EXPECT_EQ(rc, PTO_RUNTIME_ERR_INTERNAL);
        uint64_t tag = 0;
        std::memcpy(&tag, run.storage.handle(), sizeof(tag));
        const bool intact = tag == kNativeRunContextMagic;
        EXPECT_TRUE(intact);
        EXPECT_TRUE(run.runner.claim_held_by(run.storage.handle()));
        EXPECT_EQ(run.runner.drain_calls, already_drained ? 1 : 0);
        if (intact) EXPECT_EQ(run.finalize(), 0);
        EXPECT_EQ(run.runner.drain_calls, 1);
        EXPECT_FALSE(run.runner.native_run_active());
    }
}
#endif
