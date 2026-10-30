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
/*
 * The native-run operation lease and the finalization protocol, driven directly against ChipWorker.
 *
 * These are the properties no hardware run can show, because a run that happens to be fast enough
 * satisfies them by accident: which thread owns a finalization, what a second one is told, whether a
 * completed wait is repeated, and what the worker does with a run whose teardown left an ambiguous
 * tag. The runtime entries are stubs here, so each of those can be produced on demand rather than
 * waited for.
 */
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <dlfcn.h>

#include <gtest/gtest.h>

#define private public
#define protected public
// ChipWorker holds its run lane by unique_ptr, so the lane's definition has to be visible for the
// worker to be destructible here.
#include "chip_run_lane.h"
#include "chip_worker.h"
#include "runtime_module.h"
#undef private
#undef protected

#include "native_run_storage_tag.h"

namespace {

TEST(NativeRunOperationLeaseTest, RunStorageRetainsItsModuleAcrossMoves) {
    auto module = std::make_shared<RuntimeModule>(nullptr);
    std::weak_ptr<RuntimeModule> weak = module;
    ChipWorker::RuntimeStorage first(64, alignof(std::max_align_t), module);
    void *address = first.data();
    module.reset();
    EXPECT_FALSE(weak.expired());
    ChipWorker::RuntimeStorage second(std::move(first));
    EXPECT_EQ(first.data(), nullptr);
    EXPECT_EQ(second.data(), address);
    EXPECT_FALSE(weak.expired());
    first = std::move(second);
    EXPECT_EQ(second.data(), nullptr);
    EXPECT_EQ(first.data(), address);
    first = ChipWorker::RuntimeStorage{};
    EXPECT_TRUE(weak.expired());
}

// What each stub should do on its next call. Plain globals because one case drives one worker.
std::mutex g_mu;
std::condition_variable g_cv;
bool g_wait_entered = false;
bool g_wait_released = true;
int g_wait_rc = 0;
bool g_wait_throws = false;
int g_finalize_rc = 0;
bool g_finalize_throws = false;
bool g_finalize_destroys = true;
bool g_finalize_corrupts = false;
int g_wait_calls = 0;
int g_finalize_calls = 0;
int g_poll_rc = SIMPLER_NATIVE_RUN_POLL_NOT_READY;
int g_launch_rc = 0;
// The joined launch and the capability query both dereference two carriers, so both can be held
// open here while another thread tries to destroy one of them.
std::mutex g_join_mu;
std::condition_variable g_join_cv;
bool g_join_entered = false;
bool g_join_released = true;
int g_join_rc = 0;
int g_join_calls = 0;
bool g_query_entered = false;
bool g_query_released = true;
int g_query_calls = 0;
int g_device_finalize_rc = 0;
int g_device_finalize_calls = 0;
int g_destroy_context_calls = 0;
int g_comm_free_calls = 0;

int stub_release_domain(void *, uint64_t, size_t, uint32_t) {
    ++g_comm_free_calls;
    return 0;
}
int stub_release_global_domain(uint64_t) {
    ++g_comm_free_calls;
    return 0;
}
int stub_destroy_comm(void *) {
    ++g_comm_free_calls;
    return 0;
}

int prepare_run(void *, void *runtime, int32_t, const void *, const CallConfig *, const NativeRunDescriptor *) {
    // The carrier the platform would place here; only its tag matters to the worker.
    uint64_t tag = kNativeRunContextMagic;
    std::memcpy(runtime, &tag, sizeof(tag));
    return 0;
}

int launch_run(void *, void *) { return g_launch_rc; }

int poll_run(void *, void *) { return g_poll_rc; }

int wait_run(void *, void *) {
    {
        std::unique_lock<std::mutex> lk(g_mu);
        ++g_wait_calls;
        g_wait_entered = true;
        g_cv.notify_all();
        g_cv.wait(lk, [] {
            return g_wait_released;
        });
    }
    if (g_wait_throws) throw std::runtime_error("stub wait failed");
    return g_wait_rc;
}

int finalize_run(void *, void *runtime) {
    ++g_finalize_calls;
    if (g_finalize_destroys) {
        uint64_t cleared = 0;
        std::memcpy(runtime, &cleared, sizeof(cleared));
    } else if (g_finalize_corrupts) {
        uint64_t garbage = kNativeRunContextMagic ^ UINT64_C(0xff);
        std::memcpy(runtime, &garbage, sizeof(garbage));
    }
    if (g_finalize_throws) throw std::runtime_error("stub finalize failed");
    return g_finalize_rc;
}

int supports_successor(void *) { return 1; }

int launch_run_joined(void *, void *, void *predecessor_runtime) {
    {
        std::unique_lock<std::mutex> lk(g_join_mu);
        ++g_join_calls;
        g_join_entered = true;
        g_join_cv.notify_all();
        g_join_cv.wait(lk, [] {
            return g_join_released;
        });
    }
    // Read after the release, which is the dereference a destruction underneath this call would
    // have invalidated: the predecessor's carrier must still be the one this join was handed.
    uint64_t tag = 0;
    std::memcpy(&tag, predecessor_runtime, sizeof(tag));
    EXPECT_EQ(tag, kNativeRunContextMagic) << "the predecessor's carrier was destroyed under a joined launch";
    return g_join_rc;
}

int supports_joined_diagnostic_launch(void *, void *, void *predecessor_runtime) {
    {
        std::unique_lock<std::mutex> lk(g_join_mu);
        ++g_query_calls;
        g_query_entered = true;
        g_join_cv.notify_all();
        g_join_cv.wait(lk, [] {
            return g_query_released;
        });
    }
    uint64_t tag = 0;
    std::memcpy(&tag, predecessor_runtime, sizeof(tag));
    EXPECT_EQ(tag, kNativeRunContextMagic) << "the predecessor's carrier was destroyed under a capability query";
    return 1;
}

int stub_finalize_device(void *) {
    ++g_device_finalize_calls;
    return g_device_finalize_rc;
}

void stub_destroy_device_context(void *) { ++g_destroy_context_calls; }

void prime(ChipWorker &worker, uint32_t depth = 2) {
    {
        std::lock_guard<std::mutex> lk(g_mu);
        g_wait_entered = false;
        g_wait_released = true;
    }
    {
        std::lock_guard<std::mutex> lk(g_join_mu);
        g_join_entered = false;
        g_join_released = true;
        g_query_entered = false;
        g_query_released = true;
    }
    g_wait_rc = 0;
    g_wait_throws = false;
    g_finalize_rc = 0;
    g_finalize_throws = false;
    g_finalize_destroys = true;
    g_finalize_corrupts = false;
    g_wait_calls = 0;
    g_finalize_calls = 0;
    g_poll_rc = SIMPLER_NATIVE_RUN_POLL_NOT_READY;
    g_launch_rc = 0;
    g_join_rc = 0;
    g_join_calls = 0;
    g_query_calls = 0;
    g_device_finalize_rc = 0;
    g_device_finalize_calls = 0;
    g_destroy_context_calls = 0;
    g_comm_free_calls = 0;
    worker.initialized_ = true;
    // The stubs ignore it, but teardown refuses to run without one, which is the production
    // guard against closing a worker that never reached a device.
    static int device_context_placeholder = 0;
    worker.device_ctx_ = &device_context_placeholder;
    worker.destroy_device_context_fn_ = stub_destroy_device_context;
    // A reference to this executable supplies the runtime owner without another DSO.
    if (worker.runtime_module_ == nullptr) worker.runtime_module_ = std::make_shared<RuntimeModule>(nullptr);
    EXPECT_NE(worker.runtime_module_, nullptr);
    worker.pipeline_contract_ = {PTO_PIPELINE_CONTRACT_ABI_VERSION, 0, depth, {}};
    // One generation per granted set, as `init` sizes it: an unsized filter has no entry for
    // slot 0, and `is_admissible` refuses a lease outside it, so every preparation would be
    // refused as stale before reaching the stubs below.
    worker.pipeline_generations_.resize(depth);
    for (uint32_t slot = 0; slot < depth; ++slot) {
        worker.runtime_bufs_.emplace_back(64, alignof(std::max_align_t), worker.runtime_module_);
    }
    worker.endpoint_failures_.assign(worker.runtime_bufs_.size(), {});
    worker.prepare_run_fn_ = prepare_run;
    worker.launch_run_fn_ = launch_run;
    worker.poll_run_fn_ = poll_run;
    worker.wait_run_fn_ = wait_run;
    worker.finalize_run_fn_ = finalize_run;
    worker.supports_concurrent_native_prepare_fn_ = supports_successor;
}

// What the joined launch and the diagnostic capability query additionally need: both are gated on
// a launch depth above one and on the backend answering that it can order two runs.
void prime_joined(ChipWorker &worker) {
    worker.launch_depth_ = 2;
    worker.launch_run_joined_fn_ = launch_run_joined;
    worker.supports_joined_native_launch_fn_ = supports_successor;
    worker.supports_joined_diagnostic_launch_fn_ = supports_joined_diagnostic_launch;
    worker.retained_epoch_limit_granted_ = 3;
}

ChipWorkerNativeRun prepare(ChipWorker &worker, uint32_t slot, uint64_t generation) {
    ChipStorageTaskArgs args{};
    return worker.prepare_native_run_on_slot(
        1, &args, CallConfig{}, slot, generation, generation, generation, nullptr, 0, false, NativeRunProvenance{}
    );
}

void await_wait_entered() {
    std::unique_lock<std::mutex> lk(g_mu);
    ASSERT_TRUE(g_cv.wait_for(lk, std::chrono::seconds(5), [] {
        return g_wait_entered;
    }));
}

void release_wait() {
    std::lock_guard<std::mutex> lk(g_mu);
    g_wait_released = true;
    g_cv.notify_all();
}

void hold_wait() {
    std::lock_guard<std::mutex> lk(g_mu);
    g_wait_released = false;
    g_wait_entered = false;
}

void hold_join() {
    std::lock_guard<std::mutex> lk(g_join_mu);
    g_join_released = false;
    g_join_entered = false;
    g_query_released = false;
    g_query_entered = false;
}

void await_join_entered() {
    std::unique_lock<std::mutex> lk(g_join_mu);
    ASSERT_TRUE(g_join_cv.wait_for(lk, std::chrono::seconds(5), [] {
        return g_join_entered;
    }));
}

void await_query_entered() {
    std::unique_lock<std::mutex> lk(g_join_mu);
    ASSERT_TRUE(g_join_cv.wait_for(lk, std::chrono::seconds(5), [] {
        return g_query_entered;
    }));
}

void release_join() {
    std::lock_guard<std::mutex> lk(g_join_mu);
    g_join_released = true;
    g_query_released = true;
    g_join_cv.notify_all();
}

/**
 * A parked waiter thread that is always released and joined, however the case ends.
 *
 * A case that holds a waiter inside the stub's `wait` and then fails a fatal assertion returns
 * from its body with that thread still joinable, which calls `std::terminate` and destroys the
 * finding it was about to report. An exception thrown by anything between the two — a refused
 * preparation, for one — does the same. Both are cleanup, so both belong in a destructor.
 *
 * The waiter's own exception is carried out rather than escaping its thread, and `rethrow` is
 * what a case calls to make it the case's failure.
 */
class ParkedWaiter {
public:
    template <typename Fn>
    explicit ParkedWaiter(Fn &&body) {
        // Started in the body, not in a member initializer. A thread launched from an initializer
        // begins running while the members declared after it are still unconstructed, and the
        // very first thing this one may do is assign `failure_` — so the only safe order is every
        // member first, the thread last.
        thread_ = std::thread([this, body = std::forward<Fn>(body)] {
            try {
                body();
            } catch (...) {
                failure_ = std::current_exception();
            }
        });
    }

    ParkedWaiter(const ParkedWaiter &) = delete;
    ParkedWaiter &operator=(const ParkedWaiter &) = delete;

    ~ParkedWaiter() { settle(); }

    /** Release the stub and join, once. Safe to call again from the destructor. */
    void settle() {
        release_wait();
        if (thread_.joinable()) thread_.join();
    }

    /** The waiter's own exception, if it had one. Call after `settle`: the join is what makes
     *  the thread's write to it visible here. */
    void rethrow() const {
        if (failure_) std::rethrow_exception(failure_);
    }

private:
    std::exception_ptr failure_;
    std::thread thread_;
};

}  // namespace

// A refused finalization must not undo the completion its own wait established, or the next
// attempt waits again and reports whatever that second wait says instead of the first error.
TEST(NativeRunOperationLeaseTest, RefusedFinalizeKeepsItsWaitOutcomeAndDoesNotWaitTwice) {
    ChipWorker worker;
    prime(worker);
    const ChipWorkerNativeRun run = prepare(worker, 0, 1);
    worker.launch_native_run(run);

    // The teardown refuses without destroying the carrier, which is the context-refusal shape.
    g_wait_rc = -7;
    g_finalize_rc = -5;
    g_finalize_destroys = false;
    EXPECT_THROW(worker.finalize_native_run(run), std::runtime_error);
    EXPECT_EQ(g_wait_calls, 1);
    {
        std::lock_guard<std::mutex> lk(worker.native_run_mu_);
        const auto it = worker.native_runs_.find(run.run_epoch);
        ASSERT_NE(it, worker.native_runs_.end());
        EXPECT_EQ(it->second->phase, ChipWorker::NativeRunPhase::REAPED);
        EXPECT_TRUE(it->second->pending_rc_set);
        EXPECT_EQ(it->second->pending_rc, -7);
        EXPECT_FALSE(it->second->finalize_reserved);
    }

    // The retry succeeds at the teardown and reports the error the first wait established.
    g_finalize_rc = 0;
    g_finalize_destroys = true;
    EXPECT_THROW(worker.finalize_native_run(run), std::runtime_error);
    EXPECT_EQ(g_wait_calls, 1) << "the completed wait was repeated";
    EXPECT_EQ(g_finalize_calls, 2);
}

// Two finalizations of one run: one owns it, the other is told so and never reaches the runtime.
TEST(NativeRunOperationLeaseTest, SecondFinalizerIsRefusedAndNeverEntersTheRuntime) {
    ChipWorker worker;
    prime(worker);
    const ChipWorkerNativeRun run = prepare(worker, 0, 1);
    worker.launch_native_run(run);

    hold_wait();
    std::atomic<bool> first_threw{false};
    std::thread owner([&] {
        try {
            worker.finalize_native_run(run);
        } catch (...) {
            first_threw = true;
        }
    });
    await_wait_entered();

    EXPECT_THROW(worker.finalize_native_run(run), std::runtime_error);
    EXPECT_EQ(g_finalize_calls, 0) << "the refused finalizer reached the runtime";

    release_wait();
    owner.join();
    EXPECT_FALSE(first_threw.load());
    EXPECT_EQ(g_finalize_calls, 1);
    EXPECT_EQ(g_wait_calls, 1);
}

// A poll while another operation holds the lease reports no progress rather than entering
// concurrently or blocking; a wait on the same run waits for the lease instead.
TEST(NativeRunOperationLeaseTest, PollReportsNoProgressWhileAnotherOperationHoldsTheLease) {
    ChipWorker worker;
    prime(worker);
    const ChipWorkerNativeRun run = prepare(worker, 0, 1);
    worker.launch_native_run(run);

    hold_wait();
    std::thread waiter([&] {
        worker.wait_native_run(run);
    });
    await_wait_entered();

    EXPECT_FALSE(worker.poll_native_run(run));

    release_wait();
    waiter.join();
    EXPECT_EQ(g_wait_calls, 1);
    // Completion is established, so a later poll answers from the record without a device call.
    g_poll_rc = SIMPLER_NATIVE_RUN_POLL_ERROR;
    EXPECT_TRUE(worker.poll_native_run(run));
}

// §2.3.6(f) / contract row 24: later admissions grow the run table while a waiter sleeps inside
// the runtime, and the waiter is unaffected.
//
// This is a property of *where* the entry lives, so the case makes the rehash actually happen
// rather than arguing that it would be harmless. `native_runs_` maps epoch to a `unique_ptr`, and
// a waiter holds the pointee for the whole of its wait; admitting enough later runs forces the
// container to grow and relocate every node. The assertion is that the address the waiter is
// holding is still the entry its epoch resolves to afterwards, and that the wait it was in the
// middle of still returns its own outcome.
//
// The waiter is parked inside the backend's `wait` for the whole window, which is the state the
// row is about — a waiter between the lease and the completion, not one that happens to be
// scheduled elsewhere. The admissions are made from this thread, which does not weaken that: the
// waiter is genuinely inside the runtime for every one of them.
//
// `ParkedWaiter` owns the release and the join, so a fatal assertion or a throwing preparation
// below still unparks the thread instead of reaching the end of the case with it joinable.
TEST(NativeRunOperationLeaseTest, AdmittingLaterRunsRehashesTheTableUnderASleepingWaiter) {
    // One slot per run this case admits, since a preparation is refused on a slot whose
    // generation filter has no entry for it.
    constexpr uint32_t kDepth = 24;
    ChipWorker worker;
    prime(worker, kDepth);
    const ChipWorkerNativeRun run = prepare(worker, 0, 1);
    worker.launch_native_run(run);

    hold_wait();
    // Set before the waiter is released, which is anywhere from here on: the stub reads it on the
    // way out of its park.
    g_wait_rc = -7;
    ParkedWaiter waiter([&] {
        worker.wait_native_run(run);
    });
    await_wait_entered();

    const ChipWorker::NativeRunEntry *entry_before = nullptr;
    size_t buckets_before = 0;
    {
        std::lock_guard<std::mutex> lk(worker.native_run_mu_);
        const auto it = worker.native_runs_.find(run.run_epoch);
        ASSERT_NE(it, worker.native_runs_.end());
        entry_before = it->second.get();
        buckets_before = worker.native_runs_.bucket_count();
    }

    // Enough later admissions that the table must grow: libstdc++ keeps its load factor at one,
    // so a bucket count that never moved would mean no rehash happened and this case proved
    // nothing.
    //
    // Each one is launched as well as prepared, because that is the lifecycle the product
    // requires: `prepare_native_run_on_slot` admits a successor only while every run already in
    // the table is `LAUNCHED` or `REAPED`, so a queue of merely-prepared runs is refused at the
    // second one. Launching each is what makes these admissions real rather than a fixture that
    // the admission check would have to be relaxed to accept.
    std::vector<ChipWorkerNativeRun> later;
    for (uint32_t slot = 1; slot < kDepth; ++slot) {
        later.push_back(prepare(worker, slot, slot + 1));
        worker.launch_native_run(later.back());
    }
    {
        std::lock_guard<std::mutex> lk(worker.native_run_mu_);
        ASSERT_GT(worker.native_runs_.bucket_count(), buckets_before)
            << "the run table never rehashed, so this case did not reach the state row 24 is about";
        ASSERT_EQ(worker.native_runs_.size(), later.size() + 1);
        const auto it = worker.native_runs_.find(run.run_epoch);
        ASSERT_NE(it, worker.native_runs_.end()) << "the sleeping waiter's run left the table";
        EXPECT_EQ(it->second.get(), entry_before)
            << "a rehash moved the entry itself, so every waiter holding one is holding freed storage";
        EXPECT_EQ(it->second->ops_in_flight, 1u) << "the waiter's lease was lost across the rehash";
    }

    waiter.settle();
    ASSERT_NO_THROW(waiter.rethrow()) << "the parked waiter failed rather than returning its own outcome";
    EXPECT_EQ(g_wait_calls, 1) << "the waiter was made to wait a second time";
    {
        std::lock_guard<std::mutex> lk(worker.native_run_mu_);
        const auto it = worker.native_runs_.find(run.run_epoch);
        ASSERT_NE(it, worker.native_runs_.end());
        EXPECT_EQ(it->second.get(), entry_before);
        EXPECT_EQ(it->second->phase, ChipWorker::NativeRunPhase::REAPED)
            << "the waiter's own completion did not land on its own entry";
        EXPECT_TRUE(it->second->pending_rc_set);
        EXPECT_EQ(it->second->pending_rc, -7) << "the waiter's own outcome was not the one it recorded";
        EXPECT_EQ(it->second->ops_in_flight, 0u);
    }

    // The launched runs behind it are finalized here rather than left to teardown, so the
    // destructor has nothing to report and this case's own finding is the only thing in the log.
    // Their waits are the stub's, and the outcome under test has already been read above.
    g_wait_rc = 0;
    for (const ChipWorkerNativeRun &launched : later) {
        EXPECT_NO_THROW(worker.finalize_native_run(launched)) << "a later admission could not be retired";
    }
}

// A joined launch dereferences the predecessor's carrier too, so a finalization of that
// predecessor must wait for the launch to return rather than destroying it underneath. The join is
// held open inside the runtime while the other thread tries exactly that.
TEST(NativeRunOperationLeaseTest, JoinedLaunchHoldsThePredecessorAgainstConcurrentFinalize) {
    ChipWorker worker;
    prime(worker);
    prime_joined(worker);
    const ChipWorkerNativeRun ahead = prepare(worker, 0, 1);
    worker.launch_native_run(ahead);
    const ChipWorkerNativeRun behind = prepare(worker, 1, 2);

    hold_join();
    std::atomic<bool> joined{false};
    std::thread launcher([&] {
        joined = worker.launch_native_run_joined(behind, ahead);
    });
    await_join_entered();

    // The predecessor's finalization has to drain this join first. It is started here and must
    // still be outside the runtime when the join is released.
    std::atomic<bool> finalize_returned{false};
    std::thread closer([&] {
        try {
            worker.finalize_native_run(ahead);
        } catch (...) {}
        finalize_returned = true;
    });
    // Nothing proves a negative by waiting forever, so this waits long enough that a finalization
    // free to proceed would have destroyed the carrier, then checks it did not.
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_EQ(g_finalize_calls, 0) << "a finalization entered the runtime under a live joined launch";
    EXPECT_FALSE(finalize_returned.load());

    release_join();
    launcher.join();
    closer.join();
    EXPECT_TRUE(joined.load());
    EXPECT_EQ(g_join_calls, 1);
    EXPECT_EQ(g_finalize_calls, 1) << "the drained finalization never ran";
}

// A predecessor that is no longer executing is a decline, not a failure, and never reaches the
// backend.
TEST(NativeRunOperationLeaseTest, JoinedLaunchDeclinesARetiredPredecessor) {
    ChipWorker worker;
    prime(worker);
    prime_joined(worker);
    const ChipWorkerNativeRun ahead = prepare(worker, 0, 1);
    worker.launch_native_run(ahead);
    const ChipWorkerNativeRun behind = prepare(worker, 1, 2);

    worker.note_native_run_completion(ahead.run_epoch, 0);
    EXPECT_FALSE(worker.launch_native_run_joined(behind, ahead));
    EXPECT_EQ(g_join_calls, 0) << "a declined join still entered the backend";
    {
        std::lock_guard<std::mutex> lk(worker.native_run_mu_);
        const auto it = worker.native_runs_.find(behind.run_epoch);
        ASSERT_NE(it, worker.native_runs_.end());
        EXPECT_EQ(it->second->phase, ChipWorker::NativeRunPhase::PREPARED) << "a decline changed the run's state";
        EXPECT_EQ(it->second->ops_in_flight, 0u);
    }
}

// The diagnostic capability query dereferences both carriers, so it holds both leases for the
// duration of the backend call — a finalization cannot destroy either one under it.
TEST(NativeRunOperationLeaseTest, CapabilityQueryHoldsBothCarriersAgainstConcurrentFinalize) {
    ChipWorker worker;
    prime(worker);
    prime_joined(worker);
    const ChipWorkerNativeRun ahead = prepare(worker, 0, 1);
    worker.launch_native_run(ahead);
    const ChipWorkerNativeRun behind = prepare(worker, 1, 2);

    hold_join();
    std::atomic<bool> answered{false};
    std::thread asker([&] {
        answered = worker.supports_joined_diagnostic_launch(behind, ahead);
    });
    await_query_entered();

    std::atomic<bool> finalize_returned{false};
    std::thread closer([&] {
        try {
            worker.finalize_native_run(ahead);
        } catch (...) {}
        finalize_returned = true;
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_EQ(g_finalize_calls, 0) << "a finalization entered the runtime under a live capability query";
    EXPECT_FALSE(finalize_returned.load());

    release_join();
    asker.join();
    closer.join();
    EXPECT_TRUE(answered.load());
    EXPECT_EQ(g_query_calls, 1);
    EXPECT_EQ(g_finalize_calls, 1);
}

// The query is a scheduling question asked once a round: a run another operation is inside is a
// refusal for this round, never a wait, and never a backend call.
TEST(NativeRunOperationLeaseTest, CapabilityQueryRefusesRatherThanWaiting) {
    ChipWorker worker;
    prime(worker);
    prime_joined(worker);
    const ChipWorkerNativeRun ahead = prepare(worker, 0, 1);
    worker.launch_native_run(ahead);
    const ChipWorkerNativeRun behind = prepare(worker, 1, 2);

    hold_wait();
    std::thread waiter([&] {
        worker.wait_native_run(ahead);
    });
    await_wait_entered();

    const auto asked_at = std::chrono::steady_clock::now();
    EXPECT_FALSE(worker.supports_joined_diagnostic_launch(behind, ahead));
    EXPECT_LT(std::chrono::steady_clock::now() - asked_at, std::chrono::seconds(1)) << "the query waited for a lease";
    EXPECT_EQ(g_query_calls, 0) << "a refused query still entered the backend";

    release_wait();
    waiter.join();
}

// Stale identity is refused by the same validation every operation uses, so a recycled slot or a
// reused epoch cannot be handed to the backend as if it named this run.
TEST(NativeRunOperationLeaseTest, CapabilityQueryRefusesStaleIdentity) {
    ChipWorker worker;
    prime(worker);
    prime_joined(worker);
    const ChipWorkerNativeRun ahead = prepare(worker, 0, 1);
    worker.launch_native_run(ahead);
    const ChipWorkerNativeRun behind = prepare(worker, 1, 2);

    ChipWorkerNativeRun stale_generation = behind;
    stale_generation.generation = behind.generation + 1;
    EXPECT_FALSE(worker.supports_joined_diagnostic_launch(stale_generation, ahead));

    ChipWorkerNativeRun stale_slot = behind;
    stale_slot.slot_id = ahead.slot_id;
    EXPECT_FALSE(worker.supports_joined_diagnostic_launch(stale_slot, ahead));

    ChipWorkerNativeRun unknown_epoch = behind;
    unknown_epoch.run_epoch = behind.run_epoch + 1000;
    EXPECT_FALSE(worker.supports_joined_diagnostic_launch(unknown_epoch, ahead));

    ChipWorkerNativeRun predecessor_stale = ahead;
    predecessor_stale.generation = ahead.generation + 1;
    EXPECT_FALSE(worker.supports_joined_diagnostic_launch(behind, predecessor_stale));

    EXPECT_EQ(g_query_calls, 0) << "a stale pair reached the backend";
    // The same pair, correctly named, is admitted — so the refusals above are about identity and
    // not about the fixture simply never being able to answer true.
    EXPECT_TRUE(worker.supports_joined_diagnostic_launch(behind, ahead));
    EXPECT_EQ(g_query_calls, 1);
}

// A teardown that leaves neither a destroyed nor a canonical tag is isolated, and a success is
// never reported for it.
TEST(NativeRunOperationLeaseTest, UnknownTeardownIsIsolatedAndNeverReportsSuccess) {
    ChipWorker worker;
    prime(worker);
    const ChipWorkerNativeRun run = prepare(worker, 0, 1);
    worker.launch_native_run(run);

    g_finalize_destroys = false;
    g_finalize_corrupts = true;
    EXPECT_THROW(worker.finalize_native_run(run), std::runtime_error);
    {
        std::lock_guard<std::mutex> lk(worker.native_run_mu_);
        const auto it = worker.native_runs_.find(run.run_epoch);
        ASSERT_NE(it, worker.native_runs_.end());
        EXPECT_TRUE(it->second->destruction_unknown);
    }
    // Nothing further reaches an isolated run, including another finalization.
    EXPECT_THROW(worker.finalize_native_run(run), std::runtime_error);
    EXPECT_EQ(g_finalize_calls, 1);
}

// A teardown that returns success without destroying the carrier is self-contradictory: it is
// reported as a cleanup failure and the run is isolated rather than offered for another attempt.
TEST(NativeRunOperationLeaseTest, NonDestroyingSuccessIsACleanupFailure) {
    ChipWorker worker;
    prime(worker);
    const ChipWorkerNativeRun run = prepare(worker, 0, 1);

    g_finalize_destroys = false;
    g_finalize_rc = 0;
    EXPECT_THROW(worker.finalize_native_run(run), std::runtime_error);
    std::lock_guard<std::mutex> lk(worker.native_run_mu_);
    const auto it = worker.native_runs_.find(run.run_epoch);
    ASSERT_NE(it, worker.native_runs_.end());
    EXPECT_TRUE(it->second->destruction_unknown);
}

// A wait that throws never reaches the teardown, so the carrier is intact and the run may be
// finalized again.
TEST(NativeRunOperationLeaseTest, WaitThrowLeavesTheRunFinalizableAgain) {
    ChipWorker worker;
    prime(worker);
    const ChipWorkerNativeRun run = prepare(worker, 0, 1);
    worker.launch_native_run(run);

    g_wait_throws = true;
    EXPECT_THROW(worker.finalize_native_run(run), std::runtime_error);
    EXPECT_EQ(g_finalize_calls, 0) << "the teardown ran after its wait threw";
    {
        std::lock_guard<std::mutex> lk(worker.native_run_mu_);
        const auto it = worker.native_runs_.find(run.run_epoch);
        ASSERT_NE(it, worker.native_runs_.end());
        EXPECT_FALSE(it->second->destruction_unknown);
        EXPECT_FALSE(it->second->finalize_reserved);
    }

    g_wait_throws = false;
    EXPECT_NO_THROW(worker.finalize_native_run(run));
    EXPECT_EQ(g_finalize_calls, 1);
}

// Teardown closes every run it can and isolates the rest, without letting a stub's exception
// leave a noexcept path and without leaving a lease behind that would block it.
TEST(NativeRunOperationLeaseTest, TeardownClosesEveryRunAndSwallowsForeignExceptions) {
    ChipWorker worker;
    prime(worker);
    const ChipWorkerNativeRun first = prepare(worker, 0, 1);
    worker.launch_native_run(first);
    const ChipWorkerNativeRun second = prepare(worker, 1, 2);

    g_finalize_throws = true;
    g_finalize_destroys = false;
    EXPECT_NO_THROW(worker.cleanup_native_runs_noexcept());
    EXPECT_EQ(g_finalize_calls, 2) << "teardown skipped a run it owned";
    std::lock_guard<std::mutex> lk(worker.native_run_mu_);
    for (const uint64_t run_epoch : {first.run_epoch, second.run_epoch}) {
        const auto it = worker.native_runs_.find(run_epoch);
        ASSERT_NE(it, worker.native_runs_.end());
        EXPECT_TRUE(it->second->destruction_unknown);
        EXPECT_FALSE(it->second->finalize_reserved);
        EXPECT_EQ(it->second->ops_in_flight, 0u);
    }
}

// The whole close, not just its sweep: a worker whose every run closed cleanly reaches the device
// teardown and the context destruction, and gives up its run storage.
TEST(NativeRunOperationLeaseTest, WholeCloseReleasesEverythingWhenEveryRunIsProvedDestroyed) {
    ChipWorker worker;
    prime(worker);
    worker.finalize_device_fn_ = stub_finalize_device;
    worker.destroy_device_context_fn_ = stub_destroy_device_context;
    const ChipWorkerNativeRun first = prepare(worker, 0, 1);
    worker.launch_native_run(first);
    const ChipWorkerNativeRun second = prepare(worker, 1, 2);
    (void)second;

    EXPECT_NO_THROW(worker.finalize());
    EXPECT_EQ(g_finalize_calls, 2);
    EXPECT_EQ(g_device_finalize_calls, 1);
    EXPECT_EQ(g_destroy_context_calls, 1);
    EXPECT_TRUE(worker.runtime_bufs_.empty());
    std::lock_guard<std::mutex> lk(worker.native_run_mu_);
    EXPECT_TRUE(worker.native_runs_.empty());
}

// A destroyed carrier permits device teardown, but a poisoned lane cannot authorize comm free.
TEST(NativeRunOperationLeaseTest, PoisonedLaneRetainsCommOwnersAcrossDeviceTeardownRetry) {
    for (bool failed : {false, true}) {
        SCOPED_TRACE(failed);
        ChipWorker worker;
        prime(worker);
        worker.finalize_device_fn_ = stub_finalize_device;
        worker.comm_release_domain_windows_fn_ = stub_release_domain;
        worker.comm_global_domain_release_fn_ = stub_release_global_domain;
        worker.comm_destroy_fn_ = stub_destroy_comm;
        static int comm_placeholder = 0;
        const uint64_t comm = reinterpret_cast<uint64_t>(&comm_placeholder);
        ASSERT_NE(worker.create_comm_session(&comm_placeholder, nullptr, true), nullptr);
        worker.base_comm_handle_ = comm;
        worker.global_domain_ids_.insert(7);
        const auto module = worker.runtime_module_;
        worker.run_lane_ = std::make_unique<ChipRunLane>(worker);
        ChipStorageTaskArgs args{};
        ChipRun run = worker.run_lane_->submit(1, args, CallConfig{});
        g_wait_rc = failed ? -5 : 0;
        if (failed) {
            EXPECT_THROW(run.wait_until(ChipRun::Deadline::max()), std::runtime_error);
        } else {
            EXPECT_TRUE(run.wait_until(ChipRun::Deadline::max()));
        }
        EXPECT_EQ(worker.run_lane_->poisoned(), failed);
        EXPECT_EQ(g_finalize_calls, 1);
        EXPECT_EQ(worker.unproven_native_run_count(), 0u);
        uint64_t tag = kNativeRunContextMagic;
        std::memcpy(&tag, worker.runtime_bufs_[0].data(), sizeof(tag));
        EXPECT_EQ(tag, 0u);
        if (!failed) {
            EXPECT_NO_THROW(worker.finalize());
            EXPECT_EQ(g_comm_free_calls, 2);
            EXPECT_EQ(g_device_finalize_calls, 1);
            EXPECT_TRUE(module->unload_allowed_);
            EXPECT_TRUE(worker.comm_sessions_.empty());
            EXPECT_TRUE(worker.global_domain_ids_.empty());
            continue;
        }
        const auto expect_refused = [&] {
            EXPECT_THROW(worker.comm_release_domain_windows(comm, 1, 2, 0), std::runtime_error);
            EXPECT_THROW(worker.comm_global_domain_release(7), std::runtime_error);
            EXPECT_THROW(worker.comm_destroy(comm), std::runtime_error);
            EXPECT_THROW(worker.comm_destroy_all(), std::runtime_error);
            EXPECT_EQ(g_comm_free_calls, 0);
            ASSERT_EQ(worker.comm_sessions_.size(), 1u);
            EXPECT_EQ(worker.comm_sessions_[0].handle, &comm_placeholder);
            EXPECT_EQ(worker.global_domain_ids_.count(7), 1u);
        };
        expect_refused();
        g_device_finalize_rc = -9;
        EXPECT_THROW(worker.finalize(), std::runtime_error);
        EXPECT_EQ(g_device_finalize_calls, 1);
        EXPECT_EQ(g_destroy_context_calls, 0);
        EXPECT_FALSE(module->unload_allowed_);
        expect_refused();
        g_device_finalize_rc = 0;
        EXPECT_NO_THROW(worker.finalize());
        EXPECT_EQ(g_device_finalize_calls, 2);
        EXPECT_EQ(g_destroy_context_calls, 1);
        expect_refused();
        EXPECT_NO_THROW(worker.finalize());
        EXPECT_EQ(g_device_finalize_calls, 2);
        EXPECT_EQ(g_comm_free_calls, 0);
    }
}

// A run the sweep could not prove destroyed still owns whatever its carrier holds, and that
// carrier still lives in this worker's run storage. The close therefore refuses the whole
// reclamation below it rather than freeing those resources twice or unloading their owner.
TEST(NativeRunOperationLeaseTest, WholeCloseRefusesToReclaimBehindAnUnprovenRun) {
    ChipWorker worker;
    prime(worker);
    worker.finalize_device_fn_ = stub_finalize_device;
    worker.destroy_device_context_fn_ = stub_destroy_device_context;
    const ChipWorkerNativeRun run = prepare(worker, 0, 1);
    worker.launch_native_run(run);

    g_finalize_destroys = false;
    g_finalize_corrupts = true;
    EXPECT_THROW(worker.finalize(), std::runtime_error);
    EXPECT_EQ(g_device_finalize_calls, 0) << "the device teardown ran behind an undestroyed run";
    EXPECT_EQ(g_destroy_context_calls, 0) << "the context was destroyed behind an undestroyed run";
    EXPECT_NE(worker.device_ctx_, nullptr);
    EXPECT_FALSE(worker.runtime_bufs_.empty()) << "the carrier's storage was released under it";
    {
        std::lock_guard<std::mutex> lk(worker.native_run_mu_);
        const auto it = worker.native_runs_.find(run.run_epoch);
        ASSERT_NE(it, worker.native_runs_.end());
        EXPECT_TRUE(it->second->destruction_unknown);
    }
    // The refusal is not a one-shot: a second close sees the same unproven run and refuses again
    // without attempting any device call, so nothing is reclaimed by a retry either.
    EXPECT_THROW(worker.finalize(), std::runtime_error);
    EXPECT_EQ(g_device_finalize_calls, 0);
    EXPECT_EQ(g_finalize_calls, 1) << "an isolated run was handed to the runtime again";
    // The fixture's worker is destroyed holding the isolated run, which is what production does
    // too; its destructor absorbs the same refusal.
}

// Retaining a map entry preserves nothing once the owner is gone: the entry and
// the carrier's storage both belong to the worker, so its destructor would free
// those bytes whatever the run table said. What has to survive is the block, and
// the proof is that the allocator never gets it back.
//
// The carrier is given a known word before the close and read back after the
// worker is destroyed. That read is only meaningful because the block was never
// freed — which is exactly the claim — so the case also asserts the worker
// reported abandoning it, and nothing else in this fixture writes there.
TEST(NativeRunOperationLeaseTest, AnUnprovenRunsStorageOutlivesTheWorkerThatHeldIt) {
    static constexpr uint64_t kCarrierWitness = UINT64_C(0xA5A5C0FFEE5151A5);
    void *carrier = nullptr;
    {
        ChipWorker worker;
        prime(worker);
        worker.finalize_device_fn_ = stub_finalize_device;
        worker.destroy_device_context_fn_ = stub_destroy_device_context;
        const ChipWorkerNativeRun run = prepare(worker, 0, 1);
        worker.launch_native_run(run);
        carrier = worker.runtime_bufs_[run.slot_id].data();
        ASSERT_NE(carrier, nullptr);

        // A teardown that leaves the tag unreadable: the run is isolated, and
        // whether anything still reaches the carrier is exactly what the host
        // cannot tell.
        g_finalize_destroys = false;
        g_finalize_corrupts = true;
        EXPECT_THROW(worker.finalize(), std::runtime_error);

        // Detached from the owner, so the block is no longer this object's to
        // free — the run table still names the set, but that is bookkeeping and
        // this is the ownership.
        EXPECT_EQ(worker.runtime_bufs_[run.slot_id].data(), nullptr)
            << "the carrier's storage is still owned by a worker that is about to be destroyed";
        std::memcpy(carrier, &kCarrierWitness, sizeof(kCarrierWitness));
    }

    // The worker and its run table are gone. The block is not.
    uint64_t witness = 0;
    std::memcpy(&witness, carrier, sizeof(witness));
    EXPECT_EQ(witness, kCarrierWitness) << "the abandoned run storage did not survive its owner";
    // This case owns the leak it just proved, so it is the one that frees it.
    std::free(carrier);
}

// A failed launch that nonetheless reached the device records completion and its own error, so a
// later finalization reports that error instead of waiting.
TEST(NativeRunOperationLeaseTest, FailedLaunchThatCompletedKeepsItsOwnError) {
    ChipWorker worker;
    prime(worker);
    const ChipWorkerNativeRun run = prepare(worker, 0, 1);

    g_launch_rc = -11;
    g_poll_rc = SIMPLER_NATIVE_RUN_POLL_COMPLETE;
    EXPECT_THROW(worker.launch_native_run(run), std::runtime_error);
    {
        std::lock_guard<std::mutex> lk(worker.native_run_mu_);
        const auto it = worker.native_runs_.find(run.run_epoch);
        ASSERT_NE(it, worker.native_runs_.end());
        EXPECT_EQ(it->second->phase, ChipWorker::NativeRunPhase::REAPED);
        EXPECT_EQ(it->second->pending_rc, -11);
    }
    EXPECT_THROW(worker.finalize_native_run(run), std::runtime_error);
    EXPECT_EQ(g_wait_calls, 0) << "a completed run was waited on";
}
