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

#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <pthread.h>
#include <cerrno>
#include <csignal>
#include <functional>
#include <future>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "call_config.h"
#include "run_resource_arena.h"
#include "runtime_c_api.h"
#include "task_args.h"
#include "call_argument_snapshot.h"
#include "types.h"

#define private public
#include "chip_worker.h"
#include "chip_run_lane.h"
#include "queued_submission.h"
#include "native_submission_state.h"
#undef private

#include <gtest/gtest.h>

#if defined(__linux__) && defined(__GLIBCXX__)
namespace {
thread_local bool observe_resource_wait = false;
std::mutex resource_wait_entry_mu;
std::condition_variable resource_wait_entry_changed;
bool resource_wait_entered = false;
std::atomic<int> resource_wait_return{-1};
std::atomic<pthread_cond_t *> resource_wait_condition{nullptr};
std::atomic<size_t> resource_notifications{0};
thread_local bool inspect_acquire_failure = false;
thread_local pthread_mutex_t *acquire_lane_mutex = nullptr;
thread_local size_t acquire_lane_lock_count = 0;
bool acquire_failure_entered = false;
bool acquire_failure_proceed = false;

template <typename Wait>
int observe_resource_wait_call(pthread_cond_t *condition, Wait wait) {
    if (!observe_resource_wait) return wait();
    resource_wait_condition.store(condition, std::memory_order_release);
    {
        std::lock_guard<std::mutex> lock(resource_wait_entry_mu);
        resource_wait_entered = true;
    }
    resource_wait_entry_changed.notify_all();
    const int rc = wait();
    resource_wait_return.store(rc, std::memory_order_release);
    return rc;
}
}  // namespace

extern "C" int __real_pthread_mutex_lock(pthread_mutex_t *);
extern "C" int __wrap_pthread_mutex_lock(pthread_mutex_t *mutex) {
    if (inspect_acquire_failure) {
        if (acquire_lane_mutex == nullptr) acquire_lane_mutex = mutex;
        if (mutex == acquire_lane_mutex && ++acquire_lane_lock_count == 2) {
            inspect_acquire_failure = false;
            std::unique_lock<std::mutex> lock(resource_wait_entry_mu);
            acquire_failure_entered = true;
            resource_wait_entry_changed.notify_all();
            resource_wait_entry_changed.wait(lock, [] {
                return acquire_failure_proceed;
            });
        }
    }
    return __real_pthread_mutex_lock(mutex);
}
extern "C" void __real__ZNSt18condition_variable10notify_allEv(std::condition_variable *);
extern "C" void __wrap__ZNSt18condition_variable10notify_allEv(std::condition_variable *condition) {
    if (condition->native_handle() == resource_wait_condition.load(std::memory_order_acquire))
        resource_notifications.fetch_add(1, std::memory_order_release);
    __real__ZNSt18condition_variable10notify_allEv(condition);
}
extern "C" int __real_pthread_cond_wait(pthread_cond_t *, pthread_mutex_t *);
extern "C" int __wrap_pthread_cond_wait(pthread_cond_t *condition, pthread_mutex_t *mutex) {
    return observe_resource_wait_call(condition, [&] {
        return __real_pthread_cond_wait(condition, mutex);
    });
}
extern "C" int __real_pthread_cond_timedwait(pthread_cond_t *, pthread_mutex_t *, const timespec *);
extern "C" int
__wrap_pthread_cond_timedwait(pthread_cond_t *condition, pthread_mutex_t *mutex, const timespec *deadline) {
    return observe_resource_wait_call(condition, [&] {
        return __real_pthread_cond_timedwait(condition, mutex, deadline);
    });
}
#if defined(__GLIBC__)
#if __GLIBC_PREREQ(2, 30)
extern "C" int __real_pthread_cond_clockwait(pthread_cond_t *, pthread_mutex_t *, clockid_t, const timespec *);
extern "C" int __wrap_pthread_cond_clockwait(
    pthread_cond_t *condition, pthread_mutex_t *mutex, clockid_t clock, const timespec *deadline
) {
    return observe_resource_wait_call(condition, [&] {
        return __real_pthread_cond_clockwait(condition, mutex, clock, deadline);
    });
}
#endif
#endif
#endif

namespace {

std::unordered_map<void *, uint32_t> g_slots;
std::vector<bool> g_complete;
std::vector<int> g_prepare_rc;
bool g_prepare_release_proven{true};
std::vector<int> g_launch_rc;
std::vector<int> g_poll_rc;
std::vector<int> g_wait_rc;
std::vector<int> g_finalize_rc;
std::vector<bool> g_finalize_no_submission;
std::vector<uint64_t> g_epochs;
std::vector<size_t> g_prepare_count;
std::vector<uint32_t> g_prepare_purpose;
std::vector<std::array<int32_t, 2>> g_host_controls;
std::vector<uint64_t> g_control_scalars;
std::multiset<uint64_t> g_span_borrows;
std::vector<bool> g_reject_first_prepare;
size_t g_poll_count{0};
size_t g_completion_wait_count{0};
int g_completion_wait_rc{0};
std::vector<uint32_t> g_polled_slots;
std::vector<uint32_t> g_completion_waited_slots;
// When nonzero, the poll stub completes the run after this many polls. Only
// UnboundedWaitBlocksInsteadOfPolling sets it, so that a regression there fails
// on the poll count instead of looping forever.
size_t g_poll_completes_after{0};
bool g_supports_successor{true};
int g_context_rc{0};
size_t g_context_checks{0};
std::mutex g_wait_mu;
std::condition_variable g_wait_cv;
bool g_wait_entered{false};
bool g_release_wait{true};
bool g_hold_retain{false};
std::vector<std::string> g_events;
int g_retain_rc{0};
bool g_retain_output_on_error{false};
int g_release_rc{0};
uintptr_t g_next_reference{0};
std::unordered_map<void *, uint32_t> g_resource_references;
std::unordered_map<void *, uint64_t> g_resource_epochs;
ChipWorker *g_active_worker{nullptr};

int retain_resources(void *, void *runtime, void *, void **out_reference) {
    const uint32_t slot = g_slots.at(runtime);
    g_events.push_back("retain" + std::to_string(slot));
    *out_reference = nullptr;
    {
        std::unique_lock<std::mutex> lock(g_wait_mu);
        if (g_hold_retain) {
            g_wait_entered = true;
            g_wait_cv.notify_all();
            g_wait_cv.wait(lock, [] {
                return g_release_wait;
            });
        }
    }
    if (g_retain_rc != 0 && !g_retain_output_on_error) return g_retain_rc;
    *out_reference = reinterpret_cast<void *>(++g_next_reference);
    g_resource_references.emplace(*out_reference, slot);
    g_resource_epochs.emplace(*out_reference, g_epochs[slot]);
    return g_retain_rc;
}

int release_resources(void *, void *reference) {
    const uint32_t slot = g_resource_references.at(reference);
    g_events.push_back("release" + std::to_string(slot));
    if (g_release_rc != 0) return g_release_rc;
    const uint64_t epoch = g_resource_epochs.at(reference);
    g_resource_references.erase(reference);
    g_resource_epochs.erase(reference);
    const bool referenced = std::any_of(g_resource_epochs.begin(), g_resource_epochs.end(), [epoch](const auto &one) {
        return one.second == epoch;
    });
    // This fake leaf models the common API's retained loan. The real common
    // ledger is tested separately: a known-retired carrier's last reference
    // releases one loan, including when a successor reused its pipeline slot.
    bool carrier_retired;
    {
        std::lock_guard<std::mutex> lock(g_active_worker->native_run_mu_);
        carrier_retired = g_active_worker->native_runs_.count(epoch) == 0;
    }
    if (!referenced && carrier_retired) {
        const auto loan = g_span_borrows.find(static_cast<uint64_t>(slot) + 1);
        if (loan != g_span_borrows.end()) g_span_borrows.erase(loan);
    }
    return 0;
}

uint32_t slot_of(void *runtime) { return g_slots.at(runtime); }

int prepare_run(
    void *, void *runtime, int32_t, const void *arguments, const CallConfig *, const NativeRunDescriptor *descriptor
) {
    g_slots[runtime] = descriptor->pipeline_slot;
    g_complete[descriptor->pipeline_slot] = false;
    g_epochs[descriptor->pipeline_slot] = descriptor->run_epoch;
    g_events.push_back("prepare" + std::to_string(descriptor->pipeline_slot));
    ++g_prepare_count[descriptor->pipeline_slot];
    g_prepare_purpose[descriptor->pipeline_slot] = descriptor->prepare_purpose;
    const auto &args = *static_cast<const ChipStorageTaskArgs *>(arguments);
    if (args.tensor_count() != 0 && args.tensor(0).address_space == AddressSpace::HOST) {
        const auto &tensor = args.tensor(0);
        const auto *data = tensor.data_as<int32_t>();
        g_host_controls[descriptor->pipeline_slot] = {
            data[tensor.start_offset], data[tensor.start_offset + tensor.strides[0]]
        };
        g_control_scalars[descriptor->pipeline_slot] = args.scalar(0);
    }
    if (g_reject_first_prepare[descriptor->pipeline_slot] && g_prepare_count[descriptor->pipeline_slot] == 1) {
        return PTO_RUNTIME_ERR_PREPARED_INCOMPATIBLE;
    }
    if (g_prepare_rc[descriptor->pipeline_slot] != 0 && g_prepare_release_proven) {
        publish_released_preparation(runtime, descriptor->run_epoch);
    }
    return g_prepare_rc[descriptor->pipeline_slot];
}

int launch_run(void *, void *runtime) {
    const uint32_t slot = slot_of(runtime);
    g_events.push_back("launch" + std::to_string(slot));
    return g_launch_rc[slot];
}

int poll_run(void *, void *runtime) {
    ++g_poll_count;
    const uint32_t slot = slot_of(runtime);
    g_polled_slots.push_back(slot);
    if (g_poll_rc[slot] != 0) return g_poll_rc[slot];
    if (g_poll_completes_after != 0 && g_poll_count >= g_poll_completes_after) g_complete[slot] = true;
    return g_complete[slot] ? SIMPLER_NATIVE_RUN_POLL_COMPLETE : SIMPLER_NATIVE_RUN_POLL_NOT_READY;
}

int wait_run(void *, void *runtime) {
    const uint32_t slot = slot_of(runtime);
    g_events.push_back("wait" + std::to_string(slot));
    {
        std::unique_lock<std::mutex> lk(g_wait_mu);
        g_wait_entered = true;
        g_wait_cv.notify_all();
        g_wait_cv.wait(lk, [] {
            return g_release_wait;
        });
    }
    g_complete[slot] = true;
    return g_wait_rc[slot];
}

int wait_run_completion(void *, void *runtime) {
    ++g_completion_wait_count;
    g_completion_waited_slots.push_back(slot_of(runtime));
    std::unique_lock<std::mutex> lk(g_wait_mu);
    g_wait_entered = true;
    g_wait_cv.notify_all();
    g_wait_cv.wait(lk, [] {
        return g_release_wait;
    });
    if (g_completion_wait_rc != 0) return g_completion_wait_rc;
    g_complete[slot_of(runtime)] = true;
    return 0;
}

int finalize_run(void *, void *runtime, int) {
    const uint32_t slot = slot_of(runtime);
    g_events.push_back("finalize" + std::to_string(slot));
    if (g_finalize_no_submission[slot] && g_finalize_rc[slot] == 0) {
        publish_released_preparation(runtime, g_epochs[slot]);
    }
    if (g_finalize_rc[slot] == 0 && g_prepare_purpose[slot] == SIMPLER_PREPARE_CAPTURE) {
        const uint64_t epoch = g_epochs[slot];
        const bool retained = std::any_of(g_resource_epochs.begin(), g_resource_epochs.end(), [epoch](const auto &one) {
            return one.second == epoch;
        });
        if (!retained) {
            const auto loan = g_span_borrows.find(static_cast<uint64_t>(slot) + 1);
            if (loan != g_span_borrows.end()) g_span_borrows.erase(loan);
        }
    }
    return g_finalize_rc[slot];
}

int supports_successor(void *) { return g_supports_successor ? 1 : 0; }

int validate_context(void *) {
    ++g_context_checks;
    return g_context_rc;
}

void prime_worker(ChipWorker &worker, size_t depth = 2) {
    g_slots.clear();
    g_complete.assign(depth, {});
    g_prepare_rc.assign(depth, {});
    g_prepare_release_proven = true;
    g_launch_rc.assign(depth, {});
    g_poll_rc.assign(depth, {});
    g_wait_rc.assign(depth, {});
    g_finalize_rc.assign(depth, {});
    g_finalize_no_submission.assign(depth, {});
    g_epochs.assign(depth, {});
    g_prepare_count.assign(depth, {});
    g_prepare_purpose.assign(depth, {});
    g_host_controls.assign(depth, {});
    g_control_scalars.assign(depth, {});
    g_span_borrows.clear();
    g_reject_first_prepare.assign(depth, {});
    g_poll_count = 0;
    g_completion_wait_count = 0;
    g_completion_wait_rc = 0;
    g_polled_slots.clear();
    g_completion_waited_slots.clear();
    g_poll_completes_after = 0;
    g_supports_successor = true;
    g_context_rc = 0;
    g_context_checks = 0;
    {
        std::lock_guard<std::mutex> lk(g_wait_mu);
        g_wait_entered = false;
        g_release_wait = true;
    }
    g_events.clear();
    g_retain_rc = 0;
    g_retain_output_on_error = false;
    g_release_rc = 0;
    g_hold_retain = false;
    g_next_reference = 0;
    g_resource_references.clear();
    g_resource_epochs.clear();
    g_active_worker = &worker;
    worker.initialized_ = true;
    worker.pipeline_contract_ = {PTO_PIPELINE_CONTRACT_ABI_VERSION, 0, static_cast<uint32_t>(depth), {}};
    // One generation per granted set, as `init` sizes it: an unsized filter has no entry for
    // slot 0, and `is_admissible` refuses a lease outside it, so every preparation would be
    // refused as stale before reaching the stubs below.
    worker.pipeline_generations_.resize(depth);
    for (size_t slot = 0; slot < depth; ++slot)
        worker.runtime_bufs_.emplace_back(64, alignof(std::max_align_t), nullptr);
    worker.prepare_run_fn_ = prepare_run;
    worker.launch_run_fn_ = launch_run;
    worker.poll_run_fn_ = poll_run;
    worker.wait_run_fn_ = wait_run;
    worker.wait_run_completion_fn_ = wait_run_completion;
    worker.finalize_run_fn_ = finalize_run;
    worker.retain_run_resources_fn_ = retain_resources;
    worker.release_run_resources_fn_ = release_resources;
    worker.supports_concurrent_native_prepare_fn_ = supports_successor;
}

void prime_capture_worker(ChipWorker &worker, size_t depth = 2) {
    prime_worker(worker, depth);
    worker.execution_environment_ = ChipWorker::ExecutionEnvironment{static_cast<uint64_t>(getpid()), 1, 1, 0, true};
    worker.validate_caller_stream_fn_ = [](void *, void *) {
        return 0;
    };
    worker.launch_run_on_stream_fn_ = [](void *, void *runtime, void *, void *) {
        g_events.push_back("stream_launch" + std::to_string(slot_of(runtime)));
        return g_launch_rc[slot_of(runtime)];
    };
    worker.device_borrow_caller_buffers_ctx_fn_ = [](void *, const CallerBufferSpan *, uint32_t, uint64_t identity) {
        g_span_borrows.insert(identity);
        return 0;
    };
    worker.device_release_caller_buffers_ctx_fn_ = [](void *, uint64_t identity, int keep) {
        if (!keep) {
            const auto loan = g_span_borrows.find(identity);
            if (loan != g_span_borrows.end()) g_span_borrows.erase(loan);
        }
    };
    worker.run_lane_ = std::make_unique<ChipRunLane>(worker);
}

ChipStorageTaskArgs device_args() {
    ChipStorageTaskArgs args{};
    ChipTensor device{};
    device.buffer.addr = 0x5000;
    device.buffer.size = 16;
    device.shapes[0] = 4;
    device.strides[0] = 1;
    device.ndims = 1;
    device.dtype = DataType::INT32;
    device.address_space = AddressSpace::DEVICE;
    device.transfer = TensorTransfer::NONE;
    args.add_tensor(device);
    return args;
}

template <typename Finish>
void expect_capture_wait_notification(ChipRun &run, Finish finish) {
#if defined(__linux__) && defined(__GLIBCXX__)
    {
        std::lock_guard<std::mutex> lock(resource_wait_entry_mu);
        resource_wait_entered = false;
    }
    resource_wait_return.store(-1, std::memory_order_release);
    resource_wait_condition.store(nullptr, std::memory_order_release);
    resource_notifications.store(0, std::memory_order_release);
    bool completed = false;
    std::exception_ptr wait_error;
    std::thread waiter([&] {
        observe_resource_wait = true;
        try {
            completed = run.wait_until(ChipRun::Clock::now() + std::chrono::seconds(5));
        } catch (...) {
            wait_error = std::current_exception();
        }
        observe_resource_wait = false;
    });
    struct JoinWaiter {
        std::thread &thread;
        ~JoinWaiter() {
            if (!thread.joinable()) return;
            if (auto *condition = resource_wait_condition.load(std::memory_order_acquire))
                (void)::pthread_cond_broadcast(condition);
            thread.join();
        }
    } join_waiter{waiter};
    bool entered;
    {
        std::unique_lock<std::mutex> lock(resource_wait_entry_mu);
        entered = resource_wait_entry_changed.wait_for(lock, std::chrono::seconds(5), [] {
            return resource_wait_entered;
        });
    }
    finish();
    const size_t notifications = resource_notifications.load(std::memory_order_acquire);
    // A missing production notification fails directly; cleanup still wakes
    // the actual waiter, so a negative control never relies on timing out.
    if (notifications == 0) {
        if (auto *condition = resource_wait_condition.load(std::memory_order_acquire))
            (void)::pthread_cond_broadcast(condition);
    }
    waiter.join();
    EXPECT_EQ(notifications, 1);
    resource_wait_condition.store(nullptr, std::memory_order_release);
    EXPECT_TRUE(entered);
    EXPECT_TRUE(completed);
    EXPECT_EQ(wait_error, nullptr);
    // Real wait rc0 is a notification/spurious wake, never deadline expiry.
    // The predicate loop remains responsible for accepting only checked release.
    EXPECT_EQ(resource_wait_return.load(std::memory_order_acquire), 0);
#else
    finish();
    EXPECT_TRUE(run.wait_until(ChipRun::Clock::now()));
#endif
}

ChipRun submit(ChipRunLane &lane, uint64_t run_id, uint32_t slot, bool activate = true) {
    ChipStorageTaskArgs args{};
    return lane.submit(1, args, CallConfig{}, PipelineSlotLease{slot, 0, run_id}, run_id, run_id, nullptr, 0, activate);
}

}  // namespace

namespace {
int framework_transfer_rc = 0;
bool framework_destroy = true;
int transfer_completed_run(void *, void *runtime) {
    g_events.push_back("transfer" + std::to_string(slot_of(runtime)));
    if (framework_destroy) std::memset(runtime, 0, sizeof(uint64_t));
    return framework_transfer_rc;
}
ChipRun submit_framework(ChipRunLane &lane) {
    ChipStorageTaskArgs args{};
    return lane.submit(1, args, CallConfig{}, nullptr, 0, nullptr, SIMPLER_PREPARE_EAGER, nullptr, true);
}
void enable_framework_transfer(ChipWorker &worker) {
    framework_transfer_rc = 0;
    framework_destroy = true;
    worker.transfer_completed_run_fn_ = transfer_completed_run;
}
}  // namespace

TEST(ChipRunLaneTest, FrameworkOwnRetirementUsesTransferInsteadOfStrongWaitOrFinalize) {
    ChipWorker worker;
    prime_worker(worker);
    enable_framework_transfer(worker);
    ChipRunLane lane(worker);
    auto run = submit_framework(lane);
    EXPECT_FALSE(run.retire_framework(false));
    g_complete[0] = true;
    ASSERT_TRUE(run.retire_framework(false));
    EXPECT_TRUE(run.framework_transferred());
    EXPECT_TRUE(worker.native_runs_.empty());
    EXPECT_EQ(g_events, (std::vector<std::string>{"prepare0", "launch0", "transfer0"}));
    EXPECT_EQ(g_completion_wait_count, 0u);
    EXPECT_TRUE(run.retire_framework(false));
    lane.close();
    worker.finalize();
}

TEST(ChipRunLaneTest, FrameworkRequiredOwnWaitNeverBecomesAWholeStreamObserver) {
    ChipWorker worker;
    prime_worker(worker);
    enable_framework_transfer(worker);
    ChipRunLane lane(worker);
    auto run = submit_framework(lane);
    ASSERT_TRUE(run.retire_framework(true));
    EXPECT_EQ(g_completion_wait_count, 1u);
    EXPECT_EQ(g_events, (std::vector<std::string>{"prepare0", "launch0", "transfer0"}));
    lane.close();
    worker.finalize();
}

TEST(ChipRunLaneTest, FrameworkPhysicalCompletionDeadlineRetainsTheUnfinishedCarrier) {
    ChipWorker worker;
    prime_worker(worker);
    enable_framework_transfer(worker);
    ChipRunLane lane(worker);
    auto run = submit_framework(lane);
    {
        std::lock_guard<std::mutex> lock(g_wait_mu);
        g_release_wait = false;
    }
    const auto deadline = ChipRun::Clock::now() + std::chrono::milliseconds(20);
    auto retirement = std::async(std::launch::async, [&] {
        return run.retire_framework(true, deadline);
    });
    const auto status = retirement.wait_for(std::chrono::seconds(1));
    // Release the blocking native leaf before any assertion or future destruction.
    // A regression can enter it, but must not hang the test process.
    {
        std::lock_guard<std::mutex> lock(g_wait_mu);
        g_release_wait = true;
    }
    g_wait_cv.notify_all();
    const bool retired = retirement.get();
    EXPECT_EQ(status, std::future_status::ready);
    EXPECT_FALSE(retired);
    EXPECT_EQ(g_completion_wait_count, 0u);
    EXPECT_FALSE(run.framework_transferred());
    EXPECT_EQ(worker.native_runs_.size(), 1u);
    EXPECT_EQ(g_events, (std::vector<std::string>{"prepare0", "launch0"}));
    g_complete[0] = true;
    EXPECT_TRUE(run.retire_framework(false));
    EXPECT_TRUE(worker.native_runs_.empty());
    lane.close();
    worker.finalize();
}

TEST(ChipRunLaneTest, FrameworkDestroyedFactPrecedesReturnedCleanupError) {
    ChipWorker worker;
    prime_worker(worker);
    enable_framework_transfer(worker);
    ChipRunLane lane(worker);
    auto run = submit_framework(lane);
    g_complete[0] = true;
    framework_transfer_rc = -907;
    EXPECT_THROW(run.retire_framework(false), std::runtime_error);
    EXPECT_TRUE(run.framework_transferred());
    EXPECT_FALSE(run.owner_recovery_required());
    EXPECT_TRUE(worker.native_runs_.empty());
    const auto events = g_events;
    EXPECT_TRUE(run.retire_framework(false));  // cleanup fact; lane/facade retain the first error
    EXPECT_EQ(g_events, events);               // no stale storage wait or finalization

    int context = 0;
    worker.device_ctx_ = &context;
    worker.validate_execution_context_fn_ = validate_context;
    g_context_rc = PTO_RUNTIME_ERR_INVALID_STATE;
    EXPECT_THROW(lane.close(false), ExecutionContextRefused);
    g_context_rc = 0;
    EXPECT_NO_THROW(lane.close(false));
    EXPECT_THROW(lane.close(), std::runtime_error);
    worker.device_ctx_ = nullptr;
    worker.finalize();
}

TEST(ChipRunLaneTest, FrameworkKnownIntactCleanupRetryNeverRelaunchesOrStrongWaits) {
    ChipWorker worker;
    prime_worker(worker);
    enable_framework_transfer(worker);
    ChipRunLane lane(worker);
    auto run = submit_framework(lane);
    g_complete[0] = true;
    // Native runtime's canonical live tag means no destruction occurred.
    auto *storage = worker.native_runs_.at(g_epochs[0])->storage.data();
    const uint64_t canonical = kNativeRunContextMagic;
    std::memcpy(storage, &canonical, sizeof(canonical));
    framework_transfer_rc = -909;
    framework_destroy = false;
    EXPECT_THROW(run.retire_framework(false), std::runtime_error);
    EXPECT_FALSE(run.framework_transferred());
    EXPECT_FALSE(worker.native_runs_.empty());
    framework_transfer_rc = 0;
    framework_destroy = true;
    EXPECT_THROW(run.retire_framework(false), std::runtime_error);  // original -909 remains the verdict
    EXPECT_TRUE(run.framework_transferred());
    EXPECT_TRUE(worker.native_runs_.empty());
    EXPECT_EQ(g_events, (std::vector<std::string>{"prepare0", "launch0", "transfer0", "transfer0"}));
    EXPECT_THROW(lane.close(), std::runtime_error);
    worker.finalize();
}

TEST(ChipRunLaneTest, FrameworkPollFailureCannotImplicitlyStrongFinalize) {
    ChipWorker worker;
    prime_worker(worker);
    enable_framework_transfer(worker);
    ChipRunLane lane(worker);
    auto run = submit_framework(lane);
    g_poll_rc[0] = -908;
    EXPECT_THROW(run.retire_framework(false), std::runtime_error);
    EXPECT_FALSE(run.framework_transferred());
    EXPECT_EQ(g_events, (std::vector<std::string>{"prepare0", "launch0"}));
    EXPECT_FALSE(worker.native_runs_.empty());
    g_poll_rc[0] = 0;
    EXPECT_THROW(lane.close(), std::runtime_error);  // explicit recovery alone may observe/finalize
    EXPECT_TRUE(worker.native_runs_.empty());
    worker.finalize();
}

TEST(ChipRunLaneTest, FrameworkPollFailureRecoversThroughBoundedPhysicalRetirement) {
    ChipWorker worker;
    prime_worker(worker);
    enable_framework_transfer(worker);
    ChipRunLane lane(worker);
    auto run = submit_framework(lane);
    g_poll_rc[0] = -908;
    EXPECT_THROW(run.retire_framework(false), std::runtime_error);
    ASSERT_TRUE(run.owner_recovery_required());
    ASSERT_FALSE(run.framework_transferred());

    g_poll_rc[0] = 0;
    g_complete[0] = true;
    const auto deadline = ChipRun::Clock::now() + std::chrono::seconds(1);
    try {
        (void)run.retire_framework(true, deadline);
        ADD_FAILURE() << "retirement did not report the original completion error";
    } catch (const std::runtime_error &error) {
        EXPECT_NE(std::string(error.what()).find("-908"), std::string::npos);
    }
    EXPECT_TRUE(run.framework_transferred());
    EXPECT_FALSE(run.owner_recovery_required());
    EXPECT_TRUE(worker.native_runs_.empty());
    EXPECT_EQ(g_completion_wait_count, 0u);
    EXPECT_EQ(g_events, (std::vector<std::string>{"prepare0", "launch0", "transfer0"}));
    EXPECT_THROW(lane.close(), std::runtime_error);
    worker.finalize();
}

TEST(ChipRunLaneTest, BlockingSuccessorCannotObserveFrameworkPredecessorAfterCompletionWaitFailure) {
    ChipWorker worker;
    prime_worker(worker);
    enable_framework_transfer(worker);
    ChipRunLane lane(worker);
    auto framework = submit_framework(lane);
    auto ordinary = submit(lane, 102, 1, false);
    g_completion_wait_rc = -912;
    try {
        (void)ordinary.wait_until(ChipRun::Deadline::max());
        ADD_FAILURE() << "successor wait did not report its predecessor's error";
    } catch (const std::runtime_error &error) {
        EXPECT_NE(std::string(error.what()).find("-912"), std::string::npos);
    }
    EXPECT_EQ(g_completion_wait_count, 1u);
    EXPECT_FALSE(framework.framework_transferred());
    EXPECT_TRUE(framework.owner_recovery_required());
    EXPECT_NE(worker.native_runs_.count(g_epochs[0]), 0u);
    EXPECT_EQ(std::count(g_events.begin(), g_events.end(), "finalize0"), 0);

    g_completion_wait_rc = 0;
    EXPECT_THROW(lane.close(), std::runtime_error);
    EXPECT_TRUE(worker.native_runs_.empty());
    worker.finalize();
}

TEST(ChipRunLaneTest, OwnsFifoPreparationAndLaunch) {
    ChipWorker worker;
    prime_worker(worker);
    ChipRunLane lane(worker);

    ChipRun first = submit(lane, 101, 0);
    ChipRun second = submit(lane, 102, 1, false);
    EXPECT_EQ(second.preparation_disposition(), ChipRunPreparationDisposition::NATIVE_PREPARED);
    EXPECT_EQ(g_events, (std::vector<std::string>{"prepare0", "launch0", "prepare1"}));

    second.activate();
    EXPECT_FALSE(second.done());
    g_complete[0] = true;
    EXPECT_TRUE(first.done());
    EXPECT_FALSE(second.done());
    EXPECT_TRUE(second.launched());
    EXPECT_EQ(g_events, (std::vector<std::string>{"prepare0", "launch0", "prepare1", "launch1"}));

    g_complete[1] = true;
    EXPECT_TRUE(second.done());
    lane.close();
    worker.finalize();
}

TEST(ChipRunLaneTest, DoneDoesNotObserveOrDeliverTheRun) {
    ChipWorker worker;
    prime_worker(worker);
    ChipRunLane lane(worker);
    ChipRun run = submit(lane, 101, 0);
    g_complete[0] = true;
    ASSERT_TRUE(run.done());
    EXPECT_TRUE(run.done());
    EXPECT_EQ(g_events, (std::vector<std::string>{"prepare0", "launch0"}));

    ASSERT_TRUE(run.wait_until(ChipRun::Deadline::max()));
    EXPECT_EQ(g_events, (std::vector<std::string>{"prepare0", "launch0", "wait0", "finalize0"}));
    ASSERT_TRUE(run.wait_until(ChipRun::Deadline::max()));
    EXPECT_EQ(g_events, (std::vector<std::string>{"prepare0", "launch0", "wait0", "finalize0"}));
    lane.close();
    worker.finalize();
}

TEST(ChipRunLaneTest, CompletedRunContextRefusalPreservesLateObservationForRetry) {
    ChipWorker worker;
    prime_worker(worker);
    ChipRunLane lane(worker);
    ChipRun run = submit(lane, 101, 0);
    g_complete[0] = true;
    ASSERT_TRUE(run.done());
    const uint64_t epoch = g_epochs[0];
    auto *const entry = worker.native_runs_.at(epoch).get();
    void *const storage = entry->storage.data();
    int context = 0;
    worker.device_ctx_ = &context;
    const auto clear_context = [&worker](int *) {
        worker.device_ctx_ = nullptr;
    };
    const std::unique_ptr<int, decltype(clear_context)> context_guard(&context, clear_context);
    worker.validate_execution_context_fn_ = validate_context;
    g_context_rc = PTO_RUNTIME_ERR_INVALID_STATE;

    const auto events_before = g_events;
    EXPECT_TRUE(run.done());
    EXPECT_EQ(g_context_checks, 0U);
    EXPECT_THROW(run.wait_until(ChipRun::Deadline::max()), ExecutionContextRefused);
    EXPECT_FALSE(run.lane_poisoned());
    EXPECT_EQ(g_events, events_before);
    ASSERT_EQ(worker.native_runs_.at(epoch).get(), entry);
    EXPECT_EQ(entry->storage.data(), storage);
    EXPECT_FALSE(entry->finalize_reserved);
    EXPECT_FALSE(entry->result_observed);
    EXPECT_FALSE(entry->destruction_unknown);

    g_context_rc = 0;
    EXPECT_TRUE(run.wait_until(ChipRun::Deadline::max()));
    EXPECT_TRUE(worker.native_runs_.empty());
    EXPECT_EQ(g_events, (std::vector<std::string>{"prepare0", "launch0", "wait0", "finalize0"}));
    const size_t checks_after_observation = g_context_checks;
    g_context_rc = PTO_RUNTIME_ERR_INVALID_STATE;
    EXPECT_TRUE(run.done());
    EXPECT_TRUE(run.wait_until(ChipRun::Deadline::max()));
    EXPECT_EQ(g_context_checks, checks_after_observation);
    EXPECT_EQ(g_events, (std::vector<std::string>{"prepare0", "launch0", "wait0", "finalize0"}));

    g_context_rc = 0;
    ChipRun successor = submit(lane, 102, 0);
    g_complete[0] = true;
    EXPECT_TRUE(successor.wait_until(ChipRun::Deadline::max()));
    lane.close();
    worker.device_ctx_ = nullptr;
    worker.finalize();
}

// A diagnostics config is not special-cased by the lane: its collector pools and
// per-run state are built and reset under the execution claim, so its
// preparation touches nothing the predecessor is using. The lane declines to
// stage silently rather than raising, so a reinstated special case would only
// show up as lost pipeline depth.
TEST(ChipRunLaneTest, DiagnosticSuccessorPreparesLikeAnyOther) {
    ChipWorker worker;
    prime_worker(worker);
    ChipRunLane lane(worker);
    ChipStorageTaskArgs args{};
    CallConfig diagnostic;
    diagnostic.enable_pmu = true;
    diagnostic.output_prefix[0] = 'x';

    ChipRun first = submit(lane, 101, 0);
    ChipRun second = lane.submit(1, args, diagnostic, PipelineSlotLease{1, 0, 102}, 102, 102, nullptr, 0, false);
    EXPECT_EQ(second.preparation_disposition(), ChipRunPreparationDisposition::NATIVE_PREPARED);
    EXPECT_EQ(g_events, (std::vector<std::string>{"prepare0", "launch0", "prepare1"}));

    second.activate();
    g_complete[0] = true;
    EXPECT_TRUE(first.done());
    EXPECT_FALSE(second.done());
    EXPECT_EQ(g_events, (std::vector<std::string>{"prepare0", "launch0", "prepare1", "launch1"}));
    lane.close();
    worker.finalize();
}

TEST(ChipRunLaneTest, IncompatibleSuccessorRetriesAfterPredecessorFence) {
    ChipWorker worker;
    prime_worker(worker);
    ChipRunLane lane(worker);
    g_reject_first_prepare[1] = true;

    ChipRun first = submit(lane, 101, 0);
    ChipRun second = submit(lane, 102, 1, false);
    EXPECT_EQ(second.preparation_disposition(), ChipRunPreparationDisposition::VALIDATED_ONLY);
    EXPECT_EQ(g_events, (std::vector<std::string>{"prepare0", "launch0", "prepare1"}));

    second.activate();
    g_complete[0] = true;
    EXPECT_TRUE(first.done());
    EXPECT_TRUE(second.launched());
    EXPECT_EQ(g_events, (std::vector<std::string>{"prepare0", "launch0", "prepare1", "prepare1", "launch1"}));

    g_complete[1] = true;
    EXPECT_TRUE(second.done());
    lane.close();
    worker.finalize();
}

TEST(ChipRunLaneTest, EarlierActiveDispatchOrdersBeforeAnAlreadyStagedSuccessor) {
    ChipWorker worker;
    prime_worker(worker);
    ChipRunLane lane(worker);

    ChipRun successor = submit(lane, 102, 1, false);
    EXPECT_EQ(successor.preparation_disposition(), ChipRunPreparationDisposition::VALIDATED_ONLY);
    ChipRun active = submit(lane, 101, 0, true);
    EXPECT_TRUE(active.launched());
    EXPECT_EQ(successor.preparation_disposition(), ChipRunPreparationDisposition::NATIVE_PREPARED);
    EXPECT_EQ(g_events, (std::vector<std::string>{"prepare0", "launch0", "prepare1"}));

    successor.activate();
    g_complete[0] = true;
    EXPECT_TRUE(active.done());
    EXPECT_TRUE(successor.launched());
    g_complete[1] = true;
    EXPECT_TRUE(successor.done());
    lane.close();
    worker.finalize();
}

TEST(ChipRunLaneTest, AcceptsCurrentLeaseGenerationAndRejectsOlderOne) {
    ChipWorker worker;
    prime_worker(worker);
    ChipRunLane lane(worker);
    ChipRun first = submit(lane, 101, 0);
    g_complete[0] = true;
    ASSERT_TRUE(first.done());

    ChipStorageTaskArgs args{};
    ChipRun repeated = lane.submit(1, args, CallConfig{}, PipelineSlotLease{0, 0, 101}, 102, 102, nullptr, 0, true);
    g_complete[0] = true;
    ASSERT_TRUE(repeated.done());
    EXPECT_THROW(
        lane.submit(1, args, CallConfig{}, PipelineSlotLease{0, 0, 100}, 103, 103, nullptr, 0, true), std::runtime_error
    );
    lane.close();
    worker.finalize();
}

TEST(ChipRunLaneTest, ActivateStopsAtTheLaunchFenceBeforePolling) {
    ChipWorker worker;
    prime_worker(worker);
    ChipRunLane lane(worker);
    ChipRun run = submit(lane, 101, 0, false);

    run.activate();
    EXPECT_TRUE(run.launched());
    EXPECT_EQ(g_poll_count, 0);
    g_complete[0] = true;
    EXPECT_TRUE(run.done());
    EXPECT_EQ(g_poll_count, 1);
    lane.close();
    worker.finalize();
}

TEST(ChipRunLaneTest, DirectGenerationDoesNotConsumeTheFirstPipelineLease) {
    ChipWorker worker;
    prime_worker(worker);
    ChipRunLane lane(worker);
    ChipStorageTaskArgs args{};

    ChipRun direct = lane.submit(1, args, CallConfig{});
    g_complete[0] = true;
    ASSERT_TRUE(direct.done());

    ChipRun leased = submit(lane, 1, 0);
    EXPECT_TRUE(leased.launched());
    g_complete[0] = true;
    EXPECT_TRUE(leased.done());
    lane.close();
    worker.finalize();
}

TEST(ChipRunLaneTest, DirectCapacityTwoPreparesSuccessorAndBackpressuresThird) {
    ChipWorker worker;
    prime_worker(worker);
    ChipRunLane lane(worker);
    ChipStorageTaskArgs args{};

    ChipRun first = lane.submit(1, args, CallConfig{});
    ChipRun second = lane.submit(1, args, CallConfig{});
    EXPECT_TRUE(first.launched());
    EXPECT_FALSE(second.launched());
    EXPECT_EQ(second.preparation_disposition(), ChipRunPreparationDisposition::NATIVE_PREPARED);
    EXPECT_EQ(g_events, (std::vector<std::string>{"prepare0", "launch0", "prepare1"}));

    {
        std::lock_guard<std::mutex> lk(g_wait_mu);
        g_release_wait = false;
    }
    std::optional<ChipRun> third;
    std::exception_ptr submit_error;
    std::thread submitter([&] {
        try {
            third.emplace(lane.submit(1, args, CallConfig{}));
        } catch (...) {
            submit_error = std::current_exception();
        }
    });

    bool entered_wait = false;
    {
        std::unique_lock<std::mutex> lk(g_wait_mu);
        entered_wait = g_wait_cv.wait_for(lk, std::chrono::seconds(1), [] {
            return g_wait_entered;
        });
    }
    EXPECT_TRUE(entered_wait);
    EXPECT_EQ(g_prepare_count[0], 1u) << "third submit prepared before capacity released";
    EXPECT_EQ(g_prepare_count[1], 1u);
    {
        std::lock_guard<std::mutex> lk(g_wait_mu);
        g_release_wait = true;
    }
    g_wait_cv.notify_all();
    submitter.join();

    ASSERT_EQ(submit_error, nullptr);
    ASSERT_TRUE(third.has_value());
    EXPECT_TRUE(second.launched());
    EXPECT_FALSE(third->launched());
    EXPECT_EQ(third->preparation_disposition(), ChipRunPreparationDisposition::NATIVE_PREPARED);
    EXPECT_EQ(g_prepare_count[0], 2u);

    g_complete[1] = true;
    EXPECT_TRUE(second.done());
    EXPECT_TRUE(third->launched());
    g_complete[0] = true;
    EXPECT_TRUE(third->done());
    EXPECT_TRUE(first.done());
    lane.close();
    worker.finalize();
}

TEST(ChipRunLaneTest, DirectIncompatibleSuccessorRetriesAfterPredecessorFence) {
    ChipWorker worker;
    prime_worker(worker);
    ChipRunLane lane(worker);
    ChipStorageTaskArgs args{};

    ChipRun first = lane.submit(1, args, CallConfig{});
    g_reject_first_prepare[1] = true;
    ChipRun second = lane.submit(1, args, CallConfig{});
    EXPECT_EQ(second.preparation_disposition(), ChipRunPreparationDisposition::VALIDATED_ONLY);
    EXPECT_EQ(g_prepare_count[1], 1u);

    g_complete[0] = true;
    EXPECT_FALSE(second.done());
    EXPECT_TRUE(first.done());
    EXPECT_TRUE(second.launched());
    EXPECT_EQ(g_prepare_count[1], 2u);
    g_complete[1] = true;
    EXPECT_TRUE(second.done());
    lane.close();
    worker.finalize();
}

// SIMPLER_ERROR_FLOW_CONTROL_DEADLOCK (3) negated — the status a device latches for
// a flow-control deadlock. Spelled out rather than included: the SIMPLER_ERROR_*
// macros live in each runtime's private runtime_status.h, which this
// worker-level test does not compile against.
constexpr int kLatchedFlowControlDeadlock = -3;

// A latched deadlock is terminal. PTO_RUNTIME_ERR_PREPARED_INCOMPATIBLE once held
// this same value, so the lane read the deadlock as "prepare at depth one instead"
// and retried it — reporting a successful run over a device that had wedged.
TEST(ChipRunLaneTest, LatchedFlowControlDeadlockIsTerminalRatherThanRetried) {
    ChipWorker worker;
    prime_worker(worker);
    ChipRunLane lane(worker);
    ChipStorageTaskArgs args{};

    // A live predecessor is what puts the successor's prepare on the overlapping
    // path, where the incompatible-prepare retry lives.
    ChipRun first = lane.submit(1, args, CallConfig{});
    g_prepare_rc[1] = kLatchedFlowControlDeadlock;
    ChipRun second = lane.submit(1, args, CallConfig{});

    EXPECT_TRUE(second.done());
    EXPECT_THROW(second.wait_until(ChipRunLane::Deadline::max()), std::runtime_error);
    EXPECT_EQ(g_prepare_count[1], 1u) << "latched deadlock was retried as an incompatible prepare";
    EXPECT_FALSE(lane.poisoned());

    g_complete[0] = true;
    EXPECT_TRUE(first.done());
    lane.close();
    worker.finalize();
}

TEST(ChipRunLaneTest, DirectRuntimeWithoutConcurrentPrepareRetainsDepthOne) {
    ChipWorker worker;
    prime_worker(worker);
    g_supports_successor = false;
    ChipRunLane lane(worker);
    ChipStorageTaskArgs args{};

    ChipRun first = lane.submit(1, args, CallConfig{});
    ChipRun second = lane.submit(1, args, CallConfig{});

    EXPECT_TRUE(first.done());
    EXPECT_TRUE(second.launched());
    EXPECT_EQ(g_prepare_count[0], 2u);
    EXPECT_EQ(g_prepare_count[1], 0u);
    g_complete[0] = true;
    EXPECT_TRUE(second.done());
    lane.close();
    worker.finalize();
}

TEST(ChipRunLaneTest, PrepareFailureIsTerminalWithoutPoisoningTheLane) {
    ChipWorker worker;
    prime_worker(worker);
    ChipRunLane lane(worker);
    g_prepare_rc[0] = -5;

    ChipRun failed = submit(lane, 101, 0);
    EXPECT_TRUE(failed.done());
    EXPECT_THROW(failed.wait_until(ChipRunLane::Deadline::max()), RunPreparationRefused);
    EXPECT_FALSE(lane.poisoned());

    ChipRun successor = submit(lane, 102, 1);
    EXPECT_TRUE(successor.launched());
    g_complete[1] = true;
    EXPECT_TRUE(successor.done());
    lane.close();
    worker.finalize();
}

TEST(ChipRunLaneTest, TerminalAbandonPreservesPreparationCleanupProof) {
    for (const bool released : {false, true}) {
        SCOPED_TRACE(released);
        ChipWorker worker;
        prime_worker(worker);
        ChipRunLane lane(worker);
        g_prepare_rc[0] = -5;
        g_prepare_release_proven = released;
        ChipRun failed = submit(lane, 101, 0);
        ASSERT_EQ(g_prepare_count[0], 1u);
        ASSERT_TRUE(failed.done());
        EXPECT_EQ(failed.owner_recovery_required(), !released);
        const auto events = g_events;
        EXPECT_NO_THROW(failed.abandon());
        EXPECT_EQ(g_events, events);
        EXPECT_EQ(failed.owner_recovery_required(), !released);
        if (released) {
            EXPECT_NO_THROW(lane.close());
        } else {
            EXPECT_THROW(lane.close(), std::runtime_error);
        }
        worker.finalize();
    }
}

TEST(ChipRunLaneTest, CleanAbandonPublishesContextAndLastUseProof) {
    for (const bool prepared : {false, true}) {
        SCOPED_TRACE(prepared);
        ChipWorker worker;
        prime_worker(worker);
        g_supports_successor = prepared;
        ChipRunLane lane(worker);
        ChipRun first = submit(lane, 101, 0);
        ChipRun pending = submit(lane, 102, 1, false);
        ASSERT_FALSE(pending.launched());
        ASSERT_EQ(g_prepare_count[1], prepared ? 1u : 0u);
        EXPECT_TRUE(pending.owner_recovery_required());
        pending.abandon();
        EXPECT_TRUE(pending.done());
        EXPECT_FALSE(pending.owner_recovery_required());
        const auto events = g_events;
        pending.abandon();
        EXPECT_EQ(g_events, events);
        EXPECT_FALSE(pending.owner_recovery_required());
        g_complete[0] = true;
        EXPECT_TRUE(first.done());
        lane.close();
        worker.finalize();
    }
}

TEST(ChipRunLaneTest, ReclaimedLaunchFailureDoesNotPoisonTheLane) {
    ChipWorker worker;
    prime_worker(worker);
    ChipRunLane lane(worker);
    g_launch_rc[0] = -6;

    ChipRun failed = submit(lane, 101, 0);
    EXPECT_TRUE(failed.done());
    EXPECT_THROW(failed.wait_until(ChipRunLane::Deadline::max()), std::runtime_error);
    EXPECT_FALSE(lane.poisoned());

    ChipRun successor = submit(lane, 102, 1);
    EXPECT_TRUE(successor.launched());
    g_complete[1] = true;
    EXPECT_TRUE(successor.done());
    lane.close();
    worker.finalize();
}

TEST(ChipRunLaneTest, UnstartedLaunchReleasePreservesTheOriginalErrorAndAllowsRetry) {
    ChipWorker worker;
    prime_worker(worker);
    ChipRunLane lane(worker);
    g_launch_rc[0] = -6;
    g_poll_rc[0] = SIMPLER_NATIVE_RUN_POLL_COMPLETE;
    g_finalize_no_submission[0] = true;

    ChipRun failed = submit(lane, 101, 0);
    EXPECT_TRUE(failed.done());
    for (int attempt = 0; attempt < 2; ++attempt) {
        try {
            failed.wait_until(ChipRunLane::Deadline::max());
            FAIL() << "launch failure was discarded";
        } catch (const RunPreparationRefused &error) {
            EXPECT_NE(std::string(error.what()).find("launch_native_run failed with code -6"), std::string::npos);
        }
    }
    EXPECT_FALSE(lane.poisoned());
    ChipRun successor = submit(lane, 102, 1);
    EXPECT_TRUE(successor.launched());
    g_complete[1] = true;
    EXPECT_TRUE(successor.done());
    lane.close();
    worker.finalize();
}

TEST(ChipRunLaneTest, MissingUnstartedReleaseProofKeepsTheFailurePoisoned) {
    ChipWorker worker;
    prime_worker(worker);
    ChipRunLane lane(worker);
    g_launch_rc[0] = -6;
    g_poll_rc[0] = SIMPLER_NATIVE_RUN_POLL_COMPLETE;
    ChipRun failed = submit(lane, 101, 0);
    try {
        failed.wait_until(ChipRunLane::Deadline::max());
        FAIL() << "expected a terminal launch error";
    } catch (const RunPreparationRefused &) {
        FAIL() << "missing cleanup proof permitted input release";
    } catch (const std::runtime_error &) {}
    EXPECT_TRUE(lane.poisoned());
    EXPECT_THROW(lane.close(), std::runtime_error);
    worker.finalize();
}

TEST(ChipRunLaneTest, ExpiredWaitLeavesTheRunLive) {
    ChipWorker worker;
    prime_worker(worker);
    ChipRunLane lane(worker);
    ChipRun run = submit(lane, 101, 0);

    EXPECT_FALSE(run.wait_until(ChipRunLane::Clock::now()));
    EXPECT_FALSE(run.done());
    g_complete[0] = true;
    EXPECT_TRUE(run.wait_until(ChipRunLane::Deadline::max()));
    lane.close();
    worker.finalize();
}

// An unbounded wait has no deadline to end its loop, so it must reach the
// device's blocking wait rather than re-polling until the run completes. The
// poll bound is what makes that observable: a re-polling implementation calls
// poll_run without limit, and no assertion on the run's result can see it.
TEST(ChipRunLaneTest, UnboundedWaitBlocksInsteadOfPolling) {
    ChipWorker worker;
    prime_worker(worker);
    ChipRunLane lane(worker);
    ChipRun run = submit(lane, 101, 0);

    // Nothing completes this run except wait_run, so a re-polling implementation
    // would loop forever. The escape hatch keeps that a fast failure with a
    // readable message rather than a wedged suite: past the bound the stub
    // completes the run itself, and the assertions below report the poll count.
    ASSERT_FALSE(g_complete[0]);
    g_poll_count = 0;
    g_poll_completes_after = 64;
    EXPECT_TRUE(run.wait_until(ChipRunLane::Deadline::max()));
    g_poll_completes_after = 0;

    EXPECT_NE(std::find(g_events.begin(), g_events.end(), "wait0"), g_events.end())
        << "unbounded wait never reached the device's blocking wait";
    EXPECT_EQ(g_completion_wait_count, 1u);
    EXPECT_LE(g_poll_count, 2u) << "unbounded wait polled " << g_poll_count << " times instead of blocking";
    lane.close();
    worker.finalize();
}

TEST(ChipRunLaneTest, FinalizeFailurePoisonsAdmissionAndCloseReportsIt) {
    ChipWorker worker;
    prime_worker(worker);
    g_supports_successor = false;
    ChipRunLane lane(worker);
    ChipRun first = submit(lane, 101, 0);
    ChipRun queued = submit(lane, 102, 1, false);
    ASSERT_EQ(queued.preparation_disposition(), ChipRunPreparationDisposition::VALIDATED_ONLY);
    g_complete[0] = true;
    g_finalize_rc[0] = -7;
    EXPECT_TRUE(first.done());
    try {
        (void)first.wait_until(ChipRunLane::Clock::time_point::max());
        FAIL() << "expected the poisoned-lane error";
    } catch (const std::runtime_error &e) {
        EXPECT_NE(std::string(e.what()).find("chip run lane is poisoned"), std::string::npos);
        EXPECT_NE(std::string(e.what()).find("finalize_native_run failed with code -7"), std::string::npos);
    }
    EXPECT_TRUE(lane.poisoned());

    ChipStorageTaskArgs args{};
    EXPECT_THROW(
        lane.submit(1, args, CallConfig{}, PipelineSlotLease{1, 0, 103}, 103, 103, nullptr, 0, true),
        RunPreparationRefused
    );
    EXPECT_THROW(lane.close(), std::runtime_error);
    EXPECT_FALSE(queued.owner_recovery_required());
    worker.finalize();
}

TEST(ChipRunLaneTest, CloseRetryProvesCleanupAfterCanonicalFinalizeRefusal) {
    ChipWorker worker;
    prime_worker(worker);
    worker.finalize_run_fn_ = [](void *, void *storage, int deliver_outputs) {
        g_events.push_back("finalize" + std::to_string(slot_of(storage)));
        if (g_finalize_rc[0] == 0) EXPECT_EQ(deliver_outputs, 0);
        const uint64_t tag = g_finalize_rc[0] != 0 ? kNativeRunContextMagic : 0;
        std::memcpy(storage, &tag, sizeof(tag));
        return g_finalize_rc[0];
    };
    ChipRunLane lane(worker);
    auto run = submit(lane, 101, 0);
    g_complete[0] = true;
    g_finalize_rc[0] = -7;
    EXPECT_TRUE(run.done());
    EXPECT_THROW(lane.close(), std::runtime_error);
    ASSERT_TRUE(run.owner_recovery_required());

    g_finalize_rc[0] = 0;
    EXPECT_TRUE(run.retry_owner_cleanup());
    EXPECT_FALSE(run.owner_recovery_required());
    EXPECT_TRUE(worker.native_runs_.empty());
    EXPECT_THROW(run.wait_until(ChipRunLane::Clock::now()), std::runtime_error);
    EXPECT_EQ(std::count(g_events.begin(), g_events.end(), "finalize0"), 2);
    worker.finalize();
}

TEST(ChipRunLaneTest, LaunchFailureWhoseFinalizeFailsReportsThePoisonedLane) {
    ChipWorker worker;
    prime_worker(worker);
    ChipRunLane lane(worker);
    g_launch_rc[0] = -6;
    g_finalize_rc[0] = -7;

    ChipRun failed = submit(lane, 101, 0);
    EXPECT_TRUE(failed.done());
    try {
        (void)failed.wait_until(ChipRunLane::Deadline::max());
        FAIL() << "expected the poisoned-lane error";
    } catch (const std::runtime_error &e) {
        // Two independent facts, and the message carries both. Reporting only
        // the poison would lose why this run did not run, and reporting only
        // the launch failure would hide that the lane admits nothing further.
        EXPECT_NE(std::string(e.what()).find("launch_native_run failed with code -6"), std::string::npos)
            << "the run's own cause was replaced by the cleanup that failed after it";
        EXPECT_NE(std::string(e.what()).find("chip run lane is poisoned"), std::string::npos);
        EXPECT_NE(std::string(e.what()).find("finalize_native_run failed with code -7"), std::string::npos);
    }
    EXPECT_TRUE(lane.poisoned());
    EXPECT_THROW(lane.close(), std::runtime_error);
    worker.finalize();
}

TEST(ChipRunLaneTest, EarlierFailedRunRetainsItsErrorAfterLaterRunPoisonsLane) {
    ChipWorker worker;
    prime_worker(worker);
    ChipRunLane lane(worker);
    g_prepare_rc[0] = -5;

    ChipRun earlier = submit(lane, 101, 0);
    EXPECT_TRUE(earlier.done());
    EXPECT_FALSE(lane.poisoned());

    ChipRun poisoner = submit(lane, 102, 1);
    g_complete[1] = true;
    g_finalize_rc[1] = -7;
    EXPECT_TRUE(poisoner.done());
    EXPECT_FALSE(lane.poisoned());
    EXPECT_THROW(poisoner.wait_until(ChipRunLane::Deadline::max()), std::runtime_error);
    EXPECT_TRUE(lane.poisoned());

    try {
        (void)earlier.wait_until(ChipRunLane::Deadline::max());
        FAIL() << "expected the earlier run's prepare error";
    } catch (const std::runtime_error &e) {
        EXPECT_NE(std::string(e.what()).find("prepare_native_run failed with code -5"), std::string::npos);
        EXPECT_EQ(std::string(e.what()).find("chip run lane is poisoned"), std::string::npos);
        EXPECT_EQ(std::string(e.what()).find("finalize_native_run failed with code -7"), std::string::npos);
    }

    EXPECT_THROW(lane.close(), std::runtime_error);
    worker.finalize();
}

TEST(ChipRunLaneTest, PollFailureReportsTheTerminalNativeError) {
    ChipWorker worker;
    prime_worker(worker);
    ChipRunLane lane(worker);
    ChipRun run = submit(lane, 101, 0);
    g_poll_rc[0] = SIMPLER_NATIVE_RUN_POLL_ERROR;
    g_wait_rc[0] = 507015;

    EXPECT_TRUE(run.done());
    EXPECT_THROW(run.wait_until(ChipRunLane::Deadline::max()), std::runtime_error);
    try {
        run.wait_until(ChipRunLane::Deadline::max());
        FAIL() << "expected the terminal native error";
    } catch (const std::runtime_error &e) {
        EXPECT_NE(std::string(e.what()).find("chip run lane is poisoned"), std::string::npos);
        EXPECT_NE(std::string(e.what()).find("finalize_native_run failed with code 507015"), std::string::npos);
    }
    EXPECT_TRUE(lane.poisoned());
    EXPECT_THROW(lane.close(), std::runtime_error);
    worker.finalize();
}

TEST(ChipRunLaneTest, CloseDrainsAndRejectsNewSubmissions) {
    ChipWorker worker;
    prime_worker(worker);
    ChipRunLane lane(worker);
    ChipRun first = submit(lane, 101, 0);
    lane.close();
    EXPECT_TRUE(first.done());

    ChipStorageTaskArgs args{};
    EXPECT_THROW(
        lane.submit(1, args, CallConfig{}, PipelineSlotLease{1, 0, 102}, 102, 102, nullptr, 0, true), std::runtime_error
    );
    worker.finalize();
}

TEST(ChipRunLaneTest, UnknownPreparationCleanupPoisonsAndKeepsTheReleaseUnproven) {
    ChipWorker worker;
    prime_worker(worker);
    ChipRunLane lane(worker);
    g_prepare_rc[0] = -5;
    g_prepare_release_proven = false;
    ChipRun failed = submit(lane, 101, 0);
    EXPECT_TRUE(failed.done());
    try {
        failed.wait_until(ChipRunLane::Deadline::max());
        FAIL() << "unknown cleanup must fail";
    } catch (const RunPreparationRefused &) {
        FAIL() << "missing release receipt must not supply an input lifetime proof";
    } catch (const std::runtime_error &) {}
    EXPECT_TRUE(lane.poisoned());
    EXPECT_THROW(submit(lane, 102, 1), std::runtime_error);
    EXPECT_THROW(lane.close(), std::runtime_error);
    EXPECT_NO_THROW(worker.finalize());
}

TEST(ChipRunLaneTest, ReleasedPrepareReceiptIsBoundToOneEpochAndConsumedOnce) {
    alignas(8) uint64_t storage[2]{};
    publish_released_preparation(storage, 7);
    EXPECT_FALSE(consume_released_preparation(storage, sizeof(storage), 8));
    EXPECT_FALSE(consume_released_preparation(storage, sizeof(uint64_t), 7));
    EXPECT_TRUE(consume_released_preparation(storage, sizeof(storage), 7));
    EXPECT_FALSE(consume_released_preparation(storage, sizeof(storage), 7));
    EXPECT_EQ(storage[0], 0u);
    EXPECT_EQ(storage[1], 0u);
    EXPECT_FALSE(consume_released_preparation(nullptr, sizeof(storage), 7));
}

namespace {

// The parent's waiter owns the lane mutex until after the child exits.
int fork_with_lane_locked(ChipRun &run, const std::function<void()> &child_operation) {
    {
        std::lock_guard<std::mutex> lk(g_wait_mu);
        g_release_wait = false;
    }
    std::exception_ptr wait_error;
    std::thread waiter([&] {
        try {
            (void)run.wait_until(ChipRun::Deadline::max());
        } catch (...) {
            wait_error = std::current_exception();
        }
    });
    bool entered = false;
    {
        std::unique_lock<std::mutex> lk(g_wait_mu);
        entered = g_wait_cv.wait_for(lk, std::chrono::seconds(5), [] {
            return g_wait_entered;
        });
    }
    pid_t child = -1;
    int status = -1;
    if (entered) {
        child = fork();
        if (child == 0) {
            alarm(2);
            try {
                child_operation();
                _exit(0);
            } catch (...) {
                _exit(1);
            }
        }
        if (child > 0) {
            pid_t waited;
            do {
                waited = waitpid(child, &status, 0);
            } while (waited == -1 && errno == EINTR);
            EXPECT_EQ(waited, child);
        }
    }
    {
        std::lock_guard<std::mutex> lk(g_wait_mu);
        g_release_wait = true;
    }
    g_wait_cv.notify_all();
    waiter.join();
    EXPECT_TRUE(entered);
    EXPECT_GT(child, 0);
    EXPECT_EQ(wait_error, nullptr);
    return status;
}

void require_process_refusal(const std::function<void()> &operation) {
    try {
        operation();
    } catch (const ExecutionContextRefused &) {
        return;
    }
    throw std::runtime_error("inherited lane operation did not refuse its process");
}

}  // namespace

TEST(ChipRunLaneTest, TerminalDetachKeepsBothCompletedAndExecutingHandlesHostOnly) {
    EXPECT_EXIT(
        {
            ChipWorker worker;
            prime_worker(worker);
            ChipRunLane lane(worker);
            ChipRun completed = submit(lane, 1, 0);
            g_complete[0] = true;
            if (!completed.done()) _exit(2);
            ChipRun executing = submit(lane, 2, 1);
            const auto failure = std::make_exception_ptr(std::runtime_error("owned generation ended"));
            lane.detach_after_failed_teardown(failure);
            const auto events = g_events;
            for (ChipRun *run : {&completed, &executing}) {
                if (!run->done() || !run->lane_poisoned()) _exit(3);
                for (int retry = 0; retry != 2; ++retry) {
                    try {
                        (void)run->wait_until(ChipRun::Clock::now());
                        _exit(4);
                    } catch (const std::runtime_error &error) {
                        if (std::string(error.what()).find("owned generation ended") == std::string::npos) _exit(5);
                    }
                }
                run->abandon();
            }
            if (events != g_events) _exit(6);
            _exit(0);
        },
        ::testing::ExitedWithCode(0), ""
    );
}

TEST(ChipRunLaneTest, ForkedHandlesRefuseBeforeTakingAnInheritedMutex) {
    ChipWorker worker;
    prime_worker(worker);
    ChipRunLane lane(worker);
    ChipRun completed = submit(lane, 1, 0);
    g_complete[0] = true;
    ASSERT_TRUE(completed.done());
    ChipRun run = submit(lane, 2, 0);
    const int status = fork_with_lane_locked(run, [&] {
        const auto events = g_events;
        for (ChipRun *handle : {&run, &completed}) {
            require_process_refusal([&] {
                (void)handle->done();
            });
            require_process_refusal([&] {
                (void)handle->wait_until(ChipRun::Clock::now());
            });
            require_process_refusal([&] {
                handle->activate();
            });
            require_process_refusal([&] {
                handle->abandon();
            });
            require_process_refusal([&] {
                (void)handle->launched();
            });
            require_process_refusal([&] {
                (void)handle->lane_poisoned();
            });
            require_process_refusal([&] {
                (void)handle->preparation_disposition();
            });
        }
        if (events != g_events) throw std::runtime_error("child entered the runtime");
    });
    ASSERT_TRUE(WIFEXITED(status)) << "child status=" << status;
    EXPECT_EQ(WEXITSTATUS(status), 0);
    EXPECT_TRUE(run.done());
    ChipRun next = submit(lane, 3, 0);
    g_complete[0] = true;
    EXPECT_TRUE(next.done());
}

TEST(ChipRunLaneTest, ForkedLaneRefusesBeforeTakingAnInheritedMutex) {
    ChipWorker worker;
    prime_worker(worker);
    auto lane = std::make_unique<ChipRunLane>(worker);
    ChipRun run = submit(*lane, 1, 0);
    const int status = fork_with_lane_locked(run, [&] {
        const auto events = g_events;
        require_process_refusal([&] {
            (void)submit(*lane, 2, 1);
        });
        require_process_refusal([&] {
            (void)lane->submit(1, ChipStorageTaskArgs{}, CallConfig{});
        });
        require_process_refusal([&] {
            lane->drain();
        });
        require_process_refusal([&] {
            lane->close();
        });
        require_process_refusal([&] {
            (void)lane->poisoned();
        });
        lane->stop_admission();
        lane.reset();
        if (events != g_events) throw std::runtime_error("child entered the runtime");
    });
    ASSERT_TRUE(WIFEXITED(status)) << "child status=" << status;
    EXPECT_EQ(WEXITSTATUS(status), 0);
    ChipRun next = submit(*lane, 2, 0);
    g_complete[0] = true;
    EXPECT_TRUE(next.done());
}

TEST(ChipRunLaneTest, ForkedWorkerDestructionDoesNotDetachAnInheritedLane) {
    auto worker = std::make_unique<ChipWorker>();
    prime_worker(*worker);
    worker->execution_environment_ = ChipWorker::ExecutionEnvironment{static_cast<uint64_t>(getpid()), 1, 1, 0, false};
    worker->run_lane_ = std::make_unique<ChipRunLane>(*worker);
    ChipRun run = submit(*worker->run_lane_, 1, 0);
    const int status = fork_with_lane_locked(run, [&] {
        const auto events = g_events;
        worker.reset();
        require_process_refusal([&] {
            (void)run.done();
        });
        if (events != g_events) throw std::runtime_error("child entered the runtime");
    });
    ASSERT_TRUE(WIFEXITED(status)) << "child status=" << status;
    EXPECT_EQ(WEXITSTATUS(status), 0);
    EXPECT_TRUE(run.done());
    EXPECT_TRUE(worker->execution_environment_.has_value());
    ChipRun next = submit(*worker->run_lane_, 2, 0);
    g_complete[0] = true;
    EXPECT_TRUE(next.done());
    worker->finalize();
}

TEST(ChipRunLaneTest, NativeSubmissionPinsTheNativeWorkerUntilCancellation) {
    auto worker = std::make_shared<ChipWorker>();
    prime_worker(*worker);
    worker->validate_caller_stream_fn_ = [](void *, void *) {
        return 0;
    };
    worker->launch_run_on_stream_fn_ = [](void *, void *, void *, void *) {
        return 0;
    };
    std::weak_ptr<ChipWorker> observed = worker;
    auto token = worker->prepare_chip_submission(
        1, CallArgumentSnapshot(ChipStorageTaskArgs{}, {}), CallConfig{}, reinterpret_cast<void *>(1)
    );
    worker.reset();
    EXPECT_FALSE(observed.expired());
    std::thread queue([token = std::move(token), observed] {
        EXPECT_TRUE(token->cancel(std::make_exception_ptr(RunPreparationRefused("cancelled"))));
        EXPECT_FALSE(observed.expired());
        token->retire();
    });
    queue.join();
    EXPECT_TRUE(observed.expired());
    EXPECT_TRUE(g_events.empty());
}

TEST(ChipRunLaneTest, CapturePreparationRejectsProgramBeforeBackendEntry) {
    ChipWorker worker;
    prime_worker(worker);
    ChipStorageTaskArgs args{};
    EXPECT_THROW(
        worker.prepare_native_run(
            1, &args, CallConfig{}, PipelineSlotLease{0, 0, 1}, 0, 0, nullptr, 0, {}, SIMPLER_PREPARE_CAPTURE
        ),
        RunPreparationRefused
    );
    EXPECT_TRUE(g_events.empty());
    EXPECT_TRUE(worker.native_runs_.empty());
    worker.finalize();
}

TEST(ChipRunLaneTest, CapturePreparesTwoDistinctGrantedSetsWithoutLaunchingOrDraining) {
    ChipWorker worker;
    prime_worker(worker);
    worker.execution_environment_ = ChipWorker::ExecutionEnvironment{static_cast<uint64_t>(getpid()), 1, 1, 0, true};
    g_supports_successor = false;
    ChipStorageTaskArgs args{};
    auto first = worker.prepare_native_run(
        1, &args, CallConfig{}, PipelineSlotLease{0, 0, 1}, 0, 0, nullptr, 0, {}, SIMPLER_PREPARE_CAPTURE
    );
    auto second = worker.prepare_native_run(
        1, &args, CallConfig{}, PipelineSlotLease{1, 0, 2}, 0, 0, nullptr, 0, {}, SIMPLER_PREPARE_CAPTURE
    );
    EXPECT_EQ(g_events, (std::vector<std::string>{"prepare0", "prepare1"}));
    EXPECT_EQ(g_prepare_purpose, (std::vector<uint32_t>{SIMPLER_PREPARE_CAPTURE, SIMPLER_PREPARE_CAPTURE}));
    EXPECT_EQ(g_poll_count, 0);
    EXPECT_EQ(g_completion_wait_count, 0);
    EXPECT_THROW(
        worker.prepare_native_run(
            1, &args, CallConfig{}, PipelineSlotLease{0, 0, 3}, 0, 0, nullptr, 0, {}, SIMPLER_PREPARE_CAPTURE
        ),
        RunPreparationRefused
    );
    EXPECT_EQ(g_events.size(), 2);
    worker.finalize_native_run(first);
    worker.finalize_native_run(second);
    worker.finalize();
}

TEST(ChipRunLaneTest, CapturePreparedCarrierFinalizationDelegatesWithoutEagerWait) {
    ChipWorker worker;
    prime_worker(worker);
    worker.execution_environment_ = ChipWorker::ExecutionEnvironment{static_cast<uint64_t>(getpid()), 1, 1, 0, true};
    worker.validate_caller_stream_fn_ = [](void *, void *) {
        return 0;
    };
    worker.launch_run_on_stream_fn_ = [](void *, void *runtime, void *, void *) {
        g_events.push_back("stream_launch" + std::to_string(slot_of(runtime)));
        return 0;
    };
    ChipStorageTaskArgs args{};
    auto run = worker.prepare_native_run(
        1, &args, CallConfig{}, PipelineSlotLease{0, 0, 1}, 0, 0, nullptr, 0, {}, SIMPLER_PREPARE_CAPTURE
    );
    ASSERT_EQ(worker.launch_native_run(run, reinterpret_cast<void *>(1)), ChipWorker::NativeLaunch::Launched);
    worker.finalize_native_run(run);
    EXPECT_EQ(g_events, (std::vector<std::string>{"prepare0", "stream_launch0", "finalize0"}));
    EXPECT_EQ(g_completion_wait_count, 0);
    worker.finalize();
}

TEST(ChipRunLaneTest, CapturePreparationCannotJoinAnEagerActiveRun) {
    ChipWorker worker;
    prime_worker(worker);
    worker.execution_environment_ = ChipWorker::ExecutionEnvironment{static_cast<uint64_t>(getpid()), 1, 1, 0, true};
    ChipStorageTaskArgs args{};
    auto eager = worker.prepare_native_run(1, &args, CallConfig{}, PipelineSlotLease{0, 0, 1});
    ASSERT_EQ(worker.launch_native_run(eager), ChipWorker::NativeLaunch::Launched);
    EXPECT_THROW(
        worker.prepare_native_run(
            1, &args, CallConfig{}, PipelineSlotLease{1, 0, 2}, 0, 0, nullptr, 0, {}, SIMPLER_PREPARE_CAPTURE
        ),
        RunPreparationRefused
    );
    EXPECT_EQ(g_events, (std::vector<std::string>{"prepare0", "launch0"}));
    worker.finalize_native_run(eager);
    worker.finalize();
}

TEST(ChipRunLaneTest, NativeResourceReferenceSurvivesOriginalCarrierAndReleasesOnNativeThread) {
    auto worker = std::make_shared<ChipWorker>();
    prime_worker(*worker);
    ChipStorageTaskArgs args{};
    auto run = worker->prepare_native_run(1, &args, CallConfig{}, PipelineSlotLease{0, 0, 1});
    auto resources = worker->retain_native_run_resources(run, reinterpret_cast<void *>(1));
    worker->finalize_native_run(run);
    EXPECT_TRUE(worker->native_runs_.empty());
    EXPECT_EQ(worker->native_resource_references_, 1);
    std::weak_ptr<ChipWorker> observed = worker;
    worker.reset();
    EXPECT_FALSE(observed.expired());
    std::thread retirement([resources = std::move(resources)] {
        resources->release();
        resources->release();
    });
    retirement.join();
    EXPECT_TRUE(observed.expired());
    EXPECT_TRUE(g_resource_references.empty());
    EXPECT_EQ(g_events, (std::vector<std::string>{"prepare0", "retain0", "finalize0", "release0"}));
}

TEST(ChipRunLaneTest, NativeResourceAcquireFailureAllowsPreparedRetryWithoutCharge) {
    auto worker = std::make_shared<ChipWorker>();
    prime_worker(*worker);
    ChipStorageTaskArgs args{};
    auto run = worker->prepare_native_run(1, &args, CallConfig{}, PipelineSlotLease{0, 0, 1});
    g_retain_rc = -71;
    EXPECT_THROW(worker->retain_native_run_resources(run, reinterpret_cast<void *>(1)), std::runtime_error);
    EXPECT_EQ(worker->native_resource_references_, 0);
    EXPECT_TRUE(g_resource_references.empty());
    EXPECT_EQ(worker->native_runs_.at(run.run_epoch)->phase, ChipWorker::NativeRunPhase::PREPARED);
    g_retain_rc = 0;
    auto resources = worker->retain_native_run_resources(run, reinterpret_cast<void *>(1));
    EXPECT_EQ(worker->native_resource_references_, 1);
    resources->release();
    worker->finalize_native_run(run);
    worker->finalize();
    EXPECT_EQ(g_events, (std::vector<std::string>{"prepare0", "retain0", "retain0", "release0", "finalize0"}));
}

TEST(ChipRunLaneTest, NativeResourceReleaseFailurePreservesReferenceAndRefusesCloseBeforeMutation) {
    auto worker = std::make_shared<ChipWorker>();
    prime_worker(*worker);
    worker->run_lane_ = std::make_unique<ChipRunLane>(*worker);
    ChipStorageTaskArgs args{};
    auto run = worker->prepare_native_run(1, &args, CallConfig{}, PipelineSlotLease{0, 0, 1});
    auto resources = worker->retain_native_run_resources(run, reinterpret_cast<void *>(1));
    g_release_rc = -72;
    EXPECT_THROW(resources->release(), std::runtime_error);
    EXPECT_EQ(worker->native_resource_references_, 1);
    EXPECT_EQ(g_resource_references.size(), 1);
    const auto events = g_events;
    EXPECT_THROW(worker->finalize(), std::runtime_error);
    EXPECT_EQ(g_events, events);
    EXPECT_TRUE(worker->initialized_);
    EXPECT_FALSE(worker->native_runs_closing_);
    EXPECT_EQ(worker->native_runs_.at(run.run_epoch)->phase, ChipWorker::NativeRunPhase::PREPARED);
    g_release_rc = 0;
    resources->release();
    EXPECT_EQ(worker->native_resource_references_, 0);
    worker->finalize_native_run(run);
    worker->finalize();
}

TEST(ChipRunLaneTest, NativeResourceReferencesReleaseIndependentlyForTwoPreparedRuns) {
    auto worker = std::make_shared<ChipWorker>();
    prime_worker(*worker);
    worker->execution_environment_ = ChipWorker::ExecutionEnvironment{static_cast<uint64_t>(getpid()), 1, 1, 0, true};
    ChipStorageTaskArgs args{};
    auto first = worker->prepare_native_run(
        1, &args, CallConfig{}, PipelineSlotLease{0, 0, 1}, 0, 0, nullptr, 0, {}, SIMPLER_PREPARE_CAPTURE
    );
    auto second = worker->prepare_native_run(
        1, &args, CallConfig{}, PipelineSlotLease{1, 0, 2}, 0, 0, nullptr, 0, {}, SIMPLER_PREPARE_CAPTURE
    );
    auto first_resources = worker->retain_native_run_resources(first, reinterpret_cast<void *>(1));
    auto second_resources = worker->retain_native_run_resources(second, reinterpret_cast<void *>(1));
    worker->finalize_native_run(first);
    worker->finalize_native_run(second);
    first_resources->release();
    EXPECT_EQ(worker->native_resource_references_, 1);
    EXPECT_EQ(g_resource_references.size(), 1);
    EXPECT_THROW(worker->finalize(), std::runtime_error);
    second_resources->release();
    EXPECT_EQ(worker->native_resource_references_, 0);
    worker->finalize();
}

TEST(ChipRunLaneTest, DroppedUnreleasedNativeResourceReferenceRetainsExactNativeBacking) {
    EXPECT_EXIT(
        ([] {
            auto worker = std::make_shared<ChipWorker>();
            prime_worker(*worker);
            ChipStorageTaskArgs args{};
            auto run = worker->prepare_native_run(1, &args, CallConfig{}, PipelineSlotLease{0, 0, 1});
            auto resources = worker->retain_native_run_resources(run, reinterpret_cast<void *>(1));
            worker->finalize_native_run(run);
            std::weak_ptr<ChipWorker> observed = worker;
            worker.reset();
            resources.reset();
            auto retained = observed.lock();
            if (!retained || retained->native_resource_references_ != 1 || g_resource_references.size() != 1) _exit(1);
            try {
                retained->finalize();
                _exit(2);
            } catch (const std::runtime_error &) {}
            _exit(0);
        }()),
        ::testing::ExitedWithCode(0), ""
    );
}

TEST(ChipRunLaneTest, NativeResourceClosedRunRefusalPrecedesClearedRuntimeEntryInspection) {
    auto worker = std::make_shared<ChipWorker>();
    prime_worker(*worker);
    ChipStorageTaskArgs args{};
    auto run = worker->prepare_native_run(1, &args, CallConfig{}, PipelineSlotLease{0, 0, 1});
    worker->finalize_native_run(run);
    std::thread closer([&] {
        worker->finalize();
    });
    closer.join();
    ASSERT_EQ(worker->retain_run_resources_fn_, nullptr);
    ASSERT_EQ(worker->release_run_resources_fn_, nullptr);
    try {
        (void)worker->retain_native_run_resources(run, reinterpret_cast<void *>(1));
        FAIL() << "a closed worker cannot retain the old carrier";
    } catch (const std::runtime_error &error) {
        EXPECT_NE(std::string(error.what()).find("another finalization is in progress"), std::string::npos);
    }
    EXPECT_EQ(worker->native_resource_references_, 0);
    EXPECT_EQ(g_events, (std::vector<std::string>{"prepare0", "finalize0"}));
}

TEST(ChipRunLaneTest, InFlightNativeResourceAcquireRefusesCloseWithoutWaitingOrTeardown) {
    auto worker = std::make_shared<ChipWorker>();
    prime_worker(*worker);
    ChipStorageTaskArgs args{};
    auto run = worker->prepare_native_run(1, &args, CallConfig{}, PipelineSlotLease{0, 0, 1});
    {
        std::lock_guard<std::mutex> lock(g_wait_mu);
        g_hold_retain = true;
        g_release_wait = false;
    }
    std::unique_ptr<ChipRunResources> resources;
    std::exception_ptr acquire_error;
    std::thread acquiring([&] {
        try {
            resources = worker->retain_native_run_resources(run, reinterpret_cast<void *>(1));
        } catch (...) {
            acquire_error = std::current_exception();
        }
    });
    bool entered;
    {
        std::unique_lock<std::mutex> lock(g_wait_mu);
        entered = g_wait_cv.wait_for(lock, std::chrono::seconds(5), [] {
            return g_wait_entered;
        });
    }
    if (entered) {
        EXPECT_THROW(worker->finalize(), std::runtime_error);
        EXPECT_FALSE(worker->native_runs_closing_);
        EXPECT_TRUE(worker->initialized_);
        EXPECT_EQ(g_events, (std::vector<std::string>{"prepare0", "retain0"}));
    }
    {
        std::lock_guard<std::mutex> lock(g_wait_mu);
        g_release_wait = true;
    }
    g_wait_cv.notify_all();
    acquiring.join();
    ASSERT_TRUE(entered);
    ASSERT_EQ(acquire_error, nullptr);
    ASSERT_NE(resources, nullptr);
    EXPECT_EQ(worker->native_resource_references_, 1);
    worker->finalize_native_run(run);
    resources->release();
    worker->finalize();
}

TEST(ChipRunLaneTest, CaptureLanePreparesTwoGrantedSlotsAndSnapshotsWithoutOrdinaryProgress) {
    auto worker = std::make_shared<ChipWorker>();
    prime_capture_worker(*worker);
    int32_t controls[] = {99, 7, 11, 13};
    ChipStorageTaskArgs args{};
    ChipTensor host{};
    host.buffer.addr = reinterpret_cast<uintptr_t>(controls);
    host.buffer.size = sizeof(controls);
    host.shapes[0] = 2;
    host.strides[0] = 2;
    host.start_offset = 1;
    host.ndims = 1;
    host.dtype = DataType::INT32;
    host.address_space = AddressSpace::HOST;
    host.transfer = TensorTransfer::NONE;
    args.add_tensor(host);
    args.add_scalar(41);
    CallArgumentSnapshot snapshot(args, {ArgDirection::IN});
    controls[1] = 19;
    args.scalars_[0] = 22;
    auto first = worker->submit_chip_run(
        1, snapshot.args(), CallConfig{}, nullptr, 0, reinterpret_cast<void *>(1), SIMPLER_PREPARE_CAPTURE
    );
    auto second = worker->submit_chip_run(
        1, snapshot.args(), CallConfig{}, nullptr, 0, reinterpret_cast<void *>(1), SIMPLER_PREPARE_CAPTURE
    );
    EXPECT_EQ(g_host_controls, (std::vector<std::array<int32_t, 2>>{{7, 13}, {7, 13}}));
    EXPECT_EQ(g_control_scalars, (std::vector<uint64_t>{41, 41}));
    EXPECT_EQ(g_prepare_purpose, (std::vector<uint32_t>{SIMPLER_PREPARE_CAPTURE, SIMPLER_PREPARE_CAPTURE}));
    EXPECT_FALSE(first.done());
    EXPECT_FALSE(second.wait_until(ChipRun::Clock::now()));
    EXPECT_THROW(
        worker->submit_chip_run(
            1, snapshot.args(), CallConfig{}, nullptr, 0, reinterpret_cast<void *>(1), SIMPLER_PREPARE_CAPTURE
        ),
        RunPreparationRefused
    );
    EXPECT_THROW(worker->submit_chip_run(1, snapshot.args(), CallConfig{}), RunPreparationRefused);
    EXPECT_EQ(g_events, (std::vector<std::string>{"prepare0", "prepare1"}));
    EXPECT_EQ(g_poll_count, 0);
    EXPECT_EQ(g_completion_wait_count, 0);
    first.abandon();
    second.abandon();
    EXPECT_TRUE(first.done());
    EXPECT_TRUE(second.done());
    worker->finalize();
}

TEST(ChipRunLaneTest, CaptureCarrierRetirementReusesSlotsButKeepsSpansUntilCheckedRelease) {
    auto worker = std::make_shared<ChipWorker>();
    prime_capture_worker(*worker);
    auto args = device_args();
    auto first = worker->submit_chip_run(
        1, args, CallConfig{}, nullptr, 0, reinterpret_cast<void *>(1), SIMPLER_PREPARE_CAPTURE
    );
    auto second = worker->submit_chip_run(
        1, args, CallConfig{}, nullptr, 0, reinterpret_cast<void *>(1), SIMPLER_PREPARE_CAPTURE
    );
    auto first_resources = first.retain_resources();
    auto second_resources = second.retain_resources();
    first.activate();
    second.activate();
    EXPECT_TRUE(worker->native_runs_.empty());
    EXPECT_FALSE(first.done());
    EXPECT_FALSE(first.execution_ended());
    EXPECT_TRUE(first.owner_recovery_required());
    EXPECT_FALSE(second.done());
    EXPECT_EQ(g_span_borrows.size(), 2);
    auto third = worker->submit_chip_run(
        1, args, CallConfig{}, nullptr, 0, reinterpret_cast<void *>(1), SIMPLER_PREPARE_CAPTURE
    );
    auto third_resources = third.retain_resources();
    third.activate();
    EXPECT_EQ(g_span_borrows.size(), 3);
    EXPECT_THROW(worker->finalize(), std::runtime_error);
    g_release_rc = -72;
    EXPECT_THROW(first_resources->release(), std::runtime_error);
    EXPECT_FALSE(first.wait_until(ChipRun::Clock::now()));
    EXPECT_EQ(g_span_borrows.size(), 3);
    g_release_rc = 0;
    first_resources->release();
    EXPECT_TRUE(first.done());
    EXPECT_FALSE(first.owner_recovery_required());
    EXPECT_FALSE(second.done());
    EXPECT_EQ(g_span_borrows.size(), 2);
    second_resources->release();
    EXPECT_TRUE(second.done());
    EXPECT_FALSE(third.done());
    EXPECT_EQ(g_span_borrows.size(), 1);
    third_resources->release();
    EXPECT_TRUE(third.done());
    EXPECT_TRUE(g_span_borrows.empty());
    EXPECT_EQ(g_poll_count, 0);
    EXPECT_EQ(g_completion_wait_count, 0);
    worker->finalize();
}

TEST(ChipRunLaneTest, CaptureLaneDoesNotProgressAnEagerPredecessorToManufactureCapacity) {
    auto worker = std::make_shared<ChipWorker>();
    prime_capture_worker(*worker);
    auto eager = worker->submit_chip_run(1, ChipStorageTaskArgs{}, CallConfig{});
    const auto events = g_events;
    EXPECT_THROW(
        worker->submit_chip_run(
            1, ChipStorageTaskArgs{}, CallConfig{}, nullptr, 0, reinterpret_cast<void *>(1), SIMPLER_PREPARE_CAPTURE
        ),
        RunPreparationRefused
    );
    EXPECT_EQ(g_events, events);
    EXPECT_EQ(g_poll_count, 0);
    EXPECT_EQ(g_completion_wait_count, 0);
    g_complete[0] = true;
    EXPECT_TRUE(eager.wait_until(ChipRun::Deadline::max()));
    worker->finalize();
}

TEST(ChipRunLaneTest, CaptureLastUseWaitsForEveryAcquiredReferenceAndFailedAcquireCanRetry) {
    auto worker = std::make_shared<ChipWorker>();
    prime_capture_worker(*worker);
    auto run = worker->submit_chip_run(
        1, device_args(), CallConfig{}, nullptr, 0, reinterpret_cast<void *>(1), SIMPLER_PREPARE_CAPTURE
    );
    g_retain_rc = -71;
    EXPECT_THROW(run.retain_resources(), std::runtime_error);
    g_retain_rc = 0;
    auto first = run.retain_resources();
    // The real CAPI permits one reference; the fake deliberately accepts two
    // so first release cannot be mistaken for discharge of all obligations.
    auto second = run.retain_resources();
    run.activate();
    first->release();
    EXPECT_FALSE(run.done());
    EXPECT_EQ(g_span_borrows.size(), 1);
    EXPECT_TRUE(run.owner_recovery_required());
    second->release();
    EXPECT_TRUE(run.done());
    EXPECT_TRUE(g_span_borrows.empty());
    worker->finalize();
}

TEST(ChipRunLaneTest, CheckedCaptureResourceReleaseNotifiesTheActualBlockedLaneWait) {
    auto worker = std::make_shared<ChipWorker>();
    prime_capture_worker(*worker);
    auto run = worker->submit_chip_run(
        1, device_args(), CallConfig{}, nullptr, 0, reinterpret_cast<void *>(1), SIMPLER_PREPARE_CAPTURE
    );
    auto resources = run.retain_resources();
    run.activate();
    expect_capture_wait_notification(run, [&] {
        g_release_rc = -72;
        EXPECT_THROW(resources->release(), std::runtime_error);
        EXPECT_FALSE(run.done());
        EXPECT_EQ(g_span_borrows.size(), 1);
        g_release_rc = 0;
        resources->release();
    });
    EXPECT_TRUE(run.done());
    EXPECT_EQ(g_poll_count, 0);
    EXPECT_EQ(g_completion_wait_count, 0);
    worker->finalize();
}

TEST(ChipRunLaneTest, CaptureReferencePinsWorkerWithoutLaneOwnershipCycle) {
    auto worker = std::make_shared<ChipWorker>();
    prime_capture_worker(*worker);
    auto run = worker->submit_chip_run(
        1, device_args(), CallConfig{}, nullptr, 0, reinterpret_cast<void *>(1), SIMPLER_PREPARE_CAPTURE
    );
    auto resources = run.retain_resources();
    run.activate();
    std::weak_ptr<ChipWorker> observed = worker;
    worker.reset();
    EXPECT_FALSE(observed.expired());
    std::thread retirement([resources = std::move(resources)]() mutable {
        resources->release();
        resources.reset();
    });
    retirement.join();
    EXPECT_TRUE(observed.expired());
    EXPECT_TRUE(run.done());
    EXPECT_FALSE(run.owner_recovery_required());
    EXPECT_TRUE(g_span_borrows.empty());
}

TEST(ChipRunLaneTest, CaptureAcquireKeepsNativeOwnerAcrossConcurrentCloseRefusal) {
    auto worker = std::make_shared<ChipWorker>();
    prime_capture_worker(*worker);
    auto run = worker->submit_chip_run(
        1, device_args(), CallConfig{}, nullptr, 0, reinterpret_cast<void *>(1), SIMPLER_PREPARE_CAPTURE
    );
    std::weak_ptr<ChipWorker> observed = worker;
    {
        std::lock_guard<std::mutex> lock(g_wait_mu);
        g_hold_retain = true;
        g_release_wait = false;
    }
    std::unique_ptr<ChipRunResources> resources;
    std::exception_ptr error;
    std::thread acquiring([&] {
        try {
            resources = run.retain_resources();
        } catch (...) {
            error = std::current_exception();
        }
    });
    bool entered;
    {
        std::unique_lock<std::mutex> lock(g_wait_mu);
        entered = g_wait_cv.wait_for(lock, std::chrono::seconds(5), [] {
            return g_wait_entered;
        });
    }
    if (entered) {
        worker.reset();
        auto pinned = observed.lock();
        EXPECT_NE(pinned, nullptr);
        if (pinned) {
            EXPECT_THROW(pinned->finalize(), std::runtime_error);
            EXPECT_FALSE(pinned->native_runs_closing_);
        }
    }
    {
        std::lock_guard<std::mutex> lock(g_wait_mu);
        g_release_wait = true;
    }
    g_wait_cv.notify_all();
    acquiring.join();
    ASSERT_TRUE(entered);
    ASSERT_EQ(error, nullptr);
    ASSERT_NE(resources, nullptr);
    run.abandon();
    EXPECT_FALSE(run.done());
    resources->release();
    resources.reset();
    EXPECT_TRUE(observed.expired());
    EXPECT_TRUE(run.done());
    EXPECT_TRUE(g_span_borrows.empty());
}

TEST(ChipRunLaneTest, CaptureAbandonWithoutExternalReferenceNotifiesTheExistingWaiter) {
    auto worker = std::make_shared<ChipWorker>();
    prime_capture_worker(*worker);
    auto run = worker->submit_chip_run(
        1, device_args(), CallConfig{}, nullptr, 0, reinterpret_cast<void *>(1), SIMPLER_PREPARE_CAPTURE
    );
    expect_capture_wait_notification(run, [&] {
        run.abandon();
    });
    EXPECT_TRUE(run.done());
    EXPECT_TRUE(g_span_borrows.empty());
    worker->finalize();
}

TEST(ChipRunLaneTest, CaptureStopAdmissionNotifiesTheExistingWaiterAfterKnownCancellation) {
    auto worker = std::make_shared<ChipWorker>();
    prime_capture_worker(*worker);
    auto run = worker->submit_chip_run(
        1, device_args(), CallConfig{}, nullptr, 0, reinterpret_cast<void *>(1), SIMPLER_PREPARE_CAPTURE
    );
    expect_capture_wait_notification(run, [&] {
        worker->run_lane_->stop_admission();
    });
    EXPECT_TRUE(run.done());
    EXPECT_TRUE(g_span_borrows.empty());
    worker->finalize();
}

#if defined(__linux__) && defined(__GLIBCXX__)
TEST(ChipRunLaneTest, FailedCaptureAcquireCompletesConcurrentCarrierRetirement) {
    auto worker = std::make_shared<ChipWorker>();
    prime_capture_worker(*worker);
    auto run = worker->submit_chip_run(
        1, device_args(), CallConfig{}, nullptr, 0, reinterpret_cast<void *>(1), SIMPLER_PREPARE_CAPTURE
    );
    {
        std::lock_guard<std::mutex> lock(resource_wait_entry_mu);
        acquire_failure_entered = false;
        acquire_failure_proceed = false;
    }
    g_retain_rc = -71;
    std::exception_ptr error;
    std::thread acquiring([&] {
        inspect_acquire_failure = true;
        acquire_lane_mutex = nullptr;
        acquire_lane_lock_count = 0;
        try {
            (void)run.retain_resources();
        } catch (...) {
            error = std::current_exception();
        }
        inspect_acquire_failure = false;
    });
    struct JoinAcquire {
        std::thread &thread;
        ~JoinAcquire() {
            if (!thread.joinable()) return;
            {
                std::lock_guard<std::mutex> lock(resource_wait_entry_mu);
                acquire_failure_proceed = true;
            }
            resource_wait_entry_changed.notify_all();
            thread.join();
        }
    } join_acquire{acquiring};
    bool entered;
    {
        std::unique_lock<std::mutex> lock(resource_wait_entry_mu);
        entered = resource_wait_entry_changed.wait_for(lock, std::chrono::seconds(5), [] {
            return acquire_failure_entered;
        });
    }
    ASSERT_TRUE(
        entered
    ) << "failed observer must reach the actual second lane-lock entry after releasing its native op";
    expect_capture_wait_notification(run, [&] {
        run.abandon();
        EXPECT_TRUE(worker->native_runs_.empty());
        EXPECT_FALSE(run.done());
        // The failed native acquire already returned with no reference. Only
        // its lane notification is parked; checked Host retirement ends the
        // active loan before that notification establishes terminal last-use.
        EXPECT_TRUE(g_span_borrows.empty());
        {
            std::lock_guard<std::mutex> lock(resource_wait_entry_mu);
            acquire_failure_proceed = true;
        }
        resource_wait_entry_changed.notify_all();
        acquiring.join();
        EXPECT_NE(error, nullptr);
        const bool released = run.done();
        EXPECT_TRUE(released);
        EXPECT_TRUE(g_span_borrows.empty());
        // The unchanged old mechanism needs an extra explicit cancel just for
        // cleanup. The assertion above still records its missing last-use proof.
        if (!released) run.abandon();
    });
    EXPECT_TRUE(run.done());
    worker->finalize();
}
#endif

TEST(ChipRunLaneTest, CaptureEmissionDelegatesActiveStreamValidationToTheCommonBackend) {
    auto worker = std::make_shared<ChipWorker>();
    prime_capture_worker(*worker);
    auto run = worker->submit_chip_run(
        1, device_args(), CallConfig{}, nullptr, 0, reinterpret_cast<void *>(1), SIMPLER_PREPARE_CAPTURE
    );
    auto resources = run.retain_resources();
    // The ordinary validator rejects ACTIVE. Only the common caller-stream
    // launch entry may validate and classify actual recording for this intent.
    worker->validate_caller_stream_fn_ = [](void *, void *) -> int {
        return PTO_RUNTIME_ERR_UNSUPPORTED;
    };
    EXPECT_NO_THROW(run.activate());
    EXPECT_EQ(g_events, (std::vector<std::string>{"prepare0", "retain0", "stream_launch0", "finalize0"}));
    resources->release();
    worker->finalize();
}

TEST(ChipRunLaneTest, CaptureLaunchFailureDoesNotEagerPollAndPreservesTheOriginalError) {
    auto worker = std::make_shared<ChipWorker>();
    prime_capture_worker(*worker);
    auto run = worker->submit_chip_run(
        1, device_args(), CallConfig{}, nullptr, 0, reinterpret_cast<void *>(1), SIMPLER_PREPARE_CAPTURE
    );
    auto resources = run.retain_resources();
    g_launch_rc[0] = -81;
    EXPECT_THROW(run.activate(), std::runtime_error);
    EXPECT_EQ(g_poll_count, 0);
    EXPECT_EQ(g_completion_wait_count, 0);
    EXPECT_FALSE(run.done());
    resources->release();
    try {
        (void)run.wait_until(ChipRun::Clock::now());
        FAIL() << "launch error must remain observable";
    } catch (const std::runtime_error &error) {
        EXPECT_NE(std::string(error.what()).find("-81"), std::string::npos);
    }
    worker->finalize();
}

TEST(ChipRunLaneTest, CaptureCarrierCanonicalRefusalCanRetryWithoutAnotherLaunch) {
    auto worker = std::make_shared<ChipWorker>();
    prime_capture_worker(*worker);
    auto run = worker->submit_chip_run(
        1, device_args(), CallConfig{}, nullptr, 0, reinterpret_cast<void *>(1), SIMPLER_PREPARE_CAPTURE
    );
    auto resources = run.retain_resources();
    worker->finalize_run_fn_ = [](void *, void *storage, int) {
        g_events.push_back("finalize" + std::to_string(slot_of(storage)));
        const uint64_t tag = g_finalize_rc[0] != 0 ? kNativeRunContextMagic : 0;
        std::memcpy(storage, &tag, sizeof(tag));
        return g_finalize_rc[0];
    };
    g_finalize_rc[0] = -73;
    EXPECT_THROW(run.activate(), std::runtime_error);
    ASSERT_EQ(worker->native_runs_.size(), 1);
    const auto *entry = worker->native_runs_.begin()->second.get();
    EXPECT_FALSE(entry->destruction_unknown);
    EXPECT_FALSE(entry->finalize_reserved);
    EXPECT_TRUE(run.launched());
    EXPECT_FALSE(run.done());
    EXPECT_EQ(g_span_borrows.size(), 1);
    EXPECT_THROW(resources->release(), std::runtime_error);
    EXPECT_THROW(worker->finalize(), std::runtime_error);
    EXPECT_THROW(run.activate(), std::runtime_error);
    EXPECT_EQ(std::count(g_events.begin(), g_events.end(), "stream_launch0"), 1);
    const auto refused_before = g_events;
    EXPECT_THROW(worker->submit_chip_run(1, device_args(), CallConfig{}), RunPreparationRefused);
    EXPECT_THROW(
        worker->run_lane_->submit(
            1, device_args(), CallConfig{}, PipelineSlotLease{1, 0, 1}, 500, 500, nullptr, 0, true
        ),
        RunPreparationRefused
    );
    EXPECT_EQ(g_events, refused_before);
    EXPECT_EQ(g_span_borrows, (std::multiset<uint64_t>{1}));
    g_finalize_rc[0] = 0;
    try {
        run.abandon();
        FAIL() << "the original cleanup failure remains observable after safe retry";
    } catch (const std::runtime_error &error) {
        EXPECT_NE(std::string(error.what()).find("-73"), std::string::npos);
    }
    EXPECT_TRUE(worker->native_runs_.empty());
    EXPECT_EQ(std::count(g_events.begin(), g_events.end(), "finalize0"), 2);
    EXPECT_EQ(std::count(g_events.begin(), g_events.end(), "stream_launch0"), 1);
    EXPECT_FALSE(run.done());
    EXPECT_EQ(g_span_borrows.size(), 1);
    if (!worker->native_runs_.empty())
        return;  // Failed RED keeps the exact unresolved reference; no fake cleanup proof.
    resources->release();
    EXPECT_TRUE(run.done());
    EXPECT_FALSE(run.owner_recovery_required());
    EXPECT_TRUE(g_span_borrows.empty());
    EXPECT_THROW(run.wait_until(ChipRun::Clock::now()), std::runtime_error);
    worker->finalize();
}

TEST(ChipRunLaneTest, CaptureCarrierContextRefusalCanRetryAfterContextRestoration) {
    auto worker = std::make_shared<ChipWorker>();
    prime_capture_worker(*worker);
    int context = 0;
    worker->device_ctx_ = &context;
    worker->validate_execution_context_fn_ = validate_context;
    auto run = worker->submit_chip_run(
        1, device_args(), CallConfig{}, nullptr, 0, reinterpret_cast<void *>(1), SIMPLER_PREPARE_CAPTURE
    );
    auto resources = run.retain_resources();
    worker->launch_run_on_stream_fn_ = [](void *, void *storage, void *, void *) {
        g_events.push_back("stream_launch" + std::to_string(slot_of(storage)));
        g_context_rc = PTO_RUNTIME_ERR_INVALID_STATE;
        return 0;
    };
    EXPECT_THROW(run.activate(), ExecutionContextRefused);
    ASSERT_EQ(worker->native_runs_.size(), 1);
    EXPECT_FALSE(worker->native_runs_.begin()->second->destruction_unknown);
    EXPECT_FALSE(worker->native_runs_.begin()->second->finalize_reserved);
    EXPECT_EQ(std::count(g_events.begin(), g_events.end(), "finalize0"), 0);
    g_context_rc = 0;
    EXPECT_THROW(run.abandon(), ExecutionContextRefused);
    EXPECT_TRUE(worker->native_runs_.empty());
    EXPECT_EQ(std::count(g_events.begin(), g_events.end(), "stream_launch0"), 1);
    EXPECT_EQ(std::count(g_events.begin(), g_events.end(), "finalize0"), 1);
    if (!worker->native_runs_.empty()) return;  // Preserve unresolved RED state instead of forcing destruction.
    EXPECT_FALSE(run.done());
    resources->release();
    EXPECT_TRUE(run.done());
    EXPECT_TRUE(g_span_borrows.empty());
    worker->device_ctx_ = nullptr;
    worker->finalize();
}

TEST(ChipRunLaneTest, CaptureCarrierRetryKeepsTheOriginalLaunchFailure) {
    auto worker = std::make_shared<ChipWorker>();
    prime_capture_worker(*worker);
    auto run = worker->submit_chip_run(
        1, device_args(), CallConfig{}, nullptr, 0, reinterpret_cast<void *>(1), SIMPLER_PREPARE_CAPTURE
    );
    auto resources = run.retain_resources();
    worker->finalize_run_fn_ = [](void *, void *storage, int) {
        g_events.push_back("finalize" + std::to_string(slot_of(storage)));
        const uint64_t tag = g_finalize_rc[0] != 0 ? kNativeRunContextMagic : 0;
        std::memcpy(storage, &tag, sizeof(tag));
        return g_finalize_rc[0];
    };
    g_launch_rc[0] = -81;
    g_finalize_rc[0] = -73;
    try {
        run.activate();
        FAIL() << "launch and cleanup both fail";
    } catch (const std::runtime_error &error) {
        EXPECT_NE(std::string(error.what()).find("-81"), std::string::npos);
    }
    EXPECT_EQ(g_poll_count, 0);
    g_finalize_rc[0] = 0;
    try {
        run.abandon();
        FAIL() << "the original launch failure remains observable after safe retry";
    } catch (const std::runtime_error &error) {
        EXPECT_NE(std::string(error.what()).find("-81"), std::string::npos);
    }
    ASSERT_TRUE(worker->native_runs_.empty());
    EXPECT_EQ(std::count(g_events.begin(), g_events.end(), "stream_launch0"), 1);
    EXPECT_EQ(std::count(g_events.begin(), g_events.end(), "finalize0"), 2);
    resources->release();
    EXPECT_TRUE(run.done());
    EXPECT_THROW(run.wait_until(ChipRun::Clock::now()), std::runtime_error);
    worker->finalize();
}

TEST(ChipRunLaneTest, CaptureCarrierUnknownDestructionNeverReentersTheCommonFinalizer) {
    auto worker = std::make_shared<ChipWorker>();
    prime_capture_worker(*worker);
    auto run = worker->submit_chip_run(
        1, device_args(), CallConfig{}, nullptr, 0, reinterpret_cast<void *>(1), SIMPLER_PREPARE_CAPTURE
    );
    auto resources = run.retain_resources();
    worker->finalize_run_fn_ = [](void *, void *storage, int) {
        g_events.push_back("finalize" + std::to_string(slot_of(storage)));
        const uint64_t tag = UINT64_MAX;  // Neither known-intact nor known-destroyed.
        std::memcpy(storage, &tag, sizeof(tag));
        return -74;
    };
    EXPECT_THROW(run.activate(), std::runtime_error);
    ASSERT_EQ(worker->native_runs_.size(), 1);
    EXPECT_TRUE(worker->native_runs_.begin()->second->destruction_unknown);
    EXPECT_THROW(run.abandon(), std::runtime_error);
    EXPECT_THROW(run.abandon(), std::runtime_error);
    EXPECT_EQ(std::count(g_events.begin(), g_events.end(), "finalize0"), 1);
    EXPECT_EQ(std::count(g_events.begin(), g_events.end(), "stream_launch0"), 1);
    EXPECT_FALSE(run.done());
    EXPECT_TRUE(run.owner_recovery_required());
    EXPECT_EQ(g_span_borrows.size(), 1);
    EXPECT_THROW(resources->release(), std::runtime_error);
    EXPECT_THROW(worker->finalize(), std::runtime_error);
    std::weak_ptr<ChipWorker> retained = worker;
    resources.reset();  // Exact unknown Backing is deliberately retained, with no SDK/destructor retry.
    worker.reset();
    EXPECT_FALSE(retained.expired());
}

TEST(ChipRunLaneTest, RetainedCaptureLeavesItsOtherGrantAvailableForEagerExecution) {
    auto worker = std::make_shared<ChipWorker>();
    prime_capture_worker(*worker);
    auto model = worker->submit_chip_run(
        1, device_args(), CallConfig{}, nullptr, 0, reinterpret_cast<void *>(1), SIMPLER_PREPARE_CAPTURE
    );
    auto reference = model.retain_resources();
    model.activate();
    const auto before = g_events;
    std::optional<ChipRun> eager;
    try {
        eager.emplace(worker->submit_chip_run(1, device_args(), CallConfig{}));
    } catch (const std::exception &error) {
        ADD_FAILURE() << error.what();
    }
    if (eager) {
        g_complete[0] = true;
        EXPECT_TRUE(eager->wait_until(ChipRun::Clock::now()));
        EXPECT_EQ(g_polled_slots, (std::vector<uint32_t>{0, 0}));
        EXPECT_EQ(
            std::vector<std::string>(g_events.begin() + before.size(), g_events.end()),
            (std::vector<std::string>{"prepare0", "launch0", "wait0", "finalize0"})
        );
        EXPECT_FALSE(model.done());
        EXPECT_EQ(g_span_borrows, (std::multiset<uint64_t>{1}));
        EXPECT_EQ(g_resource_references.size(), 1);
    }
    reference->release();
    EXPECT_TRUE(model.done());
    worker->finalize();
}

TEST(ChipRunLaneTest, FullLiveGrantBesideRetainedModelsRefusesWithoutProgress) {
    auto worker = std::make_shared<ChipWorker>();
    prime_capture_worker(*worker);
    auto a = worker->submit_chip_run(
        1, device_args(), CallConfig{}, nullptr, 0, reinterpret_cast<void *>(1), SIMPLER_PREPARE_CAPTURE
    );
    auto ar = a.retain_resources();
    a.activate();
    auto b = worker->submit_chip_run(
        1, device_args(), CallConfig{}, nullptr, 0, reinterpret_cast<void *>(1), SIMPLER_PREPARE_CAPTURE
    );
    auto br = b.retain_resources();
    b.activate();
    auto first = worker->submit_chip_run(1, device_args(), CallConfig{});
    auto second = worker->run_lane_->submit(
        1, device_args(), CallConfig{}, PipelineSlotLease{1, 0, 1}, 100, 100, nullptr, 0, true
    );
    const auto before = g_events;
    EXPECT_THROW(worker->submit_chip_run(1, device_args(), CallConfig{}), RunPreparationRefused);
    EXPECT_EQ(g_events, before);
    EXPECT_EQ(g_poll_count, 0);
    EXPECT_EQ(g_completion_wait_count, 0);
    EXPECT_EQ(g_span_borrows.size(), 4);
    EXPECT_FALSE(a.done());
    EXPECT_FALSE(b.done());
    g_complete[0] = true;
    EXPECT_TRUE(first.wait_until(ChipRun::Deadline::max()));
    g_complete[1] = true;
    EXPECT_TRUE(second.wait_until(ChipRun::Deadline::max()));
    EXPECT_EQ(g_span_borrows, (std::multiset<uint64_t>{1, 1}));
    ar->release();
    EXPECT_TRUE(a.done());
    EXPECT_FALSE(b.done());
    EXPECT_EQ(g_span_borrows, (std::multiset<uint64_t>{1}));
    br->release();
    EXPECT_TRUE(b.done());
    EXPECT_TRUE(g_span_borrows.empty());
    worker->finalize();
}

TEST(ChipRunLaneTest, UnretiredCaptureRefusesBothEagerAdmissionPathsBeforeSideEffects) {
    auto worker = std::make_shared<ChipWorker>();
    prime_capture_worker(*worker);
    auto pending = worker->submit_chip_run(
        1, device_args(), CallConfig{}, nullptr, 0, reinterpret_cast<void *>(1), SIMPLER_PREPARE_CAPTURE
    );
    const auto before = g_events;
    EXPECT_THROW(worker->submit_chip_run(1, device_args(), CallConfig{}), RunPreparationRefused);
    std::optional<ChipRun> leased;
    try {
        leased.emplace(worker->run_lane_->submit(
            1, device_args(), CallConfig{}, PipelineSlotLease{1, 0, 1}, 100, 100, nullptr, 0, true
        ));
    } catch (const RunPreparationRefused &) {}
    EXPECT_FALSE(leased.has_value());
    EXPECT_EQ(g_events, before);
    EXPECT_EQ(g_span_borrows, (std::multiset<uint64_t>{1}));
    if (leased) leased->abandon();
    pending.abandon();
    worker->finalize();
}

TEST(ChipRunLaneTest, LargerExplicitGrantSelectsOrdinaryJoinAfterMultipleRetainedModels) {
    auto worker = std::make_shared<ChipWorker>();
    prime_capture_worker(*worker, 4);
    worker->launch_depth_ = 2;
    worker->supports_joined_native_launch_fn_ = [](void *) {
        return 1;
    };
    worker->launch_run_joined_fn_ = [](void *, void *runtime, void *predecessor) {
        g_events.push_back("join" + std::to_string(slot_of(runtime)) + "_after" + std::to_string(slot_of(predecessor)));
        return 0;
    };
    auto a = worker->submit_chip_run(
        1, device_args(), CallConfig{}, nullptr, 0, reinterpret_cast<void *>(1), SIMPLER_PREPARE_CAPTURE
    );
    auto ar = a.retain_resources();
    a.activate();
    auto b = worker->submit_chip_run(
        1, device_args(), CallConfig{}, nullptr, 0, reinterpret_cast<void *>(1), SIMPLER_PREPARE_CAPTURE
    );
    auto br = b.retain_resources();
    b.activate();
    const auto before = g_events;
    std::optional<ChipRun> first;
    try {
        first.emplace(worker->submit_chip_run(1, device_args(), CallConfig{}));
    } catch (const std::exception &error) {
        ADD_FAILURE() << error.what();
    }
    if (first) {
        auto second = worker->run_lane_->submit(
            1, device_args(), CallConfig{}, PipelineSlotLease{3, 0, 1}, 200, 200, nullptr, 0, true
        );
        second.activate();
        EXPECT_TRUE(second.launched());
        EXPECT_EQ(
            std::vector<std::string>(g_events.begin() + before.size(), g_events.end()),
            (std::vector<std::string>{"prepare0", "launch0", "prepare3", "join3_after0"})
        );
        g_complete[0] = true;
        g_complete[3] = true;
        EXPECT_TRUE(second.wait_until(ChipRun::Clock::now()));
        EXPECT_TRUE(first->wait_until(ChipRun::Clock::now()));
        for (auto slot : g_polled_slots)
            EXPECT_TRUE(slot == 0 || slot == 3);
        EXPECT_EQ(g_span_borrows, (std::multiset<uint64_t>{1, 1}));
        EXPECT_FALSE(a.done());
        EXPECT_FALSE(b.done());
        auto c = worker->submit_chip_run(1, device_args(), CallConfig{});
        auto d = worker->run_lane_->submit(
            1, device_args(), CallConfig{}, PipelineSlotLease{3, 0, 2}, 300, 300, nullptr, 0, true
        );
        auto e = worker->run_lane_->submit(
            1, device_args(), CallConfig{}, PipelineSlotLease{1, 0, 3}, 301, 301, nullptr, 0, true
        );
        auto f = worker->run_lane_->submit(
            1, device_args(), CallConfig{}, PipelineSlotLease{2, 0, 4}, 302, 302, nullptr, 0, true
        );
        const auto full_before = g_events;
        EXPECT_THROW(worker->submit_chip_run(1, device_args(), CallConfig{}), RunPreparationRefused);
        EXPECT_EQ(g_events, full_before);
        EXPECT_EQ(g_span_borrows.size(), 6);
        g_complete[0] = true;
        g_complete[3] = true;
        EXPECT_TRUE(c.wait_until(ChipRun::Clock::now()));
        EXPECT_TRUE(d.wait_until(ChipRun::Clock::now()));
        g_complete[1] = true;
        g_complete[2] = true;
        EXPECT_TRUE(e.wait_until(ChipRun::Clock::now()));
        EXPECT_TRUE(f.wait_until(ChipRun::Clock::now()));
        EXPECT_EQ(g_span_borrows, (std::multiset<uint64_t>{1, 1}));
    }
    ar->release();
    br->release();
    worker->finalize();
}

TEST(ChipRunLaneTest, RetainedCaptureReleaseDuringCreditWaitRechecksTheExecutingHead) {
    auto worker = std::make_shared<ChipWorker>();
    prime_capture_worker(*worker);
    static int context;
    worker->run_lane_.reset();
    worker->device_ctx_ = &context;
    static std::atomic<uint64_t> generation{1};
    generation.store(1);
    worker->release_run_resources_fn_ = [](void *ctx, void *reference) {
        const int rc = release_resources(ctx, reference);
        if (rc == 0) generation.fetch_add(1);
        return rc;
    };
    worker->credit_generation_fn_ = [](void *) -> uint64_t {
        return generation.load();
    };
    worker->credit_wait_fn_ = [](void *, uint64_t seen, int32_t) {
        EXPECT_EQ(seen, 1);
        std::unique_lock<std::mutex> lock(g_wait_mu);
        g_wait_entered = true;
        g_wait_cv.notify_all();
        g_wait_cv.wait(lock, [] {
            return g_release_wait;
        });
        return generation.load() != seen ? 1 : 0;
    };
    worker->run_lane_ = std::make_unique<ChipRunLane>(*worker);
    auto model = worker->submit_chip_run(
        1, device_args(), CallConfig{}, nullptr, 0, reinterpret_cast<void *>(1), SIMPLER_PREPARE_CAPTURE
    );
    auto reference = model.retain_resources();
    model.activate();
    g_launch_rc[0] = SIMPLER_RUNTIME_ERR_DIAGNOSTIC_CAPACITY_FULL;
    auto eager = worker->submit_chip_run(1, device_args(), CallConfig{});
    {
        std::lock_guard<std::mutex> lock(g_wait_mu);
        g_wait_entered = false;
        g_release_wait = false;
    }
    bool completed = false;
    std::exception_ptr error;
    std::thread waiter([&] {
        try {
            completed = eager.wait_until(ChipRun::Deadline::max());
        } catch (...) {
            error = std::current_exception();
        }
    });
    struct ReleaseAndJoin {
        std::thread &waiter;
        ~ReleaseAndJoin() {
            if (!waiter.joinable()) return;
            {
                std::lock_guard<std::mutex> lock(g_wait_mu);
                g_release_wait = true;
            }
            g_wait_cv.notify_all();
            waiter.join();
        }
    } cleanup{waiter};
    bool entered;
    {
        std::unique_lock<std::mutex> lock(g_wait_mu);
        entered = g_wait_cv.wait_for(lock, std::chrono::seconds(5), [] {
            return g_wait_entered;
        });
    }
    if (entered) {
        // Real checked release takes lane.mu while the actual credit callback
        // is parked outside it, removing the physical model prefix.
        reference->release();
        EXPECT_TRUE(model.done());
    }
    {
        std::lock_guard<std::mutex> lock(g_wait_mu);
        g_launch_rc[0] = 0;
        g_release_wait = true;
    }
    g_wait_cv.notify_all();
    waiter.join();
    EXPECT_TRUE(entered);
    EXPECT_EQ(error, nullptr);
    EXPECT_TRUE(completed);
    EXPECT_EQ(generation.load(), 2);
    EXPECT_EQ(g_completion_waited_slots, (std::vector<uint32_t>{0}));
    EXPECT_EQ(std::count(g_events.begin(), g_events.end(), "finalize0"), 2);
    if (!entered) reference->release();
    worker->device_ctx_ = nullptr;
    worker->finalize();
}

namespace {
NativeSubmissionRef origin_ref(const std::shared_ptr<NativeSubmissionState> &state) {
    auto *holder = new NativeSubmissionCapsule(state);
    NativeSubmissionRef result(holder->api);
    NativeSubmissionCapsule::release(holder);
    return result;
}

struct AclNoticeTransport {
    struct Registration {
        void (*callback)(void *);
        void *cookie;
        void notify() const { callback(cookie); }
    };
    void *active_model;
    uint32_t capture_status{1};
    int registration_rc{0};
    bool hold_query{false};
    std::vector<Registration> registrations;
};

AclNoticeTransport *acl_notice_transport = nullptr;

void bind_acl_notice_transport(ChipWorker &worker, AclNoticeTransport &transport) {
    acl_notice_transport = &transport;
    worker.acl_capture_info_fn_ = [](void *stream, void *status, void **model) {
        if (stream == nullptr) return -1;
        if (acl_notice_transport->hold_query) {
            std::unique_lock<std::mutex> lock(g_wait_mu);
            g_wait_entered = true;
            g_wait_cv.notify_all();
            g_wait_cv.wait(lock, [] {
                return g_release_wait;
            });
        }
        std::memcpy(status, &acl_notice_transport->capture_status, sizeof(uint32_t));
        *model = acl_notice_transport->active_model;
        return 0;
    };
    worker.acl_model_destroy_register_fn_ = [](void *model, void (*callback)(void *), void *cookie) {
        if (model != acl_notice_transport->active_model) return -1;
        acl_notice_transport->registrations.push_back({callback, cookie});
        return acl_notice_transport->registration_rc;
    };
}

std::shared_ptr<NativeSubmissionState>
capture_origin(const std::shared_ptr<ChipWorker> &worker, const CallArgumentSnapshot &values) {
    return worker->prepare_chip_submission(1, values, CallConfig{}, reinterpret_cast<void *>(1));
}
}  // namespace

TEST(ChipRunLaneTest, SharedOriginFrameworkClaimUsesTheSameQueueAndPhysicalTransfer) {
    auto worker = std::make_shared<ChipWorker>();
    prime_capture_worker(*worker);
    enable_framework_transfer(*worker);
    worker->observe_framework_submissions_fn_ = [](void *, int *observed) {
        *observed = 1;
        return 0;
    };
    auto state = capture_origin(worker, CallArgumentSnapshot(device_args(), {}));
    state->claim_framework();
    EXPECT_TRUE(state->framework_claimed());
    state->note_export();
    auto reference = origin_ref(state);
    int started = 1;
    EXPECT_THROW(state->retain_model_resources(started), std::runtime_error);
    EXPECT_FALSE(started);
    EXPECT_TRUE(g_events.empty());
    reference.submit();
    EXPECT_THROW(state->done(), std::logic_error);
    EXPECT_FALSE(state->retire_framework(false, ChipRun::Clock::now()));
    g_complete[0] = true;
    EXPECT_TRUE(state->retire_framework(false, ChipRun::Clock::now()));
    EXPECT_TRUE(state->framework_transferred());
    EXPECT_TRUE(worker->native_runs_.empty());
    EXPECT_EQ(worker->native_resource_references_, 0);
    EXPECT_EQ(g_completion_wait_count, 0);
    EXPECT_EQ(g_events.back(), "transfer0");
    state->retire();
    worker->finalize();
}

TEST(ChipRunLaneTest, SharedOriginCaptureRegistrationCannotBeClaimedByTheEagerObserver) {
    auto worker = std::make_shared<ChipWorker>();
    prime_capture_worker(*worker);
    int model;
    AclNoticeTransport notice{&model};
    bind_acl_notice_transport(*worker, notice);
    enable_framework_transfer(*worker);
    worker->observe_framework_submissions_fn_ = [](void *, int *observed) {
        *observed = 1;
        return 0;
    };
    auto state = capture_origin(worker, CallArgumentSnapshot(device_args(), {}));
    int started = 0;
    state->retain_model_resources(started);
    ASSERT_EQ(started, 1);
    EXPECT_THROW(state->claim_framework(), RunPreparationRefused);
    EXPECT_FALSE(state->framework_claimed());
    auto reference = origin_ref(state);
    reference.submit();
    EXPECT_EQ(worker->native_resource_references_, 1);
    EXPECT_FALSE(state->done());
    NativeSubmissionCleanup cleanup;
    state->cancel_pending(cleanup);
    EXPECT_TRUE(cleanup.carrier_retired);
    ASSERT_EQ(notice.registrations.size(), 1u);
    notice.registrations.front().notify();
    state->release_resources();
    EXPECT_TRUE(state->done());
    EXPECT_FALSE(state->framework_transferred());
    state->retire();
    worker->finalize();
}

TEST(ChipRunLaneTest, OriginStateOwnsTheHostSnapshotBeforeLateRegistration) {
    auto worker = std::make_shared<ChipWorker>();
    prime_capture_worker(*worker);
    int model;
    AclNoticeTransport notice{&model};
    bind_acl_notice_transport(*worker, notice);
    std::shared_ptr<NativeSubmissionState> state;
    {
        int32_t controls[] = {99, 7, 11, 13};
        ChipStorageTaskArgs args{};
        ChipTensor host{};
        host.buffer.addr = reinterpret_cast<uintptr_t>(controls);
        host.buffer.size = sizeof(controls);
        host.shapes[0] = 2;
        host.strides[0] = 2;
        host.start_offset = 1;
        host.ndims = 1;
        host.dtype = DataType::INT32;
        host.address_space = AddressSpace::HOST;
        host.transfer = TensorTransfer::NONE;
        args.add_tensor(host);
        args.add_scalar(41);
        state = capture_origin(worker, CallArgumentSnapshot(args, {ArgDirection::IN}));
        controls[1] = 19;
        args.scalars_[0] = 22;
        EXPECT_TRUE(g_events.empty());
    }
    auto reference = origin_ref(state);
    int started = 0;
    state->retain_model_resources(started);
    ASSERT_TRUE(started);
    EXPECT_EQ(g_host_controls[0], (std::array<int32_t, 2>{7, 13}));
    EXPECT_EQ(g_control_scalars[0], 41);
    EXPECT_EQ(g_prepare_count[0], 1);
    reference.submit();
    EXPECT_TRUE(worker->native_runs_.empty());
    EXPECT_EQ(worker->native_resource_references_, 1);
    EXPECT_FALSE(state->done());
    NativeSubmissionCleanup cleanup;
    state->cancel_pending(cleanup);
    EXPECT_EQ(cleanup.disposition, NativeSubmissionDisposition::Submitted);
    EXPECT_TRUE(cleanup.carrier_retired);
    ASSERT_EQ(notice.registrations.size(), 1u);
    notice.registrations.front().notify();
    state->release_resources();
    EXPECT_TRUE(state->done());
    state->retire();
    worker->finalize();
}

TEST(ChipRunLaneTest, OriginCapsuleViewsShareOneRegistrationAndTheOriginalResourceSlot) {
    auto worker = std::make_shared<ChipWorker>();
    prime_capture_worker(*worker);
    int model;
    AclNoticeTransport notice{&model};
    bind_acl_notice_transport(*worker, notice);
    auto state = capture_origin(worker, CallArgumentSnapshot(device_args(), {}));
    auto first = origin_ref(state);
    auto second = origin_ref(state);
    int started = 0;
    state->retain_model_resources(started);
    ASSERT_TRUE(started);
    int duplicate_started = 1;
    EXPECT_THROW(state->retain_model_resources(duplicate_started), std::runtime_error);
    EXPECT_FALSE(duplicate_started);
    EXPECT_EQ(worker->native_resource_references_, 1);
    EXPECT_EQ(g_resource_references.size(), 1);
    EXPECT_EQ(std::count(g_events.begin(), g_events.end(), "retain0"), 1);
    EXPECT_EQ(std::count(g_events.begin(), g_events.end(), "stream_launch0"), 0);
    first.submit();
    EXPECT_EQ(std::count(g_events.begin(), g_events.end(), "stream_launch0"), 1);
    EXPECT_THROW(second.submit(), std::runtime_error);
    NativeSubmissionCleanup cleanup;
    state->cancel_pending(cleanup);
    EXPECT_TRUE(cleanup.carrier_retired);
    ASSERT_EQ(notice.registrations.size(), 1u);
    notice.registrations.front().notify();
    state->release_resources();
    state->release_resources();
    EXPECT_EQ(std::count(g_events.begin(), g_events.end(), "release0"), 1);
    EXPECT_TRUE(g_span_borrows.empty());
    state->retire();
    worker->finalize();
}

TEST(ChipRunLaneTest, OriginNullAcquireFailureStillOwnsThePreparedCarrierForCancellation) {
    auto worker = std::make_shared<ChipWorker>();
    prime_capture_worker(*worker);
    auto state = capture_origin(worker, CallArgumentSnapshot(device_args(), {}));
    auto reference = origin_ref(state);
    g_retain_rc = -71;
    int started = 0;
    EXPECT_THROW(state->retain_model_resources(started), std::runtime_error);
    EXPECT_TRUE(started);
    EXPECT_EQ(worker->native_resource_references_, 0);
    EXPECT_EQ(worker->native_runs_.size(), 1);
    EXPECT_FALSE(g_span_borrows.empty());
    int duplicate_started = 1;
    EXPECT_THROW(state->retain_model_resources(duplicate_started), std::runtime_error);
    EXPECT_FALSE(duplicate_started);
    EXPECT_THROW(reference.submit(), std::runtime_error);
    EXPECT_EQ(std::count(g_events.begin(), g_events.end(), "stream_launch0"), 0);
    NativeSubmissionCleanup cleanup;
    state->cancel_pending(cleanup);
    EXPECT_EQ(cleanup.disposition, NativeSubmissionDisposition::Submitted);
    EXPECT_TRUE(cleanup.carrier_retired);
    EXPECT_TRUE(worker->native_runs_.empty());
    EXPECT_TRUE(g_span_borrows.empty());
    state->release_resources();
    EXPECT_THROW(state->wait_until(ChipRun::Clock::now()), std::runtime_error);
    state->retire();
    worker->finalize();
}

TEST(ChipRunLaneTest, OriginNonNullAcquireFailureKeepsTheSameWrapperForCheckedReleaseRetry) {
    auto worker = std::make_shared<ChipWorker>();
    prime_capture_worker(*worker);
    auto state = capture_origin(worker, CallArgumentSnapshot(device_args(), {}));
    g_retain_rc = -71;
    g_retain_output_on_error = true;
    int started = 0;
    EXPECT_THROW(state->retain_model_resources(started), std::runtime_error);
    ASSERT_TRUE(started);
    ASSERT_EQ(g_resource_references.size(), 1);
    EXPECT_EQ(worker->native_resource_references_, 1);
    const void *original_reference = g_resource_references.begin()->first;
    NativeSubmissionCleanup cleanup;
    state->cancel_pending(cleanup);
    EXPECT_EQ(cleanup.disposition, NativeSubmissionDisposition::Cancelled);
    EXPECT_TRUE(cleanup.carrier_retired);
    g_release_rc = -72;
    EXPECT_THROW(state->release_resources(), std::runtime_error);
    ASSERT_EQ(g_resource_references.size(), 1);
    EXPECT_EQ(g_resource_references.begin()->first, original_reference);
    EXPECT_EQ(worker->native_resource_references_, 1);
    EXPECT_THROW(state->retire(), RunPreparationRefused);
    g_release_rc = 0;
    state->release_resources();
    EXPECT_TRUE(g_resource_references.empty());
    EXPECT_EQ(worker->native_resource_references_, 0);
    EXPECT_EQ(std::count(g_events.begin(), g_events.end(), "retain0"), 1);
    EXPECT_EQ(std::count(g_events.begin(), g_events.end(), "release0"), 2);
    EXPECT_TRUE(g_span_borrows.empty());
    state->retire();
    worker->finalize();
}

TEST(ChipRunLaneTest, OriginModelQueryRefusalConsumesTheTokenAndStillAllowsCheckedCancellation) {
    auto worker = std::make_shared<ChipWorker>();
    prime_capture_worker(*worker);
    int model;
    AclNoticeTransport notice{&model};
    bind_acl_notice_transport(*worker, notice);
    auto state = capture_origin(worker, CallArgumentSnapshot(device_args(), {}));
    auto reference = origin_ref(state);
    int started = 0;
    state->retain_model_resources(started);
    notice.capture_status = 0;
    EXPECT_THROW(reference.submit(), std::runtime_error);
    notice.capture_status = 1;
    EXPECT_THROW(reference.submit(), std::runtime_error);
    EXPECT_EQ(std::count(g_events.begin(), g_events.end(), "stream_launch0"), 0);
    NativeSubmissionCleanup cleanup;
    state->cancel_pending(cleanup);
    EXPECT_EQ(cleanup.disposition, NativeSubmissionDisposition::Submitted);
    EXPECT_TRUE(cleanup.carrier_retired);
    ASSERT_EQ(notice.registrations.size(), 1u);
    notice.registrations.front().notify();
    state->release_resources();
    EXPECT_TRUE(worker->native_runs_.empty());
    EXPECT_TRUE(g_span_borrows.empty());
    state->retire();
    worker->finalize();
}

TEST(ChipRunLaneTest, OriginSubmittingModelQueryCannotBeCancelledOrReleaseItsResourcesEarly) {
    auto worker = std::make_shared<ChipWorker>();
    prime_capture_worker(*worker);
    int model;
    AclNoticeTransport notice{&model};
    bind_acl_notice_transport(*worker, notice);
    auto state = capture_origin(worker, CallArgumentSnapshot(device_args(), {}));
    auto reference = origin_ref(state);
    int started = 0;
    state->retain_model_resources(started);
    {
        std::lock_guard<std::mutex> lock(g_wait_mu);
        notice.hold_query = true;
        g_release_wait = false;
    }
    std::exception_ptr submission_error;
    std::thread queue([&] {
        try {
            reference.submit();
        } catch (...) {
            submission_error = std::current_exception();
        }
    });
    bool entered;
    {
        std::unique_lock<std::mutex> lock(g_wait_mu);
        entered = g_wait_cv.wait_for(lock, std::chrono::seconds(5), [] {
            return g_wait_entered;
        });
    }
    EXPECT_TRUE(entered);
    if (entered) {
        NativeSubmissionCleanup cleanup;
        state->cancel_pending(cleanup);
        EXPECT_EQ(cleanup.disposition, NativeSubmissionDisposition::Submitting);
        EXPECT_FALSE(cleanup.carrier_retired);
        EXPECT_THROW(state->release_resources(), std::runtime_error);
        EXPECT_EQ(worker->native_resource_references_, 1);
        EXPECT_EQ(std::count(g_events.begin(), g_events.end(), "finalize0"), 0);
    }
    {
        std::lock_guard<std::mutex> lock(g_wait_mu);
        g_release_wait = true;
    }
    g_wait_cv.notify_all();
    queue.join();
    EXPECT_EQ(submission_error, nullptr);
    EXPECT_EQ(std::count(g_events.begin(), g_events.end(), "stream_launch0"), 1);
    NativeSubmissionCleanup cleanup;
    state->cancel_pending(cleanup);
    EXPECT_TRUE(cleanup.carrier_retired);
    ASSERT_EQ(notice.registrations.size(), 1u);
    notice.registrations.front().notify();
    state->release_resources();
    state->retire();
    worker->finalize();
}

TEST(ChipRunLaneTest, OriginCleanupProgressDoesNotReplaceTheHistoricalLaunchError) {
    auto worker = std::make_shared<ChipWorker>();
    prime_capture_worker(*worker);
    int model;
    AclNoticeTransport notice{&model};
    bind_acl_notice_transport(*worker, notice);
    auto state = capture_origin(worker, CallArgumentSnapshot(device_args(), {}));
    auto reference = origin_ref(state);
    int started = 0;
    state->retain_model_resources(started);
    worker->finalize_run_fn_ = [](void *, void *storage, int) {
        g_events.push_back("finalize" + std::to_string(slot_of(storage)));
        const uint64_t tag = g_finalize_rc[0] != 0 ? kNativeRunContextMagic : 0;
        std::memcpy(storage, &tag, sizeof(tag));
        return g_finalize_rc[0];
    };
    g_launch_rc[0] = -81;
    g_finalize_rc[0] = -73;
    EXPECT_THROW(reference.submit(), std::runtime_error);
    EXPECT_THROW(state->release_resources(), std::runtime_error);
    NativeSubmissionCleanup cleanup;
    EXPECT_THROW(state->cancel_pending(cleanup), std::runtime_error);
    EXPECT_EQ(cleanup.disposition, NativeSubmissionDisposition::Submitted);
    EXPECT_FALSE(cleanup.carrier_retired);
    EXPECT_EQ(worker->native_resource_references_, 1);
    g_finalize_rc[0] = 0;
    EXPECT_NO_THROW(state->cancel_pending(cleanup));
    EXPECT_TRUE(cleanup.carrier_retired);
    EXPECT_EQ(std::count(g_events.begin(), g_events.end(), "stream_launch0"), 1);
    EXPECT_EQ(std::count(g_events.begin(), g_events.end(), "finalize0"), 3);
    EXPECT_EQ(g_poll_count, 0);
    try {
        state->wait_until(ChipRun::Clock::now());
        FAIL() << "the original error remains observable after successful cleanup";
    } catch (const std::runtime_error &error) {
        EXPECT_NE(std::string(error.what()).find("-81"), std::string::npos);
        EXPECT_EQ(std::string(error.what()).find("-73"), std::string::npos);
    }
    ASSERT_EQ(notice.registrations.size(), 1u);
    notice.registrations.front().notify();
    state->release_resources();
    state->retire();
    worker->finalize();
}

TEST(ChipRunLaneTest, OriginPrepareFailureKeepsItsActualCleanRollbackProof) {
    auto worker = std::make_shared<ChipWorker>();
    prime_capture_worker(*worker);
    auto state = capture_origin(worker, CallArgumentSnapshot(device_args(), {}));
    g_prepare_rc[0] = -72;
    int started = 0;
    EXPECT_THROW(state->retain_model_resources(started), std::runtime_error);
    EXPECT_TRUE(started);
    EXPECT_EQ(g_prepare_count[0], 1);
    EXPECT_EQ(std::count(g_events.begin(), g_events.end(), "retain0"), 0);
    EXPECT_TRUE(worker->native_runs_.empty());
    EXPECT_TRUE(g_span_borrows.empty());
    NativeSubmissionCleanup cleanup;
    state->cancel_pending(cleanup);
    EXPECT_EQ(cleanup.disposition, NativeSubmissionDisposition::Cancelled);
    EXPECT_TRUE(cleanup.carrier_retired);
    state->release_resources();
    state->retire();
    worker->finalize();
}

TEST(ChipRunLaneTest, OriginUnknownPrepareRollbackPinsTheExactNativeOwnersWithoutDestructorRetry) {
    auto worker = std::make_shared<ChipWorker>();
    prime_capture_worker(*worker);
    std::weak_ptr<ChipWorker> observed = worker;
    auto state = capture_origin(worker, CallArgumentSnapshot(device_args(), {}));
    auto reference = std::make_unique<NativeSubmissionRef>(origin_ref(state));
    g_prepare_rc[0] = -72;
    g_prepare_release_proven = false;
    int started = 0;
    EXPECT_THROW(state->retain_model_resources(started), std::runtime_error);
    EXPECT_TRUE(started);
    NativeSubmissionCleanup cleanup;
    state->cancel_pending(cleanup);
    EXPECT_FALSE(cleanup.carrier_retired);
    EXPECT_THROW(state->release_resources(), std::runtime_error);
    EXPECT_THROW(state->retire(), RunPreparationRefused);
    EXPECT_EQ(g_span_borrows.size(), 1);
    auto events = g_events;
    state.reset();
    worker.reset();
    std::thread queue([reference = std::move(reference)]() mutable {
        reference.reset();
    });
    queue.join();
    EXPECT_FALSE(observed.expired());
    EXPECT_EQ(g_events, events);
    EXPECT_EQ(g_span_borrows.size(), 1);
}

TEST(ChipRunLaneTest, OriginRetiredTokenRejectsRegistrationBeforeTouchingTheReleasedWorkerPin) {
    auto worker = std::make_shared<ChipWorker>();
    prime_capture_worker(*worker);
    int context = 0;
    worker->device_ctx_ = &context;
    worker->validate_execution_context_fn_ = validate_context;
    auto state = capture_origin(worker, CallArgumentSnapshot(device_args(), {}));
    NativeSubmissionCleanup cleanup;
    state->cancel_pending(cleanup);
    ASSERT_TRUE(cleanup.carrier_retired);
    state->retire();
    const size_t checks = g_context_checks;
    g_context_rc = -89;
    int started = 1;
    EXPECT_THROW(state->retain_model_resources(started), std::runtime_error);
    EXPECT_FALSE(started);
    EXPECT_EQ(g_context_checks, checks);
    g_context_rc = 0;
    std::weak_ptr<ChipWorker> observed = worker;
    worker->device_ctx_ = nullptr;
    worker.reset();
    EXPECT_TRUE(observed.expired());
    started = 1;
    EXPECT_THROW(state->retain_model_resources(started), std::runtime_error);
    EXPECT_FALSE(started);
}

TEST(ChipRunLaneTest, OriginSnapshotOnlyFactoryStillEnforcesOrdinaryStreamAdmissionAtSubmit) {
    auto worker = std::make_shared<ChipWorker>();
    prime_capture_worker(*worker);
    size_t stream_checks = 0;
    worker->device_ctx_ = &stream_checks;
    worker->validate_caller_stream_fn_ = [](void *context, void *) {
        ++*static_cast<size_t *>(context);
        return -66;
    };
    auto state = worker->prepare_chip_submission(
        1, CallArgumentSnapshot(device_args(), {}), CallConfig{}, reinterpret_cast<void *>(1)
    );
    EXPECT_EQ(stream_checks, 0);
    EXPECT_TRUE(g_events.empty());
    auto reference = origin_ref(state);
    EXPECT_THROW(reference.submit(), std::runtime_error);
    EXPECT_EQ(stream_checks, 1);
    EXPECT_TRUE(g_events.empty());
    int started = 1;
    EXPECT_THROW(state->retain_model_resources(started), std::runtime_error);
    EXPECT_FALSE(started);
    state->retire();
    worker->device_ctx_ = nullptr;
    worker->finalize();
}

TEST(ChipRunLaneTest, OriginRegistrationGateRefusesSubmitAndReportsInFlightToEveryCleanupPath) {
    auto worker = std::make_shared<ChipWorker>();
    prime_capture_worker(*worker);
    int model;
    AclNoticeTransport notice{&model};
    bind_acl_notice_transport(*worker, notice);
    auto state = capture_origin(worker, CallArgumentSnapshot(device_args(), {}));
    auto first = origin_ref(state);
    auto second = origin_ref(state);
    {
        std::lock_guard<std::mutex> lock(g_wait_mu);
        g_hold_retain = true;
        g_release_wait = false;
    }
    int started = 0;
    std::exception_ptr registration_error;
    std::thread registration([&] {
        try {
            state->retain_model_resources(started);
        } catch (...) {
            registration_error = std::current_exception();
        }
    });
    bool entered;
    {
        std::unique_lock<std::mutex> lock(g_wait_mu);
        entered = g_wait_cv.wait_for(lock, std::chrono::seconds(5), [] {
            return g_wait_entered;
        });
    }
    EXPECT_TRUE(entered);
    if (entered) {
        EXPECT_THROW(second.submit(), std::runtime_error);
        int duplicate_started = 1;
        EXPECT_THROW(state->retain_model_resources(duplicate_started), std::runtime_error);
        EXPECT_FALSE(duplicate_started);
        NativeSubmissionCleanup cleanup;
        state->cancel_pending(cleanup);
        EXPECT_EQ(cleanup.disposition, NativeSubmissionDisposition::Submitting);
        EXPECT_FALSE(cleanup.carrier_retired);
        EXPECT_FALSE(state->cancel(std::make_exception_ptr(RunPreparationRefused("concurrent cancel"))));
        EXPECT_THROW(state->release_resources(), std::runtime_error);
        EXPECT_THROW(state->retire(), RunPreparationRefused);
        EXPECT_EQ(std::count(g_events.begin(), g_events.end(), "stream_launch0"), 0);
        EXPECT_EQ(std::count(g_events.begin(), g_events.end(), "finalize0"), 0);
        EXPECT_EQ(std::count(g_events.begin(), g_events.end(), "release0"), 0);
    }
    {
        std::lock_guard<std::mutex> lock(g_wait_mu);
        g_release_wait = true;
    }
    g_wait_cv.notify_all();
    registration.join();
    EXPECT_EQ(registration_error, nullptr);
    EXPECT_TRUE(started);
    EXPECT_EQ(g_prepare_count[0], 1);
    EXPECT_EQ(std::count(g_events.begin(), g_events.end(), "retain0"), 1);
    first.submit();
    EXPECT_EQ(std::count(g_events.begin(), g_events.end(), "stream_launch0"), 1);
    NativeSubmissionCleanup cleanup;
    state->cancel_pending(cleanup);
    EXPECT_TRUE(cleanup.carrier_retired);
    ASSERT_EQ(notice.registrations.size(), 1u);
    notice.registrations.front().notify();
    state->release_resources();
    state->retire();
    worker->finalize();
}

namespace {
struct OriginContextGate {
    std::mutex mutex;
    std::condition_variable changed;
    bool hold_next{true};
    size_t destroys{0};
    bool entered{false};
    bool proceed{false};
    static int validate(void *context) {
        auto &gate = *static_cast<OriginContextGate *>(context);
        std::unique_lock<std::mutex> lock(gate.mutex);
        if (!gate.hold_next) return 0;
        gate.hold_next = false;
        gate.entered = true;
        gate.changed.notify_all();
        gate.changed.wait(lock, [&] {
            return gate.proceed;
        });
        return 0;
    }
    bool wait_for_entry() {
        std::unique_lock<std::mutex> lock(mutex);
        return changed.wait_for(lock, std::chrono::seconds(5), [&] {
            return entered;
        });
    }
    void release() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            proceed = true;
        }
        changed.notify_all();
    }
};
}  // namespace

TEST(ChipRunLaneTest, OriginOrdinarySubmitCanWinBeforeRegistrationClaimsWithoutChangingItsMode) {
    auto worker = std::make_shared<ChipWorker>();
    prime_capture_worker(*worker);
    auto state = capture_origin(worker, CallArgumentSnapshot(device_args(), {}));
    auto reference = origin_ref(state);
    OriginContextGate gate;
    worker->device_ctx_ = &gate;
    worker->validate_execution_context_fn_ = OriginContextGate::validate;
    int started = 1;
    std::exception_ptr registration_error;
    std::thread registration([&] {
        try {
            state->retain_model_resources(started);
        } catch (...) {
            registration_error = std::current_exception();
        }
    });
    const bool entered = gate.wait_for_entry();
    EXPECT_TRUE(entered);
    if (entered) EXPECT_NO_THROW(reference.submit());
    gate.release();
    registration.join();
    EXPECT_NE(registration_error, nullptr);
    EXPECT_FALSE(started);
    EXPECT_EQ(g_prepare_purpose[0], SIMPLER_PREPARE_EAGER);
    EXPECT_EQ(g_prepare_count[0], 1);
    EXPECT_EQ(std::count(g_events.begin(), g_events.end(), "retain0"), 0);
    EXPECT_EQ(std::count(g_events.begin(), g_events.end(), "stream_launch0"), 1);
    g_complete[0] = true;
    EXPECT_TRUE(state->wait_until(ChipRun::Clock::now()));
    state->retire();
    worker->device_ctx_ = nullptr;
    worker->finalize();
}

TEST(ChipRunLaneTest, OriginCancellationCanWinBeforeRegistrationClaimAndKeepsItsLocalWorkerPin) {
    auto worker = std::make_shared<ChipWorker>();
    prime_capture_worker(*worker);
    auto state = capture_origin(worker, CallArgumentSnapshot(device_args(), {}));
    OriginContextGate gate;
    worker->device_ctx_ = &gate;
    worker->validate_execution_context_fn_ = OriginContextGate::validate;
    worker->destroy_device_context_fn_ = [](void *context) {
        ++static_cast<OriginContextGate *>(context)->destroys;
    };
    int started = 1;
    std::exception_ptr registration_error;
    std::thread registration([&] {
        try {
            state->retain_model_resources(started);
        } catch (...) {
            registration_error = std::current_exception();
        }
    });
    const bool entered = gate.wait_for_entry();
    EXPECT_TRUE(entered);
    std::weak_ptr<ChipWorker> observed = worker;
    if (entered) {
        NativeSubmissionCleanup cleanup;
        state->cancel_pending(cleanup);
        EXPECT_EQ(cleanup.disposition, NativeSubmissionDisposition::Cancelled);
        EXPECT_TRUE(cleanup.carrier_retired);
        state->retire();
        worker.reset();
        EXPECT_FALSE(observed.expired());
    }
    gate.release();
    registration.join();
    EXPECT_NE(registration_error, nullptr);
    EXPECT_FALSE(started);
    EXPECT_TRUE(g_events.empty());
    EXPECT_TRUE(observed.expired());
    EXPECT_EQ(gate.destroys, 1);
}

TEST(ChipRunLaneTest, DefaultGrantRecordsFiveTokensWhileEveryModelReferenceStaysLive) {
    auto worker = std::make_shared<ChipWorker>();
    prime_capture_worker(*worker);
    int model;
    AclNoticeTransport notice{&model};
    bind_acl_notice_transport(*worker, notice);
    std::vector<std::shared_ptr<NativeSubmissionState>> states;
    std::vector<NativeSubmissionRef> references;
    size_t submitted = 0;
    for (size_t index = 0; index < 5; ++index) {
        states.push_back(capture_origin(worker, CallArgumentSnapshot(ChipStorageTaskArgs{}, {})));
        references.push_back(origin_ref(states.back()));
        int started = 0;
        try {
            states.back()->retain_model_resources(started, index == 0 ? nullptr : states.front().get());
            EXPECT_TRUE(started);
            references.back().submit();
            ++submitted;
            EXPECT_TRUE(worker->native_runs_.empty());
            EXPECT_EQ(worker->native_resource_references_, submitted);
            for (const auto &state : states)
                EXPECT_FALSE(state->done());
            const auto events = g_events;
            EXPECT_THROW(worker->finalize(), std::runtime_error);
            EXPECT_EQ(g_events, events);
        } catch (const std::exception &error) {
            ADD_FAILURE() << "node " << index << ": " << error.what();
            break;
        }
    }
    EXPECT_EQ(submitted, 5u);
    EXPECT_EQ(g_poll_count, 0);
    EXPECT_EQ(g_completion_wait_count, 0);
    ASSERT_EQ(notice.registrations.size(), 1u);
    notice.registrations.front().notify();
    for (size_t index = 0; index < references.size(); ++index) {
        NativeSubmissionCleanup cleanup;
        states[index]->cancel_pending(cleanup);
        EXPECT_TRUE(cleanup.carrier_retired);
        states[index]->release_resources();
        states[index]->retire();
    }
    EXPECT_EQ(worker->native_resource_references_, 0);
    worker->finalize();
}

namespace {
void cancel_capture_carrier(NativeSubmissionState &state) {
    NativeSubmissionCleanup cleanup;
    state.cancel_pending(cleanup);
    EXPECT_TRUE(cleanup.carrier_retired);
}
}  // namespace

TEST(ChipRunLaneTest, AclModelNoticeHoldsAllRetiredCarriersUntilTheSharedCallback) {
    int model;
    AclNoticeTransport transport{&model};
    auto worker = std::make_shared<ChipWorker>();
    prime_capture_worker(*worker);
    bind_acl_notice_transport(*worker, transport);
    std::vector<std::shared_ptr<NativeSubmissionState>> states;
    for (int i = 0; i < 3; ++i) {
        auto state = capture_origin(worker, CallArgumentSnapshot(device_args(), {}));
        int started = 0;
        state->retain_model_resources(started, states.empty() ? nullptr : states.front().get());
        EXPECT_EQ(started, 1);
        state->submit();
        EXPECT_TRUE(worker->native_runs_.empty());
        EXPECT_FALSE(state->done());
        EXPECT_FALSE(state->model_destroyed());
        const auto before = g_events;
        EXPECT_THROW(state->release_resources(), RunPreparationRefused);
        EXPECT_EQ(g_events, before);
        states.push_back(std::move(state));
    }
    ASSERT_EQ(transport.registrations.size(), 1u);
    ASSERT_EQ(worker->native_resource_references_, states.size());
    const auto before = g_events;
    const auto callback = transport.registrations.front();
    std::thread destroy([&] {
        // A runtime lock held across the callback cannot become part of
        // checked Simpler resource release.
        std::lock_guard<std::mutex> lock(worker->native_run_mu_);
        callback.notify();
    });
    destroy.join();
    EXPECT_EQ(g_events, before);
    EXPECT_EQ(worker->native_resource_references_, states.size());
    for (auto &state : states) {
        EXPECT_TRUE(state->model_destroyed());
        state->release_resources();
        EXPECT_TRUE(state->done());
        const auto released = g_events;
        state->release_resources();
        EXPECT_EQ(g_events, released);
        state->retire();
    }
    EXPECT_EQ(worker->native_resource_references_, 0u);
    worker->finalize();
}

TEST(ChipRunLaneTest, AclModelNoticeCannotReplaceCarrierRetirement) {
    int model;
    AclNoticeTransport transport{&model};
    auto worker = std::make_shared<ChipWorker>();
    prime_capture_worker(*worker);
    bind_acl_notice_transport(*worker, transport);
    auto state = capture_origin(worker, CallArgumentSnapshot(device_args(), {}));
    int started = 0;
    state->retain_model_resources(started);
    ASSERT_EQ(transport.registrations.size(), 1u);
    transport.registrations.front().notify();
    EXPECT_TRUE(state->model_destroyed());
    const auto before = g_events;
    EXPECT_THROW(state->submit(), RunPreparationRefused);
    EXPECT_THROW(state->release_resources(), RunPreparationRefused);
    EXPECT_EQ(g_events, before);
    EXPECT_EQ(worker->native_resource_references_, 1u);
    cancel_capture_carrier(*state);
    state->release_resources();
    state->retire();
    EXPECT_EQ(worker->native_resource_references_, 0u);
    worker->finalize();
}

TEST(ChipRunLaneTest, AclModelHandleReuseRequiresTheEarlierDestroyNotice) {
    int model;
    AclNoticeTransport transport{&model};
    auto worker = std::make_shared<ChipWorker>();
    prime_capture_worker(*worker);
    bind_acl_notice_transport(*worker, transport);
    int started = 0;
    auto first = capture_origin(worker, CallArgumentSnapshot(device_args(), {}));
    first->retain_model_resources(started);
    first->submit();
    auto uncertain = capture_origin(worker, CallArgumentSnapshot(device_args(), {}));
    EXPECT_THROW(uncertain->retain_model_resources(started), RunPreparationRefused);
    EXPECT_EQ(started, 1);
    cancel_capture_carrier(*uncertain);
    uncertain->release_resources();
    uncertain->retire();
    ASSERT_EQ(transport.registrations.size(), 1u);
    EXPECT_EQ(worker->native_resource_references_, 1u);

    transport.registrations.front().notify();
    auto successor = capture_origin(worker, CallArgumentSnapshot(device_args(), {}));
    successor->retain_model_resources(started);
    successor->submit();
    ASSERT_EQ(transport.registrations.size(), 2u);
    EXPECT_NE(transport.registrations[0].cookie, transport.registrations[1].cookie);
    EXPECT_TRUE(first->model_destroyed());
    EXPECT_FALSE(successor->model_destroyed());
    first->release_resources();
    first->retire();
    EXPECT_THROW(successor->release_resources(), RunPreparationRefused);
    transport.registrations.back().notify();
    successor->release_resources();
    successor->retire();
    EXPECT_EQ(worker->native_resource_references_, 0u);
    worker->finalize();
}

TEST(ChipRunLaneTest, AclModelAnchorAndSubmitRefuseAnotherActiveModel) {
    int model, other_model;
    AclNoticeTransport transport{&model};
    auto worker = std::make_shared<ChipWorker>();
    prime_capture_worker(*worker);
    bind_acl_notice_transport(*worker, transport);
    int started = 0;
    auto first = capture_origin(worker, CallArgumentSnapshot(device_args(), {}));
    first->retain_model_resources(started);
    auto wrong_stream = worker->prepare_chip_submission(
        1, CallArgumentSnapshot(device_args(), {}), CallConfig{}, reinterpret_cast<void *>(2)
    );
    EXPECT_THROW(wrong_stream->retain_model_resources(started, first.get()), RunPreparationRefused);
    cancel_capture_carrier(*wrong_stream);
    wrong_stream->release_resources();
    wrong_stream->retire();

    transport.active_model = &other_model;
    auto wrong = capture_origin(worker, CallArgumentSnapshot(device_args(), {}));
    EXPECT_THROW(wrong->retain_model_resources(started, first.get()), RunPreparationRefused);
    cancel_capture_carrier(*wrong);
    wrong->release_resources();
    wrong->retire();
    const auto before = g_events;
    EXPECT_THROW(first->submit(), RunPreparationRefused);
    EXPECT_EQ(g_events, before);
    ASSERT_EQ(transport.registrations.size(), 1u);
    cancel_capture_carrier(*first);
    EXPECT_THROW(first->release_resources(), RunPreparationRefused);
    transport.registrations.front().notify();
    first->release_resources();
    first->retire();
    worker->finalize();
}

TEST(ChipRunLaneTest, AclModelRegistrationFailurePreservesTheCallbackCookie) {
    int model;
    AclNoticeTransport transport{&model};
    transport.registration_rc = -17;
    {
        auto worker = std::make_shared<ChipWorker>();
        prime_capture_worker(*worker);
        bind_acl_notice_transport(*worker, transport);
        auto state = capture_origin(worker, CallArgumentSnapshot(device_args(), {}));
        int started = 0;
        EXPECT_THROW(state->retain_model_resources(started), RunPreparationRefused);
        EXPECT_EQ(started, 1);
        cancel_capture_carrier(*state);
        state->release_resources();
        state->retire();
        EXPECT_EQ(worker->native_resource_references_, 0u);
        worker->finalize();
    }
    // An ambiguous registration failure can leave RTS holding its cookie
    // even though no capture emission or native run ownership escaped.
    ASSERT_EQ(transport.registrations.size(), 1u);
    transport.registrations.front().notify();
    transport.registration_rc = 0;
    auto worker = std::make_shared<ChipWorker>();
    prime_capture_worker(*worker);
    bind_acl_notice_transport(*worker, transport);
    auto state = capture_origin(worker, CallArgumentSnapshot(device_args(), {}));
    int started = 0;
    state->retain_model_resources(started);
    ASSERT_EQ(transport.registrations.size(), 2u);
    cancel_capture_carrier(*state);
    transport.registrations.back().notify();
    state->release_resources();
    state->retire();
    worker->finalize();
}

TEST(ChipRunLaneTest, AclModelNoticeKeepsTheCheckedReleaseFailureRetryable) {
    int model;
    AclNoticeTransport transport{&model};
    auto worker = std::make_shared<ChipWorker>();
    prime_capture_worker(*worker);
    bind_acl_notice_transport(*worker, transport);
    auto state = capture_origin(worker, CallArgumentSnapshot(device_args(), {}));
    int started = 0;
    state->retain_model_resources(started);
    state->submit();
    ASSERT_EQ(transport.registrations.size(), 1u);
    transport.registrations.front().notify();
    g_release_rc = -72;
    EXPECT_THROW(state->release_resources(), std::runtime_error);
    EXPECT_EQ(worker->native_resource_references_, 1u);
    EXPECT_TRUE(state->model_destroyed());
    g_release_rc = 0;
    state->release_resources();
    state->retire();
    EXPECT_EQ(worker->native_resource_references_, 0u);
    worker->finalize();
}

TEST(ChipRunLaneTest, AclModelRetentionRefusesInactiveCaptureWithoutEmitting) {
    int model;
    AclNoticeTransport transport{&model};
    transport.capture_status = 0;
    auto worker = std::make_shared<ChipWorker>();
    prime_capture_worker(*worker);
    bind_acl_notice_transport(*worker, transport);
    auto state = capture_origin(worker, CallArgumentSnapshot(device_args(), {}));
    int started = 0;
    EXPECT_THROW(state->retain_model_resources(started), RunPreparationRefused);
    EXPECT_EQ(started, 1);
    EXPECT_TRUE(transport.registrations.empty());
    const auto before = g_events;
    EXPECT_THROW(state->submit(), RunPreparationRefused);
    EXPECT_EQ(g_events, before);
    cancel_capture_carrier(*state);
    state->release_resources();
    state->retire();
    worker->finalize();
}
