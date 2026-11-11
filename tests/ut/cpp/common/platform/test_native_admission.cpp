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
/** Real onboard workspace adapters and native prepare ownership over CPU transport. */
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <new>
#include <limits>
#include <array>
#include <chrono>
#include <future>
#include <fstream>
#include <iterator>
#include <optional>
#include <string>
#include <string_view>
#include "aicpu/scope_stats_collector_aicpu.h"
#include "aicpu/chip_swimlane_collector_aicpu.h"
#include "aicpu/device_run_result_base_aicpu.h"
#include "aicpu_topology_probe.h"
#include "context_preparation.h"
#include "callable.h"
#include "fixture.h"
#include "native_run_context.h"
#include "host/raii_scope_guard.h"
#include "acl/acl.h"
#define private public
#include "chip_worker.h"
#undef private
#include "chip_run_lane.h"

extern "C" int args_dump_capture_supported_impl();

namespace native_admission_test {
thread_local bool refuse_next_host_allocation = false;
thread_local size_t refuse_host_allocation_after = std::numeric_limits<size_t>::max();
thread_local size_t refused_host_allocation_bytes = 0;
thread_local size_t refused_host_allocation_count = 0;
}  // namespace native_admission_test

// Only the throwing, unaligned allocation is replaced. Its ordinary delete
// accepts malloc storage; aligned allocations retain their own allocator pair.
void *operator new(std::size_t bytes) {
    bool counted_failure = false;
    auto &remaining = native_admission_test::refuse_host_allocation_after;
    if (remaining != std::numeric_limits<size_t>::max() && remaining-- == 0) {
        remaining = std::numeric_limits<size_t>::max();
        counted_failure = true;
    }
    if (std::exchange(native_admission_test::refuse_next_host_allocation, false) || counted_failure) {
        native_admission_test::refused_host_allocation_bytes = bytes;
        ++native_admission_test::refused_host_allocation_count;
        throw std::bad_alloc();
    }
    if (void *storage = std::malloc(bytes == 0 ? 1 : bytes)) return storage;
    throw std::bad_alloc();
}

#if defined(__linux__)
namespace {
struct SwimlaneRecordGate {
    std::mutex mutex;
    std::condition_variable changed;
    bool entered{false};
    bool released{false};
    void arrive() {
        std::unique_lock<std::mutex> lock(mutex);
        entered = true;
        changed.notify_all();
        changed.wait(lock, [this] {
            return released;
        });
    }
    void release() {
        std::lock_guard<std::mutex> lock(mutex);
        released = true;
        changed.notify_all();
    }
    bool wait_entered() {
        std::unique_lock<std::mutex> lock(mutex);
        return changed.wait_for(lock, std::chrono::seconds(5), [this] {
            return entered;
        });
    }
};
std::atomic<SwimlaneRecordGate *> swimlane_record_gate{nullptr};
}  // namespace
extern "C" void *__real__ZnwmSt11align_val_t(size_t bytes, std::align_val_t alignment);
extern "C" void *__wrap__ZnwmSt11align_val_t(size_t bytes, std::align_val_t alignment) {
    if (bytes == sizeof(CollectedRecord<ChipSwimlaneAicoreTaskRecord>)) {
        if (auto *gate = swimlane_record_gate.exchange(nullptr)) gate->arrive();
    }
    return __real__ZnwmSt11align_val_t(bytes, alignment);
}
#endif

class NativeRunExecutionTestPeer {
public:
    static LaunchPermit mint(const NativeRunIdentity &identity) { return LaunchPermit(identity); }
};

namespace native_admission_test {
namespace {
int fake_current_device(void *, int *device_id) noexcept {
    *device_id = 0;
    return 0;
}
int fake_current_context(void *, void **context) noexcept {
    static int token;
    *context = &token;
    return 0;
}
int fake_create_hidden_stream(void *, void **stream) noexcept {
    static int tokens[2];
    static int next;
    *stream = &tokens[next++ % 2];
    return 0;
}
int fake_destroy_hidden_stream(void *, void *) noexcept { return 0; }
int fake_create_event(void *, void **event) noexcept {
    static int tokens[8];
    static int next;
    *event = &tokens[next++ % 8];
    return 0;
}
int fake_destroy_event(void *, void *) noexcept { return 0; }

int initialize_capture_runner(Runner &runner) {
    int rc = runner.execution_mode_latch().latch(SIMPLER_MODE_KERNEL);
    if (rc != 0) return rc;
    KernelContextOps ops{&runner,
                         fake_current_device,
                         fake_current_context,
                         fake_create_hidden_stream,
                         fake_destroy_hidden_stream,
                         fake_create_event,
                         fake_destroy_event};
    rc = runner.kernel_execution_state().initialize(0, ops);
    if (rc != 0) return rc;
    rc = runner.warm_capture_result_caches();
    return rc != 0 ? rc : runner.kernel_execution_state().mark_ready_enqueued();
}

void use_capture_preparation(Attempt &attempt) {
    attempt.descriptor.prepare_purpose = SIMPLER_PREPARE_CAPTURE;
    attempt.args.tensor(0).address_space = AddressSpace::DEVICE;
    attempt.args.tensor(0).transfer = TensorTransfer::NONE;
}

class TimeoutEnvironment {
public:
    explicit TimeoutEnvironment(const std::array<const char *, 3> &values) {
        for (size_t i = 0; i < names_.size(); ++i) {
            if (const char *old = std::getenv(names_[i])) saved_[i] = old;
            if (values[i] != nullptr) setenv(names_[i], values[i], 1);
            else unsetenv(names_[i]);
        }
    }
    ~TimeoutEnvironment() {
        for (size_t i = 0; i < names_.size(); ++i) {
            if (saved_[i]) setenv(names_[i], saved_[i]->c_str(), 1);
            else unsetenv(names_[i]);
        }
    }

private:
    static constexpr std::array<const char *, 3> names_{
        SIMPLER_OP_EXECUTE_TIMEOUT_US_ENV, SIMPLER_STREAM_SYNC_TIMEOUT_MS_ENV, SIMPLER_SCHEDULER_TIMEOUT_MS_ENV
    };
    std::array<std::optional<std::string>, 3> saved_;
};

class CallerOrderRunner : public Runner {
public:
    RunCallerOrder &caller_order(uint32_t slot) { return *submission_storage_[slot]->caller_order; }
};

// Admission follows the onboard arming prologue; transport supplies only
// kernel submission and completion in the inherited runner.
class ArmedAdmissionRunner : public CallerOrderRunner {
public:
    LaunchTransactionResult launch_execution_impl(PreparedExecution &prepared, LaunchPermit permit) override {
        prepared.streams =
            execution_mode_latch().is_kernel() ? borrowed_run_streams() : RunStreams{&cpu_stream_, &core_stream_};
        const int rc = arm_run_fence(prepared);
        if (rc != 0) return {rc, LaunchProgress::NotStarted, {}};
        return Runner::launch_execution_impl(prepared, std::move(permit));
    }

private:
    int cpu_stream_{0};
    int core_stream_{0};
};

class TimeoutRunner : public Runner {
public:
    TimeoutRunner() {
        device_id_ = -1;
        callables_.clear();
        chip_callable_buffers_.clear();
    }
    HostRuntimeTimeoutConfig timeouts() const { return timeout_config_; }
};

// Only kernel execution is supplied by the transport. The C API, boundary
// fence, strong observer, result reader and runtime copy-back stay real.
class FencedCompletionRunner : public Runner {
public:
    using Runner::Runner;
    bool publish_success{true};
    int published_error{0};
    bool refuse_retirement_owner{false};
    enum class ProgressException { None, Allocation, Other };
    ProgressException progress_exception{ProgressException::None};
    bool exception_before_proof{false};
    int progress_exception_calls{0};
    void install_throwing_boundary_leaf() {
        RunCompletionFence::DeviceEventOps ops;
        ops.create = [](void **event) {
            return aclrtCreateEventExWithFlag(event, ACL_EVENT_SYNC);
        };
        ops.record = [](void *event, void *stream) {
            return aclrtRecordEvent(event, stream);
        };
        ops.query = [this](void *event, bool *complete) {
            if (exception_before_proof) throw_progress_exception();
            aclrtEventRecordedStatus status = ACL_EVENT_RECORDED_STATUS_NOT_READY;
            const int rc = aclrtQueryEventStatus(event, &status);
            *complete = status == ACL_EVENT_RECORDED_STATUS_COMPLETE;
            return rc;
        };
        ops.wait = [this](void *event, int timeout_ms) {
            if (exception_before_proof) throw_progress_exception();
            return aclrtSynchronizeEventWithTimeout(event, timeout_ms);
        };
        ops.destroy = [](void *event) {
            return aclrtDestroyEvent(event);
        };
        submission_storage_[0]->fence = std::make_unique<RunCompletionFence>(std::move(ops));
    }
    void use_inout_argument() { callables_.at(1).signature = {ArgDirection::INOUT}; }
    std::array<void *, 2> boundary_events(const NativeRunIdentity &identity) const {
        return {
            submission_storage_[identity.pipeline_slot]->fence->boundary_event(
                identity, RunCompletionFence::StreamRole::Aicpu
            ),
            submission_storage_[identity.pipeline_slot]->fence->boundary_event(
                identity, RunCompletionFence::StreamRole::Aicore
            )
        };
    }
    int prepare_execution(
        Runtime &runtime, const CallConfig &config, uint32_t slot, const NativeRunIdentity &identity,
        std::unique_ptr<PreparedExecution> *out
    ) override {
        int rc = Runner::prepare_execution(runtime, config, slot, identity, out);
        if (rc != 0) return rc;
        (*out)->resources_owned = true;
        rc = ensure_device_run_result_region(identity, (*out)->kernel_args, (*out)->dfx);
        if (rc != 0) return rc;
        return (*out)->kernel_args.prepare_runtime_args(runtime, mem_alloc_, slot_persistent_args(slot));
    }
    LaunchTransactionResult launch_execution_impl(PreparedExecution &prepared, LaunchPermit permit) override {
        prepared.streams = {&cpu_stream_, &core_stream_};
        const int rc = arm_run_fence(prepared);
        if (rc != 0) return {rc, LaunchProgress::NotStarted, {}};
        auto result = exact_launch_transaction(
            prepared.identity, std::move(permit),
            [&](LaunchProgressSink &sink) {
                const int entry_rc = queue_run_entry(prepared, sink);
                if (entry_rc != 0) return entry_rc;
                sink.mark_submitted();
                return record_run_boundary(prepared, RunCompletionFence::StreamRole::Aicore);
            },
            [&](LaunchProgressSink &sink) {
                sink.mark_submitted();
                return record_run_boundary(prepared, RunCompletionFence::StreamRole::Aicpu);
            }
        );
        DeviceRunResultRegion published{};
        if (publish_success) {
            published.published = prepared.identity.run_epoch;
            published.verdict =
                static_cast<uint32_t>(published_error == 0 ? DeviceRunVerdict::Ok : DeviceRunVerdict::Error);
            published.completion_code = published_error;
            published.code_source =
                static_cast<uint32_t>(published_error == 0 ? DeviceRunCodeSource::None : DeviceRunCodeSource::Header);
        }
        if (prepared.kernel_args.args.run_result_data_base != 0)
            std::memcpy(
                reinterpret_cast<void *>(prepared.kernel_args.args.run_result_data_base), &published, sizeof(published)
            );
        return result;
    }
    int poll_execution(const ActiveExecution &active) override { return poll_run_fence(*active.prepared); }
    int retire_completed_execution(ActiveExecution &active) override {
        auto &prepared = *active.prepared;
        if (!prepared.resources_owned && prepared.physical_complete) return SIMPLER_NATIVE_RUN_POLL_COMPLETE;
        const int completion = poll_run_fence(prepared);
        if (completion != SIMPLER_NATIVE_RUN_POLL_COMPLETE) return completion;
        const int waits = poll_boundary_waits(prepared);
        if (waits != SIMPLER_NATIVE_RUN_POLL_COMPLETE) return waits;
        prepared.physical_complete = true;
        if (!exception_before_proof) throw_progress_exception();
        retire_run_fence(prepared);
        prepared.resources_owned = false;
        // The next allocation belongs to the common workspace observation owner.
        if (std::exchange(refuse_retirement_owner, false)) refuse_next_host_allocation = true;
        return SIMPLER_NATIVE_RUN_POLL_COMPLETE;
    }
    DrainOutcome drain_execution(ActiveExecution &active) override {
        ++drain_calls;
        auto &prepared = *active.prepared;
        const bool already_retired = !prepared.resources_owned && prepared.physical_complete;
        const int rc = wait_run_fence(prepared);
        if (prepared.physical_complete && !already_retired) {
            const int waits = poll_boundary_waits(prepared);
            if (waits != SIMPLER_NATIVE_RUN_POLL_COMPLETE) {
                return DrainOutcome::device_error(PTO_RUNTIME_ERR_INTERNAL, prepared.physical_complete);
            }
            retire_run_fence(prepared);
            prepared.resources_owned = false;
        }
        return DrainOutcome{rc, 0, prepared.physical_complete};
    }

private:
    void throw_progress_exception() {
        const auto failure = std::exchange(progress_exception, ProgressException::None);
        if (failure == ProgressException::None) return;
        ++progress_exception_calls;
        if (failure == ProgressException::Allocation) throw std::bad_alloc();
        throw std::runtime_error("completion retirement failed");
    }
    int cpu_stream_{0};
    int core_stream_{0};
};
}  // namespace

std::unique_ptr<Runner> make_fenced_completion_runner() { return std::make_unique<FencedCompletionRunner>(); }

extern "C" int simpler_execution_environment_releasable_ctx(DeviceContextHandle);

namespace {
std::vector<int> program_lane_wait_results;
std::vector<int> program_lane_finalize_results;
int program_lane_wait(void *context, void *storage) {
    const int rc = simpler_wait_run(context, storage);
    program_lane_wait_results.push_back(rc);
    return rc;
}
int program_lane_finalize(void *context, void *storage, int deliver_outputs) {
    const int rc = simpler_finalize_run_with_delivery(context, storage, deliver_outputs);
    program_lane_finalize_results.push_back(rc);
    return rc;
}
void bind_program_lane(ChipWorker &worker, FencedCompletionRunner &runner) {
    worker.initialized_ = true;
    worker.device_ctx_ = &runner;
    worker.pipeline_contract_ = {PTO_PIPELINE_CONTRACT_ABI_VERSION, 0, 2, {}};
    worker.pipeline_generations_.resize(2);
    for (int slot = 0; slot != 2; ++slot)
        worker.runtime_bufs_.emplace_back(get_runtime_size(), get_runtime_alignment(), nullptr);
    worker.prepare_run_fn_ = simpler_prepare_run;
    worker.launch_run_fn_ = simpler_launch_run;
    worker.poll_run_fn_ = simpler_poll_run;
    worker.wait_run_fn_ = program_lane_wait;
    worker.wait_run_completion_fn_ = simpler_wait_run_completion;
    worker.finalize_run_fn_ = program_lane_finalize;
    worker.supports_concurrent_native_prepare_fn_ = supports_concurrent_native_prepare_ctx;
    worker.supports_joined_native_launch_fn_ = supports_joined_native_launch_ctx;
    worker.supports_joined_diagnostic_launch_fn_ = supports_joined_diagnostic_launch_ctx;
    program_lane_wait_results.clear();
    program_lane_finalize_results.clear();
}
}  // namespace

TEST_F(NativeAdmissionTest, ReusedSlotReleasesCallerBorrowOnlyAfterItsOwnBoundaryProof) {
    for (bool own_complete : {false, true}) {
        SCOPED_TRACE(own_complete);
        // Unknown owners remain live until this process ends; no cleanup or
        // provider reset is used as a substitute for their boundary proof.
        EXPECT_EXIT(
            ([&] {
                auto require = [](bool condition, int code) {
                    if (!condition) {
                        std::fprintf(stderr, "RESIDENT_BORROW failure=%d\n", code);
                        std::_Exit(code);
                    }
                };
                FencedCompletionRunner runner;
                runner.use_inout_argument();
                ChipWorker worker;
                bind_program_lane(worker, runner);
                worker.wait_run_completion_fn_ = [](void *context, void *storage) {
                    const int rc = simpler_wait_run_completion(context, storage);
                    const auto *state = static_cast<const NativeRunContext<DeviceRunnerBase> *>(storage);
                    std::fprintf(
                        stderr, "RESIDENT_BORROW completion_rc=%d physical=%d retired=%d\n", rc,
                        state->physical_complete ? 1 : 0, state->execution_retired ? 1 : 0
                    );
                    return rc;
                };
                worker.device_malloc_ctx_fn_ = device_malloc_ctx;
                worker.device_free_caller_buffer_ctx_fn_ = device_free_caller_buffer_ctx;
                worker.device_borrow_caller_buffers_ctx_fn_ = device_borrow_caller_buffers_ctx;
                worker.device_release_caller_buffers_ctx_fn_ = device_release_caller_buffers_ctx;
                ChipRunLane lane(worker);
                const uint64_t buffer = worker.malloc(256);
                require(buffer != 0 && runner.caller_buffer_count() == 1, 1);
                ChipTensor tensor;
                const uint32_t shape[] = {256};
                tensor.init_external(
                    reinterpret_cast<void *>(buffer), 256, shape, 1, DataType::UINT8, AddressSpace::DEVICE
                );
                ChipStorageTaskArgs args;
                args.add_tensor(tensor);
                CallConfig config{};
                config.aicpu_thread_num = 2;
                for (int ring = 0; ring != RUNTIME_ENV_RING_COUNT; ++ring) {
                    config.runtime_env.ring_task_window[ring] = 4;
                    config.runtime_env.ring_heap[ring] = 1024;
                    config.runtime_env.ring_dep_pool[ring] = 4;
                }
                auto actual_state = [&]() {
                    require(worker.native_runs_.size() == 1, 2);
                    return static_cast<NativeRunContext<DeviceRunnerBase> *>(
                        worker.native_runs_.begin()->second->storage.data()
                    );
                };
                auto free_refused = [&]() {
                    try {
                        worker.free(buffer);
                    } catch (const std::runtime_error &error) {
                        return std::string(error.what()).find("still in use by a submitted run") != std::string::npos;
                    }
                    return false;
                };

                ChipRun healthy = lane.submit(1, args, config, PipelineSlotLease{0, 0, 1}, 101, 201);
                require(healthy.launched(), 3);
                const NativeRunIdentity first_identity = actual_state()->active_execution->prepared->identity;
                const auto first_events = runner.boundary_events(first_identity);
                require(first_events[0] != nullptr && first_events[1] != nullptr, 4);
                require(runner.caller_buffer_borrow_count() == 1, 5);
                require(healthy.done(), 6);
                require(
                    actual_state()->physical_complete && actual_state()->execution_retired &&
                        !actual_state()->result_observed,
                    44
                );
                require(
                    control().observed_streams.empty() && program_lane_wait_results.empty() &&
                        program_lane_finalize_results.empty(),
                    45
                );
                require(runner.caller_buffer_borrow_count() == 0 && runner.caller_buffer_retained_count() == 0, 7);
                require(healthy.wait_until(ChipRun::Deadline::max()), 8);
                require(worker.native_runs_.empty() && runner.reservations() == 0, 9);
                // A's completed proof remains in the ledger when its slot is
                // reused, while the caller allocation remains alive for B.
                require(runner.observed_run_boundaries(first_identity) == RunCompletionFence::Completion::Complete, 10);
                require(runner.caller_buffer_count() == 1, 11);

                control().events_complete = false;
                runner.published_error = -100;
                ChipRun failed = lane.submit(1, args, config, PipelineSlotLease{0, 0, 2}, 102, 202);
                require(failed.launched(), 12);
                auto *const failed_state = actual_state();
                const NativeRunIdentity second_identity = failed_state->active_execution->prepared->identity;
                require(second_identity.pipeline_slot == first_identity.pipeline_slot, 13);
                require(second_identity.generation != first_identity.generation, 14);
                require(second_identity.run_epoch != first_identity.run_epoch, 15);
                require(second_identity.dispatch_id != first_identity.dispatch_id, 16);
                require(runner.boundary_events(second_identity) == first_events, 17);
                require(runner.observed_run_boundaries(first_identity) == RunCompletionFence::Completion::Complete, 42);
                require(runner.observed_run_boundaries(second_identity) == RunCompletionFence::Completion::Pending, 43);
                require(runner.caller_buffer_borrow_count() == 1, 18);
                const size_t observer_cursor = control().observed_streams.size();
                require(!failed.done(), 19);
                require(!failed_state->physical_complete && !failed_state->execution_retired, 20);
                require(runner.observed_run_boundaries(second_identity) == RunCompletionFence::Completion::Pending, 21);
                require(free_refused(), 22);

                control().events_complete = own_complete;
                require(failed.done() == own_complete, 23);
                require(
                    !failed_state->result_observed && control().observed_streams.size() == observer_cursor &&
                        program_lane_wait_results == std::vector<int>{0} &&
                        program_lane_finalize_results == std::vector<int>{0},
                    46
                );
                require(failed_state->physical_complete == own_complete, 24);
                require(failed_state->execution_retired == own_complete, 25);
                require(runner.caller_buffer_borrow_count() == (own_complete ? 0u : 1u), 26);
                control().stream_sync_rc = 507018;
                control().event_wait_rc = own_complete ? 0 : 507018;
                try {
                    (void)failed.wait_until(ChipRun::Deadline::max());
                    std::_Exit(27);
                } catch (const std::runtime_error &error) {
                    require(std::string(error.what()).find("507018") != std::string::npos, 28);
                }
                require(control().observed_streams.size() == observer_cursor + 1, 29);
                require(program_lane_wait_results == std::vector<int>({0, 507018}), 30);
                require(program_lane_finalize_results == std::vector<int>({0, 507018}), 31);
                std::fprintf(
                    stderr,
                    "RESIDENT_BORROW observed own_complete=%d owners=%zu reservations=%zu borrow=%zu retained=%zu\n",
                    own_complete ? 1 : 0, worker.native_runs_.size(), runner.reservations(),
                    runner.caller_buffer_borrow_count(), runner.caller_buffer_retained_count()
                );
                require(worker.native_runs_.size() == (own_complete ? 0u : 1u), 32);
                require(runner.reservations() == (own_complete ? 0u : 1u), 33);
                require(runner.caller_buffer_borrow_count() == 0, 34);
                require(runner.caller_buffer_retained_count() == (own_complete ? 0u : 1u), 35);
                if (own_complete) {
                    worker.free(buffer);
                    require(runner.caller_buffer_count() == 0, 36);
                } else {
                    require(!failed_state->physical_complete && !failed_state->execution_retired, 37);
                    require(free_refused() && runner.caller_buffer_count() == 1, 38);
                }
                const size_t observations = control().observed_streams.size();
                control().stream_sync_rc = 0;
                try {
                    (void)failed.wait_until(ChipRun::Deadline::max());
                    std::_Exit(39);
                } catch (const std::runtime_error &error) {
                    require(std::string(error.what()).find("507018") != std::string::npos, 40);
                }
                require(control().observed_streams.size() == observations, 41);
                std::fprintf(
                    stdout,
                    "RESIDENT_BORROW own_complete=%d first_epoch=%llu second_epoch=%llu generations=[%llu,%llu] "
                    "same_events=1 sdk=507018 owners=%zu reservations=%zu retained=%zu free=%s\n",
                    own_complete ? 1 : 0, static_cast<unsigned long long>(first_identity.run_epoch),
                    static_cast<unsigned long long>(second_identity.run_epoch),
                    static_cast<unsigned long long>(first_identity.generation),
                    static_cast<unsigned long long>(second_identity.generation), worker.native_runs_.size(),
                    runner.reservations(), runner.caller_buffer_retained_count(), own_complete ? "released" : "refused"
                );
                std::fflush(stdout);
                std::_Exit(0);
            }()),
            ::testing::ExitedWithCode(0), ""
        );
    }
}

TEST_F(NativeAdmissionTest, KnownLaneFailureDoesNotPublishUnobservedProgramOutput) {
    struct Case {
        const char *name;
        int first_error;
        int second_error;
        bool confirm_first;
    };
    for (const auto &test :
         {Case{"healthy", 0, 0, false}, Case{"confirmed-before-fault", 0, -735, true},
          Case{"distinct-own-error", 507018, -735, false}, Case{"known-poison-native-success", 507018, 0, false}}) {
        SCOPED_TRACE(test.name);
        ASSERT_EQ(control().bytes, 0u);
        control() = {};
        FencedCompletionRunner runner;
        runner.use_inout_argument();
        ChipWorker worker;
        bind_program_lane(worker, runner);
        {
            ChipRunLane lane(worker);
            std::vector<uint8_t> first_host(16384 * sizeof(float), 0x71);
            std::vector<uint8_t> second_host(first_host.size(), 0x71);
            std::array<void *, 2> device_outputs{};
            CallConfig config{};
            config.aicpu_thread_num = 2;
            for (int ring = 0; ring != RUNTIME_ENV_RING_COUNT; ++ring) {
                config.runtime_env.ring_task_window[ring] = 4;
                config.runtime_env.ring_heap[ring] = 1024;
                config.runtime_env.ring_dep_pool[ring] = 4;
            }
            auto submit = [&](std::vector<uint8_t> &host, uint32_t slot, uint64_t run_id, uint8_t result) {
                ChipTensor tensor;
                const uint32_t shape[] = {static_cast<uint32_t>(host.size())};
                tensor.init_external(host.data(), host.size(), shape, 1, DataType::UINT8, AddressSpace::HOST);
                ChipStorageTaskArgs args;
                args.add_tensor(tensor);
                ChipRun run = lane.submit(1, args, config, PipelineSlotLease{slot, 0, 1}, run_id, run_id);
                const auto upload =
                    std::find_if(control().uploads.begin(), control().uploads.end(), [&](const auto &row) {
                        return row.second == host.data();
                    });
                EXPECT_NE(upload, control().uploads.end());
                if (upload != control().uploads.end()) {
                    device_outputs[slot] = upload->first;
                    std::memset(upload->first, result, host.size());
                }
                EXPECT_TRUE(run.done());
                EXPECT_EQ(host, std::vector<uint8_t>(host.size(), 0x71));
                return run;
            };
            auto observe = [](ChipRun &run) {
                try {
                    EXPECT_TRUE(run.wait_until(ChipRun::Deadline::max()));
                    return std::string{};
                } catch (const std::runtime_error &error) {
                    return std::string(error.what());
                }
            };
            ChipRun first = submit(first_host, 0, 1, 0x42);
            size_t first_downloads = 0;
            if (test.confirm_first) {
                const size_t cursor = control().downloads.size();
                EXPECT_TRUE(observe(first).empty());
                first_downloads =
                    std::count(control().downloads.begin() + cursor, control().downloads.end(), device_outputs[0]);
            }
            ChipRun second = submit(second_host, 1, 2, 0x53);
            if (!test.confirm_first) {
                EXPECT_TRUE(control().observed_streams.empty());
                EXPECT_EQ(control().d2h_calls, 0u);
            }
            control().stream_sync_rc = test.first_error;
            const size_t first_cursor = control().downloads.size();
            const std::string first_error = observe(first);
            if (!test.confirm_first)
                first_downloads = std::count(
                    control().downloads.begin() + first_cursor, control().downloads.end(), device_outputs[0]
                );
            control().stream_sync_rc = test.second_error;
            const size_t second_cursor = control().downloads.size();
            const std::string second_error = observe(second);
            const size_t second_downloads =
                std::count(control().downloads.begin() + second_cursor, control().downloads.end(), device_outputs[1]);
            if (test.first_error != 0) EXPECT_NE(first_error.find(std::to_string(test.first_error)), std::string::npos);
            else EXPECT_TRUE(first_error.empty());
            if (test.second_error != 0) {
                EXPECT_NE(second_error.find(std::to_string(test.second_error)), std::string::npos);
                if (test.first_error != 0)
                    EXPECT_NE(second_error.find(std::to_string(test.first_error)), std::string::npos);
            } else if (test.first_error != 0) {
                EXPECT_NE(second_error.find("507018"), std::string::npos);
                EXPECT_NE(second_error.find("run_id=1 slot=0 generation=1 dispatch_id=1"), std::string::npos);
            } else EXPECT_TRUE(second_error.empty());
            EXPECT_EQ(program_lane_wait_results, (std::vector<int>{test.first_error, test.second_error}));
            EXPECT_EQ(program_lane_finalize_results, (std::vector<int>{test.first_error, test.second_error}));
            EXPECT_TRUE(worker.native_runs_.empty());
            EXPECT_EQ(runner.reservations(), 0u);
            const bool first_delivered = test.first_error == 0;
            const bool second_delivered = test.first_error == 0 && test.second_error == 0;
            EXPECT_EQ(first_host, std::vector<uint8_t>(first_host.size(), first_delivered ? 0x42 : 0x71));
            EXPECT_EQ(second_host, std::vector<uint8_t>(second_host.size(), second_delivered ? 0x53 : 0x71));
            EXPECT_EQ(first_downloads, first_delivered ? 1u : 0u);
            EXPECT_EQ(second_downloads, second_delivered ? 1u : 0u);
            std::fprintf(
                stderr,
                "PROGRAM_LANE_COPYBACK case=%s native=[%d,%d] finalize=[%d,%d] output_reads=[%zu,%zu] first=%u "
                "second=%u owners=%zu\n",
                test.name, program_lane_wait_results.at(0), program_lane_wait_results.at(1),
                program_lane_finalize_results.at(0), program_lane_finalize_results.at(1), first_downloads,
                second_downloads, first_host.front(), second_host.front(), worker.native_runs_.size()
            );
            if (test.confirm_first) {
                const size_t observations = control().observed_streams.size();
                EXPECT_TRUE(observe(first).empty());
                EXPECT_EQ(control().observed_streams.size(), observations);
                EXPECT_EQ(first_host, std::vector<uint8_t>(first_host.size(), 0x42));
            }
            if (test.first_error || test.second_error) EXPECT_THROW(lane.close(), std::runtime_error);
            else EXPECT_NO_THROW(lane.close());
        }
        worker.device_ctx_ = nullptr;
    }
}

TEST_F(NativeAdmissionTest, OptionalResultKernelHelperKeepsIdentityWithoutPrivateSideEffects) {
    class ResultRunner : public FencedCompletionRunner {
    public:
        int prepare_result(const NativeRunIdentity &identity, KernelArgsHelper &args) {
            return ensure_device_run_result_region(identity, args, {});
        }
        size_t result_epochs() const {
            std::lock_guard<std::mutex> lock(run_result_mu_);
            return run_result_storage_.size();
        }
    } runner;
    ASSERT_EQ(initialize_capture_runner(runner), 0);
    KernelArgsHelper args;
    const size_t allocations = control().allocations;
    const size_t uploads = control().uploads.size();
    ASSERT_EQ(runner.prepare_result(NativeRunIdentity{94001, 1, 1, 0}, args), 0);
    EXPECT_EQ(args.args.run_result_data_base, 0u);
    EXPECT_EQ(args.args.run_result_epoch, 94001u);
    EXPECT_EQ(runner.result_epochs(), 0u);
    EXPECT_EQ(control().allocations, allocations);
    EXPECT_EQ(control().uploads.size(), uploads);
    EXPECT_NE(runner.prepare_result(NativeRunIdentity{0, 1, 1, 0}, args), 0);
    EXPECT_NE(runner.prepare_result(NativeRunIdentity{94002, 1, 1, 99}, args), 0);
}

TEST_F(NativeAdmissionTest, OptionalResultNativeKernelPreparationsDoNotBindAPrivateRecord) {
    for (const auto purpose : {SIMPLER_PREPARE_EAGER, SIMPLER_PREPARE_CAPTURE}) {
        SCOPED_TRACE(purpose);
        FencedCompletionRunner runner;
        ASSERT_EQ(initialize_capture_runner(runner), 0);
        Attempt attempt(runner, 0, 94003 + static_cast<uint64_t>(purpose));
        use_capture_preparation(attempt);
        attempt.descriptor.prepare_purpose = purpose;
        ASSERT_EQ(attempt.prepare(), 0);
        const auto *state = static_cast<const NativeRunContext<DeviceRunnerBase> *>(attempt.storage);
        EXPECT_EQ(state->prepared_execution->kernel_args.args.run_result_data_base, 0u);
        EXPECT_EQ(state->prepared_execution->kernel_args.args.run_result_epoch, attempt.descriptor.run_epoch);
        EXPECT_FALSE(runner.result_storage().retain_replay_evidence);
        const size_t downloads = control().d2h_calls;
        ASSERT_EQ(simpler_finalize_run(&runner, attempt.storage), 0);
        EXPECT_EQ(control().d2h_calls, downloads);
    }
}

TEST_F(NativeAdmissionTest, OptionalResultProgramAllocationFailureKeepsItsOriginalErrorAndCleanup) {
    class FailedResultRunner : public FencedCompletionRunner {
    public:
        int prepare_execution(
            Runtime &runtime, const CallConfig &config, uint32_t slot, const NativeRunIdentity &identity,
            std::unique_ptr<PreparedExecution> *out
        ) override {
            control().fail_allocation = control().allocations + 1;
            return FencedCompletionRunner::prepare_execution(runtime, config, slot, identity, out);
        }
    } runner;
    Attempt attempt(runner, 0, 94007);
    EXPECT_EQ(attempt.prepare(), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_TRUE(attempt.consume_release());
    EXPECT_EQ(runner.reservations(), 0u);
    EXPECT_EQ(runner.native_run_claim_count(), 0u);
    EXPECT_EQ(control().d2h_calls, 0u);
}

TEST_F(NativeAdmissionTest, OptionalResultKernelObserverErrorRemainsTheOriginalSignedCode) {
    EXPECT_EXIT(
        {
            class CallerRunner : public FencedCompletionRunner {
            public:
                LaunchTransactionResult
                launch_execution_impl(PreparedExecution &prepared, LaunchPermit permit) override {
                    auto result = FencedCompletionRunner::launch_execution_impl(prepared, std::move(permit));
                    if (result.rc == 0) result.rc = queue_caller_exit(prepared);
                    return result;
                }
            } runner;
            if (initialize_capture_runner(runner) != 0) std::_Exit(1);
            Attempt attempt(runner, 0, 94008);
            use_capture_preparation(attempt);
            attempt.descriptor.prepare_purpose = SIMPLER_PREPARE_EAGER;
            if (attempt.prepare() != 0) std::_Exit(2);
            auto *state = static_cast<NativeRunContext<DeviceRunnerBase> *>(attempt.storage);
            if (state->prepared_execution->kernel_args.args.run_result_data_base != 0 ||
                state->prepared_execution->kernel_args.args.run_result_epoch != attempt.descriptor.run_epoch)
                std::_Exit(3);
            int caller;
            if (simpler_launch_run_on_stream(&runner, attempt.storage, &caller, nullptr) != 0) std::_Exit(4);
            if (simpler_poll_run(&runner, attempt.storage) != SIMPLER_NATIVE_RUN_POLL_COMPLETE) std::_Exit(5);
            control().stream_sync_rc = -946;
            if (simpler_wait_run(&runner, attempt.storage) != -946) std::_Exit(6);
            control().stream_sync_rc = 0;
            if (simpler_wait_run(&runner, attempt.storage) != -946) std::_Exit(7);
            if (simpler_finalize_run(&runner, attempt.storage) != -946) std::_Exit(8);
            if (control().d2h_calls != 0) std::_Exit(9);
            std::_Exit(0);
        },
        ::testing::ExitedWithCode(0), ""
    );
}

TEST_F(NativeAdmissionTest, SuccessfulExplicitObservationDoesNotReadThePrivateResult) {
    FencedCompletionRunner runner;
    Attempt active(runner, 0, 10);
    ASSERT_EQ(active.prepare(), 0);
    ASSERT_EQ(simpler_launch_run(&runner, active.storage), 0);
    for (int poll = 0; poll != 3; ++poll) {
        EXPECT_EQ(simpler_poll_run(&runner, active.storage), SIMPLER_NATIVE_RUN_POLL_COMPLETE);
        EXPECT_EQ(control().d2h_calls, 0u);
        EXPECT_TRUE(control().observed_streams.empty());
    }
    EXPECT_EQ(simpler_wait_run(&runner, active.storage), 0);
    EXPECT_EQ(control().observed_streams.size(), 2u);
    EXPECT_EQ(control().d2h_calls, 0u);
    EXPECT_EQ(simpler_wait_run(&runner, active.storage), 0);
    EXPECT_EQ(control().observed_streams.size(), 2u);
    EXPECT_EQ(simpler_finalize_run(&runner, active.storage), 0);
    EXPECT_EQ(control().observed_streams.size(), 2u);
    EXPECT_EQ(control().d2h_calls, 0u);
    EXPECT_TRUE(active.empty());
}

TEST_F(NativeAdmissionTest, BlockingPhysicalProgressDoesNotPerformTheStrongObserver) {
    FencedCompletionRunner runner;
    Attempt first(runner, 0, 10);
    ASSERT_EQ(first.prepare(), 0);
    ASSERT_EQ(simpler_launch_run(&runner, first.storage), 0);
    ASSERT_EQ(simpler_wait_run_completion(&runner, first.storage), 0);
    EXPECT_EQ(simpler_poll_run(&runner, first.storage), SIMPLER_NATIVE_RUN_POLL_COMPLETE);
    EXPECT_EQ(control().d2h_calls, 0u);
    EXPECT_TRUE(control().observed_streams.empty());
    Attempt successor(runner, 0, 20);
    ASSERT_EQ(successor.prepare(), 0);
    ASSERT_EQ(simpler_launch_run(&runner, successor.storage), 0);
    ASSERT_EQ(simpler_wait_run_completion(&runner, successor.storage), 0);
    EXPECT_EQ(control().d2h_calls, 0u);
    EXPECT_TRUE(control().observed_streams.empty());
    EXPECT_EQ(simpler_finalize_run(&runner, first.storage), 0);
    EXPECT_EQ(control().observed_streams.size(), 2u);
    EXPECT_EQ(simpler_finalize_run(&runner, successor.storage), 0);
    EXPECT_EQ(control().observed_streams.size(), 4u);
    EXPECT_EQ(control().d2h_calls, 0u);
}

TEST_F(NativeAdmissionTest, SuccessfulObservationDoesNotRequireAPrivateTerminalPublication) {
    FencedCompletionRunner runner;
    runner.publish_success = false;
    Attempt active(runner, 0, 10);
    ASSERT_EQ(active.prepare(), 0);
    ASSERT_EQ(simpler_launch_run(&runner, active.storage), 0);
    EXPECT_EQ(simpler_wait_run(&runner, active.storage), 0);
    EXPECT_EQ(control().observed_streams.size(), 2u);
    EXPECT_EQ(simpler_finalize_run(&runner, active.storage), 0);
    EXPECT_EQ(control().d2h_calls, 0u);
    EXPECT_TRUE(active.empty());
}

TEST_F(NativeAdmissionTest, ProgramOutputStaysPrivateUntilObservationAndCopyback) {
    FencedCompletionRunner runner;
    runner.use_inout_argument();
    Attempt active(runner, 0, 10);
    ASSERT_EQ(active.prepare(), 0);
    auto upload = std::find_if(control().uploads.begin(), control().uploads.end(), [&](const auto &item) {
        return item.second == active.bytes.data();
    });
    ASSERT_NE(upload, control().uploads.end());
    void *const output_backing = upload->first;
    ASSERT_NE(output_backing, active.bytes.data());
    ASSERT_EQ(simpler_launch_run(&runner, active.storage), 0);
    std::memset(output_backing, 0x42, active.bytes.size());
    ASSERT_EQ(simpler_poll_run(&runner, active.storage), SIMPLER_NATIVE_RUN_POLL_COMPLETE);
    EXPECT_TRUE(control().observed_streams.empty());
    EXPECT_EQ(control().d2h_calls, 0u);
    EXPECT_EQ(active.bytes, std::vector<uint8_t>(active.bytes.size(), 0x71));
    EXPECT_EQ(simpler_wait_run(&runner, active.storage), 0);
    EXPECT_EQ(control().observed_streams.size(), 2u);
    EXPECT_EQ(active.bytes, std::vector<uint8_t>(active.bytes.size(), 0x71));
    EXPECT_EQ(simpler_finalize_run(&runner, active.storage), 0);
    EXPECT_EQ(active.bytes, std::vector<uint8_t>(active.bytes.size(), 0x42));
    EXPECT_EQ(control().d2h_calls, 1u);
    EXPECT_EQ(control().d2h_bytes, active.bytes.size());
}

TEST_F(NativeAdmissionTest, PhysicalProgressExceptionsKeepTheSameOwnerRetryable) {
    for (bool blocking : {false, true}) {
        for (bool before_proof : {false, true}) {
            for (bool allocation : {false, true}) {
                SCOPED_TRACE(blocking);
                SCOPED_TRACE(before_proof);
                SCOPED_TRACE(allocation);
                ASSERT_EQ(control().bytes, 0u);
                control() = {};
                FencedCompletionRunner runner;
                runner.use_inout_argument();
                runner.install_throwing_boundary_leaf();
                Attempt active(runner, 0, 10);
                ASSERT_EQ(active.prepare(), 0);
                ASSERT_EQ(simpler_launch_run(&runner, active.storage), 0);
                auto *const state = static_cast<NativeRunContext<DeviceRunnerBase> *>(active.storage);
                auto *const runtime_owner = state->runtime.get();
                auto *const execution_owner = state->active_execution.get();
                const auto charged = runner.charged();
                runner.exception_before_proof = before_proof;
                runner.progress_exception = allocation ? FencedCompletionRunner::ProgressException::Allocation :
                                                         FencedCompletionRunner::ProgressException::Other;
                bool escaped = false;
                int rc = 12345;
                try {
                    rc = blocking ? simpler_wait_run_completion(&runner, active.storage) :
                                    simpler_poll_run(&runner, active.storage);
                } catch (...) {
                    escaped = true;
                }
                EXPECT_FALSE(escaped);
                const int expected_rc = blocking ? static_cast<int>(PTO_RUNTIME_ERR_INTERNAL) :
                                                   static_cast<int>(SIMPLER_NATIVE_RUN_POLL_ERROR);
                EXPECT_EQ(rc, expected_rc);
                EXPECT_EQ(runner.progress_exception_calls, 1);
                EXPECT_FALSE(active.empty());
                EXPECT_EQ(state->runtime.get(), runtime_owner);
                EXPECT_EQ(state->active_execution.get(), execution_owner);
                EXPECT_EQ(state->active_execution->prepared->physical_complete, !before_proof);
                EXPECT_FALSE(state->physical_complete);
                EXPECT_FALSE(state->execution_retired);
                EXPECT_FALSE(state->result_observed);
                EXPECT_EQ(state->phase.load(), NativeRunPhase::Running);
                EXPECT_EQ(state->completion_rc, 0);
                EXPECT_TRUE(runner.owns_native_run_observation(active.storage));
                EXPECT_TRUE(runner.holds_epoch(10));
                EXPECT_EQ(runner.reservations(), 1u);
                EXPECT_EQ(runner.charged(), charged);
                EXPECT_EQ(control().d2h_calls, 0u);
                EXPECT_TRUE(control().observed_streams.empty());
                EXPECT_EQ(active.bytes, std::vector<uint8_t>(active.bytes.size(), 0x71));
                std::fprintf(
                    stderr, "PROGRESS_EXCEPTION entry=%s before_proof=%d allocation=%d escaped=%d rc=%d physical=%d\n",
                    blocking ? "wait_completion" : "poll", before_proof, allocation, escaped, rc,
                    state->active_execution->prepared->physical_complete
                );
                if (blocking) ASSERT_EQ(simpler_wait_run_completion(&runner, active.storage), 0);
                else ASSERT_EQ(simpler_poll_run(&runner, active.storage), SIMPLER_NATIVE_RUN_POLL_COMPLETE);
                EXPECT_EQ(runner.reservations(), 0u);
                EXPECT_TRUE(control().observed_streams.empty());
                ASSERT_EQ(simpler_wait_run(&runner, active.storage), 0);
                ASSERT_EQ(simpler_finalize_run(&runner, active.storage), 0);
                EXPECT_TRUE(active.empty());
                EXPECT_EQ(control().d2h_calls, 1u);
            }
        }
    }
}

TEST_F(NativeAdmissionTest, WorkspaceRetirementAllocationFailureRemainsRetryable) {
    for (bool blocking : {false, true}) {
        SCOPED_TRACE(blocking ? "wait_completion" : "poll");
        ASSERT_EQ(control().bytes, 0u);
        control() = {};
        FencedCompletionRunner runner;
        runner.use_inout_argument();
        Attempt active(runner, 0, 10);
        ASSERT_EQ(active.prepare(), 0);
        const auto upload = std::find_if(control().uploads.begin(), control().uploads.end(), [&](const auto &item) {
            return item.second == active.bytes.data();
        });
        ASSERT_NE(upload, control().uploads.end());
        void *const output_backing = upload->first;
        ASSERT_EQ(simpler_launch_run(&runner, active.storage), 0);
        std::memset(output_backing, 0x42, active.bytes.size());
        const auto charged = runner.charged();
        const auto staging = runner.retained_temp_addr(0);
        std::array<bool, 3> arena_refs{};
        for (size_t region = 0; region != arena_refs.size(); ++region)
            arena_refs[region] = runner.arena_referenced(arena_bank(0), region);
        EXPECT_TRUE(std::any_of(arena_refs.begin(), arena_refs.end(), [](bool held) {
            return held;
        }));
        runner.refuse_retirement_owner = true;
        bool escaped = false;
        int rc = 12345;
        try {
            rc = blocking ? simpler_wait_run_completion(&runner, active.storage) :
                            simpler_poll_run(&runner, active.storage);
        } catch (const std::bad_alloc &) {
            escaped = true;
        }
        const bool allocation_attempted = !std::exchange(refuse_next_host_allocation, false);
        EXPECT_TRUE(allocation_attempted);
        EXPECT_FALSE(runner.refuse_retirement_owner);
        EXPECT_FALSE(escaped);
        const int expected_rc =
            blocking ? static_cast<int>(PTO_RUNTIME_ERR_INTERNAL) : static_cast<int>(SIMPLER_NATIVE_RUN_POLL_ERROR);
        EXPECT_EQ(rc, expected_rc);
        EXPECT_FALSE(active.empty());
        EXPECT_TRUE(runner.owns_native_run_observation(active.storage));
        EXPECT_TRUE(runner.holds_epoch(10));
        EXPECT_EQ(runner.reservations(), 1u);
        EXPECT_EQ(runner.charged(), charged);
        EXPECT_EQ(runner.retained_temp_addr(0), staging);
        for (size_t region = 0; region != arena_refs.size(); ++region)
            EXPECT_EQ(runner.arena_referenced(arena_bank(0), region), arena_refs[region]);
        EXPECT_TRUE(control().observed_streams.empty());
        EXPECT_EQ(control().d2h_calls, 0u);
        EXPECT_EQ(active.bytes, std::vector<uint8_t>(active.bytes.size(), 0x71));
        std::fprintf(
            stderr, "RETIRE_OWNER entry=%s allocation_consumed=%d escaped=%d rc=%d reservation=%zu\n",
            blocking ? "wait_completion" : "poll", allocation_attempted, escaped, rc, runner.reservations()
        );
        if (blocking) ASSERT_EQ(simpler_wait_run_completion(&runner, active.storage), 0);
        else ASSERT_EQ(simpler_poll_run(&runner, active.storage), SIMPLER_NATIVE_RUN_POLL_COMPLETE);
        EXPECT_EQ(runner.reservations(), 0u);
        EXPECT_TRUE(control().observed_streams.empty());
        EXPECT_EQ(simpler_wait_run(&runner, active.storage), 0);
        EXPECT_EQ(control().observed_streams.size(), 2u);
        EXPECT_EQ(control().d2h_calls, 0u);
        EXPECT_EQ(simpler_finalize_run(&runner, active.storage), 0);
        EXPECT_TRUE(active.empty());
        EXPECT_FALSE(runner.native_runs_outstanding());
        EXPECT_EQ(active.bytes, std::vector<uint8_t>(active.bytes.size(), 0x42));
        EXPECT_EQ(control().d2h_calls, 1u);
        EXPECT_EQ(control().d2h_bytes, active.bytes.size());
    }
}

TEST_F(NativeAdmissionTest, PreparedBackingSurvivesSlotReuseUntilItsObservationEnds) {
    FencedCompletionRunner runner;
    runner.use_inout_argument();
    Attempt first(runner, 0, 10);
    ASSERT_EQ(first.prepare(), 0);
    auto *state = static_cast<NativeRunContext<DeviceRunnerBase> *>(first.storage);
    const std::weak_ptr<const PreparedRunResources> original = state->prepared_execution->prepared_resources;
    ASSERT_EQ(simpler_launch_run(&runner, first.storage), 0);
    ASSERT_EQ(simpler_wait_run_completion(&runner, first.storage), 0);
    EXPECT_FALSE(original.expired());
    EXPECT_FALSE(state->result_observed);
    EXPECT_EQ(runner.reservations(), 0u);
    Attempt successor(runner, 0, 20);
    ASSERT_EQ(successor.prepare(), 0);
    auto *next = static_cast<NativeRunContext<DeviceRunnerBase> *>(successor.storage);
    EXPECT_NE(original.lock(), next->prepared_execution->prepared_resources);
    ASSERT_EQ(simpler_launch_run(&runner, successor.storage), 0);
    EXPECT_FALSE(original.expired());
    EXPECT_EQ(control().d2h_calls, 0u);
    EXPECT_TRUE(control().observed_streams.empty());
    ASSERT_EQ(simpler_finalize_run_with_delivery(&runner, first.storage, 0), 0);
    EXPECT_TRUE(original.expired());
    EXPECT_EQ(first.bytes, std::vector<uint8_t>(first.bytes.size(), 0x71));
    EXPECT_EQ(control().d2h_calls, 0u);
    EXPECT_EQ(simpler_poll_run(&runner, successor.storage), SIMPLER_NATIVE_RUN_POLL_COMPLETE);
    EXPECT_EQ(simpler_finalize_run(&runner, successor.storage), 0);
}

TEST_F(NativeAdmissionTest, ReusedExecutionSlotsPreserveSixUnobservedProgramOutputs) {
    FencedCompletionRunner runner;
    ASSERT_TRUE(runner.set_run_resource_grant(2));
    runner.use_inout_argument();
    std::vector<std::unique_ptr<Attempt>> calls;
    std::vector<void *> output_backings;
    for (uint64_t invocation = 0; invocation != 6; ++invocation) {
        auto call = std::make_unique<Attempt>(runner, invocation % 2, invocation + 10);
        ASSERT_EQ(call->prepare(), 0) << "an unobserved output is not an execution-slot occupant";
        const auto upload = std::find_if(control().uploads.begin(), control().uploads.end(), [&](const auto &item) {
            return item.second == call->bytes.data();
        });
        ASSERT_NE(upload, control().uploads.end());
        void *const backing = upload->first;
        EXPECT_EQ(std::find(output_backings.begin(), output_backings.end(), backing), output_backings.end());
        output_backings.push_back(backing);
        ASSERT_EQ(simpler_launch_run(&runner, call->storage), 0);
        std::memset(backing, static_cast<int>(0x40 + invocation), call->bytes.size());
        ASSERT_EQ(simpler_poll_run(&runner, call->storage), SIMPLER_NATIVE_RUN_POLL_COMPLETE);
        EXPECT_EQ(control().d2h_calls, 0u);
        EXPECT_TRUE(control().observed_streams.empty());
        calls.push_back(std::move(call));
        for (const auto &unobserved : calls)
            EXPECT_EQ(unobserved->bytes, std::vector<uint8_t>(unobserved->bytes.size(), 0x71));
    }
    for (size_t remaining = calls.size(); remaining != 0; --remaining) {
        const size_t index = remaining - 1;
        auto &call = *calls[index];
        ASSERT_EQ(simpler_wait_run(&runner, call.storage), 0);
        ASSERT_EQ(simpler_wait_run(&runner, call.storage), 0);
        ASSERT_EQ(simpler_finalize_run(&runner, call.storage), 0);
        EXPECT_TRUE(call.empty());
        EXPECT_EQ(call.bytes, std::vector<uint8_t>(call.bytes.size(), static_cast<uint8_t>(0x40 + index)));
        EXPECT_EQ(control().d2h_calls, calls.size() - index);
        EXPECT_EQ(control().observed_streams.size(), 2 * (calls.size() - index));
    }
}

TEST_F(NativeAdmissionTest, UnmanagedWorkspaceKeepsPendingHostOutputsAcrossSlotReuse) {
    FencedCompletionRunner runner(false);
    ASSERT_FALSE(runner.workspace_enabled());
    runner.use_inout_argument();
    std::vector<std::unique_ptr<Attempt>> calls;
    std::vector<void *> output_backings;
    for (uint64_t index = 0; index != 2; ++index) {
        auto call = std::make_unique<Attempt>(runner, 0, index + 10);
        ASSERT_EQ(call->prepare(), 0);
        const auto upload = std::find_if(control().uploads.begin(), control().uploads.end(), [&](const auto &item) {
            return item.second == call->bytes.data();
        });
        ASSERT_NE(upload, control().uploads.end());
        void *const backing = upload->first;
        EXPECT_EQ(std::find(output_backings.begin(), output_backings.end(), backing), output_backings.end());
        output_backings.push_back(backing);
        ASSERT_EQ(simpler_launch_run(&runner, call->storage), 0);
        std::memset(backing, static_cast<int>(0x50 + index), call->bytes.size());
        ASSERT_EQ(simpler_poll_run(&runner, call->storage), SIMPLER_NATIVE_RUN_POLL_COMPLETE);
        EXPECT_TRUE(control().observed_streams.empty());
        EXPECT_EQ(control().d2h_calls, 0u);
        calls.push_back(std::move(call));
        for (const auto &pending : calls)
            EXPECT_EQ(pending->bytes, std::vector<uint8_t>(pending->bytes.size(), 0x71));
    }
    for (size_t remaining = calls.size(); remaining != 0; --remaining) {
        const size_t index = remaining - 1;
        auto &call = *calls[index];
        ASSERT_EQ(simpler_wait_run(&runner, call.storage), 0);
        EXPECT_EQ(call.bytes, std::vector<uint8_t>(call.bytes.size(), 0x71));
        ASSERT_EQ(simpler_finalize_run(&runner, call.storage), 0);
        EXPECT_TRUE(call.empty());
        EXPECT_EQ(call.bytes, std::vector<uint8_t>(call.bytes.size(), static_cast<uint8_t>(0x50 + index)));
        EXPECT_EQ(control().d2h_calls, calls.size() - index);
        EXPECT_EQ(control().observed_streams.size(), 2 * (calls.size() - index));
    }
}

TEST_F(NativeAdmissionTest, RetiredProgramCallKeepsItsOutputAndIdentityAcrossBindRefusal) {
    FencedCompletionRunner runner(false);
    ASSERT_EQ(runner.execution_mode_latch().latch(SIMPLER_MODE_PROGRAM), 0);
    runner.use_inout_argument();
    Attempt first(runner, 0, 10);
    ASSERT_EQ(first.prepare(), 0);
    const auto first_upload = std::find_if(control().uploads.begin(), control().uploads.end(), [&](const auto &item) {
        return item.second == first.bytes.data();
    });
    ASSERT_NE(first_upload, control().uploads.end());
    void *const first_output = first_upload->first;
    ASSERT_EQ(simpler_launch_run(&runner, first.storage), 0);
    std::memset(first_output, 0x42, first.bytes.size());
    ASSERT_EQ(simpler_poll_run(&runner, first.storage), SIMPLER_NATIVE_RUN_POLL_COMPLETE);

    Attempt second(runner, 0, 11);
    ASSERT_EQ(second.prepare(), 0);
    const auto second_upload = std::find_if(control().uploads.begin(), control().uploads.end(), [&](const auto &item) {
        return item.second == second.bytes.data();
    });
    ASSERT_NE(second_upload, control().uploads.end());
    void *const second_output = second_upload->first;
    ASSERT_NE(second_output, first_output);
    ASSERT_EQ(simpler_launch_run(&runner, second.storage), 0);
    std::memset(second_output, 0x53, second.bytes.size());
    ASSERT_EQ(simpler_poll_run(&runner, second.storage), SIMPLER_NATIVE_RUN_POLL_COMPLETE);
    const size_t owned_bytes = control().bytes;
    ASSERT_TRUE(control().observed_streams.empty());
    ASSERT_EQ(control().d2h_calls, 0u);

    control().attach_rc = -37;
    EXPECT_EQ(simpler_finalize_run(&runner, first.storage), -37);
    EXPECT_FALSE(first.empty()) << "a refused Program binding must retain its canonical carrier";
    EXPECT_EQ(control().bytes, owned_bytes) << "refused binding freed call-owned backing";
    EXPECT_TRUE(control().observed_streams.empty());
    EXPECT_EQ(control().d2h_calls, 0u);
    EXPECT_EQ(first.bytes, std::vector<uint8_t>(first.bytes.size(), 0x71));

    control().attach_rc = 0;
    EXPECT_EQ(simpler_wait_run(&runner, first.storage), 0) << "binding refusal must not cache a call error";
    EXPECT_EQ(simpler_finalize_run(&runner, first.storage), 0);
    EXPECT_EQ(first.bytes, std::vector<uint8_t>(first.bytes.size(), 0x42));
    EXPECT_EQ(simpler_wait_run(&runner, second.storage), 0);
    EXPECT_EQ(simpler_finalize_run(&runner, second.storage), 0);
    EXPECT_EQ(second.bytes, std::vector<uint8_t>(second.bytes.size(), 0x53));
}

TEST_F(NativeAdmissionTest, LateErrorsReadEachCallsRecordAfterTheSharedArenaIsReused) {
    FencedCompletionRunner runner;
    std::vector<std::unique_ptr<Attempt>> calls;
    std::vector<const void *> records;
    for (int index = 0; index != 2; ++index) {
        auto call = std::make_unique<Attempt>(runner, 0, index + 10);
        runner.published_error = -101 - index;
        ASSERT_EQ(call->prepare(), 0);
        ASSERT_EQ(simpler_launch_run(&runner, call->storage), 0);
        const auto *state = static_cast<const NativeRunContext<DeviceRunnerBase> *>(call->storage);
        records.push_back(
            reinterpret_cast<const void *>(state->active_execution->prepared->kernel_args.args.run_result_data_base)
        );
        ASSERT_EQ(simpler_poll_run(&runner, call->storage), SIMPLER_NATIVE_RUN_POLL_COMPLETE);
        calls.push_back(std::move(call));
    }
    ASSERT_NE(records[0], records[1]);
    EXPECT_EQ(control().d2h_calls, 0u);
    EXPECT_TRUE(control().observed_streams.empty());
    for (size_t remaining = calls.size(); remaining != 0; --remaining) {
        const size_t index = remaining - 1;
        auto &call = *calls[index];
        const int stream_error = -735 - static_cast<int>(index);
        control().stream_sync_rc = stream_error;
        ASSERT_EQ(simpler_wait_run(&runner, call.storage), stream_error);
        testing::internal::CaptureStderr();
        const int finalized = simpler_finalize_run(&runner, call.storage);
        const std::string log = testing::internal::GetCapturedStderr();
        EXPECT_EQ(finalized, stream_error);
        EXPECT_NE(log.find("Skipping tensor copy-back because execution failed"), std::string::npos);
        EXPECT_TRUE(call.empty());
        ASSERT_EQ(control().downloads.size(), calls.size() - index);
        EXPECT_EQ(control().downloads.back(), records[index]);
        EXPECT_EQ(control().d2h_bytes, sizeof(DeviceRunResultRegion) * (calls.size() - index));
    }
    control().stream_sync_rc = 0;
}

TEST_F(NativeAdmissionTest, CompletedBoundariesCannotHideALaterStreamObserverError) {
    // The failed observer leaves quarantined device storage. The subprocess
    // owns its final disposal without fabricating a successful drain.
    EXPECT_EXIT(
        {
            FencedCompletionRunner runner;
            runner.use_inout_argument();
            Attempt active(runner, 0, 10);
            if (active.prepare() != 0 || simpler_launch_run(&runner, active.storage) != 0) std::_Exit(1);
            auto upload = std::find_if(control().uploads.begin(), control().uploads.end(), [&](const auto &item) {
                return item.second == active.bytes.data();
            });
            if (upload == control().uploads.end()) std::_Exit(9);
            std::memset(upload->first, 0x42, active.bytes.size());
            if (simpler_poll_run(&runner, active.storage) != SIMPLER_NATIVE_RUN_POLL_COMPLETE) std::_Exit(2);
            if (control().d2h_calls != 0 || !control().observed_streams.empty()) std::_Exit(3);
            control().stream_sync_rc = -735;
            if (simpler_wait_run(&runner, active.storage) != -735) std::_Exit(4);
            const size_t observations = control().observed_streams.size();
            if (observations == 0) std::_Exit(5);
            control().stream_sync_rc = 0;
            if (simpler_wait_run(&runner, active.storage) != -735) std::_Exit(6);
            if (simpler_finalize_run(&runner, active.storage) != -735) std::_Exit(7);
            if (control().observed_streams.size() != observations) std::_Exit(8);
            if (active.bytes != std::vector<uint8_t>(active.bytes.size(), 0x71)) std::_Exit(10);
            std::_Exit(0);
        },
        ::testing::ExitedWithCode(0), ""
    );
}

TEST_F(NativeAdmissionTest, FailedObservationWithoutBoundaryProofRetainsTheCall) {
    EXPECT_EXIT(
        {
            FencedCompletionRunner runner;
            Attempt active(runner, 0, 10);
            if (active.prepare() != 0 || simpler_launch_run(&runner, active.storage) != 0) std::_Exit(1);
            control().events_complete = false;
            control().event_wait_rc = -753;
            control().stream_sync_rc = -754;
            if (simpler_poll_run(&runner, active.storage) != SIMPLER_NATIVE_RUN_POLL_NOT_READY) std::_Exit(2);
            const int first_error = simpler_wait_run(&runner, active.storage);
            if (first_error == 0) std::_Exit(3);
            if (simpler_poll_run(&runner, active.storage) == SIMPLER_NATIVE_RUN_POLL_COMPLETE) std::_Exit(4);
            if (!runner.holds_epoch(10) || runner.reservations() != 1) std::_Exit(5);
            Attempt successor(runner, 0, 20);
            if (successor.prepare() == 0) std::_Exit(6);
            if (simpler_wait_run(&runner, active.storage) != first_error) std::_Exit(7);
            std::_Exit(0);
        },
        ::testing::ExitedWithCode(0), ""
    );
}

TEST_F(NativeAdmissionTest, SdkObserverErrorOutranksPrivateFailureAndPreservesProgramOutputs) {
    for (int private_error : {-100, -5}) {
        SCOPED_TRACE(private_error);
        EXPECT_EXIT(
            {
                FencedCompletionRunner runner;
                runner.use_inout_argument();
                runner.published_error = private_error;
                Attempt active(runner, 0, 10);
                if (active.prepare() != 0 || simpler_launch_run(&runner, active.storage) != 0) std::_Exit(1);
                const auto *state = static_cast<const NativeRunContext<DeviceRunnerBase> *>(active.storage);
                const void *record = reinterpret_cast<const void *>(
                    state->active_execution->prepared->kernel_args.args.run_result_data_base
                );
                void *cpu_stream = state->active_execution->prepared->streams.aicpu;
                auto upload = std::find_if(control().uploads.begin(), control().uploads.end(), [&](const auto &item) {
                    return item.second == active.bytes.data();
                });
                if (upload == control().uploads.end()) std::_Exit(2);
                std::memset(upload->first, 0x42, active.bytes.size());
                if (simpler_poll_run(&runner, active.storage) != SIMPLER_NATIVE_RUN_POLL_COMPLETE) std::_Exit(3);
                if (control().d2h_calls != 0 || !control().observed_streams.empty()) std::_Exit(4);
                control().stream_sync_rc = 507018;
                if (simpler_wait_run(&runner, active.storage) != 507018) std::_Exit(5);
                if (control().observed_streams != std::vector<void *>{cpu_stream}) std::_Exit(6);
                if (control().downloads != std::vector<const void *>{record}) std::_Exit(7);
                control().stream_sync_rc = 0;
                if (simpler_wait_run(&runner, active.storage) != 507018) std::_Exit(8);
                if (simpler_finalize_run(&runner, active.storage) != 507018) std::_Exit(9);
                if (!active.empty()) std::_Exit(10);
                if (control().observed_streams != std::vector<void *>{cpu_stream}) std::_Exit(11);
                if (control().downloads != std::vector<const void *>{record}) std::_Exit(12);
                if (control().d2h_bytes != sizeof(DeviceRunResultRegion)) std::_Exit(13);
                if (active.bytes != std::vector<uint8_t>(active.bytes.size(), 0x71)) std::_Exit(14);
                std::_Exit(0);
            },
            ::testing::ExitedWithCode(0), ""
        );
    }
}

TEST_F(NativeAdmissionTest, SdkErrorWithoutPhysicalCompletionRetainsTheOwner) {
    EXPECT_EXIT(
        {
            FencedCompletionRunner runner;
            runner.use_inout_argument();
            runner.published_error = -100;
            Attempt active(runner, 0, 10);
            if (active.prepare() != 0 || simpler_launch_run(&runner, active.storage) != 0) std::_Exit(1);
            const auto *state = static_cast<const NativeRunContext<DeviceRunnerBase> *>(active.storage);
            const void *record = reinterpret_cast<const void *>(
                state->active_execution->prepared->kernel_args.args.run_result_data_base
            );
            void *cpu_stream = state->active_execution->prepared->streams.aicpu;
            control().events_complete = false;
            control().stream_sync_rc = 507018;
            if (simpler_poll_run(&runner, active.storage) != SIMPLER_NATIVE_RUN_POLL_NOT_READY) std::_Exit(2);
            if (control().d2h_calls != 0 || !control().observed_streams.empty()) std::_Exit(3);
            if (simpler_wait_run(&runner, active.storage) != 507018) std::_Exit(4);
            if (state->physical_complete || active.empty()) std::_Exit(5);
            if (control().observed_streams != std::vector<void *>{cpu_stream}) std::_Exit(6);
            control().stream_sync_rc = 0;
            if (simpler_wait_run(&runner, active.storage) != 507018) std::_Exit(7);
            if (simpler_finalize_run(&runner, active.storage) != 507018) std::_Exit(8);
            if (state->physical_complete || active.empty()) std::_Exit(9);
            if (!runner.holds_epoch(10) || runner.reservations() != 1) std::_Exit(10);
            Attempt successor(runner, 0, 20);
            if (successor.prepare() == 0) std::_Exit(11);
            if (control().observed_streams != std::vector<void *>{cpu_stream}) std::_Exit(12);
            if (control().downloads != std::vector<const void *>{record}) std::_Exit(13);
            if (control().d2h_bytes != sizeof(DeviceRunResultRegion)) std::_Exit(14);
            if (active.bytes != std::vector<uint8_t>(active.bytes.size(), 0x71)) std::_Exit(15);
            std::_Exit(0);
        },
        ::testing::ExitedWithCode(0), ""
    );
}

TEST_F(NativeAdmissionTest, ProgramColdCleanupDebtAbandonsOnlyAfterResetAttempt) {
    class CloseRunner : public Runner {
    public:
        void own_sources() {
            ASSERT_EQ(execution_mode_latch().latch(SIMPLER_MODE_PROGRAM), 0);
            ASSERT_NE(mem_alloc_.alloc(64), nullptr);
            ASSERT_EQ(rtStreamCreate(&stream_aicpu_, 0), 0);
        }
        int abandon_without_reset() { return abandon_common_after_device_failure(); }
    };
    CloseRunner runner;
    runner.own_sources();
    control().loader_finalize_error = -4243;
    EXPECT_EQ(finalize_device(&runner), -4243);
    EXPECT_EQ(control().bytes, 64u);
    EXPECT_EQ(runner.teardown_proof(), TeardownProof::Unresolved);
    EXPECT_EQ(runner.abandon_without_reset(), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_FALSE(control().retired);
    EXPECT_EQ(simpler_execution_environment_releasable_ctx(&runner), 0);
    control().loader_finalize_error = 0;
    EXPECT_EQ(finalize_device(&runner), 0);
    EXPECT_EQ(control().bytes, 0u);
    EXPECT_EQ(control().calls_after_retire, 0u);
}

TEST_F(NativeAdmissionTest, NativeReleasableProofReadsRealSuccessfulCloseAndOutstandingUsers) {
    Runner runner;
    EXPECT_EQ(simpler_execution_environment_releasable_ctx(&runner), 0);
    ASSERT_EQ(finalize_device(&runner), 0);
    EXPECT_EQ(simpler_execution_environment_releasable_ctx(&runner), 1);
    int owner = 0;
    ASSERT_TRUE(runner.try_reserve_native_run(&owner, 1, arena_bank(1), true));
    EXPECT_EQ(simpler_execution_environment_releasable_ctx(&runner), 0);
    const size_t close_calls = control().loader_finalize_calls;
    EXPECT_NE(finalize_device(&runner), 0);
    EXPECT_EQ(control().loader_finalize_calls, close_calls);
    runner.release_native_run_reservation(&owner);
    EXPECT_EQ(simpler_execution_environment_releasable_ctx(&runner), 0);
    EXPECT_NE(finalize_device(&runner), 0);
    EXPECT_EQ(control().loader_finalize_calls, close_calls);
    runner.release_native_run_observation(&owner);
    ASSERT_EQ(finalize_device(&runner), 0);
    EXPECT_EQ(simpler_execution_environment_releasable_ctx(&runner), 1);
}

TEST_F(NativeAdmissionTest, KernelColdCleanupFailureStopsAdmissionAndRetainsItsControlStream) {
    Runner runner;
    ASSERT_EQ(runner.execution_mode_latch().latch(SIMPLER_MODE_KERNEL), 0);
    KernelContextOps ops{&runner,
                         fake_current_device,
                         fake_current_context,
                         fake_create_hidden_stream,
                         fake_destroy_hidden_stream,
                         fake_create_event,
                         fake_destroy_event};
    ASSERT_EQ(runner.kernel_execution_state().initialize(0, ops), 0);
    ASSERT_TRUE(runner.kernel_execution_state().accepts_dispatch());
    void *stream = runner.kernel_execution_state().hidden_stream(KernelStreamKind::Aicpu);
    control().loader_finalize_error = -4243;
    ASSERT_EQ(finalize_device(&runner), -4243);
    EXPECT_EQ(runner.kernel_execution_state().phase(), KernelContextPhase::Closing);
    EXPECT_FALSE(runner.kernel_execution_state().accepts_dispatch());
    EXPECT_EQ(runner.kernel_execution_state().hidden_stream(KernelStreamKind::Aicpu), stream);
    const uint8_t binary[] = {1};
    const auto image = make_chip_callable(nullptr, 0, "orch", binary, sizeof(binary), nullptr, nullptr, 0, "");
    EXPECT_EQ(
        simpler_kernel_mode_prepare_callable(&runner, 3, image.data(), image.size()), PTO_RUNTIME_ERR_INVALID_STATE
    );
    control().loader_finalize_error = 0;
    EXPECT_EQ(finalize_device(&runner), 0);
    EXPECT_EQ(runner.kernel_execution_state().phase(), KernelContextPhase::Closed);
    EXPECT_FALSE(runner.kernel_execution_state().has_live_resources());
}

TEST_F(NativeAdmissionTest, RoundedTimeoutContinuesAndPreservesStreamCreationFailure) {
    TimeoutEnvironment env({"3000000", "4000", "2000"});
    TimeoutRunner runner;
    control().replace_op_timeout = true;
    control().actual_op_timeout_us = 4294967;
    control().fail_stream_create = 1;
    const uint8_t binary[] = {1};
    EXPECT_EQ(
        simpler_init(&runner, 0, binary, sizeof(binary), binary, sizeof(binary), nullptr, 0, nullptr, 0, nullptr, 0),
        -4242
    );
    EXPECT_EQ(control().stream_create_calls, 1u);
    EXPECT_EQ(control().op_timeout_calls, 1u);
    EXPECT_EQ(runner.device_id(), 0);
    EXPECT_EQ(runner.timeouts().op_execute_timeout_us, 3000000u);
    EXPECT_EQ(runner.timeouts().actual_op_execute_timeout_us, 4294967u);
    EXPECT_EQ(runner.timeouts().stream_sync_timeout_ms, 4000);
    EXPECT_EQ(runner.timeouts().scheduler_timeout_ms, 2000);
    EXPECT_EQ(runner.initialization_error(), nullptr);
    EXPECT_FALSE(runner.program_initialized());
    EXPECT_EQ(runner.teardown_proof(), TeardownProof::Unresolved);
    for (int attempt = 0; attempt != 2; ++attempt) {
        EXPECT_EQ(runner.attach_current_thread(0), 0);
    }
    EXPECT_EQ(control().stream_create_calls, 1u);
    EXPECT_EQ(control().op_timeout_calls, 1u);
    EXPECT_EQ(
        simpler_init(&runner, 0, binary, sizeof(binary), binary, sizeof(binary), nullptr, 0, nullptr, 0, nullptr, 0),
        PTO_RUNTIME_ERR_INVALID_STATE
    );
    EXPECT_EQ(control().op_timeout_calls, 1u);
}

TEST_F(NativeAdmissionTest, OperatorTimeoutSetterFailureIsPreservedBeforeStreams) {
    TimeoutEnvironment env({nullptr, nullptr, nullptr});
    TimeoutRunner runner;
    control().op_timeout_rc = -4243;
    control().fail_stream_create = 1;
    const uint8_t binary[] = {1};
    EXPECT_EQ(
        simpler_init(&runner, 0, binary, sizeof(binary), binary, sizeof(binary), nullptr, 0, nullptr, 0, nullptr, 0),
        -4243
    );
    EXPECT_EQ(control().stream_create_calls, 0u);
    EXPECT_EQ(runner.device_id(), 0);
    EXPECT_FALSE(runner.program_initialized());
    EXPECT_EQ(runner.teardown_proof(), TeardownProof::Unresolved);
    for (int attempt = 0; attempt != 2; ++attempt) {
        EXPECT_EQ(runner.attach_current_thread(0), -4243);
        EXPECT_EQ(runner.ensure_device_initialized(), -4243);
    }
    EXPECT_EQ(control().stream_create_calls, 0u);
    EXPECT_EQ(control().op_timeout_calls, 1u);
}

TEST_F(NativeAdmissionTest, ActualTimeoutAtDefaultSchedulerDoesNotRejectAttachment) {
    TimeoutEnvironment env({nullptr, nullptr, nullptr});
    TimeoutRunner runner;
    control().replace_op_timeout = true;
    control().actual_op_timeout_us = static_cast<uint64_t>(PLATFORM_SCHEDULER_TIMEOUT_MS) * 1000;
    EXPECT_EQ(runner.attach_current_thread(0), 0);
    EXPECT_EQ(runner.timeouts().scheduler_timeout_ms, 0);
    EXPECT_EQ(runner.timeouts().op_execute_timeout_us, PLATFORM_OP_EXECUTE_TIMEOUT_US);
    EXPECT_EQ(runner.device_id(), 0);
}

TEST_F(NativeAdmissionTest, ActualTimeoutAtStreamBudgetDoesNotRejectAttachment) {
    TimeoutEnvironment env({"3000000", "5000", "1000"});
    TimeoutRunner runner;
    control().replace_op_timeout = true;
    control().actual_op_timeout_us = 5000000;
    EXPECT_EQ(runner.attach_current_thread(0), 0);
    EXPECT_EQ(runner.timeouts().stream_sync_timeout_ms, 5000);
    EXPECT_EQ(runner.initialization_error(), nullptr);
}

TEST_F(NativeAdmissionTest, ZeroReportedTimeoutDoesNotRejectSuccessfulSetter) {
    TimeoutEnvironment env({nullptr, nullptr, nullptr});
    TimeoutRunner runner;
    control().replace_op_timeout = true;
    EXPECT_EQ(runner.attach_current_thread(0), 0);
    EXPECT_EQ(runner.timeouts().actual_op_execute_timeout_us, 0u);
    EXPECT_EQ(runner.initialization_error(), nullptr);
}

TEST_F(NativeAdmissionTest, ValidAdjustedTimeoutPreservesRequestAndIsConfiguredOnce) {
    TimeoutEnvironment env({"3000000", "8000", "1000"});
    TimeoutRunner runner;
    control().replace_op_timeout = true;
    control().actual_op_timeout_us = 4294967;
    ASSERT_EQ(runner.attach_current_thread(0), 0);
    EXPECT_EQ(runner.timeouts().op_execute_timeout_us, 3000000u);
    EXPECT_EQ(runner.timeouts().actual_op_execute_timeout_us, 4294967u);
    EXPECT_EQ(control().requested_op_timeout_us, 3000000u);
    EXPECT_EQ(runner.timeouts().stream_sync_timeout_ms, 8000);
    EXPECT_EQ(runner.attach_current_thread(0), 0);
    EXPECT_EQ(control().op_timeout_calls, 1u);
}

TEST_F(NativeAdmissionTest, ValidDownwardAdjustmentIsAccepted) {
    TimeoutEnvironment env({nullptr, nullptr, nullptr});
    TimeoutRunner runner;
    control().replace_op_timeout = true;
    control().actual_op_timeout_us = 30000000;
    EXPECT_EQ(runner.attach_current_thread(0), 0);
    EXPECT_EQ(runner.timeouts().op_execute_timeout_us, PLATFORM_OP_EXECUTE_TIMEOUT_US);
}

TEST_F(NativeAdmissionTest, BorrowedDeviceDoesNotConfigureTheGlobalOperatorTimeout) {
    TimeoutEnvironment env({"3000000", "4000", "2000"});
    TimeoutRunner runner;
    ASSERT_EQ(runner.execution_mode_latch().latch(SIMPLER_MODE_KERNEL), 0);
    KernelContextOps ops;
    ops.get_current_device = fake_current_device;
    ops.get_current_context = fake_current_context;
    ops.create_hidden_stream = fake_create_hidden_stream;
    ops.destroy_hidden_stream = fake_destroy_hidden_stream;
    ops.create_event = fake_create_event;
    ops.destroy_event = fake_destroy_event;
    ASSERT_EQ(runner.kernel_execution_state().initialize(0, ops), 0);
    control().op_timeout_rc = -4243;
    EXPECT_EQ(runner.adopt_borrowed_device(0), 0);
    EXPECT_EQ(control().op_timeout_calls, 0u);
    EXPECT_EQ(runner.timeouts().actual_op_execute_timeout_us, 0u);
}

TEST_F(NativeAdmissionTest, RegistrationRefusesWhileANativeRunIsOutstanding) {
    Runner runner;
    ASSERT_EQ(runner.execution_mode_latch().latch(SIMPLER_MODE_KERNEL), 0);
    KernelContextOps ops;
    ops.context = &runner;
    ops.get_current_device = fake_current_device;
    ops.get_current_context = fake_current_context;
    ops.create_hidden_stream = fake_create_hidden_stream;
    ops.destroy_hidden_stream = fake_destroy_hidden_stream;
    ops.create_event = fake_create_event;
    ops.destroy_event = fake_destroy_event;
    ASSERT_EQ(runner.kernel_execution_state().initialize(0, ops), 0);
    ASSERT_EQ(runner.kernel_execution_state().mark_ready_enqueued(), 0);
    ASSERT_TRUE(runner.kernel_execution_state().accepts_dispatch());

    int owner = 0;
    ASSERT_TRUE(runner.try_reserve_native_run(&owner, 1, arena_bank(1), true));
    ASSERT_TRUE(runner.native_runs_outstanding());

    const uint8_t binary[] = {0x01};
    const auto image = make_chip_callable(nullptr, 0, "orch", binary, sizeof(binary), nullptr, nullptr, 0, "");
    int32_t auto_id = 99;
    EXPECT_EQ(simpler_register_callable(&runner, 2, image.data()), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(simpler_kernel_mode_prepare_callable(&runner, 3, image.data(), image.size()), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(simpler_register_callable_v2(&runner, image.data(), image.size(), &auto_id), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(auto_id, -1);
    runner.release_native_run_reservation(&owner);
}

TEST_F(NativeAdmissionTest, AutomaticRegistrationDoesNotRetryFailedProgramInitialization) {
    class FreshRunner : public Runner {
    public:
        FreshRunner() {
            device_id_ = -1;
            callables_.clear();
            chip_callable_buffers_.clear();
        }
    } runner;
    const uint8_t binary[] = {0x01};
    control().fail_stream_create = 1;
    ASSERT_EQ(
        simpler_init(&runner, 0, binary, sizeof(binary), binary, sizeof(binary), nullptr, 0, nullptr, 0, nullptr, 0),
        -4242
    );
    ASSERT_EQ(control().stream_create_calls, 1u);
    ASSERT_EQ(runner.device_id(), 0);
    ASSERT_EQ(runner.execution_mode_latch().latched_mode(), SIMPLER_MODE_PROGRAM);
    ASSERT_EQ(runner.teardown_proof(), TeardownProof::Unresolved);

    const auto image = make_chip_callable(nullptr, 0, "orch", binary, sizeof(binary), nullptr, nullptr, 0, "");
    const auto allocations = control().allocations;
    for (int attempt = 0; attempt < 2; ++attempt) {
        int32_t id = 99;
        EXPECT_EQ(
            simpler_register_callable_v2(&runner, image.data(), image.size(), &id), PTO_RUNTIME_ERR_INVALID_STATE
        );
        EXPECT_EQ(id, -1);
        EXPECT_EQ(control().stream_create_calls, 1u);
        EXPECT_EQ(control().allocations, allocations);
        EXPECT_FALSE(runner.has_callable(0));
    }
    EXPECT_EQ(finalize_device(&runner), 0);
    int32_t id = 99;
    EXPECT_EQ(simpler_register_callable_v2(&runner, image.data(), image.size(), &id), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_EQ(id, -1);
}

TEST_F(NativeAdmissionTest, EarlyPrepareFailureDoesNotBlockTheNormalRunComposition) {
    EXPECT_EQ(simpler_prepare_run(nullptr, nullptr, 0, nullptr, nullptr, nullptr), PTO_RUNTIME_ERR_INTERNAL);
    Runner runner;
    Attempt failed(runner, 0, 10);
    failed.config.aicpu_thread_num = 1;
    EXPECT_EQ(failed.prepare(), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_TRUE(failed.consume_release());
    EXPECT_TRUE(failed.empty());
    Attempt next(runner, 0, 20);
    EXPECT_EQ(simpler_run(&runner, next.storage, 1, &next.args, &next.config, &next.descriptor), 0);
    EXPECT_TRUE(next.empty());
    EXPECT_EQ(runner.reservations(), 0u);
    EXPECT_FALSE(runner.holds_epoch(20));
}

TEST_F(NativeAdmissionTest, OwnerAllocationFailureReturnsPreparedWithoutEnteringTheBackend) {
    Runner runner;
    Runtime runtime;
    const NativeRunIdentity identity{10, 10, 10, 0};
    auto prepared = std::make_unique<Runner::PreparedExecution>(identity, runtime, CallConfig{}, 0);
    auto *original = prepared.get();
    refuse_next_host_allocation = true;
    auto outcome = runner.launch_execution(std::move(prepared), NativeRunExecutionTestPeer::mint(identity));
    const bool allocation_attempted = !std::exchange(refuse_next_host_allocation, false);
    EXPECT_TRUE(allocation_attempted);
    EXPECT_EQ(outcome.rc, PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(outcome.progress, LaunchProgress::NotStarted);
    EXPECT_EQ(outcome.prepared.get(), original);
    EXPECT_EQ(outcome.active, nullptr);
    EXPECT_FALSE(outcome.receipt.valid());
    EXPECT_EQ(runner.launch_calls, 0);
    EXPECT_EQ(runner.recovery_calls, 0);
    EXPECT_TRUE(runner.can_accept_run());
}

TEST_F(NativeAdmissionTest, FailurePolicyCannotReplaceTheResultOrLoseTheOwner) {
    Runner runner;
    Runtime runtime;
    const NativeRunIdentity identity{10, 10, 10, 0};
    auto prepared = std::make_unique<Runner::PreparedExecution>(identity, runtime, CallConfig{}, 0);
    auto *original = prepared.get();
    runner.launch_refusal_rc = -4321;
    runner.partial_launch = true;
    runner.recovery_throws = true;
    auto outcome = runner.launch_execution(std::move(prepared), NativeRunExecutionTestPeer::mint(identity));
    EXPECT_EQ(outcome.rc, -4321);
    EXPECT_EQ(outcome.progress, LaunchProgress::Partial);
    ASSERT_NE(outcome.active, nullptr);
    EXPECT_EQ(outcome.active->prepared.get(), original);
    EXPECT_FALSE(outcome.receipt.valid());
    EXPECT_FALSE(runner.can_accept_run());
    EXPECT_EQ(runner.recovery_calls, 1);
    EXPECT_EQ(runner.recovery_rc, -4321);
}

TEST_F(NativeAdmissionTest, CaptureColdPreparationsKeepTheirOwnersAndExclusiveExecutionClaims) {
    Runner runner;
    ASSERT_TRUE(runner.set_run_resource_grant(2));
    ASSERT_EQ(initialize_capture_runner(runner), 0);
    Attempt first(runner, 0, 9100), second(runner, 1, 9101);
    use_capture_preparation(first);
    use_capture_preparation(second);
    ASSERT_EQ(first.prepare(), 0);
    auto *first_state = static_cast<NativeRunContext<DeviceRunnerBase> *>(first.storage);
    const auto first_resources = first_state->prepared_execution->prepared_resources;
    std::array<void *, 3> backing;
    std::array<std::array<uint8_t, 64>, 3> original;
    for (size_t i = 0; i != backing.size(); ++i) {
        backing[i] = runner.arena_region_backing(arena_bank(0), i);
        if (backing[i] != nullptr) std::memcpy(original[i].data(), backing[i], original[i].size());
    }
    ASSERT_EQ(second.prepare(), 0);
    auto *second_state = static_cast<NativeRunContext<DeviceRunnerBase> *>(second.storage);
    EXPECT_NE(first_resources, second_state->prepared_execution->prepared_resources);
    EXPECT_EQ(runner.reservations(), 2u);
    EXPECT_EQ(runner.native_run_claim_count(), 0u);
    EXPECT_EQ(control().d2h_calls, 0u);
    EXPECT_TRUE(control().observed_streams.empty());
    for (size_t i = 0; i != backing.size(); ++i) {
        if (backing[i] != nullptr) EXPECT_EQ(std::memcmp(original[i].data(), backing[i], original[i].size()), 0);
    }
    LaunchPermit first_permit, second_permit;
    ASSERT_TRUE(runner.try_acquire_native_run(first.storage, first_state->identity(), &first_permit));
    EXPECT_FALSE(runner.try_acquire_native_run(second.storage, second_state->identity(), &second_permit));
    runner.release_native_run(first.storage);
    EXPECT_TRUE(runner.try_acquire_native_run(second.storage, second_state->identity(), &second_permit));
    runner.release_native_run(second.storage);
    ASSERT_EQ(simpler_finalize_run(&runner, first.storage), 0);
    EXPECT_EQ(runner.reservations(), 1u);
    EXPECT_EQ(second_state->phase.load(), NativeRunPhase::Prepared);
    ASSERT_EQ(simpler_finalize_run(&runner, second.storage), 0);
}

TEST_F(NativeAdmissionTest, CaptureColdPreparationKeepsSlotCapacityAndEagerReservationGuards) {
    for (bool first_is_capture : {false, true}) {
        SCOPED_TRACE(first_is_capture);
        Runner runner;
        ASSERT_TRUE(runner.set_run_resource_grant(2));
        ASSERT_EQ(initialize_capture_runner(runner), 0);
        Attempt first(runner, 0, 9200), second(runner, 1, 9201), duplicate(runner, 0, 9202), excess(runner, 2, 9203);
        use_capture_preparation(first);
        use_capture_preparation(second);
        use_capture_preparation(duplicate);
        use_capture_preparation(excess);
        if (first_is_capture) second.descriptor.prepare_purpose = SIMPLER_PREPARE_EAGER;
        else first.descriptor.prepare_purpose = SIMPLER_PREPARE_EAGER;
        ASSERT_EQ(first.prepare(), 0);
        const size_t allocations = control().allocations;
        EXPECT_NE(second.prepare(), 0);
        EXPECT_NE(duplicate.prepare(), 0);
        EXPECT_NE(excess.prepare(), 0);
        EXPECT_EQ(control().allocations, allocations);
        EXPECT_EQ(runner.reservations(), 1u);
        EXPECT_EQ(runner.native_run_claim_count(), 0u);
        ASSERT_EQ(simpler_finalize_run(&runner, first.storage), 0);
    }
}

TEST_F(NativeAdmissionTest, CaptureColdPreparationFailurePreservesTheEarlierBackingAndAllowsRetry) {
    Runner runner;
    ASSERT_TRUE(runner.set_run_resource_grant(2));
    ASSERT_EQ(initialize_capture_runner(runner), 0);
    Attempt first(runner, 0, 9300), second(runner, 1, 9301);
    use_capture_preparation(first);
    use_capture_preparation(second);
    ASSERT_EQ(first.prepare(), 0);
    auto *state = static_cast<NativeRunContext<DeviceRunnerBase> *>(first.storage);
    const auto resources = state->prepared_execution->prepared_resources;
    void *const first_heap = runner.arena_region_backing(arena_bank(0), 0);
    ASSERT_NE(first_heap, nullptr);
    std::array<uint8_t, 64> original;
    std::memcpy(original.data(), first_heap, original.size());
    const size_t allocation_failures = refused_host_allocation_count;
    refuse_next_host_allocation = true;
    const int failed_prepare_rc = second.prepare();
    const bool allocation_not_reached = std::exchange(refuse_next_host_allocation, false);
    EXPECT_NE(failed_prepare_rc, 0);
    EXPECT_FALSE(allocation_not_reached);
    EXPECT_EQ(refused_host_allocation_count, allocation_failures + 1);
    EXPECT_EQ(runner.reservations(), 1u);
    EXPECT_EQ(state->prepared_execution->prepared_resources, resources);
    EXPECT_EQ(std::memcmp(original.data(), first_heap, original.size()), 0);
    second.descriptor.run_epoch = 9302;
    ASSERT_EQ(second.prepare(), 0);
    EXPECT_EQ(std::memcmp(original.data(), first_heap, original.size()), 0);
    ASSERT_EQ(simpler_finalize_run(&runner, second.storage), 0);
    ASSERT_EQ(simpler_finalize_run(&runner, first.storage), 0);
}

TEST_F(NativeAdmissionTest, CaptureColdPreparationRejectsAnotherBindUntilItsOwnerIsPublished) {
    class PausedRunner : public Runner {
    public:
        std::promise<void> arrived;
        std::promise<void> resume;
        bool fail_first{false};
        int prepare_execution(
            Runtime &runtime, const CallConfig &config, uint32_t slot, const NativeRunIdentity &identity,
            std::unique_ptr<PreparedExecution> *out
        ) override {
            const int rc = Runner::prepare_execution(runtime, config, slot, identity, out);
            if (slot == 0 && !paused_) {
                paused_ = true;
                arrived.set_value();
                resume.get_future().wait();
                if (fail_first) return -4321;
            }
            return rc;
        }

    private:
        bool paused_{false};
    };
    for (bool fail_first : {false, true}) {
        SCOPED_TRACE(fail_first);
        PausedRunner runner;
        runner.fail_first = fail_first;
        ASSERT_TRUE(runner.set_run_resource_grant(2));
        ASSERT_EQ(initialize_capture_runner(runner), 0);
        Attempt first(runner, 0, 9400), second(runner, 1, 9401);
        use_capture_preparation(first);
        use_capture_preparation(second);
        second.config.runtime_env.ring_heap[0] *= 2;
        std::strcpy(first.config.output_prefix, "first-prepare");
        std::strcpy(second.config.output_prefix, "second-prepare");
        auto arrived = runner.arrived.get_future();
        auto pending = std::async(std::launch::async, [&] {
            return first.prepare();
        });
        const auto ready = arrived.wait_for(std::chrono::seconds(5));
        if (ready != std::future_status::ready) {
            runner.resume.set_value();
            EXPECT_EQ(pending.get(), fail_first ? -4321 : 0);
            FAIL() << "First prepare did not reach the pre-owner barrier";
        }
        void *const first_heap = runner.arena_region_backing(arena_bank(0), 0);
        const size_t allocations = control().allocations;
        const int second_rc = second.prepare();
        EXPECT_NE(second_rc, 0);
        EXPECT_EQ(control().allocations, allocations);
        EXPECT_EQ(runner.arena_region_backing(arena_bank(0), 0), first_heap);
        EXPECT_EQ(runner.output_prefix(), "first-prepare");
        runner.resume.set_value();
        EXPECT_EQ(pending.get(), fail_first ? -4321 : 0);
        if (second_rc == 0) ASSERT_EQ(simpler_finalize_run(&runner, second.storage), 0);
        if (fail_first) {
            EXPECT_TRUE(first.consume_release());
            EXPECT_EQ(runner.reservations(), 0u);
        }
        ASSERT_EQ(second.prepare(), 0);
        if (fail_first) {
            first.descriptor.run_epoch = 9402;
            ASSERT_EQ(first.prepare(), 0);
        }
        ASSERT_EQ(simpler_finalize_run(&runner, second.storage), 0);
        ASSERT_EQ(simpler_finalize_run(&runner, first.storage), 0);
    }
}

TEST_F(NativeAdmissionTest, CapacityRefusalAllowsTheSamePreparedIdentityToLaunchAgain) {
    for (bool kernel : {false, true}) {
        SCOPED_TRACE(kernel ? "Kernel" : "Program");
        ArmedAdmissionRunner runner;
        ASSERT_EQ(runner.execution_mode_latch().latch(kernel ? SIMPLER_MODE_KERNEL : SIMPLER_MODE_PROGRAM), 0);
        if (kernel) {
            KernelContextOps ops;
            ops.context = &runner;
            ops.get_current_device = fake_current_device;
            ops.get_current_context = fake_current_context;
            ops.create_hidden_stream = fake_create_hidden_stream;
            ops.destroy_hidden_stream = fake_destroy_hidden_stream;
            ops.create_event = fake_create_event;
            ops.destroy_event = fake_destroy_event;
            ASSERT_EQ(runner.kernel_execution_state().initialize(0, ops), 0);
            ASSERT_EQ(runner.kernel_execution_state().mark_ready_enqueued(), 0);
        }
        Attempt candidate(runner, 0, 10);
        if (kernel) {
            candidate.args.tensor(0).address_space = AddressSpace::DEVICE;
            candidate.args.tensor(0).transfer = TensorTransfer::NONE;
        }
        ASSERT_EQ(candidate.prepare(), 0);
        auto *state = static_cast<NativeRunContext<DeviceRunnerBase> *>(candidate.storage);
        auto *prepared = state->prepared_execution.get();
        const auto identity = prepared->identity;
        int caller;
        auto launch = [&] {
            return kernel ? simpler_launch_run_on_stream(&runner, candidate.storage, &caller, nullptr) :
                            simpler_launch_run(&runner, candidate.storage);
        };
        runner.launch_refusal_rc = SIMPLER_RUNTIME_ERR_DIAGNOSTIC_CAPACITY_FULL;
        EXPECT_EQ(launch(), SIMPLER_RUNTIME_ERR_DIAGNOSTIC_CAPACITY_FULL);
        EXPECT_EQ(state->phase.load(), NativeRunPhase::Prepared);
        EXPECT_EQ(state->prepared_execution.get(), prepared);
        EXPECT_EQ(state->active_execution, nullptr);
        EXPECT_EQ(runner.recovery_calls, 0);
        EXPECT_EQ(runner.launch_calls, 1);

        runner.launch_refusal_rc = 0;
        EXPECT_EQ(launch(), 0);
        EXPECT_EQ(runner.launch_calls, 2);
        // The transport's inherited launch does not enqueue caller dependencies.
        EXPECT_EQ(runner.caller_order(0).retire(identity), 0);
        EXPECT_EQ(simpler_wait_run(&runner, candidate.storage), 0);
        EXPECT_EQ(simpler_finalize_run(&runner, candidate.storage), 0);
        if (!candidate.empty()) EXPECT_TRUE(candidate.consume_release());
        EXPECT_TRUE(candidate.empty());
    }
}

TEST_F(NativeAdmissionTest, UnstartedLaunchRetiresOnlyItsUnsubmittedCallerOrder) {
    CallerOrderRunner runner;
    Runtime runtime;
    const NativeRunIdentity identity{10, 10, 10, 0};
    int cpu, core;
    auto &order = runner.caller_order(0);
    ASSERT_EQ(order.arm(identity, nullptr, &cpu, &core), 0);
    auto prepared = std::make_unique<Runner::PreparedExecution>(identity, runtime, CallConfig{}, 0);
    auto *original = prepared.get();
    runner.launch_refusal_rc = SIMPLER_RUNTIME_ERR_DIAGNOSTIC_CAPACITY_FULL;
    auto outcome = runner.launch_execution(std::move(prepared), NativeRunExecutionTestPeer::mint(identity));
    EXPECT_EQ(outcome.rc, SIMPLER_RUNTIME_ERR_DIAGNOSTIC_CAPACITY_FULL);
    EXPECT_EQ(outcome.progress, LaunchProgress::NotStarted);
    EXPECT_EQ(outcome.prepared.get(), original);
    EXPECT_EQ(outcome.active, nullptr);
    EXPECT_EQ(runner.recovery_calls, 0);
    EXPECT_EQ(order.arm(identity, nullptr, &cpu, &core), 0);
    EXPECT_EQ(order.retire(identity), 0);
}

TEST_F(NativeAdmissionTest, RefusedCallerRollbackKeepsActiveOwnershipAndThePrimaryError) {
    CallerOrderRunner runner;
    Runtime runtime;
    const NativeRunIdentity identity{10, 10, 10, 0};
    const NativeRunIdentity other{11, 11, 11, 0};
    int cpu, core;
    auto &order = runner.caller_order(0);
    ASSERT_EQ(order.arm(other, nullptr, &cpu, &core), 0);
    auto prepared = std::make_unique<Runner::PreparedExecution>(identity, runtime, CallConfig{}, 0);
    auto *original = prepared.get();
    runner.launch_refusal_rc = SIMPLER_RUNTIME_ERR_DIAGNOSTIC_CAPACITY_FULL;
    runner.recovery_throws = true;
    auto outcome = runner.launch_execution(std::move(prepared), NativeRunExecutionTestPeer::mint(identity));
    EXPECT_EQ(outcome.rc, SIMPLER_RUNTIME_ERR_DIAGNOSTIC_CAPACITY_FULL);
    EXPECT_EQ(outcome.progress, LaunchProgress::Partial);
    EXPECT_EQ(outcome.prepared, nullptr);
    EXPECT_NE(outcome.active, nullptr);
    if (outcome.active != nullptr) {
        EXPECT_EQ(outcome.active->prepared.get(), original);
        EXPECT_EQ(outcome.active->progress, LaunchProgress::Partial);
    }
    EXPECT_FALSE(outcome.receipt.valid());
    EXPECT_EQ(runner.recovery_calls, 1);
    EXPECT_EQ(runner.recovery_rc, SIMPLER_RUNTIME_ERR_DIAGNOSTIC_CAPACITY_FULL);
    EXPECT_FALSE(runner.can_accept_run());
    EXPECT_NE(order.arm(identity, nullptr, &cpu, &core), 0);
    EXPECT_EQ(order.retire(other), 0);
}

TEST_F(NativeAdmissionTest, PartialLaunchDoesNotRetireItsCallerOrder) {
    CallerOrderRunner runner;
    Runtime runtime;
    const NativeRunIdentity identity{10, 10, 10, 0};
    int cpu, core;
    auto &order = runner.caller_order(0);
    ASSERT_EQ(order.arm(identity, nullptr, &cpu, &core), 0);
    auto prepared = std::make_unique<Runner::PreparedExecution>(identity, runtime, CallConfig{}, 0);
    auto *original = prepared.get();
    runner.launch_refusal_rc = SIMPLER_RUNTIME_ERR_DIAGNOSTIC_CAPACITY_FULL;
    runner.partial_launch = true;
    auto outcome = runner.launch_execution(std::move(prepared), NativeRunExecutionTestPeer::mint(identity));
    EXPECT_EQ(outcome.rc, SIMPLER_RUNTIME_ERR_DIAGNOSTIC_CAPACITY_FULL);
    EXPECT_EQ(outcome.progress, LaunchProgress::Partial);
    EXPECT_EQ(outcome.prepared, nullptr);
    ASSERT_NE(outcome.active, nullptr);
    EXPECT_EQ(outcome.active->prepared.get(), original);
    EXPECT_EQ(outcome.active->progress, LaunchProgress::Partial);
    EXPECT_EQ(runner.recovery_calls, 1);
    EXPECT_EQ(order.arm(identity, nullptr, &cpu, &core), PTO_RUNTIME_ERR_INVALID_STATE);
    // The transport's inherited launch does not enqueue caller dependencies.
    EXPECT_EQ(order.retire(identity), 0);
}

TEST_F(NativeAdmissionTest, CompleteSubmissionHandsOffItsOwnerWithoutAnyLaterAllocation) {
    Runner runner;
    Runtime runtime;
    const NativeRunIdentity identity{10, 10, 10, 0};
    auto prepared = std::make_unique<Runner::PreparedExecution>(identity, runtime, CallConfig{}, 0);
    auto *original = prepared.get();
    runner.forbid_post_submit_allocation = true;
    auto outcome = runner.launch_execution(std::move(prepared), NativeRunExecutionTestPeer::mint(identity));
    const bool nothing_allocated = std::exchange(refuse_next_host_allocation, false);
    EXPECT_TRUE(nothing_allocated);
    EXPECT_EQ(outcome.rc, 0);
    EXPECT_EQ(outcome.progress, LaunchProgress::Complete);
    ASSERT_NE(outcome.active, nullptr);
    EXPECT_EQ(outcome.active->prepared.get(), original);
    EXPECT_EQ(outcome.active->progress, LaunchProgress::Complete);
    EXPECT_EQ(outcome.prepared, nullptr);
    EXPECT_TRUE(outcome.receipt.matches(identity));
}

TEST_F(NativeAdmissionTest, UnstartedLaunchCleanupPublishesOnlyItsOwnReleaseProof) {
    Runner runner;
    Attempt rejected(runner, 0, 10);
    ASSERT_EQ(rejected.prepare(), 0);
    runner.launch_refusal_rc = -4321;
    EXPECT_EQ(simpler_launch_run(&runner, rejected.storage), -4321);
    EXPECT_EQ(simpler_poll_run(&runner, rejected.storage), SIMPLER_NATIVE_RUN_POLL_COMPLETE);
    EXPECT_EQ(simpler_finalize_run(&runner, rejected.storage), 0);
    EXPECT_TRUE(rejected.consume_release());
    EXPECT_TRUE(rejected.empty());
    EXPECT_EQ(runner.reservations(), 0u);
    EXPECT_FALSE(runner.holds_epoch(10));
    runner.launch_refusal_rc = 0;
    Attempt next(runner, 0, 20);
    EXPECT_EQ(simpler_run(&runner, next.storage, 1, &next.args, &next.config, &next.descriptor), 0);
    EXPECT_TRUE(next.empty());
}

TEST_F(NativeAdmissionTest, PartialLaunchNeverPublishesAnUnstartedReleaseProof) {
    Runner runner;
    Attempt rejected(runner, 0, 10);
    ASSERT_EQ(rejected.prepare(), 0);
    runner.launch_refusal_rc = -4321;
    runner.partial_launch = true;
    EXPECT_EQ(simpler_launch_run(&runner, rejected.storage), -4321);
    EXPECT_EQ(simpler_wait_run(&runner, rejected.storage), -4321);
    EXPECT_EQ(simpler_finalize_run(&runner, rejected.storage), -4321);
    EXPECT_FALSE(rejected.consume_release());
    EXPECT_TRUE(rejected.empty());
}

TEST_F(NativeAdmissionTest, ThrowingBackendRetainsTheOwnerAndNeverPublishesNoSubmission) {
    Runner runner;
    Attempt candidate(runner, 0, 10);
    ASSERT_EQ(candidate.prepare(), 0);
    runner.launch_throws = true;
    EXPECT_EQ(simpler_launch_run(&runner, candidate.storage), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(runner.recovery_calls, 1);
    EXPECT_EQ(runner.recovery_rc, PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_FALSE(runner.can_accept_run());
    auto *state = static_cast<NativeRunContext<DeviceRunnerBase> *>(candidate.storage);
    ASSERT_NE(state->active_execution, nullptr);
    EXPECT_EQ(state->active_execution->progress, LaunchProgress::Partial);
    EXPECT_TRUE(runner.holds_epoch(10));
    EXPECT_EQ(runner.drain_calls, 0);
    Attempt blocked(runner, 1, 20);
    const auto allocations = control().allocations;
    EXPECT_EQ(blocked.prepare(), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_TRUE(blocked.empty());
    EXPECT_EQ(control().allocations, allocations);

    EXPECT_EQ(simpler_finalize_run(&runner, candidate.storage), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(runner.drain_calls, 1);
    EXPECT_FALSE(candidate.consume_release());
    EXPECT_TRUE(candidate.empty());
}

TEST_F(NativeAdmissionTest, ThrowAfterSubmissionRetainsTheCarrierAndAllBindingsUntilDrain) {
    Runner runner;
    Attempt candidate(runner, 0, 10);
    volatile int32_t accepted = 73;
    candidate.descriptor.accepted_state = &accepted;
    candidate.descriptor.accepted_value = 1;
    ASSERT_EQ(candidate.prepare(), 0);
    const auto bytes = control().bytes;
    runner.launch_throws_after_submission = true;
    EXPECT_EQ(simpler_launch_run(&runner, candidate.storage), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(runner.recovery_calls, 1);
    EXPECT_EQ(runner.recovery_rc, PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_FALSE(runner.can_accept_run());
    auto *state = static_cast<NativeRunContext<DeviceRunnerBase> *>(candidate.storage);
    ASSERT_NE(state->active_execution, nullptr);
    ASSERT_NE(state->active_execution->prepared, nullptr);
    EXPECT_EQ(state->active_execution->prepared->runtime, state->runtime.get());
    EXPECT_EQ(state->active_execution->progress, LaunchProgress::Partial);
    EXPECT_EQ(accepted, 73);
    EXPECT_EQ(simpler_poll_run(&runner, candidate.storage), SIMPLER_NATIVE_RUN_POLL_NOT_READY);
    EXPECT_EQ(runner.reservations(), 1u);
    EXPECT_TRUE(runner.holds_epoch(10));
    EXPECT_EQ(control().bytes, bytes);
    EXPECT_EQ(runner.drain_calls, 0);
    EXPECT_EQ(simpler_wait_run(&runner, candidate.storage), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(runner.drain_calls, 1);
    EXPECT_EQ(simpler_finalize_run(&runner, candidate.storage), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_FALSE(candidate.consume_release());
    EXPECT_TRUE(candidate.empty());
    EXPECT_EQ(runner.reservations(), 0u);
    EXPECT_FALSE(runner.holds_epoch(10));
    EXPECT_EQ(runner.drain_calls, 1);
}

TEST_F(NativeAdmissionTest, UnstartedLaunchWithFailedCleanupPublishesNoReleaseProof) {
    Runner runner;
    Attempt rejected(runner, 0, 10);
    ASSERT_EQ(rejected.prepare(), 0);
    runner.launch_refusal_rc = -4321;
    EXPECT_EQ(simpler_launch_run(&runner, rejected.storage), -4321);
    control().attach_rc = -77;
    EXPECT_EQ(simpler_finalize_run(&runner, rejected.storage), -77);
    EXPECT_FALSE(rejected.consume_release());
    EXPECT_FALSE(rejected.empty());
    EXPECT_EQ(runner.reservations(), 1u);
    control().attach_rc = 0;
    EXPECT_EQ(simpler_finalize_run(&runner, rejected.storage), 0);
    EXPECT_TRUE(rejected.consume_release());
    EXPECT_TRUE(rejected.empty());
    EXPECT_EQ(runner.reservations(), 0u);
}

TEST_F(NativeAdmissionTest, LostDrainAttachmentPreservesSubmittedOwnershipAndAllowsObservationRetry) {
    EXPECT_EXIT(
        {
            Runner runner;
            Attempt active(runner, 0, 10);
            if (active.prepare() != 0 || simpler_launch_run(&runner, active.storage) != 0) std::_Exit(1);
            control().attach_rc = -77;
            if (simpler_wait_run(&runner, active.storage) != -77) std::_Exit(2);
            if (!runner.accepts_new_run() || !runner.holds_epoch(10) || active.empty()) std::_Exit(3);
            if (runner.drain_calls != 0 || runner.quarantine_calls != 0 || runner.reservations() != 1) std::_Exit(4);
            control().attach_rc = 0;
            Attempt blocked(runner, 0, 20);
            if (blocked.prepare() == 0 || runner.launch_calls != 1) std::_Exit(5);
            if (simpler_wait_run(&runner, active.storage) != 0 || simpler_finalize_run(&runner, active.storage) != 0)
                std::_Exit(6);
            if (!active.empty() || runner.holds_epoch(10) || runner.native_runs_outstanding()) std::_Exit(7);
            if (blocked.prepare() != 0 || simpler_finalize_run(&runner, blocked.storage) != 0) std::_Exit(8);
            std::_Exit(0);
        },
        ::testing::ExitedWithCode(0), ""
    );
    EXPECT_EXIT(
        {
            Runner runner;
            Attempt active(runner, 0, 10);
            if (active.prepare() != 0 || simpler_launch_run(&runner, active.storage) != 0) std::_Exit(1);
            control().attach_rc = -77;
            if (simpler_finalize_run(&runner, active.storage) != -77) std::_Exit(2);
            if (runner.accepts_new_run() || active.empty() || !runner.holds_epoch(10) ||
                !runner.native_runs_outstanding())
                std::_Exit(3);
            if (runner.drain_calls != 0 || runner.quarantine_rc != -77 || runner.recovery_calls != 0 ||
                runner.reservations() != 1)
                std::_Exit(4);
            control().attach_rc = 0;
            Attempt next(runner, 1, 20);
            if (next.prepare() == 0 || runner.launch_calls != 1) std::_Exit(5);
            std::_Exit(0);
        },
        ::testing::ExitedWithCode(0), ""
    );
}

TEST_F(NativeAdmissionTest, EscapingDrainExceptionStopsAdmissionAndPreservesResources) {
    for (bool finalize_without_wait : {false, true}) {
        SCOPED_TRACE(finalize_without_wait);
        EXPECT_EXIT(
            {
                Runner runner;
                Attempt active(runner, 0, 10);
                if (active.prepare() != 0 || simpler_launch_run(&runner, active.storage) != 0) std::_Exit(1);
                runner.drain_throws = true;
                runner.quarantine_throws = true;
                const int rc = finalize_without_wait ? simpler_finalize_run(&runner, active.storage) :
                                                       simpler_wait_run(&runner, active.storage);
                if (rc != PTO_RUNTIME_ERR_INTERNAL) std::_Exit(2);
                if (runner.accepts_new_run() || active.consume_release()) std::_Exit(3);
                SimplerWorkspaceReport report{};
                if (!runner.workspace_report(&report)) std::_Exit(3);
                if (!runner.holds_epoch(10) || active.empty() || !runner.native_runs_outstanding() ||
                    runner.reservations() != 1)
                    std::_Exit(3);
                if (runner.drain_calls != 1 || runner.quarantine_calls != 1) std::_Exit(4);
                Attempt next(runner, 1, 20);
                if (next.prepare() == 0 || runner.launch_calls != 1) std::_Exit(5);
                std::_Exit(0);
            },
            ::testing::ExitedWithCode(0), ""
        );
    }
}

TEST_F(NativeAdmissionTest, DiagnosticsFailureAfterProvedCompletionDoesNotQuarantine) {
    Runner runner;
    runner.allocate_callable();
    Attempt active(runner, 0, 10);
    ASSERT_EQ(active.prepare(), 0);
    ASSERT_EQ(simpler_launch_run(&runner, active.storage), 0);
    runner.drain_result = DrainOutcome::device_complete(-59);
    EXPECT_EQ(simpler_wait_run(&runner, active.storage), -59);
    EXPECT_TRUE(runner.accepts_new_run());
    EXPECT_EQ(runner.quarantine_calls, 0);
    EXPECT_EQ(simpler_finalize_run(&runner, active.storage), -59);
    EXPECT_TRUE(active.empty());
    EXPECT_TRUE(runner.accepts_new_run());
    EXPECT_EQ(runner.quarantine_calls, 0);
    SimplerWorkspaceReport report{};
    ASSERT_TRUE(runner.workspace_report(&report));
    EXPECT_EQ(report.quarantined_blocks, 0u);
    const size_t bytes = control().bytes;
    EXPECT_EQ(simpler_unregister_callable(&runner, 1), 0);
    EXPECT_FALSE(runner.has_callable(1));
    EXPECT_EQ(control().bytes, bytes - 64);
}

TEST_F(NativeAdmissionTest, UndrainedFinalizationRetainsCallableCodeAndTables) {
    for (int failure = 0; failure < 4; ++failure) {
        SCOPED_TRACE(failure);
        EXPECT_EXIT(
            {
                Runner runner;
                runner.allocate_callable();
                Attempt active(runner, 0, 10);
                if (active.prepare() != 0) std::_Exit(1);
                if (failure == 3) {
                    runner.partial_launch = true;
                    runner.launch_refusal_rc = -4321;
                }
                if (simpler_launch_run(&runner, active.storage) != (failure == 3 ? -4321 : 0)) std::_Exit(1);
                if (failure < 2) control().attach_rc = -77;
                else if (failure == 2) runner.drain_throws = true;
                const int rc = failure < 2 ? -77 : failure == 2 ? PTO_RUNTIME_ERR_INTERNAL : -4321;
                if (failure == 0 && simpler_wait_run(&runner, active.storage) != rc) std::_Exit(2);
                if (simpler_finalize_run(&runner, active.storage) != rc) std::_Exit(3);
                control().attach_rc = 0;
                if (runner.accepts_new_run()) std::_Exit(4);
                if (failure < 3 && (!runner.native_runs_outstanding() || active.empty() || !runner.holds_epoch(10) ||
                                    runner.reservations() != 1))
                    std::_Exit(8);
                if (failure == 3 && (runner.native_runs_outstanding() || !active.empty())) std::_Exit(9);
                if (failure == 3 && (runner.recovery_calls != 1 || runner.quarantine_calls != 0)) std::_Exit(4);
                const size_t bytes = control().bytes;
                const int refusal = failure < 3 ? PTO_RUNTIME_ERR_INTERNAL : PTO_RUNTIME_ERR_INVALID_STATE;
                if (simpler_unregister_callable(&runner, 1) != refusal) std::_Exit(5);
                if (runner.unregister_callable(1) != PTO_RUNTIME_ERR_INVALID_STATE) std::_Exit(6);
                if (!runner.has_callable(1) || runner.callable_refcount() != 1 || control().bytes != bytes)
                    std::_Exit(7);
                std::_Exit(0);
            },
            ::testing::ExitedWithCode(0), ""
        );
    }
}

TEST_F(NativeAdmissionTest, DeviceHealthSuspicionDoesNotBlockProvedCallableRelease) {
    Runner runner;
    runner.allocate_callable();
    Attempt completed(runner, 0, 10);
    ASSERT_EQ(simpler_run(&runner, completed.storage, 1, &completed.args, &completed.config, &completed.descriptor), 0);
    runner.note_device_suspicion();
    ASSERT_FALSE(runner.accepts_new_run());
    ASSERT_TRUE(runner.can_accept_run());
    const size_t bytes = control().bytes;
    EXPECT_EQ(simpler_unregister_callable(&runner, 1), 0);
    EXPECT_FALSE(runner.has_callable(1));
    EXPECT_EQ(control().bytes, bytes - 64);
}

TEST_F(NativeAdmissionTest, ActualStagingAdapterTrimsIdleStorageThenRetriesOnce) {
    Runner runner;
    control().ceiling = 50 * kUnit;
    ASSERT_EQ(staging(runner, 0, 10, 20 * kUnit), 0);
    launched(runner, 0, 10);
    runner.retire(0, 10);
    const auto calls = control().allocations;
    ASSERT_EQ(staging(runner, 1, 20, 40 * kUnit), 0);
    EXPECT_EQ(control().allocations - calls, 2u);
    EXPECT_EQ(runner.retained_temp_addr(0), 0u);
    EXPECT_EQ(runner.charged(), 40 * kUnit);
    EXPECT_EQ(runner.abort_run_plan(1, 20), 0);
}

TEST_F(NativeAdmissionTest, FortyWithThirtyFreeWaitsOnlyForItsNamedTwentyByteUnitsConsumer) {
    Runner runner;
    control().ceiling = 50 * kUnit;
    ASSERT_EQ(staging(runner, 0, 10, 20 * kUnit), 0);
    launched(runner, 0, 10);
    ASSERT_EQ(staging(runner, 1, 20, 40 * kUnit), SIMPLER_RUNTIME_ERR_RESOURCE_WAIT);
    EXPECT_EQ(runner.plan(1).staging_snapshot, std::vector<uint64_t>({10}));
    const auto calls = control().allocations;
    runner.retire(2, 99);
    EXPECT_EQ(staging(runner, 1, 21, 40 * kUnit), SIMPLER_RUNTIME_ERR_RESOURCE_WAIT);
    EXPECT_EQ(control().allocations, calls);
    runner.retire(0, 10);
    EXPECT_EQ(staging(runner, 1, 22, 40 * kUnit), 0);
    EXPECT_EQ(runner.charged(), 40 * kUnit);
    EXPECT_EQ(runner.abort_run_plan(1, 22), 0);
}

TEST_F(NativeAdmissionTest, ExhaustedConsumersCannotRecaptureALaterRun) {
    Runner runner;
    control().ceiling = 50 * kUnit;
    ASSERT_EQ(staging(runner, 0, 10, 20 * kUnit), 0);
    launched(runner, 0, 10);
    ASSERT_EQ(staging(runner, 1, 20, 60 * kUnit), SIMPLER_RUNTIME_ERR_RESOURCE_WAIT);
    runner.retire(0, 10);
    EXPECT_EQ(staging(runner, 1, 21, 60 * kUnit), SIMPLER_RUNTIME_ERR_RESOURCE_UNAVAILABLE);
    EXPECT_EQ(runner.abort_run_plan(1, 21), 0);
}

TEST_F(NativeAdmissionTest, NoncapacityAdmissionFailureNeverParks) {
    Runner runner;
    runner.stop_serving();
    EXPECT_EQ(staging(runner, 0, 10, 20 * kUnit), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(runner.plan(0).parked, AutoAdmissionState::Park::None);
    EXPECT_EQ(control().allocations, 0u);
    EXPECT_EQ(runner.abort_run_plan(0, 10), 0);
}

TEST_F(NativeAdmissionTest, MultiregionFailureKeepsEveryPublishedBackingAndRollsBackOnlyNewClaims) {
    Runner runner;
    {
        DeviceRunnerBase::WorkspacePlanScope scope(0, 10);
        ASSERT_EQ(runner.setup_static_arena(0, 2 * kUnit, 2 * kUnit, 2 * kUnit), 0);
        runner.note_run_resources_proven(0, 10);
    }
    runner.note_workspace_run_fact(NativeRunIdentity{10, 10, 10, 0}, Fact::NoDeviceSubmission);
    runner.retire(0, 10);
    const std::vector<void *> old{
        runner.arena_region_backing(0, 0), runner.arena_region_backing(0, 1), runner.arena_region_backing(0, 2)
    };
    const size_t charged = runner.charged();
    control().ceiling = control().bytes + 4 * kUnit + DeviceArena::kDefaultBaseAlign - 1;
    for (uint64_t epoch : {20u, 30u, 40u}) {
        const auto staged = control().successful_allocations;
        {
            DeviceRunnerBase::WorkspacePlanScope scope(0, epoch);
            EXPECT_EQ(
                runner.setup_static_arena(0, 4 * kUnit, 4 * kUnit, 4 * kUnit), SIMPLER_RUNTIME_ERR_RESOURCE_UNAVAILABLE
            );
        }
        EXPECT_EQ(control().successful_allocations - staged, 2u);  // first region staged on both attempts
        for (size_t i = 0; i < old.size(); ++i)
            EXPECT_EQ(runner.arena_region_backing(0, i), old[i]);
        // Abort may follow attach failure, so it relinquishes claims without
        // freeing on an unproven thread. The final staged region stays charged.
        EXPECT_EQ(runner.abort_run_plan(0, epoch), 0);
        EXPECT_EQ(runner.charged(), charged + 4 * kUnit + DeviceArena::kDefaultBaseAlign - 1);
        const auto allocations = control().allocations;
        Attempt next(runner, 0, epoch + 1);
        next.config.aicpu_thread_num = 1;
        // The real C API must reclaim after attach and before launch-shape
        // validation or allocation, even when this next prepare also fails.
        EXPECT_EQ(next.prepare(), PTO_RUNTIME_ERR_INTERNAL);
        EXPECT_TRUE(next.consume_release());
        EXPECT_TRUE(next.empty());
        EXPECT_EQ(control().allocations, allocations);
        EXPECT_EQ(runner.charged(), charged);
        EXPECT_EQ(control().bytes, charged);
        for (size_t i = 0; i < old.size(); ++i)
            EXPECT_EQ(runner.arena_region_backing(0, i), old[i]);
    }
    // No further prepare: close is the final attached reclamation boundary.
    {
        DeviceRunnerBase::WorkspacePlanScope scope(0, 50);
        EXPECT_EQ(
            runner.setup_static_arena(0, 4 * kUnit, 4 * kUnit, 4 * kUnit), SIMPLER_RUNTIME_ERR_RESOURCE_UNAVAILABLE
        );
    }
    EXPECT_EQ(runner.abort_run_plan(0, 50), 0);
    EXPECT_GT(runner.charged(), charged);
    EXPECT_EQ(runner.finalize(), 0);
    EXPECT_EQ(control().bytes, 0u);
}

TEST_F(NativeAdmissionTest, NativePrepareParksAndDestroysTheAttemptBeforeAnEarlyRetryAttachFailure) {
    Runner runner;
    ASSERT_EQ(staging(runner, 0, 10, 20 * kUnit), 0);
    launched(runner, 0, 10);
    control().ceiling = control().bytes;
    Attempt candidate(runner, 1, 20, 40 * kUnit);
    ASSERT_EQ(candidate.prepare(), SIMPLER_RUNTIME_ERR_RESOURCE_WAIT);
    EXPECT_TRUE(candidate.empty());
    EXPECT_EQ(runner.reservations(), 0u);
    EXPECT_TRUE(runner.plan(1).holds_token);
    EXPECT_EQ(runner.plan(1).staging_snapshot, std::vector<uint64_t>({10}));
    const auto allocations = control().allocations;
    candidate.descriptor.run_epoch = 21;
    EXPECT_EQ(candidate.prepare(), SIMPLER_RUNTIME_ERR_RESOURCE_WAIT);
    EXPECT_EQ(control().allocations, allocations);
    control().attach_rc = -77;
    candidate.descriptor.run_epoch = 22;
    EXPECT_EQ(candidate.prepare(), -77);
    control().attach_rc = 0;
    EXPECT_TRUE(candidate.consume_release());
    EXPECT_TRUE(candidate.empty());
    EXPECT_EQ(runner.reservations(), 0u);
    EXPECT_EQ(runner.plan(1).epoch, 0u);
    EXPECT_FALSE(runner.plan(1).holds_token);
    runner.retire(0, 10);
}

TEST_F(NativeAdmissionTest, NativePrepareResumesAfterItsLaunchedConsumerRetires) {
    Runner runner;
    ASSERT_EQ(staging(runner, 0, 10, 20 * kUnit), 0);
    launched(runner, 0, 10);
    control().ceiling = control().bytes;
    Attempt candidate(runner, 1, 20);
    ASSERT_EQ(candidate.prepare(), SIMPLER_RUNTIME_ERR_RESOURCE_WAIT);
    runner.retire(0, 10);
    control().ceiling = std::numeric_limits<size_t>::max();
    candidate.descriptor.run_epoch = 21;
    ASSERT_EQ(candidate.prepare(), 0);
    EXPECT_EQ(runner.reservations(), 1u);
    EXPECT_EQ(runner.plan(1).epoch, 0u);
    ASSERT_EQ(simpler_finalize_run(&runner, candidate.storage), 0);
    EXPECT_TRUE(candidate.empty());
    EXPECT_EQ(runner.reservations(), 0u);
}

TEST_F(NativeAdmissionTest, NativeAbortAndCloseReleaseParkedPlanAndGrowthPermission) {
    Runner runner;
    ASSERT_EQ(staging(runner, 0, 10, 20 * kUnit), 0);
    launched(runner, 0, 10);
    control().ceiling = control().bytes;
    Attempt candidate(runner, 1, 20);
    ASSERT_EQ(candidate.prepare(), SIMPLER_RUNTIME_ERR_RESOURCE_WAIT);
    EXPECT_EQ(simpler_abort_run_plan_ctx(&runner, 1, 19), 0);
    EXPECT_EQ(runner.plan(1).epoch, 20u);  // stale caller cannot settle this plan
    EXPECT_EQ(simpler_abort_run_plan_ctx(&runner, 1, 20), 0);
    EXPECT_FALSE(runner.plan(1).holds_token);
    EXPECT_EQ(runner.reservations(), 0u);
    runner.retire(0, 10);
    EXPECT_EQ(finalize_device(&runner), 0);
    EXPECT_EQ(runner.charged(), 0u);
}
TEST_F(NativeAdmissionTest, CloseOwnsAParkedAttemptBeforeItsCallback) {
    Runner runner;
    ASSERT_EQ(staging(runner, 0, 10, 20 * kUnit), 0);
    launched(runner, 0, 10);
    control().ceiling = control().bytes;
    Attempt candidate(runner, 1, 20);
    ASSERT_EQ(candidate.prepare(), SIMPLER_RUNTIME_ERR_RESOURCE_WAIT);
    EXPECT_EQ(callbacks, 0);
    runner.retire(0, 10);
    ASSERT_EQ(finalize_device(&runner), 0);
    EXPECT_TRUE(candidate.empty());
    EXPECT_EQ(runner.get_prepare_plan(1), nullptr);
    EXPECT_EQ(runner.charged(), 0u);
    EXPECT_FALSE(runner.plan(1).holds_token);
    EXPECT_EQ(runner.plan(1).epoch, 0u);
    EXPECT_EQ(runner.reservations(), 0u);
}

TEST_F(NativeAdmissionTest, AWaitingPlanCannotReleaseBorrowsWhileItsNativeEpochStillOwnsExecution) {
    Runner runner;
    ASSERT_EQ(staging(runner, 0, 10, 20 * kUnit), 0);
    launched(runner, 0, 10);
    control().ceiling = control().bytes;
    Attempt candidate(runner, 1, 20);
    ASSERT_EQ(candidate.prepare(), SIMPLER_RUNTIME_ERR_RESOURCE_WAIT);
    control().ceiling = std::numeric_limits<size_t>::max();
    void *caller = runner.allocate_caller_buffer(64);
    ASSERT_NE(caller, nullptr);
    CallerDeviceBuffers::Span span{reinterpret_cast<uint64_t>(caller), 64};
    ASSERT_TRUE(runner.borrow_caller_buffers(20, &span, 1));
    // Inject only stop evidence: a native owner still names this epoch. The
    // abort guard and caller allocation table are the production implementations.
    int owner = 0;
    NativeRunIdentity identity{20, 20, 20, 1};
    LaunchPermit permit;
    ASSERT_TRUE(runner.try_reserve_native_run(&owner, 1, candidate.descriptor.arena_bank, true));
    ASSERT_TRUE(runner.try_acquire_native_run(&owner, identity, &permit));
    const auto charged = runner.charged();
    EXPECT_NE(finalize_device(&runner), 0);
    EXPECT_EQ(runner.charged(), charged);
    EXPECT_NE(simpler_abort_run_plan_ctx(&runner, 1, 20), 0);
    EXPECT_NE(runner.free_caller_buffer(caller), 0);
    EXPECT_TRUE(runner.plan(1).holds_token);
    runner.release_native_run(&owner);
    runner.release_native_run_reservation(&owner);
    ASSERT_EQ(simpler_abort_run_plan_ctx(&runner, 1, 20), 0);
    runner.release_caller_buffers(20, false);
    EXPECT_EQ(runner.free_caller_buffer(caller), 0);
    runner.retire(0, 10);
}

namespace {
void check_finalize_prefix_allocation_failure(bool launched, bool delivery_entry) {
    Runner runner;
    Attempt run(runner, 0, 8100);
    const std::string prefix(160, 'p');
    std::memcpy(run.config.output_prefix, prefix.c_str(), prefix.size() + 1);
    volatile int32_t accepted = -9;
    run.descriptor.accepted_state = &accepted;
    run.descriptor.accepted_value = 7;
    ASSERT_EQ(run.prepare(), 0);
    if (launched) ASSERT_EQ(simpler_launch_run(&runner, run.storage), 0);
    auto *state = static_cast<NativeRunContext<DeviceRunnerBase> *>(run.storage);
    const auto phase = state->phase.load();
    const int completion = state->completion_rc;
    const auto *prepared_owner = state->prepared_execution.get();
    const auto *active_owner = state->active_execution.get();
    const int32_t acceptance = accepted;
    const auto payload = run.bytes;
    const auto charged = runner.charged();
    const auto reservations = runner.reservations();
    const int launches = runner.launch_calls;
    const int drains = runner.drain_calls;
    const auto allocated = control().bytes;
    std::vector<unsigned char> carrier(get_runtime_size());
    std::memcpy(carrier.data(), run.storage, carrier.size());

    const auto finalize = [&] {
        return delivery_entry ? simpler_finalize_run_with_delivery(&runner, run.storage, 1) :
                                simpler_finalize_run(&runner, run.storage);
    };
    refused_host_allocation_bytes = refused_host_allocation_count = 0;
    refuse_next_host_allocation = true;
    bool escaped = false;
    int rc = PTO_RUNTIME_ERR_INTERNAL;
    try {
        rc = finalize();
    } catch (const std::bad_alloc &) {
        escaped = true;
    }
    const bool fault_unconsumed = std::exchange(refuse_next_host_allocation, false);
    std::printf(
        "FINALIZE_PREFIX backend=onboard launched=%d delivery_entry=%d consumed=%zu escaped=%d rc=%d bytes=%zu\n",
        launched, delivery_entry, refused_host_allocation_count, escaped, rc, refused_host_allocation_bytes
    );
    EXPECT_FALSE(fault_unconsumed);
    EXPECT_EQ(refused_host_allocation_count, 1u);
    EXPECT_GE(refused_host_allocation_bytes, prefix.size() + 1);
    EXPECT_FALSE(escaped);
    EXPECT_EQ(rc, PTO_RUNTIME_ERR_INTERNAL);
    ASSERT_EQ(state->magic, NativeRunContext<DeviceRunnerBase>::kMagic);
    EXPECT_EQ(std::memcmp(carrier.data(), run.storage, carrier.size()), 0);
    EXPECT_EQ(state->phase.load(), phase);
    EXPECT_EQ(state->completion_rc, completion);
    EXPECT_EQ(state->prepared_execution.get(), prepared_owner);
    EXPECT_EQ(state->active_execution.get(), active_owner);
    EXPECT_EQ(accepted, acceptance);
    EXPECT_EQ(run.bytes, payload);
    EXPECT_EQ(runner.charged(), charged);
    EXPECT_EQ(runner.reservations(), reservations);
    EXPECT_EQ(runner.launch_calls, launches);
    EXPECT_EQ(runner.drain_calls, drains);
    EXPECT_EQ(control().bytes, allocated);

    ASSERT_EQ(finalize(), 0);
    EXPECT_TRUE(run.empty());
    EXPECT_EQ(runner.reservations(), 0u);
    EXPECT_FALSE(runner.holds_epoch(8100));
    EXPECT_EQ(runner.drain_calls, drains + (launched ? 1 : 0));
    EXPECT_EQ(run.bytes, payload);
    Attempt successor(runner, 0, 8101);
    EXPECT_EQ(simpler_run(&runner, successor.storage, 1, &successor.args, &successor.config, &successor.descriptor), 0);
    EXPECT_TRUE(successor.empty());
    EXPECT_EQ(runner.reservations(), 0u);
}
}  // namespace

TEST_F(NativeAdmissionTest, FinalizePrefixAllocationFailureKeepsPreparedOwnerRetryable) {
    for (bool delivery_entry : {false, true}) {
        SCOPED_TRACE(delivery_entry);
        check_finalize_prefix_allocation_failure(false, delivery_entry);
    }
}

TEST_F(NativeAdmissionTest, FinalizePrefixAllocationFailureKeepsLaunchedOwnerRetryable) {
    for (bool delivery_entry : {false, true}) {
        SCOPED_TRACE(delivery_entry);
        check_finalize_prefix_allocation_failure(true, delivery_entry);
    }
}

class SharedHeapRunner : public FencedCompletionRunner {
public:
    explicit SharedHeapRunner(bool automatic = false) :
        FencedCompletionRunner(true, automatic) {
        static int binary;
        aicore_bin_handle_ = &binary;
        EXPECT_EQ(execution_mode_latch().latch(SIMPLER_MODE_PROGRAM), 0);
    }
    int reclaim_retired_images() { return reclaim_prepared_storage(); }
    std::function<void()> before_launch;
    bool refuse_capacity{false};
    LaunchTransactionResult launch_execution_impl(PreparedExecution &prepared, LaunchPermit permit) override {
        if (before_launch) before_launch();
        if (refuse_capacity) {
            return exact_launch_transaction(
                prepared.identity, std::move(permit),
                [](LaunchProgressSink &) {
                    return SIMPLER_RUNTIME_ERR_DIAGNOSTIC_CAPACITY_FULL;
                },
                [](LaunchProgressSink &) {
                    ADD_FAILURE() << "unexpected CPU submit";
                    return 0;
                }
            );
        }
        auto result = FencedCompletionRunner::launch_execution_impl(prepared, std::move(permit));
        if (result.rc == 0) EXPECT_EQ(queue_caller_exit(prepared), 0);
        return result;
    }
};

static void finish_heap_attempt(SharedHeapRunner &runner, Attempt &attempt) {
    EXPECT_EQ(simpler_finalize_run(&runner, attempt.storage), 0);
    if (!attempt.empty()) EXPECT_TRUE(attempt.consume_release());
    EXPECT_TRUE(attempt.empty());
}

TEST_F(NativeAdmissionTest, OrderedGraphHeapSharesBackingButKeepsImagesSeparate) {
    if (arena_bank(1) == 0) GTEST_SKIP() << "per-run HBG arena required";
    SharedHeapRunner runner;
    Attempt first(runner, 0, 9300), second(runner, 1, 9301);
    ASSERT_EQ(first.prepare(), 0);
    const auto heap = runner.arena_bank_gm_heap_base(0);
    ASSERT_NE(heap, 0u);
    void *image = runner.arena_region_backing(0, 2);
    ASSERT_NE(image, nullptr);
    ASSERT_EQ(simpler_launch_run(&runner, first.storage), 0);
    ASSERT_EQ(second.prepare(), 0);
    EXPECT_EQ(runner.arena_bank_gm_heap_base(1), heap);
    EXPECT_NE(runner.arena_region_backing(1, 2), image);
    finish_heap_attempt(runner, first);
    // Reclaiming the predecessor's run cannot destroy the successor's lease.
    EXPECT_EQ(runner.arena_bank_gm_heap_base(1), heap);
    finish_heap_attempt(runner, second);
}

TEST_F(NativeAdmissionTest, AutomaticAdmissionKeepsIndependentGraphHeaps) {
    if (arena_bank(1) == 0) GTEST_SKIP() << "per-run HBG arena required";
    SharedHeapRunner runner(true);
    Attempt first(runner, 0, 9310), second(runner, 1, 9311);
    ASSERT_EQ(first.prepare(), 0);
    ASSERT_EQ(simpler_launch_run(&runner, first.storage), 0);
    ASSERT_EQ(second.prepare(), 0);
    EXPECT_NE(runner.arena_bank_gm_heap_base(1), runner.arena_bank_gm_heap_base(0));
    finish_heap_attempt(runner, first);
    finish_heap_attempt(runner, second);
}

TEST_F(NativeAdmissionTest, SubmittingGraphHeapIsNotBorrowedAndRefusedRunCanRetain) {
    if (arena_bank(1) == 0) GTEST_SKIP() << "per-run HBG arena required";
    SharedHeapRunner runner;
    Attempt first(runner, 0, 9320), successor(runner, 1, 9321);
    ASSERT_EQ(first.prepare(), 0);
    const auto heap = runner.arena_bank_gm_heap_base(0);
    runner.refuse_capacity = true;
    runner.before_launch = [&] {
        auto preparation = std::async(std::launch::async, [&] {
            return successor.prepare();
        });
        ASSERT_EQ(preparation.get(), 0);
        EXPECT_NE(runner.arena_bank_gm_heap_base(1), heap);
    };
    ASSERT_EQ(simpler_launch_run(&runner, first.storage), SIMPLER_RUNTIME_ERR_DIAGNOSTIC_CAPACITY_FULL);
    runner.before_launch = {};
    auto *state = static_cast<NativeRunContext<DeviceRunnerBase> *>(first.storage);
    ASSERT_EQ(state->phase.load(), NativeRunPhase::Prepared);
    int caller = 0;
    RuntimeHandle reference = nullptr;
    ASSERT_EQ(simpler_retain_run_resources(&runner, first.storage, &caller, &reference), 0);
    ASSERT_NE(reference, nullptr);
    EXPECT_EQ(runner.arena_bank_gm_heap_base(0), heap);
    finish_heap_attempt(runner, first);
    finish_heap_attempt(runner, successor);
    EXPECT_EQ(simpler_release_run_resources(&runner, reference), 0);
}

TEST_F(NativeAdmissionTest, RetainedGraphHeapCannotBecomeSharedAfterLaunch) {
    if (arena_bank(1) == 0) GTEST_SKIP() << "per-run HBG arena required";
    SharedHeapRunner runner;
    Attempt first(runner, 0, 9330), successor(runner, 1, 9331);
    ASSERT_EQ(first.prepare(), 0);
    int caller = 0;
    RuntimeHandle reference = nullptr;
    ASSERT_EQ(simpler_retain_run_resources(&runner, first.storage, &caller, &reference), 0);
    ASSERT_EQ(simpler_launch_run(&runner, first.storage), 0);
    ASSERT_EQ(successor.prepare(), 0);
    EXPECT_NE(runner.arena_bank_gm_heap_base(0), runner.arena_bank_gm_heap_base(1));
    finish_heap_attempt(runner, first);
    finish_heap_attempt(runner, successor);
    EXPECT_EQ(simpler_release_run_resources(&runner, reference), 0);
}

TEST_F(NativeAdmissionTest, GraphHeapGrowthPreservesOldBackingAndSmallerSuccessorReusesNewGeneration) {
    if (arena_bank(1) == 0) GTEST_SKIP() << "per-run HBG arena required";
    SharedHeapRunner runner;
    Attempt first(runner, 0, 9340), larger(runner, 1, 9341), smaller(runner, 2, 9342);
    ASSERT_EQ(first.prepare(), 0);
    const auto old_heap = runner.arena_bank_gm_heap_base(0);
    ASSERT_NE(old_heap, 0u);
    std::memset(reinterpret_cast<void *>(old_heap), 0x5a, 16);
    ASSERT_EQ(simpler_launch_run(&runner, first.storage), 0);
    graph_heap_elements = 32768;
    ASSERT_EQ(larger.prepare(), 0);
    const auto new_heap = runner.arena_bank_gm_heap_base(1);
    EXPECT_NE(new_heap, old_heap);
    for (size_t i = 0; i != 16; ++i)
        EXPECT_EQ(reinterpret_cast<uint8_t *>(old_heap)[i], 0x5a);
    finish_heap_attempt(runner, first);
    ASSERT_EQ(simpler_launch_run(&runner, larger.storage), 0);
    graph_heap_elements = 16;
    ASSERT_EQ(smaller.prepare(), 0);
    EXPECT_EQ(runner.arena_bank_gm_heap_base(2), new_heap);
    finish_heap_attempt(runner, larger);
    finish_heap_attempt(runner, smaller);
}

TEST_F(NativeAdmissionTest, PreparedGraphHeapDoesNotBypassExistingAdmission) {
    if (arena_bank(1) == 0) GTEST_SKIP() << "per-run HBG arena required";
    SharedHeapRunner runner;
    Attempt first(runner, 0, 9350), successor(runner, 1, 9351);
    ASSERT_EQ(first.prepare(), 0);
    const auto allocations = control().allocations;
    EXPECT_NE(successor.prepare(), 0);
    EXPECT_EQ(control().allocations, allocations);
    int caller = 0;
    RuntimeHandle reference = nullptr;
    ASSERT_EQ(simpler_retain_run_resources(&runner, first.storage, &caller, &reference), 0);
    finish_heap_attempt(runner, first);
    EXPECT_EQ(simpler_release_run_resources(&runner, reference), 0);
}

TEST_F(NativeAdmissionTest, RetiringOldImageBankDoesNotReleaseAnotherRunsSharedHeap) {
    if (arena_bank(1) == 0) GTEST_SKIP() << "per-run HBG arena required";
    SharedHeapRunner runner;
    Attempt first(runner, 0, 9360), second(runner, 1, 9361), reused_slot(runner, 0, 9362);
    ASSERT_EQ(first.prepare(), 0);
    const auto heap = runner.arena_bank_gm_heap_base(0);
    ASSERT_EQ(simpler_launch_run(&runner, first.storage), 0);
    ASSERT_EQ(second.prepare(), 0);
    ASSERT_EQ(runner.arena_bank_gm_heap_base(1), heap);
    ASSERT_EQ(simpler_wait_run_completion(&runner, first.storage), 0);
    ASSERT_EQ(simpler_launch_run(&runner, second.storage), 0);
    ASSERT_EQ(reused_slot.prepare(), 0);
    ASSERT_EQ(runner.arena_bank_gm_heap_base(0), heap);
    void *raw = runner.arena_region_backing(1, 0);
    std::memset(reinterpret_cast<void *>(heap), 0x69, 16);
    control().selected_free = raw;
    control().selected_free_attempts = 0;
    finish_heap_attempt(runner, first);
    ASSERT_EQ(runner.reclaim_retired_images(), 0);
    ASSERT_EQ(runner.reclaim_workspace_obsolete(), 0);
    EXPECT_EQ(control().selected_free_attempts, 0u);
    for (size_t i = 0; i != 16; ++i)
        EXPECT_EQ(reinterpret_cast<uint8_t *>(heap)[i], 0x69);
    finish_heap_attempt(runner, second);
    finish_heap_attempt(runner, reused_slot);
    control().selected_free = nullptr;
}
}  // namespace native_admission_test

namespace native_admission_test {
class FrameworkCompletionRunner : public FencedCompletionRunner {
public:
    void add_host_control_signature() { callables_.at(1).signature.push_back(ArgDirection::IN); }
    void qualify_registration_transport() {
        aicpu_runtime_registered_ = true;
        aicpu_initialized_ = true;
        control().allow_registration = true;
    }
    LaunchTransactionResult launch_execution_impl(PreparedExecution &prepared, LaunchPermit permit) override {
        auto result = FencedCompletionRunner::launch_execution_impl(prepared, std::move(permit));
        if (result.rc == 0) result.rc = queue_caller_exit(prepared);
        return result;
    }
};

TEST_F(NativeAdmissionTest, FrameworkTransferEndsOnlyThePhysicalCarrierWithoutObservingItsStreams) {
    FrameworkCompletionRunner runner;
    ASSERT_EQ(initialize_capture_runner(runner), 0);
    Attempt attempt(runner, 0, 10100);
    attempt.args.tensor(0).address_space = AddressSpace::DEVICE;
    attempt.args.tensor(0).transfer = TensorTransfer::NONE;
    ASSERT_EQ(attempt.prepare(), 0);
    int caller;
    ASSERT_EQ(simpler_launch_run_on_stream(&runner, attempt.storage, &caller, nullptr), 0);
    ASSERT_EQ(simpler_poll_run(&runner, attempt.storage), SIMPLER_NATIVE_RUN_POLL_COMPLETE);
    auto *state = static_cast<NativeRunContext<DeviceRunnerBase> *>(attempt.storage);
    ASSERT_TRUE(state->execution_retired);
    ASSERT_FALSE(state->result_observed);
    control().observed_streams.clear();
    control().stream_sync_rc = -901;
    ASSERT_EQ(simpler_transfer_completed_kernel_run(&runner, attempt.storage), 0);
    EXPECT_TRUE(attempt.empty());
    EXPECT_TRUE(control().observed_streams.empty());
    EXPECT_EQ(runner.reservations(), 0u);
    EXPECT_TRUE(runner.native_runs_outstanding());
    EXPECT_TRUE(runner.fresh_callable_addition_allowed());
    EXPECT_NE(simpler_unregister_callable(&runner, 1), 0);
    int all = 77;
    EXPECT_EQ(simpler_observe_framework_submissions(&runner, &all), -901);
    EXPECT_EQ(all, 0);
    EXPECT_EQ(control().observed_streams.size(), 1u);  // original pair helper short-circuits
    EXPECT_TRUE(runner.native_runs_outstanding());
    EXPECT_FALSE(runner.fresh_callable_addition_allowed());  // original context is poisoned
    control().stream_sync_rc = 0;
    control().observed_streams.clear();
    EXPECT_EQ(simpler_observe_framework_submissions(&runner, &all), -901);
    EXPECT_EQ(all, 1);  // cleanup fact does not replace the first semantic error
    EXPECT_EQ(control().observed_streams.size(), 2u);
    EXPECT_FALSE(runner.native_runs_outstanding());
}

TEST_F(NativeAdmissionTest, FrameworkDebtPermitsAllThreeFreshCallableEntriesWithoutObservingTheOldPair) {
    FrameworkCompletionRunner runner;
    ASSERT_EQ(initialize_capture_runner(runner), 0);
    runner.qualify_registration_transport();
    std::ifstream input(NATIVE_OBSERVATION_ORCH_PATH, std::ios::binary);
    const std::vector<uint8_t> binary{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    ASSERT_FALSE(binary.empty());
    const ArgDirection signature[] = {ArgDirection::IN};
    const auto original = make_chip_callable(
        signature, 1, "framework_observation_orch", binary.data(), binary.size(), nullptr, nullptr, 0, ""
    );
    const auto image = make_chip_callable(
        signature, 1, "framework_observation_orch_b", binary.data(), binary.size(), nullptr, nullptr, 0, ""
    );
    ASSERT_EQ(simpler_unregister_callable(&runner, 1), 0);
    ASSERT_EQ(simpler_kernel_mode_prepare_callable(&runner, 1, original.data(), original.size()), 0);
    const auto original_hash = runner.callable_hash(1);
    const size_t initial_registrations = control().registered_ids.size();
    Attempt attempt(runner, 0, 10103);
    attempt.args.tensor(0).address_space = AddressSpace::DEVICE;
    attempt.args.tensor(0).transfer = TensorTransfer::NONE;
    ASSERT_EQ(attempt.prepare(), 0);
    int caller;
    ASSERT_EQ(simpler_launch_run_on_stream(&runner, attempt.storage, &caller, nullptr), 0);
    ASSERT_EQ(simpler_poll_run(&runner, attempt.storage), SIMPLER_NATIVE_RUN_POLL_COMPLETE);
    ASSERT_EQ(simpler_transfer_completed_kernel_run(&runner, attempt.storage), 0);
    control().observed_streams.clear();
    EXPECT_EQ(simpler_register_callable(&runner, 2, image.data()), 0);
    EXPECT_TRUE(runner.has_callable(2));
    EXPECT_EQ(simpler_kernel_mode_prepare_callable(&runner, 3, image.data(), image.size()), 0);
    EXPECT_TRUE(runner.has_callable(3));
    int32_t auto_id = -1;
    EXPECT_EQ(simpler_register_callable_v2(&runner, image.data(), image.size(), &auto_id), 0);
    EXPECT_GE(auto_id, 0);
    EXPECT_TRUE(runner.has_callable(auto_id));
    EXPECT_TRUE(runner.native_runs_outstanding());
    for (void *stream : control().observed_streams)
        EXPECT_EQ(stream, runner.kernel_execution_state().hidden_stream(KernelStreamKind::Aicpu));
    EXPECT_EQ(control().registered_ids.size() - initial_registrations, primary_entry_points() == nullptr ? 3u : 0u);
    EXPECT_EQ(runner.callable_hash(1), original_hash);
    EXPECT_NE(simpler_unregister_callable(&runner, 1), 0);
    EXPECT_NE(simpler_register_callable(&runner, 1, image.data()), 0);  // no reuse relaxation
    int all = 0;
    EXPECT_EQ(simpler_observe_framework_submissions(&runner, &all), 0);
    EXPECT_EQ(all, 1);
    EXPECT_FALSE(runner.native_runs_outstanding());
}

TEST_F(NativeAdmissionTest, FrameworkTransferRefusesBeforePhysicalProofAndPreservesTheOriginalOwner) {
    FrameworkCompletionRunner runner;
    ASSERT_EQ(initialize_capture_runner(runner), 0);
    Attempt attempt(runner, 0, 10101);
    attempt.args.tensor(0).address_space = AddressSpace::DEVICE;
    attempt.args.tensor(0).transfer = TensorTransfer::NONE;
    ASSERT_EQ(attempt.prepare(), 0);
    EXPECT_NE(simpler_transfer_completed_kernel_run(&runner, attempt.storage), 0);
    EXPECT_FALSE(attempt.empty());
    EXPECT_TRUE(runner.native_runs_outstanding());
    EXPECT_FALSE(runner.fresh_callable_addition_allowed());
    ASSERT_EQ(simpler_finalize_run(&runner, attempt.storage), 0);
    EXPECT_FALSE(runner.native_runs_outstanding());
}

TEST_F(NativeAdmissionTest, FrameworkTransferRefusesHostInputAtItsRuntimeBoundary) {
    FrameworkCompletionRunner runner;
    ASSERT_EQ(initialize_capture_runner(runner), 0);
    Attempt attempt(runner, 0, 10102);
    attempt.args.tensor(0).transfer = TensorTransfer::NONE;
    if (std::string_view(SIMPLER_RUNTIME_NAME) == "tensormap_and_ringbuffer") {
        EXPECT_EQ(attempt.prepare(), PTO_RUNTIME_ERR_UNSUPPORTED);
        EXPECT_TRUE(attempt.consume_release());
        EXPECT_TRUE(attempt.empty());
        EXPECT_FALSE(runner.native_runs_outstanding());
        return;
    }

    const auto host_control = attempt.args.tensor(0);
    attempt.args.tensor(0).address_space = AddressSpace::DEVICE;
    attempt.args.add_tensor(host_control);
    runner.add_host_control_signature();
    ASSERT_EQ(attempt.prepare(), 0);
    int caller;
    ASSERT_EQ(simpler_launch_run_on_stream(&runner, attempt.storage, &caller, nullptr), 0);
    ASSERT_EQ(simpler_poll_run(&runner, attempt.storage), SIMPLER_NATIVE_RUN_POLL_COMPLETE);
    const auto *state = static_cast<NativeRunContext<DeviceRunnerBase> *>(attempt.storage);
    ASSERT_TRUE(state->physical_complete);
    ASSERT_TRUE(state->execution_retired);
    ASSERT_TRUE(state->copyback_returned);
    ASSERT_TRUE(state->bindings_released);
    ASSERT_FALSE(state->result_observed);
    ASSERT_EQ(state->completion_rc, 0);
    control().observed_streams.clear();

    EXPECT_EQ(simpler_transfer_completed_kernel_run(&runner, attempt.storage), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_FALSE(attempt.empty());
    EXPECT_TRUE(runner.native_runs_outstanding());
    EXPECT_FALSE(runner.fresh_callable_addition_allowed());
    int all = 0;
    EXPECT_EQ(simpler_observe_framework_submissions(&runner, &all), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_EQ(all, 1);
    EXPECT_TRUE(control().observed_streams.empty());
    ASSERT_EQ(simpler_finalize_run(&runner, attempt.storage), 0);
    EXPECT_FALSE(runner.native_runs_outstanding());
}

TEST_F(NativeAdmissionTest, FrameworkBindingDebtDeduplicatesAndDoesNotHideAnotherRunOwner) {
    FrameworkCompletionRunner runner;
    ASSERT_EQ(initialize_capture_runner(runner), 0);
    for (uint64_t epoch = 10110; epoch != 10114; ++epoch) {
        Attempt attempt(runner, 0, epoch);
        attempt.args.tensor(0).address_space = AddressSpace::DEVICE;
        attempt.args.tensor(0).transfer = TensorTransfer::NONE;
        ASSERT_EQ(attempt.prepare(), 0);
        int caller;
        ASSERT_EQ(simpler_launch_run_on_stream(&runner, attempt.storage, &caller, nullptr), 0);
        ASSERT_EQ(simpler_poll_run(&runner, attempt.storage), SIMPLER_NATIVE_RUN_POLL_COMPLETE);
        ASSERT_EQ(simpler_transfer_completed_kernel_run(&runner, attempt.storage), 0);
        EXPECT_TRUE(runner.fresh_callable_addition_allowed());
    }
    Attempt live(runner, 1, 10120);
    live.args.tensor(0).address_space = AddressSpace::DEVICE;
    live.args.tensor(0).transfer = TensorTransfer::NONE;
    ASSERT_EQ(live.prepare(), 0);
    EXPECT_FALSE(runner.fresh_callable_addition_allowed());
    ASSERT_EQ(simpler_finalize_run(&runner, live.storage), 0);
    EXPECT_TRUE(runner.fresh_callable_addition_allowed());
    control().observed_streams.clear();
    int all = 0;
    EXPECT_EQ(simpler_observe_framework_submissions(&runner, &all), 0);
    EXPECT_EQ(all, 1);
    EXPECT_EQ(control().observed_streams.size(), 2u);
    EXPECT_FALSE(runner.native_runs_outstanding());
}

TEST_F(NativeAdmissionTest, EntryCacheQualificationRequiresObservedSuccessAndCurrentPublication) {
    class CacheRunner : public Runner {
    public:
        bool run_case(
            uint64_t epoch, bool fenced, bool matching_record, bool poll_only, uint32_t slot_index,
            bool fail_sync = false, bool resources_owned = true, int replacement = 0
        ) {
            Runtime runtime;
            CallConfig config;
            NativeRunIdentity identity{epoch, 3, epoch, slot_index};
            PreparedExecution prepared(identity, runtime, config, slot_index);
            static int streams[2];
            prepared.streams = {&streams[0], &streams[1]};
            prepared.resources_owned = resources_owned;
            auto &slot = slot_persistent_args(slot_index);
            EXPECT_EQ(ensure_device_run_result_region(identity, prepared.kernel_args, prepared.dfx), 0);
            EXPECT_EQ(prepared.kernel_args.prepare_runtime_args(runtime, mem_alloc_, slot), 0);
            LaunchProgressSink sink;
            EXPECT_EQ(prepared.kernel_args.publish_runtime_args(true, &streams[0], sink), 0);
            run_result_storage_.at(epoch).initialized = true;
            auto *record = reinterpret_cast<DeviceRunResultRegion *>(prepared.kernel_args.args.run_result_data_base);
            *record = {};
            record->published = matching_record ? epoch : epoch + 1;
            record->verdict = static_cast<uint32_t>(DeviceRunVerdict::Ok);
            EXPECT_EQ(arm_run_fence(prepared), 0);
            if (fenced) {
                auto &fence = run_fence(slot_index);
                for (auto role : {RunCompletionFence::StreamRole::Aicpu, RunCompletionFence::StreamRole::Aicore}) {
                    fence.note_kernel_submitted(identity, role);
                    EXPECT_EQ(fence.record(identity, role, &streams[0]), 0);
                }
            }
            KernelArgsHelper next;
            std::shared_ptr<PreparedSubmissionStorage> retained_submission;
            if (!resources_owned) {
                EXPECT_EQ(poll_run_fence(prepared), SIMPLER_NATIVE_RUN_POLL_COMPLETE);
                prepared.physical_complete = true;
                prepared.kernel_args.release_run_view();
                EXPECT_FALSE(slot.entry_cache_qualification.eligible());
                if (replacement == 3) EXPECT_EQ(release_slot_persistent_args(slot, mem_alloc_), 0);
                if (replacement == 4) {
                    retire_run_fence(prepared);
                    retained_submission = submission_storage_[slot_index];
                    EXPECT_EQ(prepare_run_storage(slot_index), 0);
                    EXPECT_NE(submission_storage_[slot_index], retained_submission);
                }
                if (replacement != 0) {
                    auto &next_slot = slot_persistent_args(slot_index);
                    next.set_entry_cache_epoch(epoch + 1000);
                    EXPECT_EQ(next.prepare_runtime_args(runtime, mem_alloc_, next_slot), 0);
                    if (retained_submission) EXPECT_NE(next_slot.runtime_args, slot.runtime_args);
                    if (replacement != 1) {
                        EXPECT_EQ(next.publish_runtime_args(true, &streams[0], sink), 0);
                    }
                }
            }
            if (fail_sync) control().stream_sync_rc = -81;
            if (poll_only) EXPECT_EQ(poll_run_fence(prepared), SIMPLER_NATIVE_RUN_POLL_COMPLETE);
            else EXPECT_EQ(wait_run_fence(prepared), fail_sync ? -81 : 0);
            control().stream_sync_rc = 0;
            const bool eligible = slot_persistent_args(slot_index).entry_cache_qualification.eligible();
            if (retained_submission) EXPECT_FALSE(slot.entry_cache_qualification.eligible());
            retire_run_fence(prepared);
            return eligible;
        }
    } runner;
    EXPECT_EQ(
        runner.run_case(101, true, true, false, 0),
        PLATFORM_SUPPORTS_LIVE_ENTRY_CACHE && runtime_launch_entry_args_plan(Runtime{}).supported
    );
    EXPECT_FALSE(runner.run_case(102, true, true, true, 0));
    EXPECT_EQ(
        runner.run_case(103, true, false, false, 0),
        PLATFORM_SUPPORTS_LIVE_ENTRY_CACHE && runtime_launch_entry_args_plan(Runtime{}).supported
    );
    EXPECT_EQ(
        runner.run_case(104, false, true, false, 0),
        PLATFORM_SUPPORTS_LIVE_ENTRY_CACHE && runtime_launch_entry_args_plan(Runtime{}).supported
    );
    EXPECT_FALSE(runner.run_case(105, true, true, true, 1));
    EXPECT_FALSE(runner.run_case(106, true, true, false, 0, true));
    EXPECT_EQ(
        runner.run_case(107, true, true, false, 0, false, false),
        PLATFORM_SUPPORTS_LIVE_ENTRY_CACHE && runtime_launch_entry_args_plan(Runtime{}).supported
    );
    EXPECT_FALSE(runner.run_case(108, true, true, false, 0, false, false, 1));
    EXPECT_FALSE(runner.run_case(109, true, true, false, 0, false, false, 2));
    EXPECT_FALSE(runner.run_case(110, true, true, false, 0, false, false, 3));
    EXPECT_FALSE(runner.run_case(111, true, true, false, 0, false, false, 4));
}

TEST_F(NativeAdmissionTest, LateCacheObservationSerializesWithStoragePublication) {
    class CacheRunner : public Runner {
    public:
        std::unique_lock<std::mutex> lock_storage() { return std::unique_lock<std::mutex>(prepared_storage_mutex_); }
        int observe() {
            Runtime runtime;
            CallConfig config;
            NativeRunIdentity identity{201, 3, 201, 0};
            PreparedExecution prepared(identity, runtime, config, 0);
            static int streams[2];
            prepared.streams = {&streams[0], &streams[1]};
            prepared.resources_owned = false;
            prepared.physical_complete = true;
            return wait_run_fence(prepared);
        }
    } runner;
    std::promise<void> synchronized;
    auto sync_completed = synchronized.get_future();
    std::atomic<unsigned> sync_calls{0};
    control().stream_sync_hook = [&] {
        if (sync_calls.fetch_add(1) == 1) synchronized.set_value();
    };
    auto storage_lock = runner.lock_storage();
    auto observed = std::async(std::launch::async, [&] {
        return runner.observe();
    });
    // RTS synchronization completes without the storage lock. The subsequent
    // allocation lookup shares the prepare/detach publication critical section.
    const auto sync_status = sync_completed.wait_for(std::chrono::seconds(1));
    const auto receipt_status = observed.wait_for(std::chrono::milliseconds(100));
    storage_lock.unlock();
    const int rc = observed.get();
    control().stream_sync_hook = {};
    EXPECT_EQ(sync_status, std::future_status::ready);
    EXPECT_EQ(receipt_status, std::future_status::timeout);
    EXPECT_EQ(sync_calls.load(), 2u);
    EXPECT_EQ(rc, 0);
}

TEST_F(NativeAdmissionTest, ColdResultCopyFailureRetainsAllocationForInitializationRetry) {
    Runner runner;
    control().copy_rc = -73;
    EXPECT_EQ(runner.prepare_result(), -73);
    auto &owner = runner.result_storage();
    void *allocation = owner.device_result;
    EXPECT_NE(allocation, nullptr);
    EXPECT_FALSE(owner.result_initialized);
    const auto allocations = control().allocations;
    control().copy_rc = 0;
    ASSERT_EQ(runner.prepare_result(), 0);
    EXPECT_EQ(owner.device_result, allocation);
    EXPECT_EQ(control().allocations, allocations);
    EXPECT_TRUE(owner.result_initialized);
    auto *storage = static_cast<NativeInvocationResultStorage *>(owner.device_result);
    EXPECT_EQ(storage->result.published, 0);
    EXPECT_EQ(storage->evidence.read_after_exit().kind, NativeInvocationEvidence::Kind::None);
    EXPECT_EQ(runner.close_result(), 0);
}

TEST_F(NativeAdmissionTest, FailedEvidenceReadKeepsBackingThenRetryObservesAndReleasesOnce) {
    Runner runner;
    ASSERT_EQ(runner.prepare_result(), 0);
    auto &owner = runner.result_storage();
    auto *storage = static_cast<NativeInvocationResultStorage *>(owner.device_result);
    const uint8_t payload[] = {9, 7};
    // No device tasks exist in this transport fixture: the owner has exclusive
    // access and no pending device use, independently of the stored verdict.
    storage->evidence.begin_after_exit();
    storage->evidence.complete({DeviceRunVerdict::Error, -51, DeviceRunCodeSource::Header}, payload, 2, true);
    void *allocation = owner.device_result;
    const auto bytes = control().bytes;
    control().copy_rc = -77;
    EXPECT_EQ(runner.close_result(), -77);
    EXPECT_EQ(owner.device_result, allocation);
    EXPECT_EQ(control().bytes, bytes);
    EXPECT_FALSE(owner.evidence_observed);
    control().copy_rc = 0;
    ASSERT_EQ(runner.close_result(), 0);
    EXPECT_EQ(owner.device_result, nullptr);
    EXPECT_TRUE(owner.evidence_observed);
    EXPECT_EQ(owner.observed_evidence.terminal.code, -51);
    EXPECT_EQ(owner.observed_evidence.payload[0], 9);
    EXPECT_EQ(owner.observed_evidence.payload_bytes, 2);
    const auto remaining = control().bytes;
    control().copy_rc = -88;
    EXPECT_EQ(runner.close_result(), 0);
    EXPECT_EQ(control().bytes, remaining);
    control().copy_rc = 0;
}

TEST_F(NativeAdmissionTest, FailedResultFreeRetainsObservedEvidenceWithoutRepeatingReadback) {
    Runner runner;
    ASSERT_EQ(runner.prepare_result(), 0);
    auto &owner = runner.result_storage();
    auto *storage = static_cast<NativeInvocationResultStorage *>(owner.device_result);
    storage->evidence.begin_after_exit();
    storage->evidence.complete({DeviceRunVerdict::Error, -61, DeviceRunCodeSource::ThreadRc}, nullptr, 0, true);
    void *allocation = owner.device_result;
    const auto bytes = control().bytes;
    control().free_rc = -91;
    EXPECT_EQ(runner.close_result(), -91);
    EXPECT_EQ(owner.device_result, allocation);
    EXPECT_EQ(control().bytes, bytes);
    EXPECT_TRUE(owner.evidence_observed);
    EXPECT_EQ(owner.observed_evidence.terminal.code, -61);
    control().free_rc = 0;
    control().copy_rc = -77;
    EXPECT_EQ(runner.close_result(), 0);
    EXPECT_EQ(owner.device_result, nullptr);
    EXPECT_EQ(owner.observed_evidence.terminal.code, -61);
    const auto remaining = control().bytes;
    EXPECT_EQ(runner.close_result(), 0);
    EXPECT_EQ(control().bytes, remaining);
    control().copy_rc = 0;
}

TEST_F(NativeAdmissionTest, ContextFinalizeCannotSweepResultBackingAfterReadOrFreeFailure) {
    for (bool fail_free : {false, true}) {
        Runner runner;
        ASSERT_EQ(runner.prepare_result(), 0);
        void *allocation = runner.result_storage().device_result;
        const auto bytes = control().bytes;
        control().copy_rc = fail_free ? 0 : -77;
        control().free_rc = fail_free ? -91 : 0;
        EXPECT_EQ(runner.finalize(), fail_free ? -91 : -77);
        EXPECT_EQ(runner.result_storage().device_result, allocation);
        EXPECT_EQ(control().bytes, bytes);
        control().copy_rc = 0;
        control().free_rc = 0;
        EXPECT_EQ(runner.finalize(), 0);
        EXPECT_EQ(control().bytes, 0u);
    }
}

TEST_F(NativeAdmissionTest, ExternalResourceReleaseRetriesOnlyItsFailedManagedBacking) {
    class ResourceRunner : public FencedCompletionRunner {
    public:
        explicit ResourceRunner(bool arena) :
            arena_(arena) {
            static int binary;
            aicore_bin_handle_ = &binary;
            EXPECT_EQ(execution_mode_latch().latch(SIMPLER_MODE_PROGRAM), 0);
        }
        int prepare_execution(
            Runtime &runtime, const CallConfig &config, uint32_t slot, const NativeRunIdentity &identity,
            std::unique_ptr<PreparedExecution> *out
        ) override {
            if (arena_) {
                const int rc = setup_static_arena(0, 4096, 2048, 1024);
                if (rc != 0) return rc;
            }
            return FencedCompletionRunner::prepare_execution(runtime, config, slot, identity, out);
        }

    private:
        bool arena_;
    };
    for (bool arena : {false, true}) {
        SCOPED_TRACE(arena ? "shared-memory arena" : "input staging");
        ResourceRunner runner(arena);
        int caller = 0;
        Attempt first(runner, 0, 9100);
        ASSERT_EQ(first.prepare(), 0);
        void *staging = nullptr;
        runner.get_retained_temp_buffer(0, &staging, nullptr);
        void *failed = arena ? runner.arena_region_backing(0, 1) : staging;
        ASSERT_NE(failed, nullptr);
        RuntimeHandle first_reference = nullptr;
        ASSERT_EQ(simpler_retain_run_resources(&runner, first.storage, &caller, &first_reference), 0);
        ASSERT_EQ(simpler_finalize_run(&runner, first.storage), 0);

        Attempt second(runner, 0, 9101);
        ASSERT_EQ(second.prepare(), 0);
        RuntimeHandle second_reference = nullptr;
        ASSERT_EQ(simpler_retain_run_resources(&runner, second.storage, &caller, &second_reference), 0);
        ASSERT_EQ(simpler_finalize_run(&runner, second.storage), 0);
        auto cleanup = RAIIScopeGuard([&] {
            control().selected_free_rc = 0;
            control().selected_free = nullptr;
            (void)simpler_release_run_resources(&runner, first_reference);
            (void)simpler_release_run_resources(&runner, second_reference);
        });

        control().selected_free = failed;
        control().selected_free_rc = -919;
        control().selected_free_attempts = 0;
        EXPECT_EQ(simpler_release_run_resources(&runner, first_reference), -919);
        EXPECT_EQ(control().selected_free_attempts, 1u);
        SimplerWorkspaceReport failed_report{};
        ASSERT_EQ(simpler_get_workspace_report_ctx(&runner, &failed_report, sizeof(failed_report)), 0);
        EXPECT_EQ(failed_report.release_unconfirmed_blocks, 1u);
        EXPECT_EQ(failed_report.proof_unavailable, 1u);
        const auto failed_bytes = control().bytes;
        EXPECT_EQ(simpler_release_run_resources(&runner, first_reference), -919);
        EXPECT_EQ(control().selected_free_attempts, 2u);
        EXPECT_EQ(runner.charged(), failed_report.reserved_bytes);
        EXPECT_EQ(control().bytes, failed_bytes);

        control().selected_free_rc = 0;
        EXPECT_EQ(simpler_release_run_resources(&runner, first_reference), 0);
        EXPECT_EQ(control().selected_free_attempts, 3u);
        SimplerWorkspaceReport recovered{};
        ASSERT_EQ(simpler_get_workspace_report_ctx(&runner, &recovered, sizeof(recovered)), 0);
        EXPECT_EQ(recovered.release_unconfirmed_blocks, 0u);
        EXPECT_EQ(recovered.proof_unavailable, 0u);
        EXPECT_LT(recovered.reserved_bytes, failed_report.reserved_bytes);
        EXPECT_EQ(failed_report.reserved_bytes - recovered.reserved_bytes, failed_bytes - control().bytes);
        EXPECT_EQ(simpler_release_run_resources(&runner, first_reference), PTO_RUNTIME_ERR_INVALID_ARGUMENT);
        EXPECT_EQ(control().selected_free_attempts, 3u);
        ASSERT_EQ(simpler_release_run_resources(&runner, second_reference), 0);

        Attempt larger(runner, 0, 9102, 64 * kUnit);
        const int prepare_rc = larger.prepare();
        EXPECT_EQ(prepare_rc, 0);
        if (prepare_rc == 0) EXPECT_EQ(simpler_finalize_run(&runner, larger.storage), 0);
        else EXPECT_TRUE(larger.consume_release());
        control().selected_free = nullptr;
    }
}

namespace {
class ScopeCaptureRunner : public FencedCompletionRunner {
public:
    ScopeCaptureRunner() {
        static int binary;
        aicore_bin_handle_ = &binary;
    }
    ~ScopeCaptureRunner() override {
        if (emit_scope_records) {
            set_scope_stats_enabled(false);
            set_platform_scope_stats_base(0);
            set_platform_scope_stats_run_terminal(0);
            set_platform_run_result(0, 0);
        }
        scope_stats_collector_.finalize(nullptr, [this](void *pointer) {
            return mem_alloc_.free(pointer);
        });
    }
    bool supports_scope_stats_capture() const override { return supported; }
    int prepare_collector_storage(const Runtime &, PreparedExecution &prepared) override {
        ++arm_calls;
        if (during_arm) during_arm();
        const int rc = scope_stats_collector_.init(
            1,
            [this](size_t bytes) {
                return mem_alloc_.alloc(bytes);
            },
            nullptr,
            [this](void *pointer) {
                return mem_alloc_.free(pointer);
            },
            0
        );
        prepared.kernel_args.args.scope_stats_data_base =
            reinterpret_cast<uint64_t>(scope_stats_collector_.get_scope_stats_shm_device_ptr());
        SIMPLER_SET_DFX_FLAG(prepared.kernel_args.args.enable_profiling_flag, SIMPLER_DFX_FLAG_SCOPE_STATS);
        return rc;
    }
    LaunchTransactionResult launch_execution_impl(PreparedExecution &prepared, LaunchPermit permit) override {
        if (refuse_eager) return {-887, LaunchProgress::NotStarted, {}};
        if (emit_scope_records) {
            const int bank_rc = open_run_terminal_banks_for_enabled_owners(prepared.dfx);
            if (bank_rc != 0) return {bank_rc, LaunchProgress::NotStarted, {}};
            const RunLocalDfxMetadata metadata{
                nullptr, 0, false, &prepared.artifact_identity, &prepared.scope_stats_capacity
            };
            DiagnosticAdmission admission;
            const int rc =
                start_shared_collectors_for_run(prepared.dfx, prepared.identity.run_epoch, metadata, &admission);
            if (rc != 0) return {rc, LaunchProgress::NotStarted, {}};
            resolve_run_terminal_banks(prepared.dfx, prepared.identity.run_epoch, prepared.kernel_args);
        }
        auto result = FencedCompletionRunner::launch_execution_impl(prepared, std::move(permit));
        if (result.rc == 0 && emit_scope_records) {
            set_scope_stats_enabled(true);
            set_platform_scope_stats_base(prepared.kernel_args.args.scope_stats_data_base);
            set_platform_scope_stats_run_terminal(prepared.kernel_args.args.scope_stats_run_terminal_bank);
            scope_stats_aicpu_set_orch_thread_idx(0);
            set_platform_run_result(0, prepared.identity.run_epoch);
            scope_stats_aicpu_begin_run();
            scope_stats_set_pending_site("native-eager.cpp", 120);
            scope_stats_begin(0, 1, 2, 1024, 2048, 1, 2, 3);
            scope_stats_end(0, 1, 3, 1024, 3072, 1, 3, 4);
            scope_stats_aicpu_flush_buffers();
            scope_stats_aicpu_close_run();
            EXPECT_EQ(queue_caller_exit(prepared), 0);
            ++scope_submissions;
        }
        return result;
    }
    int retire_completed_execution(ActiveExecution &active) override {
        const bool close_scope = emit_scope_records && active.prepared->resources_owned;
        const int rc = FencedCompletionRunner::retire_completed_execution(active);
        if (rc == SIMPLER_NATIVE_RUN_POLL_COMPLETE && close_scope) {
            auto &prepared = *active.prepared;
            prepared.retirement_diagnostics_rc = teardown_shared_collectors_after_run(
                prepared.dfx, prepared.pipeline_slot, prepared.identity.run_epoch, prepared.physical_complete
            );
        }
        return rc;
    }
    bool emit_scope_records{false};
    int scope_submissions{0};
    bool refuse_eager{false};
    bool window(uint64_t epoch) const { return scope_stats_collector_.recorded_window_open(epoch); }
    bool ordinary_retention() const { return scope_stats_collector_.retains_runs(); }
    bool flush(std::string *error) { return scope_stats_collector_.flush_retained_runs(5000, error); }
    bool supported{true};
    int arm_calls{0};
    std::function<void()> during_arm;
};
}  // namespace

TEST_F(NativeAdmissionTest, CaptureScopeWindowSurvivesSourceAndFailedReleaseWithoutEnablingOrdinaryRetention) {
    ScopeCaptureRunner runner;
    ASSERT_EQ(initialize_capture_runner(runner), 0);
    Attempt attempt(runner, 0, 9600);
    use_capture_preparation(attempt);
    diagnostic_path = "native-scope-capture-lifetime";
    attempt.config.enable_scope_stats = 1;
    std::strncpy(attempt.config.output_prefix, diagnostic_path.c_str(), sizeof(attempt.config.output_prefix) - 1);
    ASSERT_EQ(attempt.prepare(), 0);
    auto *state = static_cast<NativeRunContext<DeviceRunnerBase> *>(attempt.storage);
    int caller;
    RuntimeHandle reference = nullptr;
    if (!state->prepared_execution->scope_stats_capacity.required) {
        EXPECT_EQ(
            simpler_retain_run_resources(&runner, attempt.storage, &caller, &reference), PTO_RUNTIME_ERR_UNSUPPORTED
        );
        EXPECT_EQ(runner.arm_calls, 0);
        EXPECT_EQ(reference, nullptr);
        return;
    }
    ASSERT_EQ(simpler_retain_run_resources(&runner, attempt.storage, &caller, &reference), 0);
    auto release = RAIIScopeGuard([&] {
        control().stream_sync_rc = 0;
        (void)simpler_finalize_run(&runner, attempt.storage);
        (void)simpler_release_run_resources(&runner, reference);
    });
    EXPECT_FALSE(runner.ordinary_retention());
    EXPECT_TRUE(runner.window(9600));
    EXPECT_NE(state->prepared_execution->kernel_args.args.scope_stats_data_base, 0u);
    EXPECT_EQ(state->prepared_execution->kernel_args.args.scope_stats_run_terminal_bank, 0u);
    RuntimeHandle duplicate = reinterpret_cast<void *>(1);
    EXPECT_NE(simpler_retain_run_resources(&runner, attempt.storage, &caller, &duplicate), 0);
    EXPECT_EQ(duplicate, nullptr);
    EXPECT_TRUE(runner.window(9600)) << "a duplicate retain discarded the original receipt window";
    ASSERT_EQ(runner.admit_recording(*state->prepared_execution), 0);
    state->recording = true;
    state->completion_rc = 0;
    ASSERT_EQ(simpler_finalize_run(&runner, attempt.storage), 0);
    EXPECT_TRUE(runner.window(9600));
    control().stream_sync_rc = -996;
    EXPECT_EQ(simpler_release_run_resources(&runner, reference), -996);
    EXPECT_TRUE(runner.window(9600));
    control().stream_sync_rc = 0;
    ASSERT_EQ(simpler_release_run_resources(&runner, reference), 0);
    release.dismiss();
    EXPECT_FALSE(runner.window(9600));
    std::string error;
    EXPECT_TRUE(runner.flush(&error)) << error;
    EXPECT_TRUE(std::filesystem::exists(std::filesystem::path(diagnostic_path) / "scope_stats/scope_stats.jsonl"));
}

TEST_F(NativeAdmissionTest, UnsupportedCaptureDiagnosticsRefuseBeforePoolMutation) {
    for (bool other_diagnostic : {false, true}) {
        ScopeCaptureRunner runner;
        ASSERT_EQ(initialize_capture_runner(runner), 0);
        Attempt attempt(runner, 0, 9601);
        use_capture_preparation(attempt);
        attempt.config.enable_scope_stats = 1;
        attempt.config.enable_pmu = other_diagnostic ? 2 : 0;
        std::strcpy(attempt.config.output_prefix, "unused-scope-refusal");
        runner.supported = other_diagnostic;
        ASSERT_EQ(attempt.prepare(), 0);
        const auto allocations = control().allocations;
        int caller;
        RuntimeHandle reference = reinterpret_cast<void *>(1);
        EXPECT_EQ(
            simpler_retain_run_resources(&runner, attempt.storage, &caller, &reference), PTO_RUNTIME_ERR_UNSUPPORTED
        );
        EXPECT_EQ(reference, nullptr);
        EXPECT_EQ(runner.arm_calls, 0);
        EXPECT_EQ(control().allocations, allocations);
    }
}

TEST_F(NativeAdmissionTest, CaptureScopeArmingExcludesAContendingExecutionClaim) {
    ScopeCaptureRunner runner;
    ASSERT_EQ(initialize_capture_runner(runner), 0);
    Attempt first(runner, 0, 9602), second(runner, 1, 9603);
    use_capture_preparation(first);
    use_capture_preparation(second);
    first.config.enable_scope_stats = 1;
    std::strcpy(first.config.output_prefix, "native-scope-claim-exclusion");
    diagnostic_path = first.config.output_prefix;
    ASSERT_EQ(first.prepare(), 0);
    ASSERT_EQ(second.prepare(), 0);
    auto *first_state = static_cast<NativeRunContext<DeviceRunnerBase> *>(first.storage);
    auto *second_state = static_cast<NativeRunContext<DeviceRunnerBase> *>(second.storage);
    if (!first_state->prepared_execution->scope_stats_capacity.required) return;
    std::promise<void> arming, continue_arm, contender_started;
    auto continue_future = continue_arm.get_future();
    runner.during_arm = [&] {
        arming.set_value();
        continue_future.wait();
    };
    int caller;
    RuntimeHandle reference = nullptr;
    auto retain = std::async(std::launch::async, [&] {
        return simpler_retain_run_resources(&runner, first.storage, &caller, &reference);
    });
    arming.get_future().wait();
    auto contender = std::async(std::launch::async, [&] {
        contender_started.set_value();
        LaunchPermit permit;
        const bool acquired = runner.try_acquire_native_run(second.storage, second_state->identity(), &permit);
        if (acquired) runner.release_native_run(second.storage);
        return acquired;
    });
    contender_started.get_future().wait();
    EXPECT_EQ(contender.wait_for(std::chrono::milliseconds(20)), std::future_status::timeout);
    continue_arm.set_value();
    EXPECT_EQ(retain.get(), 0);
    EXPECT_TRUE(contender.get());
    ASSERT_EQ(simpler_finalize_run(&runner, first.storage), 0);
    EXPECT_EQ(simpler_release_run_resources(&runner, reference), 0);
    std::string error;
    EXPECT_TRUE(runner.flush(&error)) << error;
}

TEST_F(NativeAdmissionTest, RetainedPreparationSuccessfullyCollectsItsEagerScopeRun) {
    diagnostic_path = "native-scope-successful-eager";
    for (bool retain : {false, true}) {
        SCOPED_TRACE(retain);
        ScopeCaptureRunner runner;
        runner.set_retain_runs(retain);
        runner.emit_scope_records = true;
        ASSERT_EQ(initialize_capture_runner(runner), 0);
        Attempt attempt(runner, 0, retain ? 9611 : 9610);
        use_capture_preparation(attempt);
        attempt.config.enable_scope_stats = 1;
        const auto output = std::filesystem::path(diagnostic_path) / (retain ? "retained" : "ordinary");
        std::strcpy(attempt.config.output_prefix, output.c_str());
        ASSERT_EQ(attempt.prepare(), 0);
        auto *state = static_cast<NativeRunContext<DeviceRunnerBase> *>(attempt.storage);
        if (!state->prepared_execution->scope_stats_capacity.required)
            GTEST_SKIP() << "runtime has no ScopeStats producer";
        int caller;
        RuntimeHandle reference = nullptr;
        ASSERT_EQ(simpler_retain_run_resources(&runner, attempt.storage, &caller, &reference), 0);
        auto release = RAIIScopeGuard([&] {
            (void)simpler_finalize_run(&runner, attempt.storage);
            (void)simpler_release_run_resources(&runner, reference);
        });
        ASSERT_TRUE(runner.window(state->identity().run_epoch));
        ASSERT_EQ(simpler_launch_run_on_stream(&runner, attempt.storage, &caller, nullptr), 0);
        EXPECT_FALSE(runner.window(state->identity().run_epoch));
        EXPECT_EQ(runner.scope_submissions, 1);
        ASSERT_EQ(simpler_poll_run(&runner, attempt.storage), SIMPLER_NATIVE_RUN_POLL_COMPLETE);
        ASSERT_EQ(simpler_wait_run(&runner, attempt.storage), 0);
        ASSERT_EQ(simpler_finalize_run(&runner, attempt.storage), 0);
        ASSERT_EQ(simpler_release_run_resources(&runner, reference), 0);
        release.dismiss();
        std::string error;
        ASSERT_TRUE(runner.flush(&error)) << error;
        std::ifstream artifact(output / "scope_stats/scope_stats.jsonl");
        ASSERT_TRUE(artifact.is_open());
        const std::string text{std::istreambuf_iterator<char>(artifact), std::istreambuf_iterator<char>()};
        EXPECT_NE(text.find("native-eager.cpp"), std::string::npos) << text;
        EXPECT_EQ(std::count(text.begin(), text.end(), '\n'), 3);
        EXPECT_NE(text.find("\"total\": 2"), std::string::npos) << text;
        if (retain) {
            EXPECT_NE(text.find("\"host_received_records\": 2"), std::string::npos) << text;
            EXPECT_NE(text.find("\"counts_unknown\": false"), std::string::npos) << text;
        }
        EXPECT_EQ(text.find("\"recorded_window\": true"), std::string::npos) << text;
        EXPECT_EQ(runner.ordinary_retention(), retain);
    }
}

TEST_F(NativeAdmissionTest, RetainedPreparationSwitchingToEagerWithdrawsItsUnusedScopeWindow) {
    ScopeCaptureRunner runner;
    ASSERT_EQ(initialize_capture_runner(runner), 0);
    Attempt attempt(runner, 0, 9604);
    use_capture_preparation(attempt);
    attempt.config.enable_scope_stats = 1;
    diagnostic_path = "native-scope-unused-recording";
    std::strcpy(attempt.config.output_prefix, diagnostic_path.c_str());
    ASSERT_EQ(attempt.prepare(), 0);
    auto *state = static_cast<NativeRunContext<DeviceRunnerBase> *>(attempt.storage);
    if (!state->prepared_execution->scope_stats_capacity.required) return;
    int caller;
    RuntimeHandle reference = nullptr;
    ASSERT_EQ(simpler_retain_run_resources(&runner, attempt.storage, &caller, &reference), 0);
    auto release = RAIIScopeGuard([&] {
        (void)simpler_finalize_run(&runner, attempt.storage);
        (void)simpler_release_run_resources(&runner, reference);
    });
    EXPECT_TRUE(runner.window(9604));
    // The backend refuses before submission; withdrawal belongs to common eager
    // admission and does not depend on synthetic device-completion evidence.
    runner.refuse_eager = true;
    ASSERT_EQ(simpler_launch_run_on_stream(&runner, attempt.storage, &caller, nullptr), -887);
    EXPECT_FALSE(runner.window(9604));
    EXPECT_FALSE(runner.ordinary_retention());
    ASSERT_EQ(simpler_finalize_run(&runner, attempt.storage), 0);
    EXPECT_TRUE(attempt.consume_release());
    EXPECT_EQ(simpler_release_run_resources(&runner, reference), 0);
    release.dismiss();
}

}  // namespace native_admission_test

namespace native_admission_test {
namespace {
class PmuCaptureRunner : public FencedCompletionRunner {
public:
    PmuCaptureRunner() {
        static int binary;
        aicore_bin_handle_ = &binary;
    }
    ~PmuCaptureRunner() override {
        pmu_collector_.finalize(nullptr, [this](void *p) {
            return mem_alloc_.free(p);
        });
    }
    bool supports_pmu_capture() const override { return supported; }
    bool supported{true};
    int prepare_collector_storage(const Runtime &, PreparedExecution &prepared) override {
        ++arm_calls;
        const int rc = pmu_collector_.init(
            1, 1,
            [this](size_t n) {
                return mem_alloc_.alloc(n);
            },
            nullptr,
            [this](void *p) {
                return mem_alloc_.free(p);
            },
            0
        );
        prepared.kernel_args.args.pmu_data_base = reinterpret_cast<uint64_t>(pmu_collector_.get_pmu_shm_device_ptr());
        SIMPLER_SET_DFX_FLAG(prepared.kernel_args.args.enable_profiling_flag, SIMPLER_DFX_FLAG_PMU);
        return rc;
    }
    bool flush(std::string *error) { return pmu_collector_.flush_retained_runs(5000, error); }
    int arm_calls{0};
};
}  // namespace

TEST_F(NativeAdmissionTest, OptionalResultNativeRecordingKeepsFiveModelsWithoutPrivateHistory) {
    class CaptureRunner : public PmuCaptureRunner {
    public:
        LaunchTransactionResult launch_execution_impl(PreparedExecution &prepared, LaunchPermit permit) override {
            if (!prepared.recording) return FencedCompletionRunner::launch_execution_impl(prepared, std::move(permit));
            prepared.streams = borrowed_run_streams();
            prepared.streams.aicpu = prepared.caller_stream;
            const int rc = arm_run_fence(prepared);
            if (rc != 0) return {rc, LaunchProgress::NotStarted, {}};
            return exact_launch_transaction(
                prepared.identity, std::move(permit),
                [&](LaunchProgressSink &sink) {
                    const int entry_rc = queue_run_entry(prepared, sink);
                    if (entry_rc != 0) return entry_rc;
                    sink.mark_submitted();
                    return record_run_boundary(prepared, RunCompletionFence::StreamRole::Aicore);
                },
                [&](LaunchProgressSink &sink) {
                    sink.mark_submitted();
                    const int boundary_rc = record_run_boundary(prepared, RunCompletionFence::StreamRole::Aicpu);
                    return boundary_rc != 0 ? boundary_rc : queue_caller_exit(prepared);
                }
            );
        }
    } runner;
    ASSERT_TRUE(runner.set_run_resource_grant(2));
    ASSERT_EQ(initialize_capture_runner(runner), 0);
    std::vector<RuntimeHandle> references;
    int caller;
    const size_t downloads = control().d2h_calls;
    const auto record_allocations = [] {
        return std::count_if(
            control().allocated_device_blocks.begin(), control().allocated_device_blocks.end(), [](const auto &entry) {
                return entry.second == sizeof(NativeInvocationResultStorage);
            }
        );
    };
    const auto initial_records = record_allocations();
    const std::array<void *, 2> warm_records{
        runner.result_storage(0).device_result, runner.result_storage(1).device_result
    };
    const auto result_uploads = [&] {
        return std::count_if(control().uploads.begin(), control().uploads.end(), [&](const auto &entry) {
            return entry.first != nullptr &&
                   std::find(warm_records.begin(), warm_records.end(), entry.first) != warm_records.end();
        });
    };
    const auto initial_uploads = result_uploads();
    auto cleanup = RAIIScopeGuard([&] {
        control().recording_active = false;
        control().stream_sync_rc = 0;
        for (auto reference : references)
            (void)simpler_release_run_resources(&runner, reference);
    });
    for (uint32_t index = 0; index < 5; ++index) {
        Attempt attempt(runner, index % 2, 94100 + index);
        use_capture_preparation(attempt);
        ASSERT_FALSE(control().recording_active);
        ASSERT_EQ(attempt.prepare(), 0) << index;
        const auto *state = static_cast<const NativeRunContext<DeviceRunnerBase> *>(attempt.storage);
        EXPECT_EQ(state->prepared_execution->kernel_args.args.run_result_data_base, 0u);
        EXPECT_EQ(state->prepared_execution->kernel_args.args.run_result_epoch, attempt.descriptor.run_epoch);
        RuntimeHandle reference = nullptr;
        ASSERT_EQ(simpler_retain_run_resources(&runner, attempt.storage, &caller, &reference), 0);
        ASSERT_NE(reference, nullptr);
        references.push_back(reference);
        control().recording_active = true;
        ASSERT_EQ(simpler_launch_run_on_stream(&runner, attempt.storage, &caller, nullptr), 0);
        EXPECT_FALSE(runner.result_storage(index % 2).retain_replay_evidence);
        EXPECT_EQ(record_allocations(), initial_records);
        EXPECT_EQ(result_uploads(), initial_uploads);
        EXPECT_EQ(control().d2h_calls, downloads);
        control().recording_active = false;
        ASSERT_EQ(simpler_finalize_run(&runner, attempt.storage), 0);
    }
    control().stream_sync_rc = -947;
    EXPECT_EQ(simpler_release_run_resources(&runner, references.front()), -947);
    control().stream_sync_rc = 0;
    for (auto reference : references)
        ASSERT_EQ(simpler_release_run_resources(&runner, reference), 0);
    references.clear();
    EXPECT_EQ(control().d2h_calls, downloads);
    ASSERT_EQ(runner.finalize(), 0);
    cleanup.dismiss();
}

TEST_F(NativeAdmissionTest, CapturePmuWindowLivesThroughSourceRetirementAndFailedRelease) {
    PmuCaptureRunner runner;
    ASSERT_EQ(initialize_capture_runner(runner), 0);
    Attempt attempt(runner, 0, 9700);
    use_capture_preparation(attempt);
    diagnostic_path = "native-pmu-capture-lifetime";
    attempt.config.enable_pmu = 2;
    std::strncpy(attempt.config.output_prefix, diagnostic_path.c_str(), sizeof(attempt.config.output_prefix) - 1);
    ASSERT_EQ(attempt.prepare(), 0);
    auto *state = static_cast<NativeRunContext<DeviceRunnerBase> *>(attempt.storage);
    int caller;
    RuntimeHandle reference = nullptr;
    ASSERT_EQ(simpler_retain_run_resources(&runner, attempt.storage, &caller, &reference), 0);
    auto release = RAIIScopeGuard([&] {
        control().stream_sync_rc = 0;
        (void)simpler_finalize_run(&runner, attempt.storage);
        (void)simpler_release_run_resources(&runner, reference);
    });
    EXPECT_EQ(runner.arm_calls, 1);
    EXPECT_EQ(control().pmu_start_calls, 1u);
    EXPECT_EQ(control().pmu_stop_calls, 0u);
    EXPECT_NE(state->prepared_execution->kernel_args.args.pmu_data_base, 0u);
    EXPECT_EQ(state->prepared_execution->kernel_args.args.pmu_run_terminal_bank, 0u);
    ASSERT_EQ(runner.admit_recording(*state->prepared_execution), 0);
    state->recording = true;
    state->completion_rc = 0;
    ASSERT_EQ(simpler_finalize_run(&runner, attempt.storage), 0);
    EXPECT_EQ(control().pmu_stop_calls, 0u) << "original retirement must not release model protection";
    control().stream_sync_rc = -996;
    EXPECT_EQ(simpler_release_run_resources(&runner, reference), -996);
    EXPECT_EQ(control().pmu_stop_calls, 0u);
    control().stream_sync_rc = 0;
    ASSERT_EQ(simpler_release_run_resources(&runner, reference), 0);
    release.dismiss();
    EXPECT_EQ(control().pmu_stop_calls, 1u);
    std::string error;
    EXPECT_TRUE(runner.flush(&error)) << error;
    EXPECT_TRUE(std::filesystem::exists(std::filesystem::path(diagnostic_path) / "pmu.collection.json"));
}

namespace {
class ArgsDumpCaptureRunner : public FencedCompletionRunner {
public:
    ArgsDumpCaptureRunner() {
        static int binary;
        aicore_bin_handle_ = &binary;
    }
    ~ArgsDumpCaptureRunner() override {
        dump_collector_.finalize(nullptr, [this](void *p) {
            return mem_alloc_.free(p);
        });
    }
    bool supports_args_dump_capture() const override { return supported; }
    bool supported{true};
    bool retain_ordinary{false};
    LaunchTransactionResult launch_execution_impl(PreparedExecution &prepared, LaunchPermit permit) override {
        if (prepared.recording) return Runner::launch_execution_impl(prepared, std::move(permit));
        return FencedCompletionRunner::launch_execution_impl(prepared, std::move(permit));
    }
    bool window(uint64_t epoch) const { return dump_collector_.recorded_window_open(epoch); }
    bool ordinary_retention() const { return dump_collector_.retains_runs(); }
    bool flush(std::string *error) { return dump_collector_.flush_retained_runs(5000, error); }
    int prepare_collector_storage(const Runtime &, PreparedExecution &prepared) override {
        ++arm_calls;
        dump_collector_.configure_retained_runs(retain_ordinary, simpler::dfx::runs::kDefaultBudgetBytes);
        const int rc = dump_collector_.initialize(
            1, 0, prepared.dfx.dump_args_level,
            [this](size_t n) {
                return mem_alloc_.alloc(n);
            },
            nullptr,
            [this](void *p) {
                return mem_alloc_.free(p);
            }
        );
        prepared.kernel_args.args.dump_data_base =
            reinterpret_cast<uint64_t>(dump_collector_.get_dump_shm_device_ptr());
        SIMPLER_SET_DFX_FLAG(prepared.kernel_args.args.enable_profiling_flag, SIMPLER_DFX_FLAG_DUMP_ARGS);
        return rc;
    }
    int arm_calls{0};
};
}  // namespace

TEST_F(NativeAdmissionTest, CaptureArgsDumpColdRetentionUsesTheCommonResourceReference) {
    if (args_dump_capture_supported_impl() == 0) GTEST_SKIP() << "runtime has no task-time argument producer";
    ArgsDumpCaptureRunner runner;
    ASSERT_EQ(initialize_capture_runner(runner), 0);
    Attempt attempt(runner, 0, 9800);
    use_capture_preparation(attempt);
    diagnostic_path = "native-argsdump-capture-lifetime";
    attempt.config.enable_dump_args = static_cast<int>(DumpArgsLevel::FULL);
    std::strncpy(attempt.config.output_prefix, diagnostic_path.c_str(), sizeof(attempt.config.output_prefix) - 1);
    ASSERT_EQ(attempt.prepare(), 0);
    auto *state = static_cast<NativeRunContext<DeviceRunnerBase> *>(attempt.storage);
    int caller;
    RuntimeHandle reference = nullptr;
    auto cleanup = RAIIScopeGuard([&] {
        control().stream_sync_rc = 0;
        (void)simpler_finalize_run(&runner, attempt.storage);
        if (reference != nullptr) (void)simpler_release_run_resources(&runner, reference);
    });
    ASSERT_EQ(simpler_retain_run_resources(&runner, attempt.storage, &caller, &reference), 0);
    EXPECT_EQ(runner.arm_calls, 1);
    EXPECT_FALSE(runner.ordinary_retention());
    EXPECT_TRUE(runner.window(9800));
    RuntimeHandle duplicate = reinterpret_cast<void *>(1);
    EXPECT_NE(simpler_retain_run_resources(&runner, attempt.storage, &caller, &duplicate), 0);
    EXPECT_EQ(duplicate, nullptr);
    EXPECT_TRUE(runner.window(9800));
    EXPECT_EQ(runner.arm_calls, 1);
    EXPECT_NE(state->prepared_execution->kernel_args.args.dump_data_base, 0u);
    EXPECT_EQ(state->prepared_execution->kernel_args.args.args_dump_run_terminal_bank, 0u);
    control().recording_active = true;
    ASSERT_EQ(simpler_launch_run_on_stream(&runner, attempt.storage, &caller, nullptr), 0);
    control().recording_active = false;
    EXPECT_TRUE(state->recording);
    EXPECT_TRUE(runner.window(9800));
    EXPECT_EQ(runner.arm_calls, 1);
    ASSERT_EQ(simpler_finalize_run(&runner, attempt.storage), 0);
    control().stream_sync_rc = -996;
    EXPECT_EQ(simpler_release_run_resources(&runner, reference), -996);
    control().stream_sync_rc = 0;
    ASSERT_EQ(simpler_release_run_resources(&runner, reference), 0);
    reference = nullptr;
    cleanup.dismiss();
    std::string error;
    EXPECT_TRUE(runner.flush(&error)) << error;
    EXPECT_TRUE(std::filesystem::exists(std::filesystem::path(diagnostic_path) / "args_dump/args_dump.json"));
}

TEST_F(NativeAdmissionTest, UnsubmittedRecordingPreservesItsArgsDumpWindowUntilExplicitClose) {
    if (args_dump_capture_supported_impl() == 0) GTEST_SKIP() << "runtime has no task-time argument producer";
    for (bool retaining : {false, true}) {
        SCOPED_TRACE(retaining);
        ArgsDumpCaptureRunner runner;
        runner.retain_ordinary = retaining;
        ASSERT_EQ(initialize_capture_runner(runner), 0);
        Attempt attempt(runner, 0, 9850);
        use_capture_preparation(attempt);
        diagnostic_path = "native-argsdump-rollback";
        attempt.config.enable_dump_args = static_cast<int>(DumpArgsLevel::FULL);
        std::strncpy(attempt.config.output_prefix, diagnostic_path.c_str(), sizeof(attempt.config.output_prefix) - 1);
        ASSERT_EQ(attempt.prepare(), 0);
        int caller;
        RuntimeHandle reference = nullptr;
        auto cleanup = RAIIScopeGuard([&] {
            control().recording_active = false;
            (void)simpler_finalize_run(&runner, attempt.storage);
            if (reference != nullptr) (void)simpler_release_run_resources(&runner, reference);
        });
        ASSERT_EQ(simpler_retain_run_resources(&runner, attempt.storage, &caller, &reference), 0);
        EXPECT_EQ(runner.ordinary_retention(), retaining);
        runner.launch_refusal_rc = -989;
        control().recording_active = true;
        EXPECT_EQ(simpler_launch_run_on_stream(&runner, attempt.storage, &caller, nullptr), -989);
        control().recording_active = false;
        EXPECT_TRUE(runner.window(9850));
        EXPECT_EQ(runner.arm_calls, 1);
        EXPECT_EQ(simpler_finalize_run(&runner, attempt.storage), -989);
        EXPECT_TRUE(runner.window(9850));
        ASSERT_EQ(simpler_release_run_resources(&runner, reference), 0);
        reference = nullptr;
        cleanup.dismiss();
        std::string error;
        EXPECT_TRUE(runner.flush(&error)) << error;
        EXPECT_TRUE(std::filesystem::exists(std::filesystem::path(diagnostic_path) / "args_dump/args_dump.json"));
    }
}

TEST_F(NativeAdmissionTest, UnsupportedArgsDumpCaptureRefusesBeforeAnyPoolMutation) {
    if (args_dump_capture_supported_impl() == 0) GTEST_SKIP() << "runtime has no task-time argument producer";
    for (bool mixed : {false, true}) {
        ArgsDumpCaptureRunner runner;
        runner.supported = mixed;
        ASSERT_EQ(initialize_capture_runner(runner), 0);
        Attempt attempt(runner, 0, 9801);
        use_capture_preparation(attempt);
        attempt.config.enable_dump_args = static_cast<int>(DumpArgsLevel::FULL);
        attempt.config.enable_pmu = mixed ? 2 : 0;
        std::strcpy(attempt.config.output_prefix, "unused-dump-refusal");
        ASSERT_EQ(attempt.prepare(), 0);
        const auto allocations = control().allocations;
        int caller;
        RuntimeHandle reference = reinterpret_cast<void *>(1);
        EXPECT_EQ(
            simpler_retain_run_resources(&runner, attempt.storage, &caller, &reference), PTO_RUNTIME_ERR_UNSUPPORTED
        );
        EXPECT_EQ(reference, nullptr);
        EXPECT_EQ(runner.arm_calls, 0);
        EXPECT_EQ(control().allocations, allocations);
    }
}

TEST_F(NativeAdmissionTest, UnsupportedPmuCaptureRefusesBeforePoolOrDriverMutation) {
    for (bool mixed : {false, true}) {
        PmuCaptureRunner runner;
        runner.supported = mixed;
        ASSERT_EQ(initialize_capture_runner(runner), 0);
        Attempt attempt(runner, 0, 9701);
        use_capture_preparation(attempt);
        attempt.config.enable_pmu = 2;
        attempt.config.enable_scope_stats = mixed;
        std::strcpy(attempt.config.output_prefix, "unused-pmu-refusal");
        ASSERT_EQ(attempt.prepare(), 0);
        const auto allocations = control().allocations;
        int caller;
        RuntimeHandle reference = reinterpret_cast<void *>(1);
        EXPECT_EQ(
            simpler_retain_run_resources(&runner, attempt.storage, &caller, &reference), PTO_RUNTIME_ERR_UNSUPPORTED
        );
        EXPECT_EQ(reference, nullptr);
        EXPECT_EQ(runner.arm_calls, 0);
        EXPECT_EQ(control().allocations, allocations);
        EXPECT_EQ(control().pmu_start_calls, 0u);
    }
}

TEST_F(NativeAdmissionTest, UnavailablePmuProtectionDoesNotRejectBusinessCapture) {
    PmuCaptureRunner runner;
    ASSERT_EQ(initialize_capture_runner(runner), 0);
    Attempt attempt(runner, 0, 9702);
    use_capture_preparation(attempt);
    diagnostic_path = "native-pmu-unprotected";
    attempt.config.enable_pmu = 2;
    std::strncpy(attempt.config.output_prefix, diagnostic_path.c_str(), sizeof(attempt.config.output_prefix) - 1);
    ASSERT_EQ(attempt.prepare(), 0);
    control().pmu_start_rc = -10;
    int caller;
    RuntimeHandle reference = nullptr;
    ASSERT_EQ(simpler_retain_run_resources(&runner, attempt.storage, &caller, &reference), 0);
    EXPECT_EQ(control().pmu_start_calls, 1u);
    EXPECT_EQ(simpler_finalize_run(&runner, attempt.storage), 0);
    EXPECT_EQ(simpler_release_run_resources(&runner, reference), 0);
    std::string error;
    EXPECT_TRUE(runner.flush(&error)) << error;
    std::ifstream file(std::filesystem::path(diagnostic_path) / "pmu.collection.json");
    std::string metadata{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
    EXPECT_NE(metadata.find("occupied"), std::string::npos);
}

TEST_F(NativeAdmissionTest, TwoPmuModelsShareOneLeaseUntilBothReferencesEnd) {
    PmuCaptureRunner runner;
    ASSERT_EQ(initialize_capture_runner(runner), 0);
    diagnostic_path = "native-pmu-two-models";
    int caller;
    RuntimeHandle first_ref = nullptr, second_ref = nullptr;
    Attempt first(runner, 0, 9703), second(runner, 1, 9704);
    auto capture = [&](Attempt &attempt, const char *leaf, RuntimeHandle *reference) {
        use_capture_preparation(attempt);
        attempt.config.enable_pmu = 2;
        const std::string prefix = diagnostic_path + "/" + leaf;
        std::strncpy(attempt.config.output_prefix, prefix.c_str(), sizeof(attempt.config.output_prefix) - 1);
        EXPECT_EQ(attempt.prepare(), 0);
        EXPECT_EQ(simpler_retain_run_resources(&runner, attempt.storage, &caller, reference), 0);
        auto *state = static_cast<NativeRunContext<DeviceRunnerBase> *>(attempt.storage);
        EXPECT_EQ(runner.admit_recording(*state->prepared_execution), 0);
        state->recording = true;
        state->completion_rc = 0;
        EXPECT_EQ(simpler_finalize_run(&runner, attempt.storage), 0);
    };
    capture(first, "a", &first_ref);
    capture(second, "b", &second_ref);
    EXPECT_EQ(control().pmu_start_calls, 1u);
    EXPECT_EQ(control().pmu_stop_calls, 0u);
    EXPECT_EQ(simpler_release_run_resources(&runner, first_ref), 0);
    EXPECT_EQ(control().pmu_stop_calls, 0u) << "the second model still protects its counting window";
    EXPECT_EQ(simpler_release_run_resources(&runner, second_ref), 0);
    EXPECT_EQ(control().pmu_stop_calls, 1u);
    std::string error;
    EXPECT_TRUE(runner.flush(&error)) << error;
}

TEST_F(NativeAdmissionTest, DuplicatePmuRetainAfterUnexecutedReleasePreservesPublishedEvidence) {
    PmuCaptureRunner runner;
    ASSERT_EQ(initialize_capture_runner(runner), 0);
    Attempt attempt(runner, 0, 9705);
    use_capture_preparation(attempt);
    diagnostic_path = "native-pmu-duplicate-after-release";
    attempt.config.enable_pmu = 2;
    std::strncpy(attempt.config.output_prefix, diagnostic_path.c_str(), sizeof(attempt.config.output_prefix) - 1);
    ASSERT_EQ(attempt.prepare(), 0);
    int caller;
    RuntimeHandle reference = nullptr;
    ASSERT_EQ(simpler_retain_run_resources(&runner, attempt.storage, &caller, &reference), 0);
    ASSERT_EQ(simpler_release_run_resources(&runner, reference), 0);
    std::string error;
    ASSERT_TRUE(runner.flush(&error)) << error;
    const auto path = std::filesystem::path(diagnostic_path) / "pmu.collection.json";
    auto contents = [&] {
        std::ifstream file(path);
        return std::string{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
    };
    const auto published = contents();
    ASSERT_FALSE(published.empty());
    const auto csv = std::filesystem::path(diagnostic_path) / "pmu.csv";
    std::ifstream csv_before(csv);
    const std::string published_csv{std::istreambuf_iterator<char>(csv_before), std::istreambuf_iterator<char>()};
    const int arm_calls = runner.arm_calls;
    const auto allocation_calls = control().allocations;
    const auto lease_starts = control().pmu_start_calls;
    reference = reinterpret_cast<void *>(1);
    EXPECT_EQ(
        simpler_retain_run_resources(&runner, attempt.storage, &caller, &reference), PTO_RUNTIME_ERR_INVALID_STATE
    );
    EXPECT_EQ(reference, nullptr);
    EXPECT_EQ(contents(), published);
    std::ifstream csv_after(csv);
    EXPECT_EQ(
        (std::string{std::istreambuf_iterator<char>(csv_after), std::istreambuf_iterator<char>()}), published_csv
    );
    EXPECT_EQ(control().allocations, allocation_calls);
    EXPECT_EQ(runner.arm_calls, arm_calls);
    EXPECT_EQ(control().pmu_start_calls, lease_starts);
    EXPECT_EQ(simpler_finalize_run(&runner, attempt.storage), 0);
}

TEST_F(NativeAdmissionTest, CaptureRetentionAcceptsAnAlreadyActiveCallerWithoutChangingOrdinaryAdmission) {
    ArgsDumpCaptureRunner runner;
    ASSERT_EQ(initialize_capture_runner(runner), 0);
    Attempt attempt(runner, 0, 9980);
    use_capture_preparation(attempt);
    ASSERT_EQ(attempt.prepare(), 0);
    int caller;
    RuntimeHandle reference = nullptr;
    auto cleanup = RAIIScopeGuard([&] {
        control().recording_active = false;
        (void)simpler_finalize_run(&runner, attempt.storage);
        if (reference != nullptr) (void)simpler_release_run_resources(&runner, reference);
    });
    control().recording_active = true;
    ASSERT_EQ(simpler_retain_run_resources(&runner, attempt.storage, &caller, &reference), 0);
    ASSERT_NE(reference, nullptr);
    EXPECT_EQ(simpler_validate_caller_stream(&runner, &caller), PTO_RUNTIME_ERR_UNSUPPORTED);
    auto *state = static_cast<NativeRunContext<DeviceRunnerBase> *>(attempt.storage);
    EXPECT_FALSE(state->recording);
    EXPECT_FALSE(state->prepared_execution->recording);
    EXPECT_EQ(simpler_launch_run_on_stream(&runner, attempt.storage, &caller, nullptr), 0);
    EXPECT_TRUE(state->recording);
    control().recording_active = false;
    ASSERT_EQ(simpler_finalize_run(&runner, attempt.storage), 0);
    ASSERT_EQ(simpler_release_run_resources(&runner, reference), 0);
    reference = nullptr;
    cleanup.dismiss();
}

TEST_F(NativeAdmissionTest, CaptureRetentionRefusesInvalidatedAndQueryFailureBeforePublishingBacking) {
    ArgsDumpCaptureRunner runner;
    ASSERT_EQ(initialize_capture_runner(runner), 0);
    Attempt attempt(runner, 0, 9981);
    use_capture_preparation(attempt);
    ASSERT_EQ(attempt.prepare(), 0);
    int caller;
    RuntimeHandle reference = nullptr;
    const size_t allocations = control().allocations;
    const size_t events = control().event_create_calls;
    control().recording_invalidated = true;
    EXPECT_EQ(simpler_retain_run_resources(&runner, attempt.storage, &caller, &reference), PTO_RUNTIME_ERR_UNSUPPORTED);
    EXPECT_EQ(reference, nullptr);
    control().recording_invalidated = false;
    control().capture_info_rc = -4255;
    EXPECT_EQ(simpler_retain_run_resources(&runner, attempt.storage, &caller, &reference), -4255);
    EXPECT_EQ(reference, nullptr);
    EXPECT_EQ(control().allocations, allocations);
    EXPECT_EQ(control().event_create_calls, events);
    control().capture_info_rc = 0;
    control().recording_active = true;
    ASSERT_EQ(simpler_retain_run_resources(&runner, attempt.storage, &caller, &reference), 0);
    auto cleanup = RAIIScopeGuard([&] {
        control().recording_active = false;
        control().recording_invalidated = false;
        control().capture_info_rc = 0;
        (void)simpler_finalize_run(&runner, attempt.storage);
        (void)simpler_release_run_resources(&runner, reference);
    });
    control().recording_invalidated = true;
    EXPECT_EQ(simpler_launch_run_on_stream(&runner, attempt.storage, &caller, nullptr), PTO_RUNTIME_ERR_UNSUPPORTED);
    control().recording_invalidated = false;
    control().capture_info_rc = -4255;
    EXPECT_EQ(simpler_launch_run_on_stream(&runner, attempt.storage, &caller, nullptr), -4255);
    EXPECT_EQ(runner.launch_calls, 0);
    auto *state = static_cast<NativeRunContext<DeviceRunnerBase> *>(attempt.storage);
    EXPECT_FALSE(state->recording);
    control().capture_info_rc = 0;
    EXPECT_EQ(simpler_launch_run_on_stream(&runner, attempt.storage, &caller, nullptr), 0);
    EXPECT_TRUE(state->recording);
}

TEST_F(NativeAdmissionTest, ActiveRetentionStillRefusesKernelEagerAndProgramPreparations) {
    for (bool kernel : {false, true}) {
        ArgsDumpCaptureRunner runner;
        if (kernel) ASSERT_EQ(initialize_capture_runner(runner), 0);
        Attempt attempt(runner, 0, kernel ? 9983 : 9982);
        attempt.args.tensor(0).address_space = AddressSpace::DEVICE;
        attempt.args.tensor(0).transfer = TensorTransfer::NONE;
        ASSERT_EQ(attempt.prepare(), 0);
        int caller;
        RuntimeHandle reference = nullptr;
        const size_t allocations = control().allocations;
        const size_t events = control().event_create_calls;
        control().recording_active = true;
        EXPECT_EQ(
            simpler_retain_run_resources(&runner, attempt.storage, &caller, &reference), PTO_RUNTIME_ERR_UNSUPPORTED
        );
        EXPECT_EQ(reference, nullptr);
        EXPECT_EQ(control().allocations, allocations);
        EXPECT_EQ(control().event_create_calls, events);
        control().recording_active = false;
        ASSERT_EQ(simpler_finalize_run(&runner, attempt.storage), 0);
    }
}

class ColdKernelRunner : public PmuCaptureRunner {
public:
    ColdKernelRunner() {
        device_id_ = -1;
        // Real Kernel init registers its binary; the PMU fixture's token is not one.
        aicore_bin_handle_ = nullptr;
    }
    size_t result_epoch_count() const {
        std::lock_guard<std::mutex> lock(run_result_mu_);
        return run_result_storage_.size();
    }
    LaunchTransactionResult launch_execution_impl(PreparedExecution &prepared, LaunchPermit permit) override {
        if (!prepared.recording) return FencedCompletionRunner::launch_execution_impl(prepared, std::move(permit));
        prepared.streams = borrowed_run_streams();
        prepared.streams.aicpu = prepared.caller_stream;
        const int rc = arm_run_fence(prepared);
        if (rc != 0) return {rc, LaunchProgress::NotStarted, {}};
        return exact_launch_transaction(
            prepared.identity, std::move(permit),
            [&](LaunchProgressSink &sink) {
                const int entry_rc = queue_run_entry(prepared, sink);
                if (entry_rc != 0) return entry_rc;
                sink.mark_submitted();
                return record_run_boundary(prepared, RunCompletionFence::StreamRole::Aicore);
            },
            [&](LaunchProgressSink &sink) {
                sink.mark_submitted();
                const int boundary_rc = record_run_boundary(prepared, RunCompletionFence::StreamRole::Aicpu);
                return boundary_rc != 0 ? boundary_rc : queue_caller_exit(prepared);
            }
        );
    }
};

namespace {
int initialize_cold_kernel_runner(ColdKernelRunner &runner, uint32_t grant) {
    if (!runner.set_run_resource_grant(grant)) return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
    control().loader_start_rc = 0;
    const uint8_t binary = 1;
    CallConfig config{};
    config.aicpu_thread_num = 2;
    for (int i = 0; i < RUNTIME_ENV_RING_COUNT; ++i) {
        config.runtime_env.ring_task_window[i] = 4;
        config.runtime_env.ring_heap[i] = 1024;
        config.runtime_env.ring_dep_pool[i] = 4;
    }
    return simpler_kernel_mode_init(&runner, 0, &binary, 1, &binary, 1, &binary, 1, &config, 1);
}
size_t uploads_to(const void *target) {
    return static_cast<size_t>(std::count_if(control().uploads.begin(), control().uploads.end(), [target](auto &copy) {
        return copy.first == target;
    }));
}
}  // namespace

TEST_F(NativeAdmissionTest, KernelInitialRuntimeReservationPrecedesTheFirstActivePrepare) {
    ColdKernelRunner runner;
    ASSERT_EQ(initialize_cold_kernel_runner(runner, 3), 0);
    Runtime empty_runtime;
    const size_t extent = runtime_device_extent_size(empty_runtime);
    RecordProperty("runtime_device_extent", static_cast<int>(extent));
    RecordProperty(
        "runtime_initialized_prefix", static_cast<int>(runtime_device_initialized_prefix_size(empty_runtime))
    );
    for (uint32_t slot = 0; slot < 3; ++slot) {
        const auto &args = runner.result_storage(slot).args;
        EXPECT_NE(args.runtime_args, nullptr);
        EXPECT_EQ(args.runtime_bytes, extent);
        EXPECT_NE(args.runtime_upload_source, nullptr);
        EXPECT_EQ(args.runtime_upload_bytes, extent);
        EXPECT_FALSE(args.workers_initialized);
        const auto *workers =
            reinterpret_cast<const std::byte *>(args.runtime_args) + offsetof(DeviceRuntimeLaunchDesc, workers);
        EXPECT_EQ(uploads_to(workers), 0u);
        EXPECT_EQ(args.published_prefix_bytes, 0u);
        EXPECT_EQ(uploads_to(args.runtime_args), 0u);
    }
    const auto initial_runtime = runner.result_storage().args.runtime_args;
    const auto allocations = static_cast<size_t>(std::count_if(
        control().allocated_device_blocks.begin(), control().allocated_device_blocks.end(),
        [extent](const auto &block) {
            return block.second == extent;
        }
    ));
    const auto pinned_allocations = control().host_allocations;
    ASSERT_EQ(runner.kernel_execution_state().mark_ready_enqueued(), 0);
    Attempt attempt(runner, 0, 10100);
    use_capture_preparation(attempt);
    int caller;
    RuntimeHandle reference = nullptr;
    auto cleanup = RAIIScopeGuard([&] {
        control().recording_active = false;
        (void)simpler_finalize_run(&runner, attempt.storage);
        if (reference != nullptr) (void)simpler_release_run_resources(&runner, reference);
    });
    control().recording_active = true;
    ASSERT_EQ(attempt.prepare(), 0);
    auto &owner = runner.result_storage().args;
    EXPECT_EQ(owner.runtime_args, initial_runtime);
    EXPECT_EQ(
        static_cast<size_t>(std::count_if(
            control().allocated_device_blocks.begin(), control().allocated_device_blocks.end(),
            [extent](const auto &block) {
                return block.second == extent;
            }
        )),
        allocations
    );
    RecordProperty("first_prepare_metadata_tail", static_cast<int>(owner.runtime_upload_bytes - extent));
    // The device descriptor is fixed; host upload storage includes this run's metadata.
    if (std::string(SIMPLER_RUNTIME_NAME) == "tensormap_and_ringbuffer") {
        EXPECT_GT(owner.runtime_upload_bytes, extent);
        EXPECT_NE(owner.runtime_upload_source, nullptr);
        EXPECT_EQ(control().host_allocations, pinned_allocations + 1);
    }
    EXPECT_TRUE(owner.workers_initialized);
    EXPECT_EQ(control().capture_mode_exchanges, 2u);
    const auto *workers =
        reinterpret_cast<const std::byte *>(owner.runtime_args) + offsetof(DeviceRuntimeLaunchDesc, workers);
    EXPECT_EQ(uploads_to(workers), 1u);
    EXPECT_EQ(uploads_to(owner.runtime_args), 0u);
    ASSERT_EQ(simpler_retain_run_resources(&runner, attempt.storage, &caller, &reference), 0);
    auto *state = static_cast<NativeRunContext<DeviceRunnerBase> *>(attempt.storage);
    const std::weak_ptr<const PreparedRunResources> backing = state->prepared_execution->prepared_resources;
    ASSERT_EQ(simpler_launch_run_on_stream(&runner, attempt.storage, &caller, nullptr), 0);
    control().recording_active = false;
    ASSERT_EQ(simpler_finalize_run(&runner, attempt.storage), 0);
    EXPECT_FALSE(backing.expired());
    ASSERT_EQ(simpler_release_run_resources(&runner, reference), 0);
    reference = nullptr;
    EXPECT_TRUE(backing.expired());
    EXPECT_EQ(runner.finalize(), 0);
    EXPECT_EQ(control().live_host_allocations, 0u);
    cleanup.dismiss();
}

TEST_F(NativeAdmissionTest, ProgramRuntimeReservationRemainsLazyUntilPrepare) {
    ColdKernelRunner runner;
    control().loader_start_rc = 0;
    const uint8_t binary = 1;
    ASSERT_EQ(simpler_init(&runner, 0, &binary, 1, &binary, 1, &binary, 1, nullptr, 0, nullptr, 0), 0);
    EXPECT_EQ(runner.result_storage().args.runtime_args, nullptr);
    EXPECT_EQ(runner.result_storage().args.runtime_upload_source, nullptr);
    EXPECT_EQ(control().host_allocations, 0u);
    Attempt attempt(runner, 0, 10101);
    ASSERT_EQ(attempt.prepare(), 0);
    EXPECT_NE(runner.result_storage().args.runtime_args, nullptr);
    EXPECT_NE(runner.result_storage().args.runtime_upload_source, nullptr);
    ASSERT_EQ(simpler_finalize_run(&runner, attempt.storage), 0);
    EXPECT_EQ(runner.finalize(), 0);
    EXPECT_EQ(control().live_host_allocations, 0u);
}

TEST_F(NativeAdmissionTest, KernelRuntimeAllocationFailureRetainsEarlierGrantedBackingForCheckedClose) {
    ColdKernelRunner runner;
    constexpr uint32_t grant = 3;
    control().fail_allocation = 2;
    EXPECT_EQ(initialize_cold_kernel_runner(runner, grant), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(runner.kernel_execution_state().phase(), KernelContextPhase::Poisoned);
    for (uint32_t slot = 0; slot < grant; ++slot) {
        EXPECT_EQ(runner.result_storage(slot).device_result, nullptr);
        EXPECT_FALSE(runner.result_storage(slot).result_initialized);
    }
    auto &owner = runner.result_storage(0).args;
    ASSERT_NE(owner.runtime_args, nullptr);
    EXPECT_NE(owner.runtime_upload_source, nullptr);
    EXPECT_EQ(runner.result_storage(1).args.runtime_args, nullptr);
    EXPECT_EQ(runner.result_storage(2).args.runtime_args, nullptr);
    Runtime *retained = owner.runtime_args;
    auto restore = RAIIScopeGuard([] {
        control().selected_free_rc = 0;
    });
    control().selected_free = retained;
    control().selected_free_rc = -761;
    EXPECT_EQ(runner.finalize(), -761);
    EXPECT_EQ(owner.runtime_args, retained);
    EXPECT_GT(control().bytes, 0u);
    control().selected_free_rc = 0;
    EXPECT_EQ(runner.finalize(), 0);
    EXPECT_EQ(control().selected_free_attempts, 2u);
    EXPECT_EQ(control().bytes, 0u);
    EXPECT_EQ(control().live_host_allocations, 0u);
}

TEST_F(NativeAdmissionTest, KernelPinnedRuntimeAllocationFailurePreservesItsExactDeviceBlock) {
    ColdKernelRunner runner;
    auto restore = RAIIScopeGuard([] {
        control().host_alloc_rc = 0;
        control().selected_free_rc = 0;
    });
    control().host_alloc_rc = -762;
    EXPECT_EQ(initialize_cold_kernel_runner(runner, 2), -762);
    EXPECT_EQ(runner.kernel_execution_state().last_runtime_error(), -762);
    EXPECT_EQ(runner.kernel_execution_state().phase(), KernelContextPhase::Poisoned);
    auto &owner = runner.result_storage().args;
    Runtime *retained = owner.runtime_args;
    ASSERT_NE(retained, nullptr);
    EXPECT_NE(owner.runtime_bytes, 0u);
    EXPECT_EQ(owner.runtime_upload_source, nullptr);
    EXPECT_FALSE(owner.workers_initialized);
    EXPECT_EQ(runner.result_storage(1).args.runtime_args, nullptr);
    control().host_alloc_rc = 0;
    control().selected_free = retained;
    control().selected_free_rc = -763;
    EXPECT_EQ(runner.finalize(), -763);
    EXPECT_EQ(owner.runtime_args, retained);
    EXPECT_EQ(control().host_allocations, 1u);
    control().selected_free_rc = 0;
    EXPECT_EQ(runner.finalize(), 0);
    EXPECT_EQ(control().selected_free_attempts, 2u);
    EXPECT_EQ(control().live_host_allocations, 0u);
}

TEST_F(NativeAdmissionTest, KernelPinnedRuntimeFreeFailureRetainsTheSameSourceForCheckedRetry) {
    ColdKernelRunner runner;
    ASSERT_EQ(initialize_cold_kernel_runner(runner, 2), 0);
    auto &owner = runner.result_storage().args;
    void *retained = owner.runtime_upload_source;
    const auto bytes = owner.runtime_upload_bytes;
    ASSERT_NE(retained, nullptr);
    auto restore = RAIIScopeGuard([] {
        control().selected_host_free_rc = 0;
    });
    control().selected_host_free = retained;
    control().selected_host_free_rc = -764;
    EXPECT_EQ(runner.finalize(), -764);
    EXPECT_EQ(owner.runtime_upload_source, retained);
    EXPECT_EQ(owner.runtime_upload_bytes, bytes);
    EXPECT_EQ(owner.runtime_args, nullptr);
    EXPECT_EQ(control().live_host_allocations, 1u);
    control().selected_host_free_rc = 0;
    EXPECT_EQ(runner.finalize(), 0);
    EXPECT_EQ(control().selected_host_free_attempts, 2u);
    EXPECT_EQ(control().live_host_allocations, 0u);
}

TEST_F(NativeAdmissionTest, RetainedRuntimeGenerationKeepsItsReservedSourceThroughSlotReuse) {
    ColdKernelRunner runner;
    ASSERT_EQ(initialize_cold_kernel_runner(runner, 2), 0);
    ASSERT_EQ(runner.kernel_execution_state().mark_ready_enqueued(), 0);
    Attempt first(runner, 0, 10102);
    use_capture_preparation(first);
    ASSERT_EQ(first.prepare(), 0);
    auto &old_args = runner.result_storage().args;
    Runtime *old_runtime = old_args.runtime_args;
    void *old_source = old_args.runtime_upload_source;
    int caller;
    RuntimeHandle reference = nullptr;
    auto cleanup = RAIIScopeGuard([&] {
        control().recording_active = false;
        (void)simpler_finalize_run(&runner, first.storage);
        if (reference != nullptr) (void)simpler_release_run_resources(&runner, reference);
    });
    ASSERT_EQ(simpler_retain_run_resources(&runner, first.storage, &caller, &reference), 0);
    control().recording_active = true;
    ASSERT_EQ(simpler_launch_run_on_stream(&runner, first.storage, &caller, nullptr), 0);
    control().recording_active = false;
    ASSERT_EQ(simpler_finalize_run(&runner, first.storage), 0);
    const auto *bytes = static_cast<const uint8_t *>(old_source);
    const std::vector<uint8_t> snapshot(bytes, bytes + 32);
    Attempt next(runner, 0, 10103);
    use_capture_preparation(next);
    ASSERT_EQ(next.prepare(), 0);
    auto next_cleanup = RAIIScopeGuard([&] {
        (void)simpler_finalize_run(&runner, next.storage);
    });
    EXPECT_NE(runner.result_storage().args.runtime_args, old_runtime);
    EXPECT_NE(runner.result_storage().args.runtime_upload_source, old_source);
    EXPECT_EQ(std::memcmp(old_source, snapshot.data(), snapshot.size()), 0);
    EXPECT_EQ(old_args.runtime_args, old_runtime);
    EXPECT_EQ(old_args.runtime_upload_source, old_source);
    ASSERT_EQ(simpler_finalize_run(&runner, next.storage), 0);
    next_cleanup.dismiss();
    EXPECT_EQ(std::memcmp(old_source, snapshot.data(), snapshot.size()), 0);
    ASSERT_EQ(simpler_release_run_resources(&runner, reference), 0);
    reference = nullptr;
    EXPECT_EQ(runner.finalize(), 0);
    EXPECT_EQ(control().live_host_allocations, 0u);
    cleanup.dismiss();
}

TEST_F(NativeAdmissionTest, FiveRecordedGenerationsKeepTheirOriginalSourcesUntilEveryReferenceEnds) {
    ColdKernelRunner runner;
    ASSERT_EQ(initialize_cold_kernel_runner(runner, 2), 0);
    ASSERT_EQ(runner.kernel_execution_state().mark_ready_enqueued(), 0);
    const size_t downloads_before = control().d2h_calls;
    std::vector<RuntimeHandle> references;
    std::vector<void *> sources;
    std::vector<Runtime *> runtimes;
    std::vector<std::vector<uint8_t>> snapshots;
    int caller;
    auto cleanup = RAIIScopeGuard([&] {
        control().recording_active = false;
        for (auto reference : references)
            (void)simpler_release_run_resources(&runner, reference);
    });
    // These are CPU transport leaves; ACTIVE here is not an SDK allocation proof.
    control().recording_active = true;
    for (uint64_t index = 0; index < 5; ++index) {
        const uint32_t slot = static_cast<uint32_t>(index % 2);
        // A different extent changes the actual serialized tensor descriptor,
        // not just a pointer or a constant prefix of the upload source.
        Attempt attempt(runner, slot, 10120 + index, 16 + index);
        attempt.bytes.front() = static_cast<uint8_t>(0x20 + index);
        use_capture_preparation(attempt);
        auto retire_attempt = RAIIScopeGuard([&] {
            (void)simpler_finalize_run(&runner, attempt.storage);
        });
        ASSERT_EQ(attempt.prepare(), 0) << index;
        EXPECT_EQ(control().capture_mode_exchanges, 2u * (index + 1)) << index;
        const auto *state = static_cast<const NativeRunContext<DeviceRunnerBase> *>(attempt.storage);
        EXPECT_EQ(state->prepared_execution->kernel_args.args.run_result_data_base, 0u) << index;
        EXPECT_EQ(state->prepared_execution->kernel_args.args.run_result_epoch, attempt.descriptor.run_epoch);
        EXPECT_EQ(runner.result_epoch_count(), 0u);
        RuntimeHandle reference = nullptr;
        ASSERT_EQ(simpler_retain_run_resources(&runner, attempt.storage, &caller, &reference), 0) << index;
        ASSERT_NE(reference, nullptr);
        references.push_back(reference);
        auto &args = runner.result_storage(slot).args;
        EXPECT_EQ(std::count(runtimes.begin(), runtimes.end(), args.runtime_args), 0);
        EXPECT_EQ(std::count(sources.begin(), sources.end(), args.runtime_upload_source), 0);
        runtimes.push_back(args.runtime_args);
        sources.push_back(args.runtime_upload_source);
        ASSERT_EQ(simpler_launch_run_on_stream(&runner, attempt.storage, &caller, nullptr), 0) << index;
        ASSERT_EQ(simpler_finalize_run(&runner, attempt.storage), 0) << index;
        retire_attempt.dismiss();
        const auto *bytes = static_cast<const uint8_t *>(sources.back());
        ASSERT_GT(args.runtime_upload_bytes, 32u);
        snapshots.emplace_back(bytes, bytes + args.runtime_upload_bytes);
        for (size_t old = 0; old + 1 < snapshots.size(); ++old)
            EXPECT_NE(snapshots[old], snapshots.back()) << "distinct tensor descriptor " << index;
        for (size_t old = 0; old < sources.size(); ++old)
            EXPECT_EQ(std::memcmp(sources[old], snapshots[old].data(), snapshots[old].size()), 0)
                << "generation " << old << " after preparing " << index;
    }
    control().recording_active = false;
    for (auto &reference : references) {
        ASSERT_EQ(simpler_release_run_resources(&runner, reference), 0);
        reference = nullptr;
    }
    references.clear();
    EXPECT_EQ(control().d2h_calls, downloads_before);
    ASSERT_EQ(runner.finalize(), 0);
    EXPECT_EQ(control().live_host_allocations, 0u);
    cleanup.dismiss();
}

TEST_F(NativeAdmissionTest, CaptureModeRestoreFailureRefusesAnAlreadyPreparedPeer) {
    ColdKernelRunner runner;
    ASSERT_EQ(initialize_cold_kernel_runner(runner, 2), 0);
    ASSERT_EQ(runner.kernel_execution_state().mark_ready_enqueued(), 0);
    Attempt old(runner, 1, 10130), peer(runner, 0, 10131), replacement(runner, 1, 10132);
    use_capture_preparation(old);
    use_capture_preparation(peer);
    use_capture_preparation(replacement);
    int caller;
    RuntimeHandle old_reference = nullptr;
    RuntimeHandle peer_reference = nullptr;
    auto cleanup = RAIIScopeGuard([&] {
        control().recording_active = false;
        if (control().capture_mode_exchanges >= 2) {
            auto mode = ACL_MODEL_RI_CAPTURE_MODE_GLOBAL;
            (void)aclmdlRICaptureThreadExchangeMode(&mode);
        }
        (void)simpler_finalize_run(&runner, old.storage);
        (void)simpler_finalize_run(&runner, peer.storage);
        (void)simpler_finalize_run(&runner, replacement.storage);
        if (peer_reference != nullptr) (void)simpler_release_run_resources(&runner, peer_reference);
        if (old_reference != nullptr) (void)simpler_release_run_resources(&runner, old_reference);
    });
    control().recording_active = true;
    ASSERT_EQ(old.prepare(), 0);
    ASSERT_EQ(simpler_retain_run_resources(&runner, old.storage, &caller, &old_reference), 0);
    ASSERT_EQ(simpler_launch_run_on_stream(&runner, old.storage, &caller, nullptr), 0);
    ASSERT_EQ(simpler_finalize_run(&runner, old.storage), 0);
    ASSERT_EQ(peer.prepare(), 0);
    ASSERT_EQ(simpler_retain_run_resources(&runner, peer.storage, &caller, &peer_reference), 0);
    control().fail_capture_mode_exchange = control().capture_mode_exchanges + 2;
    ASSERT_EQ(replacement.prepare(), -872);
    ASSERT_TRUE(replacement.consume_release());
    ASSERT_EQ(runner.kernel_execution_state().phase(), KernelContextPhase::Poisoned);
    const size_t publications = control().uploads.size();
    EXPECT_EQ(simpler_launch_run_on_stream(&runner, peer.storage, &caller, nullptr), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_EQ(control().uploads.size(), publications);
    control().recording_active = false;
    auto mode = ACL_MODEL_RI_CAPTURE_MODE_GLOBAL;
    ASSERT_EQ(aclmdlRICaptureThreadExchangeMode(&mode), 0);
    ASSERT_EQ(simpler_finalize_run(&runner, peer.storage), 0);
    EXPECT_TRUE(peer.consume_release());
    ASSERT_EQ(simpler_release_run_resources(&runner, peer_reference), 0);
    peer_reference = nullptr;
    ASSERT_EQ(simpler_release_run_resources(&runner, old_reference), 0);
    old_reference = nullptr;
    EXPECT_EQ(runner.finalize(), 0);
    EXPECT_EQ(control().live_host_allocations, 0u);
    cleanup.dismiss();
}

TEST_F(NativeAdmissionTest, KernelInitConsumesOnlyMandatoryDeviceDescriptorBytes) {
    size_t mandatory_bytes = 0;
    {
        ColdKernelRunner sizing;
        ASSERT_EQ(initialize_cold_kernel_runner(sizing, 2), 0);
        for (uint32_t slot = 0; slot < 2; ++slot)
            mandatory_bytes += sizing.result_storage(slot).args.runtime_bytes;
        ASSERT_EQ(sizing.finalize(), 0);
    }
    ASSERT_EQ(control().bytes, 0u);
    ASSERT_GT(mandatory_bytes, 0u);
    control().ceiling = mandatory_bytes;
    control().allocation_failure_rc = 207001;
    ColdKernelRunner runner;
    EXPECT_EQ(initialize_cold_kernel_runner(runner, 2), 0);
    for (uint32_t slot = 0; slot < 2; ++slot) {
        EXPECT_NE(runner.result_storage(slot).args.runtime_args, nullptr);
        EXPECT_EQ(runner.result_storage(slot).device_result, nullptr);
    }
    EXPECT_EQ(control().bytes, mandatory_bytes);
    EXPECT_EQ(runner.finalize(), 0);
}

TEST_F(NativeAdmissionTest, NativeDiagnosticPrepareBeforeBeginInitializesRotatedResult) {
    ColdKernelRunner runner;
    ASSERT_EQ(initialize_cold_kernel_runner(runner, 2), 0);
    ASSERT_EQ(runner.kernel_execution_state().mark_ready_enqueued(), 0);
    RuntimeHandle reference = nullptr;
    int caller;
    auto cleanup = RAIIScopeGuard([&] {
        control().recording_active = false;
        if (reference) (void)simpler_release_run_resources(&runner, reference);
    });
    {
        Attempt first(runner, 0, 10300);
        use_capture_preparation(first);
        ASSERT_EQ(first.prepare(), 0);
        ASSERT_EQ(simpler_retain_run_resources(&runner, first.storage, &caller, &reference), 0);
        control().recording_active = true;
        ASSERT_EQ(simpler_launch_run_on_stream(&runner, first.storage, &caller, nullptr), 0);
        control().recording_active = false;
        ASSERT_EQ(simpler_finalize_run(&runner, first.storage), 0);
    }
    const size_t allocations = control().allocations;
    Attempt next(runner, 0, 10301);
    use_capture_preparation(next);
    next.config.enable_pmu = 1;
    std::strcpy(next.config.output_prefix, "requested-result");
    const auto record_allocations = [] {
        return std::count_if(
            control().allocated_device_blocks.begin(), control().allocated_device_blocks.end(), [](const auto &entry) {
                return entry.second == sizeof(NativeInvocationResultStorage);
            }
        );
    };
    const auto records_before = record_allocations();
    const auto exchanges = control().capture_mode_exchanges;
    ASSERT_FALSE(control().recording_active);
    ASSERT_EQ(next.prepare(), 0);
    EXPECT_EQ(record_allocations(), records_before + 1);
    EXPECT_GE(control().allocations, allocations);
    EXPECT_EQ(control().capture_mode_exchanges, exchanges);
    ASSERT_EQ(simpler_finalize_run(&runner, next.storage), 0);
    std::array<DeviceRunnerBase::NativeRunReservationRow, 2> reservations{};
    EXPECT_EQ(runner.snapshot_native_run_reservations(reservations.data(), reservations.size()), 0u);
    ASSERT_EQ(simpler_release_run_resources(&runner, reference), 0);
    reference = nullptr;
    EXPECT_EQ(runner.finalize(), 0);
}

TEST_F(NativeAdmissionTest, FailedEagerResultAllocationCannotBecomeUsableBeforeCheckedFree) {
    ColdKernelRunner runner;
    ASSERT_EQ(initialize_cold_kernel_runner(runner, 2), 0);
    ASSERT_EQ(runner.kernel_execution_state().mark_ready_enqueued(), 0);
    RuntimeHandle reference = nullptr;
    int caller;
    auto cleanup = RAIIScopeGuard([&] {
        control().failed_allocation_bytes = 0;
        control().selected_free_rc = 0;
        control().recording_active = false;
        if (reference) (void)simpler_release_run_resources(&runner, reference);
    });
    {
        Attempt old(runner, 0, 10310);
        use_capture_preparation(old);
        ASSERT_EQ(old.prepare(), 0);
        ASSERT_EQ(simpler_retain_run_resources(&runner, old.storage, &caller, &reference), 0);
        control().recording_active = true;
        ASSERT_EQ(simpler_launch_run_on_stream(&runner, old.storage, &caller, nullptr), 0);
        control().recording_active = false;
        ASSERT_EQ(simpler_finalize_run(&runner, old.storage), 0);
    }
    control().failed_allocation_bytes = sizeof(NativeInvocationResultStorage);
    control().allocation_failure_with_storage = true;
    control().allocation_failure_rc = -747;
    {
        Attempt failed(runner, 0, 10311);
        failed.args.tensor(0).address_space = AddressSpace::DEVICE;
        failed.args.tensor(0).transfer = TensorTransfer::NONE;
        failed.config.enable_pmu = 1;
        std::strcpy(failed.config.output_prefix, "requested-eager-result");
        ASSERT_EQ(failed.prepare(), -747);
        ASSERT_TRUE(failed.consume_release());
    }
    auto &owner = runner.result_storage();
    void *failed_address = owner.device_result;
    ASSERT_NE(failed_address, nullptr);
    EXPECT_FALSE(owner.result_initialized);
    EXPECT_EQ(uploads_to(failed_address), 0u);
    control().failed_allocation_bytes = 0;
    control().selected_free = failed_address;
    control().selected_free_rc = -751;
    {
        Attempt retry(runner, 0, 10312);
        retry.args.tensor(0).address_space = AddressSpace::DEVICE;
        retry.args.tensor(0).transfer = TensorTransfer::NONE;
        retry.config.enable_pmu = 1;
        std::strcpy(retry.config.output_prefix, "requested-eager-result");
        const int rc = retry.prepare();
        EXPECT_EQ(rc, -751);
        EXPECT_EQ(owner.device_result, failed_address);
        EXPECT_FALSE(owner.result_initialized);
        EXPECT_EQ(uploads_to(failed_address), 0u);
        EXPECT_EQ(control().selected_free_attempts, 1u);
        if (rc == 0) EXPECT_EQ(simpler_finalize_run(&runner, retry.storage), 0);
        EXPECT_TRUE(retry.consume_release());
    }
    control().selected_free_rc = 0;
    {
        Attempt retry(runner, 0, 10313);
        retry.args.tensor(0).address_space = AddressSpace::DEVICE;
        retry.args.tensor(0).transfer = TensorTransfer::NONE;
        retry.config.enable_pmu = 1;
        std::strcpy(retry.config.output_prefix, "requested-eager-result");
        ASSERT_EQ(retry.prepare(), 0);
        EXPECT_TRUE(owner.result_initialized);
        EXPECT_EQ(control().selected_free_attempts, 2u);
        ASSERT_EQ(simpler_finalize_run(&runner, retry.storage), 0);
    }
    ASSERT_EQ(simpler_release_run_resources(&runner, reference), 0);
    reference = nullptr;
    EXPECT_EQ(runner.finalize(), 0);
    cleanup.dismiss();
}

TEST_F(NativeAdmissionTest, KernelInitWithoutDiagnosticsDoesNotAllocateResultStorage) {
    ColdKernelRunner runner;
    control().failed_allocation_bytes = sizeof(NativeInvocationResultStorage);
    control().allocation_failure_rc = -747;
    ASSERT_EQ(initialize_cold_kernel_runner(runner, 2), 0);
    for (uint32_t slot = 0; slot < 2; ++slot) {
        EXPECT_NE(runner.result_storage(slot).args.runtime_args, nullptr);
        EXPECT_EQ(runner.result_storage(slot).device_result, nullptr);
    }
    EXPECT_NE(runner.kernel_execution_state().phase(), KernelContextPhase::Poisoned);
    control().failed_allocation_bytes = 0;
    EXPECT_EQ(runner.finalize(), 0);
}

TEST_F(NativeAdmissionTest, ProgramKeepsItsRecordAndPreservesAllocationFailureContract) {
    FencedCompletionRunner runner;
    ASSERT_EQ(runner.execution_mode_latch().latch(SIMPLER_MODE_PROGRAM), 0);
    control().failed_allocation_bytes = sizeof(NativeInvocationResultStorage);
    control().allocation_failure_rc = -752;
    {
        Attempt failed(runner, 0, 10320);
        EXPECT_EQ(failed.prepare(), PTO_RUNTIME_ERR_INTERNAL);
        EXPECT_TRUE(failed.consume_release());
        EXPECT_EQ(runner.result_storage().device_result, nullptr);
    }
    control().failed_allocation_bytes = 0;
    {
        Attempt succeeding(runner, 0, 10321);
        ASSERT_EQ(succeeding.prepare(), 0);
        const auto *state = static_cast<const NativeRunContext<DeviceRunnerBase> *>(succeeding.storage);
        EXPECT_NE(state->prepared_execution->kernel_args.args.run_result_data_base, 0u);
        EXPECT_EQ(state->prepared_execution->kernel_args.args.run_result_epoch, succeeding.descriptor.run_epoch);
        EXPECT_EQ(simpler_finalize_run(&runner, succeeding.storage), 0);
    }
    EXPECT_EQ(runner.finalize(), 0);
}

TEST_F(NativeAdmissionTest, ProgramResultCopyFailureKeepsItsRawCodeAndValidBackingForRetry) {
    class FailedCopyRunner : public FencedCompletionRunner {
    public:
        int prepare_execution(
            Runtime &runtime, const CallConfig &config, uint32_t slot, const NativeRunIdentity &identity,
            std::unique_ptr<PreparedExecution> *out
        ) override {
            control().copy_rc = std::exchange(copy_failure, 0);
            auto reset = RAIIScopeGuard([] {
                control().copy_rc = 0;
            });
            return FencedCompletionRunner::prepare_execution(runtime, config, slot, identity, out);
        }
        int copy_failure{-753};
    } runner;
    ASSERT_EQ(runner.execution_mode_latch().latch(SIMPLER_MODE_PROGRAM), 0);
    void *backing = nullptr;
    {
        Attempt failed(runner, 0, 10322);
        EXPECT_EQ(failed.prepare(), -753);
        EXPECT_TRUE(failed.consume_release());
        backing = runner.result_storage().device_result;
        ASSERT_NE(backing, nullptr);
        EXPECT_EQ(runner.result_storage().result_allocation_rc, 0);
        EXPECT_FALSE(runner.result_storage().result_initialized);
        ASSERT_FALSE(control().uploads.empty());
        EXPECT_EQ(control().uploads.back().first, backing);
    }
    {
        Attempt succeeding(runner, 0, 10323);
        ASSERT_EQ(succeeding.prepare(), 0);
        const auto *state = static_cast<const NativeRunContext<DeviceRunnerBase> *>(succeeding.storage);
        EXPECT_EQ(
            state->prepared_execution->kernel_args.args.run_result_data_base, reinterpret_cast<uint64_t>(backing)
        );
        EXPECT_TRUE(runner.result_storage().result_initialized);
        EXPECT_EQ(simpler_finalize_run(&runner, succeeding.storage), 0);
    }
    EXPECT_EQ(runner.finalize(), 0);
}

TEST_F(NativeAdmissionTest, DefaultAllocatorReturnsNullButTracksFailedNonNullOutput) {
    class CheckedRunner : public Runner {
    public:
        int checked_free(void *address) { return mem_alloc_.free(address); }
    } runner;
    control().failed_allocation_bytes = 64;
    control().allocation_failure_rc = -749;
    control().allocation_failure_with_storage = true;
    EXPECT_EQ(runner.allocate_tensor(64), nullptr);
    ASSERT_FALSE(control().allocated_device_blocks.empty());
    void *address = control().allocated_device_blocks.back().first;
    control().selected_free = address;
    control().selected_free_rc = -750;
    EXPECT_EQ(runner.checked_free(address), -750);
    EXPECT_EQ(control().bytes, 64u);
    control().selected_free_rc = 0;
    EXPECT_EQ(runner.checked_free(address), 0);
    EXPECT_EQ(control().bytes, 0u);
}

TEST_F(NativeAdmissionTest, NoRecordKernelStillRejectsMissingEpochAndInvalidSlot) {
    class ResultRunner : public ColdKernelRunner {
    public:
        using DeviceRunnerBase::ensure_device_run_result_region;
    } runner;
    ASSERT_EQ(initialize_cold_kernel_runner(runner, 2), 0);
    KernelArgsHelper args;
    const size_t allocations = control().allocations;
    EXPECT_NE(runner.ensure_device_run_result_region(NativeRunIdentity{0, 1, 1, 0}, args, {}), 0);
    EXPECT_NE(runner.ensure_device_run_result_region(NativeRunIdentity{1, 1, 1, 2}, args, {}), 0);
    EXPECT_EQ(control().allocations, allocations);
    EXPECT_EQ(runner.result_epoch_count(), 0u);
    EXPECT_EQ(runner.finalize(), 0);
}

TEST_F(NativeAdmissionTest, DiagnosticResultCopyFailureIsReportedOnlyWhenRequested) {
    ColdKernelRunner runner;
    control().copy_rc = 207001;
    control().copy_error_bytes = sizeof(NativeInvocationResultStorage);
    ASSERT_EQ(initialize_cold_kernel_runner(runner, 2), 0);
    ASSERT_EQ(runner.kernel_execution_state().mark_ready_enqueued(), 0);
    EXPECT_EQ(runner.result_storage().device_result, nullptr);

    Attempt attempt(runner, 0, 10323);
    use_capture_preparation(attempt);
    attempt.config.enable_pmu = 1;
    std::strcpy(attempt.config.output_prefix, "requested-result-copy");
    EXPECT_EQ(attempt.prepare(), 207001);
    EXPECT_TRUE(attempt.consume_release());
    EXPECT_NE(runner.result_storage().device_result, nullptr);
    EXPECT_FALSE(runner.result_storage().result_initialized);
    control().copy_rc = 0;
    control().copy_error_bytes = 0;
    EXPECT_EQ(runner.finalize(), 0);
}

TEST_F(NativeAdmissionTest, RetainedCallerWritesSurviveSlotReuseAndOldReferenceRelease) {
    ColdKernelRunner runner;
    ASSERT_EQ(initialize_cold_kernel_runner(runner, 2), 0);
    ASSERT_EQ(runner.kernel_execution_state().mark_ready_enqueued(), 0);
    void *a = runner.allocate_caller_buffer(64);
    void *b = runner.allocate_caller_buffer(64);
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);
    CallerDeviceBuffers::Span span_a{reinterpret_cast<uint64_t>(a), 64};
    CallerDeviceBuffers::Span span_b{reinterpret_cast<uint64_t>(b), 64};
    RuntimeHandle ar = nullptr, br = nullptr;
    void *old_source = nullptr;
    int caller;
    auto cleanup = RAIIScopeGuard([&] {
        control().recording_active = false;
        control().selected_host_free_rc = 0;
        control().selected_host_free = nullptr;
        if (ar) (void)simpler_release_run_resources(&runner, ar);
        if (br) (void)simpler_release_run_resources(&runner, br);
        runner.release_caller_buffers(1, false);
        if (a) (void)runner.free_caller_buffer(a);
        if (b) (void)runner.free_caller_buffer(b);
    });
    auto record = [&](const CallerDeviceBuffers::Span &span, uint64_t epoch, RuntimeHandle &reference) {
        if (!runner.borrow_caller_buffers(1, &span, 1)) return false;
        Attempt attempt(runner, 0, epoch);
        use_capture_preparation(attempt);
        if (attempt.prepare() != 0) return false;
        auto retire = RAIIScopeGuard([&] {
            (void)simpler_finalize_run(&runner, attempt.storage);
        });
        if (!runner.declare_caller_buffer_writes(1, &span, 1)) return false;
        if (simpler_retain_run_resources(&runner, attempt.storage, &caller, &reference) != 0) return false;
        if (old_source == nullptr) old_source = runner.result_storage(0).args.runtime_upload_source;
        // The current invocation's retained alias is not a different producer.
        EXPECT_FALSE(runner.caller_buffer_written_by_other_run(1, span.addr, span.bytes));
        control().recording_active = true;
        const int launch = simpler_launch_run_on_stream(&runner, attempt.storage, &caller, nullptr);
        control().recording_active = false;
        if (launch != 0 || simpler_finalize_run(&runner, attempt.storage) != 0) return false;
        retire.dismiss();
        return true;
    };
    ASSERT_TRUE(record(span_a, 10140, ar));
    ASSERT_NE(old_source, nullptr);
    // Same slot, different invocation: the old model is still a producer.
    EXPECT_TRUE(runner.caller_buffer_written_by_other_run(1, span_a.addr, span_a.bytes));
    ASSERT_TRUE(record(span_b, 10141, br));
    EXPECT_TRUE(runner.caller_buffer_written_by_other_run(1, span_a.addr, span_a.bytes));
    const int free_a = runner.free_caller_buffer(a);
    EXPECT_EQ(free_a, PTO_RUNTIME_ERR_INVALID_STATE);
    if (free_a == 0) a = nullptr;
    // The common release has already ended its external-use obligation and
    // reset the resources pointer when reclaim hits this real backing failure.
    control().selected_host_free = old_source;
    control().selected_host_free_rc = -919;
    ASSERT_EQ(simpler_release_run_resources(&runner, ar), -919);
    EXPECT_TRUE(runner.caller_buffer_written_by_other_run(1, span_a.addr, span_a.bytes));
    EXPECT_EQ(runner.free_caller_buffer(a), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_TRUE(runner.caller_buffer_written_by_other_run(1, span_b.addr, span_b.bytes));
    control().selected_host_free_rc = 0;
    ASSERT_EQ(simpler_release_run_resources(&runner, ar), 0);
    ar = nullptr;
    EXPECT_TRUE(runner.caller_buffer_written_by_other_run(1, span_b.addr, span_b.bytes));
    EXPECT_EQ(runner.free_caller_buffer(b), PTO_RUNTIME_ERR_INVALID_STATE);
    if (a) {
        EXPECT_EQ(runner.free_caller_buffer(a), 0);
        a = nullptr;
    }
    ASSERT_EQ(simpler_release_run_resources(&runner, br), 0);
    br = nullptr;
    EXPECT_EQ(runner.free_caller_buffer(b), 0);
    b = nullptr;
    EXPECT_EQ(runner.finalize(), 0);
    cleanup.dismiss();
}

TEST_F(NativeAdmissionTest, OrdinaryExternalReferenceKeepsItsStagingWithoutARecordedCallerSnapshot) {
    class ResourceRunner : public FencedCompletionRunner {
    public:
        ResourceRunner() {
            static int binary;
            aicore_bin_handle_ = &binary;
        }
    } runner;
    ASSERT_EQ(runner.execution_mode_latch().latch(SIMPLER_MODE_PROGRAM), 0);
    void *buffer = runner.allocate_caller_buffer(64);
    ASSERT_NE(buffer, nullptr);
    const CallerDeviceBuffers::Span span{reinterpret_cast<uint64_t>(buffer), 64};
    RuntimeHandle reference = nullptr;
    int caller;
    auto cleanup = RAIIScopeGuard([&] {
        control().selected_free_rc = 0;
        control().selected_free = nullptr;
        if (reference) (void)simpler_release_run_resources(&runner, reference);
        runner.release_caller_buffers(1, false);
        if (buffer) (void)runner.free_caller_buffer(buffer);
    });
    ASSERT_TRUE(runner.borrow_caller_buffers(1, &span, 1));
    Attempt attempt(runner, 0, 10199);
    ASSERT_EQ(attempt.prepare(), 0);
    ASSERT_TRUE(runner.declare_caller_buffer_writes(1, &span, 1));
    ASSERT_EQ(simpler_retain_run_resources(&runner, attempt.storage, &caller, &reference), 0);
    void *staging = nullptr;
    size_t staging_size = 0;
    runner.get_retained_temp_buffer(0, &staging, &staging_size);
    ASSERT_NE(staging, nullptr);
    ASSERT_GT(staging_size, 0u);
    const auto *staging_bytes = static_cast<const uint8_t *>(staging);
    const std::vector<uint8_t> snapshot(staging_bytes, staging_bytes + staging_size);
    control().selected_free = staging;
    control().selected_free_attempts = 0;
    ASSERT_EQ(simpler_finalize_run(&runner, attempt.storage), 0);
    EXPECT_EQ(control().selected_free_attempts, 0u);
    // Ordinary I/O keeps its original caller validity obligation. Retaining
    // internal backing does not introduce a recorded model's caller ledger.
    runner.release_caller_buffers(1, false);
    EXPECT_EQ(runner.caller_buffer_borrow_count(), 0u);
    EXPECT_FALSE(runner.caller_buffer_written_by_other_run(0, span.addr, span.bytes));
    EXPECT_EQ(std::memcmp(staging, snapshot.data(), snapshot.size()), 0);
    EXPECT_EQ(runner.free_caller_buffer(buffer), 0);
    buffer = nullptr;
    EXPECT_EQ(runner.finalize(), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_EQ(control().selected_free_attempts, 0u);
    ASSERT_EQ(simpler_release_run_resources(&runner, reference), 0);
    reference = nullptr;
    // The current slot cache is another existing owner: its final teardown,
    // after the external reference ended, is where this allocation is freed.
    EXPECT_EQ(runner.finalize(), 0);
    EXPECT_GT(control().selected_free_attempts, 0u);
    control().selected_free = nullptr;
    cleanup.dismiss();
}

TEST_F(NativeAdmissionTest, EagerRetirementOfCapturePreparationCannotEraseItsSuccessorOnLateFinalize) {
    class EagerRunner : public ColdKernelRunner {
    public:
        LaunchTransactionResult launch_execution_impl(PreparedExecution &prepared, LaunchPermit permit) override {
            auto result = ColdKernelRunner::launch_execution_impl(prepared, std::move(permit));
            // Like the existing ScopeCaptureRunner CPU publisher, complete
            // the real caller-order boundary after the fake device work.
            if (result.rc == 0) EXPECT_EQ(queue_caller_exit(prepared), 0);
            return result;
        }
    } runner;
    ASSERT_EQ(initialize_cold_kernel_runner(runner, 2), 0);
    ASSERT_EQ(runner.kernel_execution_state().mark_ready_enqueued(), 0);
    void *a = runner.allocate_caller_buffer(64);
    void *b = runner.allocate_caller_buffer(64);
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);
    const CallerDeviceBuffers::Span sa{reinterpret_cast<uint64_t>(a), 64}, sb{reinterpret_cast<uint64_t>(b), 64};
    RuntimeHandle ar = nullptr, br = nullptr;
    int caller;
    auto cleanup = RAIIScopeGuard([&] {
        if (ar) (void)simpler_release_run_resources(&runner, ar);
        if (br) (void)simpler_release_run_resources(&runner, br);
        runner.release_caller_buffers(1, false);
        if (a) (void)runner.free_caller_buffer(a);
        if (b) (void)runner.free_caller_buffer(b);
    });
    Attempt first(runner, 0, 10190);
    use_capture_preparation(first);
    ASSERT_TRUE(runner.borrow_caller_buffers(1, &sa, 1));
    ASSERT_EQ(first.prepare(), 0);
    ASSERT_TRUE(runner.declare_caller_buffer_writes(1, &sa, 1));
    ASSERT_EQ(simpler_retain_run_resources(&runner, first.storage, &caller, &ar), 0);
    ASSERT_EQ(simpler_launch_run_on_stream(&runner, first.storage, &caller, nullptr), 0);
    ASSERT_EQ(simpler_poll_run(&runner, first.storage), SIMPLER_NATIVE_RUN_POLL_COMPLETE);
    // Actual execution retirement released the reservation, before finalize.
    EXPECT_TRUE(runner.caller_buffer_written_by_other_run(1, sa.addr, sa.bytes));
    Attempt next(runner, 0, 10191);
    use_capture_preparation(next);
    ASSERT_TRUE(runner.borrow_caller_buffers(1, &sb, 1));
    ASSERT_EQ(next.prepare(), 0);
    ASSERT_TRUE(runner.declare_caller_buffer_writes(1, &sb, 1));
    ASSERT_EQ(simpler_retain_run_resources(&runner, next.storage, &caller, &br), 0);
    ASSERT_EQ(simpler_finalize_run(&runner, first.storage), 0);
    EXPECT_EQ(runner.free_caller_buffer(b), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_TRUE(runner.caller_buffer_written_by_other_run(2, sb.addr, sb.bytes));
    ASSERT_EQ(simpler_release_run_resources(&runner, ar), 0);
    ar = nullptr;
    EXPECT_EQ(runner.free_caller_buffer(b), PTO_RUNTIME_ERR_INVALID_STATE);
    ASSERT_EQ(simpler_finalize_run(&runner, next.storage), 0);
    ASSERT_EQ(simpler_release_run_resources(&runner, br), 0);
    br = nullptr;
    EXPECT_EQ(runner.free_caller_buffer(a), 0);
    a = nullptr;
    EXPECT_EQ(runner.free_caller_buffer(b), 0);
    b = nullptr;
    ASSERT_EQ(runner.finalize(), 0);
    cleanup.dismiss();
}

TEST_F(NativeAdmissionTest, FailedRetainedLedgerPublicationKeepsTheCarrierAndCanRetry) {
    class ResourceRunner : public ColdKernelRunner {
    public:
        size_t external_count() const { return external_run_references_.size(); }
    };
    size_t injected = 0;
    for (size_t cut = 0; cut < 24; ++cut) {
        SCOPED_TRACE(cut);
        ResourceRunner runner;
        ASSERT_EQ(initialize_cold_kernel_runner(runner, 2), 0);
        ASSERT_EQ(runner.kernel_execution_state().mark_ready_enqueued(), 0);
        void *buffer = runner.allocate_caller_buffer(64);
        ASSERT_NE(buffer, nullptr);
        const CallerDeviceBuffers::Span span{reinterpret_cast<uint64_t>(buffer), 64};
        ASSERT_TRUE(runner.borrow_caller_buffers(1, &span, 1));
        Attempt attempt(runner, 0, 10160 + cut);
        use_capture_preparation(attempt);
        ASSERT_EQ(attempt.prepare(), 0);
        auto cleanup = RAIIScopeGuard([&] {
            refuse_host_allocation_after = std::numeric_limits<size_t>::max();
            (void)simpler_finalize_run(&runner, attempt.storage);
            runner.release_caller_buffers(1, false);
            if (buffer) (void)runner.free_caller_buffer(buffer);
        });
        ASSERT_TRUE(runner.declare_caller_buffer_writes(1, &span, 1));
        int caller;
        RuntimeHandle reference = nullptr;
        const size_t before = refused_host_allocation_count;
        refuse_host_allocation_after = cut;
        const int rc = simpler_retain_run_resources(&runner, attempt.storage, &caller, &reference);
        refuse_host_allocation_after = std::numeric_limits<size_t>::max();
        if (refused_host_allocation_count != before) {
            ++injected;
            EXPECT_NE(rc, 0);
            EXPECT_EQ(reference, nullptr);
            EXPECT_EQ(runner.external_count(), 0u);
            EXPECT_FALSE(runner.caller_buffer_written_by_other_run(1, span.addr, span.bytes));
            EXPECT_EQ(runner.free_caller_buffer(buffer), PTO_RUNTIME_ERR_INVALID_STATE);
            ASSERT_EQ(simpler_retain_run_resources(&runner, attempt.storage, &caller, &reference), 0);
        } else {
            ASSERT_EQ(rc, 0);
        }
        ASSERT_NE(reference, nullptr);
        auto release = RAIIScopeGuard([&] {
            (void)simpler_release_run_resources(&runner, reference);
        });
        ASSERT_EQ(runner.external_count(), 1u);
        ASSERT_EQ(simpler_finalize_run(&runner, attempt.storage), 0);
        EXPECT_TRUE(runner.caller_buffer_written_by_other_run(1, span.addr, span.bytes));
        ASSERT_EQ(simpler_release_run_resources(&runner, reference), 0);
        release.dismiss();
        EXPECT_EQ(runner.external_count(), 0u);
        EXPECT_EQ(runner.free_caller_buffer(buffer), 0);
        buffer = nullptr;
        ASSERT_EQ(runner.finalize(), 0);
        cleanup.dismiss();
    }
    EXPECT_GE(injected, 6u);
}

TEST_F(NativeAdmissionTest, EarlyReferenceEndAndLateCarrierIdentityCannotEraseAReusedSlot) {
    ColdKernelRunner runner;
    ASSERT_EQ(initialize_cold_kernel_runner(runner, 2), 0);
    ASSERT_EQ(runner.kernel_execution_state().mark_ready_enqueued(), 0);
    void *a = runner.allocate_caller_buffer(64);
    void *b = runner.allocate_caller_buffer(64);
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);
    CallerDeviceBuffers::Span sa{reinterpret_cast<uint64_t>(a), 64};
    CallerDeviceBuffers::Span sb{reinterpret_cast<uint64_t>(b), 64};
    void *old_owner = nullptr;
    NativeRunIdentity old_identity{};
    int caller;
    auto cleanup = RAIIScopeGuard([&] {
        runner.release_caller_buffers(1, false);
        if (a) (void)runner.free_caller_buffer(a);
        if (b) (void)runner.free_caller_buffer(b);
    });
    {
        ASSERT_TRUE(runner.borrow_caller_buffers(1, &sa, 1));
        Attempt first(runner, 0, 10150);
        use_capture_preparation(first);
        ASSERT_EQ(first.prepare(), 0);
        old_owner = first.storage;
        old_identity = NativeRunIdentity{
            first.descriptor.run_epoch, first.descriptor.generation, first.descriptor.dispatch_id,
            first.descriptor.pipeline_slot
        };
        ASSERT_TRUE(runner.declare_caller_buffer_writes(1, &sa, 1));
        RuntimeHandle reference = nullptr;
        ASSERT_EQ(simpler_retain_run_resources(&runner, first.storage, &caller, &reference), 0);
        ASSERT_EQ(simpler_release_run_resources(&runner, reference), 0);
        // Ending an external reference does not end this still-prepared carrier.
        EXPECT_EQ(runner.free_caller_buffer(a), PTO_RUNTIME_ERR_INVALID_STATE);
        EXPECT_TRUE(runner.caller_buffer_written_by_other_run(2, sa.addr, sa.bytes));
        ASSERT_EQ(simpler_finalize_run(&runner, first.storage), 0);
    }
    EXPECT_EQ(runner.free_caller_buffer(a), 0);
    a = nullptr;
    ASSERT_TRUE(runner.borrow_caller_buffers(1, &sb, 1));
    Attempt next(runner, 0, 10151);
    use_capture_preparation(next);
    ASSERT_EQ(next.prepare(), 0);
    auto retire_next = RAIIScopeGuard([&] {
        (void)simpler_finalize_run(&runner, next.storage);
    });
    ASSERT_TRUE(runner.declare_caller_buffer_writes(1, &sb, 1));
    EXPECT_FALSE(runner.retire_caller_buffer_carrier(old_owner, old_identity));
    EXPECT_FALSE(runner.retire_caller_buffer_carrier(next.storage, old_identity));
    EXPECT_EQ(runner.free_caller_buffer(b), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_TRUE(runner.caller_buffer_written_by_other_run(2, sb.addr, sb.bytes));
    // This genuine reservation has no reference at all; its own checked close can retire it.
    ASSERT_EQ(simpler_finalize_run(&runner, next.storage), 0);
    retire_next.dismiss();
    EXPECT_EQ(runner.free_caller_buffer(b), 0);
    b = nullptr;
    EXPECT_EQ(runner.finalize(), 0);
    cleanup.dismiss();
}

TEST_F(NativeAdmissionTest, NativeDiagnosticPrepareReusesResultBackingAndKeepsReplayEvidence) {
    ColdKernelRunner runner;
    ASSERT_EQ(initialize_cold_kernel_runner(runner, 3), 0);
    EXPECT_EQ(runner.result_epoch_count(), 0u);
    for (uint32_t slot = 0; slot < 3; ++slot)
        EXPECT_EQ(runner.result_storage(slot).device_result, nullptr);
    ASSERT_EQ(runner.kernel_execution_state().mark_ready_enqueued(), 0);
    diagnostic_path = "kernel-initial-result-history";
    ASSERT_TRUE(std::filesystem::create_directory(diagnostic_path));
    {
        Attempt setup(runner, 0, 10007);
        use_capture_preparation(setup);
        setup.config.enable_pmu = 1;
        std::strcpy(setup.config.output_prefix, diagnostic_path.c_str());
        ASSERT_EQ(setup.prepare(), 0);
        ASSERT_EQ(simpler_finalize_run(&runner, setup.storage), 0);
    }
    void *initial_result = runner.result_storage().device_result;
    ASSERT_NE(initial_result, nullptr);
    const size_t uploads_before = uploads_to(initial_result);
    if (initial_result != nullptr) {
        auto &evidence = static_cast<NativeInvocationResultStorage *>(initial_result)->evidence;
        const uint8_t payload[] = {0x5a, 0xa6};
        evidence.begin_after_exit();
        evidence.complete({DeviceRunVerdict::Error, -744, DeviceRunCodeSource::Header}, payload, sizeof(payload), true);
    }
    const size_t initial_events = control().event_create_calls;
    const size_t initial_binaries = control().binary_register_calls;
    Attempt attempt(runner, 0, 10008);
    use_capture_preparation(attempt);
    attempt.config.enable_pmu = 1;
    std::strcpy(attempt.config.output_prefix, diagnostic_path.c_str());
    RuntimeHandle reference = nullptr;
    auto cleanup = RAIIScopeGuard([&] {
        control().recording_active = false;
        (void)simpler_finalize_run(&runner, attempt.storage);
        if (reference != nullptr) (void)simpler_release_run_resources(&runner, reference);
    });
    ASSERT_FALSE(control().recording_active);
    ASSERT_EQ(attempt.prepare(), 0);
    EXPECT_EQ(runner.result_epoch_count(), 1u);
    auto *state = static_cast<NativeRunContext<DeviceRunnerBase> *>(attempt.storage);
    EXPECT_EQ(
        state->prepared_execution->kernel_args.args.run_result_data_base, reinterpret_cast<uint64_t>(initial_result)
    );
    EXPECT_EQ(state->prepared_execution->kernel_args.args.run_result_epoch, attempt.descriptor.run_epoch);
    EXPECT_EQ(uploads_to(runner.result_storage().device_result), uploads_before);
    EXPECT_EQ(control().event_create_calls, initial_events);
    EXPECT_EQ(control().binary_register_calls, initial_binaries);
    if (initial_result != nullptr) {
        const auto evidence = static_cast<NativeInvocationResultStorage *>(initial_result)->evidence.read_after_exit();
        EXPECT_EQ(evidence.kind, NativeInvocationEvidence::Kind::Error);
        EXPECT_EQ(evidence.terminal.code, -744);
        EXPECT_EQ(evidence.fault_count, 1u);
    }
    int caller;
    ASSERT_EQ(simpler_retain_run_resources(&runner, attempt.storage, &caller, &reference), 0);
    const std::weak_ptr<const PreparedRunResources> backing = state->prepared_execution->prepared_resources;
    control().recording_active = true;
    ASSERT_EQ(simpler_launch_run_on_stream(&runner, attempt.storage, &caller, nullptr), 0);
    control().recording_active = false;
    ASSERT_EQ(simpler_finalize_run(&runner, attempt.storage), 0);
    EXPECT_FALSE(backing.expired());
    EXPECT_EQ(runner.result_storage().device_result, initial_result);
    ASSERT_EQ(simpler_release_run_resources(&runner, reference), 0);
    reference = nullptr;
    EXPECT_TRUE(backing.expired());
    EXPECT_EQ(runner.finalize(), 0);
    cleanup.dismiss();
}

TEST_F(NativeAdmissionTest, KernelInitializationPrecreatesFixedResourcesForTheActualGrant) {
    ColdKernelRunner runner;
    ASSERT_TRUE(runner.set_run_resource_grant(3));
    control().loader_start_rc = 0;
    const uint8_t binary = 1;
    CallConfig config{};
    config.aicpu_thread_num = 2;
    for (int i = 0; i < RUNTIME_ENV_RING_COUNT; ++i) {
        config.runtime_env.ring_task_window[i] = 4;
        config.runtime_env.ring_heap[i] = 1024;
        config.runtime_env.ring_dep_pool[i] = 4;
    }
    ASSERT_EQ(simpler_kernel_mode_init(&runner, 0, &binary, 1, &binary, 1, &binary, 1, &config, 1), 0);
    EXPECT_EQ(control().binary_register_calls, 1u);
    EXPECT_EQ(control().event_create_calls, static_cast<size_t>(KernelEventKind::Count) + 5u * 3u);
    EXPECT_TRUE(runner.set_run_resource_grant(3));
    EXPECT_FALSE(runner.set_run_resource_grant(2));
    EXPECT_EQ(control().event_create_calls, static_cast<size_t>(KernelEventKind::Count) + 5u * 3u);
    ASSERT_EQ(runner.kernel_execution_state().mark_ready_enqueued(), 0);
    Attempt attempt(runner, 0, 9984);
    use_capture_preparation(attempt);
    ASSERT_EQ(attempt.prepare(), 0);
    int caller;
    RuntimeHandle reference = nullptr;
    control().recording_active = true;
    const int retain_rc = simpler_retain_run_resources(&runner, attempt.storage, &caller, &reference);
    EXPECT_EQ(retain_rc, 0);
    if (retain_rc == 0) EXPECT_EQ(simpler_launch_run_on_stream(&runner, attempt.storage, &caller, nullptr), 0);
    EXPECT_EQ(control().binary_register_calls, 1u);
    EXPECT_EQ(control().event_create_calls, static_cast<size_t>(KernelEventKind::Count) + 5u * 3u);
    control().recording_active = false;
    ASSERT_EQ(simpler_finalize_run(&runner, attempt.storage), 0);
    if (reference != nullptr) EXPECT_EQ(simpler_release_run_resources(&runner, reference), 0);
    EXPECT_EQ(runner.finalize(), 0);
    EXPECT_EQ(control().live_events, 0u);
}

TEST_F(NativeAdmissionTest, FailedFixedResourceCreationKeepsHandlesThroughCheckedCloseRetry) {
    ColdKernelRunner runner;
    ASSERT_TRUE(runner.set_run_resource_grant(2));
    control().loader_start_rc = 0;
    control().fail_event_create = static_cast<size_t>(KernelEventKind::Count) + 3u;
    const uint8_t binary = 1;
    CallConfig config{};
    config.aicpu_thread_num = 2;
    for (int i = 0; i < RUNTIME_ENV_RING_COUNT; ++i) {
        config.runtime_env.ring_task_window[i] = 4;
        config.runtime_env.ring_heap[i] = 1024;
        config.runtime_env.ring_dep_pool[i] = 4;
    }
    EXPECT_EQ(simpler_kernel_mode_init(&runner, 0, &binary, 1, &binary, 1, &binary, 1, &config, 1), -4210);
    const size_t held = control().live_events;
    EXPECT_EQ(held, static_cast<size_t>(KernelEventKind::Count) + 2u);
    EXPECT_EQ(runner.kernel_execution_state().phase(), KernelContextPhase::Poisoned);
    control().event_destroy_rc = -4211;
    EXPECT_EQ(runner.finalize(), -4211);
    EXPECT_EQ(control().live_events, held);
    control().event_destroy_rc = 0;
    EXPECT_EQ(runner.finalize(), 0);
    EXPECT_EQ(control().live_events, 0u);
    EXPECT_EQ(control().event_create_calls, static_cast<size_t>(KernelEventKind::Count) + 3u);
}

// Common CAPI/owner execution with actual architecture control helpers; only
// HAL, RTS and the device loader response are CPU leaves in this target.
class ContextControlRunner : public ColdKernelRunner {
public:
    void *observed_control_stream{nullptr};
    size_t prepare_arch_calls{0};
    uint64_t ctrl_table() const { return aicore_ctrl_reg_table_dev_; }
    bool ctrl_committed() const { return aicore_ctrl_reg_table_committed_; }
    void *hidden_control_stream() const { return kernel_exec_state_.hidden_stream(KernelStreamKind::Aicpu); }
    int warm_controls() {
        const rtStream_t stream =
            execution_mode_latch().is_kernel() ? static_cast<rtStream_t>(hidden_control_stream()) : stream_aicpu_;
        return prepare_kernel_arch_context(stream);
    }
    int prepare_execution(
        Runtime &runtime, const CallConfig &config, uint32_t slot, const NativeRunIdentity &identity,
        std::unique_ptr<PreparedExecution> *out
    ) override {
        const int rc = warm_controls();
        if (rc != 0) return rc;
        std::vector<int32_t> allowed;
        if (!selected(runtime.get_aicpu_thread_num(), allowed)) return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
        for (size_t i = 0; i < allowed.size(); ++i)
            runtime.get_aicpu_allowed_cpus()[i] = allowed[i];
        runtime.set_aicpu_allowed_cpu_count(static_cast<int32_t>(allowed.size()));
        const int prepare_rc = FencedCompletionRunner::prepare_execution(runtime, config, slot, identity, out);
        if (prepare_rc != 0) return prepare_rc;
        (*out)->kernel_args.args.regs = ctrl_table();
#ifndef CONTEXT_ARCH_A5
        (*out)->kernel_args.args.ffts_base_addr = c2c_address;
#endif
        return 0;
    }
#ifdef CONTEXT_ARCH_A5
    void *scratch{nullptr};
    bool pending{false};
    simpler::a5::AicpuDeviceOccupancy occupancy{};
    simpler::a5::AicpuTopology topology{};
    bool selected(int threads, std::vector<int32_t> &allowed) {
        simpler::a5::AicpuLaunchPlan plan;
        std::string error;
        if (!simpler::a5::build_aicpu_launch_plan(topology, threads, plan, error)) return false;
        allowed = plan.allowed_cpus;
        return true;
    }
#else
    uint64_t c2c_address{0};
    bool c2c_valid{false};
    std::vector<simpler::a2a3::AicpuLogicalCpu> topology;
    bool selected(int threads, std::vector<int32_t> &allowed) {
        return simpler::a2a3::compute_allowed_cpus(topology, threads, allowed);
    }
#endif
protected:
    int prepare_kernel_arch_context(rtStream_t stream) override {
        ++prepare_arch_calls;
        observed_control_stream = stream;
#ifdef CONTEXT_ARCH_A5
        int rc = simpler::a5::ensure_ctrl_table(
            mem_alloc_, device_id_, aicore_ctrl_reg_table_dev_, aicore_ctrl_reg_table_committed_
        );
        if (rc != 0 || occupancy.occupy_valid) return rc;
        rc = simpler::a5::query_device_aicpu_occupancy(mem_alloc_, load_aicpu_op_, stream, scratch, pending, occupancy);
        if (rc != 0) return rc;
        simpler::a5::detail::CpuTopologyData driver{};
        return simpler::a5::detail::build_aicpu_topology(occupancy, "CPU-control-fixture", false, driver, topology) ?
                   0 :
                   PTO_RUNTIME_ERR_INTERNAL;
#else
        int rc = simpler::a2a3::ensure_ctrl_table(
            mem_alloc_, device_id_, aicore_ctrl_reg_table_dev_, aicore_ctrl_reg_table_committed_
        );
        if (rc != 0) return rc;
        rc = simpler::a2a3::ensure_c2c_ctrl_address(c2c_address, c2c_valid);
        if (rc != 0) return rc;
        if (topology.empty() && !simpler::a2a3::probe_aicpu_topology(static_cast<uint32_t>(device_id_), topology))
            return PTO_RUNTIME_ERR_INTERNAL;
        return 0;
#endif
    }
    int release_arch_context_resources(bool abandon) override {
#ifdef CONTEXT_ARCH_A5
        if (abandon) {
            scratch = nullptr;
            pending = false;
            return 0;
        }
        return simpler::a5::release_aicpu_query_result(mem_alloc_, observed_control_stream, scratch, pending);
#else
        (void)abandon;
        return 0;
#endif
    }
};

namespace {
void install_topology_reply(int launch_rc = 0, int sync_rc = 0) {
    control().aicpu_payload = [launch_rc, sync_rc](void *, void *args, size_t bytes, const std::string &name) {
        if (name != "simpler_aicpu_query_topology") return 0;
#ifdef CONTEXT_ARCH_A5
        EXPECT_EQ(bytes, sizeof(AicpuTopologyQueryArgs));
        auto *query = static_cast<AicpuTopologyQueryArgs *>(args);
        auto *result = reinterpret_cast<AicpuTopologyQueryResult *>(query->result_addr);
        result->occupy_rc = result->pf_occupy_rc = result->os_sched_rc = 0;
        result->occupy = result->pf_occupy = result->os_sched = control().topology_mask;
#else
        (void)args;
        (void)bytes;
#endif
        control().stream_sync_rc = sync_rc;
        return launch_rc;
    };
}
}  // namespace

TEST_F(NativeAdmissionTest, KernelContextControlsPrecedeActivePrepareOnTheActualHiddenStream) {
    ContextControlRunner runner;
    install_topology_reply();
    ASSERT_EQ(initialize_cold_kernel_runner(runner, 2), 0);
    EXPECT_EQ(runner.prepare_arch_calls, 1u);
    EXPECT_TRUE(runner.ctrl_committed());
    EXPECT_NE(runner.ctrl_table(), 0u);
    EXPECT_NE(runner.observed_control_stream, nullptr);
    EXPECT_EQ(runner.observed_control_stream, runner.hidden_control_stream());
    const auto queries = control().ctrl_query_calls;
    const auto topology_queries = control().topology_query_calls;
    const auto c2c_queries = control().c2c_calls;
    const auto copies = control().uploads.size();
    const auto syncs = control().observed_streams.size();
    ASSERT_EQ(runner.kernel_execution_state().mark_ready_enqueued(), 0);
    Attempt attempt(runner, 0, 10020);
    use_capture_preparation(attempt);
    RuntimeHandle reference = nullptr;
    auto restore = RAIIScopeGuard([&] {
        control().recording_active = false;
        (void)simpler_finalize_run(&runner, attempt.storage);
        if (reference != nullptr) (void)simpler_release_run_resources(&runner, reference);
    });
    control().recording_active = true;
    ASSERT_EQ(attempt.prepare(), 0);
    auto *native = static_cast<NativeRunContext<DeviceRunnerBase> *>(attempt.storage);
    EXPECT_EQ(native->prepared_execution->kernel_args.args.regs, runner.ctrl_table());
    std::vector<int32_t> two, three, invalid;
    EXPECT_TRUE(runner.selected(2, two));
    EXPECT_TRUE(runner.selected(3, three));
    EXPECT_EQ(two.size(), 2u);
    EXPECT_EQ(three.size(), 3u);
    EXPECT_FALSE(runner.selected(100, invalid));
    EXPECT_EQ(control().ctrl_query_calls, queries);
    EXPECT_EQ(control().topology_query_calls, topology_queries);
    EXPECT_EQ(control().c2c_calls, c2c_queries);
    // Variable runtime uploads are separate from the fixed Ctrl image.
    EXPECT_EQ(uploads_to(reinterpret_cast<void *>(runner.ctrl_table())), 1u);
    EXPECT_GE(control().uploads.size(), copies);
    EXPECT_EQ(control().observed_streams.size(), syncs);
    int caller;
    ASSERT_EQ(simpler_retain_run_resources(&runner, attempt.storage, &caller, &reference), 0);
    ASSERT_EQ(simpler_launch_run_on_stream(&runner, attempt.storage, &caller, nullptr), 0);
    control().recording_active = false;
    ASSERT_EQ(simpler_finalize_run(&runner, attempt.storage), 0);
    ASSERT_EQ(simpler_release_run_resources(&runner, reference), 0);
    reference = nullptr;
}

TEST_F(NativeAdmissionTest, KernelContextControlFailurePoisonsAndRetainsTheExactFailedTable) {
    ContextControlRunner runner;
    install_topology_reply();
    auto restore = RAIIScopeGuard([&] {
        control().copy_rc = 0;
        control().free_rc = 0;
        control().selected_free_rc = 0;
        control().ctrl_query_hook = {};
    });
    control().ctrl_query_hook = [] {
        control().copy_rc = -812;
        control().free_rc = -813;
    };
    EXPECT_EQ(initialize_cold_kernel_runner(runner, 2), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(runner.kernel_execution_state().phase(), KernelContextPhase::Poisoned);
    const auto address = runner.ctrl_table();
    ASSERT_NE(address, 0u);
    EXPECT_FALSE(runner.ctrl_committed());
    control().ctrl_query_hook = {};
    control().copy_rc = control().free_rc = 0;
    control().selected_free = reinterpret_cast<void *>(address);
    control().selected_free_rc = -814;
    EXPECT_NE(runner.finalize(), 0);
    EXPECT_EQ(runner.ctrl_table(), address);
    control().selected_free_rc = 0;
    EXPECT_EQ(runner.finalize(), 0);
    EXPECT_EQ(runner.ctrl_table(), 0u);
}

TEST_F(NativeAdmissionTest, ProgramInitializationKeepsContextControlDiscoveryLazy) {
    ContextControlRunner runner;
    install_topology_reply();
    const uint8_t binary = 1;
    control().loader_start_rc = 0;
    ASSERT_EQ(simpler_init(&runner, 0, &binary, 1, &binary, 1, &binary, 1, nullptr, 0, nullptr, 0), 0);
    EXPECT_EQ(runner.prepare_arch_calls, 0u);
    EXPECT_EQ(control().ctrl_query_calls, 0u);
    EXPECT_EQ(control().topology_query_calls, 0u);
    EXPECT_EQ(control().c2c_calls, 0u);
    EXPECT_EQ(runner.warm_controls(), 0);
    EXPECT_TRUE(runner.ctrl_committed());
}

#ifdef CONTEXT_ARCH_A5
TEST_F(NativeAdmissionTest, KernelTopologyUnknownWriterSurvivesUntilSameStreamProofAndCheckedFree) {
    ContextControlRunner runner;
    install_topology_reply(0, -817);
    auto restore = RAIIScopeGuard([&] {
        control().stream_sync_rc = 0;
        control().selected_free_rc = 0;
    });
    EXPECT_EQ(initialize_cold_kernel_runner(runner, 2), -817);
    EXPECT_EQ(runner.kernel_execution_state().phase(), KernelContextPhase::Poisoned);
    void *address = runner.scratch;
    ASSERT_NE(address, nullptr);
    ASSERT_TRUE(runner.pending);
    control().selected_free = address;
    control().selected_free_rc = -818;
    const auto attempts = control().selected_free_attempts;
    EXPECT_EQ(runner.finalize(), -817);
    EXPECT_EQ(control().selected_free_attempts, attempts);
    EXPECT_EQ(runner.scratch, address);
    control().stream_sync_rc = 0;
    EXPECT_EQ(runner.finalize(), -818);
    EXPECT_FALSE(runner.pending);
    EXPECT_EQ(runner.scratch, address);
    EXPECT_EQ(control().observed_streams.back(), runner.observed_control_stream);
    control().selected_free_rc = 0;
    EXPECT_EQ(runner.finalize(), 0);
    EXPECT_EQ(runner.scratch, nullptr);
}

TEST_F(NativeAdmissionTest, KernelTopologyFailedEnqueueKeepsItsUnprovedWriterTarget) {
    ContextControlRunner runner;
    install_topology_reply(-819);
    auto restore = RAIIScopeGuard([&] {
        control().stream_sync_rc = 0;
    });
    EXPECT_EQ(initialize_cold_kernel_runner(runner, 2), -819);
    ASSERT_TRUE(runner.pending);
    ASSERT_NE(runner.scratch, nullptr);
    control().stream_sync_rc = -820;
    EXPECT_EQ(runner.finalize(), -820);
    EXPECT_NE(runner.scratch, nullptr);
    control().stream_sync_rc = 0;
    EXPECT_EQ(runner.finalize(), 0);
}
#else
TEST_F(NativeAdmissionTest, KernelC2cSuccessHasAnExplicitValidityBitEvenForZero) {
    ContextControlRunner runner;
    control().c2c_address = 0;
    ASSERT_EQ(initialize_cold_kernel_runner(runner, 2), 0);
    EXPECT_TRUE(runner.c2c_valid);
    EXPECT_EQ(runner.c2c_address, 0u);
    const auto calls = control().c2c_calls;
    ASSERT_EQ(runner.warm_controls(), 0);
    EXPECT_EQ(control().c2c_calls, calls);
}

TEST_F(NativeAdmissionTest, KernelC2cFailureDoesNotPublishAValidContextSnapshot) {
    ContextControlRunner runner;
    control().c2c_rc = -821;
    EXPECT_EQ(initialize_cold_kernel_runner(runner, 2), -821);
    EXPECT_FALSE(runner.c2c_valid);
    EXPECT_EQ(runner.kernel_execution_state().phase(), KernelContextPhase::Poisoned);
    control().c2c_rc = 0;
    EXPECT_EQ(runner.finalize(), 0);
}
#endif

TEST_F(NativeAdmissionTest, KernelContextCtrlQueryAndAllocationFailuresDoNotCommitATable) {
    for (bool fail_query : {true, false}) {
        ContextControlRunner runner;
        install_topology_reply();
        auto restore = RAIIScopeGuard([&] {
            control().ctrl_query_rc = 0;
            control().fail_allocation = 0;
            control().ctrl_query_hook = {};
        });
        if (fail_query) control().ctrl_query_rc = -825;
        else
            control().ctrl_query_hook = [] {
                control().fail_allocation = control().allocations + 1;
            };
        EXPECT_EQ(initialize_cold_kernel_runner(runner, 2), fail_query ? -825 : PTO_RUNTIME_ERR_INTERNAL);
        EXPECT_EQ(runner.kernel_execution_state().phase(), KernelContextPhase::Poisoned);
        EXPECT_FALSE(runner.ctrl_committed());
        EXPECT_EQ(runner.ctrl_table(), 0u);
        control().ctrl_query_rc = 0;
        control().fail_allocation = 0;
        control().ctrl_query_hook = {};
        EXPECT_EQ(runner.finalize(), 0);
    }
}

TEST_F(NativeAdmissionTest, KernelContextTopologyIsDiscoveredAgainAfterItsOwnerRetires) {
    const auto initial_queries = control().topology_query_calls;
    for (uint64_t mask : {0xfu, 0xfcu}) {
        ContextControlRunner runner;
        control().topology_mask = mask;
        control().c2c_address = mask;
        install_topology_reply();
        ASSERT_EQ(initialize_cold_kernel_runner(runner, 2), 0);
        std::vector<int32_t> allowed;
        ASSERT_TRUE(runner.selected(3, allowed));
        for (int32_t cpu : allowed)
            EXPECT_NE(mask & (1ULL << cpu), 0u);
#ifndef CONTEXT_ARCH_A5
        EXPECT_TRUE(runner.c2c_valid);
        EXPECT_EQ(runner.c2c_address, mask);
#endif
        ASSERT_EQ(runner.finalize(), 0);
    }
#ifndef CONTEXT_ARCH_A5
    EXPECT_EQ(control().topology_query_calls - initial_queries, 4u);
    EXPECT_EQ(control().c2c_calls, 2u);
#else
    (void)initial_queries;
#endif
}

#ifdef CONTEXT_ARCH_A5
TEST_F(NativeAdmissionTest, KernelTopologyCompletedQueryRetainsItsFailedFreeForCheckedClose) {
    ContextControlRunner runner;
    install_topology_reply();
    auto reply = control().aicpu_payload;
    control().aicpu_payload = [reply](void *stream, void *args, size_t bytes, const std::string &name) {
        const int rc = reply(stream, args, bytes, name);
        if (name == "simpler_aicpu_query_topology") control().free_rc = -826;
        return rc;
    };
    auto restore = RAIIScopeGuard([&] {
        control().free_rc = 0;
        control().selected_free_rc = 0;
    });
    EXPECT_EQ(initialize_cold_kernel_runner(runner, 2), -826);
    ASSERT_NE(runner.scratch, nullptr);
    EXPECT_FALSE(runner.pending);
    EXPECT_FALSE(runner.occupancy.occupy_valid);
    void *address = runner.scratch;
    control().free_rc = 0;
    control().selected_free = address;
    control().selected_free_rc = -827;
    EXPECT_EQ(runner.finalize(), -827);
    EXPECT_EQ(runner.scratch, address);
    control().selected_free_rc = 0;
    EXPECT_EQ(runner.finalize(), 0);
    EXPECT_EQ(runner.scratch, nullptr);
}

TEST_F(NativeAdmissionTest, KernelTopologyReadbackFailurePreservesThePrimaryErrorAndProvedScratch) {
    ContextControlRunner runner;
    install_topology_reply();
    auto reply = control().aicpu_payload;
    control().aicpu_payload = [reply](void *stream, void *args, size_t bytes, const std::string &name) {
        const int rc = reply(stream, args, bytes, name);
        if (name == "simpler_aicpu_query_topology") control().copy_rc = -828;
        return rc;
    };
    auto restore = RAIIScopeGuard([&] {
        control().copy_rc = 0;
    });
    EXPECT_EQ(initialize_cold_kernel_runner(runner, 2), -828);
    EXPECT_EQ(runner.kernel_execution_state().last_runtime_error(), -828);
    EXPECT_FALSE(runner.pending);
    ASSERT_NE(runner.scratch, nullptr);
    EXPECT_FALSE(runner.occupancy.occupy_valid);
    control().copy_rc = 0;
    EXPECT_EQ(runner.finalize(), 0);
}
#endif

}  // namespace native_admission_test

namespace native_admission_test {
namespace {
class SwimlaneCaptureRunner : public FencedCompletionRunner {
public:
    SwimlaneCaptureRunner() {
        static int binary;
        aicore_bin_handle_ = &binary;
    }
    ~SwimlaneCaptureRunner() override {
        chip_swimlane_collector_.finish_retained_runs();
        chip_swimlane_collector_.finalize(nullptr, [this](void *pointer) {
            return mem_alloc_.free(pointer);
        });
    }
    bool supports_swimlane_capture() const override { return supported; }
    bool supported{true};
    int prepare_collector_storage(const Runtime &, PreparedExecution &prepared) override {
        ++arm_calls;
        const int rc = chip_swimlane_collector_.initialize(
            1, 1, 0, prepared.dfx.chip_swimlane_level,
            [this](size_t bytes) {
                return mem_alloc_.alloc(bytes);
            },
            nullptr,
            [this](void *pointer) {
                return mem_alloc_.free(pointer);
            }
        );
        prepared.kernel_args.args.chip_swimlane_data_base =
            reinterpret_cast<uint64_t>(chip_swimlane_collector_.get_chip_swimlane_setup_device_ptr());
        return rc;
    }
    void seed_host_preparation() {
        auto &host = host_phase_runs_[0];
        host.host_orchestrated = true;
        auto *pool = host.records.arm(true);
        host_phase_pool_append(pool, HostPhaseKind::OrchSubmitTask, 300, 350, 8802, 1);
        host_phase_pool_append(pool, HostPhaseKind::BindArenaH2d, 200, 250, 128, 0);
        host_phase_pool_append(pool, HostPhaseKind::OrchSubmitTask, 100, 150, 8801, 0);
        host.records.finish(2, 9800);
    }
    bool flush(std::string *error) { return chip_swimlane_collector_.flush_retained_runs(5000, error); }
    bool window(uint64_t epoch) const { return chip_swimlane_collector_.recorded_window_open(epoch); }
    bool ordinary_retention() const { return chip_swimlane_collector_.retains_runs(); }
    void close_timeout(int milliseconds) { timeout_config_.stream_sync_timeout_ms = milliseconds; }
    void produce_one(uint64_t epoch) {
        void *shm = chip_swimlane_collector_.get_chip_swimlane_setup_device_ptr();
        set_platform_run_result(0, epoch);
        ::set_chip_swimlane_enabled(true);
        set_platform_chip_swimlane_base(reinterpret_cast<uint64_t>(shm));
        set_platform_chip_swimlane_aicore_rotation_table(0);
        set_platform_chip_swimlane_run_terminal_bank(0);
        chip_swimlane_aicpu_init(1, ChipSwimlaneLevel::TASK_TIMING);
        chip_swimlane_aicpu_on_aicore_dispatch(0, 0, 1);
        auto *pool = get_aicore_buffer_state(shm, 0);
        auto *buffer = reinterpret_cast<ChipSwimlaneAicoreTaskBuffer *>(pool->head.current_buf_ptr);
        buffer->records[0].start_time = 100;
        buffer->records[0].end_time = 200;
        const int cores[] = {0};
        chip_swimlane_aicpu_flush(0, cores, 1);
    }
    int arm_calls{0};
};
}  // namespace

TEST_F(NativeAdmissionTest, CaptureSwimlaneWindowUsesTheCommonExternalReference) {
    SwimlaneCaptureRunner runner;
    ASSERT_EQ(initialize_capture_runner(runner), 0);
    Attempt attempt(runner, 0, 9800);
    use_capture_preparation(attempt);
    diagnostic_path = "native-swimlane-capture-lifetime";
    attempt.config.enable_chip_swimlane = 4;
    std::strncpy(attempt.config.output_prefix, diagnostic_path.c_str(), sizeof(attempt.config.output_prefix) - 1);
    ASSERT_EQ(attempt.prepare(), 0);
    runner.seed_host_preparation();
    int caller;
    RuntimeHandle reference = nullptr;
    ASSERT_EQ(simpler_retain_run_resources(&runner, attempt.storage, &caller, &reference), 0);
    auto release = RAIIScopeGuard([&] {
        (void)simpler_finalize_run(&runner, attempt.storage);
        (void)simpler_release_run_resources(&runner, reference);
    });
    EXPECT_EQ(runner.arm_calls, 1);
    EXPECT_FALSE(runner.ordinary_retention());
    auto *state = static_cast<NativeRunContext<DeviceRunnerBase> *>(attempt.storage);
    EXPECT_EQ(state->prepared_execution->kernel_args.args.chip_swimlane_run_terminal_bank, 0u);
    RuntimeHandle duplicate = reinterpret_cast<void *>(1);
    EXPECT_NE(simpler_retain_run_resources(&runner, attempt.storage, &caller, &duplicate), 0);
    EXPECT_EQ(duplicate, nullptr);
    EXPECT_TRUE(runner.window(9800));
    ASSERT_EQ(runner.admit_recording(*state->prepared_execution), 0);
    state->recording = true;
    state->completion_rc = 0;
    ASSERT_EQ(simpler_finalize_run(&runner, attempt.storage), 0);
    EXPECT_TRUE(runner.window(9800));
    control().stream_sync_rc = -996;
    EXPECT_EQ(simpler_release_run_resources(&runner, reference), -996);
    EXPECT_TRUE(runner.window(9800));
    control().stream_sync_rc = 0;
    ASSERT_EQ(simpler_release_run_resources(&runner, reference), 0);
    release.dismiss();
    std::string error;
    EXPECT_TRUE(runner.flush(&error)) << error;
    const auto path = std::filesystem::path(diagnostic_path) / "swimlane-0/records_e9800.json";
    std::ifstream artifact(path);
    ASSERT_TRUE(artifact.is_open());
    const std::string text{std::istreambuf_iterator<char>(artifact), std::istreambuf_iterator<char>()};
    EXPECT_NE(text.find("\"host_device_uploads\""), std::string::npos);
    EXPECT_NE(text.find("\"metadata_complete\": true"), std::string::npos);
    const auto first = text.find("\"task_id\": 8801");
    const auto second = text.find("\"task_id\": 8802");
    ASSERT_NE(first, std::string::npos);
    ASSERT_NE(second, std::string::npos);
    EXPECT_LT(first, second) << "the finished pool is copied directly in timestamp order";
}
}  // namespace native_admission_test

namespace native_admission_test {
#if defined(__linux__)
TEST_F(NativeAdmissionTest, SwimlaneReleaseRetriesPendingCutAfterEventRetirement) {
    SwimlaneCaptureRunner runner;
    ASSERT_EQ(initialize_capture_runner(runner), 0);
    Attempt attempt(runner, 0, 9808);
    use_capture_preparation(attempt);
    diagnostic_path = "native-swimlane-cut-retry";
    attempt.config.enable_chip_swimlane = 1;
    std::strcpy(attempt.config.output_prefix, diagnostic_path.c_str());
    ASSERT_EQ(attempt.prepare(), 0);
    int caller;
    RuntimeHandle reference = nullptr;
    ASSERT_EQ(simpler_retain_run_resources(&runner, attempt.storage, &caller, &reference), 0);
    auto *state = static_cast<NativeRunContext<DeviceRunnerBase> *>(attempt.storage);
    ASSERT_EQ(runner.admit_recording(*state->prepared_execution), 0);
    state->recording = true;
    state->completion_rc = 0;
    ASSERT_EQ(simpler_finalize_run(&runner, attempt.storage), 0);
    SwimlaneRecordGate gate;
    swimlane_record_gate.store(&gate);
    auto cleanup = RAIIScopeGuard([&] {
        swimlane_record_gate.store(nullptr);
        gate.release();
        runner.close_timeout(5000);
        (void)simpler_release_run_resources(&runner, reference);
    });
    runner.produce_one(9808);
    ASSERT_TRUE(gate.wait_entered()) << "the real producer record must reach collector allocation";
    runner.close_timeout(0);
    const int first_close = simpler_release_run_resources(&runner, reference);
    if (first_close == 0) reference = nullptr;
    ASSERT_EQ(first_close, PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_TRUE(runner.window(9808));
    swimlane_record_gate.store(nullptr);
    gate.release();
    runner.close_timeout(5000);
    ASSERT_EQ(simpler_release_run_resources(&runner, reference), 0);
    cleanup.dismiss();
    std::string error;
    ASSERT_TRUE(runner.flush(&error)) << error;
    EXPECT_FALSE(runner.window(9808));
}

#endif

TEST_F(NativeAdmissionTest, SwimlanePermanentPathRefusalDoesNotAdvertiseCapacity) {
    SwimlaneCaptureRunner runner;
    ASSERT_EQ(initialize_capture_runner(runner), 0);
    Attempt attempt(runner, 0, 9809);
    use_capture_preparation(attempt);
    diagnostic_path = "native-swimlane-permanent-path";
    std::ofstream(diagnostic_path) << "not a directory";
    attempt.config.enable_chip_swimlane = 1;
    std::strcpy(attempt.config.output_prefix, diagnostic_path.c_str());
    ASSERT_EQ(attempt.prepare(), 0);
    int caller;
    RuntimeHandle reference = nullptr;
    EXPECT_EQ(simpler_retain_run_resources(&runner, attempt.storage, &caller, &reference), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(reference, nullptr);
}

TEST_F(NativeAdmissionTest, UnsupportedSwimlaneCaptureRefusesBeforePoolMutation) {
    for (bool mixed : {false, true}) {
        SwimlaneCaptureRunner runner;
        ASSERT_EQ(initialize_capture_runner(runner), 0);
        Attempt attempt(runner, 0, 9801);
        use_capture_preparation(attempt);
        attempt.config.enable_chip_swimlane = 1;
        attempt.config.enable_pmu = mixed ? 2 : 0;
        std::strcpy(attempt.config.output_prefix, "unused-swimlane-refusal");
        runner.supported = mixed;
        ASSERT_EQ(attempt.prepare(), 0);
        const auto allocations = control().allocations;
        int caller;
        RuntimeHandle reference = reinterpret_cast<void *>(1);
        EXPECT_EQ(
            simpler_retain_run_resources(&runner, attempt.storage, &caller, &reference), PTO_RUNTIME_ERR_UNSUPPORTED
        );
        EXPECT_EQ(reference, nullptr);
        EXPECT_EQ(runner.arm_calls, 0);
        EXPECT_EQ(control().allocations, allocations);
    }
}
}  // namespace native_admission_test

namespace native_admission_test {
namespace {
class CaptureDiagnosticGateRunner : public FencedCompletionRunner {
public:
    CaptureDiagnosticGateRunner() {
        static int binary;
        aicore_bin_handle_ = &binary;
    }
    bool supports_scope_stats_capture() const override { return true; }
    bool supports_pmu_capture() const override { return true; }
    bool supports_args_dump_capture() const override { return true; }
    bool supports_swimlane_capture() const override { return true; }
    int prepare_collector_storage(const Runtime &, PreparedExecution &) override {
        ++arm_calls;
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    int arm_calls{0};
};
}  // namespace

TEST_F(NativeAdmissionTest, MixedCaptureDiagnosticsRefuseBeforeAnySharedPoolMutation) {
    const bool args_dump_supported = args_dump_capture_supported_impl() != 0;
    unsigned checked = 0;
    for (unsigned selection = 1; selection < 16; ++selection) {
        if ((selection & (selection - 1)) == 0) continue;
        if (!args_dump_supported && (selection & 4) != 0) continue;
        ++checked;
        SCOPED_TRACE(selection);
        CaptureDiagnosticGateRunner runner;
        ASSERT_EQ(initialize_capture_runner(runner), 0);
        Attempt attempt(runner, 0, 9900 + selection);
        use_capture_preparation(attempt);
        attempt.config.enable_scope_stats = (selection & 1) != 0;
        attempt.config.enable_pmu = (selection & 2) != 0 ? 2 : 0;
        attempt.config.enable_dump_args = (selection & 4) != 0 ? static_cast<int>(DumpArgsLevel::FULL) : 0;
        attempt.config.enable_chip_swimlane = (selection & 8) != 0 ? 4 : 0;
        std::strcpy(attempt.config.output_prefix, "unused-mixed-diagnostic-refusal");
        ASSERT_EQ(attempt.prepare(), 0);
        const auto allocations = control().allocations;
        const auto pmu_starts = control().pmu_start_calls;
        int caller;
        RuntimeHandle reference = reinterpret_cast<void *>(1);
        EXPECT_EQ(
            simpler_retain_run_resources(&runner, attempt.storage, &caller, &reference), PTO_RUNTIME_ERR_UNSUPPORTED
        );
        EXPECT_EQ(reference, nullptr);
        EXPECT_EQ(runner.arm_calls, 0);
        EXPECT_EQ(control().allocations, allocations);
        EXPECT_EQ(control().pmu_start_calls, pmu_starts);
    }
    RecordProperty("mixed_selections_checked", checked);
    EXPECT_EQ(checked, args_dump_supported ? 11u : 4u);
}
}  // namespace native_admission_test

namespace native_admission_test {
namespace {
class ArgsAndSwimlaneCaptureRunner : public SwimlaneCaptureRunner {
public:
    ~ArgsAndSwimlaneCaptureRunner() override {
        dump_collector_.finalize(nullptr, [this](void *pointer) {
            return mem_alloc_.free(pointer);
        });
    }
    bool supports_args_dump_capture() const override { return true; }
    int prepare_collector_storage(const Runtime &runtime, PreparedExecution &prepared) override {
        if (!prepared.dfx.dump_args_enabled())
            return SwimlaneCaptureRunner::prepare_collector_storage(runtime, prepared);
        const int rc = dump_collector_.initialize(
            1, 0, prepared.dfx.dump_args_level,
            [this](size_t bytes) {
                return mem_alloc_.alloc(bytes);
            },
            nullptr,
            [this](void *pointer) {
                return mem_alloc_.free(pointer);
            }
        );
        prepared.kernel_args.args.dump_data_base =
            reinterpret_cast<uint64_t>(dump_collector_.get_dump_shm_device_ptr());
        return rc;
    }
    bool args_window(uint64_t epoch) const { return dump_collector_.recorded_window_open(epoch); }
    bool flush_args(std::string *error) { return dump_collector_.flush_retained_runs(5000, error); }
};
}  // namespace

TEST_F(NativeAdmissionTest, SeparateDiagnosticModelsKeepIndependentCommonReferenceObligations) {
    if (args_dump_capture_supported_impl() == 0) GTEST_SKIP() << "runtime has no task-time argument producer";
    ArgsAndSwimlaneCaptureRunner runner;
    ASSERT_EQ(initialize_capture_runner(runner), 0);
    diagnostic_path = "native-separate-diagnostic-models";
    Attempt swimlane(runner, 0, 9950), args(runner, 1, 9951);
    use_capture_preparation(swimlane);
    use_capture_preparation(args);
    swimlane.config.enable_chip_swimlane = 4;
    args.config.enable_dump_args = static_cast<int>(DumpArgsLevel::FULL);
    const auto swimlane_prefix = diagnostic_path + "/swimlane";
    const auto args_prefix = diagnostic_path + "/args";
    std::strncpy(swimlane.config.output_prefix, swimlane_prefix.c_str(), sizeof(swimlane.config.output_prefix) - 1);
    std::strncpy(args.config.output_prefix, args_prefix.c_str(), sizeof(args.config.output_prefix) - 1);
    int caller;
    RuntimeHandle swimlane_ref = nullptr, args_ref = nullptr;
    auto cleanup = RAIIScopeGuard([&] {
        control().stream_sync_rc = 0;
        (void)simpler_finalize_run(&runner, swimlane.storage);
        (void)simpler_finalize_run(&runner, args.storage);
        if (swimlane_ref != nullptr) (void)simpler_release_run_resources(&runner, swimlane_ref);
        if (args_ref != nullptr) (void)simpler_release_run_resources(&runner, args_ref);
    });
    auto capture = [&](Attempt &attempt, RuntimeHandle *reference) {
        ASSERT_EQ(attempt.prepare(), 0);
        ASSERT_EQ(simpler_retain_run_resources(&runner, attempt.storage, &caller, reference), 0);
        auto *state = static_cast<NativeRunContext<DeviceRunnerBase> *>(attempt.storage);
        ASSERT_EQ(runner.admit_recording(*state->prepared_execution), 0);
        state->recording = true;
        state->completion_rc = 0;
        ASSERT_EQ(simpler_finalize_run(&runner, attempt.storage), 0);
    };
    capture(swimlane, &swimlane_ref);
    capture(args, &args_ref);
    ASSERT_NE(swimlane_ref, nullptr);
    ASSERT_NE(args_ref, nullptr);
    ASSERT_TRUE(runner.window(9950));
    ASSERT_TRUE(runner.args_window(9951));
    control().stream_sync_rc = -996;
    EXPECT_EQ(simpler_release_run_resources(&runner, swimlane_ref), -996);
    EXPECT_TRUE(runner.window(9950));
    EXPECT_TRUE(runner.args_window(9951));
    control().stream_sync_rc = 0;
    ASSERT_EQ(simpler_release_run_resources(&runner, swimlane_ref), 0);
    swimlane_ref = nullptr;
    EXPECT_TRUE(runner.args_window(9951));
    std::string error;
    EXPECT_TRUE(runner.flush(&error)) << error;
    EXPECT_FALSE(runner.window(9950));
    ASSERT_EQ(simpler_release_run_resources(&runner, args_ref), 0);
    args_ref = nullptr;
    EXPECT_TRUE(runner.flush_args(&error)) << error;
    EXPECT_FALSE(runner.args_window(9951));
    EXPECT_TRUE(std::filesystem::exists(std::filesystem::path(swimlane_prefix) / "swimlane-0/records_e9950.json"));
    EXPECT_TRUE(std::filesystem::exists(std::filesystem::path(args_prefix) / "args_dump/args_dump.json"));
    cleanup.dismiss();
}
}  // namespace native_admission_test
