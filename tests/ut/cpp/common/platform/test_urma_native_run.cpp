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

#include <cstdio>
#include <cstring>
#include "fixture.h"
#include "host_build_graph/aicpu/aicore_scheduler_error.h"
#include "urma_queue_drain.h"
#include "../../a5/runtime/urma_fault_domain.h"
#define private public
#include "chip_worker.h"
#undef private
#include "chip_run_lane.h"

using simpler::test::FaultDomain;
using simpler::test::UrmaFault;

namespace {
struct QueueMemory {
    bool device_usable;
    template <typename T>
    static bool read(uint64_t address, T &value) {
        if (address == 0) return false;
        std::memcpy(&value, reinterpret_cast<const void *>(address), sizeof(value));
        return true;
    }

    static bool write(uint64_t address, uint32_t value) {
        std::memcpy(reinterpret_cast<void *>(address), &value, sizeof(value));
        return true;
    }

    UrmaQueueState inspect_prefix(uint64_t address, uint32_t depth, uint32_t completed, uint32_t target) const {
        return device_usable ?
                   simpler::urma::inspect_cq_entries<UrmaQueueState>(*this, address, depth, completed, target) :
                   UrmaQueueState::FAILED;
    }
    UrmaQueueState ring_doorbell(uint64_t address, uint32_t value) const {
        return device_usable && write(address, value) ? UrmaQueueState::READY : UrmaQueueState::FAILED;
    }
};
}  // namespace

// The simulator owns the domain and queue memory. The signed device status,
// Native Run carrier, worker slot and caller buffer registry are exercised together.
extern "C" int acquire_hbg_urma_run(void *handle, const uint64_t *scalars, int32_t count, UrmaRunLease *use) {
    auto &domain = *static_cast<FaultDomain *>(handle);
    if (count != 1 || scalars[0] != domain.addr()) return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
    if (domain.queue_active) return COMM_DOMAIN_ADMISSION_STOPPED;
    ++domain.acquired;
    auto owner = std::shared_ptr<void>(&domain, [](void *value) {
        ++static_cast<FaultDomain *>(value)->released;
    });
    domain.queue_active = true;
    auto claim = std::shared_ptr<void>(&domain, [](void *value) {
        static_cast<FaultDomain *>(value)->queue_active = false;
    });
    use->domains.push_back(
        {domain.binding, std::move(owner), std::move(claim),
         [](void *value, UrmaQueueReceipt &receipt, UrmaWait, const UrmaLaunchPermission &permission) {
             auto &domain = *static_cast<FaultDomain *>(value);
             ++domain.drain_checks;
             if (domain.check_allocation_failures != 0) {
                 --domain.check_allocation_failures;
                 throw std::bad_alloc();
             }
             const auto gate = permission.lock();
             if (domain.drain_checks == domain.complete_on_check) domain.set_cqe(domain.row, 0, true);
             return simpler::urma::check_urma_queues(
                 QueueMemory{gate && gate->open()}, domain.addr(), domain.binding.local_rank, domain.binding.rank_count,
                 &receipt
             );
         }}
    );
    return 0;
}

namespace native_admission_test {
thread_local bool refuse_next_host_allocation = false;
namespace {

class FaultRunner : public Runner {
public:
    FaultRunner(FaultDomain &domain, bool put, UrmaFault fault) :
        domain_(domain),
        put_(put),
        fault_(fault) {
        callables_[1].signature = {put ? ArgDirection::IN : ArgDirection::OUT, ArgDirection::SCALAR};
    }

    void mark_device_unusable(int rc) override {
        revoke_urma_launches_and_mark([&] {
            Runner::mark_device_unusable(rc);
        });
    }
    void recover_device_or_mark_unusable(int rc) override {
        revoke_urma_launches_and_mark([&] {
            Runner::recover_device_or_mark_unusable(rc);
        });
    }
    uint64_t operand{};
    int status{};
    void set_stream_wait_budget(int milliseconds) { timeout_config_.stream_sync_timeout_ms = milliseconds; }
    int acquire_scheduler_notifications(
        uint32_t slot, SchedulerNotificationPurpose purpose, size_t count, SchedulerDoneNotifications *out
    ) override {
        if (purpose != SchedulerNotificationPurpose::CALENDAR_COUNTER)
            return Runner::acquire_scheduler_notifications(slot, purpose, count, out);
        struct alignas(32) Counter {
            uint32_t value{};
        };
        auto counters = std::make_shared<std::vector<Counter>>(count);
        out->owner = counters;
        for (auto &counter : *counters) {
            const auto address = reinterpret_cast<uint64_t>(&counter.value);
            out->addresses.push_back(address);
            out->add_addresses.push_back(address + 8);
        }
        return 0;
    }

    LaunchTransactionResult launch_execution_impl(PreparedExecution &prepared, LaunchPermit permit) override {
        auto launch = Runner::launch_execution_impl(prepared, std::move(permit));
        prepared.resources_owned = true;
        domain_.run_kernel(operand, put_, fault_);
        status = runtime_status_from_error_code(aicore_scheduler_runtime_error_code(domain_.lane_error));
        return launch;
    }
    int poll_execution(const ActiveExecution &) override {
        return drain_calls ? SIMPLER_NATIVE_RUN_POLL_COMPLETE : SIMPLER_NATIVE_RUN_POLL_NOT_READY;
    }
    DrainOutcome drain_execution(ActiveExecution &active) override {
        ++drain_calls;
        active.prepared->resources_owned = false;
        active.prepared->physical_complete = true;
        return DrainOutcome::device_error(status, true);
    }
    int retire_completed_execution(ActiveExecution &active) override {
        active.prepared->resources_owned = false;
        return SIMPLER_NATIVE_RUN_POLL_COMPLETE;
    }

private:
    FaultDomain &domain_;
    bool put_;
    UrmaFault fault_;
};

[[noreturn]] void exit_with_test_result() {
    const auto &result = *::testing::UnitTest::GetInstance()->current_test_info()->result();
    for (int index = 0; index < result.total_part_count(); ++index) {
        const auto &part = result.GetTestPartResult(index);
        if (part.failed()) std::fprintf(stderr, "%s:%d: %s\n", part.file_name(), part.line_number(), part.message());
    }
    std::_Exit(result.Failed() ? 1 : 0);
}

enum class HostReceiptWait { Observer, Physical, AfterTimeout };

void exercise_failure(
    bool put, UrmaFault fault, bool late_completion, HostReceiptWait wait = HostReceiptWait::Observer
) {
    FaultDomain domain;
    FaultRunner runner(domain, put, fault);
    ChipWorker worker;
    worker.initialized_ = true;
    worker.device_ctx_ = &runner;
    worker.base_comm_handle_ = reinterpret_cast<uint64_t>(&domain);
    worker.pipeline_contract_ = {PTO_PIPELINE_CONTRACT_ABI_VERSION, 0, 1, {}};
    worker.pipeline_generations_.resize(1);
    worker.runtime_bufs_.emplace_back(get_runtime_size(), get_runtime_alignment(), nullptr);
    worker.prepare_run_fn_ = simpler_prepare_run_v2;
    worker.launch_run_fn_ = simpler_launch_run;
    worker.query_run_completion_fn_ = simpler_query_run_completion;
    worker.poll_run_fn_ = simpler_poll_run;
    worker.wait_run_fn_ = simpler_wait_run;
    worker.wait_run_completion_fn_ = simpler_wait_run_completion;
    worker.finalize_run_fn_ = simpler_finalize_run_with_delivery;
    worker.device_malloc_ctx_fn_ = device_malloc_ctx;
    worker.device_free_caller_buffer_ctx_fn_ = device_free_caller_buffer_ctx;
    worker.device_borrow_caller_buffers_ctx_fn_ = device_borrow_caller_buffers_ctx;
    worker.device_release_caller_buffers_ctx_fn_ = device_release_caller_buffers_ctx;

    ChipRunLane lane(worker);
    runner.operand = worker.malloc(16);
    ChipTensor tensor;
    const uint32_t shape[]{16};
    tensor.init_external(reinterpret_cast<void *>(runner.operand), 16, shape, 1, DataType::UINT8, AddressSpace::DEVICE);
    ChipStorageTaskArgs args;
    args.add_tensor(tensor);
    args.add_scalar(domain.addr());
    CallConfig config{};
    config.aicpu_thread_num = 2;
    for (int i = 0; i < RUNTIME_ENV_RING_COUNT; ++i) {
        config.runtime_env.ring_task_window[i] = 4;
        config.runtime_env.ring_heap[i] = test_heap_capacity();
        config.runtime_env.ring_dep_pool[i] = 4;
    }
    auto run = lane.submit(1, args, config, PipelineSlotLease{0, 0, 1}, 1, 1);
    ASSERT_TRUE(run.launched());
    ASSERT_EQ(domain.posted_operand, runner.operand);
    const int error = fault == UrmaFault::HostReceipt ? SIMPLER_ERROR_NONE :
                      fault == UrmaFault::Finish      ? SIMPLER_ERROR_ASYNC_REGISTRATION_FAILED :
                      fault == UrmaFault::BadCqe      ? SIMPLER_ERROR_ASYNC_COMPLETION_INVALID :
                                                        SIMPLER_ERROR_SCHEDULER_TIMEOUT;
    ASSERT_EQ(runner.status, runtime_status_from_error_code(error));
    ASSERT_EQ(domain.sq_head[domain.row], 1u);
    ASSERT_EQ(domain.sq[domain.row].submittedWqeCount, 1u);
    ASSERT_EQ(worker.native_runs_.size(), 1u);
    auto &entry = *worker.native_runs_.begin()->second;
    auto *carrier = static_cast<NativeRunContext<DeviceRunnerBase> *>(entry.storage.data());
    const auto identity = carrier->identity();
    ChipWorkerNativeRun token{identity.pipeline_slot, identity.generation, identity.run_epoch, 1, 1};

    EXPECT_EQ(simpler_query_run_completion(&runner, carrier), SIMPLER_NATIVE_RUN_POLL_NOT_READY);
    EXPECT_EQ(domain.drain_checks, 0);
    if (fault == UrmaFault::HostReceipt) {
        if (wait != HostReceiptWait::Observer) {
            // The mock chip has stopped, and its GET is still in flight.
            carrier->active_execution->prepared->physical_complete = true;
            if (wait == HostReceiptWait::AfterTimeout) {
                runner.set_stream_wait_budget(3);
                EXPECT_THROW(run.wait_until(ChipRun::Deadline::max()), NativeRunWaitIncomplete);
                EXPECT_FALSE(lane.poisoned());
                EXPECT_GT(domain.drain_checks, 0);
                EXPECT_EQ(carrier->completion_rc, 0);
                EXPECT_EQ(domain.released, 0);
                EXPECT_TRUE(domain.queue_active);
                EXPECT_THROW(worker.free(runner.operand), std::runtime_error);
                domain.set_cqe(domain.row, 0, true);
            } else {
                domain.complete_on_check = 3;
            }
        } else {
            domain.check_allocation_failures = 2;
            for (int retry = 0; retry < 2; ++retry) {
                EXPECT_EQ(simpler_wait_run(&runner, carrier), SIMPLER_NATIVE_RUN_WAIT_INCOMPLETE);
                EXPECT_EQ(simpler_finalize_run(&runner, carrier), SIMPLER_NATIVE_RUN_WAIT_INCOMPLETE);
                EXPECT_EQ(domain.check_allocation_failures, 0);
                EXPECT_EQ(simpler_poll_run(&runner, carrier), SIMPLER_NATIVE_RUN_POLL_NOT_READY);
                EXPECT_EQ(carrier->completion_rc, 0);
                EXPECT_TRUE(domain.queue_active);
                EXPECT_THROW(worker.free(runner.operand), std::runtime_error);
            }
            domain.set_cqe(domain.row, 0, true);
            EXPECT_EQ(simpler_wait_run(&runner, carrier), 0);
        }
        EXPECT_TRUE(run.wait_until(ChipRun::Deadline::max()));
        if (wait == HostReceiptWait::Physical) EXPECT_EQ(domain.drain_checks, domain.complete_on_check);
        EXPECT_FALSE(lane.poisoned());
        EXPECT_EQ(runner.drain_calls, 1);
        EXPECT_FALSE(domain.queue_active);
        EXPECT_EQ(domain.sq_tail[domain.row], 1u);
        EXPECT_EQ(domain.cq_tail[domain.row], 1u);
        EXPECT_EQ(domain.cq_doorbell[domain.row], 1u);
        EXPECT_EQ(domain.released, 1);
        EXPECT_NO_THROW(worker.free(runner.operand));
        exit_with_test_result();
    }
    worker.wait_native_run(token);
    ASSERT_TRUE(carrier->result_observed);
    for (int retry = 0; retry < 2; ++retry) {
        EXPECT_EQ(simpler_wait_run(&runner, carrier), runner.status);
        const auto poll =
            fault == UrmaFault::BadCqe ? SIMPLER_NATIVE_RUN_POLL_ERROR : SIMPLER_NATIVE_RUN_POLL_NOT_READY;
        EXPECT_EQ(simpler_query_run_completion(&runner, carrier), poll);
        EXPECT_EQ(simpler_poll_run(&runner, carrier), poll);
        EXPECT_EQ(simpler_finalize_run(&runner, carrier), runner.status);
        EXPECT_FALSE(carrier->physical_complete);
        EXPECT_FALSE(carrier->execution_retired);
        EXPECT_EQ(carrier->host_api.urma.domains.size(), 1u);
        EXPECT_EQ(domain.released, 0);
        EXPECT_EQ(domain.done_count, 0u);
        EXPECT_EQ(domain.cq_tail[domain.row], 0u);
        EXPECT_EQ(domain.cq_doorbell[domain.row], 0u);
        EXPECT_EQ(domain.sq_tail[domain.row], 0u);
        EXPECT_EQ(runner.reservations(), 1u);
        EXPECT_EQ(runner.caller_buffer_borrow_count(), 1u);
        EXPECT_THROW(worker.free(runner.operand), std::runtime_error);
        EXPECT_THROW(worker.require_comm_reclamation_safe(), std::runtime_error);
        UrmaRunLease other;
        const uint64_t ctx = domain.addr();
        EXPECT_EQ(acquire_hbg_urma_run(&domain, &ctx, 1, &other), COMM_DOMAIN_ADMISSION_STOPPED);
    }

    const auto pending_checks = domain.drain_checks;
    EXPECT_GT(pending_checks, 0);
    if (late_completion) {
        domain.set_cqe(domain.row, 0, true);
        ASSERT_EQ(domain.cq_tail[domain.row], 0u);
        ASSERT_EQ(domain.cq_doorbell[domain.row], 0u);
        ASSERT_EQ(domain.sq_tail[domain.row], 0u);
        EXPECT_EQ(simpler_query_run_completion(&runner, carrier), SIMPLER_NATIVE_RUN_POLL_COMPLETE);
        EXPECT_EQ(simpler_query_run_completion(&runner, carrier), SIMPLER_NATIVE_RUN_POLL_COMPLETE);
        EXPECT_TRUE(domain.queue_active);
        EXPECT_EQ(domain.cq_doorbell[domain.row], 1u);
        EXPECT_EQ(domain.released, 0);
        EXPECT_THROW(worker.free(runner.operand), std::runtime_error);
        EXPECT_EQ(domain.drain_checks, pending_checks + 1);
    }
    for (int retry = 0; retry < 2; ++retry) {
        try {
            (void)run.wait_until(ChipRun::Deadline::max());
            ADD_FAILURE() << "the failed transfer returned success";
        } catch (const std::runtime_error &failure) {
            EXPECT_NE(std::string(failure.what()).find(std::to_string(runner.status)), std::string::npos)
                << failure.what();
        }
    }
    EXPECT_EQ(runner.drain_calls, 1);
    EXPECT_EQ(domain.done_count, 0u);
    EXPECT_EQ(domain.acquired, 1);
    EXPECT_EQ(domain.released, late_completion ? 1 : 0);
    EXPECT_EQ(worker.native_runs_.size(), late_completion ? 0u : 1u);
    EXPECT_EQ(runner.reservations(), late_completion ? 0u : 1u);
    EXPECT_EQ(runner.caller_buffer_retained_count(), late_completion ? 0u : 1u);
    if (late_completion) {
        EXPECT_EQ(domain.drain_checks, pending_checks + 1);
        auto &transport = control();
        transport.selected_free = reinterpret_cast<void *>(runner.operand);
        worker.free(runner.operand);
        EXPECT_EQ(transport.selected_free_attempts, 1u);
        EXPECT_EQ(runner.caller_buffer_count(), 0u);
        EXPECT_EQ(domain.sq_tail[domain.row], domain.sq_head[domain.row]);
        EXPECT_EQ(domain.cq_tail[domain.row], domain.sq[domain.row].submittedWqeCount);
    } else {
        EXPECT_THROW(worker.free(runner.operand), std::runtime_error);
        EXPECT_THROW(worker.require_comm_reclamation_safe(), std::runtime_error);
        EXPECT_EQ(runner.caller_buffer_count(), 1u);
    }
    // An unproved queue stays live until process exit. Never inject a cleanup proof.
    exit_with_test_result();
}

TEST(HbgUrmaNativeRun, SuccessfulChipWaitRetriesThePendingGet) {
    EXPECT_EXIT(exercise_failure(false, UrmaFault::HostReceipt, true), ::testing::ExitedWithCode(0), "");
}
TEST(HbgUrmaNativeRun, PhysicalWaitCompletesTheLateGetWithoutPoisoningTheLane) {
    EXPECT_EXIT(
        exercise_failure(false, UrmaFault::HostReceipt, true, HostReceiptWait::Physical), ::testing::ExitedWithCode(0),
        ""
    );
}
TEST(HbgUrmaNativeRun, PhysicalWaitTimeoutKeepsTheGetForAnotherWait) {
    EXPECT_EXIT(
        exercise_failure(false, UrmaFault::HostReceipt, true, HostReceiptWait::AfterTimeout),
        ::testing::ExitedWithCode(0), ""
    );
}
TEST(HbgUrmaNativeRun, FailedFinishReleasesOnceAfterReceiptedGet) {
    EXPECT_EXIT(exercise_failure(false, UrmaFault::Finish, true), ::testing::ExitedWithCode(0), "");
}
TEST(HbgUrmaNativeRun, ErrorCqeKeepsPutInputAndDomain) {
    EXPECT_EXIT(exercise_failure(true, UrmaFault::BadCqe, false), ::testing::ExitedWithCode(0), "");
}
TEST(HbgUrmaNativeRun, PendingTimeoutKeepsGetOutputAndDomain) {
    EXPECT_EXIT(exercise_failure(false, UrmaFault::Timeout, false), ::testing::ExitedWithCode(0), "");
}
TEST(HbgUrmaNativeRun, LateCqeAfterTimeoutReleasesPutInputOnce) {
    EXPECT_EXIT(exercise_failure(true, UrmaFault::Timeout, true), ::testing::ExitedWithCode(0), "");
}
}  // namespace
}  // namespace native_admission_test
