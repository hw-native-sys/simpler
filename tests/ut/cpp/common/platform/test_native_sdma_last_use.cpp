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

#include "fixture.h"
#define private public
#include "device_runner.h"
#undef private
#include "async_plan.h"
#include "async_execution_context.h"
#include "aicore_scheduler_error.h"
#include "backend/sdma/sdma_submission.h"
#include "scheduler/scheduler_async_execution.h"
#include "a5/runtime/host_build_graph/hbg_scheduler_test_support.h"

#include <array>
#include <unistd.h>

namespace native_admission_test {
namespace {
using namespace pto2::detail;
using ExecutionBackend = AsyncCompletionBackend<>;
constexpr uint64_t kEpoch = 20;
constexpr uint32_t kSlot = 1;
constexpr uint64_t kBytes = 4096;
constexpr uint64_t kBudget = 1 << 20;
enum class SdmaFault { PREFLIGHT, RECORD_EXPORT };

void require(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}

// Real caller-buffer borrows keep these addresses alive on failed runs.
struct SdmaAllocationFacts {
    std::array<void *, 3> allocations{}, buffers{};
};

struct SdmaIoFault {
    SdmaFault fault;
    uint64_t workspace{0}, group{0};
    uint32_t queues{0}, preflights{0}, built{0}, posted{0}, exported{0}, waited{0};
    bool preflight() {
        ++preflights;
        return fault != SdmaFault::PREFLIGHT;
    }
    bool build_session(uint32_t count, uint32_t index) {
        ++built;
        require(count == queues && index == group, "backend geometry differs from selected workspace");
        return true;
    }
    bool session_valid(uint32_t count, uint32_t index) const { return count == queues && index == group; }
    bool has_records() const { return true; }
    bool submit() {
        ++posted;
        return true;
    }
    bool event_valid(uint64_t bytes) const { return bytes == 32; }
    uint32_t record_count() const { return 1; }
    bool record(uint32_t, SdmaCompletionReceiptRecord *) {
        ++exported;
        return false;
    }
    bool wait() {
        ++waited;
        return true;
    }
};
struct CpuCache {
    std::vector<uint64_t> flushed;
    void flush_range(volatile void *address, uint32_t) { flushed.push_back(reinterpret_cast<uint64_t>(address)); }
    void finish() { std::atomic_thread_fence(std::memory_order_seq_cst); }
};

class SdmaFaultRunner : public DeviceRunner {
public:
    explicit SdmaFaultRunner(SdmaFault fault) :
        fault_(fault),
        storage_(2, 3, 1, {3, 1, 0, 0}),
        graph_(2) {
        device_id_ = 0;
        require(execution_mode_latch().latch(SIMPLER_MODE_PROGRAM) == 0, "execution mode");
        timeout_configuration_rc_ = 0;
        max_block_dim_ = block_dim_ = 1;
        require(install_staged_workspace() == 0 && set_run_resource_grant(6), "workspace admission");
        enable_automatic_admission();
        CallableState callable;
        callable.hash = callable.chip_buffer_hash = 1;
        callable.signature = {ArgDirection::INOUT, ArgDirection::INOUT, ArgDirection::INOUT};
        callable.host_orch_func_ptr = primary_entry_points();
        callables_[1] = std::move(callable);
        chip_callable_buffers_[1] = ChipCallableBuffer{};
        for (size_t i = 0; i < facts.buffers.size(); ++i) {
            facts.allocations[i] = allocate_caller_buffer(kBytes + 63);
            require(facts.allocations[i] != nullptr, "caller buffer");
            const auto aligned = (reinterpret_cast<uintptr_t>(facts.allocations[i]) + 63) & ~uintptr_t{63};
            facts.buffers[i] = reinterpret_cast<void *>(aligned);
            std::memset(facts.buffers[i], 0x5a, kBytes);
        }
    }
    int prepare_execution_streams(PreparedExecution &prepared, std::shared_ptr<const void> *lease) override {
        static int streams[2];
        prepared.streams = {&streams[0], &streams[1]};
        lease->reset();
        return 0;
    }
    int prepare_execution(
        Runtime &runtime, const CallConfig &config, uint32_t slot, const NativeRunIdentity &identity,
        std::unique_ptr<PreparedExecution> *out
    ) override {
        auto prepared = std::make_unique<PreparedExecution>(identity, runtime, config, slot);
        prepared->scheduler_runtime_mode = SCHEDULER_RUNTIME_MODE_RESIDENT_PENDING;
        prepared->resources_owned = true;
        auto *native = static_cast<NativeRunContext<DeviceRunnerBase> *>(native_storage);
        require(native != nullptr && native->identity().run_epoch == identity.run_epoch, "native identity");
        plan_image(slot, identity.run_epoch);
        submission_storage_[slot]->retain_replay_evidence = true;
        const int rc = ensure_device_run_result_region(slot, identity.run_epoch, prepared->kernel_args);
        if (rc != 0) return rc;
        *out = std::move(prepared);
        return 0;
    }
    LaunchTransactionResult launch_execution_impl(PreparedExecution &prepared, LaunchPermit permit) override {
        auto transaction = exact_launch_transaction(
            prepared.identity, std::move(permit),
            [](LaunchProgressSink &sink) {
                sink.mark_submitted();
                return 0;
            },
            [](LaunchProgressSink &sink) {
                sink.mark_submitted();
                return 0;
            }
        );
        run_progress_.claim(prepared.identity);
        (void)run_progress_.publish(prepared.identity, RunProgressState::Submitted);
        require(
            run_fence(prepared.pipeline_slot).prepare_resources() == 0 && arm_run_fence(prepared) == 0, "run fence"
        );
        require(
            record_run_boundary(prepared, RunCompletionFence::StreamRole::Aicore) == 0 &&
                record_run_boundary(prepared, RunCompletionFence::StreamRole::Aicpu) == 0,
            "stream boundaries"
        );
        run_sdk_fault();
        auto *result = static_cast<DeviceRunResultRegion *>(submission_storage_[prepared.pipeline_slot]->device_result);
        *result = {};
        result->verdict = static_cast<uint32_t>(DeviceRunVerdict::Error);
        result->completion_code = terminal_rc;
        result->code_source = static_cast<uint32_t>(DeviceRunCodeSource::Header);
        // Transport publication carries the real SDK -> scheduler mapped code.
        // There is deliberately no external-engine stop or core-retired payload.
        result->published = prepared.identity.run_epoch;
        return transaction;
    }
    bool holds_epoch() const { return workspace_.holds_references(kEpoch); }
    bool reset_blocked() const { return prepared_resources_block_fatal_reset(); }
    std::shared_ptr<const PreparedRunResources> retained() const {
        for (const auto &entry : prepared_uses_)
            if (entry.identity.run_epoch == kEpoch) return entry.use.resources();
        return {};
    }
    bool retained_image_is_same(const PreparedExecution &prepared) const {
        const void *host = nullptr;
        size_t bytes = 0;
        return prepared_scheduler_state_view(prepared, &host, &bytes) && host == scheduler_host_ &&
               bytes >= scheduler_bytes_ && async_.base == static_cast<const uint8_t *>(host) + async_offset_;
    }
    uint64_t address(size_t i) const { return reinterpret_cast<uint64_t>(facts.buffers[i]); }
    SdmaAllocationFacts facts;
    void *native_storage{nullptr};
    int32_t first_sdk_error{0}, terminal_rc{0};
    uint32_t outstanding_reservation{0};
    uint32_t receipt_count{0}, posted{0}, built{0}, exported{0}, preflights{0};
    AsyncExecutionContext *canonical{nullptr};
    SchedulerAsyncGroupWriter *writer{nullptr};

private:
    void plan_image(uint32_t slot, uint64_t epoch) {
        metadata_.resize(2);
        for (uint32_t task = 0; task < 2; ++task) {
            auto &meta = metadata_[task];
            meta.flags = SCHEDULER_TASK_EXECUTABLE;
            meta.active_mask = LaneMask::aiv0();
            meta.logical_block_num = 1;
            meta.kernel_ids[1] = 1;
            meta.timing_slot = -1;
            storage_.metadata[task] = meta;
            graph_.executable(task, 1, task == 0 ? std::vector<int32_t>{} : std::vector<int32_t>{0});
        }
        std::string error;
        SchedulerAsyncHostPlan plan;
        require(scheduler_plan_async_state(metadata_, epoch, kBudget, &plan, &error), error.c_str());
        const uint64_t summary_offset = sizeof(SchedulerRunControl);
        async_offset_ = summary_offset + sizeof(SchedulerGraphSummary);
        scheduler_bytes_ = async_offset_ + plan.image.size();
        void *device = nullptr;
        require(
            acquire_scheduler_state_storage(slot, scheduler_bytes_, 128, &device, &scheduler_host_) == 0,
            "retained scheduler allocation"
        );
        std::memset(scheduler_host_, 0, scheduler_bytes_);
        auto *run = static_cast<SchedulerRunControl *>(scheduler_host_);
        run->expected_task_count = 2;
        run->graph_summary_offset = summary_offset;
        auto *summary =
            reinterpret_cast<SchedulerGraphSummary *>(static_cast<uint8_t *>(scheduler_host_) + summary_offset);
        summary->run_epoch = epoch;
        summary->async_layout_offset = async_offset_;
        auto *base = static_cast<uint8_t *>(scheduler_host_) + async_offset_;
        std::memcpy(base, plan.image.data(), plan.image.size());
        async_ = {base, plan.layout};
    }
    void run_sdk_fault() {
        auto view = graph_.graph();
        for (uint32_t worker = 0; worker < 3; ++worker) {
            auto &context = storage_.contexts[worker];
            context.is_scheduler = 1;
            context.scheduler_count = 3;
            for (uint32_t lane = 0; lane < 3; ++lane)
                context.cluster_worker_ids[lane] = lane;
            storage_.test_graphs[worker] = view;
        }
        require(
            scheduler_init_spmd_task_records(storage_.scheduler_state->base(), storage_.layout, kEpoch),
            "scheduler records"
        );
        auto *base = storage_.scheduler_state->base();
        auto *local = storage_.local_context(&storage_.contexts[1]);
        auto &slot = local->slots[0];
        slot.state = SchedulerDispatchSlotState::READY;
        slot.generation = 1;
        slot.task_ref = 0;
        slot.block_idx = 0;
        slot.subtask_slot = 1;
        local->local_ready_mask = 1;
        auto *payload =
            scheduler_state_at<DispatchPayload>(base, local->dispatch_payload_offset(local->config.self_lane, 0));
        *payload = {};
        payload->local_context.block_num = 1;
        payload->args[SPMD_LOCAL_CONTEXT_INDEX] = reinterpret_cast<uint64_t>(&payload->local_context);
        payload->args[SPMD_GLOBAL_CONTEXT_INDEX] = reinterpret_cast<uint64_t>(&payload->global_context);
        SchedulerAsyncExecutionState<ExecutionBackend> state;
        state.view = async_;
        require(
            scheduler_async_execution_bind(&state, view, base, local, storage_.run_control, 0), "execution binding"
        );
        canonical = &payload->async_context;
        writer = reinterpret_cast<SchedulerAsyncGroupWriter *>(async_.base + async_.layout().group_writer_offset);
        AsyncCtx verified_writer;
        require(async_execution_context_load(*payload, &verified_writer), "canonical execution context");
        require(
            verified_writer.execution_context == canonical &&
                verified_writer.completion_error_code == &writer->error_code,
            "canonical writer identity"
        );
        SdmaIoFault io{fault_};
        io.workspace = address(2);
        io.group = 0;
        io.queues = 1;
        CpuCache cache;
        auto sink = [&](uint32_t, uint32_t, const SdmaCompletionReceiptRecord &) {
            ++receipt_count;
        };
        auto submit = [&] {
            return sdma_submit(
                verified_writer, SdmaOp::TGET, true, address(2), address(1), address(0) + 128, 32, 0, io, cache, sink
            );
        };
        require(!submit(), "fault must fail SDK entry");
        first_sdk_error = writer->error_code;
        require(first_sdk_error != 0, "original SDK error");
        const auto before_retry_posts = io.posted;
        require(!submit(), "sticky SDK error");
        require(
            io.posted == before_retry_posts && writer->error_code == first_sdk_error,
            "retry cannot replace error or post"
        );
        require(
            canonical->task_ref == 0 && canonical->subtask_slot == 1 &&
                canonical->region_address == reinterpret_cast<uint64_t>(async_.base),
            "canonical execution identity"
        );
        outstanding_reservation = writer->reservation.limit;
        posted = io.posted;
        built = io.built;
        exported = io.exported;
        preflights = io.preflights;
        require(receipt_count == 0 && writer->count == 0 && io.waited == 0, "no successful completion publication");
        require(
            !scheduler_async_execution_returned(&state, view, base, local, storage_.run_control, 0),
            "runtime return must propagate fault"
        );
        require(
            aicore_scheduler_runtime_error_code(scheduler_read_error(storage_.run_control)) == first_sdk_error &&
                storage_.run_control->error_task_ref == 0,
            "actual scheduler first-error authority"
        );
        auto *guards = reinterpret_cast<AsyncSdmaSubmissionGuard *>(
            async_.base + async_.layout().sdma_guard_context_offset + sizeof(AsyncSdmaGuardContext)
        );
        require(
            scheduler_gm_query(guards[0].word) == (fault_ == SdmaFault::PREFLIGHT ? 0u : 2u),
            "post failure poisons the queue while preflight failure holds no queue"
        );
        require(
            scheduler_task_control_at(base, local, 0)->state != static_cast<int64_t>(SchedulerTaskState::DONE) &&
                scheduler_task_control_at(base, local, 1)->state != static_cast<int64_t>(SchedulerTaskState::DONE),
            "no producer or consumer TaskDone"
        );
        terminal_rc = -aicore_scheduler_runtime_error_code(scheduler_read_error(storage_.run_control));
    }
    SdmaFault fault_;
    scheduler_test::FixtureStorage storage_;
    scheduler_test::BasicGraphBuffer<2> graph_;
    std::vector<SchedulerTaskMetadata> metadata_;
    SchedulerAsyncRuntimeView async_{};
    void *scheduler_host_{nullptr};
    uint64_t scheduler_bytes_{0}, async_offset_{0};
};

[[noreturn]] void check_fault_retention(SdmaFault fault) {
    try {
        SdmaFaultRunner runner(fault);
        Attempt attempt(runner, kSlot, kEpoch);
        attempt.args = {};
        for (void *buffer : runner.facts.buffers) {
            ChipTensor tensor;
            const uint32_t shape[]{kBytes};
            tensor.init_external(buffer, kBytes, shape, 1, DataType::UINT8, AddressSpace::DEVICE);
            attempt.args.add_tensor(tensor);
        }
        const std::array<CallerDeviceBuffers::Span, 3> spans{
            {{runner.address(0), kBytes}, {runner.address(1), kBytes}, {runner.address(2), kBytes}}
        };
        // Native callers establish this borrow before prepare; UNKNOWN is not
        // permission for the caller to release it after finalize.
        require(runner.borrow_caller_buffers(kSlot + 1, spans.data(), spans.size()), "caller borrow setup");
        runner.native_storage = attempt.storage;
        require(attempt.prepare() == 0, "native prepare");
        auto *native = static_cast<NativeRunContext<DeviceRunnerBase> *>(attempt.storage);
        const auto prepared_owner = native->prepared_execution->prepared_resources;
        require(
            prepared_owner && runner.retained_image_is_same(*native->prepared_execution), "same retained SDK image"
        );
        require(simpler_launch_run(&runner, attempt.storage) == 0, "transport launch");
        require(
            runner.first_sdk_error == SIMPLER_ERROR_ASYNC_COMPLETION_INVALID &&
                runner.terminal_rc == -runner.first_sdk_error,
            "semantic error bridge"
        );
        require(
            runner.outstanding_reservation == (fault == SdmaFault::PREFLIGHT ? 0u : 1u) &&
                runner.posted == (fault == SdmaFault::PREFLIGHT ? 0u : 1u) && runner.built == runner.posted &&
                runner.exported == runner.posted && runner.preflights == 1 && runner.receipt_count == 0,
            "fault stage evidence"
        );
        require(simpler_wait_run(&runner, attempt.storage) == runner.terminal_rc, "Host must preserve original error");
        require(
            !native->physical_complete && !native->execution_retired && native->active_execution &&
                native->active_execution->prepared->resources_owned &&
                runner.retained_image_is_same(*native->active_execution->prepared),
            "unproved execution retained"
        );
        require(
            simpler_finalize_run_with_delivery(&runner, attempt.storage, 0) == runner.terminal_rc && !attempt.empty(),
            "unknown final last-use"
        );
        require(
            runner.retained() && native->active_execution->prepared->prepared_resources == prepared_owner &&
                runner.holds_epoch() && runner.reset_blocked() && !runner.can_accept_run(),
            "native backing and admission retention"
        );
        for (void *buffer : runner.facts.allocations)
            require(
                runner.free_caller_buffer(buffer) == PTO_RUNTIME_ERR_INVALID_STATE,
                "actual caller borrow must refuse release"
            );
        Attempt next(runner, 2, kEpoch + 1);
        next.args = {};
        for (void *buffer : runner.facts.buffers) {
            ChipTensor tensor;
            const uint32_t shape[]{kBytes};
            tensor.init_external(buffer, kBytes, shape, 1, DataType::UINT8, AddressSpace::DEVICE);
            next.args.add_tensor(tensor);
        }
        require(next.prepare() != 0, "next admission must refuse");
        require(finalize_device(&runner) != 0 && !control().retired, "raw reset must refuse");
        require(
            runner.writer->error_code == runner.first_sdk_error &&
                runner.writer->reservation.limit == runner.outstanding_reservation && runner.writer->count == 0,
            "fault evidence survives Host refusal"
        );
        // Deliberately retained failing carriers cannot be destructed as healthy
        // runs. Child exit isolates CPU allocations; it is not a DMA stop proof.
        _exit(::testing::Test::HasFailure() ? 99 : 0);
    } catch (const std::exception &error) {
        fprintf(stderr, "SDMA retention bridge: %s\n", error.what());
        _exit(90);
    }
}

TEST_F(NativeAdmissionTest, SdmaBeforeSubmitFaultKeepsTheSameUnqualifiedNativeOwner) {
    EXPECT_EXIT(check_fault_retention(SdmaFault::PREFLIGHT), ::testing::ExitedWithCode(0), "");
}
TEST_F(NativeAdmissionTest, SdmaAfterSubmitFaultKeepsDebtAndTheSameUnqualifiedNativeOwner) {
    EXPECT_EXIT(check_fault_retention(SdmaFault::RECORD_EXPORT), ::testing::ExitedWithCode(0), "");
}
}  // namespace
}  // namespace native_admission_test
