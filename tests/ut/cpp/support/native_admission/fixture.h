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
#include <gtest/gtest.h>
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <future>
#include <functional>
#include <stdexcept>
#include <utility>
#include <limits>
#include <memory>
#include <vector>
#include "device_runner_base.h"
#include "native_run_context.h"
#include "runtime_c_api.h"
#include "task_args.h"
#include "transport.h"
#include "utils/retained_temp_bump.h"

namespace native_admission_test {
// The real prepare path needs runtime-specific orchestration entry points and
// arena-bank selection. These helpers configure its inputs; tests assert the
// resulting behavior directly in their own source files.
void *primary_entry_points();
void *alternate_entry_points();
void set_dep_gen_capture(bool enabled);
uint32_t arena_bank(uint32_t slot);

using native_admission_transport::control;
using Fact = WorkspaceManager::RunFact;
constexpr size_t kUnit = 1024;
inline int callbacks = 0;
inline bool refuse_after_callback = false;
inline std::function<void()> callback_hook;
extern thread_local bool refuse_next_host_allocation;

class Runner : public DeviceRunnerBase {
public:
    Runner() {
        device_id_ = 0;
        max_block_dim_ = block_dim_ = 1;
        EXPECT_EQ(install_staged_workspace(), 0);
        EXPECT_TRUE(set_run_resource_grant(6));
        enable_automatic_admission();
        CallableState callable;
        callable.hash = 1;
        callable.chip_buffer_hash = 1;
        callable.signature = {ArgDirection::IN};
        callable.host_orch_func_ptr = primary_entry_points();
        callables_[1] = std::move(callable);
        chip_callable_buffers_[1] = ChipCallableBuffer{};
    }
    void use_other_entry() { callables_[1].host_orch_func_ptr = alternate_entry_points(); }
    ~Runner() override { EXPECT_EQ(finalize(), 0); }
    bool can_accept_run() const override { return !admission_poisoned; }
    void recover_device_or_mark_unusable(int rc) override {
        ++recovery_calls;
        recovery_rc = rc;
        admission_poisoned = true;
        if (recovery_throws) throw std::runtime_error("recovery failed");
    }
    bool ready_to_join_launch() const override { return true; }
    int fill_persistent_arch_fields(KernelArgs *, uint64_t) override { return 0; }
    int prepare_execution(
        Runtime &runtime, const CallConfig &config, uint32_t slot, const NativeRunIdentity &identity,
        std::unique_ptr<PreparedExecution> *out
    ) override {
        *out = std::make_unique<PreparedExecution>(identity, runtime, config, slot);
        return 0;
    }
    int launch_refusal_rc{0};
    bool partial_launch{false};
    bool launch_throws{false};
    bool launch_throws_after_submission{false};
    bool forbid_post_submit_allocation{false};
    bool admission_poisoned{false};
    bool recovery_throws{false};
    int recovery_calls{0};
    int recovery_rc{0};
    int launch_calls{0};
    int drain_calls{0};
    LaunchTransactionResult launch_execution_impl(PreparedExecution &prepared, LaunchPermit permit) override {
        ++launch_calls;
        if (launch_throws) throw std::runtime_error("unknown launch progress");
        if (launch_refusal_rc != 0 && !partial_launch) {
            return LaunchTransactionResult{launch_refusal_rc, LaunchProgress::NotStarted, LaunchReceipt{}};
        }
        auto transaction = exact_launch_transaction(
            prepared.identity, std::move(permit),
            [&](LaunchProgressSink &sink) {
                sink.mark_submitted();
                return partial_launch ? launch_refusal_rc : 0;
            },
            [](LaunchProgressSink &sink) {
                sink.mark_submitted();
                return 0;
            }
        );
        if (launch_throws_after_submission) throw std::bad_alloc();
        if (forbid_post_submit_allocation) refuse_next_host_allocation = true;
        return transaction;
    }
    void abandon_prepared_execution(PreparedExecution &) noexcept override {}
    int poll_execution(const ActiveExecution &) override { return SIMPLER_NATIVE_RUN_POLL_NOT_READY; }
    DrainOutcome drain_execution(ActiveExecution &) override {
        ++drain_calls;
        return DrainOutcome::device_complete(0);
    }
    int finalize_impl() override { return finalize_common(); }
    void arm_host_dep_gen_capture(bool enabled) override { set_dep_gen_capture(enabled); }
    AutoAdmissionState::SlotPlan plan(uint32_t slot) const { return auto_admission_.plan(slot); }
    size_t charged() const { return workspace_.reserved_bytes(); }
    bool holds_epoch(uint64_t epoch) const { return workspace_.holds_references(epoch); }
    bool arena_referenced(uint32_t bank, size_t region) const {
        void *backing = arena_region_backing(bank, region);
        return backing != nullptr && workspace_.block_state(backing) == WorkspaceManager::BlockState::Referenced;
    }
    const simpler::dfx::HostPhaseRecordStore &phases(uint32_t slot) const { return host_phase_runs_[slot].records; }
    size_t reservations() const {
        size_t n = 0;
        for (const auto &r : native_run_reservations_)
            n += r.owner != nullptr;
        return n;
    }
    void retire(uint32_t slot, uint64_t epoch) {
        for (auto fact :
             {Fact::DrainAttempted, Fact::DrainProvedComplete, Fact::CopybackReturned, Fact::BindingsReleased,
              Fact::ContextDestroyed})
            note_workspace_run_fact(slot, epoch, fact);
    }
    void stop_serving() { workspace_.enter_closed(); }
};

struct Attempt {
    explicit Attempt(Runner &r, uint32_t slot, uint64_t epoch, size_t payload = 16) :
        runner(r),
        bytes(payload, 0x71) {
        storage = ::operator new(get_runtime_size(), std::align_val_t(get_runtime_alignment()));
        std::memset(storage, 0, get_runtime_size());
        descriptor.run_epoch = epoch;
        descriptor.generation = epoch;
        descriptor.dispatch_id = epoch;
        descriptor.pipeline_slot = slot;
        descriptor.arena_bank = arena_bank(slot);
        config.aicpu_thread_num = 2;
        for (int i = 0; i < RUNTIME_ENV_RING_COUNT; ++i) {
            config.runtime_env.ring_task_window[i] = 4;
            config.runtime_env.ring_heap[i] = 1024;
            config.runtime_env.ring_dep_pool[i] = 4;
        }
        ChipTensor tensor;
        uint32_t shape[] = {static_cast<uint32_t>(bytes.size())};
        tensor.init_external(bytes.data(), bytes.size(), shape, 1, DataType::UINT8, AddressSpace::HOST);
        args.add_tensor(tensor);
    }
    ~Attempt() {
        uint64_t magic = 0;
        std::memcpy(&magic, storage, sizeof(magic));
        if (magic) EXPECT_EQ(simpler_finalize_run(&runner, storage), 0);
        ::operator delete(storage, std::align_val_t(get_runtime_alignment()));
    }
    int prepare() { return simpler_prepare_run(&runner, storage, 1, &args, &config, &descriptor); }
    bool consume_release() { return consume_released_preparation(storage, get_runtime_size(), descriptor.run_epoch); }
    bool empty() const {
        uint64_t magic;
        std::memcpy(&magic, storage, sizeof(magic));
        return magic == 0;
    }
    Runner &runner;
    void *storage;
    std::vector<uint8_t> bytes;
    ChipStorageTaskArgs args;
    CallConfig config{};
    NativeRunDescriptor descriptor{};
};

class NativeAdmissionTest : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_EQ(control().bytes, 0u);
        control() = {};
        callbacks = 0;
        refuse_after_callback = false;
        callback_hook = {};
        refuse_next_host_allocation = false;
    }
    void TearDown() override {
        EXPECT_EQ(control().bytes, 0u);
        if (!diagnostic_path.empty()) std::filesystem::remove_all(diagnostic_path);
    }
    void diagnostics(Attempt &attempt) {
        const std::string path =
            std::string("native-admission-") + ::testing::UnitTest::GetInstance()->current_test_info()->name();
        ASSERT_TRUE(std::filesystem::create_directory(path));
        diagnostic_path = path;
        attempt.config.enable_dep_gen = 1;
        attempt.config.enable_chip_swimlane = static_cast<int>(ChipSwimlaneLevel::ORCH_PHASES);
        std::strncpy(attempt.config.output_prefix, diagnostic_path.c_str(), sizeof(attempt.config.output_prefix) - 1);
    }
    std::string diagnostic_path;
    static int staging(Runner &r, uint32_t slot, uint64_t epoch, size_t bytes) {
        DeviceRunnerBase::WorkspacePlanScope plan(slot, epoch);
        void *ptr = nullptr;
        size_t capacity = 0;
        return r.acquire_retained_temp(slot, bytes, &ptr, &capacity);
    }
    static void launched(Runner &r, uint32_t slot, uint64_t epoch) {
        r.note_run_resources_proven(slot, epoch);
        r.note_workspace_run_fact(slot, epoch, Fact::Launched);
    }
};

}  // namespace native_admission_test
