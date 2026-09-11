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

#include <gtest/gtest.h>

#include <array>
#include <cstring>
#include <memory>
#include <vector>

#include "callable.h"
#include "runtime.h"
#include "scheduler/scheduler.h"

#define private public
#include "scheduler/scheduler_context.h"
#undef private

namespace {
int register_writes = 0;
volatile uint32_t *last_register = nullptr;
uint32_t last_token = 0;
void callable_a() {}
void callable_b() {}
}  // namespace

// Only platform register I/O is replaced; every scheduler consumer is linked
// from its production translation unit.
uint64_t read_reg(uint64_t, RegId) { return 0; }
uint32_t reg_load_acquire(const volatile uint32_t *p) { return *p; }
void reg_store_release(volatile uint32_t *p, uint32_t value) {
    ++register_writes;
    last_register = p;
    last_token = value;
}
extern "C" uint64_t get_platform_regs() { return 0; }
extern "C" uint64_t get_platform_pmu_reg_addrs() { return 0; }
int platform_aicpu_affinity_thread_idx() { return 0; }
void platform_init_aicore_regs(uint64_t) {}
uint64_t platform_aicore_exit_deadline() { return 0; }
uint32_t platform_get_physical_cores_count() { return PLATFORM_MAX_CORES; }

extern void reset_test_reg_stub();
extern uint64_t get_test_reg_stub_value();

namespace {

class TmrSchedulerExecutionInputsTest : public ::testing::Test {
protected:
    void SetUp() override {
        sm_handle = SharedMemoryHandle::create_and_init_default(sm_arena);
        ASSERT_NE(sm_handle, nullptr);
        auto layout = SchedulerState::reserve_layout(sched_arena);
        ASSERT_NE(sched_arena.commit(), nullptr);
        ASSERT_TRUE(sched.init_data_from_layout(layout, sched_arena, sm_handle->header));
        sched.wire_arena_pointers(layout, sched_arena);
        context->sched_ = &sched;
        resident->dev.worker_count = 3;
        const uint8_t binary[] = {1, 2, 3, 4};
        a = make_callable<CORE_MAX_TENSOR_ARGS>(nullptr, 0, binary, sizeof(binary));
        b = make_callable<CORE_MAX_TENSOR_ARGS>(nullptr, 0, binary, sizeof(binary));
        reinterpret_cast<CoreCallable *>(a.data())->set_resolved_addr(reinterpret_cast<uint64_t>(&callable_a));
        reinterpret_cast<CoreCallable *>(b.data())->set_resolved_addr(reinterpret_cast<uint64_t>(&callable_b));
        table_a[0] = reinterpret_cast<uint64_t>(a.data());
        table_b[0] = reinterpret_cast<uint64_t>(b.data());
        init_slot();
    }

    void TearDown() override {
        sched.destroy();
        sched_arena.release();
        sm_arena.release();
    }

    void init_slot() {
        std::memset(&slot, 0, sizeof(slot));
        std::memset(&task, 0, sizeof(task));
        std::memset(&payload, 0, sizeof(payload));
        slot.task = &task;
        slot.payload = &payload;
        slot.task_state.store(CHIP_TASK_PENDING);
        slot.active_mask = ActiveMask(SUBTASK_MASK_AIC);
        slot.logical_block_num = 1;
        slot.total_required_subtasks = 1;
        payload.scalar_count = 1;
        payload.scalars[0] = 73;
        register_writes = 0;
        last_register = nullptr;
        last_token = 0;
        reset_test_reg_stub();
    }

    void initialize(simpler::tmr::CallableTableView table) {
        ASSERT_EQ(context->pre_handshake_init(resident.get(), 1, 1, 0, table, sm_handle->header), 0);
        EXPECT_EQ(context->get_function_bin_addr(0), table.lookup(0));
        // Hardware discovery writes this classification before the serial
        // post-handshake initializer; the test does not run a hardware handshake.
        context->core_type_compact_[0] = static_cast<uint8_t>(CoreType::AIC);
        context->core_type_compact_[1] = static_cast<uint8_t>(CoreType::AIV);
        context->core_type_compact_[2] = static_cast<uint8_t>(CoreType::AIV);
        ASSERT_EQ(context->post_handshake_init(resident.get(), table), 0);
        EXPECT_EQ(context->get_function_bin_addr(0), table.lookup(0));
        context->core_exec_states_[0].reg_addr = reinterpret_cast<uint64_t>(registers.data());
    }

    void expect_unpublished() {
        EXPECT_EQ(register_writes, 0);
        EXPECT_EQ(get_test_reg_stub_value(), 0u);
        EXPECT_EQ(sched.sm_header->sched_error_code.load(), SIMPLER_ERROR_INVALID_ARGS);
        EXPECT_EQ(context->core_exec_states_[0].running_slot_state, nullptr);
        EXPECT_EQ(context->core_exec_states_[0].pending_slot_state, nullptr);
        EXPECT_TRUE(context->core_trackers_[0].is_aic_core_idle(0));
        for (int w = 0; w < EARLY_DISPATCH_CORE_MASK_WORDS; ++w) {
            EXPECT_EQ(payload.staged_core_mask[w].load(), 0u);
        }
        EXPECT_EQ(sched.early_dispatch_doorbell_table[0].addr, 0u);
        EXPECT_EQ(sched.early_dispatch_doorbell_table[0].token, 0u);
    }

    std::unique_ptr<Runtime> resident{std::make_unique<Runtime>()};
    std::unique_ptr<SchedulerContext> context{std::make_unique<SchedulerContext>()};
    SchedulerState sched;
    SharedMemoryHandle *sm_handle{nullptr};
    DeviceArena sm_arena;
    DeviceArena sched_arena;
    ChipTaskSlotState slot{};
    TaskDescriptor task{};
    TaskPayload payload{};
    std::array<uint32_t, 1024> registers{};
    std::vector<uint8_t> a;
    std::vector<uint8_t> b;
    std::array<uint64_t, 1> table_a{};
    std::array<uint64_t, 1> table_b{};
};

TEST_F(TmrSchedulerExecutionInputsTest, ExplicitTableSurvivesHandshakeAndOrdinaryPublicationAcrossABA) {
    for (const auto *table : {&table_a, &table_b, &table_a}) {
        init_slot();
        initialize({table->data(), table->size()});
        const auto handle = context->prepare_subtask_to_core(0, 0, slot, SubtaskSlot::AIC, false, 0, false);
        ASSERT_TRUE(handle.valid);
        const auto &dispatch = context->payload_per_core_[0][handle.reg_task_id & 1u];
        EXPECT_EQ(
            dispatch.function_bin_addr,
            table == &table_a ? reinterpret_cast<uint64_t>(&callable_a) : reinterpret_cast<uint64_t>(&callable_b)
        );
        EXPECT_EQ(dispatch.src_payload, 0u);
        EXPECT_EQ(dispatch.args[0], 73u);
        context->publish_subtask_to_core(handle, 0, 0);
        EXPECT_EQ(register_writes, 1);
        EXPECT_EQ(last_token, handle.reg_task_id);
        EXPECT_EQ(
            last_register, reinterpret_cast<volatile uint32_t *>(handle.reg_addr + reg_offset(RegId::DATA_MAIN_BASE))
        );
        EXPECT_EQ(resident->dev.func_id_to_addr_[0], 0u);
    }
}

TEST_F(TmrSchedulerExecutionInputsTest, ExplicitTableBuildsEarlyPayloadAcrossABA) {
    for (const auto *table : {&table_a, &table_b, &table_a}) {
        init_slot();
        initialize({table->data(), table->size()});
        payload.early_dispatch_state.store(EARLY_DISPATCH_STAGING);
        auto idle = CoreTracker::BitStates::bit(0);
        CoreTracker::BitStates pending;
        ASSERT_EQ(context->stage_consumer_blocks(0, &slot, ResourceShape::AIC, 0, 1, idle, pending), 1);
        const auto &core = context->core_exec_states_[0];
        const auto &dispatch = context->payload_per_core_[0][core.running_reg_task_id & 1u];
        EXPECT_EQ(
            dispatch.function_bin_addr,
            table == &table_a ? reinterpret_cast<uint64_t>(&callable_a) : reinterpret_cast<uint64_t>(&callable_b)
        );
        EXPECT_EQ(dispatch.src_payload, reinterpret_cast<uint64_t>(&payload));
        EXPECT_EQ(register_writes, 1);
        EXPECT_EQ(payload.staged_core_mask[0].load(), 1u);
        EXPECT_EQ(sched.early_dispatch_doorbell_table[0].addr, core.reg_addr);
        EXPECT_EQ(resident->dev.func_id_to_addr_[0], 0u);
    }
}

TEST_F(TmrSchedulerExecutionInputsTest, InvalidMappingDoesNotPublishOrdinaryOrReleasedEarlyWork) {
    const std::array<uint64_t, 1> null_entry{0};
    const std::array<uint64_t, 1> misaligned_entry{table_a[0] + 1};
    reinterpret_cast<CoreCallable *>(b.data())->set_resolved_addr(0);
    struct InvalidMapping {
        simpler::tmr::CallableTableView table;
        int32_t func_id;
    };
    const InvalidMapping invalid[] = {
        {{}, 0},
        {{table_a.data(), table_a.size()}, -1},
        {{table_a.data(), table_a.size()}, 1},
        {{null_entry.data(), null_entry.size()}, 0},
        {{misaligned_entry.data(), misaligned_entry.size()}, 0},
        {{table_b.data(), table_b.size()}, 0},
    };
    for (const auto &mapping : invalid) {
        for (bool early : {false, true}) {
            init_slot();
            sched.sm_header->sched_error_code.store(SIMPLER_ERROR_NONE);
            initialize(mapping.table);
            task.kernel_id[0] = mapping.func_id;
            if (early) {
                payload.early_dispatch_state.store(EARLY_DISPATCH_DISPATCHED);
                auto idle = CoreTracker::BitStates::bit(0);
                CoreTracker::BitStates pending;
                context->stage_consumer_blocks(0, &slot, ResourceShape::AIC, 0, 1, idle, pending);
            } else {
                auto handle = context->prepare_subtask_to_core(0, 0, slot, SubtaskSlot::AIC, false, 0, false);
                EXPECT_FALSE(handle.valid);
                context->publish_subtask_to_core(handle, 0, 0);
            }
            expect_unpublished();
        }
    }
}

TEST_F(TmrSchedulerExecutionInputsTest, InvalidMappingDoesNotPublishGatedSyncStartCompletionPath) {
    initialize({});
    payload.early_dispatch_state.store(EARLY_DISPATCH_STAGING);
    const auto result = context->stage_sync_start_cores(&slot, 1, 0, true, false);
    EXPECT_EQ(result.staged_blocks, 1);
    expect_unpublished();
}

TEST_F(TmrSchedulerExecutionInputsTest, ExplicitTableBuildsGatedSyncStartPayloadAcrossABA) {
    for (const auto *table : {&table_a, &table_b, &table_a}) {
        init_slot();
        initialize({table->data(), table->size()});
        payload.early_dispatch_state.store(EARLY_DISPATCH_STAGING);
        const auto result = context->stage_sync_start_cores(&slot, 1, 0, true, false);
        ASSERT_EQ(result.staged_blocks, 1);
        const auto &core = context->core_exec_states_[0];
        const auto &dispatch = context->payload_per_core_[0][core.running_reg_task_id & 1u];
        EXPECT_EQ(
            dispatch.function_bin_addr,
            table == &table_a ? reinterpret_cast<uint64_t>(&callable_a) : reinterpret_cast<uint64_t>(&callable_b)
        );
        EXPECT_EQ(dispatch.src_payload, reinterpret_cast<uint64_t>(&payload));
        EXPECT_EQ(register_writes, 1);
        EXPECT_EQ(payload.staged_core_mask[0].load(), 1u);
        EXPECT_EQ(sched.early_dispatch_doorbell_table[0].addr, core.reg_addr);
        EXPECT_EQ(resident->dev.func_id_to_addr_[0], 0u);
    }
}

TEST_F(TmrSchedulerExecutionInputsTest, ProgramHandshakeWrappersConsumeResidentTable) {
    resident->dev.func_id_to_addr_[0] = table_b[0];
    resident->set_gm_sm_ptr(sm_handle->header);
    ASSERT_EQ(context->pre_handshake_init(resident.get(), 1, 1, 0), 0);
    EXPECT_EQ(context->get_function_bin_addr(0), table_b[0]);
    context->core_type_compact_[0] = static_cast<uint8_t>(CoreType::AIC);
    context->core_type_compact_[1] = static_cast<uint8_t>(CoreType::AIV);
    context->core_type_compact_[2] = static_cast<uint8_t>(CoreType::AIV);
    ASSERT_EQ(context->post_handshake_init(resident.get()), 0);
    DispatchPayload dispatch{};
    ASSERT_TRUE(context->build_payload(dispatch, slot, SubtaskSlot::AIC, 0, false));
    EXPECT_EQ(dispatch.function_bin_addr, reinterpret_cast<uint64_t>(&callable_b));
}

}  // namespace
