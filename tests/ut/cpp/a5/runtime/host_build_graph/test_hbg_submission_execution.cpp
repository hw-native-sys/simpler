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

#include "inner_kernel.h"
#include <gtest/gtest.h>
#include <array>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>
#include "async_plan.h"
#include "submission_bundle_plan.h"
#include "hbg_scheduler_test_support.h"
#include "scheduler/scheduler_submission_execution.h"

namespace {
constexpr uint64_t kEpoch = 307;
constexpr uint64_t kBudget = 1 << 24;

struct Backend {
    SchedulerSubmissionProvenanceView proof{};
    uint64_t record_address{0};
    bool bind_condition(
        const SchedulerAsyncGroupIdentity &identity, uint32_t index, const DeferredCompletionEntry &entry,
        SchedulerAsyncConditionBinding *out
    ) {
        if (identity.run_epoch != kEpoch || index != 0 || entry.addr != record_address) return false;
        *out = {0, 0};
        return true;
    }
    bool bind_condition_at_seal(
        const SchedulerAsyncGroupIdentity &identity, uint32_t index, const DeferredCompletionEntry &entry,
        const SchedulerSubmissionSealIdentity &sealing, SchedulerAsyncConditionBinding *out
    ) {
        SchedulerSubmissionProvenance record{};
        SchedulerSubmissionWorkspace workspace{};
        if (!scheduler_submission_read_seal_provenance(proof, sealing, index, entry, &record, &workspace) ||
            !bind_condition(identity, index, entry, out))
            return false;
        return record.use_index == out->binding_index;
    }
};

struct Publisher {
    bool publish_backend_metadata() { return true; }
    void flush_evidence(volatile void *, uint32_t) {}
    void finish() {}
};

struct Fixture {
    Fixture(
        uint32_t mask = 2, uint32_t blocks = 1, bool sync_start = false, uint32_t deferred_mask = 7, bool skip = false,
        bool native_child = false, bool default_guards = false, uint32_t dynamic_mask = 0
    ) :
        execution_ref(native_child ? 1 : 0),
        storage(
            4, 6, blocks * 3 + 1,
            {6, 2, 1u + (blocks > 1 || sync_start || __builtin_popcount(mask) > 1 ? 1u : 0u),
             __builtin_popcount(mask) > 1 ? 1u : 0u}
        ),
        image(nullptr, &std::free) {
        for (uint32_t task = 0; task < 4; ++task) {
            storage.metadata[task] = {};
            storage.metadata[task].active_mask = LaneMask::none();
        }
        auto &meta = storage.metadata[execution_ref];
        meta.active_mask = LaneMask((mask & 1) != 0, (mask & 2) != 0, (mask & 4) != 0);
        meta.logical_block_num = blocks;
        meta.flags = SCHEDULER_TASK_EXECUTABLE | (blocks > 1 ? SCHEDULER_TASK_SPMD : 0) |
                     (meta.active_mask.is_mix() ? SCHEDULER_TASK_MIX : 0) |
                     (sync_start ? SCHEDULER_TASK_SYNC_START : 0);
        meta.timing_slot = -1;
        for (uint32_t lane = 0; lane < 3; ++lane)
            meta.kernel_ids[lane] = (mask & (1u << lane)) ? 1 : UINT16_MAX;
        graph.mixed(execution_ref, meta.active_mask);
        if (skip) {
            meta.flags |= SCHEDULER_TASK_HAS_PREDICATE;
            graph.predicate(
                execution_ref, reinterpret_cast<uint64_t>(&predicate), 8, static_cast<uint8_t>(PredicateOp::EQ), 1
            );
        }
        graph.executable(3, 0);
        storage.metadata[3].flags = SCHEDULER_TASK_EXECUTABLE | SCHEDULER_TASK_SPMD | SCHEDULER_TASK_SYNC_START;
        storage.metadata[3].active_mask = LaneMask::aic();
        storage.metadata[3].logical_block_num = 2;
        storage.metadata[3].kernel_ids[0] = 1;
        const std::vector<SchedulerTaskMetadata> metadata(storage.metadata, storage.metadata + 4);
        caps.resize(4);
        for (size_t task = 0; task < 4; ++task)
            for (uint32_t lane = 0; lane < 3; ++lane) {
                const bool deferred = metadata[task].active_mask.has(static_cast<SubtaskSlot>(lane)) &&
                                      (task == 3 || (deferred_mask & (1u << lane)) != 0);
                caps[task][lane] = {
                    CORE_ASYNC_CAPABILITY_VERSION, deferred ? CoreAsyncMode::DEFERRED : CoreAsyncMode::SYNCHRONOUS,
                    deferred ? 2u : 0u, deferred ? CORE_ASYNC_CONDITION_TYPES : 0u
                };
            }
        std::vector<SchedulerSubmissionExecutionUse> uses;
        for (uint32_t block = 0; block < blocks; ++block)
            for (uint32_t lane = 1; lane < 3; ++lane)
                if ((mask & (1u << lane)) && !(dynamic_mask & (1u << lane)))
                    uses.push_back({execution_ref, block, lane, 0, 0});
        for (size_t task = 0; task < metadata.size(); ++task)
            for (uint32_t block = 0; block < metadata[task].logical_block_num; ++block)
                for (uint32_t lane = 0; lane < 3; ++lane)
                    if (metadata[task].active_mask.has(static_cast<SubtaskSlot>(lane))) {
                        const bool dynamic =
                            static_cast<int64_t>(task) == execution_ref && (dynamic_mask & (1u << lane)) != 0;
                        use_sets.push_back(
                            {static_cast<int64_t>(task), block, lane, dynamic ? SCHEDULER_ASYNC_INDEX_NONE : 0,
                             dynamic ? 0u : 1u}
                        );
                    }
        backend.record_address = reinterpret_cast<uint64_t>(workspace_storage.data());
        const SchedulerSubmissionWorkspaceInput workspace{
            reinterpret_cast<uint64_t>(workspace_storage.data()), workspace_storage.size(), kEpoch, 1, 48,
            SchedulerSubmissionPoolQualification::EXCLUSIVE_RUN
        };
        std::string error;
        if (!scheduler_plan_submission_bundles(
                metadata, caps, {workspace}, uses, use_sets, kEpoch, kBudget, &plan, &error
            ))
            throw std::runtime_error(error);
        SchedulerAsyncAuxiliaryBytes auxiliary{};
        auxiliary.uses = sizeof(AsyncConditionContract);
        auxiliary.scratch = plan.image.size();
        auxiliary.sdma_guard_count = default_guards ? ASYNC_SDMA_CHANNELS : 0;
        if (!scheduler_plan_async_state(metadata, caps, kEpoch, kBudget, auxiliary, &async_plan, &error))
            throw std::runtime_error(error);
        image.reset(std::aligned_alloc(128, async_plan.image.size()));
        if (!image) throw std::bad_alloc();
        std::memcpy(image.get(), async_plan.image.data(), async_plan.image.size());
        async = {static_cast<uint8_t *>(image.get()), async_plan.layout};
        std::memcpy(async.base + async.layout.scratch_offset, plan.image.data(), plan.image.size());
        EXPECT_TRUE(scheduler_submission_provenance_mount(async.base, async.layout, &backend.proof));
        for (uint64_t i = 0; i < async.layout.group_count; ++i) {
            auto *group = scheduler_async_runtime_detail::at<SchedulerAsyncGroupIdentity>(
                async, async.layout.group_identity_offset, i
            );
            const bool dynamic = group->task_ref == execution_ref && (dynamic_mask & (1u << group->subtask_slot)) != 0;
            group->binding_uses_offset = dynamic ? SCHEDULER_ASYNC_USES_DYNAMIC : async.layout.uses_offset;
            group->binding_use_count = dynamic ? 0 : 1;
        }
        graph_view = graph.graph();
        if (native_child) initialize_native_identity();
        EXPECT_TRUE(scheduler_init_spmd_task_records(base(), storage.layout, kEpoch));
        for (uint32_t worker = 0; worker < 6; ++worker) {
            auto &wire = storage.contexts[worker];
            wire.is_scheduler = 1;
            wire.scheduler_count = 6;
            for (uint32_t lane = 0; lane < 3; ++lane)
                wire.cluster_worker_ids[lane] = worker / 3 * 3 + lane;
            storage.test_graphs[worker] = view();
            locals[worker] = storage.local_context(&wire);
            states[worker].view = async;
            EXPECT_TRUE(scheduler_submission_execution_mount(states[worker], &submissions[worker]));
        }
    }
    SchedulerGraphView view() const { return graph_view; }
    void initialize_native_identity() {
        SchedulerNativeGraphHeader header{};
        header.magic = SCHEDULER_NATIVE_GRAPH_MAGIC;
        header.version = SCHEDULER_NATIVE_GRAPH_VERSION;
        header.header_bytes = sizeof(header);
        header.node_count = 4;
        header.original_task_count = 2;
        header.definition_count = header.instance_count = 1;
        header.control_inbox_count = 6;
        header.run_epoch = kEpoch;
        uint64_t cursor = sizeof(SchedulerRunControl);
        uint64_t header_offset = 0;
        const auto reserve = [&](uint64_t bytes, uint64_t &offset) {
            EXPECT_TRUE(scheduler_layout_reserve(&cursor, bytes, 128, &offset));
        };
        reserve(sizeof(header), header_offset);
        reserve(4 * sizeof(SchedulerNodeBinding), header.bindings_offset);
        reserve(2 * sizeof(int64_t), header.completion_refs_offset);
        reserve(sizeof(SchedulerGraphTemplate), header.templates_offset);
        reserve(sizeof(SchedulerGraphInstance), header.instances_offset);
        SchedulerGraphTemplate definition{};
        definition.definition_bytes = sizeof(GraphDefinition) + 2 * sizeof(int32_t);
        definition.task_count = definition.sink_count = definition.execution_count = 1;
        definition.graph_key = 777;
        reserve(definition.definition_bytes, definition.definition_offset);
        reserve(sizeof(SchedulerGraphChildPlan), definition.child_plans_offset);
        reserve(sizeof(uint16_t), definition.sinks_offset);
        EXPECT_TRUE(scheduler_layout_checked_align(cursor, 128, &header.readonly_end_offset));
        cursor = header.readonly_end_offset;
        reserve(6 * sizeof(SchedulerReadyInbox), header.control_inboxes_offset);
        EXPECT_TRUE(scheduler_layout_checked_align(cursor, 128, &header.image_bytes));
        native_image.reset(std::aligned_alloc(128, header.image_bytes));
        ASSERT_NE(native_image, nullptr);
        std::memset(native_image.get(), 0, header.image_bytes);
        auto *native_base = static_cast<uint8_t *>(native_image.get());
        *reinterpret_cast<SchedulerNativeGraphHeader *>(native_base + header_offset) = header;
        auto *bindings = reinterpret_cast<SchedulerNodeBinding *>(native_base + header.bindings_offset);
        bindings[0] = {0, 0, SchedulerNodeKind::GRAPH_ENTER, {}};
        bindings[1] = {0, 0, SchedulerNodeKind::CHILD_KERNEL, {}};
        bindings[2] = {0, 0, SchedulerNodeKind::GRAPH_EXIT, {}};
        bindings[3] = {1, SCHEDULER_NATIVE_NO_INSTANCE, SchedulerNodeKind::TOP_KERNEL, {}};
        auto *completions = reinterpret_cast<int64_t *>(native_base + header.completion_refs_offset);
        completions[0] = 2;
        completions[1] = 3;
        SchedulerGraphInstance instance{};
        instance.child_begin = 1;
        instance.child_count = 1;
        instance.exit_ref = 2;
        *reinterpret_cast<SchedulerGraphInstance *>(native_base + header.instances_offset) = instance;
        GraphDefinition wire{};
        wire.graph_key = definition.graph_key;
        wire.total_bytes = definition.definition_bytes;
        wire.task_count = 1;
        wire.off_fanin_offsets = sizeof(GraphDefinition);
        definition.fanin_offsets_offset = definition.definition_offset + wire.off_fanin_offsets;
        *reinterpret_cast<GraphDefinition *>(native_base + definition.definition_offset) = wire;
        *reinterpret_cast<SchedulerGraphTemplate *>(native_base + header.templates_offset) = definition;
        auto *child = reinterpret_cast<SchedulerGraphChildPlan *>(native_base + definition.child_plans_offset);
        child->metadata = storage.metadata[execution_ref];
        child->predicate_offset = SCHEDULER_NATIVE_NO_PREDICATE;
        graph_view.original_task_count = 2;
        graph_view.scheduler_state_address = reinterpret_cast<uint64_t>(native_image.get());
        graph_view.scheduler_state_bytes = header.image_bytes;
        graph_view.native_header_offset = header_offset;
        EXPECT_TRUE(scheduler_graph_view_valid(graph_view));
        uint64_t token = 0;
        EXPECT_EQ(scheduler_node_identity(graph_view, execution_ref, &token), SchedulerGraphResult::OK);
        EXPECT_EQ(token, TaskId::to_uint64(TaskId::make_sub_task(0, 0)));
    }
    void *base() { return storage.scheduler_state->base(); }
    SchedulerRunControl *run() { return storage.run_control; }
    DispatchPayload *payload(uint32_t worker) {
        return scheduler_state_at<DispatchPayload>(
            base(), locals[worker]->dispatch_payload_offset(locals[worker]->config.self_lane, 0)
        );
    }
    void assign(uint32_t worker, uint32_t block = 0, uint32_t lane = 1) {
        auto *local = locals[worker];
        auto &slot = local->slots[0];
        const auto &meta = storage.metadata[execution_ref];
        slot.state = SchedulerDispatchSlotState::READY;
        slot.generation = 7;
        slot.task_ref = execution_ref;
        slot.block_idx = block;
        slot.subtask_slot = lane;
        slot.cooperative = scheduler_task_requires_admission(meta);
        slot.gm_mix = scheduler_task_is_mix(meta.flags);
        local->local_ready_mask = 1;
        auto *p = payload(worker);
        *p = {};
        p->local_context.block_idx = block;
        p->local_context.block_num = meta.logical_block_num;
        p->global_context.sub_block_id = slot.gm_mix ? (local->config.self_lane == 2 ? 1 : 0) : (lane == 2 ? 1 : 0);
        p->args[SPMD_LOCAL_CONTEXT_INDEX] = reinterpret_cast<uint64_t>(&p->local_context);
        p->args[SPMD_GLOBAL_CONTEXT_INDEX] = reinterpret_cast<uint64_t>(&p->global_context);
        if (slot.cooperative) {
            auto &admission = admissions[worker];
            EXPECT_TRUE(scheduler_initialize_admission(base(), local, &admission));
            local->admission = &admission;
            admission.current_task_ref = execution_ref;
            admission.current_block_num = meta.logical_block_num;
            admission.current_flags = meta.flags;
            admission.current_active_mask = meta.active_mask;
            admission.sync_claimed =
                scheduler_task_requires_sync_start(meta.flags) && (!slot.gm_mix || local->is_mix_coordinator());
            admission.gm_mix.phase = SchedulerGmMixPhase::RUNNING;
            admission.gm_mix.slot_index = 0;
            admission.gm_mix.slot_generation = slot.generation;
            admission.gm_mix.block_idx = block;
            admission.gm_mix.subtask_slot = lane;
        }
    }
    bool bind(uint32_t worker) {
        return scheduler_async_execution_bind(&states[worker], view(), base(), locals[worker], run(), 0);
    }
    bool admit(uint32_t worker, bool *permit, SchedulerSubmissionTransactionStats *stats = nullptr) {
        return scheduler_submission_execution_admit(
            submissions[worker], &states[worker], view(), base(), locals[worker], run(), 0, permit, stats
        );
    }
    bool enter(uint32_t worker) {
        return scheduler_async_execution_permitted(&states[worker], view(), locals[worker], run(), 0);
    }
    bool returned(uint32_t worker) {
        if (!scheduler_submission_execution_returned(
                submissions[worker], &states[worker], view(), base(), locals[worker], run(), 0, backend
            ))
            return false;
        locals[worker]->local_completed_generations[0] = locals[worker]->slots[0].generation;
        return true;
    }
    AsyncSubmissionContext *context(uint32_t worker) {
        return reinterpret_cast<AsyncSubmissionContext *>(payload(worker)->async_submission_context);
    }
    uint64_t busy() {
        auto &view = submissions[0].view;
        return scheduler_gm_fetch_add(
            scheduler_submission_detail::at<SchedulerSubmissionPool>(view, view.header.pools_offset)->busy_groups,
            UINT64_C(0)
        );
    }
    void require_enter(uint32_t worker) {
        ASSERT_TRUE(bind(worker));
        bool permit = false;
        for (uint32_t attempt = 0; attempt < 64 && !permit; ++attempt)
            ASSERT_TRUE(admit(worker, &permit));
        ASSERT_TRUE(permit);
        ASSERT_TRUE(enter(worker));
    }
    int64_t execution_ref{0};
    scheduler_test::FixtureStorage storage;
    scheduler_test::BasicGraphBuffer<4> graph{4};
    SchedulerGraphView graph_view{};
    std::unique_ptr<void, decltype(&std::free)> native_image{nullptr, &std::free};
    std::vector<SchedulerAsyncCapabilities> caps;
    std::vector<SchedulerSubmissionExecutionUseSet> use_sets;
    SchedulerSubmissionHostPlan plan;
    SchedulerAsyncHostPlan async_plan;
    SchedulerAsyncRuntimeView async{};
    std::unique_ptr<void, decltype(&std::free)> image;
    alignas(128) std::array<uint8_t, 512> workspace_storage{};
    std::array<SchedulerLocalState *, 6> locals{};
    std::array<SchedulerAdmissionLocalState, 6> admissions{};
    std::array<SchedulerAsyncExecutionState<Backend>, 6> states{};
    std::array<SchedulerSubmissionExecutionView, 6> submissions{};
    Backend backend;
    uint64_t predicate{0};
};

TEST(HbgSubmissionExecution, AuxiliaryMountBindsCanonicalFrameAndZeroAttemptReturnReleasesBeforeCompletion) {
    Fixture f;
    f.assign(1);
    f.require_enter(1);
    auto *context = f.context(1);
    ASSERT_NE(context, nullptr);
    const auto &view = f.submissions[1].view;
    EXPECT_EQ(context, scheduler_submission_detail::at<AsyncSubmissionContext>(view, view.header.contexts_offset, 0));
    EXPECT_EQ(context->header.frame_address, reinterpret_cast<uint64_t>(f.payload(1)));
    EXPECT_EQ(context->header.dispatch_generation, 7u);
    EXPECT_EQ(get_async_submission_context(reinterpret_cast<int64_t *>(f.payload(1)->args)), context);
    EXPECT_EQ(f.busy(), 1u);
    ASSERT_TRUE(f.returned(1));
    EXPECT_EQ(context->evidence.flags, ASYNC_SUBMISSION_FLAGS);
    EXPECT_EQ(f.busy(), 0u);
    EXPECT_EQ(f.locals[1]->local_completed_generations[0], 7u);
    EXPECT_FALSE(f.returned(1));
    EXPECT_EQ(f.busy(), 0u);
}

TEST(HbgSubmissionExecution, FailedOrUncommittedSdkEvidenceKeepsGrantAndPreventsCompletion) {
    for (uint32_t scenario = 0; scenario < 3; ++scenario) {
        Fixture f;
        f.assign(1);
        f.require_enter(1);
        auto *context = f.context(1);
        if (scenario == 0) ASSERT_TRUE(async_submission_begin(*context));
        if (scenario == 1) async_submission_error(*context, SIMPLER_ERROR_ASYNC_REGISTRATION_FAILED);
        if (scenario == 2) context->evidence.dispatch_generation++;
        EXPECT_FALSE(f.returned(1));
        EXPECT_EQ(f.busy(), 1u);
        EXPECT_EQ(f.locals[1]->local_completed_generations[0], 0u);
        EXPECT_EQ(context->evidence.flags & ASYNC_SUBMISSION_SEALED, 0u);
    }
}

TEST(HbgSubmissionExecution, ForeignContextPointerFailsBeforeAccessingThePointedMemory) {
    Fixture f;
    f.assign(1);
    f.require_enter(1);
    f.payload(1)->async_submission_context = 1;
    EXPECT_FALSE(f.returned(1));
    EXPECT_EQ(f.busy(), 1u);
    EXPECT_EQ(f.locals[1]->local_completed_generations[0], 0u);
}

TEST(HbgSubmissionExecution, OriginalOrdinaryAdmissionDenialAbortsBeforeReturningToService) {
    Fixture f;
    f.assign(1);
    ASSERT_TRUE(f.bind(1));
    auto *local = f.locals[1];
    auto &admission = f.admissions[1];
    ASSERT_TRUE(scheduler_initialize_admission(f.base(), local, &admission));
    local->admission = &admission;
    local->slots[0].candidate_index = 0;
    admission.candidates[0].ready.task_ref = 0;
    admission.candidates[0].slot_index = 0;
    auto *control = scheduler_state_at<SchedulerAdmissionControl>(f.base(), admission.control_offset);
    scheduler_gm_store(control->commit_tail, UINT64_C(1));
    bool permit = true;
    SchedulerSubmissionTransactionStats stats{};
    ASSERT_TRUE(f.admit(1, &permit, &stats));
    EXPECT_FALSE(permit);
    EXPECT_EQ(stats.pool_cas, 1u);
    EXPECT_EQ(stats.pool_returns, 1u);
    EXPECT_EQ(f.busy(), 0u);
}

TEST(HbgSubmissionExecution, MixReservesWholeBundleUnderExistingAdmissionAndUsesPhysicalSubblock) {
    Fixture f(7);
    // The logical AIV slots can occupy either physical AIV within the cluster.
    f.assign(0, 0, 0);
    f.assign(1, 0, 2);
    f.assign(2, 0, 1);
    ASSERT_TRUE(f.bind(0));
    ASSERT_TRUE(f.bind(1));
    EXPECT_EQ(f.busy(), 0u);
    ASSERT_TRUE(f.bind(2));
    ASSERT_TRUE(f.bind(0));
    ASSERT_TRUE(f.bind(1));
    bool permit = false;
    for (uint32_t round = 0; round < 64 && !permit; ++round)
        ASSERT_TRUE(f.admit(0, &permit));
    ASSERT_TRUE(permit);
    ASSERT_TRUE(f.enter(0));
    ASSERT_TRUE(f.admit(1, &permit));
    ASSERT_TRUE(permit);
    ASSERT_TRUE(f.enter(1));
    EXPECT_EQ(f.context(1)->header.subtask_slot, 2u);
    EXPECT_EQ(f.context(1)->header.frame_sub_block_id, 0u);
    ASSERT_TRUE(f.admit(2, &permit));
    ASSERT_TRUE(permit);
    ASSERT_TRUE(f.enter(2));
    EXPECT_EQ(f.busy(), 3u);
    ASSERT_TRUE(f.returned(0));
    EXPECT_EQ(f.busy(), 3u);
    ASSERT_TRUE(f.returned(1));
    ASSERT_TRUE(f.returned(2));
    EXPECT_EQ(f.busy(), 0u);
}

TEST(HbgSubmissionExecution, SynchronousSubmitLaneInAsyncRunHasNoDeferredTokenAndStillReleases) {
    Fixture f(2, 1, false, 0);
    f.assign(1);
    f.require_enter(1);
    auto *context = f.context(1);
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(context->header.completion_mode, CoreAsyncMode::SYNCHRONOUS);
    EXPECT_EQ(context->header.use_begin, 0u);
    EXPECT_EQ(context->header.use_count, 1u);
    EXPECT_FALSE(f.payload(1)->local_context.async_ctx.task_token.is_valid());
    EXPECT_EQ(context->header.provenance_capacity, 0u);
    ASSERT_TRUE(f.returned(1));
    EXPECT_EQ(f.busy(), 0u);
}

TEST(HbgSubmissionExecution, SyncStartReservesBothBlocksTogetherBeforeEitherKernelEnters) {
    Fixture f(2, 2, true);
    f.assign(1, 0, 1);
    f.assign(4, 1, 1);
    ASSERT_TRUE(f.bind(1));
    EXPECT_EQ(f.busy(), 0u);
    ASSERT_TRUE(f.bind(4));
    ASSERT_TRUE(f.bind(1));
    bool permit = false;
    for (uint32_t round = 0; round < 64 && !permit; ++round)
        ASSERT_TRUE(f.admit(1, &permit));
    ASSERT_TRUE(permit);
    EXPECT_EQ(f.busy(), 3u);
    ASSERT_TRUE(f.enter(1));
    ASSERT_TRUE(f.admit(4, &permit));
    ASSERT_TRUE(permit);
    ASSERT_TRUE(f.enter(4));
    EXPECT_NE(f.context(1), f.context(4));
    EXPECT_EQ(f.context(4)->header.block_idx, 1u);
    ASSERT_TRUE(f.returned(1));
    EXPECT_EQ(f.busy(), 2u);
    ASSERT_TRUE(f.returned(4));
    EXPECT_EQ(f.busy(), 0u);
}

TEST(HbgSubmissionExecution, PureSpmdUsesPerExecutionSubmissionWhileKeepingOrderedCpuAdmission) {
    Fixture f(2, 2);
    f.assign(1, 0, 1);
    f.assign(4, 1, 1);
    ASSERT_TRUE(f.locals[1]->slots[0].cooperative);
    f.require_enter(1);
    EXPECT_EQ(f.busy(), 1u);
    f.require_enter(4);
    EXPECT_EQ(f.busy(), 3u);
    ASSERT_TRUE(f.returned(1));
    EXPECT_EQ(f.busy(), 2u);
    ASSERT_TRUE(f.returned(4));
    EXPECT_EQ(f.busy(), 0u);
}

TEST(HbgSubmissionExecution, MixSyncVectorLaneCanAcquireTheBundleWithoutCoordinatorClaimState) {
    Fixture f(7, 2, true);
    f.assign(1, 0, 1);
    ASSERT_FALSE(f.admissions[1].sync_claimed);
    f.require_enter(1);
    EXPECT_EQ(__builtin_popcountll(f.busy()), 4);
    EXPECT_EQ(f.context(1)->header.block_idx, 0u);
    EXPECT_EQ(f.context(1)->header.frame_sub_block_id, 0u);
    ASSERT_TRUE(f.returned(1));
    EXPECT_EQ(__builtin_popcountll(f.busy()), 3);
}

TEST(HbgSubmissionExecution, PredicateSkippedKernelUsesTheSameZeroAttemptSealAndReleaseTail) {
    Fixture f(2, 1, false, 7, true);
    f.assign(1);
    auto *local = f.locals[1];
    local->slots[0].state = SchedulerDispatchSlotState::FILLING;
    SchedulerReadyClaim ready{};
    ready.task_ref = 0;
    const SchedulerFreeSlotClaim claim{local->worker_id(), 0, local->slots[0].generation, local->config.self_lane};
    ASSERT_TRUE(scheduler_fill_dispatch_slot(f.view(), f.base(), local, f.run(), claim, ready, 0, nullptr));
    ASSERT_EQ(local->slots[0].predicate, SchedulerPredicateState::FAIL);
    f.require_enter(1);
    EXPECT_EQ(f.payload(1)->function_bin_addr, 0u);
    ASSERT_TRUE(f.returned(1));
    EXPECT_EQ(f.busy(), 0u);
    EXPECT_EQ(f.context(1)->evidence.attempted, 0u);
}

TEST(HbgSubmissionExecution, SdmaTokenRequiresMatchingPublicRecordProvenanceAndActiveGrantAtSeal) {
    for (uint32_t scenario = 0; scenario < 4; ++scenario) {
        Fixture f;
        f.assign(1);
        f.require_enter(1);
        auto &ctx = f.payload(1)->local_context.async_ctx;
        *ctx.completion_count = 1;
        auto &entry = ctx.completion_entries[0];
        entry.addr = f.backend.record_address;
        entry.backend_cookie = 0xfedcba9876543210ULL;
        entry.expected_value = 0;
        entry.engine = COMPLETION_ENGINE_SDMA;
        entry.completion_type = COMPLETION_TYPE_SDMA_EVENT_RECORD;
        entry._pad = 0;
        auto *context = f.context(1);
        ASSERT_TRUE(async_submission_begin(*context));
        Publisher publisher;
        ASSERT_TRUE(async_submission_commit(*context, publisher));
        auto *record =
            reinterpret_cast<SchedulerSubmissionProvenance *>(f.async.base + context->header.provenance_offset);
        *record = {1, COMPLETION_TYPE_SDMA_EVENT_RECORD, f.backend.record_address, 0xfedcba9876543210ULL, 0, 0, 0, 0,
                   0};
        if (scenario == 1) record->version = 0;
        if (scenario == 2) ++record->backend_cookie;
        if (scenario == 3) ++record->channel_group;
        EXPECT_EQ(f.returned(1), scenario == 0);
        EXPECT_EQ(f.busy(), scenario == 0 ? 0u : 1u);
        EXPECT_EQ(f.locals[1]->local_completed_generations[0], scenario == 0 ? 7u : 0u);
    }
}

TEST(HbgSubmissionExecution, PlannerDefaultsUnknownCapabilityAndBudgetsContextAndProvenanceRows) {
    Fixture f;
    const std::vector<SchedulerTaskMetadata> metadata(f.storage.metadata, f.storage.metadata + 4);
    const SchedulerSubmissionWorkspaceInput workspace{
        reinterpret_cast<uint64_t>(f.workspace_storage.data()), f.workspace_storage.size(), kEpoch, 1, 48,
        SchedulerSubmissionPoolQualification::EXCLUSIVE_RUN
    };
    SchedulerSubmissionHostPlan result;
    auto caps = f.caps;
    caps[0][1] = {};
    ASSERT_TRUE(scheduler_plan_submission_bundles(
        metadata, caps, {workspace}, {{0, 0, 1, 0, 0}}, f.use_sets, kEpoch, kBudget, &result, nullptr
    ));
    EXPECT_EQ(result.header.provenance_count, 64u);
    SchedulerSubmissionExecution execution{};
    std::memcpy(&execution, result.image.data() + result.header.executions_offset, sizeof(execution));
    EXPECT_EQ(execution.completion_mode, CoreAsyncMode::DEFERRED);
    EXPECT_EQ(execution.condition_capacity, 64u);
    EXPECT_EQ(execution.condition_types, CORE_ASYNC_CONDITION_TYPES);
    result = {};
    caps[0][1] = {1, CoreAsyncMode::UNKNOWN, 1, 0};
    EXPECT_FALSE(scheduler_plan_submission_bundles(
        metadata, caps, {workspace}, {{0, 0, 1, 0, 0}}, f.use_sets, kEpoch, kBudget, &result, nullptr
    ));
    EXPECT_TRUE(result.image.empty());
    EXPECT_FALSE(scheduler_plan_submission_bundles(
        metadata, f.caps, {workspace}, {{0, 0, 1, 0, 0}}, f.use_sets, kEpoch, f.plan.header.total_bytes - 1, &result,
        nullptr
    ));
    EXPECT_TRUE(result.image.empty());
    EXPECT_EQ(f.plan.header.provenance_count, 2u);
    auto bad = f.plan.header;
    bad.contexts_offset = bad.pools_offset;
    auto *base = f.submissions[0].view.base;
    std::memcpy(base, &bad, sizeof(bad));
    SchedulerSubmissionExecutionView view;
    EXPECT_FALSE(scheduler_submission_execution_mount(f.states[0], &view));
}

TEST(HbgSubmissionExecution, ImmutableUseMapMustAgreeWithTheDeferredGroupBeforeAdmission) {
    Fixture f;
    f.assign(1);
    ASSERT_TRUE(f.bind(1));
    const auto &view = f.submissions[1].view;
    *scheduler_submission_detail::at<SchedulerSubmissionExecutionUseMap>(
        view, view.header.execution_use_maps_offset
    ) = {};
    bool permit = true;
    EXPECT_FALSE(f.admit(1, &permit));
    EXPECT_FALSE(permit);
    EXPECT_EQ(f.busy(), 0u);
    EXPECT_EQ(f.payload(1)->async_submission_context, 0u);
}

TEST(HbgSubmissionExecution, DynamicExecutionCanShareARunWithQualifiedSubmissions) {
    for (uint8_t type : {COMPLETION_TYPE_COUNTER, COMPLETION_TYPE_SDMA_EVENT_RECORD}) {
        Fixture f;
        f.execution_ref = 3;
        const auto &view = f.submissions[0].view;
        uint64_t execution = SCHEDULER_ASYNC_INDEX_NONE;
        ASSERT_TRUE(scheduler_submission_execution_index(view, 3, 0, 0, &execution));
        *scheduler_submission_detail::at<SchedulerSubmissionExecutionUseMap>(
            view, view.header.execution_use_maps_offset, execution
        ) = {};
        SchedulerAsyncTaskMap map{};
        ASSERT_TRUE(scheduler_async_runtime_detail::task_map(f.async, 3, &map));
        auto *group = scheduler_async_runtime_detail::at<SchedulerAsyncGroupIdentity>(
            f.async, f.async.layout.group_identity_offset, scheduler_async_group_index(map, 0, 0)
        );
        group->binding_uses_offset = SCHEDULER_ASYNC_USES_DYNAMIC;
        group->binding_use_count = 0;
        f.assign(0, 0, 0);
        f.require_enter(0);
        ASSERT_EQ(f.context(0), nullptr);
        auto &ctx = f.payload(0)->local_context.async_ctx;
        ASSERT_NE(ctx.completion_count, nullptr);
        auto &entry = ctx.completion_entries[0];
        entry.addr = f.backend.record_address;
        entry.backend_cookie = type == COMPLETION_TYPE_COUNTER ? 0 : 1;
        entry.expected_value = type == COMPLETION_TYPE_COUNTER ? 1 : 0;
        entry.engine = COMPLETION_ENGINE_SDMA;
        entry.completion_type = type;
        *ctx.completion_count = 1;
        ASSERT_TRUE(f.returned(0));
        EXPECT_EQ(f.locals[0]->local_completed_generations[0], 7u);
        EXPECT_EQ(f.busy(), 0u);
    }
}

TEST(HbgSubmissionExecution, DynamicAivGuardCarrierSurvivesAdmissionAlongsideAQualifiedSibling) {
    Fixture f(6, 1, false, 7, false, false, true, 2);
    f.assign(1, 0, 1);
    f.require_enter(1);
    const auto address = reinterpret_cast<uint64_t>(f.async.base) + f.async.layout.sdma_guard_context_offset;
    ASSERT_EQ(f.payload(1)->async_submission_context, address);
    EXPECT_EQ(get_async_submission_context(reinterpret_cast<int64_t *>(f.payload(1)->args)), nullptr);
    ASSERT_TRUE(f.returned(1));
    f.assign(2, 0, 2);
    f.require_enter(2);
    EXPECT_NE(f.payload(2)->async_submission_context, address);
    EXPECT_NE(get_async_submission_context(reinterpret_cast<int64_t *>(f.payload(2)->args)), nullptr);
    EXPECT_TRUE(f.returned(2));
}

TEST(HbgSubmissionExecution, NativeChildFrameKeepsItsSubTaskIdentityThroughSealAndRelease) {
    Fixture f(2, 1, false, 7, false, true);
    f.assign(1);
    f.require_enter(1);
    auto *context = f.context(1);
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(context->evidence.task_ref, 1);
    EXPECT_EQ(context->header.task_token, TaskId::to_uint64(TaskId::make_sub_task(0, 0)));
    EXPECT_NE(context->header.task_token, TaskId::to_uint64(TaskId::make_global(1)));
    EXPECT_EQ(get_async_submission_context(reinterpret_cast<int64_t *>(f.payload(1)->args)), context);
    ASSERT_TRUE(f.returned(1));
    EXPECT_EQ(f.busy(), 0u);
    EXPECT_EQ(f.locals[1]->local_completed_generations[0], 7u);
}
struct CounterOnlyImage {
    SchedulerAsyncHostPlan plan;
    std::unique_ptr<void, decltype(&std::free)> image{nullptr, &std::free};
    SchedulerAsyncExecutionState<Backend> state{};

    CounterOnlyImage() {
        SchedulerTaskMetadata task{};
        task.flags = SCHEDULER_TASK_EXECUTABLE;
        task.active_mask = LaneMask::aiv0();
        task.logical_block_num = 1;
        SchedulerAsyncCapabilities caps{};
        caps[1] = {CORE_ASYNC_CAPABILITY_VERSION, CoreAsyncMode::DEFERRED, 1, CORE_ASYNC_CONDITION_COUNTER};
        std::string error;
        if (!scheduler_plan_async_state({task}, {caps}, kEpoch, kBudget, {}, &plan, &error))
            throw std::runtime_error(error);
        image.reset(std::aligned_alloc(128, plan.image.size()));
        if (!image) throw std::bad_alloc();
        std::memcpy(image.get(), plan.image.data(), plan.image.size());
        state.view = {static_cast<uint8_t *>(image.get()), plan.layout};
    }
};

TEST(HbgSubmissionExecution, CounterOnlyHostPlanMountsWithoutSubmissionScratch) {
    CounterOnlyImage f;
    ASSERT_EQ(f.plan.layout.scratch_bytes, 0u);
    ASSERT_EQ(f.plan.layout.scratch_offset, f.plan.layout.total_bytes);
    ASSERT_NE(f.plan.layout.scratch_offset, 0u);
    SchedulerSubmissionExecutionView execution;
    SchedulerSubmissionProvenanceView provenance;
    ASSERT_TRUE(scheduler_submission_execution_mount(f.state, &execution));
    ASSERT_TRUE(scheduler_submission_provenance_mount(f.state.view.base, f.state.view.layout, &provenance));
    EXPECT_FALSE(execution.view.enabled());
    EXPECT_FALSE(provenance.submissions.enabled());
    EXPECT_EQ(execution.async_offset, 0u);
    EXPECT_EQ(provenance.async_offset, 0u);
}

TEST(HbgSubmissionExecution, EmptySubmissionScratchStillRejectsInvalidOffsetsAndExtents) {
    CounterOnlyImage f;
    const uint64_t invalid_offsets[]{
        1, 128, f.plan.layout.total_bytes - 1, f.plan.layout.total_bytes + 128, UINT64_MAX
    };
    for (const uint64_t offset : invalid_offsets) {
        auto state = f.state;
        state.view.layout.scratch_offset = offset;
        SchedulerSubmissionExecutionView execution;
        SchedulerSubmissionProvenanceView provenance;
        EXPECT_FALSE(scheduler_submission_execution_mount(state, &execution)) << offset;
        EXPECT_FALSE(scheduler_submission_provenance_mount(state.view.base, state.view.layout, &provenance)) << offset;
        EXPECT_FALSE(execution.view.enabled());
        EXPECT_FALSE(provenance.submissions.enabled());
    }
    auto state = f.state;
    state.view.layout.scratch_bytes = 128;
    SchedulerSubmissionExecutionView execution;
    SchedulerSubmissionProvenanceView provenance;
    EXPECT_FALSE(scheduler_submission_execution_mount(state, &execution));
    EXPECT_FALSE(scheduler_submission_provenance_mount(state.view.base, state.view.layout, &provenance));
}

}  // namespace
