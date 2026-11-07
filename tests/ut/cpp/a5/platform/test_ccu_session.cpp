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
#include <cstring>

#include "support/ccu_session_fake.h"
#include "platform_comm/ccu/ccu_device.h"

namespace {
using Control = ccu_test::SessionControl;
CcuSessionConfig Config(const std::shared_ptr<void> &parent) {
    CcuSessionConfig c;
    c.allocation_id = 11;
    c.epoch = 29;
    c.device_id = 0;
    c.parent_owner = parent;
    c.exchange = [](uint64_t, uint32_t, const void *, uint32_t, void *, uint32_t, uint32_t) {
        return 0;
    };
    return c;
}
class CcuSessionTest : public testing::Test {
protected:
    std::shared_ptr<Control> state = std::make_shared<Control>();
    std::shared_ptr<void> parent = std::make_shared<int>(9);
    void SetUp() override { ccu_test::QueueBackend(state); }
    void TearDown() override { ccu_test::ClearBackends(); }
};
}  // namespace

TEST_F(CcuSessionTest, ConstructorHasNoSdkSideEffectsAndPartialInitializeCanAbortLocally) {
    CcuSession session(Config(parent));
    EXPECT_EQ(state->initialize_calls, 0);
    state->fail_initialize = true;
    EXPECT_EQ(session.initialize(), CcuStatus::INIT_FAILED);
    EXPECT_EQ(session.native_error(), 71);
    EXPECT_EQ(session.initialize(), CcuStatus::BUSY);
    EXPECT_EQ(session.close(CcuCloseMode::ABORT_UNPUBLISHED), CcuStatus::OK);
    EXPECT_EQ(state->close_calls, 1);
}
TEST_F(CcuSessionTest, MemoryPreparationIsIdempotentAndNeverReusesAllocationIdentity) {
    CcuSession session(Config(parent));
    ASSERT_EQ(session.initialize(), CcuStatus::OK);
    state->unsupported_memory = true;
    EXPECT_EQ(session.prepare_memory(1, 0x10000, 4096), CcuStatus::UNSUPPORTED);
    state->unsupported_memory = false;
    ASSERT_EQ(session.prepare_memory(1, 0x10000, 4096), CcuStatus::OK);
    EXPECT_EQ(session.prepare_memory(1, 0x10000, 4096), CcuStatus::OK);
    EXPECT_EQ(state->prepare_calls, 2);
    EXPECT_EQ(session.prepare_memory(1, 0x20000, 4096), CcuStatus::INVALID_ARGUMENT);
    EXPECT_EQ(session.prepare_memory(2, 0x10800, 4096), CcuStatus::INVALID_ARGUMENT);
    EXPECT_EQ(session.release_memory(1), CcuStatus::OK);
    EXPECT_EQ(session.prepare_memory(1, 0x10000, 4096), CcuStatus::INVALID_ARGUMENT);
    EXPECT_EQ(session.close(CcuCloseMode::NORMAL), CcuStatus::OK);
}
TEST_F(CcuSessionTest, PublishedLayoutResolvesByKeyAndRunLeaseFreezesHostEdits) {
    CcuSession session(Config(parent));
    ASSERT_EQ(session.initialize(), CcuStatus::OK);
    CcuProgramRef host_ref{};
    auto spec = ccu_builtin_broadcast(0);
    ASSERT_EQ(session.register_function(1, spec, host_ref), CcuStatus::OK);
    EXPECT_EQ(host_ref.context_addr, 0U);
    ASSERT_EQ(session.prepare_memory(1, 0x10000, 4096), CcuStatus::OK);
    CcuDeviceContextView view{};
    ASSERT_EQ(session.get_device_context(view), CcuStatus::OK);
    CcuDeviceContext *ctx{};
    ASSERT_EQ(ccu_get_device_context(view.addr, view.size, ctx), CcuStatus::OK);
    EXPECT_EQ(ctx->range_count, 1U);
    EXPECT_EQ(ctx->arguments_size, 56U);
    EXPECT_EQ(ctx->timeout_ns, 5000000000U);
    CcuProgramRef device_ref{};
    EXPECT_EQ(ccu_resolve_program(ctx, spec.program_key, device_ref), CcuStatus::OK);
    ASSERT_EQ(session.retain_run(), CcuStatus::OK);
    EXPECT_EQ(session.retain_run(), CcuStatus::BUSY);
    EXPECT_EQ(session.release_memory(1), CcuStatus::BUSY);
    EXPECT_EQ(session.register_function(2, spec, host_ref), CcuStatus::BUSY);
    EXPECT_EQ(session.quiesce(1), CcuStatus::BUSY);
    EXPECT_EQ(ccu_resolve_program(ctx, spec.program_key, device_ref), CcuStatus::STALE_HANDLE);
    EXPECT_EQ(session.release_run(), CcuStatus::OK);
    EXPECT_EQ(session.retain_run(), CcuStatus::STALE_HANDLE);
    EXPECT_EQ(session.quiesce(1), CcuStatus::OK);
    EXPECT_EQ(session.close(CcuCloseMode::NORMAL), CcuStatus::OK);
}
TEST_F(CcuSessionTest, AbortingReusedRegistrationDoesNotDeleteCommittedFunction) {
    CcuSession session(Config(parent));
    ASSERT_EQ(session.initialize(), CcuStatus::OK);
    CcuProgramRef ref{};
    const auto spec = ccu_builtin_broadcast(0);
    ASSERT_EQ(session.register_function(1, spec, ref), CcuStatus::OK);
    CcuDeviceContextView view{};
    ASSERT_EQ(session.get_device_context(view), CcuStatus::OK);
    ASSERT_EQ(session.register_function(2, spec, ref), CcuStatus::OK);
    EXPECT_EQ(session.abort_registration(2), CcuStatus::OK);
    EXPECT_EQ(session.abort_registration(1), CcuStatus::OK);
    EXPECT_EQ(session.get_device_context(view), CcuStatus::OK);
    EXPECT_EQ(state->publish_calls, 1);
    EXPECT_EQ(reinterpret_cast<CcuDeviceContext *>(view.addr)->binding_count, 1U);
    EXPECT_EQ(session.close(CcuCloseMode::NORMAL), CcuStatus::OK);
}
TEST_F(CcuSessionTest, RejectedSignatureCannotPublishOrReuseFailedTransaction) {
    CcuSession session(Config(parent));
    ASSERT_EQ(session.initialize(), CcuStatus::OK);
    auto spec = ccu_builtin_broadcast(0);
    spec.entry = "unsupported";
    ASSERT_EQ(ccu_compute_program_key(spec, spec.program_key), CcuStatus::OK);
    CcuProgramRef ref{};
    EXPECT_EQ(session.register_function(1, spec, ref), CcuStatus::UNSUPPORTED);
    EXPECT_EQ(session.register_function(1, spec, ref), CcuStatus::INVALID_ARGUMENT);
    CcuDeviceContextView view{};
    EXPECT_EQ(session.get_device_context(view), CcuStatus::INVALID_ARGUMENT);
    EXPECT_EQ(session.abort_registration(1), CcuStatus::OK);
    spec = ccu_builtin_broadcast(0);
    spec.signature.outputs[0].name = "src";
    ASSERT_EQ(ccu_compute_program_key(spec, spec.program_key), CcuStatus::OK);
    EXPECT_EQ(session.register_function(2, spec, ref), CcuStatus::INVALID_ARGUMENT);
    EXPECT_EQ(session.close(CcuCloseMode::ABORT_UNPUBLISHED), CcuStatus::OK);
}
TEST_F(CcuSessionTest, PublishFailurePoisonsSessionAndCloseRetryRetainsParent) {
    std::weak_ptr<void> weak = parent;
    CcuSession session(Config(parent));
    parent.reset();
    ASSERT_EQ(session.initialize(), CcuStatus::OK);
    CcuProgramRef ref{};
    ASSERT_EQ(session.register_function(1, ccu_builtin_broadcast(0), ref), CcuStatus::OK);
    state->fail_publish = true;
    CcuDeviceContextView view{};
    EXPECT_EQ(session.get_device_context(view), CcuStatus::INIT_FAILED);
    EXPECT_EQ(view.addr, 0U);
    EXPECT_EQ(session.get_device_context(view), CcuStatus::STALE_HANDLE);
    EXPECT_EQ(session.abort_registration(1), CcuStatus::RECLAMATION_UNCONFIRMED);
    state->stop = false;
    EXPECT_EQ(session.close(CcuCloseMode::ABORT_UNPUBLISHED), CcuStatus::RECLAMATION_UNCONFIRMED);
    EXPECT_FALSE(weak.expired());
    EXPECT_EQ(state->close_calls, 0);
    state->stop = true;
    state->fail_close = true;
    EXPECT_EQ(session.close(CcuCloseMode::ABORT_UNPUBLISHED), CcuStatus::RECLAMATION_UNCONFIRMED);
    EXPECT_EQ(session.native_error(), 74);
    EXPECT_FALSE(weak.expired());
    state->fail_close = false;
    EXPECT_EQ(session.close(CcuCloseMode::ABORT_UNPUBLISHED), CcuStatus::OK);
    EXPECT_TRUE(weak.expired());
    EXPECT_EQ(session.close(CcuCloseMode::NORMAL), CcuStatus::OK);
}
TEST_F(CcuSessionTest, ProgramKeyIncludesEntireDefinitionAndExcludesDisplayName) {
    auto a = ccu_builtin_broadcast(0), b = a;
    b.name = "another display name";
    CcuProgramKey key{};
    ASSERT_EQ(ccu_compute_program_key(b, key), CcuStatus::OK);
    EXPECT_EQ(std::memcmp(key.bytes, a.program_key.bytes, 32), 0);
    b.signature.scalars[0].type = CcuArgumentType::INT64;
    ASSERT_EQ(ccu_compute_program_key(b, key), CcuStatus::OK);
    EXPECT_NE(std::memcmp(key.bytes, a.program_key.bytes, 32), 0);
    b = ccu_builtin_broadcast(1);
    EXPECT_NE(std::memcmp(b.program_key.bytes, a.program_key.bytes, 32), 0);
}

TEST_F(CcuSessionTest, UserBuildLibraryAbiAndEntryAreValidatedAndAbortIsLocal) {
    CcuSession session(Config(parent));
    ASSERT_EQ(session.initialize(), CcuStatus::OK);
    auto spec = ccu_builtin_broadcast(0);
    spec.library_path = CCU_TEST_BAD_ABI_MODULE;
    spec.entry = "build_root_copy";
    ASSERT_EQ(ccu_compute_program_key(spec, spec.program_key), CcuStatus::OK);
    CcuProgramRef ref{};
    EXPECT_EQ(session.register_function(1, spec, ref), CcuStatus::ABI_MISMATCH);
    EXPECT_EQ(session.abort_registration(1), CcuStatus::OK);
    spec.library_path = CCU_TEST_USER_MODULE;
    ASSERT_EQ(ccu_compute_program_key(spec, spec.program_key), CcuStatus::OK);
    EXPECT_EQ(session.register_function(2, spec, ref), CcuStatus::OK);
    EXPECT_EQ(session.register_function(3, spec, ref), CcuStatus::OK);
    EXPECT_EQ(session.abort_registration(2), CcuStatus::OK);
    CcuDeviceContextView view{};
    EXPECT_EQ(session.get_device_context(view), CcuStatus::OK);
    EXPECT_EQ(state->publish_calls, 1);
    EXPECT_EQ(session.abort_registration(3), CcuStatus::OK);
    EXPECT_EQ(session.close(CcuCloseMode::NORMAL), CcuStatus::OK);
}

TEST_F(CcuSessionTest, MissingUserEntryIsTrackedUntilAbortAndAnotherSessionIsIndependent) {
    CcuSession first(Config(parent));
    ASSERT_EQ(first.initialize(), CcuStatus::OK);
    auto spec = ccu_builtin_broadcast(0);
    spec.library_path = CCU_TEST_USER_MODULE;
    spec.entry = "missing_entry";
    ASSERT_EQ(ccu_compute_program_key(spec, spec.program_key), CcuStatus::OK);
    CcuProgramRef ref{};
    EXPECT_EQ(first.register_function(1, spec, ref), CcuStatus::INVALID_ARGUMENT);
    EXPECT_EQ(first.abort_registration(1), CcuStatus::OK);
    auto second_state = std::make_shared<Control>();
    ccu_test::QueueBackend(second_state);
    auto config = Config(parent);
    config.epoch = 99;
    CcuSession second(std::move(config));
    ASSERT_EQ(second.initialize(), CcuStatus::OK);
    EXPECT_EQ(second_state->initialize_calls, 1);
    EXPECT_EQ(state->initialize_calls, 1);
    EXPECT_EQ(second.close(CcuCloseMode::ABORT_UNPUBLISHED), CcuStatus::OK);
    EXPECT_EQ(state->close_calls, 0);
    EXPECT_EQ(first.close(CcuCloseMode::ABORT_UNPUBLISHED), CcuStatus::OK);
}

TEST_F(CcuSessionTest, ControllableBackendKeepsDelayedRequestAndPreservesSequenceAcrossRunPublication) {
    CcuSession session(Config(parent));
    ASSERT_EQ(session.initialize(), CcuStatus::OK);
    auto spec = ccu_builtin_broadcast(0);
    CcuProgramRef host_ref{}, program{};
    ASSERT_EQ(session.register_function(1, spec, host_ref), CcuStatus::OK);
    ASSERT_EQ(session.prepare_memory(1, 0x10000, 4096), CcuStatus::OK);
    CcuDeviceContextView view{};
    ASSERT_EQ(session.get_device_context(view), CcuStatus::OK);
    auto *ctx = reinterpret_cast<CcuDeviceContext *>(view.addr);
    ASSERT_EQ(ccu_resolve_program(ctx, spec.program_key, program), CcuStatus::OK);
    struct TensorView {
        struct {
            uint64_t addr, size;
        } buffer;
        uint64_t start_offset;
        uint32_t ndims, shapes[1], strides[1], dtype;
    } tensor{{0x10000, 4096}, 0, 1, {16}, {1}, 0};
    CcuRequestRef request{};
    ASSERT_EQ(ccu_reserve_request(ctx, program, request, tensor, tensor, uint64_t{16}), CcuStatus::OK);
    ASSERT_EQ(ccu_submit_request(request), CcuStatus::OK);
    EXPECT_EQ(ccu_poll_request(request).state, CcuRequestState::PENDING);
    EXPECT_EQ(session.release_memory(1), CcuStatus::BUSY);
    state->Complete(request, {CcuRequestState::SUCCEEDED, CcuStatus::OK, 0});
    EXPECT_EQ(ccu_poll_request(request).state, CcuRequestState::SUCCEEDED);
    ccu_retire_request(request);
    ASSERT_EQ(session.prepare_memory(2, 0x20000, 4096), CcuStatus::OK);
    ASSERT_EQ(session.get_device_context(view), CcuStatus::OK);
    ctx = reinterpret_cast<CcuDeviceContext *>(view.addr);
    EXPECT_EQ(ctx->next_sequence, 2U);
    EXPECT_EQ(session.close(CcuCloseMode::NORMAL), CcuStatus::OK);
}
