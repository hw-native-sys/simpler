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

#include <cstdarg>
#include <cstdio>
#include <string>

#include "pto_orchestration_api.h"
[[noreturn]] void assert_impl(const char *, const char *, int) { throw "assert_impl"; }

namespace {

PTO2Runtime *g_bound_runtime = nullptr;

extern "C" PTO2Runtime *pto2_framework_current_runtime(void) { return g_bound_runtime; }
extern "C" void pto2_framework_bind_runtime(PTO2Runtime *rt) { g_bound_runtime = rt; }

struct FakeRuntime {
    const PTO2RuntimeOps *ops;
    PTO2ScopeMode pending_scope_mode = PTO2ScopeMode::AUTO;
    bool fatal = false;
    int submit_calls = 0;
    int alloc_calls = 0;
    int scope_begin_calls = 0;
    int scope_end_calls = 0;
    PTO2ScopeMode last_scope_mode = PTO2ScopeMode::AUTO;
    TaskOutputTensors submit_result{};
    TaskOutputTensors alloc_result{};
};

FakeRuntime *as_fake(PTO2Runtime *rt) { return reinterpret_cast<FakeRuntime *>(rt); }

TaskOutputTensors fake_submit(PTO2Runtime *rt, const MixedKernels &, const Arg &) {
    FakeRuntime *fake = as_fake(rt);
    fake->submit_calls++;
    return fake->submit_result;
}

void fake_scope_begin(PTO2Runtime *rt) {
    FakeRuntime *fake = as_fake(rt);
    fake->scope_begin_calls++;
    fake->last_scope_mode = fake->pending_scope_mode;
    fake->pending_scope_mode = PTO2ScopeMode::AUTO;
}

void fake_scope_end(PTO2Runtime *rt) { as_fake(rt)->scope_end_calls++; }
void fake_orchestration_done(PTO2Runtime *) {}
bool fake_is_fatal(PTO2Runtime *rt) { return as_fake(rt)->fatal; }

void fake_report_fatal(PTO2Runtime *, int32_t, const char *, const char *, ...) {}
void fake_log(const char *, const char *, ...) {}
uint64_t fake_get_tensor_data(PTO2Runtime *, const Tensor &, uint32_t, const uint32_t[]) { return 0; }
void fake_set_tensor_data(PTO2Runtime *, const Tensor &, uint32_t, const uint32_t[], uint64_t) {}

TaskOutputTensors fake_alloc_tensors(PTO2Runtime *rt, const Arg &) {
    FakeRuntime *fake = as_fake(rt);
    fake->alloc_calls++;
    return fake->alloc_result;
}

const PTO2RuntimeOps kFakeOps = {
    fake_submit,
    fake_scope_begin,
    fake_scope_end,
    fake_orchestration_done,
    fake_is_fatal,
    fake_report_fatal,
    fake_log,
    fake_log,
    fake_log,
    fake_log,
    fake_log,
    fake_get_tensor_data,
    fake_set_tensor_data,
    fake_alloc_tensors,
};

class RuntimeBindingGuard {
public:
    explicit RuntimeBindingGuard(PTO2Runtime *rt) { pto2_framework_bind_runtime(rt); }
    ~RuntimeBindingGuard() { pto2_framework_bind_runtime(nullptr); }
};

}  // namespace

TEST(A2A3ManualScopeApi, ScopeGuardPassesManualModeToRuntimeOps) {
    FakeRuntime runtime{};
    runtime.ops = &kFakeOps;
    RuntimeBindingGuard bind(reinterpret_cast<PTO2Runtime *>(&runtime));

    {
        PTO2ScopeGuard guard(PTO2ScopeMode::MANUAL);
        (void)guard;
    }

    EXPECT_EQ(runtime.scope_begin_calls, 1);
    EXPECT_EQ(runtime.scope_end_calls, 1);
    EXPECT_EQ(runtime.last_scope_mode, PTO2ScopeMode::MANUAL);
}

TEST(A2A3ManualScopeApi, SubmitAndAllocExposeTaskIds) {
    FakeRuntime runtime{};
    runtime.ops = &kFakeOps;
    uint32_t shape[1] = {1};
    Tensor submit_tensor = make_tensor_external(reinterpret_cast<void *>(0x10), shape, 1);
    Tensor alloc_tensor = make_tensor_external(reinterpret_cast<void *>(0x20), shape, 1);
    submit_tensor.owner_task_id = PTO2TaskId::make(0, 17);
    alloc_tensor.owner_task_id = PTO2TaskId::make(0, 23);
    runtime.submit_result.materialize_output(submit_tensor);
    runtime.alloc_result.materialize_output(alloc_tensor);
    RuntimeBindingGuard bind(reinterpret_cast<PTO2Runtime *>(&runtime));

    Arg args;
    MixedKernels kernels{};

    auto submit_out = pto2_rt_submit_task(kernels, args);
    auto alloc_out = alloc_tensors(args);

    EXPECT_EQ(runtime.submit_calls, 1);
    EXPECT_EQ(runtime.alloc_calls, 1);
    EXPECT_EQ(submit_out.task_id(), PTO2TaskId::make(0, 17));
    EXPECT_EQ(alloc_out.task_id(), PTO2TaskId::make(0, 23));
}

TEST(A2A3ManualScopeApi, ArgStoresExplicitDeps) {
    Arg args;

    args.add_dep(PTO2TaskId::make(0, 9));
    args.add_dep(PTO2TaskId::make(1, 11));

    ASSERT_EQ(args.explicit_dep_count(), 2u);
    EXPECT_EQ(args.explicit_dep(0), PTO2TaskId::make(0, 9));
    EXPECT_EQ(args.explicit_dep(1), PTO2TaskId::make(1, 11));
    EXPECT_FALSE(args.has_error);
}
