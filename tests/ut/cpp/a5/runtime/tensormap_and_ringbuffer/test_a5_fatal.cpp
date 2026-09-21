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
#include <cstdarg>
#include <cstdio>
#include <string>

#include "orchestration_api.h"
[[noreturn]] void assert_impl(const char *, const char *, int) { throw "assert_impl"; }

namespace {

RuntimeContext *g_bound_runtime = nullptr;

extern "C" RuntimeContext *framework_current_runtime(void) { return g_bound_runtime; }
extern "C" void framework_bind_runtime(RuntimeContext *rt) { g_bound_runtime = rt; }

struct FakeRuntime {
    const RuntimeOps *ops;
    ScopeMode pending_scope_mode = ScopeMode::AUTO;
    bool fatal = false;
    int submit_calls = 0;
    int alloc_calls = 0;
    int scope_begin_calls = 0;
    int scope_end_calls = 0;
    ScopeMode last_scope_mode = ScopeMode::AUTO;
    int get_calls = 0;
    int set_calls = 0;
    int report_fatal_calls = 0;
    int32_t last_fatal_code = SIMPLER_ERROR_NONE;
    std::string last_fatal_func;
    std::string last_fatal_message;
};

static_assert(offsetof(FakeRuntime, ops) == 0);  // Guard: reinterpret_cast below assumes ops is first member.
static_assert(
    offsetof(FakeRuntime, pending_scope_mode) == offsetof(RuntimeContext, pending_scope_mode)
);  // ...and pending_scope_mode follows ops, matching the RuntimeContext prefix the inline scope wrappers pun through.

FakeRuntime *as_fake(RuntimeContext *rt) { return reinterpret_cast<FakeRuntime *>(rt); }

TaskOutputTensors fake_submit(RuntimeContext *rt, const MixedKernels &, const CoreTaskArgs &) {
    as_fake(rt)->submit_calls++;
    return TaskOutputTensors{};
}

void fake_scope_begin(RuntimeContext *rt) {
    FakeRuntime *fake = as_fake(rt);
    fake->scope_begin_calls++;
    fake->last_scope_mode = fake->pending_scope_mode;
}
void fake_scope_end(RuntimeContext *rt) { as_fake(rt)->scope_end_calls++; }
void fake_orchestration_done(RuntimeContext *) {}
bool fake_is_fatal(RuntimeContext *rt) { return as_fake(rt)->fatal; }

void fake_report_fatal(RuntimeContext *rt, int32_t error_code, const char *func, const char *fmt, ...) {
    FakeRuntime *fake = as_fake(rt);
    fake->report_fatal_calls++;
    fake->fatal = true;
    fake->last_fatal_code = error_code;
    fake->last_fatal_func = func ? func : "";

    char buffer[256] = {};
    if (fmt != nullptr) {
        va_list args;
        va_start(args, fmt);
        vsnprintf(buffer, sizeof(buffer), fmt, args);
        va_end(args);
    }
    fake->last_fatal_message = buffer;
}

void fake_log(const char *, const char *, ...) {}

uint64_t fake_get_tensor_data(RuntimeContext *rt, const simpler::tmr::Tensor &, uint32_t, const uint32_t[]) {
    as_fake(rt)->get_calls++;
    return 0x1234ULL;
}

void fake_set_tensor_data(RuntimeContext *rt, const simpler::tmr::Tensor &, uint32_t, const uint32_t[], uint64_t) {
    as_fake(rt)->set_calls++;
}

TaskOutputTensors fake_alloc_tensors(RuntimeContext *rt, const CoreTaskArgs &) {
    as_fake(rt)->alloc_calls++;
    return TaskOutputTensors{};
}

TaskOutputTensors fake_submit_dummy(RuntimeContext *, const CoreTaskArgs &) { return TaskOutputTensors{}; }

const RuntimeOps kFakeOps = {
    .submit_task = fake_submit,
    .scope_begin = fake_scope_begin,
    .scope_end = fake_scope_end,
    .orchestration_done = fake_orchestration_done,
    .is_fatal = fake_is_fatal,
    .report_fatal = fake_report_fatal,
    .log_error = fake_log,
    .log_warn = fake_log,
    .log_timing = fake_log,
    .log_info = fake_log,
    .log_debug = fake_log,
    .get_tensor_data = fake_get_tensor_data,
    .set_tensor_data = fake_set_tensor_data,
    .alloc_tensors = fake_alloc_tensors,
    .submit_dummy_task = fake_submit_dummy,
    .scope_set_site = nullptr,
};

class RuntimeBindingGuard {
public:
    explicit RuntimeBindingGuard(RuntimeContext *rt) { framework_bind_runtime(rt); }
    ~RuntimeBindingGuard() { framework_bind_runtime(nullptr); }
};

TensorCreateInfo make_ci() {
    static const uint32_t kShape[1] = {1};
    return TensorCreateInfo(kShape, 1, DataType::FLOAT32);
}

}  // namespace

TEST(A5Fatal, ApiShortCircuitsAfterFatal) {
    FakeRuntime runtime{};
    runtime.ops = &kFakeOps;
    runtime.fatal = true;
    RuntimeBindingGuard bind(reinterpret_cast<RuntimeContext *>(&runtime));

    MixedKernels mixed{};
    CoreTaskArgs args;
    uint32_t indices[1] = {0};
    uint32_t shape[1] = {1};
    simpler::tmr::Tensor tensor = simpler::tmr::make_tensor_external(reinterpret_cast<void *>(0x1), shape, 1);

    EXPECT_TRUE(rt_submit_task(mixed, args).empty());
    EXPECT_TRUE(alloc_tensors(args).empty());
    EXPECT_EQ(get_tensor_data<uint64_t>(tensor, 0, indices), 0U);
    set_tensor_data<uint64_t>(tensor, 0, indices, 1U);
    rt_scope_begin();
    rt_scope_end();
    {
        ScopeGuard guard;
        (void)guard;
    }

    EXPECT_EQ(runtime.submit_calls, 0);
    EXPECT_EQ(runtime.alloc_calls, 0);
    EXPECT_EQ(runtime.get_calls, 0);
    EXPECT_EQ(runtime.set_calls, 0);
    EXPECT_EQ(runtime.scope_begin_calls, 0);
    EXPECT_EQ(runtime.scope_end_calls, 0);
    EXPECT_EQ(runtime.report_fatal_calls, 0);
}

TEST(A5Fatal, ExplicitFatalRoutesThroughOps) {
    FakeRuntime runtime{};
    runtime.ops = &kFakeOps;
    RuntimeBindingGuard bind(reinterpret_cast<RuntimeContext *>(&runtime));

    rt_report_fatal(SIMPLER_ERROR_EXPLICIT_ORCH_FATAL, "boom %d", 7);

    EXPECT_TRUE(runtime.fatal);
    EXPECT_EQ(runtime.report_fatal_calls, 1);
    EXPECT_EQ(runtime.last_fatal_code, SIMPLER_ERROR_EXPLICIT_ORCH_FATAL);
    EXPECT_EQ(runtime.last_fatal_message, "boom 7");
    EXPECT_FALSE(runtime.last_fatal_func.empty());

    MixedKernels mixed{};
    CoreTaskArgs args;
    EXPECT_TRUE(rt_submit_task(mixed, args).empty());
    EXPECT_EQ(runtime.submit_calls, 0);
}

TEST(A5Fatal, ScopeGuardForwardsModeUntilLexicalScopeExit) {
    FakeRuntime runtime{};
    runtime.ops = &kFakeOps;
    RuntimeBindingGuard bind(reinterpret_cast<RuntimeContext *>(&runtime));

    runtime.pending_scope_mode = ScopeMode::MANUAL;

    {
        SIMPLER_SCOPE_GUARD();
        EXPECT_EQ(runtime.scope_begin_calls, 1);
        EXPECT_EQ(runtime.scope_end_calls, 0);
        EXPECT_EQ(runtime.last_scope_mode, ScopeMode::AUTO);
    }

    EXPECT_EQ(runtime.scope_end_calls, 1);

    {
        SIMPLER_SCOPE_GUARD(ScopeMode::MANUAL);
        EXPECT_EQ(runtime.scope_begin_calls, 2);
        EXPECT_EQ(runtime.scope_end_calls, 1);
        EXPECT_EQ(runtime.last_scope_mode, ScopeMode::MANUAL);
    }

    EXPECT_EQ(runtime.scope_end_calls, 2);
}

TEST(A5Fatal, AllocTensorConvenienceReportsInvalidArgsInsteadOfAsserting) {
    FakeRuntime runtime{};
    runtime.ops = &kFakeOps;
    RuntimeBindingGuard bind(reinterpret_cast<RuntimeContext *>(&runtime));

    std::array<TensorCreateInfo, MAX_TENSOR_ARGS + 1> create_infos{};
    for (TensorCreateInfo &ci : create_infos) {
        ci = make_ci();
    }

    auto alloc_from_array = static_cast<TaskOutputTensors (*)(const TensorCreateInfo[], uint32_t)>(&alloc_tensors);
    EXPECT_TRUE(alloc_from_array(create_infos.data(), static_cast<uint32_t>(create_infos.size())).empty());
    EXPECT_EQ(runtime.report_fatal_calls, 1);
    EXPECT_EQ(runtime.last_fatal_code, SIMPLER_ERROR_INVALID_ARGS);
    EXPECT_EQ(runtime.alloc_calls, 0);
}
