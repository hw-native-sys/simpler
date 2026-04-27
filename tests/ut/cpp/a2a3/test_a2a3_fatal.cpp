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
/**
 * Unit tests for PTO2 A2A3 fatal error handling.
 *
 * Tests API short-circuit after fatal state, explicit fatal routing,
 * and allocation with invalid arguments.
 */

#include <gtest/gtest.h>

#include <array>
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
    bool fatal = false;
    int submit_calls = 0;
    int alloc_calls = 0;
    int scope_begin_calls = 0;
    int scope_end_calls = 0;
    int get_calls = 0;
    int set_calls = 0;
    int report_fatal_calls = 0;
    int32_t last_fatal_code = PTO2_ERROR_NONE;
    std::string last_fatal_func;
    std::string last_fatal_message;
};

static_assert(offsetof(FakeRuntime, ops) == 0);  // Guard: reinterpret_cast below assumes ops is first member.

FakeRuntime *as_fake(PTO2Runtime *rt) { return reinterpret_cast<FakeRuntime *>(rt); }

TaskOutputTensors fake_submit(PTO2Runtime *rt, const MixedKernels &, const Arg &, bool) {
    as_fake(rt)->submit_calls++;
    return TaskOutputTensors{};
}

void fake_scope_begin(PTO2Runtime *rt) { as_fake(rt)->scope_begin_calls++; }
void fake_scope_end(PTO2Runtime *rt) { as_fake(rt)->scope_end_calls++; }
void fake_orchestration_done(PTO2Runtime *) {}
bool fake_is_fatal(PTO2Runtime *rt) { return as_fake(rt)->fatal; }

void fake_report_fatal(PTO2Runtime *rt, int32_t error_code, const char *func, const char *fmt, ...) {
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

uint64_t fake_get_tensor_data(PTO2Runtime *rt, const Tensor &, uint32_t, const uint32_t[]) {
    as_fake(rt)->get_calls++;
    return 0x1234ULL;
}

void fake_set_tensor_data(PTO2Runtime *rt, const Tensor &, uint32_t, const uint32_t[], uint64_t) {
    as_fake(rt)->set_calls++;
}

TaskOutputTensors fake_alloc_tensors(PTO2Runtime *rt, const Arg &) {
    as_fake(rt)->alloc_calls++;
    return TaskOutputTensors{};
}

const PTO2RuntimeOps kFakeOps = {
    .submit_task = fake_submit,
    .scope_begin = fake_scope_begin,
    .scope_end = fake_scope_end,
    .orchestration_done = fake_orchestration_done,
    .is_fatal = fake_is_fatal,
    .report_fatal = fake_report_fatal,
    .log_error = fake_log,
    .log_warn = fake_log,
    .log_info = fake_log,
    .log_debug = fake_log,
    .log_always = fake_log,
    .get_tensor_data = fake_get_tensor_data,
    .set_tensor_data = fake_set_tensor_data,
    .alloc_tensors = fake_alloc_tensors,
};

class RuntimeBindingGuard {
public:
    explicit RuntimeBindingGuard(PTO2Runtime *rt) { pto2_framework_bind_runtime(rt); }
    ~RuntimeBindingGuard() { pto2_framework_bind_runtime(nullptr); }
};

TensorCreateInfo make_ci() {
    static const uint32_t kShape[1] = {1};
    return TensorCreateInfo(kShape, 1, DataType::FLOAT32);
}

}  // namespace

TEST(A2A3Fatal, ApiShortCircuitsAfterFatal) {
    FakeRuntime runtime{};
    runtime.ops = &kFakeOps;
    runtime.fatal = true;
    RuntimeBindingGuard bind(reinterpret_cast<PTO2Runtime *>(&runtime));

    MixedKernels mixed{};
    Arg args;
    uint32_t indices[1] = {0};
    uint32_t shape[1] = {1};
    Tensor tensor = make_tensor_external(reinterpret_cast<void *>(0x1), shape, 1);

    EXPECT_TRUE(pto2_rt_submit_task(mixed, args).empty());
    EXPECT_TRUE(alloc_tensors(args).empty());
    EXPECT_EQ(get_tensor_data<uint64_t>(tensor, 0, indices), 0U);
    set_tensor_data<uint64_t>(tensor, 0, indices, 1U);
    pto2_rt_scope_begin();
    pto2_rt_scope_end();
    {
        PTO2ScopeGuard guard;
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

TEST(A2A3Fatal, ExplicitFatalRoutesThroughOps) {
    FakeRuntime runtime{};
    runtime.ops = &kFakeOps;
    RuntimeBindingGuard bind(reinterpret_cast<PTO2Runtime *>(&runtime));

    pto2_rt_report_fatal(PTO2_ERROR_EXPLICIT_ORCH_FATAL, "boom %d", 7);

    EXPECT_TRUE(runtime.fatal);
    EXPECT_EQ(runtime.report_fatal_calls, 1);
    EXPECT_EQ(runtime.last_fatal_code, PTO2_ERROR_EXPLICIT_ORCH_FATAL);
    EXPECT_EQ(runtime.last_fatal_message, "boom 7");
    EXPECT_FALSE(runtime.last_fatal_func.empty());

    MixedKernels mixed{};
    Arg args;
    EXPECT_TRUE(pto2_rt_submit_task(mixed, args).empty());
    EXPECT_EQ(runtime.submit_calls, 0);
}

TEST(A2A3Fatal, AllocTensorConvenienceReportsInvalidArgsInsteadOfAsserting) {
    FakeRuntime runtime{};
    runtime.ops = &kFakeOps;
    RuntimeBindingGuard bind(reinterpret_cast<PTO2Runtime *>(&runtime));

    std::array<TensorCreateInfo, MAX_TENSOR_ARGS + 1> create_infos{};
    for (TensorCreateInfo &ci : create_infos) {
        ci = make_ci();
    }

    auto alloc_from_array = static_cast<TaskOutputTensors (*)(const TensorCreateInfo[], uint32_t)>(&alloc_tensors);
    EXPECT_TRUE(alloc_from_array(create_infos.data(), static_cast<uint32_t>(create_infos.size())).empty());
    EXPECT_EQ(runtime.report_fatal_calls, 1);
    EXPECT_EQ(runtime.last_fatal_code, PTO2_ERROR_INVALID_ARGS);
    EXPECT_EQ(runtime.alloc_calls, 0);
}
