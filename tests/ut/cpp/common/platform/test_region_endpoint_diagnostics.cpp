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

#include <cstdint>
#include <cstdio>
#include <string>

#include "aicpu/region_endpoint_diagnostics.h"

[[noreturn]] void assert_impl(const char *, const char *, int) { throw "assert_impl"; }

struct RuntimeContext;
struct OrchContext;
RuntimeContext *g_runtime = nullptr;
OrchContext *g_orch = nullptr;

extern "C" RuntimeContext *framework_current_runtime(void) { return g_runtime; }
extern "C" void framework_bind_runtime(RuntimeContext *runtime) { g_runtime = runtime; }
extern "C" OrchContext *framework_current_orch_context(void) { return g_orch; }
extern "C" void framework_bind_orch_context(OrchContext *orch) { g_orch = orch; }

namespace {

uint32_t g_kind = 0;
uint64_t g_session = 0;
uint64_t g_transaction = 0;
std::string g_message;

template <typename Ctx>
void capture_endpoint(
    Ctx *, uint32_t kind, uint64_t session_bits, uint64_t transaction_id, const char *, const char *message
) {
    g_kind = kind;
    g_session = session_bits;
    g_transaction = transaction_id;
    g_message = message == nullptr ? "" : message;
}

template <typename Ctx>
void install_on(void (*&slot)(Ctx *, uint32_t, uint64_t, uint64_t, const char *, const char *)) {
    slot = &capture_endpoint<Ctx>;
}

void reset_capture() {
    g_kind = 99;
    g_session = 99;
    g_transaction = 99;
    g_message.clear();
}

template <typename Ops, typename Ctx>
void bind_report(Ops &ops, Ctx &ctx, Ctx *&published) {
    ops = {};
    install_on(ops.report_endpoint_failure);
    ctx = {};
    ctx.ops = &ops;
    published = &ctx;
}

void report_samples() {
    reset_capture();
#if defined(SIMPLER_RUNTIME_host_build_graph)
    OrchOps ops{};
    OrchContext ctx{};
    bind_report(ops, ctx, g_orch);
#else
    RuntimeOps ops{};
    RuntimeContext ctx{};
    bind_report(ops, ctx, g_runtime);
#endif

    region_report_invalid_task_args(RegionTaskArgError::BAD_MAGIC_VERSION);
    EXPECT_EQ(g_kind, kEndpointFailureKindRegion);
    EXPECT_EQ(g_session, 0u);
    EXPECT_EQ(g_transaction, 0u);
    EXPECT_NE(g_message.find("Region endpoint error identity=untrusted op=decode"), std::string::npos);
    EXPECT_NE(g_message.find("kind="), std::string::npos);

    RegionViewError error{};
    error.kind = RegionViewErrorKind::TIMEOUT;
    error.op = RegionViewOp::WAIT;
    error.observed = 3;
    std::snprintf(error.message, sizeof(error.message), "line one\nline two");
    region_report_endpoint_error(0x11, 9, 4, error, 0x2000, 5);
    EXPECT_EQ(g_kind, kEndpointFailureKindRegion);
    EXPECT_EQ(g_session, 0x11u);
    EXPECT_EQ(g_transaction, 9u);
    EXPECT_NE(g_message.find("op=wait"), std::string::npos);
    EXPECT_NE(g_message.find("counter_addr=8192"), std::string::npos);
    EXPECT_NE(g_message.find("counter_operand=5"), std::string::npos);
    EXPECT_EQ(g_message.find('\n'), std::string::npos);
    EXPECT_NE(g_message.find("line one line two"), std::string::npos);
}

}  // namespace

TEST(RegionEndpointDiagnostics, ReportsIdentitySeparateFromTheDetailedLog) { report_samples(); }
