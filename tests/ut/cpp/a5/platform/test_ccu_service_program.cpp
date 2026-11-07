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
#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "ccu_service_program.h"
#include "hcomm/ccu/ccu_primitives.hpp"

namespace {
// This recorder captures the production DSL callback; it does not emulate hardware.
struct Operation {
    std::string kind, expression;
    uint64_t mask{};
};
std::vector<Operation> operations;
std::map<uint64_t, std::string> expressions;
std::set<std::string> labels;
uint64_t next_handle = 1;
std::string fail_operation;
std::string Expr(uint64_t handle) { return expressions[handle]; }
CcuResult Emit(const char *kind, std::string expression = {}, uint64_t mask = 0) {
    operations.push_back({kind, std::move(expression), mask});
    return fail_operation == kind ? CCU_E_INTERNAL : CCU_SUCCESS;
}
CcuResult Allocate(uint64_t *out) {
    *out = next_handle++;
    return CCU_SUCCESS;
}
CcuRegisteredFunction Function(uint8_t key, const char *entry) {
    CcuRegisteredFunction f;
    f.spec.entry = entry;
    f.spec.program_key.bytes[31] = key;
    f.names = {"src", "dst", "count"};
    f.argument_bytes = 56;
    f.parameters = {
        {CcuArgumentKind::TENSOR, CcuArgumentType::FLOAT32, 0, 0, 2},
        {CcuArgumentKind::TENSOR, CcuArgumentType::FLOAT32, 3, 0, 1},
        {CcuArgumentKind::SCALAR, CcuArgumentType::UINT64, 6, 0, 0}
    };
    return f;
}
int UserCopy(CcuBuildContext *builder) {
    const auto count = builder->scalar("count");
    (void)builder->require_le(count, builder->constant(1024));
    (void)builder->require_equal(count, builder->peer_scalar("count"));
    const auto bytes = builder->multiply(count, 4);
    if (builder->rank() == 0) {
        (void)builder->local_copy(builder->tensor("dst"), builder->tensor("src"), bytes);
        (void)builder->write_peer(builder->peer_tensor("dst"), builder->tensor("src"), bytes);
    }
    return builder->peer_barrier() == CcuStatus::OK ? 0 : 1;
}
class CcuServiceProgramTest : public testing::Test {
protected:
    void SetUp() override {
        operations.clear();
        expressions.clear();
        labels.clear();
        next_handle = 1;
        fail_operation.clear();
    }
    CcuResult Build(uint32_t rank, bool user = false) {
        std::vector<CcuRegisteredFunction> functions{
            Function(1, "builtin.broadcast.root0"), Function(2, "builtin.broadcast.root1")
        };
        if (user) {
            auto f = Function(3, "build_root_copy");
            f.spec.library_path = "user.so";
            f.entry = UserCopy;
            functions.push_back(std::move(f));
        }
        CcuServiceBuildArgs args{13, 17, rank, 56, 29, &functions};
        return CcuServiceProgram(&args);
    }
};
}  // namespace

extern "C" {
CcuResult CcuVariableAlloc(CcuVariableHandle *out) { return Allocate(out); }
CcuResult CcuBufferAlloc(CcuBufferHandle *out) { return Allocate(out); }
CcuResult CcuEventAlloc(CcuEventHandle *out) { return Allocate(out); }
CcuResult CcuEventGetByIndex(CcuEventHandle event, uint32_t, CcuEventHandle *out) {
    *out = event;
    return CCU_SUCCESS;
}
CcuResult CcuLocalAddrAlloc(CcuLocalAddrHandle *value, CcuAddressHandle *addr, CcuVariableHandle *token) {
    Allocate(value);
    Allocate(addr);
    return Allocate(token);
}
CcuResult CcuRemoteAddrAlloc(CcuRemoteAddrHandle *value, CcuAddressHandle *addr, CcuVariableHandle *token) {
    Allocate(value);
    Allocate(addr);
    return Allocate(token);
}
CcuResult CcuVariableAssignImm(CcuVariableHandle out, uint64_t value) {
    expressions[out] = std::to_string(value);
    return Emit("assign");
}
CcuResult CcuVariableAssignVar(CcuVariableHandle out, CcuVariableHandle value) {
    expressions[out] = Expr(value);
    return Emit("assign");
}
CcuResult CcuAddressAssignVar(CcuAddressHandle out, CcuVariableHandle value) {
    expressions[out] = Expr(value);
    return Emit("address");
}
#define CCU_TEST_BINARY(name, symbol)                                                 \
    CcuResult name(CcuVariableHandle out, CcuVariableHandle a, CcuVariableHandle b) { \
        expressions[out] = Expr(a) + symbol + Expr(b);                                \
        return Emit("arithmetic");                                                    \
    }
CCU_TEST_BINARY(CcuVariableAddVarToVar, "+")
CCU_TEST_BINARY(CcuVariableSubVarToVar, "-")
CCU_TEST_BINARY(CcuVariableMulVarToVar, "*")
#undef CCU_TEST_BINARY
#define CCU_TEST_IMMEDIATE(name, symbol)                                     \
    CcuResult name(CcuVariableHandle out, CcuVariableHandle a, uint16_t b) { \
        expressions[out] = Expr(a) + symbol + std::to_string(b);             \
        return Emit("arithmetic");                                           \
    }
CCU_TEST_IMMEDIATE(CcuVariableAddImmToVar, "+")
CCU_TEST_IMMEDIATE(CcuVariableSubImmToVar, "-")
CCU_TEST_IMMEDIATE(CcuVariableMulImmToVar, "*")
#undef CCU_TEST_IMMEDIATE
CcuResult CcuLoadArg(CcuVariableHandle out, uint32_t index) {
    expressions[out] = "arg" + std::to_string(index);
    return Emit("task_arg", {}, index);
}
CcuResult CcuLoadVarFromVarAddr(CcuVariableHandle addr, CcuVariableHandle out, uint32_t) {
    expressions[out] = "mem(" + Expr(addr) + ")";
    return Emit("load", Expr(addr));
}
CcuResult CcuStoreVarToVarAddr(CcuVariableHandle addr, CcuVariableHandle, uint32_t) {
    return Emit("store", Expr(addr));
}
CcuResult CcuEventWait(CcuEventHandle event, uint16_t mask) { return Emit("wait", std::to_string(event), mask); }
CcuResult CcuNotifyRecord(ChannelHandle, uint32_t index, uint16_t mask) {
    return Emit("notify", std::to_string(index), mask);
}
CcuResult CcuNotifyWait(ChannelHandle, uint32_t index, uint16_t mask) {
    return Emit("peer_wait", std::to_string(index), mask);
}
CcuResult CcuIfBegin(CcuVariableHandle, uint64_t, CcuConditionType, const char *label) {
    EXPECT_TRUE(labels.insert(label).second);
    return Emit("if", label);
}
CcuResult CcuIfBeginVar(CcuVariableHandle, CcuVariableHandle, CcuConditionType, const char *label) {
    EXPECT_TRUE(labels.insert(label).second);
    return Emit("if", label);
}
CcuResult CcuIfElse(const char *label) { return Emit("else", label); }
CcuResult CcuIfEnd(const char *label) { return Emit("end_if", label); }
CcuResult CcuWhileBegin(CcuVariableHandle, uint64_t, CcuConditionType, const char *label) {
    return Emit("loop", label);
}
CcuResult CcuWhileEnd(const char *label) { return Emit("end_loop", label); }
CcuResult CcuLocalCopyMemToBuffer(CcuBufferHandle, CcuLocalAddrHandle, CcuVariableHandle, CcuEventHandle, uint16_t) {
    return Emit("read_local");
}
CcuResult CcuLocalCopyBufferToMem(CcuLocalAddrHandle, CcuBufferHandle, CcuVariableHandle, CcuEventHandle, uint16_t) {
    return Emit("write_local");
}
CcuResult
CcuWriteMemToMem(ChannelHandle, CcuRemoteAddrHandle, CcuLocalAddrHandle, CcuVariableHandle, CcuEventHandle, uint16_t) {
    return Emit("write_peer");
}
}

TEST_F(CcuServiceProgramTest, LoadsStaticResourcesOutsideLoopAndCurrentParametersInsideLoop) {
    ASSERT_EQ(Build(0, true), CCU_SUCCESS);
    const auto loop = std::find_if(operations.begin(), operations.end(), [](const auto &op) {
        return op.kind == "loop";
    });
    ASSERT_NE(loop, operations.end());
    EXPECT_EQ(
        std::count_if(
            operations.begin(), loop,
            [](const auto &op) {
                return op.kind == "task_arg";
            }
        ),
        9
    );
    EXPECT_EQ(
        std::count_if(
            loop, operations.end(),
            [](const auto &op) {
                return op.kind == "task_arg";
            }
        ),
        0
    );
    EXPECT_EQ((loop + 1)->kind, "wait");
    EXPECT_EQ((loop + 1)->expression, "17");
    EXPECT_TRUE(std::any_of(loop, operations.end(), [](const auto &op) {
        return op.kind == "load" && op.expression == "arg0+32";
    }));
    EXPECT_TRUE(std::any_of(loop, operations.end(), [](const auto &op) {
        return op.kind == "load" && op.expression == "arg1+48";
    }));
    EXPECT_EQ(operations.back().kind, "end_loop");
}
TEST_F(CcuServiceProgramTest, BothRanksPublishSequenceOnlyAfterPeerStatusAndDataSynchronization) {
    for (uint32_t rank = 0; rank < 2; ++rank) {
        SetUp();
        ASSERT_EQ(Build(rank, true), CCU_SUCCESS);
        std::vector<std::string> stores;
        size_t sequence_index = 0, peer_status_index = 0;
        for (size_t i = 0; i < operations.size(); ++i) {
            if (operations[i].kind == "store") stores.push_back(operations[i].expression);
            if (operations[i].kind == "store" && operations[i].expression == "arg0+128") sequence_index = i;
            if (operations[i].kind == "peer_wait" && operations[i].mask == 4) peer_status_index = i;
        }
        ASSERT_EQ(stores.size(), 6U);
        EXPECT_EQ(stores.back(), "arg0+128");
        EXPECT_GT(sequence_index, peer_status_index);
        EXPECT_GT(peer_status_index, 0U);
        EXPECT_EQ(
            std::count_if(
                operations.begin(), operations.end(),
                [](const auto &op) {
                    return op.kind == "write_local";
                }
            ),
            rank == 0 ? 2 : 1
        );
    }
}
TEST_F(CcuServiceProgramTest, SdkRecordingErrorsAreNeverReportedAsSuccessfulBuild) {
    for (const auto *failure : {"task_arg", "loop", "if", "wait", "write_peer", "store"}) {
        SetUp();
        fail_operation = failure;
        EXPECT_EQ(Build(0), CCU_E_INTERNAL) << failure;
    }
}
