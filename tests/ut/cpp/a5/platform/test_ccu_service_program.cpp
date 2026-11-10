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
#include "support/ccu_dsl_interpreter.h"
#include "hcomm/ccu/ccu_primitives.hpp"

namespace {
// This recorder captures the production DSL callback; it does not emulate hardware.
using ccu_test::Operation;
std::vector<Operation> operations;
std::map<uint64_t, std::string> expressions;
// Arithmetic values are evaluated independently of the recorded control flow.
std::map<uint64_t, uint64_t> values;
std::set<std::string> labels;
uint64_t next_handle = 1;
uint64_t multiply_input{};
uint16_t multiply_factor{};
size_t multiply_begin{}, multiply_end{};
std::string Expr(uint64_t handle) { return expressions[handle]; }
std::string BinaryExpr(uint64_t a, uint64_t b, const char *symbol) {
    const auto rhs = Expr(b);
    if (std::string(symbol) == "+" && !rhs.empty() && std::all_of(rhs.begin(), rhs.end(), [](char ch) {
            return ch >= '0' && ch <= '9';
        }))
        return Expr(a) + symbol + rhs;
    return "var" + std::to_string(a) + symbol + "var" + std::to_string(b);
}
CcuResult Emit(const char *kind, std::string expression = {}, uint64_t mask = 0) {
    operations.push_back({kind, std::move(expression), mask});
    return CCU_SUCCESS;
}
CcuResult Allocate(uint64_t *out) {
    *out = next_handle++;
    return CCU_SUCCESS;
}
int UserMultiply(CcuBuildContext *builder) {
    multiply_begin = operations.size();
    const auto product = builder->multiply(builder->scalar(0), multiply_factor);
    const auto result = builder->require_equal(product, builder->constant(multiply_input * multiply_factor));
    multiply_end = operations.size();
    return result == CcuStatus::OK ? 0 : 1;
}
class CcuServiceProgramTest : public testing::Test {
protected:
    void SetUp() override {
        operations.clear();
        expressions.clear();
        values.clear();
        labels.clear();
        next_handle = 1;
        multiply_input = 0;
        multiply_factor = 0;
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
    values[out] = value;
    const auto result = Emit("assign_imm");
    operations.back().out = out;
    operations.back().immediate = value;
    return result;
}
CcuResult CcuVariableAssignVar(CcuVariableHandle out, CcuVariableHandle value) {
    expressions[out] = Expr(value);
    values[out] = values[value];
    const auto result = Emit("assign_var");
    operations.back().out = out;
    operations.back().a = value;
    return result;
}
CcuResult CcuAddressAssignVar(CcuAddressHandle out, CcuVariableHandle value) {
    expressions[out] = Expr(value);
    return Emit("address");
}
#define CCU_TEST_BINARY(name, symbol, kind)                                           \
    CcuResult name(CcuVariableHandle out, CcuVariableHandle a, CcuVariableHandle b) { \
        expressions[out] = BinaryExpr(a, b, #symbol);                                 \
        values[out] = values[a] symbol values[b];                                     \
        const auto result = Emit(kind);                                               \
        operations.back().out = out;                                                  \
        operations.back().a = a;                                                      \
        operations.back().b = b;                                                      \
        operations.back().arithmetic = #symbol[0];                                    \
        return result;                                                                \
    }
CCU_TEST_BINARY(CcuVariableAddVarToVar, +, "arithmetic")
CCU_TEST_BINARY(CcuVariableSubVarToVar, -, "arithmetic")
CCU_TEST_BINARY(CcuVariableMulVarToVar, *, "multiply")
#undef CCU_TEST_BINARY
#define CCU_TEST_IMMEDIATE(name, symbol, kind)                               \
    CcuResult name(CcuVariableHandle out, CcuVariableHandle a, uint16_t b) { \
        expressions[out] = Expr(a) + #symbol + std::to_string(b);            \
        values[out] = values[a] symbol b;                                    \
        return Emit(kind);                                                   \
    }
CcuResult CcuVariableAddImmToVar(CcuVariableHandle, CcuVariableHandle, uint16_t) {
    (void)Emit("immediate_add");
    return CCU_E_NOT_SUPPORT;
}
CCU_TEST_IMMEDIATE(CcuVariableSubImmToVar, -, "arithmetic")
CCU_TEST_IMMEDIATE(CcuVariableMulImmToVar, *, "multiply")
#undef CCU_TEST_IMMEDIATE
CcuResult CcuLoadArg(CcuVariableHandle out, uint32_t index) {
    expressions[out] = "arg" + std::to_string(index);
    return Emit("task_arg", {}, index);
}
CcuResult CcuLoadVarFromVarAddr(CcuVariableHandle addr, CcuVariableHandle out, uint32_t) {
    expressions[out] = "mem(" + Expr(addr) + ")";
    values[out] =
        Expr(addr) == "arg1+" + std::to_string(offsetof(CcuPositionalArguments, scalars)) ? multiply_input : 0;
    const auto result = Emit("load", Expr(addr));
    operations.back().out = out;
    operations.back().immediate = values[out];
    return result;
}
CcuResult CcuStoreVarToVarAddr(CcuVariableHandle addr, CcuVariableHandle value, uint32_t) {
    const auto result = Emit("store", Expr(addr), values[value]);
    operations.back().a = value;
    return result;
}
CcuResult CcuEventWait(CcuEventHandle event, uint16_t mask) { return Emit("wait", std::to_string(event), mask); }
CcuResult CcuNotifyRecord(ChannelHandle, uint32_t index, uint16_t mask) {
    return Emit("notify", std::to_string(index), mask);
}
CcuResult CcuNotifyWait(ChannelHandle, uint32_t index, uint16_t mask) {
    return Emit("peer_wait", std::to_string(index), mask);
}
CcuResult CcuIfBegin(CcuVariableHandle variable, uint64_t immediate, CcuConditionType condition, const char *label) {
    EXPECT_TRUE(labels.insert(label).second);
    const auto result = Emit("if", label);
    operations.back().a = variable;
    operations.back().immediate = immediate;
    operations.back().condition = condition;
    if (condition != CCU_CONDITION_EQ && condition != CCU_CONDITION_NE) return CCU_E_NOT_SUPPORT;
    return result;
}
CcuResult CcuIfBeginVar(CcuVariableHandle, CcuVariableHandle, CcuConditionType, const char *) {
    (void)Emit("variable_condition");
    return CCU_E_NOT_SUPPORT;
}
CcuResult CcuIfElse(const char *label) { return Emit("else", label); }
CcuResult CcuIfEnd(const char *label) { return Emit("end_if", label); }
CcuResult CcuWhileBegin(CcuVariableHandle variable, uint64_t immediate, CcuConditionType condition, const char *label) {
    const auto result = Emit("loop", label);
    operations.back().a = variable;
    operations.back().immediate = immediate;
    operations.back().condition = condition;
    if (condition != CCU_CONDITION_EQ && condition != CCU_CONDITION_NE) return CCU_E_NOT_SUPPORT;
    return result;
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

TEST_F(CcuServiceProgramTest, ConstantMultiplyPreservesValuesAndOverflowGuardAcrossUint16Factors) {
    for (const uint16_t factor : {4, 65535}) {
        const auto limit = factor == 0 ? UINT64_MAX : UINT64_MAX / factor;
        for (const uint64_t input : {uint64_t{0}, uint64_t{1024}, limit, limit + 1}) {
            SetUp();
            multiply_input = input;
            multiply_factor = factor;
            SCOPED_TRACE(testing::Message() << "input=" << input << " factor=" << factor);
            const std::vector<CcuRegisteredFunction> functions{{UserMultiply, {}}};
            CcuServiceBuildArgs args{13, 17, 0, sizeof(CcuPositionalArguments), 29, &functions};
            ASSERT_EQ(CcuServiceProgram(&args), CCU_SUCCESS);
            EXPECT_TRUE(std::none_of(operations.begin(), operations.end(), [](const auto &op) {
                return op.kind == "multiply" || op.kind == "immediate_add";
            }));
            std::vector<Operation> recorded(operations.begin() + multiply_begin, operations.begin() + multiply_end);
            uint64_t status_handle = 0;
            for (size_t i = 1; i < recorded.size(); ++i) {
                if (recorded[i - 1].kind == "if" && recorded[i].kind == "assign_imm" &&
                    recorded[i].immediate == static_cast<uint64_t>(CcuStatus::INVALID_ARGUMENT))
                    status_handle = recorded[i].out;
            }
            ASSERT_NE(status_handle, 0U);
            std::vector<uint64_t> registers(next_handle, 0);
            const auto execution = ccu_test::RecordedProgram(std::move(recorded)).Run(registers);
            EXPECT_LT(execution.steps, 20000U);
            EXPECT_EQ(
                registers[status_handle],
                static_cast<uint64_t>(factor != 0 && input > limit ? CcuStatus::INVALID_ARGUMENT : CcuStatus::OK)
            );
        }
    }
}
