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
#include <random>
#include <set>
#include <string>
#include <vector>

#include "ccu_service_program.h"
#include "ccu_v1_arithmetic.h"
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
std::string fail_operation;
uint64_t multiply_input{};
uint16_t multiply_factor{};
uint32_t repeated_checks{};
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
int UserMultiply(CcuBuildContext *builder) {
    multiply_begin = operations.size();
    const auto product = builder->multiply(builder->scalar("count"), multiply_factor);
    const auto result = builder->require_equal(product, builder->constant(multiply_input * multiply_factor));
    multiply_end = operations.size();
    return result == CcuStatus::OK ? 0 : 1;
}
int UserRepeatedChecks(CcuBuildContext *builder) {
    const auto value = builder->scalar("count");
    const auto bound = builder->constant(1024);
    for (uint32_t i = 0; i < repeated_checks; ++i)
        (void)builder->require_le(value, bound);
    return 0;
}
class CcuServiceProgramTest : public testing::Test {
protected:
    void SetUp() override {
        operations.clear();
        expressions.clear();
        values.clear();
        labels.clear();
        next_handle = 1;
        fail_operation.clear();
        multiply_input = 0;
        multiply_factor = 0;
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
    values[out] = Expr(addr) == "arg1+48" ? multiply_input : 0;
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

TEST_F(CcuServiceProgramTest, ServiceAndCompletionPackingNeverRecordUnsupportedMultiply) {
    for (uint32_t rank = 0; rank < 2; ++rank) {
        SetUp();
        ASSERT_EQ(Build(rank, true), CCU_SUCCESS);
        EXPECT_TRUE(std::none_of(operations.begin(), operations.end(), [](const auto &op) {
            return op.kind == "multiply" || op.kind == "immediate_add";
        }));
        const auto completion = std::find_if(operations.begin(), operations.end(), [](const auto &op) {
            return op.kind == "store" &&
                   op.expression == "arg0+" + std::to_string(offsetof(CcuRequestRecord, completed_state));
        });
        ASSERT_NE(completion, operations.end());
        EXPECT_EQ(
            completion->mask,
            (static_cast<uint64_t>(CcuStatus::EXECUTION_FAILED) << 32) | static_cast<uint64_t>(CcuRequestState::FAILED)
        );
    }
}

TEST_F(CcuServiceProgramTest, CompletionPreservesLocalErrorsAndPacksBothTerminalStates) {
    for (uint32_t rank = 0; rank < 2; ++rank) {
        SetUp();
        ASSERT_EQ(Build(rank, true), CCU_SUCCESS);
        const auto peer = std::find_if(operations.begin(), operations.end(), [](const auto &op) {
            return op.kind == "load" && op.expression == "arg3+184";
        });
        ASSERT_NE(peer, operations.end());
        const auto failed = std::find_if(peer, operations.end(), [](const auto &op) {
            return op.kind == "if" && op.expression == "failed";
        });
        ASSERT_NE(failed, operations.end());
        const auto end = std::find_if(failed, operations.end(), [](const auto &op) {
            return op.kind == "end_if" && op.expression == "failed";
        });
        ASSERT_NE(end, operations.end());
        const auto completion = std::find_if(end, operations.end(), [](const auto &op) {
            return op.kind == "store" &&
                   op.expression == "arg0+" + std::to_string(offsetof(CcuRequestRecord, completed_state));
        });
        ASSERT_NE(completion, operations.end());
        const ccu_test::RecordedProgram program(std::vector<Operation>(peer + 1, end + 1));
        for (const auto local_status : {CcuStatus::OK, CcuStatus::INVALID_ARGUMENT, CcuStatus::STALE_HANDLE}) {
            for (const auto peer_status : {CcuStatus::OK, CcuStatus::INVALID_ARGUMENT}) {
                std::vector<uint64_t> registers(next_handle, 0);
                registers[failed->a] = static_cast<uint64_t>(local_status);
                registers[peer->out] = static_cast<uint64_t>(peer_status);
                (void)program.Run(registers);
                const auto expected_status = local_status != CcuStatus::OK ? local_status :
                                             peer_status != CcuStatus::OK  ? CcuStatus::EXECUTION_FAILED :
                                                                             CcuStatus::OK;
                const auto expected_state =
                    expected_status == CcuStatus::OK ? CcuRequestState::SUCCEEDED : CcuRequestState::FAILED;
                EXPECT_EQ(registers[failed->a], static_cast<uint64_t>(expected_status));
                EXPECT_EQ(
                    registers[completion->a],
                    (static_cast<uint64_t>(expected_status) << 32) | static_cast<uint64_t>(expected_state)
                );
            }
        }
    }
}

TEST_F(CcuServiceProgramTest, ConstantMultiplyPreservesValuesAndOverflowGuardAcrossUint16Factors) {
    for (const uint16_t factor : {0, 1, 2, 3, 4, 255, 32768, 65535}) {
        const auto limit = factor == 0 ? UINT64_MAX : UINT64_MAX / factor;
        for (const uint64_t input : {uint64_t{0}, uint64_t{1}, uint64_t{1024}, limit, UINT64_MAX}) {
            SetUp();
            multiply_input = input;
            multiply_factor = factor;
            SCOPED_TRACE(testing::Message() << "input=" << input << " factor=" << factor);
            auto function = Function(1, "multiply");
            function.spec.library_path = "user.so";
            function.entry = UserMultiply;
            std::vector<CcuRegisteredFunction> functions{function};
            CcuServiceBuildArgs args{13, 17, 0, 56, 29, &functions};
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
namespace {
std::vector<std::pair<uint64_t, uint64_t>> ComparisonInputs() {
    const std::vector<uint64_t> boundaries{
        0,
        1,
        2,
        3,
        1024,
        (uint64_t{1} << 32) - 1,
        uint64_t{1} << 32,
        (uint64_t{1} << 63) - 1,
        uint64_t{1} << 63,
        UINT64_MAX - 1,
        UINT64_MAX
    };
    std::vector<std::pair<uint64_t, uint64_t>> inputs;
    for (auto lhs : boundaries)
        for (auto rhs : boundaries)
            inputs.emplace_back(lhs, rhs);
    std::mt19937_64 random(37188);
    for (size_t i = 0; i < 256; ++i)
        inputs.emplace_back(random(), random());
    return inputs;
}
}  // namespace

TEST_F(CcuServiceProgramTest, UnsignedComparisonExecutesRecordedV1InstructionsAcrossRepeatedInputs) {
    AscendC::ccu::Variable lhs, rhs;
    ccu_service_detail::ArithmeticScratch scratch;
    const auto relation = ccu_service_detail::CompareUnsigned(lhs, rhs, "comparison", scratch);
    ASSERT_NE(relation.handle, lhs.handle);
    ASSERT_NE(relation.handle, rhs.handle);
    const ccu_test::RecordedProgram program(operations);
    std::vector<uint64_t> registers(next_handle, UINT64_MAX);
    for (const auto &input : ComparisonInputs()) {
        SCOPED_TRACE(testing::Message() << "lhs=" << input.first << " rhs=" << input.second);
        registers[lhs.handle] = input.first;
        registers[rhs.handle] = input.second;
        const auto execution = program.Run(registers);
        const auto expected = input.first < input.second ? ccu_service_detail::LESS :
                              input.first > input.second ? ccu_service_detail::GREATER :
                                                           ccu_service_detail::EQUAL;
        EXPECT_EQ(registers[relation.handle], expected);
        EXPECT_EQ(registers[lhs.handle], input.first);
        EXPECT_EQ(registers[rhs.handle], input.second);
        EXPECT_EQ(execution.loop_iterations.at("comparison.bits"), 64U);
        EXPECT_EQ(execution.loop_iterations.at("comparison.shifts"), 2016U);
        EXPECT_LT(execution.steps, 20000U);
    }
    EXPECT_THROW(program.Run(registers, 100), std::runtime_error);
}

TEST_F(CcuServiceProgramTest, EqualityExecutesRecordedV1InstructionsAndPreservesInputs) {
    AscendC::ccu::Variable lhs, rhs;
    ccu_service_detail::ArithmeticScratch scratch;
    const auto difference = ccu_service_detail::EqualityDifference(lhs, rhs, "equality", scratch);
    ASSERT_NE(difference.handle, lhs.handle);
    ASSERT_NE(difference.handle, rhs.handle);
    const ccu_test::RecordedProgram program(operations);
    std::vector<uint64_t> registers(next_handle, UINT64_MAX);
    for (const auto &input : ComparisonInputs()) {
        registers[lhs.handle] = input.first;
        registers[rhs.handle] = input.second;
        const auto execution = program.Run(registers, 1000);
        EXPECT_EQ(registers[difference.handle], input.first - input.second);
        EXPECT_EQ(registers[difference.handle] == 0, input.first == input.second);
        EXPECT_EQ(registers[lhs.handle], input.first);
        EXPECT_EQ(registers[rhs.handle], input.second);
        EXPECT_EQ(execution.loop_iterations.at("equality.negate"), 64U);
    }
}

TEST_F(CcuServiceProgramTest, ConditionalLoweringPreservesAllUnsignedRelations) {
    for (const auto condition :
         {CCU_CONDITION_EQ, CCU_CONDITION_NE, CCU_CONDITION_LT, CCU_CONDITION_LE, CCU_CONDITION_GT, CCU_CONDITION_GE}) {
        SetUp();
        AscendC::ccu::Variable lhs, rhs, selected;
        ccu_service_detail::ArithmeticScratch scratch;
        selected = 0;
        ccu_service_detail::BeginIfVariables(lhs, rhs, condition, "select", scratch);
        selected = 1;
        ASSERT_EQ(CcuIfEnd("select"), CCU_SUCCESS);
        const ccu_test::RecordedProgram program(operations);
        std::vector<uint64_t> registers(next_handle, UINT64_MAX);
        for (const auto &input : ComparisonInputs()) {
            registers[lhs.handle] = input.first;
            registers[rhs.handle] = input.second;
            (void)program.Run(registers);
            bool expected = false;
            switch (condition) {
            case CCU_CONDITION_EQ:
                expected = input.first == input.second;
                break;
            case CCU_CONDITION_NE:
                expected = input.first != input.second;
                break;
            case CCU_CONDITION_LT:
                expected = input.first < input.second;
                break;
            case CCU_CONDITION_LE:
                expected = input.first <= input.second;
                break;
            case CCU_CONDITION_GT:
                expected = input.first > input.second;
                break;
            case CCU_CONDITION_GE:
                expected = input.first >= input.second;
                break;
            }
            EXPECT_EQ(registers[selected.handle], expected)
                << "condition=" << condition << " lhs=" << input.first << " rhs=" << input.second;
        }
    }
}

TEST_F(CcuServiceProgramTest, ConstantAdditionWritesThroughAliasedHandleAndWrapsAtUint64Boundary) {
    AscendC::ccu::Variable source, destination;
    ccu_service_detail::AddConstant(destination, source, UINT64_MAX);
    ccu_service_detail::AddConstant(destination, destination, 2);
    const ccu_test::RecordedProgram program(operations);
    std::vector<uint64_t> registers(next_handle, 0);
    for (const uint64_t value : {uint64_t{0}, uint64_t{7}, UINT64_MAX}) {
        registers[source.handle] = value;
        (void)program.Run(registers);
        EXPECT_EQ(registers[destination.handle], value + 1);
        EXPECT_EQ(registers[source.handle], value);
    }
}

TEST_F(CcuServiceProgramTest, SharedArithmeticScratchPreservesNestedBranchesAcrossRepeatedRequests) {
    using namespace ccu_service_detail;
    AscendC::ccu::Variable lhs, rhs, selected;
    const auto initial_handles = next_handle;
    ArithmeticScratch scratch;
    ASSERT_EQ(next_handle - initial_handles, 10U);
    const auto workspace_handles = next_handle;
    selected = 0;
    BeginIfVariables(lhs, rhs, CCU_CONDITION_LE, "outer", scratch);
    selected = 1;
    BeginIfVariables(lhs, rhs, CCU_CONDITION_EQ, "equal", scratch);
    selected = 2;
    ASSERT_EQ(CcuIfElse("equal"), CCU_SUCCESS);
    selected = 3;
    ASSERT_EQ(CcuIfEnd("equal"), CCU_SUCCESS);
    ASSERT_EQ(CcuIfElse("outer"), CCU_SUCCESS);
    selected = 4;
    BeginIfVariables(rhs, lhs, CCU_CONDITION_LT, "reversed", scratch);
    selected = 5;
    ASSERT_EQ(CcuIfEnd("reversed"), CCU_SUCCESS);
    ASSERT_EQ(CcuIfEnd("outer"), CCU_SUCCESS);
    const auto difference = EqualityDifference(lhs, lhs, "same_input", scratch);
    EXPECT_EQ(next_handle, workspace_handles);
    const ccu_test::RecordedProgram program(operations);
    std::vector<uint64_t> registers(next_handle, UINT64_MAX);
    for (const auto &input : ComparisonInputs()) {
        registers[lhs.handle] = input.first;
        registers[rhs.handle] = input.second;
        const auto execution = program.Run(registers, 40000);
        EXPECT_EQ(registers[selected.handle], input.first < input.second ? 3U : input.first == input.second ? 2U : 5U);
        EXPECT_EQ(registers[difference.handle], 0U);
        EXPECT_EQ(registers[lhs.handle], input.first);
        EXPECT_EQ(registers[rhs.handle], input.second);
        EXPECT_LT(execution.steps, 40000U);
    }
}

TEST_F(CcuServiceProgramTest, ArithmeticScratchRejectsInputsThatAliasItsTemporaryRegisters) {
    AscendC::ccu::Variable lhs, rhs;
    ccu_service_detail::ArithmeticScratch scratch;
    EXPECT_THROW(
        ccu_service_detail::CompareUnsigned(scratch.relation, rhs, "alias_comparison", scratch), std::invalid_argument
    );
    EXPECT_THROW(
        ccu_service_detail::EqualityDifference(lhs, scratch.probe_rhs, "alias_equality", scratch), std::invalid_argument
    );
    EXPECT_TRUE(operations.empty());
}

TEST_F(CcuServiceProgramTest, ServiceComparisonsShareOneRegisterWorkspaceAcrossEntries) {
    uint64_t first_entry_handles = 0;
    for (const uint32_t entries : {1U, 2U}) {
        uint64_t first_check_handles = 0;
        for (const uint32_t checks : {1U, 32U}) {
            SetUp();
            repeated_checks = checks;
            std::vector<CcuRegisteredFunction> functions;
            for (uint32_t i = 0; i < entries; ++i) {
                auto function = Function(static_cast<uint8_t>(i + 1), "repeated_checks");
                function.spec.library_path = "user.so";
                function.entry = UserRepeatedChecks;
                functions.push_back(std::move(function));
            }
            CcuServiceBuildArgs args{13, 17, 0, 56, 29, &functions};
            ASSERT_EQ(CcuServiceProgram(&args), CCU_SUCCESS);
            if (checks == 1) first_check_handles = next_handle;
            else EXPECT_EQ(next_handle, first_check_handles);
        }
        if (entries == 1) first_entry_handles = first_check_handles;
        // The extra entry owns one load's address/value/offset and its bound.
        else EXPECT_EQ(first_check_handles - first_entry_handles, 4U);
    }
}
