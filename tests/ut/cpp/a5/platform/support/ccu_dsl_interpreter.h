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
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "hcomm/ccu/ccu_primitives.hpp"

namespace ccu_test {
struct Operation {
    std::string kind, expression;
    uint64_t mask{};
    uint64_t out{}, a{}, b{}, immediate{};
    CcuConditionType condition{};
    char arithmetic{};
};

struct Interpretation {
    size_t steps{};
    std::map<std::string, size_t> loop_iterations;
};

// Interprets recorded scalar DSL operations, not data transfer or hardware timing.
class RecordedProgram {
public:
    explicit RecordedProgram(std::vector<Operation> operations) :
        operations_(std::move(operations)) {
        const auto size = operations_.size();
        ends_.assign(size, size);
        alternatives_.assign(size, size);
        starts_.assign(size, size);
        std::vector<size_t> stack;
        for (size_t i = 0; i < size; ++i) {
            const auto &op = operations_[i];
            if (op.kind == "if" || op.kind == "loop") {
                stack.push_back(i);
            } else if (op.kind == "else" || op.kind == "end_if" || op.kind == "end_loop") {
                if (stack.empty()) throw std::runtime_error("Unmatched DSL control terminator");
                const auto begin = stack.back();
                const auto &opening = operations_[begin];
                const auto expected = op.kind == "end_loop" ? "loop" : "if";
                if (opening.expression != op.expression || opening.kind != expected)
                    throw std::runtime_error("Mismatched DSL control label");
                if (op.kind == "else") {
                    if (alternatives_[begin] != size) throw std::runtime_error("Duplicate DSL else");
                    alternatives_[begin] = i;
                } else {
                    ends_[begin] = i;
                    starts_[i] = begin;
                    if (alternatives_[begin] != size) ends_[alternatives_[begin]] = i;
                    stack.pop_back();
                }
            }
        }
        if (!stack.empty()) throw std::runtime_error("Unclosed DSL control block");
    }

    Interpretation Run(std::vector<uint64_t> &registers, size_t budget = 20000) const {
        Interpretation result;
        for (size_t pc = 0; pc < operations_.size();) {
            if (++result.steps > budget) throw std::runtime_error("Recorded DSL exceeded execution budget");
            const auto &op = operations_[pc];
            if (op.kind == "assign_imm" || op.kind == "load") {
                registers.at(op.out) = op.immediate;
                ++pc;
            } else if (op.kind == "assign_var") {
                registers.at(op.out) = registers.at(op.a);
                ++pc;
            } else if (op.kind == "arithmetic") {
                if (op.arithmetic != '+') throw std::runtime_error("Unsupported V1 arithmetic in recording");
                registers.at(op.out) = registers.at(op.a) + registers.at(op.b);
                ++pc;
            } else if (op.kind == "if" || op.kind == "loop") {
                bool condition;
                if (op.condition == CCU_CONDITION_EQ) condition = registers.at(op.a) == op.immediate;
                else if (op.condition == CCU_CONDITION_NE) condition = registers.at(op.a) != op.immediate;
                else throw std::runtime_error("Unsupported V1 condition in recording");
                if (condition) {
                    if (op.kind == "loop") ++result.loop_iterations[op.expression];
                    ++pc;
                } else {
                    pc = (alternatives_[pc] != operations_.size() ? alternatives_[pc] : ends_[pc]) + 1;
                }
            } else if (op.kind == "else") {
                pc = ends_[pc] + 1;
            } else if (op.kind == "end_loop") {
                pc = starts_[pc];
            } else if (op.kind == "end_if") {
                ++pc;
            } else {
                throw std::runtime_error("Unsupported operation in scalar DSL recording: " + op.kind);
            }
        }
        return result;
    }

private:
    std::vector<Operation> operations_;
    std::vector<size_t> ends_, alternatives_, starts_;
};
}  // namespace ccu_test
