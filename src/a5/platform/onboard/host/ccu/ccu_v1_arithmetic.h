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

#include <cstdint>
#include <stdexcept>
#include <string>

#include "hcomm/ccu/ccu_primitives.hpp"

namespace ccu_service_detail {
namespace ccu = AscendC::ccu;

inline void Check(CcuResult result) {
    if (result != CCU_SUCCESS) throw std::runtime_error("CCU V1 arithmetic build failed");
}

inline void AddConstant(ccu::Variable destination, ccu::Variable source, uint64_t value) {
    ccu::Variable offset;
    offset = value;
    // V1 ADD accepts variable operands; its immediate subtypes require V2.
    destination = source + offset;
}

// One service executes these helpers sequentially. Results remain valid only
// until the next helper uses this workspace; caller-owned symbols stay outside it.
struct ArithmeticScratch {
    ccu::Variable residual_lhs, residual_rhs, probe_lhs, probe_rhs, relation;
    ccu::Variable negative_mask, shift_count, shifts_left, bits_left, decrement;

    bool contains(CcuVariableHandle handle) const {
        return handle == residual_lhs.handle || handle == residual_rhs.handle || handle == probe_lhs.handle ||
               handle == probe_rhs.handle || handle == relation.handle || handle == negative_mask.handle ||
               handle == shift_count.handle || handle == shifts_left.handle || handle == bits_left.handle ||
               handle == decrement.handle;
    }
    void validate_inputs(ccu::Variable lhs, ccu::Variable rhs) const {
        if (contains(lhs.handle) || contains(rhs.handle))
            throw std::invalid_argument("CCU arithmetic inputs must not alias scratch registers");
    }
};

inline ccu::Variable
EqualityDifference(ccu::Variable lhs, ccu::Variable rhs, const std::string &label, ArithmeticScratch &scratch) {
    scratch.validate_inputs(lhs, rhs);
    auto &negative = scratch.residual_lhs;
    auto &addend = scratch.probe_rhs;
    auto &remaining = scratch.bits_left;
    auto &decrement = scratch.decrement;
    auto &difference = scratch.relation;
    negative = 0;
    addend = rhs;
    remaining = 64;
    decrement = UINT64_MAX;
    const auto loop = label + ".negate";
    Check(CcuWhileBegin(remaining.handle, 0, CCU_CONDITION_NE, loop.c_str()));
    negative = negative + addend;
    addend = addend + addend;
    remaining = remaining + decrement;
    Check(CcuWhileEnd(loop.c_str()));
    // The 64-term geometric sum is -rhs modulo 2^64.
    difference = lhs + negative;
    return difference;
}

inline constexpr uint64_t EQUAL = 0;
inline constexpr uint64_t LESS = 1;
inline constexpr uint64_t GREATER = 2;

inline ccu::Variable
CompareUnsigned(ccu::Variable lhs, ccu::Variable rhs, const std::string &label, ArithmeticScratch &scratch) {
    scratch.validate_inputs(lhs, rhs);
    auto &residual_lhs = scratch.residual_lhs;
    auto &residual_rhs = scratch.residual_rhs;
    auto &probe_lhs = scratch.probe_lhs;
    auto &probe_rhs = scratch.probe_rhs;
    auto &relation = scratch.relation;
    auto &negative_mask = scratch.negative_mask;
    auto &shift_count = scratch.shift_count;
    auto &shifts_left = scratch.shifts_left;
    auto &bits_left = scratch.bits_left;
    auto &decrement = scratch.decrement;
    residual_lhs = lhs;
    residual_rhs = rhs;
    relation = EQUAL;
    negative_mask = UINT64_MAX;
    shift_count = 63;
    bits_left = 64;
    decrement = UINT64_MAX;
    const auto bits = label + ".bits", shifts = label + ".shifts";
    const auto lhs_set = label + ".lhs_set", rhs_clear = label + ".rhs_clear";
    const auto rhs_greater = label + ".rhs_greater", rhs_set = label + ".rhs_set";
    Check(CcuWhileBegin(bits_left.handle, 0, CCU_CONDITION_NE, bits.c_str()));
    probe_lhs = residual_lhs;
    probe_rhs = residual_rhs;
    shifts_left = shift_count;
    Check(CcuWhileBegin(shifts_left.handle, 0, CCU_CONDITION_NE, shifts.c_str()));
    probe_lhs = probe_lhs + probe_lhs;
    probe_rhs = probe_rhs + probe_rhs;
    shifts_left = shifts_left + decrement;
    Check(CcuWhileEnd(shifts.c_str()));
    // Lower bits are already zero. Shifting by 63-i isolates bit i at bit 63.
    Check(CcuIfBegin(probe_lhs.handle, 0, CCU_CONDITION_NE, lhs_set.c_str()));
    residual_lhs = residual_lhs + negative_mask;
    Check(CcuIfBegin(probe_rhs.handle, 0, CCU_CONDITION_EQ, rhs_clear.c_str()));
    relation = GREATER;
    Check(CcuIfEnd(rhs_clear.c_str()));
    Check(CcuIfElse(lhs_set.c_str()));
    Check(CcuIfBegin(probe_rhs.handle, 0, CCU_CONDITION_NE, rhs_greater.c_str()));
    relation = LESS;
    Check(CcuIfEnd(rhs_greater.c_str()));
    Check(CcuIfEnd(lhs_set.c_str()));
    Check(CcuIfBegin(probe_rhs.handle, 0, CCU_CONDITION_NE, rhs_set.c_str()));
    residual_rhs = residual_rhs + negative_mask;
    Check(CcuIfEnd(rhs_set.c_str()));
    negative_mask = negative_mask + negative_mask;
    shift_count = shift_count + decrement;
    bits_left = bits_left + decrement;
    Check(CcuWhileEnd(bits.c_str()));
    return relation;
}

inline void BeginIfVariables(
    ccu::Variable lhs, ccu::Variable rhs, CcuConditionType condition, const std::string &label,
    ArithmeticScratch &scratch
) {
    if (condition == CCU_CONDITION_EQ || condition == CCU_CONDITION_NE) {
        const auto difference = EqualityDifference(lhs, rhs, label, scratch);
        Check(CcuIfBegin(difference.handle, 0, condition, label.c_str()));
        return;
    }
    const auto relation = CompareUnsigned(lhs, rhs, label, scratch);
    switch (condition) {
    case CCU_CONDITION_LT:
        Check(CcuIfBegin(relation.handle, LESS, CCU_CONDITION_EQ, label.c_str()));
        break;
    case CCU_CONDITION_LE:
        Check(CcuIfBegin(relation.handle, GREATER, CCU_CONDITION_NE, label.c_str()));
        break;
    case CCU_CONDITION_GT:
        Check(CcuIfBegin(relation.handle, GREATER, CCU_CONDITION_EQ, label.c_str()));
        break;
    case CCU_CONDITION_GE:
        Check(CcuIfBegin(relation.handle, LESS, CCU_CONDITION_NE, label.c_str()));
        break;
    default:
        throw std::runtime_error("Unsupported CCU condition");
    }
}
}  // namespace ccu_service_detail
