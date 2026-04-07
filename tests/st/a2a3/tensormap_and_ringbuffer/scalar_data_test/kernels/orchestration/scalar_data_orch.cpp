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
 * Scalar Data Dependency Test Orchestration
 *
 * End-to-end test for get_tensor_data, set_tensor_data, and add_inout
 * with runtime-created outputs and initial value support.
 *
 * Flow:
 *   1. c = a + b           (kernel_add, runtime-created tensor)
 *   2. get_tensor_data(c, {0})   → check[0] = 2.0
 *   3. get_tensor_data(c, {100}) → check[1] = 102.0
 *   4. scalar_tensor = add_output(TensorCreateInfo, 77.0f), submit noop
 *   5. get_tensor_data(scalar_tensor, {0}) → check[2] = 77.0
 *   6. add_inout(scalar_tensor) (INOUT path), submit noop
 *   7. get_tensor_data(scalar_tensor, {0}) → check[3] = 77.0
 *   8. check[4] = 2.0 + 77.0 = 79.0  (orchestration arithmetic)
 *   9. set_tensor_data(scalar_tensor, {0}, 42.0), get_tensor_data → check[5] = 42.0
 *  10. Orch set_tensor_data(d, {0}, 10.0) → kernel_add(d, a) → check[6] = 12.0
 *  11. WAW+WAR: kernel_add reads c → set_tensor_data(c, 88.0) auto-waits → check[7] = 88.0
 *  12. External WAR with INOUT: noop(ext_b as INOUT) → set_tensor_data(ext_b) → check[8] = 55.0
 *  13. result = a + b      (kernel_add, external output via INOUT)
 */

#include <stddef.h>
#include <stdint.h>

#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

#define FUNC_ADD 0
#define FUNC_NOOP 1

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig
aicpu_orchestration_config(const ChipStorageTaskArgs &orch_args) {
    (void)orch_args;  // NOLINT(readability/casting)
    return PTO2OrchestrationConfig{
        .expected_arg_count = 4,  // a, b, result, check
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipStorageTaskArgs &orch_args) {
    // External tensors from golden.py
    Tensor ext_a = from_tensor_arg(orch_args.tensor(0));
    Tensor ext_b = from_tensor_arg(orch_args.tensor(1));
    Tensor ext_result = from_tensor_arg(orch_args.tensor(2));
    Tensor ext_check = from_tensor_arg(orch_args.tensor(3));

    uint32_t SIZE = orch_args.tensor(0).shapes[0];
    LOG_INFO("scalar_data_test: SIZE=%u, check_size=%u", SIZE, orch_args.tensor(3).shapes[0]);

    uint32_t inter_shapes[1] = {SIZE};
    TensorCreateInfo inter_ci(inter_shapes, 1, DataType::FLOAT32);

    // =========================================================
    // Step 1: c = a + b (runtime-created tensor, kernel_add)
    // =========================================================
    Arg params_c;
    params_c.add_input(ext_a);
    params_c.add_input(ext_b);
    params_c.add_output(inter_ci);
    TaskOutputTensors c_outs = pto2_rt_submit_aiv_task(FUNC_ADD, params_c);
    const Tensor &c = c_outs.get_ref(0);

    // =========================================================
    // Step 2: get_tensor_data(c, {0}) → check[0]
    //   Tests TensorMap lookup + spin-wait for kernel completion
    // =========================================================
    uint32_t idx[1] = {0};
    float c0_val = get_tensor_data<float>(c, 1, idx);
    LOG_INFO("get_tensor_data(c, {0}) = %f (expected 2.0)", static_cast<double>(c0_val));

    uint32_t check_idx[1] = {0};
    set_tensor_data(ext_check, 1, check_idx, c0_val);

    // =========================================================
    // Step 3: get_tensor_data(c, {100}) → check[1]
    //   Tests flat offset calculation for non-zero index
    // =========================================================
    idx[0] = 100;
    float c100_val = get_tensor_data<float>(c, 1, idx);
    LOG_INFO("get_tensor_data(c, {100}) = %f (expected 102.0)", static_cast<double>(c100_val));

    check_idx[0] = 1;
    set_tensor_data(ext_check, 1, check_idx, c100_val);

    // =========================================================
    // Step 4: Runtime-created scalar output with initial value
    //   Runtime allocates HeapRing buffer, writes 77.0 to element [0]
    // =========================================================
    uint32_t scalar_shapes[1] = {1};
    TensorCreateInfo scalar_ci(scalar_shapes, 1, DataType::FLOAT32);
    scalar_ci.set_initial_value(77.0f);

    Arg params_scalar;
    params_scalar.add_output(scalar_ci);
    TaskOutputTensors scalar_outs = pto2_rt_submit_aiv_task(FUNC_NOOP, params_scalar);
    const Tensor &scalar_tensor = scalar_outs.get_ref(0);

    // =========================================================
    // Step 5: get_tensor_data(scalar_tensor, {0}) → check[2]
    //   Verifies initial value was written correctly
    // =========================================================
    idx[0] = 0;
    float s0_val = get_tensor_data<float>(scalar_tensor, 1, idx);
    LOG_INFO("get_tensor_data(scalar_tensor, {0}) after init = %f (expected 77.0)", static_cast<double>(s0_val));

    check_idx[0] = 2;
    set_tensor_data(ext_check, 1, check_idx, s0_val);

    // =========================================================
    // Step 6: add_inout(scalar_tensor) second use → INOUT path
    //   Buffer already exists, so the noop just registers dependency
    // =========================================================
    {
        Arg args;
        args.add_inout(scalar_tensor);
        pto2_rt_submit_aiv_task(FUNC_NOOP, args);
    }

    // =========================================================
    // Step 7: get_tensor_data(scalar_tensor, {0}) → check[3]
    //   Value should be preserved (noop kernel didn't modify it)
    // =========================================================
    float s1_val = get_tensor_data<float>(scalar_tensor, 1, idx);
    LOG_INFO("get_tensor_data(scalar_tensor, {0}) after 2nd noop = %f (expected 77.0)", static_cast<double>(s1_val));

    check_idx[0] = 3;
    set_tensor_data(ext_check, 1, check_idx, s1_val);

    // =========================================================
    // Step 8: set_tensor_data with orchestration-computed value → check[4]
    //   Tests set_tensor_data write + orchestration arithmetic
    // =========================================================
    float combined = c0_val + s0_val;  // 2.0 + 77.0 = 79.0
    LOG_INFO(
        "Orchestration arithmetic: %f + %f = %f", static_cast<double>(c0_val), static_cast<double>(s0_val),
        static_cast<double>(combined)
    );  // NOLINT(whitespace/line_length)

    check_idx[0] = 4;
    set_tensor_data(ext_check, 1, check_idx, combined);

    // =========================================================
    // Step 9: Orch set→get round-trip on internal tensor
    //   Validates that set_tensor_data writes are visible to get_tensor_data
    //   on the same tensor. Uses scalar_tensor (currently 77.0), overwrites to 42.0.
    // =========================================================
    set_tensor_data(scalar_tensor, 1, idx, 42.0f);
    float rw_val = get_tensor_data<float>(scalar_tensor, 1, idx);
    LOG_INFO("set_tensor_data→get_tensor_data round-trip = %f (expected 42.0)", static_cast<double>(rw_val));

    check_idx[0] = 5;
    set_tensor_data(ext_check, 1, check_idx, rw_val);

    // =========================================================
    // Step 10: Orch→AICore RAW (set_tensor_data → kernel reads)
    //   Orchestration writes d[0]=10.0 via set_tensor_data, then
    //   kernel_add reads d as input: e[0] = d[0] + a[0] = 12.0
    // =========================================================
    Arg params_d;
    params_d.add_output(inter_ci);
    TaskOutputTensors d_outs = pto2_rt_submit_aiv_task(FUNC_NOOP, params_d);
    const Tensor &d = d_outs.get_ref(0);

    idx[0] = 0;
    set_tensor_data(d, 1, idx, 10.0f);

    Arg params_e;
    params_e.add_input(d);
    params_e.add_input(ext_a);
    params_e.add_output(inter_ci);
    TaskOutputTensors e_outs = pto2_rt_submit_aiv_task(FUNC_ADD, params_e);
    const Tensor &e = e_outs.get_ref(0);

    float e0_val = get_tensor_data<float>(e, 1, idx);
    LOG_INFO("Orch→AICore RAW: e[0] = %f (expected 12.0)", static_cast<double>(e0_val));

    check_idx[0] = 6;
    set_tensor_data(ext_check, 1, check_idx, e0_val);

    // =========================================================
    // Step 11: WAW + WAR on internal tensor
    //   c was written by Step 1 (kernel_add, TensorMap has producer entry).
    //   Submit a new kernel that reads c as INPUT (creates consumer dep).
    //   Then set_tensor_data(c) — no manual get_tensor_data sync.
    //   set_tensor_data internally waits for:
    //     - WAW: producer (Step 1) COMPLETED
    //     - WAR: consumer (this kernel) done (fanout_refcount check)
    //
    //   NOTE on external tensors: ext_a was read by Step 1 as INPUT,
    //   but TensorMap has no producer entry for ext_a (only consumers).
    //   set_tensor_data(ext_a) would NOT detect the reader — data race.
    //   To ensure WAR safety on external tensors, use add_inout()
    //   instead of add_input() so TensorMap tracks the access chain.
    // =========================================================
    {
        Arg args;
        args.add_input(c);
        args.add_input(ext_b);
        args.add_output(inter_ci);
        (void)pto2_rt_submit_aiv_task(FUNC_ADD, args);  // NOLINT(readability/casting)
    }

    // set_tensor_data auto-waits for producer + consumer before writing
    idx[0] = 0;
    set_tensor_data(c, 1, idx, 88.0f);
    float waw_val = get_tensor_data<float>(c, 1, idx);
    LOG_INFO("WAW+WAR: set_tensor_data(c, 88.0) after consumer = %f (expected 88.0)", static_cast<double>(waw_val));

    check_idx[0] = 7;
    set_tensor_data(ext_check, 1, check_idx, waw_val);

    // =========================================================
    // Step 12: External tensor WAR — must use add_output or add_inout, not add_input
    //
    //   For external tensors, using add_input() does NOT create a
    //   TensorMap entry. set_tensor_data would then write immediately
    //   without waiting for the reader kernel — a WAR data race.
    //
    //   Using add_output() (or add_inout()) creates a TensorMap entry,
    //   enabling set_tensor_data to detect the producer via TensorMap lookup
    //   and wait for fanout_refcount (all consumers done).
    //
    //   Here we submit noop with ext_b as write-only output (noop doesn't
    //   read data), then set_tensor_data overwrites ext_b[0] = 55.0.
    //   set_tensor_data auto-waits for the noop to complete.
    // =========================================================
    {
        Arg args;
        args.add_output(ext_b);  // write-only: creates TensorMap entry (not add_input!)
        pto2_rt_submit_aiv_task(FUNC_NOOP, args);
    }

    idx[0] = 0;
    set_tensor_data(ext_b, 1, idx, 55.0f);
    float ext_war_val = get_tensor_data<float>(ext_b, 1, idx);
    LOG_INFO(
        "External WAR (INOUT): set_tensor_data(ext_b, 55.0) = %f (expected 55.0)", static_cast<double>(ext_war_val)
    );

    check_idx[0] = 8;
    set_tensor_data(ext_check, 1, check_idx, ext_war_val);

    // Restore ext_b[0] for final result comparison
    set_tensor_data(ext_b, 1, idx, 0.0f);

    // =========================================================
    // Step 13: result = a + b (external output via add_output, kernel_add)
    // =========================================================
    {
        Arg args;
        args.add_input(ext_a);
        args.add_input(ext_b);
        args.add_output(ext_result);
        pto2_rt_submit_aiv_task(FUNC_ADD, args);
    }

    LOG_INFO("scalar_data_test: orchestration complete");
}

}  // extern "C"
