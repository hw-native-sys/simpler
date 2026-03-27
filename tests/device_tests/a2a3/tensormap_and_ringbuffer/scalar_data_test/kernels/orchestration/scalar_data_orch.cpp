/**
 * Scalar Data Dependency Test Orchestration
 *
 * End-to-end test for get_tensor_data, set_tensor_data, and add_inout
 * with initial value support.
 *
 * Flow:
 *   1. c = a + b           (kernel_add, internal tensor)
 *   2. get_tensor_data(c, {0})   → check[0] = 2.0
 *   3. get_tensor_data(c, {100}) → check[1] = 102.0
 *   4. scalar_tensor = make_tensor({1}), add_output(scalar_tensor, 77.0f), submit noop
 *   5. get_tensor_data(scalar_tensor, {0}) → check[2] = 77.0
 *   6. add_inout(scalar_tensor) (INOUT path), submit noop
 *   7. get_tensor_data(scalar_tensor, {0}) → check[3] = 77.0
 *   8. check[4] = 2.0 + 77.0 = 79.0  (orchestration arithmetic)
 *   9. set_tensor_data(scalar_tensor, {0}, 42.0), get_tensor_data → check[5] = 42.0
 *  10. Orch set_tensor_data(d, {0}, 10.0) → kernel_add(d, a) → check[6] = 12.0
 *  11. WAW+WAR: kernel_add reads c → set_tensor_data(c, 88.0) auto-waits → check[7] = 88.0
 *  12. External WAR with INOUT: noop(ext_b as INOUT) → set_tensor_data(ext_b) → check[8] = 55.0
 *  13. result = a + b      (kernel_add, external output, ext_b[0] restored)
 */

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "pto_orchestration_api.h"

#define FUNC_ADD  0
#define FUNC_NOOP 1

static uint64_t float_to_u64(float f) {
    union {
        float f32;
        uint64_t u64;
    } conv;
    conv.u64 = 0;
    conv.f32 = f;
    return conv.u64;
}

static float u64_to_float(uint64_t u) {
    union {
        uint64_t u64;
        float f32;
    } conv;
    conv.u64 = u;
    return conv.f32;
}

extern "C" {

__attribute__((visibility("default")))
PTO2OrchestrationConfig aicpu_orchestration_config(TaskArg* orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 4,  // a, b, result, check
    };
}

__attribute__((visibility("default")))
void aicpu_orchestration_entry(TaskArg* orch_args,
                               int orch_thread_num,
                               int orch_thread_index) {
    (void)orch_thread_num;
    (void)orch_thread_index;

    // External tensors from golden.py
    Tensor ext_a      = from_task_arg(orch_args[0]);
    Tensor ext_b      = from_task_arg(orch_args[1]);
    Tensor ext_result  = from_task_arg(orch_args[2]);
    Tensor ext_check   = from_task_arg(orch_args[3]);

    uint32_t SIZE = orch_args[0].tensor.shapes[0];
    LOG_INFO("scalar_data_test: SIZE=%u, check_size=%u",
             SIZE, orch_args[3].tensor.shapes[0]);

    // =========================================================
    // Step 1: c = a + b (internal tensor, kernel_add)
    // =========================================================
    uint32_t inter_shapes[1] = {SIZE};
    Tensor c = make_tensor(inter_shapes, 1, DataType::FLOAT32);

    {
        PTOParam params;
        params.add_input(ext_a);
        params.add_input(ext_b);
        params.add_output(c);
        pto2_rt_submit_aiv_task(FUNC_ADD, params);
    }

    // =========================================================
    // Step 2: get_tensor_data(c, {0}) → check[0]
    //   Tests TensorMap lookup + spin-wait for kernel completion
    // =========================================================
    uint32_t idx[1] = {0};
    uint64_t c0_raw = get_tensor_data(c, 1, idx);
    float c0_val = u64_to_float(c0_raw);
    LOG_INFO("get_tensor_data(c, {0}) = %f (expected 2.0)", (double)c0_val);

    uint32_t check_idx[1] = {0};
    set_tensor_data(ext_check, 1, check_idx, c0_raw);

    // =========================================================
    // Step 3: get_tensor_data(c, {100}) → check[1]
    //   Tests flat offset calculation for non-zero index
    // =========================================================
    idx[0] = 100;
    uint64_t c100_raw = get_tensor_data(c, 1, idx);
    LOG_INFO("get_tensor_data(c, {100}) = %f (expected 102.0)",
             (double)u64_to_float(c100_raw));

    check_idx[0] = 1;
    set_tensor_data(ext_check, 1, check_idx, c100_raw);

    // =========================================================
    // Step 4: add_inout with initial value (first use → OUTPUT path)
    //   Runtime allocates HeapRing buffer, writes 77.0 to element [0]
    // =========================================================
    uint32_t scalar_shapes[1] = {1};
    Tensor scalar_tensor = make_tensor(scalar_shapes, 1, DataType::FLOAT32);

    {
        PTOParam params;
        params.add_output(scalar_tensor, float_to_u64(77.0f));
        pto2_rt_submit_aiv_task(FUNC_NOOP, params);
    }

    // =========================================================
    // Step 5: get_tensor_data(scalar_tensor, {0}) → check[2]
    //   Verifies initial value was written correctly
    // =========================================================
    idx[0] = 0;
    uint64_t s0_raw = get_tensor_data(scalar_tensor, 1, idx);
    float s0_val = u64_to_float(s0_raw);
    LOG_INFO("get_tensor_data(scalar_tensor, {0}) after init = %f (expected 77.0)",
             (double)s0_val);

    check_idx[0] = 2;
    set_tensor_data(ext_check, 1, check_idx, s0_raw);

    // =========================================================
    // Step 6: add_inout(scalar_tensor) second use → INOUT path
    //   addr != 0, so registered as INOUT (no reallocation)
    // =========================================================
    {
        PTOParam params;
        params.add_inout(scalar_tensor);
        pto2_rt_submit_aiv_task(FUNC_NOOP, params);
    }

    // =========================================================
    // Step 7: get_tensor_data(scalar_tensor, {0}) → check[3]
    //   Value should be preserved (noop kernel didn't modify it)
    // =========================================================
    uint64_t s1_raw = get_tensor_data(scalar_tensor, 1, idx);
    LOG_INFO("get_tensor_data(scalar_tensor, {0}) after 2nd noop = %f (expected 77.0)",
             (double)u64_to_float(s1_raw));

    check_idx[0] = 3;
    set_tensor_data(ext_check, 1, check_idx, s1_raw);

    // =========================================================
    // Step 8: set_tensor_data with orchestration-computed value → check[4]
    //   Tests set_tensor_data write + orchestration arithmetic
    // =========================================================
    float combined = c0_val + s0_val;  // 2.0 + 77.0 = 79.0
    LOG_INFO("Orchestration arithmetic: %f + %f = %f",
             (double)c0_val, (double)s0_val, (double)combined);

    check_idx[0] = 4;
    set_tensor_data(ext_check, 1, check_idx, float_to_u64(combined));

    // =========================================================
    // Step 9: Orch set→get round-trip on internal tensor
    //   Validates that set_tensor_data writes are visible to get_tensor_data
    //   on the same tensor. Uses scalar_tensor (currently 77.0), overwrites to 42.0.
    // =========================================================
    set_tensor_data(scalar_tensor, 1, idx, float_to_u64(42.0f));
    uint64_t rw_raw = get_tensor_data(scalar_tensor, 1, idx);
    float rw_val = u64_to_float(rw_raw);
    LOG_INFO("set_tensor_data→get_tensor_data round-trip = %f (expected 42.0)",
             (double)rw_val);

    check_idx[0] = 5;
    set_tensor_data(ext_check, 1, check_idx, rw_raw);

    // =========================================================
    // Step 10: Orch→AICore RAW (set_tensor_data → kernel reads)
    //   Orchestration writes d[0]=10.0 via set_tensor_data, then
    //   kernel_add reads d as input: e[0] = d[0] + a[0] = 12.0
    // =========================================================
    Tensor d = make_tensor(inter_shapes, 1, DataType::FLOAT32);
    {
        PTOParam params;
        params.add_output(d);
        pto2_rt_submit_aiv_task(FUNC_NOOP, params);
    }

    idx[0] = 0;
    set_tensor_data(d, 1, idx, float_to_u64(10.0f));

    Tensor e = make_tensor(inter_shapes, 1, DataType::FLOAT32);
    {
        PTOParam params;
        params.add_input(d);
        params.add_input(ext_a);
        params.add_output(e);
        pto2_rt_submit_aiv_task(FUNC_ADD, params);
    }

    uint64_t e0_raw = get_tensor_data(e, 1, idx);
    LOG_INFO("Orch→AICore RAW: e[0] = %f (expected 12.0)",
             (double)u64_to_float(e0_raw));

    check_idx[0] = 6;
    set_tensor_data(ext_check, 1, check_idx, e0_raw);

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
        // Submit kernel that reads c as INPUT (creates fanout on c's producer)
        Tensor f = make_tensor(inter_shapes, 1, DataType::FLOAT32);
        PTOParam params;
        params.add_input(c);
        params.add_input(ext_b);
        params.add_output(f);
        pto2_rt_submit_aiv_task(FUNC_ADD, params);
    }

    // set_tensor_data auto-waits for producer + consumer before writing
    idx[0] = 0;
    set_tensor_data(c, 1, idx, float_to_u64(88.0f));
    uint64_t waw_raw = get_tensor_data(c, 1, idx);
    LOG_INFO("WAW+WAR: set_tensor_data(c, 88.0) after consumer = %f (expected 88.0)",
             (double)u64_to_float(waw_raw));

    check_idx[0] = 7;
    set_tensor_data(ext_check, 1, check_idx, waw_raw);

    // =========================================================
    // Step 12: External tensor WAR — must use INOUT, not INPUT
    //
    //   For external tensors, using add_input() does NOT create a
    //   TensorMap entry. set_tensor_data would then write immediately
    //   without waiting for the reader kernel — a WAR data race.
    //
    //   Using add_inout() creates a TensorMap entry, enabling
    //   set_tensor_data to detect the producer via TensorMap lookup
    //   and wait for fanout_refcount (all consumers done).
    //
    //   Here we submit noop with ext_b as INOUT (noop doesn't modify
    //   data), then set_tensor_data overwrites ext_b[0] = 55.0.
    //   set_tensor_data auto-waits for the noop to complete.
    // =========================================================
    {
        PTOParam params;
        params.add_inout(ext_b);   // INOUT: creates TensorMap entry (not INPUT!)
        pto2_rt_submit_aiv_task(FUNC_NOOP, params);
    }

    idx[0] = 0;
    set_tensor_data(ext_b, 1, idx, float_to_u64(55.0f));
    uint64_t ext_war_raw = get_tensor_data(ext_b, 1, idx);
    LOG_INFO("External WAR (INOUT): set_tensor_data(ext_b, 55.0) = %f (expected 55.0)",
             (double)u64_to_float(ext_war_raw));

    check_idx[0] = 8;
    set_tensor_data(ext_check, 1, check_idx, ext_war_raw);

    // Restore ext_b[0] for final result comparison
    set_tensor_data(ext_b, 1, idx, float_to_u64(0.0f));

    // =========================================================
    // Step 13: result = a + b (external output, kernel_add)
    // =========================================================
    {
        PTOParam params;
        params.add_input(ext_a);
        params.add_input(ext_b);
        params.add_output(ext_result);
        pto2_rt_submit_aiv_task(FUNC_ADD, params);
    }

    LOG_INFO("scalar_data_test: orchestration complete");
}

}  // extern "C"
