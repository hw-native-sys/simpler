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
 * fully_distributed_within_core engine — public wiring entry.
 *
 * The distributed runtime moves orchestration + scheduling + execution onto the
 * AI cores in SPMD fashion (see docs/fully_distributed_within_core.md). The
 * engine itself (per-core TensorMap, claim race over global cursors, private
 * task ring, run-ahead loop, completion-flag ring, deterministic GM output
 * heap) lives in dist_engine.cpp and is compiled into the AICPU .so so it can
 * reuse the full submit-side type set (TensorMap, MixedKernels, L0TaskArgs,
 * kernel-address resolution).
 *
 * The AICPU "stub" thread does dlopen + arena setup, then calls
 * dist_engine_register() once and publishes the returned per-core entry pointer
 * via Runtime::dist.core_main_fn. Each AICore worker thread invokes that entry,
 * which runs the orchestration entry (replaying the full submit stream) and
 * executes the tasks it wins.
 */

#pragma once

struct PTO2Runtime;
struct L2TaskArgs;
class Runtime;

// Orchestration entry signature (matches DeviceOrchestrationFunc in the AICPU
// executor): the dlopen'd user orchestration function the cores replay.
typedef void (*DistOrchFunc)(const L2TaskArgs &);

/**
 * Wire the distributed engine for one run.
 *
 * Resets the global claim cursors + completion-flag ring, (re)acquires the GM
 * output heap, stores the orchestration entry / args / PTO2Runtime, and points
 * rt->ops at the distributed ops table so the cores route rt_submit_* into the
 * distributed submit path. Must be called once on the AICPU orchestrator thread
 * before publishing Runtime::dist.go.
 *
 * Returns the address of the per-core entry function
 * (signature: void(void *runtime, int core_idx, int core_type)) to store into
 * Runtime::dist.core_main_fn. Returned as void* to keep this header light.
 */
void *dist_engine_register(
    PTO2Runtime *rt, DistOrchFunc orch_func, const L2TaskArgs *orch_args, int num_workers, Runtime *runtime
);

/**
 * Dump a per-core execution swimlane as a Chrome Trace Event JSON.
 *
 * Self-gated on the PTO_DIST_SWIMLANE env var (output file path); a no-op when
 * unset. Each executed (sub)task is one duration event laid out by physical
 * block (pid) and lane AIC/AIV0/AIV1 (tid), so the trace shows how the
 * execute-first claim race spreads work across cores (load balance, docs §6.1).
 * Must be called AFTER all workers have finished a run (single-threaded), e.g.
 * by the AICPU stub once Runtime::dist.done_count == num_workers.
 */
void dist_engine_dump_trace();
