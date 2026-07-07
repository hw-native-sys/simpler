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

#include <cstddef>
#include <cstdint>

struct PTO2Runtime;
struct L2TaskArgs;
class Runtime;

// Host-preallocated shared-segment reserve (docs §13/§15.2, 方案B).
//
// On a5 onboard, the DistGlobal segment + GM output heap MUST be reachable from
// every AICore's MPU. AICPU-side halMemAlloc returns an AICPU-process aperture
// (~0x3ffa…) that the AICore MPU does not map, so dereferencing it inside
// dist_core_main faults with "MPU address access invalid" (errcode 271). The
// host therefore pre-allocates ONE device buffer via rtMalloc(RT_MEMORY_HBM) —
// the same allocator that produces the AICore-visible tensor/runtime VAs
// (~0x1000…) — and publishes its base/size through Runtime::dist.seg_base /
// seg_size. dist_engine_register bump-carves DistGlobal + heap from it. When
// seg_base is 0 (sim, or an arch that has not wired this) the engine falls back
// to the legacy dist_alloc_gm_segment allocator, so sim/a2a3 behavior is
// unchanged.
//
// Layout inside the reserve: DistGlobal at offset 0, then the heap ring at a
// page-aligned offset after it. sizeof(DistGlobal) is dominated by
// cores[RUNTIME_MAX_WORKER] each holding a DistTensorMap
// (kRingBuckets*kBucketCapMax MapEntry ≈ 2 MiB), so it runs ~150 MiB on a
// 72-core SKU; the default heap ring is 64 MiB (kHeapRingDefault). 320 MiB
// leaves headroom for both plus alignment. A PTO_DIST_HEAP_MB that overflows
// the reserve triggers a deterministic FATAL (always_assert) in
// dist_engine_register. Device HBM is tens of GiB, so the reserve is cheap and
// allocated once (reused across runs).
constexpr size_t DIST_SEG_RESERVE_BYTES = (320ull << 20);

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
    PTO2Runtime *rt, DistOrchFunc orch_func, const L2TaskArgs *orch_args, int num_workers, Runtime *runtime,
    uint64_t orch_bind_func = 0
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

// Per-core SPMD entry. On onboard it is CCEC-compiled into the AICore binary
// and called directly by aicore_execute; on sim the AICPU .so exports it and
// each AICore worker thread reaches it via the Runtime::dist.core_main_fn
// function pointer. Marked __aicore__ so the CCEC AICore build can call it
// from aicore_execute (a __host__ function is not callable from __aicore__).
// The first parameter is __gm__ void* because on AICore the Runtime lives in
// GM (a distinct CCEC address space — a generic void* would be an illegal
// address-space cast). __aicore__/__gm__ are defined as empty for non-CCEC
// builds (AICPU aarch64, host g++, sim), so the same declaration works in
// every compile unit.
#ifndef __aicore__
#define __aicore__
#endif
#ifndef __gm__
#define __gm__
#endif
#ifdef __cplusplus
extern "C" {
#endif
__aicore__ void dist_core_main(__gm__ void *runtime_v, int core_idx, int core_type_int);
#ifdef __cplusplus
}
#endif
