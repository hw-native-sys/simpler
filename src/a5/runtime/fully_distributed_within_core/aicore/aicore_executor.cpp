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

#include "aicore/aicore.h"
#include "aicore/aicore_profiling_state.h"
#include "common/platform_config.h"  // Register-based communication
#include "pto2_dispatch_payload.h"
#include "runtime.h"
#include "dist_engine.h"

/**
 * AICore main execution loop (fully_distributed_within_core, SPMD-on-core)
 *
 * Runs the AICPU-AICore register-based handshake (phases 1-3) so the AICPU
 * shutdown/teardown path is reused unchanged, then — instead of polling
 * DATA_MAIN_BASE for AICPU-dispatched tasks — invokes the decentralized engine
 * entry on this worker thread:
 * 1. Wait for AICPU ready signal via handshake buffer
 * 2. Report physical core ID and core type, signal AICore ready
 * 3. Wait for Runtime::dist.go, then invoke Runtime::dist.core_main_fn
 * 4. Honor the EXIT signal on DATA_MAIN_BASE and ack EXITED
 *
 * The engine (compiled into the AICPU .so, but executed here on the AICore
 * thread so kernels run with this thread's sim TLS in place) replays the
 * orchestration submit stream, claims/builds the tasks it wins, executes them,
 * and sets this worker's completion flags before returning.
 * See runtime/dist_engine.* and docs/fully_distributed_within_core.md.
 *
 * @param runtime Pointer to Runtime in global memory
 * @param s_block_idx Block index (core ID)
 * @param core_type Core type (AIC or AIV)
 */
__aicore__ __attribute__((weak)) void aicore_execute(__gm__ Runtime *runtime, int s_block_idx, CoreType core_type) {
    __gm__ Handshake *my_hank = (__gm__ Handshake *)(&runtime->workers[s_block_idx]);

    // Phase 1: Wait for AICPU initialization signal
    while (my_hank->aicpu_ready == 0) {
        dcci(my_hank, SINGLE_CACHE_LINE);
        SPIN_WAIT_HINT();
    }

    // Phase 2: Report physical core ID, signal ready
    my_hank->physical_core_id = get_physical_core_id();
    OUT_OF_ORDER_STORE_BARRIER();
    my_hank->aicore_regs_ready = 1;
    dcci(&my_hank->aicore_regs_ready, SINGLE_CACHE_LINE, CACHELINE_OUT);
    while (my_hank->aicpu_regs_ready == 0) {
        dcci(&my_hank->aicpu_regs_ready, SINGLE_CACHE_LINE);
        SPIN_WAIT_HINT();
    }
    // Report initial idle status via register
    write_reg(RegId::COND, AICORE_IDLE_VALUE);

    // Phase 3: Report core type, signal ready
    my_hank->core_type = core_type;
    OUT_OF_ORDER_STORE_BARRIER();
    my_hank->aicore_done = s_block_idx + 1;  // Signal ready (use s_block_idx + 1 to avoid 0)

    dcci(my_hank, SINGLE_CACHE_LINE, CACHELINE_OUT);

    // ===========================================================================
    // fully_distributed_within_core: run orchestration + scheduling + execution
    // ON this AICore worker (SPMD). Instead of polling DATA_MAIN_BASE for
    // AICPU-dispatched tasks, each worker invokes the distributed engine entry
    // (compiled into the AICPU .so, but executed here on the AICore thread so
    // kernels run with this thread's sim TLS in place). The engine replays the
    // orchestration submit stream, claims/builds the tasks it wins, and executes
    // them; on return it has set this worker's completion flags. The worker then
    // honors the existing teardown protocol (wait for EXIT, ack EXITED) so the
    // AICPU scheduler/shutdown path is reused unchanged.
    // See runtime/dist_engine.* and docs/fully_distributed_within_core.md.
    // ===========================================================================
    // ===========================================================================
    // DEBUG: do NOT write runtime->dist.aicore_progress here — it shares a cache
    // line with dist.go (false sharing clobbers the AICPU's go=1 store).
    // Poll the PER-CORE dist_go flag. It lives in the FIRST handshake cache line
    // (offset 32), so we invalidate that line with dcci(my_hank,
    // SINGLE_CACHE_LINE) — the SAME primitive the Phase 1 aicpu_ready poll uses
    // and that is proven to deliver the AICPU's store to all 9 cores on A5.
    while (my_hank->dist_go == 0) {
        dcci(my_hank, SINGLE_CACHE_LINE);
        SPIN_WAIT_HINT();
    }
    // DEBUG: mark this worker saw go=1 and is entering dist_core_main.
    // Own cache line per worker (stride AICORE_PROGRESS_STRIDE): the dcci
    // write-back flushes the whole line, so a dense array would clobber
    // neighbors' crumbs (false sharing).
    runtime->dist.aicore_progress[s_block_idx * AICORE_PROGRESS_STRIDE] = 11;
    dcci(&runtime->dist.aicore_progress[s_block_idx * AICORE_PROGRESS_STRIDE], SINGLE_CACHE_LINE, CACHELINE_OUT);
    {
#if defined(__CPU_SIM)
        // Sim: AICore worker is a host thread; jump to the dist_core_main
        // registered by the AICPU orchestrator thread (same address space).
        DistCoreMainFn core_main = reinterpret_cast<DistCoreMainFn>(runtime->dist.core_main_fn);
        if (core_main != nullptr) {
            core_main(runtime, s_block_idx, static_cast<int>(core_type));
        } else {
            __atomic_add_fetch(&runtime->dist.done_count, 1, __ATOMIC_ACQ_REL);
        }
#else
        // Onboard: dist_core_main is CCEC-compiled into this AICore binary.
        // Pass the __gm__ Runtime* directly — dist_core_main's first parameter
        // is __gm__ void* (dist_engine.h), which matches this address space.
        // The engine recovers the GM segment base from
        // runtime->dist.global_data_base (docs §13, 方案B).
        dist_core_main(runtime, s_block_idx, static_cast<int>(core_type));
#endif
    }
    // DEBUG: mark dist_core_main returned.
    runtime->dist.aicore_progress[s_block_idx * AICORE_PROGRESS_STRIDE] = 12;
    dcci(&runtime->dist.aicore_progress[s_block_idx * AICORE_PROGRESS_STRIDE], SINGLE_CACHE_LINE, CACHELINE_OUT);

    // Signal completion via the PER-CORE dist_done flag (SECOND cache line,
    // offset 64) and dcci write-back of that line — the proven Phase 3 pattern,
    // applied to the isolated dist_done line so the AICPU's first-line
    // (dist_go) flush can never clobber it. The AICPU polls this flag instead
    // of the shared atomicAdd done_count, whose result was not visible to the
    // AICPU on A5 (cacheable-GM atomicAdd never reached an observable line).
    // On sim the done_count increment inside dist_core_main already completed
    // the wait; dist_done is best-effort there.
    my_hank->dist_done = 1u;
    dcci(&my_hank->dist_done, SINGLE_CACHE_LINE, CACHELINE_OUT);

    // Teardown: wait for the AICPU EXIT signal on DATA_MAIN_BASE and ack.
    while (true) {
        uint32_t reg_val = static_cast<uint32_t>(read_reg(RegId::DATA_MAIN_BASE));
        if (reg_val == AICORE_EXIT_SIGNAL) {
            write_reg(RegId::COND, AICORE_EXITED_VALUE);
            break;
        }
        SPIN_WAIT_HINT();
    }
    dcci(my_hank, SINGLE_CACHE_LINE, CACHELINE_OUT);
}
