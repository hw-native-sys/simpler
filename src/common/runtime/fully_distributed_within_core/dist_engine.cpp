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
 * fully_distributed_within_core engine.
 *
 * SPMD orchestration + scheduling + execution on the AI cores. See
 * docs/fully_distributed_within_core.md for the authoritative design and
 * src/.../docs/RUNTIME_LOGIC.md for the local overview.
 *
 * Each AICore worker thread runs dist_core_main(), which:
 *   1. replays the full orchestration submit stream (every core builds an
 *      identical per-core TensorMap and computes identical deterministic GM
 *      output-heap addresses; only ownership differs);
 *   2. on each rt_submit_*, races to claim the task on one of two global
 *      cursors (cube for AIC-anchored, vector for AIV-only). The winner is
 *      owner = builder = executor and builds the task into its private ring;
 *   3. runs an EXECUTE-FIRST run-ahead loop: on every submit point it first
 *      drains ready owned tasks (and pulls follower deposits), THEN claims at
 *      most this one new task. Because claim+build is fast but execute is slow,
 *      interleaving execution with claiming stops a fast core from greedily
 *      claiming a full ring of consecutive tasks: while it executes a long task
 *      other cores advance the cursor and claim subsequent tasks (load balance,
 *      see docs §6/§6.1). The ring (small, kPrivateSlots) only back-pressures
 *      when genuinely full of not-yet-ready tasks. After orchestration returns,
 *      a final loop drains the ring to completion. A task is ready once all its
 *      fan-in producers have set their entry in the global completion-flag
 *      ring; on completion the owner sets its own flag (release).
 *
 * This file is compiled into the AICPU .so (build_config aicore source_dirs do
 * not include runtime/), but dist_core_main runs ON the AICore worker threads
 * (invoked through a function pointer), so kernels execute on AICore threads
 * with their sim TLS in place.
 *
 * M2 scope: single-core tasks (1C / 1V) only — sufficient for benchmark_bgemm.
 * Multi-core co-ownership (MIX / 2V, block.won) is M3; GM heap reclamation is
 * M4. A MIX task encountered in M2 raises a fatal error.
 */

#include "dist_engine.h"

// Onboard AICore replay needs framework_bind_runtime (defined in the
// orchestration common.cpp, which is in the AICore build's source_dirs) so the
// user orchestration's current_runtime()->ops->submit_task resolves to the
// AICore engine. Forward-declared here to avoid pulling the orchestration
// header chain into the engine TU. Signature carries __gm__/__aicore__ to
// match the common.h declaration (on host/AICPU builds both are empty).
extern "C" __aicore__ void framework_bind_runtime(__gm__ PTO2Runtime *rt);
// C14 DIAG: read back the framework's bound runtime to verify the submit chain.
extern "C" __aicore__ __gm__ PTO2Runtime *framework_current_runtime();

#if !defined(SIMPLER_DIST_AICORE_ONLY)
#include "aicpu/device_malloc.h"
#include "aicpu/platform_regs.h"
#endif

#include <atomic>
#include <chrono>
#include <csignal>
#include <ctime>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <vector>

#include "callable.h"
#include "common/core_type.h"
#include "intrinsic.h"
#include "pto2_dispatch_payload.h"
#include "pto_constants.h"
#include "pto_runtime2.h"
#include "pto_submit_types.h"
#include "pto_types.h"
#include "runtime.h"
// A5 AICore cross-core atomics are CCEC builtins (atomicAdd / atomicCAS /
// atomicMin / atomicMax / atomicExch on __gm__), available implicitly — the same
// way inner_kernel.h uses get_coreid/get_ctrl without a CANN header. Used by
// Coherent<T>'s CCEC branch for cross-core RMW on uncacheable GM (docs §14 /
// user guidance: CAS is atomic on uncacheable variables on A5). No include needed.
// SPIN_WAIT_HINT is already provided by pto_runtime2_types.h (included above
// via pto_runtime2.h) using __has_include("spin_hint.h") with a no-op fallback.
// The bare include here is leftover from before the split-engine: on the AICPU
// build spin_hint.h is on the path (redundant but harmless); on the CCEC AICore
// build it is NOT on the path and a bare #include is a hard fatal error. Guard
// it so both builds stay clean.
#if __has_include("spin_hint.h")
#include "spin_hint.h"
#endif
#include "tensor.h"
#include "tensor_create_info.h"

// -----------------------------------------------------------------------------
// Compile-time gates.
//
// PTO2_PROFILING comes from profiling_config.h (default 1; a CCEC build passes
// -DPTO2_PROFILING=0). It is pulled in transitively via pto_types.h above, which
// is included before this point — so the gate below sees the real value.
//
// DIST_TRACE_ENABLED — swimlane tracing (per-task span capture + JSON dump).
// Reuses the project's PTO2_PROFILING macro: sim builds pass PTO2_PROFILING=1, so
// tracing is on there; an AICore/CCEC build that does not pass the macro gets
// `#if (PTO2_PROFILING + 0)` == `#if 0`, so all tracing code (and its host-only
// std::vector / std::chrono / clock_gettime / fprintf usage) is compiled out.
// No #ifndef fallback on purpose: undefined ⇒ off.
#define DIST_TRACE_ENABLED (PTO2_PROFILING + 0)

// DIST_SIM_HOST_CLOCK — sim-only host facilities (steady_clock now_ns() and the
// use_example_exec_time busy-wait kernel emulation). Unavailable under CCEC.
#if defined(__CCE_AICORE__) || defined(__DAV_C220__) || defined(__DAV_CUBE__) || defined(__DAV_VEC__) || \
    defined(__CCE_KT_TEST__)
#define DIST_SIM_HOST_CLOCK 0
#else
#define DIST_SIM_HOST_CLOCK 1
#endif

#if !DIST_SIM_HOST_CLOCK
// Onboard AICore (real CCEC): the user orchestration is CCEC-compiled straight
// INTO this AICore kernel image (docs §16.1 — the test's orchestration source
// is appended to the AICore CUSTOM_SOURCE_DIRS, exactly like its aic/aiv
// compute kernels). Its entry is therefore an ordinary linked symbol in the
// same binary, sharing this image's g_current_runtime / g_dist_ops.
// dist_core_main calls it DIRECTLY below — no GM blob, no cross-binary
// function-pointer jump, and no orch_bind (which existed only to bind the
// separate blob's own g_current_runtime). On sim (DIST_SIM_HOST_CLOCK=1) the
// orchestration is still dlopen'd and reached via gd->orch_func.
//
// WEAK reference on purpose: dist_engine.cpp is also compiled into the shared
// runtime AICore image that build_runtimes.py links WITHOUT any orchestration
// (there is no test source there). A strong undefined reference would fail
// that link. As a weak reference it resolves to 0 when no orchestration is
// linked in (the guarded call below is simply skipped), and binds to the
// test's in-image definition when one IS present (the real per-test image).
extern "C" __attribute__((weak)) __aicore__ void aicpu_orchestration_entry(const L2TaskArgs &orch_args);
#endif

// Tracing needs the host wall clock (now_ns lives under DIST_SIM_HOST_CLOCK), so
// the two gates cannot diverge into "trace on, host clock off". In practice both
// are off together on a CCEC build; assert it so a stray -D combination fails loud.
#if DIST_TRACE_ENABLED && !DIST_SIM_HOST_CLOCK
#error "DIST_TRACE_ENABLED requires DIST_SIM_HOST_CLOCK (swimlane uses the host clock)"
#endif

// AICore is a freestanding target: no stdio, no env, no libc stdlib. The engine
// has many diagnostic call sites (fprintf/stderr/fflush/vfprintf/getenv/atol) in
// error/watchdog/trace paths that are valuable on sim/host but cannot link on
// AICore. Rather than gate every call site individually, define no-op shims
// active ONLY when !DIST_SIM_HOST_CLOCK (i.e. the CCEC AICore build). The
// functional path never relies on their output — failure propagation uses the
// `fatal` flag in DistGlobal (set_fatal) + device atomics, not printed text.
// On sim/host (DIST_SIM_HOST_CLOCK==1) these macros are NOT defined, so the
// real libc symbols are used unchanged.
#if !DIST_SIM_HOST_CLOCK
#undef stderr
#define stderr ((void *)0)
#define fprintf(...) (0)
#define vfprintf(...) (0)
#define fflush(x) (0)
#define getenv(x) ((char *)nullptr)
#define atol(x) (0L)
#endif

namespace {

// GM segment allocation for DistGlobal + output heap. dist_engine_register runs
// on the AICPU; sim uses host malloc (shared address space with sim AICore
// threads), onboard uses halMemAlloc so every AICore can read the segment.
#if !defined(SIMPLER_DIST_AICORE_ONLY)
inline void *dist_alloc_gm_segment(size_t bytes) {
#if DIST_SIM_HOST_CLOCK
    return malloc(bytes);
#else
    return aicpu_device_malloc(bytes);
#endif
}
inline void dist_free_gm_segment(void *p) {
    if (p == nullptr) return;
#if DIST_SIM_HOST_CLOCK
    free(p);
#else
    aicpu_device_free(p);
#endif
}
inline void dist_publish_gm_segment(void *base, size_t bytes) {
    // Write-back the AICPU's DistGlobal init to HBM so every AICore reads the
    // initialized state (docs §13/§14). DIST_SIM_HOST_CLOCK is 1 for BOTH the sim
    // AICPU AND the onboard AICPU (both are GCC-compiled, not CCEC), so gating the
    // flush on !DIST_SIM_HOST_CLOCK silently dropped it on the onboard AICPU — the
    // AICPU/AICore path is NOT cache-coherent on A5, so the cores then read
    // uninitialized HBM (0xffffffff) and fault (MPU 271) on the garbage pointers.
    // The onboard AICPU is distinguished from sim by SIMPLER_DIST_AICPU_ONLY.
#if !DIST_SIM_HOST_CLOCK || defined(SIMPLER_DIST_AICPU_ONLY)
    cache_flush_range(base, bytes);
#else
    (void)base;
    (void)bytes;
#endif
}
#endif  // !SIMPLER_DIST_AICORE_ONLY

// -----------------------------------------------------------------------------
// Portable LocalContext block-index seam. The per-dispatch LocalContext lives in
// each arch's intrinsic.h and names its fields differently: a2a3 uses
// block_idx/block_num; a5 prefixes them s_block_idx/s_block_num to dodge CCE
// builtin symbol collisions on the AICore target. This shared engine sets them
// via a detection-idiom overload so the same source compiles on both arches
// without touching either intrinsic.h. The s_-prefixed overload is preferred
// when that field exists; otherwise the unprefixed one is selected.
template <typename T>
__aicore__ inline auto dist_set_local_block(T &lc, int32_t idx, int32_t num, int)
    -> decltype((void)lc.s_block_idx) {
    lc.s_block_idx = idx;
    lc.s_block_num = num;
}
template <typename T>
__aicore__ inline auto dist_set_local_block(T &lc, int32_t idx, int32_t num, long)
    -> decltype((void)lc.block_idx) {
    lc.block_idx = idx;
    lc.block_num = num;
}
template <typename T>
__aicore__ inline void dist_set_local_block(T &lc, int32_t idx, int32_t num) {
    dist_set_local_block(lc, idx, num, 0);  // int arg prefers the s_ overload
}

// -----------------------------------------------------------------------------
// global_data segment platform seam (docs §13, 方案B). The distributed runtime's
// shared state (DistGlobal) is NOT a process-global object — it is a segment the
// runtime allocates once and delivers to each AICore through the runtime objects
// the core already reads (like the arena/PTO2Runtime pointers). These are the
// only platform-specific points; everything else is portable base+offset access.
// -----------------------------------------------------------------------------
// Allocate / free the segment. Called ONLY on the AICPU (which has malloc + a GM
// allocator). Sim: host malloc (one shared address space). HW: a GM allocator that
// returns memory addressable + coherent from every AICore.
inline void *pto_gm_alloc(size_t bytes) {
#if DIST_SIM_HOST_CLOCK
    return malloc(bytes);
#else
    (void)bytes;
    return nullptr;  // TODO(a5): return a GM block (all-core addressable + coherent)
#endif
}
[[maybe_unused]] inline void pto_gm_free(void *p) {
#if DIST_SIM_HOST_CLOCK
    free(p);
#else
    (void)p;  // TODO(a5): GM free
#endif
}
// Publish AICPU writes to the segment so AICore reads observe them. Sim: one
// address space + the register()-side release fence suffice (no-op here). HW:
// cache flush/invalidate over [base, base+bytes).
inline void pto_gm_publish(void *base, size_t bytes) {
    (void)base;
    (void)bytes;
#if !DIST_SIM_HOST_CLOCK
    // TODO(a5): dcci/flush [base, base+bytes) so every AICore sees initialized state.
#endif
}
// Current core's id (index into DistGlobal::cores[]) — the ONE per-core datum on
// the HW functional path. HW: a hardware core-id register (no storage). Sim: a
// thread_local set at core_main entry (sim permits thread_local; this branch is
// NOT compiled on HW, so no process/thread global reaches the CCEC AICore path).
#if DIST_SIM_HOST_CLOCK
thread_local int32_t t_dist_core_id = -1;
inline void pto_set_core_id(int32_t id) { t_dist_core_id = id; }
inline int32_t pto_core_id() { return t_dist_core_id; }
#elif defined(SIMPLER_DIST_AICORE_ONLY)
// Onboard AICore (CCEC). The ops callbacks (dist_submit_impl, dist_is_fatal, …)
// receive only a PTO2Runtime* and must recover "which core am I" to index
// gd->cores[]/gd->layout[]. There is no thread_local on the CCEC AICore, but
// each physical core owns its own [[block_local]] storage — the same per-core
// mechanism the platform uses for profiling state (see kernel.cpp). dist_core_main
// stashes its arg-passed LOGICAL block index here at entry via pto_set_core_id
// (the same value kernel.cpp computed and threaded through aicore_execute), and
// every ops callback reads it back through pto_dist_self.
//
// This was previously a `return 0` stub, so ALL workers recovered id 0: every
// core's ops callback aliased gd->cores[0], collapsing the SPMD claim/replay so
// rt_submit_task effectively no-op'd on the 8 non-zero lanes. The run still
// completed (dist_done=9/9) but submitted ZERO tasks → the output tensor stayed
// at its zero init (golden max_diff=47). Docs §17 C14.
[[block_local]] static int32_t s_dist_core_id_hw;
__aicore__ inline void pto_set_core_id(int32_t id) { s_dist_core_id_hw = id; }
__aicore__ inline int32_t pto_core_id() { return s_dist_core_id_hw; }
#else
// Onboard AICPU (g++): ops callbacks are compiled here (g_dist_ops references
// them) but are NOT the AICPU functional path — the AICPU only registers the
// engine, it never replays orchestration — so a constant 0 is harmless. g++ does
// not understand the [[block_local]] attribute, hence the separate branch.
inline void pto_set_core_id(int32_t /*id*/) {}
inline int32_t pto_core_id() { return 0; }
#endif

// C14 DIAG: per-core count of dist_submit_impl ENTRIES (before any early return).
// Definitively answers "is submit_task actually invoked on the AICore?" without
// relying on local_index (which increments only past the early-return guards).
#if !DIST_SIM_HOST_CLOCK && defined(SIMPLER_DIST_AICORE_ONLY)
[[block_local]] static int32_t s_dbg_submit_entries;
[[block_local]] static uint64_t s_dbg_submit_rt;  // the rt dist_submit_impl actually received
#define DIST_DBG_SUBMIT_TICK(rtv) do { ++s_dbg_submit_entries; s_dbg_submit_rt = (uint64_t)(uintptr_t)(rtv); } while (0)
#define DIST_DBG_SUBMIT_RESET() do { s_dbg_submit_entries = 0; s_dbg_submit_rt = 0; } while (0)
#define DIST_DBG_SUBMIT_ENTRIES() (s_dbg_submit_entries)
#define DIST_DBG_SUBMIT_RT() (s_dbg_submit_rt)
#else
#define DIST_DBG_SUBMIT_TICK(rtv) ((void)0)
#define DIST_DBG_SUBMIT_RESET() ((void)0)
#define DIST_DBG_SUBMIT_ENTRIES() (0)
#define DIST_DBG_SUBMIT_RT() ((uint64_t)0)
#endif

// -----------------------------------------------------------------------------
// Cross-core cache-coherence seam (docs §14). A5 AICores have NO hardware cache
// coherence: a word one core writes is invisible to another until the writer
// FLUSHES it to HBM (`dcci … CACHELINE_OUT`) and the reader INVALIDATES its stale
// copy (`dcci …`) — exactly what a5's aicore_executor.cpp does by hand. So on HW
// `std::atomic`'s memory_order is necessary-but-NOT-sufficient: it orders local
// accesses but never publishes them across cores. `Coherent<T>` is a drop-in for
// `std::atomic<T>` that pairs every shared READ with an invalidate and every
// shared WRITE with a flush; on sim the hooks compile to nothing, so each access
// is exactly the original `std::atomic` op (bit-identical, zero cost).
#if !DIST_SIM_HOST_CLOCK
// AICore (CCEC, onboard). A5 AICores have no hardware cache coherence, so the
// shared segment is CACHEABLE HBM (host rtMalloc; the uncacheable double-page
// alias is not AICore-MPU-mapped) and every cross-core access must pair with a
// dcci over the cache line(s) covering [p, p+n): INVALIDATE (dcci … default)
// drops this core's stale copy so the next read pulls the writer's HBM value;
// FLUSH (dcci … CACHELINE_OUT) cleans this core's dirty line back to HBM. This
// is the same primitive a5's aicore_executor.cpp / tensormap_and_ringbuffer use
// by hand (docs §14). dcci wants a __gm__ pointer; the shared segment is GM.
constexpr uintptr_t kDcciLine = 64;
__aicore__ inline void pto_dcci_inval(__gm__ const volatile void *p, size_t n) {
    uintptr_t a = reinterpret_cast<uintptr_t>(p) & ~(kDcciLine - 1);
    const uintptr_t end = reinterpret_cast<uintptr_t>(p) + n;
    for (; a < end; a += kDcciLine) dcci(reinterpret_cast<__gm__ void *>(a), SINGLE_CACHE_LINE);
}
__aicore__ inline void pto_dcci_flush(__gm__ const volatile void *p, size_t n) {
    uintptr_t a = reinterpret_cast<uintptr_t>(p) & ~(kDcciLine - 1);
    const uintptr_t end = reinterpret_cast<uintptr_t>(p) + n;
    for (; a < end; a += kDcciLine) dcci(reinterpret_cast<__gm__ void *>(a), SINGLE_CACHE_LINE, CACHELINE_OUT);
}
#elif defined(SIMPLER_DIST_AICPU_ONLY)
// AICPU (aarch64, onboard). Same shared HBM as the AICore, reached through the
// AICPU data cache: DC CIVAC (invalidate) before reads / DC CVAC (clean) after
// writes keep the AICPU coherent with the AICore's dcci traffic.
inline void pto_dcci_inval(const volatile void *p, size_t n) { cache_invalidate_range(const_cast<const void *>(p), n); }
inline void pto_dcci_flush(const volatile void *p, size_t n) { cache_flush_range(const_cast<const void *>(p), n); }
#else
// Pure sim (host g++): one shared address space, std::atomic already coherent.
inline void pto_dcci_inval(const volatile void *, size_t) {}
inline void pto_dcci_flush(const volatile void *, size_t) {}
#endif
// Barrier used at explicit publish points (was `std::atomic_thread_fence`). On
// sim/host it maps to std::atomic_thread_fence; on AICore the cross-core shared
// path uses cacheable GM + explicit dcci (coherent_load/store) plus device
// atomics (atomicCAS/atomicAdd/atomicMax), whose ordering already provides
// release/acquire semantics, so the fence compiles to a no-op (TODO a5: add a
// pipe/store barrier if profiling shows a reordering hazard).
#if DIST_SIM_HOST_CLOCK
inline void pto_shared_fence(std::memory_order o) { std::atomic_thread_fence(o); }
#else
__aicore__ inline void pto_shared_fence(std::memory_order) {}
#endif

// AICore CCEC atomic CAS helper (defined after Coherent below; forward-declared
// here so Coherent's CCEC methods can call it). Only present when !DIST_SIM_HOST_CLOCK.
#if !DIST_SIM_HOST_CLOCK
template <class T>
__aicore__ T dist_atomic_cas(__gm__ T *p, T e, T d);
#endif

// Drop-in for std::atomic<T> on the cross-core shared path. Same size/alignment
// (single T-sized member) so DistGlobal's layout/`sizeof` is unchanged.
//
// IMPORTANT: Coherent<T> is a POD — it has NO member functions. CCEC forbids
// calling a method on a __gm__-resident object (the implicit `this` is generic
// and cannot bind to a __gm__ object), and Coherent instances live in
// DistGlobal/DistCore (GM). So all access goes through the coherent_* free
// functions below, which take a __gm__ Coherent<T>* (== Coherent<T>* on sim,
// where __gm__ is empty). The same free-function call sites therefore work on
// both builds.
//
//  * sim/host (DIST_SIM_HOST_CLOCK==1): wraps std::atomic<T>; the dcci hooks are
//    no-ops so each access is exactly the original std::atomic op (bit-identical).
//  * AICore/CCEC (DIST_SIM_HOST_CLOCK==0): the segment is CACHEABLE HBM (host
//    rtMalloc; this device has no uncacheable double-page alias — docs §16.2
//    probe). So plain load/store hit only this core's cache and are NOT
//    cross-core coherent: coherent_load INVALIDATES the line first, coherent_store
//    FLUSHES it after (dcci, §14). RMW atomicity uses the A5 hardware GM atomics
//    (atomicAdd / atomicMax / atomicCAS on __gm__) which are memory-level and
//    cross-core serialized (docs §11.1.1; CANN kernel_operator_atomic_impl.h) —
//    they read/write the true HBM value, so no surrounding dcci is needed on the
//    RMW itself. std::memory_order args are accepted but ignored (dcci + device
//    atomics provide the ordering).
template <class T>
struct Coherent {
#if DIST_SIM_HOST_CLOCK
    std::atomic<T> a;
#else
    T a;
#endif
};

// coherent_load / coherent_store / coherent_fetch_* / coherent_cas_* — free-
// function accessors for Coherent<T>. See the note above for why these are free
// functions instead of methods. Default memory_order args keep call sites terse.
//
// The value/desired params use a separate deduced type V (converted to T via
// static_cast), mirroring std::atomic's implicit-conversion behavior so callers
// can pass int literals (e.g. coherent_store(&flag, 0, mo)) to a Coherent<uint8_t>
// without a template-deduction conflict. The `expected` (e) param of CAS stays T&
// — it is in/out and must match exactly, same as std::atomic::compare_exchange.
template <class T>
__aicore__ T coherent_load(__gm__ const Coherent<T> *c, std::memory_order o = std::memory_order_seq_cst) {
#if DIST_SIM_HOST_CLOCK
    pto_dcci_inval(&c->a, sizeof(c->a));
    return c->a.load(o);
#else
    (void)o;
    // A5 onboard: the segment is CACHEABLE HBM (no uncacheable alias — the double
    // page table is unavailable on this device, docs §16.2 probe). A plain load
    // returns this core's STALE cached copy and never observes another core's
    // store/atomic. Invalidate the covering cache line(s) first so the load pulls
    // the writer's HBM value. This is the same dcci the sim seam uses; onboard it
    // is now unconditionally required (docs §11.1.1 / §16.2 C1-revised).
    pto_dcci_inval(&c->a, sizeof(c->a));
    return c->a;
#endif
}
template <class T, class V>
__aicore__ void coherent_store(__gm__ Coherent<T> *c, V v, std::memory_order o = std::memory_order_seq_cst) {
#if DIST_SIM_HOST_CLOCK
    c->a.store(static_cast<T>(v), o);
    pto_dcci_flush(&c->a, sizeof(c->a));
#else
    (void)o;
    // A5 onboard: cacheable HBM — a plain store lands only in this core's cache.
    // Flush (dcci … CACHELINE_OUT) so the covering line reaches HBM and other
    // cores (after their coherent_load invalidate) observe it.
    // NOTE(C4): flags[] is a byte-granular ring — many flags share one 64B line,
    // so this flush writes back neighbors too. Concurrent cross-core flag stores
    // on the same line can clobber (docs §16.2 item 4 / §17 C4); the real fix is
    // per-cacheline flag alignment / per-core shards, tracked separately.
    c->a = static_cast<T>(v);
    pto_dcci_flush(&c->a, sizeof(c->a));
#endif
}
template <class T, class V>
__aicore__ T coherent_fetch_add(__gm__ Coherent<T> *c, V v, std::memory_order o = std::memory_order_seq_cst) {
#if DIST_SIM_HOST_CLOCK
    pto_dcci_inval(&c->a, sizeof(c->a));
    const T r = c->a.fetch_add(static_cast<T>(v), o);
    pto_dcci_flush(&c->a, sizeof(c->a));
    return r;
#else
    (void)o;
    return atomicAdd(&c->a, static_cast<T>(v));
#endif
}
template <class T, class V>
__aicore__ T coherent_fetch_sub(__gm__ Coherent<T> *c, V v, std::memory_order o = std::memory_order_seq_cst) {
#if DIST_SIM_HOST_CLOCK
    pto_dcci_inval(&c->a, sizeof(c->a));
    const T r = c->a.fetch_sub(static_cast<T>(v), o);
    pto_dcci_flush(&c->a, sizeof(c->a));
    return r;
#else
    (void)o;
    return atomicAdd(&c->a, static_cast<T>(-static_cast<T>(v)));  // no AtomicSub builtin
#endif
}
template <class T, class V>
__aicore__ T coherent_fetch_xor(__gm__ Coherent<T> *c, V v, std::memory_order o = std::memory_order_seq_cst) {
#if DIST_SIM_HOST_CLOCK
    pto_dcci_inval(&c->a, sizeof(c->a));
    const T r = c->a.fetch_xor(static_cast<T>(v), o);
    pto_dcci_flush(&c->a, sizeof(c->a));
    return r;
#else
    (void)o;
    const T w = static_cast<T>(v);
    T cur = c->a;
    while (true) {  // no AtomicXor builtin: CAS-loop
        const T nxt = static_cast<T>(cur ^ w);
        const T old = dist_atomic_cas(&c->a, cur, nxt);
        if (old == cur) return cur;
        cur = old;
    }
#endif
}
template <class T, class V>
__aicore__ bool coherent_cas_weak(__gm__ Coherent<T> *c, T &e, V d, std::memory_order s, std::memory_order f) {
#if DIST_SIM_HOST_CLOCK
    pto_dcci_inval(&c->a, sizeof(c->a));
    const bool ok = c->a.compare_exchange_weak(e, static_cast<T>(d), s, f);
    pto_dcci_flush(&c->a, sizeof(c->a));
    return ok;
#else
    (void)s;
    (void)f;
    const T dt = static_cast<T>(d);
    const T old = dist_atomic_cas(&c->a, e, dt);
    if (old == e) return true;
    e = old;
    return false;
#endif
}
template <class T, class V>
__aicore__ bool coherent_cas_weak(__gm__ Coherent<T> *c, T &e, V d, std::memory_order o = std::memory_order_seq_cst) {
    return coherent_cas_weak(c, e, d, o, o);
}
template <class T, class V>
__aicore__ bool coherent_cas_strong(__gm__ Coherent<T> *c, T &e, V d, std::memory_order s, std::memory_order f) {
    return coherent_cas_weak(c, e, d, s, f);  // atomicCAS is already strong
}
template <class T, class V>
__aicore__ bool coherent_cas_strong(__gm__ Coherent<T> *c, T &e, V d, std::memory_order o = std::memory_order_seq_cst) {
    return coherent_cas_strong(c, e, d, o, o);
}

// CCEC atomic CAS helper: atomicCAS only supports uint32_t/uint64_t, so route
// 32-bit T through uint32_t (bit-equivalent for int32_t/uint32_t). 64-bit T
// goes direct. 8/16-bit T have no hardware CAS (the engine never CASes them —
// Coherent<uint8_t> flags are store/load only), so that path is intentionally
// not provided here; instantiating it yields a clear no-matching call error.
#if !DIST_SIM_HOST_CLOCK
template <class T>
__aicore__ T dist_atomic_cas(__gm__ T *p, T e, T d) {
    if constexpr (sizeof(T) == 8) {
        return static_cast<T>(atomicCAS(p, static_cast<uint64_t>(e), static_cast<uint64_t>(d)));
    } else {
        return static_cast<T>(atomicCAS(
            reinterpret_cast<__gm__ uint32_t *>(p), static_cast<uint32_t>(e), static_cast<uint32_t>(d)
        ));
    }
}

// CCEC atomic add helper: the GCC __atomic_add_fetch builtin lowers to an
// AtomicLoadAdd that the A5/A2A3 backends cannot select for addrspace-1 (GM)
// memory. Use the CCEC atomicAdd builtin instead (works on uncacheable GM on
// A5). atomicAdd only supports uint32_t/uint64_t — route by size like CAS.
template <class T>
__aicore__ T dist_atomic_add(__gm__ T *p, T v) {
    if constexpr (sizeof(T) == 8) {
        return static_cast<T>(atomicAdd(p, static_cast<uint64_t>(v)));
    } else {
        return static_cast<T>(atomicAdd(reinterpret_cast<__gm__ uint32_t *>(p), static_cast<uint32_t>(v)));
    }
}

// CCEC atomic fetch_max helper (A5 dav_3510). atomicMax natively supports
// int32_t/uint32_t/int64_t/uint64_t/float (kernel_operator_atomic_impl.h), so
// signed T goes direct — no uint reinterpret needed (unlike CAS). Returns the
// value BEFORE the max; the claim wins iff (old < N). Memory-level, cross-core.
template <class T>
__aicore__ T dist_atomic_max(__gm__ T *p, T v) {
    return static_cast<T>(atomicMax(p, v));
}
#endif

// Publish a task's completion flag (set to 1) with cross-core coherence but
// WITHOUT the cacheline write-back that plain coherent_store performs (docs §17
// C4/C15). The flag ring is byte-/word-dense, so many task flags share one 64B
// line; a dcci FLUSH of one flag writes back its (possibly stale) neighbours and
// can clobber another core's concurrent completion → the consumer of the
// clobbered task hangs forever waiting on flags[id]==1 (observed: t1 set, the
// structurally-identical t2 lost). Fix: set the flag via the A5 memory-level
// atomicMax(…,1), which writes the true HBM word directly and never writes back
// the shared line, so adjacent flag sets can't clobber each other. Idempotent
// (max(*,1)==1). Readers still use coherent_load's dcci-invalidate to observe it.
// sim/host: shared address space — a released store is already visible.
__aicore__ inline void dist_set_flag(__gm__ Coherent<int32_t> *c) {
#if DIST_SIM_HOST_CLOCK
    c->a.store(1, std::memory_order_release);
    pto_dcci_flush(&c->a, sizeof(c->a));  // no-op on sim; keeps parity with coherent_store
#else
    (void)dist_atomic_max(&c->a, 1);
#endif
}

// -----------------------------------------------------------------------------
// Tunables. The completion-flag ring is sized to hold an entire run without
// wrap (>= total tasks); the GM output heap is a BOUNDED RING reclaimed by the
// completion frontier (M4, §9.5/§11.4) rather than a run-sized bump.
// -----------------------------------------------------------------------------
// Kept deliberately SMALL: the out-of-order window is num_cores * kPrivateSlots,
// and this also caps how far a single core can run ahead of "ready-to-execute".
// A large ring lets one fast core greedily claim a long run of consecutive tasks
// and serialize them while other cores starve (load imbalance, docs §6.1). OoO
// capacity should come from the core-count dimension, not a deep per-core ring.
constexpr int32_t kPrivateSlots = 4;  // PRIVATE_TASK_SLOT_NUM (back-pressure cap)
// Ring slots a core reserves for draining block.won deposits addressed to its
// lane. Self-claimed tasks (consumers / single-core / own anchor subtask) may
// only occupy kPrivateSlots - kWonReserve slots, so a follower can ALWAYS pull
// and run an (immediately-ready) deposit even when its ring is otherwise full of
// not-yet-ready consumers — breaking the consumer<->deposit priority inversion.
constexpr int32_t kWonReserve = 2;
constexpr int32_t kMaxFanin = 16;        // max distinct producers a task waits on
constexpr int32_t kOutPoolSlots = 1024;  // per-core ring of materialized output Tensors
constexpr int32_t kFlagCap = 1 << 16;    // global completion-flag ring (>= total tasks)

// M4 GM-heap reclamation (§9.5/§11.4).
//   kHeapRingDefault — bounded physical heap ring (env PTO_DIST_HEAP_MB overrides,
//     in MiB). The deterministic virtual bump is unbounded; physical address is
//     (virtual_offset mod ring). A region is reused only after its previous
//     occupant's task id <= R (the reclaim frontier), enforced by back-pressure.
//   kHDefault — dependency-span bound H (env PTO_DIST_H overrides): every consumer
//     of task N has id <= N + H. R = F - H. Must be >= the graph's true heap span
//     or a producer region could be recycled while a late consumer still reads it
//     (run-time-checked → fatal "heap span exceeded").
constexpr size_t kHeapRingDefault = 64ull << 20;
constexpr int32_t kHDefault = 64;

// -----------------------------------------------------------------------------
// Per-core producer map (the "full per-core duplicate TensorMap").
//
// A faithful, compact stand-in for PTO2TensorMap: keyed by GM byte range, it
// records the most recent producer task id of each written region. INPUT/INOUT
// fan-in resolves to the producer(s) whose region overlaps. Exact-region writes
// (e.g. an INOUT accumulation chain) replace in place; new regions append.
// Every core builds an identical map by replaying the same submit stream.
// -----------------------------------------------------------------------------
// A ring slot: a written region keyed by GM byte range [lo, hi) under buffer base
// `buf_addr`, tagged with its producer task id. Compact (32B), NO chain links —
// bucket membership is positional (index / kBucketCapMax) and ordering is the
// append order, which is producer-monotonic (see KEY INVARIANT below).
struct MapEntry {
    uint64_t buf_addr;  // Tensor.buffer.addr (GM buffer base, bytes) — hash key
    uint64_t lo;        // byte offset of view origin within buffer
    uint64_t hi;        // byte offset one-past the view extent
    int32_t producer;   // task id that wrote this region
    int32_t pad_;       // keep 32B slot size
};

// Hash buckets (power of 2). Hashing by buffer BASE address groups every
// sub-region of one buffer into one chain; overlap is then tested per entry.
constexpr int32_t kRingBuckets = 128;     // number of hash buckets (power of 2)
constexpr int32_t kRingBucketShift = 7;   // log2(kRingBuckets)
// Per-bucket ring depth (compile-time max, power of 2 so `% cap` -> mask). Must
// cover the live H-window of the HOTTEST bucket: bgemm writes hundreds of disjoint
// tiles of ONE output buffer, which all hash to one bucket, so this is sized
// generously. Overflow is a deterministic FATAL (docs §12.7.2.2) — raise cap
// (Phase 2: PTO_DIST_TENSORMAP_RING_CAP) or lower H, never silently drop.
// Memory per core = kRingBuckets * kBucketCapMax * sizeof(MapEntry) = 2 MiB.
constexpr int32_t kBucketCapMax = 512;

// Active per-bucket ring depth, chosen once per run in dist_engine_register()
// (docs §12.7.2): `auto` derives it from the dependency-span H, an explicit
// PTO_DIST_TENSORMAP_RING_CAP=N overrides. Always a power of 2 in [1, kBucketCapMax]
// so lookup/insert use a mask, not a modulo. reset() copies it into the map.
// Lives in the shared segment as gd->ring_cap (docs §13); see DistGlobal.

// TensorMap operation counters (env PTO_DIST_OVERHEAD). Count the actual map work
// that differs between private (per-core replica) and shared (single global ring)
// modes. Deterministic + platform-independent, so they quantify the private-vs-
// shared orchestration overhead even where the host wall clock is unusable (sim
// device_wall_us=0; macOS spin/oversubscription inflates makespan identically for
// both modes, masking the logic difference). Gated by g_overhead_on so a normal
// run pays nothing. Reset per run in register.
//   inserts : region writes into the map — private ≈ C×D, shared ≈ D (the headline
//             SPMD-floor difference: C = cores, D = distinct output regions).
//   lookups : fan-in resolutions (identical count in both modes).
//   scans   : ring slots examined across all lookups (per-lookup cost; shared also
//             pays an atomic seq acquire on each scanned slot — private does not).
// These are a SIM-ONLY diagnostic (host wall-clock overhead study, docs §12.8.1),
// so they live only under DIST_SIM_HOST_CLOCK — on CCEC/HW they compile away
// entirely (no process-global symbol; TMOP_COUNT becomes a no-op) (docs §13.4B).
#if DIST_SIM_HOST_CLOCK
bool g_overhead_on = false;
std::atomic<uint64_t> g_tm_inserts{0};
std::atomic<uint64_t> g_tm_lookups{0};
std::atomic<uint64_t> g_tm_scans{0};
#define TMOP_COUNT(counter) \
    do {                    \
        if (g_overhead_on) (counter).fetch_add(1, std::memory_order_relaxed); \
    } while (0)
#else
#define TMOP_COUNT(counter) do {} while (0)
#endif

// Round v up to the next power of 2, clamped to [1, kBucketCapMax].
inline int32_t round_pow2_cap(int32_t v) {
    if (v < 1) v = 1;
    if (v > kBucketCapMax) return kBucketCapMax;
    int32_t p = 1;
    while (p < v) p <<= 1;
    return p;
}

// Per-core producer map — RING-PER-BUCKET (docs §12, unified private/shared;
// this is the "private" form: each core owns a full replica built by replaying
// the same submit stream, so every replica is identical, only progress differs).
//
// Structure: hash by buffer BASE into kRingBuckets buckets; each bucket is a
// BOUNDED RING of `cap` contiguous slots with head/tail cursors.
//   * insert  — reclaim retired head slots, then append at tail (tail++).
//   * lookup  — scan tail-1 .. head; the FIRST overlap is the MAX producer.
//   * retire  — O(1): raise alive_floor (= N - H). Physical reclaim is lazy
//               (insert advances head past producer < alive_floor).
//
// KEY INVARIANT (what makes the ring correct & simple): every core replays
// submits in task-id order (local_index++ in dist_submit_impl/dist_alloc_tensors)
// and insert always uses the current task id N. Hence entries appended to any one
// bucket are MONOTONICALLY NON-DECREASING in producer id. So retired entries are
// always at the head (reclaim = head++), and the newest overlapping producer is
// the first hit when scanning from the tail. Verified by a differential test
// against the former linked-list semantics over randomized SPMD-ordered streams.
//
// WHY ring vs the former bucket-chain + free-list + per-task chains (docs §12.3):
// identical O(N*H) via the same H-window reclaim and identical lookup semantics
// (max overlapping producer >= alive_floor), but contiguous storage (better
// locality, no pointer chase), no per-entry link fields, and reclaim is a cursor
// bump. `alive_floor` is N-derived (deterministic, identical on every core), so
// replicas evolve in lockstep — determinism preserved.
struct DistTensorMap {
    static constexpr int32_t kStride = kBucketCapMax;  // per-bucket slot stride
    MapEntry slots[kRingBuckets * kStride];
    uint64_t head[kRingBuckets];  // reclaim cursor (oldest live slot, monotonic)
    uint64_t tail[kRingBuckets];  // append cursor (next free slot, monotonic)
    int32_t cap;                  // active per-bucket depth (power of 2, <= kBucketCapMax)
    int32_t cap_mask;             // cap - 1 (fast `% cap`)
    int32_t alive_floor;          // producer < alive_floor == retired

    static uint32_t hash(uint64_t addr) {
        addr *= 0x9E3779B97F4A7C15ULL;  // golden-ratio multiplicative mix
        return static_cast<uint32_t>(addr >> (64 - kRingBucketShift));
    }

    template <class TRef>
    __aicore__ static void byte_range(const TRef &t, uint64_t &addr, uint64_t &lo, uint64_t &hi) {
        const uint64_t esz = get_element_size(t.dtype);
        addr = t.buffer.addr;
        lo = t.start_offset * esz;
        hi = (t.start_offset + tensor_extent_elem(t)) * esz;
    }
};

// DistTensorMap accessors are FREE FUNCTIONS (not member methods): the struct is
// GM-resident (a member of DistCore), and CCEC forbids calling a method on a
// __gm__ object. Same pattern as SharedTensorMap above.

// cap is passed in (run-configured: auto from H, or PTO_DIST_TENSORMAP_RING_CAP)
// rather than read from the segment here — this struct is defined before
// DistGlobal, so it must not depend on the DistGlobal segment (docs §13).
__aicore__ void dist_tm_reset(__gm__ DistTensorMap *m, int32_t cap_) {
    m->cap = cap_;
    m->cap_mask = cap_ - 1;
    m->alive_floor = 0;
    for (int32_t b = 0; b < kRingBuckets; b++) {
        m->head[b] = 0;
        m->tail[b] = 0;
    }
}

// Advance a bucket's head past retired entries (producer < alive_floor).
// Entries are producer-ascending, so retired ones are always at the head.
__aicore__ void dist_tm_reclaim_bucket(__gm__ DistTensorMap *m, int32_t b) {
    while (m->head[b] < m->tail[b] &&
           m->slots[b * DistTensorMap::kStride + static_cast<int32_t>(m->head[b] & m->cap_mask)].producer < m->alive_floor)
        m->head[b]++;
}

// Retire is O(1): raise the floor. Physical slots reclaimed lazily by insert.
// Same contract as before: new_floor = N - H; lookups skip producer<floor.
__aicore__ void dist_tm_advance_retire(__gm__ DistTensorMap *m, int32_t N, int32_t H) {
    const int32_t new_floor = N - H;
    if (new_floor > m->alive_floor) m->alive_floor = new_floor;
}

// Append a fresh entry for `producer`'s write of `t`'s region at the ring tail.
template <class TRef>
__aicore__ void dist_tm_insert(__gm__ DistTensorMap *m, const TRef &t, int32_t producer) {
    TMOP_COUNT(g_tm_inserts);
    uint64_t addr, lo, hi;
    DistTensorMap::byte_range(t, addr, lo, hi);
    const int32_t b = static_cast<int32_t>(DistTensorMap::hash(addr));
    dist_tm_reclaim_bucket(m, b);
    if (m->tail[b] - m->head[b] >= static_cast<uint64_t>(m->cap)) {
        // Ring full within the live window — a deterministic config error
        // (docs §12.7.2.2). NEVER silently overwrite / drop a live producer.
#if DIST_SIM_HOST_CLOCK
        fprintf(
            stderr,
            "[dist_engine] FATAL TensorMap ring overflow: bucket=%d cap=%d live=%llu — the live "
            "dependency window for this bucket exceeds the ring depth. Raise PTO_DIST_TENSORMAP_RING_CAP "
            "(<= %d) or lower PTO_DIST_H.\n",
            b, m->cap, static_cast<unsigned long long>(m->tail[b] - m->head[b]), kBucketCapMax
        );
        fflush(stderr);
        always_assert(false);
#endif
        return;
    }
    __gm__ MapEntry *e = &m->slots[b * DistTensorMap::kStride + static_cast<int32_t>(m->tail[b] & m->cap_mask)];
    e->buf_addr = addr;
    e->lo = lo;
    e->hi = hi;
    e->producer = producer;
    m->tail[b]++;
}

// Most-recent producer whose region overlaps `t`, or -1 if none. Scans the
// bucket ring newest -> oldest; producer-ascending order means the first
// overlap encountered is the MAX overlapping producer.
template <class TRef>
__aicore__ int32_t dist_tm_lookup(__gm__ const DistTensorMap *m, const TRef &t) {
    TMOP_COUNT(g_tm_lookups);
    uint64_t addr, lo, hi;
    DistTensorMap::byte_range(t, addr, lo, hi);
    const int32_t b = static_cast<int32_t>(DistTensorMap::hash(addr));
    for (uint64_t k = m->tail[b]; k > m->head[b]; k--) {
        TMOP_COUNT(g_tm_scans);
        __gm__ const MapEntry *e = &m->slots[b * DistTensorMap::kStride + static_cast<int32_t>((k - 1) & m->cap_mask)];
        if (e->producer < m->alive_floor) continue;
        if (e->buf_addr == addr && lo < e->hi && e->lo < hi) return e->producer;
    }
    return -1;
}

// -----------------------------------------------------------------------------
// SHARED TensorMap (docs §12.4-§12.7) — a SINGLE global ring-per-bucket shared by
// every core, selected by PTO_DIST_TENSORMAP_MODE=shared. Same hash/byte_range/
// overlap semantics as the private map; the only differences (docs §12.3.2) are:
//   * one copy instead of one-per-core;
//   * only ONE core appends each task (vs every core in private);
//   * concurrent readers, so slots carry a `seq` guard and cursors are atomic;
//   * reclaim by a global floor (min core progress − H − 1) instead of per-core N−H.
//
// CORRECTNESS = PRIVATE (bit-identical results, the acceptance goal). Appends are
// SERIALIZED IN TASK-ID ORDER by the global sequencer gd->tm_insert_next (see
// tm_shared_claim_append): task N is appended exactly once, only after 0..N-1 are
// appended. Because every core replays every task id in order and only advances
// past task K after K is appended (the sequencer blocks it), by the time ANY core
// resolves task N's fan-in the ring already holds every task < N — exactly what a
// private replica holds at that point. lookup() then applies BOTH the temporal
// filter (producer < N, §12.6) AND the retire floor (producer >= N−H, mirroring
// the private map's alive_floor), so it returns the identical producer the private
// replica would. Single serialized appender ⇒ no MPSC `reserve` needed; head/tail
// are plain atomics and per-slot `seq` (defense against reader/append slot reuse
// under host thread preemption, docs §12.7.1) guards concurrent readers. lookup()
// takes exactly ONE acquire (the tail snapshot) and reads all slots below it with
// RELAXED loads: because appends are serialized in id order and tail is monotonic,
// observing tail via acquire already makes every published slot < tail visible, so
// the per-slot seq is only an ABA guard (relaxed) — this amortizes the memory-order
// cost to one acquire per lookup instead of one per scanned slot (docs §12.8.1).
// -----------------------------------------------------------------------------
struct SharedRingSlot {
    uint64_t buf_addr;
    uint64_t lo;
    uint64_t hi;
    int32_t producer;
    int32_t pad_;
    Coherent<uint64_t> seq;  // absolute append index k when published; kSeqEmpty otherwise
};

struct SharedTensorMap {
    static constexpr int32_t kStride = kBucketCapMax;
    static constexpr uint64_t kSeqEmpty = ~0ull;
    SharedRingSlot slots[kRingBuckets * kStride];
    Coherent<uint64_t> head[kRingBuckets];  // reclaim cursor (oldest live slot)
    Coherent<uint64_t> tail[kRingBuckets];  // publish cursor (next free slot)
    int32_t cap;
    int32_t cap_mask;
};

// SharedTensorMap accessors are FREE FUNCTIONS (not member methods): the struct
// is GM-resident (a member of DistGlobal), and CCEC forbids calling a method on
// a __gm__ object (the implicit `this` is generic and cannot bind to __gm__).
// This mirrors the tensormap_and_ringbuffer reference, whose AICore side accesses
// GM structs only via __gm__ pointers + __aicore__ free functions.

// cap passed in (see DistTensorMap::reset): this struct precedes DistGlobal.
__aicore__ void shared_tm_reset(__gm__ SharedTensorMap *m, int32_t cap_) {
    m->cap = cap_;
    m->cap_mask = cap_ - 1;
    for (int32_t b = 0; b < kRingBuckets; b++) {
        coherent_store(&m->head[b], 0, std::memory_order_relaxed);
        coherent_store(&m->tail[b], 0, std::memory_order_relaxed);
    }
    for (int32_t i = 0; i < kRingBuckets * SharedTensorMap::kStride; i++)
        coherent_store(&m->slots[i].seq, SharedTensorMap::kSeqEmpty, std::memory_order_relaxed);
}

// Single-appender (serialized by gd->tm_insert_next). Reclaim retired head
// slots (producer <= reclaim_floor), then publish one entry at the tail.
template <class TRef>
__aicore__ void shared_tm_append(__gm__ SharedTensorMap *m, const TRef &t, int32_t producer, int32_t reclaim_floor) {
    TMOP_COUNT(g_tm_inserts);
    uint64_t addr, lo, hi;
    DistTensorMap::byte_range(t, addr, lo, hi);
    const int32_t b = static_cast<int32_t>(DistTensorMap::hash(addr));
    uint64_t h = coherent_load(&m->head[b], std::memory_order_relaxed);
    const uint64_t tl = coherent_load(&m->tail[b], std::memory_order_relaxed);
    while (h < tl &&
           m->slots[b * SharedTensorMap::kStride + static_cast<int32_t>(h & m->cap_mask)].producer <= reclaim_floor)
        h++;
    coherent_store(&m->head[b], h, std::memory_order_release);
    if (tl - h >= static_cast<uint64_t>(m->cap)) {
        // Live dependency window for this bucket exceeds ring depth — a
        // deterministic config error (docs §12.7.2). NEVER silently drop.
#if DIST_SIM_HOST_CLOCK
        fprintf(
            stderr,
            "[dist_engine] FATAL shared TensorMap ring overflow: bucket=%d cap=%d live=%llu — the "
            "live dependency window exceeds the ring depth. Raise PTO_DIST_TENSORMAP_RING_CAP (<= %d) "
            "or lower PTO_DIST_H.\n",
            b, m->cap, static_cast<unsigned long long>(tl - h), kBucketCapMax
        );
        fflush(stderr);
        always_assert(false);
#endif
        return;
    }
    const uint64_t k = tl;
    __gm__ SharedRingSlot *s = &m->slots[b * SharedTensorMap::kStride + static_cast<int32_t>(k & m->cap_mask)];
    s->buf_addr = addr;
    s->lo = lo;
    s->hi = hi;
    s->producer = producer;
    coherent_store(&s->seq, k, std::memory_order_release);       // publish slot fields
    coherent_store(&m->tail[b], k + 1, std::memory_order_release);  // publish new tail
}

// Most-recent producer overlapping `t` with producer in [floor, N), or -1.
// floor == N - H mirrors the private map's alive_floor; the < N temporal
// filter (docs §12.6) skips a fast core's future producers. Together they make
// the accepted set identical to the private replica's => identical results.
template <class TRef>
__aicore__ int32_t shared_tm_lookup(__gm__ const SharedTensorMap *m, const TRef &t, int32_t N, int32_t floor) {
    TMOP_COUNT(g_tm_lookups);
    uint64_t addr, lo, hi;
    DistTensorMap::byte_range(t, addr, lo, hi);
    const int32_t b = static_cast<int32_t>(DistTensorMap::hash(addr));
    // ONE acquire per lookup (docs §12.7.1). Appends are serialized in id order
    // (the tm_insert_next release/acquire chain links append(k)→append(k+1) even
    // across cores) and tail is monotonic, so this single acquire load of tail
    // synchronizes-with EVERY append < tl: all their slot fields + seq stores are
    // already visible. Per-slot reads below therefore use RELAXED — the seq
    // compare stays only as an ABA guard against a slot being reclaimed+reused
    // out from under the scan (head advancing concurrently). This drops the cost
    // from "one acquire per scanned slot" to "one acquire per lookup", bringing
    // shared's per-slot scan close to private's plain local read (§12.8.1).
    const uint64_t tl = coherent_load(&m->tail[b], std::memory_order_acquire);
    const uint64_t h = coherent_load(&m->head[b], std::memory_order_relaxed);
    for (uint64_t k = tl; k > h; k--) {
        TMOP_COUNT(g_tm_scans);
        const uint64_t idx = k - 1;
        __gm__ const SharedRingSlot *s =
            &m->slots[b * SharedTensorMap::kStride + static_cast<int32_t>(idx & m->cap_mask)];
        if (coherent_load(&s->seq, std::memory_order_relaxed) != idx) continue;  // empty / reclaimed+reused slot
        const int32_t p = s->producer;
        if (p >= N) continue;      // temporal filter: skip future producers (§12.6)
        if (p < floor) continue;   // retire floor: identical to private alive_floor (N-H)
        if (s->buf_addr == addr && lo < s->hi && s->lo < hi) return p;
    }
    return -1;
}

// -----------------------------------------------------------------------------
// A private-ring slot: a fully materialized, self-contained task this core owns
// and will execute itself. Holds its own copy of the argument Tensors so it can
// be executed at any later point (deferred past further orchestration).
// -----------------------------------------------------------------------------
// One traced span on a core's timeline, recorded only when swimlane tracing is
// on. `phase` distinguishes the orchestration stage so the exported lane shows
// not just kernel execution but also the work between kernels (alloc, claim/
// build, deposit drains). Laid out in the Chrome trace by physical block (pid)
// and lane (tid).
#if DIST_TRACE_ENABLED
enum class TracePhase : int32_t {
    Kernel = 0,    // incore kernel execution (or busy-wait replay)
    Alloc = 1,     // dist_alloc_tensors body (materialize + reclaim back-pressure)
    Build = 2,     // winner-only: fan-in resolution + built[] assembly (up to back-pressure)
    DrainWon = 3,  // drain_block_won pulled+built a follower deposit
    Replay = 4,    // submit replayed but claim LOST (per-core map/heap bookkeeping only)
    RingBp = 5,    // winner spun on ring/heap back-pressure (waiting for a free slot / reclaim)
    EfDrain = 6,   // execute-first drain at submit entry (deposits + ready owned tasks)
    Commit = 7,    // winner-only: alloc ring/won slot + build_ring_slot (publish the task)
};

struct TraceEvent {
    int32_t task_id;
    int32_t func_id;  // kernel id (e.g. 0=GEMM, 1=ADD); -1 if unknown
    int32_t lane;     // AIC=0 / AIV0=1 / AIV1=2
    uint8_t multicore;
    TracePhase phase;
    // Raw nanosecond timestamps — NO unit conversion on the hot path. The dump
    // stage divides by 1000 to emit microseconds (the swimlane unit).
    uint64_t ts_ns;   // start, ns from g_trace_epoch (wall clock)
    uint64_t dur_ns;  // span duration, ns (wall clock)
    // CPU time this thread actually accrued during the span (CLOCK_THREAD_CPUTIME_ID).
    // On an oversubscribed host dur_ns inflates while the thread is descheduled;
    // cpu_ns does not, so a large dur_ns with small cpu_ns == "swapped out, not work".
    // Only meaningful for non-kernel overhead spans (kernel spans set it to dur_ns).
    uint64_t cpu_ns;
};
#endif  // DIST_TRACE_ENABLED

struct RingSlot {
    bool occupied;
    // A slot can be reserved (occupied=true) before it is fully populated: the
    // submit winner grabs a slot up front so concurrent drains do not reuse it,
    // then may spin in block.won back-pressure (which itself drains Phase B)
    // before calling build_ring_slot. `built` gates execution so Phase B never
    // (re)runs a reserved-but-unbuilt slot still holding a prior occupant's
    // task_id/fanin/won linkage. build_ring_slot sets it; execute_slot clears it.
    bool built;
    int32_t task_id;
    int32_t func_id;  // kernel id of this slot's lane (swimlane label); -1 if none
    uint64_t function_bin_addr;

    int32_t tensor_count;
    int32_t scalar_count;
    Tensor tensors[MAX_TENSOR_ARGS];
    uint64_t scalars[MAX_SCALAR_ARGS];

    uint64_t args[PTO2_DISPATCH_MAX_ARGS];
    LocalContext local_ctx;
    GlobalContext global_ctx;

    int32_t fanin[kMaxFanin];
    int32_t fanin_count;

    // Multi-core (MIX / 2V) linkage. When is_multicore, the completion flag for
    // task_id is owned jointly: each co-owner decrements block.won[won_slot].remaining
    // after executing its own subtask, and the one driving it to zero publishes
    // the single global task_completed_flag. Single-core tasks set the flag directly.
    bool is_multicore;
    int32_t won_block;
    int32_t won_slot;
};

// -----------------------------------------------------------------------------
// block.won — the id-keyed anchor→follower deposit table (block-shared, §3.1).
// One BlockWon per physical block (1 AIC + 2 AIV). The anchor that wins a
// multi-core task builds its OWN physical-lane subtask into its private ring and
// deposits the remaining active-lane subtasks here; followers asynchronously
// drain the entry addressed to their physical lane (no blocking, no per-walk
// wait). Keyed by task id via per-slot task_id so concurrent multi-core tasks of
// one block never alias. `remaining` = popcount(active_mask) drives the single
// completion flag (§3.1). Lane index uses PTO2SubtaskSlot (AIC=0/AIV0=1/AIV1=2).
// -----------------------------------------------------------------------------
struct BuiltSubtask {
    bool present;
    int32_t func_id;  // kernel id of this lane's subtask (swimlane label); -1 if none
    uint64_t function_bin_addr;
    int32_t tensor_count;
    int32_t scalar_count;
    Tensor tensors[MAX_TENSOR_ARGS];
    uint64_t scalars[MAX_SCALAR_ARGS];
    int32_t fanin[kMaxFanin];
    int32_t fanin_count;
    int32_t sub_block_id;
};

struct WonSlot {
    Coherent<int32_t> state;  // 0=free, 1=published, 2=reserving
    int32_t task_id;
    Coherent<int32_t> remaining;                         // co-owners (incl. anchor) left to finish
    Coherent<int32_t> drained[PTO2_SUBTASK_SLOT_COUNT];  // 0/1 per follower lane
    BuiltSubtask lane[PTO2_SUBTASK_SLOT_COUNT];             // deposited follower subtasks
};

struct BlockWon {
    WonSlot slots[kPrivateSlots];
    // Monotone "has any anchor ever published a deposit into this block?" flag.
    // Lets follower drains short-circuit the per-slot scan for workloads with no
    // multi-core (e.g. 2V) tasks — the common case (bgemm is all single-core), so
    // every AIV core skips a 4-slot won-scan on every submit. Never reset within a
    // session; once true the scan path is taken (those workloads have real work).
    Coherent<int32_t> any_pub;
};

enum LaneId : int32_t { LANE_AIC = 0, LANE_AIV0 = 1, LANE_AIV1 = 2, LANE_NONE = -1 };

#if DIST_TRACE_ENABLED
// Swimlane tracing globals. Defined here (before DistCore) so DistCore::reset can
// see g_trace_reserve; g_trace_on / g_trace_epoch_ns sit alongside for one place.
//   g_trace_on      — set from PTO_DIST_SWIMLANE at register time; gates capture.
//   g_trace_epoch_ns — run-start epoch so every core's span ts is relative to it.
//   g_trace_reserve — per-core span reserve: 0 when off (reset never reserves, so
//     a normal run pays nothing), else a generous upper bound on spans/core so
//     push_back never reallocs mid-run (stable heap layout).
bool g_trace_on = false;
uint64_t g_trace_epoch_ns = 0;
int32_t g_trace_reserve = 0;
#endif

struct CoreLayout {
    int32_t block_id;  // physical block index
    int32_t lane;      // LaneId of this core within its block
};

// -----------------------------------------------------------------------------
// Per-core engine state (the SPMD worker context).
// -----------------------------------------------------------------------------
struct DistGlobal;  // fwd: DistCore holds a back-pointer to its owning segment

struct DistCore {
    __gm__ DistGlobal *gd;  // owning global_data segment (docs §13); set at register.
                            // Lets any helper holding `self` reach shared state as
                            // self->gd->… with no process-global symbol (CCEC-safe).
    CoreType role;
    int32_t core_idx;  // index into gd->cores[] (for trace ownership)
    int32_t block_id;  // physical block this core belongs to
    int32_t lane;      // LaneId within the block (AIC / AIV0 / AIV1)
    int32_t sub_block_id;
    int32_t local_index;  // next task id this core will see (== tasks replayed)
    uint64_t heap_next;   // deterministic GM output-heap bump cursor (bytes)

    DistTensorMap map;

    RingSlot slots[kPrivateSlots];
    int32_t occupied_count;
    int32_t owned_total;  // tasks this core claimed+executed (debug)

    Tensor outpool[kOutPoolSlots];
    int32_t outpool_head;

#if DIST_TRACE_ENABLED
    // Per-core swimlane events (only populated when tracing is on). Owned solely
    // by this core's worker thread, so push_back is lock-free.
    std::vector<TraceEvent> trace;

    // Running-cursor timestamps for lap-style tracing (see trace_lap). Each span is
    // [trace_last_ns, now); after recording, the cursor advances to now, so the next
    // span abuts this one with zero gap — the whole submit flow (incl. the orch
    // round-trip between two submits) is covered by exactly one span each, no code
    // path left un-timed. Reset at replay entry; wall + this-thread CPU clocks.
    uint64_t trace_last_ns;
    uint64_t trace_last_cpu;

    // Per-core static dependency edges (tracing only): one per fan-in resolved at
    // build time — {consumer_task, producer_task}. Dumped as Chrome-trace flow
    // events (producer's span -> consumer's span) so the swimlane shows the full
    // dependency graph; following the arrows hop-by-hop walks the chain "what is
    // this task waiting on, and what is THAT waiting on". Recorded by whichever
    // core builds the task, so every executed task contributes its in-edges.
    struct DepEdge {
        int32_t consumer_task;
        int32_t producer_task;
    };
    std::vector<DepEdge> dep_edges;

    // Per-core SLOT-RELEASE edges (tracing only): why a ringbp actually stalls.
    // When task N's owner enters the ring back-pressure, it is waiting not on N's
    // data producers but on the tasks ALREADY occupying its private ring to
    // execute (free a slot). Snapshot those occupants ({waiter=N, occupant}).
    // Dumped as flow events occupant-kernel -> ringbp: the occupant's execution is
    // the release event that ends the wait. Chains with dep_edges: ringbp -> its
    // ring occupant (slot edge) -> that occupant's data producers (dep edges).
    std::vector<DepEdge> slot_edges;
#endif  // DIST_TRACE_ENABLED
};

// DistCore::reset as a free function — DistCore is GM-resident, so its method
// cannot be called on a __gm__ object (CCEC this-addrspace constraint).
__aicore__ void dist_core_reset(
    __gm__ DistCore *self, CoreType r, int32_t block, int32_t lane_id, int32_t ring_cap
) {
    self->role = r;
    self->block_id = block;
    self->lane = lane_id;
    self->sub_block_id = (lane_id == LANE_AIV1) ? 1 : 0;
    self->local_index = 0;
    self->heap_next = 0;
    dist_tm_reset(&self->map, ring_cap);
    self->occupied_count = 0;
    self->owned_total = 0;
    self->outpool_head = 0;
    for (int32_t i = 0; i < kPrivateSlots; i++) {
        self->slots[i].occupied = false;
        self->slots[i].built = false;
    }
#if DIST_TRACE_ENABLED
    self->trace_last_ns = 0;
    self->trace_last_cpu = 0;
    self->trace.clear();
    if (g_trace_reserve > 0) self->trace.reserve(g_trace_reserve);
    self->dep_edges.clear();
    if (g_trace_reserve > 0) self->dep_edges.reserve(g_trace_reserve);
    self->slot_edges.clear();
    if (g_trace_reserve > 0) self->slot_edges.reserve(g_trace_reserve);
#endif  // DIST_TRACE_ENABLED
}

// -----------------------------------------------------------------------------
// Cursor sharding (docs §6.6). Each per-anchor-type claim cursor is split into
// kCursorShards independent sub-cursors; task id N claims on shard (N %
// kCursorShards). The shard is a pure function of N (identical on every core, no
// worker partitioning), so the claim semantics are byte-for-byte equivalent to a
// single cursor (exactly one owner per task, every core eligible) — sharding
// ONLY spreads the CAS traffic across kCursorShards cache lines, cutting the
// false-sharing / coherence contention that dominated us/task at high core
// counts (§6.5). Each sub-cursor is padded to its own cache line so adjacent
// shards never share a line; all entries init to -1 (no id claimed yet).
constexpr int32_t kCursorShards = 4;
constexpr size_t kCacheLine = 64;

struct alignas(kCacheLine) PaddedCursor {
    Coherent<int32_t> v;
    uint8_t pad[kCacheLine - sizeof(Coherent<int32_t>)];
};

// -----------------------------------------------------------------------------
// Global engine state (shared by all worker threads in this process). Cursors +
// flags live here rather than in GM because in sim every core is a host thread
// in one address space; the GM output heap below is a real shared buffer.
// -----------------------------------------------------------------------------
struct DistGlobal {
    PaddedCursor cube_cursor[kCursorShards];    // highest claimed AIC-anchored id, per shard
    PaddedCursor vector_cursor[kCursorShards];  // highest claimed AIV-only id, per shard
    PaddedCursor alloc_cursor[kCursorShards];   // highest claimed kernel-less alloc id, per shard
    // Completion-flag ring (1 == task done). int32_t (not uint8_t) so the flag can
    // be SET via the memory-level A5 atomicMax (dist_set_flag) — a byte has no
    // hardware atomic, and a plain store+dcci-flush on the dense byte ring
    // clobbers neighbouring flags across cores (docs §17 C4/C15).
    Coherent<int32_t> flags[kFlagCap];


    // M4 reclamation (§9.5/§11.4). `frontier` (F) is the global continuous
    // completion frontier — the largest prefix s.t. every task id <= F is done;
    // advanced cooperatively (CAS) by whichever core sets the flag that extends
    // the prefix. `R = frontier - H` is the reclaim frontier. `vend[N]` is the
    // cumulative virtual heap bytes through task N (deterministic & identical on
    // every core), so any core can compute the live byte window [vend[R], top).
    Coherent<int32_t> frontier;
    int32_t H;
    Coherent<uint64_t> vend[kFlagCap];

    uint8_t *heap_base;
    size_t heap_size;  // == bounded ring size

    DistOrchFunc orch_func;
    // Onboard fully_distributed_within_core only: GM address of the CCEC
    // orchestration blob's framework_bind_runtime. The blob is a separate
    // position-independent AICore image with its OWN g_current_runtime global
    // (linked from orchestration/common.cpp), so each core must bind that copy
    // to gd->rt before replaying orch_func — the runtime image's own
    // framework_bind_runtime (called below) only binds the runtime image's copy.
    // 0 on sim / host-orch paths, where the orchestration shares the binding.
    uint64_t orch_bind_func;
    const L2TaskArgs *orch_args;
    // NOTE: the GM-resident copy of the entry args (orch_args_gm) is deliberately
    // the LAST member of DistGlobal (declared after cores[] below), NOT here.
    // L2TaskArgs embeds a PTO2_PROFILING-gated DumpArgSelection member, so its
    // sizeof DIFFERS by 160 bytes between the AICPU build (profiling defaults ON →
    // member present) and the AICore CCEC build (-DPTO2_PROFILING=0 → member
    // absent). Placing it mid-struct shifted every following field (rt/runtime/
    // ops/num_workers/layout/blocks/cores) by 160 bytes on the AICPU relative to
    // the AICore — the AICore then read gd->rt at the wrong offset (=0) and
    // faulted writing through the NULL rt (MPU 271, docs §17 C10). Keeping it last
    // makes every field the AICore reads by offset layout-identical across both
    // compilers; the AICore reaches orch_args_gm only through the `orch_args`
    // pointer (an AICPU-written value) and touches only its Base (tensors_/
    // scalars_/counts), which precede DumpArgSelection and so match in both builds.
    __gm__ PTO2Runtime *rt;
    __gm__ Runtime *runtime;  // outer Runtime (for kernel-address resolution + done_count)
    // GM-resident runtime-ops table for the onboard AICore path. The static
    // g_dist_ops table bakes in link-time (base-0) function addresses that the
    // incore loader never relocates; dist_ops_refresh_aicore() fills THIS table
    // on-core with PC-relative (real loaded) addresses and rt->ops is pointed
    // here so orchestration's rt->ops->submit_task etc. branch to valid code.
    // Unused on sim/host (rt->ops points at the static g_dist_ops there).
    PTO2RuntimeOps ops;

    Coherent<int32_t> fatal;

    // Physical-block topology (1 AIC + 2 AIV per block), derived once at register
    // time from Runtime::workers[].core_type, identical to the centralized
    // scheduler's cluster discovery (AIC core b pairs with the 2b-th / (2b+1)-th
    // AIV cores in worker-index order).
    int32_t num_workers;
    int32_t num_blocks;
    CoreLayout layout[RUNTIME_MAX_WORKER];
    BlockWon blocks[RUNTIME_MAX_WORKER];  // indexed by block_id (<= num AIC)

    // Global "all cores finished orchestration replay" counter. A follower must
    // not conclude "no more pushes are coming for my lane" until every core has
    // finished replaying the submit stream (§7 tail-idle).
    Coherent<int32_t> replay_done;

    // Startup barrier: every worker thread bumps this on entry and spins until it
    // reaches num_workers before beginning replay. In sim each "core" is a host
    // pthread that the OS schedules in one at a time (hundreds of µs apart on a
    // busy box), so without this the first-claimed tasks start executing while
    // later cores have not even been scheduled — the swimlane shows a long
    // cold-start stagger that is host-scheduling noise, not engine behavior.
    // Aligning the start makes the trace reflect steady-state contention.
    Coherent<int32_t> started_count;

    // ---- Shared TensorMap mode (docs §12; PTO_DIST_TENSORMAP_MODE=shared) ----
    // One global ring shared by all cores (vs the per-core `DistCore::map`).
    SharedTensorMap shared_map;
    // Append sequencer: the next task id to be appended to `shared_map`. Serializes
    // appends in task-id order so the ring is a faithful, complete replica of what
    // any private core would hold at each task's fan-in resolution (=> identical
    // results). A core appending task N CASes this from N to kTmAppendBusy, appends,
    // then stores N+1. See tm_shared_claim_append.
    Coherent<int32_t> tm_insert_next;
    // Per-core replay progress (local_index), published each task. The shared
    // reclaim floor is (min over cores) − H − 1, so no live winner's window is
    // ever evicted (docs §12.7.1 / §9.5).
    Coherent<int32_t> core_progress[RUNTIME_MAX_WORKER];

    // ---- Run-configuration + verification, folded in from former file-scope
    // globals so NO process-global symbol is needed on CCEC/AICore (docs §13).
    // All set once at register (AICPU) and thereafter read by every core via the
    // base pointer. tensormap mode / ring cap / run-ahead bound are functional;
    // dep_sig/dep_edges are the (env-gated) dependency-graph signature oracle.
    bool tm_shared;             // PTO_DIST_TENSORMAP_MODE=shared (docs §12.2)
    int32_t ring_cap;           // per-bucket ring depth (docs §12.7.2)
    int32_t tm_runahead_max;    // shared run-ahead bound Δ_max (docs §12.7.2)
    int32_t runahead_max;       // BOTH modes: claim/replay frontier balance bound
                                // (max local_index spread across cores). 0 disables.
                                // Default = a small multiple of num_workers so the
                                // window fits every core yet caps a runaway core from
                                // racing the fetch_max claim cursor far ahead and
                                // starving laggards (load-balance knob, PTO_DIST_RUNAHEAD).
    Coherent<uint64_t> dep_sig;    // XOR of resolved fan-in edge hashes (§12.10)
    Coherent<uint64_t> dep_edges;  // count of resolved edges

    DistCore cores[RUNTIME_MAX_WORKER];

    // GM-resident copy of the orchestration entry args. The AICPU builds the
    // L2TaskArgs in its own DDR (orch_args_cached_), which the AICore cannot
    // address onboard; dist_engine_register copies it here (inside the GM
    // DistGlobal segment) and repoints `orch_args` (above) at it so every core
    // reads the args from GM. The container's TensorRefs already point into the
    // host-uploaded, GM-resident Runtime::orch_args_storage_. Harmless on sim.
    //
    // MUST stay the LAST member: L2TaskArgs's sizeof is PTO2_PROFILING-dependent
    // (DumpArgSelection member; 160 bytes) and the AICPU (profiling on) and AICore
    // (-DPTO2_PROFILING=0) builds disagree on it. As the final member its size
    // divergence only changes sizeof(DistGlobal) — the segment is allocated/
    // published with the AICPU's (larger) sizeof, and the AICore simply never
    // reads the trailing 160 bytes — instead of shifting every field after it.
    // Do NOT add new members below this one. (docs §17 C10.)
    L2TaskArgs orch_args_gm;
};

// 方案B (docs §13): the shared-state segment is NOT reached through any file-scope
// or thread_local symbol on the functional path. Its base is delivered to each
// AICore through the runtime objects the core already reads:
//   * dist_core_main gets it from runtime->dist.global_data_base (arg-passed);
//   * ops callbacks (which receive only PTO2Runtime*) get it from rt->dist_global,
//     set in register — mirroring how rt already reaches aicore_mailbox/sm_handle.
// `self` is then &gd->cores[pto_core_id()] (core id from a hardware register on HW,
// a thread_local in sim). Everything downstream threads `gd`/`self` as parameters.
__aicore__ inline __gm__ DistGlobal *pto_gd(__gm__ PTO2Runtime *rt) {
    // rt->dist_global is the GM address of the DistGlobal segment stored as a
    // uint64_t. Cast integer→__gm__ DistGlobal* directly — this is the same
    // integer→GM-pointer form CCEC accepts for runtime->dist.global_data_base.
    // (A void*→uintptr_t→__gm__ cast of a GM-loaded generic pointer makes CCEC's
    // backend fail with "error pointer address space cast"; the integer field
    // avoids that — see PTO2Runtime::dist_global.)
    return rt != nullptr ? reinterpret_cast<__gm__ DistGlobal *>(static_cast<uintptr_t>(rt->dist_global))
                         : nullptr;
}
__aicore__ inline __gm__ DistCore *pto_dist_self(__gm__ PTO2Runtime *rt) {
    __gm__ DistGlobal *gd = pto_gd(rt);
    const int32_t cid = pto_core_id();
    if (gd == nullptr || cid < 0 || cid >= RUNTIME_MAX_WORKER) return nullptr;
    return &gd->cores[cid];
}

// Diagnostics-only segment handle (NOT the functional path). dist_dump_state is a
// SIGUSR1/watchdog signal handler (signature void(int), cannot take gd) and
// dist_engine_dump_trace is a post-run swimlane dumper called arg-less from the
// AICPU stub. Both are host/sim debug aids (fprintf, chrono, env-gated) that do
// not compile onto the CCEC AICore, so a single AICPU-side pointer set in register
// is acceptable here and keeps the functional path symbol-free.
__gm__ DistGlobal *s_dump_gd = nullptr;

// Shared-vs-private TensorMap mode lives in the segment as gd->tm_shared (§12.2/§13).
// Sentinel parked in gd->tm_insert_next while a core is mid-append for a task
// (task ids are >= 0, so a negative value is unambiguously "append in progress").
constexpr int32_t kTmAppendBusy = -1;

// Shared-mode run-ahead bound (docs §12.7.2). The shared ring's live window is
// `Δ + H` where Δ = (fastest − slowest core) replay spread; a bucket overflows
// when that window (times per-task regions / hash skew) exceeds `cap`. We cap Δ
// so the window stays comfortably under `cap`, converting a would-be overflow
// into BACK-PRESSURE (the append frontier waits, draining, until the slowest core
// catches up). Set once in register from cap/H (env PTO_DIST_RUNAHEAD overrides;
// 0 disables the throttle so overflow reverts to the deterministic FATAL). 0 in
// private mode (private has no cross-core window — its reclaim is per-core N−H).
// Lives in the segment as gd->tm_runahead_max (docs §13).

// Dependency-graph signature (verification only, env PTO_DIST_DEPSIG). Each winner
// XORs a hash of every resolved fan-in edge (consumer task N, producer p) into
// gd->dep_sig; XOR is order-independent, so the value depends only on the SET of
// edges, not scheduling. private and shared modes must yield the identical
// signature => shared resolves the same dependency graph (float accumulation-order
// noise in the numeric golden makes bit-exact output comparison unreliable, so
// this graph-level oracle is the correctness gate). Reset per run in register.
// dep_sig/dep_edges live in the segment (docs §13); gd threaded in by the caller.
__aicore__ inline void dep_sig_add(__gm__ DistGlobal *gd, int32_t consumer, int32_t producer) {
    uint64_t x = (static_cast<uint64_t>(static_cast<uint32_t>(consumer)) << 32) |
                 static_cast<uint32_t>(producer);
    x *= 0x9E3779B97F4A7C15ULL;  // mix so adjacent edges don't collide trivially
    x ^= x >> 29;
    coherent_fetch_xor(&gd->dep_sig, x, std::memory_order_relaxed);
    coherent_fetch_add(&gd->dep_edges, 1, std::memory_order_relaxed);
}

#if DIST_SIM_HOST_CLOCK
// Orchestration/scheduling overhead isolation (set PTO_DIST_SKIP_EXEC=1). When
// on, execute_slot skips the actual incore kernel call — every (sub)task is
// treated as 0-cost and "completes" instantly — while ALL ownership/completion
// bookkeeping runs unchanged, so the loop terminates identically. This lets a
// benchmark measure the pure cost of on-core orchestration + claim race +
// scheduling, independent of kernel work. Outputs are NOT computed (run with
// golden checks disabled). See examples/.../runtime_overhead_test.
bool g_skip_exec = false;

// Uniform fake per-kernel execution cost (env PTO_DIST_FAKE_EXEC_NS=<ns>, sim
// only). When > 0, execute_slot "runs" EVERY incore (sub)task by busy-waiting
// this many nanoseconds instead of calling the real kernel — giving every task
// an equal, nonzero synthetic duration. This is the balance-study companion to
// PTO_DIST_SKIP_EXEC: skip-exec's 0-cost kernels let one fast core race ahead and
// win nearly all tasks (a meaningless imbalance artifact) and collapse swimlane
// bars to zero width; a uniform cost self-throttles the claim race (a core busy
// for fake_ns cannot keep winning) so the swimlane reflects GENUINE scheduling
// balance. Differs from use_example_exec_time (per-func recorded times): this is
// one uniform value for all funcs. Outputs are NOT computed (golden meaningless).
int32_t g_fake_exec_ns = 0;

inline uint64_t now_ns() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count()
    );
}

// On-core orchestration overhead summary (env PTO_DIST_OVERHEAD=1). Isolates the
// distributed runtime's own cost from the Python/sim launch overhead that swamps
// the host wall clock and from device_wall_us (which reads 0 for single cold sim
// runs). Each core times its post-barrier replay-start (t0, a local) to its
// drain-complete (t1), folds t0/t1/busy into the global min/max/sum atomics, then
// bumps g_orch_recorded; the LAST core to finish (recorded == num_workers) owns
// the complete picture and prints makespan = max(t1)-min(t0) + per-core busy
// min/avg/max. Using the last finisher + actual recorded values (not an index
// scan) is robust to sparse core_idx and needs no spin barrier. Under
// PTO_DIST_SKIP_EXEC=1 (0-cost kernels) makespan IS the pure on-core
// orchestration wall — the private-vs-shared metric. All seq_cst so the last
// finisher observes every other core's folds. (g_overhead_on declared up top.)
std::atomic<uint64_t> g_orch_t0_min{~0ull};
std::atomic<uint64_t> g_orch_t1_max{0};
std::atomic<uint64_t> g_orch_busy_min{~0ull};
std::atomic<uint64_t> g_orch_busy_max{0};
std::atomic<uint64_t> g_orch_busy_sum{0};
// Replay-phase (orch_func) wall per core. This is the map-build + fan-in-lookup +
// claim cost with NO drain-loop cross-core spin, so replay_us/task is the clean
// per-task orchestration overhead (the private-vs-shared metric) — unlike busy_us
// which is dominated by inter-core scheduling latency on an oversubscribed host.
std::atomic<uint64_t> g_orch_replay_min{~0ull};
std::atomic<uint64_t> g_orch_replay_max{0};
std::atomic<uint64_t> g_orch_replay_sum{0};
std::atomic<int32_t> g_orch_recorded{0};
inline void atomic_min_u64(std::atomic<uint64_t> &a, uint64_t v) {
    uint64_t cur = a.load(std::memory_order_seq_cst);
    while (v < cur && !a.compare_exchange_weak(cur, v, std::memory_order_seq_cst)) {
    }
}
inline void atomic_max_u64(std::atomic<uint64_t> &a, uint64_t v) {
    uint64_t cur = a.load(std::memory_order_seq_cst);
    while (v > cur && !a.compare_exchange_weak(cur, v, std::memory_order_seq_cst)) {
    }
}
#endif  // DIST_SIM_HOST_CLOCK

#if DIST_TRACE_ENABLED
// Per-thread CPU time (excludes time the thread spends descheduled). Used only by
// the swimlane to tell genuine work from host-oversubscription stalls, so it lives
// under DIST_TRACE_ENABLED (not the sim-clock gate — busy-wait never needs it).
inline uint64_t thread_cpu_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull + static_cast<uint64_t>(ts.tv_nsec);
}

// Snapshot the clock only when tracing is on (callers pass the result as a span
// start). Returns 0 otherwise so the matching trace_overhead() is a no-op.
inline uint64_t trace_now() { return g_trace_on ? now_ns() : 0; }
inline uint64_t trace_now_cpu() { return g_trace_on ? thread_cpu_ns() : 0; }

// Record a non-kernel overhead span [t0_ns, now) on this core's lane. Stores RAW
// nanoseconds (no unit conversion on the hot path — the dump stage divides by
// 1000). cpu_ns is this thread's CPU time over the span (small cpu with large dur
// == descheduled, not work). No-op unless tracing is on.
inline void trace_overhead_impl(
    __gm__ DistCore *self, int32_t task_id, int32_t func_id, TracePhase phase, uint64_t t0_ns, uint64_t t0_cpu
) {
    if (!g_trace_on) return;
    const uint64_t t1 = now_ns();
    const uint64_t c1 = thread_cpu_ns();
    self->trace.push_back(
        TraceEvent{
            task_id, func_id, self->lane, /*multicore=*/0, phase, t0_ns - g_trace_epoch_ns, t1 - t0_ns, c1 - t0_cpu
        }
    );
}

// Reset the lap cursor to "now" — call once at replay entry so the first lap span
// measures from a well-defined origin (not from an uninitialized cursor).
inline void trace_lap_reset_impl(__gm__ DistCore *self) {
    if (!g_trace_on) return;
    self->trace_last_ns = now_ns();
    self->trace_last_cpu = thread_cpu_ns();
}

// Lap-style span: record [trace_last_ns, now) then advance the cursor to now, so
// the next lap continues seamlessly from here (same idiom as pto_orchestrator's
// CYCLE_COUNT_LAP: acc += t1 - t0; t0 = t1). Every code path between two laps is
// attributed to exactly one span — no gaps, no double-counting. Stores raw ns.
inline void trace_lap_impl(__gm__ DistCore *self, int32_t task_id, int32_t func_id, TracePhase phase) {
    if (!g_trace_on) return;
    const uint64_t t1 = now_ns();
    const uint64_t c1 = thread_cpu_ns();
    self->trace.push_back(
        TraceEvent{
            task_id, func_id, self->lane, /*multicore=*/0, phase, self->trace_last_ns - g_trace_epoch_ns,
            t1 - self->trace_last_ns, c1 - self->trace_last_cpu
        }
    );
    self->trace_last_ns = t1;
    self->trace_last_cpu = c1;
}

// Trace call-site macros forward to the _impl inlines above; the #else branch below
// expands them to nothing — so call sites need no #if, and the phase enum /
// TraceEvent need not even exist when off (the preprocessor eats the whole argument
// list, TracePhase::X included). Same idiom as pto_orchestrator's CYCLE_COUNT_LAP.
#define TRACE_LAP(self, task_id, func_id, phase) trace_lap_impl((self), (task_id), (func_id), (phase))
#define TRACE_LAP_RESET(self) trace_lap_reset_impl((self))
#define TRACE_OVERHEAD(self, task_id, func_id, phase, t0_ns, t0_cpu) \
    trace_overhead_impl((self), (task_id), (func_id), (phase), (t0_ns), (t0_cpu))
#else  // !DIST_TRACE_ENABLED — tracing compiled out; call sites become no-ops.
#define TRACE_LAP(self, task_id, func_id, phase) ((void)0)
#define TRACE_LAP_RESET(self) ((void)0)
#define TRACE_OVERHEAD(self, task_id, func_id, phase, t0_ns, t0_cpu) ((void)0)
#endif  // DIST_TRACE_ENABLED

// Opt-in per-core tracing (set PTO_DIST_TRACE=1). Off by default so a passing
// run is quiet; fatal/error/heap-exhaustion diagnostics are always emitted.
__aicore__ inline bool dist_trace() {
#if DIST_SIM_HOST_CLOCK
    static const bool on = (getenv("PTO_DIST_TRACE") != nullptr);
    return on;
#else
    return false;  // AICore: no env; tracing is a host/sim-only diagnostic.
#endif
}

// -----------------------------------------------------------------------------
// Fatal / claim / execution helpers
// -----------------------------------------------------------------------------
__aicore__ inline bool fatal_set(__gm__ DistGlobal *gd) { return coherent_load(&gd->fatal, std::memory_order_acquire) != 0; }
__aicore__ inline void set_fatal(__gm__ DistGlobal *gd) { coherent_store(&gd->fatal, 1, std::memory_order_release); }

#if DIST_SIM_HOST_CLOCK
void dist_dump_state(int);  // defined below; dumps full engine state for hangs
#endif

// Env-gated stall watchdog (set PTO_DIST_WATCHDOG=<seconds>, default off). Called
// from inside the engine's spin loops on a worker thread (so fprintf is safe,
// unlike a signal handler). On the first call it records a start time; if a loop
// keeps spinning past the budget the engine is presumed deadlocked, so it dumps
// the full state once and sets fatal to unwind every core for a fast, diagnosed
// failure instead of an indefinite hang.
__aicore__ inline void watchdog(__gm__ DistGlobal *gd, uint64_t &start_ns) {
#if DIST_SIM_HOST_CLOCK
    static const long budget_s = []() -> long {
        const char *e = getenv("PTO_DIST_WATCHDOG");
        return e ? atol(e) : 0;
    }();
    if (budget_s <= 0) return;
    const uint64_t now = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count()
    );
    if (start_ns == 0) {
        start_ns = now;
        return;
    }
    if (now - start_ns > static_cast<uint64_t>(budget_s) * 1000000000ull) {
        static std::atomic<int32_t> dumped{0};
        int32_t exp = 0;
        if (dumped.compare_exchange_strong(exp, 1, std::memory_order_acq_rel)) {
            fprintf(stderr, "[dist_engine] WATCHDOG fired after %lds — presumed deadlock, dumping state\n", budget_s);
            dist_dump_state(0);
        }
        set_fatal(gd);
    }
#else
    // AICore: no host clock / no env / no stdio — watchdog is a host-only
    // diagnostic. The functional deadlock escape is the fatal flag, set
    // elsewhere; a true aicore watchdog would use a hardware timer (TODO a5).
    (void)gd;
    (void)start_ns;
#endif
}

// fetch_max claim (§11.1 / §11.1.1): returns true (WON) iff this core advanced
// the cursor to N. On A5 (dav_3510) atomicMax IS a memory-level, cross-core
// hardware atomic (CANN kernel_operator_atomic_impl.h; docs §11.1.1) — a single
// atomicMax both reads the true in-memory value and publishes the max, so there
// is NO CAS retry and NO leading coherent_load (which on cacheable GM would read
// a stale per-core cached copy). Monotonic: each task id is claimed by exactly
// one core and no id is skipped within a cursor's subsequence.
// Sim keeps the acq-rel CAS retry (host std::atomic, single address space).
__aicore__ bool claim(__gm__ Coherent<int32_t> *cursor, int32_t N) {
#if DIST_SIM_HOST_CLOCK
    int32_t c = coherent_load(cursor, std::memory_order_acquire);
    while (true) {
        if (N <= c) return false;
        if (coherent_cas_weak(cursor, c, N, std::memory_order_acq_rel, std::memory_order_acquire)) return true;
    }
#else
    const int32_t old = dist_atomic_max(&cursor->a, N);  // hardware fetch_max
    return old < N;
#endif
}

// Cooperatively advance the global completion frontier F (§11.4): after any core
// publishes flag(N), the contiguous-done prefix may have grown, so any core walks
// F forward while flag(F+1) is set. Lock-free; the CAS makes exactly one core win
// each step and the cost is amortized across all cores.
__aicore__ void advance_frontier(__gm__ DistGlobal *gd) {
    int32_t f = coherent_load(&gd->frontier, std::memory_order_acquire);
    while (true) {
        const int32_t next = f + 1;
        if (next >= kFlagCap) break;
        if (coherent_load(&gd->flags[next & (kFlagCap - 1)], std::memory_order_acquire) == 0) break;
        if (coherent_cas_weak(&gd->frontier, f, next, std::memory_order_acq_rel, std::memory_order_acquire)) {
            f = next;
        }
        // On CAS failure f was reloaded with the current value; retry.
    }
}

// Resolve a kernel id to its executable address (CoreCallable::resolved_addr()).
__aicore__ uint64_t resolve_kernel_addr(__gm__ Runtime *runtime, int32_t kernel_id) {
    if (kernel_id == INVALID_KERNEL_ID) return 0;
    // Direct field access (Runtime::get_function_bin_addr is a member method and
    // cannot be called on a __gm__ Runtime* — CCEC this-addrspace constraint).
    // func_id_to_addr_ is public (runtime.h:236) and intended for direct AICore
    // access; resolved_addr_ is a public field of CoreCallable (callable.h:62).
    if (kernel_id < 0 || kernel_id >= RUNTIME_MAX_FUNC_ID) return 0;
    const uint64_t callable_addr = runtime->func_id_to_addr_[kernel_id];
    if (callable_addr == 0) return 0;
    __gm__ const CoreCallable *callable = reinterpret_cast<__gm__ const CoreCallable *>(callable_addr);
    return callable->resolved_addr_;
}

// Execute one owned task, then publish its completion flag (release). In sim all
// cores share the address space, so the release/acquire pair is the visibility
// barrier between the kernel's output writes and a consumer's input reads.
// C16b: submit/execute-path breadcrumb. DIST_CRUMB proper is defined later (after
// dist_core_main), so this inline helper lets the submit + execute paths publish
// the same per-core stage markers into the crumb slot the AICPU waiting-loop
// scrapes. Writes + flushes unconditionally (same channel/pattern as the C16 CI
// diag); harmless on sim/AICPU builds.
static __aicore__ inline void submit_crumb(__gm__ Runtime *rt, int32_t idx, uint32_t val) {
    __gm__ volatile uint32_t *slot = &rt->dist.aicore_progress[idx * AICORE_PROGRESS_STRIDE];
    *slot = val;
    pto_dcci_flush(const_cast<__gm__ uint32_t *>(slot), sizeof(uint32_t));
    __asm__ __volatile__("" ::: "memory");
}

__aicore__ void execute_slot([[maybe_unused]] __gm__ DistCore *self, __gm__ RingSlot &s) {
    __gm__ DistGlobal *gd = self->gd;
#if DIST_SIM_HOST_CLOCK
    typedef void (*KernelFn)(int64_t *);
#else
    // AICore/CCEC: kernel args live in GM (RingSlot::args is __gm__ uint64_t[]),
    // so the kernel entry takes a __gm__-qualified int64_t pointer.
    typedef void (*KernelFn)(__gm__ int64_t *);
#endif
#if DIST_SIM_HOST_CLOCK
    // Sim-only trace-driven replay (CallConfig::use_example_exec_time): when the
    // host filled example_exec_time_ns_[func_id] > 0 for this func, "execute" it
    // by busy-waiting that many nanoseconds instead of calling the real kernel,
    // so a fast sim run reflects measured on-hardware kernel durations. 320 host
    // cores >> 72 workers, so the spin does not contend; funcs left at 0 fall
    // through to the real call below. See Runtime::example_exec_time_ns_.
    const __gm__ Runtime *rt = gd->runtime;
    int32_t sim_ns =
        (rt != nullptr && rt->use_example_exec_time_ && s.func_id >= 0 && s.func_id < RUNTIME_MAX_FUNC_ID) ?
            rt->example_exec_time_ns_[s.func_id] :
            0;
    // Uniform fake cost fallback (PTO_DIST_FAKE_EXEC_NS): applies to every kernel
    // that has no per-func recorded time. See g_fake_exec_ns.
    if (sim_ns == 0 && g_fake_exec_ns > 0) sim_ns = g_fake_exec_ns;
    if (sim_ns > 0) {
        const uint64_t t0 = now_ns();
        const uint64_t target = t0 + static_cast<uint64_t>(sim_ns);
        while (now_ns() < target) { /* spin: emulate kernel busy time */
        }
#if DIST_TRACE_ENABLED
        if (g_trace_on) {
            self->trace.push_back(
                TraceEvent{
                    s.task_id, s.func_id, self->lane, static_cast<uint8_t>(s.is_multicore ? 1 : 0), TracePhase::Kernel,
                    t0 - g_trace_epoch_ns, static_cast<uint64_t>(sim_ns), static_cast<uint64_t>(sim_ns)
                }
            );
        }
#endif
    } else if (s.function_bin_addr != 0 && !g_skip_exec) {
        // PTO_DIST_SKIP_EXEC: treat the incore task as 0-cost — skip the kernel call
        // but keep every flag/frontier/slot update below so termination is identical.
        KernelFn fn = reinterpret_cast<KernelFn>(s.function_bin_addr);
#if DIST_TRACE_ENABLED
        if (g_trace_on) {
            const uint64_t t0 = now_ns();
            fn(reinterpret_cast<int64_t *>(s.args));
            const uint64_t t1 = now_ns();
            self->trace.push_back(
                TraceEvent{
                    s.task_id, s.func_id, self->lane, static_cast<uint8_t>(s.is_multicore ? 1 : 0), TracePhase::Kernel,
                    t0 - g_trace_epoch_ns, t1 - t0, t1 - t0
                }
            );
        } else {
            fn(reinterpret_cast<int64_t *>(s.args));
        }
#else
        fn(reinterpret_cast<int64_t *>(s.args));
#endif
    }
#else   // !DIST_SIM_HOST_CLOCK — AICore/CCEC: no host clock, no busy-wait emulation.
    if (s.function_bin_addr != 0) {
        KernelFn fn = reinterpret_cast<KernelFn>(s.function_bin_addr);
        // C16: producer and consumer tasks run on DIFFERENT cores, exchanging
        // intermediate tensors (c/d/e/g) through CACHEABLE HBM (the outpool). The
        // completion FLAG is now cross-core coherent (dist_set_flag), but the
        // tensor DATA is not automatically so: this core may hold a STALE cached
        // copy of a producer's output, and its own MTE3 output may linger in cache
        // instead of reaching HBM. Mirror the standard runtime's per-task
        // coherence (a5 tensormap aicore_executor):
        //   - INVALIDATE the whole data cache BEFORE the kernel so its MTE2 loads
        //     pull the producer's freshly-written GM, not a stale line;
        //   - FLUSH the whole data cache OUT AFTER the kernel so the MTE3-written
        //     output reaches HBM before a dependent core (which invalidates before
        //     its own kernel) reads it.
        dcci(reinterpret_cast<__gm__ int32_t *>(s.args), ENTIRE_DATA_CACHE);
        __asm__ __volatile__("" ::: "memory");
        fn(reinterpret_cast<__gm__ int64_t *>(s.args));
        __asm__ __volatile__("" ::: "memory");
        dcci(reinterpret_cast<__gm__ int32_t *>(s.args), ENTIRE_DATA_CACHE, CACHELINE_OUT);
        // C16 DIAG: snapshot this task's first output element (out[0]) + task_id
        // into this core's OWN aicore_progress spare slots (+3/+4) — the proven,
        // AICPU-readable per-core crumb line (offset-safe, already dcci-coherent).
        // The AICPU FINAL dump prints slots 3/4 per core, so the union across the
        // owning cores reveals the c/d/e/g/f chain. Output tensor = last tensor.
        // C16b DIAG (temporarily disabled): the post-kernel out[0] snapshot below
        // invalidates a fixed 64 KB and dereferences the output buffer; isolating
        // whether the execute_slot hang is the kernel fn() itself or this probe.
#if 0
        if (s.tensor_count > 0) {
            __gm__ Tensor &_ot = s.tensors[s.tensor_count - 1];
            __gm__ float *_op = reinterpret_cast<__gm__ float *>(_ot.buffer.addr) + _ot.start_offset;
            pto_dcci_inval(_op, 16384 * sizeof(float));
            __gm__ volatile uint32_t *_pr =
                &gd->runtime->dist.aicore_progress[self->core_idx * AICORE_PROGRESS_STRIDE];
            _pr[3] = static_cast<uint32_t>(static_cast<int32_t>(_op[0]));       // out[0]
            _pr[4] = static_cast<uint32_t>(s.task_id);                          // which logical task
            _pr[5] = static_cast<uint32_t>(s.tensors[0].shapes[0]);             // input[0] shape[0]
            _pr[6] = static_cast<uint32_t>(s.tensors[0].buffer.size);           // input[0] buffer bytes
            _pr[7] = static_cast<uint32_t>(_ot.shapes[0]);                      // output shape[0]
            _pr[8] = static_cast<uint32_t>(_ot.buffer.size);                    // output buffer bytes
            pto_dcci_flush(const_cast<__gm__ uint32_t *>(&_pr[3]), 6 * sizeof(uint32_t));
        }
#endif
        // C16b: mark that execute_slot got PAST fn() (kernel returned) for this
        // core — distinguishes a kernel hang (never reaches here) from a
        // post-kernel/flag-publish deadlock.
        submit_crumb(gd->runtime, self->core_idx, 49);
    }
#endif  // DIST_SIM_HOST_CLOCK
    if (s.is_multicore) {
        // Joint ownership: the co-owner that drives remaining to zero (the last
        // subtask to finish) publishes the single global completion flag (§3.1),
        // then frees the block.won entry for reuse.
        __gm__ WonSlot &w = gd->blocks[s.won_block].slots[s.won_slot];
        if (coherent_fetch_sub(&w.remaining, 1, std::memory_order_acq_rel) == 1) {
            dist_set_flag(&gd->flags[s.task_id & (kFlagCap - 1)]);
            coherent_store(&w.state, 0, std::memory_order_release);  // recycle the id-keyed slot
            advance_frontier(gd);
        }
    } else {
        dist_set_flag(&gd->flags[s.task_id & (kFlagCap - 1)]);
        advance_frontier(gd);
    }
    s.built = false;
    s.occupied = false;
}

// Phase B: execute every ready owned task in the private ring. A task is ready
// once all its fan-in producers have set their completion flag (acquire).
// Returns the number of slots freed this pass.
__aicore__ int32_t drain_phase_b(__gm__ DistCore *self) {
    __gm__ DistGlobal *gd = self->gd;
    // Fast path: an empty private ring has nothing to drain. Skips the per-slot
    // scan on every submit point (called twice per task, on every core) when the
    // ring is empty — the common case for fine-grained / skip-exec workloads.
    // Behavior-identical: the loop below is a no-op when occupied_count == 0.
    if (self->occupied_count == 0) return 0;
    int32_t freed = 0;
    for (int32_t i = 0; i < kPrivateSlots; i++) {
        __gm__ RingSlot &s = self->slots[i];
        if (!s.occupied || !s.built) continue;  // skip reserved-but-unbuilt slots
        bool ready = true;
        for (int32_t f = 0; f < s.fanin_count; f++) {
            if (coherent_load(&gd->flags[s.fanin[f] & (kFlagCap - 1)], std::memory_order_acquire) == 0) {
                ready = false;
                break;
            }
        }
        if (!ready) continue;
        execute_slot(self, s);
        self->occupied_count--;
        freed++;
    }
    return freed;
}

__aicore__ int32_t alloc_ring_slot(__gm__ DistCore *self) {
    for (int32_t i = 0; i < kPrivateSlots; i++) {
        if (!self->slots[i].occupied) return i;
    }
    return -1;
}

// Kernel id for a physical lane (AIC/AIV0/AIV1) of a MixedKernels.
__aicore__ inline int32_t kernel_id_for_lane(const MixedKernels &mixed, int32_t lane) {
    switch (lane) {
    case LANE_AIC:
        return mixed.aic_kernel_id;
    case LANE_AIV0:
        return mixed.aiv0_kernel_id;
    case LANE_AIV1:
        return mixed.aiv1_kernel_id;
    default:
        return INVALID_KERNEL_ID;
    }
}

__aicore__ inline bool lane_active(const ActiveMask &M, int32_t lane) {
    return M.subtask_active(static_cast<PTO2SubtaskSlot>(lane));
}

// Materialize a private-ring slot from already-resolved components (shared by the
// owner build path and the follower drain path). `tensors`/`scalars` are copied
// in; args[] is (re)built to point at this slot's own copies so the slot is
// self-contained and executable at any later time.
// Templated on the source pointer types: callers pass either stack-local
// (generic) arrays or __gm__ arrays (e.g. BuiltSubtask::tensors), and CCEC
// forbids cross-address-space pointer casts.
template <class TPtr, class SPtr, class FPtr>
__aicore__ void build_ring_slot(
    __gm__ RingSlot *s, int32_t task_id, int32_t func_id, uint64_t fn_addr, TPtr tensors, int32_t tc,
    SPtr scalars, int32_t sc, FPtr fanin, int32_t fc, int32_t sub_block_id, bool is_multicore,
    int32_t won_block, int32_t won_slot
) {
    s->occupied = true;
    s->task_id = task_id;
    s->func_id = func_id;
    s->function_bin_addr = fn_addr;
    s->built = true;  // fully populated below — now safe for Phase B to execute
    s->tensor_count = tc;
    s->scalar_count = sc;
    for (int32_t i = 0; i < tc; i++)
        tensor_copy(&s->tensors[i], tensors[i]);
    for (int32_t j = 0; j < sc; j++)
        s->scalars[j] = scalars[j];
    int32_t n = 0;
    for (int32_t i = 0; i < tc; i++)
        s->args[n++] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&s->tensors[i]));
    for (int32_t j = 0; j < sc; j++)
        s->args[n++] = s->scalars[j];
    dist_set_local_block(s->local_ctx, 0, 1);
    {
        // AsyncCtx{) on a __gm__ field: build a zeroed local, then memcpy into GM
        // (CCEC forbids assigning a generic temporary to a __gm__ lvalue).
        AsyncCtx _zero_async{};
        memcpy(&s->local_ctx.async_ctx, &_zero_async, sizeof(AsyncCtx));
    }
    s->global_ctx.sub_block_id = sub_block_id;
    s->args[SPMD_LOCAL_CONTEXT_INDEX] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&s->local_ctx));
    s->args[SPMD_GLOBAL_CONTEXT_INDEX] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&s->global_ctx));
    s->fanin_count = fc;
    for (int32_t k = 0; k < fc; k++)
        s->fanin[k] = fanin[k];
    s->is_multicore = is_multicore;
    s->won_block = won_block;
    s->won_slot = won_slot;
}

// Reserve a free block.won slot in `block`. Returns slot index or -1 if full.
// 2V allows either AIV of the block to be an anchor, so allocation must be atomic.
__aicore__ int32_t alloc_won_slot(__gm__ DistGlobal *gd, int32_t block) {
__gm__ BlockWon &bw = gd->blocks[block];
    for (int32_t i = 0; i < kPrivateSlots; i++) {
        int32_t exp = 0;
        if (coherent_cas_strong(&bw.slots[i].state, exp, 2, std::memory_order_acq_rel, std::memory_order_relaxed)) {
            return i;
        }
    }
    return -1;
}

// True if a published block.won deposit for this core's lane has not yet been
// taken — used by the termination check to avoid finishing before draining.
__aicore__ bool has_pending_won(__gm__ DistCore *self) {
    __gm__ DistGlobal *gd = self->gd;
    if (self->lane == LANE_AIC || self->lane == LANE_NONE) return false;
    if (self->block_id < 0 || self->block_id >= gd->num_blocks) return false;  // defensive (§17 MPU-271)
__gm__ BlockWon &bw = gd->blocks[self->block_id];
    if (coherent_load(&bw.any_pub, std::memory_order_acquire) == 0) return false;  // no deposit ever published
    for (int32_t i = 0; i < kPrivateSlots; i++) {
        __gm__ WonSlot &w = bw.slots[i];
        if (coherent_load(&w.state, std::memory_order_acquire) != 1) continue;
        if (w.lane[self->lane].present && coherent_load(&w.drained[self->lane], std::memory_order_acquire) == 0) return true;
    }
    return false;
}

// Follower drain (§3.1, §6): pull every published block.won subtask addressed to
// this core's physical lane that we have not yet taken, building each into a free
// private-ring slot (back-pressure: stop when the ring is full). Non-blocking —
// if nothing is addressed to us we simply return.
__aicore__ void drain_block_won(__gm__ DistCore *self) {
    __gm__ DistGlobal *gd = self->gd;
    if (self->lane == LANE_AIC || self->lane == LANE_NONE) return;  // AIC is never a follower
    // Defensive bound (docs §17 / MPU-271 hunt): a stale/garbage block_id would make
    // gd->blocks[block_id] a wild address that faults the MPU. num_blocks is validated
    // at register; clamp here so a bad id can never index out of the blocks[] array.
    if (self->block_id < 0 || self->block_id >= gd->num_blocks) return;
__gm__ BlockWon &bw = gd->blocks[self->block_id];
    // Fast path: if no anchor has ever published a deposit into this block, there
    // is nothing to drain — skip the per-slot scan on every submit (hot path).
    if (coherent_load(&bw.any_pub, std::memory_order_acquire) == 0) return;
    for (int32_t i = 0; i < kPrivateSlots; i++) {
        __gm__ WonSlot &w = bw.slots[i];
        if (coherent_load(&w.state, std::memory_order_acquire) != 1) continue;
        if (!w.lane[self->lane].present) continue;
        int32_t exp = 0;
        if (!coherent_cas_strong(&w.drained[self->lane],
                exp, 1, std::memory_order_acq_rel, std::memory_order_relaxed
            ))
            continue;  // already taken by us on a prior pass
        int32_t si = alloc_ring_slot(self);
        if (si < 0) {
            // Ring full: hand the deposit back and let Phase B free a slot first.
            coherent_store(&w.drained[self->lane], 0, std::memory_order_release);
            return;
        }
        const __gm__ BuiltSubtask &b = w.lane[self->lane];
#if DIST_TRACE_ENABLED
        const uint64_t t_won0 = trace_now();
        const uint64_t t_won0_cpu = trace_now_cpu();
#endif
        build_ring_slot(
            &self->slots[si], w.task_id, b.func_id, b.function_bin_addr, b.tensors, b.tensor_count, b.scalars,
            b.scalar_count, b.fanin, b.fanin_count, b.sub_block_id, /*is_multicore=*/true, self->block_id, i
        );
        self->occupied_count++;
        self->owned_total++;
#if DIST_TRACE_ENABLED
        if (g_trace_on) {
            for (int32_t k = 0; k < b.fanin_count; k++)
                self->dep_edges.push_back({w.task_id, b.fanin[k]});
        }
        trace_overhead_impl(self, w.task_id, b.func_id, TracePhase::DrainWon, t_won0, t_won0_cpu);
#endif
    }
}

// -----------------------------------------------------------------------------
// Shared TensorMap helpers (docs §12.4-§12.7). Only used when gd->tm_shared.
// -----------------------------------------------------------------------------

// Slowest core's replay progress (min over all cores). O(num_workers).
__aicore__ inline int32_t tm_shared_min_progress(__gm__ DistGlobal *gd) {
    int32_t mn = INT32_MAX;
    const int32_t nw = gd->num_workers;
    for (int32_t c = 0; c < nw; c++) {
        const int32_t p = coherent_load(&gd->core_progress[c], std::memory_order_relaxed);
        if (p < mn) mn = p;
    }
    return mn;
}

// Global reclaim floor for the shared ring: producers <= this are retired. Based
// on the SLOWEST core's replay progress so no live winner's [N-H, N) window is
// ever evicted (docs §12.7.1 / §9.5). O(num_workers), called once per append.
__aicore__ inline int32_t tm_shared_reclaim_floor(__gm__ DistGlobal *gd) { return tm_shared_min_progress(gd) - gd->H - 1; }

// Run-ahead balance throttle (BOTH modes). Before advancing to task N, keep this
// core within gd->runahead_max tasks of the SLOWEST core's replay position. The
// claim (fetch_max on a monotonic cursor, §11.1) lets whichever core reaches N
// first win it, so a core that races its local_index far ahead grabs a long
// contiguous prefix and starves the cores behind it — the imbalance seen when
// cores progress at unequal rates (host oversubscription in sim; a straggler core
// on HW). Capping the spread bounds how far any core runs ahead, so laggards keep
// claiming their share. Pure WAITING (drains owned ready work + follower deposits
// meanwhile); it changes only WHICH core wins, never the deterministic map replay
// / dependency graph (DEPSIG invariant). Deadlock-free: the slowest core has
// (N - min) == 0, is never throttled, advances, and raises min_progress — which
// releases everyone ahead. Disabled when runahead_max <= 0 (zero overhead: one
// compare). core_progress[] is published by the caller before this wait so the
// min is current. NOTE: runahead_max must be >= num_workers, otherwise the window
// is too small to hold one task per core and healthy parallelism is throttled into
// idle cores (see register default).
__aicore__ inline void dist_runahead_throttle(__gm__ DistCore *self, int32_t N) {
    __gm__ DistGlobal *gd = self->gd;
    if (gd->runahead_max <= 0) return;
    uint64_t wd = 0;
    while ((N - tm_shared_min_progress(gd)) > gd->runahead_max && !fatal_set(gd)) {
        drain_block_won(self);
        if (drain_phase_b(self) == 0) {
            SPIN_WAIT_HINT();
            watchdog(gd, wd);
        }
    }
}

// Sequenced, exactly-once append claim for task N (docs §12.4). Returns true iff
// THIS core must perform the append (it won the sequencer at insert_next==N); the
// caller then appends every output region via gd->shared_map.append(...) and
// calls tm_shared_publish(N). Returns false if task N is already appended by
// another core (nothing to do) — in that case it has spun (draining ready work)
// until the append is visible, so the ring holds task N on return. Guarantees
// appends happen strictly in id order (append N only when 0..N-1 are done), which
// is what makes the shared ring content identical to a private replica.
__aicore__ inline bool tm_shared_claim_append(__gm__ DistCore *self, int32_t N) {
    __gm__ DistGlobal *gd = self->gd;
    uint64_t wd = 0;
    while (true) {
        // Run-ahead back-pressure (docs §12.7.2): keep the append frontier within
        // Δ_max of the slowest core so the shared ring's `Δ+H` live window stays
        // under `cap`. We wait here WITHOUT holding the sequencer (insert_next is
        // still == N, not BUSY), so cores behind N keep advancing (their
        // claim_append returns immediately since insert_next > their id) and raise
        // min_progress — which releases us. Deadlock-free; only throttles the
        // frontier. Disabled when gd->tm_runahead_max == 0 (=> overflow FATALs).
        if (gd->tm_runahead_max > 0 && (N - tm_shared_min_progress(gd)) > gd->tm_runahead_max) {
            if (fatal_set(gd)) return false;
            drain_block_won(self);
            if (drain_phase_b(self) == 0) {
                SPIN_WAIT_HINT();
                watchdog(gd, wd);
            }
            continue;
        }
        const int32_t v = coherent_load(&gd->tm_insert_next, std::memory_order_acquire);
        if (v > N) return false;  // task N already appended and published
        if (v == N) {
            int32_t expect = N;
            if (coherent_cas_strong(&gd->tm_insert_next,
                    expect, kTmAppendBusy, std::memory_order_acq_rel, std::memory_order_acquire))
                return true;  // we own the append for N
            // Lost the CAS (another core is appending N) — fall through to spin.
        }
        // v < N means a lower task is not yet appended, or v == kTmAppendBusy means
        // someone is mid-append; either way help drain and retry until it advances.
        if (fatal_set(gd)) return false;
        drain_block_won(self);
        if (drain_phase_b(self) == 0) {
            SPIN_WAIT_HINT();
            watchdog(gd, wd);
        }
    }
}

__aicore__ inline void tm_shared_publish(__gm__ DistGlobal *gd, int32_t N) {
    coherent_store(&gd->tm_insert_next, N + 1, std::memory_order_release);
}

// Append every OUTPUT / INOUT / OUTPUT_EXISTING region of task N to the shared
// ring, exactly once across all cores and strictly in id order. `outputs` are the
// materialized OUTPUT tensors (from result); INOUT / OUTPUT_EXISTING come from
// args. Called by EVERY core in shared mode (mirrors the unconditional private
// insert), but only the sequencer winner actually writes the ring.
inline __aicore__ void tm_shared_append_task(
    __gm__ DistCore *self, int32_t N, const L0TaskArgs &args, TaskOutputTensors &result
) {
    __gm__ DistGlobal *gd = self->gd;
    if (!tm_shared_claim_append(self, N)) return;
    const int32_t rfloor = tm_shared_reclaim_floor(gd);
    const int32_t tc = args.tensor_count();
    uint32_t out_idx = 0;
    for (int32_t i = 0; i < tc; i++) {
        const TensorArgType tag = args.tag(i);
        if (tag == TensorArgType::OUTPUT) {
            shared_tm_append(&gd->shared_map, result.get_ref(out_idx), N, rfloor);
            out_idx++;
        } else if (tag == TensorArgType::INOUT || tag == TensorArgType::OUTPUT_EXISTING) {
            shared_tm_append(&gd->shared_map, args.tensor(i).ref(), N, rfloor);
        }
    }
    tm_shared_publish(gd, N);  // sequencer: 0..N now appended
}

// -----------------------------------------------------------------------------
// Distributed submit op (replaces the centralized orchestrator submit).
//
// Every core runs this for every task (identical replay): materialize outputs
// at deterministic heap addresses, maintain the per-core producer map, then
// race to claim ownership. Only the winner builds the task into its private
// ring; losers return with map + outputs updated so downstream get_ref() and
// fan-in resolution stay consistent across cores.
// -----------------------------------------------------------------------------
__aicore__ TaskOutputTensors dist_submit_impl(__gm__ PTO2Runtime *rt, const MixedKernels &mixed, const L0TaskArgs &args) {
    DIST_DBG_SUBMIT_TICK(rt);  // C14 DIAG: count entries + record rt, before early returns
    __gm__ DistCore *self = pto_dist_self(rt);
    if (self == nullptr) return TaskOutputTensors{};
    __gm__ DistGlobal *gd = self->gd;
    __gm__ Runtime *runtime = gd->runtime;
    submit_crumb(runtime, self->core_idx, 40);  // C16b: submit entry

    // EXECUTE-FIRST (docs §6 step 0+1, §6.1): before claiming this task, pull any
    // follower deposits and execute every ready owned task. This interleaves
    // execution with claiming so a fast core does not burst-claim a full ring of
    // consecutive tasks; while it executes a (long) task other cores advance the
    // cursor and claim subsequent ones. The deterministic replay below (id bump,
    // heap bump, map maintenance) is unaffected — draining only runs/flags tasks
    // this core already owns. Every core does this on every submit point.
    //
    // Reset the lap cursor at entry so the runtime's spans never absorb the orch
    // round-trip between two submits — that time is USER orchestration code, not
    // runtime work, and would bias EfDrain if counted here. It is left un-timed on
    // purpose (a deliberate gap between submits, not a runtime span).
    TRACE_LAP_RESET(self);
    if (!fatal_set(gd)) {
        drain_block_won(self);
        submit_crumb(runtime, self->core_idx, 48);  // C16b: ef drain_block_won done, into phase_b
        drain_phase_b(self);
    }
    submit_crumb(runtime, self->core_idx, 41);  // C16b: execute-first drain done
    // Lap: the execute-first drain itself (deposits + ready owned kernels it ran).
    // Kernels show separately on the kernel sub-lane; this is the drain's own scan.
    TRACE_LAP(self, self->local_index, -1, TracePhase::EfDrain);

    const int32_t N = self->local_index++;
    const ActiveMask M = mixed.to_active_mask();
    const int32_t tc = args.tensor_count();
    if (N >= kFlagCap) {  // flag ring + vend[] are non-windowed; cap total tasks
        set_fatal(gd);
        fprintf(
            stderr, "[dist_engine] task id %d exceeds kFlagCap %d (enlarge or window the flag/vend rings)\n", N,
            kFlagCap
        );
        return TaskOutputTensors{};
    }

    // Publish this core's replay progress (BOTH modes) so the run-ahead balance
    // throttle + (shared) reclaim floor see the slowest core, then throttle so no
    // core races the claim cursor > runahead_max tasks ahead of the slowest (load
    // balance; deterministic replay below is unaffected — see dist_runahead_throttle).
    coherent_store(&gd->core_progress[self->core_idx], N, std::memory_order_relaxed);
    submit_crumb(runtime, self->core_idx, 42);  // C16b: about to run-ahead throttle
    dist_runahead_throttle(self, N);
    submit_crumb(runtime, self->core_idx, 43);  // C16b: throttle passed, entering alloc

    // (a) Deterministic GM output-heap allocation + materialization (§9.3, §11.4).
    // The virtual bump `heap_next` is unbounded and identical on every core; the
    // PHYSICAL address is (virtual mod ring). First sum this task's aligned output
    // bytes so we can keep the whole task within one ring lap: if it would straddle
    // the ring end, pad the virtual base up to the next ring boundary (deterministic
    // → every core agrees). A single task larger than the ring is unsatisfiable.
    const size_t ring = gd->heap_size;
    uint64_t total = 0;
    for (int32_t i = 0; i < tc; i++) {
        if (args.tag(i) != TensorArgType::OUTPUT) continue;
        total += PTO2_ALIGN_UP(args.tensor(i).create_info().buffer_size_bytes(), PTO2_PACKED_OUTPUT_ALIGN);
    }
    uint64_t task_base = PTO2_ALIGN_UP(self->heap_next, PTO2_PACKED_OUTPUT_ALIGN);
    if (total > 0 && gd->heap_base != nullptr) {
        if (total > ring) {
            set_fatal(gd);
            fprintf(
                stderr, "[dist_engine] task %d outputs %llu B exceed heap ring %zu B (enlarge PTO_DIST_HEAP_MB)\n", N,
                (unsigned long long)total, ring
            );
            return TaskOutputTensors{};
        }
        if ((task_base % ring) + total > ring) {
            task_base = ((task_base / ring) + 1) * ring;  // skip the ring tail; start next lap
        }
    }
    uint64_t off = 0;
    TaskOutputTensors result;
    for (int32_t i = 0; i < tc; i++) {
        if (args.tag(i) != TensorArgType::OUTPUT) continue;
        const TensorCreateInfo &ci = args.tensor(i).create_info();
        const uint64_t logical = ci.buffer_size_bytes();
        const uint64_t sz = PTO2_ALIGN_UP(logical, PTO2_PACKED_OUTPUT_ALIGN);
        // C16 DIAG: for task 0's first output, publish what dist_submit_impl reads
        // out of the (stack-resident?) create_info: its address + ndims + shape[0]
        // + logical bytes. Reveals whether the create_info pointer is stale (ndims
        // garbage) or the shape itself was built as 0.
        if (N == 0) {
            uintptr_t _cia = args.tensor(i).raw_addr();
            __gm__ volatile uint32_t *_pr =
                &gd->runtime->dist.aicore_progress[self->core_idx * AICORE_PROGRESS_STRIDE];
            _pr[9]  = static_cast<uint32_t>(_cia & 0xffffffffu);
            _pr[10] = static_cast<uint32_t>(_cia >> 32);
            _pr[11] = static_cast<uint32_t>(ci.ndims);
            _pr[12] = static_cast<uint32_t>(ci.shapes[0]);
            _pr[13] = static_cast<uint32_t>(logical);
            pto_dcci_flush(const_cast<__gm__ uint32_t *>(&_pr[9]), 5 * sizeof(uint32_t));
        }
        if (gd->heap_base == nullptr) {
            set_fatal(gd);
            fprintf(stderr, "[dist_engine] GM output heap not allocated at task %d\n", N);
            return result;
        }
        const uint64_t phys = (task_base + off) % ring;  // straddle-pad guarantees phys+logical <= ring
        __gm__ Tensor &slot_t = self->outpool[self->outpool_head];
        self->outpool_head = (self->outpool_head + 1) % kOutPoolSlots;
        init_tensor_from_create_info(slot_t, ci, gd->heap_base + phys, logical);
        result.materialize_output(slot_t);
        off += sz;
    }
    self->heap_next = task_base + off;
    // Publish cumulative virtual bytes through task N so any core can derive the
    // live window [vend[R], heap_next) for reclaim back-pressure. Deterministic, so
    // all cores store the same value (this core also reads its own writes for R<N).
    if (N >= 0 && N < kFlagCap) coherent_store(&gd->vend[N], self->heap_next, std::memory_order_relaxed);

    // Once fatal, stop claiming/executing but keep replaying the deterministic
    // allocation above so this task's `result` carries valid (materialized) output
    // refs — the orchestration may still call get_ref() on them. This degrades a
    // fatal (e.g. heap-too-small) into a clean wrong-answer failure + diagnostic
    // rather than an assertion crash mid-replay.
    if (fatal_set(gd)) return result;

    // Retire producer-map entries that have left the H span (deterministic,
    // N-derived) before this task's lookups/inserts. Bounds chain length so
    // submit stays ~O(N) instead of O(N^2). See DistTensorMap. In shared mode the
    // per-core map is unused (progress already published above for the shared
    // ring's global reclaim floor, docs §12.7.1).
    if (!gd->tm_shared)
        dist_tm_advance_retire(&self->map, N, gd->H);

    // (b) Anchor type + claim race FIRST — resolved from the mask alone (no map
    // ops, no Tensor copies). Deciding the winner up front lets the ~2/3 of cores
    // that fail type_match / lose the race SKIP the fan-in lookup below; they only
    // still perform the unconditional output insert (so every core's duplicate
    // TensorMap stays identical — §4). Competition is by anchor TYPE (§2/§3.1):
    // cube tasks (any AIC subtask) contested by AIC cores; vector tasks (AIV-only,
    // incl. 2V) by AIV cores. The cursor CAS touches no map state, so doing it
    // before the insert below does not affect the deterministic map replay.
    const uint8_t cmask = M.core_mask();
    const int32_t pc = __builtin_popcount(cmask);
    const bool has_aic = (cmask & PTO2_SUBTASK_MASK_AIC) != 0;
    const bool anchor_is_cube = has_aic;
    const bool type_match = anchor_is_cube ? (self->role == CoreType::AIC) : (self->role == CoreType::AIV);
    bool is_winner = false;
    if (type_match) {
        // Pick the shard for this task (§6.6): shard = N % kCursorShards, a pure
        // function of the task id so every core targets the same sub-cursor for N.
        __gm__ PaddedCursor *cursors = anchor_is_cube ? gd->cube_cursor : gd->vector_cursor;
        __gm__ Coherent<int32_t> *cursor = &cursors[N % kCursorShards].v;
        is_winner = claim(cursor, N);
    }

    // (c) Fan-in resolution — WINNER ONLY. Look up producers of INPUT/INOUT regions
    // BEFORE this task registers its own writes (so an INOUT does not self-match).
    // Losers never consume fanin, so they skip these lookups entirely; correctness
    // is unaffected because the map state read here is identical on every core and
    // only the owner needs the result.
    int32_t fanin[kMaxFanin];
    int32_t fc = 0;
    if (is_winner) {
        for (int32_t i = 0; i < tc; i++) {
            const TensorArgType tag = args.tag(i);
            if (tag != TensorArgType::INPUT && tag != TensorArgType::INOUT) continue;
            const Tensor &t = args.tensor(i).ref();
            if (t.manual_dep) continue;
            // Shared: query the one global ring with the same [N-H, N) window the
            // private replica uses (temporal filter + retire floor) => identical
            // producer. Private: query this core's replica (alive_floor already N-H).
            const int32_t p =
                gd->tm_shared ? shared_tm_lookup(&gd->shared_map, t, N, N - gd->H) : dist_tm_lookup(&self->map, t);
            if (p < 0) continue;
            bool dup = false;
            for (int32_t k = 0; k < fc; k++)
                if (fanin[k] == p) {
                    dup = true;
                    break;
                }
            if (!dup && fc < kMaxFanin) {
                fanin[fc++] = p;
                dep_sig_add(gd, N, p);  // graph-signature verification (env PTO_DIST_DEPSIG)
            }
        }
    }

    // (d) Register this task as the producer of its OUTPUT / INOUT / existing
    // outputs. Private: UNCONDITIONAL (every core writes its own replica, so all
    // stay identical — §4). Shared: EVERY core calls the sequencer, but only the
    // one that wins the in-id-order append actually writes the single global ring
    // (docs §12.4); it also blocks until 0..N are appended, so the ring is complete
    // for any later fan-in resolution => identical content to a private replica.
    if (gd->tm_shared) {
        tm_shared_append_task(self, N, args, result);
    } else {
        uint32_t out_idx = 0;
        for (int32_t i = 0; i < tc; i++) {
            const TensorArgType tag = args.tag(i);
            if (tag == TensorArgType::OUTPUT) {
                dist_tm_insert(&self->map, result.get_ref(out_idx), N);
                out_idx++;
            } else if (tag == TensorArgType::INOUT || tag == TensorArgType::OUTPUT_EXISTING) {
                dist_tm_insert(&self->map, args.tensor(i).ref(), N);
            }
        }
    }

    if (!is_winner) {
        TRACE_LAP(self, N, -1, TracePhase::Replay);
        return result;  // wrong type or lost the race: map updated, nothing to build
    }

    // (e) Winner only: assemble the shared argument Tensors (identical for every
    // active lane of a multi-core task — they share the task tensors, each lane
    // writing its designated output per the kernels). Inputs are copied from the
    // args; outputs are the materialized heap-addressed descriptors. Done AFTER
    // the claim so the ~2/3 of cores that fail type_match / lose the race never
    // pay these tc x sizeof(Tensor) copies.
    const uint64_t *scalars = args.scalars();
    const int32_t sc = args.scalar_count();
    Tensor built[MAX_TENSOR_ARGS];
    {
        uint32_t bo = 0;
        for (int32_t i = 0; i < tc; i++) {
            if (args.tag(i) == TensorArgType::OUTPUT) {
                built[i].copy(result.get_ref(bo));
                bo++;
            } else {
                built[i].copy(args.tensor(i).ref());
            }
        }
    }

    // ---- Winner = owner (single-core) / anchor (multi-core). ----
    // The real per-task build work (claim + fan-in lookup + built[] assembly)
    // ends here; the two back-pressure spins below are WAITING, not work, so
    // close the Build span now and time the spins separately as RingBp. Without
    // this split the spin time was misattributed to "build" (it dominated build
    // under a small ring / few blocks — it is dependency/slot wait, not cost).
    TRACE_LAP(self, N, -1, TracePhase::Build);

    // Back-pressure for self-claimed work: wait until the ring has a non-reserved
    // slot free, draining block.won deposits + ready tasks meanwhile. The reserve
    // guarantees a follower can still pull its (ready) deposits when the rest of
    // the ring is full of not-yet-ready consumers (no priority inversion).
    uint64_t wd_self = 0;
#if DIST_TRACE_ENABLED
    // Swimlane (slot-release edges): if we are about to actually wait, snapshot the
    // tasks currently occupying our ring — those are what must execute to free a
    // slot, i.e. what this ringbp truly waits on. The ring only shrinks during the
    // wait, so the entry snapshot is the complete set.
    if (g_trace_on && self->occupied_count >= kPrivateSlots - kWonReserve) {
        for (int32_t i = 0; i < kPrivateSlots; i++) {
            __gm__ const RingSlot &rs = self->slots[i];
            if (rs.occupied && rs.built) self->slot_edges.push_back({N, rs.task_id});
        }
    }
#endif
    submit_crumb(runtime, self->core_idx, 44);  // C16b: entering ring slot back-pressure
    while (self->occupied_count >= kPrivateSlots - kWonReserve && !fatal_set(gd)) {
        drain_block_won(self);
        if (drain_phase_b(self) == 0) {
            SPIN_WAIT_HINT();
            watchdog(gd, wd_self);
        }
    }
    submit_crumb(runtime, self->core_idx, 45);  // C16b: ring back-pressure cleared
    if (fatal_set(gd)) return result;

    // Heap reclaim back-pressure (§9.5/§11.4): this owner is about to build (and
    // later write) task N's outputs at deterministic physical offsets. Recycling a
    // ring region is safe only once its previous occupant's task id <= R = F - H
    // (all that occupant's consumers, which have id <= occupant+H, are done). The
    // equivalent global-derivable test is: the live virtual window (heap_next minus
    // vend[R]) must fit in the ring. Spin (draining + advancing F) until it does.
    submit_crumb(runtime, self->core_idx, 46);  // C16b: entering heap reclaim back-pressure
    if (gd->heap_base != nullptr) {
        const size_t ring = gd->heap_size;
        uint64_t wd_heap = 0;
        while (!fatal_set(gd)) {
            const int32_t f = coherent_load(&gd->frontier, std::memory_order_acquire);
            const int32_t R = f - gd->H;
            const uint64_t vstart_live = (R < 0) ? 0 : coherent_load(&gd->vend[R], std::memory_order_relaxed);
            if (self->heap_next - vstart_live <= ring) break;  // window fits — region free
            if (f >= N - 1) {  // every predecessor done yet H-window still overflows the ring
                set_fatal(gd);
                fprintf(
                    stderr,
                    "[dist_engine] heap ring %zu B too small for H=%d window at task %d (live=%llu B); "
                    "enlarge PTO_DIST_HEAP_MB or reduce PTO_DIST_H\n",
                    ring, gd->H, N, (unsigned long long)(self->heap_next - vstart_live)
                );
                return result;
            }
            drain_block_won(self);
            if (drain_phase_b(self) == 0) {
                SPIN_WAIT_HINT();
                watchdog(gd, wd_heap);
            }
        }
        if (fatal_set(gd)) return result;
    }
    submit_crumb(runtime, self->core_idx, 47);  // C16b: heap back-pressure cleared, building slot
    // Time spent in the two back-pressure spins above (ring-slot wait + heap
    // reclaim wait) — dependency/slot WAITING, kept separate from Build.
    TRACE_LAP(self, N, -1, TracePhase::RingBp);

    int32_t si = alloc_ring_slot(self);
    if (si < 0) {  // should not happen given the back-pressure gate above
        set_fatal(gd);
        fprintf(stderr, "[dist_engine] no free private-ring slot after back-pressure at task %d\n", N);
        return result;
    }
    // Reserve so concurrent drains (including the block.won back-pressure loop
    // below, which calls drain_phase_b) do not reuse this slot. Mark it unbuilt
    // so Phase B skips it until build_ring_slot populates it (avoids re-executing
    // the prior occupant's stale task_id/fanin/won linkage).
    self->slots[si].occupied = true;
    self->slots[si].built = false;

    int32_t own_lane;
    int32_t won_block = -1;
    int32_t won_slot = -1;
    bool is_multicore = (pc > 1);

    if (!is_multicore) {
        // Single core (1C / 1V): the one active lane is the only subtask. For 1V
        // the winner may be physically AIV0 or AIV1, but the active lane/kernel is
        // AIV0 (rt_submit_aiv fills aiv0). Find the single active lane.
        own_lane = has_aic ? LANE_AIC : LANE_AIV0;
    } else {
        // Multi-core (MIX / 2V): we are the anchor. Our own physical lane subtask
        // goes to our private ring; the remaining active lanes are deposited into
        // block.won for our same-block followers to drain (§3.1).
        own_lane = self->lane;
        won_block = self->block_id;
        won_slot = alloc_won_slot(gd, won_block);
        uint64_t wd_won = 0;
        while (won_slot < 0 && !fatal_set(gd)) {  // block.won full → back-pressure (drain, then retry)
            drain_block_won(self);
            if (drain_phase_b(self) == 0) {
                SPIN_WAIT_HINT();
                watchdog(gd, wd_won);
            }
            won_slot = alloc_won_slot(gd, won_block);
        }
        if (fatal_set(gd)) return result;
        __gm__ WonSlot &w = gd->blocks[won_block].slots[won_slot];
        w.task_id = N;
        coherent_store(&w.remaining, pc, std::memory_order_relaxed);
        for (int32_t L = 0; L < PTO2_SUBTASK_SLOT_COUNT; L++) {
            coherent_store(&w.drained[L], 0, std::memory_order_relaxed);
            w.lane[L].present = false;
        }
        for (int32_t L = 0; L < PTO2_SUBTASK_SLOT_COUNT; L++) {
            if (L == own_lane || !lane_active(M, L)) continue;
            __gm__ BuiltSubtask &b = w.lane[L];
            b.present = true;
            b.func_id = kernel_id_for_lane(mixed, L);
            b.function_bin_addr = resolve_kernel_addr(runtime, kernel_id_for_lane(mixed, L));
            b.tensor_count = tc;
            b.scalar_count = sc;
            for (int32_t i = 0; i < tc; i++)
                tensor_copy(&b.tensors[i], built[i]);
            for (int32_t j = 0; j < sc; j++)
                b.scalars[j] = scalars[j];
            b.fanin_count = fc;
            for (int32_t k = 0; k < fc; k++)
                b.fanin[k] = fanin[k];
            b.sub_block_id = (L == LANE_AIV1) ? 1 : 0;
        }
        pto_shared_fence(std::memory_order_release);
        coherent_store(&gd->blocks[won_block].any_pub, 1, std::memory_order_release);  // enable follower drains
        coherent_store(&w.state, 1, std::memory_order_release);                           // publish the deposits to followers
    }

    const int32_t own_sub_block = (own_lane == LANE_AIV1) ? 1 : 0;
    const int32_t own_func_id = kernel_id_for_lane(mixed, own_lane);
    build_ring_slot(
        &self->slots[si], N, own_func_id, resolve_kernel_addr(runtime, own_func_id), built, tc, scalars, sc, fanin, fc,
        own_sub_block, is_multicore, won_block, won_slot
    );
    self->occupied_count++;
    self->owned_total++;

#if DIST_TRACE_ENABLED
    if (g_trace_on) {
        for (int32_t k = 0; k < fc; k++)
            self->dep_edges.push_back({N, fanin[k]});
    }
#endif
    TRACE_LAP(self, N, -1, TracePhase::Commit);
    return result;
}

// -----------------------------------------------------------------------------
// Remaining ops — minimal stubs (bgemm exercises submit/scope/log only).
// -----------------------------------------------------------------------------
__aicore__ void dist_scope_begin(__gm__ PTO2Runtime *) {}
__aicore__ void dist_scope_end(__gm__ PTO2Runtime *) {}
__aicore__ void dist_orchestration_done(__gm__ PTO2Runtime *) {}
__aicore__ bool dist_is_fatal(__gm__ PTO2Runtime *rt) {
    __gm__ DistGlobal *gd = pto_gd(rt);
    return gd != nullptr && fatal_set(gd);
}

#if DIST_SIM_HOST_CLOCK
void dist_report_fatal(__gm__ PTO2Runtime *rt, int32_t code, const char *func, const char *fmt, ...) {
    if (__gm__ DistGlobal *gd = pto_gd(rt)) set_fatal(gd);
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "[dist_engine][FATAL][%s] code=%d: ", func ? func : "?", code);
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
}
// AICore has no va_list/fprintf and forbids variadic functions entirely; fatal
// reporting on AICore goes through set_fatal/fatal_set (both __aicore__). The
// public dist_report_fatal symbol is host/sim-only (no aicore caller).
#endif

#if DIST_SIM_HOST_CLOCK
void dist_log_error(const char *func, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "[dist_engine][E][%s] ", func ? func : "?");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
}
void dist_log_warn(const char *, const char *, ...) {}
void dist_log_debug(const char *, const char *, ...) {}
void dist_log_info_v(const char *, int, const char *, ...) {}
#endif

// Orchestration-side tensor data access (get/set_tensor_data). Replay runs on the
// AICore worker and reads/writes real GM, so these are genuine memory accesses.
// The only subtlety is read-after-write across tasks: if the region has a producer
// in this core's map, wait until that producer's completion flag is set (draining
// this core's own ring meanwhile so an owned producer actually runs). External
// tensors (no producer) are accessed immediately. Consumer (WAR) tracking is not
// modeled, mirroring the centralized runtime's documented INPUT-reader limitation.
__aicore__ void wait_producer_ready(__gm__ DistCore *self, const Tensor &t) {
    __gm__ DistGlobal *gd = self->gd;
    // Cold path (get/set_tensor_data); uses the map's current alive_floor. In
    // shared mode query the global ring with this core's [N-H, N) window (N =
    // current replay position); an over-old hit only waits on an already-set flag.
    const int32_t N = self->local_index;
    const int32_t p =
        gd->tm_shared ? shared_tm_lookup(&gd->shared_map, t, N, N - gd->H) : dist_tm_lookup(&self->map, t);
    if (p < 0) return;
    uint64_t wd = 0;
    while (!fatal_set(gd)) {
        if (coherent_load(&gd->flags[p & (kFlagCap - 1)], std::memory_order_acquire) != 0) break;
        drain_block_won(self);
        if (drain_phase_b(self) == 0) {
            SPIN_WAIT_HINT();
            watchdog(gd, wd);
        }
    }
}

__aicore__ uint64_t dist_get_tensor_data(__gm__ PTO2Runtime *rt, const Tensor &tensor, uint32_t ndims, const uint32_t *indices) {
    if (tensor.buffer.addr == 0) return 0;
    __gm__ DistCore *self = pto_dist_self(rt);
    if (self != nullptr) wait_producer_ready(self, tensor);
    const uint64_t flat = tensor.compute_flat_offset(indices, ndims);
    const uint64_t esz = get_element_size(tensor.dtype);
    uint64_t result = 0;
    memcpy(&result, reinterpret_cast<const void *>(tensor.buffer.addr + flat * esz), esz);
    return result;
}

__aicore__ void dist_set_tensor_data(
    __gm__ PTO2Runtime *rt, const Tensor &tensor, uint32_t ndims, const uint32_t *indices, uint64_t value
) {
    if (tensor.buffer.addr == 0) return;
    __gm__ DistCore *self = pto_dist_self(rt);
    if (self != nullptr) wait_producer_ready(self, tensor);
    const uint64_t flat = tensor.compute_flat_offset(indices, ndims);
    const uint64_t esz = get_element_size(tensor.dtype);
    memcpy(reinterpret_cast<void *>(tensor.buffer.addr + flat * esz), &value, esz);
}

// alloc_tensors — a kernel-less "hidden task" that only reserves GM output
// buffers (no compute). It consumes one task id, allocates its outputs on the
// deterministic heap exactly like dist_submit_impl step (a), registers itself as
// their producer, and completes INLINE (sets its own flag immediately) since no
// kernel runs. A later writer (INOUT / OUTPUT_EXISTING) becomes the new producer
// of the region, so real consumers depend on the writer, not on this alloc. Every
// core replays it identically, keeping heap addresses + maps consistent.
__aicore__ TaskOutputTensors dist_alloc_tensors(__gm__ PTO2Runtime *rt, const L0TaskArgs &args) {
    __gm__ DistCore *self = pto_dist_self(rt);
    if (self == nullptr) return TaskOutputTensors{};
    __gm__ DistGlobal *gd = self->gd;
    // EXECUTE-FIRST (docs §6 step 0+1, §6.1): every submit point first seeks an
    // execution opportunity before advancing the deterministic replay below.
    TRACE_LAP_RESET(self);  // exclude the inter-submit orch round-trip (user code) from runtime spans
    if (!fatal_set(gd)) {
        drain_block_won(self);
        drain_phase_b(self);
    }
    TRACE_LAP(self, self->local_index, -1, TracePhase::EfDrain);
    const int32_t N = self->local_index++;
    const int32_t tc = args.tensor_count();
    if (N >= kFlagCap) {
        set_fatal(gd);
        fprintf(stderr, "[dist_engine] alloc task id %d exceeds kFlagCap %d\n", N, kFlagCap);
        return TaskOutputTensors{};
    }

    // Publish progress + run-ahead balance throttle (BOTH modes; see dist_submit_impl).
    coherent_store(&gd->core_progress[self->core_idx], N, std::memory_order_relaxed);
    dist_runahead_throttle(self, N);

    // Deterministic GM heap allocation + straddle-padding (identical to submit (a)).
    const size_t ring = gd->heap_size;
    uint64_t total = 0;
    for (int32_t i = 0; i < tc; i++) {
        if (args.tag(i) != TensorArgType::OUTPUT) continue;
        total += PTO2_ALIGN_UP(args.tensor(i).create_info().buffer_size_bytes(), PTO2_PACKED_OUTPUT_ALIGN);
    }
    uint64_t task_base = PTO2_ALIGN_UP(self->heap_next, PTO2_PACKED_OUTPUT_ALIGN);
    if (total > 0 && gd->heap_base != nullptr) {
        if (total > ring) {
            set_fatal(gd);
            fprintf(
                stderr, "[dist_engine] alloc task %d outputs %llu B exceed heap ring %zu B\n", N,
                (unsigned long long)total, ring
            );
            return TaskOutputTensors{};
        }
        if ((task_base % ring) + total > ring) task_base = ((task_base / ring) + 1) * ring;
    }

    // (a) Materialize outputs + publish the deterministic heap layout — EVERY core
    // (like dist_submit_impl step (a)), so duplicate maps and vend[] stay identical.
    uint64_t off = 0;
    TaskOutputTensors result;
    for (int32_t i = 0; i < tc; i++) {
        if (args.tag(i) != TensorArgType::OUTPUT) continue;
        const TensorCreateInfo &ci = args.tensor(i).create_info();
        const uint64_t logical = ci.buffer_size_bytes();
        const uint64_t sz = PTO2_ALIGN_UP(logical, PTO2_PACKED_OUTPUT_ALIGN);
        if (gd->heap_base == nullptr) {
            set_fatal(gd);
            fprintf(stderr, "[dist_engine] GM output heap not allocated at alloc %d\n", N);
            return result;
        }
        const uint64_t phys = (task_base + off) % ring;
        __gm__ Tensor &slot_t = self->outpool[self->outpool_head];
        self->outpool_head = (self->outpool_head + 1) % kOutPoolSlots;
        init_tensor_from_create_info(slot_t, ci, gd->heap_base + phys, logical);
        result.materialize_output(slot_t);
        off += sz;
    }
    self->heap_next = task_base + off;
    if (N >= 0 && N < kFlagCap) coherent_store(&gd->vend[N], self->heap_next, std::memory_order_relaxed);
    if (fatal_set(gd)) return result;

    // (b) Register this alloc as producer of each output. Private: EVERY core
    // writes its replica (map parity). Shared: the in-id-order sequencer appends
    // once to the single global ring (docs §12.4), same as dist_submit_impl (d).
    if (gd->tm_shared) {
        tm_shared_append_task(self, N, args, result);  // progress published above
    } else {
        dist_tm_advance_retire(&self->map, N, gd->H);
        uint32_t out_idx = 0;
        for (int32_t i = 0; i < tc; i++) {
            if (args.tag(i) != TensorArgType::OUTPUT) continue;
            dist_tm_insert(&self->map, result.get_ref(out_idx), N);
            out_idx++;
        }
    }

    // (c) Single-owner election (mirrors dist_submit_impl's claim). The first core
    // to reach this alloc id wins; that core is by construction at/ahead of the
    // completion frontier (N is not yet done, so F < N), hence the winner-only
    // back-pressure below can never see heap_next < vend[F-H] and never underflows.
    // Losers have finished the deterministic bookkeeping above and return — the
    // winner alone paces reclaim and publishes the completion flag (the leading
    // core was the one gating completion before this change too, so timing is
    // unchanged; this only drops the lagging cores' redundant pass).
    bool is_winner = claim(&gd->alloc_cursor[N % kCursorShards].v, N);
    if (!is_winner) {
        TRACE_LAP(self, N, -1, TracePhase::Replay);
        return result;
    }

    // (d) Winner-only heap reclaim back-pressure: drain this core's ring while the
    // live virtual window [vend[F-H], heap_next) would overflow the physical ring.
    if (total > 0 && gd->heap_base != nullptr) {
        uint64_t wd_heap = 0;
        while (!fatal_set(gd)) {
            const int32_t f = coherent_load(&gd->frontier, std::memory_order_acquire);
            const int32_t R = f - gd->H;
            const uint64_t vstart_live = (R < 0) ? 0 : coherent_load(&gd->vend[R], std::memory_order_relaxed);
            if (self->heap_next - vstart_live <= ring) break;  // window fits — region free
            if (f >= N - 1) {
                set_fatal(gd);
                fprintf(
                    stderr, "[dist_engine] heap ring %zu B too small for H=%d window at alloc %d (live=%llu B)\n", ring,
                    gd->H, N, (unsigned long long)(self->heap_next - vstart_live)
                );
                return result;
            }
            drain_block_won(self);
            if (drain_phase_b(self) == 0) {
                SPIN_WAIT_HINT();
                watchdog(gd, wd_heap);
            }
        }
        if (fatal_set(gd)) return result;
    }

    // (e) Winner completes inline (no kernel runs).
    dist_set_flag(&gd->flags[N & (kFlagCap - 1)]);
    advance_frontier(gd);
    TRACE_LAP(self, N, -1, TracePhase::Alloc);
    return result;
}

#if DIST_SIM_HOST_CLOCK
// The ops table and registration entry are host/AICPU-only: they reference the
// variadic dist_log_* / dist_report_fatal diagnostics (CCEC has no varargs on
// AICore) and are installed by dist_engine_register from the AICPU executor.
TaskOutputTensors dist_submit_dummy(__gm__ PTO2Runtime *, const L0TaskArgs &) { return TaskOutputTensors{}; }
void dist_scope_set_site(const char *, int) {}

const PTO2RuntimeOps g_dist_ops = {
    dist_submit_impl,     dist_scope_begin,     dist_scope_end,     dist_orchestration_done, dist_is_fatal,
    dist_report_fatal,    dist_log_error,       dist_log_warn,      dist_log_debug,          dist_log_info_v,
    dist_get_tensor_data, dist_set_tensor_data, dist_alloc_tensors, dist_submit_dummy,       dist_scope_set_site,
};
#else
// AICore (CCEC) ops table. CCEC forbids variadic functions and has no
// fprintf/va_list, so the log slots are non-variadic no-op stubs cast to the
// variadic function-pointer field type (aarch64 ABI makes this safe — the
// caller's extra varargs registers are simply ignored by the callee). fatal
// reporting goes through set_fatal (coherent_store to gd->fatal), which is the
// only effect the engine observes; the AICore has no stderr to print to anyway.
// submit_task / alloc_tensors / get_tensor_data / set_tensor_data are the real
// __aicore__ engine hot-path functions (already attributed). This table is
// installed on the AICore by dist_core_main_impl before replaying the user
// orchestration, so current_runtime()->ops->submit_task resolves to the AICore
// engine — not the AICPU's (host-address) g_dist_ops.
__aicore__ TaskOutputTensors dist_submit_dummy_aicore(__gm__ PTO2Runtime *, const L0TaskArgs &) { return TaskOutputTensors{}; }
__aicore__ void dist_scope_set_site_aicore(const char *, int) {}
__aicore__ void dist_report_fatal_aicore(__gm__ PTO2Runtime *rt, int32_t /*code*/, const char * /*func*/, const char * /*fmt*/) {
    if (__gm__ DistGlobal *gd = pto_gd(rt)) set_fatal(gd);
}
// Non-variadic stubs for the log slots. The ops-table field type is variadic
// (void(*)(const char*, const char*, ...)); we cast these fixed-arg no-ops to
// that type. On AICore there is no log sink, so logging is a true no-op; the
// only diagnostic channel is gd->fatal (set by report_fatal above) which the
// engine polls via fatal_set().
__aicore__ void dist_log_error_aicore(const char *, const char *) {}
__aicore__ void dist_log_warn_aicore(const char *, const char *) {}
__aicore__ void dist_log_debug_aicore(const char *, const char *) {}
__aicore__ void dist_log_info_v_aicore(const char *, int, const char *) {}

// NON-const on the AICore build: the function-pointer fields MUST be
// (re)assigned at runtime by dist_ops_refresh_aicore() below. A static
// initializer emits each pointer as a link-time (base-0) address plus an ELF
// RELATIVE relocation that the loader would normally fix up with the load
// base — but the incore AICore loader does NOT apply relocations (it rejects
// .rela.text; .data.rel.ro RELATIVE fixups are likewise never applied). So the
// baked-in values stay base-0 and calling through them (e.g. the user
// orchestration's current_runtime()->ops->submit_task) branches to an
// unmapped low address → MPU fault (aivec error 271 → stream 507000, observed
// as every core stalling right before the orchestration replay). Re-taking
// &func on-core emits a PC-relative adrp/add that yields the REAL loaded VA, so
// dist_ops_refresh_aicore() overwrites every slot before the table is used.
PTO2RuntimeOps g_dist_ops = {
    dist_submit_impl,
    dist_scope_begin,
    dist_scope_end,
    dist_orchestration_done,
    dist_is_fatal,
    // Non-variadic stub → variadic field: aarch64 ABI drops the extra varargs,
    // so the cast is sound. CCEC forbids variadic __aicore__ functions, so the
    // stub is fixed-arg and cast here.
    reinterpret_cast<void (*)(__gm__ PTO2Runtime *, int32_t, __gm__ const char *, __gm__ const char *, ...)>(dist_report_fatal_aicore),
    reinterpret_cast<void (*)(__gm__ const char *, __gm__ const char *, ...)>(dist_log_error_aicore),
    reinterpret_cast<void (*)(__gm__ const char *, __gm__ const char *, ...)>(dist_log_warn_aicore),
    reinterpret_cast<void (*)(__gm__ const char *, __gm__ const char *, ...)>(dist_log_debug_aicore),
    reinterpret_cast<void (*)(__gm__ const char *, int, __gm__ const char *, ...)>(dist_log_info_v_aicore),
    dist_get_tensor_data,
    dist_set_tensor_data,
    dist_alloc_tensors,
    dist_submit_dummy_aicore,
    dist_scope_set_site_aicore,
};

// Rewrite every g_dist_ops slot with a PC-relative (runtime-loaded) function
// address. See the comment on g_dist_ops: the static initializer's addresses
// are base-0 (unrelocated by the incore loader); taking &func here forces CCEC
// to materialize the address relative to the kernel's load base, so the table
// becomes callable. Idempotent — every worker runs it and writes identical
// values (all cores share one image base), so the repeated stores are benign.
__aicore__ void dist_ops_refresh_aicore(__gm__ PTO2RuntimeOps *o) {
    o->submit_task = dist_submit_impl;
    o->scope_begin = dist_scope_begin;
    o->scope_end = dist_scope_end;
    o->orchestration_done = dist_orchestration_done;
    o->is_fatal = dist_is_fatal;
    o->report_fatal =
        reinterpret_cast<void (*)(__gm__ PTO2Runtime *, int32_t, __gm__ const char *, __gm__ const char *, ...)>(dist_report_fatal_aicore);
    o->log_error = reinterpret_cast<void (*)(__gm__ const char *, __gm__ const char *, ...)>(dist_log_error_aicore);
    o->log_warn = reinterpret_cast<void (*)(__gm__ const char *, __gm__ const char *, ...)>(dist_log_warn_aicore);
    o->log_debug = reinterpret_cast<void (*)(__gm__ const char *, __gm__ const char *, ...)>(dist_log_debug_aicore);
    o->log_info_v = reinterpret_cast<void (*)(__gm__ const char *, int, __gm__ const char *, ...)>(dist_log_info_v_aicore);
    o->get_tensor_data = dist_get_tensor_data;
    o->set_tensor_data = dist_set_tensor_data;
    o->alloc_tensors = dist_alloc_tensors;
    o->submit_dummy_task = dist_submit_dummy_aicore;
    o->scope_set_site = dist_scope_set_site_aicore;
}
#endif

// -----------------------------------------------------------------------------
// Deadlock diagnostics: dump the full engine state on SIGUSR1. Sim runs every
// core as a pthread in one process, so a single handler can walk gd-> Used to
// debug hangs (kill -USR1 <pid>); compiled in but inert unless signalled.
// -----------------------------------------------------------------------------
#if DIST_SIM_HOST_CLOCK
void dist_dump_state(int) {
    __gm__ DistGlobal *gd = s_dump_gd;  // diagnostics-only handle (docs §13); not the functional path
    if (gd == nullptr) return;
    fprintf(stderr, "\n===== DIST STATE DUMP =====\n");
    fprintf(
        stderr, "frontier=%d H=%d ring=%zuB replay_done=%d/%d num_blocks=%d fatal=%d\n", coherent_load(&gd->frontier),
        gd->H, gd->heap_size, coherent_load(&gd->replay_done), gd->num_workers, gd->num_blocks,
        coherent_load(&gd->fatal)
    );
    fprintf(stderr, "cube_cursor[%d]=", kCursorShards);
    for (int32_t s = 0; s < kCursorShards; s++)
        fprintf(stderr, "%d%s", coherent_load(&gd->cube_cursor[s].v), s + 1 < kCursorShards ? "," : "");
    fprintf(stderr, " vector_cursor[%d]=", kCursorShards);
    for (int32_t s = 0; s < kCursorShards; s++)
        fprintf(stderr, "%d%s", coherent_load(&gd->vector_cursor[s].v), s + 1 < kCursorShards ? "," : "");
    fprintf(stderr, "\n");
    for (int32_t c = 0; c < gd->num_workers && c < RUNTIME_MAX_WORKER; c++) {
        DistCore &co = gd->cores[c];
        fprintf(
            stderr, "core %d role=%d blk=%d lane=%d replayed=%d occ=%d owned=%d\n", c, static_cast<int>(co.role),
            co.block_id, co.lane, co.local_index, co.occupied_count, co.owned_total
        );
        for (int32_t i = 0; i < kPrivateSlots; i++) {
            __gm__ RingSlot &s = co.slots[i];
            if (!s.occupied) continue;
            int32_t unmet = -1;
            for (int32_t f = 0; f < s.fanin_count; f++)
                if (coherent_load(&gd->flags[s.fanin[f] & (kFlagCap - 1)]) == 0) {
                    unmet = s.fanin[f];
                    break;
                }
            fprintf(
                stderr, "    slot%d tid=%d built=%d mc=%d won=(%d,%d) fanin=%d unmet=%d\n", i, s.task_id, s.built,
                s.is_multicore, s.won_block, s.won_slot, s.fanin_count, unmet
            );
        }
    }
    for (int32_t b = 0; b < gd->num_blocks; b++) {
        for (int32_t i = 0; i < kPrivateSlots; i++) {
            __gm__ WonSlot &w = gd->blocks[b].slots[i];
            int32_t st = coherent_load(&w.state);
            if (st == 0) continue;
            fprintf(
                stderr, "  won blk%d slot%d state=%d tid=%d remaining=%d drained=[%d,%d,%d] present=[%d,%d,%d]\n", b, i,
                st, w.task_id, coherent_load(&w.remaining), coherent_load(&w.drained[0]), coherent_load(&w.drained[1]), coherent_load(&w.drained[2]),
                w.lane[0].present, w.lane[1].present, w.lane[2].present
            );
        }
    }
    fprintf(stderr, "===== END DUMP =====\n");
}
#endif

// -----------------------------------------------------------------------------
// Per-core entry point invoked by each AICore worker thread.
//
// This is the engine hot path. It runs ON the AICore worker (sim: a host thread
// via the core_main_fn pointer; onboard: CCEC-compiled into the AICore binary).
// On the onboard AICPU-only build (SIMPLER_DIST_AICPU_ONLY) this function is
// never invoked (the AICore binary has its own CCEC copy), but it is still
// compiled so every helper it references stays "used" — otherwise the aarch64
// -Werror=unused-function would fire on each transitively-unused helper.
// __attribute__((used)) suppresses that warning without changing codegen.
// -----------------------------------------------------------------------------
// DEBUG breadcrumb: publish a per-core stage marker into Runtime::dist so the
// AICPU orchestrator's waiting-loop log can localize where a core faults/stalls
// inside dist_core_main. Onboard (AICore CCEC) writes + dcci-flushes the line so
// the AICPU's invalidate-read sees it; sim/AICPU builds no-op. Values chosen to
// not collide with aicore_executor's 11 (pre-call) / 12 (post-return).
#if !DIST_SIM_HOST_CLOCK
// The compiler barrier ("memory" clobber) is LOAD-BEARING for diagnostics: at
// -O3 CCEC freely hoists a later (faulting) memory access ahead of these
// volatile crumb stores, so the last-retired crumb no longer marks the last
// safe statement (observed: dbg written right after crumb 90 never landed
// because the faulting deref was scheduled between them). The barrier pins
// program order across every crumb, making the crumb sequence a faithful
// bisection marker.
#define DIST_CRUMB(rt, idx, val)                                                            \
    do {                                                                                    \
        (rt)->dist.aicore_progress[(idx) * AICORE_PROGRESS_STRIDE] = (val);                 \
        dcci(&(rt)->dist.aicore_progress[(idx) * AICORE_PROGRESS_STRIDE], SINGLE_CACHE_LINE, \
             CACHELINE_OUT);                                                                \
        __asm__ __volatile__("" ::: "memory");                                              \
    } while (0)
// Companion to DIST_CRUMB: stash a 64-bit diagnostic value for THIS core in the
// spare u32 slots of its own crumb cache line (slot 0 = crumb, slots 1/2 = dbg
// lo/hi). Reuses the same proven-writable line — a separately-appended array
// faulted on-core because the device Runtime buffer does not map fields past
// aicore_progress. Same line + same dcci as the crumb, so no false sharing.
#define DIST_DBG(rt, idx, val64)                                                          \
    do {                                                                                  \
        (rt)->dist.aicore_progress[(idx) * AICORE_PROGRESS_STRIDE + 1] =                  \
            (uint32_t)((uint64_t)(val64) & 0xffffffffu);                                  \
        (rt)->dist.aicore_progress[(idx) * AICORE_PROGRESS_STRIDE + 2] =                  \
            (uint32_t)(((uint64_t)(val64) >> 32) & 0xffffffffu);                          \
        dcci(&(rt)->dist.aicore_progress[(idx) * AICORE_PROGRESS_STRIDE], SINGLE_CACHE_LINE, \
             CACHELINE_OUT);                                                              \
        __asm__ __volatile__("" ::: "memory");                                           \
    } while (0)
#else
#define DIST_CRUMB(rt, idx, val) ((void)0)
#define DIST_DBG(rt, idx, val64) ((void)0)
#endif

// DIAG bisection knobs (defined here so the preprocessor sees them before the
// first use inside dist_core_main_impl). DIST_DIAG_SKIP_ORCH skips the user
// orchestration replay; DIST_DIAG_SKIP_BIND additionally skips
// framework_bind_runtime + dist_ops_refresh_aicore so we can localize whether
// the async MPU/MTE 271 fault originates in those (incore-loaded, possibly
// unrelocated) helpers vs. later in the replayed orchestration/kernels.
#ifndef DIST_DIAG_SKIP_ORCH
#define DIST_DIAG_SKIP_ORCH 0
#endif
#ifndef DIST_DIAG_SKIP_BIND
#define DIST_DIAG_SKIP_BIND 0
#endif
#ifndef DIST_DIAG_SKIP_OPS
#define DIST_DIAG_SKIP_OPS 0
#endif

__attribute__((used)) __aicore__ void dist_core_main_impl(__gm__ void *runtime_v, int core_idx, int core_type_int) {
    if (core_idx < 0 || core_idx >= RUNTIME_MAX_WORKER) return;
    __gm__ Runtime *runtime = reinterpret_cast<__gm__ Runtime *>(runtime_v);
    // Recover the shared segment from the arg-passed base (docs §13, 方案B): no
    // process-global. On HW `core_idx` is the hardware core id; record it so the
    // ops callbacks (which only get PTO2Runtime*) can recover this same core via
    // pto_core_id(). `gd`/`self` are threaded as locals from here on.
    DIST_CRUMB(runtime, core_idx, 20);  // entered dist_core_main
    __gm__ DistGlobal *gd = reinterpret_cast<__gm__ DistGlobal *>(runtime->dist.global_data_base);
    if (gd == nullptr) return;
    DIST_CRUMB(runtime, core_idx, 21);  // gd recovered (deref of global_data_base ok)
    // DIAG: single SCALAR read of the DistGlobal segment base, with the read value
    // stashed in this core's crumb line. If crumb 41 lands (and dbg shows a sane
    // word), a scalar load of gd works and the 271 comes from a later bulk (MTE) or
    // atomic access. If it does NOT land (progress stuck at 21), the reserve base
    // is fundamentally unreachable by the AICore for *any* access.
    {
        __gm__ volatile uint32_t *_segp = reinterpret_cast<__gm__ volatile uint32_t *>(runtime->dist.global_data_base);
        pto_dcci_inval(_segp, 64);  // DIAG: invalidate this core's stale cache line first
        volatile uint32_t _seg0 = *_segp;
        (void)_seg0;
        DIST_DBG(runtime, core_idx, (uint64_t)_seg0);
        DIST_CRUMB(runtime, core_idx, 41);
    }
    pto_set_core_id(core_idx);
    DIST_DBG_SUBMIT_RESET();  // C14 DIAG: zero the per-core submit probes (block_local is not zero-init)
    __gm__ DistCore *self = &gd->cores[core_idx];
    // Re-derive self->gd from THIS core's already-validated local `gd` (recovered
    // from runtime->dist.global_data_base and confirmed readable by crumbs 91/92).
    // register() set cores[i].gd on the AICPU and flushed it, but the AICore reads
    // that field with a plain load (Coherent<T> only wraps atomics, not this raw
    // back-pointer) and on cacheable HBM may observe a STALE cross-core value —
    // e.g. a prior run's segment base after g_segment is re-carved. A stale self->gd
    // makes drain_block_won's `gd->blocks[self->block_id]` a wild address → MPU 271
    // (observed on AIV followers, docs §17). Writing it here on-core makes it a
    // same-core value equal to the validated local gd, so every later self->gd read
    // is locally coherent. Each core writes only its OWN cores[core_idx].gd (DistCore
    // is multi-MiB, so no cross-core cache-line sharing).
    self->gd = gd;
    const CoreType role = static_cast<CoreType>(core_type_int);

    // sub_block lane: only meaningful for AIV in MIX tasks (M3). bgemm's 1V add
    // ignores it, so 0 is correct for the M2 single-core scope.
    // Read fields directly: copying a __gm__ CoreLayout into a generic local would
    // need a __gm__-aware copy ctor (CCEC forbids the implicit host-attributed one).
    __gm__ const CoreLayout *lay = &gd->layout[core_idx];
    DIST_CRUMB(runtime, core_idx, 22);  // read layout ok, about to reset core
    dist_core_reset(self, role, lay->block_id, lay->lane, gd->ring_cap);
    self->core_idx = core_idx;
    DIST_CRUMB(runtime, core_idx, 23);  // core reset done, entering startup barrier
    if (dist_trace())
        fprintf(
            stderr, "[dist] core %d role=%d block=%d lane=%d START\n", core_idx, core_type_int, lay->block_id, lay->lane
        );

    // Startup barrier: wait until every worker thread has been scheduled in and
    // reached this point before anyone begins replay. In sim the OS brings the
    // host threads up one at a time, so without this the cores that start early
    // race ahead and the swimlane's first-task stagger reflects thread-wakeup
    // skew rather than engine scheduling. Bare spin (no yield) per the AICPU
    // spin-wait convention. Skipped under fatal so a failed run still tears down.
    if (!fatal_set(gd)) {
        int32_t _prev = coherent_fetch_add(&gd->started_count, 1, std::memory_order_acq_rel);
        (void)_prev;
        // C12 DIAG: barrier deadlock (271 fixed → now some cores stall at crumb 23).
        // Capture, per core, what the barrier actually observes: high32 = num_workers
        // (plain field), low32 = started_count just read. For a stuck core the LAST
        // sample tells whether started_count never reaches num_workers (lost atomic
        // increments / stale reads) or num_workers itself is wrong.
        uint64_t wd_start = 0;
        int32_t _sc = coherent_load(&gd->started_count, std::memory_order_acquire);
        while (_sc < gd->num_workers && !fatal_set(gd)) {
            DIST_DBG(runtime, core_idx,
                     ((uint64_t)(uint32_t)gd->num_workers << 32) |
                         ((uint64_t)(uint32_t)_sc << 8) | (uint64_t)(uint32_t)(_prev & 0xff));
            SPIN_WAIT_HINT();
            watchdog(gd, wd_start);
            _sc = coherent_load(&gd->started_count, std::memory_order_acquire);
        }
    }

    DIST_CRUMB(runtime, core_idx, 24);  // passed startup barrier
    // Replay the full orchestration submit stream: build the per-core map and
    // claim/build owned tasks into the private ring (back-pressure inline). MIX
    // anchors deposit follower subtasks into block.won during this replay.
    TRACE_LAP_RESET(self);  // origin for the first lap span (post-barrier, pre-replay)
#if DIST_SIM_HOST_CLOCK
    const uint64_t orch_t0 = g_overhead_on ? now_ns() : 0;
#endif
    // Onboard: the AICPU's dist_engine_register set rt->ops to the AICPU's
    // (host-address) g_dist_ops and bound the AICPU's g_current_runtime. The
    // AICore is a separate CCEC binary with its own g_dist_ops and its own
    // g_current_runtime global, so before replaying the user orchestration we
    // must (a) bind this AICore's g_current_runtime to gd->rt and (b) repoint
    // rt->ops at THIS binary's g_dist_ops. Without this, current_runtime()
    // returns NULL and rt->ops dangles into the AICPU's address space — the
    // orchestration's rt_submit_aiv_task calls silently no-op and no tasks are
    // submitted (observed: dist_done=9 but all-zero output). On sim the AICPU
    // already did both (shared address space, same g_dist_ops), so this is
    // idempotent there.
    DIST_CRUMB(runtime, core_idx, 25);  // about to bind runtime (deref gd->rt)
#if !DIST_DIAG_SKIP_BIND
    framework_bind_runtime(gd->rt);
#endif
#if !DIST_SIM_HOST_CLOCK
#if !DIST_DIAG_SKIP_OPS
    // Fill the GM-resident ops table with PC-relative (runtime-loaded) function
    // addresses and point rt->ops at it. The static g_dist_ops bakes in base-0
    // addresses that the incore loader never relocates (see g_dist_ops), so
    // calling through it faults; gd->ops holds the fixed-up copy. The rt->ops
    // field is a generic pointer, so store the GM address via an integer cast
    // (address-of-GM-object → uintptr_t → generic ptr) — the same integer-cast
    // form pto_gd() uses to dodge CCEC's "error pointer address space cast".
    dist_ops_refresh_aicore(&gd->ops);
    // C10 DIAG (round 2): the submit_task slot was confirmed a sane .text VA
    // (0x10004be03a34 == &dist_submit_impl), so the ops table IS relocated — the
    // fault is NOT a base-0 function pointer. Faulting PC 0x…263c mapped INSIDE
    // dist_core_main_impl at the crumb-26/91 region, and the dump shows a scalar
    // MPU fault (mte error: 0), not a DMA fault. Prime suspect: the very next
    // store `gd->rt->ops = …` DEREFERENCES gd->rt (a PTO2Runtime* the AICPU wrote
    // and we read via a plain load). Capture gd->rt's raw bits here — BEFORE the
    // deref — so a core that faults on `gd->rt->ops` still reports what address it
    // was about to write through. Sane device-GM VA (0x10004…/0x1000…) → rt is
    // fine and the fault is elsewhere; AICPU-aperture (0x3ff…)/garbage/base-0 → rt
    // is unreachable for the AICore write.
    {
        // C10 VERIFY: after moving orch_args_gm to the end of DistGlobal, gd->rt
        // should now read as the AICPU-published VA (0x1000…), not 0x0. Read it as
        // a raw GM qword at offsetof(rt) with a dcci-invalidate first.
        __gm__ volatile uint64_t *_rtpp = reinterpret_cast<__gm__ volatile uint64_t *>(
            static_cast<uintptr_t>(runtime->dist.global_data_base) + offsetof(DistGlobal, rt)
        );
        pto_dcci_inval(_rtpp, sizeof(uint64_t));
        DIST_DBG(runtime, core_idx, (uint64_t)(*_rtpp));
    }
    // Store the GM table address into the generic rt->ops via pure INTEGER
    // arithmetic off runtime->dist.global_data_base (the segment base, held as
    // a uint64_t). Deriving the address from an integer field — never from a
    // gm-typed &gd->ops — avoids CCEC's "error pointer address space cast"
    // backend failure on gm→generic pointer casts (same reason PTO2Runtime
    // stores dist_global as a uint64_t; see pto_gd()).
    gd->rt->ops = reinterpret_cast<const PTO2RuntimeOps *>(
        static_cast<uintptr_t>(runtime->dist.global_data_base) + offsetof(DistGlobal, ops)
    );
#endif
#else
    gd->rt->ops = &g_dist_ops;
#endif
    DIST_CRUMB(runtime, core_idx, 26);  // runtime bound, ops set
    uint64_t dbg99 = 0;  // C14 DIAG: post-orchestration submit-chain snapshot
    // Orchestration dispatch (onboard AND sim share the SAME mechanism).
    //
    // The user orchestration is compiled into a SEPARATE, position-independent
    // blob — a CCEC PC-relative AICore executable onboard
    // (kernel_compiler._compile_orchestration_dist_blob), or a dlopen'd DSO on
    // sim/host — and uploaded/loaded into GM. It is NOT linked into this runtime
    // image, so the ordinary symbol aicpu_orchestration_entry here is only a weak
    // declaration that the linker resolves to the image base (a silent no-op).
    // The §16.1 "compile orchestration INTO the image / call the symbol directly"
    // approach is not yet wired in the build (see docs C14), so reach the real
    // orchestration through the GM pointers the AICPU installed via
    // dist_engine_register:
    //   - gd->orch_bind_func binds the blob's OWN g_current_runtime to gd->rt
    //     (0 when the orchestration shares this image's binding), and
    //   - gd->orch_func is the blob's entry (a GM code address onboard).
    // Args still flow through GM (gd->orch_args → the GM-resident orch_args).
    if (gd->orch_bind_func != 0) {
        auto blob_bind = reinterpret_cast<void (*)(__gm__ PTO2Runtime *)>(gd->orch_bind_func);
        blob_bind(gd->rt);
    }
    DIST_CRUMB(runtime, core_idx, 27);  // blob bound, about to replay orchestration
    if (gd->orch_func != nullptr && gd->orch_args != nullptr && !fatal_set(gd)) {
        DIST_CRUMB(runtime, core_idx, 98);  // about to CALL orchestration blob
        gd->orch_func(*gd->orch_args);
        DIST_CRUMB(runtime, core_idx, 99);  // orchestration returned
    }
    // C15 DIAG: fn (function_bin_addr) was confirmed to resolve to a valid GM
    // kernel-blob address (0x…1c… region), so the AICore MTE exception is a bad
    // DMA inside the kernel — i.e. a tensor buffer address the kernel reads/writes
    // is invalid/unmapped. Snapshot the FIRST built slot's tensor[0] buffer
    // address (and tensor_count) BEFORE any fn() call. A device-HBM addr should
    // sit in the 0x1… region; a 0/host/garbage value pinpoints a bad arg build.
    //   bits 0-47  : tensors[0].buffer.addr (low 48b)  (input-0 device addr)
    //   bits 48-55 : pto_core_id()                      (logical id)
    //   bits 56-59 : tensor_count                       (# tensor args)
    //   bits 60-63 : self->local_index (low 4)          (tasks submitted; expect 5)
    dbg99 = 0;
    for (int32_t _si = 0; _si < kPrivateSlots; _si++) {
        __gm__ RingSlot &_s = self->slots[_si];
        if (_s.built && _s.function_bin_addr != 0) {
            dbg99 = (uint64_t)_s.tensors[0].buffer.addr & 0xffffffffffffull;
            dbg99 |= ((uint64_t)(uint32_t)(_s.tensor_count & 0xf)) << 56;
            break;
        }
    }
    dbg99 |= ((uint64_t)(uint32_t)(pto_core_id() & 0xff)) << 48;
    dbg99 |= ((uint64_t)(uint32_t)(self->local_index & 0xf)) << 60;
    DIST_CRUMB(runtime, core_idx, 28);  // orchestration replay returned
    (void)dbg99;  // C14 DIAG: unused when DIST_DBG is a no-op (sim build)
#if DIST_SIM_HOST_CLOCK
    const uint64_t orch_replay_end = g_overhead_on ? now_ns() : 0;
#endif

    // Publish "my replay is done" so followers can eventually conclude that no
    // further block.won deposits will arrive for them (§7 tail-idle).
    coherent_fetch_add(&gd->replay_done, 1, std::memory_order_acq_rel);
    DIST_CRUMB(runtime, core_idx, 29);  // replay_done published, entering drain loop

    // Drain to completion: pull any follower deposits addressed to my lane, run
    // ready tasks, and only finish once every core has finished replay (no more
    // pushes), my private ring is empty, and there is no undrained deposit left
    // for my lane.
    uint64_t wd_drain = 0;
    while (!fatal_set(gd)) {
        DIST_CRUMB(runtime, core_idx, 30);  // drain loop top, about to drain_block_won
        drain_block_won(self);
        DIST_CRUMB(runtime, core_idx, 31);  // drain_block_won returned, about to drain_phase_b
        int32_t freed = drain_phase_b(self);
        DIST_CRUMB(runtime, core_idx, 32);  // drain_phase_b returned, about to check replay/pending
        const int32_t rd = coherent_load(&gd->replay_done, std::memory_order_acquire);
        const bool all_replayed = rd >= gd->num_workers;
        const bool ring_empty = (self->occupied_count == 0);
        const bool pending = has_pending_won(self);
        DIST_CRUMB(runtime, core_idx, 33);  // has_pending_won returned
        // C15 DIAG: stuck cores hold occupied_count==1 with ring_empty==0 (all
        // replays done, no pending won) → a built ring slot is never READY, i.e.
        // its fan-in producer's completion flag is never observed as set. Scan
        // for that unready slot and publish exactly WHICH producer it waits on
        // and the LIVE flag value (0 = never seen → lost/invisible completion,
        // the C4 false-sharing / cross-core flag-visibility hazard):
        //   bits 0-11  : first unmet fanin task_id (the producer waited on)
        //   bits 12-23 : the waiting slot's own task_id (consumer)
        //   bit  24    : gd->flags[fanin & (kFlagCap-1)] value (0/1)
        //   bits 25-28 : fanin_count
        //   bit  29    : is_multicore
        //   bit  30    : found an unready built slot at all
        //   bits 32-39 : pto_core_id()
        //   bits 40-47 : self->local_index
        //   bits 48-55 : self->occupied_count
        //   bit  56    : all_replayed
        {
            uint64_t d = 0;
            for (int32_t _i = 0; _i < kPrivateSlots; _i++) {
                __gm__ RingSlot &_s = self->slots[_i];
                if (!_s.occupied || !_s.built) continue;
                int32_t _unmet = -1;
                for (int32_t _f = 0; _f < _s.fanin_count; _f++) {
                    if (coherent_load(&gd->flags[_s.fanin[_f] & (kFlagCap - 1)],
                                      std::memory_order_acquire) == 0) {
                        _unmet = _s.fanin[_f];
                        break;
                    }
                }
                if (_unmet < 0) continue;  // this slot is ready; not the culprit
                int32_t _flag = coherent_load(&gd->flags[_unmet & (kFlagCap - 1)],
                                              std::memory_order_acquire);
                d = (uint64_t)(uint32_t)(_unmet & 0xfff);
                d |= ((uint64_t)(uint32_t)(_s.task_id & 0xfff)) << 12;
                d |= _flag ? (1ull << 24) : 0;
                d |= ((uint64_t)(uint32_t)(_s.fanin_count & 0xf)) << 25;
                d |= _s.is_multicore ? (1ull << 29) : 0;
                d |= (1ull << 30);
                break;
            }
            d |= ((uint64_t)(uint32_t)(pto_core_id() & 0xff)) << 32;
            d |= ((uint64_t)(uint32_t)(self->local_index & 0xff)) << 40;
            d |= ((uint64_t)(uint32_t)(self->occupied_count & 0xff)) << 48;
            d |= all_replayed ? (1ull << 56) : 0;
            DIST_DBG(runtime, core_idx, d);
        }
        if (all_replayed && ring_empty && !pending) break;
        if (freed == 0) {
            SPIN_WAIT_HINT();
            watchdog(gd, wd_drain);
        }
    }

    if (dist_trace() || fatal_set(gd)) {
        fprintf(
            stderr, "[dist] core %d role=%d DONE replayed=%d owned=%d fatal=%d\n", core_idx, core_type_int,
            self->local_index, self->owned_total, fatal_set(gd) ? 1 : 0
        );
    }
#if DIST_SIM_HOST_CLOCK
    // On-core orchestration overhead summary (env PTO_DIST_OVERHEAD). Fold this
    // core's [t0,t1] window into the global min/max/sum, then the LAST finisher
    // prints. See g_overhead_on comment.
    if (g_overhead_on) {
        const uint64_t t1 = now_ns();
        const uint64_t busy = (t1 > orch_t0) ? (t1 - orch_t0) : 0;
        const uint64_t replay = (orch_replay_end > orch_t0) ? (orch_replay_end - orch_t0) : 0;
        atomic_min_u64(g_orch_t0_min, orch_t0);
        atomic_max_u64(g_orch_t1_max, t1);
        atomic_min_u64(g_orch_busy_min, busy);
        atomic_max_u64(g_orch_busy_max, busy);
        g_orch_busy_sum.fetch_add(busy, std::memory_order_seq_cst);
        atomic_min_u64(g_orch_replay_min, replay);
        atomic_max_u64(g_orch_replay_max, replay);
        g_orch_replay_sum.fetch_add(replay, std::memory_order_seq_cst);
        const int32_t nw = gd->num_workers;
        if (g_orch_recorded.fetch_add(1, std::memory_order_seq_cst) + 1 == nw) {
            const double makespan_us =
                (g_orch_t1_max.load(std::memory_order_seq_cst) - g_orch_t0_min.load(std::memory_order_seq_cst)) / 1000.0;
            const uint64_t bsum = g_orch_busy_sum.load(std::memory_order_seq_cst);
            const uint64_t rsum = g_orch_replay_sum.load(std::memory_order_seq_cst);
            fprintf(
                stderr,
                "[dist] OVERHEAD mode=%s cores=%d skip_exec=%d makespan_us=%.1f "
                "busy_us[min/avg/max]=%.1f/%.1f/%.1f "
                "replay_us[min/avg/max]=%.1f/%.1f/%.1f\n",
                gd->tm_shared ? "shared" : "private", nw, g_skip_exec ? 1 : 0, makespan_us,
                g_orch_busy_min.load(std::memory_order_seq_cst) / 1000.0, (bsum / static_cast<double>(nw)) / 1000.0,
                g_orch_busy_max.load(std::memory_order_seq_cst) / 1000.0,
                g_orch_replay_min.load(std::memory_order_seq_cst) / 1000.0, (rsum / static_cast<double>(nw)) / 1000.0,
                g_orch_replay_max.load(std::memory_order_seq_cst) / 1000.0
            );
        }
    }
#endif
    // Dependency-graph signature (env PTO_DIST_DEPSIG). Printed once (core 0) after
    // the global drain barrier, so every task's fan-in is resolved and gd->dep_sig is
    // final. Must be identical between private and shared modes.
    if (self->core_idx == 0 && getenv("PTO_DIST_DEPSIG") != nullptr) {
        fprintf(
            stderr, "[dist] DEPSIG mode=%s sig=%016llx edges=%llu\n", gd->tm_shared ? "shared" : "private",
            static_cast<unsigned long long>(coherent_load(&gd->dep_sig, std::memory_order_relaxed)),
            static_cast<unsigned long long>(coherent_load(&gd->dep_edges, std::memory_order_relaxed))
        );
    }
    // TensorMap op counts (env PTO_DIST_OVERHEAD). Safe to read at core 0 here:
    // every core has finished replay (all_replayed gated the drain break), so all
    // insert/lookup/scan folds are done and visible via the replay_done acq/rel
    // barrier. This is the deterministic, platform-independent overhead metric
    // (private inserts ≈ C×D vs shared ≈ D — the SPMD insert-floor difference).
#if DIST_SIM_HOST_CLOCK
    if (self->core_idx == 0 && g_overhead_on) {
        fprintf(
            stderr, "[dist] TMOPS mode=%s cores=%d inserts=%llu lookups=%llu scans=%llu\n",
            gd->tm_shared ? "shared" : "private", gd->num_workers,
            static_cast<unsigned long long>(g_tm_inserts.load(std::memory_order_relaxed)),
            static_cast<unsigned long long>(g_tm_lookups.load(std::memory_order_relaxed)),
            static_cast<unsigned long long>(g_tm_scans.load(std::memory_order_relaxed))
        );
    }
#endif
    pto_set_core_id(-1);
#if DIST_SIM_HOST_CLOCK
    __atomic_add_fetch(&runtime->dist.done_count, 1, __ATOMIC_ACQ_REL);
#else
    // A5/A2A3 backends can't lower GCC __atomic_add_fetch on addrspace-1 (GM);
    // use the CCEC atomicAdd builtin (works on uncacheable GM on A5).
    dist_atomic_add((__gm__ int32_t *)(&runtime->dist.done_count), 1);
#endif
}

}  // namespace

#if !defined(SIMPLER_DIST_AICPU_ONLY)
extern "C" __attribute__((weak)) __aicore__ void dist_core_main(__gm__ void *runtime_v, int core_idx, int core_type_int) {
    dist_core_main_impl(runtime_v, core_idx, core_type_int);
}

// Extern callable wrapper around the CCEC atomicAdd builtin: the aicore_executor
// fallback (no distributed engine registered) needs to atomically increment
// Runtime::dist.done_count, which lives in GM (addrspace 1). The GCC
// __atomic_add_fetch builtin can't lower on addrspace-1, and atomicAdd is only
// available as a CCEC builtin inside this TU's aicore compilation, so expose a
// small non-template entry point that the platform aicore_executor can call.
// Weak + aicore-only: compiled into both AIC/AIV combined objects, so the linker
// picks one definition; not built for sim (no atomicAdd on host).
#if !DIST_SIM_HOST_CLOCK
extern "C" __attribute__((weak)) __aicore__ int32_t dist_atomic_add_i32(__gm__ int32_t *p, int32_t v) {
    return static_cast<int32_t>(atomicAdd(reinterpret_cast<__gm__ uint32_t *>(p), static_cast<uint32_t>(v)));
}
#endif
#endif  // !SIMPLER_DIST_AICPU_ONLY

#if !defined(SIMPLER_DIST_AICORE_ONLY)
void *dist_engine_register(
    __gm__ PTO2Runtime *rt, DistOrchFunc orch_func, const L2TaskArgs *orch_args, int num_workers,
    __gm__ Runtime *runtime, uint64_t orch_bind_func
) {
    // Allocate the shared-state segment once (docs §13, 方案B). This runs on the
    // AICPU, which HAS malloc/a GM allocator + process globals, so a process-static
    // handle (reused across runs, re-initialized below) is legal here — the
    // no-globals constraint applies only to the AICore workers. Sim: host malloc;
    // HW: pto_gm_alloc returns a GM block addressable + coherent from every AICore.
    // The base is delivered to the workers via Runtime::dist.global_data_base and to
    // the ops callbacks via rt->dist_global; `gd` is a plain local from here on.
    // Segment source (docs §13/§15.2, 方案B):
    //  * Host-preallocated reserve (a5 onboard): runtime->dist.seg_base is a
    //    device VA from host rtMalloc(RT_MEMORY_HBM), reachable from every
    //    AICore MPU (~0x1000…). DistGlobal is carved at offset 0 and the heap
    //    just after it. AICPU-side halMemAlloc must NOT be used here — its
    //    ~0x3ffa… aperture is invisible to the AICore MPU and faults on deref
    //    (errcode 271). The daemon-static g_segment is re-carved whenever the
    //    reserve base changes (a fresh host process hands a new VA; the old
    //    one was freed), so a persistent AICPU scheduler never dereferences a
    //    stale reserve.
    //  * Legacy allocator (sim/a2a3, or seg_base==0): dist_alloc_gm_segment.
    // The host reserve (rtMalloc RT_MEMORY_HBM) is a CACHEABLE device VA. The
    // engine's cross-core Coherent<T> path on the AICore uses plain loads/stores
    // + device atomics (atomicAdd/atomicCAS) with NO dcci — it assumes the shared
    // segment is UNCACHEABLE so every core sees the same HBM word (docs §14, and
    // the Coherent<T> note above). A5 maps HBM twice; add the double-page-table
    // offset to reach the uncacheable alias of the SAME physical pages. On sim /
    // when the driver lacks the double page table this returns 0 (stay cacheable).
    const uint64_t host_seg_base_raw = static_cast<uint64_t>(runtime->dist.seg_base);
    const uint64_t host_seg_size = static_cast<uint64_t>(runtime->dist.seg_size);
    const uint64_t nocache_off = (host_seg_base_raw != 0) ? aicpu_device_nocache_offset() : 0ULL;
    const uint64_t host_seg_base = (host_seg_base_raw != 0) ? (host_seg_base_raw + nocache_off) : 0ULL;
    static __gm__ DistGlobal *g_segment = nullptr;
    static uint64_t s_reserve_base = 0;
    if (host_seg_base != 0) {
        LOG_INFO_V0(
            "[dist] host reserve: raw=0x%llx nocache_off=0x%llx base=0x%llx size=%llu sizeof(DistGlobal)=%llu",
            (unsigned long long)host_seg_base_raw, (unsigned long long)nocache_off, (unsigned long long)host_seg_base,
            (unsigned long long)host_seg_size, (unsigned long long)sizeof(DistGlobal)
        );
        always_assert(host_seg_size >= sizeof(DistGlobal));
        if (g_segment == nullptr || s_reserve_base != host_seg_base) {
            g_segment = reinterpret_cast<__gm__ DistGlobal *>(host_seg_base);
            new (g_segment) DistGlobal();  // value-init (heap_base=nullptr → carved below)
            s_reserve_base = host_seg_base;
        }
    } else if (g_segment == nullptr) {
        g_segment = static_cast<__gm__ DistGlobal *>(dist_alloc_gm_segment(sizeof(DistGlobal)));
        always_assert(g_segment != nullptr);
        new (g_segment) DistGlobal();  // value-init: zero scalars, construct atomics/vectors
    }
    __gm__ DistGlobal *gd = g_segment;
    runtime->dist.global_data_base = reinterpret_cast<uint64_t>(gd);
    rt->dist_global = reinterpret_cast<uint64_t>(gd);  // ops-callback recovery seam (no process-global)
    s_dump_gd = gd;         // diagnostics-only handle (dump_state / dump_trace)
    // Per-core back-pointer to the owning segment (docs §13.3): lets any helper that
    // holds `self` reach shared state as self->gd->…  Set for every slot before
    // workers may start.
    for (int32_t i = 0; i < RUNTIME_MAX_WORKER; i++) gd->cores[i].gd = gd;

    // GM output heap: a BOUNDED ring reclaimed by the completion frontier (M4).
    // Size from PTO_DIST_HEAP_MB (MiB) else kHeapRingDefault. Allocated once per
    // process; if a later run needs a different size, free + realloc.
    {
        size_t want = kHeapRingDefault;
        if (const char *e = getenv("PTO_DIST_HEAP_MB")) {
            const long mb = atol(e);
            if (mb > 0) want = static_cast<size_t>(mb) << 20;
        }
        if (host_seg_base != 0) {
            // Carve the heap ring from the host reserve, page-aligned just after
            // DistGlobal. No free/realloc — it is a slice of the single reserve.
            const uint64_t heap_off = (sizeof(DistGlobal) + 4095ull) & ~4095ull;
            always_assert(heap_off + want <= host_seg_size);  // FATAL if reserve too small
            gd->heap_base = reinterpret_cast<uint8_t *>(host_seg_base + heap_off);
            gd->heap_size = want;
        } else {
            if (gd->heap_base != nullptr && gd->heap_size != want) {
                dist_free_gm_segment(gd->heap_base);
                gd->heap_base = nullptr;
            }
            if (gd->heap_base == nullptr) {
                gd->heap_base = static_cast<uint8_t *>(dist_alloc_gm_segment(want));
                gd->heap_size = (gd->heap_base != nullptr) ? want : 0;
            }
        }
        // Zero the heap each run so freshly-allocated output regions read as 0,
        // matching the centralized runtime's zero-initialized GM. Kernels that
        // read a padded tile (e.g. softmax/PV where valid_len < tile width) rely
        // on the unwritten remainder being zero; an uninitialized (malloc) or
        // recycled heap would otherwise yield nondeterministic results.
        if (gd->heap_base != nullptr) memset(gd->heap_base, 0, gd->heap_size);
    }
    // Dependency-span bound H (R = F - H). Env override for graphs with longer
    // heap spans; default kHDefault.
    gd->H = kHDefault;
    if (const char *e = getenv("PTO_DIST_H")) {
        const long h = atol(e);
        if (h >= 0) gd->H = static_cast<int32_t>(h);
    }
    // Ring-per-bucket per-bucket depth (docs §12.7.2). Default `auto`: derive from
    // the dependency-span H — a bucket's live window is ~H tasks worth of entries,
    // so cap = next_pow2(8*H) gives generous headroom for multiple regions/task and
    // hash skew, clamped to [64, kBucketCapMax]. PTO_DIST_TENSORMAP_RING_CAP=N
    // overrides (rounded up to a power of 2, clamped to kBucketCapMax); =auto keeps
    // the derived value. Overflow within a run is still a deterministic FATAL in
    // insert() (never a silent drop), so an under-sized override fails loudly.
    {
        int32_t derived = round_pow2_cap(gd->H > 0 ? 8 * gd->H : 64);
        if (derived < 64) derived = 64;
        gd->ring_cap = derived;
        if (const char *e = getenv("PTO_DIST_TENSORMAP_RING_CAP")) {
            if (strcmp(e, "auto") != 0) {
                const long n = atol(e);
                if (n > 0) gd->ring_cap = round_pow2_cap(static_cast<int32_t>(n));
            }
        }
    }
    // A bucket's live window spans ~H tasks worth of entries, so H must stay below
    // the ring depth or insert() would overflow. (Necessary condition; true
    // occupancy also depends on regions/task + hash skew, guarded by insert FATAL.)
    always_assert(gd->H < gd->ring_cap - 1);
#if DIST_TRACE_ENABLED
    // Swimlane tracing gate. Capture the epoch now so every core's event ts is
    // relative to the same run start.
    g_trace_on = (getenv("PTO_DIST_SWIMLANE") != nullptr);
    g_trace_epoch_ns = now_ns();
    // Per-core span reserve: 0 when off (reset never reserves → zero overhead on a
    // normal run); a generous bound when on so push_back never reallocs for the
    // sizes we actually analyze (a realloc would perturb heap layout + add timing
    // noise to the very gaps we measure). Best-effort: a huge trace may still grow.
    g_trace_reserve = g_trace_on ? (1 << 16) : 0;
#endif
#if DIST_SIM_HOST_CLOCK
    // Overhead-isolation gate (skip incore kernel calls, keep all bookkeeping).
    g_skip_exec = (getenv("PTO_DIST_SKIP_EXEC") != nullptr);
    // Uniform fake per-kernel cost (ns). See g_fake_exec_ns.
    {
        const char *fe = getenv("PTO_DIST_FAKE_EXEC_NS");
        g_fake_exec_ns = (fe != nullptr) ? atoi(fe) : 0;
        if (g_fake_exec_ns < 0) g_fake_exec_ns = 0;
    }
    // On-core orchestration overhead summary gate (per-core makespan/busy print).
    g_overhead_on = (getenv("PTO_DIST_OVERHEAD") != nullptr);
    g_orch_recorded.store(0, std::memory_order_relaxed);
    g_orch_t0_min.store(~0ull, std::memory_order_relaxed);
    g_orch_t1_max.store(0, std::memory_order_relaxed);
    g_orch_busy_min.store(~0ull, std::memory_order_relaxed);
    g_orch_busy_max.store(0, std::memory_order_relaxed);
    g_orch_busy_sum.store(0, std::memory_order_relaxed);
    g_orch_replay_min.store(~0ull, std::memory_order_relaxed);
    g_orch_replay_max.store(0, std::memory_order_relaxed);
    g_orch_replay_sum.store(0, std::memory_order_relaxed);
#endif
#if DIST_SIM_HOST_CLOCK
    g_tm_inserts.store(0, std::memory_order_relaxed);
    g_tm_lookups.store(0, std::memory_order_relaxed);
    g_tm_scans.store(0, std::memory_order_relaxed);
#endif

    for (int32_t s = 0; s < kCursorShards; s++) {
        coherent_store(&gd->cube_cursor[s].v, -1, std::memory_order_relaxed);
        coherent_store(&gd->vector_cursor[s].v, -1, std::memory_order_relaxed);
        coherent_store(&gd->alloc_cursor[s].v, -1, std::memory_order_relaxed);
    }
    coherent_store(&gd->frontier, -1, std::memory_order_relaxed);
    for (int32_t i = 0; i < kFlagCap; i++)
        coherent_store(&gd->flags[i], 0, std::memory_order_relaxed);
    coherent_store(&gd->fatal, 0, std::memory_order_relaxed);
    coherent_store(&gd->replay_done, 0, std::memory_order_relaxed);
    coherent_store(&gd->started_count, 0, std::memory_order_relaxed);

    // TensorMap mode (docs §12.2): private (default, per-core replica) or shared
    // (one global ring-per-bucket). Selected once per run; never switched mid-run.
    coherent_store(&gd->dep_sig, 0, std::memory_order_relaxed);
    coherent_store(&gd->dep_edges, 0, std::memory_order_relaxed);
    gd->tm_shared = false;
    if (const char *e = getenv("PTO_DIST_TENSORMAP_MODE"))
        if (strcmp(e, "shared") == 0) gd->tm_shared = true;

    // Per-core replay progress. Now read by the run-ahead balance throttle in BOTH
    // modes (not just the shared reclaim floor), so zero it before any core starts.
    for (int32_t i = 0; i < RUNTIME_MAX_WORKER; i++)
        coherent_store(&gd->core_progress[i], 0, std::memory_order_relaxed);

    // Run-ahead balance bound (BOTH modes; docs §12.7.2 / load balance). Cap how
    // far any core's replay position may lead the slowest core. Derived from the
    // core count so it is a reasonable per-platform default automatically:
    // num_workers is 24 AIC + 48 AIV = 72 on a2/a3 and 36 AIC + 72 AIV = 108 on a5
    // (and their sims). The bound MUST be >= num_workers or the window cannot hold
    // one live task per core and healthy parallelism is throttled into idle cores;
    // 2x gives pipelining slack while still reining in a runaway core that would
    // otherwise claim a long contiguous prefix (fetch_max cursor) and starve
    // laggards — the imbalance observed under unequal core progress. Overridable by
    // PTO_DIST_RUNAHEAD (0 disables the throttle entirely).
    gd->runahead_max = 2 * num_workers;
    if (gd->runahead_max < 1) gd->runahead_max = 1;

    gd->tm_runahead_max = 0;
    if (gd->tm_shared) {
        shared_tm_reset(&gd->shared_map, gd->ring_cap);
        coherent_store(&gd->tm_insert_next, 0, std::memory_order_relaxed);
        // Bound run-ahead so the shared live window (Δ+H) stays well under cap,
        // leaving headroom for per-task regions + hash skew. Default keeps the
        // window <= ~3/4 cap (0 disables => FATAL on overflow instead of
        // back-pressure). PTO_DIST_RUNAHEAD overrides below.
        gd->tm_runahead_max = (3 * gd->ring_cap) / 4 - gd->H - 1;
        if (gd->tm_runahead_max < 1) gd->tm_runahead_max = 1;
    }

    // One knob: PTO_DIST_RUNAHEAD=N = "max tasks a core may run ahead of the
    // slowest". Overrides the balance bound (BOTH modes) and, in shared mode, the
    // ring back-pressure bound too. 0 disables the balance throttle (and in shared
    // mode reverts ring overflow to the deterministic FATAL).
    if (const char *e = getenv("PTO_DIST_RUNAHEAD")) {
        const long v = atol(e);
        if (v >= 0) {
            gd->runahead_max = static_cast<int32_t>(v);
            if (gd->tm_shared) gd->tm_runahead_max = static_cast<int32_t>(v);
        }
    }
    gd->orch_func = orch_func;
    gd->orch_bind_func = orch_bind_func;
    // Copy the entry args into the GM segment so the AICore can read them (the
    // AICPU-built orch_args lives in AICPU DDR). A bitwise copy is intentional:
    // L2TaskArgs deletes operator= to prevent accidental copies that could
    // dangle, but here we deliberately relocate the container while keeping its
    // TensorRefs pointing at the GM-resident Runtime::orch_args_storage_ (valid
    // on-core). Not compiled in the AICore build (register runs on the AICPU).
    if (orch_args != nullptr) {
        memcpy(static_cast<void *>(&gd->orch_args_gm), static_cast<const void *>(orch_args), sizeof(L2TaskArgs));
        gd->orch_args = &gd->orch_args_gm;
        // DIAG: log the exact addresses the AICPU is handing to the AICore so the
        // on-core DIST_DBG values can be compared against them. tensor(0).raw_addr
        // is the VA the core will dereference at crumb 96 — if it's in the AICPU
        // ~0x3ff… aperture it will fault on-core (MPU 271).
        LOG_INFO_V0(
            "[dist] register: gd=0x%llx orch_args_gm=0x%llx tensor_count=%d tensor0_addr=0x%llx",
            (unsigned long long)reinterpret_cast<uint64_t>(gd),
            (unsigned long long)reinterpret_cast<uint64_t>(&gd->orch_args_gm),
            gd->orch_args_gm.tensor_count(),
            (gd->orch_args_gm.tensor_count() > 0
                 ? (unsigned long long)gd->orch_args_gm.tensor(0).raw_addr()
                 : 0ull)
        );
    } else {
        gd->orch_args = nullptr;
    }
    gd->rt = rt;
    gd->runtime = runtime;
    // DIAG (C10 round 3): the AICore reads gd->rt as 0x0. Log the AICPU's own
    // offsetof(DistGlobal, rt) + the rt value it stored + a post-store readback.
    // Compare offsetof against the AICore's DIST_DBG(offsetof) — mismatch proves a
    // cross-compiler DistGlobal layout divergence; equal + rt!=0 here but 0 on-core
    // proves a publish/paging bug for that page.
    // NOTE: offsetof() is ill-formed on the AICPU build (DistGlobal is
    // non-standard-layout there — Coherent<T> wraps std::atomic<T>), so compute the
    // field offset by pointer subtraction instead. This non-standard-layout status
    // is itself the smoking gun: if the AICore (CCEC, Coherent<T>==plain T →
    // standard-layout) lays the struct out differently, offsetof(DistGlobal, rt)
    // there won't equal this runtime offset.
    LOG_INFO_V9(
        "[dist] register: offset_rt=%llu rt=0x%llx gd->rt(readback)=0x%llx sizeof(DistGlobal)=%llu",
        (unsigned long long)(reinterpret_cast<uintptr_t>(&gd->rt) - reinterpret_cast<uintptr_t>(gd)),
        (unsigned long long)reinterpret_cast<uint64_t>(rt),
        (unsigned long long)reinterpret_cast<uint64_t>(gd->rt),
        (unsigned long long)sizeof(DistGlobal)
    );
    // Landmark offsets (pointer subtraction; offsetof is ill-formed here) to
    // localize the 160-byte AICPU/AICore layout divergence at `rt`. Compare against
    // the AICore's packed offsetof(orch_func)|offsetof(orch_args_gm) DIST_DBG.
    LOG_INFO_V9(
        "[dist] register: off_frontier=%llu off_orch_func=%llu off_orch_args=%llu off_orch_args_gm=%llu off_ops=%llu",
        (unsigned long long)(reinterpret_cast<uintptr_t>(&gd->frontier) - reinterpret_cast<uintptr_t>(gd)),
        (unsigned long long)(reinterpret_cast<uintptr_t>(&gd->orch_func) - reinterpret_cast<uintptr_t>(gd)),
        (unsigned long long)(reinterpret_cast<uintptr_t>(&gd->orch_args) - reinterpret_cast<uintptr_t>(gd)),
        (unsigned long long)(reinterpret_cast<uintptr_t>(&gd->orch_args_gm) - reinterpret_cast<uintptr_t>(gd)),
        (unsigned long long)(reinterpret_cast<uintptr_t>(&gd->ops) - reinterpret_cast<uintptr_t>(gd))
    );

    // Derive the physical-block topology (1 AIC + 2 AIV per block) the same way
    // the centralized scheduler discovers clusters: AIC/AIV cores in worker-index
    // order, AIC[b] paired with AIV[2b] (AIV0) and AIV[2b+1] (AIV1). Followers and
    // anchors use this to address block.won deposits. See §3.1.
    gd->num_workers = num_workers;
    int32_t aic_ids[RUNTIME_MAX_WORKER];
    int32_t aiv_ids[RUNTIME_MAX_WORKER];
    int32_t naic = 0, naiv = 0;
    for (int32_t i = 0; i < num_workers && i < RUNTIME_MAX_WORKER; i++) {
        gd->layout[i].block_id = -1;
        gd->layout[i].lane = LANE_NONE;
        if (runtime->workers[i].core_type == CoreType::AIC) {
            aic_ids[naic++] = i;
        } else {
            aiv_ids[naiv++] = i;
        }
    }
    gd->num_blocks = naic;
    for (int32_t b = 0; b < naic; b++) {
        gd->layout[aic_ids[b]] = CoreLayout{b, LANE_AIC};
        if (2 * b < naiv) gd->layout[aiv_ids[2 * b]] = CoreLayout{b, LANE_AIV0};
        if (2 * b + 1 < naiv) gd->layout[aiv_ids[2 * b + 1]] = CoreLayout{b, LANE_AIV1};
        coherent_store(&gd->blocks[b].any_pub, 0, std::memory_order_relaxed);
        for (int32_t s = 0; s < kPrivateSlots; s++) {
            coherent_store(&gd->blocks[b].slots[s].state, 0, std::memory_order_relaxed);
        }
    }

    if (dist_trace()) {
        fprintf(
            stderr, "[dist] register: num_workers=%d heap_base=%p heap_size=%zu\n", num_workers,
            (void *)gd->heap_base, gd->heap_size
        );
    }

    // Install the SIGUSR1 deadlock dumper once, but only when diagnostics are
    // opted in (PTO_DIST_WATCHDOG set) — default runs install no signal handler.
    static bool handler_installed = false;
    if (!handler_installed && getenv("PTO_DIST_WATCHDOG") != nullptr) {
        signal(SIGUSR1, dist_dump_state);
        handler_installed = true;
    }

    // Publish all of the above before any worker observes Runtime::dist.go. Sim:
    // the release fence in one address space suffices; HW: flush the GM segment
    // so every AICore reads the initialized state (docs §13).
    dist_publish_gm_segment(gd, sizeof(DistGlobal));
    pto_shared_fence(std::memory_order_release);
    rt->ops = &g_dist_ops;
// Return &dist_core_main ONLY when this binary actually contains its definition.
// Sim AICPU .so: DIST_SIM_HOST_CLOCK=1 AND SIMPLER_DIST_AICPU_ONLY is NOT defined
//   → dist_core_main is compiled here (the sim AICore worker threads reach it via
//   the core_main_fn pointer). Onboard AICPU .so: DIST_SIM_HOST_CLOCK is also 1
//   (gcc, not CCEC) BUT SIMPLER_DIST_AICPU_ONLY IS defined → dist_core_main is
//   NOT compiled into this binary (it's CCEC-compiled into the AICore binary and
//   called directly by aicore_execute). Returning &dist_core_main here would
//   leave an unresolved symbol → dlopen fails → 507018. So gate on both.
#if DIST_SIM_HOST_CLOCK && !defined(SIMPLER_DIST_AICPU_ONLY)
    return reinterpret_cast<void *>(&dist_core_main);
#else
    // Onboard: dist_core_main is linked into the AICore binary; AICore workers
    // call it directly (see aicore_executor.cpp), not via this pointer.
    return nullptr;
#endif
}

#endif  // !SIMPLER_DIST_AICORE_ONLY

#if !defined(SIMPLER_DIST_AICORE_ONLY)
#if DIST_TRACE_ENABLED
void dist_engine_dump_trace() {
    if (!g_trace_on) return;
    __gm__ DistGlobal *gd = s_dump_gd;  // diagnostics-only handle (docs §13); not the functional path
    if (gd == nullptr) return;
    const char *path = getenv("PTO_DIST_SWIMLANE");
    if (path == nullptr || path[0] == '\0') return;
    FILE *f = fopen(path, "w");
    if (f == nullptr) {
        fprintf(stderr, "[dist_engine] cannot open swimlane file %s for write\n", path);
        return;
    }

    auto lane_name = [](int32_t lane) -> const char * {
        switch (lane) {
        case LANE_AIC:
            return "AIC";
        case LANE_AIV0:
            return "AIV0";
        case LANE_AIV1:
            return "AIV1";
        default:
            return "?";
        }
    };

    // Chrome Trace Event Format (https://ui.perfetto.dev / chrome://tracing).
    // Two process groups: pid = block_id is the WALL-clock swimlane; pid =
    // block_id + kCpuPid is a parallel CPU-time swimlane (same spans, width =
    // cpu_us). process_sort_index forces all wall groups above all cpu groups.
    // Dependency arrows (flow events) are emitted only in the cpu group so they
    // stay within the cpu lanes instead of tangling across the wall lanes.
    constexpr int32_t kCpuPid = 1000;
    fprintf(f, "{\n  \"displayTimeUnit\": \"ns\",\n  \"traceEvents\": [\n");
    bool first = true;
    const int32_t nw = gd->num_workers;

    // Lane/process name + sort metadata first (so idle lanes still appear).
    for (int32_t c = 0; c < nw && c < RUNTIME_MAX_WORKER; c++) {
        DistCore &co = gd->cores[c];
        if (co.block_id < 0 || co.lane < 0) continue;
        if (!first) fprintf(f, ",\n");
        first = false;
        fprintf(
            f, "    {\"ph\":\"M\",\"name\":\"process_name\",\"pid\":%d,\"args\":{\"name\":\"block%d (wall)\"}}",
            co.block_id, co.block_id
        );
        fprintf(
            f, ",\n    {\"ph\":\"M\",\"name\":\"process_sort_index\",\"pid\":%d,\"args\":{\"sort_index\":%d}}",
            co.block_id, co.block_id
        );
        fprintf(
            f,
            ",\n    {\"ph\":\"M\",\"name\":\"thread_name\",\"pid\":%d,\"tid\":%d,"
            "\"args\":{\"name\":\"%s (core%d)\"}}",
            co.block_id, co.lane, lane_name(co.lane), c
        );
        fprintf(
            f, ",\n    {\"ph\":\"M\",\"name\":\"process_name\",\"pid\":%d,\"args\":{\"name\":\"block%d (cpu)\"}}",
            co.block_id + kCpuPid, co.block_id
        );
        fprintf(
            f, ",\n    {\"ph\":\"M\",\"name\":\"process_sort_index\",\"pid\":%d,\"args\":{\"sort_index\":%d}}",
            co.block_id + kCpuPid, co.block_id + kCpuPid
        );
        fprintf(
            f,
            ",\n    {\"ph\":\"M\",\"name\":\"thread_name\",\"pid\":%d,\"tid\":%d,"
            "\"args\":{\"name\":\"%s (core%d)\"}}",
            co.block_id + kCpuPid, co.lane, lane_name(co.lane), c
        );
        // CPU-group kernel sub-lane (tid = lane + 3): kernel spans live here so a
        // ringbp bar that time-contains its releasing kernel does not nest+hide it.
        fprintf(
            f,
            ",\n    {\"ph\":\"M\",\"name\":\"thread_name\",\"pid\":%d,\"tid\":%d,"
            "\"args\":{\"name\":\"%s·kernel (core%d)\"}}",
            co.block_id + kCpuPid, co.lane + 3, lane_name(co.lane), c
        );
    }

    auto phase_name = [](TracePhase p) -> const char * {
        switch (p) {
        case TracePhase::Kernel:
            return "kernel";
        case TracePhase::Alloc:
            return "alloc";
        case TracePhase::Build:
            return "build";
        case TracePhase::DrainWon:
            return "drain_won";
        case TracePhase::Replay:
            return "orch";  // swimlane label: replayed the submit stream but LOST the claim
        case TracePhase::RingBp:
            return "ringbp";
        case TracePhase::EfDrain:
            return "efdrain";
        case TracePhase::Commit:
            return "commit";
        default:
            return "?";
        }
    };

    // Index: task_id -> its kernel span location in the CPU group, so a dep edge
    // can anchor an arrow at the producer's and consumer's actual spans.
    struct SpanLoc {
        int32_t pid;
        int32_t tid;
        double ts_us;
        double dur_us;
    };
    // In the CPU group, kernel spans go on a SEPARATE sub-lane (tid = lane +
    // kCpuKernelLane) from the build/ringbp/replay/alloc spans (tid = lane). A
    // ringbp span time-contains the kernel that ends its wait, so on one lane
    // perfetto would nest the kernel inside the ringbp bar and hide it; splitting
    // the kernel onto its own row keeps both visible.
    constexpr int32_t kCpuKernelLane = 3;
    std::vector<SpanLoc> kloc(static_cast<size_t>(kFlagCap), SpanLoc{-1, -1, 0.0, 0.0});
    for (int32_t c = 0; c < nw && c < RUNTIME_MAX_WORKER; c++) {
        DistCore &co = gd->cores[c];
        if (co.block_id < 0 || co.lane < 0) continue;
        for (const TraceEvent &e : co.trace) {
            if (e.phase != TracePhase::Kernel || e.task_id < 0 || e.task_id >= kFlagCap) continue;
            kloc[static_cast<size_t>(e.task_id)] =
                SpanLoc{co.block_id + kCpuPid, co.lane + kCpuKernelLane, e.ts_ns / 1000.0, e.cpu_ns / 1000.0};
        }
    }
    // Index: task_id -> its ringbp span in the CPU group (the arrow head for a
    // slot-release edge anchors at the ringbp's END = when the wait was satisfied).
    std::vector<SpanLoc> rbloc(static_cast<size_t>(kFlagCap), SpanLoc{-1, -1, 0.0, 0.0});
    for (int32_t c = 0; c < nw && c < RUNTIME_MAX_WORKER; c++) {
        DistCore &co = gd->cores[c];
        if (co.block_id < 0 || co.lane < 0) continue;
        for (const TraceEvent &e : co.trace) {
            if (e.phase != TracePhase::RingBp || e.task_id < 0 || e.task_id >= kFlagCap) continue;
            rbloc[static_cast<size_t>(e.task_id)] =
                SpanLoc{co.block_id + kCpuPid, co.lane, e.ts_ns / 1000.0, e.cpu_ns / 1000.0};
        }
    }

    // Duration events: kernel + non-kernel overhead spans, emitted once in the
    // wall group (pid=block) and once in the cpu group (pid=block+kCpuPid).
    for (int32_t c = 0; c < nw && c < RUNTIME_MAX_WORKER; c++) {
        DistCore &co = gd->cores[c];
        if (co.block_id < 0 || co.lane < 0) continue;
        for (const TraceEvent &e : co.trace) {
            const char *ph = phase_name(e.phase);
            char name[64];
            if (e.phase != TracePhase::Kernel) {
                snprintf(name, sizeof(name), "%s#%d", ph, e.task_id);
            } else if (e.func_id >= 0) {
                snprintf(name, sizeof(name), "f%d#%d", e.func_id, e.task_id);
            } else {
                snprintf(name, sizeof(name), "task#%d", e.task_id);
            }
            if (!first) fprintf(f, ",\n");
            first = false;
            // Convert raw ns -> us (swimlane unit) here, at dump time — never on the
            // hot path (see TraceEvent).
            const double ts_us = e.ts_ns / 1000.0;
            const double dur_us = e.dur_ns / 1000.0;
            const double cpu_us = e.cpu_ns / 1000.0;
            fprintf(
                f,
                "    {\"ph\":\"X\",\"name\":\"%s\",\"pid\":%d,\"tid\":%d,\"ts\":%.3f,\"dur\":%.3f,"
                "\"args\":{\"phase\":\"%s\",\"task_id\":%d,\"func_id\":%d,\"core\":%d,\"mc\":%d,\"cpu_us\":%.3f}}",
                name, co.block_id, co.lane, ts_us, dur_us, ph, e.task_id, e.func_id, c, e.multicore, cpu_us
            );
            fprintf(
                f,
                ",\n    {\"ph\":\"X\",\"name\":\"%s\",\"pid\":%d,\"tid\":%d,\"ts\":%.3f,\"dur\":%.3f,"
                "\"args\":{\"phase\":\"%s\",\"task_id\":%d,\"func_id\":%d,\"wall_us\":%.3f}}",
                name, co.block_id + kCpuPid, e.phase == TracePhase::Kernel ? co.lane + kCpuKernelLane : co.lane, ts_us,
                cpu_us, ph, e.task_id, e.func_id, dur_us
            );
        }
    }

    // Flow events: the full static dependency graph. One arrow per dep edge, in
    // the cpu group, from the PRODUCER kernel span's end to the CONSUMER kernel
    // span's start (time always forward: a producer completes before its consumer
    // runs). Click any task and follow arrows backward hop-by-hop to walk the
    // chain "what was this waiting on, and what was THAT waiting on".
    int32_t flow_id = 0;
    for (int32_t c = 0; c < nw && c < RUNTIME_MAX_WORKER; c++) {
        DistCore &co = gd->cores[c];
        if (co.block_id < 0 || co.lane < 0) continue;
        for (const DistCore::DepEdge &de : co.dep_edges) {
            if (de.producer_task < 0 || de.producer_task >= kFlagCap) continue;
            if (de.consumer_task < 0 || de.consumer_task >= kFlagCap) continue;
            const SpanLoc &pr = kloc[static_cast<size_t>(de.producer_task)];
            const SpanLoc &cs = kloc[static_cast<size_t>(de.consumer_task)];
            if (pr.pid < 0 || cs.pid < 0) continue;  // need both kernel spans
            fprintf(
                f, ",\n    {\"ph\":\"s\",\"name\":\"dep\",\"cat\":\"dep\",\"id\":%d,\"pid\":%d,\"tid\":%d,\"ts\":%.3f}",
                flow_id, pr.pid, pr.tid, pr.ts_us + pr.dur_us
            );
            fprintf(
                f,
                ",\n    {\"ph\":\"f\",\"name\":\"dep\",\"cat\":\"dep\",\"id\":%d,\"bp\":\"e\",\"pid\":%d,\"tid\":%d,"
                "\"ts\":%.3f}",
                flow_id, cs.pid, cs.tid, cs.ts_us
            );
            flow_id++;
        }
    }

    // Flow events (cat="slot"): slot-release edges that explain a ringbp's stall.
    // From the END of the occupant kernel's span (the moment it frees the slot) to
    // the END of the waiting ringbp span. Chains with the dep arrows: ringbp
    // --slot--> occupant kernel --dep--> the occupant's fan-in kernels.
    for (int32_t c = 0; c < nw && c < RUNTIME_MAX_WORKER; c++) {
        DistCore &co = gd->cores[c];
        if (co.block_id < 0 || co.lane < 0) continue;
        for (const DistCore::DepEdge &se : co.slot_edges) {
            if (se.producer_task < 0 || se.producer_task >= kFlagCap) continue;  // occupant
            if (se.consumer_task < 0 || se.consumer_task >= kFlagCap) continue;  // ringbp waiter
            const SpanLoc &occ = kloc[static_cast<size_t>(se.producer_task)];
            const SpanLoc &rb = rbloc[static_cast<size_t>(se.consumer_task)];
            if (occ.pid < 0 || rb.pid < 0) continue;
            double tail = occ.ts_us + occ.dur_us;      // occupant kernel end (slot freed)
            const double head = rb.ts_us + rb.dur_us;  // ringbp end (wait satisfied)
            if (tail > head) tail = head;              // keep forward in time
            fprintf(
                f,
                ",\n    {\"ph\":\"s\",\"name\":\"slot\",\"cat\":\"slot\",\"id\":%d,\"pid\":%d,\"tid\":%d,\"ts\":%.3f}",
                flow_id, occ.pid, occ.tid, tail
            );
            fprintf(
                f,
                ",\n    {\"ph\":\"f\",\"name\":\"slot\",\"cat\":\"slot\",\"id\":%d,\"bp\":\"e\",\"pid\":%d,\"tid\":%d,"
                "\"ts\":%.3f}",
                flow_id, rb.pid, rb.tid, head
            );
            flow_id++;
        }
    }

    fprintf(f, "\n  ]\n}\n");
    fclose(f);
    fprintf(stderr, "[dist_engine] swimlane trace written to %s\n", path);
}
#else   // !DIST_TRACE_ENABLED
// Tracing compiled out: keep the public symbol so aicpu_executor.cpp still links.
void dist_engine_dump_trace() {}
#endif  // DIST_TRACE_ENABLED

#endif  // !SIMPLER_DIST_AICORE_ONLY
