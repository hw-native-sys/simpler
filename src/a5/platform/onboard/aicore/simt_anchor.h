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
 * @file simt_anchor.h
 * @brief VF-classification anchor for the AICore kernel ELF (onboard / a5)
 *
 * bisheng auto-emits the two TLV records runtime reads at register time
 * (RT_FUNCTION_TYPE_COMPILER_ALLOC_UB_SIZE (7) -> Kernel::shareMemSize_ and
 * RT_FUNCTION_TYPE_AIV_TYPE_FLAG (12) -> Kernel::kernelVfType_) by statically
 * classifying which vector functional units a kernel uses. Our KERNEL_ENTRY is
 * an SU dispatcher whose vector ops live in separately compiled task .o files
 * invoked through aicore_execute, so left alone bisheng tags it NO_VF.
 *
 * This anchor reintroduces (never-executed) VF ops so the classification fires,
 * replacing the previous hand-written TLV record plus the
 * `-cce-dyn-kernel-stack-size=false` auto-emit suppression flag. It deliberately
 * uses BOTH a SIMT launch and a SIMD vector op so bisheng emits
 * AIV_TYPE = SIMD_SIMT_MIX_VF (4), NOT SIMT_VF_ONLY (3): the shared SU
 * dispatcher routes both SIMD and SIMT task .o files, and a SIMT-only tag makes
 * runtime reject every SIMD launch through it (107000 param-invalid across the
 * whole a5 ST suite — observed in CI). MIX matches the value the old
 * hand-written record pinned on purpose.
 *
 * Letting bisheng own the record removes the silent-breakage risk of the old
 * approach (a compiler change to the meta-section layout would desync the
 * hand-written bytes); the UB size is whatever bisheng computes (8 KB floor).
 *
 * Built purely from bisheng/cce compiler builtins — cce::async_invoke /
 * cce::dim3 (`__clang_cce_simt.h`), __simt_vf__ / __simd_vf__ / LAUNCH_BOUND
 * (`__clang_cce_defines.h`), threadIdx (`__clang_cce_simt_builtin_vars.h`), all
 * auto-included by ccec for the dav-c310-vec arch. No pto-isa dependency, so
 * the a5 platform runtime build stays pto-isa-free (only per-example incore
 * kernels pull pto-isa, via kernel_compiler.py).
 */

#ifndef PLATFORM_A5_AICORE_SIMT_ANCHOR_H_
#define PLATFORM_A5_AICORE_SIMT_ANCHOR_H_

#ifdef __DAV_VEC__

#include <cstdint>

// `__simt_vf__` (cce_simt_entry + noinline) marks this as the SIMT VF kernel
// bisheng's meta-section pass keys on; the threadIdx read keeps it a genuine
// vector-thread body so the kernel is not folded away.
//
// `static` is load-bearing: without it cce_simt_entry exports a GLOBAL
// `..._simt_entry` symbol, which rtRegisterAllKernel registers as a second
// launchable kernel alongside the real dispatcher entry — that extra
// registration fails the whole a5 launch path with 107000 (param-invalid). The
// anchor branch never executes (force_simt_anchor is always 0), so the kernel
// only needs to exist at compile time for classification; internal linkage
// drops the global symbol while keeping the SIMT meta tag.
static __simt_vf__ LAUNCH_BOUND(1024) __aicore__ void simt_meta_anchor_kernel(__gm__ uint32_t *sink) {
    sink[threadIdx.x] = threadIdx.x;
}

// SIMD VF companion: marks the entry as also using the SIMD vector unit so
// bisheng classifies it SIMD_SIMT_MIX_VF(4) rather than SIMT_VF_ONLY(3). The
// shared SU dispatcher routes both SIMD and SIMT task .o files, so the entry
// must advertise MIX or runtime rejects SIMD launches (107000 param-invalid).
__simd_vf__ __aicore__ void simd_meta_anchor_kernel(__ubuf__ uint32_t *ub) { ub[0] = ub[0] + 1; }

// always_inline so the async_invoke lands inside KERNEL_ENTRY's own body — the
// final link is `ld -r` (relocatable, not LTO), so cross-TU inlining requires
// the definition to be visible at the call site. Inlining is what attaches the
// SIMT tag to the *entry* function's `.ascend.meta` section, which is the one
// runtime reads at register time.
__attribute__((always_inline)) inline __aicore__ void simt_meta_anchor(__gm__ uint32_t *sink) {
    cce::async_invoke<simt_meta_anchor_kernel>(cce::dim3{1, 1, 1}, sink);
    simd_meta_anchor_kernel((__ubuf__ uint32_t *)0);
}

#endif  // __DAV_VEC__

#endif  // PLATFORM_A5_AICORE_SIMT_ANCHOR_H_
