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
 * @file pipe_sync.h
 * @brief Kernel-exit pipe drain for AIC/AIV kernels.
 *
 * Call pipe_sync() once after the final TSTORE in a kernel impl. It drains the
 * core's output pipe (FIX for AIC, MTE3 for AIV) onto the scalar pipe so the
 * GM write fully completes before kernel exit. Without this drain, the next
 * dispatched kernel can read stale GM and produce silent numerical errors.
 *
 * Usage prerequisite: this header does NOT pull in <pto/pto-inst.hpp>; it
 * leaves PTO-ISA includes to the caller. The caller must already have done
 *   #include <pto/pto-inst.hpp>
 *   using namespace pto;
 * before including pipe_sync.h, so that set_flag, wait_flag, PIPE_*, and
 * EVENT_ID7 are visible at the point where pipe_sync() is parsed.
 *
 * Core-type selection uses the kernel toolchain's predefined macros:
 *   __DAV_VEC__  → AIV build (TSTORE through MTE3)
 *   __DAV_CUBE__ → AIC build (TSTORE through FIX)
 * On hardware these are auto-defined by ccec from --cce-aicore-arch; in sim
 * they are passed explicitly by Gxx15Toolchain (see simpler_setup/toolchain.py).
 */

#pragma once

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

static __aicore__ inline void pipe_sync() {
#if defined(__DAV_VEC__)
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
#elif defined(__DAV_CUBE__)
    set_flag(PIPE_FIX, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_FIX, PIPE_S, EVENT_ID7);
#else
#error "pipe_sync.h requires __DAV_VEC__ or __DAV_CUBE__ to be defined"
#endif
}
