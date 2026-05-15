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
 * @file simt_meta.h
 * @brief SIMT metadata TLV records for AICore kernel ELF (onboard / a5)
 *
 * The legacy launch path (rtKernelLaunchWithHandleV2 + rtRegisterAllKernel)
 * requires the kernel ELF to carry two TLV records that runtime reads at
 * register time:
 *   - F_TYPE_COMPILER_ALLOC_UB_SIZE (7)  -> Kernel::shareMemSize_
 *   - F_TYPE_AIV_TYPE_FLAG          (12) -> Kernel::kernelVfType_
 * bisheng emits these only when it can statically infer the kernel uses
 * SIMT intrinsics. Our entry is an SU dispatcher (vector ops live in task
 * .o files invoked through aicore_execute), so the compiler cannot tag it.
 *
 * kernel.cpp's CMakeLists.txt pairs the hand-written record with
 * `-mllvm -cce-dyn-kernel-stack-size=false`, which stops bisheng from
 * auto-emitting a sibling `.ascend.meta.<funcname>` section. Without that
 * flag, runtime's parser (kernelInfoMap keyed by section name) would
 * overwrite our values with bisheng's NO_VF / shareMemSize=0 defaults.
 *
 * TLV type IDs mirror RT_FUNCTION_TYPE_COMPILER_ALLOC_UB_SIZE (7) and
 * RT_FUNCTION_TYPE_AIV_TYPE_FLAG (12) in CANN's runtime/runtime/elf_base.h.
 * That header is host-side (extern "C", part of the runtime API) so we
 * re-declare the two values we need rather than pull runtime headers into
 * an AICore device-side TU.
 */

#ifndef PLATFORM_A5_AICORE_SIMT_META_H_
#define PLATFORM_A5_AICORE_SIMT_META_H_

enum FuncMetaType {
    F_TYPE_COMPILER_ALLOC_UB_SIZE = 7,
    F_TYPE_AIV_TYPE_FLAG = 12,
};

// AIVType values are not exposed in any CANN C/C++ header. The canonical
// source is CANN's compiler-side Python script
// (python/site-packages/tbe/tikcpp/ascendc_identify_meta_section_info.py),
// which is what bisheng / asc_op_compiler consult when classifying kernels.
enum AIVType {
    AIV_TYPE_NO_VF = 1,
    AIV_TYPE_SIMD_VF_ONLY = 2,
    AIV_TYPE_SIMT_VF_ONLY = 3,
    AIV_TYPE_SIMD_SIMT_MIX_VF = 4,
};

struct TlvHeader {
    unsigned short type;
    unsigned short len;
};

struct FuncMetaCompilerUbSize {
    TlvHeader head;
    unsigned int ub_size;
};

struct FuncMetaAivTypeFlag {
    TlvHeader head;
    unsigned int aiv_type;
};

struct FuncLevelMeta {
    FuncMetaCompilerUbSize ub_size_meta;
    FuncMetaAivTypeFlag aiv_type_meta;
};

#endif  // PLATFORM_A5_AICORE_SIMT_META_H_
