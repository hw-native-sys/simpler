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
 * Unified invocation-args wire header (host → AICPU).
 *
 * Every kernel-mode launch ships one immutable args snapshot to the AICPU:
 * this fixed 64-byte header followed by `payload_bytes` of runtime-specific
 * payload. The header is the shared envelope both runtimes use; the payload
 * format under it belongs to each runtime (tensormap_and_ringbuffer carries
 * graph-build input, host_build_graph carries a serialized graph blob) and is
 * not constrained here. The AICPU dispatch entry validates identity,
 * generation, and capacity against this header before dispatching to the
 * runtime payload consumer — that validation lives with the consumers, not
 * in this header.
 *
 * Wire ABI: the layout below is frozen and shipped by memcpy through
 * CANN's HostArgs deep copy, so the struct is POD, position-independent, and
 * carries no pointers. All reserved fields read as zero and a consumer
 * rejects a nonzero one (fail-closed evolution, same rule as
 * SimplerKernelCtxControl).
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "execution_mode.h"

enum {
    SIMPLER_KERNEL_INVOCATION_ABI_VERSION = 1,
};

typedef struct SimplerKernelInvocationHeader {
    uint32_t abi_version;  /* must equal SIMPLER_KERNEL_INVOCATION_ABI_VERSION */
    uint32_t header_bytes; /* sizeof(SimplerKernelInvocationHeader); the payload starts here */
    uint32_t mode;         /* SimplerExecutionMode of the issuing context */
    int32_t callable_id;   /* target callable; matches the prepared registration */
    /* Residency-slot generation of the resolved callable. Compared on the
       AICPU dispatch path: replay does not return to the host, so a stale
       captured snapshot is caught on-device, not host-side. */
    uint64_t generation;
    /* Arg counts of this invocation; a consumer checks them against the
       callable's declared signature (ChipCallable sig_count / scalar_count).
       Every count in this header is int32_t, matching the callable-side
       convention; negative values are invalid. */
    int32_t tensor_count;
    int32_t scalar_count;
    /* Count of host-only duplicate tensor args (a tensor the host must read
       is passed twice: a device arg plus a host-only copy). Zero until the
       host-only copy contract lands; consumers reject nonzero meanwhile. */
    int32_t host_copy_tensor_count;
    uint32_t reserved0; /* must be zero */
    /* Byte length of the runtime-specific payload that follows this header. */
    uint64_t payload_bytes;
    uint64_t reserved[2]; /* must be all zero */
} SimplerKernelInvocationHeader;

#ifdef __cplusplus
#include <type_traits>

static_assert(
    std::is_trivially_copyable_v<SimplerKernelInvocationHeader> &&
    std::is_standard_layout_v<SimplerKernelInvocationHeader>
);
static_assert(offsetof(SimplerKernelInvocationHeader, abi_version) == 0);
static_assert(offsetof(SimplerKernelInvocationHeader, header_bytes) == 4);
static_assert(offsetof(SimplerKernelInvocationHeader, mode) == 8);
static_assert(offsetof(SimplerKernelInvocationHeader, callable_id) == 12);
static_assert(offsetof(SimplerKernelInvocationHeader, generation) == 16);
static_assert(offsetof(SimplerKernelInvocationHeader, tensor_count) == 24);
static_assert(offsetof(SimplerKernelInvocationHeader, scalar_count) == 28);
static_assert(offsetof(SimplerKernelInvocationHeader, host_copy_tensor_count) == 32);
static_assert(offsetof(SimplerKernelInvocationHeader, reserved0) == 36);
static_assert(offsetof(SimplerKernelInvocationHeader, payload_bytes) == 40);
static_assert(offsetof(SimplerKernelInvocationHeader, reserved) == 48);
static_assert(sizeof(SimplerKernelInvocationHeader) == 64);
#endif
