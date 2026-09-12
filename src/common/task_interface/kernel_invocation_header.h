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
 * this fixed header followed by `payload_bytes` of runtime-specific payload.
 * The header is the shared envelope both runtimes use; the payload format
 * under it belongs to each runtime (tensormap_and_ringbuffer carries
 * graph-build input, host_build_graph carries a serialized graph blob) and is
 * not constrained here. AICPU dispatch consumers must validate identity,
 * generation, and capacity before consuming the payload, and reject nonzero
 * host_copy_tensor_count or reserved_. Producers must zero-initialize the
 * complete header before assigning invocation fields.
 *
 * Both sides of this wire are produced by the same build (`build_runtimes.py`
 * emits the host runtime and the AICPU executor into one
 * `build/lib/{arch}/{variant}/{runtime}/`), so the struct carries no version
 * or size negotiation: evolving it means changing both sides in one tree.
 * The layout is still shipped by memcpy through CANN's HostArgs deep copy, so
 * the struct is POD, position-independent, and carries no pointers.
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "execution_mode.h"

typedef struct SimplerKernelInvocationHeader {
    uint32_t mode;       /* SimplerExecutionMode of the issuing context */
    int32_t callable_id; /* target callable; matches the prepared registration */
    /* Occupancy generation of the residency slot `callable_id` resolves to —
       a property of the slot, not of the callable in it. The counter advances
       when the slot takes a different tenant, so a generation carried by the
       callable could not detect slot reuse. Zero means "not recorded"; a live
       generation starts at 1, matching PipelineSlotLease and CanonicalIdentity.
       The comparison belongs on the AICPU dispatch path because replay does
       not return to the host, so a stale captured snapshot has to be caught
       on-device. */
    uint64_t generation;
    /* Byte length of the runtime-specific payload that follows this header. */
    uint64_t payload_bytes;
    /* Arg counts of this invocation. ChipCallable's sig_count includes both
       tensors and scalars. Consumers count its ArgDirection::SCALAR entries,
       compare scalar_count with that count, and compare tensor_count with
       sig_count minus that count. */
    int32_t tensor_count;
    int32_t scalar_count;
    /* Reserved host-only duplicate tensor count; must be zero. */
    int32_t host_copy_tensor_count;
    uint32_t reserved_; /* Must be zero. */
} SimplerKernelInvocationHeader;

#ifdef __cplusplus
#include <type_traits>

static_assert(
    std::is_trivially_copyable_v<SimplerKernelInvocationHeader> &&
    std::is_standard_layout_v<SimplerKernelInvocationHeader>
);
static_assert(sizeof(SimplerKernelInvocationHeader) == 40);
static_assert(offsetof(SimplerKernelInvocationHeader, mode) == 0);
static_assert(offsetof(SimplerKernelInvocationHeader, callable_id) == 4);
static_assert(offsetof(SimplerKernelInvocationHeader, generation) == 8);
static_assert(offsetof(SimplerKernelInvocationHeader, payload_bytes) == 16);
static_assert(offsetof(SimplerKernelInvocationHeader, tensor_count) == 24);
static_assert(offsetof(SimplerKernelInvocationHeader, scalar_count) == 28);
static_assert(offsetof(SimplerKernelInvocationHeader, host_copy_tensor_count) == 32);
static_assert(offsetof(SimplerKernelInvocationHeader, reserved_) == 36);
#endif
