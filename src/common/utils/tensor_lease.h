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

#pragma once

#include <cstddef>

/**
 * One host runtime's record of a caller tensor it staged to the device, shared
 * by every runtime that stages them.
 *
 * A `Runtime` holds the run's leases so validate can copy written tensors back
 * and then release each one the way its provenance requires. The release kind
 * is what keeps that decision out of the release loop's hands: a bump slice and
 * an owned allocation are both a `dev_ptr`, and only the staging site knows
 * which it handed over.
 *
 * Declared here rather than beside either `Runtime` because both are included
 * by AICore and AICPU translation units, which is also why this header pulls in
 * no host API: the release itself lives in `utils/tensor_lease_release.h`.
 */
enum class TensorReleaseKind {
    // device_free at end of run — the staging site owns this allocation.
    Free,
    // A slice of a buffer that outlives the run; releasing it is a no-op.
    BufferNoop,
    // Owned by someone else entirely (caller device memory); never released here.
    ExternalNoop,
};

struct TensorLease {
    void *host_ptr;
    void *dev_ptr;
    size_t size;
    // false for read-only INPUT tensors: they are never written by the kernel,
    // so the end-of-run D2H copy-back is skipped. OUTPUT/INOUT/unknown
    // keep the safe default of copying back.
    bool needs_copy_back = true;
    TensorReleaseKind release_kind = TensorReleaseKind::Free;
};
