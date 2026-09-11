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

#include <vector>

#include "common/host_api.h"
#include "utils/tensor_lease.h"

/** Per-kind tally of one release pass, for the caller's diagnostic. */
struct TensorLeaseReleaseCounts {
    int freed = 0;
    int buffer_noop = 0;
    int external_noop = 0;
};

/**
 * Release every lease of a finished run and empty the ledger.
 *
 * Host-side only, which is why this is separate from `utils/tensor_lease.h`:
 * that header is reachable from device translation units and this one reaches
 * the host API. Reports counts rather than logging them — the runtimes that
 * call it do not share one logging backend.
 */
inline TensorLeaseReleaseCounts release_tensor_leases(std::vector<TensorLease> &leases, const HostApi *api) {
    TensorLeaseReleaseCounts counts;
    for (TensorLease &lease : leases) {
        if (lease.dev_ptr == nullptr) {
            continue;
        }
        switch (lease.release_kind) {
        case TensorReleaseKind::Free:
            api->device_free(lease.dev_ptr);
            ++counts.freed;
            break;
        case TensorReleaseKind::BufferNoop:
            ++counts.buffer_noop;
            break;
        case TensorReleaseKind::ExternalNoop:
            ++counts.external_noop;
            break;
        }
    }
    leases.clear();
    return counts;
}
