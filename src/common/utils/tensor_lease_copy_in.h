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
#include "worker/runtime_c_api.h"

/**
 * Copy every input-bearing lease's current host bytes to its device buffer.
 *
 * The start-of-execution half of a run's tensor IO. Recording a lease settles
 * which device buffer a caller tensor gets; this settles what is in it. The
 * current adapter calls it during `simpler_prepare_run`, after the bind.
 *
 * An input-bearing lease with a size and no usable endpoint is an error, not a
 * transfer to skip: its device buffer is real and the kernel will read it, so
 * passing over it would hand the run whatever those bytes already held. Only a
 * lease that carries no input, or carries no bytes, is skipped.
 *
 * Host-side only, which is why this is separate from `utils/tensor_lease.h`:
 * that header is reachable from device translation units and this one reaches
 * the host API. Returns the first failing lease's code, having attempted no
 * further ones — a partially staged input set must not be launched. The ledger
 * and the buffers behind it are left standing for the caller to release.
 */
inline int copy_in_tensor_leases(const std::vector<TensorLease> &leases, const HostApi *api) {
    for (const TensorLease &lease : leases) {
        if (!lease.needs_copy_in || lease.size == 0) {
            continue;
        }
        if (lease.dev_ptr == nullptr || lease.host_ptr == nullptr) {
            return PTO_RUNTIME_ERR_INTERNAL;
        }
        const int rc = api->copy_to_device(lease.dev_ptr, lease.host_ptr, lease.size);
        if (rc != 0) {
            return rc;
        }
    }
    return 0;
}
