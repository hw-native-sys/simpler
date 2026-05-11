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
#include "aicpu/orch_so_file.h"

#include <fcntl.h>
#include <unistd.h>

#include <cstdio>

int32_t create_orch_so_file(const char *dir, int32_t callable_id, char *out_path, size_t out_path_size) {
    // Pid + callable_id naming: AICPU device libc may lack mkstemps. With
    // per-callable_id dispatch, multiple orch SOs can be resident in the
    // same device process at once (one per cid in `orch_so_table_`), so
    // the on-disk file name must be unique per cid — otherwise the
    // second cid's `O_TRUNC` would silently shred the first cid's already
    // dlopen'd file image and the next launch on cid=0 would SIGBUS.
    // callable_id < 0 is the legacy single-slot path: pid alone is fine.
    int32_t written;
    if (callable_id >= 0) {
        written = snprintf(out_path, out_path_size, "%s/libdevice_orch_%d_%d.so", dir, getpid(), callable_id);
    } else {
        written = snprintf(out_path, out_path_size, "%s/libdevice_orch_%d.so", dir, getpid());
    }
    if (written < 0 || static_cast<size_t>(written) >= out_path_size) {
        return -1;
    }
    return open(out_path, O_WRONLY | O_CREAT | O_TRUNC, 0755);
}
