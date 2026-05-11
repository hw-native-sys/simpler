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
// mkstemps is a BSD extension: glibc gates it behind _DEFAULT_SOURCE (hidden
// under strict -std=c++17 which sets __STRICT_ANSI__).  macOS libc exposes it
// unconditionally, so this only matters on Linux.
#ifndef _DEFAULT_SOURCE
#define _DEFAULT_SOURCE
#endif

#include "aicpu/orch_so_file.h"

#include <fcntl.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>

int32_t create_orch_so_file(const char *dir, int32_t callable_id, char *out_path, size_t out_path_size) {
    // mkstemps: multiple sim workers can share a process, so names must be
    // unique per call.  The "XXXXXX" template is replaced in-place.
    // callable_id is embedded purely for log readability (mkstemps already
    // guarantees uniqueness regardless).
    int32_t written;
    if (callable_id >= 0) {
        written = snprintf(out_path, out_path_size, "%s/libdevice_orch_cid%d_XXXXXX.so", dir, callable_id);
    } else {
        written = snprintf(out_path, out_path_size, "%s/libdevice_orch_XXXXXX.so", dir);
    }
    if (written < 0 || static_cast<size_t>(written) >= out_path_size) {
        return -1;
    }

    constexpr int32_t kSuffixLen = 3;  // strlen(".so")
    int32_t fd = mkstemps(out_path, kSuffixLen);
    if (fd < 0) {
        return -1;
    }
    if (fchmod(fd, 0755) != 0) {
        close(fd);
        unlink(out_path);
        return -1;
    }
    return fd;
}
