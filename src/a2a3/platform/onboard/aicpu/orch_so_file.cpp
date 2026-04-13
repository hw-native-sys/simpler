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

int32_t create_orch_so_file(const char *dir, char *out_path, size_t out_path_size) {
    // Pid-based naming: AICPU device libc may lack mkstemps, and only one
    // runtime runs per device process, so pid uniqueness is sufficient.
    int32_t written = snprintf(out_path, out_path_size, "%s/libdevice_orch_%d.so", dir, getpid());
    if (written < 0 || static_cast<size_t>(written) >= out_path_size) {
        return -1;
    }
    return open(out_path, O_WRONLY | O_CREAT | O_TRUNC, 0755);
}
