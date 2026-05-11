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
 * @file orch_so_file.h
 * @brief Orchestration SO File Creation Interface for AICPU
 *
 * Creates a writable, executable-mode file under a candidate directory for
 * staging the device orchestration shared library prior to dlopen.
 *
 * Platform Support:
 * - a2a3 (onboard): pid-based naming via open() with mode 0755. AICPU device
 *   libc may not provide mkstemps, and only one runtime runs per device process.
 * - a2a3sim (simulation): mkstemps() with fchmod(0755). Multiple sim workers
 *   can share a process, so names must be unique per call.
 */

#ifndef PLATFORM_AICPU_ORCH_SO_FILE_H_
#define PLATFORM_AICPU_ORCH_SO_FILE_H_

#include <cstddef>
#include <cstdint>

/**
 * Create a unique orchestration SO file under `dir`.
 *
 * On success, writes the full chosen path into `out_path` (null-terminated)
 * and returns an open writable fd (caller must close). Permissions are set
 * so the file is executable (0755) and suitable for dlopen.
 *
 * On failure (path too long, directory not writable, etc.), returns -1.
 * Caller is expected to try the next candidate directory.
 *
 * @param dir            Candidate directory (e.g. "/tmp")
 * @param callable_id    Per-callable_id table slot id (>= 0). Required for
 *                       uniqueness on the onboard path so concurrently-
 *                       resident orch SOs (one per cid) do not collide on
 *                       the same on-disk file. Pass -1 for the legacy
 *                       single-slot dispatch path.
 * @param out_path       Buffer that receives the full file path on success
 * @param out_path_size  Size of `out_path` in bytes
 * @return Open writable fd on success, -1 on failure
 */
int32_t create_orch_so_file(const char *dir, int32_t callable_id, char *out_path, size_t out_path_size);

#endif  // PLATFORM_AICPU_ORCH_SO_FILE_H_
