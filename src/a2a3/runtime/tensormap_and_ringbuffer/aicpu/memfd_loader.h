/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

/**
 * @file memfd_loader.h
 * @brief Memory file descriptor based SO loading for AICPU environment
 */

// Enable GNU extensions for memfd_create and MFD_CLOEXEC
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#ifndef MEMFD_LOADER_H
#define MEMFD_LOADER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <cstring>
#include <cstdio>

#include "aicpu/device_log.h"

/**
 * Load orchestration SO using memfd
 */
static inline int load_orchestration_so_with_memfd(
    const void *so_data,
    size_t so_size,
    int orch_thread_num,
    void **out_handle,
    char *out_so_path,
    int *out_memfd
) {
    *out_handle = nullptr;
    *out_memfd = -1;
    out_so_path[0] = '\0';

    if (so_data == nullptr || so_size == 0) {
        return -1;
    }

    // Create memfd
    int fd = memfd_create("libdevice_orch", MFD_CLOEXEC);

    if (fd < 0) {
        return -1;
    }

    // Write SO data to memfd
    ssize_t written = write(fd, so_data, so_size);

    if (written < 0) {
        close(fd);
        return -1;
    }
    if (written != static_cast<ssize_t>(so_size)) {
        close(fd);
        return -1;
    }

    // Reset file position to beginning before dlopen
    lseek(fd, 0, SEEK_SET);

    // Construct /proc/self/fd/N path for symlink target
    char proc_fd_path[256];
    snprintf(proc_fd_path, sizeof(proc_fd_path), "/proc/self/fd/%d", fd);

    // Create a symlink to /proc/self/fd/N with a "normal" path
    // This bypasses the AICPU dynamic linker's issue with /proc/self/fd/N paths
    char link_path[256];
    snprintf(link_path, sizeof(link_path), "/tmp/libdevice_orch_%d_%d.so", getpid(), orch_thread_num);

    int symlink_rc = symlink(proc_fd_path, link_path);
    if (symlink_rc != 0) {
        close(fd);
        return -1;
    }

    snprintf(out_so_path, 256, "%s", link_path);

    // Try dlopen from the symlink
    dlerror();
    void *handle = dlopen(out_so_path, RTLD_LAZY | RTLD_LOCAL);

    // Clean up symlink immediately after dlopen (dlopen has its own reference)
    unlink(link_path);

    if (handle == nullptr) {
        close(fd);
        return -1;
    }

    *out_handle = handle;
    *out_memfd = fd;
    return 0;
}

/**
 * Cleanup memfd-based SO
 */
static inline void cleanup_memfd_so(int memfd, void *handle) {
    if (handle != nullptr) {
        dlclose(handle);
    }
    if (memfd >= 0) {
        close(memfd);
    }
}

#ifdef __cplusplus
}
#endif

#endif  // MEMFD_LOADER_H
