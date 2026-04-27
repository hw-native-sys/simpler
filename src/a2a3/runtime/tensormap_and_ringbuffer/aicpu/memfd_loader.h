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
 * @file memfd_loader.h
 * @brief Memory file descriptor based SO loading for AICPU environment
 */

#ifndef SRC_A2A3_RUNTIME_TENSORMAP_AND_RINGBUFFER_AICPU_MEMFD_LOADER_H_
#define SRC_A2A3_RUNTIME_TENSORMAP_AND_RINGBUFFER_AICPU_MEMFD_LOADER_H_

// Enable GNU extensions for memfd_create and MFD_CLOEXEC
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

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
#include "aicpu/device_time.h"

/**
 * Load orchestration SO using memfd
 */
static inline int load_orchestration_so_with_memfd(
    const void *so_data, size_t so_size, int orch_thread_num, void **out_handle, char *out_so_path, int *out_memfd
) {
    *out_handle = nullptr;
    *out_memfd = -1;
    out_so_path[0] = '\0';

    if (so_data == nullptr || so_size == 0) {
        return -1;
    }

    uint64_t t0, t1;

    // Create memfd
    t0 = get_sys_cnt_aicpu();
    int fd = memfd_create("libdevice_orch", MFD_CLOEXEC);
    t1 = get_sys_cnt_aicpu();
    DEV_ALWAYS("[memfd_profiling] memfd_create cycles: %lu", t1 - t0);

    if (fd < 0) {
        DEV_INFO("memfd_create failed: errno=%d", errno);
        return -1;
    }

    // Write SO data to memfd
    t0 = get_sys_cnt_aicpu();
    ssize_t written = write(fd, so_data, so_size);
    t1 = get_sys_cnt_aicpu();
    DEV_INFO("[memfd_profiling] write %zu bytes cycles: %lu", so_size, t1 - t0);

    if (written < 0) {
        DEV_INFO("memfd write failed: errno=%d", errno);
        close(fd);
        return -1;
    }
    if (written != static_cast<ssize_t>(so_size)) {
        DEV_INFO("memfd partial write: %zd/%zu", written, so_size);
        close(fd);
        return -1;
    }

    // Reset file position to beginning before dlopen
    t0 = get_sys_cnt_aicpu();
    lseek(fd, 0, SEEK_SET);
    t1 = get_sys_cnt_aicpu();
    DEV_INFO("[memfd_profiling] lseek cycles: %lu", t1 - t0);

    // Construct /proc/self/fd/N path for symlink target
    char proc_fd_path[256];
    snprintf(proc_fd_path, sizeof(proc_fd_path), "/proc/self/fd/%d", fd);

    // Create a symlink to /proc/self/fd/N with a "normal" path
    // This bypasses the AICPU dynamic linker's issue with /proc/self/fd/N paths
    char link_path[256];
    snprintf(link_path, sizeof(link_path), "/tmp/libdevice_orch_%d_%d.so", getpid(), orch_thread_num);

    t0 = get_sys_cnt_aicpu();
    int symlink_rc = symlink(proc_fd_path, link_path);
    t1 = get_sys_cnt_aicpu();
    DEV_INFO("[memfd_profiling] symlink cycles: %lu", t1 - t0);

    if (symlink_rc != 0) {
        DEV_INFO("symlink failed: errno=%d", errno);
        close(fd);
        return -1;
    }

    snprintf(out_so_path, 256, "%s", link_path);

    // Try dlopen from the symlink
    t0 = get_sys_cnt_aicpu();
    dlerror();
    void *handle = dlopen(out_so_path, RTLD_LAZY | RTLD_LOCAL);
    t1 = get_sys_cnt_aicpu();
    DEV_INFO("[memfd_profiling] dlopen cycles: %lu", t1 - t0);

    // Clean up symlink immediately after dlopen (dlopen has its own reference)
    unlink(link_path);

    if (handle == nullptr) {
        const char *dl_err = dlerror();
        DEV_INFO("dlopen from memfd symlink failed: %s", dl_err ? dl_err : "unknown");
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

#endif  // SRC_A2A3_RUNTIME_TENSORMAP_AND_RINGBUFFER_AICPU_MEMFD_LOADER_H_
