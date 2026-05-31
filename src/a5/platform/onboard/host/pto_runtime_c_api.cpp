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
 * PTO Runtime C API — a5 arch-specific entries
 *
 * The shared c_api glue (simpler_init / prepare_callable / run_prepared /
 * device_*_ctx / etc.) lives in
 * `src/common/platform/onboard/host/c_api_shared.cpp` and is linked into
 * this same `libhost_runtime.so`. This file keeps only the entries that
 * need the concrete a5 `DeviceRunner` subclass (`create_device_context`)
 * plus the ACL + comm_* placeholders that ChipWorker dlsyms
 * unconditionally — the distributed runtime is not yet implemented on a5,
 * so these return not-supported sentinels at call time rather than having
 * ChipWorker probe each symbol individually.
 */

#include "device_runner.h"
#include "pto_runtime_c_api.h"

extern "C" {

DeviceContextHandle create_device_context(void) {
    try {
        return static_cast<DeviceContextHandle>(new DeviceRunner());
    } catch (...) {
        return NULL;
    }
}

/* ===========================================================================
 * ACL + comm_* placeholders (distributed runtime not yet implemented on a5)
 *
 * These exist only to satisfy ChipWorker's unconditional dlsym of the extension
 * surface — the contract is "every host_runtime.so exports the full set; a
 * runtime without a real implementation returns a not-supported result at
 * call time" rather than having ChipWorker probe each symbol individually.
 * When a5 grows real HCCL / sim distributed support these stubs get replaced
 * wholesale; no ChipWorker changes are needed.
 * =========================================================================== */

int ensure_acl_ready_ctx(DeviceContextHandle ctx, int device_id) {
    (void)ctx;
    (void)device_id;
    return 0;
}

void *create_comm_stream_ctx(DeviceContextHandle ctx) {
    (void)ctx;
    return NULL;
}

int destroy_comm_stream_ctx(DeviceContextHandle ctx, void *stream) {
    (void)ctx;
    (void)stream;
    return 0;
}

void *comm_init(int rank, int nranks, void *stream, const char *rootinfo_path) {
    (void)rank;
    (void)nranks;
    (void)stream;
    (void)rootinfo_path;
    return NULL;  // distributed runtime not yet supported on a5
}

int comm_alloc_windows(void *handle, size_t win_size, uint64_t *device_ctx_out) {
    (void)handle;
    (void)win_size;
    (void)device_ctx_out;
    return -1;
}

int comm_get_local_window_base(void *handle, uint64_t *base_out) {
    (void)handle;
    (void)base_out;
    return -1;
}

int comm_get_window_size(void *handle, size_t *size_out) {
    (void)handle;
    (void)size_out;
    return -1;
}

int comm_derive_context(
    void *handle, const uint32_t *rank_ids, size_t rank_count, uint32_t domain_rank, size_t window_offset,
    size_t window_size, uint64_t *device_ctx_out
) {
    (void)handle;
    (void)rank_ids;
    (void)rank_count;
    (void)domain_rank;
    (void)window_offset;
    (void)window_size;
    (void)device_ctx_out;
    return -1;
}

int comm_alloc_domain_windows(
    void *handle, uint64_t allocation_id, const uint32_t *rank_ids, size_t rank_count, uint32_t domain_rank,
    size_t window_size, uint64_t *device_ctx_out, uint64_t *local_window_base_out
) {
    (void)handle;
    (void)allocation_id;
    (void)rank_ids;
    (void)rank_count;
    (void)domain_rank;
    (void)window_size;
    (void)device_ctx_out;
    (void)local_window_base_out;
    return -1;
}

int comm_release_domain_windows(void *handle, uint64_t allocation_id, size_t rank_count, uint32_t domain_rank) {
    (void)handle;
    (void)allocation_id;
    (void)rank_count;
    (void)domain_rank;
    return -1;
}

int comm_barrier(void *handle) {
    (void)handle;
    return -1;
}

int comm_destroy(void *handle) {
    (void)handle;
    return -1;
}

}  // extern "C"
