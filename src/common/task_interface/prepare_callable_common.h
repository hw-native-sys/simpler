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
 * Runtime-agnostic helper for the kernel-upload half of prepare_callable_impl.
 *
 * Each runtime variant (host_build_graph, tensormap_and_ringbuffer, ...) needs
 * to: upload the ChipCallable buffer to device once, then translate each child
 * kernel's storage offset into a device address that AICPU dispatch can read
 * out of Runtime::func_id_to_addr_[]. The upload + offset arithmetic does not
 * depend on the surrounding runtime, only on the ChipCallable layout and the
 * platform's upload entry point. Callers retain the func_id range check
 * because RUNTIME_MAX_FUNC_ID lives in each runtime's own headers.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "callable.h"

struct ChildKernelAddr {
    int func_id;
    uint64_t device_addr;
};

/**
 * Output bundle from a runtime's prepare_callable_impl() — everything the
 * platform layer needs to register a callable_id with its DeviceRunner.
 *
 * Replaces the per-field Runtime::pending_* sidecar that earlier callsites
 * used as a "C ABI struggling with multiple return values" hack. Each
 * runtime variant fills only the subset it produces:
 *
 *   - host_build_graph: kernel_addrs + host_dlopen_handle + host_orch_func_ptr
 *                       (orch_so_* stay null since orch SO never crosses
 *                       host/device on this runtime)
 *   - tensormap_and_ringbuffer: kernel_addrs + orch_so_data + orch_so_size
 *                       + func_name + config_name (orch SO is uploaded to
 *                       device by DeviceRunner; host does no dlopen)
 *
 * func_name / config_name are non-empty only for the trb path; the hbg path
 * resolves its entry symbol during prepare_callable_impl and stores the
 * resulting function pointer in host_orch_func_ptr directly.
 */
struct PreparedCallableArtifacts {
    std::vector<ChildKernelAddr> kernel_addrs;
    void *host_dlopen_handle{nullptr};  // hbg only
    void *host_orch_func_ptr{nullptr};  // hbg only
    const void *orch_so_data{nullptr};  // trb only
    size_t orch_so_size{0};             // trb only
    std::string func_name;              // trb only (orch entry symbol)
    std::string config_name;            // trb only (orch config symbol)
};

/**
 * Upload the ChipCallable buffer via `upload_fn` and compute the device-side
 * address of every child kernel.
 *
 * @param callable   ChipCallable to upload. May have child_count() == 0, in
 *                   which case `out` is cleared and 0 is returned.
 * @param upload_fn  HostApi::upload_chip_callable_buffer — declared as
 *                   `uint64_t (*)(const void *)` to avoid pulling runtime
 *                   headers into task_interface. Must not be null.
 * @param out        Cleared on entry; on success, populated with one
 *                   {func_id, device_addr} entry per child kernel. Caller is
 *                   responsible for validating func_id against its runtime's
 *                   RUNTIME_MAX_FUNC_ID before writing to func_id_to_addr_[].
 * @return 0 on success (including the child_count == 0 case), -1 on argument
 *         error or upload failure.
 */
inline int upload_and_collect_child_addrs(
    const ChipCallable *callable, uint64_t (*upload_fn)(const void *), std::vector<ChildKernelAddr> *out
) {
    if (callable == nullptr || upload_fn == nullptr || out == nullptr) return -1;
    out->clear();
    if (callable->child_count() <= 0) return 0;

    uint64_t chip_dev = upload_fn(callable);
    if (chip_dev == 0) return -1;

    constexpr size_t kHeaderSize = offsetof(ChipCallable, storage_);
    out->reserve(static_cast<size_t>(callable->child_count()));
    for (int32_t i = 0; i < callable->child_count(); ++i) {
        uint64_t child_dev = chip_dev + kHeaderSize + callable->child_offset(i);
        out->push_back({callable->child_func_id(i), child_dev});
    }
    return 0;
}
