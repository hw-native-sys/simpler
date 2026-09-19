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
 * Onboard host common helpers — `KernelArgsHelper` implementation.
 *
 * Linked into both a2a3 and a5 `libhost_runtime.so`. The arch-specific
 * `KernelArgs` layout is brought in via `common/kernel_args.h` on the
 * include path (each arch CMake adds the right one).
 */

#include "device_runner_helpers.h"

#include <runtime/rt.h>

#include "acl/error_codes/rt_error_codes.h"
#include "common/unified_log.h"
#include "host/acl_error_log.h"

namespace {

int query_stream_nonblocking(rtStream_t stream, const char *name) {
    if (stream == nullptr) {
        LOG_ERROR("rtStreamQuery (%s) received a null stream", name);
        return SIMPLER_NATIVE_RUN_POLL_ERROR;
    }

    const rtError_t rc = rtStreamQuery(stream);
    if (rc == RT_ERROR_NONE) return SIMPLER_NATIVE_RUN_POLL_COMPLETE;
    if (rc == ACL_ERROR_RT_STREAM_NOT_COMPLETE) return SIMPLER_NATIVE_RUN_POLL_NOT_READY;

    LOG_ERROR("rtStreamQuery (%s) failed: %d", name, static_cast<int>(rc));
    ACL_LOG_ERROR_DETAIL(rc);
    return SIMPLER_NATIVE_RUN_POLL_ERROR;
}

int query_stream_error(rtStream_t stream, const char *name) {
    if (stream == nullptr) {
        LOG_ERROR("rtStreamQuery (%s) received a null stream", name);
        return PTO_RUNTIME_ERR_INTERNAL;
    }

    const rtError_t rc = rtStreamQuery(stream);
    if (rc == RT_ERROR_NONE || rc == ACL_ERROR_RT_STREAM_NOT_COMPLETE) return 0;

    LOG_ERROR("rtStreamQuery (%s) reports a device error: %d", name, static_cast<int>(rc));
    ACL_LOG_ERROR_DETAIL(rc);
    return static_cast<int>(rc);
}

}  // namespace

int query_stream_pair_nonblocking(rtStream_t aicpu_stream, rtStream_t aicore_stream) {
    // Query both even when the first is pending. Besides making completion a
    // true pair fence, this preserves an error from either device queue.
    const int aicpu_rc = query_stream_nonblocking(aicpu_stream, "AICPU");
    const int aicore_rc = query_stream_nonblocking(aicore_stream, "AICore");
    if (aicpu_rc == SIMPLER_NATIVE_RUN_POLL_ERROR || aicore_rc == SIMPLER_NATIVE_RUN_POLL_ERROR) {
        return SIMPLER_NATIVE_RUN_POLL_ERROR;
    }
    if (aicpu_rc == SIMPLER_NATIVE_RUN_POLL_COMPLETE && aicore_rc == SIMPLER_NATIVE_RUN_POLL_COMPLETE) {
        return SIMPLER_NATIVE_RUN_POLL_COMPLETE;
    }
    return SIMPLER_NATIVE_RUN_POLL_NOT_READY;
}

int query_stream_pair_error(rtStream_t aicpu_stream, rtStream_t aicore_stream) {
    const int aicpu_rc = query_stream_error(aicpu_stream, "AICPU");
    if (aicpu_rc != 0) return aicpu_rc;
    return query_stream_error(aicore_stream, "AICore");
}

int KernelArgsHelper::prepare_runtime_args(
    const Runtime &host_runtime, MemoryAllocator &allocator, SlotPersistentArgs &slot
) {
    if (runtime_args_state_ == RuntimeArgsState::Prepared) return PTO_RUNTIME_ERR_INVALID_STATE;
    release_run_view();
    allocator_ = &allocator;

    // Both runtime variants publish the descriptor at offset zero. Host-only
    // orchestration state and tensor leases remain outside this snapshot, and so
    // does any device-initialized tail the descriptor ends in: the block is sized
    // to the whole descriptor because the device addresses that range inside it,
    // while only the uploaded prefix is snapshotted and copied.
    const uint64_t runtime_extent = runtime_device_extent_size(host_runtime);
    // The length is a property of the runtime variant, which is fixed for a
    // runner, so a committed block always fits. A mismatch would mean the
    // block belongs to a different variant than the run being prepared.
    if (slot.runtime_args != nullptr && slot.runtime_bytes != runtime_extent) {
        LOG_ERROR(
            "runtime_args block is %llu bytes but this run needs %llu",
            static_cast<unsigned long long>(slot.runtime_bytes), static_cast<unsigned long long>(runtime_extent)
        );
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    if (slot.runtime_args == nullptr) {
        void *runtime_dev = allocator_->alloc(runtime_extent);
        if (runtime_dev == nullptr) {
            LOG_ERROR("Alloc for runtime_args failed");
            return PTO_RUNTIME_ERR_INTERNAL;
        }
        slot.runtime_args = reinterpret_cast<Runtime *>(runtime_dev);
        slot.runtime_bytes = runtime_extent;
    }
    runtime_image_.prepare(host_runtime);
    args.runtime_args = slot.runtime_args;
    runtime_args_state_ = RuntimeArgsState::Prepared;
    return 0;
}

int KernelArgsHelper::publish_runtime_args() {
    if (runtime_args_state_ != RuntimeArgsState::Prepared) return PTO_RUNTIME_ERR_INVALID_STATE;
    if (args.runtime_args == nullptr) return PTO_RUNTIME_ERR_INTERNAL;
    // The consumed snapshot is neither pending nor published during the copy.
    // Reentrant publish is rejected; copy failure leaves fresh prepare admissible.
    runtime_args_state_ = RuntimeArgsState::Empty;
    const int rc = runtime_image_.publish([this](const void *source, size_t bytes) {
        return rtMemcpy(args.runtime_args, bytes, source, bytes, RT_MEMCPY_HOST_TO_DEVICE);
    });
    if (rc != 0) {
        LOG_ERROR("runtime metadata publication failed: %d", rc);
        args.runtime_args = nullptr;
    } else {
        runtime_args_state_ = RuntimeArgsState::Published;
    }
    return rc;
}

int release_slot_persistent_args(SlotPersistentArgs &slot, MemoryAllocator &allocator) {
    int first_error = 0;
    if (slot.runtime_args != nullptr) {
        const int rc = allocator.free(slot.runtime_args);
        if (rc != 0) {
            if (first_error == 0) first_error = rc;
        } else {
            slot.runtime_args = nullptr;
            slot.runtime_bytes = 0;
        }
    }
    return first_error;
}

void abandon_slot_persistent_args(SlotPersistentArgs &slot) {
    slot.runtime_args = nullptr;
    slot.runtime_bytes = 0;
}
