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

#include <cstddef>
#include "runtime.h"
#include "aicpu/platform_entry_args.h"
#include "tensormap_and_ringbuffer/entry_args.h"

namespace simpler::tmr {
inline constexpr size_t entry_tensor_offset =
    offsetof(DeviceRuntimeLaunchDesc, orch_args_storage_) + offsetof(EntryArgsStorage, tensors_);

inline bool valid_entry_cache_route(const PlatformEntryArgs &view) {
    if (!valid_entry_cache_maintenance(view.cache_maintenance)) return false;
    if (view.cache_maintenance == static_cast<uint32_t>(EntryCacheMaintenance::Live))
        return view.source == static_cast<uint32_t>(EntryArgsSource::LaunchEnvelope);
    if (view.cache_maintenance == static_cast<uint32_t>(EntryCacheMaintenance::FullFreshen))
        return view.source == static_cast<uint32_t>(EntryArgsSource::Descriptor);
    return true;
}

static_assert(offsetof(Runtime, dev) == 0);
static_assert(entry_tensor_offset % 64 == 0);
static_assert(offsetof(DeviceRuntimeLaunchDesc, workers) % 64 == 0);
static_assert(sizeof(Tensor) % 64 == 0);

// Called after header classification and before descriptor entry values are consumed.
template <typename Clean>
bool freshen_entry_descriptor(Runtime &runtime, const PlatformEntryArgs &view, Clean clean) {
    if (!valid_entry_cache_maintenance(view.cache_maintenance)) return false;
    if (view.cache_maintenance == static_cast<uint32_t>(EntryCacheMaintenance::Live))
        return view.source == static_cast<uint32_t>(EntryArgsSource::LaunchEnvelope);
    if (view.cache_maintenance == static_cast<uint32_t>(EntryCacheMaintenance::Full)) return true;
    if (view.source != static_cast<uint32_t>(EntryArgsSource::Descriptor)) return false;
    auto &storage = runtime.dev.orch_args_storage_;
    clean(&storage, offsetof(EntryArgsStorage, tensors_));
    const uint32_t tensors = runtime.get_entry_tensor_count();
    const uint32_t scalars = runtime.get_entry_scalar_count();
    if (runtime.get_entry_args_source() != EntryArgsSource::Descriptor || tensors > CHIP_MAX_TENSOR_ARGS ||
        scalars > CHIP_MAX_SCALAR_ARGS || storage.tensor_count() < 0 || storage.scalar_count() < 0 ||
        static_cast<uint32_t>(storage.tensor_count()) != tensors ||
        static_cast<uint32_t>(storage.scalar_count()) != scalars)
        return false;
    if (tensors != 0)
        clean(
            reinterpret_cast<unsigned char *>(&runtime) + entry_tensor_offset,
            static_cast<size_t>(tensors) * sizeof(Tensor)
        );
    return true;
}

inline size_t live_entry_cleanup_prefix(const Runtime &runtime, const PlatformEntryArgs &view, bool snapshot_ok) {
    if (!snapshot_ok || view.cache_maintenance != static_cast<uint32_t>(EntryCacheMaintenance::Live) ||
        view.source != static_cast<uint32_t>(EntryArgsSource::LaunchEnvelope) ||
        runtime.get_entry_args_source() != EntryArgsSource::LaunchEnvelope ||
        view.offset != LAUNCH_ENVELOPE_HEADER_BYTES || view.args_base == nullptr ||
        view.tensor_count > CHIP_MAX_TENSOR_ARGS || view.scalar_count > CHIP_MAX_SCALAR_ARGS ||
        view.tensor_count != runtime.get_entry_tensor_count() || view.scalar_count != runtime.get_entry_scalar_count())
        return sizeof(runtime.dev);
    const size_t prefix = entry_tensor_offset + static_cast<size_t>(view.tensor_count) * sizeof(Tensor);
    return prefix < offsetof(DeviceRuntimeLaunchDesc, workers) ? prefix : sizeof(runtime.dev);
}
template <typename Clean>
void clean_entry_runtime(Runtime &runtime, const PlatformEntryArgs &view, bool snapshot_ok, Clean clean) {
    const size_t prefix = live_entry_cleanup_prefix(runtime, view, snapshot_ok);
    clean(&runtime, prefix);
    if (prefix < offsetof(DeviceRuntimeLaunchDesc, workers)) {
        constexpr size_t suffix = offsetof(DeviceRuntimeLaunchDesc, workers);
        clean(reinterpret_cast<unsigned char *>(&runtime) + suffix, sizeof(runtime.dev) - suffix);
    }
}
}  // namespace simpler::tmr
