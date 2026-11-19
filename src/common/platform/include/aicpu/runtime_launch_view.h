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
#include <cstdint>
#include <cstring>

#include "common/kernel_args.h"
#include "runtime.h"
#include "common/kernel_invocation_wire.h"

// RTS argument storage guarantees no Runtime cache-line alignment. Each AICPU
// invocation keeps its descriptor view until its own device entry returns.
class RuntimeLaunchView {
public:
    RuntimeLaunchView() = default;
    RuntimeLaunchView(const RuntimeLaunchView &) = delete;
    RuntimeLaunchView &operator=(const RuntimeLaunchView &) = delete;
    Runtime *bind(const KernelArgs &args) {
        if (args.entry_args_source != static_cast<uint32_t>(EntryArgsSource::KernelInvocation))
            return bind(args.runtime_args, args.runtime_descriptor_bytes);
        if (args.runtime_args != nullptr || args.entry_args_offset != LAUNCH_ENVELOPE_HEADER_BYTES ||
            args.runtime_descriptor_bytes != RUNTIME_LAUNCH_DESCRIPTOR_BYTES ||
            args.entry_tensor_count > CHIP_MAX_TENSOR_ARGS || args.entry_scalar_count > CHIP_MAX_SCALAR_ARGS ||
            args.entry_tensor_count + args.entry_scalar_count > CHIP_MAX_TENSOR_ARGS)
            return nullptr;

        const auto section = args.runtime_section;
        const size_t offset = simpler::tmr::kernel_runtime_descriptor_offset(
            args.entry_tensor_count, args.entry_scalar_count, alignof(DeviceRuntimeLaunchDesc)
        );
        if (section.descriptor_offset != offset || section.packet_bytes < offset ||
            section.packet_bytes - offset != args.runtime_descriptor_bytes)
            return nullptr;

        const auto *base = reinterpret_cast<const unsigned char *>(&args);
        SimplerKernelInvocationHeader header{};
        simpler::tmr::EagerKernelBinding binding{};
        std::memcpy(&header, base + args.entry_args_offset, sizeof(header));
        std::memcpy(&binding, base + args.entry_args_offset + sizeof(header), sizeof(binding));
        if (header.mode != SIMPLER_MODE_KERNEL ||
            header.tensor_count != static_cast<int32_t>(args.entry_tensor_count) ||
            header.scalar_count != static_cast<int32_t>(args.entry_scalar_count) ||
            header.payload_bytes !=
                simpler::tmr::kernel_invocation_bytes(args.entry_tensor_count, args.entry_scalar_count) -
                    sizeof(header) ||
            binding.runtime_descriptor_offset != offset ||
            binding.runtime_descriptor_bytes != args.runtime_descriptor_bytes || binding.run_epoch == 0 ||
            binding.run_epoch != args.run_result_epoch)
            return nullptr;
        return copy_descriptor(base + offset, args.runtime_descriptor_bytes);
    }

    Runtime *bind(Runtime *source, uint32_t bytes) {
        if (source == nullptr) return nullptr;
        if (bytes == 0) return source;
        if (bytes != RUNTIME_LAUNCH_DESCRIPTOR_BYTES) return nullptr;
        return copy_descriptor(source, bytes);
    }

private:
    Runtime *copy_descriptor(const void *source, uint32_t bytes) {
        constexpr uintptr_t alignment = alignof(DeviceRuntimeLaunchDesc);
        const uintptr_t address = (reinterpret_cast<uintptr_t>(storage_) + alignment - 1) & ~(alignment - 1);
        void *aligned = reinterpret_cast<void *>(address);
        std::memcpy(aligned, source, bytes);
        return static_cast<Runtime *>(aligned);
    }

    unsigned char storage_[RUNTIME_LAUNCH_DESCRIPTOR_BYTES + alignof(DeviceRuntimeLaunchDesc) - 1];
};
