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
#include "common/launch_entry_args.h"
#include "runtime.h"
#include "task_interface/kernel_invocation_header.h"

// RTS argument storage guarantees no Runtime cache-line alignment. Each AICPU
// invocation keeps its descriptor view until its own device entry returns.
class RuntimeLaunchView {
public:
    RuntimeLaunchView() = default;
    RuntimeLaunchView(const RuntimeLaunchView &) = delete;
    RuntimeLaunchView &operator=(const RuntimeLaunchView &) = delete;
    Runtime *bind(const KernelArgs &args) {
        execution_image_ = nullptr;
        execution_image_bytes_ = 0;
        const auto section = args.runtime_section;
        if (args.runtime_descriptor_bytes == 0) {
            if (section.descriptor_offset != 0 || section.packet_bytes != 0 ||
                (args.entry_args_source != static_cast<uint32_t>(EntryArgsSource::Descriptor) &&
                 args.entry_args_source != static_cast<uint32_t>(EntryArgsSource::LaunchEnvelope)))
                return nullptr;
            return args.runtime_args;
        }
        if (args.runtime_descriptor_bytes != RUNTIME_LAUNCH_DESCRIPTOR_BYTES ||
            section.descriptor_offset > section.packet_bytes ||
            args.runtime_descriptor_bytes > section.packet_bytes - section.descriptor_offset ||
            reinterpret_cast<uintptr_t>(&args) > UINTPTR_MAX - section.packet_bytes)
            return nullptr;
        if (args.entry_args_source == static_cast<uint32_t>(EntryArgsSource::Descriptor)) return bind_hbg(args);
        if (args.entry_args_source != static_cast<uint32_t>(EntryArgsSource::KernelInvocation) ||
            args.image_section.offset != 0 || args.image_section.bytes != 0 ||
            args.entry_args_offset != LAUNCH_ENVELOPE_HEADER_BYTES || args.entry_tensor_count > CHIP_MAX_TENSOR_ARGS ||
            args.entry_scalar_count > CHIP_MAX_SCALAR_ARGS ||
            args.entry_tensor_count + args.entry_scalar_count > CHIP_MAX_TENSOR_ARGS)
            return nullptr;

        const size_t offset = simpler::tmr::kernel_runtime_descriptor_offset(
            args.entry_args_offset, args.entry_tensor_count, args.entry_scalar_count, alignof(DeviceRuntimeLaunchDesc)
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

    const void *execution_image() const { return execution_image_; }
    size_t execution_image_bytes() const { return execution_image_bytes_; }

private:
    Runtime *bind_hbg(const KernelArgs &args) {
        if (args.entry_args_offset != 0 || args.entry_tensor_count != 0 || args.entry_scalar_count != 0) return nullptr;
        uint64_t graph_end = LAUNCH_ENVELOPE_HEADER_BYTES;
        if (args.graph_section_source == static_cast<uint32_t>(GraphSectionSource::LaunchEnvelope)) {
            if (args.graph_section_offset != LAUNCH_ENVELOPE_HEADER_BYTES || args.graph_section_bytes == 0)
                return nullptr;
            graph_end += args.graph_section_bytes;
        } else if (args.graph_section_source != static_cast<uint32_t>(GraphSectionSource::None) ||
                   args.graph_section_offset != 0 || args.graph_section_bytes != 0) {
            return nullptr;
        }
        const auto section = args.runtime_section;
        constexpr size_t alignment = alignof(DeviceRuntimeLaunchDesc);
        if (section.descriptor_offset != ((graph_end + alignment - 1) & ~(uint64_t(alignment) - 1))) return nullptr;
        const uint64_t descriptor_end = uint64_t(section.descriptor_offset) + args.runtime_descriptor_bytes;
        const auto image = args.image_section;
        const auto *base = reinterpret_cast<const unsigned char *>(&args);
        if (image.bytes == 0) {
            if (image.offset != 0 || descriptor_end != section.packet_bytes) return nullptr;
        } else {
            const uint64_t image_offset = hbg_execution_image_offset(descriptor_end);
            if (image.offset != image_offset || image.bytes < sizeof(HbgExecutionImage) ||
                image.offset > section.packet_bytes || image.bytes > section.packet_bytes - image.offset ||
                (uint64_t(image.offset) + image.bytes != section.packet_bytes &&
                 section.packet_bytes != HBG_DIRECT_IMAGE_MIN_PACKET_BYTES))
                return nullptr;
            execution_image_ = base + image.offset;
            execution_image_bytes_ = image.bytes;
        }
        return copy_descriptor(base + section.descriptor_offset, args.runtime_descriptor_bytes);
    }

    Runtime *copy_descriptor(const void *source, uint32_t bytes) {
        constexpr uintptr_t alignment = alignof(DeviceRuntimeLaunchDesc);
        const uintptr_t address = (reinterpret_cast<uintptr_t>(storage_) + alignment - 1) & ~(alignment - 1);
        void *aligned = reinterpret_cast<void *>(address);
        std::memcpy(aligned, source, bytes);
        return static_cast<Runtime *>(aligned);
    }

    const void *execution_image_{nullptr};
    size_t execution_image_bytes_{0};
    unsigned char storage_[RUNTIME_LAUNCH_DESCRIPTOR_BYTES + alignof(DeviceRuntimeLaunchDesc) - 1];
};
