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

#include <cstring>
#include "common/launch_entry_args.h"
#include "common/kernel_args.h"
#include "runtime.h"
#include "runtime/rts/rts_kernel.h"

namespace simpler::capture_test {

// Read the SDK's actual submitted extent before following either packet-relative section.
struct InlineCpuPacket {
    uint64_t workspace_address = 0;
    HbgExecutionImage image{};
    const unsigned char *image_section = nullptr;
    bool valid = false;
};

inline InlineCpuPacket observe_inline_cpu_packet(const rtCpuKernelArgs_t *args) {
    InlineCpuPacket result{};
    if (!args || !args->baseArgs.args || args->baseArgs.argsSize < sizeof(KernelArgs) ||
        args->baseArgs.hostInputInfoNum != 0 || args->baseArgs.hostInputInfoPtr != nullptr)
        return result;
    const auto *packet = static_cast<const unsigned char *>(args->baseArgs.args);
    KernelArgs header{};
    std::memcpy(&header, packet, sizeof(header));
    const auto bytes = args->baseArgs.argsSize;
    const auto descriptor = header.runtime_section;
    if (descriptor.packet_bytes != bytes || descriptor.descriptor_offset < sizeof(header) ||
        descriptor.descriptor_offset > bytes || header.runtime_descriptor_bytes != RUNTIME_LAUNCH_DESCRIPTOR_BYTES ||
        header.runtime_descriptor_bytes > bytes - descriptor.descriptor_offset)
        return result;
    constexpr size_t workspace_field = offsetof(DeviceRuntimeLaunchDesc, gm_sm_ptr_);
    static_assert(workspace_field + sizeof(void *) <= RUNTIME_LAUNCH_DESCRIPTOR_BYTES);
    void *workspace;
    std::memcpy(&workspace, packet + descriptor.descriptor_offset + workspace_field, sizeof(workspace));
    result.workspace_address = reinterpret_cast<uint64_t>(workspace);
    const auto section = header.image_section;
    const auto descriptor_end = descriptor.descriptor_offset + header.runtime_descriptor_bytes;
    if (section.bytes == 0) {
        result.valid = section.offset == 0 && descriptor_end == bytes &&
                       header.entry_args_source == static_cast<uint32_t>(EntryArgsSource::KernelInvocation);
        return result;
    }
    if (header.entry_args_source != static_cast<uint32_t>(EntryArgsSource::Descriptor) ||
        section.offset != hbg_execution_image_offset(descriptor_end) || section.offset > bytes ||
        section.bytes > bytes - section.offset || section.bytes < sizeof(HbgExecutionImage))
        return result;
    result.image_section = packet + section.offset;
    auto &image = result.image;
    std::memcpy(&image, result.image_section, sizeof(image));
    const uint64_t payload = section.bytes - sizeof(image);
    result.valid = image.working_address != 0 && image.bytes > 0 && image.bytes <= payload &&
                   image.mutable_prefix_bytes <= image.bytes &&
                   (payload - image.bytes) % sizeof(HbgImageReference) == 0 &&
                   image.reference_count == (payload - image.bytes) / sizeof(HbgImageReference);
    return result;
}
}  // namespace simpler::capture_test
