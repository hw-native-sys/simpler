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

#include <cstdint>
#include <new>

#include "host_build_graph/host_graph_build.h"
#include "host_build_graph/host_tensor_access.h"
#include "host_build_graph/runtime_types.h"
#include "orchestration_requirements.h"
#include "task_args.h"

namespace hbg {

// H6 wire convention:
//   [device tensors][host-only duplicates][scalars]
// host_copy_tensor_count names the trailing tensor suffix. Each host copy is
// paired, in order, with the equally-sized suffix of the DEVICE prefix. This
// makes the pairing recoverable from the existing K9 count without placing
// pointers or an index table in the invocation header.
enum class KernelExternalTensorStatus : uint8_t {
    Ok = 0,
    InvalidCounts,
    InvalidTensor,
    NonDeviceTensor,
    UnsupportedStrideFamily,
    InvalidHostCopy,
    HostCopyMismatch,
    RequirementsRejected,
    AllocationFailure,
};

inline bool valid_external_tensor_span(const ChipTensor &tensor, uint64_t *extent_out = nullptr) noexcept {
    if (tensor.buffer.addr == 0 || tensor.buffer.addr >= HEAP_VIRTUAL_BASE || tensor.buffer.size == 0 ||
        tensor.buffer.size > UINT64_MAX - tensor.buffer.addr || tensor.ndims == 0 || tensor.ndims > MAX_TENSOR_DIMS ||
        static_cast<uint8_t>(tensor.dtype) >= static_cast<uint8_t>(DataType::DATA_TYPE_NUM))
        return false;
    const uint64_t element_bytes = get_element_size(tensor.dtype);
    if (element_bytes == 0) return false;
    uint64_t extent = 1;
    for (uint32_t i = 0; i < tensor.ndims; ++i) {
        const uint64_t shape = tensor.shapes[i];
        const uint64_t stride = tensor.strides[i];
        if (shape == 0 || stride == 0 || (shape - 1) > (UINT64_MAX - extent) / stride) return false;
        extent += (shape - 1) * stride;
    }
    if (tensor.start_offset > tensor.buffer.size / element_bytes ||
        extent > tensor.buffer.size / element_bytes - tensor.start_offset)
        return false;
    if (extent_out != nullptr) *extent_out = extent;
    return true;
}

// V1 accepts dense row-major tensors and row-major views with padding between
// outer rows. The innermost dimension is unit-stride; every outer stride must
// cover the full next dimension. Transposes, broadcasts and stepped innermost
// slices need a separate specialization and are rejected before Host build.
inline bool supported_external_stride_family(const ChipTensor &tensor) noexcept {
    if (tensor.ndims == 0 || tensor.ndims > MAX_TENSOR_DIMS || tensor.strides[tensor.ndims - 1] != 1) return false;
    for (uint32_t i = tensor.ndims - 1; i > 0; --i) {
        const uint64_t inner = tensor.strides[i];
        const uint64_t shape = tensor.shapes[i];
        if (shape != 0 && inner > UINT64_MAX / shape) return false;
        if (tensor.strides[i - 1] < inner * shape) return false;
    }
    return true;
}

inline bool matching_host_copy_metadata(const ChipTensor &device, const ChipTensor &host) noexcept {
    if (device.buffer.size != host.buffer.size || device.start_offset != host.start_offset ||
        device.ndims != host.ndims || device.dtype != host.dtype)
        return false;
    for (uint32_t i = 0; i < device.ndims; ++i)
        if (device.shapes[i] != host.shapes[i] || device.strides[i] != host.strides[i]) return false;
    return true;
}

inline KernelExternalTensorStatus
validate_kernel_external_tensors(const ChipStorageTaskArgs &args, int32_t host_copy_tensor_count) noexcept {
    const int32_t tensor_count = args.tensor_count();
    const int32_t scalar_count = args.scalar_count();
    if (tensor_count < 0 || tensor_count > CHIP_MAX_TENSOR_ARGS || scalar_count < 0 ||
        scalar_count > CHIP_MAX_SCALAR_ARGS || tensor_count + scalar_count > CHIP_MAX_TENSOR_ARGS ||
        host_copy_tensor_count < 0 || host_copy_tensor_count > tensor_count / 2)
        return KernelExternalTensorStatus::InvalidCounts;
    const int32_t device_count = tensor_count - host_copy_tensor_count;
    for (int32_t i = 0; i < device_count; ++i) {
        const ChipTensor &tensor = args.tensor(i);
        if (tensor.address_space != AddressSpace::DEVICE) return KernelExternalTensorStatus::NonDeviceTensor;
        if (!valid_external_tensor_span(tensor)) return KernelExternalTensorStatus::InvalidTensor;
        if (!supported_external_stride_family(tensor)) return KernelExternalTensorStatus::UnsupportedStrideFamily;
    }
    const int32_t paired_device_begin = device_count - host_copy_tensor_count;
    for (int32_t i = 0; i < host_copy_tensor_count; ++i) {
        const ChipTensor &host = args.tensor(device_count + i);
        if (host.address_space != AddressSpace::HOST || !valid_external_tensor_span(host))
            return KernelExternalTensorStatus::InvalidHostCopy;
        if (!supported_external_stride_family(host)) return KernelExternalTensorStatus::UnsupportedStrideFamily;
        if (!matching_host_copy_metadata(args.tensor(paired_device_begin + i), host))
            return KernelExternalTensorStatus::HostCopyMismatch;
    }
    return KernelExternalTensorStatus::Ok;
}

// Validate everything before adding a readable region. On success only the
// trailing host-copy suffix is readable; DEVICE addresses have no region and a
// get_tensor_data call against one fails closed without mapping, D2H or sync.
inline KernelExternalTensorStatus prepare_kernel_external_tensors(
    const ChipStorageTaskArgs &args, int32_t host_copy_tensor_count, const HostOrchEntryPoints &entry_points,
    HostTensorAccessor &accessor
) noexcept {
    const auto tensor_status = validate_kernel_external_tensors(args, host_copy_tensor_count);
    if (tensor_status != KernelExternalTensorStatus::Ok) return tensor_status;
    if (simpler::orchestration::validate_hbg_kernel_requirements(
            entry_points.requirements_v1_available, entry_points.requirements_v1, host_copy_tensor_count
        ) != simpler::orchestration::HbgKernelRequirementsStatus::Ok)
        return KernelExternalTensorStatus::RequirementsRejected;
    if ((entry_points.requirements_v1 & simpler::orchestration::REQUIREMENT_TENSOR_DATA_READ) == 0)
        return KernelExternalTensorStatus::Ok;
    const int32_t first_host_copy = args.tensor_count() - host_copy_tensor_count;
    try {
        for (int32_t i = first_host_copy; i < args.tensor_count(); ++i) {
            const ChipTensor &host = args.tensor(i);
            if (!accessor.add_host_copy(
                    host.buffer.addr, host.buffer.size, reinterpret_cast<const void *>(host.buffer.addr)
                )) {
                accessor.close();
                return KernelExternalTensorStatus::InvalidHostCopy;
            }
        }
    } catch (const std::bad_alloc &) {
        accessor.close();
        return KernelExternalTensorStatus::AllocationFailure;
    }
    return KernelExternalTensorStatus::Ok;
}

}  // namespace hbg
