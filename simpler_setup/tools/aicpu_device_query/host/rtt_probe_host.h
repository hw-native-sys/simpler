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

#include <acl/acl.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace pto::a5 {
struct AicpuLaunchPlan;
struct AicpuTopology;
}  // namespace pto::a5

namespace aicpu_device_query {

std::string MakeRttDescriptorEntry(uint64_t fingerprint, const std::string &so_basename);
std::string MakeAffinityEnumDescriptorEntry(uint64_t fingerprint, const std::string &so_basename);
std::string MakeAffinityPreflightDescriptorEntry(uint64_t fingerprint, const std::string &so_basename);

// Full affinity preflight: serial enum of the user pool, atomic-flag orch
// election, COND die scoring. Emits schema_version=3 JSON on stdout.
bool RunAffinityPreflight(
    int device_id, void *binary_handle, uint64_t fingerprint, void *device_args, size_t device_args_bytes,
    aclrtStream stream, const pto::a5::AicpuTopology &topology
);

bool LastAffinityProbeTimedOut();

// Back-compat name used by query_device_hal.cpp.
bool RunRttProbe(
    int device_id, void *binary_handle, uint64_t fingerprint, void *device_args, size_t device_args_bytes,
    aclrtStream stream, const pto::a5::AicpuTopology &topology, const pto::a5::AicpuLaunchPlan &launch_plan
);

}  // namespace aicpu_device_query
