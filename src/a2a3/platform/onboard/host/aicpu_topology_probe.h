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
#include <vector>

namespace pto::a2a3 {

struct AicpuLogicalCpu {
    // cpu_id is an AICPU OS-global affinity ID, not a die-local OCCUPY bit.
    int32_t cpu_id;
    // cluster_id is derived from the die-local CPU ID as local_cpu_id / 4.
    int32_t cluster_id;
};

// Resolve the AICPU OS-global base for a die-local OCCUPY bitmap. A2 and A3
// both use eight global AICPU IDs per physical die.
bool resolve_aicpu_cpu_id_base(int64_t phy_die_id, int32_t &out_cpu_id_base);

// Probe host-side AICPU OCCUPY and return the user-schedulable cpu_id pool.
// OCCUPY bit positions are die-local on A2 and A3; returned cpu_ids use the
// shared AICPU OS namespace consumed by the device affinity gate.
bool probe_aicpu_topology(uint32_t device_id, std::vector<AicpuLogicalCpu> &out_user_cpus);

// Pick the active cpu_ids that should survive the on-device filter gate.
// The result order is the deterministic exec_idx order consumed by the runtime:
// for tensormap_and_ringbuffer, the highest index is the orchestrator slot.
bool compute_allowed_cpus(
    const std::vector<AicpuLogicalCpu> &user_cpus, int32_t active_count, std::vector<int32_t> &out_allowed_cpus
);

}  // namespace pto::a2a3
