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
#include <string>
#include <vector>

namespace pto::a5 {

// Reads the line-oriented companion file written by rtt_die_preflight:
//   build/config/aicpu_affinity_plan.<device_id>.cpus
//   (or the per-device path derived from SIMPLER_AICPU_AFFINITY_PLAN)
// Format:
//   schema_version=3
//   device_id=<logical ACL device id>
//   soc=<name>
//   source=<token>
//   occupy_mask=<hex or decimal u64>
//   active_count=5
//   cpus=<csv>
std::string affinity_cpus_side_path(int32_t device_id);

// Resolve the total active AICPU count from user configuration and the full
// OCCUPY launch population. requested_active_count=0 is auto. The measured RTT
// plan applies only to the exact 4-scheduler + 1-orchestrator shape.
bool resolve_aicpu_active_count(
    int32_t requested_active_count, int32_t launch_count, int32_t &active_count, bool &use_rtt_plan
);

// Select exactly active_count CPUs in ascending OCCUPY bit order. The last
// position is the orchestrator; preceding positions are scheduler slots.
bool build_occupy_contiguous_allowed(uint64_t occupy, int32_t active_count, std::vector<int32_t> &allowed_cpus);

bool load_affinity_cpus_side_file(
    const char *soc_name, int32_t device_id, uint64_t current_occupy, std::vector<int32_t> &allowed_cpus,
    std::string &plan_source
);

}  // namespace pto::a5
