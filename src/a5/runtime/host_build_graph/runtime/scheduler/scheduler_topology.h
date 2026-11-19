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

#include <stdint.h>

struct SchedulerClusterCoordinate {
    int32_t cluster_index;
    int32_t cluster_lane;
};

// The mixed-kernel launch index is the stable hardware-topology coordinate:
// AIC indices occupy [0, cluster_count), followed by the two AIV subblocks of
// every Cluster. Real A5 physical_core_id values are sparse and are only valid
// for register addressing, so they must not be treated as a dense topology.
// Scheduler selection remains dynamic and is performed after discovery.
inline __aicore__ bool scheduler_cluster_coordinate_from_worker(
    int32_t worker_id, bool is_aic, int32_t cluster_count, int32_t aiv_per_cluster,
    SchedulerClusterCoordinate *coordinate
) {
    if (coordinate == nullptr || worker_id < 0 || cluster_count <= 0 || aiv_per_cluster <= 0) return false;
    if (is_aic) {
        if (worker_id >= cluster_count) return false;
        *coordinate = {worker_id, 0};
        return true;
    }
    const int32_t aiv_launch_rank = worker_id - cluster_count;
    if (aiv_launch_rank < 0 || aiv_launch_rank >= cluster_count * aiv_per_cluster) return false;
    *coordinate = {aiv_launch_rank / aiv_per_cluster, 1 + aiv_launch_rank % aiv_per_cluster};
    return true;
}
