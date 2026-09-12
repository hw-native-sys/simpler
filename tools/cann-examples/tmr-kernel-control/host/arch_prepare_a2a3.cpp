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
#include <runtime/rt.h>
#include "arch_prepare.h"
#include "aicpu_topology_probe.h"
#include "common/platform_config.h"
#include "host/host_regs.h"

int prepare_arch_fields(KernelArgs &args, MemoryAllocator &allocator, int device) {
    int rc = init_aicore_register_addresses(&args.regs, device, allocator, AicoreRegKind::Ctrl);
    if (rc != 0) return rc;
    uint32_t length = 0;
    return rtGetC2cCtrlAddr(&args.ffts_base_addr, &length);
}

int prepare_gate_topology(control_probe::GateInit &args, int device) {
    std::vector<pto::a2a3::AicpuLogicalCpu> cpus;
    std::vector<int32_t> allowed;
    if (!pto::a2a3::probe_aicpu_topology(static_cast<uint32_t>(device), cpus) ||
        !pto::a2a3::compute_allowed_cpus(cpus, control_probe::kExecutionThreads, allowed))
        return -1;
    args.launched_threads = PLATFORM_MAX_AICPU_THREADS_JUST_FOR_LAUNCH;
    args.execution_threads = control_probe::kExecutionThreads;
    for (int32_t i = 0; i < control_probe::kExecutionThreads; ++i)
        args.allowed_cpus[i] = allowed[i];
    return 0;
}
