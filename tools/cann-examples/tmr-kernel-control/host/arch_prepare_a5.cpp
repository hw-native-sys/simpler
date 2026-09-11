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
#include "arch_prepare.h"
#include "host/host_regs.h"

int prepare_arch_fields(KernelArgs &args, MemoryAllocator &allocator, int device) {
    return init_aicore_register_addresses(&args.regs, device, allocator);
}

int prepare_gate_topology(control_probe::GateInit &, int) {
    // A5 needs its device occupancy query before choosing a native launch group.
    // Preserve the original single-CPU probe; do not invent A5 topology evidence.
    return 0;
}
