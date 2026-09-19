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
#include <pto/common/cpu_stub.hpp>

// Each kernel DSO binds its ISA operations to the runtime-owned simulation context.
extern "C" __attribute__((visibility("default"))) void
pto_sim_register_hooks(void *get_subblock_id, void *get_pipe_shared_state) {
    pto::cpu_sim::register_hooks(get_subblock_id, get_pipe_shared_state);
}
