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
#include "platform_comm/ccu/ccu_abi.h"

namespace {
using namespace simpler::ccu;
StopResult drain(void *) { return StopResult::CONFIRMED; }
PollResult poll(void *) { return PollResult::COMPLETE; }
StopResult close_program(void *handle) {
    delete static_cast<uint64_t *>(handle);
    return StopResult::CONFIRMED;
}

int32_t build(const BuildContext *context, Program *program, uint64_t address) {
    if (!context || !program || context->abi_version != ABI_VERSION || context->struct_size != sizeof(BuildContext)) {
        return -1;
    }
    *program = {ABI_VERSION, sizeof(Program), new uint64_t(address), address, drain, close_program, poll};
    return 0;
}
}  // namespace

extern "C" uint32_t simpler_ccu_module_abi_version() { return simpler::ccu::ABI_VERSION; }

extern "C" int32_t build_copy(const simpler::ccu::BuildContext *context, simpler::ccu::Program *program) {
    return build(context, program, 42);
}

extern "C" int32_t build_reduce(const simpler::ccu::BuildContext *context, simpler::ccu::Program *program) {
    return build(context, program, 43);
}
