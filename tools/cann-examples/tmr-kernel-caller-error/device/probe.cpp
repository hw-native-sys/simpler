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

#include "../protocol.h"
extern "C" __attribute__((visibility("default"))) int simpler_aicpu_init(void *) { return 0; }
extern "C" __attribute__((visibility("default"))) int simpler_aicpu_exec(void *packet) {
    if (packet == nullptr) return 2;
    return static_cast<const CallerProbeArgs *>(packet)->status == 0 ? 0 : 2;
}
