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
/**
 * The one definition of a context's execution mode, shared by every layer
 * that names it: the host-side identity latch on the platform runners, the
 * invocation wire header, and the C ABI documentation. A context's mode is
 * decided by which init entry runs first and never changes afterwards.
 */

#pragma once

typedef enum SimplerExecutionMode {
    /* Historical exclusive-device semantics, claimed by simpler_init. */
    SIMPLER_MODE_PROGRAM = 0,
    /* Borrowed-device semantics, claimed by simpler_kernel_mode_init. */
    SIMPLER_MODE_KERNEL = 1,
} SimplerExecutionMode;
