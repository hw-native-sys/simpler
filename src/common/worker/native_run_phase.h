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

/**
 * Internal phase of the caller-owned opaque native-run storage.
 *
 * Its own file because a decision can be about the phase without being about
 * the context: `native_run_context.h` carries the runtime, the host API and
 * the C ABI with it, and a caller that only names a phase would resolve
 * `runtime.h` against whichever runtime it happens to be built for.
 */
enum class NativeRunPhase : uint8_t {
    Prepared,
    Running,
    Complete,
};
