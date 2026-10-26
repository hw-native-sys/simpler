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

class Runtime;
namespace hbg {
// The boot owner calls restore after handshakes and retires after the last
// scheduler participant has shut down. The restore result survives the boot
// participant, so retirement rejects even when its own participant succeeded.
// A failed restore poisons the registered slot before completion,
// including a validation failure that never claimed an execution attempt.
// Ordinary program runs and Host launch preflight refusals are unaffected.
int restore_kernel_graph(Runtime *runtime) noexcept;
int retire_kernel_graph(Runtime *runtime, int execution_status) noexcept;
}  // namespace hbg
