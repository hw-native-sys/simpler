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
 * Per-callable_id protocol constants
 *
 * Single source of truth for the host↔AICPU per-callable_id dispatch protocol.
 * Kept separate from callable.h so the AICPU side can include it without
 * pulling in <vector>/<stdexcept>.
 *
 * Both sides must agree on these bounds:
 *   - Host: DeviceRunner::register_prepared_callable rejects out-of-range ids.
 *   - AICPU: AicpuExecutor::run guards `orch_so_table_[callable_id]` access.
 */

#pragma once

#include <cstdint>

// Hard cap on the number of distinct callable_ids that can be registered
// via Worker.register / DeviceRunner::register_prepared_callable. The AICPU
// executor reserves a fixed-size `orch_so_table_[MAX_REGISTERED_CALLABLE_IDS]`
// keyed by callable_id, so this bound is part of the host↔AICPU protocol.
constexpr int32_t MAX_REGISTERED_CALLABLE_IDS = 64;
