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
 * ChipCallConfig — per-NEXT_LEVEL-task config (block_dim, aicpu_thread_num,
 * enable_profiling). Lives here (rather than chip_worker.h) so distributed
 * task slot state can store it directly without pulling in the full
 * ChipWorker header (which depends on dist_types.h).
 */

#pragma once

struct ChipCallConfig {
    int block_dim = 24;
    int aicpu_thread_num = 3;
    bool enable_profiling = false;
    bool enable_dump_tensor = false;
};
