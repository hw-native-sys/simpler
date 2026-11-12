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
 * The arena bank selected for each pooled device region of one run.
 *
 * Each region is selected from its own declaration: a runtime may share its
 * heap while still delivering a per-run image. A missing kind keeps zero,
 * which is the only bank a context without that region can name.
 */
typedef struct RegionBanks {
    uint32_t heap;
    uint32_t sm;
    uint32_t image;
} RegionBanks;
