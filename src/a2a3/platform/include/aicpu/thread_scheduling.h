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

#include <cerrno>
#include <sched.h>

inline int use_normal_aicpu_scheduling() {
    const int policy = sched_getscheduler(0);
    if (policy < 0) return errno;
    if (policy == SCHED_OTHER) return 0;
    const sched_param param{};
    return sched_setscheduler(0, SCHED_OTHER, &param) == 0 ? 0 : errno;
}
