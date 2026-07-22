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

// PTO2SchedulerState's members are per-task/per-completion hot-path routines
// (classify_fanin_state, register_wake, drain_wiring_queue, on_mixed_task_complete,
// …) and are defined inline in pto_scheduler.h. SchedulerContext's out-of-line
// methods live in scheduler_dispatch.cpp / scheduler_cold_path.cpp /
// scheduler_completion.cpp. This translation unit carries no definitions.
