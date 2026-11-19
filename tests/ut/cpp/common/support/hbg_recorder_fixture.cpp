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

#include "host_build_graph/host/graph_recorder_pool.h"
extern "C" bool test_recorder_prewarm() { return graph_recorder_prewarm(); }
extern "C" bool test_recorder_start(void (*body)(void *), void *argument) {
    static GraphTaskArgs args;
    return graph_recorder_pool().start(args, [=](const GraphTaskArgs &) {
        body(argument);
    });
}
extern "C" void test_recorder_wait() { graph_recorder_pool().wait(); }
extern "C" void test_recorder_shutdown() { graph_recorder_pool().shutdown(); }
