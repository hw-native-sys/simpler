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
struct TestOrchestrationConfig {
    int expected_arg_count;
};
extern "C" void owner_x(const void *) {}
extern "C" void owner_y(const void *) {}
extern "C" TestOrchestrationConfig owner_x_config(const void *) { return {3}; }
extern "C" TestOrchestrationConfig owner_y_config(const void *) { return {TEST_Y_CONFIG}; }
