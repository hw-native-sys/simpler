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

#include <gtest/gtest.h>

#include "common/chip_swimlane_extension.h"
#include "common/chip_swimlane_profiling.h"
#include "host/chip_swimlane_collector.h"

TEST(ChipSwimlaneCollectorTest, BeginRunReleasesRuntimeExtensionSlots) {
    ChipSwimlaneCollector collector;

    collector.begin_run("first", ChipSwimlaneLevel::TASK_TIMING);
    ASSERT_TRUE(collector.set_json_extension(ChipSwimlaneExtensionSection::AicoreTasks, "[]"));
    EXPECT_FALSE(collector.set_json_extension(ChipSwimlaneExtensionSection::AicoreTasks, "[]"));

    collector.begin_run("second", ChipSwimlaneLevel::TASK_TIMING);
    EXPECT_TRUE(collector.set_json_extension(ChipSwimlaneExtensionSection::AicoreTasks, "[]"));
}
