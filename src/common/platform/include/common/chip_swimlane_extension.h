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

#include <cstddef>
#include <cstdint>
#include <string_view>

/** Fixed extension slots emitted by runtime-owned chip-swimlane producers. */
enum class ChipSwimlaneExtensionSection : uint32_t {
    AicoreTasks = 0,
    SchedulerTasks = 1,
    SchedulerRecords = 2,
    AicpuLifecycleRecords = 3,
    Count = 4,
};

inline constexpr bool chip_swimlane_extension_section_is_valid(ChipSwimlaneExtensionSection section) {
    return static_cast<uint32_t>(section) < static_cast<uint32_t>(ChipSwimlaneExtensionSection::Count);
}

inline constexpr const char *chip_swimlane_extension_section_name(ChipSwimlaneExtensionSection section) {
    switch (section) {
    case ChipSwimlaneExtensionSection::AicoreTasks:
        return "aicore_tasks";
    case ChipSwimlaneExtensionSection::SchedulerTasks:
        return "scheduler_tasks";
    case ChipSwimlaneExtensionSection::SchedulerRecords:
        return "scheduler_records";
    case ChipSwimlaneExtensionSection::AicpuLifecycleRecords:
        return "aicpu_lifecycle_records";
    case ChipSwimlaneExtensionSection::Count:
        break;
    }
    return nullptr;
}

inline constexpr bool chip_swimlane_extension_section_is_object(ChipSwimlaneExtensionSection section) {
    return section == ChipSwimlaneExtensionSection::SchedulerTasks ||
           section == ChipSwimlaneExtensionSection::SchedulerRecords;
}

/** Validate the bounded payload envelope; payload contents come only from internal serializers. */
inline bool
chip_swimlane_extension_has_expected_root(ChipSwimlaneExtensionSection section, std::string_view json_value) noexcept {
    if (!chip_swimlane_extension_section_is_valid(section)) return false;
    const size_t first = json_value.find_first_not_of(" \t\r\n");
    const size_t last = json_value.find_last_not_of(" \t\r\n");
    if (first == std::string_view::npos) return false;
    const bool expects_object = chip_swimlane_extension_section_is_object(section);
    return json_value[first] == (expects_object ? '{' : '[') && json_value[last] == (expects_object ? '}' : ']');
}
