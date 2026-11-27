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
#include <cstdio>
#include "launch.h"
#include "calendar_plan.h"

namespace {
struct Wire {
    WseGraphDescriptor descriptor{};
    WseNodePlan node{};
    WseBlockPlan placements[2]{};
    uint32_t indices[3]{0, 1, 1};
    wse::CalendarRow rows[2]{{0, 0, 0}, {0, 1, 0}};
};

void kernel(int64_t *) {}
}  // namespace

int main() {
    alignas(128) std::byte tasks[SCHEDULER_GRAPH_TASK_STORAGE_STRIDE]{};
    alignas(128) std::byte sram[2][16384]{};
    wse::SramRegion regions[2];
    for (size_t c = 0; c < 2; ++c) {
        const auto base = reinterpret_cast<uintptr_t>(sram[c]);
        regions[c] = {base, sizeof(sram[c]), sizeof(sram[c]) / 2, base + sizeof(sram[c]) / 2};
    }

    Wire wire;
    auto &descriptor = wire.descriptor;
    descriptor.graph = scheduler_ordinary_graph_view(reinterpret_cast<uintptr_t>(tasks), 1, 0);
    descriptor.nodes_address = reinterpret_cast<uintptr_t>(&wire.node);
    descriptor.descriptor_bytes = sizeof(wire);
    descriptor.expected_tasks = descriptor.calendar_row_count = 1;
    descriptor.calendar_core_count = 2;
    descriptor.calendar_indices_offset = offsetof(Wire, indices);
    descriptor.calendar_rows_offset = offsetof(Wire, rows);
    wire.node.entry = reinterpret_cast<uintptr_t>(kernel);
    wire.node.blocks = wire.node.block_plan_count = 1;
    wire.node.block_plans_offset = offsetof(Wire, placements);
    wire.placements[0].core_id = 0;
    wire.placements[0].thread_num = 1;

    const auto admit = [&](const char *name) {
        char error[256]{};
        auto *run = wse_create_run_v2(&descriptor, 2, 1, regions, 2, nullptr, 0, error, sizeof(error));
        std::printf("%s %s\n", name, run ? "accepted" : "rejected");
        if (!run) std::fprintf(stderr, "%s\n", error);
        wse_destroy_run(run);
    };
    admit("explicit_core_0");
    wire.indices[1] = 0;  // Move the calendar row to Core 1, keeping its explicit Core 0 placement.
    admit("wrong_calendar_core");
    wire.indices[1] = 1;
    wire.node.block_plan_count = 0;
    wire.node.block_plans_offset = 0;
    admit("missing_placement");

    descriptor.calendar_row_count = wire.indices[2] = 2;
    wire.node.blocks = wire.node.block_plan_count = 2;
    wire.node.block_plans_offset = offsetof(Wire, placements);
    wire.placements[1].core_id = 1;
    wire.placements[1].thread_num = 1;
    wire.node.sync_start = 1;
    admit("synchronous_pair");
    wire.rows[1].tick = 1;
    admit("split_sync_ticks");

    // Both synchronous tasks need the same two Cores in opposite placement order.
    // A task-by-task choice must yield one common tick per pair and the same
    // order on both Cores even with another ready SS on just one of the Cores.
    const std::vector<wse::CalendarTask> tasks_to_schedule{{{0}, {}}, {{1, 0}, {}, true}, {{0, 1}, {}, true}};
    wse::CoreCalendar calendar;
    if (!wse::build_core_calendar(tasks_to_schedule, 2, calendar) || calendar.indices != std::vector<uint32_t>{0, 3, 5})
        return 1;
    for (uint32_t i = 1; i <= 2; ++i) {
        const auto &left = calendar.rows[i];
        const auto &right = calendar.rows[i + 2];
        if (left.task != i || right.task != i || left.tick != right.tick || left.block == right.block) return 1;
    }
}
