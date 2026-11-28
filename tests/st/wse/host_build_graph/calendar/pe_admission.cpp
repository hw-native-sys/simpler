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
#include <array>
#include "launch.h"
#include "calendar_plan.h"

namespace {
struct Wire {
    WseGraphDescriptor descriptor{};
    WseNodePlan node{};
    WseBlockPlan placements[2]{};
    uint32_t indices[3]{0, 1, 1};
    wse::CalendarRow rows[2]{{0, 0, 0}, {0, 1, 0}};
    std::array<uint32_t, 9> pe_indices{0, 1, 1, 1, 1, 1, 1, 1, 1};
    uint32_t pe_rows[2]{0, 1};
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
    descriptor.pe_calendar_indices_offset = offsetof(Wire, pe_indices);
    descriptor.pe_calendar_rows_offset = offsetof(Wire, pe_rows);
    descriptor.pe_calendar_row_count = 1;
    descriptor.pe_calendar_version = wse::PE_CALENDAR_VERSION;
    wire.node.entry = reinterpret_cast<uintptr_t>(kernel);
    wire.node.blocks = wire.node.block_plan_count = 1;
    wire.node.block_plans_offset = offsetof(Wire, placements);
    wire.placements[0].core_id = 0;
    wire.placements[0].thread_num = 1;

    const auto admit = [&](const char *name) {
        char error[256]{};
        auto *run = wse_create_run_v7(&descriptor, 2, 1, regions, 2, nullptr, 0, 0, error, sizeof(error));
        std::printf("%s %s\n", name, run ? "accepted" : "rejected");
        if (!run) std::fprintf(stderr, "%s\n", error);
        const auto idle_usage = run ? wse_runtime_used(run, 1) : 0;
        wse_destroy_run(run);
        return idle_usage;
    };
    const uint64_t idle_usage = admit("explicit_core_0");
    descriptor.nodes_address =
        reinterpret_cast<uintptr_t>(&wire) + offsetof(WseGraphDescriptor, pe_calendar_indices_offset);
    admit("old_descriptor");
    descriptor.nodes_address = reinterpret_cast<uintptr_t>(&wire.node);
    descriptor.pe_calendar_version = 0;  // The v5 producer zero-initializes this header word.
    admit("old_pe_calendar_version");
    descriptor.pe_calendar_version = wse::PE_CALENDAR_VERSION;
    wire.pe_indices[1] = 0;  // Put the Block on PE 1 while its placement selects PE 0.
    admit("wrong_calendar_pe");
    wire.pe_indices[1] = 1;
    wire.pe_rows[0] = UINT32_MAX;
    admit("invalid_block_row_index");
    wire.pe_rows[0] = 0;
    descriptor.pe_calendar_rows_offset = sizeof(wire);
    admit("truncated_pe_rows");
    descriptor.pe_calendar_rows_offset = offsetof(Wire, pe_rows);
    wire.indices[1] = 0;  // Move the calendar row to Core 1, keeping its explicit Core 0 placement.
    admit("wrong_calendar_core");

    // Core 0 can reserve just enough for its own control record while Core 1 runs the SS.
    const auto original = regions[0];
    regions[0].reserve = idle_usage;
    regions[0].runtime_base = original.base + original.capacity - idle_usage;
    wire.placements[0].core_id = 1;
    wire.pe_indices = {0, 0, 0, 0, 0, 1, 1, 1, 1};
    admit("idle_core_0_at_limit");
    wire.indices[1] = 1;
    wire.placements[0].core_id = 0;
    wire.pe_indices = {0, 1, 1, 1, 1, 1, 1, 1, 1};
    admit("active_core_0_over_limit");
    regions[0] = original;
    wire.node.block_plan_count = 0;
    wire.node.block_plans_offset = 0;
    admit("missing_placement");

    descriptor.calendar_row_count = wire.indices[2] = 2;
    descriptor.pe_calendar_row_count = 2;
    wire.pe_indices = {0, 1, 1, 1, 1, 2, 2, 2, 2};
    wire.node.blocks = wire.node.block_plan_count = 2;
    wire.node.block_plans_offset = offsetof(Wire, placements);
    wire.placements[1].core_id = 1;
    wire.placements[1].thread_num = 1;
    wire.node.sync_start = 1;
    admit("synchronous_pair");
    std::swap(wire.pe_rows[0], wire.pe_rows[1]);
    admit("wrong_block_row");
    std::swap(wire.pe_rows[0], wire.pe_rows[1]);
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
    const wse::CoreCalendar placed{{0, 2, 2}, {{0, 0, 0}, {1, 0, 1}}};
    std::vector<std::vector<WseBlockPlan>> blocks(2, std::vector<WseBlockPlan>(1));
    blocks[0][0].thread_num = 2;
    blocks[0][0].thread_pes[0] = 3;
    blocks[0][0].thread_pes[1] = 2;
    blocks[1][0].thread_num = 1;
    blocks[1][0].thread_pes[0] = 3;
    wse::PeCalendar per_pe;
    if (!wse::build_pe_calendar(placed, blocks, per_pe) ||
        per_pe.indices != std::vector<uint32_t>{0, 0, 0, 1, 3, 3, 3, 3, 3} ||
        per_pe.rows != std::vector<uint32_t>{0, 0, 1})
        return 1;
    blocks[0][0].thread_pes[1] = 3;
    if (wse::build_pe_calendar(placed, blocks, per_pe)) return 1;
}
