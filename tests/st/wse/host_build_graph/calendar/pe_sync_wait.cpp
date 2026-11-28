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
#include <array>
#include <chrono>
#include <thread>
#include "pe_executor.cpp"  // NOLINT(bugprone-suspicious-include): inspect the live gate under its owning mutex.

namespace {
// The busy SS may finish or fail only after the peer really entered the start gate.
// Sampling under the PE's run mutex lets the production execution path run unmodified.
WseRun *execution;
bool fail, witnessed;
void busy(int64_t *) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (std::chrono::steady_clock::now() < deadline) {
        {
            std::lock_guard lock(execution->mutex);
            if (execution->local(1).block_state == wse::BlockState::WAITING_FOR_SYNC) {
                witnessed = true;
                if (fail) throw std::runtime_error("busy SS failed after peer arrival");
                return;
            }
        }
        std::this_thread::yield();
    }
    throw std::runtime_error("peer never arrived at the start gate");
}
void member(int64_t *) {}

struct Wire {
    WseGraphDescriptor descriptor{};
    WseNodePlan nodes[2]{};
    WseBlockPlan placements[3]{};
    uint32_t indices[3]{0, 2, 3};
    wse::CalendarRow rows[3]{{0, 0, 0}, {1, 0, 1}, {1, 1, 1}};
    uint32_t pe_indices[9]{0, 2, 2, 2, 2, 3, 3, 3, 3};
    uint32_t pe_rows[3]{0, 1, 2};
};
}  // namespace

int main(int argc, char **) {
    fail = argc > 1;
    alignas(128) std::byte tasks[2 * SCHEDULER_GRAPH_TASK_STORAGE_STRIDE]{}, sram[2][16384]{};
    wse::SramRegion regions[2];
    for (uint32_t core = 0; core < 2; ++core) {
        const auto base = reinterpret_cast<uintptr_t>(sram[core]);
        regions[core] = {base, sizeof(sram[core]), sizeof(sram[core]) / 2, base + sizeof(sram[core]) / 2};
    }
    Wire wire;
    auto &d = wire.descriptor;
    d.graph = scheduler_ordinary_graph_view(reinterpret_cast<uintptr_t>(tasks), 2, 1);
    d.nodes_address = reinterpret_cast<uintptr_t>(wire.nodes);
    d.descriptor_bytes = sizeof(wire);
    d.expected_tasks = d.calendar_core_count = 2;
    d.calendar_row_count = 3;
    d.calendar_indices_offset = offsetof(Wire, indices);
    d.calendar_rows_offset = offsetof(Wire, rows);
    d.pe_calendar_indices_offset = offsetof(Wire, pe_indices);
    d.pe_calendar_rows_offset = offsetof(Wire, pe_rows);
    d.pe_calendar_row_count = 3;
    d.pe_calendar_version = wse::PE_CALENDAR_VERSION;
    for (uint32_t task = 0; task < 2; ++task) {
        auto &node = wire.nodes[task];
        node.entry = reinterpret_cast<uintptr_t>(task ? member : busy);
        node.blocks = node.block_plan_count = task + 1;
        node.block_plans_offset = offsetof(Wire, placements) + task * sizeof(WseBlockPlan);
        node.sync_start = task;
    }
    for (auto &placement : wire.placements)
        placement.thread_num = 1;
    wire.placements[2].core_id = 1;
    char error[256]{};
    execution = wse_create_run_v7(&d, 2, 1, regions, 2, nullptr, 0, 0, error, sizeof(error));
    if (!execution) {
        std::fprintf(stderr, "%s\n", error);
        return 1;
    }
    std::vector<std::thread> pes;
    std::array<int, 2 * WSE_PE_PER_CORE> results{};
    const auto start = [&](size_t i) {
        pes.emplace_back([i, &results] {
            results[i] = wse_execute_pe(execution, i);
        });
    };
    const size_t last_pe = results.size() - 1;
    for (size_t i = 0; i < last_pe; ++i)
        start(i);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (std::chrono::steady_clock::now() < deadline) {
        {
            std::lock_guard lock(execution->mutex);
            if (execution->ready_pes == last_pe) break;
        }
        std::this_thread::yield();
    }
    start(last_pe);
    for (auto &pe : pes)
        pe.join();
    unsigned ready = 0, release = 0, begin = 0;
    uint64_t peer_ready_seq = UINT64_MAX, busy_end_seq = 0, release_seq = 0;
    uint64_t last_retired_seq = 0;
    uint32_t last_retired_core = UINT32_MAX, run_done_core = UINT32_MAX;
    bool go_from_last_pe = false;
    for (const auto &stream : execution->events)
        for (const auto &e : stream) {
            if (e.kind == WseEventKind::GO) go_from_last_pe = e.core * WSE_PE_PER_CORE + e.pe == last_pe;
            if (e.kind == WseEventKind::CORE_RETIRED && e.sequence > last_retired_seq) {
                last_retired_seq = e.sequence;
                last_retired_core = e.core;
            }
            if (e.kind == WseEventKind::RUN_DONE) run_done_core = e.core;
            if (e.task == 0 && e.kind == WseEventKind::BLOCK_END) busy_end_seq = e.sequence;
            if (e.task != 1) continue;
            if (e.kind == WseEventKind::BLOCK_READY && e.block == 1) peer_ready_seq = e.sequence;
            if (e.kind == WseEventKind::SYNC_RELEASE) release_seq = e.sequence;
            ready += e.kind == WseEventKind::BLOCK_READY;
            release += e.kind == WseEventKind::SYNC_RELEASE;
            begin += e.kind == WseEventKind::BLOCK_BEGIN;
        }
    const int expected_rc = fail ? -1 : 0;
    const bool all_retired = std::all_of(results.begin(), results.end(), [expected_rc](int rc) {
        return rc == expected_rc;
    });
    const bool ordered = fail ? busy_end_seq == 0 : peer_ready_seq < busy_end_seq && busy_end_seq < release_seq;
    const bool lifecycle =
        go_from_last_pe && (fail || (last_retired_core != UINT32_MAX && last_retired_core == run_done_core));
    const bool valid = lifecycle && witnessed && all_retired && ordered && ready == (fail ? 1u : 2u) &&
                       release == (fail ? 0u : 1u) && begin == (fail ? 0u : 2u);
    wse_destroy_run(execution);
    if (!valid) std::fprintf(stderr, "witness=%d, READY=%u, RELEASE=%u, BEGIN=%u\n", witnessed, ready, release, begin);
    return valid ? 0 : 1;
}
