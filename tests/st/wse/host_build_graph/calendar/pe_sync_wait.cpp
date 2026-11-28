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
#include "pe_swimlane.h"
#include "pe_executor.cpp"  // NOLINT(bugprone-suspicious-include): inspect the live gate under its owning mutex.

namespace {
// The busy SS may finish or fail only after the peer really entered the start gate.
// Sampling under the PE's run mutex lets the production execution path run unmodified.
WseRun *execution;
constexpr uint32_t core_count = 7;
constexpr uint32_t busy_core = 2, peer_core = 6, busy_pe = 3;
bool fail, sync_start, witnessed;
void busy(int64_t *) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (std::chrono::steady_clock::now() < deadline) {
        {
            std::lock_guard lock(execution->mutex);
            const auto reached = [](uint32_t core, uint32_t pe) {
                const auto &calendar = execution->local(core).calendars[pe];
                return sync_start ? calendar.arrived : calendar.head == calendar.size;
            };
            if (reached(busy_core, 2) && reached(peer_core, 2) && reached(peer_core, 3)) {
                witnessed = true;
                if (fail) throw std::runtime_error("busy SS failed after the three peer arrivals");
                return;
            }
        }
        std::this_thread::yield();
    }
    throw std::runtime_error("other three PEs never reached the Group");
}
void member(int64_t *) {}

struct Wire {
    WseGraphDescriptor descriptor{};
    WseNodePlan nodes[2]{};
    WseBlockPlan placements[3]{};
    uint32_t indices[core_count + 1]{0, 0, 0, 2, 2, 2, 2, 3};
    wse::CalendarRow rows[3]{{0, 0, 0}, {1, 0, 1}, {1, 1, 1}};
    uint32_t pe_indices[core_count * WSE_PE_PER_CORE + 1]{};
    uint32_t pe_rows[5]{1, 0, 1, 2, 2};
};
}  // namespace

int main(int argc, char **argv) {
    if (argc != 3) return 1;
    fail = std::string_view(argv[1]) == "fail";
    sync_start = std::string_view(argv[1]) != "independent";
    alignas(128) std::byte tasks[2 * SCHEDULER_GRAPH_TASK_STORAGE_STRIDE]{}, sram[core_count][16384]{};
    wse::SramRegion regions[core_count];
    for (uint32_t core = 0; core < core_count; ++core) {
        const auto base = reinterpret_cast<uintptr_t>(sram[core]);
        regions[core] = {base, sizeof(sram[core]), sizeof(sram[core]) / 2, base + sizeof(sram[core]) / 2};
    }
    Wire wire;
    auto &d = wire.descriptor;
    d.graph = scheduler_ordinary_graph_view(reinterpret_cast<uintptr_t>(tasks), 2, 1);
    d.nodes_address = reinterpret_cast<uintptr_t>(wire.nodes);
    d.descriptor_bytes = sizeof(wire);
    d.expected_tasks = 2;
    d.calendar_core_count = core_count;
    d.calendar_row_count = 3;
    d.calendar_indices_offset = offsetof(Wire, indices);
    d.calendar_rows_offset = offsetof(Wire, rows);
    d.pe_calendar_indices_offset = offsetof(Wire, pe_indices);
    d.pe_calendar_rows_offset = offsetof(Wire, pe_rows);
    d.pe_calendar_row_count = 5;
    d.pe_calendar_version = wse::PE_CALENDAR_VERSION;
    for (uint32_t task = 0; task < 2; ++task) {
        auto &node = wire.nodes[task];
        node.entry = reinterpret_cast<uintptr_t>(task ? member : busy);
        node.blocks = node.block_plan_count = task + 1;
        node.block_plans_offset = offsetof(Wire, placements) + task * sizeof(WseBlockPlan);
        node.sync_start = task && sync_start;
    }
    wire.placements[0] = {busy_core, 1, {busy_pe}};
    wire.placements[1] = {busy_core, 2, {2, 3}};
    wire.placements[2] = {peer_core, 2, {2, 3}};
    for (const auto &placement : wire.placements)
        for (uint32_t i = 0; i < placement.thread_num; ++i)
            ++wire.pe_indices[placement.core_id * WSE_PE_PER_CORE + placement.thread_pes[i] + 1];
    for (size_t i = 1; i < std::size(wire.pe_indices); ++i)
        wire.pe_indices[i] += wire.pe_indices[i - 1];
    char error[256]{};
    execution = wse_create_run_v8(&d, core_count, 1, regions, core_count, nullptr, 0, 1, error, sizeof(error));
    if (!execution) {
        std::fprintf(stderr, "%s\n", error);
        return 1;
    }
    std::vector<std::thread> pes;
    std::array<int, core_count * WSE_PE_PER_CORE> results{};
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
    uint64_t last_peer_seq = 0, busy_end_seq = 0, delayed_pe_seq = 0, release_seq = 0;
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
            if (e.kind == (sync_start ? WseEventKind::PE_ARRIVE : WseEventKind::THREAD_BEGIN)) {
                if (e.core == busy_core && e.pe == busy_pe) delayed_pe_seq = e.sequence;
                else last_peer_seq = std::max(last_peer_seq, e.sequence);
            }
            if (e.kind == WseEventKind::SYNC_RELEASE) release_seq = e.sequence;
            ready += e.kind == WseEventKind::PE_ARRIVE;
            release += e.kind == WseEventKind::SYNC_RELEASE;
            begin += e.kind == WseEventKind::BLOCK_BEGIN;
        }
    const int expected_rc = fail ? -1 : 0;
    const bool all_retired = std::all_of(results.begin(), results.end(), [expected_rc](int rc) {
        return rc == expected_rc;
    });
    const bool ordered = fail ? busy_end_seq == 0 && delayed_pe_seq == 0 :
                                last_peer_seq < busy_end_seq && busy_end_seq < delayed_pe_seq &&
                                    (!sync_start || delayed_pe_seq < release_seq);
    const bool lifecycle =
        go_from_last_pe && (fail || (last_retired_core != UINT32_MAX && last_retired_core == run_done_core));
    const bool valid = lifecycle && witnessed && all_retired && ordered &&
                       ready == (fail       ? 3u :
                                 sync_start ? 4u :
                                              0u) &&
                       release == (sync_start && !fail ? 1u : 0u) && begin == (fail ? 0u : 2u);
    const bool captured = fail || wse::write_pe_swimlane(argv[2], d, 1, core_count, execution, wse_run_events_v8);
    wse_destroy_run(execution);
    if (!valid) std::fprintf(stderr, "witness=%d, READY=%u, RELEASE=%u, BEGIN=%u\n", witnessed, ready, release, begin);
    return valid && captured ? 0 : 1;
}
