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
#include "swimlane_records.h"
#include "host/raii_scope_guard.h"
#include <atomic>
#include <cstddef>
#include <fstream>
#include <unistd.h>

namespace wse {
namespace {
const char *record_kind(WseEventKind kind) {
    switch (kind) {
    case WseEventKind::THREAD_BEGIN:
        return "KERNEL_BEGIN";
    case WseEventKind::THREAD_END:
        return "KERNEL_END";
    case WseEventKind::BLOCK_SKIPPED:
        return "BLOCK_SKIPPED";
    case WseEventKind::TASK_DONE:
        return "TASK_DONE";
    case WseEventKind::PE_ARRIVE:
        return "PE_ARRIVE";
    case WseEventKind::SYNC_RELEASE:
        return "SYNC_RELEASE";
    default:
        return nullptr;
    }
}

void write_tasks(std::ostream &out, const WseGraphDescriptor &descriptor) {
    const auto *nodes = reinterpret_cast<const WseNodePlan *>(descriptor.nodes_address);
    const auto *base = reinterpret_cast<const std::byte *>(&descriptor);
    for (uint64_t i = 0; i < descriptor.graph.task_count; ++i) {
        const auto &node = nodes[i];
        out << (i ? ",\n" : "") << "{\"task_id\":" << i << ",\"block_num\":" << node.blocks
            << ",\"sync_start\":" << (node.sync_start ? "true" : "false") << ",\"waits\":[";
        const auto *waits = reinterpret_cast<const uint32_t *>(base + node.waits_offset);
        for (uint32_t j = 0; j < node.wait_count; ++j)
            out << (j ? "," : "") << waits[j];
        out << "]}";
    }
}
}  // namespace

bool write_swimlane_records(
    const std::filesystem::path &output_dir, const WseGraphDescriptor &descriptor, uint64_t epoch, int32_t level,
    const WseRun *run, decltype(&wse_run_events_v8) run_events
) {
    const auto target = output_dir / "chip_swimlane_records.json";
    static std::atomic<uint64_t> next_file{0};
    const auto temp = target.string() + ".tmp." + std::to_string(getpid()) + "." + std::to_string(next_file++);
    auto cleanup = RAIIScopeGuard([&] {
        std::error_code error;
        std::filesystem::remove(temp, error);
    });
    std::ofstream out(temp);
    if (!out) return false;
    const uint32_t cores = descriptor.calendar_core_count;
    out << "{\"chip_swimlane_level\":" << level << ",\"metadata\":{\"platform\":\"" << SIMPLER_PLATFORM_NAME
        << "\",\"runtime\":\"" << SIMPLER_RUNTIME_NAME
        << "\",\"scheduler_model\":\"calendar\","
           "\"device_clock_domain\":\"host_monotonic_ns\",\"run_epoch\":"
        << epoch << ",\"core_count\":" << cores << ",\"pes_per_core\":" << WSE_PE_PER_CORE << "},\n\"tasks\":[";
    write_tasks(out, descriptor);
    out << "],\n\"pe_records\":[";
    bool first = true;
    const uint32_t lanes = cores * WSE_PE_PER_CORE;
    for (uint32_t lane = 0; lane <= lanes; ++lane) {
        uint64_t count = 0;
        const auto *events = run_events(run, lane == lanes ? UINT32_MAX : lane, &count);
        if (count && !events) return false;
        for (uint64_t i = 0; i < count; ++i) {
            const auto &e = events[i];
            const char *kind = record_kind(e.kind);
            if (!kind) continue;
            out << (first ? "" : ",\n") << "{\"kind\":\"" << kind << "\",\"seq\":" << e.sequence
                << ",\"monotonic_ns\":" << e.monotonic_ns << ",\"task_id\":" << e.task;
            first = false;
            if (lane == lanes) out << ",\"core_id\":null,\"pe_id\":null";
            else out << ",\"core_id\":" << e.core << ",\"pe_id\":" << e.pe;
            if (e.kind != WseEventKind::TASK_DONE) out << ",\"block_idx\":" << e.block;
            out << '}';
        }
    }
    out << "]}\n";
    out.close();
    if (out.fail()) return false;
    std::error_code error;
    std::filesystem::rename(temp, target, error);
    return !error;
}
}  // namespace wse
