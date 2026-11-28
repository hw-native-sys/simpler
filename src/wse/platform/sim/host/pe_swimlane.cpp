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
#include "pe_swimlane.h"
#include "host/raii_scope_guard.h"
#include <algorithm>
#include <atomic>
#include <ctime>
#include <fstream>
#include <queue>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>
#include <unistd.h>
#ifdef __APPLE__
#include <mach/mach_time.h>
#endif

namespace wse {
namespace {
// Fields below follow perfetto.protos.Trace, TracePacket, ClockSnapshot, TrackDescriptor,
// TrackEvent and DebugAnnotation. Each binary string holds at most one protobuf message.
using Message = std::string;
void varint(Message &message, uint64_t value) {
    while (value >= 128) {
        message.push_back(static_cast<char>((value & 127) | 128));
        value >>= 7;
    }
    message.push_back(static_cast<char>(value));
}
void integer(Message &message, uint32_t field, uint64_t value) {
    varint(message, field << 3);
    varint(message, value);
}
void bytes(Message &message, uint32_t field, std::string_view value) {
    varint(message, (field << 3) | 2);
    varint(message, value.size());
    message.append(value);
}
void fixed64(Message &message, uint32_t field, uint64_t value) {
    varint(message, (field << 3) | 1);
    for (unsigned shift = 0; shift < 64; shift += 8)
        message.push_back(static_cast<char>((value >> shift) & 255));
}
void packet(std::ostream &out, Message payload) {
    integer(payload, 10, 1);  // TracePacket.trusted_packet_sequence_id
    Message prefix;
    varint(prefix, 10);  // Trace.packet (length-delimited)
    varint(prefix, payload.size());
    out.write(prefix.data(), prefix.size());
    out.write(payload.data(), payload.size());
}

constexpr uint64_t MONOTONIC = 3, BOOTTIME = 6;  // perfetto.protos.BuiltinClock
bool clock_ns(clockid_t id, uint64_t &ns) {
    timespec instant{};
    if (clock_gettime(id, &instant) != 0) return false;
    ns = static_cast<uint64_t>(instant.tv_sec) * 1000000000 + instant.tv_nsec;
    return true;
}
bool write_clock_snapshot(std::ostream &out) {
    uint64_t monotonic, boot;
    if (!clock_ns(CLOCK_MONOTONIC, monotonic)) return false;
#ifdef __APPLE__
    mach_timebase_info_data_t scale{};
    if (mach_timebase_info(&scale) != KERN_SUCCESS || !scale.denom) return false;
    const auto ticks = mach_continuous_time();
    boot = ticks / scale.denom * scale.numer + (ticks % scale.denom) * scale.numer / scale.denom;
#else
    if (!clock_ns(CLOCK_BOOTTIME, boot)) return false;
#endif
    Message snapshot;
    integer(snapshot, 2, MONOTONIC);  // ClockSnapshot.primary_trace_clock
    for (const auto &[id, stamp] : {std::pair{MONOTONIC, monotonic}, std::pair{BOOTTIME, boot}}) {
        Message clock;
        integer(clock, 1, id);
        integer(clock, 2, stamp);
        bytes(snapshot, 1, clock);  // ClockSnapshot.clocks
    }
    Message payload;
    bytes(payload, 6, snapshot);  // TracePacket.clock_snapshot
    packet(out, std::move(payload));
    return true;
}

constexpr uint64_t ROOT_TRACK = 1;
uint64_t core_track(uint32_t core) { return 2 + uint64_t{core} * (WSE_PE_PER_CORE + 1); }
uint64_t pe_track(uint32_t core, uint32_t pe) { return core_track(core) + 1 + pe; }
void track(std::ostream &out, uint64_t uuid, std::string_view name, uint64_t parent = 0, uint32_t rank = 0) {
    Message desc;
    integer(desc, 1, uuid);
    bytes(desc, 2, name);
    integer(desc, 11, 3);  // TrackDescriptor.child_ordering = EXPLICIT
    if (parent) {
        integer(desc, 5, parent);
        integer(desc, 12, rank);
    }
    Message payload;
    bytes(payload, 60, desc);  // TracePacket.track_descriptor
    packet(out, std::move(payload));
}
void annotation(Message &event, std::string_view name, uint64_t value) {
    Message arg;
    bytes(arg, 10, name);    // DebugAnnotation.name
    integer(arg, 3, value);  // DebugAnnotation.uint_value
    bytes(event, 4, arg);    // TrackEvent.debug_annotations
}

struct FlowRange {
    uint64_t first, count;
};
struct TaskFlows {
    std::vector<uint64_t> arrived, released;
    std::vector<FlowRange> consumers;
    std::vector<uint64_t> predecessors;
    uint64_t endpoint_count = 0, next_endpoint = 0;
    bool done = false;
};
WseEventKind dependency_endpoint(bool sync_start) {
    return sync_start ? WseEventKind::PE_ARRIVE : WseEventKind::THREAD_BEGIN;
}
// A global event sequence identifies a single arrival or kernel start in this trace.
uint64_t flow_id(const WseEvent &event) { return event.sequence + 1; }
struct Stream {
    const WseEvent *next, *end;
    const WseEvent *open = nullptr;
    bool operator<(const Stream &other) const {
        // Merge immutable per-emitter streams in timestamp order. Simultaneous endpoints retain sequence order.
        return std::tie(next->monotonic_ns, next->sequence) > std::tie(other.next->monotonic_ns, other.next->sequence);
    }
};
uint64_t block_key(const WseEvent &e) { return (e.task << 32) | e.block; }

void add_flows(Message &event, const WseEvent &e, TaskFlows &flows, bool sync_start) {
    // TrackEvent.flow_ids starts a flow; terminating_flow_ids closes it on the receiving slice.
    if (e.kind == dependency_endpoint(sync_start)) {
        for (auto first : flows.predecessors)
            fixed64(event, 48, first + flows.next_endpoint);
        ++flows.next_endpoint;
    }
    if (e.kind == WseEventKind::TASK_DONE) {
        for (const auto &range : flows.consumers)
            for (uint64_t i = 0; i < range.count; ++i)
                fixed64(event, 47, range.first + i);
    } else if (sync_start && e.kind == WseEventKind::THREAD_BEGIN) {
        fixed64(event, 48, flow_id(e));
    } else if (sync_start && e.kind == WseEventKind::PE_ARRIVE) {
        fixed64(event, 47, flow_id(e));
    } else if (sync_start && e.kind == WseEventKind::SYNC_RELEASE) {
        for (auto id : flows.arrived)
            fixed64(event, 48, id);
        for (auto id : flows.released)
            fixed64(event, 47, id);
    }
}

void write_event(
    std::ostream &out, const WseEvent &e, uint64_t epoch, const WseNodePlan *node, TaskFlows *flows, bool skipped
) {
    using Kind = WseEventKind;
    const bool begin = e.kind == Kind::THREAD_BEGIN, end = e.kind == Kind::THREAD_END;
    const char *marker = nullptr;
    if (!begin && !end) {
        switch (e.kind) {
        case Kind::TASK_DONE:
            if (!flows || flows->consumers.empty()) return;
            marker = "TASK_DONE";
            break;
        case Kind::PE_ARRIVE:
            marker = "PE_ARRIVE";
            break;
        case Kind::SYNC_RELEASE:
            marker = "SYNC_RELEASE";
            break;
        default:
            return;
        }
    }
    Message event;
    integer(event, 9, begin ? 1 : end ? 2 : 3);  // TrackEvent.type: SLICE_BEGIN, SLICE_END, INSTANT
    integer(event, 11, e.core == UINT32_MAX ? ROOT_TRACK : pe_track(e.core, e.pe));
    bytes(event, 22, begin || end ? (skipped ? "wse.skipped" : "wse.kernel") : "wse.event");
    if (begin) {
        const std::string prefix = node->blocks > 1 ? "Group · T" : "SS · T";
        bytes(event, 23, prefix + std::to_string(e.task));  // TrackEvent.name
    } else if (marker) {
        bytes(event, 23, marker);
    }
    annotation(event, "epoch", epoch);
    annotation(event, "seq", e.sequence);
    if (e.core != UINT32_MAX) {
        annotation(event, "core_id", e.core);
        annotation(event, "pe_id", e.pe);
    }
    if (node) {
        annotation(event, "task", e.task);
        if (e.kind != Kind::TASK_DONE) annotation(event, "block_idx", e.block);
    }
    if (flows) add_flows(event, e, *flows, node->sync_start);
    Message payload;
    integer(payload, 8, e.monotonic_ns);  // TracePacket.timestamp
    integer(payload, 58, MONOTONIC);      // TracePacket.timestamp_clock_id
    bytes(payload, 11, event);            // TracePacket.track_event
    packet(out, std::move(payload));
}

bool write_events(
    std::ostream &out, const WseGraphDescriptor &descriptor, uint64_t epoch, uint32_t cores, const WseRun *run,
    decltype(&wse_run_events_v8) run_events
) {
    using Kind = WseEventKind;
    const auto *nodes = reinterpret_cast<const WseNodePlan *>(descriptor.nodes_address);
    std::vector<TaskFlows> tasks(descriptor.graph.task_count);
    uint64_t next_id = 1;
    std::unordered_set<uint64_t> skipped;
    std::priority_queue<Stream> streams;
    const uint32_t pe_count = cores * WSE_PE_PER_CORE;
    for (uint32_t index = 0; index <= pe_count; ++index) {
        uint64_t count = 0;
        const auto *events = run_events(run, index == pe_count ? UINT32_MAX : index, &count);
        if (!count) continue;
        if (!events) return false;
        streams.push({events, events + count});
        for (uint64_t i = 0; i < count; ++i) {
            const auto &e = events[i];
            if (e.task == UINT64_MAX) continue;
            if (!nodes || e.task >= descriptor.graph.task_count) return false;
            if (e.kind == Kind::BLOCK_SKIPPED) skipped.insert(block_key(e));
            auto &task = tasks[e.task];
            if (e.kind == Kind::TASK_DONE) task.done = true;
            if (e.kind == dependency_endpoint(nodes[e.task].sync_start)) ++task.endpoint_count;
            if (nodes[e.task].sync_start && (e.kind == Kind::PE_ARRIVE || e.kind == Kind::THREAD_BEGIN)) {
                if (e.sequence >= UINT64_MAX - 1) return false;
                (e.kind == Kind::PE_ARRIVE ? task.arrived : task.released).push_back(flow_id(e));
                next_id = std::max(next_id, flow_id(e) + 1);
            }
        }
    }
    // The run already validated its wait tables. Reserve one ID range per observed calendar edge,
    // so a predecessor shared by thousands of threads takes one entry until its packet is written.
    const auto *base = reinterpret_cast<const std::byte *>(&descriptor);
    for (size_t to = 0; to < tasks.size(); ++to) {
        auto &consumer = tasks[to];
        if (!consumer.endpoint_count) continue;
        const auto *waits = reinterpret_cast<const uint32_t *>(base + nodes[to].waits_offset);
        for (uint32_t i = 0; i < nodes[to].wait_count; ++i) {
            if (waits[i] >= tasks.size()) return false;
            auto &producer = tasks[waits[i]];
            if (!producer.done) continue;
            if (UINT64_MAX - next_id < consumer.endpoint_count) return false;
            producer.consumers.push_back({next_id, consumer.endpoint_count});
            consumer.predecessors.push_back(next_id);
            next_id += consumer.endpoint_count;
        }
    }
    while (!streams.empty()) {
        Stream stream = streams.top();
        streams.pop();
        const auto &e = *stream.next;
        const auto *node = e.task == UINT64_MAX ? nullptr : &nodes[e.task];
        if (e.kind == Kind::THREAD_BEGIN) {
            if (!node || stream.open) return false;
            stream.open = &e;
        } else if (e.kind == Kind::THREAD_END) {
            if (!stream.open || stream.open->task != e.task || stream.open->block != e.block ||
                stream.open->core != e.core || stream.open->pe != e.pe)
                return false;
            stream.open = nullptr;
        }
        write_event(out, e, epoch, node, node ? &tasks[e.task] : nullptr, skipped.count(block_key(e)) != 0);
        if (++stream.next != stream.end) streams.push(stream);
        else if (stream.open) return false;
    }
    return static_cast<bool>(out);
}
}  // namespace

bool write_pe_swimlane(
    const std::filesystem::path &output_dir, const WseGraphDescriptor &descriptor, uint64_t epoch, uint32_t core_count,
    const WseRun *run, decltype(&wse_run_events_v8) run_events
) {
    const auto target = output_dir / "wse_swimlane.pftrace";
    // Readers see a complete run even when a later write to the same prefix fails.
    static std::atomic<uint64_t> next_file{0};
    const auto temp = target.string() + ".tmp." + std::to_string(getpid()) + "." + std::to_string(next_file++);
    auto cleanup = RAIIScopeGuard([&] {
        std::error_code error;
        std::filesystem::remove(temp, error);
    });
    std::ofstream out(temp, std::ios::binary | std::ios::trunc);
    if (!out || !write_clock_snapshot(out)) return false;
    track(out, ROOT_TRACK, "WSE");
    for (uint32_t core = 0; core < core_count; ++core) {
        track(out, core_track(core), "Core " + std::to_string(core), ROOT_TRACK, core);
        for (uint32_t pe = 0; pe < WSE_PE_PER_CORE; ++pe)
            track(out, pe_track(core, pe), "PE " + std::to_string(pe), core_track(core), pe);
    }
    if (!write_events(out, descriptor, epoch, core_count, run, run_events)) return false;
    out.close();
    if (out.fail()) return false;
    std::error_code error;
    std::filesystem::rename(temp, target, error);
    return !error;
}
}  // namespace wse
