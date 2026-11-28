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

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <vector>

#include "calendar_completion.h"
#include "calendar_plan.h"

namespace simpler::percounter {

struct RangeCalendarInput {
    const CalendarPhase *phase;
    std::shared_ptr<const CalendarPlan> plan;
    bool sample;
};

struct RangeCalendarImage {
    enum class WaitKind { COMPLETION, PHASE_END, MATERIALIZED, FANIN, FUSED_FANIN, FUSED_PRODUCER };

    struct WaitPatch {
        uint32_t offset;
        WaitKind kind;
        uint32_t index;
        uint32_t target;
    };

    struct DispatchPatch {
        uint32_t offset;
        uint32_t global_task;
        int32_t shell;
        uint32_t role;
    };

    struct ExtraPatch {
        uint32_t offset;
        uint32_t global_task;
        int32_t shell;
    };

    struct PhaseEnd {
        uint32_t task{UINT32_MAX};
        uint32_t slot{UINT32_MAX};
        uint32_t contributors{0};
    };

    struct RangeTail {
        uint32_t range;
        uint32_t phase;
    };

    struct FusedFanin {
        std::array<uint32_t, CALENDAR_FUSED_FANIN_MAX_PRODUCERS> producers;
        uint32_t producer_count;
        uint32_t slot;
        uint32_t credits;
    };

    std::vector<std::byte> bytes;
    std::vector<WaitPatch> waits;
    std::vector<DispatchPatch> dispatches;
    std::vector<ExtraPatch> extras;
    std::vector<PhaseEnd> phase_ends;
    std::vector<RangeTail> tails;
    std::vector<uint32_t> sample_offsets;
    std::vector<uint32_t> graph_shells;
    std::vector<FusedFanin> fused_fanins;
    uint32_t peer_endpoints_offset{0};
    uint32_t peer_fanouts_offset{0};
    uint32_t peer_single_waits{0};
    uint32_t peer_double_waits{0};
    uint32_t peer_spmd_edges{0};
    uint32_t peer_spmd_wait_rows{0};
    CalendarCompletionPlan completion;
    std::array<uint64_t, 3> sync_widths{};
    uint32_t phase_slot_count{0};
    uint32_t fanin_slot_count{0};
    bool uses_cluster_input_share{false};
    bool has_task_timing{false};

    template <class T>
    uint32_t allocate(size_t count) {
        if (count == 0) return 0;
        const uint64_t offset = (bytes.size() + alignof(T) - 1) & ~uint64_t{alignof(T) - 1};
        if (offset == 0 || offset > UINT32_MAX || count > (UINT32_MAX - offset) / sizeof(T)) return 0;
        bytes.resize(offset + count * sizeof(T));
        return static_cast<uint32_t>(offset);
    }

    template <class T>
    void store(uint32_t offset, const T &value) {
        std::memcpy(bytes.data() + offset, &value, sizeof(T));
    }

    template <class T>
    T load(uint32_t offset) const {
        T result;
        std::memcpy(&result, bytes.data() + offset, sizeof(T));
        return result;
    }
};

namespace range_calendar_detail {

struct Template {
    CalendarHeader header;
    std::vector<uint32_t> indexes;
    std::vector<CalendarRow> rows;
    std::vector<CalendarLocalJoin> joins;
    std::vector<std::array<uint32_t, 3>> binding_indexes;
    std::vector<bool> polls_task_fanins;
    uint32_t binding_count{0};
    uint32_t rows_offset{0};
    uint32_t joins_offset{0};
};

inline bool decode_template(
    RangeCalendarImage &image, const CalendarPhase &phase, const CalendarPlan &plan, const HostGraphTemplate *graph,
    uint32_t clusters, Template &result
) {
    if (plan.bytes.size() < sizeof(result.header)) return false;
    std::memcpy(&result.header, plan.bytes.data(), sizeof(result.header));
    const auto &header = result.header;
    if (header.worker_count != clusters * 3 || header.row_count == 0 ||
        !calendar_detail::read_table(plan, header.index_offset, header.worker_count + 1, &result.indexes) ||
        !calendar_detail::read_table(plan, header.row_offset, header.row_count, &result.rows) ||
        !calendar_detail::read_table(plan, header.join_offset, header.join_count, &result.joins))
        return false;

    result.rows_offset = image.allocate<CalendarExecutionRow>(header.row_count);
    result.joins_offset = image.allocate<CalendarLocalJoin>(result.joins.size());
    if (result.rows_offset == 0 || (!result.joins.empty() && result.joins_offset == 0) || result.indexes.front() != 0 ||
        result.indexes.back() != header.row_count)
        return false;
    if (!result.joins.empty())
        std::memcpy(
            image.bytes.data() + result.joins_offset, result.joins.data(),
            result.joins.size() * sizeof(CalendarLocalJoin)
        );

    const auto &waits = plan.waits;
    if (waits.row_waits.size() != header.row_count || waits.row_input_owners.size() != header.row_count ||
        waits.task_fanins.size() != phase.tasks.size())
        return false;
    result.binding_indexes.resize(phase.tasks.size());
    result.polls_task_fanins.resize(phase.tasks.size());
    for (size_t task = 0; task < phase.tasks.size(); ++task) {
        // A core may cache the last dispatch in a published slice while
        // AICPU is still preparing the next one. Keep slices off the
        // same cache line so observing a newer watermark suffices.
        if (graph != nullptr && task % PERCOUNTER_GRAPH_SLICE_TASKS == 0)
            result.binding_count += result.binding_count % 2;
        result.binding_indexes[task].fill(UINT32_MAX);
        const auto mask = phase.tasks[task].active_mask;
        for (uint32_t role = 0; role < 3; ++role) {
            if ((mask & (1U << role)) || (mask == 0 && role == 1))
                result.binding_indexes[task][role] = result.binding_count++;
        }
    }
    for (uint32_t worker = 0; worker < header.worker_count; ++worker) {
        const uint32_t begin = result.indexes[worker];
        const uint32_t end = result.indexes[worker + 1];
        if (begin > end || end > header.row_count) return false;
        const uint32_t lane = worker < clusters ? 0 : 1 + (worker - clusters) % 2;
        for (uint32_t i = begin; i < end; ++i) {
            const auto &row = result.rows[i];
            const uint32_t task = static_cast<uint32_t>(percounter_order_task_id(row.entry));
            if (task >= phase.tasks.size()) return false;
            const auto &input = phase.tasks[task];
            const uint32_t role = calendar_logical_role(input.active_mask, lane);
            const uint32_t binding = result.binding_indexes[task][role];
            if (binding == UINT32_MAX || waits.task_fanins[task].size() > UINT16_MAX) return false;
            const uint32_t offset = result.rows_offset + i * sizeof(CalendarExecutionRow);
            uint32_t mix_delta = 0;
            uint16_t wait_count = waits.row_waits[i] ? static_cast<uint16_t>(waits.task_fanins[task].size()) : 0;
            if (row.local_join != CALENDAR_NO_JOIN) {
                if (row.local_join >= result.joins.size()) return false;
                mix_delta = result.joins_offset + row.local_join * sizeof(CalendarLocalJoin) - offset;
                if (result.joins[row.local_join].leader_lane != lane) wait_count = 0;
            }
            uint16_t flags = static_cast<uint16_t>(role);
            if (input.sync_start) flags |= CALENDAR_EXEC_SYNC_START;
            if (input.has_predicate) flags |= CALENDAR_EXEC_PREDICATE;
            if (input.active_mask == 0) flags |= CALENDAR_EXEC_CONTROL;
            if (wait_count != 0 && waits.summarized_fanins[task]) {
                flags |= CALENDAR_EXEC_FANIN;
                const auto &assignment = waits.row_assignments[i];
                mix_delta = assignment.begin;
                wait_count = assignment.count;
            }
            const uint32_t owner = waits.row_input_owners[i];
            if (owner != CALENDAR_NO_INPUT_OWNER) {
                const bool summary = (flags & CALENDAR_EXEC_FANIN) != 0;
                if (lane == 0 || input.is_mix || input.sync_start || input.has_predicate || input.active_mask == 0 ||
                    row.local_join != CALENDAR_NO_JOIN || (!summary && wait_count != 1) ||
                    (summary && owner != i && wait_count != 0) || owner >= header.row_count ||
                    waits.row_input_owners[owner] != owner ||
                    percounter_order_task_id(result.rows[owner].entry) != task)
                    return false;
                const uint32_t owner_worker = owner == i ? worker : (lane == 1 ? worker + 1 : worker - 1);
                const uint32_t owner_begin = result.indexes[owner_worker];
                if (owner < owner_begin || owner >= result.indexes[owner_worker + 1]) return false;
                if (owner == i) {
                    // A summary helper keeps its assignment's begin in mix_delta.
                    flags |= CALENDAR_EXEC_INPUT_OWNER;
                } else {
                    flags |= CALENDAR_EXEC_INPUT_FOLLOWER;
                    mix_delta = owner - owner_begin + 1;
                    wait_count = 0;
                }
                image.uses_cluster_input_share = true;
            }
            // A cluster follower can fall back to its producer's original DONE wait.
            result.polls_task_fanins[task] =
                result.polls_task_fanins[task] || wait_count != 0 || (flags & CALENDAR_EXEC_INPUT_FOLLOWER) != 0;
            if (graph != nullptr && graph->tasks[task].sink) flags |= CALENDAR_EXEC_SINK;
            image.store(
                offset, CalendarExecutionRow{
                            static_cast<uint32_t>(binding * sizeof(CalendarDispatch)), mix_delta,
                            static_cast<uint16_t>(percounter_order_block_idx(row.entry)),
                            static_cast<uint16_t>(percounter_order_block_num(row.entry)), wait_count, flags
                        }
            );
        }
    }
    return true;
}

struct RangeRecord {
    CalendarRange range;
    uint32_t timing;
    int32_t end_phase{-1};
};

inline uint32_t physical_task(uint32_t local, const CalendarPhase &phase, const std::vector<uint32_t> &body_base) {
    return phase.invocation < 0 ? phase.task_ids[local] : body_base[phase.invocation] + phase.task_ids[local];
}

inline uint32_t sole_sink(const CalendarPhase &phase) {
    std::vector<bool> consumed(phase.tasks.size(), false);
    for (const auto &row : phase.fanins)
        for (const int32_t input : row)
            consumed[input] = true;
    uint32_t result = UINT32_MAX;
    for (uint32_t i = 0; i < consumed.size(); ++i) {
        if (consumed[i]) continue;
        if (result != UINT32_MAX) return UINT32_MAX;
        result = i;
    }
    return result;
}

inline void materialize_fused_fanins(
    const CalendarPhase &phase, const CalendarWaitPlan &waits, RangeCalendarImage &image,
    std::vector<uint32_t> &fused_consumers, std::vector<bool> &fused_producers
) {
    if (waits.fused_fanin_credits.empty()) return;
    fused_consumers.assign(phase.tasks.size(), UINT32_MAX);
    fused_producers.resize(phase.tasks.size());
    for (size_t consumer = 0; consumer < phase.tasks.size(); ++consumer) {
        const uint32_t credits = waits.fused_fanin_credits[consumer];
        if (credits == 0) continue;
        const auto &inputs = waits.task_fanins[consumer];
        RangeCalendarImage::FusedFanin group{};
        group.producer_count = static_cast<uint32_t>(inputs.size());
        group.credits = credits;
        group.slot = image.fanin_slot_count++;
        for (size_t i = 0; i < inputs.size(); ++i) {
            group.producers[i] = phase.task_ids[inputs[i]];
            fused_producers[inputs[i]] = true;
        }
        fused_consumers[consumer] = static_cast<uint32_t>(image.fused_fanins.size());
        image.fused_fanins.push_back(group);
    }
}

// One wait per physical worker per launch. The send list can contain several workers
// (fan-out); the ordinal identifies a source within its destination's wait.
struct PeerTask {
    struct Send {
        uint32_t worker;
        uint32_t ordinal;
    };
    std::vector<Send> sends;
    uint16_t input_count{0};
    std::vector<uint32_t> spmd_targets;
    uint32_t token_begin{0};
    uint32_t fanout_offset{0};
};

inline bool prepare_peer_fanouts(RangeCalendarImage &image, std::vector<PeerTask> &peers) {
    std::map<std::vector<uint32_t>, uint32_t> sets;
    for (auto &peer : peers) {
        const auto &workers = peer.spmd_targets;
        if (workers.size() < 2) continue;
        auto [it, inserted] = sets.try_emplace(workers, 0);
        if (inserted) {
            const size_t size = sizeof(CalendarPeerFanout) + workers.size() * CALENDAR_FANOUT_TARGET_BYTES;
            const auto offset = image.allocate<uint64_t>(size / sizeof(uint64_t));
            if (offset == 0) return false;
            image.store(
                offset, CalendarPeerFanout{image.peer_fanouts_offset, static_cast<uint16_t>(workers.size()), 0, 0}
            );
            std::memcpy(
                image.bytes.data() + offset + size - workers.size() * sizeof(uint32_t), workers.data(),
                workers.size() * sizeof(uint32_t)
            );
            image.peer_fanouts_offset = offset;
            it->second = offset;
        }
        peer.fanout_offset = it->second;
    }
    return true;
}

inline void plan_spmd_notifications(
    const CalendarPhase &phase, const CalendarWaitPlan &waits, const Template &tmpl,
    std::vector<CalendarPeerEndpoint> &endpoints, std::vector<PeerTask> &peers, RangeCalendarImage &image,
    bool enable_peer_hints
) {
    // A wait must already carry a whole-task completion proof. Keep the original
    // counter and the MIX/sync_start rendezvous; only its polling mechanism changes.
    const size_t tasks = peers.size();
    std::vector<int32_t> producer_for(tasks, -1);
    const auto available = [&](size_t task) {
        const auto &shape = phase.tasks[task];
        return shape.active_mask != 0 && !shape.has_predicate && peers[task].input_count == 0 &&
               peers[task].sends.empty();
    };
    for (size_t task = 0; task < tasks; ++task) {
        const auto &inputs = waits.task_fanins[task];
        if (!available(task) || inputs.size() != 1 || phase.fanins[task].size() != 1 ||
            inputs[0] != phase.fanins[task][0] || waits.summarized_fanins[task])
            continue;
        const uint32_t source = inputs[0];
        if (!available(source) || (phase.tasks[source].active_mask & 6U) == 0) continue;
        const auto &from = phase.tasks[source];
        const auto &to = phase.tasks[task];
        if (from.block_num == 1 && from.subtask_count == 1 && to.block_num == 1 && to.subtask_count == 1) continue;
        producer_for[task] = source;
    }

    uint64_t next_token = 1;
    std::vector<bool> selected(tasks);
    for (size_t worker = 0; worker < endpoints.size(); ++worker) {
        // A worker reserved for a single row uses that row's assigned tokens.
        if (endpoints[worker].input_count != 0) continue;
        for (uint32_t i = tmpl.indexes[worker]; i < tmpl.indexes[worker + 1]; ++i) {
            const auto task = static_cast<size_t>(percounter_order_task_id(tmpl.rows[i].entry));
            const int32_t source = producer_for[task];
            if (source < 0) continue;
            const uint32_t offset = tmpl.rows_offset + i * sizeof(CalendarExecutionRow);
            auto row = image.load<CalendarExecutionRow>(offset);
            if (row.wait_count != 1 ||
                (row.flags & (CALENDAR_EXEC_FANIN | CALENDAR_EXEC_INPUT_OWNER | CALENDAR_EXEC_INPUT_FOLLOWER)))
                continue;
            if (!enable_peer_hints) {
                row.flags |= CALENDAR_EXEC_PACED_WAIT;
                image.store(offset, row);
                continue;
            }
            auto &producer = peers[source];
            if (producer.spmd_targets.empty()) {
                const uint64_t width = uint64_t{PLATFORM_CORES_PER_BLOCKDIM} * phase.tasks[source].block_num;
                if (next_token + width - 1 > UINT32_MAX) continue;
                producer.token_begin = static_cast<uint32_t>(next_token);
                next_token += width;
            }
            if (producer.spmd_targets.empty() || producer.spmd_targets.back() != worker)
                producer.spmd_targets.push_back(static_cast<uint32_t>(worker));
            endpoints[worker].input_count = CALENDAR_PEER_SPMD_ENDPOINT;
            ++image.peer_spmd_wait_rows;
            if (!selected[task]) ++image.peer_spmd_edges;
            selected[task] = true;
            row.flags |= CALENDAR_EXEC_SPMD_WAIT;
            image.store(offset, row);
        }
    }
}

inline std::vector<PeerTask> plan_peer_notifications(
    const CalendarPhase &phase, const CalendarWaitPlan &waits, const Template &tmpl, RangeCalendarImage &image,
    bool enable_peer_hints
) {
    const size_t tasks = phase.tasks.size();
    const size_t workers = tmpl.indexes.size() - 1;
    std::vector<uint32_t> owner(tasks, UINT32_MAX), row(tasks);
    for (uint32_t worker = 0; worker < workers; ++worker) {
        for (uint32_t i = tmpl.indexes[worker]; i < tmpl.indexes[worker + 1]; ++i) {
            const auto id = static_cast<size_t>(percounter_order_task_id(tmpl.rows[i].entry));
            const auto &task = phase.tasks[id];
            if (task.is_mix || task.sync_start || task.has_predicate || task.block_num != 1 ||
                task.subtask_count != 1 || task.active_mask == 0)
                continue;
            owner[id] = worker;
            row[id] = i;
        }
    }
    std::vector<PeerTask> result(tasks);
    std::vector<CalendarPeerEndpoint> endpoints(workers);
    for (uint32_t id = 0; id < tasks; ++id) {
        const uint32_t worker = owner[id];
        if (worker == UINT32_MAX || endpoints[worker].input_count != 0 || !waits.row_waits[row[id]]) continue;
        const auto &inputs = waits.task_fanins[id];
        if (inputs.empty() || inputs.size() > 2 || waits.summarized_fanins[id] ||
            waits.row_input_owners[row[id]] != CALENDAR_NO_INPUT_OWNER)
            continue;
        if (std::any_of(inputs.begin(), inputs.end(), [&](uint32_t producer) {
                return producer >= id || owner[producer] == UINT32_MAX || owner[producer] == worker;
            }))
            continue;
        const auto has_vector_sender = [&](uint32_t producer) {
            return (phase.tasks[producer].active_mask & 6U) != 0;
        };
        const bool has_sender = std::any_of(inputs.begin(), inputs.end(), has_vector_sender);
        if (!has_sender && inputs.size() != 1) continue;
        if (!has_sender || !enable_peer_hints) {
            const uint32_t offset = tmpl.rows_offset + row[id] * sizeof(CalendarExecutionRow);
            auto execution = image.load<CalendarExecutionRow>(offset);
            execution.flags |= CALENDAR_EXEC_PACED_WAIT;
            image.store(offset, execution);
        }
        if (!has_sender) continue;
        result[id].input_count = static_cast<uint16_t>(inputs.size());
        endpoints[worker].input_count = result[id].input_count;
        if (enable_peer_hints) {
            if (inputs.size() == 2) ++image.peer_double_waits;
            else ++image.peer_single_waits;
        }
        for (uint32_t i = 0; i < inputs.size(); ++i)
            if (has_vector_sender(inputs[i])) result[inputs[i]].sends.push_back({worker, i});
    }
    plan_spmd_notifications(phase, waits, tmpl, endpoints, result, image, enable_peer_hints);
    if (!enable_peer_hints) return {};

    const uint32_t first_vector_row = tmpl.indexes[workers / PLATFORM_CORES_PER_BLOCKDIM];
    for (size_t i = first_vector_row; i < tmpl.rows.size(); ++i) {
        const auto task = percounter_order_task_id(tmpl.rows[i].entry);
        if (result[task].spmd_targets.empty()) continue;
        const uint32_t offset = tmpl.rows_offset + i * sizeof(CalendarExecutionRow);
        auto execution = image.load<CalendarExecutionRow>(offset);
        execution.flags |= CALENDAR_EXEC_SPMD_SEND;
        image.store(offset, execution);
    }
    if (image.peer_single_waits != 0 || image.peer_double_waits != 0 || image.peer_spmd_edges != 0) {
        image.peer_endpoints_offset = image.allocate<CalendarPeerEndpoint>(workers);
        if (image.peer_endpoints_offset)
            std::memcpy(
                image.bytes.data() + image.peer_endpoints_offset, endpoints.data(),
                workers * sizeof(CalendarPeerEndpoint)
            );
        for (uint32_t id = 0; id < tasks; ++id) {
            if (result[id].input_count == 0 && result[id].sends.empty()) continue;
            const uint32_t offset = tmpl.rows_offset + row[id] * sizeof(CalendarExecutionRow);
            auto execution = image.load<CalendarExecutionRow>(offset);
            execution.flags |= CALENDAR_EXEC_PEER;
            image.store(offset, execution);
        }
    }
    return result;
}

inline uint32_t
emit_execution_extra(RangeCalendarImage &image, const CalendarExecutionExtra &execution, const PeerTask *peer) {
    if (peer != nullptr && !peer->spmd_targets.empty()) {
        const auto offset = image.allocate<CalendarSpmdPeerExtra>(1);
        const uint32_t count = static_cast<uint32_t>(peer->spmd_targets.size());
        if (offset == 0) return 0;
        image.store(
            offset, CalendarSpmdPeerExtra{
                        execution, {}, peer->spmd_targets.front(), count, peer->token_begin, peer->fanout_offset
                    }
        );
        if (count > 1) {
            image.waits.push_back(
                {static_cast<uint32_t>(offset + offsetof(CalendarSpmdPeerExtra, completion)),
                 RangeCalendarImage::WaitKind::COMPLETION, execution.task_id, 0}
            );
        }
        return offset;
    }
    if (peer == nullptr || (peer->sends.empty() && peer->input_count == 0)) {
        const auto offset = image.allocate<CalendarExecutionExtra>(1);
        if (offset) image.store(offset, execution);
        return offset;
    }
    const uint32_t offset = image.allocate<CalendarPeerExtra>(1);
    const uint32_t sends = image.allocate<CalendarPeerSend>(peer->sends.size());
    if (offset == 0 || (!peer->sends.empty() && sends == 0) || peer->sends.size() > UINT16_MAX) return 0;
    for (uint32_t i = 0; i < peer->sends.size(); ++i) {
        const auto &entry = peer->sends[i];
        const auto location = static_cast<uint32_t>(sends + i * sizeof(CalendarPeerSend));
        image.store(location, CalendarPeerSend{entry.worker, entry.ordinal});
    }
    image.store(
        offset, CalendarPeerExtra{execution, sends, static_cast<uint16_t>(peer->sends.size()), peer->input_count}
    );
    return offset;
}

}  // namespace range_calendar_detail

inline bool build_range_calendar(
    const std::vector<RangeCalendarInput> &inputs, const GraphPlanInput &graphs, uint32_t outer_count,
    uint32_t clusters, RangeCalendarImage *out, const SchedulerTaskMetadata *pmu_outer_tasks = nullptr,
    bool enable_peer_hints = true
) {
    using namespace range_calendar_detail;
    if (clusters == 0 || clusters > PLATFORM_MAX_BLOCKDIM || out == nullptr) return false;
    RangeCalendarImage image;
    image.bytes.resize(sizeof(CalendarRangeHeader));
    std::vector<uint32_t> body_base;
    uint32_t total_tasks = outer_count;
    for (size_t i = 0; i < graphs.invocation_count; ++i) {
        const auto &graph = graphs.invocations[i];
        const size_t tasks = graphs.templates[graph.template_index].tasks.size();
        if (tasks > UINT32_MAX - total_tasks) return false;
        body_base.push_back(total_tasks);
        total_tasks += static_cast<uint32_t>(tasks);
    }
    image.completion.tasks.resize(total_tasks);
    CompletionChains chains(image.completion);
    const uint32_t patch_offset = graphs.invocation_count == 0 ? 0 : image.allocate<CalendarDispatchPatch>(total_tasks);
    if (graphs.invocation_count != 0 && patch_offset == 0) return false;

    const uint32_t workers = clusters * 3;
    std::vector<std::vector<RangeRecord>> ranges(workers);
    std::map<const CalendarPlan *, Template> templates;
    uint32_t sequence_base = 0;
    for (uint32_t p = 0; p < inputs.size(); ++p) {
        const auto &input = inputs[p];
        if (input.phase == nullptr || input.plan == nullptr) return false;
        const auto &phase = *input.phase;
        const auto *invocation = phase.invocation < 0 ? nullptr : &graphs.invocations[phase.invocation];
        const auto *graph = invocation == nullptr ? nullptr : &graphs.templates[invocation->template_index];
        const int32_t shell = invocation == nullptr ? -1 : static_cast<int32_t>(invocation->shell_task_id);
        if (shell >= 0) image.graph_shells.push_back(static_cast<uint32_t>(shell));
        const uint32_t base = graph == nullptr ? 0 : body_base[phase.invocation];
        auto [entry, inserted] = templates.try_emplace(input.plan.get());
        auto &tmpl = entry->second;
        if (inserted && !decode_template(image, phase, *input.plan, graph, clusters, tmpl)) return false;

        std::vector<uint32_t> fused_consumers;
        std::vector<bool> fused_producers;
        std::vector<PeerTask> peer_tasks;
        if (inputs.size() == 1 && graphs.invocation_count == 0) {
            materialize_fused_fanins(phase, input.plan->waits, image, fused_consumers, fused_producers);
            peer_tasks = plan_peer_notifications(phase, input.plan->waits, tmpl, image, enable_peer_hints);
            if (!prepare_peer_fanouts(image, peer_tasks)) return false;
            if ((image.peer_single_waits || image.peer_double_waits || image.peer_spmd_edges) &&
                !image.peer_endpoints_offset)
                return false;
        }
        if (!chains.append_phase(
                phase, base, static_cast<uint32_t>(shell), graph == nullptr ? 0 : graph->sink_subtask_total,
                fused_producers
            ))
            return false;

        std::vector<bool> polled(phase.tasks.size(), false);
        for (size_t task = 0; task < phase.tasks.size(); ++task) {
            if (!tmpl.polls_task_fanins[task]) continue;
            for (int32_t predecessor : input.plan->waits.task_fanins[task]) {
                if (predecessor < 0 || static_cast<size_t>(predecessor) >= polled.size()) return false;
                polled[predecessor] = true;
            }
        }
        std::vector<uint32_t> exit_tasks;
        for (uint32_t task = 0; task < phase.tasks.size(); ++task)
            if (!polled[task] && phase.tasks[task].active_mask != 0)
                exit_tasks.push_back(physical_task(task, phase, body_base));
        const uint32_t exit_offset = image.allocate<CalendarWait>(exit_tasks.size());
        if (!exit_tasks.empty() && exit_offset == 0) return false;
        for (size_t i = 0; i < exit_tasks.size(); ++i)
            image.waits.push_back(
                {static_cast<uint32_t>(exit_offset + i * sizeof(CalendarWait)),
                 RangeCalendarImage::WaitKind::COMPLETION, exit_tasks[i], 0}
            );

        // Each Graph's patch writes own cache lines. Immutable range/wait
        // records may be read by other cores while it is being materialized.
        if (graph != nullptr) image.bytes.resize((image.bytes.size() + 63U) & ~size_t{63U});
        const uint32_t bindings = image.allocate<CalendarDispatch>(tmpl.binding_count);
        if (bindings == 0) return false;
        if (graph != nullptr) image.bytes.resize((image.bytes.size() + 63U) & ~size_t{63U});

        for (uint32_t task = 0; task < phase.tasks.size(); ++task) {
            const uint32_t global = physical_task(task, phase, body_base);
            const auto &task_input = phase.tasks[task];
            if (task_input.sync_start) {
                const uint32_t domain = task_input.is_mix ? 2 : task_input.core_type_index;
                image.sync_widths[domain] =
                    std::max(image.sync_widths[domain], static_cast<uint64_t>(task_input.block_num));
            }
            const auto &producers = input.plan->waits.task_fanins[task];
            const bool fanin = input.plan->waits.summarized_fanins[task];
            const uint32_t group = fused_consumers.empty() ? UINT32_MAX : fused_consumers[task];
            const uint32_t wait_offset = image.allocate<CalendarWait>(producers.size() + fanin);
            if (!producers.empty() && wait_offset == 0) return false;
            if (group != UINT32_MAX) {
                image.waits.push_back({wait_offset, RangeCalendarImage::WaitKind::FUSED_FANIN, group, 0});
            } else if (fanin) {
                image.waits.push_back(
                    {wait_offset, RangeCalendarImage::WaitKind::FANIN, image.fanin_slot_count++,
                     static_cast<uint32_t>(producers.size())}
                );
            }
            for (size_t i = 0; i < producers.size(); ++i) {
                const int32_t producer = producers[i];
                if (producer < 0 || static_cast<uint32_t>(producer) >= task) return false;
                // No execution row reads the input bindings of an unobserved task.
                if (!tmpl.polls_task_fanins[task] && !fused_producers.empty() && fused_producers[producer]) continue;
                const auto kind = group != UINT32_MAX ? RangeCalendarImage::WaitKind::FUSED_PRODUCER :
                                                        RangeCalendarImage::WaitKind::COMPLETION;
                image.waits.push_back(
                    {static_cast<uint32_t>(wait_offset + (i + fanin) * sizeof(CalendarWait)), kind,
                     physical_task(producer, phase, body_base), 0}
                );
            }
            if (task_input.timing_slot >= 0 && task_input.timing_slot < NUM_TASK_TIMING_SLOTS)
                image.has_task_timing = true;
            const uint32_t extra_offset = emit_execution_extra(
                image, {global, phase.task_ids[task], shell, task_input.timing_slot, 0, 0},
                peer_tasks.empty() ? nullptr : &peer_tasks[task]
            );
            if (extra_offset == 0) return false;
            image.extras.push_back({extra_offset, global, shell});
            CalendarDispatchPatch patch{};
            if (task_input.has_predicate) patch.predicate_extra_offset = extra_offset;
            for (uint32_t role = 0; role < 3; ++role) {
                const uint32_t index = tmpl.binding_indexes[task][role];
                if (index == UINT32_MAX) continue;
                const auto offset = static_cast<uint32_t>(bindings + index * sizeof(CalendarDispatch));
                image.store(
                    offset,
                    CalendarDispatch{0, 0, 0, wait_offset == 0 ? 0 : wait_offset - bindings, extra_offset - bindings}
                );
                image.dispatches.push_back({offset, global, shell, role});
                if (graph != nullptr && task_input.active_mask != 0) patch.dispatch_offsets[role] = offset;
            }
            if (graph != nullptr) image.store(patch_offset + global * sizeof(CalendarDispatchPatch), patch);
        }

        const uint32_t samples = input.sample ? image.allocate<CalendarTiming>(tmpl.header.row_count) : 0;
        if (input.sample && samples == 0) return false;
        image.sample_offsets.push_back(samples);

        const uint32_t pmu = pmu_outer_tasks == nullptr ? 0 : image.allocate<CalendarPmuCell>(tmpl.header.row_count);
        if (pmu_outer_tasks != nullptr) {
            if (pmu == 0) return false;
            for (uint32_t i = 0; i < tmpl.header.row_count; ++i) {
                const auto row = image.load<CalendarExecutionRow>(tmpl.rows_offset + i * sizeof(CalendarExecutionRow));
                if ((row.flags & CALENDAR_EXEC_CONTROL) != 0) continue;
                const auto task = static_cast<uint32_t>(percounter_order_task_id(tmpl.rows[i].entry));
                const uint32_t role = row.flags & CALENDAR_EXEC_ROLE_MASK;
                const auto kernel_id = graph == nullptr ? pmu_outer_tasks[phase.task_ids[task]].kernel_ids[role] :
                                                          graph->tasks[phase.task_ids[task]].kernel_ids[role];
                if (kernel_id == UINT16_MAX) return false;
                CalendarPmuCell cell{};
                cell.kernel_id = kernel_id;
                image.store(pmu + i * sizeof(CalendarPmuCell), cell);
            }
        }

        RangeCalendarImage::PhaseEnd end{};
        if (shell >= 0) {
            end.task = static_cast<uint32_t>(shell);
        } else if (const uint32_t sink = sole_sink(phase); sink != UINT32_MAX) {
            end.task = physical_task(sink, phase, body_base);
        }
        if (p + 1 < inputs.size() && end.task == UINT32_MAX) end.slot = image.phase_slot_count++;

        // Poll otherwise unobserved defer records after this worker's final row.
        // Other workers can finish their ranges while this one settles the records.
        uint32_t exit_worker = 0;
        if (!exit_tasks.empty()) {
            while (exit_worker < workers && tmpl.indexes[exit_worker] == tmpl.indexes[exit_worker + 1])
                ++exit_worker;
            if (exit_worker == workers) return false;
        }

        uint32_t longest = 0;
        for (uint32_t worker = 0; worker < workers; ++worker) {
            const uint32_t first = tmpl.indexes[worker];
            const uint32_t last = tmpl.indexes[worker + 1];
            longest = std::max(longest, last - first);
            uint32_t need = 0;
            const auto required = [&](uint32_t row) {
                const uint32_t task = static_cast<uint32_t>(percounter_order_task_id(tmpl.rows[row].entry));
                return std::min(
                    static_cast<uint32_t>(phase.tasks.size()),
                    (task / PERCOUNTER_GRAPH_SLICE_TASKS + 1) * PERCOUNTER_GRAPH_SLICE_TASKS
                );
            };
            for (uint32_t begin = first; begin < last;) {
                need = invocation == nullptr ? 0 : std::max(need, required(begin));
                uint32_t limit = begin + 1;
                while (limit < last && (invocation == nullptr || required(limit) <= need))
                    ++limit;
                const bool previous_phase = begin == first && p != 0;
                const uint32_t num_waits = previous_phase + (invocation != nullptr);
                const uint32_t wait_offset = image.allocate<CalendarWait>(num_waits);
                if (num_waits != 0 && wait_offset == 0) return false;
                if (previous_phase)
                    image.waits.push_back({wait_offset, RangeCalendarImage::WaitKind::PHASE_END, p - 1, 0});
                if (invocation != nullptr)
                    image.waits.push_back(
                        {static_cast<uint32_t>(wait_offset + previous_phase * sizeof(CalendarWait)),
                         RangeCalendarImage::WaitKind::MATERIALIZED, static_cast<uint32_t>(shell), need}
                    );
                const bool tail = limit == last && end.slot != UINT32_MAX;
                const bool settle = limit == last && worker == exit_worker && !exit_tasks.empty();
                if (tail) ++end.contributors;
                ranges[worker].push_back(
                    {{static_cast<uint32_t>(tmpl.rows_offset + begin * sizeof(CalendarExecutionRow)), limit - begin,
                      bindings, wait_offset, num_waits, sequence_base, 0, settle ? exit_offset : 0,
                      settle ? static_cast<uint32_t>(exit_tasks.size()) : 0,
                      shell < 0 ? UINT32_MAX : static_cast<uint32_t>(shell), begin - first,
                      pmu == 0 ? 0 : static_cast<uint32_t>(pmu + begin * sizeof(CalendarPmuCell))},
                     samples == 0 ? 0 : static_cast<uint32_t>(samples + begin * sizeof(CalendarTiming)),
                     tail ? static_cast<int32_t>(p) : -1}
                );
                begin = limit;
            }
        }
        image.phase_ends.push_back(end);
        if (longest == 0 || longest > UINT32_MAX - sequence_base) return false;
        sequence_base += longest;
    }

    uint32_t total_ranges = 0;
    bool sampled = false;
    for (const auto &worker : ranges) {
        if (worker.size() > UINT32_MAX - total_ranges) return false;
        total_ranges += static_cast<uint32_t>(worker.size());
        sampled |= std::any_of(worker.begin(), worker.end(), [](const auto &range) {
            return range.timing != 0;
        });
    }
    const uint32_t worker_offset = image.allocate<uint32_t>(workers + 1);
    const uint32_t ranges_offset = image.allocate<CalendarRange>(std::max(1U, total_ranges));
    const uint32_t sample_indexes = image.allocate<uint32_t>(sampled ? total_ranges : 0);
    const uint64_t slots = uint64_t{image.completion.slot_count} + image.phase_slot_count + image.fanin_slot_count;
    if (slots > UINT32_MAX) return false;
    const uint32_t notify_slots = image.allocate<uint64_t>(slots);
    if (worker_offset == 0 || ranges_offset == 0 || (sampled && sample_indexes == 0) ||
        (slots != 0 && notify_slots == 0))
        return false;
    uint32_t index = 0;
    for (uint32_t worker = 0; worker < workers; ++worker) {
        image.store(worker_offset + worker * sizeof(uint32_t), index);
        for (const auto &record : ranges[worker]) {
            const auto offset = static_cast<uint32_t>(ranges_offset + index * sizeof(CalendarRange));
            image.store(offset, record.range);
            if (sampled) image.store(sample_indexes + index * sizeof(uint32_t), record.timing);
            if (record.end_phase >= 0) image.tails.push_back({offset, static_cast<uint32_t>(record.end_phase)});
            ++index;
        }
    }
    image.store(worker_offset + workers * sizeof(uint32_t), index);
    image.store(
        0, CalendarRangeHeader{
               workers, worker_offset, ranges_offset, total_ranges, patch_offset, sample_indexes, notify_slots,
               static_cast<uint32_t>(slots), image.peer_endpoints_offset, image.peer_fanouts_offset
           }
    );
    *out = std::move(image);
    return true;
}

inline uint64_t range_add_command(const PercounterCounter &counter) {
    return a5_soc_sram_add_address(counter.completion_notify_address);
}

inline CalendarWait range_completion_wait(const PercounterCounter &counter, uint32_t task) {
    if (counter.calendar_defer_bit == 0)
        return {counter.completion_notify_address | CALENDAR_WAIT_NOTIFY, counter.completion_notify_target};
    const auto bit = static_cast<uint32_t>(__builtin_ctz(counter.calendar_defer_bit));
    return {
        counter.completion_notify_address | CALENDAR_WAIT_DEFER,
        (uint64_t{task} << 32) | (uint64_t{bit} << 24) | counter.completion_notify_target
    };
}

inline bool bind_range_calendar(
    RangeCalendarImage &image, uint64_t base, const PercounterLayout &layout, void *host,
    const std::vector<uint64_t> &notifications
) {
    const auto header = image.load<CalendarRangeHeader>(0);
    if (image.completion.tasks.size() != layout.task_count || notifications.size() != header.notify_slots_count)
        return false;
    for (size_t i = 0; i < notifications.size(); ++i) {
        if (a5_soc_sram_add_address(notifications[i]) == 0) return false;
        image.store(header.notify_slots_offset + static_cast<uint32_t>(i * sizeof(uint64_t)), notifications[i]);
    }
    auto *counters = region_at<PercounterCounter>(host, layout.counters_offset);
    for (size_t task_id = 0; task_id < image.completion.tasks.size(); ++task_id) {
        const auto &mapped = image.completion.tasks[task_id];
        if (mapped.slot == UINT32_MAX) continue;
        if (mapped.slot >= image.completion.slot_count) return false;
        auto &counter = counters[task_id];
        counter.completion_notify_address = notifications[mapped.slot];
        counter.completion_notify_target = mapped.target;
        counter.calendar_defer_bit = mapped.defer_bit;
    }
    auto *control = region_at<PercounterControl>(host, layout.control_offset);
    std::copy(image.sync_widths.begin(), image.sync_widths.end(), control->max_sync_blocks);
    std::map<uint32_t, uint64_t> ready_address;
    if (layout.graph_invocation_count != 0) {
        auto *records = region_at<PercounterGraphInvocation>(host, layout.graph_invocations_offset);
        std::map<uint32_t, PercounterGraphInvocation> by_shell;
        for (uint32_t i = 0; i < layout.graph_invocation_count; ++i)
            by_shell.emplace(static_cast<uint32_t>(records[i].shell_task_id), records[i]);
        if (by_shell.size() != image.graph_shells.size()) return false;
        for (size_t i = 0; i < image.graph_shells.size(); ++i) {
            const uint32_t shell = image.graph_shells[i];
            const auto record = by_shell.find(shell);
            if (record == by_shell.end()) return false;
            records[i] = record->second;
            ready_address.emplace(
                shell, base + layout.graph_invocations_offset + i * sizeof(PercounterGraphInvocation) +
                           offsetof(PercounterGraphInvocation, calendar_ready)
            );
        }
    }
    const size_t first_phase_slot = image.completion.slot_count;
    const size_t first_fanin_slot = first_phase_slot + image.phase_slot_count;
    for (const auto &group : image.fused_fanins) {
        const size_t slot = first_fanin_slot + group.slot;
        if (slot >= notifications.size()) return false;
        for (uint32_t member = 0; member < group.producer_count; ++member) {
            const uint32_t producer = group.producers[member];
            if (producer >= layout.task_count || counters[producer].completion_notify_address != 0) return false;
            auto &counter = counters[producer];
            counter.completion_notify_address = notifications[slot];
            counter.completion_notify_target = group.credits;
            counter.calendar_defer_bit = uint32_t{1} << (24 + member);
        }
    }
    std::vector<CalendarWait> phase_waits(image.phase_ends.size());
    std::vector<uint64_t> phase_add(image.phase_ends.size(), 0);
    for (size_t phase = 0; phase < image.phase_ends.size(); ++phase) {
        const auto &end = image.phase_ends[phase];
        if (end.task != UINT32_MAX) {
            const auto &counter = counters[end.task];
            if (counter.completion_notify_address == 0 || image.completion.tasks[end.task].slot == UINT32_MAX)
                return false;
            phase_waits[phase] = range_completion_wait(counter, end.task);
        } else if (end.slot != UINT32_MAX) {
            const uint64_t set = notifications[first_phase_slot + end.slot];
            phase_waits[phase] = {set | CALENDAR_WAIT_NOTIFY, end.contributors};
            phase_add[phase] = a5_soc_sram_add_address(set);
        }
    }
    for (const auto &patch : image.waits) {
        switch (patch.kind) {
        case RangeCalendarImage::WaitKind::COMPLETION:
        case RangeCalendarImage::WaitKind::FUSED_PRODUCER: {
            if (patch.index >= layout.task_count || counters[patch.index].completion_notify_address == 0) return false;
            const bool fused = image.completion.tasks[patch.index].slot == UINT32_MAX;
            if (fused != (patch.kind == RangeCalendarImage::WaitKind::FUSED_PRODUCER)) return false;
            image.store(patch.offset, range_completion_wait(counters[patch.index], patch.index));
            break;
        }
        case RangeCalendarImage::WaitKind::PHASE_END:
            if (patch.index >= phase_waits.size() || phase_waits[patch.index].read_cmd == 0) return false;
            image.store(patch.offset, phase_waits[patch.index]);
            break;
        case RangeCalendarImage::WaitKind::FANIN: {
            const size_t slot = first_fanin_slot + patch.index;
            if (slot >= notifications.size()) return false;
            image.store(patch.offset, CalendarWait{notifications[slot] | CALENDAR_WAIT_NOTIFY, patch.target});
            break;
        }
        case RangeCalendarImage::WaitKind::FUSED_FANIN: {
            if (patch.index >= image.fused_fanins.size()) return false;
            const auto &group = image.fused_fanins[patch.index];
            const size_t slot = first_fanin_slot + group.slot;
            if (slot >= notifications.size()) return false;
            const uint64_t target = (uint64_t{group.producer_count} << 32) | group.credits;
            image.store(patch.offset, CalendarWait{notifications[slot] | CALENDAR_WAIT_FUSED_FANIN, target});
            break;
        }
        case RangeCalendarImage::WaitKind::MATERIALIZED:
            if (const auto ready = ready_address.find(patch.index); ready != ready_address.end())
                image.store(patch.offset, CalendarWait{ready->second | CALENDAR_WAIT_GM_TARGET, patch.target});
            else return false;
            break;
        }
    }
    for (const auto &patch : image.dispatches) {
        auto record = image.load<CalendarDispatch>(patch.offset);
        const auto &counter = counters[patch.global_task];
        if (counter.completion_notify_address == 0) return false;
        record.done_add_cmd = range_add_command(counter);
        if (patch.shell < 0) {
            record.kernel_entry = counter.function_addresses[patch.role];
            record.args_address = counter.arguments_address;
        }
        image.store(patch.offset, record);
    }
    for (const auto &patch : image.extras) {
        auto extra = image.load<CalendarExecutionExtra>(patch.offset);
        if (patch.shell >= 0) extra.shell_add_cmd = range_add_command(counters[patch.shell]);
        else extra.predicate_address = counters[patch.global_task].predicate_address;
        image.store(patch.offset, extra);
    }
    for (const auto &tail : image.tails) {
        if (tail.phase >= phase_add.size()) return false;
        auto range = image.load<CalendarRange>(tail.range);
        range.tail_add_cmd = phase_add[tail.phase];
        image.store(tail.range, range);
    }
    return true;
}

}  // namespace simpler::percounter
