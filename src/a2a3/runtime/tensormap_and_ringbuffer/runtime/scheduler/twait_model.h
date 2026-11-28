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
/**
 * Bounded resource-cycle analysis over a caller-supplied blocking-wait
 * snapshot. Each record chains waiter -> event -> producer -> compatible
 * execution slots -> slot owner. The analyzer repeatedly eliminates tasks
 * that hold free capacity or an unblocked owner; a closed witness is returned
 * only when every compatible execution alternative of the remaining set is
 * held by the set itself. Incomplete or contradictory records yield
 * Diagnosis::Unknown rather than a fabricated cycle.
 */

#pragma once

namespace twait {

struct TaskFact {
    Identity identity{};
    bool resident{false};
    bool completed{false};
    bool complete{true};
    Event event{};
    uint32_t contributor_count{0};
    Identity contributors[max_producers]{};
    uint32_t alternative_count{0};
    CoreSlot alternatives[max_alternatives]{};
};

struct FiniteSnapshot {
    uint64_t run{0};
    bool complete{true};
    uint32_t task_count{0};
    TaskFact tasks[max_records]{};
    uint32_t slot_count{0};
    CoreSlot slots[max_records]{};
};

inline bool same_slot(const CoreSlot &left, const CoreSlot &right) {
    return left.run == right.run && left.rank == right.rank && left.lane == right.lane && left.core == right.core;
}

inline bool same_slot_fact(const CoreSlot &left, const CoreSlot &right) {
    return same_slot(left, right) && left.generation == right.generation && left.executing == right.executing &&
           left.occupied == right.occupied &&
           (!left.occupied || (left.owner == right.owner && left.token == right.token));
}

inline bool same_event(const Event &left, const Event &right) {
    return left.run == right.run && left.rank == right.rank && left.event == right.event && left.epoch == right.epoch &&
           left.condition == right.condition;
}

inline int32_t task_position(const FiniteSnapshot &snapshot, const Identity &identity) {
    for (uint32_t index = 0; index < snapshot.task_count; ++index)
        if (snapshot.tasks[index].identity == identity) return static_cast<int32_t>(index);
    return -1;
}

inline bool same_task_fact(const TaskFact &left, const TaskFact &right) {
    if (!(left.identity == right.identity) || left.resident != right.resident || left.completed != right.completed ||
        left.complete != right.complete || !same_event(left.event, right.event))
        return false;
    for (uint32_t side = 0; side < 2; ++side) {
        const auto &source = side == 0 ? left : right;
        const auto &target = side == 0 ? right : left;
        for (uint32_t index = 0; index < source.contributor_count; ++index) {
            bool found = false;
            for (uint32_t other = 0; other < target.contributor_count; ++other)
                found |= source.contributors[index] == target.contributors[other];
            if (!found) return false;
        }
        for (uint32_t index = 0; index < source.alternative_count; ++index) {
            bool found = false;
            for (uint32_t other = 0; other < target.alternative_count; ++other)
                found |= same_slot_fact(source.alternatives[index], target.alternatives[other]);
            if (!found) return false;
        }
    }
    return true;
}

inline bool normalize(const FiniteSnapshot &input, FiniteSnapshot &output) {
    if (!input.complete || input.run == 0 || input.task_count > max_records || input.slot_count == 0 ||
        input.slot_count > max_records)
        return false;
    output.run = input.run;
    for (uint32_t index = 0; index < input.task_count; ++index) {
        const auto &task = input.tasks[index];
        if (!task.complete || task.identity.run != input.run || task.contributor_count > max_producers ||
            task.alternative_count > max_alternatives)
            return false;
        const auto existing = task_position(output, task.identity);
        if (existing >= 0) {
            if (!same_task_fact(output.tasks[existing], task)) return false;
        } else {
            output.tasks[output.task_count++] = task;
        }
    }
    for (uint32_t index = 0; index < input.slot_count; ++index) {
        const auto &slot = input.slots[index];
        if (slot.run != input.run || slot.generation == 0 || !slot.executing) return false;
        bool found = false;
        for (uint32_t other = 0; other < output.slot_count; ++other) {
            if (!same_slot(output.slots[other], slot)) continue;
            if (!same_slot_fact(output.slots[other], slot)) return false;
            found = true;
        }
        if (!found) output.slots[output.slot_count++] = slot;
    }
    for (uint32_t index = 0; index < output.slot_count; ++index) {
        const auto &slot = output.slots[index];
        if (!slot.occupied) continue;
        const auto owner = task_position(output, slot.owner);
        if (owner < 0 || !output.tasks[owner].resident || output.tasks[owner].completed || slot.owner.rank != slot.rank)
            return false;
        for (uint32_t other = 0; other < index; ++other)
            if (output.slots[other].occupied && output.slots[other].owner == slot.owner) return false;
    }
    for (uint32_t index = 0; index < output.task_count; ++index) {
        const auto &task = output.tasks[index];
        if (task.resident) {
            if (task.completed || task.contributor_count == 0 || task.event.run != output.run ||
                task.event.rank != task.identity.rank || task.event.event == 0 || task.event.epoch == 0 ||
                task.event.condition == 0 || task.alternative_count != 0)
                return false;
            bool owned = false;
            for (uint32_t slot = 0; slot < output.slot_count; ++slot)
                owned |= output.slots[slot].occupied && output.slots[slot].owner == task.identity;
            if (!owned) return false;
            for (uint32_t other = 0; other < index; ++other)
                if (output.tasks[other].resident && same_event(task.event, output.tasks[other].event)) return false;
            for (uint32_t contributor = 0; contributor < task.contributor_count; ++contributor)
                if (task_position(output, task.contributors[contributor]) < 0) return false;
        } else {
            if (task.contributor_count != 0 || task.alternative_count == 0) return false;
            for (uint32_t alternative = 0; alternative < task.alternative_count; ++alternative) {
                const auto &slot = task.alternatives[alternative];
                if (slot.rank != task.identity.rank) return false;
                bool found = false;
                for (uint32_t physical = 0; physical < output.slot_count; ++physical)
                    found |= same_slot_fact(slot, output.slots[physical]);
                if (!found) return false;
            }
        }
    }
    return true;
}

inline Witness diagnose_finite(const FiniteSnapshot &input) {
    Witness result;
    FiniteSnapshot snapshot;
    if (!normalize(input, snapshot)) return result;
    bool completed[max_records]{};
    for (uint32_t index = 0; index < snapshot.task_count; ++index)
        completed[index] = snapshot.tasks[index].completed;
    bool changed = true;
    while (changed) {
        changed = false;
        for (uint32_t index = 0; index < snapshot.task_count; ++index) {
            if (completed[index]) continue;
            const auto &task = snapshot.tasks[index];
            bool enabled = task.resident;
            if (task.resident) {
                for (uint32_t contributor = 0; contributor < task.contributor_count; ++contributor)
                    enabled &= completed[task_position(snapshot, task.contributors[contributor])];
            } else {
                for (uint32_t alternative = 0; alternative < task.alternative_count; ++alternative) {
                    const auto &slot = task.alternatives[alternative];
                    enabled |= !slot.occupied || completed[task_position(snapshot, slot.owner)];
                }
            }
            if (enabled) {
                completed[index] = true;
                result.order[result.order_count++] = task.identity;
                changed = true;
            }
        }
    }
    bool all_completed = true;
    for (uint32_t index = 0; index < snapshot.task_count; ++index) {
        const auto &task = snapshot.tasks[index];
        if (completed[index]) result.terminal[result.terminal_count++] = task.identity;
        else {
            all_completed = false;
            if (task.resident) result.support[result.support_count++] = task.identity;
        }
    }
    if (all_completed) {
        result.diagnosis = Diagnosis::NoCycle;
        return result;
    }
    if (result.support_count == 0) return Witness{};
    for (uint32_t index = 0; index < snapshot.task_count; ++index) {
        const auto &task = snapshot.tasks[index];
        if (completed[index] || !task.resident) continue;
        for (uint32_t contributor = 0; contributor < task.contributor_count; ++contributor) {
            const auto producer = task_position(snapshot, task.contributors[contributor]);
            if (completed[producer]) continue;
            const auto &dependency = snapshot.tasks[producer];
            if (dependency.resident) {
                result.hops[result.count++] = {task.identity, task.event, dependency.identity, {}, false};
            } else {
                for (uint32_t alternative = 0; alternative < dependency.alternative_count; ++alternative) {
                    if (result.count >= max_hops) return Witness{};
                    result.hops[result.count++] = {
                        task.identity, task.event, dependency.identity, dependency.alternatives[alternative], true
                    };
                }
            }
            break;
        }
    }
    result.diagnosis = Diagnosis::Cycle;
    return result;
}

inline Witness diagnose(const Snapshot &input) {
    if (input.count == 0 || input.count > max_records) return Witness{};
    FiniteSnapshot snapshot;
    snapshot.run = input.records[0].waiter.run;
    for (uint32_t index = 0; index < input.count; ++index) {
        const auto &record = input.records[index];
        if (!record.complete || record.alternative_count > max_alternatives || record.waiter.run != snapshot.run ||
            record.producer.run != snapshot.run)
            return Witness{};
        auto waiter = task_position(snapshot, record.waiter);
        if (waiter < 0) {
            if (snapshot.task_count == max_records) return Witness{};
            waiter = static_cast<int32_t>(snapshot.task_count++);
            snapshot.tasks[waiter].identity = record.waiter;
            snapshot.tasks[waiter].resident = true;
            snapshot.tasks[waiter].event = record.event;
        }
        auto &task = snapshot.tasks[waiter];
        if (!same_event(task.event, record.event)) return Witness{};
        bool found = false;
        for (uint32_t contributor = 0; contributor < task.contributor_count; ++contributor)
            found |= task.contributors[contributor] == record.producer;
        if (!found) {
            if (task.contributor_count == max_producers) return Witness{};
            task.contributors[task.contributor_count++] = record.producer;
        }
    }
    for (uint32_t index = 0; index < input.count; ++index) {
        const auto &record = input.records[index];
        auto producer = task_position(snapshot, record.producer);
        if (producer >= 0 && snapshot.tasks[producer].resident) {
            if (record.producer_completed) return Witness{};
        } else {
            TaskFact fact;
            fact.identity = record.producer;
            fact.completed = record.producer_completed;
            fact.alternative_count = record.alternative_count;
            for (uint32_t alternative = 0; alternative < record.alternative_count; ++alternative)
                fact.alternatives[alternative] = record.alternatives[alternative];
            if (producer >= 0) {
                if (!same_task_fact(snapshot.tasks[producer], fact)) return Witness{};
            } else {
                if (snapshot.task_count == max_records) return Witness{};
                snapshot.tasks[snapshot.task_count++] = fact;
            }
        }
        for (uint32_t alternative = 0; alternative < record.alternative_count; ++alternative) {
            const auto &slot = record.alternatives[alternative];
            bool found = false;
            for (uint32_t other = 0; other < snapshot.slot_count; ++other) {
                if (!same_slot(snapshot.slots[other], slot)) continue;
                if (!same_slot_fact(snapshot.slots[other], slot)) return Witness{};
                found = true;
            }
            if (!found) {
                if (snapshot.slot_count == max_records) return Witness{};
                snapshot.slots[snapshot.slot_count++] = slot;
            }
        }
    }
    return diagnose_finite(snapshot);
}

}  // namespace twait
