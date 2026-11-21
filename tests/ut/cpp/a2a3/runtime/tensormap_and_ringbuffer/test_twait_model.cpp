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

#include <algorithm>
#include <numeric>
#include <queue>
#include <sstream>
#include <string>
#include <vector>
#include "scheduler/twait_contract.h"

using twait::CoreSlot;
using twait::Diagnosis;
using twait::FiniteSnapshot;
using twait::Identity;
using twait::TaskFact;
using twait::Witness;

static int position(const FiniteSnapshot &snapshot, const Identity &identity) {
    for (uint32_t index = 0; index < snapshot.task_count; ++index)
        if (snapshot.tasks[index].identity == identity) return static_cast<int>(index);
    return -1;
}

static bool enabled(const FiniteSnapshot &snapshot, int task_index, unsigned completed) {
    if ((completed & (1u << task_index)) != 0) return false;
    const auto &task = snapshot.tasks[task_index];
    if (task.resident) {
        for (uint32_t contributor = 0; contributor < task.contributor_count; ++contributor) {
            const int dependency = position(snapshot, task.contributors[contributor]);
            if (dependency < 0 || (completed & (1u << dependency)) == 0) return false;
        }
        return true;
    }
    for (uint32_t alternative = 0; alternative < task.alternative_count; ++alternative) {
        const auto &slot = task.alternatives[alternative];
        if (!slot.occupied) return true;
        const int owner = position(snapshot, slot.owner);
        if (owner >= 0 && (completed & (1u << owner)) != 0) return true;
    }
    return false;
}

static unsigned initial(const FiniteSnapshot &snapshot) {
    unsigned completed = 0;
    for (uint32_t index = 0; index < snapshot.task_count; ++index)
        if (snapshot.tasks[index].completed) completed |= 1u << index;
    return completed;
}

static Diagnosis reachable(const FiniteSnapshot &snapshot) {
    const unsigned target = (1u << snapshot.task_count) - 1;
    std::vector<bool> visited(1u << snapshot.task_count);
    std::queue<unsigned> frontier;
    frontier.push(initial(snapshot));
    while (!frontier.empty()) {
        const auto state = frontier.front();
        frontier.pop();
        if (state == target) return Diagnosis::NoCycle;
        if (visited[state]) continue;
        visited[state] = true;
        for (uint32_t task = 0; task < snapshot.task_count; ++task)
            if (enabled(snapshot, task, state)) frontier.push(state | (1u << task));
    }
    return Diagnosis::Cycle;
}

static bool contains(const Identity *identities, uint32_t count, const Identity &identity) {
    for (uint32_t index = 0; index < count; ++index)
        if (identities[index] == identity) return true;
    return false;
}

static bool check_certificate(const FiniteSnapshot &snapshot, const Witness &witness) {
    if (witness.diagnosis == Diagnosis::Unknown || witness.order_count > snapshot.task_count ||
        witness.terminal_count > snapshot.task_count || witness.support_count > snapshot.task_count ||
        witness.count > twait::max_hops)
        return false;
    unsigned completed = initial(snapshot);
    for (uint32_t step = 0; step < witness.order_count; ++step) {
        const int task = position(snapshot, witness.order[step]);
        if (task < 0 || !enabled(snapshot, task, completed)) return false;
        completed |= 1u << task;
    }
    for (uint32_t task = 0; task < snapshot.task_count; ++task) {
        const bool done = (completed & (1u << task)) != 0;
        if (contains(witness.terminal, witness.terminal_count, snapshot.tasks[task].identity) != done) return false;
    }
    if (witness.diagnosis == Diagnosis::NoCycle)
        return completed == (1u << snapshot.task_count) - 1 && witness.count == 0 && witness.support_count == 0;
    if (witness.support_count == 0 || witness.count == 0) return false;
    for (uint32_t task = 0; task < snapshot.task_count; ++task)
        if (enabled(snapshot, task, completed)) return false;
    for (uint32_t support = 0; support < witness.support_count; ++support) {
        const auto &identity = witness.support[support];
        const int task_index = position(snapshot, identity);
        if (task_index < 0 || !snapshot.tasks[task_index].resident || (completed & (1u << task_index))) return false;
        const auto &task = snapshot.tasks[task_index];
        int selected = -1;
        for (uint32_t edge = 0; edge < witness.count; ++edge) {
            const auto &hop = witness.hops[edge];
            if (!(hop.waiter == identity)) continue;
            if (!twait::same_event(hop.event, task.event)) return false;
            bool actual = false;
            for (uint32_t contributor = 0; contributor < task.contributor_count; ++contributor)
                actual |= task.contributors[contributor] == hop.producer;
            const int dependency = position(snapshot, hop.producer);
            if (!actual || dependency < 0 || (completed & (1u << dependency))) return false;
            if (selected >= 0 && selected != dependency) return false;
            selected = dependency;
            const auto &producer = snapshot.tasks[dependency];
            if (producer.resident) {
                if (hop.resource || !contains(witness.support, witness.support_count, producer.identity)) return false;
            } else {
                if (!hop.resource || !hop.core_slot.occupied ||
                    !contains(witness.support, witness.support_count, hop.core_slot.owner))
                    return false;
                bool alternative = false;
                for (uint32_t option = 0; option < producer.alternative_count; ++option)
                    alternative |= twait::same_slot_fact(hop.core_slot, producer.alternatives[option]);
                if (!alternative) return false;
            }
        }
        if (selected < 0) return false;
        const auto &producer = snapshot.tasks[selected];
        if (!producer.resident) {
            for (uint32_t option = 0; option < producer.alternative_count; ++option) {
                bool covered = false;
                for (uint32_t edge = 0; edge < witness.count; ++edge) {
                    const auto &hop = witness.hops[edge];
                    covered |= hop.waiter == identity && hop.producer == producer.identity && hop.resource &&
                               twait::same_slot_fact(hop.core_slot, producer.alternatives[option]);
                }
                if (!covered) return false;
            }
        }
    }
    for (uint32_t edge = 0; edge < witness.count; ++edge)
        if (!contains(witness.support, witness.support_count, witness.hops[edge].waiter)) return false;
    return true;
}

static std::string serialize(const FiniteSnapshot &snapshot) {
    std::ostringstream output;
    output << "finite-v1/run=" << snapshot.run << "/complete=" << snapshot.complete;
    for (uint32_t slot = 0; slot < snapshot.slot_count; ++slot) {
        const auto &fact = snapshot.slots[slot];
        output << "/S=" << fact.rank << ',' << fact.lane << ',' << fact.core << ',' << fact.generation << ','
               << fact.executing << ',' << fact.occupied << ',' << fact.owner.task << ',' << fact.owner.generation;
    }
    for (uint32_t index = 0; index < snapshot.task_count; ++index) {
        const auto &task = snapshot.tasks[index];
        output << "/T=" << task.identity.rank << ',' << task.identity.task << ',' << task.identity.generation << ','
               << task.resident << ',' << task.completed << ',' << task.complete << ',' << task.event.event << ":C";
        for (uint32_t contributor = 0; contributor < task.contributor_count; ++contributor)
            output << ',' << task.contributors[contributor].task;
        output << ":A";
        for (uint32_t alternative = 0; alternative < task.alternative_count; ++alternative)
            output << ',' << task.alternatives[alternative].rank << '.' << task.alternatives[alternative].core;
    }
    return output.str();
}

static uint64_t input_hash(const std::string &text) {
    uint64_t hash = 14695981039346656037ull;
    for (const auto character : text) {
        hash ^= static_cast<unsigned char>(character);
        hash *= 1099511628211ull;
    }
    return hash;
}

static uint64_t checked = 0;
static uint64_t digest = 0;

struct AbortEnumeration {};

static void require(bool condition, const FiniteSnapshot &snapshot, const char *label) {
    if (condition) return;
    const auto text = serialize(snapshot);
    ADD_FAILURE() << "counterexample " << label << " hash=" << input_hash(text) << ' ' << text;
    throw AbortEnumeration{};
}

static void valid(const FiniteSnapshot &snapshot, Diagnosis expected, bool certificate = true) {
    const auto witness = twait::diagnose_finite(snapshot);
    require(witness.diagnosis == expected, snapshot, "classification");
    if (certificate) require(check_certificate(snapshot, witness), snapshot, "certificate");
    ++checked;
}

static void unknown(const FiniteSnapshot &snapshot) {
    const auto witness = twait::diagnose_finite(snapshot);
    require(
        witness.diagnosis == Diagnosis::Unknown && witness.count == 0 && witness.order_count == 0 &&
            witness.terminal_count == 0 && witness.support_count == 0,
        snapshot, "invalid-definite"
    );
    ++checked;
}

static void reorder_and_invalid(const FiniteSnapshot &snapshot, Diagnosis expected) {
    auto changed = snapshot;
    std::reverse(changed.tasks, changed.tasks + changed.task_count);
    std::reverse(changed.slots, changed.slots + changed.slot_count);
    valid(changed, expected);
    changed = snapshot;
    if (changed.task_count > 0) std::rotate(changed.tasks, changed.tasks + 1, changed.tasks + changed.task_count);
    std::rotate(changed.slots, changed.slots + 1, changed.slots + changed.slot_count);
    valid(changed, expected);
    changed = snapshot;
    for (uint32_t task = 0; task < changed.task_count; ++task) {
        auto &fact = changed.tasks[task];
        std::reverse(fact.contributors, fact.contributors + fact.contributor_count);
        std::reverse(fact.alternatives, fact.alternatives + fact.alternative_count);
    }
    valid(changed, expected);
    changed = snapshot;
    changed.slots[changed.slot_count++] = changed.slots[0];
    if (changed.task_count > 0) changed.tasks[changed.task_count++] = changed.tasks[0];
    valid(changed, expected, false);
    changed = snapshot;
    changed.complete = false;
    unknown(changed);
    for (uint32_t task = 0; task < snapshot.task_count; ++task) {
        changed = snapshot;
        auto &fact = changed.tasks[task];
        fact.complete = false;
        if (fact.resident) --fact.contributor_count;
        else --fact.alternative_count;
        unknown(changed);
        changed = snapshot;
        ++changed.tasks[task].identity.run;
        unknown(changed);
    }
    for (uint32_t slot = 0; slot < snapshot.slot_count; ++slot) {
        changed = snapshot;
        ++changed.slots[slot].run;
        unknown(changed);
        changed = snapshot;
        auto conflict = changed.slots[slot];
        ++conflict.generation;
        changed.slots[changed.slot_count++] = conflict;
        unknown(changed);
        changed = snapshot;
        conflict = changed.slots[slot];
        conflict.occupied = !conflict.occupied;
        changed.slots[changed.slot_count++] = conflict;
        unknown(changed);
        if (!snapshot.slots[slot].occupied) continue;
        changed = snapshot;
        ++changed.slots[slot].owner.generation;
        unknown(changed);
        changed = snapshot;
        conflict = changed.slots[slot];
        ++conflict.owner.task;
        changed.slots[changed.slot_count++] = conflict;
        unknown(changed);
    }
}

static void relabel(
    FiniteSnapshot &snapshot, const std::vector<int> &waiters, const std::vector<int> &producers,
    const std::vector<int> &slots, bool ranks
) {
    auto rename = [&](Identity &identity) {
        if (identity.run == 0) return;
        if (identity.task < 100) identity.task = waiters[identity.task];
        else identity.task = 100 + producers[identity.task - 100];
        if (ranks) identity.rank ^= 1;
    };
    auto rename_slot = [&](CoreSlot &slot) {
        slot.core = slots[slot.core];
        if (ranks) slot.rank ^= 1;
        rename(slot.owner);
    };
    for (uint32_t slot = 0; slot < snapshot.slot_count; ++slot)
        rename_slot(snapshot.slots[slot]);
    for (uint32_t index = 0; index < snapshot.task_count; ++index) {
        auto &task = snapshot.tasks[index];
        rename(task.identity);
        if (ranks && task.resident) task.event.rank ^= 1;
        for (uint32_t contributor = 0; contributor < task.contributor_count; ++contributor)
            rename(task.contributors[contributor]);
        for (uint32_t alternative = 0; alternative < task.alternative_count; ++alternative)
            rename_slot(task.alternatives[alternative]);
    }
}

static void exercise(const FiniteSnapshot &snapshot, int resident_count, int producer_count) {
    const auto expected = reachable(snapshot);
    valid(snapshot, expected);
    digest ^= input_hash(serialize(snapshot));
    reorder_and_invalid(snapshot, expected);
    std::vector<int> waiters(resident_count), producers(producer_count), slots(snapshot.slot_count);
    std::iota(waiters.begin(), waiters.end(), 0);
    do {
        std::iota(producers.begin(), producers.end(), 0);
        do {
            std::iota(slots.begin(), slots.end(), 0);
            do {
                for (int rank_swap = 0; rank_swap < 2; ++rank_swap) {
                    auto transformed = snapshot;
                    relabel(transformed, waiters, producers, slots, rank_swap != 0);
                    valid(transformed, expected);
                }
            } while (std::next_permutation(slots.begin(), slots.end()));
        } while (std::next_permutation(producers.begin(), producers.end()));
    } while (std::next_permutation(waiters.begin(), waiters.end()));
}

static void contributors(FiniteSnapshot &snapshot, int resident_count, int producer_count, int current) {
    if (current == resident_count) {
        exercise(snapshot, resident_count, producer_count);
        return;
    }
    for (unsigned mask = 1; mask < (1u << snapshot.task_count); ++mask) {
        if (__builtin_popcount(mask) > 2) continue;
        auto &task = snapshot.tasks[current];
        task.contributor_count = 0;
        for (uint32_t identity = 0; identity < snapshot.task_count; ++identity)
            if (mask & (1u << identity))
                task.contributors[task.contributor_count++] = snapshot.tasks[identity].identity;
        contributors(snapshot, resident_count, producer_count, current + 1);
    }
}

static void alternatives(FiniteSnapshot &snapshot, int resident_count, int producer_count, int current) {
    if (current == producer_count) {
        for (unsigned completed = 0; completed < (1u << producer_count); ++completed) {
            for (int producer = 0; producer < producer_count; ++producer)
                snapshot.tasks[resident_count + producer].completed = (completed & (1u << producer)) != 0;
            contributors(snapshot, resident_count, producer_count, 0);
        }
        return;
    }
    auto &task = snapshot.tasks[resident_count + current];
    for (unsigned mask = 1; mask < (1u << snapshot.slot_count); ++mask) {
        const auto home = snapshot.slots[__builtin_ctz(mask)].rank;
        bool compatible = true;
        task.alternative_count = 0;
        for (uint32_t slot = 0; slot < snapshot.slot_count; ++slot) {
            if (!(mask & (1u << slot))) continue;
            compatible &= snapshot.slots[slot].rank == home;
            task.alternatives[task.alternative_count++] = snapshot.slots[slot];
        }
        if (!compatible) continue;
        task.identity.rank = home;
        alternatives(snapshot, resident_count, producer_count, current + 1);
    }
}

static void enumerate() {
    uint64_t canonical = 0;
    for (int slot_count = 1; slot_count <= 2; ++slot_count) {
        for (int resident_count = 0; resident_count <= slot_count; ++resident_count) {
            for (int producer_count = 0; producer_count <= 2; ++producer_count) {
                for (unsigned ranks = 0; ranks < (1u << slot_count); ++ranks) {
                    std::vector<int> owners(slot_count);
                    std::iota(owners.begin(), owners.end(), 0);
                    do {
                        if (resident_count == 0 && owners[0] != 0) continue;
                        FiniteSnapshot snapshot;
                        snapshot.run = 1;
                        snapshot.slot_count = slot_count;
                        snapshot.task_count = resident_count + producer_count;
                        for (int slot = 0; slot < slot_count; ++slot)
                            snapshot.slots[slot] = {1,
                                                    (ranks >> slot) & 1u,
                                                    static_cast<uint64_t>(7 + slot),
                                                    0,
                                                    static_cast<uint32_t>(slot),
                                                    0,
                                                    true,
                                                    false,
                                                    {}};
                        for (int task = 0; task < resident_count; ++task) {
                            auto &fact = snapshot.tasks[task];
                            auto &slot = snapshot.slots[owners[task]];
                            fact.identity = {
                                1, slot.rank, static_cast<uint64_t>(11 + task), static_cast<uint64_t>(task), 0
                            };
                            fact.resident = true;
                            fact.event = {1, slot.rank, static_cast<uint64_t>(21 + task), 31, 1};
                            slot.occupied = true;
                            slot.owner = fact.identity;
                        }
                        for (int producer = 0; producer < producer_count; ++producer)
                            snapshot.tasks[resident_count + producer].identity = {
                                1, 0, static_cast<uint64_t>(41 + producer), static_cast<uint64_t>(100 + producer), 0
                            };
                        const auto before = checked;
                        alternatives(snapshot, resident_count, producer_count, 0);
                        canonical += checked != before;
                    } while (std::next_permutation(owners.begin(), owners.end()));
                }
            }
        }
    }
    EXPECT_GT(checked, 0U);
    EXPECT_GT(canonical, 0U);
}

static FiniteSnapshot seed() {
    FiniteSnapshot snapshot;
    snapshot.run = 1;
    snapshot.task_count = 3;
    snapshot.slot_count = 2;
    snapshot.tasks[0].identity = {1, 0, 11, 0, 0};
    snapshot.tasks[0].resident = true;
    snapshot.tasks[0].event = {1, 0, 21, 31, 1};
    snapshot.tasks[0].contributor_count = 2;
    snapshot.tasks[1].identity = {1, 0, 41, 100, 0};
    snapshot.tasks[2].identity = {1, 0, 42, 101, 0};
    snapshot.tasks[0].contributors[0] = snapshot.tasks[1].identity;
    snapshot.tasks[0].contributors[1] = snapshot.tasks[2].identity;
    snapshot.slots[0] = {1, 0, 7, 0, 0, 0, true, true, snapshot.tasks[0].identity};
    snapshot.slots[1] = {1, 0, 8, 0, 1, 0, true, false, {}};
    for (int producer = 1; producer <= 2; ++producer) {
        snapshot.tasks[producer].alternative_count = 1;
        snapshot.tasks[producer].alternatives[0] = snapshot.slots[0];
    }
    return snapshot;
}

static void snapshot_controls() {
    auto snapshot = seed();
    valid(snapshot, Diagnosis::Cycle);
    reorder_and_invalid(snapshot, Diagnosis::Cycle);
    snapshot.tasks[1].alternative_count = 2;
    snapshot.tasks[1].alternatives[1] = snapshot.slots[1];
    valid(snapshot, Diagnosis::Cycle);
    snapshot.tasks[2].completed = true;
    valid(snapshot, Diagnosis::NoCycle);
    auto witness = twait::diagnose_finite(snapshot);
    require(check_certificate(snapshot, witness), snapshot, "positive-checker");
    witness.order[0] = snapshot.tasks[0].identity;
    require(!check_certificate(snapshot, witness), snapshot, "bad-order-knockout");
    snapshot = seed();
    snapshot.slots[1].occupied = true;
    snapshot.tasks[0].contributor_count = 1;
    snapshot.task_count = 3;
    snapshot.tasks[2].resident = true;
    snapshot.tasks[2].identity = {1, 0, 12, 1, 0};
    snapshot.tasks[2].completed = false;
    snapshot.tasks[2].event = {1, 0, 22, 31, 1};
    snapshot.tasks[2].contributor_count = 1;
    snapshot.tasks[2].contributors[0] = snapshot.tasks[1].identity;
    snapshot.tasks[2].alternative_count = 0;
    snapshot.slots[1].owner = snapshot.tasks[2].identity;
    snapshot.tasks[1].alternative_count = 2;
    snapshot.tasks[1].alternatives[0] = snapshot.slots[0];
    snapshot.tasks[1].alternatives[1] = snapshot.slots[1];
    valid(snapshot, Diagnosis::Cycle);
    witness = twait::diagnose_finite(snapshot);
    --witness.count;
    require(!check_certificate(snapshot, witness), snapshot, "first-alternative-knockout");
    auto pending = snapshot;
    pending.slots[0].executing = false;
    unknown(pending);
    FiniteSnapshot ring;
    ring.run = 1;
    ring.slot_count = ring.task_count = 3;
    for (int task = 0; task < 3; ++task) {
        auto &fact = ring.tasks[task];
        fact.identity = {
            1, static_cast<uint32_t>(task % 2), static_cast<uint64_t>(11 + task), static_cast<uint64_t>(task), 0
        };
        fact.resident = true;
        fact.event = {1, fact.identity.rank, static_cast<uint64_t>(21 + task), 31, 1};
        fact.contributor_count = 1;
        ring.slots[task] = {1, fact.identity.rank, 7, 0, static_cast<uint32_t>(task), 0, true, true, fact.identity};
    }
    for (int task = 0; task < 3; ++task)
        ring.tasks[task].contributors[0] = ring.tasks[(task + 1) % 3].identity;
    valid(ring, Diagnosis::Cycle);
}

TEST(TwaitModel, SnapshotIntegrityAndCertificateControls) {
    try {
        snapshot_controls();
    } catch (const AbortEnumeration &) {}
}

TEST(TwaitModel, EnumerationMatchesBruteForceReachability) {
    try {
        enumerate();
    } catch (const AbortEnumeration &) {}
}
