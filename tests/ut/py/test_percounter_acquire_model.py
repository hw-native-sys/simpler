#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Bounded interleaving model: claims, delayed publication, entry and reuse.

A block's participants are fixed by an input allocation; each owner claims
ordinary and MIX tasks in one rank order into X generic slots. SSBUF is the only receiver payload.
This is a sequentially consistent protocol model, not a weak-memory proof.
"""

from collections import deque
from dataclasses import dataclass
from itertools import combinations, permutations

import pytest


@dataclass(frozen=True)
class Block:
    task: int
    owner: int
    mask: int
    deps: int = 0
    mix: bool = False
    sync: bool = False


@dataclass(frozen=True)
class State:
    slots: tuple
    published: int
    publication_order: tuple
    observed: tuple
    ready: int
    entered: tuple
    ran: tuple
    finished: tuple
    done: int
    retired: int


def explore(blocks, lanes=3, x=2, merged=True, gate=True, rank_first=True, limit=500000):  # noqa: PLR0912, PLR0915
    n = len(blocks)
    widths = (x,) * lanes
    starts = tuple(sum(widths[:c]) for c in range(lanes))
    sources = {}
    for b, block in enumerate(blocks):
        sources.setdefault((block.owner, False if merged else block.mix), []).append(b)
    initial = State((-1,) * sum(widths), 0, (0,) * n, (0,) * n, 0, (0,) * n, (0,) * n, (0,) * n, 0, 0)
    parents: dict[State, tuple[State, str] | None] = {initial: None}
    queue = deque([initial])
    terminals = 0
    all_done = (1 << n) - 1

    def changed(s, **fields):
        return State(**(s.__dict__ | fields))

    def update(values, b, mask):
        return values[:b] + (values[b] | mask,) + values[b + 1 :]

    def complete_tasks(s):
        result = 0
        for b, block in enumerate(blocks):
            if all(s.done & (1 << k) for k, other in enumerate(blocks) if other.task == block.task):
                result |= 1 << block.task
        return result

    def shared_head(s, lane):
        visible = [
            k for k, block in enumerate(blocks) if block.mix and s.observed[k] & (1 << lane) and not s.done & (1 << k)
        ]
        return min(
            visible,
            key=lambda k: (blocks[k].task if rank_first else 0, s.publication_order[k]),
            default=-1,
        )

    while queue:
        s = queue.popleft()
        if s.retired == all_done:
            terminals += 1
            continue
        owned = [b for b in s.slots if b >= 0]
        assert len(owned) == len(set(owned)), "a block was claimed twice"
        for b, block in enumerate(blocks):
            bit = 1 << b
            assert s.observed[b] & ~block.mask == 0
            assert s.entered[b] & ~s.observed[b] == 0
            assert s.ran[b] & ~block.mask == 0
            assert s.finished[b] & ~s.ran[b] == 0
            assert not s.finished[b] or s.ready & bit
            if s.observed[b] & ~s.finished[b]:
                assert b in owned and not s.retired & bit, "an unfinished participant lost its immutable record"
            if s.done & bit:
                assert s.finished[b] == block.mask
            if s.ran[b] and (block.mix or block.sync):
                assert all(
                    not (other.mask & s.ran[b]) or s.done & (1 << k)
                    for k, other in enumerate(blocks)
                    if (other.mix or other.sync) and other.task < block.task
                )
            if b in owned:
                slot = s.slots.index(b) - starts[block.owner]
                allowed = range(x)
                assert slot in allowed, "owner exceeded its fixed slot capacity"
        done_tasks = complete_tasks(s)
        moves = []
        claimed = {b for b in s.slots if b >= 0} | {b for b in range(n) if s.retired & (1 << b)}
        for (owner, _), sequence in sources.items():
            # One owner finishes publishing a claim before making its next claim.
            if any(blocks[k].owner == owner and not s.published & (1 << k) for k in owned):
                continue
            held_tasks = {blocks[k].task for k in s.slots if k >= 0 and blocks[k].owner == owner}
            unclaimed = [b for b in sequence if b not in claimed and blocks[b].task not in held_tasks]
            if not unclaimed:
                continue
            b = unclaimed[0]
            block = blocks[b]
            if any(k >= 0 and blocks[k].owner == owner and blocks[k].task == block.task for k in s.slots):
                continue
            # A sync task can occupy at most one slot per physical owner.
            if block.sync and any(blocks[k].task == block.task for k in claimed):
                if any(blocks[k].owner == owner and blocks[k].task == block.task for k in claimed):
                    raise AssertionError("invalid sync allocation")
            allowed = range(x)
            for j in allowed:
                slot = starts[owner] + j
                if s.slots[slot] < 0:
                    slots = list(s.slots)
                    slots[slot] = b
                    moves.append((changed(s, slots=tuple(slots)), f"claim {b} owner {owner} slot {j}"))
        for b in sorted(claimed):
            block = blocks[b]
            bit = 1 << b
            if s.retired & bit:
                continue
            if not s.published & bit:
                order = list(s.publication_order)
                if block.mix:
                    order[b] = 1 + max(order[k] for k, other in enumerate(blocks) if other.owner == block.owner)
                moves.append((changed(s, published=s.published | bit, publication_order=tuple(order)), f"publish {b}"))
                continue
            if s.done & bit:
                slots = tuple(-1 if value == b else value for value in s.slots)
                moves.append((changed(s, slots=slots, retired=s.retired | bit), f"retire {b}"))
                continue
            leader = (block.mask & -block.mask).bit_length() - 1
            cohort = block.mix or block.sync
            for lane in range(lanes):
                lane_bit = 1 << lane
                if not block.mask & lane_bit:
                    continue
                if not s.observed[b] & lane_bit:
                    moves.append((changed(s, observed=update(s.observed, b, lane_bit)), f"observe {b} lane {lane}"))
                    continue
                if block.mix and shared_head(s, lane) != b:
                    continue
                if lane == leader and not s.ready & bit and done_tasks & block.deps == block.deps:
                    moves.append((changed(s, ready=s.ready | bit), f"fanin {b}"))
                if not s.ready & bit:
                    continue
                if cohort and not s.entered[b] & lane_bit:
                    # Includes unknown future work, modeling the published source boundary.
                    earlier = any(
                        (o.mix or o.sync)
                        and o.mask & lane_bit
                        and (o.task, k) < (block.task, b)
                        and not s.done & (1 << k)
                        for k, o in enumerate(blocks)
                    )
                    committed = any(s.entered[k] & lane_bit and not s.done & (1 << k) for k in range(n) if k != b)
                    if not earlier and not committed:
                        moves.append(
                            (changed(s, entered=update(s.entered, b, lane_bit)), f"enter-ready {b} lane {lane}")
                        )
                eligible = not cohort or s.entered[b] & lane_bit
                if gate and cohort:
                    eligible = eligible and s.entered[b] == block.mask
                    if block.sync:
                        eligible = eligible and all(
                            s.entered[k] == o.mask for k, o in enumerate(blocks) if o.task == block.task
                        )
                busy = any(s.ran[k] & lane_bit and not s.finished[k] & lane_bit for k in range(n))
                if eligible and not busy and not s.ran[b] & lane_bit:
                    moves.append((changed(s, ran=update(s.ran, b, lane_bit)), f"run {b} lane {lane}"))
                # Kernels may wait for every participant; finishing requires that all entered kernel.
                if s.ran[b] & lane_bit and not s.finished[b] & lane_bit and s.ran[b] == block.mask:
                    if not block.sync or all(s.ran[k] == o.mask for k, o in enumerate(blocks) if o.task == block.task):
                        moves.append((changed(s, finished=update(s.finished, b, lane_bit)), f"finish {b} lane {lane}"))
            if s.finished[b] == block.mask and (not block.mix or shared_head(s, leader) == b):
                moves.append((changed(s, done=s.done | bit), f"aggregate {b}"))
        if not moves:
            trace = []
            cursor = s
            while (parent := parents[cursor]) is not None:
                cursor, event = parent
                trace.append(event)
            return {"states": len(parents), "deadlock": list(reversed(trace)), "terminals": terminals}
        for nxt, event in moves:
            if nxt not in parents:
                parents[nxt] = (s, event)
                queue.append(nxt)
                if len(parents) > limit:
                    raise RuntimeError(f"state bound exceeded: {limit}")
    return {"states": len(parents), "deadlock": None, "terminals": terminals}


CASES = {
    "ordinary_full": (Block(0, 1, 7, mix=True), Block(1, 1, 2, 1), Block(2, 1, 2, 1)),
    "mix_full": (Block(0, 1, 2), Block(1, 1, 7, 1, True), Block(2, 1, 7, 1, True)),
    "reuse_101": (Block(0, 1, 5, mix=True), Block(1, 1, 5, 1, True)),
    "two_vector": (Block(0, 1, 6, mix=True), Block(1, 0, 1, 1)),
    "mix_spmd": (Block(0, 1, 7, mix=True), Block(0, 1, 7, mix=True)),
    "mix_revisit_successor": (
        Block(0, 1, 7, mix=True),
        Block(0, 1, 7, mix=True),
        Block(1, 1, 7, 1, mix=True),
    ),
    "pure_sync_then_mix": (Block(0, 0, 1, sync=True), Block(0, 3, 8, sync=True), Block(1, 1, 7, 1, True)),
    "mixed_sync": (Block(0, 1, 7, mix=True, sync=True), Block(0, 4, 56, mix=True, sync=True)),
    "sync_and_ordinary": (Block(0, 0, 1, sync=True), Block(0, 3, 8, sync=True), Block(1, 1, 2)),
}


@pytest.mark.parametrize("x", [1, 2, 4])
@pytest.mark.parametrize("name", CASES)
def test_bounded_claim_publish_enter_retire(name, x):
    blocks = CASES[name]
    lanes = 6 if any(b.owner >= 3 for b in blocks) else 3
    result = explore(blocks, lanes=lanes, x=x)
    assert result["deadlock"] is None, result
    assert result["terminals"] > 0


@pytest.mark.parametrize("x", [2, 4])
def test_sequence_only_shared_head_blocks_retirement_revisit(x):
    result = explore(CASES["mix_revisit_successor"], x=x, rank_first=False)
    assert result["deadlock"] is not None
    trace = result["deadlock"]
    assert trace.index("publish 2") < trace.index("publish 1"), trace
    assert trace.index("retire 0") < trace.index("publish 1"), trace


@pytest.mark.parametrize("name", ["ordinary_full", "mix_full"])
def test_independent_source_capacity_negative_controls(name):
    assert explore(CASES[name], merged=False)["deadlock"]


def boundary_witness(publish_first=True, acquire_first=True):
    writer = ("claim", "publish", "floor") if publish_first else ("claim", "floor", "publish")
    reader = ("acquire", "discover", "commit") if acquire_first else ("discover", "acquire", "commit")
    for events in permutations(writer + reader):
        if tuple(e for e in events if e in writer) != writer:
            continue
        if tuple(e for e in events if e in reader) != reader:
            continue
        published = floor = observed = boundary = False
        for event in events:
            if event == "publish":
                published = True
            elif event == "floor":
                floor = True
            elif event == "acquire":
                boundary = floor
            elif event == "discover":
                observed = published
            elif event == "commit" and boundary and not observed:
                return events
    return None


def test_future_cohort_boundary_covers_claim_to_publication_gap():
    assert boundary_witness() is None
    assert boundary_witness(publish_first=False) is not None
    assert boundary_witness(acquire_first=False) is not None


def notification_snapshot_witness(sequence_ceiling=True):
    # A and B share a task rank, but A was notified first. The reader can
    # scan A before either publication and B after both. A common acquired
    # ceiling prevents that torn scan from making B the first commitment.
    writer = ("record_a", "publish_a", "record_b", "publish_b")
    reader = ("acquire", "scan_a", "scan_b", "commit")
    for events in permutations(writer + reader):
        if tuple(e for e in events if e in writer) != writer or tuple(e for e in events if e in reader) != reader:
            continue
        sequence = ceiling = 0
        records = [0, 0]
        observed = [False, False]
        for event in events:
            if event == "record_a":
                records[0] = 1
            elif event == "publish_a":
                sequence = 1
            elif event == "record_b":
                records[1] = 2
            elif event == "publish_b":
                sequence = 2
            elif event == "acquire":
                ceiling = sequence
            elif event in ("scan_a", "scan_b"):
                index = event == "scan_b"
                observed[index] = records[index] > 0 and (not sequence_ceiling or records[index] <= ceiling)
            elif event == "commit" and observed[1] and not observed[0]:
                return events
    return None


def test_same_task_notifications_use_one_acquired_sequence_ceiling():
    assert notification_snapshot_witness() is None
    assert notification_snapshot_witness(sequence_ceiling=False) is not None


def reuse_witness(invalidate_first=True):
    writer = ("invalidate", "payload", "publish") if invalidate_first else ("payload", "invalidate", "publish")
    reader = ("open", "read", "close")
    for events in permutations(writer + reader):
        if tuple(e for e in events if e in writer) != writer or tuple(e for e in events if e in reader) != reader:
            continue
        generation = payload = 1
        opening = value = 0
        for event in events:
            if event == "invalidate":
                generation = 2
            elif event == "payload":
                payload = 3
            elif event == "publish":
                generation = 3
            elif event == "open":
                opening = generation
            elif event == "read":
                value = payload
            elif event == "close" and opening == generation and opening % 2 and value != opening:
                return events
    return None


def test_shared_payload_snapshot_rejects_reuse_overlap():
    assert reuse_witness() is None
    assert reuse_witness(invalidate_first=False) is not None


# Four physical lanes: C0, V0, C1, V1. Every task is already claimed.
LANES = ("C0", "V0", "C1", "V1")
NAMES = ("Q", "P", "S_sync_aic", "M_mix", "T_sync_aiv")
MASKS = (0b0001, 0b0010, 0b0101, 0b1100, 0b1010)
COHORT = (False, False, True, True, True)
ALL_DONE = (1 << len(NAMES)) - 1


EntryState = tuple[int, tuple[int, ...], tuple[int, ...]]


def explore_entry(deps, gate):  # noqa: PLR0912
    start: EntryState = (0, (-1,) * len(LANES), (0,) * len(NAMES))
    todo = deque([start])
    paths: dict[EntryState, EntryState | None] = {start: None}
    edges: dict[EntryState, str] = {}
    deadlocks = []
    terminals = 0
    while todo:
        state = todo.popleft()
        done, busy, armed = state
        if done == ALL_DONE:
            terminals += 1
            continue
        moves = []
        for t, mask in enumerate(MASKS):
            if COHORT[t] and all(busy[c] == t for c in range(4) if mask & (1 << c)):
                next_busy = tuple(-1 if v == t else v for v in busy)
                moves.append(((done | (1 << t), next_busy, armed), f"finish {NAMES[t]}"))
        for c in range(4):
            if busy[c] >= 0:
                continue
            for t, mask in enumerate(MASKS):
                if done & (1 << t) or not mask & (1 << c) or done & deps[t] != deps[t]:
                    continue
                if COHORT[t] and any(COHORT[u] and MASKS[u] & (1 << c) and not done & (1 << u) for u in range(t)):
                    continue
                if not COHORT[t]:
                    moves.append(((done | (1 << t), busy, armed), f"{LANES[c]} runs {NAMES[t]}"))
                elif gate and not armed[t] & (1 << c):
                    next_armed = list(armed)
                    next_armed[t] |= 1 << c
                    moves.append(((done, busy, tuple(next_armed)), f"{LANES[c]} arms {NAMES[t]}"))
                elif not gate or armed[t] == mask:
                    next_busy = list(busy)
                    next_busy[c] = t
                    moves.append(((done, tuple(next_busy), armed), f"{LANES[c]} enters {NAMES[t]}"))
                # A lane considers its lowest ready eligible task first. Arming
                # is not entry: a waiting lane can still run ordinary tasks.
                if not COHORT[t] or not gate or armed[t] == mask:
                    break
        if not moves:
            deadlocks.append(state)
        for nxt, event in moves:
            if nxt not in paths:
                paths[nxt] = state
                edges[nxt] = event
                todo.append(nxt)
    witness = []
    if deadlocks:
        s = deadlocks[0]
        while (parent := paths[s]) is not None:
            witness.append(edges[s])
            s = parent
        witness.reverse()
    return {"states": len(paths), "deadlocks": len(deadlocks), "terminal_states": terminals, "witness": witness}


def test_entry_commitment_negative_control_and_all_rank_consistent_dags():
    deps = (0, 1 << 0, 1 << 1, 1 << 2, 0)
    assert explore_entry(deps, False)["deadlocks"] > 0
    edges = [(p, t) for t in range(5) for p in range(t)]
    for choice in range(1 << len(edges)):
        deps = [0] * 5
        for bit, (producer, consumer) in enumerate(edges):
            if choice & (1 << bit):
                deps[consumer] |= 1 << producer
        result = explore_entry(tuple(deps), True)
        assert result["deadlocks"] == 0, (deps, result)
        assert result["terminal_states"] > 0


def explore_ordinary_tasks(deps, blocks, lanes, x):
    """Private task counters, ordered claims and held-task skip with later revisit."""
    entries = tuple(task for task, count in enumerate(blocks) for _ in range(count))
    starts = tuple(sum(blocks[:t]) for t in range(len(blocks)))
    task_masks = tuple(sum(1 << i for i, task in enumerate(entries) if task == t) for t in range(len(blocks)))
    initial = ((0,) * len(blocks), (-1,) * (lanes * x), 0)
    seen = {initial}
    todo = [initial]
    terminal = (1 << len(entries)) - 1
    while todo:
        tickets, owned, done = todo.pop()
        if done == terminal:
            continue
        live = [entry for entry in owned if entry >= 0]
        assert len(live) == len(set(live))
        assert all(not (done & (1 << entry)) for entry in live)
        assert sum(tickets) == len(live) + bin(done).count("1")
        ready_tasks = tuple(
            all(not (deps[t] & (1 << p)) or done & task_masks[p] == task_masks[p] for p in range(len(blocks)))
            for t in range(len(blocks))
        )
        moves = []
        for lane in range(lanes):
            positions = range(lane * x, (lane + 1) * x)
            ready = [i for i in positions if owned[i] >= 0 and ready_tasks[entries[owned[i]]]]
            if ready:
                pos = min(ready, key=lambda i: entries[owned[i]])
                slots = list(owned)
                slots[pos] = -1
                moves.append((tickets, tuple(slots), done | (1 << owned[pos])))
                continue
            free = next((i for i in positions if owned[i] < 0), None)
            if free is None:
                continue
            held = {entries[owned[i]] for i in positions if owned[i] >= 0}
            task = next((t for t, count in enumerate(blocks) if tickets[t] < count and t not in held), None)
            if task is None:
                continue
            entry = starts[task] + tickets[task]
            advanced = list(tickets)
            advanced[task] += 1
            slots = list(owned)
            slots[free] = entry
            moves.append((tuple(advanced), tuple(slots), done))
        assert moves, (deps, blocks, lanes, x, tickets, owned, done)
        for nxt in moves:
            if nxt not in seen:
                seen.add(nxt)
                todo.append(nxt)
    return len(seen)


@pytest.mark.parametrize("x", [1, 2, 4])
def test_held_task_skip_and_revisit_preserve_ordinary_progress(x):
    edges = [(p, t) for t in range(3) for p in range(t)]
    for choice in range(1 << len(edges)):
        deps = [0] * 3
        for bit, (producer, consumer) in enumerate(edges):
            if choice & (1 << bit):
                deps[consumer] |= 1 << producer
        for blocks in [(1, 3, 1), (2, 1, 2), (1, 1, 1)]:
            assert explore_ordinary_tasks(tuple(deps), blocks, 2, x) > 1


@pytest.mark.parametrize("x", [1, 2, 4])
def test_partial_sync_covers_each_small_topology_allocation(x):
    # Every choice of two clusters in a three-cluster pool. The unused
    # cluster supplies their ordinary producer, so it must remain live.
    for clusters in combinations(range(3), 2):
        remaining = next(cluster for cluster in range(3) if cluster not in clusters)
        blocks = [Block(0, remaining * 3, 1 << (remaining * 3))]
        blocks += [Block(1, cluster * 3 + 1, 7 << (cluster * 3), 1, True, True) for cluster in clusters]
        result = explore(tuple(blocks), lanes=9, x=x)
        assert result["deadlock"] is None, (clusters, result)
        assert result["terminals"] > 0

    # Every partial two-block placement over four physical vector lanes.
    for lanes in combinations([1, 2, 4, 5], 2):
        blocks = (Block(0, 0, 1), *(Block(1, lane, 1 << lane, 1, sync=True) for lane in lanes))
        result = explore(blocks, lanes=6, x=x)
        assert result["deadlock"] is None, (lanes, result)
        assert result["terminals"] > 0


def explore_mix_idle(groups, *, idle_handshake, ordinary_needed=False, stop_at_entry=False):
    """Keep an ordinary kernel in flight across other lanes' entry transitions.

    Entry readiness is global for a multi-block sync task; UB ownership is
    cluster-local. A peer blocked by a lower cohort can need the ordinary
    kernel's result before it can commit to this cohort.
    """
    participants = sum(groups)
    lanes = [i for i in range(participants.bit_length()) if participants & (1 << i)]
    owner, gated = lanes[0], lanes[-1]
    owner_group = next(group for group in groups if group & (1 << owner))
    initial = (0, 0, 0, 0)  # entered, idle, finished MIX participants, ordinary state
    seen, todo = {initial}, [initial]
    unsafe = deadlocks = terminals = 0
    while todo:
        entered, idle, finished, ordinary = todo.pop()
        if finished == participants and ordinary == 2:
            terminals += 1
            continue
        moves = []
        if (
            ordinary == 0
            and (finished & (1 << owner) or entered != participants)
            and (not stop_at_entry or not entered & (1 << owner) or finished & (1 << owner))
        ):
            moves.append((entered, idle, finished, 1))
        if ordinary == 1:
            moves.append((entered, idle, finished, 2))
        for lane in lanes:
            bit = 1 << lane
            if lane == owner and ordinary == 1:
                continue
            if not entered & bit:
                if not ordinary_needed or lane != gated or ordinary == 2:
                    moves.append((entered | bit, idle, finished, ordinary))
                continue
            if entered != participants:
                continue
            if idle_handshake and not idle & bit:
                moves.append((entered, idle | bit, finished, ordinary))
                continue
            group = next(group for group in groups if group & bit)
            if finished & bit or (idle_handshake and idle & group != group):
                continue
            if ordinary == 1 and group == owner_group and not finished & (1 << owner):
                unsafe += 1
            moves.append((entered, idle, finished | bit, ordinary))
        if not moves:
            deadlocks += 1
        for nxt in moves:
            if nxt not in seen:
                seen.add(nxt)
                todo.append(nxt)
    return {"states": len(seen), "unsafe": unsafe, "deadlocks": deadlocks, "terminals": terminals}


def test_entry_commitment_alone_can_overlap_peer_ub_use():
    assert explore_mix_idle((7,), idle_handshake=False)["unsafe"] > 0


def test_stopping_ordinary_work_at_entry_can_deadlock():
    result = explore_mix_idle((7,), idle_handshake=True, ordinary_needed=True, stop_at_entry=True)
    assert result["deadlocks"] > 0


@pytest.mark.parametrize("groups", [(3,), (5,), (6,), (7,), (7, 56)])
def test_idle_after_global_entry_preserves_resource_safety_and_progress(groups):
    for needed in (False, True):
        result = explore_mix_idle(groups, idle_handshake=True, ordinary_needed=needed)
        assert result["unsafe"] == 0, result
        assert result["deadlocks"] == 0, result
        assert result["terminals"] > 0, result
