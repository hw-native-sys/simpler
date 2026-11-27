# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Finite interleaving checker for admission, ownership and barrier workloads."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import NamedTuple


@dataclass(frozen=True)
class Workload:
    name: str
    members: tuple
    blocks: int
    dependencies: tuple = ()
    barrier: bool = False
    mapping: tuple = ()


class State(NamedTuple):
    order: tuple
    next_block: tuple
    owner: tuple
    phase: tuple
    closed: tuple


@dataclass
class Exploration:
    states: int
    terminals: int
    deadlocks: int
    overlap: bool
    b_before_unready_a: bool
    deadlock_witness: tuple


def _replace_cell(matrix, row, col, value):
    data = list(matrix[row])
    data[col] = value
    result = list(matrix)
    result[row] = tuple(data)
    return tuple(result)


def _done(state, task):
    return all(phase == 3 for phase in state.phase[task])


def _ready(state, tasks, task):
    return all(_done(state, predecessor) for predecessor in tasks[task].dependencies)


def _worker_actions(state, tasks, worker, ordered, early_close):
    active = [
        (t, b)
        for t, task in enumerate(tasks)
        for b in range(task.blocks)
        if state.owner[t][b] == worker and state.phase[t][b] in (1, 2)
    ]
    if active:
        task, block = active[0]
        phase = state.phase[task][block]
        if phase == 1 or not tasks[task].barrier or all(p >= 2 for p in state.phase[task]):
            event = "enter" if phase == 1 else "return"
            yield (
                f"{worker}:{event}:{tasks[task].name}.{block}",
                state._replace(phase=_replace_cell(state.phase, task, block, phase + 1)),
            )
        return
    choices = [t for t in state.order if worker in tasks[t].members and not state.closed[worker] & (1 << t)]
    if ordered:
        choices = choices[:1]
    for task in choices:
        spec = tasks[task]
        block = -1
        if spec.barrier:
            block = next(
                (b for b, owner in enumerate(spec.mapping) if owner == worker and state.phase[task][b] == 0), -1
            )
            can_close = block < 0
        else:
            if state.next_block[task] < spec.blocks:
                block = state.next_block[task]
            can_close = block < 0 or (
                early_close
                and any(owner == worker and state.phase[task][b] == 3 for b, owner in enumerate(state.owner[task]))
            )
        if block >= 0:
            next_block = list(state.next_block)
            if not spec.barrier:
                next_block[task] += 1
            yield (
                f"{worker}:claim:{spec.name}.{block}",
                state._replace(
                    next_block=tuple(next_block),
                    owner=_replace_cell(state.owner, task, block, worker),
                    phase=_replace_cell(state.phase, task, block, 1),
                ),
            )
        if can_close:
            closed = list(state.closed)
            closed[worker] |= 1 << task
            yield f"{worker}:close:{spec.name}", state._replace(closed=tuple(closed))


def _actions(state, tasks, workers, ordered, early_close, fixed_admission):
    for task in range(len(tasks)):
        if task in state.order or not _ready(state, tasks, task):
            continue
        if fixed_admission and task != len(state.order):
            continue
        yield f"admit:{tasks[task].name}", state._replace(order=state.order + (task,))
    for worker in range(workers):
        yield from _worker_actions(state, tasks, worker, ordered, early_close)


def _terminal(state, tasks):
    return len(state.order) == len(tasks) and all(
        _done(state, t) and all(state.closed[w] & (1 << t) for w in spec.members) for t, spec in enumerate(tasks)
    )


def explore(tasks, workers, ordered=True, early_close=False, fixed_admission=False, limit=50000):
    """Raises if the whole reachable space cannot be exhausted within limit."""
    initial = State(
        (),
        tuple(0 for _ in tasks),
        tuple(tuple(-1 for _ in range(t.blocks)) for t in tasks),
        tuple(tuple(0 for _ in range(t.blocks)) for t in tasks),
        tuple(0 for _ in range(workers)),
    )
    pending = deque([initial])
    parents: dict[State, tuple[State, str] | None] = {initial: None}
    deadlocks = terminals = 0
    overlap = b_before_unready_a = False
    witness = ()
    a = next((i for i, t in enumerate(tasks) if t.name == "A"), None)
    b = next((i for i, t in enumerate(tasks) if t.name == "B"), None)
    while pending:
        state = pending.popleft()
        if a is not None and b is not None:
            if a in state.order and b in state.order and state.order.index(a) < state.order.index(b):
                overlap |= (
                    any(p >= 2 for p in state.phase[a]) and not _done(state, a) and any(p >= 2 for p in state.phase[b])
                )
            b_before_unready_a |= any(p >= 2 for p in state.phase[b]) and not _ready(state, tasks, a)
        if _terminal(state, tasks):
            terminals += 1
            continue
        successors = tuple(_actions(state, tasks, workers, ordered, early_close, fixed_admission))
        if not successors:
            deadlocks += 1
            if not witness:
                path = []
                cursor = state
                while True:
                    parent = parents[cursor]
                    if parent is None:
                        break
                    cursor, action = parent
                    path.append(action)
                witness = tuple(reversed(path))
        for action, successor in successors:
            if successor in parents:
                continue
            if len(parents) >= limit:
                raise RuntimeError(f"incomplete exploration: state budget {limit} exhausted")
            parents[successor] = (state, action)
            pending.append(successor)
    return Exploration(len(parents), terminals, deadlocks, overlap, b_before_unready_a, witness)
