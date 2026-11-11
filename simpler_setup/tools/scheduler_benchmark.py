# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Benchmark a scheduler on one run: how long it took, and where the time went.

From a chip-swimlane capture and the deps.json of the same run, four numbers
and the two steps between them:

    measured ── dispatch cost ──> squeezed ── schedule cost ──> best  ...  bound

- ``measured``: the run, from its first AICore start to its last end;
- ``squeezed``: the same cores running the same blocks in the same order, with
  every gap removed;
- ``best``: a near-optimal order and placement, by b-level list scheduling
  improved forward-backward, or any schedule here that is shorter;
- ``bound``: a makespan no schedule of this DAG on these cores can beat.

Two references stand beside them, because neither is a step: ``fifo``
(first-in-first-out by ready time) and ``sameOrder`` (this run's start order,
on any free core).

Every number reads the DAG, the run's kernel durations and the cores, and
``squeezed`` and ``sameOrder`` also the run's own placement and order -- never a
scheduler's rules -- so runs of any scheduler, on host_build_graph or
tensormap_and_ringbuffer, are measured by one rule. Every schedule is checked
before it is reported: each unit runs once, no consumer starts before all of
its producers end, no core runs two units at once, a MIX block's slots share
one cluster and one start, and a sync-start task's blocks share one start.

See docs/dfx/scheduler-benchmark.md.

Usage::

    python -m simpler_setup.tools.scheduler_benchmark <chip_swimlane_records.json> --deps <deps.json>
"""

from __future__ import annotations

import argparse
import collections
import heapq
import json
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

from ._runtime_dispatch import HBG_RUNTIME, get, normalize_task_id_int, resolve_runtime

SLOT_COUNT = 3  # AIC, AIV0, AIV1


class BenchmarkError(ValueError):
    """The inputs do not describe one consistent run, or a schedule broke a constraint."""


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Topology:
    """Cores of each type and the cluster each belongs to.

    A capture's core ids index ``metadata.core_types``, numbered as the
    runtime's scheduler numbers its workers: AIC workers occupy [0, clusters),
    then each cluster's AIV workers in turn. Lane 0 of a cluster is its AIC,
    lanes 1..aiv_per_cluster its AIVs, and MIX slot k runs on lane k.
    """

    clusters: int
    aiv_per_cluster: int

    @property
    def core_count(self) -> int:
        return self.clusters * (1 + self.aiv_per_cluster)

    def core_type(self, core: int) -> str:
        return "aic" if core < self.clusters else "aiv"

    def coordinate(self, core: int) -> tuple[int, int]:
        if core < self.clusters:
            return core, 0
        rank = core - self.clusters
        return rank // self.aiv_per_cluster, 1 + rank % self.aiv_per_cluster

    def core_at(self, cluster: int, lane: int) -> int:
        if lane == 0:
            return cluster
        return self.clusters + cluster * self.aiv_per_cluster + (lane - 1)

    def cores_of(self, core_type: str) -> list[int]:
        if core_type == "aic":
            return list(range(self.clusters))
        return list(range(self.clusters, self.core_count))

    @staticmethod
    def from_core_types(core_types: list[str]) -> Topology:
        aic = sum(1 for t in core_types if t == "aic")
        aiv = sum(1 for t in core_types if t == "aiv")
        if aic == 0 or aiv % aic != 0 or aic + aiv != len(core_types):
            raise BenchmarkError(f"core types do not form clusters: {aic} AIC, {aiv} AIV of {len(core_types)}")
        if any(t != "aic" for t in core_types[:aic]):
            raise BenchmarkError("AIC workers must precede AIV workers")
        return Topology(clusters=aic, aiv_per_cluster=aiv // aic)


# ---------------------------------------------------------------------------
# The DAG
# ---------------------------------------------------------------------------


@dataclass
class Task:
    """One schedulable task of the DAG.

    ``slots`` are the active subtask slots (0 = AIC, 1/2 = AIV0/AIV1); a task
    with none runs no kernel and completes the instant it is ready. A Graph
    task is not a task here: it becomes a zero-slot gate before its body's
    roots and a zero-slot join after its sinks.
    """

    tid: int
    slots: tuple[int, ...]
    block_num: int
    sync_start: bool = False
    predicated: bool = False
    preds: set[int] = field(default_factory=set)
    succs: set[int] = field(default_factory=set)

    @property
    def is_mix(self) -> bool:
        return len(self.slots) > 1

    @property
    def runs_kernel(self) -> bool:
        return bool(self.slots)


@dataclass
class Dag:
    runtime: str
    tasks: dict[int, Task]
    # Topological order of every task id, computed once.
    order: list[int]


def _task_id(value) -> int:
    tid = normalize_task_id_int(value)
    if tid is None:
        raise BenchmarkError(f"task id {value!r} is not an integer")
    return tid


def _slots_of(kernel_ids: list[int]) -> tuple[int, ...]:
    return tuple(i for i, k in enumerate(kernel_ids[:SLOT_COUNT]) if k >= 0)


TASK_FIELDS = ("kernel_ids", "block_num", "sync_start", "predicated")


def _task_of(tid: int, entry: dict) -> Task:
    missing = [k for k in TASK_FIELDS if k not in entry]
    if missing:
        raise BenchmarkError(f"deps.json task {tid} lacks {', '.join(missing)}; regenerate it with a current build")
    return Task(
        tid,
        _slots_of(entry["kernel_ids"]),
        int(entry["block_num"]),
        bool(entry["sync_start"]),
        bool(entry["predicated"]),
    )


def _runtime_of(name, source: str) -> str:
    try:
        return resolve_runtime(name, source=source)
    except ValueError as exc:
        raise BenchmarkError(str(exc)) from exc


def load_dag(deps: dict) -> Dag:
    """deps.json's task DAG, every Graph task expanded into the body it replays."""
    runtime = _runtime_of(deps["runtime"], "deps.json runtime")
    shells = {_task_id(s["task_id"]): s["definition"] for s in deps.get("graph_shells", [])}
    definitions = {d["definition"]: d for d in deps.get("graph_definitions", [])}
    if shells and runtime != HBG_RUNTIME:
        raise BenchmarkError(f"deps.json names Graph tasks, which {runtime} does not have")
    tasks: dict[int, Task] = {}
    # Virtual ids for a Graph task's gate and join: negative, so they never
    # collide with a task id (whose raw value is non-negative).
    gate_of: dict[int, int] = {}
    join_of: dict[int, int] = {}

    def add(task: Task) -> None:
        if task.tid in tasks:
            raise BenchmarkError(f"task {task.tid} appears twice in the DAG")
        tasks[task.tid] = task

    def edge(pred: int, succ: int) -> None:
        if pred != succ:
            tasks[pred].succs.add(succ)
            tasks[succ].preds.add(pred)

    for entry in deps["tasks"]:
        tid = _task_id(entry["task_id"])
        if tid in shells:
            gate_of[tid], join_of[tid] = -2 * len(gate_of) - 1, -2 * len(gate_of) - 2
            add(Task(gate_of[tid], (), 1))
            add(Task(join_of[tid], (), 1))
        else:
            add(_task_of(tid, entry))

    for shell, definition_id in shells.items():
        if shell not in gate_of:
            raise BenchmarkError(f"Graph task {shell} is not in tasks[]")
        body = definitions.get(definition_id)
        if body is None:
            raise BenchmarkError(f"Graph task {shell} names missing definition {definition_id}")
        ids = _body_ids(shell, body)
        for t in body["tasks"]:
            add(_task_of(ids[t["index"]], t))
        has_pred, has_succ = set(), set()
        for p, c in body["edges"]:
            edge(ids[p], ids[c])
            has_succ.add(p)
            has_pred.add(c)
        for index, tid in ids.items():
            if index not in has_pred:
                edge(gate_of[shell], tid)
            if index not in has_succ:
                edge(tid, join_of[shell])
        # The gate also precedes the join directly, so an empty body still orders
        # the Graph task's producers before its consumers.
        edge(gate_of[shell], join_of[shell])

    # A producer that is no task -- the host creating an external buffer is
    # recorded as a creator edge from an id no submit carries -- is satisfied
    # before the run starts, so its edges constrain nothing.
    for e in deps["edges"]:
        pred, succ = _task_id(e["pred"]), _task_id(e["succ"])
        if succ not in tasks and succ not in gate_of:
            raise BenchmarkError(f"an edge leads to task {succ}, which is not in tasks[]")
        if pred in tasks or pred in join_of:
            edge(join_of.get(pred, pred), gate_of.get(succ, succ))

    return Dag(runtime, tasks, _topological_order(tasks))


def _body_ids(shell: int, body: dict) -> dict[int, int]:
    """Per body task index, the id it runs under: body task i of a Graph task
    runs under make_sub_task(the Graph task's local id, i)."""
    task_id = get(HBG_RUNTIME).TaskId
    local = task_id(shell).local_id()
    ids = {t["index"]: task_id.make_sub_task(local, t["index"]).raw for t in body["tasks"]}
    for p, c in body["edges"]:
        if p not in ids or c not in ids:
            raise BenchmarkError(f"Graph definition {body['definition']} has an edge {p} -> {c} to a task it lacks")
    return ids


def _topological_order(tasks: dict[int, Task]) -> list[int]:
    indeg = {t: len(task.preds) for t, task in tasks.items()}
    ready = collections.deque(sorted(t for t, d in indeg.items() if d == 0))
    order = []
    while ready:
        t = ready.popleft()
        order.append(t)
        for s in sorted(tasks[t].succs):
            indeg[s] -= 1
            if indeg[s] == 0:
                ready.append(s)
    if len(order) != len(tasks):
        raise BenchmarkError(f"the DAG has a cycle ({len(order)} of {len(tasks)} tasks ordered)")
    return order


# ---------------------------------------------------------------------------
# Units: what a schedule places
# ---------------------------------------------------------------------------


@dataclass
class Member:
    """One slot of one block: what holds one core."""

    slot: int
    duration: int
    # The core and start this member ran at in the capture.
    core: int
    start: int


@dataclass
class Unit:
    """What starts as one: a block of a plain task, a block of a MIX task (its
    slots on one cluster), or every block of a sync-start task.

    ``blocks`` holds, per block, that block's members. A MIX block's members
    must share one cluster; a sync-start unit's blocks must share one start.
    """

    uid: int
    task: int
    blocks: list[list[Member]]

    @property
    def members(self) -> list[Member]:
        return [m for b in self.blocks for m in b]

    def duration(self) -> int:
        return max((m.duration for m in self.members), default=0)


@dataclass
class Workload:
    dag: Dag
    topology: Topology
    units: list[Unit]
    units_of: dict[int, list[int]]  # task id -> unit ids
    measured_start: int
    measured_end: int
    # Predicated tasks that left no row, taken as skipped by their predicate.
    skipped: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class Row:
    """One AICore record: the core it ran on, its task, and its span in cycles."""

    core: int
    task: int
    start: int
    end: int


def runs_of(records: dict) -> dict[int, list[Row]]:
    """The capture's AICore rows by the run (``run_epoch``) each belongs to; task
    ids repeat across runs. A record is ``[core_id, task_token, reg_task_id,
    start_cycles, end_cycles, receive_to_start_cycles, run_epoch]``."""
    by_run: dict[int, list[Row]] = collections.defaultdict(list)
    for row in records["aicore_tasks"]:
        if len(row) != 7:
            raise BenchmarkError(f"an aicore_tasks row has {len(row)} columns; the current format has 7")
        run = row[6]
        if type(run) is not int or run < 0:
            raise BenchmarkError(f"invalid run_epoch {run!r}; a run is a nonnegative integer, 0 included")
        if int(row[4]) < int(row[3]):
            raise BenchmarkError(f"an aicore_tasks row ends before it starts: {row}")
        by_run[run].append(Row(int(row[0]), _task_id(row[1]), int(row[3]), int(row[4])))
    if not by_run:
        raise BenchmarkError("the capture has no AICore rows")
    return dict(sorted(by_run.items()))


def build_workload(rows: list[Row], dag: Dag, topology: Topology) -> Workload:
    """Attach one run's AICore rows to the DAG's units."""
    rows_of: dict[int, list[Row]] = collections.defaultdict(list)
    for row in rows:
        rows_of[row.task].append(row)
    units: list[Unit] = []
    units_of: dict[int, list[int]] = {}
    skipped: list[int] = []
    for tid in dag.order:
        task = dag.tasks[tid]
        task_rows = rows_of.pop(tid, [])
        if not task.runs_kernel:
            # A kernel-less task holds no core in the model and ends the moment
            # it is ready, though a lane may record one (some schedulers run
            # dependency-only tasks on a lane).
            units_of[tid] = []
            continue
        if task.predicated and not task_rows:
            # A task whose dispatch predicate fails never reaches a core, so none
            # of its blocks leaves a row; it ends the moment it is ready. A task
            # that left rows for only some blocks is refused below as incomplete,
            # and one the runtime still sent to a core keeps its rows, like any
            # other. benchmark() has already refused a capture that reports
            # records lost.
            units_of[tid] = []
            skipped.append(tid)
            continue
        blocks = _blocks_from_rows(dag, task, task_rows, topology)
        unit_blocks = [blocks] if task.sync_start else [[b] for b in blocks]
        units_of[tid] = []
        for ub in unit_blocks:
            units_of[tid].append(len(units))
            units.append(Unit(len(units), tid, ub))
    stray = sorted(t for t, r in rows_of.items() if r)
    if stray:
        raise BenchmarkError(_stray_message(dag, stray))
    start, end = min(r.start for r in rows), max(r.end for r in rows)
    return Workload(dag, topology, units, units_of, start, end, skipped)


def _stray_message(dag: Dag, stray: list[int]) -> str:
    runtime, hbg = get(dag.runtime), get(HBG_RUNTIME)

    # A corrupt capture can hold an id no runtime mints, which has no name and
    # no space; it is shown raw.
    def name(t: int) -> str:
        try:
            return runtime.display(t)
        except ValueError:
            return f"{t:#x}"

    def is_sub_task(t: int) -> bool:
        try:
            return hbg.TaskId(t).space() == hbg.Space.SUB_TASK
        except ValueError:
            return False

    msg = f"{len(stray)} captured task ids are not in deps.json, e.g. {', '.join(name(t) for t in stray[:3])}"
    if dag.runtime == HBG_RUNTIME and any(is_sub_task(t) for t in stray):
        msg += (
            "; they are Graph body tasks, but this deps.json records no Graph replay (graph_definitions, "
            "graph_shells): the capture and deps.json are not from the same run"
        )
    return msg


def _blocks_from_rows(dag: Dag, task: Task, rows: list[Row], topology: Topology) -> list[list[Member]]:
    """Group a task's rows into its blocks."""
    name = get(dag.runtime).display(task.tid)
    if not rows:
        raise BenchmarkError(
            f"task {name} runs a kernel and has no predicate, but left no captured rows; the capture is incomplete"
        )
    if len(rows) != task.block_num * len(task.slots):
        raise BenchmarkError(
            f"task {name} has {len(rows)} captured rows where {task.block_num} block(s) x {len(task.slots)} slot(s) "
            "ran; the capture is incomplete"
        )
    for row in rows:
        core = row.core
        if not 0 <= core < topology.core_count:
            raise BenchmarkError(f"task {name} ran on core {core}, outside the topology")
        if task.is_mix:
            if topology.coordinate(core)[1] not in task.slots:
                raise BenchmarkError(f"MIX task {name} ran on a lane its mask {task.slots} excludes")
        elif topology.core_type(core) != ("aic" if task.slots[0] == 0 else "aiv"):
            raise BenchmarkError(f"task {name} ran on a {topology.core_type(core)} core")

    def member(row: Row, slot: int) -> Member:
        return Member(slot, row.end - row.start, row.core, row.start)

    if not task.is_mix:
        return [[member(r, task.slots[0])] for r in sorted(rows, key=lambda r: (r.start, r.core))]

    # A MIX block holds one core per slot on one cluster, and a lane runs its
    # blocks in turn, so the k-th row of every lane of a cluster is one block.
    per_lane: dict[tuple[int, int], list[Row]] = collections.defaultdict(list)
    for row in rows:
        per_lane[topology.coordinate(row.core)].append(row)
    blocks = []
    for cluster in sorted({c for c, _ in per_lane}):
        lanes = [sorted(per_lane.get((cluster, s), []), key=lambda r: r.start) for s in task.slots]
        counts = {len(lane) for lane in lanes}
        if len(counts) != 1:
            raise BenchmarkError(f"MIX task {name}: cluster {cluster} has unequal rows per lane {sorted(counts)}")
        for k in range(counts.pop()):
            blocks.append([member(lanes[i][k], s) for i, s in enumerate(task.slots)])
    return blocks


# ---------------------------------------------------------------------------
# Schedules and their validation
# ---------------------------------------------------------------------------


@dataclass
class Placement:
    """Where and when one unit runs in a schedule: one start, and per member
    (in Unit.members order) the core it holds."""

    start: int
    cores: list[int]


Schedule = dict[int, Placement]  # unit id -> placement


def task_ends(workload: Workload, schedule: Schedule) -> dict[int, int]:
    """Completion time of every task, zero-slot tasks included, in a schedule."""
    end: dict[int, int] = {}
    for tid in workload.dag.order:
        uids = workload.units_of[tid]
        if uids:
            end[tid] = max(schedule[u].start + workload.units[u].duration() for u in uids)
        else:
            end[tid] = max((end[p] for p in workload.dag.tasks[tid].preds), default=0)
    return end


def makespan(workload: Workload, schedule: Schedule) -> int:
    return max(task_ends(workload, schedule).values(), default=0)


def validate(workload: Workload, schedule: Schedule, label: str) -> None:
    """Raise BenchmarkError on any broken constraint; see the module docstring.

    A Placement carries one start for all of a unit's members, so a MIX
    block's slots and a sync-start task's blocks start together by
    construction; what is checked here is everything else.
    """
    if set(schedule) != {u.uid for u in workload.units}:
        raise BenchmarkError(f"{label}: {len(schedule)} units placed, {len(workload.units)} expected")
    busy: dict[int, list[tuple[int, int, int]]] = collections.defaultdict(list)
    for unit in workload.units:
        _check_placement(workload, unit, schedule[unit.uid], label, busy)
    for core, spans in busy.items():
        spans.sort()
        last_end, last_uid = -1, None
        for s, e, uid in spans:
            if e == s:
                continue
            if s < last_end:
                raise BenchmarkError(f"{label}: core {core} runs units {last_uid} and {uid} at once")
            last_end, last_uid = e, uid
    end = task_ends(workload, schedule)
    for tid, task in workload.dag.tasks.items():
        ready = max((end[p] for p in task.preds), default=0)
        for u in workload.units_of[tid]:
            if schedule[u].start < ready:
                raise BenchmarkError(f"{label}: unit {u} starts at {schedule[u].start} before its producers end")


def _check_placement(workload: Workload, unit: Unit, p: Placement, label: str, busy: dict) -> None:
    """Cores of the right type and lane, a MIX block on one cluster; records each
    member's busy span on its core."""
    topo = workload.topology
    if p.start < 0:
        raise BenchmarkError(f"{label}: unit {unit.uid} starts at {p.start}")
    if len(p.cores) != len(unit.members):
        raise BenchmarkError(f"{label}: unit {unit.uid} has {len(p.cores)} cores for {len(unit.members)} members")
    is_mix = workload.dag.tasks[unit.task].is_mix
    index = 0
    for block in unit.blocks:
        cores = p.cores[index : index + len(block)]
        index += len(block)
        for m, c in zip(block, cores):
            if not 0 <= c < topo.core_count:
                raise BenchmarkError(f"{label}: unit {unit.uid} slot {m.slot} placed on core {c}")
            wrong_lane = is_mix and topo.coordinate(c)[1] != m.slot
            wrong_type = not is_mix and topo.core_type(c) != ("aic" if m.slot == 0 else "aiv")
            if wrong_lane or wrong_type:
                raise BenchmarkError(f"{label}: unit {unit.uid} slot {m.slot} placed on core {c}")
            busy[c].append((p.start, p.start + m.duration, unit.uid))
        clusters = {topo.coordinate(c)[0] for c in cores}
        if is_mix and len(clusters) > 1:
            raise BenchmarkError(f"{label}: MIX unit {unit.uid} spans clusters {sorted(clusters)}")


# ---------------------------------------------------------------------------
# squeezed: this run's cores and per-core order, every gap removed
# ---------------------------------------------------------------------------


def measured_schedule(workload: Workload) -> dict[int, list[tuple[int, int]]]:
    """Per unit, the (core, start) of every member as captured, relative to time 0."""
    t0 = workload.measured_start
    return {u.uid: [(m.core, m.start - t0) for m in u.members] for u in workload.units}


def _captured_core_order(units: list[Unit], measured: dict) -> dict[int, set[tuple[int, int]]]:
    """Per unit, the (unit, member index) that ran just before each of its
    members on that member's core in the capture."""
    seq: dict[int, list[tuple[int, int, int]]] = collections.defaultdict(list)
    for unit in units:
        for idx, (core, start) in enumerate(measured[unit.uid]):
            seq[core].append((start, unit.uid, idx))
    core_prev: dict[int, set[tuple[int, int]]] = collections.defaultdict(set)
    for items in seq.values():
        items.sort()
        for (_, u0, i0), (_, u1, _) in zip(items, items[1:]):
            if u0 != u1:
                core_prev[u1].add((u0, i0))
    return core_prev


def squeezed(workload: Workload) -> Schedule:
    """Each core runs what it ran, in the order it ran it, and every unit starts
    the moment its producers are done, its cores are free and -- for a MIX block
    or a sync-start task -- every one of its cores is free.

    Solved as a longest path over the constraint graph: a unit waits for the
    unit before it on each of its cores and for every producer task's end.
    Starting a unit's members together can make it longer than the run when the
    capture started them apart.
    """
    measured = measured_schedule(workload)
    units = workload.units
    core_prev = _captured_core_order(units, measured)

    # Kahn over units and tasks together. A task finishes when its units do; a
    # unit becomes eligible once its producer tasks and its core predecessors
    # are placed.
    tasks = workload.dag.tasks
    waiting_unit = {u.uid: len(core_prev[u.uid]) + len(tasks[u.task].preds) for u in units}
    waiting_task = {t: len(workload.units_of[t]) or len(tasks[t].preds) for t in tasks}
    unit_waiters: dict[int, list[int]] = collections.defaultdict(list)
    for uid, prevs in core_prev.items():
        for pu, _ in prevs:
            unit_waiters[pu].append(uid)
    start: dict[int, int] = {}
    ready_at = dict.fromkeys(tasks, 0)
    queue: collections.deque[tuple[str, int]] = collections.deque()
    # Tasks with no units and no producers finish at 0; units with nothing to
    # wait for start at 0.
    queue.extend(("task", t) for t, task in tasks.items() if not workload.units_of[t] and not task.preds)
    queue.extend(("unit", uid) for uid, w in waiting_unit.items() if w == 0)

    def finish_task(t: int) -> None:
        if workload.units_of[t]:
            end = max(start[u] + units[u].duration() for u in workload.units_of[t])
        else:
            end = ready_at[t]
        for s in tasks[t].succs:
            ready_at[s] = max(ready_at[s], end)
            if workload.units_of[s]:
                for u in workload.units_of[s]:
                    waiting_unit[u] -= 1
                    if waiting_unit[u] == 0:
                        queue.append(("unit", u))
            else:
                waiting_task[s] -= 1
                if waiting_task[s] == 0:
                    queue.append(("task", s))

    while queue:
        kind, key = queue.popleft()
        if kind == "task":
            finish_task(key)
            continue
        unit = units[key]
        s = ready_at[unit.task]
        for pu, pi in core_prev[key]:
            s = max(s, start[pu] + units[pu].members[pi].duration)
        start[key] = s
        for w in unit_waiters[key]:
            waiting_unit[w] -= 1
            if waiting_unit[w] == 0:
                queue.append(("unit", w))
        waiting_task[unit.task] -= 1
        if waiting_task[unit.task] == 0:
            finish_task(unit.task)
    if len(start) != len(units):
        raise BenchmarkError(
            f"squeezed: the captured order conflicts with the DAG ({len(start)} of {len(units)} units placed); "
            "a core ran a consumer before its producer, or a sync-start task's blocks were captured out of order"
        )
    return {uid: Placement(start[uid], [c for c, _ in measured[uid]]) for uid in start}


# ---------------------------------------------------------------------------
# List scheduling: fifo, sameOrder and best
# ---------------------------------------------------------------------------


def _shape(workload: Workload, unit: Unit) -> tuple:
    return (workload.dag.tasks[unit.task].slots, len(unit.blocks))


def _fit(workload: Workload, unit: Unit, free: set[int]) -> list[int] | None:
    """Cores for `unit` among `free`, lowest first, or None."""
    topo = workload.topology
    task = workload.dag.tasks[unit.task]
    chosen: list[int] = []
    taken: set[int] = set()
    for block in unit.blocks:
        if task.is_mix:
            for cluster in range(topo.clusters):
                cores = [topo.core_at(cluster, m.slot) for m in block]
                if all(c in free and c not in taken for c in cores):
                    chosen.extend(cores)
                    taken.update(cores)
                    break
            else:
                return None
        else:
            core_type = "aic" if block[0].slot == 0 else "aiv"
            for c in topo.cores_of(core_type):
                if c in free and c not in taken:
                    chosen.append(c)
                    taken.add(c)
                    break
            else:
                return None
    return chosen


def _list_schedule(workload: Workload, priority, label: str) -> Schedule:
    """Zero-overhead list scheduling with one queue per unit shape.

    At every event the ready unit with the smallest ``priority(uid, ready)``
    that fits starts at once, on the lowest free cores it fits. A shape whose
    head does not fit is passed over, since every unit of one shape needs the
    same cores.
    """
    dag, units = workload.dag, workload.units
    free = set(range(workload.topology.core_count))
    queues: dict[tuple, list[tuple]] = collections.defaultdict(list)  # (priority, uid)
    remaining_units = {t: len(workload.units_of[t]) for t in dag.tasks}
    remaining_preds = {t: len(dag.tasks[t].preds) for t in dag.tasks}
    events: list[tuple[int, int, int]] = []  # (time, kind, payload): kind 0 core free, 1 unit done
    schedule: Schedule = {}

    def released(t: int) -> list[int]:
        """The successors of the finished task `t` that were waiting only for it."""
        out = []
        for s in dag.tasks[t].succs:
            remaining_preds[s] -= 1
            if remaining_preds[s] == 0:
                out.append(s)
        return out

    def make_ready(ready: list[int], now: int) -> None:
        # A task without units is done the moment it is ready, which can make
        # its successors ready in turn.
        while ready:
            t = ready.pop()
            if not workload.units_of[t]:
                ready.extend(released(t))
            for uid in workload.units_of[t]:
                heapq.heappush(queues[_shape(workload, units[uid])], (priority(uid, now), uid))

    make_ready([t for t in dag.order if not dag.tasks[t].preds], 0)

    now = 0
    while True:
        while True:
            best_shape, best_cores = None, None
            for shape, q in queues.items():
                if q and (best_shape is None or q[0] < queues[best_shape][0]):
                    cores = _fit(workload, units[q[0][1]], free)
                    if cores is not None:
                        best_shape, best_cores = shape, cores
            if best_shape is None or best_cores is None:
                break
            _, uid = heapq.heappop(queues[best_shape])
            schedule[uid] = Placement(now, best_cores)
            unit = units[uid]
            for m, c in zip(unit.members, best_cores):
                free.discard(c)
                heapq.heappush(events, (now + m.duration, 0, c))
            heapq.heappush(events, (now + unit.duration(), 1, uid))
        if not events:
            break
        now = events[0][0]
        while events and events[0][0] == now:
            _, kind, payload = heapq.heappop(events)
            if kind == 0:
                free.add(payload)
            else:
                t = units[payload].task
                remaining_units[t] -= 1
                if remaining_units[t] == 0:
                    make_ready(released(t), now)
    if len(schedule) != len(units):
        raise BenchmarkError(f"{label} stalled with {len(schedule)} of {len(units)} units placed")
    return schedule


def fifo(workload: Workload) -> Schedule:
    """First-in-first-out: units queue by the time they became ready. It depends
    on nothing but the DAG and the kernel durations."""
    units = workload.units
    return _list_schedule(workload, lambda uid, ready: (ready, units[uid].task, uid), "fifo")


def same_order(workload: Workload) -> Schedule:
    """This run's start order, any free core: units queue by when the capture
    started them, and take whichever compatible cores are free."""
    started = {uid: min(s for _, s in members) for uid, members in measured_schedule(workload).items()}
    return _list_schedule(workload, lambda uid, ready: (started[uid], uid), "sameOrder")


def _task_duration(workload: Workload, tid: int) -> int:
    return max((workload.units[u].duration() for u in workload.units_of[tid]), default=0)


def b_levels(workload: Workload) -> dict[int, int]:
    """Per task, the longest chain of durations from its start to the end of
    the graph, its own duration included; kernel-less tasks add nothing."""
    dag = workload.dag
    out: dict[int, int] = {}
    for tid in reversed(dag.order):
        out[tid] = _task_duration(workload, tid) + max((out[s] for s in dag.tasks[tid].succs), default=0)
    return out


def _reversed(workload: Workload) -> Workload:
    """The same units on the DAG with every edge turned around."""
    tasks = {
        tid: replace(task, preds=set(task.succs), succs=set(task.preds)) for tid, task in workload.dag.tasks.items()
    }
    dag = Dag(workload.dag.runtime, tasks, list(reversed(workload.dag.order)))
    return replace(workload, dag=dag)


def best(workload: Workload, rounds: int = 16) -> Schedule:
    """List scheduling by b-level (longest remaining chain first), then
    forward-backward improvement.

    One improvement round schedules the reversed DAG with the units that end
    latest going first, and then the DAG again with the units whose mirrored
    backward start is earliest going first. Every schedule is a list schedule
    of the real or the reversed DAG; only forward ones are kept, and a round
    is kept only if it is shorter, so the result is never longer than the
    b-level schedule. Rounds stop at the first that does not improve.
    """
    units = workload.units
    level = b_levels(workload)
    found = _list_schedule(workload, lambda uid, ready: (-level[units[uid].task], uid), "best")
    found_len = makespan(workload, found)
    back_workload = _reversed(workload)
    for _ in range(rounds):
        end = {uid: p.start + units[uid].duration() for uid, p in found.items()}
        back = _list_schedule(back_workload, lambda uid, ready: (-end[uid], uid), "best backward")
        back_len = makespan(back_workload, back)
        mirrored = {uid: back_len - (p.start + units[uid].duration()) for uid, p in back.items()}
        fwd = _list_schedule(workload, lambda uid, ready: (mirrored[uid], uid), "best forward")
        fwd_len = makespan(workload, fwd)
        if fwd_len >= found_len:
            break
        found, found_len = fwd, fwd_len
    return found


# ---------------------------------------------------------------------------
# bound: no schedule can be shorter
# ---------------------------------------------------------------------------


def bound(workload: Workload, heads: int = 64) -> int:
    """A makespan no valid schedule can beat, summed over barrier segments.

    Within a segment, for each core type T and any set S of units that use T:
    no unit of S starts before the smallest head in S (its task's earliest
    start along the DAG), and after each one at least the smallest tail in S
    (its task's b-level minus its task's duration) remains. In between, S's
    work on T must fit on T's cores, so the segment lasts at least
    head + ceil(work / cores) + tail. Sets are taken as "head >= h and tail >= q",
    over every q and a sample of h: every distinct head when there are fewer than
    2 * `heads`, else every k-th, which keeps between `heads` and 1.5 * `heads`.
    The segment's critical path is a bound too, and the largest of them is the
    segment's.
    """
    preds = _real_preds(workload)
    topo = workload.topology
    cores = {"aic": topo.clusters, "aiv": topo.clusters * topo.aiv_per_cluster}
    total = 0
    for segment in barrier_segments(workload):
        order = [t for t in workload.dag.order if t in segment]
        dur = {t: _task_duration(workload, t) for t in order}
        succs: dict[int, list[int]] = collections.defaultdict(list)
        head: dict[int, int] = {}
        for t in order:
            inside = [p for p in preds[t] if p in segment]
            head[t] = max((head[p] + dur[p] for p in inside), default=0)
            for p in inside:
                succs[p].append(t)
        level: dict[int, int] = {}
        for t in reversed(order):
            level[t] = dur[t] + max((level[s] for s in succs[t]), default=0)
        longest = max(level.values(), default=0)
        for core_type, n in cores.items():
            items = []  # (tail, head, work on this core type)
            for t in order:
                for u in workload.units_of[t]:
                    work = sum(m.duration for m in workload.units[u].members if (m.slot == 0) == (core_type == "aic"))
                    if work:
                        items.append((level[t] - dur[t], head[t], work))
            if not items:
                continue
            items.sort(reverse=True)
            distinct = sorted({h for _, h, _ in items})
            step = max(1, len(distinct) // heads)
            for h in distinct[::step]:
                work = 0
                for tail, hd, w in items:
                    if hd >= h:
                        work += w
                        longest = max(longest, h + -(-work // n) + tail)
        total += longest
    return total


def _real_preds(workload: Workload) -> dict[int, set[int]]:
    """For every task with units, its nearest producer tasks that have units,
    looking through zero-slot tasks. A unit of t may start only once every
    unit of each of these has ended."""
    dag = workload.dag
    reach: dict[int, set[int]] = {}
    for t in dag.order:
        acc: set[int] = set()
        for p in dag.tasks[t].preds:
            if workload.units_of[p]:
                acc.add(p)
            else:
                acc |= reach[p]
        reach[t] = acc
    return {t: reach[t] for t in dag.order if workload.units_of[t]}


def barrier_segments(workload: Workload) -> list[set[int]]:
    """The core-holding tasks, split at barriers, in time order.

    A barrier is a kernel-less task that every core-holding task either
    precedes or follows. In any valid schedule its ancestors have all ended
    before its descendants start, so the segments between barriers never
    overlap and the optimum is the sum of each segment's optimum. A Graph
    task's gate and join are barriers whenever consecutive Graph tasks chain.
    """
    dag = workload.dag
    real = [t for t in dag.order if workload.units_of[t]]
    real_set = set(real)

    def reach(start: int, step) -> set[int]:
        seen = {start}
        stack = [start]
        while stack:
            for v in step(stack.pop()):
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        return seen

    barriers = set()
    for t in dag.order:
        task = dag.tasks[t]
        if workload.units_of[t] or not task.preds or not task.succs:
            continue
        if real_set <= reach(t, lambda x: dag.tasks[x].preds) | reach(t, lambda x: dag.tasks[x].succs):
            barriers.add(t)
    # A task's segment is the number of barriers it follows. Barriers are
    # totally ordered, so that count is the longest barrier chain above it.
    above: dict[int, int] = {}
    for t in dag.order:
        above[t] = max((above[p] + (1 if p in barriers else 0) for p in dag.tasks[t].preds), default=0)
    segments: dict[int, set[int]] = collections.defaultdict(set)
    for t in real:
        segments[above[t]].add(t)
    return [segments[k] for k in sorted(segments)]


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

# Each published number: its name, and what it is, as the report prints it.
CHAIN = (
    ("measured", "what the run took"),
    ("squeezed", "the same cores in the same order, every gap removed"),
    ("best", "a near-optimal order and placement"),
    ("bound", "no schedule of this DAG can be shorter"),
)
REFERENCES = (
    ("fifo", "first-in-first-out by ready time; none of this run's decisions"),
    ("sameOrder", "this run's start order, any free core"),
)
# A step is the difference of the two numbers it sits between.
STEPS = (
    ("dispatch cost", "measured", "squeezed"),
    ("schedule cost", "squeezed", "best"),
)
SCHEDULES = {"squeezed": squeezed, "best": best, "fifo": fifo, "sameOrder": same_order}


def measure(workload: Workload, us: float) -> dict[str, float]:
    """Every published number of one run, in microseconds."""
    t = {}
    for name, schedule_of in SCHEDULES.items():
        schedule = schedule_of(workload)
        validate(workload, schedule, name)
        t[name] = makespan(workload, schedule) * us
    # The b-level search can lose to a schedule it did not try, the run's own
    # included, so best is the shortest valid schedule found.
    t["best"] = min(t[name] for name in SCHEDULES)
    t["measured"] = (workload.measured_end - workload.measured_start) * us
    t["bound"] = bound(workload) * us
    if min(t[name] for name in SCHEDULES) < t["bound"] - 1e-9:
        raise BenchmarkError("a schedule is shorter than the bound; the bound or the schedule is wrong")
    return t


# The counts metadata.collection keeps of records that never reached the file.
LOSS_COUNTS = (
    "not_received_buffers",
    "unpublished_loss",
    "transport_retired",
    "transport_retired_records",
    "unattributable_handoffs",
    "cut_failed_queues",
)


def _check_complete(metadata: dict) -> None:
    """Refuse a capture that reports lost records: they would read as work that
    never ran, a predicated task's as a skip.

    Only the collection path that retains runs writes ``metadata.collection``; a
    capture from the default path carries no such report and is taken as complete.
    """
    c = metadata.get("collection")
    if c is None:
        return
    problems = [] if c["verdict"] == "published" else [f"verdict {c['verdict']}"]
    problems += [f"{k} {c[k]}" for k in LOSS_COUNTS if c[k]]
    problems += [f"{k} false" for k in ("processing_complete", "handoff_identity_verified") if not c[k]]
    if problems:
        raise BenchmarkError(f"the capture reports an incomplete collection: {', '.join(problems)}")


def benchmark(records: dict, deps: dict) -> list[dict]:
    """The report of every run in one capture, in run order."""
    dag = load_dag(deps)
    metadata = records["metadata"]
    if _runtime_of(metadata["runtime"], "metadata.runtime") != dag.runtime:
        raise BenchmarkError(f"the capture is {metadata['runtime']} but deps.json is {dag.runtime}")
    _check_complete(metadata)
    topology = Topology.from_core_types(metadata["core_types"])
    hz = metadata["clock_freq_hz"]
    if type(hz) is not int or hz <= 0:
        raise BenchmarkError(f"invalid clock_freq_hz {hz!r}; expected a positive integer")
    us = 1e6 / hz
    reports = []
    for run, rows in runs_of(records).items():
        workload = build_workload(rows, dag, topology)
        t = measure(workload, us)
        reports.append(
            {
                "run": run,
                "runtime": dag.runtime,
                "units": len(workload.units),
                "skipped": len(workload.skipped),
                "clusters": topology.clusters,
                "time_us": t,
                "step_us": {step: t[hi] - t[lo] for step, hi, lo in STEPS},
            }
        )
    return reports


def render(report: dict, label: str) -> str:
    """One run's report as the lines a reader compares between two captures."""
    t, step = report["time_us"], report["step_us"]
    width = max(len(name) for name, _ in (*CHAIN, *REFERENCES))
    shape = f"{report['runtime']}; {report['units']:,} units on {report['clusters']} clusters"
    out = [f"== {label} run {report['run']} ({shape})"]
    if report["skipped"]:
        out.append(f"  {report['skipped']:,} predicated task(s) left no AICore row and count as skipped")
    # A step is printed between the two numbers it is the difference of, so the
    # column reads top to bottom as number, step, number.
    after = {hi: name for name, hi, _ in STEPS}
    for name, what in CHAIN:
        out.append(f"  {name:<{width}}  {t[name]:12,.1f} us   {what}")
        if name in after:
            out.append(f"  {'':<{width}}  {step[after[name]]:12,.1f} us     - {after[name]}")
    out.append(f"  {'':<{width}}  {'':>12}      the shortest schedule is in [bound, best]")
    out.append("")
    for name, what in REFERENCES:
        out.append(f"  {name:<{width}}  {t[name]:12,.1f} us   {what}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Benchmark a scheduler on one run: how long it took and where it went.")
    p.add_argument("records", type=Path, help="chip_swimlane_records.json of the run to benchmark")
    p.add_argument("--deps", type=Path, required=True, help="deps.json of the same workload")
    p.add_argument("--label", help="name the capture in the report (default: the records file's directory)")
    args = p.parse_args(argv)
    try:
        reports = benchmark(
            json.loads(args.records.read_text(encoding="utf-8")), json.loads(args.deps.read_text(encoding="utf-8"))
        )
    except (BenchmarkError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyError as exc:
        print(f"error: the records or deps.json lack the field {exc}", file=sys.stderr)
        return 2
    except (TypeError, ValueError) as exc:
        print(f"error: the records or deps.json hold a field of the wrong type or value: {exc}", file=sys.stderr)
        return 2
    label = args.label or args.records.resolve().parent.name
    print("\n\n".join(render(report, label) for report in reports))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
