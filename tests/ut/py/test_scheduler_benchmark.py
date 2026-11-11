# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The scheduler benchmark's numbers, on graphs whose answers are known.

Every schedule is checked twice: its makespan against a hand-computed value or
an exhaustive search, and against the validator. The validator itself is
checked against schedules that break each constraint on purpose.
"""

from __future__ import annotations

import collections
import itertools
import json
import random

import pytest

from simpler_setup.tools import scheduler_benchmark as sb
from simpler_setup.tools.hbg.task_id import TaskId as HbgTaskId
from simpler_setup.tools.tmr.task_id import TaskId as TmrTaskId

AIC, AIV0, AIV1 = [0, -1, -1], [-1, 0, -1], [-1, -1, 0]
MIX = [0, 0, 0]
NONE = [-1, -1, -1]


class Capture:
    """A capture and its deps.json, built by hand in the current format. Cores:
    `clusters` AIC workers, then two AIV per cluster, as the runtime numbers them."""

    def __init__(self, clusters: int = 2, runtime: str = "host_build_graph"):
        self.clusters = clusters
        self.runtime = runtime
        self.tasks, self.edges, self.rows = [], [], []
        self.graph_definitions, self.graph_shells = [], []
        self.collection = None

    def task(self, tid, kernel_ids, block_num=1, sync_start=False, predicated=False):
        self.tasks.append(
            {
                "task_id": str(tid),
                "kernel_ids": kernel_ids,
                "block_num": block_num,
                "sync_start": sync_start,
                "predicated": predicated,
            }
        )
        return self

    def edge(self, pred, succ):
        self.edges.append({"pred": str(pred), "succ": str(succ)})
        return self

    def row(self, core, tid, start, end, run=1):
        self.rows.append([core, tid, len(self.rows), start, end, 0, run])
        return self

    def aiv(self, cluster, lane):
        return self.clusters + cluster * 2 + (lane - 1)

    def records(self):
        types = ["aic"] * self.clusters + ["aiv"] * (2 * self.clusters)
        metadata = {"runtime": self.runtime, "core_types": types, "clock_freq_hz": 1_000_000}
        if self.collection is not None:
            metadata["collection"] = self.collection
        return {"metadata": metadata, "aicore_tasks": self.rows}

    def deps(self):
        out = {"runtime": self.runtime, "tasks": self.tasks, "edges": self.edges, "tensors": []}
        if self.graph_shells:
            out["graph_definitions"], out["graph_shells"] = self.graph_definitions, self.graph_shells
        return out

    def workload(self):
        dag = sb.load_dag(self.deps())
        (rows,) = sb.runs_of(self.records()).values()
        return sb.build_workload(rows, dag, sb.Topology.from_core_types(self.records()["metadata"]["core_types"]))


def numbers(cap: Capture) -> dict[str, int]:
    """Every published number of `cap`'s one run, in cycles (the clock is 1 MHz)."""
    (report,) = sb.benchmark(cap.records(), cap.deps())
    return {k: round(v) for k, v in report["time_us"].items()}


def checked(w, schedule_of):
    s = schedule_of(w)
    sb.validate(w, s, schedule_of.__name__)
    return s


def sub_task(shell: int, index: int) -> int:
    return HbgTaskId.make_sub_task(HbgTaskId(shell).local_id(), index).raw


# ---------------------------------------------------------------------------
# Hand-built graphs with known answers
# ---------------------------------------------------------------------------


def test_a_chain_squeezes_out_its_gaps():
    cap = Capture().task(1, AIC).task(2, AIV0).task(3, AIC).edge(1, 2).edge(2, 3)
    cap.row(0, 1, 100, 110).row(cap.aiv(0, 1), 2, 115, 120).row(1, 3, 130, 140)
    t = numbers(cap)
    assert t["measured"] == 40
    assert t["squeezed"] == t["best"] == t["bound"] == t["fifo"] == t["sameOrder"] == 25


def test_fork_join_waits_for_both_branches():
    cap = Capture().task(1, AIC).task(2, AIV0).task(3, AIV0).task(4, AIC)
    cap.edge(1, 2).edge(1, 3).edge(2, 4).edge(3, 4)
    cap.row(0, 1, 0, 10).row(cap.aiv(0, 1), 2, 12, 32).row(cap.aiv(1, 1), 3, 12, 32).row(0, 4, 40, 45)
    w = cap.workload()
    s = checked(w, sb.fifo)
    assert s[w.units_of[4][0]].start == 30, "the join must wait for its slowest producer"
    assert numbers(cap)["squeezed"] == 35


def test_spmd_blocks_beyond_the_cores_queue_on_them():
    # Three blocks on two AIC cores: one core runs two back to back. The bound
    # is the work over the cores, 15; blocks do not split, so 20 is the optimum.
    cap = Capture().task(1, AIC, block_num=3)
    cap.row(0, 1, 0, 10).row(0, 1, 10, 20).row(1, 1, 0, 10)
    t = numbers(cap)
    assert t["squeezed"] == t["fifo"] == t["best"] == 20
    assert t["bound"] == 15


def test_mix_takes_a_whole_cluster_and_moves_to_a_free_one():
    # A long AIV0 task holds cluster 0's AIV0, so the MIX block (all three
    # lanes) cannot use cluster 0 and starts on cluster 1.
    cap = Capture().task(1, AIV0).task(2, MIX)
    cap.row(cap.aiv(0, 1), 1, 0, 50)
    cap.row(1, 2, 0, 10).row(cap.aiv(1, 1), 2, 0, 10).row(cap.aiv(1, 2), 2, 0, 10)
    w = cap.workload()
    s = checked(w, sb.fifo)
    mix = s[w.units_of[2][0]]
    assert mix.start == 0
    assert {w.topology.coordinate(c)[0] for c in mix.cores} == {1}


def test_mix_waits_when_no_cluster_is_whole():
    cap = Capture(clusters=1).task(1, AIV1).task(2, MIX)
    cap.row(cap.aiv(0, 2), 1, 0, 30)
    cap.row(0, 2, 30, 40).row(cap.aiv(0, 1), 2, 30, 40).row(cap.aiv(0, 2), 2, 30, 40)
    t = numbers(cap)
    assert t["squeezed"] == t["fifo"] == t["best"] == 40


def test_a_sync_start_task_starts_together_once_enough_cores_are_free():
    # Four AIV cores; two 20-long tasks hold two, so a three-block sync-start
    # task waits until one frees and then starts all at once.
    cap = Capture().task(1, AIV0).task(2, AIV0).task(3, AIV0, block_num=3, sync_start=True)
    cap.row(cap.aiv(0, 1), 1, 0, 20).row(cap.aiv(0, 2), 2, 0, 20)
    for core in (cap.aiv(1, 1), cap.aiv(1, 2), cap.aiv(0, 1)):
        cap.row(core, 3, 20, 30)
    w = cap.workload()
    assert len(w.units_of[3]) == 1, "a sync-start task is one unit"
    s = checked(w, sb.fifo)
    assert s[w.units_of[3][0]].start == 20
    assert len(set(s[w.units_of[3][0]].cores)) == 3


def test_the_blocks_of_a_task_without_sync_start_are_independent():
    # The same capture with task 3 not sync-start: its blocks need not wait for
    # one another, and best reads shorter.
    cap = Capture().task(1, AIV0).task(2, AIV0).task(3, AIV0, block_num=3)
    cap.row(cap.aiv(0, 1), 1, 0, 20).row(cap.aiv(0, 2), 2, 0, 20)
    for core in (cap.aiv(1, 1), cap.aiv(1, 2), cap.aiv(0, 1)):
        cap.row(core, 3, 20, 30)
    assert len(cap.workload().units_of[3]) == 3
    assert numbers(cap)["best"] == 20


def test_a_predicated_task_without_rows_keeps_the_dag_intact():
    cap = Capture().task(1, AIC).task(2, AIV0, predicated=True).task(3, AIC).edge(1, 2).edge(2, 3)
    cap.row(0, 1, 0, 10).row(0, 3, 15, 25)
    assert numbers(cap)["squeezed"] == 20


def test_a_skipped_task_holds_no_core_in_any_schedule():
    # One cluster, every core busy from 0 to 10. 3 waits only for two skipped
    # tasks, an AIC task and a two-block sync-start AIV task, which the runtime
    # retired at once without a core; no schedule may hold 3 back for a core.
    cap = Capture(clusters=1).task(1, AIC).task(2, AIV0).task(3, AIV1)
    cap.task(4, AIC, predicated=True).task(5, AIV0, block_num=2, sync_start=True, predicated=True)
    cap.edge(4, 3).edge(5, 3)
    cap.row(0, 1, 0, 10).row(cap.aiv(0, 1), 2, 0, 10).row(cap.aiv(0, 2), 3, 0, 10)
    w = cap.workload()
    assert w.units_of[4] == w.units_of[5] == []
    assert w.skipped == [4, 5]
    assert set(numbers(cap).values()) == {10}
    (report,) = sb.benchmark(cap.records(), cap.deps())
    assert "2 predicated task(s) left no AICore row and count as skipped" in sb.render(report, "case")


def test_a_predicated_task_missing_some_of_its_blocks_rows_is_refused():
    # Two blocks, one row: no predicate skips half a task, so a row was lost.
    cap = Capture().task(1, AIC, block_num=2, predicated=True).row(0, 1, 0, 10)
    with pytest.raises(sb.BenchmarkError, match="1 captured rows where 2 block.*incomplete"):
        cap.workload()


COMPLETE = {
    "verdict": "published",
    "processing_complete": True,
    "handoff_identity_verified": True,
    **dict.fromkeys(sb.LOSS_COUNTS, 0),
}


def test_a_skip_in_a_capture_that_reports_a_complete_collection_is_read():
    cap = Capture().task(1, AIC).task(2, AIC, predicated=True).row(0, 1, 0, 10)
    cap.collection = COMPLETE
    (report,) = sb.benchmark(cap.records(), cap.deps())
    assert report["skipped"] == 1


@pytest.mark.parametrize(
    "change, error",
    [
        ({"verdict": "partial_safe"}, "verdict partial_safe"),
        ({"processing_complete": False}, "processing_complete false"),
        ({"handoff_identity_verified": False}, "handoff_identity_verified false"),
        *(({k: 2}, f"{k} 2") for k in sb.LOSS_COUNTS),
    ],
)
def test_a_capture_that_reports_lost_records_is_refused(change, error):
    # The predicated task without rows would otherwise count as skipped.
    cap = Capture().task(1, AIC).task(2, AIC, predicated=True).row(0, 1, 0, 10)
    cap.collection = {**COMPLETE, **change}
    with pytest.raises(sb.BenchmarkError, match=f"incomplete collection: {error}"):
        sb.benchmark(cap.records(), cap.deps())


def test_a_skipped_task_the_runtime_sent_to_a_core_keeps_its_row():
    # a5 host_build_graph hands a task whose predicate failed to a core, which
    # returns at once: its brief row is time that core spent.
    cap = Capture(clusters=1).task(1, AIC, predicated=True).task(2, AIC).edge(1, 2)
    cap.row(0, 1, 0, 2).row(0, 2, 2, 12)
    assert len(cap.workload().units_of[1]) == 1
    assert set(numbers(cap).values()) == {12}


def test_best_is_never_longer_than_another_schedule_found():
    # One cluster. The b-level search runs the 1 -> 2 chain first and the MIX
    # task 3 last, 18 cycles; the run started 3 first and took the optimum, 16.
    cap = Capture(clusters=1).task(1, AIC, block_num=2).task(2, AIC, block_num=2).task(3, MIX).edge(1, 2)
    cap.row(0, 3, 0, 1).row(cap.aiv(0, 1), 3, 0, 3).row(cap.aiv(0, 2), 3, 0, 3)
    cap.row(0, 1, 1, 4).row(0, 1, 4, 8).row(0, 2, 8, 11).row(0, 2, 11, 16)
    w = cap.workload()
    assert sb.makespan(w, checked(w, sb.best)) == 18
    t = numbers(cap)
    assert t["best"] == t["squeezed"] == t["measured"] == 16


def test_a_long_chain_of_kernel_less_tasks_is_walked_without_recursion():
    # Deeper than Python's recursion limit allows a walk that recurses per task.
    cap = Capture().task(1, AIC).task(602, AIV0)
    cap.row(0, 1, 0, 10).row(cap.aiv(0, 1), 602, 10, 15)
    for tid in range(2, 602):
        cap.task(tid, NONE).edge(tid - 1, tid)
    cap.edge(601, 602)
    assert set(numbers(cap).values()) == {15}


def test_a_mix_tasks_rows_pair_into_blocks_by_start_on_every_lane():
    # Two 8-cycle blocks of a MIX task on one cluster, one long on its AIC and
    # the other on its AIVs. The capture lists the AIC rows out of order.
    cap = Capture(clusters=1).task(1, MIX, block_num=2)
    cap.row(0, 1, 8, 16).row(0, 1, 0, 2)
    for lane in (1, 2):
        cap.row(cap.aiv(0, lane), 1, 0, 8).row(cap.aiv(0, lane), 1, 8, 10)
    t = numbers(cap)
    assert {t[name] for name in ("measured", "squeezed", "best", "fifo", "sameOrder")} == {16}


def test_squeezed_keeps_the_captured_cores_and_their_order():
    # Two independent tasks ran back to back on one core; squeezed keeps them
    # serial even though another core was free, and the references do not.
    cap = Capture().task(1, AIC).task(2, AIC)
    cap.row(0, 1, 0, 10).row(0, 2, 10, 20)
    t = numbers(cap)
    assert t["squeezed"] == t["measured"] == 20
    assert t["fifo"] == t["sameOrder"] == t["best"] == 10


def test_best_runs_first_the_chain_head_that_the_run_and_fifo_delayed():
    # The capture started the two independent tasks first, which pushes the
    # 3 -> 4 chain back: sameOrder keeps that, best runs the chain head first.
    cap = Capture().task(1, AIC).task(2, AIC).task(3, AIC).task(4, AIC).edge(3, 4)
    cap.row(0, 1, 0, 10).row(1, 2, 0, 10).row(0, 3, 10, 20).row(0, 4, 20, 30)
    t = numbers(cap)
    assert t["sameOrder"] == t["fifo"] == 30
    assert t["best"] == t["bound"] == 20


def order_capture() -> Capture:
    """One AIC core; 1 (10) -> 2 (1), and 5 (5) independent. The run ran 1, 2, 5."""
    cap = Capture(clusters=1).task(1, AIC).task(2, AIC).task(5, AIC).edge(1, 2)
    return cap.row(0, 1, 0, 10).row(0, 2, 10, 11).row(0, 5, 11, 16)


def test_fifo_runs_the_unit_that_became_ready_first():
    w = order_capture().workload()
    s = checked(w, sb.fifo)
    assert s[w.units_of[5][0]].start == 10, "5 was ready at 0, before 2 at 10"
    assert s[w.units_of[2][0]].start == 15


def test_same_order_runs_in_the_captured_start_order():
    w = order_capture().workload()
    s = checked(w, sb.same_order)
    assert s[w.units_of[2][0]].start == 10
    assert s[w.units_of[5][0]].start == 11


def test_fifo_prefers_the_earlier_ready_time_to_the_lower_task_id():
    # Two AIC cores. 1 (10) -> 2; 3 (1) -> 4 (1) -> 5; 6 (20) independent. At 10,
    # 4 has been ready since 1 and 2 only since 10: FIFO starts 4.
    cap = Capture().task(1, AIC).task(2, AIC).task(3, AIC).task(4, AIC).task(5, AIC).task(6, AIC)
    cap.edge(1, 2).edge(3, 4).edge(4, 5)
    cap.row(0, 1, 0, 10).row(1, 3, 0, 1).row(1, 4, 1, 2).row(1, 6, 2, 22).row(0, 5, 10, 11).row(0, 2, 11, 12)
    w = cap.workload()
    assert checked(w, sb.fifo)[w.units_of[4][0]].start == 10


def test_a_sync_start_mix_task_takes_one_whole_cluster_per_block():
    cap = Capture().task(1, MIX, block_num=2, sync_start=True)
    for cluster in (0, 1):
        cap.row(cluster, 1, 0, 10).row(cap.aiv(cluster, 1), 1, 0, 10).row(cap.aiv(cluster, 2), 1, 0, 10)
    assert numbers(cap)["best"] == 10


def test_squeezed_starts_a_mix_blocks_slots_together_even_if_the_run_did_not():
    # The run started the MIX block's AIV0 slot only once the AIV0 task freed it,
    # its other slots at 0. Started together, the block ends at 15, after the
    # run's 10: dispatch cost reads negative, and the report still stands.
    cap = Capture(clusters=1).task(1, AIV0).task(2, MIX)
    cap.row(cap.aiv(0, 1), 1, 0, 5).row(0, 2, 0, 10).row(cap.aiv(0, 1), 2, 5, 6).row(cap.aiv(0, 2), 2, 0, 1)
    t = numbers(cap)
    assert (t["measured"], t["squeezed"]) == (10, 15)


def test_an_edge_from_no_task_constrains_nothing():
    # The host's own buffers reach deps.json as creator edges from no task.
    cap = Capture().task(1, AIC).edge(12345, 1).row(0, 1, 5, 15)
    assert numbers(cap)["squeezed"] == 10


def test_a_graph_body_runs_between_its_producers_and_consumers():
    cap = Capture().task(1, AIC).task(9, NONE).task(2, AIC).edge(1, 9).edge(9, 2)
    body = {"index": 0, "kernel_ids": AIV0, "block_num": 1, "sync_start": False, "predicated": False}
    cap.graph_definitions = [{"definition": 0, "tasks": [body, {**body, "index": 1}], "edges": [[0, 1]]}]
    cap.graph_shells = [{"task_id": "9", "definition": 0}]
    b0, b1 = sub_task(9, 0), sub_task(9, 1)
    cap.row(0, 1, 0, 10).row(cap.aiv(0, 1), b0, 10, 15).row(cap.aiv(0, 1), b1, 15, 20).row(0, 2, 20, 30)
    w = cap.workload()
    s = checked(w, sb.fifo)
    assert s[w.units_of[b0][0]].start == 10, "the body's root waits for the Graph task's producer"
    assert s[w.units_of[2][0]].start == 20, "the Graph task's consumer waits for the body's sink"
    assert numbers(cap)["best"] == 30


def test_an_empty_graph_body_still_orders_its_producers_before_its_consumers():
    cap = Capture().task(1, AIC).task(9, NONE).task(2, AIC).edge(1, 9).edge(9, 2)
    cap.graph_definitions = [{"definition": 0, "tasks": [], "edges": []}]
    cap.graph_shells = [{"task_id": "9", "definition": 0}]
    cap.row(0, 1, 0, 10).row(0, 2, 10, 20)
    t = numbers(cap)
    assert t["best"] == t["bound"] == t["fifo"] == 20


# ---------------------------------------------------------------------------
# best and bound
# ---------------------------------------------------------------------------


def tie_capture() -> Capture:
    """Two AIC cores; A=1 -> D=2, and independent B=3, C=2. The optimum, 4,
    starts A and C together and B after A: core 0 runs A B, core 1 runs C D."""
    cap = Capture().task(1, AIC).task(2, AIC).task(3, AIC).task(4, AIC).edge(1, 4)
    return cap.row(0, 1, 0, 1).row(0, 2, 1, 4).row(0, 3, 4, 6).row(0, 4, 6, 8)


def test_forward_backward_repairs_a_b_level_tie():
    # b-levels: A 3, B 3, C 2, D 2. b-level alone starts A and B, then C wins
    # its tie with D, so D waits for B: 5. One backward pass mirrors to
    # starting A and C first: 4, the optimum.
    w = tie_capture().workload()
    assert sb.b_levels(w) == {1: 3, 2: 3, 3: 2, 4: 2}
    assert sb.makespan(w, checked(w, sb.best)) == 4 == brute_force(w)


def test_bound_is_work_over_cores_when_that_dominates():
    w = tie_capture().workload()
    assert sb.bound(w) == 4


def test_bound_counts_work_after_its_earliest_start():
    # X=4 feeds Y1..Y3 (2 each) on two cores. The critical path is 6 and the
    # work over two cores 5, but the Ys cannot start before 4 and need 6 / 2
    # more: 7. The optimum is 8, so the bound is valid without being tight.
    cap = Capture().task(1, AIC).row(0, 1, 0, 4)
    for tid in (2, 3, 4):
        cap.task(tid, AIC).edge(1, tid).row(0, tid, 4 + 2 * (tid - 2), 6 + 2 * (tid - 2))
    w = cap.workload()
    assert sb.bound(w) == 7
    assert brute_force(w) == 8 == sb.makespan(w, checked(w, sb.best))


def test_bound_counts_work_before_its_latest_end():
    # The mirror of the head case: Y1..Y3 (2 each) -> Z (4) on two cores. The
    # Ys need 6 / 2 and Z's 4 still follows the last of them: 7; the optimum is 8.
    cap = Capture()
    for tid in (1, 2, 3):
        cap.task(tid, AIC).edge(tid, 9).row(0, tid, 2 * (tid - 1), 2 * tid)
    cap.task(9, AIC).row(0, 9, 6, 10)
    w = cap.workload()
    assert sb.bound(w) == 7
    assert brute_force(w) == 8


def test_bound_rounds_the_work_over_the_cores_up():
    cap = Capture()
    for tid in (1, 2, 3):
        cap.task(tid, AIC).row(0, tid, 3 * (tid - 1), 3 * tid)
    assert sb.bound(cap.workload()) == 5


def test_segments_on_different_core_types_add_up():
    # One cluster: two AIC tasks of 10, a barrier, then four AIV tasks of 10 on
    # the two AIV cores. Each side needs 20, and neither can overlap the other.
    cap = Capture(clusters=1).task(1, AIC).task(2, AIC).task(5, NONE).edge(1, 5).edge(2, 5)
    cap.row(0, 1, 0, 10).row(0, 2, 10, 20)
    for i, tid in enumerate((6, 7, 8, 9)):
        cap.task(tid, AIV0).edge(5, tid).row(cap.aiv(0, 1 + i % 2), tid, 20 + 10 * (i // 2), 30 + 10 * (i // 2))
    assert sb.bound(cap.workload()) == 40


def test_a_barrier_splits_the_bound_into_segments():
    # A kernel-less task every other task precedes or follows: no schedule can
    # overlap its two sides, so the bound adds theirs up.
    cap = Capture().task(1, AIC).task(2, AIC).task(5, NONE).task(3, AIC).task(4, AIC)
    cap.edge(1, 5).edge(2, 5).edge(5, 3).edge(5, 4)
    cap.row(0, 1, 0, 10).row(1, 2, 0, 4).row(0, 3, 10, 14).row(1, 4, 10, 20)
    w = cap.workload()
    assert len(sb.barrier_segments(w)) == 2
    assert sb.bound(w) == 20 == brute_force(w)


def _options(w, unit):
    """Every placement a unit can take: per block a core, or a cluster for MIX."""
    topo = w.topology
    per_block = []
    for block in unit.blocks:
        if w.dag.tasks[unit.task].is_mix:
            per_block.append([[topo.core_at(c, m.slot) for m in block] for c in range(topo.clusters)])
        else:
            per_block.append([[c] for c in topo.cores_of("aic" if block[0].slot == 0 else "aiv")])
    return [
        cores
        for combo in itertools.product(*per_block)
        for cores in [[c for part in combo for c in part]]
        if len(set(cores)) == len(cores)
    ]


def brute_force(w):
    """Minimum makespan by serial schedule generation over every precedence-
    feasible unit order and every placement choice: that space contains an
    optimal schedule, so its minimum is the optimum."""
    units = w.units
    preds = sb._real_preds(w)
    spans = []
    options = [_options(w, u) for u in units]
    for order in itertools.permutations(range(len(units))):
        pos = {u: i for i, u in enumerate(order)}
        if any(pos[pu] > pos[u.uid] for u in units for p in preds.get(u.task, ()) for pu in w.units_of[p]):
            continue
        for choice in itertools.product(*(options[u] for u in order)):
            busy, end_of = {}, {}
            for uid, cores in zip(order, choice):
                unit = units[uid]
                ready = max((end_of[pu] for p in preds.get(unit.task, ()) for pu in w.units_of[p]), default=0)
                for t in sorted({ready} | {e for c in cores for _, e in busy.get(c, []) if e >= ready}):
                    if all(
                        all(t + m.duration <= s or t >= e for s, e in busy.get(c, []))
                        for m, c in zip(unit.members, cores)
                        if m.duration > 0
                    ):
                        break
                for m, c in zip(unit.members, cores):
                    if m.duration > 0:
                        busy.setdefault(c, []).append((t, t + m.duration))
                end_of[uid] = t + unit.duration()
            spans.append(max(end_of.values(), default=0))
    return min(spans)


def random_capture(seed):
    """A small random graph -- AIC, AIV, MIX, SPMD, sync-start, kernel-less and
    skipped tasks -- and a serial capture of it: every task runs in id order,
    which is a valid execution of any DAG whose edges go up in id. At most four
    units, which keeps the exhaustive search to a fraction of a second."""
    rng = random.Random(seed)
    cap = Capture(clusters=rng.choice([1, 2]))
    t = units = 0
    for tid in itertools.count(1):
        shape = rng.choice(("aic", "aiv", "mix", "spmd", "sync", "none", "skip"))
        if shape == "spmd" and units + 2 > 4:
            shape = "aic"
        units += {"spmd": 2, "none": 0, "skip": 0}.get(shape, 1)
        if units > 4 or (units >= 3 and rng.random() < 0.3):
            break
        dur = rng.randint(1, 4)
        if shape == "aic":
            cap.task(tid, AIC).row(0, tid, t, t + dur)
        elif shape == "aiv":
            cap.task(tid, AIV0).row(cap.aiv(0, 1), tid, t, t + dur)
        elif shape == "mix":
            cap.task(tid, MIX)
            for core in (0, cap.aiv(0, 1), cap.aiv(0, 2)):
                cap.row(core, tid, t, t + dur)
        elif shape == "spmd":
            cap.task(tid, AIV0, block_num=2)
            cap.row(cap.aiv(0, 1), tid, t, t + dur).row(cap.aiv(0, 1), tid, t + dur, t + 2 * dur)
            dur *= 2
        elif shape == "sync":
            cap.task(tid, AIV0, block_num=2, sync_start=True)
            cap.row(cap.aiv(0, 1), tid, t, t + dur).row(cap.aiv(0, 2), tid, t, t + dur)
        else:  # no kernel, or a skipped one: no rows, no time
            cap.task(tid, NONE if shape == "none" else AIC, predicated=shape == "skip")
            dur = 0
        for p in range(1, tid):
            if rng.random() < 0.3:
                cap.edge(p, tid)
        t += dur
    return cap


def rich_capture(seed: int) -> Capture:
    """A larger random graph holding every shape the model knows -- AIC, AIV,
    two- and three-slot MIX with unequal slots, SPMD, sync-start, kernel-less,
    skipped-predicate and Graph tasks -- and a run of it: every block starts
    once its producers have ended and its cores are free, after a random gap."""
    rng = random.Random(seed)
    cap = Capture(clusters=2)
    free = collections.defaultdict(int)  # core -> when it frees
    end: dict[int, int] = {}  # task -> when its last block ended
    aic, aiv = list(range(cap.clusters)), [cap.aiv(c, lane) for c in range(cap.clusters) for lane in (1, 2)]

    def run(tid, kernel_ids, blocks, ready, sync=False):
        """Rows for `blocks` blocks of `tid`; returns when the last one ends."""
        slots = [s for s, k in enumerate(kernel_ids) if k >= 0]
        groups = []  # per block: [(core, duration)]
        for _ in range(blocks):
            taken = {c for g in groups for c, _ in g} if sync else set()
            if len(slots) > 1:
                clusters = [c for c in range(cap.clusters) if c not in {cap_cluster(cap, x) for x in taken}]
                cluster = min(clusters, key=lambda c: max(free[slot_core(cap, c, s)] for s in slots))
                groups.append([(slot_core(cap, cluster, s), rng.randint(1, 8)) for s in slots])
            else:
                pool = [c for c in (aic if slots[0] == 0 else aiv) if c not in taken]
                groups.append([(min(pool, key=lambda c: free[c]), rng.randint(1, 8))])
        last = ready
        for group in [[m for g in groups for m in g]] if sync else groups:
            start = max([ready] + [free[c] for c, _ in group]) + rng.randint(0, 3)
            for core, d in group:
                cap.row(core, tid, start, start + d)
                free[core] = start + d
                last = max(last, start + d)
        return last

    tids = []
    for tid in range(1, 26):
        preds = rng.sample(tids, min(len(tids), rng.randint(0, 2)))
        ready = max((end[p] for p in preds), default=0)
        shape = rng.choice(("aic", "aiv", "aiv1", "mix", "mix2", "spmd", "sync", "sync_mix", "none", "skip", "graph"))
        if shape == "graph":
            body = [(AIC, 1), (MIX, 1), ([-1, 0, -1], 2)]
            cap.task(tid, NONE)
            cap.graph_definitions.append(
                {
                    "definition": len(cap.graph_definitions),
                    "tasks": [
                        {"index": i, "kernel_ids": k, "block_num": n, "sync_start": False, "predicated": False}
                        for i, (k, n) in enumerate(body)
                    ],
                    "edges": [[0, 1], [0, 2]],
                }
            )
            cap.graph_shells.append({"task_id": str(tid), "definition": len(cap.graph_definitions) - 1})
            root = run(sub_task(tid, 0), AIC, 1, ready)
            end[tid] = max(run(sub_task(tid, i), k, n, root) for i, (k, n) in enumerate(body) if i)
        else:
            kernel_ids, blocks, sync, attrs = {
                "aic": (AIC, 1, False, {}),
                "aiv": (AIV0, 1, False, {}),
                "aiv1": (AIV1, 1, False, {}),
                "mix": (MIX, 2, False, {}),
                "mix2": ([0, 0, -1], 1, False, {}),
                "spmd": (AIV0, 3, False, {}),
                "sync": (AIV0, 2, True, {"sync_start": True}),
                "sync_mix": (MIX, 2, True, {"sync_start": True}),
                "none": (NONE, 1, False, {}),
                "skip": (AIC, 1, False, {"predicated": True}),
            }[shape]
            cap.task(tid, kernel_ids, block_num=blocks, **attrs)
            skipped = shape in ("none", "skip")
            end[tid] = ready if skipped else run(tid, kernel_ids, blocks, ready, sync)
        for p in preds:
            cap.edge(p, tid)
        tids.append(tid)
    return cap


def cap_cluster(cap: Capture, core: int) -> int:
    return core if core < cap.clusters else (core - cap.clusters) // 2


def slot_core(cap: Capture, cluster: int, slot: int) -> int:
    return cluster if slot == 0 else cap.aiv(cluster, slot)


# The numbers of rich_capture(seed), pinned: a change to any of them is a change
# to what the model computes.
REFERENCE = {
    0: {"measured": 61, "squeezed": 50, "best": 38, "bound": 31, "fifo": 48, "sameOrder": 47},
    1: {"measured": 137, "squeezed": 101, "best": 77, "bound": 60, "fifo": 77, "sameOrder": 83},
    2: {"measured": 128, "squeezed": 99, "best": 73, "bound": 55, "fifo": 81, "sameOrder": 82},
    3: {"measured": 116, "squeezed": 90, "best": 62, "bound": 49, "fifo": 75, "sameOrder": 77},
    4: {"measured": 126, "squeezed": 95, "best": 64, "bound": 52, "fifo": 67, "sameOrder": 66},
    5: {"measured": 105, "squeezed": 81, "best": 71, "bound": 53, "fifo": 74, "sameOrder": 86},
}


@pytest.mark.parametrize("seed", sorted(REFERENCE))
def test_a_rich_run_reads_the_reference_numbers(seed):
    assert numbers(rich_capture(seed)) == REFERENCE[seed]


@pytest.mark.parametrize("seed", range(60))
def test_bound_and_best_bracket_the_optimum(seed):
    cap = random_capture(seed)
    w = cap.workload()
    t = numbers(cap)
    assert t["bound"] <= brute_force(w) <= t["best"] <= min(t["fifo"], t["sameOrder"], t["squeezed"])
    assert t["squeezed"] <= t["measured"]


# ---------------------------------------------------------------------------
# The validator rejects each broken constraint
# ---------------------------------------------------------------------------


def _placed(cap):
    w = cap.workload()
    return w, {uid: sb.Placement(p.start, list(p.cores)) for uid, p in sb.fifo(w).items()}


def test_validator_rejects_a_consumer_starting_before_its_producer_ends():
    cap = Capture().task(1, AIC).task(2, AIV0).edge(1, 2)
    cap.row(0, 1, 0, 10).row(cap.aiv(0, 1), 2, 10, 15)
    w, s = _placed(cap)
    s[w.units_of[2][0]].start = 9
    with pytest.raises(sb.BenchmarkError, match="before its producers"):
        sb.validate(w, s, "t")


def test_validator_rejects_a_consumer_starting_before_a_producer_behind_a_kernel_less_task():
    cap = Capture().task(1, AIC).task(5, NONE).task(2, AIV0).edge(1, 5).edge(5, 2)
    cap.row(0, 1, 0, 10).row(cap.aiv(0, 1), 2, 10, 15)
    w, s = _placed(cap)
    s[w.units_of[2][0]].start = 9
    with pytest.raises(sb.BenchmarkError, match="before its producers"):
        sb.validate(w, s, "t")


def test_a_schedule_shorter_than_the_bound_is_refused(monkeypatch):
    cap = Capture().task(1, AIC).row(0, 1, 0, 10)
    monkeypatch.setattr(sb, "bound", lambda workload: 11)
    with pytest.raises(sb.BenchmarkError, match="shorter than the bound"):
        sb.benchmark(cap.records(), cap.deps())


def test_validator_rejects_a_mix_block_across_clusters():
    cap = Capture().task(1, MIX)
    cap.row(0, 1, 0, 10).row(cap.aiv(0, 1), 1, 0, 10).row(cap.aiv(0, 2), 1, 0, 10)
    w, s = _placed(cap)
    s[w.units_of[1][0]].cores[1] = cap.aiv(1, 1)
    with pytest.raises(sb.BenchmarkError, match="spans clusters"):
        sb.validate(w, s, "t")


def four_shapes():
    """Independent AIC (1), AIV (2), MIX (3) and AIC (4) tasks, and the fifo
    schedule of them: 1, 2 and 3 (on cluster 1) at 0, then 4 on core 0 at 10."""
    cap = Capture().task(1, AIC).task(2, AIV0).task(3, MIX).task(4, AIC)
    cap.row(0, 1, 0, 10).row(cap.aiv(0, 1), 2, 0, 10).row(0, 4, 10, 20)
    for core in (1, cap.aiv(1, 1), cap.aiv(1, 2)):
        cap.row(core, 3, 0, 10)
    return _placed(cap)


@pytest.mark.parametrize(
    "task, placement, error",
    [
        pytest.param(1, None, "3 units placed, 4 expected", id="a unit left out"),
        pytest.param(1, (-1, [0]), "starts at -1", id="a start before 0"),
        pytest.param(1, (0, []), "0 cores for 1 members", id="a block without its core"),
        pytest.param(1, (0, [99]), "placed on core 99", id="a core outside the topology"),
        pytest.param(1, (0, [3]), "placed on core 3", id="an AIC block on an AIV core"),
        pytest.param(2, (0, [1]), "placed on core 1", id="an AIV block on an AIC core"),
        pytest.param(3, (0, [1, 5, 4]), "placed on core 5", id="a MIX slot on another lane"),
        pytest.param(4, (9, [0]), "core 0 runs units", id="one cycle of overlap"),
    ],
)
def test_validator_rejects_each_broken_constraint(task, placement, error):
    w, s = four_shapes()
    sb.validate(w, s, "t")  # as placed, it is valid: 4 starts the cycle 1 ends
    (uid,) = w.units_of[task]
    if placement is None:
        del s[uid]
    else:
        s[uid] = sb.Placement(*placement)
    with pytest.raises(sb.BenchmarkError, match=error):
        sb.validate(w, s, "t")


def test_validator_rejects_two_units_on_one_core_at_once():
    cap = Capture().task(1, AIC).task(2, AIC)
    cap.row(0, 1, 0, 10).row(1, 2, 0, 10)
    w, s = _placed(cap)
    s[w.units_of[2][0]].cores = [s[w.units_of[1][0]].cores[0]]
    with pytest.raises(sb.BenchmarkError, match="at once"):
        sb.validate(w, s, "t")


# ---------------------------------------------------------------------------
# What the inputs may and may not be
# ---------------------------------------------------------------------------


def test_a_tensormap_and_ringbuffer_run_is_measured_the_same_way():
    # Its alloc tasks run no kernel and leave no row; ids are its own layout.
    alloc, a, b = (TmrTaskId.make(ring, local).raw for ring, local in ((0, 0), (1, 0), (1, 1)))
    cap = Capture(runtime="tensormap_and_ringbuffer").task(alloc, NONE).task(a, AIC).task(b, AIV0)
    cap.edge(alloc, a).edge(a, b)
    cap.row(0, a, 0, 10).row(cap.aiv(0, 1), b, 20, 25)
    (report,) = sb.benchmark(cap.records(), cap.deps())
    assert report["runtime"] == "tensormap_and_ringbuffer"
    assert round(report["time_us"]["squeezed"]) == 15


def test_graph_tasks_are_refused_outside_host_build_graph():
    cap = Capture(runtime="tensormap_and_ringbuffer").task(9, NONE)
    cap.graph_definitions = [{"definition": 0, "tasks": [], "edges": []}]
    cap.graph_shells = [{"task_id": "9", "definition": 0}]
    with pytest.raises(sb.BenchmarkError, match="does not have"):
        sb.load_dag(cap.deps())


def test_each_run_of_a_capture_is_reported_on_its_own():
    cap = Capture().task(1, AIC).task(2, AIC).edge(1, 2)
    cap.row(0, 1, 0, 10, run=1).row(0, 2, 15, 25, run=1)
    cap.row(0, 1, 500, 510, run=2).row(0, 2, 510, 520, run=2)
    reports = sb.benchmark(cap.records(), cap.deps())
    assert [r["run"] for r in reports] == [1, 2]
    assert [round(r["time_us"]["measured"]) for r in reports] == [25, 20]
    assert [round(r["time_us"]["squeezed"]) for r in reports] == [20, 20]


@pytest.mark.parametrize("columns", [5, 6])
def test_a_row_without_a_run_epoch_is_refused(columns):
    cap = Capture().task(1, AIC).row(0, 1, 0, 10)
    cap.rows[0] = cap.rows[0][:columns]
    with pytest.raises(sb.BenchmarkError, match=f"has {columns} columns; the current format has 7"):
        sb.benchmark(cap.records(), cap.deps())


@pytest.mark.parametrize("field", sb.TASK_FIELDS)
def test_a_deps_task_without_a_current_field_is_refused(field):
    cap = Capture().task(1, AIC).row(0, 1, 0, 10)
    del cap.tasks[0][field]
    with pytest.raises(sb.BenchmarkError, match=f"lacks {field}"):
        sb.benchmark(cap.records(), cap.deps())


def test_a_graph_body_task_without_a_current_field_is_refused():
    cap = Capture().task(9, NONE)
    body = {"index": 0, "kernel_ids": AIC, "block_num": 1}
    cap.graph_definitions = [{"definition": 0, "tasks": [body], "edges": []}]
    cap.graph_shells = [{"task_id": "9", "definition": 0}]
    with pytest.raises(sb.BenchmarkError, match="lacks sync_start, predicated"):
        sb.load_dag(cap.deps())


def test_a_capture_without_its_runtime_is_refused():
    cap = Capture().task(1, AIC).row(0, 1, 0, 10)
    records = cap.records()
    del records["metadata"]["runtime"]
    with pytest.raises(KeyError, match="runtime"):
        sb.benchmark(records, cap.deps())


def test_a_capture_from_another_runtime_than_deps_is_refused():
    cap = Capture().task(1, AIC).row(0, 1, 0, 10)
    records = cap.records()
    records["metadata"]["runtime"] = "tensormap_and_ringbuffer"
    with pytest.raises(sb.BenchmarkError, match="the capture is tensormap_and_ringbuffer but deps.json is"):
        sb.benchmark(records, cap.deps())


@pytest.mark.parametrize("epoch", [-1, 1.5, "1"])
def test_a_run_epoch_that_is_not_a_nonnegative_integer_is_refused(epoch):
    cap = Capture().task(1, AIC).row(0, 1, 0, 10)
    cap.rows[0][6] = epoch
    with pytest.raises(sb.BenchmarkError, match="invalid run_epoch"):
        sb.benchmark(cap.records(), cap.deps())


def test_a_capture_missing_a_tasks_rows_is_refused():
    cap = Capture().task(1, AIC).task(2, AIC).row(0, 1, 0, 10)
    with pytest.raises(sb.BenchmarkError, match="has no predicate, but left no captured rows"):
        cap.workload()


def test_a_block_missing_one_of_its_rows_is_refused_as_incomplete():
    # A MIX block whose AIV1 row was never recorded.
    cap = Capture().task(1, MIX)
    cap.row(0, 1, 0, 10).row(cap.aiv(0, 1), 1, 0, 10)
    with pytest.raises(sb.BenchmarkError, match="2 captured rows where 1 block.*incomplete"):
        cap.workload()


def test_graph_rows_without_bodies_in_deps_are_refused_by_name():
    # A deps.json that knows the Graph task only as one kernel-less task.
    cap = Capture().task(9, NONE)
    cap.row(cap.aiv(0, 1), sub_task(9, 0), 0, 5)
    with pytest.raises(sb.BenchmarkError, match="Graph body tasks.*graph_definitions"):
        cap.workload()


def test_a_row_that_ends_before_it_starts_is_refused():
    cap = Capture().task(1, AIC).row(0, 1, 10, 5)
    with pytest.raises(sb.BenchmarkError, match="ends before it starts"):
        sb.benchmark(cap.records(), cap.deps())


@pytest.mark.parametrize("hz", [0, -1, 1.5, "1000000"])
def test_a_clock_that_is_not_a_positive_integer_is_refused(hz):
    cap = Capture().task(1, AIC).row(0, 1, 0, 10)
    records = cap.records()
    records["metadata"]["clock_freq_hz"] = hz
    with pytest.raises(sb.BenchmarkError, match="clock_freq_hz"):
        sb.benchmark(records, cap.deps())


def test_a_graph_body_edge_to_a_task_the_body_lacks_is_refused():
    cap = Capture().task(9, NONE)
    cap.graph_definitions = [{"definition": 0, "tasks": [{"index": 0, "kernel_ids": AIC}], "edges": [[0, 5]]}]
    cap.graph_shells = [{"task_id": "9", "definition": 0}]
    with pytest.raises(sb.BenchmarkError, match="edge 0 -> 5 to a task it lacks"):
        sb.load_dag(cap.deps())


def test_a_deps_json_without_a_known_runtime_is_refused():
    cap = Capture(runtime="").task(1, AIC).row(0, 1, 0, 10)
    with pytest.raises(sb.BenchmarkError, match="runtime"):
        sb.benchmark(cap.records(), cap.deps())


def test_a_captured_id_no_runtime_mints_is_named_raw():
    corrupt = 3 << HbgTaskId.SPACE_SHIFT
    cap = Capture().task(1, AIC).row(0, 1, 0, 10).row(1, corrupt, 0, 10)
    with pytest.raises(sb.BenchmarkError, match=f"not in deps.json, e.g. {corrupt:#x}$"):
        cap.workload()


def _set_row(records, row):
    records["aicore_tasks"][0] = row


@pytest.mark.parametrize(
    "change, error",
    [
        pytest.param(lambda r, d: d["edges"].append({"pred": "2", "succ": "1"}), "has a cycle", id="cycle"),
        pytest.param(lambda r, d: d["tasks"].append(dict(d["tasks"][1])), "appears twice", id="duplicate_task"),
        pytest.param(lambda r, d: d["edges"].append({"pred": "1", "succ": "7"}), "leads to task 7", id="edge_to_none"),
        pytest.param(lambda r, d: r.update(aicore_tasks=[]), "no AICore rows", id="no_rows"),
        pytest.param(lambda r, d: _set_row(r, [0, 1, 0, 10]), "has 4 columns", id="short_row"),
        pytest.param(
            lambda r, d: r["metadata"].update(core_types=["aic", "aic", "aiv", "aiv", "aiv"]),
            "do not form clusters",
            id="no_clusters",
        ),
        pytest.param(
            lambda r, d: r["metadata"].update(core_types=["aiv", "aiv", "aiv", "aiv", "aic", "aic"]),
            "must precede",
            id="aiv_first",
        ),
    ],
)
def test_a_malformed_capture_is_refused_by_what_is_wrong(change, error):
    cap = Capture().task(1, AIC).task(2, AIC).edge(1, 2).row(0, 1, 0, 10).row(0, 2, 10, 20)
    records, deps = cap.records(), cap.deps()
    change(records, deps)
    with pytest.raises(sb.BenchmarkError, match=error):
        sb.benchmark(records, deps)


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


def test_the_report_puts_each_step_between_the_numbers_it_separates():
    cap = Capture().task(1, AIC).task(2, AIC).task(3, AIC).task(4, AIC).edge(3, 4)
    cap.row(0, 1, 0, 10).row(1, 2, 0, 10).row(0, 3, 15, 25).row(0, 4, 30, 40)
    (report,) = sb.benchmark(cap.records(), cap.deps())
    t, step = report["time_us"], report["step_us"]
    assert step["dispatch cost"] == t["measured"] - t["squeezed"]
    assert step["schedule cost"] == t["squeezed"] - t["best"]
    text = sb.render(report, "case")
    assert text.startswith("== case run 1 (host_build_graph; 4 units on 2 clusters)")
    names = ("measured", "dispatch cost", "squeezed", "schedule cost", "best", "bound", "fifo", "sameOrder")
    at = [text.index(name) for name in names]
    assert at == sorted(at), "the chain reads top to bottom, the references below it"


def test_the_cli_prints_the_report_and_refuses_bad_input(tmp_path, capsys):
    cap = Capture().task(1, AIC).row(0, 1, 0, 10)
    records, deps = tmp_path / "records.json", tmp_path / "deps.json"
    records.write_text(json.dumps(cap.records()))
    deps.write_text(json.dumps(cap.deps()))
    assert sb.main([str(records), "--deps", str(deps), "--label", "one"]) == 0
    assert "== one run 1" in capsys.readouterr().out
    deps.write_text(json.dumps({**cap.deps(), "runtime": "unknown"}))
    assert sb.main([str(records), "--deps", str(deps)]) == 2
    assert "error:" in capsys.readouterr().err
    records.write_text(json.dumps({"aicore_tasks": cap.rows}))
    deps.write_text(json.dumps(cap.deps()))
    assert sb.main([str(records), "--deps", str(deps)]) == 2
    assert "lack the field 'metadata'" in capsys.readouterr().err
    records.write_text(json.dumps(cap.records()))
    deps.write_text(json.dumps({**cap.deps(), "tasks": [{**cap.tasks[0], "block_num": "x"}]}))
    assert sb.main([str(records), "--deps", str(deps)]) == 2
    assert "wrong type or value" in capsys.readouterr().err
    deps.write_text(json.dumps(cap.deps()))
    records.write_text("{")
    assert sb.main([str(records), "--deps", str(deps)]) == 2
    assert "error:" in capsys.readouterr().err
    assert sb.main([str(tmp_path / "missing.json"), "--deps", str(deps)]) == 2
    assert "error:" in capsys.readouterr().err
