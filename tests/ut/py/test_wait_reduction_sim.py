#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import json

import pytest

from simpler_setup.tools.swimlane_converter import HBG_RUNTIME, TMR_RUNTIME
from simpler_setup.tools.wait_reduction_sim import (
    _scope_key,
    full_reduction,
    load_wait_graph,
    online_bitmap,
    simulate,
)


def _hbg_sub_task(parent_id, local_id):
    """A SUB_TASK id: space 1 in bits 63:62, parent in bits 51:32, index in the low 32."""
    return (1 << 62) | (parent_id << 32) | local_id


def test_scope_key_reads_each_runtimes_own_boundary():
    """A scope is what the bitmap cannot see across, and the two layouts name it differently.

    Under hbg the high word is an id space plus a parent, so comparing it as a ring
    index — which is what this counted before deps.json named its runtime — splits
    tasks of one body from each other and lumps unrelated spaces together.
    """
    # tmr: the ring index alone decides.
    assert _scope_key((1 << 32) | 7, TMR_RUNTIME) == _scope_key((1 << 32) | 9, TMR_RUNTIME)
    assert _scope_key((1 << 32), TMR_RUNTIME) != _scope_key((2 << 32), TMR_RUNTIME)

    # hbg: one modular task's body is one scope, whatever the index within it.
    assert _scope_key(_hbg_sub_task(3, 0), HBG_RUNTIME) == _scope_key(_hbg_sub_task(3, 9), HBG_RUNTIME)
    assert _scope_key(_hbg_sub_task(3, 0), HBG_RUNTIME) != _scope_key(_hbg_sub_task(4, 0), HBG_RUNTIME)
    # Tasks of the run itself share one scope, and it is not any body's.
    assert _scope_key(1, HBG_RUNTIME) == _scope_key(2, HBG_RUNTIME)
    assert _scope_key(1, HBG_RUNTIME) != _scope_key(_hbg_sub_task(0, 1), HBG_RUNTIME)


def test_scope_key_refuses_a_runtime_it_cannot_decode():
    """Guessing would silently miscount cross_scope_*, which is a number people act on."""
    for runtime in (None, "", "host_build_grpah"):
        with pytest.raises(ValueError, match="runtime"):
            _scope_key(_hbg_sub_task(3, 0), runtime)


def test_simulate_counts_cross_scope_edges_by_the_named_runtime(tmp_path):
    """An edge from a modular task into its own body crosses a scope.

    Read as tmr the high word is masked to 8 bits, which drops hbg's id space: a
    GLOBAL task and a sub-task under parent 0 would both come out as ring 0 and the
    edge would look intra-scope. Naming the runtime is what avoids that reading.
    """
    outer = "12"
    inner = str(_hbg_sub_task(0, 0))
    data = {
        "runtime": HBG_RUNTIME,
        "tasks": [{"task_id": t} for t in (outer, inner)],
        "edges": [{"pred": outer, "succ": inner, "source": "creator", "flags": ["wait", "retain"]}],
    }
    path = tmp_path / "deps.json"
    path.write_text(json.dumps(data))

    assert simulate(path, [4])["windows"]["4"]["cross_scope_wait_pairs"] == 1

    # Same graph with the name dropped: refused rather than counted under the wrong
    # layout, since the count is a number people act on.
    data.pop("runtime")
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="runtime"):
        simulate(path, [4])


def _pair_flags(edges) -> dict[tuple[str, str], frozenset[str]]:
    return {edge: frozenset(("wait", "retain")) for edge in edges}


def test_online_bitmap_matches_full_reduction_inside_window():
    order = [str(i) for i in range(5)]
    seq = {task_id: i for i, task_id in enumerate(order)}
    possible_edges = [(str(i), str(j)) for i in range(5) for j in range(i + 1, 5)]

    for mask in range(1 << len(possible_edges)):
        edges = {edge for i, edge in enumerate(possible_edges) if mask & (1 << i)}
        flags = _pair_flags(edges)
        exact = full_reduction(order, flags)
        for bl in (1, 2, 3, 4):
            actual = online_bitmap(order, seq, flags, bl)
            exact_inside_window = {edge for edge in exact if seq[edge[1]] - seq[edge[0]] <= bl}
            assert actual == exact_inside_window


def test_simulate_reports_window_cross_scope_and_resource_reductions(tmp_path):
    ring1_b = str(1 << 32)
    ring1_c = str((1 << 32) + 1)
    data = {
        "runtime": TMR_RUNTIME,
        "tasks": [{"task_id": task_id} for task_id in ("0", ring1_b, ring1_c, "3")],
        "edges": [
            {"pred": "0", "succ": ring1_b, "source": "creator", "flags": ["wait", "retain"]},
            {"pred": ring1_b, "succ": ring1_c, "source": "creator", "flags": ["wait", "retain"]},
            {"pred": "0", "succ": ring1_c, "source": "explicit", "flags": ["wait", "retain"]},
            {"pred": ring1_c, "succ": "3", "source": "creator", "flags": ["wait", "retain"]},
            {"pred": "0", "succ": "3", "source": "creator", "flags": ["wait", "retain"]},
        ],
    }
    path = tmp_path / "deps.json"
    path.write_text(json.dumps(data))

    report = simulate(path, [2])
    window = report["windows"]["2"]
    assert report["full_reduction_upper_bound"] == 2
    assert report["full_retain_classification_uncertain"] == 1
    assert window["removed"] == 1
    assert window["retain_classification_uncertain"] == 1
    assert window["redundant_window_misses"] == 1
    assert window["bitmap_misses_within_window"] == 0
    assert window["cross_scope_redundant_wait_pairs"] == 1
    assert window["cross_scope_removed"] == 1
    assert window["cross_scope_misses"] == 0
    assert window["estimated_readiness_fanout_nodes_removed"] == 1
    assert window["estimated_dep_pool_entries_removed"] == 1


def test_load_wait_graph_inserts_hidden_alloc_before_first_consumer(tmp_path):
    data = {
        "runtime": TMR_RUNTIME,
        "tasks": [{"task_id": "1"}, {"task_id": "2"}],
        "edges": [
            {"pred": "99", "succ": "1", "source": "creator", "flags": ["wait", "retain"]},
            {"pred": "1", "succ": "2", "source": "tensormap", "flags": ["wait"]},
        ],
    }
    path = tmp_path / "deps.json"
    path.write_text(json.dumps(data))

    order, seq, flags, uncertain = load_wait_graph(path)
    assert order == ["99", "1", "2"]
    assert seq == {"99": 0, "1": 1, "2": 2}
    assert flags[("99", "1")] == frozenset(("wait", "retain"))
    assert not uncertain


def test_load_wait_graph_or_accumulates_retain_only_records(tmp_path):
    data = {
        "runtime": TMR_RUNTIME,
        "tasks": [{"task_id": "1"}, {"task_id": "2"}, {"task_id": "3"}],
        "edges": [
            {"pred": "1", "succ": "2", "source": "tensormap", "flags": ["retain"]},
            {"pred": "1", "succ": "2", "source": "tensormap", "flags": ["wait"]},
            {"pred": "2", "succ": "3", "source": "tensormap", "flags": ["retain"]},
        ],
    }
    path = tmp_path / "deps.json"
    path.write_text(json.dumps(data))

    _order, _seq, flags, _uncertain = load_wait_graph(path)
    # A record that does not wait still contributes its retain to the pair, so
    # a reduced ("1", "2") counts as a demotion rather than a pure drop.
    assert flags[("1", "2")] == frozenset(("wait", "retain"))
    # A pair no record waits on is not an edge of the WAIT graph.
    assert ("2", "3") not in flags
