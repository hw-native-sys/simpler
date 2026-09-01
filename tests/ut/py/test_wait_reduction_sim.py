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

from simpler_setup.tools.wait_reduction_sim import full_reduction, load_wait_graph, online_bitmap, simulate


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


def test_simulate_reports_window_cross_ring_and_resource_reductions(tmp_path):
    ring1_b = str(1 << 32)
    ring1_c = str((1 << 32) + 1)
    data = {
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
    assert window["cross_ring_redundant_wait_pairs"] == 1
    assert window["cross_ring_removed"] == 1
    assert window["cross_ring_misses"] == 0
    assert window["estimated_readiness_fanout_nodes_removed"] == 1
    assert window["estimated_dep_pool_entries_removed"] == 1


def test_load_wait_graph_inserts_hidden_alloc_before_first_consumer(tmp_path):
    data = {
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
