# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Contract tests for simpler_setup.tools.deps_viewer text output."""

import json

from simpler_setup.tools import deps_viewer
from simpler_setup.tools.deps_viewer import _merge_task_meta_with_kernel_ids, emit_text


def test_autoload_name_map_accepts_exact_name(tmp_path):
    deps_path = tmp_path / "deps.json"
    deps_path.write_text("{}")
    (tmp_path / "name_map.json").write_text(json.dumps({"callable_id_to_name": {"7": "kernel_add"}}))

    assert deps_viewer._autoload_name_map(deps_path) == {"7": "kernel_add"}


def test_emit_text_marks_alloc_without_task_entry():
    text = emit_text(
        edges=[],
        nodes=[1],
        meta={},
        deps_path="deps.json",
        annotations={},
        tensor_table={},
        task_table={},
    )

    assert "TASK 1 kind=alloc func_id=none fanin=0 fanout=0" in text
    assert "=== TASK 1 kind=alloc func_id=none ===" in text


def test_emit_text_marks_dummy_without_kernel_slots():
    text = emit_text(
        edges=[],
        nodes=[1],
        meta={},
        deps_path="deps.json",
        annotations={},
        tensor_table={},
        task_table={1: {"task_id": 1, "kernel_ids": [-1, -1, -1]}},
    )

    assert "TASK 1 kind=dummy func_id=none fanin=0 fanout=0" in text
    assert "=== TASK 1 kind=dummy func_id=none ===" in text


def test_emit_text_marks_func_name_map_yes_only_with_named_func():
    text_no_names = emit_text(
        edges=[],
        nodes=[1],
        meta={1: {"func_id": 7}},
        deps_path="deps.json",
        annotations={},
        tensor_table={},
        task_table={1: {"task_id": 1, "kernel_ids": [7, -1, -1]}},
    )
    assert "func_name_map: no" in text_no_names

    assert "func_name_map: yes" in emit_text(
        edges=[],
        nodes=[1],
        meta={1: {"func_id": 7, "func_name": "kernel_add", "_kernel_slots": [7, -1, -1]}},
        deps_path="deps.json",
        annotations={},
        tensor_table={},
        task_table={1: {"task_id": 1, "kernel_ids": [7, -1, -1]}},
    )


def test_kernel_ids_fill_func_id_when_perf_sidecar_is_absent():
    meta = _merge_task_meta_with_kernel_ids(
        {},
        {1: {"task_id": 1, "kernel_ids": [-1, 2, -1]}},
    )

    text = emit_text(
        edges=[],
        nodes=[1],
        meta=meta,
        deps_path="deps.json",
        annotations={},
        tensor_table={},
        task_table={1: {"task_id": 1, "kernel_ids": [-1, 2, -1]}},
    )

    assert "TASK 1 kind=submit func_id=[-1,2,-1] fanin=0 fanout=0" in text
    assert "func_name_map: no" in text


def test_emit_text_marks_spmd_block_count():
    meta = _merge_task_meta_with_kernel_ids(
        {},
        {1: {"task_id": 1, "kernel_ids": [-1, 2, -1], "block_num": 4}},
    )

    text = emit_text(
        edges=[],
        nodes=[1],
        meta=meta,
        deps_path="deps.json",
        annotations={},
        tensor_table={},
        task_table={1: {"task_id": 1, "kernel_ids": [-1, 2, -1], "block_num": 4}},
    )

    assert "TASK 1 kind=submit func_id=[-1,2,-1] SPMD block num = 4 fanin=0 fanout=0" in text
    assert "=== TASK 1 kind=submit func_id=[-1,2,-1] SPMD block num = 4 ===" in text


def test_kernel_ids_render_all_active_funcs_for_mixed_task():
    meta = _merge_task_meta_with_kernel_ids(
        {},
        {
            1: {"task_id": 1, "kernel_ids": [0, 1, 2]},
            2: {"task_id": 2, "kernel_ids": [0, 1, -1]},
            3: {"task_id": 3, "kernel_ids": [-1, 3, 4]},
        },
    )

    text = emit_text(
        edges=[],
        nodes=[1, 2, 3],
        meta=meta,
        deps_path="deps.json",
        annotations={},
        tensor_table={},
        task_table={
            1: {"task_id": 1, "kernel_ids": [0, 1, 2]},
            2: {"task_id": 2, "kernel_ids": [0, 1, -1]},
            3: {"task_id": 3, "kernel_ids": [-1, 3, 4]},
        },
    )

    assert "TASK 1 kind=submit func_id=[0,1,2] fanin=0 fanout=0" in text
    assert "TASK 2 kind=submit func_id=[0,1,-1] fanin=0 fanout=0" in text
    assert "TASK 3 kind=submit func_id=[-1,3,4] fanin=0 fanout=0" in text
    assert "func_name_map: no" in text


def test_kernel_ids_infer_core_type_when_perf_sidecar_is_absent():
    meta = _merge_task_meta_with_kernel_ids(
        {},
        {
            1: {"task_id": 1, "kernel_ids": [0, -1, -1]},
            2: {"task_id": 2, "kernel_ids": [-1, 1, -1]},
            3: {"task_id": 3, "kernel_ids": [0, 1, 2]},
        },
    )

    assert meta[1]["core_type"] == "aic"
    assert meta[2]["core_type"] == "aiv"
    assert meta[3]["core_type"] == "mix"


def test_kernel_ids_use_func_name_map_when_available():
    meta = _merge_task_meta_with_kernel_ids(
        {},
        {1: {"task_id": 1, "kernel_ids": [-1, 2, -1]}},
        func_names={"2": "kernel_mul"},
    )

    text = emit_text(
        edges=[],
        nodes=[1],
        meta=meta,
        deps_path="deps.json",
        annotations={},
        tensor_table={},
        task_table={1: {"task_id": 1, "kernel_ids": [-1, 2, -1]}},
    )

    assert "TASK 1 kind=submit func_id=[-1,2,-1] fanin=0 fanout=0" in text
    assert "func_name_map: yes" in text


def test_kernel_ids_use_all_named_funcs_when_available():
    meta = _merge_task_meta_with_kernel_ids(
        {},
        {1: {"task_id": 1, "kernel_ids": [0, 1, 2]}},
        func_names={"0": "MATMUL", "1": "ADD", "2": "MUL"},
    )

    text = emit_text(
        edges=[],
        nodes=[1],
        meta=meta,
        deps_path="deps.json",
        annotations={},
        tensor_table={},
        task_table={1: {"task_id": 1, "kernel_ids": [0, 1, 2]}},
    )

    assert "TASK 1 kind=submit func_id=[0,1,2] fanin=0 fanout=0" in text
    assert "func_name_map: yes" in text


def test_emit_dot_marks_spmd_nodes_without_expanding_labels():
    task_table = {1: {"task_id": 1, "kernel_ids": [-1, 2, -1], "block_num": 4, "args": []}}
    meta = _merge_task_meta_with_kernel_ids({}, task_table)

    plain = deps_viewer.emit_dot(
        edges=[],
        nodes=[1],
        meta=meta,
        task_table=task_table,
        show_tensor_info=False,
    )

    assert 'label="1 ↓0 ↑0"' in plain
    assert 'color="#C62828"' in plain
    assert "penwidth=1.5" in plain
    assert 'style="filled"' in plain
    assert "SPMD" not in plain
    assert "4 blocks" not in plain

    rich = deps_viewer.emit_dot(
        edges=[],
        nodes=[1],
        meta=meta,
        task_table=task_table,
        tensor_table={},
    )

    assert '<TABLE BORDER="1" COLOR="#C62828"' in rich
    assert "↓0 ↑0" in rich
    assert "SPMD" not in rich
    assert "4 blocks" not in rich


def test_spmd_badges_json_includes_only_multiblock_tasks():
    task_table = {
        1: {"task_id": 1, "block_num": 1},
        2: {"task_id": 2, "block_num": 8},
    }

    badges = deps_viewer._spmd_badges_json([1, 2], task_table)

    assert badges == '{"T0_2":8}'


def test_dep_stats_json_groups_peers_by_name_hint():
    task_table = {
        2: {"task_id": 2, "kernel_ids": [-1, 7, -1]},
        3: {"task_id": 3, "kernel_ids": [-1, 8, -1]},
        4: {"task_id": 4, "kernel_ids": [-1, 7, -1]},
        5: {"task_id": 5, "kernel_ids": [-1, 9, -1]},
    }
    meta = _merge_task_meta_with_kernel_ids(
        {
            2: {"func_name": "q_proj", "func_id": 7},
            3: {"func_name": "k_proj", "func_id": 8},
            4: {"func_name": "q_proj", "func_id": 7},
        },
        task_table,
        func_names={9: "v_proj"},
    )
    stats = json.loads(
        deps_viewer._dep_stats_json(
            nodes=[1, 2, 3, 4, 5],
            edges=[(1, 5), (2, 5), (3, 5), (4, 5), (5, 1)],
            meta=meta,
            task_table=task_table,
        )
    )

    assert stats["T0_5"]["fanin"] == 4
    assert stats["T0_5"]["fanout"] == 1
    assert stats["T0_5"]["pred_by_hint"] == [["q_proj", 2], ["alloc", 1], ["k_proj", 1]]
    assert stats["T0_5"]["succ_by_hint"] == [["alloc", 1]]
    assert stats["T0_5"]["name_hint"] == "v_proj"


def test_emit_dot_handles_missing_task_table():
    dot = deps_viewer.emit_dot(edges=[], nodes=[1], meta={}, task_table=None)

    assert 'label="🔥 1 · alloc ↓0 ↑0"' in dot


def test_emit_dot_does_not_mark_alloc_only_successor_with_star():
    dot = deps_viewer.emit_dot(
        edges=[(1, 3), (2, 3)],
        nodes=[1, 2, 3],
        meta={},
        task_table={3: {"task_id": 3, "kernel_ids": [-1, 7, -1]}},
        show_tensor_info=False,
    )

    assert 'label="🔥 1 · alloc ↓0 ↑1"' in dot
    assert 'label="🔥 2 · alloc ↓0 ↑1"' in dot
    assert 'label="3 ↓2 ↑0"' in dot


def test_emit_dot_marks_star_with_alloc_and_early_dispatch_predecessors():
    dot = deps_viewer.emit_dot(
        edges=[(1, 3), (2, 3)],
        nodes=[1, 2, 3],
        meta={},
        task_table={
            2: {"task_id": 2, "kernel_ids": [-1, 6, -1], "early_dispatch": True},
            3: {"task_id": 3, "kernel_ids": [-1, 7, -1]},
        },
        show_tensor_info=False,
    )

    assert 'label="🔥 1 · alloc ↓0 ↑1"' in dot
    assert 'label="🔥 2 ↓0 ↑1"' in dot
    assert 'label="⭐ 3 ↓2 ↑0"' in dot


def test_emit_dot_does_not_mark_star_when_any_predecessor_lacks_fire():
    dot = deps_viewer.emit_dot(
        edges=[(1, 4), (2, 4), (3, 4)],
        nodes=[1, 2, 3, 4],
        meta={},
        task_table={
            2: {"task_id": 2, "kernel_ids": [-1, 6, -1], "early_dispatch": True},
            3: {"task_id": 3, "kernel_ids": [-1, 7, -1]},
            4: {"task_id": 4, "kernel_ids": [-1, 8, -1]},
        },
        show_tensor_info=False,
    )

    assert 'label="4 ↓3 ↑0"' in dot


def test_emit_html_includes_dep_stats_and_detail_panel(monkeypatch):
    monkeypatch.setattr(
        deps_viewer,
        "emit_dot",
        lambda *args, **kwargs: "digraph deps { T0_1 [label=<1>]; }",
    )
    monkeypatch.setattr(
        deps_viewer,
        "render_svg",
        lambda dot, engine="dot": b"<svg><g class='node'><title>T0_1</title></g></svg>",
    )

    html = deps_viewer.emit_html(
        edges=[(1, 2)],
        nodes=[1, 2],
        meta={2: {"func_name": "q_proj", "func_id": 7}},
        task_table={
            2: {"task_id": 2, "kernel_ids": [-1, 7, -1], "args": []},
        },
    )

    assert 'id="detail"' in html
    assert "const depStats =" in html
    assert '"T0_2"' in html
    assert '"pred_by_hint"' in html
    assert "openDetail" in html
    assert "↓N ↑M = pred / succ" in html


def test_emit_dot_hides_selected_edges_with_background_color():
    dot = deps_viewer.emit_dot(
        edges=[(1, 2), (2, 3)],
        nodes=[1, 2, 3],
        meta={},
        task_table=None,
        hidden_edges={(1, 2)},
    )

    assert 'T0_1 -> T0_2 [class="hidden-edge", color="#eef2f7", fontcolor="#eef2f7"];' in dot
    assert "T0_2 -> T0_3;" in dot


def test_emit_html_default_preserves_task_table_rendering(monkeypatch):
    captured = {}

    def fake_emit_dot(*args, **kwargs):
        captured["show_tensor_info"] = kwargs["show_tensor_info"]
        return "digraph deps { T0_1 [label=<1>]; }"

    monkeypatch.setattr(deps_viewer, "emit_dot", fake_emit_dot)
    monkeypatch.setattr(deps_viewer, "render_svg", lambda dot, engine="dot": b"<svg></svg>")

    deps_viewer.emit_html(edges=[], nodes=[1], meta={}, task_table={1: {"task_id": 1, "args": []}})

    assert captured["show_tensor_info"] is None


def test_validate_args_rejects_show_tensor_info_in_text_mode(capsys):
    rc = deps_viewer.main(["deps.json", "--show-tensor-info"])

    assert rc == 2
    assert "--show-tensor-info is only valid with --format html" in capsys.readouterr().err


def test_main_passes_task_table_when_show_tensor_info_enabled(tmp_path, monkeypatch):
    deps_json = tmp_path / "deps.json"
    deps_json.write_text("{}")

    monkeypatch.setattr(
        deps_viewer,
        "_load_deps_edges",
        lambda path: (
            [(1, 2)],
            [1, 2],
            {(1, 2): [{"arg": 0, "tensor_id": 5}]},
            {5: {"name": "T0"}},
            {1: {"task_id": 1, "args": []}},
        ),
    )
    monkeypatch.setattr(deps_viewer, "_load_task_meta", lambda path, func_names=None: {})
    monkeypatch.setattr(deps_viewer, "_autoload_name_map", lambda path: {})

    captured = {}

    def fake_emit_html(*args, **kwargs):
        task_table = kwargs["task_table"]
        show_tensor_info = kwargs["show_tensor_info"]
        captured["task_table"] = task_table
        captured["show_tensor_info"] = show_tensor_info
        return "<html></html>"

    monkeypatch.setattr(deps_viewer, "emit_html", fake_emit_html)

    rc = deps_viewer.main([str(deps_json), "--format", "html", "--show-tensor-info"])

    assert rc == 0
    assert captured["task_table"] == {1: {"task_id": 1, "args": []}}
    assert captured["show_tensor_info"] is True


def test_main_keeps_task_metadata_when_show_tensor_info_disabled(tmp_path, monkeypatch):
    deps_json = tmp_path / "deps.json"
    deps_json.write_text("{}")

    monkeypatch.setattr(
        deps_viewer,
        "_load_deps_edges",
        lambda path: (
            [(1, 2)],
            [1, 2],
            {(1, 2): [{"arg": 0, "tensor_id": 5}]},
            {5: {"name": "T0"}},
            {1: {"task_id": 1, "args": []}},
        ),
    )
    monkeypatch.setattr(deps_viewer, "_load_task_meta", lambda path, func_names=None: {})
    monkeypatch.setattr(deps_viewer, "_autoload_name_map", lambda path: {})

    captured = {}

    def fake_emit_html(*args, **kwargs):
        task_table = kwargs["task_table"]
        show_tensor_info = kwargs["show_tensor_info"]
        captured["task_table"] = task_table
        captured["show_tensor_info"] = show_tensor_info
        return "<html></html>"

    monkeypatch.setattr(deps_viewer, "emit_html", fake_emit_html)

    rc = deps_viewer.main([str(deps_json), "--format", "html"])

    assert rc == 0
    assert captured["task_table"] == {1: {"task_id": 1, "args": []}}
    assert captured["show_tensor_info"] is False


def test_transitive_reduction_keeps_non_redundant_diamond():
    # Diamond A->B, A->C, B->D, C->D — no edge is implied by another path.
    edges = [(1, 2), (1, 3), (2, 4), (3, 4)]
    kept, removed, is_dag = deps_viewer._transitive_reduction(edges, [1, 2, 3, 4])
    assert is_dag
    assert removed == []
    assert kept == sorted(edges)


def test_transitive_reduction_drops_multi_hop_shortcut():
    # Chain A->B->C->D with a long shortcut A->D implied by three hops.
    kept, removed, is_dag = deps_viewer._transitive_reduction([(1, 2), (2, 3), (3, 4), (1, 4)], [1, 2, 3, 4])
    assert is_dag
    assert removed == [(1, 4)]
    assert kept == [(1, 2), (2, 3), (3, 4)]


def test_transitive_reduction_can_drop_tensormap_shortcut():
    edges = [(1, 2), (2, 3), (1, 3)]
    annotations = {(1, 3): [{"source": "tensormap", "tensor_id": 7}]}

    kept, removed, is_dag = deps_viewer._transitive_reduction(edges, [1, 2, 3], annotations)

    assert is_dag
    assert removed == [(1, 3)]
    assert kept == [(1, 2), (2, 3)]


def test_transitive_reduction_detects_cycle_and_falls_back():
    edges = [(1, 2), (2, 1)]
    kept, removed, is_dag = deps_viewer._transitive_reduction(edges, [1, 2])
    assert not is_dag
    assert kept == edges
    assert removed == []


def _output_lifetime_graph(regions):
    """alloc->A->B->C plus direct alloc edges to B/C.

    ``regions`` maps A/B/C task ids to ``(start_offset, shape, strides)`` for
    one OUTPUT_EXISTING followed by two INOUT views of alloc-owned storage.
    """
    alloc, a, b, c = 1, 2, 3, 4
    edges = [(alloc, a), (alloc, b), (alloc, c), (a, b), (b, c)]
    annotations = {}
    task_table = {}
    for task_id in (a, b, c):
        start_offset, shape, strides = regions[task_id]
        task_table[task_id] = {
            "task_id": task_id,
            "args": [
                {
                    "idx": 0,
                    "type": "OUTPUT_EXISTING" if task_id == a else "INOUT",
                    "tensor_id": 7,
                    "dtype": "FLOAT32",
                    "shape": shape,
                    "start_offset": str(start_offset),
                    "strides": strides,
                }
            ],
        }
        annotations[(alloc, task_id)] = [
            {
                "pred": alloc,
                "succ": task_id,
                "arg": 0,
                "source": "creator",
                "tensor_id": 7,
                "consumer_dtype": "FLOAT32",
                "consumer_shape": shape,
                "consumer_start_offset": str(start_offset),
                "consumer_strides": strides,
            }
        ]
    for pred, succ in ((a, b), (b, c)):
        pred_offset, pred_shape, pred_strides = regions[pred]
        succ_offset, succ_shape, succ_strides = regions[succ]
        annotations[(pred, succ)] = [
            {
                "pred": pred,
                "succ": succ,
                "arg": 0,
                "source": "tensormap",
                "overlap": "covered",
                "tensor_id": 7,
                "consumer_dtype": "FLOAT32",
                "consumer_shape": succ_shape,
                "consumer_start_offset": str(succ_offset),
                "consumer_strides": succ_strides,
                "producer_shape": pred_shape,
                "producer_start_offset": str(pred_offset),
                "producer_strides": pred_strides,
            }
        ]
    tensor_table = {
        7: {
            "tensor_id": 7,
            "buffer_addr": "4096",
            "buffer_numel": "128",
            "dtype": "FLOAT32",
        }
    }
    return edges, [alloc, a, b, c], annotations, tensor_table, task_table


def test_transitive_reduction_v2_drops_only_full_range_middle_output():
    graph = _output_lifetime_graph(
        {
            2: (0, [2, 4], [4, 1]),
            3: (0, [8], [1]),
            4: (0, [4, 2], [2, 1]),
        }
    )

    kept, removed, is_dag = deps_viewer._transitive_reduction_v2(*graph)

    assert is_dag
    assert removed == [(1, 3)]
    assert kept == [(1, 2), (1, 4), (2, 3), (3, 4)]


def test_transitive_reduction_v2_does_not_cross_output_existing_reuse_boundary():
    alloc, a1, b1, a2, b2 = 1, 2, 3, 4, 5
    edges = [
        (alloc, a1),
        (alloc, b1),
        (alloc, a2),
        (alloc, b2),
        (a1, b1),
        (b1, a2),
        (a2, b2),
    ]
    annotations = {}
    task_table = {}
    for task_id, arg_type in (
        (a1, "OUTPUT_EXISTING"),
        (b1, "INOUT"),
        (a2, "OUTPUT_EXISTING"),
        (b2, "INOUT"),
    ):
        task_table[task_id] = {
            "task_id": task_id,
            "args": [
                {
                    "idx": 0,
                    "type": arg_type,
                    "tensor_id": 7,
                    "dtype": "FLOAT32",
                    "shape": [8],
                    "start_offset": "0",
                    "strides": [1],
                }
            ],
        }
        annotations[(alloc, task_id)] = [
            {
                "pred": alloc,
                "succ": task_id,
                "arg": 0,
                "source": "creator",
                "tensor_id": 7,
                "consumer_dtype": "FLOAT32",
                "consumer_shape": [8],
                "consumer_start_offset": "0",
                "consumer_strides": [1],
            }
        ]
    for pred, succ in ((a1, b1), (a2, b2)):
        annotations[(pred, succ)] = [
            {
                "pred": pred,
                "succ": succ,
                "arg": 0,
                "source": "tensormap",
                "overlap": "covered",
                "tensor_id": 7,
                "consumer_dtype": "FLOAT32",
                "consumer_shape": [8],
                "consumer_start_offset": "0",
                "consumer_strides": [1],
                "producer_shape": [8],
                "producer_start_offset": "0",
                "producer_strides": [1],
            }
        ]
    annotations[(b1, a2)] = [{"pred": b1, "succ": a2, "arg": -1, "source": "explicit"}]
    tensor_table = {
        7: {
            "tensor_id": 7,
            "buffer_addr": "4096",
            "buffer_numel": "128",
            "dtype": "FLOAT32",
        }
    }

    kept, removed, is_dag = deps_viewer._transitive_reduction_v2(
        edges, [alloc, a1, b1, a2, b2], annotations, tensor_table, task_table
    )

    assert is_dag
    assert removed == []
    assert kept == edges


def _write_output_lifetime_deps(tmp_path):
    edges, nodes, annotations, tensor_table, task_table = _output_lifetime_graph(
        {
            2: (0, [8], [1]),
            3: (0, [8], [1]),
            4: (0, [8], [1]),
        }
    )
    annotated_edges = []
    for pred, succ in edges:
        rows = annotations.get((pred, succ))
        if rows:
            annotated_edges.extend(rows)
        else:
            annotated_edges.append({"pred": str(pred), "succ": str(succ), "arg": -1, "source": "explicit"})
    deps = {
        "tasks": list(task_table.values()),
        "tensors": list(tensor_table.values()),
        "edges": annotated_edges,
    }
    path = tmp_path / "deps.json"
    path.write_text(json.dumps(deps))
    return path, nodes


def test_main_v2_edge_modes_keep_lifetime_boundaries_and_omit_middle(tmp_path):
    deps_path, _nodes = _write_output_lifetime_deps(tmp_path)
    reduced_out = tmp_path / "reduced_v2.txt"
    omitted_out = tmp_path / "omitted_v2.txt"

    reduced_rc = deps_viewer.main([str(deps_path), "--edge-mode", "reduced_v2", "-o", str(reduced_out)])
    omitted_rc = deps_viewer.main([str(deps_path), "--edge-mode", "omitted_v2", "-o", str(omitted_out)])

    assert reduced_rc == 0
    assert omitted_rc == 0
    assert "unique_task_edges: 4" in reduced_out.read_text()
    assert "unique_task_edges: 1" in omitted_out.read_text()
    assert "=== TASK 1" in omitted_out.read_text()
    assert "  -> 3" in omitted_out.read_text()


def _write_deps_edges(tmp_path, edges, sources=None):
    sources = sources or {}
    annotated_edges = []
    for pred, succ in edges:
        source = sources.get((pred, succ), "explicit")
        edge = {"pred": str(pred), "succ": str(succ), "arg": -1 if source == "explicit" else 0, "source": source}
        if source != "explicit":
            edge["tensor_id"] = "7"
        annotated_edges.append(edge)
    deps = {
        "tasks": [],
        "tensors": [],
        "edges": annotated_edges,
    }
    path = tmp_path / "deps.json"
    path.write_text(json.dumps(deps))
    return path


def test_main_reduced_mode_drops_edge_and_prints_removed(tmp_path, capsys):
    # A(r1t1) -> B(r1t2) -> C(r1t3) plus the implied A->C shortcut.
    ring = 1 << 32
    a, b, c = ring + 1, ring + 2, ring + 3
    deps_path = _write_deps_edges(tmp_path, [(a, b), (b, c), (a, c)])
    out = tmp_path / "graph.txt"

    rc = deps_viewer.main([str(deps_path), "--edge-mode", "reduced", "-o", str(out)])
    assert rc == 0

    text = out.read_text()
    assert "unique_task_edges: 2" in text
    # Annotations of the pruned edge are dropped too, so the count matches the
    # reduced edge set instead of over-counting (was 3 before reduction).
    assert "annotated_edges: 2" in text
    stdout = capsys.readouterr().out
    assert "removed 1 redundant edge(s)" in stdout
    # ring>=1 nodes render as the explicit (ring, local) tuple.
    assert "(1, 1) -> (1, 3)" in stdout


def test_main_reduced_mode_keeps_transitive_creator_reference(tmp_path, capsys):
    # Mirrors vector_example's t0 -> t1 -> t3 -> t4 path plus t0 -> t4:
    # t4 directly reads tensor c created by t0, so that creator-lifetime edge
    # must survive even though the task ordering is transitively covered.
    t0, t1, t3, t4 = 1, 2, 3, 4
    edges = [(t0, t1), (t1, t3), (t3, t4), (t0, t4)]
    deps_path = _write_deps_edges(tmp_path, edges, sources={(t0, t4): "creator"})
    out = tmp_path / "graph.txt"

    rc = deps_viewer.main([str(deps_path), "--edge-mode", "reduced", "-o", str(out)])

    assert rc == 0
    assert "unique_task_edges: 4" in out.read_text()
    assert "removed 0 redundant edge(s)" in capsys.readouterr().out


def test_main_full_mode_keeps_all_edges(tmp_path, capsys):
    ring = 1 << 32
    a, b, c = ring + 1, ring + 2, ring + 3
    deps_path = _write_deps_edges(tmp_path, [(a, b), (b, c), (a, c)])
    out = tmp_path / "graph.txt"

    rc = deps_viewer.main([str(deps_path), "-o", str(out)])
    assert rc == 0
    assert "unique_task_edges: 3" in out.read_text()
    assert "Transitive reduction" not in capsys.readouterr().out


def test_main_html_edge_modes_keep_full_layout_and_hide_unselected_edges(tmp_path, monkeypatch):
    ring = 1 << 32
    a, b, c = ring + 1, ring + 2, ring + 3
    all_edges = [(a, b), (b, c), (a, c)]
    deps_path = _write_deps_edges(tmp_path, all_edges)
    captures = []

    def fake_emit_html(edges, _nodes, _meta, **kwargs):
        html_edge_style = kwargs["html_edge_style"]
        captures.append(
            {
                "edges": edges,
                "annotations": kwargs["annotations"],
                "hidden_edges": html_edge_style["hidden_edges"],
                "visible_edge_count": html_edge_style["visible_edge_count"],
            }
        )
        return "<html></html>"

    monkeypatch.setattr(deps_viewer, "emit_html", fake_emit_html)

    cases = [
        ("reduced", {(a, c)}, 2),
        ("omitted", {(a, b), (b, c)}, 1),
        ("reduced_v2", {(a, c)}, 2),
        ("omitted_v2", {(a, b), (b, c)}, 1),
    ]
    for mode, expected_hidden, expected_visible_count in cases:
        rc = deps_viewer.main(
            [
                str(deps_path),
                "--format",
                "html",
                "--edge-mode",
                mode,
                "-o",
                str(tmp_path / f"{mode}.html"),
            ]
        )
        assert rc == 0

    assert len(captures) == len(cases)
    for captured, (_mode, expected_hidden, expected_visible_count) in zip(captures, cases):
        assert captured["edges"] == sorted(all_edges)
        assert set(captured["annotations"]) == set(all_edges)
        assert captured["hidden_edges"] == expected_hidden
        assert captured["visible_edge_count"] == expected_visible_count


def test_main_omitted_mode_draws_only_redundant_edges(tmp_path, capsys):
    # A(r1t1) -> B(r1t2) -> C(r1t3) plus the implied A->C shortcut. omitted
    # keeps ONLY the redundant A->C edge (complement of reduced).
    ring = 1 << 32
    a, b, c = ring + 1, ring + 2, ring + 3
    deps_path = _write_deps_edges(tmp_path, [(a, b), (b, c), (a, c)])
    out = tmp_path / "graph.txt"

    rc = deps_viewer.main([str(deps_path), "--edge-mode", "omitted", "-o", str(out)])
    assert rc == 0

    text = out.read_text()
    # Only the single redundant edge is drawn; annotations match it.
    assert "unique_task_edges: 1" in text
    assert "annotated_edges: 1" in text
    # The drawn edge is the redundant A->C, not the chain edges.
    assert "=== TASK (1, 1)" in text
    assert "  -> (1, 3)" in text
    assert "  -> (1, 2)" not in text
    stdout = capsys.readouterr().out
    assert "showing 1 redundant edge(s)" in stdout
    assert "(1, 1) -> (1, 3)" in stdout


def test_main_omitted_default_output_stem(tmp_path):
    ring = 1 << 32
    a, b, c = ring + 1, ring + 2, ring + 3
    deps_path = _write_deps_edges(tmp_path, [(a, b), (b, c), (a, c)])

    rc = deps_viewer.main([str(deps_path), "--edge-mode", "omitted"])
    assert rc == 0
    assert (tmp_path / "deps_viewer_omitted.txt").exists()
    assert not (tmp_path / "deps_viewer.txt").exists()
