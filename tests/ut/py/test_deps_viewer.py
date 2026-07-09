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

    assert 'label="1"' in plain
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
    assert "SPMD" not in rich
    assert "4 blocks" not in rich


def test_spmd_badges_json_includes_only_multiblock_tasks():
    task_table = {
        1: {"task_id": 1, "block_num": 1},
        2: {"task_id": 2, "block_num": 8},
    }

    badges = deps_viewer._spmd_badges_json([1, 2], task_table)

    assert badges == '{"T0_2":8}'


def test_emit_dot_handles_missing_task_table():
    dot = deps_viewer.emit_dot(edges=[], nodes=[1], meta={}, task_table=None)

    assert 'label="1 · alloc"' in dot


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

    def fake_emit_html(
        edges,
        nodes,
        meta,
        direction="LR",
        engine="dot",
        annotations=None,
        tensor_table=None,
        task_table=None,
        show_tensor_info=False,
    ):
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

    def fake_emit_html(
        edges,
        nodes,
        meta,
        direction="LR",
        engine="dot",
        annotations=None,
        tensor_table=None,
        task_table=None,
        show_tensor_info=False,
    ):
        captured["task_table"] = task_table
        captured["show_tensor_info"] = show_tensor_info
        return "<html></html>"

    monkeypatch.setattr(deps_viewer, "emit_html", fake_emit_html)

    rc = deps_viewer.main([str(deps_json), "--format", "html"])

    assert rc == 0
    assert captured["task_table"] == {1: {"task_id": 1, "args": []}}
    assert captured["show_tensor_info"] is False


def test_transitive_reduction_removes_implied_edge():
    # A->B->C plus the implied shortcut A->C.
    kept, removed, is_dag = deps_viewer._transitive_reduction([(1, 2), (1, 3), (2, 3)], [1, 2, 3])
    assert is_dag
    assert removed == [(1, 3)]
    assert kept == [(1, 2), (2, 3)]


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


def test_transitive_reduction_detects_cycle_and_falls_back():
    edges = [(1, 2), (2, 1)]
    kept, removed, is_dag = deps_viewer._transitive_reduction(edges, [1, 2])
    assert not is_dag
    assert kept == edges
    assert removed == []


def _write_deps_edges(tmp_path, edges):
    deps = {
        "tasks": [],
        "tensors": [],
        "edges": [{"pred": str(p), "succ": str(s), "arg": 0, "source": "creator"} for p, s in edges],
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


def test_main_full_mode_keeps_all_edges(tmp_path, capsys):
    ring = 1 << 32
    a, b, c = ring + 1, ring + 2, ring + 3
    deps_path = _write_deps_edges(tmp_path, [(a, b), (b, c), (a, c)])
    out = tmp_path / "graph.txt"

    rc = deps_viewer.main([str(deps_path), "-o", str(out)])
    assert rc == 0
    assert "unique_task_edges: 3" in out.read_text()
    assert "Transitive reduction" not in capsys.readouterr().out


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
