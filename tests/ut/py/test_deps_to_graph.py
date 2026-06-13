# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Contract tests for simpler_setup.tools.deps_to_graph text output."""

from simpler_setup.tools.deps_to_graph import _merge_task_meta_with_kernel_ids, emit_text


def test_emit_text_uses_unknown_core_without_explicit_meta_entry():
    text = emit_text(
        edges=[],
        nodes=[1],
        meta={2: {"core_type": "aiv", "func_name": "other"}},
        deps_path="deps.json",
        annotations={},
        tensor_table={},
        task_table={1: {"task_id": 1, "scope": "auto", "args": []}},
    )

    assert "TASK 1 func=unknown fanin=0 fanout=0" in text
    assert "=== TASK 1 func=unknown ===" in text


def test_emit_text_marks_func_name_map_yes_only_with_named_func():
    text_no_names = emit_text(
        edges=[],
        nodes=[1],
        meta={1: {"func_id": 7, "core_type": "aiv"}},
        deps_path="deps.json",
        annotations={},
        tensor_table={},
        task_table={1: {"task_id": 1, "scope": "auto", "args": []}},
    )
    assert "func_name_map: no" in text_no_names

    assert "func_name_map: yes" in emit_text(
        edges=[],
        nodes=[1],
        meta={1: {"func_id": 7, "func_name": "kernel_add", "core_type": "aiv"}},
        deps_path="deps.json",
        annotations={},
        tensor_table={},
        task_table={1: {"task_id": 1, "scope": "auto", "args": []}},
    )


def test_kernel_ids_fill_func_id_when_perf_sidecar_is_absent():
    meta = _merge_task_meta_with_kernel_ids(
        {},
        {1: {"task_id": 1, "scope": "auto", "kernel_ids": [-1, 2, -1], "args": []}},
    )

    text = emit_text(
        edges=[],
        nodes=[1],
        meta=meta,
        deps_path="deps.json",
        annotations={},
        tensor_table={},
        task_table={1: {"task_id": 1, "scope": "auto", "kernel_ids": [-1, 2, -1], "args": []}},
    )

    assert "TASK 1 func=f2 fanin=0 fanout=0" in text
    assert "func_name_map: no" in text


def test_kernel_ids_use_func_name_map_when_available():
    meta = _merge_task_meta_with_kernel_ids(
        {},
        {1: {"task_id": 1, "scope": "auto", "kernel_ids": [-1, 2, -1], "args": []}},
        func_names={"2": "kernel_mul"},
    )

    text = emit_text(
        edges=[],
        nodes=[1],
        meta=meta,
        deps_path="deps.json",
        annotations={},
        tensor_table={},
        task_table={1: {"task_id": 1, "scope": "auto", "kernel_ids": [-1, 2, -1], "args": []}},
    )

    assert "TASK 1 func=kernel_mul fanin=0 fanout=0" in text
    assert "func_name_map: yes" in text
