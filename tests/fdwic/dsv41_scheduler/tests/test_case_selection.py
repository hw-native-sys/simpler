# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Keep the approved scheduler coverage matrix through export and reporting."""

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.manual
SUITE = Path(__file__).resolve().parents[1]
SELECTED = {
    "pypto_lib": {"c2a_full", "moe", "layers_19_21"},
    "pypto_lib_dsl": {"c2a_full", "moe", "layers_19_21"},
    "pypto_lib_latest": {"swa", "c2a_full", "c1a_reindex", "c1a_reuse", "moe", "engram", "layers_19_21"},
}
SPEC = importlib.util.spec_from_file_location("selected_workloads", SUITE / "export_workloads.py")
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_manifests_and_tables_contain_only_selected_cases():
    total = 0
    for flavor, selected in SELECTED.items():
        folder = SUITE / "workloads" / flavor
        cases = json.loads((folder / "source.json").read_text())["cases"]
        assert set(cases) == selected
        replay = {name for name, case in cases.items() if case.get("execution") != "native_moe_host_control"}
        assert {p.parent.name for p in folder.glob("*/tasks.tsv")} == replay
        total += len(cases) * (2 if flavor == "pypto_lib_dsl" else 1)
    assert total == 16


def test_performance_table_matches_selected_variants():
    actual = []
    for line in (SUITE / "RESULTS.md").read_text().splitlines():
        cells = [cell.strip() for cell in line.split("|")[1:-1]]
        if len(cells) == 6 and re.fullmatch(r"[0-9]+\.[0-9]{6}", cells[2]):
            actual.append((cells[0], cells[1]))
    labels = [label for case, label in actual if case == "c2a_full"]
    assert len(set(labels)) == 4
    sources = ("pypto_lib", "pypto_lib_dsl", "pypto_lib_dsl", "pypto_lib_latest")
    expected = {(case, label) for label, flavor in zip(labels, sources, strict=True) for case in SELECTED[flavor]}
    assert set(actual) == expected
    assert len(actual) == 16


@pytest.mark.parametrize("flavor", SELECTED)
def test_topology_refresh_cannot_restore_removed_cases(tmp_path, flavor):
    if flavor == "pypto_lib_latest":
        pattern = ["_hc_collapse", "_hc_post", "_hc_collapse", "_hc_post"]
    elif flavor == "pypto_lib":
        pattern = ["hc_collapse_norm", "hc_post", "moe", "hc_post_2"]
    else:
        pattern = ["collapse_norm_grid", "expand_grid__2", "moe", "expand_grid"]
    names = list(dict.fromkeys([*pattern, "engram_quant"]))
    sequence = []
    for layer in range(40):
        sequence.extend(pattern)
        if layer == 0:
            sequence.append("engram_quant")
    if flavor == "pypto_lib_latest":
        sequence.append("_hc_collapse")
    tasks = [
        {"task_id": str(i), "kernel_ids": [-1, names.index(name), -1], "block_num": 1}
        for i, name in enumerate(sequence)
    ]
    capture = tmp_path / "capture"
    capture.mkdir()
    (capture / "deps.json").write_text(
        json.dumps(
            {
                "runtime": "host_build_graph",
                "tasks": tasks,
                "edges": [{"pred": str(i), "succ": str(i + 1)} for i in range(len(tasks) - 1)],
            }
        )
    )
    (capture / "name_map.json").write_text(json.dumps({"callable_id_to_name": dict(enumerate(names))}))
    output = tmp_path / "exported"
    MODULE.export(capture, flavor, output)
    produced = json.loads((output / flavor / "source.json").read_text())["cases"]
    assert set(produced) == SELECTED[flavor]
    replay = {name for name, case in produced.items() if case.get("execution") != "native_moe_host_control"}
    assert {p.name for p in (output / flavor).iterdir() if p.is_dir()} == replay


def test_run_list_excludes_removed_cached_bundles():
    listed = subprocess.check_output([sys.executable, str(SUITE / "production/list_cases.py")], text=True).splitlines()
    expected = {f"{flavor}/{case}" for flavor, cases in SELECTED.items() for case in cases}
    expected |= {f"pypto_lib_dsl_all_sync64/{case}" for case in SELECTED["pypto_lib_dsl"]}
    assert set(listed) == expected
    assert len(listed) == 16
