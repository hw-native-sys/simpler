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
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.manual
SUITE = Path(__file__).resolve().parents[1]
FLAVOR = "pypto_lib_latest"
CASES = {"swa", "c2a_full", "c1a_reindex", "c1a_reuse", "moe", "engram", "layers_19_21"}
SPEC = importlib.util.spec_from_file_location("selected_workloads", SUITE / "export_workloads.py")
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_manifests_and_tables_contain_only_selected_cases():
    workloads = SUITE / "workloads"
    assert {p.parent.name for p in workloads.glob("*/source.json")} == {FLAVOR, "native_moe"}
    assert {p.parent.name for p in (SUITE / "kernels").glob("*/manifest.json")} == {FLAVOR, "native_moe"}
    folder = workloads / FLAVOR
    cases = json.loads((folder / "source.json").read_text())["cases"]
    assert set(cases) == CASES
    assert cases["moe"]["execution"] == "native_moe_host_control"
    assert {p.parent.name for p in folder.glob("*/tasks.tsv")} == CASES - {"moe"}


def test_performance_table_matches_default_list():
    table = [
        [cell.strip() for cell in line.split("|")[1:-1]]
        for line in (SUITE / "RESULTS.md").read_text().splitlines()
        if line.startswith("|")
    ]
    rows = table[2:]
    assert [row[0] for row in rows] == [
        "swa",
        "c2a_full",
        "c1a_reindex",
        "c1a_reuse",
        "engram",
        "full_wave_mix_chain",
        "mpmd_diamond",
    ]
    for _, pgo, bound, scheduled, gap in rows:
        assert 0 < float(bound) <= float(scheduled) and gap.endswith("%")
        # Displayed times are independently rounded to six decimal places.
        assert float(gap[:-1]) == pytest.approx(100 * (float(pgo) / float(scheduled) - 1), abs=0.035)


def test_topology_refresh_cannot_restore_removed_cases(tmp_path):
    flavor = FLAVOR
    pattern = ["hc_collapse_norm", "_o_proj_b_hc_post_aic", "hc_collapse_norm_2", "_hc_post"]
    names = list(dict.fromkeys([*pattern, "engram_quant"]))
    sequence = []
    for layer in range(40):
        sequence.extend(pattern)
        if layer == 0:
            sequence.append("engram_quant")
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
    assert set(produced) == CASES
    replay = {name for name, case in produced.items() if case.get("execution") != "native_moe_host_control"}
    assert {p.name for p in (output / flavor).iterdir() if p.is_dir()} == replay


@pytest.mark.parametrize("args", [[], ["--runtime=host_build_graph"], ["--runtime=percounter"]])
def test_run_list_excludes_removed_cached_bundles(args):
    listed = subprocess.check_output(
        [sys.executable, str(SUITE / "production/list_cases.py"), *args], text=True
    ).splitlines()
    model = ["swa", "c2a_full", "c1a_reindex", "c1a_reuse", "engram"]
    assert listed == [
        *(f"{FLAVOR}/{case}" for case in model),
        "synthetic/full_wave_mix_chain",
        "synthetic/mpmd_diamond",
    ]
