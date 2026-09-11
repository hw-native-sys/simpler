# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Unit tests for A5 AICPU affinity preflight ranking and plan writes."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from simpler_setup.tools import rtt_die_preflight as preflight


def test_pick_orchestrator_min_avg_tiebreak():
    pool = [
        {"pool_idx": 0, "avg_handshake_ticks": 100},
        {"pool_idx": 1, "avg_handshake_ticks": 50},
        {"pool_idx": 2, "avg_handshake_ticks": 50},
        {"pool_idx": 3, "avg_handshake_ticks": 80},
    ]
    assert preflight.pick_orchestrator(pool) == 1


def test_pack_schedulers_phys_order_to_logical_die_contract():
    # Candidates after orch removal. Scores chosen so phys picks are:
    # die0 best=cpu10, die1 best=cpu20, die1 2nd=cpu21, die0 2nd=cpu11
    candidates = [
        {"cpu_id": 10, "die0_sum_ticks": 100, "die1_sum_ticks": 900},
        {"cpu_id": 11, "die0_sum_ticks": 200, "die1_sum_ticks": 800},
        {"cpu_id": 20, "die0_sum_ticks": 900, "die1_sum_ticks": 100},
        {"cpu_id": 21, "die0_sum_ticks": 800, "die1_sum_ticks": 150},
    ]
    allowed, schedulers = preflight.pack_schedulers_from_die_scores(candidates)
    assert allowed == [10, 11, 20, 21]
    assert [s["assigned_die"] for s in schedulers] == [0, 0, 1, 1]
    assert [s["logical_idx"] for s in schedulers] == [0, 1, 2, 3]


def test_build_allowed_cpus_from_probe_full():
    probe = {
        "schema_version": 3,
        "measurement_method": preflight.MEASUREMENT_METHOD,
        "soc_name": "Ascend950PR_9599",
        "device_id": 0,
        "user_pool_cpus": [3, 4, 5, 6, 7, 8],
        "orch_pool_idx": 5,
        "pool": [
            {
                "pool_idx": 0,
                "cpu_id": 3,
                "avg_handshake_ticks": 40,
                "die0_sum_ticks": 100,
                "die1_sum_ticks": 900,
                "is_orch": 0,
            },
            {
                "pool_idx": 1,
                "cpu_id": 4,
                "avg_handshake_ticks": 41,
                "die0_sum_ticks": 200,
                "die1_sum_ticks": 800,
                "is_orch": 0,
            },
            {
                "pool_idx": 2,
                "cpu_id": 5,
                "avg_handshake_ticks": 42,
                "die0_sum_ticks": 900,
                "die1_sum_ticks": 100,
                "is_orch": 0,
            },
            {
                "pool_idx": 3,
                "cpu_id": 6,
                "avg_handshake_ticks": 43,
                "die0_sum_ticks": 800,
                "die1_sum_ticks": 150,
                "is_orch": 0,
            },
            {
                "pool_idx": 4,
                "cpu_id": 7,
                "avg_handshake_ticks": 44,
                "die0_sum_ticks": 700,
                "die1_sum_ticks": 700,
                "is_orch": 0,
            },
            {
                "pool_idx": 5,
                "cpu_id": 8,
                "avg_handshake_ticks": 10,
                "die0_sum_ticks": 0,
                "die1_sum_ticks": 0,
                "is_orch": 1,
            },
        ],
    }
    node = preflight.build_allowed_cpus_from_probe(probe)
    assert node["orch_cpu"] == 8
    assert node["allowed_cpus"][-1] == 8
    assert len(node["allowed_cpus"]) == 5
    assert node["schedulers"][0]["assigned_die"] == 0
    assert node["schedulers"][1]["assigned_die"] == 0
    assert node["schedulers"][2]["assigned_die"] == 1
    assert node["schedulers"][3]["assigned_die"] == 1
    assert node["plan_source"] == preflight.MEASUREMENT_METHOD
    assert node["device_id"] == 0
    assert node["active_count"] == 5
    assert node["occupy_mask"] == "0x1f8"


def test_build_allowed_cpus_pool_too_small():
    probe = {
        "schema_version": 3,
        "measurement_method": preflight.MEASUREMENT_METHOD,
        "soc_name": "Ascend950PR_9599",
        "device_id": 0,
        "pool_too_small": True,
        "user_pool_cpus": [3, 4, 5],
    }
    with pytest.raises(ValueError, match="requires 4 scheduler"):
        preflight.build_allowed_cpus_from_probe(probe)


def test_contiguous_fallback_and_side_file(tmp_path: Path):
    out = tmp_path / "aicpu_affinity_plan.json"
    node = preflight.build_contiguous_fallback_node(
        soc_name="Ascend950PR_9599",
        device_id=1,
        occupy_cpus=[8, 3, 4, 5, 6, 7],
        plan_source=preflight.PLAN_SOURCE_PROBE_FAILED,
    )
    # Contiguous order preserves input order (not resorted).
    assert node["allowed_cpus"] == [8, 3, 4, 5, 6]
    assert node["orch_cpu"] == 6
    preflight.persist_device_plan(out, soc_name="Ascend950PR_9599", device_id=1, device_node=node)
    side = preflight.cpus_side_path(out)
    assert side.is_file()
    text = side.read_text(encoding="utf-8")
    assert "schema_version=3\n" in text
    assert "device_id=1\n" in text
    assert "soc=Ascend950PR_9599\n" in text
    assert f"source={preflight.PLAN_SOURCE_PROBE_FAILED}\n" in text
    assert "occupy_mask=0x1f8\n" in text
    assert "active_count=5\n" in text
    assert "cpus=8,3,4,5,6\n" in text
    assert preflight.side_file_looks_usable(side, 1)


def test_cli_fallback_occupy(tmp_path: Path):
    out = tmp_path / "plan.json"
    rc = preflight.main(
        [
            "--device",
            "0",
            "--soc",
            "Ascend950PR_9599",
            "--fallback-occupy",
            "3,4,5,6,7,8",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["socs"]["Ascend950PR_9599"]["devices"]["0"]["allowed_cpus"] == [3, 4, 5, 6, 7]
    assert loaded["socs"]["Ascend950PR_9599"]["devices"]["0"]["plan_source"] == (preflight.PLAN_SOURCE_PROBE_FAILED)
    side = preflight.cpus_side_path(out).read_text(encoding="utf-8")
    assert "cpus=3,4,5,6,7\n" in side
    assert f"source={preflight.PLAN_SOURCE_PROBE_FAILED}\n" in side


def test_validate_probe_rejects_old_schema():
    with pytest.raises(ValueError, match="schema_version"):
        preflight.validate_probe_result(
            {
                "schema_version": 2,
                "measurement_method": "serialized-cond-rtt-v2",
                "soc_name": "Ascend950PR_9599",
                "device_id": 0,
            },
            0,
        )


def test_per_device_plans_do_not_merge_or_overwrite(tmp_path: Path):
    out0 = tmp_path / "aicpu_affinity_plan.0.json"
    out1 = tmp_path / "aicpu_affinity_plan.1.json"
    node0 = preflight.build_device_node_from_allowed(
        soc_name="Ascend950PR_9599",
        device_id=0,
        allowed_cpus=[3, 4, 5, 6, 7],
        occupy_cpus=[3, 4, 5, 6, 7, 8],
        plan_source="manual",
    )
    preflight.persist_device_plan(out0, soc_name="Ascend950PR_9599", device_id=0, device_node=node0)

    node1 = preflight.build_device_node_from_allowed(
        soc_name="Ascend950PR_9599",
        device_id=1,
        allowed_cpus=[10, 11, 12, 13, 14],
        occupy_cpus=[10, 11, 12, 13, 14, 15],
        plan_source="manual",
    )
    preflight.persist_device_plan(out1, soc_name="Ascend950PR_9599", device_id=1, device_node=node1)

    loaded0 = json.loads(out0.read_text(encoding="utf-8"))
    loaded1 = json.loads(out1.read_text(encoding="utf-8"))
    assert set(loaded0["socs"]["Ascend950PR_9599"]["devices"]) == {"0"}
    assert set(loaded1["socs"]["Ascend950PR_9599"]["devices"]) == {"1"}


def test_cli_offline_allowed_cpus(tmp_path: Path):
    out = tmp_path / "plan.json"
    rc = preflight.main(
        [
            "--device",
            "2",
            "--soc",
            "Ascend950PR_9599",
            "--allowed-cpus",
            "3,4,5,6,7",
            "--occupy-cpus",
            "3,4,5,6,7,8",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["socs"]["Ascend950PR_9599"]["devices"]["2"]["allowed_cpus"] == [3, 4, 5, 6, 7]


def test_default_plan_path_is_per_device_and_honors_base_or_template(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(preflight.PLAN_ENV, raising=False)
    assert preflight.default_plan_path(2) == tmp_path / "build/config/aicpu_affinity_plan.2.json"

    monkeypatch.setenv(preflight.PLAN_ENV, str(tmp_path / "custom.json"))
    assert preflight.default_plan_path(3) == tmp_path / "custom.3.json"
    monkeypatch.setenv(preflight.PLAN_ENV, str(tmp_path / "custom.{device}.json"))
    assert preflight.default_plan_path(4) == tmp_path / "custom.4.json"
    monkeypatch.setenv(preflight.PLAN_ENV, str(tmp_path / "custom.plan"))
    assert preflight.default_plan_path(5) == tmp_path / "custom.5.plan"
    assert preflight.cpus_side_path(preflight.default_plan_path(5)) == tmp_path / "custom.5.cpus"


def test_timeout_record_is_per_device_and_clearable(tmp_path: Path):
    plan = tmp_path / "plan.2.json"
    marker = preflight.write_timeout_record(plan, 2, reason="test")
    assert marker == preflight.timeout_record_path(plan)
    assert preflight.timeout_record_looks_usable(marker, 2)
    assert not preflight.timeout_record_looks_usable(marker, 1)

    preflight.clear_timeout_record(plan)
    assert not marker.exists()


def test_helper_cache_stamp_is_atomic_and_fingerprint_specific(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(preflight, "_CACHE_STAMP", tmp_path / ".build.stamp")
    preflight._write_cache_stamp("fingerprint-a")
    assert preflight._cache_stamp_matches("fingerprint-a")
    assert not preflight._cache_stamp_matches("fingerprint-b")


def test_probe_timeout_writes_record_and_manual_probe_clears_it(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    out = tmp_path / "plan.0.json"

    def timeout_probe(_device_id: int):
        raise preflight.AffinityProbeTimeout("probe", preflight.PREFLIGHT_TIMEOUT_SECONDS)

    monkeypatch.setattr(preflight, "run_affinity_probe", timeout_probe)
    assert preflight.main(["--device", "0", "--probe", "--plan-source", "auto-first-run", "--out", str(out)]) == 1
    marker = preflight.timeout_record_path(out)
    assert preflight.timeout_record_looks_usable(marker, 0)

    def fail_probe(_device_id: int):
        raise RuntimeError("manual retry")

    monkeypatch.setattr(preflight, "run_affinity_probe", fail_probe)
    assert preflight.main(["--device", "0", "--probe", "--out", str(out)]) == 1
    assert not marker.exists()


def test_helper_build_timeout_does_not_write_timeout_record(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    out = tmp_path / "plan.0.json"

    def build_timeout(_device_id: int):
        raise subprocess.TimeoutExpired("cmake --build", preflight.HELPER_BUILD_TIMEOUT_SECONDS)

    monkeypatch.setattr(preflight, "run_affinity_probe", build_timeout)
    assert preflight.main(["--device", "0", "--probe", "--plan-source", "auto-first-run", "--out", str(out)]) == 1
    assert not preflight.timeout_record_path(out).exists()
    assert not out.exists()


def test_query_device_hal_probe_budget_starts_after_helper_build(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    host_bin = tmp_path / "query_device_hal"
    host_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    calls: list[dict[str, object]] = []
    clock = {"now": 1000.0}

    def fake_monotonic() -> float:
        return clock["now"]

    def capture_run(command, **kwargs):
        calls.append({"command": list(command), "timeout": kwargs.get("timeout")})
        return subprocess.CompletedProcess(command, 0, stdout="{}", stderr="")

    monkeypatch.setattr(preflight.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(
        preflight,
        "_ensure_query_device_hal_artifacts",
        lambda: (host_bin, {"ASCEND_HOME_PATH": "/ascend"}),
    )
    monkeypatch.setattr(preflight.subprocess, "run", capture_run)

    # Simulate a long helper build finishing before the probe starts.
    clock["now"] = 1000.0 + 120.0
    assert preflight._run_query_device_hal(0, "--rtt-json") == "{}"
    assert calls and calls[0]["command"] == [str(host_bin), "0", "--rtt-json"]
    assert calls[0]["timeout"] == pytest.approx(preflight.PREFLIGHT_TIMEOUT_SECONDS, abs=0.01)


def test_probe_failure_writes_no_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    out = tmp_path / "plan.0.json"

    def fail_probe(_device_id: int):
        raise RuntimeError("synthetic probe failure")

    monkeypatch.setattr(preflight, "run_affinity_probe", fail_probe)
    assert preflight.main(["--device", "0", "--probe", "--out", str(out)]) == 1
    assert not out.exists()
    assert not preflight.cpus_side_path(out).exists()


def test_probe_failure_does_not_modify_existing_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    out = tmp_path / "plan.0.json"
    side = preflight.cpus_side_path(out)
    out.write_text("old-json\n", encoding="utf-8")
    side.write_text("old-side\n", encoding="utf-8")

    def fail_probe(_device_id: int):
        raise RuntimeError("synthetic probe failure")

    monkeypatch.setattr(preflight, "run_affinity_probe", fail_probe)
    assert preflight.main(["--device", "0", "--probe", "--out", str(out)]) == 1
    assert out.read_text(encoding="utf-8") == "old-json\n"
    assert side.read_text(encoding="utf-8") == "old-side\n"


def test_pair_publish_rolls_back_when_side_write_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    out = tmp_path / "plan.0.json"
    side = preflight.cpus_side_path(out)
    out.write_text("old-json\n", encoding="utf-8")
    side.write_text("old-side\n", encoding="utf-8")
    node = preflight.build_device_node_from_allowed(
        soc_name="Ascend950PR_9599",
        device_id=0,
        allowed_cpus=[3, 4, 5, 6, 7],
        occupy_cpus=[3, 4, 5, 6, 7, 8],
    )
    plan = preflight.make_device_plan(soc_name="Ascend950PR_9599", device_id=0, device_node=node)
    original_write = preflight._atomic_write_text

    def fail_side(path: Path, text: str):
        if path == side:
            raise OSError("synthetic side write failure")
        original_write(path, text)

    monkeypatch.setattr(preflight, "_atomic_write_text", fail_side)
    with pytest.raises(OSError, match="synthetic side"):
        preflight.atomic_write_plan_and_side(out, plan, node)
    assert out.read_text(encoding="utf-8") == "old-json\n"
    assert side.read_text(encoding="utf-8") == "old-side\n"


@pytest.mark.parametrize(
    "allowed,occupy,error",
    [
        ([3, 4, 5, 6], [3, 4, 5, 6, 7], "allowed_cpus must contain"),
        ([3, 4, 5, 6, 7, 8], [3, 4, 5, 6, 7, 8], "allowed_cpus must contain"),
        ([3, 4, 5, 6, 64], [3, 4, 5, 6, 64], "CPU ids"),
        ([3, 4, 5, 6, 9], [3, 4, 5, 6, 7, 8], "subset"),
    ],
)
def test_manual_plan_rejects_invalid_allowed_or_occupy(allowed: list[int], occupy: list[int], error: str):
    with pytest.raises(ValueError, match=error):
        preflight.build_device_node_from_allowed(
            soc_name="Ascend950PR_9599", device_id=0, allowed_cpus=allowed, occupy_cpus=occupy
        )


def test_side_file_validation_rejects_stale_or_legacy_contract(tmp_path: Path):
    legacy = tmp_path / "legacy.cpus"
    legacy.write_text("soc=Ascend950PR_9599\nsource=manual\ncpus=3,4,5,6,7\n", encoding="utf-8")
    assert not preflight.side_file_looks_usable(legacy, 0)

    node = preflight.build_device_node_from_allowed(
        soc_name="Ascend950PR_9599",
        device_id=0,
        allowed_cpus=[3, 4, 5, 6, 7],
        occupy_cpus=[3, 4, 5, 6, 7, 8],
    )
    plan = tmp_path / "plan.0.json"
    preflight.persist_device_plan(plan, soc_name="Ascend950PR_9599", device_id=0, device_node=node)
    side = preflight.cpus_side_path(plan)
    assert preflight.side_file_looks_usable(side, 0)
    assert preflight.plan_files_look_usable(plan, 0)
    assert not preflight.side_file_looks_usable(side, 1)

    loaded = json.loads(plan.read_text(encoding="utf-8"))
    loaded["socs"]["Ascend950PR_9599"]["devices"]["0"]["plan_source"] = "auto-first-run"
    plan.write_text(json.dumps(loaded), encoding="utf-8")
    assert not preflight.plan_files_look_usable(plan, 0)

    malformed = side.read_text(encoding="utf-8").replace("cpus=3,4,5,6,7", "cpus=3,,4,5,6,7")
    side.write_text(malformed, encoding="utf-8")
    assert not preflight.side_file_looks_usable(side, 0)


def test_cmake_build_passes_project_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    calls: list[list[str]] = []

    def record(command: list[str], **_kwargs):
        calls.append(command)

    monkeypatch.setattr(preflight.subprocess, "run", record)
    preflight._cmake_build(tmp_path / "src", tmp_path / "build", {"ASCEND_HOME_PATH": "/ascend"})
    assert f"-DSIMPLER_ROOT={preflight.PROJECT_ROOT}" in calls[0]
