# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""HBG CSA contracts without a card or the large data snapshot."""

import hashlib
import importlib
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch
from simpler.task_interface import TensorTransfer

from simpler_setup import TensorArg

_PACKAGE = "examples.a5.host_build_graph.deepseek_v4_pro_csa"
workload = importlib.import_module(f"{_PACKAGE}.workload")
scene = importlib.import_module(f"{_PACKAGE}.test_deepseek_v4_pro_csa")
perf = importlib.import_module(f"{_PACKAGE}.analyze_perf")
runner = importlib.import_module(f"{_PACKAGE}.run")


def test_hbg_imports_without_the_tmr_case():
    script = textwrap.dedent("""
        import importlib
        import importlib.abc
        import sys

        class RejectTMR(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.startswith("examples.a5.tensormap_and_ringbuffer"):
                    raise ModuleNotFoundError(f"TMR case is unavailable: {fullname}")
                return None

        sys.meta_path.insert(0, RejectTMR())
        package = "examples.a5.host_build_graph.deepseek_v4_pro_csa"
        for module in ("workload", "test_deepseek_v4_pro_csa", "run", "analyze_perf"):
            importlib.import_module(f"{package}.{module}")
    """)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=workload.REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_hbg_callable_owns_all_frozen_sources():
    hbg = scene.TestDeepSeekV4ProCSAHBG
    assert len(hbg.CALLABLE["incores"]) == 54
    assert len(hbg.CALLABLE["orchestration"]["signature"]) == 52
    assert hbg.CALLABLE["orchestration"]["source"] == str(workload.SOURCE)
    sources = [hbg.CALLABLE["orchestration"], *hbg.CALLABLE["incores"]]
    relative_sources = {str(Path(item["source"]).relative_to(workload.CASE_DIR)) for item in sources}
    assert relative_sources == set(workload.fixture.manifest()["sources"])
    assert workload.fixture.DATA_DIR == workload.CASE_DIR / "data"
    assert workload.fixture.manifest()["runtime"] == "host_build_graph"
    assert hbg._st_runtime == "host_build_graph"


def test_orchestration_only_adapts_tensor_namespace_and_launch_api():
    source = workload.SOURCE.read_text()
    assert source.count(".launch_spec.set_block_num(") == 30
    original = source.replace("simpler::hbg::Tensor", "Tensor")
    original = original.replace(".launch_spec.set_block_num(", ".launch_spec.set_core_num(")
    assert hashlib.sha256(original.encode()).hexdigest() == workload.fixture.manifest()["source_orchestration_sha256"]


def test_only_attention_scale_is_host_only(monkeypatch):
    arguments = [{**arg, "shape": [1], "dtype": "torch.float32"} for arg in workload.fixture.manifest()["arguments"]]
    tensors = {arg["name"]: torch.ones(1) for arg in arguments if arg["direction"] != "OUT"}
    monkeypatch.setattr(workload.fixture, "manifest", lambda: {"arguments": arguments})
    monkeypatch.setattr(workload.fixture, "verify_snapshot", lambda: None)
    monkeypatch.setattr(workload.fixture, "load_tensor", lambda name, section: tensors[name])
    hbg = scene.TestDeepSeekV4ProCSAHBG().generate_args({})
    assert len(hbg.specs) == len(arguments) == 52
    for spec, arg in zip(hbg.specs, arguments, strict=True):
        assert spec.name == arg["name"]
        if spec.name == "hc_attn_scale":
            assert spec.transfer == TensorTransfer.NONE
        else:
            assert spec.transfer == TensorArg(spec.name, spec.value).transfer
        if arg["direction"] == "OUT":
            assert torch.equal(spec.value, torch.zeros(1))
        else:
            assert spec.value is tensors[spec.name]


def _timing_log(path, count=4):
    lines = []
    for index in range(count):
        for name, duration, clock in (
            ("chip.run", 100000 + index * 1000, ""),
            ("chip.run.runner_run.device_wall", 20000 + index * 1000, " clk=dev"),
        ):
            lines.append(
                f"[STRACE] v=1 pid=1 tid=1 inv={index + 1} hid=abc depth={name.count('.') - 1} "
                f"name={name} ts=0 dur={duration}{clock}"
            )
    path.write_text("\n".join(lines))
    return path


def test_hbg_measurement_uses_only_host_and_device_windows(tmp_path):
    log = _timing_log(tmp_path / "bench.log")
    result = perf.summarize(log, warmup=1, rounds=3)
    assert set(result["metrics"]) == {"host_us", "device_us"}
    assert result["metrics"]["device_us"]["samples"] == [21, 22, 23]
    assert result["metrics"]["device_us"]["median"] == 22
    assert result["metrics"]["host_us"]["median"] == 102


@pytest.mark.parametrize("mutation", ["host_clock", "missing_marker", "dropped", "torn", "missing_round"])
def test_invalid_hbg_measurement_is_rejected(tmp_path, mutation):
    log = _timing_log(tmp_path / "bench.log", count=1)
    text = log.read_text()
    if mutation == "host_clock":
        text = text.replace(" clk=dev", "")
    elif mutation == "missing_marker":
        text = text.splitlines()[0]
    elif mutation == "dropped":
        text += (
            "\n[HOSTLOG_DROPS] v=1 pid=1 new=1 total=1 queue_full=1 claim_exhausted=0 output_failed=0 not_admitted=0"
        )
    elif mutation == "torn":
        text += "\n[STRACE] v=1 pid=1 tid=1 inv=2"
    log.write_text(text)
    with pytest.raises(ValueError):
        perf.summarize(log, warmup=0, rounds=2 if mutation == "missing_round" else 1)


def _trace(monkeypatch, producer="aicpu", identity=True):
    raw = {
        "metadata": {
            "runtime": "host_build_graph",
            "orchestrator_source": "host",
            "host_capture": {"status": "complete", "dropped_records": 0, "recorded_records": 1, "expected_records": 1},
        },
        "scheduler_tasks": {"producer": producer, "records": [[0]]},
        "scheduler_records": {
            "streams": [
                {
                    "producer": producer,
                    "records": [[0]],
                    "capture": {"committed": 1, "dropped": 0, "truncated": False},
                }
            ]
        },
        "host_orchestrator_phases": [[{}]],
    }
    parsed = {
        "tasks": [{"dispatch_time_us": 1, "finish_time_us": 2}],
        "timeline_metadata": {"clock_alignment": {"status": "bounded"}, "cross_domain_latency_available": True},
    }
    summary = {} if identity is None else {"block_identity_verified": identity}
    monkeypatch.setattr(runner, "validate_task_capture", lambda *args: (raw, parsed, summary))
    return raw, parsed


def test_aicpu_fallback_is_visible_and_fails_distributed_gate(monkeypatch):
    _trace(monkeypatch)
    result = runner.validate_trace(None, None, 4)
    assert result["scheduler_producer"] == "aicpu"
    assert result["distributed_scheduler_verified"] is False
    with pytest.raises(ValueError, match="Distributed scheduler required"):
        runner.validate_trace(None, None, 4, require_aicore=True)


def test_aicore_trace_passes_distributed_gate(monkeypatch):
    _trace(monkeypatch, "aicore")
    result = runner.validate_trace(None, None, 4, require_aicore=True)
    assert result["scheduler_producer"] == "aicore"
    assert result["distributed_scheduler_verified"] is True
    assert result["clock_alignment_status"] == "bounded"


@pytest.mark.parametrize("identity", (False, None))
def test_resident_counts_without_explicit_block_identity_cannot_pass_strict_gate(monkeypatch, identity):
    _trace(monkeypatch, "aicore", identity)
    with pytest.raises(ValueError, match="identity"):
        runner.validate_trace(None, None, 4, require_aicore=True)


def test_l1_cannot_establish_scheduler_provenance(monkeypatch):
    raw, _ = _trace(monkeypatch, "aicore")
    raw["metadata"].pop("orchestrator_source")
    raw.pop("scheduler_tasks")
    raw.pop("host_orchestrator_phases")
    result = runner.validate_trace(None, None, 1)
    assert result["scheduler_producer"] is None
    assert result["distributed_scheduler_verified"] is False
    with pytest.raises(ValueError, match="Distributed scheduler required"):
        runner.validate_trace(None, None, 1, require_aicore=True)


@pytest.mark.parametrize(
    "mutation",
    [
        "runtime",
        "producer",
        "records",
        "join",
        "stream",
        "dropped",
        "truncated",
        "committed",
        "host",
        "phases",
        "host_source",
        "host_rows",
    ],
)
def test_invalid_l4_capture_is_rejected(monkeypatch, mutation):
    raw, parsed = _trace(monkeypatch)
    if mutation == "runtime":
        raw["metadata"]["runtime"] = "tensormap_and_ringbuffer"
    elif mutation == "producer":
        raw["scheduler_tasks"]["producer"] = None
    elif mutation == "records":
        raw["scheduler_tasks"]["records"] = []
    elif mutation == "join":
        parsed["tasks"][0]["finish_time_us"] = None
    elif mutation == "stream":
        raw["scheduler_records"]["streams"][0]["producer"] = "aicore"
    elif mutation in ("dropped", "truncated", "committed"):
        raw["scheduler_records"]["streams"][0]["capture"][mutation] = 5
    elif mutation == "host":
        raw["metadata"]["host_capture"]["status"] = "incomplete"
    elif mutation == "phases":
        raw["host_orchestrator_phases"] = []
    elif mutation == "host_source":
        raw["metadata"].pop("orchestrator_source")
    elif mutation == "host_rows":
        raw["host_orchestrator_phases"] = [[]]
    with pytest.raises(ValueError):
        runner.validate_trace(None, None, 4)
