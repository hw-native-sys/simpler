# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CPU contract checks for the self-contained HBG CSA workload."""

import ast
import importlib

import pytest
import torch

_PACKAGE = "examples.a5.host_build_graph.deepseek_v4_pro_csa"
fixture = importlib.import_module(f"{_PACKAGE}.fixture")
golden = importlib.import_module(f"{_PACKAGE}.golden")
config = importlib.import_module(f"{_PACKAGE}.kernel_config")
perf = importlib.import_module(f"{_PACKAGE}.analyze_perf")


def test_frozen_callable_and_source_contract():
    manifest = fixture.manifest()
    assert len(manifest["arguments"]) == len(config.ORCHESTRATION["signature"]) == 52
    assert len(manifest["files"]) == 58
    assert len(config.KERNELS) == 54
    assert [item["func_id"] for item in config.KERNELS] == list(range(54))
    for arg, direction in zip(manifest["arguments"], config.ORCHESTRATION["signature"], strict=True):
        assert arg["direction"] == direction.name
    assert sum(arg["direction"] != "IN" for arg in manifest["arguments"]) == 7
    for path, digest in manifest["sources"].items():
        assert fixture.sha256(fixture.CASE_DIR / path) == digest


def test_no_pypto_imports():
    for path in fixture.CASE_DIR.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                assert all(not name.name.startswith("pypto") for name in node.names), path
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("pypto"), path


def test_snapshot_import_is_validated_and_non_overwriting(tmp_path, monkeypatch):
    source, destination = tmp_path / "source", tmp_path / "destination"
    (source / "in").mkdir(parents=True)
    original = source / "in/x.pt"
    original.write_bytes(b"frozen-test-payload")
    metadata = {"files": {"in/x.pt": {"bytes": original.stat().st_size, "sha256": fixture.sha256(original)}}}
    monkeypatch.setattr(fixture, "manifest", lambda: metadata)
    fixture.import_snapshot(source, destination)
    fixture.import_snapshot(source, destination)
    target = destination / "in/x.pt"
    target.write_bytes(b"different-payload")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        fixture.import_snapshot(source, destination)
    assert target.read_bytes() == b"different-payload"
    with pytest.raises(ValueError, match="does not match"):
        fixture.verify_snapshot(destination)
    with pytest.raises(FileNotFoundError, match="Missing CSA snapshot"):
        fixture.verify_snapshot(tmp_path / "absent")


def _small_fixture():
    outputs = {
        "compress_state": torch.zeros(3, 4, 2048),
        "inner_compress_state": torch.zeros(3, 4, 512),
        "kv_cache": torch.zeros(1, 128, 1, 512, dtype=torch.bfloat16),
        "cmp_kv": torch.zeros(1, 128, 1, 512, dtype=torch.bfloat16),
        "idx_kv_cache": torch.zeros(1, 128, 1, 128, dtype=torch.float8_e4m3fn),
        "idx_kv_scale": torch.zeros(1, 128, 1, 1),
        "x_out": torch.zeros(8, 4, 8),
    }
    inputs = {
        name: torch.arange(8)
        for name in (
            "state_slot_mapping",
            "inner_state_slot_mapping",
            "ori_slot_mapping",
            "idx_slot_mapping",
            "cmp_slot_mapping",
        )
    }
    inputs.update(
        {
            "compress_state_block_table": torch.zeros(4, 1, dtype=torch.int32),
            "cmp_norm_w": torch.ones(512, dtype=torch.bfloat16),
            "position_ids": torch.tensor([0, 1] * 4, dtype=torch.int32),
            "freqs_cos": torch.ones(4, 32, dtype=torch.bfloat16),
            "freqs_sin": torch.zeros(4, 32, dtype=torch.bfloat16),
        }
    )
    return outputs, inputs


def test_all_seven_outputs_are_checked():
    expected, inputs = _small_fixture()
    golden.validate_outputs(expected, expected, inputs)
    incomplete = dict(expected)
    incomplete.pop("idx_kv_scale")
    with pytest.raises(AssertionError, match="All seven"):
        golden.validate_outputs(incomplete, expected, inputs)


@pytest.mark.parametrize("name", list(golden.comparators()))
def test_each_output_detects_wrong_mapped_results(name):
    expected, inputs = _small_fixture()
    actual = {key: value.clone() for key, value in expected.items()}
    if name == "x_out":
        actual[name].fill_(99)
    else:
        actual[name].reshape(actual[name].shape[0] * actual[name].shape[1], -1)[0].fill_(99)
    with pytest.raises(AssertionError, match=name if name != "compress_state" else "compress"):
        golden.validate_outputs(actual, expected, inputs)


@pytest.mark.parametrize("name", [name for name in golden.comparators() if name != "x_out"])
def test_unmapped_rows_must_remain_exact(name):
    expected, inputs = _small_fixture()
    actual = {key: value.clone() for key, value in expected.items()}
    actual[name].reshape(actual[name].shape[0] * actual[name].shape[1], -1)[8].fill_(1)
    with pytest.raises(AssertionError, match="unmapped"):
        golden.validate_outputs(actual, expected, inputs)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_output_cannot_pass(value):
    expected, inputs = _small_fixture()
    actual = dict(expected, x_out=expected["x_out"].clone())
    actual["x_out"].view(-1)[0] = value
    with pytest.raises(AssertionError, match="illegal values"):
        golden.validate_outputs(actual, expected, inputs)


def test_cache_comparator_validates_state_independently():
    expected, inputs = _small_fixture()
    actual = {key: value.clone() for key, value in expected.items()}
    actual["compress_state"][0, 0].fill_(10)
    ok, _ = golden.comparators()["cmp_kv"](
        actual["cmp_kv"],
        expected["cmp_kv"],
        actual_outputs=actual,
        expected_outputs=expected,
        inputs=inputs,
        rtol=1e-2,
        atol=1e-2,
    )
    assert not ok


def test_mapping_corruption_is_not_tolerated():
    expected, inputs = _small_fixture()
    inputs["ori_slot_mapping"][1] = inputs["ori_slot_mapping"][0]
    with pytest.raises(AssertionError, match="duplicate"):
        golden.validate_outputs(expected, expected, inputs)


def _timing_log(path, count):
    lines = []
    for index in range(count):
        for name, start, duration in (
            ("chip.run", 0, 100000),
            ("chip.run.runner_run.device_wall", 100, 20000 + index * 1000),
        ):
            lines.append(
                f"[STRACE] v=1 pid=1 tid=1 inv={index + 1} hid=abc depth={name.count('.') - 1} "
                f"name={name} ts={start} dur={duration}" + (" clk=dev" if "device_wall" in name else "")
            )
    path.write_text("\n".join(lines))
    return path


def test_performance_uses_device_window_and_excludes_warmup(tmp_path):
    summary = perf.summarize(_timing_log(tmp_path / "run.log", 4), warmup=1, rounds=3)
    assert summary["metrics"]["device_us"]["samples"] == [21, 22, 23]
    assert summary["metrics"]["device_us"]["median"] == 22
    assert summary["metrics"]["host_us"]["median"] == 100


def test_incomplete_benchmark_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Expected 4 rounds"):
        perf.summarize(_timing_log(tmp_path / "run.log", 3), warmup=1, rounds=3)


def test_empty_benchmark_is_rejected(tmp_path):
    log = tmp_path / "empty.log"
    log.write_text("")
    with pytest.raises(ValueError, match="Expected one CSA callable"):
        perf.summarize(log)


def test_host_clock_cannot_replace_device_timing(tmp_path):
    log = _timing_log(tmp_path / "run.log", 1)
    log.write_text(log.read_text().replace(" clk=dev", ""))
    with pytest.raises(ValueError, match="device clock domain"):
        perf.summarize(log, warmup=0, rounds=1)


def test_partial_trace_record_is_rejected(tmp_path):
    log = _timing_log(tmp_path / "run.log", 1)
    log.write_text(log.read_text() + "\n[STRACE] v=1 pid=1 tid=1 inv=2")
    with pytest.raises(ValueError, match="incomplete records"):
        perf.summarize(log, warmup=0, rounds=1)


def test_dropped_trace_record_is_rejected(tmp_path):
    log = _timing_log(tmp_path / "run.log", 1)
    log.write_text(
        log.read_text()
        + "\n[HOSTLOG_DROPS] v=1 pid=1 new=1 total=1 queue_full=1 claim_exhausted=0 output_failed=0 not_admitted=0"
    )
    with pytest.raises(ValueError, match="dropped records"):
        perf.summarize(log, warmup=0, rounds=1)
