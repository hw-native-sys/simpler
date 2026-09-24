# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
DRIVER = ROOT / "examples/a2a3/host_build_graph/qwen3_14b_decode_worker_submit/main.py"


def _load_driver():
    spec = importlib.util.spec_from_file_location("_qwen_worker_submit_test_driver", DRIVER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeTaskArgs:
    def __init__(self):
        self.items = []

    def add_tensor(self, tensor, tag):
        self.items.append((tensor, tag))


class _FakeBuffer:
    def __init__(self, name):
        self.name = name

    def tensor(self, shape, dtype):
        return self.name, tuple(shape), dtype


def test_manifest_freezes_bounded_worker_submit_contract():
    manifest = json.loads(
        (ROOT / "examples/a2a3/host_build_graph/qwen3_14b_decode_worker_submit/workload_manifest.json").read_text()
    )
    assert manifest["model"] == "Qwen3-14B"
    assert manifest["batch"] == 16
    assert manifest["execution"] == {
        "platform": "a2a3",
        "runtime": "host_build_graph",
        "worker_level": 3,
        "endpoint": "single_local_chip",
        "task_shape": "single_NEXT_LEVEL",
        "depths": [1, 2],
        "hardware_qualified_launch_depths": [1],
    }
    assert manifest["kv_cache"]["address_space"] == "HOST"
    assert manifest["kv_cache"]["mutable_per_run"] is True


def test_task_args_preserve_signature_directions(monkeypatch):
    driver = _load_driver()
    monkeypatch.setattr(driver, "TaskArgs", _FakeTaskArgs)
    monkeypatch.setattr(driver, "torch_dtype_to_datatype", lambda dtype: SimpleNamespace(value=7))
    specs = [
        SimpleNamespace(name="input", shape=(2,), dtype="FLOAT32"),
        SimpleNamespace(name="state", shape=(2,), dtype="FLOAT32"),
        SimpleNamespace(name="output", shape=(2,), dtype="FLOAT32"),
    ]
    signature = [driver.ArgDirection.IN, driver.ArgDirection.INOUT, driver.ArgDirection.OUT]
    common = {"input": _FakeBuffer("common-input")}
    run = {"state": _FakeBuffer("run-state"), "output": _FakeBuffer("run-output")}

    args = driver._task_args(specs, signature, common, run)

    assert args.items == [
        (("common-input", (2,), 7), driver.TensorArgType.INPUT),
        (("run-state", (2,), 7), driver.TensorArgType.INOUT),
        (("run-output", (2,), 7), driver.TensorArgType.OUTPUT_EXISTING),
    ]


@pytest.mark.parametrize("depth", [0, 3])
def test_cli_rejects_runtime_depth_outside_bounded_range(depth):
    driver = _load_driver()
    with pytest.raises(SystemExit):
        driver.parse_args(["--depth", str(depth)])


@pytest.mark.parametrize("depth", [1, 2])
def test_driver_propagates_launch_depth_to_worker(monkeypatch, depth):
    driver = _load_driver()
    events = []

    class Handle:
        def wait(self):
            events.append("wait")

    class Worker:
        def __init__(self, **kwargs):
            assert kwargs["launch_depth"] == depth

        def register(self, chip):
            return chip

        def init(self):
            events.append("init")

        def submit(self, callback, args, config):
            callback(SimpleNamespace(submit_next_level=lambda *a, **k: events.append("dispatch")), args, config)
            return Handle()

        def close(self):
            events.append("close")

    base = SimpleNamespace(
        _chip_spec=lambda *args: {"name": "decode"},
        l3_compile_cache_key=lambda *args: "key",
        _build_config=lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(driver, "_base_driver", lambda: base)
    monkeypatch.setattr(driver, "compile_chip_callable_spec", lambda *args: object())
    monkeypatch.setattr(driver, "Worker", Worker)
    monkeypatch.setattr(driver, "_build_host_buffers", lambda *a, **k: ([], [], {}, [{} for _ in range(depth)]))
    assert driver.run([0], depth=depth, seed=1, seq_len=2, skip_golden=True) == 0
    assert events == ["init", *(["dispatch"] * depth), *(["wait"] * depth), "close"]
