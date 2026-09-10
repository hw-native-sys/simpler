# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Device-free lifecycle tests for the standalone Qwen decode drivers."""

from __future__ import annotations

import importlib.util
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from simpler import task_interface

ROOT = Path(__file__).resolve().parents[3]


class _Payload:
    def __init__(self, name: str):
        self.name = name
        self.shape = (1,)
        self.dtype = torch.float32
        self.device = SimpleNamespace(type="cpu")

    def numel(self):
        return 1

    def element_size(self):
        return 4

    def is_contiguous(self):
        return True


class _FakeBuffer:
    def __init__(self, index: int):
        self.index = index

    def tensor(self, shape, dtype):
        return self.index, shape, dtype


class _FakeTaskArgs:
    def __init__(self):
        self.tensors = []

    def add_tensor(self, tensor, tag) -> None:
        self.tensors.append((tensor, tag))


class _FakeCallConfig:
    def __init__(self):
        self.runtime_env = SimpleNamespace(ring_task_window=-1, ring_heap=-1, ring_dep_pool=-1)
        self.enable_chip_swimlane = 0
        self.enable_dump_args = 0
        self.enable_pmu = 0
        self.enable_dep_gen = False
        self.enable_scope_stats = False
        self.output_prefix = ""


class _FakeWorker:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.events = []
        self.allocations = []
        self.uploads = []
        self.runs = []
        self.copy_from_count = 0
        self.closed = False
        self.__class__.instances.append(self)

    def register(self, chip):
        self.events.append("register")
        return ("handle", chip)

    def init(self) -> None:
        self.events.append("init")

    def malloc(self, nbytes: int) -> _FakeBuffer:
        buffer = _FakeBuffer(len(self.allocations))
        self.allocations.append((nbytes, buffer))
        self.events.append("malloc")
        return buffer

    def copy_to(self, buffer: _FakeBuffer, tensor: _Payload) -> None:
        self.uploads.append((buffer.index, tensor.name))
        self.events.append("copy_to")

    def run(self, handle, task_args, config) -> None:
        self.runs.append((handle, task_args, config))
        self.events.append("run")

    def copy_from(self, actual, buffer) -> None:
        self.copy_from_count += 1

    def free(self, buffer) -> None:
        self.events.append(f"free:{buffer.index}")

    def close(self) -> None:
        self.closed = True
        self.events.append("close")


def _load_driver(relative_path: str):
    module_name = f"_test_qwen_driver_{relative_path.replace('/', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, ROOT / relative_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Qwen driver at {relative_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _fixture_stream(driver, *, n_layers: int):
    """Yield tiny payloads and assert the consumer releases each before next()."""
    previous = None
    for param in driver.param_specs(n_layers):
        if previous is not None:
            assert previous() is None
        payload = _Payload(param.name)
        previous = weakref.ref(payload)
        yield param.name, payload
        del payload
    assert previous is not None and previous() is None


@pytest.mark.parametrize(
    ("relative_path", "platform", "expected_dep_pool"),
    [
        ("examples/a2a3/tensormap_and_ringbuffer/qwen3_14b_decode/main.py", "a2a3", 0),
        ("examples/a5/tensormap_and_ringbuffer/qwen3_14b_decode/main.py", "a5", 65536),
    ],
)
@pytest.mark.parametrize("runtime", ["host_build_graph", "tensormap_and_ringbuffer"])
def test_standalone_driver_keeps_one_device_fixture_across_rounds(
    monkeypatch, relative_path: str, platform: str, expected_dep_pool: int, runtime: str
):
    driver = _load_driver(relative_path)
    _FakeWorker.instances.clear()
    chip = object()

    monkeypatch.setattr(driver, "Worker", _FakeWorker)
    monkeypatch.setattr(task_interface, "TaskArgs", _FakeTaskArgs)
    monkeypatch.setattr(driver, "CallConfig", _FakeCallConfig)
    monkeypatch.setattr(driver, "compile_chip_callable_spec", lambda *args: chip)
    monkeypatch.setattr(driver, "l3_compile_cache_key", lambda *args: "cache-key")
    monkeypatch.setattr(
        driver,
        "log_torch_backend_autoload_once",
        lambda: _FakeWorker.instances[-1].events.append("autoload"),
    )
    monkeypatch.setattr(
        driver,
        "effective_diagnostic_options",
        lambda *args, **kwargs: SimpleNamespace(
            chip_swimlane=0,
            dump_args=0,
            pmu=0,
            dep_gen=False,
            scope_stats=False,
            swimlane_overhead=False,
        ),
    )
    monkeypatch.setattr(
        driver,
        "param_tensors",
        lambda *, seed, seq_len, n_layers: _fixture_stream(driver, n_layers=n_layers),
    )

    def unexpected_golden(*args, **kwargs):
        raise AssertionError("--skip-golden must not materialize or compute a golden")

    monkeypatch.setattr(driver, "_decode_generate_inputs", unexpected_golden)
    monkeypatch.setattr(driver, "_decode_golden", unexpected_golden)

    assert driver.run([7], platform, runtime=runtime, rounds=3, skip_golden=True) == 0

    [worker] = _FakeWorker.instances
    names = [param.name for param in driver.param_specs(driver.N_LAYERS)]
    assert worker.kwargs == {"level": 2, "platform": platform, "runtime": runtime, "device_id": 7}
    assert len(worker.allocations) == 20 == len(names)
    assert [name for _, name in worker.uploads] == [name for name in names if name != "out"]
    assert len(worker.runs) == 3
    assert len({id(task_args) for _, task_args, _ in worker.runs}) == 1
    assert len({id(config) for _, _, config in worker.runs}) == 1
    task_args = worker.runs[0][1]
    signature = driver.TestQwen314BDecode.CALLABLE["orchestration"]["signature"]
    assert [tag for _, tag in task_args.tensors] == [
        {
            driver.D.IN: task_interface.TensorArgType.INPUT,
            driver.D.INOUT: task_interface.TensorArgType.INOUT,
            driver.D.OUT: task_interface.TensorArgType.OUTPUT_EXISTING,
        }[direction]
        for direction in signature
    ]
    assert worker.runs[0][2].runtime_env.ring_dep_pool == expected_dep_pool
    assert worker.copy_from_count == 0
    assert worker.events.index("autoload") > max(i for i, event in enumerate(worker.events) if event == "copy_to")
    assert worker.events.index("autoload") < worker.events.index("run")
    assert [event for event in worker.events if event.startswith("free:")] == [f"free:{i}" for i in reversed(range(20))]
    assert worker.closed


@pytest.mark.parametrize(
    ("relative_path", "platform"),
    [
        ("examples/a2a3/tensormap_and_ringbuffer/qwen3_14b_decode/main.py", "a2a3"),
        ("examples/a5/tensormap_and_ringbuffer/qwen3_14b_decode/main.py", "a5"),
    ],
)
def test_correctness_reuses_one_materialized_fixture(monkeypatch, relative_path: str, platform: str) -> None:
    driver = _load_driver(relative_path)
    _FakeWorker.instances.clear()
    chip = object()
    names = [param.name for param in driver.param_specs(driver.N_LAYERS)]
    fixture = SimpleNamespace(**{name: _Payload(name) for name in names})
    generated = []
    golden_calls = []
    compared = []
    finalized = []

    monkeypatch.setattr(driver, "Worker", _FakeWorker)
    monkeypatch.setattr(task_interface, "TaskArgs", _FakeTaskArgs)
    monkeypatch.setattr(driver, "CallConfig", _FakeCallConfig)
    monkeypatch.setattr(driver, "compile_chip_callable_spec", lambda *args: chip)
    monkeypatch.setattr(driver, "l3_compile_cache_key", lambda *args: "cache-key")
    monkeypatch.setattr(
        driver,
        "log_torch_backend_autoload_once",
        lambda: _FakeWorker.instances[-1].events.append("autoload"),
    )
    monkeypatch.setattr(
        driver,
        "effective_diagnostic_options",
        lambda _rounds, **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(driver, "build_output_prefix", lambda _label: Path("diagnostic-output"))
    monkeypatch.setattr(
        driver,
        "finalize_diagnostic_outputs",
        lambda *args, **kwargs: finalized.append((args, kwargs)),
    )
    monkeypatch.setattr(
        driver,
        "_decode_generate_inputs",
        lambda **kwargs: generated.append(kwargs) or fixture,
    )
    monkeypatch.setattr(driver, "_decode_golden", lambda value, **kwargs: golden_calls.append((value, kwargs)))
    monkeypatch.setattr(
        driver,
        "_copy_and_compare",
        lambda worker, buffers, value: compared.append((worker, buffers, value)),
    )
    monkeypatch.setattr(
        driver,
        "param_tensors",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("correctness mode must not generate a second fixture")),
    )

    assert driver.run([7], platform, enable_chip_swimlane=3, enable_swimlane_overhead=True) == 0

    [worker] = _FakeWorker.instances
    assert generated == [{"seed": 1234, "seq_len": 3500, "n_layers": driver.N_LAYERS}]
    assert golden_calls == [(fixture, {"n_layers": driver.N_LAYERS})]
    assert len(compared) == 1 and compared[0][0] is worker and compared[0][2] is fixture
    assert [name for _, name in worker.uploads] == [name for name in names if name != "out"]
    assert worker.runs[0][2].enable_chip_swimlane == 3
    assert worker.runs[0][2].output_prefix == "diagnostic-output"
    assert worker.events.index("autoload") > max(i for i, event in enumerate(worker.events) if event == "copy_to")
    assert worker.events.index("autoload") < worker.events.index("run")
    assert len(finalized) == 1
    assert finalized[0][1]["chip_swimlane"] == 3
    assert finalized[0][1]["swimlane_overhead"] is True


@pytest.mark.parametrize(
    ("relative_path", "platform"),
    [
        ("examples/a2a3/tensormap_and_ringbuffer/qwen3_14b_decode/main.py", "a2a3"),
        ("examples/a5/tensormap_and_ringbuffer/qwen3_14b_decode/main.py", "a5"),
    ],
)
def test_standalone_main_configures_cli_log_level(monkeypatch, relative_path: str, platform: str) -> None:
    driver = _load_driver(relative_path)
    configured = []
    runs = []

    monkeypatch.setattr(driver, "configure_logging", configured.append)
    monkeypatch.setattr(driver, "run", lambda *args, **kwargs: runs.append((args, kwargs)) or 0)

    assert driver.main(["-p", platform, "--log-level", "info", "--compile-only"]) == 0
    assert configured == ["info"]
    assert len(runs) == 1
    assert runs[0][1]["compile_only"] is True
