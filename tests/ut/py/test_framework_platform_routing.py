# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CPU routing barriers; compiler/framework/device execution is not qualified here."""

import importlib.util
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_EXAMPLES = _ROOT / "examples/workers/l2/framework_eager"
_COMPILED = (
    "main",
    "alias_inout",
    "value_arguments",
    "persistent_state",
    "scope_statistics",
    "argument_capture",
    "lifecycle",
)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _shared_consumers():
    consumers = set()
    for path in (_ROOT / "tests/st").glob("**/framework_eager/test_*.py"):
        module = _load(path, "platform_scene_metadata")
        marks = getattr(module, "pytestmark", [])
        if not any(mark.name == "platforms" and "a5" in mark.args[0] for mark in marks):
            continue
        example = getattr(module, "_EXAMPLE", None)
        assert isinstance(example, Path) and example.is_file(), (path, example)
        assert example.parent == _EXAMPLES, (path, example)
        consumers.add(example.stem)
    assert consumers, "No shared A3/A5 framework consumers declared by scene tests"
    return sorted(consumers)


_CONSUMERS = _shared_consumers()


@pytest.fixture
def consumer(request, monkeypatch):
    reference = _load(_EXAMPLES / "main.py", "platform_reference")
    monkeypatch.setitem(sys.modules, "main", reference)
    module = reference if request.param == "main" else _load(_EXAMPLES / f"{request.param}.py", "platform_consumer")
    for loader in ("_helpers", "_reference", "_bridge_helpers"):
        if hasattr(module, loader):
            monkeypatch.setattr(module, loader, lambda: reference)
    return module, reference


@pytest.mark.parametrize("consumer", _CONSUMERS, indirect=True)
@pytest.mark.parametrize("selected", [None, "a2a3", "a5"])
def test_real_consumer_cli_routes_selected_platform(consumer, selected, tmp_path, monkeypatch):
    module, reference = consumer
    bridge = tmp_path / "_simpler_framework_queue_cpu.so"
    bridge.touch()
    monkeypatch.setattr(reference, "build_bridge", lambda directory: bridge)
    needs_output = "root" in inspect.signature(module.run).parameters
    calls = []
    monkeypatch.setattr(module, "run", lambda *args, **kwargs: calls.append((args, kwargs)))
    argv = [str(module.__file__), "--runtime", "host_build_graph", "--build-dir", str(tmp_path)]
    if needs_output:
        argv += ["--output-dir", str(tmp_path / "exports")]
    if selected is not None:
        argv += ["--platform", selected]
    monkeypatch.setattr(sys, "argv", argv)
    module.main()
    assert len(calls) == 1 and calls[0][1]["platform"] == (selected or "a2a3")
    assert calls[0][0][:2] == ("host_build_graph", 0)


@pytest.mark.parametrize("consumer", _COMPILED, indirect=True)
def test_real_compile_boundary_separates_platform_cache_entries(consumer, monkeypatch):
    import importlib  # noqa: PLC0415

    scene_test = importlib.import_module("simpler_setup.scene_test")
    module, _ = consumer
    cache, calls = {}, []

    def compile_boundary(spec, platform, runtime, key):
        calls.append((platform, runtime, key))
        return cache.setdefault(key, object())

    monkeypatch.setattr(scene_test, "compile_chip_callable_spec", compile_boundary)
    a3 = module._compile("host_build_graph", platform="a2a3")
    a5 = module._compile("host_build_graph", platform="a5")
    a3_again = module._compile("host_build_graph", platform="a2a3")
    assert [(platform, runtime) for platform, runtime, _ in calls] == [
        ("a2a3", "host_build_graph"),
        ("a5", "host_build_graph"),
        ("a2a3", "host_build_graph"),
    ]
    assert a3 is a3_again and a3 is not a5
    assert calls[0][2] == calls[2][2] and calls[0][2] != calls[1][2]


@pytest.mark.parametrize("consumer", _CONSUMERS, indirect=True)
@pytest.mark.parametrize("platform", ["a2a3", "a5"])
def test_real_run_constructs_worker_on_selected_platform(consumer, platform, tmp_path, monkeypatch):
    module, reference = consumer
    streams = []

    def new_stream(**kwargs):
        stream = SimpleNamespace(npu_stream=len(streams) + 17)
        streams.append(stream)
        return stream

    torch = SimpleNamespace(npu=SimpleNamespace(set_device=lambda device: None, Stream=new_stream))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    monkeypatch.setattr(reference, "_load_bridge", lambda path: object())
    if hasattr(module, "_context_api"):
        # Context creation is a recording boundary in this Worker-routing probe;
        # the consumer's own contracts exercise restoration bookkeeping.
        monkeypatch.setattr(module, "_context_api", lambda: object())
        monkeypatch.setattr(module, "_CallerContexts", lambda *args: SimpleNamespace(create=lambda: None, base=None))
    observed = []

    class ConstructionStopped(Exception):
        pass

    def create_worker(**kwargs):
        observed.append(kwargs)
        raise ConstructionStopped

    monkeypatch.setitem(sys.modules, "simpler.worker", SimpleNamespace(Worker=create_worker))
    arguments = ["host_build_graph", 0, Path("recording.so")]
    if "root" in inspect.signature(module.run).parameters:
        arguments.append(tmp_path / "exports")
    with pytest.raises(ConstructionStopped):
        module.run(*arguments, platform=platform)
    assert observed == [dict(level=2, device_id=0, platform=platform, runtime="host_build_graph")]
