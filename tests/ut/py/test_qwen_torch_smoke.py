# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Completion-smoke receipts preserve execution and cleanup failures."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

CASE = Path(__file__).resolve().parents[3] / "examples/a2a3/host_build_graph/qwen3_14b_serving_effective"


@pytest.fixture
def smoke(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(CASE))
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    spec = importlib.util.spec_from_file_location("qwen_torch_smoke_test", CASE / "run_torch_eager.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    state = SimpleNamespace(primary=None, terminal=None, cleanup=None, bad_shape=False, backend=None)
    stream = SimpleNamespace(npu_stream=123, synchronize=lambda: None)

    def cpu(function):
        return lambda *args, **kwargs: function(*args, **{**kwargs, "device": "cpu"})

    module.torch = SimpleNamespace(
        device=lambda _: torch.device("cpu"),
        npu=SimpleNamespace(set_device=lambda _: None, current_stream=lambda _: stream),
        tensor=cpu(torch.tensor),
        arange=cpu(torch.arange),
        ones=cpu(torch.ones),
        int32=torch.int32,
        int64=torch.int64,
        float32=torch.float32,
        isfinite=torch.isfinite,
    )

    class Backend:
        def __init__(self, **kwargs):
            self.steps, self.closed, self.failed = 0, False, False
            self.cleanup_error = None
            self.values = {"out": torch.ones((1, 152064))}
            state.backend = self

        def __call__(self, *args):
            if state.primary:
                self.failed = True
                raise state.primary
            self.steps += 1
            return torch.ones((1, 7)) if state.bad_shape else self.values["out"]

        def close(self):
            if state.cleanup:
                self.cleanup_error = state.cleanup
                raise state.terminal if state.terminal is not None else state.cleanup
            self.closed = True

    monkeypatch.setattr(module, "TorchDecode", Backend)
    monkeypatch.setattr(module, "inspect_artifact", lambda _: SimpleNamespace(source_hashes={"source": "digest"}))
    monkeypatch.setattr(module, "validate_checkpoint", lambda _: {"config.json": "digest"})
    monkeypatch.setattr(module, "validate_prompt_ids", lambda _: None)
    monkeypatch.setattr(module, "iter_kernel_weights", lambda _: iter((("weight", torch.ones(1)),)))
    monkeypatch.setattr(module, "rope_tables", lambda _: (torch.ones(3466, 128), torch.ones(3466, 128)))
    monkeypatch.setattr(module, "allocate_layer_caches", lambda **kwargs: (torch.ones((2, 448, 128, 1, 1)),) * 2)
    monkeypatch.setattr(module.importlib.metadata, "version", lambda _: "test")
    prompt = tmp_path / "prompt.json"
    prompt.write_text("[1, 2]")
    output = tmp_path / "receipt.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_torch_eager.py",
            "--artifact",
            str(tmp_path),
            "--model",
            str(tmp_path),
            "--prompt-ids",
            str(prompt),
            "--device",
            "0",
            "--output",
            str(output),
        ],
    )
    return SimpleNamespace(module=module, state=state, output=output)


@pytest.mark.parametrize(
    "execution_fails,close_fails,replay_terminal",
    [(False, False, False), (True, False, False), (False, True, False), (True, True, False), (True, True, True)],
)
def test_smoke_always_writes_receipt_and_preserves_primary(smoke, execution_fails, close_fails, replay_terminal):
    primary, cleanup = RuntimeError("execution failed"), RuntimeError("cleanup failed")
    smoke.state.primary = primary if execution_fails else None
    smoke.state.cleanup = cleanup if close_fails else None
    smoke.state.terminal = primary if replay_terminal else None
    if execution_fails or close_fails:
        with pytest.raises(RuntimeError) as caught:
            smoke.module.main()
        assert caught.value is (primary if execution_fails else cleanup)
    else:
        assert smoke.module.main() == 0
    receipt = json.loads(smoke.output.read_text())
    assert receipt["status"] == ("failed" if execution_fails or close_fails else "passed")
    assert receipt["model_sha256"] == {"config.json": "digest"}
    assert receipt["closed"] is not close_fails
    if execution_fails:
        assert receipt["error"] == "RuntimeError: execution failed"
        assert receipt["position"] == 3338 and receipt["step"] == 0
    if close_fails:
        assert receipt["cleanup_error"] == "RuntimeError: cleanup failed"


def test_model_failure_is_recorded_before_device_work(smoke, monkeypatch):
    def fail(_):
        raise ValueError("model checksum mismatch")

    monkeypatch.setattr(smoke.module, "validate_checkpoint", fail)
    with pytest.raises(ValueError, match="checksum"):
        smoke.module.main()
    assert smoke.state.backend is None
    assert json.loads(smoke.output.read_text())["error"] == "ValueError: model checksum mismatch"


def test_close_preserves_terminal_error_and_records_cleanup_error(smoke, monkeypatch):
    primary = RuntimeError("terminal execution failed")
    smoke.state.cleanup = OSError("worker cleanup failed")

    def finish_steps(*args):
        smoke.state.terminal = primary

    monkeypatch.setattr(smoke.module, "_run_steps", finish_steps)
    with pytest.raises(RuntimeError) as caught:
        smoke.module.main()
    assert caught.value is primary
    receipt = json.loads(smoke.output.read_text())
    assert receipt["error"] == "RuntimeError: terminal execution failed"
    assert receipt["cleanup_error"] == "OSError: worker cleanup failed"
    assert receipt["status"] == "failed" and not receipt["closed"]


def test_smoke_gate_has_shape_and_position_context(smoke):
    smoke.state.bad_shape = True
    with pytest.raises(RuntimeError, match="step=0.*position=3338.*shape"):
        smoke.module.main()
    assert json.loads(smoke.output.read_text())["status"] == "failed"


def test_empty_weight_conversion_fails_before_backend_creation(smoke, monkeypatch):
    monkeypatch.setattr(smoke.module, "iter_kernel_weights", lambda _: iter(()))
    with pytest.raises(RuntimeError, match="no weights"):
        smoke.module.main()
    assert smoke.state.backend is None
    assert json.loads(smoke.output.read_text())["status"] == "failed"


def test_receipt_failure_does_not_replace_execution_error(smoke, monkeypatch):
    error = RuntimeError("execution failed")
    smoke.state.primary = error

    def fail_write(*args, **kwargs):
        raise OSError("receipt storage failed")

    monkeypatch.setattr(Path, "write_text", fail_write)
    with pytest.raises(RuntimeError) as caught:
        smoke.module.main()
    assert caught.value is error and smoke.state.backend.closed


def test_explicit_smoke_gate_remains_active_under_python_optimized():
    code = (
        "import sys, types; "
        f"sys.path.insert(0, {str(CASE)!r}); "
        "sys.modules['torch_npu'] = types.SimpleNamespace(); "
        "from run_torch_eager import _check; _check(False, 'optimized smoke gate')"
    )
    result = subprocess.run(
        [sys.executable, "-O", "-c", code],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "TORCH_DEVICE_BACKEND_AUTOLOAD": "0"},
    )
    assert result.returncode != 0 and "RuntimeError: optimized smoke gate" in result.stderr
