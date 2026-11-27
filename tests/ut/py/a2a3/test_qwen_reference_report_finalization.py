# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CPU-only failure injection through the reference consumer's real entry point."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

CASE = Path(__file__).resolve().parents[4] / "examples/a2a3/host_build_graph/qwen3_14b_serving_effective"


@pytest.fixture
def consumer(monkeypatch, tmp_path):
    worker = Mock()
    handle = Mock(_run_id=7)
    worker.submit.return_value = handle
    fixture = Mock(steps=1, num_pages=1, golden={"logits": torch.ones(1, 4)})
    step = SimpleNamespace(
        seq_lens=torch.tensor([1], dtype=torch.int32),
        slot_mapping=torch.tensor([0], dtype=torch.int32),
        block_table=torch.tensor([[0]], dtype=torch.int32),
        input_token_ids=torch.tensor([0], dtype=torch.int32),
        expected_output_token_ids=torch.tensor([0], dtype=torch.int32),
    )
    adapter = Mock(completed_steps=1)
    adapter.next_step.return_value = step
    imports = {
        "callable_bridge": {"inspect_artifact": Mock(return_value=Mock(source_hashes={}))},
        "real_worker_submit": {
            "BATCH": 1,
            "HEAD_DIM": 1,
            "HEADS": 1,
            "HIDDEN": 1,
            "LAYERS": 1,
            "PADDED_VOCAB": 4,
            "PAGE": 1,
            "_bind": Mock(),
            "_host_buffer": Mock(),
            "_view": lambda buffer, shape, dtype: buffer,
        },
        "reference_fixture": {"ReferenceFixture": Mock(return_value=fixture)},
        "safetensors.torch": {"load_file": Mock(), "save_file": Mock()},
        "simpler.task_interface": {
            "CallConfig": SimpleNamespace,
            "TensorArgType": SimpleNamespace(INPUT=0, OUTPUT_EXISTING=1),
        },
        "simpler.worker": {"Worker": Mock(return_value=worker)},
        "standalone_adapter": {"StandaloneDecodeAdapter": Mock(return_value=adapter)},
        "weights": {"iter_kernel_weights": Mock(), "rope_tables": Mock()},
        "simpler_setup.torch_interop": {"torch_dtype_to_datatype": Mock()},
    }
    for name, attributes in imports.items():
        monkeypatch.setitem(sys.modules, name, SimpleNamespace(**attributes))
    spec = importlib.util.spec_from_file_location("qwen_reference_report_consumer", CASE / "reference_worker_submit.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / "output"
    args = SimpleNamespace(
        fixture=tmp_path,
        artifact=tmp_path,
        model=tmp_path,
        output=output,
        device=0,
        steps=1,
        kv_reference=None,
        depth2_probe=False,
    )
    (tmp_path / "SHA256SUMS").write_text("fixture identity")
    (tmp_path / "distributed_meta.json").write_text('{"params": []}')
    runtime = tmp_path / "build/lib/a2a3/onboard/host_build_graph"
    runtime.mkdir(parents=True)
    (runtime / "runtime.so").write_bytes(b"runtime identity")
    monkeypatch.setattr(module, "__file__", str(tmp_path / "examples/a2a3/host_build_graph/case/consumer.py"))
    monkeypatch.setattr(module.argparse.ArgumentParser, "parse_args", lambda self: args)
    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **k: "test-head")
    monkeypatch.setattr(module.torch, "set_num_threads", lambda count: None)

    def allocate(_worker, devices, hosts, _fixture, _model):
        for name, shape, dtype in (
            ("seq_lens", (1,), torch.int32),
            ("slot_mapping", (1,), torch.int32),
            ("block_table", (1,), torch.int32),
            ("sampled_ids_in", (1, 8), torch.int32),
            ("sampled_ids", (1, 8), torch.int32),
            ("out", (1, 4), torch.float32),
        ):
            hosts[name] = torch.zeros(shape, dtype=dtype)
            devices[name] = name

    monkeypatch.setattr(module, "allocate_resident", allocate)

    def copy_from(buffer, device):
        buffer.fill_(1 if device == "out" else 0)

    worker.copy_from.side_effect = copy_from
    return SimpleNamespace(module=module, worker=worker, handle=handle, fixture=fixture, output=output)


def _contexts(error):
    errors = []
    while error is not None:
        assert error not in errors, "exception context must not contain a cycle"
        errors.append(error)
        error = error.__context__
    return errors


@pytest.mark.parametrize(
    "failure_stage,close_fails,write_fails",
    [
        (None, False, False),
        ("register", False, False),
        ("init", False, False),
        ("result", False, False),
        (None, True, False),
        ("result", True, False),
        (None, False, True),
        ("result", True, True),
    ],
    ids=["success", "register", "init", "execution", "close", "execution-close", "write", "all-fail"],
)
def test_reference_report_finalizes_after_cleanup(consumer, monkeypatch, failure_stage, close_fails, write_fails):
    execution_error = RuntimeError("execution failed") if failure_stage else None
    underlying_error = ValueError("underlying execution failure")
    if execution_error:
        execution_error.__context__ = underlying_error
    close_error = RuntimeError("cleanup failed") if close_fails else None
    write_errors = []
    if failure_stage:
        target = consumer.handle if failure_stage == "result" else consumer.worker
        getattr(target, failure_stage).side_effect = execution_error
    consumer.worker.close.side_effect = close_error
    write_text = Path.write_text
    reports = []

    def write_report(path, text, *args, **kwargs):
        if path.name == "result.json":
            consumer.worker.close.assert_called_once_with()
            reports.append(json.loads(text))
            if write_fails:
                error = OSError("report storage failed")
                write_errors.append(error)
                raise error
        return write_text(path, text, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", write_report)
    if execution_error or close_error or write_fails:
        with pytest.raises(BaseException) as caught:
            consumer.module.cli()
        assert caught.value is (execution_error or close_error or write_errors[0])
        chain = _contexts(caught.value)
        if execution_error:
            assert underlying_error in chain
        if close_error:
            assert close_error in chain
        assert all(error in chain for error in write_errors)
    else:
        assert consumer.module.cli() == 0
    consumer.worker.close.assert_called_once_with()
    if write_fails:
        assert not (consumer.output / "result.json").exists()
    else:
        report = json.loads((consumer.output / "result.json").read_text())
        assert report == reports[0]
        assert len(reports) == 1
        assert report["status"] == ("failed" if execution_error or close_error else "passed")
        if execution_error or close_error:
            assert report["error"] == f"RuntimeError: {execution_error or close_error}"
        if close_error:
            assert report["cleanup_error"] == "RuntimeError: cleanup failed"
        if failure_stage is None:
            assert report["completed_steps"] == 1
            assert len(report["steps"]) == 1


@pytest.mark.parametrize("report_operation", ["exists", "write_text"])
def test_reference_setup_failure_keeps_original_error_when_reporting_fails(consumer, monkeypatch, report_operation):
    error = ValueError("invalid fixture")
    consumer.fixture.verify_model.side_effect = error
    write_errors = []

    def fail_write(*args, **kwargs):
        write_error = OSError("report storage failed")
        write_errors.append(write_error)
        raise write_error

    monkeypatch.setattr(Path, report_operation, fail_write)
    with pytest.raises(ValueError) as caught:
        consumer.module.cli()
    assert caught.value is error
    assert _contexts(error) == [error, *write_errors]
    consumer.module.Worker.assert_not_called()


def test_reference_successful_system_exit_does_not_write_failure(consumer, monkeypatch):
    def exit_successfully():
        consumer.output.mkdir()
        consumer.module._LAST_OUTPUT["path"] = consumer.output
        raise SystemExit(0)

    monkeypatch.setattr(consumer.module, "main", exit_successfully)
    with pytest.raises(SystemExit) as caught:
        consumer.module.cli()
    assert caught.value.code == 0
    assert not (consumer.output / "result.json").exists()
