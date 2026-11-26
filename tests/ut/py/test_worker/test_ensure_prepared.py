# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Unit tests for the module-level ``_ensure_prepared`` helper.

``_ensure_prepared`` is the ``_CTRL_PREPARE`` registration path: it stages a
callable's slot exactly once. There is no lazy/run-time preparation — the run
path (``_TASK_READY``) only consumes an already-prepared slot and raises if it
is missing, so that behavior is covered by the chip-loop tests, not here.
"""

import ctypes
from unittest.mock import MagicMock

import pytest
from simpler import worker as worker_module
from simpler.callable_identity import build_chip_callable_descriptor, compute_callable_hashid
from simpler.task_interface import ChipCallable, CoreCallable
from simpler.worker import Worker, _ensure_prepared

from simpler_setup import a5_input_cache


class TestEnsurePrepared:
    def test_prepares_slot(self, capsys):
        cw = MagicMock()
        callable_obj = object()
        registry = {2: callable_obj}
        prepared: set[int] = set()

        _ensure_prepared(cw, registry, prepared, 2, device_id=0)

        cw.register_callable.assert_called_once_with(2, callable_obj)
        assert prepared == {2}
        # Preparation is a normal control-flow step now — no warning is emitted.
        assert capsys.readouterr().err == ""

    def test_already_prepared_short_circuits(self):
        cw = MagicMock()
        # cid is already in `prepared`; helper must skip lookup and
        # slot preparation entirely.  Pass an empty registry to prove the
        # lookup never happens (otherwise registry.get would return None
        # and the helper would raise).
        _ensure_prepared(cw, {}, {5}, 5, device_id=0)
        cw.register_callable.assert_not_called()

    def test_missing_cid_raises(self):
        with pytest.raises(RuntimeError, match="cid 9 not in registry"):
            _ensure_prepared(MagicMock(), {}, set(), 9, device_id=0)


def _raw_chip_callable():
    code = bytes.fromhex("00f06140")  # A5 RET
    child = CoreCallable.build(signature=[], binary=code)
    return code, ChipCallable.build(signature=[], func_name="entry", binary=b"host", children=[(2, child), (4, child)])


def test_a5_worker_register_enriches_only_the_uploaded_copy(monkeypatch):
    code, target = _raw_chip_callable()
    # Both CANN decoders accept this four-byte kernel. Use its actual footer so this test
    # also runs on hosts without CANN and still exercises the native checksum and wire packing.
    footer = bytes.fromhex("4843503101000003e607843c7f782c79") + bytes(24)
    analyzer = MagicMock(return_value=code + footer)
    monkeypatch.setenv("ASCEND_HOME_PATH", "mock-toolchain")
    monkeypatch.setattr(a5_input_cache, "annotate_a5_hbg_text", analyzer)
    monkeypatch.setattr(worker_module, "_monotonic", lambda: 100.0)
    worker = Worker(level=2, platform="a5", runtime="host_build_graph")
    worker._lifecycle = worker_module._Lifecycle.READY
    chip_worker = MagicMock()
    worker._chip_worker = chip_worker
    original = ctypes.string_at(target.buffer_ptr(), target.buffer_size())
    descriptor = build_chip_callable_descriptor(target=target, platform="a5", runtime="host_build_graph")
    handle = worker.register(target)
    assert handle.hashid == compute_callable_hashid(descriptor)
    assert worker.register(target).hashid == handle.hashid
    chip_worker.register_callable.assert_called_once()
    analyzer.assert_called_once_with(code, deadline=100.25)
    upload = chip_worker.register_callable.call_args.args[1]
    assert upload is not target
    assert ctypes.string_at(target.buffer_ptr(), target.buffer_size()) == original
    expected_child = CoreCallable.build(signature=[], binary=code + footer)
    expected = ctypes.string_at(expected_child.buffer_ptr(), expected_child.buffer_size())
    for index in range(2):
        child = upload.child(index)
        assert ctypes.string_at(child.buffer_ptr(), child.buffer_size()) == expected


@pytest.mark.parametrize(
    "platform,runtime", [("a5", "host_build_graph"), ("a5sim", "host_build_graph"), ("a5", "tensormap_and_ringbuffer")]
)
def test_chip_registration_missing_decoder_or_unsupported_target_keeps_original(
    monkeypatch, tmp_path, platform, runtime
):
    _, target = _raw_chip_callable()
    monkeypatch.setenv("ASCEND_HOME_PATH", str(tmp_path))
    cw = MagicMock()
    prepared: set[int] = set()
    _ensure_prepared(cw, {2: target}, prepared, 2, device_id=0, platform=platform, runtime=runtime)
    _ensure_prepared(cw, {2: target}, prepared, 2, device_id=0, platform=platform, runtime=runtime)
    cw.register_callable.assert_called_once_with(2, target)


def test_each_upload_observes_the_current_decoder(monkeypatch, tmp_path):
    _, target = _raw_chip_callable()
    monkeypatch.setenv("ASCEND_HOME_PATH", str(tmp_path))
    tool = tmp_path / "bin/llvm-objdump"
    tool.parent.mkdir()
    deadline = worker_module._monotonic() + 5.0
    assert (
        worker_module._chip_callable_for_upload(target, "a5", "host_build_graph", analysis_deadline=deadline) is target
    )
    tool.write_text("#!/bin/sh\nprintf '%s\\n' '0: 4061f000 RET'\n")
    tool.chmod(0o755)
    assert (
        worker_module._chip_callable_for_upload(target, "a5", "host_build_graph", analysis_deadline=deadline)
        is not target
    )
    tool.write_text("#!/bin/sh\nexit 1\n")
    assert (
        worker_module._chip_callable_for_upload(target, "a5", "host_build_graph", analysis_deadline=deadline) is target
    )


def test_late_prepare_falls_back_when_its_short_analysis_budget_is_spent(monkeypatch):
    code, target = _raw_chip_callable()
    footer = bytes.fromhex("4843503101000003e607843c7f782c79") + bytes(24)
    clock = [100.0]

    def slow_proof(text, *, deadline):
        assert text == code and deadline == 100.25
        clock[0] = deadline
        return text + footer

    monkeypatch.setattr(worker_module, "_monotonic", lambda: clock[0])
    monkeypatch.setattr(a5_input_cache, "annotate_a5_hbg_text", slow_proof)
    monkeypatch.setenv("ASCEND_HOME_PATH", "mock-toolchain")
    cw = MagicMock()
    prepared = set()
    _ensure_prepared(cw, {1: target}, prepared, 1, device_id=0, platform="a5", runtime="host_build_graph")
    cw.register_callable.assert_called_once_with(1, target)
    assert prepared == {1}


def test_chip_startup_shares_analysis_budget_and_uploads_every_callable(monkeypatch):
    _, target = _raw_chip_callable()
    footer = bytes.fromhex("4843503101000003e607843c7f782c79") + bytes(24)
    clock = [100.0]
    deadlines = []

    def analyze(text, *, deadline):
        deadlines.append(deadline)
        clock[0] += 1.0 if len(deadlines) == 1 else 4.0
        return text + footer

    cw = MagicMock()
    cw.pipeline_depth = 1
    monkeypatch.setattr(worker_module, "_ChipWorker", lambda: cw)
    monkeypatch.setattr(worker_module, "_initialize_chip_runtime", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker_module, "_run_chip_main_loop", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker_module, "_monotonic", lambda: clock[0])
    monkeypatch.setattr(a5_input_cache, "annotate_a5_hbg_text", analyze)
    monkeypatch.setenv("ASCEND_HOME_PATH", "mock-toolchain")
    buf = memoryview(bytearray(worker_module.mailbox_size(1)))
    worker_module._chip_process_loop(
        buf,
        None,
        0,
        dict.fromkeys((0, 1, 2), target),
        {},
        {},
        worker_module.mint_owner_instance_id(),
        worker_module.mint_owner_instance_id(),
        worker_module.mint_owner_instance_id(),
        platform="a5",
        runtime="host_build_graph",
        startup_deadline=110.0,
    )
    assert deadlines == [105.0, 105.0]
    uploads = [call.args for call in cw.register_callable.call_args_list]
    assert [cid for cid, _ in uploads] == [0, 1, 2]
    assert uploads[0][1] is not target
    assert uploads[1][1] is target and uploads[2][1] is target
