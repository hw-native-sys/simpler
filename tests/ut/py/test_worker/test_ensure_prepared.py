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

from unittest.mock import MagicMock

import pytest
from simpler import worker as worker_module
from simpler.task_interface import ChipCallable, CoreCallable
from simpler.worker import _ensure_prepared


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


def test_chip_startup_uploads_every_callable(monkeypatch):
    child = CoreCallable.build(signature=[], binary=bytes.fromhex("00f06140"))
    target = ChipCallable.build(signature=[], func_name="entry", binary=b"host", children=[(2, child)])

    cw = MagicMock()
    cw.pipeline_depth = 1
    monkeypatch.setattr(worker_module, "_ChipWorker", lambda: cw)
    monkeypatch.setattr(worker_module, "_initialize_chip_runtime", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker_module, "_run_chip_main_loop", lambda *args, **kwargs: None)
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
    )
    uploads = [call.args for call in cw.register_callable.call_args_list]
    assert [cid for cid, _ in uploads] == [0, 1, 2]
    assert all(upload is target for _, upload in uploads)
