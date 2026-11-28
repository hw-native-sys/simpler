# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Held admission carries local preparation order independently of global success."""

import ast
import inspect
from types import SimpleNamespace

import pytest
from simpler import worker as module


@pytest.mark.parametrize("stale", [False, True])
def test_held_grant_prepares_before_enqueue_without_authorizing_execution(stale):
    tree = ast.parse(inspect.getsource(module._run_chip_main_loop))
    helper = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "enqueue_held_if_authorized"
    )
    helper.body = [ast.Global(names=node.names) if isinstance(node, ast.Nonlocal) else node for node in helper.body]
    events = []
    staged = SimpleNamespace(
        native_activated=False,
        enqueued_held=False,
        prepare_ordered=False,
        entry_addr=100,
        entry_generation=9,
        entry_buf=b"owned-entry",
        identity=(1, 2, 3),
    )

    def enqueue():
        assert staged.prepare_ordered, "A held run must acquire preparation order before native enqueue"
        events.append("enqueue")
        return True

    staged.chip_run = SimpleNamespace(_authorize_prepare=lambda: events.append("prepare"), _enqueue_held=enqueue)
    context = {
        name: getattr(module, name)
        for name in (
            "_FRAME_STAGED",
            "_OFF_ENTRY_ENQUEUE_BEFORE_SUCCESS",
            "_OFF_ENTRY_PEER_CANCEL_REQUESTED",
            "_OFF_ENTRY_GENERATION",
            "_format_exc",
        )
    }
    context.update(
        shutdown_message="",
        device_id=0,
        _StagedRun=SimpleNamespace,
        _mailbox_load_i32=lambda address: int(address == 100 + module._OFF_ENTRY_ENQUEUE_BEFORE_SUCCESS),
        _mailbox_load_u64=lambda address: 8 if stale else 9,
        _read_run_entry_identity=lambda buffer: staged.identity,
        republish_native_preparation=lambda run: events.append("publish-prepared"),
    )
    code = compile(ast.fix_missing_locations(ast.Module(body=[helper], type_ignores=[])), module.__file__, "exec")
    exec(code, context)  # noqa: S102 -- execute the production helper with controlled mailbox observations
    invoke = context["enqueue_held_if_authorized"]
    assert invoke(staged, module._FRAME_STAGED) is not stale
    assert not staged.native_activated
    if stale:
        assert not events and not staged.prepare_ordered and not staged.enqueued_held
    else:
        assert events == ["prepare", "publish-prepared", "enqueue"]
        assert staged.enqueued_held
        assert invoke(staged, module._FRAME_STAGED)
        assert events == ["prepare", "publish-prepared", "enqueue"]
