# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Execute each example's actual finally block without requiring kernel compilation."""

import ast
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("example", ["run.py", "run_mx.py"])
@pytest.mark.parametrize("close_fails", [False, True])
def test_example_cleanup_always_closes_and_preserves_original_error(example, close_fails):
    root = Path(__file__).resolve().parents[3]
    path = root / "examples/a5/tensormap_and_ringbuffer/host_ddr_gather" / example
    module = ast.parse(path.read_text())
    run = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "run")
    guarded = next(node for node in run.body if isinstance(node, ast.Try) and node.finalbody)
    original = KeyboardInterrupt("interrupted after submit")
    cleanup_error = RuntimeError("completion unproven or unregister refused")
    worker = Mock()
    worker.release_buffer.side_effect = cleanup_error
    worker.free.side_effect = cleanup_error
    if close_fails:
        worker.close.side_effect = cleanup_error
    # Keep the source finally block intact: only the hardware-dependent body is replaced.
    guarded = ast.Try(
        body=[ast.Raise(exc=ast.Name(id="original", ctx=ast.Load()))],
        handlers=[],
        orelse=[],
        finalbody=guarded.finalbody,
    )
    code = compile(ast.fix_missing_locations(ast.Module(body=[guarded], type_ignores=[])), str(path), "exec")
    values = dict(worker=worker, tables=[object()], table=object(), handles=[object()], original=original, sys=sys)
    with pytest.raises(KeyboardInterrupt) as caught:
        exec(code, values)
    assert caught.value is original
    worker.close.assert_called_once_with()
    if close_fails:
        assert caught.value.__cause__ is cleanup_error


@pytest.mark.parametrize("example", ["run.py", "run_mx.py"])
def test_example_init_failure_still_closes(example, monkeypatch):
    import runpy  # noqa: PLC0415

    root = Path(__file__).resolve().parents[3]
    path = root / "examples/a5/tensormap_and_ringbuffer/host_ddr_gather" / example
    run = runpy.run_path(str(path))["run"]
    worker = Mock()
    original = RuntimeError("initialization refused")
    worker.init.side_effect = original
    monkeypatch.setitem(run.__globals__, "Worker", Mock(return_value=worker))
    monkeypatch.setitem(run.__globals__, "build_callable", Mock())
    monkeypatch.setitem(run.__globals__, "compile_chip_callable_spec", Mock())
    with pytest.raises(RuntimeError) as caught:
        run()
    assert caught.value is original
    worker.close.assert_called_once_with()
