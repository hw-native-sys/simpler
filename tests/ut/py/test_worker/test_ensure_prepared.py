# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Unit tests for the module-level ``_ensure_prepared`` helper.

The two chip-child loops both rely on this helper to handle the
``_CTRL_PREPARE`` prewarm path (``lazy=False``) and the ``_TASK_READY``
safety-net path (``lazy=True``). The lazy branch is not reachable from a
normal end-to-end run — it only fires when prewarm was skipped — so it
must be exercised directly here.
"""

from unittest.mock import MagicMock

import pytest
from simpler.worker import _ensure_prepared


class TestEnsurePrepared:
    def test_lazy_warns_and_prepares(self, capsys):
        cw = MagicMock()
        callable_obj = object()
        registry = {7: callable_obj}
        prepared: set[int] = set()

        _ensure_prepared(cw, registry, prepared, 7, lazy=True, device_id=3)

        cw.prepare_callable.assert_called_once_with(7, callable_obj)
        assert prepared == {7}
        err = capsys.readouterr().err
        assert "WARN: lazy-prepare cid=7" in err
        assert "dev=3" in err

    def test_eager_does_not_warn(self, capsys):
        cw = MagicMock()
        callable_obj = object()
        registry = {2: callable_obj}
        prepared: set[int] = set()

        _ensure_prepared(cw, registry, prepared, 2, lazy=False, device_id=0)

        cw.prepare_callable.assert_called_once_with(2, callable_obj)
        assert prepared == {2}
        assert capsys.readouterr().err == ""

    def test_already_prepared_short_circuits(self):
        cw = MagicMock()
        # cid is already in `prepared`; helper must skip lookup and
        # prepare_callable entirely.  Pass an empty registry to prove the
        # lookup never happens (otherwise registry.get would return None
        # and the helper would raise).
        _ensure_prepared(cw, {}, {5}, 5, lazy=True, device_id=0)
        cw.prepare_callable.assert_not_called()

    def test_missing_cid_raises(self):
        with pytest.raises(RuntimeError, match="cid 9 not in registry"):
            _ensure_prepared(MagicMock(), {}, set(), 9, lazy=False, device_id=0)
