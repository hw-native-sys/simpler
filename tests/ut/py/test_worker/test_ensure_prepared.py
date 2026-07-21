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
from simpler.worker import _ensure_prepared


class TestEnsurePrepared:
    def test_prepares_slot(self, capsys):
        cw = MagicMock()
        callable_obj = object()
        registry = {2: callable_obj}
        prepared: set[int] = set()

        _ensure_prepared(cw, registry, prepared, 2, device_id=0)

        cw._register_callable_at_slot.assert_called_once_with(2, callable_obj)
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
        cw._register_callable_at_slot.assert_not_called()

    def test_missing_cid_raises(self):
        with pytest.raises(RuntimeError, match="cid 9 not in registry"):
            _ensure_prepared(MagicMock(), {}, set(), 9, device_id=0)
