# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: PLC0415
"""Nothing requests workspace ownership any more, and the old spellings still work.

The onboard program init entry installs the manager for every caller, so the
host-side question is no longer *which* route asks — it is that no route has to,
and that the spellings which used to ask are still accepted.

Scope, exactly: these read the Python surface. They do not reach the install,
the same-handle refusals or the ledger — the first two are the C entry's and
are established by review of its call sites, and the ledger's own behaviour is
covered by the C++ WorkspaceManager cases. `test_teardown_proof.cpp` covers the
lifecycle rule the entry applies, and states its own limits.
"""

from __future__ import annotations

import inspect

_COMMON = {"platform": "a2a3", "runtime": "host_build_graph"}


def test_the_chip_child_entry_carries_no_management_request():
    """The internal per-topology flag is gone: a forked child's four regions are
    owned because the init entry owns them, not because the parent said so."""
    from simpler.worker import _chip_process_loop

    assert "manage_workspace" not in inspect.signature(_chip_process_loop).parameters


def test_no_worker_route_computes_a_management_predicate():
    """The routing predicate is removed outright rather than left as a caller
    of an ignored keyword."""
    from simpler.worker import Worker

    assert not hasattr(Worker, "_chip_children_manage_workspace")


def test_chip_worker_init_keeps_the_retired_keywords_name_position_and_default():
    """The Python signature a caller binds against: the parameter is still
    there, still after `workspace_budget_bytes`, and still defaults to False.

    This checks the signature only. Whether the native call still accepts the
    same positional list is the binding's, and is covered by the build.
    """
    from simpler.task_interface import ChipWorker

    parameters = list(inspect.signature(ChipWorker.init).parameters)
    assert "manage_workspace" in parameters
    assert inspect.signature(ChipWorker.init).parameters["manage_workspace"].default is False
    # Position matters as much as presence for the positional native call.
    assert parameters.index("manage_workspace") == parameters.index("workspace_budget_bytes") + 1


def test_the_level_two_route_no_longer_names_the_retired_keyword(monkeypatch):
    """`_init_level2` reaches `ChipWorker.init` without mentioning it at all.

    What a caller who *does* pass it gets is not asserted here: the value is
    ignored in `ChipWorker::init`, which this level cannot observe.
    """
    import simpler.worker as worker_mod

    import simpler_setup.runtime_builder as rb_mod

    seen: list[dict] = []

    class _RecordingChip:
        def init(self, *_a, **kwargs):
            seen.append(dict(kwargs))

        def _register_callable_at_slot(self, *_a, **_k):  # pragma: no cover
            pass

    class _FakeBuilder:
        def __init__(self, *_a, **_k):
            pass

        def get_binaries(self, *_a, **_k):
            return object()

    monkeypatch.setattr(worker_mod, "ChipWorker", _RecordingChip)
    monkeypatch.setattr(rb_mod, "RuntimeBuilder", _FakeBuilder)
    worker = worker_mod.Worker(2, device_id=0, **_COMMON)
    worker._init_level2()

    assert len(seen) == 1
    assert "manage_workspace" not in seen[0]
