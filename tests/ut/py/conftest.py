# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Pytest configuration for Python unit tests (tests/ut/py/).

Adds project directories to sys.path so that simpler_setup, task_interface,
and host_worker modules are importable without installing the package, and adds
this directory itself so test-only helper modules (e.g. ``_cpp_constant_parser``)
are importable by bare name under ``--import-mode=importlib``.
"""

import sys
from multiprocessing import resource_tracker
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path

import pytest

from simpler_setup.platform_info import ARCHS, PLATFORM_MAP

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
for _d in [_ROOT, _ROOT / "python", Path(__file__).resolve().parent]:
    _s = str(_d)
    if _s not in sys.path:
        sys.path.insert(0, _s)


_PYTHON_UT = Path(__file__).resolve().parent
_ONBOARD = {platform for platform, (_, target) in PLATFORM_MAP.items() if target == "onboard"}


def _case_arches(item, directories):
    if directories and directories[0] in ARCHS:
        return {directories[0]}
    declaration = item.get_closest_marker("ut_arches")
    if declaration:
        return set(declaration.args)
    # A shared test can use one platform per parameter row. Unknown values
    # (for example invalid-input tests) have no architectural restriction.
    params = getattr(getattr(item, "callspec", None), "params", {})
    arches = set()
    for name in ("arch", "platform"):
        value = params.get(name)
        if not isinstance(value, str):
            continue
        if value in PLATFORM_MAP:
            arches.add(PLATFORM_MAP[value][0])
        elif value in ARCHS:
            arches.add(value)
    if arches:
        return arches
    platforms = item.get_closest_marker("platforms")
    if platforms:
        return {PLATFORM_MAP[p][0] for p in platforms.args[0] if p in PLATFORM_MAP}
    return set()


def pytest_itemcollected(item):
    if _PYTHON_UT not in item.path.parents:
        return
    dirs = item.path.relative_to(_PYTHON_UT).parts[:-1]
    owner = dirs[0] if dirs and dirs[0] in ARCHS else None
    declared = item.get_closest_marker("ut_arches")
    if declared and (not declared.args or declared.kwargs or any(arch not in ARCHS for arch in declared.args)):
        raise pytest.UsageError(f"{item.nodeid}: ut_arches takes architecture names from {ARCHS}")
    if owner and declared and set(declared.args) != {owner}:
        raise pytest.UsageError(f"{item.nodeid}: ut_arches disagrees with its {owner} directory")

    resources = [r for r in ("hardware", "toolkit") if item.get_closest_marker(f"requires_{r}")]
    if not resources:
        return
    if len(resources) != 1:
        raise pytest.UsageError(f"{item.nodeid}: select one execution resource, hardware or toolkit")
    resource = resources[0]
    if {r for r in dirs if r in ("hardware", "toolkit", "nohardware")} != {resource}:
        raise pytest.UsageError(f"{item.nodeid}: requires_{resource} tests belong in a {resource}/ directory")
    platforms = item.get_closest_marker("platforms")
    names = platforms.args[0] if platforms and len(platforms.args) == 1 else ()
    supported = set(names) if isinstance(names, (list, tuple)) and all(isinstance(p, str) for p in names) else set()
    if not supported or not supported.issubset(_ONBOARD):
        raise pytest.UsageError(f"{item.nodeid}: requires_{resource} needs an explicit list of onboard platforms")
    if owner and supported != {owner}:
        raise pytest.UsageError(f"{item.nodeid}: onboard platforms disagree with its {owner} directory")
    params = getattr(getattr(item, "callspec", None), "params", {})
    for field in ("arch", "platform"):
        value = params.get(field)
        if isinstance(value, str) and (value in ARCHS or value in PLATFORM_MAP) and value not in supported:
            raise pytest.UsageError(f"{item.nodeid}: parameter {field}={value} disagrees with its onboard platforms")


def pytest_collection_modifyitems(config, items):
    selected = config.getoption("--ut-arches")
    if selected is None:
        return
    keep, deselected = [], []
    for item in items:
        if _PYTHON_UT in item.path.parents:
            arches = _case_arches(item, item.path.relative_to(_PYTHON_UT).parts[:-1])
            if arches and arches.isdisjoint(selected):
                deselected.append(item)
                continue
        keep.append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = keep


@pytest.fixture
def native_host_memory():
    """Emulate native POSIX ownership without Python resource-tracker ownership."""
    posixshmem = pytest.importorskip("_posixshmem")

    class NativeOwner(SharedMemory):
        def __init__(self, *, size, name=None):
            super().__init__(name=name, create=True, size=size)
            resource_tracker.unregister(self._name, "shared_memory")

        def unlink(self):
            posixshmem.shm_unlink(self._name)

    owners = []

    def create(*, size=4096, name=None):
        owner = NativeOwner(size=size, name=name)
        owners.append(owner)
        return owner

    yield create
    for owner in owners:
        owner.close()
        try:
            owner.unlink()
        except FileNotFoundError:
            pass
