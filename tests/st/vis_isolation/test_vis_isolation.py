#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Onboard ST: chip init works when ``ASCEND_RT_VISIBLE_DEVICES`` renumbers the device.

A launcher that isolates NPUs (task queue, container plugin, vLLM-ascend-style
worker-per-card) sets ``ASCEND_RT_VISIBLE_DEVICES`` and then hands simpler
logical ids starting at 0. ACL honors the variable for its own entry points, but
the chip-init path also makes **direct driver calls that bypass ACL** —
``halMemCtl`` / ``halGetDeviceInfo*`` on a2a3, ``halResMap`` /
``halGetDeviceInfo*`` / ``dsmi_get_device_info`` on a5 — and those index the
driver-visible space. Every one of them must translate through
``common/acl_hal_device.h::acl_to_hal_device_id``; a site that forgets targets
the wrong device and fails with ``halMemCtl rc=42`` in
``init_aicore_register_addresses``.

``init_aicore_register_addresses`` and ``probe_aicpu_topology`` run on every
onboard chip bring-up, so *any* scene test launched under the variable covers
all of those call sites. ``dummy_task`` is the cheapest one.

Allocating a comm domain crosses a second, independent device-id layer that chip
init never reaches, so it gets its own case: ``aclrtMemAccessDesc::location.id``
is consumed in the driver-visible space on a2a3 with CANN 9.0 even though
``aclrtMemSetAccess`` is an ACL entry point, and a logical id there fails with
507899. The A5 ACL runtime instead validates this field in the ACL-logical space.
Its sibling ``aclrtPhysicalMemProp::location.id`` is logical, so the two must
not be treated alike on the measured a2a3 stack — see
``common/acl_hal_device.h``.

The scene test runs in a subprocess: the variable has to be set before the
process performing ACL init starts, and mutating it in-process would leak into
the pooled workers the rest of the session shares.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

RUNTIME = "tensormap_and_ringbuffer"
_ST_ROOT = Path(__file__).resolve().parent.parent

# Platform -> the scene test the subprocess runs. Any onboard scene test would
# exercise the same chip-init call sites; these are the smallest per arch.
_DUMMY_TASK = {
    "a2a3": _ST_ROOT / "a2a3" / RUNTIME / "dummy_task" / "test_dummy_task.py",
    "a5": _ST_ROOT / "a5" / RUNTIME / "dummy_task" / "test_dummy_task.py",
}

# A comm domain maps device memory, which is a second device-id layer the chip-init path never
# reaches. On a2a3/CANN 9.0, `aclrtMemAccessDesc::location.id` is consumed in the driver-visible
# space, so a logical id there fails with 507899 on every card pair except an identity mapping.
# Both ranks take part, so this one runs on all the granted cards rather than a single logical id.
_COMM_DOMAIN = _ST_ROOT / "worker" / "comm_domain" / "async_notify" / "test_async_notify.py"

# The subprocess re-runs a full scene test (build cache hit + one chip bring-up).
_TIMEOUT_S = 600


def _remapped_visible(st_device_ids):
    """Return the granted cards as an ascending list, skipping if they map to themselves.

    The list must be ascending: CANN rejects any other order outright
    (`Runtime::GetVisibleDevices`, RT_ALL_ORDER_ERROR), which voids the whole mapping and makes even
    rtSetDevice fail with 107001. Position i is logical id i, so a card whose id differs from its
    position is what makes a translation observable — under an identity grant both id spaces hold
    the same number and the test would pass without proving anything.
    """
    visible = sorted(int(d) for d in st_device_ids)
    if all(card == i for i, card in enumerate(visible)):
        pytest.skip(f"granted cards {visible} map to themselves; no remap to verify")
    return visible


@pytest.mark.platforms(["a2a3", "a5"])
@pytest.mark.device_count(2)
@pytest.mark.runtime(RUNTIME)
def test_chip_init_under_visible_devices(st_platform, st_device_ids):
    """Two granted cards, addressed by a logical id that is not its own card id."""
    scene_test = _DUMMY_TASK[st_platform]
    assert scene_test.is_file(), f"missing scene test: {scene_test}"

    visible = _remapped_visible(st_device_ids)
    logical = next(i for i, card in enumerate(visible) if card != i)
    expected = visible[logical]

    env = {**os.environ, "ASCEND_RT_VISIBLE_DEVICES": ",".join(str(d) for d in visible)}
    proc = subprocess.run(
        [sys.executable, str(scene_test), "-p", st_platform, "-d", str(logical)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=_TIMEOUT_S,
        check=False,
    )
    if proc.returncode != 0:
        print(proc.stdout, file=sys.stderr)
    assert proc.returncode == 0, (
        f"dummy_task failed under ASCEND_RT_VISIBLE_DEVICES={env['ASCEND_RT_VISIBLE_DEVICES']} "
        f"(logical {logical} -> driver-visible {expected}), rc={proc.returncode}"
    )


@pytest.mark.platforms(["a2a3", "a5"])
@pytest.mark.device_count(2)
@pytest.mark.runtime(RUNTIME)
def test_comm_domain_under_visible_devices(st_platform, st_device_ids):
    """A comm domain allocates and maps its windows when the cards are renumbered."""
    assert _COMM_DOMAIN.is_file(), f"missing scene test: {_COMM_DOMAIN}"

    visible = _remapped_visible(st_device_ids)
    # Every rank participates, and under isolation the ranks are logical 0..N-1 whatever the
    # granted cards are.
    logical = ",".join(str(i) for i in range(len(visible)))

    env = {**os.environ, "ASCEND_RT_VISIBLE_DEVICES": ",".join(str(d) for d in visible)}
    proc = subprocess.run(
        [sys.executable, str(_COMM_DOMAIN), "-p", st_platform, "-d", logical],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=_TIMEOUT_S,
        check=False,
    )
    if proc.returncode != 0:
        print(proc.stdout, file=sys.stderr)
    assert proc.returncode == 0, (
        f"comm_domain/async_notify failed under "
        f"ASCEND_RT_VISIBLE_DEVICES={env['ASCEND_RT_VISIBLE_DEVICES']} "
        f"(logical {logical} -> driver-visible {visible}), rc={proc.returncode}"
    )
