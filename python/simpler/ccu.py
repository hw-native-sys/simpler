# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Internal CCU registration and execution ownership for a chip process."""

from __future__ import annotations

import hashlib
import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path


def ccu_sdk_roots() -> tuple[Path, Path]:
    """Return the CANN installation and current Host architecture roots in lookup order."""
    home = os.environ.get("ASCEND_HOME_PATH")
    if not home:
        raise FileNotFoundError("CANN CCU SDK requires ASCEND_HOME_PATH to point to the CANN installation")
    sdk = Path(home).absolute()
    return sdk, sdk / f"{platform.machine()}-linux"


def ccu_sdk_root() -> Path:
    """Locate the host CANN SDK used by CCU modules and their build identity."""
    roots = ccu_sdk_roots()
    headers = ("include/hcomm/ccu/ccu_res.h", "include/acl/acl.h")
    for root in roots:
        if all((root / header).is_file() for header in headers):
            return root
    raise FileNotFoundError(f"CANN CCU SDK requires {', '.join(headers)}; searched {', '.join(map(str, roots))}")


def runtime_build_identity() -> tuple[bytes, bytes]:
    """Return the SDK and pinned PTO-ISA SHA-256 bytes stored in a CCU manifest."""
    from simpler_setup.pto_isa import ensure_pto_isa_root  # noqa: PLC0415 -- CCU-only toolchain discovery

    candidates = tuple(root / "include/hcomm/ccu/ccu_res.h" for root in ccu_sdk_roots())
    sdk_header = next((path for path in candidates if path.is_file()), None)
    if sdk_header is None:
        raise FileNotFoundError(f"CANN CCU SDK requires ccu_res.h; searched {', '.join(map(str, candidates))}")
    isa_header = Path(ensure_pto_isa_root()) / "include/pto/pto-inst.hpp"
    return hashlib.sha256(sdk_header.read_bytes()).digest(), hashlib.sha256(isa_header.read_bytes()).digest()


class ChipCcuBinding:
    """Callable registration shared by independently prepared, serial executions."""

    def __init__(self, worker, communicator: int, registration: int, timeout: float):
        self.worker = worker
        self.communicator = communicator
        self.registration = registration
        self.timeout = timeout
        self.generation = 0
        self.active = False

    def prepare(self, run_id, allocation_id):
        self.generation += 1
        self.worker.prepare_ccu_run(self.communicator, self.registration, self.generation, run_id, allocation_id)
        return ChipCcuRun(self, self.generation)


class ChipCcuRun:
    def __init__(self, binding, generation):
        self.binding = binding
        self.generation = generation

    def _control(self, operation):
        binding = self.binding
        return binding.worker.control_ccu_run(binding.communicator, binding.registration, operation, self.generation)

    def begin(self):
        if self.binding.active:
            raise RuntimeError("CCU predecessor has no confirmed stop")
        self.binding.active = True
        try:
            self._control(1)
        except BaseException:
            self.abort()
            raise

    def finish(self):
        try:
            deadline = time.monotonic() + self.binding.timeout
            while self._control(3):
                if time.monotonic() >= deadline:
                    if self._control(9):
                        raise RuntimeError("CCU mission stop is unconfirmed")
                    break
            self._control(7)
        except BaseException:
            self.abort()
            raise
        self.binding.active = False

    def discard(self):
        self._control(7)

    def abort(self):
        # Keep active set until a confirmed stop; a successful abort call alone is insufficient.
        try:
            self._control(4)
        except Exception:
            pass


@dataclass(frozen=True)
class CcuRegistrationHandle:
    """A domain-scoped kernel registration on one target chip worker."""

    domain: object
    owner: object
    callable_handle: object
    ccu_id: int
    worker: int
    registration_id: int
    allocation_id: int
    generation: int

    def require_live(self, owner):
        self.domain._require_ready()
        if owner is not self.owner:
            raise ValueError("CCU registration belongs to a different Worker")
        if self.domain._ccu_registration_failed:
            raise RuntimeError("CCU domain registration cleanup is unconfirmed")
        if (self.domain.allocation_id, self.domain.generation) != (self.allocation_id, self.generation):
            raise RuntimeError("CCU registration domain generation is stale")


class ChipCcuRunSet:
    """One native execution collection, completed before the FIFO successor."""

    def __init__(self, worker, communicator, registrations, generation, dispatch_id, natural_exit_wait=0.1):
        self.worker = worker
        self.communicator = communicator
        self.natural_exit_wait = natural_exit_wait
        self.run_set = worker.prepare_ccu_run_set(communicator, registrations, generation, dispatch_id)
        self.started = False

    def _control(self, operation):
        return self.worker.control_ccu_run_set(self.communicator, self.run_set, operation)

    def begin(self):
        self.started = True
        try:
            self._control(1)
        except BaseException as launch_error:
            try:
                self.abort()
            except BaseException as cleanup_error:
                raise launch_error from cleanup_error
            raise

    def finish(self):
        deadline = time.monotonic() + self.natural_exit_wait
        while self._control(3):
            if time.monotonic() >= deadline:
                if self._control(9):
                    raise RuntimeError("CCU run-set stop is unconfirmed")
                break
        self._control(7)
        self.started = False

    def discard(self):
        self._control(7)

    def abort(self):
        try:
            self._control(4)
        except Exception:
            # Native ownership retains unconfirmed resources and closes admission.
            pass
