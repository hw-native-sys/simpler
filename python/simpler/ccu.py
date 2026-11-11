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
import time
from pathlib import Path


def runtime_build_identity() -> tuple[bytes, bytes]:
    from simpler_setup.pto_isa import ensure_pto_isa_root  # noqa: PLC0415 -- CCU-only toolchain discovery

    sdk_header = Path(os.environ["ASCEND_HOME_PATH"]) / "aarch64-linux/include/hcomm/ccu/ccu_res.h"
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
                    raise TimeoutError(
                        f"CCU exit protocol timed out for registration {self.binding.registration}, "
                        f"generation {self.generation}. Check the user kernel trigger and exit logic."
                    )
        except BaseException:
            self.abort()
            raise
        self.binding.active = False
        self._control(7)

    def discard(self):
        self._control(7)

    def abort(self):
        # Keep active set until a confirmed stop; a successful abort call alone is insufficient.
        try:
            self._control(4)
        except Exception:
            pass
