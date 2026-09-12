# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Case-owned L2 device tensors shared by scene and streaming drivers."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class ResidentTaskArgs:
    """Own fixed device addresses until release; upload each input exactly once.

    ``add`` consumes one CPU contiguous fixture at a time, so a streaming driver
    can discard each large weight before materializing its successor. Callers
    retain ordinary host outputs separately when they need copy-back.
    """

    def __init__(self, worker):
        self.worker = worker
        self.buffers = {}
        self.tensors = {}
        self.directions = {}

    def add(self, name, host, direction):
        from simpler.task_interface import ArgDirection as D  # noqa: PLC0415

        from simpler_setup.torch_interop import torch_dtype_to_datatype  # noqa: PLC0415

        if name in self.directions:
            raise ValueError(f"Duplicate resident tensor {name!r}")
        if direction not in (D.IN, D.OUT, D.INOUT):
            raise ValueError(f"Resident tensor {name!r} has an unsupported direction")
        if host.device.type != "cpu" or not host.is_contiguous():
            raise ValueError(f"Resident tensor {name!r} must be a contiguous CPU tensor")
        size = host.numel() * host.element_size()
        if not size:
            # An empty tensor names no device bytes. Leaving it unrecorded keeps
            # it on the ordinary host-staging path, so `build_args` callers must
            # reconcile their own argument list -- see the count check there.
            return
        buf = self.worker.malloc(size)
        try:
            if direction != D.OUT:
                self.worker.copy_to(buf, host)
            tensor = buf.tensor(tuple(host.shape), int(torch_dtype_to_datatype(host.dtype).value))
        except BaseException:
            try:
                self.worker.free(buf)
            except Exception as exc:  # noqa: BLE001 -- preserve the construction failure
                logger.warning("Resident tensor cleanup failed: %s", exc)
            raise
        self.buffers[name] = buf
        self.tensors[name] = tensor
        self.directions[name] = direction

    def build_args(self, expected_count=None):
        """Build all-resident L2 args, preserving direction.

        ``expected_count`` is the caller's tensor-argument count. A mismatch means
        `add` skipped an empty tensor, so every later argument would shift against
        the orchestration signature -- reject that rather than dispatch a silently
        misaligned argument list.
        """
        from simpler.task_interface import ArgDirection as D  # noqa: PLC0415
        from simpler.task_interface import TaskArgs, TensorArgType  # noqa: PLC0415

        if expected_count is not None and expected_count != len(self.tensors):
            raise ValueError(
                f"build_args expected {expected_count} resident tensors but holds {len(self.tensors)}; "
                "an empty tensor cannot be resident -- keep it on the host-staging path instead."
            )
        tags = {D.IN: TensorArgType.INPUT, D.OUT: TensorArgType.OUTPUT_EXISTING, D.INOUT: TensorArgType.INOUT}
        args = TaskArgs()
        for name, tensor in self.tensors.items():
            args.add_tensor(tensor, tags[self.directions[name]])
        return args

    def copy_back(self, test_args, names):
        for name in names:
            if name in self.buffers:
                self.worker.copy_from(getattr(test_args, name), self.buffers[name])

    def release(self):
        """Release in LIFO order, even when an individual free fails."""
        self.tensors.clear()
        self.directions.clear()
        while self.buffers:
            _, buf = self.buffers.popitem()
            try:
                self.worker.free(buf)
            except Exception as exc:  # noqa: BLE001 -- attempt all frees, preserve the test result
                logger.warning("Resident tensor cleanup failed: %s", exc)

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.release()
