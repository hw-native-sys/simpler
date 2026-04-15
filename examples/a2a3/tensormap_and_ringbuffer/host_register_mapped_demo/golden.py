# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Golden script for the host_register_mapped demo."""

import atexit
import ctypes
import logging

import numpy as np
import torch

from simpler.task_interface import host_free, host_malloc, host_register_mapped, host_unregister_mapped

logger = logging.getLogger(__name__)

__outputs__ = ["mapped_out"]

RTOL = 1e-5
ATOL = 1e-5

ROWS = 128
COLS = 128
SIZE = ROWS * COLS

_MAPPED_STATE = {}


def _cleanup_mapped_state() -> None:
    host_ptr = _MAPPED_STATE.get("host_ptr", 0)
    if not host_ptr:
        _MAPPED_STATE.clear()
        return

    try:
        if _MAPPED_STATE.get("registered", False):
            host_unregister_mapped(host_ptr)
    except Exception as exc:  # noqa: BLE001
        logger.warning("host_unregister_mapped cleanup failed: %s", exc)

    try:
        host_free(host_ptr)
    except Exception as exc:  # noqa: BLE001
        logger.warning("host_free cleanup failed: %s", exc)

    _MAPPED_STATE.clear()


atexit.register(_cleanup_mapped_state)


def generate_inputs(params: dict) -> list:
    del params
    _cleanup_mapped_state()

    host_ptr = host_malloc(SIZE * ctypes.sizeof(ctypes.c_float))
    if host_ptr == 0:
        raise RuntimeError("host_malloc returned null")

    host_buf = (ctypes.c_float * SIZE).from_address(host_ptr)
    host_np = np.ctypeslib.as_array(host_buf)
    host_np[:] = np.arange(SIZE, dtype=np.float32)
    host_tensor = torch.from_numpy(host_np)

    try:
        mapped_dev_ptr = host_register_mapped(host_ptr, host_tensor.numel() * host_tensor.element_size())
    except Exception:
        host_free(host_ptr)
        raise

    mapped_out = torch.zeros_like(host_tensor)

    _MAPPED_STATE.update(
        {
            "host_ptr": host_ptr,
            "mapped_dev_ptr": mapped_dev_ptr,
            "registered": True,
            "host_buf": host_buf,
            "host_np": host_np,
            "host_tensor": host_tensor,
        }
    )

    logger.info(
        "host_register_mapped_demo: host_ptr=0x%x mapped_dev_ptr=0x%x size=%d",
        host_ptr,
        mapped_dev_ptr,
        host_tensor.numel() * host_tensor.element_size(),
    )

    return [
        ("mapped_out", mapped_out),
        ("mapped_dev_ptr", ctypes.c_uint64(mapped_dev_ptr)),
    ]


def compute_golden(tensors: dict, params: dict) -> None:
    del params
    host_tensor = _MAPPED_STATE["host_tensor"]
    tensors["mapped_out"][:] = host_tensor + 1.0


def post_run_collect(outputs: dict, params: dict) -> None:
    del outputs
    del params
    _cleanup_mapped_state()
