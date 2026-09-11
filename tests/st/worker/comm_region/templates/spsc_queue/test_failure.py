#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Scene-layer SPSC queue construction and poison failures. Sim-only."""

from __future__ import annotations

import struct

import pytest
from simpler.task_interface import TaskArgs

from ._helpers import (
    _SMALL_ECHO,
    _TIMEOUT_S,
    FAULT_PLATFORMS,
    close_owned_workers,
    create_queue,
    echo_payload,
    enqueue_bytes,
    make_l3_worker,
    stream_config,
    submit_queue,
)


def _exception_text(exc: BaseException) -> str:
    parts = [str(exc)]
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        parts.append(str(current))
        current = current.__cause__ if current.__cause__ is not None else current.__context__
    return "\n".join(parts)


@pytest.mark.platforms(FAULT_PLATFORMS)
@pytest.mark.device_count(1)
@pytest.mark.runtime("tensormap_and_ringbuffer")
def test_timeout_is_nonterminal(st_platform, st_device_ids):
    worker, _chip_handle = make_l3_worker(st_platform, int(st_device_ids[0]))
    primary = None
    try:

        def orch(orch_handle, _args, cfg):
            queue = create_queue(orch_handle)
            with pytest.raises(TimeoutError, match="timed out"):
                queue.output.peek(timeout=0.2)
            assert queue.input.try_enqueue(_SMALL_ECHO, len(_SMALL_ECHO)) is True
            queue.free()

        worker.run(orch, args=None, config=stream_config())
    except BaseException as exc:
        primary = exc
        raise
    finally:
        close_owned_workers(primary, worker)


@pytest.mark.platforms(FAULT_PLATFORMS)
@pytest.mark.device_count(1)
@pytest.mark.runtime("tensormap_and_ringbuffer")
def test_peer_rejects_zero_transaction(st_platform, st_device_ids):
    worker, chip_handle = make_l3_worker(st_platform, int(st_device_ids[0]))
    primary = None
    try:

        def orch(orch_handle, _args, cfg):
            queue = create_queue(orch_handle)
            scalars = list(queue.chip_task_arg_scalars())
            scalars[2] = 0
            task_args = TaskArgs()
            for value in scalars:
                task_args.add_scalar(int(value))
            orch_handle.submit_next_level(chip_handle, task_args, cfg, worker=0)

        with pytest.raises(BaseException) as excinfo:  # noqa: PT011
            worker.run(orch, args=None, config=stream_config())
        text = _exception_text(excinfo.value)
        assert "transaction=" not in text
    except BaseException as exc:
        primary = exc
        raise
    finally:
        close_owned_workers(primary, worker)


@pytest.mark.platforms(FAULT_PLATFORMS)
@pytest.mark.device_count(1)
@pytest.mark.runtime("tensormap_and_ringbuffer")
def test_peer_rejects_wrong_magic(st_platform, st_device_ids):
    worker, chip_handle = make_l3_worker(st_platform, int(st_device_ids[0]))
    primary = None
    try:

        def orch(orch_handle, _args, cfg):
            queue = create_queue(orch_handle)
            scalars = list(queue.chip_task_arg_scalars())
            scalars[0] = 0x4C33513200010001
            task_args = TaskArgs()
            for value in scalars:
                task_args.add_scalar(int(value))
            orch_handle.submit_next_level(chip_handle, task_args, cfg, worker=0)

        with pytest.raises(BaseException) as excinfo:  # noqa: PT011
            worker.run(orch, args=None, config=stream_config())
        text = _exception_text(excinfo.value)
        assert "transaction=" not in text
    except BaseException as exc:
        primary = exc
        raise
    finally:
        close_owned_workers(primary, worker)


@pytest.mark.platforms(FAULT_PLATFORMS)
@pytest.mark.device_count(1)
@pytest.mark.runtime("tensormap_and_ringbuffer")
def test_malformed_descriptor_poisons_with_identity(st_platform, st_device_ids):
    worker, chip_handle = make_l3_worker(st_platform, int(st_device_ids[0]))
    primary = None
    try:

        def orch(orch_handle, _args, cfg):
            queue = create_queue(orch_handle)
            enqueue_bytes(queue, echo_payload(b"bad-seq"))
            smashed = bytearray(8)
            struct.pack_into("<Q", smashed, 0, 99)
            queue.region.payload_write(queue.layout.input_desc_offset, smashed, 8)
            submit_queue(orch_handle, chip_handle, queue, cfg)
            queue.output.peek(timeout=_TIMEOUT_S)

        with pytest.raises(BaseException) as excinfo:  # noqa: PT011
            worker.run(orch, args=None, config=stream_config())
        text = _exception_text(excinfo.value)
        assert "remote abort" in text
    except BaseException as exc:
        primary = exc
        raise
    finally:
        close_owned_workers(primary, worker)


@pytest.mark.platforms(FAULT_PLATFORMS)
@pytest.mark.device_count(1)
@pytest.mark.runtime("tensormap_and_ringbuffer")
def test_stale_release_poisons_locally(st_platform, st_device_ids):
    worker, chip_handle = make_l3_worker(st_platform, int(st_device_ids[0]))
    primary = None
    saw_local_poison = False
    try:

        def orch(orch_handle, _args, cfg):
            nonlocal saw_local_poison
            queue = create_queue(orch_handle)
            submit_queue(orch_handle, chip_handle, queue, cfg)
            enqueue_bytes(queue, echo_payload(b"once"))
            message = queue.output.peek(timeout=_TIMEOUT_S)
            queue.output.release(message)
            with pytest.raises(RuntimeError, match="not active"):
                queue.output.release(message)
            saw_local_poison = True

        try:
            worker.run(orch, args=None, config=stream_config())
        except BaseException:
            if not saw_local_poison:
                raise
        assert saw_local_poison
    except BaseException as exc:
        primary = exc
        raise
    finally:
        close_owned_workers(primary, worker)
