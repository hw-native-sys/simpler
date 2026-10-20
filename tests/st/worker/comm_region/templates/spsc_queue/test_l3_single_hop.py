#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""L3→L2 SPSC queue acceptance scene: binding, duplex DATA/ERROR, wrap, STOP, free."""

from __future__ import annotations

import pytest
from simpler.comm_region_template import SpscQueueOpcode
from simpler.worker import Worker

from ._helpers import (
    _POST_STOP_MARKER,
    _QUEUE_DEPTH,
    _SMALL_ECHO,
    _TIMEOUT_S,
    SUCCESS_PLATFORMS,
    build_chip_callable,
    close_owned_workers,
    compute_payload,
    create_queue,
    drain_message,
    echo_once,
    echo_payload,
    enqueue_bytes,
    error_payload,
    expected_compute_payload,
    region_lifecycle,
    stream_config,
    submit_queue,
    wrap_payload,
)


@pytest.mark.platforms(SUCCESS_PLATFORMS)
@pytest.mark.device_count(1)
@pytest.mark.runtime("tensormap_and_ringbuffer")
def test_l3_single_hop(st_platform, st_device_ids):
    worker = Worker(
        level=3,
        device_ids=[int(st_device_ids[0])],
        num_sub_workers=0,
        platform=st_platform,
        runtime="tensormap_and_ringbuffer",
    )
    recorder = None
    transactions: list[int] = []
    primary = None
    try:
        chip_handle = worker.register(build_chip_callable(st_platform))
        lifecycle = region_lifecycle()
        recorder = lifecycle.install_lifecycle_recorder(worker)
        worker.init()

        def orch(orch_handle, _args, cfg):
            queue = create_queue(orch_handle)
            transactions.append(int(queue._endpoint_binding.transaction_id))
            layout = queue.layout

            for _ in range(_QUEUE_DEPTH):
                assert queue.input.try_enqueue(_SMALL_ECHO, len(_SMALL_ECHO)) is True
            assert queue.input.try_enqueue(_SMALL_ECHO, len(_SMALL_ECHO)) is False
            assert queue.output.try_peek() is None

            submit_queue(orch_handle, chip_handle, queue, cfg)
            for _ in range(_QUEUE_DEPTH):
                opcode, payload, _offset = drain_message(queue)
                assert opcode is SpscQueueOpcode.DATA
                assert payload == _SMALL_ECHO
            assert queue.output.try_peek() is None

            wrap_messages = [wrap_payload(seed) for seed in range(3)]
            wrap_offsets = []
            for payload in wrap_messages:
                enqueue_bytes(queue, payload)
                opcode, got, offset = drain_message(queue)
                assert opcode is SpscQueueOpcode.DATA
                assert got == payload
                wrap_offsets.append(offset)
            assert wrap_offsets[1] > wrap_offsets[0]
            assert wrap_offsets[2] == layout.output_arena_offset

            queue.input.enqueue(None, nbytes=0, timeout=_TIMEOUT_S)
            opcode, payload, offset = drain_message(queue)
            assert opcode is SpscQueueOpcode.DATA
            assert payload == b""
            assert offset == 0

            err1 = error_payload(b"e1")
            echo = echo_payload(b"ok")
            err2 = error_payload(b"e2")
            enqueue_bytes(queue, err1)
            opcode, payload, _offset = drain_message(queue)
            assert opcode is SpscQueueOpcode.ERROR
            assert payload == err1
            enqueue_bytes(queue, echo)
            opcode, payload, _offset = drain_message(queue)
            assert opcode is SpscQueueOpcode.DATA
            assert payload == echo
            enqueue_bytes(queue, err2)
            opcode, payload, _offset = drain_message(queue)
            assert opcode is SpscQueueOpcode.ERROR
            assert payload == err2

            compute = compute_payload(3.0)
            enqueue_bytes(queue, compute)
            opcode, payload, _offset = drain_message(queue)
            assert opcode is SpscQueueOpcode.DATA
            assert payload == expected_compute_payload(3.0)

            queue.request_stop(timeout=_TIMEOUT_S)
            assert queue.input.try_enqueue(_SMALL_ECHO, len(_SMALL_ECHO)) is False
            opcode, payload, _offset = drain_message(queue)
            assert opcode is SpscQueueOpcode.DATA
            assert payload == _POST_STOP_MARKER
            assert queue.output.try_peek() is None

            queue.free()
            with pytest.raises(RuntimeError):
                queue.input.try_enqueue(_SMALL_ECHO, len(_SMALL_ECHO))

        def orch_unfreed(orch_handle, _args, cfg):
            queue = create_queue(orch_handle)
            transactions.append(int(queue._endpoint_binding.transaction_id))
            submit_queue(orch_handle, chip_handle, queue, cfg)
            echo_once(queue, free=False)

        worker.run(orch, args=None, config=stream_config())
        worker.run(orch_unfreed, args=None, config=stream_config())
        assert transactions == [1, 2]
        lifecycle.assert_lifecycle_rounds(recorder, 2)
    except BaseException as exc:
        primary = exc
        raise
    finally:
        try:
            close_owned_workers(primary, worker)
        finally:
            if recorder is not None and recorder.restore is not None:
                recorder.restore()
