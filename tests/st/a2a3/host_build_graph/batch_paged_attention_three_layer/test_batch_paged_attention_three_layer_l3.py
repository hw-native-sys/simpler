#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Level-3 coverage for streaming HostGraph paged attention."""

import threading
import time

import torch
from simpler.l3_l2_message_queue import L3L2QueueOpcode
from simpler.request_session import (
    HostGraphToken,
    append_host_graph_prepared_request_args,
    append_host_graph_token_stream_args,
    decode_host_graph_token,
)
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, Tensor, scene_test
from simpler_setup.goldens.paged_attention import compute_golden as _pa_compute_golden
from simpler_setup.goldens.paged_attention import generate_inputs as _pa_generate_inputs
from simpler_setup.scene_test import CallableNamespace, _build_l3_task_args, _compare_outputs

_STREAM_TIMEOUT_S = 30.0
_CROSS_REQUEST_STAGGER_S = 0.02
_REQUEST_A = 1
_REQUEST_B = 2

_SMALL_PARAMS = {
    "batch": 1,
    "num_heads": 16,
    "kv_head_num": 1,
    "head_dim": 16,
    "block_size": 16,
    "context_len": 33,
    "max_model_len": 256,
    "dtype": "bfloat16",
    "layer_count": 3,
    "layers_per_epoch": 1,
}
_STRESS_PARAMS = {
    "batch": 16,
    "num_heads": 16,
    "kv_head_num": 1,
    "head_dim": 128,
    "block_size": 128,
    "context_len": 8192,
    "max_model_len": 8192,
    "dtype": "bfloat16",
    "layer_count": 120,
    "layers_per_epoch": 40,
}
_STRESS_CONFIG = {
    "device_count": 1,
    "num_sub_workers": 0,
    "aicpu_thread_num": 4,
    "block_dim": 24,
    "runtime_env": {
        "ring_task_window": 32768,
        "ring_heap": 1024 * 1024 * 1024,
        "ring_dep_pool": 32768,
    },
}


def _read_token(queue, request_id: int, expected_seq: int, expected_epochs: int) -> tuple[HostGraphToken, object]:
    message = queue.output.peek(timeout=_STREAM_TIMEOUT_S)
    assert message.opcode == L3L2QueueOpcode.DATA
    payload = bytearray(message.payload_nbytes)
    queue.output.read_into(message, payload)
    token = decode_host_graph_token(payload)
    assert token.request_id == request_id
    assert token.token_seq == expected_seq
    assert token.token_id == expected_seq
    assert token.is_final == (expected_seq == expected_epochs)
    assert token.synthetic
    assert token.status == 0
    return token, message


def run_dag(orch, callables, task_args, config, *, request_id=_REQUEST_A, token_sink=None):
    chip_args, _ = _build_l3_task_args(task_args, callables.paged_attention_sig)
    layer_count = int(chip_args.scalar(1))
    layers_per_epoch = int(chip_args.scalar(2))
    expected_epochs = (layer_count + layers_per_epoch - 1) // layers_per_epoch
    queue = orch.create_l3_l2_queue(worker_id=0, depth=1, input_arena_bytes=64, output_arena_bytes=4096)
    append_host_graph_token_stream_args(chip_args, queue, request_id)
    callables.keep(chip_args)
    orch.submit_next_level(callables.paged_attention, chip_args, config, worker=0)

    for seq in range(1, expected_epochs + 1):
        token, message = _read_token(queue, request_id, seq, expected_epochs)
        if token_sink is not None:
            token_sink(token)
        queue.output.release(message)


@scene_test(level=3, runtime="host_build_graph")
class TestBatchPagedAttentionThreeLayerL3(SceneTestCase):
    RTOL = 1e-3
    ATOL = 1e-3

    CALLABLE = {
        "orchestration": run_dag,
        "callables": [
            {
                "name": "paged_attention",
                "orchestration": {
                    "source": "kernels/orchestration/paged_attention_three_layer_orch.cpp",
                    "function_name": "aicpu_orchestration_entry",
                    "signature": [D.IN, D.IN, D.IN, D.IN, D.IN, D.OUT],
                },
                "incores": [
                    {
                        "func_id": 0,
                        "name": "QK",
                        "source": "kernels/aic/aic_qk_matmul.cpp",
                        "core_type": "aic",
                        "signature": [D.IN, D.IN, D.OUT],
                    },
                    {
                        "func_id": 1,
                        "name": "SF",
                        "source": "kernels/aiv/aiv_softmax_prepare.cpp",
                        "core_type": "aiv",
                        "signature": [D.IN, D.OUT, D.OUT, D.OUT],
                    },
                    {
                        "func_id": 2,
                        "name": "PV",
                        "source": "kernels/aic/aic_pv_matmul.cpp",
                        "core_type": "aic",
                        "signature": [D.IN, D.IN, D.OUT],
                    },
                    {
                        "func_id": 3,
                        "name": "UP",
                        "source": "kernels/aiv/aiv_online_update.cpp",
                        "core_type": "aiv",
                        "signature": [D.IN, D.IN, D.IN, D.INOUT, D.INOUT, D.INOUT, D.INOUT],
                    },
                ],
            }
        ],
    }

    CASES = [
        {
            "name": "SmallThreeLayerStreaming",
            "platforms": ["a2a3sim", "a2a3"],
            "config": {"device_count": 1, "num_sub_workers": 0, "aicpu_thread_num": 4, "block_dim": 9},
            "params": _SMALL_PARAMS,
        },
        {
            "name": "ConcurrentTwoRequestOverlapProfile",
            "platforms": ["a2a3"],
            "config": _STRESS_CONFIG,
            "manual": True,
            "params": _STRESS_PARAMS,
        },
    ]

    def _run_and_validate_l3(self, worker, compiled_callables, sub_handles, case, **kwargs):
        if case["name"] != "ConcurrentTwoRequestOverlapProfile":
            return super()._run_and_validate_l3(worker, compiled_callables, sub_handles, case, **kwargs)
        return self._run_concurrent_requests(worker, compiled_callables, sub_handles, case)

    def _build_submission(self, orch, callables, request, request_id: int, *, prepared: bool):
        chip_args, _ = _build_l3_task_args(request, callables.paged_attention_sig)
        layer_count = int(chip_args.scalar(1))
        layers_per_epoch = int(chip_args.scalar(2))
        expected_epochs = (layer_count + layers_per_epoch - 1) // layers_per_epoch
        queue = orch.create_l3_l2_queue(worker_id=0, depth=1, input_arena_bytes=64, output_arena_bytes=4096)
        if prepared:
            append_host_graph_prepared_request_args(chip_args, request_id)
        append_host_graph_token_stream_args(chip_args, queue, request_id)
        callables.keep(chip_args)
        return chip_args, queue, expected_epochs

    def _relay_tokens(self, queue, request_id: int, expected_epochs: int, emitter) -> None:
        for seq in range(1, expected_epochs + 1):
            token, message = _read_token(queue, request_id, seq, expected_epochs)
            print(
                f"L3_TOKEN_RECEIVED request_id={request_id} token_seq={seq} "
                f"received_ns={time.monotonic_ns()} final={token.is_final}"
            )
            try:
                emitter.emit(token, final=token.is_final)
                print(
                    f"USER_TOKEN_DELIVERED request_id={request_id} token_seq={seq} "
                    f"delivered_ns={time.monotonic_ns()} final={token.is_final}"
                )
            finally:
                queue.output.release(message)

    def _run_concurrent_requests(self, worker, compiled_callables, sub_handles, case):
        params = case["params"]
        requests = {request_id: self.generate_args(params) for request_id in (_REQUEST_A, _REQUEST_B)}
        goldens = {request_id: request.clone() for request_id, request in requests.items()}
        for golden in goldens.values():
            self.compute_golden(golden, params)
        config = self._build_config(case["config"])
        callables = CallableNamespace({**compiled_callables, **sub_handles})
        request_a_submitted = threading.Event()

        def request_orch(orch, request, request_id, emitter, cfg):
            prepared = request_id == _REQUEST_B
            chip_args, queue, expected_epochs = self._build_submission(
                orch, callables, request, request_id, prepared=prepared
            )
            if prepared:
                start_ns = time.monotonic_ns()
                emitter.prepare_host_request(
                    0, request_id, callables.paged_attention, chip_args, cfg, arena_bank=1, timeout=_STREAM_TIMEOUT_S
                )
                print(f"HOST_REQUEST_PREPARED request_id={request_id} start_ns={start_ns} end_ns={time.monotonic_ns()}")
            orch.submit_next_level(callables.paged_attention, chip_args, cfg, worker=0)
            if request_id == _REQUEST_A:
                request_a_submitted.set()
            self._relay_tokens(queue, request_id, expected_epochs, emitter)

        def consume(stream, output, errors):
            try:
                output.extend(stream)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        with worker.open_request_session(request_orch, max_active_runs=2) as session:
            streams = {
                _REQUEST_A: session.submit(requests[_REQUEST_A], config=config, request_id=_REQUEST_A),
            }
            assert request_a_submitted.wait(_STREAM_TIMEOUT_S)
            # Place B's Host O behind A's first publication instead of profiling
            # two simultaneous cold starts.
            time.sleep(_CROSS_REQUEST_STAGGER_S)
            streams[_REQUEST_B] = session.submit(requests[_REQUEST_B], config=config, request_id=_REQUEST_B)
            tokens = {request_id: [] for request_id in streams}
            errors = []
            consumers = [
                threading.Thread(target=consume, args=(stream, tokens[request_id], errors))
                for request_id, stream in streams.items()
            ]
            for consumer in consumers:
                consumer.start()
            for consumer in consumers:
                consumer.join(_STREAM_TIMEOUT_S)
                assert not consumer.is_alive(), "request stream consumer timed out"
            if errors:
                raise errors[0]

        expected_epochs = (int(params["layer_count"]) + int(params["layers_per_epoch"]) - 1) // int(
            params["layers_per_epoch"]
        )
        for request_id in streams:
            assert [token.token_seq for token in tokens[request_id]] == list(range(1, expected_epochs + 1))
            _compare_outputs(
                requests[request_id], goldens[request_id], requests[request_id].tensor_names(), self.RTOL, self.ATOL
            )

    def generate_args(self, params):
        specs = [
            Tensor(name, value.share_memory_()) if isinstance(value, torch.Tensor) else Scalar(name, value)
            for name, value in _pa_generate_inputs(params)
        ]
        specs.extend(
            [
                Scalar("layer_count", params["layer_count"]),
                Scalar("layers_per_epoch", params.get("layers_per_epoch", 1)),
            ]
        )
        return TaskArgsBuilder(*specs)

    def compute_golden(self, args, params):
        tensors = {spec.name: spec.value for spec in args.specs if isinstance(spec, Tensor)}
        _pa_compute_golden(tensors, params)
        for spec in args.specs:
            if isinstance(spec, Tensor) and spec.name in tensors:
                getattr(args, spec.name)[:] = tensors[spec.name]


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
