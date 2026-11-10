#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Real PER_PEER queue use and deferred completion for configurable URMA jetties."""

import json
import statistics

import torch
from simpler.comm_domain import UrmaConfig
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import CommBufferSpec, DataType, TaskArgs, TensorArgType

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, TensorArg, scene_test
from simpler_setup.scene_test import _rehosted_buffer_for, _rehosted_ref_for

RECORD_WORDS = 112
OBSERVATION_WORDS = 32
HEADER_WORDS = 16
QUEUE_WORDS = 12
MAX_JETTIES = 8
TILE_ELEMS = 8192
CHUNK_ELEMS = 256 * 1024 * 1024 // 4
RESERVED_BYTES = 256
COUNTER_MASK = 2**32 - 1


def multi_jetty_orch(orch, callables, task_args, config):
    elems, slots = int(task_args.elems), int(task_args.slots)
    count = slots * elems
    with orch.allocate_domain(
        workspace_engines={
            "urma": UrmaConfig(jetties_per_aiv=int(task_args.jetties), sq_depth=int(task_args.sq_depth)),
        },
        name="urma_multi_jetty",
        workers=[0, 1],
        window_size=max(RESERVED_BYTES + 2 * count * 4, 4 * 1024 * 1024),
        buffers=[
            CommBufferSpec(name="reserved", dtype="int32", count=64, nbytes=RESERVED_BYTES),
            CommBufferSpec(name="source", dtype="float32", count=count, nbytes=count * 4),
            CommBufferSpec(name="receive", dtype="float32", count=count, nbytes=count * 4),
        ],
    ) as domain:
        # PUT reads the local source; GET reads the peer's independently initialized window.
        rank = 0 if int(task_args.put) else 1
        orch.copy_to(domain[rank].buffers["source"], _rehosted_buffer_for(task_args, task_args.source))
        args = TaskArgs()
        for name in ("source", "receive"):
            args.add_tensor(domain[0].buffers[name].tensor((count,), DataType.FLOAT32), TensorArgType.INPUT)
        for name in ("out", "result", "records", "observations"):
            args.add_tensor(_rehosted_ref_for(task_args, getattr(task_args, name)), TensorArgType.OUTPUT_EXISTING)
        args.add_scalar(domain.device_context(0, workspace_engines=("urma",)))
        for name in ("elems", "slots", "waves", "repeats", "put"):
            args.add_scalar(getattr(task_args, name))
        orch.submit_next_level(callables.multi_jetty, args, config, worker=0)


def _case(name, *, jetties=2, elems=CHUNK_ELEMS + TILE_ELEMS, put=0, waves=2, repeats=1, slots=1, **extra):
    case = {
        "name": name,
        "platforms": ["a5"],
        "config": {"device_count": 2, "num_sub_workers": 0},
        "params": {
            "jetties": jetties,
            "elems": elems,
            "put": put,
            "waves": waves,
            "repeats": repeats,
            "slots": slots,
            **extra,
        },
    }
    if extra.get("wrap"):
        # A single scheduler owner bounds the queue owners used across tasks.
        case["config"]["aicpu_thread_num"] = 2
    return case


@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class _UrmaMultiJetty(SceneTestCase):
    CALLABLE = {
        "orchestration": multi_jetty_orch,
        "callables": [
            {
                "name": "multi_jetty",
                "orchestration": {
                    "source": "kernels/orchestration/multi_jetty_orch.cpp",
                    "function_name": "aicpu_orchestration_entry",
                    "signature": [D.IN, D.IN, D.OUT, D.OUT, D.OUT, D.OUT] + [D.IN] * 6,
                },
                "incores": [
                    {
                        "func_id": 0,
                        "source": "kernels/aiv/kernel_transfer.cpp",
                        "core_type": "aiv",
                        "signature": [D.IN, D.IN, D.OUT, D.OUT] + [D.IN] * 5,
                    },
                    {
                        "func_id": 1,
                        "source": "kernels/aiv/kernel_observe.cpp",
                        "core_type": "aiv",
                        "signature": [D.IN, D.OUT, D.IN],
                    },
                    {
                        "func_id": 2,
                        "source": "kernels/aiv/kernel_consumer.cpp",
                        "core_type": "aiv",
                        "signature": [D.IN, D.OUT, D.IN],
                    },
                ],
            }
        ],
    }

    def generate_args(self, params):
        elems, slots = params["elems"], params["slots"]
        assert elems > 0 and elems % TILE_ELEMS == 0
        phases = params["waves"] + params["put"]
        pattern = torch.arange(TILE_ELEMS, dtype=torch.float32).remainder_(251).add_(10)
        source = pattern.repeat(slots, elems // TILE_ELEMS)
        for sender in range(slots):
            source[sender].add_(10000 * sender)
            for chunk, offset in enumerate(range(0, elems, CHUNK_ELEMS)):
                source[sender, offset : offset + CHUNK_ELEMS].add_(1000 * chunk)
        return TaskArgsBuilder(
            TensorArg("source", source.flatten()),
            TensorArg("out", torch.zeros(slots * elems)),
            TensorArg("result", torch.zeros(slots * elems)),
            TensorArg("records", torch.zeros(phases * slots * RECORD_WORDS, dtype=torch.int64)),
            TensorArg("observations", torch.zeros(phases * slots * OBSERVATION_WORDS, dtype=torch.int64)),
            *(Scalar(key, params[key]) for key in ("elems", "slots", "waves", "repeats", "put", "jetties")),
            Scalar("sq_depth", params.get("sq_depth", 0)),
        )

    def compute_golden(self, args, params):
        pass

    def compare_outputs(self, test_args, golden_args, output_names, params):
        assert torch.equal(test_args.out, test_args.source), "remote data differs after deferred completion"
        # Bound comparison temporaries even for transfers larger than 512 MiB.
        for offset in range(0, test_args.source.numel(), 1024 * 1024):
            end = offset + 1024 * 1024
            assert torch.equal(test_args.result[offset:end], test_args.source[offset:end] + 1), "consumer mismatch"
        slots, jetties, waves = (params[key] for key in ("slots", "jetties", "waves"))
        phases = waves + params["put"]
        chunks = (params["elems"] + CHUNK_ELEMS - 1) // CHUNK_ELEMS
        records = test_args.records.reshape(phases, slots, RECORD_WORDS)
        observations = test_args.observations.reshape(phases, slots, OBSERVATION_WORDS)
        assert torch.all(records[:, :, 0] == 1), f"failed transfer records: {records[:, :, :HEADER_WORDS]}"
        assert torch.all(observations[:, :, 1] == 1), "dependent queue observer did not finish"
        assert torch.all(records[:, :, 2] == jetties), "workspace ignored the requested jetty count"
        assert torch.all(records[:, :, 15] == 0), "completion registration reported an error"
        assert torch.all(records[:, :, 12] == chunks)
        assert torch.all(observations[:, :, 0] >= records[:, :, 7]), "observer started before posting returned"
        queue_history = {}
        reused = 0
        durations = []
        first_rows = []
        for phase in range(phases):
            assert len(set(records[phase, :, 1].tolist())) == slots, "SPMD senders share a queue owner"
            phase_rows = set()
            for logical in range(slots):
                record = records[phase, logical]
                observed = observations[phase, logical]
                repeats = params["repeats"] if phase < waves else 1
                assert int(record[5]) == chunks * repeats, "wrong number of per-chunk completion targets"
                assert int(record[14]) == logical
                assert int(record[3]) == 1
                sender = int(record[1])
                rows_per_peer = int(record[4])
                assert 0 <= sender * jetties < rows_per_peer
                queues = record[HEADER_WORDS:].reshape(MAX_JETTIES, QUEUE_WORDS)
                tails = observed[2 : 2 + MAX_JETTIES * 2].reshape(MAX_JETTIES, 2)
                for jetty in range(jetties):
                    queue = queues[jetty]
                    row = int(queue[0])
                    if params.get("sq_depth", 0):
                        assert int(queue[1]) >= params["sq_depth"], "SQ capacity is below the requested depth"
                    assert row == rows_per_peer + sender * jetties + jetty
                    assert row not in phase_rows, "distinct AIV queue groups overlap"
                    phase_rows.add(row)
                    required = (chunks // jetties + int(jetty < chunks % jetties)) * repeats
                    posted_bb = (int(queue[7]) - int(queue[3])) & COUNTER_MASK
                    posted_cqe = (int(queue[8]) - int(queue[5])) & COUNTER_MASK
                    assert posted_bb == posted_cqe == required, (
                        f"queue {row}: expected {required}, got {queue.tolist()}"
                    )
                    assert int(queue[11]) == required, "event targets do not match actual queue use"
                    assert int(queue[9]) == int(queue[7]) and int(queue[10]) == int(queue[8])
                    assert int(tails[jetty, 0]) == int(queue[7]), "consumer released before the SQ target"
                    assert int(tails[jetty, 1]) == int(queue[8]), "consumer released before the CQ target"
                    assert int(queue[4]) == int(queue[3]) and int(queue[6]) == int(queue[5])
                    if row in queue_history:
                        prior = queue_history[row]
                        assert int(queue[3]) == prior[2] and int(queue[5]) == prior[3], "reuse reset queue counters"
                        prior[2:4] = [int(queue[7]), int(queue[8])]
                        reused += 1
                    else:
                        queue_history[row] = [
                            int(queue[3]),
                            int(queue[5]),
                            int(queue[7]),
                            int(queue[8]),
                            int(queue[1]),
                            int(queue[2]),
                        ]
                    if phase == 0:
                        first_rows.append(
                            {
                                "sender": sender,
                                "row": row,
                                "posted_bb": posted_bb,
                                "posted_cqe": posted_cqe,
                                "completed_bb": (int(tails[jetty, 0]) - int(queue[4])) & COUNTER_MASK,
                                "completed_cqe": (int(tails[jetty, 1]) - int(queue[6])) & COUNTER_MASK,
                            }
                        )
                if phase < waves:
                    durations.append((int(observed[0]) - int(record[6])) * 1e6 / int(record[8]))
        wrapped = [
            row
            for row, (b0, c0, b1, c1, bd, cd) in queue_history.items()
            if ((b1 - b0) & COUNTER_MASK) > bd and ((c1 - c0) & COUNTER_MASK) > cd
        ]
        if params.get("wrap"):
            assert wrapped, f"no actual SQ/CQ wrapped: {queue_history}"
            assert reused > 0, "wrap case did not reuse any sender queue"
        first = records[0, 0]
        first_queues = first[HEADER_WORDS:].reshape(MAX_JETTIES, QUEUE_WORDS)[:jetties]
        if params.get("equal_sq_capacity"):
            capacity = int(first_queues[:, 1].sum())
            capacities = self.__dict__.setdefault("_equal_sq_capacities", {})
            reference = capacities.setdefault(params["put"], capacity)
            assert capacity == reference, "provider returned unequal aggregate SQ capacities for the comparison"
        median_us = statistics.median(durations[slots:] or durations)
        print(
            "URMA_MULTI_JETTY "
            + json.dumps(
                {
                    "jetties": jetties,
                    "requested_sq_depth": params.get("sq_depth", 0),
                    "bytes_per_request": params["elems"] * 4,
                    "put": bool(params["put"]),
                    "senders": slots,
                    "repeats_per_task": params["repeats"],
                    "observed_completion_median_us": median_us,
                    "observed_bytes_per_us": params["elems"] * 4 * params["repeats"] / median_us,
                    "workspace_bytes": int(first[9]),
                    "queue_context_rows": int(first[10]),
                    "sq_depths": first_queues[:, 1].tolist(),
                    "cq_depths": first_queues[:, 2].tolist(),
                    "sender_sq_cq_payload_bytes": int((first_queues[:, 1:3] * 64).sum()),
                    "first_phase_queues": first_rows,
                    "reused_queue_observations": reused,
                    "wrapped_rows": wrapped,
                },
                sort_keys=True,
            )
        )


@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestUrmaMultiJettyBoundaries(_UrmaMultiJetty):
    CASES = [
        _case(f"{'put' if put else 'get'}_{boundary}_j{jetties}", jetties=jetties, elems=elems, put=put)
        for put in (0, 1)
        for jetties in (1, 2)
        for boundary, elems in (
            ("below", CHUNK_ELEMS - TILE_ELEMS),
            ("exact", CHUNK_ELEMS),
            ("above", CHUNK_ELEMS + TILE_ELEMS),
            ("tail", 2 * CHUNK_ELEMS + TILE_ELEMS),
        )
    ]


@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestUrmaMultiJettySmallRequests(_UrmaMultiJetty):
    CASES = [
        _case(f"small_{'put' if put else 'get'}", elems=TILE_ELEMS, put=put, repeats=16, waves=4) for put in (0, 1)
    ]


@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestUrmaMultiJettySenders(_UrmaMultiJetty):
    CASES = [_case("two_senders", slots=2, put=1, repeats=4, waves=3)]


@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestUrmaMultiJettyDepth(_UrmaMultiJetty):
    CASES = [_case("explicit_sq_depth", elems=TILE_ELEMS, put=1, repeats=16, waves=4, sq_depth=1024)]


@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestUrmaMultiJettyQueueWrap(_UrmaMultiJetty):
    CASES = [_case("small_queue_wrap", elems=TILE_ELEMS, put=1, waves=4097, repeats=8, slots=2, wrap=True)]


@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestUrmaMultiJettyTiming(_UrmaMultiJetty):
    CASES = [
        {
            **_case(
                f"timing_{'put' if put else 'get'}_{budget}_j{jetties}",
                jetties=jetties,
                elems=2 * CHUNK_ELEMS + TILE_ELEMS,
                put=put,
                waves=21,
                sq_depth=0 if budget == "default_depth" else 2048 // jetties,
                equal_sq_capacity=budget == "equal_sq_capacity",
            ),
            "manual": True,
        }
        for put in (0, 1)
        for budget in ("default_depth", "equal_sq_capacity")
        for jetties in (1, 2)
    ]


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
