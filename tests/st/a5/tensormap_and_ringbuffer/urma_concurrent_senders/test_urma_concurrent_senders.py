#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""URMA queue ownership and deferred data visibility on real A5 devices."""

import torch
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import CommBufferSpec, DataType, TaskArgs, TensorArgType

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, TensorArg, scene_test
from simpler_setup.scene_test import _rehosted_buffer_for, _rehosted_ref_for

RECORD_WORDS = 32
TILE_ELEMS = 8192
RESERVED_BYTES = 256


def concurrent_senders_orch(orch, callables, task_args, config):
    peers = int(task_args.peers)
    slots = int(task_args.slots)
    elems = int(task_args.elems)
    count = slots * elems
    with orch.allocate_domain(
        workspace_engines=("urma",),
        name="urma_concurrent_senders",
        workers=list(range(peers + 1)),
        window_size=max(RESERVED_BYTES + (peers + 1) * count * 4, 4 * 1024 * 1024),
        buffers=[
            CommBufferSpec(name="reserved", dtype="int32", count=64, nbytes=RESERVED_BYTES),
            CommBufferSpec(name="source", dtype="float32", count=peers * count, nbytes=peers * count * 4),
            CommBufferSpec(name="receive", dtype="float32", count=count, nbytes=count * 4),
        ],
    ) as domain:
        orch.copy_to(domain[0].buffers["source"], _rehosted_buffer_for(task_args, task_args.source))
        args = TaskArgs()
        for name in ("source", "receive"):
            args.add_tensor(
                domain[0].buffers[name].tensor((peers * count if name == "source" else count,), DataType.FLOAT32),
                TensorArgType.INPUT,
            )
        for name in ("out", "result", "records", "layout"):
            args.add_tensor(_rehosted_ref_for(task_args, getattr(task_args, name)), TensorArgType.OUTPUT_EXISTING)
        args.add_scalar(domain.device_context(0, workspace_engines=("urma",)))
        for name in ("elems", "slots", "peers", "waves", "repeats", "independent"):
            args.add_scalar(getattr(task_args, name))
        orch.submit_next_level(callables.concurrent_senders, args, config, worker=0)


@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class _UrmaConcurrentSenders(SceneTestCase):
    CALLABLE = {
        "orchestration": concurrent_senders_orch,
        "callables": [
            {
                "name": "concurrent_senders",
                "orchestration": {
                    "source": "kernels/orchestration/concurrent_senders_orch.cpp",
                    "function_name": "aicpu_orchestration_entry",
                    "signature": [D.IN, D.IN, D.OUT, D.OUT, D.OUT, D.OUT] + [D.IN] * 7,
                },
                "incores": [
                    {
                        "func_id": 0,
                        "source": "kernels/aiv/kernel_transfer.cpp",
                        "core_type": "aiv",
                        "signature": [D.IN, D.IN, D.OUT, D.OUT] + [D.IN] * 11,
                    },
                    {
                        "func_id": 1,
                        "source": "kernels/aiv/kernel_consumer.cpp",
                        "core_type": "aiv",
                        "signature": [D.IN, D.OUT, D.IN],
                    },
                    {
                        "func_id": 2,
                        "source": "kernels/aic/kernel_idle.cpp",
                        "core_type": "aic",
                        "signature": [D.IN, D.IN, D.OUT, D.OUT] + [D.IN] * 11,
                    },
                ],
            }
        ],
    }

    def generate_args(self, params):
        slots, peers, waves = (params[key] for key in ("slots", "peers", "waves"))
        elems = params.get("elems", TILE_ELEMS)
        source = torch.arange(slots * elems, dtype=torch.int64).remainder_(251).to(torch.float32)
        source = source.repeat(peers).reshape(peers, -1)
        source.add_(torch.arange(peers, dtype=torch.float32).reshape(-1, 1) * 1000 + 10)
        source = source.flatten()
        marker_words = waves * slots * 16 if params.get("independent") == 3 else 0
        return TaskArgsBuilder(
            TensorArg("source", source),
            TensorArg("out", torch.zeros(peers * slots * elems)),
            TensorArg("result", torch.zeros(peers * slots * elems)),
            TensorArg("records", torch.zeros((waves + 1) * slots * RECORD_WORDS + marker_words, dtype=torch.int64)),
            TensorArg("layout", torch.zeros(16, dtype=torch.int64)),
            Scalar("elems", elems),
            Scalar("slots", slots),
            Scalar("peers", peers),
            Scalar("waves", waves),
            Scalar("repeats", params["repeats"]),
            Scalar("independent", params.get("independent", 0)),
        )

    def compute_golden(self, args, params):
        pass

    def compare_outputs(self, test_args, golden_args, output_names, params):
        width = int(test_args.layout[0])
        slots, peers, waves = (params[key] for key in ("slots", "peers", "waves"))
        elems = params.get("elems", TILE_ELEMS)
        assert 0 < width <= slots
        expected = torch.zeros(peers, slots * elems)
        expected[:, : width * elems] = test_args.source.reshape(peers, -1)[:, : width * elems]
        assert torch.equal(test_args.out.reshape(peers, -1), expected), "PUT -> deferred GET data mismatch"
        assert torch.equal(test_args.result.reshape(peers, -1), expected + 1), "deferred GET consumer mismatch"
        records = test_args.records[: (waves + 1) * slots * RECORD_WORDS].reshape(waves + 1, slots, RECORD_WORDS)[
            :, :width
        ]
        assert torch.all(records[:, :, 0] == 1), f"missing/failed transfer records: {records[:, :, :8]}"
        for wave in range(waves + 1):
            rows = records[wave]
            if wave in (0, waves) or params.get("independent", 0) not in (1, 3):
                assert len(set(rows[:, 1].tolist())) == width, "concurrent senders reused one physical queue owner"
            assert torch.all(rows[:, 6] == 1), "expected the default single jetty per AIV and peer"
            assert torch.all(rows[:, 23] > rows[:, 1] * rows[:, 6]), "sender exceeds the per-peer queue range"
            peer_one_rows = rows[:, 23] + rows[:, 1] * rows[:, 6]
            assert torch.equal(rows[:, 5], peer_one_rows), "queue row differs from peer and stable AIV identity"
            assert torch.equal(rows[:, 24], peer_one_rows), "peer 1 completion used another queue row"
            chunks = (elems * 4 + 256 * 1024 * 1024 - 1) // (256 * 1024 * 1024)
            repetitions = params["repeats"] if wave < waves else 1
            assert torch.equal(rows[:, 7], rows[:, 6] * chunks * peers * repetitions), "missing deferred records"
            assert torch.all(rows[:, 15] == 0), "workspace is not PER_PEER"
            assert torch.all(rows[:, 20] == peers), "not every peer was submitted"
            assert torch.all(rows[:, 31] == 0), "URMA completion registration reported an error"
            assert torch.all(rows[:, 13] > rows[:, 12]), "first queue did not post any request"
            assert torch.equal(rows[:, 26], rows[:, 29] + rows[:, 28]), "peer 1 address differs from its MR offset"
            if peers == 2:
                peer_two_rows = 2 * rows[:, 23] + rows[:, 1] * rows[:, 6]
                assert torch.equal(rows[:, 25], peer_two_rows), "peer 2 completion used another queue row"
                assert torch.all(rows[:, 25] != rows[:, 24]), "different peers shared a physical queue row"
                assert torch.all(rows[:, 14] > rows[:, 22]), "peer 2 queue did not post any request"
                assert torch.equal(rows[:, 27], rows[:, 30] + rows[:, 28]), "peer 2 address differs from its MR offset"
            if params.get("independent") == 2 and wave < waves:
                assert sorted(rows[:, 18].tolist()) == [0, 1], "MIX did not exercise both runtime AIV lanes"
                assert len(set(rows[:, 1].tolist())) == 2, "MIX siblings shared a stable sender identity"
        if width > 1:
            first = records[0]
            assert int(first[:, 8].max()) < int(first[:, 9].min()), "producer intervals did not overlap"
        if params.get("wrap"):
            owners = {}
            for row in records[:waves].reshape(-1, RECORD_WORDS):
                sender = int(row[1])
                if sender not in owners:
                    owners[sender] = [
                        int(row[12]),
                        int(row[16]),
                        int(row[13]),
                        int(row[17]),
                        int(row[11]),
                        int(row[10]),
                    ]
                else:
                    owners[sender][2] = max(owners[sender][2], int(row[13]))
                    owners[sender][3] = max(owners[sender][3], int(row[17]))
            wrapped = [sender for sender, (c0, b0, c1, b1, cd, bd) in owners.items() if c1 - c0 > cd and b1 - b0 > bd]
            assert wrapped, f"no SQ/CQ pair wrapped: {owners}"
            print(f"URMA wrapped queue owners: {wrapped}")
        if params.get("independent") == 3:
            pending = records[1, :, 12] - records[1, :, 19]
            print(f"URMA reuse observation, submitted-minus-completed at entry: {pending.tolist()}")
        print(
            f"URMA PER_PEER acceptance: senders={width}, peers={peers}, waves={waves}, "
            f"jetties={sorted(set(records[:, :, 6].flatten().tolist()))}, "
            f"sq_depth={int(records[0, 0, 10])}, cq_depth={int(records[0, 0, 11])}"
        )
        if params.get("independent") == 2:
            print(
                f"URMA MIX identities [stable, logical, raw CCE block, raw CCE subblock]: "
                f"{records[0, :, 1:5].tolist()}, runtime lanes={records[0, :, 18].tolist()}"
            )
        if peers == 2:
            print(f"URMA per-peer queue rows [peer 1, peer 2]: {records[0, :, 24:26].tolist()}")


@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestUrmaSamePeer(_UrmaConcurrentSenders):
    CASES = [
        {
            "name": "same_peer",
            "platforms": ["a5"],
            "config": {"device_count": 2, "num_sub_workers": 0},
            "params": {"slots": 2, "peers": 1, "waves": 3, "repeats": 4},
        }
    ]


@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestUrmaConcurrentTasks(_UrmaConcurrentSenders):
    CASES = [
        {
            "name": "concurrent_tasks",
            "platforms": ["a5"],
            "config": {"device_count": 2, "num_sub_workers": 0},
            "params": {"slots": 2, "peers": 1, "waves": 3, "repeats": 4, "independent": 1},
        }
    ]


@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestUrmaMixSiblings(_UrmaConcurrentSenders):
    CASES = [
        {
            "name": "mix_siblings",
            "platforms": ["a5"],
            "config": {"device_count": 2, "num_sub_workers": 0},
            "params": {"slots": 2, "peers": 1, "waves": 3, "repeats": 4, "independent": 2},
        }
    ]


@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestUrmaTwoPeers(_UrmaConcurrentSenders):
    CASES = [
        {
            "name": "two_peers",
            "platforms": ["a5"],
            "config": {"device_count": 3, "num_sub_workers": 0},
            "params": {"slots": 1, "peers": 2, "waves": 3, "repeats": 4},
        }
    ]


@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestUrmaQueueWrap(_UrmaConcurrentSenders):
    CASES = [
        {
            "name": "queue_wrap",
            "platforms": ["a5"],
            # One scheduler owner keeps the two producers on a bounded set of queues for physical ring reuse.
            "config": {"device_count": 2, "num_sub_workers": 0, "aicpu_thread_num": 2},
            "params": {"slots": 2, "peers": 1, "waves": 4097, "repeats": 8, "independent": 1, "wrap": True},
        }
    ]


@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestUrmaReuseObservation(_UrmaConcurrentSenders):
    CASES = [
        {
            "name": "reuse_observation",
            "platforms": ["a5"],
            "config": {"device_count": 2, "num_sub_workers": 0},
            "params": {"slots": 2, "peers": 1, "waves": 2, "repeats": 8, "independent": 3},
        }
    ]


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
