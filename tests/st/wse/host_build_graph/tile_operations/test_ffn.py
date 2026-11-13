# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Independent logical FFN blocks using the temporary WSE operation interface."""

import json
from functools import lru_cache
from pathlib import Path

import pytest
import torch
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import CallConfig, DataType, TaskArgs, TensorArgType
from simpler.worker import Worker

from tests.st.wse.host_build_graph.fixed_sram_layout import copy_from, copy_to, layout_view, release_view
from tests.st.wse.host_build_graph.scene_helpers import build_callable
from tests.st.wse.host_build_graph.scene_helpers import ffn_inputs as shared_ffn_inputs

pytestmark = [pytest.mark.platforms(["wsesim"]), pytest.mark.runtime("host_build_graph"), pytest.mark.device_count(1)]
HERE = Path(__file__).parent


@lru_cache
def build_ffn(platform):
    signatures = [[D.IN, D.IN, D.OUT, D.OUT], [D.IN, D.OUT], [D.IN, D.IN, D.OUT]]
    return build_callable(
        platform,
        HERE / "kernels",
        "ffn_orch.cpp",
        [D.IN] * 3 + [D.OUT] * 4,
        [(f"ffn_{stage}.cpp", signature) for stage, signature in enumerate(signatures)],
    )


def ffn_inputs(seed):
    return shared_ffn_inputs(seed, 40)


def audit_ffn(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    header, events = rows[0], rows[1:]
    assert header["core_count"] == 40 and header["expected_tasks"] == 5
    done = [e for e in events if e["kind"] == 6]
    assert len(done) == 5 and sorted(e["value"] for e in done) == [1, 2, 3, 4, 5]
    for task in (1, 2, 3):
        for kind in (16, 17):
            actual = [e for e in events if e["kind"] == kind and e["task"] == task]
            assert sorted(e["value"] * 40 + e["block"] for e in actual) == list(range(160))
            assert {(e["core"], e["pe"]) for e in actual} == {(c, p) for c in range(40) for p in range(4)}
    for previous, following in ((1, 2), (2, 3)):
        completed = next(e["seq"] for e in done if e["task"] == previous)
        assert completed < min(e["seq"] for e in events if e["kind"] == 4 and e["task"] == following)
    for task, generations in ((1, 1), (3, 40)):
        for core in range(40):
            last_consumed = -1
            for generation in range(1, generations + 1):
                group = [
                    e for e in events if e["task"] == task and e["core"] == core and e.get("generation") == generation
                ]
                assert {e["slot"] for e in group} == {0 if task == 1 else 1}
                publish = [e for e in group if e["kind"] == 12]
                reads = [e for e in group if e["kind"] == 14]
                consumed = [e for e in group if e["kind"] == 15]
                assert len(publish) == 1 and publish[0]["pe"] == 0
                assert {e["pe"] for e in reads} == {0, 1, 2, 3} and len(reads) == 4
                assert {e["pe"] for e in consumed} == {0, 1, 2, 3} and len(consumed) == 4
                assert last_consumed < publish[0]["seq"] < min(e["seq"] for e in reads)
                for pe in range(4):
                    assert next(e["seq"] for e in reads if e["pe"] == pe) < next(
                        e["seq"] for e in consumed if e["pe"] == pe
                    )
                last_consumed = max(e["seq"] for e in consumed)
    return header["epoch"]


def test_ffn_independent_pe_graph_replay(tmp_path, st_platform):
    worker = Worker(level=2, platform=st_platform, runtime="host_build_graph", device_id=0)
    buffers = []
    try:
        worker.init()
        handle = worker.register(build_ffn(st_platform))
        epochs = []
        for seed in (0, 1):
            inputs, golden = ffn_inputs(seed)
            tensors = inputs + [torch.full_like(t, -12345) for t in golden] + [torch.full((160, 2), -1.0)]
            args = TaskArgs()
            for index, tensor in enumerate(tensors):
                buffer = layout_view(worker, tensor.nbytes, core=len(buffers) % 40)
                buffers.append(buffer)
                copy_to(worker, buffer, tensor)
                args.add_tensor(
                    buffer.tensor(shapes=tuple(tensor.shape), dtype=DataType.FLOAT32),
                    TensorArgType.INPUT if index < 3 else TensorArgType.OUTPUT_EXISTING,
                )
            config = CallConfig()
            config.output_prefix = str(tmp_path / str(seed))
            config.enable_dep_gen = True
            worker.run(handle, args, config)
            for index, expected in enumerate(golden, 3):
                copy_from(worker, tensors[index], buffers[index])
                torch.testing.assert_close(tensors[index], expected, rtol=0, atol=0)
            copy_from(worker, tensors[6], buffers[6])
            assert tensors[6].tolist() == [[b % 40, b // 40] for b in range(160)]
            epochs.append(audit_ffn(tmp_path / str(seed) / "wse_events.jsonl"))
            assert (tmp_path / str(seed) / "deps.json").is_file()
            for buffer in buffers:
                release_view(worker, buffer)
            buffers.clear()
        assert len(set(epochs)) == 2
        worker.unregister(handle)
    finally:
        worker.close()


def test_ffn_pending_runs_keep_separate_shared_state(tmp_path, st_platform):
    worker = Worker(level=2, platform=st_platform, runtime="host_build_graph", device_id=0)
    pending = []
    try:
        worker.init()
        handle = worker.register(build_ffn(st_platform))
        for seed in (3, 4):
            inputs, golden = ffn_inputs(seed)
            tensors = inputs + [torch.full_like(t, -12345) for t in golden] + [torch.full((160, 2), -1.0)]
            args = TaskArgs()
            buffers = []
            for index, tensor in enumerate(tensors):
                buffer = layout_view(worker, tensor.nbytes, core=len(buffers) % 40)
                buffers.append(buffer)
                copy_to(worker, buffer, tensor)
                args.add_tensor(
                    buffer.tensor(shapes=tuple(tensor.shape), dtype=DataType.FLOAT32),
                    TensorArgType.INPUT if index < 3 else TensorArgType.OUTPUT_EXISTING,
                )
            config = CallConfig()
            config.output_prefix = str(tmp_path / str(seed))
            run = worker.submit(handle, args, config)
            pending.append((seed, run, args, buffers, tensors, golden))
        epochs = []
        for seed, run, args, buffers, tensors, golden in reversed(pending):
            run.wait()
            for index, expected in enumerate(golden, 3):
                copy_from(worker, tensors[index], buffers[index])
                torch.testing.assert_close(tensors[index], expected, rtol=0, atol=0)
            epochs.append(audit_ffn(tmp_path / str(seed) / "wse_events.jsonl"))
        assert len(set(epochs)) == 2
        for _, _, _, buffers, _, _ in pending:
            for buffer in buffers:
                release_view(worker, buffer)
        worker.unregister(handle)
    finally:
        worker.close()
