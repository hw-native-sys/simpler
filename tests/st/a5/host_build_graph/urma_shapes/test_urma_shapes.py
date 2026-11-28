# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Real URMA submissions from SPMD, MIX and predicated Graph calendar rows."""

import json
from copy import deepcopy
from pathlib import Path

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, scene_test
from simpler_setup.tools.hbg.task_id import TaskId
from tests.st.a5.host_build_graph.urma_device.test_urma_device import _HbgUrmaDevice
from tests.st.a5.tensormap_and_ringbuffer.urma_deferred_completion.test_urma_deferred_completion import _case


def _spec(function):
    spec = deepcopy(_HbgUrmaDevice.CALLABLE)
    chip = spec["callables"][0]
    kernels = Path(__file__).parent / "kernels"
    chip["orchestration"].update(source=str(kernels / "shapes.cpp"), function_name=function)
    get = dict(source=str(kernels / "transfer.cpp"), core_type="aiv", signature=[D.IN, D.IN, D.INOUT] + [D.IN] * 4)
    chip["incores"][0] = dict(func_id=0, **get)
    cube = dict(func_id=2, source=str(kernels / "notifier.cpp"), core_type="aic", signature=get["signature"])
    if function == "predicated_graph":
        gate = kernels.parents[2] / "host_build_graph/predicated_dispatch/kernels/aic/kernel_write_gate.cpp"
        chip["incores"].append(dict(func_id=3, source=str(gate), core_type="aic", signature=[D.OUT]))
    elif function == "graph_mix":
        chip["incores"].extend([cube, dict(func_id=4, **get)])
    elif function.startswith("pending_gets"):
        chip["incores"].append(cube)
        if function == "pending_gets_aiv":
            chip["incores"].append(dict(cube, func_id=4, core_type="aiv"))
    return spec


class _ShapeCase(_HbgUrmaDevice):
    SLOTS = 4
    PREDICATED = False
    MIX = False
    PROGRESS = False
    AIV_NOTIFIER = False

    def _segments(self, tensor):
        return (tensor[tensor.numel() // 2 :] if self.PREDICATED else tensor).view(self.SLOTS, -1)

    def compute_golden(self, args, params):
        for rank in (0, 1):
            out = getattr(args, f"out_{rank}")
            local = getattr(args, f"in_{rank}")
            out.copy_(getattr(args, f"in_{1 - rank}"))
            if self.PREDICATED:
                half = out.numel() // 2
                out[:half].copy_(local[:half] + 1)  # The first Graph's predicated GET is skipped.
            segments = self._segments(out)
            segments[:, :32].fill_(-77)
            segments[:, -32:].fill_(-77)
            if self.PROGRESS:
                segments[0, -32:-29] = torch.tensor([self.SLOTS, 0, 1], dtype=out.dtype)
            getattr(args, f"result_{rank}").copy_(out + 1)

    def compare_outputs(self, actual, golden, output_names, params):
        for rank in (0, 1):
            name = f"out_{rank}"
            segments = self._segments(getattr(actual, name))
            words = segments[:, :10].to(torch.int64)
            sender, row, bb, cqe, block, lane, worker, rows_per_peer, jetties, valid = words.T
            assert (valid == 1).all() and (jetties == 1).all()
            assert ((0 <= sender) & (sender < rows_per_peer)).all()
            assert torch.equal(row, (1 - rank) * rows_per_peer + sender)
            assert len(set((worker - sender).tolist())) == 1 and (worker > sender).all()
            assert (bb > 0).all() and (cqe > 0).all()
            if self.MIX:
                assert block.tolist() == [0, 0, 1, 1] and lane.tolist() == [0, 1, 0, 1]
                assert (sender[1::2] == sender[::2] + 1).all()
            else:
                assert block.tolist() == list(range(self.SLOTS))
            if self.PROGRESS:
                assert 1 < len(set(sender.tolist())) < self.SLOTS
                for owner in set(sender.tolist()):
                    for target in (bb, cqe):
                        own = target[sender == owner].tolist()
                        assert len(set(own)) == len(own)
                assert segments[0, -32:-29].tolist() == [self.SLOTS, 0, 1]
                if self.AIV_NOTIFIER:
                    notifier = int(segments[0, -16])
                    assert 1 <= notifier <= rows_per_peer[0]
                    self._segments(getattr(golden, name))[0, -16] = notifier
                    self._segments(getattr(golden, f"result_{rank}"))[0, -16] = notifier + 1
            else:
                assert len(set(sender.tolist())) == self.SLOTS
            result = self._segments(getattr(actual, f"result_{rank}"))
            assert torch.equal(result[:, :10], segments[:, :10] + 1)
            # All dynamic header fields have been checked, including their consumer's copy.
            self._segments(getattr(golden, name))[:, :10].copy_(segments[:, :10])
            self._segments(getattr(golden, f"result_{rank}"))[:, :10].copy_(result[:, :10])
            print(
                f"URMA rank={rank} senders={len(set(sender.tolist()))} groups={self.SLOTS} "
                f"first [sender,row,BB,CQE]={words[:4, :4].tolist()}",
                flush=True,
            )
        super().compare_outputs(actual, golden, output_names, params)

    def _run_and_validate(self, *args, **kwargs):
        super()._run_and_validate(*args, **kwargs)
        if kwargs.get("enable_chip_swimlane", 0) < 2:
            return
        paths = sorted(Path(kwargs["output_prefix"]).glob("rank*/d*/chip_swimlane_records.json"))
        assert len(paths) == 2
        for path in paths:
            events = json.loads(path.read_text())["calendar_task_events"]
            assert events["dropped"] == 0
            records = events["records"]
            submitted = [e for e in records if e["kind"] == "defer_registered"]
            settled = [e for e in records if e["kind"] == "deferred_completion_issue"]
            ready = [e for e in records if e["kind"] == "defer_signal_ready"]
            refs = {e["deferred_group"] for e in submitted}
            assert len(submitted) == len(settled) == len(ready) == len(refs) == self.SLOTS
            assert refs == {e["deferred_group"] for e in settled} == {e["deferred_group"] for e in ready}
            assert all(e["contribution"] == 1 for e in settled)
            kernels = [e for e in records if e["kind"] == "kernel_end"]
            if self.PREDICATED:
                # Only the consumer of shell 3 needs the active GET. Shell 1 skips its predicated body row.
                skipped_id = TaskId.make_sub_task(1, 1).raw
                source = TaskId.make_sub_task(3, 1).raw
                skipped = [e for e in records if e["task_id"] == skipped_id and e["kind"] == "completion_issue"]
                assert len(skipped) == 1 and skipped[0]["contribution"] == 1
                assert all(e["task_id"] != skipped_id for e in kernels)
                early = [e for e in kernels if e["task_id"] == 2]
                assert len(early) == 1 and skipped[0]["cycles"] <= early[0]["kernel_start_cycles"]
                consumers = (4,)
            elif self.MIX:
                source = TaskId.make_sub_task(0, 0).raw
                roles = sorted((e["block_index"], e["cluster_lane"]) for e in kernels if e["task_id"] == source)
                assert roles == [(b, lane) for b in range(2) for lane in range(3)]
                assert sorted((e["block_index"], e["cluster_lane"]) for e in submitted) == [
                    (b, lane) for b in range(2) for lane in (1, 2)
                ]
                consumers = (1,)
            else:
                source = 0
                consumers = range(1, self.SLOTS + 1) if self.PROGRESS else (1,)
            assert {e["task_id"] for e in submitted} == {source}
            dependent = [e for e in kernels if e["task_id"] in consumers]
            assert sorted(e["task_id"] for e in dependent) == list(consumers)
            assert max(e["cycles"] for e in ready) <= min(e["kernel_start_cycles"] for e in dependent)
            if self.AIV_NOTIFIER:
                probe, release = (
                    next(e for e in records if e["task_id"] == t and e["kind"] == "kernel_end")
                    for t in (self.SLOTS + 1, self.SLOTS + 2)
                )
                assert probe["cycles"] <= release["kernel_start_cycles"]
                lane = release["worker_id"]
                begin = next(
                    e["cycles"] for e in records if e["kind"] == "row_begin" and e["task_id"] == release["task_id"]
                )
                previous = max(
                    (
                        e["cycles"]
                        for e in records
                        if e["worker_id"] == lane and e["kind"] == "local_retired" and e["cycles"] <= begin
                    ),
                    default=0,
                )
                scans = [e for e in events["range_records"] if e["kind"] == "ready_row_scan" and e["worker_id"] == lane]
                assert any(previous <= e["scan_start_cycles"] < e["cycles"] <= begin for e in scans)
                assert all(
                    e["cycles"] <= release["kernel_start_cycles"] or e["scan_start_cycles"] >= release["cycles"]
                    for e in scans
                )
                cohort = [e for e in kernels if e["task_id"] == self.SLOTS + 3]
                assert sorted(e["block_index"] for e in cohort) == [0, 1]
                assert len({e["worker_id"] for e in cohort}) == 2
                assert max(e["finish_cycles"] for e in dependent) <= min(e["kernel_start_cycles"] for e in cohort)


@scene_test(level=3, runtime="host_build_graph")
class TestUrmaSpmd(_ShapeCase):
    CALLABLE = _spec("spmd_get")
    CASES = _case("sync_start")


@scene_test(level=3, runtime="host_build_graph")
class TestUrmaGraphPredicate(_ShapeCase):
    SLOTS = 1
    PREDICATED = True
    CALLABLE = _spec("predicated_graph")
    CASES = _case("false_and_true_sinks", elems=65536)


@scene_test(level=3, runtime="host_build_graph")
class TestUrmaGraphMix(_ShapeCase):
    MIX = True
    CALLABLE = _spec("graph_mix")
    CASES = _case("sync_start_mix_shell", elems=65536)


@scene_test(level=3, runtime="host_build_graph")
class TestUrmaPendingProgress(_ShapeCase):
    SLOTS = 128
    PROGRESS = True
    CALLABLE = _spec("pending_gets")
    CASES = _case("late_notifier", elems=1 << 20)


@scene_test(level=3, runtime="host_build_graph")
class TestUrmaPendingAivProgress(_ShapeCase):
    SLOTS = 128
    PROGRESS = True
    AIV_NOTIFIER = True
    CALLABLE = _spec("pending_gets_aiv")
    CASES = _case("late_aiv_with_predecessor", elems=1 << 20)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
