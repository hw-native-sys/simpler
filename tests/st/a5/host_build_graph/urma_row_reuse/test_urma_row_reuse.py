# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import json
from copy import deepcopy
from pathlib import Path

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, scene_test
from tests.st.a5.host_build_graph.urma_device.test_urma_device import _HbgUrmaDevice
from tests.st.a5.tensormap_and_ringbuffer.urma_deferred_completion.test_urma_deferred_completion import _case


def _callable(entry):
    spec = deepcopy(_HbgUrmaDevice.CALLABLE)
    chip = spec["callables"][0]
    kernels = Path(__file__).parent / "kernels"
    chip["orchestration"].update(source=str(kernels / "row_reuse.cpp"), function_name=entry)
    chip["incores"] = chip["incores"][:1]  # Use the guarded Device GET kernel from urma_device.
    if entry == "reuse_row":
        for func_id, core in ((1, "aiv"), (2, "aic")):
            chip["incores"].append(
                {
                    "func_id": func_id,
                    "source": str(kernels / "observe_row.cpp"),
                    "core_type": core,
                    "signature": [D.INOUT, D.IN, D.IN, D.IN],
                }
            )
    return spec


def _check_samples(output, rank):
    names = ["phase", "worker", "sender_plus_one", "rows_per_peer", "registered", "settled"]
    for group in ("older", "newer"):
        names.extend(f"{group}_{field}" for field in ("ref", "row", "bb", "cqe", "claimed", "retired", "pending"))
    names.extend(("sq_tail", "cq_tail", "cq_doorbell"))
    words = output.view(torch.int32)
    offsets = (0, 32, 64)
    assert words.numel() >= offsets[-1] + len(names)
    samples = [dict(zip(names, words[offset : offset + len(names)].tolist())) for offset in offsets]
    print(f"URMA row snapshots rank {rank}: {json.dumps(samples)}", flush=True)
    before, newer, older = samples
    assert [s["phase"] for s in samples] == [1, 2, 3]
    assert [s["registered"] for s in samples] == [1, 2, 2]
    assert [s["settled"] for s in samples] == [0, 1, 2]
    assert newer["worker"] != older["worker"]
    row = (1 - rank) * before["rows_per_peer"] + before["sender_plus_one"] - 1
    assert before["sender_plus_one"] > 0 and before["older_ref"] > 0
    for key in ("sender_plus_one", "rows_per_peer", "older_ref", "older_row", "older_bb", "older_cqe"):
        assert before[key] == newer[key] == older[key], key
    assert before["older_row"] == newer["newer_row"] == older["newer_row"] == row
    assert older["newer_ref"] == newer["newer_ref"] > 0 and older["newer_ref"] != older["older_ref"]
    assert [s["older_claimed"] for s in samples] == [0, 0, 1]
    assert [s["older_retired"] for s in samples] == [0, 0, 1]
    assert [s["older_pending"] for s in samples] == [1, 1, 0]
    for s in (newer, older):
        assert s["newer_retired"] == 1 and s["newer_pending"] == 0
        assert before["sq_tail"] < s["older_bb"] < s["newer_bb"] == s["sq_tail"]
        assert before["cq_tail"] + 1 == s["older_cqe"] == s["newer_cqe"] - 1
        assert s["cq_tail"] == s["newer_cqe"] and s["cq_doorbell"] == (s["cq_tail"] & 0xFFFFFF)
    assert before["cq_doorbell"] == (before["cq_tail"] & 0xFFFFFF)
    for key in ("sq_tail", "cq_tail", "cq_doorbell"):
        assert newer[key] == older[key], key


def _check_events(path, terminal):
    events = json.loads(path.read_text())["calendar_task_events"]
    assert events["dropped"] == 0
    records = events["records"]
    registered = sorted((e for e in records if e["kind"] == "defer_registered"), key=lambda e: e["cycles"])
    assert len(registered) == (1 if terminal else 2)
    completions = []
    for event in registered:
        task = [e for e in records if e["task_id"] == event["task_id"]]
        assert not any(e["kind"] == "completion_issue" for e in task)
        (poll,) = (e for e in task if e["kind"] == "defer_poll_begin")
        (done,) = (e for e in task if e["kind"] == "deferred_completion_issue")
        assert event["cycles"] <= poll["cycles"] <= done["cycles"]
        assert event["deferred_group"] == poll["deferred_group"] == done["deferred_group"]
        assert event["worker_id"] == poll["source_worker_id"] == done["source_worker_id"]
        assert poll["worker_id"] == done["worker_id"] and done["contribution"] == 1
        completions.append((poll, done))
    if terminal:
        assert {e["task_id"] for e in records if e["kind"] == "kernel_end"} == {registered[0]["task_id"]}
    else:
        first, second = registered
        assert first["task_id"] != second["task_id"] and first["worker_id"] == second["worker_id"]
        assert first["cluster_lane"] in (1, 2)
        (old_poll, _), (new_poll, new_done) = completions
        assert new_poll["cluster_lane"] == 0 and old_poll["cluster_lane"] in (1, 2)
        assert new_done["cycles"] <= old_poll["cycles"]


class _RowCase(_HbgUrmaDevice):
    TERMINAL = False

    def compute_golden(self, args, params):
        for rank in (0, 1):
            out = getattr(args, f"out_{rank}")
            out.copy_(getattr(args, f"in_{1 - rank}"))
            for part in out.chunk(1 if self.TERMINAL else 2):
                part[:32].fill_(-77.0)
                part[-32:].fill_(-77.0)

    def compare_outputs(self, test_args, golden_args, output_names, params):
        if not self.TERMINAL:
            for rank in (0, 1):
                name = f"result_{rank}"
                _check_samples(getattr(test_args, name), rank)
            output_names = [name for name in output_names if name not in ("result_0", "result_1")]
        super().compare_outputs(test_args, golden_args, output_names, params)

    def _run_and_validate(self, *args, **kwargs):
        super()._run_and_validate(*args, **kwargs)
        if kwargs.get("enable_chip_swimlane", 0) >= 2:
            paths = sorted(Path(kwargs["output_prefix"]).glob("rank*/d*/chip_swimlane_records.json"))
            assert [p.parent.parent.name for p in paths] == ["rank0", "rank1"]
            for path in paths:
                _check_events(path, self.TERMINAL)


@scene_test(level=3, runtime="host_build_graph")
class TestUrmaRowReuse(_RowCase):
    CALLABLE = _callable("reuse_row")
    CASES = _case("newer_before_older")


@scene_test(level=3, runtime="host_build_graph")
class TestUrmaTerminalGet(_RowCase):
    TERMINAL = True
    CALLABLE = _callable("terminal_get")
    CASES = _case("phase_exit_get")


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
