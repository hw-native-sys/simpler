# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Freeze per-call diagnostics inside real framework queue callbacks.

The four invocations turn collection off, select two different detail/event
combinations, then turn it off again. Caller configs are changed only after
Worker.submit consumes them. Export retirement follows one framework boundary;
native admission and diagnostic collection may independently wait.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import threading
from pathlib import Path

import main as reference

_SELECTIONS = ((0, 0), (1, 2), (3, 4), (0, 0))
_UNPROVEN = []


def _submit_frame(frame, runs):
    config = frame["config"]
    frame["callback_thread"] = threading.get_native_id()
    runs.append(frame["worker"].submit(frame["callable"], frame["args"], config, caller_stream=frame["raw_stream"]))
    # The caller object is mutable; the accepted invocation owns its selection
    # and destination independently of these later changes.
    config.enable_chip_swimlane = 4
    config.enable_pmu = 1
    config.output_prefix = str(frame["poison"])


def _check_exports(root: Path, runtime: str):
    from simpler_setup.tools.swimlane_converter import read_perf_data  # noqa: PLC0415

    epochs = []
    for index, (level, event) in enumerate(_SELECTIONS):
        folder = root / str(index)
        swimlane, pmu = folder / "chip_swimlane_records.json", folder / "pmu.csv"
        if level == 0:
            artifacts = [
                path
                for pattern in ("chip_swimlane_records.json", "pmu.csv", "swimlane-*/records_e*.json")
                for path in folder.glob(pattern)
            ]
            assert not artifacts, (index, artifacts)
            continue
        data = read_perf_data(swimlane)
        assert data["runtime"] == runtime and data["chip_swimlane_level"] == level, (index, data)
        assert data["tasks"], (index, data)
        task_epochs = {int(task["run_epoch"]) for task in data["tasks"]}
        assert len(task_epochs) == 1 and 0 not in task_epochs, (index, task_epochs)
        if level >= 3:
            assert any(data.get("scheduler_records", [])), (index, data)
            assert {
                int(record["run_epoch"]) for stream in data["scheduler_records"] for record in stream
            } == task_epochs
        else:
            assert not data.get("scheduler_records"), (index, data)
        with pmu.open(newline="") as source:
            rows = list(csv.DictReader(source))
        assert rows and {int(row["event_type"]) for row in rows} == {event}, (index, rows)
        assert all(int(row["pmu_total_cycles"]) > 0 for row in rows), (index, rows)
        pmu_epochs = {int(row["run_epoch"]) for row in rows}
        # Both producers stamp get_platform_run_result_epoch(), not separate
        # collector-local counters.
        assert pmu_epochs == task_epochs, (index, pmu_epochs, task_epochs)
        epoch = next(iter(task_epochs))
        assert epoch not in epochs, (index, epoch, epochs)
        epochs.append(epoch)
    poison = root / "poison"
    assert not list(poison.rglob("*")), list(poison.rglob("*"))
    return epochs


def run(runtime: str, device: int, bridge_path: Path, root: Path, *, platform="a2a3"):
    import torch  # noqa: PLC0415
    import torch_npu  # noqa: PLC0415, F401 -- registers the framework backend
    from simpler.task_interface import CallConfig  # noqa: PLC0415
    from simpler.worker import Worker  # noqa: PLC0415

    root.mkdir(parents=True, exist_ok=False)
    poison = root / "poison"
    poison.mkdir()
    bridge = reference._load_bridge(bridge_path)
    torch.npu.set_device(device)
    stream = torch.npu.Stream(device=device)
    raw_stream = stream.npu_stream
    worker, handle = None, None
    frames, queued, runs = [], [], []
    retired = False
    try:
        worker = Worker(level=2, device_id=device, platform=platform, runtime=runtime)
        handle = worker.register(reference._compile(runtime, platform=platform))
        worker.init(borrowed_context_generation=1)
        frames, weak_tensors = reference._prepare_frames(torch, worker, handle, stream, raw_stream, len(_SELECTIONS))
        for index, frame in enumerate(frames):
            folder = root / str(index)
            folder.mkdir()
            config = CallConfig()
            config.enable_chip_swimlane, config.enable_pmu = _SELECTIONS[index]
            config.output_prefix = str(folder)
            frame.update(config=config, poison=poison)
        gc.collect()
        assert all(ref() is not None for ref in weak_tensors)
        caller_thread = threading.get_native_id()
        task_queue = os.environ.get("TASK_QUEUE_ENABLE", "1")
        with torch.npu.stream(stream):
            for index, frame in enumerate(frames):
                a, b, output = frame["tensors"]
                a.fill_(index + 1)
                b.fill_(index + 3)
                output.fill_(-999)
                queued.append(bridge.enqueue(lambda frame=frame: _submit_frame(frame, runs)))
                torch.add(output, output, out=frame["result"])
            bridge.finish_current_stream()
        for call in queued:
            call.check()
            assert (call.thread_id != caller_thread) == (task_queue == "1")
        assert len(runs) == len(_SELECTIONS)
        observed = []
        for index, frame in enumerate(frames):
            value = frame["result"].cpu()
            assert torch.equal(value, torch.full(reference.SHAPE, 8 + 4 * index, dtype=value.dtype))
            observed.append(value[0, 0].item())
        # Default collectors export on retirement. All successors are already
        # complete; this is not an extra wait inserted between submissions.
        for run_handle in runs:
            run_handle.wait()
        worker.close()
        for frame in frames:
            for buffer in frame["buffers"]:
                buffer.close()
        retired = True
        epochs = _check_exports(root, runtime)
        summary = dict(
            platform=platform,
            runtime=runtime,
            task_queue=task_queue,
            calls=len(runs),
            selections=_SELECTIONS,
            observed=observed,
            run_epochs=epochs,
            caller_thread=caller_thread,
            callback_threads=[call.thread_id for call in queued],
            caller_configs_mutated=all(
                frame["config"].enable_chip_swimlane == 4
                and frame["config"].enable_pmu == 1
                and frame["config"].output_prefix == str(poison)
                for frame in frames
            ),
            verified=True,
        )
        print("FRAMEWORK_DIAGNOSTIC_SELECTION_PASS " + json.dumps(summary, sort_keys=True), flush=True)
        return summary
    finally:
        if not retired:
            _UNPROVEN.append((worker, handle, frames, queued, runs, stream))
            try:
                os.write(2, b"FRAMEWORK_DIAGNOSTICS_UNPROVEN: caller owners retained after failure.\n")
            except OSError:
                pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", choices=["tensormap_and_ringbuffer", "host_build_graph"], required=True)
    parser.add_argument("--platform", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    options = parser.parse_args()
    modules = list(options.build_dir.glob("_simpler_framework_queue*.so"))
    if len(modules) > 1:
        parser.error("--build-dir contains multiple framework queue bridges")
    path = modules[0] if modules else reference.build_bridge(options.build_dir)
    run(options.runtime, options.device, path, options.output_dir, platform=options.platform)


if __name__ == "__main__":
    main()
