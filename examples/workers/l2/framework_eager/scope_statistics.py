# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Per-call ScopeStats attribution through the actual Torch emission queue.

TMR records the root and two nested user scopes. HBG has no device scope
producer and exports metadata only. This consumer measures neither overlap nor
performance; ordinary diagnostic retirement may apply native backpressure.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import threading
from pathlib import Path

import main as reference

_SELECTIONS = (False, True, True, False)
_KERNEL = Path(__file__).with_name("kernels") / "scope_statistics_orchestration.cpp"
_UNPROVEN = []


def _compile(runtime, platform="a2a3"):
    from simpler.task_interface import ArgDirection  # noqa: PLC0415

    from simpler_setup.scene_test import compile_chip_callable_spec  # noqa: PLC0415

    directions = [ArgDirection.IN, ArgDirection.IN, ArgDirection.OUT]
    spec = {
        "orchestration": {
            "source": str(_KERNEL),
            "function_name": "framework_scope_statistics",
            "config_name": "framework_scope_statistics_config",
            "signature": directions,
        },
        "incores": [
            {
                "func_id": 0,
                "source": str(reference.HERE.parent / "vector_add/kernels/aiv/vector_add_kernel.cpp"),
                "core_type": "aiv",
                "signature": directions,
            }
        ],
    }
    return compile_chip_callable_spec(spec, platform, runtime, f"framework-scope-statistics-{platform}-{runtime}")


def _scope_sites():
    # ScopeStatsRecord stores the basename in a 32-byte NUL-terminated field.
    lines = [
        number
        for number, line in enumerate(_KERNEL.read_text().splitlines(), 1)
        if line.strip().startswith("SIMPLER_SCOPE()")
    ]
    assert len(lines) == 2, lines
    return [f"{_KERNEL.name[:31]}:{number}" for number in lines]


def _check_scope_artifact(path: Path, runtime: str):
    metadata, *records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert metadata["fatal"] is False and metadata["dropped"] == 0, metadata
    assert metadata["total"] == len(records), (metadata, records)
    assert not metadata.get("counts_unknown", False), metadata
    assert metadata.get("collection_verdict", "published") == "published", metadata
    if runtime == "host_build_graph":
        assert not records, records
        return None
    assert runtime == "tensormap_and_ringbuffer"
    assert len(records) == 6, records
    outer, inner = _scope_sites()
    expected = [
        ("(unknown):0", "begin", 0, 0),
        (outer, "begin", 1, 1),
        (inner, "begin", 2, 2),
        (inner, "end", 2, 2),
        (outer, "end", 1, 1),
        ("(unknown):0", "end", 0, 0),
    ]
    assert [(r["site"], r["phase"], r["depth"], r["ring"]) for r in records] == expected, records
    epochs = {r["run_epoch"] for r in records}
    assert len(epochs) == 1 and type(next(iter(epochs))) is int and next(iter(epochs)) > 0, epochs
    for key in ("task_window_max", "heap_max", "dep_pool_max"):
        assert len(metadata[key]) == 4 and all(type(n) is int and n > 0 for n in metadata[key]), metadata
    assert metadata["tensormap_max"] > 0, metadata
    for record in records:
        for resource in ("task_window", "heap", "dep_pool"):
            assert 0 <= record[resource + "_start"] <= record[resource + "_end"], record
        assert 0 <= record["tensormap"] <= metadata["tensormap_max"], record
    # These are cumulative allocation frontiers, not asynchronous live counts.
    # Each user scope submits one task; the outer also allocates its temporary.
    for begin, end in ((records[1], records[4]), (records[2], records[3])):
        assert end["task_window_end"] > begin["task_window_end"], (begin, end)
    assert records[4]["heap_end"] > records[1]["heap_end"], records
    return next(iter(epochs))


def _check_exports(root: Path, runtime: str):
    epochs = []
    for index, enabled in enumerate(_SELECTIONS):
        folder = root / str(index)
        artifacts = list(folder.rglob("scope_stats.jsonl"))
        if not enabled:
            assert not artifacts, (index, artifacts)
            continue
        expected = folder / "scope_stats/scope_stats.jsonl"
        assert artifacts == [expected], (index, artifacts)
        epoch = _check_scope_artifact(expected, runtime)
        if epoch is not None:
            assert epoch not in epochs, (index, epoch, epochs)
            epochs.append(epoch)
    assert not list((root / "poison").rglob("scope_stats.jsonl")), "post-submit prefix mutation affected export"
    return epochs


def _submit_frame(frame, runs):
    config = frame["config"]
    frame["callback_thread"] = threading.get_native_id()
    runs.append(frame["worker"].submit(frame["callable"], frame["args"], config, caller_stream=frame["raw_stream"]))
    config.enable_scope_stats = not _SELECTIONS[frame["index"]]
    config.output_prefix = str(frame["poison"])


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
        handle = worker.register(_compile(runtime, platform=platform))
        worker.init(borrowed_context_generation=1)
        frames, references = reference._prepare_frames(torch, worker, handle, stream, raw_stream, len(_SELECTIONS))
        for index, frame in enumerate(frames):
            folder = root / str(index)
            folder.mkdir()
            config = CallConfig()
            config.enable_scope_stats = _SELECTIONS[index]
            config.output_prefix = str(folder)
            frame.update(config=config, poison=poison)
        gc.collect()
        assert all(ref() is not None for ref in references)
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
            assert torch.equal(value, torch.full(reference.SHAPE, 10 + 6 * index, dtype=value.dtype))
            observed.append(value[0, 0].item())
        for run_handle in runs:
            run_handle.wait()
        epochs = _check_exports(root, runtime)
        worker.close()
        for frame in frames:
            for buffer in frame["buffers"]:
                buffer.close()
        retired = True
        summary = dict(
            platform=platform,
            runtime=runtime,
            task_queue=task_queue,
            calls=len(runs),
            selections=list(_SELECTIONS),
            observed=observed,
            run_epochs=epochs,
            user_scope_sites=_scope_sites(),
            metadata_only=runtime == "host_build_graph",
            caller_thread=caller_thread,
            callback_threads=[call.thread_id for call in queued],
            caller_configs_mutated=all(
                frame["config"].enable_scope_stats != _SELECTIONS[frame["index"]]
                and frame["config"].output_prefix == str(poison)
                for frame in frames
            ),
            caller_tensors_retained=all(ref() is not None for ref in references),
            verified=True,
        )
        print("FRAMEWORK_SCOPE_STATISTICS_PASS " + json.dumps(summary, sort_keys=True), flush=True)
        return summary
    finally:
        if not retired:
            _UNPROVEN.append((worker, handle, frames, queued, runs, stream))
            try:
                os.write(2, b"FRAMEWORK_SCOPE_STATISTICS_UNPROVEN: caller/runtime owners retained.\n")
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
    bridge = modules[0] if modules else reference.build_bridge(options.build_dir)
    run(options.runtime, options.device, bridge, options.output_dir, platform=options.platform)


if __name__ == "__main__":
    main()
