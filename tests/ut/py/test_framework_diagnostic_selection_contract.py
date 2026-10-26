# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Actual consumer/checker CPU barriers; no Torch-NPU or RTS execution is emulated.

The recording submit boundary freezes values and writes synthetic artifacts only
when the real consumer retires its handles. The canonical swimlane reader still
decodes every fixture. Device ordering and real collector output belong to ST.
"""

import csv
import gc
import importlib.util
import json
import sys
import threading
import weakref
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
from _capture_builder import Capture, legal_phase_names
from _task_ids import hbg_sub_task, tmr_task

from simpler_setup.tools.swimlane_converter import read_perf_data

_ROOT = Path(__file__).resolve().parents[3]
_CONSUMER = _ROOT / "examples/workers/l2/framework_eager/diagnostic_selection.py"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def consumer(monkeypatch):
    monkeypatch.setitem(sys.modules, "main", _load(_CONSUMER.with_name("main.py"), "diagnostic_reference"))
    return _load(_CONSUMER, "diagnostic_consumer")


def _write(folder, runtime, level, event, epoch):
    if not level:
        return
    task = hbg_sub_task(1, 0) if runtime == "host_build_graph" else tmr_task(1, 0)
    capture = Capture(runtime=runtime, level=level)
    if level == 1:
        capture.aicore_task(task_id=task, start=120, end=180, run_epoch=epoch)
    else:
        capture.task(task_id=task, start=120, end=180, run_epoch=epoch)
    if level >= 3:
        capture.sched_phase(phase=sorted(legal_phase_names(runtime, "aicpu"))[0], start=100, end=200, run_epoch=epoch)
    (folder / "chip_swimlane_records.json").write_text(json.dumps(capture.build()))
    with (folder / "pmu.csv").open("w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["event_type", "run_epoch", "pmu_total_cycles"])
        writer.writerow([event, epoch, 123])


@pytest.fixture
def exports(tmp_path, consumer):
    (tmp_path / "poison").mkdir()
    for index, (level, event) in enumerate(consumer._SELECTIONS):
        folder = tmp_path / str(index)
        folder.mkdir()
        _write(folder, "host_build_graph", level, event, index + 10)
    return tmp_path


def test_canonical_reader_accepts_per_call_selections(consumer, exports):
    assert consumer._check_exports(exports, "host_build_graph") == [11, 12]


def test_off_call_rejects_retained_swimlane_form_without_fixed_files(consumer, exports):
    retained = exports / "3/swimlane-0"
    retained.mkdir()
    capture = Capture(runtime="host_build_graph", level=1).aicore_task(
        task_id=hbg_sub_task(1, 0), start=120, end=180, run_epoch=13
    )
    artifact = retained / "records_e13.json"
    artifact.write_text(json.dumps(capture.build()))
    assert {task["run_epoch"] for task in read_perf_data(artifact)["tasks"]} == {13}
    assert not (exports / "3/chip_swimlane_records.json").exists()
    assert not (exports / "3/pmu.csv").exists()
    with pytest.raises(AssertionError):
        consumer._check_exports(exports, "host_build_graph")


def test_off_calls_allow_host_logs_without_diagnostic_records(consumer, exports):
    for index in (0, 3):
        (exports / str(index) / "host.123.log").write_text("Host completion log\n")
    assert consumer._check_exports(exports, "host_build_graph") == [11, 12]


@pytest.mark.parametrize(
    "defect",
    ["stale_level", "stale_event", "duplicate_epoch", "mixed_epoch", "stale_phase", "off_record", "poison", "missing"],
)
def test_actual_checker_rejects_misattribution(consumer, exports, defect):
    folder = exports / "2"
    if defect == "stale_level":
        _write(folder, "host_build_graph", 1, 4, 12)
    elif defect == "stale_event":
        _write(folder, "host_build_graph", 3, 2, 12)
    elif defect == "duplicate_epoch":
        _write(folder, "host_build_graph", 3, 4, 11)
    elif defect == "mixed_epoch":
        with (folder / "pmu.csv").open("a") as out:
            out.write("4,11,123\n")
    elif defect == "stale_phase":
        path = folder / "chip_swimlane_records.json"
        data = json.loads(path.read_text())
        data["scheduler_records"]["streams"][0]["records"][0]["run_epoch"] = 11
        path.write_text(json.dumps(data))
    elif defect == "off_record":
        _write(exports / "3", "host_build_graph", 1, 2, 13)
    elif defect == "poison":
        (exports / "poison/pmu.csv").write_text("late export")
    elif defect == "missing":
        (folder / "pmu.csv").unlink()
    with pytest.raises((AssertionError, FileNotFoundError)):
        consumer._check_exports(exports, "host_build_graph")


@pytest.fixture
def transport(consumer, monkeypatch):
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    monkeypatch.setenv("TASK_QUEUE_ENABLE", "0")
    import torch  # noqa: PLC0415 -- disable backend autoload before importing CPU Torch

    events, snapshots, refs, workers = [], [], [], []
    state = SimpleNamespace(fail_boundary=False, fail_retire=False)
    error = RuntimeError("completion is unknown")
    stream = SimpleNamespace(npu_stream=17)

    class Run:
        def __init__(self, selection, epoch):
            self.selection, self.epoch = selection, epoch

        def wait(self):
            assert events.count("submit") == 4 and "framework boundary" in events
            if state.fail_retire:
                raise error
            events.append("retire")
            folder, runtime, level, event = self.selection
            _write(folder, runtime, level, event, self.epoch)

    class Worker:
        closed = False

        def __init__(self, *, runtime, **kwargs):
            self.runtime = runtime
            workers.append(self)

        def register(self, spec):
            return "callable"

        def init(self, **kwargs):
            assert kwargs == {"borrowed_context_generation": 1}

        def submit(self, handle, args, config, *, caller_stream):
            assert handle == "callable" and caller_stream == 17 and not self.closed
            selection = (Path(config.output_prefix), self.runtime, config.enable_chip_swimlane, config.enable_pmu)
            snapshots.append(selection)
            events.append("submit")
            torch.add(args[0], args[1], out=args[2])
            run = Run(selection, len(snapshots))
            refs.append(weakref.ref(run))
            return run

        def close(self):
            assert events.count("retire") == 4 and all(ref() is not None for ref in refs)
            events.append("close")
            self.closed = True

    class Buffer:
        closed = False

        def close(self):
            assert workers[0].closed
            self.closed = True
            events.append("buffer close")

    def prepare(torch, worker, handle, actual_stream, raw_stream, calls):
        assert actual_stream is stream and raw_stream == 17 and calls == 4
        frames, weak_tensors = [], []
        for index in range(calls):
            tensors = tuple(torch.empty(consumer.reference.SHAPE) for _ in range(3))
            weak_tensors.extend(weakref.ref(tensor) for tensor in tensors)
            frames.append(
                dict(
                    index=index,
                    tensors=tensors,
                    args=tensors,
                    buffers=[Buffer() for _ in tensors],
                    worker=worker,
                    callable=handle,
                    raw_stream=raw_stream,
                    result=torch.empty(consumer.reference.SHAPE),
                )
            )
        return frames, weak_tensors

    def enqueue(callback):
        callback()
        return SimpleNamespace(check=lambda: None, thread_id=threading.get_native_id())

    def finish():
        events.append("framework boundary")
        if state.fail_boundary:
            raise error

    bridge = SimpleNamespace(enqueue=enqueue, finish_current_stream=finish)
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            set_device=lambda device: None, Stream=lambda **kwargs: stream, stream=lambda actual: nullcontext()
        ),
        raising=False,
    )
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "simpler.worker", SimpleNamespace(Worker=Worker))
    monkeypatch.setattr(consumer.reference, "_compile", lambda runtime: "compiled")
    monkeypatch.setattr(consumer.reference, "_load_bridge", lambda path: bridge)
    monkeypatch.setattr(consumer.reference, "_prepare_frames", prepare)
    return SimpleNamespace(events=events, snapshots=snapshots, state=state, error=error, workers=workers, refs=refs)


@pytest.mark.parametrize("runtime", ["host_build_graph", "tensormap_and_ringbuffer"])
def test_whole_consumer_freezes_config_and_retires_after_final_boundary(consumer, transport, tmp_path, runtime):
    summary = consumer.run(runtime, 0, Path("recording.so"), tmp_path / "diagnostics")
    assert summary["observed"] == [8, 12, 16, 20] and summary["run_epochs"] == [2, 3]
    assert summary["caller_configs_mutated"]
    assert [(level, event) for _, _, level, event in transport.snapshots] == [(0, 0), (1, 2), (3, 4), (0, 0)]
    assert (
        transport.events == ["submit"] * 4 + ["framework boundary"] + ["retire"] * 4 + ["close"] + ["buffer close"] * 12
    )
    assert not consumer._UNPROVEN


@pytest.mark.parametrize("failure", ["fail_boundary", "fail_retire"])
def test_whole_consumer_unknown_completion_keeps_owners(consumer, transport, tmp_path, failure):
    setattr(transport.state, failure, True)
    with pytest.raises(RuntimeError) as caught:
        consumer.run("host_build_graph", 0, Path("recording.so"), tmp_path / "diagnostics")
    assert caught.value is transport.error
    gc.collect()
    worker, _, frames, queued, runs, _ = consumer._UNPROVEN[0]
    assert worker is transport.workers[0] and not worker.closed
    assert len(frames) == len(queued) == len(runs) == 4
    assert all(ref() is run for ref, run in zip(transport.refs, runs))
    assert all(not buffer.closed for frame in frames for buffer in frame["buffers"])


@pytest.mark.parametrize("defect", [None, "wrong_value", "wrong_selection", "mixed_epoch", "missing_mutation"])
def test_real_st_rejects_incomplete_summary(tmp_path, monkeypatch, defect):
    st = _load(_ROOT / "tests/st/a2a3/framework_eager/test_diagnostic_selection.py", "diagnostic_st")
    summary = dict(
        runtime="host_build_graph",
        task_queue="1",
        verified=True,
        observed=[8, 12, 16, 20],
        calls=4,
        selections=[[0, 0], [1, 2], [3, 4], [0, 0]],
        caller_configs_mutated=True,
        run_epochs=[2, 3],
        caller_thread=1,
        callback_threads=[2] * 4,
    )
    if defect == "wrong_value":
        summary["observed"] = [8, 12, 16, -1998]
    elif defect == "wrong_selection":
        summary["selections"] = [[0, 0], [1, 2], [1, 2], [0, 0]]
    elif defect == "mixed_epoch":
        summary["run_epochs"] = [2, 2]
    elif defect == "missing_mutation":
        summary["caller_configs_mutated"] = False
    result = SimpleNamespace(
        returncode=0, stdout="FRAMEWORK_DIAGNOSTIC_SELECTION_PASS " + json.dumps(summary), stderr=""
    )

    def run(*args, **kwargs):
        assert kwargs["env"]["MAX_JOBS"] == "2"
        return result

    monkeypatch.setattr(st, "subprocess", SimpleNamespace(run=run))
    request = SimpleNamespace(config=SimpleNamespace(getoption=lambda name: "0"))
    if defect is None:
        st.test_framework_diagnostic_selection("host_build_graph", "1", tmp_path, tmp_path, request)
    else:
        with pytest.raises(AssertionError):
            st.test_framework_diagnostic_selection("host_build_graph", "1", tmp_path, tmp_path, request)
