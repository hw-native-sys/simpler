# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CPU barriers for the real ScopeStats consumer and its artifact checker.

The recording transport uses CPU tensors and writes controlled schema fixtures at
retirement. It proves caller bookkeeping/checking, not RTS execution, Torch-NPU
ordering or real ScopeStats collection; those require the marked scene cases.
"""

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

_ROOT = Path(__file__).resolve().parents[3]
_CONSUMER = _ROOT / "examples/workers/l2/framework_eager/scope_statistics.py"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def consumer(monkeypatch):
    monkeypatch.setitem(sys.modules, "main", _load(_CONSUMER.with_name("main.py"), "scope_reference"))
    return _load(_CONSUMER, "scope_consumer")


def _fixture_records():
    # The fixture computes sites from the C++ input, never from exported records.
    source = _CONSUMER.with_name("kernels") / "scope_statistics_orchestration.cpp"
    sites = [
        f"{source.name[:31]}:{i}"
        for i, line in enumerate(source.read_text().splitlines(), 1)
        if line.strip().startswith("SIMPLER_SCOPE()")
    ]
    outer, inner = sites
    identities = [
        ("(unknown):0", "begin", 0),
        (outer, "begin", 1),
        (inner, "begin", 2),
        (inner, "end", 2),
        (outer, "end", 1),
        ("(unknown):0", "end", 0),
    ]
    records = []
    for index, (site, phase, depth) in enumerate(identities):
        allocated = int(phase == "end" and depth > 0)
        records.append(
            dict(
                site=site,
                phase=phase,
                depth=depth,
                ring=depth,
                task_window_start=0,
                task_window_end=allocated,
                heap_start=0,
                heap_end=65536 * allocated,
                dep_pool_start=0,
                dep_pool_end=0,
                tensormap=0,
                run_epoch=1,
                buf_seq=0,
            )
        )
    return records


def _write(folder, runtime, epoch):
    path = folder / "scope_stats/scope_stats.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    records = _fixture_records() if runtime == "tensormap_and_ringbuffer" else []
    records = [{**r, "run_epoch": epoch} for r in records]
    meta = dict(
        fatal=False,
        dropped=0,
        total=len(records),
        task_window_max=[64] * 4,
        heap_max=[1048576] * 4,
        dep_pool_max=[1024] * 4,
        tensormap_max=65536,
    )
    path.write_text("\n".join(json.dumps(r) for r in [meta, *records]) + "\n")
    return path


@pytest.fixture
def exports(tmp_path):
    (tmp_path / "poison").mkdir()
    for index in range(4):
        (tmp_path / str(index)).mkdir()
        if index in (1, 2):
            _write(tmp_path / str(index), "tensormap_and_ringbuffer", 40 + index)
    return tmp_path


def test_checker_accepts_nested_scope_identity_and_cumulative_resources(consumer, exports):
    assert consumer._check_exports(exports, "tensormap_and_ringbuffer") == [41, 42]
    for index in (0, 3):
        (exports / str(index) / "host.log").write_text("ordinary host log\n")
    (exports / "poison/host.log").write_text("ordinary host log\n")
    assert consumer._check_exports(exports, "tensormap_and_ringbuffer") == [41, 42]


@pytest.mark.parametrize(
    "defect",
    [
        "stale_epoch",
        "mixed_epoch",
        "wrong_site",
        "missing_end",
        "wrong_ring",
        "reordered",
        "no_task_alloc",
        "no_heap_alloc",
    ],
)
def test_real_checker_rejects_foreign_or_incomplete_scope_records(consumer, exports, defect):
    path = exports / "2/scope_stats/scope_stats.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if defect == "stale_epoch":
        for record in rows[1:]:
            record["run_epoch"] = 41
    elif defect == "mixed_epoch":
        rows[-1]["run_epoch"] = 41
    elif defect == "wrong_site":
        rows[2]["site"] = "foreign.cpp:1"
    elif defect == "missing_end":
        rows.pop(4)
        rows[0]["total"] -= 1
    elif defect == "wrong_ring":
        rows[3]["ring"] = 1
    elif defect == "reordered":
        rows[3], rows[4] = rows[4], rows[3]
    elif defect == "no_task_alloc":
        rows[4]["task_window_end"] = rows[3]["task_window_end"]
    elif defect == "no_heap_alloc":
        rows[5]["heap_end"] = rows[2]["heap_end"]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(AssertionError):
        consumer._check_exports(exports, "tensormap_and_ringbuffer")


@pytest.mark.parametrize(
    "defect", ["dropped", "fatal", "unknown", "partial", "missing", "off_record", "poison_record", "duplicate_path"]
)
def test_real_checker_rejects_unhealthy_or_misrouted_exports(consumer, exports, defect):
    path = exports / "2/scope_stats/scope_stats.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if defect == "dropped":
        rows[0]["dropped"] = 1
    elif defect == "fatal":
        rows[0]["fatal"] = True
    elif defect == "unknown":
        rows[0].update(fatal=None, dropped=None, total=None, counts_unknown=True)
    elif defect == "partial":
        rows[0]["collection_verdict"] = "partial_safe"
    elif defect == "missing":
        path.unlink()
    elif defect == "off_record":
        _write(exports / "3", "tensormap_and_ringbuffer", 43)
    elif defect == "poison_record":
        _write(exports / "poison", "tensormap_and_ringbuffer", 43)
    elif defect == "duplicate_path":
        _write(exports / "2/retained-0", "tensormap_and_ringbuffer", 42)
    if defect != "missing":
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(AssertionError):
        consumer._check_exports(exports, "tensormap_and_ringbuffer")


def test_hbg_is_metadata_only_and_rejects_fabricated_samples(consumer, exports):
    for index in (1, 2):
        _write(exports / str(index), "host_build_graph", index)
    assert consumer._check_exports(exports, "host_build_graph") == []
    _write(exports / "2", "tensormap_and_ringbuffer", 42)
    with pytest.raises(AssertionError):
        consumer._check_exports(exports, "host_build_graph")


@pytest.fixture
def transport(consumer, monkeypatch):
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    monkeypatch.setenv("TASK_QUEUE_ENABLE", "0")
    import torch  # noqa: PLC0415 -- CPU package, without backend autoload

    events, refs, workers, snapshots = [], [], [], []
    stream = SimpleNamespace(npu_stream=17)
    error = RuntimeError("completion is unknown")
    state = SimpleNamespace(fail_boundary=False, fail_retire=False, fail_checker=False)

    class Run:
        def __init__(self, selection, epoch):
            self.selection, self.epoch = selection, epoch

        def wait(self):
            assert events.count("submit") == 4 and "boundary" in events
            if state.fail_retire:
                raise error
            events.append("retire")
            folder, runtime, enabled = self.selection
            if enabled:
                _write(folder, runtime, self.epoch)

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
            assert (handle, caller_stream) == ("callable", 17) and not self.closed
            selection = Path(config.output_prefix), self.runtime, config.enable_scope_stats
            snapshots.append(selection)
            events.append("submit")
            # Only independent CPU arithmetic; no native scheduling is emulated.
            torch.add(2 * args[0], args[1], out=args[2])
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
            events.append("buffer close")
            self.closed = True

    def prepare(torch, worker, handle, actual, raw, calls):
        assert actual is stream and raw == 17 and calls == 4
        frames, weak_tensors = [], []
        for index in range(calls):
            tensors = tuple(torch.empty(consumer.reference.SHAPE) for _ in range(3))
            weak_tensors.extend(weakref.ref(t) for t in tensors)
            frames.append(
                dict(
                    index=index,
                    tensors=tensors,
                    args=tensors,
                    buffers=[Buffer() for _ in tensors],
                    worker=worker,
                    callable=handle,
                    raw_stream=raw,
                    result=torch.empty(consumer.reference.SHAPE),
                )
            )
        return frames, weak_tensors

    def enqueue(callback):
        callback()
        return SimpleNamespace(check=lambda: None, thread_id=threading.get_native_id())

    def finish():
        events.append("boundary")
        if state.fail_boundary:
            raise error

    actual_check = consumer._check_exports

    def check(*args):
        events.append("check exports")
        if state.fail_checker:
            raise error
        return actual_check(*args)

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
    monkeypatch.setattr(consumer, "_compile", lambda runtime: "compiled")
    monkeypatch.setattr(consumer, "_check_exports", check)
    monkeypatch.setattr(consumer.reference, "_load_bridge", lambda path: bridge)
    monkeypatch.setattr(consumer.reference, "_prepare_frames", prepare)
    return SimpleNamespace(events=events, refs=refs, workers=workers, snapshots=snapshots, state=state, error=error)


@pytest.mark.parametrize("runtime", ["tensormap_and_ringbuffer", "host_build_graph"])
def test_whole_consumer_freezes_each_config_and_exports_before_close(consumer, transport, tmp_path, runtime):
    result = consumer.run(runtime, 0, Path("recording.so"), tmp_path / "diagnostics")
    assert result["observed"] == [10, 16, 22, 28] and result["caller_configs_mutated"]
    assert result["run_epochs"] == ([2, 3] if runtime == "tensormap_and_ringbuffer" else [])
    assert [enabled for _, _, enabled in transport.snapshots] == [False, True, True, False]
    assert [path.name for path, _, _ in transport.snapshots] == ["0", "1", "2", "3"]
    assert (
        transport.events
        == ["submit"] * 4 + ["boundary"] + ["retire"] * 4 + ["check exports", "close"] + ["buffer close"] * 12
    )
    assert not consumer._UNPROVEN


@pytest.mark.parametrize("failure", ["fail_boundary", "fail_retire", "fail_checker"])
def test_unknown_completion_or_export_failure_retains_all_owners(consumer, transport, tmp_path, failure):
    setattr(transport.state, failure, True)
    with pytest.raises(RuntimeError) as caught:
        consumer.run("tensormap_and_ringbuffer", 0, Path("recording.so"), tmp_path / "diagnostics")
    assert caught.value is transport.error
    gc.collect()
    worker, _, frames, queued, runs, _ = consumer._UNPROVEN[0]
    assert worker is transport.workers[0] and not worker.closed
    assert len(frames) == len(queued) == len(runs) == 4
    assert all(ref() is run for ref, run in zip(transport.refs, runs))
    assert all(not b.closed for frame in frames for b in frame["buffers"])


@pytest.mark.parametrize(
    "defect", [None, "wrong_value", "wrong_selection", "mixed_epoch", "missing_mutation", "wrong_thread"]
)
def test_actual_st_rejects_invalid_summary(tmp_path, monkeypatch, defect):
    st = _load(_ROOT / "tests/st/a2a3/framework_eager/test_scope_statistics.py", "scope_st")
    summary = dict(
        runtime="tensormap_and_ringbuffer",
        task_queue="1",
        verified=True,
        calls=4,
        selections=[False, True, True, False],
        observed=[10, 16, 22, 28],
        run_epochs=[41, 42],
        user_scope_sites=["first", "second"],
        metadata_only=False,
        caller_thread=1,
        callback_threads=[2] * 4,
        caller_configs_mutated=True,
        caller_tensors_retained=True,
    )
    if defect == "wrong_value":
        summary["observed"] = [10, 16, 22, -999]
    elif defect == "wrong_selection":
        summary["selections"] = [True] * 4
    elif defect == "mixed_epoch":
        summary["run_epochs"] = [41, 41]
    elif defect == "missing_mutation":
        summary["caller_configs_mutated"] = False
    elif defect == "wrong_thread":
        summary["callback_threads"] = [1] * 4
    result = SimpleNamespace(returncode=0, stdout="FRAMEWORK_SCOPE_STATISTICS_PASS " + json.dumps(summary), stderr="")

    def run(*args, **kwargs):
        assert kwargs["env"]["MAX_JOBS"] == "2"
        return result

    monkeypatch.setattr(st, "subprocess", SimpleNamespace(run=run))
    request = SimpleNamespace(config=SimpleNamespace(getoption=lambda name: "0"))
    if defect is None:
        st.test_framework_scope_statistics_each_invocation("tensormap_and_ringbuffer", "1", tmp_path, tmp_path, request)
    else:
        with pytest.raises(AssertionError):
            st.test_framework_scope_statistics_each_invocation(
                "tensormap_and_ringbuffer", "1", tmp_path, tmp_path, request
            )
