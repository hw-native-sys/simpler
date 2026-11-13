# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""What the early-enqueue suites' joined-launch evidence reader accepts as proof of an ordering.

Both `tensormap_and_ringbuffer` early-enqueue suites decide their whole-operator ordering claim
from records their chip children write to per-pid host-log files. Two properties of that reader are
not observable from a passing hardware run, because a hardware run that happens to be fast enough
satisfies them by accident:

delayed   a child writes the joined-launch record when a successor's native submission completes
          and a run's boundary markers when that run retires. They are separate records at
          separate times, so a wait satisfied by the first can be read before the second exists —
          and the drain loop is the only thing that advances the cursor, so a row unread by then is
          absent for good.
scoped    a run key is ``(dispatch id, pipeline slot)`` and each chip child allocates both for
          itself, so two endpoints of one Worker both name ``(1, 0)``. A cursor opened for the
          second endpoint still sees whatever the first child writes late.

The suites' own functions are loaded and exercised here, per arch, rather than reimplemented: a
copy of the rule would pass while the rule itself regressed. Nothing here touches a device — the
cursor is planted, so the reader is driven over records that no run produced.
"""

import ast
import contextlib
import importlib.util
import inspect
import mmap
import sys
import textwrap
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
from simpler.buffer import create_host_shared_buffer, mint_owner_instance_id

#: The suites carrying this reader. Both are loaded, because they hold separate copies of it and a
#: repair applied to one would otherwise go missing from the other.
_SUITES = {
    "a2a3_tmr": "st/a2a3/tensormap_and_ringbuffer/early_enqueue/test_early_enqueue.py",
    "a5_tmr": "st/a5/tensormap_and_ringbuffer/early_enqueue/test_early_enqueue.py",
}

_PREDECESSOR = (1, 0)
_SUCCESSOR = (2, 1)
_PAIR = (_PREDECESSOR, _SUCCESSOR)

_REFILL_SUITES = {
    "a2a3_tmr": ("st/a2a3/tensormap_and_ringbuffer/test_dfx_joined_launch_attribution.py", 64),
    "a2a3_hbg": ("st/a2a3/host_build_graph/test_dfx_joined_launch_attribution.py", 513),
    "a5_tmr": ("st/a5/tensormap_and_ringbuffer/test_dfx_joined_launch_attribution.py", 64),
    "a5_hbg": ("st/a5/host_build_graph/test_dfx_joined_launch_attribution.py", 513),
}


def _refill_scene_body(relative_path):
    """Load the actual consumer and reader without importing a native runtime.

    Execution and diagnostic leaves below are controlled CPU models. The submit/wait loop,
    full-tile checks and final reuse/join assertions are the scene's unmodified AST.
    """
    import torch  # noqa: PLC0415

    path = Path(__file__).resolve().parents[2] / relative_path
    source = ast.parse(path.read_text())
    constants = {
        "SIZE",
        "DTYPE",
        "_SPIN_ITERS",
        "_REFILL_INPUT_VALUES",
        "_CHAIN_LENGTH",
        "_JOIN_SPAN",
        "_OWNER_ARTIFACTS",
    }
    functions = {"_golden", "_attributes", "_join_records", "_established_pairs", "_find_artifacts"}
    methods = {"_refill_owns_its_records", "_drain"}
    nodes = []
    for node in source.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in constants for target in node.targets
        ):
            nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in functions:
            nodes.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == "_ChildTrace":
            nodes.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == "TestDfxJoinedLaunchAttribution":
            for member in node.body:
                if isinstance(member, ast.FunctionDef) and member.name in methods:
                    nodes.append(member)
                elif isinstance(member, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == "DEVICE_OWNERS" for target in member.targets
                ):
                    nodes.append(member)
    namespace = {"torch": torch, "contextlib": contextlib, "Path": Path}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102
    return namespace


@pytest.mark.parametrize("name", sorted(_REFILL_SUITES))
@pytest.mark.parametrize("fail_at_flush", [False, True])
def test_dfx_refill_offers_successors_before_strong_observation(name, tmp_path, fail_at_flush):
    """A stream observer can finish later executions without observing their own results.

    Two physical credits admit new work independently of retained semantic results. Waiting
    on an old, unobserved handle drains whatever is currently on that shared stream; it does
    not become a no-op merely because another handle's wait finished its physical execution.
    The records here model that ordering, never hardware or SDK support.
    """
    relative_path, additions = _REFILL_SUITES[name]
    scene = _refill_scene_body(relative_path)
    torch = scene["torch"]
    jobs, events, spans, tiles = [], [], [], []
    generations = [0, 0]
    tensor_refs = []

    def tracked_frombuffer(*args, **kwargs):
        tensor = torch.frombuffer(*args, **kwargs)
        tensor_refs.append(weakref.ref(tensor))
        return tensor

    owner_patterns = {
        owner: pattern.replace("*", "1")
        for owner, pattern in scene["_OWNER_ARTIFACTS"].items()
        if owner in scene["DEVICE_OWNERS"]
    }

    def views(buffers):
        return [torch.frombuffer(buffer.shm.buf, dtype=scene["DTYPE"], count=16384) for buffer in buffers]

    def retire(job):
        job.ended = True
        events.append(("physical_end", job.epoch))

    def observe(job):
        events.append(("wait", job.epoch))
        if job.observed:
            return
        # The strong observer waits for the currently offered stream suffix, but delivers
        # only this handle's result. A later handle still owes its own strong observation.
        for pending in jobs:
            if not pending.ended:
                retire(pending)
        a, b, output = views(job.buffers)
        value = (5, 11, 17, 23, 29, 35)[job.epoch - 1]
        assert torch.all(a == value) and torch.all(b == value + 1)
        output.fill_(value + additions * (value + 1))
        job.observed = True

    def submit(_worker, buffers, prefix, spin_iters):
        assert spin_iters == scene["_SPIN_ITERS"]
        live = [job for job in jobs if not job.ended]
        if len(live) == 2:
            retire(live.pop(0))
        free_slot = next(slot for slot in range(2) if all(job.slot != slot for job in live))
        generations[free_slot] += 1
        job = SimpleNamespace(
            epoch=len(jobs) + 1,
            slot=free_slot,
            generation=generations[free_slot],
            prefix=prefix,
            buffers=buffers,
            ended=False,
            observed=False,
        )
        if live:
            spans.append(
                SimpleNamespace(
                    name=scene["_JOIN_SPAN"],
                    pid=4242,
                    attrs=f"p_epoch={live[-1].epoch} s_epoch={job.epoch} observed=1 unfired=1 rc=0",
                )
            )
        jobs.append(job)
        events.append(("submit", job.epoch))
        return SimpleNamespace(wait=lambda: observe(job))

    def flush():
        assert len(jobs) == 6 and all(job.observed for job in jobs)
        events.append(("flush", len(jobs)))
        if fail_at_flush:
            raise RuntimeError("injected diagnostic failure")
        for job in jobs:
            for pattern in owner_patterns.values():
                artifact = job.prefix / pattern
                artifact.parent.mkdir(parents=True, exist_ok=True)
                artifact.touch()

    def attributed(prefix, index, _simulated, _what, *, input_value):
        job = jobs[index]
        assert prefix == job.prefix and input_value == (5, 11, 17, 23, 29, 35)[index]
        return {
            "epoch": job.epoch,
            "run_id": job.epoch,
            "owner_instance_id": "controlled-owner",
            "host_pid": 4242,
            "pipeline_slot": job.slot,
            "pipeline_generation": job.generation,
        }

    def allclose(actual, expected, **kwargs):
        assert actual.numel() == expected.numel() == 16384
        tiles.append(actual.numel())
        return torch.allclose(actual, expected, **kwargs)

    scene["torch"] = SimpleNamespace(frombuffer=tracked_frombuffer, allclose=allclose)
    case = SimpleNamespace(
        RTOL=0,
        ATOL=0,
        _diagnostic_config=lambda prefix, **_kwargs: prefix,
        _submit=submit,
        _drain=scene["_drain"].__func__,
        _checked_owners=lambda: list(owner_patterns),
        _assert_one_run_owns_this_namespace=attributed,
        _wait_for_both_runs=lambda *_args, **_kwargs: spans,
    )
    worker = SimpleNamespace(
        create_buffer=lambda nbytes: SimpleNamespace(shm=SimpleNamespace(buf=bytearray(nbytes))),
        flush_diagnostics=flush,
    )
    if fail_at_flush:
        with pytest.raises(RuntimeError, match="injected diagnostic failure") as failure:
            scene["_refill_owns_its_records"](case, worker, tmp_path / name, False, None)
        assert failure.value.__traceback__ is not None
        assert not any(ref() is not None for ref in tensor_refs), "traceback retained CPU tensor exports"
        return
    try:
        scene["_refill_owns_its_records"](case, worker, tmp_path / name, False, None)
    except AssertionError as error:
        pytest.fail(f"{error}\ncontrolled two-credit/strong-observer events: {events}")
    assert tiles == [16384] * 6
    assert events.index(("submit", 6)) < events.index(("wait", 1))
    assert [epoch for event, epoch in events if event == "wait"] == list(range(1, 7))


def _load_suite(name, relative_path, monkeypatch):
    """One early-enqueue scene module, loaded by path.

    By path rather than by import, so this does not depend on the scene tree being an importable
    package, and under a private module name so the ``SceneTestCase`` classes it defines are not
    reachable for collection from this lane.
    """
    path = Path(__file__).resolve().parents[2] / relative_path
    spec = importlib.util.spec_from_file_location(f"_evidence_reader_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("wrong_output", [False, True])
def test_hbg_control_refusal_traceback_does_not_pin_host_views(monkeypatch, wrong_output):
    import torch  # noqa: PLC0415

    module = _load_suite("a5_hbg", "st/a5/host_build_graph/early_enqueue/test_early_enqueue.py", monkeypatch)
    case = module.TestA5ThreeRunCapacity()
    buffers, errors = [], []
    owner = mint_owner_instance_id()

    def create_buffer(size):
        buffer = create_host_shared_buffer(size, owner, len(buffers) + 1)
        buffers.append(buffer)
        return buffer

    def copy_from(buffer, device):
        torch.frombuffer(buffer.shm.buf, dtype=torch.float32).copy_(device)

    class Handle:
        def __init__(self, source, control, target):
            self.source, self.control, self.target = source, control, target
            self.error = RuntimeError("prepare_native_run failed") if isinstance(control, torch.Tensor) else None
            if self.error is not None:
                errors.append(self.error)

        def wait(self):
            if self.error is not None:
                raise self.error
            source = torch.frombuffer(self.source.shm.buf, dtype=torch.float32)
            control = torch.frombuffer(self.control.shm.buf, dtype=torch.float32)
            self.target.copy_(case._link(source, control))
            if wrong_output:
                self.target[-1] += 1

    monkeypatch.setattr(case, "_require_three_sets", lambda *_: None)
    monkeypatch.setattr(
        case, "_submit_link", lambda _worker, source, control, target, *_a, **_kw: Handle(source, control, target)
    )
    monkeypatch.setattr(module, "_RunTrace", lambda *_: None)
    monkeypatch.setattr(module, "_await_child_log_line", lambda *_: "require_host_tensor refused DEVICE")
    worker = SimpleNamespace(
        create_buffer=create_buffer,
        alloc_child_tensor=lambda _device, shape, _dtype: torch.empty(shape),
        copy_from=copy_from,
        free=lambda *_: None,
    )
    try:
        if wrong_output:
            with pytest.raises(AssertionError, match="predecessor's own result"):
                case.test_a_device_control_tensor_is_refused_until_it_is_copied_to_the_host("a5", worker)
        else:
            case.test_a_device_control_tensor_is_refused_until_it_is_copied_to_the_host("a5", worker)
        assert errors and errors[0].__traceback__ is not None
        for buffer in buffers:
            buffer.close()
    finally:
        for error in errors:
            error.__traceback__ = None
        for buffer in buffers:
            buffer.close()


def _controlled_stream_fault_case(
    scene,
    monkeypatch,
    tmp_path,
    *,
    alter_log=None,
    alter_error=None,
    wrong_success_output=False,
    copied_failed_output=False,
    fatal_succeeds=False,
    missing_frame=False,
    fault_index=None,
):
    """Actual scene/reader with synthetic transport, native log and stream outcomes.

    No native or SDK execution occurs here. A real hardware run must supply these records from
    its own child. The controlled parent snapshots and error strings preserve their full identity
    so the actual scene cannot qualify a foreign or merely poisoned error by accident.
    """
    arch = scene.split("_", 1)[0]
    wide = scene.endswith("wide")
    module = _load_suite(f"fault_{scene}", _SUITES[f"{arch}_tmr"], monkeypatch)
    name = (
        "TestA5TmrWideFailureAttribution"
        if wide
        else ("TestA2a3TmrEarlyEnqueueErrorAttribution" if arch == "a2a3" else "TestA5TmrEarlyEnqueueErrorAttribution")
    )
    case = object.__new__(getattr(module, name))
    count = 5 if wide else 2
    original = (1 if wide else 0) if fault_index is None else fault_index
    pid, device = 4242, 2
    directory = tmp_path / "child-logs"
    directory.mkdir()
    log = directory / f"host.{pid}.log"
    log.touch()
    submitted, waited, snapshots, trackers, sampler_events = [], [], [], [], []

    def identity(index):
        return (1, index + 1, index if wide else index % 2, 1, index + 1, index + 1)

    def append(text):
        with log.open("a") as stream:
            stream.write(text + "\n")

    def span(index, name, attrs):
        return f"[STRACE] v=1 pid={pid} tid={pid} inv={100 + index} hid=abc depth=0 name={name} ts=100 dur=1 {attrs}"

    def error():
        _, run, slot, generation, dispatch, epoch = identity(original)
        return (
            f"chip run lane is poisoned: finalize_native_run failed with code 507018 "
            f"(run_id={run} slot={slot} generation={generation} dispatch_id={dispatch} run_epoch={epoch})"
        )

    class Buffer:
        def __init__(self, nbytes):
            self.shm = SimpleNamespace(buf=memoryview(bytearray(nbytes)))

    class Handle:
        def __init__(self, index, buffers):
            self.index, self.buffers = index, buffers

        def wait(self):
            waited.append(self.index)
            if self.index < original:
                views = [module.torch.frombuffer(b.shm.buf, dtype=module.torch.float32) for b in self.buffers]
                views[2].copy_(views[0] + 64 * views[1])
                if wrong_success_output:
                    views[2][-1] += 1
                return
            if self.index == original:
                _, _run, slot, generation, dispatch, epoch = identity(original)
                append(
                    f"[mono_ns=100][T0x1092][ERROR] log_run_failure: Run device failure: device={device} "
                    f"runner=0xabc rc=507018 run_epoch={epoch} generation={generation} "
                    f"dispatch_id={dispatch} pipeline_slot={slot} cid=1 chip_hash=0x1 chip_dev=0x2 runtime_args=0x3"
                )
            if copied_failed_output:
                module.torch.frombuffer(self.buffers[2].shm.buf, dtype=module.torch.float32)[-1] = 1
            if self.index == count - 1:
                if alter_log is not None:
                    log.write_text(alter_log(log.read_text()))
                if fatal_succeeds:
                    return
            raise RuntimeError(error() if alter_error is None else alter_error(error()))

    original_trackers = module._CaseRuns

    class CapturedCaseRuns(original_trackers):
        def __init__(self, frames):
            super().__init__(frames)
            trackers.append(self)

    def submit(_worker, buffers, _prefix, **_kwargs):
        index = len(submitted)
        submitted.append(index)
        ident = identity(index)
        if not missing_frame or index != count - 1:
            snapshots.append((ident, module._TASK_LAUNCHED, module._TASK_ACCEPTED))
        for tracker in trackers:
            tracker.note(list(snapshots))
        _, run, slot, generation, dispatch, epoch = ident
        append(
            span(
                index,
                "chip.run",
                f"run_id={run} dispatch_id={dispatch} slot_id={slot} generation={generation} run_epoch={epoch}",
            )
        )
        append(span(index, module._BOUNDARY_SPAN, f"dev_id={device} aic_start=1 aic_rc=0 wo_end=2 wo_rc=0 ts_hz=1000"))
        if index == 1:
            append(span(index, module._JOIN_SPAN, "p_disp=1 p_slot=0 s_disp=2 s_slot=1 observed=1 unfired=1 rc=0"))
        return Handle(index, buffers)

    monkeypatch.setattr(module, "_child_log_directory", lambda: str(directory))
    monkeypatch.setattr(module, "_frame_views", lambda *_args, **_kwargs: contextlib.nullcontext(snapshots))
    monkeypatch.setattr(module, "_coherent_snapshot", lambda _frames: list(snapshots))
    monkeypatch.setattr(module, "_CaseRuns", CapturedCaseRuns)
    monkeypatch.setattr(module, "_EVIDENCE_BUDGET_S", 0.0)
    monkeypatch.setattr(case, "_submit_vector", submit)
    if wide:
        original_observer = module._WideWindowObserver

        class CapturedObserver(original_observer):
            def __enter__(self):
                result = super().__enter__()
                sampler_events.append("ready")
                return result

            def __exit__(self, *args):
                result = super().__exit__(*args)
                assert not self._thread.is_alive()
                sampler_events.append("stopped")
                return result

        monkeypatch.setattr(module, "_WideWindowObserver", CapturedObserver)
    mapping = mmap.mmap(-1, 64)
    mapping_view = memoryview(mapping)
    worker = SimpleNamespace(
        _chip_pids=[pid],
        _config={"device_ids": [device]},
        _chip_run_resource_set_counts=[count],
        _chip_shms=[SimpleNamespace(buf=mapping_view)],
        create_buffer=Buffer,
    )

    def invoke():
        try:
            if wide:
                case.test_one_failure_amid_a_wide_prefix_is_its_own("a5", worker)
            else:
                case.test_a_failing_successor_reports_its_own_status(arch, worker)
        finally:
            # This is real buffer-protocol backing. A leaked sampler export makes close fail,
            # including on a controlled scene failure; shutdown is never mocked away.
            mapping_view.release()
            mapping.close()

    return SimpleNamespace(
        invoke=invoke,
        submitted=submitted,
        waited=waited,
        count=count,
        log=log,
        sampler_events=sampler_events,
        worker=worker,
        handle_type=Handle,
    )


@pytest.mark.parametrize("scene", ["a2a3_narrow", "a5_narrow", "a5_wide"])
@pytest.mark.parametrize("wrong_success_output", [False, True])
def test_fault_tracebacks_do_not_pin_host_outputs(scene, wrong_success_output, monkeypatch, tmp_path):
    controlled = _controlled_stream_fault_case(
        scene, monkeypatch, tmp_path, fault_index=1, wrong_success_output=wrong_success_output
    )
    buffers, errors = [], []
    owner = mint_owner_instance_id()

    def create_buffer(size):
        buffer = create_host_shared_buffer(size, owner, len(buffers) + 1)
        buffers.append(buffer)
        return buffer

    original_wait = controlled.handle_type.wait

    def retain_failure(handle):
        try:
            return original_wait(handle)
        except RuntimeError as error:
            errors.append(error)
            raise

    monkeypatch.setattr(controlled.worker, "create_buffer", create_buffer)
    monkeypatch.setattr(controlled.handle_type, "wait", retain_failure)
    try:
        if wrong_success_output:
            with pytest.raises(AssertionError, match="successful call.*incorrect output"):
                controlled.invoke()
        else:
            controlled.invoke()
        assert errors and all(error.__traceback__ is not None for error in errors)
        for buffer in buffers:
            buffer.close()
    finally:
        for error in errors:
            error.__traceback__ = None
        for buffer in buffers:
            buffer.close()


@pytest.mark.parametrize(
    "scene,fault_index",
    [
        ("a2a3_narrow", 0),
        ("a2a3_narrow", 1),
        ("a5_narrow", 0),
        ("a5_narrow", 1),
        ("a5_wide", 1),
        ("a5_wide", 4),
    ],
)
def test_queued_fault_uses_owned_stream_observation_instead_of_kernel_isolation(
    scene, fault_index, monkeypatch, tmp_path
):
    controlled = _controlled_stream_fault_case(scene, monkeypatch, tmp_path, fault_index=fault_index)
    controlled.invoke()
    assert controlled.submitted == list(range(controlled.count))
    assert controlled.waited == list(range(controlled.count)), "every accepted handle must still be drained"
    if scene.endswith("wide"):
        assert controlled.sampler_events == ["ready", "stopped"]


@pytest.mark.parametrize("scene", ["a2a3_narrow", "a5_narrow", "a5_wide"])
def test_queued_fault_uses_owned_native_error_without_device_timing(scene, monkeypatch, tmp_path):
    controlled = _controlled_stream_fault_case(
        scene,
        monkeypatch,
        tmp_path,
        alter_log=lambda text: "\n".join(
            line for line in text.splitlines() if "name=chip.run.runner_run.device_boundaries " not in line
        )
        + "\n",
    )
    controlled.invoke()
    assert controlled.waited == list(range(controlled.count))


@pytest.mark.parametrize("scene", ["a2a3_narrow", "a5_narrow", "a5_wide"])
@pytest.mark.parametrize(
    "fault", ["bare_poison", "unsupported_status", "copied_failure", "fatal_success", "missing_frame"]
)
def test_queued_fault_keeps_failure_output_identity_and_drain_guards(scene, fault, monkeypatch, tmp_path):
    controls = {
        "bare_poison": {"alter_error": lambda _error: "chip run lane is poisoned"},
        "unsupported_status": {"alter_error": lambda error: error.replace("507018", "507046")},
        "copied_failure": {"copied_failed_output": True},
        "fatal_success": {"fatal_succeeds": True},
        "missing_frame": {"missing_frame": True},
    }
    controlled = _controlled_stream_fault_case(scene, monkeypatch, tmp_path, **controls[fault])
    reason = {
        "bare_poison": "unsupported or unowned stream error",
        "unsupported_status": "unsupported or unowned stream error",
        "copied_failure": "failed observation copied output",
        "fatal_success": "completed successfully",
        "missing_frame": "five accepted native identities" if scene.endswith("wide") else "two runs were not both seen",
    }[fault]
    with pytest.raises(AssertionError, match=reason):
        controlled.invoke()
    assert controlled.submitted == list(range(controlled.count))
    assert controlled.waited == list(range(controlled.count))


def test_wide_stream_error_does_not_waive_any_successful_output(monkeypatch, tmp_path):
    controlled = _controlled_stream_fault_case("a5_wide", monkeypatch, tmp_path, wrong_success_output=True)
    with pytest.raises(AssertionError):
        controlled.invoke()
    assert controlled.waited == list(range(5))


@pytest.mark.parametrize("scene", ["a2a3_narrow", "a5_narrow"])
def test_queued_stream_error_still_requires_this_exact_pair_to_have_joined(scene, monkeypatch, tmp_path):
    controlled = _controlled_stream_fault_case(
        scene,
        monkeypatch,
        tmp_path,
        alter_log=lambda text: "\n".join(
            line for line in text.splitlines() if "name=chip.run.joined_launch " not in line
        )
        + "\n",
    )
    with pytest.raises(AssertionError, match="never recorded"):
        controlled.invoke()
    assert controlled.waited == [0, 1]


def _synthetic_stream_rows(*, run=1, dispatch=1, slot=0, generation=1, epoch=1, device=2, pid=4242):
    """Exact existing record schemas, deliberately synthetic CPU parser input."""
    head = f"[STRACE] v=1 pid={pid} tid={pid} inv=100 hid=abc depth=0 "
    root = head + (
        f"name=chip.run ts=100 dur=1 run_id={run} dispatch_id={dispatch} slot_id={slot} "
        f"generation={generation} run_epoch={epoch}\n"
    )
    boundary = head + (
        f"name=chip.run.runner_run.device_boundaries ts=100 dur=1 dev_id={device} "
        "aic_start=1 aic_rc=0 wo_end=2 wo_rc=0 ts_hz=1000\n"
    )
    native = (
        f"[mono_ns=100][T0x1092][ERROR] log_run_failure: Run device failure: device={device} "
        f"runner=0xabc rc=507018 run_epoch={epoch} generation={generation} dispatch_id={dispatch} "
        f"pipeline_slot={slot} cid=1 chip_hash=0x1 chip_dev=0x2 runtime_args=0x3\n"
    )
    error = (
        f"chip run lane is poisoned: finalize_native_run failed with code 507018 "
        f"(run_id={run} slot={slot} generation={generation} dispatch_id={dispatch} run_epoch={epoch})"
    )
    return root, boundary, native, error


def _owned_stream_cursor(suite, monkeypatch, tmp_path, before="", device=2):
    path = tmp_path / "host.4242.log"
    path.write_text(before)
    monkeypatch.setattr(suite, "_child_log_directory", lambda: str(tmp_path))
    return suite._RunTrace(SimpleNamespace(_chip_pids=[4242], _config={"device_ids": [device]})), path


@pytest.mark.parametrize("zero_values", [False, True])
def test_owned_stream_failure_joins_exact_fields_including_zero(suite, monkeypatch, tmp_path, zero_values):
    values = dict(run=0, dispatch=0, slot=0, generation=0, epoch=0, device=0) if zero_values else {}
    root, boundary, native, error = _synthetic_stream_rows(**values)
    trace, path = _owned_stream_cursor(suite, monkeypatch, tmp_path, device=0 if zero_values else 2)
    path.write_text(native)
    trace.take()
    assert trace.stream_failure(4242, error) is None
    with path.open("a") as stream:
        stream.write(root + boundary)
    trace.take()
    observation = trace.stream_failure(4242, error)
    assert observation is not None
    assert observation["run_id"] == (0 if zero_values else 1)
    assert observation["slot_id"] == 0
    assert observation["rc"] == 507018


def test_native_failure_accepts_the_hosts_actual_null_pointer_format(suite, monkeypatch, tmp_path):
    rendered = suite.ctypes.create_string_buffer(32)
    snprintf = suite.ctypes.CDLL(None).snprintf
    snprintf.argtypes = [suite.ctypes.c_void_p, suite.ctypes.c_size_t, suite.ctypes.c_char_p]
    snprintf.restype = suite.ctypes.c_int
    count = snprintf(rendered, len(rendered), b"%p", suite.ctypes.c_void_p(None))
    assert 0 < count < len(rendered)
    root, boundary, native, error = _synthetic_stream_rows()
    trace, path = _owned_stream_cursor(suite, monkeypatch, tmp_path)
    path.write_text(root + boundary + native.replace("runtime_args=0x3", "runtime_args=" + rendered.value.decode()))
    trace.take()
    assert trace.stream_failure(4242, error) is not None


def test_owned_native_failure_with_exact_root_needs_no_device_boundary(suite, monkeypatch, tmp_path):
    root, _boundary, native, error = _synthetic_stream_rows()
    trace, path = _owned_stream_cursor(suite, monkeypatch, tmp_path)
    path.write_text(root + native)
    trace.take()
    assert trace.stream_failure(4242, error) is not None, (
        "failed strong observation need not emit device timing; its owned native record still names the device"
    )


@pytest.mark.parametrize("field", ["device", "run_epoch", "generation", "dispatch_id", "pipeline_slot", "rc"])
def test_owned_stream_failure_rejects_each_mismatched_native_field(suite, monkeypatch, tmp_path, field):
    root, boundary, native, error = _synthetic_stream_rows()
    trace, path = _owned_stream_cursor(suite, monkeypatch, tmp_path)
    import re  # noqa: PLC0415 -- control mutates one existing native field

    native = re.sub(rf"\b{field}=-?\d+", f"{field}=999", native)
    path.write_text(root + boundary + native)
    trace.take()
    assert trace.stream_failure(4242, error) is None


@pytest.mark.parametrize("field", ["run_id", "slot", "generation", "dispatch_id", "run_epoch"])
def test_owned_stream_failure_rejects_each_mismatched_handle_field(suite, monkeypatch, tmp_path, field):
    root, boundary, native, error = _synthetic_stream_rows()
    trace, path = _owned_stream_cursor(suite, monkeypatch, tmp_path)
    import re  # noqa: PLC0415 -- control mutates one handle identity field

    path.write_text(root + boundary + native)
    trace.take()
    error = re.sub(rf"\b{field}=\d+", f"{field}=999", error)
    assert trace.stream_failure(4242, error) is None


@pytest.mark.parametrize(
    "fault",
    [
        "stale",
        "foreign_file",
        "foreign_root_pid",
        "truncated",
        "conflicting_root",
        "conflicting_device",
        "conflicting_status",
        "second_invocation",
        "wrong_boundary_device",
        "malformed_root",
        "malformed_boundary",
    ],
)
def test_owned_stream_failure_refuses_absent_or_ambiguous_provenance(suite, monkeypatch, tmp_path, fault):
    root, boundary, native, error = _synthetic_stream_rows()
    trace, path = _owned_stream_cursor(suite, monkeypatch, tmp_path, native if fault == "stale" else "")
    if fault == "stale":
        with path.open("a") as stream:
            stream.write(root + boundary)
    elif fault == "foreign_file":
        path.write_text(root + boundary)
        (tmp_path / "host.999.log").write_text(native)
    elif fault == "foreign_root_pid":
        path.write_text(root.replace("pid=4242", "pid=999") + boundary + native)
    elif fault == "truncated":
        path.write_text(root + boundary + native.split(" cid=", 1)[0] + "~\n")
    elif fault == "conflicting_root":
        path.write_text(root + root.replace("run_id=1", "run_id=99") + boundary + native)
    elif fault == "conflicting_device":
        path.write_text(root + boundary + boundary.replace("dev_id=2", "dev_id=99") + native)
    elif fault == "conflicting_status":
        path.write_text(root + boundary + native + native.replace("rc=507018", "rc=507046"))
    elif fault == "wrong_boundary_device":
        path.write_text(root + boundary.replace("dev_id=2", "dev_id=999") + native)
    elif fault == "malformed_root":
        path.write_text(root + root.replace("generation=1", "generation=~") + boundary + native)
    elif fault == "malformed_boundary":
        path.write_text(root + boundary + boundary.replace("dev_id=2", "dev_id=~") + native)
    else:
        path.write_text(root + root.replace("inv=100", "inv=101") + boundary + native)
    trace.take()
    assert trace.stream_failure(4242, error) is None


@pytest.mark.parametrize("config", [None, {}, {"device_ids": []}, {"device_ids": [2, 3]}])
def test_stream_identity_needs_explicit_unambiguous_pid_device_binding(suite, monkeypatch, tmp_path, config):
    root, boundary, native, error = _synthetic_stream_rows()
    path = tmp_path / "host.4242.log"
    path.touch()
    monkeypatch.setattr(suite, "_child_log_directory", lambda: str(tmp_path))
    worker = SimpleNamespace(_chip_pids=[4242])
    if config is not None:
        worker._config = config
    trace = suite._RunTrace(worker)
    path.write_text(root + boundary + native)
    trace.take()
    assert trace.run_of_invocation[4242, 100] == (1, 0), "the existing reader does not require the new binding"
    assert trace.stream_failure(4242, error) is None


def test_stream_identity_uses_actual_pid_device_order_without_timing(suite, monkeypatch, tmp_path):
    for pid in (4242, 5353):
        (tmp_path / f"host.{pid}.log").touch()
    monkeypatch.setattr(suite, "_child_log_directory", lambda: str(tmp_path))
    trace = suite._RunTrace(SimpleNamespace(_chip_pids=[4242, 5353], _config={"device_ids": [7, 2]}))
    root, _boundary, native, error = _synthetic_stream_rows(pid=5353, device=2)
    (tmp_path / "host.5353.log").write_text(root + native)
    trace.take()
    assert trace.stream_failure(4242, error) is None
    assert trace.stream_failure(5353, error)["device"] == 2


def test_stream_identity_rejects_native_device_outside_its_owned_pid_binding(suite, monkeypatch, tmp_path):
    root, _boundary, native, error = _synthetic_stream_rows(device=7)
    trace, path = _owned_stream_cursor(suite, monkeypatch, tmp_path, device=2)
    path.write_text(root + native)
    trace.take()
    assert trace.stream_failure(4242, error) is None


def test_propagated_lane_error_needs_an_independently_failed_handle(suite, monkeypatch, tmp_path):
    root, boundary, native, error = _synthetic_stream_rows()
    second_root, second_boundary, _native, _error = _synthetic_stream_rows(run=2, dispatch=2, slot=1, epoch=2)
    trace, path = _owned_stream_cursor(suite, monkeypatch, tmp_path)
    path.write_text(
        root
        + boundary
        + native
        + second_root.replace("inv=100", "inv=101")
        + second_boundary.replace("inv=100", "inv=101")
    )
    trace.take()
    expected = suite.torch.ones(16384)
    # A matching native record alone cannot make a successful first handle into an SDK error.
    # The second handle's poison text must anchor to an actual failed first observation.
    with pytest.raises(AssertionError, match="no handle independently reported"):
        suite._assert_stream_observations(
            trace,
            4242,
            _planted_case_runs(suite, [(1, 0), (2, 1)]),
            [(expected.clone(), expected, None), (suite.torch.zeros(16384), expected, error)],
            [],
            "controlled propagated error",
        )


@pytest.mark.parametrize("blocked", [False, True], ids=["stopped", "still-reading"])
def test_wide_fault_sampler_keeps_real_mapping_until_checked_thread_shutdown(blocked, monkeypatch, tmp_path):
    module = _load_suite("a5_fault_sampler", _SUITES["a5_tmr"], monkeypatch)
    mapping = mmap.mmap(-1, 64)
    mapping_view = memoryview(mapping)
    worker = SimpleNamespace(_chip_shms=[SimpleNamespace(buf=mapping_view)])
    entered, release = module.threading.Event(), module.threading.Event()
    calls = []

    def snapshot(_entries):
        calls.append(mapping[0])
        if blocked and len(calls) == 2:
            entered.set()
            assert release.wait(5.0)
            calls.append(mapping[0])
        return []

    monkeypatch.setattr(module, "_coherent_snapshot", snapshot)
    observer = module._WideWindowObserver(worker, [], _planted_case_runs(module, []), 5)
    try:
        observer.__enter__()
        assert observer.samples >= 1, "the protected observer must be ready before the case submits"
        if blocked:
            assert entered.wait(2.0)
            monkeypatch.setattr(module, "_OBSERVER_JOIN_BUDGET_S", 0.0)
            with pytest.raises(AssertionError, match="did not exit"):
                observer.__exit__(None, None, None)
            mapping_view.release()
            with pytest.raises(BufferError):
                mapping.close()
        else:
            observer.__exit__(None, None, None)
            mapping_view.release()
            mapping.close()
    finally:
        release.set()
        observer._thread.join(2.0)
        assert not observer._thread.is_alive()
        observer._shutdown()
        mapping_view.release()
        mapping.close()
    assert calls and all(value == 0 for value in calls)


@pytest.fixture(params=sorted(_SUITES), ids=sorted(_SUITES))
def suite(request, monkeypatch):
    return _load_suite(request.param, _SUITES[request.param], monkeypatch)


@pytest.fixture
def hbg_suite(monkeypatch):
    return _load_suite("a5_hbg", "st/a5/host_build_graph/early_enqueue/test_early_enqueue.py", monkeypatch)


def _planted_trace(suite, pids, identities, boundaries):
    """A suite's own ``_RunTrace`` with its parsed state planted and no files behind it.

    Built through ``__new__`` so the real ``boundaries_of_run`` lookup is what the checks below
    exercise. Each child is given a path that does not exist, which is a state the cursor already
    tolerates, so reading it yields nothing rather than raising.
    """
    trace = object.__new__(suite._RunTrace)
    trace._pids = list(pids)
    trace._paths = {pid: Path(f"/nonexistent/host.{pid}.log") for pid in pids}
    trace._offsets = dict.fromkeys(pids, 0)
    trace._origins = dict.fromkeys(pids, 0)
    trace.run_of_invocation = dict(identities)
    trace.boundary_of_invocation = dict(boundaries)
    return trace


def _planted_case_runs(suite, keys):
    """A suite's own ``_CaseRuns`` holding exactly these keys, with no mailbox behind it."""
    case_runs = object.__new__(suite._CaseRuns)
    case_runs.floor = 0
    case_runs._keys = set(keys)
    return case_runs


def _planted_boundary(device, aicore_start, whole_operator_end):
    """One boundary row in the shape ``_RunTrace._note_boundaries`` produces.

    A position the emitter could not take carries a non-zero rc and no time, which is why an
    absent reading is ``None`` here rather than zero.
    """
    return {
        "device": device,
        "aicore_start": aicore_start,
        "aicore_rc": 0 if aicore_start is not None else 1,
        "whole_operator_end": whole_operator_end,
        "whole_operator_rc": 0 if whole_operator_end is not None else 1,
        "hz": 1_000_000,
    }


def _planted_join(pid, predecessor, successor):
    """One joined-launch record in the shape ``_join_record`` produces."""
    return {
        "pid": pid,
        "inv": 0,
        "predecessor": predecessor,
        "successor": successor,
        "observed": True,
        "unfired": True,
        "query_rc": 0,
    }


def test_a_late_successor_boundary_holds_the_evidence_wait_open(suite):
    """The join record is not evidence that the readings the ordering claim needs have arrived."""
    pid = 4242
    records = [_planted_join(pid, _PREDECESSOR, _SUCCESSOR)]
    case_runs = _planted_case_runs(suite, [_PREDECESSOR, _SUCCESSOR])
    identities = {(pid, 1): _PREDECESSOR, (pid, 2): _SUCCESSOR}

    half = _planted_trace(suite, [pid], identities, {(pid, 1): _planted_boundary(6, 100, 200)})
    assert suite._established_pairs(records, case_runs, pid) == {_PAIR}, (
        "the record does establish the pair, so what follows is about the readings it does not carry"
    )
    assert not suite._boundaries_settled(half, records, case_runs, pid), (
        "the wait was satisfied while the successor's boundary row was still missing, which is the "
        "premature read this predicate exists to prevent"
    )
    assert suite._whole_operator_order(half, records, case_runs, pid) == (set(), set(), set(), {_PAIR}), (
        "a pair with one half of its readings must be unmeasured, never ordered"
    )

    whole = _planted_trace(
        suite,
        [pid],
        identities,
        {(pid, 1): _planted_boundary(6, 100, 200), (pid, 2): _planted_boundary(6, 300, 400)},
    )
    assert suite._boundaries_settled(whole, records, case_runs, pid), (
        "with both rows present the wait must stop, or a passing case would always burn its budget"
    )
    assert suite._whole_operator_order(whole, records, case_runs, pid) == ({_PAIR}, set(), set(), set())


def test_a_successor_released_too_early_is_still_overlapping(suite):
    """Waiting for the readings does not soften what they then have to say.

    The predicate holds out for a comparable pair, so the one case that must never be reported as
    an absence is a pair whose readings are present and wrong.
    """
    pid = 4242
    records = [_planted_join(pid, _PREDECESSOR, _SUCCESSOR)]
    case_runs = _planted_case_runs(suite, [_PREDECESSOR, _SUCCESSOR])
    trace = _planted_trace(
        suite,
        [pid],
        {(pid, 1): _PREDECESSOR, (pid, 2): _SUCCESSOR},
        {(pid, 1): _planted_boundary(6, 100, 400), (pid, 2): _planted_boundary(6, 300, 500)},
    )
    assert suite._boundaries_settled(trace, records, case_runs, pid)
    ordered, overlapping, within_tick, unmeasured = suite._whole_operator_order(trace, records, case_runs, pid)
    assert (ordered, overlapping, within_tick, unmeasured) == (set(), {_PAIR}, set(), set()), (
        "a successor released before its predecessor's whole operator ended must stay an overlap"
    )


def test_another_endpoints_record_is_not_this_endpoints_evidence(suite):
    """Two endpoints of one Worker name their runs identically, so the pid has to decide."""
    mine, theirs = 4242, 5353
    case_runs = _planted_case_runs(suite, [_PREDECESSOR, _SUCCESSOR])
    records = [_planted_join(theirs, _PREDECESSOR, _SUCCESSOR)]
    trace = _planted_trace(
        suite,
        [mine, theirs],
        {(theirs, 1): _PREDECESSOR, (theirs, 2): _SUCCESSOR},
        {(theirs, 1): _planted_boundary(4, 100, 200), (theirs, 2): _planted_boundary(4, 300, 400)},
    )

    assert not suite._established_pairs(records, case_runs, mine), (
        "the other endpoint's record established a pair for this endpoint, so a claim about this "
        "device could be carried by another device's runs"
    )
    assert not suite._boundaries_settled(trace, records, case_runs, mine)
    assert suite._established_pairs(records, case_runs, theirs) == {_PAIR}, (
        "the record is real evidence for the endpoint that wrote it; only its attribution is scoped"
    )
    # Unscoped is what the defect looked like: the pair is established, and its readings are found
    # on whichever child happens to hold rows under the same key.
    assert suite._whole_operator_order(trace, records, case_runs)[0] == {_PAIR}

    detail = suite._boundary_detail(trace, {_PAIR})
    assert set(detail) == {(mine, _PAIR), (theirs, _PAIR)}, (
        "a per-pair key would report only the child read last, hiding which endpoint the readings came from"
    )
    assert detail[mine, _PAIR] == {"predecessor": None, "successor": None}
    assert suite._boundary_detail(trace, {_PAIR}, mine) == {(mine, _PAIR): {"predecessor": None, "successor": None}}


def _three_run_wait_predicate(suite, trace, case_runs):
    """Use the scene's actual wait condition, not a separately reimplemented oracle."""
    method = suite.TestA5ThreeRunCapacity.test_three_runs_are_launched_and_accepted_at_once
    tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
    waits = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_await_records"
    ]
    assert len(waits) == 1 and isinstance(waits[0].args[3], ast.Lambda)
    expression = ast.Expression(waits[0].args[3])
    return eval(
        compile(expression, inspect.getfile(method), "eval"), {**vars(suite), "trace": trace, "case_runs": case_runs}
    )


@pytest.mark.parametrize("last_start", [500, 400, 350], ids=["ordered", "within_tick", "overlapping"])
def test_hbg_three_run_wait_drains_a_late_boundary_without_hiding_overlap(hbg_suite, monkeypatch, last_start):
    pid, third = 4242, (3, 2)
    runs = _planted_case_runs(hbg_suite, [_PREDECESSOR, _SUCCESSOR, third])
    trace = _planted_trace(
        hbg_suite,
        [pid],
        {(pid, 1): _PREDECESSOR, (pid, 2): _SUCCESSOR, (pid, 3): third},
        {(pid, 1): _planted_boundary(0, 100, 200), (pid, 2): _planted_boundary(0, 300, 400)},
    )
    joins = [_planted_join(pid, _PREDECESSOR, _SUCCESSOR), _planted_join(pid, _SUCCESSOR, third)]
    reads = []

    def take():
        reads.append(None)
        if len(reads) == 1:
            return joins
        trace.boundary_of_invocation[pid, 3] = _planted_boundary(0, last_start, 600)
        return []

    monkeypatch.setattr(trace, "take", take)
    records = []
    assert hbg_suite._await_records(trace, records, 1.0, _three_run_wait_predicate(hbg_suite, trace, runs))
    assert len(reads) == 2, "the join rows arrived before the last boundary; the real wait must keep draining"
    ordered, overlapping, within_tick, unmeasured = hbg_suite._whole_operator_order(trace, records, runs)
    assert not unmeasured
    assert _PAIR in ordered
    last_pair = (_SUCCESSOR, third)
    assert (last_pair in overlapping) == (last_start < 400)
    assert (last_pair in within_tick) == (last_start == 400)
    assert (last_pair in ordered) == (last_start > 400)


def test_hbg_three_run_wait_does_not_accept_a_permanently_missing_boundary(hbg_suite, monkeypatch):
    pid, third = 4242, (3, 2)
    runs = _planted_case_runs(hbg_suite, [_PREDECESSOR, _SUCCESSOR, third])
    trace = _planted_trace(
        hbg_suite,
        [pid],
        {(pid, 1): _PREDECESSOR, (pid, 2): _SUCCESSOR, (pid, 3): third},
        {(pid, 1): _planted_boundary(0, 100, 200), (pid, 2): _planted_boundary(0, 300, 400)},
    )
    joins = [_planted_join(pid, _PREDECESSOR, _SUCCESSOR), _planted_join(pid, _SUCCESSOR, third)]
    monkeypatch.setattr(trace, "take", lambda: joins)
    records = []
    assert not hbg_suite._await_records(trace, records, 0, _three_run_wait_predicate(hbg_suite, trace, runs))
    assert hbg_suite._whole_operator_order(trace, records, runs)[3] == {(_SUCCESSOR, third)}


@pytest.mark.parametrize("name", sorted(_REFILL_SUITES))
def test_dfx_failure_keeps_initial_timing_outside_log_tail(name):
    scene = _refill_scene_body(_REFILL_SUITES[name][0])
    spans = [
        SimpleNamespace(pid=42, inv=inv, name="chip.run.prepare_execution", ts=inv * 100, dur=10, attrs="")
        for inv in (7, 8, 9)
    ]
    spans.append(
        SimpleNamespace(pid=42, inv=8, name="chip.run.joined_launch", ts=999, dur=0, attrs="observed=1 unfired=0 rc=0")
    )
    trace = SimpleNamespace(spans=lambda: spans, window_text=lambda: "tail omits first pair")
    result = scene["_ChildTrace"].safe_window_text(trace)
    assert "inv=7 name=chip.run.prepare_execution ts=700 dur=10" in result
    assert "inv=8 name=chip.run.joined_launch ts=999 dur=0 observed=1 unfired=0 rc=0" in result
    assert "inv=9" not in result
    assert result.endswith("tail omits first pair")
