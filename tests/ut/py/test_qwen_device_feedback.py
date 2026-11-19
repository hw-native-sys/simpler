# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DEVICE feedback, bounded admission, and result qualification contracts."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from simpler_setup.torch_interop import torch_dtype_to_datatype

CASE = Path(__file__).resolve().parents[3] / "examples/a2a3/host_build_graph/qwen3_14b_serving_effective"
sys.path.insert(0, str(CASE))
import device_feedback_worker_submit as feedback  # noqa: E402
from clone_worker_submit import JoinedEvidenceError, _await_joined_evidence, _first_join_evidence  # noqa: E402


def test_every_successor_consumes_the_actual_device_output():
    initial = object()
    cache = object()
    resident = {"sampled_ids_in": initial, "k_cache": cache}
    outputs = [{"sampled_ids": object(), "out": object()} for _ in range(3)]
    metadata = [{"seq_lens": object()} for _ in range(3)]
    for i in range(5):
        frame = i % 3
        previous = None if i == 0 else outputs[(i - 1) % 3]
        row = feedback._feedback_buffers(resident, outputs[frame], metadata[frame], previous)
        assert row["sampled_ids_in"] is (initial if previous is None else previous["sampled_ids"])
        assert row["sampled_ids"] is outputs[frame]["sampled_ids"]
        assert row["seq_lens"] is metadata[frame]["seq_lens"]
        assert row["k_cache"] is cache
    assert resident == {"sampled_ids_in": initial, "k_cache": cache}


@pytest.mark.parametrize("steps", [1, 2, 3, 127])
@pytest.mark.parametrize("depth", [1, 2])
def test_chain_refills_before_waiting_for_the_successor(steps, depth):
    active = []
    events = []

    def submit(index, _args):
        assert len(active) < depth
        active.append(index)
        events.append(("submit", index))

        def result(timeout):
            assert timeout == 120
            assert active.pop(0) == index
            events.append(("complete", index))
            handle._run_id = None

        handle = SimpleNamespace(result=result, _run_id=index + 10)
        return handle

    def prepare(index):
        assert len(active) < depth
        events.append(("prepare", index))

    def finish(index, handle):
        events.append(("readback", index))

    run_ids = feedback._run_device_chain(steps, prepare, submit, finish, depth=depth)
    assert run_ids == list(range(10, steps + 10))
    assert not active
    if steps >= 3 and depth == 2:
        assert events[:8] == [
            ("prepare", 0),
            ("submit", 0),
            ("prepare", 1),
            ("submit", 1),
            ("complete", 0),
            ("readback", 0),
            ("prepare", 2),
            ("submit", 2),
        ]
        assert events.index(("submit", 2)) < events.index(("complete", 1))


@pytest.mark.parametrize("failure", ["submit", "complete"])
def test_failure_stops_admission_and_drains_submitted_consumers(failure):
    submitted = []
    completed = []
    original = RuntimeError("producer failed")

    def submit(index, _args):
        if failure == "submit" and index == 1:
            raise original
        submitted.append(index)

        def result(timeout):
            assert timeout == 120
            if failure == "complete" and index == 0:
                raise original
            completed.append(index)

        return SimpleNamespace(result=result, _run_id=index + 10)

    with pytest.raises(RuntimeError) as caught:
        feedback._run_device_chain(5, lambda _: None, submit, lambda *_: None)
    assert caught.value is original
    assert submitted == ([0] if failure == "submit" else [0, 1])
    assert (0 if failure == "submit" else 1) in completed


@pytest.mark.parametrize("steps,depth", [(0, 3), (-1, 3), (2, 0), (2, 4)])
def test_invalid_chain_never_submits(steps, depth):
    with pytest.raises(ValueError):
        feedback._run_device_chain(
            steps, lambda _: None, lambda *_: pytest.fail("invalid submission"), lambda *_: None, depth=depth
        )


@pytest.mark.parametrize("token", [11, 892])
def test_sentinel_rejects_a_missing_write_even_for_repeated_expected_tokens(token):
    expected = torch.tensor([token], dtype=torch.int32)
    sampled = torch.full((1, 8), -1, dtype=torch.int32)
    with pytest.raises(AssertionError, match="sampled token mismatch"):
        feedback._check_sampled(sampled, expected, 1)
    sampled[:, 0] = token
    assert feedback._check_sampled(sampled, expected, 1) == [token]


def test_nonfinite_logits_cannot_pass_a_correct_token():
    with pytest.raises(AssertionError, match="logits failed"):
        feedback._check_logits(
            torch.full((1, 2), float("nan")),
            torch.tensor([1.0, 2.0]),
            0,
        )


def test_serial_launch_needs_completion_identity_and_ordered_device_boundaries():
    spans = _spans()
    spans[2].name = "chip.run.runner_run.aicpu_launch"
    assert _first_join_evidence(spans, (10, 11)) is None
    assert _first_join_evidence(spans, (10, 11), allow_serial=True)["kind"] == "serial_launch"
    spans[-1].attrs = spans[-1].attrs.replace("aic_start=101", "aic_start=99")
    with pytest.raises(AssertionError, match="overlapped"):
        _first_join_evidence(spans, (10, 11), allow_serial=True)


def test_failed_join_is_not_reclassified_as_serial():
    spans = _spans()
    spans.append(SimpleNamespace(pid=1, inv=2, name="chip.run.runner_run.aicpu_launch", attrs=""))
    spans[2].attrs = spans[2].attrs.replace("rc=0", "rc=1")
    assert _first_join_evidence(spans, (10, 11), allow_serial=True) is None


def _spans():
    return [
        SimpleNamespace(pid=1, inv=1, name="chip.run", attrs="run_id=10 dispatch_id=20 slot_id=0 run_epoch=1"),
        SimpleNamespace(pid=1, inv=2, name="chip.run", attrs="run_id=11 dispatch_id=21 slot_id=1 run_epoch=2"),
        SimpleNamespace(
            pid=1,
            inv=2,
            name="chip.run.joined_launch",
            attrs="p_disp=20 p_slot=0 p_epoch=1 s_disp=21 s_slot=1 s_epoch=2 observed=1 unfired=1 rc=0",
        ),
        SimpleNamespace(
            pid=1,
            inv=1,
            name="chip.run.runner_run.device_boundaries",
            attrs="dev_id=0 ts_hz=1000000 wo_rc=0 wo_end=100 aic_rc=0 aic_start=10",
        ),
        SimpleNamespace(
            pid=1,
            inv=2,
            name="chip.run.runner_run.device_boundaries",
            attrs="dev_id=0 ts_hz=1000000 wo_rc=0 wo_end=200 aic_rc=0 aic_start=101",
        ),
    ]


@pytest.mark.parametrize("replacement", ["observed=0", "unfired=0", "rc=1", "p_disp=19", "p_epoch=999", "s_epoch=998"])
def test_fallback_or_stale_edge_cannot_qualify(replacement):
    spans = _spans()
    key = replacement.split("=")[0]
    spans[2].attrs = " ".join(replacement if p.startswith(key + "=") else p for p in spans[2].attrs.split())
    assert _first_join_evidence(spans, (10, 11)) is None


@pytest.mark.parametrize("replacement", ["wo_end=0", "wo_rc=1", "ts_hz=0"])
def test_invalid_device_boundary_is_not_serial_order_evidence(replacement):
    spans = _spans()
    key = replacement.split("=")[0]
    spans[3].attrs = " ".join(replacement if p.startswith(key + "=") else p for p in spans[3].attrs.split())
    with pytest.raises(AssertionError):
        _first_join_evidence(spans, (10, 11))


def test_missing_joined_trace_fails_instead_of_silent_serial_success(tmp_path):
    with pytest.raises(JoinedEvidenceError) as caught:
        _await_joined_evidence(tmp_path, [1], ((10, 11), (11, 12)), timeout=0)
    assert caught.value.joined_count == 0
    assert caught.value.missing_pairs == ((10, 11), (11, 12))


def test_serial_policy_still_rejects_missing_trace(tmp_path):
    with pytest.raises(JoinedEvidenceError):
        _await_joined_evidence(tmp_path, [1], ((10, 11),), timeout=0, allow_serial=True)


def test_retained_output_capacity_is_checked_before_allocating():
    resident = {name: SimpleNamespace(nbytes=64) for name in ("sampled_ids", "next_hidden")}
    worker = SimpleNamespace(device_memory_info=lambda: SimpleNamespace(free_bytes=127))
    with pytest.raises(MemoryError, match="retained step outputs"):
        feedback._allocate_frames(worker, resident, {"out": object()}, None, 2, [])


@pytest.fixture
def chain_runtime(monkeypatch, tmp_path):
    for name, value in (("BATCH", 2), ("PADDED_VOCAB", 3), ("HIDDEN", 4)):
        monkeypatch.setattr(feedback, name, value)

    class Buffer:
        next_id = 0
        space_next = {False: 0, True: 0}

        def __init__(self, shape, dtype, *, device=False):
            Buffer.next_id += 1
            self.identity = SimpleNamespace(buffer_id=Buffer.next_id)
            self.nbytes = torch.empty(shape, dtype=dtype).numel() * torch.empty((), dtype=dtype).element_size()
            Buffer.space_next[device] += 1
            self.base = Buffer.space_next[device] * 4096
            self.shm = SimpleNamespace(buf=bytearray(self.nbytes))
            self.value = feedback._view(self, shape, dtype)
            self.device = device
            self.closed = False

        def close(self):
            assert not worker.pending
            self.closed = True

    class Worker:
        def __init__(self):
            self.pending = []
            self.next_id = 1
            self.hosts = []
            self.closed = False
            self.fail_submit = None
            self.fail_complete = None
            self.fail_close = False
            self.timeout = False
            self.timeout_once = False
            self.skip_sample_write = False
            self.skip_sample_at = None
            self.skip_logit_at = None
            self.fixed_token = None
            self.copy_active = []
            self.allocations = []
            self._chip_pids = []

        def register(self, chip):
            return chip

        def init(self):
            pass

        def alloc_child_tensor(self, _rank, shape, dtype):
            dtype = {torch_dtype_to_datatype(t).value: t for t in (torch.int32, torch.float32, torch.bfloat16)}[dtype]
            buffer = Buffer(shape, dtype, device=True)
            self.allocations.append(buffer)
            return buffer

        def device_memory_info(self):
            return SimpleNamespace(free_bytes=1 << 30)

        def copy_to(self, destination, source):
            self.copy_active.append(len(self.pending))
            assert not self.pending
            destination.value.copy_(source.value)

        def copy_from(self, destination, source):
            self.copy_active.append(len(self.pending))
            assert not self.pending
            destination.value.copy_(source.value)

        def submit(self, dispatch, args, config):
            index = self.next_id - 1
            if index == self.fail_submit:
                raise RuntimeError("submit failed")
            holder = []
            orch = SimpleNamespace(submit_next_level=lambda _chip, values, _cfg, worker: holder.append(values))
            dispatch(orch, args, config)
            values = holder[0]
            handle = SimpleNamespace(_run_id=self.next_id)
            self.next_id += 1
            self.pending.append((handle, values))

            def result(timeout):
                assert timeout == 120
                if self.timeout:
                    raise TimeoutError("consumer still executing")
                if self.timeout_once:
                    self.timeout_once = False
                    raise TimeoutError("temporary fence timeout")
                if handle._run_id is None:
                    if hasattr(handle, "error"):
                        raise handle.error
                    return
                current, output = self.pending.pop(0)
                assert current is handle
                handle._run_id = None
                if index == self.fail_complete:
                    handle.error = RuntimeError("producer failed")
                    raise handle.error
                if not self.skip_sample_write and index != self.skip_sample_at:
                    output["sampled_ids"].value[:, 0] = (
                        output["sampled_ids_in"].value[:, 0] + 1 if self.fixed_token is None else self.fixed_token
                    )
                if index != self.skip_logit_at:
                    output["out"].value.copy_(torch.tensor([[1.0, 2.0, 3.0]]).repeat(2, 1))

            handle.result = result
            return handle

        def close(self):
            if self.timeout:
                raise TimeoutError("close deferred with live consumers")
            self.pending.clear()
            self.closed = True
            if self.fail_close:
                raise RuntimeError("close failed")

    worker = Worker()

    def host_buffer(_worker, value):
        host = Buffer(tuple(value.shape), value.dtype)
        host.value.copy_(value)
        worker.hosts.append(host)
        return host

    def resident(_worker, devices, hosts, _fixture, _model, **_kwargs):
        for name, shape, dtype in (
            ("sampled_ids_in", (2, 8), torch.int32),
            ("sampled_ids", (2, 8), torch.int32),
            ("out", (2, 3), torch.float32),
            ("next_hidden", (2, 4), torch.bfloat16),
            ("seq_lens", (2,), torch.int32),
            ("slot_mapping", (2,), torch.int32),
            ("block_table", (4,), torch.int32),
        ):
            hosts[name] = host_buffer(worker, torch.zeros(shape, dtype=dtype))
            devices[name] = Buffer(shape, dtype, device=True)

    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "distributed_meta.json").write_text('{"params": []}')
    reference = tmp_path / "reference"
    reference.mkdir()
    (reference / "SHA256SUMS").write_text("sealed")
    fixture = SimpleNamespace(
        root=reference,
        steps=127,
        num_pages=4,
        metadata={"first_token_id": torch.tensor([10], dtype=torch.int32)},
        golden={
            "decode_output_token_ids": torch.arange(11, 138, dtype=torch.int32),
            "logits": torch.tensor([[1.0, 2.0, 3.0]]).repeat(127, 1),
        },
        verify_model=lambda _: None,
        step=lambda i: {
            "seq_lens": torch.full((2,), 128 + i + 1, dtype=torch.int32),
            "slot_mapping": torch.tensor([128 + i, 384 + i], dtype=torch.int32),
            "block_table": torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
        },
    )
    arguments = SimpleNamespace(
        fixture=reference,
        artifact=artifact,
        model=tmp_path,
        kv_reference=tmp_path,
        output=tmp_path / "output",
        steps=3,
        device=0,
    )
    monkeypatch.setattr(feedback, "_parse_args", lambda: arguments)
    monkeypatch.setattr(feedback, "ReferenceFixture", lambda _: fixture)
    monkeypatch.setattr(feedback, "Worker", lambda **_: worker)
    monkeypatch.setattr(feedback, "_host_buffer", host_buffer)
    monkeypatch.setattr(feedback, "allocate_resident", resident)
    monkeypatch.setattr(feedback, "load_kv_reference", lambda *_: {})
    monkeypatch.setattr(
        feedback, "inspect_artifact", lambda _: SimpleNamespace(compile=lambda **_: object(), source_hashes={})
    )
    monkeypatch.setattr(feedback, "_bind", lambda _specs, row, **_: row)
    monkeypatch.setattr(feedback, "validate_kv", lambda *_: [{"passed": True}])
    monkeypatch.setattr(
        feedback,
        "_await_joined_evidence",
        lambda _logs, _pids, pairs, **_: {"pair_count": len(pairs), "serial_pairs": ()},
    )
    monkeypatch.setattr(
        feedback,
        "_submission_evidence",
        lambda _logs, ids, **kwargs: [
            {"run_id": i, "prepare_only": int(i % 2 == 0), "enqueued": i in kwargs.get("completed_run_ids", ())}
            for i in ids
        ],
    )
    return worker, arguments


def test_three_step_device_chain_reads_back_only_after_drain(chain_runtime):
    worker, arguments = chain_runtime
    assert feedback.main() == 0
    report = json.loads((arguments.output / "result.json").read_text())
    assert [s["sampled"] for s in report["steps"]] == [[11, 11], [12, 12], [13, 13]]
    assert report["run_ids"] == [1, 2, 3]
    assert report["joined_count"] == 2
    assert report["fallback_count"] == 0
    assert worker.copy_active and not any(worker.copy_active)
    assert report["accepted_count"] == report["enqueued_count"] == report["completed_count"] == 3
    assert report["staged_count"] == 1
    assert report["retired_count"] == 3
    assert worker.closed and all(h.closed for h in worker.hosts)


def test_long_chain_uses_fixed_frames_and_resets_reused_outputs(chain_runtime):
    worker, arguments = chain_runtime
    arguments.steps = 127
    assert feedback.main() == 0
    report = json.loads((arguments.output / "result.json").read_text())
    assert len(worker.allocations) == (arguments.steps - 1) + feedback.FRAME_COUNT - 1
    assert len(report["feedback_buffers"]) == arguments.steps
    assert [s["sampled"] for s in report["steps"]] == [[i, i] for i in range(11, 138)]
    assert [s["frame"] for s in report["steps"]] == [i % feedback.FRAME_COUNT for i in range(127)]
    assert report["accepted_count"] == report["completed_count"] == report["retired_count"] == 127


def test_reused_frame_missing_write_cannot_pass_repeated_tokens(chain_runtime):
    worker, arguments = chain_runtime
    arguments.steps = 5
    worker.fixed_token = 11
    worker.skip_sample_at = 3
    fixture = feedback.ReferenceFixture(arguments.fixture)
    fixture.golden["decode_output_token_ids"].fill_(11)
    with pytest.raises(AssertionError, match="sampled token mismatch at step 3"):
        feedback.main()
    report = json.loads((arguments.output / "result.json").read_text())
    assert report["accepted_count"] == report["completed_count"] == 5
    assert len(report["steps"]) == 5


@pytest.mark.parametrize("missing", ["frame", "run", "launch"])
def test_submission_evidence_requires_native_state_proof(tmp_path, monkeypatch, missing):
    spans = [
        SimpleNamespace(pid=1, inv=1, name="node.frame_submit", attrs="run_id=7 dispatch_id=9 prepare_only=1"),
        SimpleNamespace(
            pid=2, inv=3, name="chip.run", attrs="run_id=7 dispatch_id=9 slot_id=0 generation=2 run_epoch=4"
        ),
        SimpleNamespace(pid=2, inv=3, name="chip.run.runner_run.aicpu_launch", attrs=""),
    ]
    (tmp_path / "host.1.log").write_text("trace")
    monkeypatch.setattr(feedback, "parse_spans", lambda _: spans)
    assert feedback._submission_evidence(tmp_path, [7], completed_run_ids=(7,)) == [
        {
            "run_id": 7,
            "dispatch_id": 9,
            "slot_id": 0,
            "generation": 2,
            "run_epoch": 4,
            "prepare_only": 1,
            "enqueued": True,
        }
    ]
    spans.pop({"frame": 0, "run": 1, "launch": 2}[missing])
    with pytest.raises(AssertionError):
        feedback._submission_evidence(tmp_path, [7])


def test_observed_serial_launch_is_reported(chain_runtime, monkeypatch):
    worker, arguments = chain_runtime
    monkeypatch.setattr(
        feedback,
        "_await_joined_evidence",
        lambda *_args, **_kwargs: {
            "pair_count": 1,
            "serial_pairs": ((1, 2),),
        },
    )
    assert feedback.main() == 0
    report = json.loads((arguments.output / "result.json").read_text())
    assert report["fallback_edges"][0]["reason"] == "serial_launch"


def test_serial_launch_after_refill_is_reported(chain_runtime, monkeypatch):
    worker, arguments = chain_runtime
    monkeypatch.setattr(
        feedback,
        "_await_joined_evidence",
        lambda *_args, **_kwargs: {
            "pair_count": 1,
            "serial_pairs": ((2, 3),),
        },
    )
    with pytest.raises(AssertionError, match="after the initial submissions"):
        feedback.main()
    report = json.loads((arguments.output / "result.json").read_text())
    assert report["fallback_edges"] == [{"predecessor": 2, "successor": 3, "reason": "serial_launch"}]


def test_chain_rejects_an_all_fallback_trace(chain_runtime, monkeypatch):
    worker, arguments = chain_runtime
    monkeypatch.setattr(
        feedback,
        "_await_joined_evidence",
        lambda _logs, _pids, pairs, **_: {"pair_count": 0, "serial_pairs": tuple(pairs)},
    )
    with pytest.raises(AssertionError, match="fell back for every pair"):
        feedback.main()
    report = json.loads((arguments.output / "result.json").read_text())
    assert report["joined_count"] == 0
    assert report["fallback_count"] == 2


@pytest.mark.parametrize("failure", ["fail_submit", "fail_complete", "fail_close", "skip_sample_write"])
def test_chain_failure_is_reported_and_private_hosts_are_closed(chain_runtime, failure):
    worker, arguments = chain_runtime
    setattr(worker, failure, 1 if failure in ("fail_submit", "fail_complete") else True)
    with pytest.raises((RuntimeError, AssertionError)):
        feedback.main()
    report = json.loads((arguments.output / "result.json").read_text())
    assert report["status"] == "failed"
    assert len(report["run_ids"]) == report["accepted_count"]
    if failure != "fail_complete":
        assert report["enqueued_count"] == report["accepted_count"]
    else:
        assert report["enqueued_count"] is None
        assert report["observed_enqueued_count"] == report["completed_count"]
    assert worker.closed
    assert all(h.closed == (failure != "fail_close") for h in worker.hosts)


def test_wait_and_close_timeout_preserve_live_consumers(chain_runtime):
    worker, arguments = chain_runtime
    worker.timeout = True
    with pytest.raises(TimeoutError, match="consumer still executing"):
        feedback.main()
    report = json.loads((arguments.output / "result.json").read_text())
    assert len(worker.pending) == 2
    assert worker.next_id == 3
    assert all(not host.closed for host in worker.hosts)
    assert report["run_ids"] == [1, 2]
    assert report["completed_count"] == report["retired_count"] == 0
    assert not report["worker_closed"]


def test_wait_timeout_followed_by_drain_preserves_error_and_retires(chain_runtime):
    worker, arguments = chain_runtime
    worker.timeout_once = True
    with pytest.raises(TimeoutError, match="temporary fence timeout"):
        feedback.main()
    report = json.loads((arguments.output / "result.json").read_text())
    assert report["run_ids"] == [1, 2]
    assert report["completed_count"] == report["retired_count"] == 2
    assert report["status"] == "failed"
    assert report["worker_closed"]
    assert not worker.pending
    assert all(host.closed for host in worker.hosts)


def test_failure_distinguishes_missing_trace_from_zero_enqueues(chain_runtime, monkeypatch):
    worker, arguments = chain_runtime
    worker.fail_complete = 0
    monkeypatch.setattr(feedback, "_submission_evidence", lambda *_args, **_kwargs: [])
    with pytest.raises(RuntimeError, match="producer failed"):
        feedback.main()
    report = json.loads((arguments.output / "result.json").read_text())
    assert report["run_ids"] == [1, 2]
    assert report["accepted_count"] == 2
    assert report["completed_count"] == 1
    assert report["enqueued_count"] is report["staged_count"] is None
    assert report["observed_enqueued_count"] == 0


def test_submission_evidence_consumes_lazy_parser_while_file_is_open(tmp_path):
    records = [
        ("node.frame_submit", "run_id=7 dispatch_id=9 prepare_only=1"),
        ("chip.run", "run_id=7 dispatch_id=9 slot_id=0 generation=2 run_epoch=4"),
        ("chip.run.runner_run.aicpu_launch", ""),
    ]
    (tmp_path / "host.1.log").write_text(
        "\n".join(
            f"[STRACE] v=1 pid=1 tid=1 inv=3 hid=abc depth=0 name={name} ts=100 dur=20 {attrs}"
            for name, attrs in records
        )
    )
    assert feedback._submission_evidence(tmp_path, [7], completed_run_ids=(7,))[0]["run_epoch"] == 4
    with pytest.raises(AssertionError, match="launch unproven"):
        feedback._submission_evidence(tmp_path, [7])
    rows = feedback._submission_evidence(tmp_path, [7], require_all=False)
    assert rows[0]["enqueued"] is False


def test_reused_host_logit_frame_missing_write_fails_before_reuse(chain_runtime):
    worker, arguments = chain_runtime
    arguments.steps = 5
    worker.skip_logit_at = 2
    with pytest.raises(AssertionError, match="logits failed at step 2"):
        feedback.main()
    assert worker.next_id == 5
