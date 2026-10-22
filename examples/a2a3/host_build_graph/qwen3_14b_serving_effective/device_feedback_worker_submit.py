# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qualify a bounded Qwen DEVICE feedback chain through Worker.submit."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from callable_bridge import _sha256, inspect_artifact
from clone_worker_submit import (
    JoinedEvidenceError,
    _assert_disjoint_storage,
    _attributes,
    _await_joined_evidence,
    _source_head,
)
from real_worker_submit import BATCH, HIDDEN, PADDED_VOCAB, _bind, _host_buffer, _view
from reference_fixture import ReferenceFixture
from reference_worker_submit import (
    COSINE_MIN,
    LOGIT_RELATIVE_L2_MAX,
    allocate_resident,
    load_kv_reference,
    metrics,
    validate_kv,
)
from simpler.task_interface import CallConfig, TensorArgType
from simpler.worker import Worker

from simpler_setup.tools.strace_timing import parse_spans
from simpler_setup.torch_interop import torch_dtype_to_datatype

_METADATA = ("seq_lens", "slot_mapping", "block_table")
LAUNCH_DEPTH = 2
FRAME_COUNT = LAUNCH_DEPTH


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fixture", "artifact", "model", "kv-reference", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--steps", type=int, default=127)
    return parser.parse_args()


def _feedback_buffers(resident, outputs, metadata, previous_output=None):
    buffers = dict(resident)
    buffers.update(metadata)
    buffers.update(outputs)
    buffers["sampled_ids_in"] = (
        resident["sampled_ids_in"] if previous_output is None else previous_output["sampled_ids"]
    )
    return buffers


def _run_device_chain(steps, prepare, submit, finish, *, depth=LAUNCH_DEPTH, completed=None):
    if steps < 1 or depth not in (1, 2):
        raise ValueError("require positive steps and bounded depth one or two")
    pending = []
    run_ids = []
    completed = set() if completed is None else completed
    try:
        for index in range(steps):
            if len(pending) == depth:
                predecessor_index, predecessor = pending[0]
                predecessor.result(timeout=120)
                completed.add(predecessor_index)
                finish(predecessor_index, predecessor)
                pending.pop(0)
            bound_args = prepare(index)
            handle = submit(index, bound_args)
            run_ids.append(handle._run_id)
            pending.append((index, handle))
        for index, handle in pending:
            handle.result(timeout=120)
            completed.add(index)
            finish(index, handle)
    except BaseException as error:
        # Already submitted consumers retain their buffers until their own drain or Worker.close.
        for index, handle in pending:
            if index in completed:
                continue
            try:
                handle.result(timeout=120)
                completed.add(index)
            except BaseException:
                pass
        raise error
    return run_ids


def _check_sampled(sampled, expected_tokens, index):
    if not torch.equal(sampled[:, 0], expected_tokens):
        raise AssertionError(f"sampled token mismatch at step {index}")
    return sampled[:, 0].tolist()


def _check_logits(logits, expected_logits, index):
    checks = [
        metrics(row[: expected_logits.shape[-1]], expected_logits[i] if expected_logits.ndim == 2 else expected_logits)
        for i, row in enumerate(logits)
    ]
    if not all(m["finite"] and m["relative_l2"] <= LOGIT_RELATIVE_L2_MAX and m["cosine"] >= COSINE_MIN for m in checks):
        raise AssertionError(f"logits failed at step {index}")
    return checks


def _allocate_frames(worker, resident, resident_hosts, fixture, steps, hosts):
    # Tokens stay resident through their device consumer and the final host audit.
    outputs = [{"out": resident_hosts["out"], "next_hidden": resident["next_hidden"]}]
    sampled = [resident["sampled_ids"]]
    required = (steps - 1) * int(sampled[0].nbytes) + (FRAME_COUNT - 1) * int(resident["next_hidden"].nbytes)
    available = int(worker.device_memory_info().free_bytes)
    if required > available:
        raise MemoryError(f"retained step outputs need at least {required} free bytes; device reports {available}")
    metadata = [{name: resident_hosts[name] for name in _METADATA}]
    sample = fixture.step(0)
    for _index in range(1, FRAME_COUNT):
        out = _host_buffer(worker, torch.full((BATCH, PADDED_VOCAB), float("nan")))
        hosts.append(out)
        row = {
            "out": out,
            "next_hidden": worker.alloc_child_tensor(0, (BATCH, HIDDEN), torch_dtype_to_datatype(torch.bfloat16).value),
        }
        outputs.append(row)
        step_hosts = {}
        for name in _METADATA:
            value = sample[name].reshape(-1) if name == "block_table" else sample[name]
            host = _host_buffer(worker, value)
            hosts.append(host)
            step_hosts[name] = host
        metadata.append(step_hosts)
    sentinel = resident_hosts["sampled_ids"]
    _view(sentinel, (BATCH, 8), torch.int32).fill_(-1)
    worker.copy_to(sampled[0], sentinel)
    for _index in range(1, steps):
        buffer = worker.alloc_child_tensor(0, (BATCH, 8), torch_dtype_to_datatype(torch.int32).value)
        sampled.append(buffer)
        worker.copy_to(buffer, sentinel)
    host_rows = [{**source, "out": output["out"]} for source, output in zip(metadata, outputs)]
    device_rows = [{"next_hidden": row["next_hidden"]} for row in outputs]
    device_rows.extend({"sampled_ids": buffer} for buffer in sampled)
    for rows in (host_rows, device_rows):
        storage = []
        for row in rows:
            current = {
                name: (buffer.identity.buffer_id, int(buffer.base), int(buffer.nbytes)) for name, buffer in row.items()
            }
            for previous in storage:
                _assert_disjoint_storage(previous, current, "decode frames share storage")
            storage.append(current)
    return outputs, metadata, sampled


def _submission_evidence(log_directory, run_ids, *, completed_run_ids=(), require_all=True):
    frames, runs, launches = {}, {}, set()
    for path in log_directory.glob("host.*.log"):
        with path.open() as stream:
            for span in parse_spans(stream):
                if span.name == "node.frame_submit":
                    fields = _attributes(span)
                    frames[int(fields["run_id"])] = fields
                elif span.name == "chip.run":
                    fields = _attributes(span)
                    runs[int(fields["run_id"])] = (span.pid, span.inv, fields)
                elif span.name == "chip.run.runner_run.aicpu_launch":
                    launches.add((span.pid, span.inv))
    rows = []
    for run_id in run_ids:
        if run_id not in frames or run_id not in runs:
            if require_all:
                raise AssertionError(f"submission/completion trace missing for run {run_id}")
            continue
        pid, inv, fields = runs[run_id]
        if frames[run_id]["dispatch_id"] != fields["dispatch_id"]:
            raise AssertionError(f"native launch identity missing for run {run_id}")
        enqueued = run_id in completed_run_ids and (pid, inv) in launches
        if require_all and not enqueued:
            raise AssertionError(f"successful native launch unproven for run {run_id}")
        rows.append(
            {
                "run_id": run_id,
                "dispatch_id": int(fields["dispatch_id"]),
                "slot_id": int(fields["slot_id"]),
                "generation": int(fields["generation"]),
                "run_epoch": int(fields["run_epoch"]),
                "prepare_only": int(frames[run_id]["prepare_only"]),
                "enqueued": enqueued,
            }
        )
    return rows


def _feedback_evidence(log_directory, child_pids, run_ids):
    pairs = tuple(zip(run_ids, run_ids[1:]))
    if not pairs:
        return {"joined_count": 0, "refill_joined_count": 0, "fallback_count": 0, "fallback_edges": []}
    evidence = _await_joined_evidence(log_directory, child_pids, pairs, allow_serial=True)
    serial = set(evidence["serial_pairs"])
    edges = [
        {
            "predecessor": first,
            "successor": second,
            "reason": "serial_launch",
        }
        for first, second in pairs
        if (first, second) in serial
    ]
    return {
        "native_execution": evidence,
        "joined_count": evidence["pair_count"],
        "refill_joined_count": sum(pair not in serial for pair in pairs[1:]),
        "fallback_count": len(serial),
        "fallback_edges": edges,
    }


def main():  # noqa: PLR0912, PLR0915 -- one Worker owns the bounded chain and its retained results
    args = _parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    log_directory = None
    worker = None
    hosts = []
    error = None
    completed = set()
    report = {
        "status": "running",
        "steps_requested": args.steps,
        "launch_depth": LAUNCH_DEPTH,
        "frame_count": FRAME_COUNT,
        "accepted_count": 0,
        "staged_count": None,
        "enqueued_count": None,
        "completed_count": 0,
        "retired_count": 0,
        "steps": [],
        "run_ids": [],
    }
    try:
        torch.set_num_threads(8)
        fixture = ReferenceFixture(args.fixture)
        if not 1 <= args.steps <= fixture.steps:
            raise ValueError("steps must be within the frozen reference")
        fixture.verify_model(args.model)
        reference_digest = _sha256(fixture.root / "SHA256SUMS")
        kv_reference = load_kv_reference(args.kv_reference, reference_digest)
        artifact = inspect_artifact(args.artifact)
        chip = artifact.compile(seed_outputs=("out",))
        report.update(
            code_head=_source_head(Path(__file__).resolve().parents[4]),
            source_hashes=artifact.source_hashes,
            reference_sums_sha256=reference_digest,
            adjacent_pair_count=args.steps - 1,
            consumer_source_sha256={
                name: _sha256(Path(__file__).with_name(name))
                for name in (
                    Path(__file__).name,
                    "clone_worker_submit.py",
                    "reference_worker_submit.py",
                    "callable_bridge.py",
                )
            },
        )
        specs = json.loads((args.artifact / "distributed_meta.json").read_text())["params"]
        directions = {
            p["name"].split("__ssa_", 1)[0]: TensorArgType.INPUT
            if p["direction"] == "In"
            else TensorArgType.OUTPUT_EXISTING
            for p in specs
        }
        directions["out"] = TensorArgType.INOUT
        worker = Worker(
            level=3,
            output_root=args.output,
            platform="a2a3",
            runtime="host_build_graph",
            device_ids=[args.device],
            num_sub_workers=0,
            launch_depth=LAUNCH_DEPTH,
        )
        chip_handle = worker.register(chip)
        worker.init()
        from _task_interface import _host_log_directory  # noqa: PLC0415
        from simpler.task_interface import _bind_host_log_session_directory  # noqa: PLC0415

        log_directory = Path(_host_log_directory() or _bind_host_log_session_directory())
        child_pids = list(worker._chip_pids)
        report["host_trace"] = {"directory": str(log_directory), "child_pids": child_pids}
        resident, resident_hosts = {}, {}
        try:
            allocate_resident(worker, resident, resident_hosts, fixture, args.model, host_staged=(*_METADATA, "out"))
        finally:
            hosts.extend(resident_hosts.values())
        initial = _view(resident_hosts["sampled_ids_in"], (BATCH, 8), torch.int32)
        initial.zero_()
        initial[:, 0] = fixture.metadata["first_token_id"].repeat(BATCH)
        worker.copy_to(resident["sampled_ids_in"], resident_hosts["sampled_ids_in"])
        outputs, metadata, sampled_outputs = _allocate_frames(
            worker, resident, resident_hosts, fixture, args.steps, hosts
        )
        report["retained_token_bytes"] = sum(int(buffer.nbytes) for buffer in sampled_outputs)
        report["device_frame_bytes"] = sum(int(row["next_hidden"].nbytes) for row in outputs)
        report["host_logit_bytes"] = sum(int(row["out"].nbytes) for row in outputs)
        report["metadata_bytes"] = sum(int(host.nbytes) for row in metadata for host in row.values())
        config = CallConfig()
        config.enable_dep_gen = False
        config.enable_chip_swimlane = 0
        report["feedback_buffers"] = [
            {"buffer_id": buffer.identity.buffer_id, "base": int(buffer.base)} for buffer in sampled_outputs
        ]

        def prepare(index):
            frame = index % FRAME_COUNT
            output = {**outputs[frame], "sampled_ids": sampled_outputs[index]}
            previous_output = None if index == 0 else {"sampled_ids": sampled_outputs[index - 1]}
            _view(output["out"], (BATCH, PADDED_VOCAB), torch.float32).fill_(float("nan"))
            values = fixture.step(index)
            for name in _METADATA:
                value = values[name].reshape(-1) if name == "block_table" else values[name]
                _view(metadata[frame][name], tuple(value.shape), value.dtype).copy_(value)
            buffers = _feedback_buffers(resident, output, metadata[frame], previous_output)
            return _bind(specs, buffers, directions=directions, cache_pages=fixture.num_pages)

        def submit(index, bound_args):
            def dispatch(orch, _args, cfg):
                orch.submit_next_level(chip_handle, bound_args, cfg, worker=0)

            handle = worker.submit(dispatch, args=None, config=config)
            report["accepted_count"] += 1
            report["run_ids"].append(handle._run_id)
            if handle._run_id is None:
                raise AssertionError("Worker.submit did not expose a native run id")
            return handle

        def finish(index, _handle):
            frame = index % FRAME_COUNT
            logits = _view(outputs[frame]["out"], (BATCH, PADDED_VOCAB), torch.float32)
            row = {"step": index, "logits": _check_logits(logits, fixture.golden["logits"][index], index)}
            row["run_id"] = report["run_ids"][index]
            row["frame"] = frame
            report["steps"].append(row)

        started = time.perf_counter()
        run_ids = _run_device_chain(args.steps, prepare, submit, finish, depth=LAUNCH_DEPTH, completed=completed)
        report["chain_wall_s"] = time.perf_counter() - started
        report["run_ids"] = run_ids
        if any(run_id is None for run_id in run_ids):
            raise AssertionError("Worker.submit did not expose a native run id for every step")
        report.update(_feedback_evidence(log_directory, child_pids, run_ids))
        if args.steps > 1 and report["joined_count"] == 0:
            raise AssertionError("supported DEVICE feedback chain fell back for every pair")
        if args.steps > LAUNCH_DEPTH and report["refill_joined_count"] == 0:
            raise AssertionError("DEVICE feedback did not join after the initial submissions")
        for index, output in enumerate(sampled_outputs):
            worker.copy_from(resident_hosts["sampled_ids"], output)
            sampled = _view(resident_hosts["sampled_ids"], (BATCH, 8), torch.int32)
            expected_tokens = fixture.golden["decode_output_token_ids"][index].repeat(BATCH)
            report["steps"][index]["sampled"] = _check_sampled(sampled, expected_tokens, index)
        report["submission_trace"] = _submission_evidence(log_directory, run_ids, completed_run_ids=run_ids)
        report["enqueued_count"] = len(report["submission_trace"])
        report["staged_count"] = sum(row["prepare_only"] for row in report["submission_trace"])
        checks = validate_kv(worker, resident, fixture, kv_reference, args.steps)
        report["kv_failures"] = [{"layer": c["layer"], "kind": c["kind"]} for c in checks if not c["passed"]]
        if report["kv_failures"]:
            raise AssertionError("DEVICE feedback appended KV failed")
        report["status"] = "passed"
    except BaseException as exc:
        error = exc
        report["status"] = "capacity_blocked" if isinstance(exc, MemoryError) else "failed"
        if isinstance(exc, JoinedEvidenceError):
            report["evidence_missing_pairs"] = exc.missing_pairs
    finally:
        report["completed_count"] = len(completed)
        worker_closed = False
        if worker is not None:
            try:
                worker.close()
                worker_closed = True
            except BaseException as exc:
                report["cleanup_error"] = f"{type(exc).__name__}: {exc}"
                if error is None:
                    error = exc
        for host in hosts if worker is None or worker_closed else ():
            try:
                host.close()
            except BaseException as exc:
                report["cleanup_error"] = f"{type(exc).__name__}: {exc}"
                if error is None:
                    error = exc
        report["worker_closed"] = worker_closed
        if worker_closed:
            report["retired_count"] = report["accepted_count"]
            report["retirement_basis"] = "worker_close"
        if report["run_ids"] and report["enqueued_count"] is None:
            try:
                completed_ids = {report["run_ids"][index] for index in completed}
                rows = _submission_evidence(
                    log_directory, report["run_ids"], completed_run_ids=completed_ids, require_all=False
                )
                report["submission_trace"] = rows
                report["observed_enqueued_count"] = sum(row["enqueued"] for row in rows)
                report["observed_staged_count"] = sum(row["prepare_only"] for row in rows)
                if len(rows) == len(report["run_ids"]):
                    report["staged_count"] = report["observed_staged_count"]
                    if report["observed_enqueued_count"] == len(rows):
                        report["enqueued_count"] = len(rows)
            except BaseException as exc:
                report["submission_evidence_error"] = f"{type(exc).__name__}: {exc}"
        if error is not None:
            if report["status"] == "passed":
                report["status"] = "failed"
            report["error"] = f"{type(error).__name__}: {error}"
        try:
            (args.output / "result.json").write_text(json.dumps(report, indent=2))
        except BaseException as exc:
            if error is None:
                error = exc
            else:
                exc.__context__ = error.__context__
                error.__context__ = exc
    if error is not None:
        raise error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
