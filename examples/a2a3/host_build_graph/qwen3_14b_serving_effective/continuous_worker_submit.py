# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qualify bounded, staggered Qwen requests with continuous DEVICE feedback."""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Any

import torch
from callable_bridge import _sha256, inspect_artifact
from clone_worker_submit import _assert_disjoint_storage, _await_joined_evidence, _source_head
from device_feedback_worker_submit import (
    _METADATA,
    _allocate_frames,
    _check_logits,
    _check_sampled,
    _feedback_buffers,
    _submission_evidence,
)
from real_worker_submit import BATCH, PADDED_VOCAB, _bind, _view
from reference_fixture import ReferenceFixture
from reference_worker_submit import allocate_resident, load_kv_reference, validate_kv
from simpler.task_interface import CallConfig, TensorArgType
from simpler.worker import Worker

_PRIVATE = ("k_cache", "v_cache", "sampled_ids_in", "sampled_ids", "next_hidden")
_RETAINED = []


def _run_requests(lengths, start, submit, finish, checkpoint):
    """Keep two independent chains pending and replace requests at audited boundaries."""
    if len(lengths) < 3 or any(length < 2 for length in lengths):
        raise ValueError("require at least three requests with at least two steps each")
    waiting = deque(enumerate(lengths))
    live = {}
    pending = deque()
    next_slot = 0

    def admit(slot):
        request, length = waiting.popleft()
        state = {"request": request, "slot": slot, "length": length, "submitted": 0, "completed": 0}
        start(state)
        live[slot] = state

    def complete():
        state, step, handle = pending[0]
        handle.result(timeout=120)
        finish(state, step, handle)
        state["completed"] += 1
        pending.popleft()

    try:
        admit(0)
        admit(1)
        while live:
            finished = [state for state in live.values() if state["completed"] == state["length"]]
            if finished:
                while pending:
                    complete()
                finished = [state for state in live.values() if state["completed"] == state["length"]]
                checkpoint(tuple(live.values()), tuple(finished))
                for state in finished:
                    del live[state["slot"]]
                survivors = tuple(live)
                for slot in (0, 1):
                    if slot not in live and waiting:
                        admit(slot)
                next_slot = survivors[0] if survivors else next(iter(live), 0)
                continue
            candidates = [
                state
                for state in live.values()
                if state["submitted"] < state["length"] and state["submitted"] - state["completed"] < 2
            ]
            if len(pending) == 4 or not candidates:
                complete()
                continue
            state = next((state for state in candidates if state["slot"] == next_slot), candidates[0])
            for _ in range(
                min(
                    4 - len(pending),
                    2 - (state["submitted"] - state["completed"]),
                    state["length"] - state["submitted"],
                )
            ):
                step = state["submitted"]
                handle = submit(state, step)
                pending.append((state, step, handle))
                state["submitted"] += 1
            next_slot = 1 - state["slot"]
    except BaseException:
        for _state, _step, handle in pending:
            try:
                handle.result(timeout=120)
            except BaseException:
                pass
        raise


def _storage(buffers):
    return {name: (value.identity.buffer_id, int(value.base), int(value.nbytes)) for name, value in buffers.items()}


def _arrival_predecessor(rows, completed):
    if rows and rows[-1]["run_id"] not in completed:
        return rows[-1]["run_id"]
    return None


def _assert_slot_storage(slots):
    device, host = [], []
    for slot in slots:
        device.append(
            {
                **{name: slot["device"][name] for name in _PRIVATE},
                **{f"token_{i}": value for i, value in enumerate(slot["tokens"])},
                **{f"scratch_{i}": row["next_hidden"] for i, row in enumerate(slot["outputs"])},
            }
        )
        host.append(
            {
                **slot["host"],
                **{
                    f"metadata_{i}_{name}": value
                    for i, row in enumerate(slot["metadata"])
                    for name, value in row.items()
                },
                **{f"logits_{i}": row["out"] for i, row in enumerate(slot["outputs"])},
            }
        )
    for rows in (device, host):
        _assert_disjoint_storage(_storage(rows[0]), _storage(rows[1]), "live requests share private storage")


def _joined_refills(rows, serial_pairs):
    count = sum(
        second["refill"] and (first["run_id"], second["run_id"]) not in serial_pairs
        for first, second in zip(rows, rows[1:])
    )
    if not count:
        raise AssertionError("continuous refill has no joined evidence")
    return count


def _close_resources(worker, hosts, slots, report):
    report["worker_closed"] = False
    if worker is not None:
        try:
            worker.close()
        except BaseException:
            _RETAINED.append((worker, hosts, slots))
            raise
        report["worker_closed"] = True
    first_error = None
    for host in hosts:
        try:
            host.close()
        except BaseException as exc:
            first_error = first_error or exc
    if first_error is not None:
        raise first_error


def _restore(worker, slot, fixture):
    host = worker.create_buffer(int(slot["device"]["k_cache"].nbytes) // fixture.layers)
    try:
        for layer in range(fixture.layers):
            for kind, value in fixture.layer(layer):
                _view(host, tuple(value.shape), value.dtype).copy_(value)
                worker.copy_to(
                    slot["device"]["k_cache" if kind == "key" else "v_cache"],
                    host,
                    dst_offset=layer * int(host.nbytes),
                )
    finally:
        worker.release_buffer(host)
    sentinel = slot["host"]["sampled_ids"]
    _view(sentinel, (BATCH, 8), torch.int32).fill_(-1)
    for token in slot["tokens"]:
        worker.copy_to(token, sentinel)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fixture", "kv-reference", "artifact", "model", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--request-steps", type=int, nargs="+", default=[63, 127, 95, 127, 97, 127])
    parser.add_argument("--pipeline-depth", type=int, default=8)
    parser.add_argument("--launch-depth", type=int, choices=(2, 4), default=4)
    parser.add_argument("--inject-partial-at", type=int)
    return parser.parse_args()


def main():  # noqa: PLR0912, PLR0915 -- one Worker owns the request pool and its cleanup
    args = _parse_args()
    if len(args.request_steps) < 3 or min(args.request_steps) < 2 or args.pipeline_depth < 4:
        raise ValueError("require at least three multi-step requests and at least four resource sets")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "host").mkdir()
    from _task_interface import _set_host_log_directory  # noqa: PLC0415

    _set_host_log_directory(str(args.output / "host"))
    worker, error = None, None
    hosts, slots, runs, completed = [], [], [], set()
    handles, attempts = {}, []
    prefix_cache = {}
    rows, checkpoints, arrival_pairs = [], [], []
    report = {"status": "running", "requests": [], "runs": rows, "checkpoints": checkpoints}
    try:
        torch.set_num_threads(8)
        fixture = ReferenceFixture(args.fixture)
        if max(args.request_steps) > fixture.steps:
            raise ValueError("request length exceeds the frozen reference")
        fixture.verify_model(args.model)
        reference_digest = _sha256(args.fixture / "SHA256SUMS")
        expected_kv = load_kv_reference(args.kv_reference, reference_digest)
        artifact = inspect_artifact(args.artifact)
        chip = artifact.compile(seed_outputs=("out",))
        specs = json.loads((args.artifact / "distributed_meta.json").read_text())["params"]
        directions = {
            item["name"].split("__ssa_", 1)[0]: TensorArgType.INPUT
            if item["direction"] == "In"
            else TensorArgType.OUTPUT_EXISTING
            for item in specs
        }
        directions["out"] = TensorArgType.INOUT
        worker = Worker(
            level=3,
            platform="a2a3",
            runtime="host_build_graph",
            device_ids=[args.device],
            num_sub_workers=0,
            pipeline_depth=args.pipeline_depth,
            launch_depth=args.launch_depth,
        )
        chip_handle = worker.register(chip)
        worker.init()
        from _task_interface import _host_log_directory  # noqa: PLC0415

        log_directory = Path(_host_log_directory())
        shared = None
        for _ in range(2):
            slot: dict[str, Any] = {"device": {}, "host": {}}
            slots.append(slot)
            try:
                allocate_resident(
                    worker,
                    slot["device"],
                    slot["host"],
                    fixture,
                    args.model,
                    shared_devices=shared,
                    host_staged=(*_METADATA, "out"),
                )
            finally:
                hosts.extend(slot["host"].values())
            shared = {name: value for name, value in slot["device"].items() if name not in _PRIVATE}
            initial = _view(slot["host"]["sampled_ids_in"], (BATCH, 8), torch.int32)
            initial.zero_()
            initial[:, 0] = fixture.metadata["first_token_id"].repeat(BATCH)
            worker.copy_to(slot["device"]["sampled_ids_in"], slot["host"]["sampled_ids_in"])
            slot["outputs"], slot["metadata"], slot["tokens"] = _allocate_frames(
                worker,
                slot["device"],
                slot["host"],
                fixture,
                max(args.request_steps),
                hosts,
            )
        _assert_slot_storage(slots)
        config = CallConfig()
        config.enable_dep_gen = False
        config.enable_chip_swimlane = 0
        config.output_prefix = str(args.output)
        report.update(
            code_head=_source_head(Path(__file__).resolve().parents[4]),
            consumer_sha256=_sha256(Path(__file__)),
            pipeline_depth=args.pipeline_depth,
            launch_depth=args.launch_depth,
            request_slots=2,
            granted_sets=worker._chip_run_resource_set_counts,
            reference_sums_sha256=reference_digest,
            host_trace=str(log_directory),
            source_hashes=artifact.source_hashes,
        )

        def start(state):
            slot = slots[state["slot"]]
            if state["request"] >= 2:
                _restore(worker, slot, fixture)
            state["audited"] = 0
            state["rows"] = []
            report["requests"].append(
                {
                    "request_id": state["request"],
                    "slot": state["slot"],
                    "steps": state["length"],
                    "private_storage": _storage({name: slot["device"][name] for name in _PRIVATE}),
                    "status": "live",
                }
            )

        def submit(state, step):
            slot = slots[state["slot"]]
            frame = step % 2
            output = {**slot["outputs"][frame], "sampled_ids": slot["tokens"][step]}
            _view(output["out"], (BATCH, PADDED_VOCAB), torch.float32).fill_(float("nan"))
            values = fixture.step(step)
            for name in _METADATA:
                value = values[name].reshape(-1) if name == "block_table" else values[name]
                _view(slot["metadata"][frame][name], tuple(value.shape), value.dtype).copy_(value)
            previous = None if step == 0 else {"sampled_ids": slot["tokens"][step - 1]}
            bound = _bind(
                specs,
                _feedback_buffers(slot["device"], output, slot["metadata"][frame], previous),
                directions=directions,
                cache_pages=fixture.num_pages,
            )
            attempt = len(runs)
            attempt_row = {
                "request_id": state["request"],
                "step": step,
                "attempt": len(attempts),
                "status": "building",
            }
            attempts.append(attempt_row)
            predecessor = _arrival_predecessor(rows, completed) if step == 0 else None

            def dispatch(orch, _args, cfg):
                orch.submit_next_level(chip_handle, bound, cfg, worker=0)
                if attempt == args.inject_partial_at:
                    raise RuntimeError("injected partial Qwen submission")

            try:
                handle = worker.submit(dispatch, args=None, config=config)
            except BaseException as exc:
                attempt_row.update(
                    status="submission_failed",
                    error=f"{type(exc).__name__}: {exc}",
                    native_identity="unavailable_from_failed_submit",
                )
                raise
            attempt_row["status"] = "accepted"
            run_id = handle._run_id
            if run_id is None:
                raise AssertionError("submission has no native run identity")
            if predecessor is not None:
                arrival_pairs.append((predecessor, run_id))
            runs.append(run_id)
            handles[run_id] = handle
            row = {
                "request_id": state["request"],
                "step": step,
                "run_id": run_id,
                "slot": state["slot"],
                "frame": frame,
                "input_buffer": int((slot["device"]["sampled_ids_in"] if step == 0 else slot["tokens"][step - 1]).base),
                "output_buffer": int(slot["tokens"][step].base),
                "refill": state["completed"] > 0 and _arrival_predecessor(rows, completed) is not None,
            }
            rows.append(row)
            state["rows"].append(row)
            return handle

        def finish(state, step, _handle):
            row = state["rows"][step]
            completed.add(row["run_id"])
            row["retired"] = bool(_handle._cleanup_published)
            if not row["retired"]:
                raise AssertionError("completed run has unpublished cleanup")
            if step + 1 < state["submitted"]:
                try:
                    worker.free(slots[state["slot"]]["tokens"][step])
                except RuntimeError as exc:
                    if "still referenced" not in str(exc):
                        raise
                    row["last_consumer_release_refused"] = True
                else:
                    raise AssertionError("producer output freed before its DEVICE consumer retired")
            logits = _view(slots[state["slot"]]["outputs"][step % 2]["out"], (BATCH, PADDED_VOCAB), torch.float32)
            row["logits"] = _check_logits(logits, fixture.golden["logits"][step], step)

        def checkpoint(live, finished):
            for state in live:
                slot = slots[state["slot"]]
                for step in range(state["audited"], state["completed"]):
                    worker.copy_from(slot["host"]["sampled_ids"], slot["tokens"][step])
                    sampled = _view(slot["host"]["sampled_ids"], (BATCH, 8), torch.int32)
                    state["rows"][step]["sampled"] = _check_sampled(
                        sampled,
                        fixture.golden["decode_output_token_ids"][step].repeat(BATCH),
                        step,
                    )
                state["audited"] = state["completed"]
            for state in finished:
                checks = validate_kv(
                    worker,
                    slots[state["slot"]]["device"],
                    fixture,
                    expected_kv,
                    state["length"],
                    prefix_cache=prefix_cache,
                )
                report["requests"][state["request"]]["kv"] = checks
                if not all(check["passed"] for check in checks):
                    raise AssertionError(f"request {state['request']} KV mismatch")
                report["requests"][state["request"]]["status"] = "retired"
            checkpoints.append(
                {
                    "completed_runs": len(completed),
                    "retired_requests": [s["request"] for s in finished],
                    "live_requests": [s["request"] for s in live if s not in finished],
                    "boundary": "drained_result_audit_and_slot_reuse",
                }
            )

        _run_requests(args.request_steps, start, submit, finish, checkpoint)
        execution = _await_joined_evidence(
            log_directory, list(worker._chip_pids), tuple(zip(runs, runs[1:])), allow_serial=True
        )
        serial_pairs = {tuple(pair) for pair in execution["serial_pairs"]}
        replacements = {row["run_id"] for row in rows if row["request_id"] >= 2 and row["step"] == 0}
        if not any(second in replacements for _first, second in arrival_pairs):
            raise AssertionError("no replacement request arrived behind a live predecessor")
        if any(pair in serial_pairs for pair in arrival_pairs):
            raise AssertionError("an arriving request did not join its executing predecessor")
        report.update(native_execution=execution, arrivals={"pair_count": len(arrival_pairs), "pairs": arrival_pairs})
        feedback_pairs = tuple(
            (first["run_id"], second["run_id"])
            for first, second in zip(rows, rows[1:])
            if first["request_id"] == second["request_id"]
        )
        feedback_joined = sum(pair not in serial_pairs for pair in feedback_pairs)
        if not feedback_joined:
            raise AssertionError("no request-local DEVICE feedback edge joined")
        report.update(
            device_feedback={"pair_count": feedback_joined, "fallback_count": len(feedback_pairs) - feedback_joined},
            refill_joined_count=_joined_refills(rows, serial_pairs),
            fallback_count=len(serial_pairs),
            retirement_basis="run_cleanup_and_request_last_consumer_audit",
        )
        report["submission_trace"] = _submission_evidence(log_directory, runs, completed_run_ids=completed)
        report["staged_count"] = sum(row["prepare_only"] for row in report["submission_trace"])
        report["status"] = "passed"
    except BaseException as exc:
        error = exc
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        try:
            _close_resources(worker, hosts, slots, report)
        except BaseException as exc:
            report["cleanup_error"] = f"{type(exc).__name__}: {exc}"
            error = error or exc
        report.update(
            accepted_count=len(runs),
            enqueued_count=len(report.get("submission_trace", [])) if report.get("status") == "passed" else None,
            retired_requests=sum(row["status"] == "retired" for row in report["requests"]),
        )
        for row in rows:
            handle = handles[row["run_id"]]
            row["retired"] = bool(handle._cleanup_published)
            if handle._terminal:
                try:
                    handle.result(timeout=0)
                except BaseException as exc:
                    row["terminal_error"] = f"{type(exc).__name__}: {exc}"
                else:
                    completed.add(row["run_id"])
        report.update(
            attempts=attempts,
            completed_count=len(completed),
            retired_count=sum(row.get("retired", False) for row in rows),
            stop_admission=error is not None,
        )
        if error is not None and runs:
            try:
                observed = _submission_evidence(
                    log_directory,
                    runs,
                    completed_run_ids=completed,
                    require_all=False,
                )
                report.update(
                    submission_trace=observed,
                    observed_enqueued_count=sum(row["enqueued"] for row in observed),
                    observed_staged_count=sum(row["prepare_only"] for row in observed),
                    unobserved_run_ids=sorted(set(runs) - {row["run_id"] for row in observed}),
                )
            except BaseException as exc:
                report["submission_evidence_error"] = f"{type(exc).__name__}: {exc}"
        if error is not None:
            report["status"] = "failed"
            report.setdefault("error", f"{type(error).__name__}: {error}")
            for request in report["requests"]:
                if request["status"] != "retired":
                    request["status"] = "interrupted"
            for attempt in attempts:
                if attempt["status"] == "submission_failed":
                    report["requests"][attempt["request_id"]].update(
                        status="failed",
                        error=attempt["error"],
                    )
        (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    if error is not None:
        raise error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
