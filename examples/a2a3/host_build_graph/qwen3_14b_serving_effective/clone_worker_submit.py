# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qualify independent Qwen requests through the public Worker.submit path."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import torch
from callable_bridge import inspect_artifact
from real_worker_submit import BATCH, PADDED_VOCAB, _bind, _view
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
from standalone_adapter import StandaloneDecodeAdapter

from simpler_setup.tools.strace_timing import parse_spans

_HOST_INPUTS = ("seq_lens", "slot_mapping", "block_table", "sampled_ids_in")
_PRIVATE = (
    "k_cache",
    "v_cache",
    "out",
    "sampled_ids",
    "next_hidden",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--kv-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--steps", type=int, default=127)
    return parser.parse_args()


def _private_storage(devices):
    return {
        name: (devices[name].identity.buffer_id, int(devices[name].base), int(devices[name].nbytes))
        for name in _PRIVATE
    }


def _assert_disjoint_storage(first, second, label):
    identities = {identity for identity, _, _ in first.values()}
    if identities & {identity for identity, _, _ in second.values()}:
        raise AssertionError(label)
    if any(
        left_base < right_base + right_bytes and right_base < left_base + left_bytes
        for _, left_base, left_bytes in first.values()
        for _, right_base, right_bytes in second.values()
    ):
        raise AssertionError(label)


def _assert_private_storage(first, second):
    first_storage, second_storage = _private_storage(first), _private_storage(second)
    _assert_disjoint_storage(first_storage, second_storage, "live requests share request-private storage")
    return first_storage, second_storage


def _assert_private_host_storage(first, second):
    first_storage = _private_host_storage(first)
    second_storage = _private_host_storage(second)
    _assert_disjoint_storage(first_storage, second_storage, "live requests share request-private host storage")
    return first_storage, second_storage


def _private_host_storage(hosts):
    return {
        name: (hosts[name].identity.buffer_id, int(hosts[name].base), int(hosts[name].nbytes)) for name in _HOST_INPUTS
    }


def _prepare_step(hosts, adapter):
    step = adapter.next_step()
    values = {
        "seq_lens": step.seq_lens,
        "slot_mapping": step.slot_mapping,
        "block_table": step.block_table.reshape(-1),
        "sampled_ids_in": torch.zeros((BATCH, 8), dtype=torch.int32),
    }
    values["sampled_ids_in"][:, 0] = step.input_token_ids
    for name, value in values.items():
        _view(hosts[name], tuple(value.shape), value.dtype).copy_(value)
    return step


def _reset_sampled_ids(host, device, copy_to):
    sampled_ids = _view(host, (BATCH, 8), torch.int32)
    sampled_ids.fill_(-1)
    copy_to(device, host)


def _run_decode_pairs(steps, prepare_pair, submit, finish_pair):
    for _ in range(steps):
        prepare_pair()
        first = (0, *submit(0))
        if first[2].done:
            raise RuntimeError("first request completed before the second request could be submitted")
        second = (1, *submit(1))
        finish_pair((first, second))


def _source_head(root):
    try:
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _attributes(span):
    return dict(part.split("=", 1) for part in span.attrs.split() if "=" in part)


def _first_join_evidence(spans, run_ids):
    runs = {}
    boundaries = {}
    joins = []
    for span in spans:
        if span.name not in {"chip.run", "chip.run.runner_run.device_boundaries", "chip.run.joined_launch"}:
            continue
        fields = _attributes(span)
        if span.name == "chip.run" and int(fields.get("run_id", -1)) in run_ids:
            runs[int(fields["run_id"])] = (span.pid, span.inv, int(fields["dispatch_id"]), int(fields["slot_id"]))
        elif span.name == "chip.run.runner_run.device_boundaries":
            boundaries[span.pid, span.inv] = fields
        elif span.name == "chip.run.joined_launch":
            joins.append((span.pid, fields))
    if len(runs) != 2:
        return None
    first, second = (runs[run_id] for run_id in run_ids)
    if first[0] != second[0]:
        raise AssertionError("requests ran on different chip children")
    for pid, fields in joins:
        if (
            pid == first[0]
            and int(fields.get("p_disp", -1)) == first[2]
            and int(fields.get("p_slot", -1)) == first[3]
            and int(fields.get("s_disp", -1)) == second[2]
            and int(fields.get("s_slot", -1)) == second[3]
            and fields.get("observed") == "1"
            and fields.get("unfired") == "1"
            and fields.get("rc") == "0"
        ):
            predecessor = boundaries.get(first[:2])
            successor = boundaries.get(second[:2])
            if predecessor is None or successor is None:
                return None
            if predecessor.get("wo_rc") != "0" or successor.get("aic_rc") != "0":
                raise AssertionError("device boundary timestamps are unavailable")
            if predecessor.get("dev_id") != successor.get("dev_id") or predecessor.get("ts_hz") != successor.get(
                "ts_hz"
            ):
                raise AssertionError("request device clocks are not comparable")
            end, start = int(predecessor["wo_end"]), int(successor["aic_start"])
            if start < end:
                raise AssertionError("successor operator overlapped its predecessor")
            return {
                "predecessor_dispatch": first[2],
                "successor_dispatch": second[2],
                "serial_device_ticks": start - end,
            }
    return None


def _await_joined_evidence(log_directory, child_pids, run_pairs, timeout=10.0):
    deadline = time.monotonic() + timeout
    offsets = {pid: 0 for pid in child_pids}
    spans = []
    pending = list(run_pairs)
    evidence = {}
    while True:
        for pid in child_pids:
            path = log_directory / f"host.{pid}.log"
            try:
                blob = path.read_bytes()
            except OSError:
                continue
            window = blob[offsets[pid] :]
            consumed = window.rfind(b"\n") + 1
            if consumed > 0:
                offsets[pid] += consumed
                spans.extend(parse_spans(window[:consumed].decode("utf-8", errors="replace").splitlines()))
        for run_ids in tuple(pending):
            pair_evidence = _first_join_evidence(spans, run_ids)
            if pair_evidence is not None:
                evidence[run_ids] = pair_evidence
                pending.remove(run_ids)
        if not pending:
            ticks = [item["serial_device_ticks"] for item in evidence.values()]
            return {
                "pair_count": len(evidence),
                "first": evidence[run_pairs[0]],
                "serial_device_ticks_min": min(ticks),
                "serial_device_ticks_max": max(ticks),
            }
        if time.monotonic() >= deadline:
            raise AssertionError(f"joined native launch evidence missing for {len(pending)} request pairs")
        time.sleep(0.05)


def main() -> int:  # noqa: PLR0912, PLR0915 -- one Worker owns the two request lifetimes
    args = _parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    host_log_directory = args.output / "host"
    host_log_directory.mkdir()
    from _task_interface import _set_host_log_directory  # noqa: PLC0415

    _set_host_log_directory(str(host_log_directory))
    report = {"status": "running", "launch_depth": 2, "steps_requested": args.steps, "requests": [[], []]}
    worker = None
    error = None
    hosts = []
    child_pids = []
    log_directory = None
    try:
        torch.set_num_threads(8)
        fixture = ReferenceFixture(args.fixture)
        if not 1 <= args.steps <= fixture.steps:
            raise ValueError("steps must be within the frozen reference")
        fixture.verify_model(args.model)
        reference_digest = hashlib.sha256((fixture.root / "SHA256SUMS").read_bytes()).hexdigest()
        kv_reference = load_kv_reference(args.kv_reference, reference_digest)
        artifact = inspect_artifact(args.artifact)
        chip = artifact.compile()
        report["reference_sums_sha256"] = reference_digest
        report["source_hashes"] = artifact.source_hashes
        repo_root = Path(__file__).resolve().parents[4]
        report["code_head"] = _source_head(repo_root)
        specs = json.loads((args.artifact / "distributed_meta.json").read_text())["params"]
        directions = {
            p["name"].split("__ssa_", 1)[0]: (
                TensorArgType.INPUT if p["direction"] == "In" else TensorArgType.OUTPUT_EXISTING
            )
            for p in specs
        }
        worker = Worker(
            level=3,
            platform="a2a3",
            runtime="host_build_graph",
            device_ids=[args.device],
            num_sub_workers=0,
            launch_depth=2,
        )
        chip_handle = worker.register(chip)
        worker.init()
        from _task_interface import _host_log_directory  # noqa: PLC0415
        from simpler.task_interface import _bind_host_log_session_directory  # noqa: PLC0415

        log_directory = Path(_host_log_directory() or _bind_host_log_session_directory())
        child_pids = list(worker._chip_pids)
        devices = [{}, {}]
        hosts = [{}, {}]
        layer_cache = {}
        allocate_resident(
            worker, devices[0], hosts[0], fixture, args.model, layer_cache=layer_cache, host_staged=_HOST_INPUTS
        )
        shared = {name: device for name, device in devices[0].items() if name not in _PRIVATE}
        required = sum(int(devices[0][name].nbytes) for name in _PRIVATE)
        available = int(worker.device_memory_info().free_bytes)
        report["second_request_minimum_bytes"] = required
        report["free_bytes_before_second_request"] = available
        if available < required:
            raise MemoryError(f"second request needs at least {required} free bytes; device reports {available}")
        allocate_resident(
            worker,
            devices[1],
            hosts[1],
            fixture,
            args.model,
            shared_devices=shared,
            layer_cache=layer_cache,
            host_staged=_HOST_INPUTS,
        )
        layer_cache.clear()
        storage = _assert_private_storage(*devices)
        host_storage = _assert_private_host_storage(*hosts)
        report["private_storage"] = [
            {
                name: {"buffer_id": identity, "base": base, "nbytes": nbytes}
                for name, (identity, base, nbytes) in request.items()
            }
            for request in storage
        ]
        report["private_host_storage"] = [
            {
                name: {"buffer_id": identity, "base": base, "nbytes": nbytes}
                for name, (identity, base, nbytes) in request.items()
            }
            for request in host_storage
        ]
        adapter_fixture = fixture.adapter_fixture()
        adapters = [StandaloneDecodeAdapter(adapter_fixture) for _ in range(2)]
        config = CallConfig()
        config.enable_dep_gen = False
        config.enable_chip_swimlane = 0
        config.output_prefix = str(args.output)

        def prepare_pair():
            for request in range(2):
                _reset_sampled_ids(hosts[request]["sampled_ids"], devices[request]["sampled_ids"], worker.copy_to)

        def submit(request):
            request_devices = devices[request]
            request_hosts = hosts[request]
            step = _prepare_step(request_hosts, adapters[request])
            task_buffers = dict(request_devices)
            task_buffers.update({name: request_hosts[name] for name in _HOST_INPUTS})
            task_args = _bind(specs, task_buffers, directions=directions, cache_pages=fixture.num_pages)

            def dispatch(orch, _args, cfg):
                orch.submit_next_level(chip_handle, task_args, cfg, worker=0)

            handle = worker.submit(dispatch, args=None, config=config)
            entry = {"step": step.index, "run_id": handle._run_id, "input": step.input_token_ids.tolist()}
            report["requests"][request].append(entry)
            return step, handle, entry

        def finish_pair(pair):
            for request, step, handle, entry in pair:
                handle.result(timeout=120)
            for request, step, _handle, entry in pair:
                for name in ("sampled_ids", "out"):
                    worker.copy_from(hosts[request][name], devices[request][name])
                sampled = _view(hosts[request]["sampled_ids"], (BATCH, 8), torch.int32).clone()
                logits = _view(hosts[request]["out"], (BATCH, PADDED_VOCAB), torch.float32).clone()
                expected_logits = fixture.golden["logits"][step.index]
                checks = [
                    metrics(
                        row[: expected_logits.shape[-1]],
                        expected_logits[i] if expected_logits.ndim == 2 else expected_logits,
                    )
                    for i, row in enumerate(logits)
                ]
                entry["sampled"] = sampled[:, 0].tolist()
                entry["logits"] = checks
                entry["logits_passed"] = all(
                    m["finite"] and m["relative_l2"] <= LOGIT_RELATIVE_L2_MAX and m["cosine"] >= COSINE_MIN
                    for m in checks
                )
                adapters[request].complete_step(sampled)

        _run_decode_pairs(args.steps, prepare_pair, submit, finish_pair)
        run_pairs = tuple((first["run_id"], second["run_id"]) for first, second in zip(*report["requests"]))
        report["native_joined"] = _await_joined_evidence(log_directory, child_pids, run_pairs)
        report["first_in_flight_after_second_submit"] = True
        prefix_cache = {}
        report["kv_passed"] = []
        report["kv_failures"] = []
        for request in range(2):
            checks = validate_kv(worker, devices[request], fixture, kv_reference, args.steps, prefix_cache=prefix_cache)
            passed = all(check["passed"] for check in checks)
            report["kv_passed"].append(passed)
            report["kv_failures"].append(
                [{"layer": check["layer"], "kind": check["kind"]} for check in checks if not check["passed"]]
            )
    except BaseException as exc:
        error = exc
        report["status"] = "capacity_blocked" if isinstance(exc, MemoryError) else "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if worker is not None:
            try:
                worker.close()
            except BaseException as exc:
                report["cleanup_error"] = f"{type(exc).__name__}: {exc}"
                if error is None:
                    error = exc
                    report["status"] = "failed"
        if "native_joined" not in report and log_directory is not None:
            report["native_join_error"] = "native trace could not be read before cleanup"
        if error is None and report.get("status") == "running":
            failed_logits = [
                (request, entry["step"])
                for request, entries in enumerate(report["requests"])
                for entry in entries
                if not entry["logits_passed"]
            ]
            if failed_logits:
                error = AssertionError(f"logits failed for requests {failed_logits}")
            elif not report.get("native_joined"):
                error = AssertionError("native joined-launch evidence is missing")
            elif report.get("kv_passed") != [True, True]:
                error = AssertionError("appended KV failed for one or more requests")
        if error is None:
            report["status"] = "passed"
        elif report.get("status") == "running":
            report["status"] = "failed"
        for request_hosts in hosts:
            for host in request_hosts.values():
                try:
                    host.close()
                except BaseException as exc:
                    report["cleanup_error"] = f"{type(exc).__name__}: {exc}"
                    if error is None:
                        error = exc
                        report["status"] = "failed"
        if error is not None and "error" not in report:
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
