#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: PLC0415
"""Exercise Graph Calendar cache modes inside the frozen DSpark EP2 full decode."""

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import dspark_benchmark as dspark

ORCH = Path("next_levels/l2_decode_fwd_dspark/orchestration/l2_decode_fwd_dspark.cpp")
REGION = re.compile(
    r"(?P<collapse>    // Spmd hc_collapse_norm_spmd_10: hc_collapse_norm_10\n"
    r".*?    TaskOutputTensors task_207_outs = rt_submit_aiv_task\(160, params_t207\);\n)"
    r"(?P<between>    int64_t hidden_rows_inline315195 = .*?)"
    r"(?P<select>    // Spmd lm_head_select_rows_spmd_0: lm_head_select_rows_0\n"
    r".*?    TaskOutputTensors task_208_outs = rt_submit_aiv_task\(161, params_t208\);\n)"
    r"    TaskId selected_ready_inline315200 = task_208_outs\.task_id\(\);\n",
    re.S,
)
OUTPUTS = (
    "state_tokens",
    "state_meta",
    "active_widths",
    "accepted_ids",
    "accepted_counts",
    "num_sampled",
    "last_sampled",
    "draft_ids",
    "committed",
)
ACTIVE_REQUESTS = (1, 6, 11, 12)
TARGET_ROWS_PER_REQUEST = dspark.TARGET_ROWS // dspark.BATCH
MARKOV_ROWS_PER_REQUEST = 5
WALL_METRICS = ("p50_aicore_wall_us", "p95_aicore_wall_us")
REGRESSION_LIMITS: dict[str, float] = dict(zip(WALL_METRICS, (5.0, 10.0)))


def wrap_orchestration(source: str, mode: str) -> str:
    """Record only the local Markov head prologue before the existing TP push."""
    matches = list(REGION.finditer(source))
    if (
        len(matches) != 1
        or source.count('extern "C" {') != 1
        or source.count("task_208_outs") != 2
        or source.count("int64_t live_rows = ") != 1
        or f"int64_t markov_rows = (B_DYN * {MARKOV_ROWS_PER_REQUEST});" not in source
    ):
        raise ValueError("prepared artifact does not contain the expected DSpark Markov prologue")
    match = matches[0]
    tensors = (
        "ext_markov_final_norm_weight",
        "mix_flat_inline315109",
        "x_flat_inline315108",
        "markov_row_indices",
        "markov_flat_hidden",
        "markov_normalized",
        "selected_hidden_inline315199",
    )
    scalars = ("tokens_inline315107", "active_rows_inline315106", "hidden_rows_inline315195")
    body = match["collapse"] + "\n" + match["select"]
    if any(not re.search(rf"\b{name}\b", body) for name in (*tensors, *scalars)):
        raise ValueError("prepared Markov prologue argument bindings changed")
    # Shape determines the existing block count. The kernel still receives the three
    # invocation scalars by inheritance, including active_rows on every run.
    original_shape = "((tokens_inline315107 + 1) / 2)"
    if body.count(original_shape) != 1:
        raise ValueError("prepared Markov prologue launch spec changed")
    body = body.replace(original_shape, "((int64_t{graph.tensor(2).ref().shapes[0]} + 1) / 2)")
    values = {name: f"graph.tensor({i}).ref()" for i, name in enumerate(tensors)}
    values.update({name: f"graph.scalar({i})" for i, name in enumerate(scalars)})
    for name, value in values.items():
        body = re.sub(rf"\b{name}\b", value, body)
    body = re.sub(r"TaskOutputTensors task_20[78]_outs = ", "", body)
    helper = (
        "namespace {\nvoid submit_markov_head_prologue(const GraphTaskArgs &graph) {\n"
        + body
        + '}\n}  // namespace\n\nextern "C" {'
    )
    options = "" if mode == "exact" else ", simpler::hbg::GraphSubmitOptions{.calendar_hash = 0}"
    submit = (
        match["between"]
        + "    GraphTaskArgs markov_prologue;\n"
        + f"    markov_prologue.add_input({', '.join(tensors[:4])});\n"
        + f"    markov_prologue.add_output({', '.join(tensors[4:])});\n"
        + "    // The active requests form a prefix. Normalize their Markov rows and zero the padded rows.\n"
        + f"    int64_t active_rows = std::min<int64_t>({scalars[1]}, "
        + f"(live_rows / {TARGET_ROWS_PER_REQUEST}) * {MARKOV_ROWS_PER_REQUEST});\n"
        + f"    markov_prologue.add_scalar({scalars[0]}, active_rows, {scalars[2]});\n"
        + "    TaskId selected_ready_inline315200 =\n"
        + f"        rt_submit_graph(&submit_markov_head_prologue, markov_prologue{options}).task_id;\n"
    )
    modified = source[: match.start()] + submit + source[match.end() :]
    return modified.replace('extern "C" {', helper, 1)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def kernel_hashes(runtime_dir):
    cache = runtime_dir / ORCH.parents[1] / "cache"
    hashes = sorted(file_hash(path) for path in cache.glob("incore_*.bin"))
    if not hashes:
        raise ValueError(f"no cached business kernels in {cache}")
    return hashes


def prepare(args, daily):
    frozen = read_json(args.fixture / "prepared.json")
    if frozen["case"] != dspark.make_case(2) or set(frozen["artifact_runtimes"].values()) != {"percounter"}:
        raise ValueError("expected the frozen DSpark 8K EP2 Percounter fixture")
    if args.lib_commit != frozen["lib_commit"]:
        raise ValueError("use the Lib revision that produced the frozen fixture")
    original_build = Path(frozen["runtime_dir"])
    source = (original_build / ORCH).read_text(encoding="utf-8")
    wrapped = wrap_orchestration(source, args.mode)
    args.output.mkdir(parents=True, exist_ok=False)
    work_dir = args.scratch / args.output.name
    shutil.copytree(original_build, work_dir)
    (work_dir / ORCH).write_text(wrapped, encoding="utf-8")
    for name in ("frozen", "inputs.pkl"):
        (args.output / name).symlink_to((args.fixture / name).resolve(strict=True))
    meta = {
        **frozen,
        "runtime_dir": str(work_dir),
        "sampling": {"warmup": args.warmup, "rounds": args.rounds},
        "calendar_mode": args.mode,
        "fixture_source_sha256": file_hash(original_build / ORCH),
        "orchestration_sha256": file_hash(work_dir / ORCH),
        "fixture_inputs_sha256": file_hash(args.output / "inputs.pkl"),
    }
    dspark.prime(args, daily, meta)
    meta["business_kernel_sha256"] = kernel_hashes(work_dir)
    write_json(args.output / "prepared.json", meta)


@contextmanager
def gate_runtime_scope(runtime):
    from simpler_setup.runtime_builder import RuntimeBuilder

    original = RuntimeBuilder("a5").get_binaries("percounter")
    identity = {
        "runtime_sha256": {
            name: file_hash(path.resolve(strict=True)) for name, path in vars(original).items() if path is not None
        },
        "gate_sha256": None,
    }
    if runtime is None:
        yield identity
        return

    gate_tool = Path(__file__).resolve().parents[4] / "tests/fdwic/dsv41_scheduler/production/device_start_gate.py"
    spec = importlib.util.spec_from_file_location("dspark_device_start_gate", gate_tool)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load the device-start gate validator from {gate_tool}")
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    manifest = validator.validate_runtime_gate(runtime, "percounter")
    if any(validator.file_stamp(item["source"]) != item["stamp"] for item in manifest["original_binaries"].values()):
        raise ValueError("the original runtime changed after preparation of the device-start gate")
    expected = Path(manifest["wrappers"]["percounter"]["original"]).resolve()
    variant = runtime / "percounter"
    if original.host_path.resolve() != expected:
        raise ValueError(f"device-start gate wraps {expected}, but the worker selected {original.host_path}")
    snapshot = runtime.parents[1]
    identity["gate_sha256"] = {
        path: file_hash(snapshot / path)
        for path in manifest["snapshot_files"]
        if path.startswith("a5/onboard/percounter/")
        or path in ("a5/dispatcher/libsimpler_aicpu_dispatcher.so", "pto_isa_build.json")
    }
    get_binaries = RuntimeBuilder.get_binaries

    def gated_binaries(builder, name, build=False, **kwargs):
        if builder.platform != "a5" or name != "percounter":
            return get_binaries(builder, name, build=build, **kwargs)
        if build:
            raise ValueError("prepare the runtime and the device-start gate before allocating devices")
        binaries = get_binaries(builder, name, **kwargs)
        if binaries.host_path.resolve() != expected:
            raise ValueError(f"device-start gate wraps {expected}, but the worker selected {binaries.host_path}")
        return replace(
            binaries,
            host_path=variant / "libhost_runtime.so",
            aicpu_path=variant / "libaicpu_kernel.so",
            aicore_path=variant / "aicore_kernel.o",
            dispatcher_path=runtime.parent / "dispatcher/libsimpler_aicpu_dispatcher.so",
        )

    with patch.object(RuntimeBuilder, "get_binaries", gated_binaries):
        yield identity


def request_order(first):
    offset = ACTIVE_REQUESTS.index(first)
    return ACTIVE_REQUESTS[offset:] + ACTIVE_REQUESTS[:offset]


def run(args, daily, meta):
    import cloudpickle
    import golden
    import torch
    from golden import runner
    from pypto.ir import DistributedConfig
    from pypto.runtime import bench
    from utils import ENGRAM_SHARD_ROWS, EngramPayloadSource

    devices = [int(value) for value in os.environ["TASK_DEVICE"].split(",")]
    if len(devices) != 2 or len(set(devices)) != 2:
        raise ValueError("TASK_DEVICE must contain two distinct devices")
    sampling = meta["sampling"]
    requests = request_order(args.first_active_requests)
    gate_runtime = args.gate_runtime.resolve() if args.gate_runtime is not None else None
    attempt = args.output / f"first-{args.first_active_requests}"
    attempt.mkdir()
    write_json(attempt / "prepared.json", meta)
    os.environ.update(
        PYPTO_BENCH="1",
        PYPTO_BENCH_WARMUP=str(sampling["warmup"]),
        PYPTO_BENCH_ROUNDS=str(sampling["rounds"]),
        SIMPLER_DEVICE_STRACE_ENABLE="1",
    )
    os.chdir(args.output)
    entry = dspark.run_entry(args.lib, daily, meta["case"], devices, "dspark_calendar_entry")
    with (args.output / "inputs.pkl").open("rb") as stream:
        specs, config = cloudpickle.load(stream)
    config = dict(config, distributed_config=DistributedConfig(device_ids=devices, num_sub_workers=0))
    data = args.output / "frozen/data"
    frozen = dspark.load_frozen_inouts(args.lib / dspark.ENTRY, specs, data)
    ordered_args = runner._ordered_args
    original_resident = runner._run_l3_resident
    rounds = []
    outputs = {}
    slots = torch.arange(dspark.BATCH, dtype=torch.int32)

    class CaptureComplete(Exception):
        """Stop after raw collection and leave timing reduction outside the allocation."""

    def record_outputs(tensors):
        count = rounds[-1]
        result = {name: tensors[name].detach().clone() for name in OUTPUTS}
        reference = outputs.setdefault(count, result)
        if any(not torch.equal(value, reference[name]) for name, value in result.items()):
            raise ValueError(f"DSpark output changed between replays with {count} active requests")

    def restore(ordered_specs, tensors, *args, **kwargs):
        if kwargs.get("benchmark_dispatch_index") != len(rounds):
            raise ValueError("unexpected DSpark benchmark dispatch index")
        if rounds:
            record_outputs(tensors)
        elif not torch.equal(tensors["slot_ids"], slots.expand(len(devices), -1)):
            raise ValueError("expected the B12 fixture with all slots active")
        for name, value in frozen.items():
            tensors[name].view(torch.uint8).copy_(value)
        for name in OUTPUTS:
            if name not in frozen:
                tensors[name].zero_()
        count = requests[len(rounds) % len(requests)]
        tensors["slot_ids"][:, :count].copy_(slots[:count])
        tensors["slot_ids"][:, count:].fill_(-1)
        for name in ("attention_num_tokens", "moe_num_tokens"):
            tensors[name].fill_(count * TARGET_ROWS_PER_REQUEST)
        rounds.append(count)
        return ordered_args(ordered_specs, tensors, *args, **kwargs)

    def capture(log, **kwargs):
        (attempt / "benchmark.strace.log").write_text(log, encoding="utf-8")
        raise CaptureComplete

    def snapshot(compiled, specs, tensors, *call_args, **kwargs):
        try:
            return original_resident(compiled, specs, tensors, *call_args, **kwargs)
        except CaptureComplete:
            record_outputs(tensors)
            torch.save(outputs, attempt / "decode-outputs.pt")
            return None

    with (
        gate_runtime_scope(gate_runtime) as runtime_identity,
        daily.cached_native_only(),
        ExitStack() as sources,
        patch.object(runner, "_ordered_args", restore),
        patch.object(runner, "_run_l3_resident", snapshot),
        patch.object(bench, "_parse_stats_from_strace", capture),
        patch.object(bench, "_STRACE_LOG_LEVEL", "info"),
    ):
        windows = [
            sources.enter_context(EngramPayloadSource([(None, 0, 2 * ENGRAM_SHARD_ROWS * 256)])) for _ in devices
        ]
        started_at_ns = time.time_ns()
        result = golden.run(
            fn=entry["l3_decode_fwd_dspark"],
            specs=specs,
            config=config,
            golden_fn=None,
            golden_data=str(data),
            runtime_dir=meta["runtime_dir"],
            window_sources={"engram_payload": windows},
            persistent=True,
            reset_persistent_windows=False,
        )
    write_json(
        attempt / "device-status.json",
        {
            "devices": devices,
            "inout_resets": len(rounds),
            "passed": result.passed,
            "error": result.error,
            "active_requests_order": rounds,
            "runtime_identity": runtime_identity,
            "started_at_ns": started_at_ns,
            "command": sys.argv,
        },
    )
    if not result.passed or len(rounds) != sum(sampling.values()) or set(outputs) != set(requests):
        raise RuntimeError(result.error or "DSpark replay did not complete every frozen dispatch and collect outputs")


def wall_summary(samples):
    return {
        "p50_aicore_wall_us": statistics.median(samples),
        "p95_aicore_wall_us": statistics.quantiles(samples, n=20, method="inclusive")[-1],
    }


def record_rank_value(records, fields, value):
    invocation = int(fields["inv"])
    pid = int(fields["pid"])
    if pid in records[invocation]:
        raise ValueError(f"duplicate {fields['name']} record for invocation {invocation}, pid {pid}")
    records[invocation][pid] = value


def read_arm_trace(stream, kind):
    trace = SimpleNamespace(
        device_ids=defaultdict(dict),
        gates=defaultdict(dict),
        launch_ts=defaultdict(dict),
        aicore=defaultdict(dict),
        cache_events=defaultdict(Counter),
        training_us=defaultdict(dict),
        lookup_keys=defaultdict(list),
    )
    for line in stream:
        fields = dict(re.findall(r"(\w+)=([^\s]+)", line))
        name = fields.get("name")
        if "[DEVICE_START_GATE]" in line:
            epoch, device, peer = (int(fields[key]) for key in ("epoch", "device", "peer_epoch"))
            start, end = (int(fields[f"gate_{boundary}_cycles"]) for boundary in ("begin", "end"))
            if peer != epoch or start <= 0 or end < start or device in trace.gates[epoch]:
                raise ValueError(f"invalid gate rendezvous: {line.rstrip()}")
            trace.gates[epoch][device] = (int(fields["rank"]), end)
        elif "[STRACE]" in line:
            if name == "chip.run.runner_run.aicpu_launch":
                record_rank_value(trace.launch_ts, fields, int(fields["ts"]))
            elif name == "chip.run.runner_run.device_wall":
                record_rank_value(trace.device_ids, fields, int(fields["dev_id"]))
            elif name == "chip.run.runner_run.aicore_wall":
                values = tuple(int(fields[key]) for key in ("dur", "aicore_start_cycle", "aicore_cnt_hz"))
                record_rank_value(trace.aicore, fields, values)
        elif "A5 calendar: " in line and fields.get("kind") == kind:
            event = line.split("A5 calendar: ", 1)[1].split()[0]
            pid, key = int(fields["pid"]), fields["key"]
            if event == "STORE":
                trace.training_us[pid][key] = int(fields["training_us"])
            elif event in {"MISS", "HIT"}:
                if event == "HIT" and key not in trace.training_us[pid]:
                    raise ValueError(f"hit before successful PGO publication for pid {pid}, key {key}")
                trace.lookup_keys[pid].append(key)
            else:
                raise ValueError(f"unexpected graph cache event: {line.rstrip()}")
            trace.cache_events[pid][event] += 1
    return trace


def start_timing(trace, invocations):
    enqueue_skew = []
    offset_spread = []
    for invocation in invocations:
        launch = trace.launch_ts[invocation].values()
        enqueue_skew.append((max(launch) - min(launch)) / 1000.0)
        if trace.gates:
            offsets = []
            for pid, (_, start_tick, frequency) in trace.aicore[invocation].items():
                _, gate_end = trace.gates[invocation][trace.device_ids[invocation][pid]]
                if start_tick < gate_end or frequency <= 0:
                    raise ValueError(f"business AICore started before its gate at invocation {invocation}, pid {pid}")
                offsets.append((start_tick - gate_end) * 1e6 / frequency)
            offset_spread.append(max(offsets) - min(offsets))

    def distribution(values):
        return {
            "p50_us": statistics.median(values),
            "p95_us": statistics.quantiles(values, n=20, method="inclusive")[-1],
        }

    return {
        "gate_epochs_per_rank": len(trace.gates),
        "host_aicpu_enqueue_skew": distribution(enqueue_skew),
        "gate_end_to_aicore_start_offset_spread": distribution(offset_spread) if trace.gates else None,
    }


def parse_arm(output, *, require_start_gate=False):
    meta = read_json(output / "prepared.json")
    status = read_json(output / "device-status.json")
    warmup, rounds = (meta["sampling"][phase] for phase in ("warmup", "rounds"))
    total = rounds + warmup
    devices = set(status["devices"])
    if not status["passed"] or status["inout_resets"] != total or len(devices) != 2:
        raise ValueError(f"{output} did not run all rounds on two devices")
    observed_order = status["active_requests_order"]
    if len(observed_order) != total:
        raise ValueError(f"{output} has an incomplete active-request sequence")
    requests = request_order(observed_order[0])
    if observed_order != [requests[i % len(requests)] for i in range(total)]:
        raise ValueError(f"{output} did not restore the expected active-request sequence")
    kind = "graph_args" if meta["calendar_mode"] == "exact" else "graph_bucket"
    with (output / "benchmark.strace.log").open(encoding="utf-8") as stream:
        trace = read_arm_trace(stream, kind)
    ranks = set(trace.cache_events)
    expected_invocations = set(range(1, total + 1))
    if (
        set(trace.device_ids) != expected_invocations
        or any(row != trace.device_ids[1] for row in trace.device_ids.values())
        or set(trace.device_ids[1].values()) != devices
        or set(trace.device_ids[1]) != ranks
        or len(ranks) != 2
        or any(
            set(table) != expected_invocations or any(set(row) != ranks for row in table.values())
            for table in (trace.aicore, trace.launch_ts)
        )
    ):
        raise ValueError(f"{output} is missing timing, physical device or Graph cache evidence for either rank")
    if (require_start_gate or trace.gates) and (
        set(trace.gates) != expected_invocations
        or any(set(row) != devices or {event[0] for event in row.values()} != {0, 1} for row in trace.gates.values())
    ):
        raise ValueError(f"{output} is missing matching gate peer epochs")
    plans = len(requests) if kind == "graph_args" else 1
    expected = Counter(MISS=plans, STORE=plans, HIT=total - plans)
    for pid, events in trace.cache_events.items():
        keys = trace.lookup_keys[pid]
        if (
            events != expected
            or len(keys) != total
            or len(set(keys)) != plans
            or any(key != keys[index % plans] for index, key in enumerate(keys))
        ):
            raise ValueError(f"unexpected {kind} cache activity on pid {pid}: {events}, keys={keys}")
    samples = {i: max(event[0] for event in trace.aicore[i].values()) / 1000.0 for i in range(warmup + 1, total + 1)}
    by_variant = {count: [] for count in ACTIVE_REQUESTS}
    for invocation, wall in samples.items():
        by_variant[observed_order[invocation - 1]].append(wall)
    return {
        "devices": status["devices"],
        "sampling": meta["sampling"],
        "runtime_identity": status["runtime_identity"],
        "started_at_ns": status["started_at_ns"],
        "command": status["command"],
        "active_requests_order": observed_order,
        "first_active_requests": requests[0],
        "device_by_pid": trace.device_ids[1],
        **start_timing(trace, samples),
        "training_us_by_key_and_pid": trace.training_us,
        "training_us_by_pid": {pid: sum(costs.values()) for pid, costs in trace.training_us.items()},
        "graph_events_by_pid": trace.cache_events,
        "keys_by_pid": {pid: values[:plans] for pid, values in trace.lookup_keys.items()},
        **wall_summary(list(samples.values())),
        "active_request_variants": {
            count: {"normalized_rows": count * MARKOV_ROWS_PER_REQUEST, "rounds": len(values), **wall_summary(values)}
            for count, values in by_variant.items()
        },
        "slowest_rank_aicore_wall_us": list(samples.values()),
    }


def compare(args):
    import torch

    exact = read_json(args.exact / "prepared.json")
    bucket = read_json(args.bucket / "prepared.json")
    if (exact["calendar_mode"], bucket["calendar_mode"]) != ("exact", "bucket"):
        raise ValueError("select one exact arm and one explicit bucket arm")
    matched = (
        "case",
        "fixture",
        "sampling",
        "lib_commit",
        "fixture_source_sha256",
        "fixture_inputs_sha256",
        "business_kernel_sha256",
    )
    if any(exact[key] != bucket[key] for key in matched):
        raise ValueError("EP2 arms differ in input, sampling, workload source, or business kernels")
    records = {
        arm: parse_arm(path, require_start_gate=args.require_start_gate)
        for arm, path in (("exact", args.exact), ("bucket", args.bucket))
    }
    for key in ("gate_epochs_per_rank", "devices", "runtime_identity", "active_requests_order"):
        if records["exact"][key] != records["bucket"][key]:
            raise ValueError(f"EP2 arms differ in {key}")
    exact_outputs = torch.load(args.exact / "decode-outputs.pt", map_location="cpu", weights_only=True)
    bucket_outputs = torch.load(args.bucket / "decode-outputs.pt", map_location="cpu", weights_only=True)
    if set(exact_outputs) != set(ACTIVE_REQUESTS) or set(bucket_outputs) != set(ACTIVE_REQUESTS):
        raise ValueError("EP2 arms must report outputs for every active-request input")
    for count in ACTIVE_REQUESTS:
        for name in OUTPUTS:
            if not torch.equal(exact_outputs[count][name], bucket_outputs[count][name]):
                raise ValueError(f"DSpark output differs for {name} at {count} active requests")
    if args.reference_outputs is not None:
        reference = torch.load(args.reference_outputs, map_location="cpu", weights_only=True)
        for name in OUTPUTS:
            if not torch.equal(exact_outputs[dspark.BATCH][name], reference[name]):
                raise ValueError(f"Full-B12 DSpark output differs from the unwrapped reference for {name}")

    def deltas(exact_stats, bucket_stats):
        return {name: 100 * (bucket_stats[name] / exact_stats[name] - 1) for name in WALL_METRICS}

    variant_deltas = {
        count: deltas(
            records["exact"]["active_request_variants"][count], records["bucket"]["active_request_variants"][count]
        )
        for count in ACTIVE_REQUESTS
    }
    result = {
        **records,
        "run_order": sorted(records, key=lambda arm: records[arm]["started_at_ns"]),
        "kernel_count": len(exact["business_kernel_sha256"]),
        "output_elements_equal_by_active_requests": {
            count: sum(exact_outputs[count][name].numel() for name in OUTPUTS) for count in ACTIVE_REQUESTS
        },
        "unwrapped_reference_active_requests": dspark.BATCH if args.reference_outputs is not None else None,
        "unwrapped_reference_sha256": file_hash(args.reference_outputs) if args.reference_outputs else None,
        "percentage_delta": deltas(records["exact"], records["bucket"]),
        "active_request_variant_percentage_deltas": variant_deltas,
        "performance_limits_pct": REGRESSION_LIMITS,
        "within_performance_limits": all(
            delta <= REGRESSION_LIMITS[name] for variant in variant_deltas.values() for name, delta in variant.items()
        ),
    }
    write_json(args.output, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(required=True, dest="stage")
    for stage in ("prepare", "run"):
        command = stages.add_parser(stage)
        command.add_argument("--lib", required=True, type=Path)
        command.add_argument("--output", required=True, type=Path)
        if stage == "prepare":
            command.add_argument("--fixture", required=True, type=Path)
            command.add_argument("--scratch", required=True, type=Path)
            command.add_argument("--mode", required=True, choices=("exact", "bucket"))
            command.add_argument("--warmup", default=10, type=int)
            command.add_argument("--rounds", default=100, type=int)
        else:
            command.add_argument("--gate-runtime", type=Path, help="validated test wrapper runtime/a5/onboard")
            command.add_argument("--first-active-requests", type=int, choices=ACTIVE_REQUESTS, default=1)
    command = stages.add_parser("compare")
    for flag in ("exact", "bucket", "output"):
        command.add_argument(f"--{flag}", required=True, type=Path)
    command.add_argument("--reference-outputs", type=Path)
    command.add_argument("--require-start-gate", action="store_true")
    args = parser.parse_args()
    if args.stage == "compare":
        compare(args)
        return
    args.lib, args.output = args.lib.resolve(), args.output.resolve()
    args.lib_commit = subprocess.check_output(["git", "-C", str(args.lib), "rev-parse", "HEAD"], text=True).strip()
    sys.path[:0] = [str(args.lib), str((args.lib / dspark.ENTRY).parent)]
    daily = dspark.daily_runner(args.lib)
    if args.stage == "prepare":
        args.fixture, args.scratch = args.fixture.resolve(), args.scratch.resolve()
        if args.warmup < len(ACTIVE_REQUESTS) or args.rounds < 2 * len(ACTIVE_REQUESTS):
            parser.error("each variant needs at least one warmup and two measured rounds")
        prepare(args, daily)
    else:
        meta = read_json(args.output / "prepared.json")
        if meta["lib_commit"] != args.lib_commit:
            raise ValueError("use the Lib revision that produced the frozen fixture")
        run(args, daily, meta)


if __name__ == "__main__":
    main()
