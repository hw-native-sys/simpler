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
# Resolve Lib before importing its harness. Keep report-only commands independent of Torch.
"""Benchmark the A5 DSpark 8K Lib case and export its Percounter swimlanes."""

import argparse
import ast
import copy
import importlib.util
import inspect
import json
import math
import os
import runpy
import statistics
import subprocess
import sys
import tempfile
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ENTRY = Path("models/deepseek_v4_1_flash/decode_fwd_dspark.py")
BATCH = 12
START_POSITION = 8192
TARGET_ROWS = 72
ARTIFACT_RUNTIME = {"host_build_graph": "percounter", "tensormap_and_ringbuffer": "tensormap_and_ringbuffer"}


def daily_runner(lib):
    spec = importlib.util.spec_from_file_location("lib_ci_perf", lib / ".github/scripts/ci_perf.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load the Lib performance runner from {lib}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_case(ep):
    return {
        "case_id": f"dsv4.1-flash-decode-fwd-dspark-ep{ep}-tp{ep}-b12-8k",
        "entrypoint": str(ENTRY),
        "platform": "a5",
        "device_count": ep,
        "arguments": [
            "--ep",
            str(ep),
            "--dp",
            str(ep),
            "--lm-head-tp",
            str(ep),
            "--experts-per-rank",
            "12",
            "--draft-experts-per-rank",
            "4",
            "--balanced-routing",
        ],
    }


def apply_fixture(fn, specs, ep):
    import torch
    from golden import TensorSpec

    entry = fn._func
    m = entry.__globals__
    if (m["DECODE_BATCH"], m["EP_SIZE"], m["TARGET_ROWS"]) != (BATCH, ep, TARGET_ROWS):
        raise ValueError("DSpark entry does not match the requested B12/72-row topology")
    target = m["target_fwd"]
    state = m["state_ops"]
    limit = START_POSITION + m["ROWS"] + 8
    pages = (limit + m["BLOCK_SIZE"] - 1) // m["BLOCK_SIZE"]
    extents = {
        m["TARGET_FWD_FWD_ORI_BLOCKS_DYN"].name: target.N_LAYERS * BATCH * 2,
        m["TARGET_FWD_FWD_CMP_BLOCKS_DYN"].name: target.COMPRESSED_SOURCE_UNITS * BATCH * pages,
        m["TARGET_FWD_FWD_INDEX_BLOCKS_DYN"].name: target.COMPRESSED_SOURCE_UNITS * BATCH * pages,
        m["FWD_C2A_TABLE_DYN"].name: pages,
        m["FWD_C1A_TABLE_DYN"].name: pages,
        m["WINDOW_TABLE_DYN"].name: 2,
        state.TABLE_DYN.name: 2,
        m["DRAFT_ORI_BLOCKS_DYN"].name: BATCH * 2,
        m["TARGET_FWD_ROPE_ROWS_DYN"].name: limit,
        m["STATE_OPS_ROPE_ROWS_DYN"].name: limit,
    }

    def page_table(width):
        return torch.arange(BATCH * width, dtype=torch.int32).reshape(1, BATCH, width).repeat(ep, 1, 1)

    windows = page_table(2)
    compressed = page_table(pages)
    cos, sin = target.prepare_forward_rope_tables(limit)
    replacements = {
        "target_window_table": windows,
        "draft_table": windows[:, None].repeat(1, m["DSPARK_DRAFT_LAYERS"], 1, 1),
        "target_c2a_table": compressed,
        "target_c1a_table": compressed,
        "c2a_index_block_table": compressed,
        "c1a_index_block_table": compressed,
        "freqs_cos": cos.unsqueeze(0).repeat(ep, 1, 1, 1),
        "freqs_sin": sin.unsqueeze(0).repeat(ep, 1, 1, 1),
        "attention_num_tokens": TARGET_ROWS,
        "moe_num_tokens": TARGET_ROWS,
    }
    params = inspect.signature(entry).parameters
    named = {spec.name: spec for spec in specs if isinstance(spec, TensorSpec)}
    if not replacements.keys() <= named.keys():
        raise ValueError("DSpark fixture arguments changed")
    for name, spec in named.items():
        dims = params[name].annotation.shape
        spec.shape = [extents.get(getattr(dim, "name", ""), old) for dim, old in zip(dims, spec.shape, strict=True)]
        if name in replacements:
            spec.init_value = replacements[name]
        if isinstance(spec.init_value, torch.Tensor) and tuple(spec.init_value.shape) != tuple(spec.shape):
            if name != "draft_window_cache":
                raise ValueError(f"{name} still has its short-context initializer")
            spec.init_value = torch.zeros(spec.shape, dtype=torch.uint8)
            _, factors = m["unpack_quantized_cache"](spec.init_value, False)
            factors.fill_(1)

    named["slot_ids"].init_value[:] = torch.arange(BATCH, dtype=torch.int32)
    named["generations"].init_value.fill_(1)
    meta = named["state_meta"].init_value
    for index, value in (
        (state.STATE_VALID, 1),
        (state.STATE_GENERATION, 1),
        (state.STATE_ANCHOR, START_POSITION),
        (state.STATE_COMMITTED, START_POSITION),
        (state.STATE_DRAFT_COUNT, m["ROWS"] - 1),
        (state.STATE_LIMIT, limit),
    ):
        meta[..., index] = value
    named["state_tokens"].init_value.fill_(1)
    for name, spec in named.items():
        if isinstance(spec.init_value, torch.Tensor) and tuple(spec.init_value.shape) != tuple(spec.shape):
            raise ValueError(f"{name} has a mismatched fixture shape")
    return {
        "batch_per_card": BATCH,
        "start_position": START_POSITION,
        "target_rows_per_card": TARGET_ROWS,
        "window_pages": 2,
        "compressed_pages": pages,
    }


def run_entry(lib, daily, case, devices, name):
    sys.argv = daily.model_argv({"platform": "a5"}, case, devices)
    return runpy.run_path(str(lib / ENTRY), run_name=name)


def compile_cached(build, runner):
    from pypto.runtime.device_runner import _compile_and_assemble

    children = sorted((build / "next_levels").glob("*/kernel_config.py"))
    if not children:
        raise ValueError("no compiled DSpark chip callables")
    runner._setup_runtime_dir(str(build), compile_label="DSpark benchmark", platform="a5")
    for child in children:
        _compile_and_assemble(child.parent, "a5")
    return {child.parent.name: runner._artifact_runtime(child) for child in children}


def prime(args, daily, meta):
    """Populate and verify CPU-side caches for an existing prepared fixture."""
    from golden import runner

    os.chdir(args.output)
    build = Path(meta["runtime_dir"])
    expected = meta["artifact_runtimes"]
    if compile_cached(build, runner) != expected:
        raise ValueError("prepared chip callable runtime changed")
    with daily.cached_native_only():
        if compile_cached(build, runner) != expected:
            raise ValueError("prepared chip callable runtime changed")
    print("CPU-only replay cache check passed.", flush=True)


def prepare(args, daily, case):
    import cloudpickle
    import golden
    import torch
    from golden import ScalarSpec, TensorSpec, runner

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if (output / "prepared.json").exists():
        raise ValueError("use a new output directory")
    original = golden.run
    manifest = daily.load_manifest("a5")
    sampling = manifest["sampling"]
    os.environ["PYPTO_BENCH"] = "0"
    os.environ["PYPTO_LIB_A5_RUNTIME"] = args.runtime
    prepared = None

    def capture(**kwargs):
        nonlocal prepared
        if prepared is not None:
            raise ValueError("DSpark must call golden.run once")
        fixture = apply_fixture(kwargs["fn"], kwargs["specs"], args.ep)
        prepared = {
            "case": case,
            "fixture": fixture,
            "sampling": sampling,
            "lib_commit": args.lib_commit,
            "runtime": args.runtime,
        }
        print(json.dumps(prepared), flush=True)
        if args.stage == "inspect":
            return SimpleNamespace(passed=True)
        result = original(**{**kwargs, "compile_only": True})
        if not result.passed:
            raise RuntimeError(result.error or "DSpark compilation failed")
        build = Path(result.work_dir).resolve()
        runtimes = compile_cached(build, runner)
        if set(runtimes.values()) != {ARTIFACT_RUNTIME[args.runtime]}:
            raise ValueError(f"expected {ARTIFACT_RUNTIME[args.runtime]} callables, got {runtimes}")
        specs = kwargs["specs"]
        tensors = [spec for spec in specs if isinstance(spec, TensorSpec)]
        scalars = [spec for spec in specs if isinstance(spec, ScalarSpec)]
        torch.manual_seed(0)
        runner._prepare_inputs(specs, tensors, scalars, None, output / "frozen", True)
        cleaned = [copy.copy(spec) for spec in specs]
        for spec in cleaned:
            if isinstance(spec, TensorSpec):
                spec.init_value = None
        with (output / "inputs.pkl").open("wb") as f:
            cloudpickle.dump((cleaned, kwargs["config"]), f)
        prepared.update(runtime_dir=str(build), artifact_runtimes=runtimes)
        return result

    os.chdir(output)
    golden.run = capture
    try:
        run_entry(args.lib, daily, case, list(range(args.ep)), "__main__")
    finally:
        golden.run = original
    if prepared is None:
        raise ValueError("entry did not provide a DSpark fixture")
    if args.stage == "prepare":
        daily.write_json(output / "prepared.json", prepared)


def load_frozen_inouts(entry, specs, data):
    import torch

    module = ast.parse(entry.read_bytes(), filename=str(entry))
    fn = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "l3_decode_fwd_dspark")
    names = {
        arg.arg
        for arg in fn.args.args
        if isinstance(arg.annotation, ast.Subscript)
        and isinstance(arg.annotation.value, ast.Attribute)
        and arg.annotation.value.attr == "InOut"
    }
    if not {"state_meta", "state_tokens"} <= names <= {spec.name for spec in specs}:
        raise ValueError("DSpark InOut signature changed")
    return {
        name: torch.load(data / "in" / f"{name}.pt", weights_only=True, mmap=True).view(torch.uint8) for name in names
    }


@contextmanager
def restore_inouts_between_dispatches(entry, specs, data):
    """Replay the frozen in/out arguments before each warmup and timed dispatch."""
    import torch

    inouts = load_frozen_inouts(entry, specs, data)
    frozen = {index: inouts[spec.name] for index, spec in enumerate(specs) if spec.name in inouts}
    bench = importlib.import_module("pypto.runtime.bench")
    original = bench._dispatch_loop
    resets = SimpleNamespace(count=0)

    def dispatch_loop(handle, args, *, rounds, warmup, dispatch_config):
        for _ in range(rounds + warmup):
            for index, reference in frozen.items():
                args[index].view(torch.uint8).copy_(reference)
            resets.count += 1
            handle(*args, config=dispatch_config)

    bench._dispatch_loop = dispatch_loop
    try:
        yield resets
    finally:
        bench._dispatch_loop = original


@contextmanager
def restore_inouts_between_trace_passes(entry, specs, data):
    """Restore the same host InOut tensors before PyPTO's graph and timing dispatches."""
    import torch
    from golden.runner import _strip_ssa_suffix
    from pypto.runtime import distributed_runner as runtime

    frozen = load_frozen_inouts(entry, specs, data)
    resets = SimpleNamespace(count=0)

    def with_restore(dispatch):
        # Both one-shot and prepared dispatch receive the live host tensors by name.
        def run(owner, scope, tensors, *args, **kwargs):
            inouts = {}
            for param, tensor in tensors.items():
                name = _strip_ssa_suffix(param)
                if name in frozen:
                    if name in inouts:
                        raise ValueError(f"Ambiguous host InOut tensor {name} for DSpark trace replay")
                    inouts[name] = tensor
            for name, reference in frozen.items():
                tensor = inouts.get(name)
                if tensor is None or not isinstance(tensor, torch.Tensor):
                    raise TypeError(f"Expected host InOut tensor {name} for DSpark trace replay")
                tensor.view(torch.uint8).copy_(reference)
            resets.count += 1
            return dispatch(owner, scope, tensors, *args, **kwargs)

        return run

    prepared = runtime.DistributedWorker
    with (
        patch.object(runtime, "_dispatch", with_restore(runtime._dispatch)),
        patch.object(prepared, "_dispatch_prepared", with_restore(prepared._dispatch_prepared)),
    ):
        yield resets


def replay(args, daily, meta):
    import cloudpickle
    import golden

    case = meta["case"]
    selected = [int(value) for value in os.environ["TASK_DEVICE"].split(",")]
    if len(selected) != case["device_count"] or len(set(selected)) != len(selected):
        raise ValueError("TASK_DEVICE does not match the prepared topology")
    sampling = meta["sampling"]
    trace_level = args.trace_level if args.stage == "trace" else 0
    os.environ.update(
        PYPTO_BENCH="0" if trace_level else "1",
        PYPTO_BENCH_WARMUP=str(sampling["warmup"]),
        PYPTO_BENCH_ROUNDS=str(sampling["rounds"]),
        SIMPLER_DEVICE_STRACE_ENABLE="1",
    )
    os.chdir(args.output)
    entry = run_entry(args.lib, daily, case, selected, "dspark_bench_entry")
    with (args.output / "inputs.pkl").open("rb") as f:
        specs, cfg = cloudpickle.load(f)
    from pypto.ir import DistributedConfig

    cfg = dict(
        cfg,
        distributed_config=DistributedConfig(device_ids=selected, num_sub_workers=0),
        enable_chip_swimlane=trace_level,
    )
    data = args.output / "frozen/data"
    restore = restore_inouts_between_trace_passes if trace_level else restore_inouts_between_dispatches
    resets_scope = restore(args.lib / ENTRY, specs, data)
    with daily.cached_native_only(), resets_scope as resets:
        result = golden.run(
            fn=entry["l3_decode_fwd_dspark"],
            specs=specs,
            config=cfg,
            golden_fn=None,
            golden_data=str(data),
            runtime_dir=meta["runtime_dir"],
        )
    if trace_level:
        if not result.passed:
            raise RuntimeError(result.error or "DSpark trace replay failed")
        if resets.count != 2:
            raise ValueError(f"Expected frozen InOut resets for both trace passes, got {resets.count}")
        return
    record = {"passed": result.passed, "error": result.error, "devices": selected}
    if result.passed:
        if resets.count != sampling["rounds"] + sampling["warmup"]:
            raise ValueError(f"expected a reset per dispatch, got {resets.count}")
        record["inout_resets"] = resets.count
        record["benchmark"] = daily.benchmark_payload(result.bench)
        record["device_wall_samples"] = [
            {str(pid): [item.device_wall_us for item in dispatches] for pid, dispatches in row.items()}
            for row in result.bench.rounds_dispatches
        ]
    daily.write_json(args.output / "raw-result.json", record)
    if not result.passed:
        raise RuntimeError(result.error or "DSpark replay failed")


def capture_trace(args, daily, meta):
    import fcntl

    from pypto.runtime import runner

    if set(meta["artifact_runtimes"].values()) != {"percounter"}:
        raise ValueError("DSpark trace replay requires a prepared A5 Percounter artifact")
    build = Path(meta["runtime_dir"])
    raw = build / "dfx_outputs"
    captures = args.output / "traces"
    captures.mkdir(exist_ok=True)
    with (build / ".dspark-trace.lock").open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if raw.is_dir() and any(raw.iterdir()):
            raise FileExistsError(f"Archive the existing capture before replay: {raw}")
        # Keep PyPTO's two-pass pairing and name maps. Render only after release.
        with patch.object(runner, "_generate_swimlane", return_value=None):
            replay(args, daily, meta)
        records = list(raw.glob("rank*/d[0-9]*/chip_swimlane_records.json"))
        ranks = {record.parent.parent.name for record in records}
        if len(ranks) != meta["case"]["device_count"]:
            raise ValueError(f"Missing rank captures under {raw}: {sorted(ranks)}")
        capture = Path(tempfile.mkdtemp(prefix=f"l{args.trace_level}-", dir=captures))
        raw.rename(capture / "dfx_outputs")
    print(capture)


def worker_span_us(records, deps):
    blocks = records["aicore_tasks"]
    clock = records["metadata"]["clock_freq_hz"]
    if not blocks or clock <= 0 or any(not 0 < block[3] <= block[4] for block in blocks):
        raise ValueError("Missing or invalid AICore block timing")
    if records.get("percounter_task_events", {}).get("dropped", 0):
        raise ValueError("Percounter scheduler records were dropped")
    expected = Counter()
    for task in deps["tasks"]:
        slots = sum(kernel >= 0 for kernel in task["kernel_ids"])
        if slots:
            task_id = int(task["task_id"])
            expected.update({(task_id, block): slots for block in range(task["block_num"])})
    actual = Counter((block[1], block[7]) for block in blocks)
    if actual != expected:
        missing = sum((expected - actual).values())
        extra = sum((actual - expected).values())
        raise ValueError(f"AICore block coverage differs from dep-gen ({missing} missing, {extra} extra)")
    return (max(block[4] for block in blocks) - min(block[3] for block in blocks)) * 1e6 / clock


def convert_trace(raw, perfetto, level, span):
    with tempfile.TemporaryDirectory(dir=perfetto.parent) as directory:
        converted = Path(directory) / perfetto.name
        subprocess.run(
            [sys.executable, "-m", "simpler_setup.tools.swimlane_converter", str(raw), "-o", str(converted)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        events = json.loads(converted.read_text(encoding="utf-8"))["traceEvents"]
        tracks = {event["args"]["name"]: event["pid"] for event in events if event.get("name") == "process_name"}
        if level == 2 and not {"Scheduler View", "AICPU Lifecycle"} <= tracks.keys():
            raise ValueError(f"Missing A5 calendar scheduler tracks in the export of {raw}")
        worker_pid = tracks.get("Worker View")
        blocks = [event for event in events if event.get("pid") == worker_pid and event.get("ph") == "X"]
        if worker_pid is None or not blocks:
            raise ValueError(f"No Worker View intervals in the export of {raw}")
        exported_span = max(block["ts"] + block["dur"] for block in blocks) - min(block["ts"] for block in blocks)
        if not math.isclose(span, exported_span, abs_tol=0.01):
            raise ValueError(f"Worker View span mismatch: raw {span:.3f} us, Perfetto {exported_span:.3f} us")
        converted.replace(perfetto)


def export_trace(args):
    source = args.trace_root or args.output / "traces"
    level = args.trace_level
    candidates = []
    skipped = []
    capture_errors = (OSError, ValueError, KeyError, IndexError, TypeError)
    for raw in sorted(source.rglob("rank*/d[0-9]*/chip_swimlane_records.json")):
        try:
            records = json.loads(raw.read_text(encoding="utf-8"))
            if records["chip_swimlane_level"] != level:
                continue
            deps = json.loads(raw.with_name("deps.json").read_text(encoding="utf-8"))
            candidates.append((worker_span_us(records, deps), raw))
        except capture_errors as error:
            skipped.append({"raw": str(raw), "reason": str(error)})
    out = args.output / "swimlanes"
    out.mkdir(parents=True, exist_ok=True)
    perfetto = out / f"fastest-l{level}.perfetto.json"
    for span, raw in sorted(candidates):
        try:
            convert_trace(raw, perfetto, level, span)
        except subprocess.CalledProcessError as error:
            detail = error.stderr.strip() if error.stderr else str(error)
            skipped.append({"raw": str(raw), "reason": detail.splitlines()[-1] if detail else str(error)})
        except capture_errors as error:
            skipped.append({"raw": str(raw), "reason": str(error)})
        else:
            break
    else:
        if skipped:
            print(json.dumps({"skipped": skipped}, indent=2), file=sys.stderr)
        raise ValueError(f"No complete, convertible L{level} worker captures under {source}")
    report = {
        "level": level,
        "rank": raw.parent.parent.name,
        "dispatch": raw.parent.name,
        "worker_span_us": span,
        "candidate_count": len(candidates),
        "raw": str(raw),
        "perfetto": str(perfetto),
        "skipped": skipped,
    }
    (out / f"fastest-l{level}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def rank_medians(rows):
    if not rows or not all(
        dispatches and all(math.isfinite(value) and value > 0 for value in dispatches)
        for row in rows
        for dispatches in row.values()
    ):
        return None
    return {rank: statistics.median(sum(row[rank]) for row in rows) for rank in rows[0]}


def summarize(args, daily, meta):
    payload = daily.read_json(args.output / "raw-result.json")
    case, sampling = meta["case"], meta["sampling"]
    if payload.get("inout_resets") != sampling["rounds"] + sampling["warmup"]:
        raise ValueError("no verified reset of frozen InOut state for each dispatch")
    effective = daily.effective_us(payload, case, sampling)
    bench = payload["benchmark"]
    effective_ranks = rank_medians(bench["samples"])
    wall = payload.get("device_wall_samples", [])
    wall_ranks = rank_medians(wall)
    wall_fastest = wall_critical = None
    if wall_ranks is not None:
        wall_payload = {**payload, "benchmark": {**bench, "samples": wall}}
        wall_fastest = daily.effective_us(wall_payload, case, sampling)
        wall_critical = statistics.median(max(map(sum, row.values())) for row in wall)
    if effective_ranks is None and wall_ranks is None:
        raise ValueError("both effective and device-wall timing are unavailable")

    report = {
        **meta,
        "devices": payload["devices"],
        "inout_resets": payload["inout_resets"],
        "effective_median_us": effective if effective_ranks is not None else None,
        "effective_rank_medians_us": effective_ranks,
        "device_wall_fastest_rank_median_us": wall_fastest,
        "device_wall_slowest_rank_per_round_median_us": wall_critical,
        "device_wall_rank_medians_us": wall_ranks,
        "verification": "synthetic fixture execution",
    }
    daily.write_json(args.output / "summary.json", report)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("inspect", "prepare", "prime", "run", "summarize", "trace", "export-trace"))
    parser.add_argument("--lib", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ep", type=int, choices=(2, 8))
    parser.add_argument("--runtime", choices=tuple(ARTIFACT_RUNTIME), default="host_build_graph")
    parser.add_argument("--trace-level", type=int, choices=(1, 2, 3, 4), default=2)
    parser.add_argument("--trace-root", type=Path, help="Existing capture tree to scan with export-trace")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.stage == "export-trace":
        if args.trace_root is not None:
            args.trace_root = args.trace_root.resolve()
        export_trace(args)
        return
    if args.lib is None:
        parser.error("--lib is required for this stage")
    args.lib = args.lib.resolve()
    args.lib_commit = subprocess.check_output(["git", "-C", str(args.lib), "rev-parse", "HEAD"], text=True).strip()
    sys.path[:0] = [str(args.lib), str((args.lib / ENTRY).parent)]
    daily = daily_runner(args.lib)
    if args.stage in ("prepare", "inspect"):
        if args.ep is None:
            parser.error("--ep is required for preparation")
        prepare(args, daily, make_case(args.ep))
        return
    meta = daily.read_json(args.output / "prepared.json")
    if args.lib_commit != meta["lib_commit"]:
        raise ValueError("use the same Lib revision for preparation and replay")
    if args.stage == "prime":
        prime(args, daily, meta)
    elif args.stage == "run":
        replay(args, daily, meta)
    elif args.stage == "trace":
        capture_trace(args, daily, meta)
    else:
        summarize(args, daily, meta)


if __name__ == "__main__":
    main()
