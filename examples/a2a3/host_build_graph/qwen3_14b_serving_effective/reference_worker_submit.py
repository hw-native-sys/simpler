# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qualify real autoregressive Qwen decode through Worker.submit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import torch
from callable_bridge import inspect_artifact
from real_worker_submit import (
    BATCH,
    HEAD_DIM,
    HEADS,
    HIDDEN,
    LAYERS,
    PADDED_VOCAB,
    PAGE,
    _bind,
    _host_buffer,
    _view,
)
from reference_fixture import ReferenceFixture
from safetensors.torch import load_file, save_file
from simpler.task_interface import CallConfig, TensorArgType
from simpler.worker import Worker
from standalone_adapter import StandaloneDecodeAdapter
from weights import iter_kernel_weights, rope_tables

from simpler_setup.torch_interop import torch_dtype_to_datatype

LOGIT_RELATIVE_L2_MAX = 0.05
COSINE_MIN = 0.999

_LAST_OUTPUT: dict[str, Path] = {}


def metrics(actual, expected):
    actual, expected = actual.float(), expected.float()
    delta = actual - expected
    return {
        "finite": bool(torch.isfinite(actual).all()),
        "relative_l2": float(delta.norm() / expected.norm().clamp_min(1e-12)),
        "max_abs": float(delta.abs().max()),
        "cosine": float(torch.nn.functional.cosine_similarity(actual.reshape(1, -1), expected.reshape(1, -1))),
    }


def validate_kv(worker, devices, fixture, expected, steps, *, prefix_cache=None):
    shape = (fixture.num_pages, PAGE, HEADS, HEAD_DIM)
    nbytes = math.prod(shape) * 2
    staging = worker.create_buffer(nbytes)
    rows = []
    prefix_cache = {} if prefix_cache is None else prefix_cache
    try:
        for layer in range(LAYERS):
            prefix = prefix_cache.get(layer)
            if prefix is None:
                prefix = prefix_cache[layer] = load_file(str(fixture.root / f"layer_{layer:02d}.safetensors"))
            for name, kind in (("k_cache", "key"), ("v_cache", "value")):
                worker.copy_from(staging, devices[name], src_offset=layer * nbytes)
                physical = _view(staging, shape, torch.bfloat16)
                logical = physical.reshape(BATCH, fixture.pages_per_request * PAGE, HEADS, HEAD_DIM).permute(0, 2, 1, 3)
                prefix_equal = all(torch.equal(row[:, : fixture.length], prefix[kind][0]) for row in logical)
                tail_zero = bool((logical[:, :, fixture.length + steps :] == 0).all())
                added = logical[:, :, fixture.length : fixture.length + steps]
                reference = expected[f"{kind}_{layer:02d}"][:, :steps]
                checks = [metrics(row, reference) for row in added]
                rows.append(
                    {
                        "layer": layer,
                        "kind": kind,
                        "prefix_exact": prefix_equal,
                        "unused_tail_zero": tail_zero,
                        "rows": checks,
                        "passed": prefix_equal
                        and tail_zero
                        and all(
                            m["finite"] and m["relative_l2"] <= LOGIT_RELATIVE_L2_MAX and m["cosine"] >= COSINE_MIN
                            for m in checks
                        ),
                    }
                )
                del physical, logical, added
    finally:
        staging.close()
    return rows


def load_kv_reference(root, reference_digest):
    manifest = json.loads((root / "manifest.json").read_text())
    path = root / "appended.safetensors"
    if manifest["reference_sums_sha256"] != reference_digest:
        raise ValueError("Appended KV reference identity mismatch")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != manifest["appended_sha256"]:
        raise ValueError("Appended KV reference checksum mismatch")
    return load_file(str(path))


def allocate_resident(worker, devices, hosts, fixture, model, *, shared_devices=None, layer_cache=None, host_staged=()):
    def upload(name, tensor):
        device = worker.alloc_child_tensor(0, tuple(tensor.shape), torch_dtype_to_datatype(tensor.dtype).value)
        devices[name] = device
        staging = _host_buffer(worker, tensor)
        try:
            worker.copy_to(device, staging)
        finally:
            staging.close()

    if shared_devices is None:
        for name, value in iter_kernel_weights(model):
            upload(name, value)
            print("uploaded " + name, flush=True)
        del value
        for name, value in zip(("rope_cos", "rope_sin"), rope_tables(model)):
            upload(name, value)
    else:
        devices.update(shared_devices)
    kv_shape = (LAYERS * fixture.num_pages * HEADS * PAGE, HEAD_DIM)
    for name in ("k_cache", "v_cache"):
        devices[name] = worker.alloc_child_tensor(0, kv_shape, torch_dtype_to_datatype(torch.bfloat16).value)
    for layer in range(LAYERS):
        if layer_cache is None:
            values = fixture.layer(layer)
        else:
            if layer not in layer_cache:
                layer_cache[layer] = tuple(fixture.layer(layer))
            values = layer_cache[layer]
        for kind, physical in values:
            staging = _host_buffer(worker, physical)
            try:
                worker.copy_to(
                    devices["k_cache" if kind == "key" else "v_cache"],
                    staging,
                    dst_offset=layer * physical.numel() * physical.element_size(),
                )
            finally:
                staging.close()
    print("uploaded all 40 KV layers", flush=True)
    for name, shape, dtype in (
        ("seq_lens", (BATCH,), torch.int32),
        ("slot_mapping", (BATCH,), torch.int32),
        ("block_table", (BATCH * 32,), torch.int32),
        ("out", (BATCH, PADDED_VOCAB), torch.float32),
        ("sampled_ids_in", (BATCH, 8), torch.int32),
        ("sampled_ids", (BATCH, 8), torch.int32),
        ("next_hidden", (BATCH, HIDDEN), torch.bfloat16),
    ):
        hosts[name] = _host_buffer(worker, torch.zeros(shape, dtype=dtype))
        if name not in host_staged:
            devices[name] = worker.alloc_child_tensor(0, shape, torch_dtype_to_datatype(dtype).value)
            worker.copy_to(devices[name], hosts[name])


def main():  # noqa: PLR0912, PLR0915 -- execution and finalization share one report
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--kv-reference", type=Path)
    parser.add_argument(
        "--depth2-probe",
        action="store_true",
        help="record a requested depth=2 probe and run the safe serial depth=1 fallback",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    _LAST_OUTPUT["path"] = args.output
    repo_root = Path(__file__).resolve().parents[4]
    depth2_fallback = args.depth2_probe
    torch.set_num_threads(8)
    fixture = ReferenceFixture(args.fixture)
    if not 1 <= args.steps <= fixture.steps:
        raise ValueError("steps must be within the frozen reference")
    fixture.verify_model(args.model)
    adapter = StandaloneDecodeAdapter(fixture.adapter_fixture())
    artifact = inspect_artifact(args.artifact)
    chip = artifact.compile()
    specs = json.loads((args.artifact / "distributed_meta.json").read_text())["params"]
    directions = {
        p["name"].split("__ssa_", 1)[0]: TensorArgType.INPUT
        if p["direction"] == "In"
        else TensorArgType.OUTPUT_EXISTING
        for p in specs
    }
    report = {
        "status": "running",
        "steps_requested": args.steps,
        "launch_depth": 1,
        "launch_depth_requested": 2 if args.depth2_probe else 1,
        "depth2_policy": "safe_serial_fallback" if args.depth2_probe else "serial_default",
        "depth2_conclusion": (
            "safe_serial_fallback_host_sampled_token_feedback" if depth2_fallback else "not_requested"
        ),
        "eos_policy": "fixed dispatch count; compare all frozen tokens",
        "logit_gate": {
            "relative_l2_max": LOGIT_RELATIVE_L2_MAX,
            "cosine_min": COSINE_MIN,
            "tokens": "exact",
        },
        "reference_sums_sha256": hashlib.sha256((args.fixture / "SHA256SUMS").read_bytes()).hexdigest(),
        "code_head": subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True).strip(),
        "source_hashes": artifact.source_hashes,
        "steps": [],
    }
    report["consumer_source_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(Path(__file__).parent.glob("*.py"))
    }
    report["runtime_binaries_sha256"] = {
        str(path.relative_to(repo_root)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((repo_root / "build/lib/a2a3/onboard/host_build_graph").glob("*"))
        if path.is_file()
    }
    if not report["runtime_binaries_sha256"]:
        raise FileNotFoundError("no host_build_graph runtime binaries found")
    report["torch_version"] = torch.__version__
    report["kv_numerical_gate"] = {
        "relative_l2_max": LOGIT_RELATIVE_L2_MAX,
        "cosine_min": COSINE_MIN,
        "prefix": "bitwise unchanged",
        "unused_tail": "zero",
    }
    kv_reference = None
    if args.kv_reference is not None:
        kv_reference = load_kv_reference(args.kv_reference, report["reference_sums_sha256"])
    worker = Worker(
        level=3,
        platform="a2a3",
        runtime="host_build_graph",
        device_ids=[args.device],
        num_sub_workers=0,
        launch_depth=1,
    )
    devices, hosts = {}, {}
    failure = None
    try:
        chip_handle = worker.register(chip)
        worker.init()
        allocate_resident(worker, devices, hosts, fixture, args.model)
        config = CallConfig()
        config.enable_dep_gen = False
        config.enable_chip_swimlane = 0
        for index in range(args.steps):
            step = adapter.next_step()
            values = {
                "seq_lens": step.seq_lens,
                "slot_mapping": step.slot_mapping,
                "block_table": step.block_table.reshape(-1),
                "sampled_ids_in": torch.zeros((BATCH, 8), dtype=torch.int32),
                "sampled_ids": torch.full((BATCH, 8), -1, dtype=torch.int32),
            }
            values["sampled_ids_in"][:, 0] = step.input_token_ids
            for name, value in values.items():
                _view(hosts[name], tuple(value.shape), value.dtype).copy_(value)
                worker.copy_to(devices[name], hosts[name])
            task_args = _bind(specs, devices, directions=directions, cache_pages=fixture.num_pages)

            def submit_next_level(orch, _args, cfg):
                orch.submit_next_level(chip_handle, task_args, cfg, worker=0)

            handle = worker.submit(submit_next_level, args=None, config=config)
            run_id = handle._run_id
            handle.result(timeout=120)
            for name in ("sampled_ids", "out"):
                worker.copy_from(hosts[name], devices[name])
            sampled = _view(hosts["sampled_ids"], (BATCH, 8), torch.int32).clone()
            logits = _view(hosts["out"], (BATCH, PADDED_VOCAB), torch.float32).clone()
            expected_logits = fixture.golden["logits"][index]
            row_metrics = [
                metrics(
                    row[: expected_logits.shape[-1]],
                    expected_logits[row_index] if expected_logits.ndim == 2 else expected_logits,
                )
                for row_index, row in enumerate(logits)
            ]
            entry = {
                "index": index,
                "run_id": run_id,
                "input": step.input_token_ids.tolist(),
                "sampled": sampled[:, 0].tolist(),
                "expected": step.expected_output_token_ids.tolist(),
                "argmax": logits.argmax(dim=1).tolist(),
                "logits": row_metrics,
                "seq_lens": step.seq_lens.tolist(),
                "slot_mapping": step.slot_mapping.tolist(),
            }
            report["steps"].append(entry)
            print(json.dumps(entry), flush=True)
            passed = all(
                m["finite"] and m["relative_l2"] <= LOGIT_RELATIVE_L2_MAX and m["cosine"] >= COSINE_MIN
                for m in row_metrics
            )
            if index == 0 or not passed or not torch.equal(sampled[:, 0], step.expected_output_token_ids):
                save_file({"logits": logits, "sampled": sampled}, str(args.output / f"step-{index:03d}.safetensors"))
            if kv_reference is not None and (index in (0, 117, 118, args.steps - 1) or not passed):
                kv_checks = validate_kv(worker, devices, fixture, kv_reference, index + 1)
                entry["kv"] = kv_checks
                passed = passed and all(check["passed"] for check in kv_checks)
            adapter.complete_step(sampled)
            if not passed:
                raise AssertionError(f"Numerical gate failed at step {index}")
        report["status"] = "passed"
        report["completed_steps"] = adapter.completed_steps
    except BaseException as error:
        failure = error
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        try:
            worker.close()
        except BaseException as error:
            report["cleanup_error"] = f"{type(error).__name__}: {error}"
            if failure is None:
                failure = error
                report["status"] = "failed"
                report["error"] = report["cleanup_error"]
            else:
                error.__context__ = failure.__context__
                failure.__context__ = error
        try:
            (args.output / "result.json").write_text(json.dumps(report, indent=2))
        except BaseException as error:
            if failure is None:
                failure = error
            else:
                error.__context__ = failure.__context__
                failure.__context__ = error
    if failure is not None:
        raise failure
    return 0


def cli():
    try:
        return main()
    except BaseException as error:
        if isinstance(error, SystemExit) and error.code in (None, 0):
            raise
        output_path = _LAST_OUTPUT.get("path")
        if output_path is not None:
            result_path = output_path / "result.json"
            try:
                if not result_path.exists():
                    result_path.write_text(
                        json.dumps({"status": "failed", "error": f"{type(error).__name__}: {error}"}, indent=2)
                    )
            except BaseException as report_error:
                report_error.__context__ = error.__context__
                error.__context__ = report_error
        raise


if __name__ == "__main__":
    raise SystemExit(cli())
