#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run full Qwen requests through the existing PyPTO Serving Program engine."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import importlib
import importlib.metadata
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_sequences(fixture, golden, prompt_ids, *, requests, max_tokens):
    """Join the prefill token to its complete, checked decode feedback chain."""
    first = fixture.metadata["first_generated_token_ids"].tolist()
    prompts = fixture.metadata["prompt_token_ids"].tolist()
    inputs = golden["decode_input_token_ids"].tolist()
    outputs = golden["decode_output_token_ids"].tolist()
    if len(prompts) != 16 or any(row != prompt_ids for row in prompts):
        raise ValueError("serving tokenizer input differs from the frozen prompt IDs")
    if len(first) != 16 or len(inputs) != 127 or len(outputs) != 127:
        raise ValueError("reference must contain 16 requests and 127 decode steps")
    if any(len(row) != 16 for row in inputs + outputs):
        raise ValueError("reference decode rows must contain 16 tokens")
    if inputs != [first, *outputs[:-1]]:
        raise ValueError("reference decode inputs break the autoregressive token chain")
    if requests not in (1, 16) or not 2 <= max_tokens <= 128:
        raise ValueError("require 1 or 16 requests and 2..128 generated tokens")
    return [[first[row], *(step[row] for step in outputs[: max_tokens - 1])] for row in range(requests)]


async def collect_request(engine, prompt, config, expected, prompt_tokens, row):
    started = time.perf_counter()
    first_at = last_at = None
    terminal = False
    outputs = engine.add_request(row["request_id"], prompt, config)
    async with contextlib.aclosing(outputs):
        async for output in outputs:
            now = time.perf_counter()
            if terminal:
                raise RuntimeError("serving emitted output after the terminal response")
            if output.token_id is not None:
                row["tokens"].append(output.token_id)
                first_at = now if first_at is None else first_at
                last_at = now
                if row["tokens"] != expected[: len(row["tokens"])]:
                    raise ValueError(f"token mismatch for {row['request_id']} at index {len(row['tokens']) - 1}")
            if output.prompt_tokens != prompt_tokens or output.completion_tokens != len(row["tokens"]):
                raise ValueError("serving token counts differ from the observed request")
            if output.finished:
                terminal = True
                row["finish_reason"] = output.finish_reason
                row["terminal_tokens"] = list(output.token_ids)
                if output.finish_reason != "FINISHED_LENGTH":
                    raise RuntimeError(f"unexpected serving finish reason: {output.finish_reason}")
                if list(output.token_ids) != expected or row["tokens"] != expected:
                    raise ValueError("terminal token history differs from the complete reference")
    if not terminal or first_at is None or last_at is None:
        raise RuntimeError("serving stream ended without a terminal response")
    row.update(
        token_gate="passed",
        ttft_seconds=first_at - started,
        tpot_seconds=(last_at - first_at) / (len(expected) - 1),
        end_to_end_seconds=time.perf_counter() - started,
    )


async def run_requests(engine, prompt, config, expected, *, prompt_tokens, rounds, report):
    """Drain or cancel every request before checked engine shutdown."""
    failure = None
    report["closed"] = False
    try:
        started = time.perf_counter()
        report["engine_start_attempted"] = True
        await engine.start()
        report["engine_started"] = True
        report["startup_seconds"] = time.perf_counter() - started
        for round_index in range(rounds):
            tasks = []
            for index, tokens in enumerate(expected):
                row = {
                    "request_id": f"qwen-{round_index}-{index}",
                    "round": round_index,
                    "tokens": [],
                    "token_gate": "not_passed",
                }
                report["requests"].append(row)
                tasks.append(asyncio.create_task(collect_request(engine, prompt, config, tokens, prompt_tokens, row)))
            try:
                await asyncio.gather(*tasks)
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                for task, row in zip(tasks, report["requests"][-len(tasks) :]):
                    if task.cancelled():
                        row["error"] = "CancelledError"
                    elif task.exception() is not None:
                        error = task.exception()
                        row["error"] = f"{type(error).__name__}: {error}"
    except BaseException as error:
        failure = error
    finally:
        try:
            if await engine.stop() is not True:
                raise RuntimeError("serving workers were not confirmed reaped")
            report["closed"] = True
        except BaseException as error:
            report["cleanup_error"] = f"{type(error).__name__}: {error}"
            if failure is None:
                failure = error
    if failure is not None:
        raise failure


def source_identity(path):
    path = Path(path).resolve(strict=True)
    result = {"path": str(path), "sha256": sha256(path)}
    tracked = subprocess.run(
        ["git", "-C", str(path.parent), "ls-files", "--error-unmatch", "--", path.name],
        check=False,
        capture_output=True,
        text=True,
    )
    if tracked.returncode:
        return result
    revision = subprocess.run(
        ["git", "-C", str(path.parent), "rev-parse", "HEAD"], check=False, capture_output=True, text=True
    )
    if revision.returncode == 0:
        result["checkout_head"] = revision.stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(path.parent), "status", "--porcelain"], check=False, capture_output=True, text=True
        )
        if status.returncode:
            raise RuntimeError("cannot inspect source checkout status")
        result["checkout_status"] = status.stdout
    return result


def package_identity(name):
    distribution = importlib.metadata.distribution(name)
    result: dict[str, Any] = {"version": distribution.version}
    origin = distribution.read_text("direct_url.json")
    if origin is not None:
        result["origin"] = json.loads(origin)
        url = urlparse(result["origin"]["url"])
        if url.scheme == "file":
            source = Path(unquote(url.path)) / "pyproject.toml"
            if source.is_file():
                result["source"] = source_identity(source)
    return result


def prepare(args, report):
    from fixture import load_fixture  # noqa: PLC0415
    from pypto_serving.config.types import GenerateConfig, RuntimeConfig  # noqa: PLC0415
    from pypto_serving.model.common.executor.utils import build_pypto_run_config  # noqa: PLC0415
    from pypto_serving.model.qwen.npu_executor import _find_pypto_lib_qwen14b_dir  # noqa: PLC0415
    from pypto_serving.model.tokenizer import load_tokenizer  # noqa: PLC0415
    from pypto_serving.serving.engine.async_engine import AsyncLLMEngine, EngineConfig  # noqa: PLC0415

    run_config = build_pypto_run_config(platform="a2a3", device_ids=[args.device])
    if run_config.distributed_config.runtime != "tensormap_and_ringbuffer":
        raise ValueError("the installed serving compiler does not select the TMR baseline")
    report["loaded_modules"] = {}
    for name in ("simpler", "_task_interface", "pypto", "pypto_serving", "torch", "torch_npu"):
        module = importlib.import_module(name)
        report["loaded_modules"][name] = source_identity(module.__file__)
        report["loaded_modules"][name]["version"] = getattr(module, "__version__", None)
    report["packages"] = {name: package_identity(name) for name in ("simpler", "pypto", "pypto-serving")}
    kernel_dir = _find_pypto_lib_qwen14b_dir()
    report["kernel_sources"] = {path.name: source_identity(path) for path in sorted(kernel_dir.glob("*.py"))}
    report["consumer"] = source_identity(__file__)
    report["cann_root"] = os.environ.get("ASCEND_HOME_PATH")
    fixture = load_fixture(args.fixture)
    fixture.require_executable()
    fixture.verify_external_inputs(args.model)
    fixture.verify_artifact_and_golden()
    names = {entry["name"] for entry in fixture.manifest["checkpoint_files"]}
    index = json.loads((args.model / "model.safetensors.index.json").read_text())
    required = {"config.json", "model.safetensors.index.json", "tokenizer.json", *index["weight_map"].values()}
    if not required.issubset(names):
        raise ValueError("fixture must authenticate model config, tokenizer, index and every weight shard")
    model_config = json.loads((args.model / "config.json").read_text())
    if model_config.get("num_hidden_layers") != 40 or model_config.get("hidden_size") != 5120:
        raise ValueError("require the real 40-layer Qwen3-14B checkpoint")
    tokenizer = load_tokenizer(str(args.model))
    prompt = args.prompt.read_text(encoding="utf-8")
    ids = tokenizer.encode(prompt)
    expected = expected_sequences(
        fixture, fixture.load_golden(), ids, requests=args.requests, max_tokens=args.max_tokens
    )
    report["inputs"] = {
        "fixture_sums_sha256": sha256(args.fixture / "SHA256SUMS"),
        "checkpoint_files": fixture.manifest["checkpoint_files"],
        "prompt_ids": ids,
        "prompt_sha256": sha256(args.prompt),
        "reference_versions": fixture.manifest["versions"],
        "expected_tokens": expected,
    }
    runtime = RuntimeConfig(
        page_size=128,
        max_batch_size=16,
        max_seq_len=4096,
        max_new_tokens=args.max_tokens,
        device="cpu",
        kv_dtype="bfloat16",
        weight_dtype="bfloat16",
        max_num_batched_tokens=4096,
        ring_dep_pool=262144,
        ring_task_window=262144,
        ring_heap=4 << 30,
    )
    config = EngineConfig(
        model_id="qwen3-14b-program",
        model_dir=str(args.model),
        platform="a2a3",
        device_id=args.device,
        device_ids=(args.device,),
        executor_cls="PyptoQwen14BExecutor",
        runtime_config=runtime,
        max_num_running_reqs=args.requests,
        max_num_scheduled_tokens=4096,
        enable_prefix_cache=False,
        enable_chunk_prefill=True,
        async_scheduling=False,
    )
    generation = GenerateConfig(
        max_new_tokens=args.max_tokens, temperature=0.0, top_p=1.0, top_k=None, stream=True, ignore_eos=True
    )
    return AsyncLLMEngine(config=config, tokenizer=tokenizer), prompt, generation, expected, len(ids)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--requests", type=int, choices=(1, 16), default=16)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args(argv)
    if not 2 <= args.max_tokens <= 128 or args.rounds < 1 or args.device < 0:
        parser.error("require 2..128 tokens, positive rounds and a nonnegative device")
    args.output.mkdir(parents=True, exist_ok=False)
    args.output = args.output.resolve()
    args.model, args.fixture, args.prompt = args.model.resolve(), args.fixture.resolve(), args.prompt.resolve()
    report = {
        "status": "running",
        "cell": "B-TMR",
        "mode": "program",
        "execution": "eager",
        "runtime": "tensormap_and_ringbuffer",
        "requests": [],
        "closed": False,
        "engine_start_attempted": False,
        "engine_started": False,
        "scope": "serving prefill, decode and greedy sampling; full token gate",
        "tensor_numerics": "not_observed",
        "six_cell_acceptance": "not_qualified",
        "request_config": {"requests": args.requests, "max_tokens": args.max_tokens, "rounds": args.rounds},
        "timing_scope": "host-observed serving latency; includes scheduler; not device or host-submit latency",
        "scheduler": {
            "async_scheduling": False,
            "chunked_prefill": True,
            "max_num_scheduled_tokens": 4096,
            "long_prefill_token_threshold": 2048,
        },
    }
    result_path = args.output / "result.json"
    result_path.write_text(json.dumps(report, indent=2) + "\n")
    previous_build_dir = os.environ.get("PYPTO_PROG_BUILD_DIR")
    os.environ["PYPTO_PROG_BUILD_DIR"] = str(args.output / "artifacts")
    failure = None
    try:
        engine, prompt, generation, expected, prompt_tokens = prepare(args, report)
        asyncio.run(
            run_requests(
                engine, prompt, generation, expected, prompt_tokens=prompt_tokens, rounds=args.rounds, report=report
            )
        )
        report["status"] = "token_gate_passed"
    except BaseException as error:
        failure = error
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        if previous_build_dir is None:
            os.environ.pop("PYPTO_PROG_BUILD_DIR", None)
        else:
            os.environ["PYPTO_PROG_BUILD_DIR"] = previous_build_dir
        try:
            report["artifact_sha256"] = {
                str(path.relative_to(args.output)): sha256(path)
                for path in sorted((args.output / "artifacts").rglob("*"))
                if path.is_file()
            }
        except BaseException as error:
            report["artifact_error"] = f"{type(error).__name__}: {error}"
            report["status"] = "failed"
            if failure is None:
                failure = error
        try:
            result_path.write_text(json.dumps(report, indent=2) + "\n")
        except BaseException as error:
            if failure is None:
                failure = error
    if failure is not None:
        raise failure
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
