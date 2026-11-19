# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run the fixed Qwen eager cohort through the native or Simpler vLLM worker."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def _generate(llm, ids, sampling, args, report):
    for _ in range(args.rounds):
        start = time.perf_counter()
        llm.sleep(level=0, mode="keep")
        llm.enqueue([{"prompt_token_ids": ids} for _ in range(args.requests)], sampling, use_tqdm=False)
        llm.wake_up(tags=["scheduling"])
        result = llm.wait_for_completion(use_tqdm=False)
        tokens = [list(request.outputs[0].token_ids) for request in result]
        if len(tokens) != args.requests or any(len(row) != args.max_tokens for row in tokens):
            raise RuntimeError("the complete workload did not return the configured token count")
        report["rounds"].append({"tokens": tokens, "wall_seconds": time.perf_counter() - start})
    if args.backend == "simpler":
        stats = llm.collective_rpc("simpler_stats")
        report["simpler"] = stats
        if stats != [{"steps": args.rounds * (args.max_tokens - 1), "failed": False}]:
            raise RuntimeError(f"decode dispatches do not reconcile with returned tokens: {stats}")
    if args.reference is not None:
        reference = json.loads(args.reference.read_text())
        if reference["backend"] != "native" or [r["tokens"] for r in report["rounds"]] != [
            r["tokens"] for r in reference["rounds"]
        ]:
            raise RuntimeError("generated token sequences differ from the native reference")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt-ids", type=Path, required=True)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--backend", choices=("native", "simpler"), default="simpler")
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--requests", type=int, choices=(1, 16), default=16)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 2 <= args.max_tokens <= 128 or args.rounds < 1:
        parser.error("max-tokens must be in [2, 128] and rounds must be positive")
    if args.backend == "simpler" and args.artifact is None:
        parser.error("--artifact is required for the Simpler backend")
    if args.output.exists():
        parser.error("output must be a new file")
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(args.device)
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    case = str(Path(__file__).resolve().parent)
    os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, (case, os.environ.get("PYTHONPATH"))))
    from torch_decode import validate_prompt_ids  # noqa: PLC0415
    from validate_real_inputs import validate_checkpoint  # noqa: PLC0415
    from vllm import LLM, SamplingParams  # noqa: PLC0415

    report = {"backend": args.backend, "requests": args.requests, "rounds": [], "simpler": None, "closed": False}
    llm, failure = None, None
    try:
        ids = json.loads(args.prompt_ids.read_text())
        validate_prompt_ids(ids)
        report["model_sha256"] = validate_checkpoint(args.model)
        options = {}
        if args.backend == "simpler":
            options.update(
                worker_cls="vllm_eager.SimplerWorker",
                additional_config={"simpler": {"artifact": str(args.artifact.resolve(strict=True))}},
            )
        llm = LLM(
            model=str(args.model.resolve(strict=True)),
            dtype="bfloat16",
            enforce_eager=True,
            max_model_len=4096,
            max_num_seqs=args.requests,
            max_num_batched_tokens=max(4096, args.requests * 3338),
            enable_prefix_caching=False,
            enable_chunked_prefill=False,
            async_scheduling=False,
            block_size=128,
            num_gpu_blocks_override=449,
            **options,
        )
        sampling = SamplingParams(temperature=0, max_tokens=args.max_tokens)
        _generate(llm, ids, sampling, args, report)
    except BaseException as error:
        failure = error
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        if llm is not None:
            try:
                if args.backend == "simpler":
                    llm.collective_rpc("close_simpler")
            except BaseException as error:
                report["cleanup_error"] = f"{type(error).__name__}: {error}"
                if failure is None:
                    failure = error
            try:
                llm.llm_engine.engine_core.shutdown()
                report["closed"] = True
            except BaseException as error:
                report["shutdown_error"] = f"{type(error).__name__}: {error}"
                if failure is None:
                    failure = error
        report["status"] = "passed" if failure is None else "failed"
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
        except BaseException as error:
            if failure is None:
                failure = error
    if failure is not None:
        raise failure
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
