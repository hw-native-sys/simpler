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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt-ids", type=Path, required=True)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--backend", choices=("native", "simpler"), default="simpler")
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.max_tokens <= 128 or args.rounds < 1:
        parser.error("max-tokens must be in [1, 128] and rounds must be positive")
    if args.backend == "simpler" and args.artifact is None:
        parser.error("--artifact is required for the Simpler backend")
    ids = json.loads(args.prompt_ids.read_text())
    if not isinstance(ids, list) or len(ids) != 3338 or any(type(value) is not int or value < 0 for value in ids):
        parser.error("prompt-ids must contain exactly 3338 token IDs")
    if args.output.exists():
        parser.error("output must be a new file")
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(args.device)
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    case = str(Path(__file__).resolve().parent)
    os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, (case, os.environ.get("PYTHONPATH"))))
    from torch_decode import validate_prompt_ids  # noqa: PLC0415
    from vllm import LLM, SamplingParams  # noqa: PLC0415

    validate_prompt_ids(ids)
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
        max_num_seqs=16,
        max_num_batched_tokens=16 * 3338,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        async_scheduling=False,
        num_gpu_blocks_override=449,
        **options,
    )
    sampling = SamplingParams(temperature=0, max_tokens=args.max_tokens)
    rows = []
    try:
        expected_steps = 0
        for _ in range(args.rounds):
            start = time.perf_counter()
            result = llm.generate([{"prompt_token_ids": ids} for _ in range(16)], sampling, use_tqdm=False)
            elapsed = time.perf_counter() - start
            tokens = [list(request.outputs[0].token_ids) for request in result]
            if len(tokens) != 16 or not all(tokens):
                raise RuntimeError("the complete cohort did not return generated tokens")
            expected_steps += max(map(len, tokens)) - 1
            rows.append({"tokens": tokens, "wall_seconds": elapsed})
        stats = llm.collective_rpc("simpler_stats") if args.backend == "simpler" else None
        if stats is not None and (len(stats) != 1 or stats[0] != {"steps": expected_steps, "failed": False}):
            raise RuntimeError(f"decode dispatches do not reconcile with returned tokens: {stats}")
        if args.reference is not None:
            reference = json.loads(args.reference.read_text())["rounds"]
            if [row["tokens"] for row in rows] != [row["tokens"] for row in reference]:
                raise RuntimeError("generated token sequences differ from the native reference")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"backend": args.backend, "rounds": rows, "simpler": stats}, indent=2) + "\n")
    finally:
        if args.backend == "simpler":
            llm.collective_rpc("close_simpler")
        llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
