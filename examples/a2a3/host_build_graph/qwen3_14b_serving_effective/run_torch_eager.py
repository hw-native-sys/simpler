# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exercise the fixed decode artifact with real weights and controlled zero-prefix KV."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch_npu  # noqa: F401
from callable_bridge import inspect_artifact
from torch_decode import (
    BATCH,
    BLOCKS_PER_ROW,
    MAX_DECODE_LEN,
    PAGE,
    TorchDecode,
    allocate_layer_caches,
    validate_prompt_ids,
)
from validate_real_inputs import validate_checkpoint
from weights import PADDED_VOCAB, iter_kernel_weights, rope_tables


def _check(condition, message):
    if not condition:
        raise RuntimeError(message)


def _run_steps(backend, caches, token, device, stream, receipt):
    caller_stream = stream.npu_stream
    blocks = torch.arange(BLOCKS_PER_ROW, dtype=torch.int32, device=device).reshape(1, -1)
    for step in range(2):
        position = 3338 + step
        receipt.update(step=step, position=position)
        context = f"step={step} position={position}"
        page = int(blocks[0, position // PAGE])
        metadata = SimpleNamespace(
            block_tables=blocks,
            seq_lens=torch.tensor([position + 1], dtype=torch.int32, device=device),
            slot_mapping=torch.tensor([page * PAGE + position % PAGE], dtype=torch.int32, device=device),
        )
        output = backend(token, torch.tensor([position], dtype=torch.int64, device=device), metadata, ("controlled",))
        _check(
            output.shape == (1, PADDED_VOCAB) and output.dtype == torch.float32,
            f"{context}: invalid logits shape={tuple(output.shape)} dtype={output.dtype}",
        )
        valid = output[:, :151936]
        _check(bool(torch.isfinite(valid).all()), f"{context}: non-finite logits")
        token = valid.argmax(-1)
        for cache in caches:
            value = cache[:, page, position % PAGE]
            _check(bool(torch.isfinite(value).all()), f"{context}: non-finite KV")
            _check(bool((value.flatten(1).count_nonzero(dim=1) > 0).all()), f"{context}: missing layer KV write")
        _check(torch.npu.current_stream(device).npu_stream == caller_stream, f"{context}: caller stream changed")
        receipt["steps"].append(
            {
                "position": position,
                "shape": list(output.shape),
                "dtype": str(output.dtype),
                "selected_token": int(token[0]),
                "kv_written": True,
            }
        )
        print("TORCH_DECODE_STEP " + json.dumps(receipt["steps"][-1]), flush=True)
    _check(backend.steps == 2 and not backend.failed, "decode did not complete both smoke steps")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt-ids", type=Path, required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be a new file")
    receipt = {
        "workload": "real-weights-controlled-zero-prefix-kv",
        "task_queue": os.environ.get("TASK_QUEUE_ENABLE"),
        "active_requests": 1,
        "steps": [],
        "caller_stream_after_close": False,
    }
    backend, failure = None, None
    try:
        receipt["framework_versions"] = {
            name: importlib.metadata.version(name) for name in ("torch", "torch-npu", "simpler")
        }
        prompt = json.loads(args.prompt_ids.read_text())
        validate_prompt_ids(prompt)
        receipt["model_sha256"] = validate_checkpoint(args.model)
        inspected = inspect_artifact(args.artifact)
        receipt["artifact_source_hashes"] = inspected.source_hashes
        torch.npu.set_device(args.device)
        device = torch.device(f"npu:{args.device}")
        stream = torch.npu.current_stream(device)
        packed = {}
        for name, tensor in iter_kernel_weights(args.model):
            packed[name] = tensor.to(device)
            print(f"TORCH_WEIGHT_READY {name} shape={list(tensor.shape)}", flush=True)
        _check(bool(packed), "checkpoint conversion returned no weights")
        del tensor
        cos, sin = rope_tables(args.model)
        packed.update(rope_cos=cos[:MAX_DECODE_LEN].to(device), rope_sin=sin[:MAX_DECODE_LEN].to(device))
        caches = allocate_layer_caches(pages=BATCH * BLOCKS_PER_ROW, device=device)
        backend = TorchDecode(artifact=args.artifact, packed=packed, caches=caches, device=device)
        receipt["bound_parameters"] = {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)} for name, value in backend.values.items()
        }
        _run_steps(
            backend, caches, torch.tensor([prompt[-1]], dtype=torch.int64, device=device), device, stream, receipt
        )
    except BaseException as error:
        failure = error
        receipt["error"] = f"{type(error).__name__}: {error}"
    finally:
        if backend is not None:
            try:
                backend.close()
                _check(backend.closed, "decode backend did not close")
                torch.ones(1, device=device).add_(1)
                stream.synchronize()
                _check(
                    torch.npu.current_stream(device).npu_stream == stream.npu_stream,
                    "caller stream changed after close",
                )
                receipt["caller_stream_after_close"] = True
            except BaseException as error:
                receipt["cleanup_error"] = f"{type(error).__name__}: {error}"
                if failure is None:
                    failure = error
                    receipt["error"] = receipt["cleanup_error"]
        receipt.update(
            status="passed" if failure is None else "failed",
            closed=backend.closed if backend else False,
            decode_dispatches=backend.steps if backend else 0,
        )
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(receipt, indent=2) + "\n")
        except BaseException as error:
            if failure is None:
                failure = error
            else:
                print(f"TORCH_RECEIPT_WRITE_FAILED {type(error).__name__}: {error}", flush=True)
    if failure is not None:
        raise failure
    print(
        "TORCH_ARTIFACT_SMOKE_PASS "
        + json.dumps({"output": str(args.output), "dispatches": receipt["decode_dispatches"]}),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
