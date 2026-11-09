# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compare real-prefill single-request Qwen decode with Torch eager."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from callable_bridge import inspect_artifact
from reference_worker_submit import COSINE_MIN, LOGIT_RELATIVE_L2_MAX, metrics
from safetensors.torch import save_file
from torch_decode import (
    BATCH,
    BLOCKS_PER_ROW,
    HEAD_DIM,
    HEADS,
    LAYERS,
    MAX_DECODE_LEN,
    PAGE,
    TorchDecode,
    allocate_layer_caches,
    load_prefill_kv,
    validate_prompt_ids,
)
from validate_real_inputs import sha256_file, validate_checkpoint
from weights import PADDED_VOCAB, iter_kernel_weights, rope_tables

TOKENS, VOCAB, PROMPT_LENGTH = 128, 151936, 3338
MODEL_REVISION = "40c069824f4251a91eefaf281ebe4c544efd3e18"
EOS = (151645, 151643)


def kv_names():
    return tuple(f"{kind}_{layer:02d}" for layer in range(LAYERS) for kind in ("key", "value"))


def compare(actual, expected):
    if actual.shape != expected.shape or not actual.numel():
        raise ValueError("numerical tensor shape differs from the reference or is empty")
    result = metrics(actual, expected)
    return {
        **result,
        "exact": torch.equal(actual, expected),
        "passed": result["finite"]
        and result["relative_l2"] <= LOGIT_RELATIVE_L2_MAX
        and result["cosine"] >= COSINE_MIN,
    }


def compare_kv(actual, expected):
    if actual.ndim != 4 or actual.shape != expected.shape or not actual.shape[2]:
        raise ValueError("appended KV shape differs from the reference or is empty")
    steps = [compare(actual[:, :, i], expected[:, :, i]) for i in range(actual.shape[2])]
    return {
        **compare(actual, expected),
        "passed": all(step["passed"] for step in steps),
        "first_different_decode": next((i + 1 for i, step in enumerate(steps) if not step["exact"]), None),
        "first_out_of_tolerance_decode": next((i + 1 for i, step in enumerate(steps) if not step["passed"]), None),
    }


def verify_inputs(model, manifest):
    if (model / "model.safetensors").exists():
        raise ValueError("reference requires the pinned sharded checkpoint without a single-file override")
    expected = json.loads(manifest.read_text())
    if expected["revision"] != MODEL_REVISION:
        raise ValueError("input manifest revision differs from the qualified checkpoint")
    shards = {entry["path"]: entry for entry in expected["shards"]}
    if len(shards) != len(expected["shards"]) or set(shards).intersection(expected["small_files"]):
        raise ValueError("input manifest contains duplicate file records")
    records = {**expected["small_files"], **shards}
    root = model.resolve()
    for name in records:
        if not (model / name).resolve().is_relative_to(root):
            raise ValueError(f"model input path escapes checkpoint root: {name}")
    observed = validate_checkpoint(model)
    required = set(observed) | {
        path.name for path in model.iterdir() if path.is_file() and path.suffix in (".json", ".txt", ".jinja", ".model")
    }
    if missing := required.difference(records):
        raise ValueError(f"missing file digests: {', '.join(sorted(missing))}")
    for name, record in records.items():
        path = (model / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"model input path escapes checkpoint root: {name}")
        digest = observed[name] if name in observed else sha256_file(path)
        if digest != record["sha256"]:
            raise ValueError(f"model input checksum mismatch: {name}")
    return {"revision": expected["revision"], "files": {name: entry["sha256"] for name, entry in records.items()}}


def snapshot_kv(cache, start, stop):
    layers = cache.to_legacy_cache()
    if len(layers) != LAYERS or any(len(pair) != 2 for pair in layers):
        raise ValueError("reference cache must contain K and V for all layers")
    result = {}
    for layer, pair in enumerate(layers):
        for kind, value in zip(("key", "value"), pair):
            if value.shape != (1, HEADS, stop, HEAD_DIM) or value.dtype != torch.bfloat16:
                raise ValueError("reference KV shape or dtype differs from the fixed contract")
            result[f"{kind}_{layer:02d}"] = value[:, :, start:stop].to("cpu", copy=True).contiguous()
    return result


def generate_reference(model_dir, prompt, output, device):
    from transformers import AutoModelForCausalLM  # noqa: PLC0415

    if importlib.metadata.version("transformers") != "4.57.1":
        raise ValueError("Torch reference requires transformers==4.57.1")
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_dir,
            local_files_only=True,
            use_safetensors=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        .to(device)
        .eval()
    )
    result = model(torch.tensor([prompt], device=device), use_cache=True, logits_to_keep=1)
    prefill = snapshot_kv(result.past_key_values, 0, len(prompt))
    save_file(prefill, str(output / "prefill.safetensors"))
    logits, tokens = [], []
    for index in range(TOKENS):
        if result.logits.shape != (1, 1, VOCAB):
            raise ValueError("reference logits shape differs from the fixed vocabulary")
        current = result.logits[0, -1].to(device="cpu", dtype=torch.float32, copy=True)
        if not bool(torch.isfinite(current).all()):
            raise ValueError("reference produced non-finite logits")
        token = int(current.argmax())
        logits.append(current)
        tokens.append(token)
        print(f"REFERENCE_TOKEN {index + 1} {token}", flush=True)
        if token in EOS or index == TOKENS - 1:
            break
        result = model(
            torch.tensor([[token]], device=device),
            past_key_values=result.past_key_values,
            use_cache=True,
            logits_to_keep=1,
        )
    golden = {
        "tokens": torch.tensor(tokens, dtype=torch.int64),
        "logits": torch.stack(logits),
        **snapshot_kv(result.past_key_values, len(prompt), len(prompt) + len(tokens) - 1),
    }
    save_file(golden, str(output / "reference.safetensors"))
    return prefill, golden


def validate_reference(prefill, golden):
    names = set(kv_names())
    if set(prefill) != names or set(golden) != names | {"tokens", "logits"}:
        raise ValueError("reference must contain every layer's K/V and complete logits/tokens")
    if golden["tokens"].shape != (TOKENS,) or golden["tokens"].dtype != torch.int64:
        raise ValueError("reference must contain exactly 128 integer tokens")
    if bool(((golden["tokens"] < 0) | (golden["tokens"] >= VOCAB)).any()):
        raise ValueError("reference token is outside the valid vocabulary")
    if golden["logits"].shape != (TOKENS, VOCAB) or golden["logits"].dtype != torch.float32:
        raise ValueError("reference logits shape or dtype differs from the contract")
    for name in names:
        for value, length in ((prefill[name], PROMPT_LENGTH), (golden[name], TOKENS - 1)):
            if value.shape != (1, HEADS, length, HEAD_DIM) or value.dtype != torch.bfloat16:
                raise ValueError("reference KV shape or dtype differs from the contract")
    if any(not bool(torch.isfinite(value).all()) for value in (*prefill.values(), *golden.values())):
        raise ValueError("reference contains non-finite values")


def inspect_cache(caches, blocks, prefill, length, steps):
    records, appended = [], {}
    indices = blocks[0].long()
    for layer in range(LAYERS):
        for cache, kind in zip(caches, ("key", "value")):
            logical = cache[layer].index_select(0, indices).flatten(0, 1).permute(1, 0, 2).unsqueeze(0).cpu()
            name = f"{kind}_{layer:02d}"
            if steps:
                appended[name] = logical[:, :, length : length + steps].contiguous()
            records.append(
                {
                    "tensor": name,
                    "prefix_exact": torch.equal(logical[:, :, :length], prefill[name]),
                    "tail_zero": bool((logical[:, :, length + steps :] == 0).all()),
                    "finite": bool(torch.isfinite(logical).all()),
                }
            )
    return records, appended


def run_decode(*, artifact, packed, prefill, golden, device, output, forced, receipt=None):
    validate_reference(prefill, golden)
    receipt = {} if receipt is None else receipt
    caches = allocate_layer_caches(pages=BATCH * BLOCKS_PER_ROW, device=device)
    # The non-contiguous table exercises the physical page crossing at position 3456.
    block_ids = torch.arange(BLOCKS_PER_ROW, dtype=torch.int32).roll(7)
    blocks = block_ids.to(device).reshape(1, -1)
    layers = tuple((prefill[f"key_{layer:02d}"], prefill[f"value_{layer:02d}"]) for layer in range(LAYERS))
    load_prefill_kv(caches, layers, blocks)
    imported, _ = inspect_cache(caches, blocks, prefill, PROMPT_LENGTH, 0)
    if not all(row["prefix_exact"] and row["tail_zero"] and row["finite"] for row in imported):
        raise ValueError("prefill import failed exact all-layer restoration")
    backend = TorchDecode(artifact=artifact, packed=packed, caches=caches, device=device)
    stream = torch.npu.current_stream(device)
    tokens, logits, rows, appended = [int(golden["tokens"][0])], [], [], {}
    receipt.update(tokens=tokens, positions=rows, caller_stream_after_close=False)
    failure = None
    try:
        for step in range(TOKENS - 1):
            position = PROMPT_LENGTH + step
            token = int(golden["tokens"][step]) if forced else tokens[-1]
            page = int(block_ids[position // PAGE])
            slot = page * PAGE + position % PAGE
            receipt.update(step=step, position=position)
            metadata = SimpleNamespace(
                block_tables=blocks,
                seq_lens=torch.tensor([position + 1], dtype=torch.int32, device=device),
                slot_mapping=torch.tensor([slot], dtype=torch.int32, device=device),
            )
            result = backend(
                torch.tensor([token], device=device), torch.tensor([position], device=device), metadata, ("request",)
            )
            if result.shape != (1, PADDED_VOCAB) or result.dtype != torch.float32:
                raise ValueError("decode output differs from the FP32 logits ABI")
            current = result[0, :VOCAB].to("cpu", copy=True)
            if not bool(torch.isfinite(current).all()):
                raise ValueError("decode produced non-finite logits")
            logits.append(current)
            tokens.append(int(current.argmax()))
            rows.append({"position": position, "page": page, "slot": slot, "token": tokens[-1]})
            print(("COMMON_PREFIX " if forced else "FREE_GENERATION ") + json.dumps(rows[-1]), flush=True)
            if not forced and tokens[-1] in EOS:
                break
        records, appended = inspect_cache(caches, blocks, prefill, PROMPT_LENGTH, backend.steps)
        receipt.update(
            prefill_and_tail=records,
            token_equal=tokens == golden["tokens"].tolist(),
            page_boundary_covered=any(row["position"] % PAGE == 0 for row in rows),
            appended_kv={name: compare_kv(value, golden[name]) for name, value in appended.items()} if forced else {},
            logits=[compare(value, golden["logits"][i + 1]) for i, value in enumerate(logits)] if forced else [],
        )
    except BaseException as error:
        failure = error
        receipt["error"] = f"{type(error).__name__}: {error}"
    finally:
        try:
            backend.close()
            torch.ones(1, device=device).add_(1)
            stream.synchronize()
            receipt["caller_stream_after_close"] = torch.npu.current_stream(device).npu_stream == stream.npu_stream
        except BaseException as error:
            cleanup = backend.cleanup_error
            if cleanup is None and error is not failure:
                cleanup = error
            if cleanup is not None:
                receipt["cleanup_error"] = f"{type(cleanup).__name__}: {cleanup}"
            if failure is None:
                failure = error
                receipt["error"] = f"{type(error).__name__}: {error}"
        receipt.update(closed=backend.closed, failed=backend.failed, decode_dispatches=backend.steps)
        try:
            if logits:
                save_file({**appended, "logits": torch.stack(logits)}, str(output))
        except BaseException as error:
            receipt["diagnostic_error"] = f"{type(error).__name__}: {error}"
            if failure is None:
                failure = error
                receipt["error"] = receipt["diagnostic_error"]
    if failure is not None:
        raise failure
    return receipt


def validation_passed(report):
    names = set(kv_names())
    for mode in ("common_prefix", "free_generation"):
        result = report.get(mode, {})
        if any(key in result for key in ("error", "cleanup_error", "diagnostic_error")):
            return False
        if not all(
            result.get(key) is True
            for key in ("token_equal", "closed", "page_boundary_covered", "caller_stream_after_close")
        ):
            return False
        if result.get("failed") is not False or result.get("decode_dispatches") != TOKENS - 1:
            return False
        if len(result.get("tokens", [])) != TOKENS:
            return False
        records = result.get("prefill_and_tail", [])
        if len(records) != len(names) or {row.get("tensor") for row in records} != names:
            return False
        if not all(all(row.get(key) is True for key in ("prefix_exact", "tail_zero", "finite")) for row in records):
            return False
    common = report["common_prefix"]
    kv, logits = common.get("appended_kv", {}), common.get("logits", [])
    return (
        set(kv) == names
        and len(logits) == TOKENS - 1
        and all(row.get("passed") is True for row in (*kv.values(), *logits))
    )


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("artifact", "model", "prompt-ids", "input-manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", type=int, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "failed",
        "contract": {"relative_l2_max": LOGIT_RELATIVE_L2_MAX, "cosine_min": COSINE_MIN, "generated_tokens": TOKENS},
    }
    failure = None
    try:
        prompt = json.loads(args.prompt_ids.read_text())
        validate_prompt_ids(prompt)
        report["model"] = verify_inputs(args.model, args.input_manifest)
        report["prompt_ids"] = prompt
        artifact = inspect_artifact(args.artifact)
        report["artifact_sources"] = artifact.source_hashes
        report["versions"] = {
            name: importlib.metadata.version(name) for name in ("torch", "torch-npu", "transformers", "simpler")
        }
        import torch_npu  # noqa: F401, PLC0415

        torch.npu.set_device(args.device)
        device = torch.device(f"npu:{args.device}")
        prefill, golden = generate_reference(args.model, prompt, args.output, device)
        validate_reference(prefill, golden)
        gc.collect()
        torch.npu.empty_cache()
        packed = {}
        for name, value in iter_kernel_weights(args.model):
            packed[name] = value.to(device)
            del value
        cos, sin = rope_tables(args.model)
        packed.update(rope_cos=cos[:MAX_DECODE_LEN].to(device), rope_sin=sin[:MAX_DECODE_LEN].to(device))
        for mode, forced in (("common_prefix", True), ("free_generation", False)):
            report[mode] = {}
            run_decode(
                artifact=args.artifact,
                packed=packed,
                prefill=prefill,
                golden=golden,
                device=device,
                output=args.output / f"{mode}.safetensors",
                forced=forced,
                receipt=report[mode],
            )
        report["status"] = "passed" if validation_passed(report) else "failed"
    except BaseException as error:
        failure = error
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        try:
            (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        except OSError as error:
            if failure is None:
                raise
            print(f"receipt write failed: {error}", file=sys.stderr)
    if failure is not None:
        raise failure
    print("TORCH_NUMERICAL_" + report["status"].upper(), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
