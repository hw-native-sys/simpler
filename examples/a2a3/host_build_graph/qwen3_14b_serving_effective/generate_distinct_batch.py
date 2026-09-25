# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#!/usr/bin/env python3
"""Generate a compact distinct-request Qwen KV identity reference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch_npu
import transformers
from safetensors.torch import save_file
from transformers import AutoTokenizer, Qwen3ForCausalLM

BATCH = 16
PROMPT_LEN = 3338
MARKER_START = 3300
MARKER_LEN = 8
DEFAULT_STEPS = 3
LAYERS = 40
HEADS = 8
HEAD_DIM = 128
TOPK = 16


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tensor_hash(tensor) -> str:
    return hashlib.sha256(tensor.contiguous().view(torch.uint16).numpy().tobytes()).hexdigest()


def token_hash(values) -> str:
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


def main() -> None:  # noqa: PLR0912, PLR0915
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--decode-steps", type=int, default=DEFAULT_STEPS)
    args = parser.parse_args()
    if not 1 <= args.decode_steps <= 127:
        raise ValueError("decode steps must be 1..127")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    torch.npu.set_device(args.device)
    device = f"npu:{args.device}"

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)
    base_text = args.prompt.read_text()
    base_ids = tokenizer(base_text, add_special_tokens=False).input_ids
    if len(base_ids) != PROMPT_LEN:
        raise ValueError(f"base prompt token count {len(base_ids)} != {PROMPT_LEN}")
    prompt_ids, prompt_texts, markers = [], [], []
    for row in range(BATCH):
        marker = f" Request identifier {row:02d} is active."
        encoded = tokenizer(marker, add_special_tokens=False).input_ids
        if len(encoded) != MARKER_LEN:
            raise ValueError(f"marker {row} has {len(encoded)} tokens, expected {MARKER_LEN}")
        ids = list(base_ids[:MARKER_START]) + encoded
        if len(ids) != PROMPT_LEN:
            raise ValueError(f"marker row {row} does not produce {PROMPT_LEN} tokens")
        prompt_ids.append(ids)
        prompt_texts.append(tokenizer.decode(ids, skip_special_tokens=False))
        markers.append(marker)
    if len({tuple(row) for row in prompt_ids}) != BATCH:
        raise ValueError("distinct prompt rows collapsed")

    model = (
        Qwen3ForCausalLM.from_pretrained(
            args.model,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        .eval()
        .to(device)
    )
    first_tokens = []
    prefill_logits_rows = []
    output_rows = []
    input_rows = []
    topk_rows = []
    topvalue_rows = []
    logits_rows = []
    prefix_hashes = {}
    shared_prefix_written = False
    with torch.inference_mode():
        for row, ids in enumerate(prompt_ids):
            prompt = torch.tensor([ids], dtype=torch.long, device=device)
            prefill = model(input_ids=prompt, use_cache=True, logits_to_keep=1)
            cache = prefill.past_key_values
            prefill_logits = prefill.logits[0, -1].float().cpu()
            first = int(prefill_logits.argmax())
            if not torch.isfinite(prefill_logits).all():
                raise ValueError(f"non-finite prefill logits for row {row}")
            first_tokens.append(first)
            prefill_logits_rows.append(prefill_logits)
            keys_by_layer, values_by_layer = [], []
            for layer in range(LAYERS):
                keys = cache.layers[layer].keys.cpu().contiguous()
                values = cache.layers[layer].values.cpu().contiguous()
                if tuple(keys.shape) != (1, HEADS, PROMPT_LEN, HEAD_DIM):
                    raise ValueError(f"unexpected key shape at layer {layer}: {tuple(keys.shape)}")
                prefix_key = keys[0, :, :MARKER_START].contiguous()
                prefix_value = values[0, :, :MARKER_START].contiguous()
                tail_key = keys[0, :, MARKER_START:].contiguous()
                tail_value = values[0, :, MARKER_START:].contiguous()
                prefix_hashes.setdefault(
                    str(layer),
                    {
                        "key_sha256": tensor_hash(prefix_key),
                        "value_sha256": tensor_hash(prefix_value),
                    },
                )
                if row == 0:
                    save_file(
                        {"key": prefix_key, "value": prefix_value},
                        str(args.output / f"layer_{layer:02d}_prefix.safetensors"),
                    )
                    shared_prefix_written = True
                else:
                    if tensor_hash(prefix_key) != prefix_hashes[str(layer)]["key_sha256"]:
                        raise ValueError(f"prefix key differs for row {row}, layer {layer}")
                    if tensor_hash(prefix_value) != prefix_hashes[str(layer)]["value_sha256"]:
                        raise ValueError(f"prefix value differs for row {row}, layer {layer}")
                keys_by_layer.append(tail_key)
                values_by_layer.append(tail_value)
            save_file(
                {"key": torch.stack(keys_by_layer), "value": torch.stack(values_by_layer)},
                str(args.output / f"request_{row:02d}_tail.safetensors"),
            )
            token = torch.tensor([[first]], dtype=torch.long, device=device)
            row_inputs, row_outputs, row_topids, row_topvalues, row_logits = [], [], [], [], []
            for step in range(args.decode_steps):
                row_inputs.append(int(token.item()))
                result = model(input_ids=token, past_key_values=cache, use_cache=True, logits_to_keep=1)
                logits = result.logits[0, -1].float().cpu()
                if not torch.isfinite(logits).all():
                    raise ValueError(f"non-finite decode logits row={row}, step={step}")
                values, indices = torch.topk(logits, TOPK)
                sampled = int(logits.argmax())
                row_outputs.append(sampled)
                row_topids.append(indices.to(torch.int32))
                row_topvalues.append(values)
                row_logits.append(logits)
                token = torch.tensor([[sampled]], dtype=torch.long, device=device)
            input_rows.append(torch.tensor(row_inputs, dtype=torch.int32))
            output_rows.append(torch.tensor(row_outputs, dtype=torch.int32))
            topk_rows.append(torch.stack(row_topids))
            topvalue_rows.append(torch.stack(row_topvalues))
            logits_rows.append(torch.stack(row_logits))
            del cache, prefill
            torch.npu.empty_cache()
            print(f"REQUEST {row} first={first} outputs={row_outputs}", flush=True)

    if not shared_prefix_written:
        raise AssertionError("no shared prefix written")
    save_file(
        {
            "prompt_token_ids": torch.tensor(prompt_ids, dtype=torch.int32),
            "first_token_ids": torch.tensor(first_tokens, dtype=torch.int32),
            "prefill_logits": torch.stack(prefill_logits_rows),
        },
        str(args.output / "metadata.safetensors"),
    )
    save_file(
        {
            "decode_input_token_ids": torch.stack(input_rows, dim=1),
            "decode_output_token_ids": torch.stack(output_rows, dim=1),
            "decode_topk_token_ids": torch.stack(topk_rows, dim=1),
            "decode_topk_values": torch.stack(topvalue_rows, dim=1),
        },
        str(args.output / "decode.safetensors"),
    )
    save_file(
        {f"row_{row:02d}": logits_rows[row] for row in range(BATCH)},
        str(args.output / "decode_logits.safetensors"),
    )
    for row, text in enumerate(prompt_texts):
        (args.output / f"prompt_{row:02d}.txt").write_text(text)

    index = json.loads((args.model / "model.safetensors.index.json").read_text())
    manifest = {
        "schema": "qwen-reference-distinct-batch-v1",
        "model": "Qwen3-14B",
        "batch": BATCH,
        "prompt_tokens": PROMPT_LEN,
        "layers": LAYERS,
        "kv_heads": HEADS,
        "head_dim": HEAD_DIM,
        "dtype": "bfloat16",
        "layout": "B,H,S,D logical; physical pages are BSND",
        "marker_window": {"start": MARKER_START, "length": MARKER_LEN},
        "decode_steps": args.decode_steps,
        "capacity_decode_steps": 127,
        "topk": TOPK,
        "sampling": "greedy argmax",
        "eos_policy": "fixed identity dispatch count; EOS does not stop this qualification",
        "shared_prefix": "all rows share the verified prefix before marker_window; each row has independent tail KV",
        "metadata": "metadata.safetensors",
        "decode_reference": "decode.safetensors",
        "decode_logits": "decode_logits.safetensors",
        "requests": [
            {
                "request_id": f"qwen-distinct-{row:02d}",
                "prompt_path": f"prompt_{row:02d}.txt",
                "prompt_token_ids_sha256": token_hash(prompt_ids[row]),
                "first_token_id": first_tokens[row],
                "tail_path": f"request_{row:02d}_tail.safetensors",
            }
            for row in range(BATCH)
        ],
        "reference_single_bundle_sha256": "6659b433406cc99d9bd20df604e3faf10f07526e3d4b752af1bb4dc3035b73e3",
        "checkpoint_sha256": {
            name: digest(args.model / name)
            for name in ["config.json", "model.safetensors.index.json", *sorted(set(index["weight_map"].values()))]
        },
        "tokenizer_sha256": digest(args.model / "tokenizer.json"),
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "attention": "eager",
        "device": device,
        "prefix_hashes": prefix_hashes,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    provenance = {
        "generator_sha256": digest(Path(__file__).resolve()),
        "prompt_source_sha256": digest(args.prompt),
        "distinct_prompt_rows": BATCH,
        "marker_tokens": markers,
        "validation": {
            "all_prompt_rows_distinct": True,
            "shared_prefix_bitwise_equal": True,
            "decode_steps": args.decode_steps,
        },
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    files = sorted(path for path in args.output.iterdir() if path.is_file() and path.name != "SHA256SUMS")
    (args.output / "SHA256SUMS").write_text("".join(f"{digest(path)}  {path.name}\n" for path in files))
    print(json.dumps({"output": str(args.output), "first_tokens": first_tokens, "files": len(files)}), flush=True)


if __name__ == "__main__":
    main()
