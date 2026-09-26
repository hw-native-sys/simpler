# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Load validated logical KV and materialize independent paged request storage."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import load_file


class ReferenceFixture:
    def __init__(self, root, batch=16, page_size=128):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.validation = json.loads((self.root / "validation.json").read_text())
        required = {
            "schema": "qwen-reference-logical-kv-v1",
            "layers": 40,
            "kv_heads": 8,
            "head_dim": 128,
            "dtype": "bfloat16",
            "layout": "B,H,S,D",
        }
        if any(self.manifest.get(k) != v for k, v in required.items()):
            raise ValueError("Reference geometry does not match Qwen3-14B ABI")
        if not self.validation["reference_roundtrip_passed"]:
            raise ValueError("Reference KV has not passed restoration validation")
        if batch < 1 or page_size != 128:
            raise ValueError("Require positive batch and page_size=128")
        for line in (self.root / "SHA256SUMS").read_text().splitlines():
            expected, name = line.split(maxsplit=1)
            target = (self.root / name).resolve()
            if not target.is_relative_to(self.root.resolve()):
                raise ValueError("Checksum path escapes reference root")
            h = hashlib.sha256()
            with target.open("rb") as f:
                for block in iter(lambda: f.read(8 << 20), b""):
                    h.update(block)
            if h.hexdigest() != expected:
                raise ValueError(f"Checksum mismatch: {name}")
        self.batch, self.page_size = batch, page_size
        self.length = int(self.manifest["prompt_tokens"])
        self.steps = int(self.validation["decode_steps"])
        if self.length < 1 or self.steps < 1 or self.length + self.steps > 4096:
            raise ValueError("Reference sequence exceeds decode ABI capacity")
        capacity_steps = int(self.manifest.get("capacity_decode_steps", self.steps))
        self.pages_per_request = (self.length + capacity_steps + page_size - 1) // page_size
        self.num_pages = self.batch * self.pages_per_request
        self.block_table = torch.full((self.batch, 32), -1, dtype=torch.int32)
        self.block_table[:, : self.pages_per_request] = torch.arange(self.num_pages, dtype=torch.int32).view(batch, -1)
        self.metadata = load_file(str(self.root / "metadata.safetensors"))
        self.golden = load_file(str(self.root / "decode.safetensors"))

    def adapter_fixture(self):
        """Translate logical reference metadata to the standalone decode contract."""
        steps = [self.step(i) for i in range(self.steps)]
        golden = {
            name: torch.stack([s[name] for s in steps]).to(torch.int32)
            for name in ("seq_lens", "slot_mapping", "input_token_ids", "reference_output_token_ids")
        }
        golden["decode_input_token_ids"] = golden.pop("input_token_ids")
        golden["decode_output_token_ids"] = golden.pop("reference_output_token_ids")
        return SimpleNamespace(
            manifest={
                "decode_dispatches_remaining": self.steps,
                "physical_layout": {"page_size": self.page_size, "num_pages": self.num_pages},
            },
            metadata={
                "first_generated_token_ids": self.metadata["first_token_id"].repeat(self.batch).to(torch.int32),
                "block_table": self.block_table.clone(),
            },
            load_golden=lambda: golden,
        )

    def verify_model(self, model):
        provenance = json.loads((self.root / "provenance.json").read_text())
        for name, expected in provenance["checkpoint_sha256"].items():
            path = (Path(model) / name).resolve()
            if not path.is_relative_to(Path(model).resolve()):
                raise ValueError("Checkpoint path escapes model root")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 << 20), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                raise ValueError(f"Checkpoint checksum mismatch: {name}")

    def step(self, index):
        if not 0 <= index < self.steps:
            raise ValueError("Decode step out of range")
        position = self.length + index
        pages = self.block_table[:, position // self.page_size]
        return {
            "seq_lens": torch.full((self.batch,), position + 1, dtype=torch.int32),
            "slot_mapping": pages * self.page_size + position % self.page_size,
            "block_table": self.block_table.clone(),
            "input_token_ids": self.golden["decode_input_token_ids"][index].repeat(self.batch),
            "reference_output_token_ids": self.golden["decode_output_token_ids"][index].repeat(self.batch),
        }

    def layer(self, layer):
        """Yield key/value physical pages [page, token, head, dim] for one layer.

        Each request has independent pages. Only unused tail capacity is zero.
        Caller uploads layers at layer * num_pages * 8 * page_size * 128 elements.
        """
        if not 0 <= layer < self.manifest["layers"]:
            raise ValueError("Layer index out of range")
        logical = load_file(str(self.root / f"layer_{layer:02d}.safetensors"))
        for value in logical.values():
            if (
                value.shape != (1, 8, self.length, 128)
                or value.dtype != torch.bfloat16
                or not torch.isfinite(value).all()
            ):
                raise ValueError(f"Invalid KV tensor in layer {layer}")
        rows = [(logical["key"][0], logical["value"][0])] * self.batch
        for kind_index, kind in enumerate(("key", "value")):
            physical = torch.zeros((self.num_pages, self.page_size, 8, 128), dtype=torch.bfloat16)
            for row, tensors in enumerate(rows):
                padded = torch.zeros((8, self.pages_per_request * self.page_size, 128), dtype=torch.bfloat16)
                padded[:, : self.length] = tensors[kind_index]
                pages = padded.reshape(8, self.pages_per_request, self.page_size, 128).permute(1, 2, 0, 3)
                physical[row * self.pages_per_request : (row + 1) * self.pages_per_request] = pages
            yield kind, physical

    def validate_batch_layout(self):
        for index in (0, 117, 118, 126):
            step = self.step(index)
            assert torch.unique(step["slot_mapping"]).numel() == self.batch
            assert (step["slot_mapping"] >= 0).all()
            assert (step["slot_mapping"] < self.num_pages * self.page_size).all()
        for layer in range(self.manifest["layers"]):
            logical = load_file(str(self.root / f"layer_{layer:02d}.safetensors"))
            for kind, physical in self.layer(layer):
                for row in range(self.batch):
                    ids = self.block_table[row, : self.pages_per_request].long()
                    back = physical[ids].permute(2, 0, 1, 3).reshape(1, 8, -1, 128)[:, :, : self.length]
                    assert torch.equal(back, logical[kind]), (layer, kind, row)
                # Verify an actual write to request zero cannot alter request one.
                if self.batch > 1:
                    sibling_page = int(self.block_table[1, 0])
                    sibling = physical[sibling_page].clone()
                    physical[int(self.block_table[0, 0])].fill_(123)
                    assert torch.equal(physical[sibling_page], sibling)
        return {
            "batch": self.batch,
            "num_pages": self.num_pages,
            "pages_per_request": self.pages_per_request,
            "all_40_layers_16_rows_bitwise_roundtrip": True,
            "request_storage_independent": True,
        }
