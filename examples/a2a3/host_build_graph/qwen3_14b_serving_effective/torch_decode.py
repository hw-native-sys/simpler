# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Fixed-shape Qwen decode over caller-owned torch storage."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from weights import HEAD_DIM
from weights import NUM_LAYERS as LAYERS

BATCH, HEADS, PAGE = 16, 8, 128
MAX_DECODE_LEN = 3338 + 128
BLOCKS_PER_ROW = (MAX_DECODE_LEN + PAGE - 1) // PAGE
WAIT_TIMEOUT = 120
_GENERATIONS = itertools.count(1)
PROMPT_SHA256 = "83d41b1d59bda5281cf2489944632d71247d38c4fdf37bf347d4935297d4d00d"


def validate_prompt_ids(ids):
    if not isinstance(ids, list) or len(ids) != 3338 or any(type(value) is not int or value < 0 for value in ids):
        raise ValueError("prompt must contain exactly 3338 token IDs")
    if hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest() != PROMPT_SHA256:
        raise ValueError("prompt token IDs differ from the qualified workload")


def allocate_layer_caches(*, pages, device):
    if pages < BATCH * BLOCKS_PER_ROW:
        raise ValueError(f"fixed Qwen decode requires at least {BATCH * BLOCKS_PER_ROW} pages")
    return tuple(
        torch.zeros((LAYERS, pages, PAGE, HEADS, HEAD_DIM), dtype=torch.bfloat16, device=device) for _ in range(2)
    )


class TorchDecode:
    def __init__(self, *, artifact: Path, packed, caches, device):
        from callable_bridge import inspect_artifact  # noqa: PLC0415
        from real_worker_submit import _dtype, _shape  # noqa: PLC0415
        from simpler.task_interface import CallConfig  # noqa: PLC0415

        self.artifact = inspect_artifact(artifact)
        if "sampled_ids_host" in self.artifact.parameter_names:
            raise ValueError("torch eager decode requires the 25-argument DEVICE-output ABI")
        self.device = torch.device(device)
        self.caches = caches
        self.worker = None
        self.run = None
        self.sources = []
        self.failed = False
        self.closed = False
        self.request_id = None
        self.position = None
        self.blocks = None
        self.padding_pages = None
        self.steps = 0
        self.cleanup_error = None
        self.config = CallConfig()
        self.config.enable_dep_gen = False
        self.config.enable_chip_swimlane = 0
        pages = caches[0].shape[1]
        expected = (LAYERS, pages, PAGE, HEADS, HEAD_DIM)
        for cache in caches:
            if tuple(cache.shape) != expected or not cache.is_contiguous() or cache.dtype != torch.bfloat16:
                raise ValueError("KV backing must be contiguous BF16 layer/BSND storage")
        self.values = {**packed, "k_cache": caches[0].view(-1, HEAD_DIM), "v_cache": caches[1].view(-1, HEAD_DIM)}
        specs = json.loads((artifact / "distributed_meta.json").read_text())["params"]
        for spec in specs:
            name = spec["name"].split("__ssa_", 1)[0]
            shape, dtype = _shape(name, spec["shape"], pages=pages), _dtype(spec["dtype"])
            if name == "block_table":
                shape = (BATCH * BLOCKS_PER_ROW,)
            if name not in self.values:
                self.values[name] = torch.zeros(shape, dtype=dtype, device=self.device)
            value = self.values[name]
            if tuple(value.shape) != shape or value.dtype != dtype or not value.is_contiguous():
                raise ValueError(f"tensor does not match callable ABI: {name}")
            if value.device != self.device:
                raise ValueError(f"tensor is on another device: {name}")

    def _initialize(self):
        from simpler.buffer import Buffer  # noqa: PLC0415
        from simpler.task_interface import ArgDirection, TaskArgs, Tensor, TensorArgType  # noqa: PLC0415
        from simpler.worker import Worker  # noqa: PLC0415

        from simpler_setup.torch_interop import torch_dtype_to_datatype  # noqa: PLC0415

        signature = self.artifact.callable_spec["orchestration"]["signature"]
        if len(self.artifact.parameter_names) != len(signature):
            raise ValueError("callable parameter and signature counts differ")
        callable_ = self.artifact.compile()
        self.worker = Worker(level=2, device_id=self.device.index, platform="a2a3", runtime="host_build_graph")
        self.callable = self.worker.register(callable_)
        self.worker.init(borrowed_context_generation=next(_GENERATIONS))
        self.args = TaskArgs()
        for name, direction in zip(self.artifact.parameter_names, signature):
            value = self.values[name]
            source = Buffer.wrap(
                address=value.data_ptr(),
                nbytes=value.numel() * value.element_size(),
                location=self.worker.device_location,
            )
            self.sources.append(source)
            access = TensorArgType.INPUT if direction == ArgDirection.IN else TensorArgType.OUTPUT_EXISTING
            if direction == ArgDirection.INOUT or name in ("k_cache", "v_cache"):
                access = TensorArgType.INOUT
            self.args.add_tensor(Tensor(source, shapes=value.shape, dtype=torch_dtype_to_datatype(value.dtype)), access)

    def _prepare_single_request(self, input_ids, positions, metadata, request_ids):
        """Build the fixed batch with independent pages around one live request."""
        if tuple(input_ids.shape) != (1,) or tuple(positions.shape) != (1,):
            raise ValueError("single-request padding requires one token and one position")
        if len(request_ids) != 1 or (self.steps and request_ids[0] != self.request_id):
            raise ValueError("decode requires the same single request")
        table = metadata.block_tables
        if table.ndim != 2 or table.shape[0] != 1 or table.shape[1] > 32:
            raise ValueError("single-request block table must be [1, <=32]")
        if tuple(metadata.seq_lens.shape) != (1,) or tuple(metadata.slot_mapping.shape) != (1,):
            raise ValueError("single-request metadata must describe exactly one request")
        device = input_ids.device
        table = table.to(device=device, dtype=torch.int32)
        active_position = int(positions[0])
        if not 3338 <= active_position < MAX_DECODE_LEN - 1:
            raise ValueError("fixed decode exceeds the qualified generation range")
        if int(metadata.seq_lens[0]) != active_position + 1:
            raise ValueError("KV length does not include exactly the current decode token")
        active_blocks = (active_position + PAGE) // PAGE
        if active_blocks > table.shape[1]:
            raise ValueError("single-request block table is shorter than the active position")
        row_pages = tuple(table[0].tolist())
        active_pages = row_pages[:active_blocks]
        columns = self.values["block_table"].numel() // BATCH
        if any(page != -1 for page in row_pages[columns:]):
            raise ValueError("block table has populated columns beyond the fixed callable capacity")
        reserved = tuple(page for page in row_pages[:columns] if page >= 0)
        pages = self.caches[0].shape[1]
        if (
            len(set(reserved)) != len(reserved)
            or any(page >= pages for page in reserved)
            or any(page < 0 for page in active_pages)
        ):
            raise ValueError("active request KV pages are outside the backing capacity")
        if self.blocks is not None and active_pages[: len(self.blocks)] != self.blocks:
            raise ValueError("live request KV pages cannot be replaced")
        expected_slot = active_pages[active_position // PAGE] * PAGE + active_position % PAGE
        if int(metadata.slot_mapping[0]) != expected_slot:
            raise ValueError("slot mapping disagrees with the single request")
        expected_position = 3338 if self.position is None else self.position + 1
        if active_position != expected_position:
            raise ValueError("decode positions must be consecutive after the 3338-token prefill")
        padding_count = (BATCH - 1) * BLOCKS_PER_ROW
        if self.padding_pages is None or set(reserved).intersection(self.padding_pages):
            used = set(reserved)
            selected = [page for page in range(pages - 1, -1, -1) if page not in used]
            if len(selected) < padding_count:
                raise ValueError("not enough disjoint KV pages for inactive fixed-shape rows")
            self.padding_pages = tuple(selected[:padding_count])
        padded_table = torch.full((BATCH, columns), -1, dtype=torch.int32, device=device)
        width = min(table.shape[1], columns)
        padded_table[0, :width].copy_(table[0, :width])
        padded_table[1:, :BLOCKS_PER_ROW] = torch.tensor(self.padding_pages, dtype=torch.int32, device=device).reshape(
            BATCH - 1, BLOCKS_PER_ROW
        )
        padded_seq_lens = metadata.seq_lens.to(device=device, dtype=torch.int32).reshape(-1)[:1].repeat(BATCH)
        padded_slots = padded_table[:, active_position // PAGE] * PAGE + active_position % PAGE
        # Keep a valid token in every fixed ABI row; the inactive rows are
        # isolated by their private pages and are omitted from the returned logits.
        padded_input = input_ids[0].repeat(BATCH)
        padded_metadata = SimpleNamespace(
            seq_lens=padded_seq_lens, block_tables=padded_table, slot_mapping=padded_slots
        )
        return padded_input, padded_metadata, active_position, active_pages

    def __call__(self, input_ids, positions, metadata, request_ids):
        if self.failed or self.closed:
            raise RuntimeError("failed decode backend cannot admit another step")
        try:
            input_ids, metadata, position, blocks = self._prepare_single_request(
                input_ids, positions, metadata, request_ids
            )
            stream = torch.npu.current_stream(self.device)
        except ValueError:
            raise
        except BaseException:
            self.failed = True
            raise
        try:
            self.values["sampled_ids_in"][:, 0].copy_(input_ids)
            self.values["seq_lens"].copy_(metadata.seq_lens)
            self.values["block_table"].view(BATCH, -1).copy_(metadata.block_tables)
            self.values["slot_mapping"].copy_(metadata.slot_mapping)
            # Flush the framework emission queue and finish prior consumers before reusing storage.
            stream.synchronize()
            if self.worker is None:
                self._initialize()
            if self.worker is None:
                raise RuntimeError("decode Worker was not initialized")
            self.run = self.worker.submit(self.callable, self.args, self.config, caller_stream=stream.npu_stream)
            self.run.wait(timeout=WAIT_TIMEOUT)
        except TimeoutError as error:
            self.failed = True
            raise TimeoutError(f"decode step={self.steps} position={position} timed out") from error
        except BaseException:
            self.failed = True
            raise
        self.position = position
        self.request_id = request_ids[0]
        self.blocks = blocks
        self.steps += 1
        return self.values["out"][: len(request_ids)]

    def close(self):  # noqa: PLR0912 -- cleanup retains primary and cleanup errors separately
        if self.closed:
            return
        primary_error = None
        cleanup_errors = []
        try:
            if self.run is not None:
                try:
                    self.run.wait(timeout=WAIT_TIMEOUT)
                except TimeoutError as error:
                    self.failed = True
                    raise TimeoutError(f"decode close step={self.steps} position={self.position} timed out") from error
                except BaseException as error:
                    primary_error = error
                    self.run = None
            try:
                torch.npu.current_stream(self.device).synchronize()
            except BaseException as error:
                cleanup_errors.append(error)
            if self.worker is not None:
                try:
                    self.worker.close()
                except BaseException as error:
                    cleanup_errors.append(error)
                else:
                    self.worker = None
            remaining_sources = []
            for source in self.sources:
                try:
                    source.close()
                except BaseException as error:
                    cleanup_errors.append(error)
                    remaining_sources.append(source)
            self.sources = remaining_sources
            if cleanup_errors:
                self.cleanup_error = cleanup_errors[0]
            if not cleanup_errors:
                self.closed = True
            if primary_error is not None:
                if cleanup_errors:
                    primary_error.__context__ = cleanup_errors[0]
                raise primary_error
            if cleanup_errors:
                raise cleanup_errors[0]
        except BaseException:
            self.failed = True
            raise
        self.closed = True
