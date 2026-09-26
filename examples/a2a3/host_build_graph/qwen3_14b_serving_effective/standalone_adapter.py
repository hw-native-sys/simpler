# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Stateful Qwen decode adapter for one fixture-backed Worker.submit stream.

The adapter owns host metadata progression and the sampled-token feedback edge.
It deliberately leaves model/artifact allocation to the caller: a serving worker
can bind the generated artifact ABI to its own resident buffers, then use this
class to submit one immutable metadata snapshot per decode step.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Callable

import torch

BATCH = 16


@dataclass(frozen=True)
class DecodeStep:
    """Immutable host inputs for one decode dispatch."""

    index: int
    input_token_ids: torch.Tensor
    seq_lens: torch.Tensor
    slot_mapping: torch.Tensor
    block_table: torch.Tensor
    expected_output_token_ids: torch.Tensor


class StandaloneDecodeAdapter:
    """Advance fixture metadata and enforce the autoregressive token edge.

    ``fixture`` is the object returned by ``fixture.load_fixture`` and ``golden``
    is its ``load_golden()`` result. A caller must invoke ``complete_step`` only
    after the corresponding ``RunHandle`` has reached a terminal state. This
    makes host-visible sampled output and slot reuse share one explicit boundary.
    """

    def __init__(self, fixture: Any, golden: dict[str, torch.Tensor] | None = None) -> None:
        self.fixture = fixture
        self.golden = golden if golden is not None else fixture.load_golden()
        self.metadata = fixture.metadata
        self.steps = int(fixture.manifest["decode_dispatches_remaining"])
        self.page_size = int(fixture.manifest["physical_layout"]["page_size"])
        self.blocks_per_row = int(self.metadata["block_table"].shape[1])
        self._step = 0
        self._inflight = False
        self._last_output = self.metadata["first_generated_token_ids"].clone()
        self._block_table = self.metadata["block_table"].clone()
        self._observed: list[torch.Tensor] = []

        self._validate_contract()

    @property
    def completed_steps(self) -> int:
        return self._step

    @property
    def finished(self) -> bool:
        return self._step == self.steps

    @property
    def observed_tokens(self) -> tuple[torch.Tensor, ...]:
        return tuple(item.clone() for item in self._observed)

    def _validate_contract(self) -> None:
        if self.metadata["block_table"].shape != (BATCH, self.blocks_per_row):
            raise ValueError("fixture block_table must have one row per request")
        required = {"decode_input_token_ids", "decode_output_token_ids", "seq_lens", "slot_mapping"}
        missing = required - set(self.golden)
        if missing:
            raise ValueError(f"golden is missing tensors: {sorted(missing)}")
        for name in required:
            value = self.golden[name]
            if value.dtype != torch.int32 or tuple(value.shape) != (self.steps, BATCH):
                raise ValueError(f"golden tensor contract mismatch: {name}")
        if self.metadata["first_generated_token_ids"].shape != (BATCH,):
            raise ValueError("first_generated_token_ids must have shape [16]")

    def _install_step_metadata(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_lens = self.golden["seq_lens"][index].clone()
        slot_mapping = self.golden["slot_mapping"][index].clone()
        positions = seq_lens - 1
        if not torch.equal(slot_mapping.remainder(self.page_size), positions.remainder(self.page_size)):
            raise ValueError(f"golden slot mapping offset mismatch at step {index}")
        logical_blocks = positions.div(self.page_size, rounding_mode="floor")
        page_ids = slot_mapping.div(self.page_size, rounding_mode="floor")
        if int(logical_blocks.min()) < 0 or int(logical_blocks.max()) >= self.blocks_per_row:
            raise ValueError(f"golden logical block exceeds fixture table at step {index}")
        if int(page_ids.min()) < 0 or int(page_ids.max()) >= int(self.fixture.manifest["physical_layout"]["num_pages"]):
            raise ValueError(f"golden page id exceeds fixture storage at step {index}")
        for row in range(BATCH):
            self._block_table[row, int(logical_blocks[row])] = page_ids[row]
        return seq_lens, slot_mapping, self._block_table.clone()

    def next_step(self) -> DecodeStep:
        """Return the next immutable step and reserve the stream until completion."""
        if self.finished:
            raise RuntimeError("decode stream is already complete")
        if self._inflight:
            raise RuntimeError("the previous decode step has not completed")
        index = self._step
        expected_input = self.metadata["first_generated_token_ids"].clone() if index == 0 else self._last_output.clone()
        golden_input = self.golden["decode_input_token_ids"][index]
        if not torch.equal(expected_input, golden_input):
            raise ValueError(f"golden input token chain diverged at step {index}")
        seq_lens, slot_mapping, block_table = self._install_step_metadata(index)
        self._inflight = True
        return DecodeStep(
            index=index,
            input_token_ids=expected_input,
            seq_lens=seq_lens,
            slot_mapping=slot_mapping,
            block_table=block_table,
            expected_output_token_ids=self.golden["decode_output_token_ids"][index].clone(),
        )

    def complete_step(self, sampled_ids: torch.Tensor) -> None:
        """Consume sampled IDs after the run fence and advance the feedback edge."""
        if not self._inflight:
            raise RuntimeError("no decode step is waiting for completion")
        values = sampled_ids
        if values.dtype != torch.int32:
            values = values.to(torch.int32)
        if values.ndim == 2 and values.shape[0] == BATCH and values.shape[1] >= 1:
            values = values[:, 0]
        if tuple(values.shape) != (BATCH,):
            raise ValueError(f"sampled_ids must have shape [16] or [16, N], got {tuple(values.shape)}")
        expected = self.golden["decode_output_token_ids"][self._step]
        if not torch.equal(values, expected):
            mismatch = int((values != expected).nonzero(as_tuple=False)[0][0])
            raise ValueError(f"sampled token mismatch at step {self._step}, row {mismatch}")
        self._observed.append(values.clone())
        self._last_output = values.clone()
        self._step += 1
        self._inflight = False

    def submit_step(
        self,
        worker: Any,
        *,
        chip_handle: Any,
        task_args: Any,
        config: Any,
        worker_id: int = 0,
    ) -> tuple[DecodeStep, Any]:
        """Submit the reserved step and return its metadata plus ``RunHandle``.

        The caller must wait on the returned handle, read sampled IDs from its
        ABI-owned output buffer, and call ``complete_step`` before requesting the
        next step.
        """
        step = self.next_step()

        def submit_next_level(orch: Any, _args: Any, cfg: Any) -> None:
            orch.submit_next_level(chip_handle, task_args, cfg, worker=worker_id)

        try:
            handle = worker.submit(submit_next_level, args=None, config=config)
        except BaseException:
            self._inflight = False
            raise
        return step, handle


def bind_task_args(
    task_args: Any,
    parameter_specs: Sequence[Any],
    signature: Sequence[Any],
    buffers: dict[str, Any],
    *,
    direction_to_tag: Callable[[Any], Any],
    dtype_to_runtime: Callable[[str], Any],
) -> Any:
    """Bind generated host ABI buffers while preserving explicit directions."""
    if len(parameter_specs) != len(signature):
        raise ValueError("parameter specs and callable signature have different lengths")
    for spec, direction in zip(parameter_specs, signature):
        buffer = buffers.get(spec.name)
        if buffer is None:
            raise ValueError(f"missing ABI buffer: {spec.name}")
        task_args.add_tensor(
            buffer.tensor(tuple(spec.shape), dtype_to_runtime(spec.dtype)),
            direction_to_tag(direction),
        )
    return task_args
