# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Immutable configuration used by both domain allocation backends."""

from __future__ import annotations

import operator
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .task_interface import CommBufferSpec


@dataclass(frozen=True)
class CommDomainRef:
    """Creator identity and generation of an already established domain."""

    owner_instance_id: bytes
    domain_id: int
    generation: int

    def __post_init__(self):
        if not isinstance(self.owner_instance_id, bytes) or len(self.owner_instance_id) != 8:
            raise ValueError("domain creator must be an 8-byte instance identity")
        if not any(self.owner_instance_id) or not 0 <= self.domain_id < 2**64 or not 0 < self.generation < 2**64:
            raise ValueError("invalid communication domain reference")


@dataclass(frozen=True)
class DomainConfig:
    members: tuple
    window_size: int
    buffers: tuple
    retain_after_run: bool
    topology: tuple
    workspace_engines: tuple[str, ...] = ()

    @classmethod
    def capture(cls, members, window_size, buffers, retain_after_run, topology, workspace_engines=()):
        snapshot = tuple(
            (
                str(b.name),
                str(b.dtype),
                operator.index(b.count),
                operator.index(b.nbytes),
                bool(b.load_from_host),
                bool(b.store_to_host),
            )
            for b in buffers
        )
        members = tuple(members)
        window_size = operator.index(window_size)
        if not members or len(members) > 64 or len(set(members)) != len(members):
            raise ValueError("domain members must contain 1 to 64 distinct chips")
        if window_size <= 0:
            raise ValueError("domain window_size must be positive")
        if window_size % 64:
            raise ValueError("domain window_size must be a multiple of 64 bytes")
        if len({b[0] for b in snapshot}) != len(snapshot) or any(not b[0] or b[3] <= 0 for b in snapshot):
            raise ValueError("domain buffers require unique names and positive nbytes")
        for name, _dtype, _count, nbytes, _load_from_host, _store_to_host in snapshot:
            if nbytes % 64:
                raise ValueError(
                    f"domain buffer {name!r}: nbytes={nbytes} must be a multiple of 64 bytes; "
                    "the caller must pad the physical buffer size and include it in window_size"
                )
        if sum(b[3] for b in snapshot) > window_size:
            raise ValueError("domain buffers exceed window_size")
        if isinstance(workspace_engines, str) or workspace_engines is None:
            raise TypeError("workspace_engines must be a sequence of engine names")
        names = tuple(workspace_engines)
        if any(name not in ("sdma", "urma", "rdma") for name in names) or len(set(names)) != len(names):
            raise ValueError("workspace_engines requires distinct names from sdma, urma, rdma")
        workspace_engines = tuple(sorted(names))
        if "rdma" in workspace_engines and len(workspace_engines) > 1:
            raise ValueError("RDMA domains do not map peer windows; use separate domains for other engines")
        return cls(members, window_size, snapshot, bool(retain_after_run), tuple(topology), workspace_engines)

    def workspace_mask(self) -> int:
        return sum(1 << ("sdma", "urma", "rdma").index(name) for name in self.workspace_engines)

    def specs(self) -> list[CommBufferSpec]:
        from .task_interface import CommBufferSpec  # noqa: PLC0415

        return [CommBufferSpec(*buffer) for buffer in self.buffers]

    def difference(self, other: DomainConfig) -> str:
        for item in fields(self):
            before, after = getattr(self, item.name), getattr(other, item.name)
            if before != after:
                return f"{item.name}: {before!r} != {after!r}"
        return ""
