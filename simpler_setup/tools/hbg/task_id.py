# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The Python mirror of ``simpler::hbg::TaskId`` (src/common/host_build_graph/task_id.h).

Mirrors both directions, so this file is the one place the layout is stated on the
Python side: a mint here composes the same fields the decode here reads, and both
read their positions from the constants below.

The ``make_*`` constructors are for a test that has to hand a decoder an id, and for
nothing else. Every id a real run carries is minted by the host or device runtime, so
DFX tooling decodes ids and never mints one -- a tool calling ``make_*`` is stating a
layout it should be asking this class about.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from simpler_setup.tools._runtime_dispatch import normalize_task_id_int


class Space(enum.Enum):
    """Which id space a task id belongs to, held in the top two bits. Mirrors ``TaskId::Space``."""

    GLOBAL = 0
    SUB_TASK = 1
    PARAM = 2


@dataclass(frozen=True)
class TaskId:
    """A 64-bit hbg task handle: ``(id space << 62) | (parent << 32) | local id``."""

    raw: int

    # Position knowledge: the single Python-side source of truth for this layout.
    SPACE_SHIFT = 62
    PARENT_SHIFT = 32
    PARENT_BITS = 20
    PARENT_MASK = (1 << PARENT_BITS) - 1
    # The space occupies everything above PARENT_SHIFT's field up to bit 63, which is
    # two bits; C++ reads it as the enum's own width.
    _SPACE_MASK = 0x3
    # The low field ends where the parent field begins, which is what bounds a local
    # id. Derived rather than written out so one shift states both boundaries.
    _LOCAL_MASK = (1 << PARENT_SHIFT) - 1

    _INVALID_RAW = (1 << 64) - 1

    @classmethod
    def parse(cls, v):
        """``TaskId`` for any int-convertible value (negative values are normalized to unsigned), else None."""
        raw = normalize_task_id_int(v)
        return None if raw is None else cls(raw)

    @classmethod
    def make_global(cls, local_id):
        """A task of the run itself: space GLOBAL, local id in the low 32. Mirrors ``TaskId::make_global``."""
        return cls((Space.GLOBAL.value << cls.SPACE_SHIFT) | (local_id & cls._LOCAL_MASK))

    @classmethod
    def make_sub_task(cls, parent_id, local_id):
        """A materialized sub-task: space SUB_TASK, parent in bits 51:32, index in the low 32.

        Mirrors ``TaskId::make_sub_task``, including its truncation: the C++ mint masks
        the parent to ``PARENT_MASK`` and narrows the local id to 32 bits, so a value too
        wide for its field is silently cut rather than refused. A mirror that refused
        instead would make a test disagree with the runtime it stands in for.
        """
        return cls(
            (Space.SUB_TASK.value << cls.SPACE_SHIFT)
            | ((parent_id & cls.PARENT_MASK) << cls.PARENT_SHIFT)
            | (local_id & cls._LOCAL_MASK)
        )

    @classmethod
    def make_param(cls, param_index):
        """A boundary parameter: space PARAM, index in the low 32, parent zero. Mirrors ``TaskId::make_param``."""
        return cls((Space.PARAM.value << cls.SPACE_SHIFT) | (param_index & cls._LOCAL_MASK))

    def is_valid(self):
        return self.raw != self._INVALID_RAW

    def space(self):
        """This id's Space, read from the same field ``TaskId::space()`` reads.

        Raises:
            ValueError: the top two bits are ``11`` (the invalid sentinel, or a
                corrupt id) -- carries the raw value so the caller can locate which
                record produced it. A filter picking this runtime's ids out of a mixed
                stream still gets its answer: another space compares unequal, and only
                a field no mint can produce raises.
        """
        value = (self.raw >> self.SPACE_SHIFT) & self._SPACE_MASK
        try:
            return Space(value)
        except ValueError:
            raise ValueError(
                f"task_id {self.raw:#018x} has id space {value} (top two bits), which no real "
                f"space takes; expected one of {[s.value for s in Space]} -- either the invalid "
                f"sentinel or a corrupt record"
            ) from None

    def local_id(self):
        return self.raw & self._LOCAL_MASK

    def parent_id(self):
        """The modular task this id belongs to, meaningful for SUB_TASK only."""
        return (self.raw >> self.PARENT_SHIFT) & self.PARENT_MASK

    def is_global(self):
        return self.space() is Space.GLOBAL

    def space_name(self):
        return self.space().name

    def display(self):
        """``t{local}`` for GLOBAL, ``g{parent}t{local}`` for SUB_TASK, ``p{local}`` for PARAM."""
        space = self.space()
        if space is Space.SUB_TASK:
            return f"g{self.parent_id()}t{self.local_id()}"
        if space is Space.PARAM:
            return f"p{self.local_id()}"
        return f"t{self.local_id()}"
