# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The Python mirror of ``simpler::tmr::TaskId`` (src/common/tensormap_and_ringbuffer/task_id.h).

Mirrors both directions, so this file is the one place the layout is stated on the
Python side: a mint here composes the same field the decode here reads, and both read
its position from the constants below.

``make()`` is for a test that has to hand a decoder an id, and for nothing else. Every
id a real run carries is minted by the host or device runtime, so DFX tooling decodes
ids and never mints one -- a tool calling ``make()`` is stating a layout it should be
asking this class about.
"""

from __future__ import annotations

from dataclasses import dataclass

from simpler_setup.tools._runtime_dispatch import normalize_task_id_int


@dataclass(frozen=True)
class TaskId:
    """A 64-bit tmr task handle: ``(ring_id << 32) | local_id``."""

    raw: int

    # task_id.h inlines this shift as a literal in ring()/make() rather than naming
    # it -- this is still the single Python-side source of truth for that value.
    RING_SHIFT = 32
    # The width C++ gives the field by returning it as uint8_t. Named rather than
    # inlined so the layout test has a value to compare the header against: a ring
    # widened there and not here reads back a ring the runtime cannot mint.
    RING_BITS = 8
    # The largest ring a mint can encode. ``ring()`` refuses anything above it rather
    # than masking, so bits 63:40 cannot be dropped without being reported.
    RING_MASK = (1 << RING_BITS) - 1
    # The low field ends where the ring field begins, which is what bounds a local id.
    # Derived rather than written out so one shift states both boundaries.
    _LOCAL_MASK = (1 << RING_SHIFT) - 1

    _INVALID_RAW = (1 << 64) - 1

    @classmethod
    def parse(cls, v):
        """``TaskId`` for any int-convertible value (negative values are normalized to unsigned), else None."""
        raw = normalize_task_id_int(v)
        return None if raw is None else cls(raw)

    @classmethod
    def make(cls, ring_id, local_id):
        """``(ring_id << 32) | local_id``. Mirrors ``TaskId::make``.

        Mirrors its truncation too: the C++ mint takes a ``uint8_t`` ring and a
        ``uint32_t`` local id, so a value too wide for its field is silently cut rather
        than refused. A mirror that refused instead would make a test disagree with the
        runtime it stands in for.
        """
        return cls(((ring_id & cls.RING_MASK) << cls.RING_SHIFT) | (local_id & cls._LOCAL_MASK))

    def is_valid(self):
        return self.raw != self._INVALID_RAW

    def ring(self):
        """This id's ring, read from the same field ``TaskId::ring()`` reads.

        Raises:
            ValueError: the field holds more than ``RING_BITS`` can, which no minted
                handle does -- a ring id is a ``uint8_t``, so bits 63:40 of a real
                handle are zero. Carries the raw value so the caller can locate which
                record produced it. C++ narrows with ``static_cast<uint8_t>`` and would
                report a ring the runtime cannot mint; reporting the id instead is what
                keeps a corrupt record from rendering as a plausible one.
        """
        value = self.raw >> self.RING_SHIFT
        if value > self.RING_MASK:
            raise ValueError(
                f"task_id {self.raw:#018x} has ring {value} (bits 63:32), which exceeds the "
                f"{self.RING_BITS}-bit field a mint can fill -- either the invalid sentinel "
                f"or a corrupt record"
            )
        return value

    def local_id(self):
        return self.raw & self._LOCAL_MASK

    def display(self):
        return f"r{self.ring()}t{self.local_id()}"
