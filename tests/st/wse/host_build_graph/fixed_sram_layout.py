# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Test data layouts over one ordinary device Buffer; no per-Core allocation API."""

import json
from dataclasses import dataclass
from weakref import WeakKeyDictionary

from simpler_setup.environment import PROJECT_ROOT

_LAYOUTS = WeakKeyDictionary()


@dataclass
class Window:
    layout: object
    offset: int
    nbytes: int
    released: bool = False

    @property
    def root(self):
        return self.layout.buffer

    @property
    def base(self):
        return self.root.base + self.offset

    @property
    def identity(self):
        return self.root.identity

    def tensor(self, **kwargs):
        return self.root.tensor(**{**kwargs, "byte_offset": self.offset + kwargs.get("byte_offset", 0)})


class Layout:
    def __init__(self, worker):
        spec = json.loads((PROJECT_ROOT / "src/wse/platform/config/default_simulation.json").read_text())
        self.cores = spec["hardware"]["core_count"]

        def expand(value):
            return [value] * self.cores if isinstance(value, int) else value

        capacities = expand(spec["hardware"]["sram_bytes_per_core"])
        reserves = expand(spec["runtime"]["sram_reserve_bytes_per_core"])
        self.sizes = [c - r for c, r in zip(capacities, reserves, strict=True)]
        self.starts = [sum(self.sizes[:core]) for core in range(self.cores)]
        self.cursors = [0] * self.cores
        self.buffer = worker.malloc(sum(self.sizes))
        self.views = 0


def layout_view(worker, size, *, core=0, alignment=64):
    layout = _LAYOUTS.get(worker)
    if layout is None:
        layout = _LAYOUTS[worker] = Layout(worker)
    offset = (layout.cursors[core] + alignment - 1) & -alignment
    assert offset + size <= layout.sizes[core], "test layout exceeds its fixed Core window"
    layout.cursors[core] = offset + size
    layout.views += 1
    return Window(layout, layout.starts[core] + offset, size)


def copy_to(worker, window, value, **kwargs):
    if isinstance(window, Window):
        kwargs["dst_offset"] = kwargs.get("dst_offset", 0) + window.offset
        window = window.root
    worker.copy_to(window, value, **kwargs)


def copy_from(worker, value, window, **kwargs):
    if isinstance(window, Window):
        kwargs["src_offset"] = kwargs.get("src_offset", 0) + window.offset
        window = window.root
    worker.copy_from(value, window, **kwargs)


def release_view(worker, window):
    if not isinstance(window, Window):
        worker.free(window)
        return
    assert not window.released, "test view released twice"
    layout = window.layout
    if layout.views == 1:
        worker.free(layout.buffer)
        del _LAYOUTS[worker]
    layout.views -= 1
    window.released = True
