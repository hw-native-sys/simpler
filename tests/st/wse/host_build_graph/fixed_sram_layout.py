# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Place test objects at aligned offsets in Worker-owned per-Core SRAM windows."""

from dataclasses import dataclass, field
from weakref import WeakKeyDictionary

_LAYOUTS = WeakKeyDictionary()


@dataclass
class Layout:
    cursors: dict = field(default_factory=dict)
    views: int = 0


@dataclass
class Window:
    address: object
    nbytes: int
    released: bool = False

    @property
    def base(self):
        return self.address.base

    @property
    def identity(self):
        return self.address.identity

    def tensor(self, **kwargs):
        return self.address.tensor(**kwargs)


def layout_view(worker, size, *, core=0, alignment=64):
    layout = _LAYOUTS.setdefault(worker, Layout())
    local = worker.get_base_addr(coreid=core)
    offset = (layout.cursors.get(core, 0) + alignment - 1) & -alignment
    assert offset + size <= local.nbytes, "test layout exceeds its fixed Core window"
    layout.cursors[core] = offset + size
    layout.views += 1
    return Window(local + offset, size)


def copy_to(worker, window, value, **kwargs):
    worker.copy_to(window.address, value, **kwargs)


def copy_from(worker, value, window, **kwargs):
    worker.copy_from(value, window.address, **kwargs)


def release_view(worker, window):
    assert not window.released, "test view released twice"
    window.released = True
    layout = _LAYOUTS[worker]
    layout.views -= 1
    if layout.views == 0:
        del _LAYOUTS[worker]
