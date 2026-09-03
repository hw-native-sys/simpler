# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Simpler runtime — public Python surface.

Host-side log filter setup happens during worker initialization. A hierarchical
`Worker` seeds the parent before its first fork; `ChipWorker.init` (see
`simpler.task_interface`) repeats that initialization in each child before the
C++ `_ChipWorker.init` loads runtime modules. The native extension owns the
shared state; each host module contains its own logger implementation and binds
to that state immediately after it is loaded. Onboard `simpler_init` also maps
the threshold onto CANN's coarser dlog ladder.

`Worker` and the `task_interface` submodule resolve on first attribute access
rather than at import time: both pull in the `_task_interface` extension, so
eager imports here would make `import simpler` fail wherever that extension is
missing or stale, including for callers that only want the logging helpers.
"""

import importlib
from typing import Any

# Importing _log configures the Python-side simpler logger to TIMING if unset.
from ._log import (
    DEFAULT_THRESHOLD,
    NUL,
    TIMING,
    get_current_config,
    get_logger,
)

__all__ = [
    "DEFAULT_THRESHOLD",
    "Worker",
    "register_chip_control_extension",
    "NUL",
    "TIMING",
    "get_current_config",
    "get_logger",
    "comm_endpoints",
    "task_interface",
]

# name -> (module, attribute). Resolved by __getattr__ on first access.
_LAZY_ATTRS = {
    "Worker": (f"{__name__}.worker", "Worker"),
    "register_chip_control_extension": (
        f"{__name__}.worker",
        "register_chip_control_extension",
    ),
}
_LAZY_SUBMODULES = ("comm_endpoints", "task_interface")


def __getattr__(name: str) -> Any:
    """Resolve the extension-backed surface on first access (PEP 562)."""
    if name in _LAZY_ATTRS:
        module_name, attribute = _LAZY_ATTRS[name]
        return getattr(importlib.import_module(module_name), attribute)
    if name in _LAZY_SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
