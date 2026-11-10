# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Immutable communication workspace engine configuration."""

from __future__ import annotations

import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class SdmaConfig:
    """SDMA resources with the provider's default configuration."""


@dataclass(frozen=True)
class RdmaConfig:
    """RDMA resources with the provider's default configuration."""


@dataclass(frozen=True)
class UrmaConfig:
    """PER_PEER queues per AIV; sq_depth requests provider depth (zero selects its default).

    Effective SQ capacity is provider-reported and can differ from the request.
    """

    jetties_per_aiv: int = 1
    sq_depth: int = 0

    def __post_init__(self):
        for name in ("jetties_per_aiv", "sq_depth"):
            value = getattr(self, name)
            if isinstance(value, bool):
                raise TypeError(f"{name} must be an integer, not bool")
            object.__setattr__(self, name, operator.index(value))
        if not 1 <= self.jetties_per_aiv <= 8:
            raise ValueError("jetties_per_aiv must be between 1 and 8; the device also limits rows per peer")
        if self.sq_depth != 0 and (self.sq_depth < 1 or self.sq_depth >= 2**31 or self.sq_depth & (self.sq_depth - 1)):
            raise ValueError("sq_depth must be zero or a power of two below 2**31")


WorkspaceEngineConfig = Union[SdmaConfig, UrmaConfig, RdmaConfig]
WorkspaceEngines = Union[Sequence[str], Mapping[str, WorkspaceEngineConfig]]
_CONFIG_TYPES = {"sdma": SdmaConfig, "urma": UrmaConfig, "rdma": RdmaConfig}


def normalize_workspace_configs(workspace_engines) -> tuple[tuple[str, WorkspaceEngineConfig], ...]:
    if isinstance(workspace_engines, str) or workspace_engines is None:
        raise TypeError("workspace_engines must be a sequence of engine names or a configuration mapping")
    names = tuple(workspace_engines)
    if any(not isinstance(name, str) or name not in _CONFIG_TYPES for name in names) or len(set(names)) != len(names):
        raise ValueError("workspace_engines requires distinct names from sdma, urma, rdma")
    if "rdma" in names and len(names) > 1:
        raise ValueError("RDMA domains do not map peer windows; use separate domains for other engines")
    configs = []
    for name in sorted(names):
        config = workspace_engines[name] if isinstance(workspace_engines, Mapping) else _CONFIG_TYPES[name]()
        if type(config) is not _CONFIG_TYPES[name]:
            raise TypeError(f"workspace_engines[{name!r}] requires {_CONFIG_TYPES[name].__name__}")
        configs.append((name, config))
    return tuple(configs)


def normalize_workspace_engines(workspace_engines) -> tuple[str, ...]:
    return tuple(name for name, _ in normalize_workspace_configs(workspace_engines))
