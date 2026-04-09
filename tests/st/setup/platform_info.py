# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Platform, arch, and runtime discovery utilities.

Shared module used by RuntimeBuilder and build_runtimes.py
to eliminate duplicated platform mapping
and runtime/variant discovery logic.
"""

import importlib.util
from pathlib import Path

from .environment import PROJECT_ROOT

PLATFORM_MAP: dict[str, tuple[str, str]] = {
    "a2a3": ("a2a3", "onboard"),
    "a2a3sim": ("a2a3", "sim"),
    "a5": ("a5", "onboard"),
    "a5sim": ("a5", "sim"),
}

_REVERSE_MAP: dict[tuple[str, str], str] = {v: k for k, v in PLATFORM_MAP.items()}

ARCHS = ("a2a3", "a5")
TARGETS = ("host", "aicpu", "aicore")


def parse_platform(platform: str) -> tuple[str, str]:
    """Parse platform string into (arch, variant).

    >>> parse_platform("a2a3sim")
    ('a2a3', 'sim')
    """
    if platform not in PLATFORM_MAP:
        raise ValueError(f"Unknown platform: {platform}. Supported: {', '.join(PLATFORM_MAP)}")
    return PLATFORM_MAP[platform]


def to_platform(arch: str, variant: str) -> str:
    """Convert (arch, variant) back to platform string.

    >>> to_platform("a2a3", "sim")
    'a2a3sim'
    """
    key = (arch, variant)
    if key not in _REVERSE_MAP:
        raise ValueError(f"No platform for ({arch}, {variant})")
    return _REVERSE_MAP[key]


def discover_runtimes(arch: str) -> list[str]:
    """Return sorted list of runtime names that have a build_config.py."""
    runtime_base = PROJECT_ROOT / "src" / arch / "runtime"
    if not runtime_base.is_dir():
        return []
    return sorted(d.name for d in runtime_base.iterdir() if d.is_dir() and (d / "build_config.py").exists())


def discover_variants(arch: str) -> list[str]:
    """Return sorted list of platform variants (excluding include/src dirs)."""
    platform_base = PROJECT_ROOT / "src" / arch / "platform"
    if not platform_base.is_dir():
        return []
    return sorted(d.name for d in platform_base.iterdir() if d.is_dir() and d.name not in ("include", "src"))


def load_build_config(config_path: Path) -> dict:
    """Load BUILD_CONFIG dict from a build_config.py file."""
    spec = importlib.util.spec_from_file_location("build_config", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load build config from {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.BUILD_CONFIG
