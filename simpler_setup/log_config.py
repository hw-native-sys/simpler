# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared CLI log-level helper.

The CLI accepts a string from {debug, V0..V9, info, warn, error, null} or a
raw integer; we map it to a Python `logging` level and call
`logging.getLogger("simpler").setLevel(...)`. The C++ side picks up the same
level via `simpler_init` at `Worker.init()` time (one-shot snapshot) — there
is no env var; the Python "simpler" logger is the single source of truth.

pytest is intentionally not touched — it has its own `--log-cli-level` and
pyproject `log_cli_level` knobs.
"""

from __future__ import annotations

import logging

# Recognised level names → Python integer level.
# V0..V9 are simpler's INFO sub-tiers (15..24); INFO == V5 == 20.
_NAME_TO_LEVEL = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warn": logging.WARNING,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "null": 60,
    "v0": 15,
    "v1": 16,
    "v2": 17,
    "v3": 18,
    "v4": 19,
    "v5": 20,
    "v6": 21,
    "v7": 22,
    "v8": 23,
    "v9": 24,
}

LOG_LEVEL_CHOICES = [
    "debug",
    "v0",
    "v1",
    "v2",
    "v3",
    "v4",
    "v5",
    "v6",
    "v7",
    "v8",
    "v9",
    "info",
    "warn",
    "error",
    "null",
]
DEFAULT_LOG_LEVEL = "v5"  # = INFO = simpler default threshold


def parse_level(level: str | int) -> int:
    """Translate a CLI-style level into a Python logger level integer.

    Accepts either a name from `LOG_LEVEL_CHOICES` (case-insensitive) or a
    raw integer. Unknown names fall back to V5 (INFO) — silently — to match
    the previous behaviour that mapped unknown strings to INFO.
    """
    if isinstance(level, int):
        return level
    name = str(level).lower()
    return _NAME_TO_LEVEL.get(name, _NAME_TO_LEVEL[DEFAULT_LOG_LEVEL])


def configure_logging(log_level: str | int = DEFAULT_LOG_LEVEL) -> None:
    """Configure the simpler-namespaced logger from a CLI-style level.

    Args:
        log_level: name (case-insensitive) or raw integer; see LOG_LEVEL_CHOICES.
    """
    level = parse_level(log_level)
    simpler_logger = logging.getLogger("simpler")
    simpler_logger.setLevel(level)
    # Ensure root has at least one handler so the message reaches stderr;
    # this matches the prior behaviour for first-time CLI invocations.
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        root.addHandler(handler)
