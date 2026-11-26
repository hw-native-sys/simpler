# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared CLI log-level helper.

The CLI accepts a string from {debug, info, timing, warn, error, null} or a
raw integer; we map it to a Python `logging` level and call
`logging.getLogger("simpler").setLevel(...)`. The C++ side picks up the same
level through the native state seeded at `Worker.init()` time (one-shot
snapshot) — there is no env var; the Python "simpler" logger is the single
source of truth.

pytest is intentionally not touched — it has its own `--log-cli-level` and
pyproject `log_cli_level` knobs.
"""

from __future__ import annotations

import logging

TIMING = 25

# Recognised level names → Python integer level.
_NAME_TO_LEVEL = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "timing": TIMING,
    "warn": logging.WARNING,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "null": 60,
}

LOG_LEVEL_CHOICES = ["debug", "info", "timing", "warn", "error", "null"]
DEFAULT_LOG_LEVEL = "timing"


def parse_level(level: str | int) -> int:
    """Translate a CLI-style level into a Python logger level integer.

    Accepts either a name from `LOG_LEVEL_CHOICES` (case-insensitive) or a
    raw integer. Unknown names fall back to TIMING.
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
    logging.basicConfig(format="[%(levelname)s] %(message)s")
