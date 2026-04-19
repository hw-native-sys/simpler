# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared CLI log-level helper.

Used by SceneTestCase.run_module (`python test_*.py`) so the standalone entry
point wires `--log-level` the same way as the rest of the codebase and defaults
to INFO. Also propagates via `PTO_LOG_LEVEL` env var so subprocesses spawned
by the parallel scheduler inherit the level.

pytest is intentionally not touched — it has its own `--log-cli-level` and
pyproject `log_cli_level` knobs.
"""

import logging
import os

LOG_LEVEL_CHOICES = ["error", "warn", "info", "debug"]
DEFAULT_LOG_LEVEL = "info"

_LEVEL_MAP = {
    "error": logging.ERROR,
    "warn": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
}


def configure_logging(log_level: str = DEFAULT_LOG_LEVEL) -> None:
    """Configure root logger for a CLI entry point.

    Args:
        log_level: one of "error" / "warn" / "info" / "debug" (case-insensitive).
            Unknown values fall back to INFO.
    """
    log_level = log_level.lower()
    level = _LEVEL_MAP.get(log_level, logging.INFO)
    logging.basicConfig(
        level=level,
        format="[%(levelname)s] %(message)s",
        force=True,
    )
    os.environ["PTO_LOG_LEVEL"] = log_level
