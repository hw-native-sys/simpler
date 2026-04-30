# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Simpler unified logging — single Python-side knob, propagated to C++.

User-facing API: just `logging.getLogger("simpler").setLevel(int)`. Optional
constants (V0..V9, NUL) for readability. The integer threshold encodes both
severity and INFO sub-verbosity:

    DEBUG    = 10
    V0..V4   = 15..19      (sub-INFO, more verbose)
    V5       = 20  (= INFO; default threshold)
    V6..V9   = 21..24      (above-INFO, more must-see)
    WARN     = 30
    ERROR    = 40
    NUL      = 60          (suppress all)

C++ side uses two axes (severity enum, info_v int) — `_split_threshold()`
converts the Python integer into that pair, which `Worker.init()` forwards
to `ChipWorker::init(..., log_level, log_info_v)` once. The platform SO's
`simpler_init()` then propagates that snapshot to HostLogger, runner state,
and (onboard only) CANN dlog.

This module configures the "simpler" logger at import so that an unconfigured
user gets the V5 default rather than Python's WARNING root inheritance.
"""

import logging

# DEFAULT_LOG_THRESHOLD is exposed by the _task_interface nanobind module so
# Python and C++ share one constant. During a fresh `pip install -e .` the
# pre-existing .so may be stale or absent, so fall back to the hardcoded
# value (kept in sync manually with src/{a5,a2a3}/platform/src/host/host_log.h).
try:
    from _task_interface import DEFAULT_LOG_THRESHOLD as _NATIVE_DEFAULT  # pyright: ignore[reportMissingImports]
except (ImportError, AttributeError):
    _NATIVE_DEFAULT = 20

# Public verbosity constants (Python integer levels).
V0 = 15
V1 = 16
V2 = 17
V3 = 18
V4 = 19
V5 = 20  # alias of logging.INFO
V6 = 21
V7 = 22
V8 = 23
V9 = 24
NUL = 60

DEFAULT_THRESHOLD = _NATIVE_DEFAULT  # 20 (V5)

_LOGGER_NAME = "simpler"
_logger = logging.getLogger(_LOGGER_NAME)
if _logger.level == logging.NOTSET:
    _logger.setLevel(DEFAULT_THRESHOLD)

# Severity enum integers — must mirror C++ simpler::log::LogLevel.
_SEV_DEBUG = 0
_SEV_INFO = 1
_SEV_WARN = 2
_SEV_ERROR = 3
_SEV_NUL = 4


def get_logger() -> logging.Logger:
    """Return the simpler-namespaced Python logger."""
    return _logger


def _split_threshold(t: int) -> tuple[int, int]:
    """Convert a Python integer threshold into the C++ (severity, info_v) pair.

    Severity is the floor (CANN-aligned: 0=DEBUG..4=NUL); info_v is the INFO
    verbosity threshold (only meaningful when severity == INFO, otherwise 0).

    Banding:
        t <= 10           → (DEBUG, 0)        # everything visible
        11 <= t <= 14     → (DEBUG, 0)        # below V0; treat as DEBUG band
        15 <= t <= 24     → (INFO,  t - 15)   # V0..V9 → info_v 0..9
        25 <= t <= 39     → (WARN,  0)        # below WARN; round up
        40 <= t <= 59     → (ERROR, 0)
        t >= 60           → (NUL,   0)
    """
    if t <= 14:
        return (_SEV_DEBUG, 0)
    if t <= 24:
        return (_SEV_INFO, t - 15)
    if t <= 39:
        return (_SEV_WARN, 0)
    if t <= 59:
        return (_SEV_ERROR, 0)
    return (_SEV_NUL, 0)


def get_current_config() -> tuple[int, int]:
    """Return current (severity, info_v) for forwarding to ChipWorker.init().

    Reads the simpler logger's effective level — which respects user
    setLevel() calls and falls back to DEFAULT_THRESHOLD when unconfigured
    (we set that at module import).
    """
    return _split_threshold(_logger.getEffectiveLevel())
