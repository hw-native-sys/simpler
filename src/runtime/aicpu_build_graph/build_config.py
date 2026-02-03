# Runtime build configuration
# All paths are relative to this file's directory (src/runtime/)

import os
from pathlib import Path


def _optional_kernels_aicpu_dir() -> list[str]:
    """
    Optional: include example-provided AICPU graph builder sources.

    Convention:
      examples/<example>/kernels/aicpu/*.cpp

    The runner sets PTO_KERNELS_DIR to the selected kernels directory.
    """
    kernels_dir = os.environ.get("PTO_KERNELS_DIR")
    if not kernels_dir:
        return []
    d = (Path(kernels_dir) / "aicpu").resolve()
    if not d.is_dir():
        return []
    return [str(d)]


_AICPU_EXTRA_DIRS = _optional_kernels_aicpu_dir()

BUILD_CONFIG = {
    "aicore": {
        "include_dirs": ["runtime"],
        "source_dirs": ["aicore", "runtime"]
    },
    "aicpu": {
        "include_dirs": ["runtime"] + _AICPU_EXTRA_DIRS,
        "source_dirs": ["aicpu", "runtime"] + _AICPU_EXTRA_DIRS
    },
    "host": {
        "include_dirs": ["runtime"],
        "source_dirs": ["host", "runtime"]
    }
}
