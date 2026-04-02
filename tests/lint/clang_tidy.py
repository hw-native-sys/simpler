# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run clang-tidy on changed files using per-target compile databases.

For each changed file, every build/cache/<arch>/<variant>/<runtime>/<target>/
compile_commands.json that contains the file is used as-is (no merging).
This ensures each file is analysed with the exact flags of its compilation unit.

If no sim build cache exists, the sim runtimes are built first:
    python examples/scripts/build_runtimes.py
"""

import json
import os
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_BUILD_RUNTIMES = _ROOT / "examples" / "scripts" / "build_runtimes.py"
_CACHE_DIR = _ROOT / "build" / "cache"

# Suppress compiler flags that are valid for GCC but unknown to clang.
_SUPPRESS_ARGS = [
    "--extra-arg=-Wno-unknown-warning-option",
    "--extra-arg=-Wno-unused-command-line-argument",
]


def _ensure_sim_cache() -> None:
    """Build all detectable sim runtimes if no sim compile databases exist."""
    if list(_CACHE_DIR.glob("*/sim/*/*/compile_commands.json")):
        return

    print("No sim compile databases found — building runtimes...", flush=True)
    result = subprocess.run(
        [sys.executable, str(_BUILD_RUNTIMES), "--platforms", "a2a3sim", "a5sim"], check=False, cwd=_ROOT
    )
    if result.returncode != 0:
        print("ERROR: build_runtimes.py failed; cannot run clang-tidy", file=sys.stderr)
        sys.exit(result.returncode)


def _build_file_index() -> dict[str, list[Path]]:
    """Return a mapping from absolute source path to the db directories that compile it.

    Each db directory is a build/cache/<arch>/<variant>/<runtime>/<target>/
    folder that contains a compile_commands.json covering the file.
    Only sim variant databases are used (avoids cross-compiler sysroot issues).
    """
    index: dict[str, list[Path]] = {}
    for db_file in sorted(_CACHE_DIR.glob("*/sim/*/*/compile_commands.json")):
        for entry in json.loads(db_file.read_text()):
            filepath = entry["file"]
            index.setdefault(filepath, []).append(db_file.parent)
    return index


def main() -> int:
    changed = [os.path.abspath(f) for f in sys.argv[1:]]
    if not changed:
        return 0

    _ensure_sim_cache()
    file_index = _build_file_index()

    if not file_index:
        print("ERROR: no sim compile databases found under build/cache/*/sim/", file=sys.stderr)
        return 1

    failed: set[str] = set()
    for f in changed:
        db_dirs = file_index.get(f)
        if not db_dirs:
            continue  # file not in any sim compile database (e.g. onboard-only)

        for db_dir in db_dirs:
            result = subprocess.run(
                ["clang-tidy", f"-p={db_dir}", "--quiet"] + _SUPPRESS_ARGS + [f],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.stdout:
                print(result.stdout, end="")
            if result.returncode != 0:
                failed.add(f)

    if failed:
        print(f"\n{len(failed)} file(s) failed clang-tidy")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
