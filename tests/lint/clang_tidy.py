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
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_BUILD_RUNTIMES = _ROOT / "examples" / "scripts" / "build_runtimes.py"
_CACHE_DIR = _ROOT / "build" / "cache"

for _import_dir in (_ROOT / "python", _ROOT / "examples" / "scripts"):
    _import_dir_str = str(_import_dir)
    if _import_dir_str not in sys.path:
        sys.path.insert(0, _import_dir_str)

from platform_info import load_build_config, to_platform  # noqa: E402
from runtime_compiler import RuntimeCompiler  # noqa: E402

# Suppress compiler flags that are valid for GCC but unknown to clang.
_SUPPRESS_ARGS = [
    "--extra-arg=-Wno-unknown-warning-option",
    "--extra-arg=-Wno-unused-command-line-argument",
]

# GCC-only flags to strip from compile_commands.json before passing to clang-tidy.
_GCC_ONLY_FLAGS = {"-fno-gnu-unique"}


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


def _strip_gcc_flags(command: str) -> str:
    """Remove GCC-only flags that clang/clang-tidy does not understand."""
    parts = shlex.split(command)
    filtered_parts = [p for p in parts if p not in _GCC_ONLY_FLAGS]
    return shlex.join(filtered_parts)


def _resolve_target_dirs(config_dir: Path, build_config: dict, target: str) -> tuple[list[str], list[str]]:
    """Resolve include and source dirs for a target from build_config."""
    cfg = build_config[target]
    include_dirs = [str((config_dir / p).resolve()) for p in cfg["include_dirs"]]
    source_dirs = [str((config_dir / p).resolve()) for p in cfg["source_dirs"]]
    return include_dirs, source_dirs


def _parse_db_path(db_file: Path) -> tuple[str, str, str, str]:
    """Return (arch, variant, runtime, target) for a compile database path."""
    try:
        arch, variant, runtime_name, target, filename = db_file.relative_to(_CACHE_DIR).parts
    except ValueError as exc:
        raise RuntimeError(f"compile database is outside build/cache: {db_file}") from exc

    if filename != "compile_commands.json":
        raise RuntimeError(f"unexpected compile database file name: {db_file}")

    return arch, variant, runtime_name, target


def _reconfigure_compile_database(db_file: Path) -> None:
    """Delete the broken target build dir and rerun CMake configure for it."""
    arch, variant, runtime_name, target = _parse_db_path(db_file)
    platform = to_platform(arch, variant)
    config_path = _ROOT / "src" / arch / "runtime" / runtime_name / "build_config.py"
    if not config_path.is_file():
        raise RuntimeError(f"build config not found for compile database recovery: {config_path}")

    build_config = load_build_config(config_path)
    if target not in build_config:
        raise RuntimeError(f"target '{target}' not found in build config: {config_path}")

    include_dirs, source_dirs = _resolve_target_dirs(config_path.parent, build_config, target)
    compiler = RuntimeCompiler.get_instance(platform=platform)
    build_target = getattr(compiler, f"{target}_target", None)
    if build_target is None:
        raise RuntimeError(f"runtime compiler has no target configuration for '{target}'")

    target_build_dir = db_file.parent
    print(f"WARNING: reconfiguring broken compile database: {db_file}", file=sys.stderr)
    if target_build_dir.exists():
        shutil.rmtree(target_build_dir)
    target_build_dir.mkdir(parents=True, exist_ok=True)

    cmake_cmd = [
        "cmake",
        build_target.get_root_dir(),
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
    ] + build_target.gen_cmake_args(include_dirs, source_dirs)
    compiler._run_build_step(cmake_cmd, str(target_build_dir), target.upper(), "CMake configuration")


def _parse_compile_database(raw: str, db_file: Path) -> list[dict]:
    """Parse compile_commands.json content and reject empty or malformed payloads."""
    if not raw.strip():
        raise ValueError(f"empty compile database: {db_file}")
    entries = json.loads(raw)
    if not isinstance(entries, list):
        raise ValueError(f"compile database is not a JSON array: {db_file}")
    return entries


def _load_compile_database(db_file: Path) -> tuple[str, list[dict]]:
    """Load a compile database, rebuilding its target cache dir when it is broken."""
    raw = db_file.read_text()
    try:
        return raw, _parse_compile_database(raw, db_file)
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"WARNING: invalid compile database detected: {exc}", file=sys.stderr)
        _reconfigure_compile_database(db_file)

    rebuilt_raw = db_file.read_text()
    return rebuilt_raw, _parse_compile_database(rebuilt_raw, db_file)


def _build_file_index() -> dict[str, list[Path]]:
    """Return a mapping from absolute source path to the db directories that compile it.

    Each db directory is a build/cache/<arch>/<variant>/<runtime>/<target>/
    folder that contains a compile_commands.json covering the file.
    Only sim variant databases are used (avoids cross-compiler sysroot issues).

    When the compile database contains GCC-only flags, it is modified
    in-place to remove them so that clang-tidy can parse the commands.
    """
    index: dict[str, list[Path]] = {}
    for db_file in sorted(_CACHE_DIR.glob("*/sim/*/*/compile_commands.json")):
        raw, entries = _load_compile_database(db_file)
        needs_filter = any(flag in raw for flag in _GCC_ONLY_FLAGS)
        if needs_filter:
            for entry in entries:
                if "command" in entry:
                    entry["command"] = _strip_gcc_flags(entry["command"])
            db_file.write_text(json.dumps(entries, indent=2))
        for entry in entries:
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
