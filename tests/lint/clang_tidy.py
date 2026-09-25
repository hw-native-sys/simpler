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
    python simpler_setup/build_runtimes.py
"""

import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from clang_tidy_paths import should_run_clang_tidy

_ROOT = Path(__file__).resolve().parents[2]
_BUILD_RUNTIMES = _ROOT / "simpler_setup" / "build_runtimes.py"
_CACHE_DIR = _ROOT / "build" / "cache"


def _macos_isysroot_args() -> list[str]:
    """On macOS, Homebrew clang-tidy is built with a hard-coded default sysroot
    (e.g. `MacOSX26.sdk`) that often does not exist on a fresh checkout. Without
    the right sysroot the libc++ → libc include cascade breaks and any C++
    header transitively pulling in `<wchar.h>` etc. fails — typical symptom is
    `'algorithm' file not found`.

    We inject two flags:
      -isysroot $SDK                    → fixes the libc / framework lookup
      -isystem $SDK/usr/include/c++/v1  → forces clang-tidy to use Apple SDK
                                          libc++ instead of Homebrew's bundled
                                          libc++ (driver default), so lint sees
                                          the same C++ stdlib headers that the
                                          Apple-Clang build actually compiles
                                          against.

    Apple Clang resolves the SDK via xcrun at runtime and needs no flag, so
    both build and lint converge on the same headers. No-op on Linux.
    """
    if platform.system() != "Darwin":
        return []
    try:
        sdk_path = subprocess.check_output(["xcrun", "--show-sdk-path"], text=True).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    if not sdk_path:
        return []
    return [
        f"--extra-arg=-isysroot{sdk_path}",
        f"--extra-arg=-isystem{sdk_path}/usr/include/c++/v1",
    ]


# Suppress compiler flags that are valid for GCC but unknown to clang.
_SUPPRESS_ARGS = [
    "--extra-arg=-Wno-unknown-warning-option",
    "--extra-arg=-Wno-unused-command-line-argument",
] + _macos_isysroot_args()

# GCC-only flags to strip from compile_commands.json before passing to clang-tidy.
_GCC_ONLY_FLAGS = {"-fno-gnu-unique"}

# A compiler whose program name carries a target-triple prefix — conda-forge and
# crosstool GCC ship `<triple>-g++` with `g++` as a symlink to it, and CMake
# records the resolved name. clang-tidy reads that prefix as its target triple,
# then matches no installed GCC and receives no C++ standard library include
# dirs: every `#include <cstdint>` fails, and statements that fail to build are
# dropped from the AST, which turns non-trivial constructors into
# modernize-use-equals-default reports. Every target in a sim database is
# host-compiled, so the prefix names nothing clang-tidy needs.
_TRIPLE_PREFIXED_DRIVER = re.compile(
    r"^(?:[A-Za-z0-9_]+-)+(?P<driver>gcc|g\+\+|cc|c\+\+|clang|clang\+\+)(?P<suffix>-[0-9.]+)?$"
)


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


def _strip_target_triple(compiler: str) -> str:
    """Return the compiler path with any target-triple prefix dropped from its name."""
    path = Path(compiler)
    match = _TRIPLE_PREFIXED_DRIVER.match(path.name)
    if match is None:
        return compiler
    return str(path.with_name(match["driver"] + (match["suffix"] or "")))


def _rewrite_argv(argv: list[str]) -> list[str]:
    """Drop GCC-only flags and any target-triple prefix from a compile command."""
    if not argv:
        return argv
    return [_strip_target_triple(argv[0])] + [arg for arg in argv[1:] if arg not in _GCC_ONLY_FLAGS]


def _rewrite_entry(entry: dict) -> bool:
    """Rewrite one compile database entry in place; return True when it changed."""
    changed = False
    if "command" in entry:
        argv = shlex.split(entry["command"])
        rewritten = _rewrite_argv(argv)
        if rewritten != argv:
            entry["command"] = shlex.join(rewritten)
            changed = True
    if "arguments" in entry:
        rewritten = _rewrite_argv(entry["arguments"])
        if rewritten != entry["arguments"]:
            entry["arguments"] = rewritten
            changed = True
    return changed


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


def _import_checkout_project() -> tuple:
    """Import `simpler_setup` from this checkout and return the recovery entry points.

    Which source tree a recovery configures is decided by that package's
    `PROJECT_ROOT`, and only this checkout's tree is the one the databases under
    `build/cache/` and the changed paths both name. A wheel install of this
    project carries a second physical copy of `src/` under
    `simpler_setup/_assets` and reports that copy as its root; the hook runs as
    a script, so `sys.path[0]` is `tests/lint` and the repo root is otherwise
    absent from the path, which is what lets the installed copy answer the
    import. A database configured against it names platform sources under paths
    no changed file matches, and compiles the checkout's own sources with both
    copies of `src/common` on the include path — where `#pragma once` is per
    file, so every shared type arrives twice and reads as a redefinition of
    itself.
    """
    if sys.path[0] != str(_ROOT):
        sys.path.insert(0, str(_ROOT))
    from simpler_setup.environment import PROJECT_ROOT  # noqa: PLC0415
    from simpler_setup.platform_info import load_build_config, to_platform  # noqa: PLC0415
    from simpler_setup.runtime_compiler import RuntimeCompiler  # noqa: PLC0415

    if Path(PROJECT_ROOT).resolve() != _ROOT:
        raise RuntimeError(
            f"simpler_setup reports its project root as {PROJECT_ROOT}, not this checkout ({_ROOT}); "
            "a compile database configured from another source tree cannot describe these files"
        )
    return load_build_config, to_platform, RuntimeCompiler


def _reconfigure_compile_database(db_file: Path) -> None:
    """Delete the broken target build dir and rerun CMake configure for it."""
    load_build_config, to_platform, RuntimeCompiler = _import_checkout_project()

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

    # The host target's CMakeLists refuses to configure without this, so a
    # recovery that omitted it produced a fatal configure, left the database
    # empty, and sent clang-tidy on to analyse the file with no include paths at
    # all. The resulting `file not found` errors then degrade the AST enough that
    # unrelated checks misfire on bodies they cannot resolve, so the omission
    # cost far more than the failed recovery. Named as the real build names it.
    cmake_defines = {"SIMPLER_RUNTIME_NAME": runtime_name} if target == "host" else None
    cmake_cmd = [
        "cmake",
        build_target.get_root_dir(),
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
    ] + build_target.gen_cmake_args(include_dirs, source_dirs, cmake_defines=cmake_defines)
    compiler._run_build_step(cmake_cmd, str(target_build_dir), target.upper(), "CMake configuration")


def _parse_compile_database(raw: str, db_file: Path) -> list[dict]:
    """Parse compile_commands.json content and reject empty or malformed payloads."""
    if not raw.strip():
        raise ValueError(f"empty compile database: {db_file}")
    entries = json.loads(raw)
    if not isinstance(entries, list):
        raise ValueError(f"compile database is not a JSON array: {db_file}")
    for entry in entries:
        if not isinstance(entry, dict) or "file" not in entry:
            raise ValueError(f"compile database entry is not an object naming a file: {db_file}")
    return entries


def _load_compile_database(db_file: Path) -> list[dict]:
    """Load a compile database, rebuilding its target cache dir when it is broken."""
    if not db_file.is_file():
        print(f"WARNING: compile database disappeared, skipping: {db_file}", file=sys.stderr)
        return []

    try:
        return _parse_compile_database(db_file.read_text(), db_file)
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"WARNING: invalid compile database detected: {exc}", file=sys.stderr)
        _reconfigure_compile_database(db_file)

    if not db_file.is_file():
        print(f"WARNING: compile database recovery produced no file, skipping: {db_file}", file=sys.stderr)
        return []

    try:
        return _parse_compile_database(db_file.read_text(), db_file)
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"WARNING: recovered compile database is still invalid, skipping: {exc}", file=sys.stderr)
        return []


def _publish_compile_database(db_file: Path, entries: list[dict]) -> None:
    """Replace a database's content by rename, so no reader ever sees it empty.

    Every invocation of this hook reads all of these files, and pre-commit runs
    a hook it is not told to serialize as several processes over disjoint chunks
    of the changed files. A write in place is therefore observable by a peer as
    a zero-byte database, which reads as a broken cache and sends that peer
    through a full CMake reconfigure of the target instead of a replay.
    """
    tmp = db_file.with_name(f"{db_file.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(entries, indent=2))
    os.replace(tmp, db_file)


def _build_file_index() -> dict[str, list[Path]]:
    """Return a mapping from absolute source path to the db directories that compile it.

    Each db directory is a build/cache/<arch>/<variant>/<runtime>/<target>/
    folder that contains a compile_commands.json covering the file.
    Only sim variant databases are used (avoids cross-compiler sysroot issues).

    When a compile command carries GCC-only flags or a target-triple-prefixed
    compiler name, the database's content is replaced so that clang-tidy can
    replay it.
    """
    index: dict[str, list[Path]] = {}
    for db_file in sorted(_CACHE_DIR.glob("*/sim/*/*/compile_commands.json")):
        entries = _load_compile_database(db_file)
        changed = False
        for entry in entries:
            changed |= _rewrite_entry(entry)
        if changed:
            _publish_compile_database(db_file, entries)
        for entry in entries:
            filepath = entry["file"]
            index.setdefault(filepath, []).append(db_file.parent)
    return index


def main() -> int:
    changed = [os.path.abspath(path) for path in sys.argv[1:] if should_run_clang_tidy(path)]
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
