#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Build a benchmark-only TMR runtime in an isolated source snapshot.

The ordinary source tree and build/lib are never patched or overwritten.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
PATCH = Path(__file__).with_name("tmr_timing.patch")
RUNTIME = "tensormap_and_ringbuffer"
MARKER = "dsv41_timing.json"
METRIC = "first_kernel_start_to_last_kernel_end"
TIMING_SOURCES = (
    "src/a5/runtime/tensormap_and_ringbuffer/aicore/aicore_executor.cpp",
    "src/a5/runtime/tensormap_and_ringbuffer/aicpu/aicpu_executor.cpp",
    "src/a5/runtime/tensormap_and_ringbuffer/runtime/runtime.h",
)
COMPONENTS = ("aicore_kernel.o", "libaicpu_kernel.so")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stage_sources(source, snapshot):
    """Apply the small timing delta only to a fresh, independent source copy."""
    source, snapshot = Path(source).resolve(), Path(snapshot).resolve()
    if snapshot.exists():
        raise ValueError("Use a new source snapshot directory")
    if any(snapshot.is_relative_to(source / name) for name in ("src", "cmake", "simpler_setup", "python")):
        raise ValueError("The source snapshot must be outside production source directories")
    before = {name: digest(source / name) for name in TIMING_SOURCES}
    patch_paths = [line[6:] for line in PATCH.read_text().splitlines() if line.startswith("+++ b/")]
    if sorted(patch_paths) != sorted(TIMING_SOURCES):
        raise ValueError("Timing overlay must contain exactly the three benchmark instrumentation files")
    snapshot.mkdir(parents=True)
    for name in ("src", "cmake", "simpler_setup", "python"):
        shutil.copytree(source / name, snapshot / name, ignore=shutil.ignore_patterns("__pycache__", "_assets"))
    for name in ("pto_isa.pin", "pyproject.toml", "CMakeLists.txt"):
        if (source / name).is_file():
            shutil.copy2(source / name, snapshot / name)
    command = ["git", "apply", "--unsafe-paths", f"--directory={snapshot}", "--whitespace=error-all"]
    subprocess.run([*command, "--check", str(PATCH)], cwd=source, check=True, capture_output=True, text=True)
    subprocess.run([*command, str(PATCH)], cwd=source, check=True, capture_output=True, text=True)
    if {name: digest(source / name) for name in TIMING_SOURCES} != before:
        raise ValueError("Production sources changed during snapshot creation")
    return {
        "patch_sha256": digest(PATCH),
        "original_sources": before,
        "patched_sources": {name: digest(snapshot / name) for name in TIMING_SOURCES},
    }


def validate_runtime(runtime):
    """Reject a stock or mixed TMR build before requesting NPU devices."""
    variant = Path(runtime).resolve() / RUNTIME
    try:
        manifest = json.loads((variant / MARKER).read_text())
        if manifest["schema"] != 1 or manifest["test_only"] is not True or manifest["metric"] != METRIC:
            raise ValueError("Invalid test-only timing marker")
        if set(manifest["components"]) != set(COMPONENTS):
            raise ValueError("Incomplete test-only timing components")
        for name in COMPONENTS:
            if digest(variant / name) != manifest["components"][name]:
                raise ValueError(f"Test-only TMR component mismatch: {name}")
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"L0 AICore wall requires a matching test-only TMR build: {error}. "
            "Run production/tmr_runtime.py --output-dir NEW_DIR --pto-isa PINNED_CHECKOUT first."
        ) from error
    return manifest


def build(args):
    source, output, pto = args.source_root.resolve(), args.output_dir.resolve(), args.pto_isa.resolve()
    if output.exists():
        raise ValueError("Build the test-only runtime into a new output directory")
    protected = ("src", "cmake", "simpler_setup", "python", "build/lib", "build/cache")
    if any(output.is_relative_to(source / name) for name in protected):
        raise ValueError("Use a dedicated benchmark directory outside production sources and build outputs")
    required = (source / "pto_isa.pin").read_text().strip()
    # A local clone fixes the header revision without checking out or changing
    # the caller's shared PTO-ISA worktree.
    subprocess.run(["git", "-C", str(pto), "cat-file", "-e", required + "^{commit}"], check=True)
    output.mkdir(parents=True)
    snapshot = output / "source"
    manifest = stage_sources(source, snapshot)
    pinned = snapshot / "build/pto-isa"
    pinned.parent.mkdir(parents=True)
    subprocess.run(["git", "clone", "--shared", "--no-checkout", str(pto), str(pinned)], check=True)
    subprocess.run(["git", "-C", str(pinned), "checkout", "--detach", required], check=True)
    scratch = output / "tmp"
    scratch.mkdir()
    driver = """
import runpy, sys
from pathlib import Path
source, output = map(Path, sys.argv[1:])
sys.meta_path[:] = [f for f in sys.meta_path if type(f).__module__ != '_editable_skbc_simpler']
sys.path[:0] = [str(source / 'python'), str(source)]
from simpler_setup.environment import PROJECT_ROOT
assert PROJECT_ROOT.resolve() == source.resolve(), PROJECT_ROOT
sys.argv = [str(source / 'simpler_setup/build_runtimes.py'), '--platforms', 'a5',
            '--runtime', 'tensormap_and_ringbuffer', '--lib-dir', str(output / 'runtime'),
            '--cache-dir', str(output / 'cache')]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    env = dict(os.environ, TMPDIR=str(scratch), TMP=str(scratch), TEMP=str(scratch))
    with (output / "build.log").open("w") as log:
        subprocess.run(
            [sys.executable, "-c", driver, str(snapshot), str(output)],
            cwd=snapshot,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    runtime = output / "runtime/a5/onboard"
    manifest.update(
        schema=1,
        test_only=True,
        metric=METRIC,
        pto_isa_commit=required,
        components={name: digest(runtime / RUNTIME / name) for name in COMPONENTS},
    )
    (runtime / RUNTIME / MARKER).write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    validate_runtime(runtime)
    print(f"Test-only TMR runtime: {runtime}")


def main():
    if sys.argv[1:2] == ["validate"]:
        parser = argparse.ArgumentParser(description="Validate the benchmark-only TMR timing runtime")
        parser.add_argument("runtime", type=Path)
        args = parser.parse_args(sys.argv[2:])
        try:
            validate_runtime(args.runtime)
        except ValueError as error:
            parser.error(str(error))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pto-isa", type=Path, required=True, help="Local checkout containing the pinned commit")
    build(parser.parse_args())


if __name__ == "__main__":
    main()
