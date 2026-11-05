# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: PLC0415
"""Frozen CSA input/output snapshot and source-checkout identity."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

CASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CASE_DIR.parents[3]
DATA_DIR = CASE_DIR / "data"
RUNTIME = "host_build_graph"


def manifest():
    return json.loads((CASE_DIR / "manifest.json").read_text())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_snapshot(data_dir=DATA_DIR):
    for relative, expected in manifest()["files"].items():
        path = Path(data_dir) / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing CSA snapshot: {path}; run fixture.py <original-build>/data first")
        if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            raise ValueError(f"CSA snapshot does not match the frozen baseline: {path}")


def import_snapshot(source, destination=DATA_DIR):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    verify_snapshot(source)
    for relative, expected in manifest()["files"].items():
        target = destination / relative
        if target.exists() and sha256(target) != expected["sha256"]:
            raise FileExistsError(f"Refusing to overwrite a different snapshot: {target}")
    for relative in manifest()["files"]:
        target = destination / relative
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / relative, target)
    verify_snapshot(destination)


def load_tensor(name, section, data_dir=DATA_DIR):
    import torch

    tensor = torch.load(Path(data_dir) / section / f"{name}.pt", map_location="cpu", weights_only=True)
    spec = next(arg for arg in manifest()["arguments"] if arg["name"] == name)
    if list(tensor.shape) != spec["shape"] or str(tensor.dtype) != spec["dtype"] or not tensor.is_contiguous():
        raise ValueError(f"Unexpected shape/dtype/layout for {section}/{name}")
    return tensor


def checkout_identity():
    import _task_interface
    import torch

    from simpler_setup.environment import PROJECT_ROOT

    if Path(PROJECT_ROOT).resolve() != REPO_ROOT:
        raise RuntimeError(f"CSA must use this simpler checkout ({REPO_ROOT}), not {PROJECT_ROOT}; reinstall editable")
    if torch.__version__.split("+")[0].split(".")[:2] != ["2", "12"]:
        raise RuntimeError(f"The frozen CSA baseline requires Torch 2.12; got {torch.__version__}")

    def git(*args):
        return subprocess.check_output(["git", "-C", str(REPO_ROOT), *args], text=True).strip()

    commit = git("rev-parse", "HEAD")
    if _task_interface.__build_commit__ != commit:
        raise RuntimeError("simpler bindings are stale; run pip install --no-build-isolation -e .")
    runtime = REPO_ROOT / "build/lib/a5/onboard" / RUNTIME
    binaries = [runtime / name for name in ("libhost_runtime.so", "libaicpu_kernel.so", "aicore_kernel.o")]
    return {
        "runtime": RUNTIME,
        "fixture_id": manifest()["fixture_id"],
        "fixture_manifest_sha256": sha256(CASE_DIR / "manifest.json"),
        "torch": torch.__version__,
        "simpler_commit": commit,
        "simpler_branch": git("branch", "--show-current"),
        "tracked_diff_sha256": hashlib.sha256(
            git("diff", "HEAD", "--", "src", "python", "simpler_setup").encode()
        ).hexdigest(),
        "project_root": str(PROJECT_ROOT),
        "runtime_sha256": {path.name: sha256(path) for path in binaries},
        "case_sources_match_baseline": all(
            sha256(CASE_DIR / path) == digest for path, digest in manifest()["sources"].items()
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Import or verify the frozen CSA snapshot without regenerating golden."
    )
    parser.add_argument("source", nargs="?", type=Path, help="Data directory from the original validated build")
    args = parser.parse_args()
    if args.source is not None:
        import_snapshot(args.source)
    verify_snapshot()
    print(json.dumps(checkout_identity(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
