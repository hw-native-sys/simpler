# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared path and identity helpers for the real vLLM eager integration.

The CI identity recorder (`.github/actions/setup-vllm-eager/identity.py`) and
the scene-test fixture in `tests/st/a2a3/vllm_eager/` must resolve the same
adapter checkout and NNAL/ATB install; both import this module so the executed
adapter and the recorded receiver cannot diverge.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

ADAPTER_CASE = Path("examples/a2a3/host_build_graph/qwen3_14b_serving_effective")
RUNNER = "run_vllm_eager.py"


def _git(cwd: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(cwd), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _checkout() -> Path:
    return Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())


def _file_identity(path: Path) -> dict[str, str]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest()}


def adapter_runner(root: Path | str | None = None) -> Path:
    """Resolve the fixed-shape eager adapter runner.

    ``VLLM_ADAPTER_ROOT`` names a checkout (or the case directory itself) and
    wins over ``root``; without either, the current repository is used.
    """
    value = os.environ.get("VLLM_ADAPTER_ROOT")
    base = Path(value) if value else (Path(root) if root is not None else _checkout())
    candidates = (base / RUNNER, base / ADAPTER_CASE / RUNNER)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ValueError(
        f"vLLM adapter runner not found under {base} (tried {', '.join(str(candidate) for candidate in candidates)}); "
        f"set VLLM_ADAPTER_ROOT to the pinned adapter checkout or its {ADAPTER_CASE.name} directory"
    )


def adapter_identity(runner: Path | str) -> dict[str, object]:
    """Record the adapter runner bytes and, for a checkout, its revision.

    ``modified`` is true when the runner itself differs from its committed
    revision, which keeps an unvetted adapter visible in the CI receipt.
    """
    runner = Path(runner).resolve()
    record: dict[str, object] = {"runner": _file_identity(runner)}
    head = _git(runner.parent, "rev-parse", "HEAD")
    if head is None:
        return record
    record["head"] = head
    record["root"] = _git(runner.parent, "rev-parse", "--show-toplevel")
    status = _git(runner.parent, "status", "--porcelain", "--", str(runner))
    record["modified"] = bool(status)
    return record


def atb_home(candidates: list[Path] | None = None) -> Path:
    """Locate the NNAL/ATB installation whose version.info the tests check.

    ``candidates`` overrides discovery for tests; when omitted, the caller's
    ``ATB_ENV``/``ATB_HOME_PATH``/``ATB_HOME`` and the default install location
    are used in that order.
    """
    if candidates is None:
        candidates = []
        atb_env = os.environ.get("ATB_ENV")
        if atb_env:
            candidates.append(Path(atb_env).resolve().parent)
        for name in ("ATB_HOME_PATH", "ATB_HOME"):
            value = os.environ.get(name)
            if value:
                candidates.append(Path(value))
        candidates.append(Path("/usr/local/Ascend/nnal/atb/latest"))
    for base in candidates:
        for candidate in (base, base / "latest", *base.parents):
            if (candidate / "version.info").is_file():
                return candidate
    return candidates[0]
