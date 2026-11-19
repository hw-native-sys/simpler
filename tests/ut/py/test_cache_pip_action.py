# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exercise the cache action with pip's platform-specific directory output."""

import os
import subprocess
from pathlib import Path

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    "cache_dir", ["/home/runner/.cache/pip", "/Users/runner/Library/Caches/pip", "/custom cache/pip"]
)
def test_cache_uses_the_directory_reported_by_pip(tmp_path, cache_dir):
    action = yaml.safe_load((_ROOT / ".github/actions/cache-pip/action.yml").read_text())
    steps = action["runs"]["steps"]
    cache = next(step for step in steps if step.get("uses", "").startswith("actions/cache@"))
    output = tmp_path / "github-output"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python = bin_dir / "python"
    python.write_text('#!/bin/sh\n[ "$*" = "-m pip cache dir" ] || exit 2\nprintf "%s\\n" "$PIP_REPORTED_CACHE"\n')
    python.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "GITHUB_OUTPUT": str(output),
        "PIP_REPORTED_CACHE": cache_dir,
    }
    outputs = {}
    for step in steps[: steps.index(cache)]:
        subprocess.run(["bash", "-e", "-o", "pipefail", "-c", step["run"]], env=env, check=True)
        for line in output.read_text().splitlines():
            key, value = line.split("=", 1)
            outputs[f"${{{{ steps.{step['id']}.outputs.{key} }}}}"] = value
    assert outputs.get(cache["with"]["path"], cache["with"]["path"]) == cache_dir
