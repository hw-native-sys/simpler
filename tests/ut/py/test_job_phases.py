# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Keep every reusable CI job on the shared preparation order.

The skeleton is copied into each workflow on purpose: the run payload is the
job's own content. The preparation order is the shared part, so that is what
these tests read. A job may omit a preparation step; it must not reorder the
ones it has, start with anything but its target checkout, or install the
environment more than once.
"""

from pathlib import Path

import pytest
import yaml

WORKFLOW_DIR = Path(__file__).parents[3] / ".github/workflows"

PREPARATION = (
    "Checkout target",
    "Validate manual mode",
    "Checkout Python setup action",
    "Set up Python",
    "Verify Python >= 3.10",
    "Set up C++ compiler",
    "Verify provisioned toolchain",
    "Cache pip packages",
    "Set up environment",
)


def _job_workflows() -> list[str]:
    # _detect-changes.yml produces job gates rather than running a job body.
    return sorted(path.name for path in WORKFLOW_DIR.glob("_*.yml") if path.name != "_detect-changes.yml")


@pytest.mark.parametrize("workflow", _job_workflows())
def test_preparation_steps_keep_their_order(workflow: str) -> None:
    # BaseLoader keeps a bare `on:` a string, and every workflow on disk is
    # checked without being listed here.
    document = yaml.load((WORKFLOW_DIR / workflow).read_text(), Loader=yaml.BaseLoader)
    jobs = list(document["jobs"].values())
    assert len(jobs) == 1, workflow
    names = [step["name"] for step in jobs[0]["steps"]]
    assert names[0] == "Checkout target", workflow
    present = [name for name in names if name in PREPARATION]
    assert present == [name for name in PREPARATION if name in present], workflow
    assert names.count("Set up environment") == 1, workflow
