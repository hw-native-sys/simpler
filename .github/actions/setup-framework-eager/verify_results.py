# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Require the exact four framework eager combinations to pass in JUnit output."""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

EXPECTED = {
    f"tests.st.a2a3.framework_eager.test_framework_eager::test_framework_queue_orders_public_worker[{queue}-{runtime}]"
    for queue in ("0", "1")
    for runtime in ("tensormap_and_ringbuffer", "host_build_graph")
}


def verify_results(path: Path) -> None:
    root = ET.parse(path).getroot()
    for suite in root.iter("testsuite"):
        if any(int(suite.get(outcome, "0")) != 0 for outcome in ("failures", "errors", "skipped")):
            raise ValueError("Framework eager suite reports a failure, error or skip")
    cases = list(root.iter("testcase"))
    observed = [f"{case.get('classname')}::{case.get('name')}" for case in cases]
    if len(observed) != len(EXPECTED) or set(observed) != EXPECTED:
        raise ValueError(f"Expected the four framework eager nodeids; observed {observed}")
    for case, nodeid in zip(cases, observed):
        if any(child.tag in {"failure", "error", "skipped"} for child in case):
            raise ValueError(f"Framework eager did not pass: {nodeid}")
    print("FRAMEWORK_CI_VERIFIED " + ", ".join(sorted(observed)), flush=True)


if __name__ == "__main__":
    verify_results(Path(sys.argv[1]))
