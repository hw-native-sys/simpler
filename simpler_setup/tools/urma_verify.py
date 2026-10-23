# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run the mandatory A5 URMA cases; absent, skipped or unexecuted cases fail."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path

import pytest

from simpler_setup.pto_isa import read_pto_isa_pin
from simpler_setup.runtime_builder import RuntimeBuilder

CASES = (
    "TestUrmaDeferredCompletionDemo",
    "TestUrmaRepeatedPosts",
    "TestUrmaDomainLifecycle",
    "test_urma_native_teardown",
)
TEST_PATH = "examples/a5/tensormap_and_ringbuffer/urma_deferred_completion_demo"


class ExecutionGate:
    def __init__(self, cases=CASES):
        self.cases = cases
        self.selected = set()
        self.executed = set()
        self.problems = []

    def pytest_collection_finish(self, session):
        self.selected = {item.nodeid for item in session.items}
        if len(self.selected) != len(self.cases) or any(
            sum(case in node for node in self.selected) != 1 for case in self.cases
        ):
            self.problems.append(f"expected exactly {self.cases}; collected {sorted(self.selected)}")

    def pytest_runtest_logreport(self, report):
        if report.skipped or report.failed or getattr(report, "wasxfail", None):
            self.problems.append(f"{report.nodeid}: {report.when} {report.outcome}")
        if report.when == "call" and report.passed:
            self.executed.add(report.nodeid)

    def pytest_sessionfinish(self, session):
        if not self.selected or self.executed != self.selected:
            self.problems.append("not every required case executed successfully")
        if self.problems:
            session.exitstatus = pytest.ExitCode.TESTS_FAILED


def verify_artifacts():
    if os.environ.get("SIMPLER_ENABLE_PTO_URMA_WORKSPACE", "").upper() not in {"1", "ON", "YES", "TRUE"}:
        raise RuntimeError("URMA verification requires explicit SIMPLER_ENABLE_PTO_URMA_WORKSPACE=ON")
    if os.environ.get("SIMPLER_ENABLE_PTO_RDMA_WORKSPACE", "").upper() != "OFF":
        raise RuntimeError("URMA verification requires explicit SIMPLER_ENABLE_PTO_RDMA_WORKSPACE=OFF")
    binaries = RuntimeBuilder("a5").get_binaries("tensormap_and_ringbuffer")
    host = ctypes.CDLL(str(binaries.host_path))
    host.comm_async_workspace_backend.restype = ctypes.c_char_p
    host.comm_pto_isa_build_commit.restype = ctypes.c_char_p
    backend = host.comm_async_workspace_backend().decode()
    pin = host.comm_pto_isa_build_commit().decode()
    if backend != "URMA" or pin != read_pto_isa_pin():
        raise RuntimeError(f"loaded runtime backend/pin mismatch: {backend} / {pin}")
    return {
        "backend": backend,
        "pto_pin": pin,
        "binaries": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (binaries.host_path, binaries.aicpu_path, binaries.aicore_path)
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument("--report", type=Path, default=Path("outputs/urma-verification.json"))
    args = parser.parse_args(argv)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    gate = ExecutionGate()
    evidence = {}
    status = 1
    try:
        evidence = verify_artifacts()
        status = int(
            pytest.main(
                [
                    TEST_PATH,
                    "--platform",
                    "a5",
                    "--runtime",
                    "tensormap_and_ringbuffer",
                    "--device",
                    args.device,
                    "--require-pto-isa",
                    "--pto-session-timeout",
                    "1200",
                    "-v",
                    "-s",
                    f"--junitxml={args.report.with_suffix('.xml')}",
                ],
                plugins=[gate],
            )
        )
    except Exception as error:
        gate.problems.append(str(error))
    finally:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        evidence.update(
            selected=sorted(gate.selected), executed=sorted(gate.executed), problems=gate.problems, exit_status=status
        )
        args.report.write_text(json.dumps(evidence, indent=2) + "\n")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
