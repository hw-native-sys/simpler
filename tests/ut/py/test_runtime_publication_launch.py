# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exercise onboard execution state against the built host runtime, with RTS stubs.

The hardware marker selects CI jobs that build the onboard libraries and have
CANN headers. The probe does not initialize or execute on a device.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_CASES = [
    pytest.param(arch, runtime, id=f"{arch}-{runtime}", marks=pytest.mark.platforms([arch]))
    for arch in ("a2a3", "a5")
    for runtime in ("tensormap_and_ringbuffer", "host_build_graph")
]


@pytest.mark.parametrize(
    "probe",
    [
        "runtime_publication_launch",
        "run_execution_streams",
        "run_caller_launch",
        "run_observer",
        "bootstrap_error",
        "runtime_timeout_configuration",
    ],
)
@pytest.mark.requires_hardware
@pytest.mark.parametrize(("arch", "runtime"), _CASES)
def test_onboard_execution_contract(arch: str, runtime: str, probe: str, tmp_path: Path):
    cache = _ROOT / "build/cache" / arch / "onboard" / runtime / "host"
    library = _ROOT / "build/lib" / arch / "onboard" / runtime / "libhost_runtime.so"
    database = cache / "compile_commands.json"
    assert library.is_file(), f"Build the {arch} onboard runtimes before this test: {library}"
    assert database.is_file(), f"RuntimeBuilder's host compile database is required: {database}"
    entries = json.loads(database.read_text())
    entry = next(
        item for item in entries if Path(item["file"]) == _ROOT / f"src/{arch}/platform/onboard/host/device_runner.cpp"
    )
    args = entry.get("arguments") or shlex.split(entry["command"])
    # The same compiler, generated profiling headers and definitions must see
    # the same C++ object layouts as the library under test.
    command = []
    arguments = iter(args)
    for arg in arguments:
        if arg == "-o":
            next(arguments)
        elif arg != "-c" and arg != entry["file"]:
            command.append(arg)
    cann = Path(os.environ["ASCEND_HOME_PATH"])
    binary = tmp_path / probe
    if probe in ("run_caller_launch", "run_observer") and arch == "a5":
        # Compile the real backend so this executable's strong scheduler-export
        # observer replaces its private weak leaf. The drain decision stays real.
        command.append(entry["file"])
    if probe in ("run_caller_launch", "run_observer"):
        command += [
            str(_ROOT / "src/common/log/host_log.cpp"),
            str(_ROOT / "src/common/log/unified_log_host.cpp"),
        ]
    command += [
        "-UNDEBUG",
        str(_ROOT / f"tests/ut/py/support/{'run_caller_launch' if probe == 'run_observer' else probe}_probe.cpp"),
        # The probe reaches its test fixture by path from the unit-test root, the
        # same way the C++ targets do, so a fixture that moves stays reachable
        # from both without this line changing.
        f"-I{_ROOT / 'tests/ut/cpp'}",
        str(library),
        f"-Wl,-rpath,{library.parent}",
        f"-L{cann / 'lib64'}",
        f"-L{cann / 'runtime/lib64'}",
        f"-Wl,-rpath,{cann / 'lib64'}",
        f"-Wl,-rpath,{cann / 'runtime/lib64'}",
        "-lruntime",
        "-lascendcl",
        "-pthread",
        "-ldl",
        "-o",
        str(binary),
    ]
    build = subprocess.run(command, cwd=entry["directory"], capture_output=True, text=True, timeout=120, check=False)
    assert build.returncode == 0, build.stdout + build.stderr
    if probe == "run_observer":
        modes = (0, 1, 2, 3) if (arch, runtime) == ("a5", "host_build_graph") else (0,)
        for mode in modes:
            for scenario in (
                "success-record-error",
                "failed-record-success",
                "missing-record-success",
                "stale-record-success",
                "malformed-record-success",
                "copy-error-success",
                "error-record-error",
                "retired-error-record",
                "copy-error-error",
                "core-error",
                "program-output",
                "output-copy-error",
                "kernel-no-copy",
                "unknown-stop",
                "late-stream-error",
            ) + (("clock-export",) if runtime == "host_build_graph" else ()):
                observed = subprocess.run(
                    [str(binary), "--observer", scenario, str(mode)],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                assert observed.returncode == 0, observed.stdout + observed.stderr
                assert "PASS:" in observed.stdout
        return
    run_args = [str(binary)]
    result = subprocess.run(run_args, capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS:" in result.stdout
    if probe == "run_caller_launch":
        assert "external resources: public retain/finalize/reuse/release and retries passed" in result.stderr
        assert "capture release: timeout/error retain, completion retry and duplicate rejection passed" in result.stderr
        _check_close_failure_cases(binary)
    if probe == "runtime_timeout_configuration":
        output = result.stdout + result.stderr
        assert output.count("Configured timeout ordering warning:") == 1
        assert (
            "scheduler=2000 ms, requested_op_execute=3000000 us, actual_op_execute=4294967 us, stream_sync=4000 ms"
        ) in output
        assert "op-execute timeout must be below stream-sync timeout" in output
        assert "Increase SIMPLER_STREAM_SYNC_TIMEOUT_MS" in output
        assert "Continuing initialization; budgets unchanged" in output
    if probe == "bootstrap_error":
        assert "SO cache failed: stage=args errno=22" in result.stdout + result.stderr
        for stage in ("verify", "link", "unlink"):
            assert f"SO cache failed: stage={stage} errno=5" in result.stdout + result.stderr
        assert "SO cache failed: stage=write errno=28" in result.stdout + result.stderr
        assert "device filesystem block_bytes=4096 free_blocks=0" in result.stdout + result.stderr
        assert "device filesystem query failed: errno=38" in result.stdout + result.stderr
        assert "aclrtMalloc(result) failed: -1515" in result.stdout + result.stderr
        assert "aclrtMemcpy(result initialization) failed: -1616" in result.stdout + result.stderr


def _check_close_failure_cases(binary: Path):
    # Each close-failure case has a fresh process. Unconfirmed resets retain
    # the context and backing.
    for scenario in (
        "reset-failure",
        "probe-failure",
        "acl-finalize-retry",
        "borrowed",
        "held-program",
        "held-borrowed",
        "external-program",
        "external-borrowed",
        "unused",
        "unused-release-retry",
        "unused-sticky-close",
    ):
        failed_reset = subprocess.run([str(binary), scenario], capture_output=True, text=True, timeout=30, check=False)
        assert failed_reset.returncode == 0, failed_reset.stdout + failed_reset.stderr
        if scenario.startswith("unused"):
            assert "standalone collector unused: preparation retired without submission" in failed_reset.stderr
            continue
        if scenario.startswith(("held-", "external-")):
            assert f"standalone collector {scenario}: external close deferred; readers_" in failed_reset.stderr
            continue
        expected = (
            "standalone collector acl-finalize-retry: confirmed reset reused; original error="
            if scenario == "acl-finalize-retry"
            else "standalone collector borrowed: context retained, no reset"
            if scenario == "borrowed"
            else f"standalone collector {scenario}: unconfirmed context retained"
        )
        assert expected in failed_reset.stderr
        if scenario in ("reset-failure", "probe-failure"):
            assert (
                f"standalone collector {scenario}: same-runner retry confirmed; original error=-739"
                in failed_reset.stderr
            )
