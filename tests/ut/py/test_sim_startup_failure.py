# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Exercise partial startup through the installed, real simulation runtimes."""

import errno
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_FAULTS = (
    ("bind", "sim-aic-1"),
    ("bind", "sim-aicpu-0"),
    ("bind", "sim-aicpu-1"),
    ("create", "sim-aic-0"),
    ("create", "sim-aicpu-0"),
)
pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="ELF pthread interposition requires Linux")


@pytest.fixture(scope="module")
def startup_faults(tmp_path_factory):
    directory = tmp_path_factory.mktemp("sim-startup-faults")
    template = Path(__file__).with_name("support").joinpath("sim_startup_fault.cpp").read_text()
    libraries = {}
    for mode, thread in _FAULTS:
        source = directory / f"{mode}-{thread}.cpp"
        source.write_text(
            template.replace("@THREAD@", thread).replace(
                "constexpr bool fail_creation = false;",
                f"constexpr bool fail_creation = {str(mode == 'create').lower()};",
            )
        )
        library = source.with_suffix(".so")
        subprocess.run(
            ["c++", "-std=c++17", "-shared", "-fPIC", str(source), "-pthread", "-ldl", "-o", str(library)],
            check=True,
            capture_output=True,
            timeout=60,
        )
        libraries[mode, thread] = library
    return libraries


def _run_scene(arch, runtime, library):
    base = "tests/st" if runtime == "host_build_graph" else "examples"
    scene = _ROOT / base / arch / runtime / "vector_example/test_vector_example.py"
    env = dict(os.environ)
    if library is not None:
        env["LD_PRELOAD"] = (env["LD_PRELOAD"] + ":" if env.get("LD_PRELOAD") else "") + str(library)
    command = [
        sys.executable,
        "-m",
        "pytest",
        str(scene),
        "--platform",
        arch + "sim",
        "--log-level",
        "info",
        "--manual",
        "include",
        "--resource-child",
        "-p",
        "no:xdist",
        "-q",
        "-s",
        "--pto-session-timeout",
        "60",
    ]
    with subprocess.Popen(
        command,
        cwd=_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    ) as child:
        try:
            output, _ = child.communicate(timeout=90)
        except subprocess.TimeoutExpired:
            # A missing startup gate can strand native threads and child workers.
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            output, _ = child.communicate()
            pytest.fail("Simulation startup failed to terminate:\n" + output[-12000:])
        return child.returncode, output


@pytest.mark.parametrize("arch", ["a2a3", "a5"])
@pytest.mark.parametrize("runtime", ["host_build_graph", "tensormap_and_ringbuffer"])
def test_sim_startup_without_injection(arch, runtime):
    code, output = _run_scene(arch, runtime, None)
    assert code == 0, output
    assert "1 passed" in output, output


@pytest.mark.parametrize("arch", ["a2a3", "a5"])
@pytest.mark.parametrize("runtime", ["host_build_graph", "tensormap_and_ringbuffer"])
@pytest.mark.parametrize(("mode", "thread"), _FAULTS)
def test_sim_partial_startup_failure_is_drained(arch, runtime, mode, thread, startup_faults):
    code, output = _run_scene(arch, runtime, startup_faults[mode, thread])
    marker = "SIM_STARTUP_CREATE_EAGAIN" if mode == "create" else "SIM_STARTUP_BIND_ENOMEM"
    assert output.count(marker) == 1, output
    if mode == "bind":
        assert f"thread={thread} pthread_error={errno.ENOMEM}" in output, output
    assert code == 1, output
    assert "TIMEOUT:" not in output, output
    assert "launch_native_run failed with code -1" in output, output
    # This diagnostic follows stop_run_producers/join in each real runner.
    assert "All threads completed" in output, output
    assert "AICPU execution failed with rc=-1" in output, output


def test_creation_fault_ignores_unrelated_parent(startup_faults, tmp_path):
    source = Path(__file__).with_name("support") / "sim_startup_parent_probe.cpp"
    executable = tmp_path / "parent-probe"
    subprocess.run(
        ["c++", "-std=c++17", "-pthread", str(source), "-o", str(executable)],
        check=True,
        capture_output=True,
        timeout=60,
    )
    env = dict(os.environ)
    library = startup_faults["create", "sim-aic-0"]
    env["LD_PRELOAD"] = (env["LD_PRELOAD"] + ":" if env.get("LD_PRELOAD") else "") + str(library)
    result = subprocess.run([str(executable)], check=False, env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stderr.count("SIM_STARTUP_CREATE_EAGAIN") == 1
