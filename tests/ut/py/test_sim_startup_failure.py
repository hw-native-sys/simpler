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
    assert "launch_native_run failed with code -1 (" in output, output
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


@pytest.mark.parametrize("arch", ["a2a3", "a5"])
@pytest.mark.parametrize("runtime", ["host_build_graph", "tensormap_and_ringbuffer"])
@pytest.mark.parametrize("binding", ["core_tls", "subblock", "cluster", "affinity"])
def test_sim_core_identity_failure_is_drained(arch, runtime, binding, tmp_path):
    # Wrapper key count is checked indirectly by the exact identity-source assertion below.
    key_count = 6 if arch == "a2a3" else 10
    skip = {"core_tls": 1, "subblock": key_count + 1, "cluster": key_count + 2, "affinity": 1}[binding]
    thread = "sim-aicpu-1" if binding == "affinity" else "sim-aic-1"
    template = Path(__file__).with_name("support").joinpath("sim_startup_fault.cpp").read_text()
    source = tmp_path / "identity.cpp"
    source.write_text(
        template.replace("@THREAD@", thread).replace(
            "constexpr int skip_bindings = 0;", f"constexpr int skip_bindings = {skip};"
        )
    )
    library = source.with_suffix(".so")
    subprocess.run(
        ["c++", "-std=c++17", "-shared", "-fPIC", str(source), "-pthread", "-ldl", "-o", str(library)],
        check=True,
        capture_output=True,
        timeout=60,
    )
    code, output = _run_scene(arch, runtime, library)
    assert output.count("SIM_STARTUP_BIND_ENOMEM") == 1, output
    diagnostic = "AICPU affinity TLS" if binding == "affinity" else "AICore thread TLS"
    assert f"{diagnostic} binding failed: {errno.ENOMEM}" in output, output
    if binding in ("subblock", "cluster"):
        assert f"Sim {binding} identity binding failed: {errno.ENOMEM}" in output, output
    assert code == 1, output
    assert "launch_native_run failed with code -1000 (" in output, output
    assert "Simulated startup cancelled; all created core threads joined" in output, output
    assert "TIMEOUT:" not in output, output


@pytest.mark.parametrize("arch", ["a2a3", "a5"])
def test_sim_key_creation_failure_allows_same_worker_retry(arch, tmp_path):
    source = Path(__file__).with_name("support") / "sim_key_create_fault.cpp"
    library = tmp_path / "key-fault.so"
    subprocess.run(
        ["c++", "-std=c++17", "-shared", "-fPIC", str(source), "-ldl", "-o", str(library)],
        check=True,
        capture_output=True,
        timeout=60,
    )
    script = r"""
import importlib
import sys
from simpler import Worker
arch = sys.argv[1]
module = importlib.import_module(f"tests.st.{arch}.host_build_graph.graph_execution.test_graph_execution")
cls = getattr(module, "TestGraphExecutionHostBuildGraph" + ("A5" if arch == "a5" else ""))
scene = cls()
compiled = scene.build_callable(arch + "sim")
worker = Worker(level=2, device_id=0, platform=arch + "sim", runtime="host_build_graph")
try:
    worker.init()
    try:
        scene._run_and_validate_l2(worker, compiled, scene.CASES[1])
    except RuntimeError as error:
        print("EXPECTED_LOAD_FAILURE", str(error), flush=True)
    else:
        raise AssertionError("injected key creation failure did not reject the run")
    scene._run_and_validate_l2(worker, compiled, scene.CASES[1])
    print("SAME_WORKER_RETRY_PASSED", flush=True)
finally:
    cls._st_l2_handle = None
    worker.close()
"""
    env = dict(os.environ)
    env["LD_PRELOAD"] = str(library) + (":" + env["LD_PRELOAD"] if env.get("LD_PRELOAD") else "")
    result = subprocess.run(
        [sys.executable, "-c", script, arch],
        check=False,
        cwd=_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert output.count("SIM_CORE_KEY_CREATE_EAGAIN") == 1, output
    assert f"AICore SO failed to create TLS keys: {errno.EAGAIN}" in output, output
    assert "EXPECTED_LOAD_FAILURE" in output, output
    assert "SAME_WORKER_RETRY_PASSED" in output, output
