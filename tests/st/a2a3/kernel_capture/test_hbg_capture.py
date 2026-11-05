# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Actual HBG native prepare/capture/replay with no warmup user execution."""

from __future__ import annotations

import ctypes
import os
import subprocess
from pathlib import Path

import pytest
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import ChipCallable, CoreCallable

from simpler_setup.elf_parser import extract_text_section
from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.pto_isa import ensure_pto_isa_root

_ROOT = Path(__file__).resolve().parents[4]
_NATIVE = Path(__file__).with_name("native")
pytestmark = [pytest.mark.requires_hardware, pytest.mark.platforms(["a2a3"]), pytest.mark.runtime("host_build_graph")]


def build_probe(build: Path, *, retained_models: bool = False):
    build.mkdir(parents=True, exist_ok=True)
    compiler = KernelCompiler("a2a3")
    kernel = compiler.compile_incore(
        str(_NATIVE / ("hbg_inout_step.cpp" if retained_models else "hbg_step.cpp")),
        core_type="aiv",
        pto_isa_root=ensure_pto_isa_root(),
        extra_include_dirs=compiler.get_orchestration_include_dirs("host_build_graph"),
        build_dir=str(build),
    )
    orch = compiler.compile_orchestration(
        "host_build_graph", str(_NATIVE / "hbg_orchestration.cpp"), build_dir=str(build)
    )
    child = CoreCallable.build(
        signature=[D.INOUT, D.SCALAR] if retained_models else [D.IN, D.OUT, D.SCALAR],
        binary=extract_text_section(kernel),
    )
    chip = ChipCallable.build(
        signature=[D.INOUT, D.SCALAR] if retained_models else [D.IN, D.OUT, D.IN, D.SCALAR],
        func_name="hbg_inout_orchestration" if retained_models else "hbg_capture_orchestration",
        binary=orch,
        children=[(0, child)],
    )

    (build / "callable.bin").write_bytes(ctypes.string_at(int(chip.buffer_ptr()), int(chip.buffer_size())))
    cann = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest"))
    runtime = _ROOT / "build/lib/a2a3/onboard/host_build_graph"
    command = [
        "c++",
        "-std=c++17",
        "-O2",
        str(_NATIVE / ("hbg_retained_models.cpp" if retained_models else "hbg_driver.cpp")),
        *([] if retained_models else [str(_NATIVE / "capture_submit_fault.cpp")]),
        f"-I{_ROOT / 'src/common/worker'}",
        f"-I{_ROOT / 'src/common/task_interface'}",
        f"-I{_ROOT / 'src/common'}",
        *[f"-I{path}" for path in compiler.get_orchestration_include_dirs("host_build_graph")],
        f"-I{cann / 'include'}",
        f"-I{cann / 'pkg_inc'}",
        f"-I{cann / 'pkg_inc/runtime'}",
        f"-I{cann / 'pkg_inc/profiling'}",
        f"-I{cann / 'pkg_inc/runtime/runtime'}",
        f"-L{runtime}",
        f"-Wl,-rpath,{runtime}",
        "-lhost_runtime",
        f"-L{cann / 'lib64'}",
        f"-Wl,-rpath,{cann / 'lib64'}",
        "-lascendcl",
        "-lruntime",
        "-ldl",
        "-o",
        str(build / "hbg_driver"),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=180, check=False)
    (build / "driver-build.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    return [
        str(build / "hbg_driver"),
        str(build / "callable.bin"),
        str(runtime / "libaicpu_kernel.so"),
        str(runtime / "aicore_kernel.o"),
        str(_ROOT / "build/lib/a2a3/dispatcher/libsimpler_aicpu_dispatcher.so"),
    ]


@pytest.fixture(scope="module")
def hbg_capture_probe(tmp_path_factory):
    build = tmp_path_factory.mktemp("hbg-capture")
    return build, build_probe(build)


def test_hbg_retained_models_keep_persistent_inout(tmp_path, request):
    binaries = build_probe(tmp_path, retained_models=True)
    device = str(request.config.getoption("--device")).split("-")[0].split(",")[0]
    logs = tmp_path / "ascend-retained-models"
    logs.mkdir()
    result = subprocess.run(
        [binaries[0], device, *binaries[1:]],
        env=dict(os.environ, ASCEND_PROCESS_LOG_PATH=str(logs)),
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    (tmp_path / "retained-models.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}\nDevice logs: {logs}"
    assert "hbg_retained_models PASS models=2 references=2 replays=6 eager=1 elements=16" in result.stdout
    for stage in ("a-recorded", "b-prepared", "b-recorded", "b1", "a1", "b2", "a2", "b3", "b-after-a-release", "eager"):
        assert f"state={stage} verified=48" in result.stdout
    for stage in ("a-retains-callable", "a-retains-context", "b-retains-callable", "b-retains-context"):
        assert f"stage={stage} rc=0" in result.stdout
    assert "stage=same-carrier-and-slot rc=0" in result.stdout


@pytest.mark.parametrize(
    "scenario",
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    ids=[
        "dependency",
        "early-dispatch",
        "nested-graph",
        "destroy-unexecuted",
        "reject-duplicate",
        "reject-predecessor",
        "readonly-host-snapshot",
        "cancel-active-capture-and-recover",
        "capture-end-rejection-and-recover",
        "unjoined-capture-end-and-recover",
        "invalidated-capture-and-recover",
        "injected-partial-capture-and-recover",
        "two-device-timeouts-and-same-model-recovery",
        "post-handshake-boot-failure-and-same-model-recovery",
    ],
)
def test_hbg_model_replays_after_host_finalize(hbg_capture_probe, request, scenario):
    build, binaries = hbg_capture_probe
    device = str(request.config.getoption("--device")).split("-")[0].split(",")[0]
    logs = build / f"ascend-{scenario}"
    logs.mkdir()
    env = dict(os.environ, ASCEND_PROCESS_LOG_PATH=str(logs))
    if scenario == 12:
        env.update(
            SIMPLER_SCHEDULER_TIMEOUT_MS="200",
            SIMPLER_OP_EXECUTE_TIMEOUT_US="3000000",
            SIMPLER_STREAM_SYNC_TIMEOUT_MS="6000",
        )
    result = subprocess.run(
        [binaries[0], device, str(scenario), *binaries[1:]],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    (build / f"run-{scenario}.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}\nDevice logs: {logs}"
    if scenario == 4:
        assert "stage=reject-duplicate-capture-launch rc=0" in result.stdout
    if scenario == 5:
        assert "stage=reject-capture-predecessor rc=0" in result.stdout
    if 7 <= scenario <= 11:
        assert "stage=finalize-during-capture rc=0" in result.stdout
        assert "stage=release-live-capture-rejected rc=0" in result.stdout
        assert "stage=release-refusal-keeps-capture rc=0" in result.stdout
        assert "cancelled-capture-cleaned cycle=0" in result.stdout
        assert result.stdout.count("stage=capture-did-not-execute rc=0") == 2
        assert result.stdout.count("stage=destroy-model rc=0") == (1 if scenario >= 9 else 2)
        assert result.stdout.count("stage=release-after-destroy rc=0") == 2
        assert result.stdout.count("verified=16") == 3
    if scenario == 8:
        assert "stage=capture-end-invalid-argument rc=0" in result.stdout
    if 9 <= scenario <= 11:
        assert "stage=failed-end-no-model rc=0" in result.stdout
        assert "stage=other-capture-ended rc=0" in result.stdout
        assert "stage=external-reference-retains-callable rc=0" in result.stdout
    if scenario == 9:
        assert "failed-capture-end rc=107025 expected=107025" in result.stdout
    if scenario in (10, 11):
        assert "failed-capture-end rc=507903 expected=507903" in result.stdout
        assert "stage=failed-other-end-no-model rc=0" in result.stdout
    if scenario == 11:
        assert "partial-capture core=1 cpu=0 after=0 boundary=0 conflict=107032 launch=107032" in result.stdout
        assert "stage=partial-capture-stops-submission rc=0" in result.stdout
        assert "finalize-capture rc=107032 expected=107032" in result.stdout
    if scenario == 12:
        for round_id in (1, 2):
            assert f"fault-replay={round_id} sync=507011" in result.stdout
            assert f"fault-replay={round_id} executed=16" in result.stdout
        device_log = "\n".join(p.read_text(errors="replace") for p in logs.glob("debug/device-*/*.log"))
        assert device_log.count("TIMEOUT_EXIT after_idle_iterations=") == 2, device_log
        assert result.stdout.count("verified=16") == 3
        assert "stage=release-after-destroy rc=0" in result.stdout
    if scenario == 13:
        for round_id in (1, 2):
            assert f"boot-failure={round_id} sync=507011" in result.stdout
        assert result.stdout.count("stage=boot-failure-no-user-task rc=0") == 2
        assert result.stdout.count("stage=boot-failure-retains-callable rc=0") == 2
        assert result.stdout.count("stage=boot-failure-execution-ended rc=0") == 2
        assert "stage=restore-injected-arena-binding rc=0" in result.stdout
        device_log = "\n".join(p.read_text(errors="replace") for p in logs.glob("debug/device-*/*.log"))
        assert device_log.count("host-orch: prebuilt_arena_base is null") == 2, device_log
        assert result.stdout.count("verified=16") == 3
        assert "stage=release-after-destroy rc=0" in result.stdout
    captures = 2 if 7 <= scenario <= 11 else 1
    replays = 0 if scenario == 3 else (5 if scenario in (12, 13) else 3)
    assert (
        f"hbg_capture PASS scenario={scenario} replays={replays} prepare={captures} launch={captures}" in result.stdout
    )


if __name__ == "__main__":
    import sys

    print("\n".join(build_probe(Path(sys.argv[1]))))
