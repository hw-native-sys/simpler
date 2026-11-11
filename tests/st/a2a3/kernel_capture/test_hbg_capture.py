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
import json
import logging
import os
import struct
import subprocess
import tempfile
from collections import Counter
from itertools import chain
from pathlib import Path

import pytest
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import ChipCallable, CoreCallable

from simpler_setup.elf_parser import extract_text_section
from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.pto_isa import ensure_pto_isa_root
from simpler_setup.runtime_builder import RuntimeBuilder

_ROOT = Path(__file__).resolve().parents[4]
_NATIVE = Path(__file__).with_name("native")
pytestmark = [pytest.mark.requires_hardware, pytest.mark.platforms(["a2a3"]), pytest.mark.runtime("host_build_graph")]


def build_probe(
    build: Path,
    *,
    retained_models: bool = False,
    args_dump: bool = False,
    runtime_name: str = "host_build_graph",
    platform: str = "a2a3",
):
    build.mkdir(parents=True, exist_ok=True)
    compiler = KernelCompiler(platform)
    kernel = compiler.compile_incore(
        str(_NATIVE / ("hbg_inout_step.cpp" if retained_models else "hbg_step.cpp")),
        core_type="aiv",
        pto_isa_root=ensure_pto_isa_root(),
        extra_include_dirs=compiler.get_orchestration_include_dirs(runtime_name),
        build_dir=str(build),
    )
    orch = compiler.compile_orchestration(
        runtime_name,
        str(_NATIVE / ("args_dump_orchestration.cpp" if args_dump else "hbg_orchestration.cpp")),
        build_dir=str(build),
    )
    child = CoreCallable.build(
        signature=[D.INOUT, D.SCALAR] if retained_models else [D.IN, D.OUT, D.SCALAR],
        binary=extract_text_section(kernel),
    )
    chip = ChipCallable.build(
        signature=[D.INOUT, D.SCALAR] if retained_models else [D.IN, D.OUT, D.IN, D.SCALAR],
        func_name=(
            "args_dump_capture_orchestration"
            if args_dump
            else "hbg_inout_orchestration"
            if retained_models
            else "hbg_capture_orchestration"
        ),
        binary=orch,
        children=[(0, child)],
    )

    (build / "callable.bin").write_bytes(ctypes.string_at(int(chip.buffer_ptr()), int(chip.buffer_size())))
    cann = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest"))
    runtime = RuntimeBuilder(platform).get_binaries(runtime_name, build=False)
    runtime_dir = runtime.host_path.parent
    assert runtime.dispatcher_path is not None
    command = [
        "c++",
        "-std=c++17",
        "-O2",
        str(_NATIVE / ("hbg_retained_models.cpp" if retained_models else "hbg_driver.cpp")),
        *([] if retained_models else [str(_NATIVE / "capture_submit_fault.cpp")]),
        f"-I{_ROOT / 'src/common/worker'}",
        f"-I{_ROOT / 'src/common/task_interface'}",
        f"-I{_ROOT / 'src/common'}",
        *[f"-I{path}" for path in compiler.get_orchestration_include_dirs(runtime_name)],
        f"-I{cann / 'include'}",
        f"-I{cann / 'pkg_inc'}",
        f"-I{cann / 'pkg_inc/runtime'}",
        f"-I{cann / 'pkg_inc/profiling'}",
        f"-I{cann / 'pkg_inc/runtime/runtime'}",
        f"-L{runtime_dir}",
        f"-Wl,-rpath,{runtime_dir}",
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
        str(runtime.aicpu_path),
        str(runtime.aicore_path),
        str(runtime.dispatcher_path),
    ]


@pytest.fixture(scope="module")
def hbg_capture_probe(tmp_path_factory, request):
    build = tmp_path_factory.mktemp("hbg-capture")
    return build, build_probe(build, platform=request.config.getoption("--platform"))


@pytest.mark.platforms(["a2a3", "a5"])
def test_hbg_retained_models_keep_persistent_inout(tmp_path, request):
    binaries = build_probe(tmp_path, retained_models=True, platform=request.config.getoption("--platform"))
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


def _run_capture_probe(command, run, env):
    # Streams reach disk before subprocess.run returns or raises TimeoutExpired.
    try:
        with (run / "stdout.log").open("w") as stdout, (run / "stderr.log").open("w") as stderr:
            result = subprocess.run(
                command,
                env=env,
                stdout=stdout,
                stderr=stderr,
                text=True,
                timeout=180,
                check=False,
            )
    finally:
        # task-submit can run as root; the artifact uploader runs as the runner user.
        try:
            for path in chain((run,), run.rglob("*")):
                try:
                    if not path.is_symlink():
                        path.chmod(0o755 if path.is_dir() else 0o644)
                except OSError as error:
                    logging.warning("Capture log permission normalization failed for %s: %s", path, error)
        except OSError as error:
            logging.warning("Capture log permission normalization traversal failed for %s: %s", run, error)
    result.stdout = (run / "stdout.log").read_text(errors="replace")
    result.stderr = (run / "stderr.log").read_text(errors="replace")
    return result


@pytest.mark.parametrize(
    "scenario",
    [
        *(pytest.param(scenario, marks=pytest.mark.platforms(["a2a3", "a5"])) for scenario in range(12)),
        # These two injections assert the A2/A3 scheduler and boot failure paths.
        12,
        13,
    ],
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
    _, binaries = hbg_capture_probe
    device = str(request.config.getoption("--device")).split("-")[0].split(",")[0]
    output = _ROOT / "outputs/hbg-capture"
    output.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix=f"scenario-{scenario}-device-{device}-", dir=output))
    logs = run / "ascend"
    logs.mkdir()
    env = dict(os.environ, ASCEND_PROCESS_LOG_PATH=str(logs))
    if scenario == 12:
        env.update(
            SIMPLER_SCHEDULER_TIMEOUT_MS="200",
            SIMPLER_OP_EXECUTE_TIMEOUT_US="3000000",
            SIMPLER_STREAM_SYNC_TIMEOUT_MS="6000",
        )
    result = _run_capture_probe([binaries[0], device, str(scenario), *binaries[1:]], run, env)
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


@pytest.mark.parametrize("replay", [0, 1], ids=["zero", "interleaved-models"])
def test_hbg_retained_models_publish_pmu_samples(tmp_path, request, replay):
    from collections import Counter  # noqa: PLC0415

    from simpler_setup.tools.pmu_samples import read_pmu_samples  # noqa: PLC0415

    binaries = build_probe(tmp_path, retained_models=True)
    device = str(request.config.getoption("--device")).split("-")[0].split(",")[0]
    logs = tmp_path / "ascend-pmu"
    logs.mkdir()
    output = tmp_path / "pmu"
    result = subprocess.run(
        [binaries[0], device, *binaries[1:], str(output), str(replay)],
        env=dict(os.environ, ASCEND_PROCESS_LOG_PATH=str(logs)),
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    (tmp_path / "pmu-native.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "stage=pmu-flush rc=0" in result.stdout
    ordinary = read_pmu_samples(output / "eager/pmu.csv")
    assert ordinary["records"] and all(int(row["pmu_total_cycles"]) > 0 for row in ordinary["records"])
    for leaf, event, rounds in (("a", 2, 2), ("b", 4, 4)):
        captured = read_pmu_samples(output / leaf / "pmu.csv", recorded=True)
        assert captured["collection"]["event_type"] == event
        assert len(captured["records"]) == replay * rounds * len(ordinary["records"])
        assert all(int(row["pmu_total_cycles"]) > 0 for row in captured["records"])
        if replay:
            assert max(Counter(row["task_id"] for row in captured["records"]).values()) >= rounds


@pytest.mark.parametrize("replay", [0, 1], ids=["zero", "interleaved-models"])
def test_hbg_retained_models_publish_swimlane_observations(tmp_path, request, replay):
    import sys  # noqa: PLC0415
    from collections import Counter  # noqa: PLC0415

    binaries = build_probe(tmp_path, retained_models=True)
    device = str(request.config.getoption("--device")).split("-")[0].split(",")[0]
    logs = tmp_path / "ascend-swimlane"
    logs.mkdir()
    output = tmp_path / "swimlane"
    result = subprocess.run(
        [binaries[0], device, *binaries[1:], str(output), str(replay), "swimlane"],
        env=dict(os.environ, ASCEND_PROCESS_LOG_PATH=str(logs)),
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    (tmp_path / "swimlane-native.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "stage=swimlane-flush rc=0" in result.stdout
    if replay:
        assert result.stdout.index("state=eager verified=48") < result.stdout.index(
            "state=b-after-a-release verified=48"
        )
    ordinary = json.loads((output / "eager/chip_swimlane_records.json").read_text())
    assert ordinary["aicore_tasks"] and not ordinary["metadata"].get("collection", {}).get("recorded_window")
    assert not (output / "unused").exists()
    for leaf, rounds in (("a", 2), ("b", 4)):
        [path] = (output / leaf).glob("swimlane-*/records_e*.json")
        before = path.read_bytes()
        raw = json.loads(before)
        collection = raw["metadata"]["collection"]
        assert raw["chip_swimlane_level"] == 4
        assert collection["recorded_window"] is True and collection["counts_unknown"] is True
        assert collection["processing_complete"] and collection["metadata_complete"], collection
        assert all(collection[key] is None for key in ("device_total", "device_dropped", "fatal", "reconciliation"))
        # This fixed one-kernel graph has one task per replay; this is not an exported replay denominator.
        assert len(raw["aicore_tasks"]) == replay * rounds * len(ordinary["aicore_tasks"])
        if replay:
            assert max(Counter(row[1] for row in raw["aicore_tasks"]).values()) >= rounds
        trace_path = path.with_name("observations.json")
        converted = subprocess.run(
            [sys.executable, "-m", "simpler_setup.tools.swimlane_converter", str(path), "-o", str(trace_path)],
            cwd=_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        (path.parent / "converter.log").write_text(converted.stdout + converted.stderr)
        assert converted.returncode == 0, converted.stdout + converted.stderr
        assert path.read_bytes() == before
        trace = json.loads(trace_path.read_text())
        assert trace["metadata"]["clock_domain"] == "device_syscnt_cycles"
        assert trace["metadata"]["clock_alignment"]["status"] == "unavailable"
        assert trace["metadata"]["collection"] == collection
        events = [event for event in trace["traceEvents"] if event["ph"] == "X"]
        assert all(event["ph"] in ("M", "X") for event in trace["traceEvents"])
        observed = [event for event in events if event["args"]["population"] == "aicore_tasks"]
        assert len(observed) == len(raw["aicore_tasks"])
        assert [event["args"]["record_index"] for event in observed] == list(range(len(raw["aicore_tasks"])))
        host = json.loads((path.parent / trace["metadata"]["host_preparation_file"]).read_text())
        assert host["metadata"]["clock_domain"] == "host_monotonic_ns" and host["metadata"]["preparation_only"] is True
        # Frozen preparation exists once even for zero replay; no Host rows are multiplied by rounds.
        preparation_count = sum(len(stream) for stream in raw.get("host_orchestrator_phases", [])) + len(
            raw.get("host_device_uploads", [])
        )
        assert preparation_count > 0
        assert len([event for event in host["traceEvents"] if event["ph"] == "X"]) == preparation_count


def check_args_dump_output(output: Path, replays: int):
    from simpler_setup.tools import dump_viewer  # noqa: PLC0415

    for leaf, initial, increment, rounds, recorded in (
        ("a", 10, -3, replays, True),
        ("b", 1000, 5, replays, True),
        ("eager", -24680, 17, 1, False),
    ):
        directory = output / leaf / "args_dump"
        manifest = json.loads((directory / "args_dump.json").read_text())
        entries = manifest["args"]
        assert bool(manifest.get("recorded_window")) is recorded, manifest
        if recorded:
            assert manifest["counts_unknown"] is True, manifest
            assert manifest["dropped_records"] is None, manifest
            assert manifest["dropped_records_complete"] is False, manifest
            assert manifest["collection_verdict"] == "aggregate_counts_unknown", manifest
        else:
            assert manifest["dropped_records"] == 0, manifest
        assert manifest.get("host_discarded_args", 0) == 0, manifest
        assert manifest.get("metadata_discarded_records", 0) == 0, manifest
        assert manifest["dump_args_level"] == (3 if recorded else 1), manifest
        assert manifest["total_args"] == len(entries), manifest
        if not rounds:
            assert not entries and manifest["bin_file"] is None, manifest
            continue

        assert entries, manifest
        assert len({entry["run_epoch"] for entry in entries}) == 1, entries
        binary = directory / manifest["bin_file"]
        scalar_by_task = {}
        scalar_occurrences = Counter()
        for entry in entries:
            if entry["kind"] == "scalar":
                assert entry["arg_index"] == 1 and entry["dtype"] == "INT32", entry
                assert entry["stage"] == "before_dispatch", entry
                value = entry["value"]
                # Signed INT32 JSON values preserve the complete inline scalar bit pattern.
                assert struct.pack("<i", value) in (struct.pack("<i", increment), struct.pack("<i", 7)), entry
                identity = (entry["task_id"], tuple(entry["func_id"]))
                assert scalar_by_task.setdefault(identity, value) == value, entry
                scalar_occurrences[identity] += 1
        assert len(scalar_by_task) == 2 and set(scalar_by_task.values()) == {increment, 7}, scalar_by_task
        assert scalar_occurrences == Counter({identity: rounds for identity in scalar_by_task}), scalar_occurrences
        expected = Counter()
        for round_index in range(rounds):
            seed = initial + 100 * round_index
            for scalar, before, after in (
                (increment, seed, seed + increment),
                (7, seed + increment, seed + increment + 7),
            ):
                for stage, base in (("before_dispatch", before), ("after_completion", after)):
                    expected[(stage, scalar, tuple(base + 7 * i for i in range(16)))] += 1
        observed = Counter()
        ranges = set()
        for index, entry in enumerate(entries):
            exported = dump_viewer.export_arg(entry, binary, directory, entry_index=index if recorded else None)
            assert exported.exists()
            if entry["kind"] == "scalar":
                assert f"# value: {entry['value']}" in exported.read_text()
                continue
            assert entry["role"] == "inout" and entry["arg_index"] == 0, entry
            assert entry["dtype"] == "INT32" and entry["shape"] == [16], entry
            assert not entry["truncated"] and not entry.get("host_discarded"), entry
            payload = dump_viewer.read_arg_data(binary, entry["bin_offset"], entry["bin_size"])
            values = struct.unpack("<16i", payload)
            scalar = scalar_by_task[(entry["task_id"], tuple(entry["func_id"]))]
            observed[(entry["stage"], scalar, values)] += 1
            assert f"[0] {values[0]}\n" in exported.read_text()
            assert f"[15] {values[-1]}\n" in exported.read_text()
            ranges.add((entry["bin_offset"], entry["bin_size"]))
        assert observed == expected, (leaf, observed, expected)
        tensors = [entry for entry in entries if entry["kind"] != "scalar"]
        assert len(ranges) == len(tensors), "separate samples alias exported payload offsets"
        if recorded:
            assert len(list((directory / "txt").glob("*.txt"))) == len(entries)


def run_args_dump_probe(binaries, device, output: Path, replays: int):
    logs = output.parent / "ascend-argsdump"
    logs.mkdir()
    result = subprocess.run(
        [binaries[0], str(device), *binaries[1:], str(output), str(replays), "argsdump"],
        env=dict(os.environ, ASCEND_PROCESS_LOG_PATH=str(logs)),
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    (output.parent / "argsdump-native.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "stage=argsdump-flush rc=0" in result.stdout
    assert f"argsdump_retained_models PASS models=2 replays={2 * replays}" in result.stdout
    check_args_dump_output(output, replays)


@pytest.fixture(scope="module")
def hbg_args_dump_probe(tmp_path_factory):
    return build_probe(tmp_path_factory.mktemp("hbg-argsdump-capture"), retained_models=True, args_dump=True)


@pytest.mark.parametrize("replays", [0, 3])
def test_hbg_args_dump_capture_exports_task_time_values(hbg_args_dump_probe, tmp_path, request, replays):
    device = str(request.config.getoption("--device")).split("-")[0].split(",")[0]
    run_args_dump_probe(hbg_args_dump_probe, device, tmp_path / "dump", replays)
