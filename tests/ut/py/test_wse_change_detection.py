# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Replay the canonical change detector, with independent sibling architectures."""

import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[3]


def _detect(tmp_path, files, *, repo=None, base=None):
    workflow = yaml.safe_load((_ROOT / ".github/workflows/_detect-changes.yml").read_text())
    script = next(step["run"] for step in workflow["jobs"]["detect"]["steps"] if step.get("id") == "check")
    # File-list checks start after attribution; Git checks execute the whole step.
    if repo is None:
        script = script[script.index('if [ -z "$FILES" ]; then') :]
    output = tmp_path / "outputs"
    env = dict(os.environ, FILES=files, GITHUB_OUTPUT=str(output))
    if repo is not None:
        env.update(BASE_REPOSITORY="test/base", BASE_REF="main", BASE_SHA=base or "", HEAD_SHA="")
    result = subprocess.run(
        ["bash", "-e", "-c", script], cwd=repo, env=env, check=True, capture_output=True, text=True, timeout=10
    )
    flags = dict(line.split("=", 1) for line in output.read_text().splitlines())
    return flags, result.stdout


@pytest.mark.parametrize(
    ("files", "arches", "categories"),
    [
        ("docs/a.md", (False, False, False), (False, False)),
        ("tests/fdwic/dsv41_scheduler/production/export_cases.py", (False, False, False), (False, False)),
        ("src/wse/docs/sram.md", (False, False, False), (False, False)),
        (".gitignore", (False, False, False), (False, False)),
        ("src/wse/platform/sim/host/device_runner.cpp", (False, False, True), (True, True)),
        ("tests/st/wse/host_build_graph/test_equivalence.py", (False, False, True), (True, True)),
        ("src/a5/platform/sim/host/device_runner.cpp", (False, True, False), (True, True)),
        ("tests/st/a5/host_build_graph/vector_example/test_vector_example.py", (False, True, False), (True, True)),
        ("src/a2a3/platform/sim/host/device_runner.cpp", (True, False, False), (True, True)),
        ("simpler_setup/kernel_compiler.py", (True, True, True), (True, True)),
        (".github/workflows/_st-sim-wse.yml", (True, True, True), (True, True)),
        ("unclassified/new_file.cpp", (True, True, True), (True, True)),
        ("", (True, True, True), (True, True)),
        ("tests/ut/cpp/wse/test_runtime.cpp", (False, False, True), (False, True)),
        ("src/a2a3/runtime.cpp\nsrc/a5/runtime.cpp", (True, True, False), (True, True)),
        ("src/a2a3/runtime.cpp\nsrc/wse/runtime.cpp", (True, False, True), (True, True)),
        ("src/a5/runtime.cpp\nsrc/wse/runtime.cpp", (False, True, True), (True, True)),
        ("tests/st/wse/test_scene.py\ntests/ut/cpp/a5/test_runtime.cpp", (False, True, True), (True, True)),
        ("src/wse/docs/sram.md\nsrc/a5/runtime.cpp", (False, True, False), (True, True)),
    ],
)
def test_architecture_flags_from_actual_detector(tmp_path, files, arches, categories):
    flags, stdout = _detect(tmp_path, files)
    assert tuple(flags[arch + "_changed"] == "true" for arch in ("a2a3", "a5", "wse")) == arches
    assert tuple(flags[category + "_affected"] == "true" for category in ("st", "ut")) == categories
    if files:
        for arch, affected in zip(("a2a3", "a5", "wse"), arches, strict=True):
            if affected:
                assert f"Files affecting {arch}:\n" in stdout
                assert f"skipping {arch}" not in stdout
            else:
                assert f"All changes are confined to other architectures or non-code; skipping {arch}" in stdout
                assert f"Files affecting {arch}:" not in stdout
    if not files:
        assert flags["st_affected"] == flags["ut_affected"] == "true"


def test_workflow_consumers_use_declared_outputs():
    text = (_ROOT / ".github/workflows/_detect-changes.yml").read_text()
    workflow = yaml.load(text, Loader=yaml.BaseLoader)
    outputs = set(workflow["on"]["workflow_call"]["outputs"])
    for path in (_ROOT / ".github/workflows").glob("*.yml"):
        consumer = path.read_text()
        assert set(re.findall(r"needs\.detect-changes\.outputs\.(\w+)", consumer)) <= outputs


@pytest.mark.parametrize(
    "files",
    [
        "tests/fdwic/dsv41_scheduler/production/record_tool.cpp",
        "tests/fdwic/dsv41_scheduler/tests/test_dsv41_half_sync.py",
        "tests/fdwic/dsv41_scheduler/production/requirements-bound.txt",
        "tests/fdwic/dsv41_scheduler/production/sync_bound.py\ndocs/benchmark.md",
    ],
)
def test_manual_only_keeps_precommit_but_no_automatic_test_jobs(tmp_path, files):
    flags, _ = _detect(tmp_path, files)
    assert flags["non_code_only"] == "false"
    assert flags["tests_only"] == "true"
    assert _selected_jobs(flags) == {"detect-changes", "pre-commit"}


def _selected_jobs(flags):
    workflow = yaml.safe_load((_ROOT / ".github/workflows/ci.yml").read_text())
    selected = set()
    for name, job in workflow["jobs"].items():
        expression = job.get("if", "True")
        expression = (
            re.sub(r"needs\.detect-changes\.outputs\.(\w+)", lambda match: repr(flags[match[1]]), expression)
            .replace("&&", " and ")
            .replace("||", " or ")
        )
        if eval(expression, {"__builtins__": {}}):
            selected.add(name)
    return selected


@pytest.mark.parametrize(
    "path",
    [
        "examples/a5/tensormap_and_ringbuffer/host_ddr_map/run_mx.py",
        "examples/a5/tensormap_and_ringbuffer/host_ddr_gather/kernels/aiv/gather.cpp",
        "examples/a5/tensormap_and_ringbuffer/vector_example/test_vector_example.py",
        "tests/st/a5/host_build_graph/vector_example/test_vector_example.py",
    ],
)
def test_a5_ut_payload_changes_select_the_real_ut_jobs(tmp_path, path):
    assert (_ROOT / path).is_file()
    flags, _ = _detect(tmp_path, path)
    assert flags["a5_changed"] == flags["ut_affected"] == "true"
    assert _selected_jobs(flags) & {"ut", "ut-a2a3", "ut-a5"} == {"ut", "ut-a5"}


@pytest.mark.parametrize(
    ("path", "hardware_jobs"),
    [
        ("examples/a5/new_example/main.py", {"ut-a5"}),
        ("examples/a5/new_example/kernels/aiv/kernel.cpp", {"ut-a5"}),
        ("examples/a2a3/new_example/main.py", {"ut-a2a3"}),
        ("examples/a2a3/new_example/kernels/aiv/kernel.cpp", {"ut-a2a3"}),
        ("examples/wse/new_example/main.py", set()),
        ("examples/workers/l2/new_example/main.py", {"ut-a2a3", "ut-a5"}),
        ("tests/st/a5/new_scene/test_scene.py", {"ut-a5"}),
        (
            "tests/st/a5/host_build_graph/graph_execution/kernels/orchestration/graph_execution_orch.cpp",
            {"ut-a5"},
        ),
        ("tests/st/a2a3/new_scene/test_scene.py", {"ut-a2a3"}),
        ("tests/st/a2a3/host_build_graph/graph_execution/test_graph_execution.py", {"ut-a2a3"}),
        ("tests/st/wse/new_scene/test_scene.py", set()),
        ("tests/st/common/new_scene/test_scene.py", {"ut-a2a3", "ut-a5"}),
    ],
)
def test_example_and_scene_changes_select_ut_by_architecture(tmp_path, path, hardware_jobs):
    flags, _ = _detect(tmp_path, path)
    selected = _selected_jobs(flags)
    assert flags["st_affected"] == flags["ut_affected"] == "true"
    assert selected & {"ut", "ut-a2a3", "ut-a5"} == {"ut"} | hardware_jobs
    assert "packaging-matrix" not in selected
    assert "profiling-flags-smoke" not in selected


@pytest.mark.parametrize(
    ("files", "hardware_jobs"),
    [
        ("src/a5/runtime/host_build_graph/runtime.h", {"ut-a5"}),
        ("src/a5/platform/onboard/host/device_runner.cpp", {"ut-a5"}),
        ("src/a5/platform/onboard/host/aicpu_topology_probe_extra.cpp", {"ut-a5"}),
        ("src/a2a3/runtime/host_build_graph/runtime.h", {"ut-a2a3"}),
        ("src/wse/platform/sim/host/device_runner.cpp", set()),
        ("tests/ut/cpp/a5/test_runtime.cpp", {"ut-a5"}),
        ("tests/ut/cpp/a2a3/test_runtime.cpp", {"ut-a2a3"}),
        ("tests/ut/cpp/wse/test_runtime.cpp", set()),
        ("src/common/worker/chip_worker.cpp", {"ut-a2a3", "ut-a5"}),
        ("python/simpler/worker.py", {"ut-a2a3", "ut-a5"}),
        ("tests/ut/py/test_buffer.py", {"ut-a2a3", "ut-a5"}),
        ("examples/a5/new_example/main.py\nexamples/a2a3/new_example/main.py", {"ut-a2a3", "ut-a5"}),
        ("examples/a5/new_example/main.py\nexamples/a2a3/new_example/README.md", {"ut-a5"}),
        ("examples/a5/new_example/main.py\npython/simpler/worker.py", {"ut-a2a3", "ut-a5"}),
    ],
)
def test_hardware_ut_jobs_respect_architecture_in_shared_and_mixed_changes(tmp_path, files, hardware_jobs):
    flags, _ = _detect(tmp_path, files)
    assert _selected_jobs(flags) & {"ut", "ut-a2a3", "ut-a5"} == {"ut"} | hardware_jobs


@pytest.mark.parametrize(
    "path",
    [
        "src/a5/platform/onboard/host/aicpu_topology_probe.cpp",
        "src/a5/platform/onboard/host/aicpu_affinity_select.cpp",
        "src/a5/platform/onboard/host/aicpu_topology_probe.h",
        "src/a5/platform/onboard/host/aicpu_cpu_topo_fallback.json",
        "src/a5/platform/include/common/platform_config.h",
    ],
)
def test_shared_device_query_inputs_select_both_hardware_ut_jobs(tmp_path, path):
    assert (_ROOT / path).is_file()
    flags, _ = _detect(tmp_path, path)
    assert (flags["a2a3_changed"], flags["a5_changed"], flags["wse_changed"]) == ("true", "true", "false")
    assert _selected_jobs(flags) & {"ut", "ut-a2a3", "ut-a5"} == {"ut", "ut-a2a3", "ut-a5"}


@pytest.mark.parametrize(("source_arch", "target_arch"), [("a2a3", "a5"), ("a5", "a2a3")])
def test_real_cross_architecture_rename_selects_both_hardware_ut_jobs(tmp_path, source_arch, target_arch):
    repo = tmp_path / "repo"
    repo.mkdir()
    env = dict(
        os.environ,
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_CONFIG_NOSYSTEM="1",
        GIT_AUTHOR_NAME="CI",
        GIT_AUTHOR_EMAIL="ci@example.invalid",
        GIT_COMMITTER_NAME="CI",
        GIT_COMMITTER_EMAIL="ci@example.invalid",
    )

    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=repo, env=env, check=True, capture_output=True, text=True
        ).stdout.strip()

    git("init", "-b", "main")
    git("config", "diff.renames", "true")
    source = f"src/{source_arch}/platform/onboard/aicore/sdma_warmup_kernel.cpp"
    target = f"src/{target_arch}/platform/onboard/aicore/sdma_warmup_kernel.cpp"
    (repo / source).parent.mkdir(parents=True)
    payload = _ROOT / "src/a2a3/platform/onboard/aicore/sdma_warmup_kernel.cpp"
    (repo / source).write_bytes(payload.read_bytes())
    git("add", ".")
    git("commit", "-m", "base")
    base = git("rev-parse", "HEAD")
    (repo / target).parent.mkdir(parents=True)
    git("mv", source, target)
    git("commit", "-m", "move across architectures")
    assert git("diff", "--name-status", base + "...HEAD") == f"R100\t{source}\t{target}"
    flags, stdout = _detect(tmp_path, "", repo=repo, base=base)
    assert source in stdout and target in stdout
    assert flags["a2a3_changed"] == flags["a5_changed"] == "true"
    assert _selected_jobs(flags) & {"ut", "ut-a2a3", "ut-a5"} == {"ut", "ut-a2a3", "ut-a5"}


@pytest.mark.parametrize(
    "path",
    [
        "examples/a5/new_example/README.md",
        "examples/a2a3/new_example/README.md",
        "examples/wse/new_example/README.md",
        "examples/a5/tensormap_and_ringbuffer/host_ddr_map/README.md",
        "tests/st/a5/host_build_graph/vector_example/README.md",
        "tests/st/a2a3/host_build_graph/graph_execution/README.md",
        "tests/st/a5/host_build_graph/graph_execution/README.md",
    ],
)
def test_example_and_scene_docs_still_skip_ut(tmp_path, path):
    flags, _ = _detect(tmp_path, path)
    assert flags["ut_affected"] == "false"
    assert not {"ut", "ut-a2a3", "ut-a5"} & _selected_jobs(flags)


def test_example_change_is_selected_with_manual_benchmark_and_docs(tmp_path):
    files = "\n".join(
        [
            "tests/fdwic/dsv41_scheduler/production/export_cases.py",
            "examples/a5/new_example/main.py",
            "docs/ci.md",
        ]
    )
    flags, _ = _detect(tmp_path, files)
    assert _selected_jobs(flags) & {"ut", "ut-a2a3", "ut-a5"} == {"ut", "ut-a5"}


@pytest.mark.parametrize(
    ("files", "expected"),
    [
        ("docs/ci.md", False),
        ("src/a2a3/runtime/host_build_graph/runtime.h", False),
        ("src/wse/platform/sim/host/device_runner.cpp", False),
        ("src/a5/runtime/host_build_graph/runtime.h", True),
        ("simpler_setup/test_environments.py", True),
        ("tests/ut/cpp/a5/test_runtime.cpp", False),
        ("tests/st/a5/host_build_graph/test_scene.py", True),
    ],
)
def test_automatic_a5_framework_uses_actual_architecture_and_scene_flags(tmp_path, files, expected):
    flags, _ = _detect(tmp_path, files)
    assert ("st-framework-a5" in _selected_jobs(flags)) == expected


@pytest.mark.parametrize(
    ("other", "arches", "categories"),
    [
        ("src/a5/runtime/host_build_graph/runtime.h", (False, True, False), (True, True)),
        ("tests/st/a2a3/vector/test_vector.py", (True, False, False), (True, True)),
        ("tests/ut/py/test_buffer.py", (True, True, True), (False, True)),
        ("simpler_setup/kernel_compiler.py", (True, True, True), (True, True)),
        (".github/workflows/ci.yml", (True, True, True), (True, True)),
        ("unclassified/new.cpp", (True, True, True), (True, True)),
        ("tests/fdwic/dsv41_scheduler_other/new.py", (True, True, True), (True, True)),
    ],
)
def test_manual_exclusion_does_not_hide_other_changes(tmp_path, other, arches, categories):
    files = "tests/fdwic/dsv41_scheduler/production/export_cases.py\n" + other
    flags, _ = _detect(tmp_path, files)
    assert tuple(flags[arch + "_changed"] == "true" for arch in ("a2a3", "a5", "wse")) == arches
    assert tuple(flags[category + "_affected"] == "true" for category in ("st", "ut")) == categories
    if other == ".github/workflows/ci.yml":
        workflow = yaml.safe_load((_ROOT / ".github/workflows/ci.yml").read_text())
        assert _selected_jobs(flags) == set(workflow["jobs"])


def test_unknown_diff_still_selects_every_automatic_job(tmp_path):
    flags, _ = _detect(tmp_path, "")
    workflow = yaml.safe_load((_ROOT / ".github/workflows/ci.yml").read_text())
    assert _selected_jobs(flags) == set(workflow["jobs"])
