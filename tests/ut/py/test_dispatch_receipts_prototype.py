# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CPU-only real subprocess proof for default-environment receipt aggregation."""

import json
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from simpler_setup.test_environments import verify_result_files

ROOT = Path(__file__).resolve().parents[3]
HOOKS = (
    "pytest_addoption",
    "pytest_configure",
    "pytest_itemcollected",
    "pytest_collection_modifyitems",
    "pytest_collection_finish",
    "pytest_runtestloop",
    "pytest_runtest_makereport",
    "pytest_sessionfinish",
)
CASES = """
import pytest

@pytest.mark.device_count(1)
@pytest.mark.runtime("host_build_graph")
@pytest.mark.execution_mode("eager")
def test_resource():
    pass

@pytest.mark.execution_mode("eager")
class TestL2:
    _st_level = 2
    _st_runtime = "tensormap_and_ringbuffer"
    CASES = [{"name": "cpu", "platforms": ["a2a3"]}]
    def test_run(self):
        pass

@pytest.mark.execution_mode("capture_replay")
def test_capture_leftover():
    pass
"""


@pytest.fixture
def execution():
    # Keep the generated checkout inside this worktree so real git identity is used.
    with tempfile.TemporaryDirectory(prefix="receipt-cpu-", dir=ROOT) as temporary:
        directory = Path(temporary)
        (directory / "pytest.ini").write_text("[pytest]\n")
        (directory / "conftest.py").write_text(f"""
import importlib.util
import sys
spec = importlib.util.spec_from_file_location("receipt_real_hooks", {str(ROOT / "conftest.py")!r})
real = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = real
spec.loader.exec_module(real)
real.ensure_pto_isa_root = lambda **kwargs: None
for name in {HOOKS!r}:
    globals()[name] = getattr(real, name)
""")

        def run(source=CASES, *extra, receipts=True, extra_files=None, require_capture=True):
            (directory / "test_cpu.py").write_text(source)
            for filename, content in (extra_files or {}).items():
                (directory / filename).write_text(content)
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "test_cpu.py",
                    "--confcutdir",
                    str(directory),
                    "--platform",
                    "a2a3",
                    "--max-parallel",
                    "1",
                    *(["--environment-receipts"] if receipts else []),
                    *(["--require-execution-mode", "capture_replay"] if receipts and require_capture else []),
                    "--junitxml",
                    str(directory / "results.xml"),
                    *extra,
                ],
                cwd=directory,
                env={
                    **os.environ,
                    "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
                    "PYTHONPATH": os.pathsep.join((str(ROOT), str(ROOT / "python"))),
                },
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            return result, directory / "results.plan.json", sorted(directory.glob("**/results.xml"))

        yield run


def test_real_resource_l2_and_leftover_receipts(execution):
    result, plan, paths = execution()
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(paths) == 3
    assert len(json.loads(plan.read_text())["cases"]) == 3
    verify_result_files(plan, paths, dispatcher=True)
    summary = json.loads(plan.with_suffix(".coverage.json").read_text())
    assert summary["groups"] == {
        "eager": {"selected": 2, "executed": 2, "passed": 2},
        "capture_replay": {"selected": 1, "executed": 1, "passed": 1},
    }


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "source", "empty", "skip"])
def test_actual_shard_corruption_is_rejected(execution, mutation):
    result, plan, paths = execution()
    assert result.returncode == 0, result.stdout + result.stderr
    if mutation == "missing":
        paths.pop()
    elif mutation == "duplicate":
        paths.append(paths[0])
    else:
        tree = ET.parse(paths[0])
        case = next(tree.getroot().iter("testcase"))
        if mutation == "source":
            source_property = case.find("properties/property[@name='simpler_source']")
            assert source_property is not None
            source_property.set("value", "foreign")
        elif mutation == "empty":
            tree.getroot().clear()
        else:
            ET.SubElement(case, "skipped")
        tree.write(paths[0])
    with pytest.raises(ValueError):
        verify_result_files(plan, paths, dispatcher=True)


def test_real_resource_skip_fails_parent(execution):
    source = CASES.replace("def test_resource():\n    pass", 'def test_resource():\n    pytest.skip("unexpected")')
    result, _, _ = execution(source)
    assert result.returncode != 0
    assert "VERIFICATION_FAILED" in result.stderr


def test_baseline_default_does_not_make_a_plan(execution):
    # The opt-in is necessary: existing default runs do not emit strict receipts.
    result, plan, _ = execution(receipts=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not plan.exists()


def test_required_capture_filtered_out_fails(execution):
    result, _, _ = execution(CASES, "-k", "not capture_leftover")
    assert result.returncode != 0
    assert "Required execution modes not selected" in result.stderr


def test_missing_mode_marker_fails(execution):
    result, _, _ = execution(CASES.replace('@pytest.mark.execution_mode("capture_replay")', ""))
    assert result.returncode != 0
    assert "Missing or conflicting execution_mode" in result.stderr


def test_top_level_xdist_requires_parent_plan(execution):
    result, _, _ = execution(CASES, "-p", "xdist.plugin", "-n", "2")
    assert result.returncode != 0
    assert "parent plan" in result.stderr


PARALLEL = CASES.replace(
    "def test_run(self):\n        pass",
    """@pytest.mark.parametrize("index", range(8))
    def test_run(self, request, index):
        import os
        request.node.user_properties.append(("worker", os.environ["PYTEST_XDIST_WORKER"]))""",
)


def test_real_l2_xdist_uses_both_workers(execution):
    result, plan, paths = execution(PARALLEL, "-p", "xdist", "--max-parallel", "2", "--device", "0-1", "--dist", "load")
    assert result.returncode == 0, result.stdout + result.stderr
    workers = {
        p.get("value") for path in paths for p in ET.parse(path).getroot().iter("property") if p.get("name") == "worker"
    }
    assert workers == {"gw0", "gw1"}
    assert len(json.loads(plan.read_text())["cases"]) == 10
    verify_result_files(plan, paths, dispatcher=True)


def test_real_l2_default_loadfile_schedule(execution):
    additional = (
        "import pytest\n" + PARALLEL.split("class TestL2:")[1].split('@pytest.mark.execution_mode("capture_replay")')[0]
    )
    additional = additional.replace(
        "import pytest\n", 'import pytest\n@pytest.mark.execution_mode("eager")\nclass TestMore:'
    )
    result, plan, paths = execution(
        PARALLEL,
        "test_more.py",
        "-p",
        "xdist",
        "--max-parallel",
        "2",
        "--device",
        "0-1",
        extra_files={"test_more.py": additional},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    workers = {
        p.get("value") for path in paths for p in ET.parse(path).getroot().iter("property") if p.get("name") == "worker"
    }
    assert workers == {"gw0", "gw1"}
    assert len(json.loads(plan.read_text())["cases"]) == 18
    verify_result_files(plan, paths, dispatcher=True)


POISON = (
    CASES
    + """
@pytest.mark.execution_mode("eager")
class TestPoison:
    _st_level = 2
    _st_runtime = "tensormap_and_ringbuffer"
    CASES = [{"name": "cpu", "platforms": ["a2a3"]}]
    def test_run(self, request):
        import os
        import sys
        if os.environ.get("SIMPLER_L2_POISON_SINK"):
            sys.modules["receipt_real_hooks"]._register_l2_poison_skip(request.node)
            pytest.skip("poison guard")
        assert True
"""
)


@pytest.mark.parametrize("parallel", [False, True])
def test_poison_retry_only_replays_registered_case(execution, parallel):
    args = ("-p", "xdist", "--max-parallel", "2", "--device", "0-1", "--dist", "load") if parallel else ()
    result, plan, paths = execution(POISON, *args)
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads(plan.with_suffix(".results.json").read_text())
    assert len(manifest["retries"]) == 1
    retry_path, authorization = next(iter(manifest["retries"].items()))
    assert authorization["nodeids"] == ["test_cpu.py::TestPoison::test_run"]
    assert len(list(ET.parse(retry_path).getroot().iter("testcase"))) == 1
    verify_result_files(plan, paths, dispatcher=True, retries=manifest["retries"])
    summary = json.loads(plan.with_suffix(".coverage.json").read_text())
    assert summary["attempts"] == 5
    assert summary["recovered"] == 1
    assert summary["groups"]["eager"]["passed"] == 3


@pytest.mark.parametrize("fault", ["original_failure", "retry_failure", "retry_skip", "unregistered_skip"])
def test_poison_retry_does_not_hide_failures(execution, fault):
    source = POISON
    if fault == "original_failure":
        source = source.replace('pytest.skip("poison guard")', 'pytest.fail("original failure")')
    elif fault == "retry_failure":
        source = source.replace("assert True", 'pytest.fail("retry failure")')
    elif fault == "retry_skip":
        source = source.replace("assert True", 'pytest.skip("still poisoned")')
    else:
        source = source.replace('sys.modules["receipt_real_hooks"]._register_l2_poison_skip(request.node)', "pass")
    result, _, _ = execution(source, "-p", "xdist", "--max-parallel", "2", "--device", "0-1")
    assert result.returncode != 0
    assert "VERIFICATION_FAILED" in result.stderr


def test_retry_preserves_parameter_instance_identity(execution):
    source = POISON.replace(
        "    def test_run(self, request):",
        '    @pytest.mark.parametrize("index", range(3))\n    def test_run(self, request, index):',
    ).replace(
        'if os.environ.get("SIMPLER_L2_POISON_SINK"):', 'if os.environ.get("SIMPLER_L2_POISON_SINK") and index != 1:'
    )
    result, plan, paths = execution(source, "-p", "xdist", "--max-parallel", "2", "--device", "0-1", "--dist", "load")
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads(plan.with_suffix(".results.json").read_text())
    authorization = next(iter(manifest["retries"].values()))
    assert authorization["nodeids"] == ["test_cpu.py::TestPoison::test_run[0]", "test_cpu.py::TestPoison::test_run[2]"]
    verify_result_files(plan, paths, dispatcher=True, retries=manifest["retries"])
    summary = json.loads(plan.with_suffix(".coverage.json").read_text())
    assert summary["attempts"] == 8
    assert summary["recovered"] == 2


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "source", "environment", "plan"])
def test_real_xdist_receipts_reject_corruption(execution, mutation):
    result, plan, paths = execution(PARALLEL, "-p", "xdist", "--max-parallel", "2", "--device", "0-1", "--dist", "load")
    assert result.returncode == 0, result.stdout + result.stderr
    for path in paths:
        tree = ET.parse(path)
        cases = list(tree.getroot().iter("testcase"))
        if len(cases) == 8:
            break
    else:
        pytest.fail("Missing actual xdist JUnit")
    case = cases[0]
    suite = next(suite for suite in tree.getroot().iter("testsuite") if case in suite)
    if mutation == "missing":
        suite.remove(case)
    elif mutation == "duplicate":
        suite.append(case)
    else:
        property_element = case.find(f"properties/property[@name='simpler_{mutation}']")
        assert property_element is not None
        property_element.set("value", "foreign")
    tree.write(path)
    with pytest.raises(ValueError):
        verify_result_files(plan, paths, dispatcher=True)
    assert not plan.with_suffix(".coverage.json").exists()


def test_xdist_worker_crash_cannot_qualify_coverage(execution):
    source = PARALLEL.replace(
        "request.node.user_properties.append",
        "if index == 0:\n            os._exit(23)\n        request.node.user_properties.append",
    )
    result, plan, _ = execution(
        source, "-p", "xdist", "--max-parallel", "2", "--device", "0-1", "--max-worker-restart", "0"
    )
    assert result.returncode != 0
    assert "VERIFICATION_FAILED" in result.stderr
    assert not plan.with_suffix(".coverage.json").exists()


@pytest.mark.parametrize("mutation", ["missing_file", "missing_case", "duplicate", "no_authorization", "wrong_node"])
def test_retry_evidence_is_mandatory(execution, mutation):
    result, plan, paths = execution(POISON)
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads(plan.with_suffix(".results.json").read_text())
    retries = manifest["retries"]
    retry_path = Path(next(iter(retries)))
    if mutation == "missing_file":
        retry_path.unlink()
    elif mutation == "no_authorization":
        retries = {}
    elif mutation == "wrong_node":
        retries[str(retry_path)]["nodeids"] = ["test_cpu.py::test_capture_leftover"]
    else:
        tree = ET.parse(retry_path)
        case = next(tree.getroot().iter("testcase"))
        suite = next(suite for suite in tree.getroot().iter("testsuite") if case in suite)
        if mutation == "missing_case":
            suite.remove(case)
        else:
            suite.append(case)
        tree.write(retry_path)
    with pytest.raises((ValueError, OSError)):
        verify_result_files(plan, paths, dispatcher=True, retries=retries)
