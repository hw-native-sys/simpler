# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Real pytest selection and receipts, with only package preflight replaced."""

import builtins
import copy
import importlib.util
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import pytest

from simpler_setup import test_environments

_ROOT = Path(__file__).resolve().parents[3]
_HOOKS = (
    "pytest_addoption",
    "pytest_configure",
    "pytest_itemcollected",
    "pytest_collection_modifyitems",
    "pytest_collection_finish",
    "pytest_runtestloop",
    "pytest_sessionfinish",
)
_DIRECT = """
import pytest

@pytest.mark.test_environment("torch_npu")
@pytest.mark.parametrize("index", range({count}))
def test_direct(index):
    assert index >= 0
"""


def _run_pytest(
    directory,
    source,
    *args,
    profile="torch_npu",
    path="tests/new_place/test_direct.py",
    capable=True,
    extra_conftest="",
):
    directory.mkdir(parents=True, exist_ok=True)
    test_path = directory / path
    test_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.write_text(source)
    (directory / "pytest.ini").write_text("[pytest]\naddopts = --import-mode=importlib\nmarkers =\n    selected_tag\n")
    (directory / "conftest.py").write_text(
        f"""
import builtins
import importlib.util
import sys
from pathlib import Path

real_import = builtins.__import__
def without_framework(name, *args, **kwargs):
    if name.split(".")[0] in ("torch", "torch_npu"):
        raise AssertionError("CPU contract cannot import a framework backend")
    return real_import(name, *args, **kwargs)
builtins.__import__ = without_framework

spec = importlib.util.spec_from_file_location("real_test_conftest", {str(_ROOT / "conftest.py")!r})
real = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = real
spec.loader.exec_module(real)
# These CPU cases do not compile PTO kernels or need a managed checkout.
real.ensure_pto_isa_root = lambda **kwargs: None
for name in {_HOOKS!r}:
    globals()[name] = getattr(real, name)

def package_preflight(name, *, packages=True):
    Path({str(directory / "preflight-called")!r}).write_text(name)
    if not {capable!r}:
        raise ValueError("CPU capability is missing")
    return {{"environment": name, "CPU_contract": True}}
real._test_environments.preflight = package_preflight
{extra_conftest}
"""
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(test_path),
            "-c",
            str(directory / "pytest.ini"),
            "--rootdir",
            str(directory),
            "--platform",
            "a2a3",
            "--test-environment",
            profile,
            "--junitxml",
            str(directory / "results.xml"),
            "--confcutdir",
            str(directory),
            "-q",
            *args,
        ],
        cwd=_ROOT,
        env={
            **{key: value for key, value in os.environ.items() if key not in {"GIT_DIR", "GIT_WORK_TREE"}},
            "PYTHONPATH": os.pathsep.join((str(_ROOT), str(_ROOT / "python"))),
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        },
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    (directory / "pytest.stdout").write_text(result.stdout)
    (directory / "pytest.stderr").write_text(result.stderr)
    return result, directory / "results.plan.json", directory / "results.xml"


def _success(result):
    assert result.returncode == 0, result.stdout + result.stderr


def test_actual_collection_plan_and_junit_follow_added_cases_and_changed_paths(tmp_path):
    for count, path in ((4, "tests/first/test_one.py"), (5, "moved/elsewhere/test_two.py")):
        result, plan_path, xml_path = _run_pytest(tmp_path / str(count), _DIRECT.format(count=count), path=path)
        _success(result)
        plan = json.loads(plan_path.read_text())
        assert len(plan["cases"]) == count
        assert all(case["nodeid"].startswith(path + "::") for case in plan["cases"])
        test_environments.verify_results(plan_path, xml_path)
        assert "TEST_ENVIRONMENT_VERIFIED" in result.stdout


_FILTERS = """
import pytest

def test_default():
    pass

@pytest.mark.test_environment("torch_npu")
@pytest.mark.platforms(["a2a3"])
@pytest.mark.selected_tag
def test_framework_keep():
    pass

@pytest.mark.test_environment("torch_npu")
@pytest.mark.manual
def test_framework_manual():
    pass

@pytest.mark.test_environment("torch_npu")
@pytest.mark.platforms(["a5"])
def test_framework_other_platform():
    pass
"""


@pytest.mark.parametrize(
    ("profile", "args", "expected"),
    [
        ("default", (), {"test_default"}),
        ("torch_npu", (), {"test_framework_keep"}),
        ("torch_npu", ("--manual", "only"), {"test_framework_manual"}),
        ("torch_npu", ("--manual", "include", "-m", "selected_tag", "-k", "keep"), {"test_framework_keep"}),
    ],
)
def test_real_filters_are_orthogonal(tmp_path, profile, args, expected):
    result, plan_path, xml_path = _run_pytest(tmp_path, _FILTERS, *args, profile=profile)
    _success(result)
    assert {case.get("name") for case in ET.parse(xml_path).getroot().iter("testcase")} == expected
    assert plan_path.exists() == (profile != "default")


def test_collect_only_does_not_probe_dependencies_or_freeze_a_plan(tmp_path):
    result, plan, _ = _run_pytest(tmp_path, _DIRECT.format(count=2), "--collect-only", capable=False)
    _success(result)
    assert not plan.exists()
    assert not (tmp_path / "preflight-called").exists()


def test_selected_missing_capability_fails_before_test_execution(tmp_path):
    source = _DIRECT.format(count=1).replace("assert index >= 0", f'open({str(tmp_path / "body-ran")!r}, "w").close()')
    result, plan, _ = _run_pytest(tmp_path, source, capable=False)
    assert result.returncode != 0
    assert "CPU capability is missing" in result.stdout + result.stderr
    assert not plan.exists()
    assert not (tmp_path / "body-ran").exists()


def test_runtime_skip_is_rejected_after_the_real_junit_plugin_writes_results(tmp_path):
    source = _DIRECT.format(count=1).replace("assert index >= 0", 'pytest.skip("runtime dependency vanished")')
    result, plan, xml = _run_pytest(tmp_path, source)
    assert result.returncode == 1
    assert plan.exists()
    assert next(ET.parse(xml).getroot().iter("testcase")).find("skipped") is not None
    assert "TEST_ENVIRONMENT_VERIFICATION_FAILED" in result.stderr


@pytest.mark.parametrize(
    "declaration",
    [
        '@pytest.mark.test_environment("unknown")',
        '@pytest.mark.test_environment("default")\n@pytest.mark.test_environment("torch_npu")',
        "@pytest.mark.test_environment()",
    ],
)
def test_invalid_declarations_cannot_hide_behind_default_or_k_filter(tmp_path, declaration):
    source = f"import pytest\n{declaration}\ndef test_hidden():\n    pass\n"
    result, _, _ = _run_pytest(tmp_path, source, "-k", "not hidden", profile="default")
    assert result.returncode != 0
    assert "test environment" in result.stdout + result.stderr or "test_environment" in result.stdout + result.stderr


@pytest.fixture(scope="module")
def passing_receipt(tmp_path_factory):
    result, plan, xml = _run_pytest(tmp_path_factory.mktemp("environment_receipt"), _DIRECT.format(count=2))
    _success(result)
    test_environments.verify_results(plan, xml)
    return plan, xml


@pytest.mark.parametrize(
    "change",
    [
        "empty",
        "missing",
        "extra",
        "duplicate",
        "failure",
        "error",
        "skipped",
        "source",
        "environment",
        "classname",
    ],
)
def test_real_verifier_rejects_corrupted_actual_junit(tmp_path, passing_receipt, change):
    plan, xml = passing_receipt
    tree = ET.parse(xml)
    suite = next(tree.getroot().iter("testsuite"))
    cases = list(suite.iter("testcase"))
    if change == "empty":
        for case in cases:
            suite.remove(case)
    elif change == "missing":
        suite.remove(cases[0])
    elif change in ("extra", "duplicate"):
        other = copy.deepcopy(cases[0])
        if change == "extra":
            properties = other.find("properties")
            assert properties is not None
            next(p for p in properties if p.get("name") == "simpler_nodeid").set("value", "foreign::test_case")
        suite.append(other)
    elif change in ("failure", "error", "skipped"):
        ET.SubElement(cases[0], change)
    elif change in ("source", "environment"):
        properties = cases[0].find("properties")
        assert properties is not None
        next(p for p in properties if p.get("name") == "simpler_" + change).set("value", "wrong")
    else:
        cases[0].set("classname", "wrong.module")
    mutated = tmp_path / "mutated.xml"
    tree.write(mutated)
    with pytest.raises(ValueError):
        test_environments.verify_results(plan, mutated)


@pytest.mark.parametrize("args", [("--resource-child",), ("--level", "2")])
def test_strict_environment_rejects_unsupported_dispatch_modes(tmp_path, args):
    source = _DIRECT.format(count=1)
    if args == ("--level", "2"):
        source = """
import pytest
@pytest.mark.test_environment("torch_npu")
class TestScene:
    _st_level = 2
    _st_runtime = "host_build_graph"
    CASES = [{"name": "a", "platforms": ["a2a3"]}]
    def test_run(self):
        raise AssertionError("unsupported dispatcher executed")
"""
    result, plan, _ = _run_pytest(tmp_path, source, *args)
    assert result.returncode != 0
    assert not plan.exists()
    assert "direct pytest" in result.stdout + result.stderr or "dispatcher" in result.stdout + result.stderr


def test_device_count_dispatch_is_refused_in_strict_environment(tmp_path):
    source = _DIRECT.format(count=1).replace(
        "@pytest.mark.test_environment", "@pytest.mark.device_count(1)\n@pytest.mark.test_environment"
    )
    result, plan, _ = _run_pytest(tmp_path, source)
    assert result.returncode != 0
    assert "dispatcher" in result.stdout + result.stderr
    assert not plan.exists()


def test_resource_and_l2_commands_keep_environment_selection():
    spec = importlib.util.spec_from_file_location("environment_command_conftest", _ROOT / "conftest.py")
    assert spec is not None and spec.loader is not None
    root_conftest = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = root_conftest
    spec.loader.exec_module(root_conftest)

    class Config:
        invocation_params = SimpleNamespace(args=("tests/st", "--test-environment", "default"))

        def getoption(self, name, default=None):
            return {"--test-environment": "default"}.get(name, default)

    job = SimpleNamespace(nodeid="test_example.py::test_one", runtime="host_build_graph", kind="standalone")
    command = root_conftest._resource_child_command(job, [0], "a2a3", "exclude", Config())
    assert command[command.index("--test-environment") + 1] == "default"
    session = SimpleNamespace(config=Config())
    l2 = root_conftest._base_pytest_argv(session) + ["--runtime", "host_build_graph", "--level", "2"]
    assert l2[l2.index("--test-environment") + 1] == "default"


@pytest.mark.parametrize("mode", ["empty", "xdist"])
def test_strict_session_refuses_empty_or_parallel_execution(tmp_path, mode):
    extra = ""
    args = ()
    if mode == "empty":
        args = ("-k", "not direct")
    else:
        extra = """
original_configure = pytest_configure
def pytest_configure(config):
    original_configure(config)
    config.option.numprocesses = 2
"""
    result, plan, _ = _run_pytest(tmp_path, _DIRECT.format(count=1), *args, extra_conftest=extra)
    assert result.returncode != 0
    assert not plan.exists()
    message = "nonempty successful collection" if mode == "empty" else "direct pytest execution only"
    assert message in result.stdout + result.stderr


def test_precompile_uses_the_same_final_environment_selection(tmp_path):
    source = """
from pathlib import Path
import pytest

class TestDefault:
    _st_level = 2
    _st_runtime = "host_build_graph"
    CASES = [{"platforms": ["a2a3"]}]
    @staticmethod
    def compile_chip_callable(platform):
        Path("compiled-default").touch()
    def test_run(self):
        pass

@pytest.mark.test_environment("torch_npu")
class TestProfile(TestDefault):
    @staticmethod
    def compile_chip_callable(platform):
        assert platform == "a2a3"
        Path("compiled-profile").touch()
"""
    source = source.replace('Path("compiled-default")', f"Path({str(tmp_path / 'compiled-default')!r})")
    source = source.replace('Path("compiled-profile")', f"Path({str(tmp_path / 'compiled-profile')!r})")
    extra = """
import pytest
from simpler_setup.tools.scene_test_compile import compile_collected_scene_tests

class CompileSelected:
    def pytest_collection_finish(self, session):
        count, failures = compile_collected_scene_tests(session.items, "a2a3", max_workers=1)
        assert count == 1 and not failures

original_configure = pytest_configure
def pytest_configure(config):
    original_configure(config)
    config.pluginmanager.register(CompileSelected())
"""
    result, plan, _ = _run_pytest(tmp_path, source, "--collect-only", capable=False, extra_conftest=extra)
    _success(result)
    assert (tmp_path / "compiled-profile").exists()
    assert not (tmp_path / "compiled-default").exists()
    assert not (tmp_path / "preflight-called").exists()
    assert not plan.exists()


def test_actual_preflight_checks_metadata_without_importing_backends(tmp_path, monkeypatch):
    requirements = test_environments.requirements("torch_npu")
    home = tmp_path / "cann"
    home.mkdir()
    (home / "ascend_toolkit_install.info").write_text("version=9.0.0\n")
    monkeypatch.setenv("ASCEND_HOME_PATH", str(home))
    state: dict[str, object] = {"missing": None}
    versions = dict(requirements)

    def find_spec(name):
        assert name in requirements
        return None if state["missing"] == name else SimpleNamespace(origin="/cpu-contract/" + name)

    real_import = builtins.__import__

    def without_framework(name, *args, **kwargs):
        assert name.split(".")[0] not in {"torch", "torch_npu"}
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_framework)
    monkeypatch.setattr(test_environments.importlib.util, "find_spec", find_spec)
    monkeypatch.setattr(test_environments.importlib.metadata, "version", versions.__getitem__)
    record = test_environments.preflight("torch_npu")
    assert {name: value["version"] for name, value in record["packages"].items()} == requirements
    for missing in ("torch", "torch_npu"):
        state["missing"] = missing
        with pytest.raises(ValueError, match="module is missing"):
            test_environments.preflight("torch_npu")
    state["missing"] = None
    versions["torch_npu"] = "wrong-version"
    with pytest.raises(ValueError, match="found wrong-version"):
        test_environments.preflight("torch_npu")


def _recording_direct_source(directory):
    return (
        _DIRECT.format(count=2)
        .replace("import pytest", "import os\nfrom pathlib import Path\nimport pytest")
        .replace(
            "assert index >= 0",
            f'Path({str(directory)!r}, "body-" + str(index)).write_text('
            'os.environ.get("PYTEST_XDIST_WORKER", "controller"))',
        )
    )


@pytest.mark.parametrize("arguments", [("-n", "2"), ("--dist", "load", "--tx", "2*popen")])
def test_real_xdist_controller_refuses_strict_environment_before_bodies(tmp_path, arguments):
    result, plan, _ = _run_pytest(
        tmp_path, _recording_direct_source(tmp_path), "-p", "xdist.plugin", *arguments, "--max-worker-restart=0"
    )
    assert result.returncode != 0, result.stdout + result.stderr
    assert not list(tmp_path.glob("body-*"))
    assert not plan.exists()
    assert "direct pytest execution only" in result.stdout + result.stderr


def test_real_xdist_n_zero_preserves_direct_collection_and_receipts(tmp_path):
    result, plan, xml = _run_pytest(tmp_path, _recording_direct_source(tmp_path), "-p", "xdist.plugin", "-n", "0")
    _success(result)
    assert {path.name: path.read_text() for path in tmp_path.glob("body-*")} == {
        "body-0": "controller",
        "body-1": "controller",
    }
    test_environments.verify_results(plan, xml)


def test_real_xdist_workerinput_is_rejected_even_after_worker_options_are_reset(tmp_path):
    # A default controller reaches a real xdist worker. The worker selects the
    # strict profile before the production configure hook; workerinput is real.
    extra = f"""
import json
original_configure = pytest_configure
def pytest_configure(config):
    if hasattr(config, "workerinput"):
        record = {{"workerinput": config.workerinput,
                   "numprocesses": config.getoption("numprocesses", default=None),
                   "dist": config.getoption("dist", default=None)}}
        Path({str(tmp_path)!r}, "worker-" + config.workerinput["workerid"] + ".json").write_text(json.dumps(record))
        config.option.test_environment = "torch_npu"
    original_configure(config)
"""
    result, _, _ = _run_pytest(
        tmp_path,
        _recording_direct_source(tmp_path),
        "-p",
        "xdist.plugin",
        "-n",
        "1",
        "--max-worker-restart=0",
        profile="default",
        extra_conftest=extra,
    )
    records = [json.loads(path.read_text()) for path in tmp_path.glob("worker-*.json")]
    assert records and records[0]["workerinput"]["workerid"] == "gw0"
    assert records[0]["numprocesses"] is None
    assert result.returncode != 0, result.stdout + result.stderr
    assert not list(tmp_path.glob("body-*"))
    assert "direct pytest execution only" in result.stdout + result.stderr


def _installed_toolchain(home, version, *, nested=False, script=True):
    manifest_home = home / f"{os.uname().machine}-linux" if nested else home
    manifest_home.mkdir(parents=True)
    (manifest_home / "ascend_toolkit_install.info").write_text(f"version={version}\n")
    if script:
        (home / "set_env.sh").write_text("# Selection tests never execute this script.\n")
    return home / "set_env.sh"


def test_toolchain_selection_uses_installed_manifest_and_deduplicates_aliases(tmp_path, monkeypatch, capsys):
    root = tmp_path / "Ascend"
    active = root / "cann-beta"
    _installed_toolchain(active, "9.2.0-beta.1")
    _installed_toolchain(root / "ascend-toolkit" / "9.0.0", "9.2.0-beta.1")
    exact = root / "ascend-toolkit" / "release-without-version-in-name"
    expected = _installed_toolchain(exact, "9.0.0", nested=True)
    (root / "ascend-toolkit" / "latest").symlink_to(exact, target_is_directory=True)
    monkeypatch.setattr(test_environments, "_ASCEND_ROOT", root)
    monkeypatch.setenv("ASCEND_HOME_PATH", str(active))

    assert test_environments.select_toolchain("torch_npu") == expected
    inventory = json.loads(capsys.readouterr().out.removeprefix("TEST_ENVIRONMENT_TOOLCHAINS "))
    assert sum(entry["home"] == str(exact) for entry in inventory["installed"]) == 1


def test_toolchain_selection_preserves_an_exact_active_toolkit(tmp_path, monkeypatch):
    root = tmp_path / "Ascend"
    _installed_toolchain(root / "cann-first", "9.0.0")
    active = tmp_path / "active-toolkit"
    expected = _installed_toolchain(active, "9.0.0")
    monkeypatch.setattr(test_environments, "_ASCEND_ROOT", root)
    monkeypatch.setenv("ASCEND_HOME_PATH", str(active))

    assert test_environments.select_toolchain("torch_npu") == expected


@pytest.mark.parametrize("version,script", [("9.2.0-beta.1", True), ("9.0.0", False)])
def test_toolchain_selection_refuses_incompatible_or_incomplete_installation(tmp_path, monkeypatch, version, script):
    root = tmp_path / "Ascend"
    active = root / "ascend-toolkit" / "9.0.0"
    _installed_toolchain(active, version, script=script)
    monkeypatch.setattr(test_environments, "_ASCEND_ROOT", root)
    monkeypatch.setenv("ASCEND_HOME_PATH", str(active))

    with pytest.raises(ValueError, match=r"No installed CANN 9\.0\.0 with set_env\.sh"):
        test_environments.select_toolchain("torch_npu")
