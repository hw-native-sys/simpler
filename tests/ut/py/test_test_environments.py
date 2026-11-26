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
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import pytest

from simpler_setup import test_environments

_ROOT = Path(__file__).resolve().parents[3]
_HOOKS = (
    "pytest_addoption",
    "pytest_configure",
    "pytest_sessionstart",
    "pytest_itemcollected",
    "pytest_collection_modifyitems",
    "pytest_collection_finish",
    "pytest_runtestloop",
    "pytest_runtest_makereport",
    "pytest_sessionfinish",
)
_DIRECT = """
import pytest

@pytest.mark.test_environment("torch_npu")
@pytest.mark.parametrize("index", range({count}))
def test_direct(index):
    assert index >= 0
"""


def _pytest_subprocess_env():
    """Start an independent controller without inheriting an outer xdist role."""
    return {
        **{
            key: value
            for key, value in os.environ.items()
            if key not in {"GIT_DIR", "GIT_WORK_TREE"} and not key.startswith("PYTEST_XDIST_")
        },
        "PYTHONPATH": os.pathsep.join((str(_ROOT), str(_ROOT / "python"))),
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    }


def _run_pytest(
    directory,
    source,
    *args,
    profile="torch_npu",
    path="tests/new_place/test_direct.py",
    capable=True,
    extra_conftest="",
    input_text=None,
    path_after_separator=False,
    ini_options="",
):
    directory.mkdir(parents=True, exist_ok=True)
    if path_after_separator:
        git_dir = subprocess.check_output(
            ["git", "-C", str(_ROOT), "rev-parse", "--absolute-git-dir"], text=True
        ).strip()
        (directory / ".git").write_text(f"gitdir: {git_dir}\n")
    test_path = directory / path
    test_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.write_text(source)
    (directory / "pytest.ini").write_text(
        "[pytest]\naddopts = --import-mode=importlib\nmarkers =\n    selected_tag\n" + ini_options
    )
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
            *([] if path_after_separator else [str(test_path)]),
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
            *(["--", str(test_path)] if path_after_separator else []),
        ],
        cwd=directory if path_after_separator else _ROOT,
        env=_pytest_subprocess_env(),
        capture_output=True,
        text=True,
        input=input_text,
        timeout=60,
        check=False,
    )
    (directory / "pytest.stdout").write_text(result.stdout)
    (directory / "pytest.stderr").write_text(result.stderr)
    return result, directory / "results.plan.json", directory / "results.xml"


def _success(result):
    assert result.returncode == 0, result.stdout + result.stderr


def test_direct_parallel_shards_use_distinct_devices_and_complete_receipts(tmp_path):
    source = """
import os
from pathlib import Path
import pytest

@pytest.mark.test_environment("torch_npu")
@pytest.mark.parametrize("index", range(8))
def test_direct(index, request):
    device = request.config.getoption("--device")
    Path(__file__).with_name(f"case-{index}").write_text(f"{device}:{os.getpid()}")
"""
    result, plan, xml = _run_pytest(
        tmp_path, source, "--device", "1,3,5,7", "--max-parallel", "4", "--environment-receipts"
    )
    _success(result)
    records = [path.read_text().split(":") for path in tmp_path.rglob("case-*")]
    assert len(records) == 8
    assert {device for device, _ in records} == {"1", "3", "5", "7"}
    assert len({pid for _, pid in records}) == 4
    test_environments.verify_results(plan, xml)


def test_direct_parallel_preserves_child_failure_and_all_results(tmp_path):
    source = _DIRECT.format(count=8).replace("assert index >= 0", "assert index != 3")
    result, plan, xml = _run_pytest(
        tmp_path, source, "--device", "1,3,5,7", "--max-parallel", "4", "--environment-receipts"
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert len(list(ET.parse(xml).getroot().iter("testcase"))) == 8
    with pytest.raises(ValueError, match="failure"):
        test_environments.verify_results(plan, xml)


def test_direct_parallel_lastfailed_keeps_all_shards_and_drives_rerun(tmp_path, monkeypatch):
    source = """
import os
import pytest

@pytest.mark.test_environment("torch_npu")
@pytest.mark.parametrize("index", range(4))
def test_direct(index):
    if os.environ.get("FRAMEWORK_CACHE_PASS") != "1":
        assert index >= 2
"""
    cache = tmp_path / ".pytest_cache/v/cache/lastfailed"
    cache.parent.mkdir(parents=True)
    unrelated = {"other.py::test_old": True}
    cache.write_text(json.dumps(unrelated))
    result, _, _ = _run_pytest(tmp_path, source, "--device=1,3", "--max-parallel=2")
    assert result.returncode == 1, result.stdout + result.stderr
    failed = {f"tests/new_place/test_direct.py::test_direct[{index}]": True for index in (0, 1)}
    assert json.loads(cache.read_text()) == {**unrelated, **failed}
    monkeypatch.setenv("FRAMEWORK_CACHE_PASS", "1")
    rerun, plan, xml = _run_pytest(tmp_path, source, "--device=1,3", "--max-parallel=2", "--lf")
    _success(rerun)
    assert {case["nodeid"] for case in json.loads(plan.read_text())["cases"]} == set(failed)
    test_environments.verify_results(plan, xml)
    assert json.loads(cache.read_text()) == unrelated


@pytest.mark.parametrize("capture", ["fd", "no", "tee-sys"])
def test_direct_parallel_preserves_visible_success_output(tmp_path, capture):
    source = _DIRECT.format(count=4).replace("assert index >= 0", "print(f'case-output-{index}', flush=True)")
    result, plan, xml = _run_pytest(tmp_path, source, "--device=1,3", "--max-parallel=2", f"--capture={capture}")
    _success(result)
    assert "[scheduler] START" in result.stdout
    for index in range(4):
        assert result.stdout.count(f"case-output-{index}") == (0 if capture == "fd" else 1)
    test_environments.verify_results(plan, xml)


@pytest.mark.parametrize("configuration", ["cli", "ini"])
def test_direct_file_logging_runs_in_controller_and_preserves_all_records(tmp_path, configuration):
    source = """
import logging
import time
from pathlib import Path
import pytest

@pytest.mark.test_environment("torch_npu")
@pytest.mark.parametrize("index", range(2))
def test_direct(index, request):
    root = Path(__file__).parent
    if request.config.getoption("--environment-parent-plan"):
        ready = root / f"ready-{index}"
        ready.touch()
        other = root / f"ready-{1-index}"
        deadline = time.monotonic() + 10
        while not other.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert other.exists()
        if index == 1:
            written = root / "written-0"
            while not written.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert written.exists()
    logging.getLogger("exclusive-file-records").info("FILE_RECORD_%s", index)
    if index == 0:
        (root / "written-0").touch()
"""
    output = tmp_path / "run.log"
    flags = ["--log-file", str(output), "--log-file-level=INFO"] if configuration == "cli" else []
    ini = f"log_file = {output}\nlog_file_level = INFO\n" if configuration == "ini" else ""
    result, plan, xml = _run_pytest(tmp_path, source, "--device=1,3", "--max-parallel=2", *flags, ini_options=ini)
    _success(result)
    content = output.read_text()
    for index in range(2):
        assert f"FILE_RECORD_{index}" in content, content
    assert "[scheduler] START" not in result.stdout
    test_environments.verify_results(plan, xml)


@pytest.mark.parametrize("flags", [("--log-file", os.devnull), ("-p", "no:logging")])
def test_direct_without_file_logging_keeps_parallel_execution(tmp_path, flags):
    result, plan, xml = _run_pytest(tmp_path, _DIRECT.format(count=2), "--device=1,3", *flags)
    _success(result)
    assert result.stdout.count("[scheduler] START") == 2
    test_environments.verify_results(plan, xml)


def test_direct_callback_failure_preserves_completed_results(tmp_path):
    extra = """
original_print = builtins.print
def fail_completion(*args, **kwargs):
    if args and str(args[0]).startswith("[framework]"):
        raise OSError("completion output failure")
    return original_print(*args, **kwargs)
builtins.print = fail_completion
"""
    result, _, xml = _run_pytest(
        tmp_path, _DIRECT.format(count=4), "--device=1,3", "--max-parallel=2", extra_conftest=extra
    )
    assert result.returncode != 0
    assert "completion output failure" in result.stdout + result.stderr
    cases = list(ET.parse(xml).getroot().iter("testcase"))
    assert 2 <= len(cases) <= 4
    assert all(case.find("failure") is None and case.find("error") is None for case in cases)


@pytest.mark.skipif(os.name != "posix", reason="POSIX signal cleanup")
@pytest.mark.parametrize("stop", ["callback", "SIGTERM", "SIGALRM"])
def test_direct_interrupted_shards_preserve_already_produced_logs(tmp_path, stop):
    source = """
import os
import signal
import time
from pathlib import Path
import pytest

@pytest.mark.test_environment("torch_npu")
@pytest.mark.parametrize("index", range(2))
def test_direct(index):
    root = Path(__file__).parent
    print(f"SHARD_DIAGNOSTIC_{index}", flush=True)
    if index == 1:
        (root / "slow-ready").touch()
        signal.pause()
    else:
        ready = root / "slow-ready"
        deadline = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists()
        if STOP != "callback":
            os.kill(os.getppid(), getattr(signal, STOP))
            signal.pause()
""".replace("STOP", repr(stop))
    extra = (
        """
original_print = builtins.print
def fail_completion(*args, **kwargs):
    if args and str(args[0]).startswith("[framework]"):
        raise OSError("completion output failure")
    return original_print(*args, **kwargs)
builtins.print = fail_completion
"""
        if stop == "callback"
        else ""
    )
    result, _, _ = _run_pytest(
        tmp_path,
        source,
        "--device=1,3",
        "--max-parallel=2",
        "-s",
        "--pto-session-timeout=60",
        extra_conftest=extra,
    )
    assert result.returncode != 0, result.stdout + result.stderr
    if stop == "callback":
        assert "completion output failure" in result.stdout + result.stderr
    else:
        expected = 124 if stop == "SIGALRM" else 128 + getattr(signal, stop)
        assert result.returncode == expected, result.stdout + result.stderr
    for index in range(2):
        log = next(tmp_path.glob(f"results.plan-shards-*/shard-{index}/pytest.log"), None)
        assert log is not None, result.stdout + result.stderr
        assert f"SHARD_DIAGNOSTIC_{index}" in log.read_text()


@pytest.mark.parametrize("parallel", ["1", "2"])
def test_direct_fanout_preserves_selection_and_execution_options(tmp_path, parallel):
    source = """
import os
from pathlib import Path
import pytest

@pytest.mark.test_environment("torch_npu")
@pytest.mark.parametrize("index", range(8))
def test_direct(index, request):
    assert request.config.getoption("--rounds") == 3
    Path(__file__).with_name(f"case-{index}").write_text(str(os.getpid()))
"""
    result, plan, xml = _run_pytest(
        tmp_path,
        source,
        "--device=1,3,5,7",
        f"--max-parallel={parallel}",
        "--rounds=3",
        "-k",
        "not 7",
        "--environment-receipts",
    )
    _success(result)
    records = list(tmp_path.rglob("case-*"))
    assert len(records) == 7
    assert not list(tmp_path.rglob("case-7"))
    assert len({path.read_text() for path in records}) == int(parallel)
    test_environments.verify_results(plan, xml)


@pytest.mark.parametrize("limit", ["0", "invalid"])
def test_direct_parallel_rejects_invalid_limits(tmp_path, limit):
    result, _, _ = _run_pytest(tmp_path, _DIRECT.format(count=4), "--device", "1,3", "--max-parallel", limit)
    assert result.returncode == 4, result.stdout + result.stderr
    assert "--max-parallel" in result.stdout + result.stderr


@pytest.mark.parametrize("warning_error", [False, True])
@pytest.mark.parametrize("path", ["tests/new_place/test_direct.py", "--device=literal.py"])
def test_direct_parallel_keeps_options_before_path_separator(tmp_path, warning_error, path):
    source = _DIRECT.format(count=4)
    if warning_error:
        source = source.replace("assert index >= 0", "__import__('warnings').warn('must fail')")
    result, plan, xml = _run_pytest(
        tmp_path,
        source,
        "--device=1,3",
        "--max-parallel=2",
        "-k",
        "not 3",
        "-W",
        "error::UserWarning",
        "--environment-receipts",
        path_after_separator=True,
        path=path,
    )
    assert result.returncode == (1 if warning_error else 0), result.stdout + result.stderr
    assert "[scheduler] START" in result.stdout
    cases = list(ET.parse(xml).getroot().iter("testcase"))
    assert len(cases) == 3
    assert all("[3]" not in case.get("name") for case in cases)
    if warning_error:
        assert all(case.find("failure") is not None for case in cases)
    else:
        test_environments.verify_results(plan, xml)


def test_path_separator_contract_accepts_relative_worktree_metadata(tmp_path, monkeypatch):
    alias = tmp_path / "linked-checkout"
    alias.mkdir()
    git_dir = subprocess.check_output(["git", "-C", str(_ROOT), "rev-parse", "--absolute-git-dir"], text=True).strip()
    (alias / ".git").write_text(f"gitdir: {os.path.relpath(git_dir, alias)}\n")
    for name in ("conftest.py", "python", "simpler_setup"):
        (alias / name).symlink_to(_ROOT / name, target_is_directory=(_ROOT / name).is_dir())
    monkeypatch.setattr(sys.modules[__name__], "_ROOT", alias)
    result, plan, xml = _run_pytest(
        tmp_path / "run", _DIRECT.format(count=2), "--device=1,3", path_after_separator=True
    )
    _success(result)
    test_environments.verify_results(plan, xml)


@pytest.mark.parametrize(
    ("flags", "failures"),
    [
        (("-x",), 1),
        (("--maxfail=1",), 1),
        (("--maxfail", "2"), 2),
        (("--maxfail=0",), 6),
        (("--stepwise",), 1),
        (("--stepwise-skip",), 2),
        (("--stepwise-reset",), 1),
    ],
)
def test_direct_failure_limit_applies_to_the_whole_run(tmp_path, flags, failures):
    source = """
from pathlib import Path
import pytest

@pytest.mark.test_environment("torch_npu")
@pytest.mark.parametrize("index", range(6))
def test_direct(index):
    Path(__file__).with_name(f"executed-{index}").touch()
    assert index < 0
"""
    result, _, xml = _run_pytest(tmp_path, source, "--device=1,3", "--max-parallel=2", *flags)
    assert result.returncode == 1, result.stdout + result.stderr
    executed = {path.name for path in tmp_path.rglob("executed-*")}
    assert executed == {f"executed-{index}" for index in range(failures)}, result.stdout + result.stderr
    assert len(list(ET.parse(xml).getroot().iter("failure"))) == failures
    assert ("[scheduler] START" in result.stdout) == (failures == 6)


@pytest.mark.parametrize("debugger", ["--pdb", "--trace"])
@pytest.mark.parametrize("devices", ["1", "1,3"])
def test_direct_interactive_debugger_runs_in_controller(tmp_path, debugger, devices):
    source = _DIRECT.format(count=2)
    if debugger == "--pdb":
        source = source.replace("assert index >= 0", "assert index < 0")
    extra = f"""
import os

def pytest_sessionstart(session):
    if not session.config.getoption("--environment-parent-plan", default=None):
        Path({str(tmp_path / "controller-pid")!r}).write_text(str(os.getpid()))

def pytest_enter_pdb(config, pdb):
    with Path({str(tmp_path)!r}, f"debugger-{{os.getpid()}}").open("a") as output:
        output.write("enter\\n")
"""
    result, plan, xml = _run_pytest(
        tmp_path,
        source,
        "--device",
        devices,
        "--max-parallel",
        "2",
        debugger,
        extra_conftest=extra,
        input_text="continue\ncontinue\n",
    )
    assert result.returncode == (1 if debugger == "--pdb" else 0), result.stdout + result.stderr
    pid = (tmp_path / "controller-pid").read_text()
    records = list(tmp_path.glob("debugger-*"))
    assert [path.name for path in records] == [f"debugger-{pid}"]
    assert records[0].read_text().splitlines() == ["enter", "enter"]
    assert result.stdout.count("(Pdb)") >= 2
    assert "[scheduler] START" not in result.stdout
    cases = list(ET.parse(xml).getroot().iter("testcase"))
    assert len(cases) == 2
    if debugger == "--pdb":
        assert all(case.find("failure") is not None for case in cases)
    else:
        test_environments.verify_results(plan, xml)


def test_unknown_environment_with_junit_remains_usage_error(tmp_path):
    result, plan, xml = _run_pytest(tmp_path, _DIRECT.format(count=1), profile="unknown")
    assert result.returncode == 4, result.stdout + result.stderr
    assert "Unknown test environment 'unknown'" in result.stderr
    assert "INTERNALERROR" not in result.stdout + result.stderr
    assert not plan.exists()
    assert not xml.exists()
    assert not list(tmp_path.glob("*.lock"))
    assert not (tmp_path / "preflight-called").exists()


@pytest.mark.parametrize("parallel", ["1", "2"])
def test_junit_path_expands_environment_variables_exactly_once(tmp_path, monkeypatch, parallel):
    monkeypatch.setenv("FRAMEWORK_REPORT", "$FRAMEWORK_TAG")
    monkeypatch.setenv("FRAMEWORK_TAG", "other")
    result, _, _ = _run_pytest(
        tmp_path,
        _DIRECT.format(count=4),
        "--device=1,3",
        f"--max-parallel={parallel}",
        "--junitxml",
        str(tmp_path / "$FRAMEWORK_REPORT.xml"),
    )
    _success(result)
    xml = tmp_path / "$FRAMEWORK_TAG.xml"
    plan = xml.with_suffix(".plan.json")
    assert not (tmp_path / "other.xml").exists()
    assert xml.with_name(xml.name + ".lock").exists()
    test_environments.verify_results(plan, xml)


@pytest.mark.parametrize("receipts", [False, True])
def test_direct_parallel_keeps_parent_receipt_mode(tmp_path, receipts):
    flags = ["--environment-receipts"] if receipts else []
    result, plan, xml = _run_pytest(tmp_path, _DIRECT.format(count=4), "--device", "1,3", "--max-parallel", "2", *flags)
    _success(result)
    test_environments.verify_results(plan, xml)


@pytest.mark.parametrize("parallel", ["1", "2"])
def test_direct_parallel_keeps_warning_errors(tmp_path, parallel):
    source = _DIRECT.format(count=4).replace("assert index >= 0", "__import__('warnings').warn('must fail')")
    result, _, xml = _run_pytest(
        tmp_path, source, "--device", "1,3", "--max-parallel", parallel, "-W", "error::UserWarning"
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert len(list(ET.parse(xml).getroot().iter("failure"))) == 4


@pytest.mark.parametrize("xml_state", ["missing", "malformed", "unknown-encoding"])
@pytest.mark.parametrize("returncode", [0, 9])
def test_direct_parallel_keeps_readable_results_after_shard_crash(tmp_path, xml_state, returncode):
    source = f"""
import os
from pathlib import Path
import pytest

@pytest.mark.test_environment("torch_npu")
@pytest.mark.parametrize("index", range(4))
def test_direct(index, request):
    if index == 0:
        if {xml_state!r} == "malformed":
            Path(request.config.getoption("xmlpath")).write_text("<broken>")
        elif {xml_state!r} == "unknown-encoding":
            Path(request.config.getoption("xmlpath")).write_text(
                '<?xml version="1.0" encoding="no-such-codec"?><testsuites/>'
            )
        os._exit({returncode})
"""
    result, _, xml = _run_pytest(tmp_path, source, "--device", "1,3", "--max-parallel", "2")
    assert result.returncode == 1, result.stdout + result.stderr
    assert len(list(ET.parse(xml).getroot().iter("testcase"))) == 2
    assert "TEST_ENVIRONMENT_SHARD_UNREADABLE" in result.stderr
    assert "test_direct[0]" in result.stderr and "test_direct[2]" in result.stderr


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="Linux process-state inspection")
@pytest.mark.parametrize(("mode", "startup_delay"), [("timeout", 0), ("timeout", 5), ("terminate", 0)])
def test_direct_timeout_kills_shards_and_descendants(tmp_path, mode, startup_delay):
    source = """
import os
import subprocess
import sys
import time
from pathlib import Path
import pytest

@pytest.mark.test_environment("torch_npu")
@pytest.mark.parametrize("index", range(2))
def test_direct(index):
    time.sleep(STARTUP_DELAY)
    pidfile = str(Path(__file__).with_name(f"descendant-{index}"))
    program = ("import os,signal; from pathlib import Path; "
               "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
               "signal.signal(signal.SIGUSR1,signal.SIG_IGN); "
               f"marker=Path({pidfile!r}); pending=marker.with_name('.preparing-'+marker.name); "
               "pending.write_text(str(os.getpid())); pending.replace(marker); "
               "signal.pause()")
    subprocess.run([sys.executable, "-c", program], check=True)
""".replace("STARTUP_DELAY", str(startup_delay))
    extra = f"""
import os, signal, threading, time

def pytest_sessionstart(session):
    if session.config.getoption("--environment-parent-plan", default=None):
        return
    def interrupt_when_ready():
        end = time.monotonic() + 20
        while time.monotonic() < end:
            if len(list(Path({str(tmp_path)!r}).rglob("descendant-*"))) == 2:
                Path({str(tmp_path / "interrupt-ready")!r}).touch()
                if {mode!r} == "timeout":
                    signal.setitimer(signal.ITIMER_REAL, 0.1)
                else:
                    os.kill(os.getpid(), signal.SIGTERM)
                return
            time.sleep(0.01)
    threading.Thread(target=interrupt_when_ready, daemon=True).start()
"""
    try:
        result, _, _ = _run_pytest(
            tmp_path,
            source,
            "--device",
            "1,3",
            "--max-parallel",
            "2",
            "--pto-session-timeout",
            "30",
            extra_conftest=extra,
        )
        assert result.returncode == (124 if mode == "timeout" else 143), result.stdout + result.stderr
        assert (tmp_path / "interrupt-ready").exists(), result.stdout + result.stderr
        markers = list(tmp_path.rglob("descendant-*"))
        assert len(markers) == 2, result.stdout + result.stderr
        for marker in markers:
            stat = Path(f"/proc/{int(marker.read_text())}/stat")
            deadline = time.monotonic() + 5
            while True:
                try:
                    state = stat.read_text().rsplit(")", 1)[1].split()[0]
                except (FileNotFoundError, ProcessLookupError):
                    break
                if state in {"Z", "X", "x"}:
                    break
                assert time.monotonic() < deadline, f"{marker}: process is still {state}"
                time.sleep(0.01)
    finally:
        for marker in tmp_path.rglob("descendant-*"):
            try:
                os.kill(int(marker.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.parametrize(
    ("parallel", "collect_only", "kill_owner", "output_kind"),
    [
        (1, False, False, "same"),
        (2, False, False, "same"),
        (1, True, False, "same"),
        (2, True, False, "same"),
        (1, False, True, "same"),
        (2, False, False, "same-plan"),
        (2, False, False, "owner-plan"),
    ],
)
def test_same_result_output_is_exclusive_until_owner_exits(
    tmp_path, monkeypatch, parallel, collect_only, kill_owner, output_kind
):
    import socket  # noqa: PLC0415

    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw0")
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(2)
    listener.settimeout(15)
    source = f"""
import os
import socket
from pathlib import Path
import pytest

@pytest.mark.test_environment("torch_npu")
@pytest.mark.parametrize("index", range(2))
def test_direct(index):
    actor = os.environ["OUTPUT_LOCK_ACTOR"]
    Path(__file__).with_name(f"entered-{{actor}}-{{index}}").touch()
    if actor == "owner":
        with socket.create_connection({listener.getsockname()!r}, timeout=15) as connection:
            assert connection.recv(1) == b"g"
"""
    scaffold, plan, xml = _run_pytest(tmp_path, source, "--collect-only")
    _success(scaffold)
    command = list(scaffold.args)
    command.remove("--collect-only")
    command.extend(["--device", "1,3", "--max-parallel", str(parallel)])
    env = _pytest_subprocess_env()
    clients = []
    owner = subprocess.Popen(
        command,
        cwd=_ROOT,
        env={**env, "OUTPUT_LOCK_ACTOR": "owner"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        for _ in range(parallel):
            clients.append(listener.accept()[0])
        original_plan = plan.read_bytes()
        original_xml = xml.read_bytes()
        contender_command = list(command)
        if output_kind == "same-plan":
            contender_command[contender_command.index("--junitxml") + 1] = str(xml.with_suffix(".junit"))
        elif output_kind == "owner-plan":
            contender_command[contender_command.index("--junitxml") + 1] = str(plan)
        contender = subprocess.run(
            contender_command + (["--collect-only"] if collect_only else []),
            cwd=_ROOT,
            env={**env, "OUTPUT_LOCK_ACTOR": "contender"},
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        assert contender.returncode == 4, contender.stdout + contender.stderr
        assert "output already in use" in contender.stderr
        assert not list(tmp_path.rglob("entered-contender-*"))
        assert plan.read_bytes() == original_plan
        assert xml.read_bytes() == original_xml
        if output_kind == "same-plan":
            assert not xml.with_suffix(".junit").exists()
        assert owner.poll() is None
        if kill_owner:
            owner.kill()
        else:
            for client in clients:
                client.sendall(b"g")
            for _ in range(2 - parallel):
                client = listener.accept()[0]
                clients.append(client)
                client.sendall(b"g")
        output, _ = owner.communicate(timeout=20)
        assert owner.returncode == (-signal.SIGKILL if kill_owner else 0), output
        if not kill_owner:
            test_environments.verify_results(plan, xml)
        rerun = subprocess.run(
            command,
            cwd=_ROOT,
            env={**env, "OUTPUT_LOCK_ACTOR": "after"},
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        _success(rerun)
        assert len(list(tmp_path.rglob("entered-after-*"))) == 2
        test_environments.verify_results(plan, xml)
    finally:
        for client in clients:
            client.close()
        listener.close()
        if owner.poll() is None:
            owner.terminate()
        owner.communicate(timeout=10)


def test_parallel_runs_share_output_parent_without_sharing_shards(tmp_path, monkeypatch):
    import socket  # noqa: PLC0415
    import threading  # noqa: PLC0415

    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw0")
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(4)
    listener.settimeout(20)
    port = listener.getsockname()[1]
    source = f"""
import socket
import pytest

@pytest.mark.test_environment("torch_npu")
@pytest.mark.parametrize("name", ["alpha", "bravo", "charlie", "delta"])
def test_direct(name):
    with socket.create_connection(("127.0.0.1", {port}), timeout=15) as connection:
        assert connection.recv(1) == b"g"
"""
    result, _, _ = _run_pytest(tmp_path, source, "--collect-only")
    _success(result)
    errors = []

    def release_all():
        clients = []
        try:
            for _ in range(4):
                clients.append(listener.accept()[0])
            for client in clients:
                client.sendall(b"g")
        except OSError as error:
            errors.append(error)
        finally:
            for client in clients:
                client.close()
            listener.close()

    gate = threading.Thread(target=release_all, daemon=True)
    gate.start()
    children = []
    try:
        for label, expression in (("first", "alpha or bravo"), ("second", "charlie or delta")):
            command = [
                sys.executable,
                "-m",
                "pytest",
                str(tmp_path / "tests/new_place/test_direct.py"),
                "-c",
                str(tmp_path / "pytest.ini"),
                "--rootdir",
                str(tmp_path),
                "--confcutdir",
                str(tmp_path),
                "--platform",
                "a2a3",
                "--test-environment",
                "torch_npu",
                "--device",
                "1,3",
                "--max-parallel",
                "2",
                "-k",
                expression,
                "--junitxml",
                str(tmp_path / f"{label}.xml"),
            ]
            children.append(
                subprocess.Popen(
                    command,
                    cwd=_ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=_pytest_subprocess_env(),
                )
            )
        for child in children:
            output, _ = child.communicate(timeout=30)
            assert child.returncode == 0, output
        assert not errors
        for label in ("first", "second"):
            test_environments.verify_results(tmp_path / f"{label}.plan.json", tmp_path / f"{label}.xml")
        assert len(list(tmp_path.glob("*-shards-*"))) == 2
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait()
        listener.close()
        gate.join(timeout=2)


def test_actual_collection_plan_and_junit_follow_added_cases_and_changed_paths(tmp_path):
    for count, path in ((4, "tests/first/test_one.py"), (5, "moved/elsewhere/test_two.py")):
        result, plan_path, xml_path = _run_pytest(tmp_path / str(count), _DIRECT.format(count=count), path=path)
        _success(result)
        plan = json.loads(plan_path.read_text())
        assert len(plan["cases"]) == count
        assert all(case["nodeid"].startswith(path + "::") for case in plan["cases"])
        test_environments.verify_results(plan_path, xml_path)
        assert "TEST_ENVIRONMENT_VERIFIED" in result.stdout
        plan_lines = [line for line in result.stdout.splitlines() if line.startswith("TEST_ENVIRONMENT_PLAN ")]
        assert len(plan_lines) == 1 and len(plan_lines[0]) < 512


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


@pytest.mark.parametrize(
    ("platform", "expected"),
    [("a2a3", {"test_shared"}), ("a2a3sim", {"test_shared", "test_simulator_only"})],
)
def test_scene_method_platform_limits_the_class_cases(tmp_path, platform, expected):
    source = """
import pytest

class TestScene:
    CASES = [{"platforms": ["a2a3", "a2a3sim"]}]

    def test_shared(self):
        pass

    @pytest.mark.platforms(["a2a3sim"])
    def test_simulator_only(self):
        pass
"""
    result, _, xml_path = _run_pytest(tmp_path, source, "--platform", platform, profile="default")
    _success(result)
    assert {case.get("name") for case in ET.parse(xml_path).getroot().iter("testcase")} == expected


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
        "plan_identity",
    ],
)
def test_real_verifier_rejects_corrupted_actual_junit(tmp_path, passing_receipt, change):
    plan, xml = passing_receipt
    if change == "plan_identity":
        changed_plan = json.loads(plan.read_text())
        changed_plan["identity"]["cann"] = {"version": "unqualified"}
        mutated_plan = tmp_path / "mutated.plan.json"
        mutated_plan.write_text(json.dumps(changed_plan))
        with pytest.raises(ValueError, match="wrong source, environment or plan"):
            test_environments.verify_results(mutated_plan, xml)
        return
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
    monkeypatch.setattr(test_environments.platform, "machine", lambda: "aarch64")
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


def test_vllm_environment_registry_and_constraints():
    profile = test_environments.ENVIRONMENTS["vllm_torch_npu"]
    assert profile.execution == "direct"
    assert profile.cann_minimum == (9, 0)
    assert dict(profile.modules) == {
        "torch": "torch",
        "torch-npu": "torch_npu",
        "numpy": "numpy",
        "vllm": "vllm",
        "vllm-ascend": "vllm_ascend",
    }
    assert test_environments.requirements("vllm_torch_npu") == {
        "torch": "2.10.0+cpu",
        "torch-npu": "2.10.0",
        "vllm": "0.20.2",
        "vllm-ascend": "0.20.2rc1",
        "triton-ascend": "3.2.1",
        "numpy": "1.26.4",
    }


def test_vllm_preflight_checks_metadata_without_importing_backends(tmp_path, monkeypatch):
    monkeypatch.setattr(test_environments.platform, "machine", lambda: "aarch64")
    profile = test_environments.ENVIRONMENTS["vllm_torch_npu"]
    requirements = test_environments.requirements("vllm_torch_npu")
    home = tmp_path / "cann"
    home.mkdir()
    (home / "ascend_toolkit_install.info").write_text("version=9.0.0\n")
    monkeypatch.setenv("ASCEND_HOME_PATH", str(home))
    state: dict[str, object] = {"missing": None}
    versions = dict(requirements)
    modules = {module for _, module in profile.modules}

    def find_spec(name):
        assert name in modules
        return None if state["missing"] == name else SimpleNamespace(origin="/cpu-contract/" + name)

    real_import = builtins.__import__

    def without_backend(name, *args, **kwargs):
        assert name.split(".")[0] not in {"torch", "torch_npu", "vllm", "vllm_ascend"}
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_backend)
    monkeypatch.setattr(test_environments.importlib.util, "find_spec", find_spec)
    monkeypatch.setattr(test_environments.importlib.metadata, "version", versions.__getitem__)
    record = test_environments.preflight("vllm_torch_npu")
    expected = {name: versions[name] for name, _ in profile.modules}
    assert {name: value["version"] for name, value in record["packages"].items()} == expected
    for missing in modules:
        state["missing"] = missing
        with pytest.raises(ValueError, match="module is missing"):
            test_environments.preflight("vllm_torch_npu")
    state["missing"] = None
    versions["vllm"] = "0.19.0"
    with pytest.raises(ValueError, match="found 0.19.0"):
        test_environments.preflight("vllm_torch_npu")


def test_vllm_environment_selection_is_isolated(tmp_path):
    source = """
import pytest

def test_default():
    pass

@pytest.mark.test_environment("torch_npu")
def test_torch():
    pass

@pytest.mark.test_environment("vllm_torch_npu")
def test_vllm():
    pass
"""
    for profile, expected in (("default", {"test_default"}), ("vllm_torch_npu", {"test_vllm"})):
        result, plan, xml = _run_pytest(tmp_path / profile, source, profile=profile)
        _success(result)
        assert {case.get("name") for case in ET.parse(xml).getroot().iter("testcase")} == expected
        assert plan.exists() == (profile != "default")


def test_target_version_profile_reuses_the_declared_framework_suite():
    profile = test_environments.ENVIRONMENTS["torch_npu_2_10"]
    assert profile.execution == "direct"
    assert profile.cann_minimum == (9, 0)
    assert profile.suite == "torch_npu"
    assert dict(profile.modules) == {"torch": "torch", "torch-npu": "torch_npu"}
    requirements = test_environments.requirements("torch_npu_2_10")
    assert requirements["torch"] == "2.10.0+cpu"
    assert requirements["torch-npu"] == "2.10.0"


def test_suite_must_name_a_registered_profile(monkeypatch):
    monkeypatch.setitem(
        test_environments.ENVIRONMENTS,
        "broken",
        test_environments.TestEnvironment("direct", suite="missing"),
    )
    with pytest.raises(ValueError, match="unknown suite"):
        test_environments.declared_suite("broken")


def test_target_version_profile_selects_only_the_declared_framework_suite(tmp_path):
    source = """
import pytest

def test_default():
    pass

@pytest.mark.test_environment("torch_npu")
def test_framework():
    pass

@pytest.mark.test_environment("vllm_torch_npu")
def test_vllm():
    pass
"""
    result, plan_path, xml_path = _run_pytest(tmp_path, source, profile="torch_npu_2_10")
    _success(result)
    assert {case.get("name") for case in ET.parse(xml_path).getroot().iter("testcase")} == {"test_framework"}
    plan = json.loads(plan_path.read_text())
    assert plan["environment"] == "torch_npu_2_10"
    assert plan["suite"] == "torch_npu"
    test_environments.verify_results(plan_path, xml_path)
    assert "TEST_ENVIRONMENT_VERIFIED" in result.stdout


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
    manifest_home = home / f"{test_environments.platform.machine()}-linux" if nested else home
    manifest_home.mkdir(parents=True)
    (manifest_home / "ascend_toolkit_install.info").write_text(f"version={version}\n")
    if script:
        (home / "set_env.sh").write_text("# Selection tests never execute this script.\n")
    return home / "set_env.sh"


@pytest.mark.parametrize("command", ["select", "preflight", "requirements"])
def test_bootstrap_script_runs_without_installed_sdk(tmp_path, command):
    home = tmp_path / "cann"
    script = _installed_toolchain(home, "9.0.0")
    output = tmp_path / "github-output"
    env = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}}
    env.update(ASCEND_HOME_PATH=str(home), GITHUB_OUTPUT=str(output))
    # -S excludes site-packages and editable-install hooks, as before CI setup.
    result = subprocess.run(
        [sys.executable, "-S", str(_ROOT / "simpler_setup/test_environments.py"), command, "torch_npu"],
        cwd=_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    if command == "select":
        assert output.read_text() == f"cann_env={script}\n"
    elif command == "preflight":
        record = json.loads(result.stdout.removeprefix("TEST_ENVIRONMENT_TOOLCHAIN "))
        assert record["cann_home"] == str(home)
        assert "packages" not in record
    else:
        assert result.stdout.splitlines() == [
            f"{name}=={version}" for name, version in test_environments.requirements("torch_npu").items()
        ]


def test_toolchain_selection_uses_installed_manifest_and_deduplicates_aliases(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(test_environments.platform, "machine", lambda: "aarch64")
    root = tmp_path / "Ascend"
    active = root / "cann-beta"
    _installed_toolchain(active, "8.9.0")
    _installed_toolchain(root / "ascend-toolkit" / "9.0.0", "8.9.0")
    exact = root / "ascend-toolkit" / "release-without-version-in-name"
    expected = _installed_toolchain(exact, "9.0.0", nested=True)
    (root / "ascend-toolkit" / "latest").symlink_to(exact, target_is_directory=True)
    monkeypatch.setattr(test_environments, "_ASCEND_ROOT", root)
    monkeypatch.setenv("ASCEND_HOME_PATH", str(active))

    assert test_environments.select_toolchain("torch_npu") == expected
    inventory = json.loads(capsys.readouterr().out.removeprefix("TEST_ENVIRONMENT_TOOLCHAINS "))
    assert sum(entry["home"] == str(exact) for entry in inventory["installed"]) == 1


def test_toolchain_selection_preserves_an_exact_active_toolkit(tmp_path, monkeypatch):
    monkeypatch.setattr(test_environments.platform, "machine", lambda: "aarch64")
    root = tmp_path / "Ascend"
    _installed_toolchain(root / "cann-first", "9.0.0")
    active = tmp_path / "active-toolkit"
    expected = _installed_toolchain(active, "9.0.0")
    monkeypatch.setattr(test_environments, "_ASCEND_ROOT", root)
    monkeypatch.setenv("ASCEND_HOME_PATH", str(active))

    assert test_environments.select_toolchain("torch_npu") == expected


@pytest.mark.parametrize("version,script", [("8.9.0", True), ("9.0.0", False)])
def test_toolchain_selection_refuses_incompatible_or_incomplete_installation(tmp_path, monkeypatch, version, script):
    monkeypatch.setattr(test_environments.platform, "machine", lambda: "aarch64")
    root = tmp_path / "Ascend"
    active = root / "ascend-toolkit" / "9.0.0"
    _installed_toolchain(active, version, script=script)
    monkeypatch.setattr(test_environments, "_ASCEND_ROOT", root)
    monkeypatch.setenv("ASCEND_HOME_PATH", str(active))

    with pytest.raises(ValueError, match=r"No installed CANN meeting the minimum with set_env\.sh"):
        test_environments.select_toolchain("torch_npu")


@pytest.mark.parametrize(
    "version,machine,qualified",
    [
        ("9.0.0", "aarch64", True),
        ("9.0.0", "x86_64", True),
        ("9.2.0-beta.2", "aarch64", True),
        ("9.2.0-beta.2", "x86_64", True),
        ("9.2.0-beta.1", "aarch64", True),
        ("9.2.0", "aarch64", True),
        ("9.3", "aarch64", True),
        ("9.10.0", "x86_64", True),
        ("10.0", "x86_64", True),
        ("10.0.0-beta.1", "aarch64", True),
        ("9", "x86_64", False),
        ("9.0", "aarch64", True),
        ("9.0.1-beta.1", "x86_64", True),
        ("9.0.0-beta.2", "aarch64", True),
        ("9.0.0-rc.1", "x86_64", True),
        ("9.1.T500", "aarch64", True),
        ("9.0.RC1", "x86_64", True),
        ("9.0garbage", "x86_64", False),
        ("9.0-", "aarch64", False),
        ("10", "x86_64", False),
        ("8.99", "aarch64", False),
        ("8.9.99", "aarch64", False),
        ("unknown", "aarch64", False),
        ("9.2broken", "x86_64", False),
        ("9.2.0.1", "x86_64", True),
        ("9.2.0-beta", "aarch64", True),
        ("09.2.0", "aarch64", True),
        ("", "aarch64", False),
    ],
)
def test_toolchain_qualification_matches_selection_and_preflight(
    tmp_path, monkeypatch, capsys, version, machine, qualified
):
    home = tmp_path / "misleading-name-9.0.0"
    expected = _installed_toolchain(home, version)
    monkeypatch.setattr(test_environments.platform, "machine", lambda: machine)
    monkeypatch.setenv("ASCEND_HOME_PATH", str(home))
    real_import = builtins.__import__

    def without_framework(name, *args, **kwargs):
        assert name.split(".")[0] not in {"torch", "torch_npu"}
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_framework)
    if qualified:
        assert test_environments.select_toolchain("torch_npu", roots=[]) == expected
        record = test_environments.preflight("torch_npu", packages=False)
        assert record["cann"]["version"] == version
        assert record["machine"] == machine
    else:
        with pytest.raises(ValueError, match="No installed CANN meeting the minimum"):
            test_environments.select_toolchain("torch_npu", roots=[])
        with pytest.raises(ValueError, match="does not meet the minimum version"):
            test_environments.preflight("torch_npu", packages=False)
    inventory = json.loads(capsys.readouterr().out.removeprefix("TEST_ENVIRONMENT_TOOLCHAINS "))
    assert inventory["installed"][0]["version"] == version
    assert inventory["machine"] == machine


@pytest.mark.parametrize("active_version,fallback_version", [("9.0.0", "9.2.0-beta.2"), ("9.2.0-beta.2", "9.0.0")])
def test_each_qualified_active_toolchain_precedes_fallback(tmp_path, monkeypatch, active_version, fallback_version):
    active = tmp_path / "active"
    expected = _installed_toolchain(active, active_version)
    fallback = tmp_path / "fallback"
    _installed_toolchain(fallback, fallback_version)
    monkeypatch.setattr(test_environments.platform, "machine", lambda: "aarch64")
    monkeypatch.setenv("ASCEND_HOME_PATH", str(active))
    assert test_environments.select_toolchain("torch_npu", roots=[fallback]) == expected


@pytest.mark.parametrize("machine,active_version", [("aarch64", "8.9.0"), ("x86_64", "9.0garbage")])
def test_unqualified_active_falls_back_but_preflight_checks_actual_active(
    tmp_path, monkeypatch, machine, active_version
):
    active = tmp_path / "active"
    _installed_toolchain(active, active_version)
    fallback = tmp_path / "fallback"
    expected = _installed_toolchain(fallback, "9.0.0")
    later = tmp_path / "later"
    _installed_toolchain(later, "9.0.0")
    monkeypatch.setattr(test_environments.platform, "machine", lambda: machine)
    monkeypatch.setenv("ASCEND_HOME_PATH", str(active))
    assert test_environments.select_toolchain("torch_npu", roots=[fallback, later]) == expected
    with pytest.raises(ValueError, match="does not meet the minimum version"):
        test_environments.preflight("torch_npu", packages=False)
    monkeypatch.setenv("ASCEND_HOME_PATH", str(expected.parent))
    assert test_environments.preflight("torch_npu", packages=False)["cann"]["version"] == "9.0.0"


def test_final_collection_plan_preserves_preflight_identity(tmp_path):
    result, plan_path, _ = _run_pytest(tmp_path, _DIRECT.format(count=1))
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(plan_path.read_text())["identity"] == {"environment": "torch_npu", "CPU_contract": True}


def test_toolchain_manifest_is_required_by_selection_and_preflight(tmp_path, monkeypatch):
    home = tmp_path / "cann-9.2.0-beta.2"
    home.mkdir()
    (home / "set_env.sh").write_text("# No manifest; the directory name is not qualification.\n")
    monkeypatch.setattr(test_environments.platform, "machine", lambda: "aarch64")
    monkeypatch.setenv("ASCEND_HOME_PATH", str(home))
    with pytest.raises(ValueError, match="No installed CANN meeting the minimum"):
        test_environments.select_toolchain("torch_npu", roots=[])
    with pytest.raises(ValueError, match="CANN install manifest missing"):
        test_environments.preflight("torch_npu", packages=False)
