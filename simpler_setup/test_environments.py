# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Declared pytest environments and strict receipts for direct test execution.

Environment selection does not import optional backends. Tests must defer those
imports until fixtures or device-owning subprocesses execute.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TestEnvironment:
    execution: str
    constraints: str | None = None
    cann_version: str | None = None


ENVIRONMENTS = {
    "default": TestEnvironment("dispatcher"),
    "torch_npu": TestEnvironment(
        "direct", ".github/actions/setup-framework-eager/constraints.txt", "9.0.0"
    ),
}
_PROPERTY_PREFIX = "simpler_"


def _environment(name: str) -> TestEnvironment:
    if name not in ENVIRONMENTS:
        raise ValueError(f"Unknown test environment {name!r}; choose from {', '.join(ENVIRONMENTS)}")
    return ENVIRONMENTS[name]


def environment_for_item(item) -> str:
    """Identical inherited declarations are idempotent; conflicts are errors."""
    declared = set()
    for marker in item.iter_markers("test_environment"):
        if marker.kwargs or len(marker.args) != 1 or not isinstance(marker.args[0], str):
            raise ValueError(f"{item.nodeid}: test_environment requires one environment name")
        _environment(marker.args[0])
        declared.add(marker.args[0])
    if len(declared) > 1:
        raise ValueError(f"{item.nodeid}: conflicting test environments {sorted(declared)}")
    return next(iter(declared), "default")


def select_environment(items, name: str):
    _environment(name)
    selected, deselected = [], []
    for item in items:
        (selected if environment_for_item(item) == name else deselected).append(item)
    return selected, deselected


def environment_argv(config) -> list[str]:
    return ["--test-environment", config.getoption("--test-environment", default="default")]


def _checkout() -> Path:
    return Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())


def _source() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def requirements(name: str) -> dict[str, str]:
    profile = _environment(name)
    if profile.constraints is None:
        return {}
    lines = (_checkout() / profile.constraints).read_text().splitlines()
    return dict(line.strip().split("==", 1) for line in lines if line.strip() and not line.startswith("#"))


def preflight(name: str, *, packages: bool = True) -> dict:
    """Validate the active toolchain and package metadata without importing them."""
    profile = _environment(name)
    record = {"environment": name, "python": sys.version, "executable": sys.executable,
              "machine": platform.machine(), "libc": platform.libc_ver()}
    if profile.cann_version is not None:
        home = Path(os.environ["ASCEND_HOME_PATH"]).resolve()
        candidates = [home / "ascend_toolkit_install.info", home / f"{platform.machine()}-linux/ascend_toolkit_install.info"]
        manifest = next((path for path in candidates if path.is_file()), None)
        if manifest is None:
            raise ValueError(f"CANN install manifest missing under active ASCEND_HOME_PATH={home}")
        fields = dict(line.split("=", 1) for line in manifest.read_text().splitlines() if "=" in line)
        record.update(cann_home=str(home), manifest=str(manifest), cann=fields)
        if fields.get("version") != profile.cann_version:
            raise ValueError(f"{name} requires CANN {profile.cann_version}; active manifest reports {fields}")
    if packages:
        installed = {}
        for module, expected in requirements(name).items():
            spec = importlib.util.find_spec(module)
            if spec is None:
                raise ValueError(f"Selected test environment {name} requires {module}=={expected}; module is missing")
            version = importlib.metadata.version(module)
            if version != expected:
                raise ValueError(f"Selected test environment {name} requires {module}=={expected}; found {version}")
            installed[module] = {"version": version, "path": spec.origin}
        record["packages"] = installed
    return record


def _plan_digest(plan: dict) -> str:
    return hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()


def prepare_session(session) -> None:
    """Freeze the final collection once; default keeps its existing dispatcher."""
    import pytest  # noqa: PLC0415
    from _pytest.junitxml import mangle_test_address  # noqa: PLC0415

    config = session.config
    name = config.getoption("--test-environment", default="default")
    if _environment(name).execution != "direct" or config.getoption("collectonly", default=False):
        return
    xmlpath = config.getoption("xmlpath", default=None)
    if not xmlpath:
        raise pytest.UsageError(f"{name} requires --junitxml to verify every selected test")
    if session.testsfailed or not session.items:
        raise pytest.UsageError(f"{name} requires a nonempty successful collection")
    if config.getoption("--resource-child", default=False) or config.getoption("numprocesses", default=None):
        raise pytest.UsageError(f"{name} strict receipts support direct pytest execution only")
    cases = []
    for item in session.items:
        if environment_for_item(item) != name:
            raise pytest.UsageError(f"Wrong environment in final collection: {item.nodeid}")
        if item.get_closest_marker("device_count") or getattr(getattr(item, "cls", None), "_st_level", None):
            raise pytest.UsageError(f"{name} cannot aggregate dispatcher results: {item.nodeid}")
        if any(key.startswith(_PROPERTY_PREFIX) for key, _ in item.user_properties):
            raise pytest.UsageError(f"{item.nodeid}: reserved test receipt property")
        address = mangle_test_address(item.nodeid)
        prefix = config.getoption("junitprefix", default=None)
        if prefix:
            address.insert(0, prefix)
        cases.append({"nodeid": item.nodeid, "classname": ".".join(address[:-1]), "name": address[-1]})
    if len({case["nodeid"] for case in cases}) != len(cases):
        raise pytest.UsageError("Duplicate nodeids in selected test environment")
    try:
        identity = preflight(name)
    except (ValueError, OSError, KeyError, importlib.metadata.PackageNotFoundError) as error:
        raise pytest.UsageError(str(error)) from error
    plan = {"source": _source(), "environment": name, "platform": config.getoption("--platform", default=None),
            "cases": cases}
    destination = Path(xmlpath).with_suffix(".plan.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(plan, indent=2) + "\n")
    digest = _plan_digest(plan)
    for item in session.items:
        item.user_properties.extend([
            ("simpler_nodeid", item.nodeid), ("simpler_source", plan["source"]),
            ("simpler_environment", name), ("simpler_plan", digest),
        ])
    session._test_environment_receipt = (destination, Path(xmlpath))
    print("TEST_ENVIRONMENT_PLAN " + json.dumps({"path": str(destination), "plan": plan, "identity": identity}), flush=True)


def verify_results(plan_path: Path, xml_path: Path) -> None:
    plan = json.loads(plan_path.read_text())
    profile = _environment(plan["environment"])
    if profile.execution != "direct" or plan["source"] != _source():
        raise ValueError("Test plan source or execution environment does not match this checkout")
    expected = {case["nodeid"]: case for case in plan["cases"]}
    if not expected or len(expected) != len(plan["cases"]):
        raise ValueError("Test plan must contain unique, nonempty nodeids")
    root = ET.parse(xml_path).getroot()
    for suite in root.iter("testsuite"):
        if any(int(suite.get(outcome, "0")) for outcome in ("failures", "errors", "skipped")):
            raise ValueError("Selected test environment reports a failure, error or skip")
    observed = set()
    for case in root.iter("testcase"):
        properties = [(p.get("name"), p.get("value")) for p in case.findall("properties/property")]
        values = dict(properties)
        nodeid = values.get("simpler_nodeid")
        if nodeid not in expected or nodeid in observed:
            raise ValueError(f"Missing, foreign or duplicate test receipt: {nodeid}")
        target = expected[nodeid]
        required = {"simpler_nodeid": nodeid, "simpler_source": plan["source"],
                    "simpler_environment": plan["environment"], "simpler_plan": _plan_digest(plan)}
        for key, value in required.items():
            if properties.count((key, value)) != 1 or sum(name == key for name, _ in properties) != 1:
                raise ValueError(f"Test receipt has wrong source, environment or plan: {nodeid}")
        if case.get("classname") != target["classname"] or case.get("name") != target["name"]:
            raise ValueError(f"Test receipt has wrong JUnit identity: {nodeid}")
        if any(child.tag in {"failure", "error", "skipped"} for child in case):
            raise ValueError(f"Selected test did not pass: {nodeid}")
        observed.add(nodeid)
    if observed != set(expected):
        raise ValueError(f"Tests selected but not executed: {sorted(set(expected) - observed)}")
    print("TEST_ENVIRONMENT_VERIFIED " + ", ".join(sorted(observed)), flush=True)


def finish_session(session) -> None:
    receipt = getattr(session, "_test_environment_receipt", None)
    if receipt is not None:
        try:
            verify_results(*receipt)
        except (ValueError, OSError, KeyError, ET.ParseError) as error:
            session.exitstatus = 1
            print(f"TEST_ENVIRONMENT_VERIFICATION_FAILED {error}", file=sys.stderr, flush=True)


def main() -> None:
    command, *args = sys.argv[1:]
    if command == "requirements":
        print("\n".join(f"{name}=={version}" for name, version in requirements(args[0]).items()))
    elif command == "preflight":
        print("TEST_ENVIRONMENT_TOOLCHAIN " + json.dumps(preflight(args[0], packages=False)), flush=True)
    elif command == "verify":
        verify_results(Path(args[0]), Path(args[1]))
    else:
        raise SystemExit(f"Unknown test environment command {command!r}")


if __name__ == "__main__":
    main()
