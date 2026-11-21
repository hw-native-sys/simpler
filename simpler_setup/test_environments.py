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
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

# Framework toolchain selection runs this file before the SDK is installed.
if __package__:
    from simpler_setup import test_coverage
else:
    import test_coverage


@dataclass(frozen=True)
class TestEnvironment:
    execution: str
    constraints: str | None = None
    cann_minimum: tuple[int, int] | None = None
    modules: tuple[tuple[str, str], ...] = ()
    # Version-qualified profiles can qualify a suite that declares another
    # environment name. The suite name must itself be a registered profile.
    suite: str | None = None


ENVIRONMENTS = {
    "default": TestEnvironment("dispatcher"),
    "torch_npu": TestEnvironment(
        "direct",
        ".github/actions/setup-framework/constraints.txt",
        (9, 0),
        (("torch", "torch"), ("torch_npu", "torch_npu")),
    ),
    # Target vLLM-Ascend 0.20.2rc1 stack (issue #2715): the same declared
    # framework eager suite on Torch-NPU 2.10.0 instead of the pinned 2.7.1.
    "torch_npu_2_10": TestEnvironment(
        "direct",
        ".github/actions/setup-vllm-eager/constraints.txt",
        (9, 0),
        (("torch", "torch"), ("torch-npu", "torch_npu")),
        suite="torch_npu",
    ),
    "vllm_torch_npu": TestEnvironment(
        "direct",
        ".github/actions/setup-vllm-eager/constraints.txt",
        (9, 0),
        (
            ("torch", "torch"),
            ("torch-npu", "torch_npu"),
            ("numpy", "numpy"),
            ("vllm", "vllm"),
            ("vllm-ascend", "vllm_ascend"),
        ),
    ),
}
_PROPERTY_PREFIX = "simpler_"
_ASCEND_ROOT = Path("/usr/local/Ascend")


def _environment(name: str) -> TestEnvironment:
    if name not in ENVIRONMENTS:
        raise ValueError(f"Unknown test environment {name!r}; choose from {', '.join(ENVIRONMENTS)}")
    profile = ENVIRONMENTS[name]
    if profile.suite is not None and profile.suite not in ENVIRONMENTS:
        raise ValueError(f"Test environment {name!r} declares unknown suite {profile.suite!r}")
    return profile


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


def declared_suite(name: str) -> str:
    """Return the suite name that items declare for this environment profile."""
    profile = _environment(name)
    return profile.suite or name


def select_environment(items, name: str):
    declared = declared_suite(name)
    selected, deselected = [], []
    for item in items:
        (selected if environment_for_item(item) == declared else deselected).append(item)
    return selected, deselected


def environment_argv(config) -> list[str]:
    return ["--test-environment", config.getoption("--test-environment", default="default")]


def validate_execution(config) -> None:
    """Reject distributed execution before a controller can start workers."""
    import pytest  # noqa: PLC0415

    name = config.getoption("--test-environment", default="default")
    if name == "default" and config.getoption("--environment-receipts", default=False):
        distributed = bool(config.getoption("numprocesses", default=None)) or hasattr(config, "workerinput")
        if distributed and not config.getoption("--environment-parent-plan", default=None):
            raise pytest.UsageError("Distributed default receipts require an immutable parent plan")
    if _environment(name).execution != "direct":
        return
    distributed = bool(config.getoption("numprocesses", default=None)) or (
        config.getoption("dist", default="no") != "no" and bool(config.getoption("tx", default=()))
    )
    if (
        distributed
        or hasattr(config, "workerinput")
        or config.getoption("looponfail", default=False)
        or config.getoption("--resource-child", default=False)
    ):
        raise pytest.UsageError(f"{name} strict receipts support direct pytest execution only")


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


def _manifest(home: Path) -> tuple[Path, dict[str, str]]:
    candidates = [
        home / "ascend_toolkit_install.info",
        home / f"{platform.machine()}-linux/ascend_toolkit_install.info",
    ]
    manifest = next((path for path in candidates if path.is_file()), None)
    if manifest is None:
        raise ValueError(f"CANN install manifest missing under {home}")
    fields = dict(line.split("=", 1) for line in manifest.read_text().splitlines() if "=" in line)
    return manifest, fields


def _accepts_cann_version(profile: TestEnvironment, version: str | None) -> bool:
    """Compare CANN major/minor lines; separated release/build labels are opaque."""
    match = re.fullmatch(r"([0-9]+)\.([0-9]+)(?:[._+-][A-Za-z0-9]+)*", version or "")
    if match is None or profile.cann_minimum is None:
        return False
    return (int(match[1]), int(match[2])) >= profile.cann_minimum


def select_toolchain(name: str, *, roots: list[Path] | None = None) -> Path:
    """Choose an installed toolkit meeting the minimum CANN version line without changing the host default."""
    profile = _environment(name)
    if not profile.cann_minimum:
        raise ValueError(f"{name} does not declare a CANN toolchain")
    machine = platform.machine()
    active = os.environ.get("ASCEND_HOME_PATH")
    if roots is None:
        roots = sorted(_ASCEND_ROOT.glob("cann*"))
        roots += sorted((_ASCEND_ROOT / "ascend-toolkit").glob("*"))
    roots = ([Path(active)] if active else []) + roots
    seen = set()
    installed = []
    matches = []
    for root in roots:
        home = root.resolve()
        if home in seen or not home.is_dir():
            continue
        seen.add(home)
        entry: dict[str, object] = {"home": str(home)}
        try:
            manifest, fields = _manifest(home)
            script = home / "set_env.sh"
            entry.update(
                manifest=str(manifest),
                version=fields.get("version", "unknown"),
                set_env=str(script),
                set_env_exists=script.is_file(),
            )
            if _accepts_cann_version(profile, fields.get("version")) and script.is_file():
                matches.append(script)
        except (OSError, ValueError) as error:
            entry["error"] = str(error)
        installed.append(entry)
    print(
        "TEST_ENVIRONMENT_TOOLCHAINS " + json.dumps({"environment": name, "machine": machine, "installed": installed}),
        flush=True,
    )
    if not matches:
        raise ValueError(
            f"No installed CANN meeting the minimum with set_env.sh for {name} on {machine}; "
            f"requires CANN {profile.cann_minimum[0]}.{profile.cann_minimum[1]} or newer; see toolchain inventory"
        )
    return matches[0]


def preflight(name: str, *, packages: bool = True) -> dict:
    """Validate the active toolchain and package metadata without importing them."""
    profile = _environment(name)
    record = {
        "environment": name,
        "python": sys.version,
        "executable": sys.executable,
        "machine": platform.machine(),
        "libc": platform.libc_ver(),
    }
    if profile.cann_minimum:
        home = Path(os.environ["ASCEND_HOME_PATH"]).resolve()
        manifest, fields = _manifest(home)
        record.update(cann_home=str(home), manifest=str(manifest), cann=fields)
        if not _accepts_cann_version(profile, fields.get("version")):
            raise ValueError(
                f"{name} active CANN does not meet the minimum version; "
                f"requires CANN {profile.cann_minimum[0]}.{profile.cann_minimum[1]} or newer; "
                f"active manifest reports {fields}"
            )
    if packages:
        installed = {}
        pinned = requirements(name)
        for distribution, module in profile.modules:
            expected = pinned[distribution]
            spec = importlib.util.find_spec(module)
            if spec is None:
                raise ValueError(f"Selected test environment {name} requires {module}=={expected}; module is missing")
            version = importlib.metadata.version(distribution)
            if version != expected:
                raise ValueError(f"Selected test environment {name} requires {module}=={expected}; found {version}")
            installed[distribution] = {"module": module, "version": version, "path": spec.origin}
        record["packages"] = installed
    return record


def _plan_digest(plan: dict) -> str:
    return hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()


def child_receipt_argv(config, nodeids=None) -> list[str]:
    """Allocate an exclusive result path without changing resource scheduling."""
    plan = getattr(config, "_environment_plan", None)
    if plan is None:
        return []
    directory = Path(tempfile.mkdtemp(prefix="shard-", dir=plan.parent))
    result = directory / "results.xml"
    config._environment_results.append(result)
    feature_args = ["--verify-feature-coverage"] if config.getoption("--verify-feature-coverage", default=False) else []
    assigned = [arg for nodeid in nodeids or [] for arg in ("--environment-shard-case", nodeid)]
    return [
        "--environment-receipts",
        *feature_args,
        "--environment-parent-plan",
        str(plan),
        *assigned,
        "--junitxml",
        str(result),
    ]


def retry_receipt_argv(config, sink: Path, original: str) -> list[str]:
    """Authorize one bounded retry from the existing poison-skip sink."""
    records = [json.loads(line) for line in sink.read_text().splitlines() if line.strip()]
    nodeids = sorted({record["nodeid"] for record in records})
    plan = json.loads(config._environment_plan.read_text())
    expected = {case["nodeid"] for case in plan["cases"]}
    if not nodeids or not set(nodeids) <= expected:
        raise ValueError("Poison retry contains unplanned nodeids")
    args = child_receipt_argv(config)
    retries = getattr(config, "_environment_retries", {})
    retries[args[-1]] = {"original": original, "nodeids": nodeids}
    config._environment_retries = retries
    for nodeid in nodeids:
        args.extend(["--environment-retry-case", nodeid])
    return args


def execution_mode(item, default=None) -> str:
    return test_coverage.execution_mode(item, default=default)


def prepare_session(session) -> None:  # noqa: PLR0912
    """Freeze the final collection once; default keeps its existing dispatcher."""
    import pytest  # noqa: PLC0415

    # Pytest owns the spelling of its JUnit classname and parameterized name.
    from _pytest.junitxml import mangle_test_address  # noqa: PLC0415

    config = session.config
    name = config.getoption("--test-environment", default="default")
    opted_in = config.getoption("--environment-receipts", default=False)
    if (_environment(name).execution != "direct" and not opted_in) or config.getoption("collectonly", default=False):
        return
    declared = declared_suite(name)
    xmlpath = config.getoption("xmlpath", default=None)
    if not xmlpath:
        raise pytest.UsageError(f"{name} requires --junitxml to verify every selected test")
    if session.testsfailed or not session.items:
        raise pytest.UsageError(f"{name} requires a nonempty successful collection")
    validate_execution(config)
    cases = []
    for item in session.items:
        if environment_for_item(item) != declared:
            raise pytest.UsageError(f"Wrong environment in final collection: {item.nodeid}")
        if _environment(name).execution == "direct" and (
            item.get_closest_marker("device_count") or getattr(getattr(item, "cls", None), "_st_level", None)
        ):
            raise pytest.UsageError(f"{name} cannot aggregate dispatcher results: {item.nodeid}")
        if any(key.startswith(_PROPERTY_PREFIX) for key, _ in item.user_properties):
            raise pytest.UsageError(f"{item.nodeid}: reserved test receipt property")
        address = mangle_test_address(item.nodeid)
        prefix = config.getoption("junitprefix", default=None)
        if prefix:
            address.insert(0, prefix)
        cases.append({"nodeid": item.nodeid, "classname": ".".join(address[:-1]), "name": address[-1]})
        if opted_in:
            try:
                if config.getoption("--verify-feature-coverage", default=False):
                    feature = test_coverage.features(item, config.getoption("--platform"), name)
                    cases[-1]["feature"] = feature
                    # Program/auxiliary cases carry no declared mode; receipts still
                    # record the eager execution they imply.
                    cases[-1]["execution_mode"] = feature["mode"] or "eager"
                else:
                    # Receipts without coverage cannot know the product; an
                    # undeclared mode is the eager execution such cases imply.
                    cases[-1]["execution_mode"] = execution_mode(item, default="eager")
            except ValueError as error:
                raise pytest.UsageError(str(error)) from error
    if len({case["nodeid"] for case in cases}) != len(cases):
        raise pytest.UsageError("Duplicate nodeids in selected test environment")
    try:
        identity = preflight(name)
    except (
        ValueError,
        OSError,
        KeyError,
        subprocess.CalledProcessError,
        importlib.metadata.PackageNotFoundError,
    ) as error:
        raise pytest.UsageError(str(error)) from error
    try:
        source = _source()
    except (OSError, subprocess.CalledProcessError) as error:
        raise pytest.UsageError(f"Cannot identify the test source checkout: {error}") from error
    plan = {
        "source": source,
        "environment": name,
        "suite": declared,
        "platform": config.getoption("--platform", default=None),
        "identity": identity,
        "cases": cases,
    }
    parent = config.getoption("--environment-parent-plan", default=None)
    if config.getoption("--verify-feature-coverage", default=False) and not parent:
        try:
            plan["coverage"] = test_coverage.policy(config, cases)
        except ValueError as error:
            raise pytest.UsageError(str(error)) from error
    if opted_in and not parent:
        required = set(config.getoption("--require-execution-mode", default=[]))
        missing = required - {case["execution_mode"] for case in cases}
        if missing:
            raise pytest.UsageError(f"Required execution modes not selected: {sorted(missing)}")
    destination = Path(parent) if parent else Path(xmlpath).resolve().with_suffix(".plan.json")
    if parent:
        parent_plan = json.loads(destination.read_text())
        if any(plan[key] != parent_plan[key] for key in ("source", "environment", "platform")):
            raise pytest.UsageError("Child source, environment or platform differs from parent plan")
        expected = {case["nodeid"]: case for case in parent_plan["cases"]}
        if any(expected.get(case["nodeid"]) != case for case in cases):
            raise pytest.UsageError("Child selected a case outside the parent plan")
        assigned = config.getoption("--environment-shard-case", default=[])
        if assigned and set(assigned) != {case["nodeid"] for case in cases}:
            raise pytest.UsageError("Child selection differs from its assigned plan members")
        plan = parent_plan
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(plan, indent=2) + "\n")
    digest = _plan_digest(plan)
    for item in session.items:
        item.user_properties.extend(
            [
                ("simpler_nodeid", item.nodeid),
                ("simpler_source", plan["source"]),
                ("simpler_environment", name),
                ("simpler_plan", digest),
            ]
        )
        if "coverage" in plan:
            item.user_properties.append(("simpler_executed", "false"))
    if not parent:
        session._test_environment_receipt = (destination, Path(xmlpath))
        if opted_in and _environment(name).execution == "dispatcher":
            config._environment_plan = destination
            config._environment_results = [Path(xmlpath)]
    # The plan itself is the file at ``path``; the console line only identifies
    # it, so every dispatched child does not re-echo the whole case list.
    print(
        "TEST_ENVIRONMENT_PLAN "
        + json.dumps({"path": str(destination), "digest": digest, "cases": len(plan["cases"])}),
        flush=True,
    )


def verify_results(plan_path: Path, xml_path: Path) -> None:
    verify_result_files(plan_path, [xml_path])


def _retry_targets(paths, expected, retries, dispatcher):
    retry_targets = {}
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate result files")
    for retry_path, authorization in (retries or {}).items():
        original = authorization["original"]
        nodes = authorization["nodeids"]
        if not dispatcher or original == retry_path or original not in paths or retry_path not in paths or not nodes:
            raise ValueError("Invalid retry result paths")
        if len(set(nodes)) != len(nodes):
            raise ValueError("Duplicate retry authorization")
        for nodeid in nodes:
            if nodeid not in expected or nodeid in retry_targets:
                raise ValueError("Foreign or repeated retry authorization")
            retry_targets[nodeid] = {(original, "skipped"), (retry_path, "passed")}
    return retry_targets


class ResultVerificationError(ValueError):
    """Failed verification with source-checked partial execution evidence."""

    def __init__(self, errors, summary):
        super().__init__("; ".join(errors))
        self.summary = summary


def _result_roots(xml_paths, retries, optional_skips=False):
    roots = []
    errors = []
    for xml_path in xml_paths:
        try:
            root = ET.parse(xml_path).getroot()
        except (OSError, ET.ParseError) as error:
            errors.append(f"Unreadable result {xml_path}: {error}")
            continue
        roots.append((str(xml_path), root))
        for suite in root.iter("testsuite"):
            outcomes = ("failures", "errors") if retries or optional_skips else ("failures", "errors", "skipped")
            if any(int(suite.get(outcome, "0")) for outcome in outcomes):
                errors.append("Selected test environment reports a failure, error or skip")
            if (retries or optional_skips) and int(suite.get("skipped", "0")) != len(suite.findall("testcase/skipped")):
                errors.append("Retry suite skip count disagrees with its receipts")
    return roots, errors


def _verified_case(case, expected, plan):
    properties = [(p.get("name"), p.get("value")) for p in case.findall("properties/property")]
    values = dict(properties)
    nodeid = values.get("simpler_nodeid")
    if nodeid not in expected:
        raise ValueError(f"Missing, foreign or duplicate test receipt: {nodeid}")
    target = expected[nodeid]
    required = {
        "simpler_nodeid": nodeid,
        "simpler_source": plan["source"],
        "simpler_environment": plan["environment"],
        "simpler_plan": _plan_digest(plan),
    }
    for key, value in required.items():
        if properties.count((key, value)) != 1 or sum(name == key for name, _ in properties) != 1:
            raise ValueError(f"Test receipt has wrong source, environment or plan: {nodeid}")
    if case.get("classname") != target["classname"] or case.get("name") != target["name"]:
        raise ValueError(f"Test receipt has wrong JUnit identity: {nodeid}")
    outcome = "skipped" if case.find("skipped") is not None else "passed"
    if any(child.tag in {"failure", "error"} for child in case):
        outcome = "failed"
    executed = outcome == "passed"
    if "coverage" in plan:
        flags = [value for key, value in properties if key == "simpler_executed"]
        if len(flags) != 1 or flags[0] not in ("true", "false"):
            raise ValueError(f"Missing or invalid execution evidence: {nodeid}")
        executed = flags[0] == "true"
        if outcome == "passed" and not executed:
            raise ValueError(f"Passing receipt without call execution: {nodeid}")
    return nodeid, outcome, executed


def _coverage_summary(plan, expected, observed, recovered, executions, invalid):
    counts = {}
    for case in expected.values():
        mode = case["execution_mode"]
        counts[mode] = counts.get(mode, 0) + 1
    summary = {
        "source": plan["source"],
        "environment": plan["environment"],
        "platform": plan["platform"],
        "identity": plan["identity"],
        "attempts": sum(len(attempts) for attempts in observed.values()),
        "recovered": len(recovered),
        "groups": {mode: {"selected": count, "executed": count, "passed": count} for mode, count in counts.items()},
    }
    if "coverage" in plan:
        summary.update(test_coverage.summary(plan, observed, recovered, executions, invalid))
    return summary


def verify_result_files(  # noqa: PLR0912
    plan_path: Path, xml_paths: list[Path], *, dispatcher: bool = False, retries: dict | None = None
) -> None:
    """Verify the union of explicit result files against one immutable plan."""
    retries = retries or {}
    if dispatcher:
        plan_path.with_suffix(".coverage.json").unlink(missing_ok=True)
    plan = json.loads(plan_path.read_text())
    if "coverage" in plan:
        plan_path.with_suffix(".coverage.json").unlink(missing_ok=True)
    profile = _environment(plan["environment"])
    try:
        source = _source()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"Cannot identify the test source checkout: {error}") from error
    if profile.execution != ("dispatcher" if dispatcher else "direct") or plan["source"] != source:
        raise ValueError("Test plan source or execution environment does not match this checkout")
    if plan.get("suite", plan["environment"]) != (profile.suite or plan["environment"]):
        raise ValueError("Test plan suite does not match its environment profile")
    expected = {case["nodeid"]: case for case in plan["cases"]}
    if not expected or len(expected) != len(plan["cases"]):
        raise ValueError("Test plan must contain unique, nonempty nodeids")
    retry_targets = _retry_targets([str(path) for path in xml_paths], expected, retries, dispatcher)
    optional_skips = dispatcher and any(case.get("feature", {}).get("allowed_skip") for case in expected.values())
    roots, errors = _result_roots(xml_paths, retries, optional_skips)
    observed = {}
    executions = {}
    invalid = set()
    for path, case in ((path, case) for path, root in roots for case in root.iter("testcase")):
        try:
            nodeid, outcome, executed = _verified_case(case, expected, plan)
            if path in retries and nodeid not in retries[path]["nodeids"]:
                raise ValueError(f"Unauthorized case in retry result: {nodeid}")
        except ValueError as error:
            errors.append(str(error))
            invalid.update(p.get("value") for p in case.findall("properties/property[@name='simpler_nodeid']"))
            continue
        executions[nodeid] = executions.get(nodeid, False) or executed
        attempts = observed.setdefault(nodeid, [])
        attempts.append((path, outcome))
        if outcome == "failed" or (
            outcome == "skipped"
            and (path, outcome) not in retry_targets.get(nodeid, set())
            and not test_coverage.allows_skip(plan, expected[nodeid], case)
        ):
            errors.append(f"Selected test did not pass: {nodeid}")
    if set(observed) != set(expected):
        errors.append(f"Tests selected but not executed: {sorted(set(expected) - set(observed))}")
    recovered = set()
    for nodeid, attempts in observed.items():
        if nodeid in retry_targets:
            if len(attempts) != 2 or set(attempts) != retry_targets[nodeid]:
                errors.append(f"Incomplete or invalid retry: {nodeid}")
                invalid.add(nodeid)
            else:
                recovered.add(nodeid)
        elif len(attempts) != 1:
            errors.append(f"Missing, foreign or duplicate test receipt: {nodeid}")
            invalid.add(nodeid)
    recovered -= invalid
    summary = None
    if dispatcher or "coverage" in plan:
        summary = _coverage_summary(plan, expected, observed, recovered, executions, invalid)
        summary["errors"] = list(dict.fromkeys(errors))
        incomplete = "coverage" in plan and any(group["missing"] for group in summary["groups"].values())
        summary["verification_status"] = "incomplete" if incomplete else "failed" if errors else "verified"
    if errors:
        raise ResultVerificationError(list(dict.fromkeys(errors)), summary if "coverage" in plan else None)
    if summary is not None:
        plan_path.with_suffix(".coverage.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        "TEST_ENVIRONMENT_VERIFIED "
        + json.dumps({"cases": len(observed), "plan": plan_path.name, "digest": _plan_digest(plan)}),
        flush=True,
    )


def finish_session(session) -> None:
    receipt = getattr(session, "_test_environment_receipt", None)
    if receipt is not None:
        try:
            results = getattr(session.config, "_environment_results", None)
            if results is None:
                verify_results(*receipt)
            else:
                retries = getattr(session.config, "_environment_retries", {})
                receipt[0].with_suffix(".results.json").write_text(
                    json.dumps({"files": [str(path) for path in results], "retries": retries}, indent=2) + "\n"
                )
                verify_result_files(receipt[0], results, dispatcher=True, retries=retries)
        except (ValueError, OSError, KeyError, ET.ParseError) as error:
            session.exitstatus = 1
            print(f"TEST_ENVIRONMENT_VERIFICATION_FAILED {error}", file=sys.stderr, flush=True)


def main() -> None:
    command, *args = sys.argv[1:]
    if command == "requirements":
        selected = requirements(args[0])
        if len(args) > 1:
            selected = {args[1]: selected[args[1]]}
        print("\n".join(f"{name}=={version}" for name, version in selected.items()))
    elif command == "select":
        script = select_toolchain(args[0])
        print("TEST_ENVIRONMENT_TOOLCHAIN_SELECTED " + str(script), flush=True)
        if os.environ.get("GITHUB_OUTPUT"):
            with open(os.environ["GITHUB_OUTPUT"], "a") as output:
                output.write(f"cann_env={script}\n")
    elif command == "preflight":
        print("TEST_ENVIRONMENT_TOOLCHAIN " + json.dumps(preflight(args[0], packages=False)), flush=True)
    elif command == "verify":
        verify_results(Path(args[0]), Path(args[1]))
    else:
        raise SystemExit(f"Unknown test environment command {command!r}")


if __name__ == "__main__":
    main()
