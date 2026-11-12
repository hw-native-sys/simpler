# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Feature requirements over the existing environment and invocation scope."""

from __future__ import annotations

RUNTIMES = ("host_build_graph", "tensormap_and_ringbuffer")


def _declaration(item, name, choices=None, class_default=None):
    values = set()
    for marker in item.iter_markers(name):
        if marker.kwargs or len(marker.args) != 1 or not isinstance(marker.args[0], str):
            raise ValueError(f"{item.nodeid}: invalid {name}")
        value = marker.args[0]
        if choices and value not in choices:
            raise ValueError(f"{item.nodeid}: invalid {name} {value!r}")
        values.add(value)
    if len(values) > 1:
        raise ValueError(f"{item.nodeid}: conflicting {name}")
    value = next(iter(values), None)
    if value is None and class_default:
        value = getattr(getattr(item, "cls", None), class_default, None)
        if value is not None and choices and value not in choices:
            raise ValueError(f"{item.nodeid}: invalid {class_default} {value!r}")
    return value


def execution_mode(item, default=None):
    mode = _declaration(item, "execution_mode", ("eager", "capture_replay"), "_st_execution_mode")
    if mode is None:
        if default is not None:
            return default
        raise ValueError(f"Missing or conflicting execution_mode: {item.nodeid}")
    return mode


def features(item, platform, environment):
    # Program is the default: a case declares a product only when it is Kernel
    # or auxiliary, so ordinary tests do not repeat the default marker.
    product = (
        _declaration(item, "runtime_product", ("program", "kernel", "auxiliary"), "_st_runtime_product") or "program"
    )
    if product == "kernel":
        mode = execution_mode(item)
    else:
        if any(True for _ in item.iter_markers("execution_mode")):
            raise ValueError(f"{item.nodeid}: execution_mode is Kernel-only")
        mode = None
    runtimes = {
        _declaration(item, "runtime", (*RUNTIMES, "common")),
        _declaration(item, "coverage_runtime", (*RUNTIMES, "common")),
        getattr(getattr(item, "cls", None), "_st_runtime", None),
        getattr(getattr(item, "callspec", None), "params", {}).get("runtime"),
    } - {None}
    if len(runtimes) != 1 or not runtimes <= {*RUNTIMES, "common"}:
        raise ValueError(f"{item.nodeid}: missing or conflicting feature classification")
    if environment == "torch_npu" and product != "kernel":
        raise ValueError(f"{item.nodeid}: framework consumers must declare Kernel")
    result = {"product": product, "runtime": runtimes.pop(), "mode": mode, "allowed_skip": None}
    declarations = list(item.iter_markers("coverage_skip"))
    if len(declarations) > 1:
        raise ValueError(f"{item.nodeid}: conflicting skip exceptions")
    if declarations:
        marker = declarations[0]
        reason = marker.kwargs.get("reason")
        platforms = marker.kwargs.get("platforms")
        if (
            marker.args
            or set(marker.kwargs) != {"reason", "platforms"}
            or not isinstance(reason, str)
            or not reason.strip()
            or not isinstance(platforms, (list, tuple))
            or not platforms
            or any(p not in ("a2a3", "a5") for p in platforms)
        ):
            raise ValueError(f"{item.nodeid}: skip exception requires exact reason and explicit platforms")
        if environment == "torch_npu":
            raise ValueError(f"{item.nodeid}: framework skip exceptions are forbidden")
        if platform in platforms:
            result["allowed_skip"] = reason
    return result


def group(feature):
    """Program and auxiliary groups are mode-free; execution mode only splits Kernel."""
    if feature["product"] == "kernel":
        return f"kernel/{feature['runtime']}/{feature['mode']}"
    return f"{feature['product']}/{feature['runtime']}"


def _partition(config):
    """Return the partition the CI invocation declares; never infer it from -m spelling."""
    return config.getoption("--coverage-partition", default="all")


def remember_candidates(config, items):
    """Snapshot canonical scope before pytest applies ad-hoc -k/-m filters."""
    if not config.getoption("--verify-feature-coverage", default=False):
        return
    if config.getoption("--environment-parent-plan", default=None):
        return
    platform = config.getoption("--platform")
    if platform not in ("a2a3", "a5"):
        raise ValueError("Feature qualification requires an onboard platform")
    environment = config.getoption("--test-environment")
    partition = _partition(config)
    candidates = {}
    for item in items:
        sdma = bool(item.get_closest_marker("sdma"))
        fault = bool(item.get_closest_marker("sdma_fault"))
        member = {"all": True, "main": not sdma and not fault, "sdma": sdma and not fault, "sdma_fault": fault}[
            partition
        ]
        if member:
            candidates[item.nodeid] = features(item, platform, environment)
    config._feature_candidates = candidates


def policy(config, cases):
    platform = config.getoption("--platform")
    environment = config.getoption("--test-environment")
    manual = config.getoption("--manual")
    partition = _partition(config)
    runtime = config.getoption("--runtime")
    required = set()
    gaps = set()
    for rt in RUNTIMES:
        if runtime and rt != runtime:
            continue
        if environment == "default":
            required.add(f"program/{rt}")
        required.add(f"kernel/{rt}/eager")
        capture = f"kernel/{rt}/capture_replay"
        if environment == "torch_npu" and any(group(value) == capture for value in config._feature_candidates.values()):
            required.add(capture)
        elif environment == "default" and platform in ("a2a3", "a5") and rt == "host_build_graph":
            required.add(capture)
        else:
            gaps.add(capture)
    baseline = set(required)
    candidates = config._feature_candidates
    eligible = {group(value) for value in candidates.values() if value["allowed_skip"] is None}
    narrowed = (
        manual == "only"
        or partition in ("sdma", "sdma_fault")
        or config.getoption("--level") is not None
        or config.getoption("--exclude-level") in (2, 3)
    )
    if narrowed:
        required &= {group(value) for value in candidates.values()}
    missing = required - eligible
    if missing:
        raise ValueError(f"Required feature groups absent: {sorted(missing)}")
    expected = {
        nodeid for nodeid, value in candidates.items() if group(value) in required and value["allowed_skip"] is None
    }
    selected = {case["nodeid"] for case in cases}
    if expected - selected:
        raise ValueError(f"Required feature cases deselected: {sorted(expected - selected)}")
    conflicts = {case["nodeid"]: group(case["feature"]) for case in cases if group(case["feature"]) in gaps}
    if conflicts:
        raise ValueError(f"Tests claim a feature not delivered by the declared coverage policy: {conflicts}")
    return {
        "scope": {
            "platform": platform,
            "environment": environment,
            "manual": manual,
            "partition": partition,
            "runtime": runtime,
            "level": config.getoption("--level"),
            "exclude_level": config.getoption("--exclude-level"),
        },
        "required_groups": sorted(required),
        "required_nodeids": sorted(expected),
        "not_in_scope": sorted(baseline - required),
        "not_delivered": sorted(gaps),
    }


def allows_skip(plan, target, case):
    if plan["environment"] != "default" or target["nodeid"] in plan.get("coverage", {}).get("required_nodeids", []):
        return False
    reason = target.get("feature", {}).get("allowed_skip")
    skipped = case.find("skipped")
    return bool(
        reason and skipped is not None and skipped.get("type") == "pytest.skip" and skipped.get("message") == reason
    )


def summary(plan, observed, recovered, executions, invalid):
    groups = {}
    for target in plan["cases"]:
        key = group(target["feature"])
        counts = groups.setdefault(
            key, {"selected": 0, "executed": 0, "passed": 0, "failed": 0, "skipped": 0, "missing": 0}
        )
        counts["selected"] += 1
        nodeid = target["nodeid"]
        counts["executed"] += int(executions.get(nodeid, False))
        attempts = observed.get(nodeid, [])
        if any(outcome == "failed" for _, outcome in attempts):
            final = "failed"
        elif nodeid in invalid or not attempts:
            final = "missing"
        elif nodeid in recovered:
            final = "passed"
        else:
            final = attempts[-1][1]
        counts[final] += 1
    return {
        "groups": groups,
        "scope": plan["coverage"]["scope"],
        "not_in_scope": plan["coverage"]["not_in_scope"],
        "not_delivered": plan["coverage"]["not_delivered"],
    }
