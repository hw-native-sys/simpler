# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Strict CI receipts reject skipped, missing, duplicated or foreign cases."""

import json
import xml.etree.ElementTree as ET

import pytest

from simpler_setup.test_environments import _plan_digest, _source, verify_results


def _write(tmp_path, names, outcome=None, *, platform="a2a3"):
    plan = {
        "source": _source(),
        "environment": "torch_npu",
        "platform": platform,
        "cases": [{"nodeid": name, "classname": "tests.test_example", "name": name.split("::")[-1]} for name in names],
    }
    plan_path = tmp_path / "results.plan.json"
    plan_path.write_text(json.dumps(plan))
    suite = ET.Element("testsuite")
    for planned in plan["cases"]:
        case = ET.SubElement(suite, "testcase", classname=planned["classname"], name=planned["name"])
        properties = ET.SubElement(case, "properties")
        for name, value in (
            ("simpler_nodeid", planned["nodeid"]),
            ("simpler_source", plan["source"]),
            ("simpler_environment", plan["environment"]),
            ("simpler_plan", _plan_digest(plan)),
        ):
            ET.SubElement(properties, "property", name=name, value=value)
        if outcome:
            ET.SubElement(case, outcome)
    path = tmp_path / "results.xml"
    ET.ElementTree(suite).write(path)
    return plan_path, path


@pytest.mark.parametrize("platform", ["a2a3", "a5"])
def test_collected_environment_plan_passes(tmp_path, capsys, platform):
    verify_results(
        *_write(
            tmp_path, ["tests/test_example.py::test_first", "tests/test_example.py::test_second"], platform=platform
        )
    )
    assert "TEST_ENVIRONMENT_VERIFIED " in capsys.readouterr().out


@pytest.mark.parametrize("outcome", ["skipped", "error", "failure"])
def test_environment_plan_requires_actual_passes(tmp_path, outcome):
    with pytest.raises(ValueError, match="did not pass"):
        verify_results(*_write(tmp_path, ["tests/test_example.py::test_first"], outcome))


@pytest.mark.parametrize("mutation", ["empty", "missing", "duplicate", "foreign"])
def test_environment_plan_requires_exact_nodeids(tmp_path, mutation):
    plan, path = _write(tmp_path, ["tests/test_example.py::test_first", "tests/test_example.py::test_second"])
    tree = ET.parse(path)
    root = tree.getroot()
    if mutation == "empty":
        root.clear()
    elif mutation == "missing":
        root.remove(root[-1])
    elif mutation == "duplicate":
        root.append(root[0])
    else:
        nodeid_property = root[0].find("properties/property")
        assert nodeid_property is not None
        nodeid_property.set("value", "tests/foreign.py::test_foreign")
    tree.write(path)
    with pytest.raises(ValueError, match="not executed|foreign or duplicate"):
        verify_results(plan, path)


@pytest.mark.parametrize("outcome", ["failures", "errors", "skipped"])
def test_environment_plan_rejects_suite_level_errors(tmp_path, outcome):
    plan, path = _write(tmp_path, ["tests/test_example.py::test_first"])
    tree = ET.parse(path)
    tree.getroot().set(outcome, "1")
    tree.write(path)
    with pytest.raises(ValueError, match="reports a failure"):
        verify_results(plan, path)
