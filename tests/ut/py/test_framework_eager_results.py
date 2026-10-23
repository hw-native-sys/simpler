# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Strict CI receipts reject skipped, missing, duplicated or foreign cases."""

import importlib.util
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[3] / ".github/actions/setup-framework-eager/verify_results.py"
_SPEC = importlib.util.spec_from_file_location("framework_eager_results", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _write(path, names, outcome=None):
    suite = ET.Element("testsuite")
    for nodeid in names:
        classname, name = nodeid.split("::")
        case = ET.SubElement(suite, "testcase", classname=classname, name=name)
        if outcome:
            ET.SubElement(case, outcome)
    ET.ElementTree(suite).write(path)
    return path


def test_exact_framework_matrix_passes(tmp_path, capsys):
    _MODULE.verify_results(_write(tmp_path / "results.xml", sorted(_MODULE.EXPECTED)))
    assert "FRAMEWORK_CI_VERIFIED " in capsys.readouterr().out


@pytest.mark.parametrize("outcome", ["skipped", "error", "failure"])
def test_framework_matrix_requires_actual_passes(tmp_path, outcome):
    with pytest.raises(ValueError, match="did not pass"):
        _MODULE.verify_results(_write(tmp_path / "results.xml", sorted(_MODULE.EXPECTED), outcome))


@pytest.mark.parametrize("mutation", ["empty", "missing", "duplicate", "foreign"])
def test_framework_matrix_requires_exact_nodeids(tmp_path, mutation):
    names = sorted(_MODULE.EXPECTED)
    if mutation == "empty":
        names = []
    elif mutation == "missing":
        names.pop()
    elif mutation == "duplicate":
        names[-1] = names[0]
    else:
        names[-1] = names[-1].replace("test_framework_queue_orders_public_worker", "test_foreign")
    with pytest.raises(ValueError, match="four framework eager nodeids"):
        _MODULE.verify_results(_write(tmp_path / "results.xml", names))


@pytest.mark.parametrize("outcome", ["failures", "errors", "skipped"])
def test_framework_matrix_rejects_suite_level_errors(tmp_path, outcome):
    path = _write(tmp_path / "results.xml", sorted(_MODULE.EXPECTED))
    tree = ET.parse(path)
    tree.getroot().set(outcome, "1")
    tree.write(path)
    with pytest.raises(ValueError, match="suite reports"):
        _MODULE.verify_results(path)
