# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Collect-only inventory of the actual pytest selection and feature policy.

Load with ``-p simpler_setup.tools.collection_inventory --collection-inventory PATH``.
This plugin never runs fixtures or test bodies.
"""

import json
from collections import Counter
from pathlib import Path

import pytest

from simpler_setup import test_coverage


def pytest_addoption(parser):
    parser.addoption("--collection-inventory", help="Write the final collect-only selection as JSON")


def pytest_configure(config):
    if config.getoption("--collection-inventory") and not config.getoption("collectonly"):
        raise pytest.UsageError("--collection-inventory requires --collect-only")


@pytest.hookimpl(trylast=True)
def pytest_collection_finish(session):
    config = session.config
    path = config.getoption("--collection-inventory")
    if not path:
        return
    cases = []
    errors = []
    for item in session.items:
        case = {"nodeid": item.nodeid}
        try:
            case["feature"] = test_coverage.features(
                item, config.getoption("--platform"), config.getoption("--test-environment")
            )
        except ValueError as exc:
            errors.append(str(exc))
        cases.append(case)
    result = {
        "arguments": list(config.invocation_params.args),
        "collected": len(cases),
        "duplicates": sorted(key for key, count in Counter(c["nodeid"] for c in cases).items() if count > 1),
        "classification_errors": errors,
        "collection_errors": session.testsfailed,
        "cases": cases,
    }
    if config.getoption("--verify-feature-coverage") and not errors and not session.testsfailed:
        result["policy"] = test_coverage.policy(config, cases)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
