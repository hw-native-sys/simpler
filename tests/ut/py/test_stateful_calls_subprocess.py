# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Keep standalone stateful calls in pytest's dependency context."""

import json
import runpy
from pathlib import Path

import pytest

_CASE = Path(__file__).resolve().parents[3] / "tests/st/common/kernel_eager_state/test_stateful_calls.py"


@pytest.mark.parametrize("pythonpath", ["unset", "shadow"])
@pytest.mark.parametrize(
    ("platform", "runtime", "borrowed", "host_controls", "budget"),
    [
        ("a2a3sim", "host_build_graph", False, False, None),
        ("a5sim", "tensormap_and_ringbuffer", False, False, None),
        ("a5", "host_build_graph", True, True, 128 << 20),
    ],
)
def test_stateful_subprocess_preserves_imports_and_arguments(
    tmp_path, monkeypatch, pythonpath, platform, runtime, borrowed, host_controls, budget
):
    parent = tmp_path / "parent"
    child = tmp_path / "child"
    parent.mkdir()
    child.mkdir()
    dependency = "stateful_dependency_context"
    (parent / f"{dependency}.py").write_text('origin = "parent"\n')
    (child / f"{dependency}.py").write_text('origin = "shadow"\n')
    monkeypatch.syspath_prepend(str(parent))
    if pythonpath == "unset":
        monkeypatch.delenv("PYTHONPATH", raising=False)
    else:
        monkeypatch.setenv("PYTHONPATH", str(child))

    result_path = tmp_path / "result.json"
    entry = child / "entry.py"
    entry.write_text(
        "import json, sys\n"
        "from pathlib import Path\n"
        f"from {dependency} import origin\n"
        f"Path({str(result_path)!r}).write_text(json.dumps(dict(origin=origin, argv=sys.argv, name=__name__)))\n"
    )
    check = runpy.run_path(str(_CASE))["_check"]
    # The real launcher runs a child that reports its dependency and argument context.
    monkeypatch.setitem(check.__globals__, "__file__", str(entry))
    check(platform, [3], runtime, borrowed, host_controls, budget)

    result = json.loads(result_path.read_text())
    assert result["origin"] == "parent"
    assert result["name"] == "__main__"
    assert result["argv"] == [
        str(entry),
        platform,
        runtime,
        "3",
        str(int(borrowed)),
        str(int(host_controls)),
        str(budget) if budget is not None else "none",
    ]
