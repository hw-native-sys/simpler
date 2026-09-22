# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A scene-test class's ``CASES`` gate **every** item it owns, not just ``test_run``.

The root conftest deselects any item whose class declares ``CASES`` that match no
platform. That is a whole-class gate: a hand-written method on such a class
disappears with it, silently, and a method inventory cannot see the difference.
``tests/st/**/prepared_callable`` depends on this, where the intentional
registration-failure case lives in its own class and must stay selected while the
generic ``test_run`` inherited into that class must not run.

These cases drive the real hook and real pytest collection over a synthetic
module, so they fail if either the gate or the suppression idiom changes. They
touch no device and import no runtime.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]

# Re-export the real hooks under the synthetic rootdir, so the behaviour under
# test is the repository's own and not a restatement of it.
_CONFTEST = """
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("_real_root_conftest", r"{root}/conftest.py")
_real = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_real)

pytest_addoption = _real.pytest_addoption
pytest_collection_modifyitems = _real.pytest_collection_modifyitems
"""

_MODULE = '''
class _Base:
    """Stands in for SceneTestCase: owns the generic runner the gate protects."""

    CASES = []

    def test_run(self):
        pass


class TestMatching(_Base):
    CASES = [{"name": "c", "platforms": ["a5", "a5sim"]}]
    _st_runtime = "tensormap_and_ringbuffer"
    _st_level = 2

    def test_handwritten(self):
        pass


class TestNoMatchingCase(_Base):
    CASES = []
    _st_runtime = "tensormap_and_ringbuffer"
    _st_level = 2

    def test_handwritten(self):
        pass


class TestRunnerSuppressed(_Base):
    CASES = [{"name": "c", "platforms": ["a5", "a5sim"]}]
    _st_runtime = "tensormap_and_ringbuffer"
    _st_level = 2

    def test_run(self, *args, **kwargs):
        raise AssertionError("must not be collected")

    test_run.__test__ = False

    def test_handwritten(self):
        pass
'''


@pytest.fixture(scope="module")
def collected(tmp_path_factory):
    """Node ids pytest selects for the synthetic module under ``--platform a5``."""
    workdir = tmp_path_factory.mktemp("scene_gate")
    (workdir / "conftest.py").write_text(textwrap.dedent(_CONFTEST).format(root=_ROOT))
    (workdir / "test_synthetic_scene.py").write_text(textwrap.dedent(_MODULE))

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "--platform", "a5", "-p", "no:randomly"],
        cwd=workdir,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"collection failed:\n{result.stdout}\n{result.stderr}"
    return {line.split("::", 1)[1] for line in result.stdout.splitlines() if "::" in line}


def test_a_matching_class_keeps_its_hand_written_method(collected):
    assert "TestMatching::test_handwritten" in collected
    assert "TestMatching::test_run" in collected


def test_no_matching_case_deselects_the_whole_class(collected):
    """The gate that made an empty-CASES dedicated class silently untested."""
    assert "TestNoMatchingCase::test_run" not in collected
    assert "TestNoMatchingCase::test_handwritten" not in collected, (
        "CASES gate every item of the class, so an empty list drops hand-written methods too"
    )


def test_suppressing_the_runner_keeps_the_hand_written_method(collected):
    """The shape ``prepared_callable``'s registration-failure class relies on."""
    assert "TestRunnerSuppressed::test_run" not in collected, "__test__ = False has to stop collection"
    assert "TestRunnerSuppressed::test_handwritten" in collected, (
        "suppressing the inherited runner must not cost the class its own test"
    )


def test_the_real_registration_failure_class_uses_that_shape():
    """Tie the synthetic shape to the classes it was written for."""
    import ast  # noqa: PLC0415

    for arch in ("a5", "a2a3"):
        path = _ROOT / "tests/st" / arch / "tensormap_and_ringbuffer/prepared_callable/test_prepared_callable.py"
        tree = ast.parse(path.read_text(), filename=str(path))
        cls = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "TestPreparedCallableRegistrationFailure"
        )
        names = [n.name for n in cls.body if isinstance(n, ast.FunctionDef)]
        assert "test_prewarm_failure_rolls_back_registration" in names
        assert "test_run" in names, "the inherited runner has to be shadowed, not left inherited"
        assert any(
            isinstance(n, ast.Assign) and any(isinstance(t, ast.Attribute) and t.attr == "__test__" for t in n.targets)
            for n in cls.body
        ), "the shadow has to carry __test__ = False"
        assert any(
            isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "CASES" for t in n.targets)
            for n in cls.body
        ), "the class has to keep platform CASES metadata or the gate deselects it"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
