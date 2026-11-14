# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Project includes must stay visible to the percounter isolation check."""

from tests.lint import check_percounter_fork_isolation as checker


def test_angle_project_headers_are_followed(tmp_path, monkeypatch):
    monkeypatch.setattr(checker, "REPO", tmp_path)
    root = tmp_path / "root.h"
    nested = tmp_path / "scheduler"
    nested.mkdir()
    leaf = nested / "forked.h"
    root.write_text("#include <scheduler/forked.h>\n#include <cstdint>\n")
    leaf.write_text("")
    reached, unresolved = checker.closure([root], [tmp_path])
    assert leaf in reached
    assert unresolved == {}


def test_unresolved_project_angle_header_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(checker, "REPO", tmp_path)
    (tmp_path / "scheduler").mkdir()
    root = tmp_path / "root.h"
    root.write_text("#include <scheduler/missing.h>\n")
    _, unresolved = checker.closure([root], [tmp_path])
    assert "scheduler/missing.h" in unresolved


def test_angle_header_does_not_search_including_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(checker, "REPO", tmp_path)
    local = tmp_path / "local"
    search = tmp_path / "search"
    local.mkdir()
    search.mkdir()
    root = local / "root.h"
    root.write_text("#include <tensor.h>\n")
    (local / "tensor.h").write_text("")
    correct = search / "tensor.h"
    correct.write_text("")
    reached, _ = checker.closure([root], [search])
    assert correct in reached
    assert local / "tensor.h" not in reached
