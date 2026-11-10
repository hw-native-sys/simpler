# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Runtime cache validity must follow source bytes, including exported snapshots."""

import json
import os

import pytest

from simpler_setup import runtime_builder as rb


def stamp(root, monkeypatch):
    monkeypatch.setattr(rb, "PROJECT_ROOT", root)
    builder = rb.RuntimeBuilder.__new__(rb.RuntimeBuilder)
    return builder._build_cache_stamp("pto-pin")


def test_snapshot_and_documentation_edits_preserve_cache(tmp_path, monkeypatch):
    source = tmp_path / "src" / "runtime.h"
    source.parent.mkdir()
    source.write_text("int version = 1;\n")
    before = stamp(tmp_path, monkeypatch)
    assert before, "Exported source must have a stable build identity without Git"
    (tmp_path / "README.md").write_text("Documentation update")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git/HEAD").write_text("different-commit")
    assert stamp(tmp_path, monkeypatch) == before
    cache = tmp_path / "cache"
    rb._invalidate_cache_if_stale(cache, before)
    (cache / "executor.o").write_bytes(b"existing object")
    rb._invalidate_cache_if_stale(cache, stamp(tmp_path, monkeypatch))
    assert (cache / "executor.o").read_bytes() == b"existing object"


def test_same_mtime_header_edit_and_added_removed_source_invalidate(tmp_path, monkeypatch):
    source = tmp_path / "src" / "runtime.h"
    source.parent.mkdir()
    source.write_text("int version = 1;\n")
    before = stamp(tmp_path, monkeypatch)
    times = source.stat()
    source.write_text("int version = 2;\n")
    os.utime(source, ns=(times.st_atime_ns, times.st_mtime_ns))
    changed = stamp(tmp_path, monkeypatch)
    assert changed != before
    extra = source.with_name("new.cpp")
    extra.write_text("int new_function();")
    assert stamp(tmp_path, monkeypatch) != changed
    extra.unlink()
    assert stamp(tmp_path, monkeypatch) == changed


def test_regular_source_edit_retains_incremental_objects(tmp_path):
    cache = tmp_path / "cache"
    before = {"src/host.cpp": {"sha256": "old", "mtime_ns": 100}}
    rb._invalidate_cache_if_stale(cache, "first:pto-isa=pin", inputs=before)
    obj = cache / "executor.o"
    obj.write_bytes(b"unrelated device object")
    os.utime(obj, ns=(200, 200))
    after = {"src/host.cpp": {"sha256": "new", "mtime_ns": 300}}
    rb._invalidate_cache_if_stale(cache, "second:pto-isa=pin", inputs=after)
    assert obj.read_bytes() == b"unrelated device object"


def test_backdated_edit_still_cleans_objects(tmp_path):
    cache = tmp_path / "cache"
    before = {"src/common.h": {"sha256": "old", "mtime_ns": 100}}
    rb._invalidate_cache_if_stale(cache, "first:pto-isa=pin", inputs=before)
    obj = cache / "executor.o"
    obj.write_bytes(b"stale device object")
    os.utime(obj, ns=(200, 200))
    after = {"src/common.h": {"sha256": "new", "mtime_ns": 100}}
    rb._invalidate_cache_if_stale(cache, "second:pto-isa=pin", inputs=after)
    assert not obj.exists()


def test_compiler_replacement_cleans_even_with_same_source_stamp(tmp_path):
    cache = tmp_path / "cache"
    before = {"@toolchain/aicore/cxx_path": {"sha256": "compiler-v1", "mtime_ns": 0}}
    rb._invalidate_cache_if_stale(cache, "sources:pto-isa=pin", inputs=before)
    obj = cache / "executor.o"
    obj.write_bytes(b"old compiler object")
    after = {"@toolchain/aicore/cxx_path": {"sha256": "compiler-v2", "mtime_ns": 0}}
    rb._invalidate_cache_if_stale(cache, "sources:pto-isa=pin", inputs=after)
    assert not obj.exists()


def test_pin_change_cleans_even_with_identical_source_files(tmp_path):
    cache = tmp_path / "cache"
    files = {"src/runtime.cpp": {"sha256": "same", "mtime_ns": 100}}
    rb._invalidate_cache_if_stale(cache, "sources:pto-isa=old", inputs=files)
    obj = cache / "executor.o"
    obj.write_bytes(b"old ISA object")
    rb._invalidate_cache_if_stale(cache, "sources:pto-isa=new", inputs=files)
    assert not obj.exists()


def test_missing_input_manifest_fails_closed(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / rb._GIT_COMMIT_FILE).write_text("sources:pto-isa=pin")
    obj = cache / "executor.o"
    obj.write_bytes(b"unverified object")
    rb._invalidate_cache_if_stale(cache, "sources:pto-isa=pin", inputs={})
    assert not obj.exists()


@pytest.mark.parametrize("manifest", [[], {"stamp": "same"}, {"stamp": "same", "files": {"source": None}}])
def test_malformed_input_manifest_fails_closed(tmp_path, manifest):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / rb._GIT_COMMIT_FILE).write_text("same")
    (cache / ".runtime_inputs.json").write_text(json.dumps(manifest))
    obj = cache / "executor.o"
    obj.write_bytes(b"unverified object")
    rb._invalidate_cache_if_stale(cache, "same", inputs={})
    assert not obj.exists()
