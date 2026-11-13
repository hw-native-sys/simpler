# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""WSE discovery and cache contracts stay independent of A5."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from simpler.callable_identity import build_chip_callable_descriptor
from simpler.task_interface import ChipCallable

from simpler_setup import runtime_builder
from simpler_setup.kernel_compiler import KernelCompiler
from simpler_setup.platform_info import (
    discover_runtimes,
    parse_platform,
    to_platform,
)
from simpler_setup.pto_isa import ensure_pto_isa_root, read_pto_isa_pin
from simpler_setup.runtime_builder import RuntimeBuilder
from simpler_setup.runtime_compiler import RuntimeCompiler


def test_wse_supports_only_sim_hbg():
    assert parse_platform("wsesim") == ("wse", "sim")
    assert to_platform("wse", "sim") == "wsesim"
    assert discover_runtimes("wse") == ["host_build_graph"]
    with pytest.raises(ValueError):
        parse_platform("wse")
    builder = RuntimeBuilder("wsesim")
    with pytest.raises(ValueError, match="not available"):
        builder._validate_runtime("tensormap_and_ringbuffer")
    assert builder._resolve_build_pto_isa_commit() == read_pto_isa_pin()


def test_wse_orchestration_uses_own_headers_and_identity():
    wse = KernelCompiler("wsesim")
    a5 = KernelCompiler("a5sim")
    includes = wse.get_orchestration_include_dirs("host_build_graph")
    assert any("/src/wse/" in str(p) for p in includes)
    assert all("/src/a5/" not in str(p) for p in includes)
    assert wse.incore_compile_cache_token("") != a5.incore_compile_cache_token("aiv")
    chip = ChipCallable.build(signature=[], func_name="empty", binary=b"identical-bytes", children=[])
    assert build_chip_callable_descriptor(
        target=chip, platform="wsesim", runtime="host_build_graph"
    ) != build_chip_callable_descriptor(target=chip, platform="a5sim", runtime="host_build_graph")


def test_source_manifest_matches_copied_files():
    root = Path(__file__).resolve().parents[3]
    manifest = json.loads((root / "src/wse/source-copy-manifest.json").read_text())
    for entry in manifest["files"]:
        assert hashlib.sha256((root / entry["destination"]).read_bytes()).hexdigest() == entry["sha256"]


def test_real_scene_cache_keeps_wse_and_a5_separate(tmp_path, monkeypatch):
    from simpler_setup import SceneTestCase, scene_test  # noqa: PLC0415
    from simpler_setup.scene_test import clear_compile_cache  # noqa: PLC0415

    kernel = tmp_path / "kernel.cpp"
    kernel.write_text('extern "C" void kernel_entry() {}\n')
    orchestration = tmp_path / "orchestration.cpp"
    orchestration.write_text('extern "C" void empty() {}\n')

    @scene_test(level=2, runtime="host_build_graph")
    class CacheProbe(SceneTestCase):
        CALLABLE = {
            "orchestration": {"source": str(orchestration), "function_name": "empty", "signature": []},
            "incores": [{"func_id": 0, "source": str(kernel), "core_type": "", "signature": []}],
        }

    cases, originals = {}, {}
    for platform, core_type in (("a5sim", "aiv"), ("wsesim", "")):
        case = cases[platform] = CacheProbe()
        case.CALLABLE = {
            **CacheProbe.CALLABLE,
            "incores": [{**CacheProbe.CALLABLE["incores"][0], "core_type": core_type}],
        }
        originals[platform] = case.build_callable(platform)
        assert case.build_callable(platform) is originals[platform]
    clear_compile_cache()

    def unexpected_compile(*args, **kwargs):
        pytest.fail("complete persistent cache entry was recompiled")

    monkeypatch.setattr(KernelCompiler, "compile_incore", unexpected_compile)
    monkeypatch.setattr(KernelCompiler, "compile_orchestration", unexpected_compile)
    descriptors = {}
    for platform, original in originals.items():
        cached = cases[platform].build_callable(platform)
        descriptors[platform] = build_chip_callable_descriptor(
            target=cached, platform=platform, runtime="host_build_graph"
        )
        assert descriptors[platform] == build_chip_callable_descriptor(
            target=original, platform=platform, runtime="host_build_graph"
        )
    assert descriptors["a5sim"] != descriptors["wsesim"]
    clear_compile_cache()


def test_runtime_components_follow_platform_targets():
    wse = RuntimeCompiler("wsesim")
    assert wse.aicpu_target is None
    assert wse.host_target is not None and wse.aicore_target is not None
    for platform in ("a2a3sim", "a5sim"):
        compiler = RuntimeCompiler(platform)
        assert all(getattr(compiler, name + "_target") is not None for name in ("host", "aicpu", "aicore"))


def test_wse_binary_lookup_uses_two_components(tmp_path, monkeypatch):
    builder = RuntimeBuilder("wsesim")
    monkeypatch.setattr(builder, "_requires_pto_isa_metadata_validation", lambda: False)
    monkeypatch.setattr(builder, "_resolve_sim_context_path", lambda: None)
    for target in ("host", "aicore"):
        path = tmp_path / getattr(builder._runtime_compiler, f"{target}_target").get_binary_name()
        path.write_bytes(b"lookup-only")
    bins = builder._lookup_binaries("host_build_graph", tmp_path)
    assert bins.aicpu_path is None
    assert bins.host_path.is_file() and bins.aicore_path.is_file()


def test_a5_binary_lookup_still_requires_aicpu(tmp_path, monkeypatch):
    builder = RuntimeBuilder("a5sim")
    monkeypatch.setattr(builder, "_requires_pto_isa_metadata_validation", lambda: False)
    for target in ("host", "aicore"):
        (tmp_path / getattr(builder._runtime_compiler, f"{target}_target").get_binary_name()).touch()
    with pytest.raises(FileNotFoundError, match="libaicpu_kernel"):
        builder._lookup_binaries("host_build_graph", tmp_path)


def test_wse_cache_keeps_runtime_and_pto_identity(monkeypatch):
    monkeypatch.setattr(runtime_builder, "_runtime_source_stamp", lambda _: "source-content")
    builder = RuntimeBuilder("wsesim")
    stamp = builder._build_cache_stamp("fixed-pto")
    assert stamp == "source-content:pto-isa=fixed-pto"
    assert stamp == builder._build_cache_stamp("fixed-pto")


def test_runtime_source_cache_preserves_existing_input_scope(tmp_path):
    from simpler_setup.runtime_builder import _runtime_source_stamp  # noqa: PLC0415

    for arch in ("common", "a2a3", "a5", "wse"):
        path = tmp_path / "src" / arch / "kernel.h"
        path.parent.mkdir(parents=True)
        path.write_text("first")
    before = _runtime_source_stamp(tmp_path)
    (tmp_path / "src/wse/kernel.h").write_text("second")
    after = _runtime_source_stamp(tmp_path)
    assert before != after
    (tmp_path / "src/common/kernel.h").write_text("shared update")
    assert _runtime_source_stamp(tmp_path) != after


@pytest.mark.parametrize("gcc15", [False, True])
def test_automatic_platform_discovery_uses_shared_host_compilers(monkeypatch, gcc15):
    from simpler_setup import build_runtimes  # noqa: PLC0415

    available = {"gcc", "g++"} | ({"gcc-15", "g++-15"} if gcc15 else set())
    monkeypatch.setattr(build_runtimes.shutil, "which", lambda name: name if name in available else None)
    platforms = build_runtimes.detect_buildable_platforms()
    assert platforms == ["a2a3sim", "a5sim", "wsesim"]


@pytest.mark.parametrize("core_type", ["aic", "aiv"])
def test_wse_target_keeps_peer_semantics_on_reused_thread(tmp_path, core_type):
    from tests.ut.py.test_cpu_sim_target import _KERNEL, _RUN  # noqa: PLC0415

    isa_root = ensure_pto_isa_root()
    source = tmp_path / "target.cpp"
    source.write_text(_KERNEL)
    artifacts = {}
    tokens = {}
    for platform in ("a2a3sim", "a5sim", "wsesim"):
        compiler = KernelCompiler(platform)
        role = "" if platform == "wsesim" else core_type
        binary = compiler.compile_incore(str(source), core_type=role, pto_isa_root=isa_root)
        path = tmp_path / f"{platform}.so"
        path.write_bytes(binary)
        artifacts[platform] = str(path)
        tokens[platform] = compiler.incore_compile_cache_token(role)
    assert len({json.dumps(token, sort_keys=True) for token in tokens.values()}) == 3

    expected = {"a2a3sim": [1.0] * 8 + [2.0, 0.0], "a5sim": [0.0] * 8 + [5.0, 17.0], "wsesim": [0.0] * 8 + [5.0, 17.0]}
    env = dict(os.environ)
    env.pop("PTO_CPU_SIM_ARCH", None)
    for order in (("a2a3sim", "a5sim", "wsesim"), ("wsesim", "a5sim", "a2a3sim")):
        result = subprocess.run(
            [sys.executable, "-c", _RUN, *(artifacts[p] for p in order)],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
        assert json.loads(result.stdout) == [[p, expected[p]] for p in order * 2]
