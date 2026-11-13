# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""WSE compiler inputs use the ordinary include closure and cache schema."""

import ctypes

import pytest

from simpler_setup import scene_test_cache as cache
from simpler_setup.kernel_compiler import KernelCompiler


@pytest.mark.parametrize("platform", ["a2a3sim", "a5sim", "wsesim"])
@pytest.mark.parametrize("role,macro", [("", None), ("aic", "-D__DAV_CUBE__"), ("aiv", "-D__DAV_VEC__")])
def test_sim_role_macros_follow_the_explicit_argument(platform, role, macro):
    compiler = KernelCompiler(platform)
    flags = compiler._sim_incore_compile_flags(role)
    actual = {flag for flag in flags if flag in {"-D__DAV_CUBE__", "-D__DAV_VEC__"}}
    assert actual == ({macro} if macro else set())
    token = compiler.incore_compile_cache_token(role)
    assert token["core_type"] == role and token["flags"] == flags
    assert set(token) == {"schema", "logic", "identity", "core_type", "flags", "linker", "pto_isa_commit"}


def test_only_included_wse_headers_invalidate_real_kernel_cache(tmp_path, monkeypatch):
    compiler = KernelCompiler("wsesim")
    include = tmp_path / "platform/include"
    operations = tmp_path / "platform/sim/operations"
    (include / "wse").mkdir(parents=True)
    operations.mkdir(parents=True)
    public = include / "wse/probe.h"
    public.write_text('#include "../../sim/operations/semantic.h"\n')
    semantic = operations / "semantic.h"
    semantic.write_text("inline int wse_value() { return 7; }\n")
    source = tmp_path / "kernel.cpp"
    source.write_text('#include <wse/probe.h>\nextern "C" int result() { return wse_value(); }\n')
    include_dirs = [*compiler.get_incore_include_dirs(), str(include)]
    metadata = {"platform": "wsesim", "compiler": compiler.incore_compile_cache_token("")}
    monkeypatch.setattr(cache, "KERNEL_CACHE_DIR", tmp_path / "cache")
    compiles = []

    def key():
        return cache.compile_incore_artifact_key(metadata, source, include_dirs)

    def compile_kernel():
        compiles.append(1)
        return compiler.compile_incore(str(source), core_type="", extra_include_dirs=[str(include)])

    def result(name, binary):
        path = tmp_path / (name + ".so")
        path.write_bytes(binary)
        lib = ctypes.CDLL(str(path))
        lib.result.argtypes = []
        lib.result.restype = ctypes.c_int
        return lib.result()

    first_key = key()
    first = cache.get_or_compile_incore(first_key, compile_kernel)
    assert result("first", first) == 7
    assert len(compiles) == 1
    (operations / "unused.h").write_text("unrelated WSE operation changed\n")
    assert key() == first_key
    assert cache.get_or_compile_incore(key(), compile_kernel) == first
    assert len(compiles) == 1

    peer_source = tmp_path / "peer.cpp"
    peer_source.write_text('extern "C" int result() { return 5; }\n')
    peers = {platform: KernelCompiler(platform) for platform in ("a2a3sim", "a5sim")}

    def peer_keys():
        return {
            platform: cache.compile_incore_artifact_key(
                {"platform": platform, "compiler": peer.incore_compile_cache_token("aiv")},
                peer_source,
                peer.get_incore_include_dirs(),
            )
            for platform, peer in peers.items()
        }

    before = peer_keys()
    semantic.write_text("inline int wse_value() { return 19; }\n")
    assert key() != first_key
    assert peer_keys() == before
    second = cache.get_or_compile_incore(key(), compile_kernel)
    assert result("second", second) == 19
    assert len(compiles) == 2
    semantic.write_text("inline int wse_value() { return 7; }\n")
    assert key() == first_key
    assert cache.get_or_compile_incore(key(), compile_kernel) == first
    assert len(compiles) == 2
