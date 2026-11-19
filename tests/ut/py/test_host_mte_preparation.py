# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host MTE scenes use the common precompiler and consume saved inputs."""

from types import SimpleNamespace

import pytest

from examples.a5.host_mte_domain import run as runner
from examples.a5.host_mte_domain import test_host_mte_domain as entrypoint
from simpler_setup import scene_test_cache
from simpler_setup.scene_test import clear_compile_cache
from simpler_setup.tools.scene_test_compile import compile_collected_scene_tests


class _CompilerBoundary:
    """Supply compiler output without requiring a device SDK in a CPU unit test."""

    _sanitizers = ""

    def __init__(self, platform):
        self.platform = platform

    def get_orchestration_include_dirs(self, runtime):
        return []

    def get_incore_include_dirs(self):
        return []

    def get_orchestration_cache_inputs(self, runtime):
        return [], []

    def compile_cache_token(self, runtime, core_types):
        return {"runtime": runtime, "test": "host-mte-preparation"}

    def incore_compile_cache_token(self, core_type):
        return {"core_type": core_type, "test": "host-mte-preparation"}

    def compile_orchestration(self, runtime, source):
        return b"prepared-orchestration"

    def compile_incore(self, source, **kwargs):
        return b"prepared-incore"


def test_common_precompiler_prepares_all_host_mte_scenes(monkeypatch, tmp_path):
    pto_root = tmp_path / "pto-isa"
    (pto_root / "include" / "pto").mkdir(parents=True)
    monkeypatch.setattr("simpler_setup.kernel_compiler.KernelCompiler", _CompilerBoundary)
    monkeypatch.setattr("simpler_setup.pto_isa.ensure_pto_isa_root", lambda: str(pto_root))
    monkeypatch.setattr(scene_test_cache, "KERNEL_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(runner, "ARTIFACTS", tmp_path / "inputs")
    scenes = [value for name, value in vars(entrypoint).items() if name.startswith("Test")]
    items = [SimpleNamespace(cls=scene, iter_markers=lambda name=None: iter(())) for scene in scenes]
    clear_compile_cache()
    try:
        # The simulated compiler route bypasses hardware ELF extraction. No Worker is created.
        assert compile_collected_scene_tests(items, "a5sim", max_workers=1) == (3, [])
        clear_compile_cache()

        def forbidden(*_args, **_kwargs):
            pytest.fail("scene execution must reuse cached kernels and saved inputs")

        monkeypatch.setattr(_CompilerBoundary, "compile_orchestration", forbidden)
        monkeypatch.setattr(_CompilerBoundary, "compile_incore", forbidden)
        monkeypatch.setattr(runner, "generate_inputs", forbidden)
        for scene in scenes:
            assert "peer_read" in scene().build_callable("a5sim")
            inputs = runner.load_inputs(scene._st_runtime, runner.ARTIFACTS)
            assert len(inputs) == 12
            assert inputs["forward_1_2"][0] == 220000
            assert inputs["reverse_0_1"][0] == 1100000
    finally:
        clear_compile_cache()


@pytest.mark.parametrize("runtime", runner.RUNTIMES)
@pytest.mark.parametrize("artifacts", ["missing", "corrupt", "ready"])
def test_inputs_never_fall_back_to_generation(monkeypatch, tmp_path, runtime, artifacts):
    if artifacts != "missing":
        runner.prepare_inputs(runtime, tmp_path)
        if artifacts == "corrupt":
            (tmp_path / runtime / "inputs.npz").write_bytes(b"corrupted")

    def forbidden(*_args, **_kwargs):
        pytest.fail("scene execution must not generate inputs")

    monkeypatch.setattr(runner, "generate_inputs", forbidden)
    if artifacts == "ready":
        inputs = runner.load_inputs(runtime, tmp_path)
        assert inputs["forward_1_2"].shape == (16384,)
        assert inputs["forward_1_2"][0] == 220000
    else:
        with pytest.raises(FileNotFoundError if artifacts == "missing" else ValueError):
            runner.load_inputs(runtime, tmp_path)
