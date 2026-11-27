# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compile each runtime with its platform toolchain and verify the artifacts."""

import pytest


class TestRuntimeBuilderIntegration:
    """Integration tests that actually compile all platform x runtime combinations.

    Test parametrization is handled dynamically by conftest.py based on:
    - Available platforms discovered from src/*/platform/{onboard,sim}/
    - Available runtimes per architecture from src/{arch}/runtime/*/build_config.py
    - --platform filter if specified on command line
    """

    @pytest.fixture(autouse=True)
    def _reset_compiler_singleton(self):
        """Reset RuntimeCompiler singleton-per-platform cache so each test gets fresh instances."""
        from simpler_setup.runtime_compiler import RuntimeCompiler  # noqa: PLC0415

        yield
        RuntimeCompiler._instances.clear()

    def test_get_binaries_returns_valid_paths(self, platform, runtime_name):
        """get_binaries(build=True) produces RuntimeBinaries with existing files."""
        from simpler_setup.runtime_builder import RuntimeBinaries, RuntimeBuilder  # noqa: PLC0415

        builder = RuntimeBuilder(platform=platform)
        result = builder.get_binaries(runtime_name, build=True)

        assert isinstance(result, RuntimeBinaries)
        targets = ("host", "aicore") if platform == "wsesim" else ("host", "aicpu", "aicore")
        if "aicpu" not in targets:
            assert result.aicpu_path is None
        for label in targets:
            path = getattr(result, f"{label}_path")
            assert path is not None
            assert path.is_file(), f"{label} binary not found: {path}"
            assert path.stat().st_size > 0, f"{label} binary is empty: {path}"
