# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A5 debug profiles retain independent build state."""

from unittest.mock import patch

import pytest

from simpler_setup import runtime_builder as rb


@pytest.fixture(autouse=True)
def _clear_profile_environment(monkeypatch):
    monkeypatch.delenv("SIMPLER_AICORE_BUILD_PROFILE", raising=False)


@patch("simpler_setup.runtime_builder.RuntimeCompiler")
def test_dev_profile_does_not_overwrite_release_paths(_compiler):
    release = rb.RuntimeBuilder("a5")
    dev = rb.RuntimeBuilder("a5", aicore_build_profile="dev")
    assert dev._LIB_DIR == release._LIB_DIR / "profiles" / "dev"
    assert dev._CACHE_DIR == release._CACHE_DIR / "profiles" / "dev"
    assert rb.RuntimeBuilder("a5")._LIB_DIR == release._LIB_DIR


@pytest.mark.parametrize("platform,profile", [("a5", "typo"), ("a2a3", "dev"), ("a5sim", "dev")])
def test_invalid_profile_fails_before_compiler_creation(platform, profile):
    with patch("simpler_setup.runtime_builder.RuntimeCompiler") as compiler:
        with pytest.raises(ValueError):
            rb.RuntimeBuilder(platform, aicore_build_profile=profile)
        compiler.get_instance.assert_not_called()


@patch("simpler_setup.runtime_builder.RuntimeCompiler")
def test_environment_selects_dev_for_build_and_lookup(_compiler, monkeypatch):
    monkeypatch.setenv("SIMPLER_AICORE_BUILD_PROFILE", "dev")
    selected = rb.RuntimeBuilder("a5")
    explicit = rb.RuntimeBuilder("a5", aicore_build_profile="dev")
    release = rb.RuntimeBuilder("a5", aicore_build_profile="release")
    assert selected.aicore_build_profile == "dev"
    assert selected._LIB_DIR == explicit._LIB_DIR == release._LIB_DIR / "profiles" / "dev"
    assert selected._CACHE_DIR == explicit._CACHE_DIR == release._CACHE_DIR / "profiles" / "dev"


def test_invalid_environment_fails_before_compiler_creation(monkeypatch):
    monkeypatch.setenv("SIMPLER_AICORE_BUILD_PROFILE", "typo")
    with patch("simpler_setup.runtime_builder.RuntimeCompiler") as compiler:
        with pytest.raises(ValueError, match="Unknown AICore build profile"):
            rb.RuntimeBuilder("a5")
        compiler.get_instance.assert_not_called()


@pytest.mark.parametrize("build", [False, True])
@patch("simpler_setup.runtime_builder.RuntimeCompiler")
def test_dev_rejects_unvalidated_runtime_before_lookup(_compiler, build):
    builder = rb.RuntimeBuilder("a5", aicore_build_profile="dev")
    with patch.object(builder, "_validate_runtime") as validate:
        with pytest.raises(ValueError, match="supports only A5 host_build_graph"):
            builder.get_binaries("tensormap_and_ringbuffer", build=build)
        validate.assert_not_called()
