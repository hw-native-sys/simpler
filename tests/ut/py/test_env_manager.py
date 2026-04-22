# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for python/env_manager.py - environment variable cache."""

import pytest
from simpler import env_manager


@pytest.fixture(autouse=True)
def _reset_cache():
    """Reset env_manager cache between tests."""
    env_manager._cache.clear()
    yield
    env_manager._cache.clear()


class TestGet:
    def test_uncached_returns_none(self):
        assert env_manager.get("NONEXISTENT_VAR_12345") is None

    def test_after_ensure(self, monkeypatch):
        monkeypatch.setenv("TEST_ENV_VAR_XYZ", "hello")
        env_manager.ensure("TEST_ENV_VAR_XYZ")
        assert env_manager.get("TEST_ENV_VAR_XYZ") == "hello"


class TestEnsure:
    def test_returns_value_when_set(self, monkeypatch):
        monkeypatch.setenv("TEST_ENSURE_VAR", "value123")
        result = env_manager.ensure("TEST_ENSURE_VAR")
        assert result == "value123"

    def test_raises_when_unset(self, monkeypatch):
        monkeypatch.delenv("UNSET_VAR_99999", raising=False)
        with pytest.raises(EnvironmentError, match="not set"):
            env_manager.ensure("UNSET_VAR_99999")

    def test_raises_when_empty(self, monkeypatch):
        monkeypatch.setenv("EMPTY_VAR_TEST", "")
        with pytest.raises(EnvironmentError, match="not set"):
            env_manager.ensure("EMPTY_VAR_TEST")

    def test_caching(self, monkeypatch):
        monkeypatch.setenv("CACHED_VAR", "original")
        env_manager.ensure("CACHED_VAR")

        # Change the env var - cached value should persist
        monkeypatch.setenv("CACHED_VAR", "changed")
        result = env_manager.ensure("CACHED_VAR")
        assert result == "original"  # Returns cached, not re-read

    def test_caching_skips_none_check(self, monkeypatch):
        monkeypatch.setenv("CACHE_TEST_2", "val")
        env_manager.ensure("CACHE_TEST_2")

        # Even if we remove from env, cache returns the value
        monkeypatch.delenv("CACHE_TEST_2")
        result = env_manager.ensure("CACHE_TEST_2")
        assert result == "val"
