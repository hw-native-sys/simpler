# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Unit tests for the sanitizer-selection logic (pure functions, no toolchain)."""

from __future__ import annotations

import pytest

from simpler_setup import sanitizers as san


@pytest.mark.parametrize(
    "selection,expected",
    [
        ("none", ""),
        ("", ""),
        (None, ""),
        ("asan", "address,undefined"),
        ("ubsan", "undefined"),
        ("tsan", "thread"),
        ("address,leak", "address,leak"),  # raw tokens pass through unchanged
    ],
)
def test_resolve(selection, expected):
    assert san.resolve(selection) == expected


def test_validate_allows_compatible_combos(monkeypatch):
    monkeypatch.setattr(san.sys, "platform", "linux")  # tsan is Linux-only
    san.validate("address,undefined")  # asan + ubsan is fine
    san.validate("thread")  # tsan alone is fine on Linux
    san.validate("")  # empty is a no-op


@pytest.mark.parametrize("incompatible", ["address", "leak", "memory"])
def test_validate_rejects_thread_with_address_family(monkeypatch, incompatible):
    monkeypatch.setattr(san.sys, "platform", "linux")
    with pytest.raises(ValueError, match="thread"):
        san.validate(f"thread,{incompatible}")


def test_validate_rejects_tsan_off_linux(monkeypatch):
    monkeypatch.setattr(san.sys, "platform", "darwin")
    with pytest.raises(ValueError, match="Linux-only"):
        san.validate("thread")


@pytest.mark.parametrize(
    "tokens,expected",
    [
        ("", None),
        ("undefined", "libubsan.so"),
        ("address,undefined", "libasan.so"),  # asan runtime covers ubsan
        ("leak", "liblsan.so"),  # standalone LSan
        ("thread", "libtsan.so"),
    ],
)
def test_preload_lib(tokens, expected):
    assert san.preload_lib(tokens) == expected


@pytest.mark.parametrize(
    "platform,expected",
    [
        ("a2a3sim", "g++-15"),
        ("a5sim", "g++-15"),
        ("a2a3", "g++"),
        ("a5", "g++"),
    ],
)
def test_host_cxx(platform, expected):
    # Sim unifies on g++-15 (matches kernels); onboard host uses plain g++.
    assert san.host_cxx(platform) == expected


def test_preload_command_uses_platform_compiler():
    cmd = san.preload_command("address,undefined", "a2a3sim")
    assert "g++-15" in cmd and "libasan.so" in cmd
    assert san.preload_command("", "a2a3sim") is None
