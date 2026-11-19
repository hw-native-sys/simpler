# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import hashlib

import pytest
from simpler.ccu import ccu_sdk_root, runtime_build_identity


@pytest.mark.parametrize("home", [None, ""])
def test_ccu_sdk_reports_missing_environment(monkeypatch, home):
    if home is None:
        monkeypatch.delenv("ASCEND_HOME_PATH", raising=False)
    else:
        monkeypatch.setenv("ASCEND_HOME_PATH", home)
    with pytest.raises(FileNotFoundError, match="ASCEND_HOME_PATH"):
        ccu_sdk_root()


@pytest.mark.parametrize("machine", ["aarch64", "x86_64"])
@pytest.mark.parametrize("layout", ["root", "arch", "symlink"])
def test_runtime_build_identity_uses_host_sdk(tmp_path, monkeypatch, machine, layout):
    sdk = tmp_path / "cann"
    root = sdk if layout == "root" else sdk / f"{machine}-linux"
    header = root / "include/hcomm/ccu/ccu_res.h"
    header.parent.mkdir(parents=True)
    header.write_bytes(b"host SDK")
    acl = root / "include/acl/acl.h"
    acl.parent.mkdir()
    acl.touch()
    if layout == "symlink":
        (sdk / "include").symlink_to(root / "include", target_is_directory=True)
    isa = tmp_path / "pto-isa"
    isa_header = isa / "include/pto/pto-inst.hpp"
    isa_header.parent.mkdir(parents=True)
    isa_header.write_bytes(b"pinned ISA")
    monkeypatch.setenv("ASCEND_HOME_PATH", str(sdk))
    monkeypatch.setattr("platform.machine", lambda: machine)
    monkeypatch.setattr("simpler_setup.pto_isa.ensure_pto_isa_root", lambda: str(isa))

    assert runtime_build_identity() == (
        hashlib.sha256(b"host SDK").digest(),
        hashlib.sha256(b"pinned ISA").digest(),
    )


@pytest.mark.parametrize("machine,foreign", [("aarch64", "x86_64"), ("x86_64", "aarch64")])
def test_ccu_sdk_does_not_select_another_host_architecture(tmp_path, monkeypatch, machine, foreign):
    for name in ("include/hcomm/ccu/ccu_res.h", "include/acl/acl.h"):
        header = tmp_path / f"{foreign}-linux" / name
        header.parent.mkdir(parents=True, exist_ok=True)
        header.touch()
    monkeypatch.setenv("ASCEND_HOME_PATH", str(tmp_path))
    monkeypatch.setattr("platform.machine", lambda: machine)

    with pytest.raises(FileNotFoundError, match="CANN CCU SDK requires") as error:
        ccu_sdk_root()
    assert f"{machine}-linux" in str(error.value)


def test_ccu_sdk_prefers_the_active_root_layout(tmp_path, monkeypatch):
    for root in (tmp_path, tmp_path / "x86_64-linux"):
        for name in ("include/hcomm/ccu/ccu_res.h", "include/acl/acl.h"):
            header = root / name
            header.parent.mkdir(parents=True, exist_ok=True)
            header.touch()
    monkeypatch.setenv("ASCEND_HOME_PATH", str(tmp_path))
    monkeypatch.setattr("platform.machine", lambda: "x86_64")

    assert ccu_sdk_root() == tmp_path
