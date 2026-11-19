# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Real pinned URMA manager and production host cleanup with stateful HCCL mocks."""

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from simpler_setup.pto_isa import ensure_pto_isa_root

ROOT = Path(__file__).resolve().parents[3]


def _require_compatible_cann_headers(cann, compiler):
    if not (cann / "include/hccl/hccl_channel.h").is_file():
        pytest.skip("requires compatible CANN headers")
    result = subprocess.run(
        compiler
        + ["-std=c++17", "-fsyntax-only", "-x", "c++", "-", f"-I{cann / 'include'}", f"-I{cann / 'pkg_inc/base'}"],
        input=(
            "#include <hccl/hccl.h>\n"
            "#include <hccl/hccl_res.h>\n"
            "#include <hccl/hccl_channel.h>\n"
            "int main() {\n"
            "  (void)&HcclChannelAcquireWithConfig;\n"
            "  static_assert(HCCL_CHANNEL_CONFIG_SHARED_QUEUE_TAG_MAX_LEN > 0);\n"
            "}\n"
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip("requires compatible CANN headers")


@pytest.fixture(scope="module")
def urma_lifecycle_binary(tmp_path_factory):
    build = tmp_path_factory.mktemp("urma_lifecycle")
    cann = Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/cann"))
    compiler = shlex.split(os.environ["CXX"]) if os.environ.get("CXX") else [shutil.which("g++-15") or "c++"]
    _require_compatible_cann_headers(cann, compiler)
    isa = Path(ensure_pto_isa_root())
    sdma = build / "pto/comm/async/sdma/sdma_workspace_manager.hpp"
    sdma.parent.mkdir(parents=True)
    sdma.write_text((ROOT / "tests/ut/py/support/unused_sdma_workspace_manager.h").read_text())
    log = build / "common/unified_log.h"
    log.parent.mkdir()
    log.write_text(
        "#pragma once\n"
        + "".join(f"#define LOG_{level}(...) ((void)0)\n" for level in ("ERROR", "WARN", "INFO", "DEBUG"))
    )
    includes = [
        build,
        isa / "include",
        isa / "pkg_inc",
        cann / "include",
        cann / "pkg_inc/base",
        ROOT / "src/common",
        ROOT / "src/common/platform/include",
        ROOT / "src/common/task_interface",
        ROOT / "src/a5/platform/include",
        ROOT / "src/a5/platform/onboard/host",
    ]
    binary = build / "urma_lifecycle"
    result = subprocess.run(
        compiler
        + [
            "-std=c++17",
            "-UNDEBUG",
            "-ffunction-sections",
            "-fdata-sections",
            "-Wl,--gc-sections",
            *[f"-I{p}" for p in includes],
            str(ROOT / "tests/ut/py/support/urma_lifecycle_host.cpp"),
            str(ROOT / "src/common/platform_comm/ccu/ccu_domain_owner.cpp"),
            str(ROOT / "src/common/platform_comm/ccu/ccu_run_lifecycle.cpp"),
            "-ldl",
            "-lrt",
            "-pthread",
            "-o",
            str(binary),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"URMA lifecycle fixture compilation failed:\n{result.stdout}\n{result.stderr}", pytrace=False)
    return binary


@pytest.mark.parametrize(
    "scenario",
    [
        "notify_registration",
        "channel_acquire",
        "destroy_failure",
        "context_upload",
        "late_import",
        "early_get_device",
        "peer_import",
        "peer_reserve",
        "peer_map",
        "success",
        "undrained",
        "cqe_pending",
        "base_destroy",
        "abort_unpublished",
        "release_destroy_failure",
        "finalize_live",
        "finalize_undrained",
        "overlap_subset",
        "overlap_reordered",
        "overlap_release_failure",
    ],
)
def test_urma_lifecycle(urma_lifecycle_binary, scenario):
    subprocess.run(
        [str(urma_lifecycle_binary), scenario],
        check=True,
        timeout=5 if scenario == "abort_unpublished" else 20,
    )


@pytest.mark.parametrize("compile_success", [True, False])
def test_urma_fixture_uses_selected_host_compiler(tmp_path, tmp_path_factory, monkeypatch, compile_success):
    marker = tmp_path / "compiler-arguments.json"
    compiler = tmp_path / "selected compiler"
    compiler.write_text(
        f"#!{sys.executable}\n"
        "import json, pathlib, sys\n"
        f"pathlib.Path({str(marker)!r}).write_text(json.dumps(sys.argv[1:]))\n"
        "print('selected host compiler diagnostic', file=sys.stderr)\n"
        f"sys.exit({0 if compile_success else 7})\n"
    )
    compiler.chmod(0o755)
    monkeypatch.setenv("CXX", f"{shlex.quote(str(compiler))} --compiler-option")
    monkeypatch.setenv("ASCEND_HOME_PATH", str(tmp_path))
    monkeypatch.setattr(sys.modules[__name__], "_require_compatible_cann_headers", lambda *_args: None)
    monkeypatch.setattr(sys.modules[__name__], "ensure_pto_isa_root", lambda: str(tmp_path))

    if compile_success:
        urma_lifecycle_binary.__wrapped__(tmp_path_factory)
    else:
        with pytest.raises(pytest.fail.Exception, match="selected host compiler diagnostic"):
            urma_lifecycle_binary.__wrapped__(tmp_path_factory)

    arguments = json.loads(marker.read_text())
    assert arguments[0] == "--compiler-option"
    assert str(ROOT / "tests/ut/py/support/urma_lifecycle_host.cpp") in arguments


@pytest.mark.parametrize("missing", ["sdk", "acquire", "tag_limit", None])
def test_urma_fixture_checks_sdk_capabilities(tmp_path, tmp_path_factory, monkeypatch, missing):
    compiler = shutil.which("g++-15") or shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a host C++ compiler")
    if missing != "sdk":
        headers = tmp_path / "include/hccl"
        headers.mkdir(parents=True)
        (headers / "hccl.h").write_text("#pragma once\n")
        (headers / "hccl_channel.h").write_text(
            "" if missing == "acquire" else "void HcclChannelAcquireWithConfig();\n"
        )
        (headers / "hccl_res.h").write_text(
            "" if missing == "tag_limit" else "constexpr unsigned HCCL_CHANNEL_CONFIG_SHARED_QUEUE_TAG_MAX_LEN = 256;\n"
        )
    monkeypatch.setenv("ASCEND_HOME_PATH", str(tmp_path))
    monkeypatch.setenv("CXX", shlex.quote(compiler))

    def begin_fixture_build():
        raise RuntimeError("SDK accepted; beginning fixture build")

    monkeypatch.setattr(sys.modules[__name__], "ensure_pto_isa_root", begin_fixture_build)
    exception = pytest.skip.Exception if missing else RuntimeError
    message = "requires compatible CANN headers" if missing else "SDK accepted; beginning fixture build"
    with pytest.raises(exception, match=message):
        urma_lifecycle_binary.__wrapped__(tmp_path_factory)
