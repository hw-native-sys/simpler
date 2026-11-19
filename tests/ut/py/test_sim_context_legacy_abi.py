# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A separately built legacy consumer must still load against Sim Context."""

import subprocess
import sys
from pathlib import Path


def test_legacy_device_binding_consumer(tmp_path):
    root = Path(__file__).resolve().parents[3]
    context = root / "build/lib/libcpu_sim_context.so"
    assert context.is_file(), "Build the simulation package before running this test"
    source = tmp_path / "legacy.cpp"
    # Deliberately use the old declaration, independent of the current header.
    source.write_text(
        'extern "C" void pto_cpu_sim_bind_device(int);\n'
        'extern "C" void legacy_bind(int id) { pto_cpu_sim_bind_device(id); }\n'
    )
    consumer = tmp_path / "legacy.so"
    flags = ["-dynamiclib", "-undefined", "dynamic_lookup"] if sys.platform == "darwin" else ["-shared"]
    subprocess.run(["c++", "-fPIC", *flags, str(source), "-o", str(consumer)], check=True, capture_output=True)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import ctypes, os, sys
context = ctypes.CDLL(sys.argv[1], mode=os.RTLD_GLOBAL | os.RTLD_NOW)
consumer = ctypes.CDLL(sys.argv[2], mode=os.RTLD_NOW)
consumer.legacy_bind.argtypes = [ctypes.c_int]
consumer.legacy_bind.restype = None
context.pto_cpu_sim_get_bound_device.restype = ctypes.c_int
consumer.legacy_bind(7)
assert context.pto_cpu_sim_get_bound_device() == 7
consumer.legacy_bind(-1)
assert context.pto_cpu_sim_get_bound_device() == -1
""",
            str(context),
            str(consumer),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
