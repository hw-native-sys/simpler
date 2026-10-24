# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Record the actual framework and runtime modules inside the exclusive CI task."""

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import _task_interface
import torch
import torch_npu

from simpler_setup.environment import PROJECT_ROOT
from simpler_setup.test_environments import preflight


def _identity(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    assert _task_interface.__build_commit__ == source, (_task_interface.__build_commit__, source)
    preflight("torch_npu")
    modules = {
        name: _identity(module.__file__)
        for name, module in (("torch", torch), ("torch_npu", torch_npu), ("native", _task_interface))
    }
    runtimes = {
        runtime: {
            name: _identity(PROJECT_ROOT / "build/lib/a2a3/onboard" / runtime / name)
            for name in ("libhost_runtime.so", "libaicpu_kernel.so", "aicore_kernel.o")
        }
        for runtime in ("host_build_graph", "tensormap_and_ringbuffer")
    }
    record = {
        "source": source,
        "python": sys.version,
        "executable": sys.executable,
        "machine": platform.machine(),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "modules": modules,
        "runtimes": runtimes,
    }
    destination = Path("outputs/framework-eager-ci/identity.json")
    destination.write_text(json.dumps(record, indent=2) + "\n")
    print("FRAMEWORK_CI_RUNTIME " + json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
