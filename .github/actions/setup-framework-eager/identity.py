# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Record the actual framework and runtime modules inside the exclusive CI task."""

import argparse
import hashlib
import importlib
import json
import platform
import subprocess
import sys
from pathlib import Path


def _identity(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _runtimes(platform_name, project_root):
    if platform_name not in ("a2a3", "a5"):
        raise ValueError("platform must be a2a3 or a5")
    return {
        runtime: {
            name: _identity(project_root / "build/lib" / platform_name / "onboard" / runtime / name)
            for name in ("libhost_runtime.so", "libaicpu_kernel.so", "aicore_kernel.o")
        }
        for runtime in ("host_build_graph", "tensormap_and_ringbuffer")
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("a2a3", "a5"), required=True)
    args = parser.parse_args()

    environment = importlib.import_module("simpler_setup.test_environments").preflight("torch_npu")
    project_root = importlib.import_module("simpler_setup.environment").PROJECT_ROOT
    native = importlib.import_module("_task_interface")
    torch = importlib.import_module("torch")
    torch_npu = importlib.import_module("torch_npu")

    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    assert native.__build_commit__ == source, (native.__build_commit__, source)
    modules = {
        name: _identity(module.__file__)
        for name, module in (("torch", torch), ("torch_npu", torch_npu), ("native", native))
    }
    runtimes = _runtimes(args.platform, project_root)
    record = {
        "source": source,
        "platform": args.platform,
        "environment": environment,
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
