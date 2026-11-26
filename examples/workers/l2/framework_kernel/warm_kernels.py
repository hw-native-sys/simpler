# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Populate the framework consumers' kernel cache without initializing devices."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib
import json
import os
from pathlib import Path


def warm_kernels(platform):
    from simpler_setup.environment import PROJECT_ROOT  # noqa: PLC0415
    from simpler_setup.pto_isa import ensure_pto_isa_root  # noqa: PLC0415
    from simpler_setup.scene_test_cache import KERNEL_CACHE_DIR  # noqa: PLC0415

    checkout = ensure_pto_isa_root()
    records = []
    for runtime in ("host_build_graph", "tensormap_and_ringbuffer"):
        consumers = [
            ("main", {}),
            ("alias_inout", {}),
            ("argument_capture", {}),
            ("scope_statistics", {}),
            ("persistent_state", {}),
            ("lifecycle", {"name": "x"}),
            ("lifecycle", {"name": "y"}),
            ("value_arguments", {"host_control": False}),
        ]
        if runtime == "host_build_graph":
            consumers.append(("value_arguments", {"host_control": True}))
        for name, options in consumers:
            module = importlib.import_module(name)
            callable_obj = module._compile(runtime, platform=platform, **options)
            raw = ctypes.string_at(int(callable_obj.buffer_ptr()), int(callable_obj.buffer_size()))
            record = {
                "consumer": name,
                "runtime": runtime,
                "options": options,
                "size": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
            records.append(record)
            print("FRAMEWORK_KERNEL_WARMED " + json.dumps(record, sort_keys=True), flush=True)
    return {
        "platform": platform,
        "project_root": str(PROJECT_ROOT),
        "cache": str(KERNEL_CACHE_DIR),
        "pto_isa_root": str(checkout),
        "callables": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", required=True, choices=("a2a3", "a5"))
    parser.add_argument("--output", required=True, type=Path)
    options = parser.parse_args()
    if os.environ.get("TASKQUEUE_INSIDE") and os.environ.get("TASK_DEVICE"):
        parser.error("kernel warm-up must run outside a device allocation")
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    options.output.unlink(missing_ok=True)
    record = warm_kernels(options.platform)
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(record, indent=2) + "\n")
    print("FRAMEWORK_KERNEL_WARMUP " + str(options.output), flush=True)


if __name__ == "__main__":
    main()
