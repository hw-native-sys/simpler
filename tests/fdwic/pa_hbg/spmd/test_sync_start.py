# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Resident acceptance with real block barriers, early return and independent full-capacity tasks."""

from simpler_setup import SceneTestCase, scene_test

if __package__:
    from .test_spmd_contract import TestHbgSpmdContract as _SpmdContract
else:
    from test_spmd_contract import TestHbgSpmdContract as _SpmdContract


@scene_test(level=2, runtime="host_build_graph")
class TestHbgSyncStart(_SpmdContract):
    CASES = [
        {
            "name": f"{mode_name}_mask{mask}_n{blocks}",
            "platforms": ["a5sim"] if blocks in (8, 16) else (["a5"] if blocks >= 32 else ["a5", "a5sim"]),
            "params": {"mask": mask, "blocks": blocks, "tasks": 2, "tail_delay": 8192, "sync_mode": mode},
        }
        for mask in (1, 2, 4, 3, 5, 6, 7)
        for blocks in ((2, 7, 8, 16, 32, 64) if mask in (2, 4) else (2, 7, 8, 32))
        for mode_name, mode in (("barrier", 1), ("early_return", 2))
    ] + [
        {
            "name": "reuse_early_return_mix",
            "platforms": ["a5", "a5sim"],
            "params": {"mask": 7, "blocks": 2, "tasks": 17, "tail_delay": 4096, "sync_mode": 2},
        }
    ]


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
