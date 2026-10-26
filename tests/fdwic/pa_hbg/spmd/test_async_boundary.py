# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Resident deferred-registration refusal, without issuing asynchronous work.

This checks the public kernel API's return values, not Host detection of async
kernels, deferred completion support, or hardware SDMA fallback.
"""

import copy

from simpler_setup import SceneTestCase, scene_test

if __package__:
    from . import test_spmd_contract as _spmd
else:
    import test_spmd_contract as _spmd

_ASYNC_COMPLETE = 0xC34


@scene_test(level=2, runtime="host_build_graph")
class TestHbgAsyncBoundary(_spmd.TestHbgSpmdContract):
    CALLABLE = copy.deepcopy(_spmd.TestHbgSpmdContract.CALLABLE)
    for _kernel in CALLABLE["incores"]:
        if _kernel["func_id"] < 3:
            _kernel["name"] = f"async_boundary_{_kernel['func_id']}"
            _kernel["source"] = f"kernels/async_boundary_{_kernel['func_id']}.cpp"
    del _kernel

    CASES = [
        {
            "name": f"async_refusal_mask{mask}_n{blocks}",
            "platforms": ["a5", "a5sim"],
            "params": {"mask": mask, "blocks": blocks, "tasks": 2, "tail_delay": 4096},
        }
        for mask in (1, 2, 4, 3, 5, 6, 7)
        for blocks in (1, 2)
    ]

    def compute_golden(self, args, params):
        super().compute_golden(args, params)
        records = args.records.view(params["tasks"], params["blocks"] * 3 + 1, _spmd._RECORD_WORDS)
        for block in range(params["blocks"]):
            for slot in range(3):
                if params["mask"] & (1 << slot):
                    records[:, block * 3 + slot, 15] = _ASYNC_COMPLETE

    def _run_and_validate(self, worker, callable_obj, case, *args, **kwargs):
        assert not (kwargs.get("skip_golden", False) or case.get("skip_golden", self.SKIP_GOLDEN)), (
            "async boundary acceptance cannot skip golden or invocation provenance"
        )
        kwargs["rounds"] = max(3, kwargs.get("rounds", 1))
        return super()._run_and_validate(worker, callable_obj, case, *args, **kwargs)


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
