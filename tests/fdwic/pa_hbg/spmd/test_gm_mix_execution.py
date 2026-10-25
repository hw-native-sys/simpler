# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Native GM Mix identities, completion visibility, task overlap and retained-run checks.

The probe audits logical execution events, not physical lane placement or a full
admission/ack/seal trace. Every actual invocation requires fresh resident provenance.
"""

from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, scene_test

if __package__:
    from .test_ordered_execution import TestHbgOrderedExecution as _OrderedExecution
    from .test_ordered_execution import _case
else:
    from test_ordered_execution import TestHbgOrderedExecution as _OrderedExecution
    from test_ordered_execution import _case

_MASKS = (3, 5, 6, 7)
_PAIRS = ((3, 7), (7, 3), (6, 1), (1, 6), (5, 2), (2, 5), (6, 6), (7, 7))


@scene_test(level=2, runtime="host_build_graph")
class TestHbgGmMixExecution(_OrderedExecution):
    CALLABLE = {
        "orchestration": dict(_OrderedExecution.CALLABLE["orchestration"]),
        "incores": [dict(entry) for entry in _OrderedExecution.CALLABLE["incores"]]
        + [
            {
                "func_id": 2,
                "name": "ordered_record_slot_2",
                "source": "kernels/ordered_record_slot_2.cpp",
                "core_type": "aiv",
                "signature": [D.INOUT, D.IN, D.IN],
            }
        ],
    }
    CASES = (
        [_case(f"mix_mask{mask}_n{blocks}", "shape", mask, blocks=blocks) for mask in _MASKS for blocks in (1, 2, 65)]
        + [
            _case(
                f"mix_fanin_mask{mask}_slow_{position}",
                "fanin",
                mask,
                blocks=65,
                delay_block=position,
                tail_delay=262144,
            )
            for mask in _MASKS
            for position in ("head", "middle", "tail")
        ]
        + [
            _case(
                f"mix_{scenario}_mask{mask}_to{other}",
                scenario,
                mask,
                blocks=65,
                other_mask=other,
                tail_delay=8000000 if scenario == "overlap" else 262144,
            )
            for scenario in ("overlap", "edge")
            for mask, other in _PAIRS
        ]
        + [
            _case(f"mix_retained_large_small_mask{mask}", "shape", mask, invocation_blocks=(512, 1, 65, 1, 512))
            for mask in _MASKS
        ]
    )


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
