# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 FLASH decode on host_build_graph: same program, host-run orchestration.

The 43-layer network, its 368 kernels and its fixture come from the
``tensormap_and_ringbuffer`` case; only the runtime changes. HBG compiles the
orchestration with the host g++, runs it on the host CPU instead of the AICPU,
and ships the built SM image to the device, which boots scheduler-only.

``kernels/orchestration/decode_fwd_graph.cpp`` is that case's orchestration
with the runtime untouched, recast as a Graph: the whole forward pass is cut
into Graph blocks covering all 43 layers, so the host records eight
Definitions and submits 129 tasks itself, the rest being built on the recording
threads. The ten ``recv_count_out`` reads that drove the MoE per-expert tile
loops (a task-produced tensor, unreadable while the host builds the graph) are
now dispatch predicates the scheduler evaluates on device. Tensor initialization
is shared with the TMR case and runs in AIV kernels, and the attention/FFN scale
factors travel as kernel tensor inputs read from GM, so the host never writes a
GM-heap device address and never copies tensor data into task scalars. The
orchestration source is therefore the TMR one recast as a Graph, with no
runtime-specific rewrite. ``skip_golden`` is inherited from the TMR case, which
is itself a completion/smoke case.

Host construction, Graph recording (129 host submissions) and device replay all
complete, both ranks ``outcome=0``. See README.md for the measurements and for
the fixes that got the replay running.

    python examples/a2a3/host_build_graph/deepseek_v4_flash_decode/\\
test_deepseek_v4_flash_decode.py -p a2a3 -d <d0>,<d1>
"""

import copy
import importlib.util
import sys
from pathlib import Path

import pytest

from simpler_setup import SceneTestCase, scene_test
from simpler_setup.goldens.deepseek_v4_flash_decode import N_RANKS, generate_inputs

HERE = Path(__file__).resolve().parent
TMR_CASE_DIR = HERE.parents[1] / "tensormap_and_ringbuffer/deepseek_v4_flash_decode"


def _load_tmr_case():
    module_name = "_dsv4_flash_tmr_base"
    spec = importlib.util.spec_from_file_location(module_name, TMR_CASE_DIR / "test_deepseek_v4_flash_decode.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the TMR deepseek_v4_flash_decode case")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_TMR = _load_tmr_case()


def _host_build_graph_callable():
    """Same CALLABLE as the TMR case, with kernel sources re-pointed at the TMR dir
    and the orchestration swapped for the Graph-form variant.

    ``decode_fwd_graph.cpp`` is the tensormap_and_ringbuffer orchestration with
    the runtime untouched and no runtime-specific rewrite. HBG runs the
    orchestrator on the host before the device executes anything, so the ten
    ``get_tensor_data(recv_count_out)`` reads (a task-produced tensor that drove
    the MoE per-expert tile loops) had no value to return; both runtimes now use
    a static per-expert tile grid whose tasks carry a dispatch predicate on
    ``recv_count_out[expert][0]``, read by the scheduler on device, and the
    ``exp_gate_up_act*`` kernels derive ``valid_rows`` from the count tensor in
    ``kernel_entry``. The one ``get_tensor_data`` read left
    (``ext_num_tokens_per_owner``) is an external tensor and drives
    ``set_block_num``, which a predicate cannot express; the former 30 scale
    reads are gone — the consuming kernels read the scale tensors from GM
    directly. The shared orchestration submits the same device-side
    initialization kernels under both runtimes.

    The case is ``skip_golden`` because the TMR case it is derived from is: it
    measures host-side graph construction and device execution, not numerics.
    """
    callable_config = copy.deepcopy(_TMR.TestDeepseekV4FlashDecode.CALLABLE)
    chip = callable_config["callables"][0]
    chip["orchestration"]["source"] = str(HERE / "kernels/orchestration/decode_fwd_graph.cpp")
    for incore in chip["incores"]:
        incore["source"] = str(TMR_CASE_DIR / incore["source"])
    return callable_config


@pytest.mark.resource_last
@scene_test(level=3, runtime="host_build_graph")
class TestDeepseekV4FlashDecodeHostBuildGraph(SceneTestCase):
    """DSv4 FLASH EP2/TP2 decode with the orchestration built on the host."""

    CALLABLE = _host_build_graph_callable()
    CASES = [
        {
            "name": "DecodeFwdEP2TP2",
            "platforms": ["a2a3"],
            "skip_golden": True,
            "config": {
                "device_count": N_RANKS,
                "num_sub_workers": 0,
                # Ring sizing matches the TMR case: both runtimes now build the
                # same static per-expert tile grid, so both allocate tile scratch
                # for all 32 experts of every MoE layer.
                "runtime_env": {
                    "ring_task_window": 16384,
                    "ring_heap": 2 << 30,
                    "ring_dep_pool": 16384,
                },
            },
            "params": {"seed": 1234},
        }
    ]

    def generate_args(self, params):
        return generate_inputs(params.get("seed", 1234))

    def compute_golden(self, args, params):
        raise NotImplementedError(
            "deepseek_v4_flash_decode is a completion/smoke case (skip_golden): no "
            "full-network torch reference exists upstream either. Component-level "
            "goldens live with the standalone kernels in pypto-lib."
        )


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
