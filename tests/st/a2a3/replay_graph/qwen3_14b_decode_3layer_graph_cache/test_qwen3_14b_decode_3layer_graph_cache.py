#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import copy
import importlib.util
import sys
from pathlib import Path

from simpler_setup import SceneTestCase, scene_test
from simpler_setup.goldens.qwen3_14b_decode_3layer import compute_golden as _decode_golden
from simpler_setup.goldens.qwen3_14b_decode_3layer import generate_inputs as _decode_generate_inputs


_REPO_ROOT = Path(__file__).resolve().parents[5]
_EXAMPLE_DIR = _REPO_ROOT / "examples/a2a3/replay_graph/qwen3_14b_decode_3layer"
_QWEN_SRC = "../../../../../examples/a2a3/replay_graph/qwen3_14b_decode_3layer/kernels"

_spec = importlib.util.spec_from_file_location("_qwen3_14b_decode_3layer_example", _EXAMPLE_DIR / "test_qwen3_14b_decode.py")
_example = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules[_spec.name] = _example
_spec.loader.exec_module(_example)


def _graph_cache_callable():
    callable_cfg = copy.deepcopy(_example.TestQwen314BDecode3Layer.CALLABLE)
    callable_cfg["orchestration"]["source"] = "kernels/orchestration/decode_fwd_layers_graph_cache.cpp"
    for incore in callable_cfg["incores"]:
        src = incore["source"]
        if src.startswith("kernels/"):
            incore["source"] = f"{_QWEN_SRC}/{src[len('kernels/'):]}"
    return callable_cfg


@scene_test(level=2, runtime="replay_graph")
class TestQwen314BDecode3LayerGraphCache(SceneTestCase):
    RTOL = _example.TestQwen314BDecode3Layer.RTOL
    ATOL = _example.TestQwen314BDecode3Layer.ATOL
    CALLABLE = _graph_cache_callable()

    CASES = [
        {
            "name": "GraphCacheStressBatch16Seq3500",
            "platforms": ["a2a3"],
            "config": {"aicpu_thread_num": 4, "block_dim": 0, "enable_graph_cache": True, "rounds": 2},
            "params": {"seed": 1234, "seq_len": 3500},
        },
    ]

    def generate_args(self, params):
        return _decode_generate_inputs(params.get("seed", 1234), params.get("seq_len", 3500))

    def compute_golden(self, args, params):
        _decode_golden(args)

if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
