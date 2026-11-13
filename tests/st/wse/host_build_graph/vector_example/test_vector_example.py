# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Four-task vector DAG using explicit views of fixed SRAM."""

import json
from functools import lru_cache
from pathlib import Path

import pytest
import torch
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import CallConfig, DataType, TaskArgs, TensorArgType
from simpler.worker import Worker

from simpler_setup.scene_test import _bound_host_logs
from tests.st.wse.host_build_graph.fixed_sram_layout import copy_from, copy_to, layout_view, release_view
from tests.st.wse.host_build_graph.scene_helpers import build_callable

pytestmark = [pytest.mark.platforms(["wsesim"]), pytest.mark.runtime("host_build_graph"), pytest.mark.device_count(1)]


@lru_cache
def build_vector(platform):
    return build_callable(
        platform,
        Path(__file__).parent / "kernels",
        "orchestration/example_orch.cpp",
        [D.IN, D.IN] + [D.OUT] * 4,
        [
            ("aiv/kernel_add.cpp", [D.IN, D.IN, D.OUT]),
            ("aiv/kernel_add_scalar.cpp", [D.IN, D.OUT]),
            ("aiv/kernel_mul.cpp", [D.IN, D.IN, D.OUT]),
        ],
    )


def vector_args(worker, seed=0):
    count = 128 * 128
    a = torch.arange(count, dtype=torch.float32) % 13 - 5 + seed
    b = torch.full((count,), 3.0 + seed)
    values = [a, b, *[torch.full((count,), -1000.0 - seed) for _ in range(4)]]
    windows, args = [], TaskArgs()
    for index, value in enumerate(values):
        window = layout_view(worker, value.nbytes, core=index)
        copy_to(worker, window, value)
        windows.append(window)
        args.add_tensor(
            window.tensor(shapes=(count,), dtype=DataType.FLOAT32),
            TensorArgType.INPUT if index < 2 else TensorArgType.OUTPUT_EXISTING,
        )
    return args, windows, values, (a + b + 1) * (a + b + 2)


def check_vector(worker, windows, values, expected):
    copy_from(worker, values[2], windows[2])
    torch.testing.assert_close(values[2], expected, rtol=0, atol=0)


def test_vector_fixed_sram(tmp_path, st_platform, request):
    worker = Worker(level=2, platform=st_platform, runtime="host_build_graph", device_id=0)
    worker.init()
    try:
        handle = worker.register(build_vector(st_platform))
        args, windows, values, expected = vector_args(worker)
        config = CallConfig()
        config.output_prefix = str(tmp_path)
        config.enable_dep_gen = request.config.getoption("--enable-dep-gen")
        worker.run(handle, args, config)
        check_vector(worker, windows, values, expected)
        rows = [json.loads(line) for line in (tmp_path / "wse_events.jsonl").read_text().splitlines()]
        assert rows[0]["expected_tasks"] == 4
        if config.enable_dep_gen:
            assert (tmp_path / "deps.json").is_file()
        for window in windows:
            release_view(worker, window)
        worker.unregister(handle)
    except BaseException:
        for log in _bound_host_logs():
            print(log.read_text(), flush=True)
        raise
    finally:
        worker.close()
