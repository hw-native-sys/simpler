# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Preserve vector math and four-task dependencies across input rebinding.

Native WSE Graph equivalence is covered in direct_launch/test_direct_launch.py.
"""

import json

import pytest
from simpler.task_interface import CallConfig
from simpler.worker import Worker

from tests.st.wse.host_build_graph.fixed_sram_layout import release_view
from tests.st.wse.host_build_graph.vector_example.test_vector_example import build_vector, check_vector, vector_args

pytestmark = [pytest.mark.platforms(["wsesim"]), pytest.mark.runtime("host_build_graph"), pytest.mark.device_count(1)]


def test_vector_dependencies_and_rebinding(tmp_path, st_platform):
    worker = Worker(level=2, platform=st_platform, runtime="host_build_graph", device_id=0)
    worker.init()
    try:
        handle = worker.register(build_vector(st_platform))
        for repeat in range(2):
            args, windows, values, expected = vector_args(worker, repeat)
            config = CallConfig()
            config.output_prefix = str(tmp_path / str(repeat))
            worker.run(handle, args, config)
            check_vector(worker, windows, values, expected)
            for window in windows:
                release_view(worker, window)
            rows = [json.loads(line) for line in (tmp_path / str(repeat) / "wse_events.jsonl").read_text().splitlines()]
            assert rows[0]["expected_tasks"] == 4
            done = {e["task"]: e["seq"] for e in rows[1:] if e["kind"] == 6}
            begin = {e["task"]: e["seq"] for e in rows[1:] if e["kind"] == 4}
            assert len(done) == 4
            for producer, consumer in [(0, 1), (0, 2), (1, 3), (2, 3)]:
                assert done[producer] < begin[consumer]
        worker.unregister(handle)
    finally:
        worker.close()
