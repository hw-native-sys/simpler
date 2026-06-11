#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""scope_stats smoke — capture pipeline produces a usable ``scope_stats.jsonl``.

Re-uses ``vector_example`` (outer executor scope + one inner ``PTO2_SCOPE()``).
With ``--enable-scope-stats`` the platform collector
(``scope_stats_collector_aicpu.h``) appends one record per scope boundary
(begin and end) into a pooled buffer that streams to the host, which writes
NDJSON. Enabling the flag is the entire user surface for the new API — the
runtime takes care of the ``set_pending_site`` / ``scope_stats_begin`` /
``scope_stats_end`` calls. Schema lives in ``docs/dfx/scope-stats.md`` §3.

Output (``scope_stats.jsonl``): line 1 is run metadata
(``{"version":5,"fatal":bool,"dropped":uint,"total":uint}``); each subsequent
line is one scope-boundary record carrying task/heap/dep_pool start-end.
"""

import json

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, Tensor, scene_test
from simpler_setup.scene_test import _outputs_dir, _sanitize_for_filename

KERNELS_BASE = "../../../../../../examples/a2a3/tensormap_and_ringbuffer/vector_example/kernels"
_REQUIRED_RECORD_FIELDS = {
    "site",
    "phase",
    "depth",
    "ring",
    "task_window_start",
    "task_window_end",
    "heap_start",
    "heap_end",
    "dep_pool_start",
    "dep_pool_end",
    "tensormap",
}


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestScopeStats(SceneTestCase):
    """Vector example with --enable-scope-stats, then assert scope_stats.jsonl."""

    CALLABLE = {
        "orchestration": {
            "source": f"{KERNELS_BASE}/orchestration/example_orchestration.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.IN, D.OUT],
        },
        "incores": [
            {
                "func_id": 0,
                "source": f"{KERNELS_BASE}/aiv/kernel_add.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 1,
                "source": f"{KERNELS_BASE}/aiv/kernel_add_scalar.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.OUT],
            },
            {
                "func_id": 2,
                "source": f"{KERNELS_BASE}/aiv/kernel_mul.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.OUT],
            },
        ],
    }

    CASES = [
        {
            "name": "default",
            "platforms": ["a2a3sim", "a2a3"],
            "config": {"aicpu_thread_num": 4, "block_dim": 3},
            "params": {},
        },
    ]

    def generate_args(self, params):
        SIZE = 128 * 128
        return TaskArgsBuilder(
            Tensor("a", torch.full((SIZE,), 2.0, dtype=torch.float32)),
            Tensor("b", torch.full((SIZE,), 3.0, dtype=torch.float32)),
            Tensor("f", torch.zeros(SIZE, dtype=torch.float32)),
        )

    def compute_golden(self, args, params):
        args.f[:] = (args.a + args.b + 1) * (args.a + args.b + 2) + (args.a + args.b)

    def test_run(self, st_platform, st_worker, request):
        super().test_run(st_platform, st_worker, request)
        if not request.config.getoption("--enable-scope-stats", default=False):
            return
        for case in self.CASES:
            if st_platform in case["platforms"]:
                self._validate_scope_stats_artifact(case)

    def _validate_scope_stats_artifact(self, case):
        safe_label = _sanitize_for_filename(f"TestScopeStats_{case['name']}")
        matches = sorted(_outputs_dir().glob(f"{safe_label}_*"), key=lambda p: p.stat().st_mtime)
        assert matches, (
            f"no output directory under {_outputs_dir()} matching {safe_label}_* — "
            f"--enable-scope-stats was on but the run produced no per-case output dir"
        )
        path = matches[-1] / "scope_stats" / "scope_stats.jsonl"
        assert path.exists(), f"scope_stats.jsonl missing under {matches[-1]} — collector finalize failed?"
        lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
        assert lines, f"scope_stats.jsonl empty under {matches[-1]}"
        meta = json.loads(lines[0])
        assert meta.get("version") == 5, f"unexpected schema version: {meta!r}"
        assert meta.get("fatal") is False, f"run latched fatal: {meta!r}"
        assert meta.get("dropped", 0) == 0, f"records dropped on device: {meta!r}"
        assert "dep_pool_max" in meta, f"metadata missing dep_pool_max: {meta!r}"
        records = [json.loads(ln) for ln in lines[1:]]
        # outer (executor) + inner PTO2_SCOPE, each emitting a begin and an end
        # record → ≥4 records.
        assert len(records) >= 4, f"expected ≥4 begin/end records, got {records!r}"
        for rec in records:
            assert _REQUIRED_RECORD_FIELDS <= rec.keys(), f"record missing fields: {rec!r}"


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
