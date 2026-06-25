#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""args_dump profiling smoke — capture pipeline produces a usable
``args_dump/`` directory.

Re-uses ``vector_example`` (5 submit_task calls). With ``--dump-args`` the
AICPU writer captures task dump records into a unified manifest + raw-byte
payload pair under ``<output_prefix>/args_dump/``. Smoke asserts:
manifest exists + parses, the ``bin_file`` field it names exists, entries
use the unified schema, and no legacy args-only manifest is emitted.
"""

import json
import subprocess
import sys

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, Tensor, scene_test
from simpler_setup.scene_test import _outputs_dir, _sanitize_for_filename

KERNELS_BASE = "../../../../../../examples/a2a3/tensormap_and_ringbuffer/vector_example/kernels"


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestArgsDump(SceneTestCase):
    """args_dump capture smoke, level-aware on the ``--dump-args`` level.

    Uses ``partial_dump_orch`` (5 tasks; four carry ``dump(...)`` markers) so a
    single orchestration exercises both modes:

    - ``--dump-args 1`` (partial): only marked args are captured — task
      ``0x..00`` via no-arg ``dump()`` (all tensor + scalar args), task
      ``0x..01`` via ``dump(t2_addend)`` (scalar-only), task ``0x..02`` via
      ``dump(d, inter_ci, t3_count)`` (mixed tensor + scalar, input ``e``
      excluded), and task ``0x..03`` via no-arg ``dump()`` (all tensor args).
      Mode is latched host-side before dispatch, so it is race-free regardless
      of submission order.
    - ``--dump-args 2`` (full): markers are ignored, every task is dumped.

    The dump level comes straight from the CLI ``--dump-args`` value
    (no per-case override).
    """

    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/partial_dump_orch.cpp",
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
        level = int(request.config.getoption("--dump-args", default=0))
        if not level:
            return
        safe_label = _sanitize_for_filename("TestArgsDump_default")
        matches = sorted(_outputs_dir().glob(f"{safe_label}_*"), key=lambda p: p.stat().st_mtime)
        assert matches, "args dump output directory missing"
        dump_dir = matches[-1] / "args_dump"
        assert dump_dir.is_dir(), f"args_dump/ missing under {matches[-1]} — dump capture failed?"
        manifest = dump_dir / "args_dump.json"
        assert manifest.exists(), f"args_dump.json missing under {dump_dir} — collector finalize failed?"
        with manifest.open() as f:
            data = json.load(f)
        bin_name = data.get("bin_file")
        tensors = data.get("args", [])
        assert tensors, f"args_dump.json has no entries: {data}"
        if level == 3:
            # full_json_only: metadata only, no payload and no .bin file.
            assert bin_name is None, f"level 3 manifest should have bin_file=null: {data}"
            assert not (dump_dir / "args.bin").exists(), "level 3 must not write args.bin"
            assert all(t.get("bin_size") == 0 for t in tensors), tensors
        else:
            assert bin_name, f"manifest missing bin_file field: {data}"
            bin_path = dump_dir / bin_name
            assert bin_path.exists(), f"manifest names bin_file={bin_name!r} but {bin_path} not found"
            assert bin_path.stat().st_size > 0, "args.bin is empty"

        # Unified manifest (#792): tensors and scalar args share one
        # args_dump.json keyed by a "kind" field; no separate legacy sidecar files.
        assert not (dump_dir / "tensor_dump.json").exists(), "tensor_dump.json should not be emitted"
        assert not (dump_dir / "kernel_args_dump.json").exists(), "kernel_args_dump.json should not be emitted"
        assert all("kind" in t for t in tensors), tensors
        scalar_entries = [t for t in tensors if t.get("kind") == "scalar"]
        assert all(t.get("stage") == "before_dispatch" for t in scalar_entries), scalar_entries
        assert all(t.get("bin_size") == 0 for t in scalar_entries), scalar_entries
        assert all("value" in t for t in scalar_entries), scalar_entries

        # Level-aware checks operate on the tensor entries.
        tensor_entries = [t for t in tensors if t.get("kind") == "tensor"]
        task_ids = {t["task_id"] for t in tensor_entries}
        if level == 1:
            # Partial: only the selected tensor/scalar args, race-free (host-latched).
            assert len(tensor_entries) == 7, f"partial expected 7 tensor entries, got {len(tensor_entries)}"
            assert task_ids == {
                "0x0000000100000000",
                "0x0000000100000002",
                "0x0000000100000003",
            }
            # Task granularity: dump() captured all tensor args on task 0.
            t00 = sorted(t["arg_index"] for t in tensor_entries if t["task_id"] == "0x0000000100000000")
            assert t00 == [0, 1]
            # Mixed granularity: dump(d, inter_ci, t3_count) captured tensor args
            # 0 + 2, not arg 1 (e).
            t02 = sorted(t["arg_index"] for t in tensor_entries if t["task_id"] == "0x0000000100000002")
            assert t02 == [0, 2]
            # Tensor-only task granularity: dump() captured all three tensor args.
            t03 = sorted(t["arg_index"] for t in tensor_entries if t["task_id"] == "0x0000000100000003")
            assert t03 == [0, 1, 2]
            scalar_by_task = {
                task_id: sorted(t["arg_index"] for t in scalar_entries if t["task_id"] == task_id)
                for task_id in {t["task_id"] for t in scalar_entries}
            }
            assert scalar_by_task == {
                "0x0000000100000000": [2, 3],
                "0x0000000100000001": [2],
                "0x0000000100000002": [3],
            }
            ambiguous_scalars = [
                (t["task_id"], t["arg_index"]) for t in scalar_entries if t.get("arg_index_ambiguous", False)
            ]
            assert ambiguous_scalars == [("0x0000000100000002", 3)]
        else:
            # Full (level 2 or 3): markers ignored — every one of the 5 tasks is dumped.
            assert len(task_ids) >= 5, f"full dump should cover all 5 tasks, got {sorted(task_ids)}"

        # ---- Tool smoke: dump_viewer ----
        # Exit-code-only check; the no-filter default lists every captured
        # arg without exporting. A schema change that breaks the viewer
        # fires here in the same CI step that produced the dump.
        subprocess.run(
            [sys.executable, "-m", "simpler_setup.tools.dump_viewer", str(dump_dir)],
            check=True,
            timeout=60,
        )


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
