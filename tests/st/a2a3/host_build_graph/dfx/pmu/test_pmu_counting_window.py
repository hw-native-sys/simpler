#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""PMU counters over a window held open long enough for interference to land.

``tensormap_and_ringbuffer/dfx/pmu`` covers the capture pipeline's schema on a
payload of microsecond-scale tasks. It cannot see a window that *stops
counting*, and that is this subsystem's live failure: the AICore PMU control
register is programmed out-of-band from the AICPU, TS writes the same register,
and nothing arbitrates between them. A utilization query on the same chip, or a
background periodic source, resets the register mid-run and every counter from
that point reads zero.

What the window's length does to the rate is the whole reason this case exists:
the loss climbs from a minority of runs at a half-second window to most runs at
four seconds. A payload of microsecond-scale tasks therefore reproduces it so
rarely that a clean run is not evidence of anything, which is why the existing
smoke passes while the defect is live.

How long the window may be is set elsewhere, not chosen here: the onboard lanes
cap a single task at `SIMPLER_SCHEDULER_TIMEOUT_MS=2000`, so the window is
about 0.9 s and the loss is a minority of runs. See `_SPIN_ITERS` for what that
makes a pass worth.

The payload is ``worker_async_fifo``'s: one spinning ``delayed_add`` followed by
a 512-step scalar chain, 513 tasks in all. It is borrowed rather than copied so
that the spin and the chain have one definition.
"""

import ctypes
import time

import torch
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import TensorTransfer

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, TensorArg, scene_test
from simpler_setup.scene_test import _outputs_dir, _sanitize_for_filename

_KERNELS = "../../worker_async_fifo/kernels"

#: Elements per tensor; the kernels work one 128x128 float tile.
_SIZE = 128 * 128
#: Scalar-add steps the orchestration submits after the spinning task. Must
#: agree with `kChainLength` in pipelined_vector_orch.cpp.
_CHAIN_LENGTH = 512
#: Iterations of the first task's device spin, which is what holds the counting
#: window open. `worker_async_fifo`'s own count, and the longest this may be:
#: the onboard lanes export `SIMPLER_SCHEDULER_TIMEOUT_MS=2000`, and a single
#: task that outlasts that budget is reaped as a scheduler timeout rather than
#: measured. This spin runs about 0.9 s, so the window is under half the
#: budget.
#:
#: That bounds how much of the defect one run can see. At a window this long
#: the counters are lost in a minority of runs, not in most of them — the rate
#: climbs with the window and only a multi-second window makes a single pass
#: meaningful. So a pass here is weak evidence and a failure is strong
#: evidence: this case is a regression barrier, which accumulates across CI
#: runs, rather than a detector that settles the question in one.
_SPIN_ITERS = 200_000_000
#: Floor on the spinning task's counter. The spin measures 1.6333e9 cycles, and
#: that is a property of the loop's pipeline rather than of the core's clock:
#: at four times this count it varied by 1294 cycles — two parts in ten million
#: — across five runs. The floor sits far below it anyway, because its job is
#: to catch a window that stopped early, not to pin the measurement.
_SPIN_CYCLES_FLOOR = 1_000_000_000


@scene_test(level=2, runtime="host_build_graph")
class TestPmuCountingWindow(SceneTestCase):
    """Every row of a 513-task run counted, over a window held open ~0.9 s."""

    CALLABLE = {
        "orchestration": {
            "source": f"{_KERNELS}/orchestration/pipelined_vector_orch.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [D.IN, D.IN, D.OUT, D.IN],
        },
        "incores": [
            {
                "func_id": 0,
                "source": f"{_KERNELS}/aiv/delayed_add.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 1,
                "source": f"{_KERNELS}/aiv/kernel_add_scalar.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.OUT],
            },
        ],
    }

    CASES = [
        {
            "name": "spin",
            # Onboard only: the register and its other writer are on silicon,
            # and a spin this long under simulation is not a test anyone waits
            # for.
            "platforms": ["a2a3"],
            # Manual, so the ordinary sweep does not collect it. The sweep
            # passes no `--enable-pmu`, which leaves every assertion below
            # skipped while the payload still spends its spin on a device —
            # cost with nothing checked. The onboard DFX smokes ask for it by
            # name with `--manual include`, and that is where the assertions
            # are live.
            "manual": True,
            "config": {"device_count": 1},
            "params": {},
        },
    ]

    def generate_args(self, params):
        return TaskArgsBuilder(
            TensorArg("a", torch.full((_SIZE,), 2.0, dtype=torch.float32)),
            TensorArg("b", torch.full((_SIZE,), 3.0, dtype=torch.float32)),
            TensorArg("out", torch.zeros(_SIZE, dtype=torch.float32)),
            # host_build_graph runs the orchestration on the host at prepare
            # time, and this is the tensor it reads element 0 of to get every
            # chain step's addend. A transferred operand is not readable there,
            # so this one stays where the host can read it.
            TensorArg("control", torch.full((_SIZE,), 1.0, dtype=torch.float32), transfer=TensorTransfer.NONE),
            Scalar("spin_iters", ctypes.c_int64(_SPIN_ITERS)),
        )

    def compute_golden(self, args, params):
        args.out[:] = args.a + args.b + _CHAIN_LENGTH * args.control[0]

    def test_run(self, st_platform, st_worker, request):
        # Taken before the run so the artifact search cannot settle on a stale
        # directory from an earlier invocation with the same label.
        run_marker = int(time.time())
        super().test_run(st_platform, st_worker, request)
        if not request.config.getoption("--enable-pmu", default=0):
            return
        for case in self._matching_cases(st_platform, request):
            self._assert_every_row_counted(case, run_marker)

    def _assert_every_row_counted(self, case, run_marker):
        safe_label = _sanitize_for_filename(f"TestPmuCountingWindow_{case['name']}")
        matches = [p for p in _outputs_dir().glob(f"{safe_label}_*") if p.stat().st_mtime >= run_marker]
        assert matches, f"no PMU output directory for '{safe_label}' created this run — PMU capture failed?"
        run_dir = max(matches, key=lambda p: p.stat().st_mtime)
        csv = run_dir / "pmu.csv"
        assert csv.exists(), f"pmu.csv missing under {run_dir} — PMU capture failed?"
        lines = csv.read_text().splitlines()
        assert len(lines) >= 2, f"pmu.csv has no data rows (only header): {lines}"
        header_cols = lines[0].split(",")
        cycles_index = header_cols.index("pmu_total_cycles")
        func_index = header_cols.index("func_id")

        cycles = []
        func_ids = []
        for row in lines[1:]:
            cols = row.split(",")
            assert len(cols) == len(header_cols), f"row has {len(cols)} columns, header has {len(header_cols)}: {row}"
            cycles.append(int(cols[cycles_index]))
            func_ids.append(int(cols[func_index]))

        # The population first, because every check below is about the rows
        # that are present and none of them notices rows that are missing: the
        # orchestration submits one spinning task and `_CHAIN_LENGTH` scalar
        # steps, so a file holding only the spin's row satisfies "every row
        # counted" and "the longest row is long enough" while 512 records are
        # gone.
        spins = [i for i, f in enumerate(func_ids) if f == 0]
        steps = [i for i, f in enumerate(func_ids) if f == 1]
        assert len(spins) == 1 and len(steps) == _CHAIN_LENGTH, (
            f"expected 1 spinning task (func_id=0) and {_CHAIN_LENGTH} chain steps (func_id=1), got "
            f"{len(spins)} and {len(steps)} across {len(cycles)} rows: records were dropped or the "
            f"orchestration's shape changed, and the counter checks below cannot see a missing row"
        )

        # Per row, not in aggregate. The spinning task alone satisfying an "any
        # row is non-zero" check is the shape this failure actually takes, and
        # it is why the existing smoke's assertion cannot see it.
        zeros = [i for i, c in enumerate(cycles) if c == 0]
        assert not zeros, (
            f"{len(zeros)} of {len(cycles)} PMU rows report zero cycles over a window held open for "
            f"{_SPIN_ITERS} spin iterations. The rows carry this run's task identities, so the records "
            f"were produced and attributed and only the counter values are zero: the window stopped "
            f"counting rather than never having been armed. First row indices: {zeros[:16]}"
        )

        # A window disabled partway through the spin leaves a row that is
        # non-zero but short, which the check above cannot see: it was observed
        # at 1.1% and at 3.0% of the expected count with every row non-zero.
        # Held against the spinning task's own row rather than against max(),
        # which would let any other row stand in for it.
        spin_cycles = cycles[spins[0]]
        assert spin_cycles >= _SPIN_CYCLES_FLOOR, (
            f"the spinning task counted {spin_cycles} cycles against a floor of {_SPIN_CYCLES_FLOOR} for "
            f"{_SPIN_ITERS} iterations: non-zero but short, so the window was counting and stopped before "
            f"the task ended. Highest four rows overall: {sorted(cycles)[-4:]}"
        )
