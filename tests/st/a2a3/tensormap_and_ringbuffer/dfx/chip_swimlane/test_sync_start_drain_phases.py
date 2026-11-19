#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Scheduler phase accounting for ready and early cross-owner sync-start tasks on A2/A3."""

from collections import Counter
from pathlib import Path

from simpler_setup import SceneTestCase, scene_test
from simpler_setup.tools.swimlane_converter import read_perf_data
from tests.st.a2a3.tensormap_and_ringbuffer.spmd_sync_start_aiv.test_spmd_sync_start_aiv import (
    TestSpmdSyncStartAiv as _ReadyAiv,
)
from tests.st.a2a3.tensormap_and_ringbuffer.spmd_sync_start_early_dispatch.test_spmd_sync_start_early_dispatch import (
    TestSpmdSyncStartEarlyDispatch as _EarlyMix,
)

DRAIN_PHASES = ("drain", "drain_prepare", "drain_publish")


class _SchedulerPhases:
    def _build_config(self, *args, **kwargs):
        config = super()._build_config(*args, **kwargs)
        self._phase_path = (
            Path(config.output_prefix) / "chip_swimlane_records.json" if config.enable_chip_swimlane >= 3 else None
        )
        return config

    def phase_work(self):
        perf = read_perf_data(self._phase_path)
        work = Counter()
        for thread in perf["scheduler_records"]:
            for record in thread:
                work[record["phase"]] += record["tasks_processed"]
        return work


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestReadySyncStartDispatch(_SchedulerPhases, _ReadyAiv):
    CASES = [{"name": "global_aiv", "platforms": ["a2a3sim", "a2a3"], "manual": True, "params": {}}]

    def compare_outputs(self, test_args, golden_args, output_names, params):
        super().compare_outputs(test_args, golden_args, output_names, params)
        if self._phase_path is None:
            return
        work = self.phase_work()
        blocks = sum(int(width) for width in test_args.layout[::2])
        assert work["dispatch"] == blocks, f"expected {blocks} AIV dispatches, got {work}; {self._phase_path}"
        assert not any(phase in work for phase in DRAIN_PHASES), f"ready AIV entered drain: {work}; {self._phase_path}"


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestSyncStartDrainPhases(_SchedulerPhases, _EarlyMix):
    CASES = [
        {
            "name": "early_global_mix",
            "platforms": ["a2a3sim", "a2a3"],
            "manual": True,
            "config": {"aicpu_thread_num": 4},
            "params": {},
        }
    ]

    def compare_outputs(self, test_args, golden_args, output_names, params):
        super().compare_outputs(test_args, golden_args, output_names, params)
        if self._phase_path is None:
            return
        work = self.phase_work()
        blocks = int(test_args.layout[0])
        # Drain counts logical blocks. Prepare and publish count one AIC and two AIVs per MIX block.
        expected = (blocks, 3 * blocks, 3 * blocks)
        assert tuple(work[phase] for phase in DRAIN_PHASES) == expected, (
            f"expected {dict(zip(DRAIN_PHASES, expected, strict=True))}, got {work}; {self._phase_path}"
        )


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
