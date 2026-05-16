# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for the RunTiming binding returned by Worker.run / run_prepared.

Pure-Python unit tests — no device, no kernels. Verifies that the nanobind
bound class behaves as documented at the boundary so callers (benchmark
scripts, perf dashboards) can rely on the contract without spinning up a
worker.
"""

import inspect

import pytest
from _task_interface import RunTiming  # pyright: ignore[reportMissingImports]


class TestRunTimingConstruction:
    def test_default_constructor_zeros(self):
        t = RunTiming()
        assert t.host_wall_ns == 0
        assert t.device_wall_ns == 0
        assert t.host_wall_us == 0.0
        assert t.device_wall_us == 0.0

    def test_explicit_values(self):
        t = RunTiming(1500, 2500)
        assert t.host_wall_ns == 1500
        assert t.device_wall_ns == 2500

    def test_device_wall_defaults_to_zero(self):
        # The (host_ns, device_ns=0) constructor is the path Worker.run uses
        # for L3+ DAGs, where per-task device cycles aren't aggregated yet.
        t = RunTiming(9999)
        assert t.host_wall_ns == 9999
        assert t.device_wall_ns == 0


class TestRunTimingUnitConversion:
    def test_ns_to_us_host(self):
        assert RunTiming(1_000, 0).host_wall_us == pytest.approx(1.0)
        assert RunTiming(1_500_000, 0).host_wall_us == pytest.approx(1500.0)

    def test_ns_to_us_device(self):
        assert RunTiming(0, 1_000).device_wall_us == pytest.approx(1.0)
        assert RunTiming(0, 7_250).device_wall_us == pytest.approx(7.25)

    def test_us_returns_float(self):
        t = RunTiming(1000, 2000)
        assert isinstance(t.host_wall_us, float)
        assert isinstance(t.device_wall_us, float)

    def test_ns_returns_int(self):
        t = RunTiming(1000, 2000)
        assert isinstance(t.host_wall_ns, int)
        assert isinstance(t.device_wall_ns, int)


class TestRunTimingRepr:
    def test_repr_includes_both_walls(self):
        r = repr(RunTiming(1500, 2500))
        assert "RunTiming" in r
        assert "host_wall_us=1.5" in r
        assert "device_wall_us=2.5" in r

    def test_repr_zero(self):
        r = repr(RunTiming())
        assert "host_wall_us=0" in r
        assert "device_wall_us=0" in r


class TestRunTimingExports:
    """Guard against the symbol getting dropped from the public surface — it's
    re-exported from simpler.worker so end-user code can write
    `from simpler.worker import RunTiming`.
    """

    def test_exported_from_task_interface_module(self):
        import _task_interface  # noqa: PLC0415

        assert hasattr(_task_interface, "RunTiming")
        assert _task_interface.RunTiming is RunTiming

    def test_exported_from_simpler_worker(self):
        from simpler.worker import RunTiming as WorkerRunTiming  # noqa: PLC0415

        assert WorkerRunTiming is RunTiming


class TestWorkerRunSignature:
    """Worker.run / run_prepared signature contract: documented to return a
    RunTiming. We can't actually drive a worker without a device, but we can
    inspect the wrapper signature + docstring to catch accidental drift
    (e.g. someone reverting the return to None).
    """

    def test_worker_run_return_annotation(self):
        from simpler.worker import Worker  # noqa: PLC0415

        sig = inspect.signature(Worker.run)
        # The annotation is the RunTiming class itself (PEP 484 style).
        assert sig.return_annotation is RunTiming, (
            f"Worker.run lost its RunTiming return annotation; got {sig.return_annotation!r}"
        )

    def test_worker_run_prepared_return_annotation(self):
        from simpler.worker import Worker  # noqa: PLC0415

        sig = inspect.signature(Worker.run_prepared)
        assert sig.return_annotation is RunTiming

    def test_chip_worker_wrapper_returns_run_timing(self):
        # The simpler.task_interface.ChipWorker wrapper is what
        # Worker._chip_worker uses internally; it must forward the return
        # rather than swallowing it.
        from simpler.task_interface import ChipWorker  # noqa: PLC0415  # pyright: ignore[reportAttributeAccessIssue]

        src = inspect.getsource(ChipWorker.run_prepared)
        # Surgical drift guard: the wrapper must `return self._impl.run_prepared(...)`
        # rather than calling it for side effects only. If someone reverts to
        # the side-effect-only form, RunTiming would silently become None for
        # every caller without anything else breaking.
        assert "return self._impl.run_prepared" in src
