# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Build a chip-swimlane capture the way a run would write one.

A capture is a wire format, and a test that hand-writes the dict is free to
write one no run could produce. That freedom has cost real coverage: fixtures
have named a runtime whose data they did not carry, carried a phase no
scheduler emits, and omitted the runtime name entirely -- and each of those
passed, because the only thing checking the document's shape was the tool
under test, which the fixture was written against.

So the shape is stated here once. A caller describes a task or a phase; the
column order, the row width, the join key and ``run_epoch`` come from this
module. What a caller cannot express is a document that contradicts itself.

Four things are enforced, each for a fixture that got them wrong:

  The runtime is required -- there is no default, because a capture that does
  not name one cannot be decoded and no run writes one.

  A phase name must be one the named runtime can record. ``tools/hbg`` and
  ``tools/tmr`` already state their own vocabularies, so this reads them
  rather than keeping a third list.

  Row widths and ``run_epoch`` are generated, never passed in.

  A task's AICore row and scheduler row are written together by ``task()``, so
  their join key agrees by construction.

Testing a malformed document stays possible, and stays visible: ``build()``
returns a plain dict, so a test that wants a bad one mutates the result and
says so at the call site. A ``strict=False`` switch would instead make "I am
testing a rejection" and "I forgot a field" look identical.

Two fixture shapes, two entry points. ``Capture`` builds what a collector
writes, for tests that decode. ``sched_record`` builds what a decode produces,
for tests that go straight to the renderer with ``scheduler_phases=``. The
second is a different vocabulary -- microseconds rather than cycles, ``phase``
rather than ``kind`` -- so it is a separate function rather than a mode of the
first.
"""

from __future__ import annotations

from simpler_setup.tools._runtime_dispatch import HBG_RUNTIME, get, resolve_runtime

_DEFAULT_CLOCK_FREQ_HZ = 50_000_000
# The arch a stream reports. Fixed because it is a label on the capture rather
# than part of its shape: nothing the tools decide reads it, and no test needs
# two arches in one document.
_PLATFORM = "a5"


def sched_record(*, phase, start, end, run_epoch=0, **fields):
    """One scheduler-phase record in decoded form, for a renderer's input.

    ``run_epoch`` is stated because the renderer reads it off every record to
    stamp the event it draws -- a record without one is not something a decode
    emits. The rest of a record's fields vary by phase and are passed through.
    """
    return {
        "phase": phase,
        "start_time_us": start,
        "end_time_us": end,
        "run_epoch": run_epoch,
        **fields,
    }


def legal_phase_names(runtime_name):
    """Every phase name a capture of this runtime may carry.

    Read from the runtime's own wire vocabulary (``PHASES``) rather than assembled
    from the tables a renderer or a report happens to key by. Those answer
    presentation questions -- what colour a bar is, what a report row says -- and a
    phase can be legal on the wire without appearing in either.
    """
    return get(runtime_name).PHASES


class Capture:
    """A capture under construction. See the module docstring."""

    def __init__(
        self,
        *,
        runtime,
        level=2,
        clock_freq_hz=_DEFAULT_CLOCK_FREQ_HZ,
        core_types=("aiv",),
        scheduler_producer="aicpu",
    ):
        self.runtime = resolve_runtime(runtime, source="Capture(runtime=...)")
        self.level = level
        self.clock_freq_hz = clock_freq_hz
        self.core_types = list(core_types)
        self._aicore_rows = []
        self._scheduler_rows = []
        self._sched_streams = {}
        self._host_orch_phases = {}
        self._aicpu_orch_phases = {}
        self._host_uploads = []
        self._lifecycle_records = []
        self._metadata_extra = {}
        # Who timed the per-task rows. Below the level that captures phase
        # streams this is all a capture says about its scheduler, so it is
        # stated here; adding an AICore stream sets it to match.
        self._scheduler_producer = scheduler_producer
        self._next_reg_task_id = 1

    # -- metadata ---------------------------------------------------------

    def metadata(self, **fields):
        """Add metadata fields beyond the four every capture carries."""
        self._metadata_extra.update(fields)
        return self

    def host_orchestrated(self, *, origin_ns, clock_domain_id=None, records=1, **capture_fields):
        """Declare the host-orchestration metadata a host-side capture carries.

        ``clock_domain_id`` is omitted unless given: the collector writes it
        only when it could identify the boot it measured against, so a capture
        without one is a real capture, and a test about that case has to be
        able to build it.

        Extra keyword arguments land in ``host_capture`` beside the counters
        below. Like a lifecycle record it is a flat object with nothing for a
        caller to get structurally wrong, and which of its fields a test is
        about is the test's business.
        """
        self._metadata_extra.update(
            {
                "orchestrator_source": "host",
                "host_orchestration_origin_ns": origin_ns,
                "host_capture": {
                    "status": "complete",
                    "expected_records": records,
                    "recorded_records": records,
                    "dropped_records": 0,
                    "error": None,
                    **capture_fields,
                },
            }
        )
        if clock_domain_id is not None:
            self._metadata_extra["host_clock_domain_id"] = clock_domain_id
        return self

    # -- per-task rows ----------------------------------------------------

    def task(
        self,
        *,
        task_id,
        core_id=0,
        start,
        end,
        dispatch=None,
        finish=None,
        receive_to_start=0,
        run_epoch=0,
        reg_task_id=None,
    ):
        """One task's AICore and scheduler rows, sharing a join key.

        ``dispatch`` defaults to just before the kernel starts and ``finish``
        to just after it ends, which is the ordering the decoder requires; a
        test that cares about those gaps passes its own.
        """
        reg = self._next_reg_task_id if reg_task_id is None else reg_task_id
        if reg_task_id is None:
            self._next_reg_task_id += 1
        self.aicore_task(
            task_id=task_id,
            core_id=core_id,
            start=start,
            end=end,
            receive_to_start=receive_to_start,
            run_epoch=run_epoch,
            reg_task_id=reg,
        )
        self.scheduler_task(
            core_id=core_id,
            reg_task_id=reg,
            dispatch=start - 1 if dispatch is None else dispatch,
            finish=end + 1 if finish is None else finish,
            run_epoch=run_epoch,
        )
        return self

    def aicore_task(self, *, task_id, core_id=0, start, end, receive_to_start=0, run_epoch=0, reg_task_id=None):
        """An AICore row on its own, for a capture that has no scheduler timing."""
        reg = self._next_reg_task_id if reg_task_id is None else reg_task_id
        if reg_task_id is None:
            self._next_reg_task_id += 1
        self._aicore_rows.append([core_id, task_id, reg, start, end, receive_to_start, run_epoch])
        return self

    def scheduler_task(self, *, core_id=0, reg_task_id, dispatch, finish, run_epoch=0):
        """A scheduler row on its own, for a task whose AICore row is elsewhere."""
        self._scheduler_rows.append([core_id, reg_task_id, dispatch, finish, run_epoch])
        return self

    # -- phase records ----------------------------------------------------

    def sched_phase(
        self,
        *,
        phase,
        start,
        end,
        thread=0,
        task_id=None,
        tasks_processed=1,
        loop_iter=1,
        run_epoch=0,
        producer="aicpu",
        metrics=None,
    ):
        """One scheduler-phase record on one thread's stream.

        ``metrics`` are the per-record side values the exporter writes in a
        parallel array; the index tying one to its record is filled in here.

        Raises:
            ValueError: the named runtime does not record this phase.
        """
        legal = legal_phase_names(self.runtime)
        if phase not in legal:
            raise ValueError(
                f"{self.runtime} does not record the phase {phase!r}; it records {sorted(legal)}. "
                "A capture carrying a phase its runtime never emits is one no run could write."
            )
        stream = self._sched_streams.setdefault(
            thread,
            {
                "platform": _PLATFORM,
                "producer": producer,
                "scheduler_id": thread,
                "worker_id": thread,
                "core_type": "aicpu",
                "physical_core_id": None,
                "capture": {"committed": 0, "dropped": 0, "truncated": False},
                "records": [],
                "metrics": [],
            },
        )
        stream["records"].append(
            {
                "start_cycles": start,
                "end_cycles": end,
                "run_epoch": run_epoch,
                "loop_iter": loop_iter,
                "kind": phase,
                "tasks_processed": tasks_processed,
                "task_id": task_id,
            }
        )
        stream["capture"]["committed"] = len(stream["records"])
        if metrics is not None:
            stream["metrics"].append({"record_index": len(stream["records"]) - 1, **metrics})
        return self

    def aicore_sched_phase(
        self,
        *,
        phase,
        start,
        end,
        scheduler_id=0,
        task_id=None,
        tasks_processed=1,
        loop_iter=0,
        worker_id=None,
        physical_core_id=None,
        metrics=None,
    ):
        """One record on an AICore scheduler's stream.

        A second producer, not a variant of the first: these records come from
        a scheduler running on an AICore, so the stream names a vector core and
        a physical core id where the AICPU's names neither. The three ids are
        independent -- a scheduler index, the worker slot it occupies and the
        core it runs on -- and default to the same value only for brevity.

        Raises:
            ValueError: this runtime has no AICore scheduler.
        """
        if self.runtime != HBG_RUNTIME:
            raise ValueError(
                f"{self.runtime} has no AICore scheduler, so it records no stream with producer 'aicore'. "
                "Only host_build_graph hands scheduling to an AICore."
            )
        self.sched_phase(
            phase=phase,
            start=start,
            end=end,
            thread=scheduler_id,
            task_id=task_id,
            tasks_processed=tasks_processed,
            loop_iter=loop_iter,
            producer="aicore",
            metrics=metrics,
        )
        stream = self._sched_streams[scheduler_id]
        stream["core_type"] = "aiv"
        stream["worker_id"] = scheduler_id if worker_id is None else worker_id
        stream["physical_core_id"] = scheduler_id if physical_core_id is None else physical_core_id
        # A run that schedules on an AICore says so on both sections: the
        # exporter writes scheduler_tasks.producer and the stream's producer
        # from the same path, so a document naming one of each is not one it
        # could write.
        self._scheduler_producer = "aicore"
        return self

    def aicpu_lifecycle(self, *, thread_id, **cycles):
        """One AICPU thread's lifecycle record.

        The cycle fields pass through: unlike a task row, a lifecycle record is
        a flat object with no width or join key for a caller to get wrong, and
        which of its phases a test cares about is the test's business.
        """
        self._lifecycle_records.append({"aicpu_thread_id": thread_id, **cycles})
        return self

    def host_orch_phase(self, *, task_id, start_ns, end_ns, thread=0, submit_idx=0):
        """One host-orchestrator submit record."""
        self._host_orch_phases.setdefault(thread, []).append(
            {"submit_idx": submit_idx, "task_id": task_id, "start_host_ns": start_ns, "end_host_ns": end_ns}
        )
        return self

    def aicpu_orch_phase(self, *, task_id, start, end, thread=0, submit_idx=0, run_epoch=0):
        """One AICPU-orchestrator submit record."""
        self._aicpu_orch_phases.setdefault(thread, []).append(
            {
                "submit_idx": submit_idx,
                "task_id": task_id,
                "start_cycles": start,
                "end_cycles": end,
                "run_epoch": run_epoch,
            }
        )
        return self

    def host_upload(self, *, phase, start_ns, end_ns, detail=0):
        """One host-to-device upload span."""
        self._host_uploads.append({"phase": phase, "start_host_ns": start_ns, "end_host_ns": end_ns, "detail": detail})
        return self

    # -- result -----------------------------------------------------------

    def build(self):
        """The capture as a plain dict, ready to write as JSON or decode."""
        document = {
            "chip_swimlane_level": self.level,
            "metadata": {
                "runtime": self.runtime,
                "clock_freq_hz": self.clock_freq_hz,
                "num_cores": len(self.core_types),
                "core_types": list(self.core_types),
                **self._metadata_extra,
            },
            "aicore_tasks": [list(row) for row in self._aicore_rows],
        }
        if self._scheduler_rows:
            document["scheduler_tasks"] = {
                "producer": self._scheduler_producer,
                "records": [list(row) for row in self._scheduler_rows],
            }
        if self._sched_streams:
            document["scheduler_records"] = {
                "streams": [self._sched_streams[thread] for thread in sorted(self._sched_streams)]
            }
        if self._host_orch_phases:
            document["host_orchestrator_phases"] = [
                self._host_orch_phases[thread] for thread in sorted(self._host_orch_phases)
            ]
        if self._aicpu_orch_phases:
            document["aicpu_orchestrator_phases"] = [
                self._aicpu_orch_phases[thread] for thread in sorted(self._aicpu_orch_phases)
            ]
        if self._host_uploads:
            document["host_device_uploads"] = list(self._host_uploads)
        if self._lifecycle_records:
            document["aicpu_lifecycle_records"] = list(self._lifecycle_records)
        return document
