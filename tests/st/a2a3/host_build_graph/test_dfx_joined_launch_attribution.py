#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Two collecting runs through ``Worker.submit``, joined on the device, each owning its artifacts.

Early enqueue already carried pairs that collect nothing. The case this suite is about is the one
that used to be refused outright — two runs with every diagnostic owner on, offered to the device
together — and it asserts the two things that capability has to be:

  joined        on hardware, the successor's submission completed while its predecessor was still
                executing. The witness is the receipt the runner publishes on
                ``chip.run.joined_launch`` for a join that succeeded, read in the direction it was
                published and only where it says so: the query answered (``observed=1``,
                ``rc=0``) and found the predecessor's whole-operator boundary **unfired**. A
                successor that joined only after its predecessor had finished publishes a record
                too, with ``unfired=0``, and must not pass. Both runs' epochs are in that record,
                so the evidence names *these* two diagnostic runs and in which order.
  attributed    each artifact carries exactly one run identity, the two runs' identities differ,
                and the identity inside each artifact is the one the submitting child independently
                recorded in ``dispatch_identity.json`` beside it. Two files with two different
                epochs prove nothing on their own: swapping them would pass. Agreeing with the
                sidecar written by a different writer, in a different language, from the same
                frame identity, is what makes a file *that run's*.

Every owner this backend has a device producer for is enabled at once, and each one's artifact is
located in the retained writer's own namespace — not at a default-mode path:

  chip_swimlane  ``swimlane-<n>/records_e<epoch>.json``, one JSON object whose
                 ``metadata.collection`` carries the run epoch, the terminal verdict and the
                 dispatch identity. This is the owner the identity assertions read.
  args_dump      ``args_dump/args_dump.json``, the manifest naming that run's payload file.
  pmu            ``pmu.csv``.

``collect_across_runs`` is on — declared on the class, which is where the framework grants it: it
partitions the Worker group and reaches ``Worker(collect_across_runs=...)``, because retention is
granted once and changes what a returning ``run()`` means. That is the mode the capability is:
retained, background publication, two runs open at once. With it off the collectors keep one
accumulator and reset it per run, which is the default path the no-diagnostics case holds as the
control — and under retention that control reads its output tree only after
``flush_diagnostics()``, the barrier that says the files up to here exist.

**Simulation.** ``simpler_launch_run_joined`` is ``PTO_RUNTIME_ERR_UNSUPPORTED`` on the simulated
backend and ``supports_joined_native_launch_ctx`` answers 0, because a simulated context takes the
execution claim at prepare and never holds two runs. So the sim rows assert the other half: the
pair serializes, no join is published, and every attribution claim above still holds. The joined
receipt is asserted on the onboard rows, where a join is reachable.

**On host_build_graph.** The graph is built on the host, so dep_gen and scope_stats have no device
producer here and are not terms in the pair decision. The owners judged are the swimlane, args-dump
and PMU, and the artifact the identity assertions read is the swimlane's, exactly as above: it is
the one owner every backend has a device producer for.

The mixed-mode case is a requirement rather than a degradation. A pair whose settings disagree —
a different swimlane level, a different PMU event mode — cannot join, because each is one word in
a device header the device latches once. The runs then go in queue order, publish no joined
receipt, and *both* keep a complete diagnostic. Nothing is turned off to make them fit. That
absence is read only once both runs' own host spans are in the window, which the per-process
writer's ordering makes a boundary rather than a deadline.

Artifacts are resolved by searching below the prefix, never by assuming a flat path: a chip child
rewrites its own output into a ``rank<N>/d<N>`` namespace of its own, and which child served a run
is not this process's to predict.
"""

import contextlib
import json
import os
import struct
import time
from collections import Counter
from pathlib import Path

import torch
from simpler.task_interface import ArgDirection as D
from simpler.task_interface import DataType, TaskArgs, Tensor, TensorArgType, TensorTransfer

from simpler_setup import SceneTestCase, scene_test
from simpler_setup.tools.strace_timing import parse_spans

#: The payload that keeps a run on the device long enough for its successor to be offered beside
#: it, with the chain and the spin this tree's own early-enqueue suite uses.
KERNELS_BASE = "worker_async_fifo/kernels"
_CHAIN_LENGTH = 512
#: Bounded so a regression fails on an assertion rather than on the op-execute timeout. Zero in
#: simulation, where the kernel runs on a host thread and no pair can be joined anyway.
_SPIN_ITERS = 200_000_000

#: What each run of a pair fills its two inputs with — `value` and `value + 1`. No two runs share
#: one, so a record holding the other run's value is the two collections crossing in one artifact.
_PAIR_INPUT_VALUES = (5.0, 11.0)

SIZE = 128 * 128
DTYPE = torch.float32
_F32 = DataType.FLOAT32

#: A swimlane level the joined pair decision admits. Levels 3 and 4 arm phase
#: pools whose ordering has not been established and are refused.
_SWIMLANE_TASK_TIMING = 1
#: Any PMU event mode works; what matters is that a joined pair names the same
#: one, because it selects the counter group the device programs.
_PMU_PIPE_UTILIZATION = 2

#: Each owner's published artifact, as a pattern below the run's own namespace. These are the
#: *retained* writers' names: the swimlane reserves a directory per run and writes one file per
#: epoch into it, and the default-mode `chip_swimlane_records.json` is not written on this path.
_OWNER_ARTIFACTS = {
    "chip_swimlane": "swimlane-*/records_e*.json",
    "dump_args": "args_dump/args_dump.json",
    "pmu": "pmu.csv",
    "dep_gen": "deps.json",
    "scope_stats": "scope_stats/scope_stats.jsonl",
}
#: The owner whose artifact carries the run epoch *and* the dispatch identity, which is what the
#: attribution assertions read. Every backend has a device producer for it.
_IDENTITY_OWNER = "chip_swimlane"
#: The verdicts that leave a readable artifact. A run that settled as any of these published a
#: complete terminal status; anything else means the collector could not settle this run.
_PUBLISHING_VERDICTS = frozenset({"published", "partial_safe", "partial_cut_unknown"})

#: The span one *successful* joined launch publishes, carrying both runs' epochs and what the
#: successor read of its predecessor's boundary once its own submission had completed. The runner
#: emits it only once that submission has succeeded, and emits it whatever the ordering query
#: answered — so its presence is the join and its fields are the ordering.
_JOIN_SPAN = "chip.run.joined_launch"
#: A run's own enclosing host span, written when that run finalizes. One per run, and — because a
#: process's host log is drained by a single writer in enqueue order — always *after* any joined
#: record that run published. Seeing both runs' is therefore the observation boundary: any joined
#: record for the pair is already in the window, so an absence is an absence.
_RUN_SPAN = "chip.run"
#: How long to wait for that boundary. The writers are asynchronous, so this covers the lag
#: between a record being accepted and written — not the runs, which have been waited for.
#: Exhausting it is a failed observation, which is reported as one rather than as an absence.
_EVIDENCE_BUDGET_S = 10.0

_DIR_TAGS = {
    D.IN: TensorArgType.INPUT,
    D.OUT: TensorArgType.OUTPUT_EXISTING,
    D.INOUT: TensorArgType.INOUT,
}


def _golden(a, b):
    return a + b + _CHAIN_LENGTH * b


def _find_artifacts(prefix: Path, owner: str) -> list:
    """This run's published files for `owner`, wherever its child put them.

    A chip child rewrites its output into a ``rank<N>/d<N>`` namespace it owns, so the path below a
    run's prefix is the child's to choose. Searching for the writer's own pattern is what makes the
    assertion about the run that named this prefix rather than about a layout.
    """
    if not prefix.exists():
        return []
    return sorted(prefix.rglob(_OWNER_ARTIFACTS[owner]))


def _read_collection(path: Path) -> dict:
    """The ``metadata.collection`` object a retained swimlane artifact carries."""
    document = json.loads(path.read_text())
    metadata = document.get("metadata", {})
    collection = metadata.get("collection")
    assert collection is not None, f"{path} carries no retained collection verdict: {sorted(metadata)}"
    return collection


def _read_sidecar(path: Path) -> dict:
    """The dispatch identity the submitting child wrote beside its artifacts."""
    return json.loads(path.read_text())


def _aicore_executions(path: Path, epoch) -> Counter:
    """How many AICore executions this run had per task, from the swimlane's own rows.

    `aicore_tasks` rows are `[core_id, task_token, reg_task_id, start, end, receive_to_start,
    run_epoch]`, one per dispatched subtask. This is the run's independent record of what
    executed: it comes from a different producer than PMU does, so a PMU row that is absent
    cannot hide behind the absence of the evidence for it. A mix task appears once per subtask,
    which is why these are counted rather than collected into a set.
    """
    document = json.loads(path.read_text())
    rows = document.get("aicore_tasks", [])
    return Counter(row[1] for row in rows if len(row) >= 7 and row[6] == epoch)


def _read_args_manifest(path: Path) -> dict:
    """One run's argument-dump manifest."""
    return json.loads(path.read_text())


def _first_float32(payload: bytes, arg: dict) -> float:
    """The first element of the payload one argument record names."""
    return struct.unpack_from("<f", payload, arg["bin_offset"])[0]


def _record_value(payload: bytes, tensor_args: list, task_id: int, arg_index: int, stage: str, epoch) -> float:
    """The value one named argument of one task reports at one capture stage, in this run.

    Matched on all four of run, task, argument index and stage, because a capture is one record per
    `(run, task, argument, stage)` and nothing weaker identifies one: the union of a task's
    arguments at one stage is satisfied by a value that landed on the wrong argument. Exactly one
    record may match, and an index the producer could not resolve is not an index — a record
    carrying `arg_index_ambiguous` is refused rather than read as the argument it names.
    """
    matches = [
        arg
        for arg in tensor_args
        if int(arg["task_id"], 16) == task_id
        and arg.get("arg_index") == arg_index
        and arg.get("stage") == stage
        and arg.get("run_epoch") == epoch
    ]
    assert len(matches) == 1, (
        f"run {epoch}'s task 0x{task_id:x} has {len(matches)} records for argument {arg_index} at {stage}, "
        f"where a capture is one record per run, task, argument and stage"
    )
    assert matches[0].get("arg_index_ambiguous") is not True, (
        f"run {epoch}'s task 0x{task_id:x} reports argument {arg_index} at {stage} with an unresolved index"
    )
    return _first_float32(payload, matches[0])


def _cycles_by_core(rows: list) -> str:
    """Each core's row count and its highest cycle total, for a zero-cycle failure to carry.

    A counting window is per core, so a run whose rows are all zero and a run where only some
    cores report are different defects, and the rows already hold which is which. This is the
    message's content only — the assertion it accompanies is unchanged.
    """
    per_core: dict = {}
    for row in rows:
        counted, highest = per_core.get(row[1], (0, 0))
        per_core[row[1]] = (counted + 1, max(highest, int(row[5])))
    return ", ".join(
        f"core {core}: {counted} rows, max {highest} cycles" for core, (counted, highest) in sorted(per_core.items())
    )


def _execution_order(path: Path, epoch) -> list:
    """The tasks this run executed, ordered by when the swimlane says each started.

    Identity for the content checks below has to come from somewhere other than the content.
    These rows are the chip swimlane's, written by a different producer from the argument dump,
    and `start_cycles` is column 3 — so which of two tasks ran first is established without
    reading a single argument byte. A mix task appears once per subtask; the earliest start of
    any of them is the task's.
    """
    document = json.loads(path.read_text())
    earliest = {}
    for row in document.get("aicore_tasks", []):
        if len(row) < 7 or row[6] != epoch:
            continue
        token, start = row[1], row[3]
        if token not in earliest or start < earliest[token]:
            earliest[token] = start
    return [token for token, _ in sorted(earliest.items(), key=lambda pair: pair[1])]


def _read_pmu_rows(path: Path):
    """One run's PMU CSV as its header fields and its data rows."""
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    assert lines, f"{path} is empty"
    header = lines[0].split(",")
    return header, [line.split(",") for line in lines[1:]]


def _read_scope_stats(path: Path):
    """One scope-stats artifact's metadata line and the run identities in its records."""
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    assert lines, f"{path} is empty"
    metadata = json.loads(lines[0])
    records = [json.loads(line) for line in lines[1:]]
    assert records, f"{path} carries metadata but no records"
    return metadata, {record["run_epoch"] for record in records}


def _attributes(span) -> dict:
    """One span's ``k=v`` attribute string as a dict."""
    return dict(pair.split("=", 1) for pair in span.attrs.split() if "=" in pair)


def _join_records(spans, pid) -> list:
    """Every joined-launch record one child published, as records rather than as a set.

    The runner emits this span whatever its ordering query answered, so the fields decide what a
    record means and none of them may be discarded: ``observed`` says the query was answered,
    ``unfired`` says the predecessor's whole-operator boundary had not fired when the successor's
    submission completed, and ``rc`` says why a query could not answer. A record that cannot be
    read is kept as itself, because an unreadable record is a different finding from no record.
    """
    records = []
    for span in spans:
        if span.name != _JOIN_SPAN or span.pid != pid:
            continue
        attributes = _attributes(span)
        try:
            records.append(
                {
                    "successor": int(attributes["s_epoch"]),
                    "predecessor": int(attributes["p_epoch"]),
                    "observed": attributes["observed"] == "1",
                    "unfired": attributes["unfired"] == "1",
                    "query_rc": int(attributes["rc"]),
                }
            )
        except (KeyError, ValueError):
            records.append({"malformed": span.attrs})
    return records


def _established_pairs(records) -> set:
    """The ``(predecessor, successor)`` epochs whose ordering these records establish.

    A record counts only when the query answered, it answered 0, and the predecessor's
    whole-operator boundary had **not** fired: that pair's successor completed its native
    submission before its predecessor completed, which is the only reading that places the two on
    the device together. A fired boundary means the submission and the completion raced and
    nothing is claimed; a failed query counts for neither side. Both are reported by the caller
    rather than read as a success.

    Ordered, never a set of two epochs: a joined API invocation taken after the predecessor had
    completed would satisfy an unordered membership test, which is exactly the implementation this
    acceptance has to reject.
    """
    return {
        (record["predecessor"], record["successor"])
        for record in records
        if record.get("observed")
        and record.get("unfired")
        and record.get("query_rc") == 0
        and record["successor"] != record["predecessor"]
    }


def _child_log_directory() -> str:
    """Where this worker's chip children append their host logs.

    The children inherited the parent's logger state at fork, so either the parent was already
    bound and they write where it writes, or neither was and they bind the same process-tree spool
    the parent's own module constant names.
    """
    from _task_interface import _host_log_directory  # noqa: PLC0415
    from simpler.task_interface import _bind_host_log_session_directory  # noqa: PLC0415

    return _host_log_directory() or _bind_host_log_session_directory()


def _with_child_evidence(trace, what: str, call):
    """Run `call`, attaching the children's log window to whatever it raises.

    The HBG failure this exists for happens inside the drain, before any assertion of this
    suite's own runs — so a message built at an assertion would never be reached. The original
    exception stays the cause of the one raised here, and a window that could not be read is
    reported rather than swallowed: losing the first failure to a second one while collecting
    evidence about it would be the worse trade.
    """
    try:
        return call()
    except BaseException as error:
        raise RuntimeError(f"{what}: {error}\n{trace.safe_window_text()}") from error


class _ChildTrace:
    """A byte window over each of this worker's chip children's host logs.

    The host log is one append-only file per process (``host.<pid>.log``), so a window into it is
    a byte offset per file. A count over the concatenation of several files is not a window: a
    file that sorts earlier appending between two reads shifts every later record.

    Opened at the files' current size, so the window begins where this object was created — which
    is what keeps an earlier pair's records, and an earlier session that happened to reuse a pid,
    out of a later pair's evidence. The offsets are never advanced: a caller polls the same window
    until the records it needs are in it.
    """

    def __init__(self, worker):
        directory = Path(_child_log_directory())
        self._paths = {pid: directory / f"host.{pid}.log" for pid in worker._chip_pids}  # noqa: SLF001
        self._offsets = {pid: self._size(path) for pid, path in self._paths.items()}

    @staticmethod
    def _size(path: Path) -> int:
        try:
            return path.stat().st_size
        except OSError:
            return 0

    def spans(self) -> list:
        """Every whole-line span this worker's own children appended since the window opened."""
        collected = []
        for pid, path in self._paths.items():
            blob = b""
            with contextlib.suppress(OSError):
                blob = path.read_bytes()
            window = blob[self._offsets[pid] :]
            consumed = window.rfind(b"\n") + 1
            if consumed <= 0:
                continue
            lines = window[:consumed].decode("utf-8", errors="replace").splitlines()
            collected.extend(span for span in parse_spans(lines) if span.pid == pid)
        return collected

    def window_text(self, *, budget: int = 16384) -> str:
        """The bytes each child appended since this window opened, for a failure to carry.

        The collectors state why a run was refused in the child's own host log, and nothing
        collects that file in ordinary CI — so a failure that does not carry the window is a
        failure with no cause attached. This is that window verbatim rather than parsed: the
        reason lines are not spans and no parser here knows their shape.

        Only the tail is read, not read-then-sliced: a diagnostic run's log is as large as the
        run was long, and a failure report must not pull all of it into memory to quote 16 KiB.
        The tail is where a refusal is written, since it is stated as the run ends.

        Every child is named even when it appended nothing, and a file that is missing or
        shorter than its own offset says so — a truncated window is itself evidence about the
        run, and reporting it as empty would hide that.
        """
        chunks = []
        for pid, path in sorted(self._paths.items()):
            opened_at = self._offsets[pid]
            try:
                with path.open("rb") as handle:
                    size = os.fstat(handle.fileno()).st_size
                    if size < opened_at:
                        chunks.append(
                            f"--- child {pid}: {path} is {size} B, shorter than the {opened_at} B this "
                            f"window opened at; it was replaced or truncated ---"
                        )
                        continue
                    if size == opened_at:
                        chunks.append(f"--- child {pid}: nothing appended since byte {opened_at} ---")
                        continue
                    start = max(opened_at, size - budget)
                    handle.seek(start)
                    kept = handle.read(size - start)
            except OSError as error:
                chunks.append(f"--- child {pid}: {path} unreadable ({error.__class__.__name__}) ---")
                continue
            elided = start - opened_at
            head = (
                f"--- child {pid}: {size - opened_at} B since byte {opened_at}"
                + (f", first {elided} B elided" if elided else "")
                + " ---"
            )
            chunks.append(head + "\n" + kept.decode("utf-8", errors="replace").rstrip("\n"))
        return "\n".join(chunks) if chunks else "--- this worker reported no chip children ---"

    def safe_window_text(self) -> str:
        """`window_text` that never raises, so it cannot replace the failure it is reporting."""
        try:
            return self.window_text()
        except BaseException as error:  # noqa: BLE001 -- reported in place of the window it could not read
            return f"--- the child log window could not be read: {error!r} ---"


@scene_test(level=3, runtime="host_build_graph", collect_across_runs=True)
class TestDfxJoinedLaunchAttribution(SceneTestCase):
    """Two diagnostic runs over the public API, joined on the device, each owning its records."""

    #: Every owner this runtime has a device producer for. host_build_graph builds its graph on
    #: the host, so dep_gen and scope_stats have none there and its copy names a different set.
    #: Which argument of each marked task carries the handoff, by the index it holds in its own
    #: task's argument list: the first task's output tensor, then the second task's input and its
    #: output. The first task is `kDelayedAdd(a, b, out)`, the second `kAddScalar(src, out)`.
    PRODUCER_OUTPUT_ARG = 2
    CONSUMER_INPUT_ARG = 0
    CONSUMER_OUTPUT_ARG = 1

    DEVICE_OWNERS = ("chip_swimlane", "dump_args", "pmu")

    CALLABLE = {
        "callables": [
            {
                "name": "vector",
                "orchestration": {
                    "source": f"{KERNELS_BASE}/orchestration/pipelined_vector_orch.cpp",
                    "function_name": "aicpu_orchestration_entry",
                    "signature": [D.IN, D.IN, D.OUT, D.IN],
                },
                "incores": [
                    {
                        "func_id": 0,
                        "source": f"{KERNELS_BASE}/aiv/delayed_add.cpp",
                        "core_type": "aiv",
                        "signature": [D.IN, D.IN, D.OUT],
                    },
                    {
                        "func_id": 1,
                        "source": f"{KERNELS_BASE}/aiv/kernel_add_scalar.cpp",
                        "core_type": "aiv",
                        "signature": [D.IN, D.OUT],
                    },
                ],
            },
        ],
    }

    #: `launch_depth=2` authorizes a successor beside its predecessor; the retained mode the
    #: capability *is* comes from the class's own `collect_across_runs`. The control varies the
    #: diagnostics and keeps both runs, so what it controls for is the diagnostics rather than the
    #: depth or the mode.
    CASES = [
        {
            "name": "agreeing_pair",
            "platforms": ["a2a3sim", "a2a3"],
            "config": {
                "device_count": 1,
                "num_sub_workers": 0,
                "launch_depth": 2,
            },
        },
    ]

    def test_run(self, st_platform, st_worker, tmp_path):
        worker = st_worker
        simulated = st_platform.endswith("sim")

        # One window per pair, opened before the pair is submitted, so no pair can supply another
        # pair's evidence.
        trace = _ChildTrace(worker)
        agreeing = self._assert_pair_owns_its_records(
            worker, tmp_path / "agreeing", simulated, disagree=False, trace=trace
        )
        self._assert_join_evidence(trace, agreeing, expect_joined=not simulated, what="an agreeing pair")

        trace = _ChildTrace(worker)
        mixed = self._assert_pair_owns_its_records(worker, tmp_path / "mixed", simulated, disagree=True, trace=trace)
        self._assert_join_evidence(trace, mixed, expect_joined=False, what="a disagreeing pair")

        self._assert_no_artifact_without_diagnostics(worker, tmp_path / "off")

    # -- configurations --------------------------------------------------

    def _diagnostic_config(self, prefix: Path, *, shifted: bool):
        """Every owner this runtime has a device producer for, enabled at once.

        `shifted` moves exactly one setting without turning any owner off, which is what makes a
        pair disagree for the reason the pair decision is about: two runs naming different settings
        for one shared device word, rather than one run collecting less than the other.
        """
        owners = set(type(self).DEVICE_OWNERS)
        return self._build_config(
            self.CASES[0]["config"],
            enable_chip_swimlane=(2 if shifted else _SWIMLANE_TASK_TIMING) if "chip_swimlane" in owners else 0,
            enable_dump_args="dump_args" in owners,
            enable_pmu=_PMU_PIPE_UTILIZATION if "pmu" in owners else 0,
            enable_dep_gen="dep_gen" in owners,
            enable_scope_stats="scope_stats" in owners,
            output_prefix=str(prefix),
        )

    def _checked_owners(self):
        """The owners this backend publishes a file for."""
        return [owner for owner in type(self).DEVICE_OWNERS if owner in _OWNER_ARTIFACTS]

    # -- submission ------------------------------------------------------

    def _chip_args(self, buffers, spin_iters):
        """One chip task's args: this run's three tensors, then the control this orch reads.

        Host orchestration reads the addend separately from its H2D device operand, which is what
        the fourth signature entry is: the same buffer, named again with no transfer of its own.
        """
        signature = type(self)._st_chip_handles["vector_sig"]
        args = TaskArgs()
        for buffer, direction in zip(buffers, signature):
            args.add_tensor(buffer.tensor((SIZE,), _F32), _DIR_TAGS[direction])
        args.add_tensor(
            Tensor(buffers[1], shapes=(SIZE,), dtype=_F32),
            TensorArgType.INPUT,
            transfer=TensorTransfer.NONE,
        )
        args.add_scalar(spin_iters)
        return args

    def _submit(self, worker, buffers, chip_config, spin_iters):
        handle = type(self)._st_chip_handles["vector"]

        def graph(orch, _args, _cfg):
            orch.submit_next_level(handle, self._chip_args(buffers, spin_iters), chip_config, worker=0)

        return worker.submit(graph)

    def _submit_pair(self, worker, root: Path, simulated: bool, *, disagree: bool):
        """Two submissions, each with its own buffers and its own destination.

        Both are in flight before either is waited on: that is what offers the pair to the backend
        at all. Waiting on the first before submitting the second would serialize them here, in the
        test, and both the join and the attribution assertion would be about nothing.

        Every accepted handle is returned even when a later submission raises, because an accepted
        run is the caller's to drain whatever happened after it: abandoning one leaves its slot and
        its buffers held for the worker's life.
        """
        root.mkdir(parents=True, exist_ok=True)
        spin_iters = 0 if simulated else _SPIN_ITERS
        handles = []
        views = []
        try:
            for index, value in enumerate(_PAIR_INPUT_VALUES):
                nbytes = SIZE * DTYPE.itemsize
                buffers = [worker.create_buffer(nbytes) for _ in range(3)]
                tensors = [torch.frombuffer(buffer.shm.buf, dtype=DTYPE, count=SIZE) for buffer in buffers]
                tensors[0].fill_(value)
                tensors[1].fill_(value + 1.0)
                tensors[2].zero_()
                prefix = root / f"run{index}"
                # Only the second run's settings move, so the pair agrees or disagrees by exactly
                # one word.
                chip_config = self._diagnostic_config(prefix, shifted=disagree and index == 1)
                handles.append(self._submit(worker, buffers, chip_config, spin_iters))
                views.append((tensors, prefix))
        except BaseException:
            # Whatever was accepted still has to retire, and its views have to go before anything
            # can unlink the memory behind them. The drain's own failure must not replace the one
            # being unwound, so it is suppressed here and the first cause is re-raised.
            with contextlib.suppress(BaseException):
                self._drain(handles)
            views.clear()
            raise
        return handles, views

    @staticmethod
    def _drain(handles):
        """Wait on every accepted handle, reporting the first failure last.

        A failure in one wait must not skip the rest: each remaining run is still on the device and
        still owns its slot.
        """
        first_error = None
        for handle in handles:
            try:
                handle.wait()
            except BaseException as error:  # noqa: BLE001 -- re-raised below, after every run is drained
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error

    # -- the cases -------------------------------------------------------

    def _assert_pair_owns_its_records(self, worker, root: Path, simulated: bool, *, disagree: bool, trace):
        """Submit a pair, check both results and every artifact, and return what each run is.

        The returned records are in submission order, so the first is the predecessor and the
        second the successor — which is the direction the joined-launch assertion requires.

        `trace` is the window these children's reasons are read out of when anything here fails.
        """
        what = "a disagreeing pair" if disagree else "an agreeing pair"
        return _with_child_evidence(
            trace, what, lambda: self._pair_owns_its_records(worker, root, simulated, disagree=disagree, what=what)
        )

    def _pair_owns_its_records(self, worker, root: Path, simulated: bool, *, disagree: bool, what: str):
        """The pair case itself, with its failures carried by the caller above."""
        handles, views = self._submit_pair(worker, root, simulated, disagree=disagree)
        try:
            self._drain(handles)
            results = [
                torch.allclose(tensors[2], _golden(tensors[0], tensors[1]), rtol=self.RTOL, atol=self.ATOL)
                for tensors, _prefix in views
            ]
            prefixes = [prefix for _tensors, prefix in views]
        finally:
            # Dropped before the framework-driven close can unlink the shm.
            views.clear()

        for index, ok in enumerate(results):
            assert ok, f"{what}: run {index} did not produce its own golden result"

        worker.flush_diagnostics()

        runs = []
        for index, prefix in enumerate(prefixes):
            for owner in self._checked_owners():
                found = _find_artifacts(prefix, owner)
                assert found, (
                    f"{what}: run {index} enabled {owner} and published no artifact below {prefix}; "
                    f"an owner that did not admit the run is indistinguishable from one that did "
                    f"and wrote nothing"
                )
            runs.append(self._assert_one_run_owns_this_namespace(prefix, index, simulated, what))

        assert runs[0]["epoch"] != runs[1]["epoch"], (
            f"{what}: both artifacts carry run epoch {runs[0]['epoch']}, so one run published under the other's name"
        )
        assert runs[0]["run_id"] != runs[1]["run_id"], (
            f"{what}: both runs reported dispatch identity {runs[0]['run_id']}"
        )
        return runs

    def _assert_one_run_owns_this_namespace(self, prefix: Path, index: int, simulated: bool, what: str) -> dict:
        """Everything one run's own artifacts must agree about. Returns what identifies it."""
        identity_files = _find_artifacts(prefix, _IDENTITY_OWNER)
        assert len(identity_files) == 1, (
            f"{what}: run {index} published {len(identity_files)} {_IDENTITY_OWNER} artifacts below {prefix}; "
            f"one run writes one"
        )
        collection = _read_collection(identity_files[0])

        # The terminal status, as the writer settled it.
        assert collection.get("verdict") in _PUBLISHING_VERDICTS, (
            f"{what}: run {index} settled as {collection.get('verdict')!r}, which leaves no readable artifact"
        )
        assert collection.get("processing_complete") is True, f"{what}: run {index} was published unprocessed"
        assert collection.get("not_received_buffers") == 0, (
            f"{what}: run {index} published without {collection.get('not_received_buffers')} of its buffers"
        )
        assert collection.get("unpublished_loss") == 0, (
            f"{what}: run {index} lost {collection.get('unpublished_loss')} records before publication"
        )

        epoch = collection.get("run_epoch")
        assert epoch, f"{what}: run {index}'s artifact carries no run identity: {collection!r}"

        # The sidecar the submitting child wrote beside it, from the same frame identity through a
        # different writer. Agreement is what makes this file *this* run's rather than a file that
        # happens to sit under this prefix.
        sidecars = sorted(prefix.rglob("dispatch_identity.json"))
        assert len(sidecars) == 1, f"{what}: run {index} has {len(sidecars)} dispatch identities below {prefix}"
        sidecar = _read_sidecar(sidecars[0])
        dispatch = collection.get("dispatch", {})
        assert dispatch.get("status") == "identified", (
            f"{what}: run {index}'s artifact names no submission: {dispatch!r}"
        )
        for field in ("run_id", "owner_instance_id", "endpoint_dispatch_id", "pipeline_slot", "pipeline_generation"):
            assert dispatch.get(field) == sidecar.get(field), (
                f"{what}: run {index}'s artifact reports {field}={dispatch.get(field)!r} where the child that "
                f"submitted it recorded {sidecar.get(field)!r}"
            )

        # Every other owner that carries per-record identities must name the same run.
        if "scope_stats" in self._checked_owners():
            metadata, record_epochs = _read_scope_stats(_find_artifacts(prefix, "scope_stats")[0])
            assert metadata.get("fatal") is False, f"{what}: run {index} latched a scope-stats fatal: {metadata!r}"
            assert metadata.get("dropped", 0) == 0, f"{what}: run {index} dropped scope records: {metadata!r}"
            assert record_epochs == {epoch}, (
                f"{what}: run {index}'s scope records name {sorted(record_epochs)} where its swimlane artifact "
                f"names {epoch} — one run's records were attributed to another run's artifact"
            )

        # The argument dump, read for its content rather than for its existence. These runs
        # execute a chain of tasks that each name three tensors, so a manifest with no args is an
        # owner that was admitted and captured nothing — which the file alone cannot be told from
        # one that captured everything there was. Every record must also name this run: an arg
        # carrying the other epoch is the pair's two collections crossing.
        if "dump_args" in self._checked_owners():
            manifest_path = _find_artifacts(prefix, "dump_args")[0]
            manifest = _read_args_manifest(manifest_path)
            assert manifest.get("collection_verdict") in _PUBLISHING_VERDICTS, (
                f"{what}: run {index} settled its args dump as {manifest.get('collection_verdict')!r}"
            )
            assert manifest.get("counts_unknown") is False, (
                f"{what}: run {index} published an args dump whose producer accounting was unreadable"
            )
            args = manifest.get("args", [])
            assert manifest.get("total_args", 0) > 0 and args, (
                f"{what}: run {index} enabled the argument dump over tasks that carry tensor arguments and "
                f"recorded none of them; {manifest_path} is the empty manifest an owner with no capture "
                f"publishes"
            )
            arg_epochs = {arg.get("run_epoch") for arg in args}
            assert arg_epochs == {epoch}, (
                f"{what}: run {index}'s argument records name {sorted(arg_epochs)} where its swimlane artifact "
                f"names {epoch}"
            )
            # The payload the manifest points at, which is what makes the records content rather
            # than a list of shapes.
            bin_file = manifest.get("bin_file")
            assert bin_file, f"{what}: run {index}'s args dump names no payload file"
            payload = manifest_path.parent / bin_file
            assert payload.is_file() and payload.stat().st_size > 0, (
                f"{what}: run {index}'s args dump names {bin_file}, which holds no bytes"
            )

            # And the bytes are this run's own data. The first task's inputs are the buffers this
            # process seeded, and each run seeded a value no other run used, so decoding them is
            # what separates "a payload exists" from "the payload is this run's". A record whose
            # content came from the pair's other run, or from before that run filled its inputs,
            # shows up here and nowhere else.
            payload_bytes = payload.read_bytes()
            seeded = {_PAIR_INPUT_VALUES[index], _PAIR_INPUT_VALUES[index] + 1.0}
            tensor_args = [
                arg
                for arg in args
                if arg.get("dtype") == "FLOAT32" and arg.get("kind") == "tensor" and arg.get("bin_size", 0) >= 4
            ]
            # The handoff between the two marked tasks, at pinned arithmetic values, read one
            # argument at a time. The orchestration marks the first two tasks: the first computes
            # `a + b` into a buffer, the second reads that buffer and adds `b` to it, and every
            # later task in the chain writes the same buffer again. So the values below are the
            # ones a per-task capture must report, and a capture taken at the run's boundaries
            # reports the pre-run content on one side and the chain's final value on the other.
            #
            # Which of the two ran first comes from the swimlane's `start_cycles`, not from the
            # bytes being checked, so no value here is both the oracle and the subject.
            a = _PAIR_INPUT_VALUES[index]
            b = a + 1.0
            final = float(_golden(a, b))
            captured_tasks = {int(arg["task_id"], 16) for arg in tensor_args}
            assert len(captured_tasks) == 2, (
                f"{what}: run {index} captured {sorted(captured_tasks)}; the workload marks exactly two tasks, "
                f"so anything else means the records are not the ones this check reads"
            )
            marked = [token for token in _execution_order(identity_files[0], epoch) if token in captured_tasks]
            assert len(marked) == 2, (
                f"{what}: the swimlane orders {marked} of the two captured tasks {sorted(captured_tasks)}; both "
                f"must appear there for their order to be established independently"
            )
            producer_id, consumer_id = marked
            cls = type(self)

            # The inputs this process seeded, each against the argument that reads it, so a record
            # holding the pair's other run's values — or this buffer's content from before the run
            # filled it — fails on the argument that carried it.
            for arg_index, seed in enumerate((a, b)):
                read = _record_value(payload_bytes, tensor_args, producer_id, arg_index, "before_dispatch", epoch)
                assert read == seed, (
                    f"{what}: run {index}'s first marked task reports {read} for argument {arg_index} at its own "
                    f"dispatch where it reads {seed}; this run seeded {sorted(seeded)}"
                )

            produced = _record_value(
                payload_bytes, tensor_args, producer_id, cls.PRODUCER_OUTPUT_ARG, "after_completion", epoch
            )
            assert produced == a + b, (
                f"{what}: run {index}'s first marked task reports {produced} for argument "
                f"{cls.PRODUCER_OUTPUT_ARG} at its own completion where it computed a + b = {a + b}; the buffer "
                f"held {sorted(seeded)} before the run and {final} after"
            )
            consumed = _record_value(
                payload_bytes, tensor_args, consumer_id, cls.CONSUMER_INPUT_ARG, "before_dispatch", epoch
            )
            assert consumed == a + b, (
                f"{what}: run {index}'s second marked task reports {consumed} for argument "
                f"{cls.CONSUMER_INPUT_ARG} at its own dispatch where it reads the a + b = {a + b} its predecessor "
                f"wrote; the buffer held {sorted(seeded)} before the run and {final} after"
            )
            consumer_output = _record_value(
                payload_bytes, tensor_args, consumer_id, cls.CONSUMER_OUTPUT_ARG, "after_completion", epoch
            )
            assert consumer_output == a + 2.0 * b, (
                f"{what}: run {index}'s second marked task reports {consumer_output} for argument "
                f"{cls.CONSUMER_OUTPUT_ARG} at its own completion where it computed (a + b) + b = {a + 2.0 * b}; "
                f"{final} is what the chain left in that buffer"
            )

        # PMU, against what this run is independently known to have executed. A count of rows
        # cannot catch a lost measurement, and the cells cannot vouch for themselves, so the
        # expectation comes from the swimlane's own AICore rows: one PMU row per executed
        # subtask, naming the same task. The cycle *value* is asserted only on hardware — a
        # simulated backend programs no counters, so a zero there is honest rather than a loss.
        #
        # The cycle check below stays exact. What its message may not do is name a cause: rows
        # that carry correct identities and zero totals are equally consistent with a framework
        # enable that did not stick, a counting window the AICore never gated open, a counting
        # range that measured nothing, and a register table whose entry for this core is not
        # this core's. So the message reports the observation and lists what it does not
        # separate.
        if "pmu" in self._checked_owners():
            header, rows = _read_pmu_rows(_find_artifacts(prefix, "pmu")[0])
            assert header[:6] == ["thread_id", "core_id", "task_id", "func_id", "core_type", "pmu_total_cycles"], (
                f"{what}: run {index}'s PMU csv does not carry the columns this check reads: {header!r}"
            )
            for row in rows:
                assert len(row) >= 6 and row[2].strip(), (
                    f"{what}: run {index} published a PMU row naming no task: {row!r}"
                )
            executed = _aicore_executions(identity_files[0], epoch)
            assert executed, (
                f"{what}: run {index} published no AICore task rows, so there is no independent record of "
                f"what it executed to check its measurements against"
            )
            measured = Counter(int(row[2], 16) for row in rows)
            assert measured == executed, (
                f"{what}: run {index} executed {sum(executed.values())} AICore subtasks and measured "
                f"{sum(measured.values())}; unmeasured {sorted((executed - measured).elements())}, "
                f"measured-but-not-executed {sorted((measured - executed).elements())}"
            )
            if not simulated:
                assert any(int(row[5]) > 0 for row in rows), (
                    f"{what}: run {index} measured every executed subtask and reports zero cycles on all "
                    f"{len(rows)} of its PMU rows; per core: {_cycles_by_core(rows)}. The rows carry this run's "
                    f"own task identities, so the records were produced and attributed — what is zero is only the "
                    f"counter values. This does not establish which of a lost framework enable, an AICore gate "
                    f"that never opened, a counting range that measured nothing, or a register table entry that "
                    f"is not this core's produced it. The collector states this run's control read-back on one "
                    f"`PmuCollector terminal: run ...` line, and that line reports two instants from two "
                    f"separate fields: `enabled` / `not-enabled` are what each core's init read **before any "
                    f"task ran**, and `held` / `lost` are what its own close read **after every task had**. The "
                    f"pair is the discriminator — `enabled` with `lost` is a window that was armed and stopped "
                    f"being armed, `not-enabled` is one that never took the write, and `enabled` with neither "
                    f"close code is a core whose close never ran. Both are read through the address the AICPU "
                    f"programmed and neither observes the AICore's own gate, so neither is proof a counter was "
                    f"counting. The per-code counts cover every producer that claimed the run; the "
                    f"physical id and register values are the *first* reporting producer's only. That line, when "
                    f"it was emitted and the window below reaches it, is in the attached child log; its absence "
                    f"here says nothing about what was observed"
                )

        return {"epoch": epoch, "run_id": sidecar.get("run_id"), "host_pid": sidecar.get("host_pid")}

    def _wait_for_both_runs(self, trace, pid: int, what: str) -> list:
        """Block until this child's own span for each of the pair's runs is readable.

        This is the observation boundary, and it is a proof rather than a timeout. A process's
        host log is drained by one writer in enqueue order; a joined launch is published while the
        successor's submission completes, and each run's enclosing ``chip.run`` span is written
        when that run finalizes — after it. So both runs' spans being in the window means any
        joined record this pair produced is already in it too, which is what lets an absence be
        read as an absence rather than as a read taken too early.
        """
        deadline = time.monotonic() + _EVIDENCE_BUDGET_S
        while True:
            spans = trace.spans()
            published = sum(1 for span in spans if span.pid == pid and span.name == _RUN_SPAN)
            if published >= 2:
                return spans
            assert time.monotonic() < deadline, (
                f"{what}: only {published} of this pair's two runs had published their own {_RUN_SPAN} span on "
                f"child {pid} within {_EVIDENCE_BUDGET_S:.0f}s, so nothing can be concluded from the joined "
                f"records in that window"
            )
            time.sleep(0.05)

    def _assert_join_evidence(self, trace, runs, *, expect_joined: bool, what: str):
        """Whether the device held these two runs together, by the runner's own receipt.

        The receipt is read in the direction it was published — the successor is the run submitted
        second — and only an answered query that found the predecessor's boundary *unfired* counts.
        A joined API invocation made after the predecessor had already completed publishes a
        record too, and it must not pass as evidence.
        """
        predecessor, successor = runs[0], runs[1]
        assert successor["host_pid"] == predecessor["host_pid"], (
            f"{what}: the pair was served by children {predecessor['host_pid']} and {successor['host_pid']}, "
            f"so no join between them was ever possible"
        )
        pid = successor["host_pid"]
        spans = self._wait_for_both_runs(trace, pid, what)
        records = _join_records(spans, pid)
        established = _established_pairs(records)
        wanted = (predecessor["epoch"], successor["epoch"])

        if expect_joined:
            assert wanted in established, (
                f"{what}: no joined launch established that epoch {successor['epoch']} completed its submission "
                f"before epoch {predecessor['epoch']} completed. The records on child {pid} were {records}; an "
                f"implementation that serialized every diagnostic run, and one whose successor joined only after "
                f"its predecessor had finished, both look exactly like this.\n{trace.safe_window_text()}"
            )
        else:
            assert wanted not in established and wanted[::-1] not in established, (
                f"{what}: the device joined epochs {wanted}, which this pair must not be. The records on child "
                f"{pid} were {records}"
            )

    def _assert_no_artifact_without_diagnostics(self, worker, root: Path):
        """The control: a destination named, every owner off, and nothing written there.

        The prefix is set exactly as a collecting run sets it, so this says "a run with somewhere
        to write wrote nothing" rather than "a run with nowhere to write wrote nothing" — which
        would hold however much the collectors published. No spin: there is no window to observe
        here, only an output tree to find empty.

        The flush is what makes the emptiness a finding on this Worker. Retention is on for the
        class, so a returning `run()` no longer implies a file exists; without the barrier an
        artifact this control is supposed to catch could simply not have been written yet.
        """
        trace = _ChildTrace(worker)
        return _with_child_evidence(
            trace, "the no-diagnostics control", lambda: self._no_artifact_without_diagnostics(worker, root)
        )

    def _no_artifact_without_diagnostics(self, worker, root: Path):
        """The control itself, with its failures carried by the caller above."""
        root.mkdir(parents=True, exist_ok=True)
        nbytes = SIZE * DTYPE.itemsize
        buffers = [worker.create_buffer(nbytes) for _ in range(3)]
        tensors = [torch.frombuffer(buffer.shm.buf, dtype=DTYPE, count=SIZE) for buffer in buffers]
        tensors[0].fill_(3.0)
        tensors[1].fill_(4.0)
        tensors[2].zero_()
        quiet = self._build_config(self.CASES[0]["config"], output_prefix=str(root))
        ok = False
        try:
            self._submit(worker, buffers, quiet, 0).wait()
            ok = torch.allclose(tensors[2], _golden(tensors[0], tensors[1]), rtol=self.RTOL, atol=self.ATOL)
        finally:
            tensors.clear()
        assert ok, "the default path stopped producing its own result"
        worker.flush_diagnostics()
        for owner in self._checked_owners():
            assert not _find_artifacts(root, owner), f"a run with no diagnostics enabled published a {owner} artifact"


if __name__ == "__main__":
    SceneTestCase.run_module(__name__)
