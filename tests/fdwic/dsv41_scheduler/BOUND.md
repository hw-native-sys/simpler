# Synchronization-constrained lower bound

The table uses a conditional model of each case, independently of the runtime
being compared. It preserves `sync_start` as an admission constraint, even
when scheduler execution costs zero. Half-sync and all-sync have separate
L1 captures, durations, models and results.

## Inputs and model

Use a validated isolated HBG L1 capture of the exact bundle: one warmup and
ten measured rounds. Retain `deps.json`, `source-task-map.txt`, execution
identities and the matching `plan-rank0.bin`. The plan supplies the actual
sync flags; neither a task name nor its block count can substitute for them.

`ideal_bound.read_case` audits execution coverage, the completion DAG,
topology consistency, numerical validation and profiling level. Communication,
deferred completion and early dependency release are excluded.

Each round models non-preemptive `(task, block, subtask)` intervals using their
measured kernel durations, 32 AIC/64 AIV capacities, MIX subtasks sharing a
block start, and sync-start tasks sharing a start across **all** blocks. Task
completion is the latest subtask end; every successor waits for all captured
completion dependencies. Zero barrier overhead does not remove co-residency.

Specific cluster placement is relaxed. Scheduling, notification and teardown
cost zero. Durations round **down** to 100 ns, a conservative relaxation rather
than an assumed hardware latency. The objective minimizes final completion.
The older critical-path/work-volume formula is only a preliminary relaxation.

For example, on two workers a 100-tick single block can overlap two independent
10-tick blocks and finish in 100 ticks. Requiring those two blocks to start
together raises the model optimum to 110 ticks, with unchanged kernels and DAG.

## Reproduce

Install the optional analysis dependency inside the worktree virtual environment;
ordinary runtime builds and benchmark execution do not need OR-Tools:

```bash
source .venv/bin/activate
pip install -r tests/fdwic/dsv41_scheduler/production/requirements-bound.txt
```

Capture the selected variant through the automatically allocating launcher:

```bash
export DSV41=tests/fdwic/dsv41_scheduler
bundle="$PREPARED/cases/pypto_lib_dsl_all_sync64/c2a_full"
capture="$RUN_DIR/l1/c2a_full-all"
bash "$DSV41/run.sh" "$bundle" "$DSV41_RUNTIME" "$capture" 1 1 10
PYTHONPATH=.:python python "$DSV41/production/sync_bound.py" \
  "$bundle" "$capture/rank0" "$RUN_DIR/bounds/c2a_full-all" --time-limit 30
```

The time limit applies to each measured round. Reusing an output directory
refines unfinished samples using their previous bounds and feasible schedules;
an input/model fingerprint rejects results from a different problem. Optimal
samples are retained. A serial capacity-respecting schedule supplies a hint,
never a replacement lower bound. Feasible results are independently checked
for capacities, durations, precedence and common starts.

`summary.json` records the mean proven bound, mean feasible time when available,
per-round solver status and `all_optimal`. Per-round files retain the schedule
and problem fingerprint. Keep generated artifacts on the experiment host.

## Interpret and present

The table uses the average **proven lower bound**, not the feasible schedule's
upper bound on the model optimum. Only `OPTIMAL` closes the gap. If any round
is unclosed, mark its case's bound and percentage with a dagger. Do not replace
the missing proof with a runtime measurement.

Measured durations include intra-kernel waits and depend on cache behavior,
concurrency and L1 instrumentation. This is neither an absolute hardware limit
nor necessarily a strict bound on another L0 run. Differences between half-sync
and all-sync include both their constraints and separately measured durations.

Follow the [fixed table contract](README.md#performance-presentation-contract).
Communication rows have no bound or percentage. The best measured time is the
smaller HBG/percounter value actually displayed in the row.

The model follows the official [OR-Tools scheduling guide](https://developers.google.com/optimization/scheduling/job_shop)
and [CP-SAT status definitions](https://developers.google.com/optimization/cp/cp_solver).
