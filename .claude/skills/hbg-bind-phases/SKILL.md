---
name: hbg-bind-phases
description: Measure host_build_graph's `bind` phases — the host-side stage between "caller data in place" and "device can start" (orchestration, Definition upload, H2D), also called the bind path or control plane — on the dsv4 and qwen decode cases, and compare two branches. Use when the user asks how long bind or the control plane takes, whether a change moved host_orch / graph_upload / arena_h2d, or to A/B a host-side change. This is the HOST stage with the device run skipped — for on-device latency use `benchmark` or `perf-example-device` instead.
---

# Measuring the `host_build_graph` bind phases

What the phases are, why each switch is there, and the reference numbers:
[`docs/dfx/hbg-bind-phases.md`](../../../docs/dfx/hbg-bind-phases.md).
Read it before interpreting any number. This file is the invocation.

The device run is skipped, so a case whose device execution does not complete
still yields its whole host picture.

## The recipe

Fill in `NAME`, `RUN`, `DEVICES`, `TIMEOUT`, `CASE` from the case table and
`ENVS`, `TAIL` from the mode table, then run it verbatim.
`pip install --no-build-isolation -e .` first, and again after every `HEAD` move.

```bash
HEAD_SHA=$(git rev-parse --short HEAD)
NAME="<case>_<mode>"                    # e.g. qwen_numbers — see below
RUN=<n>                                 # repetition: 1, 2, … within one A/B session
LOG="outputs/bind_${NAME}_${HEAD_SHA}_r${RUN}.log"; mkdir -p outputs
[ -e "$LOG" ] && { echo "refusing to overwrite $LOG — bump RUN"; exit 1; }
MARK="outputs/.bind_start"; : >"$MARK"   # fixed mtime; "$LOG" keeps being appended to

DEVICES=<N>                                            # per case
TIMEOUT=<seconds>                                      # per case
ENVS="SIMPLER_HBG_BIND_BREAKDOWN_ENABLE=1 \
TORCH_DEVICE_BACKEND_AUTOLOAD=0 SIMPLER_SKIP_DEVICE_RUN=1"          # + mode delta
CASE="examples/.../<entry>.py -p a2a3"                # exactly as the case table gives it
TAIL="--rounds 6 --log-level timing"                                 # per mode

.claude/skills/onboard-arch-precheck/check.sh a2a3 || exit 1
echo "[stamp] $HEAD_SHA env $ENVS python $CASE $TAIL" >"$LOG"
task-submit --device auto --device-num "$DEVICES" --timeout "$TIMEOUT" --max-time "$TIMEOUT" \
  --run "env $ENVS python $CASE $TAIL -d \$TASK_DEVICE" >>"$LOG" 2>&1
grep -c 'name=chip.run.bind\.' "$LOG"   # numbers mode: must be > 0
```

`env` prefixes the assignments so they survive coming from a variable, and the
`[stamp]` line is the same string that runs — which is what makes two logs
comparable. Never hand-edit one arm's command without the other's.

**`$LOG` is opened with a truncating `>`, so its name has to separate every run
this skill tells you to make.** `NAME` separates the two cases and the two modes,
`HEAD_SHA` the two arms of a comparison, and `RUN` the repetitions — the
comparison protocol runs each arm more than once, so `base, measure, base,
measure` is two runs per commit that agree on case, mode and sha. The guard turns
a forgotten `RUN` into a refusal; without all three fields a later run silently
destroys an earlier log, and its own output looks entirely normal.

**`--log-level timing` pins a condition rather than enabling anything.**
`python/simpler/_log.py` puts the `simpler` logger at TIMING on import and
`Worker` snapshots that one logger for the host log and for every forked chip
child, so the segment spans are already on without it. What passing it
buys is the `[stamp]` line: the level is the one measurement condition with no
record of its own in the log — unlike `torch_backend_autoload` — so a run that
omits the flag leaves it implicit in whatever that commit's default happens to
be. Two arms of a cross-commit A/B can then run at different levels with neither
stamp showing it.

**In timeline mode that last check reports 0 on a run that worked.** A diagnostic
flag makes `CallConfig.output_prefix` non-empty, and a non-empty prefix moves
every host-log record — every `[STRACE]` span included — out of
`$LOG` into `outputs/<case>_<ts>/host.<pid>.log`, one file per process. Grep those
instead, and feed the finisher both (see the timeline command below).

## The two cases

| Property | dsv4 FLASH decode | qwen3-14b decode |
| -------- | ----------------- | ---------------- |
| `NAME` prefix | `dsv4` | `qwen` |
| `DEVICES` | 2 | 1 |
| entry point | standalone driver — it owns its `Worker`, so no `--case` / `--manual` / `--level` and no module runner | standalone driver with the same ownership model |
| `CASE` | `examples/a2a3/host_build_graph/deepseek_v4_flash_decode/main.py -p a2a3 --skip-golden` | `examples/a2a3/host_build_graph/qwen3_14b_decode/main.py -p a2a3 --skip-golden` |
| parameters | child memory; **`--skip-golden` is not optional here** — without it the driver first streams a 42.6 GiB-per-rank fixture you are not measuring | device-resident buffers; the valid fixture is streamed once, and `--skip-golden` skips only torch/D2H |
| `TIMEOUT` | 3600 (cold compile is minutes) | 2400 |
| emits `torch_backend_autoload` | **no** — the parser says so above the table, and a dsv4 A/B has no in-log witness for the autoload state | yes |

Neither case needs a `--level`: each driver runs its `Worker` in this process,
so its chip children write straight to the log the `task-submit --run` command
is redirected into. (Under the scene-test module runner they did not: it
captured the child's stdout and the log ended up with zero segment spans
while the run still passed.)

## The two modes

| Field | numbers | timeline |
| ----- | ------- | -------- |
| `NAME` suffix | `_numbers` | `_timeline` |
| `ENVS` delta | none | `+ SIMPLER_HBG_HOST_PHASE_RECORDS_ENABLE=1` |
| `TAIL` | `--rounds 6 --log-level timing` | `--rounds 1 --enable-pmu 2 --log-level timing` |
| finish with | the parser, below | `strace_timing`, below |

`--rounds > 1` force-disables every diagnostic (it warns per flag), so one run
gives statistics or a timeline, never both. The diagnostic flag in timeline mode
is there to make `CallConfig.output_prefix` non-empty, which is what gives the
per-event artifact a directory; `--enable-chip-swimlane` raises
`NotImplementedError` at level 3.

```bash
# numbers — the segment spans are in $LOG, since no diagnostic flag is on
python -m simpler_setup.tools.hbg_bind_phases "$LOG"

# timeline — only an artifact newer than this run's own marker counts
RECORDS=$(find outputs -name host_phase_records.jsonl -newer "$MARK")
echo "$RECORDS"          # must name exactly one file; empty ⇒ this run wrote none,
                         # so stop rather than reading a previous run's artifact
D=$(dirname "$RECORDS")
grep -c 'name=chip.run.bind\.' "$D"/host.*.log   # must be > 0; $LOG has none in this mode
# The spans are split — the invoking process wrote its own to $LOG, each chip
# child wrote its own under $D — so parse the concatenation, not either half.
cat "$LOG" "$D"/host.*.log > "$D/bind_timeline.log"
python -m simpler_setup.tools.strace_timing "$D/bind_timeline.log" \
  --host-phase-records "$RECORDS" \
  --swimlane "$D/host_swimlane.json"                          # load in ui.perfetto.dev
```

## Before reporting a number

- `grep -c 'name=chip.run.bind\.'` was `> 0` — in `$LOG` for numbers mode, in
  `$D/host.*.log` for timeline mode — and the table shows a `[stamp]` line.
- Quote the stamp with the number. A number without the command and commit that
  produced it cannot be compared to anything.
- Before an A/B, check the recording sites at both commits as required by
  **Comparing two branches** in the doc. Identical segment names do not guarantee
  identical coverage: #2353 makes `graph_upload` copy-only and leaves Definition
  preparation outside the reported three-segment subtotal. Do not report that
  subtotal as the complete control plane or infer a speedup across this boundary.
- Follow the remaining comparison rules in that section, including interleaving
  the arms and taking minima of per-bind sums.

For on-device latency instead, use [`benchmark`](../benchmark/SKILL.md) or
[`perf-example-device`](../perf-example-device/SKILL.md).
