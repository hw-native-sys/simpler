---
name: hbg-bind-phases
description: Measure host_build_graph's `bind` phases — the host-side stage between "caller data in place" and "device can start" (orchestration, Definition upload, H2D), also called the bind path or control plane — on the dsv4 and qwen decode cases, and compare two branches. Use when the user asks how long bind or the control plane takes, whether a change moved host_orch / graph_upload / sm_h2d / arena_h2d, or to A/B a host-side change. This is the HOST stage with the device run skipped — for on-device latency use `benchmark` or `perf-example-device` instead.
---

# Measuring the `host_build_graph` bind phases

What the phases are, why each switch is there, and the reference numbers:
[`docs/dfx/hbg-bind-phases.md`](../../../docs/dfx/hbg-bind-phases.md).
Read it before interpreting any number. This file is the invocation.

The device run is skipped, so a case whose device execution does not complete
still yields its whole host picture.

## The recipe

Fill in `ENVS`, `CASE`, `TAIL` from the two tables below, then run it verbatim.
`pip install --no-build-isolation -e .` first, and again after every `HEAD` move.

```bash
HEAD_SHA=$(git rev-parse --short HEAD)
LOG="outputs/bind_${HEAD_SHA}.log"; mkdir -p outputs
MARK="outputs/.bind_start"; : >"$MARK"   # fixed mtime; "$LOG" keeps being appended to

ENVS="SIMPLER_HBG_BIND_BREAKDOWN_ENABLE=1 SIMPLER_LOG_LEVEL=TIMING \
TORCH_DEVICE_BACKEND_AUTOLOAD=0 SIMPLER_SKIP_DEVICE_RUN=1"          # + mode delta
CASE="examples/.../test_<case>.py -p a2a3 \
--case <Class>:: --skip-golden"                       # + manual mode and level, per case
TAIL="--rounds 6"                                                    # per mode

.claude/skills/onboard-arch-precheck/check.sh a2a3 || exit 1
echo "[stamp] $HEAD_SHA env $ENVS python $CASE $TAIL" >"$LOG"
task-submit --device auto --device-num <N> --timeout 3600 --max-time 3600 \
  --run "env $ENVS python $CASE $TAIL -d \$TASK_DEVICE" >>"$LOG" 2>&1
grep -c 'bind phase=' "$LOG"          # must be > 0
```

`env` prefixes the assignments so they survive coming from a variable, and the
`[stamp]` line is the same string that runs — which is what makes two logs
comparable. Never hand-edit one arm's command without the other's.

## The two cases

| Property | dsv4 FLASH decode | qwen3-14b decode |
| -------- | ----------------- | ---------------- |
| `--device-num` | 2 | 1 |
| `CASE` example | `examples/a2a3/host_build_graph/deepseek_v4_flash_decode/test_deepseek_v4_flash_decode.py` | `examples/a2a3/host_build_graph/qwen3_14b_decode/test_qwen3_14b_decode.py` |
| `--case` | `TestDeepseekV4FlashDecodeHostBuildGraph::` | `TestQwen314BDecodeHostBuildGraph::` |
| manual mode | none | **add** `--manual only` |
| level | `level=3`, so **add** `--runtime host_build_graph --level 3` | `level=2`, add nothing |
| timeout | 3600 (cold compile is minutes) | 2400 |

The `--level 3` addition is not optional for dsv4: without it the module runner
captures the child's stdout and the log ends up with zero `bind phase=` lines
while the run still passes.

## The two modes

| Field | numbers | timeline |
| ----- | ------- | -------- |
| `ENVS` delta | none | `+ SIMPLER_HBG_HOST_PHASE_RECORDS_ENABLE=1` |
| `TAIL` | `--rounds 6` | `--rounds 1 --enable-pmu 2` |
| finish with | the parser, below | `strace_timing`, below |

`--rounds > 1` force-disables every diagnostic (it warns per flag), so one run
gives statistics or a timeline, never both. The diagnostic flag in timeline mode
is there to make `CallConfig.output_prefix` non-empty, which is what gives the
per-event artifact a directory; `--enable-chip-swimlane` raises
`NotImplementedError` at level 3.

```bash
# numbers
python -m simpler_setup.tools.hbg_bind_phases "$LOG" --rounds 6

# timeline — only an artifact newer than this run's own marker counts
RECORDS=$(find outputs -name host_phase_records.jsonl -newer "$MARK")
echo "$RECORDS"          # must name exactly one file; empty ⇒ this run wrote none,
                         # so stop rather than reading a previous run's artifact
python -m simpler_setup.tools.strace_timing "$LOG" --host-phase-records "$RECORDS" \
  --swimlane "$(dirname "$RECORDS")/host_swimlane.json"      # load in ui.perfetto.dev
```

## Before reporting a number

- `grep -c 'bind phase='` was `> 0`, and the table shows a `[stamp]` line.
- Quote the stamp with the number. A number without the command and commit that
  produced it cannot be compared to anything.
- Comparing two branches has three more rules, all of them learned the hard way —
  follow **Comparing two branches** in the doc rather than reasoning it out here.

For on-device latency instead, use [`benchmark`](../benchmark/SKILL.md) or
[`perf-example-device`](../perf-example-device/SKILL.md).
