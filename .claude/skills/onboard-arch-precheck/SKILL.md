---
name: onboard-arch-precheck
description: Detect the host's actual Ascend silicon and refuse mismatched `--platform` onboard hardware test invocations BEFORE any device is locked. MUST invoke this skill before running pytest or task-submit commands that use `--platform a2a3` or `--platform a5` (onboard only — sim variants pass through). Use when invoking onboard hardware tests, repro'ing flaky-test reports, or wrapping pytest in task-submit. Skip for `--platform a2a3sim` / `--platform a5sim` (silicon-agnostic).
---

# Onboard Arch Precheck

Gate to prevent running wrong-arch onboard tests on a dev box. CI already
handles this (each onboard runner is labeled with its arch and only gets
matching jobs). Local hardware work bypasses that protection — this skill
restores it.

## Why this exists (DO NOT STRIP — non-obvious failure-mode context)

This repo supports two onboard arches:

- **`a2a3`** — Ascend 910 family (`Ascend910B*` = Atlas A2 silicon;
  `Ascend910_93*` = Atlas A3 silicon; both build via `dav-c220` and share
  `src/a2a3/`)
- **`a5`** — different silicon (`Ascend950*` family, builds via `dav-c310`,
  `src/a5/`)

A given dev box has ONE of them. **Running tests built for the wrong arch
produces error cascades that look like genuine refactor bugs but aren't.**
Typical signatures:

| Error code | Surface | Meaning when arch-mismatched |
| ---------- | ------- | ---------------------------- |
| **507018** (`ACL_ERROR_RT_AICPU_EXCEPTION`) | `aclrtSynchronizeStreamWithTimeout` | AICPU kernel built against the wrong silicon contract faults at runtime; host sees a timeout / aicpu exception. |
| **507899** | `rtStreamCreate` / `rtMalloc` | Driver rejects an operation because the resource model doesn't match the silicon. |

**Critical**: BOTH codes can also be GENUINE bugs (real flaky tests,
real refactor regressions, real AICPU OS issues — see issue #822 for a
historical example). This is precisely what makes the misdiagnosis costly.
People chase a phantom regression for hours when the actual cause was
running the wrong `--platform`. This skill is cheap insurance against that.

## When to invoke

Invoke `check.sh` BEFORE running ANY of these commands:

- `pytest ... --platform a2a3 ...` (onboard)
- `pytest ... --platform a5 ...` (onboard)
- `task-submit ... --run "... pytest ... --platform a2a3 ..."`
- `task-submit ... --run "... pytest ... --platform a5 ..."`
- `python test_*.py -p a2a3 ...` / `-p a5 ...` (standalone scene-test runner)
- Any wrapper script that calls the above

**Skip the check entirely** when the platform string is a sim variant:

- `--platform a2a3sim`
- `--platform a5sim`
- `-p a2a3sim` / `-p a5sim`

Sim variants build and run on host CPU only — they cannot mismatch
silicon by definition. Wasting the precheck on them adds latency.

## How to use

```bash
# Single check; exit non-zero with a clear message on mismatch.
.claude/skills/onboard-arch-precheck/check.sh a2a3

# Embedded in a workflow:
.claude/skills/onboard-arch-precheck/check.sh "$ARCH" || exit 1
pytest ... --platform "$ARCH" ...

# Before task-submit (gate BEFORE the device lock):
.claude/skills/onboard-arch-precheck/check.sh "$ARCH" || exit 1
task-submit --device auto --device-num 1 --run "pytest ... --platform $ARCH ..."
```

`check.sh` exits:

- `0` — silicon matches the requested arch, or the arch is a sim variant (pass through)
- `2` — silicon mismatch, with an explanation of the detected arch and the 507018 / 507899 failure modes printed to stderr
- `1` — unable to detect silicon (npu-smi missing, unrecognized chip, etc.)

## Detection method

Same authoritative source as `tools/cann-examples/query/` — the CANN
`platform_config/<SoC>.ini` `Short_SoC_version` field — but read via
shell so the precheck does not depend on the `query` binary being built
and never has to bind an ACL device (which would risk contending with
in-flight `task-submit` device locks).

Pipeline:

1. `npu-smi info -t board -i 0 -c 0` → get `Chip Name` + `NPU Name`
   (~600 ms, no ACL init, no device binding).
2. Construct CANN SoC name per family:
   - `Ascend910` + `B*` NPU → `Ascend910B3` (or B1/B2/B4)
   - `Ascend910` + numeric NPU → `Ascend910_9392` (Atlas A3 SKUs)
   - `Ascend950` + NPU → glob for `Ascend950DT_<NPU>` or `Ascend950PR_<NPU>`
3. Read `Short_SoC_version=` from
   `${ASCEND_HOME_PATH}/{aarch64,x86_64}-linux/data/platform_config/<SoC>.ini`.
4. Map `Short_SoC_version` → repo arch (must stay in sync with
   `docs/hardware/chip-architecture.md` and `query.cpp::arch_from_short_soc`):

   | `Short_SoC_version` | Repo arch |
   | ------------------- | --------- |
   | `Ascend910B` | `a2a3` |
   | `Ascend910_93` | `a2a3` |
   | `Ascend950` | `a5` |

The skill **never hardcodes "this box is a2a3"** — every check
re-derives the answer from `npu-smi` + the CANN ini files. The result
is cached at `/tmp/onboard-arch-precheck.cache` (format
`arch|soc|short_soc`) with a 1-hour TTL so repeat invocations stay fast
(~5 ms). Mismatch errors include the specific SoC + Short_SoC_version
so you know exactly what silicon was detected (e.g.
`Detected silicon: a2a3 (SoC=Ascend910_9392, Short_SoC_version=Ascend910_93)`).

## What this does NOT cover

- **Multi-arch hosts**: not a real configuration in this org's fleet. If
  one ever exists, the cache will need to be keyed by `npu-smi info -i N`
  per-device. The skill currently inspects device 0 only.
- **Sim builds**: intentionally pass through — they're silicon-agnostic by
  construction (`src/{arch}/platform/sim/`).
- **vNPU / virtualization**: irrelevant here — vNPU slicing happens within
  one silicon family; the `Ascend910` vs `Ascend950` discrimination is
  unaffected.
- **CANN version drift**: this is a HARDWARE-side check. If CANN is
  installed for the wrong silicon, separate `ASCEND_HOME_PATH` /
  `ASCEND_DRIVER_PATH` errors will trip during the actual build / run,
  not here.

## Related

- [task-submit-isolation.md](../../rules/task-submit-isolation.md) — sister
  rule that says all hardware work goes through `task-submit`. The
  precheck runs BEFORE `task-submit` to avoid wasteful device lock
  acquisition on a doomed-to-fail invocation.
- [chip-architecture.md SoC → arch mapping](../../../docs/hardware/chip-architecture.md#identifying-which-chip-generation-you-have) —
  the authoritative table for `Short_SoC_version` → repo arch. The
  `npu-smi` `Chip Name` is a coarser view (family-level only), which is
  fine for the arch gate purpose.
