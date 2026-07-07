# Onboard detection + isolation procedures

Shared reference for the `test-*`, `perf-*`, and `profile` skills. Holds the
platform / device / CI-pin / isolation logic so it lives in one place — when
the chip mapping, idle-device rule, CI job names, or `task-submit` wrapping
change, update them **here** only.

> This file mirrors the conventions of two authoritative neighbours; defer to
> them rather than re-deriving:
>
> - Platform/silicon detection → the
>   [`onboard-arch-precheck`](../skills/onboard-arch-precheck/SKILL.md) skill
>   (reads the CANN `Short_SoC_version`; never hardcodes "this box is a2a3").
> - Device selection + PTO-ISA pin → the
>   [`benchmark`](../skills/benchmark/SKILL.md) skill's Step 2 / Step 3.

## A. Precheck + platform (onboard / hardware)

Onboard runs must gate on real silicon **before** locking a device — a
wrong-arch `--platform` produces 507018 / 507899 cascades that look like real
bugs. Both detecting the platform and refusing a mismatch are handled by the
`onboard-arch-precheck` skill, so do not re-implement chip parsing here.

1. Pick the arch the user is targeting (`a2a3` or `a5`). If the user did not
   say, derive it from the example/test path (`.../a2a3/...` → `a2a3`,
   `.../a5/...` → `a5`); default `a2a3`.
2. Run the gate — it refuses a mismatch (exit 2) or undetectable silicon
   (exit 1) before any device is touched:

   ```bash
   .claude/skills/onboard-arch-precheck/check.sh "$ARCH" || exit 1
   ```

3. The gate caches the *detected* arch at `/tmp/onboard-arch-precheck.cache`
   (`arch|soc|short_soc`). Use that as the authoritative platform instead of
   re-parsing `npu-smi`:

   ```bash
   PLATFORM=$(cut -d'|' -f1 /tmp/onboard-arch-precheck.cache)
   ```

## B. Platform (simulation)

Sim variants are silicon-agnostic — no precheck, no device.

- From a path: `.../a2a3/...` → `a2a3sim`; `.../a5/...` → `a5sim`.
- Otherwise default `a2a3sim`. (Map `950` → `a5sim` only if you already know
  the box is a5; when unsure, default `a2a3sim`.)

## C. Select idle device(s)

An *idle* device is one whose **HBM-Usage is 0** in `npu-smi info` output.
(Mirrors the `benchmark` skill's Step 2.)

- **Single device:** pick the lowest-ID idle device. If none is free, report
  and stop.
- **Device range (multi-device runs):** from the idle devices take **at most
  4** and find the **longest consecutive sub-range**; pass it as
  `--device <start>-<end>`. If no consecutive pair exists, use the lowest-ID
  idle device as `--device <id>`. If none is free, report and stop.

When wrapping in `task-submit` (§E), prefer `--device auto --device-num <N>`
and let it pick free dies instead of selecting here.

## D. Extract CI timeout from `.github/workflows/ci.yml`

PTO-ISA reproducibility comes from `pto_isa.pin`; local test commands should
not pass a separate PTO-ISA commit. For full-pipeline / per-runtime runs,
copy the CI session timeout for the relevant job:

  ```bash
  PTO_SESSION_TIMEOUT=$(grep -oP '(?<=--pto-session-timeout )\d+' .github/workflows/ci.yml | head -1)
  ```

## E. Isolate onboard runs through `task-submit`

Per [running-onboard.md](../rules/running-onboard.md), every onboard
invocation on a shared box must hold a `task-submit` device lock when the tool
is available. Wrap the run; let `task-submit` own the device:

```bash
if command -v task-submit >/dev/null 2>&1; then
  task-submit --timeout <t> --max-time <t> --device auto --device-num <N> \
    --run "<the python/pytest command, using \$TASK_DEVICE for -d/--device>"
else
  echo "[WARN] task-submit not found; running unlocked — results may be noisy if another process is on this NPU"
  <the python/pytest command, using the §C-selected device>
fi
```

- For pytest (multi-device) pass `--device-num` matching the range size; for a
  standalone `python test_*.py -d` use `--device-num 1` and `-d $TASK_DEVICE`.
- Set `--timeout`/`--max-time` to cover the run (e.g. 1800 for a single
  example, longer for a full pipeline).
- When `task-submit` is absent, fall back to the §C-selected idle device and
  document that the run is unlocked.
