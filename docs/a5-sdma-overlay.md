# a5 SDMA Workspace Overlay — Isolation Status

The a5 PTO-ISA async-SDMA workspace overlay is merged but **gated off by
default**. This doc explains why, what is gated, and how to develop against it
or re-enable it once the a5 CANN environment supports it.

> Tracking: PR [#1179](https://github.com/hw-native-sys/simpler/pull/1179),
> issue [#1315](https://github.com/hw-native-sys/simpler/issues/1315)
> (re-enable checklist). PTO-ISA version guard for a5 overlays: [#1351](https://github.com/hw-native-sys/simpler/issues/1351)
> (closed via [#1404](https://github.com/hw-native-sys/simpler/pull/1404)).
> Pin path transport: [#1403](https://github.com/hw-native-sys/simpler/issues/1403).

---

## What the overlay is

`ensure_sdma_workspace()` (in `src/a5/platform/onboard/host/comm_hccl.cpp`)
pre-allocates the per-rank PTO-ISA async-SDMA scratch workspace via
`aclnnShmemSdmaStarsQuery*` and mirrors its address into
`CommContext.workSpace`. SDMA producer kernels (`SdmaTget` / `TGET_ASYNC`)
read `workSpace` to submit SQEs to the SDMA hardware engine; the deferred
completion machinery polls workspace event-records to signal consumers.

This is the a5 mirror of the a2a3 SDMA path, which is always on.

## Why it is gated off

Symbol presence is not enough. `aclnnShmemSdmaStarsQuery*` was introduced in
**CANN 9.0** and on 9.1.T500 the symbols live in `libopapi.so` (not
`libascendcl.so`). Calling the API can still fail at STARS sync time and
poison the AICPU context:

| CANN | Failure |
| ---- | ------- |
| 9.1.T500 | STARS streams created but `aclrtSynchronizeStream` fails → AICPU exception `0x715002a` → every later kernel launch reports `507018`. **Breaks all a5 comm cases**, not just SDMA. Reconfirmed 2026-07-21 on current `pto_isa.pin` (`83d01313…`): overlay ON → `Created 48 STARS streams OK` → sync fail → `507018` / `0x715002a`. |
| 9.1.0 (`timestamp=20260625`) | `HcclCommInitRootInfo` returns `HCCL_E_INTERNAL (4)` — base HCCL never comes up. |

`ensure_sdma_workspace` runs inside `comm_alloc_windows` / `alloc_domain`, so
every communication case flows through it. Leaving it on unconditionally would
poison the whole a5 comm test surface. The overlay was verified **working** on
a separate a5 box whose CANN completes the primitive (`sdma_async_completion_demo`
passes), so this remains an environment issue, not a simpler code defect.

## Current state (default OFF)

Controlled by the build-time macro / env var `SIMPLER_ENABLE_PTO_SDMA_WORKSPACE`
(default unset → OFF):

- `ensure_sdma_workspace` is a no-op (`#else (void)h;`).
- `host_runtime.so` does **not** link `libnnopbase` and does not reference
  `aclnnShmemSdmaStarsQuery`.
- `CommContext.workSpace` stays `0`; SDMA producer kernels self-skip
  (`if (comm_ctx->workSpace == 0) { pipe_barrier(PIPE_ALL); return; }`).
- `sdma_async_completion_demo` `pytest.skip`s.
- All other (non-SDMA) comm cases run normally.

### Gating points

Everything below keys on the a5 async workspace overlays. When SDMA
(`SIMPLER_ENABLE_PTO_SDMA_WORKSPACE`) or URMA (`SIMPLER_ENABLE_PTO_URMA_WORKSPACE`)
is ON the a5 onboard host `.so` embeds pto-isa headers, so the same pin /
metadata / staleness / ccache guard that protects a2a3 onboard applies to a5
too (#1351); when both are OFF none of it runs.

The pin-resolved checkout path is passed to host CMake as `-DPTO_ISA_ROOT=`
(from `ensure_pto_isa_root()` / `pto_isa.pin`). It is **not** transported via
`os.environ["PTO_ISA_ROOT"]` or `$ENV{PTO_ISA_ROOT}` (#1403).

| File / mechanism | What |
| ---------------- | ---- |
| `src/a5/platform/onboard/host/CMakeLists.txt` | `option(SIMPLER_ENABLE_PTO_SDMA_WORKSPACE ... OFF)`; under overlay ON: require `-DPTO_ISA_ROOT=` + pto-isa include; `SIMPLER_PTO_ISA_BUILD_COMMIT` cache-bust define |
| `src/a5/platform/onboard/host/comm_hccl.cpp` | `ensure_sdma_workspace` body under `#ifdef SIMPLER_ENABLE_PTO_SDMA_WORKSPACE` |
| `simpler_setup/runtime_compiler.py` | `_init_a5` resolves pin into `self.pto_isa_root` when overlay ON (no env export) |
| `simpler_setup/runtime_builder.py` (`platform_embeds_pto_isa`) | Predicate for pin/metadata/stamp/validation; host compile passes `-DPTO_ISA_ROOT=` + overlay cmake defines |
| `examples/a5/.../sdma_async_completion_demo/test_sdma_async_completion_demo.py` | `pytest.skip` when env var unset |

## Developing under the isolation

- **Non-SDMA comm work** (HCCL, P2P, notify, allreduce, …): unaffected. The
  overlay being off is transparent to these paths.
- **New SDMA features**: build locally with
  `SIMPLER_ENABLE_PTO_SDMA_WORKSPACE=ON` (see below) on a box whose CANN
  **behaviorally** supports it. The kernel-side SDMA primitives (`SdmaTget`,
  `SdmaTput`, `TGET_ASYNC`, `TPUT_ASYNC`) live in pto-isa and are independent
  of this host-side workspace gate.
- **Do not** make `ensure_sdma_workspace` unconditional again without a
  passing `sdma_async_completion_demo` + non-SDMA allreduce regression on the
  target CANN.

## Re-enabling (once CANN is ready)

Checklist before flipping anything:

- [x] #1351 resolved (PTO-ISA version guard covers a5 overlays)
- [ ] CANN **behaviorally** completes `aclnnShmemSdmaStarsQuery` (not just symbols)
- [ ] Option A verified; only then consider Option B

Symbol probe (opapi — `libascendcl.so` is often 0 and misleading):

```bash
nm -D "$ASCEND_HOME_PATH/lib64/libopapi.so" | grep -c SdmaStars   # expect > 0
```

**Behavior probe** (must PASS; symbols alone are insufficient):

```bash
export SIMPLER_ENABLE_PTO_SDMA_WORKSPACE=ON
# pto_isa.pin → ensure_pto_isa_root() → -DPTO_ISA_ROOT= (no manual export)
python examples/a5/tensormap_and_ringbuffer/sdma_async_completion_demo/test_sdma_async_completion_demo.py \
    -p a5 -d <ids>
python -m pytest examples/workers/l3/allreduce_distributed/test_allreduce.py \
    -v --platform a5 --device <ids> -k onephase
```

**Option A — CI / local env var only (no default flip):**

```bash
export SIMPLER_ENABLE_PTO_SDMA_WORKSPACE=ON
pip install --no-build-isolation -e .
```

**Option B — make SDMA the a5 default** (do **not** do this while 9.1.T500-class
CANN is the CI/default environment): flip `option(... OFF)` → `set(... ON)`,
make `_init_a5` resolve the pin unconditionally, and remove the demo skip gate.
