# a5 SDMA Workspace Overlay — Isolation Status

The a5 PTO-ISA async-SDMA workspace overlay is merged but **gated off by
default**. This doc explains why, what is gated, and how to develop against it
or re-enable it once the a5 CANN environment supports it.

> Tracking: PR [#1179](https://github.com/hw-native-sys/simpler/pull/1179),
> issue [#1315](https://github.com/hw-native-sys/simpler/issues/1315)
> (re-enable checklist).

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

The available a5 CANN drops do not expose a working
`aclnnShmemSdmaStarsQuery`:

| CANN | Failure |
| ---- | ------- |
| 9.1.T500 | STARS streams created but `aclrtSynchronizeStream` fails → AICPU exception `0x715002a` → every later kernel launch reports `507018`. **Breaks all a5 comm cases**, not just SDMA. |
| 9.1.0 (`timestamp=20260625`) | `HcclCommInitRootInfo` returns `HCCL_E_INTERNAL (4)` — base HCCL never comes up. |

`ensure_sdma_workspace` runs inside `comm_alloc_windows` / `alloc_domain`, so
every communication case flows through it. Leaving it on unconditionally would
poison the whole a5 comm test surface. The overlay was verified **working** on
a separate a5 box whose CANN exposes the primitive (`sdma_async_completion_demo`
passes), so this is an environment issue, not a code defect.

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

| File | What |
| ---- | ---- |
| `src/a5/platform/onboard/host/CMakeLists.txt` | `option(SIMPLER_ENABLE_PTO_SDMA_WORKSPACE ... OFF)`; `PTO_ISA_ROOT` check + pto-isa include guarded |
| `src/a5/platform/onboard/host/comm_hccl.cpp` | `ensure_sdma_workspace` body under `#ifdef SIMPLER_ENABLE_PTO_SDMA_WORKSPACE` |
| `simpler_setup/runtime_compiler.py` | `_init_a5` `PTO_ISA_ROOT` ensure + `_sdma_workspace_enabled()` |
| `simpler_setup/runtime_builder.py` | `_compile_target` forwards the env var to the host CMake define |
| `examples/a5/.../sdma_async_completion_demo/test_sdma_async_completion_demo.py` | `pytest.skip` when env var unset |

## Developing under the isolation

- **Non-SDMA comm work** (HCCL, P2P, notify, allreduce, …): unaffected. The
  overlay being off is transparent to these paths.
- **New SDMA features**: build locally with
  `SIMPLER_ENABLE_PTO_SDMA_WORKSPACE=ON` (see below) on a box whose CANN
  supports it. The kernel-side SDMA primitives (`SdmaTget`, `SdmaTput`,
  `TGET_ASYNC`, `TPUT_ASYNC`) live in pto-isa and are independent of this
  host-side workspace gate.
- **Do not** make `ensure_sdma_workspace` unconditional again without
  confirming the target CANN exposes a working `aclnnShmemSdmaStarsQuery`.

## Re-enabling (once CANN is ready)

Verify the primitive first:

```bash
nm -D $ASCEND_HOME_PATH/lib64/libascendcl.so | grep -ic SdmaStars   # must be > 0
```

**Option A — CI env var only (no code change):**

```bash
export SIMPLER_ENABLE_PTO_SDMA_WORKSPACE=ON
export PTO_ISA_ROOT=<managed checkout>
pip install -e .
```

The env-var→CMake-define forwarding and the test skip gate pick it up
automatically.

**Option B — make SDMA the a5 default** (revert the 5 gating points above):
flip `option(... OFF)` → `set(... ON)`, make `_init_a5`'s `PTO_ISA_ROOT`
ensure unconditional (mirror `_init_a2a3`), and remove the test skip gate.

Verify:

```bash
python -m pytest examples/a5/tensormap_and_ringbuffer/sdma_async_completion_demo/test_sdma_async_completion_demo.py \
    -v --platform a5 --device <ids> -s          # must PASS, not skip
python -m pytest examples/workers/l3/allreduce_distributed/test_allreduce.py \
    -v --platform a5 --device <ids> -k onephase  # regression must stay green
```
