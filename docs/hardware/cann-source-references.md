# CANN Source References (gitcode.com/cann/*)

When you need to verify what the closed CANN `.so` / `.ko` shipped in
`/usr/local/Ascend/` actually does — memory attribute, ioctl handler,
HCCL channel API, HCCP RDMA — the public source is at
`gitcode.com/cann/`. This page is the **lookup table** for the three
repos we hit most often in simpler dev, plus the convention for
cloning them when you need a working tree.

## When you need them

You usually do **not** need a local clone. Most lookups are one-shot
("what does `halMemCtl(REG_AIC_CTRL)` actually do?") and can be
answered by browsing the repo on gitcode.com. Clone only when:

- you want to grep across hundreds of files for an `addr_type` or
  message-ID dispatch path,
- you're cross-referencing a behavior at two file paths simultaneously
  and need both checked out,
- you're prototyping a fix you might propose upstream.

If you only need to *quote* a file path in a simpler PR / doc /
investigation, link the gitcode URL directly — no clone needed.

## The three repos

| repo | role | URL | recent tag (worth pinning) |
| ---- | ---- | --- | -------------------------- |
| `cann/driver` | Kernel-side driver: `halMemCtl`, `pgprot_*`, ioctl handlers, channel msg routing, svm/devmm/dvpp/queue/buff/comm subsystems. Closed `.ko`s in `/usr/lib/modules/.../updates/` are root-readonly; this is the source. | <https://gitcode.com/cann/driver> | `master` (no version tags); use a date-pinned commit if you cite specific behavior |
| `cann/hccl` | HCCL collective-comm library: `HcclCommInitRootInfo`, channel API (`HcclChannelAcquire`, `HcclGetHcclBuffer`), collective-op decomposition. Backs `libhccl.so`. | <https://gitcode.com/cann/hccl> | `v8.5.0` (matches CANN 8.5.0) |
| `cann/hcomm` | HCOMM lower-level comm: RDMA service (`rs_drv_*`), HCCP control plane, PCIe + UB transport primitives, algorithm impls beneath HCCL. | <https://gitcode.com/cann/hcomm> | `v8.5.0` |

The version tags matter when your closed-install `.so` is a specific
CANN release — pinning the source to the same tag ensures structures,
opCodes, and msgIDs match. `cann/driver` has no version tags (only
`master`); pick a commit close to your install timestamp if precision
matters.

Other repos exist under the `cann` org (compiler, op-libraries,
toolkit, etc.) — not covered here because we don't hit them often
from simpler dev. Browse [gitcode.com/cann](https://gitcode.com/cann)
to find them.

## Clone-to-build/ convention

Follows the same rule as [`.claude/skills/multi-repo-setup/SKILL.md`](../../.claude/skills/multi-repo-setup/SKILL.md):
**external sources are cloned under the simpler worktree's `build/`
directory.** `build/` is `.gitignore`d so the clones never get
committed, and they stay co-located with the simpler you're testing.

```bash
BUILD="$(git rev-parse --show-toplevel)/build"
mkdir -p "$BUILD"

# Clone-or-update each repo on demand. Most investigations only need
# one of these; clone what you need.

# Driver (no version tags — uses master HEAD)
if [ ! -d "$BUILD/cann-driver/.git" ]; then
    git clone --depth 1 https://gitcode.com/cann/driver.git "$BUILD/cann-driver"
else
    git -C "$BUILD/cann-driver" fetch origin master --quiet
    git -C "$BUILD/cann-driver" reset --hard origin/master
fi

# HCCL pinned at v8.5.0
if [ ! -d "$BUILD/hccl/.git" ]; then
    git clone --branch v8.5.0 --depth 1 https://gitcode.com/cann/hccl.git "$BUILD/hccl"
else
    git -C "$BUILD/hccl" fetch origin v8.5.0 --quiet
    git -C "$BUILD/hccl" checkout v8.5.0
fi

# HCOMM pinned at v8.5.0
if [ ! -d "$BUILD/hcomm/.git" ]; then
    git clone --branch v8.5.0 --depth 1 https://gitcode.com/cann/hcomm.git "$BUILD/hcomm"
else
    git -C "$BUILD/hcomm" fetch origin v8.5.0 --quiet
    git -C "$BUILD/hcomm" checkout v8.5.0
fi
```

Browsing happens via your editor against `build/cann-driver/`,
`build/hccl/`, `build/hcomm/`. Don't `cd` into them for long-running
work — the gitignore keeps them out of `git status`, but adding files
inside them from a simpler-side workflow is a smell.

## Per-repo cheat sheet

### `cann/driver` — when investigating MMIO / kernel behavior

What it answers — entry points that come up in simpler debugging:

| Question | File |
| -------- | ---- |
| What is the actual page-table attribute for AIC_CTRL MMIO? | `src/sdk_driver/svm/v2/master/pmaster/svm_master_remote_map.c:1673` (uses `devmm_make_nocache_pgprot` → `pgprot_device()` → `MT_DEVICE_nGnRE`) |
| What does `halMemCtl(ADDR_MAP_TYPE_REG_AIC_CTRL)` dispatch to? | `src/ascend_hal/svm/v2/devmm/devmm_map_dev_reserve.c:184` (handler table) |
| What channel message does the master kernel send for a remote mmap? | `src/sdk_driver/svm/v2/master/comm/svm_master_addr_map.c:33` (`DEVMM_CHAN_MAP_DEV_RESERVE_H2D_ID`) |
| What are the `pgprot_*` factories? | `src/sdk_driver/svm/v2/common/svm_mem_mng.c:23-37` |
| Linux pgprot alias macros (`pgprot_device` etc.) | `src/sdk_driver/kernel_adapt/include/ka_memory_pub.h:183` |

This is the source backing
[`docs/hardware/mmio-performance.md`](mmio-performance.md)'s "Memory
attribute — proven from driver source" section. If you change any
claim there, cross-check against this repo.

### `cann/hccl` — when investigating collective-comm channel APIs

Backs the comm-spike work in `~/workspace/hw-native-sys/comm-spike/`.
Most useful subtrees:

- `src/common/`, `src/ops/` — HCCL op decompositions and runtime
  glue.
- `include/hccl_res.h` (also at
  `/usr/local/Ascend/ascend-toolkit/latest/include/hccl/hccl_res.h`)
  — `HcclGetHcclBuffer`, `HcclChannelAcquire`,
  `HcclChannelGetHcclBuffer` — the channel API that gives access to
  the cross-rank symmetric-memory windows.

### `cann/hcomm` — when investigating sub-HCCL transport / HCCP

The lower-level RDMA + PCIe + UB transport beneath HCCL. Use when an
HCCL-layer question pushes down into the actual transport handshake.
Most useful subtrees:

- `src/platform/hccp/` — HCCP control-plane source. `external_depends/`
  and `pkg_inc/` here have the device-side base headers (`plog.h`,
  etc.) that the host SDK uses.
- `src/platform/hccp/rdma_service/rs_drv_*.c` — RDMA setup and
  packet-level transport.
- `src/framework/`, `src/algorithm/` — algorithm-side helpers HCCL
  uses to dispatch into the transport.

## Related

- [`.claude/skills/multi-repo-setup/SKILL.md`](../../.claude/skills/multi-repo-setup/SKILL.md)
  — the simpler-ecosystem version (simpler, pypto, pypto-lib, pto-isa).
- [`mmio-performance.md`](mmio-performance.md) — driver-source trace
  that cites `cann/driver` paths.
- [`cache-coherency.md`](cache-coherency.md) — the AICore / AICPU
  coherency rules, complemented by the kernel mmap attributes in the
  driver source.
